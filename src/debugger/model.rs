//! Egui-free view-model for the visual debugger: pure functions that project
//! live emulator state (`TileArray`, `InterpreterEngine`) into plain structs
//! the GUI layer can render without touching device internals directly.

use std::collections::BTreeSet;

use crate::device::array::TileArray;
use crate::device::tile::Tile;
use crate::device::{PortDirection, PortType};
use crate::interpreter::{CoreStatus, InterpreterEngine};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileKindDisplay {
    Shim,
    Mem,
    Core,
}

impl TileKindDisplay {
    fn of(tile: &Tile) -> Self {
        if tile.is_shim() {
            TileKindDisplay::Shim
        } else if tile.is_mem() {
            TileKindDisplay::Mem
        } else {
            TileKindDisplay::Core
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileCell {
    pub col: u8,
    pub row: u8,
    pub kind: TileKindDisplay,
}

/// Flat list of every tile position in the array, tagged with its kind.
pub fn tile_grid(array: &TileArray) -> Vec<TileCell> {
    array
        .iter()
        .map(|t| TileCell { col: t.col, row: t.row, kind: TileKindDisplay::of(t) })
        .collect()
}

/// Sorted, distinct axis values present in a tile grid.
pub fn tile_axes(cells: &[TileCell]) -> (Vec<u8>, Vec<u8>) {
    let cols = cells.iter().map(|cell| cell.col).collect::<BTreeSet<_>>().into_iter().collect();
    let rows = cells.iter().map(|cell| cell.row).collect::<BTreeSet<_>>().into_iter().collect();
    (cols, rows)
}

/// What a tile is doing, for display. Egui-free, testable. Derived from
/// CoreStatus for compute tiles, and from DMA/stream activity for shim/mem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileState {
    NotEnabled,
    Ready,
    Running,
    WaitLock(u8),
    WaitDma(u8),
    WaitStream(u8),
    WaitBank,
    Done,
    Error,
    Dma,
    Stream,
    Idle,
}

impl TileState {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotEnabled => "---",
            Self::Ready => "RDY",
            Self::Running => "RUN",
            Self::WaitLock(_) => "LOCK",
            Self::WaitDma(_) | Self::Dma => "DMA",
            Self::WaitStream(_) | Self::Stream => "STR",
            Self::WaitBank => "BANK",
            Self::Done => "DONE",
            Self::Error => "ERR",
            Self::Idle => "IDLE",
        }
    }

    pub fn letter(&self) -> char {
        // -=disabled, Y=ready, R=running, L=lock, D=DMA wait, S=stream wait,
        // B=bank, O=done, E=error, M=DMA moving, T=stream moving, I=idle.
        match self {
            Self::NotEnabled => '-',
            Self::Ready => 'Y',
            Self::Running => 'R',
            Self::WaitLock(_) => 'L',
            Self::WaitDma(_) => 'D',
            Self::WaitStream(_) => 'S',
            Self::WaitBank => 'B',
            Self::Done => 'O',
            Self::Error => 'E',
            Self::Dma => 'M',
            Self::Stream => 'T',
            Self::Idle => 'I',
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }
}

/// Full screen-reader sentence for one tile.
pub fn accessible_label(col: u8, row: u8, kind: TileKindDisplay, state: &TileState) -> String {
    let kind = match kind {
        TileKindDisplay::Shim => "Shim",
        TileKindDisplay::Mem => "Memory",
        TileKindDisplay::Core => "Core",
    };
    let state = match state {
        TileState::NotEnabled => "not enabled".into(),
        TileState::Ready => "ready".into(),
        TileState::Running => "running".into(),
        TileState::WaitLock(id) => format!("waiting on lock {id}"),
        TileState::WaitDma(channel) => format!("waiting on DMA channel {channel}"),
        TileState::WaitStream(port) => format!("waiting on stream port {port}"),
        TileState::WaitBank => "waiting on memory bank".into(),
        TileState::Done => "done".into(),
        TileState::Error => "error".into(),
        TileState::Dma => "DMA active".into(),
        TileState::Stream => "stream active".into(),
        TileState::Idle => "idle".into(),
    };
    format!("{kind} ({col},{row}): {state}")
}

/// Classify one tile for display.
pub fn tile_state(engine: &InterpreterEngine, col: u8, row: u8) -> TileState {
    let array = &engine.device().array;
    let Some(tile) = array.get(col, row) else {
        return TileState::NotEnabled;
    };
    if !is_column_enabled(engine, col) {
        return TileState::NotEnabled;
    }

    if tile.is_compute() {
        if !engine.is_core_enabled(col as usize, row as usize) {
            return TileState::NotEnabled;
        }
        return match engine.core_status(col as usize, row as usize) {
            Some(CoreStatus::Ready) => TileState::Ready,
            Some(CoreStatus::Running) => TileState::Running,
            Some(CoreStatus::WaitingLock { raw_lock_id }) => TileState::WaitLock(raw_lock_id),
            Some(CoreStatus::WaitingDma { channel }) => TileState::WaitDma(channel),
            Some(CoreStatus::WaitingStream { port }) => TileState::WaitStream(port),
            Some(CoreStatus::WaitBank) => TileState::WaitBank,
            Some(CoreStatus::Halted) => TileState::Done,
            Some(CoreStatus::Error) => TileState::Error,
            None => TileState::NotEnabled,
        };
    }

    if array.dma_engine(col, row).is_some_and(|dma| dma.any_channel_active()) {
        TileState::Dma
    } else if tile
        .stream_switch
        .masters
        .iter()
        .chain(&tile.stream_switch.slaves)
        .any(|port| port.cycle_beat)
    {
        TileState::Stream
    } else {
        TileState::Idle
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PortWire {
    pub index: u8,
    pub direction: PortDirection,
    pub port_type: PortType,
    pub moving: bool,
    pub active: bool,
    pub stalled: bool,
    pub route_to: Option<(u8, u8, u8)>,
}

/// Every stream port on one tile, masters then slaves.
pub fn tile_ports(array: &TileArray, col: u8, row: u8) -> Vec<PortWire> {
    let Some(tile) = array.get(col, row) else {
        return Vec::new();
    };
    tile.stream_switch
        .masters
        .iter()
        .chain(&tile.stream_switch.slaves)
        .map(|port| PortWire {
            index: port.index,
            direction: port.direction,
            port_type: port.port_type,
            moving: port.cycle_beat,
            active: port.cycle_active,
            stalled: port.cycle_stalled,
            route_to: port.route_to,
        })
        .collect()
}

/// True if the column is participating (any enabled core / active tile).
pub fn is_column_enabled(engine: &InterpreterEngine, col: u8) -> bool {
    engine.device().array.clock().is_column_active(col)
}

#[derive(Clone, Debug)]
pub struct ChannelSnapshot {
    pub index: u8,
    pub state: String,
    pub current_bd: Option<u8>,
    pub queued_bd: Option<u8>,
    pub queue_len: usize,
}

#[derive(Clone, Debug)]
pub struct PortSnapshot {
    pub label: String,
    pub active: bool,
    pub stalled: bool,
}

#[derive(Clone, Debug)]
pub struct TileSnapshot {
    pub col: u8,
    pub row: u8,
    pub kind: TileKindDisplay,
    pub core_status: Option<String>,
    pub pc: Option<u32>,
    pub dma: Vec<ChannelSnapshot>,
    pub locks: Vec<i8>,
    pub mem_size: usize,
    pub mem_peek: Vec<u32>,
    pub ports: Vec<PortSnapshot>,
}

/// Point-in-time snapshot of one tile's live state, or `None` if `(col, row)`
/// is outside the array.
pub fn tile_snapshot(engine: &InterpreterEngine, col: u8, row: u8) -> Option<TileSnapshot> {
    let array = &engine.device().array;
    let tile = array.get(col, row)?;
    let kind = TileKindDisplay::of(tile);

    // Core status + live PC (only meaningful on compute tiles).
    let core_status = engine.core_status(col as usize, row as usize).map(|s| format!("{s:?}"));
    let pc = engine.core_context(col as usize, row as usize).map(|c| c.pc());

    // DMA channels (coarse live view).
    let mut dma = Vec::new();
    if let Some(eng) = array.dma_engine(col, row) {
        for ch in 0..eng.channel_count() as u8 {
            dma.push(ChannelSnapshot {
                index: ch,
                state: format!("{:?}", eng.channel_state(ch)),
                current_bd: eng.current_bd(ch),
                queued_bd: eng.queued_bd(ch),
                queue_len: eng.task_queue_size(ch),
            });
        }
    }

    // All 64 locks (effective value accounts for pending updates). Tiles with
    // fewer real locks (e.g. 16 on compute) report 0 for the unused indices --
    // `effective_lock_value` already bounds-checks against the backing Vec.
    let locks: Vec<i8> = (0..64).map(|i| tile.effective_lock_value(i)).collect();

    // Memory: size + a small word peek (first 8 words).
    let mem_size = tile.data_memory().len();
    let mem_peek: Vec<u32> = (0..8).filter_map(|w| tile.read_data_u32(w * 4)).collect();

    // Stream ports: master + slave activity.
    let mut ports = Vec::new();
    for (i, p) in tile.stream_switch.masters.iter().enumerate() {
        ports.push(PortSnapshot { label: format!("M{i}"), active: p.cycle_active, stalled: p.cycle_stalled });
    }
    for (i, p) in tile.stream_switch.slaves.iter().enumerate() {
        ports.push(PortSnapshot { label: format!("S{i}"), active: p.cycle_active, stalled: p.cycle_stalled });
    }

    Some(TileSnapshot { col, row, kind, core_status, pc, dma, locks, mem_size, mem_peek, ports })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{PortDirection, PortType};
    use crate::interpreter::InterpreterEngine;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin")
    }

    #[test]
    fn tile_grid_matches_npu1_layout() {
        let engine = InterpreterEngine::new_npu1();
        let grid = tile_grid(&engine.device().array);
        // Every position in the flat array yields exactly one cell.
        let (cols, rows) = (engine.device().array.cols(), engine.device().array.rows());
        assert_eq!(grid.len(), cols as usize * rows as usize, "one cell per tile");
        // Row 0 is all shim; row 1 is all mem; rows >=2 are core.
        for c in &grid {
            match c.row {
                0 => assert_eq!(c.kind, TileKindDisplay::Shim),
                1 => assert_eq!(c.kind, TileKindDisplay::Mem),
                _ => assert_eq!(c.kind, TileKindDisplay::Core),
            }
        }
    }

    #[test]
    fn tile_axes_are_distinct_values_reported_by_the_grid() {
        let cells = vec![
            TileCell { col: 7, row: 9, kind: TileKindDisplay::Core },
            TileCell { col: 3, row: 2, kind: TileKindDisplay::Shim },
            TileCell { col: 7, row: 2, kind: TileKindDisplay::Mem },
        ];

        assert_eq!(tile_axes(&cells), (vec![3, 7], vec![2, 9]));
    }

    #[test]
    fn tile_state_classifies_a_known_kernel_core() {
        let path = fixture();
        if !path.exists() {
            eprintln!(
                "SKIP tile_state_classifies_a_known_kernel_core: fixture not built at {}",
                path.display()
            );
            return;
        }
        let engine = crate::loading::load_engine(&path).expect("load add_one_using_dma");

        let state = tile_state(&engine, 0, 2);

        assert_eq!(state, TileState::Ready);
        assert_eq!(state.code(), "RDY");
    }

    #[test]
    fn tile_state_letters_are_injective() {
        let states = [
            TileState::NotEnabled,
            TileState::Ready,
            TileState::Running,
            TileState::WaitLock(5),
            TileState::WaitDma(1),
            TileState::WaitStream(3),
            TileState::WaitBank,
            TileState::Done,
            TileState::Error,
            TileState::Dma,
            TileState::Stream,
            TileState::Idle,
        ];

        assert_eq!(states.iter().map(TileState::letter).collect::<HashSet<_>>().len(), states.len());
        assert!(TileState::Error.is_error());
        assert!(!TileState::Running.is_error());
    }

    #[test]
    fn accessible_label_includes_kind_position_and_full_state() {
        assert_eq!(
            accessible_label(2, 3, TileKindDisplay::Core, &TileState::WaitLock(5)),
            "Core (2,3): waiting on lock 5"
        );
    }

    #[test]
    fn tile_ports_projects_compute_masters_then_slaves() {
        let engine = InterpreterEngine::new_npu1();
        let ports = tile_ports(&engine.device().array, 0, 2);

        assert!(!ports.is_empty());
        assert_eq!(ports[0].index, 0);
        assert_eq!(ports[0].direction, PortDirection::Master);
        assert_eq!(ports[0].port_type, PortType::Core);
        assert_eq!(ports[23].index, 0);
        assert_eq!(ports[23].direction, PortDirection::Slave);
        assert_eq!(ports[23].port_type, PortType::Core);
    }

    #[test]
    fn column_enabled_tracks_the_real_column_gate() {
        let gated = InterpreterEngine::new_npu1();
        assert!(!is_column_enabled(&gated, 0));

        let mut active = InterpreterEngine::new_npu1();
        active.device_mut().array.clock_mut().ungate_all();
        assert!(is_column_enabled(&active, 0));
    }

    #[test]
    fn tile_snapshot_reports_locks_and_memory() {
        let engine = InterpreterEngine::new_npu1();
        // A compute tile exists at (0,2) on NPU1.
        let snap = tile_snapshot(&engine, 0, 2).expect("compute tile exists");
        assert_eq!(snap.locks.len(), 64);
        assert!(snap.mem_size > 0);
        assert_eq!(snap.col, 0);
        assert_eq!(snap.row, 2);
    }

    #[test]
    fn tile_snapshot_none_for_missing_tile() {
        let engine = InterpreterEngine::new_npu1();
        assert!(tile_snapshot(&engine, 99, 99).is_none());
    }
}
