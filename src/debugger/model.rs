//! Egui-free view-model for the visual debugger: pure functions that project
//! live emulator state (`TileArray`, `InterpreterEngine`) into plain structs
//! the GUI layer can render without touching device internals directly.

use std::collections::BTreeSet;

use crate::device::array::TileArray;
use crate::device::dma::{DimensionConfig, DmaStall, IterationConfig};
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
    pub progress: f32,
    pub phase: &'static str,
    pub stall: Option<DmaStall>,
    pub queue_bds: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DmaBdDetail {
    pub id: u8,
    pub base_addr: u64,
    pub length: u32,
    pub dimensions: [DimensionConfig; 4],
    pub iteration: IterationConfig,
    pub acquire_lock: Option<u8>,
    pub acquire_value: i8,
    pub release_lock: Option<u8>,
    pub release_value: i8,
    pub next_bd: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct DmaChannelDetail {
    pub index: u8,
    pub phase: &'static str,
    pub stall: Option<DmaStall>,
    pub current_bd: Option<DmaBdDetail>,
    pub queued_bd_ids: Vec<u8>,
    pub bytes_transferred: u64,
    pub total_bytes: u64,
    pub current_address: Option<u64>,
    pub remaining_bytes: u64,
    pub lock_wait_cycles: u64,
    pub cycles_spent: u64,
    pub transfers_completed: u64,
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
                progress: eng.get_transfer(ch).map_or(0.0, |transfer| transfer.progress()),
                phase: eng.channel_phase(ch),
                stall: eng.channel_stall_reason(ch),
                queue_bds: eng.queued_bd_ids(ch).into_iter().take(8).collect(),
            });
        }
    }

    // The tile's real lock bank (16 on compute/shim, 64 on mem). Effective value
    // accounts for pending updates this cycle.
    let locks: Vec<i8> = (0..tile.locks.len()).map(|i| tile.effective_lock_value(i)).collect();

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

/// Full DMA detail built only for the selected channel.
pub fn dma_channel_detail(
    engine: &InterpreterEngine,
    col: u8,
    row: u8,
    channel: u8,
) -> Option<DmaChannelDetail> {
    let dma = engine.device().array.dma_engine(col, row)?;
    if channel as usize >= dma.channel_count() {
        return None;
    }

    let current_bd = dma.current_bd(channel).and_then(|id| {
        dma.get_bd(id).map(|bd| DmaBdDetail {
            id,
            base_addr: bd.base_addr,
            length: bd.length,
            dimensions: [bd.d0, bd.d1, bd.d2, bd.d3],
            iteration: bd.iteration,
            acquire_lock: bd.acquire_lock,
            acquire_value: bd.acquire_value,
            release_lock: bd.release_lock,
            release_value: bd.release_value,
            next_bd: bd.next_bd,
        })
    });
    let transfer = dma.get_transfer(channel);
    let stats = dma.channel_stats(channel)?;

    Some(DmaChannelDetail {
        index: channel,
        phase: dma.channel_phase(channel),
        stall: dma.channel_stall_reason(channel),
        current_bd,
        queued_bd_ids: dma.queued_bd_ids(channel),
        bytes_transferred: transfer.map_or(0, |transfer| transfer.bytes_transferred),
        total_bytes: transfer.map_or(0, |transfer| transfer.total_bytes),
        current_address: transfer.map(|transfer| transfer.address_gen.current()),
        remaining_bytes: transfer.map_or(0, |transfer| transfer.remaining_bytes()),
        lock_wait_cycles: stats.lock_wait_cycles,
        cycles_spent: stats.cycles_spent,
        transfers_completed: stats.transfers_completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::dma::{BdConfig, DimensionConfig, DmaStall, IterationConfig};
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
        // Compute tiles own 16 locks (mem tiles have 64); the snapshot reports
        // the real bank size, not a padded 64.
        assert_eq!(snap.locks.len(), 16);
        assert!(snap.mem_size > 0);
        assert_eq!(snap.col, 0);
        assert_eq!(snap.row, 2);
    }

    #[test]
    fn tile_snapshot_none_for_missing_tile() {
        let engine = InterpreterEngine::new_npu1();
        assert!(tile_snapshot(&engine, 99, 99).is_none());
    }

    #[test]
    fn dma_snapshot_carries_live_progress_phase_stall_and_queue() {
        let mut engine = InterpreterEngine::new_npu1();
        let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
        dma.configure_bd(0, BdConfig::simple_1d(0x400, 64).with_acquire(5, 1)).unwrap();
        dma.configure_bd(3, BdConfig::simple_1d(0x800, 32)).unwrap();
        dma.configure_bd(5, BdConfig::simple_1d(0xc00, 32)).unwrap();
        assert!(dma.enqueue_task(2, 0, 0, false));
        assert!(dma.enqueue_task(2, 3, 0, false));
        assert!(dma.enqueue_task(2, 5, 0, false));

        let snap = tile_snapshot(&engine, 0, 2).unwrap();
        let channel = snap.dma.iter().find(|channel| channel.index == 2).unwrap();

        assert_eq!(channel.progress, 0.0);
        assert_eq!(channel.phase, "AcquiringLock");
        assert_eq!(channel.stall, Some(DmaStall::LockWait));
        assert_eq!(channel.queue_bds, vec![3, 5]);
        assert_eq!(channel.queue_len, 2);
    }

    #[test]
    fn dma_channel_detail_projects_current_bd_live_position_queue_and_stats() {
        let mut engine = InterpreterEngine::new_npu1();
        let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
        let mut current = BdConfig::simple_1d(0x400, 64).with_acquire(5, 1).with_release(6, -1);
        current.d0 = DimensionConfig::new(4, 4);
        current.d1 = DimensionConfig::new(3, 32);
        current.d2 = DimensionConfig::new(2, -64);
        current.d3 = DimensionConfig::new(5, 256);
        current.iteration = IterationConfig::new(2, 7);
        dma.configure_bd(0, current).unwrap();
        dma.configure_bd(3, BdConfig::simple_1d(0x800, 32)).unwrap();
        dma.configure_bd(5, BdConfig::simple_1d(0xc00, 32)).unwrap();
        assert!(dma.enqueue_task(2, 0, 0, false));
        assert!(dma.enqueue_task(2, 3, 0, false));
        assert!(dma.enqueue_task(2, 5, 0, false));
        let expected_total_bytes = dma.get_transfer(2).unwrap().total_bytes;

        let detail = dma_channel_detail(&engine, 0, 2, 2).unwrap();
        let bd = detail.current_bd.unwrap();

        assert_eq!(detail.index, 2);
        assert_eq!(detail.phase, "AcquiringLock");
        assert_eq!(detail.stall, Some(DmaStall::LockWait));
        assert_eq!(bd.id, 0);
        assert_eq!(bd.base_addr, 0x400);
        assert_eq!(bd.length, 64);
        assert_eq!((bd.dimensions[1].size, bd.dimensions[1].stride), (3, 32));
        assert_eq!((bd.dimensions[2].size, bd.dimensions[2].stride), (2, -64));
        assert_eq!((bd.iteration.wrap, bd.iteration.stepsize), (2, 7));
        assert_eq!((bd.acquire_lock, bd.acquire_value), (Some(5), 1));
        assert_eq!((bd.release_lock, bd.release_value), (Some(6), -1));
        assert_eq!(detail.queued_bd_ids, vec![3, 5]);
        assert_eq!(detail.bytes_transferred, 0);
        assert_eq!(detail.total_bytes, expected_total_bytes);
        assert_eq!(detail.current_address, Some(0x400));
        assert_eq!(detail.remaining_bytes, expected_total_bytes);
        assert_eq!(detail.lock_wait_cycles, 0);
        assert_eq!(detail.cycles_spent, 0);
        assert_eq!(detail.transfers_completed, 0);
        assert!(dma_channel_detail(&engine, 0, 2, 99).is_none());
    }

    #[test]
    fn dma_channel_detail_projects_partial_progress_and_completed_stats() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().array.clock_mut().ungate_all();
        {
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            dma.configure_bd(0, BdConfig::simple_1d(0x400, 32)).unwrap();
            dma.configure_bd(1, BdConfig::simple_1d(0x800, 64)).unwrap();
            assert!(dma.enqueue_task(2, 0, 0, false));
            assert!(dma.enqueue_task(2, 1, 0, false));
        }

        let mut host_memory = crate::device::HostMemory::new();
        let mut guard = 0;
        loop {
            engine.device_mut().array.step_dma(0, 2, &mut host_memory).unwrap();
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.pop_stream_out_for_channel(2).is_some() {}
            let transfer = dma.get_transfer(2);
            if transfer.is_some_and(|transfer| {
                transfer.bd_index == 1
                    && transfer.bytes_transferred > 0
                    && transfer.bytes_transferred < transfer.total_bytes
            }) {
                break;
            }
            guard += 1;
            assert!(guard < 200, "fixture never reached partial progress on the second BD");
        }

        let detail = dma_channel_detail(&engine, 0, 2, 2).unwrap();
        let bd = detail.current_bd.as_ref().unwrap();
        assert_eq!(bd.id, 1);
        assert!(detail.bytes_transferred > 0);
        assert!(detail.bytes_transferred < detail.total_bytes);
        assert!(detail.current_address.unwrap() > bd.base_addr);
        assert_eq!(detail.remaining_bytes, detail.total_bytes - detail.bytes_transferred);
        assert_eq!(detail.transfers_completed, 1);
        assert!(detail.cycles_spent > 0);

        let snapshot = tile_snapshot(&engine, 0, 2).unwrap();
        let channel = snapshot.dma.iter().find(|channel| channel.index == 2).unwrap();
        assert!(channel.progress > 0.0 && channel.progress < 1.0);
    }
}
