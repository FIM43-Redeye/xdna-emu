//! Egui-free view-model for the visual debugger: pure functions that project
//! live emulator state (`TileArray`, `InterpreterEngine`) into plain structs
//! the GUI layer can render without touching device internals directly.

use crate::device::array::TileArray;
use crate::device::tile::Tile;
use crate::interpreter::InterpreterEngine;

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
    use crate::interpreter::InterpreterEngine;

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
