//! DMA subsystem signal mapping.
//!
//! Maps aiesimulator VCD signal paths under the `dma` scope to
//! [`StatePath`] values for both compute/mem tiles and shim tiles.
//!
//! # VCD hierarchy
//!
//! For mem tiles and shim tiles, DMA channels are directly under the `dma`
//! scope of each tile:
//!
//! ```text
//! tile_0_1.dma.s2mm_state0.cur_bd
//! tile_0_1.dma.s2mm_state0.status
//! tile_0_1.dma.mm2s_state1.address
//! ```
//!
//! For compute tiles (array group), there is an additional `mm` scope level
//! between the tile and the `dma` scope:
//!
//! ```text
//! tile_0_3.mm.dma.s2mm_state0.cur_bd
//! ```
//!
//! This module provides a single [`dma_mapping`] function that covers the
//! `dma` scope in both cases. The caller is responsible for registering it
//! with the correct scope chain (i.e., registering a separate `mm` subsystem
//! wrapper for compute tiles is handled upstream in the mapping tree builder).
//!
//! # Signal coverage
//!
//! Each channel group (`s2mm_stateN` / `mm2s_stateN`) contains ~20 fields.
//! The mapping covers every field that has a corresponding [`StatePath`]
//! variant. Fields present in the VCD but not yet in [`StatePath`]
//! (e.g. `cur_bd_enable_acquire`, `lanes`, `channel_running`) are intentionally
//! omitted and will be flagged by the coverage audit (Task 6).

use crate::vcd::mapping::{NestedSignalFactory, SubsystemMapping};
use crate::vcd::state_path::{DmaDir, StatePath, Subsystem};

// ---------------------------------------------------------------------------
// Signal table helpers
// ---------------------------------------------------------------------------

/// Build the list of `(name, width, factory)` child signal definitions for
/// one DMA channel direction. `dir` is baked into the factory closures.
///
/// The order matches the aiesimulator VCD declaration order for readability,
/// though the mapping is order-independent at runtime.
fn channel_signals(dir: DmaDir) -> Vec<(&'static str, u32, NestedSignalFactory)> {
    match dir {
        DmaDir::S2mm => s2mm_signals(),
        DmaDir::Mm2s => mm2s_signals(),
    }
}

/// BD state and channel signals for S2MM (stream-to-memory, inbound) channels.
///
/// Field names are taken verbatim from aiesimulator VCD output (ground truth).
/// Width values are informational and match the declared VCD wire widths.
fn s2mm_signals() -> Vec<(&'static str, u32, NestedSignalFactory)> {
    vec![
        // BD descriptor fields
        ("cur_bd", 32, |col, row, ch| StatePath::DmaCurrentBd {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_valid", 1, |col, row, ch| StatePath::DmaBdValid {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_lock_acq_ID: lock to acquire before starting the BD
        ("cur_bd_lock_acq_ID", 32, |col, row, ch| StatePath::DmaLockAcqId {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_acquire_value: value to acquire the lock at
        ("cur_bd_acquire_value", 7, |col, row, ch| StatePath::DmaLockAcqValue {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_release_value: value to release the lock to after BD completes
        ("cur_bd_release_value", 7, |col, row, ch| StatePath::DmaLockRelValue {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_length", 32, |col, row, ch| StatePath::DmaBdLength {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_next_BD: next BD index for linked-list chaining
        ("cur_bd_next_BD", 32, |col, row, ch| StatePath::DmaNextBd {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_use_next_BD: whether to follow the next-BD chain
        ("cur_bd_use_next_BD", 1, |col, row, ch| StatePath::DmaUseNextBd {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_tlast_suppress", 1, |col, row, ch| StatePath::DmaTlastSuppress {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_iteration_stepsize", 32, |col, row, ch| StatePath::DmaIterStepsize {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_iteration_current", 32, |col, row, ch| StatePath::DmaIterCurrent {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_iteration_wrap", 32, |col, row, ch| StatePath::DmaIterWrap {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // cur_bd_packet_ID: packet ID in packet-mode transfers
        ("cur_bd_packet_ID", 32, |col, row, ch| StatePath::DmaPacketId {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("cur_bd_enable_packet", 1, |col, row, ch| StatePath::DmaEnablePacket {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        // Channel status and pipeline observability signals
        ("status", 32, |col, row, ch| StatePath::DmaStatus {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("processed_stream", 32, |col, row, ch| StatePath::DmaProcessedStream {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("processed_mem", 32, |col, row, ch| StatePath::DmaProcessedMem {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("address", 32, |col, row, ch| StatePath::DmaAddress {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
        ("data", 128, |col, row, ch| StatePath::DmaData {
            col,
            row,
            dir: DmaDir::S2mm,
            ch,
        }),
    ]
}

/// BD state and channel signals for MM2S (memory-to-stream, outbound) channels.
///
/// The field set is identical to S2MM; only the direction discriminant differs.
fn mm2s_signals() -> Vec<(&'static str, u32, NestedSignalFactory)> {
    vec![
        ("cur_bd", 32, |col, row, ch| StatePath::DmaCurrentBd {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_valid", 1, |col, row, ch| StatePath::DmaBdValid {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_lock_acq_ID", 32, |col, row, ch| StatePath::DmaLockAcqId {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_acquire_value", 7, |col, row, ch| StatePath::DmaLockAcqValue {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_release_value", 7, |col, row, ch| StatePath::DmaLockRelValue {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_length", 32, |col, row, ch| StatePath::DmaBdLength {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_next_BD", 32, |col, row, ch| StatePath::DmaNextBd {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_use_next_BD", 1, |col, row, ch| StatePath::DmaUseNextBd {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_tlast_suppress", 1, |col, row, ch| StatePath::DmaTlastSuppress {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_iteration_stepsize", 32, |col, row, ch| StatePath::DmaIterStepsize {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_iteration_current", 32, |col, row, ch| StatePath::DmaIterCurrent {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_iteration_wrap", 32, |col, row, ch| StatePath::DmaIterWrap {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_packet_ID", 32, |col, row, ch| StatePath::DmaPacketId {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("cur_bd_enable_packet", 1, |col, row, ch| StatePath::DmaEnablePacket {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("status", 32, |col, row, ch| StatePath::DmaStatus {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("processed_stream", 32, |col, row, ch| StatePath::DmaProcessedStream {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("processed_mem", 32, |col, row, ch| StatePath::DmaProcessedMem {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("address", 32, |col, row, ch| StatePath::DmaAddress {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
        ("data", 128, |col, row, ch| StatePath::DmaData {
            col,
            row,
            dir: DmaDir::Mm2s,
            ch,
        }),
    ]
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the DMA subsystem mapping for a compute or mem tile.
///
/// Returns a [`SubsystemMapping`] covering the VCD `dma` scope. Each channel
/// group `s2mm_stateN` (0..`s2mm_count`) and `mm2s_stateN` (0..`mm2s_count`)
/// contains 19 mapped signals.
///
/// # Channel counts per tile type
///
/// Tile type | S2MM | MM2S
/// ----------|------|-----
/// Compute   |  2   |  2
/// Mem       |  6   |  6 (AIE2 mem tile)
///
/// These counts come from the mlir-aie device model
/// (`tools/aie-device-models.json`).
pub fn dma_mapping(s2mm_count: u8, mm2s_count: u8) -> SubsystemMapping {
    SubsystemMapping::new("dma", Subsystem::Dma)
        .nested_group("s2mm_state", s2mm_count, channel_signals(DmaDir::S2mm))
        .nested_group("mm2s_state", mm2s_count, channel_signals(DmaDir::Mm2s))
}

/// Build the DMA subsystem mapping for a shim tile.
///
/// Shim tiles expose a `dma` scope with the same channel group structure as
/// compute/mem tiles, but with different channel counts and additional
/// AXI-MM-specific signals (which are not yet in [`StatePath`] and will be
/// flagged by the coverage audit).
///
/// # Channel counts per shim tile type (AIE2 Phoenix)
///
/// The Phoenix shim row has 2 S2MM and 2 MM2S channels per tile per the
/// mlir-aie device model.
pub fn shim_dma_mapping(s2mm_count: u8, mm2s_count: u8) -> SubsystemMapping {
    // Shim DMA uses the same VCD scope and field structure as tile DMA.
    // Shim-specific signals (AXI burst state, req_addr, etc.) are not yet
    // covered by StatePath -- the coverage audit will enumerate them.
    dma_mapping(s2mm_count, mm2s_count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    // -- Resolution tests --

    #[test]
    fn dma_resolves_s2mm_cur_bd() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaCurrentBd {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_ch1_cur_bd() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state1", "cur_bd"], 1, 2);
        assert_eq!(
            result,
            Some(StatePath::DmaCurrentBd {
                col: 1,
                row: 2,
                dir: DmaDir::S2mm,
                ch: 1,
            })
        );
    }

    #[test]
    fn dma_resolves_mm2s_status() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["mm2s_state1", "status"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaStatus {
                col: 0,
                row: 1,
                dir: DmaDir::Mm2s,
                ch: 1,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_bd_length() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_length"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaBdLength {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_lock_acq_id() {
        // Real VCD field name: cur_bd_lock_acq_ID (capital ID)
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_lock_acq_ID"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaLockAcqId {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_acquire_value() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_acquire_value"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaLockAcqValue {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_release_value() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_release_value"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaLockRelValue {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_next_bd() {
        // Real VCD field name: cur_bd_next_BD (capital BD)
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_next_BD"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaNextBd {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_use_next_bd() {
        // Real VCD field name: cur_bd_use_next_BD (capital BD)
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_use_next_BD"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaUseNextBd {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_tlast_suppress() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_tlast_suppress"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaTlastSuppress {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_packet_id() {
        // Real VCD field name: cur_bd_packet_ID (capital ID)
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_packet_ID"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaPacketId {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_enable_packet() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_enable_packet"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaEnablePacket {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_iter_fields() {
        let mapping = dma_mapping(2, 2);
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_iteration_stepsize"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaIterStepsize {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_iteration_current"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaIterCurrent {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let result = mapping.resolve(&["s2mm_state0", "cur_bd_iteration_wrap"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaIterWrap {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_s2mm_processed_and_addr_data() {
        let mapping = dma_mapping(2, 2);

        let r = mapping.resolve(&["s2mm_state0", "processed_stream"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaProcessedStream {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let r = mapping.resolve(&["s2mm_state0", "processed_mem"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaProcessedMem {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let r = mapping.resolve(&["s2mm_state0", "address"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaAddress {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let r = mapping.resolve(&["s2mm_state0", "data"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaData {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn dma_resolves_mm2s_channel_fields() {
        let mapping = dma_mapping(2, 2);

        let r = mapping.resolve(&["mm2s_state0", "cur_bd"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaCurrentBd {
                col: 0,
                row: 1,
                dir: DmaDir::Mm2s,
                ch: 0,
            })
        );
        let r = mapping.resolve(&["mm2s_state0", "address"], 0, 1);
        assert_eq!(
            r,
            Some(StatePath::DmaAddress {
                col: 0,
                row: 1,
                dir: DmaDir::Mm2s,
                ch: 0,
            })
        );
        let r = mapping.resolve(&["mm2s_state1", "data"], 3, 2);
        assert_eq!(
            r,
            Some(StatePath::DmaData {
                col: 3,
                row: 2,
                dir: DmaDir::Mm2s,
                ch: 1,
            })
        );
    }

    // -- Boundary and rejection tests --

    #[test]
    fn dma_rejects_out_of_range_channel() {
        let mapping = dma_mapping(2, 2);
        // Channel 2 is out of range for count=2 (valid: 0, 1).
        assert_eq!(mapping.resolve(&["s2mm_state2", "cur_bd"], 0, 0), None);
        assert_eq!(mapping.resolve(&["mm2s_state2", "cur_bd"], 0, 0), None);
    }

    #[test]
    fn dma_rejects_unknown_field() {
        let mapping = dma_mapping(2, 2);
        assert_eq!(
            mapping.resolve(&["s2mm_state0", "nonexistent_field"], 0, 0),
            None
        );
    }

    #[test]
    fn dma_rejects_wrong_scope_name() {
        let mapping = dma_mapping(2, 2);
        // Only accepts "dma" as scope.
        assert_eq!(mapping.resolve(&["cur_bd"], 0, 0), None);
        assert_eq!(
            mapping.resolve(&["s2mm_state0", "cur_bd", "extra"], 0, 0),
            None
        );
    }

    #[test]
    fn dma_scope_name_is_dma() {
        // The VCD scope name must be "dma" so the mapping tree can route to it.
        let mapping = dma_mapping(2, 2);
        assert_eq!(mapping.scope_name(), "dma");
    }

    // -- Enumeration tests --

    #[test]
    fn dma_enumerates_all_channels() {
        let mapping = dma_mapping(2, 2);
        let paths = mapping.enumerate(0, 1);
        // 19 signals per channel, 2 S2MM + 2 MM2S channels = 76 paths
        assert_eq!(paths.len(), 76);
    }

    #[test]
    fn dma_enumerate_minimum_40_signals() {
        // Task requirement: at least 10 fields per channel * 4 channels = 40.
        let mapping = dma_mapping(2, 2);
        let paths = mapping.enumerate(0, 1);
        assert!(paths.len() >= 40);
    }

    #[test]
    fn dma_enumerates_6_channel_memtile() {
        // Mem tiles have 6 S2MM and 6 MM2S channels.
        let mapping = dma_mapping(6, 6);
        let paths = mapping.enumerate(0, 1);
        // 19 signals x 12 channels = 228
        assert_eq!(paths.len(), 228);
    }

    #[test]
    fn dma_enumerate_contains_both_directions() {
        let mapping = dma_mapping(2, 2);
        let paths = mapping.enumerate(0, 1);

        let s2mm_count = paths
            .iter()
            .filter(|p| matches!(p, StatePath::DmaCurrentBd { dir: DmaDir::S2mm, .. }))
            .count();
        let mm2s_count = paths
            .iter()
            .filter(|p| matches!(p, StatePath::DmaCurrentBd { dir: DmaDir::Mm2s, .. }))
            .count();
        // One DmaCurrentBd per channel per direction.
        assert_eq!(s2mm_count, 2);
        assert_eq!(mm2s_count, 2);
    }

    #[test]
    fn dma_enumerate_tile_coordinates_propagated() {
        let mapping = dma_mapping(2, 2);
        let paths_a = mapping.enumerate(0, 1);
        let paths_b = mapping.enumerate(3, 5);

        // No overlap: all paths from tile (0,1) have col=0 row=1.
        assert!(paths_a.iter().all(|p| p.tile() == (0, 1)));
        assert!(paths_b.iter().all(|p| p.tile() == (3, 5)));
    }

    // -- Shim DMA tests --

    #[test]
    fn shim_dma_has_same_structure() {
        let shim = shim_dma_mapping(2, 2);
        let tile = dma_mapping(2, 2);
        assert_eq!(shim.enumerate(0, 0).len(), tile.enumerate(0, 1).len());
    }

    #[test]
    fn shim_dma_resolves_signals() {
        let shim = shim_dma_mapping(2, 2);
        let result = shim.resolve(&["s2mm_state0", "cur_bd"], 0, 0);
        assert_eq!(
            result,
            Some(StatePath::DmaCurrentBd {
                col: 0,
                row: 0,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
    }

    #[test]
    fn shim_dma_scope_name_is_dma() {
        let shim = shim_dma_mapping(2, 2);
        assert_eq!(shim.scope_name(), "dma");
    }
}
