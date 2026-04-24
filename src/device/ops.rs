//! Device-facing operation vocabulary (`DeviceOp`).
//!
//! `DeviceOp` is the arch-generic operation boundary between the
//! parser and `device::state`. Produced by
//! `crate::parser::cdo::semantics::lower()`; consumed by
//! `crate::device::state::apply()`.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md
//! §"DeviceOp vocabulary" for the design rationale (Option B naming:
//! existing `DmaDirection` reused, `u8` channel, fresh `TileAddr`).
//!
//! Stage 8b Half 2 Task 10: this module is defined but not yet
//! wired up. Task 11 rewrites `semantics::lower` to emit DeviceOp;
//! Task 12 migrates `device::state::apply` to consume it.

use smallvec::SmallVec;
use xdna_archspec::types::{DmaDirection, TileAddr};

/// Arch-generic device-facing operation.
///
/// Each variant maps to a well-defined hardware effect. Register-level
/// variants (`RegWrite`/`RegWrite64`/`RegMask`/`RegMask64`/`RegBurst`)
/// are the escape hatch: ~75% of CDO commands lower to one of these
/// and the emulator reassembles structured state (BDs, locks, routes)
/// on trigger-register writes the same way real silicon does.
///
/// Structured variants (`CoreEnable`, `DmaStart`) are local
/// promotions: a single register write to a well-known offset whose
/// effect is clearer as a typed op than as an opaque register write.
#[derive(Debug, Clone)]
pub enum DeviceOp {
    // --- Register-level writes (escape hatch; ~75% of CDO commands) ---

    /// Single 32-bit register write.
    /// Produced by: `CdoRaw::Write`, and by `CdoRaw::Write64` after
    /// address truncation (see `RegMask` doc for the CDO-Write64
    /// width-naming note).
    RegWrite { tile: TileAddr, offset: u32, value: u32 },

    /// Masked register write: `*reg = (*reg & !mask) | (value & mask)`.
    /// Produced by: `CdoRaw::MaskWrite`, and by `CdoRaw::MaskWrite64`
    /// after address truncation (Write64/MaskWrite64 in CDO mean
    /// 64-bit *address*, not 64-bit value -- the high 32 bits of the
    /// address are always zero for AIE and the written value is u32).
    RegMask { tile: TileAddr, offset: u32, mask: u32, value: u32 },

    /// Bulk write: `words.len()` consecutive 32-bit words starting at
    /// `offset`.
    ///
    /// Inline `SmallVec` capacity is 64 words (256 bytes), sized
    /// generously to avoid heap allocation for typical NPU burst
    /// patterns (BD configuration, program loads). Larger bursts spill
    /// to the heap.
    /// Produced by: `CdoRaw::DmaWrite`.
    RegBurst { tile: TileAddr, offset: u32, words: SmallVec<[u32; 64]> },

    // --- Coarse control (local single-register promotions) ---

    /// Enable or disable a compute core.
    /// Produced by: `CdoRaw::Write` or `CdoRaw::MaskWrite` targeting
    /// the Core_Control register.
    ///
    /// `enabled` is the decoded enable state (bit 0 of the final value);
    /// `value` is the raw 32-bit register value (or, for MaskWrite, the
    /// mask-blended value) that state stores into `tile.core.control`
    /// for readback-correctness. Carrying the raw word keeps the typed
    /// op self-describing so `device::state` never has to re-derive it.
    CoreEnable { tile: TileAddr, enabled: bool, value: u32 },

    /// Start a DMA channel.
    /// Produced by: `CdoRaw::Write` targeting a DMA channel Start_Queue
    /// register.
    ///
    /// `bd_id` is the full 32-bit value written to Start_Queue; it
    /// carries the starting BD index (via `start_bd_id` field) plus
    /// repeat count, token-issue enable, and other channel-start
    /// parameters that `device::state` decodes before enqueueing the
    /// task. Passing the raw value avoids splitting field extraction
    /// between the parser and state.
    DmaStart { tile: TileAddr, channel: u8, dir: DmaDirection, bd_id: u32 },

    // --- Synchronization / timing ---

    /// Poll a register until `(value & mask) == expected`.
    ///
    /// On real hardware: blocks until the condition is met (e.g. DMA
    /// completion). In the emulator: currently a logged no-op because
    /// register writes are synchronous; retained to preserve semantic
    /// information for future cycle-accurate work.
    /// Produced by: `CdoRaw::MaskPoll`, `CdoRaw::MaskPoll64`.
    MaskPoll { tile: TileAddr, offset: u32, mask: u32, expected: u32 },

    /// Timing delay for N cycles.
    ///
    /// On real hardware: inserts a wait. In the emulator: no-op.
    /// Produced by: `CdoRaw::Delay`.
    Delay { cycles: u32 },

    /// Debug sequence marker. `value` is an opaque tag; always a no-op
    /// in `device::state`, useful for trace and test tooling.
    /// Produced by: `CdoRaw::Marker`.
    Marker { value: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(col: u8, row: u8) -> TileAddr {
        TileAddr::new(col, row)
    }

    #[test]
    fn reg_write_constructs() {
        let op = DeviceOp::RegWrite { tile: t(0, 2), offset: 0x1000, value: 0xDEAD };
        match op {
            DeviceOp::RegWrite { tile, offset, value } => {
                assert_eq!(tile, t(0, 2));
                assert_eq!(offset, 0x1000);
                assert_eq!(value, 0xDEAD);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn reg_mask_applies_shape() {
        let op = DeviceOp::RegMask { tile: t(2, 1), offset: 0x100, mask: 0xFF00, value: 0x5500 };
        if let DeviceOp::RegMask { mask, value, .. } = op {
            assert_eq!(mask, 0xFF00);
            assert_eq!(value, 0x5500);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn reg_burst_inline_capacity_avoids_heap() {
        // Under 64 words: stays inline.
        let words: SmallVec<[u32; 64]> = (0..32).collect();
        let op = DeviceOp::RegBurst { tile: t(0, 2), offset: 0x0, words };
        if let DeviceOp::RegBurst { words, .. } = op {
            assert_eq!(words.len(), 32);
            assert!(!words.spilled());
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn reg_burst_large_spills_to_heap() {
        // Over 64 words: spills.
        let words: SmallVec<[u32; 64]> = (0..128).collect();
        let op = DeviceOp::RegBurst { tile: t(0, 2), offset: 0x0, words };
        if let DeviceOp::RegBurst { words, .. } = op {
            assert_eq!(words.len(), 128);
            assert!(words.spilled());
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn core_enable_variant() {
        let on = DeviceOp::CoreEnable { tile: t(0, 2), enabled: true, value: 0x1 };
        let off = DeviceOp::CoreEnable { tile: t(0, 2), enabled: false, value: 0x0 };
        assert!(matches!(on, DeviceOp::CoreEnable { enabled: true, value: 0x1, .. }));
        assert!(matches!(off, DeviceOp::CoreEnable { enabled: false, value: 0x0, .. }));
    }

    #[test]
    fn dma_start_both_directions() {
        let s2mm = DeviceOp::DmaStart {
            tile: t(0, 0),
            channel: 0,
            dir: DmaDirection::S2mm,
            bd_id: 0x7,
        };
        let mm2s = DeviceOp::DmaStart {
            tile: t(0, 0),
            channel: 1,
            dir: DmaDirection::Mm2s,
            bd_id: 0x3,
        };
        assert!(matches!(
            s2mm,
            DeviceOp::DmaStart { dir: DmaDirection::S2mm, channel: 0, bd_id: 0x7, .. }
        ));
        assert!(matches!(
            mm2s,
            DeviceOp::DmaStart { dir: DmaDirection::Mm2s, channel: 1, bd_id: 0x3, .. }
        ));
    }

    #[test]
    fn mask_poll_constructs() {
        let op = DeviceOp::MaskPoll {
            tile: t(0, 2),
            offset: 0x1D004,
            mask: 0x3,
            expected: 0x2,
        };
        if let DeviceOp::MaskPoll { mask, expected, .. } = op {
            assert_eq!(mask, 0x3);
            assert_eq!(expected, 0x2);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn delay_constructs() {
        let op = DeviceOp::Delay { cycles: 100 };
        assert!(matches!(op, DeviceOp::Delay { cycles: 100 }));
    }

    #[test]
    fn marker_constructs() {
        let op = DeviceOp::Marker { value: 0xDEAD_BEEF };
        assert!(matches!(op, DeviceOp::Marker { value: 0xDEAD_BEEF }));
    }

    #[test]
    fn is_clone() {
        let op = DeviceOp::RegWrite { tile: t(0, 0), offset: 0, value: 0 };
        let _cloned = op.clone();
    }
}
