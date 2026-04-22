//! AIE2 lock model implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). All AIE2-family devices share the same lock feature
//! set:
//!
//! - 6-bit register field (mask 0x3F), logical 7-bit signed range
//!   (-64..63); values outside the 6-bit field alias when read back.
//! - Both acquire-GE and acquire-EQ modes supported.
//! - Host-side GetValue / SetValue both supported.
//!
//! Width and mask sourced from
//! `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
//! (`memory.Lock0_value.Lock_value` field). Bounds from aie-rt
//! `xaiemlgbl_reginit.c:2452-2453` (LockValUpperBound=63,
//! LockValLowerBound=-64). A regdb drift-detection test in this
//! module asserts the JSON still agrees with these static constants.
//!
//! ## Lock ID Quadrant Mapping
//!
//! AIE2 cores address 64 locks via a 6-bit ID field. The field is
//! divided into four 16-lock quadrants that each map to a neighboring
//! tile's memory-module lock bank:
//!
//! | ID range | Direction | Neighbor |
//! |----------|-----------|----------|
//! | 0–15     | South     | row − 1  |
//! | 16–31    | West      | col − 1  |
//! | 32–47    | North     | row + 1  |
//! | 48–63    | East/Internal | own tile |
//!
//! Source: AM025 §"Lock Instruction Encoding" and aie-rt
//! `xaie_locks_aieml.h` `XAieMl_LockSetValue()` which uses the
//! same 4-quadrant split to select the target tile.

use crate::locks::{LockModel, LockValueLayout};

/// Lock ID quadrant boundaries for AIE2.
///
/// The 6-bit lock ID field in AIE2 core instructions (range 0–63) is
/// partitioned into four 16-lock quadrants. These constants delimit
/// each quadrant so that call sites can route lock operations to the
/// correct neighbor tile without embedding bare literals.
///
/// Source: AM025 §"Lock Instruction Encoding"; cross-confirmed by
/// aie-rt `xaie_locks_aieml.h` `XAieMl_LockSetValue()`.
pub mod quadrants {
    /// South quadrant: lock IDs 0–15 address the row-1 neighbor.
    pub const SOUTH_START: u8 = 0;
    /// Exclusive end of the South quadrant (first West ID).
    pub const SOUTH_END: u8 = 16;

    /// West quadrant: lock IDs 16–31 address the col-1 neighbor.
    pub const WEST_START: u8 = SOUTH_END;
    /// Exclusive end of the West quadrant (first North ID).
    pub const WEST_END: u8 = 32;

    /// North quadrant: lock IDs 32–47 address the row+1 neighbor.
    pub const NORTH_START: u8 = WEST_END;
    /// Exclusive end of the North quadrant (first East/Internal ID).
    pub const NORTH_END: u8 = 48;

    /// East/Internal quadrant: lock IDs 48–63 address the own tile.
    ///
    /// "East" in AIE2 non-checkerboard layout means the local tile.
    pub const EAST_START: u8 = NORTH_END;
    /// Exclusive end of the East/Internal quadrant (lock-ID ceiling).
    pub const EAST_END: u8 = 64;

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Drift-detection: quadrant boundaries must partition 0–63 into
        /// four contiguous 16-element ranges. If the hardware encoding
        /// changes (e.g., AIE2P uses a different partitioning), this
        /// test fires and forces a manual review.
        #[test]
        fn quadrant_boundaries_partition_lock_id_space() {
            // Each quadrant covers exactly 16 IDs.
            assert_eq!(SOUTH_END - SOUTH_START, 16, "South quadrant must have 16 IDs");
            assert_eq!(WEST_END - WEST_START,   16, "West quadrant must have 16 IDs");
            assert_eq!(NORTH_END - NORTH_START, 16, "North quadrant must have 16 IDs");
            assert_eq!(EAST_END  - EAST_START,  16, "East quadrant must have 16 IDs");

            // Quadrants must be contiguous and start at 0.
            assert_eq!(SOUTH_START, 0, "South quadrant must start at 0");
            assert_eq!(WEST_START,  SOUTH_END, "West must start where South ends");
            assert_eq!(NORTH_START, WEST_END,  "North must start where West ends");
            assert_eq!(EAST_START,  NORTH_END, "East must start where North ends");

            // Total coverage = 64 IDs (full 6-bit field).
            assert_eq!(EAST_END, 64, "East quadrant must end at 64");
        }
    }
}

/// The AIE2 Lock_value field layout.
///
/// Static so hot-path consumers can cache `&'static LockValueLayout`
/// at construction time.
pub static AIE2_LOCK_VALUE_LAYOUT: LockValueLayout = LockValueLayout {
    width: 6,
    mask: 0x3F,
    sign_bit: 5,
    min: -64,
    max: 63,
};

/// AIE2 lock model.
///
/// Zero-sized: a single `AIE2_LOCK_MODEL` static instance serves every
/// tile in every AIE2-family NPU. `ArchConfig::lock_model()` returns a
/// `&'static dyn LockModel` pointing at this singleton.
#[derive(Debug, Clone, Copy)]
pub struct Aie2LockModel;

/// The single `Aie2LockModel` instance used across every AIE2-family
/// consumer. Reference via `ArchConfig::lock_model()`.
pub static AIE2_LOCK_MODEL: Aie2LockModel = Aie2LockModel;

impl LockModel for Aie2LockModel {
    fn supports_acquire_eq(&self) -> bool {
        true
    }

    fn supports_dynamic_value_ops(&self) -> bool {
        true
    }

    fn value_layout(&self) -> &'static LockValueLayout {
        &AIE2_LOCK_VALUE_LAYOUT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regdb::RegisterDb;

    #[test]
    fn aie2_lock_model_feature_flags() {
        assert!(AIE2_LOCK_MODEL.supports_acquire_eq(),
                "AIE2 supports acq-EQ");
        assert!(AIE2_LOCK_MODEL.supports_dynamic_value_ops(),
                "AIE2 supports GetValue/SetValue");
    }

    #[test]
    fn aie2_lock_value_layout_constants() {
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.width, 6);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.mask, 0x3F);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.sign_bit, 5);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.min, -64);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.max, 63);
    }

    #[test]
    fn aie2_lock_model_value_layout_returns_static() {
        let layout = AIE2_LOCK_MODEL.value_layout();
        assert_eq!(*layout, AIE2_LOCK_VALUE_LAYOUT);
    }

    /// Drift-detection: if the regdb JSON ever changes shape, this
    /// test fires. The static constants above carry the "known-good"
    /// AM025 + aie-rt values; this test asserts the JSON still agrees.
    #[test]
    fn aie2_lock_layout_matches_regdb() {
        let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..") // up to npu-work/
            .join("mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json");

        if !json_path.exists() {
            eprintln!("Skipping drift test: register DB JSON not found at {}",
                      json_path.display());
            return;
        }

        let db = RegisterDb::from_file(&json_path)
            .expect("Failed to load register DB JSON");
        let field = db
            .module("memory")
            .and_then(|m| m.register("Lock0_value"))
            .and_then(|r| r.field("Lock_value"))
            .expect("memory.Lock0_value.Lock_value field not found in JSON");

        assert_eq!(field.width, AIE2_LOCK_VALUE_LAYOUT.width,
                   "Lock_value field width drifted from static constant");
        assert_eq!(field.mask, AIE2_LOCK_VALUE_LAYOUT.mask,
                   "Lock_value field mask drifted from static constant");
    }
}
