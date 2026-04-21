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

use crate::locks::{LockModel, LockValueLayout};

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
