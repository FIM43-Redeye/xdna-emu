//! Tests for the LockModel trait and LockValueLayout carrier.
//!
//! The AIE2 concrete impl tests (and the regdb drift-detection test)
//! live alongside the impl in `aie2/locks.rs`.

use super::LockValueLayout;

/// Fixture matching AIE2's Lock_value field (width=6, mask=0x3F,
/// sign_bit=5, min=-64, max=63). Duplicated here so the trait-side
/// tests are independent of the AIE2 impl.
fn aie2_layout() -> LockValueLayout {
    LockValueLayout { width: 6, mask: 0x3F, sign_bit: 5, min: -64, max: 63 }
}

#[test]
fn sign_extend_zero_is_zero() {
    assert_eq!(aie2_layout().sign_extend(0), 0);
}

#[test]
fn sign_extend_max_positive() {
    // 0x1F = 31 -- max positive for 6-bit signed is 31.
    assert_eq!(aie2_layout().sign_extend(31), 31);
}

#[test]
fn sign_extend_min_negative_for_field() {
    // 0x20 = 0b100000 -- sign bit set, all-zeros payload -> -32.
    assert_eq!(aie2_layout().sign_extend(0x20), -32);
}

#[test]
fn sign_extend_all_bits_set_is_minus_one() {
    // 0x3F = all 6 bits set -> -1.
    assert_eq!(aie2_layout().sign_extend(0x3F), -1);
}

#[test]
fn sign_extend_masks_extra_bits() {
    // 0xFF = all 8 bits set; upper bits outside mask must be ignored
    // before sign-extend.
    assert_eq!(aie2_layout().sign_extend(0xFF), -1);
}
