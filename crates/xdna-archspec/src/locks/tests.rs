//! Tests for the LockModel trait and LockValueLayout carrier.
//!
//! The AIE2 concrete impl tests (and the regdb drift-detection test)
//! live alongside the impl in `aie2/locks.rs`.

use super::LockValueLayout;

/// Fixture matching AIE2's Lock_value field (width=6, mask=0x3F).
/// Duplicated here so the trait-side tests are independent of the AIE2
/// impl.
fn aie2_layout() -> LockValueLayout {
    LockValueLayout { width: 6, mask: 0x3F }
}

#[test]
fn decode_keeps_the_unsigned_field() {
    let layout = aie2_layout();
    assert_eq!(layout.decode(0), 0);
    assert_eq!(layout.decode(31), 31);
    // Bit 5 is a count bit, not a sign bit (silicon: 40 is a count of 40).
    assert_eq!(layout.decode(0x20), 32);
    assert_eq!(layout.decode(40), 40);
    assert_eq!(layout.decode(0x3F), 63);
}

#[test]
fn decode_masks_bits_above_the_field() {
    // Silicon: writing 0xFF reads back 0x3F.
    assert_eq!(aie2_layout().decode(0xFF), 63);
}

#[test]
fn max_is_the_field_mask() {
    assert_eq!(aie2_layout().max(), 63);
}
