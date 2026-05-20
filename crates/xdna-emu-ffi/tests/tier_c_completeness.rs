//! Tier C completeness: lock the FFI surface against drift.

use xdna_emu::*;

#[test]
fn halt_reason_wedge_recovered_has_discriminant_four() {
    assert_eq!(XdnaEmuHaltReason::WedgeRecovered as u32, 4);
}

#[test]
fn context_state_discriminants_match_spec() {
    assert_eq!(XdnaEmuContextState::Connected as u32, 0);
    assert_eq!(XdnaEmuContextState::Stopped as u32, 1);
    assert_eq!(XdnaEmuContextState::Failed as u32, 2);
}

#[test]
fn reset_context_signature_takes_context_id() {
    // Type-level check: the fn pointer must accept (handle, context_id).
    type FnReset = unsafe extern "C" fn(*mut XdnaEmuHandle, u32) -> XdnaEmuResult;
    let _: FnReset = xdna_emu_reset_context;
}

#[test]
fn get_context_state_signature_matches_spec() {
    type FnGet = unsafe extern "C" fn(*mut XdnaEmuHandle, u32, *mut XdnaEmuContextState, *mut u64) -> i32;
    let _: FnGet = xdna_emu_get_context_state;
}
