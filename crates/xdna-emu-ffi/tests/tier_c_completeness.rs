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

#[test]
fn firmware_component_signatures_match_the_public_contract() {
    type FnLoad = unsafe extern "C" fn(*mut XdnaEmuHandle, *const u8, u64) -> XdnaEmuResult;
    type FnBoot = unsafe extern "C" fn(*mut XdnaEmuHandle, u64) -> XdnaEmuResult;
    type FnRead = unsafe extern "C" fn(*mut XdnaEmuHandle, u32, *mut u32) -> XdnaEmuResult;
    type FnWrite = unsafe extern "C" fn(*mut XdnaEmuHandle, u32, u32) -> XdnaEmuResult;
    type FnMap = unsafe extern "C" fn(*mut XdnaEmuHandle, u64, *mut u8, u64) -> XdnaEmuResult;
    type FnUnmap = unsafe extern "C" fn(*mut XdnaEmuHandle, u64, u64) -> XdnaEmuResult;

    let _: FnLoad = xdna_emu_load_firmware;
    let _: FnBoot = xdna_emu_boot_firmware;
    let _: FnRead = xdna_emu_firmware_read_host_sram32;
    let _: FnWrite = xdna_emu_firmware_write_host_sram32;
    let _: FnRead = xdna_emu_firmware_read_host32;
    let _: FnWrite = xdna_emu_firmware_write_host32;
    let _: FnMap = xdna_emu_map_host_memory;
    let _: FnUnmap = xdna_emu_unmap_host_memory;
}
