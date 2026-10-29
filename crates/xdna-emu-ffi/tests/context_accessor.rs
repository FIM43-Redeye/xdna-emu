use xdna_emu::*;

#[test]
fn get_context_state_returns_connected_for_default_context_after_create() {
    let handle = unsafe { xdna_emu_create() };
    assert!(!handle.is_null());

    let mut state: u32 = 99;
    let mut counter: u64 = 99;
    let rc = unsafe {
        xdna_emu_get_context_state(
            handle,
            0,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, 0, "expected Success");
    assert_eq!(state, XdnaEmuContextState::Connected as u32);
    assert_eq!(counter, 0);

    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn get_context_state_returns_invalid_for_unknown_context_id() {
    let handle = unsafe { xdna_emu_create() };
    let mut state: u32 = 0;
    let mut counter: u64 = 0;
    let rc = unsafe {
        xdna_emu_get_context_state(
            handle,
            999,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, -2, "expected invalid-context-id");
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn get_context_state_rejects_null_handle() {
    let mut state: u32 = 0;
    let mut counter: u64 = 0;
    let rc = unsafe {
        xdna_emu_get_context_state(
            std::ptr::null_mut(),
            0,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, -1, "expected null-handle");
}

#[test]
fn reset_context_with_default_id_succeeds_on_fresh_handle() {
    let handle = unsafe { xdna_emu_create() };
    let rc = unsafe { xdna_emu_reset_context(handle, 0) };
    assert_eq!(rc, XdnaEmuResult::Success);
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn reset_context_returns_error_for_invalid_id() {
    let handle = unsafe { xdna_emu_create() };
    let rc = unsafe { xdna_emu_reset_context(handle, 999) };
    assert_eq!(rc, XdnaEmuResult::ExecutionError);
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn reset_context_with_same_id_twice_succeeds() {
    // Idempotent: calling reset on an already-Connected context is fine.
    let handle = unsafe { xdna_emu_create() };
    let rc1 = unsafe { xdna_emu_reset_context(handle, 0) };
    assert_eq!(rc1, XdnaEmuResult::Success);
    let rc2 = unsafe { xdna_emu_reset_context(handle, 0) };
    assert_eq!(rc2, XdnaEmuResult::Success);
    unsafe { xdna_emu_destroy(handle) };
}
