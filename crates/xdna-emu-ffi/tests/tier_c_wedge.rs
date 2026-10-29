//! Tier C: verify the refactored run loop preserves the happy-path
//! Completed return. The full wedge -> WedgeRecovered cycle is exercised
//! at the device-state layer in Task 15 (the FFI-end-to-end wedge fixture
//! is deferred -- no wedging xclbin in our corpus today).

use xdna_emu::{xdna_emu_create, xdna_emu_destroy, xdna_emu_run, xdna_emu_set_max_cycles, XdnaEmuHaltReason};

#[test]
fn run_preserves_happy_path_completed_after_classifier_refactor() {
    // Minimal handle, no NPU instructions, no cores enabled -- the engine
    // halts immediately at warm-up, then the classifier hits NaturalCompletion
    // (syncs trivially satisfied: none pending). Refactor regression guard.
    let handle = unsafe { xdna_emu_create() };
    assert!(!handle.is_null());

    unsafe { xdna_emu_set_max_cycles(handle, 1000) };
    let status = unsafe { xdna_emu_run(handle) };

    assert!(matches!(status.halt_reason, XdnaEmuHaltReason::Completed), "got {:?}", status.halt_reason);

    unsafe { xdna_emu_destroy(handle) };
}
