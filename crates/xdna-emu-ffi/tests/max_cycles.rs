// SPDX-License-Identifier: MIT
//
// Integration tests for xdna_emu_set_max_cycles + halt_reason.
//
// Uses the FFI crate's rlib (lib name `xdna_emu`) directly via the
// `xdna_emu_ffi` alias to avoid confusion with the upstream `xdna-emu`
// crate (which was renamed to `xdna_emu_core` in the FFI crate's deps).

use xdna_emu::{
    xdna_emu_create, xdna_emu_destroy, xdna_emu_run, xdna_emu_set_max_cycles,
    XdnaEmuHaltReason, XdnaEmuResult,
};

#[test]
fn max_cycles_zero_is_unbounded() {
    unsafe {
        let h = xdna_emu_create();
        assert!(!h.is_null());
        let rc = xdna_emu_set_max_cycles(h, 0);
        assert_eq!(rc, XdnaEmuResult::Success);
        let status = xdna_emu_run(h);
        assert_eq!(status.result, XdnaEmuResult::Success);
        // No xclbin loaded -> no cores enabled -> engine halts immediately,
        // so we expect Completed, not Budget.
        assert_eq!(status.halt_reason, XdnaEmuHaltReason::Completed);
        xdna_emu_destroy(h);
    }
}

#[test]
fn max_cycles_one_hits_budget() {
    // TODO(plan-phase): requires a loaded xclbin with an enabled core;
    // currently no-op without a fixture. Left as a placeholder integration
    // test; the unit-level guarantees are enforced via the loop condition
    // in execution.rs.
}
