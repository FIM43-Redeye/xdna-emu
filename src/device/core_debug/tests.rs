//! Tests for the core debug subsystem.

use super::*;

// -- Initial state --

#[test]
fn initial_state_is_reset_disabled() {
    let state = CoreDebugState::new();
    assert!(!state.is_enabled(), "should start disabled");
    assert!(!state.is_halted(), "should start not halted");
    assert!(!state.is_done(), "should start not done");
    assert!(state.is_reset(), "should start in reset");
    assert_eq!(state.read_pc(), 0);
    assert_eq!(state.read_sp(), 0);
    assert_eq!(state.read_lr(), 0);
}

#[test]
fn initial_status_register_shows_reset() {
    let state = CoreDebugState::new();
    let status = state.read_status();
    // Reset bit should be set (bit 1), enable bit clear (bit 0).
    assert_eq!(status & (1 << STATUS_RESET_LSB), 1 << STATUS_RESET_LSB);
    assert_eq!(status & (1 << STATUS_ENABLE_LSB), 0);
}

// -- Status register packing --

#[test]
fn status_packs_enable_bit() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_ENABLE_LSB), 0);
    assert_eq!(status & (1 << STATUS_RESET_LSB), 0);
}

#[test]
fn status_packs_done_bit() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.done = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_DONE_LSB), 0);
}

#[test]
fn status_packs_debug_halt() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.halted = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_DEBUG_HALT_LSB), 0);
}

#[test]
fn status_packs_memory_stall_all_directions() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.mem_stall = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_MEMORY_STALL_S_LSB), 0);
    assert_ne!(status & (1 << STATUS_MEMORY_STALL_W_LSB), 0);
    assert_ne!(status & (1 << STATUS_MEMORY_STALL_N_LSB), 0);
    assert_ne!(status & (1 << STATUS_MEMORY_STALL_E_LSB), 0);
}

#[test]
fn status_packs_lock_stall_all_directions() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.lock_stall = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_LOCK_STALL_S_LSB), 0);
    assert_ne!(status & (1 << STATUS_LOCK_STALL_W_LSB), 0);
    assert_ne!(status & (1 << STATUS_LOCK_STALL_N_LSB), 0);
    assert_ne!(status & (1 << STATUS_LOCK_STALL_E_LSB), 0);
}

#[test]
fn status_packs_stream_stall() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.stream_stall = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_STREAM_STALL_SS0_LSB), 0);
    assert_ne!(status & (1 << STATUS_STREAM_STALL_MS0_LSB), 0);
}

#[test]
fn status_packs_cascade_stall() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.cascade_stall = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_CASCADE_STALL_SCD_LSB), 0);
    assert_ne!(status & (1 << STATUS_CASCADE_STALL_MCD_LSB), 0);
}

#[test]
fn status_packs_ecc_error() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.ecc_error = true;
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_ECC_ERROR_STALL_LSB), 0);
    assert_ne!(status & (1 << STATUS_ERROR_HALT_LSB), 0);
}

#[test]
fn status_no_stalls_when_clear() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    let status = state.read_status();
    assert_eq!(status, 0);
}

// -- Control register --

#[test]
fn control_enable_sets_enabled() {
    let mut state = CoreDebugState::new();
    state.write_control(CTRL_ENABLE_MASK);
    assert!(state.is_enabled());
    assert!(!state.is_reset());
}

#[test]
fn control_reset_clears_state() {
    let mut state = CoreDebugState::new();
    // Set up some state first.
    state.enabled = true;
    state.reset = false;
    state.done = true;
    state.halted = true;
    state.pc = 0x1234;
    state.mem_stall = true;

    // Write reset.
    state.write_control(CTRL_RESET_MASK);
    assert!(state.is_reset());
    assert!(!state.is_enabled());
    assert!(!state.is_done());
    assert!(!state.is_halted());
    assert_eq!(state.read_pc(), 0);
    assert!(!state.mem_stall);
}

#[test]
fn control_enable_and_reset_both_set() {
    let mut state = CoreDebugState::new();
    state.write_control(CTRL_ENABLE_MASK | CTRL_RESET_MASK);
    // Both bits set -- reset takes priority on state clearing.
    assert!(state.is_reset());
    // Enable bit is stored but reset clears runtime state.
    assert!(state.is_enabled());
}

#[test]
fn control_roundtrip_read() {
    let mut state = CoreDebugState::new();
    state.write_control(CTRL_ENABLE_MASK);
    assert_eq!(state.read_control(), CTRL_ENABLE_MASK);

    state.write_control(CTRL_RESET_MASK);
    assert_eq!(state.read_control(), CTRL_RESET_MASK);

    state.write_control(0);
    assert_eq!(state.read_control(), 0);
}

// -- PC/SP/LR round-trip --

#[test]
fn pc_roundtrip() {
    let mut state = CoreDebugState::new();
    state.update_pc(0x1234);
    assert_eq!(state.read_pc(), 0x1234);
}

#[test]
fn sp_roundtrip() {
    let mut state = CoreDebugState::new();
    state.update_sp(0xABCD);
    assert_eq!(state.read_sp(), 0xABCD);
}

#[test]
fn lr_roundtrip() {
    let mut state = CoreDebugState::new();
    state.update_lr(0x5678);
    assert_eq!(state.read_lr(), 0x5678);
}

#[test]
fn pc_masked_to_20_bits() {
    let mut state = CoreDebugState::new();
    state.update_pc(0xFFFF_FFFF);
    assert_eq!(state.read_pc(), ADDR_MASK_20BIT);
}

#[test]
fn sp_masked_to_20_bits() {
    let mut state = CoreDebugState::new();
    state.update_sp(0xFFFF_FFFF);
    assert_eq!(state.read_sp(), ADDR_MASK_20BIT);
}

#[test]
fn lr_masked_to_20_bits() {
    let mut state = CoreDebugState::new();
    state.update_lr(0xFFFF_FFFF);
    assert_eq!(state.read_lr(), ADDR_MASK_20BIT);
}

// -- Halt / Resume --

#[test]
fn halt_when_enabled() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    assert!(state.request_halt());
    assert!(state.is_halted());
}

#[test]
fn halt_when_disabled_fails() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    assert!(!state.request_halt());
    assert!(!state.is_halted());
}

#[test]
fn halt_when_already_halted_fails() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.request_halt();
    assert!(!state.request_halt());
}

#[test]
fn resume_when_halted() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.request_halt();
    assert!(state.request_resume());
    assert!(!state.is_halted());
}

#[test]
fn resume_when_not_halted_fails() {
    let mut state = CoreDebugState::new();
    assert!(!state.request_resume());
}

// -- Stall updates --

#[test]
fn update_stalls_reflected_in_status() {
    let mut state = CoreDebugState::new();
    state.reset = false;

    state.update_stalls(true, false, false, false);
    assert_ne!(state.read_status() & (1 << STATUS_MEMORY_STALL_S_LSB), 0);
    assert_eq!(state.read_status() & (1 << STATUS_LOCK_STALL_S_LSB), 0);

    state.update_stalls(false, true, false, false);
    assert_eq!(state.read_status() & (1 << STATUS_MEMORY_STALL_S_LSB), 0);
    assert_ne!(state.read_status() & (1 << STATUS_LOCK_STALL_S_LSB), 0);

    state.update_stalls(false, false, true, false);
    assert_ne!(state.read_status() & (1 << STATUS_STREAM_STALL_SS0_LSB), 0);

    state.update_stalls(false, false, false, true);
    assert_ne!(state.read_status() & (1 << STATUS_CASCADE_STALL_SCD_LSB), 0);
}

#[test]
fn stalls_clear_when_updated_false() {
    let mut state = CoreDebugState::new();
    state.reset = false;

    state.update_stalls(true, true, true, true);
    assert_ne!(state.read_status() & (1 << STATUS_MEMORY_STALL_S_LSB), 0);
    assert_ne!(state.read_status() & (1 << STATUS_LOCK_STALL_S_LSB), 0);

    state.update_stalls(false, false, false, false);
    assert_eq!(state.read_status(), 0);
}

// -- Done bit --

#[test]
fn done_bit_behavior() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    assert!(!state.is_done());

    state.set_done(true);
    assert!(state.is_done());
    assert_ne!(state.read_status() & (1 << STATUS_DONE_LSB), 0);

    state.set_done(false);
    assert!(!state.is_done());
    assert_eq!(state.read_status() & (1 << STATUS_DONE_LSB), 0);
}

// -- Single-step mode --

#[test]
fn single_step_via_debug_control0() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;

    // Write single-step count = 3 (bits [5:2]).
    let value = 3u32 << DBG_CTRL0_SSTEP_COUNT_LSB;
    state.write_debug_control0(value);

    assert!(state.single_step);
    assert_eq!(state.single_step_count, 3);

    // Read back.
    let readback = state.read_debug_control0();
    assert_eq!((readback & DBG_CTRL0_SSTEP_COUNT_MASK) >> DBG_CTRL0_SSTEP_COUNT_LSB, 3);
}

#[test]
fn single_step_zero_count_disables() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.single_step = true;
    state.single_step_count = 5;

    state.write_debug_control0(0);
    assert!(!state.single_step);
    assert_eq!(state.single_step_count, 0);
}

// -- Debug_Control0 halt/resume via register write --

#[test]
fn debug_control0_halt_bit() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;

    state.write_debug_control0(DBG_CTRL0_HALT_MASK);
    assert!(state.is_halted());

    state.write_debug_control0(0);
    assert!(!state.is_halted());
}

// -- Debug_Status --

#[test]
fn debug_status_reflects_halt() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.request_halt();

    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_HALTED_LSB), 0);
}

#[test]
fn debug_status_clear_when_not_halted() {
    let state = CoreDebugState::new();
    assert_eq!(state.read_debug_status(), 0);
}

// -- Register interface --

#[test]
fn register_read_core_status() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    let val = state.read_register(REG_CORE_STATUS).unwrap();
    assert_ne!(val & (1 << STATUS_ENABLE_LSB), 0);
}

#[test]
fn register_read_core_control() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.reset = false;
    let val = state.read_register(REG_CORE_CONTROL).unwrap();
    assert_eq!(val, CTRL_ENABLE_MASK);
}

#[test]
fn register_read_pc() {
    let mut state = CoreDebugState::new();
    state.update_pc(0x4200);
    let val = state.read_register(REG_CORE_PC).unwrap();
    assert_eq!(val, 0x4200);
}

#[test]
fn register_read_sp() {
    let mut state = CoreDebugState::new();
    state.update_sp(0x1000);
    let val = state.read_register(REG_CORE_SP).unwrap();
    assert_eq!(val, 0x1000);
}

#[test]
fn register_read_lr() {
    let mut state = CoreDebugState::new();
    state.update_lr(0x2000);
    let val = state.read_register(REG_CORE_LR).unwrap();
    assert_eq!(val, 0x2000);
}

#[test]
fn register_read_debug_control0() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.halted = true;
    let val = state.read_register(REG_DEBUG_CONTROL0).unwrap();
    assert_ne!(val & DBG_CTRL0_HALT_MASK, 0);
}

#[test]
fn register_read_debug_control1() {
    let mut state = CoreDebugState::new();
    state.debug_ctrl1 = 0x1234_5678;
    let val = state.read_register(REG_DEBUG_CONTROL1).unwrap();
    assert_eq!(val, 0x1234_5678);
}

#[test]
fn register_read_debug_control2() {
    let mut state = CoreDebugState::new();
    state.debug_ctrl2 = 0x0000_000F;
    let val = state.read_register(REG_DEBUG_CONTROL2).unwrap();
    assert_eq!(val, 0x0000_000F);
}

#[test]
fn register_read_debug_status() {
    let state = CoreDebugState::new();
    let val = state.read_register(REG_DEBUG_STATUS).unwrap();
    assert_eq!(val, 0);
}

#[test]
fn register_read_unknown_returns_none() {
    let state = CoreDebugState::new();
    assert!(state.read_register(0xDEAD).is_none());
}

#[test]
fn register_write_core_control() {
    let mut state = CoreDebugState::new();
    assert!(state.write_register(REG_CORE_CONTROL, CTRL_ENABLE_MASK));
    assert!(state.is_enabled());
    assert!(!state.is_reset());
}

#[test]
fn register_write_debug_control0() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    assert!(state.write_register(REG_DEBUG_CONTROL0, DBG_CTRL0_HALT_MASK));
    assert!(state.is_halted());
}

#[test]
fn register_write_debug_control1() {
    let mut state = CoreDebugState::new();
    assert!(state.write_register(REG_DEBUG_CONTROL1, 0x7F7F_7F7F));
    assert_eq!(state.debug_ctrl1, 0x7F7F_7F7F);
}

#[test]
fn register_write_debug_control2() {
    let mut state = CoreDebugState::new();
    assert!(state.write_register(REG_DEBUG_CONTROL2, 0x0F));
    assert_eq!(state.debug_ctrl2, 0x0F);
}

#[test]
fn register_write_readonly_accepted_but_ignored() {
    let mut state = CoreDebugState::new();
    state.update_pc(0x100);
    // Writing to read-only registers should return true but not change state.
    assert!(state.write_register(REG_CORE_STATUS, 0xFFFF_FFFF));
    assert!(state.write_register(REG_CORE_PC, 0xFFFF));
    assert!(state.write_register(REG_CORE_SP, 0xFFFF));
    assert!(state.write_register(REG_CORE_LR, 0xFFFF));
    assert!(state.write_register(REG_DEBUG_STATUS, 0xFF));
    // PC should not have changed.
    assert_eq!(state.read_pc(), 0x100);
}

#[test]
fn register_write_unknown_returns_false() {
    let mut state = CoreDebugState::new();
    assert!(!state.write_register(0xDEAD, 0));
}

// -- Combined scenarios --

#[test]
fn full_lifecycle_enable_run_halt_resume_done() {
    let mut state = CoreDebugState::new();

    // Step 1: unreset and enable.
    state.write_control(CTRL_ENABLE_MASK);
    assert!(state.is_enabled());
    assert!(!state.is_reset());

    // Step 2: simulate execution -- update PC.
    state.update_pc(0x100);
    assert_eq!(state.read_pc(), 0x100);

    // Step 3: halt.
    assert!(state.request_halt());
    assert!(state.is_halted());
    assert_ne!(state.read_status() & (1 << STATUS_DEBUG_HALT_LSB), 0);

    // Step 4: resume.
    assert!(state.request_resume());
    assert!(!state.is_halted());

    // Step 5: done.
    state.set_done(true);
    assert!(state.is_done());
    assert_ne!(state.read_status() & (1 << STATUS_DONE_LSB), 0);
}

#[test]
fn reset_during_execution_clears_everything() {
    let mut state = CoreDebugState::new();

    // Set up mid-execution state.
    state.write_control(CTRL_ENABLE_MASK);
    state.update_pc(0x500);
    state.update_sp(0x1000);
    state.update_lr(0x200);
    state.update_stalls(true, false, true, false);
    state.request_halt();

    // Reset.
    state.write_control(CTRL_RESET_MASK);

    // Everything cleared.
    assert!(state.is_reset());
    assert!(!state.is_halted());
    assert!(!state.is_done());
    assert_eq!(state.read_pc(), 0);
    assert_eq!(state.read_sp(), 0);
    assert_eq!(state.read_lr(), 0);
    assert!(!state.mem_stall);
    assert!(!state.stream_stall);
}

#[test]
fn status_register_combined_bits() {
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.enabled = true;
    state.done = true;
    state.halted = true;
    state.mem_stall = true;
    state.lock_stall = true;

    let status = state.read_status();

    // Check all expected bits are set.
    assert_ne!(status & (1 << STATUS_ENABLE_LSB), 0);
    assert_ne!(status & (1 << STATUS_DONE_LSB), 0);
    assert_ne!(status & (1 << STATUS_DEBUG_HALT_LSB), 0);
    assert_ne!(status & (1 << STATUS_MEMORY_STALL_S_LSB), 0);
    assert_ne!(status & (1 << STATUS_LOCK_STALL_S_LSB), 0);

    // Check unexpected bits are clear.
    assert_eq!(status & (1 << STATUS_RESET_LSB), 0);
    assert_eq!(status & (1 << STATUS_STREAM_STALL_SS0_LSB), 0);
    assert_eq!(status & (1 << STATUS_CASCADE_STALL_SCD_LSB), 0);
}
