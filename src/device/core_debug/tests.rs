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
fn status_packs_error_halt_without_ecc() {
    // set_error_halt flips the generic error bit (decode failure, missing
    // program memory, executor Error). Error_Halt fires; ECC_Error_Stall
    // stays clear because this is not an ECC-induced error.
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.set_error_halt(true);
    let status = state.read_status();
    assert_ne!(status & (1 << STATUS_ERROR_HALT_LSB), 0);
    assert_eq!(status & (1 << STATUS_ECC_ERROR_STALL_LSB), 0);
}

#[test]
fn write_control_reset_clears_error_halt() {
    // Reset bit clears error_halt alongside the other transient state.
    let mut state = CoreDebugState::new();
    state.reset = false;
    state.set_error_halt(true);
    assert_ne!(state.read_status() & (1 << STATUS_ERROR_HALT_LSB), 0);
    state.write_control(0x2); // bit 1 = reset
    assert_eq!(state.read_status() & (1 << STATUS_ERROR_HALT_LSB), 0);
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

// ----------------------------------------------------------------------
// Event-driven debug halt (Debug_Control1 / Debug_Control2 / Debug_Status)
// ----------------------------------------------------------------------

/// Build a Debug_Control1 value with the given event IDs in their fields.
fn make_dbg_ctrl1(resume: u8, sstep: u8, event0: u8, event1: u8) -> u32 {
    ((resume as u32) & 0x7F)
        | (((sstep as u32) & 0x7F) << 8)
        | (((event0 as u32) & 0x7F) << 16)
        | (((event1 as u32) & 0x7F) << 24)
}

#[test]
fn check_event_halt_no_config_does_not_halt() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.halted = false;
    // debug_ctrl1 left at 0 -- all event IDs are 0, the EVENT_NONE
    // sentinel. No event should ever halt.
    state.check_event_halt(16);
    state.check_event_halt(42);
    assert!(!state.is_halted(), "no halt event configured -> no halt");
}

#[test]
fn check_event_halt_event0_match_halts_and_latches_cause() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, 16, 0); // Event0 = 16 (WATCHPOINT_0)
    state.check_event_halt(16);
    assert!(state.is_halted(), "event 16 must trigger halt via Event0");
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_HALTED_LSB), 0, "Debug_Status[0] aggregate halt bit");
    assert_ne!(status & (1 << DBG_STS_EVENT0_HALTED_LSB), 0, "Debug_Status[5] Event0 cause latch");
    // Other cause bits must remain clear.
    assert_eq!(status & (1 << DBG_STS_EVENT1_HALTED_LSB), 0);
    assert_eq!(status & (1 << DBG_STS_MEM_STALL_HALTED_LSB), 0);
}

#[test]
fn check_event_halt_event1_match_halts_and_latches_cause() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, 0, 28); // Event1 = 28 (ACTIVE_CORE)
    state.check_event_halt(28);
    assert!(state.is_halted());
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_EVENT1_HALTED_LSB), 0, "Event1 cause latch");
    assert_eq!(status & (1 << DBG_STS_EVENT0_HALTED_LSB), 0, "Event0 must stay clear");
}

#[test]
fn check_event_halt_unrelated_event_does_not_halt() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, 16, 0);
    state.check_event_halt(17); // Adjacent event ID; must not match.
    assert!(!state.is_halted());
}

#[test]
fn check_event_halt_resume_event_clears_halt_and_causes() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(40, 0, 16, 0);
    state.check_event_halt(16); // Halt via Event0.
    assert!(state.is_halted());
    state.check_event_halt(40); // Resume via Resume_Event.
    assert!(!state.is_halted(), "resume event must clear halt");
    let status = state.read_debug_status();
    assert_eq!(status, 0, "all halt cause latches must be cleared on resume");
}

#[test]
fn check_event_halt_event_0_id_never_matches() {
    // Even if some caller passes event_id=0 (the EVENT_NONE sentinel),
    // it must never match a configured halt event -- otherwise a slot
    // with debug_ctrl1=0 would halt on every "no event" notification.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = 0; // halt event0 = 0, halt event1 = 0
    state.check_event_halt(0);
    assert!(!state.is_halted());
}

#[test]
fn stall_halt_mem_disabled_by_default_does_not_halt() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    // debug_ctrl2 = 0 -> all stall-halt enables off.
    state.update_stalls(true, false, false, false);
    assert!(!state.is_halted(), "mem stall alone must not halt without enable");
}

#[test]
fn stall_halt_mem_enabled_halts_on_mem_stall() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl2 = 1 << DBG_CTRL2_MEM_STALL_HALT_LSB;
    state.update_stalls(true, false, false, false);
    assert!(state.is_halted());
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_MEM_STALL_HALTED_LSB), 0);
}

#[test]
fn stall_halt_lock_enabled_halts_on_lock_stall() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl2 = 1 << DBG_CTRL2_LOCK_STALL_HALT_LSB;
    state.update_stalls(false, true, false, false);
    assert!(state.is_halted());
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_LOCK_STALL_HALTED_LSB), 0);
}

#[test]
fn stall_halt_stream_enabled_halts_on_stream_stall() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl2 = 1 << DBG_CTRL2_STREAM_STALL_HALT_LSB;
    state.update_stalls(false, false, true, false);
    assert!(state.is_halted());
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_STREAM_STALL_HALTED_LSB), 0);
}

#[test]
fn stall_halt_unenabled_categories_do_not_halt() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl2 = 1 << DBG_CTRL2_MEM_STALL_HALT_LSB; // mem only
                                                           // Lock and stream stalls are active but their enables are off.
    state.update_stalls(false, true, true, false);
    assert!(!state.is_halted(), "only enabled categories may trigger halt");
}

#[test]
fn debug_status_aggregates_multiple_causes() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl2 = 1 << DBG_CTRL2_MEM_STALL_HALT_LSB;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, 16, 0);

    // Trigger via mem stall first.
    state.update_stalls(true, false, false, false);
    // Then via Event0 -- request_halt is idempotent but the cause
    // latch should also set independently.
    state.check_event_halt(16);

    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_MEM_STALL_HALTED_LSB), 0);
    assert_ne!(status & (1 << DBG_STS_EVENT0_HALTED_LSB), 0);
}

// ----------------------------------------------------------------------
// PC_Event0..3 + Debug_Control2.PC_Event_Halt
// ----------------------------------------------------------------------

/// Build a PC_Event* register raw value with VALID and a 14-bit address.
fn make_pc_event(valid: bool, address: u32) -> u32 {
    let mut v = address & PC_EVENT_ADDRESS_MASK;
    if valid {
        v |= PC_EVENT_VALID_MASK;
    }
    v
}

#[test]
fn pc_event_registers_default_zero() {
    let state = CoreDebugState::new();
    assert_eq!(state.read_register(REG_PC_EVENT0), Some(0));
    assert_eq!(state.read_register(REG_PC_EVENT1), Some(0));
    assert_eq!(state.read_register(REG_PC_EVENT2), Some(0));
    assert_eq!(state.read_register(REG_PC_EVENT3), Some(0));
}

#[test]
fn pc_event_register_round_trips_writeable_bits() {
    let mut state = CoreDebugState::new();
    // Set bits outside the writable mask -- they must be dropped.
    assert!(state.write_register(REG_PC_EVENT0, 0xFFFF_FFFF));
    let read = state.read_register(REG_PC_EVENT0).unwrap();
    assert_eq!(read, PC_EVENT_WRITE_MASK, "non-writable bits must be masked off");
    // VALID stays set.
    assert_ne!(read & PC_EVENT_VALID_MASK, 0);
    // 14-bit address.
    assert_eq!(read & PC_EVENT_ADDRESS_MASK, PC_EVENT_ADDRESS_MASK);
}

#[test]
fn pc_event_match_broadcasts_core_pc_event() {
    // PC_Event0 valid + matching PC, with HaltEvent0 wired to Core_PC_0
    // (event 16): the match must trigger halt via Event0 cause bit.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_0, 0);

    state.update_pc(0x100);

    assert!(state.is_halted(), "PC match must broadcast Core_PC_0 -> HaltEvent0");
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_EVENT0_HALTED_LSB), 0, "Event0 cause latched");
    // PC_Event_Halt gate disabled -> halt_cause_pc_event must NOT be latched.
    assert_eq!(
        status & (1 << DBG_STS_PC_EVENT_HALTED_LSB),
        0,
        "PC_Event cause should only latch when Debug_Control2.PC_Event_Halt is set"
    );
}

#[test]
fn pc_event_invalid_does_not_fire() {
    // VALID=0 with matching address -> no event.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(false, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_0, 0);
    state.update_pc(0x100);
    assert!(!state.is_halted(), "VALID=0 must inhibit PC match");
}

#[test]
fn pc_event_address_mismatch_does_not_fire() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_0, 0);
    state.update_pc(0x104);
    assert!(!state.is_halted(), "address mismatch must not fire");
}

#[test]
fn pc_event_halt_gate_latches_pc_event_cause() {
    // Debug_Control2.PC_Event_Halt + valid PC_Event match: halts and
    // latches halt_cause_pc_event independently of Debug_Control1.HaltEvent*.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x200);
    state.debug_ctrl2 = 1 << DBG_CTRL2_PC_EVENT_HALT_LSB;
    // No HaltEvent wiring -- the gate alone must halt.
    state.update_pc(0x200);
    assert!(state.is_halted(), "PC_Event_Halt gate must halt on any PC event");
    let status = state.read_debug_status();
    assert_ne!(
        status & (1 << DBG_STS_PC_EVENT_HALTED_LSB),
        0,
        "halt_cause_pc_event must latch under PC_Event_Halt gate"
    );
}

#[test]
fn pc_event_halt_gate_disabled_no_pc_cause_latch() {
    // Without the gate, even a fired PC event does not latch the
    // PC_Event halt cause (only the Event0/1 cause via HaltEvent path).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event1 = make_pc_event(true, 0x300);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, 0, EVENT_CORE_PC_1);
    state.update_pc(0x300);
    assert!(state.is_halted());
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_EVENT1_HALTED_LSB), 0);
    assert_eq!(status & (1 << DBG_STS_PC_EVENT_HALTED_LSB), 0);
}

#[test]
fn pc_range_0_1_fires_when_in_range() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.pc_event1 = make_pc_event(true, 0x200);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_RANGE_0_1, 0);
    state.update_pc(0x150);
    assert!(state.is_halted(), "Core_PC_Range_0_1 must fire when PC is in [PC0, PC1]");
}

#[test]
fn pc_range_0_1_outside_does_not_fire() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.pc_event1 = make_pc_event(true, 0x200);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_RANGE_0_1, 0);
    state.update_pc(0x300);
    assert!(!state.is_halted(), "PC outside range must not fire");
}

#[test]
fn pc_range_requires_both_endpoints_valid() {
    // PC_Event0 valid, PC_Event1 invalid -> no range event.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.pc_event1 = make_pc_event(false, 0x200);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_RANGE_0_1, 0);
    state.update_pc(0x150);
    assert!(!state.is_halted(), "range needs both endpoints VALID");
}

#[test]
fn pc_range_swapped_endpoints_still_defines_range() {
    // PC0 > PC1: still treated as a range (lo=PC1, hi=PC0).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x200);
    state.pc_event1 = make_pc_event(true, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_RANGE_0_1, 0);
    state.update_pc(0x150);
    assert!(state.is_halted(), "range works regardless of endpoint order");
}

#[test]
fn pc_range_2_3_fires_when_in_range() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event2 = make_pc_event(true, 0x400);
    state.pc_event3 = make_pc_event(true, 0x500);
    state.debug_ctrl1 = make_dbg_ctrl1(0, 0, EVENT_CORE_PC_RANGE_2_3, 0);
    state.update_pc(0x480);
    assert!(state.is_halted(), "Core_PC_Range_2_3 must fire when PC is in [PC2, PC3]");
}

#[test]
fn pc_event_resume_clears_pc_event_cause() {
    // Halt via PC_Event_Halt gate, then send a Resume event.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.debug_ctrl2 = 1 << DBG_CTRL2_PC_EVENT_HALT_LSB;
    // Resume on a non-PC event so we can drive resume directly.
    state.debug_ctrl1 = make_dbg_ctrl1(40, 0, 0, 0);
    state.update_pc(0x100);
    assert!(state.is_halted());
    state.check_event_halt(40); // resume
    assert!(!state.is_halted(), "resume event must clear halt");
    let status = state.read_debug_status();
    assert_eq!(status, 0, "all halt-cause latches must clear on resume");
}

#[test]
fn reset_clears_pc_event_registers() {
    let mut state = CoreDebugState::new();
    state.write_register(REG_PC_EVENT0, make_pc_event(true, 0x100));
    state.write_register(REG_PC_EVENT3, make_pc_event(true, 0x3FFF));
    // Assert reset.
    state.write_control(CTRL_RESET_MASK);
    assert_eq!(state.read_register(REG_PC_EVENT0), Some(0));
    assert_eq!(state.read_register(REG_PC_EVENT3), Some(0));
}

#[test]
fn pc_event_match_after_reset_does_not_fire() {
    // Configuration written, then reset, then PC update at the previously
    // matched address: must not fire because PC_Event0 was cleared.
    let mut state = CoreDebugState::new();
    state.write_register(REG_PC_EVENT0, make_pc_event(true, 0x100));
    state.write_register(REG_DEBUG_CONTROL1, make_dbg_ctrl1(0, 0, EVENT_CORE_PC_0, 0));
    state.write_control(CTRL_RESET_MASK);
    state.write_control(CTRL_ENABLE_MASK);
    state.update_pc(0x100);
    assert!(!state.is_halted(), "PC_Event registers must be cleared by reset");
}

// ----------------------------------------------------------------------
// Single-step-on-event (Debug_Control1.SSTEP_EVENT)
// ----------------------------------------------------------------------

#[test]
fn sstep_event_match_sets_pending_latch() {
    // §5.1 principled split: this is the documented AFTER-commit case.
    // check_event_halt is called directly with no PC_Event slot wired, so
    // the before-commit seam (has_sync_sstep_pc_trap_at) does not engage;
    // the SSTEP_EVENT latch arms and the halt is deferred to consume (the
    // triggering bundle commits first). Watchpoint/mem/lock/range-wired
    // single-step stays on this path by design (no coherent before-commit
    // point -- arming known only post-bundle).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 16, 0, 0);
    state.check_event_halt(16);
    assert!(state.pending_single_step, "matching event must arm the SSTEP latch");
    assert!(!state.is_halted(), "halt is deferred to consume, so the triggering bundle commits first");
}

#[test]
fn sstep_event_consume_halts_and_clears_latch() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 16, 0, 0);
    state.check_event_halt(16);
    let drained = state.consume_pending_single_step();
    assert!(drained, "consume must report the event was queued");
    assert!(state.is_halted(), "consume must halt the core");
    assert!(!state.pending_single_step, "latch must clear after consume");
}

#[test]
fn sstep_event_consume_when_idle_returns_false() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    let drained = state.consume_pending_single_step();
    assert!(!drained, "no event queued -> consume reports false");
    assert!(!state.is_halted());
}

#[test]
fn sstep_event_unrelated_event_does_not_arm() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 16, 0, 0);
    state.check_event_halt(17); // adjacent event ID, must not match
    assert!(!state.pending_single_step);
}

#[test]
fn sstep_event_zero_id_never_matches() {
    // SSTEP_EVENT field = 0 (EVENT_NONE) must not arm on event_id=0.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = 0;
    // check_event_halt short-circuits on event_id=0, but defense-in-depth:
    // even if the field were 0 and a non-zero event fired, no match occurs.
    state.check_event_halt(0);
    assert!(!state.pending_single_step);
    state.check_event_halt(42);
    assert!(!state.pending_single_step);
}

#[test]
fn sstep_event_via_pc_event_path() {
    // §5.1 principled split (Maya 2026-05-19): SSTEP_EVENT wired to a
    // point PC event (Core_PC_0) is the before-commit-eligible case --
    // the arming condition (PC match) is known before the bundle, so the
    // pre-execute seam halts BEFORE the bundle commits, parallel to G1's
    // PC_Event_Halt seam (no deferred pending_single_step round-trip).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_0, 0, 0);

    assert!(state.has_sync_sstep_pc_trap_at(0x100));
    assert!(!state.is_halted(), "query alone does not halt");
    state.consume_sync_sstep_pc_trap(0x100);
    assert!(state.is_halted(), "PC-wired single-step halts before-commit");

    // Idempotent: consumed at this PC, does not re-fire after resume
    // (mirrors consume_sync_pc_trap; re-arming is the §8-tracked edge).
    state.request_resume();
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));
}

#[test]
fn sstep_pc_trap_only_for_point_pc_events() {
    // Watchpoint / non-PC SSTEP_EVENT is NOT before-commit eligible: no
    // PC_Event slot mapping -> query is false (it stays on the unchanged
    // after-commit check_event_halt -> pending_single_step path).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    // SSTEP_EVENT = 32 (a non-PC event id, e.g. a watchpoint).
    state.debug_ctrl1 = make_dbg_ctrl1(0, 32, 0, 0);
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));

    // PC-range single-step is deliberately bucketed after-commit too.
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_RANGE_0_1, 0, 0);
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));
}

#[test]
fn sstep_pc_trap_requires_valid_matching_pc_event() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_2, 0, 0);
    // PC_Event2 not VALID -> no trap.
    state.pc_event2 = make_pc_event(false, 0x200);
    assert!(!state.has_sync_sstep_pc_trap_at(0x200));
    // VALID but PC mismatch -> no trap.
    state.pc_event2 = make_pc_event(true, 0x200);
    assert!(!state.has_sync_sstep_pc_trap_at(0x208));
    // VALID and matching -> trap.
    assert!(state.has_sync_sstep_pc_trap_at(0x200));
}

#[test]
fn sstep_event_resume_cancels_pending() {
    // If a resume event arrives between the SSTEP arming and the
    // coordinator's consume call, the pending latch must clear so the
    // single-step doesn't fire after the host has unhalted.
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(40, 16, 0, 0); // resume=40, sstep=16
    state.check_event_halt(16);
    assert!(state.pending_single_step);
    state.check_event_halt(40); // resume
    assert!(!state.pending_single_step, "resume must cancel a pending single-step");
    let drained = state.consume_pending_single_step();
    assert!(!drained, "consume after resume sees nothing");
    assert!(!state.is_halted());
}

#[test]
fn sstep_event_reset_clears_pending() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 16, 0, 0);
    state.check_event_halt(16);
    assert!(state.pending_single_step);
    state.write_control(CTRL_RESET_MASK);
    assert!(!state.pending_single_step, "reset must clear pending single-step");
}

#[test]
fn sstep_event_coexists_with_halt_event() {
    // SSTEP_EVENT and HaltEvent0 wired to the same event ID. Both fire on
    // a match: the halt is immediate (HaltEvent path), and pending is
    // also set (idempotent -- consume drains it but request_halt is
    // already true).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, 16, 16, 0); // sstep=halt=16
    state.check_event_halt(16);
    assert!(state.is_halted(), "HaltEvent0 must halt immediately");
    assert!(state.pending_single_step, "SSTEP latch also set -- separate paths");
    let status = state.read_debug_status();
    assert_ne!(status & (1 << DBG_STS_EVENT0_HALTED_LSB), 0);
    state.consume_pending_single_step();
    assert!(state.is_halted(), "still halted");
}

// -- §5.2 Count-step state machine --

#[test]
fn count_step_arm_from_debug_control0() {
    // Debug_Control0[5:2]=N (halt bit clear) arms a live N-budget.
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x10); // N=4 (4<<2), halt bit [0]=0
    assert_eq!(s.count_step_remaining, Some(4));
    assert!(!s.is_halted(), "arming alone does not halt");
}

#[test]
fn count_step_zero_disables() {
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x10);
    s.write_debug_control0(0x00); // N=0 -> disabled
    assert_eq!(s.count_step_remaining, None);
    assert!(!s.tick_count_step(), "disabled budget never halts");
    assert!(!s.is_halted());
}

#[test]
fn count_step_decrements_then_halts_before_n_plus_1() {
    // N=2: bundles 1 and 2 commit (tick after each), halt latched on the
    // 2nd tick so the existing is_halted gate blocks bundle 3.
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x08); // N=2 (2<<2)
    assert!(!s.tick_count_step(), "after bundle 1: budget 2->1, no halt");
    assert_eq!(s.count_step_remaining, Some(1));
    assert!(s.tick_count_step(), "after bundle 2: budget expires, halt");
    assert!(s.is_halted());
    assert!(s.halt_cause_count_step);
    assert_eq!(s.count_step_remaining, None, "expiry clears the budget");
}

#[test]
fn count_step_expiry_clears_no_rearm_on_resume() {
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x04); // N=1
    assert!(s.tick_count_step(), "N=1 halts on the first tick");
    assert_eq!(s.count_step_remaining, None);
    s.request_resume(); // resume must NOT re-arm the budget
    assert_eq!(s.count_step_remaining, None);
    assert!(!s.tick_count_step(), "post-expiry ticks are no-ops until a fresh write");
}

#[test]
fn count_step_halt_bit_precedence_with_latent_budget() {
    // 0x11 = halt bit [0]=1 AND Single_Step_Count=4. Bit[0]'s immediate
    // halt takes precedence; the N-budget is still armed (latent).
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x11);
    assert!(s.is_halted(), "halt bit [0] halts immediately (precedence)");
    assert_eq!(s.count_step_remaining, Some(4), "budget armed latent");
}

#[test]
fn count_step_expiry_cause_cleared_on_resume() {
    // Count-step expiry latches halt_cause_count_step; an event-resume
    // must clear it in lockstep with the other halt_cause_* latches
    // (clear_halt_causes symmetry). The budget rule is independent:
    // resume does NOT re-arm -- count_step_remaining stays None.
    let mut s = CoreDebugState::new();
    s.enabled = true;
    // Resume on event id 5 (arbitrary non-zero, not colliding with sstep/halt).
    s.debug_ctrl1 = make_dbg_ctrl1(5, 0, 0, 0);
    s.write_debug_control0(0x04); // N=1
    assert!(s.tick_count_step(), "N=1 expires on first tick");
    assert!(s.halt_cause_count_step, "expiry latches the cause");
    assert!(s.is_halted());
    assert_eq!(s.count_step_remaining, None, "expiry cleared the budget");

    s.check_event_halt(5); // Debug_Resume_Core_Event fires
    assert!(!s.halt_cause_count_step, "resume clears the count-step cause latch");
    assert!(!s.is_halted(), "resume unhalts");
    assert_eq!(s.count_step_remaining, None, "resume does NOT re-arm the budget");
}
