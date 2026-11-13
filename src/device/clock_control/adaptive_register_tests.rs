//! Adaptive clock-gate controls: the enable bits of Module_Clock_Control and the
//! stream-switch Abort_Period register.
//!
//! These behaviours come from the AM025 register database and have not been
//! checked on silicon.

use super::*;

const MCC_COMPUTE: u32 = 0x60000;
const MCC_MEMTILE: u32 = 0xFFF00;
const ABORT_COMPUTE: u32 = 0x3FF38;
const ABORT_MEMTILE: u32 = 0xB0F38;

const SS_ADAPTIVE: u32 = 1 << 4;
const DMA_ADAPTIVE: u32 = 1 << 5;

fn ungated() -> ClockController {
    let mut clock = ClockController::new(5, 6);
    clock.write_register(2, 0, COLUMN_CLOCK_CONTROL_OFFSET, 1);
    clock
}

fn idle_ss(clock: &mut ClockController, col: u8, row: u8, cycles: u32) {
    for _ in 0..cycles {
        clock.tick_adaptive_ss(col, row, false);
    }
}

fn idle_dma(clock: &mut ClockController, col: u8, row: u8, cycles: u32) {
    for _ in 0..cycles {
        clock.tick_adaptive_dma(col, row, false);
    }
}

#[test]
fn defaults_engage_both_gates_at_128_idle_cycles() {
    let mut clock = ungated();
    idle_ss(&mut clock, 2, 3, 127);
    idle_dma(&mut clock, 2, 3, 127);
    assert!(!clock.is_adaptive_ss_engaged(2, 3));
    assert!(!clock.is_adaptive_dma_engaged(2, 3));
    idle_ss(&mut clock, 2, 3, 1);
    idle_dma(&mut clock, 2, 3, 1);
    assert!(clock.is_adaptive_ss_engaged(2, 3));
    assert!(clock.is_adaptive_dma_engaged(2, 3));
}

#[test]
fn enables_read_as_set_before_any_write() {
    let clock = ungated();
    for (row, _) in [(3, MCC_COMPUTE), (1, MCC_MEMTILE), (0, MCC_MEMTILE)] {
        assert!(clock.is_adaptive_ss_enabled(2, row), "row {row}");
        assert!(clock.is_adaptive_dma_enabled(2, row), "row {row}");
    }
}

#[test]
fn clearing_bit4_stops_the_stream_switch_gate_only() {
    for (row, mcc, reset) in [(3, MCC_COMPUTE, 0x37), (1, MCC_MEMTILE, 0x33), (0, MCC_MEMTILE, 0x3B)] {
        let mut clock = ungated();
        clock.write_register(2, row, mcc, reset & !SS_ADAPTIVE);
        idle_ss(&mut clock, 2, row, 100_000);
        idle_dma(&mut clock, 2, row, 100_000);
        assert!(!clock.is_adaptive_ss_engaged(2, row), "row {row}: switch gate must never engage");
        assert!(clock.is_adaptive_dma_engaged(2, row), "row {row}: DMA gate unaffected");
    }
}

#[test]
fn clearing_bit5_stops_the_dma_gate_only() {
    for (row, mcc, reset) in [(3, MCC_COMPUTE, 0x37), (1, MCC_MEMTILE, 0x33), (0, MCC_MEMTILE, 0x3B)] {
        let mut clock = ungated();
        clock.write_register(2, row, mcc, reset & !DMA_ADAPTIVE);
        idle_ss(&mut clock, 2, row, 100_000);
        idle_dma(&mut clock, 2, row, 100_000);
        assert!(clock.is_adaptive_ss_engaged(2, row), "row {row}: switch gate unaffected");
        assert!(!clock.is_adaptive_dma_engaged(2, row), "row {row}: DMA gate must never engage");
    }
}

#[test]
fn setting_the_enable_bit_again_re_arms_the_gate() {
    let mut clock = ungated();
    clock.write_register(2, 3, MCC_COMPUTE, 0x37 & !SS_ADAPTIVE);
    idle_ss(&mut clock, 2, 3, 500);
    assert!(!clock.is_adaptive_ss_engaged(2, 3));
    clock.write_register(2, 3, MCC_COMPUTE, 0x37);
    // The idle counter ran on while the gate was disabled, so it engages at once.
    assert!(clock.is_adaptive_ss_engaged(2, 3));
}

#[test]
fn abort_period_sets_the_stream_switch_threshold_to_two_to_the_n() {
    for (row, offset) in [(3, ABORT_COMPUTE), (1, ABORT_MEMTILE), (0, ABORT_COMPUTE)] {
        for n in [3u32, 4, 7, 12] {
            let mut clock = ungated();
            clock.write_register(2, row, offset, n);
            idle_ss(&mut clock, 2, row, (1 << n) - 1);
            assert!(!clock.is_adaptive_ss_engaged(2, row), "row {row} N={n}: one cycle short");
            idle_ss(&mut clock, 2, row, 1);
            assert!(clock.is_adaptive_ss_engaged(2, row), "row {row} N={n}: at 2^N");
        }
    }
}

#[test]
fn abort_period_has_no_effect_on_the_dma_gate() {
    // The database defines an Abort_Period for the stream-switch gate only.
    let mut clock = ungated();
    clock.write_register(2, 3, ABORT_COMPUTE, 3);
    idle_dma(&mut clock, 2, 3, 8);
    assert!(!clock.is_adaptive_dma_engaged(2, 3));
    idle_dma(&mut clock, 2, 3, 120);
    assert!(clock.is_adaptive_dma_engaged(2, 3));
}

#[test]
fn abort_period_uses_only_the_low_four_bits() {
    let mut clock = ungated();
    clock.write_register(2, 3, ABORT_COMPUTE, 0xFFF3);
    idle_ss(&mut clock, 2, 3, 8);
    assert!(clock.is_adaptive_ss_engaged(2, 3), "field [3:0] = 3");
}

#[test]
fn abort_period_outside_the_documented_range_is_applied_as_written() {
    // Documented range is 3-12; the value is still 2^N for N in 0..=15.
    let mut clock = ungated();
    clock.write_register(2, 3, ABORT_COMPUTE, 0);
    idle_ss(&mut clock, 2, 3, 1);
    assert!(clock.is_adaptive_ss_engaged(2, 3), "N=0: threshold 1");

    let mut clock = ungated();
    clock.write_register(2, 3, ABORT_COMPUTE, 15);
    idle_ss(&mut clock, 2, 3, 32_767);
    assert!(!clock.is_adaptive_ss_engaged(2, 3));
    idle_ss(&mut clock, 2, 3, 1);
    assert!(clock.is_adaptive_ss_engaged(2, 3), "N=15: threshold 32768");
}

#[test]
fn abort_period_and_enable_registers_read_back() {
    let mut clock = ungated();
    assert_eq!(clock.read_register(2, 3, ABORT_COMPUTE), Some(7));
    assert_eq!(clock.read_register(2, 1, ABORT_MEMTILE), Some(7));
    clock.write_register(2, 3, ABORT_COMPUTE, 5);
    assert_eq!(clock.read_register(2, 3, ABORT_COMPUTE), Some(5));
}

#[test]
fn masked_write_helper_leaves_other_module_bits_alone() {
    // Clearing only bit 4 keeps Core, Memory and Stream_Switch clocks running.
    let mut clock = ungated();
    clock.write_register(2, 3, MCC_COMPUTE, 0x37 & !SS_ADAPTIVE);
    assert!(clock.is_module_active(2, 3, ModuleKind::Core));
    assert!(clock.is_module_active(2, 3, ModuleKind::Memory));
    assert!(clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
}
