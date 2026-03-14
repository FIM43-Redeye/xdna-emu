//! Per-module tile timer emulation.
//!
//! Each AIE2 tile module (core, memory, memtile, shim) has a 64-bit free-running
//! timer that increments every clock cycle. The timer can be read, written, and
//! reset via memory-mapped registers. A trigger mechanism generates an event when
//! the timer reaches a programmed value.
//!
//! # Register Layout (from aie-rt xaiemlgbl_params.h)
//!
//! Each module's timer occupies a 256-byte register block with identical internal
//! layout. The base address varies by module type:
//!
//! | Module          | Base    | Source constant (aie-rt)                    |
//! |-----------------|---------|---------------------------------------------|
//! | Memory (compute)| 0x14000 | `XAIEMLGBL_MEMORY_MODULE_TIMER_CONTROL`    |
//! | Core (compute)  | 0x34000 | `XAIEMLGBL_CORE_MODULE_TIMER_CONTROL`      |
//! | MemTile         | 0x94000 | `XAIEMLGBL_MEM_TILE_MODULE_TIMER_CONTROL`  |
//! | Shim/PL         | 0x34000 | `XAIEMLGBL_PL_MODULE_TIMER_CONTROL`        |
//!
//! Within each block, the register offsets from the base are:
//!
//! | Register          | Offset | Description                                |
//! |-------------------|--------|--------------------------------------------|
//! | Timer_Control     | +0x000 | Reset bit [31], reset event [14:8] or [15:8] |
//! | Trig_Event_Low    | +0x0F0 | Trigger threshold low 32 bits              |
//! | Trig_Event_High   | +0x0F4 | Trigger threshold high 32 bits             |
//! | Timer_Low         | +0x0F8 | Current timer value low 32 bits (read/write) |
//! | Timer_High        | +0x0FC | Current timer value high 32 bits (read/write) |
//!
//! # Trigger Mechanism
//!
//! The timer can generate a "Timer_Value_Reached" event when the 64-bit timer
//! value matches the 64-bit trigger threshold (Trig_Event_Low | Trig_Event_High).
//! The default threshold is 0xFFFFFFFF_FFFFFFFF (maximum), which effectively
//! disables the trigger.
//!
//! # Reset Mechanism
//!
//! Writing 1 to the Reset bit (bit 31 of Timer_Control) resets the timer to 0.
//! The Reset_Event field configures which event will auto-reset the timer (used
//! by the timer sync protocol in aie-rt's `XAie_SyncTimer`).
//!
//! # Derived From
//!
//! - aie-rt: `driver/src/timer/xaie_timer.c` (MIT, Xilinx)
//! - aie-rt: `driver/src/global/xaiemlgbl_params.h` (register offsets and fields)
//! - aie-rt: `driver/src/global/xaiemlgbl_reginit.c` (TimerMod init structs)

/// Internal register offsets relative to the timer block base.
///
/// These are constant across all module types (core, memory, memtile, shim).
/// Derived from aie-rt xaiemlgbl_params.h: the difference between each
/// module's TIMER_CONTROL and TIMER_TRIG_EVENT_LOW_VALUE is always 0xF0,
/// and the four data registers are contiguous at +0xF0, +0xF4, +0xF8, +0xFC.
mod reg {
    /// Timer_Control register (reset bit, reset event selection).
    pub const CONTROL: u32 = 0x000;
    /// Trigger event threshold, low 32 bits of the 64-bit comparison value.
    pub const TRIG_EVENT_LOW: u32 = 0x0F0;
    /// Trigger event threshold, high 32 bits of the 64-bit comparison value.
    pub const TRIG_EVENT_HIGH: u32 = 0x0F4;
    /// Current timer value, low 32 bits (readable and writable).
    pub const TIMER_LOW: u32 = 0x0F8;
    /// Current timer value, high 32 bits (readable and writable).
    pub const TIMER_HIGH: u32 = 0x0FC;
}

/// Bit field definitions for the Timer_Control register.
///
/// Layout (from aie-rt xaiemlgbl_params.h):
/// ```text
/// [31]    Reset       -- write 1 to reset timer to 0
/// [14:8]  Reset_Event -- 7-bit event ID that triggers auto-reset (core/mem/PL)
/// [15:8]  Reset_Event -- 8-bit event ID for memtile (wider field)
/// [7:0]   Reserved
/// ```
mod control {
    /// Bit 31: writing 1 resets the timer value to 0.
    pub const RESET_BIT: u32 = 31;
    pub const RESET_MASK: u32 = 1 << RESET_BIT;

    /// Bits [14:8] (core/memory/PL) or [15:8] (memtile): event that triggers
    /// auto-reset. We use the wider 8-bit mask which is a superset.
    pub const RESET_EVENT_LSB: u32 = 8;
    pub const RESET_EVENT_MASK: u32 = 0x0000_FF00;
}

/// Per-module tile timer.
///
/// Emulates the hardware timer found in each AIE2 tile module. The timer is
/// a 64-bit counter that increments by 1 each clock cycle when the tile is
/// running. It provides:
///
/// - Read/write access to the current 64-bit value (split across two 32-bit
///   registers for hardware compatibility)
/// - A programmable 64-bit trigger threshold that generates a
///   "Timer_Value_Reached" event when the timer matches the threshold
/// - Reset functionality (immediate via register write, or deferred via event)
/// - A configurable reset event for the timer sync protocol
#[derive(Debug, Clone)]
pub struct TileTimer {
    /// Current 64-bit timer value. Incremented by `tick()` each cycle.
    value: u64,

    /// Trigger threshold low 32 bits. Default 0xFFFFFFFF (per aie-rt).
    trig_low: u32,

    /// Trigger threshold high 32 bits. Default 0xFFFFFFFF (per aie-rt).
    trig_high: u32,

    /// Timer_Control register value (reset bit + reset event field).
    /// The reset bit is write-only and self-clearing in hardware; we clear
    /// it after processing.
    control: u32,

    /// Whether the trigger event fired on the most recent tick. Downstream
    /// consumers (event broadcast, trace unit) can poll this.
    pub trigger_fired: bool,
}

impl TileTimer {
    /// Create a new timer with default state.
    ///
    /// Initial value is 0, trigger thresholds are 0xFFFFFFFF (disabled),
    /// matching the hardware reset defaults from aie-rt params.
    pub fn new() -> Self {
        Self {
            value: 0,
            trig_low: 0xFFFF_FFFF,
            trig_high: 0xFFFF_FFFF,
            control: 0,
            trigger_fired: false,
        }
    }

    /// Advance the timer by one clock cycle.
    ///
    /// Wraps around on 64-bit overflow (matching hardware behavior -- the timer
    /// is a free-running counter with no saturation). Checks the trigger
    /// threshold after incrementing and sets `trigger_fired` if the timer
    /// value matches.
    pub fn tick(&mut self) {
        self.value = self.value.wrapping_add(1);
        self.check_trigger();
    }

    /// Read the low 32 bits of the timer value.
    ///
    /// This is how aie-rt's `XAie_ReadTimer` reads the timer: low first,
    /// then high, combined into a 64-bit value.
    pub fn read_low(&self) -> u32 {
        self.value as u32
    }

    /// Read the high 32 bits of the timer value.
    pub fn read_high(&self) -> u32 {
        (self.value >> 32) as u32
    }

    /// Write the low 32 bits of the timer value.
    ///
    /// Preserves the high 32 bits. This is used by host software to set
    /// the timer to a specific value.
    pub fn write_low(&mut self, val: u32) {
        self.value = (self.value & 0xFFFF_FFFF_0000_0000) | val as u64;
    }

    /// Write the high 32 bits of the timer value.
    ///
    /// Preserves the low 32 bits.
    pub fn write_high(&mut self, val: u32) {
        self.value = (self.value & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
    }

    /// Get the full 64-bit timer value.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Reset the timer value to 0.
    ///
    /// This is the behavior triggered by writing 1 to bit 31 of Timer_Control,
    /// or by the reset event firing. Matches aie-rt `XAie_ResetTimer`.
    pub fn reset(&mut self) {
        self.value = 0;
        self.trigger_fired = false;
    }

    /// Get the currently configured reset event ID.
    ///
    /// Returns the 8-bit event ID from the Reset_Event field of Timer_Control.
    /// A value of 0 means no reset event is configured.
    pub fn reset_event(&self) -> u8 {
        ((self.control & control::RESET_EVENT_MASK) >> control::RESET_EVENT_LSB) as u8
    }

    /// Read a timer register by offset relative to the timer block base.
    ///
    /// Returns `Some(value)` if the offset maps to a valid timer register,
    /// `None` otherwise. This is the register-space interface used by the
    /// tile's MMIO dispatch.
    ///
    /// # Offsets
    ///
    /// | Offset | Register         |
    /// |--------|-----------------|
    /// | 0x000  | Timer_Control   |
    /// | 0x0F0  | Trig_Event_Low  |
    /// | 0x0F4  | Trig_Event_High |
    /// | 0x0F8  | Timer_Low       |
    /// | 0x0FC  | Timer_High      |
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        match offset {
            reg::CONTROL => Some(self.control),
            reg::TRIG_EVENT_LOW => Some(self.trig_low),
            reg::TRIG_EVENT_HIGH => Some(self.trig_high),
            reg::TIMER_LOW => Some(self.read_low()),
            reg::TIMER_HIGH => Some(self.read_high()),
            _ => None,
        }
    }

    /// Write a timer register by offset relative to the timer block base.
    ///
    /// Returns `true` if the offset mapped to a valid timer register and the
    /// write was handled, `false` otherwise.
    ///
    /// The Timer_Control reset bit (bit 31) is self-clearing: writing 1 resets
    /// the timer to 0 and the bit reads back as 0. This matches the hardware
    /// behavior described in aie-rt's `XAie_ResetTimer`, which writes the
    /// reset bit via `XAie_MaskWrite32` and expects it to auto-clear.
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        match offset {
            reg::CONTROL => {
                // Store the control register value (reset event field persists).
                // If the reset bit is set, reset the timer and clear the bit.
                self.control = value & !control::RESET_MASK;
                if value & control::RESET_MASK != 0 {
                    self.reset();
                }
                true
            }
            reg::TRIG_EVENT_LOW => {
                self.trig_low = value;
                true
            }
            reg::TRIG_EVENT_HIGH => {
                self.trig_high = value;
                true
            }
            reg::TIMER_LOW => {
                self.write_low(value);
                true
            }
            reg::TIMER_HIGH => {
                self.write_high(value);
                true
            }
            _ => false,
        }
    }

    /// Check whether the timer value matches the trigger threshold.
    ///
    /// Called internally after each tick. Sets `trigger_fired` to true on the
    /// exact cycle the timer reaches the threshold value. The trigger fires
    /// only on the transition (not continuously while equal), matching the
    /// hardware's edge-triggered behavior.
    fn check_trigger(&mut self) {
        let threshold = (self.trig_high as u64) << 32 | self.trig_low as u64;
        self.trigger_fired = self.value == threshold;
    }
}

impl Default for TileTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_value_is_zero() {
        let timer = TileTimer::new();
        assert_eq!(timer.value(), 0);
        assert_eq!(timer.read_low(), 0);
        assert_eq!(timer.read_high(), 0);
    }

    #[test]
    fn tick_increments_by_one() {
        let mut timer = TileTimer::new();
        timer.tick();
        assert_eq!(timer.value(), 1);
        timer.tick();
        assert_eq!(timer.value(), 2);
        timer.tick();
        assert_eq!(timer.value(), 3);
    }

    #[test]
    fn read_low_high_split() {
        let mut timer = TileTimer::new();

        // Set a value that spans both halves.
        // 0x0000_0001_0000_0000 = high=1, low=0
        timer.write_high(1);
        timer.write_low(0);
        assert_eq!(timer.value(), 0x0000_0001_0000_0000);
        assert_eq!(timer.read_low(), 0);
        assert_eq!(timer.read_high(), 1);

        // 0xDEAD_BEEF_CAFE_BABE
        timer.write_high(0xDEAD_BEEF);
        timer.write_low(0xCAFE_BABE);
        assert_eq!(timer.read_high(), 0xDEAD_BEEF);
        assert_eq!(timer.read_low(), 0xCAFE_BABE);
        assert_eq!(timer.value(), 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn write_low_preserves_high() {
        let mut timer = TileTimer::new();
        timer.write_high(0xAAAA_AAAA);
        timer.write_low(0x1111_1111);
        assert_eq!(timer.read_high(), 0xAAAA_AAAA);
        assert_eq!(timer.read_low(), 0x1111_1111);

        // Overwrite low, high should be preserved.
        timer.write_low(0x2222_2222);
        assert_eq!(timer.read_high(), 0xAAAA_AAAA);
        assert_eq!(timer.read_low(), 0x2222_2222);
    }

    #[test]
    fn write_high_preserves_low() {
        let mut timer = TileTimer::new();
        timer.write_low(0xBBBB_BBBB);
        timer.write_high(0x1111_1111);
        assert_eq!(timer.read_low(), 0xBBBB_BBBB);
        assert_eq!(timer.read_high(), 0x1111_1111);

        // Overwrite high, low should be preserved.
        timer.write_high(0x3333_3333);
        assert_eq!(timer.read_low(), 0xBBBB_BBBB);
        assert_eq!(timer.read_high(), 0x3333_3333);
    }

    #[test]
    fn overflow_wraps_to_zero() {
        let mut timer = TileTimer::new();
        // Set to u64::MAX - 1 so next two ticks go to MAX then wrap.
        timer.write_low(0xFFFF_FFFE);
        timer.write_high(0xFFFF_FFFF);
        assert_eq!(timer.value(), u64::MAX - 1);

        timer.tick();
        assert_eq!(timer.value(), u64::MAX);

        timer.tick();
        assert_eq!(timer.value(), 0, "64-bit overflow should wrap to 0");
    }

    #[test]
    fn low_word_overflow_carries_to_high() {
        let mut timer = TileTimer::new();
        timer.write_low(0xFFFF_FFFF);
        timer.write_high(0);
        assert_eq!(timer.value(), 0x0000_0000_FFFF_FFFF);

        timer.tick();
        assert_eq!(timer.read_low(), 0);
        assert_eq!(timer.read_high(), 1, "carry from low word to high word");
    }

    #[test]
    fn reset_clears_to_zero() {
        let mut timer = TileTimer::new();
        timer.write_low(0x1234_5678);
        timer.write_high(0x9ABC_DEF0);
        assert_ne!(timer.value(), 0);

        timer.reset();
        assert_eq!(timer.value(), 0);
        assert_eq!(timer.read_low(), 0);
        assert_eq!(timer.read_high(), 0);
    }

    #[test]
    fn register_read_control() {
        let mut timer = TileTimer::new();
        // Default control is 0.
        assert_eq!(timer.read_register(reg::CONTROL), Some(0));

        // Write a reset event.
        timer.write_register(reg::CONTROL, 0x0000_3F00);
        assert_eq!(timer.read_register(reg::CONTROL), Some(0x0000_3F00));
    }

    #[test]
    fn register_write_control_reset_bit() {
        let mut timer = TileTimer::new();
        // Set timer to a nonzero value.
        timer.write_low(42);
        assert_eq!(timer.value(), 42);

        // Write control with reset bit set (bit 31).
        timer.write_register(reg::CONTROL, 0x8000_0000);

        // Timer should be reset to 0.
        assert_eq!(timer.value(), 0);

        // Reset bit should self-clear (reads back as 0).
        assert_eq!(timer.read_register(reg::CONTROL), Some(0));
    }

    #[test]
    fn register_write_control_reset_with_event() {
        let mut timer = TileTimer::new();
        timer.write_low(100);

        // Write control with both reset bit and event ID 0x2A (bits [14:8]).
        // Event value = 0x2A << 8 = 0x2A00
        timer.write_register(reg::CONTROL, 0x8000_2A00);

        // Timer should be reset.
        assert_eq!(timer.value(), 0);

        // Reset bit should self-clear, but event field should persist.
        assert_eq!(timer.read_register(reg::CONTROL), Some(0x0000_2A00));
        assert_eq!(timer.reset_event(), 0x2A);
    }

    #[test]
    fn register_read_write_trig_low() {
        let mut timer = TileTimer::new();
        // Default trigger low is 0xFFFFFFFF per aie-rt.
        assert_eq!(timer.read_register(reg::TRIG_EVENT_LOW), Some(0xFFFF_FFFF));

        timer.write_register(reg::TRIG_EVENT_LOW, 0x0000_1000);
        assert_eq!(timer.read_register(reg::TRIG_EVENT_LOW), Some(0x0000_1000));
    }

    #[test]
    fn register_read_write_trig_high() {
        let mut timer = TileTimer::new();
        // Default trigger high is 0xFFFFFFFF per aie-rt.
        assert_eq!(timer.read_register(reg::TRIG_EVENT_HIGH), Some(0xFFFF_FFFF));

        timer.write_register(reg::TRIG_EVENT_HIGH, 0x0000_0042);
        assert_eq!(timer.read_register(reg::TRIG_EVENT_HIGH), Some(0x0000_0042));
    }

    #[test]
    fn register_read_write_timer_low() {
        let mut timer = TileTimer::new();
        assert_eq!(timer.read_register(reg::TIMER_LOW), Some(0));

        timer.write_register(reg::TIMER_LOW, 0xABCD_EF01);
        assert_eq!(timer.read_register(reg::TIMER_LOW), Some(0xABCD_EF01));
        assert_eq!(timer.read_low(), 0xABCD_EF01);
    }

    #[test]
    fn register_read_write_timer_high() {
        let mut timer = TileTimer::new();
        assert_eq!(timer.read_register(reg::TIMER_HIGH), Some(0));

        timer.write_register(reg::TIMER_HIGH, 0x1234_5678);
        assert_eq!(timer.read_register(reg::TIMER_HIGH), Some(0x1234_5678));
        assert_eq!(timer.read_high(), 0x1234_5678);
    }

    #[test]
    fn register_invalid_offset_returns_none() {
        let timer = TileTimer::new();
        assert_eq!(timer.read_register(0x004), None);
        assert_eq!(timer.read_register(0x100), None);
        assert_eq!(timer.read_register(0xFFF), None);
    }

    #[test]
    fn register_invalid_offset_write_returns_false() {
        let mut timer = TileTimer::new();
        assert!(!timer.write_register(0x004, 42));
        assert!(!timer.write_register(0x100, 42));
        assert!(!timer.write_register(0xFFF, 42));
    }

    #[test]
    fn register_round_trip_all() {
        let mut timer = TileTimer::new();

        // Write all writable registers via the register interface.
        assert!(timer.write_register(reg::CONTROL, 0x0000_1F00));
        assert!(timer.write_register(reg::TRIG_EVENT_LOW, 0xAAAA_AAAA));
        assert!(timer.write_register(reg::TRIG_EVENT_HIGH, 0xBBBB_BBBB));
        assert!(timer.write_register(reg::TIMER_LOW, 0xCCCC_CCCC));
        assert!(timer.write_register(reg::TIMER_HIGH, 0xDDDD_DDDD));

        // Read them all back.
        assert_eq!(timer.read_register(reg::CONTROL), Some(0x0000_1F00));
        assert_eq!(timer.read_register(reg::TRIG_EVENT_LOW), Some(0xAAAA_AAAA));
        assert_eq!(timer.read_register(reg::TRIG_EVENT_HIGH), Some(0xBBBB_BBBB));
        assert_eq!(timer.read_register(reg::TIMER_LOW), Some(0xCCCC_CCCC));
        assert_eq!(timer.read_register(reg::TIMER_HIGH), Some(0xDDDD_DDDD));
    }

    #[test]
    fn trigger_fires_at_threshold() {
        let mut timer = TileTimer::new();

        // Set trigger threshold to value 5.
        timer.write_register(reg::TRIG_EVENT_LOW, 5);
        timer.write_register(reg::TRIG_EVENT_HIGH, 0);

        // Tick 1-4: no trigger.
        for _ in 0..4 {
            timer.tick();
            assert!(!timer.trigger_fired, "should not fire before threshold");
        }

        // Tick 5: trigger fires.
        timer.tick();
        assert_eq!(timer.value(), 5);
        assert!(timer.trigger_fired, "should fire at threshold");

        // Tick 6: trigger no longer fires (edge-triggered, not level).
        timer.tick();
        assert_eq!(timer.value(), 6);
        assert!(!timer.trigger_fired, "should not fire after threshold");
    }

    #[test]
    fn trigger_disabled_by_default() {
        let mut timer = TileTimer::new();

        // Default thresholds are 0xFFFFFFFF_FFFFFFFF. The trigger fires only
        // when the timer reaches exactly u64::MAX after incrementing.
        // With default thresholds, the trigger should not fire for normal
        // values.
        for _ in 0..100 {
            timer.tick();
            assert!(!timer.trigger_fired);
        }
    }

    #[test]
    fn trigger_fires_at_u64_max_threshold() {
        let mut timer = TileTimer::new();

        // Default thresholds are 0xFFFFFFFF for both, meaning threshold = u64::MAX.
        // Set timer to u64::MAX - 1, then tick once to reach u64::MAX.
        timer.write_low(0xFFFF_FFFE);
        timer.write_high(0xFFFF_FFFF);

        timer.tick();
        assert_eq!(timer.value(), u64::MAX);
        assert!(timer.trigger_fired, "should fire at u64::MAX with default thresholds");
    }

    #[test]
    fn trigger_high_threshold() {
        let mut timer = TileTimer::new();

        // Set threshold to 0x0000_0001_0000_0003 (high=1, low=3).
        timer.write_register(reg::TRIG_EVENT_LOW, 3);
        timer.write_register(reg::TRIG_EVENT_HIGH, 1);

        // Set timer just below threshold.
        timer.write_low(0x0000_0002);
        timer.write_high(0x0000_0001);
        assert_eq!(timer.value(), 0x0000_0001_0000_0002);

        timer.tick();
        assert_eq!(timer.value(), 0x0000_0001_0000_0003);
        assert!(timer.trigger_fired, "should fire when both halves match");
    }

    #[test]
    fn multiple_independent_timers() {
        let mut t1 = TileTimer::new();
        let mut t2 = TileTimer::new();
        let mut t3 = TileTimer::new();

        // Tick t1 three times, t2 once, t3 not at all.
        t1.tick();
        t1.tick();
        t1.tick();
        t2.tick();

        assert_eq!(t1.value(), 3);
        assert_eq!(t2.value(), 1);
        assert_eq!(t3.value(), 0);

        // They should be completely independent.
        t3.write_low(999);
        assert_eq!(t1.value(), 3, "t1 unaffected by t3 write");
        assert_eq!(t2.value(), 1, "t2 unaffected by t3 write");
        assert_eq!(t3.value(), 999);
    }

    #[test]
    fn reset_event_field() {
        let mut timer = TileTimer::new();

        // No reset event by default.
        assert_eq!(timer.reset_event(), 0);

        // Set reset event to 0x3F (via register write, no reset bit).
        timer.write_register(reg::CONTROL, 0x0000_3F00);
        assert_eq!(timer.reset_event(), 0x3F);

        // Set reset event to 0x7F with reset bit -- timer resets but event
        // field persists.
        timer.write_low(42);
        timer.write_register(reg::CONTROL, 0x8000_7F00);
        assert_eq!(timer.value(), 0);
        assert_eq!(timer.reset_event(), 0x7F);
    }

    #[test]
    fn default_trait() {
        let timer = TileTimer::default();
        assert_eq!(timer.value(), 0);
        assert_eq!(timer.trig_low, 0xFFFF_FFFF);
        assert_eq!(timer.trig_high, 0xFFFF_FFFF);
    }

    #[test]
    fn tick_after_write_continues_from_written_value() {
        let mut timer = TileTimer::new();
        timer.write_low(100);
        timer.write_high(0);

        timer.tick();
        assert_eq!(timer.value(), 101);

        timer.tick();
        assert_eq!(timer.value(), 102);
    }

    #[test]
    fn reset_clears_trigger_fired() {
        let mut timer = TileTimer::new();
        timer.write_register(reg::TRIG_EVENT_LOW, 1);
        timer.write_register(reg::TRIG_EVENT_HIGH, 0);

        timer.tick();
        assert!(timer.trigger_fired);

        timer.reset();
        assert!(!timer.trigger_fired, "reset should clear trigger_fired");
    }

    #[test]
    fn control_reset_via_register_clears_trigger() {
        let mut timer = TileTimer::new();
        timer.write_register(reg::TRIG_EVENT_LOW, 1);
        timer.write_register(reg::TRIG_EVENT_HIGH, 0);

        timer.tick();
        assert!(timer.trigger_fired);

        // Reset via control register.
        timer.write_register(reg::CONTROL, control::RESET_MASK);
        assert!(!timer.trigger_fired, "control reset should clear trigger_fired");
        assert_eq!(timer.value(), 0);
    }

    #[test]
    fn trigger_threshold_update_mid_count() {
        let mut timer = TileTimer::new();

        // Tick to 10.
        for _ in 0..10 {
            timer.tick();
        }
        assert_eq!(timer.value(), 10);
        assert!(!timer.trigger_fired);

        // Set threshold to current value + 5 = 15.
        timer.write_register(reg::TRIG_EVENT_LOW, 15);
        timer.write_register(reg::TRIG_EVENT_HIGH, 0);

        // Tick to 14: no trigger.
        for _ in 0..4 {
            timer.tick();
            assert!(!timer.trigger_fired);
        }

        // Tick to 15: trigger fires.
        timer.tick();
        assert_eq!(timer.value(), 15);
        assert!(timer.trigger_fired);
    }
}
