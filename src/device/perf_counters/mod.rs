//! Performance counter runtime logic for AMD AIE2 NPU tiles.
//!
//! Each AIE2 tile module (core, memory, memtile, shim) has hardware performance
//! counters that count cycles while active. The counters are controlled by
//! start/stop/reset events and can fire threshold events when the count reaches
//! a configured event value.
//!
//! # Hardware Behavior (derived from aie-rt `xaie_perfcnt.c`)
//!
//! - **Start event**: When the configured start event fires, the counter
//!   begins incrementing by 1 each cycle.
//! - **Stop event**: When the configured stop event fires, the counter
//!   stops incrementing (value is preserved).
//! - **Reset event**: When the configured reset event fires, the counter
//!   value is zeroed.
//! - **Event value (threshold)**: When the counter reaches this value, a
//!   `PERF_CNT_N` event is generated (hardware event ID = base + counter_index).
//! - **Direct write**: The host can write a counter value directly via
//!   register write (e.g., to preset or clear it).
//!
//! # Per-Module Counter Counts (from aie-rt `xaiemlgbl_reginit.c`)
//!
//! | Module      | Counter Count | Event Width |
//! |-------------|---------------|-------------|
//! | Core        | 4             | 7 bits      |
//! | Memory      | 2             | 7 bits      |
//! | MemTile     | 4             | 8 bits      |
//! | Shim (PL)   | 2             | 7 bits      |
//!
//! # Register Layout (from aie-rt `xaiemlgbl_params.h`)
//!
//! Each module's perf counter block has a consistent relative layout:
//!
//! | Offset   | Register                   | Content                              |
//! |----------|----------------------------|--------------------------------------|
//! | +0x00    | Performance_Control0       | cnt0/cnt1 start/stop events          |
//! | +0x04    | Performance_Control1       | cnt2/cnt3 start/stop (4-counter) or reset (2-counter) |
//! | +0x08    | Performance_Control2       | cnt0-3 reset events (4-counter)      |
//! | +0x20    | Performance_Counter0       | Counter 0 value (R/W)                |
//! | +0x24    | Performance_Counter1       | Counter 1 value (R/W)                |
//! | +0x28    | Performance_Counter2       | Counter 2 value (4-counter only)     |
//! | +0x2C    | Performance_Counter3       | Counter 3 value (4-counter only)     |
//! | +0x80    | Performance_Counter0_Event_Value | Threshold for counter 0       |
//! | +0x84    | Performance_Counter1_Event_Value | Threshold for counter 1       |
//! | +0x88    | Performance_Counter2_Event_Value | Threshold (4-counter only)    |
//! | +0x8C    | Performance_Counter3_Event_Value | Threshold (4-counter only)    |
//!
//! Absolute offsets per module (from aie-rt `xaiemlgbl_params.h`):
//!
//! | Module      | Control0    | Counter0    | EventValue0   |
//! |-------------|-------------|-------------|---------------|
//! | Core        | 0x0003_1500 | 0x0003_1520 | 0x0003_1580   |
//! | Memory      | 0x0001_1000 | 0x0001_1020 | 0x0001_1080   |
//! | MemTile     | 0x0009_1000 | 0x0009_1020 | 0x0009_1080   |
//! | Shim (PL)   | 0x0003_1000 | 0x0003_1020 | 0x0003_1080   |
//!
//! # Control Register Bit Fields
//!
//! Start/stop control packs two counters per register with a 16-bit shift
//! (per aie-rt `StartStopShift = 16`):
//! - bits [6:0] / [7:0]: counter_lo start event
//! - bits [14:8] / [15:8]: counter_lo stop event
//! - bits [22:16] / [23:16]: counter_hi start event
//! - bits [30:24] / [31:24]: counter_hi stop event
//!
//! Reset control packs up to 4 counters with an 8-bit shift
//! (per aie-rt `ResetShift = 8`):
//! - bits [6:0] / [7:0]: counter 0 reset event
//! - bits [14:8] / [15:8]: counter 1 reset event
//! - bits [22:16] / [23:16]: counter 2 reset event (4-counter only)
//! - bits [30:24] / [31:24]: counter 3 reset event (4-counter only)

/// Maximum number of performance counters per module (core module has 4).
/// Per aie-rt: Core=4, Memory=2, MemTile=4, Shim(PL)=2.
pub const MAX_PERF_COUNTERS: usize = 4;

/// Runtime state for a single performance counter.
///
/// Tracks whether the counter is actively counting and fires threshold
/// events. Configuration (start/stop/reset event IDs and event values) lives
/// in the parent `PerfCounterBank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CounterState {
    /// Counter is idle (not counting). Enters this state on reset or
    /// initial power-on. Transitions to Active when the start event fires.
    Idle,
    /// Counter is actively incrementing each cycle. Transitions to Stopped
    /// when the stop event fires, or back to Idle on reset event.
    Active,
    /// Counter was stopped by the stop event. Value is preserved.
    /// A subsequent start event re-activates counting from the current value.
    Stopped,
}

impl Default for CounterState {
    fn default() -> Self {
        CounterState::Idle
    }
}

/// A bank of performance counters for one tile module.
///
/// Provides the runtime counting logic that complements the existing
/// register-level configuration in `tile.rs::PerfCounters`. This struct
/// owns the full state: configuration fields, counter values, and runtime
/// active/idle/stopped state.
///
/// # Event-Driven Operation
///
/// Performance counters are event-driven. Call `handle_event()` whenever
/// a hardware event fires in the owning module. If the event matches a
/// counter's start, stop, or reset event, the counter transitions state.
/// Call `tick()` once per cycle to increment all active counters.
///
/// # Threshold Events
///
/// When a counter value reaches its configured `event_value`, the counter
/// generates a `PERF_CNT_N` event. The caller is responsible for routing
/// this event back into the tile's event system. `tick()` returns a list
/// of counter indices that fired threshold events this cycle.
#[derive(Debug, Clone)]
pub struct PerfCounterBank {
    /// Number of valid counters in this module (2 or 4).
    num_counters: usize,

    /// Start event for each counter (7- or 8-bit event ID, 0 = no event configured).
    start_event: [u8; MAX_PERF_COUNTERS],

    /// Stop event for each counter.
    stop_event: [u8; MAX_PERF_COUNTERS],

    /// Reset event for each counter.
    reset_event: [u8; MAX_PERF_COUNTERS],

    /// Current counter values (32-bit, wrapping).
    counter_value: [u32; MAX_PERF_COUNTERS],

    /// Event value (threshold) for each counter. When `counter_value[i]`
    /// reaches `event_value[i]`, a threshold event fires.
    event_value: [u32; MAX_PERF_COUNTERS],

    /// Runtime state for each counter.
    state: [CounterState; MAX_PERF_COUNTERS],
}

impl PerfCounterBank {
    /// Create a new performance counter bank.
    ///
    /// `num_counters` must be <= `MAX_PERF_COUNTERS` (4). Per aie-rt:
    /// - Core module: 4 counters
    /// - Memory module: 2 counters
    /// - MemTile module: 4 counters
    /// - Shim (PL) module: 2 counters
    pub fn new(num_counters: usize) -> Self {
        assert!(
            num_counters <= MAX_PERF_COUNTERS,
            "num_counters ({}) exceeds MAX_PERF_COUNTERS ({})",
            num_counters,
            MAX_PERF_COUNTERS
        );
        Self {
            num_counters,
            start_event: [0; MAX_PERF_COUNTERS],
            stop_event: [0; MAX_PERF_COUNTERS],
            reset_event: [0; MAX_PERF_COUNTERS],
            counter_value: [0; MAX_PERF_COUNTERS],
            event_value: [0; MAX_PERF_COUNTERS],
            state: [CounterState::Idle; MAX_PERF_COUNTERS],
        }
    }

    /// Number of counters in this bank.
    pub fn num_counters(&self) -> usize {
        self.num_counters
    }

    // -- Counter Value Access --

    /// Read the current value of a counter.
    ///
    /// Returns 0 for out-of-range indices (matching hardware behavior where
    /// reads to non-existent registers return 0).
    pub fn read_counter(&self, index: usize) -> u32 {
        if index < self.num_counters {
            self.counter_value[index]
        } else {
            0
        }
    }

    /// Write a counter value directly (register write).
    ///
    /// This sets the counter to any arbitrary value. Per aie-rt
    /// `XAie_PerfCounterSet`, this is a direct 32-bit write to the
    /// counter register. Does not affect counting state.
    pub fn write_counter(&mut self, index: usize, value: u32) {
        if index < self.num_counters {
            self.counter_value[index] = value;
        }
    }

    // -- Event Value (Threshold) Access --

    /// Read the event value (threshold) for a counter.
    ///
    /// When the counter reaches this value, a `PERF_CNT_N` event fires.
    pub fn read_event_value(&self, index: usize) -> u32 {
        if index < self.num_counters {
            self.event_value[index]
        } else {
            0
        }
    }

    /// Write the event value (threshold) for a counter.
    ///
    /// Per aie-rt `XAie_PerfCounterEventValueSet`.
    pub fn write_event_value(&mut self, index: usize, value: u32) {
        if index < self.num_counters {
            self.event_value[index] = value;
        }
    }

    // -- Control Register Access --

    /// Read a start/stop control register value.
    ///
    /// Reconstructs the 32-bit register from the stored start/stop events
    /// for two counters (counter_lo and counter_hi). Layout matches
    /// aie-rt PerfMod with StartStopShift=16.
    ///
    /// `event_width` is 7 for core/memory/shim, 8 for memtile.
    pub fn read_control_start_stop(
        &self,
        counter_lo: usize,
        counter_hi: usize,
        event_width: u32,
    ) -> u32 {
        let _ = event_width; // Width is used for masking on write; reads return full stored value
        let mut val = 0u32;
        if counter_lo < self.num_counters {
            val |= self.start_event[counter_lo] as u32;
            val |= (self.stop_event[counter_lo] as u32) << 8;
        }
        if counter_hi < self.num_counters {
            val |= (self.start_event[counter_hi] as u32) << 16;
            val |= (self.stop_event[counter_hi] as u32) << 24;
        }
        val
    }

    /// Write a start/stop control register.
    ///
    /// Extracts start and stop event IDs for two counters from a 32-bit
    /// register value. Layout per aie-rt: StartStopShift=16, event fields
    /// at bits [event_width-1:0] for start and [event_width+7:8] for stop,
    /// then shifted by 16 for the upper counter.
    pub fn write_control_start_stop(
        &mut self,
        value: u32,
        counter_lo: usize,
        counter_hi: usize,
        event_width: u32,
    ) {
        let mask = (1u32 << event_width) - 1;
        if counter_lo < self.num_counters {
            self.start_event[counter_lo] = (value & mask) as u8;
            self.stop_event[counter_lo] = ((value >> 8) & mask) as u8;
        }
        if counter_hi < self.num_counters {
            self.start_event[counter_hi] = ((value >> 16) & mask) as u8;
            self.stop_event[counter_hi] = ((value >> 24) & mask) as u8;
        }
    }

    /// Read the reset control register value.
    ///
    /// Reconstructs the 32-bit register from stored reset events.
    /// Layout: ResetShift=8, each counter's reset event packed at
    /// bits [7:0], [15:8], [23:16], [31:24].
    pub fn read_control_reset(&self) -> u32 {
        let mut val = 0u32;
        for i in 0..self.num_counters {
            val |= (self.reset_event[i] as u32) << (8 * i);
        }
        val
    }

    /// Write the reset control register.
    ///
    /// Per aie-rt: ResetShift=8, reset events packed at 8-bit boundaries.
    pub fn write_control_reset(&mut self, value: u32, event_width: u32) {
        let mask = (1u32 << event_width) - 1;
        for i in 0..self.num_counters {
            self.reset_event[i] = ((value >> (8 * i)) & mask) as u8;
        }
    }

    // -- Runtime Event Handling --

    /// Check if a counter is currently active (counting).
    pub fn is_active(&self, index: usize) -> bool {
        if index < self.num_counters {
            self.state[index] == CounterState::Active
        } else {
            false
        }
    }

    /// Notify the counter bank that a hardware event has fired.
    ///
    /// Checks each counter to see if the event matches its start, stop,
    /// or reset event. Multiple counters can respond to the same event.
    ///
    /// Event 0 is treated as "no event configured" -- a counter with
    /// start_event=0 will never start from an event with ID 0.
    /// This matches aie-rt behavior where event 0 is XAIE_EVENT_NONE.
    ///
    /// State transitions:
    /// - Start event: Idle -> Active, Stopped -> Active
    /// - Stop event: Active -> Stopped
    /// - Reset event: any state -> Idle, counter value zeroed
    ///
    /// Note: if both start and reset fire on the same event ID, reset
    /// takes priority (counter is zeroed and goes to Idle).
    pub fn handle_event(&mut self, event_id: u8) {
        if event_id == 0 {
            return; // Event 0 = NONE, never matches
        }

        for i in 0..self.num_counters {
            // Reset takes priority over start/stop
            if self.reset_event[i] == event_id && self.reset_event[i] != 0 {
                self.counter_value[i] = 0;
                self.state[i] = CounterState::Idle;
                continue;
            }

            // Stop takes priority over start (if same event is both start and stop,
            // the counter stops -- this matches the hardware behavior where a
            // counter configured with start==stop would toggle, but since we
            // process stop before start, stop wins)
            if self.stop_event[i] == event_id && self.stop_event[i] != 0 {
                if self.state[i] == CounterState::Active {
                    self.state[i] = CounterState::Stopped;
                }
                continue;
            }

            if self.start_event[i] == event_id && self.start_event[i] != 0 {
                if self.state[i] != CounterState::Active {
                    self.state[i] = CounterState::Active;
                }
            }
        }
    }

    /// Advance all active counters by one cycle.
    ///
    /// Returns a vector of counter indices that reached their event value
    /// threshold this cycle. The caller should generate the corresponding
    /// `PERF_CNT_N` events.
    ///
    /// Counter values wrap at u32::MAX (matching 32-bit hardware counters).
    pub fn tick(&mut self) -> Vec<usize> {
        let mut threshold_events = Vec::new();

        for i in 0..self.num_counters {
            if self.state[i] == CounterState::Active {
                self.counter_value[i] = self.counter_value[i].wrapping_add(1);

                // Check threshold: fire event when counter reaches event_value.
                // event_value of 0 means no threshold configured.
                if self.event_value[i] != 0
                    && self.counter_value[i] == self.event_value[i]
                {
                    threshold_events.push(i);
                }
            }
        }

        threshold_events
    }

    // -- Register Interface --

    /// Read a register by offset relative to the performance counter block base.
    ///
    /// This provides a unified register read interface using the block-relative
    /// offsets documented in the module header. Returns `None` for unrecognized
    /// offsets.
    ///
    /// `event_width` controls bit field widths (7 for core/mem/shim, 8 for memtile).
    pub fn read_register(&self, offset: u32, event_width: u32) -> Option<u32> {
        match offset {
            // Control0: cnt0/cnt1 start/stop
            0x00 => Some(self.read_control_start_stop(0, 1, event_width)),
            // Control1: cnt2/cnt3 start/stop (4-counter) or unused (2-counter)
            0x04 => {
                if self.num_counters > 2 {
                    Some(self.read_control_start_stop(2, 3, event_width))
                } else {
                    Some(0)
                }
            }
            // Control2 (4-counter) or Control1 (2-counter): reset events
            0x08 => Some(self.read_control_reset()),
            // Counter values: 0x20, 0x24, 0x28, 0x2C
            off @ 0x20..=0x2C if (off - 0x20) % 4 == 0 => {
                let idx = ((off - 0x20) / 4) as usize;
                Some(self.read_counter(idx))
            }
            // Event values: 0x80, 0x84, 0x88, 0x8C
            off @ 0x80..=0x8C if (off - 0x80) % 4 == 0 => {
                let idx = ((off - 0x80) / 4) as usize;
                Some(self.read_event_value(idx))
            }
            _ => None,
        }
    }

    /// Write a register by offset relative to the performance counter block base.
    ///
    /// Returns `true` if the offset was recognized and handled.
    ///
    /// `event_width` controls bit field widths (7 for core/mem/shim, 8 for memtile).
    pub fn write_register(&mut self, offset: u32, value: u32, event_width: u32) -> bool {
        match offset {
            0x00 => {
                self.write_control_start_stop(value, 0, 1, event_width);
                true
            }
            0x04 => {
                if self.num_counters > 2 {
                    self.write_control_start_stop(value, 2, 3, event_width);
                }
                true
            }
            0x08 => {
                self.write_control_reset(value, event_width);
                true
            }
            off @ 0x20..=0x2C if (off - 0x20) % 4 == 0 => {
                let idx = ((off - 0x20) / 4) as usize;
                self.write_counter(idx, value);
                true
            }
            off @ 0x80..=0x8C if (off - 0x80) % 4 == 0 => {
                let idx = ((off - 0x80) / 4) as usize;
                self.write_event_value(idx, value);
                true
            }
            _ => false,
        }
    }

    /// Get the start event for a counter.
    pub fn start_event(&self, index: usize) -> u8 {
        if index < self.num_counters {
            self.start_event[index]
        } else {
            0
        }
    }

    /// Get the stop event for a counter.
    pub fn stop_event(&self, index: usize) -> u8 {
        if index < self.num_counters {
            self.stop_event[index]
        } else {
            0
        }
    }

    /// Get the reset event for a counter.
    pub fn reset_event(&self, index: usize) -> u8 {
        if index < self.num_counters {
            self.reset_event[index]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction and defaults --

    #[test]
    fn new_core_module_has_4_counters() {
        let bank = PerfCounterBank::new(4);
        assert_eq!(bank.num_counters(), 4);
    }

    #[test]
    fn new_memory_module_has_2_counters() {
        let bank = PerfCounterBank::new(2);
        assert_eq!(bank.num_counters(), 2);
    }

    #[test]
    fn new_counters_default_to_zero_and_idle() {
        let bank = PerfCounterBank::new(4);
        for i in 0..4 {
            assert_eq!(bank.read_counter(i), 0);
            assert_eq!(bank.read_event_value(i), 0);
            assert_eq!(bank.start_event(i), 0);
            assert_eq!(bank.stop_event(i), 0);
            assert_eq!(bank.reset_event(i), 0);
            assert!(!bank.is_active(i));
        }
    }

    #[test]
    #[should_panic(expected = "num_counters (5) exceeds MAX_PERF_COUNTERS (4)")]
    fn new_panics_on_too_many_counters() {
        PerfCounterBank::new(5);
    }

    #[test]
    fn new_zero_counters_is_valid() {
        let bank = PerfCounterBank::new(0);
        assert_eq!(bank.num_counters(), 0);
    }

    // -- Direct counter value read/write --

    #[test]
    fn write_counter_then_read_back() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_counter(0, 0xDEAD_BEEF);
        bank.write_counter(3, 0xCAFE_BABE);
        assert_eq!(bank.read_counter(0), 0xDEAD_BEEF);
        assert_eq!(bank.read_counter(3), 0xCAFE_BABE);
        // Unwritten counters remain 0
        assert_eq!(bank.read_counter(1), 0);
        assert_eq!(bank.read_counter(2), 0);
    }

    #[test]
    fn write_counter_out_of_range_is_noop() {
        let mut bank = PerfCounterBank::new(2);
        bank.write_counter(2, 42);
        assert_eq!(bank.read_counter(2), 0);
    }

    #[test]
    fn read_counter_out_of_range_returns_zero() {
        let bank = PerfCounterBank::new(2);
        assert_eq!(bank.read_counter(5), 0);
    }

    // -- Event value (threshold) read/write --

    #[test]
    fn write_event_value_then_read_back() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_event_value(0, 1000);
        bank.write_event_value(3, 5000);
        assert_eq!(bank.read_event_value(0), 1000);
        assert_eq!(bank.read_event_value(3), 5000);
    }

    #[test]
    fn write_event_value_out_of_range_is_noop() {
        let mut bank = PerfCounterBank::new(2);
        bank.write_event_value(3, 42);
        assert_eq!(bank.read_event_value(3), 0);
    }

    // -- Control register write/read round-trip --

    #[test]
    fn control_start_stop_round_trip_7bit() {
        let mut bank = PerfCounterBank::new(4);
        // Pack: start0=42, stop0=43, start1=44, stop1=45
        let value = 42 | (43 << 8) | (44 << 16) | (45 << 24);
        bank.write_control_start_stop(value, 0, 1, 7);

        assert_eq!(bank.start_event(0), 42);
        assert_eq!(bank.stop_event(0), 43);
        assert_eq!(bank.start_event(1), 44);
        assert_eq!(bank.stop_event(1), 45);

        // Read back should reconstruct the same value
        let readback = bank.read_control_start_stop(0, 1, 7);
        assert_eq!(readback, value);
    }

    #[test]
    fn control_start_stop_round_trip_8bit_memtile() {
        let mut bank = PerfCounterBank::new(4);
        // MemTile uses 8-bit event fields, can have values > 127
        let value = 200u32 | (201 << 8) | (202 << 16) | (203 << 24);
        bank.write_control_start_stop(value, 0, 1, 8);

        assert_eq!(bank.start_event(0), 200);
        assert_eq!(bank.stop_event(0), 201);
        assert_eq!(bank.start_event(1), 202);
        assert_eq!(bank.stop_event(1), 203);
    }

    #[test]
    fn control_start_stop_7bit_masks_high_bits() {
        let mut bank = PerfCounterBank::new(4);
        // Value 0xFF should be masked to 0x7F for 7-bit events
        let value = 0xFF | (0xFF << 8);
        bank.write_control_start_stop(value, 0, 1, 7);
        assert_eq!(bank.start_event(0), 0x7F);
        assert_eq!(bank.stop_event(0), 0x7F);
    }

    #[test]
    fn control_reset_round_trip() {
        let mut bank = PerfCounterBank::new(4);
        let value = 5 | (6 << 8) | (7 << 16) | (8 << 24);
        bank.write_control_reset(value, 7);

        assert_eq!(bank.reset_event(0), 5);
        assert_eq!(bank.reset_event(1), 6);
        assert_eq!(bank.reset_event(2), 7);
        assert_eq!(bank.reset_event(3), 8);

        let readback = bank.read_control_reset();
        assert_eq!(readback, value);
    }

    #[test]
    fn control_reset_2_counter_module() {
        let mut bank = PerfCounterBank::new(2);
        let value = 10 | (20 << 8) | (30 << 16) | (40 << 24);
        bank.write_control_reset(value, 7);

        // Only first 2 counters should be set
        assert_eq!(bank.reset_event(0), 10);
        assert_eq!(bank.reset_event(1), 20);
        // Out of range
        assert_eq!(bank.reset_event(2), 0);
        assert_eq!(bank.reset_event(3), 0);
    }

    // -- Start/Stop/Reset Event Handling --

    #[test]
    fn counter_starts_on_start_event() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(42 | (0 << 8), 0, 1, 7); // start=42, stop=0

        assert!(!bank.is_active(0));
        bank.handle_event(42);
        assert!(bank.is_active(0));
    }

    #[test]
    fn counter_stops_on_stop_event() {
        let mut bank = PerfCounterBank::new(4);
        // start=42, stop=43
        bank.write_control_start_stop(42 | (43 << 8), 0, 1, 7);

        bank.handle_event(42); // start
        assert!(bank.is_active(0));

        bank.handle_event(43); // stop
        assert!(!bank.is_active(0));
    }

    #[test]
    fn counter_increments_each_tick_while_active() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(42 | (43 << 8), 0, 1, 7);

        // Not active yet -- tick should not increment
        bank.tick();
        assert_eq!(bank.read_counter(0), 0);

        // Start counting
        bank.handle_event(42);
        bank.tick();
        assert_eq!(bank.read_counter(0), 1);
        bank.tick();
        assert_eq!(bank.read_counter(0), 2);
        bank.tick();
        assert_eq!(bank.read_counter(0), 3);

        // Stop counting
        bank.handle_event(43);
        bank.tick();
        assert_eq!(bank.read_counter(0), 3); // Value preserved
    }

    #[test]
    fn counter_does_not_increment_when_idle() {
        let mut bank = PerfCounterBank::new(2);
        // No start event configured
        for _ in 0..100 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 0);
        assert_eq!(bank.read_counter(1), 0);
    }

    #[test]
    fn reset_event_zeroes_counter_and_goes_idle() {
        let mut bank = PerfCounterBank::new(4);
        // start=10, stop=0
        bank.write_control_start_stop(10, 0, 1, 7);
        // reset=20
        bank.write_control_reset(20, 7);

        bank.handle_event(10); // start
        for _ in 0..50 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 50);
        assert!(bank.is_active(0));

        bank.handle_event(20); // reset
        assert_eq!(bank.read_counter(0), 0);
        assert!(!bank.is_active(0));
    }

    #[test]
    fn reset_event_takes_priority_over_start() {
        let mut bank = PerfCounterBank::new(4);
        // Both start and reset are the same event
        bank.write_control_start_stop(10, 0, 1, 7); // start=10
        bank.write_control_reset(10, 7); // reset=10

        bank.write_counter(0, 100);
        bank.handle_event(10);
        // Reset should take priority: counter zeroed, state is Idle
        assert_eq!(bank.read_counter(0), 0);
        assert!(!bank.is_active(0));
    }

    #[test]
    fn multiple_counters_independent() {
        let mut bank = PerfCounterBank::new(4);
        // Counter 0: start=10, stop=11
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
        // Counter 2: start=20, stop=21
        bank.write_control_start_stop(20 | (21 << 8), 2, 3, 7);

        // Start counter 0 only
        bank.handle_event(10);
        for _ in 0..5 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 5);
        assert_eq!(bank.read_counter(2), 0);

        // Now start counter 2
        bank.handle_event(20);
        for _ in 0..3 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 8); // 5 + 3
        assert_eq!(bank.read_counter(2), 3);

        // Stop counter 0
        bank.handle_event(11);
        for _ in 0..2 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 8); // Stopped
        assert_eq!(bank.read_counter(2), 5); // Still counting
    }

    #[test]
    fn stopped_counter_can_restart() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

        bank.handle_event(10); // start
        for _ in 0..5 {
            bank.tick();
        }
        bank.handle_event(11); // stop
        assert_eq!(bank.read_counter(0), 5);

        bank.handle_event(10); // restart from current value
        for _ in 0..3 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 8); // 5 + 3, continues from where stopped
    }

    #[test]
    fn event_0_never_matches() {
        let mut bank = PerfCounterBank::new(4);
        // start_event is 0 (default)
        bank.handle_event(0);
        assert!(!bank.is_active(0)); // Should not start
    }

    #[test]
    fn direct_write_does_not_affect_state() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

        // Directly write a counter value while idle
        bank.write_counter(0, 1000);
        assert_eq!(bank.read_counter(0), 1000);
        assert!(!bank.is_active(0)); // Still idle

        // Start and verify it counts from the written value
        bank.handle_event(10);
        bank.tick();
        assert_eq!(bank.read_counter(0), 1001);
    }

    // -- Threshold Events --

    #[test]
    fn tick_fires_threshold_event_when_counter_reaches_event_value() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);
        bank.write_event_value(0, 5);

        bank.handle_event(10); // start

        // Tick 4 times -- no threshold yet
        for _ in 0..4 {
            let events = bank.tick();
            assert!(events.is_empty());
        }
        assert_eq!(bank.read_counter(0), 4);

        // Tick once more -- counter hits 5 = event_value
        let events = bank.tick();
        assert_eq!(events, vec![0]);
        assert_eq!(bank.read_counter(0), 5);
    }

    #[test]
    fn threshold_event_fires_only_once_at_exact_value() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);
        bank.write_event_value(0, 3);

        bank.handle_event(10);

        // Tick to threshold
        bank.tick(); // 1
        bank.tick(); // 2
        let events = bank.tick(); // 3 -- fires
        assert_eq!(events, vec![0]);

        // Subsequent ticks should not re-fire
        let events = bank.tick(); // 4
        assert!(events.is_empty());
    }

    #[test]
    fn event_value_zero_means_no_threshold() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);
        // event_value defaults to 0

        bank.handle_event(10);
        for _ in 0..100 {
            let events = bank.tick();
            assert!(events.is_empty());
        }
    }

    #[test]
    fn multiple_counters_can_fire_threshold_same_tick() {
        let mut bank = PerfCounterBank::new(4);
        // Both counter 0 and counter 1 start on event 10
        bank.write_control_start_stop(10 | (10 << 16), 0, 1, 7);
        bank.write_event_value(0, 3);
        bank.write_event_value(1, 3);

        bank.handle_event(10);
        bank.tick(); // 1
        bank.tick(); // 2
        let events = bank.tick(); // 3
        assert_eq!(events, vec![0, 1]);
    }

    #[test]
    fn counter_wraps_at_u32_max() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);
        bank.write_counter(0, u32::MAX);

        bank.handle_event(10);
        bank.tick();
        assert_eq!(bank.read_counter(0), 0); // Wrapped
    }

    // -- Register Interface --

    #[test]
    fn register_read_write_round_trip_control0() {
        let mut bank = PerfCounterBank::new(4);
        let value = 42 | (43 << 8) | (44 << 16) | (45 << 24);
        assert!(bank.write_register(0x00, value, 7));
        assert_eq!(bank.read_register(0x00, 7), Some(value));
    }

    #[test]
    fn register_read_write_round_trip_control1_4counter() {
        let mut bank = PerfCounterBank::new(4);
        let value = 10 | (11 << 8) | (12 << 16) | (13 << 24);
        assert!(bank.write_register(0x04, value, 7));
        assert_eq!(bank.read_register(0x04, 7), Some(value));
    }

    #[test]
    fn register_read_write_round_trip_reset() {
        let mut bank = PerfCounterBank::new(4);
        let value = 5 | (6 << 8) | (7 << 16) | (8 << 24);
        assert!(bank.write_register(0x08, value, 7));
        assert_eq!(bank.read_register(0x08, 7), Some(value));
    }

    #[test]
    fn register_read_write_counter_values() {
        let mut bank = PerfCounterBank::new(4);
        assert!(bank.write_register(0x20, 0xAAAA, 7));
        assert!(bank.write_register(0x24, 0xBBBB, 7));
        assert!(bank.write_register(0x28, 0xCCCC, 7));
        assert!(bank.write_register(0x2C, 0xDDDD, 7));

        assert_eq!(bank.read_register(0x20, 7), Some(0xAAAA));
        assert_eq!(bank.read_register(0x24, 7), Some(0xBBBB));
        assert_eq!(bank.read_register(0x28, 7), Some(0xCCCC));
        assert_eq!(bank.read_register(0x2C, 7), Some(0xDDDD));
    }

    #[test]
    fn register_read_write_event_values() {
        let mut bank = PerfCounterBank::new(4);
        assert!(bank.write_register(0x80, 1000, 7));
        assert!(bank.write_register(0x84, 2000, 7));
        assert!(bank.write_register(0x88, 3000, 7));
        assert!(bank.write_register(0x8C, 4000, 7));

        assert_eq!(bank.read_register(0x80, 7), Some(1000));
        assert_eq!(bank.read_register(0x84, 7), Some(2000));
        assert_eq!(bank.read_register(0x88, 7), Some(3000));
        assert_eq!(bank.read_register(0x8C, 7), Some(4000));
    }

    #[test]
    fn register_unknown_offset_returns_none() {
        let bank = PerfCounterBank::new(4);
        assert_eq!(bank.read_register(0x10, 7), None);
        assert_eq!(bank.read_register(0x30, 7), None);
        assert_eq!(bank.read_register(0x90, 7), None);
    }

    #[test]
    fn register_unknown_offset_write_returns_false() {
        let mut bank = PerfCounterBank::new(4);
        assert!(!bank.write_register(0x10, 42, 7));
        assert!(!bank.write_register(0xFF, 42, 7));
    }

    // -- Integration-style tests --

    #[test]
    fn full_lifecycle_start_count_threshold_stop_reset() {
        let mut bank = PerfCounterBank::new(4);

        // Configure counter 0: start=10, stop=11, reset=12, threshold=5
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
        bank.write_control_reset(12, 7);
        bank.write_event_value(0, 5);

        // Idle: should not count
        bank.tick();
        assert_eq!(bank.read_counter(0), 0);

        // Start
        bank.handle_event(10);
        assert!(bank.is_active(0));

        // Count to threshold
        for _ in 0..4 {
            let ev = bank.tick();
            assert!(ev.is_empty());
        }
        assert_eq!(bank.read_counter(0), 4);

        let ev = bank.tick(); // counter hits 5
        assert_eq!(ev, vec![0]);
        assert_eq!(bank.read_counter(0), 5);

        // Counter keeps going past threshold
        bank.tick();
        assert_eq!(bank.read_counter(0), 6);

        // Stop
        bank.handle_event(11);
        assert!(!bank.is_active(0));
        bank.tick();
        assert_eq!(bank.read_counter(0), 6); // Preserved

        // Reset
        bank.handle_event(12);
        assert_eq!(bank.read_counter(0), 0);
        assert!(!bank.is_active(0));
    }

    #[test]
    fn two_counter_module_ignores_counters_2_and_3() {
        let mut bank = PerfCounterBank::new(2);

        // Configure via register: counter 2/3 should be ignored
        let value = 10 | (11 << 8) | (20 << 16) | (21 << 24);
        bank.write_control_start_stop(value, 0, 1, 7);

        assert_eq!(bank.start_event(0), 10);
        assert_eq!(bank.stop_event(0), 11);
        assert_eq!(bank.start_event(1), 20);
        assert_eq!(bank.stop_event(1), 21);

        // Write to counter index 2 should be silently ignored
        bank.write_counter(2, 42);
        assert_eq!(bank.read_counter(2), 0);

        // But counter 0 and 1 should work fine
        bank.handle_event(10);
        bank.tick();
        assert_eq!(bank.read_counter(0), 1);
    }

    #[test]
    fn stop_event_preserves_value_across_restart() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

        // First run: count to 10
        bank.handle_event(10);
        for _ in 0..10 {
            bank.tick();
        }
        bank.handle_event(11);
        assert_eq!(bank.read_counter(0), 10);

        // Second run: count continues from 10
        bank.handle_event(10);
        for _ in 0..5 {
            bank.tick();
        }
        assert_eq!(bank.read_counter(0), 15);
    }

    #[test]
    fn reset_while_stopped_zeroes_value() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
        bank.write_control_reset(12, 7);

        bank.handle_event(10); // start
        for _ in 0..5 {
            bank.tick();
        }
        bank.handle_event(11); // stop
        assert_eq!(bank.read_counter(0), 5);

        bank.handle_event(12); // reset while stopped
        assert_eq!(bank.read_counter(0), 0);
        assert!(!bank.is_active(0));
    }

    #[test]
    fn threshold_wraps_around() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);
        bank.write_counter(0, u32::MAX - 1);
        bank.write_event_value(0, 1); // threshold at value 1

        bank.handle_event(10);
        // u32::MAX - 1 -> u32::MAX (no threshold, event_value=1 != u32::MAX)
        let ev = bank.tick();
        assert!(ev.is_empty());
        // u32::MAX -> 0 (wraps, 0 != 1)
        let ev = bank.tick();
        assert!(ev.is_empty());
        // 0 -> 1 (matches threshold!)
        let ev = bank.tick();
        assert_eq!(ev, vec![0]);
    }

    #[test]
    fn handle_event_with_no_matching_counter_is_noop() {
        let mut bank = PerfCounterBank::new(4);
        bank.write_control_start_stop(10, 0, 1, 7);

        // Event 99 does not match any configured event
        bank.handle_event(99);
        assert!(!bank.is_active(0));
        assert_eq!(bank.read_counter(0), 0);
    }

    #[test]
    fn same_event_for_multiple_counters() {
        let mut bank = PerfCounterBank::new(4);
        // Counter 0 and counter 1 both start on event 10
        bank.write_control_start_stop(10 | (10 << 16), 0, 1, 7);

        bank.handle_event(10);
        assert!(bank.is_active(0));
        assert!(bank.is_active(1));

        bank.tick();
        assert_eq!(bank.read_counter(0), 1);
        assert_eq!(bank.read_counter(1), 1);
    }
}
