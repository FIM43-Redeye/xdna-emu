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
/// register-level configuration managed by the Tile struct. This struct
/// owns the full state: configuration fields, counter values, and runtime
/// active/idle/stopped state.
///
/// # Event-Driven Operation
///
/// Performance counters are event-driven. Call `handle_event()` whenever
/// a hardware event fires in the owning module. If the event matches a
/// counter's start, stop, or reset event, the counter transitions state.
/// Call `tick_active_cycles()` or `tick_idle_cycles()` once per cycle to
/// increment active counters. ACTIVE_CORE-configured counters (event 0x1C)
/// are level-gated: they tick only during cycles when the core is in
/// Execute state.
///
/// # Threshold Events
///
/// When a counter value reaches its configured `event_value`, the counter
/// generates a `PERF_CNT_N` event. The caller is responsible for routing
/// this event back into the tile's event system. The per-cycle tick methods
/// return a list of counter indices that fired threshold events this cycle.
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

    /// Reset to construction defaults, preserving `num_counters`.
    ///
    /// Mirrors a real-HW column reset on hw_context teardown: zeroes
    /// counter values, event thresholds, control events, and runtime
    /// state. CDO/PDI replay and the patched insts.bin re-configure
    /// thresholds and start/stop events before the next run consumes
    /// them, so clearing here is safe and matches HW behavior.
    pub fn reset(&mut self) {
        *self = Self::new(self.num_counters);
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
    pub fn read_control_start_stop(&self, counter_lo: usize, counter_hi: usize, event_width: u32) -> u32 {
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
    /// - Reset event: counter value zeroed; run state preserved
    ///
    /// Reset is independent of the start/stop state machine: per aie-rt's
    /// `XAie_PerfCounterReset`, a reset zeros the counter register but does
    /// not stop it. The common "self-reset" pattern (`reset_event =
    /// PERF_CNT_N`) relies on this -- after the threshold fires and the
    /// counter zeroes, it keeps ticking and fires again every period.
    pub fn handle_event(&mut self, event_id: u8) {
        if event_id == 0 {
            return; // Event 0 = NONE, never matches
        }

        for i in 0..self.num_counters {
            // Reset zeros the counter value but does not change run state.
            if self.reset_event[i] == event_id && self.reset_event[i] != 0 {
                self.counter_value[i] = 0;
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

    /// Advance every Active counter by one cycle and return the indices of
    /// any that crossed their threshold this cycle.
    ///
    /// Hardware perf counters are duration counters: once a `start_event`
    /// pulses the counter into the Active state, it ticks every cycle until
    /// a `stop_event` halts it, regardless of whether the core itself is
    /// executing or stalled. ACTIVE_CORE (0x1C) and DISABLED_CORE
    /// (`XAIE_EVENT_DISABLED_CORE`) are edge events used as start/stop
    /// triggers, NOT level-asserted per-cycle gates -- so the counter must
    /// NOT pause when the core stalls.
    ///
    /// Counter values wrap at u32::MAX (matching 32-bit hardware counters).
    pub fn tick(&mut self) -> Vec<usize> {
        let mut threshold_events = Vec::new();

        for i in 0..self.num_counters {
            if self.state[i] != CounterState::Active {
                continue;
            }

            self.counter_value[i] = self.counter_value[i].wrapping_add(1);

            // Check threshold: fire event when counter reaches event_value.
            // event_value of 0 means no threshold configured.
            if self.event_value[i] != 0 && self.counter_value[i] == self.event_value[i] {
                threshold_events.push(i);
            }
        }

        threshold_events
    }

    /// Deprecated: use [`tick`]. Retained for source compatibility with
    /// callers that pre-date the duration-counter fix; both variants now
    /// behave identically.
    #[deprecated(note = "perf counters tick every cycle regardless of core state; use tick()")]
    pub fn tick_active_cycles(&mut self) -> Vec<usize> {
        self.tick()
    }

    /// Deprecated: use [`tick`]. See [`tick_active_cycles`].
    #[deprecated(note = "perf counters tick every cycle regardless of core state; use tick()")]
    pub fn tick_idle_cycles(&mut self) -> Vec<usize> {
        self.tick()
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
mod tests;
