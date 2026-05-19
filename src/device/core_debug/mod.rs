//! Core debug subsystem for AIE2 compute tiles.
//!
//! Handles external observation and control of a running AIE core via
//! the core status, core control, and debug control registers. All bit
//! positions and register offsets are derived from aie-rt
//! `xaiemlgbl_params.h` (XAIEMLGBL_CORE_MODULE_*).
//!
//! # Register Map (from aie-rt)
//!
//! | Register        | Offset   | Key Fields                                        |
//! |-----------------|----------|---------------------------------------------------|
//! | Core_Control    | 0x32000  | [1] reset, [0] enable                             |
//! | Core_Status     | 0x32004  | [20] done, [19] error_halt, [16] debug_halt, etc. |
//! | Debug_Control0  | 0x32010  | [5:2] single_step_count, [0] debug_halt_bit       |
//! | Debug_Control1  | 0x32014  | Event-based halt/resume/single-step config         |
//! | Debug_Control2  | 0x32018  | Stall-to-halt enables (stream, lock, mem, PC)      |
//! | Debug_Status    | 0x3201C  | Which halt cause is active                         |
//! | Core_PC         | 0x31100  | [19:0] program counter (20-bit)                   |
//! | Core_SP         | 0x31120  | [19:0] stack pointer (20-bit)                     |
//! | Core_LR         | 0x31130  | [19:0] link register (20-bit)                     |

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Register offsets (from aie-rt xaiemlgbl_params.h)
// ---------------------------------------------------------------------------

/// Core_Control register offset within a tile.
const REG_CORE_CONTROL: u32 = 0x0003_2000;
/// Core_Status register offset within a tile.
const REG_CORE_STATUS: u32 = 0x0003_2004;
/// Debug_Control0 register offset within a tile.
const REG_DEBUG_CONTROL0: u32 = 0x0003_2010;
/// Debug_Control1 register offset within a tile.
const REG_DEBUG_CONTROL1: u32 = 0x0003_2014;
/// Debug_Control2 register offset within a tile.
const REG_DEBUG_CONTROL2: u32 = 0x0003_2018;
/// Debug_Status register offset within a tile.
const REG_DEBUG_STATUS: u32 = 0x0003_201C;
/// PC_Event0..3 register offsets (each: bit 31 VALID, bits [13:0] PC_ADDRESS).
const REG_PC_EVENT0: u32 = 0x0003_2020;
const REG_PC_EVENT1: u32 = 0x0003_2024;
const REG_PC_EVENT2: u32 = 0x0003_2028;
const REG_PC_EVENT3: u32 = 0x0003_202C;
/// Core_PC register offset within a tile.
const REG_CORE_PC: u32 = 0x0003_1100;
/// Core_SP register offset within a tile.
const REG_CORE_SP: u32 = 0x0003_1120;
/// Core_LR register offset within a tile.
const REG_CORE_LR: u32 = 0x0003_1130;

// ---------------------------------------------------------------------------
// PC_Event* bit positions (per aie-rt xaiemlgbl_params.h)
// ---------------------------------------------------------------------------

/// PC_Event*: bit 31 VALID enable.
const PC_EVENT_VALID_MASK: u32 = 0x8000_0000;
/// PC_Event*: bits [13:0] 14-bit PC_ADDRESS.
const PC_EVENT_ADDRESS_MASK: u32 = 0x0000_3FFF;
/// Writable bits in PC_Event* (per AM025 register mask 0x80003FFF).
const PC_EVENT_WRITE_MASK: u32 = PC_EVENT_VALID_MASK | PC_EVENT_ADDRESS_MASK;

// ---------------------------------------------------------------------------
// Core event IDs broadcast on PC matches (from xaie_events_aieml.h)
// ---------------------------------------------------------------------------

/// XAIEML_EVENTS_CORE_PC_0 = 16. Fires when PC matches PC_Event0.
const EVENT_CORE_PC_0: u8 = 16;
/// XAIEML_EVENTS_CORE_PC_1 = 17.
const EVENT_CORE_PC_1: u8 = 17;
/// XAIEML_EVENTS_CORE_PC_2 = 18.
const EVENT_CORE_PC_2: u8 = 18;
/// XAIEML_EVENTS_CORE_PC_3 = 19.
const EVENT_CORE_PC_3: u8 = 19;
/// XAIEML_EVENTS_CORE_PC_RANGE_0_1 = 20. Fires while PC is in [PC0, PC1]
/// (both PC_Event0 and PC_Event1 must be VALID).
const EVENT_CORE_PC_RANGE_0_1: u8 = 20;
/// XAIEML_EVENTS_CORE_PC_RANGE_2_3 = 21.
const EVENT_CORE_PC_RANGE_2_3: u8 = 21;

// ---------------------------------------------------------------------------
// Core_Status bit positions (from xaiemlgbl_params.h)
// ---------------------------------------------------------------------------

/// Bit 0: core is enabled.
const STATUS_ENABLE_LSB: u32 = 0;
/// Bit 1: core is in reset.
const STATUS_RESET_LSB: u32 = 1;
/// Bits 2-5: memory stall (S, W, N, E directions).
const STATUS_MEMORY_STALL_S_LSB: u32 = 2;
const STATUS_MEMORY_STALL_W_LSB: u32 = 3;
const STATUS_MEMORY_STALL_N_LSB: u32 = 4;
const STATUS_MEMORY_STALL_E_LSB: u32 = 5;
/// Bits 6-9: lock stall (S, W, N, E directions).
const STATUS_LOCK_STALL_S_LSB: u32 = 6;
const STATUS_LOCK_STALL_W_LSB: u32 = 7;
const STATUS_LOCK_STALL_N_LSB: u32 = 8;
const STATUS_LOCK_STALL_E_LSB: u32 = 9;
/// Bit 10: stream stall SS0.
const STATUS_STREAM_STALL_SS0_LSB: u32 = 10;
/// Bit 12: stream stall MS0.
const STATUS_STREAM_STALL_MS0_LSB: u32 = 12;
/// Bits 14-15: cascade stall (SCD, MCD).
const STATUS_CASCADE_STALL_SCD_LSB: u32 = 14;
const STATUS_CASCADE_STALL_MCD_LSB: u32 = 15;
/// Bit 16: debug halt.
const STATUS_DEBUG_HALT_LSB: u32 = 16;
/// Bit 17: ECC error stall.
const STATUS_ECC_ERROR_STALL_LSB: u32 = 17;
/// Bit 19: error halt.
const STATUS_ERROR_HALT_LSB: u32 = 19;
/// Bit 20: core done.
const STATUS_DONE_LSB: u32 = 20;

// ---------------------------------------------------------------------------
// Core_Control bit positions
// ---------------------------------------------------------------------------

/// Bit 0: enable.
const CTRL_ENABLE_LSB: u32 = 0;
const CTRL_ENABLE_MASK: u32 = 1 << CTRL_ENABLE_LSB;
/// Bit 1: reset.
const CTRL_RESET_LSB: u32 = 1;
const CTRL_RESET_MASK: u32 = 1 << CTRL_RESET_LSB;

// ---------------------------------------------------------------------------
// Debug_Control0 bit positions
// ---------------------------------------------------------------------------

/// Bit 0: debug halt request.
const DBG_CTRL0_HALT_LSB: u32 = 0;
const DBG_CTRL0_HALT_MASK: u32 = 1 << DBG_CTRL0_HALT_LSB;
/// Bits [5:2]: single-step count.
const DBG_CTRL0_SSTEP_COUNT_LSB: u32 = 2;
const DBG_CTRL0_SSTEP_COUNT_MASK: u32 = 0xF << DBG_CTRL0_SSTEP_COUNT_LSB;

// ---------------------------------------------------------------------------
// Debug_Control1 bit positions (per AM025)
// ---------------------------------------------------------------------------

/// Bits [6:0]: resume event ID (clears halt when fired).
const DBG_CTRL1_RESUME_EVENT_LSB: u32 = 0;
const DBG_CTRL1_EVENT_MASK: u32 = 0x7F;
/// Bits [14:8]: single-step event ID. When the configured event fires,
/// the core completes the current bundle and then halts (interpretation
/// (a): triggering bundle is the last to commit). Drained by the
/// coordinator via `consume_pending_single_step` after each step.
const DBG_CTRL1_SSTEP_EVENT_LSB: u32 = 8;
/// Bits [22:16]: halt event 0 ID.
const DBG_CTRL1_HALT_EVENT0_LSB: u32 = 16;
/// Bits [30:24]: halt event 1 ID.
const DBG_CTRL1_HALT_EVENT1_LSB: u32 = 24;

// ---------------------------------------------------------------------------
// Debug_Control2 bit positions (per AM025; stall-to-halt enables)
// ---------------------------------------------------------------------------

/// Bit 0: halt on any PC event firing (independent of Debug_Control1.HaltEvent*).
const DBG_CTRL2_PC_EVENT_HALT_LSB: u32 = 0;
/// Bit 1: halt on memory stall.
const DBG_CTRL2_MEM_STALL_HALT_LSB: u32 = 1;
/// Bit 2: halt on lock stall.
const DBG_CTRL2_LOCK_STALL_HALT_LSB: u32 = 2;
/// Bit 3: halt on stream stall.
const DBG_CTRL2_STREAM_STALL_HALT_LSB: u32 = 3;

// ---------------------------------------------------------------------------
// Debug_Status bit positions
// ---------------------------------------------------------------------------

/// Bit 0: halted by debug halt request (any cause).
const DBG_STS_HALTED_LSB: u32 = 0;
/// Bit 1: halted due to PC event.
const DBG_STS_PC_EVENT_HALTED_LSB: u32 = 1;
/// Bit 2: halted due to memory stall.
const DBG_STS_MEM_STALL_HALTED_LSB: u32 = 2;
/// Bit 3: halted due to lock stall.
const DBG_STS_LOCK_STALL_HALTED_LSB: u32 = 3;
/// Bit 4: halted due to stream stall.
const DBG_STS_STREAM_STALL_HALTED_LSB: u32 = 4;
/// Bit 5: halted due to Debug_Halt_Core_Event0.
const DBG_STS_EVENT0_HALTED_LSB: u32 = 5;
/// Bit 6: halted due to Debug_Halt_Core_Event1.
const DBG_STS_EVENT1_HALTED_LSB: u32 = 6;

/// 20-bit address mask for PC, SP, and LR values.
const ADDR_MASK_20BIT: u32 = 0x000F_FFFF;

// ---------------------------------------------------------------------------
// Core Debug State
// ---------------------------------------------------------------------------

/// External observation and control state for an AIE2 compute core.
///
/// This struct models the host-visible registers that allow debuggers and
/// management software to inspect and control core execution. It does NOT
/// drive the interpreter -- it is a projection of interpreter state into
/// the register space, plus a latch for external halt/resume requests.
///
/// The status register layout matches the hardware exactly (per aie-rt
/// `xaiemlgbl_params.h`), so host software reading these registers gets
/// the same bit patterns as on real silicon.
#[derive(Debug, Clone)]
pub struct CoreDebugState {
    // -- Core state bits (mirrored from interpreter) --
    /// Core is enabled (executing or ready to execute).
    pub(super) enabled: bool,
    /// Core is held in reset.
    pub(super) reset: bool,
    /// Core has reached end of program.
    pub(super) done: bool,

    // -- Stall indicators (updated by interpreter each cycle) --
    /// Memory stall: core blocked on memory access.
    /// Packed as 4 directional bits (S, W, N, E) in the status register.
    pub(super) mem_stall: bool,
    /// Lock stall: core blocked on lock acquire.
    /// Packed as 4 directional bits (S, W, N, E) in the status register.
    pub(super) lock_stall: bool,
    /// Stream stall: core blocked on stream read/write.
    pub(super) stream_stall: bool,
    /// Cascade stall: core blocked on cascade input/output.
    pub(super) cascade_stall: bool,

    // -- Debug control --
    /// Core halted by debug request (Debug_Control0 bit 0).
    pub(super) halted: bool,
    /// ECC error has halted the core.
    pub(super) ecc_error: bool,
    /// Generic execution error has halted the core (decode failure,
    /// unhandled instruction, missing program memory). Distinct from
    /// `ecc_error` -- both contribute to the Error_Halt status bit (19),
    /// but only `ecc_error` raises ECC_ERROR_STALL (bit 17).
    pub(super) error_halt: bool,
    /// Single-step mode active.
    pub(super) single_step: bool,
    /// Single-step count (Debug_Control0 bits [5:2]).
    pub(super) single_step_count: u8,

    // -- Observable registers --
    /// Program counter (20-bit).
    pub(super) pc: u32,
    /// Stack pointer (20-bit).
    sp: u32,
    /// Link register (20-bit).
    lr: u32,

    // -- Debug_Control1 raw value (event-based halt/resume config) --
    pub(super) debug_ctrl1: u32,
    // -- Debug_Control2 raw value (stall-to-halt enables) --
    pub(super) debug_ctrl2: u32,

    // -- PC_Event0..3 raw register values (VALID at bit 31, PC_ADDRESS at [13:0]) --
    pub(super) pc_event0: u32,
    pub(super) pc_event1: u32,
    pub(super) pc_event2: u32,
    pub(super) pc_event3: u32,

    // -- Halt-cause latches (Debug_Status bits 1-6) --
    /// Set when a configured PC event triggered the halt. Not currently
    /// driven (PC_Event registers aren't modeled).
    pub(super) halt_cause_pc_event: bool,
    /// Set when memory stall + Debug_Control2.MEM_STALL_HALT triggered halt.
    pub(super) halt_cause_mem_stall: bool,
    /// Set when lock stall + Debug_Control2.LOCK_STALL_HALT triggered halt.
    pub(super) halt_cause_lock_stall: bool,
    /// Set when stream stall + Debug_Control2.STREAM_STALL_HALT triggered halt.
    pub(super) halt_cause_stream_stall: bool,
    /// Set when an event matching Debug_Halt_Core_Event0 fired.
    pub(super) halt_cause_event0: bool,
    /// Set when an event matching Debug_Halt_Core_Event1 fired.
    pub(super) halt_cause_event1: bool,
    /// Set when an event matching Debug_Control1.SSTEP_EVENT fired during
    /// the current step. Drained by the coordinator after the step
    /// completes via `consume_pending_single_step`, which clears the latch
    /// and requests halt.
    pub(super) pending_single_step: bool,

    /// PC of the most recently consumed synchronous PC_Event pre-execute
    /// trap. When set, the pre-execute seam in the coordinator skips the
    /// match check for exactly one step (the step immediately after resume
    /// from a pre-execute halt), allowing the core to execute the trap
    /// bundle and advance past it. Cleared by
    /// `clear_sync_trap_consumed()` after the bundle executes.
    ///
    /// Phase B Unit 1, spec §5.1: the before-commit seam must not re-fire
    /// the same PC_Event match on the step immediately following resume.
    pub(super) sync_trap_consumed_at: Option<u32>,

    /// Live count-step budget (Debug_Control0[5:2] Single_Step_Count).
    /// `Some(n)` = n committed bundles remain before a before-commit halt;
    /// `None` = count-step disabled. Distinct from the raw
    /// `single_step_count` config field: this is the decrementing live
    /// counter. G2 (NPU1 silicon, 2026-05-19): count-step is live hardware;
    /// `N=4` halts in the prologue before the first store. Modeling
    /// decisions (silicon-unobservable edges, spec §5.2, Maya 2026-05-19):
    /// N counts committed bundles, halt fires before the (N+1)th commits;
    /// count+halt-bit (0x11) -> bit[0] immediate-halt precedence, budget
    /// armed latent; expiry clears the budget, only a fresh Debug_Control0
    /// write re-arms (request_resume never re-arms).
    pub(super) count_step_remaining: Option<u32>,
    /// Latched cause: the core was halted by count-step budget expiry.
    pub(super) halt_cause_count_step: bool,
}

impl Default for CoreDebugState {
    /// Initial state matches hardware reset: core in reset, not enabled,
    /// PC/SP/LR all zero. Per aie-rt, CORE_CONTROL_RESET_DEFVAL = 0x1,
    /// CORE_CONTROL_ENABLE_DEFVAL = 0x0.
    fn default() -> Self {
        Self {
            enabled: false,
            reset: true,
            done: false,
            mem_stall: false,
            lock_stall: false,
            stream_stall: false,
            cascade_stall: false,
            halted: false,
            ecc_error: false,
            error_halt: false,
            single_step: false,
            single_step_count: 0,
            pc: 0,
            sp: 0,
            lr: 0,
            debug_ctrl1: 0,
            debug_ctrl2: 0,
            pc_event0: 0,
            pc_event1: 0,
            pc_event2: 0,
            pc_event3: 0,
            halt_cause_pc_event: false,
            halt_cause_mem_stall: false,
            halt_cause_lock_stall: false,
            halt_cause_stream_stall: false,
            halt_cause_event0: false,
            halt_cause_event1: false,
            pending_single_step: false,
            sync_trap_consumed_at: None,
            count_step_remaining: None,
            halt_cause_count_step: false,
        }
    }
}

impl CoreDebugState {
    /// Create a new core debug state in hardware-reset configuration.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Status register (read-only from host perspective)
    // -----------------------------------------------------------------------

    /// Pack the current state into the Core_Status register format.
    ///
    /// Bit layout per aie-rt `xaiemlgbl_params.h`:
    /// ```text
    /// [0]  Enable
    /// [1]  Reset
    /// [2]  Memory_Stall_S    [3] W   [4] N   [5] E
    /// [6]  Lock_Stall_S      [7] W   [8] N   [9] E
    /// [10] Stream_Stall_SS0  [12] Stream_Stall_MS0
    /// [14] Cascade_Stall_SCD [15] Cascade_Stall_MCD
    /// [16] Debug_Halt
    /// [17] ECC_Error_Stall
    /// [18] ECC_Scrubbing_Stall (not modeled)
    /// [19] Error_Halt
    /// [20] Core_Done
    /// [21] Processor_Bus_Stall (not modeled)
    /// ```
    ///
    /// Directional stalls: when a stall is active, we set all 4 directional
    /// bits for that stall type. The emulator does not track per-direction
    /// stall granularity -- this is a simplification that is conservative
    /// (any host software checking "any memory stall" will see it).
    pub fn read_status(&self) -> u32 {
        let mut val = 0u32;

        if self.enabled {
            val |= 1 << STATUS_ENABLE_LSB;
        }
        if self.reset {
            val |= 1 << STATUS_RESET_LSB;
        }

        // Memory stall: set all 4 directional bits.
        if self.mem_stall {
            val |= 1 << STATUS_MEMORY_STALL_S_LSB;
            val |= 1 << STATUS_MEMORY_STALL_W_LSB;
            val |= 1 << STATUS_MEMORY_STALL_N_LSB;
            val |= 1 << STATUS_MEMORY_STALL_E_LSB;
        }

        // Lock stall: set all 4 directional bits.
        if self.lock_stall {
            val |= 1 << STATUS_LOCK_STALL_S_LSB;
            val |= 1 << STATUS_LOCK_STALL_W_LSB;
            val |= 1 << STATUS_LOCK_STALL_N_LSB;
            val |= 1 << STATUS_LOCK_STALL_E_LSB;
        }

        // Stream stall: set both SS0 and MS0 bits.
        if self.stream_stall {
            val |= 1 << STATUS_STREAM_STALL_SS0_LSB;
            val |= 1 << STATUS_STREAM_STALL_MS0_LSB;
        }

        // Cascade stall: set both SCD and MCD bits.
        if self.cascade_stall {
            val |= 1 << STATUS_CASCADE_STALL_SCD_LSB;
            val |= 1 << STATUS_CASCADE_STALL_MCD_LSB;
        }

        if self.halted {
            val |= 1 << STATUS_DEBUG_HALT_LSB;
        }
        if self.ecc_error {
            val |= 1 << STATUS_ECC_ERROR_STALL_LSB;
            val |= 1 << STATUS_ERROR_HALT_LSB;
        }
        if self.error_halt {
            val |= 1 << STATUS_ERROR_HALT_LSB;
        }
        if self.done {
            val |= 1 << STATUS_DONE_LSB;
        }

        val
    }

    // -----------------------------------------------------------------------
    // Control register (write from host)
    // -----------------------------------------------------------------------

    /// Handle a write to the Core_Control register.
    ///
    /// Per aie-rt, this register has two bits:
    /// - Bit 0: enable (start core execution)
    /// - Bit 1: reset (hold core in reset)
    ///
    /// aie-rt uses `XAie_MaskWrite32` -- only bits within the mask are
    /// modified. We accept the full value and apply both bits.
    pub fn write_control(&mut self, value: u32) {
        self.enabled = (value & CTRL_ENABLE_MASK) != 0;
        self.reset = (value & CTRL_RESET_MASK) != 0;

        // Reset clears runtime state.
        if self.reset {
            self.done = false;
            self.halted = false;
            self.mem_stall = false;
            self.lock_stall = false;
            self.stream_stall = false;
            self.cascade_stall = false;
            self.ecc_error = false;
            self.error_halt = false;
            self.single_step = false;
            self.single_step_count = 0;
            self.pc = 0;
            self.sp = 0;
            self.lr = 0;
            self.pc_event0 = 0;
            self.pc_event1 = 0;
            self.pc_event2 = 0;
            self.pc_event3 = 0;
            self.pending_single_step = false;
            self.clear_halt_causes();
        }
    }

    /// Read the Core_Control register value.
    pub fn read_control(&self) -> u32 {
        let mut val = 0u32;
        if self.enabled {
            val |= CTRL_ENABLE_MASK;
        }
        if self.reset {
            val |= CTRL_RESET_MASK;
        }
        val
    }

    // -----------------------------------------------------------------------
    // Debug registers
    // -----------------------------------------------------------------------

    /// Read the program counter (20-bit).
    pub fn read_pc(&self) -> u32 {
        self.pc & ADDR_MASK_20BIT
    }

    /// Read the stack pointer (20-bit).
    pub fn read_sp(&self) -> u32 {
        self.sp & ADDR_MASK_20BIT
    }

    /// Read the link register (20-bit).
    pub fn read_lr(&self) -> u32 {
        self.lr & ADDR_MASK_20BIT
    }

    /// Update the program counter. Called by the interpreter after each
    /// instruction. Value is masked to 20 bits per hardware. Also drives
    /// PC_Event0..3 matching, which broadcasts Core_PC_n events through
    /// the standard event halt path and (when Debug_Control2.PC_Event_Halt
    /// is set) latches halt_cause_pc_event.
    pub fn update_pc(&mut self, pc: u32) {
        self.pc = pc & ADDR_MASK_20BIT;
        self.check_pc_events(pc);
    }

    /// Update the stack pointer. Value is masked to 20 bits per hardware.
    pub fn update_sp(&mut self, sp: u32) {
        self.sp = sp & ADDR_MASK_20BIT;
    }

    /// Update the link register. Value is masked to 20 bits per hardware.
    pub fn update_lr(&mut self, lr: u32) {
        self.lr = lr & ADDR_MASK_20BIT;
    }

    // -----------------------------------------------------------------------
    // Stall updates (called by interpreter each cycle)
    // -----------------------------------------------------------------------

    /// Update stall indicators from the interpreter.
    ///
    /// These are transient -- they reflect the current cycle's state and
    /// should be called every cycle (or at least before any status read).
    pub fn update_stalls(&mut self, mem: bool, lock: bool, stream: bool, cascade: bool) {
        self.mem_stall = mem;
        self.lock_stall = lock;
        self.stream_stall = stream;
        self.cascade_stall = cascade;
        // Re-evaluate stall-halt enables: a stall going hot with the
        // matching Debug_Control2 bit set must immediately trigger halt.
        self.check_stall_halt();
    }

    // -----------------------------------------------------------------------
    // Halt / Resume
    // -----------------------------------------------------------------------

    /// Request a debug halt. Returns true if the halt took effect
    /// (core was enabled and not already halted).
    ///
    /// Per aie-rt `_XAie_CoreDebugCtrlHalt`, this writes the debug halt
    /// bit in Debug_Control0.
    pub fn request_halt(&mut self) -> bool {
        if self.enabled && !self.halted {
            self.halted = true;
            true
        } else {
            false
        }
    }

    /// Request a debug resume. Returns true if the resume took effect
    /// (core was halted).
    ///
    /// Per aie-rt `XAie_CoreDebugUnhalt`, this clears the debug halt
    /// bit in Debug_Control0.
    pub fn request_resume(&mut self) -> bool {
        if self.halted {
            self.halted = false;
            true
        } else {
            false
        }
    }

    /// Check if the core is halted by debug request.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Check if the core is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if the core is in reset.
    pub fn is_reset(&self) -> bool {
        self.reset
    }

    /// Check if the core is done.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Set the done bit. Called when the core reaches end of program.
    pub fn set_done(&mut self, done: bool) {
        self.done = done;
    }

    /// Set the enable bit. Called when the core is started.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the ECC error state.
    pub fn set_ecc_error(&mut self, error: bool) {
        self.ecc_error = error;
    }

    /// Set the generic error_halt state (decode failure, unhandled
    /// instruction, missing program memory). Surfaces in Core_Status bit 19
    /// alongside `ecc_error`. The interpreter calls this when it transitions
    /// to `CoreStatus::Error`; reset_event clears it via `write_control`.
    pub fn set_error_halt(&mut self, error: bool) {
        self.error_halt = error;
    }

    // -----------------------------------------------------------------------
    // Event-driven halt (Debug_Control1 / Debug_Control2)
    // -----------------------------------------------------------------------

    /// Decoded Debug_Halt_Core_Event0 from Debug_Control1 [22:16].
    fn debug_halt_event0(&self) -> u8 {
        ((self.debug_ctrl1 >> DBG_CTRL1_HALT_EVENT0_LSB) & DBG_CTRL1_EVENT_MASK) as u8
    }

    /// Decoded Debug_Halt_Core_Event1 from Debug_Control1 [30:24].
    fn debug_halt_event1(&self) -> u8 {
        ((self.debug_ctrl1 >> DBG_CTRL1_HALT_EVENT1_LSB) & DBG_CTRL1_EVENT_MASK) as u8
    }

    /// Decoded Debug_Resume_Core_Event from Debug_Control1 [6:0].
    fn debug_resume_event(&self) -> u8 {
        ((self.debug_ctrl1 >> DBG_CTRL1_RESUME_EVENT_LSB) & DBG_CTRL1_EVENT_MASK) as u8
    }

    /// Decoded Debug_Single_Step_Event from Debug_Control1 [14:8].
    fn debug_sstep_event(&self) -> u8 {
        ((self.debug_ctrl1 >> DBG_CTRL1_SSTEP_EVENT_LSB) & DBG_CTRL1_EVENT_MASK) as u8
    }

    /// Notify the debug subsystem that an event has fired. If the event
    /// matches a configured Debug_Halt_Core_EventN, halts the core and
    /// latches the corresponding Debug_Status cause bit. If it matches
    /// Debug_Resume_Core_Event, resumes the core and clears all latched
    /// cause bits.
    ///
    /// AM025 Debug_Control1 encodes 7-bit event IDs; event 0 is the
    /// EVENT_NONE sentinel and never matches (mirrors aie-rt's event-0 =
    /// disabled convention). Tile-level dispatchers (notify_*_trace_event)
    /// already short-circuit on event_id 0, but we double-check here so
    /// direct callers can't accidentally trigger halts on unconfigured
    /// slots.
    pub fn check_event_halt(&mut self, event_id: u8) {
        if event_id == 0 {
            return;
        }
        let halt_e0 = self.debug_halt_event0();
        let halt_e1 = self.debug_halt_event1();
        let resume_e = self.debug_resume_event();
        let sstep_e = self.debug_sstep_event();
        if halt_e0 != 0 && event_id == halt_e0 {
            self.halt_cause_event0 = true;
            self.request_halt();
        }
        if halt_e1 != 0 && event_id == halt_e1 {
            self.halt_cause_event1 = true;
            self.request_halt();
        }
        if resume_e != 0 && event_id == resume_e {
            self.clear_halt_causes();
            self.pending_single_step = false;
            self.request_resume();
        }
        if sstep_e != 0 && event_id == sstep_e {
            self.pending_single_step = true;
        }
    }

    /// Drain the SSTEP_EVENT latch. Returns true (and requests halt) if a
    /// single-step event was queued during the just-completed step.
    /// Called by the coordinator after each core step. The latch clears
    /// on resume too, so a manual resume between event and consume cancels
    /// the pending single-step.
    ///
    /// Per AM025 there is no dedicated Debug_Status cause bit for
    /// single-step halts -- the aggregate `halted` bit is the only signal.
    pub fn consume_pending_single_step(&mut self) -> bool {
        if self.pending_single_step {
            self.pending_single_step = false;
            self.request_halt();
            true
        } else {
            false
        }
    }

    /// Decrement the count-step budget by one committed bundle. The
    /// coordinator calls this after each committed bundle, adjacent to
    /// `consume_pending_single_step`. On expiry it latches `halted` (via
    /// `request_halt`) and the count-step cause; the existing
    /// `interpreter.rs` `is_halted` gate then prevents the next bundle from
    /// committing — the before-commit-of-bundle-(N+1) boundary G2 derived
    /// (spec §5.2). Expiry clears the budget; only a fresh Debug_Control0
    /// write re-arms. Returns true iff this tick expired the budget.
    pub fn tick_count_step(&mut self) -> bool {
        match self.count_step_remaining {
            Some(n) if n > 1 => {
                self.count_step_remaining = Some(n - 1);
                false
            }
            Some(_) => {
                self.count_step_remaining = None;
                self.halt_cause_count_step = true;
                self.request_halt();
                true
            }
            None => false,
        }
    }

    /// Decode a PC_Event* raw register value to its address (or None if
    /// VALID=0). Per AM025 PC_Event* layout (bit 31 VALID, bits [13:0] address).
    fn pc_event_address(raw: u32) -> Option<u32> {
        if raw & PC_EVENT_VALID_MASK != 0 {
            Some(raw & PC_EVENT_ADDRESS_MASK)
        } else {
            None
        }
    }

    /// Returns true when Debug_Control2.PC_Event_Halt (bit 0) is set.
    fn pc_event_halt_enabled(&self) -> bool {
        (self.debug_ctrl2 >> DBG_CTRL2_PC_EVENT_HALT_LSB) & 1 != 0
    }

    // -----------------------------------------------------------------------
    // Synchronous PC_Event trap query (Phase B Unit 1, spec §5.1)
    // -----------------------------------------------------------------------

    /// Return true if the core has an armed synchronous PC_Event trap at `pc`
    /// that has not yet been consumed for this step.
    ///
    /// A trap is armed when:
    /// - At least one of PC_Event0..3 is VALID and its 14-bit PC_ADDRESS
    ///   matches the low 14 bits of `pc`, AND
    /// - Debug_Control2.PC_Event_Halt (bit 0) is set, AND
    /// - The trap has not already been consumed at this PC by a prior call to
    ///   `consume_sync_pc_trap()` (suppresses re-fire for one step after resume).
    ///
    /// This is a non-committing query: it does not halt the core or modify any
    /// state. The coordinator calls this before executing the bundle; if it
    /// returns true, the coordinator calls `consume_sync_pc_trap()` and returns
    /// `StepResult::DebugHalt` without executing the bundle.
    ///
    /// Per G1 silicon observation (NPU1, 2026-05-18): synchronous PC-event
    /// breakpoints halt BEFORE the trap bundle commits. See findings doc.
    pub fn has_sync_pc_trap_at(&self, pc: u32) -> bool {
        if !self.pc_event_halt_enabled() {
            return false;
        }
        // Suppress re-fire for exactly one step after resume from a
        // pre-execute halt at this same PC.
        if self.sync_trap_consumed_at == Some(pc & PC_EVENT_ADDRESS_MASK) {
            return false;
        }
        let pc14 = pc & PC_EVENT_ADDRESS_MASK;
        let matches_event =
            |raw: u32| -> bool { Self::pc_event_address(raw).map_or(false, |addr| addr == pc14) };
        matches_event(self.pc_event0)
            || matches_event(self.pc_event1)
            || matches_event(self.pc_event2)
            || matches_event(self.pc_event3)
    }

    /// Consume the synchronous PC_Event trap at `pc`: halt the core and record
    /// the consumed PC so the same trap does not re-fire immediately on resume.
    ///
    /// Called by the coordinator when `has_sync_pc_trap_at(pc)` returns true,
    /// before the bundle executes. Sets `halt_cause_pc_event` (matching HW
    /// Debug_Status behavior) and requests halt.
    pub fn consume_sync_pc_trap(&mut self, pc: u32) {
        self.sync_trap_consumed_at = Some(pc & PC_EVENT_ADDRESS_MASK);
        self.halt_cause_pc_event = true;
        self.request_halt();
    }

    /// True iff an *event-driven* single-step (Debug_Control1[14:8]
    /// SSTEP_EVENT) is wired to a *point* PC event (Core_PC_0..3) that is
    /// VALID and matches `pc`, and this PC has not already been consumed.
    /// This is the before-commit-eligible single-step case (§5.1 principled
    /// split, Maya 2026-05-19): the arming condition (PC match) is known
    /// *before* the bundle, so silicon halts before the bundle commits — the
    /// same boundary as the G1 PC_Event_Halt seam. Watchpoint/mem/lock and
    /// PC-*range*-wired SSTEP_EVENT have no coherent before-commit point and
    /// stay after-commit via the unchanged check_event_halt ->
    /// pending_single_step -> consume_pending_single_step path (documented
    /// modeling decision).
    pub fn has_sync_sstep_pc_trap_at(&self, pc: u32) -> bool {
        let pc14 = pc & PC_EVENT_ADDRESS_MASK;
        if self.sync_trap_consumed_at == Some(pc14) {
            return false;
        }
        let raw = match self.debug_sstep_event() {
            EVENT_CORE_PC_0 => self.pc_event0,
            EVENT_CORE_PC_1 => self.pc_event1,
            EVENT_CORE_PC_2 => self.pc_event2,
            EVENT_CORE_PC_3 => self.pc_event3,
            _ => return false,
        };
        Self::pc_event_address(raw).map_or(false, |addr| addr == pc14)
    }

    /// Consume a before-commit PC-wired single-step trap: latch the PC-event
    /// halt cause (Debug_Status has no dedicated single-step cause bit —
    /// aggregate only; a PC-wired single-step *is* a PC event firing), mark
    /// this PC consumed so it does not re-fire after resume (mirrors
    /// `consume_sync_pc_trap`; re-arming is the §8-tracked edge), and request
    /// the halt.
    pub fn consume_sync_sstep_pc_trap(&mut self, pc: u32) {
        self.sync_trap_consumed_at = Some(pc & PC_EVENT_ADDRESS_MASK);
        self.halt_cause_pc_event = true;
        self.request_halt();
    }

    /// Clear the sync-trap-consumed record after the trap bundle has *retired*.
    ///
    /// Called by the coordinator ONLY on `StepResult::Continue` (the trap
    /// bundle ran and PC moved off TRAP_PC), so the next time the core
    /// returns to that PC the trap fires again. It must NOT be cleared on
    /// `DebugHalt`/`WaitLock`: those can leave PC pinned at TRAP_PC with the
    /// bundle un-retired, and clearing the latch there re-arms the
    /// pre-execute seam and swallows the host resume (review S1).
    pub fn clear_sync_trap_consumed(&mut self) {
        self.sync_trap_consumed_at = None;
    }

    /// Drive PC_Event0..3 matching against the new PC value.
    ///
    /// Per AM025 / xaie_events_aieml.h:
    /// - PC_Event0..3 each broadcast Core_PC_0..3 (event IDs 16..19) when
    ///   VALID=1 and the 14-bit PC_ADDRESS field equals the low 14 bits of
    ///   the current PC.
    /// - PC_Range_0_1 (event 20) fires while PC is within [PC_Event0,
    ///   PC_Event1] (both must be VALID; the lower address is treated as
    ///   the range start regardless of which slot holds it).
    /// - PC_Range_2_3 (event 21) is the symmetric case for slots 2/3.
    /// - When Debug_Control2.PC_Event_Halt is set and any of the above
    ///   fires, the core is halted and halt_cause_pc_event is latched.
    /// - Independent of the gate, each fired event is also routed through
    ///   `check_event_halt`, so Debug_Control1.HaltEvent0/1 wired to one of
    ///   these IDs will halt with the matching Event0/Event1 cause bit.
    fn check_pc_events(&mut self, pc: u32) {
        let pc14 = pc & PC_EVENT_ADDRESS_MASK;
        let pc0 = Self::pc_event_address(self.pc_event0);
        let pc1 = Self::pc_event_address(self.pc_event1);
        let pc2 = Self::pc_event_address(self.pc_event2);
        let pc3 = Self::pc_event_address(self.pc_event3);

        let mut any_pc_event_fired = false;
        if pc0 == Some(pc14) {
            any_pc_event_fired = true;
            self.check_event_halt(EVENT_CORE_PC_0);
        }
        if pc1 == Some(pc14) {
            any_pc_event_fired = true;
            self.check_event_halt(EVENT_CORE_PC_1);
        }
        if pc2 == Some(pc14) {
            any_pc_event_fired = true;
            self.check_event_halt(EVENT_CORE_PC_2);
        }
        if pc3 == Some(pc14) {
            any_pc_event_fired = true;
            self.check_event_halt(EVENT_CORE_PC_3);
        }
        if let (Some(a), Some(b)) = (pc0, pc1) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            if pc14 >= lo && pc14 <= hi {
                any_pc_event_fired = true;
                self.check_event_halt(EVENT_CORE_PC_RANGE_0_1);
            }
        }
        if let (Some(a), Some(b)) = (pc2, pc3) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            if pc14 >= lo && pc14 <= hi {
                any_pc_event_fired = true;
                self.check_event_halt(EVENT_CORE_PC_RANGE_2_3);
            }
        }

        if any_pc_event_fired && self.pc_event_halt_enabled() {
            self.halt_cause_pc_event = true;
            self.request_halt();
        }
    }

    /// Consult Debug_Control2 stall-halt enables and trigger halt if any
    /// enabled stall condition is currently active. Called from
    /// `update_stalls` so every stall-state update gets re-evaluated.
    fn check_stall_halt(&mut self) {
        let ctrl2 = self.debug_ctrl2;
        if (ctrl2 >> DBG_CTRL2_MEM_STALL_HALT_LSB) & 1 != 0 && self.mem_stall {
            self.halt_cause_mem_stall = true;
            self.request_halt();
        }
        if (ctrl2 >> DBG_CTRL2_LOCK_STALL_HALT_LSB) & 1 != 0 && self.lock_stall {
            self.halt_cause_lock_stall = true;
            self.request_halt();
        }
        if (ctrl2 >> DBG_CTRL2_STREAM_STALL_HALT_LSB) & 1 != 0 && self.stream_stall {
            self.halt_cause_stream_stall = true;
            self.request_halt();
        }
    }

    /// Clear all halt-cause latches (called on resume).
    fn clear_halt_causes(&mut self) {
        self.halt_cause_pc_event = false;
        self.halt_cause_mem_stall = false;
        self.halt_cause_lock_stall = false;
        self.halt_cause_stream_stall = false;
        self.halt_cause_event0 = false;
        self.halt_cause_event1 = false;
        self.halt_cause_count_step = false;
    }

    // -----------------------------------------------------------------------
    // Debug_Control0 register interface
    // -----------------------------------------------------------------------

    /// Write Debug_Control0. Handles halt bit and single-step count.
    pub(super) fn write_debug_control0(&mut self, value: u32) {
        let halt_req = (value & DBG_CTRL0_HALT_MASK) != 0;
        if halt_req {
            self.request_halt();
        } else {
            self.request_resume();
        }

        let sstep_count = ((value & DBG_CTRL0_SSTEP_COUNT_MASK) >> DBG_CTRL0_SSTEP_COUNT_LSB) as u8;
        self.single_step_count = sstep_count;
        self.single_step = sstep_count > 0;

        // §5.2 count-step arm (G2 silicon-derived, 2026-05-19). A non-zero
        // Single_Step_Count arms a live N-committed-bundle budget; N=0 disables.
        // Independent of the halt bit: for 0x11 the bit[0] immediate halt above
        // takes precedence; the budget is still armed (latent) and applies if
        // the core later resumes (spec §5.2 modeling decision).
        self.count_step_remaining = if sstep_count > 0 {
            Some(sstep_count as u32)
        } else {
            None
        };
    }

    /// Read Debug_Control0.
    pub(super) fn read_debug_control0(&self) -> u32 {
        let mut val = 0u32;
        if self.halted {
            val |= DBG_CTRL0_HALT_MASK;
        }
        val |= (self.single_step_count as u32) << DBG_CTRL0_SSTEP_COUNT_LSB;
        val
    }

    // -----------------------------------------------------------------------
    // Debug_Status register (read-only)
    // -----------------------------------------------------------------------

    /// Read Debug_Status. Bit 0 is the aggregate "any halt" indicator;
    /// bits 1-6 are per-cause latches (PC event, mem/lock/stream stall,
    /// Event0, Event1) that survive until the next resume clears them.
    /// Per AM025 Debug_Status field layout.
    pub(super) fn read_debug_status(&self) -> u32 {
        let mut val = 0u32;
        if self.halted {
            val |= 1 << DBG_STS_HALTED_LSB;
        }
        if self.halt_cause_pc_event {
            val |= 1 << DBG_STS_PC_EVENT_HALTED_LSB;
        }
        if self.halt_cause_mem_stall {
            val |= 1 << DBG_STS_MEM_STALL_HALTED_LSB;
        }
        if self.halt_cause_lock_stall {
            val |= 1 << DBG_STS_LOCK_STALL_HALTED_LSB;
        }
        if self.halt_cause_stream_stall {
            val |= 1 << DBG_STS_STREAM_STALL_HALTED_LSB;
        }
        if self.halt_cause_event0 {
            val |= 1 << DBG_STS_EVENT0_HALTED_LSB;
        }
        if self.halt_cause_event1 {
            val |= 1 << DBG_STS_EVENT1_HALTED_LSB;
        }
        val
    }

    // -----------------------------------------------------------------------
    // Register-space interface
    // -----------------------------------------------------------------------

    /// Read a register by its tile-relative offset. Returns `Some(value)`
    /// if the offset matches a known register, `None` otherwise.
    ///
    /// This provides the register-space interface for the tile's MMIO
    /// routing to call into.
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        match offset {
            REG_CORE_CONTROL => Some(self.read_control()),
            REG_CORE_STATUS => Some(self.read_status()),
            REG_DEBUG_CONTROL0 => Some(self.read_debug_control0()),
            REG_DEBUG_CONTROL1 => Some(self.debug_ctrl1),
            REG_DEBUG_CONTROL2 => Some(self.debug_ctrl2),
            REG_DEBUG_STATUS => Some(self.read_debug_status()),
            REG_PC_EVENT0 => Some(self.pc_event0),
            REG_PC_EVENT1 => Some(self.pc_event1),
            REG_PC_EVENT2 => Some(self.pc_event2),
            REG_PC_EVENT3 => Some(self.pc_event3),
            REG_CORE_PC => Some(self.read_pc()),
            REG_CORE_SP => Some(self.read_sp()),
            REG_CORE_LR => Some(self.read_lr()),
            _ => None,
        }
    }

    /// Write a register by its tile-relative offset. Returns `true` if the
    /// offset matched a known writable register, `false` otherwise.
    ///
    /// Read-only registers (Core_Status, Debug_Status, Core_PC, Core_SP,
    /// Core_LR) are silently ignored per hardware behavior (writes to
    /// read-only registers are dropped).
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        match offset {
            REG_CORE_CONTROL => {
                self.write_control(value);
                true
            }
            REG_DEBUG_CONTROL0 => {
                self.write_debug_control0(value);
                true
            }
            REG_DEBUG_CONTROL1 => {
                self.debug_ctrl1 = value;
                true
            }
            REG_DEBUG_CONTROL2 => {
                self.debug_ctrl2 = value;
                true
            }
            REG_PC_EVENT0 => {
                self.pc_event0 = value & PC_EVENT_WRITE_MASK;
                true
            }
            REG_PC_EVENT1 => {
                self.pc_event1 = value & PC_EVENT_WRITE_MASK;
                true
            }
            REG_PC_EVENT2 => {
                self.pc_event2 = value & PC_EVENT_WRITE_MASK;
                true
            }
            REG_PC_EVENT3 => {
                self.pc_event3 = value & PC_EVENT_WRITE_MASK;
                true
            }
            // Read-only registers: accept the write (return true) but drop it.
            REG_CORE_STATUS | REG_DEBUG_STATUS | REG_CORE_PC | REG_CORE_SP | REG_CORE_LR => true,
            _ => false,
        }
    }
}
