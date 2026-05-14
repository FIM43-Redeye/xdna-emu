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
/// Core_PC register offset within a tile.
const REG_CORE_PC: u32 = 0x0003_1100;
/// Core_SP register offset within a tile.
const REG_CORE_SP: u32 = 0x0003_1120;
/// Core_LR register offset within a tile.
const REG_CORE_LR: u32 = 0x0003_1130;

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
// Debug_Status bit positions
// ---------------------------------------------------------------------------

/// Bit 0: halted by debug halt request.
const DBG_STS_HALTED_LSB: u32 = 0;

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
    /// instruction. Value is masked to 20 bits per hardware.
    pub fn update_pc(&mut self, pc: u32) {
        self.pc = pc & ADDR_MASK_20BIT;
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

    /// Read Debug_Status. Shows which halt cause is active.
    pub(super) fn read_debug_status(&self) -> u32 {
        let mut val = 0u32;
        if self.halted {
            val |= 1 << DBG_STS_HALTED_LSB;
        }
        // Additional halt causes would be set here if we track them.
        // For now, only the direct debug halt is modeled.
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
            // Read-only registers: accept the write (return true) but drop it.
            REG_CORE_STATUS | REG_DEBUG_STATUS | REG_CORE_PC | REG_CORE_SP | REG_CORE_LR => true,
            _ => false,
        }
    }
}
