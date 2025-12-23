//! Core traits for the interpreter.
//!
//! These traits define the abstraction boundaries that allow different
//! implementations to be swapped. For example:
//!
//! - `Decoder`: Can be a fast pattern-matcher or a full TableGen-derived decoder
//! - `Executor`: Can be a fast "instant" executor or a cycle-accurate pipeline model
//! - `StateAccess`: Allows different state representations (debugging, checkpointing)
//!
//! # Design Philosophy
//!
//! The traits are designed to be:
//! - **Testable**: Easy to mock for unit testing
//! - **Swappable**: Different implementations for different use cases
//! - **Minimal**: Only the essential operations, no implementation details

use thiserror::Error;

// Re-export the actual VliwBundle from bundle module
pub use super::bundle::VliwBundle;

// Re-export Tile from device module
pub use crate::device::tile::Tile;

// ExecutionContext is now provided by the state module.
// Re-export it here for backwards compatibility.
pub use super::state::ExecutionContext;

/// Condition flags for branching decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    /// Zero flag: set when result is zero.
    pub z: bool,
    /// Negative flag: set when result is negative (sign bit set).
    pub n: bool,
    /// Carry flag: set on unsigned overflow.
    pub c: bool,
    /// Overflow flag: set on signed overflow.
    pub v: bool,
}

impl Flags {
    /// Create flags from a 32-bit result value.
    #[inline]
    pub fn from_result(result: u32) -> Self {
        Self {
            z: result == 0,
            n: (result as i32) < 0,
            c: false,
            v: false,
        }
    }

    /// Create flags from an addition operation (with carry/overflow detection).
    #[inline]
    pub fn from_add(a: u32, b: u32, result: u32) -> Self {
        let a_sign = (a >> 31) != 0;
        let b_sign = (b >> 31) != 0;
        let r_sign = (result >> 31) != 0;

        Self {
            z: result == 0,
            n: r_sign,
            c: result < a, // Unsigned overflow
            v: (a_sign == b_sign) && (r_sign != a_sign), // Signed overflow
        }
    }

    /// Create flags from a subtraction operation.
    #[inline]
    pub fn from_sub(a: u32, b: u32, result: u32) -> Self {
        let a_sign = (a >> 31) != 0;
        let b_sign = (b >> 31) != 0;
        let r_sign = (result >> 31) != 0;

        Self {
            z: result == 0,
            n: r_sign,
            c: a >= b, // No borrow occurred
            v: (a_sign != b_sign) && (r_sign != a_sign), // Signed overflow
        }
    }
}

/// Result of executing a bundle or instruction.
#[derive(Debug, Clone)]
pub enum ExecuteResult {
    /// Continue to next instruction (PC + bundle.size).
    Continue,

    /// Branch to target address.
    Branch {
        /// Target program counter.
        target: u32,
    },

    /// Stall waiting on lock acquisition.
    WaitLock {
        /// Lock ID being waited on.
        lock_id: u8,
    },

    /// Stall waiting on DMA completion.
    WaitDma {
        /// DMA channel being waited on.
        channel: u8,
    },

    /// Core has halted (normal termination).
    Halt,

    /// Error during execution.
    Error {
        /// Human-readable error message.
        message: String,
    },
}

/// Errors that can occur during instruction decoding.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Not enough bytes to decode instruction.
    #[error("incomplete instruction: need {needed} bytes, have {have}")]
    Incomplete {
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        have: usize,
    },

    /// Unknown opcode encountered.
    #[error("unknown opcode 0x{opcode:08X} at PC 0x{pc:04X}")]
    UnknownOpcode {
        /// The opcode value.
        opcode: u32,
        /// Program counter where error occurred.
        pc: u32,
    },

    /// Invalid slot combination in VLIW bundle.
    #[error("invalid slot combination in bundle at PC 0x{pc:04X}")]
    InvalidSlotCombination {
        /// Program counter where error occurred.
        pc: u32,
    },

    /// Invalid register encoding.
    #[error("invalid register {reg} in instruction at PC 0x{pc:04X}")]
    InvalidRegister {
        /// The invalid register number.
        reg: u8,
        /// Program counter where error occurred.
        pc: u32,
    },
}

/// Trait for instruction decoding.
///
/// Implementations decode raw bytes into structured `VliwBundle` representations.
/// Different implementations can provide different trade-offs:
///
/// - `FastDecoder`: Quick pattern matching, may miss edge cases
/// - `TableGenDecoder`: Full accuracy from parsed TableGen definitions
///
/// # Example
///
/// ```ignore
/// let decoder = FastDecoder::new();
/// let bytes = &program_memory[pc..];
/// let bundle = decoder.decode(bytes, pc)?;
/// println!("Decoded {} byte instruction", bundle.size);
/// ```
pub trait Decoder: Send + Sync {
    /// Decode bytes at the given PC into a VLIW bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Slice of program memory starting at PC
    /// * `pc` - Current program counter (for error reporting)
    ///
    /// # Returns
    ///
    /// The decoded bundle or a decode error.
    fn decode(&self, bytes: &[u8], pc: u32) -> Result<VliwBundle, DecodeError>;

    /// Get the size of the next instruction without full decode.
    ///
    /// This is useful for fast PC advancement when you don't need
    /// the full decoded bundle.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Slice of program memory starting at PC
    ///
    /// # Returns
    ///
    /// The instruction size in bytes (4, 8, or 16).
    fn instruction_size(&self, bytes: &[u8]) -> Result<u8, DecodeError>;
}

/// Trait for executing decoded bundles.
///
/// Implementations can vary in accuracy and performance:
///
/// - `FastExecutor`: Executes all slots "instantly", no pipeline modeling
/// - `CycleAccurateExecutor`: Models pipeline stages and hazards
///
/// # Example
///
/// ```ignore
/// let mut executor = FastExecutor::new();
/// let result = executor.execute(&bundle, &mut ctx, &mut tile);
/// match result {
///     ExecuteResult::Continue => ctx.pc += bundle.size as u32,
///     ExecuteResult::Branch { target } => ctx.pc = target,
///     _ => {}
/// }
/// ```
pub trait Executor: Send {
    /// Execute a single bundle on the given context.
    ///
    /// # Arguments
    ///
    /// * `bundle` - The decoded VLIW bundle to execute
    /// * `ctx` - Mutable execution context (registers, flags)
    /// * `tile` - Mutable tile state (memory, locks, DMA)
    ///
    /// # Returns
    ///
    /// The result of execution (continue, branch, wait, halt, or error).
    fn execute(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> ExecuteResult;

    /// Check if this executor models cycle-accurate behavior.
    ///
    /// Returns `true` if the executor models pipeline stages, hazards,
    /// and accurate timing. Returns `false` for fast/instant execution.
    fn is_cycle_accurate(&self) -> bool;
}

/// Trait for accessing processor state.
///
/// This abstraction allows different state representations:
/// - Direct access to register arrays
/// - Logged/traced access for debugging
/// - Checkpointed access for time-travel debugging
///
/// # Register Organization
///
/// AIE2 has several register files:
/// - Scalar GPR: 32 × 32-bit general purpose registers
/// - Pointer registers: 8 × 20-bit address registers
/// - Modifier registers: 8 × 20-bit for post-modify addressing
/// - Vector registers: 32 × 256-bit SIMD registers
/// - Accumulator registers: 8 × 512-bit for MAC operations
pub trait StateAccess {
    /// Read a scalar general-purpose register.
    fn read_scalar(&self, reg: u8) -> u32;

    /// Write a scalar general-purpose register.
    fn write_scalar(&mut self, reg: u8, value: u32);

    /// Read a vector register (as 8 × u32).
    fn read_vector(&self, reg: u8) -> [u32; 8];

    /// Write a vector register.
    fn write_vector(&mut self, reg: u8, value: [u32; 8]);

    /// Get the current program counter.
    fn pc(&self) -> u32;

    /// Set the program counter.
    fn set_pc(&mut self, pc: u32);

    /// Get the current condition flags.
    fn flags(&self) -> Flags;

    /// Set the condition flags.
    fn set_flags(&mut self, flags: Flags);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags_from_result() {
        // Zero
        let f = Flags::from_result(0);
        assert!(f.z);
        assert!(!f.n);

        // Positive
        let f = Flags::from_result(42);
        assert!(!f.z);
        assert!(!f.n);

        // Negative (sign bit set)
        let f = Flags::from_result(0x8000_0000);
        assert!(!f.z);
        assert!(f.n);
    }

    #[test]
    fn test_flags_from_add() {
        // Simple add, no overflow
        let f = Flags::from_add(10, 20, 30);
        assert!(!f.z);
        assert!(!f.n);
        assert!(!f.c);
        assert!(!f.v);

        // Result is zero
        let f = Flags::from_add(0, 0, 0);
        assert!(f.z);

        // Unsigned overflow (carry)
        let f = Flags::from_add(0xFFFF_FFFF, 2, 1);
        assert!(f.c);

        // Signed overflow
        let f = Flags::from_add(0x7FFF_FFFF, 1, 0x8000_0000);
        assert!(f.v);
    }

    #[test]
    fn test_flags_from_sub() {
        // Simple sub
        let f = Flags::from_sub(30, 20, 10);
        assert!(!f.z);
        assert!(!f.n);
        assert!(f.c); // No borrow

        // Result is zero
        let f = Flags::from_sub(10, 10, 0);
        assert!(f.z);

        // Borrow occurred
        let f = Flags::from_sub(10, 20, 0xFFFF_FFF6);
        assert!(!f.c);

        // Signed underflow
        let f = Flags::from_sub(0x8000_0000, 1, 0x7FFF_FFFF);
        assert!(f.v);
    }

    #[test]
    fn test_decode_error_display() {
        let e = DecodeError::Incomplete { needed: 4, have: 2 };
        assert!(e.to_string().contains("incomplete"));

        let e = DecodeError::UnknownOpcode { opcode: 0xDEAD, pc: 0x100 };
        assert!(e.to_string().contains("0000DEAD"));
    }

    #[test]
    fn test_execute_result_variants() {
        let r = ExecuteResult::Continue;
        assert!(matches!(r, ExecuteResult::Continue));

        let r = ExecuteResult::Branch { target: 0x100 };
        assert!(matches!(r, ExecuteResult::Branch { target: 0x100 }));

        let r = ExecuteResult::WaitLock { lock_id: 5 };
        assert!(matches!(r, ExecuteResult::WaitLock { lock_id: 5 }));
    }
}
