//! Register file implementations for AIE2.
//!
//! AIE2 has several register files:
//!
//! - **Scalar GPR**: 32 x 32-bit general purpose registers (r0-r31)
//! - **Special**: 16 x 32-bit special-purpose registers (lr, LS, LE, LC, etc.)
//! - **Pointer**: 8 x 20-bit address registers (p0-p7)
//! - **Modifier**: 8 x 20-bit post-modify registers (m0-m7)
//! - **Vector**: 32 x 256-bit SIMD registers (v0-v31)
//! - **Accumulator**: 8 x 512-bit MAC accumulators (acc0-acc7)
//!
//! The scalar register file is extended to 48 entries to include special-
//! purpose registers at indices 32-47. In AIE2 hardware these are separate
//! registers (not part of the GPR file), but for simplicity we store them
//! in the same array with dedicated index constants.

use std::fmt;

/// Number of scalar registers including special-purpose slots.
/// Indices 0-31 are GPRs (r0-r31); 32-47 are special registers.
pub const NUM_SCALAR_REGS: usize = 48;

/// Number of general-purpose scalar registers (r0-r31).
pub const NUM_SCALAR_GPRS: usize = 32;

// Special register indices (32+). These are NOT part of the r0-r31 GPR file
// in hardware; they are separate special-purpose registers. We map them to
// high indices in the scalar array for convenience.
/// Link register (lr) -- return address saved by jl/call.
pub const LR_REG_INDEX: u8 = 32;
/// Loop start address register (LS).
pub const LS_REG_INDEX: u8 = 33;
/// Loop end address register (LE).
pub const LE_REG_INDEX: u8 = 34;
/// Loop count register (LC).
pub const LC_REG_INDEX: u8 = 35;
/// Decompress pointer register (DP).
pub const DP_REG_INDEX: u8 = 36;
/// Core ID register (CORE_ID) -- read-only in hardware.
pub const CORE_ID_REG_INDEX: u8 = 37;

/// Number of pointer registers.
pub const NUM_POINTER_REGS: usize = 8;

/// Number of modifier registers.
pub const NUM_MODIFIER_REGS: usize = 8;

/// Number of vector registers.
pub const NUM_VECTOR_REGS: usize = 32;

/// Number of accumulator registers.
pub const NUM_ACCUMULATOR_REGS: usize = 8;

/// Mask for 20-bit pointer/modifier values.
const PTR_MASK: u32 = 0x000F_FFFF;

/// Scalar register file including special-purpose registers.
///
/// Indices 0-31: general-purpose registers r0-r31.
/// Indices 32-47: special-purpose registers (lr, LS, LE, LC, DP, CORE_ID).
///
/// In AIE2 hardware, the special registers are separate from the GPR file.
/// We store them in the same array for simplicity, using dedicated index
/// constants (LR_REG_INDEX, etc.) to access them.
#[derive(Clone)]
pub struct ScalarRegisterFile {
    regs: [u32; NUM_SCALAR_REGS],
}

impl Default for ScalarRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [0; NUM_SCALAR_REGS],
        }
    }

    /// Read a register. Indices 0-31 are GPRs; 32-47 are special registers.
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        let idx = (reg as usize) % NUM_SCALAR_REGS;
        self.regs[idx]
    }

    /// Write a register. Indices 0-31 are GPRs; 32-47 are special registers.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        let idx = (reg as usize) % NUM_SCALAR_REGS;
        self.regs[idx] = value;
    }

    /// Get a slice of GPR registers only (r0-r31) for debugging/display.
    pub fn as_slice(&self) -> &[u32] {
        &self.regs[..NUM_SCALAR_GPRS]
    }
}

impl fmt::Debug for ScalarRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only show non-zero registers
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .collect();

        if non_zero.is_empty() {
            write!(f, "ScalarRegisterFile {{ all zero }}")
        } else {
            write!(f, "ScalarRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "r{}: 0x{:08X}", reg, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// Pointer register file.
///
/// 8 × 20-bit address registers (p0-p7).
/// Used for memory addressing with optional post-modify.
#[derive(Clone)]
pub struct PointerRegisterFile {
    regs: [u32; NUM_POINTER_REGS],
}

impl Default for PointerRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl PointerRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [0; NUM_POINTER_REGS],
        }
    }

    /// Read a pointer register (0-7).
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        self.regs[(reg & 0x07) as usize]
    }

    /// Write a pointer register (0-7).
    /// Value is masked to 20 bits.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        self.regs[(reg & 0x07) as usize] = value & PTR_MASK;
    }

    /// Add offset to a pointer register (with wrapping at 20 bits).
    #[inline]
    pub fn add(&mut self, reg: u8, offset: i32) {
        let idx = (reg & 0x07) as usize;
        let new_value = self.regs[idx].wrapping_add(offset as u32) & PTR_MASK;
        self.regs[idx] = new_value;
    }
}

impl fmt::Debug for PointerRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .collect();

        if non_zero.is_empty() {
            write!(f, "PointerRegisterFile {{ all zero }}")
        } else {
            write!(f, "PointerRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "p{}: 0x{:05X}", reg, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// Modifier register file.
///
/// 8 × 20-bit registers (m0-m7) for post-modify addressing.
/// After a load/store, the modifier is added to the pointer.
#[derive(Clone)]
pub struct ModifierRegisterFile {
    regs: [u32; NUM_MODIFIER_REGS],
}

impl Default for ModifierRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl ModifierRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [0; NUM_MODIFIER_REGS],
        }
    }

    /// Read a modifier register (0-7).
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        self.regs[(reg & 0x07) as usize]
    }

    /// Read as signed value (sign-extended from 20 bits).
    #[inline]
    pub fn read_signed(&self, reg: u8) -> i32 {
        let val = self.read(reg);
        // Sign-extend from 20 bits
        if val & 0x0008_0000 != 0 {
            (val | 0xFFF0_0000) as i32
        } else {
            val as i32
        }
    }

    /// Write a modifier register (0-7).
    /// Value is masked to 20 bits.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        self.regs[(reg & 0x07) as usize] = value & PTR_MASK;
    }
}

impl fmt::Debug for ModifierRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .collect();

        if non_zero.is_empty() {
            write!(f, "ModifierRegisterFile {{ all zero }}")
        } else {
            write!(f, "ModifierRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "m{}: 0x{:05X}", reg, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// Vector register file.
///
/// 32 × 256-bit SIMD registers (v0-v31).
/// Each register holds 8 × 32-bit, 16 × 16-bit, or 32 × 8-bit elements.
#[derive(Clone)]
pub struct VectorRegisterFile {
    /// Each register is 256 bits = 8 × u32
    regs: [[u32; 8]; NUM_VECTOR_REGS],
}

impl Default for VectorRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [[0; 8]; NUM_VECTOR_REGS],
        }
    }

    /// Read a vector register as 8 × u32.
    #[inline]
    pub fn read(&self, reg: u8) -> [u32; 8] {
        self.regs[(reg & 0x1F) as usize]
    }

    /// Write a vector register from 8 × u32.
    #[inline]
    pub fn write(&mut self, reg: u8, value: [u32; 8]) {
        self.regs[(reg & 0x1F) as usize] = value;
    }

    /// Read a single lane (0-7) as u32.
    #[inline]
    pub fn read_lane(&self, reg: u8, lane: u8) -> u32 {
        self.regs[(reg & 0x1F) as usize][(lane & 0x07) as usize]
    }

    /// Write a single lane (0-7).
    #[inline]
    pub fn write_lane(&mut self, reg: u8, lane: u8, value: u32) {
        self.regs[(reg & 0x1F) as usize][(lane & 0x07) as usize] = value;
    }

    /// Read entire register as bytes (for memory operations).
    pub fn read_bytes(&self, reg: u8) -> [u8; 32] {
        let words = self.read(reg);
        let mut bytes = [0u8; 32];
        for (i, word) in words.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Write entire register from bytes.
    pub fn write_bytes(&mut self, reg: u8, bytes: &[u8; 32]) {
        let mut words = [0u32; 8];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        self.write(reg, words);
    }
}

impl fmt::Debug for VectorRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.iter().any(|x| *x != 0))
            .collect();

        if non_zero.is_empty() {
            write!(f, "VectorRegisterFile {{ all zero }}")
        } else {
            writeln!(f, "VectorRegisterFile {{")?;
            for (reg, val) in non_zero {
                write!(f, "  v{}: [", reg)?;
                for (i, lane) in val.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "0x{:08X}", lane)?;
                }
                writeln!(f, "]")?;
            }
            write!(f, "}}")
        }
    }
}

/// Accumulator register file.
///
/// 8 × 512-bit registers (acc0-acc7) for multiply-accumulate operations.
/// Each register holds 8 × 64-bit or 16 × 32-bit accumulators.
#[derive(Clone)]
pub struct AccumulatorRegisterFile {
    /// Each register is 512 bits = 8 × u64
    regs: [[u64; 8]; NUM_ACCUMULATOR_REGS],
}

impl Default for AccumulatorRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl AccumulatorRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [[0; 8]; NUM_ACCUMULATOR_REGS],
        }
    }

    /// Read an accumulator register as 8 × u64.
    #[inline]
    pub fn read(&self, reg: u8) -> [u64; 8] {
        self.regs[(reg & 0x07) as usize]
    }

    /// Write an accumulator register from 8 × u64.
    #[inline]
    pub fn write(&mut self, reg: u8, value: [u64; 8]) {
        self.regs[(reg & 0x07) as usize] = value;
    }

    /// Read a single lane (0-7) as u64.
    #[inline]
    pub fn read_lane(&self, reg: u8, lane: u8) -> u64 {
        self.regs[(reg & 0x07) as usize][(lane & 0x07) as usize]
    }

    /// Write a single lane (0-7).
    #[inline]
    pub fn write_lane(&mut self, reg: u8, lane: u8, value: u64) {
        self.regs[(reg & 0x07) as usize][(lane & 0x07) as usize] = value;
    }

    /// Clear (zero) an accumulator register.
    #[inline]
    pub fn clear(&mut self, reg: u8) {
        self.regs[(reg & 0x07) as usize] = [0; 8];
    }

    /// Accumulate: add values to existing accumulator lanes.
    pub fn accumulate(&mut self, reg: u8, values: [u64; 8]) {
        let idx = (reg & 0x07) as usize;
        for (lane, val) in values.iter().enumerate() {
            self.regs[idx][lane] = self.regs[idx][lane].wrapping_add(*val);
        }
    }
}

impl fmt::Debug for AccumulatorRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.iter().any(|x| *x != 0))
            .collect();

        if non_zero.is_empty() {
            write!(f, "AccumulatorRegisterFile {{ all zero }}")
        } else {
            writeln!(f, "AccumulatorRegisterFile {{")?;
            for (reg, val) in non_zero {
                write!(f, "  acc{}: [", reg)?;
                for (i, lane) in val.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "0x{:016X}", lane)?;
                }
                writeln!(f, "]")?;
            }
            write!(f, "}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Scalar Register Tests ==========

    #[test]
    fn test_scalar_read_write() {
        let mut regs = ScalarRegisterFile::new();

        regs.write(0, 0xDEAD_BEEF);
        assert_eq!(regs.read(0), 0xDEAD_BEEF);

        regs.write(31, 0xCAFE_BABE);
        assert_eq!(regs.read(31), 0xCAFE_BABE);
    }

    #[test]
    fn test_scalar_special_registers() {
        let mut regs = ScalarRegisterFile::new();

        // Index 32 is the LR register slot, NOT r0.
        regs.write(LR_REG_INDEX, 0x1234);
        assert_eq!(regs.read(LR_REG_INDEX), 0x1234);
        assert_eq!(regs.read(0), 0); // r0 unaffected

        // Other special registers are independent.
        regs.write(LC_REG_INDEX, 100);
        assert_eq!(regs.read(LC_REG_INDEX), 100);
        assert_eq!(regs.read(LR_REG_INDEX), 0x1234); // lr unaffected

        // Wrapping still applies at NUM_SCALAR_REGS (48).
        regs.write(48, 0xABCD); // wraps to index 0 (r0)
        assert_eq!(regs.read(0), 0xABCD);
    }

    #[test]
    fn test_scalar_debug_format() {
        let mut regs = ScalarRegisterFile::new();
        regs.write(5, 42);

        let debug = format!("{:?}", regs);
        assert!(debug.contains("r5"));
        assert!(debug.contains("0x0000002A"));
    }

    // ========== Pointer Register Tests ==========

    #[test]
    fn test_pointer_read_write() {
        let mut regs = PointerRegisterFile::new();

        regs.write(0, 0x1000);
        assert_eq!(regs.read(0), 0x1000);

        // Value masked to 20 bits
        regs.write(1, 0xFFFF_FFFF);
        assert_eq!(regs.read(1), 0x000F_FFFF);
    }

    #[test]
    fn test_pointer_add() {
        let mut regs = PointerRegisterFile::new();

        regs.write(0, 0x1000);
        regs.add(0, 0x100);
        assert_eq!(regs.read(0), 0x1100);

        // Negative offset
        regs.add(0, -0x50);
        assert_eq!(regs.read(0), 0x10B0);

        // Wrapping at 20 bits
        regs.write(1, 0x000F_FFF0);
        regs.add(1, 0x20);
        assert_eq!(regs.read(1), 0x0000_0010); // Wrapped
    }

    // ========== Modifier Register Tests ==========

    #[test]
    fn test_modifier_read_write() {
        let mut regs = ModifierRegisterFile::new();

        regs.write(0, 0x100);
        assert_eq!(regs.read(0), 0x100);
    }

    #[test]
    fn test_modifier_signed_read() {
        let mut regs = ModifierRegisterFile::new();

        // Positive value
        regs.write(0, 0x100);
        assert_eq!(regs.read_signed(0), 0x100);

        // Negative value (sign bit set in 20-bit)
        regs.write(1, 0x000F_FF00); // -256 in 20-bit signed
        assert_eq!(regs.read_signed(1), -256);
    }

    // ========== Vector Register Tests ==========

    #[test]
    fn test_vector_read_write() {
        let mut regs = VectorRegisterFile::new();

        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        assert_eq!(regs.read(0), data);
    }

    #[test]
    fn test_vector_lane_access() {
        let mut regs = VectorRegisterFile::new();

        regs.write_lane(5, 3, 0xDEAD_BEEF);
        assert_eq!(regs.read_lane(5, 3), 0xDEAD_BEEF);
        assert_eq!(regs.read_lane(5, 0), 0); // Other lanes unaffected
    }

    #[test]
    fn test_vector_byte_access() {
        let mut regs = VectorRegisterFile::new();

        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        regs.write_bytes(0, &bytes);

        let read_back = regs.read_bytes(0);
        assert_eq!(read_back, bytes);
    }

    // ========== Accumulator Register Tests ==========

    #[test]
    fn test_accumulator_read_write() {
        let mut regs = AccumulatorRegisterFile::new();

        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        assert_eq!(regs.read(0), data);
    }

    #[test]
    fn test_accumulator_clear() {
        let mut regs = AccumulatorRegisterFile::new();

        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        regs.clear(0);
        assert_eq!(regs.read(0), [0; 8]);
    }

    #[test]
    fn test_accumulator_accumulate() {
        let mut regs = AccumulatorRegisterFile::new();

        regs.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);
        regs.accumulate(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(regs.read(0), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_accumulator_lane_access() {
        let mut regs = AccumulatorRegisterFile::new();

        regs.write_lane(2, 5, 0xCAFE_BABE_DEAD_BEEF);
        assert_eq!(regs.read_lane(2, 5), 0xCAFE_BABE_DEAD_BEEF);
    }
}
