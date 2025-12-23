//! AIE2 instruction decoder.
//!
//! AIE2 uses a VLIW (Very Long Instruction Word) architecture with 128-bit
//! instruction words containing multiple slots for parallel operations.
//!
//! # Instruction Format
//!
//! Each 128-bit instruction can contain:
//! - Scalar operation (32-bit)
//! - Vector operation (64-bit)
//! - Memory operation (32-bit)
//!
//! # Simplified Decoder
//!
//! This decoder recognizes common patterns without fully decoding the
//! complex VLIW format. It's designed for visualization and debugging
//! rather than cycle-accurate simulation.

use thiserror::Error;

/// Instruction decode error.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Not enough bytes for instruction.
    #[error("Incomplete instruction: need at least 4 bytes")]
    Incomplete,

    /// Unknown instruction encoding.
    #[error("Unknown instruction encoding: 0x{0:08X}")]
    Unknown(u32),
}

/// Decoded instruction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionKind {
    /// No operation (may include delay slots).
    Nop,

    /// Move register to register.
    Move { dst: u8, src: u8 },

    /// Load from memory.
    Load { dst: u8, base: u8, offset: i16 },

    /// Store to memory.
    Store { src: u8, base: u8, offset: i16 },

    /// Arithmetic operation.
    Arith { op: ArithOp, dst: u8, src1: u8, src2: u8 },

    /// Branch instruction.
    Branch { target: u32, condition: BranchCond },

    /// Call subroutine.
    Call { target: u32 },

    /// Return from subroutine.
    Return,

    /// Acquire lock.
    LockAcquire { lock_id: u8, value: i8 },

    /// Release lock.
    LockRelease { lock_id: u8, value: i8 },

    /// Vector operation (simplified).
    Vector { op: VectorOp },

    /// DMA operation.
    Dma { channel: u8, bd: u8, start: bool },

    /// Unknown/unrecognized instruction.
    Unknown { raw: u32 },
}

/// Arithmetic operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// Branch condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCond {
    Always,
    Equal,
    NotEqual,
    LessThan,
    GreaterEqual,
}

/// Vector operation (simplified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorOp {
    Vmul,
    Vadd,
    Vmac,
    Vload,
    Vstore,
    Other,
}

/// A decoded instruction with metadata.
#[derive(Debug, Clone)]
pub struct Instruction {
    /// The instruction kind.
    pub kind: InstructionKind,
    /// Size in bytes (4, 8, or 16 for VLIW).
    pub size: u8,
    /// Raw bytes of the instruction.
    pub raw: Vec<u8>,
}

impl Instruction {
    /// Decode an instruction from bytes.
    ///
    /// Returns the decoded instruction and its size.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < 4 {
            return Err(DecodeError::Incomplete);
        }

        let word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // AIE2 instruction format detection based on opcode patterns.
        // This is a simplified decoder - real AIE2 has complex VLIW encoding.

        let (kind, size) = decode_instruction(word0, bytes);

        let raw = bytes[..size as usize].to_vec();

        Ok(Self { kind, size, raw })
    }

    /// Check if this is a control flow instruction.
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self.kind,
            InstructionKind::Branch { .. }
                | InstructionKind::Call { .. }
                | InstructionKind::Return
        )
    }

    /// Check if this is a memory operation.
    pub fn is_memory_op(&self) -> bool {
        matches!(
            self.kind,
            InstructionKind::Load { .. } | InstructionKind::Store { .. }
        )
    }

    /// Check if this is a lock operation.
    pub fn is_lock_op(&self) -> bool {
        matches!(
            self.kind,
            InstructionKind::LockAcquire { .. } | InstructionKind::LockRelease { .. }
        )
    }

    /// Get a human-readable disassembly string.
    pub fn disassemble(&self) -> String {
        match &self.kind {
            InstructionKind::Nop => "nop".to_string(),
            InstructionKind::Move { dst, src } => format!("mov r{}, r{}", dst, src),
            InstructionKind::Load { dst, base, offset } => {
                format!("ldr r{}, [r{}, #{}]", dst, base, offset)
            }
            InstructionKind::Store { src, base, offset } => {
                format!("str r{}, [r{}, #{}]", src, base, offset)
            }
            InstructionKind::Arith { op, dst, src1, src2 } => {
                let op_str = match op {
                    ArithOp::Add => "add",
                    ArithOp::Sub => "sub",
                    ArithOp::Mul => "mul",
                    ArithOp::And => "and",
                    ArithOp::Or => "orr",
                    ArithOp::Xor => "eor",
                    ArithOp::Shl => "lsl",
                    ArithOp::Shr => "lsr",
                };
                format!("{} r{}, r{}, r{}", op_str, dst, src1, src2)
            }
            InstructionKind::Branch { target, condition } => {
                let cond_str = match condition {
                    BranchCond::Always => "b",
                    BranchCond::Equal => "beq",
                    BranchCond::NotEqual => "bne",
                    BranchCond::LessThan => "blt",
                    BranchCond::GreaterEqual => "bge",
                };
                format!("{} 0x{:04X}", cond_str, target)
            }
            InstructionKind::Call { target } => format!("call 0x{:04X}", target),
            InstructionKind::Return => "ret".to_string(),
            InstructionKind::LockAcquire { lock_id, value } => {
                format!("lock.acquire {}, {}", lock_id, value)
            }
            InstructionKind::LockRelease { lock_id, value } => {
                format!("lock.release {}, {}", lock_id, value)
            }
            InstructionKind::Vector { op } => {
                let op_str = match op {
                    VectorOp::Vmul => "vmul",
                    VectorOp::Vadd => "vadd",
                    VectorOp::Vmac => "vmac",
                    VectorOp::Vload => "vld",
                    VectorOp::Vstore => "vst",
                    VectorOp::Other => "vop",
                };
                op_str.to_string()
            }
            InstructionKind::Dma { channel, bd, start } => {
                if *start {
                    format!("dma.start ch{}, bd{}", channel, bd)
                } else {
                    format!("dma.wait ch{}", channel)
                }
            }
            InstructionKind::Unknown { raw } => format!(".word 0x{:08X}", raw),
        }
    }
}

/// Decode an instruction word.
///
/// This is a simplified decoder that recognizes common patterns.
/// Real AIE2 decoding would need the full VLIW slot parsing.
fn decode_instruction(word0: u32, bytes: &[u8]) -> (InstructionKind, u8) {
    // Check for special instruction patterns

    // NOP pattern: 0x00000000 or 0x15010040 (common AIE NOP)
    if word0 == 0x0000_0000 || word0 == 0x1501_0040 {
        return (InstructionKind::Nop, 4);
    }

    // Extract opcode fields - AIE2 uses different encoding regions
    let opcode_high = (word0 >> 28) & 0xF;
    let opcode_mid = (word0 >> 24) & 0xF;
    let opcode_low = (word0 >> 20) & 0xF;

    match opcode_high {
        // Branch/Call patterns (0x0X, 0x1X often)
        0x0 | 0x1 => {
            if word0 & 0xFFFF_0000 == 0x0 {
                // Likely a return or short branch
                if word0 == 0 {
                    return (InstructionKind::Nop, 4);
                }
                return (
                    InstructionKind::Branch {
                        target: word0 & 0xFFFF,
                        condition: BranchCond::Always,
                    },
                    4,
                );
            }

            // Check for return pattern
            if word0 & 0xFF00_0000 == 0x1000_0000 {
                return (InstructionKind::Return, 4);
            }

            // Generic branch
            (
                InstructionKind::Branch {
                    target: word0 & 0x00FF_FFFF,
                    condition: BranchCond::Always,
                },
                4,
            )
        }

        // Load/Store patterns
        0x2 | 0x3 => {
            let is_store = opcode_mid & 1 != 0;
            let reg = ((word0 >> 16) & 0xF) as u8;
            let base = ((word0 >> 12) & 0xF) as u8;
            let offset = (word0 & 0xFFF) as i16;

            if is_store {
                (
                    InstructionKind::Store {
                        src: reg,
                        base,
                        offset,
                    },
                    4,
                )
            } else {
                (
                    InstructionKind::Load {
                        dst: reg,
                        base,
                        offset,
                    },
                    4,
                )
            }
        }

        // Arithmetic patterns
        0x4 | 0x5 | 0x6 | 0x7 => {
            let dst = ((word0 >> 16) & 0xF) as u8;
            let src1 = ((word0 >> 12) & 0xF) as u8;
            let src2 = ((word0 >> 8) & 0xF) as u8;

            let op = match opcode_mid {
                0 => ArithOp::Add,
                1 => ArithOp::Sub,
                2 => ArithOp::Mul,
                3 => ArithOp::And,
                4 => ArithOp::Or,
                5 => ArithOp::Xor,
                6 => ArithOp::Shl,
                7 => ArithOp::Shr,
                _ => ArithOp::Add,
            };

            (InstructionKind::Arith { op, dst, src1, src2 }, 4)
        }

        // Move patterns
        0x8 => {
            let dst = ((word0 >> 16) & 0xF) as u8;
            let src = ((word0 >> 12) & 0xF) as u8;
            (InstructionKind::Move { dst, src }, 4)
        }

        // Lock operations
        0x9 => {
            let is_release = opcode_mid & 1 != 0;
            let lock_id = ((word0 >> 8) & 0x3F) as u8;
            let value = (word0 & 0xFF) as i8;

            if is_release {
                (InstructionKind::LockRelease { lock_id, value }, 4)
            } else {
                (InstructionKind::LockAcquire { lock_id, value }, 4)
            }
        }

        // DMA operations
        0xA => {
            let channel = ((word0 >> 16) & 0x3) as u8;
            let bd = ((word0 >> 8) & 0xF) as u8;
            let start = word0 & 1 != 0;
            (InstructionKind::Dma { channel, bd, start }, 4)
        }

        // Vector operations (0xB-0xF typically)
        0xB | 0xC | 0xD | 0xE | 0xF => {
            // Check for VLIW format (larger instructions)
            let has_vliw = bytes.len() >= 16 && is_vliw_header(word0);

            let op = match opcode_mid {
                0 => VectorOp::Vmul,
                1 => VectorOp::Vadd,
                2 => VectorOp::Vmac,
                3 => VectorOp::Vload,
                4 => VectorOp::Vstore,
                _ => VectorOp::Other,
            };

            if has_vliw {
                (InstructionKind::Vector { op }, 16)
            } else {
                (InstructionKind::Vector { op }, 4)
            }
        }

        _ => (InstructionKind::Unknown { raw: word0 }, 4),
    }
}

/// Check if a word looks like a VLIW header.
fn is_vliw_header(word: u32) -> bool {
    // VLIW instructions typically have specific marker bits
    // This is a heuristic - real detection needs more context
    (word >> 24) >= 0xB0 && (word & 0x8000_0000) != 0
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.disassemble())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_nop() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        let inst = Instruction::decode(&bytes).unwrap();
        assert_eq!(inst.kind, InstructionKind::Nop);
        assert_eq!(inst.size, 4);
    }

    #[test]
    fn test_decode_aie_nop() {
        let bytes = [0x40, 0x00, 0x01, 0x15]; // 0x15010040 little-endian
        let inst = Instruction::decode(&bytes).unwrap();
        assert_eq!(inst.kind, InstructionKind::Nop);
    }

    #[test]
    fn test_decode_incomplete() {
        let bytes = [0x00, 0x00];
        assert!(Instruction::decode(&bytes).is_err());
    }

    #[test]
    fn test_disassemble() {
        let inst = Instruction {
            kind: InstructionKind::Arith {
                op: ArithOp::Add,
                dst: 0,
                src1: 1,
                src2: 2,
            },
            size: 4,
            raw: vec![0; 4],
        };
        assert_eq!(inst.disassemble(), "add r0, r1, r2");
    }

    #[test]
    fn test_control_flow_detection() {
        let branch = Instruction {
            kind: InstructionKind::Branch {
                target: 0x100,
                condition: BranchCond::Always,
            },
            size: 4,
            raw: vec![0; 4],
        };
        assert!(branch.is_control_flow());

        let add = Instruction {
            kind: InstructionKind::Arith {
                op: ArithOp::Add,
                dst: 0,
                src1: 1,
                src2: 2,
            },
            size: 4,
            raw: vec![0; 4],
        };
        assert!(!add.is_control_flow());
    }

    #[test]
    fn test_lock_detection() {
        let acq = Instruction {
            kind: InstructionKind::LockAcquire {
                lock_id: 5,
                value: 1,
            },
            size: 4,
            raw: vec![0; 4],
        };
        assert!(acq.is_lock_op());
    }
}
