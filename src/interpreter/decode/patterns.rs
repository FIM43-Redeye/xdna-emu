//! Pattern-based instruction decoder.
//!
//! This decoder uses pattern matching on instruction words to decode
//! AIE2 instructions. It's designed to handle common patterns accurately
//! while providing graceful fallback for unrecognized encodings.
//!
//! # Pattern Recognition
//!
//! The decoder uses a hierarchical approach:
//! 1. Check for known NOP patterns
//! 2. Extract high nibble for coarse classification
//! 3. Apply slot-specific decoding rules
//! 4. Fall back to Unknown for unrecognized patterns

use super::{extract_imm, extract_reg, opcodes, Aie2Slot, SlotDecode};
use crate::interpreter::bundle::{
    BranchCondition, BundleFormat, ElementType, MemWidth, Operand, Operation, PostModify,
    ShufflePattern, SlotIndex, SlotOp, VliwBundle,
};
use crate::interpreter::traits::{DecodeError, Decoder};

/// Pattern-based decoder for AIE2 instructions.
///
/// This decoder recognizes common instruction patterns and produces
/// accurate decodes for typical programs. Unrecognized patterns are
/// decoded as `Operation::Unknown` to allow continued execution.
#[derive(Debug, Clone, Default)]
pub struct PatternDecoder {
    /// Statistics for unrecognized patterns (for debugging).
    unknown_count: u64,
}

impl PatternDecoder {
    /// Create a new pattern decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the count of unknown patterns encountered.
    pub fn unknown_count(&self) -> u64 {
        self.unknown_count
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.unknown_count = 0;
    }

    /// Decode the first instruction word.
    fn decode_word0(&self, word0: u32, bytes: &[u8], pc: u32) -> Result<VliwBundle, DecodeError> {
        let mut bundle = VliwBundle::from_raw(bytes, pc);

        // Check for NOP patterns first
        if word0 == opcodes::NOP_ZERO || word0 == opcodes::NOP_AIE {
            bundle.set_slot(SlotOp::nop(SlotIndex::Scalar0));
            return Ok(bundle);
        }

        // Extract high nibble for classification
        let high_nibble = (word0 >> 28) & 0xF;

        let slot_decode = match high_nibble {
            opcodes::high_nibble::BRANCH => self.decode_branch(word0),
            opcodes::high_nibble::CALL => self.decode_call_return(word0),
            opcodes::high_nibble::LOAD => self.decode_load(word0),
            opcodes::high_nibble::STORE => self.decode_store(word0),
            opcodes::high_nibble::ARITH_0
            | opcodes::high_nibble::ARITH_1
            | opcodes::high_nibble::ARITH_2
            | opcodes::high_nibble::ARITH_3 => self.decode_arith(word0, high_nibble),
            opcodes::high_nibble::MOVE => self.decode_move(word0),
            opcodes::high_nibble::LOCK => self.decode_lock(word0),
            opcodes::high_nibble::DMA => self.decode_dma(word0),
            opcodes::high_nibble::VEC_0
            | opcodes::high_nibble::VEC_1
            | opcodes::high_nibble::VEC_2
            | opcodes::high_nibble::VEC_3
            | opcodes::high_nibble::VEC_4 => self.decode_vector(word0, high_nibble, bytes),
            _ => SlotDecode {
                slot: Aie2Slot::Alu,
                op: Operation::Unknown { opcode: word0 },
                sources: vec![],
                dest: None,
            },
        };

        bundle.set_slot(slot_decode.to_slot_op());
        Ok(bundle)
    }

    /// Decode a branch instruction.
    fn decode_branch(&self, word0: u32) -> SlotDecode {
        // Branch target is typically in lower bits
        let target = word0 & 0x00FF_FFFF;

        // Condition is encoded in mid bits
        let cond_bits = (word0 >> 24) & 0xF;
        let condition = match cond_bits {
            0 => BranchCondition::Always,
            1 => BranchCondition::Equal,
            2 => BranchCondition::NotEqual,
            3 => BranchCondition::Less,
            4 => BranchCondition::GreaterEqual,
            5 => BranchCondition::LessEqual,
            6 => BranchCondition::Greater,
            7 => BranchCondition::Negative,
            8 => BranchCondition::PositiveOrZero,
            _ => BranchCondition::Always,
        };

        SlotDecode {
            slot: Aie2Slot::Alu,
            op: Operation::Branch { condition },
            sources: vec![Operand::Immediate(target as i32)],
            dest: None,
        }
    }

    /// Decode a call or return instruction.
    fn decode_call_return(&self, word0: u32) -> SlotDecode {
        // Check for return pattern
        if word0 & 0x00FF_0000 == 0x0000_0000 {
            return SlotDecode {
                slot: Aie2Slot::Alu,
                op: Operation::Return,
                sources: vec![],
                dest: None,
            };
        }

        let target = word0 & 0x00FF_FFFF;
        SlotDecode {
            slot: Aie2Slot::Alu,
            op: Operation::Call,
            sources: vec![Operand::Immediate(target as i32)],
            dest: None,
        }
    }

    /// Decode a load instruction.
    fn decode_load(&self, word0: u32) -> SlotDecode {
        let dst = extract_reg(word0, 16, 0x1F);
        let base = extract_reg(word0, 8, 0x7);
        let offset = extract_imm(word0, 0, 8);

        // Determine width from sub-opcode
        let width_bits = (word0 >> 21) & 0x7;
        let width = match width_bits {
            0 => MemWidth::Byte,
            1 => MemWidth::HalfWord,
            2 => MemWidth::Word,
            3 => MemWidth::DoubleWord,
            _ => MemWidth::Word,
        };

        SlotDecode {
            slot: Aie2Slot::Lda,
            op: Operation::Load {
                width,
                post_modify: PostModify::None,
            },
            sources: vec![Operand::Memory {
                base,
                offset: offset as i16,
            }],
            dest: Some(Operand::ScalarReg(dst)),
        }
    }

    /// Decode a store instruction.
    fn decode_store(&self, word0: u32) -> SlotDecode {
        let src = extract_reg(word0, 16, 0x1F);
        let base = extract_reg(word0, 8, 0x7);
        let offset = extract_imm(word0, 0, 8);

        let width_bits = (word0 >> 21) & 0x7;
        let width = match width_bits {
            0 => MemWidth::Byte,
            1 => MemWidth::HalfWord,
            2 => MemWidth::Word,
            3 => MemWidth::DoubleWord,
            _ => MemWidth::Word,
        };

        SlotDecode {
            slot: Aie2Slot::St,
            op: Operation::Store {
                width,
                post_modify: PostModify::None,
            },
            sources: vec![
                Operand::ScalarReg(src),
                Operand::Memory {
                    base,
                    offset: offset as i16,
                },
            ],
            dest: None,
        }
    }

    /// Decode an arithmetic instruction.
    fn decode_arith(&self, word0: u32, high_nibble: u32) -> SlotDecode {
        let dst = extract_reg(word0, 16, 0xF);
        let src1 = extract_reg(word0, 12, 0xF);
        let src2 = extract_reg(word0, 8, 0xF);

        // Operation type from mid nibble
        let op_bits = (word0 >> 24) & 0xF;

        let op = match (high_nibble, op_bits) {
            (0x4, 0) | (0x4, _) => Operation::ScalarAdd,
            (0x5, 0) | (0x5, _) => Operation::ScalarSub,
            (0x6, 0) | (0x6, _) => Operation::ScalarMul,
            (0x7, 0) => Operation::ScalarAnd,
            (0x7, 1) => Operation::ScalarOr,
            (0x7, 2) => Operation::ScalarXor,
            (0x7, 3) => Operation::ScalarShl,
            (0x7, 4) => Operation::ScalarShr,
            (0x7, 5) => Operation::ScalarSra,
            _ => Operation::ScalarAdd,
        };

        SlotDecode {
            slot: Aie2Slot::Alu,
            op,
            sources: vec![Operand::ScalarReg(src1), Operand::ScalarReg(src2)],
            dest: Some(Operand::ScalarReg(dst)),
        }
    }

    /// Decode a move instruction.
    fn decode_move(&self, word0: u32) -> SlotDecode {
        let dst = extract_reg(word0, 16, 0x1F);
        let src = extract_reg(word0, 8, 0x1F);

        // Check for immediate move
        let is_imm = (word0 >> 23) & 1 != 0;

        if is_imm {
            let imm = extract_imm(word0, 0, 16);
            SlotDecode {
                slot: Aie2Slot::Mv,
                op: Operation::ScalarMovi { value: imm },
                sources: vec![],
                dest: Some(Operand::ScalarReg(dst)),
            }
        } else {
            SlotDecode {
                slot: Aie2Slot::Mv,
                op: Operation::ScalarMov,
                sources: vec![Operand::ScalarReg(src)],
                dest: Some(Operand::ScalarReg(dst)),
            }
        }
    }

    /// Decode a lock operation.
    fn decode_lock(&self, word0: u32) -> SlotDecode {
        let is_release = (word0 >> 24) & 1 != 0;
        let lock_id = extract_reg(word0, 8, 0x3F);
        let _value = extract_reg(word0, 0, 0xFF);

        let op = if is_release {
            Operation::LockRelease
        } else {
            Operation::LockAcquire
        };

        SlotDecode {
            slot: Aie2Slot::Alu,
            op,
            sources: vec![Operand::Lock(lock_id)],
            dest: None,
        }
    }

    /// Decode a DMA operation.
    fn decode_dma(&self, word0: u32) -> SlotDecode {
        let channel = extract_reg(word0, 16, 0x3);
        let bd = extract_reg(word0, 8, 0xF);
        let is_start = word0 & 1 != 0;

        let op = if is_start {
            Operation::DmaStart
        } else {
            Operation::DmaWait
        };

        SlotDecode {
            slot: Aie2Slot::Alu,
            op,
            sources: vec![
                Operand::DmaChannel(channel),
                Operand::BufferDescriptor(bd),
            ],
            dest: None,
        }
    }

    /// Decode a vector operation.
    fn decode_vector(&self, word0: u32, high_nibble: u32, bytes: &[u8]) -> SlotDecode {
        // Check for VLIW format (larger instruction)
        let is_vliw = bytes.len() >= 16 && (word0 & 0x8000_0000) != 0 && high_nibble >= 0xB;

        // Operation type from sub-opcode
        let op_bits = (word0 >> 24) & 0xF;

        // Default element type (will be refined with TableGen)
        let element_type = match (word0 >> 20) & 0x3 {
            0 => ElementType::Int8,
            1 => ElementType::Int16,
            2 => ElementType::Int32,
            _ => ElementType::Int32,
        };

        let op = match op_bits {
            0 => Operation::VectorMul { element_type },
            1 => Operation::VectorAdd { element_type },
            2 => Operation::VectorMac { element_type },
            3 => Operation::Load {
                width: MemWidth::Vector256,
                post_modify: PostModify::None,
            },
            4 => Operation::Store {
                width: MemWidth::Vector256,
                post_modify: PostModify::None,
            },
            5 => Operation::VectorSub { element_type },
            6 => Operation::VectorMin { element_type },
            7 => Operation::VectorMax { element_type },
            8 => Operation::VectorShuffle {
                pattern: ShufflePattern::Identity,
            },
            _ => Operation::Unknown { opcode: word0 },
        };

        // For VLIW, we'd need to decode additional words
        // For now, mark as vector slot

        SlotDecode {
            slot: if is_vliw { Aie2Slot::Lng } else { Aie2Slot::Vec },
            op,
            sources: vec![],
            dest: None,
        }
    }
}

impl Decoder for PatternDecoder {
    fn decode(&self, bytes: &[u8], pc: u32) -> Result<VliwBundle, DecodeError> {
        if bytes.len() < 4 {
            return Err(DecodeError::Incomplete {
                needed: 4,
                have: bytes.len(),
            });
        }

        let word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        self.decode_word0(word0, bytes, pc)
    }

    fn instruction_size(&self, bytes: &[u8]) -> Result<u8, DecodeError> {
        if bytes.len() < 2 {
            return Err(DecodeError::Incomplete {
                needed: 2,
                have: bytes.len(),
            });
        }

        // Check for 2-byte NOP
        if bytes.len() >= 2 && bytes[0] == 0 && bytes[1] == 0 {
            // Could be 2-byte or 4-byte NOP, assume 4
            return Ok(4);
        }

        if bytes.len() < 4 {
            return Err(DecodeError::Incomplete {
                needed: 4,
                have: bytes.len(),
            });
        }

        let word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let high_nibble = (word0 >> 28) & 0xF;

        // Check for long format (6 bytes)
        if high_nibble >= 0xB && (word0 & 0x0080_0000) != 0 {
            return Ok(6);
        }

        // Check for VLIW format (16 bytes)
        if bytes.len() >= 16 && high_nibble >= 0xB && (word0 & 0x8000_0000) != 0 {
            return Ok(16);
        }

        // Default to 4 bytes
        Ok(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_nop_zero() {
        let decoder = PatternDecoder::new();
        let bytes = [0x00, 0x00, 0x00, 0x00];

        let bundle = decoder.decode(&bytes, 0).unwrap();

        assert!(bundle.is_nop());
        assert_eq!(bundle.size(), 4);
    }

    #[test]
    fn test_decode_nop_aie() {
        let decoder = PatternDecoder::new();
        let bytes = [0x40, 0x00, 0x01, 0x15]; // 0x15010040 little-endian

        let bundle = decoder.decode(&bytes, 0).unwrap();

        assert!(bundle.is_nop());
    }

    #[test]
    fn test_decode_incomplete() {
        let decoder = PatternDecoder::new();
        let bytes = [0x00, 0x00];

        let result = decoder.decode(&bytes, 0);

        assert!(matches!(result, Err(DecodeError::Incomplete { .. })));
    }

    #[test]
    fn test_decode_arith_add() {
        let decoder = PatternDecoder::new();
        // Construct ADD r0, r1, r2 pattern: 0x4X_XX_12_00
        let bytes = [0x00, 0x12, 0x00, 0x40];

        let bundle = decoder.decode(&bytes, 0).unwrap();

        let slot = bundle.slot(SlotIndex::Scalar0).unwrap();
        assert!(matches!(slot.op, Operation::ScalarAdd));
    }

    #[test]
    fn test_decode_branch() {
        let decoder = PatternDecoder::new();
        // Branch to 0x100
        let bytes = [0x00, 0x01, 0x00, 0x00]; // Target in lower bits

        let bundle = decoder.decode(&bytes, 0).unwrap();

        assert!(bundle.has_control_flow());
    }

    #[test]
    fn test_decode_load() {
        let decoder = PatternDecoder::new();
        // Load pattern: 0x2X...
        let bytes = [0x10, 0x02, 0x01, 0x20]; // Load with base r0, offset 0x10

        let bundle = decoder.decode(&bytes, 0).unwrap();

        let slot = bundle.slot(SlotIndex::Load).unwrap();
        assert!(matches!(slot.op, Operation::Load { .. }));
    }

    #[test]
    fn test_decode_lock_acquire() {
        let decoder = PatternDecoder::new();
        // Lock acquire: 0x90...
        let bytes = [0x01, 0x05, 0x00, 0x90]; // Lock 5, value 1

        let bundle = decoder.decode(&bytes, 0).unwrap();

        let slot = bundle.slot(SlotIndex::Scalar0).unwrap();
        assert!(matches!(slot.op, Operation::LockAcquire));
    }

    #[test]
    fn test_instruction_size_default() {
        let decoder = PatternDecoder::new();
        let bytes = [0x00, 0x00, 0x00, 0x40]; // Simple instruction

        assert_eq!(decoder.instruction_size(&bytes).unwrap(), 4);
    }

    #[test]
    fn test_instruction_size_incomplete() {
        let decoder = PatternDecoder::new();
        let bytes = [0x00];

        assert!(matches!(
            decoder.instruction_size(&bytes),
            Err(DecodeError::Incomplete { .. })
        ));
    }
}
