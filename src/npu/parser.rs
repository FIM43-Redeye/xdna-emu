//! NPU instruction stream parser.
//!
//! Parses the binary format used by mlir-aie for NPU instructions.

use super::NpuOpcode;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

/// A parsed NPU instruction.
#[derive(Debug, Clone)]
pub enum NpuInstruction {
    /// Write a single 32-bit value to a register.
    Write32 {
        /// Register offset (absolute address in NPU address space).
        reg_off: u32,
        /// Value to write.
        value: u32,
    },

    /// Write a block of 32-bit values starting at an address.
    BlockWrite {
        /// Starting register offset.
        reg_off: u32,
        /// Values to write (each written to consecutive 4-byte addresses).
        values: Vec<u32>,
    },

    /// Write a value with a mask (read-modify-write).
    MaskWrite {
        /// Register offset.
        reg_off: u32,
        /// Value to write (after masking).
        value: u32,
        /// Mask: only bits set in mask are modified.
        mask: u32,
    },

    /// Poll a register until (value & mask) matches expected.
    MaskPoll {
        /// Register offset.
        reg_off: u32,
        /// Expected value (after masking).
        value: u32,
        /// Mask to apply before comparison.
        mask: u32,
    },

    /// Patch an address with a host buffer address.
    DdrPatch {
        /// Register address to patch.
        reg_addr: u32,
        /// Argument index (which host buffer to use).
        arg_idx: u8,
        /// Offset to add to the buffer address.
        arg_plus: u32,
    },

    /// Sync/wait operation.
    Sync {
        /// Channel number.
        channel: u8,
        /// Column number.
        column: u8,
        /// Direction (0 = S2MM, 1 = MM2S).
        direction: u8,
        /// Number of columns.
        column_num: u8,
        /// Row number.
        row: u8,
        /// Number of rows.
        row_num: u8,
    },

    /// Unknown instruction (for debugging).
    Unknown {
        /// The opcode byte.
        opcode: u8,
        /// Raw bytes following the opcode.
        data: Vec<u8>,
    },
}

/// A stream of NPU instructions parsed from binary data.
pub struct NpuInstructionStream {
    instructions: Vec<NpuInstruction>,
}

impl NpuInstructionStream {
    /// Parse NPU instructions from binary data.
    ///
    /// The format has a file header followed by instruction opcodes.
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("NPU instruction data too short".to_string());
        }

        let mut cursor = Cursor::new(data);
        let mut instructions = Vec::new();

        // Parse file header
        // Format appears to be:
        // [0-3]: Magic/version (0x06030100 for optimized format)
        // [4-7]: Flags
        // [8-11]: Number of operations
        // [12-15]: Total size in bytes
        let magic = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let _flags = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let num_ops = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let total_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

        // Validate header
        if magic != 0x06030100 {
            return Err(format!("Unknown NPU instruction magic: 0x{:08X}", magic));
        }

        if total_size as usize > data.len() {
            return Err(format!(
                "NPU instruction size {} exceeds data length {}",
                total_size,
                data.len()
            ));
        }

        // Parse instructions
        for i in 0..num_ops {
            let pos = cursor.position() as usize;
            if pos >= data.len() {
                break;
            }

            log::debug!("Parsing instruction {} at offset 0x{:X} ({})", i, pos, pos);

            match Self::parse_instruction(&mut cursor, data) {
                Ok(instr) => {
                    let end_pos = cursor.position() as usize;
                    log::debug!("  Instruction {} parsed, ends at 0x{:X}, size={}", i, end_pos, end_pos - pos);
                    instructions.push(instr);
                }
                Err(e) => {
                    // Log but continue - some instructions might be unknown
                    log::warn!("Failed to parse NPU instruction at offset {}: {}", pos, e);
                    break;
                }
            }
        }

        Ok(Self { instructions })
    }

    /// Parse a single instruction from the cursor.
    ///
    /// NPU instruction format:
    /// - Standard ops (opcode < 128): 8-byte header (4 opcode + 4 zeros), then op-specific fields
    /// - Custom ops (opcode >= 128): 4-byte header + 4-byte size + payload
    fn parse_instruction(cursor: &mut Cursor<&[u8]>, _data: &[u8]) -> Result<NpuInstruction, String> {
        // Read 4-byte opcode header (opcode byte + 3 padding bytes)
        let opcode_byte = cursor.read_u8().map_err(|e| e.to_string())?;
        let _padding1 = cursor.read_u24::<LittleEndian>().map_err(|e| e.to_string())?;

        let opcode = NpuOpcode::from(opcode_byte);

        // Standard ops have an additional 4 bytes of zeros after the opcode header
        // Custom ops (>= 128) have size field immediately after opcode header
        let is_custom = opcode_byte >= 128;

        if is_custom {
            // Custom op: 4-byte header + 4-byte size + payload
            let size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
            let header_size = 8u32; // opcode header + size field
            let remaining = size.saturating_sub(header_size);

            // Read payload
            let mut payload = vec![0u8; remaining as usize];
            cursor.read_exact(&mut payload).map_err(|e| e.to_string())?;

            if opcode == NpuOpcode::Tct {
                // TCT (Transaction Control Token) = Sync instruction
                // Format (from AIETargetNPU.cpp appendSync):
                //   words[0] = opcode (0x80)
                //   words[1] = size (16 bytes)
                //   words[2] = direction | (row << 8) | (column << 16)
                //   words[3] = (row_num << 8) | (column_num << 16) | (channel << 24)
                //
                // Payload is 8 bytes (words 2-3 after 8-byte header)
                let word2 = if payload.len() >= 4 {
                    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])
                } else {
                    0
                };
                let word3 = if payload.len() >= 8 {
                    u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]])
                } else {
                    0
                };

                let direction = (word2 & 0xFF) as u8;
                let row = ((word2 >> 8) & 0xFF) as u8;
                let column = ((word2 >> 16) & 0xFF) as u8;
                let row_num = ((word3 >> 8) & 0xFF) as u8;
                let column_num = ((word3 >> 16) & 0xFF) as u8;
                let channel = ((word3 >> 24) & 0xFF) as u8;

                log::debug!(
                    "Sync parsed: dir={} row={} col={} ch={}",
                    direction, row, column, channel
                );

                Ok(NpuInstruction::Sync {
                    channel,
                    column,
                    direction,
                    column_num,
                    row,
                    row_num,
                })
            } else if opcode == NpuOpcode::DdrPatch {
                // DDR patch payload structure (from AIETargetNPU.cpp appendAddressPatch):
                // Instruction is 12 words total (48 bytes):
                //   words[0] = opcode (0x81)
                //   words[1] = size (48)
                //   words[2-4] = unused
                //   words[5] = action (0)
                //   words[6] = reg_addr (the BD address field to patch)
                //   words[7] = unused
                //   words[8] = arg_idx (which host buffer)
                //   words[9] = unused
                //   words[10] = arg_plus (offset within buffer)
                //   words[11] = unused
                //
                // Payload starts after 8-byte header (opcode + size), so payload offsets:
                //   payload[16..20] = words[6] = reg_addr
                //   payload[24..28] = words[8] = arg_idx
                //   payload[32..36] = words[10] = arg_plus

                let reg_addr = if payload.len() >= 20 {
                    u32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]])
                } else {
                    0
                };

                let arg_idx = if payload.len() >= 28 {
                    payload[24]  // Only the low byte is used
                } else {
                    0
                };

                let arg_plus = if payload.len() >= 36 {
                    u32::from_le_bytes([payload[32], payload[33], payload[34], payload[35]])
                } else {
                    0
                };

                log::debug!(
                    "DdrPatch parsed: reg_addr=0x{:08X} arg_idx={} arg_plus={}",
                    reg_addr, arg_idx, arg_plus
                );

                Ok(NpuInstruction::DdrPatch {
                    reg_addr,
                    arg_idx,
                    arg_plus,
                })
            } else {
                Ok(NpuInstruction::Unknown {
                    opcode: opcode_byte,
                    data: payload,
                })
            }
        } else {
            // Standard op: 8-byte header (4 opcode + 4 zeros), then op-specific fields
            let _padding2 = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

            match opcode {
                NpuOpcode::Write32 => {
                    // Write32: 8-byte header + 8-byte reg_off (u64) + 4-byte value + 4-byte size = 24 bytes
                    // RegOff is u64 aligned, so we read it as u64
                    let reg_off = cursor.read_u64::<LittleEndian>().map_err(|e| e.to_string())?;
                    let value = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let _size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    Ok(NpuInstruction::Write32 { reg_off: reg_off as u32, value })
                }

                NpuOpcode::BlockWrite => {
                    // BlockWrite: 8-byte header + 4-byte reg_off + 4-byte size + payload
                    let reg_off = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

                    // Size includes full instruction (16-byte header + payload)
                    let header_size = 16u32;
                    let payload_bytes = size.saturating_sub(header_size);
                    let num_words = payload_bytes / 4;

                    let mut values = Vec::with_capacity(num_words as usize);
                    for _ in 0..num_words {
                        let val = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                        values.push(val);
                    }

                    Ok(NpuInstruction::BlockWrite { reg_off, values })
                }

                NpuOpcode::MaskWrite => {
                    // MaskWrite: 8-byte header + 8-byte reg_off + 4-byte value + 4-byte mask + 4-byte size = 28 bytes
                    let reg_off = cursor.read_u64::<LittleEndian>().map_err(|e| e.to_string())?;
                    let value = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let mask = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let _size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    Ok(NpuInstruction::MaskWrite { reg_off: reg_off as u32, value, mask })
                }

                NpuOpcode::MaskPoll => {
                    // MaskPoll: 8-byte header + 8-byte reg_off + 4-byte value + 4-byte mask + 4-byte size = 28 bytes
                    let reg_off = cursor.read_u64::<LittleEndian>().map_err(|e| e.to_string())?;
                    let value = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let mask = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    let _size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
                    Ok(NpuInstruction::MaskPoll { reg_off: reg_off as u32, value, mask })
                }

                _ => {
                    // Unknown standard op - skip 8 bytes (we already read 8 byte header)
                    let mut buf = vec![0u8; 8];
                    cursor.read_exact(&mut buf).ok();
                    Ok(NpuInstruction::Unknown {
                        opcode: opcode_byte,
                        data: buf,
                    })
                }
            }
        }
    }

    /// Get the parsed instructions.
    pub fn instructions(&self) -> &[NpuInstruction] {
        &self.instructions
    }

    /// Get the number of instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        let result = NpuInstructionStream::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_opcode_from_u8() {
        assert_eq!(NpuOpcode::from(0), NpuOpcode::Write32);
        assert_eq!(NpuOpcode::from(1), NpuOpcode::BlockWrite);
        assert_eq!(NpuOpcode::from(3), NpuOpcode::MaskWrite);
        assert_eq!(NpuOpcode::from(128), NpuOpcode::Tct);
        assert_eq!(NpuOpcode::from(129), NpuOpcode::DdrPatch);
        assert_eq!(NpuOpcode::from(200), NpuOpcode::Unknown);
    }
}
