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
#[derive(Debug)]
pub struct NpuInstructionStream {
    instructions: Vec<NpuInstruction>,
}

impl NpuInstructionStream {
    /// Parse NPU instructions from binary data.
    ///
    /// Accepts two formats:
    /// - Raw binary: 16-byte header (magic 0x06030100) followed by opcodes
    /// - ELF wrapper: standard ELF with `.ctrltext` section containing raw bytes
    ///   (produced by `aiebu_asm` via `aiecc.py --aie-generate-elf`)
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("NPU instruction data too short".to_string());
        }

        // Detect ELF magic (\x7FELF) and extract .ctrltext section.
        if data.len() >= 4 && &data[..4] == b"\x7fELF" {
            return Self::parse_elf(data);
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

    /// Extract raw NPU instructions from an ELF wrapper.
    ///
    /// XRT's `xrt::elf` API uses ELF files produced by `aiebu_asm`. The raw
    /// NPU instruction bytes live in the `.ctrltext` section (PROGBITS,
    /// ALLOC|EXECUTE). We extract that section and parse it as raw binary.
    ///
    /// These ELFs are non-standard (repurposed e_machine, unusual PHDR layout)
    /// so we parse only the ELF header and section headers rather than using
    /// goblin's full parser which rejects them.
    fn parse_elf(data: &[u8]) -> Result<Self, String> {
        use goblin::elf32::header::{Header as Elf32Header, SIZEOF_EHDR};
        use goblin::elf32::section_header::SectionHeader;

        if data.len() < SIZEOF_EHDR {
            return Err("Instruction ELF too small for ELF32 header".to_string());
        }

        let header = Elf32Header::from_bytes(
            data[..SIZEOF_EHDR]
                .try_into()
                .map_err(|_| "Instruction ELF header size mismatch")?,
        );

        let shoff = header.e_shoff as usize;
        let shnum = header.e_shnum as usize;
        let shentsize = header.e_shentsize as usize;
        let shstrndx = header.e_shstrndx as usize;

        if shentsize == 0 || shnum == 0 {
            return Err("Instruction ELF has no section headers".to_string());
        }

        let shdr_end = shoff + shnum * shentsize;
        if shdr_end > data.len() {
            return Err(format!(
                "Section headers extend past end of ELF (need {}, have {})",
                shdr_end, data.len()
            ));
        }

        let shdrs = SectionHeader::from_bytes(&data[shoff..shdr_end], shnum);

        // Locate the section name string table.
        if shstrndx >= shnum {
            return Err("Invalid e_shstrndx".to_string());
        }
        let strtab_sh = &shdrs[shstrndx];
        let strtab_off = strtab_sh.sh_offset as usize;
        let strtab_size = strtab_sh.sh_size as usize;
        if strtab_off + strtab_size > data.len() {
            return Err("Section string table extends past end of ELF".to_string());
        }
        let strtab = &data[strtab_off..strtab_off + strtab_size];

        // Find .ctrltext by name.
        for sh in &shdrs {
            let name_off = sh.sh_name as usize;
            if name_off >= strtab.len() {
                continue;
            }
            let name_end = strtab[name_off..]
                .iter()
                .position(|&b| b == 0)
                .map_or(strtab.len(), |p| name_off + p);
            let name = std::str::from_utf8(&strtab[name_off..name_end]).unwrap_or("");

            if name == ".ctrltext" {
                let offset = sh.sh_offset as usize;
                let size = sh.sh_size as usize;
                if offset + size > data.len() {
                    return Err(format!(
                        ".ctrltext section extends past end of ELF \
                         (offset={}, size={}, file={})",
                        offset, size, data.len()
                    ));
                }
                let ctrltext = &data[offset..offset + size];
                log::info!(
                    "Extracted {} bytes from .ctrltext section in instruction ELF",
                    size
                );
                return Self::parse(ctrltext);
            }
        }

        Err("Instruction ELF has no .ctrltext section".to_string())
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

                // Payload must be at least 36 bytes to contain all fields.
                // The instruction is 48 bytes total (40 bytes payload after
                // 8-byte header), but we only need through offset 35.
                if payload.len() < 36 {
                    return Err(format!(
                        "DdrPatch payload too short: {} bytes (need at least 36)",
                        payload.len()
                    ));
                }

                let reg_addr = u32::from_le_bytes([
                    payload[16], payload[17], payload[18], payload[19],
                ]);
                let arg_idx = payload[24];
                let arg_plus = u32::from_le_bytes([
                    payload[32], payload[33], payload[34], payload[35],
                ]);

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

                // Opcodes that the emulator can safely skip.
                // Noop: does nothing by definition.
                // MaskPoll/MaskPollBusy: emulator writes are synchronous.
                // Preempt: no preemption in emulator.
                // LoadPdi/LoadPmStart/LoadPmEndInternal: firmware-level ops.
                // CreateScratchpad/UpdateStateTable/UpdateReg/UpdateScratch: firmware state.
                NpuOpcode::Noop => {
                    log::debug!("NPU NOOP");
                    // Consume the remaining 4 bytes of the standard 8-byte header
                    let _zeros = cursor.read_u32::<LittleEndian>().ok();
                    Ok(NpuInstruction::Unknown { opcode: opcode_byte, data: vec![] })
                }

                NpuOpcode::Preempt | NpuOpcode::MaskPollBusy |
                NpuOpcode::LoadPdi | NpuOpcode::LoadPmStart |
                NpuOpcode::CreateScratchpad | NpuOpcode::UpdateStateTable |
                NpuOpcode::UpdateReg | NpuOpcode::UpdateScratch |
                NpuOpcode::LoadPmEndInternal => {
                    log::debug!("NPU opcode {:?} -- firmware-level, skipping", opcode);
                    let _zeros = cursor.read_u32::<LittleEndian>().ok();
                    // These may have payloads; skip 8 more bytes conservatively
                    let mut buf = vec![0u8; 8];
                    cursor.read_exact(&mut buf).ok();
                    Ok(NpuInstruction::Unknown { opcode: opcode_byte, data: buf })
                }

                // ConfigShimDmaBd (14) and ConfigShimDmaDmaBufBd (15) configure
                // shim DMA BDs as complete 8-word transactions. Not yet
                // implemented -- panic loudly so we never silently skip a BD
                // configuration that would change DMA behavior.
                NpuOpcode::ConfigShimDmaBd | NpuOpcode::ConfigShimDmaDmaBufBd => {
                    panic!(
                        "NPU opcode {:?} ({:#04x}) is not implemented. \
                         This instruction configures a shim DMA BD with \
                         multi-dimensional addressing parameters; silently \
                         skipping it would corrupt DMA transfers. Implement \
                         in src/npu/parser.rs and src/npu/executor.rs before \
                         running tests that emit it.",
                        opcode, opcode_byte
                    );
                }

                _ => {
                    panic!(
                        "NPU unknown opcode {} ({:#04x}) at offset {:#X}. \
                         Silently skipping unknown opcodes is unsafe -- they \
                         may carry payload bytes whose absence corrupts \
                         downstream parsing, or they may configure hardware \
                         state that affects test results. Add a handler in \
                         src/npu/parser.rs.",
                        opcode_byte, opcode_byte, cursor.position().saturating_sub(4)
                    );
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
    fn test_parse_elf_magic_triggers_elf_path() {
        // A truncated ELF that starts with \x7FELF but isn't valid.
        let data = b"\x7fELF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = NpuInstructionStream::parse(data);
        assert!(result.is_err());
        // Should complain about ELF parsing, not about "Unknown NPU instruction magic"
        let err = result.unwrap_err();
        assert!(
            err.contains("ELF") || err.contains("elf"),
            "Expected ELF-related error, got: {}",
            err
        );
    }

    #[test]
    fn test_parse_elf_no_ctrltext() {
        // Build a minimal valid ELF32 LE with no .ctrltext section.
        let elf = build_minimal_elf32(None);
        let result = NpuInstructionStream::parse(&elf);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("no .ctrltext"),
            "Expected missing .ctrltext error"
        );
    }

    #[test]
    fn test_parse_elf_with_ctrltext() {
        // Build a minimal raw NPU instruction stream (header only, 0 ops).
        let raw_npu: Vec<u8> = vec![
            0x00, 0x01, 0x03, 0x06, // magic LE = 0x06030100
            0x00, 0x00, 0x00, 0x00, // flags
            0x00, 0x00, 0x00, 0x00, // num_ops = 0
            0x10, 0x00, 0x00, 0x00, // total_size = 16
        ];
        let elf = build_minimal_elf32(Some(&raw_npu));
        let result = NpuInstructionStream::parse(&elf);
        assert!(result.is_ok(), "ELF parse failed: {:?}", result.err());
        assert_eq!(result.unwrap().len(), 0);
    }

    /// Build a minimal ELF32 little-endian file, optionally with a
    /// `.ctrltext` section containing `ctrltext_data`.
    fn build_minimal_elf32(ctrltext_data: Option<&[u8]>) -> Vec<u8> {
        let mut buf = Vec::new();

        // String table: \0 + ".shstrtab\0" + optional ".ctrltext\0"
        let mut strtab = vec![0u8]; // index 0 = empty
        let shstrtab_name_idx = strtab.len();
        strtab.extend_from_slice(b".shstrtab\0");
        let ctrltext_name_idx = if ctrltext_data.is_some() {
            let idx = strtab.len();
            strtab.extend_from_slice(b".ctrltext\0");
            idx
        } else {
            0
        };

        // Section count: null + .shstrtab + optional .ctrltext
        let num_sections: u16 = if ctrltext_data.is_some() { 3 } else { 2 };
        let shentsize: u16 = 40; // sizeof(Elf32_Shdr)

        // Layout: ELF header (52) | ctrltext data | strtab | section headers
        let ehdr_size: usize = 52;
        let ctrltext_len = ctrltext_data.map_or(0, |d| d.len());
        let ctrltext_offset = ehdr_size;
        let strtab_offset = ctrltext_offset + ctrltext_len;
        let shdr_offset = strtab_offset + strtab.len();

        // ELF header (52 bytes)
        buf.extend_from_slice(b"\x7fELF");       // e_ident[0..4]: magic
        buf.push(1);                               // EI_CLASS: ELFCLASS32
        buf.push(1);                               // EI_DATA: ELFDATA2LSB
        buf.push(1);                               // EI_VERSION: EV_CURRENT
        buf.extend_from_slice(&[0; 9]);             // EI_PAD
        buf.extend_from_slice(&2u16.to_le_bytes()); // e_type: ET_EXEC
        buf.extend_from_slice(&0u16.to_le_bytes()); // e_machine
        buf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        buf.extend_from_slice(&0u32.to_le_bytes()); // e_entry
        buf.extend_from_slice(&0u32.to_le_bytes()); // e_phoff
        buf.extend_from_slice(&(shdr_offset as u32).to_le_bytes()); // e_shoff
        buf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        buf.extend_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
        buf.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
        buf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        buf.extend_from_slice(&shentsize.to_le_bytes()); // e_shentsize
        buf.extend_from_slice(&num_sections.to_le_bytes()); // e_shnum
        buf.extend_from_slice(&1u16.to_le_bytes()); // e_shstrndx = 1

        assert_eq!(buf.len(), ehdr_size);

        // .ctrltext data (if present)
        if let Some(data) = ctrltext_data {
            buf.extend_from_slice(data);
        }

        // String table content
        buf.extend_from_slice(&strtab);

        // Section headers
        assert_eq!(buf.len(), shdr_offset);

        // SHdr 0: null section (40 zero bytes)
        buf.extend_from_slice(&[0u8; 40]);

        // SHdr 1: .shstrtab
        buf.extend_from_slice(&(shstrtab_name_idx as u32).to_le_bytes()); // sh_name
        buf.extend_from_slice(&3u32.to_le_bytes()); // sh_type = SHT_STRTAB
        buf.extend_from_slice(&0u32.to_le_bytes()); // sh_flags
        buf.extend_from_slice(&0u32.to_le_bytes()); // sh_addr
        buf.extend_from_slice(&(strtab_offset as u32).to_le_bytes()); // sh_offset
        buf.extend_from_slice(&(strtab.len() as u32).to_le_bytes()); // sh_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // sh_link
        buf.extend_from_slice(&0u32.to_le_bytes()); // sh_info
        buf.extend_from_slice(&1u32.to_le_bytes()); // sh_addralign
        buf.extend_from_slice(&0u32.to_le_bytes()); // sh_entsize

        // SHdr 2: .ctrltext (if present)
        if ctrltext_data.is_some() {
            buf.extend_from_slice(&(ctrltext_name_idx as u32).to_le_bytes()); // sh_name
            buf.extend_from_slice(&1u32.to_le_bytes()); // sh_type = SHT_PROGBITS
            buf.extend_from_slice(&6u32.to_le_bytes()); // sh_flags = ALLOC|EXEC
            buf.extend_from_slice(&0u32.to_le_bytes()); // sh_addr
            buf.extend_from_slice(&(ctrltext_offset as u32).to_le_bytes()); // sh_offset
            buf.extend_from_slice(&(ctrltext_len as u32).to_le_bytes()); // sh_size
            buf.extend_from_slice(&0u32.to_le_bytes()); // sh_link
            buf.extend_from_slice(&0u32.to_le_bytes()); // sh_info
            buf.extend_from_slice(&1u32.to_le_bytes()); // sh_addralign
            buf.extend_from_slice(&0u32.to_le_bytes()); // sh_entsize
        }

        buf
    }

    #[test]
    fn test_parse_real_insts_elf() {
        // If a build artifact exists, verify we can parse the real thing.
        let elf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("build/peano/add_one_objFifo_elf/insts.elf");
        if !elf_path.exists() {
            eprintln!("Skipping: {} not found", elf_path.display());
            return;
        }
        let data = std::fs::read(&elf_path).unwrap();
        let result = NpuInstructionStream::parse(&data);
        assert!(result.is_ok(), "Failed to parse real insts.elf: {:?}", result.err());
        let stream = result.unwrap();
        assert!(stream.len() > 0, "Expected non-empty instruction stream from real ELF");
    }

    #[test]
    fn test_opcode_from_u8() {
        // Standard ops (0-4)
        assert_eq!(NpuOpcode::from(0), NpuOpcode::Write32);
        assert_eq!(NpuOpcode::from(1), NpuOpcode::BlockWrite);
        assert_eq!(NpuOpcode::from(2), NpuOpcode::BlockSet);
        assert_eq!(NpuOpcode::from(3), NpuOpcode::MaskWrite);
        assert_eq!(NpuOpcode::from(4), NpuOpcode::MaskPoll);
        // Extended ops (5-15) per xdna-driver numbering
        assert_eq!(NpuOpcode::from(5), NpuOpcode::Noop);
        assert_eq!(NpuOpcode::from(6), NpuOpcode::Preempt);
        assert_eq!(NpuOpcode::from(14), NpuOpcode::ConfigShimDmaBd);
        assert_eq!(NpuOpcode::from(15), NpuOpcode::ConfigShimDmaDmaBufBd);
        // Custom ops (128+)
        assert_eq!(NpuOpcode::from(128), NpuOpcode::Tct);
        assert_eq!(NpuOpcode::from(129), NpuOpcode::DdrPatch);
        assert_eq!(NpuOpcode::from(130), NpuOpcode::ReadRegs);
        assert_eq!(NpuOpcode::from(200), NpuOpcode::LoadPmEndInternal);
        // Unknown
        assert_eq!(NpuOpcode::from(99), NpuOpcode::Unknown);
        assert_eq!(NpuOpcode::from(254), NpuOpcode::Unknown);
    }
}
