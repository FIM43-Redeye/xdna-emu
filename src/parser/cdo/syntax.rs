//! Syntax: typed commands (CdoRaw) decoded from CDO byte frames.
//! Shape-aware of command opcodes but not of their device-level effects.
//!
//! `Cdo` owns the parsed header + raw command stream and exposes a
//! `commands()` iterator that yields `CdoRaw` -- typed commands with
//! decoded payloads. Device-level interpretation (address decoding,
//! BD field parsing, tile routing, etc.) is the semantics layer's job;
//! this module is deliberately arch-blind for the common path.
//!
//! Note: `CdoRaw::decode_aie_address` retains an AIE2-specific helper
//! for legacy callers; Stage 8b Half 2 will relocate this kind of
//! arch-aware decoding into the semantics layer behind `ArchHandle`.

use anyhow::{anyhow, bail, Result};
use zerocopy::FromBytes;

use super::framing::{CdoVersion, RawCdoHeader, CDO_HEADER_SIZE, CDO_MAGIC_CDO, CDO_MAGIC_XLNX};

/// CDO command opcodes (CDOv2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CdoOpcode {
    // General Commands (0x1xx)
    EndMark = 0x100,
    MaskPoll = 0x101,
    MaskWrite = 0x102,
    Write = 0x103,
    Delay = 0x104,
    DmaWrite = 0x105,
    MaskPoll64 = 0x106,
    MaskWrite64 = 0x107,
    Write64 = 0x108,
    DmaXfer = 0x109,
    InitSeq = 0x10A,
    CframeRead = 0x10B,
    Set = 0x10C,
    DmaWriteKeyhole = 0x10D,
    SsitSyncMaster = 0x10E,
    SsitSyncSlaves = 0x10F,
    SsitWaitSlaves = 0x110,
    Nop = 0x111,
    GetDeviceId = 0x112,
    EventLogging = 0x113,
    SetBoard = 0x114,
    GetBoard = 0x115,
    SetPlmWdt = 0x116,
    LogString = 0x117,
    LogAddress = 0x118,
    Marker = 0x119,
    Proc = 0x11A,
    BlockBegin = 0x11B,
    BlockEnd = 0x11C,
    Break = 0x11D,
    OtCheck = 0x11E,
    PsmSequence = 0x11F,
    PlmUpdate = 0x120,
    ScatterWrite = 0x121,
    ScatterWrite2 = 0x122,
    TamperTrigger = 0x123,
    SetIpiAccess = 0x125,

    // PM Commands (0x2xx)
    PmGetApiVersion = 0x201,
    PmGetDeviceStatus = 0x203,
    PmRequestDevice = 0x20D,
    PmReleaseDevice = 0x20E,
    PmResetAssert = 0x211,
    PmClockEnable = 0x224,

    // Unknown/other
    Unknown(u16),
}

impl From<u16> for CdoOpcode {
    fn from(v: u16) -> Self {
        match v {
            0x100 => Self::EndMark,
            0x101 => Self::MaskPoll,
            0x102 => Self::MaskWrite,
            0x103 => Self::Write,
            0x104 => Self::Delay,
            0x105 => Self::DmaWrite,
            0x106 => Self::MaskPoll64,
            0x107 => Self::MaskWrite64,
            0x108 => Self::Write64,
            0x109 => Self::DmaXfer,
            0x10A => Self::InitSeq,
            0x10B => Self::CframeRead,
            0x10C => Self::Set,
            0x10D => Self::DmaWriteKeyhole,
            0x10E => Self::SsitSyncMaster,
            0x10F => Self::SsitSyncSlaves,
            0x110 => Self::SsitWaitSlaves,
            0x111 => Self::Nop,
            0x112 => Self::GetDeviceId,
            0x113 => Self::EventLogging,
            0x114 => Self::SetBoard,
            0x115 => Self::GetBoard,
            0x116 => Self::SetPlmWdt,
            0x117 => Self::LogString,
            0x118 => Self::LogAddress,
            0x119 => Self::Marker,
            0x11A => Self::Proc,
            0x11B => Self::BlockBegin,
            0x11C => Self::BlockEnd,
            0x11D => Self::Break,
            0x11E => Self::OtCheck,
            0x11F => Self::PsmSequence,
            0x120 => Self::PlmUpdate,
            0x121 => Self::ScatterWrite,
            0x122 => Self::ScatterWrite2,
            0x123 => Self::TamperTrigger,
            0x125 => Self::SetIpiAccess,
            0x201 => Self::PmGetApiVersion,
            0x203 => Self::PmGetDeviceStatus,
            0x20D => Self::PmRequestDevice,
            0x20E => Self::PmReleaseDevice,
            0x211 => Self::PmResetAssert,
            0x224 => Self::PmClockEnable,
            other => Self::Unknown(other),
        }
    }
}

/// Decoded CDO command with payload
#[derive(Debug, Clone)]
pub enum CdoRaw {
    /// Write value to 32-bit address
    Write { address: u32, value: u32 },

    /// Masked write: *addr = (*addr & ~mask) | (value & mask)
    MaskWrite { address: u32, mask: u32, value: u32 },

    /// Write value to 64-bit address
    Write64 { address: u64, value: u32 },

    /// Masked write to 64-bit address
    MaskWrite64 { address: u64, mask: u32, value: u32 },

    /// DMA write: bulk data to address
    DmaWrite { address: u32, data: Vec<u8> },

    /// Poll register until (val & mask) == expected
    MaskPoll { address: u32, mask: u32, expected: u32 },

    /// Poll 64-bit address
    MaskPoll64 { address: u64, mask: u32, expected: u32 },

    /// Delay for N cycles
    Delay { cycles: u32 },

    /// No operation (padding)
    Nop { words: u16 },

    /// End marker
    EndMark,

    /// Debug marker
    Marker { value: u32 },

    /// Unknown/unhandled command
    Unknown { opcode: u16, payload: Vec<u32> },
}

impl CdoRaw {
    /// Returns the target address for address-based commands
    pub fn address(&self) -> Option<u64> {
        match self {
            Self::Write { address, .. } => Some(*address as u64),
            Self::MaskWrite { address, .. } => Some(*address as u64),
            Self::Write64 { address, .. } => Some(*address),
            Self::MaskWrite64 { address, .. } => Some(*address),
            Self::DmaWrite { address, .. } => Some(*address as u64),
            Self::MaskPoll { address, .. } => Some(*address as u64),
            Self::MaskPoll64 { address, .. } => Some(*address),
            _ => None,
        }
    }

    /// Decode tile coordinates from AIE address
    /// Returns (column, row, offset) for standard AIE2 addressing
    pub fn decode_aie_address(&self) -> Option<(u8, u8, u32)> {
        use xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK};
        let addr = self.address()?;
        // AIE2: col = bits[29:25], row = bits[24:20], offset = bits[19:0]
        let col = ((addr >> TILE_COL_SHIFT) & 0x1F) as u8;
        let row = ((addr >> TILE_ROW_SHIFT) & 0x1F) as u8;
        let offset = (addr as u32) & TILE_OFFSET_MASK;
        Some((col, row, offset))
    }
}

/// Parsed CDO container
pub struct Cdo<'a> {
    data: &'a [u8],
    pub header: RawCdoHeader,
    commands_offset: usize,
}

impl<'a> Cdo<'a> {
    /// Parse a CDO from raw bytes
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < CDO_HEADER_SIZE {
            bail!(
                "CDO too small: {} bytes (minimum {})",
                data.len(),
                CDO_HEADER_SIZE
            );
        }

        let (header, _) = RawCdoHeader::read_from_prefix(data)
            .map_err(|e| anyhow!("Failed to parse CDO header: {:?}", e))?;

        // Validate magic
        if header.ident_word != CDO_MAGIC_CDO && header.ident_word != CDO_MAGIC_XLNX {
            bail!(
                "Invalid CDO magic: 0x{:08X} (expected 0x{:08X} or 0x{:08X})",
                header.ident_word,
                CDO_MAGIC_CDO,
                CDO_MAGIC_XLNX
            );
        }

        // Validate checksum
        let computed_checksum =
            !(header.num_words + header.ident_word + header.version + header.cdo_length);
        if header.checksum != computed_checksum {
            // Warn but don't fail -- some tools may not set checksum correctly.
            log::warn!(
                "CDO checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
                computed_checksum, header.checksum
            );
        }

        Ok(Self {
            data,
            header,
            commands_offset: CDO_HEADER_SIZE,
        })
    }

    /// Get the CDO version
    pub fn version(&self) -> CdoVersion {
        CdoVersion::from(self.header.version)
    }

    /// Get the magic identifier
    pub fn magic(&self) -> &'static str {
        if self.header.ident_word == CDO_MAGIC_CDO {
            "CDO"
        } else {
            "XLNX"
        }
    }

    /// Get the total command length in words
    pub fn command_length_words(&self) -> usize {
        self.header.cdo_length as usize
    }

    /// Get the raw command data
    pub fn command_data(&self) -> &'a [u8] {
        let end = (self.commands_offset + self.header.cdo_length as usize * 4).min(self.data.len());
        &self.data[self.commands_offset..end]
    }

    /// Iterate over all commands
    pub fn commands(&self) -> CdoRawIterator<'a> {
        CdoRawIterator {
            data: self.command_data(),
            offset: 0,
        }
    }

    /// Count commands by type
    pub fn command_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for cmd in self.commands() {
            let name = match &cmd {
                CdoRaw::Write { .. } => "WRITE",
                CdoRaw::MaskWrite { .. } => "MASK_WRITE",
                CdoRaw::Write64 { .. } => "WRITE64",
                CdoRaw::MaskWrite64 { .. } => "MASK_WRITE64",
                CdoRaw::DmaWrite { .. } => "DMA_WRITE",
                CdoRaw::MaskPoll { .. } => "MASK_POLL",
                CdoRaw::MaskPoll64 { .. } => "MASK_POLL64",
                CdoRaw::Delay { .. } => "DELAY",
                CdoRaw::Nop { .. } => "NOP",
                CdoRaw::EndMark => "END_MARK",
                CdoRaw::Marker { .. } => "MARKER",
                CdoRaw::Unknown { opcode, .. } => {
                    let key = format!("UNKNOWN(0x{:03X})", opcode);
                    *counts.entry(key).or_insert(0) += 1;
                    continue;
                }
            };
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Print a summary of the CDO contents
    pub fn print_summary(&self) {
        println!("CDO Summary");
        println!("===========");
        println!("Magic: {} (0x{:08X})", self.magic(), self.header.ident_word);
        println!("Version: {:?}", self.version());
        println!("Length: {} words ({} bytes)", self.command_length_words(), self.command_length_words() * 4);
        println!();
        println!("Command counts:");
        let counts = self.command_counts();
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in sorted {
            println!("  {}: {}", name, count);
        }
    }
}

/// Iterator over CDO commands
pub struct CdoRawIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for CdoRawIterator<'a> {
    type Item = CdoRaw;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + 4 > self.data.len() {
            return None;
        }

        // Read command word: [31:16] = payload_length, [15:0] = opcode
        let cmd_word = u32::from_le_bytes([
            self.data[self.offset],
            self.data[self.offset + 1],
            self.data[self.offset + 2],
            self.data[self.offset + 3],
        ]);
        self.offset += 4;

        let opcode = CdoOpcode::from((cmd_word & 0xFFFF) as u16);
        let inline_len = ((cmd_word >> 16) & 0xFF) as usize;

        // CDO v2 extended length: when the 8-bit inline length field is 0xFF,
        // the actual payload length is in the next word (supports payloads > 254 words).
        // See Xilinx CDO library: CDO_MAX_INLINE_PAYLOAD_LEN = 255 sentinel.
        let payload_len = if inline_len == 0xFF {
            if self.offset + 4 > self.data.len() {
                self.offset = self.data.len();
                return Some(CdoRaw::Unknown {
                    opcode: (cmd_word & 0xFFFF) as u16,
                    payload: Vec::new(),
                });
            }
            let actual_len = u32::from_le_bytes([
                self.data[self.offset],
                self.data[self.offset + 1],
                self.data[self.offset + 2],
                self.data[self.offset + 3],
            ]) as usize;
            self.offset += 4;
            actual_len
        } else {
            inline_len
        };

        // Read payload words
        let payload_bytes = payload_len * 4;
        if self.offset + payload_bytes > self.data.len() {
            // Truncated command - return what we can
            self.offset = self.data.len();
            return Some(CdoRaw::Unknown {
                opcode: (cmd_word & 0xFFFF) as u16,
                payload: Vec::new(),
            });
        }

        let payload: Vec<u32> = (0..payload_len)
            .map(|i| {
                let start = self.offset + i * 4;
                u32::from_le_bytes([
                    self.data[start],
                    self.data[start + 1],
                    self.data[start + 2],
                    self.data[start + 3],
                ])
            })
            .collect();

        self.offset += payload_bytes;

        // Decode command
        let cmd = match opcode {
            CdoOpcode::EndMark => CdoRaw::EndMark,

            CdoOpcode::Write if payload_len >= 2 => CdoRaw::Write {
                address: payload[0],
                value: payload[1],
            },

            CdoOpcode::MaskWrite if payload_len >= 3 => CdoRaw::MaskWrite {
                address: payload[0],
                mask: payload[1],
                value: payload[2],
            },

            CdoOpcode::Write64 if payload_len >= 3 => CdoRaw::Write64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                value: payload[2],
            },

            CdoOpcode::MaskWrite64 if payload_len >= 4 => CdoRaw::MaskWrite64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                mask: payload[2],
                value: payload[3],
            },

            CdoOpcode::DmaWrite if payload_len >= 2 => {
                // CDO v2 DmaWrite format: [addr_hi, addr_lo, data...]
                // addr_hi:addr_lo form a 64-bit destination address.
                // Data length is (payload_len - 2) * 4 bytes.
                // For AIE tiles, addr_hi is always 0 and addr_lo is the tile address.
                let addr_hi = payload[0];
                let addr_lo = payload[1];
                let address = if addr_hi == 0 {
                    addr_lo
                } else {
                    // AIE tile addresses are 32-bit. A non-zero addr_hi means
                    // this targets DDR above 4GB, which the emulator's host
                    // memory model does not support. Warn and use low 32 bits.
                    log::warn!(
                        "CDO DmaWrite has 64-bit addr 0x{:08X}_{:08X} (>4GB); \
                         emulator only supports 32-bit tile addresses, using low 32 bits",
                        addr_hi, addr_lo
                    );
                    addr_lo
                };
                let data: Vec<u8> = payload[2..]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                CdoRaw::DmaWrite { address, data }
            }

            CdoOpcode::MaskPoll if payload_len >= 3 => CdoRaw::MaskPoll {
                address: payload[0],
                mask: payload[1],
                expected: payload[2],
            },

            CdoOpcode::MaskPoll64 if payload_len >= 4 => CdoRaw::MaskPoll64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                mask: payload[2],
                expected: payload[3],
            },

            CdoOpcode::Delay if payload_len >= 1 => CdoRaw::Delay { cycles: payload[0] },

            CdoOpcode::Nop => CdoRaw::Nop {
                words: payload_len as u16,
            },

            CdoOpcode::Marker if payload_len >= 1 => CdoRaw::Marker { value: payload[0] },

            _ => CdoRaw::Unknown {
                opcode: (cmd_word & 0xFFFF) as u16,
                payload,
            },
        };

        Some(cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_conversion() {
        assert_eq!(CdoOpcode::from(0x103), CdoOpcode::Write);
        assert_eq!(CdoOpcode::from(0x105), CdoOpcode::DmaWrite);
        assert_eq!(CdoOpcode::from(0x111), CdoOpcode::Nop);
        assert!(matches!(CdoOpcode::from(0xFFF), CdoOpcode::Unknown(0xFFF)));
    }

    #[test]
    fn test_parse_minimal_cdo() {
        // Minimal valid CDO: header only, no commands
        let mut data = vec![0u8; 20];
        // num_words = 4
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        // ident = CDO
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        // version = 2.0
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        // length = 0
        data[12..16].copy_from_slice(&0u32.to_le_bytes());
        // checksum
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200 + 0);
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        let cdo = Cdo::parse(&data).unwrap();
        assert_eq!(cdo.magic(), "CDO");
        assert_eq!(cdo.version(), CdoVersion::V2_00);
        assert_eq!(cdo.command_length_words(), 0);
    }

    #[test]
    fn test_parse_cdo_with_write() {
        let mut data = vec![0u8; 32];
        // Header
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&3u32.to_le_bytes()); // 3 words of commands
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200 + 3);
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        // Command: WRITE (0x103) with payload_len=2
        let cmd_word: u32 = 0x0002_0103; // payload=2, opcode=0x103
        data[20..24].copy_from_slice(&cmd_word.to_le_bytes());
        data[24..28].copy_from_slice(&0x0020_0000u32.to_le_bytes()); // address
        data[28..32].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // value

        let cdo = Cdo::parse(&data).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 1);

        match &commands[0] {
            CdoRaw::Write { address, value } => {
                assert_eq!(*address, 0x0020_0000);
                assert_eq!(*value, 0xDEADBEEF);
            }
            _ => panic!("Expected Write command"),
        }
    }

    #[test]
    fn test_aie_address_decode() {
        // Address for tile (1, 2) with offset 0x1234
        let addr: u32 = (1 << 25) | (2 << 20) | 0x1234;
        let cmd = CdoRaw::Write {
            address: addr,
            value: 0,
        };
        let (col, row, offset) = cmd.decode_aie_address().unwrap();
        assert_eq!(col, 1);
        assert_eq!(row, 2);
        assert_eq!(offset, 0x1234);
    }

    #[test]
    fn test_parse_cdo_with_dma_write() {
        let mut data = vec![0u8; 44];
        // Header
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&6u32.to_le_bytes()); // 6 words: 1 cmd + 2 addr + 2 data + 1 nop
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200 + 6);
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        // DMA_WRITE command: opcode=0x105, payload_len=4 (addr_hi, addr_lo, 2 data words)
        // CDO v2 format: [addr_hi, addr_lo, data...]
        let cmd_word: u32 = 0x0004_0105;
        data[20..24].copy_from_slice(&cmd_word.to_le_bytes());
        data[24..28].copy_from_slice(&0u32.to_le_bytes());           // addr_hi = 0
        data[28..32].copy_from_slice(&0x0022_0000u32.to_le_bytes()); // addr_lo = tile(0,2) program mem
        data[32..36].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());  // data word 0
        data[36..40].copy_from_slice(&0xCAFEBABEu32.to_le_bytes());  // data word 1

        // NOP to fill out the 6 words
        data[40..44].copy_from_slice(&0x0000_0111u32.to_le_bytes());

        let cdo = Cdo::parse(&data).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 2);

        match &commands[0] {
            CdoRaw::DmaWrite { address, data } => {
                assert_eq!(*address, 0x0022_0000);
                assert_eq!(data.len(), 8);
                assert_eq!(data[0..4], [0xEF, 0xBE, 0xAD, 0xDE]); // little-endian
                assert_eq!(data[4..8], [0xBE, 0xBA, 0xFE, 0xCA]);
            }
            _ => panic!("Expected DmaWrite command"),
        }
    }

    #[test]
    fn test_parse_cdo_extended_length() {
        // Test CDO extended length: when inline payload_len == 0xFF,
        // the actual length is in the next word.
        // Build a DmaWrite with 260 payload words (> 254, triggers extended format).
        let payload_words = 260usize; // addr_hi + addr_lo + 258 data words
        let cdo_len = 1 + 1 + payload_words; // cmd_word + extended_len_word + payload
        let total_bytes = CDO_HEADER_SIZE + cdo_len * 4;
        let mut data = vec![0u8; total_bytes];

        // Header
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&(cdo_len as u32).to_le_bytes());
        let checksum = !(4u32.wrapping_add(CDO_MAGIC_CDO).wrapping_add(0x0200).wrapping_add(cdo_len as u32));
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        let mut off = CDO_HEADER_SIZE;

        // DMA_WRITE with extended length: inline_len=0xFF sentinel
        let cmd_word: u32 = 0x00FF_0105;
        data[off..off + 4].copy_from_slice(&cmd_word.to_le_bytes());
        off += 4;

        // Extended length word: actual payload = 260 words
        data[off..off + 4].copy_from_slice(&(payload_words as u32).to_le_bytes());
        off += 4;

        // payload[0] = addr_hi = 0
        data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        off += 4;

        // payload[1] = addr_lo = tile(0,2) program memory
        data[off..off + 4].copy_from_slice(&0x0022_0000u32.to_le_bytes());
        off += 4;

        // payload[2..260] = data (258 words of pattern)
        for i in 0..258u32 {
            data[off..off + 4].copy_from_slice(&i.to_le_bytes());
            off += 4;
        }

        let cdo = Cdo::parse(&data).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 1);

        match &commands[0] {
            CdoRaw::DmaWrite { address, data } => {
                assert_eq!(*address, 0x0022_0000);
                // 258 data words * 4 bytes = 1032 bytes
                assert_eq!(data.len(), 258 * 4);
                // First data word should be 0
                assert_eq!(data[0..4], 0u32.to_le_bytes());
                // Second data word should be 1
                assert_eq!(data[4..8], 1u32.to_le_bytes());
            }
            _ => panic!("Expected DmaWrite command"),
        }
    }

    #[test]
    fn test_parse_multiple_commands() {
        let mut data = vec![0u8; 40];
        // Header
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&5u32.to_le_bytes()); // 5 words
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200 + 5);
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        // NOP command (0x111, 0 payload)
        data[20..24].copy_from_slice(&0x0000_0111u32.to_le_bytes());

        // WRITE command (0x103, 2 payload)
        let cmd_word: u32 = 0x0002_0103;
        data[24..28].copy_from_slice(&cmd_word.to_le_bytes());
        data[28..32].copy_from_slice(&0x1234u32.to_le_bytes()); // address
        data[32..36].copy_from_slice(&0x5678u32.to_le_bytes()); // value

        // END_MARK (0x100, 0 payload)
        data[36..40].copy_from_slice(&0x0000_0100u32.to_le_bytes());

        let cdo = Cdo::parse(&data).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 3);

        assert!(matches!(commands[0], CdoRaw::Nop { words: 0 }));
        assert!(matches!(commands[1], CdoRaw::Write { address: 0x1234, value: 0x5678 }));
        assert!(matches!(commands[2], CdoRaw::EndMark));
    }
}
