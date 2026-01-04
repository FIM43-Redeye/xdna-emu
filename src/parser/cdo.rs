//! CDO (Configuration Data Object) parser
//!
//! CDO is AMD's format for device configuration commands. It contains
//! a sequence of register writes, DMA transfers, and control operations
//! that configure the AIE array.
//!
//! # Format Overview
//!
//! ```text
//! +------------------------------------------+
//! | Header (20 bytes)                        |
//! |   NumWords: 4                            |
//! |   IdentWord: "CDO\0" or "XLNX"           |
//! |   Version: e.g., 0x0200                  |
//! |   CDOLength: words (excluding header)    |
//! |   CheckSum: one's complement             |
//! +------------------------------------------+
//! | Command stream                           |
//! |   Each command: [len:16|opcode:16] + payload
//! +------------------------------------------+
//! ```
//!
//! # Example
//!
//! ```no_run
//! use xdna_emu::parser::Cdo;
//!
//! // CDO data typically comes from an XCLBIN's AIE Partition section
//! let cdo_bytes: Vec<u8> = std::fs::read("path/to/cdo.bin")?;
//! let cdo = Cdo::parse(&cdo_bytes)?;
//!
//! println!("CDO version: {:?}", cdo.version());
//! for cmd in cdo.commands() {
//!     println!("{:?}", cmd);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{anyhow, bail, Result};
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// CDO magic: "CDO\0" in little-endian
pub const CDO_MAGIC_CDO: u32 = 0x004F4443;
/// CDO magic: "XLNX" in little-endian
pub const CDO_MAGIC_XLNX: u32 = 0x584C4E58;

/// CDO header size in bytes
pub const CDO_HEADER_SIZE: usize = 20;

/// Scan for CDO magic within a buffer and return offset if found
pub fn find_cdo_offset(data: &[u8]) -> Option<usize> {
    // Search for CDO magic ("CDO\0" or "XLNX")
    let cdo_magic = CDO_MAGIC_CDO.to_le_bytes();
    let xlnx_magic = CDO_MAGIC_XLNX.to_le_bytes();

    // CDO header: NumWords (4) + IdentWord (4) + ...
    // So magic is at offset 4 in header
    for i in 0..data.len().saturating_sub(CDO_HEADER_SIZE) {
        if i + 8 <= data.len() {
            let word_at_4 = &data[i + 4..i + 8];
            if word_at_4 == cdo_magic || word_at_4 == xlnx_magic {
                // Verify it looks like a valid CDO header
                let num_words = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
                if num_words == 4 {
                    // Looks like a valid CDO header
                    return Some(i);
                }
            }
        }
    }
    None
}

/// CDO version values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdoVersion {
    /// Version 1.50 - CDOv2 format with CDOv1 NPI/CFU commands
    V1_50,
    /// Version 2.00 - CDOv2 format with SEM commands
    V2_00,
    /// Unknown version
    Unknown(u32),
}

impl From<u32> for CdoVersion {
    fn from(v: u32) -> Self {
        match v {
            0x0132 => Self::V1_50,
            0x0200 => Self::V2_00,
            other => Self::Unknown(other),
        }
    }
}

/// Raw CDO header (20 bytes)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawCdoHeader {
    /// Number of remaining words in header (usually 4)
    pub num_words: u32,
    /// Identification word: "CDO\0" or "XLNX"
    pub ident_word: u32,
    /// Format version
    pub version: u32,
    /// Length in 32-bit words (excluding header)
    pub cdo_length: u32,
    /// One's complement checksum of header fields
    pub checksum: u32,
}

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
pub enum CdoCommand {
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

impl CdoCommand {
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
        use crate::device::registers_spec::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK};
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
            // Warn but don't fail - some tools may not set checksum correctly
            eprintln!(
                "Warning: CDO checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
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
    pub fn commands(&self) -> CdoCommandIterator<'a> {
        CdoCommandIterator {
            data: self.command_data(),
            offset: 0,
        }
    }

    /// Count commands by type
    pub fn command_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for cmd in self.commands() {
            let name = match &cmd {
                CdoCommand::Write { .. } => "WRITE",
                CdoCommand::MaskWrite { .. } => "MASK_WRITE",
                CdoCommand::Write64 { .. } => "WRITE64",
                CdoCommand::MaskWrite64 { .. } => "MASK_WRITE64",
                CdoCommand::DmaWrite { .. } => "DMA_WRITE",
                CdoCommand::MaskPoll { .. } => "MASK_POLL",
                CdoCommand::MaskPoll64 { .. } => "MASK_POLL64",
                CdoCommand::Delay { .. } => "DELAY",
                CdoCommand::Nop { .. } => "NOP",
                CdoCommand::EndMark => "END_MARK",
                CdoCommand::Marker { .. } => "MARKER",
                CdoCommand::Unknown { opcode, .. } => {
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
pub struct CdoCommandIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for CdoCommandIterator<'a> {
    type Item = CdoCommand;

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
        let payload_len = (cmd_word >> 16) as usize;

        // Read payload words
        let payload_bytes = payload_len * 4;
        if self.offset + payload_bytes > self.data.len() {
            // Truncated command - return what we can
            self.offset = self.data.len();
            return Some(CdoCommand::Unknown {
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
            CdoOpcode::EndMark => CdoCommand::EndMark,

            CdoOpcode::Write if payload_len >= 2 => CdoCommand::Write {
                address: payload[0],
                value: payload[1],
            },

            CdoOpcode::MaskWrite if payload_len >= 3 => CdoCommand::MaskWrite {
                address: payload[0],
                mask: payload[1],
                value: payload[2],
            },

            CdoOpcode::Write64 if payload_len >= 3 => CdoCommand::Write64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                value: payload[2],
            },

            CdoOpcode::MaskWrite64 if payload_len >= 4 => CdoCommand::MaskWrite64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                mask: payload[2],
                value: payload[3],
            },

            CdoOpcode::DmaWrite if payload_len >= 2 => {
                let address = payload[0];

                // Handle embedded format (addr=0 means target is in payload[1])
                if address == 0 && payload_len >= 2 {
                    // Embedded format: [0, target_addr, data...]
                    // No separate byte_len - data length is (payload_len - 2) * 4
                    let target_addr = payload[1];
                    let data: Vec<u8> = payload[2..]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    CdoCommand::DmaWrite { address: target_addr, data }
                } else {
                    // Standard format: [addr, byte_len, data...]
                    let byte_len = payload[1] as usize;
                    let data: Vec<u8> = payload[2..]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .take(byte_len)
                        .collect();
                    CdoCommand::DmaWrite { address, data }
                }
            }

            CdoOpcode::MaskPoll if payload_len >= 3 => CdoCommand::MaskPoll {
                address: payload[0],
                mask: payload[1],
                expected: payload[2],
            },

            CdoOpcode::MaskPoll64 if payload_len >= 4 => CdoCommand::MaskPoll64 {
                address: ((payload[0] as u64) << 32) | (payload[1] as u64),
                mask: payload[2],
                expected: payload[3],
            },

            CdoOpcode::Delay if payload_len >= 1 => CdoCommand::Delay { cycles: payload[0] },

            CdoOpcode::Nop => CdoCommand::Nop {
                words: payload_len as u16,
            },

            CdoOpcode::Marker if payload_len >= 1 => CdoCommand::Marker { value: payload[0] },

            _ => CdoCommand::Unknown {
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
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<RawCdoHeader>(), 20);
    }

    #[test]
    fn test_magic_constants() {
        assert_eq!(CDO_MAGIC_CDO, 0x004F4443);
        assert_eq!(CDO_MAGIC_XLNX, 0x584C4E58);
        // "CDO\0" in bytes (little-endian)
        assert_eq!(&CDO_MAGIC_CDO.to_le_bytes(), b"CDO\0");
        // "XNLX" in bytes (little-endian) - note: constant name is historical
        assert_eq!(&CDO_MAGIC_XLNX.to_le_bytes(), b"XNLX");
    }

    #[test]
    fn test_version_conversion() {
        assert_eq!(CdoVersion::from(0x0132), CdoVersion::V1_50);
        assert_eq!(CdoVersion::from(0x0200), CdoVersion::V2_00);
        assert!(matches!(CdoVersion::from(0x9999), CdoVersion::Unknown(0x9999)));
    }

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
            CdoCommand::Write { address, value } => {
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
        let cmd = CdoCommand::Write {
            address: addr,
            value: 0,
        };
        let (col, row, offset) = cmd.decode_aie_address().unwrap();
        assert_eq!(col, 1);
        assert_eq!(row, 2);
        assert_eq!(offset, 0x1234);
    }

    #[test]
    fn test_find_cdo_offset() {
        // CDO header embedded at offset 100 in larger buffer
        let mut data = vec![0xFFu8; 200];

        // Write CDO header at offset 100
        let offset = 100;
        data[offset..offset + 4].copy_from_slice(&4u32.to_le_bytes());           // num_words
        data[offset + 4..offset + 8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes()); // ident
        data[offset + 8..offset + 12].copy_from_slice(&0x0200u32.to_le_bytes());  // version
        data[offset + 12..offset + 16].copy_from_slice(&0u32.to_le_bytes());      // length
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200);
        data[offset + 16..offset + 20].copy_from_slice(&checksum.to_le_bytes());  // checksum

        let found = find_cdo_offset(&data);
        assert_eq!(found, Some(100));
    }

    #[test]
    fn test_find_cdo_offset_xlnx() {
        // Test XLNX magic variant
        let mut data = vec![0u8; 50];
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_XLNX.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&0u32.to_le_bytes());

        assert_eq!(find_cdo_offset(&data), Some(0));
    }

    #[test]
    fn test_find_cdo_offset_not_found() {
        let data = vec![0xFFu8; 100];
        assert_eq!(find_cdo_offset(&data), None);
    }

    #[test]
    fn test_parse_cdo_with_dma_write() {
        let mut data = vec![0u8; 48];
        // Header
        data[0..4].copy_from_slice(&4u32.to_le_bytes());
        data[4..8].copy_from_slice(&CDO_MAGIC_CDO.to_le_bytes());
        data[8..12].copy_from_slice(&0x0200u32.to_le_bytes());
        data[12..16].copy_from_slice(&7u32.to_le_bytes()); // 7 words: 1 cmd + 2 header + 4 data
        let checksum = !(4 + CDO_MAGIC_CDO + 0x0200 + 7);
        data[16..20].copy_from_slice(&checksum.to_le_bytes());

        // DMA_WRITE command: opcode=0x105, payload_len=6 (addr, len, 4 data words)
        let cmd_word: u32 = 0x0006_0105;
        data[20..24].copy_from_slice(&cmd_word.to_le_bytes());
        data[24..28].copy_from_slice(&0x0002_0000u32.to_le_bytes()); // address (tile 0,2)
        data[28..32].copy_from_slice(&8u32.to_le_bytes());           // byte length
        data[32..36].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());  // data word 0
        data[36..40].copy_from_slice(&0xCAFEBABEu32.to_le_bytes());  // data word 1
        // Padding words
        data[40..44].copy_from_slice(&0u32.to_le_bytes());
        data[44..48].copy_from_slice(&0u32.to_le_bytes());

        let cdo = Cdo::parse(&data).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 1);

        match &commands[0] {
            CdoCommand::DmaWrite { address, data } => {
                assert_eq!(*address, 0x0002_0000);
                assert_eq!(data.len(), 8);
                assert_eq!(data[0..4], [0xEF, 0xBE, 0xAD, 0xDE]); // little-endian
                assert_eq!(data[4..8], [0xBE, 0xBA, 0xFE, 0xCA]);
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

        assert!(matches!(commands[0], CdoCommand::Nop { words: 0 }));
        assert!(matches!(commands[1], CdoCommand::Write { address: 0x1234, value: 0x5678 }));
        assert!(matches!(commands[2], CdoCommand::EndMark));
    }
}
