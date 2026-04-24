//! Framing: byte-level CDO container parsing (header, version, command
//! length frames). Independent of command opcode meaning.
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
}
