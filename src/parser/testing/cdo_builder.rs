//! Fluent builder for minimal-valid CDO byte streams.
//!
//! Produces bytes accepted by [`crate::parser::cdo::Cdo::parse`]. Handles
//! checksum, cdo_length (in 32-bit words), and little-endian encoding
//! automatically. The command stream is a simple sequence of
//! `[payload_len:16 | opcode:16]`-framed entries; see
//! [`crate::parser::cdo::framing`] for the format.
//!
//! Not covered: CDOv2 extended-length commands (payload > 254 words).
//! Ordinary `with_dma_write` payloads stay under that limit in practice;
//! if a test needs the 0xFF sentinel path, extend the builder.

use crate::parser::cdo::framing::{CDO_HEADER_SIZE, CDO_MAGIC_CDO};

// CDO opcode discriminants. CdoOpcode is non-unit-only (has Unknown(u16)),
// so `as u16` casting is not permitted; mirror the values from syntax.rs.
const OP_END_MARK: u16 = 0x100;
const OP_MASK_WRITE: u16 = 0x102;
const OP_WRITE: u16 = 0x103;
const OP_DELAY: u16 = 0x104;
const OP_DMA_WRITE: u16 = 0x105;
const OP_MARKER: u16 = 0x119;

/// Builds a byte stream that parses as a valid CDO container.
///
/// ```ignore
/// let bytes = CdoBuilder::new()
///     .with_write32(0x0020_0000, 0xDEADBEEF)
///     .with_end_mark()
///     .build();
/// let cdo = Cdo::parse(&bytes).unwrap();
/// ```
pub struct CdoBuilder {
    version: u32,
    ident: u32,
    /// Raw 32-bit command words appended so far.
    command_words: Vec<u32>,
}

impl CdoBuilder {
    /// New builder with defaults: ident = "CDO\0", version = 0x0200 (V2.00).
    pub fn new() -> Self {
        Self { version: 0x0200, ident: CDO_MAGIC_CDO, command_words: Vec::new() }
    }

    /// Override the version field (useful for Unknown-version negative tests).
    pub fn with_version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Override the identification word (e.g., `CDO_MAGIC_XLNX`, or an
    /// invalid value for bad-magic negative tests).
    pub fn with_ident(mut self, ident: u32) -> Self {
        self.ident = ident;
        self
    }

    /// Emit a WRITE command: `*address = value`.
    pub fn with_write32(mut self, address: u32, value: u32) -> Self {
        self.push_cmd(OP_WRITE, &[address, value]);
        self
    }

    /// Emit a MASK_WRITE command: `*addr = (*addr & ~mask) | (value & mask)`.
    pub fn with_mask_write32(mut self, address: u32, mask: u32, value: u32) -> Self {
        self.push_cmd(OP_MASK_WRITE, &[address, mask, value]);
        self
    }

    /// Emit a DMA_WRITE command with CDOv2 `[addr_hi, addr_lo, data...]`
    /// framing. `data` is zero-padded to a 4-byte boundary.
    pub fn with_dma_write(mut self, address: u32, data: Vec<u8>) -> Self {
        let mut padded = data;
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        let mut payload = Vec::with_capacity(2 + padded.len() / 4);
        payload.push(0u32); // addr_hi
        payload.push(address); // addr_lo
        for chunk in padded.chunks_exact(4) {
            payload.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        self.push_cmd(OP_DMA_WRITE, &payload);
        self
    }

    /// Emit a MARKER command with the given tag value.
    pub fn with_marker(mut self, value: u32) -> Self {
        self.push_cmd(OP_MARKER, &[value]);
        self
    }

    /// Emit a DELAY command (cycles).
    pub fn with_delay(mut self, cycles: u32) -> Self {
        self.push_cmd(OP_DELAY, &[cycles]);
        self
    }

    /// Emit an END_MARK command.
    pub fn with_end_mark(mut self) -> Self {
        self.push_cmd(OP_END_MARK, &[]);
        self
    }

    /// Finalize: produce the byte stream with a correctly-populated header
    /// (magic, version, `cdo_length`, one's-complement checksum) followed
    /// by the command stream.
    pub fn build(self) -> Vec<u8> {
        let cdo_len_words = self.command_words.len() as u32;
        let num_words = 4u32;
        let checksum = !(num_words
            .wrapping_add(self.ident)
            .wrapping_add(self.version)
            .wrapping_add(cdo_len_words));

        let mut out = Vec::with_capacity(CDO_HEADER_SIZE + self.command_words.len() * 4);
        out.extend_from_slice(&num_words.to_le_bytes());
        out.extend_from_slice(&self.ident.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&cdo_len_words.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        for word in &self.command_words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Push a command: first word is `[payload_len:8 | 0:8 | opcode:16]`,
    /// followed by payload words. `payload_len` must fit in 8 bits
    /// (< 0xFF); this builder does not emit the extended-length sentinel.
    fn push_cmd(&mut self, opcode: u16, payload: &[u32]) {
        let payload_len = payload.len() as u32;
        debug_assert!(payload_len < 0xFF, "CdoBuilder does not emit CDOv2 extended-length (payload >= 0xFF)");
        let cmd_word = (payload_len << 16) | opcode as u32;
        self.command_words.push(cmd_word);
        self.command_words.extend_from_slice(payload);
    }
}

impl Default for CdoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::cdo::framing::CdoVersion;
    use crate::parser::cdo::syntax::{Cdo, CdoRaw};

    #[test]
    fn empty_builder_parses() {
        let bytes = CdoBuilder::new().build();
        let cdo = Cdo::parse(&bytes).expect("minimal CDO should parse");
        assert_eq!(cdo.version(), CdoVersion::V2_00);
        assert_eq!(cdo.command_length_words(), 0);
        assert_eq!(cdo.commands().count(), 0);
    }

    #[test]
    fn round_trip_write_and_end_mark() {
        let bytes = CdoBuilder::new().with_write32(0x0020_0000, 0xDEADBEEF).with_end_mark().build();

        let cdo = Cdo::parse(&bytes).expect("builder should produce a parseable CDO");
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 2);

        match &commands[0] {
            CdoRaw::Write { address, value } => {
                assert_eq!(*address, 0x0020_0000);
                assert_eq!(*value, 0xDEADBEEF);
            }
            other => panic!("expected Write, got {other:?}"),
        }
        assert!(matches!(commands[1], CdoRaw::EndMark));
    }

    #[test]
    fn round_trip_mask_write_dma_marker_delay() {
        let bytes = CdoBuilder::new()
            .with_mask_write32(0x1000, 0x0000_00FF, 0x42)
            .with_dma_write(0x0022_0000, vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
            .with_marker(0xCAFE)
            .with_delay(100)
            .with_end_mark()
            .build();

        let cdo = Cdo::parse(&bytes).unwrap();
        let commands: Vec<_> = cdo.commands().collect();
        assert_eq!(commands.len(), 5);

        assert!(matches!(commands[0], CdoRaw::MaskWrite { address: 0x1000, mask: 0x0000_00FF, value: 0x42 }));
        match &commands[1] {
            CdoRaw::DmaWrite { address, data } => {
                assert_eq!(*address, 0x0022_0000);
                assert_eq!(data.len(), 8);
                assert_eq!(data[0], 0x11);
                assert_eq!(data[7], 0x88);
            }
            other => panic!("expected DmaWrite, got {other:?}"),
        }
        assert!(matches!(commands[2], CdoRaw::Marker { value: 0xCAFE }));
        assert!(matches!(commands[3], CdoRaw::Delay { cycles: 100 }));
        assert!(matches!(commands[4], CdoRaw::EndMark));
    }

    #[test]
    fn unknown_version_still_parses() {
        let bytes = CdoBuilder::new().with_version(0x9999).build();
        let cdo = Cdo::parse(&bytes).unwrap();
        assert!(matches!(cdo.version(), CdoVersion::Unknown(0x9999)));
    }
}
