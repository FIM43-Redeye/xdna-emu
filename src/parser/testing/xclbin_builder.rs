//! Fluent builder for minimal-valid XCLBIN byte streams.
//!
//! Produces bytes accepted by [`crate::parser::Xclbin::from_file`]. Layout
//! (all little-endian) matches the existing parser in `xclbin.rs`:
//!
//! ```text
//! offset 0x000   magic = b"xclbin2\0"     (8 bytes)
//! offset 0x008   signature_length         (4 bytes, zero)
//! offset 0x00C   reserved                 (28 bytes, zero)
//! offset 0x028   keyblock                 (256 bytes, zero)
//! offset 0x128   unique_id                (8 bytes, zero)
//! offset 0x130   RawHeader                (152 bytes, num_sections set)
//! offset 0x1C8   section-headers table    (40 bytes per entry)
//! offset (past)  section payloads, back-to-back
//! ```
//!
//! Not covered: signed XCLBINs (signature_length non-zero, signature
//! payload at EOF), platform VBNV string, UUID-stable layouts for
//! cross-run comparison. For tests that care about those, use a real
//! XCLBIN fixture.
//!
//! The parser requires `Xclbin::from_file` (it memory-maps the file),
//! so the test round-trip here writes to a tempfile.

use crate::parser::xclbin::{SectionKind, XCLBIN_MAGIC};

const HEADER_OFFSET: usize = 0x130;
const SECTIONS_OFFSET: usize = 0x1C8;
const RAW_HEADER_SIZE: usize = 152;
const SECTION_HEADER_SIZE: usize = 40;

/// Builds a byte stream that parses as a valid XCLBIN.
///
/// ```ignore
/// let bytes = XclbinBuilder::new()
///     .with_partition(cdo_bytes)
///     .build();
/// // write to tempfile, then Xclbin::from_file(path)
/// ```
pub struct XclbinBuilder {
    sections: Vec<(u32, Vec<u8>)>,
}

impl XclbinBuilder {
    /// New builder with no sections.
    pub fn new() -> Self {
        Self { sections: Vec::new() }
    }

    /// Add an AIE_PARTITION section with the given bytes.
    pub fn with_partition(mut self, bytes: Vec<u8>) -> Self {
        // SectionKind is non-unit-only (has Unknown(u32)), so we can't
        // `as u32`-cast the variant. Use the literal discriminant from
        // `xclbin.rs` (AiePartition = 32).
        self.sections.push((32, bytes));
        self
    }

    /// Add an arbitrary section with the given kind discriminant and data.
    pub fn with_section(mut self, kind: u32, bytes: Vec<u8>) -> Self {
        self.sections.push((kind, bytes));
        self
    }

    /// Finalize: produce the byte stream.
    pub fn build(self) -> Vec<u8> {
        let num_sections = self.sections.len();
        let sections_table_size = num_sections * SECTION_HEADER_SIZE;
        let payload_offset = SECTIONS_OFFSET + sections_table_size;

        // Layout: assign each section a back-to-back offset after the table.
        let mut section_offsets = Vec::with_capacity(num_sections);
        let mut cursor = payload_offset;
        for (_, data) in &self.sections {
            section_offsets.push(cursor);
            cursor += data.len();
        }
        let total_size = cursor;

        let mut out = vec![0u8; total_size];

        // Magic
        out[0..8].copy_from_slice(&XCLBIN_MAGIC);
        // signature_length, reserved, keyblock, unique_id all stay zero.

        // RawHeader at 0x130. Most fields are allowed to be zero; we only
        // need `length` and `num_sections` (the latter is at byte offset
        // 128 + 16 = 144 within RawHeader, i.e. absolute 0x130 + 144 =
        // 0x1C0).
        //
        // RawHeader layout:
        //   0   u64  length
        //   8   u64  time_stamp
        //   16  u64  feature_rom_time_stamp
        //   24  u16  version_patch
        //   26  u8   version_major
        //   27  u8   version_minor
        //   28  u16  mode
        //   30  u16  action_mask
        //   32  [u8;16]  interface_uuid
        //   48  [u8;64]  platform_vbnv
        //   112 [u8;16]  xclbin_uuid
        //   128 [u8;16]  debug_bin
        //   144 u32  num_sections
        //   148 u32  _padding
        let h = HEADER_OFFSET;
        // length
        out[h..h + 8].copy_from_slice(&(total_size as u64).to_le_bytes());
        // num_sections
        out[h + 144..h + 148].copy_from_slice(&(num_sections as u32).to_le_bytes());

        // Section headers at 0x1C8. RawSectionHeader layout (40 bytes):
        //   0   u32  section_kind
        //   4   [u8;16]  section_name
        //   20  u32  _padding (implicit, for u64 alignment)
        //   24  u64  section_offset
        //   32  u64  section_size
        for (i, ((kind, data), &offset)) in self.sections.iter().zip(section_offsets.iter()).enumerate() {
            let sh = SECTIONS_OFFSET + i * SECTION_HEADER_SIZE;
            out[sh..sh + 4].copy_from_slice(&kind.to_le_bytes());
            // section_name: leave zero (null-terminated empty string).
            out[sh + 24..sh + 32].copy_from_slice(&(offset as u64).to_le_bytes());
            out[sh + 32..sh + 40].copy_from_slice(&(data.len() as u64).to_le_bytes());

            // Payload
            out[offset..offset + data.len()].copy_from_slice(data);
        }

        out
    }

    /// Convenience: build + write to a tempfile, returning the handle so
    /// it lives as long as the parsed `Xclbin`.
    #[cfg(test)]
    pub fn build_to_tempfile(self) -> std::io::Result<tempfile::NamedTempFile> {
        use std::io::Write;
        let bytes = self.build();
        let mut tmp = tempfile::NamedTempFile::new()?;
        tmp.write_all(&bytes)?;
        tmp.flush()?;
        Ok(tmp)
    }
}

impl Default for XclbinBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Size-assertion: if RawHeader ever changes layout, this file has the
// offsets wrong. Keep the constant in sync.
const _: () = assert!(
    RAW_HEADER_SIZE == 152,
    "XclbinBuilder assumes RawHeader is 152 bytes -- update offsets if changed"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Xclbin;

    #[test]
    fn empty_builder_parses() {
        let tmp = XclbinBuilder::new().build_to_tempfile().unwrap();
        let xclbin = Xclbin::from_file(tmp.path()).expect("empty XCLBIN should parse");
        assert_eq!(xclbin.num_sections(), 0);
        assert_eq!(xclbin.sections().count(), 0);
    }

    #[test]
    fn round_trip_single_partition() {
        let payload: Vec<u8> = (0u8..64).collect();
        let tmp = XclbinBuilder::new()
            .with_partition(payload.clone())
            .build_to_tempfile()
            .unwrap();
        let xclbin = Xclbin::from_file(tmp.path()).unwrap();

        assert_eq!(xclbin.num_sections(), 1);
        let sections: Vec<_> = xclbin.sections().collect();
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].kind, SectionKind::AiePartition);
        assert_eq!(sections[0].data, &payload[..]);

        let partition = xclbin.aie_partition().unwrap();
        assert_eq!(partition.data, &payload[..]);
    }

    #[test]
    fn round_trip_two_sections_different_kinds() {
        let metadata = b"{\"ok\":true}".to_vec();
        let partition: Vec<u8> = vec![0xAA; 32];
        // SectionKind can't `as u32`-cast (non-unit-only); use literal
        // 25 = AieMetadata per xclbin.rs.
        let tmp = XclbinBuilder::new()
            .with_section(25, metadata.clone())
            .with_partition(partition.clone())
            .build_to_tempfile()
            .unwrap();
        let xclbin = Xclbin::from_file(tmp.path()).unwrap();

        assert_eq!(xclbin.num_sections(), 2);

        let meta = xclbin.aie_metadata().unwrap();
        assert_eq!(meta.data, &metadata[..]);

        let part = xclbin.aie_partition().unwrap();
        assert_eq!(part.data, &partition[..]);
    }
}
