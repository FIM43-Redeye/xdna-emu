//! XCLBIN container format parser
//!
//! XCLBIN is AMD/Xilinx's container format for FPGA and NPU binaries.
//! See docs/formats/xclbin.md for format specification.
//!
//! # Example
//! ```no_run
//! use xdna_emu::parser::xclbin::Xclbin;
//!
//! let xclbin = Xclbin::from_file("path/to/binary.xclbin")?;
//! for section in xclbin.sections() {
//!     println!("{}: {} bytes", section.name(), section.size());
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::Path;
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// Magic bytes for XCLBIN format: "xclbin2\0"
pub const XCLBIN_MAGIC: [u8; 8] = *b"xclbin2\0";

/// Section type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SectionKind {
    Bitstream = 0,
    ClearingBitstream = 1,
    EmbeddedMetadata = 2,
    Firmware = 3,
    DebugData = 4,
    SchedFirmware = 5,
    MemTopology = 6,
    Connectivity = 7,
    IpLayout = 8,
    DebugIpLayout = 9,
    DesignCheckPoint = 10,
    ClockFreqTopology = 11,
    Mcs = 12,
    Bmc = 13,
    BuildMetadata = 14,
    KeyvalueMetadata = 15,
    UserMetadata = 16,
    DnaCertificate = 17,
    Pdi = 18,
    BitstreamPartialPdi = 19,
    PartitionMetadata = 20,
    EmulationData = 21,
    SystemMetadata = 22,
    SoftKernel = 23,
    AskFlash = 24,
    AieMetadata = 25,
    AskGroupTopology = 26,
    AskGroupConnectivity = 27,
    SmartNic = 28,
    AieResources = 29,
    Overlay = 30,
    VenderMetadata = 31,
    AiePartition = 32,
    IpMetadata = 33,
    AieResourcesBin = 34,
    AieTraceMetadata = 35,
    Unknown(u32),
}

impl From<u32> for SectionKind {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Bitstream,
            1 => Self::ClearingBitstream,
            2 => Self::EmbeddedMetadata,
            3 => Self::Firmware,
            4 => Self::DebugData,
            5 => Self::SchedFirmware,
            6 => Self::MemTopology,
            7 => Self::Connectivity,
            8 => Self::IpLayout,
            9 => Self::DebugIpLayout,
            10 => Self::DesignCheckPoint,
            11 => Self::ClockFreqTopology,
            12 => Self::Mcs,
            13 => Self::Bmc,
            14 => Self::BuildMetadata,
            15 => Self::KeyvalueMetadata,
            16 => Self::UserMetadata,
            17 => Self::DnaCertificate,
            18 => Self::Pdi,
            19 => Self::BitstreamPartialPdi,
            20 => Self::PartitionMetadata,
            21 => Self::EmulationData,
            22 => Self::SystemMetadata,
            23 => Self::SoftKernel,
            24 => Self::AskFlash,
            25 => Self::AieMetadata,
            26 => Self::AskGroupTopology,
            27 => Self::AskGroupConnectivity,
            28 => Self::SmartNic,
            29 => Self::AieResources,
            30 => Self::Overlay,
            31 => Self::VenderMetadata,
            32 => Self::AiePartition,
            33 => Self::IpMetadata,
            34 => Self::AieResourcesBin,
            35 => Self::AieTraceMetadata,
            other => Self::Unknown(other),
        }
    }
}

/// Raw XCLBIN header structure (152 bytes)
/// Matches `struct axlf_header` from xclbin.h
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawHeader {
    pub length: u64,
    pub time_stamp: u64,
    pub feature_rom_time_stamp: u64,
    pub version_patch: u16,
    pub version_major: u8,
    pub version_minor: u8,
    pub mode: u16,
    pub action_mask: u16,
    pub interface_uuid: [u8; 16],
    pub platform_vbnv: [u8; 64],
    pub xclbin_uuid: [u8; 16],
    pub debug_bin: [u8; 16],
    pub num_sections: u32,
    // 4 bytes padding for 8-byte alignment (so section headers align properly)
    _padding: u32,
}

/// Raw section header structure (40 bytes)
/// Matches `struct axlf_section_header` from xclbin.h
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawSectionHeader {
    pub section_kind: u32,
    pub section_name: [u8; 16],
    // 4 bytes implicit padding here for u64 alignment
    pub section_offset: u64,
    pub section_size: u64,
}

/// Parsed section with reference to underlying data
#[derive(Debug)]
pub struct Section<'a> {
    pub kind: SectionKind,
    pub name: String,
    pub offset: u64,
    pub data: &'a [u8],
}

impl<'a> Section<'a> {
    /// Returns the section name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the section size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the raw section data
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

/// Parsed XCLBIN file
pub struct Xclbin {
    /// Memory-mapped file data
    _mmap: Mmap,
    /// Raw pointer to mapped data (valid for lifetime of mmap)
    data: *const u8,
    data_len: usize,
    /// Parsed header
    pub header: RawHeader,
    /// Section headers
    section_headers: Vec<RawSectionHeader>,
}

// Safety: Xclbin only contains the mmap and derived pointers
// The mmap is Send+Sync, and we only read from the data
unsafe impl Send for Xclbin {}
unsafe impl Sync for Xclbin {}

impl Xclbin {
    /// Load and parse an XCLBIN file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        // Memory-map the file for efficient access
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap {}", path.display()))?;

        Self::from_mmap(mmap)
    }

    /// Parse from a memory-mapped buffer
    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let data = mmap.as_ptr();
        let data_len = mmap.len();

        // Need at least magic + signature_length + reserved + keyblock + unique_id + header
        const MIN_SIZE: usize = 8 + 4 + 28 + 256 + 8 + 152;
        if data_len < MIN_SIZE {
            bail!("File too small: {} bytes (minimum {})", data_len, MIN_SIZE);
        }

        let data_slice = &mmap[..];

        // Validate magic
        if data_slice[0..8] != XCLBIN_MAGIC {
            bail!(
                "Invalid magic: expected {:?}, got {:?}",
                XCLBIN_MAGIC,
                &data_slice[0..8]
            );
        }

        // Parse header (starts at offset 0x130 = 304)
        const HEADER_OFFSET: usize = 0x130;
        let (header, _) = RawHeader::read_from_prefix(&data_slice[HEADER_OFFSET..])
            .map_err(|e| anyhow!("Failed to parse header: {:?}", e))?;

        // Validate section count
        let num_sections = header.num_sections as usize;
        if num_sections > 0x10000 {
            bail!("Invalid section count: {}", num_sections);
        }

        // Parse section headers (start at offset 0x1C8 = 456)
        const SECTIONS_OFFSET: usize = 0x1C8;
        let mut section_headers = Vec::with_capacity(num_sections);

        for i in 0..num_sections {
            let offset = SECTIONS_OFFSET + i * 40;
            if offset + 40 > data_len {
                bail!("Section header {} extends past end of file", i);
            }

            let (section, _) = RawSectionHeader::read_from_prefix(&data_slice[offset..])
                .map_err(|e| anyhow!("Failed to parse section header {}: {:?}", i, e))?;

            section_headers.push(section);
        }

        Ok(Self {
            _mmap: mmap,
            data,
            data_len,
            header,
            section_headers,
        })
    }

    /// Get the underlying data slice
    fn data(&self) -> &[u8] {
        // Safety: data pointer is valid for the lifetime of self._mmap
        unsafe { std::slice::from_raw_parts(self.data, self.data_len) }
    }

    /// Returns the total file length
    pub fn length(&self) -> u64 {
        self.header.length
    }

    /// Returns the XCLBIN UUID
    pub fn uuid(&self) -> uuid::Uuid {
        uuid::Uuid::from_bytes(self.header.xclbin_uuid)
    }

    /// Returns the platform string (e.g., "xilinx:npu1:...")
    pub fn platform(&self) -> String {
        let bytes = &self.header.platform_vbnv;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }

    /// Returns the number of sections
    pub fn num_sections(&self) -> usize {
        self.section_headers.len()
    }

    /// Iterate over all sections
    pub fn sections(&self) -> impl Iterator<Item = Section<'_>> {
        let data = self.data();
        self.section_headers.iter().map(move |hdr| {
            let offset = hdr.section_offset as usize;
            let size = hdr.section_size as usize;
            let end = (offset + size).min(data.len());
            let section_data = if offset < data.len() {
                &data[offset..end]
            } else {
                &[]
            };

            // Parse name (null-terminated)
            let name_end = hdr
                .section_name
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(16);
            let name = String::from_utf8_lossy(&hdr.section_name[..name_end]).into_owned();

            Section {
                kind: SectionKind::from(hdr.section_kind),
                name,
                offset: hdr.section_offset,
                data: section_data,
            }
        })
    }

    /// Find a section by kind
    pub fn find_section(&self, kind: SectionKind) -> Option<Section<'_>> {
        self.sections().find(|s| s.kind == kind)
    }

    /// Find all sections of a given kind
    pub fn find_sections(&self, kind: SectionKind) -> Vec<Section<'_>> {
        self.sections().filter(|s| s.kind == kind).collect()
    }

    /// Get the AIE partition section (contains PDI/CDO for NPU)
    pub fn aie_partition(&self) -> Option<Section<'_>> {
        self.find_section(SectionKind::AiePartition)
    }

    /// Get the AIE metadata section (JSON)
    pub fn aie_metadata(&self) -> Option<Section<'_>> {
        self.find_section(SectionKind::AieMetadata)
    }

    /// Print a summary of the XCLBIN contents
    pub fn print_summary(&self) {
        println!("XCLBIN Summary");
        println!("==============");
        println!("UUID: {}", self.uuid());
        println!("Platform: {}", self.platform());
        println!(
            "Version: {}.{}.{}",
            self.header.version_major, self.header.version_minor, self.header.version_patch
        );
        println!("Total size: {} bytes", self.length());
        println!("Sections: {}", self.num_sections());
        println!();

        for (i, section) in self.sections().enumerate() {
            println!(
                "  [{:2}] {:?} \"{}\" @ 0x{:x}, {} bytes",
                i,
                section.kind,
                section.name,
                section.offset,
                section.size()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constant() {
        assert_eq!(&XCLBIN_MAGIC, b"xclbin2\0");
    }

    #[test]
    fn test_section_kind_conversion() {
        assert_eq!(SectionKind::from(6), SectionKind::MemTopology);
        assert_eq!(SectionKind::from(32), SectionKind::AiePartition);
        assert!(matches!(SectionKind::from(999), SectionKind::Unknown(999)));
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<RawHeader>(), 152);
    }

    #[test]
    fn test_section_header_size() {
        // 4 (kind) + 16 (name) + 4 (padding) + 8 (offset) + 8 (size) = 40
        assert_eq!(std::mem::size_of::<RawSectionHeader>(), 40);
    }

    // Integration test with real XCLBIN file
    #[test]
    fn test_parse_real_xclbin() {
        let test_xclbin = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";

        if !std::path::Path::new(test_xclbin).exists() {
            eprintln!("Skipping real XCLBIN test: file not found");
            return;
        }

        let xclbin = Xclbin::from_file(test_xclbin).unwrap();

        // Should have valid UUID
        let uuid = xclbin.uuid();
        assert!(!uuid.is_nil());

        // Should have at least one section
        assert!(xclbin.num_sections() > 0);

        // Should have AIE Partition section for NPU binaries
        let aie_partition = xclbin.aie_partition();
        assert!(aie_partition.is_some(), "Expected AIE_PARTITION section");

        let partition = aie_partition.unwrap();
        assert_eq!(partition.kind, SectionKind::AiePartition);
        assert!(partition.size() > 0);

        // Verify we can iterate all sections without panicking
        let sections: Vec<_> = xclbin.sections().collect();
        assert!(!sections.is_empty());

        // All sections should have valid offsets (within file bounds)
        let file_len = xclbin.length();
        for section in &sections {
            assert!(
                section.offset + section.size() as u64 <= file_len,
                "Section {} extends past file end",
                section.name()
            );
        }
    }

    #[test]
    fn test_xclbin_sections_types() {
        let test_xclbin = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";

        if !std::path::Path::new(test_xclbin).exists() {
            return;
        }

        let xclbin = Xclbin::from_file(test_xclbin).unwrap();

        // Collect all section kinds we see
        let kinds: Vec<_> = xclbin.sections().map(|s| s.kind).collect();

        // NPU xclbins typically have MEM_TOPOLOGY and AIE_PARTITION
        assert!(
            kinds.contains(&SectionKind::MemTopology) || kinds.contains(&SectionKind::AiePartition),
            "Expected at least MEM_TOPOLOGY or AIE_PARTITION section"
        );
    }
}
