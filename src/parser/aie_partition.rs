//! AIE Partition section parser
//!
//! The AIE Partition section in an XCLBIN contains the PDI/CDO data
//! needed to configure the AIE array. This module parses the nested
//! structure to extract the actual CDO commands.
//!
//! # Structure Hierarchy
//!
//! ```text
//! aie_partition (184 bytes)
//!   └── aie_partition_info (88 bytes)
//!   └── aie_pdi[] array
//!         ├── pdi_image (raw CDO bytes)
//!         └── cdo_groups[] array
//! ```

use anyhow::{anyhow, bail, Result};
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// Array offset reference (size + offset from section start)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct ArrayOffset {
    pub count: u32,   // Number of elements
    pub offset: u32,  // Byte offset from section start
}

/// CDO type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CdoType {
    Unknown = 0,
    Primary = 1,
    Lite = 2,
    PrePost = 3,
}

impl From<u8> for CdoType {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Primary,
            2 => Self::Lite,
            3 => Self::PrePost,
            _ => Self::Unknown,
        }
    }
}

/// CDO group metadata (96 bytes)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawCdoGroup {
    pub mpo_name: u32,           // Offset to name string
    pub cdo_type: u8,            // CDO_Type enum
    pub padding: [u8; 3],
    pub pdi_id: u64,             // PDI ID
    pub dpu_kernel_ids: ArrayOffset,  // Array of kernel IDs
    pub pre_cdo_groups: ArrayOffset,  // Array of pre-CDO group IDs
    pub reserved: [u8; 64],
}

/// AIE PDI container (96 bytes)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawAiePdi {
    pub uuid: [u8; 16],          // PDI container UUID
    pub pdi_image: ArrayOffset,  // PDI/CDO image data
    pub cdo_groups: ArrayOffset, // Array of cdo_group
    pub reserved: [u8; 64],
}

/// AIE partition info (88 bytes)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawAiePartitionInfo {
    pub column_width: u16,           // Width of partition in columns
    pub padding: [u8; 6],
    pub start_columns: ArrayOffset,  // Array of start column IDs (u16)
    pub reserved: [u8; 72],
}

/// AIE partition header (184 bytes)
#[derive(Debug, Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct RawAiePartition {
    pub schema_version: u8,
    pub padding0: [u8; 3],
    pub mpo_name: u32,               // Offset to name string
    pub operations_per_cycle: u32,   // For TOPS calculation
    pub padding1: [u8; 4],
    pub inference_fingerprint: u64,  // Hash of inference function
    pub pre_post_fingerprint: u64,   // Hash of pre/post processing
    pub info: RawAiePartitionInfo,   // Partition info (88 bytes)
    pub aie_pdi: ArrayOffset,        // Array of aie_pdi
    pub kernel_commit_id: u32,       // Git commit ID offset
    pub reserved: [u8; 52],
}

/// Parsed AIE PDI with resolved data references
#[derive(Debug)]
pub struct AiePdi<'a> {
    pub uuid: uuid::Uuid,
    pub pdi_image: &'a [u8],
    pub cdo_type: CdoType,
}

/// Parsed AIE Partition section
#[derive(Debug)]
pub struct AiePartition<'a> {
    data: &'a [u8],
    pub header: RawAiePartition,
}

impl<'a> AiePartition<'a> {
    /// Parse an AIE Partition section from raw bytes
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < std::mem::size_of::<RawAiePartition>() {
            bail!(
                "AIE Partition too small: {} bytes (need {})",
                data.len(),
                std::mem::size_of::<RawAiePartition>()
            );
        }

        let (header, _) = RawAiePartition::read_from_prefix(data)
            .map_err(|e| anyhow!("Failed to parse AIE partition header: {:?}", e))?;

        Ok(Self { data, header })
    }

    /// Get the partition name (if any)
    pub fn name(&self) -> Option<String> {
        if self.header.mpo_name == 0 {
            return None;
        }
        let offset = self.header.mpo_name as usize;
        if offset >= self.data.len() {
            return None;
        }
        let name_bytes = &self.data[offset..];
        let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_bytes.len());
        Some(String::from_utf8_lossy(&name_bytes[..end]).into_owned())
    }

    /// Get the column width
    pub fn column_width(&self) -> u16 {
        self.header.info.column_width
    }

    /// Get start columns for this partition
    pub fn start_columns(&self) -> Vec<u16> {
        let arr = &self.header.info.start_columns;
        if arr.count == 0 || arr.offset == 0 {
            return Vec::new();
        }

        let offset = arr.offset as usize;
        let count = arr.count as usize;

        if offset + count * 2 > self.data.len() {
            return Vec::new();
        }

        (0..count)
            .filter_map(|i| {
                let start = offset + i * 2;
                if start + 2 <= self.data.len() {
                    Some(u16::from_le_bytes([self.data[start], self.data[start + 1]]))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate over PDI containers in this partition
    pub fn pdis(&self) -> impl Iterator<Item = AiePdi<'a>> {
        let arr = &self.header.aie_pdi;
        let pdi_count = arr.count as usize;
        let pdi_offset = arr.offset as usize;
        let data = self.data;

        (0..pdi_count).filter_map(move |i| {
            let offset = pdi_offset + i * std::mem::size_of::<RawAiePdi>();
            if offset + std::mem::size_of::<RawAiePdi>() > data.len() {
                return None;
            }

            let (raw_pdi, _) = RawAiePdi::read_from_prefix(&data[offset..]).ok()?;

            // Get PDI image data
            let img_offset = raw_pdi.pdi_image.offset as usize;
            let img_size = raw_pdi.pdi_image.count as usize;

            if img_offset + img_size > data.len() {
                return None;
            }

            let pdi_image = &data[img_offset..img_offset + img_size];

            // Get CDO type from first cdo_group if available
            let cdo_type = if raw_pdi.cdo_groups.count > 0 {
                let grp_offset = raw_pdi.cdo_groups.offset as usize;
                if grp_offset + std::mem::size_of::<RawCdoGroup>() <= data.len() {
                    if let Ok((grp, _)) = RawCdoGroup::read_from_prefix(&data[grp_offset..]) {
                        CdoType::from(grp.cdo_type)
                    } else {
                        CdoType::Unknown
                    }
                } else {
                    CdoType::Unknown
                }
            } else {
                CdoType::Unknown
            };

            Some(AiePdi {
                uuid: uuid::Uuid::from_bytes(raw_pdi.uuid),
                pdi_image,
                cdo_type,
            })
        })
    }

    /// Get the first (primary) PDI image
    pub fn primary_pdi(&self) -> Option<AiePdi<'a>> {
        self.pdis().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<ArrayOffset>(), 8);
        assert_eq!(std::mem::size_of::<RawCdoGroup>(), 96);
        assert_eq!(std::mem::size_of::<RawAiePdi>(), 96);
        assert_eq!(std::mem::size_of::<RawAiePartitionInfo>(), 88);
        assert_eq!(std::mem::size_of::<RawAiePartition>(), 184);
    }

    #[test]
    fn test_cdo_type_conversion() {
        assert_eq!(CdoType::from(0), CdoType::Unknown);
        assert_eq!(CdoType::from(1), CdoType::Primary);
        assert_eq!(CdoType::from(2), CdoType::Lite);
        assert_eq!(CdoType::from(3), CdoType::PrePost);
        assert_eq!(CdoType::from(99), CdoType::Unknown);
    }

    #[test]
    fn test_parse_minimal_partition() {
        // Create a minimal valid partition (header only, no PDIs)
        let mut data = vec![0u8; 256];

        // schema_version = 0
        data[0] = 0;
        // mpo_name = 0 (no name)
        // operations_per_cycle = 1000
        data[8..12].copy_from_slice(&1000u32.to_le_bytes());
        // column_width = 4 (at offset 0x20 in aie_partition_info)
        data[0x20..0x22].copy_from_slice(&4u16.to_le_bytes());

        let partition = AiePartition::parse(&data).unwrap();

        assert_eq!(partition.column_width(), 4);
        assert_eq!(partition.name(), None);
        assert!(partition.start_columns().is_empty());
    }

    #[test]
    fn test_parse_partition_with_name() {
        let mut data = vec![0u8; 300];

        // mpo_name = 200 (offset to name string)
        data[4..8].copy_from_slice(&200u32.to_le_bytes());

        // Name string at offset 200
        let name = b"test_partition\0";
        data[200..200 + name.len()].copy_from_slice(name);

        let partition = AiePartition::parse(&data).unwrap();

        assert_eq!(partition.name(), Some("test_partition".to_string()));
    }

    #[test]
    fn test_parse_partition_with_start_columns() {
        let mut data = vec![0u8; 300];

        // column_width = 2
        data[0x20..0x22].copy_from_slice(&2u16.to_le_bytes());

        // start_columns array_offset at 0x28: count=2, offset=200
        data[0x28..0x2C].copy_from_slice(&2u32.to_le_bytes());  // count
        data[0x2C..0x30].copy_from_slice(&200u32.to_le_bytes()); // offset

        // Start columns at offset 200: [0, 2]
        data[200..202].copy_from_slice(&0u16.to_le_bytes());
        data[202..204].copy_from_slice(&2u16.to_le_bytes());

        let partition = AiePartition::parse(&data).unwrap();

        assert_eq!(partition.column_width(), 2);
        let cols = partition.start_columns();
        assert_eq!(cols, vec![0, 2]);
    }

    #[test]
    fn test_parse_too_small_data() {
        let data = vec![0u8; 100]; // Less than 184 bytes
        let result = AiePartition::parse(&data);
        assert!(result.is_err());
    }

    // Integration test with real XCLBIN file
    #[test]
    fn test_parse_real_aie_partition() {
        use crate::parser::xclbin::{SectionKind, Xclbin};

        let test_xclbin = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";

        if !std::path::Path::new(test_xclbin).exists() {
            eprintln!("Skipping real AIE partition test: file not found");
            return;
        }

        let xclbin = Xclbin::from_file(test_xclbin).unwrap();
        let section = xclbin.find_section(SectionKind::AiePartition)
            .expect("XCLBIN should have AIE_PARTITION section");

        let partition = AiePartition::parse(section.data()).unwrap();

        // Should have valid column width (typically 1-8 for NPU)
        let width = partition.column_width();
        assert!(width > 0 && width <= 8, "Invalid column width: {}", width);

        // Should have at least one PDI
        let pdis: Vec<_> = partition.pdis().collect();
        assert!(!pdis.is_empty(), "Expected at least one PDI");

        // First PDI should have valid UUID and image data
        let pdi = &pdis[0];
        assert!(!pdi.uuid.is_nil());
        assert!(!pdi.pdi_image.is_empty(), "PDI image should not be empty");
    }

    #[test]
    fn test_parse_pdi_and_find_cdo() {
        use crate::parser::cdo::find_cdo_offset;
        use crate::parser::xclbin::{SectionKind, Xclbin};

        let test_xclbin = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";

        if !std::path::Path::new(test_xclbin).exists() {
            return;
        }

        let xclbin = Xclbin::from_file(test_xclbin).unwrap();
        let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
        let partition = AiePartition::parse(section.data()).unwrap();

        // Get PDI and find CDO within it
        let pdi = partition.primary_pdi().expect("Should have primary PDI");

        // Should be able to find CDO offset within PDI image
        let cdo_offset = find_cdo_offset(pdi.pdi_image);
        assert!(cdo_offset.is_some(), "Should find CDO within PDI image");
    }
}
