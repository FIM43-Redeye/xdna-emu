# AIE Partition Format Reference

> **Source**: XRT `include/xrt/detail/xclbin.h`
> **License**: Apache 2.0

The AIE Partition section in an XCLBIN contains the configuration data needed
to program the AIE array. It includes partition metadata, PDI images (which
contain CDO commands), and CDO group definitions.

## Structure Hierarchy

```
AIE Partition Section
├── aie_partition (184 bytes) - Main partition header
│   ├── aie_partition_info (88 bytes) - Column/row configuration
│   │   └── start_columns[] - Array of starting column indices
│   └── aie_pdi[] - Array of PDI containers
│       ├── pdi_image - Raw PDI/CDO binary data
│       └── cdo_groups[] - CDO group metadata
│           └── Kernel IDs, pre-CDO references
└── String table (name, commit ID, etc.)
```

## aie_partition (184 bytes)

The main partition header.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 1 | `schema_version` | Schema version (default 0) |
| 0x01 | 3 | `padding0` | Alignment padding |
| 0x04 | 4 | `mpo_name` | Offset to partition name string |
| 0x08 | 4 | `operations_per_cycle` | For TOPS calculation |
| 0x0C | 4 | `padding` | Alignment padding |
| 0x10 | 8 | `inference_fingerprint` | Hash of inference function |
| 0x18 | 8 | `pre_post_fingerprint` | Hash of pre/post processing |
| 0x20 | 88 | `info` | aie_partition_info structure |
| 0x78 | 8 | `aie_pdi` | array_offset to PDI array |
| 0x80 | 4 | `kernel_commit_id` | Offset to git commit string |
| 0x84 | 52 | `reserved` | Reserved for future use |

## aie_partition_info (88 bytes)

Partition layout configuration.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 2 | `column_width` | Width of partition in columns |
| 0x02 | 6 | `padding` | Alignment padding |
| 0x08 | 8 | `start_columns` | array_offset to u16[] of start column IDs |
| 0x10 | 72 | `reserved` | Reserved |

## aie_pdi (96 bytes)

PDI container with CDO image data.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 16 | `uuid` | PDI container UUID |
| 0x10 | 8 | `pdi_image` | array_offset to raw PDI bytes |
| 0x18 | 8 | `cdo_groups` | array_offset to cdo_group[] |
| 0x20 | 64 | `reserved` | Reserved |

## cdo_group (96 bytes)

CDO group metadata.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | `mpo_name` | Offset to group name string |
| 0x04 | 1 | `cdo_type` | CDO type (see below) |
| 0x05 | 3 | `padding` | Alignment |
| 0x08 | 8 | `pdi_id` | PDI ID |
| 0x10 | 8 | `dpu_kernel_ids` | array_offset to u64[] kernel IDs |
| 0x18 | 8 | `pre_cdo_groups` | array_offset to u32[] pre-CDO group IDs |
| 0x20 | 64 | `reserved` | Reserved |

## CDO Type Values

| Value | Name | Description |
|-------|------|-------------|
| 0 | Unknown | Unknown CDO type |
| 1 | Primary | Primary CDO (main configuration) |
| 2 | Lite | Lite CDO (minimal reconfiguration) |
| 3 | PrePost | Pre/Post processing CDO |

## array_offset (8 bytes)

Generic pointer to variable-length array.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | `count` | Number of elements |
| 0x04 | 4 | `offset` | Byte offset from section start |

## PDI Image Format

The `pdi_image` data pointed to by aie_pdi contains:

1. **PDI Header** - Xilinx/AMD proprietary header (sync words, metadata)
2. **CDO Section** - Embedded CDO commands

To find the CDO within the PDI:
- Scan for CDO magic: `"CDO\0"` (0x004F4443) or `"XLNX"` (0x584C4E58)
- Magic appears at offset +4 in CDO header (after `num_words` field)
- Verify `num_words` field equals 4

## Example: Parsing AIE Partition

```rust
use xdna_emu::parser::{Xclbin, AiePartition, Cdo};
use xdna_emu::parser::xclbin::SectionKind;
use xdna_emu::parser::cdo::find_cdo_offset;

let xclbin = Xclbin::from_file("design.xclbin")?;

if let Some(section) = xclbin.find_section(SectionKind::AiePartition) {
    let partition = AiePartition::parse(section.data())?;

    println!("Column width: {}", partition.column_width());
    println!("Start columns: {:?}", partition.start_columns());

    for pdi in partition.pdis() {
        println!("PDI UUID: {}", pdi.uuid);

        if let Some(cdo_offset) = find_cdo_offset(pdi.pdi_image) {
            let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;
            println!("CDO commands: {}", cdo.commands().count());
        }
    }
}
```

## References

- XRT xclbin.h: https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/xrt/detail/xclbin.h
- AMD Versal ACAP documentation
