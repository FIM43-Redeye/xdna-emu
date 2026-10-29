# XCLBIN Format Reference

> **Source**: XRT `/opt/xilinx/xrt/include/xrt/detail/xclbin.h`
> **License**: Apache 2.0 / GPL v2 (dual licensed)

XCLBIN is AMD/Xilinx's container format for FPGA and NPU binaries. It packages
bitstreams, ELF executables, metadata, and configuration data into a single file.

## File Layout

```
+------------------------------------------+
| Magic: "xclbin2\0" (8 bytes)             |
+------------------------------------------+
| Signature length (4 bytes, -1 = none)    |
+------------------------------------------+
| Reserved (28 bytes, filled with 0xFF)    |
+------------------------------------------+
| Key block (256 bytes, signature data)    |
+------------------------------------------+
| Unique ID (8 bytes)                      |
+------------------------------------------+
| Header (152 bytes)                       |
+------------------------------------------+
| Section headers (40 bytes each)          |
|   ... m_numSections entries ...          |
+------------------------------------------+
| Section data (variable, 8-byte aligned)  |
|   ... payloads for each section ...      |
+------------------------------------------+
```

## Main Structure: `axlf`

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x000 | 8 | `m_magic` | `"xclbin2\0"` |
| 0x008 | 4 | `m_signature_length` | -1 if unsigned |
| 0x00C | 28 | `reserved` | Filled with 0xFF |
| 0x028 | 256 | `m_keyBlock` | Signature for validation |
| 0x128 | 8 | `m_uniqueId` | Unique identifier |
| 0x130 | 152 | `m_header` | See `axlf_header` below |
| 0x1C8 | 40×N | `m_sections` | Section header array |

**Total fixed header size**: 0x1C8 (456 bytes) before sections

## Header: `axlf_header` (152 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 8 | `m_length` | Total file size in bytes |
| 0x08 | 8 | `m_timeStamp` | Unix timestamp of creation |
| 0x10 | 8 | `m_featureRomTimeStamp` | Feature ROM timestamp |
| 0x18 | 2 | `m_versionPatch` | Patch version |
| 0x1A | 1 | `m_versionMajor` | Major version |
| 0x1B | 1 | `m_versionMinor` | Minor version |
| 0x1C | 2 | `m_mode` | XCLBIN_MODE enum |
| 0x1E | 2 | `m_actionMask` | Action flags (AM_LOAD_AIE, etc.) |
| 0x20 | 16 | `m_interface_uuid` | Interface UUID |
| 0x30 | 64 | `m_platformVBNV` | Platform string (null-terminated) |
| 0x70 | 16 | `uuid` | UUID of this xclbin |
| 0x80 | 16 | `m_debug_bin` | Debug binary name |
| 0x90 | 4 | `m_numSections` | Number of section headers |

## Section Header: `axlf_section_header` (40 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | `m_sectionKind` | Section type enum |
| 0x04 | 16 | `m_sectionName` | Human-readable name |
| 0x14 | 8 | `m_sectionOffset` | File offset to data |
| 0x1C | 8 | `m_sectionSize` | Size of section data |

## Section Types (NPU-relevant)

| Value | Name | Description |
|-------|------|-------------|
| 2 | `EMBEDDED_METADATA` | XML metadata |
| 6 | `MEM_TOPOLOGY` | Memory bank definitions |
| 7 | `CONNECTIVITY` | Kernel-to-memory connections |
| 8 | `IP_LAYOUT` | IP core addresses |
| 18 | `PDI` | Programmable Device Image (contains CDO) |
| 25 | `AIE_METADATA` | AIE-specific JSON metadata |
| 32 | `AIE_PARTITION` | AIE partition info with PDI images |

## AIE Partition Structure

The `AIE_PARTITION` section (kind=32) contains:

```c
struct aie_partition {
    uint8_t  schema_version;      // Usually 0
    uint8_t  padding0[3];
    uint32_t mpo_name;            // Offset to name string
    uint32_t operations_per_cycle;
    uint8_t  padding[4];
    uint64_t inference_fingerprint;
    uint64_t pre_post_fingerprint;
    struct aie_partition_info info;  // 88 bytes
    struct array_offset aie_pdi;     // PDI array
    // ... more fields
};
```

The `aie_pdi` array contains the actual PDI images with CDO data.

## PDI Structure

```c
struct aie_pdi {
    uuid_t uuid;                    // 16 bytes
    struct array_offset pdi_image;  // Offset/size to raw PDI
    struct array_offset cdo_groups; // CDO group definitions
    uint8_t reserved[64];
};
```

## Parsing Strategy

1. **Validate magic** - Must be `"xclbin2\0"`
2. **Read header** - Get section count from `m_numSections`
3. **Parse section headers** - Build section table
4. **Find AIE_PARTITION** - Contains the NPU configuration
5. **Extract PDI/CDO** - The actual tile configuration data

## Example Hex (from real xclbin)

```
00000000: 7863 6c62 696e 3200  "xclbin2\0"
00000008: ffff ffff           signature_length = -1 (unsigned)
0000000c: ffff ffff ...       reserved (28 bytes of 0xFF)
...
00000128: dda2 4969 0000 0000  unique_id
00000130: 9723 0000 0000 0000  m_length (file size)
...
000001c8: 0600 0000           first section kind (MEM_TOPOLOGY=6)
000001cc: 6d61 696e 5f6d ...  "main_mem_topolo\0"
```

## References

- XRT source: https://github.com/Xilinx/XRT
- Header: `/opt/xilinx/xrt/include/xrt/detail/xclbin.h`
- mlir-aie: https://github.com/Xilinx/mlir-aie
