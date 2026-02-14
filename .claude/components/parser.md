# Parser

Binary format parsers for AMD XDNA NPU binaries: XCLBIN containers, ELF core executables, and CDO configuration.

Read this file when working on anything in `src/parser/`.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports `Xclbin`, `AiePartition`, `Cdo`, `AieElf` |
| `xclbin.rs` | `Xclbin` -- XCLBIN container parser (sections, metadata, AIE partition extraction) |
| `aie_partition.rs` | `AiePartition` -- extracts PDI/CDO data and per-core ELF binaries from the AIE partition section |
| `elf.rs` | `AieElf` -- AIE core ELF parser (text/data segments, entry point, `MemoryRegion` mapping) |
| `cdo.rs` | `Cdo` -- Configuration Data Object parser (tile setup commands: register writes, DMA config, routing) |

## Binary Format Chain

Real NPU binaries follow this nesting structure:

```
XCLBIN container
  +-- AIE Partition section (PDI format)
  |     +-- CDO commands (register writes, DMA setup, stream routing)
  |     +-- Core ELF binaries (one per active compute tile)
  +-- Other sections (metadata, debug info, etc.)
```

The typical load sequence:
1. `Xclbin::parse(data)` -- parse the container, identify sections
2. `AiePartition::extract(&xclbin)` -- pull out the AIE partition
3. `Cdo::parse(pdi_data)` -- decode configuration commands
4. `AieElf::parse(elf_data)` -- load per-core executables

## Key Types

- `Xclbin` -- parsed container with section iterator
- `SectionKind` -- identifies section types (AIE_PARTITION, MEM_TOPOLOGY, etc.)
- `AiePartition` -- CDO + ELF data extracted from the partition
- `Cdo` -- list of CDO commands (register writes, block writes, mask writes)
- `AieElf` -- parsed ELF with segments mapped to tile memory layout
- `MemoryRegion` -- ELF segment mapped to a tile memory address range

## Conventions

- All parsers are zero-copy where possible, operating on byte slices
- CDO commands are the primary mechanism for configuring the NPU array (DMA descriptors, stream routing, lock initialization)
- ELF files target individual compute tiles; the CDO specifies which tile each ELF belongs to
- The CDO `find_cdo_offset()` helper locates the CDO payload within PDI data
