# CDO (Configuration Data Object) Format Reference

> **Source**: mlir-aie `third_party/bootgen/cdo-*.{h,c}`
> **License**: Apache 2.0
> **AMD Docs**: [UG1304 - Versal ACAP System Software Developers Guide](https://docs.amd.com/r/en-US/ug1304-versal-acap-ssdg/Configuration-Data-Object)

CDO is AMD's format for device configuration commands. It's a sequence of
register writes, DMA transfers, and control operations that configure the
AIE array - routing, DMA descriptors, memory initialization, and core startup.

**Implementation**: See `src/parser/cdo.rs`

## File Layout

```
+------------------------------------------+
| Header (20 bytes)                        |
+------------------------------------------+
| Command stream (variable)                |
|   Command 1                              |
|   Command 2                              |
|   ...                                    |
+------------------------------------------+
```

## CDO Header (20 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | `NumWords` | Remaining words in header (usually 4) |
| 0x04 | 4 | `IdentWord` | Magic (see below) |
| 0x08 | 4 | `Version` | Format version |
| 0x0C | 4 | `CDOLength` | Length in words (excluding header) |
| 0x10 | 4 | `CheckSum` | One's complement of header sum |

### Magic Values

| Value | Bytes | Description |
|-------|-------|-------------|
| `0x584C4E58` | "XNLX" | Bootgen/XLNX format |
| `0x004F4443` | "CDO\0" | AIE CDO driver format |

Both magic values are valid and indicate CDO format.

### Version Values

| Value | Meaning |
|-------|---------|
| `0x0132` | Version 1.50 (CDOv2 format, allows CDOv1 NPI/CFU commands) |
| `0x0200` | Version 2.00 (CDOv2 format, SEM commands replace NPI/CFU) |

## Command Format

Each command is one or more 32-bit words:

```
Word 0: [31:16] = payload_length, [15:0] = opcode
Word 1..N: command-specific payload
```

For commands with payload_length=0, only the opcode word is present.

## Command Opcodes (CDOv2)

### General Commands (0x1xx)

| Opcode | Name | Payload | Description |
|--------|------|---------|-------------|
| 0x100 | END_MARK | 0 | End marker |
| 0x101 | MASK_POLL | 3 | Poll register until (val & mask) == expected |
| 0x102 | MASK_WRITE | 3 | Write (reg & ~mask) | (val & mask) |
| 0x103 | WRITE | 2 | Write value to 32-bit address |
| 0x104 | DELAY | 1 | Delay N cycles |
| 0x105 | DMA_WRITE | 2+N | Write N words to address |
| 0x106 | MASK_POLL64 | 4 | 64-bit address mask poll |
| 0x107 | MASK_WRITE64 | 4 | 64-bit address mask write |
| 0x108 | WRITE64 | 3 | Write to 64-bit address |
| 0x109 | DMA_XFER | varies | DMA transfer |
| 0x10D | DMA_WRITE_KEYHOLE | varies | Keyhole DMA write |
| 0x111 | NOP | 0-N | No operation (padding) |
| 0x119 | MARKER | varies | Debug marker |

### Power Management Commands (0x2xx)

These control clocks, resets, and power domains. Less relevant for emulation
but may appear in CDO streams:

| Opcode | Name |
|--------|------|
| 0x201 | PM_GET_API_VERSION |
| 0x20D | PM_REQUEST_DEVICE |
| 0x20E | PM_RELEASE_DEVICE |
| 0x211 | PM_RESET_ASSERT |
| 0x224 | PM_CLOCK_ENABLE |

## Key Commands for AIE Emulation

### WRITE (0x103)

```
Word 0: [31:16] = 2, [15:0] = 0x103
Word 1: address (32-bit)
Word 2: value
```

Used for: Lock configuration, single register writes

### DMA_WRITE (0x105)

```
Word 0: [31:16] = 2+N, [15:0] = 0x105
Word 1: address (32-bit)
Word 2: length in bytes
Word 3..N+2: data words
```

Used for: ELF loading, memory initialization, bulk configuration

### MASK_WRITE (0x102)

```
Word 0: [31:16] = 3, [15:0] = 0x102
Word 1: address
Word 2: mask
Word 3: value
```

Result: `*addr = (*addr & ~mask) | (value & mask)`

Used for: Partial register updates, bitfield modifications

### WRITE64 (0x108)

```
Word 0: [31:16] = 3, [15:0] = 0x108
Word 1: address_hi
Word 2: address_lo
Word 3: value
```

Used for: Shim DMA configuration (64-bit address space)

## Address Space

CDO addresses map to the AIE array's memory-mapped register space:

```
Address = (col << COL_SHIFT) | (row << ROW_SHIFT) | register_offset

For AIE2 (NPU):
  COL_SHIFT = 25
  ROW_SHIFT = 20
```

### Example Address Decoding

```
Address: 0x020A0000
  col = (0x020A0000 >> 25) & 0x1F = 1
  row = (0x020A0000 >> 20) & 0x1F = 10  (but only 6 rows, so row=2 in tile coords)
  offset = 0x020A0000 & 0xFFFFF = 0xA0000
```

## Register Regions (per tile)

| Offset Range | Description |
|--------------|-------------|
| 0x00000-0x1FFFF | Tile memory |
| 0x20000-0x2FFFF | Memory module registers |
| 0x30000-0x3FFFF | Tile control |
| 0x40000-0x4FFFF | Stream switch |
| 0x60000-0x6FFFF | DMA |

## Parsing Strategy for Emulation

1. **Validate header** - Check magic `0x584C4E58`
2. **Parse commands sequentially** - Each command is self-describing
3. **Execute WRITE/DMA_WRITE** - These configure the tile state
4. **Track MASK_WRITE** - Partial register updates
5. **Ignore PM commands** - Power management not needed for emulation
6. **Record memory writes** - For ELF and data initialization

## What CDO Configures

For NPU emulation, CDO commands set up:

1. **Stream switch routing** - Which ports connect to which
2. **DMA buffer descriptors** - Address, length, stride patterns
3. **Lock initial values** - Synchronization state
4. **Core memory** - ELF code loaded via DMA_WRITE
5. **Tile memory initialization** - Constants, lookup tables

## Example CDO Fragment

```
# Write to stream switch (col=0, row=2)
0x00020103  # WRITE, payload=2
0x000680A0  # address: tile(0,2) stream switch config
0x80000005  # value: route config

# DMA write to tile memory
0x00040105  # DMA_WRITE, payload=4
0x00060000  # address: tile(0,2) memory base
0x00000008  # length: 8 bytes
0xDEADBEEF  # data word 0
0xCAFEBABE  # data word 1
```

## References

- Bootgen source: https://github.com/Xilinx/bootgen
- mlir-aie CDO: `third_party/bootgen/cdo-binary.c`
- AMD UG1304: Versal ACAP System Software Developers Guide
