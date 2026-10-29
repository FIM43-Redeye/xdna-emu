# AIE ELF Format Reference

> **Source**: llvm-aie `llvm/include/llvm/BinaryFormat/ELF.h`
> **Compiler**: Peano (https://github.com/Xilinx/llvm-aie)

AIE cores use standard ELF32 format with custom machine type and architecture
flags. Each tile runs its own ELF binary loaded via CDO DMA_WRITE commands.

**Implementation**: See `src/parser/elf.rs`

## ELF Header Fields

| Field | Value | Description |
|-------|-------|-------------|
| `e_machine` | `264` (0x108) | `EM_AIE` - AMD/Xilinx AI Engine |
| `e_flags` | See below | Architecture variant |
| Class | ELF32 | 32-bit ELF |
| Endian | Little | Little-endian |

## Architecture Flags (e_flags)

| Value | Name | Description |
|-------|------|-------------|
| `0x01` | `EF_AIE_AIE1` | Original AI Engine (Versal) |
| `0x02` | `EF_AIE_AIE2` | AI Engine ML (Phoenix/HawkPoint NPU) |
| `0x03` | `EF_AIE_AIE2P` | AI Engine ML+ (Strix/Krackan NPU) |

## Memory Layout

AIE cores have separate address spaces for program and data memory:

```
+------------------+  0x00000000
| Program Memory   |  64KB (16K instructions)
| (.text section)  |  Code loaded here
+------------------+  0x0000FFFF

      (gap)

+------------------+  0x00070000
| Data Memory      |  64KB per bank
| Bank 0           |  Stack, buffers
+------------------+  0x00080000
```

### Memory Regions

| Address Range | Region | Description |
|--------------|--------|-------------|
| 0x00000000 - 0x0000FFFF | Program | Instruction memory (.text) |
| 0x00070000 - 0x0007FFFF | Data | Local data memory |
| 0x00080000+ | Unknown | Memory-mapped registers, neighbor banks |

## Common Sections

| Section | Description |
|---------|-------------|
| `.text` | Program code (VLIW instructions) |
| `.comment` | Compiler version string |
| `.symtab` | Symbol table |
| `.strtab` | String table |

## Common Symbols

| Symbol | Address | Description |
|--------|---------|-------------|
| `__start` | 0x00000000 | Entry point (reset vector) |
| `main` | varies | Main kernel function |
| `_main_init` | varies | Initialization code |
| `core_X_Y` | varies | Per-tile kernel (X=col, Y=row) |
| `_sp_start_value_DM_stack` | 0x00070000 | Stack pointer initial value |
| `objFifo_*_buff_*` | 0x0007XXXX | Object FIFO buffer addresses |

## Example Symbol Table

```
Address    Size  Type     Name
0x00000000   32  FUNC     __start
0x00000020  608  FUNC     core_0_2
0x00000280   80  FUNC     _main_init
0x00070000    0  NOTYPE   _sp_start_value_DM_stack
0x00070400    0  NOTYPE   objFifo_out1_buff_0
0x00074000    0  NOTYPE   objFifo_out1_buff_1
0x00078000    0  NOTYPE   objFifo_in1_cons_buff_0
0x0007C000    0  NOTYPE   objFifo_in1_cons_buff_1
```

## Instruction Format

AIE uses a VLIW (Very Long Instruction Word) architecture. Instructions are
variable-width, typically 128 bits (16 bytes) for full bundles or smaller
for compressed encodings.

The ISA is documented in llvm-aie TableGen files:
- `llvm/lib/Target/AIE2/AIE2InstrInfo.td`
- `llvm/lib/Target/AIE2/AIE2InstrFormats.td`

## Loading via CDO

ELF code is loaded to tile program memory via CDO DMA_WRITE commands:

```
DMA_WRITE to tile(0,2) @ 0x00000000
  Length: 720 bytes
  Data: <.text section contents>
```

The CDO commands configure:
1. Program memory (code from .text)
2. DMA descriptors (buffer addresses from symbols)
3. Stream switch routing

## Example: Parsing AIE ELF

```rust
use xdna_emu::parser::AieElf;

let data = std::fs::read("core_0_2.elf")?;
let elf = AieElf::parse(&data)?;

println!("Architecture: {:?}", elf.architecture());
println!("Entry: 0x{:X}", elf.entry_point());

// Get program code
if let Some(text) = elf.text_section() {
    println!("Code size: {} bytes", text.len());
}

// List functions
for func in elf.functions() {
    println!("0x{:08X} {} ({} bytes)",
        func.address, func.name, func.size);
}

// Find buffer addresses
for sym in elf.data_symbols() {
    println!("Buffer: {} @ 0x{:08X}", sym.name, sym.address);
}
```

## Compiler Information

The `.comment` section contains compiler version information:

```
clang version 20.0.0 (https://github.com/Xilinx/llvm-aie ...)
Linker: LLD 20.0.0 (https://github.com/Xilinx/llvm-aie ...)
```

## References

- llvm-aie: https://github.com/Xilinx/llvm-aie
- ELF specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
- AMD AM020: AIE-ML Architecture Manual
