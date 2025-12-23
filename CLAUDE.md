# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

**xdna-emu** is an open-source, cycle-accurate emulator and visual debugger for AMD XDNA NPUs (Ryzen AI). It fills the gap left by AMD's proprietary aiesimulator (which is license-gated, CLI-only, and not cycle-accurate).

### Goals
1. **Binary-Compatible Emulation**: Load real `.xclbin` binaries, execute them faithfully
2. **Visual Debugging**: See tiles, data flow, DMA transfers, locks, routing in real-time
3. **Hardware Validation**: Run in emulator → run on real NPU → compare results
4. **Optional Compilation**: Invoke Peano compiler to build kernels, run immediately

### The Dream Workflow
```
Write kernel.cc
    ↓
xdna-emu compiles via Peano → kernel.xclbin
    ↓
xdna-emu emulates (visual, step-through, breakpoints)
    ↓
"Run on Hardware" button
    ↓
Same binary executes on real NPU via XRT
    ↓
Compare emulated vs actual results
```

## Target Devices

| Device | Codename | Architecture | Array Size |
|--------|----------|--------------|------------|
| NPU1 | Phoenix/HawkPoint | AIE2 (XDNA) | 4 cols × 6 rows |
| NPU2 | Strix | AIE2P (XDNA2) | 4 cols × 6 rows |
| NPU3 | Strix Halo | AIE2P (XDNA2) | 8 cols × 6 rows |
| NPU4 | Krackan | AIE2P (XDNA2) | 4 cols × 6 rows |

**Not in scope**: Versal FPGAs (AIE + PL fabric) - different use case, no local hardware.

## Architecture Knowledge Sources

### AMD Documentation
- **AM020**: AIE-ML (AIE2) Architecture Manual
- **AM025**: AIE-ML Register Reference
- Links: https://docs.amd.com/

### Binary Formats
- **XCLBIN**: Container (ELF cores + PDI/CDO configuration)
- **ELF**: Per-core executables
- **CDO**: Configuration Data Objects (DMA descriptors, routing)

### ISA Reference
- **llvm-aie (Peano)**: `llvm/lib/Target/AIE2/` has TableGen instruction definitions
- Repository: https://github.com/Xilinx/llvm-aie

## What We're Emulating

### Per Tile
- **Compute Core**: VLIW processor, 256-bit vector unit
- **Local Memory**: 64KB (compute tile), 512KB (mem tile)
- **DMA Engine**: 2 channels, n-dimensional addressing
- **Locks**: 64 synchronization primitives
- **Stream Switch**: Circuit/packet routing

### Array Level
- **Shim Tiles**: DDR interface via NoC
- **Mem Tiles**: Shared memory between columns
- **Routing**: Configured stream switches

### Execution Model
- Cores run independently (no global clock)
- DMAs operate concurrently with cores
- Locks synchronize between tiles
- Stream switches route based on CDO configuration

## Technical Decisions

### Language: Rust

Chosen for:
- Memory safety (emulator = complex state machine)
- Excellent binary parsing (`goblin`, `nom`)
- Built-in profiling (`cargo flamegraph`, `cargo bench`)
- Modern tooling (`cargo` just works)

### GUI Framework: egui

Using `eframe` (egui framework) for the visual debugger:
- Pure Rust, no external dependencies
- Immediate mode - simple state management
- Cross-platform (Linux, Windows, macOS, web)

## Repository Structure

```
xdna-emu/
├── README.md             # User documentation
├── ROADMAP.md            # Development plan
├── CLAUDE.md             # AI assistant context
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI + GUI entry point
│   ├── lib.rs
│   ├── parser/           # Binary format parsers
│   │   ├── xclbin.rs     # XCLBIN container
│   │   ├── aie_partition.rs
│   │   ├── cdo.rs        # CDO commands
│   │   └── elf.rs        # AIE ELF files
│   ├── device/           # Device state model
│   │   ├── registers.rs  # Register definitions
│   │   ├── tile.rs       # Single tile state
│   │   ├── array.rs      # Tile array
│   │   └── state.rs      # CDO application
│   ├── emu/              # Emulation engine
│   │   ├── instruction.rs # Instruction decoder
│   │   ├── core.rs       # Core executor
│   │   └── engine.rs     # Multi-core coordinator
│   ├── visual/           # GUI (egui)
│   │   ├── app.rs        # Main application
│   │   ├── tile_grid.rs  # Tile array view
│   │   ├── tile_detail.rs # Detail panels
│   │   ├── controls.rs   # Run/Step/Reset
│   │   └── memory_view.rs # Hex viewer
│   └── integration/      # External tools (planned)
│       └── mod.rs
└── docs/
    └── formats/          # Binary format docs
```

## Related Resources

- **mlir-aie**: `/home/triple/npu-work/mlir-aie` - test binaries, device models
- **llvm-aie**: https://github.com/Xilinx/llvm-aie - ISA definitions
- **XRT**: https://github.com/Xilinx/XRT - runtime (installed at /opt/xilinx/xrt)
- **aie-rt**: https://github.com/Xilinx/aie-rt - low-level register definitions

## Build Commands

```bash
# Build
cargo build

# Build optimized
cargo build --release

# Run
cargo run -- path/to/binary.xclbin

# Test
cargo test

# Benchmark
cargo bench

# Profile (generates flamegraph.svg)
cargo install flamegraph
cargo flamegraph --release -- path/to/binary.xclbin
```

## How To Begin

Read ROADMAP.md for the development plan and current status.
