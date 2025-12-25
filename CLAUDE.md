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

## Current Status

See [ROADMAP.md](ROADMAP.md) for high-level progress, or dive into:
- [Phase 1: Core Accuracy](docs/roadmap/phase1-core-accuracy.md) - 🟢 Mostly Complete (55% binary recognition)
- [Phase 2: Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) - 🔴 Not Started
- [Phase 3: Developer Experience](docs/roadmap/phase3-developer-experience.md) - 🟡 GUI Exists
- [Phase 4: Validation & Testing](docs/roadmap/phase4-validation-testing.md) - 🟡 265 Tests
- [Phase 5: Production Readiness](docs/roadmap/phase5-production-readiness.md) - 🔴 Not Started
- [Phase 6: Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) - 🔴 Not Started

**Current focus**: Phase 1 - extracting slots from multi-slot bundles (currently 55%)

## Target Devices

| Device | Codename | Architecture | Array Size | Status |
|--------|----------|--------------|------------|--------|
| NPU1 | Phoenix/HawkPoint | AIE2 (XDNA) | 4 cols × 6 rows | **Primary target** |
| NPU2 | Strix | AIE2P (XDNA2) | 4 cols × 6 rows | Planned |
| NPU3 | Strix Halo | AIE2P (XDNA2) | 8 cols × 6 rows | Planned |
| NPU4 | Krackan | AIE2P (XDNA2) | 4 cols × 6 rows | Planned |

We're starting with **Phoenix (NPU1/AIE2)** because it's the hardware we have. The architecture is generic - AIE2P support will be incremental once AIE2 is solid.

**Not in scope**: Versal FPGAs (AIE + PL fabric) - different use case, no local hardware.

## Architecture Knowledge Sources

### AMD Documentation
- **AM020**: AIE-ML (AIE2) Architecture Manual
- **AM025**: AIE-ML Register Reference
- Links: https://docs.amd.com/
- Path: xdna-emu/docs/xdna/ (use the files in the folders, not the PDFs)

### Binary Formats
- **XCLBIN**: Container (ELF cores + PDI/CDO configuration)
- **ELF**: Per-core executables
- **CDO**: Configuration Data Objects (DMA descriptors, routing)

### ISA Reference
- **llvm-aie (Peano)**: `llvm/lib/Target/AIE/` has TableGen instruction definitions
- Repository: https://github.com/Xilinx/llvm-aie
- Local clone: `../llvm-aie`

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
├── ROADMAP.md            # High-level development plan
├── CLAUDE.md             # AI assistant context (this file)
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
│   ├── interpreter/      # New modular interpreter
│   │   ├── bundle/       # VLIW bundle handling
│   │   ├── decode/       # Instruction decoding (pattern + TableGen)
│   │   ├── state/        # Register files, execution context
│   │   ├── execute/      # Execution units (scalar, vector, memory, control)
│   │   ├── core/         # Per-core interpreter
│   │   └── engine/       # Multi-core coordinator
│   ├── tablegen/         # TableGen parser for llvm-aie
│   │   ├── parser.rs     # Regex-based .td file parsing
│   │   ├── types.rs      # SlotDef, FormatClass, InstrDef, SemanticOp
│   │   └── resolver.rs   # Compute encodings from format classes
│   ├── emu_stub/         # Legacy emulation (being replaced)
│   ├── visual/           # GUI (egui)
│   │   ├── app.rs        # Main application
│   │   ├── tile_grid.rs  # Tile array view
│   │   ├── tile_detail.rs # Detail panels
│   │   ├── controls.rs   # Run/Step/Reset
│   │   └── memory_view.rs # Hex viewer
│   └── integration/      # External tools (planned)
│       └── mod.rs
└── docs/
    └── roadmap/          # Detailed phase documentation
        ├── phase1-core-accuracy.md
        ├── phase2-toolchain-integration.md
        ├── phase3-developer-experience.md
        ├── phase4-validation-testing.md
        ├── phase5-production-readiness.md
        ├── phase6-community-ecosystem.md
        └── tablegen-assessment.md
```

## Related Resources

- **mlir-aie**: `/home/triple/npu-work/mlir-aie` - test binaries, device models
- **llvm-aie**: `../llvm-aie` (local clone) - ISA definitions, TableGen files
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

# Test (267 tests)
cargo test

# Benchmark
cargo bench

# Profile (generates flamegraph.svg)
cargo install flamegraph
cargo flamegraph --release -- path/to/binary.xclbin
```

## How To Begin

1. Read [ROADMAP.md](ROADMAP.md) for the high-level development plan
2. Check [Phase 1](docs/roadmap/phase1-core-accuracy.md) for current implementation details
3. Run `cargo test` to verify everything works
4. See the "Next Steps" section in Phase 1 for current work items
