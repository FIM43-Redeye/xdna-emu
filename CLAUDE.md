# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Setup

**Start Claude from the parent `npu-work` directory**, not from xdna-emu directly:

```bash
cd /home/triple/npu-work
claude
```

This ensures:
- NPU development environment auto-activates (Python 3.13, mlir-aie, Peano compiler)
- XRT and hardware access are configured
- Helper functions like `npu-compile` and `npu-run-quiet` are available

The workspace contains:
- `xdna-emu/` - This emulator project (main focus)
- `mlir-aie/` - MLIR-based AIE compiler and test suite
- `llvm-aie/` - Peano compiler (LLVM with AIE backend)

See `/home/triple/npu-work/CLAUDE.md` for environment details.

---

There is one absolutely critical rule to always keep to: **REFER TO THE DOCUMENTATION.** The way to do so is described below, using Explore agents, since the body of documentation is very large.

## Project Vision

**xdna-emu** is an open-source, cycle-accurate emulator and visual debugger for AMD XDNA NPUs (Ryzen AI). It fills the gap left by AMD's proprietary aiesimulator (which is license-gated, CLI-only, and not cycle-accurate).

### Goals
1. **Binary-Compatible Emulation**: Load real `.xclbin` binaries, execute them faithfully
2. **Visual Debugging**: See tiles, data flow, DMA transfers, locks, routing in real-time
3. **Hardware Validation**: Run in emulator -> run on real NPU -> compare results
4. **Optional Compilation**: Invoke Peano compiler to build kernels, run immediately

### The Dream Workflow
```
Write kernel.cc
    |
xdna-emu compiles via Peano -> kernel.xclbin
    |
xdna-emu emulates (visual, step-through, breakpoints)
    |
"Run on Hardware" button
    |
Same binary executes on real NPU via XRT
    |
Compare emulated vs actual results
```

## Current Status

See [ROADMAP.md](ROADMAP.md) for detailed status with confidence markers
(VERIFIED / OBSERVED / CLAIMED).

| Phase | Status | Summary |
|-------|--------|---------|
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | Functional | Unit tests pass; real-binary coverage is thin |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | Not started | Peano-first; Vitis deferred to post-1.0 |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | GUI exists | GUI renders; debugging features not built |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | In progress | Test harness exists; coverage has major gaps |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | Not started | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | Not started | |

Run `cargo test --lib` to see the current test count. Do not rely on
numbers written in documentation -- they go stale within a session.

## Target Devices

These are the official device names from the xdna-driver source:

| Driver ID | Product Name | Codename | Architecture | Array Size | Status |
|-----------|--------------|----------|--------------|------------|--------|
| NPU1 | Ryzen AI | Phoenix/Hawk Point | AIE2 (XDNA) | 5 cols x 6 rows | **Primary target** |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P (XDNA2) | 8+ cols x 6 rows | Planned |
| NPU6 | (TBD) | Krackan | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |

Array sizes include the shim tile row (row 0). Driver IDs NPU2/NPU3 are
prototypes marked for deprecation -- not consumer devices.

We are starting with **Phoenix (NPU1/AIE2)** because it is the hardware we have.
AIE2P support will be incremental once AIE2 is solid.

### mlir-aie Device Target Naming

mlir-aie uses different device names that can be confusing:

| mlir-aie target | Maps to Driver | Architecture |
|-----------------|----------------|--------------|
| `npu1`, `npu1_Xcol` | NPU1 (Phoenix) | AIE2 |
| `npu2`, `npu2_Xcol` | NPU4 (Strix) | AIE2P |
| `xcvc1902` | N/A (Versal) | AIE1 |
| `xcve2802` | N/A (Versal) | AIE2 |

**Not in scope**: Versal FPGAs (AIE1 + PL fabric) - different use case, no local hardware. However, AIE1 support may be added later since the TableGen parser handles multiple architectures.

## Architecture Knowledge Sources

**CRITICAL: Consult the architecture documentation when implementing any hardware feature.**

Many debugging hours have been lost to assumptions about register layouts, address mappings, and data formats. The AIE-ML architecture has specific conventions that differ from intuition:
- Lock registers are 16 bytes apart, not 4
- Core lock IDs 48-63 map to memory module locks 0-15
- BD fields span multiple words with specific bit layouts
- DMA addressing uses word units, not bytes

When implementing anything related to registers, DMA, locks, or memory:
1. First read the relevant AM020/AM025 section
2. Check the extracted docs in `docs/xdna/` (text files, not PDFs)
3. Verify bit layouts and address calculations against the reference
4. Add comments citing the specific register/section

**Use Explore agents for documentation research.** The architecture docs are extensive (AM020 alone spans 100+ pages). Rather than reading files directly and wasting context, spawn an Explore agent with subagent_type='Explore' to navigate the docs and extract the specific details needed. This is ideal for questions like "how does AIE-ML memory addressing work?" or "what is the BD lock field format?"

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

## Component Documentation

Detailed documentation for each module lives in `.claude/components/`. Read
the relevant file when working on that area of the codebase.

| Component | File | When to Read |
|-----------|------|-------------|
| Device model | [`.claude/components/device.md`](.claude/components/device.md) | Working on tiles, array, DMA, streams, locks, host memory (`src/device/`) |
| Interpreter | [`.claude/components/interpreter.md`](.claude/components/interpreter.md) | Working on instruction decode, execution, timing, multi-core (`src/interpreter/`) |
| Parser | [`.claude/components/parser.md`](.claude/components/parser.md) | Working on XCLBIN, ELF, or CDO parsing (`src/parser/`) |
| TableGen | [`.claude/components/tablegen.md`](.claude/components/tablegen.md) | Working on ISA definitions, decoder tables, llvm-aie integration (`src/tablegen/`) |
| Testing | [`.claude/components/testing.md`](.claude/components/testing.md) | Working on tests, test runner, FFI, NPU instructions, config (`src/testing/`, `src/npu/`, `src/ffi/`, `tests/`) |
| Visual | [`.claude/components/visual.md`](.claude/components/visual.md) | Working on the GUI debugger (`src/visual/`) |

Top-level source files not covered by component docs:
- `src/main.rs` -- CLI and GUI entry point
- `src/lib.rs` -- crate root, module declarations
- `src/integration/mod.rs` -- external tool integration (placeholder)

## Related Resources

- **xdna-driver**: `/home/triple/npu-work/xdna-driver` - Linux kernel driver, authoritative device definitions
- **mlir-aie**: `/home/triple/npu-work/mlir-aie` - test binaries, device models
- **llvm-aie**: `../llvm-aie` (local clone) - ISA definitions, TableGen files
- **XRT**: https://github.com/Xilinx/XRT - runtime (installed at /opt/xilinx/xrt)
- **aie-rt**: https://github.com/Xilinx/aie-rt - low-level register definitions

## Available Development Tools

### XRT Tools (in /opt/xilinx/xrt/bin/)
- **xclbinutil**: Essential tool for XCLBIN inspection, extraction, and manipulation
  - `xclbinutil --info --input file.xclbin` - dump sections and metadata
  - `xclbinutil --dump-section` - extract specific sections
- **xrt-smi**: Device management and diagnostics

### MLIR-AIE Tools (in mlir-aie/my_install/ or ironenv/)
- **llvm-objdump**: Disassemble AIE ELF binaries
  - `llvm-objdump -d file.elf` - disassemble to see instruction mnemonics
- **aie-translate**: MLIR to various formats
- **aie-opt**: MLIR optimization passes

### RyzenAI-SW Tools
- Located in `/home/triple/npu-work/RyzenAI-SW/`
- Contains NPU driver source and examples

## Build Commands

```bash
# Build
cargo build

# Build optimized
cargo build --release

# Run
cargo run -- path/to/binary.xclbin

# Test
./scripts/run-tests.sh          # All tests (doc tests run with nice 19)
./scripts/run-tests.sh --lib    # Fast: library tests only
./scripts/run-tests.sh --doc    # Doc tests only (nice'd, limited parallelism)
cargo test --lib                # Direct: library tests without script

# Benchmark
cargo bench

# Profile (generates flamegraph.svg)
cargo install flamegraph
cargo flamegraph --release -- path/to/binary.xclbin
```

**Note on doc tests**: Doc tests spawn separate processes that each load TableGen
files from llvm-aie. The test script runs them with `nice -n 19` and limited
parallelism to avoid overwhelming the system during other work.

## How To Begin

1. Read [ROADMAP.md](ROADMAP.md) for the development plan and confidence markers
2. Check the relevant [phase documentation](docs/roadmap/) for current details
3. Run `cargo test --lib` to verify everything works
4. Read the component doc (`.claude/components/`) for the module you are working on
