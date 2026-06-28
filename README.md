# xdna-emu

Open-source emulator and visual debugger for AMD XDNA NPUs (Ryzen AI).
Loads real `.xclbin` binaries, executes them faithfully, and provides a
drop-in emulation backend for the AMD NPU development ecosystem.

## Overview

xdna-emu is a from-scratch emulator for the AIE2 tile processor array found
in AMD Ryzen AI hardware. It targets binary compatibility: the same xclbin
that runs on real silicon runs in the emulator, producing identical results.

The emulator plugs into AMD's XRT runtime via a driver plugin, so existing
XRT-based test programs run against the emulator unmodified. Set `XDNA_EMU=1`
and your test binary talks to the emulator instead of the NPU.

### Design Principles

- **Derive from the toolchain.** Hardware behavior is extracted from
  open-source sources (aie-rt, llvm-aie, mlir-aie), not hardcoded. The
  TableGen-driven ISA decoder has zero hardcoded encodings. BD field parsing
  has zero hardcoded bit positions. See [CLAUDE.md](CLAUDE.md) for the full
  source hierarchy.

- **Data-driven architecture.** Instruction definitions are parsed from
  llvm-aie TableGen files into O(1) lookup tables. Register layout comes from
  the AM025 register database JSON. Device topology comes from the mlir-aie
  device model. When the toolchain evolves, the emulator evolves with it.

- **Correctness before performance.** The emulator is a development tool, not
  a hardware substitute. Getting the right answer matters more than getting it
  fast.

## Features

- **Binary parsing**: XCLBIN containers, AIE Partition, CDO command streams, ELF loading
- **VLIW execution**: 128-bit bundle decoding with 7-slot dispatch (scalar, vector, memory, control)
- **ISA coverage**: Scalar ALU, vector element ops, SRS/UPS type conversion, permutations, pack/unpack, matrix multiply
- **DMA engine**: Multi-dimensional addressing (up to 4D), BD chaining, repeat count, stream I/O, zero-padding
- **Synchronization**: Lock acquire/release with deadlock detection, cross-tile memory access
- **Stream routing**: Circuit-switched and packet-switched routing with configurable latency
- **Timing model**: AM020-based operation latencies, hazard detection, branch delay slots
- **Multi-core coordination**: Tile array orchestration, inter-tile stream routing, shim tile DMA
- **XRT plugin**: Drop-in driver plugin -- existing XRT test programs run against the emulator
- **Trace comparison**: Binary trace infrastructure for comparing emulator vs real NPU execution
- **Dual-compiler testing**: Bridge test suite validates with both Chess and Peano compilers
- **Visual debugger**: egui-based GUI showing tile status, registers, memory, locks, DMA state

## Quick Start

### Build

```bash
cargo build --release
```

### Test

```bash
# Library tests (fast, recommended)
cargo test --lib

# All tests including doc tests
./scripts/run-tests.sh
```

Run `cargo test --lib` to see the current test count. Do not rely on numbers
written in documentation -- they go stale quickly.

### Run

```bash
# Launch GUI
./target/release/xdna-emu

# Launch GUI with xclbin preloaded
./target/release/xdna-emu --gui path/to/kernel.xclbin

# CLI mode
./target/release/xdna-emu --dump-state path/to/kernel.xclbin
```

### XRT Bridge Tests

The bridge test suite exercises the full hardware-equivalent flow:
`test.exe -> XRT -> plugin -> emulator`. This is the primary validation path.

```bash
# Run all bridge tests (requires XRT + NPU hardware for HW comparison)
./scripts/emu-bridge-test.sh

# Emulator-only (no hardware needed)
./scripts/emu-bridge-test.sh --no-hw

# Single test
./scripts/emu-bridge-test.sh --no-hw add_one_using_dma
```

See [CLAUDE.md](CLAUDE.md) for XRT plugin build and install instructions.

## Target Devices

| Driver ID | Product | Codename | Architecture | Array | Status |
|-----------|---------|----------|--------------|-------|--------|
| NPU1 | Ryzen AI | Phoenix/Hawk Point | AIE2 (XDNA) | 5 cols x 6 rows | **Primary target** |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P (XDNA2) | 8+ cols x 6 rows | Planned |
| NPU6 | (TBD) | Krackan | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |

Array sizes include the shim tile row (row 0). We are building on
**Phoenix (NPU1/AIE2)** first because it is the hardware we have. AIE2P
support will be incremental -- the architectures share most of their design.

## Project Structure

```
src/
  parser/       Binary format parsers (XCLBIN, CDO, ELF)
  device/       Device state model (tiles, registers, DMA, stream switch, locks)
  interpreter/  VLIW execution engine (decode, execute, timing, coordination)
  npu/          NPU instruction processor (shim tile configuration)
  testing/      Test infrastructure and xclbin test suite runner
  trace/        Execution tracing, event logging, binary trace comparison
  vcd/          VCD waveform export and trace cycle analysis
  debug/        Debug utilities and memory watch infrastructure
  fuzzer/       Kernel fuzzer infrastructure (early)
  integration/  Toolchain integration (aiesimulator, Chess, bridge, elfanalyzer)
  visual/       egui-based visual debugger
crates/
  xdna-archspec/  TableGen ISA definitions, device model, register specs
  xdna-emu-ffi/   C FFI cdylib for the XRT plugin
xrt-plugin/     XRT driver plugin (C++, delegates to Rust emulator via FFI)
scripts/        Build, test, and bridge test scripts
tools/          Trace tooling (mlir-trace-inject, parse-trace, trace-sweep,
                trace-prepare); deprecated/ holds pre-IRON-API versions
docs/           Architecture docs, format specs, roadmap phases
```

## Documentation

| Document | Content |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Development status with confidence markers |
| [CLAUDE.md](CLAUDE.md) | Architecture guide, correctness principles, source hierarchy |
| [docs/roadmap/](docs/roadmap/) | Detailed per-phase documentation |
| [docs/formats/](docs/formats/) | Binary format specifications (XCLBIN, CDO, ELF, AIE Partition) |
| [.claude/components/](.claude/components/) | Deep-dive component docs (device, interpreter, parser, tablegen, testing, visual) |

## How It Works

### What We Emulate (Per Tile)

- **Compute Core**: VLIW processor with 8 functional slots, 256-bit vector unit
- **Local Memory**: 64 KB per compute tile, 512 KB per memory tile
- **DMA Engine**: 2 channels with multi-dimensional addressing and BD chaining
- **Locks**: Synchronization primitives with acquire/release semantics
- **Stream Switch**: Circuit-switched and packet-switched routing between tiles

### Execution Flow

1. Parse the `.xclbin` container (XCLBIN -> AIE Partition -> CDO + ELF sections)
2. Configure the tile array from CDO commands (DMA descriptors, routing, locks)
3. Load ELF binaries into per-tile memory
4. Execute all cores concurrently with coordinated DMA and stream routing
5. Compare output buffers against expected results

### XRT Plugin Architecture

The XRT plugin (`xrt-plugin/`) replaces the real XDNA kernel driver. XRT
loads the shared library, which delegates to the Rust emulator through
C FFI bindings (`src/ffi/`). From the application's perspective, it is
talking to real hardware.

## Acknowledgments

This project was written almost entirely by [Claude](https://www.anthropic.com/claude)
via [Claude Code](https://claude.ai/code), with human guidance and direction.

## License

[MIT License](LICENSE)
