# xdna-emu

Open-source emulator for AMD XDNA NPUs (Ryzen AI). Binary-compatible execution engine with TableGen-driven ISA decoding, mock XRT integration, and visual debugging.

**Status**: 1,000+ unit tests | 77K lines Rust | Active daily development

## What This Is

A from-scratch emulator that loads real `.xclbin` binaries and executes them faithfully against the XDNA architecture. The goal is a drop-in development and testing backend for the AMD NPU ecosystem — write a kernel, emulate it, validate it, then run it on hardware.

### Key Technical Decisions

- **TableGen-driven ISA decoder**: Instruction definitions are parsed directly from [llvm-aie](https://github.com/Xilinx/llvm-aie) TableGen files and resolved into O(1) lookup tables. Zero hardcoded encodings.
- **Register database from AM025**: BD field parsing and register layout are fully data-driven from the mlir-aie register database JSON. Zero hardcoded bit positions.
- **Mock XRT + C FFI bridge**: A reimplementation of the XRT API (`mock_xrt/`) backed by the emulator through a C FFI layer (`include/xdna_emu.h`), enabling existing XRT-based test programs to run against the emulator unmodified.
- **Derive from the toolchain**: All hardware behavior is derived from open-source sources (aie-rt, llvm-aie, mlir-aie) and hardware observation. See [CLAUDE.md](CLAUDE.md) for the full source hierarchy.

## Features

- **Binary parsing**: XCLBIN, AIE Partition, CDO command streams, ELF loading
- **VLIW execution**: 128-bit bundle decoding with 7-slot dispatch (scalar, vector, memory, control)
- **ISA coverage**: Scalar ALU, vector element ops, SRS/UPS type conversion, permutations, pack/unpack, matrix multiply
- **DMA engine**: Multi-dimensional addressing, BD chaining, repeat count, stream I/O, compression
- **Synchronization**: Lock acquire/release with deadlock detection, barriers, cross-tile memory latency
- **Stream routing**: Circuit-switched and packet-switched routing with configurable latency
- **Timing model**: AM020-based operation latencies, hazard detection, bank conflict modeling
- **Multi-core coordination**: Tile array orchestration, inter-tile stream routing, shim tile DMA
- **Visual debugger**: egui-based GUI showing tile status, registers, memory, locks, DMA state

## Quick Start

```bash
cargo build --release

# Launch GUI
./target/release/xdna-emu

# Launch GUI with xclbin preloaded
./target/release/xdna-emu --gui path/to/aie.xclbin

# CLI mode
./target/release/xdna-emu --dump-state path/to/aie.xclbin
```

## Project Structure

```
src/
├── parser/       # Binary format parsers (XCLBIN, CDO, ELF)
├── device/       # Device state model (tiles, registers, DMA, stream switch)
├── interpreter/  # VLIW execution engine (decode, execute, timing, coordination)
├── tablegen/     # TableGen parser and encoding resolver (from llvm-aie)
├── npu/          # NPU instruction processor (shim tile configuration)
├── testing/      # Test infrastructure and xclbin test suite runner
├── trace/        # Execution tracing and event logging
├── ffi/          # C FFI bindings for external integration
├── integration/  # Toolchain integration (Chess/Peano builds)
└── visual/       # egui-based visual debugger
mock_xrt/         # Mock XRT library (C++) backed by emulator
include/          # C FFI header (xdna_emu.h)
```

## Target Devices

| Driver ID | Product | Codename | Architecture | Array |
|-----------|---------|----------|--------------|-------|
| NPU1 | Ryzen AI | Phoenix | AIE2 | 4x5 |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P | 4x5 |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P | 8x5 |
| NPU6 | (TBD) | Krackan | AIE2P | 4x5 |

Primary target is **Phoenix (NPU1/AIE2)**. AIE2P support is planned and incremental — the architectures share most of their design.

## Documentation

- [ROADMAP.md](ROADMAP.md) — Development status with confidence markers (VERIFIED / OBSERVED / CLAIMED)
- [CLAUDE.md](CLAUDE.md) — Architecture guide and correctness principles
- `docs/roadmap/` — Detailed per-phase documentation
- `docs/formats/` — Binary format specifications

## Acknowledgments

This project was written almost entirely by [Claude](https://www.anthropic.com/claude) via [Claude Code](https://claude.ai/code), with human guidance and direction.

## License

[X11 License](LICENSE) — Any usage with attribution, advertisement barred
