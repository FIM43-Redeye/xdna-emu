# xdna-emu

Open-source emulator and visual debugger for AMD XDNA NPUs (Ryzen AI).

**Status**: ~90% binary compatible | 564 tests passing

## Features

- **Binary Parsing**: Load real `.xclbin` files (XCLBIN, AIE Partition, CDO, ELF)
- **Device Emulation**: Simulates NPU1 (Phoenix) tile arrays, NPU4-6 (Strix) planned
- **ISA Coverage**: Scalar (div, select), vector (element ops, shifts), matrix multiply, convolution, type conversion
- **Multi-Tile DMA**: Stream routing, lock synchronization, memory tile support
- **Visual Debugger**: GUI showing tile status, memory, locks, DMA, registers
- **Step Debugging**: Run, step, breakpoints, speed control

## Quick Start

```bash
# Build
cargo build --release

# Launch GUI
./target/release/xdna-emu

# Launch GUI with xclbin preloaded
./target/release/xdna-emu --gui path/to/aie.xclbin

# CLI mode: parse and dump state
./target/release/xdna-emu --dump-state path/to/aie.xclbin
```

## GUI

The visual debugger shows:
- **Left panel**: Tile array grid with color-coded status (green=running, yellow=waiting, red=halted)
- **Right panel**: Selected tile details (core state, registers, locks, DMA, memory hex view)
- **Top controls**: Run/Pause, Step, Reset, speed slider

Drag & drop `.xclbin` files onto the window to load them.

## Target Devices

| Driver ID | Product | Codename | Architecture | Array |
|-----------|---------|----------|--------------|-------|
| NPU1 | Ryzen AI | Phoenix | AIE2 | 4x5 |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P | 4x5 |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P | 8x5 |
| NPU6 | (TBD) | Krackan | AIE2P | 4x5 |

*Note: NPU2/NPU3 are prototypes, not consumer devices.*

## Project Structure

```
src/
├── parser/     # Binary format parsers (XCLBIN, CDO, ELF)
├── device/     # Device state model (tiles, registers, DMA)
├── emu/        # Emulation engine (instruction decode, execution)
└── visual/     # GUI (egui-based visual debugger)
```

## Documentation

- `docs/formats/` - Binary format specifications
- `ROADMAP.md` - Development roadmap
- `CLAUDE.md` - AI assistant context

## Acknowledgments

This project was written almost entirely by [Claude Opus 4.5](https://www.anthropic.com/claude) via [Claude Code](https://claude.ai/code), with human guidance and direction.

## License

[X11 License](LICENSE) - Any usage with attribution, advertisement barred
