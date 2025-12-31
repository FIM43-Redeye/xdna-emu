# xdna-emu

Open-source emulator and visual debugger for AMD XDNA NPUs (Ryzen AI).

**Status**: ~85% binary compatible | 564 tests passing

## Features

- **Binary Parsing**: Load real `.xclbin` files (XCLBIN, AIE Partition, CDO, ELF)
- **Device Emulation**: Simulates NPU1/NPU2 tile arrays (5x6 grid)
- **ISA Coverage**: Scalar, vector, matrix multiply, convolution (VMAC/VMSC), type conversion
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

| Device | Codename | Architecture | Array |
|--------|----------|--------------|-------|
| NPU1 | Phoenix/HawkPoint | AIE2 | 5x6 |
| NPU2 | Strix | AIE2P | 5x6 |
| NPU3 | Strix Halo | AIE2P | 9x6 |
| NPU4 | Krackan | AIE2P | 5x6 |

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

[Unlicense](LICENSE) - Public Domain
