# Device Model

Hardware state representation for AMD XDNA NPU tiles, arrays, and peripherals.

Read this file when working on anything in `src/device/`.

## Files

### Top-Level

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports, `AieArch` enum (Aie2/Aie2P) |
| `aie2_spec.rs` | Architecture constants derived from AM020 (register offsets, memory sizes, tile geometry) |
| `arch_config.rs` | `ArchConfig` trait and device configuration (`Aie2Config`, `Aie2pConfig`) |
| `registers.rs` | `RegisterInfo`, `RegisterModule`, `TileAddress` -- CDO address decoding |
| `registers_spec.rs` | Register specification constants (offsets, field masks) |
| `tile.rs` | `Tile` state: memory, locks, DMA BDs, core state, stream switch |
| `array.rs` | `TileArray` -- the complete device (5 cols x 6 rows for NPU1) |
| `state.rs` | `DeviceState` -- applies CDO commands to the array |
| `host_memory.rs` | `HostMemory` -- simulated DDR with named regions and direction tracking |
| `stream_switch.rs` | `StreamSwitch` -- per-tile circuit/packet routing |
| `stream_router.rs` | `StreamRouter` -- global stream routing across the array |

### DMA Subsystem (`dma/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, DMA architecture documentation, re-exports |
| `engine.rs` | `DmaEngine` -- main DMA controller, channel management, BD execution |
| `transfer.rs` | `Transfer`, `TransferState`, `TransferDirection` -- transfer state machine |
| `addressing.rs` | `AddressGenerator`, `DimensionConfig` -- multi-dimensional addressing (1D-4D) |
| `timing.rs` | `DmaTimingConfig`, `ChannelTimingState`, `TransferPhase` -- cycle-accurate DMA timing |
| `bd.rs` | Buffer descriptor configuration and parsing |
| `channel.rs` | `ChannelState`, `ChannelId`, `ChannelType` -- per-channel state |
| `stream_io.rs` | `StreamData` -- stream switch integration for DMA transfers |
| `compression.rs` | DMA data compression/decompression support |

## Key Types

- `TileArray` -- the full NPU device; created via `TileArray::npu1()`
- `Tile` -- individual tile state (compute, mem, or shim)
- `TileAddress` -- CDO address decoder (`TileAddress::decode(0x02232000)`)
- `DeviceState` -- wraps `TileArray` and applies CDO configuration
- `DmaEngine` -- per-tile DMA with 16 BDs and 4 channels
- `StreamRouter` -- routes stream data between tiles
- `HostMemory` -- DDR simulation with `MemoryRegion` tracking

## Tile Types

Row 0 = shim (DDR interface), row 1 = mem tile (512KB), rows 2-5 = compute (64KB each).

## Conventions and Gotchas

- **Address decoding**: CDO addresses encode column/row/offset in a single u32. `TileAddress::decode()` extracts these. See `registers.rs` for the bit layout.
- **Lock spacing**: Lock registers are 16 bytes apart (not 4). Core lock IDs 48-63 map to memory module locks 0-15.
- **DMA addressing**: Uses 32-bit word units, not byte addresses. Multiply by 4 when comparing to byte offsets.
- **BD fields**: Span multiple 32-bit words with specific bit layouts defined in AM020. See `bd.rs` for field extraction.

## Known Issues

- **DMA dual abstraction**: Both `channel.rs` (`ChannelState`) and `engine.rs` (`ChannelState` within `DmaEngine`) track channel state. These need to be unified.
- **Unwrap calls**: `engine.rs` has unwrap() calls that should use expect() with descriptive messages.
- **BD field parsing**: Correctness depends on AM020 interpretation that has not been cross-checked against the aie-rt source.

## Architecture References

- aie-rt: `../aie-rt/driver/src/` (official Xilinx, branch xlnx_rel_v2025.2)
- AM020: AIE-ML Architecture Manual (DMA, locks, memory layout)
- AM025: Register Reference (offsets, field definitions)
- Extracted docs: `docs/xdna/` (text format, use Explore agents to navigate)

## Trace Unit

The trace subsystem (`src/trace/`) handles emulator event tracing:
- `mod.rs` -- Perfetto JSON trace output, event recording
- `store.rs` -- trace event storage
- `compare.rs` -- binary trace comparison (HW vs EMU)
- `vcd.rs` -- aiesimulator VCD trace parser

The trace unit records DMA transfers, lock operations, core events, and
stream switch activity. Output is Perfetto JSON viewable at ui.perfetto.dev.
