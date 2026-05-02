# Device Model

Hardware state representation for AMD XDNA NPU tiles, arrays, and peripherals.

Read this file when working on anything in `src/device/`.

## Layout

The post-refactor device layout is mostly directories of small modules.
The flat-file layout an earlier version of this doc described no longer
matches reality.

### Top-Level (`src/device/`)

| Path | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports, and `AieArch` indirection through `arch_handle.rs` |
| `arch_handle.rs` | `ArchHandle` -- runtime device handle that fans out to the right `xdna_archspec` model |
| `model.rs` | Device-level model glue (size, tile-kind classification, lookups) |
| `registers.rs` | `RegisterInfo`, `RegisterModule`, `TileAddress` -- CDO address decoding |
| `host_memory.rs` | `HostMemory` -- simulated DDR with named regions and direction tracking |
| `banking.rs` | Memory bank conflict modeling |
| `ops.rs` | `DeviceOp` -- the 8-variant operation vocabulary that CDO and NPU paths share |
| `timer.rs` | Tile timer model |
| `aiert_validation.rs` | aie-rt validation hooks |

### Subsystem Directories

| Path | Purpose |
|------|---------|
| `state/` | `DeviceState` -- applies `DeviceOp` to the array; per-tile compute/memtile dispatch and effects |
| `array/` | `TileArray` -- the device tile grid, with control + DMA + routing helpers |
| `tile/` | `Tile` state: memory, locks, registers, edge config, per-tile streams, core state |
| `dma/` | DMA engine: BDs, transfer state machine, addressing, timing, stream I/O, FIFO, token, compression |
| `stream_switch/` | Per-tile and global stream routing |
| `regdb/` | Register database glue (consumes the AM025 JSON via xdna-archspec) |
| `events/` | Event tracking (broadcasts, timer-triggered events, port events) |
| `perf_counters/` | Performance counter banks (per-tile) |
| `trace_unit/` | Trace unit hardware model: packet stream emission for HW-compatible traces |
| `core_debug/` | Per-core debug state (stalls, halts, instrumentation) |
| `control_packets/` | Control packet handling (kernel-driven register writes) |
| `interrupts/` | Interrupt model |

## Key Types

- `TileArray` -- the full NPU device; created via `TileArray::npu1()`
- `Tile` -- individual tile state (compute, mem, or shim)
- `TileAddress` -- CDO address decoder (`TileAddress::decode(0x02232000)`)
- `DeviceState` -- wraps `TileArray` and applies `DeviceOp`s
- `DeviceOp` -- 8-variant op vocabulary the CDO parser and NPU instruction
  executor both feed (Write32, BlockWrite, MaskWrite, MaskPoll, ...)
- `DmaEngine` -- per-tile DMA with 16 BDs and 4 channels
- `HostMemory` -- DDR simulation with `MemoryRegion` tracking
- `ArchHandle` -- runtime arch indirection (Aie2 vs Aie2P)

## Tile Types

Row 0 = shim (DDR interface), row 1 = mem tile (512KB), rows 2-5 = compute (64KB each).

## Conventions and Gotchas

- **Address decoding**: CDO addresses encode column/row/offset in a single u32.
  `TileAddress::decode()` extracts these. See `registers.rs` for the bit layout.
- **Lock spacing**: Lock registers are 16 bytes apart (not 4). Core lock IDs
  48-63 map to memory module locks 0-15.
- **DMA addressing**: Uses 32-bit word units, not byte addresses. Multiply by 4
  when comparing to byte offsets.
- **BD fields**: Span multiple 32-bit words with bit layouts defined in the
  AM025 register database JSON. `bd.rs` field extraction is fully data-driven
  (zero hardcoded bit positions).
- **DeviceOp**: When adding a new write path, prefer adding a `DeviceOp`
  variant over reaching directly into `Tile::registers`. The same op flows
  through CDO parsing and NPU instruction execution -- keeping the vocabulary
  unified is what made the refactor worthwhile.

## Architecture References

- aie-rt: `../aie-rt/driver/src/` (official Xilinx, branch xlnx_rel_v2025.2)
- xdna-archspec: `crates/xdna-archspec/` (the runtime arch source-of-truth)
- AM020: AIE-ML Architecture Manual (DMA, locks, memory layout)
- AM025: Register Reference (offsets, field definitions; consumed as JSON)
- Extracted docs: `docs/xdna/` (text format, use Explore agents to navigate)

## Trace Unit (`src/device/trace_unit/`)

The trace unit emits HW-compatible packet streams from EMU-side events
(DMA transfers, lock operations, core events, stream switch activity,
broadcasts). Output is decoded by the same mlir-aie `parse_trace`
machinery that handles real-NPU traces, then converted to flat events
JSON or Perfetto JSON via `tools/parse-trace.py`.

The comparison side lives in `src/trace/` (mode-0 `compare.rs`, mode-2
`compare_mode2.rs` + `mode2_decode.rs`, VCD `vcd.rs`).
