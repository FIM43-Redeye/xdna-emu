# Control Packet Implementation Design

**Date**: 2026-03-03
**Status**: Design approved, implementation pending

## Context

Control packets are the NPU's runtime reconfiguration mechanism. They flow
through the packet-switched stream network and write (or read) tile registers
dynamically -- while cores are running. This enables:

- Setting lock values to gate core execution stages
- Reconfiguring DMA buffer descriptors and channels mid-flight
- Changing stream switch routing at runtime
- Reading tile state back to the host

The emulator already has substantial control packet infrastructure:

- **Working**: Packet switch routing (packet_enable fix from 76f6946), TileCtrl
  master port identification and draining (array.rs), control packet state
  machine (header parse -> data collect -> execute), OP_WRITE / OP_BLOCK_WRITE
  / OP_WRITE_INCR execution, CDO packet slot and master packet configuration.

- **Broken**: Register dispatch gap (Section 1), OP_READ not implemented
  (Section 2).

### Test Landscape

| Test | Status | Issue |
|------|--------|-------|
| add_one_ctrl_packet | XFAIL | Opaque input (test runner), possibly OP_READ |
| add_one_ctrl_packet_4_cores | XFAIL | Same |
| add_one_ctrl_packet_col_overlay | XFAIL | Same + column overlay |
| ctrl_packet_reconfig | Build XFAIL | Peano legalization bug (llvm-aie#480) |
| ctrl_packet_reconfig_1x4_cores | Unknown | Needs reconfig support |
| ctrl_packet_reconfig_4x1_cores | Unknown | Needs reconfig support |

Chess-compiled variants of add_one_ctrl_packet already pass (16/16) as
unexpected passes, suggesting the hardware path works when data is correct.

## Design

### Section 1: Register Dispatch Fix

**Problem**: Control packet writes call `tile.write_register(offset, value)`
directly, which handles only a subset of register types (compute-tile DMA BDs,
locks, basic channel control). CDO writes go through
`DeviceState::write_register()` which does full module-based dispatch including
MemTile DMA BDs, MemTile stream switch, core module registers, and shim DMA
channels.

For ctrl_packet_reconfig, control packets rewrite MemTile DMA BDs and channels
at runtime. Without full dispatch, these writes update the register HashMap but
never modify the structured BD/channel state that the DMA engine actually reads.

**Fix**: Add `DeviceState::ctrl_packet_write(col, row, offset, value)` that
reconstructs the full tile address and routes through the existing module
dispatch. `execute_ctrl_packet()` calls this instead of `self.write_register()`.

This requires `execute_ctrl_packet()` to have access to `DeviceState` rather
than operating on `Tile` alone. The cleanest approach: move control packet
execution to `array.rs` where `DeviceState` context is available, or pass a
callback/closure that routes writes through `DeviceState`.

Implementation choice: use a closure or trait object passed from
`route_tile_switches_to_ctrl()` in array.rs. The tile's
`process_ctrl_packet_word()` collects the header and data (stateful), then
calls a provided write function when execution is needed. This keeps the state
machine in Tile but routes writes through DeviceState.

### Section 2: OP_READ + Response Routing

**Hardware behavior**: When a control packet carries operation=1 (READ), the
tile reads N consecutive 32-bit registers starting at the header address, then
generates a response packet that flows back through the stream switch to the
originator (typically shim DMA S2MM).

**Response packet format** (deduced from add_one_ctrl_packet test.cpp):
1. Stream packet header (pkt_id = response_id from ctrl pkt header)
2. Data words (N register values, where N = beats from ctrl pkt header)

**Response routing path**:
```
Tile (0,2) TileCtrl slave port
  -> stream switch packet routing (configured via CDO packet_flow)
  -> MemTile stream switch
  -> Shim stream switch
  -> Shim DMA S2MM channel
  -> Host memory (ctrlOut buffer)
```

**Implementation**:
1. `execute_ctrl_packet()` for OP_READ: read N registers via `read_register()`
2. Build response: encode stream packet header + raw data words
3. Push response into TileCtrl slave port on the tile's stream switch
4. Existing packet routing infrastructure handles the rest

The TileCtrl slave port needs to be identified (port 0 on compute tiles per
stream_switch.rs:460). Data pushed here enters the packet routing pipeline
where configured slave slots and master packet configs route it to the correct
destination.

**Open question**: Does the response include a control packet header, or just
raw data? The test expects 8 values in ctrlOut (2 reads x 4 beats each). If
each read produces 4 data words with no header overhead, the response is just
data. If there's a control packet header, it would be 5 words per read (1
header + 4 data). The drop_header configuration on the TileCtrl master port
determines whether the stream packet header is forwarded.

### Section 3: Test Verification via mock_xrt

Instead of fixing the npu-test runner's Opaque input problem, verify through
mock_xrt test.exe which provides correct binary compatibility:

- test.exe constructs control packet headers with proper bitfields
- test.exe opens ctrlpkt.bin for reconfig tests
- mock_xrt passes buffers to emulator via FFI
- Shim DMA reads host memory and sends through packet switch

Verification steps:
1. Build mock_xrt (CMake)
2. Compile add_one_ctrl_packet/test.cpp against mock_xrt
3. Run test.exe and check output
4. Repeat for ctrl_packet_reconfig variants

### Section 4: ctrl_packet_reconfig Support

The reconfig tests use pre-compiled ctrlpkt.bin (generated by aiecc.py with
`--aie-generate-ctrlpkt`). The control packet sequence rewrites DMA BDs and
channel configurations to switch between kernels at runtime.

With the register dispatch fix (Section 1), this should work automatically:
control packet writes to DMA BD registers go through the same path as CDO
writes, updating structured state. DMA engine picks up new BD configuration
on next channel start.

Column overlay support (base + main device architecture) may require additional
CDO parsing work if the overlay format differs from standard CDO. This is
deferred if it proves complex.

## Files to Modify

| File | Change |
|------|--------|
| `src/device/tile.rs` | Split execute_ctrl_packet into collect + dispatch; accept write callback |
| `src/device/array.rs` | Pass DeviceState write function to ctrl packet handler |
| `src/device/state.rs` | Add ctrl_packet_write(col, row, offset, value) method |
| `src/device/stream_switch.rs` | Identify TileCtrl slave port for response injection |
| `tests/test_overrides.toml` | Update XFAIL expectations as tests start passing |

## Not In Scope

- Parity validation (calculated in tests but not checked on real hardware)
- Async execution in mock_xrt (currently synchronous, sufficient for testing)
- npu-test runner Opaque input parsing (mock_xrt handles this)
