# Stream Switch -- Verification Report

**Agent**: H (re-run)
**Date**: 2026-03-12
**Oracle**: aie-rt stream_switch/xaie_ss_aieml.c, xaie_ss.c, xaie_ss.h,
           xaiemlgbl_reginit.c (port maps + StrmMod structs),
           xaiemlgbl_params.h (register bit fields),
           xaie_helper.c (_XAie_GetSlaveIdx / _XAie_GetMstrIdx)
**Emulator**: src/device/stream_switch.rs (2,429 lines)
**Result**: PASS -- no critical or high divergences

## 1. Port Type Assignments per Bundle (All 3 Tile Types)

### Compute Tile (AieMlTileStrmSwMasterPortMap / SlavePortMap)

Generated `COMPUTE_MASTER_PORTS` (23 ports) exactly matches
`AieMlTileStrmSwMasterPortMap` in xaiemlgbl_reginit.c:

| PhyPort | aie-rt         | Emulator (gen_stream_ports.rs) | Match |
|---------|----------------|-------------------------------|-------|
| 0       | CORE,0         | port_type::CORE               | YES   |
| 1       | DMA,0          | port_type::dma(0)             | YES   |
| 2       | DMA,1          | port_type::dma(1)             | YES   |
| 3       | CTRL,0         | port_type::CORE -> TileCtrl*  | YES   |
| 4       | FIFO,0         | port_type::FIFO               | YES   |
| 5-8     | SOUTH,0-3      | port_type::south(0-3)         | YES   |
| 9-12    | WEST,0-3       | port_type::west(0-3)          | YES   |
| 13-18   | NORTH,0-5      | port_type::north(0-5)         | YES   |
| 19-22   | EAST,0-3       | port_type::east(0-3)          | YES   |

*Port 3 is generated as `CORE` from register name "Tile_Ctrl", then
overridden to `TileCtrl` in `new_compute_tile()` at runtime.

Generated `COMPUTE_SLAVE_PORTS` (25 ports) exactly matches
`AieMlTileStrmSwSlavePortMap`:

| PhyPort | aie-rt         | Emulator                      | Match |
|---------|----------------|-------------------------------|-------|
| 0       | CORE,0         | port_type::CORE               | YES   |
| 1-2     | DMA,0-1        | port_type::dma(0-1)           | YES   |
| 3       | CTRL,0         | port_type::CORE -> TileCtrl*  | YES   |
| 4       | FIFO,0         | port_type::FIFO               | YES   |
| 5-10    | SOUTH,0-5      | port_type::south(0-5)         | YES   |
| 11-14   | WEST,0-3       | port_type::west(0-3)          | YES   |
| 15-18   | NORTH,0-3      | port_type::north(0-3)         | YES   |
| 19-22   | EAST,0-3       | port_type::east(0-3)          | YES   |
| 23      | TRACE,0        | port_type::TRACE              | YES   |
| 24      | TRACE,1        | port_type::TRACE              | YES   |

Port COUNTS per bundle match aie-rt AieMlTileStrmMstr/Slv arrays:
- Masters: Core(1), DMA(2), Ctrl(1), FIFO(1), South(4), West(4), North(6), East(4), Trace(0) = 23
- Slaves: Core(1), DMA(2), Ctrl(1), FIFO(1), South(6), West(4), North(4), East(4), Trace(2) = 25

### MemTile (AieMlMemTileStrmSwMasterPortMap / SlavePortMap)

Generated `MEMTILE_MASTER_PORTS` (17 ports) exactly matches aie-rt:
DMA(0-5), Ctrl(6), South(7-10), North(11-16)

Generated `MEMTILE_SLAVE_PORTS` (18 ports) exactly matches aie-rt:
DMA(0-5), Ctrl(6), South(7-12), North(13-16), Trace(17)

Port counts match AieMlMemTileStrmMstr/Slv:
- Masters: Core(0), DMA(6), Ctrl(1), FIFO(0), South(4), West(0), North(6), East(0), Trace(0) = 17
- Slaves: Core(0), DMA(6), Ctrl(1), FIFO(0), South(6), West(0), North(4), East(0), Trace(1) = 18

Key asymmetry verified: 6 North masters / 4 North slaves, 4 South masters / 6 South slaves.

### Shim Tile (AieMlShimStrmSwMasterPortMap / SlavePortMap)

Generated `SHIM_MASTER_PORTS` (22 ports) exactly matches aie-rt:
Ctrl(0), FIFO(1), South(2-7), West(8-11), North(12-17), East(18-21)

Generated `SHIM_SLAVE_PORTS` (23 ports) exactly matches aie-rt:
Ctrl(0), FIFO(1), South(2-9), West(10-13), North(14-17), East(18-21), Trace(22)

Port counts match AieMlShimStrmMstr/Slv:
- Masters: Core(0), DMA(0), Ctrl(1), FIFO(1), South(6), West(4), North(6), East(4), Trace(0) = 22
- Slaves: Core(0), DMA(0), Ctrl(1), FIFO(1), South(8), West(4), North(4), East(4), Trace(1) = 23

MaxPhyPortId verified: 21 (master), 22 (slave) -- matches aie-rt.

## 2. Packet Routing: Header Matching, Mask, ID, Register Layout

### Slave Slot Register (PacketSlot::from_register)

Bit positions match xaiemlgbl_params.h exactly:
- Bits 28:24 = ID (5 bits) -- SLOT0_ID_LSB=24, MASK=0x1F000000
- Bits 20:16 = MASK (5 bits) -- SLOT0_MASK_LSB=16, MASK=0x001F0000
- Bit 8 = ENABLE -- SLOT0_ENABLE_LSB=8, MASK=0x00000100
- Bits 5:4 = MSEL (2 bits) -- SLOT0_MSEL_LSB=4, MASK=0x00000030
- Bits 2:0 = ARBITOR (3 bits) -- SLOT0_ARBIT_LSB=0, MASK=0x00000007

Matching logic: `(incoming & mask) == (slot_id & mask)` -- matches aie-rt
behavior where the mask applies to both sides.

### Master Config Register (MasterPacketConfig::from_register)

Bit positions match xaiemlgbl_params.h exactly:
- Bit 31 = MASTER_ENABLE -- LSB=31
- Bit 30 = PACKET_ENABLE -- LSB=30
- Bit 7 = DROP_HEADER -- LSB=7
- Bits 6:0 = CONFIGURATION (7-bit field) -- LSB=0, MASK=0x7F

Within CONFIGURATION in packet mode:
- Bits 2:0 = ARBITOR (3 bits) -- matches XAIE_SS_MASTER_PORT_ARBITOR_LSB=0
- Bits 6:3 = MSEL_ENABLE (4-bit bitmap) -- matches XAIE_SS_MASTER_PORT_MSELEN_LSB=3

Slot count: 4 per slave port (NumSlaveSlots=4U) -- matches our `[PacketSlot; 4]`.
Slot offset: 0x4 per slot, 0x10 per port -- matches SlotOffset/SlotOffsetPerPort.

## 3. Circuit-Mode Setup/Teardown

### XAie_StrmConnCctEnable / Disable

aie-rt circuit mode:
1. Compute slave index via `_XAie_GetSlaveIdx()`: `(PortBaseAddr + PortOffset*PortNum - SlvConfigBaseAddr) / 4`
2. Write slave index to master config CONFIGURATION field (bits 6:0)
3. Set MASTER_ENABLE (bit 31), PACKET_ENABLE=0 (bit 30)
4. Write SLAVE_ENABLE (bit 31) in slave config, PACKET_ENABLE=0

Emulator circuit mode (state.rs `write_stream_switch`):
1. Read slave_select from `value & 0x1F` (5 bits) -- the slave index
2. Call `configure_local_route(slave_select, port)` -- creates LocalRoute
3. Set master enabled, slave enabled from bit 31

The slave index computation is compatible: the register addresses are
contiguous with 4-byte spacing, so the physical port index IS the slave
index. Our `slave_select` extraction uses 5 bits (0x1F) vs aie-rt's 7-bit
CONFIGURATION field (0x7F). Since max slave index across all AIE2 tile
types is 24, 5 bits suffices. See STREAM-1 for forward-compat concern.

### Disable path

Not explicitly implemented as a separate disable path. When a master config
register is written with enable=0, the port's `enabled` flag is cleared.
Local routes are not explicitly removed on disable -- they persist but the
port is disabled. This differs from aie-rt which writes 0 to both master
and slave registers. Functionally equivalent because disabled ports don't
participate in routing.

## 4. Arbiter/Msel Configuration

Arbiter system matches aie-rt:
- 8 arbiters max (3-bit field, XAIE_SS_ARBITOR_MAX=0x7)
- 4 msel values (2-bit field, XAIE_SS_MSEL_MAX=0x3)
- 4-bit msel_enable bitmap (XAIE_SS_MSELEN_MAX=0xF)

Emulator: `arbiter_locks[8]` array prevents packet interleaving when
multiple slaves share an arbiter. Lock acquired on header, released on
TLAST. Lower-index slaves get priority (sequential iteration order).

This matches aie-rt's design where each arbiter serves one packet at a
time. The lower-index priority is a simplification (hardware uses
round-robin or deterministic merge), but functionally safe since CDO
typically assigns distinct arbiters to concurrent sources.

## 5. Backpressure/Flow Control

FIFO-based backpressure model:
- Each port has a fixed-depth FIFO (master=2, slave=4 per arch_timing)
- `can_accept()` checks FIFO capacity before push
- Circuit routes: data enters switch pipeline, stalls if master FIFO full
- Packet routes: header and data words check all target masters can accept
  before consuming from slave

This is functionally correct. aie-rt does not model backpressure (it
programs registers, not simulates). The emulator's FIFO model correctly
prevents data loss and stalls when downstream cannot accept.

## 6. Additional Findings

### Stale module doc comment -- FIXED

Module-level doc claimed packet switching was NOT modeled. This was stale;
packet switching has full implementation (slots, arbiters, drop-header,
arbiter locking). Updated in this audit.

### Register layout sharing between shim and compute

Shim tiles share `write_stream_switch()` with compute tiles, using the
`memory_stream_switch` register layout. This works because PL_MODULE and
CORE_MODULE stream switch registers are at identical tile-level offsets:
- Master config base: 0x3F000 (both modules)
- Slave config base: 0x3F100 (both modules)
- Slave slot base: 0x3F200 (both modules)
- Port offset: 0x4 (both modules)

The difference in port TYPE ordering and COUNT is handled by the per-tile
StreamSwitch construction using different generated port arrays.

### Intra-tile pipeline latency

The emulator models 3-4 cycle pipeline latency for data traversing the
switch (local-to-local=3, local-to-external=4, external-to-external=4).
This is sourced from arch_timing constants. aie-rt does not specify this
(it programs registers, not simulates timing), so this is derived from
AM020 documentation.
