# Trace Unit -- Verification Report

Audit of `src/device/trace_unit.rs` and `src/trace/mod.rs` against
aie-rt `driver/src/trace/xaie_trace.c`, `events/xaie_events_aieml.h`,
and `global/xaiemlgbl_params.h`.

## Summary

The trace unit is functionally correct after one critical encoding fix
applied during this audit. The register interface, state machine, event
ID mapping, and packet format all match aie-rt.

## Register Interface

### Trace_Control0 (offset +0x00)

| Field        | aie-rt definition                  | Our impl        | Status |
|--------------|------------------------------------|------------------|--------|
| mode[1:0]    | Core: LSB=0, mask=0x03             | `value & 0x3`    | MATCH  |
| start[22:16] | Core: 7-bit (0x7F0000)             | `(value>>16)&0xFF` | OK (1) |
| stop[30:24]  | Core: 7-bit (0x7F000000)           | `(value>>24)&0xFF` | OK (1) |

(1) We extract 8 bits; aie-rt uses 7 for core/mem, 8 for MemTile.
    The extra bit is always 0 for 7-bit modules. Not a functional issue.

### Trace_Control1 (offset +0x04)

| Field             | aie-rt definition        | Our impl              | Status |
|-------------------|--------------------------|-----------------------|--------|
| packet_type[14:12]| LSB=12, mask=0x7000      | `(value>>12)&0x7`     | MATCH  |
| packet_id[4:0]    | LSB=0, mask=0x1F         | `value & 0x1F`        | MATCH  |

### Trace_Status (offset +0x08)

Not implemented (no read_register). See catalog.

### Trace_Event0 (offset +0x10) / Trace_Event1 (offset +0x14)

| Field          | aie-rt definition              | Our impl            | Status |
|----------------|--------------------------------|----------------------|--------|
| event0[6:0]    | 7-bit at [6:0]                 | `value & 0xFF`       | OK (1) |
| event1[14:8]   | 7-bit at [14:8]                | `(value>>8)&0xFF`    | OK (1) |
| event2[22:16]  | 7-bit at [22:16]               | `(value>>16)&0xFF`   | OK (1) |
| event3[30:24]  | 7-bit at [30:24]               | `(value>>24)&0xFF`   | OK (1) |

Identical layout for Event1 with events 4-7.

## Register Routing

Dispatch in `state.rs` uses regdb-derived offsets:
- Core module: 0x340D0..0x340E4 (offset - base = 0x00..0x14) -- CORRECT
- Memory module: 0x140D0..0x140E4 -- CORRECT
- MemTile: 0x940D0..0x940E4 -- CORRECT
- Shim/PL: uses core_events base (0x340D0) -- CORRECT (same address)

## State Machine

| State     | aie-rt enum              | Our enum         | Status |
|-----------|--------------------------|------------------|--------|
| Idle      | XAIE_TRACE_IDLE (0)      | TraceState::Idle | MATCH  |
| Running   | XAIE_TRACE_RUNNING (1)   | TraceState::Running | MATCH |
| Stopped   | XAIE_TRACE_OVERRUN (2)   | TraceState::Stopped | (2)   |

(2) aie-rt calls state 2 "OVERRUN", meaning the stop event has fired or
    the buffer overflowed. We call it "Stopped" which is functionally
    equivalent for our purposes (we never overflow since we emit packets
    immediately).

Transitions:
- Idle -> Running on start_event: MATCH (aie-rt configures, HW transitions)
- Running -> Stopped on stop_event: MATCH
- Events ignored when not Running: MATCH

## Trace Mode

| Mode       | aie-rt enum            | Our enum            | Status |
|------------|------------------------|---------------------|--------|
| EventTime  | XAIE_TRACE_EVENT_TIME (0) | TraceMode::EventTime | MATCH |
| EventPC    | XAIE_TRACE_EVENT_PC (1)   | TraceMode::EventPc   | MATCH |
| Execution  | XAIE_TRACE_INST_EXEC (2)  | TraceMode::Execution | MATCH |

Memory module: Mode field is FEATURE_UNAVAILABLE (no mode bits in register).
Our code reads bits [1:0] which are always 0 -- benign.

## Event ID Mapping

All event IDs verified against `xaie_events_aieml.h`:

### Core Module
- INSTR_EVENT_0=33, INSTR_EVENT_1=34: MATCH
- MEMORY_STALL=23, STREAM_STALL=24, LOCK_STALL=26: MATCH
- ACTIVE=28, DISABLED=29: MATCH
- INSTR_VECTOR=37, INSTR_LOAD=38, INSTR_STORE=39: MATCH
- PORT_IDLE_0=74 (stride 4): MATCH
- EDGE_DETECTION_EVENT_0=13: MATCH

### Memory Module (Compute Tile)
- DMA_S2MM_0_START_TASK=19: MATCH (XAIEML_EVENTS_MEM_DMA_S2MM_0_START_TASK=19)
- LOCK_SEL0_ACQ_GE=45, LOCK_0_REL=46 (stride 4): MATCH
- CONFLICT_DM_BANK_0=77: MATCH

### MemTile Module
- DMA_S2MM_SEL0_START_TASK=21: MATCH (XAIEML_EVENTS_MEM_TILE_DMA_S2MM_SEL0_START_TASK=21)
- LOCK_SEL0_ACQ_GE=47 (stride 4): MATCH
- PORT_IDLE_0=79: MATCH
- CONFLICT_DM_BANK_0=112: MATCH

### PL/Shim Module
- DMA_S2MM_0_START_TASK=14: MATCH (XAIEML_EVENTS_PL_DMA_S2MM_0_START_TASK=14)
- LOCK_0_ACQ_GE=40 (stride 4, 6 locks): MATCH
- PORT_IDLE_0=77: MATCH

## Byte Encoding (Fixed During Audit)

### Single0: 0b0EEETTTT (1 byte)
- Event slot in bits [6:4], delta in bits [3:0]: MATCH (was always correct)

### Single1: 0b100EEETT TTTTTTTT (2 bytes)
- Event slot in bits [4:2], delta hi in bits [1:0]: FIXED
- Previously had slot in bits [6:4], causing wrong decode for slots 1-7

### Single2: 0b101EEETT TTTTTTTT TTTTTTTT (3 bytes)
- Event slot in bits [4:2], delta hi in bits [1:0]: FIXED
- Same misalignment as Single1, now corrected

### Start marker: 0xF0 + 7 bytes big-endian timer
- Format matches mlir-aie decode exactly: MATCH

### Pad byte: 0xFE
- Matches mlir-aie trim logic: MATCH

## Packet Format

8 words (32 bytes): header + 7 data words.

| Field      | mlir-aie parse.py         | Our impl           | Status |
|------------|---------------------------|---------------------|--------|
| pkt_id[4:0]  | bits [4:0]             | `pkt_id & 0x1F`    | MATCH  |
| pkt_type[12:11] | bits [12:11]        | `pkt_type << 11`    | MATCH  |
| row[20:16]    | bits [20:16]           | `row << 16`         | MATCH  |
| col[28:21]    | bits [28:21]           | `col << 21`         | MATCH  |
| parity[31]    | odd parity             | `count_ones() % 2`  | MATCH  |

Data words: 28 bytes packed big-endian into 7 u32s: MATCH.

## Performance Counters

aie-rt has separate PerfCnt configuration (not in trace module). Our
emulator does not implement perf counters. Perf counter events (IDs 5-8)
can still be configured in trace slots but will never fire. This is a
known limitation, not a divergence.

## Test Coverage

18 trace_unit tests pass, including a new exhaustive round-trip test that
verifies encoding for all 8 slots across all 3 Single formats against
the mlir-aie decode algorithm.
