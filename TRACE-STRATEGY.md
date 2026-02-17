# Trace-Driven Validation Strategy

## The Goal

Build a **logic fuzzer** that generates valid AIE2 kernels, runs them on both
the emulator and real NPU hardware, collects per-tile execution traces from
each, and **automatically flags any behavioral difference**. This is our path
to 100% hardware compatibility.

```
                    +------------------+
                    |  Logic Fuzzer    |
                    |  (valid kernels) |
                    +--------+---------+
                             |
                     kernel.xclbin
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |    Emulator      |          |    Real NPU     |
     |  (xdna-emu)      |          |  (via XRT)      |
     +--------+---------+          +--------+---------+
              |                             |
        trace.json                    trace.json
        (Perfetto)                    (Perfetto)
              |                             |
              +--------------+--------------+
                             |
                    +--------v--------+
                    |  Trace Differ   |
                    |  (flag deltas)  |
                    +-----------------+
```

Any difference in event ordering, timing, lock behavior, DMA sequencing, or
output data is a bug in the emulator. The fuzzer runs continuously, shrinking
failing cases to minimal reproducers.

## Why This Works

The AIE2 hardware has **built-in trace units in every tile**. Each tile can
record 8 configurable events (lock acquire, DMA start, instruction class,
port activity, stalls, etc.) into a packet-switched trace stream that gets
routed through the stream switch network to DDR. The mlir-aie project already
has a complete Python toolchain for configuring, collecting, and parsing these
traces into Chrome Trace Event Format (viewable in Perfetto).

If we emit the same format from the emulator, we get **apple-to-apple
comparisons** between simulated and real execution. No need to build custom
tooling -- the infrastructure already exists.

---

## What Exists Today

### Hardware Trace Infrastructure (mlir-aie)

The AIE2 trace system is production-grade and extensively documented.

**Key files:**

| File | Purpose |
|------|---------|
| `mlir-aie/python/utils/trace/setup.py` | Generates trace configuration (CDO register writes) |
| `mlir-aie/python/utils/trace/parse.py` | Parses raw trace packets to Perfetto JSON |
| `mlir-aie/python/utils/trace/events/aie2.py` | AIE2 event code definitions |
| `mlir-aie/python/utils/trace/utils.py` | Packet de-interleaving, byte stream decoding |
| `mlir-aie/python/utils/trace/config.py` | TraceConfig class for end-to-end workflow |

**Trace hardware per tile:**
- 8 configurable event slots per tile
- Packet-switched routing through stream switch to shim DMA to DDR
- Timer synchronization via broadcast events across all tiles
- Three trace types per compute tile: core module, memory module, combined

**Available events (partial list):**

| Category | Events |
|----------|--------|
| Instructions | `INSTR_VECTOR`, `INSTR_LOAD`, `INSTR_STORE`, `INSTR_CALL`, `INSTR_RETURN` |
| Locks | `INSTR_LOCK_ACQUIRE_REQ`, `INSTR_LOCK_RELEASE_REQ`, `LOCK_STALL` |
| Stalls | `MEMORY_STALL`, `STREAM_STALL`, `CASCADE_STALL` |
| DMA | `DMA_S2MM_x_START_TASK`, `DMA_S2MM_x_FINISHED_TASK`, `DMA_MM2S_x_START_TASK` |
| Ports | `PORT_RUNNING_x`, `PORT_IDLE_x`, `PORT_STALLED_x`, `PORT_TLAST_x` |
| User | `USER_EVENT_0` through `USER_EVENT_3`, `INSTR_EVENT_0`, `INSTR_EVENT_1` |
| Broadcast | `BROADCAST_0` through `BROADCAST_15` |

**Trace control registers:**

| Register | Address (core) | Purpose |
|----------|---------------|---------|
| Trace Control 0 | 0x340D0 | Start/stop events, mode (event-time, event-PC, execution) |
| Trace Control 1 | 0x340D4 | Packet type and ID for routing |
| Trace Event Group 0 | 0x340E0 | Events 0-3 (7 bits each, packed into 32 bits) |
| Trace Event Group 1 | 0x340E4 | Events 4-7 |
| Port Selection 0 | 0x3FF00 | Map stream switch ports to port events 0-3 |
| Port Selection 1 | 0x3FF04 | Map stream switch ports to port events 4-7 |
| Timer Control | 0x34000 | Event to reset timer on (for synchronization) |

Memory module uses base 0x14xxxx, memtile uses 0x94xxxx.

### Output Format: Chrome Trace Event / Perfetto JSON

```json
[
  {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "core_trace for tile(0,2)"}},
  {"name": "thread_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "INSTR_VECTOR"}},
  {"name": "thread_name", "ph": "M", "pid": 0, "tid": 7, "args": {"name": "LOCK_STALL"}},
  {"name": "LOCK_STALL",  "ts": 1,    "ph": "B", "pid": 0, "tid": 7, "args": {}},
  {"name": "LOCK_STALL",  "ts": 2065, "ph": "E", "pid": 0, "tid": 7, "args": {}},
  {"name": "INSTR_VECTOR","ts": 2070, "ph": "B", "pid": 0, "tid": 0, "args": {}},
  {"name": "INSTR_VECTOR","ts": 2071, "ph": "E", "pid": 0, "tid": 0, "args": {}}
]
```

Fields:
- `ph`: Phase -- `"M"` (metadata), `"B"` (begin), `"E"` (end)
- `pid`: One per (tile, trace_type) pair
- `tid`: Event slot index (0-7), mapped to event name via metadata
- `ts`: Timestamp in cycles

Viewable at https://ui.perfetto.dev/ -- drag and drop the JSON file.

### XRT Runtime Profiling

| Tool | Location | Purpose |
|------|----------|---------|
| `xrt::aie::profiling` | `/opt/xilinx/xrt/include/xrt/xrt_aie.h` | Performance counter API |
| `xrt-tracer` | `/opt/xilinx/xrt/bin/unwrapped/xrt-tracer` | Binary trace capture |
| `npu_perf_trace.sh` | `/opt/xilinx/xrt/share/amdxdna/` | perf-based kernel tracing |
| `npu_perf_analyze.sh` | `/opt/xilinx/xrt/share/amdxdna/` | Trace analysis |

### Kernel Driver Tracing

| Component | Location | Purpose |
|-----------|----------|---------|
| Tracepoints | `xdna-driver/src/driver/amdxdna/amdxdna_trace.h` | Job lifecycle, mailbox, IRQ |
| Debugfs | `aie2_debugfs.c` | FW log/trace dump, telemetry, health |
| FW trace | `amdxdna_dpt.h/c` | Firmware circular buffer tracing |
| Event schema | `tools/bins/configs/trace_events.json` | 27 event types, 9 categories |

---

## Implementation Plan

### Phase 1: Emulator Trace Output (Perfetto JSON)

**Goal:** Make xdna-emu emit the same Perfetto JSON format as the hardware
trace parser. This is purely emulator-side work with no hardware dependency.

**What we have today:** `EventLog` in `context.rs` already records timestamped
events per core (instruction start/complete, lock acquire/release/contention,
DMA start/complete, branch taken, memory conflict, halt). It just needs a
Perfetto JSON serializer.

**Mapping emulator events to hardware trace events:**

| Emulator EventType | Hardware Trace Event | Notes |
|--------------------|---------------------|-------|
| `InstructionStart { pc }` | `INSTR_VECTOR`, `INSTR_LOAD`, etc. | Need to classify by instruction type |
| `LockAcquireStart` | `INSTR_LOCK_ACQUIRE_REQ` | Direct mapping |
| `LockAcquired` | (end of LOCK_STALL) | |
| `LockContention` | `LOCK_STALL` (begin) | |
| `LockReleased` | `INSTR_LOCK_RELEASE_REQ` | Direct mapping |
| `DmaStart` | `DMA_S2MM_x_START_TASK` / `DMA_MM2S_x_START_TASK` | Need channel direction |
| `DmaComplete` | `DMA_S2MM_x_FINISHED_TASK` / `DMA_MM2S_x_FINISHED_TASK` | |
| `MemoryConflict` | `MEMORY_STALL` | |
| `BranchTaken` | (no direct hardware trace event) | Could use `INSTR_CALL` / `INSTR_RETURN` |
| `Halt` | (core becomes idle) | |

**Work items:**

1. Add a `TraceExporter` that converts `EventLog` to Perfetto JSON
2. Expand `EventType` to distinguish instruction classes (vector, load, store,
   stream, cascade) to match hardware granularity
3. Add configurable event selection (8 events per tile, mirroring hardware)
4. Add DMA direction and channel to DMA events
5. Add port activity tracking (RUNNING/IDLE/STALLED per port)
6. Emit per-tile process metadata matching mlir-aie conventions

### Phase 2: Hardware Trace Collection

**Goal:** Collect real hardware traces for comparison. This uses the existing
mlir-aie trace infrastructure -- no new tools needed.

**Workflow:**

```bash
# 1. Add trace configuration to the kernel's MLIR
#    (configure_packet_tracing_aie2 in the runtime sequence)

# 2. Compile and run on hardware
npu-compile kernel_with_trace.py
npu-run kernel_with_trace.xclbin

# 3. Parse trace buffer to Perfetto JSON
python mlir-aie/python/utils/trace/parse.py \
    --input trace.txt \
    --mlir kernel_with_trace.mlir \
    --output hw_trace.json

# 4. Also run on emulator
cargo run -- kernel_with_trace.xclbin --trace emu_trace.json
```

**Work items:**

1. Script to inject trace configuration into existing test kernels
2. Validate that our existing test xclbins can have tracing added
3. Collect reference traces for the 9 currently-passing tests
4. Document the exact trace event selection used (for reproducibility)

### Phase 3: Trace Comparison

**Goal:** Automated diffing of emulator vs. hardware traces with meaningful
error reporting.

**Comparison levels (increasing strictness):**

1. **Event ordering:** Do the same events happen in the same order?
   (Ignoring exact cycle counts -- just the sequence.)

2. **Relative timing:** Are event durations proportional? (e.g., if hardware
   shows a lock stall of N cycles, does the emulator show a similar stall?)

3. **Cycle-accurate timing:** Do events happen at the exact same cycle?
   (This is the ultimate goal but may need per-event tolerance bands.)

4. **Output data:** Does the kernel produce identical output buffers?
   (We already check this in the test harness.)

**Comparison algorithm:**

```
For each tile in both traces:
    Align by start event (timer reset broadcast)
    Extract event sequence: [(event_name, begin_cycle, end_cycle), ...]
    Compare sequences:
        - Missing events (in one trace but not the other)
        - Reordered events (same events, different order)
        - Timing deltas (same events, same order, different cycles)
    Report per-tile summary:
        - MATCH: identical within tolerance
        - TIMING_DRIFT: same sequence, cycle differences
        - ORDER_MISMATCH: events in different order
        - MISSING_EVENT: events present in only one trace
```

**Work items:**

1. Trace alignment (match start events between emulator and hardware)
2. Event sequence extraction and normalization
3. Diff algorithm with configurable tolerance
4. Report generator (summary + detailed per-tile diffs)
5. Integration into test harness (`--compare-trace hw_trace.json`)

### Phase 4: Logic Fuzzer

**Goal:** Generate random valid AIE2 kernels that exercise diverse behavior,
run them on both emulator and hardware, flag differences.

**What "valid kernel" means:**
- Compiles successfully with Peano
- Has well-defined inputs and expected outputs
- Uses a subset of AIE2 features (expandable over time)
- Terminates deterministically

**Kernel generation strategy:**

```
Template:
    1. Allocate N input buffers, M output buffers (random sizes)
    2. Configure DMA: input buffers -> tile memory via objFifos
    3. Generate compute kernel:
        a. Load from input buffers
        b. Apply random but deterministic operations:
           - Vector add/sub/mul with constants
           - Scalar arithmetic
           - Element-wise transforms
           - Shuffle/permute
        c. Store to output buffers
    4. Configure DMA: tile memory -> output buffers
    5. Synchronize (locks/tokens)
```

**Fuzzer dimensions:**

| Dimension | Range | Purpose |
|-----------|-------|---------|
| Buffer count | 1-4 | Test DMA channel scheduling |
| Buffer size | 256B-64KB | Test DMA addressing modes |
| Element type | int8, int16, int32, bf16 | Test type handling |
| Operations | add, sub, mul, shift, logical | Test ALU paths |
| Tile count | 1-4 | Test multi-core coordination |
| DMA pattern | linear, 2D, 3D | Test addressing generators |
| Double buffering | yes/no | Test lock ping-pong |
| Vector width | 256-bit | Test vector pipeline |

**Shrinking:** When a difference is found, the fuzzer reduces the kernel to
the smallest reproducer -- fewer operations, smaller buffers, single tile --
while preserving the difference. This gives us minimal test cases for debugging.

**Work items:**

1. Kernel template generator (Python, outputs MLIR or C++ for Peano)
2. Random operation sequence generator with seed for reproducibility
3. Expected output calculator (golden reference, run on host CPU)
4. Harness: compile -> run on emulator -> run on hardware -> diff traces
5. Shrinking algorithm for failing cases
6. CI integration (nightly fuzzing runs)

---

## Architecture Decisions

### Why Perfetto JSON (not a custom format)?

1. **Already the standard.** mlir-aie's trace parser outputs it. No conversion
   needed for hardware traces.
2. **Visualization is free.** Drag-and-drop to https://ui.perfetto.dev/ for
   interactive timeline exploration.
3. **Diffing is straightforward.** JSON arrays of events are easy to parse,
   align, and compare programmatically.
4. **Community familiarity.** Anyone who has used Chrome DevTools or Android
   systrace knows this format.

### Why logic fuzzing (not random instruction fuzzing)?

Random instruction streams are mostly invalid on AIE2 (wrong VLIW slot
packing, illegal operand combinations, missing synchronization). The kernel
would crash or hang rather than produce comparable traces.

Logic fuzzing generates **semantically valid programs** that the compiler
can optimize and schedule. The randomness is in the computation graph, not
the instruction encoding. This means:

- Every generated kernel compiles and runs to completion
- Outputs are deterministically computable from inputs
- The compiler's scheduler exercises real pipeline timing
- DMA/lock/routing interactions are exercised naturally

### Why start with event ordering (not cycle accuracy)?

Cycle-accurate comparison requires modeling NoC latency, memory bank conflicts,
DMA pipeline depth, and other micro-architectural details we haven't fully
validated. Starting with event ordering catches **functional bugs** (wrong
lock sequencing, missing DMA transfers, incorrect data routing) which are
higher-severity than timing bugs.

Once event ordering matches, we progressively tighten to relative timing,
then cycle-accurate. Each level catches a different class of bugs.

---

## Success Criteria

| Milestone | Criteria |
|-----------|----------|
| Phase 1 complete | Emulator emits Perfetto JSON; viewable in Perfetto; covers all current EventTypes |
| Phase 2 complete | 5+ test kernels have matching hardware traces collected |
| Phase 3 complete | Automated diff tool runs in CI; reports pass/fail per test |
| Phase 4 complete | Fuzzer generates 100+ kernels/hour; all pass event-ordering comparison |
| Validation target | Zero event-ordering mismatches on 10,000 fuzzed kernels |

---

## References

- [Perfetto Trace Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview) -- Chrome Trace Event Format specification
- `mlir-aie/python/utils/trace/__init__.py` -- mlir-aie trace documentation
- `mlir-aie/python/utils/trace/setup.py` -- Trace configuration reference (52KB)
- `mlir-aie/python/utils/trace/parse.py` -- Trace parser reference (31KB)
- `mlir-aie/python/utils/trace/events/aie2.py` -- AIE2 event code definitions
- `mlir-aie/test/parse-trace/` -- Example traces and golden outputs
- AMD AM020 Ch4 -- AIE-ML trace architecture
- AMD AM025 -- Trace control register definitions
