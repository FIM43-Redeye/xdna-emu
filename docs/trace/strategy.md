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

### Phases 1-3: Trace Emission, Collection, and Comparison -- LANDED

The trace pipeline is built. Both EMU and HW emit comparable trace data
through a single decoder, output buffers are diffed, and cycle counts are
reported per (tile, event).

**Pipeline:**

| Stage | Tool / Component |
|------|------------------|
| Inject trace routing into kernel MLIR | `tools/mlir-trace-inject.py` (uses mlir-aie's declarative IRON API) |
| Drive multi-batch event sweep | `bridge-runner/bridge-trace-runner` |
| Decode raw HW trace to events JSON / cycles / Perfetto | `tools/parse-trace.py` (wraps mlir-aie's `parse_trace`) |
| Emit EMU-side trace | `src/device/trace_unit/`, `src/trace/` -- writes the same packet-stream format HW does |
| Compare HW vs EMU events JSON | `src/bin/trace_compare.rs` (Rust binary at `target/release/trace-compare`) |
| End-to-end orchestration | `scripts/emu-bridge-test.sh --trace=sweep` |

The Rust comparator handles event-sequence diffing with configurable
tolerance, anchor alignment via the BROADCAST_15 timer-reset broadcast,
and per-tile summary classification (MATCH / DRIFT / ORDER_MISMATCH /
MISSING_EVENT). Output buffers are diffed by the bridge harness alongside
trace events.

Two trace modes are supported on both EMU and HW:
- **Mode 0 (event-time):** the original event-stream format. Default for
  bridge tests.
- **Mode 2 (execution / INST_EXEC):** branch-only PC trace per AM020
  ch.2. EMU mode-2 encoder + comparator landed under A.2b.

**Open follow-ons:**
- #305: generalize mode-2 baseline collection across all bridge tests
- #321: fix EMU broadcast/trace-stop timing so late-kernel events are captured
- #322: populate per-NPU-instruction cycle costs (currently default 1)
- #323: empirical calibration of those costs against HW timing
- #306: investigate Peano trace BO empty failure (Chess works; Peano lowering issue)

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

| Milestone | Criteria | Status |
|-----------|----------|--------|
| Phase 1 complete | Emulator emits HW-compatible trace; viewable in Perfetto via parse-trace.py | DONE |
| Phase 2 complete | 5+ test kernels have matching hardware traces collected | DONE (trace-sweep covers most bridge tests) |
| Phase 3 complete | Automated diff tool runs in bridge harness; reports pass/fail per test | DONE (`trace_compare.rs`; CI not yet wired) |
| Phase 4 complete | Fuzzer generates 100+ kernels/hour; all pass event-ordering comparison | NOT STARTED |
| Validation target | Zero event-ordering mismatches on 10,000 fuzzed kernels | NOT STARTED |

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
