# Plan: Trace Unification Across Emulator, Simulator, and Hardware

Status: Phase 1 Complete (2026-02-17)

## The Vision

Load traces from three different sources into one viewer, side by side,
for the same test binary:

```
Real NPU hardware  -->  Perfetto trace  -->  |
aiesimulator       -->  Perfetto trace  -->  |  Perfetto UI
xdna-emu           -->  Perfetto trace  -->  |  (unified view)
```

This gives us differential debugging: see where the emulator diverges
from real silicon, at the event level, visually.

## Why This Matters

- **Correctness**: When the emulator produces wrong output, we can see
  exactly which DMA transfer, lock acquisition, or instruction sequence
  diverged from hardware.
- **Performance**: Cycle-level traces show where the emulator's timing
  model differs from real silicon (stall patterns, pipeline behavior).
- **Debugging**: Instead of printf-debugging differences, we get a
  visual timeline showing all three sources aligned by event type.

## What Already Exists

### Our Emulator (src/trace.rs)

Already exports Perfetto JSON. Current event coverage:

- Per-tile instruction events (vector, load, store, call, return)
- Lock acquire/release with stall durations
- DMA lifecycle (start task, BD complete, task complete)
- DMA stall reasons (lock contention, stream starvation)
- Stream get/put operations
- Branch events
- Core state transitions (active, disabled, stalled)

Format: Chrome Trace Event Format JSON, viewable at ui.perfetto.dev.
Each tile gets a unique PID, events are grouped by category into threads.

### Real NPU Hardware Traces (mlir-aie)

**Trace capture**: aie-rt provides `XAie_Trace*()` functions to configure
hardware trace units. Traces are collected into DDR buffers as packets.

**Trace parsing**: `mlir-aie/python/utils/trace/parse.py` converts raw
trace packets into Perfetto JSON. Understands all four packet types
(Core, Memory, ShimTile, MemTile) and maps them to named events.

**Event enums**: Auto-generated from aie-rt headers in
`mlir-aie/install/python/aie/utils/trace/events/aie2.py`. Four classes:
CoreEvent (128 events), MemEvent (128), ShimTileEvent (128),
MemTileEvent (160+). Each event has a numeric code and uppercase name.

**Trace modes** (from aie-rt):
- `EVENT_TIME` -- event timestamps
- `EVENT_PC` -- program counter at each event
- `INST_EXEC` -- instruction-level execution trace

### aiesimulator

**VCD output**: Full wire-level Value Change Dump files. SystemC 1.0
format, picosecond timescale. Contains every signal transition in every
tile -- complete but very low-level. Files can be 16MB+ for simple tests.

**Existing VCD files**: Already produced by our unit test runs at
`build/unit_tests/*/--simulation-cycle-timeout.vcd`.

**eventanalyze** (aietools): BLOCKED -- requires `compiler_report.json`
from full Vitis flow; segfaults on empty/missing JSON. Our aiecc.py
builds do not produce this file. See Phase 1 findings below.

### xdna-driver Kernel Traces

Linux tracepoints for host-side events:
- `amdxdna_debug_point` -- generic debug events
- `xdna_job` -- job lifecycle (fence, seqno, operation ID)
- `mbox_set_tail/head` -- mailbox ring buffer management
- `npu_perf_trace.sh` -- captures via `perf record`
- 26 event types across 9 categories (scheduling, PDI load, preemption,
  power management, L2 cache, errors)

These are useful for understanding the host-device interaction but not
needed for the initial trace unification (tile-level is the priority).

---

## Phase 1 Findings: Format Compatibility (COMPLETE)

Investigation date: 2026-02-17. Examined all four sources in detail.

### Format Comparison Matrix

| Aspect | Our Emulator (`trace.rs`) | mlir-aie (`parse.py`) | aiesimulator VCD |
|--------|--------------------------|----------------------|------------------|
| Format | Chrome Trace JSON array | Chrome Trace JSON array | IEEE 1364 VCD |
| Timestamp unit | Cycles (as microseconds) | Cycles | Picoseconds |
| Event phases | B, E, M | B, E, M | Level signals (1/0) |
| PID scheme | Sequential per (type, col, row) | Sequential per (type, location) | N/A (signal hierarchy) |
| TID scheme | By event category (0-7) | By trace slot index (0-7) | N/A |

### Event Name Alignment

Core instruction events match EXACTLY between our emulator and mlir-aie
(both derived from aie-rt `CoreEvent` enum):

| Our Emulator | mlir-aie CoreEvent | Code | Match |
|---|---|---|---|
| INSTR_VECTOR | INSTR_VECTOR | 37 | Exact |
| INSTR_LOAD | INSTR_LOAD | 38 | Exact |
| INSTR_STORE | INSTR_STORE | 39 | Exact |
| INSTR_CALL | INSTR_CALL | 35 | Exact |
| INSTR_RETURN | INSTR_RETURN | 36 | Exact |
| INSTR_STREAM_GET | INSTR_STREAM_GET | 40 | Exact |
| INSTR_STREAM_PUT | INSTR_STREAM_PUT | 41 | Exact |
| INSTR_LOCK_ACQUIRE_REQ | INSTR_LOCK_ACQUIRE_REQ | 44 | Exact |
| INSTR_LOCK_RELEASE_REQ | INSTR_LOCK_RELEASE_REQ | 45 | Exact |
| MEMORY_STALL | MEMORY_STALL | 23 | Exact |
| LOCK_STALL | LOCK_STALL | 26 | Exact |
| STREAM_STALL | STREAM_STALL | 24 | Exact |

DMA and lock events DIVERGE -- ours are generic, mlir-aie is
channel/selector-specific:

| Our Emulator | mlir-aie MemEvent | Issue |
|---|---|---|
| DMA_START_TASK | DMA_S2MM_0_START_TASK (19) | We lack channel info |
| DMA_FINISHED_BD | DMA_S2MM_0_FINISHED_BD (23) | Same |
| DMA_FINISHED_TASK | DMA_S2MM_0_FINISHED_TASK (27) | Same |
| DMA_STALLED_LOCK | DMA_S2MM_0_STALLED_LOCK (31) | Same |
| DMA_STREAM_STARVATION | DMA_S2MM_0_STREAM_STARVATION (35) | Same |
| LOCK_ACQ | LOCK_SEL0_ACQ_EQ (44) | We lack selector+mode |
| LOCK_REL | LOCK_0_REL (46) | We lack lock ID in name |
| ACTIVE_CORE | ACTIVE (28) | Minor name mismatch |
| DISABLED_CORE | DISABLED (29) | Minor name mismatch |
| BRANCH_TAKEN | *(no equivalent)* | Emulator-only event |

### Process Name Format Difference

- Ours: `core_trace for tile(2,0)` (with parentheses)
- mlir-aie: `core_trace for tile2,0` (no parentheses)

Minor but affects visual alignment in side-by-side view.

### Two Abstraction Levels of Events

1. **Hardware trace events** (mlir-aie IntEnum): 128 codes per module,
   matching what the silicon's trace unit actually emits. Our emulator
   naturally maps here.

2. **Simulation events** (aietools event_type_table.txt): 95+ higher-level
   types that eventanalyze derives from VCD signal analysis. Different
   abstraction, different naming, different tool ecosystem.

### eventanalyze: Dead End

- Requires `compiler_report.json` from full Vitis/aiecompiler flow
- Our aiecc.py/mlir-aie builds do NOT produce this file
- Segfaults on empty JSON placeholder
- Documented output formats: `--text`, `--ctf`, `--csv`, `--wdb`
  (NO Perfetto JSON option)
- Related tools tried: `vcdanalyze` (same crash), `eventanalyze_vcd`
  (same options as `eventanalyze`)
- `aiesimulator --online` passes to vcdanalyze, same metadata requirement

### VCD Structure: Self-Describing Event Signals

The VCD files contain **labeled event_trace signals** that map directly
to hardware event codes. Three tile-type hierarchies:

```
Shim:    tl...math_engine.shim.tile_X_Y.event_trace.event{N}_{name}
MemTile: tl...math_engine.mem_row.tile_X_Y.event_trace.event{N}_{name}
Compute: tl...math_engine.array.tile_X_Y.cm.event_trace.event{N}_{name}
```

The event number in the signal name directly corresponds to the mlir-aie
enum code. Examples from test 08_tile_locks:

```
event28_active          = CoreEvent.ACTIVE (28)
event29_disabled        = CoreEvent.DISABLED (29)
event37_instr_vector    = CoreEvent.INSTR_VECTOR (37)
event44_instr_lock_acquire_req = CoreEvent.INSTR_LOCK_ACQUIRE_REQ (44)
```

These are 1-bit level signals: `b1` = event is active, `b0` = event
ends. VCD only records changes, so transitions map directly to Perfetto
B/E pairs.

### VCD File Statistics (test 08_tile_locks)

| Metric | Value |
|--------|-------|
| File size | 17 MB |
| Total lines | 343,251 |
| Declared signals | 129,345 |
| Timescale | 1 ps |
| Clock period | 952 ps (~1.05 GHz) |
| Simulation duration | ~4.1M ps (~4,300 cycles) |
| Active compute tile | tile_7_3 only |
| Tile types in hierarchy | shim (row 0), mem_row (row 1-2), array (row 3+) |

Additional signals beyond event_trace:
- ISS pipeline: `cm.proc.iss.{pm_rd_in, dme_rda_s_in, ...}` (compute tiles)
- Core status: `cm.proc.core_status.{enable, lock_stall_S, stream_stall_SS0, ...}`
- Stream switch: `stream_switch.{from_sNorth0.data, event_idle_sNorth0, ...}`
- DMA state: `dma.{s2mm_state0.valid_resp, ...}`

---

## Updated Plan

### Phase 2: Minor Emulator Alignment (Easy Wins)

Fix the small differences found in Phase 1:
- Rename `ACTIVE_CORE` -> `ACTIVE`, `DISABLED_CORE` -> `DISABLED`
- Change process name format from `tile(R,C)` to `tileR,C`
- These are trivial `trace.rs` changes

### Phase 3: Custom VCD-to-Perfetto Parser (KEY DELIVERABLE)

**Goal**: Write a Rust module that converts aiesimulator VCD files to
Perfetto JSON, bypassing eventanalyze entirely.

Since eventanalyze is blocked and VCD is self-describing, this is both
necessary and advantageous (zero proprietary tool dependencies, full
control over output format).

**Design**:

1. Parse VCD header to build signal map:
   signal_id -> (tile_col, tile_row, tile_type, event_number, event_name)

2. Scan VCD body for transitions on event_trace signals:
   - `b1 <id>` at timestamp T -> Begin event
   - `b0 <id>` at timestamp T' -> End event

3. Convert timestamps: picoseconds -> cycles (divide by clock period)

4. Emit Perfetto JSON matching our emulator's conventions:
   - Same PID/TID scheme
   - Same event names (from signal name suffix)
   - Same B/E/M phase structure

**Signal name parsing**: Extract tile coords and event info from:
```
tl...math_engine.{shim|mem_row|array}.tile_{col}_{row}.{cm.|}event_trace.event{N}_{name}
```

**Filtering**: Only parse event_trace signals (ignore the other 129K+
signals). This makes the parser fast despite large VCD files.

**Module location**: `src/trace/vcd.rs` (alongside existing `trace.rs`)

**Estimated scope**: ~300-500 lines of Rust, well-tested.

### Phase 4: Enrich Emulator DMA/Lock Events

Thread channel/selector info through emulator events so they match
mlir-aie's channel-specific naming:
- `DMA_START_TASK` -> `DMA_S2MM_0_START_TASK` etc.
- `LOCK_ACQ` -> `LOCK_SEL0_ACQ_EQ` etc.

This requires changes in `src/device/dma.rs`, `src/device/locks.rs`,
and `src/trace.rs`.

### Phase 5: Hardware Trace Capture Pipeline

**Goal**: Make it easy to capture a hardware trace for any test we run.

This requires:
1. Configure trace units before test execution (via CDO or runtime API)
2. Allocate DDR buffer for trace packets
3. Run the test
4. Read trace buffer, parse with mlir-aie's parse.py
5. Output Perfetto JSON alongside test results

Investigate whether our npu-runner tool can be extended to handle trace
capture, or whether mlir-aie already has a test harness for this.

### Phase 6: GUI Integration (Optional)

**Goal**: Load and display traces in our egui-based visual debugger.

Options:
- Embed a trace timeline view in the GUI (significant effort)
- Just auto-open Perfetto UI with the trace file (minimal effort)
- Write a merged trace that combines emulator + hardware (moderate)

The pragmatic choice is probably "just open Perfetto UI" for now, and
add inline trace visualization later when the GUI is more mature.

## Event Mapping Reference

Updated with confirmed mappings from Phase 1 investigation. Event codes
from mlir-aie `aie2.py` enums.

### Core Events (CoreEvent enum, compute tiles)

| Our Event | mlir-aie Name | Code | VCD Signal Suffix |
|-----------|--------------|------|-------------------|
| INSTR_VECTOR | INSTR_VECTOR | 37 | event37_instr_vector |
| INSTR_LOAD | INSTR_LOAD | 38 | event38_instr_load |
| INSTR_STORE | INSTR_STORE | 39 | event39_instr_store |
| INSTR_CALL | INSTR_CALL | 35 | event35_instr_call |
| INSTR_RETURN | INSTR_RETURN | 36 | event36_instr_return |
| INSTR_STREAM_GET | INSTR_STREAM_GET | 40 | event40_instr_stream_get |
| INSTR_STREAM_PUT | INSTR_STREAM_PUT | 41 | event41_instr_stream_put |
| INSTR_LOCK_ACQUIRE_REQ | INSTR_LOCK_ACQUIRE_REQ | 44 | event44_instr_lock_acquire_req |
| INSTR_LOCK_RELEASE_REQ | INSTR_LOCK_RELEASE_REQ | 45 | event45_instr_lock_release_req |
| MEMORY_STALL | MEMORY_STALL | 23 | event23_memory_stall |
| LOCK_STALL | LOCK_STALL | 26 | event26_lock_stall |
| STREAM_STALL | STREAM_STALL | 24 | event24_stream_stall |
| ACTIVE_CORE | ACTIVE | 28 | event28_active |
| DISABLED_CORE | DISABLED | 29 | event29_disabled |
| BRANCH_TAKEN | *(none)* | -- | *(emulator only)* |

### Memory Events (MemEvent/MemTileEvent, memory tiles)

| Our Event | mlir-aie Name | Code | VCD Signal Suffix |
|-----------|--------------|------|-------------------|
| DMA_START_TASK | DMA_S2MM_SEL0_START_TASK | 21 | event21_dma_s2mm_sel0_start_task |
| DMA_FINISHED_BD | DMA_S2MM_SEL0_FINISHED_BD | 25 | event25_dma_s2mm_sel0_finished_bd |
| DMA_FINISHED_TASK | DMA_S2MM_SEL0_FINISHED_TASK | 29 | event29_dma_s2mm_sel0_finished_task |
| DMA_STALLED_LOCK | DMA_S2MM_SEL0_STALLED_LOCK | 33 | event33_dma_s2mm_sel0_stalled_lock_acquire |
| DMA_STREAM_STARVATION | DMA_S2MM_SEL0_STREAM_STARVATION | 37 | event37_dma_s2mm_sel0_stream_starvation |
| LOCK_ACQ | LOCK_SEL0_ACQ_EQ | 46 | (memtile numbering varies) |
| LOCK_REL | LOCK_SEL0_RELEASE | -- | (memtile numbering varies) |

## Dependencies

- **Perfetto UI**: https://ui.perfetto.dev (web-based, no install needed)
- **GTKWave**: For raw VCD inspection (apt install gtkwave)
- **mlir-aie trace parser**: mlir-aie/python/utils/trace/parse.py
- **mlir-aie event enums**: mlir-aie/install/python/aie/utils/trace/events/aie2.py
- **npu-runner**: Our hardware test runner (already built)

Note: eventanalyze is NOT a dependency. We bypass it entirely.

## Answered Questions (from Phase 1)

1. **Do we use the same event name strings?**
   Core events: YES (exact match). DMA/lock events: NO (ours are generic,
   mlir-aie is channel-specific). Fix planned in Phase 4.

2. **Do we use the same PID/TID assignment scheme?**
   Similar but not identical. PID: both use sequential per (type, tile).
   TID: ours groups by event category, mlir-aie uses trace slot index.
   Process name format differs (parens vs no parens). Fix in Phase 2.

3. **Are timestamps on the same scale?**
   Both emulator and mlir-aie use cycles. VCD uses picoseconds (952 ps
   per cycle in simulation). VCD parser will convert ps -> cycles.

4. **Can Perfetto overlay two separate trace files?**
   Still unanswered -- need to test manually. If not, we write a merge
   tool (trivial: concatenate JSON arrays, shift PIDs to avoid collision).

5. **eventanalyze Perfetto output?**
   ANSWERED: eventanalyze does NOT produce Perfetto JSON. Its formats are
   text, CTF, CSV, WDB. This is moot since we're writing our own parser.

## Open Questions

1. Can Perfetto UI merge two separate JSON trace files into one view?
   (If not, we need to write a merge tool.)

2. What's the overhead of hardware trace capture? Does it affect timing?
   (Probably minimal -- traces use dedicated hardware units.)

3. The VCD event_trace signals appear to be level-based (stay high while
   condition is true), not pulse-based. Need to verify this interpretation
   across more test cases and event types.

4. The 952 ps clock period: is this configurable, or fixed for AIE2
   simulation? Need to handle variable clock periods in the VCD parser.
