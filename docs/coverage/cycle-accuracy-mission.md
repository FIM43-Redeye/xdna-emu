# Cycle-accuracy mission

The reason this doc exists: cycle-accuracy work fans out across many
subsystems (instruction costs, broadcast network, trace pipelining,
timer sync, DMA pipeline, NoC arbitration), and we keep getting
pulled into the next-most-visible rabbit hole at the expense of a
coherent picture. This file is the **single index** for every
cycle-accuracy effort. If you're touching anything that affects
when an event/instruction/state-change happens at cycle granularity,
your work belongs in this index.

This is NOT an architecture doc — see [aie2/subsystem-index.md](aie2/subsystem-index.md)
for "what subsystems exist." This doc tracks "where are we on making
each subsystem cycle-accurate."

## Mission statement

xdna-emu's value proposition is binary-compatible, cycle-accurate
emulation of AMD XDNA NPUs. "Cycle-accurate" specifically means:
for any kernel that runs on real silicon, the EMU produces a sequence
of events (instruction retires, DMA completions, lock acquires, trace
frame emissions, etc.) at *the same per-cycle positions* as HW,
within a small bounded skew.

The current state is "structurally accurate, cycle-approximate." We
match what HW does (events fire, in the right order), but we don't
yet match exactly when each one happens at cycle granularity. The
2-PC mode-2 residual on `add_one_using_dma.chess` is a typical
manifestation: the PC sequences agree structurally; they disagree
by 1 cycle at each end of the trace window.

The mission: close that gap, subsystem by subsystem, by deriving
each subsystem's cycle behavior from authoritative sources rather
than guessing.

## Authoritative source hierarchy

Repeated from [CLAUDE.md](../../CLAUDE.md) so cycle-accuracy work
defaults to the right reference:

1. **Open-source toolchain** — aie-rt, llvm-aie, mlir-aie. Apache
   2.0/MIT. Primary reference. If a subsystem's cycle behavior is
   in a TableGen file or aie-rt source, that's the answer.
2. **Real-hardware measurement** — a kernel that runs on the NPU
   and is instrumented to measure the quantity we care about.
   Use this when (1) is silent.
3. **aietools SystemC simulator** — closed-source binaries, but
   parts of the headers and data files are readable (see
   [aietools SystemC angle](#aietools-systemc-angle) below).
   Reading reference for hardware behavior the open toolchain
   doesn't document. Never copy code; read, understand, write
   original.
4. **AM020/AM025 architecture references** — the silicon spec,
   filling in gaps not covered above.

## Open cycle-accuracy items

Status legend: PROPOSED (idea, not started) / IN-PROGRESS / BLOCKED
/ DEFERRED (paused with reason) / FIXED (closed).

### 1. Per-instruction cycle costs (the calibration framework)

**Status**: IN-PROGRESS. Phases 1-3 landed in #322/#323. Phase 4+
ongoing.

**What**: Each VLIW instruction variant has a per-cycle cost
(retire latency, hazard implications). The cost framework in
`src/npu/cycle_cost.rs` and the calibration sweeps in
`tools/calibration/` measure these against real HW.

**Why it matters**: this is the dominant source of end-to-end
cycle drift. If a kernel's body is even 0.1% off per
instruction, end-to-end timing diverges by enough to shift
event-arrival cycles into different alignment buckets — which
is what makes the trace start/stop pipelined window unobservable
on `add_one_using_dma`.

**Cross-references**:
- Tasks: #322, #323, #336-#339, #342-#344
- Findings: `docs/archive/findings/2026-05-04-control-path-cycle-calibration.md`
- Code: `src/npu/cycle_cost.rs`, `tools/calibration/`

### 2. Multi-tile timer sync (broadcast-driven reset)

**Status**: FIXED 2026-05-04 (#318 family).

**What**: `XAie_SyncTimer` aligns all per-tile timers to cycle 0
in the same hardware cycle by exploiting a broadcast event. The
EMU now consumes `Timer_Control.Reset_Event` via a `pending_reset`
latch on `TileTimer`.

**Cross-references**:
- Detail: [timer-sync-gap.md](timer-sync-gap.md)
- Code: `src/device/timer.rs`, `src/device/tile/mod.rs`

### 3. Trace controller pipelined start/stop

**Status**: FIXED in principle (model + unit tests landed
2026-05-04), but not empirically observable on the only mode-2
test we have. Stays in tree because the model is internally
consistent and may matter for other kernels.

**What**: HW pipelines the Idle→Running and Running→Stopped
transitions on the trace controller by 1 cycle. Same-cycle
events as the start/stop trigger are dropped.

**Cross-references**:
- Detail: [trace-start-stop-latency-gap.md](trace-start-stop-latency-gap.md)
- Code: `src/device/trace_unit/mod.rs`

### 4. Broadcast event propagation latency

**Status**: PROPOSED. Not started.

**What**: Each tile-to-tile hop in the broadcast event network
has some non-zero cycle cost in real HW (almost certainly 1
cycle per hop, but unconfirmed). The EMU currently models 0
cycles — events teleport from source to destination tiles
within the same cycle.

**Why it matters**: contributes to the 2-PC gap on
`add_one_using_dma.chess` mode-2: HW's start_event arrives
at the trace tile a couple of cycles after the perf-counter
overflow, putting the start-cycle within the kernel's branch
retire window. EMU's 0-cycle model means the start lands
much earlier in kernel-relative time, missing the window.

**How to derive the value**:

a. **HW measurement (preferred per source hierarchy 2)**: build
   a calibration kernel with two perf counters — one in the
   broadcast source tile counting cycles, one in a destination
   tile counting broadcast event arrivals. Sweep across
   tile-to-tile distances. Per-hop cycle cost falls out of the
   delta. This fits the existing calibration sweep harness in
   `tools/calibration/`.

b. **aietools SystemC inspection (faster, read-only)**: the
   `EventBroadcast` SystemC module is in
   `amd-unified-software/aietools/lib/lnx64.o/libaie2_cluster_msm_v1_0_0_dbg.osci.so`.
   Closed binary, but symbol names suggest it has thread/method
   handles for register IO and one for propagation, with
   internal state. Some neighbouring SystemC sources are
   plain text — see [aietools SystemC angle](#aietools-systemc-angle)
   below.

c. **AM020/AM025 reading**: AM020 ch.2 line 351-353 describes
   the broadcast OR-tree but does not give cycle counts.
   Likely silent on this. Confirm before assuming.

**Open question**: do the four directions have symmetric latency,
or is the AIE-ML→memory module path different (per AM020 ch.2
TIP about east-broadcast being internally connected)?

### 5. NoC / AXI / DMA pipeline timings

**Status**: DEFERRED.

**What**: NoC arbitration, DMA request pipeline, AXI
transaction latency. Currently fudged.

**Cross-references**:
- Findings: `docs/archive/findings/2026-05-04-control-path-cycle-calibration.md`
- Tasks: #335 (constants extracted), #337 (default latencies
  disassembled from libaie2_cluster_msm)

### 6. Peano mode-2 trace BO empty (misdiagnosis trail)

**Status**: NOT A BUG IN OUR CYCLE-ACCURACY MODEL. Investigated
2026-05-04, re-evaluated 2026-05-05 — the original "Peano-empty-
trace" framing was wrong on two counts. Stays in this index as
a breadcrumb so future-us doesn't re-derive the same wrong path.

**What we thought was happening**: Peano-compiled kernels were
producing empty mode-2 trace BOs on HW where Chess-compiled
equivalents weren't, suggesting either a fixed-length trace
window issue or a missing `gen_trace_done_aie2`-style sync per
upstream [mlir-aie #2001](https://github.com/Xilinx/mlir-aie/issues/2001)
/ [PR #2058](https://github.com/Xilinx/mlir-aie/pull/2058).

**What actually happened**:

1. Our `tools/mlir-trace-inject.py` already emits the post-#2058
   pattern. The declarative `aie.trace.host_config` /
   `aie.trace.start_config` ops it inserts lower (via
   `AIEXInlineTraceConfig`) into the same packet-routed trace
   flow + `gen_trace_done_aie2`-equivalent event-fire that
   PR #2058 prescribes for runtime sequences. We were not
   missing the upstream fix.

2. The Peano-vs-Chess empirical differential came from a single
   2026-05-01 sweep run that was contaminated by trace-BD reuse
   across batches (the issue tracked + fixed under #311 and the
   2026-05-04 trace-sweep contamination fix). The "tests
   affected" list we recorded was an artifact of which
   compiler-test pair fell into which empty batch, not a real
   compiler-driven HW behavior.

**Separate, real follow-up**: 2026-05-04 single-run bridge
results show ~70 tests with empty HW `trace_raw.bin` (only
`dmabd_task_queue` for both compilers has data). Uniform
across Chess + Peano, so not a #2058 thing. Tracked as its own
investigation, not under this entry.

**Cross-references**:
- Detail (with the full correction trail): [peano-trace-window-gap.md](peano-trace-window-gap.md)
- Task #306 (closed; original conclusion was wrong, see detail doc)
- Task #311 (real source of the contamination)
- Upstream: mlir-aie #2001, PR #2058 (both already in our checkout AND already in our injector's lowering output)

### 7. Bank conflict event-fire (for cycle-accurate event subsystem)

**Status**: PROPOSED.

**What**: We *detect* bank conflicts for stall purposes but
don't *emit* `MEM_CONFLICT_*` events to the event subsystem.
Mode-2 / VCD comparators see the divergence at the event level.

**Cross-references**: the cycle-accuracy gaps tracked in this document

## aietools SystemC angle

The `*.osci.so` libraries in `amd-unified-software/aietools/lib/lnx64.o/`
are closed binaries (we've disassembled them for #337 to extract
control-path defaults), but they are *not* the whole story. The
`amd-unified-software/aietools/data/` tree has 898+ plain-text
files in `aie_ml/` alone — `.h`, `.cpp`, `.txt`, `.json`, `.py`.
Symbolic event names (`me_events.h`), vector op semantics
(`python_model/model/`), and instruction definitions
(`me_native.h`, `me_streams.h`, `me_locks.h`, etc.) are all
readable. Read-only reference per CLAUDE.md's source policy —
read, understand, write original — but read freely.

**TODO for an Explore agent (deferred)**: scan
`amd-unified-software/aietools/data/aie_ml/`, `aie2ps/`, and
`include/` for any text files mentioning broadcast latency,
trace pipeline depth, broadcast network behavior, or per-cycle
constants. Likely names to grep for: `broadcast`, `propagation`,
`pipeline`, `latency`, `_delay`, cycle counts in tables. Also
worth scanning `me_native.h` and friends for any
`set_delay_cycles`/`assume_cycles` macros that document HW
latencies in the kernel-side intrinsics. Report file paths and
exact strings; no interpretation needed.

## How to add an item to this index

When you discover a cycle-accuracy gap (anywhere — bridge test
divergence, code review, doc reading), add an entry under
"Open cycle-accuracy items" with:

- **Status**: one of the legend values.
- **What**: one sentence on what behavior is/isn't modeled.
- **Why it matters**: which symptom, in which test, would
  this close.
- **How to derive the value**: pointer to authoritative source,
  or "needs measurement" with a sketch.
- **Cross-references**: tasks, findings, code.

If the item already has a deep-dive doc in `docs/coverage/` or
a finding under `docs/superpowers/findings/`, link to it rather
than duplicating content here.
