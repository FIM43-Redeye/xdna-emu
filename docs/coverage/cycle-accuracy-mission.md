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

### 8. Adaptive clock-gate execution consumption (wake-on-event)

**Status**: SHIPPED 2026-05-25 (same-day deferral and re-enablement).
Wake events 1-3 implemented; Wake 4 (cascade) deferred to land
alongside Core-adaptive gating.

**What**: AM025 adaptive clock-gating engages a per-tile-module
gate after `2^abort_period` idle cycles (default 128). Silicon
stops the clocked domain on engagement and resumes it on external
wake events. Without wake-on-event coverage, consuming the gate
to skip execution stalled lock-mediated tests on the
"control-packet enqueue onto a gated tile" pattern.

**Investigation**: surfaced 2026-05-25 via the
`ctrl_packet_reconfig` family (4 variants), `core_dmas` /
`memtile_dmas` / `tile_dmas` `dma_configure_task_lock` and
`blockwrite_using_locks` clusters all wedging in
`AcquiringLock` with `lock_value=0`, all unblocked when adaptive
consumption is removed. Diagnosed with bridge-sweep correlation
against the pre-clock-control baseline at `20260521`. Consumption
was deferred (5cfe9c4) the same day pending the wake-event work
described below.

**Wake events implemented** (see code references for each):

- **Wake 1 -- register write into a gated module's address range.**
  Implemented as `DeviceState::wake_adaptive_for_subsystem` invoked
  from `write_register`, `mask_write_register`, and `dma_write`
  (dispatch.rs). Maps `SubsystemKind::Dma | Lock | DataMemory` to
  `wake_adaptive_dma` (Memory clock bit shared with DMA on
  compute/memtile; shim DMA has its own bit but same call) and
  `SubsystemKind::StreamSwitch` to `wake_adaptive_ss`. ClockControl
  writes are skipped at the dispatcher level since they have their
  own counter-reset logic on column / module ungate transitions.
  This is the wedge-unblocker for `ctrl_packet_reconfig`: the
  control packet write that enqueues a task on an idle gated tile
  wakes that tile's DMA counter before the next step.

- **Wake 2 -- stream beat into a slave port of a gated SS module.**
  Implemented via the existing Phase-5 SS branch: any successful
  slave-push sets `cycle_active = true` on the port (ports.rs);
  Phase 5 OR-folds `cycle_active` across all SS ports on the tile
  and calls `tick_adaptive_ss(active=true)`, resetting the SS
  counter. End-of-cycle wake rather than emit-site wake, which is
  slightly closer to silicon's wake-up latency than instantaneous
  reset would be.

- **Wake 3 -- lock-value change reaching a gated DMA tile.**
  Decomposes into two sub-cases, both already covered without an
  additional explicit wake call: (a) cross-tile lock changes
  arrive as stream-routed control packets that decode into
  register writes to the local Lock_value register -- covered by
  Wake 1; (b) same-tile lock changes happen because a channel is
  in AcquiringLock, which `ChannelFsm::is_active()` includes, so
  `any_channel_has_pending_work()` is true and the Phase-5 tick
  produces `tick_adaptive_dma(active=true)`. The "DMA gated AND
  channel in AcquiringLock" scenario from the original sketch is
  impossible in our FSM model.

- **Wake 4 -- cascade transfer arriving at a gated compute Core.**
  DEFERRED. No Core adaptive counter exists today; this wake
  becomes meaningful only when Core-side adaptive gating lands.
  Tracked in the `clock_control` coverage entry.

**Verification**: full bridge sweep 2026-05-25 (consumption
re-enabled, EMU-only, 80 tests, -j16). All 13 formerly-wedged
tests pass: `ctrl_packet_reconfig` × 4 variants, `core_dmas` /
`memtile_dmas` / `tile_dmas` `dma_configure_task_lock`,
`memtile_dmas` / `tile_dmas` `writebd*`,
`tile_dmas/blockwrite_using_locks`. Two residual failures are
both pre-existing and out of scope: `memtile_dmas/
blockwrite_using_locks (chess)` TIMEOUT (separate `lock_value=0`
wedge that reproduces with adaptive fully off -- next priority),
and `vec_mul_trace_distribute_lateral (chess)` compile fail
(unrelated). Lib tests: 3206 pass.

**Cross-references**:
- Spec: `docs/superpowers/specs/2026-05-24-clock-control-design.md`
- Code: `src/device/clock_control/mod.rs` (`wake_adaptive_dma/_ss`),
  `src/device/state/dispatch.rs` (`wake_adaptive_for_subsystem`),
  `src/device/array/routing.rs` (Phase 5 SS branch, Wake 2 chain),
  `src/device/clock_control/integration_tests.rs` (Wake 1-3 tests)

### 9. Perf-counter-driven trace event emission (LOCK_STALL et al.)

**Status**: PROPOSED. Blocked on HW ground-truth measurement campaign.

**What**: HW's trace controller emits state events like LOCK_STALL,
MEMORY_STALL, STREAM_STALL not as raw level signals but via the
correlation between a perf counter (typically PERF_CTRL0 counting
ACTIVE_CORE with a configurable threshold like 1024 cyc) and the
underlying stall state. On rollover the configured PERF_CNT_N slot
fires AND, in the same cycle, whichever stall is active gets stamped
into its slot. EMU currently treats LOCK_STALL as an edge event
(initial emit on WaitLock entry, via `cycle_accurate.rs:821`) plus
a per-retry emit (`interpreter.rs:728-743`, with
`LOCK_STALL_TRACE_PERIOD = 1`). This over-emits on long-stall tests
(2701 events vs HW's ~44 on `_diag_phase_b_add_one_instrumented`)
and approximately matches on short-stall tests (22 vs 24 on
`add_one_using_dma`).

**Why it matters**: cycle-accuracy comparisons that read the trace
decoder's `ts` field are tripped by event-count differences (since
`ts = soc + 1 + events_before`). Today's stage-decomposition for
`#355a` was misdiagnosed as a stage-5 regression because the
over-emission inflated `ts` by ~2686 cyc. Tighter measurement
discipline (always use `soc`) handles this for tooling, but
ultimately the emission semantics should match HW so the `ts` field
is trustworthy too.

**How to derive the value**:

a. **HW measurement (preferred)**: re-measure LOCK_STALL counts on
   a calibration set (`add_one_using_dma`, `_diag_phase_b_*`,
   `dynamic_object_fifo/*`, a few of the `objectfifo_repeat/*`),
   correlating against each test's perf-counter CDO writes. Use
   `soc` field for the cycle window. The 2026-05-11 "~4400 events
   at PC 832" claim cannot be trusted -- it was made before the
   ts/soc gotcha was understood. Phase C's 44 events with ~1024-cyc
   spacing is more credible.

b. **CDO parser**: extract each tile's PERF_CTRL0 / PERF_CTRL1
   configuration at xclbin-load time, store on the tile's trace
   state, and tick during WaitingLock / WaitingDma / WaitingStream
   (since the core stays ACTIVE during stalls per AM020).

**Open question**: do other state events (MEMORY_STALL, STREAM_STALL,
PORT_STALL) follow the same perf-counter-driven sampling, or are
they edge-only? The 2026-05-12 MEMORY_STALL fix in
`docs/archive/findings/2026-05-11-emu-dma-pipeline-too-fast-misses-stalls.md`
addressed a different bug (S2MM phantom bank claims during stall),
not the emission cadence.

**Cross-references**:
- Detail: [`docs/superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md)
- Currently P3 (measurement-discipline-only); P2 is the real fix
- Code: `src/interpreter/core/interpreter.rs:97-101`,
  `src/interpreter/core/interpreter.rs:728-743`,
  `src/interpreter/execute/cycle_accurate.rs:818-822`

### 10. Shim streaming throughput modeling

**Status**: PROPOSED. Blocked on HW measurement of shim DDR egress
rate as a function of BD size and access pattern.

**What**: EMU models shim DMA cold-start (`shim_ddr_cold_start_cycles
= 1500`, commit `3357b7c`) as a pure pre-data delay, after which
words stream out instantly at the configured `words_per_cycle = 4`
rate. HW behaves differently: cold-start is shorter (~tRCD + tRP +
tRAS, sub-microsecond on modern DDR), but the **streaming itself is
throughput-bound** -- the shim DMA's task duration is dominated by
egress rate from DDR, not by cold-start. On
`_diag_phase_b_add_one_instrumented.chess` this surfaces as a
74%-closed-but-wrong-shape stage 1+2:

| Sub-stage | HW | EMU (SoC) | Gap |
|-----------|---:|---:|---:|
| 1a: shim dispatch -> shim FINISHED_TASK | 1682 | 2054 | -372 |
| 1b: shim FINISHED_TASK -> memtile S2MM done | 1017 | -27 | +1044 |

EMU's memtile S2MM finishes 27 cyc *before* the shim itself reports
done. Real silicon has memtile finishing 1017 cyc *after* shim done
(the propagation/commit tail of continuous streaming). The cold-start
knob cannot fix 1b -- adding cycles to MemoryLatency shifts both
endpoints, leaving the sub-stage delta invariant.

**Why it matters**: this is the dominant residual on the
calibration test that drove Phase A/B/C of cycle-accuracy work
(`_diag_phase_b_add_one_instrumented`). Closing it would bring EMU
within stage-3-noise-floor of HW on that test, completing the
#355a roadmap. Likely contributes proportionally on every test
that hits shim DDR -- which is most of them.

**How to derive the value**:

a. **HW measurement (preferred)**: build a calibration kernel that
   varies (i) BD payload size, (ii) access pattern (linear /
   strided / cross-bank). Instrument with shim and memtile
   FINISHED_BD trace anchors. Measure shim_dispatch -> shim_done
   and shim_dispatch -> memtile_s2mm_done deltas; the slope vs
   BD size gives the per-word egress rate, the intercept gives
   true cold-start. Two-tier model (cold-start opens row, then
   sustained throughput) is the expected shape.

b. **AM020 DDR timing tables**: precharge (tRP), activate (tRCD),
   CAS (tCL), refresh-time-rolled-into-stream contributions.
   These give the structural floor for cold-start; not a substitute
   for HW measurement of the streaming rate.

c. **aietools SystemC reading**: the
   `libaie2_cluster_msm_v1_0_0_dbg.osci.so` model contains a
   shim DMA dispatcher; symbol names may indicate the streaming
   rate model. Probably worth a read pass once we're doing the
   HW measurement, as a cross-check on the rate.

**Open questions**:
- Does memtile S2MM also have a throughput-bound regime, or is
  its rate so fast (SRAM-backed) that it effectively matches the
  shim's egress rate? Phase C's stage 3 (memtile S2MM -> MM2S) at
  HW=13 / EMU=15 cyc suggests memtile internals are already
  well-calibrated.
- How does multi-channel arbitration interact with the rate?
  `add_one_using_dma` only exercises a single shim S2MM ch; a
  multi-channel test would calibrate the arbitration cost.

**Cross-references**:
- Detail: [`docs/superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md`](../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md)
- Phase C: `docs/archive/findings/2026-05-10-phase-c-stage-attribution.md`
- Parent item: #5 (NoC / AXI / DMA pipeline timings, DEFERRED)
- Code: `crates/xdna-archspec/src/model_builder.rs:189`,
  `src/device/dma/engine/stepping.rs` `consume_first_bd_bonus`,
  `crates/xdna-archspec/src/types.rs` `DmaTiming.words_per_cycle`

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
