# A.2: PC-anchored validation via mode-1 trace sweeps + perfcnt cycle clock

**Status:** spec, pre-implementation. Brainstorm cycle complete on 2026-04-25;
ready for `writing-plans` to produce the implementation plan.

## Problem

The Phase E validation pipeline today runs an event-time (mode 0) sweep over
8 trace slots per tile and joins HW vs EMU per event by cycle anchors. Two
fundamental problems:

1. **HW jitter.** Real silicon does not run two consecutive batches with
   bit-identical cycle traces. Cross-batch cycle anchors are noisy, so any
   joining that compares "HW cycle X for batch N to EMU cycle X for batch
   N" is bounded above by HW reproducibility, not by emulator accuracy.

2. **Cycle-span scalar misleads when event sets disagree** (cascade_flows
   finding, 2026-04-25). When HW fires `INSTR_VECTOR + LOCK_STALL` and EMU
   fires `MEMORY_STALL + INSTR_LOCK_*` for the same kernel,
   `max(ts) - min(ts)` measures different things and the resulting drift
   ratio is meaningless.

The validation reframe: PCs are kernel-state quantities and are invariant
across batches and across HW jitter; cycles are timing quantities that
drift. Move the joining anchor from cycles to PCs by switching the sweep
to mode 1 (EVENT_PC), and use perfcnt overflow events as a deterministic
in-batch cycle clock.

## Goal

A validation pipeline where:

1. Every kernel-execution event (INSTR_VECTOR, LOCK_STALL, etc.) recorded
   on a core trace unit carries the PC at which it fired.
2. HW vs EMU comparison is "does the same event fire at the same set of
   PCs?" — a structural correctness signal independent of HW timing
   jitter.
3. Cycles come from a per-tile performance counter overflowing on a
   known period (default 1024 cycles), giving deterministic in-batch
   cycle anchors at known PCs.
4. The sweep covers cores in mode 1 simultaneously with memmod / memtile
   / shim in mode 0 (since the Mode bitfield doesn't exist on those
   modules), so a single per-batch run produces validation data for all
   tile types in parallel.
5. A single mode-2 (INST_EXEC) capture at the end of each test stores
   HW instruction-stream baselines as fixtures for a future A.2b spec
   that will implement EMU mode-2 + instruction-stream comparison.

## Non-Goals

- EMU mode-2 encoder. Out of scope; A.2b owns it.
- Instruction-stream diff (LCS-style). Deferred — both because mode-2
  EMU isn't built and because tolerance design needs its own brainstorm.
- A new mlir-aie dialect op for perfcnt sugar. Pragmatic path uses the
  existing `aie.trace.reg` primitive. Optional follow-up PR adds an
  `aie.trace.perf_counter` op upstream.

## Scope cuts confirmed against authoritative sources

- **Mode 1 is core-only.** `aie_registers_aie2.json` shows the `Mode`
  bitfield in `Trace_Control0` exists for the core module only; memmod,
  memory_tile, and shim variants of that register have only
  `Trace_Start_Event` / `Trace_Stop_Event`. Memmod/memtile/shim continue
  in mode 0; the EMU enforces this at config-time.
- **Perfcnt is universal.** All four module types
  (core / memory / memory_tile / shim) have Performance_Control + at
  least Performance_Counter0 + Performance_Counter0_Event_Value in the
  regdb. Cycle anchoring works on every tile type.
- **Mode-1 PC source is "current core PC" only.** Memory-side events
  (DMA fire, lock acquire/release) propagating into a core's trace flow
  have no PC; they're recorded with `pc = 0` sentinel and excluded from
  PC-set comparisons.

## Architecture

```
[1] Inject     mlir-trace-inject.py
                 --trace-mode event_pc
                 --core-grounding PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1
                 --memmod-grounding PERF_CNT_0
                 --memtile-grounding PERF_CNT_0
                 --shim-grounding PERF_CNT_0
                 --core-sweep <NAMES|"all">
                 --memmod-sweep <NAMES|"all">
                 --memtile-sweep <NAMES|"all">
                 --shim-sweep <NAMES|"all">
                 --perfcnt-period 1024
                                          ↓
                          traced-mode1.mlir
                          (aie.trace + aie.trace.config@perf_<type>_<col>_<row>)
                                          ↓
[2] Compile    aiecc.py
                                          ↓
                          traced-mode1.xclbin (perfcnt + trace baked in)
                                          ↓
[3] Sweep      tools/trace-sweep.py
                 --mode event_pc
                 --core-grounding ... (+ all the same flags for parity)
                                          ↓
                          batch_NN/{hw,emu}/trace_raw.bin     (mode-1 sweep batches)
                          batch_M2/hw/trace_raw.bin           (single mode-2 baseline, HW only)
                                          ↓
[4] Decode     parse-trace.py --decoder ours --trace-mode event_pc | inst_exec
                                          ↓
                          batch_NN/{hw,emu}/trace.events.json
                          batch_M2/hw/trace.events.json
                                          ↓
[5] Compare    trace-compare --sweep <dir> --pc-anchored
                                          ↓
                          Per-test PC-anchored report
                          + perfcnt-anchored cycle bands
                          + mode-2 baseline pass-through (decoded but not compared)
```

**Per-batch invariants:**

- The xclbin bakes in mode 1 (cores) / mode 0 (others), perfcnt config,
  and grounding event slot assignments. The sweep only patches the
  *swept* slots per batch via `trace-patch-events.py`; grounding slots
  stay fixed.
- All tile types advance one batch in lockstep. Total batch count =
  `ceil(max(per-cursor sweep length / per-cursor sweep capacity))`.
  Cursors that exhaust their sweep list emit batches with grounding
  slots only.
- Per-batch cycle anchors come from *that batch's own* perfcnt overflow
  stream — never compared across batches.
- PC sets per event-name are compared *within a batch* on HW vs EMU.
  Sweep-level aggregation is a roll-up, not a re-join.
- HW grounding-event PCs (INSTR_EVENT_0/1) must be batch-invariant; the
  orchestrator self-checks this and flags `unsafe_for_pc_join=true` in
  the manifest if they drift, falling back to per-batch-only joining.

## Components

### Component 1 — MLIR injection (`tools/mlir-trace-inject.py`)

**New CLI surface:**

| Flag | Default | Semantics |
|------|---------|-----------|
| `--trace-mode` | `event_time` | `event_time` (mode 0, back-compat) or `event_pc` (mode 1, cores only) |
| `--core-grounding` | `PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1` | Comma-separated event names for fixed slots |
| `--memmod-grounding` | `PERF_CNT_0` | Same |
| `--memtile-grounding` | `PERF_CNT_0` | Same |
| `--shim-grounding` | `PERF_CNT_0` | Same |
| `--core-sweep-events` | unset (= today's 5 default core events) | Override sweep list. Special value `all` = enumerate from event header |
| `--memmod-sweep-events` | unset (= no memmod injection) | Same |
| `--memtile-sweep-events` | unset (= no memtile injection) | Same |
| `--shim-sweep-events` | unset (= no shim injection) | Same |
| `--perfcnt-period` | `1024` | Cycles between PERF_CNT_0_EVENT fires |

**Per-tile MLIR emission**: for each tile in scope, emit:

1. **`aie.trace @trace_<type>_<col>_<row>(%tile)` block** with:
   - `aie.trace.mode EventPC` (cores when `--trace-mode=event_pc`) or
     `EventTime` (others, or back-compat).
   - `aie.trace.packet packet_type=<type-specific>`.
   - `aie.trace.event @<name>` for each grounding event in fixed slot
     positions, then sweep events in remaining slots.
   - `aie.trace.start broadcast=15` / `aie.trace.stop broadcast=14`
     (existing convention).

2. **`aie.trace.config @perf_<type>_<col>_<row>(%tile)`** if `PERF_CNT_0`
   is in the grounding set:
   - `aie.trace.reg register="Performance_Control0" field="Cnt0_Start_Event" value=ACTIVE`
   - `aie.trace.reg register="Performance_Control1" field="Cnt0_Reset_Event" value=PERF_CNT_0`
   - `aie.trace.reg register="Performance_Counter0_Event_Value" value=<period>`
   - The `AIEXInlineTraceConfig` pass lowers each `trace.reg` into
     `npu.write32` ops in the runtime sequence, so this works without
     any new mlir-aie dialect ops.

3. **`aie.trace.host_config buffer_size = N`** in the runtime sequence
   (existing; bump default to `65536` to accommodate full-event sweeps
   plus mode-2 baselines).

4. **`aie.trace.start_config @<sym>`** for each `aie.trace` decl AND
   each `aie.trace.config` perfcnt block, in the runtime sequence
   prologue.

**Refusal / warning conditions:**

- `--trace-mode event_pc` with non-empty `--memmod-sweep-events`,
  `--memtile-sweep-events`, or `--shim-sweep-events`: **warn and
  continue**. The non-core tile configs land in mode 0 (regdb-forced —
  the Mode field doesn't exist on those Trace_Control0 variants); cores
  go to mode 1; the warning surfaces the asymmetry. This matches the
  hardware reality that DMA, locks, and stream events legitimately
  propagate into a core's trace flow with no PC source, and aborting
  the injection would leave non-core sweep coverage entirely off the
  table.

- File already contains `aie.trace`: existing exit-2 behavior.

### Component 2 — EMU implementation (`src/device/`)

#### 2a. Mode-1 trace encoder for core trace units

`src/device/trace_unit/mod.rs`:

- `mode_supports_pc()` helper, true only for cores.
- `set_mode(...)`: panics in dev / errors in release if `EventPc` is
  set on a non-core trace unit.
- `notify_event(hw_event_id, cycle, pc: Option<u32>)` — extended with
  `pc` parameter:
  - Mode 0 path: encodes `(event_bits, cycle_delta)` as today.
  - Mode 1 path:
    - `pc.unwrap_or(0)` for the encoding. `pc == 0` is the "no anchor"
      sentinel (excluded from PC-set comparisons downstream).
    - Rate-limited warning when `pc` is `None` (signals a memory-side
      event firing into a core's trace flow).
- Mode-1 frame encoding follows `tools/trace_decoder/modes/mode1.py`.
  Byte-equivalence with the in-tree decoder is the gate.

#### 2b. Performance counter implementation

New submodule `src/device/tile/perfcnt.rs`. State per counter:

```rust
pub struct PerfCounter {
    value: u32,
    event_value: u32,
    start_event: u8,
    stop_event: u8,
    reset_event: u8,
    running: bool,
}
```

Register mapping (canonical names from regdb):

| Module | Registers |
|--------|-----------|
| core | `Performance_Control0/1/2`, `Performance_Counter0..3`, `Performance_Counter0..3_Event_Value` |
| memory (memmod) | `Performance_Control0/1`, `Performance_Counter0/1`, `Performance_Counter0/1_Event_Value` |
| memory_tile | `Performance_Control0/1/2`, `Performance_Counter0..3`, `Performance_Counter0..3_Event_Value` |
| shim | `Performance_Ctrl0/1`, `Performance_Counter0/1`, `Performance_Counter0/1_Event_Value` |

Behavior, on each cycle:

1. If `running`, increment `value`.
2. If `value >= event_value`, fire `PERF_CNT<N>_EVENT` (event ID looked
   up from the regdb).
3. If `reset_event` matches the firing event, reset `value` to 0.
   (The "free-run" pattern uses `reset_event = PERF_CNT<N>` so the
   counter resets immediately on its own overflow.)
4. Start/stop events toggle `running`.

The tile-level event dispatch updates perfcnt state *before*
propagating to the trace unit, so the perfcnt-driven event lands in
the same cycle's trace frame.

#### 2c. PC threading from coordinator to trace unit

`src/interpreter/engine/coordinator.rs`. Every event-fire site that is
core-instruction-driven (INSTR_VECTOR, INSTR_LOCK_*, INSTR_EVENT_0/1)
must thread the current PC into `notify_event`. Memory-side fire sites
(DMA, lock, stream) pass `None`.

This is the riskiest piece because it touches the main execution loop.
The plan must enumerate the full event-fire site list and pair each
with a PC-availability decision.

### Component 3 — Sweep orchestration (`tools/trace-sweep.py`)

**New CLI:**

| Flag | Default | Semantics |
|------|---------|-----------|
| `--mode` | `event_time` | `event_time` or `event_pc`. Drives `trace-patch-events.py --mode`. |
| `--core-grounding` etc. | as in component 1 | Mirror of injection-time flags so the sweep knows which slots not to overwrite. |
| `--core-sweep / --memmod-sweep / --memtile-sweep / --shim-sweep` | `all` | Which events to sweep per tile type. |
| `--perfcnt-period` | `1024` | Reserved for sanity-checking against the xclbin's baked-in value. |
| `--with-mode2-baseline` | `true` | Emit one final mode-2 batch per test (HW only). |

**Per-tile sweep cursors:** one per (tile, module-type), each holding
the sweep list filtered against the grounding set, a position index,
and `remaining_slots = 8 - len(grounding_for_this_module)`.

**Per-batch generation:**

1. For batch *N*, each cursor consumes its next `remaining_slots`
   events. Exhausted cursors emit grounding-only.
2. Build a multi-tile patch list: `[(col, row, tile_type, slot_event_ids[]), ...]`.
3. Apply via `trace-patch-events.py --multi-tile <json>` (extending the
   existing tool to take a JSON spec; current tool does one tile per
   subprocess, which is `O(N_tiles)` overhead per batch).
4. Run `bridge-trace-runner --batch-stdin` HW + EMU sides (existing
   batch loop).
5. Decode via `parse-trace.py --decoder ours --trace-mode event_pc` →
   `batch_NN/{hw,emu}/trace.events.json`.

**Mode-2 finishing batch:** after the mode-1 sweep completes:

1. Apply a mode-2 patch via `trace-patch-events.py --mode inst_exec`
   (`--mode 2`). The trace event slots are inert in mode 2 (the trace
   unit ignores them and emits per-cycle E_atom/N_atom records plus
   New_PC branch records instead), so the patch only flips the
   Trace_Control0 Mode bits; slot configs from the mode-1 sweep can
   stay in place untouched.
2. **Perfcnt is not needed in mode 2.** Mode 2 records every cycle as
   an E_atom (executed) or N_atom (stalled) frame, with New_PC frames
   on every taken branch. The cycle clock is intrinsic — perfcnt
   overflow events have no special record type in the mode-2 stream
   and contribute only as ordinary E/N_atom records like any other
   cycle. The Performance_Counter0 config baked into the xclbin
   continues to free-run but is invisible to the mode-2 decoder.
3. Run HW only — EMU mode-2 is out of scope; orchestrator passes
   `--no-emu` for this batch and `bridge-trace-runner` itself rejects
   the EMU side cleanly if invoked anyway.
4. Decode via `parse-trace.py --trace-mode inst_exec` →
   `mode2-baseline/<test>/<tile>.events.json`. Trace bin preserved
   alongside under the same path.
5. The compare layer emits a "mode-2 baseline captured, comparison
   deferred to A.2b" note in the per-test summary.

**Cross-batch consistency self-check:** orchestrator records the HW
grounding-event PC sets per batch. Verifies INSTR_EVENT_0/1 PCs are
batch-invariant before declaring sweep complete. Drift flags
`unsafe_for_pc_join=true` in `sweep-manifest.json`.

### Component 4 — PC-anchored compare logic (`src/trace/compare.rs`)

New entry point: `compare_pc_anchored(...)`.

**Per-batch, per-tile, partition by `pkt_type`:**

- `pkt_type == 0` (core, mode 1) → PC-anchored path.
- `pkt_type ∈ {1,2,3}` (memmod / memtile / shim, mode 0) → existing
  cycle-anchored path, **augmented** with perfcnt-anchor cycle bands.

**PC-anchored path:**

1. Drop events with `ts == 0` (no-PC sentinel) into an "unanchored"
   bucket; report counts but exclude from set/multiset diff.
2. For each event-name `X` in the slot config:
   - `pc_set_hw = { e.ts for e in hw_events if e.name == X }`
   - `pc_set_emu = { e.ts for e in emu_events if e.name == X }`
   - **Set-diff (headline):** `hw_only = pc_set_hw - pc_set_emu`,
     `emu_only = pc_set_emu - pc_set_hw`. Pass = both empty.
   - **Multiset-diff (body):** `pc_count_hw[pc] = count`, same for emu.
     Per-PC delta. Threshold-flagged divergences (default threshold:
     any non-zero delta is reported; configurable).

**Cycle band derivation (perfcnt overflow stream):**

1. Filter to events with `name == "PERF_CNT_0_EVENT"`. Each carries
   `(slot=anchor, ts=PC_at_overflow)`.
2. Sequence `[PC_overflow_0, PC_overflow_1, ...]` where overflow `n`
   occurred at exactly `period * n` cycles after kernel start
   (deterministic by perfcnt construction).
3. For any other event firing at PC `p`, find bracketing perfcnt
   overflows and compute `cycle_est = period * (n_below + (p - PC_below) / max(1, PC_above - PC_below))`.
4. HW vs EMU per-(event, PC): compute `delta_cycles = abs(emu_est - hw_est)`.
   Tolerance: `period / 2` for HW jitter; flag values exceeding this.

**Output structure:**

```rust
pub struct PCAnchoredReport {
    pub pkt_type: u32,
    pub set_diff: HashMap<EventName, (HashSet<Pc>, HashSet<Pc>)>,
    pub multiset_diff: HashMap<EventName, HashMap<Pc, (u32, u32, i32)>>,
    pub cycle_bands: HashMap<EventName, HashMap<Pc, CycleBand>>,
    pub unanchored_count_hw: usize,
    pub unanchored_count_emu: usize,
}
```

`BatchResult` extended with `pc_anchored: Option<PCAnchoredReport>` and
`mode: TraceMode` per tile so the dispatch is explicit.

### Component 5 — Sweep aggregation

`compare_sweep_dir_with_opts(...)` extension. Per-test summary:

- **Coverage matrix:** `event_name → batch → "swept" | "absent" | "grounding"`.
- **Per-event divergence summary:** total `set_diff` size + multiset
  magnitude across all batches the event appeared in. Sorted descending.
- **Cycle-band summary:** average `delta_cycles` per event across all
  (PC, batch) pairs. Detects systemic skew (EMU consistently faster /
  slower) vs per-event anomalies.
- **Self-check report:** any batches with `unsafe_for_pc_join=true`
  surface as warnings.
- **Mode-2 baseline note:** lists `<test>/<tile>` mode-2 captures
  available for downstream A.2b consumption.

`format_report` adds three new sections: "PC-anchored coverage",
"PC-anchored divergences", "Perfcnt-anchored cycle deltas".

## Error handling

| Condition | Behavior |
|-----------|----------|
| Injection emits mode=1 with non-core sweep events | warn (does not abort). Non-core configs land in mode 0; mode-1 PC semantics apply only to cores. |
| Trace_Control0 mode=1 write hits a non-core trace_unit on EMU | error (panic in dev, log+skip in release). regdb-derived. |
| Mode-1 event fires with `pc = None` | rate-limited warning, encode `pc=0` sentinel. Excluded from PC-set diffs. |
| `parse-trace.py` mode-1 decode finds every `ts == 0` | warn (suspicious — every event unanchored); proceed. |
| Sweep batch fails (TDR, EMU panic) | record failure marker in manifest with diagnostic; continue to next batch. |
| Cross-batch grounding-PC drift | manifest flags `unsafe_for_pc_join=true`; comparison emits warning and falls back to per-batch-only joining. |
| `compare_pc_anchored` finds zero events in a batch | report `BatchResult { status: "empty_trace" }` rather than failing. |
| EMU asked to run mode-2 batch | `bridge-trace-runner` returns clean error; sweep orchestrator skips EMU automatically (mode-2 is HW-only). |
| Mode-2 decode fails on a baseline | log warning, omit from manifest mode-2 list; rest of sweep unaffected. |

## Testing strategy

**Unit (`cargo test --lib`):**

1. `trace_unit::mode1_encoder` — golden bytes for synthetic event
   sequences match `tools/trace_decoder/fixtures/mode1_*`.
2. `trace_unit::mode1_non_core_panics` — config-time enforcement.
3. `trace_unit::mode1_no_pc_sentinel` — encodes `pc=0` and warns when
   called with `None`.
4. `perfcnt::overflow_fires_event` — counter reaches threshold, fires
   `PERF_CNT0_EVENT`, optionally self-resets when configured to.
5. `perfcnt::start_stop_reset` — start/stop event toggles `running`,
   reset event zeros `value`.
6. `perfcnt::all_module_types` — each of core/memmod/memtile/shim
   instantiates its perfcnt count correctly per the regdb (4 / 2 / 4 / 2).
7. `compare::pc_anchored_set_diff` — golden HW/EMU JSON pairs produce
   expected set diffs.
8. `compare::pc_anchored_multiset_diff` — counts and deltas correct.
9. `compare::cycle_band_interp` — linear interpolation between perfcnt
   anchors hits expected values; tolerance flagging works.
10. `compare::pc_anchored_unanchored_excluded` — `ts=0` events
    contribute to unanchored count, not to PC-set diff.

**Tool tests (Python, mirrors existing patterns):**

11. `tools/test_trace_inject.py` — extend with mode-1 + perfcnt config
    cases; injected MLIR round-trips through `Module.parse()`.
12. `tools/test_trace_sweep.py` — multi-tile lockstep batch generator,
    cursor exhaustion handling, grounding-PC consistency check, mode-2
    finishing batch.
13. `tools/test_trace_decoder.py` (existing) — already covers mode-1
    decode via fixtures; no new cases needed unless EMU-emitted bytes
    expose decoder edge cases.

**Integration (bridge tests):**

14. End-to-end mode-1 sweep on `add_one`. Manifest: every core event
    set-diff-empty or with explainable small divergence. Used as the
    "pipeline lights up" gate before full bridge suite re-runs.
15. Comparison against existing mode-0 sweep on the same test —
    perfcnt-anchored cycle deltas should match the old
    `EMU_SECONDS_PER_CYCLE`-derived cycle-diff in ballpark.
16. Mode-2 baseline capture on `add_one` — HW-only run produces non-
    empty `mode2-baseline/add_one/0_2.events.json`.

## Validation gate

A.2 is "done" when:

- (a) All unit + tool tests in this spec pass.
- (b) Integration sweep on `add_one` (or equivalent small test)
  produces a manifest with `unsafe_for_pc_join=false`.
- (c) `cargo test --lib` baseline holds at 2755+ / 5 ignored
  throughout.
- (d) Mode-2 baseline captured for at least one test, confirming the
  hook works end-to-end.
- (e) The full bridge test suite has not regressed.

## Implementation order

This is a sketch; the writing-plans phase will produce the executable
plan with concrete commits.

1. **EMU perfcnt registers** for all four module types.
   Self-contained; tests via direct register-write fixtures.
2. **EMU mode-1 encoder** (cores only) + sentinel handling.
   Tests via byte-equivalence against in-tree decoder fixtures.
3. **PC threading from coordinator to trace unit.**
   Highest-risk implementation. Per-event-fire-site audit.
4. **`tools/mlir-trace-inject.py`** extension: per-module-type
   grounding/sweep flags, perfcnt config emission, mode selector.
5. **`trace-patch-events.py --multi-tile`** extension.
6. **`tools/trace-sweep.py`** extension: per-tile cursors,
   lockstep batching, mode-2 finishing batch, grounding-PC
   consistency check.
7. **`src/trace/compare.rs`** extension: `compare_pc_anchored`,
   cycle-band interpolation, multiset-diff.
8. **`compare_sweep_dir_with_opts` + `format_report`**: aggregation,
   coverage matrix, per-event summary, mode-2 note.
9. **Integration test gate** on `add_one` or equivalent.

Each numbered step is a discrete commit; steps 1-3 land before any
tool-side work, since all tooling consumes EMU output.

## Why deferred until 2026-04-25

Original Phase 1 spec marked A.2 as the largest validation-strategy
reframe. Prerequisites done in the 2026-04-25 hygiene pass and earlier
threads:

- In-tree decoder authoritative (B thread, `d026e40`).
- Cycle accumulator confirmed deterministic, EMU_SECONDS_PER_CYCLE
  retired (A.5, `daed9a7`).
- cascade_flows finding correctly framed (A.1 + correction in
  `07128a2`).
- Subsystem 8 audit confirmed no parser-side blockers.

A.2 is a Subsystem-9-shaped piece: encoder + perfcnt + coordinator
threading + tool surface + compare logic. Folding it into a hygiene
pass would either cut corners or balloon scope. Picking it up here with
its own brainstorm + plan + execute cycle is per the original "each
hygiene item warrants its own brainstorm cycle" principle.
