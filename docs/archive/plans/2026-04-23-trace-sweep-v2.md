# Trace sweep v2 — grounded multi-tile event matrix

**Date:** 2026-04-23
**Status:** v1 landed, v2 specified, ready to execute
**Owner:** this session (resume-able after compaction)

## Context

This work underpins the emulator refactor validation story. The trace-sweep
infrastructure exists so we can, after any non-trivial emulator change,
run a single command that produces a per-(test, tile, event) matrix
comparing HW vs EMU. A regression shows up as a cell flipping from
`MATCH` to `DRIFT(Hx/Ey)` or `EMU-MISS`. Without this, "did we break
anything?" is answered by running a handful of hand-picked tests with a
hand-picked event set, and silently missing regressions on any event we
didn't think to check.

The goal of v2 is to make the sweep *complete* — all 577 events across
all 4 tile types — and *mergeable*, so traces from separate batch runs
can be stitched into one unified timeline per tile via a grounding
event.

## What v1 landed (completed)

Three commits on branch `dev`, pushed as of this document:

- `0c0bb04` `feat(trace-sweep): compile-once / patch-many event sweep infrastructure`
- `0aee320` `fix(trace-sweep): column-agnostic event matching + handle both empty-trace shapes`
- `f1fb71b` `feat(bridge-test): NO_CORE classification for DMA-only passthrough tests`

Tools:
- `tools/trace-patch-events.py` — patches `TRACE_EVENT0`/`TRACE_EVENT1`
  for a single tile in an existing `insts.bin`. Handles all 4 tile types
  (core/memmod/memtile/shim). Opcode-aware instruction walking mirrors
  `src/npu/parser.rs`; keep them in sync.
- `tools/trace-sweep.py` — orchestrator. For each batch of 8 events:
  patch → `bridge-trace-runner` (HW) → `bridge-trace-runner` (EMU) →
  `parse-trace.py` → relabel slot names → record. Treats empty traces
  as successful-zero, not failure. Filters HW events by `(row, pkt_type)`
  only because HW uses absolute columns while EMU uses relative.

Event enumeration sourced from the build-tree
`mlir-aie/build/include/xaienginecdo_static/xaiengine/xaie_events_aieml.h`:
- core: 127 events
- memmod: 161 events
- memtile: 161 events
- shim: 128 events
- **total: 577**

Demonstrated on `vector_scalar_using_dma.chess`, 16 core events, HW+EMU,
44s. Correctly attributed the MATCH→DRIFT regression to specific events
(LOCK_STALL 17/1, INSTR_LOCK_ACQUIRE_REQ 8/2, INSTR_LOCK_RELEASE_REQ
8/4, INSTR_EVENT_0/1 4/2 — all EMU undercount, all feed into #138).

Register-write audit finding (pre-build): mlir-aie's
`AIEInlineTraceConfigPass` emits trace event-selection as
`aiex.npu.write32` into `runtime_sequence`, which becomes Write32 ops in
`insts.bin`. Byte-patching the 4 value bytes at offset +16 of each
matching record is sufficient — no CDO regeneration needed. See
`lib/Dialect/AIEX/Transforms/AIEInlineTraceConfig.cpp:110`.

## v2 scope

Four deliverables. Order them by dependency: 1→2→3→4.

### 1. Grounding event + cross-batch merge

**Goal:** turn N batch runs into one unified trace with all events on
one timeline per tile. This is the payoff of all the lateral-routing
setup — we get "every tile traced for every event" without ever running
more than 8 events at a time.

**Approach:**

- Dedicate one of the 8 slots in every batch to a *grounding event*
  that fires at a deterministic cycle. Candidates, in preferred order:
  1. `USER_EVENT_1` raised from the runtime_sequence via
     `aiex.npu.write32` to `Event_Generate`. Fires at a single known
     cycle (start of trace) on every run.
  2. `INSTR_EVENT_0` if the kernel raises it at a consistent point
     (some do, some don't — not reliable across tests).
  3. Broadcast-start event (already wired up by AIEInsertTraceFlows).
- The grounding event occupies slot 0 of every batch. The other 7 slots
  carry sweep events.
- At merge time: for each batch, find the grounding event's trace
  timestamp. That becomes the anchor. Subtract it from every other
  event's timestamp in that batch. All batches are now on a common
  "cycles since anchor" axis.
- Concatenate the relabeled events across batches. Dedup only if the
  grounding event itself gets duplicated (it should appear once per
  batch after anchoring).

**Output:** per-tile merged event stream, ordered by cycles-since-anchor.
Schema:
```
{
  "tile": {"col": C, "row": R, "type": "core"},
  "grounding_event": "USER_EVENT_1",
  "batches_merged": N,
  "events": [
    {"name": "INSTR_VECTOR",   "ts_anchored": 1183, "source_batch": 0},
    {"name": "LOCK_STALL",     "ts_anchored": 1197, "source_batch": 2},
    ...
  ]
}
```

**Implementation notes:**
- `trace-patch-events.py` already handles any event ID in any slot;
  no changes needed to the patcher.
- `trace-sweep.py` needs: `--ground-event NAME` flag that (a) forces the
  event into slot 0 of every batch, (b) drops it from the per-event
  result rows (it's instrumentation, not a swept event), (c) writes a
  merged-events.json alongside the matrix JSON.
- Caveat: the grounding event reserves a slot, so max useful events
  per batch drops from 8 to 7. With 577 events total this means 83
  batches instead of 73, ~20s each on EMU → ~28 min full sweep. HW
  faster, probably ~14 min.
- Verify: grounding event must fire before any other event to serve as
  anchor. For `USER_EVENT_1` via `Event_Generate` at the very start of
  `runtime_sequence`, this holds. Sanity-check by asserting that every
  batch's grounding-event timestamp is strictly less than every other
  event's.

### 2. Full-sweep validation harness

**Goal:** "sweep everything" becomes one command. Used after each
refactor step as the go/no-go gate.

**Approach:**

- New script `scripts/trace-sweep-all.sh` that:
  - Takes a test name (or `--all` for the current Phase B 7-test batch)
  - Runs `trace-sweep.py` for each (test × compiler × traced tile) combo
  - Writes per-combo JSON + a consolidated summary
- Per-test tile discovery: parse `aie-hw-cycles-traced.mlir` to find
  which tiles have `aie.trace` ops. Run sweep once per tile per
  applicable tile type.
- Expected runtime for a single test with grounding + full sweep:
  ~30 min per side (HW + EMU), ~60 min total. For all 7 Phase B tests,
  ~7 hours end-to-end. Sweep in background, emit progress.
- Cache: `trace-sweep.py` already reuses the same xclbin/insts across
  batches. For multi-test runs, add an `--only-if-stale` flag that
  short-circuits when the input xclbin hasn't changed and a cached
  result exists.

**Output layout:**
```
build/trace-sweep-results/YYYYMMDD/
  <test>.<compiler>.<tile>.json        # raw per-batch data
  <test>.<compiler>.<tile>.merged.json # grounding-merged stream
  <test>.<compiler>.<tile>.matrix.csv  # tabular view for diff
  summary.json                         # all combos at once
  regressions.txt                      # cells that flipped MATCH→DRIFT
```

### 3. Summary / matrix renderer

**Goal:** make the matrix readable. Rendered output drops into PR
descriptions and regression reports.

**Approach:**

- New `scripts/show-sweep-matrix.py`:
  - Input: one matrix JSON or a directory of them
  - Output options: text table (current one-off script), markdown,
    CSV, or tiny HTML heatmap (fire-count ratio as color)
- Compact text form (for sanity checks):
  ```
  vector_scalar_using_dma.chess core@(0,2)     core@(1,2) memmod@(0,2) ...
  INSTR_VECTOR              MATCH(100/100)  MATCH          MISS
  INSTR_EVENT_0             DRIFT(H4/E2)    -              -
  LOCK_STALL                DRIFT(H17/E1)   -              -
  ...
  ```
- Diff form: given an old JSON and a new JSON, show only cells that
  changed. This is the "did the refactor regress anything?" view.

### 4. Multi-tile sweeps

**Goal:** sweep multiple tiles in one compile. Saves recompile cost
when a test has more than one traced tile (cascade_flows, 1x4_cores).

**Approach:**

- The existing xclbin already has trace flows for every tile mentioned
  in the injected MLIR (our `mlir-trace-inject.py` injects one
  `aie.trace` per compute tile).
- Patcher currently updates one tile at a time. Add a
  `--tiles-events-json` mode that accepts a JSON manifest mapping
  `(col, row, type) → [event_ids]` and patches all of them in one
  pass.
- Sweep orchestrator gains a `--tiles <spec>` flag; each batch patches
  every tile before running. Output matrix becomes 3D: tile × event ×
  side.

**Priority:** do this last. Single-tile v1 already produces useful data;
multi-tile multiplies fidelity but isn't required for regression
verification.

## Open design questions

- **Grounding event choice.** `USER_EVENT_1` via `Event_Generate` is
  cleanest but requires emitting an extra `aiex.npu.write32` into the
  runtime_sequence. AIEInsertTraceFlows already emits this for
  broadcast-start tiles (lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp:714);
  for non-broadcast tiles we'd patch it in ourselves or fall back to a
  broadcast event. Need to verify every traced tile gets a USER_EVENT_1
  trigger without extra patching.
- **Per-event cycle attribution.** v1 reports batch-level cycles only
  (last event timestamp in the buffer). After merging, we could emit
  per-event "first-seen cycle" and "period" numbers — but those only
  work when the event fires multiple times. Decide: include in v2 or
  defer to v3?
- **Merging when batches return different cycle counts.** A 16-event
  sweep on EMU showed 20612 cycles vs HW's 44870 (the buffer-fill
  asymmetry is #138). Once we anchor on the grounding event, cycle
  counts shouldn't conflict — but we should assert and surface if
  anchored timestamps differ between batches for the same event
  (would indicate non-determinism or a grounding-event bug).

## Success criteria

v2 is done when all of the following hold:

1. Running `scripts/trace-sweep-all.sh` on the current `dev` branch
   produces the baseline matrix for Phase B tests in <8 hours.
2. That baseline reproduces on a second run with zero diff (determinism
   check).
3. Taking a known-regressing patch (e.g., reverting `f6b9880` which added
   `INSTR_LOCK_ACQUIRE_REQ` emission) and re-running produces a diff
   that calls out exactly those events as regressed, with no false
   positives on unrelated events.
4. The grounding-event merge produces a timeline where events from
   different batches are correctly interleaved by cycle. Verify with a
   hand-crafted test that fires known events at known cycles.

## Known limitations carried into v2

From v1 smoke tests:

- **EMU trace buffer fills faster than HW for the same kernel** (#138).
  Visible as ~2× fewer events captured on EMU side. Sweep still works,
  but DRIFT classifications reflect both real emission gaps and buffer
  asymmetry. Track separately.
- **HW can fail runtime with state=8 when input BO binding isn't
  first-slot** (#137). Affects `ctrl_packet_reconfig` specifically.
  Sweep treats as `HW-err` — not a v2 bug, just noise to skip.
- **EMU port events (PORT_RUNNING_*, PORT_IDLE_*, PORT_TLAST_*) never
  fire.** Visible in the v1 demo. Either emulator doesn't model port
  activity events, or they're conditional on PortEvent register
  configuration we aren't writing. File as a real EMU gap when v2 fully
  surveys port events.

## Pointers for resuming after compaction

If this context is cleared and you need to resume:

1. Read this file.
2. Read `tools/trace-sweep.py` and `tools/trace-patch-events.py` to
   re-familiarize with v1 shape.
3. `git log --oneline origin/dev..dev` to see what's landed since origin.
4. Verify v1 still works:
   ```
   source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
   python3 tools/trace-sweep.py \
     --test vector_scalar_using_dma --compiler chess \
     --col 0 --row 2 --tile-type core \
     --events INSTR_EVENT_0,INSTR_LOCK_ACQUIRE_REQ \
     --out /tmp/claude-1000/smoke.json --no-hw
   ```
5. Start v2 work from deliverable 1 (grounding event + merge).

## Relationship to other tasks

- `#131` stays `in_progress` until v2 ships.
- `#137`, `#138`, `#139` surface cleanly through the sweep once v2 is
  running but are fixes on the emulator side, separate from the sweep
  infrastructure itself.
