# Three-way timing calibration (Stage 2, task #72)

**Status**: BUILD COMPLETE through item 8 (2026-06-06); steps 9 (--aiesim-timing harness integration) and 10 (full 72-kernel campaign) pending.

## PIVOT: in-process NPU1 VCD, not standalone vc2802 (2026-06-06, Maya)

The original plan pulled the aiesim **timing** VCD from the **standalone**
aiesimulator (`--aiesim`, Phase 5c), which only models AIE2 as `xcve2802`
(Versal 38x11). That forced a region-dependent geometry remap (vc2802 -> NPU1)
on every anchor -- a remap `aiesim-validate.py:574` admits was never even
implemented in `vcd_compare`.

Maya's question -- "why normalize at all, didn't we build an NPU1 JSON?" --
exposed the real cause: normalization was an artifact of using the *wrong*
aiesim. The NPU1 JSON drives the **in-process cluster** (the correctness
oracle), which simply wasn't dumping a VCD. The design defaulted to the
standalone because it already had `--dump-vcd` wired.

**The in-process cluster can dump its own NPU1-native VCD.** It is a real
SystemC sim (`sc_bootstrap.cpp` owns `sc_main`/`sc_start`, links aietools
`libsystemc`, hosts the `aie_cluster` model). The cluster model exposes the
standard SystemC trace hook -- `math_engine_base.h:124`:
`virtual void add_sc_traces(sc_trace_file *tf) = 0;` -- and our `sc_main`
already holds the top object (`aiesim_top::math_engine()`, constructed at
`sc_bootstrap.cpp:209`). So `sc_create_vcd_trace_file()` + `top->add_sc_traces(tf)`
yields a VCD in **NPU1 5x6 geometry, identical to the trace-BO/HW side**.

Consequences:
- **Zero geometry normalization** -- pure NPU1<->NPU1 anchor alignment.
- **One run, one config** -- the same in-process execution produces correctness
  *and* timing, so the "report timing only where standalone-VCD and
  NPU1-correctness configs agree" caveat (old decision 3) evaporates.
- **Truer Phoenix oracle** -- NPU1-native, not Versal vc2802 (wrong array size /
  different NoC). This is what we want post-Strix.

The standalone `--aiesim` path is now **very deprecated** (Maya) -- kept only as
an optional independent cross-check, to be cut once the bridge timing path is
solid. New build order is tasks #87 (wire `add_sc_traces`) -> #88 (in-process
NPU1 mapping tree + coverage) -> #84 (anchor extractor, no normalization). The
"geometry wrinkle" section below is retained for historical context but no
longer governs the primary path.

**Decisions (2026-06-06, Maya):**
1. Architecture: **C -> B** -- total-cycle direct three-way first, then layer
   per-anchor detail on the same comparator. `vcd_compare` interp<->aiesim kept
   as a free cross-check column.
2. Envelope: **per-anchor (B) on all 72** aiesim-capable PASS kernels (full
   exhaustive Phoenix capture; aiesim standalone is slow but the window is ours).
3. Geometry: report aiesim timing only on kernels where the standalone-VCD
   config and the NPU1-correctness config agree on correctness (default; Maya
   to confirm if she wants otherwise).

**Step 0 feasibility -- DONE (2026-06-06).** `vcd-compare --coverage` on a real
standalone-aiesim VCD (`build/experiments/aiesim-real-harness/pkg/chess_run.vcd`)
maps 16944 DMA / 10336 lock / 17024 event / 8968 stream signals (43.6% overall;
the unmapped remainder is `SystemC.tl.aie_logical` internals, irrelevant to
timing anchors). The anchor vocabulary is extractable from the aiesim VCD. Gate
PASSED.

## Data contract (schema-first)

One **timing record** per (kernel, source). Option C populates `total_cycles`
only; Option B adds `anchors`. Sources: `hw`, `interp`, `aiesim`.

```json
{
  "kernel": "add_one_using_dma",
  "compiler": "chess",
  "source": "hw",
  "total_cycles": 12345,
  "anchors": [
    {"col": 1, "row": 2, "kind": "dma_s2mm0_start", "cycle": 1000},
    {"col": 1, "row": 2, "kind": "dma_s2mm0_done",  "cycle": 1200},
    {"col": 0, "row": 2, "kind": "lock4_acquire",   "cycle": 980}
  ]
}
```

- `hw`/`interp` records derive from `parse-trace.py` output (total = `--out-cycles`;
  anchors reduced from `--out-events`).
- `aiesim` records derive from the standalone VCD via `src/vcd` (total = active
  span across mapped signals; anchors from `DmaFsmState`/`Lock` transitions).
- Anchor coords are normalized to **NPU1 geometry** (the aiesim/xcve2802 row
  offset is removed at extraction time) so all three align on `(col,row,kind)`.
- The comparator reads the three records per kernel, aligns on `(col,row,kind)`,
  and reports per-anchor cycle deltas + total-cycle drift, **HW as ground truth**.

Goal: confirm aiesim's *timing* fidelity against real Phoenix, alongside
correctness, across the aiesim-capable kernel envelope (the 72 PASS kernels
from Stage 1, task #71). This is the calibration that lets aiesim stand in as a
Phoenix timing oracle after the Strix swap.

This doc exists because "three-way timing" is not a single obvious pipeline --
the three sources expose timing through two incompatible mechanisms, and
bridging them forks into three real architectures. Pick one before building.

## The core constraint

| Source | Timing mechanism | Common with |
|--------|------------------|-------------|
| Real HW (Phoenix NPU1) | AIE **trace BO** -> `parse-trace.py` -> events (per-slot, absolute `ts`) + total-cycle scalar (`max(ts)-min(ts)`) | interp |
| Interpreter EMU | **trace BO** (same pipeline as HW) AND its own **VCD** (`src/vcd/emit.rs`, feature-gated) | HW *and* aiesim |
| aiesim (AMD standalone) | **VCD** waveform -> `src/vcd/` mapping -> per-signal `SignalTimeline` of `(cycle, value)` transitions | interp |

HW can *only* produce trace-BO events (no VCD). aiesim can *only* produce a VCD
(the in-process NPU1 cluster does **not** model the trace unit -- proven
2026-06-06: it rejects trace-register AXI accesses, trace BO comes back empty).
The interpreter is the only side fluent in both.

What already exists:
- HW <-> interp timing: trace-BO cycles, both sides, via `_run_trace_cycles_pipeline`
  + `parse-trace.py` (`--out-cycles`, `--out-events`). Mature.
- interp <-> aiesim timing: `vcd_compare` (`src/bin/vcd_compare.rs`, `src/vcd/compare.rs`)
  aligns interp-VCD vs aiesim-VCD signal-for-signal, reports per-signal
  `TimingOffset { offset_cycles }` (median cycle delta) / ExactMatch / Mismatch.
  Mature.
- HW <-> aiesim timing: **nothing.** This is the missing leg.

## The geometry wrinkle

The aiesim VCD is produced by the **standalone aiesimulator** on a `xcve2802`
(Versal AIE2) chess build via the `--aiesim` path (Phase 5c), NOT the in-process
NPU1 cluster used for correctness. Tiles are architecturally identical but:
- Geometry is Versal 38x11, not NPU1 5x6.
- Row offset: NPU1 core row N == xcve2802 core row N+1 (per `aiesim-validate.py:54`).
- So timing-aiesim and correctness-aiesim are two different aiesim configs; any
  HW<->aiesim alignment must apply the row offset and restrict to the columns
  the kernel actually uses.

(Open feasibility item for any VCD-based option: verify the standalone `--aiesim`
VCD path still runs and yields usable DMA/lock timelines on a couple of corpus
kernels before committing. The `src/vcd` subsystem exists to consume it, so it
has worked before, but it may be stale.)

## Architecture options

### Option A -- Interpreter-as-pivot (transitivity)

Compose the two existing pipelines. Report HW<->interp (trace cycles) and
interp<->aiesim (VCD offsets) side by side; aiesim's drift-from-HW is inferred
through the interpreter.

- **Build**: minimal -- both halves exist; mostly a report that joins them.
- **Strength**: leverages the mature `vcd_compare`; fast to stand up.
- **Weakness**: indirect. Interpreter timing error pollutes *both* legs, so a
  clean "aiesim matches silicon" number is inferred, never measured. If interp
  is the thing whose accuracy we're least sure of, pivoting on it is circular.

### Option B -- Direct common-anchor timeline

Define a small, robust **macro-anchor vocabulary** that BOTH the trace BO and
the aiesim VCD can yield without lossy slot reconstruction: per-tile DMA-channel
first-start / last-done, lock-event cycles, core compute start/end. Build a
`VCD -> anchor-timeline` extractor (reuse `src/vcd::extract_timelines` +
`DmaFsmState`/`Lock` mappings) and a `trace-events -> anchor-timeline` reducer
(group `parse-trace` events by tile/channel). One comparator aligns all three on
shared anchors; reports per-anchor cycle deltas + total-cycle drift, HW as
ground truth.

- **Build**: most new code (two extractors + comparator + geometry alignment).
- **Strength**: *direct* HW<->aiesim timing. Robust anchors dodge the lossy
  "which trace slot fired" reconstruction the VCD can't support.
- **Weakness**: most effort; geometry/row-offset alignment; depends on the
  standalone VCD feasibility item above.

### Option C -- Total-cycle direct (pragmatic first cut)

Compare just **total kernel cycles** three ways (HW trace span, interp trace
span, aiesim VCD total span), plus optionally per-tile active-window
(first-activity .. last-activity cycle). One scalar + a few per-tile numbers per
kernel.

- **Build**: small; direct three-way; robust.
- **Strength**: an early *direct* HW<->aiesim number across the whole envelope;
  catches gross drift immediately; natural foundation to layer B onto.
- **Weakness**: coarse -- misses per-event structure (a kernel can have the
  right total but wrong internal cadence).

## Recommendation

Target **B** (direct per-anchor is the real calibration signal), but build in
the order **C -> B**: land Option C first so we have a working direct three-way
total-cycle number across the full envelope early, then layer B's per-anchor
detail on the same comparator. Treat Option A's `vcd_compare` interp<->aiesim
output as a *cross-check* column, not the primary signal -- it's free and
catches interp-vs-aiesim divergence the anchor view might smooth over.

Step 0 regardless of choice: verify the standalone `--aiesim` VCD path runs on
~2 corpus kernels and that `src/vcd` maps its DMA/lock signals (feasibility gate
for B and C).

## The interpreter VCD leg (Maya, 2026-06-06)

Once VCD timing extraction is trustworthy, the interpreter's *own* VCD output
(`src/vcd/emit.rs` `VcdRecorder`, behind the `vcd-recording` feature) becomes a
usable timing artifact too. That gives the interpreter **two** timing
representations and closes every edge of the triangle with a *direct*
measurement:

- interp **trace BO** -> cycles, comparable to **HW** (existing).
- interp **own VCD** -> cycles/anchors, comparable to **aiesim** (this leg).

The anchor/`cycle_span` extractor is device-parameterized (takes a `MappingTree`),
so it would run on an interp VCD (NPU1 tree) with no new code, and the existing
`vcd-compare --emu <interp.vcd> --sim <aiesim.vcd>` already aligns two VCDs
signal-for-signal with per-signal `TimingOffset` (median cycle delta). The
consumer side is free.

**BUT the producer side is not built.** `VcdRecorder` (`emit.rs`) has zero
callers in the execution path -- the interpreter does **not** emit a VCD today.
Wiring it requires instrumenting the device state machine to record every
DMA/lock/stream/core state change (the archived Phase C Task 9, explicitly
deferred). So this leg is a real follow-on, tracked as task #86.

Why it doesn't block the campaign: the interpreter is **already** in the
three-way via its **trace BO** (same `bridge-trace-runner -> parse-trace` path
as HW), so it gets a trace-cycles/anchors column directly comparable to HW.
Only the *direct FSM-level* interp<->aiesim VCD edge is missing, and it is **not
Phoenix-gated** (both sides available anytime) -- do it after the HW<->aiesim
campaign.

Anchor matching approach (Maya, 2026-06-06): **fixed canonical anchor set** --
a small universal vocabulary (DMA ch0/ch1 first-start & last-done, lock
acquire/release, compute start/end), extracted from the VCD via FSM/lock
transitions and from the trace BO by recognizing matching event names; report
only anchors both sides carry.

## Canonical anchor vocabulary (grounded on real data, 2026-06-06)

Inspected a real EMU `events.json` (`add_one_using_dma.chess.emu`). The trace BO
carries DMA start/done as named events with a precise `soc` (start-of-cycle
absolute) field, tagged by `(col, row)`:

| Anchor kind | Trace BO event name | aiesim/interp VCD source |
|-------------|---------------------|--------------------------|
| `dma_s2mm{ch}_start` | `DMA_S2MM_{ch}_START_TASK` | `DmaFsmState{dir=S2mm,ch}` first leave-idle |
| `dma_s2mm{ch}_done`  | `DMA_S2MM_{ch}_FINISHED_TASK` | `DmaFsmState{dir=S2mm,ch}` done / last change |
| `dma_mm2s{ch}_start` | `DMA_MM2S_{ch}_START_TASK` | `DmaFsmState{dir=Mm2s,ch}` first leave-idle |
| `dma_mm2s{ch}_done`  | `DMA_MM2S_{ch}_FINISHED_TASK` | `DmaFsmState{dir=Mm2s,ch}` done / last change |

DMA START/FINISHED tasks are the strongest universal anchors -- present, named,
and semantically unambiguous on both sides. Lock (`LOCK_STALL`,
`INSTR_LOCK_ACQUIRE_REQ`) and `PORT_RUNNING_N` events exist too and can extend
the set later, but the DMA task anchors are the v1 vocabulary.

Notes for the B build:
- Cycle anchor: use the trace event's `soc` field (precise start-of-cycle), not
  `ts`. VCD side: change-time / derived period.
- Coordinates: trace BO is NPU1 geometry already; aiesim VCD is xcve2802 -- apply
  the row offset (NPU1 row N == vc2802 row N+1) at VCD extraction so both align
  on `(col,row,kind)`. Shim DMA (row 0) needs explicit care -- verify the aiesim
  VCD exposes shim-tile DmaFsmState before relying on shim anchors.
- Alignment: absolute cycle zeros differ between sources (sim start vs trace
  start), so the comparator aligns all sources on a **shared reference anchor**
  -- the `(col,row,kind)` present in every available source, earliest by HW
  cycle -- and expresses every anchor relative to that reference within its own
  source before computing per-anchor deltas vs HW. (An earlier draft normalized
  each source to its *own* earliest anchor; that silently assumes the first
  event is the same across sources, injecting a spurious constant offset when it
  isn't. The shared reference removes that failure mode.)

## Build plan

1. **C-aiesim extractor**: `vcd-compare --cycles <vcd>` -> total active-cycle
   span, symmetric with `parse-trace.py --out-cycles`. (TDD against an existing
   corpus VCD.)
2. **Timing-record emitters**: small adapters producing the data-contract JSON
   for each source (hw/interp from parse-trace output; aiesim from vcd-compare).
3. **C comparator + report**: three total-cycle records per kernel -> drift
   table, HW ground truth.
4. **(#87) In-process VCD dump**: env-gated `add_sc_traces` in `sc_bootstrap.cpp`
   sc_main -> NPU1-native VCD. Verify non-degenerate timeline under the chunked
   `sc_pause`/`sc_start` driver model.
5. **(#88) In-process NPU1 mapping tree**: `--coverage` the new VCD to discover
   the `aiesim_top.*` hierarchy; add `build_npu1_inproc_mapping_tree`. Confirm
   `DmaFsmState`/`Lock` resolve.
6. **(#84) B-aiesim anchor extractor** -- DONE. `vcd-compare --anchors <vcd>`
   (`src/vcd/anchors.rs`) reduces each DMA channel's typed `status`
   ([`StatePath::DmaStatus`]) timeline to anchors: `dma_{dir}{ch}_start` =
   first leave-idle change, `dma_{dir}{ch}_done` = last change. **No
   normalization** (native NPU1 tree). Verified on the in-process fixture: 16
   anchors across shim/mem/compute, cycles matching raw VCD inspection.
   - Encoding note: the `status` FSM is `0`=idle, `1`=acquire, `2`=run. Shim
     channels return to idle (`0->1->2->0`); compute/mem oscillate `1<->2` per
     BD and the capture ends mid-run, so "done" is universally the last change,
     not a return to idle. The rule needs only "0 is idle".
   - The richer per-channel `start_task`/`finished_task` pulse signals exist but
     are inconsistently emitted (shim does not pulse them), so `status` is the
     robust universal anchor source.
7. **(#84) B trace-anchor reducer** -- DONE. `tools/trace-anchors.py` reduces
   `parse-trace --out-events` JSON to the same anchors: `DMA_{DIR}_{ch}_{START,
   FINISHED}_TASK` -> canonical kind, `soc` cycle (first-`soc` for `_start`,
   last-`soc` for `_done`, matching the VCD first-leave-idle/last-change rule).
   Event names are grounded on mlir-aie's `python/utils/trace/setup.py` enum.
8. **(#84) B comparator** -- DONE. `timing-three-way.py --per-anchor` aligns
   anchors on `(col,row,kind)` against a **shared reference anchor** (the key
   present in every source, earliest by HW cycle; marked `*` in the report) and
   reports per-anchor deltas vs HW (+ per-kernel and overall mean |Δ|). The
   shared reference makes deltas independent of both absolute origins and of
   which channel each source arms first -- a selftest locks in that it reports 0
   drift on a shared anchor where per-source-earliest would have reported a
   bogus 600-cycle offset. End-to-end verified on real fixture data: a
   +50000-cycle origin shift aligns away cleanly and an injected +30-cycle drift
   surfaces as the sole nonzero delta.
9. **(#85) Harness integration**: `--aiesim-timing` mode in emu-bridge-test.sh
   joining all three from the **in-process** run (correctness + timing in one).
10. **(#85) Campaign**: all 72 kernels, three-way, correctness + timing; close-out
    finding. (Standalone `--aiesim` path deprecated -- optional cross-check only.)
