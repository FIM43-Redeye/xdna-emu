# Three-way timing calibration (Stage 2, task #72)

**Status**: DESIGN APPROVED (2026-06-06). Building.

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

## Build plan

1. **C-aiesim extractor**: `vcd-compare --cycles <vcd>` -> total active-cycle
   span, symmetric with `parse-trace.py --out-cycles`. (TDD against an existing
   corpus VCD.)
2. **Timing-record emitters**: small adapters producing the data-contract JSON
   for each source (hw/interp from parse-trace output; aiesim from vcd-compare).
3. **C comparator + report**: three total-cycle records per kernel -> drift
   table, HW ground truth.
4. **B-aiesim anchor extractor**: `DmaFsmState`/`Lock` transitions -> anchors,
   NPU1-normalized coords.
5. **B trace-anchor reducer**: `parse-trace --out-events` -> anchors.
6. **B comparator**: extend (3) with per-anchor alignment + deltas.
7. **Harness integration**: a `--aiesim-timing` mode in emu-bridge-test.sh that
   runs the standalone `--aiesim` VCD path on the envelope and joins all three.
8. **Campaign**: all 72 kernels, three-way, correctness + timing; close-out
   finding.
