# Jitter-grounding spikes -- measured foundation for #140's grounding rule

**Date:** 2026-06-23
**Reproduce:** `build/experiments/spike-jitter-grounding/{spike.py,spike2.py}`
(gitignored; HW, add_one_using_dma, chess). Run with
`env -u XDNA_EMU PYTHONPATH=tools <ironenv>/python build/.../spikeN.py`.

These two HW spikes settled the model for the explicit-jitter-robust-grounding
design (`docs/superpowers/specs/2026-06-23-explicit-jitter-robust-grounding-design.md`)
*before* committing the plan. They overturned the design's first premise twice.

## Why we spiked

An Opus adversarial review of the first design draft returned NEEDS-REDESIGN with
3 verified CRITICALs, the central one being: cross-run offsets subtract two
**unsynchronized per-tile timers**, so "Q=0 / single-run exactness" was unsound.
Rather than redesign on speculation, we measured.

## Spike 1 (10 runs): the jitter model was inverted

The draft assumed the deterministic part of the through-core span is the
post-delivery remainder (bracket at `MM2S_FINISHED`) and the jitter is delivery.
Measurement showed the opposite:

- `MM2S_START -> S2MM_START` (both shim, one domain) was stable ~935 in 9/10 runs
  while the FINISHED/delivery events scattered ~3700 cycles. The deterministic
  thing is the **issue cadence**, not the delivery; FINISHED **is** the jitter.
- One run's whole shim pipeline issued +3000 cycles late vs the core PERF_CNT_2,
  yet its within-shim span was still 937 -> **cross-domain timer skew is real but
  cancels within a domain** (CRITICAL-1 scoped: cross-tile only; through-core,
  being within-shim, is unaffected).
- One within-domain span outlier remained (890 vs 935): the open question.

## Spike 2 (20 runs, core lock/stall/contention signature): DECISIVE

Added core `INSTR_LOCK_ACQUIRE_REQ/RELEASE_REQ`, `ACTIVE`, `LOCK_STALL`,
`MEMORY_STALL`, `LOCK_ACCESS_TO_UNAVAILABLE`, `GROUP_STALL` to read each run's
structure.

**The deterministic unit is a within-domain milestone-bounded segment:**

```
core INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ = 22 cycles
ALL 20 runs, range 0 -- including run_01 (span 3354), run_12 (acquire@5029),
run_16 (span 900).
```

- Lock STRUCTURE identical every run (15 acquires / 16 releases / 0
  lock-unavailable / 0 memory-stall): the span outliers are NOT an explainable
  kernel-level structural class. They are delivery jitter on the WAIT (when the
  acquire fires: absolute 990..5029) plus trace-load perturbation.
- **Observer effect:** spike 1 (1 core trace event) -> span stable ~935; spike 2
  (7 core events incl. high-frequency ACTIVE ~132x / GROUP_STALL ~149x) -> span
  scattered 900..3354. Heavy trace perturbs timing. BUT the compute segment held
  exactly 22 under heavy trace -> **within-domain exact-agreement grounding is
  observer-effect-robust**; low-frequency event selection is a refinement, not a
  correctness requirement.

## The model the plan is built on

- **Deterministic, cycle-accurate, Q=0 literally:** within-timer-domain (same
  tile) segments whose per-run offset agrees EXACTLY (range 0). Exact agreement is
  equality, not statistics -- it both classifies and verifies magnitude.
- **Jitter = named gaps:** within-domain offsets that do NOT agree exactly (they
  bundle a wait) and all cross-domain offsets. Existence + orientation only.
- **Through-core deliverable:** `gap + (exact 22-cycle compute) + gap`, never one
  deterministic number for the bundled 935 span.
- **Deferred (own plans):** cross-domain timer-sync (BROADCAST_15 timer-reset);
  active low-frequency event selection.

## Spike 3 (15 runs, memtile PORT_RUNNING) + 2nd Opus review: Q=0 does not hold for PRODUCIBLE segments

The 2nd Opus review (of the measured rewrite) verified two facts that reshape the
plan:
- **C1:** the trace timer is per-MODULE, not per-tile (separate `core_timer` /
  `mem_timer`, decoder resets `timer=0` per `(pkt_type,row,col)`). So
  `same_domain` must key on `(col,row,pkt_type)`, not `(col,row)`. (Trivial fix;
  the spike's core->core 22-cycle segment was already single-module, so the
  measurement stands.)
- **C2:** the core `INSTR_LOCK_ACQUIRE_REQ/RELEASE_REQ` segment (the range-0
  exact one) is NOT producible -- `event_map` orients only PORT_RUNNING / DMA
  events, not instruction-lock events, and there is no edge kind for an
  acquire->release instruction pair. The producible within-domain pairs are
  PORT_RUNNING / DMA on the same module.

Spike 3 measured the producible candidates (memtile buffer relays, both ends
pkt=3, same module/timer), 15 runs:

```
PR0->PR4   range 1   [29, 30]        (the Plan-2 relay)
PR1->PR4   range 1   [-79, -78]
PR0->PR1   range 2   [107, 108, 109]
PR0->PR5   range 4 ; PR1->PR5 range 4 ; PR4->PR5 range 4
```

**None are range 0.** The producible within-domain deterministic segments carry a
small bimodal/trimodal +/-1..4 spread -- a measurement/decode quantum (classic
timestamp-frame quantization), three orders of magnitude below real jitter
(within-domain 1..4 vs jitter 1000s). So:

- The deterministic-vs-jitter DISCRIMINATION is crisp and unambiguous (1..4 vs
  1000s).
- But `Q=0` (literal exactness) holds only for the non-producible core-lock
  segment. For producible segments, grounding needs either a DERIVED measurement
  quantum Q (~2..4, from decode/frame resolution -- the "derived measurement
  floor" Maya pre-approved, NOT a tuned tolerance), or a decoder-precision fix to
  reach true range 0, or exposing the instruction-event layer to make the range-0
  core segments producible.

This is the open keystone decision (it touches the no-statistics line) -- see the
spec's Q section + the plan fork.

## Decode diagnosis (runs3): the decoder is CORRECT; the +/-1..4 is genuine hardware on level events

Diagnosing the +/-1..4 on the producible memtile relays (PORT_RUNNING pairs).

**FALSE ALARM CORRECTED.** An initial hand-rolled command walk appeared to show two
of our decode paths disagreeing on PR4's first occurrence by 28 cycles (335478 vs
335506). That was a BUG IN THE DIAGNOSTIC, not the decoder: the walk omitted
`RepeatCmd` handling, and there are two `RepeatCmd count=14` early in the memtile
stream (335478 + 28 = 335506 exactly). The real decoder is self-consistent:
`parse_trace` / `rebuild_timeline_mode0` reproduce the events.json exactly
(slot4 first soc = 335506). **No decoder inconsistency exists.**

What the (now-correct) diagnosis shows:

- **EDGE / instruction events decode range-0 EXACT** -- the core compute segment
  (LOCK_ACQUIRE_REQ -> LOCK_RELEASE_REQ, single-fire) was 22 cycles, range 0 over
  20 runs. The decoder handles single-fire edge events precisely.
- **LEVEL events (PORT_RUNNING) carry genuine +/-1..4 HARDWARE variation.** Since
  the decoder is a deterministic, self-consistent function of the bytes, and the
  raw byte streams genuinely differ run-to-run (first PR0 span 2747 vs 3608
  cycles), the +/-1..4 is in the captured SIGNAL, not a decode artifact. Level
  (port-active span) timing simply varies +/-1..4 at the hardware.

**Reframe:** "decoder-precision fix first" is moot -- there is no decode bug to
fix; the decoder is correct and edge events are already exact. The precision
problem (spike 3) and the producibility problem (C2) share one root: we measured
on PORT_RUNNING (producible but genuinely +/-1..4 at the hardware) instead of edge
events (exact but not yet producible). The path to cycle-exact grounding is to
**ground on EDGE events.** The proven-exact edge segment (core compute = 22)
requires exposing instruction-lock events (the C2 static-orientation layer).

**Residual tooling due-diligence (Maya's concern):** our `trace_decoder` is
self-consistent and edge-exact, but its docstring notes it "mirrors the public
surface of mlir-aie's parse_trace so this can drop in *once validated*" -- it has
NOT been validated against the upstream mlir-aie reference decoder. That
validation is worth doing for tooling confidence, independent of this finding.

## Decoder validation vs upstream mlir-aie (Maya's tooling concern): PASS

Our `trace_decoder` had never been validated against the upstream
`aie.utils.trace.parse_trace` (its docstring: "once validated"). Validated now via
the existing `tools/parse-trace.py` dual-backend wrapper, on NATIVE (unpatched)
traces whose bytes match their MLIR (the patched HwInstrument traces are
apples-to-oranges because our runtime patcher reconfigures tiles the upstream
decoder reads from the original MLIR).

Method (`build/experiments/spike-jitter-grounding/validate_decoder.py` + a
dma_passthrough check): decode the same trace.bin with both backends, group by
`(pkt,row,col,slot)`, require count agreement AND relative-timing agreement
(within a timer domain the two differ only by a constant ts-origin convention
offset; after removing it every timestamp must match).

Result -- **PERFECT AGREEMENT**:
- 4 core-traced calibration kernels (write32_compute/mem/shim, blockwrite):
  12/12 groups MATCH, including a slot with 4714 firings.
- dma_passthrough_w32_n12 (native, traces core AND memtile): 7/7 MATCH,
  including 4 MEMTILE slots (level events) at 12 firings each, exact.

**Conclusions:**
- Our decoder is validated against upstream: count + relative timing agree
  exactly across core and memtile, edge and level events, on native traces.
- The earlier patched-trace memtile divergence (ours 11 vs upstream 4) was
  PATCH CONTAMINATION (our patcher's memtile config vs the MLIR's), not a bug.
- Therefore the +/-1..4 on level events (spike 3) is GENUINE HARDWARE signal
  variation -- both decoders agree on the bytes; the bytes themselves vary
  run-to-run. There is nothing to "fix" in the decoder.
- Durable follow-up worth doing: graduate `validate_decoder.py` into a permanent
  regression test (ours-vs-upstream on native traces), closing the "once
  validated" gap for good.

With the decoder trusted, the grounding direction stands: ground on EDGE events
(validated exact), expose instruction-lock events for the through-core compute
segment (C2 layer); level events are genuinely fuzzy hardware -> gaps.

## Opus review findings, re-adjudicated against measurement

- CRITICAL-1 (timer skew): confirmed real; scoped to cross-tile, cancels
  within-domain -> does not block within-shim through-core.
- CRITICAL-2 (decoder Q justification stale): upheld -> dropped from the design.
- CRITICAL-3 (classify_edge DDR-source absent from dump): premise moot -- the real
  classifier is same-domain + exact-agreement, derived from the event key.
- IMPORTANT-5 (single-run can't verify magnitude): dissolved -- grounding uses
  exact cross-run agreement (range 0, proven), which verifies magnitude without
  statistics. "Single-run" relaxes to "exact agreement across the run set."
