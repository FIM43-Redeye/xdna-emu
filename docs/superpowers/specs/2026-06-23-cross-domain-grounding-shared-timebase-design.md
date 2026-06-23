# Cross-Domain Grounding: Reproduction Targets on Shared Timebase -- Design

**Status:** approved model (brainstorm + 2 toolchain investigations + 3 rounds
adversarial review + direct HW-data verification + an aiesim-oracle investigation,
2026-06-23). This is the **final** form of this spec. The first two forms
(co-observed-anchor, then naive "drop the same_domain gate") and the third
(skew-decomposition, which tried to extract cross-domain causal latency) were each
falsified by review + data and corrected. The reason the third form was abandoned
-- cross-domain causal latency is undeterminable from a trace alone -- is fully
characterized in [`../../trace/cross-domain-skew-limit.md`](../../trace/cross-domain-skew-limit.md);
**read that first.** This spec is the small engine-side change that the boundary
permits. The next #140 frontier after the within-domain jitter-grounding rule
(`2026-06-23-explicit-jitter-robust-grounding-design.md`, merged 2026-06-23).
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Evidence:** `build/experiments/gap140/nondeterminism/add_one_using_dma/`
(20 HW runs); `build/experiments/bcast-bridge/FINDINGS.md`; the limit doc.

## The resolved understanding (one paragraph; full version in the limit doc)

The AIE-ML array shares one BROADCAST_15-synced trace timebase (verified). For a
cross-domain event pair, the recorded offset is `Δwall + skew` -- causal latency
plus broadcast-arrival skew -- and **the trace gives only the sum.** A trace in
isolation cannot split them (three independent walls: §6 of the limit doc), so the
inference engine cannot ground a cross-domain edge as a *causal* segment. It can,
however, record the exact raw offset, which agrees across runs to range 0 -- a
valid **reproduction target** for the emulator. (The decomposition itself is
recoverable later, by the emulator, via skew-free in-domain compute verification
plus toolchain inter-tile latencies; that is downstream Rust work, not this plan
-- limit doc §7-8.)

## What this spec does (the whole change)

Cross-domain edges already fall to `Gap` today (the `same_domain` gate in
`ground_edge` sends them there -- correct, unchanged). Three additions:

1. **Reproduction-target annotation on `Gap`.** A `Gap` gains an optional
   `reproduction_offset`: the exact raw `soc` offset (`offset_exact`, the anchor
   cancels in the pairwise difference so it works cross-domain) **iff** the offset
   agrees across runs (`range <= Q`); otherwise `None`. A jittery cross-domain edge
   (`range > Q`) gets `None` and stays a pure named gap -- the truthful outcome. No
   tolerance, no decomposition: this is the measurable sum, recorded as the byte
   target the emulator must reproduce.

2. **Async-CDC gap-only guard (an existing same-domain bug this surfaced).** Shim
   NoC-egress completion events (`DMA_*_FINISHED` on the shim) currently
   false-ground **same-domain** as a Segment with offset 0 (e.g.
   `DMA_S2MM_0_FINISHED_TASK <- DMA_MM2S_0_FINISHED_TASK`, both `1|0|2`). Their
   timing crosses the async 1 GHz<->960 MHz NoC FIFO to DDR (AM020 CDC) and carries
   non-deterministic synchronizer latency -- it is **not** a cycle-deterministic
   causal fact even though a 20-run sample happened to show range 0. These events
   are flagged **gap-only by event semantics**: they never ground as a Segment
   (within-domain) and never carry a `reproduction_offset` (cross-domain), because
   no deterministic byte target exists for them. The flagged set is derived from
   the event semantics (shim DMA completion crossing to DDR), not hardcoded per
   kernel.

3. **Plumbing + report.** Thread `reproduction_offset` through `facts.py` (the
   `derives` gap fact) and the report projection (`engine.py` /
   `run_experiment.py`) so a cross-domain gap surfaces its byte target. Minimal:
   one optional numeric field, not the deferred structured-facts migration.

## Components

### `tools/inference/grounding.py`

- `Gap` gains `reproduction_offset: Optional[int] = None`.
- `ground_edge`: for a cross-domain edge, compute `offset_exact(...)`; if not
  `None`, return `Gap(child, parent, reproduction_offset=that)`; else
  `Gap(child, parent)`. Within-domain Segment path unchanged. **If either endpoint
  is an async-CDC event (the guard set), return `Gap(child, parent)` with no
  `reproduction_offset` regardless of same/cross-domain or exact agreement** --
  this is the bug fix (within-domain async-CDC no longer grounds as a Segment).
- A small, semantics-derived predicate `is_async_cdc(event_key) -> bool` (shim row,
  NoC-egress `DMA_*_FINISHED`) -- derived from the event name + shim tile kind, the
  same partition the decoder already applies; no new dump field.

### `tools/inference/facts.py` + consumers

The `derives` gap fact carries `reproduction_offset` (backward-compatible accessor,
default `None`). `engine.py` projects it into the gap report rows.

### `tools/inference/run_experiment.py::write_report`

A cross-domain gap row shows its `reproduction_offset` (the byte target) or marks
it absent (jitter / async-CDC). Within-domain segment rows unchanged.

## Provable within-domain regression safety

For within-domain edges that are NOT async-CDC, behavior is byte-identical to
today's Segment path (the guard and the reproduction-offset annotation touch only
the Gap path and the async-CDC set). A regression test on the merged within-domain
fixtures asserts identical segment output. The only intended within-domain behavior
change is the async-CDC fix: the shim `DMA_*_FINISHED` pair becomes a Gap (was a
spurious offset-0 Segment).

## Q -- the measurement floor

`Q = 0`, the single shared comparator (`verifier.py`), used cross-domain unchanged.
The raw cross-domain offset agrees exactly (range 0, verified). A non-zero
cross-domain `Q` is a falsification (diagnose: genuine jitter, async-CDC,
decoder-frame artifact), never an adjustable tolerance.

## Scope

- **In scope:** the three additions above -- cross-domain `Gap` reproduction-target
  annotation, the async-CDC gap-only guard, the facts/report plumbing, and tests.
- **Out (this plan):** any attempt to ground cross-domain edges as causal segments;
  any skew model in the engine. Cross-domain causal facts are out by the boundary,
  not deferred (limit doc).

## Connection to the emulator (downstream, not this plan)

The recorded `reproduction_offset` is the byte target the emulator's broadcast
model will be validated against. The emulator path -- give
`src/device/events/broadcast.rs` a forward per-hop flood model, verify the compute
model skew-free in-domain, then measure the broadcast skew as the cross-domain
residual (`measured − verified Δwall`, toolchain stream/DMA latencies subtracted)
-- is the real cross-domain causal work, and it is **named Rust follow-on work**,
not part of this Python inference plan. See limit doc §7-9.

## Testing

- **Offline units:** `Gap` round-trips `reproduction_offset`; cross-domain
  `ground_edge` returns a Gap with the exact reproduction offset for the verified
  `-2/+2/+4` pairs and `None` for a jittery cross-domain pair (fixtures from the
  20-run data); the async-CDC guard turns the shim `DMA_*_FINISHED` same-domain pair
  into a Gap with no reproduction offset (regression of the false-ground bug);
  `is_async_cdc` classifies shim NoC-egress completions and nothing else; the
  within-domain byte-identical regression; facts/report reproduction-offset
  round-trip.
- **HW acceptance gate:** on real NPU1, the verified cross-domain pairs surface
  their exact reproduction offsets; a shim NoC-egress event stays a Gap with no
  reproduction offset; within-domain segments unchanged; the suite
  (add_one_using_dma + add_one_objFifo + vector_scalar_using_dma) reaches a terminal
  state with zero regressions.

## Plan sequence (one serious thing)

1. **Implement** the `Gap.reproduction_offset` field, the `ground_edge`
   cross-domain annotation, the `is_async_cdc` predicate + gap-only guard, the
   facts/report plumbing, and the test changes.
2. **Validate** offline + the HW gate.

(No calibration spike: there is nothing to calibrate -- the reproduction offset is
the directly-measured exact value.)

## Out of scope (deferred, named follow-ons)

- **The emulator broadcast forward-model + skew-residual measurement** (the real
  cross-domain causal work) -- limit doc §7-9; Rust, its own plan.
- **NoC-egress causal timing** -- irreducibly non-deterministic (async CDC);
  gap-only, never promised.
- **Full structured facts-schema migration** beyond the one `reproduction_offset`
  field.

## Correctness principle

A cross-domain edge is a Gap carrying the exact, directly-measured raw offset as a
reproduction target -- never a causal segment, because causal latency is
undeterminable from a trace alone (limit doc). Async-CDC events are gap-only by
event semantics. The exact-agreement test (`range <= Q`, single shared `Q=0`) is
unchanged. DERIVE FROM THE TOOLCHAIN throughout: the async-CDC set from event
semantics, the offset from measurement, nothing hardcoded per kernel.
