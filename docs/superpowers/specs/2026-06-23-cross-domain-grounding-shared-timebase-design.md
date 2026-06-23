# Cross-Domain Grounding on the Already-Shared Timebase -- Design

**Status:** approved model (brainstorm + 2 toolchain investigations + 2 rounds
adversarial review + direct HW-data verification, 2026-06-23). **Supersedes** the
co-observed-anchor design (`2026-06-23-cross-domain-timer-sync-grounding-design.md`,
removed) -- the second adversarial round + a direct check of 20-run HW data
falsified that design's premise. The next #140 frontier after the within-domain
jitter-grounding rule
(`2026-06-23-explicit-jitter-robust-grounding-design.md`, merged 2026-06-23).
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Evidence:** `build/experiments/gap140/nondeterminism/add_one_using_dma/`
(20 HW runs); compiled `aie_traced.mlir` Timer_Control writes; the verification
tables in "The discovery" below.

## The discovery (this design's foundation -- measured, not assumed)

The trace timebase is **already synchronized, cycle-exact, across the AIE-ML
array.** mlir-aie's trace lowering (`AIEInsertTraceFlows.cpp` Phase 4b/4f) compiles
`Timer_Control.Reset_Event = BROADCAST_15` into every traced array tile, and the
shim generates the `USER_EVENT_1` that becomes that broadcast. Verified in the
compiled binary (`aie_traced.mlir`): core `Timer_Control` (0x34000) and memmod
`Timer_Control` (0x14000) at the compute tile both = 0x7A00 (Reset_Event = 122 =
BROADCAST_15); shim `Timer_Control` (row 0) = 0x7F00 (Reset_Event = 127 =
USER_EVENT_1, the generator).

20 HW runs of add_one_using_dma, first-occurrence `soc` offsets:

| pair | offset | range over 20 runs |
|------|--------|--------------------|
| core(1,2) <-> memmod(1,2) -- same-tile, cross-module | -2 | **0** |
| core(1,2) <-> memtile(1,1) -- cross-tile | +2 | **0** |
| memmod(1,2) <-> memtile(1,1) -- cross-tile | +4 | **0** |

The offsets are additively consistent (`2 - 4 = -2`) and rock-stable while the
absolute timestamps swing ~14,000 cycles run-to-run. That swing is the shared
broadcast-generation instant `T_gen` (host-trigger jitter), which **cancels in
any offset**; the small fixed integers are deterministic broadcast propagation
between tiles/modules. **Conclusion: cross-domain trace offsets are already
cycle-exact for the array.**

### The falsified premise (and an honest correction)

The superseded design (and the original within-domain spike) assumed cross-domain
timers were *unsynchronized* and that we had *patched out* the broadcast sync.
Both are false:

- Our capture pipeline does **not** patch out the broadcast trace-start.
  `trace_capture.py::configure_batch` patches only the trace `mode` field; it never
  sets `start_event`, so the compiled `BROADCAST_15` start and the compiled
  `Timer_Control.Reset_Event = BROADCAST_15` remain live. (The earlier
  "[VERIFIED] we patched it out" claim trusted the `trace-patch-events.py`
  *docstring* -- a capability the standard pipeline never invokes -- instead of
  checking the actual capture path. Corrected.)
- The original spike's "cross-domain timer skew is real" (the +3000-cycle shim
  observation) was a **misdiagnosis**: on an already-synced timebase, a +3000
  shim-vs-core offset is genuine workload/delivery jitter (correctly a Gap), not
  timer skew. The remedy is not to synchronize timers (already synchronized) but
  to let the exact-agreement test classify each cross-domain edge.

## Goal

Extend grounding to **cross-domain offsets** by recognizing the timebase is
already shared. Within the array, a deterministic cross-domain edge becomes an
exact **Segment**; a jittery one (e.g. a delivery-gated wait) stays a named
**Gap** -- the identical discrimination the within-domain rule already performs.
No statistical inference; exact cross-run agreement (`range <= Q`, Q=0).

## Architecture (the whole change)

The within-domain rule's `same_domain` gate exists only because of the
now-falsified "cross-domain timers are unsynced" premise. Since they are
synchronized, the gate is an artificial barrier that discards groundable
cross-domain edges. The change:

```
# today (grounding.py::ground_edge)
if same_domain(child, parent):
    off = offset_exact(run_dirs, child, parent, anchor_key)
    if off is not None: return Segment(child, parent, off)
return Gap(child, parent)

# new
off = offset_exact(run_dirs, child, parent, anchor_key)
if off is not None: return Segment(child, parent, off)
return Gap(child, parent)
```

`offset_exact` already anchors both endpoints to a common anchor and tests
`range <= Q`; on the shared timebase the anchor value **cancels in the pairwise
offset** (`offset = child_soc - parent_soc`), so the result is well-defined
cross-domain. Deterministic cross-domain edges (propagation + deterministic
causal latency) ground as Segments with the measured exact offset; jittery edges
fail exact agreement and stay Gaps. **That is the design.** Everything the
superseded spec proposed -- a co-observed sync broadcast, `alignment.py`,
`broadcast_routing.py`/hop derivation, a new injector tool, a pluggable alignment
seam -- is unnecessary: the silicon already provides the shared timebase, and the
measured exact offset already includes propagation (no hop correction needed,
because we measure the offset rather than predict it).

### Provable regression safety

Removing the gate does **not** change the within-domain code path: today's
`same_domain` branch is exactly `offset_exact -> Segment|Gap`, which is what
remains after removal. Within-domain results are byte-identical by construction;
cross-domain edges that previously short-circuited to Gap now get tested. A
regression test on the merged within-domain fixtures asserts identical output.

## Scope: what grounds, what doesn't

- **In scope (cycle-exact, shared timebase):** all AIE-ML-clock domains -- core,
  memory module, mem-tile, and shim *control* events. The shim timer is
  array-clocked (1 GHz, AM020 ch.1 single-array-clock-domain) and resets
  deterministically on USER_EVENT_1.
- **Naturally gaps (not a special case):** **NoC-egress timing** -- a shim
  DMA-to-DDR completion (`DMA_*_FINISHED`) fires after a NoC round-trip across the
  async 1 GHz<->960 MHz FIFO (AM020: "Asynchronous, clock domain crossing"). Its
  CDC synchronizer latency is non-deterministic, so its cross-run offset will
  exceed `Q` and the exact-agreement test will correctly classify it as a Gap. No
  special exclusion mechanism is needed -- but this is the one place to watch for
  a *rare-CDC-jitter false Segment* (an event that happens to show `range 0` in a
  finite sample yet is physically cross-clock). Such a false Segment is a known
  suspect, not a promised cycle-exact result; if it ever surfaces it is diagnosed,
  not trusted.

## Residual check before grounding is trusted cross-tile (a small spike)

The 20-run data definitively covers same-tile (core<->memmod) and intra-array
cross-tile (core/memmod <-> mem-tile). It does **not** directly cover the shim:
no batch co-traced a shim row-0 event with a core row-2 event, so shim<->array
exactness is *inferred* (deterministic USER_EVENT_1 reset) but not measured. The
plan's first step is a cheap confirming spike: one batch co-tracing a shim
*control* event (e.g. `DMA_MM2S_0_START_TASK`) and a core event over ~20 runs;
confirm the control offset is exact (`range 0`, Segment) while a shim
*delivery* event (`DMA_S2MM_0_FINISHED_TASK`) is jittery (Gap). If shim control
is not exact, the scope narrows to non-shim tiles and the finding is recorded.

## Facts & reporting

No fact-schema change is required: a cross-domain Segment is an ordinary
`derives` Segment (`kind="segment"`, the measured offset). The offset legitimately
includes broadcast propagation -- which is correct for trace *reproduction* (the
emulator must reproduce the real trace's timestamps, propagation included). An
optional, deferred enhancement may flag a Segment as cross-domain in the report
for human readability; it is **not** needed for grounding and must not grow the
fact arity (the structured-facts migration stays deferred).

## Q -- the measurement floor

`Q = 0`, empirically confirmed cross-domain by the 20-run tables above (range 0
for every deterministic pair). A non-zero cross-domain `Q` is a **falsification**
(diagnose the residual: genuine jitter, a NoC-CDC term, a decoder-frame artifact),
never a tuned constant. The cross-domain check uses the same literal-`0`
comparator as within-domain (a single shared `Q=0`; do not introduce an
adjustable cross-domain tolerance).

## Testing

- **Offline units:** cross-domain `ground_edge` (Segment for a deterministic
  cross-domain pair, Gap for a jittery one) using fixtures derived from the
  verified 20-run offsets; the within-domain regression (byte-identical output
  on the merged fixtures, proving the gate removal changed nothing within-domain).
- **HW acceptance gate:** on real NPU1, the same-tile core<->memmod handoff and an
  intra-array cross-tile pair ground to their exact offsets (matching the
  measured -2 / +2 / +4); a shim *delivery* event stays a Gap; the suite reaches
  a terminal state; the within-domain segments are unchanged.

## Out of scope (deferred, named follow-ons)

- **Shim NoC-egress cycle-accuracy** (rate-aware handling of the async CDC) -- may
  be irreducibly Q>0; not promised.
- **Optional cross-domain report flag** (facts-schema enhancement) -- deferred.
- **Re-examining the within-domain spec's C1 wording** ("timer resets per
  (pkt_type,row,col)" implied separate origins; the timers are separate *counters*
  with a *common* origin). A docs follow-up, not code.
- **Mode A as a deliberate probe** -- the timebase sync is already Mode A; a future
  effort may *vary* the reset config to probe behavior (the config-fidelity
  interest), but that is exploration, not needed for grounding.

## Correctness principle

Domain classification remains the per-module key `(col,row,pkt_type)`, but
cross-domain offsets are groundable because the array shares one broadcast-synced
timebase (verified, not assumed). The deterministic/jitter discrimination is the
measured exact-agreement test (`range <= Q`, Q=0, single shared comparator); the
measured offset is the ground truth (no hop prediction, no hardcoded constant).
Scope is the single AIE-ML clock domain; NoC-egress naturally gaps. DERIVE FROM
THE TOOLCHAIN -- and from the silicon's own measured behavior -- throughout.
