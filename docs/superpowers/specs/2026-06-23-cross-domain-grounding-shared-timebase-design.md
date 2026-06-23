# Cross-Domain Causal Grounding via Skew Decomposition -- Design

**Status:** approved model (brainstorm + 2 toolchain investigations + 3 rounds
adversarial review + direct HW-data verification + a skew-derivation
investigation, 2026-06-23). This is the **third** form of this spec; the first two
(co-observed-anchor, then naive "drop the same_domain gate") were each falsified
by review + data and corrected. Supersedes both. The next #140 frontier after the
within-domain jitter-grounding rule
(`2026-06-23-explicit-jitter-robust-grounding-design.md`, merged 2026-06-23).
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Evidence:** `build/experiments/gap140/nondeterminism/add_one_using_dma/`
(20 HW runs); compiled `aie_traced.mlir`; the verification tables herein.

## The discovery (the verified foundation)

The trace timebase is **already synchronized, cycle-exact, across the AIE-ML
array.** mlir-aie's trace lowering compiles `Timer_Control.Reset_Event =
BROADCAST_15` into every traced array tile (verified in `aie_traced.mlir`: core
0x34000 and memmod 0x14000 both = 0x7A00 = Reset_Event 122 = BROADCAST_15; shim
= 0x7F00 = USER_EVENT_1, the generator). Our capture pipeline leaves this live
(`trace_capture.py` patches only the trace `mode`, never `start_event`). Across
20 HW runs, first-occurrence `soc` offsets:

| pair | offset | range / 20 runs |
|------|--------|-----------------|
| core(1,2) <-> memmod(1,2) -- same-tile, cross-module | -2 | **0** |
| core(1,2) <-> memtile(1,1) -- cross-tile | +2 | **0** |
| memmod(1,2) <-> memtile(1,1) -- cross-tile | +4 | **0** |

Additively consistent (`2 - 4 = -2`), dead-stable while absolute timestamps swing
~14,000 cycles run-to-run. (This corrects the original within-domain spike's
"cross-domain timer skew" conclusion -- the timers are synced; what that spike saw
was workload jitter on a synced timebase.)

## The base-fact decomposition

For an event `x` in timer domain A and `y` in domain B, on this shared timebase:

```
offset(x,y) = soc_A(x) - soc_B(y)
            = [wall(x) - wall(y)]          +  [origin_B - origin_A]
            =   causal latency (x->y)       +     skew(A,B)
```

`skew(A,B) = origin_B - origin_A` is the difference in *when each domain's timer
was reset by BROADCAST_15* -- a fixed per-domain-pair propagation constant
(verified: the `-2/+2/+4` above), entirely separable from the causal term by
subtraction. Within one domain `skew = 0`, so offset *is* causal latency (why the
within-domain rule is sound). Cross-domain, the raw offset is **skew-polluted**;
grounding it directly would report propagation skew indistinguishably from causal
latency (the FATAL flaw the third review round found). The fix is to **derive the
skew and subtract it**, recovering the causal latency.

## The skew model: derive the structure, calibrate the constants, validate

A skew-derivation investigation established that the skew is *partly* derivable:

- **Inter-tile hop delay** is structurally modelable as ~1 cycle per row/col hop of
  the broadcast flood (corroborated by our `bcast_bridge.cpp` / `effects.rs`
  models -- but NOT by an independent aie-rt/AM025/AM020 timing constant; "1
  cycle/hop" is a model invariant, not a documented HW figure).
- **Intra-tile module delay** (the `-2` between a tile's core and mem modules) is
  **undocumented in every source** -- a silicon micro-constant (differing pipeline
  depths for the two modules' broadcast inputs). It is *not derivable*; it must be
  calibrated empirically.

So skew handling is a **structural model with empirically-calibrated micro-constants,
validated against silicon** (Maya's decision, the honest form of "derive + validate"):

```
origin_delay(col, row, module) = (row - shim_row) * HOP + intra_module_offset(module)
skew(A, B) = origin_delay(B) - origin_delay(A)
```

- The **form** is derived from the broadcast network (flood topology + per-module
  reset path).
- `HOP` and `intra_module_offset(core|mem|memtile)` are **calibrated** against the
  measured per-domain-pair skew (the stable additive component of cross-domain
  offsets; cleanest from near-broadcast reference events). They are few; the
  measured skews over-determine them.
- **Validation:** the fitted model must predict the measured skew with ~0 residual
  across the observed domain-pairs, and generalize to additional tile-pairs /
  kernels. A non-trivial residual falsifies the model (do not tune to hide it).
- The constants live in a small data-driven table (`skew_model`), with the shim
  broadcast source taken from the trace config. The model is derived/validated,
  never a per-kernel hardcode.

The feasibility/calibration spike (plan step 1) establishes `HOP` and the
`intra_module_offset` table and confirms the residual is ~0 before the grounding
change is written.

## Grounding design: a cross-domain Segment is a causal-latency Segment

Remove the `same_domain` gate in `ground_edge` (its sole use; verified). For a
cross-domain edge, compute the raw exact offset (via `offset_exact`, which already
works cross-domain -- the anchor cancels in the pairwise difference), look up
`skew(A,B)` from the model, and return a Segment whose **offset is the causal
latency** `raw - skew`, with the skew recorded as provenance:

```
raw  = offset_exact(run_dirs, child, parent, anchor)   # exact, range<=Q
skew = skew_model.skew(domain(parent), domain(child))   # derived+calibrated constant
Segment(child, parent, offset = raw - skew, skew = skew, raw = raw)   # offset is CAUSAL
```

This makes a cross-domain Segment **uniform with a within-domain Segment**: both
mean "child occurs `offset` causal cycles after parent" (within-domain `skew=0`,
so `offset = raw`). That uniformity dissolves the mechanical bugs the third round
found, rather than patching each:

- **`check_ordering`** (the FATAL: it rejected the raw `-2`) now operates on the
  causal offset (`raw - skew`), which is `>= 0` for a correctly-oriented edge.
  Within-domain behavior is unchanged (`skew=0`). No skew-pollution rejections.
  (It consumes the *derived* skew model, not a grounding product -- no circular
  dependency.)
- **`assemble`/Timeline** is monotonic because it accumulates causal offsets, not
  skew-polluted raw offsets.
- **No negative-offset sign hazard** in `check_additivity` or consumers: causal
  offsets compose additively and are non-negative for causal chains.
- **Byte-identical reproduction is preserved**: `raw = offset + skew` is
  recoverable from the Segment's provenance, so the emulator still has the exact
  trace timestamp to match.

### Provable within-domain regression safety

For within-domain edges, `same_domain` was true and `skew=0`, so the new path
(`offset = raw - 0 = raw`) is byte-identical to today's `same_domain` branch. A
regression test on the merged within-domain fixtures (no skew) asserts identical
output.

## The async-CDC guard (an existing bug this surfaced)

The third review found that a NoC-egress event already false-grounds **same-domain
today**: `DMA_S2MM_0_FINISHED_TASK <- DMA_MM2S_0_FINISHED_TASK` (shim, both
`1|0|2`) grounds as a Segment with offset 0. These shim DMA-to-DDR completion
events fire after a round-trip across the async 1 GHz<->960 MHz NoC FIFO (AM020
CDC); their timing carries non-deterministic synchronizer latency and is **not**
a cycle-accurate causal fact, even though a 20-run sample happened to show range
0. This is gate-independent and must be fixed regardless: flag known async-CDC
events (NoC-egress `DMA_*_FINISHED` on the shim) as **gap-only** -- they never
ground as Segments. The flagged set is derived from the event semantics (shim DMA
completion crossing to DDR), not hardcoded per kernel.

## Scope

- **In scope (cycle-exact, shared AIE-ML timebase):** causal grounding across core,
  memory module, mem-tile, and shim *control* events -- all 1 GHz, BROADCAST_15
  (shim: USER_EVENT_1) reset, skew modeled.
- **Out (gap-only):** NoC-egress *timing* (shim DMA-to-DDR completion) -- async CDC,
  not cycle-deterministic.

## Q -- the measurement floor

`Q = 0`, the single shared comparator (verifier.py), used cross-domain unchanged.
The raw cross-domain offset agrees exactly (range 0, verified); after subtracting
the constant skew the causal latency is equally exact. A non-zero cross-domain `Q`
is a falsification (diagnose: genuine jitter, async-CDC, decoder-frame artifact),
never an adjustable tolerance.

## Facts & reporting

A Segment's `offset` is the causal latency in all cases (within- and cross-domain),
keeping `kind="segment"` and the existing accessors' meaning intact. The skew and
raw value are Segment provenance (for reproduction + audit). This is a **minimal,
bounded** addition -- a Segment gains two numeric provenance fields, not the
deferred full structured-facts migration; the report projection
(`engine.py`/`run_experiment.py`) is extended to surface them. Cross-domain
Segments are distinguishable by `skew != 0`.

## Connection to the emulator (downstream value, not this plan)

The emulator's broadcast model (`src/device/events/broadcast.rs`) is currently
*atomic* -- it does not reproduce the `-2/+2/+4` skew. The skew model's constants
are exactly the ground-truth targets for making the emulator's broadcast timing
cycle-accurate, which is a real step toward byte-identical traces. That Rust work
is a named follow-on, not part of this (Python inference) plan.

## Testing

- **Offline units:** the skew model (`origin_delay`/`skew` predict the measured
  `-2/+2/+4`); cross-domain `ground_edge` (Segment with causal offset = raw-skew
  for a deterministic pair, Gap for a jittery one) on fixtures from the verified
  20-run data; `check_ordering` on a cross-domain edge with raw offset `-2`,
  skew `-2` -> causal `0` -> accepted (the regression of the FATAL bug); the
  within-domain byte-identical regression; the async-CDC gap-only guard; the two
  existing cross-domain-`->`-Gap tests inverted to cross-domain-`->`-Segment.
- **HW acceptance gate:** on real NPU1, the same-tile core<->memmod and an
  intra-array cross-tile pair ground to causal offsets after skew subtraction; the
  skew model's residual is ~0; a shim NoC-egress event stays a Gap; within-domain
  segments unchanged; suite reaches terminal state.

## Plan sequence (one serious thing, with a calibration gate)

1. **Skew-model calibration spike (gate):** from the 20-run data (+ a cheap
   confirming run if needed), fit `HOP` + `intra_module_offset`, validate residual
   ~0 and additive consistency; record the constants. If the model can't predict
   the measured skew, stop and diagnose.
2. **Implement** `skew_model`, the `ground_edge` cross-domain branch (causal offset
   + provenance), the `check_ordering` skew-correction, the async-CDC gap-only
   guard, facts/report provenance, and the test changes.
3. **Validate** offline + HW gate.

## Out of scope (deferred, named follow-ons)

- **Per-hop broadcast delay in the Rust emulator** (make `broadcast.rs`
  cycle-accurate to reproduce the skew) -- the downstream byte-identical step.
- **Distant-tile / multi-column / loop-kernel skew generalization** beyond the
  calibration set -- validated incrementally as more kernels are traced.
- **NoC-egress causal timing** (rate-aware async-CDC handling) -- likely
  irreducibly Q>0; not promised.
- **Full structured facts-schema migration** beyond the two skew/raw provenance
  fields.

## Correctness principle

Cross-domain offsets are groundable because the array shares one BROADCAST_15-synced
timebase (verified). A Segment's offset is always a causal latency: within-domain
directly, cross-domain after subtracting a skew whose *structure* is derived from
the broadcast network and whose *micro-constants* are calibrated against and
validated by measured silicon skew (residual ~0, never tuned to pass). The
exact-agreement test (`range <= Q`, single shared Q=0) is unchanged; async-CDC
events are gap-only by event semantics. DERIVE FROM THE TOOLCHAIN -- structure from
the toolchain, the few undocumented micro-constants from silicon ground truth --
throughout.
