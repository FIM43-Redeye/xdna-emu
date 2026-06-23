# Cross-Domain Timer-Sync Grounding -- Design

**Status:** approved model (brainstorm + toolchain investigation + 4-lens
adversarial spec review, 2026-06-23). The next #140 frontier after the
within-domain jitter-grounding rule
(`docs/superpowers/specs/2026-06-23-explicit-jitter-robust-grounding-design.md`,
merged 2026-06-23). This is the **umbrella design**; it is delivered as a tiered
sequence of plans (see "Plan sequence"), of which **Tier 1 (same-tile
cross-module)** is the immediate target.
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Evidence:** toolchain investigation + adversarial review (this doc's
"Investigation" and "Adversarial review outcomes" sections). A feasibility HW
spike, gated on the instrumentation prerequisite plan, validates the model before
the Tier-1 grounding plan is written.

## Goal

The within-domain grounding rule grounds cycle-accurate offsets only *within* a
single per-module timer domain `(col, row, pkt_type)`; every cross-domain edge is
a named Gap because each module's trace timer runs on its own origin. This design
extends grounding to **cross-domain offsets** within the AIE-ML clock domain:
where a shared timebase can be established, a cross-domain edge becomes an exact
**Segment** instead of a Gap.

The lever is the AIE2 event-broadcast network: a single broadcast event observed
in two domains pins both to a common instant. No statistical inference -- exact
cross-run agreement (equality, `range <= Q`), the same discriminator the
within-domain rule uses. No median, no MAD, no outlier tolerance, no tuned
epsilon, **and no magnitude bounds masquerading as exactness** (see "Adversarial
review outcomes").

## Investigation (measured foundation, with honest confidence markers)

A toolchain investigation (aie-rt, AM025 register DB, mlir-aie, AM020, our own
trees) plus a 4-lens adversarial review established the following. Confidence is
marked because two originally-asserted "facts" did not survive scrutiny.

- **[VERIFIED] The HW timer-sync mechanism is real.** `Timer_Control.Reset_Event`
  (bits 14:8, confirmed in `aie_registers_aie2.json`: `bit_range [8,14]`,
  "Event No that will reset Internal Timer") selects an event whose firing resets
  a module's timer to zero. `Event_Broadcast0..15` broadcast an event array-wide.
  mlir-aie `AIEInsertTraceFlows.cpp` (Phase 4b/4f) wires shim `USER_EVENT_1` ->
  `Event_Broadcast15` -> each traced tile's `Timer_Control.Reset_Event =
  BROADCAST_15` (15 is the conventional default from `AIETraceOps.td`,
  configurable, not hardwired).
- **[VERIFIED] Our pipeline deliberately patches it out.**
  `tools/trace-patch-events.py` replaces the broadcast-driven trace start with
  kernel-internal events "to eliminate the broadcast-latency component of
  entry/exit jitter." That choice -- not a hardware limit -- is why our
  cross-domain timestamps currently sit on independent per-tile origins.
- **[VERIFIED] Our decoder retains per-tile absolute timer values.** We read
  `StartCmd.timer_value` and compute `soc` (drift-corrected cycle-of-signal).
  mlir-aie's main cross-tile decode path does *not* carry `timer_value` forward
  (`parse.py:228`: `timer = 0  # TODO ... sync this between trace types and
  row,col`). (Correction from an earlier draft: mlir-aie does not "disable"
  `timer_value` entirely -- it uses it in one path and zeroes it in the
  cross-tile path. The net effect -- no cross-tile sync -- is the same.)
- **[VERIFIED -- DECISIVE SCOPE CONSTRAINT] The AIE-ML array is a single clock
  domain; the NoC interface is an async crossing.** AM020 (chapter-1 Table 1):
  AIE-ML array clock 1 GHz; NoC clock 960 MHz, "**Asynchronous, clock domain
  crossing (CDC)**"; "clock domain crossing at the NoC interface tile."
  Consequence: two timers in the AIE-ML domain (core, memory, memtile, and the
  shim **PL-module timer**, all 1 GHz) reset to a common instant stay aligned for
  all elapsed time -- an additive `align_D` is valid. But any event whose
  timestamp reflects **NoC egress** (a shim DMA-to-DDR completion) crosses the
  1 GHz<->960 MHz async FIFO, whose CDC synchronizer latency is non-deterministic
  and cannot be removed by any additive (or even rate-corrected) integer. The
  +3000-cycle skew measured in the within-domain spike was a shim-vs-core
  *start-offset* (both AIE-ML domain, cancellable), NOT rate drift -- so
  intra-array shim<->core *control* events are safe; NoC-egress *timing* is not.
- **[MODELED, NOT HW-MEASURED] Broadcast per-hop propagation is deterministic,
  ~1 cycle/hop.** Our `aiesim-bridge/src/bcast_bridge.cpp` models one row per
  posedge and was HW-validated for *arrival* (the broadcast demonstrably reaches
  tile(0,2) on silicon with matching structural trace events) -- but arrival !=
  per-hop cycle count. mlir-aie's own note (`trace/__init__.py:165`) says "a few
  clock cycles between tiles," not "exactly one." **The exact per-hop constant is
  an open question, deferred to the cross-tile tier's spike. Tier 1 does not
  depend on it (see below).** The broadcast network has its own register space
  (`Event_Broadcast*`, `Event_Broadcast_Block_*`) entirely separate from the
  stream-switch; "no backpressure" is architecturally plausible but not cited
  from an available source -- the cross-tile tier must validate determinism
  empirically, not assume it.

**The decisive consequence (the math).** If two domains share one broadcast
reference, the run-to-run jitter cancels. For an event `e` in domain `D` with raw
timestamp `soc_D(e)` measured from `D`'s timer origin `origin_D`, and a sync
broadcast generated at instant `T_gen` arriving at `D` at `T_gen + hop_D`:

```
soc_D(e)     = wall(e) - origin_D
soc_D(bcast) = (T_gen + hop_D) - origin_D
```

Define the **global timestamp** `gts(e) = soc_D(e) + align_D` with
`align_D = hop_D - soc_D(bcast)`. Then:

```
gts(e) = soc_D(e) + hop_D - soc_D(bcast) = wall(e) - T_gen
```

`origin_D` (each domain's jittery reset) and `T_gen` (the jittery host trigger)
both cancel. **`hop_D` does NOT cancel -- it corrects the propagation delay
already baked into the measured `soc_D(bcast)`** (correction: an adversary claimed
it cancels and that hop-routing is therefore unnecessary; the algebra shows
omitting `hop_D` leaves a per-domain `-hop_D` residual, so it genuinely matters
for cross-tile pairs). A cross-domain offset `gts(child) - gts(parent)` equals
the true `wall(child) - wall(parent)` -- jitter-free -- **provided both domains
share the AIE-ML clock and the same `T_gen`**.

## Why Tier 1 (same-tile cross-module) is the tractable first target

The core and memory modules of one tile share `(col, row)` but have distinct
`pkt_type` -> distinct timer domains (C1), yet they are co-located:

- **`hop_D` is equal for both -> `Delta hop = 0`.** The broadcast reaches both
  modules of a tile at the same hop count, so the propagation correction drops
  out entirely. **Tier 1 needs no `broadcast_routing.py`** -- not because
  `hop_D` cancels, but because the two endpoints are at the same tile.
- **Both are in the AIE-ML clock domain.** No NoC crossing, no rate drift, no CDC
  jitter. The model's additive-`align_D` premise holds exactly.
- **It carries a free independent guard.** `Delta hop = 0` predicts
  `soc_core(bcast) == soc_mem(bcast)` within `Q`. That equality is an
  *independent* generation-identity check (it does not use the lock handoff it
  helps validate), catching the broadcast-generation-mismatch *bias* that the
  `range <= Q` jitter test structurally cannot (a consistent-but-wrong anchor
  passes a jitter test).

The Tier-1 deliverable -- the cross-domain **lock handoff** (memory-module
`LOCK_REL` -> core `INSTR_LOCK_ACQUIRE_REQ`) grounding to a *fixed* latency --
is therefore on the safe side of every flaw the review found.

## The alignment model & the pluggable seam

`align_D` is the seam that makes the later HW-timer-reset mode a config swap:

| Mode | `align_D` | Per-run input | Tier |
|------|-----------|---------------|------|
| **Co-observed anchor** | `hop_D - soc_D(sync_broadcast)` (Tier 1: `hop_D` term is 0) | the sync broadcast's first-occurrence `soc` in `D` | 1 (this), 2 |
| **HW timer-reset (mode A)** | `hop_D` | none -- the reset folds `T_gen + hop_D` into `origin_D` | deferred |

The grounding rule consumes `align_D` via a thin function and is agnostic to how
it was produced. **We do NOT build a `DomainAlignment` interface or a
`ResetAlignment` stub now** (YAGNI -- one real implementor; a stub cannot prove
interface fit, and the modes have different input arity). We build the
`align_D(...)` function for the anchor mode and keep a one-paragraph note here on
how mode A plugs in, so re-enabling it later (for the config-fidelity probing
that motivates it) is a localized addition, not a rewrite.

**Regression safety (hard requirement, not "conceptual unification").** The
existing within-domain path stays byte-for-byte: `ground_edge`'s `same_domain`
branch is the current code with **zero new parameters reaching it**. The
cross-domain branch is a separate function (`ground_cross_domain(...)`) that
`ground_edge` dispatches to only when `same_domain` is false. The "unification is
conceptual" framing from the first draft is removed -- it invited a shared-`gts`
refactor that would route within-domain edges through `align_D` and introduce a
broadcast dependency, regressing the merged 22-cycle segment. The regression test
runs on a fixture **with no sync broadcast** to prove no broadcast dependency
leaked into the within-domain path.

## Adversarial review outcomes (what changed, and why)

The 4-lens review (algebra/physics, toolchain facts, feasibility, integration)
reshaped this spec. Resolved here:

1. **Clock-domain (FATAL):** scoped out NoC-egress timing; Tier 1 is intra-array.
2. **`hop_D` does not cancel (corrected an adversary):** retained as the
   cross-tile correction; Tier 1 sidesteps it via `Delta hop = 0`.
3. **Per-hop = 1 is asserted, not measured:** demoted to [MODELED]; the cross-tile
   tier's spike must measure the constant (two edges, differing hop-deltas), not
   assume it.
4. **Generation-identity bias:** added a one-shot broadcast guarantee + the
   same-tile consistency check; a wrong-but-consistent anchor is *rejected*, not
   silently grounded.
5. **"Small fixed latency" was a tolerance:** purged. The handoff latency is
   asserted *fixed across runs* (`range <= Q`); no magnitude bound appears.
6. **`Q = 0` "by construction" was wrong cross-domain:** it is empirical (two
   counters); the spike confirms it. A non-zero cross-domain `Q` is a
   *falsification of the model* (diagnose the residual), never a documented
   constant to pass the gate.
7. **`check_lock_handoff` is dead code** (only `check_ordering` is wired into the
   live path) and sharpening it to consume `align_D` would make it non-independent
   (a falsifier that uses the alignment it validates checks self-consistency, not
   correctness). It is wired in honestly; the *independent* validations are the
   same-tile consistency check and the within-domain byte-for-byte regression.
8. **Scope was >=3 plans:** split (see below).
9. **soc-degeneracy:** the sync broadcast must be a *dedicated* broadcast fired
   *after* trace start, not the trace-start trigger itself (else `soc_D(bcast)`
   collapses to 0 and the anchor silently becomes mode A).
10. **Slot budget / memmod at 7/8:** the instrumentation plan owns the batching
    plan so the broadcast + cross-domain lock events co-trace without overrun.

## Plan sequence (tiered; one serious thing per plan)

1. **Prerequisite plan -- sync-broadcast instrumentation.** Generate a dedicated
   one-shot sync broadcast (a `USER_EVENT` -> `Event_BroadcastN` on a channel
   *not* used for trace start, default flood routing) and record `BROADCAST_N` as
   a trace event in the relevant domains. Owns: the slot-budget/batching plan; the
   memmod lock-event characterization (identify the `add_one` lock number, add
   `MEM_LOCK_*` events); and any new injector tooling (`trace-patch-events.py` is
   rewrite-only -- generating a fresh broadcast may require an `inject-write32`
   tool analogous to `inject-maskpoll.py`, unless reusing a compiled broadcast).
   Mirrors how the instruction-event layer preceded the within-domain rule.
2. **Feasibility spike (gate)**, on the instrumentation output: ~20 HW runs;
   confirm (a) the same-tile consistency check `soc_core(bcast) == soc_mem(bcast)`
   holds (`range <= Q`), and (b) the cross-domain lock handoff grounds to a fixed
   latency (`range = 0`). If it holds, write Tier 1; if not, diagnose before
   committing. The spec is updated with the measured outcome.
3. **Tier 1 plan -- same-tile cross-module grounding** (this design's immediate
   scope): the lock handoff, co-observed anchor, no routing, AIE-ML clock only.
4. **Later tiers (named follow-ons):** cross-tile AIE-ML grounding (builds
   `broadcast_routing.py` + measures the per-hop constant); NoC-egress (needs a
   rate-aware mode, possibly infeasible due to CDC jitter); mode A HW timer-reset
   (config-fidelity probing).

## Components (Tier 1 scope)

New code is additive; the within-domain path is untouched. (Instrumentation lives
in the prerequisite plan, not here.)

### 1. Alignment function (`tools/inference/alignment.py`, new)

`align_D(domain, run_dir) -> int = -soc_D(sync_broadcast first occurrence)`
(the `hop_D` term is omitted for Tier 1: same-tile pairs share an equal `hop_D`,
so it is common to both domains and cancels in the cross-domain offset -- adding
it back is the documented cross-tile extension point). Plus `broadcast_consistent(run_dir) -> bool`: the
same-tile `soc_core(bcast) == soc_mem(bcast)` within `Q` generation-identity
guard. A one-paragraph note documents the mode-A computation.

### 2. Grounding extension (`tools/inference/grounding.py`)

`ground_edge`: `same_domain` -> today's raw-`soc` path (current code, no new
params). Else -> `ground_cross_domain(...)`: require `broadcast_consistent`;
compute the `gts`-offset per run via `align_D`; exact-agreement (`range <= Q`) ->
Segment (cross-domain), else Gap. Cross-domain Segments are flagged minimally (a
boolean + the anchor broadcast key) -- NOT a structured-provenance payload (that
is the deferred facts-schema migration; do not smuggle it in).

### 3. Facts + reporting (`facts.py`, `engine.py`, `run_experiment.py`)

The minimal cross-domain flag rides the existing 4-arg `derives` shape
(`kind="segment"`, no new kind string). The report projection in `engine.py` and
`run_experiment.py` is extended to surface the cross-domain flag -- a known,
in-scope edit to the report schema (the first draft wrongly implied a free
`args[4]` slot the projection would carry automatically).

### 4. Falsifier (`tools/inference/verifier.py` + live wiring)

Wire `check_lock_handoff` into the live verification path (today only
`check_ordering` runs). For the cross-domain handoff it asserts the aligned
release->acquire latency is *fixed* across runs (`range <= Q`), no magnitude
bound. Independence is provided by `broadcast_consistent` + the within-domain
regression, not by the handoff check itself.

### Data flow (additions in **bold**)

```
loop -> run_experiment -> engine (chainer -> rules -> grounding/verifier) -> report
                                                          |
                          **alignment.py** (consumed only on the cross-domain branch)
```

## Q -- the measurement floor

`Q = 0`, the within-domain floor, carried to cross-domain **empirically** (two
physical counters, not one -- so not "by construction"). The feasibility spike
confirms `range = 0` for the same-tile handoff and the consistency check before
Tier 1 is written. A non-zero cross-domain `Q` falsifies the model (diagnose the
residual: generation mismatch, a real sub-cycle phase term, decoder-frame
interaction); it is never adopted as a tuned constant to pass the gate.

## Testing

- **Offline units:** `align_D` (anchor computation, same-tile `hop=0`);
  `broadcast_consistent` (passes equal, rejects mismatched); cross-domain
  `ground_edge` (Segment when exact + consistent, Gap otherwise); the
  within-domain regression on a no-broadcast fixture (proves no broadcast
  dependency leaked); `check_lock_handoff` fixed-latency assertion.
- **HW acceptance gate:** the cross-domain lock handoff grounds to a fixed
  latency on real NPU1; the same-tile consistency check holds; the just-merged
  within-domain segments still ground identically, byte-for-byte; the suite
  reaches a terminal state.

## Out of scope (deferred, named follow-ons -- one serious thing at a time)

- **Cross-tile AIE-ML grounding** (`broadcast_routing.py` BFS over
  `Event_Broadcast_Block_*` masks + topology; spike to measure the per-hop
  constant). The `hop_D` term and routing live here.
- **NoC-egress timing** (shim DMA-to-DDR completion): crosses the async NoC CDC;
  needs a rate-aware mode and may be irreducibly Q>0. Possibly infeasible;
  explicitly *not promised*.
- **Mode A: HW timer-reset** (`BROADCAST_15 -> Timer_Control.Reset_Event`) for
  config-fidelity probing. The `align_D` seam is built to accept it.
- **Per-hop broadcast delay in the Rust emulator** (currently atomic flood in
  `src/device/events/broadcast.rs`): a separate fidelity task.
- **Active low-frequency event selection; full structured facts-schema
  migration.**

## Correctness principle

Domain classification is the per-module timer key `(col, row, pkt_type)`; the
deterministic/jitter discrimination is the measured exact-agreement test
(`range <= Q`); `align_D` cancels run jitter by construction; cross-domain
grounding is scoped to the single AIE-ML clock domain; no magnitude bound or
tuned tolerance is used anywhere; `hop_D` (when the cross-tile tier needs it) is
derived from the configured broadcast routing, never hardcoded. DERIVE FROM THE
TOOLCHAIN holds throughout.
