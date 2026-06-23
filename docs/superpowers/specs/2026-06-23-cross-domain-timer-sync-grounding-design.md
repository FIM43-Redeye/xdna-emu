# Cross-Domain Timer-Sync Grounding -- Design

**Status:** approved model (brainstorm + toolchain investigation, 2026-06-23).
The next #140 frontier after the within-domain jitter-grounding rule
(`docs/superpowers/specs/2026-06-23-explicit-jitter-robust-grounding-design.md`,
merged 2026-06-23).
**Issue:** #140 (byte-identical emulator/HW trace reports).
**Evidence:** toolchain investigation (this doc's "Investigation" section); a
feasibility HW spike is the gate before the implementation plan (see "Feasibility
gate").

## Goal

The within-domain grounding rule grounds cycle-accurate offsets only *within* a
single per-module timer domain `(col, row, pkt_type)`; every cross-domain edge
(core compute -> shim DMA out, memory-module release -> core acquire, etc.) is a
named Gap because each module's trace timer runs on its own origin. This design
extends grounding to **cross-domain offsets**: where a shared timebase can be
established, a cross-domain edge becomes an exact **Segment** instead of a Gap.

The lever is the AIE2 event-broadcast network: a single broadcast event observed
in two domains pins both to a common instant, and the broadcast's deterministic
propagation delay is the only residual -- a derivable constant, not jitter.

No statistical inference: exact cross-run agreement (equality, `range <= Q`), the
same discriminator the within-domain rule uses. No median, no MAD, no tuned
tolerance.

## Investigation (this is the design's measured foundation)

A toolchain investigation (aie-rt, AM025 register DB, mlir-aie, our own trees)
established the physics this design rests on:

- **The HW timer-sync mechanism is real.** `Timer_Control.Reset_Event`
  (bits 14:8) selects an event whose firing resets a module's timer to zero.
  `Event_Broadcast0..15` registers broadcast an event to the whole array; the
  standard pattern (mlir-aie `AIEInsertTraceFlows.cpp` Phase 4b/4f) wires a shim
  `USER_EVENT_1` -> `Event_Broadcast15` -> every traced tile's
  `Timer_Control.Reset_Event = BROADCAST_15`, resetting all timers from one
  event to a common t=0.
- **Our pipeline deliberately patches it out.** `tools/trace-patch-events.py`
  replaces the broadcast-driven trace start with kernel-internal events "to
  eliminate the broadcast-latency component of entry/exit jitter." That choice --
  not any hardware limit -- is why our cross-domain timestamps currently sit on
  independent per-tile origins (spike run 07's +3000-cycle cross-domain skew).
- **Our decoder retains per-tile absolute timer values.** We read
  `StartCmd.timer_value` and compute `soc` (drift-corrected cycle-of-signal);
  mlir-aie's decoder disables `timer_value` entirely (`# TODO ... sync this
  between trace types and row,col`). We are ahead here -- we keep what we need to
  align.
- **Broadcast propagation is DETERMINISTIC: exactly 1 cycle per hop.** Confirmed
  by our own HW-validated model (`aiesim-bridge/src/bcast_bridge.cpp`: "the
  broadcast ripples up one row per posedge -- matching real-HW propagation
  timing. One-cycle latency = one broadcast hop = faithful"), by mlir-aie's own
  note (`trace/__init__.py:165`: "a few clock cycles between tiles", which it
  declines to compensate), and by the network being dedicated single-bit wiring,
  physically separate from the stream-switch data network -- no backpressure, no
  contention, no data dependence.
- **Hop count is a fixed function of the configured routing.** Broadcast floods
  in four cardinal directions, gated per-channel per-direction by
  `Event_Broadcast_Block_{North,South,East,West}_*` masks. The hop distance from
  the sync source to any domain is therefore derivable from the routing config +
  array topology (default masks -> Manhattan shortest path).

**The decisive consequence (the math).** If two domains share one broadcast
reference, the run-to-run jitter cancels. For an event `e` in domain `D` with raw
timestamp `soc_D(e)` measured from `D`'s timer origin `origin_D`, and a sync
broadcast generated at instant `T_gen` arriving at `D` at `T_gen + hop_D`:

```
soc_D(e)      = wall(e) - origin_D
soc_D(bcast)  = (T_gen + hop_D) - origin_D
=> wall(e) - T_gen = soc_D(e) - soc_D(bcast) + hop_D
```

Define the **global timestamp** `gts(e) = soc_D(e) + align_D`. Then `origin_D`
(each domain's jittery reset) and `T_gen` (the jittery host trigger) both cancel,
leaving only `align_D` -- a per-domain quantity whose only run-independent
residual is `hop_D`, the derivable propagation constant. A cross-domain offset
`gts(child) - gts(parent)` is therefore exact up to derivable integers, with all
jitter cancelled.

## The alignment model & the pluggable seam

The entire design reduces to how `align_D` is computed, and that is the seam that
makes mode A (HW timer-reset) a later config swap rather than a rewrite:

| Mode | `align_D` | Per-run input |
|------|-----------|---------------|
| **Co-observed anchor (this plan)** | `hop_D - soc_D(sync_broadcast)` | the sync broadcast's first-occurrence `soc` in `D` |
| **HW timer-reset (later plan, mode A)** | `hop_D` | none -- the reset already folds `T_gen + hop_D` into `origin_D` |

Both modes feed the **same** `DomainAlignment` interface (`align(domain, run) ->
int`). The grounding rule consumes `align_D` and is agnostic to which mode
produced it. Re-enabling the HW reset later means adding a second provider; the
grounding/reporting math does not change.

**Why mode A matters later (config-fidelity probing).** Re-enabling the literal
HW timer-reset lets us probe how the silicon's own synchronization behaves
(broadcast trigger gating, reset propagation), which is a first-class goal --
hence the seam is built now even though the provider is deferred.

**Regression safety.** Within-domain grounding is the degenerate case: both
endpoints share `D`, so `align_D` cancels identically and `hop` is zero -- it
reduces to today's raw-`soc` offset. The existing within-domain path keeps using
raw `soc` with no broadcast dependency; cross-domain is a *new branch* taken only
when `same_domain` is false. The unification is conceptual; the working path is
not rewritten.

**Built-in falsifier.** A cross-domain lock handoff (memory-module `LOCK_RELEASE`
-> core `LOCK_ACQUIRE`) is causally near-instantaneous on silicon, so once
aligned it must ground to a *small fixed* latency across runs. This sharpens the
existing `check_lock_handoff` inequality (`release.ts <= acquire.ts`) into an
exact cross-domain handoff-latency check, and is an independent on-silicon
validation of the whole alignment model.

## Components (tight scope)

New code is additive; the existing within-domain path is untouched.

### 1. Sync-broadcast instrumentation (`tools/trace-patch-events.py`, `tools/trace_capture.py`)

The data-production prerequisite. Generate one sync broadcast (a `USER_EVENT`
wired to `Event_BroadcastN`, default flood routing) early in the run, and add the
corresponding `BROADCAST_N` event to the trace-event menu of each domain to be
cross-aligned. Costs one trace slot per domain; fires once -> clean
first-occurrence anchor. Purely additive: it does **not** touch the timer-reset
config (that is mode A, later).

### 2. Hop-delay derivation (`tools/inference/broadcast_routing.py`, new)

`hop_delay(source_tile, domain_tile) -> int`: BFS the broadcast flood over the
array topology (`tools/aie-device-models.json`) honoring the configured
`Event_Broadcast_Block_*` masks, returning hop count. Default masks -> Manhattan
shortest path; non-default -> derived from the actual routing. Mirrors the flood
semantics already proven in `aiesim-bridge/src/bcast_bridge.cpp` and
`src/device/events/broadcast.rs`. This is the "derive from the toolchain" core
and the basis for config-fidelity probing.

### 3. Alignment provider (`tools/inference/alignment.py`, new)

The seam. `DomainAlignment` interface -> `align(domain, run_dir) -> int`.
`AnchorEventAlignment` (this plan): `hop_delay(src, D) - soc_D(sync_broadcast
first occurrence)`. A documented `ResetAlignment` stub records the mode-A
computation (`hop_delay` only) so the later swap is obvious and the interface is
proven to fit both.

### 4. Grounding extension (`tools/inference/grounding.py`)

`ground_edge`: if `same_domain` -> today's raw-`soc` path (unchanged). Else ->
cross-domain path: compute the `gts`-offset per run via the alignment provider,
apply the exact-agreement test (`range <= Q`) -> Segment carrying alignment
provenance (sync broadcast used, `hop_delD - hop_delP`, per-run `align` values
for audit); else Gap. Within-domain Segments carry empty provenance -- same
`kind="segment"`, no new kind string.

### 5. Reporting + falsifier (`tools/inference/run_experiment.py`, `tools/inference/verifier.py`)

Cross-domain Segments report alongside within-domain ones, provenance attached.
`check_lock_handoff` sharpens to the exact cross-domain handoff-latency check.

### Data flow (additions in **bold**)

```
loop -> run_experiment -> engine (chainer -> rules -> grounding/verifier) -> report
                                                          |
                          **alignment.py <- broadcast_routing.py**
                          (consumed only on the cross-domain branch)
```

## Q -- the measurement floor

`Q = 0`, the same measured within-domain floor, carried to cross-domain by
construction: once `align_D` cancels `origin_D` and `T_gen` and `hop_D` removes
the propagation constant, a deterministic cross-domain relationship has zero
cross-run delta. The feasibility spike confirms `Q = 0` cross-domain empirically
before the plan is written; if a kernel ever exposes a genuine discrete quantum
it is documented as that quantum, never tuned to pass a test.

## Feasibility gate (spike before the plan)

This design has a single load-bearing empirical claim: the cross-domain
`gts`-offset agrees exactly across runs once hop-corrected. It earns its plan the
same way the within-domain rule did -- a cheap HW spike (~20 runs):

1. Generate a sync broadcast; record it as a trace event in two domains.
2. Per run, compute `align_D` for each domain and the cross-domain `gts`-offset
   of a known causal edge (the cross-domain lock handoff is the ideal target).
3. Confirm `range = 0` across runs (the handoff lands on a small fixed latency).

**If it holds**, the model is proven on silicon and the plan is written against
the measured results. **If not**, we diagnose before committing implementation
(propagation jitter? broadcast not recorded in both domains? hop-delta wrong?).
The spec is updated with the spike's measured outcome before planning.

## Testing

- **Offline units:** `hop_delay` (flood BFS, default + blocked masks); `align`
  (anchor-event computation); cross-domain `ground_edge` (Segment when exact, Gap
  when not); provenance round-trip; the sharpened `check_lock_handoff`.
- **HW acceptance gate:** the cross-domain lock handoff grounds exact on real
  NPU1; a within-domain regression check (the just-merged segments still ground
  identically, byte-for-byte); the suite reaches a terminal state. Same green-bar
  bar as the within-domain rule.

## Out of scope (deferred, named follow-ons -- one serious thing at a time)

- **Mode A: re-enabling the HW timer-reset** (`BROADCAST_15 ->
  Timer_Control.Reset_Event`). The seam and the `ResetAlignment` stub are built
  now; the live provider + config-fidelity probing are a later plan.
- **Per-hop broadcast delay in the Rust emulator.** The emulator currently floods
  broadcasts atomically (`src/device/events/broadcast.rs`,
  `effects.rs::propagate_broadcasts`); cycle-accurate per-hop delay there is a
  separate fidelity task.
- **Active low-frequency event selection** (observer-effect refinement).
- **Multi-column / multi-source broadcast topologies** beyond what the spike
  kernel exercises.

## Correctness principle

Domain classification is the per-module timer key `(col, row, pkt_type)`; the
deterministic/jitter discrimination is the measured exact-agreement test
(`range <= Q`); `align_D` cancels run jitter by construction; `hop_D` is derived
from the configured broadcast routing, never hardcoded. DERIVE FROM THE TOOLCHAIN
holds throughout.
