# Faithful Timer-Sync: Cross-Domain Skew via Broadcast-Propagation Modeling

**Arc design / umbrella spec.** 2026-06-28.

This document is the governing plan for modeling AIE2 (NPU1 / Phoenix) per-tile
timer synchronization faithfully, so the trace inference engine can ground
cross-column causal edges with real reproduction offsets. It supersedes the
sequencing and gate design of the parked route-1 spec
([`2026-06-27-timer-sync-route1-emulator-forward-model-design.md`](2026-06-27-timer-sync-route1-emulator-forward-model-design.md))
and folds in its round-3 adversarial review (commit `bddaf514`). It decomposes
the work into five sub-projects (SP-1..SP-5); each gets its own design + plan +
implementation cycle. We continue from this write-up.

**Why an umbrella doc now:** this is plausibly *the* last major Phoenix march.
The campaign is large (four-to-five sub-projects with real dependency
structure), it revises a parked design rather than starting fresh, and its
foundation rests on findings that were imprecise in the canonical docs until the
2026-06-28 deep dive. It deserves a document we can stand on.

---

## 1. Why this matters -- the critical path

The trace inference engine's cross-column edge **grounding** is the immediate
open frontier of #140. Today the engine attaches only the *raw* fused offset to
a cross-column coupling -- `raw = Delta_wall + skew` -- and flags it as a
cross-domain `Gap`, never a grounded causal segment. It cannot **decompose**
that raw offset into the true parent->child latency (`Delta_wall`), because the
two columns' trace timers live in different timer domains with unknown skew. The
standing rule (`cross-domain-skew-limit.md` Sec.9) is therefore "never emit a
cross-domain offset *as* a causal cycle."

**Decision (Maya, 2026-06-28): cross-column edges must carry the decomposed
causal offset (`Delta_wall`), not merely exist.** That makes the cross-domain
skew a required quantity, which is what this arc produces -- and it **amends
skew-limit Sec.9**: a modeled, validated skew table lets a cross-domain coupling
ground as a real causal segment for the first time, alongside (not replacing)
the existing raw `reproduction_offset`.

The dependency chain this unblocks:

```
faithful timer-sync (this arc)
  -> cross-column couplings ground a decomposed causal offset (Delta_wall)
  -> inference engine cross-column grounding closes
  -> #140 "done"
  -> framework tenants 4 (locks/streams) + 5 (multi-tile) unpark
  -> clean_release(Aie2) goes green
  -> Phoenix retirement is reachable
```

Everything funnels through here. See
[[project_inference_engine_brainstorm_inflight]],
[[project_framework_arc_inflight]].

---

## 2. The reframe -- this is the within-domain-exactness campaign from the other end

Cross-domain skew is **not measurable from traces alone**. For a parent event in
domain A and child in domain B, the raw trace-timer offset decomposes as:

```
raw_offset = child_ts - parent_ts
           = [W(child) - W(parent)]  +  [origin_A - origin_B]
           =      Delta_wall          +        skew(A,B)
```

Two unknowns, one equation, and no second independent traceable signal exists (a
second broadcast on the same path has its arrival delay cancel against the reset
delay -- the three walls in
[`docs/trace/cross-domain-skew-limit.md`](../../trace/cross-domain-skew-limit.md)).

The only way through is **traces plus a within-domain-verified emulator**:

```
skew(A,B) = HW_raw_offset  -  W_sim(Delta_wall)
```

where `W_sim(Delta_wall)` comes from an emulator whose within-domain timing is
byte-exact (so `skew == 0` by definition within a domain, and the cross-domain
residual isolates the real skew). This is not circular: it is traces plus an
independently-validated model.

**Consequence:** the timer-sync campaign and the within-domain-exactness
campaign (device-model audit, relay-fill, send-cadence) are the *same campaign
viewed from opposite ends*. The within-domain Q=0 prerequisite was substantially
cleared this session:

- **Recv side closed** (2026-06-27): memtile S2MM `PORT_RUNNING_0`
  `[34,16,14] -> [16,16,16,16]` == HW, `sum(PORT_RUNNING)==words`.
- **Send side substantially resolved** (2026-06-28): `[16,16,16,16] ->
  [16,16,15,8,8,1]` vs HW `[8,8,14,2,14,2,6,8,1]`; both recv ports exact;
  residual is a narrowed **cold-start transient** (known-fidelity-gaps row 51).

The gate this arc rests on is therefore now reachable.

---

## 3. Findings that shape the arc (2026-06-28 deep dive)

Three load-bearing facts, established/corrected by the deep dive. The first
corrects imprecise shorthand in the canonical docs.

### 3.1 aiesim cannot oracle the skew -- wired topology, dormant propagation

The prior canonical claim that aiesim leaves the inter-tile broadcast network
"unwired" is **imprecise**. The accurate picture:

- **Topology is wired.** The wiring diagnostic confirms all 50 EventBroadcast
  objects have their directional channel pointers connected; every
  memtile<->memtile / memtile<->compute / compute<->compute hop is a shared
  channel; block masks at reset-default (unblocked).
- **Propagation is dormant.** The consumer lists are *populated* (consB 1-4
  across 170 wired channels, none empty -- `FINDINGS.md` L145-147), yet the
  array/compute EventBroadcast units emit *zero* value events across a whole run
  until our bridge intervenes (before bridge: 0 array broadcast signals; after:
  112). The mechanism: `generate_outputs` is gated on a per-EB credit counter
  (`0x192a`) that only a tile's *own* events advance via `read_inputs`, and the
  sub-model seams are dropped (the shim->memtile input is a dangling placeholder;
  memtile->compute is an unbridged seam), so the array EBs never get credited and
  never fire. It is a **partial cluster instantiation** that leaves the inter-tile
  seams for the harness to bridge (it exports `CoreModule::event_broadcast_write`
  for exactly that -- suggestive of intent, not dispositive proof of it).
- **Therefore circular.** Our `aiesim-bridge/src/bcast_bridge.cpp` injects the
  flood "one row per posedge by construction," so any skew aiesim reports is our
  own injection cadence reflected back. aiesim **cannot** serve as the skew
  oracle.

(aiesim remains a faithful oracle for *within-domain compute timing* -- it is
cycle-exact there. The circularity is specific to inter-tile broadcast skew.)

### 3.2 The per-hop broadcast latency is a silicon-only quantity

Absent from every accessible artifact:

- **Decrypted device model** (`build/experiments/aiesim-device-decrypt/VC2802.plaintext.json`):
  contains broadcast *configuration* (16 channels/module, event IDs
  `broadcast_0`=107 .. `broadcast_15`=122, memtile `broadcast_a_15`=157,
  `timer_sync`=3, the four directional block-mask register offsets, timer
  control offsets) but **zero propagation timing**. Every timing field is
  data-path/structural (`StreamSwitch.delay=2`, memory latencies, FIFO depths).
  (`NPU1.json` and siblings are encrypted, but they are *our own* device dumps,
  not an AMD oracle -- decrypting them would surface no latency the schema lacks.)
- **aie-rt** `XAie_SyncTimer`: silent; assumes timers reset "at the same time,"
  no cycle guarantee.
- **AM020 Ch.2**: describes the broadcast OR-tree directional routing rules but
  gives **no cycle counts**.
- **mlir-aie** trace utils: only "a slight delay... a few clock cycles between
  tiles" -- vague observation, not a spec; plus the known `timer=0` sync TODO.

It must be measured on silicon. It is small and deterministic: add_one HW shows
stable `core-memtile=+2`, `memmod-memtile=+4`, `core-memmod=-2` (range 0 over 20
runs).

### 3.3 The mechanism is fully toolchain-derivable

The flood *structure* needs no measurement: topology + OR-tree directional rules
(AM020 Ch.2) + per-module event IDs (AM025 regdb / device-model config). Only
the per-hop latency constants (`d_h`, `d_v`, intra-tile core/mem offset) are
silicon-measured. This cleanly separates a toolchain-grounded mechanism from a
handful of calibrated constants -- a faithful model, not a fudge.

---

## 4. The decomposition (SP-1 .. SP-5)

Dependency shape: `SP-1 -> SP-2 -> SP-4a` (EMU byte-match) and `SP-1 + SP-3 ->
SP-4b` (engine causal decomposition); `SP-3` runs alongside SP-1/SP-2; `SP-5`
slots into SP-1's parameters whenever we choose to spend the silicon. Note SP-4b
reads `origin_D` (SP-1) and `W_sim` directly and does **not** depend on SP-2's
trace-origin reconciliation.

### SP-1 -- Faithful broadcast flood (emulator core)

**The foundation. Fully buildable now; no HW, no kernel dependency;
unit-testable in isolation.**

- Add per-hop propagation delay to `propagate_broadcasts`
  (`src/device/state/effects.rs`): each tile's timer reset fires at
  `origin_D = n_h*d_h + n_v*d_v` from the flood source, not at the same cycle.
- Replace the current DFS/LIFO frontier with a real **min-latency wavefront**
  (Dijkstra/BFS by cumulative latency). The OR-tree means the earliest-arriving
  direction wins, so `origin_D` is the minimum-latency arrival -- a DFS frontier
  can assign the wrong arrival.
- **Derive broadcast event IDs from regdb** instead of the hardcoded
  `CORE_BROADCAST_BASE=107` / `MEMTILE_BROADCAST_BASE=142` bases (kills the
  157-vs-122 indexed-mapping bug class).
- Model the **intra-tile core/mem module asymmetry** (different broadcast input
  pipeline depths -> the +2/+4-style offset).
- Latency constants (`d_h`, `d_v`, intra-tile offset) become **named parameters
  with documented placeholder values**, structured so SP-5 calibration is a
  drop-in.
- **Carry forward route-1's Component-3 correctness hazards** (from the parked
  spec; this arc supersedes route-1's sequencing/gate but must not lose these,
  and all three are code-confirmed live against the wired flood->timer path at
  `tile/mod.rs`): (i) the latched `pending_reset` must apply even to a
  clock-gated module that isn't ticked, or `origin_D` corrupts; (ii) the flood
  must complete *before* the first in-window traced event; (iii) the shim must
  self-reset its own timer when it generates the sync event.
- **Tests:** pure unit tests on flood arrival ordering/timing (tile at
  hop-distance N resets at the N-weighted origin). Mechanism + directional rules
  ground in AM020 Ch.2 + regdb.

Existing code today: `propagate_broadcasts` floods with **zero** per-hop delay
(all tiles see `current_cycle`); `src/device/timer.rs` already does
event-driven reset (`notify_event -> pending_reset -> tick`).

### SP-2 -- Trace-origin reconciliation

- Each traced module's trace **Start-frame absolute origin** reflects its
  `origin_D` (the flood-reset arrival cycle). Today all modules share global
  `cycle = 0`; the trace unit's `timer` field is dead code and the real origin
  is the global `cycle` argument in `notify_*_trace_event`.
- **Hard invariant:** within-domain deltas stay **byte-identical** -- only the
  Start-frame absolute value shifts. The existing trace sweep is the regression
  gate (origin-invariant).
- This is what makes cross-domain trace timestamps actually *carry* the modeled
  skew. Depends on SP-1 (needs `origin_D`).

### SP-3 -- The validation kernel (gate-carrier)

Round-3 problem 3: no existing kernel carries the gate (two_col's on-chip
cross-domain pairs terminate on HW-silent circular-BD task events;
matrix_transpose is single-core shim<->core, DDR-excluded). So we hand-write
one.

- **Bidirectional on-chip dataflow** (distribute + gather), >=2 distinct hop
  counts, **direction-diverse** (rank-2 in `(Delta_hops, direction)`),
  **non-circular-BD** so events fire *in-window*, traceable on real NPU1.
- Within-domain segments must be clean (Q=0) -- designed to **anchor past any
  cold-start transient** (the send-side residual from row 51).
- Compiler/IRON work; parallelizable with SP-1/SP-2.

### SP-4 -- Cross-domain gate + inference integration

This splits into two pieces with different dependencies and different epistemic
standing. Keeping them separate matters: 4a is uncontroversial faithful
reproduction; 4b changes a standing rule.

**SP-4a -- EMU cross-domain byte-match (needs SP-1 + SP-2).** With the flood
delays (SP-1) and trace-origin reconciliation (SP-2) in place, the emulator's
trace reproduces HW's cross-domain *raw* offsets (`W_sim(Delta_wall) + skew`).
No rule change -- this is just the emulator becoming faithful across domains.
Gate: `emu_raw == hw_raw` on the validation kernel's on-chip cross-domain pairs.

**SP-4b -- Engine skew-export + causal decomposition (needs SP-1 + SP-3; *not*
SP-2; rev2: P1/P2 deferred to SP-5).** Where round-3 problems 1 and 2 get fixed and where skew-limit Sec.9 is
amended.

- **Coupling-latency validation moves to an in-domain round-trip gate.** A
  same-domain A->B->A segment is a within-domain segment (Q=0 ground truth), so
  on-chip coupling latencies (`ROUTE_PER_HOP`, DMA pipeline) are pinned there --
  *not* in the cross-domain gate. (Round-3 problem 1: on a monotone data path,
  data-coupling and skew share the same hop count `h`, so a hop-count
  cross-domain fit returns `d + ROUTE_error` and matches HW by construction
  regardless of a wrong `ROUTE_PER_HOP` -- the gate is illusory.)
- **The cross-domain solve keys on direction diversity (rank-2), not hop
  count**, and solves separate `d_h` / `d_v`. (Round-3 problem 2: the flood is
  omnidirectional, so `origin_D = n_h*d_h + n_v*d_v`, not scalar `d*(n_h+n_v)`.)
- **Solve `skew(A,B)` and export it to the inference engine.** The skew solver
  reads `origin_D` (SP-1) and `W_sim` (the emulator's global-cycle wall time),
  *not* the encoded trace -- so it does not need SP-2. The engine then grounds a
  cross-domain coupling as a causal segment with `causal_offset = raw_offset -
  skew(A,B) = Delta_wall`.
- **This is a NEW grounded quantity, distinct from `reproduction_offset`.** The
  existing `reproduction_offset` (the raw fused `Delta_wall + skew`, which is
  what *reproduces* the trace; `grounding.py:105`) is **kept unchanged**; the
  decomposed `causal_offset` is added alongside it. SP-4b must therefore
  deliberately relax the `timeline.py` cross-domain fail-loud guard (which today
  forbids emitting a cross-domain offset as a causal cycle) and amend skew-limit
  Sec.9 -- it does **not** silently reuse the existing field. The `is_async_cdc`
  shim/DDR pairs stay existence-only: their skew is non-deterministic, out of
  scope.
- Depends on SP-1, SP-3.

### SP-5 -- Phoenix silicon characterization (deferred, Phoenix-gated)

- Measure `d_h` / `d_v` / intra-tile-offset across a tile-distance sweep on
  Phoenix; replace SP-1's placeholders.
- Per the sequencing decision (a), this comes **last** -- but SP-1 makes the
  constants parameters precisely so this is a drop-in, and the
  within-domain-exact emulator is what lets us *interpret* the capture.
- This is the one irreducible silicon input. Bank it before the Phoenix swap.
- **Candidate direct-measurement / cross-check instrument:** a compute-path timer
  read (`__builtin_aiev2_read_tm`) with a two-source broadcast trigger solves
  `d_h`/`d_v` directly on silicon, independent of the trace-decomposition route.
  See `docs/trace/cross-domain-skew-limit.md` §8 route 3b. Reasoned, untested;
  SP-5 evaluates it alongside the tile-distance sweep.

---

## 5. Revisions from the parked route-1 spec

The parked route-1 design assumed a single scalar `d` and an over-determined
*hop-count* cross-domain gate. Round-3 proved both wrong. This arc bakes in both
fixes from the start:

| Parked route-1 | This arc |
|----------------|----------|
| Single scalar per-hop `d` | Separate `d_h` / `d_v` (SP-1), solved direction-diverse (SP-4) |
| DFS/LIFO broadcast frontier | Real min-latency wavefront (SP-1) |
| Over-determined hop-count cross-domain gate | Direction-diverse rank-2 gate (SP-4) |
| Coupling latency validated in cross-domain gate | In-domain round-trip gate (SP-4) |
| Relied on existing kernels | Purpose-built rank-2 kernel (SP-3) |
| Parked on relay-fill prerequisite | Prerequisite substantially cleared (section 2) |

The measure-first spec
([`2026-06-27-broadcast15-skew-silicon-characterization-design.md`](2026-06-27-broadcast15-skew-silicon-characterization-design.md))
remains superseded (it inverted the skew-limit doc's route priority and leaned
on the non-deterministic shim).

---

## 6. Governing constraints and decisions

- **Faithful mechanism over cosmetic byte-match** (Maya). We model what the
  silicon does -- the broadcast flood with real per-hop propagation -- not an
  offset table that merely makes EMU traces look like HW traces.
- **Sequence (a)** (Maya): build the mechanism + gate first (toolchain-grounded,
  no HW), slot in measured constants (SP-5) later. Lets us validate the
  mechanism is right before spending silicon time, and the characterization
  needs the within-domain-exact emulator to interpret anyway.
- **The Phoenix window is ours to decide** (Maya): no clock pressure. Do it
  right. (See [[project_strix_swap_replaces_phoenix]].)
- **Within-domain Q=0 is the load-bearing prerequisite.** The whole gate rests
  on it; substantially cleared this session, finished as SP-3's kernel requires.
- **Keystone (inference engine):** claim only what we measure / what the binary
  contains / what follows by verified rule. No statistical inference (exact
  agreement is equality, not stats).
- **Derive from the toolchain.** Mechanism from AM020 Ch.2 + regdb; only the
  per-hop latency constants are silicon-measured, and that fact is documented as
  such.

---

## 7. Open questions and risks

- **Cold-start transient vs the validation kernel.** The send-side residual
  (row 51) is a within-domain opening transient. SP-3 must design segments that
  anchor past it; if it bleeds into the kernel's within-domain segments, SP-3 or
  a small send-side follow-up must address it first.
- **`d_h` vs `d_v`.** Do horizontal and vertical hops actually differ? SP-1
  keeps them separate to be safe; SP-5 measures whether they collapse.
- **Intra-tile core/mem asymmetry.** Its source is broadcast input pipeline
  depth; SP-1 models it as a per-module offset. Confirm the sign/magnitude
  against the add_one `+2/+4/-2` signature.
- **Canonical-doc correction -- DONE (2026-06-28).** The "unwired" shorthand in
  `known-fidelity-gaps.md` and `cross-domain-skew-limit.md` is corrected to the
  grounded mechanism (section 3.1): topology wired and consumer lists *populated*,
  but the array/compute EBs stay dormant because their `generate_outputs`
  credit-gate is never advanced across the unbridged sub-model seams. (The first
  correction pass said "consumer lists empty / `cons=-2`" -- that was itself a
  misdiagnosis caught in review, contradicted by `bcast-bridge/FINDINGS.md`
  L145-147; re-corrected.)

---

## 8. Next step

Drill **SP-1** (faithful broadcast flood) into a full design -- its own spec
under `docs/superpowers/specs/`, then `writing-plans`. SP-1 is the foundation,
needs no HW or kernel, and is unit-testable in isolation.
