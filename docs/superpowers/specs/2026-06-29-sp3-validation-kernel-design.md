# SP-3: The Validation Kernel (Gate-Carrier) -- Design

**Sub-project design. 2026-06-29.** Third of the five sub-projects in the
[timer-sync faithful-broadcast arc](2026-06-28-timer-sync-faithful-broadcast-arc.md)
(#140). SP-1 (faithful broadcast flood) and SP-2 (trace-origin reconciliation)
are merged; this drills SP-3 into a full design ahead of `writing-plans`.

---

## 1. Goal and role in the arc

**Goal:** hand-write one AIE2 (NPU1 / Phoenix) kernel whose on-chip trace
carries a **rank-2, direction-diverse set of cross-domain causal couplings**, so
the later sub-projects have a real artifact to validate against. No existing
kernel carries this gate (arc §4, round-3 problem 3): `two_col`'s on-chip
cross-domain pairs terminate on HW-silent circular-BD task events, and
`matrix_transpose` is single-core shim<->core with the DDR crossing excluded.

**Role.** SP-3 is the **gate-carrier**, not the gate. It delivers the kernel and
proves it traces cleanly on real silicon. The EMU byte-match (SP-4a), the skew
solve and engine grounding change (SP-4b), and the measured `d_h`/`d_v` constants
(SP-5) all consume this kernel but are out of scope here (§10).

SP-3 is HW-focused by deliberate choice: HW has been the problematic side of this
campaign, and the EMU-side reproduction is SP-4a's problem by design.

---

## 2. Requirements (from arc §4 SP-3)

The kernel must:

1. Do **bidirectional on-chip dataflow** (distribute + gather).
2. Produce couplings with **>=2 distinct hop counts** (non-degenerate).
3. Be **direction-diverse**: the couplings' flood-hop-vector differences are
   **rank-2** (so `d_h` and `d_v` are separable, not just their sum).
4. Use **non-circular BDs** so trace events fire **in-window** (the failure mode
   that makes `two_col` HW-silent).
5. Be **traceable on real NPU1**.
6. Have **clean (Q=0) within-domain anchor segments**, designed to **anchor past
   the cold-start transient** (the send-side residual, known-fidelity-gaps row 51).
7. (SP-2 final-review precondition) **reset the timer on every traced
   cross-domain tile** -- so the EMU's unconditional trace origin_offset and its
   conditional timer reset agree (§7).

---

## 3. Topology and coupling geometry

### 3.1 Tiles (Option A -- minimal 2-column diagonal)

```
            col 0          col 1
  row 2   core(0,2)      core(1,2)      compute (+const payload)
  row 1   mem(0,1)                      memtile: distribute + gather
  row 0   shim(0,0)                     DDR feed/drain + sync generator
```

Four tiles. `shim(0,0)` feeds from and drains to DDR and generates the trace-sync
broadcast; `mem(0,1)` distributes to and gathers from the two compute cores;
`core(0,2)` and `core(1,2)` run a trivial `+const` payload.

### 3.2 The geometry fact that makes this read off the grid

The BROADCAST_15 flood originates at the shim corner `(col 0, row 0)`, and every
tile sits at `col >= 0, row >= 0`. So each module's flood-arrival delay is
`origin = n_h*d_h + n_v*d_v` with `n_h = col`, `n_v = row` (min-latency Manhattan
from the corner). A coupling between two tiles therefore has flood-hop-vector
difference equal to the **coordinate difference** of the tiles:

```
Δn = (col_parent - col_child,  row_parent - row_child)
```

No reasoning about the flood source per coupling -- the tile grid gives it
directly.

### 3.3 The coupling set

The on-chip cross-domain couplings (events in different `col|row|pkt_type`
domains -- here, different rows -> different domains):

| Coupling                       | Δn (col, row) | Manhattan hops | Constrains (in SP-4b) |
|--------------------------------|---------------|----------------|-----------------------|
| `mem(0,1)` <-> `core(0,2)`     | (0, 1)        | 1              | `d_v`                 |
| `mem(0,1)` <-> `core(1,2)`     | (1, 1)        | 2              | `d_h + d_v`           |

- **Rank-2:** `(0,1)` and `(1,1)` are linearly independent (`det = -1 != 0`), so
  `d_h` and `d_v` are separable. Requirement 3 met.
- **>=2 distinct hop counts:** {1, 2}. Requirement 2 met.
- **Bidirectional:** distribute gives `mem -> core` couplings, gather gives
  `core -> mem` couplings (same Δn magnitudes, opposite causal direction).
  Requirement 1 met.

The diagonal coupling mixes `d_h` and `d_v`, so SP-4b's solve reads `d_v` off the
vertical coupling and `d_h = (diagonal skew) - d_v`. Under Q=0 both skews are
exact, so the subtraction adds no error.

### 3.4 Why Option A over the axis-clean alternative

An axis-clean variant (a pure-horizontal `mem(0,1) <-> mem(1,1)` coupling that
isolates `d_h` directly, plus a 2-hop vertical) was considered and rejected as
the *starting* shape:

- The **linearity sweep is SP-5's job**, not SP-3's -- so SP-3 only needs
  separability (rank-2) and non-degeneracy (>=2 hops), not many distances.
- **Every extra traced tile adds trace-routing traffic** that can perturb the
  within-domain timing we need at Q=0. Minimal traced-tile count is the
  conservative choice for the fragile thing (clean HW trace).
- It is far less raw MLIR to hand-write and debug.

Option B (axis-clean) is the documented fallback if the diagonal coupling does
not resolve cleanly on HW (§11).

---

## 4. Authoring approach: raw MLIR + the task API

**Raw MLIR, not IRON Python.** IRON is built for useful dataflow, not for forcing
in-window trace events; its ObjectFifo lowering auto-generates the circular
`next_bd` chains that make `two_col` HW-silent, and `iter_count` is only a partial
escape. Raw MLIR with the **DMA task API** gives the control requirement 4 needs:

```
aiex.dma_configure_task -> aiex.dma_start_task -> aiex.dma_await_task -> aiex.dma_free_task
```

Each task runs a terminating BD chain (`aie.end`, not `aie.next_bd ^self`), so its
`DMA_*_START_TASK` / `DMA_*_FINISHED_TASK` events fire **once, in-window**. Pattern
reference: `mlir-aie/test/npu-xrt/shim_dma_bd_reuse/aie.mlir` (fire-and-forget
tasks). Trace-config reference: `mlir-aie/test/npu-xrt/vec_mul_event_trace/aie.mlir`.

---

## 5. Dataflow (concrete)

```
shim(0,0) --MM2S--> mem(0,1)              feed from DDR        [async-CDC, NOT coupled]
mem(0,1)  --MM2S--> core(0,2)  +const     distribute (vertical)
mem(0,1)  --MM2S--> core(1,2)  +const     distribute (diagonal)
core(0,2) --MM2S--> mem(0,1)              gather (vertical)
core(1,2) --MM2S--> mem(0,1)              gather (diagonal)
mem(0,1)  --MM2S--> shim(0,0)             drain to DDR         [async-CDC, NOT coupled]
```

The shim<->DDR transfers are `is_async_cdc` (skew is non-deterministic, out of
scope per skew-limit §5); they exist only to move data and are never coupled. The
four on-chip transfers carry the gate.

**Compute payload is deliberately trivial** (`+const` elementwise). The cores need
only consume and produce so their DMA boundary events fire; the arithmetic is
irrelevant to the timing gate, so the simplest correct thing wins.

**Iterations.** The kernel runs a small bounded number of distribute/compute/gather
iterations via repeated in-window tasks, so the measured task boundaries land in
steady state (§8). All BD chains terminate; nothing loops circularly.

---

## 6. Trace configuration and event selection

Each on-chip tile -- `mem(0,1)`, `core(0,2)`, `core(1,2)` -- gets an `aie.trace`
block (the high-level op lowers correctly from raw MLIR):

- **Distinct packet id + type per tile** (mem vs core), so the decode separates
  streams.
- **Coupling-carrier events: DMA task boundaries.**
  `DMA_MM2S_0_START_TASK` / `DMA_MM2S_0_FINISHED_TASK`,
  `DMA_S2MM_0_START_TASK` / `DMA_S2MM_0_FINISHED_TASK`. These are single in-window
  lifecycle events -- *not* per-word cadence -- so they sidestep the cold-start
  transient by construction (§8).
- **`PORT_RUNNING_n` traced but not relied on.** Cheap to include and the engine
  may still use it, but the gate does not depend on the per-word port cadence that
  the row-51 transient perturbs.
- **`aie.trace.start broadcast=15` / `aie.trace.stop broadcast=14` on every traced
  tile** (§7).

Trace output routing (packet flows to the shim trace DMA, host buffer config)
follows the standard mlir-aie trace plumbing -- no novelty there.

---

## 7. The SP-4a precondition: uniform broadcast-15 timer reset

SP-2's final review surfaced an asymmetry in the EMU: it sets each traced module's
trace `origin_offset` **unconditionally** on every reached tile, but only **resets
the module timer when the tile's `reset_event` matches the broadcast**. Trace and
timer therefore share the value but can disagree on whether it is *applied*.

For EMU and HW to agree on this kernel, **every traced cross-domain tile must
actually reset its timer on the flood.** This is satisfied by construction:
`aie.trace.start broadcast=15` lowers to `Timer_Control.Reset_Event = BROADCAST_15`
(skew-limit §1), so a uniform `broadcast=15` start on all three traced tiles makes
every one reset on the same flood.

**Invariant (spec-checked):** all three traced tiles use `broadcast=15` as the
trace-start / timer-reset event -- no tile started on a different event, no
exceptions. The plan will assert this explicitly so the implementer cannot quietly
skip it on one tile.

---

## 8. Cold-start transient avoidance

Arc §7 flags the send-side cold-start transient (known-fidelity-gaps row 51) as a
risk: an opening per-word port-cadence wobble. Two design choices remove it:

1. **Couple on task boundaries, not port cadence (§6).** The transient perturbs
   `PORT_RUNNING` cadence -- the durations between successive port toggles -- not a
   task's START/FINISH timestamp. Task-boundary couplings are immune by
   construction.
2. **Anchor past the opening.** The kernel streams a few iterations so the measured
   boundaries fall in steady state; if HW shows any opening wobble, the first
   iteration is discarded as warm-up. The within-domain anchor segments (§9) are
   positioned in the same steady region.

If the transient nonetheless bleeds into the within-domain segments, the fallback
is a small send-side follow-up (arc §7) -- but the task-boundary choice is expected
to make that unnecessary.

---

## 9. Within-domain anchors and Q=0

Alongside the cross-domain couplings, the engine needs **clean within-domain
segments** -- parent->child pairs in the *same* `col|row|pkt_type` domain -- as
Q=0 ground truth (skew is identically zero within a domain, so the offset *is*
causal latency; skew-limit §7). These come from same-module event pairs, e.g. a
single core's `DMA_S2MM_0_FINISHED_TASK -> DMA_MM2S_0_START_TASK` lifecycle, or the
memtile's own task boundaries.

`Q = 0` is a **measured property, not a tuned tolerance** (`verifier.py`): a
within-domain offset either agrees exactly across all HW runs (cross-run range 0)
or it does not qualify. The kernel must produce within-domain segments that come
out at range 0 across ~20 HW runs.

---

## 10. Scope boundary and acceptance criteria

### In scope

- The raw-MLIR kernel (dataflow + trivial payload + trace config).
- Build integration into the bridge-test harness (dual-compiler; runs on NPU1).
- A HW run (x~20 for Q=0) confirming the decoded trace yields:
  - the **rank-2 cross-domain coupling set** -- Δn `(0,1)` and `(1,1)`, hop counts
    {1,2} -- in both causal directions (distribute and gather);
  - **Q=0 within-domain anchor segments** (cross-run range 0).
- Confirmation that the **inference engine ingests** this trace and identifies
  those couplings (as cross-domain gaps with reproduction offsets -- grounding them
  with a decomposed causal offset is SP-4b).

### Out of scope (forward pointers)

- **SP-4a** -- EMU reproduces HW's cross-domain raw offsets byte-for-byte (needs
  SP-1+SP-2, both merged, + this kernel). No EMU dependency *inside* SP-3.
- **SP-4b** -- skew solve + engine grounding change + skew-limit §9 amendment.
- **SP-5** -- the measured `d_h`/`d_v`/intra-tile constants (and the route-3b
  compute-path direct-measurement cross-check, skew-limit §8).

### Acceptance test (one sentence)

*The kernel runs on NPU1, its trace decodes to a rank-2 cross-domain coupling
structure with Q=0 within-domain anchors, and the inference engine ingests it.*

---

## 11. Open questions and risks

- **Does the diagonal coupling resolve cleanly on HW?** `mem(0,1) -> core(1,2)`
  crosses a column and a row. If the decode or the engine does not surface it as a
  clean single coupling, fall back to **Option B** (axis-clean: a pure-horizontal
  `mem(0,1) <-> mem(1,1)` coupling isolating `d_h`, plus a 2-hop vertical). Option B
  costs two extra traced tiles and a less-common memtile->memtile cross-column
  route, which is why it is the fallback, not the start.
- **Trace-buffer / routing pressure.** Three traced tiles + shim trace share the
  trace plumbing; contention could in principle perturb Q=0. Minimal traced-tile
  count (§3.4) is the hedge; if Q!=0 appears, suspect routing contention before the
  emulator.
- **Cold-start transient bleed (§8).** Expected handled by task-boundary coupling;
  the send-side follow-up is the fallback.

---

## 12. Provenance of behavioral knowledge

Mechanism and topology derive from the toolchain: trace-config / timer-reset
lowering (`AIEInsertTraceFlows`, skew-limit §1), the DMA task API
(`shim_dma_bd_reuse`), the event set (`vec_mul_event_trace`), and the engine's
domain/coupling/Q=0 semantics (`grounding.py`, `verifier.py`). The only
silicon-measured quantities (`d_h`, `d_v`, intra-tile offset) are SP-5's, not
SP-3's -- SP-3 produces the artifact that lets SP-5 measure them.
