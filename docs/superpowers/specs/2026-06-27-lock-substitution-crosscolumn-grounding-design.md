# Design: Lock-Substitution Cross-Column Grounding (#140 P2)

> **SUPERSEDED (2026-06-27).** An adversarial review found this approach would
> ground a *vacuous* edge: the ledger's cross-column orientation is not
> physically faithful (DMA-buffer-relay reachability mixes the distribute and
> gather directions; the compute input S2MM DMA never appears as a consumer in
> any cross-column pair — verified). Substituting a lock onto that orientation,
> with two always-on endpoints (`PORT_RUNNING_*` + `LOCK_*_REL`), would flip the
> test `unobserved → grounded` by co-firing alone — the "tuned to pass" the
> connectivity module forbids. The lock observable is sound; what it would attach
> to is not. Direction pivoted to building a **faithful cross-column dataflow
> orientation** first (the generator's deferred Tier E). The HW root-cause work
> in this doc (circular-BD DMA silent; `LOCK_1_REL` fires) remains valid and
> feeds the new design. See the new spec (faithful cross-column orientation).

**Date:** 2026-06-27
**Context:** connectivity sub-project (#140). P1 made the cross-column
classification honest (grounded / observed_but_ungrounded / unobserved). P2 was
to *prove* a grounded cross-column edge on real hardware. The P2 diagnosis
(HW-confirmed) showed why two_col's cross-column conversations stay `unobserved`,
and this design closes that gap.

## Problem (HW-confirmed)

A cross-column conversation `memtile(col1) → compute(col2)` is oriented in the
ledger as `producer → consumer-S2MM-DMA`, via route reachability over the
stream-switch route graph. That consumer endpoint is the compute tile's S2MM DMA
task event (`DMA_S2MM_0_START_TASK` / `_FINISHED_TASK`).

On HW, that endpoint is **structurally silent**: the compute objectfifo DMA is a
circular BD ring (`two_col.config.json` compute BD0: `use_next_bd:true,
next_bd:1, lock_acq_id:0, lock_rel_id:1`) enqueued once at startup, before the
trace window opens. `DMA_*_START_TASK` / `FINISHED_TASK` are per-task-ENQUEUE
events, so they fire zero times in-window. The per-iteration data movement is
gated by **locks**.

HW evidence (`build/experiments/two_col_p2_spike/`):
- Broad menu (both DMA channels, both directions, start+finished + locks): the
  compute memmod records **zero DMA events** and **lock events every iteration**.
- Lock-confirm capture: `LOCK_1_REL` (the BD0 rel-lock) fires 64×/run on every
  compute tile (253 total). Locks 0–3 all fire uniformly (double-buffered
  objectfifo: locks 0/1 input pair, 2/3 output pair).

So the events that fire (locks) have no cross-column orientation; the event that
is oriented cross-column (consumer DMA) is silent. The conversation cannot
ground.

## Insight

The receiving S2MM DMA leaves an observable trace exactly when it commits the
arrived data: it releases its "buffer full" lock (`lock_rel_id`). That
lock-release is a faithful proxy for the silent `DMA_S2MM_FINISHED_TASK` — the
two are one physical act ("the receiving DMA wrote the buffer and signalled
ready"). The lock is **derivable from the consumer DMA's own BD**, so we ground
on the *specific* receiving-DMA rel-lock, not generic tile lock activity.

## Decisions (settled in brainstorm)

1. **Arrival observable:** the consumer DMA's lock-release `LOCK_<rel_id>_REL`
   (primary). The consumer core's lock-acquire is a documented fallback, built
   **only if** the primary proves fragile on HW (it did not — `LOCK_1_REL`
   fires). v1 = primary only.
2. **Placement:** Python generator / event layer (`tools/config_extract/` +
   `tools/inference/selfmodel.py`). The route graph stays a pure
   routing/reachability structure; "which trace event observes this silent
   endpoint" is an event-observability concern that already lives in Python
   (event_map, selfmodel menu, candidate_pairs). No Rust change, no dump schema
   change, no fixture regen. The dump already carries `dma_channels[].start_bd`
   and `bds[].lock_rel_id`.

## Architecture

Three changes, all Python, composing with P1's existing classifier.

### 1. Menu extension — `tools/inference/selfmodel.py`

`_MENU[1]` (memmod) gains a generous lock rel-event set so the consumer DMA's
`LOCK_<rel_id>_REL` is traced. Consistent with the existing "generous per-type
menu, pruned by `never_fired`" philosophy (the selfmodel docstring already frames
the menu as a deliberate over-approximation).

The menu is per-tile-TYPE; the derived rel-lock varies per tile/BD. A generous
span of rel-locks (e.g. `LOCK_0_REL..LOCK_3_REL`, covering typical
double-buffered input/output lock pairs) covers the tiles we have; `never_fired`
prunes any that don't fire for a given kernel. The 8-slot budget is shared with
the existing DMA-start events; the exact composition is an implementation detail
(keep a couple of DMA-start events for kernels whose DMA is not circular, plus
the rel-lock span).

### 2. Lock-proxy orientation — `tools/config_extract/`

A new derivation the generator invokes: for each cross-column route pair whose
**consumer endpoint is a compute S2MM DMA event**, derive the receiving DMA's
rel-lock and emit an **additional** candidate pair

```
(producer_observable_event,  compute LOCK_<rel_id>_REL)
```

Derivation (all from the dump, derive-from-toolchain):

```
consumer DMA port  --(route graph already gives this)-->  (col,row, S2MM, channel)
  -> tile.dma_channels[dir=s2mm, index=channel].start_bd
  -> tile.bds[start_bd].lock_rel_id            (= the "full" lock the DMA releases)
  -> event name  LOCK_<lock_rel_id>_REL  on (col,row,pkt=1)
```

The emitted pair is cross-domain (memtile pkt 3 → compute pkt 1) and carries a
cite naming the structural justification (the receiving DMA's BD rel-lock), so
`audit_ledger` can re-derive and check it. It is *additional* — the silent
DMA→DMA route pair is left intact (it harms nothing; see §3).

Precision note: the lock is the rel-lock of the **specific** DMA that receives
the producer's stream, not "any lock on the tile." Locks 0–3 all fire, but only
`lock_rel_id` is the arrival signal for *this* conversation.

### 3. P1 classifier — `tools/inference/connectivity.py` (expected: no change)

The new pair flows into `cross_domain_pairs`. When both endpoints fire, `weave`
grounds a `CrossTrackEdge` and `classify_connectivity` marks the **tile-pair**
grounded — no `unobserved` flag. Because the classifier works at tile-pair
granularity, the still-silent DMA→DMA pair for the same tile-pair does not force
an `unobserved` flag once the lock-proxy pair grounds the tile-pair. We verify
this composition holds; we expect zero classifier change.

## Data flow

```
dump
  → generator: route-reachability pairs (DMA endpoints, some silent)
              + NEW lock-proxy pairs (observable rel-lock endpoints)
  → candidate_pairs → cross_domain_pairs
  → weave  → CrossTrackEdge (both endpoints fired)
  → classify_connectivity → grounded   (tile-pair memtile~compute)
```

## Testing

- **Unit (derivation):** given a dump with a cross-column circular-BD consumer,
  the lock-proxy derivation emits the correct `(producer, LOCK_<rel_id>_REL)`
  pair with `rel_id` read from the right BD; negative cases (non-cross-column
  consumer, missing BD) emit nothing.
- **Unit (generator):** `generate_ledger` includes the lock-proxy entries with a
  well-formed, auditable cite; `audit_ledger` accepts them.
- **Integration (P1 classifier):** a synthetic both-ends-observed cross-column
  lock-proxy pair grounds the tile-pair (status `grounded`, no flag), while a
  silent DMA pair on the same tile-pair does not add an `unobserved` flag.
- **Real-capture (the proof):** two_col captured on real NPU1 with the extended
  menu → the cross-column conversation `1|1 ~ 2|4` (and the other three) grounds:
  `connectivity` status `grounded`, no `connectivity_unobserved` flag for those
  tile-pairs, zero defects. This is the cross-column grounding proof the
  sub-project was after.

## Out of scope (v1)

- The core-acquire fallback (build only if the DMA-rel proxy proves fragile;
  HW shows it is not).
- Any change to the Rust route-graph resolver or the dump schema.
- Non-objectfifo / non-circular dataflows beyond what the generous menu +
  derivation already cover; the derivation is general (reads BDs), but only
  two_col is HW-validated here.

## Known assumptions (HW-validated)

- `LOCK_<rel_id>_REL` fires on the compute memmod and maps to the physical lock
  the BD releases through the lock-event-selection. **Confirmed:** `LOCK_1_REL`
  fires 64×/run on all four compute tiles.
