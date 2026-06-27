# Design: Cross-Column Edge Grounding via Lock-Substitution (v2, post-orientation-fix)

**Date:** 2026-06-27
**Context:** connectivity sub-project (#140), P2. Supersedes
`2026-06-27-lock-substitution-crosscolumn-grounding-design.md` (v1), which was
written against a buggy ledger orientation. The orientation inversion is now
fixed (commit: `_resolve_tile_dma` S2MM/MM2S direction), so the cross-column
consumer is correctly the compute **input S2MM** DMA, and v1's adversarial-review
blockers C1/C2/I5 dissolve. This v2 carries forward the still-valid points
(I3 audit branch, I4 validity guard) and is honest about the existence-only
nature of cross-domain grounding (C2).

## Problem (now that orientation is faithful)

With the inversion fixed, two_col's cross-column conversations are correctly
oriented: `producer = memtile/shim MM2S → consumer = compute S2MM` (verified:
`consumer=2|4|1|DMA_S2MM_0_START_TASK <- producer=1|1|3|PORT_RUNNING_0`). But the
consumer endpoint (compute S2MM DMA task event) is **structurally silent** — the
objectfifo DMA is a circular BD ring enqueued before the trace window, so
`DMA_S2MM_*_START_TASK/FINISHED_TASK` fire zero times in-window (HW-confirmed,
finding 2026-06-27). The receiving DMA's per-iteration observable is its
buffer-full lock release: for two_col compute BD0 (`lock_rel_id=1`),
`LOCK_1_REL` fires 64×/run on every compute tile (HW-confirmed). So the
conversation is honestly `unobserved` (consumer endpoint never fires), and to
*prove* a grounded cross-column edge we substitute the silent consumer with its
own rel-lock event.

This is now sound where v1 was not: the lock attaches to the **correctly
oriented** consumer (compute S2MM input), and the arrival lock (LOCK_1) and the
silent consumer endpoint coincide — the divergence v1's review flagged (I5) was
itself an artifact of the inversion.

## Decisions (from brainstorm, unchanged)

1. **Arrival observable:** consumer DMA's lock-release `LOCK_<rel_id>_REL`
   (primary). Core-acquire fallback documented, out of v1/v2 scope (HW shows the
   primary fires).
2. **Placement:** Python generator / event layer.

## Architecture

### 1. Menu extension — `tools/inference/selfmodel.py`

`_MENU[1]` (memmod) gains a generous lock rel-event span (e.g.
`LOCK_0_REL..LOCK_3_REL`, covering input/output double-buffer rel-locks) so the
consumer DMA's `LOCK_<rel_id>_REL` is traced. Generous over-approximation pruned
by `never_fired`, consistent with the existing menu philosophy. 8-slot budget is
fine — the memmod carries no anchor (anchor rides the core), so all 8 slots are
available; current 3 events + 4 rel-locks = 7 ≤ 8.

### 2. Lock-proxy orientation — `tools/config_extract/`

A new derivation the generator invokes. For each cross-column ledger pair whose
**consumer (child) endpoint is a compute S2MM DMA event**, emit an *additional*
entry substituting the consumer with its rel-lock:

```
parse channel from the event name        DMA_S2MM_<ch>_*  -> ch
locate the S2MM DMA channel              tile.dma_channels[dir=s2mm, index=ch].start_bd
guard the BD is configured               tile.bds[start_bd].valid  (else: emit nothing)   # I4
read the buffer-full lock                tile.bds[start_bd].lock_rel_id  -> rel
emit  (parent = producer,  child = compute LOCK_<rel>_REL on (col,row,pkt=1))
```

Validity/precision guards (I4): channel comes from the event name (not guessed),
and the BD must be `valid` — `start_bd=0` is also the unconfigured-channel
default, so an invalid BD0 must yield no pair, not a bogus `LOCK_0_REL`.

The entry is cross-domain (memtile pkt 3 / shim pkt 2 → compute pkt 1) and
*additional* — the silent S2MM route pair is left intact (harmless; the
classifier grounds at tile-pair granularity). Pair-tuple order follows the
`(child, parent)` codebase convention (M6).

### 3. audit_ledger branch — `tools/config_extract/generator.py` (I3)

`resolve_event_port` returns `None` for lock events, and `audit_ledger` rejects
any non-`route`/`program` kind and any `b` that doesn't resolve to a route node.
So the lock-proxy entry needs its own cite kind + audit branch, modeled on the
existing `program_order:` special-case: a `lock_proxy:` cite that names the
consumer DMA channel + rel-lock, and an audit branch that re-derives
`channel → start_bd → bds[start_bd].lock_rel_id → LOCK_<rel>_REL` from the dump
and checks the cited lock matches. This branch is the trust anchor for the
feature — it must be re-derivable, not asserted.

### 4. P1 classifier — `tools/inference/connectivity.py` (no change expected)

The lock-proxy entry flows into `cross_domain_pairs`; when both endpoints fire,
`weave` grounds a `CrossTrackEdge` and `classify_connectivity` marks the
tile-pair `grounded`. Tile-pair granularity absorbs the still-silent S2MM pair.
Verify; expect zero change.

## Honesty: what the grounded edge claims (C2)

The grounded cross-column edge is **existence-only**: it witnesses that the
configured producer→consumer path was *active* in the run (both endpoints
fired), backed by **faithful structural reachability** (post-fix). It is NOT a
cycle-exact timing relationship — cross-timer-domain cycle grounding needs
timer-sync (BROADCAST_15), which is deferred. This is exactly the standard the
engine already applies to every cross-domain edge; with faithful orientation it
is an honest existence witness, not co-firing tuned to pass. Both endpoints are
high-frequency (PORT_RUNNING level + LOCK_REL per-iteration), so the witness is
"the path ran," not "these two specific events are causally adjacent" — stated
plainly so the claim is not over-read. The producer PORT_RUNNING is also a
tile-level observable that can feed multiple consumers (M7); the *orientation*
(which consumer) is carried by the route graph, not the co-firing.

## Testing

- **Unit (derivation):** cross-column S2MM-consumer pair → correct
  `(LOCK_<rel>_REL, producer)` with `rel` from the right BD; `bd.valid=false` →
  no pair; channel parsed from event name; non-cross-column / non-S2MM consumer
  → no pair.
- **Unit (audit):** `audit_ledger` accepts a well-formed `lock_proxy:` entry and
  rejects one whose cited lock does not re-derive from the dump.
- **Integration (P1):** a both-ends-observed cross-column lock-proxy pair grounds
  the tile-pair (`grounded`, no flag); the silent S2MM pair adds no `unobserved`
  flag for that tile-pair.
- **Real-capture (the proof):** two_col on real NPU1 with the extended menu →
  the cross-column conversation `1|1 ~ 2|4` (and the other three) grounds:
  `grounded`, no `connectivity_unobserved` flag, zero defects. The cross-column
  grounding proof the sub-project was after.

## Out of scope (v2)

- Core-acquire fallback (primary fires).
- Cross-timer-domain cycle-exact grounding (timer-sync; deferred).
- Rust route-graph / dump-schema changes.

## Validated assumptions (HW)

- Faithful orientation post-fix: compute S2MM is the cross-column consumer
  (verified offline on the fixed code).
- `LOCK_1_REL` (BD0 rel-lock) fires 64×/run on all four compute tiles
  (HW-confirmed).
- add_one grounding un-regressed by the orientation fix (HW: `placed`, core
  segment exact, shim gaps intact).
