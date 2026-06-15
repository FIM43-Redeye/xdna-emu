# Byte-identical held-level trace encoding -- scoping (task #141)

**Date:** 2026-06-14
**Status:** Scoping. Derives HW's held-level framing from a config-matched
HW+EMU capture of `add_one_using_dma` and proposes the re-architecture, with an
honest assessment of what byte-identity is and isn't achievable.
**Supersedes the *mechanism* of** the EMU continuous-hold choice in
[2026-06-08 skip-token spec](2026-06-08-skip-token-held-level-encoding.md) for
the dense-event case; that spec's upstream-oracle correctness still holds.

## The divergence (root, confirmed at byte level)

EMU and HW both encode held levels with skip tokens, but choose **opposite
strategies**:

- **EMU -- continuous hold.** One `Event(active, cyc=0)` then one big
  `Repeat1`, *by design* so the upstream oracle keeps the level continuously
  active across the whole span (`commit_pending_frame`, `frame_held` branch).
- **HW -- per-edge re-checkpoint.** On every change to the tile's active slot
  set, HW re-emits the **full** active set as a triplet:
  `Event(bits, cyc=G)` + `Event(bits, cyc=0)` + optional `Repeat0(R)`.
  Held levels are thus re-emitted whenever *any* slot in the tile changes.

Through the in-tree decoder this renders HW as **2 records per segment** (the
`cyc=G` and `cyc=0` frames share one soc) vs EMU's **1 record per interval** --
the 2x raw-count factor. Same decoded timeline; different bytes.

## HW framing rule (derived from config-matched capture)

memtile (3,1,1), `add_one_using_dma`, steady state -- every segment:

```
Event(bits, cyc=G)   <- full active set, G-cycle lead
Event(bits, cyc=0)   <- ALWAYS present: re-arm so the following Repeat extends
Repeat0(R)           <- optional; R in {4,6,12,14} (<=15 -> Repeat0, 1 byte)
```

Observed steady-state loop: `[slot5 R=14], [slot1 R=6], [slot1 R=6]` repeating.

**Firmly established:**
1. A frame is emitted on **every** tile slot-set change, carrying the full
   current active set (so held levels re-emit alongside any neighbour's edge).
2. Each emission is the `cyc=G` + `cyc=0` (+ `Repeat`) triplet; the `cyc=0`
   arm frame is invariant.
3. `Repeat0` (<=15) for short holds; HW uses `Repeat1` only for long isolated
   holds (e.g. LOCK_STALL 6354 -> `Repeat1(1023)` chunks), i.e. when nothing
   else in the tile forces intervening frames.

**NOT cleanly derivable from samples:** the exact split of a segment's total
hold into `G` (lead-frame cycles) vs `R` (Repeat). G ranges 1..61 with no
closed form against the timeline; it reflects HW trace-unit internal
buffering/flush cadence. This is the crux risk below.

## Re-architecture proposal

Replace the EMU `frame_held` continuous-hold path with HW's per-edge triplet:

1. **Emit on every active-set change** carrying the full `active` snapshot (EMU
   already commits on changes; the snapshot is already `held_mask | pulses`).
2. **Per emission**: `emit_event_frame(active, G)`, `emit_event_frame(active,
   0)`, then `Repeat` for the residual hold. This is exactly the existing
   *opening-a-hold* branch (lines 972-975) -- generalise it to the continuing
   case instead of the `emit_skip_run(gap-1)` + single `cyc=0` form.
3. **Repeat0 vs Repeat1**: emit `Repeat0` when the run <=15, else `Repeat1`
   (the emitters already exist; selection logic is the change).
4. **G/R split**: tune empirically against config-matched captures via a
   round-trip byte-diff TDD harness (below).

### Regression strategy (the gate)

- **Keep green:** the upstream-oracle LOCK_STALL long-span test (#98/#99) --
  the isolated-hold case must still decode to one continuous span under
  upstream `parse_trace`. The per-edge triplet uses `cyc>0` frames that upstream
  treats as deactivate/reactivate, so the *isolated* path must still collapse to
  Repeat1 without spurious intervening frames. This is the main hazard.
- **New gate:** byte-diff EMU `trace_raw.bin` against the config-matched HW
  capture for `add_one_using_dma` (and a few more kernels) -- the direct
  byte-identity assertion.

## Honest assessment (radical candor)

- **Achievable:** structural identity (per-edge triplet, full-set re-emit,
  Repeat0 usage) and byte-identity *for captured kernels* via empirical G/R
  fitting. This already collapses the 2x record factor at the source and makes
  the comparator's interval-normalisation redundant (kept as a safety net).
- **Uncertain:** *perfect* byte-identity across **all** kernels. The G/R split
  depends on HW trace-unit flush timing we can only observe, not derive. We may
  match captured cases byte-for-byte yet drift on unseen ones. If that proves
  true, the honest landing is "structurally identical + byte-identical on the
  validated corpus," not a universal guarantee.
- **Cost/risk:** touches the #99/#100 skip-token encoder that gates the whole
  trace-fidelity line. Non-trivial; worth doing carefully behind the regression
  gate, not rushed.

## Recommendation

Proceed with the per-edge-triplet re-architecture **behind both gates**, fitting
G/R against config-matched captures. Set expectations at "byte-identical on the
validated corpus." If G/R generalisation fails, fall back to structural identity
(2-records/segment, right cadence) which is itself a large fidelity gain and
leaves the interval comparator as the equivalence check.
