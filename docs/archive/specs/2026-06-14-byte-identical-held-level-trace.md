# Byte-identical held-level trace encoding -- scoping (task #141)

**Date:** 2026-06-14
**Status:** Scoped, evidence-backed. HW's held-level mechanism is now derived
from three converging sources (mlir-aie decoder, aie-rt/AM025 trace registers,
and a 140-capture HW corpus), correcting the first-pass "not derivable" guess.
Proposes the re-architecture (P1) and reframes byte-identity as encoder (P1) +
cycle-exact timeline (P2), with the byte-diff doubling as a timeline-fidelity
oracle. Awaiting go/no-go on starting the P1 encoder rewrite.
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

## HW mechanism (derived from toolchain + 140-capture corpus)

Three independent dives (mlir-aie decoder, aie-rt/AM025 trace registers,
140 HW chess captures) converge on a single mechanism. The earlier "G/R split
is not derivable / reflects internal flush cadence" claim was **wrong** -- it
analysed one port in isolation. Corrected picture:

1. **Emit on every watched-event edge, carrying the FULL active mask.** The HW
   trace unit re-states the entire asserted slot set on any slot's edge.
   Confirmed at byte level: when slot 4 turns on while slot 0 is held, the
   frame carries both bits (`mask=10001000`). The decoder's own model (each
   `EventCmd` = "snapshot of the currently asserted mask") matches silicon.
2. **The lead-frame `cyc` = delta since the previous emitted packet.** The "G"
   that looked like a free 1..61 parameter is just the inter-edge spacing of
   the *whole tile's combined* slot timeline. Derivable from the event
   timeline, not a flush knob.
3. **No hardware periodic flush exists.** `Trace_Control0/1`, `Trace_Status`
   carry no timer / sync-interval / keepalive field (AM025). The only forced
   emission is a `Sync` on 56-bit timer overflow (`0x3FFFF`), irrelevant at
   kernel lengths.
4. **Held-level duration is a greedy repeat-token fill, re-checkpointed at
   token saturation.** Corpus evidence (held-level re-checkpoint gap
   histogram):
   - `gap=1024` huge spike = `Repeat1(1023)` + implicit `+1` -> the 10-bit
     token **saturating** and forcing a full-mask re-emit. Pure encoder
     artifact, fully derivable.
   - long tail is **16-aligned** (368, 384, ... 1024, 1520) = `Event+Repeat0`
     (~16 cyc) chains accumulating.
   - `Repeat1(1023)` is the dominant token (2741 occurrences) -- HW readily
     encodes long isolated holds; the earlier "caps at 15" read was a
     one-kernel red herring.

So the cadence is the **streaming encoder's greedy token-fill** (emit the
largest repeat token that fits the gap to the next edge; re-checkpoint the full
mask when it saturates), layered on the **real event timeline**. Both are
deterministic.

## The decomposition: byte-identity = P1 (encoder) + P2 (cycle-exact timeline)

Byte-identity cleanly splits into two independent problems:

- **P1 -- the encoder.** Full-mask-on-every-edge + `cyc`=inter-edge-delta +
  greedy repeat-fill (`Repeat1` for runs >=16, `Repeat0` for <16) +
  re-checkpoint on saturation. **Derivable now** from toolchain + corpus.
  Bounded, achievable; this is the #141 re-architecture.
- **P2 -- the event timeline.** Whether each port *actually* toggles on the
  same cycle as HW. This is the existing #140 model-fidelity frontier (DDR
  burst cadence, receive-FIFO calibration). If our timeline is off by a cycle,
  the deltas diverge and bytes won't match.

**P1 alone buys structural identity** (per-edge full-mask re-checkpoint) and
collapses the 2x record factor at the source, making the comparator's
interval-normalisation redundant (keep it as a safety net).

**True byte-identity = P1 + P2.** The bytes match only when the encoder matches
*and* the timeline is cycle-exact. Which gives the prize:

> Once P1 lands, **byte-diff against the HW capture is the tightest fidelity
> oracle available** -- EMU bytes matching HW bytes for a kernel *proves* the
> event timeline is cycle-exact for it. Byte-identity becomes a measurement
> tool for the #140 timeline work, not a standalone purity goal.

## Re-architecture proposal (P1)

Replace the EMU `frame_held` continuous-hold path with HW's edge-driven model:

1. **Emit on every active-set change** carrying the full `active` snapshot (EMU
   already commits on changes; the snapshot is already `held_mask | pulses`).
2. **Per emission**: `emit_event_frame(active, delta)`, `emit_event_frame(active,
   0)`, then greedy `Repeat` fill for the residual hold. This generalises the
   existing *opening-a-hold* branch (lines 972-975) to the continuing case
   instead of the `emit_skip_run(gap-1)` + single `cyc=0` form.
3. **Greedy repeat-fill**: for a gap to the next edge, emit `Repeat1(1023)`
   chunks while >=16 remain, then `Repeat0(<=15)` remainder; re-checkpoint the
   full mask at each saturation boundary. (Emitters already exist; the
   selection + re-checkpoint logic is the change.) Validate the exact token
   choice against captures in the byte-diff TDD loop below -- this is encoder
   detail, fit it there, not a blocker to the design.

### Regression strategy (the gate)

- **Keep green:** the upstream-oracle LOCK_STALL long-span test (#98/#99). HW's
  own trace decodes correctly under upstream `parse_trace` (that's what the tool
  is *for*), so faithful HW-identical emission is upstream-correct by
  construction -- the hazard is only the isolated-hold path still collapsing to
  the right `Repeat1` chunks without spurious intervening frames.
- **New gate:** byte-diff EMU `trace_raw.bin` against the config-matched HW
  capture, per kernel. This is simultaneously the byte-identity assertion *and*
  the P2 timeline-fidelity oracle.

## Honest assessment (radical candor)

- **Achievable now (P1):** structural identity + correct encoder cadence. Kills
  the 2x record factor at the source.
- **Gated on P2 for true byte-identity:** the bytes only match where the event
  timeline is already cycle-exact. For kernels still carrying #140 timeline gaps
  (DDR burst, FIFO calibration) the byte-diff will *fail until the timeline is
  fixed* -- which is the point: it becomes the gate that tells us the timeline
  is right. Honest landing: "encoder byte-identical now; full byte-identity
  tracks timeline fidelity, kernel by kernel."
- **Cost/risk:** touches the #99/#100 skip-token encoder that gates the whole
  trace-fidelity line. Worth doing carefully behind both gates, not rushed.

## Recommendation

Land the P1 encoder re-architecture **behind both gates**, and adopt the
per-kernel byte-diff as the new P2 timeline-fidelity oracle (subsuming the
interval comparator). Expectation: encoder-identical immediately; full
byte-identity per kernel as #140 timeline gaps close. This turns #141 from a
purity exercise into the sharpest validation signal for the timeline work
we are already doing.
