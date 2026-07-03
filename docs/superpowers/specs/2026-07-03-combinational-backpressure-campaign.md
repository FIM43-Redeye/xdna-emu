# Combinational backpressure campaign (#140 SP-4a offset)

**Date:** 2026-07-03  **Status:** DESIGN -- Maya-approved direction (chose this
over "gate execution on dispatch" and "document as limit").
**Goal oracle:** lean-kernel prod->consA first LOCK_STALL **-62 -> +2** (HW
range-0). Secondary: of_out drain dump + shim starvation (+1491 -> ~+14) close as
a side effect (same root).

## The gap (empirically nailed, 2026-07-03)

Both HW and EMU pre-fill the pipeline ~equally (in-core cntr: prod~12/consA~5 HW,
prod~10/consA~4 EMU) -- so the offset is **NOT a fill-depth gap**. It is a
**backpressure-TIMING gap**:

- **HW:** a cold terminal (un-armed shim S2MM) holds TREADY low **combinationally
  end-to-end**; when the shim is cold, *every* upstream stage backs up in ~the
  same cycle, so both cores hit their first LOCK_STALL ~together -- **prod +22,
  consA +24 (producer-first, 2cy spread)**.
- **EMU:** backpressure is a **stage-by-stage FIFO fill** (each hop stalls only
  when the next FIFO fills, one hop per routing cycle), so the drain-side stage
  (of_j, nearest the cold shim) backs up first and the source (of_src/prod) backs
  up ~63-104cy later -- **consumer-first, 63-104cy spread** (true cold-start:
  consA cyc506, prod cyc610; resume: consA 7367, prod 7427 -> trace offset -62).

The flip is real at BOTH true-cold-start and the trace-resume phase, so it is not
merely a trace-arm-phase artifact (though the ~6791cy dispatch delay does mean the
trace measures the resume, not the initial stall -- secondary).

## What has been tried and ruled out (do NOT re-chase)

- **Fix (a)** `channel_is_started`->source-MM2S gate (blocks the of_out MM2S when
  the shim is un-armed): IMPLEMENTED (uncommitted, `routing.rs:894-952`) and
  VERIFIED ACTIVE (blocks exactly memtile(1,1) MM2S ch1). Necessary (keeps the
  FABRIC empty) but **INSUFFICIENT**: offset stays -62, starvation +1491. It gates
  only the terminal circuit's source, not the transitive chain -- the memtile
  MEMORY buffers still fill stage-by-stage.
- **Drain-rate throttle** (`XDNA_EMU_S2MM_COLD_COOLDOWN`, parked): meters drain
  rate, not backpressure timing -> offset pinned -62 across 10x sweep.
- **memtile_lock_release_latency 63->0:** no effect on offset.
- **memtile_first_bd_startup** (post-transfer hold): flips offset DISCRETELY
  -62 <-> +63 (at ~2000), never +2; improves starvation 1491->595. A real lever
  (slows the fill) but overshoots and does not tighten the 63cy spread.

## The mechanism to build

Make a cold terminal's backpressure propagate **combinationally (same cycle)**
across the coupled objectfifo chain back to the producer, so upstream stages
stall ~together (2cy), not 63-104cy apart. Two propagation domains:

1. **Within a circuit (pure stream route):** end-to-end TREADY. A circuit whose
   terminal S2MM can't accept (un-armed, or buffer full) carries NO data and holds
   its source MM2S THIS cycle. Fix (a) is the un-armed special case; generalize to
   "terminal not ready for ANY reason" and make it a per-cycle combinational
   resolve (source ready = terminal ready, transitively through the fabric hops in
   one cycle).
2. **Across circuits (objectfifo link + core loop coupling):** the HARD part. The
   chain of_src -> of_d -> [consA core] -> of_j -> of_out -> shim couples through
   memtile links (of_src->of_d, of_j->of_out are explicit `objectfifo_link`) AND
   through the consumer core's loop (consA consumes of_d, produces of_j -- implicit
   in the core program). Pure-DMA transitivity resolves the link edges but NOT the
   core-loop edge. Candidate: when the terminal shim is cold, treat the memtile
   S2MM that FEEDS the cold MM2S's buffer (via link) as also-cold, and recurse; the
   core-loop edge is handled naturally because a backpressured consumer stops
   releasing its input lock the same way it does today, just without the per-hop
   FIFO-fill delay. Need to verify the core-loop edge doesn't reintroduce the
   spread.

## Implementation plan (incremental, measure each step against the lean oracle)

Step 0 (done): fix (a) active; baseline offset -62, starvation +1491.
Step 1: make the of_out-buffer-full -> of_j-S2MM-cannot-accept coupling
  COMBINATIONAL (same cycle the of_out MM2S is gated, the linked of_j S2MM stops
  accepting), so consA's of_j MM2S backpressures in-cycle. Measure offset.
Step 2: propagate through consA's loop (consA stalled on of_j-Produce stops
  consuming of_d in-cycle) and the of_d/of_src memtile relay, so prod backpressures
  in-cycle. Measure offset -> expect it to tighten toward the 2cy spread.
Step 3: reconcile with the memtile STARTING latency (the +22 transient) if a
  residual producer-first magnitude remains.
Gate at each step: `cargo test --lib` green; recv slots [16,16,16,16]; data
byte-identical; steady-state of_out 64cy/obj; the send-cadence corpus no-regress.

## Risk / watch

- Blast radius: the routing/backpressure model is shared by ALL kernels. The
  combinational resolve must be behavior-neutral when no terminal is cold (steady
  state) and for interior started-S2MM chains (byte-identical). Guard with the
  same `channel_is_started` predicate fix (a) uses -- only cold terminals trigger
  the combinational hold.
- Do NOT let the combinational hold create a per-cycle infinite loop or a
  borrow tangle in the phased `route_*` functions. A fixpoint pass (iterate the
  transitive backpressure to convergence within one cycle) may be needed.
- The parked drain-throttle + memtile-startup are separate levers; keep them
  parked (default off) until this lands, then re-evaluate whether any is still
  needed for the drain-side.

## RESULTS (2026-07-03): mechanism proven, offset is order+rate, starvation orthogonal

Built the principled graph-reachability backpressure (`src/device/array/backpressure.rs`,
env `XDNA_EMU_SP4A_BP_MODE`, default 0 = off / byte-identical, 3585/0 green). It reuses
`route_graph`'s E2/E3/P2 relay detectors + `build_mm2s_terminal_map` fabric edges, reverses
them, and BFS-backward from each cold terminal S2MM to gate every transitively-feeding
node combinationally (accept-gate interior S2MM at mode>=1, drain-gate MM2S at mode>=2).

Measured on the lean oracle:
| config | prod | consA | offset | starvation |
|---|---|---|---|---|
| baseline (fix a) | shim+661 | shim+599 | -62 | +1491 |
| mode1 (gate consA of_j) | shim+104 | shim+42 | -62 | +1491 |
| mode1 + forced core edge (gate prod) | shim+57 | shim+179 | **+122** | +1491 |

Findings:
1. **Mechanism works + is controllable.** HW's +2 is BRACKETED: gating consA-only holds -62;
   extending reach to prod FLIPS the sign to producer-first (+122). The order-lever is real.
2. **`CoreLockRelay` does not emit consA's through-core edge for this kernel** (strict /
   passthrough-only; consA computes or is Peano-built). Principled reach stops at consA; a
   looser backpressure-specific core edge is needed to reach prod. Confirmed by forcing it
   (`XDNA_EMU_SP4A_BP_FORCE_CORE`, debug-only).
3. **Asymmetric data-gate is STRUCTURALLY IMPOSSIBLE here.** prod's output (of_src) is the
   sole source of consA's input (of_d) -- one linear pipe. Any gate stalling prod's output
   starves consA's input (verified: forcing prod's gate pushed consA to shim+179, first-stall
   TYPE flipping of_j-Produce -> of_d-Consume starvation). So a pure topology gate cannot make
   prod stall-first-while-consA-fed; that is a FILL-RATE property (HW pre-fills ~10 objects
   fast+uniform ~22cy, then backpressures prod; EMU pre-fills slow+staggered).
4. **Starvation (+1491) is INVARIANT across every gating mode** -> NOT backpressure-topology;
   drain-side/throughput. The one-root hope (offset+starvation from one backpressure change)
   is falsified. But the parked `memtile_first_bd_startup` lever moved BOTH offset and
   starvation -> fill-RATE is the shared root of both remaining oracles.

**Revised plan for exact +2 (Maya: push-to-+2):** combinational-backpressure (order) COMPOSES
with a fill-rate correction (timing). Keep the backpressure infra as the order-setter; next =
characterize + correct the fill-rate (why prod pumps to stall at shim+104 vs HW +22). Same
lever is expected to close the orthogonal starvation.

### CORRECTION (2026-07-03, initial-fill probe): forward-relay starvation FALSIFIED; -62 is resume-frame

Added a cycle-stamped S2MM-accept probe (`XDNA_EMU_SP4A_RELAY_TRACE`, debug) run capped at
`XDNA_EMU_MAX_CYCLES=800` to see the INITIAL fill (before the shim dispatches at ~6791).
Measured (baseline, mode 0):
- of_src reaches memtile (1,1) S2MM ch0 at **cyc 63** (prod->memtile fabric latency), then
  streams 1 word/cy contiguously.
- of_d reaches consA (1,3) S2MM ch0 at **cyc 153** (memtile forward-relay adds ~90cy), then
  streams 1 word/cy contiguously (400 words by cyc558).

**consA is NOT starved in the initial fill** -- it receives of_d steadily from cyc153. So the
"consA waits 506cy for first of_d" forward-relay-starvation hypothesis is FALSIFIED (the 506
was a stale recollection / different measurement). The initial fill is healthy and
producer-led (of_src at 63 precedes of_d at 153).

**Therefore the -62 lives in the RESUME frame.** The EMU over-fills the pipeline in the first
~600cy, then all cores idle ~6800cy until the shim dispatches (~6791). The oracle trace arms
at shim START, so it measures the RESUME re-stall, not the producer-led initial fill. During
resume the shim drains from the CONSUMER end (shim->of_out->of_j->consA->of_d->of_src->prod),
so consA re-stalls before prod -- consumer-first, structurally. The combinational backpressure
moves absolute resume timing but CANNOT flip this order (the shim inherently frees the consumer
end first).

**Consequence:** exact +2 (producer-first) requires the oracle to measure the producer-led
INITIAL fill, i.e. eliminating the ~6800cy over-fill / aligning shim dispatch with core start
(the DISPATCH axis -- the earlier dispatch-gate hedge, which got -62 -> -14). The
combinational-backpressure routing change closes the STARVATION-side fabric over-feed but does
NOT, by itself, flip the offset sign. This reframes the campaign: offset sign = dispatch/frame,
not backpressure topology. NEEDS MAYA STEER (dispatch axis was previously set aside as a hedge).

### HW CAPTURE (2026-07-03): dispatch-axis FALSIFIED; campaign REVALIDATED

Ran the lean kernel on REAL silicon via the same runner (`run_hw.sh`, `env -u XDNA_EMU`),
same decode + measure. Result:
- HW shim S2MM_0 START soc = **428654**; prod first LOCK_STALL **428697 (+43)**, consA
  **428699 (+45)**; offset **+2**; starvation **+13**. (Matches the oracle targets exactly.)

**Decisive: HW ALSO measures a RESUME.** HW's first LOCK_STALLs land just +43/+45 AFTER shim
dispatch -- the shim dispatches late and the cores stall right after, exactly like EMU. So the
dispatch-frame is NOT the discriminator (the "over-fill is an EMU artifact / fix dispatch"
hypothesis is FALSIFIED). Both worlds resume from shim dispatch; the difference is purely the
RESUME ORDER: HW producer-first (+2), EMU consumer-first (-62). **This is exactly what the
combinational-backpressure campaign targets -- the premise HOLDS.** The dispatch-axis detour is
dead.

**Refined target (sharp).** In EMU mode 1, consA is ALREADY HW-correct (+42 vs HW +43/+45).
The entire error is PROD: EMU +104 vs HW +43. Cause: in mode 1 the backpressure from consA's
stall reaches prod via natural FIFO-fill (+62cy propagation). Forcing the core edge pulled prod
to +57 (right direction) but starved consA to +179 because it ACCEPT-gated consA's input (of_d).
=> Build a combinational drain-back that pulls prod ~61cy earlier WITHOUT starving consA: model
the consA-stall -> prod-stall as a LOCK-RELEASE-HALT propagation (consA stalled on of_j-Produce
stops RELEASING of_d's consume-lock, so upstream sees the buffer full -- but consA keeps its
buffered of_d, not starved), not a data-accept block. That is the next build.

### MECHANISM NAILED (2026-07-03): the RESUME cascade must be combinational too

The HW puzzle: prod re-stalls FIRST (+43) despite being FURTHEST from the draining shim. The
only explanation -- HW's resume cascade is COMBINATIONAL. When the shim starts draining, the
freeing (lock releases: of_out drained -> of_j space -> consA produces -> of_d space -> of_src
space -> prod produces) propagates up the WHOLE chain in ~one cycle, so all cores resume
TOGETHER and then re-stall in producer->consumer PROGRAM ORDER (prod first, +2). EMU's cascade
is PER-HOP (one stage per routing cycle), so cores resume STAGGERED (consA first, nearest the
shim) and re-stall consumer-first (-62). The +62 is the per-hop resume-cascade propagation time
from shim up to prod.

**Unified root:** matching HW needs BOTH directions combinational -- the stall-propagation
(backpressure, which the mode-1 gate already does: consA lands +42 = HW) AND the
unstall-propagation (resume/freeing, still per-hop: prod lands +104 not +43). Both are the same
thing: the chain's lock/backpressure state must resolve to a FIXPOINT each cycle when a cold
terminal is in the path, not one hop per cycle. This is the "fixpoint pass" the Risk section
flagged.

**Proposed architecture (touches per-cycle lock/route resolution, blast
radius on all kernels, guarded by cold-terminal):** each cycle, over the precomputed
cold-terminal reach, iterate the transitive backpressure+freeing to convergence within the
cycle (bounded by chain length) so lock releases and stalls propagate end-to-end in one
emulated cycle instead of one hop. Keep the static reachability (built) as the topology; add the
per-cycle fixpoint as the dynamic resolver. Expected to land prod ~+43 (resume in program order)
AND close the starvation (same combinational freeing removes the drain-side over-feed).

### CHEAP CONFIRMATION (2026-07-03): resume-cascade is a pure propagation delay -- fixpoint validated

Traced the mode-1 resume cascade in the events timeline (pure analysis, no rebuild):
```
+0   shim START; memtile ports begin
+32  consA acquire-req
+42  consA LOCK_STALL          <- consA re-stalls, HW-correct (+43)
+66  memtile PORT_RUNNING_5    <- of_out drain finally starts
+94  prod acquire-req          <- prod's FIRST produce attempt
+104 prod LOCK_STALL           <- prod re-stalls, should be +43
```
**prod is idle (no acquire attempt) until +94** -- the freeing cascade takes ~94cy to propagate
one-hop-per-cycle up to it. consA is active from +32 and re-stalls normally at +42 (NOT starved).
So the +62 error is a pure resume-cascade propagation delay; collapsing it to one cycle (the
fixpoint) lands prod at program-order re-stall (~+43) WITHOUT touching consA. Same in-cycle
freeing starts of_out draining at ~+0 not +66 => starvation fix too. Fixpoint architecture
CONFIRMED. Building it now (guarded by cold-terminal + mode knob, mode 0 byte-identical).

## Artifacts
- Root-cause: `build/experiments/sp4a-drainthrottle/OFFSET-ROOT-CAUSE.md`
- Fix (a) code: `src/device/array/routing.rs` (`route_dma_to_tile_switches`,
  `build_mm2s_terminal_map`, `walk_to_terminal_s2mm`), uncommitted.
- Oracle harness: `build/experiments/sp4a-drainthrottle/{run_emu.sh,measure.py}`.
