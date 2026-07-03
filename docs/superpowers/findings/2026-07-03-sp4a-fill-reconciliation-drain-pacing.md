# SP-4a: both pipelines pre-fill; the divergence is shim-drain cold-start pacing -- BUT that is a SEPARATE gap from the cross-domain offset

**Date:** 2026-07-03  **Issue:** #140 (SP-4a cold-start fill-state)
**Status:** DRAIN-SIDE MECHANISM NAILED. **PARTIAL CORRECTION (2026-07-03, same
day):** the drain-side reconciliation below stands and is now *fixable* (a shim
S2MM drain throttle kills the dump + moves shim starvation +1491->+190; see
`build/experiments/sp4a-drainthrottle/FINDING.md`). But this doc's claim that
fixing the drain pacing would carry the **prod->consA offset -52 -> +2 is
EMPIRICALLY FALSE** -- the offset is PINNED at -62 across every drain-throttle
config (10x range of cooldown/decay). The offset is a SEPARATE upstream gap
(forward-fill / core-engagement order: EMU engages consA before prod, ~600cy
late; HW engages prod first, ~22cy after shim START). The 2026-06-29
SP4A-LEAN-FINDING was right that the offset lever is the memtile relay cadence,
NOT the shim cold-start. Neither removing (`cold=0`) nor metering the drain
touches the offset. **Two independent gaps; the offset is now its own campaign
(Maya: offset first, park the drain fix env-gated at default 0).**

Converges with the 2026-06-29 finding
(`2026-06-29-coldstart-headstart-trace-baseline.md`) on the DRAIN side.
**Working evidence (gitignored):** `build/experiments/pathA-cntr-spike/`:
`w5-window/` (delayed-drain fill depth, HW+EMU), `w6-hopladder/` (hop-scaling),
`w8-drainpace/` (drain-gated consA + dispatch-latency variance).

## The one-line result

HW and EMU **both pre-fill the pipeline to full** before the drain starts. The
"HW empty at baseline" that three prior sessions chased is a **trace artifact**
(the trace starts at the baseline and cannot see the pre-baseline fill). The
real, only divergence is **shim-drain cold-start pacing**: HW's shim S2MM
backpressures the drain at cold-start so the memtile delivers gradually (RAMP);
the EMU's doesn't, so the memtile dumps the backlog (~3.5 objects in one 228cy
span). Forward path and steady-state are bit-faithful.

## Why in-core cntr, not trace (Maya's call, vindicated)

The trace's blind spot -- it begins at `start_trace`/`DMA_S2MM_0_START_TASK`, so
the entire pre-baseline fill is invisible -- is exactly what produced the "HW
empty" misdiagnosis. Prior sessions inferred "empty" from the shim starving at
t+14, conflating **shim-ingress-empty** with **pipeline-empty**. In-core cntr
(free-running tile timer, read via the `read_cntr.s` MOV_CNTR stub) sees the
fill directly:

| observable (in-core cntr) | HW | EMU |
|---|---|---|
| producer fill depth pre-drain (`w5`) | **12** | 10 |
| consA fill depth pre-drain (`w5`,`w8`) | **5** | 4 |
| pipeline upstream at drain start | **FULL** | **FULL** |

Both pre-fill. Shim-ingress is empty in both (nothing delivered yet); that is not
the same as the pipeline being empty. **This is why every prior fill-state fix
failed: they aimed to make the EMU pipeline *empty* to match a phantom -- moving
it away from HW, which is also full.**

## The divergence: shim-drain cold-start pacing (RAMP vs DUMP)

Localized by prior work's span decode (span-count-verified) and consistent with
all in-core data. Initial of_out send port (memtile->shim), begin@rel-baseline:
- **HW slot5:** `(1103,48) (1152,64) (1217,64)...` -- first object delayed to
  +1103, then one object per ~64cy as **separate spans** (per-object gaps).
- **EMU slot5:** `(534,228) (763,64)...` -- first object early (+534), then
  **~3.5 objects in one contiguous 228cy span** (no per-object gap), then 64cy.

Mechanism: HW's shim S2MM DDR-write path warms cold (NoC arbitration + DDR row
open), so it accepts the first word only ~+1103 and meters subsequent objects
one per DDR-write; the memtile is backpressured into a gradual ramp. The EMU's
shim S2MM accepts the pre-filled backlog at full rate with no per-object
backpressure -> contiguous dump. The dump is what makes the EMU shim starve only
at +1492 (drains a delivered backlog) vs HW +14 (nothing delivered yet).

## Hop-scaling: terminal-dominated, NOT relay-accumulated (`w6`)

A ladder of relay depths (producer->shim direct; +1 memtile hop; lean 1-consumer;
2-consumer) measured the cold-start via the producer stall. It does **not** grow
with hop count (lean stall ~968cy mean vs 2-consumer ~781cy -- within noise). The
forward relay is fast (producer fast-fills 7; the near hops warm during the fill
and overlap away). So the cold-start is concentrated at the **terminal (shim)**,
not summed across the relay. aie-rt grounds the per-channel mechanism (each DMA
channel's first BD pays an unamortized STARTING cost against an empty 4-deep
START_QUEUE: fetch BD -> decode 4-D descriptor -> acquire lock), but the
observable is terminal-gated.

## The array is deterministic; all jitter is off-array (Infinity Fabric)

- **Cold-start magnitude is variable** (`w6`: 765-1386cy over 5 runs, +/-23%).
  Since the AIE array is deterministic, this jitter is entirely the shim S2MM ->
  DDR/NoC (Infinity Fabric) terminal, not the array.
- **The array-side fill is exactly deterministic**: across 6 `w8` runs the
  dispatch latency swung 24.7k-40.7k cy yet consA filled to **exactly 5 every
  run**. The fill does not depend on the (jittery, off-array) dispatch timing.

This is the clean model split: **array side deterministic + faithful (leave it
alone); terminal (shim/DDR) carries the cold-start pacing AND the jitter.**

## Dispatch-latency surprise (characterized, deferred -- see known-fidelity-gaps)

`w8` also exposed a large gap: the base latency from core-start to the first
drain dispatch is **~25-32k cy on HW (mean 30.5k, +/-16%, occasional 40k)** vs
the EMU's ~673cy. Decomposed: a ~25-32k fixed base (host->NPU launch + firmware
startup, jittery -> off-array) + **~112cy per runtime-sequence instruction**
(controller processing rate, from W5's 1024-write32 pad) + scales with CDO size.
It is **orthogonal to the SP-4a oracle** (both pipelines are equally full when
the drain dispatches, whenever that is) and off-array, so the EMU correctly does
not model it (modeling it would inject host jitter without improving array
fidelity). Not yet decomposed into host-launch vs firmware-config -- logged in
known-fidelity-gaps for a later dedicated marker-probe.

## The fix + its oracle (any drain-pacing change must satisfy ALL)

Model the shim S2MM cold-start **backpressure/ramp** on the drain so the memtile
meters of_out gradually instead of dumping. **DRAIN-SIDE oracle only** (the
offset is NOT part of this -- see the correction at the top):
- forward slots 0+4 stay `[2,2,...]` (do NOT touch the faithful forward path);
- return/drain steady-state stays 64cy@~65cy;
- cold-start initial burst moves: of_out (slot5) 228 -> ~64; of_j (slot1) 17->~11;
- shim S2MM first starvation +1492/+1684 -> ~+14 (lean oracle);
- `cargo test --lib` green + bridge-corpus spot-check, no regression.

The `prod->consA offset -52 -> +2` is a SEPARATE oracle for the upstream
core-engagement campaign, NOT satisfiable by the drain throttle (proven:
build/experiments/sp4a-drainthrottle/). The drain throttle is implemented and
PARKED env-gated (`XDNA_EMU_S2MM_COLD_COOLDOWN`/`_DECAY`, default 0 = off) until
the offset campaign completes and it can be calibrated + landed without conflating
the two.
