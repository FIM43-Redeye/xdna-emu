# SP-4a fixpoint surgery -- RESUME POINTER (post-compaction handoff)

**Date:** 2026-07-03  **Issue:** #140 SP-4a  **Status:** checkpoint committed;
surgery is the NEXT build. Full campaign narrative + all measurements live in
[`2026-07-03-combinational-backpressure-campaign.md`](2026-07-03-combinational-backpressure-campaign.md)
-- read that first. This file is the crisp "where we are / what's next" so the
transition doesn't lose state.

## The one-line state

Static combinational-backpressure infra is BUILT + COMMITTED (mode 0 default =
byte-identical, 3585/0 green). The lean-oracle offset is still -62 at mode 0 and
does not reach +2 by gating alone. HW capture + full diagnosis are done and prove
the remaining fix. NEXT = the invasive per-cycle cold-chain fixpoint.

## What is committed (the checkpoint)

- `src/device/array/backpressure.rs` -- graph-reachability combinational
  backpressure. Reverses route_graph's fabric + relay (E2/E3/P2) edges, BFS-back
  from each cold terminal S2MM, gates the reachable set. Env `XDNA_EMU_SP4A_BP_MODE`
  (default 0 = off/byte-identical; 1 = accept-gate interior S2MM; 2 = + drain-gate
  MM2S). Cached on `TileArray.backpressure_reach`, invalidated on reset.
- `route_streams` builds it lazily; `route_dma_to_tile_switches` consumes the
  drain gate; `route_tile_switches_to_dma` consumes the accept gate.
- Parked drain-throttle knobs (timing.rs/channel.rs/stepping.rs, default off).
- HW harness `build/experiments/sp4a-drainthrottle/run_hw.sh` (gitignored).

## HW ground truth (the oracle, re-measured on silicon 2026-07-03)

Lean kernel, real NPU via `run_hw.sh` (`env -u XDNA_EMU`, same runner+decode):
- shim S2MM_0 START soc=428654; prod first LOCK_STALL **+43**, consA **+45**;
  offset **+2**; starvation **+13**. (Matches SP4A-HW-TARGETS.)
- HW dispatches the shim LATE and stalls just after -> HW ALSO measures a RESUME,
  same frame as EMU. Dispatch-frame is NOT the discriminator (that hypothesis is
  dead). The difference is the resume ORDER.

## The mechanism (nailed, do NOT re-derive)

- Both worlds over-fill then resume at shim dispatch. HW's resume cascade is
  COMBINATIONAL: the freeing (shim drains of_out -> of_j space -> consA produces
  -> of_d space -> of_src space -> prod produces) propagates up the whole chain in
  ~one cycle, so all cores resume TOGETHER and re-stall in producer->consumer
  PROGRAM ORDER (prod +43, consA +45, +2).
- EMU's cascade is PER-HOP (Phase 3 `step_all_dma` advances each DMA one BD-word
  per cycle; Phase 4 `route_streams` one hop per cycle). So the freeing crawls up
  one stage per cycle -> cores resume STAGGERED (consA first, nearest the shim) ->
  re-stall consumer-first (-62).
- Confirmed by the mode-1 resume timeline: consA active +32, re-stalls +42
  (HW-correct, NOT starved); **prod is idle -- no acquire attempt until +94** --
  because the freeing takes ~94cy to reach it; prod re-stalls +104. The +62 error
  is a PURE resume-cascade propagation delay.

## What gating alone can/can't do (measured)

- mode 1 (accept-gate interior S2MM): consA lands +42 (=HW), prod +104. Offset
  still -62. Backpressure/order lever works but can't pull prod.
- Extending reach to prod requires a through-core edge (CoreLockRelay does NOT
  emit consA's of_d->of_j edge for this kernel). Forcing it (throwaway test, now
  removed) flipped offset to +122 -- OVERSHOOT, because accept-gating consA's INPUT
  (of_d) starves it (of_d is prod's output on a LINEAR pipe -- can't gate prod's
  output without starving consA's input).
- => Pure static gating cannot hit +2. The resume must be made combinational.

## THE NEXT BUILD: per-cycle cold-chain fixpoint (the surgery)

Per-cycle phase order (src/device/array/routing.rs, the `step`-side fn ~line 140+):
Phase 2 resolve_lock_requests -> 2.5 reset drain counters -> **Phase 3
`step_all_dma`** (each DMA one BD-word) -> **Phase 4 `route_streams`** (one hop) ->
4.5 cascade -> 5 adaptive tick. The one-DMA-stage-per-cycle limit in Phase 3 IS
the ~94cy cascade.

**Goal:** when a cold terminal is armed/draining, make the freeing propagate up its
reachable cold chain within ONE emulated cycle, so prod resumes with consA and
re-stalls in program order (~+43). Same combinational freeing starts of_out
draining at ~+0 not +66 -> also closes starvation (+1491 -> ~+13). Both oracles.

**Open design question (resolve with Maya before/while building):** how to
multi-step the cold-chain DMAs to convergence within one cycle WITHOUT breaking
lock/buffer consistency for the rest of the array. Candidate approaches:
1. Bounded per-cycle fixpoint over the cold reach: repeat {step cold-chain DMAs;
   route their words} up to chain-length K until no progress. Needs a
   step-only-these-channels variant (step_all_dma steps everything; iterating it
   over-advances all tiles).
2. One-shot resume flush at the cold->armed transition (rare, once/kernel): at the
   dispatch cycle, run K extra cold-chain resolution passes. Risk: over-advances
   the array timer by K once.
Both must be guarded by the cold-terminal predicate + the mode knob so mode 0 and
all no-cold-terminal kernels stay byte-identical. Watch: don't create a per-cycle
infinite loop / borrow tangle; blast radius = shared route/lock resolution.

**Gate each step against:** lean oracle (`build/experiments/sp4a-drainthrottle/`,
`run_emu.sh` for EMU, `run_hw.sh` for HW, `measure.py`; targets prod ~+43, offset
+2, starvation ~+13); `cargo test --lib` green; recv slots [16,16,16,16];
byte-identical data; steady-state of_out 64cy/obj; send-cadence corpus no-regress.

## Reproduce the current baseline

```
cargo build -p xdna-emu-ffi
cd build/experiments/sp4a-drainthrottle
XDNA_EMU_SP4A_BP_MODE=1 ./run_emu.sh 0 0 chk   # consA +42, prod +104, offset -62
./run_hw.sh chk_hw                              # HW: offset +2, starvation +13
```
