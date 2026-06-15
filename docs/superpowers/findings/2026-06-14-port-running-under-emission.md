# PORT_RUNNING memtile under-emission — root cause + fix (task #140)

**Date:** 2026-06-14 (root-cause + model fix landed)
**Kernel:** `add_one_using_dma` (memtile 0,1; event ports 0,1,4,5 active)
**Status:** root cause confirmed; model fix landed; two residual gaps scoped

## The signal

The stochastic-aware comparator flagged memtile `PORT_RUNNING_*` as
**deterministic** (HW std 0) yet EMU under-emits. Per-port raw record counts
(HW run_01 vs add_one_using_dma.chess.emu) read as ~2:1 under-emission. The
first-pass framing (encoding mismatch + "EMU delivers at half granularity")
was only half right and is superseded below.

## What the raw model actually produces (the reframe)

Instrumenting the coordinator's port-edge emission (`cycle_beat`/`cycle_active`
transition site) for the in-process run gives the model's raw running
intervals *before* trace encoding:

| event port | role | model intervals (before) | HW pulses |
|------------|------|-------------------------:|----------:|
| 1 | MM2S -> compute (send) | 8 | 8 |
| 5 | MM2S -> shim (send)    | 4 | 4 |
| 0 | S2MM <- shim (recv)    | 1 | ~5-6 |
| 4 | S2MM <- compute (recv) | 1 | ~8-9 |

So the model was **already correct on the MM2S send ports** (1, 5) and wrong
**only on the S2MM receive ports** (0, 4), which sat in one continuous running
interval where HW toggles many times. The "EMU half-granularity on all ports"
read was an artifact of mis-pairing trace records.

## Root cause (confirmed)

Send vs receive marked port activity differently:

- **Send (MM2S) ports** set `cycle_active` in `StreamPort::push()` — true only
  on cycles where a beat is actually sent. Correct: they idle under
  backpressure.
- **Receive (S2MM) ports** had no `pop()`-side marking; their activity came
  entirely from `begin_routing_cycle()` seeding `cycle_active = has_data()`. So
  a receive port was "running" whenever its FIFO merely *held buffered data*,
  even on cycles where no beat crossed it.

The compute MM2S delivers in 8 discrete 1-2 cycle bursts (FSM log), but the
memtile receive port stayed continuously active because residual FIFO data kept
`has_data()` true across the inter-burst gaps. **The model conflated "holds
buffered data" with "running."** HW-faithful semantics: *running* = a beat
crosses this cycle; *stalled* = holds data but blocked; *idle* = no data.

## The fix (landed)

`cycle_active` has a second consumer — adaptive clock gating
(`routing.rs` Phase 5), for which "holds buffered data" is the *correct* signal
(a backpressured stream switch is awake, consuming clock). So `cycle_active`
was left untouched and a distinct per-cycle signal was added:

- New `StreamPort::cycle_beat` — set true on `push()` **and** `pop()` (any beat
  crossing), reset (no `has_data()` seed) at `begin_routing_cycle()`.
- Coordinator PORT_RUNNING/IDLE now follows `cycle_beat`, not `cycle_active`.

Result (same run, after fix): receive port 4 went **1 -> 5 intervals** (toward
HW's ~8-9), send ports 1/5 **unchanged-correct** (8/4), clock gating untouched.
Full lib suite green (3515 + 2 new `cycle_beat` tests). Low blast radius.

## Residual gaps (scoped, not yet closed)

1. **eport 0 (S2MM <- shim) still 1 interval.** This port idles on HW because
   the shim delivers host-DDR in bursts; that is exactly what the
   parameterizable DDR burst model (`src/device/dma/burst.rs`, default-off)
   introduces. Enabling/calibrating burst is the lever here, not a port fix.
2. **eport 4 at 5 vs HW ~8-9.** Overlapping bursts at the receive FIFO bridge
   some inter-burst gaps that HW keeps separate. Fine calibration (FIFO depth /
   inter-tile latency), secondary.
3. **Encoding factor (separate from the model).** HW emits 2 records per pulse
   (B/E collapse at the sub-tick HW pulse width); EMU emits ~1 record per
   running interval. This is a uniform ~2x factor in raw record counts and is a
   trace encoding/decoder-semantics question, arguably a comparator concern
   rather than a model bug. The stochastic comparator's level-event handling
   should account for it (interval/duration-aware, not raw count).

## Fresh-decode verification (current code, both fixes)

Re-decoded `add_one_using_dma` EMU with the `cycle_beat` fix via the in-tree
decoder (same backend as the HW captures), to stand on verified ground rather
than the earlier stale bridge artifact:

| port | HW raw | HW iv | EMU raw | EMU iv | note |
|------|-------:|------:|--------:|-------:|------|
| PORT_RUNNING_1 | 16 | 8 | 8 | 8 | send port -- exact match |
| PORT_RUNNING_5 |  8 | 4 | 4 | 4 | send port -- exact match |
| PORT_RUNNING_0 | 11 | 6 | 4 | 3 | recv<-shim: 2->3 (burst gap) |
| PORT_RUNNING_4 | 16 | 8 | 5 | 5 | recv<-compute: 1->5 (calibration) |

Confirms the encoding reality empirically: **HW emits ~2 records per running
pulse** (a same-soc begin/end pair from its ~1-cycle pulse), **current EMU emits
1 record per running interval** (the assert; the held span renders as a single
record through the in-tree decoder). This is a benign representation difference,
not a behavioural mismatch -- interval counting normalises it, and the two
send ports become exact matches. The `cycle_beat` fix is confirmed on fresh
code (recv ports 1->5 and 2->3). Stochastic verdict on fresh EMU: 3-of-10 OUT
(was 6-of-10 raw), all three now *under*-emission: PORT_RUNNING_0 (-4.4 sigma),
PORT_RUNNING_4 (-3.0 sigma), STREAM_STARVATION_0 (-1.0 sigma). The earlier
PORT_RUNNING_4 +3.9 sigma was stale-artifact noise (fresh is -3.0 sigma).

## Why deterministic (vs the stochastic STREAM_STARVATION)

PORT_RUNNING toggling is governed by structural BD/buffer/lock-handshake
boundaries, so it is deterministic across runs (std 0). This is a different
root cause from the shim STREAM_STARVATION stochasticity (memoryless DDR
arrival jitter) — which is exactly why the stochastic-aware comparator was able
to separate them.
