# #140/#141: the PORT_RUNNING cadence metric was counting frame-records, not spans

**Date:** 2026-06-16
**Status:** Root cause found. Reframes #140 (was: stochastic DDR-delivery calibration)
and #141 (encoder is faithful, not the defect). The committed stochastic DDR
("phoenix") model is now known to reproduce a **metric artifact**, not real
hardware behavior.

## TL;DR

The `port-cadence-baseline.py` metric counted **trace frame-records** (every frame
that *names* a port slot), not **spans** (continuous runs of the port being
active). Held PORT_RUNNING levels are re-checkpointed in `cycles==0` frames every
time a *concurrent* signal toggles -- HW does this too -- so the frame count is
inflated and cross-contaminated by other signals' edge timing, and it manufactures
run-to-run **variance that does not exist on silicon**.

Measured correctly (span-based, idle-gap>2), NPU1 `add_one_using_dma` PORT_RUNNING
is **deterministic** across 15 HW runs:

| port | **HW** (15 runs) | EMU phoenix (removed) | **EMU default** (phoenix gone) | gated by |
|------|:----------------:|:---------------------:|:------------------------------:|:--------:|
| PORT_RUNNING_0 (recv<-shim)     | **1** (std 0) | 2 | **1 (match)** | shim DDR |
| PORT_RUNNING_1 (recv<-compute)  | **5** (std 0) | 6 | 6 (+1) | compute core |
| PORT_RUNNING_4 (send->compute)  | **3** (std 0) | 6 | 5 (+2) | compute core |
| PORT_RUNNING_5 (send->shim)     | **4** (std 0) | 4 | **4 (match)** | shim DDR |

The "DDR jitter" (slot0 `5.65 ± 0.48`) that the entire stochastic delivery model
was built to reproduce **is not present at the span level**. It was
re-checkpoint-timing noise in the broken metric.

**Localization (2026-06-16, later same day).** With the phoenix model removed
(now the default), the two **shim-DDR-gated** ports (slot0 recv<-shim, slot5
send->shim) match HW **exactly**; only the two **compute-core-gated** ports
(slot1 recv<-compute, slot4 send->compute) diverge. The phoenix model actually
made slot0 *worse* (2 vs 1) by perturbing a path that was already correct. This
is airtight evidence the residual gap is **not** DMA/DDR delivery -- it is the
compute core's buffer release/acquire phasing (see "The real #140" below).

## How we got here (the methodology trail -- three stacked errors)

1. **Bad oracle #1 -- in-process DMA byte-counter proxy.** Disagreed with the
   decoded trace on counts *and* direction-of-effect (said "Lever 1 helps"; the
   real path said it hurts). `stream_in`/`stream_out` FIFOs sit between the byte
   counter and the port.
2. **Bad oracle #2 -- raw `cycle_beat` port probe.** Even reading the true port
   signal, raw beat-runs (slot0 = 2 continuous spans) disagreed with the decoded
   trace (slot0 = 5). The transform was downstream.
3. **The metric itself.** `parse_trace` emits one event record per frame that
   names a slot; `port-cadence-baseline.py` counted those (unique `soc`). A
   continuously-held slot0 that spans 3 slot4 toggles -> 4+ slot0 frame-records
   -> "4 sub-bursts" that are really 1 span. The gap pattern `[26,27,9,4]` was
   *slot4's edge timing* bleeding into slot0.

Stale-`.so` trap: `cargo build -p xdna-emu-ffi` silently did not relink
`target/debug/libxdna_emu.so` (hardlink); `touch crates/xdna-emu-ffi/src/lib.rs`
forced it. The FFI/plugin path also runs the kernel at **col 1**, not col 0 like
the in-process path -- probes must auto-detect the active column.

See [[feedback_port_cadence_oracle_only_emu_decoded]].

## Why the encoder is NOT the defect (#141 is faithful)

Disassembling HW's real `add_one_using_dma` memtile trace (raw bytes from the
15-run baseline) to mode-0 commands and comparing to our EMU output: **identical
frame grammar** for every transition type --
- open-after-gap -> `S(slot, gap)` then `S(slot, 0)` (the two-frame open the
  skip-token spec flagged as "unsettled" -- it is exactly what HW does),
- hold -> `Repeat0/1`,
- concurrent join / re-checkpoint -> `M([held, joiner], 0)` then `M([held], 0)`.

Every difference is a *count* (deltas, repeat lengths, number of toggles) = the
underlying port-signal timing, not the encoding. And the round-trip is clean: our
EMU bytes decode (oracle B/E) back to slot0 = **2 spans** = its real 2-span signal
(re-checkpoints correctly persist). The HW reference frames in
`docs/superpowers/specs/2026-06-08-skip-token-held-level-encoding.md` (`[13..16]`)
confirm HW re-checkpoints held levels on foreign toggles -- so suppressing
re-checkpoints would be *wrong*; matching HW means matching the signal edges.

## The real #140 (deterministic, no RNG) -- compute-core release phasing

EMU inserts genuine `>2`-cycle idle gaps in PORT_RUNNING that HW does not, on the
two **compute-core-gated** ports only (slot0/slot5, shim-DDR-gated, match exactly):
- **slot4 (send->compute): EMU 5 sub-bursts vs HW 3** -- the headline.
- slot1 (recv<-compute): EMU 6 vs HW 5.

Aligned to the first begin (one run), the shapes are:
```
HW slot4:  [0,93](93)  gap51  [144,161](17)  gap57  [218,237](19)            = 3 runs
EMU slot4: [0,33] gap27 [60,86] gap40 [126,144] gap56 [200,218] gap48 [266,284] = 5 runs
```
HW **front-loads** a single 93-cycle continuous opening burst, then trickles;
EMU fragments that opening into [0,33]+gap27+[60,86] before settling. The
mechanism (confirmed by `XDNA_EMU_XFORM_PROBE`): the memtile MM2S is
compute-backpressured (compute `in1_prod` lock = 0 almost continuously), sending
one 8-word buffer per core-release, perfectly serialized to the core cadence
(slot4 edges ~64cy apart = the core per-buffer period). On silicon the core
releases/acquires buffers slightly earlier, so the small stream FIFO never fully
drains during the opening and the port stays continuously asserted. The
**steady-state** gaps already match (~50-56cy both); only the warmup/opening is
fragmented. The DMA/stream model itself is HW-faithful (FIFO depths 4/4/2 from
AM020, instantaneous backpressure). The empirical front-loading is real; its
*attribution* was corrected twice below.

**Refined root cause -- ATTEMPT 1 (WRONG, kept for the trail).** The first
refinement (2026-06-16) read the flat ~64-71cy/buffer (XFORM_PROBE: `in1_cons=2`,
consumption uniform 64/71/64/71) as "the core's whole per-buffer period is honest
instruction cost, and HW front-loads because its VLIW core runs the add-loop
faster (~2-3 vs ~7 cy/element) -- HW pipelining/forwarding that EMU over-charges
as hazard stalls." This framed the fix as **"Improve interpreter VLIW core-timing
fidelity (ILP/hazard modeling)."** It is **mechanically impossible** -- see below.

**Refined root cause -- ATTEMPT 2 (CORRECTED, 2026-06-16 cont.).** A code+ISA
scout (two Explore passes + chess-ELF disasm) overturned Attempt 1:

1. **EMU inserts ZERO load-use/RAW hazard stalls.** The `HazardDetector`
   (`src/interpreter/timing/hazards.rs`) is **dormant** -- `check_scalar_read` /
   `check_vector_read` / `check_operation` are never called in the execution path
   (grep-verified: only internal callers in `hazards.rs:258/260` and tests). It
   *records* writes but never *checks* reads. So there are **no hazard stalls to
   over-charge** -- Attempt 1's mechanism cannot exist.
2. **EMU per-bundle cost = 1cy** (`record_instruction(1)`, cycle_accurate.rs:960)
   **+ 1cy per lock transaction** (`record_stall(1)`, cycle_accurate.rs:955 -- the
   only `record_stall` in the executor). Load latency is a **deferred-write only**
   (changes which value a read sees, never the cycle count). EMU therefore
   **counts compiled bundles at the HW 1-bundle/cycle issue rate**, plus the
   lock-arb charge. Nothing else.
3. **The compute loop is SCALAR, Chess-hand-software-pipelined** (disasm of
   `add_one_using_dma/chess/.../main_core_0_2.elf`, `core_0_2` @ 0xe0): per 8-word
   buffer, 8 individual `lda` then 8 `st`+`add.nc #1` (0x170-0x1d6). The 8 loads
   are issued up front so the 7cy load latency is covered before the first `add`
   fires -- the compiler did the pipelining; EMU's deferred-write reproduces it
   faithfully. **There is no vectorized add-loop where forwarding would matter.**
4. **The dominant per-buffer cost is lock function-call overhead.** Each
   acquire/release is a `jl` to a one-instruction function (`acq`/`rel` @
   0x330/0x350) wrapped in 5+5 branch-delay slots ~= 13cy/call. Four lock
   calls/buffer ~= **52cy**, vs ~16cy of load/store -- which is where the observed
   64-71cy/buffer comes from. **HW runs the identical binary, so it pays the same
   overhead.** "Core runs the add-loop faster" was the wrong picture: most of the
   period is lock-call branch-delay slots, not adds.

**Corrected divergence candidates.** EMU's compute extent is ~47cy longer than HW
(span extent 284 vs 237) and fragments more. Since EMU faithfully executes the
compiled bundles at HW's issue rate, the core-period divergence can only be:
- **Lever A -- the +1cy/lock-arbitration charge** (`record_stall(1)`, added
  2026-06-07 to match an observed HW LOCK_STALL pulse). ~4 lock txns/buffer x ~4
  buffers ~= +16cy. May be over-applied, or the real HW pulse may overlap activity
  rather than serialize. Blast radius = every kernel with locks.
- **Lever B -- lock-acquire stall duration** driven by *when the DMA delivers /
  releases buffers* (stream/FIFO coupling). A DMA-timing interaction, NOT core
  timing.

**Open gap.** HW's per-buffer period was inferred from port spans -- see the A/B
measurement below, which resolved it.

## A/B split MEASURED (2026-06-16 cont. 2)

Built a controlled comparison from existing trace artifacts
(`build/experiments/gap140/ab_split.py`): EMU lock-arb **OFF**
(`baseline-pre-lockarb`, before commit `737f5505`) vs **ON** (`baseline-cycle-only`,
after it) vs **HEAD** (fresh `multirun --emu` on current code) vs **HW**
(`baseline-pre-lockarb` HW pair). Per-port span sub-burst counts (idle-gap>2):

| slot (CORRECTED direction) | EMU arb-OFF | EMU arb-ON | EMU HEAD | HW |
|---|---|---|---|---|
| 0  recv<-shim    | 1 | 1 | 1 | 1 |
| 1  recv<-compute | 6 | 6 | 6 | 5 |
| 4  send->compute | 1 | 1 | 5 | 3 |
| 5  send->shim    | 4 | 4 | 4 | 4 |

**Result 1 -- Lever A (lock-arb) is DEAD.** OFF and ON columns are *byte-identical*
span timelines (same harness, same day, only `737f5505` differs). The +1cy/lock-txn
charge changes compute-core cycle counts but does NOT touch the memtile port
cadence. Ruled out as a #140 lever. (Lever B, the lock-acquire/DMA-coupling
framing, is also not quite right -- see Result 3.)

**Result 2 -- PORT DIRECTIONS NAILED BY DECODED-TRACE CAUSALITY (and the
in-process probe mislabels them).** There was an interim mis-correction: the
2026-06-14 FSM instrumentation labels port1=send/port4=recv, and `5675bdd5`'s
commit message calls the port it changed (1->5) a "receive port", so this doc was
briefly edited to slot1=send/slot4=recv. **That is WRONG.** The decoded trace on a
SHARED absolute timebase (`build/experiments/gap140/ab_timeline.py`) shows the
pipeline firing in exact dataflow order in BOTH worlds:
```
DDR -> memtile -> compute -> memtile -> DDR
PORT_0   @0  ->  PORT_4  @27-35  ->  PORT_1  @113-129  ->  PORT_5  @190-225
```
A core cannot emit output before it ingests input, so the compute-port that fires
FIRST (PORT_4) is **send->compute (input)** and the later one (PORT_1) is
**recv<-compute (output)**. The MLIR dataflow confirms: `flow(shim->mt DMA0)`,
`flow(mt DMA0->compute)`, `flow(compute DMA0->mt DMA1)`, `flow(mt DMA1->shim)`.
**Correct, final: PORT_0=recv<-shim, PORT_4=send->compute, PORT_1=recv<-compute,
PORT_5=send->shim** (= the original prior-session labels). The 2026-06-14 FSM
probe's port-direction attribution was wrong -- the in-process probe misleads for
DIRECTIONS too, not just cadence; the decoded trace is the only oracle. (`5675bdd5`
changed PORT_4 = the memtile MM2S-0 *send-to-compute* flow; its "receive port"
naming came from the same mislabeling.)

**Result 3 -- the real divergence is slot4 (send->compute, INPUT delivery)
OPENING CLUSTERING, not lock-arb and not core-ILP.** Only slot4 diverges
meaningfully (slot0/5 match, slot1 +1). The `1->5` swing was commit `5675bdd5`
(beat-crossing PORT_RUNNING) -- so the 06-14 `slot4=1` is a *stale-semantics
artifact* (old `has_data()`-seed model held the port continuously "running"), NOT
a real "too smooth" state.

**The "two oracles disagree" dissolves under the decoded oracle.** Raw spans
(before idle-gap>2 grouping):
```
HW  slot4: [0,8] g1 [9,17] g1 [18,94] g50 [144,163] g56 [219,238]   (5 raw runs -> 3 groups)
EMU slot4: [0,33] g27 [60,86] g40 [126,144] g56 [200,218] g48 [266,284]  (5 raw runs -> 5 groups)
```
BOTH emit **5 raw beat-runs**. The "HW ~8-9 pulses" that made it look like EMU
*under*-emits came from the in-process `cycle_beat` probe (the known-misleading
path, [[feedback_port_cadence_oracle_only_emu_decoded]]), NOT the decoded oracle.
One signal only: HW and EMU both emit 5 input-beat-runs; the difference is **gap
pattern**. HW *clusters* the opening three runs (gaps 1,1 -> continuous `[0,94]`)
then settles ~50-56cy; EMU *spreads* the opening (gaps 27,40) then matches
steady-state. **Headline: HW streams the opening memtile->compute INPUT delivery
to fill both initially-free input slots (producer lock init=2) back-to-back; EMU
sends one 8-word buffer, waits ~27cy for the core to free a slot, then the next --
it does not exploit the input double-buffer depth at the opening.** Steady-state
matches; only the opening fill transient diverges. Next target: the memtile
MM2S-0 -> compute S2MM-0 opening delivery / lock-acquire timing (overlaps #132).

**Output path (PORT_1, recv<-compute) is the *close* port** (EMU 6 vs HW 5) -- the
secondary, not the headline. The headline is the INPUT-delivery port (slot4,
send->compute). The compute core's instruction timing itself is faithful (no
hazard stalls, 1cy/bundle); the gap is purely how the memtile *feeds* the core at
the opening.

## Fix landed: stream egress metered at AXI4-Stream rate (2026-06-16, commit `67987db3`)

Per-cycle DMA word throughput was `words_per_cycle=4` (the 128-bit tile data
MEMORY bus width) for ALL non-shim transfers -- including MM2S stream egress. But
the AIE2 inter-tile stream is a 32-bit AXI4-Stream = **1 word/cyc/port**. Direct
proof in the XFORM probe (active window ~cyc 7912): the memtile MM2S-0 (`ch6`)
advances ~4 words/cyc while the compute S2MM-0 (`c2ch0`) ingests ~1 word/cyc. The
MM2S bursts the memory-read rate into the shallow (correct, 2-4 word) FIFO, fills
both compute slots, then sticks at `56/64` for ~28cy -- that stall IS the slot4
gap.

**The fix** (`build/experiments/gap140/ab_split.py` measures it): a data-driven
`stream_words_per_cycle=1` threaded through archspec + the emulator
`DmaTimingConfig`; `do_transfer_cycle` picks the rate by the NARROWEST interface
crossed (shim<->DDR=shim; crosses a stream=stream rate 1; else memory bus 4).
Supersedes the deleted `MM2S_1WORD` env hack with a principled model.

**Corroboration:** three chained-BD interval tests asserted the 4-word rate for
MM2S transfers while citing HW `add_one slot0 on16 off1` in their own comments --
self-contradictory (`on16` = 16 cycles/16-word-BD = 1 word/cyc). The fix makes
them produce interval 17 (16 data + 1 bubble) = HW on16/off1 for the first time.

**Result -- real but PARTIAL.** slot4 (send->compute) opening continuous send
**33 -> 46cy** (toward HW's 94), first stall **27 -> 16cy**; steady-state period
preserved (gaps ~40-56 vs HW ~50-56). But slot4 is **still 5 sub-bursts vs HW 3**
-- a 16cy stall still splits the opening at ~46cy. slot0 (recv<-shim) shifted
1->2 sub-bursts, structurally TOWARD HW (HW slot0 is also 2 raw spans `[0,50]
g1 [51,68]`) but with a 7cy gap where HW has 1. 3499 lib tests green.

**Residual (open):** the egress rate was *a* factor, not the whole opening
transient. slot4 (memtile->compute) is fed by slot0 (shim->memtile); the opening
is the **shim->memtile->compute 2-stage relay fill**. The memtile can only
forward what the shim has delivered, so the 46cy cap is the 2-stage startup, not
the egress rate. That relay-fill transient (and the slot0 7-vs-1 gap) is the next
target.

## Consequences / decisions

1. **The stochastic DDR ("phoenix") model modeled a ghost -- REMOVED
   (2026-06-16).** HW PORT_RUNNING is deterministic (std 0, 15 runs).
   `BurstParams::AIE2_DDR_PHOENIX`, the seeded PRNG, system-entropy seeding, the
   100-run calibration -- all reproduced variance silicon does not have, *and*
   made slot0 worse (2 vs 1). `src/device/dma/burst.rs` deleted; the
   `XDNA_EMU_DDR_PROFILE`/`_DDR_BURST_*`/`_DDR_SCRIPT` env vars and the
   `XDNA_EMU_MM2S_1WORD` egress-rate experiment (Lever 1, also moot once the gap
   localized to the core) removed with it. `bd_switch_bubble_cycles` (the
   HW-confirmed 1-cycle per-BD bubble) is unrelated and kept. Default-path cycle
   accuracy is unchanged (it was off by default); 3514 lib tests green.
2. **The metric is fixed:** `tools/port-span-cadence.py` (span-based, oracle B/E,
   idle-gap>2). `port-cadence-baseline.py` is superseded for cadence work.
3. **`docs/known-fidelity-gaps.md` row 50** (PORT_RUNNING recv-port) rewritten:
   the gap is real but *deterministic*, the phoenix "1-sigma band-match" was an
   artifact, and the root cause is the lock-arb / DMA-coupling pair (Levers A/B)
   -- NOT VLIW ILP/hazard under-modeling.

## Scaffolding landed (default-off diagnostics for the implementation pass)

- `XDNA_EMU_XFORM_PROBE` (coordinator): per-cycle memtile/compute DMA + lock +
  port-beat timeline, auto-detects the active column. `[XEDGE]` logs the exact
  PORT_RUNNING level edges fed to the encoder. **Kept** -- it is the tool for the
  core-phasing characterization (it shows exactly when each port stops beating
  and what the DMA/locks are doing).

The `XDNA_EMU_DDR_SCRIPT` (scripted delivery) and `XDNA_EMU_MM2S_1WORD` (egress
rate) experiment levers were removed with the phoenix model -- both targeted DDR
delivery / egress rate, which the localization ruled out as the source.

All remaining diagnostics gated and default-off; 3514 lib tests green.

## HW disambiguation (2026-06-16 cont. 3): H-A confirmed, real gap = PORT_STALLED

The "stream-rate fix" left a residual (slot4 5 vs HW 3 sub-bursts) framed two ways:
**H-A** -- EMU's per-beat PORT_RUNNING is correct and the divergence is
consumer-pacing/word-flow; **H-B** -- HW *holds* PORT_RUNNING through backpressure
where EMU drops it. The existing single-event capture (PORT_RUNNING only) and the
toolchain docs could not decide it, so we ran the decisive HW experiment: add
`PORT_STALLED_4` (+0/1/5) to the memtile trace capture via the
`XDNA_TRACE_MEMTILE_EVENTS` env override (read by emu-bridge-test.sh:1765; the
`PORT_RUNNING_4` port-selection is shared, so `PORT_STALLED_4` reads the same
physical port) and decode both on real NPU1.

**Verdict: H-A. PORT_RUNNING is already correct.** On HW the memtile send port's
`PORT_RUNNING_4` and `PORT_STALLED_4` are **perfectly complementary** -- every
PORT_RUNNING gap (`g27`, `g58`, `g67`) is exactly filled by a PORT_STALLED
interval; together they tile `[33,271]` continuously (port engaged the whole
transfer, alternating beating <-> backpressured). HW's RUNNING **drops** during
backpressure, exactly like EMU's per-beat model. (Had we guessed H-B we'd have
broken correct semantics.) The original #140 PORT_RUNNING sub-burst residual is a
small opening-timing detail (EMU stalls ~4-10cy earlier -> one extra group);
semantics are sound.

**The real gap the experiment exposed: EMU emitted ZERO PORT_STALLED on
circuit-routed DMA ports.** `cycle_stalled` was set only in the packet-route path
(`step_packet_routes`); circuit routes (most inter-tile DMA, incl. memtile
MM2S->compute) skipped it silently. So HW shows complementary RUNNING/STALLED
tiling and EMU showed RUNNING with silent gaps.

**Fix (commit `a02187e4`).** `StreamSwitch::mark_stalled_ports()`, run once after
all routing each cycle: a port that holds data but saw no beat this cycle was
backpressured -> `cycle_stalled |= has_data() && !cycle_beat`. Evaluated
*post-routing* (the local route runs in two passes around inter-tile propagation)
so it reflects the final `cycle_beat`, keeping RUNNING and STALLED exclusive. A
first attempt set the stall inside the circuit route's `!can_pop` branch, but the
per-cycle probe (`s4st` readout added to XFORM_PROBE) showed it co-asserting with
`cycle_beat` during FIFO-fill (the slave port beats on DMA *push* while the route
can't *forward*) -> RUNNING/STALLED overlapped. The post-routing `!cycle_beat`
definition removed all overlap (0 cycles with both set) and reproduces HW's
complementary tiling. `cycle_stalled` is trace-only (consumed solely by
PORT_STALLED emission), so data movement, cycle counts, and PORT_RUNNING baselines
are unchanged; the default corpus capture (PORT_RUNNING only) is untouched. 3499
lib tests green.

**Validation oracle:** `build/experiments/gap140/disambig_stalled.py` (HW vs EMU
RUNNING/STALLED overlay); HW capture under
`build/bridge-test-results/20260616/add_one_using_dma.chess.hw`.

**Open residual:** the small opening-timing difference (EMU one extra RUNNING
burst; EMU first stall ~17cy vs HW ~27cy) -- a fine DMA/double-buffer-fill detail,
not semantics. Steady-state (`~9-10cy` bursts, `~56-63cy` stalls) matches HW.

## Confirm-first on the opening residual (2026-06-16 cont. 4)

Maya pushed back on parking it at a threshold: byte-identical is the north star,
and a *consumer-pacing* difference is concerning because it implies our **core**
timing model is off, not just a DMA cosmetic. Agreed -- so we ran a confirm-first
probe before any model change.

**Hypothesis tested:** "EMU fills one compute input buffer before backpressuring;
HW overlaps both (double-buffer)." If true, at the first slot4 (memtile MM2S0)
stall the compute `in1_cons_prod_lock` would still be `1` (one buffer free).

**Result: DISPROVEN.** Per-cycle XFORM probe (`build/experiments/gap140/xform_probe_exclfix.log`),
compute tile (0,2), `c2lk[0]` = `objFifo_in1_cons_prod_lock` (init 2):

```
cyc=7937  s4=1 s4st=0   c2lk=[1,0,2,0]   slot4 RUNNING, in1_prod=1
cyc=7954  s4=1 s4st=0   c2lk=[0,0,2,0]   in1_prod hits 0: S2MM took BOTH buffers
cyc=7967  s4=1 s4st=0   c2lk=[0,1,1,0]   core consuming + producing (out_prod 2->1)
cyc=7981  s4=0 s4st=1   c2lk=[0,1,1,0]   FIRST slot4 STALL, in1_prod=0
```

At the first stall `in1_prod = 0` -- **both** compute input buffers were already
filled. EMU does exploit the double-buffer depth. The burst width is **not** gated
by buffer count, because the core consumes concurrently (cons-lock and out-prod tick
during the burst). The earlier "10 ~= 8+2 vs 19 ~= 16+3" arithmetic was coincidence.

**The real residual is consumer-drain cadence.** Full HW `PORT_RUNNING_4` spans
(two independent persisted silicon runs agree, `20260614` + `baseline-cycle-only`):

```
HW  durs: 8, 8, 14, 61, 19, 19      <- one 61cy contiguous burst at the opening (cyc 33->94)
EMU durs: 17, 16, 10, 10, 10, 9, 18 <- broken into ~16/10cy bursts with backpressure stalls
```

HW pipelines nearly the whole 64-word transfer in one sustained burst once warm,
because its core drains input buffers fast enough to keep accepting. EMU stalls
when both compute buffers transiently fill -- i.e. EMU's core is not freeing input
buffers at HW's opening-warmup rate.

**Why this matters (Maya's point):** this is in tension with the prior conclusion
"the core instruction timing is faithful." That conclusion was about *bundle-issue
rate* and *lock function-call overhead* (both correct and HW-calibrated); it did
**not** test whether the core's per-buffer **drain cadence** sustains a continuous
downstream stream the way HW's does. The bridge comparator (raw re-checkpoint
records) rates `PORT_RUNNING_4` 8/8 OK within +1cy, so this is sub-threshold -- but
we don't want a threshold, we want identity.

**Status: DOCUMENTED, next-session target.** Confirm-first paid off twice: it ruled
out a non-fix (buffers are already overlapped) and pointed the frontier at the core.

**Resume direction:** instrument the compute core's per-buffer drain cadence --
cycles from `in1_cons_cons_lock` acquire to `in1_cons_prod_lock` release per 8-word
buffer -- and diff EMU vs HW across the opening warmup. The question is whether
EMU's core frees input buffers at HW's rate before steady-state, or whether a
warmup/pipeline-fill effect lets HW's core run ahead. Artifacts on disk:
`build/experiments/gap140/{xform_probe_exclfix.log, disambig_stalled.py}`;
HW refs `build/bridge-test-results/{20260614,baseline-cycle-only}/add_one_using_dma.chess.hw/`.
