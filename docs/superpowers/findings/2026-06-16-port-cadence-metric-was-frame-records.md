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

**Open gap.** HW's per-buffer period was inferred from port spans, not directly
measured, so the A-vs-B split is unresolved -- that is the decisive next
measurement. **Output path ruled out** (`out1_prod` never 0, `out1_cons` always
0). aiesim ruled out as oracle (PORT_RUNNING-silent). HW PORT_RUNNING spans
(slot1/slot4 -> 5/3) are the only oracle; preserve the #135 steady-period match.

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
