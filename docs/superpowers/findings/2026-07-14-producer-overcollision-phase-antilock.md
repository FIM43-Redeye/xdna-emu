# The producer "over-collision" is a DMA granule-fetch phase anti-lock + round-robin over-loss

Date: 2026-07-14
Branch: master (post `feat/core-memory-stall-model` merge)
Hardware: real Phoenix NPU1 (AIE2), clean steady-state probe, 3 runs, run-to-run identical.
Cross-checked by Codex (adversarial, empirical) and a dedicated EMU-internals investigation.

## Summary

The roadmap carried a "producer over-collision: EMU 17 stalls vs HW 1, ~10x" as an open bug.
It is neither 10x nor what it looked like. On a clean steady-state kernel the producer core
stall rate is **EMU 30 vs HW 18-20 per run (~1.6x)**, and it is the net of **two compensating
model errors**:

1. **EMU generates ~16x FEWER core-vs-DMA bank conflicts** (68 vs ~1100 CONFLICT_DM_BANK
   cycles/run) -- a deterministic **phase anti-lock**, not a density error.
2. **EMU's per-bank round-robin loses 44% of the conflicts it does have**, where HW gives the
   core near-priority and loses only **1.6%** (the DMA absorbs the loss via egress-FIFO slack,
   `MM2S_STARVATION = 0`).

Few-but-lossy (EMU) vs many-but-rarely-lost (HW) partly cancel to a ~1.6x net. The cancellation
is not robust (of_q0_rich's 16 short-object boundaries stack it to the "17" that started this;
the reference-capture "1" was a coverage undercount -- the same 16/9 trap as the width finding).

## The clean measurement (producer_probe, immune to Q=0 late-arming)

Generator `tools/experiments/producer_probe.py`: a compute tile (0,2) MM2S self-loops draining
a 256-word `dma_buf` (logical bank 0) while a DENSE march-store core (`march_buf[i]=i`, 1
store/cycle, `march_buf` logical bank 0) runs concurrently. Variants idle / apart (march in
logical bank 2, control) / collide (same logical bank). STALLED_LOCK->FINISHED_BD brackets each
transfer (per the width finding's method). Robust raw totals (never the buggy per-bracket
normalization -- see Traps):

| variant | EMU MEMORY_STALL | HW MEMORY_STALL | EMU CONFLICT | HW CONFLICT |
|---|---:|---:|---:|---:|
| idle    | 1  | 1  | 3   | 7    |
| apart   | 0  | 0  | 0   | 0    |
| collide | 30 | 18-20 | 68 | 1098-1103 |

collide-apart isolates the producer core-vs-MM2S rate. apart=0 confirms the march self-conflicts
zero and only the same-logical-bank overlap matters.

## Error #1: deterministic phase anti-lock (the 16x)

Both sides are individually faithful:
- **Core:** the ELF march is a zero-overhead single-bundle hardware loop (`main_core_0_2.elf`
  `0x270`: `st r24,[p0],#4; add`, `ls==le`, `lc=1500`) = 1 store/cycle. EMU demands a bank every
  march cycle (28,159 demand cy = 85% of window), matching the ELF. Core density NOT too sparse.
- **DMA:** MM2S granule fetch = one 1-cycle bank pulse per 4 drained words (egress FIFO 12,
  stream 1 word/cy), phys bank alternating per granule. EMU ~1024 granules, 531/531 banks --
  matches the 16 B / 4-beat width finding.

Both the core (per-store) and the DMA (per-granule) alternate the two interleaved physical banks
of logical bank 0 with **period 8 at the same rate**. In the emulator's fixed step schedule the
loop-entry transient sets a ~4-cycle (half-period) relative offset and PINS it: the DMA fires its
bank-X pulse exactly on the cycle the core is on bank-(not X). Measured: of 908 both-active
cycles only 68 are same-bank (7.5% vs ~50% by chance) -- **92.5% anti-correlated, 6.7x below
chance**. The DMA sails through the dense core's "gaps." HW phase-locks the OPPOSITE way (~1082
DMA losses over ~1024 granules => nearly every granule collides). Same period-8 structure,
opposite phase.

Anchors: `src/device/dma/engine/stepping.rs:311` `next_granule_fetch` (presents the DMA bank
demand on a stream-drain schedule decoupled from the core -- where the dodging phase is born);
`:156` `channel_bank_mask` (routes it to the peek); `src/interpreter/engine/coordinator.rs`
`arbitrate_memory_banks` (~2075-2108: the two independently-phased streams meet with nothing
coupling them).

## Error #2: round-robin vs HW core-priority

`src/device/bank_arbiter.rs` is pure per-bank round-robin over all requesters (3 core ports +
DMA channels by ordinal). On the few conflicts it has, EMU loses 44% to the DMA. HW loses 1.6%
(core wins ~98%; DMA absorbs via FIFO). The width finding already saw this from the other side
("the DMA loses 78%... does not win by priority"). Faithful model: core-CLASS beats DMA-CLASS,
round-robin WITHIN a class (preserves the HW-exact consumer results, which are ~91% core-vs-core
self-conflict).

## The fix is ONE two-part change (do not do either half alone)

- core-priority alone (68 conflicts x 1.6%) -> ~1 stall: UNDERSHOOTS HW's 18-20.
- denser conflicts alone (1100 x 44% round-robin) -> ~480 stalls: massive OVERSHOOT.
- both (1100 x 1.6%) -> ~18: matches HW.

Error #1's faithful fix must break the deterministic anti-lock so the DMA collides at ~HW rate
against a dense same-logical-bank core -- WITHOUT a fragile fixed phase-offset constant (that
would be calibration) and WITHOUT changing the core march cost (1/cy correct) or granule cadence
(1-in-4 correct). Candidate framings: (a) align the fetch phase in-phase with the core on physical
bank X; (b) model the micro-timing (fetch jitter / retry-while-denied drift) the fixed-schedule
emulator lacks; (c) the physical framing -- a dense core owning logical bank 0 uses one of its two
interleaved physical banks EVERY cycle, so an MM2S reading the same logical bank cannot reliably
find a free slot and must contend+re-present every granule. (c) may point at arbitrating the
contention such that a dense core denies the DMA regardless of exact sub-bank phase.

## What Codex refuted (adopted; do not repeat)

- `producer_probe_measure.py` divided whole-run MEMORY_STALL area by the bracketed-transfer
  count -> reported 2.0 where per-transfer is 1.875 (fixed: divide by all FINISHED_BD).
- The "boundary vs steady-state" split I briefly claimed was a STALLED_LOCK-bracket artifact --
  the stalls occur during active MM2S streaming, not at idle boundaries. Retracted.
- "core priority overshoots to ~0.07" conflated absolute priority (zeros DMA-caused stalls by
  construction) with the real urgency behavior. The real caution is the AM020 round-robin
  citation, in tension with HW's empirical ~98% core win.
- Cross-kernel per-word extrapolation (256->1024) is unsupported (the probe is phase-dependent).

## Traps for anyone re-running this

- Compare INTERVAL AREA, never decoded record counts (Event+Repeat undercount ~2.2x); ts never soc.
- of_q0_rich is Q=0: the mem-unit trace arms late and catches only ~2-3 tail reps of 16. It
  CANNOT give an absolute producer stall rate. Use producer_probe (self-looping BD, steady-state).
- The STALLED_LOCK->FINISHED_BD bracket is for DURATION; do not localize stalls to it.

## Status (2026-07-15): LANDED

The two-part fix this finding called for landed as designed, across three commits:
core-class-priority arbiter (`5da4cfd8`), DMA egress-FIFO urgency override +
backoff-on-denial granule fetch (`d77c391c`, the backoff drift is what breaks the
anti-lock), and threading that urgency into bank arbitration (`e8e254aa`). Plan/spec:
`docs/superpowers/plans/2026-07-15-producer-collision-C-cycle.md`,
`docs/superpowers/specs/2026-07-15-producer-collision-C-cycle-spec.md`. Gate
(`.superpowers/sdd/task-4-emu-gate.md`): CONFLICT count LANDED -- this finding's
anti-locked 68 -> 870 (collide) / 900 (collide_read_dense), within ~20% of HW's
1098-1103 / ~1162, apart stays 0. The residual MEMORY_STALL undercount on the same
probe (EMU 0 vs HW 18-20/24) is NOT a repeat of this anti-lock and NOT a calibration
miss -- it is a small sub-cycle phase-average tail below the deterministic lockstep
model's resolution, documented as its own residual in `docs/known-fidelity-gaps.md`
(class file `docs/fidelity-gaps/core-compute-timing.md`) and left as a signpost to
the future variable-latency memory model, not a promise it recovers exactly 18-20.
