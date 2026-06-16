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

| port | **HW** (15 runs) | **EMU** (phoenix) | old frame metric (HW / EMU) |
|------|:----------------:|:-----------------:|:---------------------------:|
| PORT_RUNNING_0 (recv<-shim)     | **1** (std 0) | 2 | 5.65 / 5 |
| PORT_RUNNING_1 (recv<-compute)  | **5** (std 0) | 6 | 8 / 8 |
| PORT_RUNNING_4 (send->compute)  | **3** (std 0) | 6 | 7.26 / 7 |
| PORT_RUNNING_5 (send->shim)     | **4** (std 0) | 4 | 4 / 4 |

The "DDR jitter" (slot0 `5.65 ± 0.48`) that the entire stochastic delivery model
was built to reproduce **is not present at the span level**. It was
re-checkpoint-timing noise in the broken metric.

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

## The real #140 (deterministic, no RNG)

EMU inserts genuine `>2`-cycle idle gaps in PORT_RUNNING that HW does not:
- **slot4 (send->compute): EMU 6 sub-bursts vs HW 3** -- the headline. Our
  memtile->compute send is choppier than silicon.
- slot0 (recv<-shim): EMU 2 vs HW 1 -- we split the continuous delivery once.
- slot1: EMU 6 vs HW 5.
- slot5: EMU 4 vs HW 4 -- already exact.

This is now a clean, deterministic, reproducible target: find the spurious gaps
and remove them. No band-matching, no seeds.

## Consequences / decisions

1. **The stochastic DDR ("phoenix") model models a ghost.** HW PORT_RUNNING is
   deterministic (std 0, 15 runs). `BurstParams::AIE2_DDR_PHOENIX`, the seeded
   PRNG, system-entropy seeding, the 100-run calibration -- all reproduce
   variance silicon does not have. **Decision pending: revert, or repurpose to a
   deterministic delivery-shaping knob.** (Deferred to the post-compaction
   implementation pass.)
2. **The metric is fixed:** `tools/port-span-cadence.py` (span-based, oracle B/E,
   idle-gap>2). `port-cadence-baseline.py` is superseded for cadence work.
3. **`docs/known-fidelity-gaps.md` row 50** (PORT_RUNNING recv-port) needs
   rewriting: the gap is real but *deterministic*, and the phoenix "1-sigma
   band-match" was an artifact.

## Scaffolding landed (default-off diagnostics for the implementation pass)

- `XDNA_EMU_XFORM_PROBE` (coordinator): per-cycle memtile/compute DMA + lock +
  port-beat timeline, auto-detects the active column. `[XEDGE]` logs the exact
  PORT_RUNNING level edges fed to the encoder.
- `XDNA_EMU_DDR_SCRIPT` (burst.rs): scripted `(burst,gap)` delivery -- may become
  the deterministic delivery-shaping mechanism.
- `XDNA_EMU_MM2S_1WORD` (stepping.rs): AXI4-Stream 1-word/cyc egress experiment
  (Lever 1) -- inconclusive on its own; kept default-off.

All gated, default-off, 3534 lib tests green.
