---
name: '#355a host_memory_latency response curve and first calibration step'
description: Empirical response of EMU pipeline fill to host_memory_latency_cycles is linear with slope 1. Bumping 100 -> 500 closes ~10% of the EMU/HW gap; closing the rest needs structural changes (stream-switch fabric, multi-stage backpressure) AND/OR ground-truth HW cycle measurement via read_aie_reg.
type: project
---

# #355a -- host_memory_latency calibration: linear response, structural gap remains

## TL;DR

Bumped `host_memory_latency_cycles` from 100 to 500 as a first calibration
step. Measured response curve: pipeline fill increases by exactly the
delta in `host_memory_latency_cycles`, slope = 1.0. With host_lat=500,
EMU pipeline fill on `add_one_using_dma` is 2719 cyc vs HW's ~6000 cyc
(2.2x ratio, down from 2.6x).

The remaining gap cannot be closed by tuning `host_memory_latency`
alone -- a value of ~3700 would be needed to match HW, which is
implausible for any realistic PCIe + NoC + DDR fill estimate. Closing
the gap end-to-end requires either:

1. Adding the missing structural latencies (stream-switch fabric,
   memtile DMA broadcast/fanout, NoC contention).
2. Direct ground-truth HW cycle measurement via the now-working
   `xrt::hw_context::read_aie_reg` path, which lets us calibrate
   against actual cycle counts rather than inferred trace metrics.

## Measurement

Same harness as `2026-05-05-355-cycle-divergence-diagnosis.md`. EMU
pipeline fill = number of `tile(0,2) ch2.*granted=false` log lines
between the compute core's first `LockAcquire raw=49 ... WAIT` and
its corresponding `LockAcquire raw=49 ... SUCCESS`.

| host_memory_latency_cycles | EMU pipeline fill | EMU/HW ratio |
|----------------------------|-------------------|--------------|
| 100 (baseline)             | 2319 cyc          | 2.59x        |
| 500                        | 2719 cyc          | 2.21x        |

Slope = 1.0 cycle of pipeline fill per cycle of host_memory_latency,
matching the FSM design: shim DMA enters `HostPipelineLatency` for
exactly `host_memory_latency_cycles` cycles before the first word of
the BD's transfer. The first BD on the input chain is the bottleneck
that gates downstream stages, so its host-fill latency translates 1:1
into pipeline-fill cycles.

To match HW (~6000 cyc) by tuning host_memory_latency alone:
`6000 - 2319 + 100 = 3781 cyc`. That's ~9.4 us at 400 MHz, well above
any plausible PCIe + NoC + DDR fill latency. Defensible Phoenix-class
estimates are 250-500 cyc, hence the choice of 500 as the new baseline.

## Why 500, not 100

The original `100` was acknowledged in code as an initial guess that
"we never validated against HW." Phoenix Ryzen AI hardware has roughly:

- PCIe round-trip ≈ 250 ns ≈ 100 cyc at 400 MHz.
- NoC traversal ≈ 50-100 ns ≈ 20-40 cyc.
- DDR access ≈ 100-150 ns ≈ 40-60 cyc.

Sum of the typical components: ~150-200 cyc one-way, ~300-400 cyc for
a full read round-trip including arbitration. 500 cyc rounds up
slightly to account for the worst-case fill (no banks open, NoC
contention with other transfers). Still in the plausible range; not
calibrated against ground-truth.

## Where the rest of the gap lives

The 11.2x gap on the trace-event side (HW core wait 2839 cyc vs EMU
253 cyc) decomposes into two roughly-independent problems:

1. **EMU core boots too fast** (~2.9x). The compute core reaches its
   first lock acquire at ~2085 cyc EMU vs ~6000 cyc HW (inferred from
   PERF_CNT_0 spacing of 6 events at 1024-cyc threshold). Likely
   instruction-execution timing being too aggressive on certain
   bundles (vector ops, SRS, or specific intrinsics).

2. **EMU DMA pipeline fills too fast** (current bridge-log measurement).
   Even if host_memory_latency is tuned correctly, the chain's overall
   fill is ~3700+ cyc shorter than HW. Suspects:
   - Stream-switch fabric per-hop latency (3-4 cyc/hop x 2-3 hops),
     model values exist (`local_to_local_latency: 3`) but are not
     consumed by the data-path FSM.
   - Memtile DMA broadcast/fanout per-port arbitration.
   - Missing NoC bandwidth contention (multiple shim transfers
     compete for the same NoC link in HW).

## Recommended next steps

Listed in order of expected payoff per engineering hour:

1. **Wire stream-switch fabric latency into the data path FSM.** The
   constants are already in `StreamSwitchTiming` but no consumer reads
   them. Should add 6-12 cyc per stream hop to pipeline fill.

2. **Calibrate against HW ground truth.** The `read_aie_reg` path now
   works on Phoenix (op-table fix landed 2026-05-05). Use it to read
   `Performance_Counter0` from the compute tile after a kernel run,
   giving us a direct cycle measurement that bypasses trace
   interpretation. Calibrate cycle-cost framework knobs against that
   measurement.

3. **Add NoC bandwidth/contention model.** Heavier engineering; not
   needed for single-input/single-output kernels, but matters for
   matmul-cascade-style tests with concurrent shim transfers.

## Validation

- `cargo test --lib`: 2863 passed, 0 failed (no test pins the
  host_memory_latency value).
- Bridge test on `add_one_using_dma` with chess + peano: both PASS.

## See also

- `docs/superpowers/findings/2026-05-05-355-cycle-divergence-diagnosis.md`
  -- original diagnosis and metric definitions.
- `docs/superpowers/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md`
  -- read_aie_reg path is now usable on Phoenix; see Step 2 above.
- `crates/xdna-archspec/src/model_builder.rs:165` -- the constant
  changed by this finding.
- task #355a -- still open; this finding documents progress, not
  completion.
