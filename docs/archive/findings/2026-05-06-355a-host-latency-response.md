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

1. Adding the missing structural latencies. **Note (2026-05-07):**
   stream-switch fabric latency was originally listed here but a
   follow-up audit found it is already correctly applied (intra-tile
   via `LocalRoute.latency`, inter-tile via `ROUTE_PER_HOP`). The
   leading remaining suspect is **DMA engine internal pipeline costs**
   (shim + memtile BD-startup, address generation, request issue),
   plus memtile DMA broadcast/fanout and NoC contention.
2. Direct ground-truth HW cycle measurement via the now-validated
   `AIE_RW_ACCESS` path on compute tiles, which lets us calibrate
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
   fill is ~3700+ cyc shorter than HW. Suspects (re-evaluated 2026-05-07):
   - ~~Stream-switch fabric per-hop latency~~ -- **already applied**.
     Intra-tile: `LocalRoute.latency` set per-route by `with_port_latency`
     (port-type-derived: 3 for local-to-local, 4 for paths involving
     external) and consumed via `switch_pipeline.cycles_remaining` in
     `StreamSwitch::step`. Inter-tile: `ROUTE_PER_HOP = 4` applied in
     `array::routing::propagate_inter_tile`. Total per-hop contribution
     is ~7-11 cyc, far too small to account for the ~3700 cyc gap.
     (Dead-code cleanup opportunity: the `local_latency`/`external_latency`
     fields on the `StreamSwitch` struct itself are written at construction
     time but never read -- the live data-path uses `LocalRoute.latency`
     and `ROUTE_PER_HOP` instead. Safe to delete.)
   - Shim and memtile DMA engine internal pipeline costs (BD-startup,
     address generation, request issue, response handling) likely
     under-modeled in EMU. This is the leading suspect for the gap.
   - Memtile DMA broadcast/fanout per-port arbitration.
   - Missing NoC bandwidth contention (multiple shim transfers
     compete for the same NoC link in HW).

## Recommended next steps

Listed in order of expected payoff per engineering hour:

1. ~~Wire stream-switch fabric latency into the data path FSM.~~
   **Withdrawn 2026-05-07.** Audit found that both intra-tile and
   inter-tile latency are already wired into the data-path FSM (see
   Suspects above). The original observation about unused fields was
   incomplete -- it spotted dead code on the `StreamSwitch` struct but
   missed the live `LocalRoute.latency` / `ROUTE_PER_HOP` consumers.
   Stream-switch contribution to the gap is at most ~50 cyc; cannot
   close ~3700 cyc.

2. **Audit DMA engine internal pipeline timing.** Now the leading
   suspect. Shim and memtile DMA engines have BD-startup, address
   generation, request issue, and response handling phases. Compare
   EMU's stepping logic against aie-rt's actual DMA programming
   sequence (`../aie-rt/driver/src/dma/xaie_dma_aieml.c`) and AM020 Ch3
   for the cycle costs of each phase. This is where the 3700-cyc gap
   most likely lives.

3. **Calibrate against HW ground truth via `AIE_RW_ACCESS`.** The path
   now works on Phoenix for compute-tile registers and DM (op-table fix
   landed 2026-05-05; functional validation completed 2026-05-07, see
   `2026-05-07-aie-rw-access-memtile-dm-half-impl.md`). Read `TIMER_LOW`
   on each compute tile pre/post-kernel for ground-truth total cycles
   per tile -- bypasses trace interpretation entirely. Memtile reads
   are NOT available (firmware bug); compute tile coverage is what
   matters for cycle-cost calibration. Calibrate framework knobs
   against this measurement.

4. **Add NoC bandwidth/contention model.** Heavier engineering; not
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
