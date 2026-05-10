---
name: '#355a Phase A: cycle-anchored HW vs EMU on add_one_using_dma'
description: First cycle-accurate trace measurement on both sides. Pipeline fill (compute LOCK_STALL -> first ACQUIRE_REQ) is 3750 cyc on HW vs 329 cyc on EMU -- an 11.4x ratio. That gap is structural, not in the kernel iteration itself. Shim BD lifecycle is 3.0-3.5x too fast on EMU. The earlier waterfall analysis underestimated the chain because it measured EMU-internal phase residency, not HW-grounded propagation. AM020's documented per-stage latencies don't account for thousands of cycles of additional HW serialization.
type: project
---

# `#355a` Phase A: cycle-anchored measurement on `add_one_using_dma`

## TL;DR

Wired `XDNA_TRACE_MODE=event_time` through `trace-prepare.py` -> 
`mlir-trace-inject.py` and ran the bridge test on `add_one_using_dma`
with both HW and EMU producing mode-0 (cycle-delta) traces. Joined the
parse-trace events with their slot names and computed first-event
cycle deltas per side.

The headline number:

| Metric                                       |     HW |    EMU | Ratio  |
|----------------------------------------------|-------:|-------:|-------:|
| Pipeline fill (LOCK_STALL -> ACQUIRE_REQ)    |   3750 |    329 | 11.4x  |
| Shim MM2S 0 BD lifecycle (start -> finish)   |   2849 |    943 |  3.0x  |
| Shim S2MM 0 BD lifecycle (start -> finish)   |   3210 |    926 |  3.5x  |

EMU's pipeline propagation is **11.4x too fast** for the kernel-visible
fill window. The earlier waterfall claim that "the chain is correct
because EMU completes it in ~547 cyc which matches the model" was
correct *for what the model says* but wrong about whether the model is
calibrated -- HW takes thousands more cycles than AM020's documented
per-stage latencies suggest.

## Methodology

Phase A was a no-instrumentation first pass: re-run with mode-0 traces
on both sides, decode existing events, compare cycle deltas.

Plumbing change to enable it:

- `tools/trace-prepare.py`: added `--trace-mode` CLI flag forwarded
  to `mlir-trace-inject.py`.
- `tools/mlir-trace-inject.py`: already had `--trace-mode` (default
  `event_pc`); now reachable through the bridge orchestrator.
- `scripts/emu-bridge-test.sh`: reads `XDNA_TRACE_MODE` env and
  passes it to trace-prepare. Default behavior (event_pc) unchanged.

Run:

```bash
rm -rf ../mlir-aie/build/test/npu-xrt/add_one_using_dma/{chess,peano,traced,test.exe,test.cpp}
XDNA_TRACE_MODE=event_time ./scripts/emu-bridge-test.sh --chess-only 'add_one_using_dma$'
```

Decode (per-side `events.json`):

```bash
python3 tools/parse-trace.py --trace-bin <trace_raw.bin> \
  --xclbin-mlir <input_with_addresses.mlir> \
  --trace-mode event_time --out-events <out>
```

Then a one-off Python script joins events with slot_names (the
parse-trace JSON keeps them separate -- a future PR could merge them
in the parser itself).

Anchor: `DMA_MM2S_0_START_TASK` on shim col 1 row 0 -- the runtime
sequence's first `dma_memcpy_nd` push (host -> memtile input). All
deltas reported relative to that anchor.

## Detailed event timeline (cycles relative to shim MM2S 0 start)

```
event                                               HW rel   EMU rel   ratio
-----------------------------------------------------------------------------
compute kernel LOCK_STALL (first acq blocked)          148        49     3.0x
shim MM2S 0 start (host->memtile)                        0         0
shim S2MM 0 start (memtile->host)                      889      1116     0.8x
compute kernel first acquire SUCCESS                  3898       378    10.3x
shim MM2S 0 finished                                  2849       943     3.0x
shim S2MM 0 finished                                  4099      2042     2.0x
```

Notes:

1. **Compute LOCK_STALL fires very close to shim push on both sides**
   (148 vs 49 cyc after anchor). So the compute boot vs runtime sequence
   *ordering* is roughly aligned -- compute runs and hits its first
   blocked acquire shortly after the runtime sequence pushes the input
   DMA on both HW and EMU. The 3x difference here is small in absolute
   terms.

2. **Shim S2MM 0 (output BD push) is dispatched 889 cyc after MM2S on
   HW vs 1116 cyc on EMU.** Within 25% on both sides. So the runtime
   sequence interpretation speed is roughly comparable -- not a
   per-instruction dispatch gap.

3. **The 11.4x pipeline-fill gap** is what dominates the apparent total
   trace-span ratio. It's not kernel iteration timing (the May 5
   finding's measurement of ~1.0x kernel iteration speed still holds);
   it's the time the kernel waits between issuing its first acquire
   and seeing the lock fire.

4. **Shim BD lifecycle is 3x slower on HW.** With 64 words and AM020's
   documented 4 bytes/cyc, theoretical lifetime is ~64 cyc; HW shows
   2849 cyc (44.5 cyc/word effective), EMU 943 cyc. Both are dominated
   by backpressure from the kernel's consumption rate, but HW
   backpressures 3x harder, consistent with each stage of the
   downstream chain being slower in absolute cycles.

## How this overrules the earlier waterfall finding

The Phase 1 waterfall (`2026-05-10-dma-waterfall-pipeline-fill-decomp.md`)
concluded "the chain is correct because EMU completes it in 547 cyc
which matches per-stage costs from the model." That conclusion was
internally consistent -- the model's per-stage costs DO sum to ~547
cyc -- but it implicitly assumed AM020's documented latencies are
calibrated against HW.  This measurement shows they are not.

The implication: there is an additional ~3000 cycles of HW serialization
in the chain that AM020 does not document.  Almost certainly some
combination of:

- **Cross-tile lock release-to-acquire propagation.** AM020 doesn't
  document the latency for a memtile S2MM lock-release event to
  propagate to a memtile MM2S waiting on the same lock.  In real
  hardware this likely involves the lock unit's NoC packet machinery
  rather than a 1-cycle handoff.
- **Memtile DMA channel-to-data-port arbitration.** Memtile has 6
  channels per direction sharing 6 data ports.  When channel 0
  releases and channel 0 (different direction) wants to acquire, there
  may be many cycles of arbitration even with no contention.
- **NoC bandwidth or fabric pipeline latency** beyond the
  3-cyc-per-hop documented stream-switch latency.  The shim->memtile
  link traverses array-interface NoC routing that AM020 chapter 3
  doesn't decompose into cycle costs.
- **Shim DMA host-DDR latency itself.** Our model uses
  `host_memory_latency_cycles=500`; the May 6 finding's response curve
  showed pipeline fill responds 1:1 to this knob.  Pushing this from
  500 to ~3500 would close most of the gap on its own, but that
  exceeds any plausible PCIe + NoC + DDR estimate (~250-1000 cyc),
  so it's at most part of the answer.

## What Phase A doesn't tell us

We have aggregate cycle deltas, but no decomposition of *where* the
HW chain spends those 3000+ extra cycles.  To attribute them we need
finer-grained instrumentation, which is Phase B/C:

- **Phase B (instrument runtime sequence + post-DMA points):** insert
  `aiex.npu.write32` user-event triggers in the MLIR runtime sequence
  at multiple instrumentation points so per-segment cycle deltas
  become directly observable on both sides.  Example anchors: just
  before each `dma_memcpy_nd`, just after each, and just before
  `dma_wait`.

- **Phase C (compute-kernel boot anchor):** add a user-event trigger
  as the compute kernel's first instruction (modify the .cc source
  file, or have aiecc inject it).  Distinguishes "kernel boot" from
  "kernel hits first acquire" in the trace.

- **Per-stage timing on HW:** re-run with `dma-fill-measure.py` once
  the existing `parse-trace.py` is patched to actually populate
  `name` from `slot_names` (currently the JSON has them in separate
  fields and downstream tools have to do the join).  The measure tool
  already extracts `t_first_dma_start`, `t_first_dma_finished`,
  `dma_roundtrip` per tile -- those extractions just need named
  events to land in the events array.

## Recommended next direction

The 11.4x pipeline-fill gap is the calibration target.  Options:

1. **Phase B instrumentation.**  Best ROI: insert user events at known
   landmarks and build a per-stage HW vs EMU timeline.  We can then
   pinpoint which stage's cycles are short (or absent) in EMU.

2. **Patch parse-trace.py to populate event names.**  Cheap follow-up
   so dma-fill-measure.py and similar tools get usable events without
   each tool joining slot_names.  Independent of the calibration.

3. **Calibrate `host_memory_latency_cycles`.**  Bumping it from 500
   towards the upper plausible end (1000-1500) will close some
   fraction of the gap and is harmless if HW reads support a higher
   value.  But Phase B should come first to know whether host-pipe
   is the right knob or not.

## See also

- `2026-05-05-355-cycle-divergence-diagnosis.md` -- original
  decomposition, claimed kernel-execution at ~1.0x ratio.  Holds up
  under this measurement.
- `2026-05-06-355a-host-latency-response.md` -- host_lat tuning,
  named DMA pipeline cost as suspect.
- `2026-05-10-dma-waterfall-pipeline-fill-decomp.md` -- waterfall
  analysis. Its claim "EMU chain is correct" is **superseded** by
  this finding -- the chain is correct *to AM020's documented
  latencies*, but those latencies don't capture all HW serialization.
- `tools/dma-fill-measure.py`, `tools/dma-waterfall.py`, plus the new
  `--trace-mode` plumbing in `trace-prepare.py` /
  `emu-bridge-test.sh`.
- task #355a -- Phase A complete; Phase B is next.
