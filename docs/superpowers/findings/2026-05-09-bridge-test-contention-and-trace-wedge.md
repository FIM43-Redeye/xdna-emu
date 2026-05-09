---
name: 'Bridge test reliability: HW/EMU contention masked all v1 "real bugs"; trace-injection wedges memtile-only tests'
description: The mysterious memtile_dmas/cascade/packet_flow HW failures observed in v1 sweeps were all artifacts of EMU at -j16 starving HW dispatch. Switching Phase 3+4 to sequential (HW first solo, then EMU parallel) makes 95/95 HW tests pass cleanly. Five script tooling improvements landed. One real bug remains open: trace injection on shim+memtile-only tests (entire memtile_dmas family + others) reproducibly TDRs the NPU; both shim-only and memtile-only injection variants fail independently, so the cause is something common to all trace setups on no-compute-tile devices.
type: project
---

# Bridge test reliability + trace-injection wedge -- 2026-05-09

## TL;DR

A long debugging session unraveled the "memtile_dmas tests fail on real
hardware" mystery into three independent causes:

1. **CPU contention masquerading as HW bugs.** Phase 3+4 originally ran
   HW (-j1) and EMU (-j16) concurrently. EMU at -j16 starved HW's
   userspace dispatch enough to cause spurious TDRs and even data
   corruption (FAIL) on memtile_dmas tests. Targeted reliability test
   (5 v1-failing tests x 3 iterations, HW solo) showed **30/30 PASS**.
   Hypothesis confirmed.

2. **Trace pipeline crash on shim+memtile-only tests.** `mlir-trace-inject.py`
   bailed early when a device had no compute tiles, then `_build_trace_config`
   crashed downstream on `min(t["col"] for t in tiles_traced)`. Affected
   the entire `memtile_dmas/*` family plus several others. Now fixed
   (commit `534383e`).

3. **Trace injection wedges memtile-only tests on real HW.** With trace
   injection enabled, `memtile_dmas/dma_configure_task_lock` reproducibly
   TDRs the NPU within 9-15s. Both shim-only and memtile-only injection
   variants fail, so the cause is shared trace plumbing, not a
   tile-type-specific issue. **Open bug.**

The HW-only sweep with sequential ordering ran 95/95 PASS in 26 minutes,
0 TDRs.

## Bridge test improvements landed

Five commits on `dev` between `9e5401e` and `9a0b793`:

- `425e798` -- bridge-test: parallel Phase 5b, sequential Phase 3+4,
  elapsed-time progress
- `7e9c071` -- bridge-test: skip Phase 5 trace compare when HW or EMU
  is disabled
- `534383e` -- mlir-trace-inject: don't skip devices without compute
  tiles
- `9a0b793` -- bridge-test: nuke stale per-test artifacts for sides not
  in this run

(Also one earlier commit during the same session for trace-inject
itself.)

The Phase 5b refactor splits the event sweep into two arms: HW serial
(NPU is single-tenant), EMU parallel via `xargs -P $JOBS` (no device
contention). Same HW-first ordering as Phase 3+4. This is where the
biggest wall-clock savings come from on `--sweep` runs.

The stale-artifact guard fixes a misleading-report bug we walked into:
running with `--no-emu` left old `.emu/` dirs in place from a previous
run, so Phase 5 silently compared today's fresh HW against yesterday's
stale EMU and the report displayed misleading "PASS PASS CLEAN" rows.
The guard nukes the disabled side's artifacts before Phase 3+4 starts.

## Bug #6: trace injection wedges memtile-only tests

### Repro

```bash
pkexec modprobe -r amdxdna && pkexec modprobe amdxdna
xrt-smi validate                      # confirm NPU healthy
nice -n 19 ./scripts/emu-bridge-test.sh --no-emu \
    'memtile_dmas/dma_configure_task_lock$'
```

Result: HW TDR at ~10s (chess) / ~16s (peano), 100% reproducible.
Without trace (`--no-trace`), same test passes in ~5s.

### Bisect findings (2026-05-09)

Manually edited the `trace-prepare.py` invocation in
`scripts/emu-bridge-test.sh` to omit one tile type at a time:

| Variant | Result |
|---|---|
| Both shim + memtile (default) | TDR |
| Shim only (`--shim-sweep-events all`, no memtile) | TDR |
| Memtile only (`--memtile-sweep-events all`, no shim) | TDR |
| No trace at all (`--no-trace`) | PASS |

So the wedge isn't tile-type-specific. Both modes share:

1. **Shim DMA channel 1** as the trace output sink. Both modes set up
   shim BD 15 and `dma_configure_task` on channel 1 to drain trace
   packets to a host buffer.
2. **Trace control register prologue writes** before the test's own
   `dma_configure_task` ops run (lines 12-29 in the
   `traced/aie_traced.mlir` for this test).
3. **address_patch with arg_idx=2** -- the trace BO patch targets arg
   slot 2, but the original `runtime_sequence` declares only 2 args
   (slots 0,1). Either the test_traced.cpp passes a 3rd arg slot that
   XRT honors, or this is a real protocol gap.

### Likely root causes (need verification)

- **Shim S2MM channel 1 conflict.** The test's data flows use shim
  S2MM channel 0; trace uses channel 1. They shouldn't conflict on
  paper, but the trace prologue writes to nearby register addresses
  (0x340D0..0x340E4 for shim Trace_Control + Trace_Event regs), and
  there's a `maskwrite32` to address 0x1D2C8 with mask 0xFF00 that
  warrants checking against the regdb.
- **Memtile register 0x94000 write at line 22** with value 0x9D00.
  This is on memtile (row=1) and might be touching memtile DMA or
  stream switch state the test relies on.
- **arg_idx=2 protocol.** Worth checking `test_traced.cpp` (or the
  template `tools/templates/test_traced.cpp.in` if it exists) to see
  whether 3 BOs are passed and how arg_idx maps.

### Suggested next steps for the open fix

In rough order of cheap-first:

1. Read `test_traced.cpp` for this test and confirm it passes 3 BOs
   (input, output, trace_bo). Verify XRT arg_idx mapping is sane.
2. Diff the trace prologue writes against AM025 regdb to identify
   which registers are being touched and whether they overlap with
   the test's memtile DMA channel state.
3. Capture the runtime mailbox traffic via the new debugfs interface
   (#358) to see what firmware sees vs what the kernel queues.
4. Try a "shim DMA channel 1 only, no register prologue" variant by
   manually editing the traced MLIR -- isolate whether the prologue
   writes alone are enough to wedge.

## Tests confirmed PASSING with HW solo (no contention, no trace)

From `bahg41enj` reliability run (3 iterations, 30/30 PASS):

- `cascade_flows` (chess)
- `memtile_dmas/blockwrite_using_locks` (chess)
- `memtile_dmas/dma_configure_task_lock` (chess + peano)
- `memtile_dmas/dma_configure_task_token` (chess + peano)
- `memtile_dmas/writebd` (chess)
- `memtile_dmas/writebd_tokens` (chess)
- `packet_flow_fanout` (chess + peano)

All previously suspected of being real DMA bugs. None are.

## HW-only full sweep summary

`bridge-sweep-20260509-hw-only.log` (started 15:39, 26m runtime):

- Chess: 61/62 compiled, **53/53 HW pass, 0 fail, 0 TDR** (10 skip)
- Peano: 54/55 compiled, **42/42 HW pass, 0 fail, 0 TDR** (11 skip,
  1 XFAIL)
- Trace prep: 59 OK, 10 FAIL (the bug #1 cluster, fixed in this
  session -- expect 69 OK on next sweep)

10 trace prep FAILs all hit the `min() iterable argument is empty`
crash now fixed by `534383e`:

- `bd_chain_repeat_on_memtile`
- `dynamic_object_fifo/sliding_window_conditional`
- `matrix_multiplication_using_cascade`
- `memtile_dmas/{blockwrite_using_locks, dma_configure_task_lock,
   dma_configure_task_token, writebd, writebd_tokens}`
- `objectfifo_repeat/simple_repeat`
- `ctrl_packet_reconfig_1x4_cores`

## Other artifacts from the session

- **PR amd/xdna-driver#1298** -- 25 typo fixes across driver, shim,
  uapi headers, and tools. Doc-only, doesn't depend on anything else.
- **`tools/dma-fill-measure.py`** built earlier in the day (commit
  `6b31370`) but not yet exercised against fresh data. The HW-only
  sweep produced 60 chess `.sweep/` dirs that could be ingested for
  #359 calibration once the EMU side is also captured.

## See also

- `docs/superpowers/findings/2026-05-06-355a-host-latency-response.md`
  -- prior #359/#355a calibration progress (the original target this
  session was supposed to advance, before the bridge test mess
  surfaced).
- task #359 -- still in_progress; this session unblocks it by
  proving HW results are reliable.
- task #378 -- open follow-up for upstream maskwrite32 trace patch.
