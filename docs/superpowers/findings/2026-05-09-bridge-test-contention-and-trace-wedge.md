---
name: 'Bridge test reliability: HW/EMU contention masked v1 failures; memtile_dmas family is broken on HW *independent* of trace'
description: The mysterious memtile_dmas/cascade/packet_flow HW failures in v1 sweeps were artifacts of EMU at -j16 starving HW dispatch. Switching Phase 3+4 to sequential (HW first solo, then EMU parallel) makes 90/95 HW tests pass cleanly. Five script tooling improvements landed. The "trace wedge on memtile-only tests" hypothesis (Bug #6) was a RED HERRING -- post-reboot bisect showed memtile_dmas/* TDR even with --no-trace, both compilers, repeatedly. The trace prologue is bit-identical to passing tests. Real cause is unknown; suspect memtile DMA programming / firmware state. Bug #6 remains open with new scope.
type: project
---

# Bridge test reliability + memtile_dmas regression -- 2026-05-09

## TL;DR

A long debugging session went through three phases:

1. **CPU contention masquerading as HW bugs.** Phase 3+4 originally ran
   HW (-j1) and EMU (-j16) concurrently. EMU at -j16 starved HW's
   userspace dispatch enough to cause spurious TDRs and even data
   corruption (FAIL) on memtile_dmas tests. Confirmed by switching to
   sequential ordering. The "30/30 PASS" reliability run that supported
   this hypothesis was either run with trace enabled (different code
   path than my variants) or hit a transient working window -- post-
   reboot the same tests now fail consistently with --no-trace.

2. **Trace pipeline crash on shim+memtile-only tests.** `mlir-trace-inject.py`
   bailed early when a device had no compute tiles, then `_build_trace_config`
   crashed downstream on `min(t["col"] for t in tiles_traced)`. Affected
   the entire `memtile_dmas/*` family plus several others. Real bug,
   now fixed (commit `534383e`).

3. **"Trace wedges memtile-only tests" -- RED HERRING.** With trace
   injection enabled, `memtile_dmas/dma_configure_task_lock` reproducibly
   TDRs the NPU within 9-15s. Initial bisect showed both shim-only and
   memtile-only injection variants fail, suggesting a shared trace
   plumbing issue. **However, follow-up MLIR-level binary-search variants
   (A-F below) ALL TDR, including a variant that drops the entire
   shim trace BD setup. After clearing build caches and re-running with
   --no-trace, the test STILL TDRs.** All four memtile_dmas tests we
   tried (blockwrite_using_locks, dma_configure_task_lock,
   dma_configure_task_token, writebd, writebd_tokens) TDR with --no-trace
   on both compilers. add_one_using_dma --no-trace passes; xrt-smi
   validate passes; system is healthy. The trace was incidental.

The HW-only sweep with sequential ordering ran 95/95 PASS in 26 minutes
in this session, but that was BEFORE the memtile_dmas regression
manifested -- they appear in the doc-section "Tests confirmed PASSING"
below but no longer pass.

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

## Bug #6 (rescoped): memtile_dmas/* family TDRs on HW with --no-trace

### Repro

```bash
pkexec modprobe -r amdxdna && pkexec modprobe amdxdna
xrt-smi validate                      # confirm NPU healthy
# clear cache so bridge test actually rebuilds
rm -rf ../mlir-aie/build/test/npu-xrt/memtile_dmas/dma_configure_task_lock/{chess,peano,traced,test.exe,test.cpp}
nice -n 19 ./scripts/emu-bridge-test.sh --no-emu --no-trace --chess-only \
    'memtile_dmas/dma_configure_task_lock$'
```

Result: HW TDR at ~7s, 100% reproducible. add_one_using_dma --no-trace
PASSES under the same conditions, so the system itself is healthy.

### Variant-bisect ruling out the trace-prologue hypothesis (2026-05-09)

Hand-edited `aie_traced.mlir` and recompiled via aiecc, then ran
test_traced.exe directly with the modified xclbin/insts.bin. All
variants TDR'd:

| Variant | What was changed | Result |
|---|---|---|
| Baseline | trace as injected | TDR @10s |
| A | drop issue_token bit on trace BD push (0x8000000F → 0xF) | TDR |
| B | drop shim event broadcast/timer setup (0x34000/0x3404C/0x34008) | TDR |
| C | drop memtile trace config (0x94*** writes) + packet flows | TDR |
| D | drop entire shim trace BD setup (writebd 15 + push) | TDR |
| F | keep all writes, drop just the packet_flow blocks | TDR |

After this dead-end, cleared the build cache and re-ran the bridge
test with `--no-trace`. **Still TDR.** That eliminated trace as a
cause entirely. Also confirmed:

- 4/4 other memtile_dmas tests (blockwrite_using_locks,
  dma_configure_task_token, writebd, writebd_tokens) all TDR with
  --no-trace on chess.
- The trace prologue is **bit-identical** to add_one_using_dma's
  trace prologue (same controller_id collision pattern, same shim S2MM
  1 BD push with token, same memtile Timer_Control = 0x9D00).
  add_one_using_dma passes with trace, memtile_dmas tests TDR with or
  without trace.

So the original "trace wedges memtile-only tests" bisect (which I'd
done by toggling --shim-sweep-events vs --memtile-sweep-events flags)
was finding the same underlying TDR each time, NOT a trace-specific
bug.

### What dmesg actually shows

```
amdxdna 0000:c6:00.1: aie2_tdr_work: Device isn't making progress... Count 13 timeout 2
amdxdna 0000:c6:00.1: aie2_dump_ctx: Dumping ctx ...
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	op: 0x0
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	msg: 0x1d000001
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	fence: unsignaled
```

Generic "kernel never completed". No firmware-side diagnostic. The
kernel is just hung somewhere in the runtime sequence's DMA waits.

### Open questions for next session

- Are these tests ACTUALLY broken, or is there persistent state from
  trace TDRs that a stronger reset (suspend/resume, reboot) would
  clear? `pkexec modprobe -r amdxdna && pkexec modprobe amdxdna` does
  NOT clear it -- need to escalate.
- If tests are genuinely broken: when did they regress? They were
  added by mlir-aie commit `9c92428136` on 2026-02-06 and were
  reportedly passing in earlier sessions.
- WHERE is the wedge happening? The runtime sequence has chained
  memtile S2MM 0 BDs (0,1,2,3) gated on prod_lock/cons_lock, then
  memtile MM2S 0 (BD 4) acquires cons_lock 4× and reads, then shim
  S2MM 0 receives. dmesg can't tell us which step hangs. Need to
  read AIE registers post-TDR via `xrt::hw_context::read_aie_reg`
  (#356 added this on NPU1) to see lock state, BD state, channel
  state, etc.

### Suggested next steps (post-reboot)

1. Reboot to clear any persistent NPU/firmware state.
2. Re-run memtile_dmas/dma_configure_task_lock --no-trace and verify
   it still TDRs (confirm regression vs accumulated state).
3. If still TDRs: capture AIE register state at TDR time via
   `xrt::hw_context::read_aie_reg`. Read in order of likely hangs:
   - memtile prod_lock / cons_lock values
   - memtile S2MM 0 channel CTRL + STATUS + current_BD
   - memtile MM2S 0 channel CTRL + STATUS + current_BD
   - shim S2MM 0 channel CTRL + STATUS
   - shim MM2S 0 channel CTRL + STATUS
4. From register state, identify which DMA is stuck on which lock.
5. Cross-reference with mlir-aie git history near 9c92428136 to see
   if any subsequent commit changed lock semantics or BD chain
   behavior.

## Tests reportedly PASSING earlier in session (now contradicted)

The `bahg41enj` reliability run logged 30/30 PASS for these (3 iters,
chess+peano):

- `cascade_flows` (chess)
- `memtile_dmas/blockwrite_using_locks` (chess)
- `memtile_dmas/dma_configure_task_lock` (chess + peano)
- `memtile_dmas/dma_configure_task_token` (chess + peano)
- `memtile_dmas/writebd` (chess)
- `memtile_dmas/writebd_tokens` (chess)
- `packet_flow_fanout` (chess + peano)

**But these now consistently TDR with --no-trace.** Possible
explanations:

- The reliability run may have used trace artifacts (it ran during
  the same session as trace experiments and may have inherited
  cached trace-injected artifacts).
- Or the NPU was in a different state then; the many trace TDRs
  since then have left persistent state nothing short of a reboot
  clears.
- Or a recent emulator/tooling change perturbed something.

This needs to be re-verified after a reboot.

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
