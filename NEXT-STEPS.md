# Next Steps

Working notes for resuming from 2026-05-05. Granular task state lives in
the task list (#XXX references below); this file is the orientation
document — read it first when picking up where we left off.

## State at session end (2026-05-05)

### What landed today

- **#351 fixed**: `aiex.npu.address_patch arg_idx` is the *BO arg
  index* in the kernel regmap (after the opcode/instr_BO/ninstr
  prefix), not the XRT kernel slot. Our injector was passing
  `3 + max_existing_memref_args` (= 6 for 3-memref tests), which made
  firmware patch BD15 with garbage. Fix: pass `max_existing_memref_args`
  (the BO index) for `arg_idx` and keep `3 + max_existing_memref_args`
  as the host-side XRT slot. Verified on real HW with
  `add_one_using_dma`: trace_raw.bin now populated on both compilers,
  both sides. See `docs/superpowers/findings/2026-05-05-trace-arg-idx-bug.md`.
- **Bridge test default flipped**: `NO_TRACE=false` is the new default.
  The old `NO_TRACE=true` made `--no-trace` runs silently produce no
  trace data while looking like a successful run. The script now does
  what its name suggests by default.
- **Calibration framework + cycle-cost model** committed (sweep
  analyzers, sweep configs, provisional NPU1 fast-mode constants).
- **Trace start pipelining + multi-tile timer reset** committed.
  Closes some of the gaps documented in `docs/coverage/`.

### What we discovered but haven't acted on yet

#### `read_aie_reg` is **NOT** firmware-blocked on NPU1 — driver-table-blocked

**Major correction to the 2026-05-05 morning diagnosis.** Phoenix
firmware (1.5.5.391, mailbox protocol 5.8) DOES implement
`MSG_OP_AIE_RW_ACCESS` (opcode 0x203). The previous claim that "Phoenix
firmware never implemented it" was wrong — the only obstacle was the
*driver-side* op-table check.

How we proved it: added a debugfs `nputest` test_case04 (raw mailbox
send-and-capture) to `aie2_debugfs.c`, sent opcode 0x203 with a
register-read request for TIMER_LOW (col=0, row=2, addr=0x340F8),
and got back `status = 0x0 (AIE2_STATUS_SUCCESS)` with a
register value that monotonically advanced ~13.4k cycles per read at
400 MHz MP-NPU clock — exactly matching shell loop wall-time. The
firmware understood the opcode, parsed the request, and returned a
valid value.

The actual block: `aie2_is_supported_msg()` iterates
`ndev->priv->optional_msg`. On Phoenix that table only contained
`MSG_OP_CHAIN_EXEC_NPU` — no entry for `MSG_OP_AIE_RW_ACCESS`. AMD's
`npu1_regs.c` simply omitted the entry; nothing about the firmware
gates the opcode.

**Fix applied 2026-05-05 (this session):**

```c
// xdna-driver/src/driver/amdxdna/npu1_regs.c
const struct msg_op_ver npu1_msg_op_tbl[] = {
    { AIE2_FW_VERSION(5, 8), MSG_OP_CHAIN_EXEC_NPU },
    { AIE2_FW_VERSION(5, 8), MSG_OP_AIE_RW_ACCESS },  /* added */
    { 0 },
};
```

Driver rebuilt and loaded. End-to-end XRT path verified:
`bridge-trace-runner --read-perf-counter` now returns
`perf_ok:true, core_cycles:0` instead of throwing on EOPNOTSUPP. The
zero comes from the *separate* lifecycle bug below — counter never
got configured.

**Outstanding lifecycle bug** in `bridge-runner/bridge-trace-runner.cpp`:
the pre-launch `read_aie_reg`/`write_aie_reg` calls (lines 1717-1724,
which initialize PERF_CTRL0 and zero PERF_COUNTER0 BEFORE `run.start()`)
fail with EINVAL. Reason: `hwctx->num_col` is set inside the runqueue
connect path that fires on first kernel launch (`aie2_ctx_runqueue.c:316`,
`ctx->num_col = part_num_col(part)`). Until `run.start()` is called,
num_col is zero, and the partition-range check at `aie2_pci.c:1635`
fails. The post-wait read at line 1740 succeeds fine.

Workable options (none implemented yet):
- Issue a no-op kernel run first to allocate the partition, then setup,
  then run the real kernel.
- Cache `core_cycles_start` from the post-wait read of run N, use it as
  the baseline for run N+1's delta computation. Ugly but free.
- Patch the driver to allocate the partition at hwctx creation rather
  than first kernel run. Bigger surgery, risks regressing real workloads.

**A different lifecycle bug we hit and survived**: writing PERF_CTRL0
via raw mailbox (no active context) wedged the firmware, leading to a
PM-suspend cascade that flushed our stale message into a freed-stack
callback → kernel page-fault Oops in `aie2_dbgfs_raw_resp_cb`. Root
cause was a stack-allocated response struct in our debugfs handler;
fixed in the same session by moving to heap and intentionally leaking
on timeout (commit included). Lesson: register writes that target
per-tile state with no allocated context will hang firmware — only
register *reads* are safe via the raw debugfs path without a partition.

References:
- src: `xdna-driver/src/driver/amdxdna/aie2_debugfs.c` (test_case04 added)
- src: `xdna-driver/src/driver/amdxdna/npu1_regs.c` (op-table entry added)
- src: `xdna-driver/src/driver/amdxdna/aie2_message.c:1772`
  (`aie2_rw_aie_reg`'s table check — same code, now satisfied)
- src: `xdna-driver/src/driver/amdxdna/npu4_regs.c:40`
  (Strix's entry — was correct, ours wasn't)
- finding: `docs/superpowers/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md`
  (the breakthrough doc, supersedes the calibration-doc's read_aie_reg analysis)
- task: #356 (re-opened; in-progress for the npu1_regs.c entry verification)
- task: #357 (re-scope: NPU4 still wanted for cross-arch confirmation, but Phoenix is no longer blocked)

#### EMU trace divergence has multiple distinct causes (#321 re-scope)

On `add_one_using_dma` post-fix, HW captures 67 trace events spanning
~12.6k cycles (slots 0/6/7 = PERF_CNT_0, LOCK_STALL,
INSTR_LOCK_ACQUIRE_REQ); EMU captures only 16 events spanning 264
cycles, all slot 7 (INSTR_LOCK_ACQUIRE_REQ). The original task #321
("trace-stop timing") was too narrow:

- **#321** (now narrowed): trace-stop timing — verify
  USER_EVENT_2 / `trace_done` broadcast fires at the right cycle in
  EMU vs HW.
- **#353**: EMU never emits `LOCK_STALL` events at all. Either the
  EMU's lock-acquire model treats stalls as instant, or the
  `LOCK_STALL` event ID isn't wired into the trace unit's
  event-detection path.
- **#354**: EMU never emits `PERF_CNT_0` anchor pulses. Either the
  perf counter isn't incrementing, or the threshold-match isn't
  emitting the event into the trace stream, or the kernel doesn't
  reach the threshold (depends on #355).
- **#355**: EMU's effective kernel-activity cycle range is ~50x
  shorter than HW. This is probably **upstream of #353 and #354** —
  if EMU runs the kernel in 250 cycles when HW takes 12.6k, neither
  PERF_CNT_0 thresholds (every 1024 cy) nor LOCK_STALL events have
  enough wall-time to fire.

**Recommended order**: investigate #355 first. If we fix the cycle
model to put HW and EMU on comparable timelines, #353 and #354 may
resolve as side effects. #321 (trace-stop timing) is probably the
last to address — it's only meaningful once events generate at the
right rate.

This sequence overlaps with #342 (differential validation of
calibrated `CycleCostModel`) — they're the same problem viewed from
different angles. Worth treating as one workstream.

## Pending work (in priority-suggested order)

1. **Verify #351 fix on full bridge sweep** — single test verified
   today; need to run the whole matrix to confirm no regressions and
   measure populated-trace rate across all ~75 tests. Single command:
   `./scripts/emu-bridge-test.sh --compile`.

2. **#355 → #354 → #353 → #321** — EMU trace divergence chain.
   Start with #355 cycle-range investigation; expect that the upstream
   cycle-cost gap is responsible for most of the missing-events
   symptoms. Coordinate with #342 differential validation.

3. **NPU1 driver patch experiment** (one-liner) — when we have time
   for a low-priority diagnostic, patch `aie2_rw_aie_reg` to skip the
   firmware-feature check, rebuild via DKMS (we do this anyway for
   driver updates), retry `bridge-trace-runner --read-perf-counter`.
   Also fix the partition-range lifecycle bug in
   `bridge-trace-runner.cpp` regardless. Expected return: `-EINVAL`
   from firmware "unknown opcode" rejection; small chance of useful
   signal. Cleanup if it doesn't pan out: revert the one-liner.

4. **#357 re-test on NPU4 when hardware arrives** — Strix firmware
   ≥ 6.24 implements `MSG_OP_AIE_RW_ACCESS`, so the same
   `bridge-trace-runner --read-perf-counter` flow that fails on Phoenix
   should succeed on Strix. If it does, on-NPU
   `Performance_Counter0` readback becomes a trace-independent ground
   truth and major chunks of our trace pipeline become optional rather
   than load-bearing.

5. **Calibration follow-ups** (#342, #347) — differential validation,
   broadcast event propagation latency measurement.

## Cleanup before next session

- `/tmp/claude-1000/livepatch-aie2/` — experimental kmodule, can be
  deleted (`/tmp` is wiped on reboot anyway, but for clarity).
- `/tmp/claude-1000/perfcnt-test/` — input/output/trace artifacts from
  the read_aie_reg experiment; same.
- The task list is current; tasks #353, #354, #355, #357 are pending
  and self-contained.

## What didn't get committed but is worth preserving

Nothing — all source changes from today are committed (eight commits,
last one being the cycle_cost.rs comment correction). The findings
docs include the corrected `read_aie_reg` analysis. Working tree is
clean apart from this NEXT-STEPS.md file.
