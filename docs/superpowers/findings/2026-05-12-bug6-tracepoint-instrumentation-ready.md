---
name: bug #6 tracepoint instrumentation ready, pass-state baseline captured 2026-05-12
description: Built tools/bug6-trace.sh which uses the amdxdna driver's existing kernel tracepoints (xdna_job, mbox_set_tail/head, mbox_irq_handle, mbox_rx_worker) to capture the full host-firmware-IRQ-fence chain in a single pkexec call. Captured the pass-state baseline trace for writebd. Reframed two prior assumptions: (1) "total completed jobs N" in dmesg is the post-fini count and is normal for any passing test, not a hang-specific smoking gun; (2) the bug 6 trigger is "first writebd after sufficient device idle / fresh fd state," not specifically "fresh boot" -- a reboot might just give us more pass-state. Investigation now gated on natural recurrence of the hang, at which point a single command captures everything needed for diff.
type: finding
---

# Bug #6 -- tracepoint instrumentation ready, pass-state baseline captured

Picks up from
[2026-05-12-bug6-state-dependent-post-num_rqs-fix.md](2026-05-12-bug6-state-dependent-post-num_rqs-fix.md)
which paused with "next investigation = driver-side instrumentation."
This session built that instrumentation, but using **existing kernel
tracepoints** rather than new printks -- no driver code change required.

## TL;DR

- The amdxdna driver already has tracepoints
  (`amdxdna_trace/xdna_job`, `mbox_set_tail`, `mbox_set_head`,
  `mbox_irq_handle`, `mbox_rx_worker`, `uc_irq_handle`, `uc_wakeup`)
  that cover every boundary in the submit-firmware-IRQ-fence chain
  for a kernel job.  No code change needed to instrument bug 6.
- Built [tools/bug6-trace.sh](../../../tools/bug6-trace.sh) which
  enables those events, runs a single test under timeout, and
  snapshots trace + dmesg in one pkexec call.
- Captured the **pass-state baseline trace** for `writebd` -- driver-
  level kernel execution is 1.33 ms end-to-end, with a clean fixed
  pattern (see "Baseline" below).  Any hang trace can be diffed
  against this to localize the failing boundary in seconds.
- **Reframed** the morning's "HW reports 1 completed but host never
  sees" theory: `ctx.NNN.M total completed jobs N` is logged by
  `aie2_ctx_fini` in `aie2_ctx.c:695` and is the post-mortem count
  of `aie2_sched_notify` calls during the ctx lifetime.  For any
  passing writebd, that count is **1** -- it's normal, not a smoking
  gun.  The morning's framing made the hang look like "missing
  user-job completion among 1 reported completion," but actually
  it's just "the 1 expected completion didn't happen in time."
- **Reframed** the bug 6 trigger condition: morning notes show the
  initial hang was "session start," and ONE modprobe reload cleared
  it.  No fresh-boot specificity proven.  The trigger is more likely
  "first writebd after sufficient device idle / fresh fd state,"
  which a fresh boot is just one path to.  A reboot today might
  give us only more pass-state.  Investigation is now gated on
  **natural recurrence** rather than forcing the trigger.

## What got built

### `tools/bug6-trace.sh`

```
./tools/bug6-trace.sh <label> <test-dir> [timeout-sec]
```

Single pkexec invocation that:

1. Sizes trace ringbuffer (16 MB / CPU), enables the 7
   amdxdna_trace events, clears + starts tracing.
2. Drops to user via `runuser` and runs `test.exe -x aie.xclbin
   -k MLIR_AIE -i insts.bin` under `timeout`.
3. Stops tracing, snapshots `/sys/kernel/tracing/trace` and `dmesg
   -T --since=<start>`, chowns everything back to user.

Outputs to `xdna-emu/build/experiments/bug6/<label>.{trace,dmesg,test.log,meta,rc}`.

The single-pkexec design matters -- two sequential pkexec calls
(setup + snapshot) leave a wide auth-prompt-delay gap during which
the test can finish before the snapshot is taken; the timing
artifacts ruined the first run.  See
[[lump-pkexec-into-single-call]] memory entry.

### Pass-state baseline (writebd, chess, current `5CA2BD72` srcversion)

Captured 23 trace events for a passing writebd run:

```
mailbox 145 = MGMT channel,    mailbox 136 = USER channel

T+0     mbox_set_tail mgmt   id=0x..27  op=0x02     hwctx start
T+27us  mbox_irq_handle mgmt
T+67us  mbox_set_head mgmt   id=0x..27  op=0x02     <- FW ack

T+212us mbox_set_tail mgmt   id=0x..28  op=0x106    mgmt
T+260us mbox_set_head mgmt   id=0x..28  op=0x106    <- FW ack

T+330us mbox_set_tail USER   id=0x..00  op=0x11     config_ctx
T+330us xdna_job seq#0 "job run" op=0  (OP_EXECBUF) <- drm_sched submits
T+395us mbox_set_head USER   id=0x..00  op=0x11     <- FW ack

T+1.15ms mbox_set_tail USER  id=0x..01  op=0x18     EXEC kernel
T+1.29ms mbox_set_head USER  id=0x..01  op=0x18     <- FW ack
T+1.32ms xdna_job seq#0 "signaling fence"           <- FENCE -> userspace wakes
T+1.33ms xdna_job seq#0 "job free"

T+2.0ms mbox_set_tail mgmt   id=0x..29  op=0x03     hwctx teardown
```

Total driver-level execution: ~1.33 ms.  USER-channel opcodes
0x11 (CONFIG_CU/equivalent) and 0x18 (EXEC) are the two messages
that matter for "did the kernel run."  Saved at
`build/experiments/bug6/pass-baseline-v2.{trace,dmesg,test.log,meta}`.

### Discrimination ladder for hang traces

Compare a hang trace against the baseline above; the **first** trace
event the hang is missing tells us the failing boundary:

| Missing event after | Failure mode |
|---|---|
| `mbox_set_tail USER op=0x18` | Driver never submitted the user kernel (host-side bug, before mailbox post) |
| `mbox_irq_handle` after that set_tail | Firmware stopped responding (firmware hang or IRQ delivery broken) |
| `mbox_rx_worker` after IRQ | RX worker scheduling broken (kworker / softirq issue) |
| `mbox_set_head` for op=0x18 | Response received but msg_id dispatch lost it |
| `xdna_job "signaling fence"` for the right seq | Handler ran but `dma_fence_signal` skipped or wrong job (handler bug in `aie2_sched_resp_handler`) |
| (everything fires, userspace still hangs) | Fence-to-syncobj-to-XRT propagation issue (drm_sched fence wiring) |

Each hypothesis was previously listed under "What's unknown" in the
prior finding doc; the trace pipeline now answers them in one shot.

## What's NEXT

**On next session, before running anything device-touching, run:**

```bash
./tools/bug6-trace.sh first-attempt \
    ~/npu-work/mlir-aie/build/test/npu-xrt/memtile_dmas/writebd/chess 30
```

If that hangs (rc=124, elapsed ~30s), we have the hang trace.
Diff against `pass-baseline-v2.trace`, walk the discrimination
ladder above, and the failing boundary tells us exactly which code
path to investigate next.

If it passes (rc=0, elapsed <2s), the device is already in
pass-state.  Re-label as `pass-NNN` and try again on the next
fresh session.

## Why we didn't reboot today

Earlier session evidence (NOTES.md from `build/experiments/2026-05-12-bug6-rebase-attempt`):

- Morning: writebd hung at *session-start*, not after reboot
  specifically.  ONE modprobe reload cleared it.
- Today (evening): writebd has been passing reliably (6+ runs,
  0.21 s each).  Uptime 1h7m, lots of recent device activity --
  we're firmly in "warm/exercised" state.

The bug 6 trigger is *more specific* than "post-boot" -- it's some
combination of device idleness and ctx-fd state that a reboot is
just *one* way to reach.  Rebooting today, with all the recent
activity, would likely have given us another pass-state run.  Maya
flagged this correctly when we discussed wrap-up.

## Cross-references

- [2026-05-12-bug6-state-dependent-post-num_rqs-fix.md](2026-05-12-bug6-state-dependent-post-num_rqs-fix.md) -- prior pause point, with the hypothesis list this work makes testable
- `build/experiments/2026-05-12-bug6-rebase-attempt/NOTES.md` -- morning context including "hang at session start, cleared by one reload"
- `build/experiments/2026-05-12-bug6-rebase-attempt/POST-REBOOT-FINDINGS.md` -- the reboot that exposed the (now-fixed) num_rqs=0 bug
- `tools/bug6-trace.sh` -- the trace capture script
- `build/experiments/bug6/pass-baseline-v2.{trace,dmesg,test.log,meta}` -- the captured baseline
- xdna-driver `aie2_ctx.c:217` (`aie2_sched_notify`) -- the only path that increments `ctx->completed`; lines 432-434 show OP_NOOP increments without firmware roundtrip
- xdna-driver `amdxdna_trace.h` -- the tracepoint definitions
