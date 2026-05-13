---
name: 'CHAIN_EXEC_NPU silently dropped by Phoenix firmware for some workloads; fake-fence-on-teardown masks it as 30s "completions"'
description: Trace evidence shows Phoenix NPU1 firmware (FW 1.5.5.391, protocol 5.8) accepts MSG_OP_CHAIN_EXEC_NPU (op 0x18) submissions for ctrl_packet workloads but never sends back the per-ctx mailbox response. Bridge-trace-runner waits 30s, gives up, destroys the ctx -- which triggers mailbox_release_msg -> notify_cb(NULL, 0) -> aie2_sched_resp_handler -> dma_fence_signal with cmd state ERT_CMD_STATE_TIMEOUT. Tests can still PASS if the kernel actually ran (output buffer correct), they just waste 30s per submission. The upstream NPU1 gating entry that enables this path was added in ae559d3 on assumption ("FW protocol 5.8 supports it") with no verification comment, unlike our verified MSG_OP_AIE_RW_ACCESS entry. Audit shows AIE2_PREEMPT is correctly NOT enabled on NPU1; AIE2_NPU_COMMAND is the only feature on NPU1 that may be enabled-but-broken. Bug #6 ("memtile_dmas hang") is plausibly a manifestation of the same firmware behavior.
type: project
---

# CHAIN_EXEC_NPU silent-drop on Phoenix -- 2026-05-13

## TL;DR

**Update 2026-05-13 (post-reboot probe):** A `ctrl_packet_reconfig`-
family probe with `tdr_dump_ctx=N` showed that the silent-drop
behavior is NOT general to ctrl_packet workloads. Out of
`reconfig` / `_1x4_cores` / `_4x1_cores` / `_elf` variants, only
`_4x1_cores (peano)` failed -- and that one had AMD-Vi IOMMU page
faults at addresses 0x0, 0x20, ..., 0xc0, indicating peano emitted
zeroed buffer base addresses. **That's a separate, identifiable
peano bug, documented in
`2026-05-13-ctrl-packet-reconfig-4x1-peano-iommu-fault.md`.**
The original `add_one_ctrl_packet (chess)` wedge had ZERO IOMMU
faults and remains the unresolved mystery -- mode A below.


`MSG_OP_CHAIN_EXEC_NPU` (opcode 0x18) once it reports protocol 5.8.
A single-line entry in `npu1_msg_op_tbl` plus an `AIE2_NPU_COMMAND`
feature flag in `npu1_fw_feature_table` together flip the driver's
exec path from `legacy_exec_message_ops` (uses `CHAIN_EXEC_BUFFER_CF`,
op 0x12) to `npu_exec_message_ops` (uses `CHAIN_EXEC_NPU`, op 0x18).

Trace evidence from a 26-minute bridge-test sweep (FW 1.5.5.391,
protocol 5.8, kernel 7.0.6-custom+, amdxdna 2.23.0):

- 561 / 585 jobs got proper IRQ responses on the per-ctx mailbox
  (latency: median 0ms, max 2ms).
- 24 / 585 jobs got **no IRQ at all** on the per-ctx mailbox.
  Latency from `xdna_job: ... job run` to `xdna_job: ... signaling
  fence`: 32.110-32.323s, distributed within 0.21s of each other.
  That tight clustering = a hardcoded timeout.
- All 24 slow jobs were ctrl_packet workloads (23 from a wedged
  `add_one_ctrl_packet (chess)` sweep, 1 from
  `ctrl_packet_reconfig_4x1_cores (peano)` during the HW phase).

The 30s number is the bridge-trace-runner's `run.wait(std::chrono::
seconds(30))` (bridge-runner/bridge-trace-runner.cpp:1910). When that
timeout fires, the runner destroys the ctx via `MSG_OP_DESTROY_CONTEXT`
(op 0x3) on the management mailbox. Driver-side ctx teardown iterates
pending messages on the per-ctx channel and calls `notify_cb(NULL, 0)`
on each (`amdxdna_mailbox.c:957` -> `mailbox_release_msg` -> line 267).
For job submissions, `notify_cb` is `aie2_sched_resp_handler`. With
NULL data it sets the cmd state to `ERT_CMD_STATE_TIMEOUT` (via
`aie2_ctx_cmd_health_data`) but **still calls `aie2_sched_notify`**,
which calls `dma_fence_signal(job->fence)` (`aie2_ctx.c:229`). The
syncobj wait in the runner then returns "success" 0.4ms later.

Net effect: tests can PASS if the actual kernel ran on the array
(output buffer matches), they just waste 30s per submission. Tests
hit BUDGET if 30s exceeds the per-test wall-clock budget.

## Trace evidence

Trace + dmesg captured by the new amdxdna-trace daemon (commit
`ac1d318`):
- `build/bridge-test-results/20260513/amdxdna.trace` (1.3 MB, 13289 events)
- `build/bridge-test-results/20260513/amdxdna.dmesg` (352 KB)

### Latency distribution

```
Jobs by job_run -> signaling_fence latency:
  <10ms:  561
  10ms-1s:  0
  1-10s:    0
  10-30s:   0
  >=30s:   24
```

Bimodal: microseconds or 32 seconds, nothing in between.

### Slow ctxs are deterministic per submission

```
ctx.86231.13   delta=32.239s   ctrl_packet sweep, event 1 of 24
ctx.86231.14   delta=32.110s   event 2
ctx.86231.15   delta=32.174s   event 3
... (12 more in the 32.17-32.20 range) ...
ctx.86231.24   delta=32.175s   event 23 (last completed before SIGTERM)
ctx.86231.25                   submitted at 1897.04s, in flight at snapshot
```

Mean 32.18s, std-dev 0.04s across 23 contexts. That's not firmware
slowness -- that's a userspace timer.

### Mailbox traffic during the 32s wait

Slow ctx (ctx.86231.13) full sequence:
```
1509.964215  mbox_set_tail: xdna_mailbox.135 id 0x1d000000 opcode 0x11 (CONFIG_CU)
1509.964241  xdna_job: ctx.86231.13 job run
1509.964280  mbox_set_tail: xdna_mailbox.135 id 0x1d000001 opcode 0x18 (CHAIN_EXEC_NPU)
                     ... 32 seconds of complete silence on mailbox.135 ...
1542.202390  mbox_set_tail: xdna_mailbox.145 id 0x1d0000f8 opcode 0x3  (DESTROY_CONTEXT, on mgmt mbox)
1542.202698  mbox_irq_handle: xdna_mailbox.145
1542.202727  mbox_set_head:   xdna_mailbox.145 id 0x1d0000f8 opcode 0x3
1542.202760  xdna_job: ctx.86231.13 signaling fence       <- fake!
1542.202769  xdna_job: ctx.86231.13 job free
```

Fast ctx (ctx.55316.1) full sequence:
```
309.553697  mbox_set_tail: xdna_mailbox.136 id 0x1d000000 opcode 0x11 (CONFIG_CU)
309.553720  xdna_job: ctx.55316.1 job run
309.553722  mbox_irq_handle: xdna_mailbox.136                <- IRQ in 2us
309.553738  mbox_set_head:   xdna_mailbox.136 id 0x1d000000 opcode 0x11
309.554891  mbox_set_tail:   xdna_mailbox.136 id 0x1d000001 opcode 0x18 (CHAIN_EXEC_NPU)
309.555148  mbox_irq_handle: xdna_mailbox.136                <- IRQ in 257us
```

Difference is on the per-ctx mailbox after the op-0x18 submission:
fast ctxs get an IRQ in microseconds, slow ctxs get nothing for 32
seconds. The mgmt mailbox (.145) keeps responding fine throughout
both cases.

## The fake-fence-on-teardown path

Reading the kernel paths:

`amdxdna_mailbox.c:957` -- `xa_for_each(&mb_chann->chan_xa, msg_id, mb_msg)
  mailbox_release_msg(mb_chann, mb_msg);` runs during channel teardown.

`mailbox_release_msg` at line 262-269:
```c
static void mailbox_release_msg(struct mailbox_channel *mb_chann,
                                struct mailbox_msg *mb_msg)
{
    MB_DBG(mb_chann, "msg_id 0x%x msg opcode 0x%x",
           mb_msg->pkg.header.id, mb_msg->pkg.header.opcode);
    mb_msg->notify_cb(mb_msg->handle, NULL, 0);
    kfree(mb_msg);
}
```

`aie2_sched_resp_handler` at `aie2_ctx.c:243` (the `notify_cb`):
```c
if (unlikely(!data)) {
    aie2_ctx_cmd_health_data(job->ctx, cmd_abo);  // sets cmd state TIMEOUT
    goto out;                                       // jumps to aie2_sched_notify
}
```

`aie2_sched_notify` at `aie2_ctx.c:218`:
```c
ctx->completed++;
reset_tdr_timer(job);
trace_xdna_job(&job->base, ctx->name, "signaling fence", job->seq, job->opcode);
job->state = JOB_STATE_DONE;
dma_fence_signal(job->fence);   // <-- this is the fake signal
```

So a job whose firmware response is missing gets:
- cmd state = `ERT_CMD_STATE_TIMEOUT`
- fence = signaled
- syncobj wait in userspace returns success
- TDR timer reset

The TIMEOUT state is technically observable by userspace, but
bridge-trace-runner doesn't appear to check it after `run.wait()`
returns (need to confirm).

## Capability gating audit

NPU1 enables exactly two configurable features via
`npu1_fw_feature_table` (`npu1_regs.c:79-82`):

```c
static const struct aie2_fw_feature_tbl npu1_fw_feature_table[] = {
    { .feature = AIE2_NPU_COMMAND, .min_fw_version = AIE2_FW_VERSION(5, 8) },
    { 0 }
};
```

Compare to NPU4 (Strix), `npu4_regs.c:56-60`:

```c
const struct aie2_fw_feature_tbl npu4_fw_feature_table[] = {
    { .feature = AIE2_NPU_COMMAND, .min_fw_version = AIE2_FW_VERSION(6, 15) },
    { .feature = AIE2_PREEMPT,     .min_fw_version = AIE2_FW_VERSION(6, 12) },
    { 0 }
};
```

| Feature | NPU1 (Phoenix) | NPU4 (Strix) | Driver enforces? | Status |
|---|---|---|---|---|
| `AIE2_NPU_COMMAND` | FW >= 5.8 -> ENABLED | FW >= 6.15 -> ENABLED | Yes (`aie2_msg_init` switches `exec_msg_ops`) | **Enabled but broken for some workloads on NPU1** |
| `AIE2_PREEMPT`     | not in table -> NOT enabled | FW >= 6.12 -> ENABLED | Yes (`aie2_message.c:1304` `-EOPNOTSUPP`) | Properly gated |

The opcode-level table (`npu1_msg_op_tbl`) has matching entries:
```c
const struct msg_op_ver npu1_msg_op_tbl[] = {
    { AIE2_FW_VERSION(5, 8), MSG_OP_CHAIN_EXEC_NPU },   // upstream, no verification comment
    { AIE2_FW_VERSION(5, 8), MSG_OP_AIE_RW_ACCESS },    // FIM43-Redeye, with verification comment
    { 0 },
};
```

So the only NPU1 capability flag that is "enabled by upstream
presumption with no verification" is `AIE2_NPU_COMMAND` /
`MSG_OP_CHAIN_EXEC_NPU`. Everything else is either not enabled, or
properly gated to fail.

## Provenance of the upstream entry

`git blame src/driver/amdxdna/npu1_regs.c`:
```
ae559d3e (Nishad Saraf 2025-12-15) { AIE2_FW_VERSION(5, 8), MSG_OP_CHAIN_EXEC_NPU },
5bb4d851 (FIM43-Redeye  2026-05-05) /* Phoenix firmware (verified at FW 1.5.5.391, protocol 5.8) implements
5bb4d851 (FIM43-Redeye  2026-05-05)  * MSG_OP_AIE_RW_ACCESS (opcode 0x203). Verified by sending the opcode... */
5bb4d851 (FIM43-Redeye  2026-05-05) { AIE2_FW_VERSION(5, 8), MSG_OP_AIE_RW_ACCESS },
```

Commit `ae559d3 "Use unified firmware version for protocol
compatibility (#922)"` -- no per-platform verification, no test, no
comment justifying the NPU1 entry. The upstream pattern appears to
be: "if FW protocol >= N, assume opcode is supported."

That assumption falsifies on Phoenix at FW 1.5.5.391 protocol 5.8 for
ctrl_packet workloads -- the firmware accepts the message (no error
returned through the mgmt mailbox, no `mailbox_send_msg` failure) but
never sends back the per-ctx response.

## The `aie2_hmm_invalidate` deadlock (post-cleanup wedge mode)

When the runner times out and tries to clean up, `munmap()` of the
staged buffer triggers the MMU notifier `amdxdna_hmm_invalidate`,
which calls the per-device hook `aie2_hmm_invalidate`
(`aie2_ctx.c:1017-1027`):

```c
void aie2_hmm_invalidate(struct amdxdna_gem_obj *abo, unsigned long cur_seq)
{
    struct drm_gem_object *gobj = to_gobj(abo);

    /*
     * Must wait forever, otherwise, memory was unmapped then FW might crash.
     * In case FW not response, TDR will terminal context execution and unref all BOs.
     */
    dma_resv_wait_timeout(gobj->resv, DMA_RESV_USAGE_BOOKKEEP,
                          false /* non-interruptible */, MAX_SCHEDULE_TIMEOUT);
}
```

`MAX_SCHEDULE_TIMEOUT` = wait forever. The driver design *explicitly*
relies on TDR to terminate the ctx if firmware doesn't respond.

But our `/etc/modprobe.d/amdxdna.conf` setting `tdr_dump_ctx=1` makes
TDR dump-only -- it never calls `aie2_rq_stop_all/restart_all` (that
recovery path is what we disabled because it wedges `synchronize_srcu`
on `modprobe -r`; see
`docs/superpowers/findings/2026-05-09-bridge-test-contention-and-trace-wedge.md`).

So the chain is:
1. Firmware silently drops CHAIN_EXEC_NPU response
2. Runner times out at 30s, calls munmap to clean up
3. munmap -> mmu_notifier -> aie2_hmm_invalidate -> dma_resv_wait_timeout(MAX_SCHEDULE_TIMEOUT)
4. dma_resv waits for the unsignaled fence (firmware never responded)
5. TDR fires every 2s, dumps ctx state, does NOT recover (because dump_only=1)
6. Process pinned in D-state forever, holding /dev/accel/accel0
7. modprobe -r would hang on synchronize_srcu (process never releases device)
8. **Reboot is the only escape**

Pid 95080 in this session was wedged exactly this way. Stack trace:
```
dma_fence_default_wait+0x180/0x2a0
dma_fence_wait_timeout+0x7c/0x5a0
dma_resv_wait_timeout+0x8d/0x120
amdxdna_hmm_invalidate+0xa6/0x1a0 [amdxdna]
__mmu_notifier_invalidate_range_start+0x2d7/0x3c0
unmap_vmas+0xbd/0x240
unmap_region+0x17e/0x2c0
__vm_munmap+0x10e/0x320
```

Pid 86231 (the earlier wedge from this session) was in
`drm_syncobj_array_wait_timeout` (the `run.wait(30s)` path) -- that
one cleared eventually because syncobj_wait IS bounded. The 95080
wedge is in the cleanup path AFTER the bounded wait, and the cleanup
path uses `MAX_SCHEDULE_TIMEOUT`.

Three interconnected issues:

| Issue | What | Mitigation |
|---|---|---|
| **A** | Firmware silently drops some `CHAIN_EXEC_NPU` responses | None yet (this is the root) |
| **B** | `aie2_hmm_invalidate` waits forever, depends on TDR | Driver design, not configurable |
| **C** | TDR `dump_only=1` skips the recovery B depends on | We chose this to avoid `synchronize_srcu` wedge in modprobe -r |

Without C the synchronize_srcu wedge happens; with C the
hmm_invalidate wedge happens. They're mutually exclusive in the
current design. Resolving A is the only clean fix.

### PCIe layer is healthy

Pre-reboot AER counter snapshot (all zero):
```
aer_dev_correctable: TOTAL_ERR_COR=0 (all sub-counters 0)
aer_dev_fatal:       (all sub-counters 0)
aer_dev_nonfatal:    (all sub-counters 0)
```

Confirms the silent-drop is at the firmware layer, not PCIe -- no
malformed TLPs, no completion timeouts at the link layer, no
correctable errors. Firmware genuinely chose not to respond.

### Pre-reboot artifacts saved

`build/experiments/2026-05-13-chain-exec-npu-pre-reboot/`:
- `pid95080-stack.txt` -- full kernel stack of the hmm_invalidate wedge
- `pid95080-meta.txt` -- /proc/PID/status, wchan
- `pid95080-lsof.txt` -- open file descriptors
- `ps-snapshot.txt` -- process tree at wedge time
- `dmesg-since-boot.log` -- complete dmesg before ringbuffer wraps
- `instrumented-sweep.log` -- full sweep log

## Open questions

1. **Why does the firmware silently drop the response?** It clearly
   processes some `CHAIN_EXEC_NPU` submissions (561 fast jobs in the
   trace did fine). What's different about the 24 that hung? Are
   they all ctrl_packet variants or something more specific? Check
   the inst_buf / save_buf / restore_buf contents of a slow vs fast
   submission.

2. **Does the kernel actually run on the array even when the
   completion is dropped?** If yes, output buffer comparison would
   succeed and the test "PASSes" -- which is what we observe in the
   HW phase. If no, output buffer comparison would fail. We have
   no direct evidence either way; need to instrument the data path.

3. **Is this related to Bug #6?** `memtile_dmas/*` tests TDR within
   ~9-15s on this same hardware (`2026-05-09-bridge-test-contention-
   and-trace-wedge.md`). 9-15s is short for a hardware fault but
   long for a firmware response timer. Bug #6 might be a stronger
   manifestation of the same firmware behavior -- the firmware drops
   the response AND something downstream (TDR timer, hwctx state)
   gets stuck rather than gracefully timing out.

4. **What does the NPU4 fast path look like for the same workloads?**
   We don't have NPU4 hardware, but if the firmware version table is
   accurate, NPU4 has had `CHAIN_EXEC_NPU` since FW 6.15 -- meaningfully
   longer history of validation. Worth comparing trace output if
   anyone with Strix can run the same sweep.

5. **Is `bridge-trace-runner` checking `ERT_CMD_STATE_TIMEOUT` after
   `run.wait()` returns?** If not, that's a separate runner bug --
   "wait succeeded but state is TIMEOUT" should be reported as
   failure, not silent 30s "PASS". Read confirms wait at line 1910
   but the state-check logic after is unclear.

## Probe matrix

The mystery has shape: firmware accepts the message via mgmt mailbox,
processes the command for 561/585 jobs without issue, but silently
drops the per-ctx response for 24 ctrl_packet jobs. Variables worth
controlling for:

| Axis | What it'd mean if it matters |
|---|---|
| `EXEC_NPU_TYPE` (PARTIAL_ELF=2 / PREEMPT=3 / ELF=4) | Phoenix supports some types via op 0x18 but not others |
| Compiler (chess vs peano) | Chess emits something firmware doesn't recognize; peano emits the safer subset |
| inst_buf contents | Specific instruction sequences trigger a firmware bug (overflow, malformed packet, undocumented form) |
| Per-ctx mailbox channel index (.135 vs .136) | Channel allocation has state we don't see; some channels work, others don't |
| ctx column placement (`@[1, 2]` vs other) | Phoenix has per-column firmware state |
| Submission rate | Back-pressure / queue overflow (less likely now that we pin to one job in flight) |

Concrete probes ordered cheap to costly:

1. **Trace a passing ctrl_packet test.** `ctrl_packet_reconfig (chess)`
   PASSed in HW phase. If it hits the same 32s pattern, we know
   "ctrl_packet always silently drops" -- the HW-phase pass was lucky
   (kernel ran, output matched, runner didn't notice TIMEOUT). If it
   completes in ms, `add_one_ctrl_packet` is structurally different
   from `ctrl_packet_reconfig`.

2. **Compare slot payloads.** The driver builds `cmd_chain_slot_npu`
   in `aie2_message.c` around lines 1148 (`EXEC_NPU_TYPE_PARTIAL_ELF`),
   1183 (`PREEMPT`), 1218 (`ELF`). Add a debug print of `npu_slot->type`,
   `inst_size`, `arg_cnt` for slow vs fast jobs. If the slow ones
   consistently use `EXEC_NPU_TYPE_ELF` (=4) while fast use
   `PARTIAL_ELF` (=2), that's the answer.

3. **Force legacy path.** Edit NPU1 tables to remove `AIE2_NPU_COMMAND`
   and `MSG_OP_CHAIN_EXEC_NPU`, rebuild driver, retry. Tells us whether
   the bug is opcode-level (CHAIN_EXEC_NPU specifically) or something
   deeper.

4. **Read `dump_ctx` output more carefully.** Slow ctxs all show
   `op: 0x0` in `aie2_dump_ctx`, which is suspicious -- most jobs
   would have a real ERT_CMD op. Is `op: 0x0` really `OP_NOOP`,
   or is the dump path stale (pre-`aie2_init_exec_req`)?

5. **Check firmware error/telemetry registers.** The amdxdna driver's
   `aie2_get_app_health` is "unsupported" on this FW (dmesg confirms),
   but Phoenix might expose error state via `MSG_OP_GET_TELEMETRY`
   (0x4) or similar. If we can poll for error count after a slow
   submission, we'd know whether the firmware is logging an internal
   error.

6. **Enable firmware-side tracing.** `MSG_OP_START_FW_TRACE`,
   `MSG_OP_SET_FW_TRACE_CATEGORIES` exist. If the firmware self-traces
   the silent-drop path, that's gold. Probably gated behind
   `aie2_is_supported_msg` which fails on NPU1 -- needs checking
   against `npu1_msg_op_tbl`.

Cheapest first probe is #1 (re-run with `-v ctrl_packet_reconfig`)
because we already have infrastructure; no driver rebuild needed.

## Verification plan

The fastest experimental confirmation:

1. **Force legacy path on NPU1.** Edit `src/driver/amdxdna/npu1_regs.c`
   to remove the `AIE2_NPU_COMMAND` entry from `npu1_fw_feature_table`
   and the `CHAIN_EXEC_NPU` entry from `npu1_msg_op_tbl`. Rebuild
   driver via `cd ~/npu-work/xdna-driver/build; ./build.sh -release
   -refresh_dkms`.

2. **Re-run the same sweep** with no other changes:
   ```bash
   nice -n 19 ./scripts/emu-bridge-test.sh --sweep 2>&1 | tee /tmp/claude-1000/legacy-path-sweep.log
   ```

3. **Predictions:**
   - If hypothesis is right: ctrl_packet sweeps complete in seconds
     each (using `MSG_OP_CHAIN_EXEC_BUFFER_CF` op 0x12 instead of 0x18),
     no 32s waits, no fake-fence path triggered.
   - If hypothesis is wrong: ctrl_packet still wedges -- silent-drop
     is in the firmware regardless of message opcode, or in something
     else entirely.

4. **Trace daemon stays on** -- the `amdxdna-trace.sh` daemon already
   covers sweep mode (commit `ac1d318`), so a second wedge will be
   captured at `build/bridge-test-results/<date>/amdxdna.{trace,dmesg}`
   for diffing.

If the legacy path fixes it, that's enough evidence to file a finding
upstream. If it doesn't, we know the gating fix isn't sufficient and
we need to look deeper at the firmware itself.

## What we shipped this session

- `ac1d318` -- sweep: enforce one-job-in-flight; extend amdxdna trace
  to cover sweep mode. (This still helps narrow the wedge regardless
  of root cause -- without these changes the trace daemon would have
  exited before sweep mode and we'd have nothing.)
- This finding doc.

## Pickup pointer

Maya is rebooting (current bridge-trace-runner is in D-state from the
SIGTERM mid-CHAIN_EXEC_NPU-submit cleanup). After reboot:
1. Pull this doc back up.
2. Decide on the verification approach (probably the legacy-path
   force-and-test above, but Maya wanted to think about deeper probes
   first -- "why is the firmware not replying" is the real question).
