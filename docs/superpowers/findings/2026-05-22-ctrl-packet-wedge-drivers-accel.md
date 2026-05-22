---
name: 'ctrl_packet sweep still wedges Phoenix on the drivers/accel kernel tree; TDR recovery still incomplete, but driver reload recovers cleanly'
description: A full ctrl_packet event sweep run against the canonical drivers/accel/amdxdna kernel tree (kernel 7.0.9-custom, DKMS xrt-amdxdna 2.23.0, FW 1.5.5.391) wedged the NPU on add_one_ctrl_packet, both compilers -- the same test as the 2026-05-13 src/driver finding. The driver-tree swap does not fix the wedge: drivers/accel gates AIE2_NPU_COMMAND identically (FW>=5.8 on NPU1), so it takes the same CHAIN_EXEC_NPU op-0x18 path. The wedge mode is harder than the 2026-05-13 clean 32s silent-drop: aie2_tdr_detect fires, then the mgmt mailbox rejects DESTROY_CONTEXT (op 0x3, status 0x2000006) and CREATE_CONTEXT (op 0x2, status 0x2000003) -- the device cannot make new contexts. This confirms 2026-05-20 (TDR recovery incomplete on Phoenix) on the new tree with TDR recovery active by default (tdr_dump_only=false). New data point: a plain modprobe -r/modprobe reload recovered the device in 3s with no synchronize_srcu hang and no reboot -- this is the benign wedge class (HW runner exited, no D-state process pinning /dev/accel/accel0, mgmt mailbox responsive).
type: project
---

# ctrl_packet sweep wedge on drivers/accel -- 2026-05-22

## TL;DR

The hypothesis under test: "past NPU wedges may have been an artifact of
building the driver from the obsolete `src/driver/amdxdna` tree; the
canonical `drivers/accel/amdxdna` tree might have fixed them."

**Answer: no.** A full ctrl_packet event sweep against the `drivers/accel`
tree wedged the NPU on `add_one_ctrl_packet` (both compilers) -- the exact
test from `2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`. The wedge
is firmware-level; the driver tree does not change it.

Three parts characterized:

1. **Trigger** -- unchanged. `drivers/accel` gates `AIE2_NPU_COMMAND`
   identically (`FW >= 5.8` on NPU1), so Phoenix still takes the
   `CHAIN_EXEC_NPU` (op 0x18) path where the firmware silent-drop lives.
2. **Wedge mode** -- harder than 2026-05-13. Not a clean 32s silent-drop;
   `aie2_tdr_detect` fires, then the mgmt mailbox rejects context
   create/destroy. The device cannot allocate new hwctx.
3. **Recovery** -- the bright spot. `modprobe -r amdxdna && modprobe
   amdxdna` recovered the device in 3s, `xrt-smi validate` passed
   afterward. No `synchronize_srcu` hang, no reboot.

## Environment

| Axis | Value |
|---|---|
| Kernel | `7.0.9-custom` (build `#19`, 2026-05-21 14:09) |
| Driver tree | `drivers/accel/amdxdna` (canonical / mainline mirror) |
| Driver pkg | DKMS `xrt-amdxdna/2.23.0`, module ver `2e962728` |
| Module code | `drivers/accel` + mailbox UAF fix (PR #1348, code == `b1d58df`) |
| Firmware | `1.5.5.391` (protocol 5.8 per 2026-05-13) |
| Device | NPU1 / Phoenix, `0000:c6:00.1`, `[1022:1502]` |

The DKMS module is built by `make -C drivers/accel/amdxdna` (confirmed in
`/usr/src/xrt-amdxdna-2.23.0/dkms.conf`). The loaded module IS the
canonical tree.

### Module would not load at first -- RANDSTRUCT mismatch

Unrelated to the NPU, but worth recording: `amdxdna` failed to load with
`Exec format error` / `module amdxdna: .gnu.linkonce.this_module section
size must match the kernel's built struct module size at run time`. Cause:
the kernel was rebuilt (`vmlinuz` 14:25) *after* DKMS compiled the module
(`.ko` 12:36). `CONFIG_RANDSTRUCT` re-randomizes `struct module`'s layout
per kernel build, so a module compiled against the earlier build cannot
load into the later one. Fix: forced DKMS rebuild --
`dkms build xrt-amdxdna/2.23.0 -k 7.0.9-custom --force && dkms install ...
--force` (plain `dkms install --force` re-copies the stale artifact; it
does not recompile). `ryzen_smu` is broken the same way -- not relevant to
NPU work, left as-is.

## Static analysis: same gating, same path

`npu1_fw_feature_table` in `npu1_regs.c`, both trees:

```c
/* drivers/accel */
static const struct amdxdna_fw_feature_tbl npu1_fw_feature_table[] = {
	{ .major = 5, .min_minor = 7 },
	{ .features = BIT_U64(AIE2_NPU_COMMAND), .major = 5, .min_minor = 8 },
	{ 0 }
};

/* src/driver */
static const struct aie2_fw_feature_tbl npu1_fw_feature_table[] = {
	{ .feature = AIE2_NPU_COMMAND, .min_fw_version = AIE2_FW_VERSION(5, 8) },
	{ 0 }
};
```

Identical gating: `AIE2_NPU_COMMAND` enabled at `FW >= 5.8` on NPU1. With
FW 1.5.5.391 (protocol 5.8), both trees flip Phoenix onto the
`CHAIN_EXEC_NPU` (op 0x18) exec path. `drivers/accel` has no
`npu1_msg_op_tbl` -- the `msg_op_ver` table is an `src/driver`-ism;
mainline drives the path purely from the feature table -- but the result
is the same. Live confirmation: an `xrt-smi validate` run issued **45,020
op-0x18 submissions** with zero error.

## The sweep

`./scripts/emu-bridge-test.sh --sweep ctrl_packet` -- all 7 ctrl_packet
npu-xrt tests, both compilers, HW + EMU + Phase-5b event sweep.

**Functional bridge: 7/7 PASS** on HW for both compilers. Single-shot
submissions sail through.

**Phase 5b HW event sweep: `add_one_ctrl_packet` FAILED, both compilers.**
The sweep fires 25 event batches per test; the higher submission volume is
what rolls the firmware's probabilistic silent-drop (~4% per 2026-05-13)
often enough to hit. The sweep runner (`bridge-trace-runner`, kept alive
across batches via `--batch-stdin`) reported `HW batch 1/25: hw_ok=False`
then exited:

```
[sweep-lockstep] HW batch 1/25: hw_ok=False hw_cyc=None
[sweep-lockstep] sweep raised: RuntimeError: HW runner has exited
```

The other 4 sweeps (`_4_cores`, `_col_overlay` x2) ran OK -- they
completed *before* the wedge cascaded.

## The wedge (dmesg)

```
00:30:20  aie2_tdr_detect: TDR timeout detected
00:30:20  aie_send_mgmt_msg_wait: command opcode 0x3 failed, status 0x2000006
00:30:20  aie2_destroy_context_req: Destroy context failed, ret -22
00:30:24  aie2_tdr_detect: TDR timeout detected
00:30:24  aie_send_mgmt_msg_wait: command opcode 0x2 failed, status 0x2000003
00:30:24  aie2_xrs_load: create context failed, ret -22
00:30:24  aie2_alloc_resource: Allocate AIE resource failed, ret -22
00:30:24  aie2_hwctx_init: Alloc hw resource failed, ret -22
00:30:24  amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -22
   ... ~100 create_hwctx failures across 5 PIDs over 00:30:24-00:30:29 ...
00:30:29  (last NPU activity -- run moved to the emulator-only EMU arm)
```

Opcode map: `0x2` = `CREATE_CONTEXT`, `0x3` = `DESTROY_CONTEXT` (mgmt
mailbox). After TDR fires, both fail -- `DESTROY` with status `0x2000006`,
`CREATE` with status `0x2000003`. The `-22` is `-EINVAL`, the driver's
collapse of any non-OK mailbox status.

Key distinction from 2026-05-13: that finding's wedge was a *clean* 32s
silent-drop -- firmware drops the per-ctx response, the driver fake-signals
the fence on teardown, the test PASSes (just slow). This run hit the
harder mode: the mgmt mailbox stays *responsive* (it returns status codes,
it does not time out) but the firmware **refuses to allocate new
contexts**. The device is unusable for new work until the driver is
reloaded.

`bridge-trace-runner` *exited* rather than hanging in D-state -- because
the failure surfaces as an ioctl error (`-EINVAL` from
`create_hwctx_ioctl`), not as an unsignaled fence. This matters for
recovery (below).

## Recovery -- clean, no reboot

TDR recovery on `drivers/accel` runs by default (`tdr_dump_only=false`).
The dmesg shows `aie2_tdr_detect` *did* fire -- recovery was attempted --
and the device still ended wedged (create/destroy rejected). That confirms
`2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md` on the new tree.

But the wedge is the **benign class**:

- The HW runner **exited** -- no process pinned in D-state holding
  `/dev/accel/accel0`. So `modprobe -r` has nothing to block on; the
  `synchronize_srcu` wedge (the `AIE_RW_ACCESS` poison mode) does not
  apply.
- The mgmt mailbox stayed **responsive** throughout (it answered
  create/destroy with error status rather than going silent).

Result: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`
completed in **3.0s**. Clean re-probe (`[drm] Initialized
amdxdna_accel_driver 0.15.0`, full mailbox handshake, zero errors).
`xrt-smi validate` afterward: latency 136us, throughput 16490 op/s,
both tests PASSED.

So for this wedge class the recovery escalation stops at step 1 (driver
reload). No bridge PM-cycle, no SBR, no reboot.

## Verdict

| Question | Answer |
|---|---|
| Were past wedges a `src/driver`-tree artifact? | **No.** Firmware-level; `drivers/accel` hits them identically. |
| Does `drivers/accel` change the trigger? | No -- same `AIE2_NPU_COMMAND` gating, same op-0x18 path. |
| Does `drivers/accel` TDR recovery fix the wedge? | No -- TDR fires but recovery is incomplete (confirms 2026-05-20). |
| Is the device still recoverable without reboot? | **Yes** -- driver reload, 3s, for this (runner-exited, mailbox-alive) class. |

The `drivers/accel` tree is a cleaner, mainline codebase and is the right
thing to run -- but it is not a fix for the Phoenix firmware silent-drop.
The NPU remains wedgeable by ctrl_packet sweeps. The only clean fix is
still upstream: either firmware, or gating `AIE2_NPU_COMMAND` off for
NPU1 until the firmware silent-drop is resolved (the verification probe
proposed in 2026-05-13 -- force the legacy `CHAIN_EXEC_BUFFER_CF` op-0x12
path -- remains the cleanest test, and would now need to be done against
`drivers/accel`'s feature-table-only gating).

**Update 2026-05-22 (traced re-run).** The silent-drop has since been
captured directly, on both the driver verbose mailbox log and the kernel
tracepoints -- see
[`2026-05-22-chain-exec-npu-silent-drop-captured.md`](2026-05-22-chain-exec-npu-silent-drop-captured.md).
The "mgmt mailbox responsive but refuses to allocate contexts" symptom
recorded here is now decoded: a dropped op-0x18 exec leaves a compute
column whose job is hung, `DESTROY_CONTEXT` fails `AIE2_STATUS_MGMT_ERT_BUSY`
(0x2000006) because the management firmware cannot reclaim that column,
the column leaks, and once the pool is exhausted every `CREATE_CONTEXT`
fails `AIE2_STATUS_MGMT_ERT_NOAVAIL` (0x2000003).

## Environment cleanup done this session

The `/etc/modprobe.d/` config was stale for `drivers/accel` -- it set five
parameters that the new tree does not have (`autosuspend_ms`,
`fw_log_level`, `fw_trace_categories`, `fw_log_size`, `fw_trace_size`,
`poll_fw_trace`), all logged by modprobe as "unknown parameter ... ignored".

- `amdxdna-verbose-debug.conf` trimmed to `options amdxdna dyndbg=+p`
  (the one parameter that still applies).
- `amdxdna.conf` (the `autosuspend_ms=-1` pin) deleted -- `drivers/accel`
  has no such module param. Replaced with
  `/etc/udev/rules.d/71-amdxdna-no-autosuspend.rules`, which sets
  `power/control=on` on the PCI device (standard runtime-PM, the
  mainline mechanism).
- `drivers/accel` module parameters: `aie2_max_col`, `tdr_dump_only`,
  `tdr_timeout_ms`, `force_cmdlist`, `force_iova`.

CLAUDE.md's "amdxdna module options pinned" and "NPU recovery" sections
still reference `src/driver`-only knobs (`autosuspend_ms`, `tdr_dump_ctx`)
and need a pass.

## Artifacts

`build/experiments/2026-05-22-ctrl-packet-wedge-drivers-accel/`:
- `sweep.log` -- full bridge sweep output
- `dmesg-live.log` -- continuous ISO-timestamped dmesg across the run
  (~49 MB; `dyndbg=+p` is verbose -- consider gzip)
- `amdxdna.dmesg`, `amdxdna.trace` -- amdxdna-trace daemon snapshot.
  NOTE: `amdxdna.trace` is empty (0 entries). The `amdxdna-trace.sh`
  daemon hardcoded the obsolete `src/driver` trace subsystem name
  (`amdxdna_trace`); the `drivers/accel` tree registers tracepoints
  under `amdxdna`, so every event silently failed to enable. Fixed
  2026-05-22 (runtime subsystem resolution) -- future runs capture
  mailbox traffic.
- `add_one_ctrl_packet.{chess,peano}.sweep.log` -- the failed sweep logs
- `smoke-2.log` -- pre-sweep single-test path check (passed)
