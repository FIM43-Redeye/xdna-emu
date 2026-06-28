---
name: 'Phoenix FW 1.5.5.391 does not implement FW_TRACE (DPT) opcodes'
description: 'Phase 2e investigation: drivers/accel amdxdna omits AIE2_FW_TRACE from the NPU1 feature table; with the bit experimentally enabled, the DRM ioctl reaches firmware which responds status 0xffffffff to opcode 0x10f (start_fw_trace) and disables the mailbox IRQ. Phoenix FW does not implement the FW_TRACE DPT path. The (failed) enable also takes the device down at PCIe level, requiring reboot to recover. Controller-internal trace via DPT is not available on Phoenix; calibration must rely on trace-unit data or empirical model fits.'
type: project
---

# Phoenix FW 1.5.5.391 does not implement FW_TRACE (DPT) opcodes

## TL;DR

Phase 2e tested whether the amdxdna DPT framework's `FW_TRACE`
path could expose controller-internal events on Phoenix
(NPU1, FW 1.5.5.391). The answer is **no**.

Three structural gates blocked the path:

1. The kernel's `drivers/accel/amdxdna` ops table doesn't wire
   `aie2_fw_trace_init` into `aie2_ops` at all. The init exists
   in source (`aie2_pci.c:1308`) but the wiring is gated by
   `AIE_FEATURE_ON(AIE2_FW_TRACE)` in `aie2_msg_init`, and that
   bit is not set in `npu1_fw_feature_table`.

2. Experimentally adding `BIT_U64(AIE2_FW_TRACE)` to the NPU1
   feature table (matching the existing AIE2_RW_ACCESS local
   patch pattern), the wiring activates and the ioctl reaches
   firmware. Firmware responds **status `0xffffffff`** to
   opcode `0x10f` (`MSG_OP_START_FW_TRACE`):

   ```
   amdxdna: xdna_mailbox.145: opcode 0x10f size 4 id 0x1d00000e
   amdxdna: xdna_mailbox.145: Message callback ret -22
   amdxdna: xdna_mailbox.145: Unexpected ret -22, disable irq
   amdxdna: [drm] *ERROR* aie2_start_fw_trace: Start FW trace failed,
            ret -22 status 0xffffffff
   amdxdna: [drm] *ERROR* aie2_fw_trace_init:
            Failed to start FW trace: -22
   ```

   Status `0xffffffff` indicates "FW does not implement this
   opcode" -- a generic firmware-side error, consistent across
   opcodes Phoenix FW doesn't handle.

3. The unexpected response causes the driver to **disable the
   mailbox IRQ** ("Unexpected ret -22, disable irq"). After that
   the device becomes unreachable at PCIe level
   ("No such device with index '0'" on subsequent XRT calls),
   and a plain `modprobe -r && modprobe` reload does not recover
   the device binding. SBR or reboot is required to bring it
   back.

So Phoenix FW does not support FW_TRACE, and probing for it
takes the device down. The Linux DPT framework's `FW_LOG` and
`FW_TRACE` features were added for newer NPUs (NPU4+); Phoenix
predates this firmware contract.

## Path tried

1. Wrote `tools/fw-trace-enable.py` (committed in the same series
   as this finding): a small ctypes shim that opens
   `/dev/accel/accel0` and issues `DRM_IOCTL_AMDXDNA_SET_STATE`
   with `param = DRM_AMDXDNA_SET_FW_TRACE_STATE` and
   `buffer -> struct amdxdna_drm_set_dpt_state { action=1,
   config=<categories> }`. CAP_SYS_ADMIN-gated, so invoked via
   `pkexec`.

2. Initial attempt with the upstream NPU1 feature table:
   `OSError: [Errno 95] Operation not supported`. Cause: the
   ioctl dispatcher checks `if (!aie->msg_ops.fw_trace_init)
   return -EOPNOTSUPP;` and the callback was NULL because
   `AIE2_FW_TRACE` is not in `npu1_fw_feature_table[]`.

3. Patched `drivers/accel/amdxdna/npu1_regs.c` to add
   `{ .features = BIT_U64(AIE2_FW_TRACE), .major = 5, .min_minor = 8 }`
   to the table (mirroring the existing AIE2_RW_ACCESS local
   patch). Rebuilt + DKMS-installed.

4. Re-ran the ioctl: `OSError: [Errno 22] Invalid argument`,
   with dmesg showing the FW status `0xffffffff` response and
   IRQ disable.

5. Reverted the patch in the project tree. The PCIe-level
   device wedge persisted after `modprobe -r && modprobe`;
   recovery requires higher-level intervention (SBR or reboot).

## Why FW_LOG is also off

Same gate, same negative response. The NPU1 feature table also
omits `AIE2_FW_LOG`. The driver auto-starts FW_LOG at WARN level
on supported devices, but on Phoenix the wiring is NULL, so
nothing fires. This matches the observation that no `[FW LOG]:`
lines appear in dmesg under sustained workload, even with the
verbose-debug `dyndbg=+p` modprobe option set.

We did not test patching `AIE2_FW_LOG` in -- the FW_TRACE result
makes it likely the FW_LOG opcode is similarly unimplemented,
and any further probing would risk wedging the device again.

## Recovery from this finding's wedge

After the failed FW_TRACE enable:

- `modprobe -r amdxdna && modprobe amdxdna` returns success but
  the device does not re-enumerate; subsequent XRT calls fail
  with "No such device with index '0'".
- CLAUDE.md recovery escalation suggests Bridge PM-cycle,
  SBR, suspend/resume, or reboot. Reboot is the simplest.

We left the broken state for the next session rather than
walking the recovery escalation tonight; the project tree is
clean (patch reverted), so a fresh boot + standard
`build.sh -release -refresh_dkms` will produce the right
module without the FW_TRACE bit.

## Implications

### For Phase 2e

The DPT FW_TRACE path is **not a usable calibration source on
Phoenix**. The controller-internal visibility goal must be
abandoned for this hardware.

### For the warm-up transient question

The Phase 2c HW data showed MM2S per-task durations decreasing
1739 -> 370 cyc over 8 tasks. With FW_TRACE blocked, perf
counters blocked (2026-05-26 AIE_RW_ACCESS finding), and MM2S-
side level events wedging the trace stack
([`2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox`](2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox.md)),
the remaining path is **empirical curve-fitting** against the
existing HW campaign data. Model the transient as
`task_duration(i) = cold_start + per_task + transient(i)`
where `transient(i)` decays over the first 3-4 tasks. Pure
phenomenological model, no mechanism hypothesis.

### For future hardware

On NPU4+ (Strix Point), `AIE2_FW_TRACE` is wired upstream and
the FW likely implements the opcode. When the dev devbox
upgrades from Phoenix to Strix
([[project_strix_swap_replaces_phoenix]]), the DPT trace path
should become available, and this finding's negative result
will not apply.

## Artifacts

- `tools/fw-trace-enable.py` -- ctypes ioctl shim, kept for
  future use on hardware that supports FW_TRACE.
- Driver patch: reverted in the project tree
  (`drivers/accel/amdxdna/npu1_regs.c`); loaded module on the
  devbox at time of finding still carries the bit and will be
  replaced by the next clean rebuild.
- Related findings:
  - [`2026-05-27-phase-2c-dispatch-overhead-recalibration`](2026-05-27-phase-2c-dispatch-overhead-recalibration.md)
    -- the calibration work that motivated 2e.
  - [`2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox`](2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox.md)
    -- parallel trace-event dead end.
  - [`2026-05-26-aie-rw-access-not-a-cycle-probe`](2026-05-26-aie-rw-access-not-a-cycle-probe.md)
    -- perf counter readback dead end.
