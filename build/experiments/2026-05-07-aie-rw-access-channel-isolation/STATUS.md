# Session-end status (2026-05-07)

## What's installed right now

The **minimal patch** (remove `if (-ETIME) destroy mgmt_chann` from
`aie2_send_mgmt_msg_wait`) is:

- Applied as uncommitted edit to
  `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_message.c`.
- Built via `~/npu-work/xdna-driver/build/build.sh -release`.
- Installed via DKMS (`xrt-amdxdna/2.23.0`).
- Module `amdxdna` was reloaded; running on the NPU as of session end.

The architectural patch (`architectural-patch.diff`) was tested and
reverted — firmware refuses opcode 0x203 on user channel with
`AIE2_STATUS_INVALID_COMMAND`. The diff is preserved as a reference.

## Current device state

After the latest run, AIE_RW_ACCESS to memtile and unclaimed cells
will hang firmware (it's a firmware-side bug, the patch only prevents
the cascade). Subsequent AIE_RW_ACCESS calls also hang in this run's
state -- something about the V3/V4 hangs leaves the firmware unable
to service further AIE_RW_ACCESS.

It is unknown whether `modprobe -r amdxdna` will cleanly suspend the
firmware in this state -- with the patch, mgmt_chann survives so
`aie2_suspend_fw` will at least attempt to send. If FW suspend hangs,
modprobe will take 5s but should complete (driver waits on the
mailbox). SMU wedge from the previous failure mode (modprobe-r
suspend_fw fails ENODEV → SMU stop fails) should NOT happen because
mgmt_chann is alive.

If a clean test is needed, simplest path is reboot. If experimentation
with the recoverable state is wanted, try `pkexec modprobe -r amdxdna
&& pkexec modprobe amdxdna` first and observe whether the device
recovers without reboot. (NEW EXPERIMENT for next session.)

## Findings this session

### What works
- compute reg via AIE_RW_ACCESS: V0/V1/V2/V5 -- 100% reliable, exact
  round-trips, sub-100µs RTT
- compute DM via AIE_RW_ACCESS: V6 -- single read returned a value
  successfully
- minimal patch (mgmt_chann survives -ETIME): mailbox isolation
  achieved, M0 ran cleanly after V4 hung

### What's broken
- memtile reads via AIE_RW_ACCESS: register space (0x940F8 TIMER_LOW)
  AND data memory (0x10000) both hang firmware indefinitely. Rate
  hypothesis ruled out (1s pre-sleep didn't help).
- unclaimed cell reads (via AIE_RW_ACCESS): hang the firmware
  identically to memtile reads. Updates the prior
  per-tile-claim-authorization finding -- the "INVALID_PARAM" in that
  finding was likely a cascade artifact, not actual firmware
  authorization enforcement.
- architectural fix (route 0x203 through user channel): firmware
  refused with INVALID_COMMAND. 0x203 is hard-routed to mgmt only.

## Unfinished investigation threads

1. **Test driver-reload recovery without reboot**: with the minimal
   patch, modprobe -r/+ should not wedge SMU. Verify next session.

2. **Verify the firmware cascade after V3 hang**: does the firmware
   really need to be "reset" by reboot, or does it auto-recover after
   some quiet period? Could try a `sleep 60` after V4 hang and see if
   subsequent AIE_RW_ACCESS recovers.

3. **Step-debugger / emulator readback design implication**: with
   memtile reads broken on Phoenix, any debug-readback design has to
   either (a) use compute-tile reads only, (b) use a different
   readback mechanism (CDO write of "dump-to-buffer" sequence),
   or (c) wait for AMD to fix firmware. (b) seems most promising.

4. **Upstream the minimal patch to xdna-driver**: defer until we have
   a clean reproduction case and write-up suitable for AMD. The patch
   is 4 lines but the rationale needs the finding doc.

## File trail

- `/home/triple/npu-work/xdna-emu/docs/superpowers/findings/2026-05-07-aie-rw-access-memtile-dm-half-impl.md`
  -- main finding doc, updated this session with all the new
  evidence
- `/home/triple/npu-work/xdna-emu/docs/superpowers/findings/2026-05-06-aie-rw-access-tile-claim-authorization.md`
  -- prior finding (still valid for the per-tile filtering at
  authorization layer; the "INVALID_PARAM" claim about unclaimed
  cells needs revisiting)
- `architectural-patch.diff` (this dir) -- the user-channel routing
  attempt, preserved for reference
- `minimal-patch.diff` (this dir) -- the destroy-removal patch (mostly
  matches the actual installed change, minus my expanded comment)
- `RUNBOOK.md` (this dir) -- procedure for build/install/test
- `STATUS.md` (this file) -- session-end snapshot
