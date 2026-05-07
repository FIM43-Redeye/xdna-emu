# AIE_RW_ACCESS channel-isolation experiment runbook

## Context

`aie2_send_mgmt_msg_wait` destroys `ndev->mgmt_chann` on `-ETIME`
timeout. AIE_RW_ACCESS (0x203) goes through this path. Memtile reads
hang the firmware indefinitely, the 5s timeout fires, mgmt_chann dies,
device enters "no-mailbox" state, modprobe -r cascades to SMU wedge.

Two patches available:

1. **Architectural** (`architectural-patch.diff`): Route AIE_RW_ACCESS
   through the per-hwctx user mailbox (`ctx->priv->mbox_chann`) instead
   of mgmt. A hung memtile read only kills that hwctx, not device mgmt.
   This is the **applied state on disk** as of session end (uncommitted
   in xdna-driver working tree).

2. **Minimal** (`minimal-patch.diff`): Just remove the
   destroy-on-ETIME from `aie2_send_mgmt_msg_wait`. Smaller change,
   doesn't depend on firmware accepting 0x203 on user channel. Risk:
   if the protocol genuinely needs the destroy on timeout (out-of-seq
   late response), leaving the channel up could confuse future msgs.

## Apply / build / install (post-reboot)

The architectural patch is **already applied** to xdna-driver source.
To build and install:

```bash
cd /home/triple/npu-work/xdna-driver/build
./build.sh -release
pkexec bash -c '
  dkms remove xrt-amdxdna/2.23.0 -k $(uname -r) && \
  dkms install xrt-amdxdna/2.23.0 -k $(uname -r) && \
  modprobe -r amdxdna && modprobe amdxdna'
```

Then test:

```bash
cd /home/triple/npu-work/xdna-emu/tools/validate-readback/build
./validate-readback
```

Expected outcome:
- L0/L1/V0/V1/V2/V5/V6/V3 pass (compute reg works, compute DM TBD,
  unclaimed cell still throws EINVAL).
- V4 (memtile read) returns -ETIMEDOUT (-62) cleanly, but the
  device REMAINS USABLE for subsequent loads. Specifically:
  - `xdna_mailbox.MGMT` (irq 145) stays alive.
  - Only the per-hwctx user channel dies.
  - A second `validate-readback` invocation succeeds at xclbin-load
    + dummy run, can repeat the test.

## If firmware refuses 0x203 on user channel

The per-hwctx user channel is created when the hwctx starts. The
firmware accepts `0x11` (CONFIG_CU) and `0x18` (CHAIN_EXEC_NPU) on it
already. But `0x203` may be hard-routed to mgmt by the firmware. If
so, V0/V1 will fail (any AIE_RW_ACCESS will fail) -- look for
INVALID_COMMAND status in dmesg.

Revert the architectural patch:
```bash
cd /home/triple/npu-work/xdna-driver
git checkout src/driver/amdxdna/aie2_message.c \
              src/driver/amdxdna/aie2_pci.c \
              src/driver/amdxdna/aie2_pci.h
```

Then apply the minimal patch instead:
```bash
patch -p1 < /home/triple/npu-work/xdna-emu/build/experiments/2026-05-07-aie-rw-access-channel-isolation/minimal-patch.diff
```

Rebuild + reinstall as above. Test again. With the minimal patch:
- AIE_RW_ACCESS goes through mgmt as before
- V4 hang -> -ETIMEDOUT, but mgmt_chann is NOT destroyed
- Subsequent ops still work (probably; depends on whether protocol is corrupted)

## Revert everything

If both patches break things:
```bash
cd /home/triple/npu-work/xdna-driver && git checkout src/driver/amdxdna/
```

Then rebuild + DKMS as above to restore upstream behavior.
