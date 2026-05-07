---
name: AIE_RW_ACCESS to memtile DM is a half-implementation -- writes ack, reads hang
description: Reads/writes via xrt::hw_context to memtile data-memory addresses (offset 0x10000 confirmed) on Phoenix expose a firmware bug -- writes return SUCCESS without verifying the store, but readbacks never respond. The user-context mailbox channel times out after 5s and is destroyed by the driver, and the cascade kills mgmt_chann too, wedging the SMU.
type: finding
---

# AIE_RW_ACCESS memtile-DM half-implementation

## TL;DR

`xrt::hw_context::write_aie_reg` to a CLAIMED memtile's data-memory
address returns SUCCESS at the firmware mailbox layer (driver logs the
write, response 0x203 size 8 with success status). But the matching
`read_aie_reg` of the same address NEVER GETS A RESPONSE: firmware
silently drops the request, and after 5s the driver destroys the
user-context mailbox channel. The destruction cascade reaches mgmt_chann
during modprobe-r, producing `aie2_suspend_fw: Failed to suspend fw,
ret -19` and `smu cmd 4 failed, 0xff`. Reboot is required.

This is the **worst possible failure mode** for a debug/readback API:
writes claim to succeed without performing any verification, and
reads — the only way to verify writes — silently hang. We have no way
to confirm whether the writes went anywhere real.

## How it manifested

`tools/validate-readback/validate-readback.cpp` had a V4 test that did a
write+read round-trip on memtile (col 0, row 1) at offsets 0x10000 and
0x10100 (using the kernel `add_one_using_dma`'s claimed memtile, so
authorization was not the issue).

dmesg (T14978 = the test's PID, mailbox.145 = user-context channel):

```
[249.341098] AIE tile write: ctx 1 col 0 row 1 addr 0x10000 size 4
[249.341112] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001c
[249.341169] xdna_mailbox.145: resp opcode 0x203 size 8 id 0x1d00001c   <-- ACK
[249.341184] AIE reg write ctx 5 row 1 col 0 addr 0x10000 value 0xbeef00aa

[249.341196] AIE tile write: ctx 1 col 0 row 1 addr 0x10100 size 4
[249.341209] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001d
[249.341268] xdna_mailbox.145: resp opcode 0x203 size 8 id 0x1d00001d   <-- ACK
[249.341284] AIE reg write ctx 5 row 1 col 0 addr 0x10100 value 0xbeef00bb

[249.341296] AIE tile read: ctx 1 col 0 row 1 addr 0x10000 size 4
[249.341309] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001e   <-- never responded
[254.814693] xdna_mailbox.145: IRQ disabled and RX work cancelled        <-- 5.5s later
[254.814784] aie2_rw_aie_reg: AIE reg read failed, ret -62               <-- ETIMEDOUT
```

Both writes complete in microseconds. The read on the same address
hangs indefinitely. The driver kills the channel after the 5s mailbox
timeout. modprobe-r then cascades to SMU wedge.

## What address space triggers it

Memtile data-memory range. Per `xaiemlgbl_params.h`:
- `XAIEMLGBL_MEM_TILE_MODULE_DATAMEMORY = 0x00000` (DATAMEMORY base)
- 512 KB total -- DM occupies `0x00000-0x7FFFF` of the memtile's
  per-tile address space.

The failing offset (0x10000) is 64 KB into DM. We have not yet
characterized whether the same hang occurs at all DM offsets or only
some; the test that would have probed this would have caused another
SMU wedge. Treat **all of memtile DM** as suspect via AIE_RW_ACCESS
until proven otherwise.

Memtile register space (`>0x80000`) status is currently TBD; the
redesigned `validate-readback` reads `XAIEMLGBL_MEM_TILE_MODULE_TIMER_LOW`
(`0x940F8`) post-reboot to find out.

Compute-tile **register** access works fine (V0/V1/V2 confirmed
`PERF_COUNTER0` and `TIMER_LOW` round-trips at `(col, 2)`). Compute-tile
**DM** access is also TBD; the redesigned test reads compute DM at
offset `0x100` to characterize.

## Why "half-implementation" not "incomplete"

- An "incomplete" address would return `AIE2_STATUS_INVALID_PARAM`
  fast, the driver would translate to `-EINVAL`, and the channel
  would survive (we observed this for unclaimed cells).
- A "half-implemented" address ACCEPTS the write opcode-side
  (firmware ACKs with SUCCESS, no error) but the read opcode is
  silently unhandled. Firmware never replies, leaving the request
  in the mailbox queue until timeout.

The write-ack-without-verification means we cannot trust *any* memtile
DM write through this path. The firmware may be discarding the write,
writing to the wrong address, or actually writing — we have no way to
distinguish without a working read.

## Recovery cost

Each memtile-DM read attempt costs:
- 5s of wall time (mailbox timeout)
- The user-context mailbox channel (destroyed at timeout)
- Eventual SMU wedge during the next modprobe-r (mgmt fini cascade)
- A reboot to recover the SMU

This is significantly worse than the unclaimed-cell finding
(2026-05-06-aie-rw-access-tile-claim-authorization.md): unclaimed cells
fail-fast and only wedge the SMU through accumulated EINVALs. The
memtile-DM read wedges in a single shot.

## Implications

For step-debugger / emulator-readback designs:

1. **AIE_RW_ACCESS cannot be relied on for memtile DM access.** Even
   if the xclbin claims the memtile, DM reads hang.

2. **Writes through AIE_RW_ACCESS are not self-verifying.** Even on
   address ranges where reads work (compute-tile registers), the SUCCESS
   response means "firmware accepted the request," not "the write was
   committed and is observable." For memtile DM specifically, the
   write returns SUCCESS but the verification path is broken.

3. **Step-debugger designs that depend on reading arbitrary tile state
   need an alternative path** for memtile DM. Candidates: instrument
   the kernel to write its DM contents to a buffer the host can read
   via a normal output BO; use shim-DMA-mediated readback; or explore
   whether the firmware's coredump path (MSG_OP_GET_COREDUMP, currently
   not implemented on Phoenix per
   2026-05-06-npu1-msg-op-capability-survey.md) would have exposed this.

4. **Don't probe memtile DM at all in casual diagnostics.** Each probe
   destroys a mailbox channel and likely the SMU. Use the
   `--probe-dm-danger` flag in `validate-readback` only to reproduce
   this finding deliberately, then reboot.

## What it does NOT prove

- Whether the hang is offset-specific (some DM addresses might work).
  Cannot probe without paying the SMU-reboot cost.
- Whether memtile **register** access works. The redesigned V4 test
  in `validate-readback` reads `MEM_TILE_TIMER_LOW (0x940F8)` to
  find out.
- Whether compute-tile DM access works. The redesigned V6 test reads
  compute DM at offset `0x100` to find out.
- Whether shim-tile (row 0) DM/registers are accessible at all. Not
  exercised yet.
- Whether the bug is FW 1.5.5.391 specific. We have only this firmware
  loadable.

## See also

- `xdna-driver` `aie2_message.c:68-72` -- driver-side timeout path that
  becomes -ETIMEDOUT (-62).
- `xdna-driver` `aie2_pci.c:1564` -- AIE_RW_ACCESS read IOCTL handler.
- `aie-rt` `xaiemlgbl_params.h` -- memtile register layout.
- `2026-05-06-aie-rw-access-tile-claim-authorization.md` -- companion
  finding: per-tile-claim authorization. Together with this finding,
  characterizes AIE_RW_ACCESS limits on Phoenix.
- `2026-05-06-npu1-msg-op-capability-survey.md` -- broader NPU1 mailbox
  capability picture. AIE_RW_ACCESS (0x203) is the only non-trivial
  read/write path that's implemented; this finding shows even it has
  a hard hole.
- `xdna-emu/tools/validate-readback/validate-readback.cpp` -- redesigned
  to skip the dangerous DM round-trip by default; `--probe-dm-danger`
  reproduces.
