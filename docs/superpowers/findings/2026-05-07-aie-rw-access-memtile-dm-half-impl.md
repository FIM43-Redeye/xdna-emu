---
name: AIE_RW_ACCESS reads against the memtile (row 1) hang the firmware on Phoenix
description: Reads via xrt::hw_context::read_aie_reg targeting any memtile address tested so far (DM 0x10000 and register-space TIMER_LOW 0x940F8) hang firmware indefinitely. Writes silently ack with SUCCESS but cannot be verified. The user-context mailbox channel times out after 5s and is destroyed by the driver. Modprobe -r afterward cascades to SMU wedge, requiring reboot.
type: finding
---

# AIE_RW_ACCESS memtile half-implementation

## TL;DR

`xrt::hw_context::read_aie_reg` against ANY memtile (row 1) address
tested so far — including both data memory (offset 0x10000) and
register space (`MEM_TILE_MODULE_TIMER_LOW` at 0x940F8) — never
receives a response from firmware. The user-context mailbox channel
times out after 5s and is destroyed by the driver.

Writes to memtile DM ack with SUCCESS at the mailbox layer (driver logs
the write, response 0x203 size 8 with success status), but since
readbacks hang we cannot verify the writes actually committed anywhere.
Memtile-register writes were not tested; assume the same write-acks/
read-hangs pattern.

After the user-context mailbox dies, the device enters a degraded state:
new xclbins can still be loaded and contexts created, but kernel jobs
fail with status 255 (`EXEC_CMD -EIO`). Running `modprobe -r amdxdna`
from this state cascades to SMU wedge (`smu cmd 4 failed, 0xff`),
requiring reboot.

This is the **worst possible failure mode** for a debug/readback API:
writes claim to succeed without performing any verification, and
reads — the only way to verify writes — silently hang. We have no way
to confirm whether the writes went anywhere real.

## How it manifested

Two independent reproductions:

### Reproduction 1: memtile DM round-trip (write + write + read)

`tools/validate-readback/validate-readback.cpp` V4 test did a
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

### Reproduction 2: memtile register-space single read (no preceding writes)

After updating V4 to read `MEM_TILE_MODULE_TIMER_LOW` (0x940F8) -- a
read-only register, no preceding writes from this test, just five
successful V2 reads on a compute tile immediately prior:

```
[79.674796] AIE tile read: ctx 1 col 0 row 2 addr 0x31520 size 4    <-- compute reg V2 read
[79.674806] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001a
[79.674889] xdna_mailbox.145: resp opcode 0x203 size 8 id 0x1d00001a  <-- ACK ~80us
[79.674924] AIE tile read: ctx 1 col 0 row 2 addr 0x31520 size 4
[79.674935] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001b
[79.674993] xdna_mailbox.145: resp opcode 0x203 size 8 id 0x1d00001b  <-- ACK ~60us

[79.675034] AIE tile read: ctx 1 col 0 row 1 addr 0x940f8 size 4    <-- memtile reg
[79.675045] xdna_mailbox.145: req opcode 0x203 size 24 id 0x1d00001c <-- never responded
[84.727322] xdna_mailbox.145: IRQ disabled and RX work cancelled     <-- 5s later
[84.727442] aie2_rw_aie_reg: AIE reg read failed, ret -62           <-- ETIMEDOUT
```

The mailbox handled the V2 compute reads with ~60-80 µs round-trip
each. The memtile read on the SAME mailbox channel never got a reply.
This rules out any "mailbox saturation / hammering too fast" theory:
the firmware was responsive throughout V2, then specifically dropped
the memtile read.

### Common pattern

Both writes and reads to memtile DM at offsets 0x10000 / 0x10100 ack
with SUCCESS for writes; both reads (DM and register space) hang.
modprobe-r afterward cascades through `aie2_suspend_fw -19` and
`smu cmd 4 failed, 0xff`, requiring reboot.

## What address space triggers it

The entire memtile (row 1) appears unreachable for AIE_RW_ACCESS reads.
Two distinct address regions confirmed broken:

- **Memtile DM** at offset `0x10000` (64 KB into the 512 KB
  `XAIEMLGBL_MEM_TILE_MODULE_DATAMEMORY` region, which spans
  `0x00000-0x7FFFF`).
- **Memtile registers** at offset `0x940F8`
  (`XAIEMLGBL_MEM_TILE_MODULE_TIMER_LOW`, well into reg space).

These cover both halves of the memtile address map, so absent further
testing assume **all memtile reads** via AIE_RW_ACCESS hang. (Each
attempt costs a reboot, so we did not exhaustively probe.)

Compute-tile **register** access works fine (V0/V1/V2 confirmed
`PERF_COUNTER0` and `TIMER_LOW` round-trips at `(col, 2)`). Compute-tile
**DM** access is currently TBD; the redesigned `validate-readback` test
reads compute DM at offset `0x100` (V6) to characterize. Will run
post-reboot.

Shim-tile (row 0) access via AIE_RW_ACCESS not exercised yet; it could
be a third "broken" tile type or a fourth working one. Note that
shim-tile reads weren't in the previous tests because shim has minimal
register space and no DM in the conventional sense.

## Address-space hypothesis ruled out

Initial concern: maybe we're using the wrong address. Specifically,
maybe firmware expects a different mapping for memtile (e.g., the
addresses in `xaiemlgbl_params.h` are nominally for Versal AIE-ML and
Phoenix XDNA's firmware uses a different layout).

The `add_one_using_dma` xclbin's CDO programs memtile (col 0, row 1)
DMA BD0 at offset `0xA0000`, which matches `xaiemlgbl_params.h`'s
`XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_0 = 0x000A0000` exactly. Other
memtile DMA offsets in the same CDO (`0xA0040`, `0xA0060`, `0xA0300`,
`0xA0320`, `0xA0340`, `0xA0360`) all align with the published register
spacing. The kernel runs successfully, so firmware is honoring those
addresses correctly during CDO load.

So firmware DOES interpret `(col, row, addr)` with `addr` as the
tile-module-relative offset, the same convention we used for memtile
TIMER_LOW (`0x940F8`). The address space is right; the AIE_RW_ACCESS
read code path is the broken thing -- specifically, the variant that
services memtile reads.

## Why CDO writes work but AIE_RW_ACCESS reads don't

Two different firmware code paths:

- **CDO load** (xclbin programming): firmware streams a sequence of
  WRITE/MASK_WRITE commands from the xclbin's CDO blob, addresses
  preconfigured in the binary. Memtile DMA, lock, and stream-switch
  registers all get written this way successfully -- the kernel runs.
- **AIE_RW_ACCESS** (runtime mailbox opcode 0x203): firmware receives
  a single (col, row, addr, [value]) tuple via mailbox, performs the
  access, and replies. Compute-tile reads/writes work; memtile reads
  hang (and probably memtile writes "succeed" without committing).

The CDO path is exercised every time the kernel loads -- it's
load-bearing for production. The AIE_RW_ACCESS path is exercised only
when something explicitly calls `read_aie_reg`/`write_aie_reg`, which
is rare in production. A bug in AIE_RW_ACCESS for memtile would not
have been noticed by AMD's own test suite.

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

## Recovery cost and the mailbox-channel cascade

Each memtile read attempt costs:
- 5s of wall time (mailbox timeout)
- The user-context mailbox channel (destroyed at timeout)
- THE MGMT CHANNEL TOO (the test-process exit cleanup hits ENODEV on
  a follow-up mgmt op via the dead user channel and tears down
  mgmt_chann as well)
- The device enters a "no live mailbox" state: subsequent `xrt-smi
  validate` / new hwctx creation fails with `part_ctx_start: connect
  failed, err -19`, EXEC_CMD returns -EIO
- Eventual SMU wedge during the next modprobe-r (mgmt fini cascade
  is the trigger but the damage was already done at process exit)
- A reboot to recover

Confirmed timeline from one such failure (timestamps in seconds since
boot):
```
79.675034  AIE tile read row 1 col 0 addr 0x940f8 (req)
84.727322  mailbox.145 IRQ disabled (5s timeout)
84.727431  mailbox.145 released
84.735345  mailbox.136 (MGMT) IRQ disabled
84.735393  mailbox.136 released
... 162s gap during which xrt-smi validate / our retry tried to load ...
246.669915 part_ctx_start: ctx.X.1 connect failed, err -19
247.761074 ctx.X.1 deleted, status 255 (EXEC_CMD -EIO at userland)
```

This is significantly worse than the unclaimed-cell finding
(2026-05-06-aie-rw-access-tile-claim-authorization.md): unclaimed cells
fail-fast and only wedge the SMU through accumulated EINVALs. A
memtile read wedges the entire device in a single shot, and the wedge
happens at TEST EXIT time (cleanup), not at modprobe -r time.

**Refinement of "modprobe -r causes the wedge"**: the wedge doesn't
require modprobe. The test process's own exit cleanup path tears down
both mailbox channels. Modprobe -r afterward fails the SMU stop only
because there's no mgmt channel left to talk through. The actual root
cause is the firmware-side hang on memtile read; everything else is
the driver's defensive cleanup propagating the damage.

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
