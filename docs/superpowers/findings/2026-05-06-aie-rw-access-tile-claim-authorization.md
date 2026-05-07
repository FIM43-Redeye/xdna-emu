---
name: AIE_RW_ACCESS enforces per-tile-claim authorization on Phoenix
description: Reads/writes via xrt::hw_context::read_aie_reg/write_aie_reg are scoped to tiles the hwctx's xclbin explicitly claims, NOT to the partition column. Discovered the hard way -- a sweep of unclaimed rows wedged the SMU.
type: finding
---

# AIE_RW_ACCESS tile-claim authorization

## TL;DR

`xrt::hw_context::read_aie_reg` / `write_aie_reg` (which dispatch
`MSG_OP_AIE_RW_ACCESS` / opcode `0x203` to firmware) are authorized
**per-tile**, scoped to the set of tiles the hwctx's xclbin explicitly
claims. Reads or writes targeting a tile in the partition column but
NOT in the kernel's tile set return `AIE2_STATUS_INVALID_PARAM` (4-byte
short response). The driver translates the non-success status to
`-EINVAL` (see `aie2_message.c:68-72`).

This is meaningful for step-debugger and emulator-readback design: you
cannot use AIE_RW_ACCESS to inspect arbitrary tiles in the array
without first standing up an hwctx whose kernel claims those tiles.

## How it manifested

Working on validate-readback's V2 cross-row sweep (rows 2-5 of the
partition column) on top of the dummy `add_one_using_dma` xclbin (which
claims only tiles `(0,0)`, `(0,1)`, `(0,2)` -- shim, memtile, compute):

- Reads at `(0, 2)` (claimed) succeeded -- this matches V0/V1.
- Reads at `(0, 3)`, `(0, 4)`, `(0, 5)` (unclaimed but in partition)
  returned `EINVAL`.
- Even the read at `(0, 2)` in the sweep failed `EINVAL`, contradicting
  the V0/V1 success at the same cell moments earlier. The most likely
  reason is mailbox channel state corruption from the accumulating
  unauthorized requests.

After enough INVALID_PARAM responses (somewhere in the V2/V3/V4
sequence), `mgmt_chann` timed out and was destroyed by the driver
(`aie2_message.c:62-66`). Subsequent operations returned `-ENODEV`,
and `modprobe -r` then failed at `aie2_mgmt_fw_fini: suspend_fw failed`,
SMU-wedging the device until reboot. (See
`2026-05-06-npu1-msg-op-capability-survey.md` for the SMU-wedge pattern
and recovery escalation.)

## Why "claimed by the hwctx" not "in the partition column"

The kernel's `aie.mlir` (or equivalent) lists which `aie.tile(c, r)`
the kernel needs. The xclbin's CDO encodes this; the firmware's
`MSG_OP_CREATE_CONTEXT` payload conveys it. AIE_RW_ACCESS authorization
is checked on the firmware side against THIS set, not the partition's
column range.

Verifying this via the actual code path:

- Driver-side allows any (col, row) within `hwctx->num_col` and
  `ndev->metadata.rows` (`aie2_pci.c:1635-1646`). Per-tile filtering
  is NOT in the driver.
- Firmware's `MSG_OP_AIE_RW_ACCESS` handler (closed source) returns
  `AIE2_STATUS_INVALID_PARAM = 0x02000004` for unauthorized tiles. The
  driver translates non-success status to `-EINVAL` via
  `aie2_send_mgmt_msg_wait` line 68-72.

## Implications for step-debugger and emulator readback

To inspect a tile via AIE_RW_ACCESS at runtime:

1. The hwctx's xclbin must claim that tile (via `aie.tile` declaration
   or equivalent).
2. A "scratch buffer" tile that the kernel doesn't actively use is
   acceptable -- it just needs to be in the claim list.
3. For a future "step the NPU one cycle at a time" design that wants
   to read arbitrary tile state: build a debug xclbin that claims the
   FULL array (5 cols × 6 rows on Phoenix). That single xclbin can
   then back any single-step inspection session.

## Implications for validate-readback test design

- Only sweep tiles the kernel's xclbin actually claims. For
  `add_one_using_dma`, that's `(col, 0)`, `(col, 1)`, `(col, 2)`.
- "Authorization probe" reads (intentional unauthorized cell) are
  useful but should be done LAST and exactly ONCE -- accumulating
  INVALID_PARAM responses risks mailbox channel timeout.
- Functional sweeps that need wider tile coverage need a different
  xclbin (a "verify_4x4"-equivalent that claims all 4 cols × 6 rows).

## What it does NOT prove

- Whether the firmware enforcement is fully correct (e.g., does it
  reject reads to a "claimed" tile that's *not currently powered*?
  We'd need a kernel that selectively powers tiles to exercise this).
- Whether AIE_RW_ACCESS works against compute tiles that are claimed
  but in an unhealthy state (halted, error, etc.).
- Whether tile-type matters beyond claim-status. (V4 will exercise
  memtile to confirm, post-reboot.)

## See also

- `xdna-driver` `aie2_pci.c:1564` -- AIE_RW_ACCESS read IOCTL handler.
- `xdna-driver` `aie2_message.c:68-72` -- driver-side translation of
  firmware non-success status to `-EINVAL`.
- `xdna-emu/tools/validate-readback/validate-readback.cpp` -- V3
  unclaimed-cell probe will reproduce this finding cleanly post-reboot.
- `2026-05-06-npu1-msg-op-capability-survey.md` -- SMU-wedge pattern
  and recovery escalation that this discovery hit.
