# NPU firmware suspend / waitmode path (Phoenix LX7, FW 1.5.5.391)

**Status:** Suspend/halt path reverse-engineered (2026-05-20). The firmware's
main event loop, its event dispatch, and the quiesce-then-`waiti` routine are
mapped and connected to the driver's `MSG_OP_SUSPEND` handshake. One hop --
host opcode to internal event code -- is still open.

**Related:**
- [`2026-05-20-npu-firmware-format.md`](2026-05-20-npu-firmware-format.md)
  -- format + architecture (Xtensa LX7, load base, the Ghidra pipeline).
- [`2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`](2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md)
  -- the recovery investigation that motivated reading FW internals.

## Why this matters

When the NPU wedges, driver reload fails because `aie2_smu_start()` runs a
defensive SMU `POWER_OFF` that returns `0xFF` (CMD_FAIL). The SMU is healthy;
it cannot *complete* `POWER_OFF` because the NPU it is trying to power-gate
will not quiesce. "Quiesce" has a precise meaning, and this document pins it
down: the LX7 management firmware must reach a `waiti` (Xtensa wait-for-
interrupt halt). The suspend path is the routine that gets it there.

The SMU itself runs separate firmware and is not in the LX7 blob -- see the
format doc. This document is purely about the **LX7 management firmware**.

## The SUSPEND handshake -- driver side (VERIFIED from source)

`aie2_suspend_fw()` (`xdna-driver/src/driver/amdxdna/aie2_message.c:114`)
does two things in order:

| Step | Action |
|------|--------|
| 1 | Send `MSG_OP_SUSPEND` (0x101) over the management mailbox via `aie2_send_mgmt_msg_wait()`. Payload is empty -- `suspend_req` is a single `place_holder` u32. |
| 2 | `aie2_psp_waitmode_poll(ndev->psp_hdl)` -- poll `PSP_PWAITMODE_REG` until `(mode_reg & 0x1) == 1`. |

`PSP_PWAITMODE_REG` maps to `MPNPU_PWAITMODE = 0x3010034`
(`npu1_regs.c:11`), in the PUB register aperture. A SUSPEND is not "done"
when the FW acks the mailbox message -- it is done when `PWAITMODE` bit 0
goes high. That bit reflects the LX7 core being in `waiti` state.

## The firmware side (REVERSE-ENGINEERED)

All addresses are rebased to load base `0x08ad3000`. The FW sees the PUB
aperture at `0x27010000` (the driver sees the same registers through its
PCIe BAR at `0x03010000`).

### Main event loop -- `FUN_08ad85f8`

The firmware's main supervisor loop. Pseudocode:

```
init (8 setup calls)
loop:
    if [0x27010AC0] != 0:      # PUB control/status reg: nonzero => idle-ok
        waiti 0x0              # halt until interrupt
    event = FUN_08b04428(-1, 1)   # dequeue next event code
    dispatch on event:
        0xe  -> FUN_08ad8190      quiesce all 6 columns + waiti   <-- HALT
        0xf  -> FUN_08ad8480      mailbox transport processor
        0x11 -> FUN_08ad81e0      per-column status aggregation
        0x12 -> FUN_08ad8b08
        (codes <= 0xd handled in separate blocks)
```

`FUN_08ad85f8` has no callers -- it is a top-level entry point (the main
thread). It was invisible until the `SeedFunctions` pipeline fix (below).

### The quiesce-and-halt routine -- `FUN_08ad8190` (event 0xe)

```
loop 6x   (the 6 columns; per-column state at 0x0E740 + col*0x1b8):
    if column active (bit 3 of per-column state byte):
        FUN_08ad70b8(column)        # per-column teardown
write -1 to a global state slot
FUN_08adc858()
OR a bit into PUB reg 0x27010AC0
log-checkpoint(0x11)
waiti 0x0                           # <-- HALT; reflected in PWAITMODE
retw.n
```

This is the firmware action that satisfies the driver's `waitmode` poll.

### Per-column teardown -- `FUN_08ad70b8`

Takes a column index, disables that column's resources, then issues
`idtlb` + `dsync` (data-TLB invalidate and sync -- the LX7 has an MMU).
Also called from a per-context destroy path, not only from suspend.

### `waiti` and `PWAITMODE` (CLAIMED -- mechanism inferred)

`waiti` appears at four sites; `FUN_08ad85f8` (idle) and `FUN_08ad8190`
(suspend) are the relevant two. The firmware does **not** write
`0x27010034` -- consistent with `PWAITMODE` being a hardware-latched
status of the core's wait-mode output pin rather than a software-written
register. The driver polling `PWAITMODE` bit 0 is therefore polling "is
the LX7 core halted" directly. The pin-latch mechanism is inferred, not
yet verified against hardware docs.

## How the wedge happens

The chain, end to end:

> LX7 wedged (stuck in a transaction, faulted, spinning) -> `FUN_08ad8190`
> never runs to completion -> `waiti` never executes -> `PWAITMODE` bit 0
> never asserts -> `aie2_psp_waitmode_poll` times out -> SUSPEND fails ->
> the NPU never reaches a quiesced state -> the SMU `POWER_OFF` handshake
> has no quiesced partner to acknowledge it -> `0xFF` CMD_FAIL.

This makes the earlier "POWER_OFF needs FW cooperation" hypothesis
concrete: the cooperation *is* the LX7 reaching its `waiti`.

## Tooling: `SeedFunctions` (recovering indirectly-reached code)

Ghidra's auto-analysis is call-graph driven -- it disassembles from known
entry points and follows calls. This firmware reaches a large amount of
code only *indirectly* (interrupt/exception vectors, dispatch-table
handlers, `callx` targets), so that code was never disassembled and its
inbound calls never registered. `FUN_08ad8190` showed "0 callers" purely
because its caller (`FUN_08ad85f8`) lived in an un-analyzed gap.

`tools/ghidra-scripts/SeedFunctions.java` scans the whole image for the
Xtensa windowed-ABI `entry` prologue and creates a function at every one,
then re-runs analysis over the recovered code. It is wired into
`tools/ghidra-npu-fw.sh` as a postScript before the dump. Effect on this
firmware: 590 -> 623 functions, and the entire suspend chain became
visible. Any future RE question needs the complete code, not the
call-reachable slice.

## Open question: opcode -> event translation

The host sends mailbox opcode `MSG_OP_SUSPEND = 0x101`; the FW main loop
dispatches on internal **event 0xe**. The translation between them is not
yet pinned. It is *not* an opcode comparison chain and *not* a simple
`{opcode, handler}` table -- both were scanned for and neither exists.
The event codes being distinct for suspend (0xe) vs. generic message
(0xf) suggests the mailbox ISR peeks the incoming opcode and routes
control-plane messages (SUSPEND/RESUME) to dedicated events. Confirming
that is the next step. It is a detail, not load-bearing for the wedge
conclusion above.

## Address reference

| Symbol | Address | Role |
|--------|---------|------|
| `FUN_08ad85f8` | `0x08ad85f8` | Main FW event loop |
| `FUN_08ad8190` | `0x08ad8190` | Quiesce 6 columns + `waiti` (event 0xe) |
| `FUN_08ad70b8` | `0x08ad70b8` | Per-column teardown |
| `FUN_08ad8480` | `0x08ad8480` | Mailbox transport processor (event 0xf) |
| `FUN_08ad81e0` | `0x08ad81e0` | Per-column status aggregation (event 0x11) |
| `FUN_08b04428` | `0x08b04428` | Event-queue dequeue primitive |
| PWAITMODE | FW `0x27010034` / drv `0x3010034` | Core wait-mode status |

## Source policy

Static analysis of AMD's NPU firmware as a reading reference, per the
xdna-emu source policy (top-level CLAUDE.md). No firmware code is copied
into the emulator. The knowledge informs original emulator implementations
and the driver-side recovery fix.
