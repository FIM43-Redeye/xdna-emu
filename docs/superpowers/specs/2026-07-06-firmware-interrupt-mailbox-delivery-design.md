# Firmware interrupt + mailbox event delivery (M2c iter18 / the (C) endgame)

**Date:** 2026-07-06
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Status:** Design approved (brainstorming), pending spec review -> implementation plan.

## Problem

The M2c walk-and-stub boot now runs the real firmware into its **cooperative
task scheduler / main dispatch loop** and then spins in unbounded recursion
(stack descends ~144 B/cycle). This is NOT a decode/mapping wall and NOT an
emulator logic bug (confirmed below) -- it is the firmware waiting for
asynchronous events the emulator never delivers.

### Root cause (iter18, Phase-1 root-cause complete)

- The dispatcher `0xd7f0` picks the current task `sub = struct2[0x28]`
  (`struct2 = 0x2250`; the current task swaps once `0x10f10 -> 0x9040` then
  sticks), sets its state byte `sub[0x2c] = 6`, runs a worker, then branches on
  the per-task **done flag** `sub[0x30]`: nonzero -> **return/unwind**; zero ->
  **recurse** (`callx8 0x588c` -> `0x8770` -> `0xc530` -> `0x7fc4` -> `0xd7f0`).
- Store-watch proof: `sub[0x30]` (`0x9070` / `0x10f40`) is written **zero times**
  by any instruction in the entire boot (while `sub[0x2c]` is written every
  pass). Since the task structs are low-window, any store would route through
  `local_data` and be visible. No instruction sets the done flag.
- No external reads occur in the loop -> it is not polling.

### Why this is faithful, not a bug (the (B) confirmation)

The firmware's main idle loop is at **vaddr `0xc8cc`**:

```
loop:  a4 = global[0x84]            ; handler pointer
       if (a4 != 0) callx8 a4       ; DISPATCH the scheduler
0xc8eb waiti 0                      ; SLEEP until interrupt
0xc8ee j loop
```

That is the textbook interrupt-driven `dispatch(); waiti();` structure. The
scheduler is *supposed* to process ready work, return, and let `waiti` sleep
the CPU until the next interrupt (mailbox doorbell / DMA completion) marks a
task done (`sub[0x30]`) and wakes it. Our dispatcher recurses instead of
returning purely because **no interrupt ever delivers a task-completion event**.
The recursion is downstream of missing events. The fix is event delivery.

## Goal

Model **faithful Xtensa interrupt delivery** and a **mailbox doorbell** as a
real interrupt source, so the firmware's own handler runs, completes tasks, the
scheduler unwinds to the `WAITI` idle loop, and a host-injected mailbox command
is processed end-to-end with a response. Dispatch timings then *emerge* from
real firmware execution -- the firmware-emulation endgame.

## Design decisions (from brainstorming)

- **Target:** the full **host-command round-trip** (host writes a mailbox
  command -> doorbell IRQ -> firmware dispatches -> response), not just "reach
  idle."
- **Faithfulness:** faithful interrupt **mechanism** (derived from QEMU /
  Xtensa ISA) as the floor; the mailbox->interrupt-bit **wiring** derived
  **empirically** by observing the firmware's own `INTENABLE`/vector/doorbell
  setup, enriched with any driver/datasheet detail we can get, **but never
  gated on** a full interrupt-controller datasheet. Synthetic completion
  (poking flags directly) is rejected -- it would prevent timings from emerging.

## What already exists (build on, don't rebuild)

- `PS.INTLEVEL` field + `intlevel()`/`set_intlevel()` (`regfile.rs`).
- `rsil` implemented (`interp/system.rs`): captures full old PS, sets INTLEVEL.
- `Step::Wait(WaitReason)` with `Waiti` / `MailboxEmpty` variants
  (`interp/mod.rs`) -- the wait-return type is already defined.
- Exception vector machinery (`VECBASE`, `raise_general_exception`, EXCCAUSE
  incl. `LEVEL1_INTERRUPT`) and `Op::Waiti` decode.
- Mailbox/command protocol facts from prior recon (driver `amdxdna_mailbox.*`):
  X2I (host->fw) / I2X (fw->host) rings, head/tail regs in the `0x27010dxx`
  block, 16-byte wire header `{total_size, sz_ver, id, opcode}`, opcode set
  (`EXEC_DPU 0x10`, `CALIBRATE_CLOCK 0x11C`, ...), id magic `0x1D000000`.

## Components (six independently testable units)

1. **Interrupt registers** (`regfile.rs`): `INTENABLE` (SR 0xE4),
   `INTERRUPT` pending (SR 0xE2 read), `INTSET` (SR 0xE2 write-to-set),
   `INTCLEAR` (SR 0xE3 write-to-clear). Per-bit pending & enable. No
   `CCOUNT`/`CCOMPARE` (firmware uses zero timer interrupts). Wire these into
   `write_sr`/`read_sr` (currently log-and-drop unmodeled SRs).

2. **Delivery** (`interp/mod.rs`): after each step (and when re-checking a
   blocked `WAITI`), if `(INTERRUPT & INTENABLE)` has a bit at a level greater
   than `PS.INTLEVEL` with `PS.EXCM == 0`, take the interrupt: `EPC1 <- PC`,
   set PS (EXCM / raise INTLEVEL per the level), vector to
   `VECBASE + Level1InterruptVectorOffset`. Derived from QEMU
   `HELPER(check_interrupts)` / the Xtensa ISA. This is what makes the
   firmware's own handler run. Offset(s) discovered in Phase 0.

3. **`WAITI` blocks** (`interp/control.rs`): set `PS.INTLEVEL = imm`; if no
   deliverable interrupt is pending, return `Step::Wait(Waiti)`. `boot_to_idle`
   treats WAITI-with-nothing-pending as `reached_idle`. A newly-pending
   interrupt makes the next check deliver instead of wait.

4. **Mailbox/doorbell source** (`mmio.rs`): model the `0x27xxxxxx` block --
   X2I/I2X rings with head/tail pointers, the 16-byte wire header. A host write
   to the doorbell register sets the mailbox `INTERRUPT` bit (bit # from
   Phase 0).

5. **Host-injection API** (`FirmwareProcessor`):
   `inject_mailbox_command(opcode, payload)` -- writes the wire message into
   X2I, bumps the tail, rings the doorbell (sets the pending bit);
   `read_mailbox_response()` -- reads I2X.

6. **Verification**: boot to idle (WAITI), inject a simple handshake/version
   command, assert the firmware's real handler runs, processes it, and writes a
   response to I2X.

**End-to-end data flow:** `inject -> doorbell write -> INTERRUPT bit set ->
(WAITI wakes / next step delivers) -> firmware Level-1 handler -> reads X2I ring
-> dispatches opcode -> sets task done -> scheduler unwinds -> writes I2X
response -> returns -> WAITI idle`.

## Phasing

**Phase 0 -- Observe the wiring** (RE, bounded). Pin: which `INTENABLE` bits
the firmware sets during init; the Level-1 interrupt vector/handler location
(`VECBASE + offset`); the mailbox doorbell register + which `INTERRUPT` bit it
drives; the X2I/I2X ring addresses; **and what event the init-time stuck task
awaits** (self-generated init event vs. first host command). Output: a findings
doc that makes Phases 1-4 concrete. Diagnostics available: `XDNA_FW_CALLS`,
`XDNA_FW_STOP_PC`, the trace-to-wall probe; add SR-write logging for
`INTENABLE`/vector setup.

**Phase 1 -- Interrupt mechanism** (deterministic, derive-from-toolchain).
Registers + delivery + `WAITI`-blocks. Unit-tested in isolation against QEMU
semantics (synthetic pending bit -> vector to a test handler; WAITI wakes). No
firmware binary needed to test.

**Phase 2 -- Unwind to idle.** With the mechanism live, deliver the event
Phase 0 identified. The stuck task completes, the recursion unwinds, boot
reaches the `0xc8cc` `WAITI` idle loop -> `reached_idle = true`. First
end-to-end proof on real firmware.

**Phase 3 -- Mailbox protocol + injection.** Model the X2I/I2X rings + doorbell
in `mmio.rs`; add `inject_mailbox_command` / `read_mailbox_response`.

**Phase 4 -- Round-trip.** Boot to idle, inject a simple handshake/version
command, verify the firmware's own handler processes it and writes a response.
Dispatch timing emerges from real execution.

## Risks / open questions (resolved by Phase 0)

- **Phase 0 gates the specifics** of Phases 2 and 4: until we observe the real
  wiring we are partly designing against the known driver protocol.
- **Phase 2's event may be the mailbox itself.** If the firmware will not idle
  until it has serviced an initial host handshake, Phases 2 and 3/4 merge.
  Phase 0 resolves whether the init-time stuck task awaits a self-generated
  event or the first host command.
- The exact Level-1 interrupt vector offset for this custom AMD core is
  unknown until Phase 0 (iter7 already found the core's exception offsets differ
  from the standard Tensilica configs -- KERNEL=0x2e0, DOUBLE=0x31c -- so the
  interrupt vector offset must be observed, not assumed).

## Testing

- Phases 1: hermetic unit tests (no firmware) for register semantics, delivery
  gating (INTLEVEL/EXCM masking, level priority), and WAITI wake/block.
- Phases 2/4: firmware-gated integration tests (skip without the binary, like
  the existing boot tests) asserting `reached_idle` and a correct I2X response.
- Every phase keeps `cargo test --lib` green.

## Non-goals (this iteration)

- Timer/CCOUNT interrupts (firmware uses none).
- Multiple simultaneous interrupt levels beyond what the firmware exercises.
- The full opcode set -- Phase 4 proves the round-trip with one simple command;
  additional opcodes are follow-on work.
