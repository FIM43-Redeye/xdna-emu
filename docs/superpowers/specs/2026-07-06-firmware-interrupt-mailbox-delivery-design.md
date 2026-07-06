# Firmware interrupt + mailbox event delivery (M2c iter18 / the (C) endgame)

**Date:** 2026-07-06
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Status:** Design approved (brainstorming), pending spec review -> implementation plan.

## Problem

The M2c walk-and-stub boot now runs the real firmware into its **cooperative
task scheduler / main dispatch loop** and then spins in unbounded recursion
(stack descends ~144 B/cycle). This is NOT a decode/mapping wall and -- strongly
indicated, see the (B) analysis below -- NOT an emulator logic bug either: it is
the firmware waiting for asynchronous events the emulator never delivers.

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

### Why this is faithful, not a bug (the (B) indication)

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

**How strong is this claim?** Strongly indicated, not yet fully closed. The
store-watch proof shows `sub[0x30]` is written zero times *by any CPU store
routed through `local_data`* -- but that is the only writer class it observes.
On real silicon a task-completion flag is frequently written by a **DMA
descriptor** the firmware programmed, with the interrupt merely *signaling* the
completion rather than the handler's own `s32i` setting the flag. So two
faithful shapes remain open, and **Phase 0 must decide which**:

- **(i) handler writes the flag** -- the delivered Level-1 handler runs, reads
  the ring, and stores `sub[0x30]` itself. Modeling interrupt delivery alone
  closes the loop.
- **(ii) DMA writes the flag** -- the interrupt only signals; the actual store
  comes from a DMA engine the emulator does not yet model. Here Phase 2 grows a
  minimal DMA-completion write, not just interrupt delivery.

Until Phase 0 identifies the completion *writer*, the root cause is "firmware
waiting on an undelivered async event" -- which is faithful either way -- but
the specific Phase 2 fix branches on (i) vs (ii).

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

Verified against the tree (not taken on faith):

- `PS.INTLEVEL` field + `intlevel()`/`set_intlevel()` (`regfile.rs`) -- real,
  round-trip tested, `set_intlevel` provably leaves CALLINC undisturbed.
- `rsil` implemented (`interp/system.rs`): captures the **full** old PS (tested
  with PS seeded outside the INTLEVEL nibble), sets INTLEVEL.
- `Step::Wait(WaitReason)` with `Waiti` / `MailboxEmpty` variants
  (`interp/mod.rs`) -- the wait-return type is already defined.
- `Op::Waiti` decodes correctly against both oracles.
- `VECBASE` + `raise_general_exception` machinery exists (the iter13 syscall
  path routes through it).

**Must be built, NOT reuse (the spec's earlier draft overclaimed these):**

- `EXCCAUSE = LEVEL1_INTERRUPT (4)` -- currently only a doc comment enumerating
  QEMU's cause enum; no `const`, nothing uses it. Add the const + set it on
  interrupt entry.
- **`rfe` (return from exception/level-1 interrupt)** -- not decoded, not
  implemented. Today only `rfwo`/`rfwu` exist; `decode/control.rs` explicitly
  punts the `rfe`/`rfde` encoding family as "boot path doesn't hit." Interrupts
  are the **first** code path in the project that requires a handler to
  *return* architecturally. See Component 2b.

- Mailbox/command protocol facts from prior recon (driver `amdxdna_mailbox.*`):
  X2I (host->fw) / I2X (fw->host) rings, head/tail regs in the `0x27010dxx`
  block, 16-byte wire header `{total_size, sz_ver, id, opcode}`, opcode set
  (`EXEC_DPU 0x10`, `CALIBRATE_CLOCK 0x11C`, ...), id magic `0x1D000000`.

## Components (independently testable units)

**Scope decision -- Level-1 only.** We model a single interrupt priority:
**level-1** (the low-priority, EXCM-masked class), which is what a mailbox
doorbell on a management core is in practice. High-priority interrupts (L>=2,
with `EPS[n]`/`EPC[n]` banks and `rfi n` return) are explicitly out of scope
until the firmware is observed to use one -- Phase 0 confirms level-1. This
keeps delivery to a single faithful mechanism instead of a level-1/high-level
blend, and means the per-interrupt priority *level map* (which is fixed in the
core's TIE config and NOT firmware-observable) collapses to "everything modeled
is level-1."

1. **Interrupt registers** (`regfile.rs`): `INTENABLE` (SR 0xE4),
   `INTERRUPT` pending (SR 0xE2 read), `INTSET` (SR 0xE2 write-to-set),
   `INTCLEAR` (SR 0xE3 write-to-clear). Per-bit pending & enable. No
   `CCOUNT`/`CCOMPARE` (firmware uses zero timer interrupts). Wire these into
   `write_sr`/`read_sr` (currently log-and-drop unmodeled SRs). Add a
   `LEVEL1_INTERRUPT = 4` cause const. (No `PS_UM` -- see Component 2.)

2. **Delivery** (`interp/mod.rs`): after each step (and when re-checking a
   blocked `WAITI`), a level-1 interrupt is deliverable iff
   `(INTERRUPT & INTENABLE) != 0` **and** `PS.INTLEVEL == 0` **and**
   `PS.EXCM == 0` (derived from the Xtensa ISA / QEMU `handle_interrupt`).
   When deliverable, take it by **reusing the existing, silicon-validated
   `raise_general_exception(PC, LEVEL1_INTERRUPT)`** path -- a level-1 interrupt
   shares the general user/kernel exception vector and dispatches on EXCCAUSE,
   so it is exactly a general exception with `cause = 4`. That path already
   does the faithful entry: `EPC1 <- PC`, `EXCCAUSE <- 4`, `PS.EXCM <- 1`,
   and routes to `GENERAL_EXCEPTION_HANDLER` (the absolute `0x2958` the iter13
   work proved is the real handler on this core -- **not** VECBASE-relative,
   which dead-ends at a zero dispatch pointer early in boot). PS.INTLEVEL is
   **not** raised for level-1 (EXCM does the masking), and **PS.UM is not
   touched** -- on real Xtensa the vector is selected by *reading* UM, not
   writing it, and this firmware is kernel-only (UM implicitly 0), so mirroring
   `raise_general_exception` (which sets EXCM only) is the faithful entry.

   The one thing delivery must confirm (Phase 0) that the syscall path did not
   need: that the shared handler at `0x2958`, **post-init**, has a working
   `EXCCAUSE == 4` (level-1 interrupt) dispatch arm. The syscall path exercises
   the `cause = 1` arm early in boot; the interrupt exercises the `cause = 4`
   arm after init has run. Phase 0 verifies the handler routes cause-4 to real
   interrupt-servicing rather than a stub.

2b. **Return from interrupt -- `rfe`** (`decode/*` + `interp/control.rs`):
   decode and implement `rfe` (encoding family currently punted). Semantics:
   `PS.EXCM <- 0; PC <- EPC1`. This is the terminating instruction of the
   level-1 handler; **without it the handler cannot return and Phase 2 walls
   immediately.** (`rfi n` for high-level interrupts stays unimplemented --
   out of scope per the level-1 decision.)

3. **`WAITI` retires, then blocks** (`interp/control.rs`): `waiti imm` sets
   `PS.INTLEVEL = imm` and **advances PC past the WAITI** (the instruction has
   retired -- the CPU is now idle *after* it), then, if no interrupt is
   deliverable, returns `Step::Wait(Waiti)`. The PC-advance is load-bearing:
   when the interrupt later arrives, `EPC1 <- PC` captures the instruction
   *after* WAITI (the `j loop` at `0xc8ee`), so `rfe` resumes the idle loop and
   re-dispatches -- rather than returning onto the WAITI and re-sleeping
   forever (a livelock that would masquerade as the current recursion). This
   changes today's "WAITI does not advance PC" behavior. **Change surface** (not
   just new tests): (a) the boot-idle harness currently detects idle as
   `Step::Wait && cpu.pc == pre_step_pc` (`mod.rs:227`, and sibling copies in
   the probe/test loops at ~796/~953) -- with WAITI now advancing PC that
   equality no longer holds, so idle detection must key on
   `Step::Wait(Waiti)` regardless of PC movement; (b) the two existing waiti
   unit tests assert `cpu.pc == 0` (`control.rs:267`, `:287`) and the
   "deliberately does NOT advance pc" doc comments must be updated. A
   newly-pending interrupt makes the next check deliver instead of wait.

4. **Mailbox/doorbell source** (`mmio.rs`): the `0x27xxxxxx` block is already
   plain RAM, so X2I/I2X message bytes already stick with zero new plumbing.
   Add only what the round-trip needs: (a) the **doorbell write side-effect** --
   a write to the doorbell register sets the mailbox `INTERRUPT` bit (bit # from
   Phase 0); (b) minimal head/tail pointer bumps. A full ring abstraction
   (wraparound, sizing, general header machinery) is deliberately *not* built
   here -- YAGNI until more than one opcode is exercised.

5. **Host-injection API** (`FirmwareProcessor`):
   `inject_mailbox_command(opcode, payload)` -- writes the wire message into
   X2I, bumps the tail, rings the doorbell (sets the pending bit);
   `read_mailbox_response()` -- reads I2X.

6. **Verification**: boot to idle (WAITI), inject a simple handshake/version
   command, assert the firmware's real handler runs, processes it, and writes a
   response to I2X.

**End-to-end data flow:** `inject -> doorbell write -> INTERRUPT bit set ->
(WAITI already retired -> next check delivers) -> Level-1 entry via
raise_general_exception(cause=4) (EPC1<-PC, EXCM=1, route to 0x2958) -> firmware
handler reads X2I ring -> dispatches opcode -> sets task done -> scheduler
unwinds -> writes I2X response -> handler `rfe` -> back onto j loop -> WAITI
idle`.

## Phasing

**Phase 0 -- Observe the wiring** (RE, bounded). Firmware-observable, pin by
watching the boot: which `INTENABLE` bits the firmware sets during init;
confirmation the doorbell is **level-1** (not high-priority) and that the idle
loop is `waiti 0` (INTLEVEL 0 -- a `waiti 1+` would mask level-1 and never
wake); that the shared exception handler at `0x2958` has a working
**`EXCCAUSE == 4` (level-1 interrupt) dispatch arm post-init** (the syscall
path only exercised the cause-1 arm early in boot -- confirm cause-4 reaches
real interrupt servicing, not a stub); the X2I/I2X ring addresses; **what event
the init-time stuck task awaits** (self-generated init event vs. first host
command); **and who writes the task done-flag `sub[0x30]`** -- a CPU store in
the handler (shape (i)) vs. a DMA/peripheral (shape (ii), which pulls a minimal
DMA-completion write into Phase 2). Output: a findings doc that makes Phases 1-4
concrete.

*Not firmware-observable -- assume a faithful default, mark as a calibration
knob, validate against HW rather than blocking Phase 0 on RE:*
- **Which `INTERRUPT` bit the doorbell drives** and its **edge-vs-level
  trigger**. Default: a single dedicated bit, **edge-triggered** (INTSET sets,
  handler acks via INTCLEAR). A level-triggered doorbell would instead require
  clearing the *source* register, and if acked via INTCLEAR alone would re-fire
  forever -- so if idle isn't reached, flip this knob.
- The **per-interrupt priority level** -- fixed in the core's TIE config, not a
  register the firmware writes. Collapsed to level-1 by the scope decision.

Diagnostics available: `XDNA_FW_CALLS`, `XDNA_FW_STOP_PC`, the trace-to-wall
probe; add SR-write logging for `INTENABLE`/vector setup and store-watch on
`sub[0x30]` to settle the writer question.

**Phase 1 -- Interrupt mechanism** (deterministic, derive-from-toolchain).
Registers + delivery + `rfe` + `WAITI`-retire. Unit-tested in isolation against
QEMU semantics with no firmware binary: synthetic pending bit -> level-1 entry
(EPC1<-PC, EXCCAUSE=4, EXCM=1, routed via raise_general_exception) -> `rfe` ->
EXCM cleared, PC restored to EPC1; WAITI advances PC then blocks, and a pending
bit wakes it to deliver.
Note: delivery (Component 2) and return (Component 2b) are a **coupled pair** --
they must land together; a handler that vectors but can't `rfe` is untestable.

**Phase 2 -- Unwind to idle.** With the mechanism live, deliver the event
Phase 0 identified. Branches on the Phase-0 completion-writer finding: shape (i)
-- the delivered handler stores `sub[0x30]` itself, so interrupt delivery alone
closes the loop; shape (ii) -- add a minimal DMA-completion write that sets
`sub[0x30]`, with the interrupt signaling it. Either way the stuck task
completes, the recursion unwinds, boot reaches the `0xc8cc` `WAITI` idle loop ->
`reached_idle = true`. First end-to-end proof on real firmware.

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
- **The completion writer may be DMA, not the CPU handler** (shape (ii)). The
  store-watch only observed CPU stores; if `sub[0x30]` is DMA-written on
  silicon, Phase 2 needs a minimal DMA-completion write. Phase 0 settles this
  before Phase 2 is designed in detail.
- **Doorbell trigger type (edge vs level) is assumed, not observed.** If idle
  isn't reached with the edge default, flip the calibration knob (Phase 0
  fallbacks) before suspecting the mechanism.
- **Interrupt entry reuses the proven `raise_general_exception` route to the
  absolute handler `0x2958`, not a VECBASE-relative interrupt vector.** iter13
  proved the VECBASE-relative kernel/user exception-vector stub reads a zero
  dispatch pointer early in boot; the direct route is the faithful one for this
  core. The open Phase-0 question is not the *vector offset* but whether that
  shared handler's `EXCCAUSE == 4` arm does real interrupt servicing post-init.

## Testing

- Phases 1: hermetic unit tests (no firmware) for register semantics, delivery
  gating (INTLEVEL==0 / EXCM==0 masking), the level-1 entry sequence
  (EPC1/EXCCAUSE=4/EXCM, routed to 0x2958), `rfe` return (EXCM cleared,
  PC<-EPC1), and WAITI-retire-then-wake/block; plus the idle-detection harness
  change (key on `Step::Wait(Waiti)`, not PC-stability).
- Phases 2/4: firmware-gated integration tests (skip without the binary, like
  the existing boot tests) asserting `reached_idle` and a correct I2X response.
- Every phase keeps `cargo test --lib` green.

## Non-goals (this iteration)

- Timer/CCOUNT interrupts (firmware uses none).
- **High-priority interrupts (L>=2)**: the `EPS[n]`/`EPC[n]` banks and `rfi n`
  return are not modeled. Level-1 only, until Phase 0 shows the firmware uses a
  higher level (it is not expected to for a mailbox doorbell).
- Nested/re-entrant level-1 interrupts: EXCM masks level-1 during the handler,
  so by construction the handler runs to `rfe` without re-entry. Not modeled.
  (Corollary: because the handler runs EXCM=1, `window_check` is suppressed
  inside it -- iter17 gates on `WOE && !EXCM` -- so the handler spills windows
  in software via `rotw`, exactly as the existing `0x2958` exception handler
  already does. Don't expect `window_check` to fire mid-handler.)
- Multiple simultaneous interrupt levels beyond what the firmware exercises.
- The full opcode set -- Phase 4 proves the round-trip with one simple command;
  additional opcodes are follow-on work.
