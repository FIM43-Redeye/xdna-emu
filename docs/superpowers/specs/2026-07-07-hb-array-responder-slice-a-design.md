# H-b Slice A — Minimal Array Responder (Falsifier Experiment)

**Issue:** #140 (firmware-emulation dream / boot-to-idle)
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Date:** 2026-07-07
**Status:** Design approved, pending adversarial review

## Context

The iter18 firmware RE (documented in
`docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`)
charted the firmware<->array boundary to completion and bottomed out in a
**paradox**: every concrete *local* completion mechanism is excluded by
experiment, yet boot demonstrably *can* advance.

- **Interrupt-driven: excluded.** INTENABLE=0x1 (mailbox doorbell only, no AIE
  line); the busy dispatch loop holds PS.INTLEVEL=2 with zero level-0 windows
  over 400k steps, so a level-1 IRQ can never be delivered while boot is stuck.
- **Synchronous poll: excluded.** Seeding the poll flags (`XDNA_FW_TRACE_SEEDPOLL`,
  bits=0xb) alias-correctly makes `FUN_00008c68` take its full ack+consume path
  6660x, but boot does not advance.
- **Tag/response: degenerate.** The poll's tag arg a7=0, so `[0xf9e8]==0` matches
  trivially -- there is no nonzero tag the array must echo.

Yet forcing the pending mask `[0x10f40]` directly cascades boot 58k -> 623097
(~565k steps). So something real sets that mask, and static charting cannot see
what.

**The reframe:** static RE cannot resolve the paradox. The completion has to
traverse **two subsystems**, and the `force_done` experiment (which wrote the
mask directly) bypassed both:

1. **Array side** -- the array does the column work and *writes* a completion
   signal (poll bit, ack, event register).
2. **Firmware delivery side** -- that written signal must be *delivered* to the
   pending mask `[0x10f40]`, which readies the next task.

Slice A builds the array actually running against the descriptor and instruments
which subsystem is the gate. The live array becomes the instrument that resolves
the paradox static RE could not.

## Prior facts this design rests on

Established during iter18 (all committed on this branch):

- **Firmware and array are 100% isolated.** No `crate::device` / `crate::interpreter`
  reference exists under `src/firmware/`. The firmware interp runs in a sealed
  sandbox over its own flat `Bus` (six `Vec<u8>` backing stores in
  `src/firmware/mmio.rs`). The bridge is entirely new code.
- **The polled addresses are Mailbox aperture, not Array aperture.** In the
  firmware bus map, `0x27000000..0x28000000` is `Region::Mailbox` (plain-RAM
  stub); the real AIE Array aperture is `0x04000000..0x08000000` (dropped/logged
  stub). Every address stuck boot polls -- `0x2727n000`, `0x27010d28`, ack
  `0x2727n114` -- lands in the Mailbox RAM stub today.
- **The descriptor.** Task 0x10f10's work-fn (shared executor `0x588c`) builds a
  7-word descriptor at firmware DRAM `0xfae0`:
  `{[0]=1 valid, [4]=1, [8]=0xf colmask, [c]=0, [10]=task_ptr, [14]=0, [18]=0}`,
  cache-flushes (Dhwbi), then polls `0x8c68`. No tag/sequence field.
- **The poll handshake.** Per-column poll struct at `0xf9e0 + k*0x60`
  (`[0]` byte bit3 = pending, `[8]` = tag matched via `Bne a9,a2`); HW page
  `0x2727n000` bit0/bit1; ack `0x2727n114`.
- **The delivery machinery.** `deliver_pending_events` (`0xcadc`) ->
  `wake_tasks_by_event_mask` (`0xd84c`) sets `[0x10f40]`. Event source is HW reg
  `0x27010d28`, read only by `sched_event_poll` (`0x5524`, never reached
  naturally) or via interrupt (never delivered at INTLEVEL 2).
- **Alias-correct data path.** `Cpu::data_read32/data_write32/data_read8/data_write8`
  are DTLB-translation-authoritative; `Bus::data_load32` is physical/24-bit-masked.
  All writeback in this experiment MUST use the `Cpu::data_*` API.

## Scope

**In scope (Slice A):** a disposable falsifier experiment that emits the
completion signals a finished column-init would produce, event-ordered on the
real descriptor post, and instruments write-vs-deliver separately.

**Out of scope (deferred to A1 / B):**

- Real per-column *compute* (the `task_ptr` semantics remain unknown) -- A1.
- A durable firmware<->`DeviceState` Bus-integrated bridge -- B.
- Any firmware-control-flow manipulation (no direct mask poke, no forced
  event-poll). The experiment emits array-side signals only and lets the
  firmware's own machinery succeed or fail.

## Design

### 1. Shape and coupling -- probe-owned, reuse the harness

Slice A is an experiment, not the B bridge, so it stays out of the firmware
`Bus` hot path entirely. New probe in `src/firmware/mod.rs`,
`m2c_probe_array_responder`, reusing the existing step-loop pattern (same bones
as `m2c_probe_exec_trace` and its `XDNA_FW_TRACE_SEEDPOLL` seeding). It owns the
`Proc`, steps the CPU, watches for the descriptor post, and injects the
writeback via the alias-correct `cpu.data_write32` / `data_write8`.

Zero `Bus` refactor, zero touch to `src/device/`. When A2 reveals the real
mechanism, that is when B's real Bus-integrated bridge gets built -- designed to
fit the truth instead of a guess.

### 2. Trigger -- descriptor post, event-ordered

The failed synchronous experiment seeded blindly every step. A2's whole
difference is *ordering*: it fires only when the firmware actually posts.

- **Trigger condition:** descriptor at `0xfae0` shows valid (`[0xfae0]==1`) with
  colmask (`[0xfae8]`) nonzero.
- **Dedup:** respond once per post generation. Track the last-responded
  descriptor state (e.g. a hash of the 7 words, or a monotonic generation
  counter derived from re-posts) so a single post fires exactly one response and
  a genuine re-post fires a fresh one.
- **The one open factual item (plan task 1):** confirm the exact handoff signal
  the firmware uses to mean "descriptor ready" -- valid-bit set, the Dhwbi
  cache-flush, or a mailbox doorbell write. The valid-bit trigger above is the
  working assumption; task 1 confirms it before the rest is built.

### 3. What it emits -- the completion a finished column-init would produce

On a post with colmask=0xf, for each column `k` in the mask, write via the
`Cpu::data_*` API:

- Poll handshake: `0x2727k000` bit0/bit1, ack `0x2727k114`, RAM `0xf9e0 + k*0x60`
  byte bit3.
- Event register: `0x27010d28` with the completion event id.

No firmware-control-flow manipulation. We emit the array-side signals and let
the firmware's own machinery deliver (or fail to deliver) to the mask.

**Note on the event id:** the exact bit(s) `0x27010d28` should carry for the
column-init completion event is not yet pinned. Slice A writes a best-guess
(all-columns-complete pattern) and the instrumentation records whether any
firmware read consumes it; if delivery fails purely on a wrong event id, the
diagnostic still localizes the gate to delivery and the id becomes a follow-up
sweep. This is called out as a known residual, not a hidden assumption.

### 4. Instrumentation -- the two-bit diagnostic (the actual payload)

- **Bit (a) "consumed":** does the firmware poll observe our writes
  (`FUN_00008c68` takes the ack path)? Expected yes -- the synchronous
  experiment showed this fires 6660x. Instrument by counting ack-path entries
  after the post.
- **Bit (b) "delivered":** does the firmware's *own* machinery set the pending
  mask `[0x10f40]` (not us), and does boot advance past 623k? This is the real
  unknown and directly tests the delivery-side hypothesis. Instrument by watching
  `[0x10f40]` for a firmware-driven transition to nonzero and tracking the PC /
  step count for advance past the 623k wall.

### 5. Success criteria -- every outcome is informative

| (a) consumed | (b) delivered | Interpretation | Next |
|---|---|---|---|
| yes | yes, boot advances | The array-write was the whole gate. Mechanism found. | B builds the real responder |
| yes | no | Gate is delivery, not array-write. Paradox localizes to interp event/interrupt modeling. | Next slice: delivery path in the interp |
| no | -- | Trigger/handoff model is wrong. | Refine task 1 |

**Honest expectation on record:** (b) is likely to come back **no** (static
analysis predicts sched_event_poll never reached, interrupts dead at INTLEVEL 2).
That outcome is a *win*: it converts a static inference into live evidence and
points the next slice at the interp's event delivery instead of the array. If
(b) comes back yes, better still -- we found the gate.

## Testing

Per the repo's derive-from-toolchain and self-check norms:

- The probe is driven by env vars (following the existing `m2c_probe_*`
  convention) and prints a structured verdict (the two bits + advance status).
- One runnable self-check: a unit test that constructs the responder's
  writeback for a synthetic colmask and asserts the exact addresses/bits written
  via the `Cpu::data_*` API match the mapped handshake layout (guards against
  the writeback drifting from the charted addresses). No hardware needed.
- `cargo test --lib` stays green (branch baseline: lib 4031/0/31).

## Risks and mitigations

- **Wrong handoff trigger** -> plan task 1 pins it before the rest is built;
  bit (a)=no catches a wrong trigger at runtime.
- **Wrong event id on `0x27010d28`** -> called out as a known residual; the
  diagnostic still localizes the gate; id becomes a follow-up sweep if delivery
  hinges on it.
- **Aliasing** -> mandatory `Cpu::data_*` API (the retired alias tax makes all
  reads/writes translation-authoritative by construction).
- **Scope creep into A1/B** -> the responder emits signals only, never runs real
  column compute and never manipulates firmware control flow; those are
  explicitly deferred.
```
