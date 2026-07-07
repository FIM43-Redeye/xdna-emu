# H-b Slice A — Delivery/Scheduler-Side Gate Probe

**Issue:** #140 (firmware-emulation dream / boot-to-idle)
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Date:** 2026-07-07
**Status:** Design approved (revised after adversarial review killed the original array-responder design)

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
(~565k steps). So something real sets that mask, and static charting alone
cannot see what.

## The paradox localizes to the delivery/scheduler subsystem

The completion has to traverse **two subsystems**, and the `force_done`
experiment (which wrote the mask directly) bypassed both:

1. **Array side** -- the array does the column work and *writes* a completion
   signal (poll bit, ack, event register).
2. **Firmware delivery/scheduler side** -- that written signal must be
   *delivered* to the pending mask `[0x10f40]`, which readies the next task, and
   the busy dispatch loop must *yield* for that to matter.

`force_done` proved subsystem-2's downstream works (mask set -> boot cascades).
The synchronous experiment proved subsystem-1's poll signals are already-drained
no-ops (the poll consumes bit3, `[a8] &= 0xf7`, then finds it clear next cycle).
THE PIN's sharpened conclusion states it directly: the array responder's job is
*NOT* tag-matched poll completions -- the open gate is **"what makes the busy
dispatch loop EXIT / a task get readied -- a scheduler-spanning loop-exit
condition."**

**Therefore the gate is interp-side, in delivery/scheduling.** It is not any
value the array writes into RAM.

## Why the original array-responder design was rejected

The first revision of this spec proposed an "A2 responder": a probe that, on the
firmware's descriptor post at `0xfae0`, emits the completion signals a finished
column-init would produce (poll bits, ack, event reg `0x27010d28`), and
instruments write-vs-deliver. An adversarial Opus review (2026-07-07) killed it
as theater, and the critique holds:

- **The `delivered` outcome is a constant, not an unknown.** The responder writes
  only Mailbox/DRAM RAM. The delivery gate `interrupt_deliverable()` (interp
  `mod.rs:475`) needs `intlevel()==0 && (interrupt & intenable)!=0`, and all
  three fail during the wall. Nothing an array-RAM-writer emits touches INTLEVEL,
  INTENABLE, or `cpu.interrupt`, and `sched_event_poll` (the only polled reader
  of `0x27010d28`) is never reached. So an event-register write has **no
  reachable consumer by construction of the interp** -- the outcome is decided by
  state the experiment never perturbs.
- **It contradicts THE PIN.** It builds the poll responder that finding calls a
  no-op and never touches the loop-exit condition that finding calls the real
  gate.
- **The dedup is ill-defined.** The descriptor is byte-identical every re-post
  with no sequence field, and the poll consumes (clears) bit3 each cycle -- so a
  once-per-post responder drains then goes silent, and an every-cycle responder
  is the already-falsified blind seed.

The lesson banked: the array-side signal was never the unknown. The gate is the
firmware's own delivery/scheduler control flow.

## Scope

Two experiments, in order, both interp-side where the gate actually is:

- **(ii) Static loop-exit trace (first).** Determine the busy dispatch loop's
  actual exit/yield conditions and which one the firmware is waiting on. Cheap,
  no manufactured state, frames (i).
- **(i) Counterfactual delivery probe (second, framed by (ii)).** A ~20-30 line
  extension of `m2c_probe_inject_interrupt` that manufactures a delivery window
  and tests whether an AIE-completion event, once deliverable, actually readies
  task 0x10f10 through the ISR chain.

**Out of scope (deferred):**

- Any array-side responder / column-compute (the original A2 and A1) -- only
  revisit *after* (i) confirms an AIE event, once delivered, can ready the task;
  only then does emitting the event have a consumer worth being faithful to.
- The durable firmware<->`DeviceState` bridge (B).

## Design

### Part (ii) — static loop-exit trace

**Question:** why does task 0x10f10 busy-spin on the `0x8c68` poll instead of
yielding to the scheduler, and what exact condition would make the dispatch loop
exit or the task block/re-ready?

**Method (static + light dynamic, no state forcing):**

1. Delimit the loop body from known anchors: dispatcher (`task_dispatcher`
   `0xd7f0`) -> shared work-fn (`0x588c`) -> poll (`FUN_00008c68` `0x8c88`) ->
   back to dispatcher.
2. Enumerate every branch in that body that could **exit** the loop or **yield**
   (a call into the scheduler that blocks the current task / selects another).
   For each, record the condition tested and the memory/register it reads.
3. Determine whether 0x10f10 ever invokes a yield/block primitive (sets its state
   byte `[task+0x2c]` to a blocked value, or calls the scheduler to re-select) --
   or whether it unconditionally busy-spins.
4. Map why 0x10f10 stays the current task: the scheduler's current-task selection
   (`[SCHED+40]=0x2278`), the ready set, and whether any *other* task is ready
   but not selected.

**Deliverable:** a precise statement of the loop-exit/yield condition and the
specific value/event the firmware waits on to take it -- which tells us whether
the exit is **event-driven** (validating (i)'s counterfactual shape) or
**synchronous** (in which case (i) is the wrong shape and we redirect).

### Part (i) — counterfactual delivery probe

**Question:** if an AIE-completion event were made deliverable, does the ISR
chain ready task 0x10f10 and advance boot -- i.e., is the *only* thing missing
the delivery window?

**Method (extend `m2c_probe_inject_interrupt`):**

1. Warm to the steady-state poll (existing probe capability).
2. Arm the AIE-completion bit in INTENABLE (the bit identified by (ii), or a
   small sweep if (ii) leaves it open).
3. Open a level-0 window -- force `PS.INTLEVEL=0` at the poll instruction (or at
   the yield point (ii) identifies, if one exists).
4. Seed `0x27010d28` with the completion event and set `cpu.interrupt |= bit`;
   step.
5. Instrument the ISR chain end-to-end: does `FUN_00005580` run ->
   `wake_tasks_by_event_mask(1<<id)` (`0xd84c`) -> `[0x10f40]` set (by firmware,
   not us) -> dispatcher completes 0x10f10 -> boot advances past 623k?

**This is a counterfactual:** it manufactures the delivery window by force to
isolate "*if* delivered, does the ISR ready the task?" It does not by itself
reveal how the real firmware opens the window -- that is (ii)'s job, and is why
(ii) runs first.

### Instrumentation and success criteria

| Part | Signal | Interpretation |
|---|---|---|
| (ii) | loop-exit condition is event-driven | (i)'s counterfactual is the right shape; proceed |
| (ii) | loop-exit condition is synchronous (no yield/block; waits on a memory value) | (i) is the wrong shape; redirect to what writes that value |
| (i) | ISR chain fires, `[0x10f40]` set by fw, boot advances | Gate narrows to "why the loop never yields a level-0 window" -- the scheduler question, the real H-b task |
| (i) | ISR chain fires but no advance | Gate is deeper (ready-list / idle transition) |
| (i) | ISR chain does not fire | ISR delivery path itself is broken or the event id is wrong -- sweep id / recheck INTENABLE bit |

## Testing

- Both experiments are driven by env vars (existing `m2c_probe_*` convention) and
  print a structured verdict.
- (i) leaves one runnable self-check: a unit test asserting the probe's INTENABLE
  arm + INTLEVEL force + `cpu.interrupt` set produce a `interrupt_deliverable()
  == true` precondition for a synthetic event (guards the counterfactual actually
  opens the window it claims to). No hardware needed.
- `cargo test --lib` stays green (branch baseline: lib 4031/0/31).

## Risks and mitigations

- **(i) is counterfactual (manufactured window)** -> (ii) runs first to confirm
  the gate is event-driven and to frame what the real window is; (i) is read as
  "is delivery the *only* missing piece," not "this is how the fw does it."
- **Forcing `PS.INTLEVEL`/INTENABLE could itself corrupt** -> control-register
  forces are more surgical than RAM writes, and we discriminate on the *ISR
  chain firing* (FUN_00005580 -> wake -> mask -> dispatch), not merely on "did
  boot move," so a corruption-driven wander is distinguishable from real
  delivery.
- **Wrong event id / INTENABLE bit** -> (ii) aims to pin them; a small sweep is
  the fallback, and "ISR does not fire" is an explicit, informative outcome.
- **(ii) finds the exit is synchronous** -> that is an informative result, not a
  failure: it redirects the whole investigation to the memory value the loop
  waits on, and we skip (i).
```
