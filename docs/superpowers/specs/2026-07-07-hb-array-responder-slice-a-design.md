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

Two decidable static/observational scans first, then a conditional probe -- all
interp-side where the gate actually is. The scans replace the original "classify
the loop exit" framing, which an adversarial review showed is undecidable (the
loop never exits in EMU, so there is no observable exit to classify).

- **(ii-a) Enumerate every writer of the readiness field `[task+0x30]` (first).**
  Find all store sites and trace why none fire. Decidable, existing tools.
- **(ii-b) Audit the boot-long history of INTENABLE / PS.INTLEVEL writes (first).**
  One observed yes/no: does the firmware ever arm a non-mailbox (AIE) interrupt
  bit or lower INTLEVEL toward 0? Decidable from the transition log.
- **(i) Conditional delivery replay/confirmation (only if (ii-b) shows an
  attempt).** Replay the firmware's own arm/window; or, if kept purely as a
  plumbing confirmation, clearly labeled as "confirm the ISR->mask wiring
  `force_done` bypassed," never billed as "locate the gate."

**Out of scope (deferred):**

- Any array-side responder / column-compute (the original A2 and A1) -- only
  revisit *after* (i) confirms an AIE event, once delivered, can ready the task;
  only then does emitting the event have a consumer worth being faithful to.
- The durable firmware<->`DeviceState` bridge (B).

## Design

The reframe (after the plan-stage adversarial review) turns (ii) from an
undecidable classification into two decidable scans whose outcomes fork the
investigation on observed evidence, neither branch pre-excluded. Both reuse
existing probes -- no new probe is required for (ii).

### Part (ii-a) — enumerate every writer of the readiness field `[task+0x30]`

**Question:** what are *all* the code paths that could ready task 0x10f10 (set
its `[task+0x30]` field, `=0x10f40` for the first task), and why does none fire?
We "know" `wake_tasks_by_event_mask` writes it; we have never confirmed that is
the *only* writer. A non-event writer would break the paradox outright.

**Fact that sets the method:** `[0x10f40]` is addressed base-relative
(`[task_ptr + 0x30]`), never as an L32r literal (RE: "the completion target
`+0x30` is IMPLICIT in the task-struct layout, not bound per-request"). So a
value/literal xref finds nothing; a base+offset store-site scan is required.

**Method (existing probes):**

1. `m2c_probe_store_search` (`XDNA_FW_STORE_DISP=0x30`, its default) -- enumerate
   every `s32i/s16i/s8i ?, [?+0x30]` site. Over-matches (any base), so filter to
   task-pointer bases by disassembling each hit's function.
2. `m2c_probe_call_xref` (`XDNA_FW_XREF=0xd84c,0xcadc`) -- backward callers of
   `wake_tasks_by_event_mask` / `deliver_pending_events`; trace the chain up to
   the entry condition that never fires. (Direct calls only; a `callx*` gap is
   itself a finding -- indirect/table dispatch.)
3. `m2c_probe_poll_watch` (`XDNA_FW_POLL_ADDR=0x10f40`) -- full-boot runtime
   watch (alias-safe) confirming nothing writes `0x10f40` in EMU, i.e. the
   statically-found writers never fire.

**Deliverable:** the complete writer map -- every path that could ready the task
and, for each, the exact condition/PC where it stalls.

### Part (ii-b) — audit the INTENABLE / PS.INTLEVEL write history — the GATE

**Question (decidable yes/no):** over a full boot, does the firmware ever arm a
non-mailbox (AIE-candidate) bit in INTENABLE, or lower PS.INTLEVEL toward 0 --
i.e. does it *attempt* to open an interrupt-delivery window?

**Method (existing probe):** `m2c_probe_intenable_watch` already runs boot and
diffs `intenable` / `interrupt` / `intlevel()` each step, recording every
transition with its causing PC, plus `armed_at` (first non-zero INTENABLE) and
`first_level0_after_arm`. Read its transition log directly. (Modeling notes to
respect when reading: interrupt entry raises EXCM, not INTLEVEL; the only
intlevel mutators are `wsr.ps` / `rsil` / `waiti`; `xsr` is unmodeled but the
firmware uses none.)

**GATE verdict from the observed log:**

- **Attempt present** -- firmware writes INTENABLE with a non-`0x1` value and/or
  lowers INTLEVEL in the stuck region. The window is real; if delivery still
  fails, either the window closes before delivery or the interp mishandles it.
  -> proceed to (i), *replaying the firmware's own attempt* (not a counterfactual).
- **No attempt** -- INTENABLE only ever `0x1`, INTLEVEL never drops below 2. The
  readiness path is not interrupt-based; the completion must come via a (ii-a)
  path that stalls for a non-interrupt reason. -> skip (i); follow (ii-a)'s stall
  points. This is a real, informative result, not a failure.

### Part (i) — conditional delivery replay / plumbing confirmation

**Runs only on an (ii-b) "attempt present" verdict.** Then extend
`m2c_probe_inject_interrupt` to reproduce the firmware's *own* observed arm +
window and instrument whether the ISR chain (`0x2958` -> `0xd84c` -> `[0x10f40]`
set by fw -> dispatcher completes 0x10f10 -> advance) fires.

If (ii-b) is "no attempt" but we still want the isolated plumbing check (does
`ISR -> wake -> mask -> dispatch` work *when* a window is open?), it may be run
as an explicit, **clearly-labeled counterfactual** -- "confirm the wiring
`force_done` bypassed," never "locate the gate" -- and only with the safeguards
below, since forcing a window the firmware never opens is otherwise an artifact
generator.

**Required safeguards (from the review):**

- Force `set_intlevel(0)` **once** to open a single window, not every step at the
  poll PC (which livelocks: the poll never executes, and an unacked synthetic bit
  re-fires `0x8c88`<->`0x2958` forever, masquerading as "ISR fires, no advance").
- Add explicit **livelock detection** (same `0x8c88`<->`0x2958` pair N times ->
  report "unacked re-entry," distinct from a real "deeper gate").
- Isolate the AIE bit: set `XDNA_FW_INT_LINE` to the candidate bit so the
  mailbox doorbell (`0x1`) is not fired alongside it.

### Success criteria

| Part | Observed | Interpretation |
|---|---|---|
| (ii-a) | a writer of `[task+0x30]` outside the event path | paradox broken -- follow that writer's stall |
| (ii-a) | `wake_tasks` is the sole writer | readiness is event-gated; (ii-b) decides how |
| (ii-b) | INTENABLE gains a non-`0x1` bit and/or INTLEVEL drops | firmware attempts a window -> (i) replays it |
| (ii-b) | INTENABLE stays `0x1`, INTLEVEL never < 2 | not interrupt-based -> follow (ii-a)'s stall; skip (i) |
| (i) | ISR chain fires, `[0x10f40]` set by fw, boot advances | the replayed window is sufficient; gate is "why fw doesn't reach it sooner" |
| (i) | livelock detected (unacked re-entry) | harness artifact, not a gate -- discard |
| (i) | `Step::Unknown` at the exception decode gap (`0xd903`) | ISR taken but faults on an unmodeled op -- a decoder gap, not a gate |

## Testing

- Both scans are existing env-var-driven probes printing structured verdicts; no
  new code, so no new self-check for (ii).
- (i), *if it runs and adds code*, leaves one **non-tautological** check: run a
  few `step()`s with the window force applied and assert the probe's
  `first_gen_exc` becomes `Some` (i.e. the force actually causes delivery) --
  not a restatement of `interrupt_deliverable()`.
- `cargo test --lib` stays green (branch baseline: lib 4031/0/31).

## Risks and mitigations

- **(ii-b) classification bias** -> removed: the verdict is read from an *observed*
  transition log (does INTENABLE/INTLEVEL change, yes/no), not inferred from an
  unobservable exit.
- **(i) counterfactual artifacts** -> (i) is now conditional on (ii-b) showing a
  real attempt; if run as a pure plumbing check it is labeled as such and carries
  force-once + livelock detection + the decode-gap outcome, so an artifact is not
  mistaken for a gate.
- **store_search over-matches** -> filter by disassembling each hit; `poll_watch`
  cross-checks at runtime.
- **`callx*` caller gap in call_xref** -> treated as a positive finding
  (table/indirect dispatch), not a dead end.
```
