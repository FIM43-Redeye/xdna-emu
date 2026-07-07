# H-b: Array Seam Integration — Proving the Completion Path

**Issue:** #140 (firmware-emulation dream / boot-to-idle)
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Date:** 2026-07-07
**Status:** KILLED by adversarial review (2026-07-07). The mailbox-doorbell seam
hypothesis is falsified: (1) no INTLEVEL window opens on worker-return (the
dispatcher is entered at level 2; `wsr.ps` restores 2, not 0); (2) the interp
models only one FLIX bundle shape, so the ISR can't be PC-traced without a
decode-implementation project; (3) the doorbell (bit 0) is the host->fw mailbox
line, the wrong event. See the finding's Session-5 section and
`project_firmware_emulation_dream` (BANKED). Kept for the record.

## Context

The firmware-side ledger is closed and the interpreter is exonerated (Session-5,
`docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`). What we
now know, disasm-verified:

- The busy-loop wall is exactly the condition `[task+0x30]==0` (task 0x10f10's
  pending mask, `= [0x10f40]`). The dispatcher (`task_dispatcher` 0xd7f0) enters a
  level-2 critical section (`0xd7f3 rsil a2,2`), and:
  - if `[task+0x30] != 0`: `0xd839 bnez.n a10 -> 0xd845` skips the run-fn, restores
    PS (`0xd845 wsr.ps a2` -> drops INTLEVEL), returns -> task done, boot advances.
  - if `[task+0x30] == 0`: calls the worker run-fn `0x588c` at `0xd842` INSIDE the
    critical section; the worker blocks in the per-column poll (`FUN_00008c68`
    0x8c88) waiting for completion that never arrives, so it never returns, so
    `0xd845` is never reached and INTLEVEL is held at 2.
- The INTLEVEL=2 hold is FAITHFUL firmware behavior, correctly modeled (the
  `wsr.ps` restore exists and executes; it is gated behind the run-fn returning).
  No interp fix is a prerequisite.
- The downstream chain is proven functional: force_done (setting `[0x10f40]`)
  cascades boot 58k->623k via the same `0xd839->0xd845` branch real delivery takes.
- The only writer of `[0x10f40]` is the wake path (`FUN_00005580+0x289 ->
  wake_tasks_by_event_mask(1<<id)`), which is array-gated: reachable on silicon only
  after array-side per-column completion. Its static dispatch is undecidable
  (0x5580/0x5524 have zero pointers); it is reached via the ISR or a
  runtime-computed run-fn, both of which depend on array activity absent in EMU.

## The seam hypothesis (the thing under test)

Stitching the dispatcher logic to the interrupt state yields a concrete, testable
mechanism for how array completion advances boot:

1. The worker posts the colmask-0xf descriptor (flushed to DRAM `0xfae0`) and
   **blocks in the poll** `0x8c68` (reads per-column pages `0x2727_n000` bit0/bit1).
   While it blocks it never returns -> INTLEVEL held at 2.
2. When the columns complete and the poll **succeeds, the worker returns** ->
   dispatcher reaches `0xd845 wsr.ps` -> **INTLEVEL drops to 0**. This return is the
   delivery window.
3. The **mailbox doorbell (INTENABLE bit 0) is the one armed interrupt.** If the
   array, on completing, rings that doorbell (with a completion message), it
   **delivers in that window** -> the ISR (`0x2958`, which carries a static hint
   straight at the waker `[0x293c]=wake_tasks+0x18`) -> the wake path -> sets
   `[0x10f40]` -> next dispatcher cycle takes the `0xd839->0xd845` "task done"
   branch -> **boot advances past the wall.**

This explains why Session-3's poll-seeding did not advance boot: seeding the poll
lets the worker return (opening the window) but rings no doorbell, so nothing
delivers in the window, `[task+0x30]` is never set, and the worker just re-posts.
Poll-completion and task-done are TWO signals; the array must produce both.

**Calibration:** the ISR->wake link is FLIX-misaligned under static disasm and not
cleanly decidable statically. This is the LEADING hypothesis, not proven. The build
is the instrument that confirms or refutes it, and Part 1 tests its hardest link
first and cheapest.

## Why this is NOT the killed "A2 responder"

The array-responder design killed at spec stage (an adversarial review; see the
concluded `2026-07-07-hb-array-responder-slice-a-design.md`) was theater because
its signals had **no reachable consumer**: it wrote the event register `0x27010d28`
(reader `sched_event_poll` never reached) and the poll pages (drained to a no-op),
while interrupt delivery was dead (INTLEVEL=2, no window). Its "delivered" outcome
was a constant fixed by state it never perturbed.

This design is built on the mechanism A2 lacked, and the difference is load-bearing:

- **The consumer is the armed mailbox doorbell (bit 0), not the unreached event
  poll.** Delivery is via the ISR, which IS entered once a window opens.
- **The window is real and we know how it opens:** the worker returns on
  poll-success -> `0xd845 wsr.ps` drops INTLEVEL to 0. A2 never opened a window;
  this design does (by satisfying the poll so the worker returns) AND rings the
  doorbell so an interrupt is pending when the window opens.
- **Part 1 tests consumer-reachability directly and first.** The exact thing that
  made A2 theater (no reachable consumer) is the first thing this design falsifies
  or confirms, cheaply, before any build.

If Part 1 shows the ISR does NOT reach the wake path, the mailbox seam is refuted
and we have saved the entire array build — a cheap, informative kill, not theater.

## Scope

Risk-first, three parts. Compute fidelity is deferred: the first cut proves the
COMPLETION PROTOCOL, not tile math.

- **Part 1 — ISR pre-flight (cheapest, first).** Force the mailbox ISR to run and
  trace it PC-following to confirm (or refute) that it reaches `wake_tasks` and sets
  `[0x10f40]`, and to surface any FLIX decode gap in the never-executed ISR path.
- **Part 2 — completion-protocol responder.** On the firmware's descriptor post,
  drive the full seam: satisfy the poll pages (worker returns, window opens) + ring
  the doorbell with the completion message (from Part 1) so it delivers in the
  window. Observe whether boot advances.
- **Part 3 — observe & attribute.** Boot advances past the wall -> #140 boot-to-idle
  closed for this path. Otherwise, the firmware's response localizes the broken link.

**Out of scope (deferred to the real bridge, "B"):**
- The durable firmware<->`DeviceState` in-Bus bridge.
- Real per-column tile/DMA compute (the responder stubs completion).
- The responder is designed to GROW into the real array integration: the
  descriptor-consume and completion-signal interfaces stay; only the stubbed
  "columns complete" step is later replaced by real `TileArray` execution.

## Design

### Part 1 — ISR pre-flight

**Question:** if the mailbox doorbell delivers in an open window, does the ISR
(`0x2958`) reach `wake_tasks_by_event_mask` and set `[0x10f40]` — and does the interp
decode the ISR's (FLIX) bundles when executing that never-run path?

**Method (extend `m2c_probe_inject_interrupt`):**
1. Warm to steady state. At the worker-return point (or by forcing `set_intlevel(0)`
   once at `0xd845`, the real restore site), open a genuine window.
2. Ring the doorbell: `cpu.interrupt |= 0x1` (the armed mailbox bit), and seed the
   mailbox message region the ISR reads (address discovered by tracing, not guessed).
3. Trace PC-following from `0x2958` through the ISR: record the exact instruction
   path, whether it reaches `wake_tasks` (0xd84c), whether `[0x10f40]` is set by
   firmware, and any `Step::Unknown` (a FLIX decode gap) with its PC.

**Success criteria:**
| Observed | Meaning |
|---|---|
| ISR runs, reaches `wake_tasks`, sets `[0x10f40]` | seam's ISR link CONFIRMED; proceed to Part 2 |
| ISR runs but does not reach the wake path | mailbox seam REFUTED; origination is the computed run-fn path — redirect, build saved |
| `Step::Unknown` at a PC in the ISR | FLIX decode gap in never-run code; fix THAT instruction concretely (targeted decode fix), then re-run Part 1 |

### Part 2 — completion-protocol responder

**Trigger:** the firmware's descriptor post — descriptor at `0xfae0` valid
(`[0xfae0]==1`) with colmask (`[0xfae8]`) nonzero, deduped per post.

**Action on post (colmask=0xf):**
1. For each column k in the mask, satisfy the poll: set `0x2727_k000` bit0/bit1 and
   RAM `0xf9e0+k*0x60` bit3 (the pages the worker's poll reads), so the worker's
   poll succeeds and it RETURNS — opening the INTLEVEL window at `0xd845`.
2. Ring the doorbell: set the mailbox message (the completion, per Part 1's decoded
   format) and `cpu.interrupt |= 0x1`, so the interrupt is pending when the window
   opens.
3. Let the firmware run and observe.

All writes use the alias-correct `Cpu::data_*` API. This is a probe-owned
orchestrator (owns the firmware `Proc`); no `DeviceState` yet — the "array" is the
completion protocol. Compute is stubbed: we assert "columns complete" and produce
the signals, we do not run tile math.

**Success criteria:**
| Observed | Meaning |
|---|---|
| boot advances past the wall (dispatcher takes `0xd839->0xd845`, `[0x10f40]` set by fw, progress to idle) | seam CONFIRMED end-to-end; #140 boot-to-idle closed for this path |
| worker returns (window opens) but no delivery / `[0x10f40]` unset | doorbell/message wrong; refine from Part 1's decoded ISR |
| worker never returns | poll-satisfaction wrong (pages/format); refine the poll write |
| `Step::Unknown` on the newly-run path | decode gap; targeted fix, re-run |

### Part 3 — observe & attribute

Run to a budget past 623k; classify the outcome per the tables above. Every outcome
localizes the next step. A confirmed advance is validated by the ISR->wake->mask->
dispatch order firing before progress (not a corruption wander).

## Coupling and growth path

Probe-owned orchestrator in `src/firmware/mod.rs` (the harness the whole
investigation used), reusing the `inject_interrupt` machinery for Part 1 and a new
`m2c_probe_array_seam` for Part 2. When the seam is confirmed, the responder's
stubbed "columns complete" step is replaced by real `TileArray`/`DeviceState`
execution (the durable bridge, B), keeping the descriptor-consume and
completion-signal interfaces.

## Testing

- Both parts are env-var-driven probes printing a structured verdict.
- Part 1, if it adds decode-gap fixes, carries a targeted unit test per fixed
  instruction (decode of that bundle at that PC produces the correct op).
- Any Part-2 code carries a self-check that the responder's writes land at the
  charted poll/doorbell addresses via `Cpu::data_*`.
- `cargo test --lib` stays green (branch baseline: lib 4031/0/31, +N for new checks).

## Risks and mitigations

- **Mailbox seam is wrong (ISR doesn't reach wake).** Part 1 falsifies it cheaply,
  before any build; the alternative (computed run-fn) becomes the redirect.
- **FLIX decode gap in the never-run ISR/wake path.** Expected possibility; fix the
  specific instruction concretely (reactive, targeted), do NOT RE the whole
  config-specific bundle format.
- **Wrong doorbell message content.** Part 1 decodes what the ISR reads; Part 2 uses
  that, not a guess.
- **A2-redux critique.** Addressed structurally: the consumer here is the armed
  doorbell + a real window, and Part 1 tests consumer-reachability first — the exact
  gap that made A2 theater.
- **Corruption wander mistaken for success.** Advance is only credited when the
  ISR->wake->mask->dispatch order is observed firing before progress.
