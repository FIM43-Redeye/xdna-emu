# H-b Delivery/Scheduler-Side Gate Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Locate the real boot-to-idle gate by (ii) statically tracing why the firmware dispatch loop never yields, then (i) testing whether an AIE-completion event, once made deliverable, readies the stuck task via the ISR chain.

**Architecture:** Two experiments, interp-side where the paradox localizes the gate. (ii) is investigation using existing `m2c_probe_*` tools; its deliverable is a documented loop-exit/yield condition that gates (i). (i) is a ~30-line extension of `m2c_probe_inject_interrupt` that arms the AIE INTENABLE bit and forces a level-0 window, then instruments the ISR chain end to end. No array-side code, no `src/device/` coupling.

**Tech Stack:** Rust; the in-tree Xtensa firmware interpreter (`src/firmware/`); the existing probe harness in `src/firmware/mod.rs` (env-var driven `#[test]` probes, run under `XDNA_FW_PROBE=1`).

## Global Constraints

- **Derive from the toolchain / observe; never hardcode a guessed value.** Addresses used here are all from the committed iter18 RE (`docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`) and THE PIN.
- **All firmware memory access uses the alias-correct `Cpu::data_read32/data_write32/data_read8/data_write8` API** (translation-authoritative), never bare `Bus` load/store.
- **No firmware-control-flow forcing beyond the two named interp-state forces in (i)** (arm INTENABLE, `set_intlevel(0)`); no RAM pokes to manufacture completion.
- **`cargo test --lib` stays green** (branch baseline: lib 4031 pass / 0 fail / 31 ignored). Probes are `#[ignore]`-style (they early-return unless `XDNA_FW_PROBE=1`), so they must not perturb the default count.
- **Commit messages end with:** `Generated using Claude Code.` No emoji.
- **Findings recorded in** `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md` (append Session-5 sections), not new files.

**Key addresses (from the RE, verbatim):**
- `task_dispatcher` = `0xd7f0`; done-flag check site = `0xd828` (`l32i.n a10,[a4+0x30]`)
- shared work-fn (executor) = `0x588c`; poll `FUN_00008c68` entry = `0x8c88`
- `wake_tasks_by_event_mask` = `0xd84c`; `deliver_pending_events` = `0xcadc`
- message dispatcher `FUN_00005580`; `sched_event_poll` = `0x5524` (reads `0x27010d28`)
- event/status reg = `0x27010d28`; GENERAL_EXCEPTION_HANDLER = `0x2958`
- scheduler current-task ptr = `[0x2278]`; first task = `0x10f10`, its pending mask = `[0x10f40]`; task B = `0x9040`
- task state byte = `[task+0x2c]` (6 = ready); done-flag = `[task+0x30]`
- boot wall step count ≈ 58754; force_done cascade reaches ≈ 623097

---

## Task 1 (ii-a): Delimit the dispatch loop body and enumerate every exit/yield branch

**Files:**
- Modify (append findings only): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`
- Tools used (no code change): existing `m2c_probe_disasm_range`, `m2c_probe_exec_trace`, `m2c_probe_call_xref` in `src/firmware/mod.rs`

**Interfaces:**
- Consumes: the committed RE addresses above.
- Produces: a written "loop-body branch table" — for each branch in the dispatcher→work-fn→poll cycle that can leave the loop or call the scheduler, its PC, the condition tested, and the memory/register read. Task 2 consumes this table.

This task is investigation, not TDD. Each step is run-a-probe / record-a-fact.

- [ ] **Step 1: Disassemble the dispatcher body.** Run:
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_DIS_LO=0xd7f0 XDNA_FW_DIS_HI=0xd8a0 \
    cargo test --lib m2c_probe_disasm_range -- --nocapture 2>&1 | tee /home/triple/.claude/jobs/13115116/tmp/dis_dispatcher.log
  ```
  Read `dis_dispatcher.log`. Expected: the dispatcher's task-select + done-flag check (`0xd828`) + the call to the work-fn.

- [ ] **Step 2: Disassemble the work-fn and poll entry.** Run the same probe for `XDNA_FW_DIS_LO=0x588c XDNA_FW_DIS_HI=0x5980` and for `XDNA_FW_DIS_LO=0x8c88 XDNA_FW_DIS_HI=0x8d40`, tee to `dis_workfn.log` / `dis_poll.log`. Read both.

- [ ] **Step 3: Identify branch/call instructions that could exit or yield.** From the three disassembly logs, list every `beqz/bnez/bne/beq/ball/...` and every `call*/callx*` in the loop body. For each, note the register/condition and (from the preceding `l32i`) the memory address it derives from. Record as the branch table.

- [ ] **Step 4: Confirm the steady-state loop actually cycles these PCs.** Run:
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_TRACE_FROM=58000 XDNA_FW_TRACE_LEN=2000 \
    cargo test --lib m2c_probe_exec_trace -- --nocapture 2>&1 | tee /home/triple/.claude/jobs/13115116/tmp/trace_loop.log
  ```
  Read `trace_loop.log`. Confirm the PC cycle is dispatcher(`0xd7f0`)→work-fn(`0x588c`)→poll(`0x8c88`)→dispatcher and record the observed period. Mark which branch-table entries are actually reached vs skipped in steady state.

- [ ] **Step 5: Record the branch table** in the RE findings doc under a new `## Session-5 (ii-a): dispatch-loop exit/yield branch table` heading, with a row per branch: `PC | insn | condition | memory read | reached-in-steady-state? | exit-or-yield?`.

- [ ] **Step 6: Commit.**
  ```bash
  git add docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "docs(#140): iter18 Session-5 (ii-a) -- dispatch-loop exit/yield branch table

Generated using Claude Code."
  ```

---

## Task 2 (ii-b): Determine why task 0x10f10 busy-spins instead of yielding — GATE

**Files:**
- Modify (append findings only): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`
- Tools used (no code change): existing `m2c_probe_current_task_timeline`, `m2c_probe_exec_trace`

**Interfaces:**
- Consumes: Task 1's branch table.
- Produces: (a) a statement of whether 0x10f10 ever calls a yield/block primitive or unconditionally busy-spins; (b) the exact value/event the loop waits on to exit; (c) **the GATE verdict: is the exit event-driven (Task 3 proceeds) or synchronous (Task 3 is replaced by a redirect)?** Task 3 consumes the GATE verdict, the INTENABLE bit candidate, and the event-id candidate.

- [ ] **Step 1: Map the current-task timeline.** Run:
  ```bash
  XDNA_FW_PROBE=1 cargo test --lib m2c_probe_current_task_timeline -- --nocapture 2>&1 | tee /home/triple/.claude/jobs/13115116/tmp/task_timeline.log
  ```
  Read `task_timeline.log`. Record: at the wall (~58754), what is `[0x2278]` (current task), and does it ever change after the wall? Confirm 0x10f10 is pinned as current task (or record what supersedes it).

- [ ] **Step 2: Check for a yield/block primitive in the loop body.** From Task 1's branch table, identify any call that writes the current task's state byte `[task+0x2c]` to a non-ready value or re-enters the scheduler. If none is reached in steady state, record "no yield reached — unconditional busy-spin." If one exists but is not taken, record the condition that gates it and the memory address that condition reads.

- [ ] **Step 3: Identify the wait value.** For the branch that keeps the loop cycling (the poll's fall-through, per the RE the `FUN_00008c68` consume path that clears bit3), record the exact memory location and value the loop is waiting to observe, and whether any *other* task is in the ready set but unselected (read task B `0x9040` state byte `[0x906c]` and its pending mask).

- [ ] **Step 4: Classify the exit.** Decide, from Steps 1-3:
  - **Event-driven** if the loop-exit / task-ready transition depends on the pending mask `[0x10f40]` being set by `wake_tasks_by_event_mask`, whose only trigger is an event through `0x27010d28` (i.e., the force_done target is downstream of event delivery). → Task 3 proceeds.
  - **Synchronous** if the loop exits purely on a memory value written outside the event path (no dependence on `0x10f40` / `wake_tasks`). → Task 3 is out of scope; record the redirect target (the writer of that value) and stop.

- [ ] **Step 5: Record the GATE verdict** in the RE doc under `## Session-5 (ii-b): why 0x10f10 busy-spins + GATE`, including the INTENABLE-bit candidate for the AIE completion line and the event-id candidate for `0x27010d28` (from Task 1's reads of the event path), or an explicit "unknown — sweep in Task 3" if not derivable statically.

- [ ] **Step 6: Commit.**
  ```bash
  git add docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "docs(#140): iter18 Session-5 (ii-b) -- 0x10f10 busy-spin cause + event-vs-synchronous GATE

Generated using Claude Code."
  ```

**GATE:** If Step 4 classified the exit as **synchronous**, STOP here — Task 3 does not apply; hand the redirect target back for a new investigation. Proceed to Task 3 only on an **event-driven** verdict.

---

## Task 3 (i): Extend `m2c_probe_inject_interrupt` to open a real delivery window

**Files:**
- Modify: `src/firmware/mod.rs` (the `m2c_probe_inject_interrupt` fn, currently `src/firmware/mod.rs:1412-1540`)
- Modify (append findings): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`

**Interfaces:**
- Consumes: from Task 2 — the GATE=event-driven verdict, the INTENABLE-bit candidate (`XDNA_FW_INT_ARM`), and the event-id/status value (`XDNA_FW_INT_STATUS`, already an input).
- Produces: the (i) verdict — does the ISR chain (`GEN_EXC 0x2958` → `wake 0xd84c` → `[0x10f40]` set → dispatcher completes 0x10f10 → advance past 623k) fire once a level-0 window is open?

The current probe seeds `0x27010d28` and sets `cpu.interrupt`, but delivery never happens because INTENABLE lacks the AIE bit and INTLEVEL stays 2. This task adds two named interp-state forces to open the window, then reuses the probe's existing ISR-chain instrumentation (`first_gen_exc`, `first_wake`, `first_pending`, `min_intlevel`, `lvl0_windows`).

- [ ] **Step 1: Write the failing self-check test.** Add to `src/firmware/mod.rs` (near the other probe tests, not `#[ignore]`-gated — this one runs in the default suite):
  ```rust
  #[test]
  fn inject_window_makes_interrupt_deliverable() {
      // The (i) counterfactual precondition: arming an INTENABLE bit + forcing
      // INTLEVEL=0 + setting the matching cpu.interrupt bit must make the interp
      // report the interrupt deliverable. Guards that the probe's window-opening
      // forces actually open the window they claim to.
      use crate::firmware::xtensa::interp::Cpu;
      let mut cpu = Cpu::default();
      cpu.intenable = 0x0000_0002; // a non-mailbox (AIE-candidate) bit
      cpu.interrupt = 0x0000_0002;
      cpu.regs.set_intlevel(0);
      assert!(
          cpu.interrupt_deliverable(),
          "arm+force+set must make the interrupt deliverable"
      );
      // And with INTLEVEL raised it must NOT be deliverable (the pre-force state).
      cpu.regs.set_intlevel(2);
      assert!(!cpu.interrupt_deliverable(), "INTLEVEL=2 must mask delivery");
  }
  ```

- [ ] **Step 2: Run it to confirm it compiles and passes** (it exercises existing API — this is a guard, expected PASS; if it fails, the API assumption is wrong and the probe extension below must be adjusted to the real signatures):
  ```bash
  cargo test --lib inject_window_makes_interrupt_deliverable 2>&1 | tail -20
  ```
  Expected: `test ... ok`. If `interrupt_deliverable` is not `pub`, make it `pub(crate)` in `src/firmware/xtensa/interp/mod.rs` and re-run.

- [ ] **Step 3: Add the two window-opening inputs to the probe.** In `m2c_probe_inject_interrupt`, after the existing `let reseed = ...` line (currently `src/firmware/mod.rs:1432`), add:
  ```rust
  // (i) window-opening forces: arm the AIE completion bit in INTENABLE and
  // force a level-0 window at the poll, so a seeded event is actually
  // deliverable (the steady-state loop holds INTLEVEL=2 with INTENABLE=0x1).
  let arm = env_hex("XDNA_FW_INT_ARM", 0); // AIE INTENABLE bit candidate (Task 2)
  let force_lvl0_at = env_hex("XDNA_FW_INT_FORCELVL_PC", 0x8c88); // poll entry
  let force_lvl0 = std::env::var("XDNA_FW_INT_FORCELVL").is_ok();
  ```

- [ ] **Step 4: Apply the INTENABLE arm at injection.** Replace the injection block (currently `src/firmware/mod.rs:1473-1475`):
  ```rust
  // Inject.
  let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
  proc.cpu.interrupt |= fire;
  ```
  with:
  ```rust
  // Inject: arm the AIE bit (so it is not masked), seed the status reg, and
  // raise the interrupt line.
  if arm != 0 {
      proc.cpu.intenable |= arm;
  }
  let fire = fire | arm; // fire the armed bit even if INTENABLE was 0x1 before
  let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
  proc.cpu.interrupt |= fire;
  eprintln!("  armed INTENABLE|={arm:#010x} -> INTENABLE={:#010x}", proc.cpu.intenable);
  ```
  (Note: `fire` is shadowed here; the earlier `let fire = if line == 0 {...}` at `:1451` stays, this re-binds after arming so an armed bit is fired even when the pre-arm INTENABLE was just `0x1`.)

- [ ] **Step 5: Force a level-0 window at the poll PC each step.** In the run loop, immediately after `let pc = proc.cpu.pc;` (currently `src/firmware/mod.rs:1489`), add:
  ```rust
  if force_lvl0 && pc == force_lvl0_at {
      proc.cpu.regs.set_intlevel(0); // open a deliverable window at the poll
  }
  ```

- [ ] **Step 6: Build and run the self-check plus a compile check of the probe.**
  ```bash
  cargo test --lib inject_window_makes_interrupt_deliverable 2>&1 | tail -20
  cargo build --lib 2>&1 | tail -20
  ```
  Expected: test `ok`, build clean.

- [ ] **Step 7: Run the (i) experiment** with Task 2's INTENABLE bit and event id (example values shown; substitute Task 2's):
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_INT_ARM=0x2 XDNA_FW_INT_STATUS=0x2 \
    XDNA_FW_INT_FORCELVL=1 XDNA_FW_INT_FORCELVL_PC=0x8c88 XDNA_FW_INT_RUN=800000 \
    cargo test --lib m2c_probe_inject_interrupt -- --nocapture 2>&1 | tee /home/triple/.claude/jobs/13115116/tmp/inject_window.log
  ```
  Read `inject_window.log`. The instrumentation already prints `interrupt taken (@0x2958)`, `wake_tasks (@0xd84c)`, `[0x10f40] pending set`, `min intlevel`, `level-0 windows`, and the final pc / advance.

- [ ] **Step 8: If the ISR chain does not fire, sweep the INTENABLE bit / event id.** If `first_gen_exc` is `None` with a window open (`lvl0_windows > 0`), re-run Step 7 varying `XDNA_FW_INT_ARM` over the candidate bits from Task 2's event-path reads (e.g. `0x2`, `0x4`, `0x8`, ...) and `XDNA_FW_INT_STATUS` to match. Record which, if any, causes `0x2958` to be entered.

- [ ] **Step 9: Record the (i) verdict** in the RE doc under `## Session-5 (i): counterfactual delivery probe`, mapping the observed outcome to the spec's success table (ISR fires + advances → gate is "why no window"; fires + no advance → deeper; never fires → delivery path/id wrong). Note `lvl0_windows` and whether the advance (if any) is real (ISR-driven) vs a corruption wander (distinguished by whether `0x2958`→`0xd84c`→`0x10f40` fired in order before the advance).

- [ ] **Step 10: Confirm the default suite is still green.**
  ```bash
  cargo test --lib 2>&1 | tail -15
  ```
  Expected: `4032 passed` (baseline 4031 + the new self-check); 0 failed. The probe itself early-returns without `XDNA_FW_PROBE`, so it does not run in the default suite.

- [ ] **Step 11: Commit.**
  ```bash
  git add src/firmware/mod.rs docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "feat(#140): iter18 Session-5 (i) -- counterfactual delivery probe (arm INTENABLE + force level-0 window)

Extends m2c_probe_inject_interrupt to open a real delivery window (arm the AIE
INTENABLE bit, force PS.INTLEVEL=0 at the poll) and instrument whether an
AIE-completion event, once deliverable, readies task 0x10f10 through the ISR
chain (0x2958 -> 0xd84c -> [0x10f40] -> dispatch). Adds a self-check that the
window-opening forces make interrupt_deliverable() true.

Generated using Claude Code."
  ```

---

## Self-Review

**Spec coverage:**
- Spec Part (ii) static loop-exit trace → Tasks 1 + 2. Covered.
- Spec Part (i) counterfactual delivery probe → Task 3. Covered.
- Spec GATE (event-driven vs synchronous) → Task 2 Step 4 + the explicit GATE between Task 2 and 3. Covered.
- Spec success table → Task 3 Step 9 maps outcomes to it. Covered.
- Spec self-check (window actually opens) → Task 3 Steps 1-2. Covered.
- Spec "no array code / no device coupling" → no task touches `src/device/`. Covered.

**Placeholder scan:** No "TBD"/"handle edge cases". The only runtime-supplied values (INTENABLE bit, event id) are explicit env-var inputs with a Task-2 provenance and a Task-3 Step-8 sweep fallback — code is complete, values are inputs, per the spec's known-residual handling.

**Type consistency:** `set_intlevel(u32)` (interp `mod.rs:1141`), `cpu.intenable`/`cpu.interrupt` (`u32`, `mod.rs:301-304`), `interrupt_deliverable()` (`mod.rs:476`), `data_read32/data_write32` all match the read code. `env_hex` closure is already defined in the probe. The shadowing of `fire` is called out explicitly.

**Note on TDD framing:** Tasks 1-2 are investigation (existing tools → documented findings), not test-first code; they carry no unit test because they add no code. Task 3 adds code and carries the `inject_window_makes_interrupt_deliverable` self-check, per the repo's one-runnable-check norm.
