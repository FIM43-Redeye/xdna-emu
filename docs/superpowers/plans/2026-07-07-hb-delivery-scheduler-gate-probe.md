# H-b Delivery/Scheduler-Side Gate Probe — Implementation Plan (reframed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, from observed evidence, how the boot-to-idle gate could ever clear — by (ii-a) enumerating every writer of the task-readiness field and (ii-b) auditing whether the firmware ever attempts an interrupt-delivery window — then (i) only conditionally replaying a real attempt.

**Architecture:** All work is interp-side, where the paradox localizes the gate. (ii-a) and (ii-b) are observational: run existing `m2c_probe_*` tools over a full boot and read their structured output; deliverables are documented findings. (i) is a conditional ~30-line extension of `m2c_probe_inject_interrupt`, run only if (ii-b) observes a real window attempt. No array-side code, no `src/device/` coupling, no new probes for (ii).

**Tech Stack:** Rust; the in-tree Xtensa firmware interpreter (`src/firmware/`); the existing env-var-driven `#[test]` probe harness in `src/firmware/mod.rs` (run under `XDNA_FW_PROBE=1`).

## Global Constraints

- **Derive from the toolchain / observe; never hardcode a guessed value.** All addresses are from the committed iter18 RE (`docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`) and THE PIN.
- **All firmware memory access uses the alias-correct `Cpu::data_read32/data_write32/...` API** (translation-authoritative), never bare `Bus` load/store.
- **No firmware-control-flow forcing except in (i)'s two named, safeguarded interp-state forces** (arm INTENABLE, `set_intlevel(0)` once).
- **Never pipe `cargo`/probe runs through `tail`/`head`/`grep`.** Redirect to a log file under `/home/triple/.claude/jobs/13115116/tmp/` and Read it. Probe runs are EMU-heavy (full boot ≈ 1.5M interp steps, tens of seconds+); run the long ones with `run_in_background` and Read the log when done.
- **`cargo test --lib` stays green** (branch baseline: lib 4031 pass / 0 fail / 31 ignored). Assert "0 failed" and "+N vs baseline", never an absolute hardcoded count.
- **Commit messages end with:** `Generated using Claude Code.` No emoji.
- **Findings recorded in** `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md` (append Session-5 sections).

**Key addresses (from the RE, verbatim):**
- `task_dispatcher` = `0xd7f0`; done-flag / readiness check site = `0xd828` (`l32i.n a10,[a4+0x30]`, `a4`=current task)
- shared work-fn = `0x588c`; poll `FUN_00008c68` entry = `0x8c88`
- `wake_tasks_by_event_mask` = `0xd84c`; `deliver_pending_events` = `0xcadc`
- event/status reg = `0x27010d28`; GENERAL_EXCEPTION_HANDLER = `0x2958`; exception decode gap = `0xd903`
- scheduler current-task ptr = `[0x2278]`; first task = `0x10f10`, its readiness field `[task+0x30]` = `[0x10f40]`; task B = `0x9040`
- boot wall ≈ 58754 steps; force_done cascade reaches ≈ 623097

**Verified probe facts (from source, not assumed):**
- `m2c_probe_disasm_range`: env `XDNA_FW_DISASM="start:end"` (single var). Static linear disasm over the range.
- `m2c_probe_store_search`: env `XDNA_FW_STORE_DISP` (hex, default `0x30`). Lists every `S32i/S16i/S8i` with that displacement (any base) + a disp histogram.
- `m2c_probe_call_xref`: env `XDNA_FW_XREF` (comma-sep hex targets). BACKWARD (who-calls-X), DIRECT calls only (`callx*` not shown).
- `m2c_probe_poll_watch`: env `XDNA_FW_POLL_ADDR` (comma-sep hex, required) + `XDNA_FW_MAX` (default 1_500_000). Reads each addr every step, logs value changes `(n, pc, addr, old->new)`; alias-safe.
- `m2c_probe_intenable_watch`: NO env (MAX=1_000_000 hardcoded). Diffs `intenable`/`interrupt`/`intlevel()` each step, logs every transition with PC+symbol, plus `armed_at` and `first_level0_after_arm`.
- `m2c_probe_inject_interrupt` (`src/firmware/mod.rs:1412`): env `XDNA_FW_INT_WARMUP` (60000), `XDNA_FW_INT_STATUS` (hex, ffffffff), `XDNA_FW_INT_LINE` (hex, 0=>use INTENABLE), `XDNA_FW_INT_RUN` (400000), `XDNA_FW_INT_RESEED`. Already tracks `first_gen_exc`/`first_wake`/`first_pending`/`min_intlevel`/`lvl0_windows`.
- Interp API: `pub intenable`/`pub interrupt` (`interp/mod.rs:301,304`), `regs.intlevel()`/`regs.set_intlevel(u32)` (`regfile.rs:119,125`), `interrupt_deliverable()` is `pub` (`interp/mod.rs:475`), `Cpu::new(entry: u32)` (no `Default`).

---

## Task 1 (ii-a): Enumerate every writer of the readiness field `[task+0x30]`

**Files:**
- Modify (append findings only): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`
- Tools (no code change): `m2c_probe_store_search`, `m2c_probe_disasm_range`, `m2c_probe_call_xref`, `m2c_probe_poll_watch`

**Interfaces:**
- Consumes: the committed RE addresses.
- Produces: the **writer map** — every store site that could write `[task+0x30]`, whether its base is a task pointer, the condition gating it, and (from the runtime watch) confirmation that none fire in EMU. Task 2 and the final verdict consume this.

Investigation, not TDD — each step runs a probe and records a fact.

- [ ] **Step 1: Enumerate `+0x30` store sites.** Run (background; Read the log after):
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_STORE_DISP=0x30 \
    cargo test --lib m2c_probe_store_search -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/store30.log 2>&1
  ```
  Read `store30.log`. Record every `pc symbol op` hit.

- [ ] **Step 2: Filter to task-pointer bases.** For each hit's function, disassemble it to see whether the store's base register holds a task/scheduler pointer (vs an unrelated struct). Run per candidate function, e.g.:
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_DISASM=0xd84c:0xd8c0 \
    cargo test --lib m2c_probe_disasm_range -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/dis_wake.log 2>&1
  ```
  Repeat for `deliver_pending_events` (`XDNA_FW_DISASM=0xcadc:0xcb60`) and any other `+0x30` store site from Step 1. Read each log. Record which sites actually target the task-readiness field.

- [ ] **Step 3: Backward call-graph to the readiness writers.** Run:
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_XREF=0xd84c,0xcadc \
    cargo test --lib m2c_probe_call_xref -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/xref_wake.log 2>&1
  ```
  Read `xref_wake.log`. Record the direct callers of `wake_tasks_by_event_mask` and `deliver_pending_events`. If a caller set is empty (only `callx*` reaches them), record that as an indirect/table-dispatch finding and note it — that itself explains why the path is unreached in steady state.

- [ ] **Step 4: Runtime confirmation the writers never fire.** Run (background; long — full boot):
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_POLL_ADDR=0x10f40 \
    cargo test --lib m2c_probe_poll_watch -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/poll_10f40.log 2>&1
  ```
  Read `poll_10f40.log`. Expected: `[0x10f40]` never changes across the boot (no writer fires). Record the observed result (any change, with its PC, would itself break the paradox).

- [ ] **Step 5: Record the writer map** in the RE doc under `## Session-5 (ii-a): readiness-field writer map`, with: the store-site list (task-pointer-based ones flagged), the caller chain to each writer, the `callx*`/indirect note if any, and the runtime "never written" confirmation. State explicitly whether `wake_tasks_by_event_mask` is the SOLE writer or whether a non-event writer exists.

- [ ] **Step 6: Commit.**
  ```bash
  git add docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "docs(#140): iter18 Session-5 (ii-a) -- readiness-field [task+0x30] writer map

Generated using Claude Code."
  ```

---

## Task 2 (ii-b): Audit the INTENABLE / PS.INTLEVEL write history — the GATE

**Files:**
- Modify (append findings only): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`
- Tool (no code change): `m2c_probe_intenable_watch`

**Interfaces:**
- Consumes: Task 1's writer map (to interpret whether the event path is the only readiness route).
- Produces: **the GATE verdict** — does the firmware ever attempt an interrupt-delivery window (arm a non-`0x1` INTENABLE bit and/or lower INTLEVEL)? — plus, on an attempt, the specific INTENABLE bit and PC to replay in Task 3. The GATE decides whether Task 3 runs.

- [ ] **Step 1: Run the INTENABLE/INTLEVEL audit.** Run (background; long — full boot):
  ```bash
  XDNA_FW_PROBE=1 \
    cargo test --lib m2c_probe_intenable_watch -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/intenable.log 2>&1
  ```
  Read `intenable.log`. Record: every INTENABLE transition (value + PC), every INTLEVEL transition (value + PC), `armed_at`, and `first_level0_after_arm`.

- [ ] **Step 2: Read the observed verdict** (this is a direct yes/no from the log, not an inference):
  - Does INTENABLE ever hold a value other than `0x1` (i.e. a non-mailbox bit armed)? If yes, record which bit and the PC that armed it.
  - Does INTLEVEL ever drop below 2 in the stuck region (after the wall ≈58754)? If yes, record the PC (`wsr.ps`/`rsil`/`waiti`) and whether `first_level0_after_arm` is set.

- [ ] **Step 3: Classify the GATE.**
  - **Attempt present** (INTENABLE gains a non-`0x1` bit and/or INTLEVEL drops): the firmware tries to open a window. → Task 3 runs, replaying the observed arm bit + window PC.
  - **No attempt** (INTENABLE only `0x1`, INTLEVEL never < 2): readiness is not interrupt-based. → Task 3 does NOT run; the completion route is a (ii-a) path that stalls for a non-interrupt reason. Hand back the (ii-a) stall points for a new investigation.

- [ ] **Step 4: Record the GATE verdict** in the RE doc under `## Session-5 (ii-b): INTENABLE/INTLEVEL audit + GATE`, stating the observed transitions, the attempt/no-attempt verdict, and (on attempt) the INTENABLE bit + PC for Task 3, or (on no-attempt) the redirect to (ii-a)'s stall.

- [ ] **Step 5: Commit.**
  ```bash
  git add docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "docs(#140): iter18 Session-5 (ii-b) -- INTENABLE/INTLEVEL audit + delivery-window GATE

Generated using Claude Code."
  ```

**GATE:** Proceed to Task 3 ONLY on an **attempt-present** verdict. On **no-attempt**, STOP — the plan's decidable core is complete; the result (readiness is not interrupt-based) redirects to a fresh (ii-a)-stall investigation, out of this plan's scope.

---

## Task 3 (i, CONDITIONAL): Replay the firmware's observed delivery-window attempt

**Runs only if Task 2's GATE = attempt-present.** Extends `m2c_probe_inject_interrupt` to reproduce the firmware's *own* observed arm bit + window (from Task 2 Step 4) and instrument whether the ISR chain readies task 0x10f10.

**Files:**
- Modify: `src/firmware/mod.rs` (the `m2c_probe_inject_interrupt` fn, `src/firmware/mod.rs:1412-1540`)
- Modify (append findings): `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`

**Interfaces:**
- Consumes: Task 2's INTENABLE arm bit (`XDNA_FW_INT_ARM`), the window PC (`XDNA_FW_INT_FORCELVL_PC`), the event/status value (`XDNA_FW_INT_STATUS`).
- Produces: the (i) verdict mapped to the spec success table (ISR fires + advances / livelock artifact / decode-gap fault).

- [ ] **Step 1: Add the safeguarded window-opening inputs.** In `m2c_probe_inject_interrupt`, after `let reseed = ...` (`src/firmware/mod.rs:1432`), add:
  ```rust
  // (i) window-opening (safeguarded): arm the firmware's OWN observed AIE bit
  // and force ONE level-0 window at the observed PC, so a seeded event is
  // actually deliverable. Force-once + livelock detection avoid the re-entry
  // artifact (poll never runs; an unacked bit re-fires 0x8c88<->0x2958 forever).
  let arm = env_hex("XDNA_FW_INT_ARM", 0);            // INTENABLE bit from (ii-b)
  let force_pc = env_hex("XDNA_FW_INT_FORCELVL_PC", 0x8c88);
  let force_lvl0 = std::env::var("XDNA_FW_INT_FORCELVL").is_ok();
  const GEN_EXC_PC: u32 = 0x2958;
  const POLL_PC: u32 = 0x8c88;
  ```

- [ ] **Step 2: Arm the bit and isolate it at injection.** Replace the injection block (`src/firmware/mod.rs:1473-1475`):
  ```rust
  // Inject.
  let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
  proc.cpu.interrupt |= fire;
  ```
  with:
  ```rust
  // Inject: arm the observed AIE bit, seed the status reg, raise ONLY the
  // armed line (XDNA_FW_INT_LINE isolates it from the 0x1 mailbox doorbell).
  if arm != 0 {
      proc.cpu.intenable |= arm;
  }
  let fire = if line != 0 { line } else { fire | arm };
  let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
  proc.cpu.interrupt |= fire;
  eprintln!("  armed INTENABLE|={arm:#010x} -> INTENABLE={:#010x}, firing {fire:#010x}", proc.cpu.intenable);
  ```

- [ ] **Step 3: Force ONE window + detect livelock.** Before the run loop (near `src/firmware/mod.rs:1477`, after `let mut n = warmup;`), add:
  ```rust
  let mut window_opened = false;
  let mut reentry_pairs = 0u64;   // 0x8c88 -> 0x2958 ping-pong count
  let mut last_was_poll = false;
  ```
  Then inside the loop, immediately after `let pc = proc.cpu.pc;` (`src/firmware/mod.rs:1489`), add:
  ```rust
  if force_lvl0 && !window_opened && pc == force_pc {
      proc.cpu.regs.set_intlevel(0);   // open exactly one window
      window_opened = true;
  }
  // Livelock detector: repeated poll -> gen-exc transitions = unacked re-entry.
  if pc == GEN_EXC_PC && last_was_poll {
      reentry_pairs += 1;
      if reentry_pairs >= 32 {
          stop = format!("livelock: unacked re-entry {POLL_PC:#x}<->{GEN_EXC_PC:#x} x{reentry_pairs}");
          break;
      }
  }
  last_was_poll = pc == POLL_PC;
  ```

- [ ] **Step 4: Add a non-tautological self-check.** Add near the probe tests in `src/firmware/mod.rs` (runs in the default suite — NOT gated on `XDNA_FW_PROBE`):
  ```rust
  #[test]
  fn forced_window_delivers_a_pending_interrupt() {
      // Guards (i)'s actual force logic: arming a bit + set_intlevel(0) + a
      // pending bit must let a few steps TAKE the interrupt (reach the general
      // exception vector), not merely report deliverable. Uses a tiny synthetic
      // program is overkill; assert the delivery precondition transitions across
      // a real step of a fresh CPU whose PC we point at a nop-like fetch.
      use crate::firmware::xtensa::interp::Cpu;
      let mut cpu = Cpu::new(0);
      cpu.intenable = 0x0000_0002;
      cpu.interrupt = 0x0000_0002;
      cpu.regs.set_intlevel(2);
      assert!(!cpu.interrupt_deliverable(), "INTLEVEL=2 masks delivery");
      cpu.regs.set_intlevel(0);
      assert!(cpu.interrupt_deliverable(), "open window must make it deliverable");
  }
  ```
  (Note: this still reduces to the deliverability predicate because a full stepped-delivery check needs a loaded image; the real delivery is exercised by the experiment run in Step 6. If a stepped check is wanted, gate it behind the firmware image like the probes. Keep this as the compile/precondition guard; do NOT bill it as proving the experiment.)

- [ ] **Step 5: Build and run the self-check.** Run:
  ```bash
  cargo test --lib forced_window_delivers_a_pending_interrupt \
    > /home/triple/.claude/jobs/13115116/tmp/selfcheck.log 2>&1
  cargo build --lib > /home/triple/.claude/jobs/13115116/tmp/build.log 2>&1
  ```
  Read both logs. Expected: test `ok`, build clean.

- [ ] **Step 6: Run the (i) replay** with Task 2's observed arm bit / window PC / event value (substitute the real values; example placeholders shown):
  ```bash
  XDNA_FW_PROBE=1 XDNA_FW_INT_ARM=<bit-from-ii-b> XDNA_FW_INT_LINE=<bit-from-ii-b> \
    XDNA_FW_INT_STATUS=<event-val> XDNA_FW_INT_FORCELVL=1 XDNA_FW_INT_FORCELVL_PC=<pc-from-ii-b> \
    XDNA_FW_INT_RUN=800000 \
    cargo test --lib m2c_probe_inject_interrupt -- --nocapture \
    > /home/triple/.claude/jobs/13115116/tmp/replay.log 2>&1
  ```
  Read `replay.log`. Note `first_gen_exc`, `first_wake`, `first_pending`, `lvl0_windows`, the stop reason (real advance vs `livelock:` vs `Unknown at 0xd903`), and the final pc / advance.

- [ ] **Step 7: Record the (i) verdict** in the RE doc under `## Session-5 (i): delivery-window replay`, mapping the outcome to the spec success table: ISR fires + `[0x10f40]` set by fw + advance (window sufficient); `livelock:` stop (harness artifact — discard); `Unknown at 0xd903` (ISR taken, faults on unmodeled op — decoder gap, not a gate).

- [ ] **Step 8: Confirm the default suite is still green.** Run:
  ```bash
  cargo test --lib > /home/triple/.claude/jobs/13115116/tmp/fulltest.log 2>&1
  ```
  Read `fulltest.log`. Expected: 0 failed; pass count = baseline + 1 (the new self-check). The probe early-returns without `XDNA_FW_PROBE`, so it does not run in the default suite.

- [ ] **Step 9: Commit.**
  ```bash
  git add src/firmware/mod.rs docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md
  git commit -m "feat(#140): iter18 Session-5 (i) -- safeguarded delivery-window replay

Extends m2c_probe_inject_interrupt to replay the firmware's OWN observed
INTENABLE arm + level-0 window (force-once, livelock-detected, AIE bit isolated
from the mailbox doorbell) and instrument whether the ISR chain readies task
0x10f10. Conditional on the (ii-b) audit observing a real window attempt.

Generated using Claude Code."
  ```

---

## Self-Review

**Spec coverage:**
- Spec (ii-a) writer enumeration → Task 1 (store_search + disasm + call_xref + poll_watch). Covered.
- Spec (ii-b) INTENABLE/INTLEVEL audit + GATE → Task 2 (intenable_watch). Covered.
- Spec (i) conditional replay + safeguards (force-once, livelock detect, isolate bit) → Task 3 Steps 1-3, gated on Task 2. Covered.
- Spec success table (incl. livelock artifact, decode-gap outcome) → Task 3 Step 7. Covered.
- Spec "no new probe for (ii), no device coupling" → Tasks 1-2 add no code; no task touches `src/device/`. Covered.

**Placeholder scan:** No "TBD"/"handle edge cases". Task 3's `<bit-from-ii-b>` etc. are explicit outputs of Task 2, not open placeholders — the code is complete; the values are experiment inputs the GATE supplies. If Task 2 = no-attempt, Task 3 does not run and the placeholders never resolve, by design.

**Type consistency:** `set_intlevel(u32)` / `intlevel()` (regfile), `intenable`/`interrupt` (`u32`, interp mod.rs:301/304), `interrupt_deliverable()` (`pub`, mod.rs:475), `Cpu::new(0)` (not `Default`), `env_hex` closure already defined in the probe, `data_write32`/`data_read32` — all match verified source. `fire` is re-bound (not shadowed-then-unused): `if line != 0 { line } else { fire | arm }`.

**Env-var correctness (the prior plan's failure mode):** every probe invocation uses the verified var — `XDNA_FW_DISASM` (single start:end), `XDNA_FW_STORE_DISP`, `XDNA_FW_XREF`, `XDNA_FW_POLL_ADDR`, `m2c_probe_intenable_watch` (no env), `XDNA_FW_INT_*`. No `DIS_LO/HI`, no `TRACE_FROM/LEN`.

**Constraint compliance:** no `cargo | tail`; all runs redirect to a job-dir log and Read; no absolute pass-count assertion (0 failed + baseline+1).

**TDD framing:** Tasks 1-2 are observation (existing tools → findings), no code, no unit test. Task 3 adds code and carries the `forced_window_delivers_a_pending_interrupt` guard, explicitly scoped as a precondition check (the experiment run is the real evidence), per the review's non-tautology note.
