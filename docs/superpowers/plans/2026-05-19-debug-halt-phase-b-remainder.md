# debug_halt Phase B Remainder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After each implementation Unit, run the two-stage review (spec-compliance + code-quality, fresh sonnet, parallel) + controller adjudication that verifies by reading code — the established discipline for this subsystem (same as Phase B Units 1/1b).

**Goal:** Close out Phase B of `debug_halt` conclusively: implement the §5.2 count-step state machine (HW-anchored by G2) and the §5.1 single-step halt boundary (the principled split Maya chose 2026-05-19), flipping the `debug_halt` coverage verdict to `Full`.

**Architecture:** Two scoped, hardware-free Rust units on the existing `core_debug` "projection + latch, does not drive the interpreter" substrate. Unit 2 adds a live count-step budget armed by `Debug_Control0[5:2]`, decremented per committed bundle by a new coordinator consumer adjacent to `consume_pending_single_step`; expiry latches `halted` and the existing `interpreter.rs:181` `is_halted` gate produces the before-commit-of-N+1 boundary G2 derived. Unit 3 routes PC-wired event single-step (Debug_Control1[14:8] SSTEP_EVENT == Core_PC_0..3) through the existing Unit-1 pre-execute seam (before-commit, arming known pre-bundle); watchpoint/mem/lock/range-wired SSTEP_EVENT stays after-commit as a documented modeling decision (no coherent before-commit point).

**Tech Stack:** Rust (`src/device/core_debug/`, `src/interpreter/engine/coordinator.rs`), `cargo test --lib`, `crates/xdna-archspec` coverage spine.

**Spec:** `docs/superpowers/specs/2026-05-18-debug-halt-design.md` §5.1, §5.2, §5.3, §6, §8.
**Findings (derived inputs, do not re-derive):** `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` G2 + "Phase B inputs (closing)".

---

## Conventions for every task

- Emulator repo: `/home/triple/npu-work/xdna-emu` (branch `dev`). All Rust changes commit here.
- After any Rust change: `cargo build` (the XRT plugin loads `libxdna_emu.so` at runtime; coordinator changes affect it — rebuild per CLAUDE.md build discipline even though Phase B remainder is unit-tested, not bridge-tested).
- Run unit tests sandbox-safe: `TMPDIR=/tmp/claude-1000 cargo test --lib`.
- No emoji. End every commit message with `Generated using Claude Code.` (internal project — no pre-approval).
- The PostToolUse rustfmt hook auto-formats edited `.rs` files; do not hand-format.
- No hardware in this plan. No bridge runs. Pure `cargo test --lib`.
- Decisions locked by Maya 2026-05-19 (do not re-litigate): **principled split** for §5.1; **accepted modeling readings** for §5.2 (N counts committed bundles, halt fires before the (N+1)th commits; count+halt-bit `0x11` → bit[0] immediate-halt precedence, budget armed latent; expiry clears the budget, only a fresh `Debug_Control0` write re-arms).

---

## File structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `docs/superpowers/specs/2026-05-18-debug-halt-design.md` | §5.1/§5.2/§5.3/§6/§8 updated to the locked decisions (coherence before implementation) | 1 |
| `docs/superpowers/plans/2026-05-19-debug-halt-phase-b-remainder.md` | This plan | 1 |
| `src/device/core_debug/mod.rs` | `count_step_remaining`/`halt_cause_count_step` state; `write_debug_control0` arm; `tick_count_step`; `has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap` | 2,3 |
| `src/interpreter/engine/coordinator.rs` | Count-step tick consumer; sstep-PC pre-execute seam extension | 2,3 |
| `src/device/core_debug/tests.rs` | Count-step unit tests; sstep-PC before-commit tests; rewrite `sstep_event_via_pc_event_path`; clarify `sstep_event_match_sets_pending_latch` | 2,3 |
| `crates/xdna-archspec/src/coverage/units.rs` | `debug_halt` narrative + `Modeled { completeness: Full }` | 4 |
| `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` | Closing note: Units 2/3 implemented | 4 |

---

## Task 1: Spec + plan coherence (committed before any implementation)

**Goal:** Make the spec reflect the two locked decisions so implementation is written against a coherent spec, per the established "spec/plan/findings coherence committed before each implementation" discipline. No code.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-18-debug-halt-design.md`
- Create (this file): `docs/superpowers/plans/2026-05-19-debug-halt-phase-b-remainder.md`

- [ ] **Step 1: Rewrite spec §5.1 (single-step halt boundary — resolved, principled split).**

Replace the §5.1 "Single-step halts — deferred to §5.2/G2, not this unit." bullet and its trailing "Two existing tests ... stay valid until then." sentence with the resolution:

> - **Single-step halts — RESOLVED (Phase B remainder Unit 3, principled split, Maya 2026-05-19).** Event-driven single-step (Debug_Control1[14:8] SSTEP_EVENT) splits by arming-condition observability:
>   - **PC-wired single-step** — SSTEP_EVENT == a point PC event (Core_PC_0..3). The arming condition is a PC match, known *before* the bundle, so silicon halts *before* the bundle commits — the same boundary as the G1 PC_Event_Halt seam. Routed through the existing Unit-1 pre-execute seam via `has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap`.
>   - **Watchpoint / mem / lock / PC-range-wired single-step** — arming is only known *after* the bundle runs; there is no coherent before-commit point. **Documented modeling decision:** stays after-commit via the unchanged `check_event_halt` → `pending_single_step` → `consume_pending_single_step` path. PC-range single-step (Core_PC_Range_0_1/2_3) is bucketed here deliberately: a range's before-commit boundary (range entry vs. each in-range step) is itself ambiguous, so it takes the defensible after-commit reading.
>   - Test disposition: `core_debug/tests.rs` `sstep_event_via_pc_event_path` is rewritten to assert the before-commit query path (PC-wired); `sstep_event_match_sets_pending_latch` stays valid (it exercises the bare `check_event_halt` latch with no PC_Event wired, i.e. the after-commit path) — comment clarified.

- [ ] **Step 2: Make spec §5.2 modeling decisions explicit.**

In §5.2, replace "unobservable edges (re-arm on resume, count=0, halt-bit interaction) are implemented to the most natural reading and documented inline as explicit modeling decisions, citing the finding." with the locked readings:

> Silicon-unobservable edges are implemented to these explicit modeling decisions (Maya 2026-05-19; documented inline citing the G2 finding; finer characterization remains the §8 count-step forward-commitment):
> - **N counts committed bundles; the halt fires *before* the (N+1)th commits.** Implemented as a post-execute per-committed-bundle decrement; on expiry `request_halt()` latches `halted` and the existing `interpreter.rs:181` `is_halted` gate prevents the next bundle — before-commit of bundle N+1. Consistent with G1 and with the G2 observation (`N=4` halted in the prologue, `LANDED:0`).
> - **count + halt-bit (`0x11`):** the halt bit `[0]`'s immediate halt takes precedence (documented async halt); the N-budget is still armed and latent, applying only if the core is later resumed.
> - **Resume / re-arm:** budget expiry clears the budget (`None`); `request_resume()` never re-arms; only a fresh `Debug_Control0` write with `Single_Step_Count > 0` re-arms (mirrors `N=0` = disabled, write = arm).

- [ ] **Step 3: Update spec §5.3 component boundaries.** Add a "Phase B remainder (Units 2/3)" block listing the exact loci: `core_debug/mod.rs` (`count_step_remaining`/`halt_cause_count_step`, `write_debug_control0` arm, `tick_count_step`, `has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap`); `coordinator.rs` (count-step tick adjacent to `consume_pending_single_step`; sstep-PC seam check adjacent to the Unit-1 `has_sync_pc_trap_at` seam); `core_debug/tests.rs` (count-step + sstep-PC tests, the 1046 rewrite); `units.rs` (→ `Full`). Mark §5.1/§5.2 as "Phase B remainder, this plan" rather than "deferred".

- [ ] **Step 4: Update spec §6 testing + §8.** In §6: add the count-step arm/tick/expire/resume unit tests and the sstep-PC before-commit tests; state that after this remainder `Modeled { completeness: Full }` is reached (G1 + routing + G2 count-step + single-step boundary all closed) and the coverage artifacts are regenerated with zero drift. In §8: confirm the **count-step finer-characterization** forward-commitment stays open (decrement cadence / exact instruction boundary / larger-N / `0x11` on silicon — only `N=4` was observed); the Phase B remainder ships the natural reading, it does not close the §8 tracker.

- [ ] **Step 5: Commit (xdna-emu `dev`).**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/specs/2026-05-18-debug-halt-design.md docs/superpowers/plans/2026-05-19-debug-halt-phase-b-remainder.md
git commit -m "debug-halt: Phase B remainder -- spec+plan coherence (principled split + locked count-step readings)

Spec 5.1 single-step boundary RESOLVED as the principled split (PC-wired
single-step -> before-commit via the Unit-1 seam; watchpoint/range -> documented
after-commit). Spec 5.2 count-step modeling decisions made explicit (N counts
committed bundles, halt before N+1; 0x11 halt-bit precedence + latent budget;
expiry clears, only a fresh write re-arms). 5.3/6/8 updated; plan saved.
Decisions locked with Maya 2026-05-19.

Generated using Claude Code."
```

---

## Task 2: Phase B Unit 2 — §5.2 count-step state machine

**Goal:** `Debug_Control0[5:2]` `Single_Step_Count` arms a live N-committed-bundle budget; a coordinator consumer decrements it per committed bundle and halts at expiry (before the (N+1)th bundle commits), matching G2.

**Files:**
- Modify: `src/device/core_debug/mod.rs`
- Modify: `src/interpreter/engine/coordinator.rs`
- Test: `src/device/core_debug/tests.rs`

- [ ] **Step 1: Write the failing unit tests.**

Append to `src/device/core_debug/tests.rs` (in the same `#[cfg(test)] mod`/`tests` module the existing `sstep_event_*` tests live in, so `CoreDebugState`/`make_*` are in scope):

```rust
#[test]
fn count_step_arm_from_debug_control0() {
    // Debug_Control0[5:2]=N (halt bit clear) arms a live N-budget.
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x10); // N=4 (4<<2), halt bit [0]=0
    assert_eq!(s.count_step_remaining, Some(4));
    assert!(!s.is_halted(), "arming alone does not halt");
}

#[test]
fn count_step_zero_disables() {
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x10);
    s.write_debug_control0(0x00); // N=0 -> disabled
    assert_eq!(s.count_step_remaining, None);
    assert!(!s.tick_count_step(), "disabled budget never halts");
    assert!(!s.is_halted());
}

#[test]
fn count_step_decrements_then_halts_before_n_plus_1() {
    // N=2: bundles 1 and 2 commit (tick after each), halt latched on the
    // 2nd tick so the existing is_halted gate blocks bundle 3.
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x08); // N=2 (2<<2)
    assert!(!s.tick_count_step(), "after bundle 1: budget 2->1, no halt");
    assert_eq!(s.count_step_remaining, Some(1));
    assert!(s.tick_count_step(), "after bundle 2: budget expires, halt");
    assert!(s.is_halted());
    assert!(s.halt_cause_count_step);
    assert_eq!(s.count_step_remaining, None, "expiry clears the budget");
}

#[test]
fn count_step_expiry_clears_no_rearm_on_resume() {
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x04); // N=1
    assert!(s.tick_count_step(), "N=1 halts on the first tick");
    assert_eq!(s.count_step_remaining, None);
    s.request_resume(); // resume must NOT re-arm the budget
    assert_eq!(s.count_step_remaining, None);
    assert!(!s.tick_count_step(), "post-expiry ticks are no-ops until a fresh write");
}

#[test]
fn count_step_halt_bit_precedence_with_latent_budget() {
    // 0x11 = halt bit [0]=1 AND Single_Step_Count=4. Bit[0]'s immediate
    // halt takes precedence; the N-budget is still armed (latent).
    let mut s = CoreDebugState::new();
    s.enabled = true;
    s.write_debug_control0(0x11);
    assert!(s.is_halted(), "halt bit [0] halts immediately (precedence)");
    assert_eq!(s.count_step_remaining, Some(4), "budget armed latent");
}
```

- [ ] **Step 2: Run, verify they fail.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib count_step_ 2>&1`
Expected: compile error / FAIL — `count_step_remaining`, `halt_cause_count_step`, `tick_count_step` do not exist yet.

- [ ] **Step 3: Add the state fields.**

In `src/device/core_debug/mod.rs`, in the `CoreDebugState` struct (after the `pc_event0..3` block near line 258), add:

```rust
    /// Live count-step budget (Debug_Control0[5:2] Single_Step_Count).
    /// `Some(n)` = n committed bundles remain before a before-commit halt;
    /// `None` = count-step disabled. Distinct from the raw
    /// `single_step_count` config field: this is the decrementing live
    /// counter. G2 (NPU1 silicon, 2026-05-19): count-step is live hardware;
    /// `N=4` halts in the prologue before the first store. Modeling
    /// decisions (silicon-unobservable edges, spec §5.2, Maya 2026-05-19):
    /// N counts committed bundles, halt fires before the (N+1)th commits;
    /// count+halt-bit (0x11) -> bit[0] immediate-halt precedence, budget
    /// armed latent; expiry clears the budget, only a fresh Debug_Control0
    /// write re-arms (request_resume never re-arms).
    pub(super) count_step_remaining: Option<u32>,
    /// Latched cause: the core was halted by count-step budget expiry.
    pub(super) halt_cause_count_step: bool,
```

In the `impl Default for CoreDebugState` block (near lines 292-329), add to the struct literal (alongside `sync_trap_consumed_at: None,`):

```rust
            count_step_remaining: None,
            halt_cause_count_step: false,
```

- [ ] **Step 4: Arm the budget in `write_debug_control0`.**

In `src/device/core_debug/mod.rs`, replace `write_debug_control0` (lines 854-865) body's tail so the function reads:

```rust
pub(super) fn write_debug_control0(&mut self, value: u32) {
    let halt_req = (value & DBG_CTRL0_HALT_MASK) != 0;
    if halt_req {
        self.request_halt();
    } else {
        self.request_resume();
    }

    let sstep_count = ((value & DBG_CTRL0_SSTEP_COUNT_MASK) >> DBG_CTRL0_SSTEP_COUNT_LSB) as u8;
    self.single_step_count = sstep_count;
    self.single_step = sstep_count > 0;

    // §5.2 count-step arm (G2 silicon-derived, 2026-05-19). A non-zero
    // Single_Step_Count arms a live N-committed-bundle budget; N=0 disables.
    // Independent of the halt bit: for 0x11 the bit[0] immediate halt above
    // takes precedence; the budget is still armed (latent) and applies if
    // the core later resumes (spec §5.2 modeling decision).
    self.count_step_remaining = if sstep_count > 0 { Some(sstep_count as u32) } else { None };
}
```

- [ ] **Step 5: Add `tick_count_step`.**

In `src/device/core_debug/mod.rs`, immediately after `consume_pending_single_step` (ends line 680), add:

```rust
/// Decrement the count-step budget by one committed bundle. The
/// coordinator calls this after each committed bundle, adjacent to
/// `consume_pending_single_step`. On expiry it latches `halted` (via
/// `request_halt`) and the count-step cause; the existing
/// `interpreter.rs` `is_halted` gate then prevents the next bundle from
/// committing — the before-commit-of-bundle-(N+1) boundary G2 derived
/// (spec §5.2). Expiry clears the budget; only a fresh Debug_Control0
/// write re-arms. Returns true iff this tick expired the budget.
pub fn tick_count_step(&mut self) -> bool {
    match self.count_step_remaining {
        Some(n) if n > 1 => {
            self.count_step_remaining = Some(n - 1);
            false
        }
        Some(_) => {
            self.count_step_remaining = None;
            self.halt_cause_count_step = true;
            self.request_halt();
            true
        }
        None => false,
    }
}
```

- [ ] **Step 6: Run the unit tests, verify they pass.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib count_step_ 2>&1`
Expected: all five `count_step_*` tests PASS.

- [ ] **Step 7: Wire the coordinator consumer.**

In `src/interpreter/engine/coordinator.rs`, at the `consume_pending_single_step` call site (line 712), add the count-step tick immediately after it:

```rust
            tile.core_debug.consume_pending_single_step();
            // §5.2 count-step: decrement the live budget per committed
            // bundle; expiry latches halted, the is_halted gate blocks the
            // next bundle (before-commit of N+1, G2-derived).
            tile.core_debug.tick_count_step();
```

- [ ] **Step 8: Full suite + build.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1` — expected: green (xdna-emu; no regressions in existing `sstep_event_*` / async tests).
Then: `cargo build 2>&1` — expected: clean (coordinator change picked up; CLAUDE.md rebuild discipline).

- [ ] **Step 9: Commit.**

```bash
cd /home/triple/npu-work/xdna-emu
git add src/device/core_debug/mod.rs src/device/core_debug/tests.rs src/interpreter/engine/coordinator.rs
git commit -m "debug-halt Phase B Unit 2: count-step state machine (Debug_Control0[5:2])

Single_Step_Count arms a live N-committed-bundle budget; a coordinator
consumer adjacent to consume_pending_single_step decrements per committed
bundle and on expiry latches halted -- the existing is_halted gate then
blocks bundle N+1 (before-commit, G2-derived NPU1 silicon 2026-05-19).
Locked modeling decisions: N counts committed bundles; 0x11 -> bit[0]
immediate-halt precedence + latent budget; expiry clears, only a fresh
write re-arms. Pure-logic, unit-tested; preserves core_debug's
projection+latch character.

Generated using Claude Code."
```

---

## Task 3: Phase B Unit 3 — §5.1 single-step halt boundary (principled split)

**Goal:** PC-wired event single-step (Debug_Control1[14:8] SSTEP_EVENT == Core_PC_0..3, matching the current PC) halts *before* the bundle commits via the existing Unit-1 pre-execute seam. Watchpoint/mem/lock/range-wired SSTEP_EVENT stays after-commit (documented; `check_event_halt` path unchanged).

**Files:**
- Modify: `src/device/core_debug/mod.rs`
- Modify: `src/interpreter/engine/coordinator.rs`
- Test: `src/device/core_debug/tests.rs`

- [ ] **Step 1: Write the failing / rewritten unit tests.**

Rewrite the existing `sstep_event_via_pc_event_path` test (tests.rs lines ~1047-1060) to assert the before-commit query path:

```rust
#[test]
fn sstep_event_via_pc_event_path() {
    // §5.1 principled split (Maya 2026-05-19): SSTEP_EVENT wired to a
    // point PC event (Core_PC_0) is the before-commit-eligible case --
    // the arming condition (PC match) is known before the bundle, so the
    // pre-execute seam halts BEFORE the bundle commits, parallel to G1's
    // PC_Event_Halt seam (no deferred pending_single_step round-trip).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_0, 0, 0);

    assert!(state.has_sync_sstep_pc_trap_at(0x100));
    assert!(!state.is_halted(), "query alone does not halt");
    state.consume_sync_sstep_pc_trap(0x100);
    assert!(state.is_halted(), "PC-wired single-step halts before-commit");

    // Idempotent: consumed at this PC, does not re-fire after resume
    // (mirrors consume_sync_pc_trap; re-arming is the §8-tracked edge).
    state.request_resume();
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));
}

#[test]
fn sstep_pc_trap_only_for_point_pc_events() {
    // Watchpoint / non-PC SSTEP_EVENT is NOT before-commit eligible: no
    // PC_Event slot mapping -> query is false (it stays on the unchanged
    // after-commit check_event_halt -> pending_single_step path).
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.pc_event0 = make_pc_event(true, 0x100);
    // SSTEP_EVENT = 32 (a non-PC event id, e.g. a watchpoint).
    state.debug_ctrl1 = make_dbg_ctrl1(0, 32, 0, 0);
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));

    // PC-range single-step is deliberately bucketed after-commit too.
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_RANGE_0_1, 0, 0);
    assert!(!state.has_sync_sstep_pc_trap_at(0x100));
}

#[test]
fn sstep_pc_trap_requires_valid_matching_pc_event() {
    let mut state = CoreDebugState::new();
    state.enabled = true;
    state.debug_ctrl1 = make_dbg_ctrl1(0, EVENT_CORE_PC_2, 0, 0);
    // PC_Event2 not VALID -> no trap.
    state.pc_event2 = make_pc_event(false, 0x200);
    assert!(!state.has_sync_sstep_pc_trap_at(0x200));
    // VALID but PC mismatch -> no trap.
    state.pc_event2 = make_pc_event(true, 0x200);
    assert!(!state.has_sync_sstep_pc_trap_at(0x208));
    // VALID and matching -> trap.
    assert!(state.has_sync_sstep_pc_trap_at(0x200));
}
```

Then clarify the comment on `sstep_event_match_sets_pending_latch` (tests.rs ~989) — the test body is unchanged (it still passes; it exercises the bare `check_event_halt` latch with no PC_Event wired, i.e. the after-commit path). Replace its comment block with:

```rust
    // §5.1 principled split: this is the documented AFTER-commit case.
    // check_event_halt is called directly with no PC_Event slot wired, so
    // the before-commit seam (has_sync_sstep_pc_trap_at) does not engage;
    // the SSTEP_EVENT latch arms and the halt is deferred to consume (the
    // triggering bundle commits first). Watchpoint/mem/lock/range-wired
    // single-step stays on this path by design (no coherent before-commit
    // point -- arming known only post-bundle).
```

- [ ] **Step 2: Run, verify failure.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib sstep_ 2>&1`
Expected: compile error / FAIL — `has_sync_sstep_pc_trap_at`, `consume_sync_sstep_pc_trap` do not exist; rewritten `sstep_event_via_pc_event_path` cannot compile.

- [ ] **Step 3: Add the before-commit query + consume.**

In `src/device/core_debug/mod.rs`, immediately after `consume_sync_pc_trap` (ends line 746), add:

```rust
/// True iff an *event-driven* single-step (Debug_Control1[14:8]
/// SSTEP_EVENT) is wired to a *point* PC event (Core_PC_0..3) that is
/// VALID and matches `pc`, and this PC has not already been consumed.
/// This is the before-commit-eligible single-step case (§5.1 principled
/// split, Maya 2026-05-19): the arming condition (PC match) is known
/// *before* the bundle, so silicon halts before the bundle commits — the
/// same boundary as the G1 PC_Event_Halt seam. Watchpoint/mem/lock and
/// PC-*range*-wired SSTEP_EVENT have no coherent before-commit point and
/// stay after-commit via the unchanged check_event_halt ->
/// pending_single_step -> consume_pending_single_step path (documented
/// modeling decision).
pub fn has_sync_sstep_pc_trap_at(&self, pc: u32) -> bool {
    let pc14 = pc & PC_EVENT_ADDRESS_MASK;
    if self.sync_trap_consumed_at == Some(pc14) {
        return false;
    }
    let raw = match self.debug_sstep_event() {
        EVENT_CORE_PC_0 => self.pc_event0,
        EVENT_CORE_PC_1 => self.pc_event1,
        EVENT_CORE_PC_2 => self.pc_event2,
        EVENT_CORE_PC_3 => self.pc_event3,
        _ => return false,
    };
    Self::pc_event_address(raw).map_or(false, |addr| addr == pc14)
}

/// Consume a before-commit PC-wired single-step trap: latch the PC-event
/// halt cause (Debug_Status has no dedicated single-step cause bit —
/// aggregate only; a PC-wired single-step *is* a PC event firing), mark
/// this PC consumed so it does not re-fire after resume (mirrors
/// `consume_sync_pc_trap`; re-arming is the §8-tracked edge), and request
/// the halt.
pub fn consume_sync_sstep_pc_trap(&mut self, pc: u32) {
    self.sync_trap_consumed_at = Some(pc & PC_EVENT_ADDRESS_MASK);
    self.halt_cause_pc_event = true;
    self.request_halt();
}
```

(`debug_sstep_event()` is a private method on the same `impl`; `pub fn` here calls it fine. `EVENT_CORE_PC_0..3` and `EVENT_CORE_PC_RANGE_0_1` are module constants already in scope; `tests.rs` references them via the same path the existing `EVENT_CORE_PC_0` use at line 1047 does.)

- [ ] **Step 4: Extend the coordinator pre-execute seam.**

In `src/interpreter/engine/coordinator.rs`, in the Unit-1 seam block (lines ~624-674), immediately after the `has_sync_pc_trap_at` `if { ... continue; }` block, add the sibling check:

```rust
            // §5.1 principled split (Maya 2026-05-19): PC-wired event
            // single-step also halts before-commit (arming = PC match,
            // known pre-bundle). Same seam, same skip-the-bundle semantics
            // as the G1 PC_Event_Halt path above.
            if tile.core_debug.has_sync_sstep_pc_trap_at(next_pc) {
                tile.core_debug.consume_sync_sstep_pc_trap(next_pc);
                all_halted = false;
                any_running = true;
                continue;
            }
```

(Use the exact `all_halted`/`any_running`/`continue` trio as written in the existing `has_sync_pc_trap_at` block — copy it verbatim from that block so the loop bookkeeping is identical.)

- [ ] **Step 5: Run the targeted tests, verify pass.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib sstep_ 2>&1`
Expected: `sstep_event_via_pc_event_path`, `sstep_pc_trap_only_for_point_pc_events`, `sstep_pc_trap_requires_valid_matching_pc_event` PASS; `sstep_event_match_sets_pending_latch` and the other existing `sstep_event_*` tests (consume/idle/unrelated/zero-id/resume/reset/coexists) still PASS unchanged.

- [ ] **Step 6: Full suite + build.**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1` — expected: fully green (xdna-emu). The Unit-2 `count_step_*` tests still pass; existing async/PC-event/Unit-1 tests untouched.
Then: `cargo build 2>&1` — expected: clean.

- [ ] **Step 7: Commit.**

```bash
cd /home/triple/npu-work/xdna-emu
git add src/device/core_debug/mod.rs src/device/core_debug/tests.rs src/interpreter/engine/coordinator.rs
git commit -m "debug-halt Phase B Unit 3: single-step halt boundary -- principled split

PC-wired event single-step (Debug_Control1[14:8] SSTEP_EVENT == Core_PC_0..3,
matching PC) halts BEFORE the bundle commits via the existing Unit-1
pre-execute seam (has_sync_sstep_pc_trap_at / consume_sync_sstep_pc_trap):
the arming condition is a PC match, known pre-bundle. Watchpoint/mem/lock and
PC-range-wired SSTEP_EVENT keep the unchanged after-commit
check_event_halt->pending_single_step path -- documented modeling decision (no
coherent before-commit point). sstep_event_via_pc_event_path rewritten to the
before-commit query; the bare-event latch test clarified. Principled split
locked with Maya 2026-05-19.

Generated using Claude Code."
```

---

## Task 4: Finalize — coverage `Full`, findings coherence, regroup checkpoint

**Goal:** Flip the `debug_halt` coverage verdict to `Full`, record the closure in the findings doc, leave Phase B conclusively closed, stop at the regroup.

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/units.rs`
- Modify: `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Rewrite the `debug_halt` coverage entry.**

In `crates/xdna-archspec/src/coverage/units.rs` (the `debug_halt` entry, lines ~156-159), replace the narrative string with:

```
"Halt + status bits, synchronous PC-event breakpoints (before-commit pre-execute seam, G1 silicon-derived 2026-05-18), async halt paths, and debug-register read/write routing all modeled. Count-step (Debug_Control0[5:2] Single_Step_Count) is a live N-committed-bundle budget that halts before the (N+1)th bundle commits (G2 silicon-derived 2026-05-19). Event-driven single-step boundary is the principled split: PC-wired (SSTEP_EVENT==Core_PC_0..3) halts before-commit via the same seam; watchpoint/mem/lock/range-wired stays after-commit (documented modeling decision -- no coherent before-commit point). Open (tracked, spec section 8): count-step finer silicon characterization (decrement cadence / larger-N / 0x11-on-silicon -- only N=4 observed)."
```

Change the completeness marker from
`Modeled { completeness: Partial { missing: "single-step / count-step PC trap (G2/section 5.2)".into() } }`
to:
`Modeled { completeness: Full }`

- [ ] **Step 2: Regenerate coverage artifacts, zero drift.**

Run the spec §6-prescribed regeneration: `cargo run -p xdna-archspec --example gen_coverage_artifacts 2>&1`.
- If that example exists: it rewrites the generated artifact(s); `git diff --stat` must show only the intended `debug_halt` narrative/marker change (zero unrelated drift). Stage the regenerated artifact(s).
- If it does not exist (coverage is build-time-validated via `crates/xdna-archspec/build.rs` `enforce_coverage`/`build_gate`): run `cargo build -p xdna-archspec 2>&1` and confirm the build gate passes with the new `Full` marker (a failed gate prints the offending unit). Record which path applied in the commit message.

Either way: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-archspec 2>&1` green, and `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1` green (xdna-emu + xdna-archspec, full).

- [ ] **Step 3: Add the closing note to the findings doc.**

In `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`, in the "Phase B inputs (closing)" section, update bullet (b) and (c) to record implementation completion: append to the doc a short final paragraph under that section:

> **Phase B remainder — IMPLEMENTED (2026-05-19).** Unit 2 shipped the §5.2
> count-step state machine (`Debug_Control0[5:2]` arms a live
> N-committed-bundle budget; `tick_count_step` decrements per committed
> bundle; expiry latches `halted` so the `is_halted` gate blocks bundle
> N+1 — before-commit, G2-anchored). Unit 3 shipped the §5.1 single-step
> halt boundary as the principled split (PC-wired SSTEP_EVENT →
> before-commit via the Unit-1 seam; watchpoint/range → documented
> after-commit). `debug_halt` coverage is now `Modeled { Full }`. The §8
> count-step finer-characterization forward-commitment remains open
> (decrement cadence / larger-N / `0x11`-on-silicon — only `N=4`
> observed); the natural reading is shipped and documented inline.

- [ ] **Step 4: Commit.**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/units.rs docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
# include any regenerated coverage artifact from Step 2 if the example exists
git commit -m "debug-halt: Phase B complete -- coverage Full, findings closed

Units 2 (count-step state machine) + 3 (single-step principled split)
shipped; debug_halt -> Modeled { completeness: Full }. Coverage artifacts
regenerated zero-drift / build gate green. Findings doc records Phase B
remainder implemented; the section 8 count-step finer-characterization
forward-commitment stays open (N=4 only observed -- natural reading shipped).

Generated using Claude Code."
```

- [ ] **Step 5: STOP at the regroup.**

Phase B is conclusively closed. Do **not** start the §8 forward-commitments. Surface to Maya: Phase B complete (Units 1/1b/2/3, coverage `Full`); the §8 landscape — the two actionable-now items (`OUTBUF_ADDR` robustness; `Core_Status` RESET-bit `enable_core()`/`write_control()` reconciliation) vs. the genuinely HW-budget-gated trackers (count-step finer characterization; resume HW-verification) — for her to direct which to pick up next, per the plan→execute→regroup→next-plan rhythm.

---

## Self-review

**Spec coverage (spec §5.1, §5.2, §5.3, §6, §8):**
- §5.1 single-step halt boundary (principled split) → Task 1 Step 1 (spec coherence), Task 3 (implementation: `has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap` + seam extension), Task 3 Step 1 (test rewrite + watchpoint-after-commit clarification). Covered.
- §5.2 count-step state machine + the three locked modeling decisions → Task 1 Step 2 (spec coherence), Task 2 (state, arm, `tick_count_step`, coordinator wiring, tests). The "guarding before-commit assertion" of spec §6: resolved as the spec-§6 option (c) pure state-machine assertion (`count_step_decrements_then_halts_before_n_plus_1`) — explicitly justified because the literal store-did-not-land claim is HW-probe-covered (Exp2 `LANDED:0`) and the is_halted gate that produces before-commit-of-N+1 is unchanged Unit-1 territory. Surfaced here, not silently downgraded.
- §5.3 component boundaries → Task 1 Step 3; loci match Tasks 2/3 exactly (`count_step_remaining`/`halt_cause_count_step`, `write_debug_control0`, `tick_count_step`, `has_sync_sstep_pc_trap_at`/`consume_sync_sstep_pc_trap`, coordinator adjacency, tests, `units.rs`).
- §6 testing + completeness `Full` → Task 4 (units.rs marker, artifact regen zero-drift, full `cargo test --lib`).
- §8 count-step forward-commitment stays open → Task 1 Step 4 + Task 4 Step 3 (explicitly recorded as still-open, not closed).

**Placeholder scan:** No "TBD/TODO". The only conditional is Task 4 Step 2 (regen example present vs. build-time gate) — both branches are concrete commands with concrete expected outcomes and a "record which applied" instruction, because the spec (§6) and the codebase audit disagree on the mechanism's name; this is a real ambiguity resolved by trying the prescribed command and falling back to the verified build gate, not a placeholder.

**Type / identifier consistency:** `count_step_remaining: Option<u32>`, `halt_cause_count_step: bool`, `tick_count_step(&mut self) -> bool`, `has_sync_sstep_pc_trap_at(&self, u32) -> bool`, `consume_sync_sstep_pc_trap(&mut self, u32)` are spelled identically across Tasks 1-4, the struct, Default, the methods, the tests, and the coordinator calls. Constants (`DBG_CTRL0_SSTEP_COUNT_MASK/LSB`, `EVENT_CORE_PC_0..3`, `EVENT_CORE_PC_RANGE_0_1`, `PC_EVENT_ADDRESS_MASK`) match mod.rs definitions verbatim. `make_dbg_ctrl1(resume, sstep, event0, event1)` / `make_pc_event(valid, addr)` argument order matches tests.rs:625/780. `0x10`=N4, `0x08`=N2, `0x04`=N1, `0x11`=halt|N4 — arithmetic checked against `DBG_CTRL0_SSTEP_COUNT_LSB=2`.
