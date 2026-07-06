# Firmware Interrupt + Mailbox Event Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver faithful level-1 Xtensa interrupts (plus a mailbox doorbell as an interrupt source) so the real XDNA firmware's own handler runs, tasks complete, the scheduler unwinds to its `WAITI` idle loop, and a host-injected mailbox command round-trips end-to-end.

**Architecture:** A level-1 interrupt IS a general exception with `EXCCAUSE=4`, so delivery reuses the existing, silicon-validated `raise_general_exception` path (which already routes to the real handler at absolute `0x2958`, sets EXCM, saves EPC1). We add: the four interrupt SRs (INTENABLE/INTERRUPT/INTSET/INTCLEAR), an `rfe` return op, an interrupt-deliverability check between instructions, a `WAITI`-retires-then-halts model, and a minimal mailbox doorbell that sets the pending bit. Phase 0 (RE) pins the observed wiring the firmware-gated phases consume.

**Tech Stack:** Rust; the in-tree Xtensa interpreter under `src/firmware/xtensa/`; firmware-gated integration tests that skip without the local `npu.dev.sbin` binary.

## Global Constraints

- **DERIVE FROM THE TOOLCHAIN.** Interrupt mechanism is derived from the Xtensa ISA / QEMU `target/xtensa` (`handle_interrupt`, `translate_rfe`, `HELPER(waiti)`). Never hardcode what can be extracted; comment the hardware-fact source, not tool internals.
- **Level-1 only.** No high-priority interrupts (`EPS[n]`/`EPC[n]`/`rfi n`), no timer/CCOUNT, no nested/re-entrant level-1 (EXCM masks it during the handler).
- **No synthetic completion.** Never poke a task done-flag directly; task completion must emerge from the firmware's own handler (or a modeled DMA write), so timings emerge from real execution.
- **`cargo test --lib` stays green after every task.** Firmware binary: `/home/triple/npu-work/xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin` (not in repo; firmware-gated tests skip when absent).
- **No emoji. Commit messages end with:** `Generated using Claude Code.`
- Spec: `docs/superpowers/specs/2026-07-06-firmware-interrupt-mailbox-delivery-design.md`.

---

## File Structure

- `src/firmware/xtensa/interp/mod.rs` — Cpu interrupt state (`interrupt`/`intenable`/`halted` fields), SR routing (`read_sr`/`write_sr`), the `EXCCAUSE_LEVEL1_INTERRUPT` const + SR-number consts, `interrupt_deliverable`, delivery integration at the top of `step()`.
- `src/firmware/xtensa/decode/mod.rs` — `Op::Rfe` variant + its `max_ar` arm (`None`).
- `src/firmware/xtensa/decode/control.rs` — `rfe` decode (`decode_rrr`).
- `src/firmware/xtensa/interp/control.rs` — `rfe` exec + the `WAITI`-retire/halt change; update the two existing `waiti` tests.
- `src/firmware/mod.rs` — `boot_to_idle` idle-detection change (key on `Step::Wait`), the Phase 2/3/4 harness + inject API.
- `src/firmware/mmio.rs` — mailbox doorbell side-effect (Phase 3).
- `docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md` — Phase 0 output (created in Task 0).

---

## Task 0: Phase 0 — Observe the interrupt wiring (RE)

Reverse-engineering, not TDD. Deliverable is a committed findings doc that answers every question below with evidence, making Tasks 5/6/7 concrete. Firmware-gated (needs the local binary + the existing boot harness).

**Files:**
- Create: `docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md`

**Interfaces:**
- Produces (consumed by Tasks 5/6/7, referenced by findings-doc field name):
  - `INTENABLE_BITS` — the bitmask the firmware writes to INTENABLE during init.
  - `DOORBELL_INT_BIT` — which INTERRUPT bit the mailbox doorbell drives (assume a single dedicated bit if unobservable).
  - `DOORBELL_TRIGGER` — edge or level (assume edge if unobservable).
  - `IDLE_WAITI_LEVEL` — the imm on the idle-loop `waiti` (expected 0).
  - `HANDLER_CAUSE4_ARM` — whether `0x2958`'s `EXCCAUSE==4` arm reaches real interrupt servicing post-init (yes/no + evidence).
  - `X2I_ADDR` / `I2X_ADDR` — mailbox ring base addresses.
  - `AWAITED_EVENT` — what the init-time stuck task awaits (self-generated vs first host command).
  - `COMPLETION_WRITER` — who writes `sub[0x30]`: CPU handler (shape (i)) or DMA/peripheral (shape (ii)).

- [ ] **Step 1: Add SR-write logging for interrupt setup**

Temporarily raise the `write_sr`/`read_sr` log level or add a targeted `eprintln!` for SR `0xE2/0xE3/0xE4` and `SR_VECBASE`, so a boot run prints every INTENABLE/INTERRUPT/INTCLEAR/vector write with its PC. (Revert before Task 1 — no permanent per-SR logging.) Also confirm the existing store-watch already covers `sub[0x30]` (`0x9070`/`0x10f40`).

- [ ] **Step 2: Run the boot probe and capture the trace**

Run the firmware boot via the existing trace-to-wall probe with `XDNA_FW_CALLS` and a store-watch on `sub[0x30]`, redirecting to a file (never pipe through tail/grep):

```bash
XDNA_FW_CALLS=1 XDNA_FW_STOREWATCH=0x9070,0x10f40 RUST_LOG=debug \
  cargo test --lib -- firmware::tests::m2c_boot_advances_into_c_runtime --nocapture \
  > /tmp/claude-1000/-home-triple-npu-work-xdna-emu/*/scratchpad/phase0-boot.log 2>&1
```

Read the log (Read tool, not cat). Expected: INTENABLE writes visible; the idle loop reaching `waiti` at `0xc8eb`.

- [ ] **Step 3: Answer each Phase-0 question with evidence, write the findings doc**

For each interface field above, record the observed value (or the assumed default + why it wasn't observable) with a PC / disassembly citation. Structure the doc: Problem recap, one section per field with evidence, and a "Phase 2 branch" section stating shape (i) vs (ii) for `COMPLETION_WRITER`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md
git commit -m "docs(#140): Phase 0 findings -- firmware interrupt wiring (iter18)

Generated using Claude Code."
```

---

## Task 1: Interrupt special registers (INTENABLE/INTERRUPT/INTSET/INTCLEAR)

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs` (Cpu fields; SR consts; `EXCCAUSE_LEVEL1_INTERRUPT`; `read_sr`/`write_sr` arms; `Cpu::new`)
- Test: `src/firmware/xtensa/interp/mod.rs` (unit test in the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces:
  - `pub interrupt: u32` and `pub intenable: u32` fields on `Cpu`.
  - `pub const EXCCAUSE_LEVEL1_INTERRUPT: u32 = 4;`
  - SR routing: read `0xE2` → `interrupt`, read `0xE4` → `intenable`; write `0xE2` (INTSET) → `interrupt |= v`, write `0xE3` (INTCLEAR) → `interrupt &= !v`, write `0xE4` (INTENABLE) → `intenable = v`.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/firmware/xtensa/interp/mod.rs`:

```rust
#[test]
fn interrupt_srs_route_intset_intclear_intenable() {
    // INTSET (0xE2 write) sets pending bits by OR; INTCLEAR (0xE3 write)
    // clears by AND-NOT; INTERRUPT (0xE2 read) returns pending; INTENABLE
    // (0xE4) is a plain read/write register. SR numbers per QEMU
    // target/xtensa: INTSET/INTERRUPT=226(0xE2), INTCLEAR=227(0xE3),
    // INTENABLE=228(0xE4).
    let mut cpu = Cpu::new(0);
    cpu.write_sr(0xE4, 0x0000_00F0); // INTENABLE = bits 4-7
    assert_eq!(cpu.read_sr(0xE4), 0x0000_00F0);
    cpu.write_sr(0xE2, 0b1010); // INTSET bits 1,3
    assert_eq!(cpu.read_sr(0xE2), 0b1010, "INTERRUPT read returns the set bits");
    cpu.write_sr(0xE2, 0b0100); // INTSET bit 2 -> OR, not overwrite
    assert_eq!(cpu.read_sr(0xE2), 0b1110);
    cpu.write_sr(0xE3, 0b0010); // INTCLEAR bit 1
    assert_eq!(cpu.read_sr(0xE2), 0b1100, "INTCLEAR clears only its bits");
}
```

`write_sr`/`read_sr` are private; the test is in the same module so it can call them.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib firmware::xtensa::interp::tests::interrupt_srs_route_intset_intclear_intenable`
Expected: FAIL — `0xE2/0xE3/0xE4` currently hit the log-and-drop arm; reads return 0.

- [ ] **Step 3: Add the fields, consts, and SR routing**

In `src/firmware/xtensa/interp/mod.rs`, near the other SR-number consts (around line 60-80) add:

```rust
/// INTERRUPT (read) / INTSET (write) special register (`cpu.h` sregs index
/// 226 = 0xE2). Reading returns pending interrupt bits; writing SETS bits
/// (OR). Verified against QEMU target/xtensa.
const SR_INTERRUPT: u8 = 0xE2;
/// INTCLEAR (write) special register (index 227 = 0xE3): writing CLEARS the
/// named pending interrupt bits (AND-NOT).
const SR_INTCLEAR: u8 = 0xE3;
/// INTENABLE special register (index 228 = 0xE4): per-bit interrupt enable
/// mask.
const SR_INTENABLE: u8 = 0xE4;
```

Near `EXCCAUSE_SYSCALL` (line 112) add:

```rust
/// EXCCAUSE value for a level-1 interrupt (`LEVEL1_INTERRUPT`, the 5th entry
/// / index 4 in QEMU's cause enum). A level-1 interrupt shares the general
/// user/kernel exception vector and dispatches on EXCCAUSE, so it is a
/// general exception with this cause -- delivery reuses
/// `raise_general_exception`.
pub const EXCCAUSE_LEVEL1_INTERRUPT: u32 = 4;
```

Add fields to the `Cpu` struct (after `threadptr`, before `fr` around line 275):

```rust
    /// INTERRUPT pending-interrupt bits (Xtensa INTERRUPT SR read / INTSET SR
    /// write). A modeled interrupt source (the mailbox doorbell) sets a bit;
    /// `INTCLEAR` clears it; the handler acks by clearing. Level-1 only.
    pub interrupt: u32,
    /// INTENABLE per-bit interrupt-enable mask (Xtensa INTENABLE SR). An
    /// interrupt is deliverable only if its bit is set here.
    pub intenable: u32,
```

Initialize both to 0 in `Cpu::new` (add `interrupt: 0, intenable: 0,`).

In `write_sr`'s match (line 462) add before the `_` arm:

```rust
            SR_INTERRUPT => self.interrupt |= value, // INTSET: set bits by OR
            SR_INTCLEAR => self.interrupt &= !value, // INTCLEAR: clear bits
            SR_INTENABLE => self.intenable = value,
```

In `read_sr`'s match (line 490) add before the `_` arm:

```rust
            SR_INTERRUPT => self.interrupt,
            SR_INTENABLE => self.intenable,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib firmware::xtensa::interp::tests::interrupt_srs_route_intset_intclear_intenable`
Expected: PASS

- [ ] **Step 5: Run the full lib suite, then commit**

Run: `cargo test --lib`
Expected: PASS (existing count + 1)

```bash
git add src/firmware/xtensa/interp/mod.rs
git commit -m "feat(#140): iter18 -- interrupt SRs (INTENABLE/INTERRUPT/INTSET/INTCLEAR) + LEVEL1 cause

Generated using Claude Code."
```

---

## Task 2: `rfe` decode + execute (return from level-1 interrupt/exception)

**Files:**
- Modify: `src/firmware/xtensa/decode/mod.rs` (`Op::Rfe` variant + `max_ar` arm)
- Modify: `src/firmware/xtensa/decode/control.rs` (`decode_rrr` arm + decode test)
- Modify: `src/firmware/xtensa/interp/control.rs` (exec arm + test)

**Interfaces:**
- Consumes: `Cpu::epc1`, `RegFile::clear_excm` (both exist).
- Produces: `Op::Rfe`; `control::exec` handles it as `clear_excm(); pc = epc1`.

- [ ] **Step 1: Write the failing decode test**

In `src/firmware/xtensa/decode/control.rs` `mod tests`, add:

```rust
#[test]
fn decodes_rfe() {
    // rfe: same RFEI encoding family as rfwo/rfwu (op1=0,op2=0,r=3,t=0),
    // with s=0 (rfwo is s=4, rfwu s=5). Bytes `00 30 00`. Return from a
    // level-1 interrupt / general exception: PS.EXCM<-0, PC<-EPC1.
    let d = decode(&[0x00, 0x30, 0x00], 0xc8f0);
    assert_eq!(d.len, 3);
    assert!(matches!(d.op, Op::Rfe), "got {:?}", d.op);
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test --lib firmware::xtensa::decode::control::tests::decodes_rfe`
Expected: FAIL — `00 30 00` currently decodes to `Op::Unknown` (r=3/t=0/s=0 is unclaimed).

- [ ] **Step 3: Add the `Op::Rfe` variant, `max_ar` arm, and decode arm**

In `src/firmware/xtensa/decode/mod.rs`, add the variant near `Rfwo`/`Rfwu` (line 222-227):

```rust
    /// `rfe`: Return From Exception (level-1). The interrupt/exception
    /// sibling of `rfwo`/`rfwu`: leaves exception mode (PS.EXCM<-0) and
    /// resumes at EPC1. Terminates the firmware's level-1 interrupt handler.
    /// Encoding: RFEI family, s=0 (`00 30 00`). Semantics per QEMU
    /// `translate_rfe`: `PS &= ~PS_EXCM; jump(EPC1)`.
    Rfe,
```

In `max_ar` (line 1303), add `Rfe` to the `None` arm:

```rust
            Entry { .. } | Retw | RetwN | Rfwo | Rfwu | Rfe => None,
```

In `src/firmware/xtensa/decode/control.rs` `decode_rrr` (after the rfwu arm, line 56):

```rust
        // rfe (RFEI group r==3, t==0, s==0): return from level-1
        // interrupt/exception. Same encoding family as rfwo(s=4)/rfwu(s=5).
        // Bytes `00 30 00`. QEMU translate_rfe: PS.EXCM<-0, PC<-EPC1.
        (0x0, 0x0) if r == 3 && t == 0 && s == 0 => Some(Op::Rfe),
```

- [ ] **Step 4: Run the decode test, verify it passes**

Run: `cargo test --lib firmware::xtensa::decode::control::tests::decodes_rfe`
Expected: PASS

- [ ] **Step 5: Write the failing exec test**

In `src/firmware/xtensa/interp/control.rs`, add a new test module at the end:

```rust
#[cfg(test)]
mod rfe_tests {
    use super::super::{mapped_cpu, Step};
    use super::super::super::regfile::PS_EXCM;
    use crate::firmware::mmio::Bus;

    #[test]
    fn rfe_clears_excm_and_resumes_at_epc1() {
        // rfe (`00 30 00`): leave exception mode and jump to EPC1 -- the
        // inverse of the EXCM-set entry raise_general_exception performs.
        let rom = vec![0x00, 0x30, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_excm();
        cpu.epc1 = 0xc8ee; // the instruction after the idle-loop waiti
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.ps & PS_EXCM, 0, "rfe leaves exception mode");
        assert_eq!(cpu.pc, 0xc8ee, "rfe resumes at EPC1");
    }
}
```

- [ ] **Step 6: Run it, verify it fails**

Run: `cargo test --lib firmware::xtensa::interp::control::rfe_tests`
Expected: FAIL — `Op::Rfe` decodes now but no exec arm handles it, so `step()` panics ("not handled by any category").

- [ ] **Step 7: Add the exec arm**

In `src/firmware/xtensa/interp/control.rs`, extend the `Op::Rfwo | Op::Rfwu` arm's neighborhood with a new arm (after line 129):

```rust
        Op::Rfe => {
            // Return from level-1 interrupt/exception (QEMU translate_rfe):
            // leave exception mode and resume at EPC1. Unlike rfwo/rfwu it
            // does NOT touch WINDOWSTART/WINDOWBASE -- a level-1 interrupt
            // shares the general exception vector, not a window vector, so no
            // window frame was spilled/filled to undo.
            cpu.regs.clear_excm();
            cpu.pc = cpu.epc1;
            Some(Step::Ran)
        }
```

- [ ] **Step 8: Run the exec test + full suite, verify pass**

Run: `cargo test --lib firmware::xtensa::interp::control::rfe_tests`
Expected: PASS
Run: `cargo test --lib`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/firmware/xtensa/decode/mod.rs src/firmware/xtensa/decode/control.rs src/firmware/xtensa/interp/control.rs
git commit -m "feat(#140): iter18 -- rfe decode + exec (return from level-1 interrupt)

Generated using Claude Code."
```

---

## Task 3: `WAITI` retires-then-halts + idle detection

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs` (`Cpu.halted` field + `Cpu::new`; `step()` halt short-circuit)
- Modify: `src/firmware/xtensa/interp/control.rs` (`Waiti` exec + update the two existing `waiti` tests)
- Modify: `src/firmware/mod.rs` (`boot_to_idle` idle detection)

**Interfaces:**
- Produces: `pub halted: bool` on `Cpu`; `Waiti` advances PC, sets `halted`, returns `Step::Wait(Waiti)`; `step()` returns `Step::Wait(Waiti)` while halted with nothing deliverable; `boot_to_idle` sets `reached_idle` on any `Step::Wait`.
- Consumed by Task 4 (delivery clears `halted`).

- [ ] **Step 1: Write the failing test**

In `src/firmware/xtensa/interp/control.rs` `mod waiti_tests`, add:

```rust
#[test]
fn waiti_advances_pc_and_halts_then_re_waits() {
    // New model: waiti RETIRES (advances PC past itself) and halts. With no
    // deliverable interrupt, re-stepping stays halted and keeps returning
    // Wait, with PC parked AFTER the waiti (so a later interrupt's EPC1
    // points at the next instruction, not back onto the waiti).
    let rom = vec![0x00, 0x70, 0x00]; // waiti 0 @ pc 0
    let mut bus = Bus::new(rom);
    let mut cpu = mapped_cpu(0);
    match cpu.step(&mut bus) {
        Step::Wait(reason) => assert_eq!(reason, WaitReason::Waiti),
        other => panic!("expected Wait(Waiti), got {:?}", other),
    }
    assert_eq!(cpu.pc, 3, "waiti advances PC past itself (retires)");
    assert!(cpu.halted, "waiti halts the CPU");
    // Re-step: still halted, nothing pending -> Wait again, PC unchanged.
    assert!(matches!(cpu.step(&mut bus), Step::Wait(WaitReason::Waiti)));
    assert_eq!(cpu.pc, 3);
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test --lib firmware::xtensa::interp::control::waiti_tests::waiti_advances_pc_and_halts_then_re_waits`
Expected: FAIL — current `waiti` leaves `pc==0` and there is no `halted` field.

- [ ] **Step 3: Add the `halted` field**

In `src/firmware/xtensa/interp/mod.rs`, add to `Cpu` (after `fr`):

```rust
    /// True when a `waiti` has retired and the CPU is idling until a
    /// deliverable interrupt arrives (Xtensa "waiti" halt state, QEMU
    /// `env->halted`). Cleared when an interrupt is delivered.
    pub halted: bool,
```

Initialize `halted: false` in `Cpu::new`.

- [ ] **Step 4: Rewrite the `Waiti` exec arm**

In `src/firmware/xtensa/interp/control.rs`, replace the `Op::Waiti { imm }` arm (lines 170-181) with:

```rust
        Op::Waiti { imm } => {
            // Faithful Xtensa waiti (QEMU HELPER(waiti)): set PS.INTLEVEL,
            // RETIRE (advance PC past the instruction), and halt until a
            // deliverable interrupt arrives. Advancing PC is load-bearing:
            // when the interrupt is later taken, EPC1 captures the
            // instruction AFTER waiti (the idle loop's `j loop`), so `rfe`
            // resumes the loop and re-dispatches -- rather than returning
            // onto the waiti and re-sleeping forever.
            cpu.regs.set_intlevel(*imm);
            cpu.pc = pc.wrapping_add(len as u32);
            cpu.halted = true;
            Some(Step::Wait(WaitReason::Waiti))
        }
```

- [ ] **Step 5: Add the halt short-circuit in `step()`**

In `src/firmware/xtensa/interp/mod.rs` `step()` (line 589), insert at the very top, before the fastpath block:

```rust
        // A halted CPU (post-waiti) runs no instructions; it only yields Wait
        // until Task 4's interrupt delivery unhalts it. (Delivery is checked
        // ahead of this in Task 4; until then a halted CPU stays halted.)
        if self.halted {
            return Step::Wait(WaitReason::Waiti);
        }
```

- [ ] **Step 6: Update the two existing `waiti` tests**

In `src/firmware/xtensa/interp/control.rs`, `waiti_sets_intlevel_and_yields_without_advancing_pc` and `waiti_nonzero_level_is_recorded` assert `cpu.pc == 0`. Change both `assert_eq!(cpu.pc, 0, ...)` lines to `assert_eq!(cpu.pc, 3, "waiti now retires -- advances PC past itself");` and update the first test's doc comment ("pc must NOT advance ...") to describe the retire-then-halt model. Rename `waiti_sets_intlevel_and_yields_without_advancing_pc` to `waiti_sets_intlevel_and_yields_after_advancing_pc`.

- [ ] **Step 7: Update `boot_to_idle` idle detection**

In `src/firmware/mod.rs` `boot_to_idle` (line 225-232), replace the `Step::Wait(reason)` arm's PC-stability check:

```rust
                Step::Wait(reason) => {
                    // Interrupt delivery is checked ahead of execution
                    // (Task 4), so a returned Wait means nothing was
                    // deliverable -- the CPU is genuinely idle in its
                    // command-loop waiti. (With waiti now retiring, keying on
                    // PC-stability would miss the first idle step.)
                    reached_idle = true;
                    wait_reason = Some(reason);
                    break;
                }
```

- [ ] **Step 8: Run the tests + full suite**

Run: `cargo test --lib firmware::xtensa::interp::control::waiti_tests`
Expected: PASS (all three)
Run: `cargo test --lib`
Expected: PASS. If a firmware-gated boot test's idle assertion shifts by one instruction, that is the intended retire change — verify it still reports `reached_idle` where it did before.

- [ ] **Step 9: Commit**

```bash
git add src/firmware/xtensa/interp/mod.rs src/firmware/xtensa/interp/control.rs src/firmware/mod.rs
git commit -m "feat(#140): iter18 -- waiti retires-then-halts; boot_to_idle keys on Wait

Generated using Claude Code."
```

---

## Task 4: Level-1 interrupt delivery (the coupled core)

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs` (`interrupt_deliverable` + delivery at top of `step()`)
- Test: `src/firmware/xtensa/interp/mod.rs` (unit tests, no firmware)

**Interfaces:**
- Consumes: `interrupt`/`intenable` (Task 1), `EXCCAUSE_LEVEL1_INTERRUPT` (Task 1), `raise_general_exception` (exists), `halted` (Task 3), `Op::Rfe` (Task 2).
- Produces: interrupts delivered between instructions when `intlevel()==0 && !excm() && (interrupt & intenable)!=0`, routed via `raise_general_exception(pc, 4)`; delivery clears `halted`.

- [ ] **Step 1: Write the failing end-to-end test**

In `src/firmware/xtensa/interp/mod.rs` `mod tests`, add (the delivery routes to the absolute handler `GENERAL_EXCEPTION_HANDLER = 0x2958`, so map that page and stage an `rfe` there):

```rust
#[test]
fn level1_interrupt_delivers_runs_handler_rfe_resumes() {
    use super::GENERAL_EXCEPTION_HANDLER; // 0x2958
    // A pending+enabled bit with INTLEVEL==0, EXCM==0 is deliverable. The
    // firmware body is a single nop (`f0 20 00`) at 0x100; the "handler" is
    // an rfe (`00 30 00`) staged at the real handler address. Delivery:
    // EPC1<-PC(0x100), EXCCAUSE=4, EXCM set, PC<-0x2958. Then rfe resumes at
    // EPC1 with EXCM clear.
    let mut rom = vec![0u8; 0x295b];
    rom[0x100..0x103].copy_from_slice(&[0xf0, 0x20, 0x00]); // nop
    rom[0x2958..0x295b].copy_from_slice(&[0x00, 0x30, 0x00]); // rfe
    let mut bus = Bus::new(rom);
    let mut cpu = mapped_cpu(0x100);
    // Map the handler's page (0x2000) -- different 4KB page than 0x100.
    cpu.mmu.write_tlb(false, 0x2000 | 0x1, 0x2000 | 1);
    cpu.intenable = 0b10;
    cpu.interrupt = 0b10; // pending + enabled

    // Delivery happens BEFORE the nop executes.
    match cpu.step(&mut bus) {
        Step::Exception { cause, pc } => {
            assert_eq!(cause, super::EXCCAUSE_LEVEL1_INTERRUPT);
            assert_eq!(pc, GENERAL_EXCEPTION_HANDLER);
        }
        other => panic!("expected interrupt delivery, got {:?}", other),
    }
    assert_eq!(cpu.epc1, 0x100, "EPC1 = the instruction the interrupt preempted");
    assert!(cpu.regs.excm(), "handler runs with EXCM set");

    // Handler acks its source (INTCLEAR) then returns via rfe.
    cpu.interrupt = 0;
    assert!(matches!(cpu.step(&mut bus), Step::Ran)); // rfe
    assert!(!cpu.regs.excm(), "rfe cleared EXCM");
    assert_eq!(cpu.pc, 0x100, "rfe resumed at EPC1 -- the preempted instruction");

    // With the source cleared, the preempted nop now runs normally.
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.pc, 0x103);
}

#[test]
fn interrupt_masked_by_intlevel_excm_and_disable() {
    // Not deliverable when: INTLEVEL!=0 (waiti raised it), or EXCM set
    // (already in a handler), or the enable bit is clear.
    let mut cpu = Cpu::new(0);
    cpu.interrupt = 0b1;
    cpu.intenable = 0b0; // disabled
    assert!(!cpu.interrupt_deliverable());
    cpu.intenable = 0b1;
    assert!(cpu.interrupt_deliverable());
    cpu.regs.set_intlevel(1); // level-1 masked
    assert!(!cpu.interrupt_deliverable());
    cpu.regs.set_intlevel(0);
    cpu.regs.set_excm(); // in a handler
    assert!(!cpu.interrupt_deliverable());
}

#[test]
fn pending_interrupt_wakes_a_halted_waiti() {
    // A halted CPU (post-waiti 0) with a newly-pending+enabled bit delivers
    // on the next step instead of re-waiting.
    let mut rom = vec![0u8; 0x295b];
    rom[0x2958..0x295b].copy_from_slice(&[0x00, 0x30, 0x00]); // rfe
    let mut bus = Bus::new(rom);
    let mut cpu = mapped_cpu(0);
    cpu.mmu.write_tlb(false, 0x2000 | 0x1, 0x2000 | 1);
    cpu.halted = true;
    cpu.pc = 0xc8ee; // parked after the idle-loop waiti
    cpu.intenable = 0b1;
    cpu.interrupt = 0b1;
    match cpu.step(&mut bus) {
        Step::Exception { cause, .. } => assert_eq!(cause, super::EXCCAUSE_LEVEL1_INTERRUPT),
        other => panic!("expected delivery to wake the halt, got {:?}", other),
    }
    assert!(!cpu.halted, "delivery unhalts the CPU");
    assert_eq!(cpu.epc1, 0xc8ee, "EPC1 = the post-waiti instruction");
}
```

- [ ] **Step 2: Run them, verify they fail**

Run: `cargo test --lib firmware::xtensa::interp::tests::level1_interrupt`
Expected: FAIL — no `interrupt_deliverable`; delivery not wired; the halt short-circuit returns Wait instead of delivering.

- [ ] **Step 3: Add `interrupt_deliverable` and wire delivery**

In `src/firmware/xtensa/interp/mod.rs`, add a method in the `impl Cpu` block near `window_check`:

```rust
    /// True when a level-1 interrupt may be delivered right now: a pending bit
    /// is enabled (`INTERRUPT & INTENABLE`), the current level is 0 (a nonzero
    /// PS.INTLEVEL -- e.g. a `waiti 1+` -- masks level-1), and PS.EXCM is
    /// clear (we are not already inside a handler). The `intlevel()==0` gate
    /// is the level-1 specialization of "interrupt level > PS.INTLEVEL"; this
    /// interpreter models level-1 only. Derived from QEMU
    /// `xtensa_cpu_has_work` / `handle_interrupt`.
    pub fn interrupt_deliverable(&self) -> bool {
        self.regs.intlevel() == 0
            && !self.regs.excm()
            && (self.interrupt & self.intenable) != 0
    }
```

At the very top of `step()`, BEFORE the `self.halted` short-circuit added in Task 3 (delivery must win over re-waiting):

```rust
        // Interrupts are checked between instructions (faithful Xtensa). A
        // deliverable level-1 interrupt IS a general exception with
        // EXCCAUSE=4: reuse the proven raise_general_exception path, which
        // sets EPC1<-PC, EXCM, and routes to the real handler at 0x2958.
        // EXCM is clear here (interrupt_deliverable gates on it), so this
        // never mis-routes to the double vector.
        if self.interrupt_deliverable() {
            self.halted = false;
            return self.raise_general_exception(self.pc, EXCCAUSE_LEVEL1_INTERRUPT);
        }
```

Order in `step()`: (1) delivery check, (2) `halted` short-circuit (Task 3), (3) fastpath, (4) fetch/decode/window_check/exec.

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cargo test --lib firmware::xtensa::interp::tests::level1_interrupt firmware::xtensa::interp::tests::interrupt_masked firmware::xtensa::interp::tests::pending_interrupt_wakes`
Expected: PASS

- [ ] **Step 5: Full suite + commit**

Run: `cargo test --lib`
Expected: PASS

```bash
git add src/firmware/xtensa/interp/mod.rs
git commit -m "feat(#140): iter18 -- level-1 interrupt delivery via raise_general_exception(cause=4)

Generated using Claude Code."
```

---

## Task 5: Phase 2 — Unwind to idle on real firmware (firmware-gated)

**Files:**
- Modify: `src/firmware/mod.rs` (a firmware-gated integration test that sets INTENABLE + delivers the Phase-0 event, asserts `reached_idle`)

**Interfaces:**
- Consumes: the Task 0 findings-doc fields `INTENABLE_BITS`, `DOORBELL_INT_BIT`, `AWAITED_EVENT`, `COMPLETION_WRITER`, `IDLE_WAITI_LEVEL`; the Task 4 delivery mechanism.

**Branch on `COMPLETION_WRITER` from the findings doc:**
- **Shape (i) — CPU handler writes `sub[0x30]`:** delivering the interrupt is sufficient; the firmware's handler runs and stores the flag itself.
- **Shape (ii) — DMA writes `sub[0x30]`:** additionally model a minimal DMA-completion write that sets the flag, with the interrupt signaling it. Add that write to the harness at the point Phase 0 identified; keep it derived from the observed descriptor, not a synthetic poke.

- [ ] **Step 1: Write the failing firmware-gated test**

In `src/firmware/mod.rs` (near the existing `m2c_boot_advances_into_c_runtime`), add — using the same firmware-load/skip guard the existing boot tests use (copy that guard verbatim from the neighbouring test):

```rust
#[test]
fn m2c_boot_unwinds_to_waiti_idle() {
    // [firmware-load guard copied from m2c_boot_advances_into_c_runtime --
    //  returns early if npu.dev.sbin is absent]
    let mut proc = /* load as the existing boot test does */;

    // Enable the interrupt(s) the firmware arms during init and raise the
    // doorbell bit, per Phase 0 findings
    // (docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md:
    //  INTENABLE_BITS, DOORBELL_INT_BIT). Boot to the idle loop.
    proc.cpu.intenable = /* INTENABLE_BITS from findings */;
    proc.cpu.interrupt = 1 << /* DOORBELL_INT_BIT from findings */;
    // Shape (ii) only: also stage the DMA-completion write Phase 0 identified.

    let report = proc.boot_to_idle(500_000);
    assert!(report.reached_idle, "firmware unwinds to the waiti idle loop");
    assert_eq!(report.last_pc, /* 0xc8ee post-waiti, per Phase 0 */,
        "idle at the command-loop waiti");
}
```

Fill the `/* ... */` slots from the committed Phase 0 findings doc — they are concrete observed constants, not placeholders to invent.

- [ ] **Step 2: Run it, verify it fails (or is skipped without the binary)**

Run: `cargo test --lib firmware::tests::m2c_boot_unwinds_to_waiti_idle -- --nocapture`
Expected: with the binary present, FAIL initially if the event/flag path isn't complete (the recursion persists); without the binary, SKIP. Diagnose with `XDNA_FW_CALLS` if it walls — do NOT add a second fix on top; return to the findings doc.

- [ ] **Step 3: Complete the shape-(i)/(ii) path until the test passes**

Shape (i): confirm delivery reaches `0x2958`'s cause-4 arm (Phase 0 `HANDLER_CAUSE4_ARM`) and the handler stores the flag. Shape (ii): add the minimal DMA-completion write. Iterate against the findings, not by guessing.

- [ ] **Step 4: Run it, verify it passes; full suite**

Run: `cargo test --lib firmware::tests::m2c_boot_unwinds_to_waiti_idle`
Expected: PASS (with binary)
Run: `cargo test --lib`
Expected: PASS

- [ ] **Step 5: Write the iter18 Phase-2 finding + commit**

Document the unwind (which shape, the event, the handler path) in `docs/superpowers/findings/2026-07-06-iter18-phase2-unwind-to-idle.md`.

```bash
git add src/firmware/mod.rs docs/superpowers/findings/2026-07-06-iter18-phase2-unwind-to-idle.md
git commit -m "feat(#140): iter18 Phase 2 -- firmware unwinds to waiti idle via delivered interrupt

Generated using Claude Code."
```

---

## Task 6: Phase 3 — Mailbox doorbell + host-injection API

**Files:**
- Modify: `src/firmware/mmio.rs` (doorbell-write side-effect: set the mailbox INTERRUPT bit)
- Modify: `src/firmware/mod.rs` (`inject_mailbox_command` / `read_mailbox_response` on `FirmwareProcessor`)
- Test: both files (a hermetic mmio test + a firmware-gated API test)

**Interfaces:**
- Consumes: findings-doc `X2I_ADDR`, `I2X_ADDR`, `DOORBELL_INT_BIT`, the doorbell register address; `Cpu.interrupt`.
- Produces:
  - `FirmwareProcessor::inject_mailbox_command(&mut self, opcode: u32, payload: &[u8])` — writes the 16-byte wire header `{total_size, sz_ver, id, opcode}` (id magic `0x1D000000`) + payload into X2I, bumps the tail, sets `cpu.interrupt |= 1 << DOORBELL_INT_BIT`.
  - `FirmwareProcessor::read_mailbox_response(&self) -> Option<Vec<u8>>` — reads a completed I2X message, else `None`.

- [ ] **Step 1: Write the failing hermetic doorbell test**

In `src/firmware/mmio.rs` `mod tests`, add a test that a write to the doorbell register (address from findings) sets the expected `cpu.interrupt` bit. (The `0x27xxxxxx` block is already plain RAM per `mmio.rs:21`; this adds ONLY the doorbell side-effect, no ring abstraction.) Since the doorbell side-effect must reach `cpu.interrupt`, model it as: the store path recognizes the doorbell address and records a pending doorbell on the `Bus`, which `step()`/the processor folds into `cpu.interrupt`. Write the test to the interface you choose; keep it minimal.

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test --lib firmware::mmio::tests::doorbell`
Expected: FAIL

- [ ] **Step 3: Implement the doorbell side-effect + inject/read API**

Add the doorbell recognition in `mmio.rs` (single address compare; `// ponytail:` note that only the one doorbell reg is modeled). Add `inject_mailbox_command`/`read_mailbox_response` to `FirmwareProcessor` in `mod.rs`, writing/reading the wire header at `X2I_ADDR`/`I2X_ADDR` from the findings.

- [ ] **Step 4: Run the hermetic test + full suite**

Run: `cargo test --lib firmware::mmio::tests::doorbell`
Expected: PASS
Run: `cargo test --lib`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/firmware/mmio.rs src/firmware/mod.rs
git commit -m "feat(#140): iter18 Phase 3 -- mailbox doorbell side-effect + host-injection API

Generated using Claude Code."
```

---

## Task 7: Phase 4 — Host-command round-trip (firmware-gated)

**Files:**
- Modify: `src/firmware/mod.rs` (a firmware-gated round-trip test)

**Interfaces:**
- Consumes: Task 5 (boot to idle), Task 6 (inject/read API), findings-doc opcode for a simple handshake/version command.

- [ ] **Step 1: Write the failing round-trip test**

```rust
#[test]
fn m2c_mailbox_command_round_trips() {
    // [firmware-load guard]
    let mut proc = /* load + boot_to_idle to the waiti idle loop (Task 5) */;
    assert!(/* reached idle */);

    // Inject a simple handshake/version command; the doorbell sets the
    // INTERRUPT bit, delivery runs the firmware's own handler, which reads
    // X2I, dispatches, and writes an I2X response.
    proc.inject_mailbox_command(/* opcode from findings */, &[]);
    // Step until the firmware writes a response or returns to idle.
    for _ in 0..500_000 {
        let step = proc.cpu.step(&mut proc.bus);
        if proc.read_mailbox_response().is_some() { break; }
        if matches!(step, Step::Unknown { .. }) { break; }
    }
    let resp = proc.read_mailbox_response().expect("firmware wrote an I2X response");
    assert!(!resp.is_empty(), "response carries the handshake reply");
}
```

- [ ] **Step 2: Run it, verify it fails (or skips without the binary)**

Run: `cargo test --lib firmware::tests::m2c_mailbox_command_round_trips -- --nocapture`
Expected: FAIL/SKIP. Diagnose walls with `XDNA_FW_CALLS`; iterate against the real handler path, not synthetic pokes.

- [ ] **Step 3: Complete the round-trip until it passes**

- [ ] **Step 4: Full suite + finding + commit**

Run: `cargo test --lib`
Expected: PASS

Document the round-trip (opcode, handler path, emergent dispatch timing) in `docs/superpowers/findings/2026-07-06-iter18-phase4-round-trip.md`.

```bash
git add src/firmware/mod.rs docs/superpowers/findings/2026-07-06-iter18-phase4-round-trip.md
git commit -m "feat(#140): iter18 Phase 4 -- host mailbox command round-trips through real firmware

Generated using Claude Code."
```

---

## Self-Review Notes

- **Spec coverage:** Component 1 (interrupt SRs) → Task 1; Component 2 (delivery) → Task 4; Component 2b (`rfe`) → Task 2; Component 3 (`WAITI` retire + harness) → Task 3; Component 4 (doorbell) → Task 6; Component 5 (inject API) → Task 6; Component 6 (verification) → Tasks 5 & 7. Phase 0 → Task 0. Every spec phase maps to a task.
- **Firmware-gated data dependency:** Tasks 0/5/6/7 reference observed constants from the Task 0 findings doc by field name. These are a genuine RE data dependency the spec deliberately gates on Phase 0 — not invented placeholders. Phase 1 (Tasks 1-4) is fully concrete and needs no firmware.
- **Type consistency:** `interrupt`/`intenable`/`halted` (Cpu fields), `interrupt_deliverable()`, `EXCCAUSE_LEVEL1_INTERRUPT`, `Op::Rfe`, `inject_mailbox_command`/`read_mailbox_response` are named identically across all tasks that use them.
- **Coupling honored:** delivery (Task 4) lands after `rfe` (Task 2) and `WAITI`-halt (Task 3), since its end-to-end test exercises all three.
