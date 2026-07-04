//! Xtensa interpreter core: fetch/decode/execute over the `Bus` for the
//! non-windowed base-ISA subset plus the windowed-call ABI (M1.5).
//!
//! Windowed-ABI model (matches the Xtensa windowed-register option; the real
//! firmware uses only call8/callx8, so PS.CALLINC is always 2):
//! - `call8`/`callx8` stash the return address (with the call size in bits
//!   31:30) into a8 of the *caller's* window and set PS.CALLINC -- they do
//!   NOT rotate WINDOWBASE.
//! - `entry` rotates WINDOWBASE forward by PS.CALLINC, allocates the callee
//!   frame, and marks the new frame live in WINDOWSTART; the return address
//!   the caller left in a8 thereby becomes the callee's a0.
//! - `retw`/`retw.n` rotate WINDOWBASE back by the call size in `a0[31:30]`
//!   and return to `a0[29:0]`.
//! - Window overflow (raised by `entry` when the forward rotation would
//!   overwrite a still-live frame) and underflow (raised by `retw` when the
//!   frame being returned to was spilled) vector to a VECBASE-relative window
//!   handler. This firmware manages its windows with a software spill-all
//!   routine (rotw + s32i.n) rather than installing these architectural
//!   handlers, so the raise+vector path here is the faithful CPU mechanism,
//!   unit-proven below; whether an image installs a handler is decided at
//!   boot (M1.7).
//!
//! Split into per-category execute modules (`arith`/`mem`/`branch`/`control`/
//! `system`) as a scaffold for M2a's additional opcodes: `Cpu`, `Step`,
//! `step()`, the window-exception raise, and (M2a Task 9) the general
//! (non-window) exception raise stay here as shared machinery; `step()`'s
//! dispatch tries each category's `exec` in turn (`None` = not this
//! category's op), preserving the exact original per-op behavior.
//!
//! **General-exception raise (M2a Task 9).** A second, separate raise path
//! from the window-exception one above: `syscall` and integer
//! divide-by-zero (`quou`/`remu`/`rems`) both vector through
//! [`Cpu::raise_general_exception`] to the VECBASE-relative KernelException
//! vector (this firmware runs kernel-mode, PS.UM=0 -- see
//! [`KERNEL_EXCEPTION_VECTOR_OFFSET`] for the PS.UM selection we don't
//! model), recording a REAL architectural EXCCAUSE value (unlike the window
//! path's synthetic diagnostic cause IDs) that a handler would read to
//! dispatch on the specific cause.

mod arith;
mod branch;
mod control;
mod mem;
mod system;

use super::decode::{self, Op};
use super::regfile::{RegFile, NUM_FRAMES};
use crate::firmware::Bus;

// Special-register numbers (Xtensa SR encoding). Only the ones the firmware
// boot path touches, plus the windowed/exception SRs the M2 command loop will
// need, are modeled; every other SR is a logged no-op in `write_sr`/`read_sr`.
/// SAR (shift amount register).
const SR_SAR: u8 = 0x03;
/// WINDOWBASE.
const SR_WINDOWBASE: u8 = 0x48;
/// WINDOWSTART.
const SR_WINDOWSTART: u8 = 0x49;
/// EPC1 (exception program counter, level 1).
const SR_EPC1: u8 = 0xB1;
/// PS (processor state).
const SR_PS: u8 = 0xE6;
/// VECBASE (relocatable vector base).
const SR_VECBASE: u8 = 0xE7;
/// EXCCAUSE (exception cause register). Verified against `xtensa-modules.c`'s
/// `Opcode_rsr_exccause_Slot_inst_encode` template (`0x03e800` -> op1=3,op2=0,
/// sr=0xE8) -- M2a Task 9.
const SR_EXCCAUSE: u8 = 0xE8;

/// MMU config special registers (QEMU `cpu.h` sregs indices; PTEVADDR/ITLBCFG/
/// DTLBCFG cross-checked against decode/system.rs's existing decode tests).
const SR_PTEVADDR: u8 = 0x53;
const SR_RASID: u8 = 0x5A;
const SR_ITLBCFG: u8 = 0x5B;
const SR_DTLBCFG: u8 = 0x5C;

/// User-register (TIE) number `wur`/`rur` write/read into `cpu.vecbase`, via
/// [`Cpu::write_ur`]. NOT the same namespace as [`SR_VECBASE`] (special
/// registers and user registers are architecturally separate 8-bit spaces
/// that just happen to share this numeric value on this firmware) -- see
/// `decode::Op::Wur`'s doc for the full derivation, including the
/// discrepancy between AMD's vendored generic `xtensa-modules.c` (which
/// names UR 0xE7 "threadptr" in its stock reference core config) and the
/// real firmware's own Ghidra-derived naming ("VECBASE").
const UR_VECBASE: u8 = 0xE7;

/// Diagnostic cause code for a window-overflow exception. Xtensa vectors
/// window exceptions through dedicated VECBASE-relative addresses rather than
/// reporting them in EXCCAUSE, so this is our own identifier for the
/// [`Step::Exception`] `cause` channel, not an architectural EXCCAUSE value.
pub const CAUSE_WINDOW_OVERFLOW: u32 = 0x1000;
/// Diagnostic cause code for a window-underflow exception (see
/// [`CAUSE_WINDOW_OVERFLOW`]).
pub const CAUSE_WINDOW_UNDERFLOW: u32 = 0x1001;

/// EXCCAUSE value for `syscall`: an ARCHITECTURAL cause code (unlike the
/// window causes above, which are this interpreter's own diagnostic IDs).
/// Verified against QEMU `target/xtensa/cpu.h`'s exception-cause enum
/// (`ILLEGAL_INSTRUCTION_CAUSE = 0, SYSCALL_CAUSE, ...` -- SYSCALL_CAUSE is
/// the second entry, value 1), matching the brief's "EXCCAUSE = 1" directly.
pub const EXCCAUSE_SYSCALL: u32 = 1;
/// EXCCAUSE value for integer divide-by-zero (`quou`/`remu`/`rems` dividing
/// by zero -- see `interp::arith::exec`). Verified against QEMU
/// `target/xtensa/cpu.h`'s exception-cause enum: counting from
/// `ILLEGAL_INSTRUCTION_CAUSE = 0`, `INTEGER_DIVIDE_BY_ZERO_CAUSE` is the
/// 7th entry (index 6): `ILLEGAL_INSTRUCTION(0), SYSCALL(1),
/// INSTRUCTION_FETCH_ERROR(2), LOAD_STORE_ERROR(3), LEVEL1_INTERRUPT(4),
/// ALLOCA(5), INTEGER_DIVIDE_BY_ZERO(6)`.
pub const EXCCAUSE_INTEGER_DIVIDE_BY_ZERO: u32 = 6;

/// VECBASE-relative offset of the (non-window) general exception vector
/// this firmware uses -- where `syscall`, integer divide-by-zero, and every
/// other synchronous general exception vectors to (the handler then reads
/// EXCCAUSE to dispatch on the specific cause; unlike the window vectors,
/// there is only ONE such vector per privilege mode for the whole
/// general-exception class).
///
/// Xtensa actually has TWO general-exception vectors, and QEMU
/// `target/xtensa/exc_helper.c` selects between them by PS.UM at raise time:
/// `vector = (env->sregs[PS] & PS_UM) ? EXC_USER : EXC_KERNEL` -- i.e. the
/// KernelExceptionVector (`XCHAL_KERNEL_VECOFS`, VECBASE+0x300) when
/// PS.UM=0, the UserExceptionVector (`XCHAL_USER_VECOFS`, VECBASE+0x340)
/// when PS.UM=1. This bare-metal NPU management firmware runs entirely in
/// kernel mode (PS.UM=0 -- it never enters user mode), so we assume the
/// kernel vector unconditionally and deliberately DO NOT model PS.UM
/// selection (YAGNI: mgmt firmware never runs user-mode).
///
/// `0x300` verified as a STABLE architectural constant (not core-specific)
/// by cross-checking `XCHAL_KERNEL_VECOFS` across five independent QEMU
/// `target/xtensa/core-*/core-isa.h` configs (dc233c, de233_fpu,
/// sample_controller, de212, test_mmuhifi_c3) -- all agree on `0x300`
/// despite having entirely different `VECBASE` reset addresses, the same
/// cross-config-agreement technique already used to derive
/// `window_vector_offset`'s 0x00/0x40/.../0x140 table.
const KERNEL_EXCEPTION_VECTOR_OFFSET: u32 = 0x300;

/// VECBASE-relative offset of the window-exception vector for call size `k`
/// (1/2/3 quads = call4/8/12) and direction. Standard Xtensa window-vector
/// layout: Overflow4 +0x00, Underflow4 +0x40, Overflow8 +0x80, Underflow8
/// +0xC0, Overflow12 +0x100, Underflow12 +0x140.
///
/// `k` is the faulting call size: `a0[31:30]` for underflow (unambiguous --
/// the returning frame's own size), and PS.CALLINC for overflow. The overflow
/// vector is properly selected by the size of the *spilled* frame; this
/// firmware uses only call8, so every frame is 8-register and CALLINC coincides
/// with it. If mixed call sizes ever appear, re-derive the overflow selector
/// against hardware -- no oracle for it exists in the call8-only firmware.
fn window_vector_offset(overflow: bool, k: u32) -> u32 {
    let size_base = match k {
        1 => 0x00,
        2 => 0x80,
        3 => 0x100,
        // Only call4/8/12 (k=1/2/3) exist; clamp anything else to the k=1
        // vector rather than compute an out-of-table offset.
        _ => 0x00,
    };
    size_base + if overflow { 0x00 } else { 0x40 }
}

/// Why the interpreter yielded instead of running another instruction.
///
/// Not yet produced by this milestone -- no wait-capable op (`WAITI`, a
/// device poll) is implemented here -- but defined now per the M1 interface
/// so the co-sim scheduler seam (design spec section 6, M3) has a stable
/// type to match on once `WAITI` and `sysstub.rs` poll/spin-detection (M1.6)
/// land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitReason {
    /// Blocked on `WAITI` with the mailbox ring empty.
    MailboxEmpty,
    /// Blocked in a `WAITI` wait-for-interrupt instruction generally.
    Waiti,
    /// Spinning on repeated reads of `addr`, waiting for it to change (a
    /// device status poll; detected by `sysstub.rs`'s spin-detection, M1.6).
    PollSpin { addr: u32 },
}

/// The outcome of one `Cpu::step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// One instruction executed; `pc` advanced past it.
    Ran,
    /// The interpreter yielded without executing further; see `WaitReason`.
    Wait(WaitReason),
    /// An Xtensa exception was raised with cause code `cause`; `pc` is the
    /// exception vector the CPU jumped to (also now in `Cpu::pc`). The faulting
    /// instruction's restart address is saved in `Cpu::epc1`.
    Exception { cause: u32, pc: u32 },
    /// The instruction at `pc` isn't executed by this interpreter yet -- an
    /// opcode the decode module itself doesn't recognize (`Op::Unknown`). `pc` is
    /// left unchanged (the instruction was not executed), so re-stepping
    /// re-reports the same unimplemented instruction rather than silently
    /// skipping past it. `word` is the raw fetched bytes, packed
    /// little-endian, for diagnostics.
    Unknown { pc: u32, word: u32 },
}

/// The Xtensa interpreter core: program counter plus the windowed register file.
pub struct Cpu {
    /// Program counter -- byte address into the firmware's address space.
    pub pc: u32,
    /// Windowed register file (AR0-63, WINDOWBASE/WINDOWSTART, SAR, PS).
    pub regs: RegFile,
    /// Exception vector base. Window overflow/underflow vectors are computed
    /// as `vecbase + window_vector_offset(..)`. Set directly for now; the
    /// firmware's own VECBASE write (`wur`) is wired with the special-register
    /// support in M1.6/M1.7.
    pub vecbase: u32,
    /// Saved restart PC for the most recent exception (Xtensa EPC1). Window
    /// exceptions are restartable, so this holds the faulting instruction's
    /// own address -- the handler re-executes it after spilling/filling.
    pub epc1: u32,
    /// Xtensa MMU-v3 state (TLBs + config regs). Translation flows through
    /// `Cpu::translate`; `Bus` only ever sees physical addresses.
    pub mmu: super::mmu::Mmu,
}

impl Cpu {
    /// Create a CPU with `pc` at `entry`, a fresh zeroed register file, and a
    /// zero vector base (boot sets `vecbase` before enabling window exceptions).
    pub fn new(entry: u32) -> Self {
        Self {
            pc: entry,
            regs: RegFile::new(),
            vecbase: 0,
            epc1: 0,
            mmu: super::mmu::Mmu::new(),
        }
    }

    /// Raise a window overflow (`overflow=true`) or underflow exception for a
    /// call of size `k` quads faulting at `faulting_pc`: save the restart PC to
    /// [`Cpu::epc1`], enter exception mode, vector to the VECBASE-relative
    /// window handler, and report it via [`Step::Exception`]. Called by
    /// `control::exec`'s `Entry`/`Retw` handling.
    fn raise_window_exception(&mut self, faulting_pc: u32, overflow: bool, k: u32) -> Step {
        self.epc1 = faulting_pc;
        self.regs.set_excm();
        let vector = self.vecbase.wrapping_add(window_vector_offset(overflow, k));
        self.pc = vector;
        let cause = if overflow {
            CAUSE_WINDOW_OVERFLOW
        } else {
            CAUSE_WINDOW_UNDERFLOW
        };
        Step::Exception { cause, pc: vector }
    }

    /// Raise a general (non-window) exception with architectural cause code
    /// `cause` (e.g. [`EXCCAUSE_SYSCALL`], [`EXCCAUSE_INTEGER_DIVIDE_BY_ZERO`]):
    /// record it in EXCCAUSE, save the restart PC to [`Cpu::epc1`], enter
    /// exception mode, and vector to the VECBASE-relative KernelException
    /// vector ([`KERNEL_EXCEPTION_VECTOR_OFFSET`] -- this firmware runs
    /// kernel-mode; see that constant's doc for the PS.UM selection we
    /// deliberately don't model). A NEW, SEPARATE path from
    /// [`Cpu::raise_window_exception`] (M1.5) -- window exceptions vector
    /// through their own dedicated addresses and use a synthetic internal
    /// cause ID, not EXCCAUSE; this path is the real Xtensa EXCCAUSE-based
    /// mechanism every synchronous general exception shares. Called by
    /// `system::exec`'s `Syscall` handling and `arith::exec`'s `Quou`/
    /// `Remu`/`Rems` divide-by-zero handling.
    fn raise_general_exception(&mut self, faulting_pc: u32, cause: u32) -> Step {
        self.regs.exccause = cause;
        self.epc1 = faulting_pc;
        self.regs.set_excm();
        let vector = self.vecbase.wrapping_add(KERNEL_EXCEPTION_VECTOR_OFFSET);
        self.pc = vector;
        Step::Exception { cause, pc: vector }
    }

    /// Shared `call8`/`callx8` effect: stash the return address (with the call
    /// size, 2, in bits 31:30) into a8 of the current window, record the call
    /// increment in PS.CALLINC for the callee's `entry`, and jump to `target`.
    /// WINDOWBASE is NOT rotated here -- `entry` does that. Called by
    /// `control::exec`'s `Call8`/`Callx8` handling.
    fn enter_call(&mut self, pc: u32, len: u8, target: u32) {
        let ret = pc.wrapping_add(len as u32);
        self.regs.write_ar(8, (2 << 30) | (ret & 0x3FFF_FFFF));
        self.regs.set_callinc(2);
        self.pc = target;
    }

    /// Route a `wsr.<sr>` write to the modeled state for the special registers
    /// the interpreter tracks (SAR/WINDOWBASE/WINDOWSTART/EPC1/PS/VECBASE),
    /// plus the MMU-config SRs (PTEVADDR/RASID/ITLBCFG/DTLBCFG, routed into
    /// `cpu.mmu` -- M2b Task 4); any other SR is logged and dropped -- their
    /// effect is on hardware state this phase doesn't simulate, not on the
    /// interpreter's registers. Called by `system::exec`'s `Wsr` handling.
    fn write_sr(&mut self, sr: u8, value: u32) {
        match sr {
            SR_SAR => self.regs.sar = value,
            SR_WINDOWBASE => self.regs.windowbase = value % NUM_FRAMES,
            SR_WINDOWSTART => self.regs.windowstart = value,
            SR_EPC1 => self.epc1 = value,
            SR_PS => self.regs.ps = value,
            SR_VECBASE => self.vecbase = value,
            SR_PTEVADDR => self.mmu.ptevaddr = value,
            SR_RASID => self.mmu.write_rasid(value),
            SR_ITLBCFG => self.mmu.itlbcfg = value,
            SR_DTLBCFG => self.mmu.dtlbcfg = value,
            _ => log::debug!(
                "firmware interp: wsr.0x{:02x} = 0x{:08x} (unmodeled SR; logged no-op)",
                sr,
                value
            ),
        }
    }

    /// Route a `rsr.<sr> at` read to the modeled state for the same
    /// special registers [`Cpu::write_sr`] tracks, plus [`SR_EXCCAUSE`]; any
    /// other SR returns 0 and is logged (mirrors `write_sr`'s
    /// log-and-no-op treatment of unmodeled SRs, so a read from an
    /// unmodeled register is visible in the log rather than silently
    /// returning a plausible-looking value). Called by `system::exec`'s
    /// `Rsr` handling.
    fn read_sr(&self, sr: u8) -> u32 {
        match sr {
            SR_SAR => self.regs.sar,
            SR_WINDOWBASE => self.regs.windowbase,
            SR_WINDOWSTART => self.regs.windowstart,
            SR_EPC1 => self.epc1,
            SR_PS => self.regs.ps,
            SR_PTEVADDR => self.mmu.ptevaddr,
            SR_RASID => self.mmu.rasid,
            SR_ITLBCFG => self.mmu.itlbcfg,
            SR_DTLBCFG => self.mmu.dtlbcfg,
            SR_VECBASE => self.vecbase,
            SR_EXCCAUSE => self.regs.exccause,
            _ => {
                log::debug!("firmware interp: rsr.0x{:02x} (unmodeled SR; logged, returning 0)", sr);
                0
            }
        }
    }

    /// Route a `wur at,<ur>` write to the modeled user-register state: only
    /// [`UR_VECBASE`] is modeled (routed to `cpu.vecbase`, the SAME field
    /// `write_sr`'s `SR_VECBASE` arm uses -- see `decode::Op::Wur`'s doc for
    /// why: the real firmware's own naming, per Ghidra, calls this UR
    /// "VECBASE", so this models it as an alternate WUR-based access path to
    /// the identical architectural state `wsr.vecbase` sets); any other UR
    /// is logged and dropped. Called by `system::exec`'s `Wur` handling.
    fn write_ur(&mut self, ur: u8, value: u32) {
        match ur {
            UR_VECBASE => self.vecbase = value,
            _ => log::debug!(
                "firmware interp: wur.0x{:02x} = 0x{:08x} (unmodeled UR; logged no-op)",
                ur,
                value
            ),
        }
    }

    /// Fetch, decode, and execute one instruction from `bus` at `self.pc`.
    ///
    /// Advances `pc` by the decoded instruction length on every executed op.
    /// Implements the non-windowed base-ISA subset (`movi`/`movi.n`/`mov.n`,
    /// `l32i`/`l32i.n`/`l32r`, `or`, `extui`, `isync`/`witlb` as logged
    /// no-ops) and the windowed-call ABI (`call8`/`callx8`/`entry`/`retw`/
    /// `retw.n`, with the window overflow/underflow raise path). Anything
    /// the decode module doesn't recognize returns `Step::Unknown` without advancing
    /// `pc`; a raised window exception returns `Step::Exception` and leaves
    /// `pc` at the window-exception vector.
    ///
    /// Dispatches to `mem::exec`/`arith::exec`/`control::exec`/`system::exec`/
    /// `branch::exec` in turn: each returns `Some(Step)` if `decoded.op` is
    /// one of its ops, `None` otherwise, so the next category gets a turn.
    ///
    /// **Zero-overhead-loop loop-back (M2a Task 7).** After the dispatched
    /// op retires, this checks whether `pc` should loop back to LBEG: it
    /// fires only when the executed instruction advanced `pc` SEQUENTIALLY
    /// (`self.pc == pc + len`, the plain fall-through address) to exactly
    /// `regs.lend`, with `regs.lcount != 0`. "Sequentially" is exactly the
    /// distinction every op category's own tail already encodes: `mem`/
    /// `arith`/`system`/`Entry` unconditionally set `cpu.pc = pc + len`; a
    /// not-taken `branch::exec` also falls through to `pc + len`; but a
    /// taken branch, `jx`/`call8`/`callx8`/`retw`/`retw.n`, a skipped
    /// `loopnez`, or a window-exception vector all leave `cpu.pc` somewhere
    /// else. So `self.pc == pc + len` after the dispatch is precisely "this
    /// instruction did NOT itself redirect control flow" -- comparing
    /// addresses is enough to recover that distinction without threading a
    /// separate flag through every category's `exec`.
    ///
    /// This mirrors QEMU `target/xtensa/translate.c`'s `gen_check_loop_end`:
    /// it's reachable only when an instruction's own `translate()` left
    /// `is_jmp == DISAS_NEXT` (didn't perform its own jump), and for a
    /// conditional branch specifically only on the not-taken arm (the taken
    /// arm calls `gen_jumpi` directly, bypassing the check) -- i.e. the same
    /// "sequential vs. self-redirected" split this implements via address
    /// comparison. **Deferred edge case**: a taken branch/jump whose target
    /// coincidentally equals `pc + len` (an offset-0 branch to the very next
    /// instruction) would be misclassified as sequential by this proxy, where
    /// real hardware (via QEMU's `is_jmp`-based mechanism) would not
    /// loop-check it. This is architecturally possible but does not occur in
    /// this firmware (a branch/jump target equal to its own fall-through
    /// address is dead code no real compiler emits) and is not modeled as a
    /// distinct case -- see the task-7 report for the full argument.
    pub fn step(&mut self, bus: &mut Bus) -> Step {
        let pc = self.pc;
        let bytes = [bus.load8(pc), bus.load8(pc.wrapping_add(1)), bus.load8(pc.wrapping_add(2))];
        let decoded = decode::decode(&bytes, pc);
        if let Op::Unknown { word } = decoded.op {
            return Step::Unknown { pc, word };
        }
        let len = decoded.len;
        let step = mem::exec(self, bus, &decoded.op, pc, len)
            .or_else(|| arith::exec(self, bus, &decoded.op, pc, len))
            .or_else(|| control::exec(self, bus, &decoded.op, pc, len))
            .or_else(|| system::exec(self, bus, &decoded.op, pc, len))
            .or_else(|| branch::exec(self, bus, &decoded.op, pc, len))
            .unwrap_or_else(|| panic!("decoded op {:?} not handled by any category", decoded.op));
        if step == Step::Ran
            && self.pc == pc.wrapping_add(len as u32)
            && self.pc == self.regs.lend
            && self.regs.lcount != 0
        {
            self.regs.lcount -= 1;
            self.pc = self.regs.lbeg;
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    #[test]
    fn unknown_step_leaves_pc_unchanged_for_reinspection() {
        // `00 00 00` decodes to Op::Unknown (op0=0 RRR with no matching
        // sub-op). The point here is specifically that `pc` does NOT advance
        // on Step::Unknown, so a caller that re-steps sees the same
        // unimplemented instruction again rather than skipping past it.
        let rom = vec![0x00, 0x00, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        match cpu.step(&mut bus) {
            Step::Unknown { pc, word } => {
                assert_eq!(pc, 0);
                assert_eq!(word, 0);
            }
            other => panic!("expected Step::Unknown, got {:?}", other),
        }
        assert_eq!(cpu.pc, 0);
    }

    #[test]
    fn loop_back_fires_on_a_not_taken_branchs_sequential_fallthrough() {
        // `beqz.n a2, 0x4` (`8c 02`, narrow, len 2) @ pc 0. With AR[2]!=0 the
        // branch is NOT taken, so branch::exec's tail falls through to
        // `pc+len == 2` exactly like any other sequential op -- pinning that
        // the loop-back mechanism reacts to a not-taken CONDITIONAL BRANCH's
        // fallthrough the same way it does to a plain mem/arith op's tail
        // (both funnel through the identical `self.pc == pc+len` check in
        // `step()`), not just to the Loop/Loopnez-body ops exercised
        // elsewhere.
        let rom = vec![0x8c, 0x02];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 1); // nonzero -> beqz.n not taken
        cpu.regs.lend = 2; // == the not-taken fallthrough pc
        cpu.regs.lcount = 5;
        cpu.regs.lbeg = 0x100;

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x100, "not-taken fallthrough hit LEND -> looped back to LBEG");
        assert_eq!(cpu.regs.lcount, 4, "LCOUNT decremented");
    }

    #[test]
    fn loop_back_does_not_fire_on_a_taken_branch_even_when_its_target_is_lend() {
        // Same instruction, but AR[2]==0 -> beqz.n IS taken, jumping to its
        // decoded target (4) -- which is deliberately set equal to LEND
        // here. This is not a pathological/deferred edge: it's the common
        // real-world case of an early-exit branch jumping straight to a
        // loop's exit address. Per QEMU `translate.c` (`gen_brcond` only
        // calls the loop-end check on the NOT-taken arm; the taken arm calls
        // `gen_jumpi` directly) and this module's `step()` doc, a TAKEN
        // branch must never trigger loop-back, regardless of where it lands
        // -- `self.pc` (4) != `pc+len` (2) here, so the sequential guard
        // correctly excludes it even though `self.pc == lend` would
        // otherwise hold.
        let rom = vec![0x8c, 0x02];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0); // zero -> beqz.n taken
        cpu.regs.lend = 4; // == the branch's own taken target
        cpu.regs.lcount = 9;
        cpu.regs.lbeg = 0x200;

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 4, "taken branch lands on its target, no loop-back redirect");
        assert_eq!(cpu.regs.lcount, 9, "LCOUNT untouched -- taken control flow never checks LEND");
    }

    #[test]
    fn undecoded_opcode_returns_unknown() {
        // 0xff is not a valid op0 format selector (the decode module treats 0xE/0xF
        // as a single undecodable byte); must report Unknown, not panic.
        let rom = vec![0xff, 0xff, 0xff];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Unknown { .. }));
        assert_eq!(cpu.pc, 0);
    }
}
