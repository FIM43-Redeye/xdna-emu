//! Xtensa interpreter core: fetch/decode/execute over the `Bus` for the
//! non-windowed base-ISA subset plus the windowed-call ABI (M1.5; the full
//! call4/call8/call12/callx4/callx8/callx12 family completed M2c Phase 2
//! iter1 -- the reset/MMU-setup prologue uses only call8/callx8, but the C
//! runtime uses all three sizes).
//!
//! Windowed-ABI model (matches the Xtensa windowed-register option; the call
//! increment `k` is 1/2/3 quads for call4/call8/call12 respectively, always
//! recorded in PS.CALLINC by the call and read back by `entry`/`retw`):
//! - `call{4,8,12}`/`callx{4,8,12}` stash the return address (with `k` in bits
//!   31:30) into a[4k] of the *caller's* window and set PS.CALLINC -- they do
//!   NOT rotate WINDOWBASE.
//! - `entry` rotates WINDOWBASE forward by PS.CALLINC, allocates the callee
//!   frame, and marks the new frame live in WINDOWSTART; the return address
//!   the caller left in a[4k] thereby becomes the callee's a0.
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
//! **General-exception raise (M2a Task 9; routing corrected in M2c iter13).**
//! A second, separate raise path from the window-exception one above:
//! `syscall` and integer divide-by-zero (`quou`/`remu`/`rems`) both vector
//! through [`Cpu::raise_general_exception`] to the firmware's unified
//! general-exception handler ([`GENERAL_EXCEPTION_HANDLER`]) -- unless PS.EXCM
//! is already set, in which case it is a double fault and vectors to the
//! VECBASE-relative [`DOUBLE_EXCEPTION_VECTOR_OFFSET`]. The raise records a
//! REAL architectural EXCCAUSE value (unlike the window path's synthetic
//! diagnostic cause IDs) that the handler reads to dispatch on the specific
//! cause.

mod arith;
mod branch;
mod control;
mod fastpath;
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
/// INTERRUPT (read) / INTSET (write) special register (`cpu.h` sregs index
/// 226 = 0xE2). Reading returns pending interrupt bits; writing SETS bits
/// (OR). Verified against QEMU target/xtensa.
const SR_INTERRUPT: u8 = 0xE2;
/// INTCLEAR (write) special register (index 227 = 0xE3): writing CLEARS the
/// named pending interrupt bits (AND-NOT). Write-only -- a read of 0xE3 is
/// architecturally undefined and falls through to the unmodeled-SR path
/// (returns 0), so no read arm is wired for it (unlike 0xE2/0xE4).
const SR_INTCLEAR: u8 = 0xE3;
/// INTENABLE special register (index 228 = 0xE4): per-bit interrupt enable
/// mask.
const SR_INTENABLE: u8 = 0xE4;
/// SCOMPARE1 special register (index 12): the compare value the atomic
/// `s32c1i` tests the memory word against before storing (`xtensa-modules.c`
/// SREG SCOMPARE1 == 12).
const SR_SCOMPARE1: u8 = 12;

/// MMU config special registers (QEMU `cpu.h` sregs indices; PTEVADDR/ITLBCFG/
/// DTLBCFG cross-checked against decode/system.rs's existing decode tests).
const SR_PTEVADDR: u8 = 0x53;
const SR_RASID: u8 = 0x5A;
const SR_ITLBCFG: u8 = 0x5B;
const SR_DTLBCFG: u8 = 0x5C;

/// User-register (TIE) number `wur`/`rur` for THREADPTR, routed into
/// `cpu.threadptr` via [`Cpu::write_ur`]. NOT the same as [`SR_VECBASE`]:
/// special registers and user registers are architecturally separate 8-bit
/// spaces that merely share the numeric value 0xE7. AMD's vendored generic
/// `xtensa-modules.c` names UR 0xE7 "threadptr" (correct); the firmware's own
/// Ghidra disassembly mislabels this instance "VECBASE". The DYNAMIC boot
/// behavior settles it as threadptr, not vecbase: the firmware writes a
/// transient stack pointer (`sp+16`) here and immediately `syscall`s (a
/// syscall-argument pattern -- you never point the exception vector base at
/// the stack), and routing it to `cpu.vecbase` corrupted VECBASE so the
/// syscall vectored into a zero page; routing it to a separate `threadptr`
/// preserves the real prologue-set VECBASE (0x800) so the syscall reaches its
/// loaded handler. See `decode::Op::Wur`'s doc.
const UR_THREADPTR: u8 = 0xE7;

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
/// EXCCAUSE value for a level-1 interrupt (`LEVEL1_INTERRUPT`, the 5th entry
/// / index 4 in QEMU's cause enum). A level-1 interrupt shares the general
/// user/kernel exception vector and dispatches on EXCCAUSE, so it is a
/// general exception with this cause -- delivery reuses
/// `raise_general_exception`.
pub const EXCCAUSE_LEVEL1_INTERRUPT: u32 = 4;
/// EXCCAUSE value for integer divide-by-zero (`quou`/`remu`/`rems` dividing
/// by zero -- see `interp::arith::exec`). Verified against QEMU
/// `target/xtensa/cpu.h`'s exception-cause enum: counting from
/// `ILLEGAL_INSTRUCTION_CAUSE = 0`, `INTEGER_DIVIDE_BY_ZERO_CAUSE` is the
/// 7th entry (index 6): `ILLEGAL_INSTRUCTION(0), SYSCALL(1),
/// INSTRUCTION_FETCH_ERROR(2), LOAD_STORE_ERROR(3), LEVEL1_INTERRUPT(4),
/// ALLOCA(5), INTEGER_DIVIDE_BY_ZERO(6)`.
pub const EXCCAUSE_INTEGER_DIVIDE_BY_ZERO: u32 = 6;

/// Absolute image address of the firmware's UNIFIED general-exception handler
/// -- where `syscall`, integer divide-by-zero, and every other synchronous
/// non-double general exception vectors to. The handler reads EXCCAUSE to
/// dispatch on the specific cause (`rsr.exccause; bnei a3,1` = syscall path,
/// which advances EPC1 by 3; it then saves the full a0-a15 + SAR/PS/EPC1/
/// loop/FPU context to an exception frame). Unlike the window and double
/// vectors, this is NOT a VECBASE-relative vector: it is reached at a fixed
/// address in the firmware's own code, so VECBASE is irrelevant to the target.
///
/// DERIVED EMPIRICALLY (M2c iter13): the value was pinned by a redirect
/// experiment -- routing the boot's one user-mode `syscall` here is the ONLY
/// target that clears the "main returns" wall at 0x2000e035 (routing to any
/// VECBASE-region stub, including iter7's 0x28b4, still walls). The handler
/// itself was located by scanning the image for `rsr.exccause` sites (only 3
/// exist) and reading the one with the `bnei a3,1` syscall dispatch.
///
/// FIXME(iter13-B): this is the pragmatic "route directly to the handler"
/// model. On real hardware the CPU vectors a general exception to a
/// VECBASE-relative UserExceptionVector / KernelExceptionVector (selected by
/// PS.UM), whose stub reaches THIS handler via a RUNTIME-INSTALLED dispatch
/// pointer that `init` writes into RAM -- a pointer our boot never populates
/// (the 0xe1fc dispatch slot reads zero). We proved no static path reaches the
/// handler: no literal equals its address in either window, no `j` reaches it,
/// and an exhaustive 1024-offset VECBASE sweep finds nothing. Modeling the
/// real runtime handler-registration (so kernel/user route through their true
/// per-mode vectors) is deferred -- it needs RE of `init`'s dispatch-table
/// setup, entangled with the Harvard IRAM/DRAM split. iter7's
/// `0x28b4`-as-kernel-vector was a mislabel (0x28b4 uses the interrupted
/// `a1`/`a4` as a live call-frame -- architecturally impossible for a vector).
/// Full derivation: `docs/superpowers/findings/2026-07-05-iter13-user-exception-vector-routing.md`.
const GENERAL_EXCEPTION_HANDLER: u32 = 0x2958;

/// MMU-fault EXCCAUSE values (`cpu.h:266-294`). Derived from QEMU; these are
/// the architectural cause codes a TLB miss/multi-hit/privilege/prohibited
/// fault reports through the same EXCCAUSE channel as syscall/divide-by-zero.
pub const EXCCAUSE_INST_TLB_MISS: u32 = 16;
pub const EXCCAUSE_INST_TLB_MULTI_HIT: u32 = 17;
pub const EXCCAUSE_INST_FETCH_PRIVILEGE: u32 = 18;
pub const EXCCAUSE_INST_FETCH_PROHIBITED: u32 = 20;
pub const EXCCAUSE_LOAD_STORE_TLB_MISS: u32 = 24;
pub const EXCCAUSE_LOAD_STORE_TLB_MULTI_HIT: u32 = 25;
pub const EXCCAUSE_LOAD_STORE_PRIVILEGE: u32 = 26;
pub const EXCCAUSE_LOAD_PROHIBITED: u32 = 28;
pub const EXCCAUSE_STORE_PROHIBITED: u32 = 29;

/// VECBASE-relative offset of the DoubleExceptionVector (`EXC_DOUBLE`): a
/// fault raised while PS.EXCM is already set vectors here instead of the
/// unified general handler (`exc_helper.c:56-58`).
///
/// `0x31c` is DERIVED FROM THIS CORE (M2c iter13, correcting iter7's SUSPECT
/// standard-config `0x3C0`). The real double handler sits at `VECBASE + 0x31c
/// = 0xb1c` -- confirmed by disassembly: `wsr.excsave1/2/5/6; rsr.exccause;
/// bne a2,a3; ...; rsr.depc; wsr.depc; rfde`. The terminating `rfde` (Return
/// From Double Exception) and the `depc` (Double Exception PC) manipulation
/// are unambiguous double-handler markers. The old `0x3C0` -> 0xbc0 is zeros
/// in the image (not a handler). Still on an unexercised path (no double fault
/// occurs in the boot yet), but now pinned rather than guessed.
const DOUBLE_EXCEPTION_VECTOR_OFFSET: u32 = 0x31c;

/// EXCVADDR special register (`cpu.h` sregs index 238 = 0xEE).
const SR_EXCVADDR: u8 = 0xEE;

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
    /// Faulting virtual address of the most recent load/store/fetch fault
    /// (Xtensa EXCVADDR); set by `translate` before raising (`exc_helper.c:73`).
    pub excvaddr: u32,
    /// When true (default), `step()` collapses recognized large contiguous-fill
    /// loops via `fastpath::try_fill_loop` instead of grinding them. Tests flip
    /// it off to compare fast-path output against per-iteration interpretation.
    pub fastpath_enabled: bool,
    /// Xtensa THREADPTR user (TIE) register (`wur`/`rur` UR 0xE7). The firmware
    /// stores a stack struct pointer here to pass an argument across a `syscall`
    /// trap. SEPARATE from `vecbase` despite sharing the number 0xE7 -- SR and
    /// UR are architecturally distinct register spaces; see `Cpu::write_ur` and
    /// `decode::Op::Wur`.
    pub threadptr: u32,
    /// INTERRUPT pending-interrupt bits (Xtensa INTERRUPT SR read / INTSET SR
    /// write). A modeled interrupt source (the mailbox doorbell) sets a bit;
    /// `INTCLEAR` clears it; the handler acks by clearing. Level-1 only.
    pub interrupt: u32,
    /// INTENABLE per-bit interrupt-enable mask (Xtensa INTENABLE SR). An
    /// interrupt is deliverable only if its bit is set here.
    pub intenable: u32,
    /// SCOMPARE1 (Xtensa SR 12): the compare value for the atomic
    /// compare-and-store `s32c1i`. `wsr.scompare1` sets it; `s32c1i` stores
    /// only when the memory word equals it. Load-bearing for the firmware's
    /// lock-free scheduler queues.
    pub scompare1: u32,
    /// Floating-point register file (f0-f15), stored as RAW 32-bit bit patterns
    /// -- this interpreter models NO floating-point semantics. The firmware's
    /// general-exception handler save/restores FP context with `lsi`/`ssi`
    /// (load/store single); those move bit patterns to and from memory, which
    /// an opaque register file reproduces exactly. If a boot path ever performs
    /// actual FP ARITHMETIC (`add.s`/`mul.s`/...), that becomes its own scoped
    /// decision -- until then FP is pure storage. (iter15.)
    pub fr: [u32; 16],
    /// True when a `waiti` has retired and the CPU is idling until a
    /// deliverable interrupt arrives (Xtensa "waiti" halt state, QEMU
    /// `env->halted`). Cleared when an interrupt is delivered.
    pub halted: bool,
}

/// Which access class a translation is for -- selects ITLB vs DTLB and the
/// permission subset checked (`Cpu::translate`'s `is_write` argument to
/// `Mmu::translate`: 2=fetch, 0=load, 1=store).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Fetch,
    Load,
    Store,
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
            excvaddr: 0,
            fastpath_enabled: true,
            threadptr: 0,
            interrupt: 0,
            intenable: 0,
            scompare1: 0,
            fr: [0; 16],
            halted: false,
        }
    }

    /// Translate a virtual address to physical for `access`, or return a
    /// ready-to-propagate `Step::Exception` on a TLB/permission fault. The one
    /// chokepoint for all address translation; `Bus` downstream sees only
    /// physical addresses. `mmu_idx` (the ring `Mmu::translate` checks
    /// privilege against) is hardwired to 0 -- this firmware runs kernel-only,
    /// PS.RING is never set (M2b Task 6 flagged confirming this; it is
    /// deliberate).
    pub fn translate(&mut self, bus: &mut Bus, vaddr: u32, access: Access) -> Result<u32, Step> {
        let is_write = match access {
            Access::Fetch => 2u8,
            Access::Load => 0,
            Access::Store => 1,
        };
        match self.mmu.translate(bus, vaddr, is_write, 0) {
            Ok(t) => Ok(t.paddr),
            Err(fault) => {
                // EPC1 = the faulting INSTRUCTION's pc (self.pc), EXCVADDR =
                // the fault target (vaddr) -- matches QEMU's
                // `exception_cause_vaddr(env, env->pc, cause, address)`
                // (exc_helper.c). These coincide for a Fetch fault (you fault
                // trying to execute *at* self.pc, so vaddr == self.pc at that
                // call site) but differ for Load/Store, where self.pc is the
                // load/store instruction and vaddr is the memory address it
                // was accessing -- using self.pc here is correct for both,
                // with no special-casing needed in Task 9's call sites.
                self.excvaddr = fault.vaddr;
                Err(self.raise_general_exception(self.pc, fault.cause))
            }
        }
    }

    /// Canonical translation-aware CPU data read: translate (DTLB, load), then
    /// D-side physical route. The SOLE entry for "what the CPU reads at virtual
    /// data address V" -- executor ops and probes share it, so probe-vs-CPU
    /// disagreement is structurally impossible. Faults propagate as Step::Exception
    /// exactly as Cpu::translate raises them.
    pub fn data_read32(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u32, Step> {
        let paddr = self.translate(bus, vaddr, Access::Load)?;
        Ok(bus.data_load32(paddr))
    }
    pub fn data_read8(&mut self, bus: &mut Bus, vaddr: u32) -> Result<u8, Step> {
        let paddr = self.translate(bus, vaddr, Access::Load)?;
        Ok(bus.data_load8(paddr))
    }
    pub fn data_write32(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
        let paddr = self.translate(bus, vaddr, Access::Store)?;
        bus.data_store32(paddr, v);
        Ok(())
    }
    pub fn data_write8(&mut self, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
        let paddr = self.translate(bus, vaddr, Access::Store)?;
        bus.data_store8(paddr, v);
        Ok(())
    }

    /// Raise a window overflow (`overflow=true`) or underflow exception for a
    /// call of size `k` quads faulting at `faulting_pc`: save the restart PC to
    /// [`Cpu::epc1`], enter exception mode, vector to the VECBASE-relative
    /// window handler, and report it via [`Step::Exception`]. Called by
    /// `control::exec`'s `Entry`/`Retw` handling.
    /// `rotate_delta` rotates WINDOWBASE to position the exception handler's
    /// window over the frame it spills/fills (+n forward for overflow, -k back
    /// for underflow); `vector_quads` (1/2/3) selects the Overflow/Underflow
    /// 4/8/12 vector. The pre-rotation WINDOWBASE is saved in PS.OWB so `rfwo`/
    /// `rfwu` can restore it. Mirrors QEMU `HELPER(window_check)` /
    /// `HELPER(test_underflow_retw)`.
    fn raise_window_exception(
        &mut self,
        faulting_pc: u32,
        overflow: bool,
        rotate_delta: i32,
        vector_quads: u32,
    ) -> Step {
        self.regs.set_owb(self.regs.windowbase);
        self.regs.rotate(rotate_delta);
        self.epc1 = faulting_pc;
        self.regs.set_excm();
        let vector = self.vecbase.wrapping_add(window_vector_offset(overflow, vector_quads));
        self.pc = vector;
        let cause = if overflow {
            CAUSE_WINDOW_OVERFLOW
        } else {
            CAUSE_WINDOW_UNDERFLOW
        };
        Step::Exception { cause, pc: vector }
    }

    /// Windowed-register-ABI overflow guard, run in [`Cpu::step`] BEFORE
    /// executing any instruction whose highest AR operand ([`Op::max_ar`])
    /// reaches beyond the current frame into a live older one. `max_reg`'s quad
    /// (`max_reg / 4`) is the window requirement `w`; if a live older frame
    /// sits within `w` quads ahead of WINDOWBASE, spill it (raise
    /// WindowOverflow) so the instruction's register access can't corrupt it,
    /// and re-execute the faulting instruction after `rfwo`. Returns `Some(step)`
    /// when it raised, `None` when the access fits.
    ///
    /// Without this, a full window (every frame live) lets a high-register write
    /// -- e.g. loading an outgoing-argument or return-address register into
    /// a8..a15 -- clobber the oldest frame's saved registers *before* the
    /// entry-time overflow can spill them, losing that frame's return address
    /// (the pc=0 boot wall this closes). Suppressed unless window exceptions are
    /// enabled (PS.WOE set, PS.EXCM clear): the spill handler itself runs with
    /// EXCM=1 and uses `s32e` on a8..a15, which must not re-fault. Mirrors QEMU
    /// `gen_window_check` -> `HELPER(window_check)`; `w == 0` (max_reg < 4, the
    /// always-live current quad) never overflows via `overflow_check`.
    fn window_check(&mut self, max_reg: u8, faulting_pc: u32) -> Option<Step> {
        if !self.regs.window_exceptions_enabled() {
            return None;
        }
        let (n, spill_quads) = self.regs.overflow_check((max_reg / 4) as u32)?;
        Some(self.raise_window_exception(faulting_pc, true, n as i32, spill_quads))
    }

    /// True when a level-1 interrupt may be delivered right now: a pending bit
    /// is enabled (`INTERRUPT & INTENABLE`), the current level is 0 (a nonzero
    /// PS.INTLEVEL -- e.g. a `waiti 1+` -- masks level-1), and PS.EXCM is
    /// clear (we are not already inside a handler). The `intlevel()==0` gate
    /// is the level-1 specialization of "interrupt level > PS.INTLEVEL"; this
    /// interpreter models level-1 only. Derived from QEMU
    /// `xtensa_cpu_has_work` / `handle_interrupt`.
    pub fn interrupt_deliverable(&self) -> bool {
        self.regs.intlevel() == 0 && !self.regs.excm() && (self.interrupt & self.intenable) != 0
    }

    /// Raise a general (non-window) exception with architectural cause code
    /// `cause` (e.g. [`EXCCAUSE_SYSCALL`], [`EXCCAUSE_INTEGER_DIVIDE_BY_ZERO`]):
    /// record it in EXCCAUSE, save the restart PC to [`Cpu::epc1`], enter
    /// exception mode, and vector to the firmware's unified general-exception
    /// handler ([`GENERAL_EXCEPTION_HANDLER`], a fixed image address -- see its
    /// doc for why it is NOT VECBASE-relative and the FIXME on the runtime
    /// dispatch we don't yet model) -- UNLESS PS.EXCM is already set, in which
    /// case this is a double fault and vectors instead to the VECBASE-relative
    /// [`DOUBLE_EXCEPTION_VECTOR_OFFSET`] DoubleExceptionVector
    /// (`exc_helper.c:56-58`; closes M2a carry-forward finding 9a). A NEW,
    /// SEPARATE path from [`Cpu::raise_window_exception`] (M1.5) -- window
    /// exceptions vector through their own dedicated addresses and use a
    /// synthetic internal cause ID, not EXCCAUSE; this path is the real
    /// Xtensa EXCCAUSE-based mechanism every synchronous general exception
    /// shares. Called by `system::exec`'s `Syscall` handling, `arith::exec`'s
    /// `Quou`/`Remu`/`Rems` divide-by-zero handling, and `Cpu::translate`'s
    /// MMU fault path (M2b Task 7).
    fn raise_general_exception(&mut self, faulting_pc: u32, cause: u32) -> Step {
        self.regs.exccause = cause;
        self.epc1 = faulting_pc;
        let vector = if self.regs.excm() {
            // Fault while already in exception mode -> DoubleExceptionVector
            // (exc_helper.c:56-58). Closes M2a carry-forward 9a.
            self.vecbase.wrapping_add(DOUBLE_EXCEPTION_VECTOR_OFFSET)
        } else {
            // Non-double general exception -> the unified handler (iter13).
            // Absolute image address, not VECBASE-relative.
            GENERAL_EXCEPTION_HANDLER
        };
        self.regs.set_excm();
        self.pc = vector;
        Step::Exception { cause, pc: vector }
    }

    /// Shared windowed-call effect for a call increment `k` quads (call4/
    /// callx4 k=1, call8/callx8 k=2, call12/callx12 k=3): stash the return
    /// address (with `k` in bits 31:30) into a[4k] of the current window,
    /// record `k` in PS.CALLINC for the callee's `entry`, and jump to
    /// `target`. WINDOWBASE is NOT rotated here -- `entry` does that (reading
    /// `k` back out of PS.CALLINC, which is why `entry`/`retw` need no change
    /// to support the wider call family: they already read `k` dynamically
    /// via `callinc()` / `a0[31:30]`). Per the windowed-call ABI (QEMU
    /// `target/xtensa/translate.c`'s `gen_callw`; real Xtensa). Called by
    /// `control::exec`'s `Call4`/`Call8`/`Call12`/`Callx4`/`Callx8`/`Callx12`
    /// handling.
    fn enter_call(&mut self, pc: u32, len: u8, target: u32, k: u32) {
        let ret = pc.wrapping_add(len as u32);
        self.regs.write_ar((4 * k) as u8, (k << 30) | (ret & 0x3FFF_FFFF));
        self.regs.set_callinc(k);
        self.pc = target;
    }

    /// Execute a FLIX `xt_format1` bundle ([`Op::Flix1`]) with parallel read-all/
    /// write-all semantics. Real Xtensa VLIW slots read their sources from
    /// PRE-bundle register state and commit all their writes together; a naive
    /// sequential execution would wrongly let a later slot observe an earlier
    /// slot's write. We model the real behavior by snapshotting the register
    /// file, running each slot op against a fresh copy of that snapshot (so every
    /// slot reads pre-bundle values), and merging each op's register changes into
    /// a `result` file that becomes the post-bundle state. Memory stores go
    /// straight to the bus (they persist correctly across the per-op snapshot
    /// restores); the (at most one) control-flow slot (`j`/`jx`) runs LAST so its
    /// PC redirect wins over the other slots' `pc = pc + 8` fall-through.
    ///
    /// Window overflow is checked once for the widest register across all slots
    /// (one spill covers the whole bundle; on overflow the bundle re-runs after
    /// `rfwo`, pc unchanged). A slot op that faults (MMU/exception) aborts the
    /// bundle immediately with that Step (pc/regs already set by the fault path),
    /// matching a bundle that faults mid-issue. `ops.len() >= 2` always (0/1 real
    /// ops collapse to Nop/the single op at decode).
    fn exec_flix1_bundle(&mut self, bus: &mut Bus, ops: &[Op], pc: u32) -> Step {
        if let Some(mr) = ops.iter().filter_map(|o| o.max_ar()).max() {
            if let Some(step) = self.window_check(mr, pc) {
                return step;
            }
        }
        let snapshot = self.regs.clone();
        let mut result = snapshot.clone();
        let is_ctrl = |o: &Op| matches!(o, Op::J { .. } | Op::Jx { .. });
        // Non-control slots first; the control-flow slot (if any) last.
        let order = ops.iter().filter(|o| !is_ctrl(o)).chain(ops.iter().filter(|o| is_ctrl(o)));
        for op in order {
            self.regs = snapshot.clone(); // every slot reads pre-bundle state
            let step = mem::exec(self, bus, op, pc, 8)
                .or_else(|| arith::exec(self, bus, op, pc, 8))
                .or_else(|| control::exec(self, bus, op, pc, 8))
                .or_else(|| system::exec(self, bus, op, pc, 8))
                .or_else(|| branch::exec(self, bus, op, pc, 8))
                .unwrap_or_else(|| panic!("flix1 slot op {op:?} not handled by any category"));
            if step != Step::Ran {
                return step; // faulting slot aborts the bundle
            }
            // Merge this slot's AR/SAR writes into the accumulated result.
            for i in 0..16u8 {
                let v = self.regs.read_ar(i);
                if v != snapshot.read_ar(i) {
                    result.write_ar(i, v);
                }
            }
            if self.regs.sar != snapshot.sar {
                result.sar = self.regs.sar;
            }
        }
        self.regs = result;
        // Zero-overhead-loop back-edge (mirrors step()'s tail): a bundle sitting
        // exactly at LEND with a live loop count wraps to LBEG.
        if self.pc == pc.wrapping_add(8) && self.pc == self.regs.lend && self.regs.lcount != 0 {
            self.regs.lcount -= 1;
            self.pc = self.regs.lbeg;
        }
        Step::Ran
    }

    /// Route a `wsr.<sr>` write to the modeled state for the special registers
    /// the interpreter tracks (SAR/WINDOWBASE/WINDOWSTART/EPC1/PS/VECBASE),
    /// plus the MMU-config SRs (PTEVADDR/RASID/ITLBCFG/DTLBCFG, routed into
    /// `cpu.mmu` -- M2b Task 4) and EXCVADDR (M2b Task 7); any other SR is
    /// logged and dropped -- their effect is on hardware state this phase
    /// doesn't simulate, not on the interpreter's registers. Called by
    /// `system::exec`'s `Wsr` handling.
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
            SR_EXCVADDR => self.excvaddr = value,
            SR_INTERRUPT => self.interrupt |= value, // INTSET: set bits by OR
            SR_INTCLEAR => self.interrupt &= !value, // INTCLEAR: clear bits
            SR_INTENABLE => self.intenable = value,
            SR_SCOMPARE1 => self.scompare1 = value,
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
            SR_EXCVADDR => self.excvaddr,
            SR_INTERRUPT => self.interrupt,
            SR_INTENABLE => self.intenable,
            SR_SCOMPARE1 => self.scompare1,
            _ => {
                log::debug!("firmware interp: rsr.0x{:02x} (unmodeled SR; logged, returning 0)", sr);
                0
            }
        }
    }

    /// Route a `wur at,<ur>` write to the modeled user-register state: only
    /// [`UR_THREADPTR`] is modeled (routed to `cpu.threadptr`). This is the
    /// Xtensa THREADPTR TIE register -- a DISTINCT register from VECBASE despite
    /// sharing the number 0xE7 (SR and UR are separate spaces). Writing it must
    /// NOT touch `cpu.vecbase`: the firmware sets threadptr to a syscall-argument
    /// pointer and traps, and clobbering VECBASE here sent that syscall into an
    /// unmapped page (see [`UR_THREADPTR`] and `decode::Op::Wur`). Any other UR
    /// is logged and dropped. Called by `system::exec`'s `Wur` handling.
    fn write_ur(&mut self, ur: u8, value: u32) {
        match ur {
            UR_THREADPTR => self.threadptr = value,
            _ => log::debug!(
                "firmware interp: wur.0x{:02x} = 0x{:08x} (unmodeled UR; logged no-op)",
                ur,
                value
            ),
        }
    }

    /// Route a `rur arr,<ur>` read to the modeled user-register state -- the
    /// READ sibling of [`Cpu::write_ur`]. Only [`UR_THREADPTR`] is modeled
    /// (returns `cpu.threadptr`); every other UR -- notably `fcr` (0xE8) and
    /// `fsr` (0xE9), the FPU control/status registers the firmware's exception
    /// handler saves -- reads 0, since this interpreter models no FPU or other
    /// TIE state. Called by `system::exec`'s `Rur` handling.
    fn read_ur(&self, ur: u8) -> u32 {
        match ur {
            UR_THREADPTR => self.threadptr,
            _ => {
                log::debug!("firmware interp: rur.0x{:02x} (unmodeled UR; reads 0)", ur);
                0
            }
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
        // Interrupts are checked between instructions (faithful Xtensa). A
        // deliverable level-1 interrupt IS a general exception with
        // EXCCAUSE=4: reuse the proven raise_general_exception path, which
        // sets EPC1<-PC, EXCM, and routes to the real handler at 0x2958.
        // EXCM is clear here (interrupt_deliverable gates on it), so this
        // never mis-routes to the double vector. This check precedes the
        // `halted` short-circuit below so delivery wins over re-waiting --
        // a pending interrupt must wake a halted (post-waiti) CPU rather
        // than let it re-park.
        if self.interrupt_deliverable() {
            self.halted = false;
            return self.raise_general_exception(self.pc, EXCCAUSE_LEVEL1_INTERRUPT);
        }

        // A halted CPU (post-waiti) runs no instructions; it only yields Wait
        // until a deliverable interrupt arrives (checked above, ahead of this).
        if self.halted {
            return Step::Wait(WaitReason::Waiti);
        }

        // Fill-loop fast-path: if poised at a large contiguous-fill loop's
        // back-edge, collapse all its iterations in one shot (byte-identical to
        // grinding). The recognizer's own cheap gate returns immediately when
        // the PC is not at a large loop start, so this adds ~two comparisons to
        // the common path.
        if self.fastpath_enabled {
            if let Some(step) = fastpath::try_fill_loop(self, bus) {
                return step;
            }
        }

        let pc = self.pc;
        // Translate + fetch the first byte; its op0 nibble fixes the
        // instruction length (`decode::decode`'s own rule, `decode/mod.rs`
        // ~1046-1051: narrow `.n` ops are 2 bytes, 0xE/0xF are 1, everything
        // else is 3), so we translate/fetch ONLY the bytes the instruction
        // occupies -- never faulting on a speculative byte past the real
        // length in a possibly-unmapped next page.
        let phys0 = match self.translate(bus, pc, Access::Fetch) {
            Ok(p) => p,
            Err(step) => return step,
        };
        let b0 = bus.fetch8(pc, phys0);
        let op0 = b0 & 0xF;
        // op0 0xE/0xF are 8-byte FLIX bundles (xt_format1/xt_format2), narrow
        // .n ops (0x8..=0xD) are 2 bytes, everything else 3 -- fetch exactly
        // the instruction's bytes so we never fault on a speculative byte past
        // its real length in a possibly-unmapped next page.
        let need: usize = if op0 == 0xE || op0 == 0xF {
            8
        } else if (0x8..=0xD).contains(&op0) {
            2
        } else {
            3
        };
        let mut buf = [b0, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
        for i in 1..need {
            let phys_i = match self.translate(bus, pc.wrapping_add(i as u32), Access::Fetch) {
                Ok(p) => p,
                Err(step) => return step,
            };
            buf[i] = bus.fetch8(pc.wrapping_add(i as u32), phys_i);
        }
        let decoded = decode::decode(&buf[..need], pc);
        match &decoded.op {
            Op::Unknown { word } => return Step::Unknown { pc, word: *word },
            // FLIX xt_format1 bundle carrying >=2 real ops: parallel executor,
            // NOT the per-category dispatch (which never sees this variant).
            Op::Flix1 { ops } => return self.exec_flix1_bundle(bus, ops, pc),
            _ => {}
        }
        // Windowed-register-ABI overflow: before an instruction accesses a
        // register in a8..a15 that overlaps a live older frame, spill that
        // frame (WindowOverflow) so the access can't corrupt it. The faulting
        // instruction re-executes after `rfwo`, so `pc` must not advance --
        // hence this precedes exec and returns straight through. See
        // `Op::max_ar` / `Cpu::window_check`.
        if let Some(max_reg) = decoded.op.max_ar() {
            if let Some(step) = self.window_check(max_reg, pc) {
                return step;
            }
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

/// Test-only helper shared by every `step()`-driven test across `interp` and
/// its per-category siblings (`arith`/`mem`/`branch`/`control`/`system`):
/// builds a `Cpu` at `entry` with its 4KB code page identity-mapped R+X in
/// the ITLB. Fetch always translates now (this `step()`'s Task 8 change, with
/// no "MMU off" special case), so every test that calls `step()` needs its
/// code page mapped before the first fetch -- this keeps that one-time setup
/// DRY.
///
/// Installed into ITLB **way 1**, deliberately not way 0: `witlb`/`iitlb`
/// decode their AS operand's way index from its low bits (`Mmu::write_tlb`/
/// `invalidate_tlb`), so any test that drives one of those ops through an
/// unset (zero-defaulted) address register targets way 0/VPN 0 by accident --
/// which would silently evict this very fetch mapping mid-test if it lived
/// there too (caught by `system::wdtlb_iitlb_idtlb_dsync_...`'s `iitlb a5`
/// with an unset `a5`). Way 1 sidesteps that whole collision class.
///
/// A test whose `step()` calls cross into a second code page (a jump/
/// exception vector landing outside `entry`'s page) installs the extra page
/// itself via `cpu.mmu.write_tlb` -- way-1 autorefill indexing (`ei =
/// (vaddr>>12)&0x3`) only collides with THIS mapping when the other page
/// shares `entry`'s page low 2 page-number bits, which none of the two-page
/// tests here do.
#[cfg(test)]
pub(crate) fn mapped_cpu(entry: u32) -> Cpu {
    let mut cpu = Cpu::new(entry);
    let page = entry & 0xfffff000;
    cpu.mmu.write_tlb(false, page | 0x1, page | 1); // R+X identity, way 1
    cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    use super::super::regfile::{PS_EXCM, PS_WOE};

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

    #[test]
    fn window_check_spills_before_high_reg_write_into_full_window() {
        // Full window: 8 packed call8 frames (WS bits at every odd quad),
        // WINDOWBASE=1. An instruction touching a8 (max_ar=8 -> w=2) finds the
        // first live older frame 2 quads ahead (n=2) -> Overflow8.
        let mut cpu = mapped_cpu(0x1000);
        cpu.vecbase = 0x800;
        cpu.regs.windowbase = 1;
        cpu.regs.windowstart = 0xaaaa;
        cpu.regs.ps = PS_WOE; // window exceptions enabled, not in a handler
        let step = cpu.window_check(8, 0x1000).expect("full window must overflow");
        assert_eq!(step, Step::Exception { cause: CAUSE_WINDOW_OVERFLOW, pc: 0x880 });
        assert_eq!(cpu.regs.windowbase, 3, "rotated +n=2 onto the frame to spill");
        assert_eq!(cpu.regs.owb(), 1, "pre-rotation WINDOWBASE saved in PS.OWB");
        assert!(cpu.regs.excm(), "handler runs in exception mode");
        assert_eq!(cpu.epc1, 0x1000, "faulting instr re-executes after rfwo");
        assert_eq!(cpu.pc, 0x880, "Overflow8 vector = vecbase + 0x80");
    }

    #[test]
    fn window_check_suppressed_in_exception_mode() {
        // The spill handler itself uses s32e on a8..a15; with PS.EXCM set the
        // check must be a no-op or it would re-fault forever.
        let mut cpu = mapped_cpu(0x1000);
        cpu.regs.windowbase = 1;
        cpu.regs.windowstart = 0xaaaa;
        cpu.regs.ps = PS_WOE | PS_EXCM;
        assert!(cpu.window_check(8, 0x1000).is_none());
        assert_eq!(cpu.regs.windowbase, 1, "state untouched when suppressed");
    }

    #[test]
    fn window_check_ignores_current_frame_registers() {
        // a0..a3 (max_ar < 4 -> w=0) are always the current frame -- never an
        // overflow even in a full window.
        let mut cpu = mapped_cpu(0x1000);
        cpu.regs.windowbase = 1;
        cpu.regs.windowstart = 0xaaaa;
        cpu.regs.ps = PS_WOE;
        assert!(cpu.window_check(3, 0x1000).is_none());
    }

    #[test]
    fn unknown_step_leaves_pc_unchanged_for_reinspection() {
        // `00 00 00` decodes to Op::Unknown (op0=0 RRR with no matching
        // sub-op). The point here is specifically that `pc` does NOT advance
        // on Step::Unknown, so a caller that re-steps sees the same
        // unimplemented instruction again rather than skipping past it.
        let rom = vec![0x00, 0x00, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
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
        let mut cpu = mapped_cpu(0);
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
        let mut cpu = mapped_cpu(0);
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
        let mut cpu = mapped_cpu(0);
        assert!(matches!(cpu.step(&mut bus), Step::Unknown { .. }));
        assert_eq!(cpu.pc, 0);
    }

    // -- M2b Task 7: Cpu::translate seam + MMU EXCCAUSE + double-fault ----

    #[test]
    fn translate_returns_paddr_on_hit() {
        use crate::firmware::mmio::Bus;
        let mut cpu = Cpu::new(0);
        let mut bus = Bus::new(vec![0u8; 16]);
        // Install a DTLB mapping and translate a load through it.
        cpu.mmu.write_tlb(true, 0x0009_0000 | 0x3, 0x4000_1000 | 0);
        let paddr = cpu.translate(&mut bus, 0x4000_1abc, Access::Load).expect("hit");
        assert_eq!(paddr, 0x0009_0abc);
    }

    #[test]
    fn translate_raises_itlb_miss_as_exception() {
        use crate::firmware::mmio::Bus;
        // pc == the fetch address: a real Fetch call site translates at
        // self.pc, so this pins cpu.pc to the vaddr being fetched rather
        // than leaving them decoupled (self.pc is what raise_general_exception
        // uses for EPC1; vaddr is what becomes EXCVADDR -- for Fetch the two
        // are the same address).
        let mut cpu = Cpu::new(0x2000_0340);
        cpu.vecbase = 0x4000_0000;
        let mut bus = Bus::new(vec![0u8; 16]);
        // No mapping, no page table -> ITLB miss -> Step::Exception at the
        // unified general-exception handler (iter13; absolute, not VECBASE-rel).
        let err = cpu.translate(&mut bus, 0x2000_0340, Access::Fetch).unwrap_err();
        match err {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, 16); // INST_TLB_MISS
                assert_eq!(pc, GENERAL_EXCEPTION_HANDLER);
            }
            other => panic!("expected Exception, got {:?}", other),
        }
        assert_eq!(cpu.regs.exccause, 16);
        assert_eq!(cpu.excvaddr, 0x2000_0340);
        assert_eq!(cpu.epc1, 0x2000_0340);
    }

    #[test]
    fn double_fault_vectors_to_double_exception_offset() {
        use crate::firmware::mmio::Bus;
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x4000_0000;
        cpu.regs.set_excm(); // already in exception mode
        let mut bus = Bus::new(vec![0u8; 16]);
        let err = cpu.translate(&mut bus, 0x2000_0340, Access::Fetch).unwrap_err();
        match err {
            // A fault while PS.EXCM is set is a double fault -> VECBASE+0x31c
            // (the real rfde-terminated handler, iter13; was 0x3C0).
            Step::Exception { pc, .. } => assert_eq!(pc, 0x4000_0000 + DOUBLE_EXCEPTION_VECTOR_OFFSET),
            other => panic!("expected Exception, got {:?}", other),
        }
    }

    // -- Task 3: canonical Cpu::data_* translation-aware accessor --------

    #[test]
    fn data_accessor_low_hits_dram_high_translates() {
        use crate::firmware::mmio::Bus;
        let mut cpu = Cpu::new(0);
        let mut bus = Bus::new(vec![]);
        // Low window is covered by the varway56 way-6 ei0 identity region.
        cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
        cpu.data_write32(&mut bus, 0x0000_2278, 0x9040).expect("low D-side write");
        assert_eq!(cpu.data_read32(&mut bus, 0x0000_2278).expect("low D-side read"), 0x9040);
        assert_eq!(bus.load_local32(0x2278), 0x9040, "landed in DRAM, identity paddr");
        // High: map a data page and round-trip through translation.
        cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x4000_0000 | 0);
        cpu.data_write32(&mut bus, 0x4000_0010, 0xbeef).expect("high write translates");
        assert_eq!(cpu.data_read32(&mut bus, 0x4000_0010).expect("high read"), 0xbeef);
        assert_eq!(bus.data_load32(0x08b0_0010), 0xbeef, "translated to RAM paddr");
    }

    #[test]
    fn data_accessor_fault_propagates_as_exception() {
        use crate::firmware::mmio::Bus;
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x4000_0000;
        let mut bus = Bus::new(vec![]);
        let pc_before = cpu.pc;
        // Unmapped high page -> DTLB miss -> Step::Exception. Cpu::translate raises
        // the exception itself (not deferred to a caller), so pc is not left at some
        // ad hoc mid-op value: it vectors to the handler, and EPC1 -- not pc -- is
        // where the pre-fault pc is preserved.
        match cpu.data_read32(&mut bus, 0x5000_0000) {
            Err(Step::Exception { cause, .. }) => assert_eq!(cause, 24),
            other => panic!("expected LOAD_STORE_TLB_MISS, got {other:?}"),
        }
        assert_eq!(cpu.epc1, pc_before, "EPC1 preserves the pre-fault pc");
        assert_eq!(cpu.pc, GENERAL_EXCEPTION_HANDLER, "fault vectors pc to the handler");
    }

    // -- M2b Task 8: fetch wired through MMU translation (page-safe) -----

    #[test]
    fn fetch_translates_through_itlb() {
        use crate::firmware::mmio::Bus;
        // Map virtual code page 0x20000000 -> physical ROM 0, execute a nop.n there.
        // nop.n = `3d f0` (VERIFIED vector from op-vectors; op0=0xD narrow, 2 bytes,
        // pure pc-advance -- ideal for a fetch-path test).
        let mut rom = vec![0u8; 16];
        rom[0] = 0x3d;
        rom[1] = 0xf0;
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x2000_0000);
        // ITLB entry: VPN 0x20000000 -> paddr 0, attr 1 (R+X), way 0.
        cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x2000_0000 | 0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x2000_0002); // advanced in VIRTUAL space
    }

    #[test]
    fn fetch_miss_raises_exception_not_unknown() {
        use crate::firmware::mmio::Bus;
        let mut bus = Bus::new(vec![0u8; 16]);
        let mut cpu = Cpu::new(0x2000_0340);
        cpu.vecbase = 0x4000_0000;
        // No ITLB mapping, no page table -> ITLB miss, NOT Step::Unknown.
        match cpu.step(&mut bus) {
            Step::Exception { cause, .. } => assert_eq!(cause, 16),
            other => panic!("expected ITLB-miss Exception, got {:?}", other),
        }
    }

    #[test]
    fn narrow_op_at_page_end_does_not_fault_on_third_byte() {
        use crate::firmware::mmio::Bus;
        // A 2-byte op whose bytes are the last two of a 4KB page: the (unmapped)
        // next page must NOT be touched. Physical page 0 holds the op at 0xFFE.
        let mut rom = vec![0u8; 0x1000];
        rom[0xFFE] = 0x3d; // nop.n (VERIFIED vector)
        rom[0xFFF] = 0xf0;
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x2000_0FFE);
        // Map only virtual page 0x20000000 -> paddr 0 (R+X). The next virtual
        // page 0x20001000 is deliberately left unmapped.
        cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x2000_0000 | 0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran), "must not fault on the 3rd byte");
        assert_eq!(cpu.pc, 0x2000_1000);
    }

    // -- Task 4: level-1 interrupt delivery -------------------------------

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

    #[test]
    fn excm_masks_still_pending_interrupt_across_multi_instruction_handler() {
        // The interrupt bit stays PENDING+ENABLED for the whole handler; EXCM
        // (set on entry, cleared only by rfe) must suppress re-delivery across
        // every handler instruction. Handler at 0x2958: nop; nop; wsr.intclear
        // a2 (ack the source); rfe. Proves (a) no nested/re-entrant delivery
        // while EXCM is set even though the line is still asserted, and (b)
        // once the handler acks and returns, exactly the one original delivery
        // happened -- no spurious re-fire. This locks down the edge-vs-level
        // re-delivery contract (handler must ack via INTCLEAR).
        let mut rom = vec![0u8; 0x2964];
        rom[0x100..0x103].copy_from_slice(&[0xf0, 0x20, 0x00]); // nop (preempted body)
        rom[0x2958..0x295b].copy_from_slice(&[0xf0, 0x20, 0x00]); // handler: nop
        rom[0x295b..0x295e].copy_from_slice(&[0xf0, 0x20, 0x00]); // handler: nop
        rom[0x295e..0x2961].copy_from_slice(&[0x20, 0xe3, 0x13]); // wsr.intclear a2
        rom[0x2961..0x2964].copy_from_slice(&[0x00, 0x30, 0x00]); // rfe
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.mmu.write_tlb(false, 0x2000 | 0x1, 0x2000 | 1); // handler page
        cpu.intenable = 0b10;
        cpu.interrupt = 0b10;
        cpu.regs.write_ar(2, 0b10); // the bit wsr.intclear will ack

        // Delivery preempts the body nop.
        assert!(matches!(cpu.step(&mut bus), Step::Exception { .. }));
        assert_eq!(cpu.pc, super::GENERAL_EXCEPTION_HANDLER);
        assert!(cpu.regs.excm());

        // Two handler nops run with EXCM set and the bit STILL pending -- each
        // must be a plain Ran, never a re-delivered Exception.
        for expected_pc in [0x295b, 0x295e] {
            assert!(
                matches!(cpu.step(&mut bus), Step::Ran),
                "no re-delivery while EXCM masks the still-pending bit"
            );
            assert!(cpu.regs.excm(), "still in the handler");
            assert_eq!(cpu.interrupt, 0b10, "source still asserted, not yet acked");
            assert_eq!(cpu.pc, expected_pc);
        }

        // wsr.intclear a2 acks the source; rfe returns.
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // wsr.intclear
        assert_eq!(cpu.interrupt, 0, "handler acked the source via INTCLEAR");
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // rfe
        assert!(!cpu.regs.excm(), "rfe left exception mode");
        assert_eq!(cpu.pc, 0x100, "resumes at the preempted body instruction");

        // With the source acked, the body nop runs -- no spurious re-delivery.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x103);
    }
}
