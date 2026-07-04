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
//! `step()`, and the window-exception raise stay here as shared machinery;
//! `step()`'s dispatch tries each category's `exec` in turn (`None` = not
//! this category's op), preserving the exact original per-op behavior.

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

/// Diagnostic cause code for a window-overflow exception. Xtensa vectors
/// window exceptions through dedicated VECBASE-relative addresses rather than
/// reporting them in EXCCAUSE, so this is our own identifier for the
/// [`Step::Exception`] `cause` channel, not an architectural EXCCAUSE value.
pub const CAUSE_WINDOW_OVERFLOW: u32 = 0x1000;
/// Diagnostic cause code for a window-underflow exception (see
/// [`CAUSE_WINDOW_OVERFLOW`]).
pub const CAUSE_WINDOW_UNDERFLOW: u32 = 0x1001;

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
}

impl Cpu {
    /// Create a CPU with `pc` at `entry`, a fresh zeroed register file, and a
    /// zero vector base (boot sets `vecbase` before enabling window exceptions).
    pub fn new(entry: u32) -> Self {
        Self { pc: entry, regs: RegFile::new(), vecbase: 0, epc1: 0 }
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
    /// the interpreter tracks (SAR/WINDOWBASE/WINDOWSTART/EPC1/PS/VECBASE);
    /// any other SR (the MMU-config registers ITLBCFG/DTLBCFG/PTEVADDR/RASID
    /// the boot sequence programs, and everything not yet modeled) is logged
    /// and dropped -- their effect is on hardware state this phase doesn't
    /// simulate (the MMU is `mmu.rs`/M2), not on the interpreter's registers.
    /// Called by `system::exec`'s `Wsr` handling.
    fn write_sr(&mut self, sr: u8, value: u32) {
        match sr {
            SR_SAR => self.regs.sar = value,
            SR_WINDOWBASE => self.regs.windowbase = value % NUM_FRAMES,
            SR_WINDOWSTART => self.regs.windowstart = value,
            SR_EPC1 => self.epc1 = value,
            SR_PS => self.regs.ps = value,
            SR_VECBASE => self.vecbase = value,
            _ => log::debug!(
                "firmware interp: wsr.0x{:02x} = 0x{:08x} (unmodeled SR; logged no-op)",
                sr,
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
    pub fn step(&mut self, bus: &mut Bus) -> Step {
        let pc = self.pc;
        let bytes = [bus.load8(pc), bus.load8(pc.wrapping_add(1)), bus.load8(pc.wrapping_add(2))];
        let decoded = decode::decode(&bytes, pc);
        if let Op::Unknown { word } = decoded.op {
            return Step::Unknown { pc, word };
        }
        mem::exec(self, bus, &decoded.op, pc, decoded.len)
            .or_else(|| arith::exec(self, bus, &decoded.op, pc, decoded.len))
            .or_else(|| control::exec(self, bus, &decoded.op, pc, decoded.len))
            .or_else(|| system::exec(self, bus, &decoded.op, pc, decoded.len))
            .or_else(|| branch::exec(self, bus, &decoded.op, pc, decoded.len))
            .unwrap_or_else(|| panic!("decoded op {:?} not handled by any category", decoded.op))
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
