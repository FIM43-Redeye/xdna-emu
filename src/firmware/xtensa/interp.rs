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
    /// opcode `decode.rs` itself doesn't recognize (`Op::Unknown`). `pc` is
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
    /// window handler, and report it via [`Step::Exception`].
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
    /// WINDOWBASE is NOT rotated here -- `entry` does that.
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
    /// `decode.rs` doesn't recognize returns `Step::Unknown` without advancing
    /// `pc`; a raised window exception returns `Step::Exception` and leaves
    /// `pc` at the window-exception vector.
    pub fn step(&mut self, bus: &mut Bus) -> Step {
        let pc = self.pc;
        let bytes = [bus.load8(pc), bus.load8(pc.wrapping_add(1)), bus.load8(pc.wrapping_add(2))];
        let decoded = decode::decode(&bytes, pc);
        match decoded.op {
            Op::MoviN { t, imm } | Op::Movi { t, imm } => {
                self.regs.write_ar(t, imm as u32);
            }
            Op::MovN { t, s } => {
                let v = self.regs.read_ar(s);
                self.regs.write_ar(t, v);
            }
            Op::Or { r, s, t } => {
                let v = self.regs.read_ar(s) | self.regs.read_ar(t);
                self.regs.write_ar(r, v);
            }
            Op::Extui { r, t, shiftimm, maskimm } => {
                // `maskimm` here is already the field WIDTH (1-16): decode.rs
                // resolves the raw 4-bit encoded value (0-15) to
                // `maskimm_m1 + 1` before handing it to us. So the mask is
                // `(1<<maskimm)-1`, NOT `(1<<(maskimm+1))-1` -- that second
                // formula is for the raw encoded field decode.rs has already
                // adjusted away.
                let mask = (1u32 << maskimm) - 1;
                let v = (self.regs.read_ar(t) >> shiftimm) & mask;
                self.regs.write_ar(r, v);
            }
            Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
                let addr = self.regs.read_ar(s).wrapping_add(imm);
                let v = bus.load32(addr);
                self.regs.write_ar(t, v);
            }
            Op::L32r { t, target } => {
                let v = bus.load32(target);
                self.regs.write_ar(t, v);
            }
            Op::Isync => {
                log::debug!("firmware interp: isync at 0x{:08x} (no modeled pipeline effect)", pc);
            }
            Op::Witlb { t, s } => {
                log::debug!(
                    "firmware interp: witlb a{},a{} at 0x{:08x} (no-op; mmu.rs models TLB state later)",
                    t,
                    s,
                    pc
                );
            }
            Op::Wdtlb { t, s } => {
                log::debug!(
                    "firmware interp: wdtlb a{},a{} at 0x{:08x} (no-op; mmu.rs models TLB state later)",
                    t,
                    s,
                    pc
                );
            }
            Op::Iitlb { s } => {
                log::debug!(
                    "firmware interp: iitlb a{} (=0x{:08x}) at 0x{:08x} (no-op; mmu.rs later)",
                    s,
                    self.regs.read_ar(s),
                    pc
                );
            }
            Op::Idtlb { s } => {
                log::debug!(
                    "firmware interp: idtlb a{} (=0x{:08x}) at 0x{:08x} (no-op; mmu.rs later)",
                    s,
                    self.regs.read_ar(s),
                    pc
                );
            }
            Op::Wsr { sr, t } => {
                let value = self.regs.read_ar(t);
                self.write_sr(sr, value);
            }
            Op::Dsync => {
                log::debug!("firmware interp: dsync at 0x{:08x} (no modeled pipeline effect)", pc);
            }
            Op::Jx { s } => {
                self.pc = self.regs.read_ar(s);
                return Step::Ran;
            }
            Op::Call8 { target } => {
                self.enter_call(pc, decoded.len, target);
                return Step::Ran;
            }
            Op::Callx8 { s } => {
                let target = self.regs.read_ar(s);
                self.enter_call(pc, decoded.len, target);
                return Step::Ran;
            }
            Op::Entry { s, imm } => {
                let k = self.regs.callinc();
                // Overflow: rotating WINDOWBASE forward by `k` quads would
                // expose quads (windowbase+1 ..= windowbase+k) -- if any is
                // still live (an older frame the window has wrapped back onto),
                // its registers must be spilled first.
                if k > 0 && self.regs.window_exceptions_enabled() {
                    let wb = self.regs.windowbase;
                    if (1..=k).any(|i| self.regs.frame_live(wb + i)) {
                        return self.raise_window_exception(pc, true, k);
                    }
                }
                // Read the caller's `as` (stack pointer) in the OLD window,
                // decrement by the frame size, rotate, then write the new sp
                // into the callee's `as` in the NEW window.
                let sp = self.regs.read_ar(s).wrapping_sub(imm);
                self.regs.rotate(k as i32);
                self.regs.mark_frame_live(self.regs.windowbase);
                self.regs.write_ar(s, sp);
            }
            Op::Retw | Op::RetwN => {
                // Call size (quads) the matching call recorded in a0[31:30].
                let a0 = self.regs.read_ar(0);
                let k = a0 >> 30;
                // Underflow: the frame we return into (windowbase - k) must be
                // live; if an earlier overflow spilled it, it needs filling.
                if k > 0 && self.regs.window_exceptions_enabled() {
                    let wb = self.regs.windowbase;
                    let caller = (wb as i32 - k as i32).rem_euclid(16) as u32;
                    if !self.regs.frame_live(caller) {
                        return self.raise_window_exception(pc, false, k);
                    }
                }
                self.regs.clear_frame_live(self.regs.windowbase);
                self.regs.rotate(-(k as i32));
                // Return address is 30-bit; the top 2 bits follow the current
                // PC's region (both zero for this firmware's low code space).
                self.pc = (a0 & 0x3FFF_FFFF) | (pc & 0xC000_0000);
                return Step::Ran;
            }
            Op::Unknown { word } => {
                return Step::Unknown { pc, word };
            }
        }
        self.pc = pc.wrapping_add(decoded.len as u32);
        Step::Ran
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    // movi a2, 5 (0c 52 as movi.n) then or a3,a2,a2 (20 32 20) -> a3 == 5.
    #[test]
    fn executes_movi_n_then_or() {
        let rom = vec![0x0c, 0x52, /* movi.n a2,5 */ 0x30, 0x32, 0x20 /* or a3,a2,a2 */];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 5);
    }

    #[test]
    fn executes_wide_movi() {
        // The 3-byte movi form (distinct from movi.n above). Vector from
        // M1.1: d2 a0 ac = movi a13, 172.
        let rom = vec![0xd2, 0xa0, 0xac];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(13), 172);
    }

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
        // 0xff is not a valid op0 format selector (decode.rs treats 0xE/0xF
        // as a single undecodable byte); must report Unknown, not panic.
        let rom = vec![0xff, 0xff, 0xff];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Unknown { .. }));
        assert_eq!(cpu.pc, 0);
    }

    #[test]
    fn executes_mov_n() {
        // mov.n a11, a3 -- `bd 03` (M1.1 vector).
        let rom = vec![0xbd, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0x1234);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(11), 0x1234);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l32i_n_loads_from_bus() {
        // l32i.n a4,a5,16 -- `48 45` (M1.1 vector).
        let rom = vec![0x48, 0x45];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b00010, 0xdeadbeef);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(5, 0x08b00000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xdeadbeef);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l32i_loads_from_bus() {
        // l32i a5,a2,40 -- `52 22 0a` (M1.1 vector).
        let rom = vec![0x52, 0x22, 0x0a];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b00028, 0x1122_3344); // base + 40 (0x28)
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0x08b00000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x1122_3344);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_l32r_loads_from_resolved_target() {
        // l32r a2, 0x2d158 -- `21 bd e7` @ pc 0x33262 (M1.1 vector, target
        // already verified against real firmware disassembly in decode.rs).
        // Build a ROM big enough to hold both the instruction and the
        // literal-pool value it targets, with zeros in between.
        let mut rom = vec![0u8; 0x33265];
        rom[0x2d158..0x2d158 + 4].copy_from_slice(&0xcafe_babeu32.to_le_bytes());
        rom[0x33262..0x33262 + 3].copy_from_slice(&[0x21, 0xbd, 0xe7]);
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x33262);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0xcafe_babe);
        assert_eq!(cpu.pc, 0x33265);
    }

    #[test]
    fn executes_extui_extracts_low_16_bits() {
        // extui a3,a3,0,16 -- `30 30 f4` (M1.1 vector): shiftimm=0,
        // maskimm=16 (a 16-bit field). All-ones source isolates whether the
        // mask width is `maskimm` (correct, giving 0xffff) vs. an
        // off-by-one `maskimm+1` (would give 0x1ffff).
        let rom = vec![0x30, 0x30, 0xf4];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0xffff_ffff);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 0xffff);
    }

    #[test]
    fn wsr_routes_modeled_special_registers() {
        // wsr.vecbase a2 (`20 e7 13`), wsr.ps a3 (`30 e6 13`): the modeled SRs
        // must land in cpu.vecbase / regs.ps. Oracle: SR numbers per the
        // Xtensa encoding (VECBASE=0xE7, PS=0xE6); decode verified in decode.rs.
        let rom = vec![0x20, 0xe7, 0x13, /* wsr.vecbase a2 */ 0x30, 0xe6, 0x13 /* wsr.ps a3 */];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0x0001_8000);
        cpu.regs.write_ar(3, 0x0004_0010);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, 0x0001_8000);
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.ps, 0x0004_0010);
    }

    #[test]
    fn wsr_to_unmodeled_mmu_config_is_a_no_op_that_advances() {
        // wsr.itlbcfg a2 (`20 5b 13`, boot vector): an unmodeled MMU-config
        // SR must not touch the interpreter's registers, but must still
        // advance pc like any executed instruction.
        let rom = vec![0x20, 0x5b, 0x13];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert_eq!(cpu.regs.ps, 0, "unmodeled SR write left PS untouched");
    }

    #[test]
    fn jx_jumps_to_register_target() {
        // jx a3 (`a0 03 00`, boot vector): pc becomes AR[3], no advance-past.
        let rom = vec![0xa0, 0x03, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0x2000_0340);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x2000_0340);
    }

    #[test]
    fn wdtlb_iitlb_idtlb_dsync_are_logged_no_ops() {
        // The remaining boot MMU-setup ops all advance pc with no modeled
        // register/memory effect: wdtlb a7,a4 (`70 e4 50`), iitlb a5
        // (`00 45 50`), idtlb a5 (`00 c5 50`), dsync (`30 20 00`).
        let rom = vec![0x70, 0xe4, 0x50, 0x00, 0x45, 0x50, 0x00, 0xc5, 0x50, 0x30, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        for expected_pc in [3u32, 6, 9, 12] {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
            assert_eq!(cpu.pc, expected_pc);
        }
    }

    #[test]
    fn witlb_and_isync_are_logged_no_ops() {
        // witlb a7,a4 (`70 64 50`) then isync (`00 20 00`) -- M1.1 vectors,
        // concatenated. Neither has a modeled register/memory effect; both
        // must still advance pc like any executed instruction.
        let rom = vec![0x70, 0x64, 0x50, 0x00, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 6);
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::firmware::mmio::Bus;
    use crate::firmware::xtensa::regfile::PS_WOE;

    #[test]
    fn entry_allocates_frame_and_sets_stack() {
        // entry a1, 32 (`36 41 00`). With no preceding call PS.CALLINC=0, so
        // the window doesn't rotate, but the frame is still allocated: a1
        // (stack) is decremented by the frame size.
        let rom = vec![0x36, 0x41, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(1, 0x0000_1000); // sp
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(1), 0x0000_1000 - 32);
    }

    #[test]
    fn call8_stashes_return_and_defers_rotation() {
        // Faithful Xtensa: call8 does NOT rotate WINDOWBASE (entry does). It
        // records PS.CALLINC=2 and stashes the return address -- with the call
        // size in bits 31:30 -- into a8 of the *caller's* window. Oracle
        // vector `e5 20 f9` @ pc 0x3a034 -> call8 0x33244. (The bytes must sit
        // at 0x3a034 in the image, since the interp fetches from pc.)
        let mut rom = vec![0u8; 0x3a037];
        rom[0x3a034..0x3a037].copy_from_slice(&[0xe5, 0x20, 0xf9]);
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x3a034);
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, wb0, "call8 must not rotate the window");
        assert_eq!(cpu.pc, 0x33244, "call8 jumps to target");
        assert_eq!(cpu.regs.callinc(), 2, "call8 records CALLINC=2");
        // a8 = (2<<30) | ((next_pc) & 0x3FFFFFFF); next_pc = 0x3a034 + 3.
        assert_eq!(cpu.regs.read_ar(8), 0x8000_0000 | 0x3a037);
    }

    #[test]
    fn callx8_takes_target_from_register() {
        // callx8 a5 (`e0 05 00`): register-indirect form; target comes from
        // a5. Same return/CALLINC effect as call8, no rotation.
        let rom = vec![0xe0, 0x05, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(5, 0x000a_bcd0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x000a_bcd0);
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.regs.callinc(), 2);
        assert_eq!(cpu.regs.read_ar(8), 0x8000_0000 | 0x3); // next_pc = 0 + 3
    }

    #[test]
    fn call8_entry_retw_round_trip() {
        // End-to-end with real oracle vectors: call8 @0x3a034 -> entry @0x33244
        // (`36 41 00`, frame 0x20) -> retw. Proves the rotation-at-entry model
        // threads both the callee stack pointer and the return address, and
        // that retw restores the window and returns to the post-call PC.
        let mut rom = vec![0u8; 0x3a037];
        rom[0x33244..0x33247].copy_from_slice(&[0x36, 0x41, 0x00]); // entry a1,0x20
        rom[0x33247..0x3324a].copy_from_slice(&[0x90, 0x00, 0x00]); // retw
        rom[0x3a034..0x3a037].copy_from_slice(&[0xe5, 0x20, 0xf9]); // call8 0x33244
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x3a034);
        cpu.regs.write_ar(1, 0x0000_2000); // caller sp

        // call8: no rotation, return addr into caller a8, jump to entry.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x33244);
        assert_eq!(cpu.regs.windowbase, 0);

        // entry: rotate +2 (CALLINC), allocate frame, thread sp + return addr.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 2, "entry rotates by CALLINC=2");
        assert_eq!(cpu.regs.read_ar(1), 0x0000_2000 - 0x20, "callee sp = caller sp - frame");
        assert_eq!(cpu.regs.read_ar(0), 0x8000_0000 | 0x3a037, "caller a8 becomes callee a0");
        assert_eq!(cpu.pc, 0x33247);

        // retw: restore the window and return to the instruction after call8.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 0, "retw rotates back by a0[31:30]=2");
        assert_eq!(cpu.pc, 0x3a037, "retw returns to a0[29:0]");
    }

    #[test]
    fn retw_n_restores_window_like_retw() {
        // retw.n (`1d f0`): narrow form, identical window-restore semantics.
        let rom = vec![0x1d, 0xf0];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.windowbase = 2;
        cpu.regs.windowstart = (1 << 2) | (1 << 0); // current + caller frames live
        cpu.regs.write_ar(0, (2 << 30) | 0x0000_0555); // a0: call size 2, ret 0x555
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.pc, 0x0000_0555);
    }

    #[test]
    fn entry_raises_window_overflow_and_vectors_to_stub_handler() {
        // Force overflow: with WOE enabled and CALLINC=2, entry would rotate
        // onto quad 2, which WINDOWSTART marks live -> WindowOverflow8. It
        // must vector to VECBASE + 0x80, save the restart PC, enter EXCM, and
        // NOT mutate the window. Then the stub handler at the vector runs as
        // ordinary instructions.
        let mut rom = vec![0u8; 0x1083];
        rom[0..3].copy_from_slice(&[0x36, 0x41, 0x00]); // entry a1,0x20 @ pc 0
        rom[0x1080..0x1083].copy_from_slice(&[0x00, 0x20, 0x00]); // isync (stub handler)
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x1000;
        cpu.regs.set_callinc(2);
        cpu.regs.ps |= PS_WOE;
        cpu.regs.windowbase = 0;
        cpu.regs.windowstart = (1 << 0) | (1 << 2); // quad 2 live -> overflow

        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, CAUSE_WINDOW_OVERFLOW);
                assert_eq!(pc, 0x1080, "WindowOverflow8 vector = VECBASE + 0x80");
            }
            other => panic!("expected overflow exception, got {:?}", other),
        }
        assert_eq!(cpu.pc, 0x1080, "pc left at the window-exception vector");
        assert_eq!(cpu.epc1, 0, "restartable: EPC1 = faulting entry's own pc");
        assert!(cpu.regs.excm(), "handler runs with EXCM set");
        assert_eq!(cpu.regs.windowbase, 0, "faulting entry did not rotate");
        assert_eq!(cpu.regs.windowstart, (1 << 0) | (1 << 2), "WINDOWSTART untouched");

        // The stub handler dispatches and executes as an ordinary instruction.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x1083);
    }

    #[test]
    fn retw_raises_window_underflow_and_vectors() {
        // Force underflow: WOE enabled, a0 encodes call size 2, but the frame
        // being returned to (windowbase - 2 = quad 2) is NOT live in
        // WINDOWSTART (an earlier overflow spilled it) -> WindowUnderflow8 at
        // VECBASE + 0xC0. The retw must not rotate or clear the frame.
        let rom = vec![0x90, 0x00, 0x00]; // retw @ pc 0
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x2000;
        cpu.regs.ps |= PS_WOE;
        cpu.regs.windowbase = 4;
        cpu.regs.windowstart = 1 << 4; // current frame only; caller (quad 2) spilled
        cpu.regs.write_ar(0, (2 << 30) | 0x0000_1234);

        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, CAUSE_WINDOW_UNDERFLOW);
                assert_eq!(pc, 0x20c0, "WindowUnderflow8 vector = VECBASE + 0xC0");
            }
            other => panic!("expected underflow exception, got {:?}", other),
        }
        assert_eq!(cpu.pc, 0x20c0);
        assert_eq!(cpu.epc1, 0, "restartable: EPC1 = faulting retw's own pc");
        assert!(cpu.regs.excm());
        assert_eq!(cpu.regs.windowbase, 4, "faulting retw did not rotate");
        assert_eq!(cpu.regs.windowstart, 1 << 4, "frame not cleared on underflow");
    }

    #[test]
    fn no_overflow_when_woe_disabled() {
        // The exact WINDOWSTART that overflowed above must NOT raise when WOE
        // is clear: detection is gated on PS.WOE. entry rotates normally.
        let rom = vec![0x36, 0x41, 0x00]; // entry a1,0x20
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.set_callinc(2); // WOE left clear
        cpu.regs.windowbase = 0;
        cpu.regs.windowstart = (1 << 0) | (1 << 2);
        cpu.regs.write_ar(1, 0x0000_1000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 2, "entry still rotates when WOE off");
    }
}
