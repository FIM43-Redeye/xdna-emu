//! Xtensa interpreter core: fetch/decode/execute over the `Bus` for the
//! non-windowed base-ISA subset. Windowed call ops (`entry`, `call8`, `retw`,
//! ...) are already decoded by `decode.rs` but their windowed-ABI execution
//! semantics (`RegFile::rotate`, window overflow/underflow) land in M1.5 --
//! here they fall through to `Step::Unknown`, same as any opcode `decode.rs`
//! itself doesn't recognize.

use super::decode::{self, Op};
use super::regfile::RegFile;
use crate::firmware::Bus;

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
    /// An Xtensa exception was raised at `pc` with cause code `cause`.
    Exception { cause: u32, pc: u32 },
    /// The instruction at `pc` isn't executed by this interpreter yet --
    /// either an opcode `decode.rs` itself doesn't recognize (`Op::Unknown`),
    /// or a windowed-call op deferred to M1.5. `pc` is left unchanged (the
    /// instruction was not executed), so re-stepping re-reports the same
    /// unimplemented instruction rather than silently skipping past it.
    /// `word` is the raw fetched bytes, packed little-endian, for
    /// diagnostics.
    Unknown { pc: u32, word: u32 },
}

/// The Xtensa interpreter core: program counter plus the windowed register file.
pub struct Cpu {
    /// Program counter -- byte address into the firmware's address space.
    pub pc: u32,
    /// Windowed register file (AR0-63, WINDOWBASE/WINDOWSTART, SAR, PS).
    pub regs: RegFile,
}

impl Cpu {
    /// Create a CPU with `pc` at `entry` and a fresh, zeroed register file.
    pub fn new(entry: u32) -> Self {
        Self { pc: entry, regs: RegFile::new() }
    }

    /// Fetch, decode, and execute one instruction from `bus` at `self.pc`.
    ///
    /// Advances `pc` by the decoded instruction length on every executed op.
    /// Implements the non-windowed base-ISA subset (`movi`/`movi.n`/`mov.n`,
    /// `l32i`/`l32i.n`/`l32r`, `or`, `extui`, `isync`/`witlb` as logged
    /// no-ops); windowed-call ops and anything `decode.rs` doesn't recognize
    /// return `Step::Unknown` without advancing `pc`.
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
            Op::Entry { .. } | Op::Call8 { .. } => {
                return Step::Unknown { pc, word: word_of(&bytes) };
            }
            Op::Unknown { word } => {
                return Step::Unknown { pc, word };
            }
        }
        self.pc = pc.wrapping_add(decoded.len as u32);
        Step::Ran
    }
}

/// Pack 3 fetched bytes into a little-endian diagnostic word (the same
/// packing `decode::decode` uses internally for its own `Op::Unknown` word).
fn word_of(bytes: &[u8; 3]) -> u32 {
    (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16)
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
    fn windowed_op_returns_unknown_until_m1_5() {
        let rom = vec![0x36, 0x41, 0x00]; // entry a1, 32
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Unknown { .. }));
    }

    #[test]
    fn unknown_step_leaves_pc_unchanged_for_reinspection() {
        // Same `entry` vector as above; the point here is specifically that
        // `pc` does NOT advance on Step::Unknown, so a caller that re-steps
        // sees the same unimplemented instruction again rather than skipping
        // past it silently.
        let rom = vec![0x36, 0x41, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        match cpu.step(&mut bus) {
            Step::Unknown { pc, word } => {
                assert_eq!(pc, 0);
                assert_eq!(word, 0x00_41_36);
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
