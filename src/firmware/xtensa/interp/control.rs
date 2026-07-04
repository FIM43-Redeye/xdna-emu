//! Windowed-call ABI execute: `call8`, `callx8`, `entry`, `retw`/`retw.n`,
//! and `jx`.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`Jx`/`Call8`/`Callx8`/
/// `Entry`/`Retw`/`RetwN`); `None` otherwise, so `step()` tries the next
/// category. Unlike `mem`/`arith`/`system`, these ops set `cpu.pc` themselves
/// (a plain jump target, `enter_call`'s target, a window-exception vector, or
/// the windowed return address) rather than falling through to a common
/// `pc += len` tail.
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::Jx { s } => {
            cpu.pc = cpu.regs.read_ar(*s);
            Some(Step::Ran)
        }
        Op::Call8 { target } => {
            cpu.enter_call(pc, len, *target);
            Some(Step::Ran)
        }
        Op::Callx8 { s } => {
            let target = cpu.regs.read_ar(*s);
            cpu.enter_call(pc, len, target);
            Some(Step::Ran)
        }
        Op::Entry { s, imm } => {
            let k = cpu.regs.callinc();
            // Overflow: rotating WINDOWBASE forward by `k` quads would
            // expose quads (windowbase+1 ..= windowbase+k) -- if any is
            // still live (an older frame the window has wrapped back onto),
            // its registers must be spilled first.
            if k > 0 && cpu.regs.window_exceptions_enabled() {
                let wb = cpu.regs.windowbase;
                if (1..=k).any(|i| cpu.regs.frame_live(wb + i)) {
                    return Some(cpu.raise_window_exception(pc, true, k));
                }
            }
            // Read the caller's `as` (stack pointer) in the OLD window,
            // decrement by the frame size, rotate, then write the new sp
            // into the callee's `as` in the NEW window.
            let sp = cpu.regs.read_ar(*s).wrapping_sub(*imm);
            cpu.regs.rotate(k as i32);
            cpu.regs.mark_frame_live(cpu.regs.windowbase);
            cpu.regs.write_ar(*s, sp);
            cpu.pc = pc.wrapping_add(len as u32);
            Some(Step::Ran)
        }
        Op::Retw | Op::RetwN => {
            // Call size (quads) the matching call recorded in a0[31:30].
            let a0 = cpu.regs.read_ar(0);
            let k = a0 >> 30;
            // Underflow: the frame we return into (windowbase - k) must be
            // live; if an earlier overflow spilled it, it needs filling.
            if k > 0 && cpu.regs.window_exceptions_enabled() {
                let wb = cpu.regs.windowbase;
                let caller = (wb as i32 - k as i32).rem_euclid(16) as u32;
                if !cpu.regs.frame_live(caller) {
                    return Some(cpu.raise_window_exception(pc, false, k));
                }
            }
            cpu.regs.clear_frame_live(cpu.regs.windowbase);
            cpu.regs.rotate(-(k as i32));
            // Return address is 30-bit; the top 2 bits follow the current
            // PC's region (both zero for this firmware's low code space).
            cpu.pc = (a0 & 0x3FFF_FFFF) | (pc & 0xC000_0000);
            Some(Step::Ran)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Cpu, Step};
    use crate::firmware::mmio::Bus;

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
}

#[cfg(test)]
mod window_tests {
    use super::super::{Cpu, Step, CAUSE_WINDOW_OVERFLOW, CAUSE_WINDOW_UNDERFLOW};
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
