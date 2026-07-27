//! Windowed-call ABI execute: `call8`, `callx8`, `entry`, `retw`/`retw.n`,
//! `jx`, exception returns, the zero-overhead-loop setup ops `loop`/`loopnez` (M2a Task 7 --
//! the loop-BACK itself, on retirement at LEND, lives in `interp::Cpu::step`,
//! not here; this module only handles the `loop`/`loopnez` instructions that
//! arm the loop registers), and (M2a Task 8) the software window-spill
//! primitive `rotw`, the wait instruction `waiti`, and the plain
//! (non-windowed) call ABI `call0`/`ret.n`.

use super::{Cpu, Step, WaitReason};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`Jx`/`Call8`/`Callx8`/
/// `Entry`/`Retw`/`RetwN`/`Rfe`/`Rfde`/`Loop`/`Loopnez`/`Rotw`/`Waiti`/
/// `Call0`/`RetN`);
/// `None` otherwise, so `step()` tries the next category. Unlike `mem`/
/// `arith`/`system`, these ops set `cpu.pc` themselves (a plain jump target,
/// `enter_call`'s target, a window-exception vector, the windowed or plain
/// return address, the loop body's entry point or, for a skipped
/// `loopnez`, LEND) rather than falling through to a common `pc += len`
/// tail -- except `Waiti`, which deliberately leaves `cpu.pc` untouched
/// (see its match arm).
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::Jx { s } => {
            cpu.pc = cpu.regs.read_ar(*s);
            Some(Step::Ran)
        }
        Op::Call4 { target } => {
            cpu.enter_call(pc, len, *target, 1);
            Some(Step::Ran)
        }
        Op::Call8 { target } => {
            cpu.enter_call(pc, len, *target, 2);
            Some(Step::Ran)
        }
        Op::Call12 { target } => {
            cpu.enter_call(pc, len, *target, 3);
            Some(Step::Ran)
        }
        Op::Callx0 { s } => {
            // Non-windowed indirect call: mirror Call0, but the target comes
            // from a register, so it must be read BEFORE a0 is overwritten
            // (s could be 0, aliasing a0 itself).
            let target = cpu.regs.read_ar(*s);
            cpu.regs.write_ar(0, pc.wrapping_add(len as u32));
            cpu.pc = target;
            Some(Step::Ran)
        }
        Op::Callx4 { s } => {
            let target = cpu.regs.read_ar(*s);
            cpu.enter_call(pc, len, target, 1);
            Some(Step::Ran)
        }
        Op::Callx8 { s } => {
            let target = cpu.regs.read_ar(*s);
            cpu.enter_call(pc, len, target, 2);
            Some(Step::Ran)
        }
        Op::Callx12 { s } => {
            let target = cpu.regs.read_ar(*s);
            cpu.enter_call(pc, len, target, 3);
            Some(Step::Ran)
        }
        Op::Entry { s, imm } => {
            let k = cpu.regs.callinc();
            // Overflow: rotating WINDOWBASE forward by `k` quads would
            // expose quads (windowbase+1 ..= windowbase+k) -- if any is
            // still live (an older frame the window has wrapped back onto),
            // its registers must be spilled first.
            if k > 0 && cpu.regs.window_exceptions_enabled() {
                // QEMU window_check: `n` = quads to the frame that must spill,
                // `spill_quads` = its size (Overflow4/8/12). Rotate WINDOWBASE
                // forward by `n` so the handler's a0.. cover that frame.
                if let Some((n, spill_quads)) = cpu.regs.overflow_check(k) {
                    return Some(cpu.raise_window_exception(pc, true, n as i32, spill_quads));
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
            // The returning frame is done -> clear its live bit first (QEMU
            // translate_retw does this unconditionally, before the check).
            cpu.regs.clear_frame_live(cpu.regs.windowbase);
            // Underflow: the frame we return into (windowbase - k) must be
            // live; if an earlier overflow spilled it, rotate WINDOWBASE back
            // by `k` and vector to the Underflow<4k> handler to refill it.
            if k > 0 && cpu.regs.window_exceptions_enabled() {
                let wb = cpu.regs.windowbase;
                let caller = (wb as i32 - k as i32).rem_euclid(16) as u32;
                if !cpu.regs.frame_live(caller) {
                    return Some(cpu.raise_window_exception(pc, false, -(k as i32), k));
                }
            }
            cpu.regs.rotate(-(k as i32));
            // Return address is 30-bit; the top 2 bits follow the current
            // PC's region (both zero for this firmware's low code space).
            cpu.pc = (a0 & 0x3FFF_FFFF) | (pc & 0xC000_0000);
            Some(Step::Ran)
        }
        Op::Rfwo | Op::Rfwu => {
            // Return from window overflow/underflow (QEMU `translate_rfw`):
            // leave exception mode, then update WINDOWSTART for the frame the
            // handler just processed at the CURRENT WINDOWBASE -- rfwo CLEARS
            // its bit (the frame was spilled to memory, no longer in the
            // register file), rfwu SETS it (the frame was restored). Finally
            // restore WINDOWBASE from PS.OWB and resume at EPC1 (the faulting
            // `entry`/`retw`, which now re-executes without re-faulting).
            cpu.regs.clear_excm();
            let wb = cpu.regs.windowbase;
            if matches!(op, Op::Rfwo) {
                cpu.regs.clear_frame_live(wb);
            } else {
                cpu.regs.mark_frame_live(wb);
            }
            cpu.regs.windowbase = cpu.regs.owb();
            cpu.pc = cpu.epc1;
            Some(Step::Ran)
        }
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
        Op::Rfde => {
            // QEMU translate_rfde: return to the double-fault restart PC.
            // EXCM deliberately remains set because this resumes the outer
            // exception handler; its later rfe performs the eventual clear.
            cpu.pc = cpu.depc;
            Some(Step::Ran)
        }
        Op::Loop { s, end } => {
            cpu.regs.lcount = cpu.regs.read_ar(*s).wrapping_sub(1);
            cpu.regs.lbeg = pc.wrapping_add(len as u32);
            cpu.regs.lend = *end;
            // Unconditional fall-through -- unlike loopnez/loopgtz, plain
            // `loop` has no zero-trip-count skip check (if AR[s]==0, LCOUNT
            // wraps to u32::MAX and the body still runs, a real Xtensa
            // hardware footgun compilers avoid by using loopnez when a zero
            // count is possible).
            cpu.pc = pc.wrapping_add(len as u32);
            Some(Step::Ran)
        }
        Op::Loopnez { s, end } => {
            let count = cpu.regs.read_ar(*s);
            // LBEG/LEND/LCOUNT are set UNCONDITIONALLY, before the zero
            // check -- matches QEMU `translate_loop` (the SR writes are
            // emitted ahead of the `AR[s]==0` conditional branch, not
            // gated on the body path) and real Xtensa hardware, not a
            // QEMU-only artifact: see `Op::Loopnez`'s doc in decode/mod.rs.
            cpu.regs.lcount = count.wrapping_sub(1);
            cpu.regs.lbeg = pc.wrapping_add(len as u32);
            cpu.regs.lend = *end;
            cpu.pc = if count == 0 {
                *end // skip the body entirely
            } else {
                pc.wrapping_add(len as u32) // fall through into the body
            };
            Some(Step::Ran)
        }
        Op::Rotw { imm } => {
            // Software window-spill primitive: just rotates WINDOWBASE by
            // the signed delta (RegFile::rotate already wraps mod
            // NUM_FRAMES) -- no overflow/underflow check, unlike
            // entry/retw. This firmware's own spill-all routine (rotw +
            // s32i.n) is the software substitute for the architectural
            // window-exception handlers this interpreter otherwise models.
            cpu.regs.rotate(*imm);
            cpu.pc = pc.wrapping_add(len as u32);
            Some(Step::Ran)
        }
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
        Op::Call0 { target } => {
            // Plain (non-windowed) call: stash the return address plainly
            // in a0 (no call-size packing into bits 31:30 -- that packing
            // is specific to call8/callx8's windowed-ABI a0/a8 handoff) and
            // jump. PS.CALLINC is NOT touched -- there is no `entry` on the
            // other end of a call0, so nothing needs to rotate the window.
            let ret = pc.wrapping_add(len as u32);
            cpu.regs.write_ar(0, ret);
            cpu.pc = *target;
            Some(Step::Ran)
        }
        Op::RetN => {
            // Plain (non-windowed) return: pc = AR[0], no window rotation
            // (unlike Retw/RetwN, which also rotate WINDOWBASE by
            // a0[31:30]) -- the a0 value here is a full 32-bit address, not
            // packed with a call size.
            cpu.pc = cpu.regs.read_ar(0);
            Some(Step::Ran)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{mapped_cpu, Step};
    use crate::firmware::mmio::Bus;

    #[test]
    fn jx_jumps_to_register_target() {
        // jx a3 (`a0 03 00`, boot vector): pc becomes AR[3], no advance-past.
        let rom = vec![0xa0, 0x03, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0x2000_0340);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x2000_0340);
    }
}

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

    #[test]
    fn rfde_resumes_at_depc_without_clearing_excm() {
        let rom = vec![0x00, 0x32, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_excm();
        cpu.depc = 0x1234_5678;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_ne!(cpu.regs.ps & PS_EXCM, 0, "rfde remains in exception mode");
        assert_eq!(cpu.pc, 0x1234_5678, "rfde resumes at DEPC");
    }
}

#[cfg(test)]
mod rotw_tests {
    use super::super::{mapped_cpu, Step};
    use crate::firmware::mmio::Bus;

    #[test]
    fn rotw_moves_windowbase_by_the_signed_immediate() {
        // rotw 0x1 (`10 80 40`, oracle vector): WINDOWBASE moves by +1, the
        // decoded simm4 -- no window-overflow/underflow check (this firmware
        // spills/fills its own windows in software via rotw, unlike
        // entry/retw's architectural detection).
        let rom = vec![0x10, 0x80, 0x40];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.windowbase = 3;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 4);
        assert_eq!(cpu.pc, 3, "falls through, ordinary pc+len advance");
    }

    #[test]
    fn rotw_negative_immediate_wraps_mod_num_frames() {
        // rotw -1: same field layout as the +1 oracle vector (`10 80 40`)
        // with t (simm4) flipped to 0xF (sign_extend4(0xF) == -1) --
        // confirmed against xtensa-lx106-elf-objdump, which decodes it the
        // same way it fails on the +1 vector (`excw`, the windowed-register
        // option gap this file's module doc already documents for
        // entry/retw/callx8/jx), so the field position is pinned by the
        // real oracle vector, not invented. From WINDOWBASE=0, -1 must wrap
        // to NUM_FRAMES-1 (15), exercising RegFile::rotate's mod-16 wrap.
        let rom = vec![0xf0, 0x80, 0x40];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.windowbase = 0;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 15, "0 + (-1) mod 16 == 15");
    }
}

#[cfg(test)]
mod waiti_tests {
    use super::super::{mapped_cpu, Step, WaitReason};
    use crate::firmware::mmio::Bus;

    #[test]
    fn waiti_sets_intlevel_and_yields_after_advancing_pc() {
        // waiti 0 (`00 70 00`, oracle vector): sets PS.INTLEVEL, RETIRES (pc
        // advances past the instruction), and returns Step::Wait. Real
        // hardware (QEMU HELPER(waiti)) advances pc then halts until a
        // deliverable interrupt arrives -- it does not stall in place. PC
        // parked after waiti is load-bearing: EPC1 on a later interrupt must
        // point at the next instruction, not back onto the waiti.
        let rom = vec![0x00, 0x70, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_intlevel(3); // pre-existing level, must be overwritten
        match cpu.step(&mut bus) {
            Step::Wait(reason) => assert_eq!(reason, WaitReason::Waiti),
            other => panic!("expected Step::Wait(Waiti), got {:?}", other),
        }
        assert_eq!(cpu.regs.intlevel(), 0, "PS.INTLEVEL set from the decoded imm4");
        assert_eq!(cpu.pc, 3, "waiti now retires -- advances PC past itself");
    }

    #[test]
    fn waiti_nonzero_level_is_recorded() {
        // Same instruction family, nonzero imm4: `00 75 00` -> waiti 5 (s is
        // byte1's low nibble, so t(byte0 hi)=0/r(byte1 hi)=7 stay fixed at
        // the oracle vector's values while s(byte1 lo)=5 carries the level).
        let rom = vec![0x00, 0x75, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        match cpu.step(&mut bus) {
            Step::Wait(reason) => assert_eq!(reason, WaitReason::Waiti),
            other => panic!("expected Step::Wait(Waiti), got {:?}", other),
        }
        assert_eq!(cpu.regs.intlevel(), 5);
        assert_eq!(cpu.pc, 3, "waiti now retires -- advances PC past itself");
    }

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
}

#[cfg(test)]
mod plain_call_tests {
    use super::super::{mapped_cpu, Step};
    use crate::firmware::mmio::Bus;

    #[test]
    fn call0_stashes_return_in_a0_and_jumps_no_window_effect() {
        // call0 (`85 ec ff` @ pc 0xe1ce -> target 0xe098, firmware oracle
        // vector): AR[0] = pc+3 (plain, no call-size packing into bits
        // 31:30 the way call8/callx8's enter_call does), pc = target, and
        // CALLINC/WINDOWBASE are left untouched -- there is no matching
        // `entry` for a call0/ret.n pair.
        let mut rom = vec![0u8; 0xe1d1];
        rom[0xe1ce..0xe1d1].copy_from_slice(&[0x85, 0xec, 0xff]);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0xe1ce);
        let callinc0 = cpu.regs.callinc();
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0xe098, "call0 jumps to the decoded target");
        assert_eq!(cpu.regs.read_ar(0), 0xe1ce + 3, "AR[0] = pc+3, no window-size packing");
        assert_eq!(cpu.regs.callinc(), callinc0, "call0 does not touch PS.CALLINC");
        assert_eq!(cpu.regs.windowbase, wb0, "call0 does not rotate the window");
    }

    #[test]
    fn ret_n_returns_to_a0_without_window_rotation() {
        // ret.n (`0d f0`): pc = AR[0], plainly -- no window rotate (unlike
        // retw.n, which also rotates WINDOWBASE by a0[31:30]).
        let rom = vec![0x0d, 0xf0];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.windowbase = 5;
        cpu.regs.write_ar(0, 0x0000_1234);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x0000_1234);
        assert_eq!(cpu.regs.windowbase, 5, "ret.n does not rotate the window");
    }

    #[test]
    fn call0_ret_n_round_trip() {
        // End-to-end with the real call0 oracle vector: call0 @0xe1ce ->
        // (stub body, immediately returns) -> ret.n. Proves AR[0] threads
        // the return address through to a real ret.n.
        let mut rom = vec![0u8; 0xe1d1];
        rom[0xe098..0xe09a].copy_from_slice(&[0x0d, 0xf0]); // ret.n at the target
        rom[0xe1ce..0xe1d1].copy_from_slice(&[0x85, 0xec, 0xff]); // call0 0xe098
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0xe1ce);

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0xe098);

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0xe1ce + 3, "ret.n returns to the instruction after call0");
    }

    // -- M2c Phase 2 iter1: callx0, the non-windowed indirect call ---------

    #[test]
    fn callx0_reads_target_before_overwriting_a0_when_they_alias() {
        // callx0 a0 (`c0 00 00`): s==0 makes the target register alias a0
        // itself. callx0 must read AR[s] BEFORE writing the return address
        // into a0, or the target would be clobbered by its own return
        // address before the jump reads it.
        let rom = vec![0xc0, 0x00, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(0, 0x0000_2000); // target, aliasing a0
        let callinc0 = cpu.regs.callinc();
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x0000_2000, "jumped to the pre-overwrite value of AR[s]/a0");
        assert_eq!(cpu.regs.callinc(), callinc0, "callx0 does not touch PS.CALLINC");
        assert_eq!(cpu.regs.windowbase, wb0, "callx0 does not rotate the window");
    }

    #[test]
    fn callx0_writes_full_32bit_return_no_k_packing() {
        // callx0 a5 (`c0 05 00`), non-aliasing target register: pc = AR[5],
        // and a0 gets the PLAIN pc+len return address -- no call-size
        // packing into bits 31:30 the way call4/call8/call12/callx*'s
        // enter_call does (there is no matching entry/retw for a callx0).
        let rom = vec![0xc0, 0x05, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0x0000_9000);
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x0000_9000, "callx0 jumps to AR[s]");
        assert_eq!(cpu.regs.read_ar(0), 0 + 3, "a0 = full 32-bit pc+len, no call-size packing");
        assert_eq!(cpu.regs.windowbase, wb0, "callx0 does not rotate the window");
    }
}

#[cfg(test)]
mod window_tests {
    use super::super::{mapped_cpu, Step, CAUSE_WINDOW_OVERFLOW, CAUSE_WINDOW_UNDERFLOW};
    use crate::firmware::mmio::Bus;
    use crate::firmware::xtensa::regfile::PS_WOE;

    #[test]
    fn entry_allocates_frame_and_sets_stack() {
        // entry a1, 32 (`36 41 00`). With no preceding call PS.CALLINC=0, so
        // the window doesn't rotate, but the frame is still allocated: a1
        // (stack) is decremented by the frame size.
        let rom = vec![0x36, 0x41, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
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
        let mut cpu = mapped_cpu(0x3a034);
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
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0x000a_bcd0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x000a_bcd0);
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.regs.callinc(), 2);
        assert_eq!(cpu.regs.read_ar(8), 0x8000_0000 | 0x3); // next_pc = 0 + 3
    }

    // -- M2c Phase 2 iter1: windowed-call family completion -----------------

    #[test]
    fn call4_stashes_return_and_sets_callinc_1() {
        // call4 (`d5 20 f9` @ pc 0x3a034 -> target 0x33244): identical bytes
        // to the call8 oracle vector with only the n field (byte0 bits 5:4)
        // flipped from 2 to 1 -- the shared CALLN target formula discards
        // byte0's bits 0-5 entirely, so the target is unchanged. The callee's
        // a0 is a[4*k] = a4 (k=1), not call8's a8.
        let mut rom = vec![0u8; 0x3a037];
        rom[0x3a034..0x3a037].copy_from_slice(&[0xd5, 0x20, 0xf9]);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x3a034);
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, wb0, "call4 must not rotate the window");
        assert_eq!(cpu.pc, 0x33244, "call4 jumps to target");
        assert_eq!(cpu.regs.callinc(), 1, "call4 records CALLINC=1");
        // a4 = (1<<30) | ((next_pc) & 0x3FFFFFFF); next_pc = 0x3a034 + 3.
        assert_eq!(cpu.regs.read_ar(4), 0x4000_0000 | 0x3a037);
    }

    #[test]
    fn call12_stashes_return_and_sets_callinc_3() {
        // call12 (`f5 20 f9` @ pc 0x3a034 -> target 0x33244): same target
        // formula, n=3. The callee's a0 is a[4*3] = a12.
        let mut rom = vec![0u8; 0x3a037];
        rom[0x3a034..0x3a037].copy_from_slice(&[0xf5, 0x20, 0xf9]);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x3a034);
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, wb0, "call12 must not rotate the window");
        assert_eq!(cpu.pc, 0x33244, "call12 jumps to target");
        assert_eq!(cpu.regs.callinc(), 3, "call12 records CALLINC=3");
        // a12 = (3<<30) | ((next_pc) & 0x3FFFFFFF); next_pc = 0x3a034 + 3.
        assert_eq!(cpu.regs.read_ar(12), 0xC000_0000 | 0x3a037);
    }

    #[test]
    fn callx4_takes_target_from_register() {
        // callx4 a5 (`d0 05 00`): register-indirect form, k=1; target comes
        // from a5. Same return/CALLINC(1) effect as call4, no rotation.
        let rom = vec![0xd0, 0x05, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0x000a_bcd0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x000a_bcd0);
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.regs.callinc(), 1);
        assert_eq!(cpu.regs.read_ar(4), 0x4000_0000 | 0x3); // next_pc = 0 + 3
    }

    #[test]
    fn callx12_takes_target_from_register() {
        // callx12 a5 (`f0 05 00`): register-indirect form, k=3; target comes
        // from a5. Same return/CALLINC(3) effect as call12, no rotation.
        let rom = vec![0xf0, 0x05, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0x000a_bcd0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x000a_bcd0);
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.regs.callinc(), 3);
        assert_eq!(cpu.regs.read_ar(12), 0xC000_0000 | 0x3); // next_pc = 0 + 3
    }

    #[test]
    fn call4_entry_retw_round_trip_rotates_by_1() {
        // End-to-end with the k=1 vector: call4 @0x3a034 -> entry @0x33244
        // (the same entry oracle vector, frame 0x20) -> retw. Proves entry
        // rotates WINDOWBASE by CALLINC=1 (not call8's 2), the caller's a4
        // becomes the callee's a0, and retw reads k=1 back out of
        // a0[31:30] to rotate back by exactly 1 (not always -2).
        let mut rom = vec![0u8; 0x3a037];
        rom[0x33244..0x33247].copy_from_slice(&[0x36, 0x41, 0x00]); // entry a1,0x20
        rom[0x33247..0x3324a].copy_from_slice(&[0x90, 0x00, 0x00]); // retw
        rom[0x3a034..0x3a037].copy_from_slice(&[0xd5, 0x20, 0xf9]); // call4 0x33244
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x3a034);
        cpu.mmu.write_tlb(false, 0x33000 | 0x1, 0x33000 | 0);
        cpu.regs.write_ar(1, 0x0000_2000); // caller sp

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // call4
        assert_eq!(cpu.pc, 0x33244);
        assert_eq!(cpu.regs.windowbase, 0);

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // entry
        assert_eq!(cpu.regs.windowbase, 1, "entry rotates by CALLINC=1");
        assert_eq!(cpu.regs.read_ar(1), 0x0000_2000 - 0x20, "callee sp = caller sp - frame");
        assert_eq!(cpu.regs.read_ar(0), 0x4000_0000 | 0x3a037, "caller a4 becomes callee a0");
        assert_eq!(cpu.pc, 0x33247);

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // retw
        assert_eq!(cpu.regs.windowbase, 0, "retw rotates back by a0[31:30]=1");
        assert_eq!(cpu.pc, 0x3a037, "retw returns to a0[29:0]");
    }

    #[test]
    fn call12_entry_retw_round_trip_rotates_by_3() {
        // Same shape as the k=1 round trip, with call12 (k=3): entry rotates
        // by 3, the caller's a12 becomes the callee's a0, retw rotates back
        // by 3.
        let mut rom = vec![0u8; 0x3a037];
        rom[0x33244..0x33247].copy_from_slice(&[0x36, 0x41, 0x00]); // entry a1,0x20
        rom[0x33247..0x3324a].copy_from_slice(&[0x90, 0x00, 0x00]); // retw
        rom[0x3a034..0x3a037].copy_from_slice(&[0xf5, 0x20, 0xf9]); // call12 0x33244
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x3a034);
        cpu.mmu.write_tlb(false, 0x33000 | 0x1, 0x33000 | 0);
        cpu.regs.write_ar(1, 0x0000_2000); // caller sp

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // call12
        assert_eq!(cpu.pc, 0x33244);

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // entry
        assert_eq!(cpu.regs.windowbase, 3, "entry rotates by CALLINC=3");
        assert_eq!(cpu.regs.read_ar(1), 0x0000_2000 - 0x20, "callee sp = caller sp - frame");
        assert_eq!(cpu.regs.read_ar(0), 0xC000_0000 | 0x3a037, "caller a12 becomes callee a0");
        assert_eq!(cpu.pc, 0x33247);

        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // retw
        assert_eq!(cpu.regs.windowbase, 0, "retw rotates back by a0[31:30]=3");
        assert_eq!(cpu.pc, 0x3a037, "retw returns to a0[29:0]");
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
        let mut cpu = mapped_cpu(0x3a034);
        // call8 jumps to 0x33244, in a DIFFERENT 4KB page than the entry
        // page mapped_cpu already covers -- map it too (way 0, ei=3, no
        // collision with the entry page's ei=2 slot).
        cpu.mmu.write_tlb(false, 0x33000 | 0x1, 0x33000 | 0);
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
        let mut cpu = mapped_cpu(0);
        cpu.regs.windowbase = 2;
        cpu.regs.windowstart = (1 << 2) | (1 << 0); // current + caller frames live
        cpu.regs.write_ar(0, (2 << 30) | 0x0000_0555); // a0: call size 2, ret 0x555
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 0);
        assert_eq!(cpu.pc, 0x0000_0555);
    }

    #[test]
    fn entry_raises_window_overflow_and_vectors_to_stub_handler() {
        // Force overflow: a full call8-nested window (WINDOWSTART bits every 2
        // quads) with CALLINC=2 -- entry's forward rotation wraps onto a live
        // frame -> WindowOverflow8 at VECBASE+0x80. Per QEMU window_check the
        // exception ROTATES WINDOWBASE forward to the frame being spilled (n=2),
        // saves the old base in PS.OWB, saves the restart PC, and enters EXCM.
        // Then the stub handler at the vector runs as ordinary instructions.
        let mut rom = vec![0u8; 0x1083];
        rom[0..3].copy_from_slice(&[0x36, 0x41, 0x00]); // entry a1,0x20 @ pc 0
        rom[0x1080..0x1083].copy_from_slice(&[0x00, 0x20, 0x00]); // isync (stub handler)
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        // The overflow vector (VECBASE+0x80 = 0x1080) lands in a DIFFERENT
        // 4KB page than pc 0 -- map it too (way 0, ei=1, no collision with
        // page 0's ei=0 slot).
        cpu.mmu.write_tlb(false, 0x1000 | 0x1, 0x1000 | 0);
        cpu.vecbase = 0x1000;
        cpu.regs.set_callinc(2);
        cpu.regs.ps |= PS_WOE;
        cpu.regs.windowbase = 14; // newest frame; +2 wraps onto the live quad 0
        cpu.regs.windowstart = 0x5555; // call8 frames at quads 0,2,..14

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
        assert_eq!(cpu.regs.windowbase, 0, "rotated +n=2 onto the frame to spill (14+2 mod 16)");
        assert_eq!(cpu.regs.owb(), 14, "pre-rotation WINDOWBASE saved in PS.OWB");
        assert_eq!(cpu.regs.windowstart, 0x5555, "WINDOWSTART unchanged by the raise itself");

        // The stub handler dispatches and executes as an ordinary instruction.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 0x1083);
    }

    #[test]
    fn retw_raises_window_underflow_and_vectors() {
        // Force underflow: WOE enabled, a0 encodes call size 2, but the frame
        // being returned to (windowbase - 2 = quad 2) is NOT live in
        // WINDOWSTART (an earlier overflow spilled it) -> WindowUnderflow8 at
        // VECBASE + 0xC0. Per QEMU the returning frame's bit is cleared, then
        // WINDOWBASE rotates back by k=2 to the frame being refilled, with the
        // old base saved in PS.OWB.
        let rom = vec![0x90, 0x00, 0x00]; // retw @ pc 0
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
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
        assert_eq!(cpu.regs.windowbase, 2, "rotated back by k=2 to the frame to refill");
        assert_eq!(cpu.regs.owb(), 4, "pre-rotation WINDOWBASE saved in PS.OWB");
        assert_eq!(cpu.regs.windowstart, 0, "returning frame's live bit cleared");
    }

    #[test]
    fn rfwo_clears_spilled_frame_and_restores_windowbase_from_owb() {
        let rom = vec![0x00, 0x34, 0x00]; // rfwo @ pc 0
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_owb(5); // pre-overflow WINDOWBASE stashed by the raise
        cpu.regs.windowbase = 2; // rotated onto the spilled frame during the handler
        cpu.regs.windowstart = 1 << 2; // that frame is still marked live
        cpu.regs.set_excm();
        cpu.epc1 = 0x1234;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert!(!cpu.regs.excm(), "rfwo leaves exception mode");
        assert_eq!(cpu.regs.windowstart, 0, "spilled frame's WINDOWSTART bit cleared");
        assert_eq!(cpu.regs.windowbase, 5, "WINDOWBASE restored from PS.OWB");
        assert_eq!(cpu.pc, 0x1234, "resumes at EPC1 (the faulting entry)");
    }

    #[test]
    fn rfwu_marks_refilled_frame_and_restores_windowbase_from_owb() {
        let rom = vec![0x00, 0x35, 0x00]; // rfwu @ pc 0
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_owb(6);
        cpu.regs.windowbase = 3; // rotated onto the frame being refilled
        cpu.regs.windowstart = 0; // its bit was clear (spilled) -- rfwu re-marks it
        cpu.regs.set_excm();
        cpu.epc1 = 0x2000;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert!(!cpu.regs.excm());
        assert_eq!(cpu.regs.windowstart, 1 << 3, "refilled frame's WINDOWSTART bit set");
        assert_eq!(cpu.regs.windowbase, 6, "WINDOWBASE restored from PS.OWB");
        assert_eq!(cpu.pc, 0x2000);
    }

    #[test]
    fn s32e_stores_then_l32e_reads_back_windowed() {
        // s32e a3,a5,-16 (`30 c5 49`) then l32e a4,a5,-16 (`40 c5 09`): the
        // spill lands at AR[5]-16 and the fill reads the same word back. Low
        // (local_data) address, but translation is authoritative for D-side
        // accesses too (Task 4), so the low-window page still needs a DTLB entry.
        let mut rom = vec![0x30, 0xc5, 0x49, 0x40, 0xc5, 0x09];
        rom.resize(0x1000, 0);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.mmu.write_tlb(true, 0x0 | 0x3, 0x0 | 0); // low-window DTLB identity, page 0
        cpu.regs.write_ar(5, 0x0000_1000); // base; -16 -> 0xFF0 (low window)
        cpu.regs.write_ar(3, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // s32e
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // l32e
        assert_eq!(cpu.regs.read_ar(4), 0xdead_beef, "l32e reads back what s32e stored");
    }

    #[test]
    fn no_overflow_when_woe_disabled() {
        // The exact WINDOWSTART that overflowed above must NOT raise when WOE
        // is clear: detection is gated on PS.WOE. entry rotates normally.
        let rom = vec![0x36, 0x41, 0x00]; // entry a1,0x20
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.set_callinc(2); // WOE left clear
        cpu.regs.windowbase = 0;
        cpu.regs.windowstart = (1 << 0) | (1 << 2);
        cpu.regs.write_ar(1, 0x0000_1000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, 2, "entry still rotates when WOE off");
    }
}

#[cfg(test)]
mod loop_tests {
    use super::super::{mapped_cpu, Step};
    use crate::firmware::mmio::Bus;

    #[test]
    fn loop_repeats_body_ar_minus_one_times_then_falls_through() {
        // loop a4, 0x7 (`76 84 03`, imm8=3 -> LEND = 0+4+3 = 7) @ pc 0; body
        // = `addi.n a5,a5,1` (`1b 55`) then `addi.n a6,a6,1` (`1b 66`) --
        // TWO DISTINCT registers, so the test can tell "body ran N times"
        // (both a5 and a6 incremented N times) apart from "only the first
        // body op ran" (a bug that would leave a6 at 0 while a5 still
        // advances) -- a plain single-register counter couldn't catch that.
        // Marker `isync` (`00 20 00`) sits exactly at LEND (pc 7) to prove
        // execution proceeds past the loop once it's exhausted. AR[4]=3 ->
        // LCOUNT = AR[4]-1 = 2 loop-BACKS, so the body runs 3 TOTAL times
        // (1 initial pass-through + 2 loop-backs) -- matches the brief's
        // "AR[4]=3 -> body runs exactly 3 times."
        let rom = vec![
            0x76, 0x84, 0x03, // loop a4, 0x7
            0x1b, 0x55, // addi.n a5,a5,1
            0x1b, 0x66, // addi.n a6,a6,1
            0x00, 0x20, 0x00, // isync (marker, past LEND)
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 3);

        // loop itself: sets LBEG/LEND/LCOUNT, falls through into the body.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3, "falls through to LBEG");
        assert_eq!(cpu.regs.lbeg, 3);
        assert_eq!(cpu.regs.lend, 7);
        assert_eq!(cpu.regs.lcount, 2, "LCOUNT = AR[4]-1 = 2");

        // Drive exactly 6 body-instruction steps (3 iterations x 2 ops).
        for _ in 0..6 {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
        }
        assert_eq!(cpu.regs.read_ar(5), 3, "body ran exactly 3 times (a5 leg)");
        assert_eq!(cpu.regs.read_ar(6), 3, "both ops in the body ran every iteration (a6 leg)");
        assert_eq!(cpu.pc, 7, "loop exhausted (LCOUNT==0), fell through to LEND");
        assert_eq!(cpu.regs.lcount, 0);

        // Marker past LEND executes normally -- pc proceeds past the loop,
        // proving the loop-back doesn't re-fire once LCOUNT is exhausted.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 10);
    }

    #[test]
    fn loopnez_skips_body_entirely_when_count_is_zero() {
        // loopnez a3, 0x7 (`76 93 03`, imm8=3 -> LEND=7, same byte layout as
        // the loop test above bar the r/s nibble) @ pc 0. With AR[3]==0, the
        // body must be skipped entirely: pc jumps straight to LEND, and the
        // marker there executes normally.
        let rom = vec![
            0x76, 0x93, 0x03, // loopnez a3, 0x7
            0x1b, 0x55, // addi.n a5,a5,1 (body -- must NOT execute)
            0x1b, 0x66, // addi.n a6,a6,1 (body -- must NOT execute)
            0x00, 0x20, 0x00, // isync (marker, at LEND)
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0);

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 7, "AR[3]==0 -> pc jumps straight to LEND, body skipped");
        assert_eq!(cpu.regs.read_ar(5), 0, "body never executed (a5 leg)");
        assert_eq!(cpu.regs.read_ar(6), 0, "body never executed (a6 leg)");
        assert_eq!(cpu.regs.lbeg, 3);
        assert_eq!(cpu.regs.lend, 7);
        assert_eq!(
            cpu.regs.lcount, 0xFFFF_FFFF,
            "AR[3]-1 wraps -- loop registers are set unconditionally even \
             though the body is skipped (matches QEMU translate_loop: the \
             LCOUNT/LBEG/LEND writes are unconditional, emitted before the \
             AR[s]==0 branch, not gated on the body actually running)"
        );

        // Marker at LEND executes normally.
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 10);
    }

    #[test]
    fn loopnez_falls_through_into_body_when_count_is_nonzero() {
        // Same program as the skip test, but AR[3]=2 (nonzero): loopnez must
        // fall through into the body exactly like plain `loop` would, with
        // LCOUNT = AR[3]-1 = 1 (one loop-back, body runs twice total).
        let rom = vec![
            0x76, 0x93, 0x03, // loopnez a3, 0x7
            0x1b, 0x55, // addi.n a5,a5,1
            0x1b, 0x66, // addi.n a6,a6,1
            0x00, 0x20, 0x00, // isync (marker, at LEND)
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 2);

        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3, "AR[3]!=0 -> falls through into the body");
        assert_eq!(cpu.regs.lcount, 1);

        for _ in 0..4 {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
        }
        assert_eq!(cpu.regs.read_ar(5), 2, "body ran exactly twice (a5 leg)");
        assert_eq!(cpu.regs.read_ar(6), 2, "body ran exactly twice (a6 leg)");
        assert_eq!(cpu.pc, 7);
        assert_eq!(cpu.regs.lcount, 0);
    }
}
