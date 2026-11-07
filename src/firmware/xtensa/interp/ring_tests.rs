//! Access ring and privileged instructions under the MMU Option.
//!
//! The ring of an access is CRING, `PS.EXCM ? 0 : PS.RING` [RM 4.4.1.4,
//! p.88], except for L32E/S32E, which use PS.RING [RM 4.6.2.3, p.148; L32E
//! p.377; S32E p.509]. A matching entry whose ring is below the access ring
//! raises InstFetchPrivilegeCause (18) or LoadStorePrivilegeCause (26)
//! [RM 4.6.2.3, p.148; Table 4-64, p.89-90]. Privileged instructions raise
//! PrivilegedCause (8) at CRING != 0 [RM 4.4.1.4, p.88; Table 4-64, p.89].

use super::super::regfile::{PS_EXCM, PS_UM};
use super::{unmapped_mmu, Cpu, ExceptionVectors, Step};
use crate::firmware::xtensa::flat_bus::FlatBus;
use crate::firmware::xtensa::mmu::{cause, Tlb};

const PS_RING1: u32 = 1 << 6;
const CODE: u32 = 0x0000_0000;
const DATA: u32 = 0x0100_0000;
const VECBASE: u32 = 0x4000_0000;
const PRIVILEGED_CAUSE: u32 = 8;

/// A CPU at `CODE` whose code page (ITLB way 1) and data page (DTLB way 0)
/// are mapped at the given rings, both identity and with full rights.
fn cpu_with_rings(code_ring: u32, data_ring: u32, ps: u32) -> Cpu {
    let mut cpu = Cpu::new(CODE, unmapped_mmu(), ExceptionVectors::default());
    cpu.vecbase = VECBASE;
    cpu.mmu.write_entry(Tlb::Instruction, CODE | 1, CODE | (code_ring << 4) | 0x1);
    cpu.mmu.write_entry(Tlb::Data, DATA, DATA | (data_ring << 4) | 0x3);
    cpu.regs.ps = ps;
    cpu.regs.write_ar(2, DATA);
    cpu.regs.write_ar(9, DATA + 12); // l32e/s32e a4,a9,-12 -> DATA
    cpu
}

fn bus_with(code: &[u8]) -> FlatBus {
    let mut bus = FlatBus::new(code.to_vec());
    bus.write32(DATA, 0x1122_3344);
    bus
}

const L32I_N_A7_A2_0: [u8; 2] = [0x78, 0x02];
const S32I_N_A7_A2_0: [u8; 2] = [0x79, 0x02];
const L32E_A4_A9_M12: [u8; 3] = [0x40, 0xd9, 0x09];
const S32E_A4_A9_M12: [u8; 3] = [0x40, 0xd9, 0x49];
const RSIL_A2_2: [u8; 3] = [0x20, 0x62, 0x00];
const WSR_A2_PS: [u8; 3] = [0x20, 0xe6, 0x13];
const WSR_A2_SAR: [u8; 3] = [0x20, 0x03, 0x13];

fn expect_fault(step: Step, want: u32) {
    match step {
        Step::Exception { cause, .. } => assert_eq!(cause, want, "exception cause"),
        other => panic!("expected exception {want}, got {other:?}"),
    }
}

#[test]
fn cring_is_ps_ring_outside_exception_mode_and_zero_inside() {
    let mut cpu = cpu_with_rings(0, 0, PS_RING1);
    assert_eq!(cpu.regs.ps_ring(), 1);
    assert_eq!(cpu.regs.cring(), 1);
    cpu.regs.ps = PS_RING1 | PS_EXCM;
    assert_eq!(cpu.regs.ps_ring(), 1, "PS.RING is kept while EXCM is set");
    assert_eq!(cpu.regs.cring(), 0, "EXCM forces CRING to 0");
    cpu.regs.ps = 3 << 6;
    assert_eq!(cpu.regs.cring(), 3);
}

#[test]
fn a_load_at_ring1_from_a_ring0_page_is_a_privilege_fault() {
    let mut cpu = cpu_with_rings(1, 0, PS_RING1);
    let mut bus = bus_with(&L32I_N_A7_A2_0);
    expect_fault(cpu.step(&mut bus), cause::LOAD_STORE_PRIVILEGE);
    assert_eq!(cpu.excvaddr, DATA, "EXCVADDR is the load's address");
    assert_eq!(cpu.epc1, CODE, "EPC1 is the faulting load");
    assert_eq!(cpu.regs.read_ar(7), 0, "the load did not complete");
    assert!(cpu.regs.excm());
    assert_eq!(cpu.regs.cring(), 0, "the handler runs at ring 0");
}

#[test]
fn a_store_at_ring1_to_a_ring0_page_is_a_privilege_fault() {
    let mut cpu = cpu_with_rings(1, 0, PS_RING1);
    cpu.regs.write_ar(7, 0xdead_beef);
    let mut bus = bus_with(&S32I_N_A7_A2_0);
    expect_fault(cpu.step(&mut bus), cause::LOAD_STORE_PRIVILEGE);
    assert_eq!(bus.read32(DATA), 0x1122_3344, "the store did not happen");
}

#[test]
fn a_load_at_ring1_from_a_ring1_page_completes() {
    let mut cpu = cpu_with_rings(1, 1, PS_RING1);
    let mut bus = bus_with(&L32I_N_A7_A2_0);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.read_ar(7), 0x1122_3344);
}

#[test]
fn exception_mode_accesses_ring0_pages_whatever_ps_ring_says() {
    let mut cpu = cpu_with_rings(0, 0, PS_RING1 | PS_EXCM);
    let mut bus = bus_with(&L32I_N_A7_A2_0);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.read_ar(7), 0x1122_3344);
}

#[test]
fn a_fetch_at_ring1_from_a_ring0_page_is_a_privilege_fault() {
    let mut cpu = cpu_with_rings(0, 1, PS_RING1);
    let mut bus = bus_with(&L32I_N_A7_A2_0);
    expect_fault(cpu.step(&mut bus), cause::INST_FETCH_PRIVILEGE);
    assert_eq!(cpu.excvaddr, CODE);
    assert_eq!(cpu.epc1, CODE);
}

#[test]
fn a_fault_at_ring1_with_um_set_takes_the_user_vector() {
    // PS.UM selects the vector, not the privilege [RM 4.6.1.2, p.144].
    let mut cpu = cpu_with_rings(1, 0, PS_RING1 | PS_UM);
    let mut bus = bus_with(&L32I_N_A7_A2_0);
    match cpu.step(&mut bus) {
        Step::Exception { cause, pc } => {
            assert_eq!(cause, cause::LOAD_STORE_PRIVILEGE);
            assert_eq!(pc, VECBASE + ExceptionVectors::default().user);
        }
        other => panic!("expected exception, got {other:?}"),
    }
}

#[test]
fn l32e_translates_at_ps_ring_not_cring() {
    // The window handlers run with EXCM set (CRING 0) and reach the
    // interrupted code's stack at PS.RING [RM L32E p.377].
    let mut cpu = cpu_with_rings(0, 0, PS_RING1 | PS_EXCM);
    let mut bus = bus_with(&L32E_A4_A9_M12);
    expect_fault(cpu.step(&mut bus), cause::LOAD_STORE_PRIVILEGE);
    assert_eq!(cpu.excvaddr, DATA);

    let mut cpu = cpu_with_rings(0, 1, PS_RING1 | PS_EXCM);
    let mut bus = bus_with(&L32E_A4_A9_M12);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.read_ar(4), 0x1122_3344);
}

#[test]
fn s32e_translates_at_ps_ring_not_cring() {
    let mut cpu = cpu_with_rings(0, 0, PS_RING1 | PS_EXCM);
    cpu.regs.write_ar(4, 0xdead_beef);
    let mut bus = bus_with(&S32E_A4_A9_M12);
    expect_fault(cpu.step(&mut bus), cause::LOAD_STORE_PRIVILEGE);
    assert_eq!(bus.read32(DATA), 0x1122_3344);

    let mut cpu = cpu_with_rings(0, 1, PS_RING1 | PS_EXCM);
    cpu.regs.write_ar(4, 0xdead_beef);
    let mut bus = bus_with(&S32E_A4_A9_M12);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(bus.read32(DATA), 0xdead_beef);
}

#[test]
fn l32e_at_ring1_is_a_privileged_instruction() {
    // "if CRING != 0 then Exception (PrivilegedInstructionCause)" [RM p.377].
    let mut cpu = cpu_with_rings(1, 1, PS_RING1);
    let mut bus = bus_with(&L32E_A4_A9_M12);
    expect_fault(cpu.step(&mut bus), PRIVILEGED_CAUSE);
    assert_eq!(cpu.regs.read_ar(4), 0);
    assert_eq!(cpu.epc1, CODE);
}

#[test]
fn rsil_at_ring1_is_a_privileged_instruction() {
    let mut cpu = cpu_with_rings(1, 1, PS_RING1);
    let mut bus = bus_with(&RSIL_A2_2);
    expect_fault(cpu.step(&mut bus), PRIVILEGED_CAUSE);
    assert_eq!(cpu.regs.read_ar(2), DATA, "rsil wrote nothing");
    assert_eq!(cpu.regs.intlevel(), 0, "PS.INTLEVEL unchanged");
}

#[test]
fn wsr_to_ps_at_ring1_is_privileged_but_wsr_to_sar_is_not() {
    // WSR with special register number >= 64 is privileged [RM WSR p.561].
    let mut cpu = cpu_with_rings(1, 1, PS_RING1);
    cpu.regs.write_ar(2, 0);
    let mut bus = bus_with(&WSR_A2_PS);
    expect_fault(cpu.step(&mut bus), PRIVILEGED_CAUSE);
    assert_eq!(cpu.regs.ps_ring(), 1, "PS was not written");

    let mut cpu = cpu_with_rings(1, 1, PS_RING1);
    cpu.regs.write_ar(2, 7);
    let mut bus = bus_with(&WSR_A2_SAR);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.sar, 7);
}

#[test]
fn privileged_instructions_run_at_ring0_and_in_exception_mode() {
    let mut cpu = cpu_with_rings(0, 0, 0);
    let mut bus = bus_with(&RSIL_A2_2);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.intlevel(), 2);

    let mut cpu = cpu_with_rings(0, 0, PS_RING1 | PS_EXCM);
    let mut bus = bus_with(&RSIL_A2_2);
    assert_eq!(cpu.step(&mut bus), Step::Ran);
    assert_eq!(cpu.regs.intlevel(), 2);
}

/// Fill loop `s8i a3,a5,0; addi.n a5,a5,1` (bytes as in the fast-path tests)
/// at `CODE`, `n` iterations into `dest`, run under `ps`; returns the CPU
/// after the first step that is not `Ran` or after `max` steps.
fn run_fill(fast: bool, dest: u32, dest_ring: u32, n: u32, max: u32) -> (Cpu, FlatBus, Option<u32>) {
    let mut cpu = Cpu::new(CODE, unmapped_mmu(), ExceptionVectors::default());
    cpu.vecbase = VECBASE;
    cpu.fastpath_enabled = fast;
    cpu.mmu.write_entry(Tlb::Instruction, CODE | 1, CODE | (1 << 4) | 0x1);
    // 1 MB way-4 data page at `dest`.
    cpu.mmu
        .write_entry(Tlb::Data, (dest & 0xfff0_0000) | 4, (dest & 0xfff0_0000) | (dest_ring << 4) | 0x3);
    let mut bus = FlatBus::new(vec![0x32, 0x45, 0x00, 0x1b, 0x55]);
    cpu.regs.ps = PS_RING1;
    cpu.regs.lbeg = CODE;
    cpu.regs.lend = CODE + 5;
    cpu.regs.lcount = n - 1;
    cpu.regs.write_ar(5, dest);
    cpu.regs.write_ar(3, 0xab);
    let mut fault = None;
    for _ in 0..max {
        match cpu.step(&mut bus) {
            Step::Ran => {}
            Step::Exception { cause, .. } => {
                fault = Some(cause);
                break;
            }
            other => panic!("unexpected {other:?}"),
        }
        if cpu.regs.lcount == 0 && cpu.pc == CODE + 5 {
            break;
        }
    }
    (cpu, bus, fault)
}

#[test]
fn the_fill_fast_path_honours_the_access_ring() {
    const DEST: u32 = 0x0210_0000;
    const N: u32 = 0x2000;
    for fast in [true, false] {
        let (cpu, bus, fault) = run_fill(fast, DEST, 0, N, 16);
        assert_eq!(fault, Some(cause::LOAD_STORE_PRIVILEGE), "fast={fast}");
        assert_eq!(cpu.excvaddr, DEST);
        assert_eq!(bus.read8(DEST), 0, "nothing filled into the ring-0 page");
    }
    // A ring-1 destination still collapses in one step at ring 1.
    let (cpu, bus, fault) = run_fill(true, DEST, 1, N, 1);
    assert_eq!(fault, None);
    assert_eq!(cpu.regs.lcount, 0, "collapsed in one step");
    assert_eq!(bus.read8(DEST + N - 1), 0xab);
}
