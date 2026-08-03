use super::*;
use crate::firmware::mmio::{Region, StubAccess};

#[test]
fn m2c_low_instruction_window_is_the_header_stripped_body() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    for vaddr in [0x100, 0xa3c, 0xa78, 0x50e8, 0xc847] {
        for i in 0..4 {
            assert_eq!(
                proc.bus.fetch8(vaddr + i, vaddr + i),
                raw[(vaddr + LOW_VMA_FILE_OFFSET + i) as usize],
                "low instruction byte at {:#x}",
                vaddr + i,
            );
        }
    }

    for i in 0..4 {
        assert_eq!(
            proc.bus.fetch8(0x2000_0340 + i, 0x340 + i),
            raw[(0x340 + PSP_LOAD_OFFSET + i) as usize],
            "high boot-alias byte at {:#x}",
            0x2000_0340 + i,
        );
    }
    assert_eq!(proc.entry, 0x100);
}

#[test]
fn m2c_exception_vectors_match_the_executed_firmware() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let decode_at = |proc: &mut FirmwareProcessor, pc| {
        let bytes: [u8; 3] = std::array::from_fn(|i| proc.bus.fetch8(pc + i as u32, pc + i as u32));
        decode::decode(&bytes, pc).op
    };
    assert!(matches!(decode_at(&mut proc, 0xa98), Op::Rsr { sr: 0xc0, .. }));
    assert!(matches!(decode_at(&mut proc, 0xa9d), Op::Wsr { sr: 0xc0, .. }));
    assert!(matches!(decode_at(&mut proc, 0xaac), Op::Rfde));

    for _ in 0..100_000 {
        match proc.cpu.step(&mut proc.bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(
                    cause,
                    crate::firmware::xtensa::interp::EXCCAUSE_SYSCALL,
                    "the first boot exception must be syscall",
                );
                assert_eq!(pc, 0xa3c, "the first syscall must enter UserExceptionVector");
                return;
            }
            Step::Ran => {}
            step => panic!("boot stopped before its first syscall: {step:?}"),
        }
    }
    panic!("boot did not raise its first syscall within 100k instructions");
}

#[test]
fn m2c_loader_rejects_an_image_without_segment_b() {
    let mut raw = vec![0u8; SEG_B_FILE_START as usize];
    raw[0x10..0x14].copy_from_slice(b"$PS1");
    let declared = raw.len() as u32 - 0x100;
    raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
    let image = FirmwareImage::parse(&raw).expect("container header");
    let err = match FirmwareProcessor::try_load_m2c(image) {
        Ok(_) => panic!("truncated Phoenix load map was accepted"),
        Err(error) => error,
    };
    assert!(matches!(err, FirmwareError::Truncated { .. }), "got {err}");
}

#[test]
fn m2c_loader_rejects_an_image_extending_past_the_rom_aperture() {
    let mut raw = vec![0u8; 0x0400_005d];
    raw[0x10..0x14].copy_from_slice(b"$PS1");
    let declared = raw.len() as u32 - 0x100;
    raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
    let image = FirmwareImage::parse(&raw).expect("container header");

    assert!(
        FirmwareProcessor::try_load_m2c(image).is_err(),
        "a segment-A byte outside the ROM aperture was accepted",
    );
}

#[test]
fn m2c_boot_with_device_routes_firmware_array_writes() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(path).expect("read firmware");
    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let mut processor = FirmwareProcessor::try_load_m2c(image).expect("load Phoenix firmware");
    let mut device = crate::device::DeviceState::new_npu1();
    device.write_tile_register(4, 0, 0x000f_ff20, 1);
    assert!(device.array.clock().is_column_active(4));

    let report = processor.boot_to_idle_with_device(&mut device, 200_000);

    assert!(report.reached_idle, "firmware did not reach idle: {report:?}");
    assert!(!device.array.clock().is_column_active(4));
}

#[test]
fn m2c_boot_publishes_alive_state_through_host_sram() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let report = proc.boot_to_idle(200_000);
    assert!(report.reached_idle, "firmware did not reach idle: {report:?}");

    assert_eq!(proc.bus.host_sram_load32(0x030b_b020), 0x5550_4e5f, "management-channel descriptor magic",);
    assert_eq!(
        proc.bus.host_sram_load32(0x030b_f000),
        0x030b_b000,
        "FW_ALIVE_OFF must publish the descriptor's device address",
    );

    proc.bus.host_sram_store32(0x030b_f000, 0);
    assert_eq!(proc.bus.host_sram_load32(0x030b_f000), 0, "the driver's clear reaches local SRAM");
}

#[test]
fn boots_real_firmware_from_pinned_entry() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");

    let mut proc = FirmwareProcessor::load(img, BOOT_ENTRY);
    let report = proc.boot_to_idle(5_000_000);

    eprintln!("=== M1.7 boot observation ===");
    eprintln!("entry            = {:#x}", proc.entry);
    eprintln!("instrs_executed  = {}", report.instrs_executed);
    eprintln!("last_pc          = {:#x}", report.last_pc);
    eprintln!("reached_idle     = {}", report.reached_idle);
    eprintln!("wait_reason      = {:?}", report.wait_reason);
    eprintln!("unresolved_spin  = {:?}", report.unresolved_spin);
    eprintln!("unknown_op       = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#08x}")));
    eprintln!("window_exceptions= {}", report.window_exceptions);
    eprintln!("funcs_entered    = {:?}", report.funcs_entered);

    // Coherence assertion (the entry-pinning check): from BOOT_ENTRY the
    // interpreter decodes and runs a coherent MMU-setup stream -- it does
    // NOT desync into Unknown within the first handful of instructions.
    // The prologue is 0x320..0x399 (movi.n/wsr/witlb/wdtlb/iitlb/idtlb/or/
    // dsync/isync/l32r) before the `jx` into virtual space at 0x399.
    assert!(
        report.instrs_executed > 20,
        "entry {BOOT_ENTRY:#x} desynced early: only {} instrs, last_pc={:#x}, unknown={:?}",
        report.instrs_executed,
        report.last_pc,
        report.unknown_op,
    );

    // The prologue is straight-line MMU setup: no windowed calls, so H1
    // (window overflow dormant) holds across everything observable this
    // phase -- H2 (overflow fires) cannot be reached before the MMU wall.
    assert_eq!(report.window_exceptions, 0, "no window exception in the boot prologue");
}

#[test]
fn ctxsw_call0_target_uses_matching_plus_100_section() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let caller = 0x2a86;
    let caller_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(caller + i as u32, caller + i as u32));
    let target = match decode::decode(&caller_bytes, caller).op {
        Op::Call0 { target } => target,
        op => panic!("expected context-switch Call0 at {caller:#x}, got {op:?}"),
    };
    assert_eq!(target, 0xdf98);

    let target_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
    assert!(
        matches!(decode::decode(&target_bytes, target).op, Op::Rsr { sr: 72, t: 2 }),
        "the +0x100 context-switch caller must reach its matching window-rotation helper, \
             not the base-framed bytes at {target:#x}: {:02x?}",
        &target_bytes[..3],
    );
}

#[test]
fn ctxsw_callx4_target_uses_matching_plus_100_section() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let load = 0x2abe;
    let load_bytes: [u8; 8] = std::array::from_fn(|i| proc.bus.fetch8(load + i as u32, load + i as u32));
    let literal_vaddr = match decode::decode(&load_bytes, load).op {
        Op::L32r { t: 4, target } => target,
        op => panic!("expected context-restore L32r at {load:#x}, got {op:?}"),
    };
    let target = u32::from_le_bytes(std::array::from_fn(|i| {
        proc.bus.fetch8(literal_vaddr + i as u32, literal_vaddr + i as u32)
    }));
    assert_eq!(target, 0x2568);

    let target_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
    assert!(
        matches!(decode::decode(&target_bytes, target).op, Op::Entry { .. }),
        "the +0x100 context-restore caller must reach its matching callback, \
             not the zero-filled base view at {target:#x}: {:02x?}",
        &target_bytes[..3],
    );

    for literal in (0x2510u32..=0x2544).step_by(4) {
        let file = (literal + LOW_VMA_FILE_OFFSET) as usize;
        let expected = u32::from_le_bytes(raw[file..file + 4].try_into().unwrap());
        assert_eq!(
            proc.bus.inst_load32_overlay(literal, literal),
            expected,
            "context-restore L32R at {literal:#x} must use the callback's matching literal pool",
        );
    }
}

#[test]
fn m2c_event_service_calls_use_matching_plus_100_sections() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    for (caller, expected_target) in [
        (0x598c, 0x94b8),
        (0x5995, 0x956c),
        (0x599d, 0x958c),
        (0x59ad, 0x94b8),
        (0x59b5, 0x94f8),
        (0x59bc, 0x94d8),
        (0x59c5, 0x954c),
        (0x59d6, 0x952c),
        (0x59de, 0x95b4),
    ] {
        let caller_bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(caller + i as u32, caller + i as u32));
        let target = match decode::decode(&caller_bytes, caller).op {
            Op::Call8 { target } => target,
            op => panic!("expected event-service Call8 at {caller:#x}, got {op:?}"),
        };
        assert_eq!(target, expected_target, "event-service call at {caller:#x}");

        let target_bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
        assert!(
            matches!(decode::decode(&target_bytes, target).op, Op::Entry { .. }),
            "the +0x100 event-service caller at {caller:#x} must reach its matching \
             callee, not the base-framed function tail at {target:#x}: {:02x?}",
            &target_bytes[..3],
        );
    }

    let literal_vaddr = 0x32f0;
    let literal = u32::from_le_bytes(std::array::from_fn(|i| {
        proc.bus.fetch8(literal_vaddr + i as u32, literal_vaddr + i as u32)
    }));
    assert_eq!(
        literal, 0xf2e0,
        "the +0x100 event-service caller must read its matching metadata-table literal",
    );
}

#[test]
fn exception_report_call_uses_matching_plus_100_section() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let caller = 0xc4ca;
    let caller_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(caller + i as u32, caller + i as u32));
    let target = match decode::decode(&caller_bytes, caller).op {
        Op::Call8 { target } => target,
        op => panic!("expected exception-report Call8 at {caller:#x}, got {op:?}"),
    };
    assert_eq!(target, 0x7f20);

    let target_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
    assert!(
        matches!(decode::decode(&target_bytes, target).op, Op::Entry { .. }),
        "the +0x100 exception-report caller must reach its matching callee, \
             not the base-framed store tail at {target:#x}: {:02x?}",
        &target_bytes[..3],
    );
}

#[test]
fn scheduler_mmio_lookup_call_uses_matching_plus_100_section() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let caller = 0x7d78;
    let caller_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(caller + i as u32, caller + i as u32));
    let target = match decode::decode(&caller_bytes, caller).op {
        Op::Call8 { target } => target,
        op => panic!("expected scheduler MMIO-lookup Call8 at {caller:#x}, got {op:?}"),
    };
    assert_eq!(target, 0x8934);

    let target_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
    assert!(
        matches!(decode::decode(&target_bytes, target).op, Op::Entry { .. }),
        "the scheduler's +0x100 call must reach the matching MMIO lookup, \
             not the base-framed function interior at {target:#x}: {:02x?}",
        &target_bytes[..3],
    );

    let literal_insn = 0x8937;
    let literal_bytes: [u8; 8] =
        std::array::from_fn(|i| proc.bus.fetch8(literal_insn + i as u32, literal_insn + i as u32));
    let literal_vaddr = match decode::decode(&literal_bytes, literal_insn).op {
        Op::L32r { t: 5, target } => target,
        op => panic!("expected MMIO-table L32r at {literal_insn:#x}, got {op:?}"),
    };
    assert_eq!(literal_vaddr, 0x349c);
    assert_eq!(
        proc.bus.inst_load32_overlay(literal_vaddr, literal_vaddr),
        0x2722_0000,
        "the matching MMIO lookup must use its device-register table base",
    );
}

/// Characterization lock (MMU data-path design, 2026-07-06): the
/// translation-authoritative data path depends on the low DRAM window being
/// TLB-covered from reset through steady state. Assert the STRUCTURAL facts,
/// not just a point sample: way-6 ei0 is the reset identity region, the
/// prologue clears entry 1 (code region) not entry 0, asid resolves ring 0,
/// and every low-window data probe still translates identity at steady state.
#[test]
fn low_window_dram_is_translation_covered_from_reset() {
    use crate::firmware::xtensa::mmu::Mmu;
    // (a) Reset populates way-6 ei0 = VPN 0 -> paddr 0, asid 1, attr 3, variable.
    let fresh = Mmu::new_with_varway56(true);
    let e0 = fresh.dtlb[6][0];
    assert_eq!(e0.vaddr, 0);
    assert_eq!(e0.paddr, 0);
    assert_eq!(e0.asid, 1);
    assert_eq!(e0.attr, 3, "RWX -- grants low-window read AND write");
    assert!(e0.variable);
    // (c) asid 1 always resolves to ring 0 (write_rasid forces the ring-0 byte).
    let mut r = Mmu::new_with_varway56(true);
    r.write_rasid(0x08070605);
    assert_eq!(r.lookup(0x0000_f9e0, true).expect("low window resolves").ring, 0);

    // (b) + (d) need the real boot; skip cleanly without the binary.
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary absent -- structural (a)/(c) still checked");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    // Sample at END OF STEADY STATE: run until the boot reaches the syscall
    // context-switch entry (CTXSW_CALLEE_LO=0x2630) or walls. The invariant is
    // "low-window identity from reset THROUGH STEADY STATE" -- the context
    // switch is precisely where steady state ends: it reprograms processor/MMU
    // state (PS/EPC restore + address-space swap), so past it the low-window
    // identity mapping is legitimately no longer guaranteed. Before iter20's
    // +0x100 overlays, 0x2630 ran a mis-fetched `call0` and walled one instr
    // later at 0x44a34, so this loop happened to stop here anyway; now 0x2630
    // runs for real, so we stop explicitly at its entry (frontier-independent).
    for _ in 0..300_000 {
        if proc.cpu.pc == CTXSW_CALLEE_LO {
            break;
        }
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
    }
    // (b) prologue invalidated way-6 entry 1 (code region), left entry 0 (low window).
    assert_eq!(proc.cpu.mmu.dtlb[6][0].asid, 1, "low-window entry 0 still live");
    assert_eq!(proc.cpu.mmu.dtlb[6][1].asid, 0, "code-region entry 1 invalidated");
    // (d) every low-window data address the firmware touches translates identity.
    for a in [0x0000_f9e0u32, 0x0000_9070, 0x0000_2278, 0x0000_2250, 0x0000_22bc] {
        let t = proc.cpu.mmu.translate(&mut proc.bus, a, 0 /*load*/, 0).expect("resolves");
        assert_eq!(t.paddr, a, "low-window data translates identity");
    }
}

/// M2b Task 10 (#140): an OBSERVATION run, not a pass/fail correctness
/// test. Boots the real firmware with the now-live MMU (M2b Tasks 1-9)
/// and records what the autorefill mechanism actually computes at the
/// `jx` target, plus the operands of every `witlb`/`wdtlb`/`iitlb`/
/// `idtlb` the boot prologue issues -- the empirical starting point for
/// M2c's page-table-data reconstruction. See
/// `docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md`
/// for the write-up this test's output feeds.
#[test]
fn characterize_real_firmware_autorefill() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load(img, BOOT_ENTRY);

    // One recorded `witlb`/`wdtlb`/`iitlb`/`idtlb`: the AS operand (way
    // index + VPN) and, for the two install ops, the AT operand
    // (paddr|attr) -- the firmware's own INTENDED region map. This is
    // the concrete artifact M2c needs (Task 8 already found these all
    // target fixed ways 5/6, so they're currently no-ops against this
    // MMU model -- the central M2c "varway56" question).
    struct TlbOp {
        pc: u32,
        mnemonic: &'static str,
        way: u32,
        as_: u32,
        at: Option<u32>,
    }
    let mut tlb_ops: Vec<TlbOp> = Vec::new();

    const MAX_STEPS: u32 = 200;
    let mut n = 0u32;
    let stop_reason = loop {
        if n >= MAX_STEPS {
            break format!("step cap ({MAX_STEPS}) reached, stuck at pc={:#x}", proc.cpu.pc);
        }

        let pc = proc.cpu.pc;
        // Peek (no side effects), same pattern as
        // `coverage_scan::zero_unknown_in_boot_prologue`'s `is_jx` check
        // -- witlb/wdtlb/iitlb/idtlb only READ their AR operands, so
        // recording them before the step executes is equivalent to
        // after, but keeps the established "peek, then step" shape.
        let peek =
            [proc.bus.peek8(pc), proc.bus.peek8(pc.wrapping_add(1)), proc.bus.peek8(pc.wrapping_add(2))];
        match decode::decode(&peek, pc).op {
            Op::Witlb { t, s } => tlb_ops.push(TlbOp {
                pc,
                mnemonic: "witlb",
                way: proc.cpu.regs.read_ar(s) & 0x7,
                as_: proc.cpu.regs.read_ar(s),
                at: Some(proc.cpu.regs.read_ar(t)),
            }),
            Op::Wdtlb { t, s } => tlb_ops.push(TlbOp {
                pc,
                mnemonic: "wdtlb",
                way: proc.cpu.regs.read_ar(s) & 0xf,
                as_: proc.cpu.regs.read_ar(s),
                at: Some(proc.cpu.regs.read_ar(t)),
            }),
            Op::Iitlb { s } => tlb_ops.push(TlbOp {
                pc,
                mnemonic: "iitlb",
                way: proc.cpu.regs.read_ar(s) & 0x7,
                as_: proc.cpu.regs.read_ar(s),
                at: None,
            }),
            Op::Idtlb { s } => tlb_ops.push(TlbOp {
                pc,
                mnemonic: "idtlb",
                way: proc.cpu.regs.read_ar(s) & 0xf,
                as_: proc.cpu.regs.read_ar(s),
                at: None,
            }),
            _ => {}
        }

        let step = proc.cpu.step(&mut proc.bus);

        // Same counting convention as `FirmwareProcessor::boot_to_idle`:
        // an executed instruction (including one that raises a fault)
        // counts; `Step::Unknown` did not execute (pc unchanged), so it's
        // a stop reason, not an executed instruction.
        match step {
            Step::Ran => {
                n += 1;
            }
            Step::Exception { cause, pc: vector_pc } => {
                n += 1;
                break format!("Exception cause={cause} vector_pc={vector_pc:#x}");
            }
            Step::Wait(reason) => {
                n += 1;
                break format!("Wait({reason:?})");
            }
            Step::Unknown { pc, word } => break format!("Unknown pc={pc:#x} word={word:#010x}"),
        }
    };

    // The boot must reach the wall via the live MMU: an ITLB-miss
    // Exception (cause 16, INST_TLB_MISS) raised by the `jx` target's
    // fetch fault. Checked BEFORE reading `excvaddr` below, since that
    // field is only meaningful as "the jx target" once this specific
    // fault path is confirmed to be what actually happened -- a
    // step-cap timeout or a Wait would leave `excvaddr` holding
    // something else (or its zeroed reset value).
    assert!(
        stop_reason.starts_with("Exception cause=16"),
        "expected the boot to stop at the jx target's ITLB miss (cause 16, INST_TLB_MISS) -- a \
             different outcome means cpu.excvaddr below would not be the jx-target vaddr this \
             characterization assumes: {stop_reason}",
    );

    // The autorefill anchor numbers (`get_pte`, `mmu.rs`). PTEVADDR and
    // the faulting vaddr are both LIVE-READ off the CPU -- `mmu.ptevaddr`
    // (programmed by the prologue's own `wsr.ptevaddr`) and
    // `cpu.excvaddr` (set by `Cpu::translate`'s fault path, interp/mod.rs,
    // to whatever vaddr actually faulted -- NOT assumed from static
    // analysis of the prologue's `jx` operand). `pt_vaddr` is then
    // computed from those two live values by the same formula production
    // code uses (`get_pte`). `pt_lookup` is a read-only probe of whether
    // anything in the DTLB actually covers the computed address (it
    // doesn't -- the firmware's own high-region witlb targeted fixed
    // ways 5/6, which never took per the loop above).
    let ptevaddr = proc.cpu.mmu.ptevaddr;
    let jx_target = proc.cpu.excvaddr;
    let pt_vaddr = (ptevaddr | (jx_target >> 10)) & !3;
    let pt_lookup = proc.cpu.mmu.lookup(pt_vaddr, true);

    eprintln!("=== M2b Task 10: real-firmware autorefill characterization ===");
    eprintln!("instructions executed = {n}");
    eprintln!("stop reason           = {stop_reason}");
    eprintln!("last_pc               = {:#x}", proc.cpu.pc);
    eprintln!("PTEVADDR              = {ptevaddr:#x}");
    eprintln!("jx target (excvaddr)  = {jx_target:#x}");
    eprintln!("computed pt_vaddr     = {pt_vaddr:#x}");
    eprintln!("pt_vaddr DTLB lookup  = {pt_lookup:?}");
    eprintln!("firmware TLB-setup operands during the prologue:");
    for op in &tlb_ops {
        eprintln!("  {:#x}: {} way={} AS={:#x} AT={:?}", op.pc, op.mnemonic, op.way, op.as_, op.at);
    }

    // Sanity checks only -- NOT a boot-success assertion. M2b is not
    // expected to get past the wall (M2c supplies the missing
    // page-table data); these confirm the *mechanism* ran faithfully.
    assert_eq!(
        ptevaddr, 0x3c00_0000,
        "boot prologue should program PTEVADDR via wsr.ptevaddr to 0x3c000000 -- if this drifts, the \
             M2c pt_vaddr derivation (docs/superpowers/specs/2026-07-04-m2b-mmu-mechanism-design.md) needs \
             re-deriving",
    );
    assert_eq!(
        jx_target, 0x2000_0340,
        "the jx target (live-read from cpu.excvaddr, not a literal) drifted from the known boot-\
             prologue jx destination -- re-derive the M2c pt_vaddr math (docs/superpowers/specs/2026-07-04-\
             m2b-mmu-mechanism-design.md) if this genuinely changed",
    );
    assert!(
        !tlb_ops.is_empty(),
        "boot prologue issued no witlb/wdtlb/iitlb/idtlb -- expected several (already observed during \
             M2a/M2b); an empty list means the boot desynced before reaching them",
    );
    assert!(
        tlb_ops.iter().all(|op| op.way == 5 || op.way == 6),
        "expected every boot TLB-setup op to target fixed ways 5/6 (Task 8 finding); a different way \
             changes the M2c varway56 framing entirely: {:#?}",
        tlb_ops.iter().map(|op| (op.pc, op.mnemonic, op.way)).collect::<Vec<_>>(),
    );
    assert!(
        pt_lookup.is_err(),
        "expected the PTE address {pt_vaddr:#x} to be unmapped (the firmware's own high-region map \
             never took) -- a hit here would mean something now covers it and the wall should have moved",
    );
}

/// Boot the pinned firmware through go-alive to its natural scheduler wait.
#[test]
fn m2c_boot_reaches_natural_scheduler_wait() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let report = proc.boot_to_idle(200_000);
    assert!(report.reached_idle, "firmware did not reach its natural scheduler wait: {report:?}");
    assert_eq!(report.wait_reason, Some(WaitReason::Waiti));
    assert_eq!(report.last_pc & 0x00ff_ffff, 0x0000_c84a);
    assert_eq!(report.window_exceptions, 0);
    assert_eq!(report.unresolved_spin, None);
    assert_eq!(report.unknown_op, None);
}

#[test]
fn m2c_source_46_returns_to_idle() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let boot = proc.boot_to_idle(200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    assert_ne!(proc.bus.data_load32(0x2720_0304) & (1 << 14), 0, "source 46 is not enabled");
    assert_ne!(proc.cpu.intenable & 1, 0, "Xtensa level-1 input is not enabled");

    assert!(proc.bus.assert_management_source(46));
    assert_eq!(proc.bus.data_load32(0x2720_03b4), 1 << 14);
    assert_eq!(proc.bus.data_load32(0x2720_03c4), 46);

    let handled = proc.boot_to_idle(200_000);
    assert!(handled.reached_idle, "source-46 handler did not return to idle: {handled:?}");
    assert_eq!(handled.unknown_op, None);
    assert_eq!(handled.unresolved_spin, None);
    assert_eq!(proc.bus.data_load32(0x2720_03b4), 0);
    assert_eq!(proc.bus.data_load32(0x2720_03c4), 0);
    assert_eq!(proc.cpu.interrupt & 1, 0);
}

struct PinnedMgmtChannel {
    x2i_tail: u32,
    i2x_head: u32,
    next_id: u32,
    async_registrations: Vec<(u32, u64)>,
}

impl PinnedMgmtChannel {
    fn new() -> Self {
        Self { x2i_tail: 0, i2x_head: 0, next_id: 0x1d00_0000, async_registrations: Vec::new() }
    }

    fn publish(&mut self, proc: &mut FirmwareProcessor, opcode: u32, body: &[u32]) -> (u32, u32) {
        let body_bytes = body.len() as u32 * 4;
        let packet_bytes = 16 + body_bytes;
        assert!(self.x2i_tail + packet_bytes <= 1024, "test sequence wrapped the X2I ring");

        let id = self.next_id;
        let header = [body_bytes, 0x0001_0000 | body_bytes, id, opcode];
        for (index, word) in header.iter().chain(body).enumerate() {
            proc.bus.host_store32(0x030b_c000 + self.x2i_tail + index as u32 * 4, *word);
        }
        self.x2i_tail += packet_bytes;
        let old_i2x_tail = proc.bus.host_load32(0x030e_d000);
        proc.bus.host_store32(0x030e_c000, self.x2i_tail);
        self.next_id = self.next_id.wrapping_add(1);
        (id, old_i2x_tail)
    }

    fn deliver(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) -> (u32, u32) {
        let published = self.publish(proc, opcode, body);

        let report = proc.boot_to_idle_with_device(device, 250_000);
        assert!(report.reached_idle, "opcode {opcode:#x} did not return to idle: {report:?}");
        assert_eq!(report.unknown_op, None, "opcode {opcode:#x}");
        assert_eq!(report.unresolved_spin, None, "opcode {opcode:#x}");
        assert_eq!(
            proc.bus.host_load32(0x030e_c004),
            self.x2i_tail,
            "firmware did not consume opcode {opcode:#x}",
        );

        published
    }

    fn finish_transact(
        &mut self,
        proc: &mut FirmwareProcessor,
        opcode: u32,
        id: u32,
        old_i2x_tail: u32,
    ) -> Vec<u32> {
        assert_eq!(old_i2x_tail, self.i2x_head, "unconsumed I2X data before opcode {opcode:#x}");

        let body_bytes = proc.bus.host_load32(0x030b_d000 + self.i2x_head);
        assert_ne!(body_bytes, 0, "opcode {opcode:#x} produced no response");
        assert_eq!(body_bytes & 3, 0, "opcode {opcode:#x} response is not word-aligned");
        assert_eq!(
            proc.bus.host_load32(0x030b_d004 + self.i2x_head),
            0x0001_0000 | body_bytes,
            "opcode {opcode:#x} response protocol",
        );
        assert_eq!(proc.bus.host_load32(0x030b_d008 + self.i2x_head), id, "opcode {opcode:#x} response ID");
        assert_eq!(
            proc.bus.host_load32(0x030b_d00c + self.i2x_head),
            opcode,
            "opcode {opcode:#x} response opcode",
        );

        let body = (0..body_bytes / 4)
            .map(|word| proc.bus.host_load32(0x030b_d010 + self.i2x_head + word * 4))
            .collect::<Vec<_>>();
        self.i2x_head += 16 + body_bytes;
        assert_eq!(
            proc.bus.host_load32(0x030e_d000),
            self.i2x_head,
            "opcode {opcode:#x} published extra I2X data",
        );
        assert_eq!(
            proc.bus.take_pending_msix_mask(),
            1 << 14,
            "opcode {opcode:#x} did not publish exactly one management MSI-X edge",
        );
        proc.bus.host_store32(0x030e_d004, self.i2x_head);
        proc.bus.host_store32(0x030e_d008, 0);
        body
    }

    fn transact(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) -> Vec<u32> {
        let (id, old_i2x_tail) = self.deliver(proc, device, opcode, body);
        self.finish_transact(proc, opcode, id, old_i2x_tail)
    }

    fn post(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) -> u32 {
        let (id, old_i2x_tail) = self.deliver(proc, device, opcode, body);
        assert_eq!(
            proc.bus.host_load32(0x030e_d000),
            old_i2x_tail,
            "posted opcode {opcode:#x} unexpectedly responded synchronously",
        );
        id
    }

    fn initialize(&mut self, proc: &mut FirmwareProcessor, device: &mut crate::device::DeviceState) {
        assert_eq!(self.transact(proc, device, 0x10a, &[2, 1, 0]), [0]);
        assert_eq!(self.transact(proc, device, 0x10a, &[4, 1, 0]), [0]);
        assert_eq!(self.transact(proc, device, 0x103, &[0]), [0]);
        assert_eq!(self.transact(proc, device, 0x101, &[0]), [0]);
        assert_eq!(self.transact(proc, device, 0x102, &[0]), [0]);
        assert_eq!(self.transact(proc, device, 0x10a, &[1, 1, 0]), [0]);
        assert_eq!(
            self.transact(proc, device, 0x108, &[0]),
            [0, 1, 5, 5, 391],
            "pinned 1502_00 firmware version",
        );

        let aie_version = self.transact(proc, device, 0x0f, &[0]);
        assert_eq!(aie_version.len(), 2);
        assert_eq!(aie_version[0], 0);
        assert_ne!(aie_version[1], 0);

        let tile_info = self.transact(proc, device, 0x0e, &[0]);
        assert_eq!(tile_info.len(), 12);
        assert_eq!(tile_info[0], 0);
        let reported_cols = (tile_info[3] & 0xffff) as usize;
        assert!((1..=device.cols()).contains(&reported_cols));

        for col in 0..reported_cols {
            let address = 0x1000_0000 + col as u32 * 0x2000;
            let id = self.post(proc, device, 0x10c, &[address, 0, 0x2000]);
            self.async_registrations.push((id, address as u64));
        }
    }

    fn create_context(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        requested_col: u8,
        requested_cols: u8,
    ) -> PinnedContextChannel {
        let response = self.transact(
            proc,
            device,
            0x02,
            &[
                1, // AIE2
                u32::from_le_bytes([requested_col, requested_cols, 0, 0]),
                1, // one CQ pair, PASID 0
                0,
                0,
                0,
                2, // PRIORITY_HIGH
            ],
        );
        PinnedContextChannel::from_create_response(&response)
    }
}

#[test]
fn m2c_core_error_reaches_registered_async_buffer_through_signed_firmware() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let Some(mlir_aie) = std::env::var_os("MLIR_AIE_PATH") else {
        eprintln!("skip: MLIR_AIE_PATH is not set");
        return;
    };
    let Some(error_pdi) = std::env::var_os("XDNA_ERROR_PDI") else {
        eprintln!("skip: error-enabled PDI not present (set XDNA_ERROR_PDI)");
        return;
    };
    let (_, mut insts, functional) = load_frozen_chess_context_fixture(
        std::path::Path::new(&mlir_aie),
        "add_one_using_dma",
        "aie.xclbin",
        9671,
        300,
        1,
        &[1, 2, 3, 4],
    );
    let terminal_tct = insts.len() - 16;
    assert_eq!(u32::from_le_bytes(insts[terminal_tct..terminal_tct + 4].try_into().unwrap()), 0x80);
    assert_eq!(u32::from_le_bytes(insts[terminal_tct + 4..terminal_tct + 8].try_into().unwrap()), 16);
    let event_address = crate::device::registers::TileAddress::encode(
        0,
        2,
        crate::device::regdb::device_reg_layout().core_events.event_generate,
    );
    insts.extend_from_slice(&0u64.to_le_bytes());
    insts.extend_from_slice(&u64::from(event_address).to_le_bytes());
    insts.extend_from_slice(
        &(xdna_archspec::aie2::trace_events::core_events::DECOMPRESSION_UNDERFLOW as u32).to_le_bytes(),
    );
    insts.extend_from_slice(&24u32.to_le_bytes());
    let num_ops = u32::from_le_bytes(insts[8..12].try_into().unwrap()) + 1;
    let total_size = insts.len() as u32;
    insts[8..12].copy_from_slice(&num_ops.to_le_bytes());
    insts[12..16].copy_from_slice(&total_size.to_le_bytes());
    let pdi = std::fs::read(error_pdi).expect("read error-enabled PDI");
    let raw = std::fs::read(path).expect("read firmware");
    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let mut proc = FirmwareProcessor::load_m2c(image);
    let mut engine = crate::interpreter::engine::InterpreterEngine::new_npu1();

    let boot = proc.boot_to_idle_with_device(engine.device_mut(), 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    const ASYNC_BASE: u64 = 0x1000_0000;
    const ASYNC_BUFFER_SIZE: usize = 0x2000;
    const HOST_HEAP_BASE: u64 = 0x6000_0000;
    const HEAP_SIZE: usize = 0x0400_0000;
    const DEVICE_HEAP_BASE: u64 = 0x0400_0000;
    const PDI_DEVICE_ADDR: u64 = 0x0402_0000;
    const PDI_HOST_ADDR: u64 = 0x6002_0000;
    const INST_DEVICE_ADDR: u64 = 0x0402_8000;
    const INST_HOST_ADDR: u64 = 0x6002_8000;
    const INPUT_A_ADDR: u64 = 0x6400_0000;
    const INPUT_B_ADDR: u64 = 0x6400_1000;
    const OUTPUT_ADDR: u64 = 0x6400_2000;
    let async_bytes = engine.device().cols() * ASYNC_BUFFER_SIZE;
    engine
        .host_memory_mut()
        .allocate_region("pinned Phoenix async-event buffers", ASYNC_BASE, async_bytes)
        .expect("allocate async-event buffers");
    engine
        .host_memory_mut()
        .allocate_region("pinned Phoenix context heap", HOST_HEAP_BASE, HEAP_SIZE)
        .expect("allocate context heap");
    engine
        .host_memory_mut()
        .allocate_region("pinned Phoenix data BOs", INPUT_A_ADDR, 0x3000)
        .expect("allocate data BOs");

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, engine.device_mut());
    let mut context = management.create_context(&mut proc, engine.device_mut(), 1, 1);
    assert_eq!(
        management.transact(
            &mut proc,
            engine.device_mut(),
            0x106,
            &[context.context_id, HOST_HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER",
    );
    engine.host_memory_mut().write_bytes(PDI_HOST_ADDR, &pdi);
    let (config_id, config_x2i_tail, _, config_report, _) = pump_pinned_context_command(
        &mut proc,
        &mut engine,
        &mut context,
        0x11,
        &pinned_config_cu_body(PDI_DEVICE_ADDR, functional),
        4,
    );
    assert_eq!(config_report.stop, RuntimePumpStop::ResponseCompleted, "{config_report:?}");
    consume_pinned_context_response(
        &mut proc,
        &mut context,
        config_id,
        config_x2i_tail,
        0x11,
        &[0],
        &config_report,
        "CONFIG_CU",
    );

    let input = (1u32..=64).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    engine.host_memory_mut().write_bytes(INPUT_A_ADDR, &input);
    let exec_body = pinned_chained_exec_body(
        engine.host_memory_mut(),
        DEVICE_HEAP_BASE,
        HOST_HEAP_BASE,
        INST_DEVICE_ADDR,
        INST_HOST_ADDR,
        &insts,
        INPUT_A_ADDR,
        INPUT_B_ADDR,
        OUTPUT_ADDR,
    );
    let old_i2x_tail = proc.bus.host_load32(0x030e_d000);
    let (exec_id, exec_x2i_tail, _, exec_report, _) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context, 0x18, &exec_body, 100_000);
    assert_eq!(exec_report.stop, RuntimePumpStop::ResponseCompleted, "{exec_report:?}");
    consume_pinned_context_response(
        &mut proc,
        &mut context,
        exec_id,
        exec_x2i_tail,
        0x18,
        &[0, 0, 0],
        &exec_report,
        "CHAIN_EXEC_NPU",
    );
    assert_eq!(
        (0..64)
            .map(|index| engine.host_memory().read_u32(OUTPUT_ADDR + index * 4))
            .collect::<Vec<_>>(),
        (2..=65).collect::<Vec<_>>(),
        "frozen kernel output",
    );

    let report = pump_runtime(&mut proc, &mut engine, 8, 200_000, |firmware, _| {
        firmware.bus.host_load32(0x030e_d000) != old_i2x_tail
    });
    assert_eq!(
        report.stop,
        RuntimePumpStop::ResponseCompleted,
        "async error produced no firmware response: {report:?}"
    );

    let response_id = proc.bus.host_load32(0x030b_d008 + management.i2x_head);
    let &(_, buffer_address) = management
        .async_registrations
        .iter()
        .find(|(id, _)| *id == response_id)
        .expect("firmware response did not consume a registered async request");
    assert_eq!(
        management.finish_transact(&mut proc, 0x10c, response_id, old_i2x_tail),
        [0, 0],
        "REGISTER_ASYNC_EVENT response",
    );

    let words = (0..6)
        .map(|word| engine.host_memory().read_u32(buffer_address + word * 4))
        .collect::<Vec<_>>();
    assert_eq!(&words[..2], &[1, 0], "aie_err_info count and return code");
    // The driver names word 2 `rsvd` and never reads it; signed firmware uses
    // it as a private payload cursor, so it is not part of the driver contract.
    assert_eq!(&words[3..], &[0x0000_0102, 1, 70], "one core-event record");

    let reregister_id = management.post(
        &mut proc,
        engine.device_mut(),
        0x10c,
        &[buffer_address as u32, (buffer_address >> 32) as u32, ASYNC_BUFFER_SIZE as u32],
    );
    management.async_registrations.push((reregister_id, buffer_address));
    let old_i2x_tail = proc.bus.host_load32(0x030e_d000);
    engine.device_mut().write_tile_register(
        1,
        3,
        crate::device::regdb::device_reg_layout().core_events.event_generate,
        xdna_archspec::aie2::trace_events::core_events::DECOMPRESSION_UNDERFLOW as u32,
    );
    let report = pump_runtime(&mut proc, &mut engine, 8, 200_000, |firmware, _| {
        firmware.bus.host_load32(0x030e_d000) != old_i2x_tail
    });
    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "second async response: {report:?}");
    assert_eq!(
        management.finish_transact(&mut proc, 0x10c, reregister_id, old_i2x_tail),
        [0, 0],
        "second REGISTER_ASYNC_EVENT response",
    );
    let words = (0..6)
        .map(|word| engine.host_memory().read_u32(buffer_address + word * 4))
        .collect::<Vec<_>>();
    assert_eq!(&words[..2], &[1, 0], "second aie_err_info count and return code");
    assert_eq!(&words[3..], &[0x0000_0103, 1, 70], "second core-event record");
}

#[test]
fn m2c_runtime_pump_delivers_each_pinned_driver_initialization_response() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut engine = crate::interpreter::engine::InterpreterEngine::new_npu1();

    let boot = proc.boot_to_idle_with_device(engine.device_mut(), 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut channel = PinnedMgmtChannel::new();
    for (opcode, body) in
        [(0x10a, &[2, 1, 0][..]), (0x10a, &[4, 1, 0][..]), (0x103, &[0][..]), (0x101, &[0][..])]
    {
        let (id, old_i2x_tail) = channel.publish(&mut proc, opcode, body);
        let report = pump_runtime(&mut proc, &mut engine, 1, 200_000, |_, _| false);
        assert_eq!(report.stop, RuntimePumpStop::ArrayIdleFirmwareWaiting, "opcode {opcode:#x}: {report:?}");
        assert_eq!(
            proc.bus.host_load32(0x030e_c004),
            channel.x2i_tail,
            "firmware did not consume opcode {opcode:#x}",
        );
        assert_eq!(channel.finish_transact(&mut proc, opcode, id, old_i2x_tail), [0]);
        let idle = pump_runtime(&mut proc, &mut engine, 1, 200_000, |_, _| false);
        assert_eq!(
            idle.stop,
            RuntimePumpStop::ArrayIdleFirmwareWaiting,
            "post-ack service after opcode {opcode:#x}: {idle:?}",
        );
    }
}

struct PinnedCq {
    head_addr: u32,
    tail_addr: u32,
    buf_addr: u32,
    buf_size: u32,
}

impl PinnedCq {
    fn from_words(words: &[u32]) -> Self {
        assert_eq!(words.len(), 4);
        assert!(words[..3].iter().all(|value| value & 3 == 0), "unaligned CQ descriptor: {words:x?}");
        assert_ne!(words[3], 0, "zero-sized CQ descriptor");
        Self { head_addr: words[0], tail_addr: words[1], buf_addr: words[2], buf_size: words[3] }
    }
}

struct PinnedContextChannel {
    context_id: u32,
    x2i: PinnedCq,
    i2x: PinnedCq,
    x2i_tail: u32,
    i2x_head: u32,
    next_id: u32,
}

impl PinnedContextChannel {
    fn from_create_response(response: &[u32]) -> Self {
        assert_eq!(response.len(), 19);
        assert_eq!(response[0], 0, "CREATE_CONTEXT status");
        assert_ne!(response[1], u32::MAX, "invalid firmware context ID");
        assert_eq!((response[2] >> 16) & 0xff, 1, "allocated CQ pairs");
        Self {
            context_id: response[1],
            x2i: PinnedCq::from_words(&response[3..7]),
            i2x: PinnedCq::from_words(&response[7..11]),
            x2i_tail: 0,
            i2x_head: 0,
            next_id: 0x1d00_0000,
        }
    }

    fn post(&mut self, bus: &mut Bus, opcode: u32, body: &[u32]) -> (u32, u32, u32) {
        let body_bytes = body.len() as u32 * 4;
        let packet_bytes = 16 + body_bytes;
        assert!(
            self.x2i_tail + packet_bytes <= self.x2i.buf_size,
            "test sequence wrapped the context X2I ring"
        );
        assert_eq!(
            bus.host_load32(self.x2i.head_addr),
            self.x2i_tail,
            "unconsumed context X2I data before opcode {opcode:#x}",
        );

        let id = self.next_id;
        let header = [body_bytes, 0x0001_0000 | body_bytes, id, opcode];
        for (index, word) in header.iter().chain(body).enumerate() {
            bus.host_store32(self.x2i.buf_addr + self.x2i_tail + index as u32 * 4, *word);
        }
        let old_i2x_tail = bus.host_load32(self.i2x.tail_addr);
        self.x2i_tail += packet_bytes;
        self.next_id = self.next_id.wrapping_add(1);
        bus.host_store32(self.x2i.tail_addr, self.x2i_tail);
        (id, self.x2i_tail, old_i2x_tail)
    }

    fn consume_response(&mut self, bus: &mut Bus, id: u32, opcode: u32) -> Vec<u32> {
        let body_bytes = bus.host_load32(self.i2x.buf_addr + self.i2x_head);
        assert_ne!(body_bytes, 0, "opcode {opcode:#x} produced no response");
        assert_eq!(body_bytes & 3, 0, "opcode {opcode:#x} response is not word-aligned");
        let packet_bytes = 16 + body_bytes;
        assert!(
            self.i2x_head + packet_bytes <= self.i2x.buf_size,
            "test sequence wrapped the context I2X ring"
        );
        assert_eq!(
            bus.host_load32(self.i2x.buf_addr + self.i2x_head + 4),
            0x0001_0000 | body_bytes,
            "opcode {opcode:#x} response protocol",
        );
        assert_eq!(
            bus.host_load32(self.i2x.buf_addr + self.i2x_head + 8),
            id,
            "opcode {opcode:#x} response ID",
        );
        assert_eq!(
            bus.host_load32(self.i2x.buf_addr + self.i2x_head + 12),
            opcode,
            "opcode {opcode:#x} response opcode",
        );

        let body = (0..body_bytes / 4)
            .map(|word| bus.host_load32(self.i2x.buf_addr + self.i2x_head + 16 + word * 4))
            .collect::<Vec<_>>();
        self.i2x_head += packet_bytes;
        assert_eq!(
            bus.host_load32(self.i2x.tail_addr),
            self.i2x_head,
            "opcode {opcode:#x} published extra I2X data",
        );
        bus.host_store32(self.i2x.head_addr, self.i2x_head);
        bus.host_store32(self.i2x.head_addr + 4, 0);
        body
    }
}

#[test]
fn m2c_pinned_initialization_create_context_programs_shared_array() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();

    let boot = proc.boot_to_idle_with_device(&mut device, 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);
    for col in 0..device.cols() {
        assert!(
            !device.array.clock().is_column_active(col as u8),
            "column {col} active before pinned initialization"
        );
    }

    let mut channel = PinnedMgmtChannel::new();
    channel.initialize(&mut proc, &mut device);

    let requested_col = 1u8;
    let column_state_before_context = (0..device.cols())
        .map(|col| device.array.clock().is_column_active(col as u8))
        .collect::<Vec<_>>();
    assert!(
        column_state_before_context.iter().all(|&active| active),
        "RESUME did not ungate all physical columns: {column_state_before_context:?}",
    );
    proc.bus.arm_probe();
    let _context = channel.create_context(&mut proc, &mut device, requested_col, 1);
    let array_accesses = proc.bus.take_probe();

    assert_eq!(
        proc.bus.data_load32(0x21cc + 0x4c),
        u32::MAX,
        "APP-ERT did not complete its endpoint-6 startup handshake",
    );
    assert_eq!(
        (proc.bus.data_load32(0x13df0), proc.bus.data_load32(0x13df4)),
        (0x20, 0x20),
        "endpoint 6 did not consume both APP-ERT startup messages",
    );

    let array_writes = array_accesses
        .iter()
        .filter(|access| access.region == Region::Array && access.is_write)
        .collect::<Vec<_>>();
    assert!(!array_writes.is_empty(), "CREATE_CONTEXT performed no array MMIO writes: {array_accesses:#x?}");
    assert!(
        array_writes
            .iter()
            .all(|access| Bus::decode_array_addr(access.addr).0 == requested_col),
        "CREATE_CONTEXT programmed outside requested column {requested_col}: {array_writes:#x?}",
    );
    let clock_writes = array_writes.iter().filter(|access| {
        Bus::decode_array_addr(access.addr).2 == crate::device::clock_control::COLUMN_CLOCK_CONTROL_OFFSET
    });
    assert!(
        clock_writes.clone().any(|access| access.value & 1 == 0)
            && clock_writes.clone().any(|access| access.value & 1 != 0),
        "CREATE_CONTEXT did not gate and re-enable requested column {requested_col}: {array_writes:#x?}",
    );
    // Once APP-ERT reaches its normal all-events wait, the scheduler releases
    // the requested column clock through its idle resource callback.
    assert!(
        !device.array.clock().is_column_active(requested_col),
        "idle APP-ERT left requested column {requested_col} active: {array_writes:#x?}"
    );
    for col in 0..device.cols() {
        if col as u8 != requested_col {
            assert_eq!(
                device.array.clock().is_column_active(col as u8),
                column_state_before_context[col],
                "CREATE_CONTEXT changed unrequested column {col}",
            );
        }
    }
}

#[test]
fn m2c_clean_destroy_fully_reclaims_context() {
    const HEAP_SIZE: usize = 0x0400_0000;
    const HEAP_A: u64 = 0x6000_0000;
    const HEAP_B: u64 = 0x6800_0000;
    const DEVICE_WORD: u32 = 0x0400_1000;
    const HEAP_OFFSET: u64 = 0x1000;
    const CONTEXT_LIMIT: u32 = 6;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();
    let mut host_memory = crate::device::HostMemory::new();

    let boot = proc.boot_to_idle_with_device(&mut device, 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, &mut device);
    let first = management.create_context(&mut proc, &mut device, 1, 1);
    assert!(
        first.context_id < CONTEXT_LIMIT,
        "firmware context ID {} exceeds the six-slot table",
        first.context_id
    );

    host_memory
        .allocate_region("first context heap", HEAP_A, HEAP_SIZE)
        .expect("allocate first heap");
    host_memory
        .allocate_region("replacement context heap", HEAP_B, HEAP_SIZE)
        .expect("allocate replacement heap");
    host_memory.write_u32(HEAP_A + HEAP_OFFSET, 0xaaaa_aaaa);
    host_memory.write_u32(HEAP_B + HEAP_OFFSET, 0xbbbb_bbbb);

    assert_eq!(
        management.transact(
            &mut proc,
            &mut device,
            0x106,
            &[first.context_id, HEAP_A as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER for first context",
    );
    assert_eq!(
        proc.bus
            .with_device_and_host_memory(&mut device, &mut host_memory)
            .data_load32(DEVICE_WORD),
        0xaaaa_aaaa,
        "first context mapping did not select its heap",
    );

    assert_eq!(management.transact(&mut proc, &mut device, 0x03, &[first.context_id]), [0]);
    assert_ne!(
        proc.bus
            .with_device_and_host_memory(&mut device, &mut host_memory)
            .data_load32(DEVICE_WORD),
        0xaaaa_aaaa,
        "destroyed context still selected its old heap",
    );

    let replacement = management.create_context(&mut proc, &mut device, 1, 1);
    assert_eq!(replacement.context_id, first.context_id, "clean destroy did not release its firmware slot");
    assert_eq!(
        management.transact(
            &mut proc,
            &mut device,
            0x106,
            &[replacement.context_id, HEAP_B as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER for replacement context",
    );
    assert_eq!(
        proc.bus
            .with_device_and_host_memory(&mut device, &mut host_memory)
            .data_load32(DEVICE_WORD),
        0xbbbb_bbbb,
        "reused context slot did not select its replacement heap",
    );
}

#[test]
fn m2c_destroy_rejects_nonempty_completion_ring_until_drained() {
    // The pinned firmware allocates slot 5 first. Its ordinary DESTROY_CONTEXT
    // path requires this AIE completion ring's producer and consumer to match.
    const FIRST_CONTEXT: u32 = 5;
    const COMPLETION_RING_CONSUMER: u32 = 0x2402_c214;
    const COMPLETION_RING_PRODUCER: u32 = 0x2402_c218;
    const MGMT_ERT_BUSY: u32 = 0x0200_0006;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();

    let boot = proc.boot_to_idle_with_device(&mut device, 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, &mut device);
    let context = management.create_context(&mut proc, &mut device, 1, 1);
    assert_eq!(context.context_id, FIRST_CONTEXT);
    let consumer = proc.bus.data_load32(COMPLETION_RING_CONSUMER);
    assert_eq!(
        proc.bus.data_load32(COMPLETION_RING_PRODUCER),
        consumer,
        "fresh completion ring is not empty"
    );

    proc.bus.data_store32(COMPLETION_RING_PRODUCER, (consumer + 1) & 0x7f);
    assert_eq!(
        management.transact(&mut proc, &mut device, 0x03, &[context.context_id]),
        [MGMT_ERT_BUSY],
        "nonempty completion ring must block ordinary DESTROY_CONTEXT",
    );

    proc.bus.data_store32(COMPLETION_RING_PRODUCER, consumer);
    assert_eq!(management.transact(&mut proc, &mut device, 0x03, &[context.context_id]), [0]);
    assert_eq!(
        management.create_context(&mut proc, &mut device, 1, 1).context_id,
        context.context_id,
        "drained context was not reclaimed",
    );
}

#[test]
fn m2c_reclaimed_slot_rejects_destroy_but_accepts_precreate_map() {
    const HEAP_BASE: u64 = 0x6000_0000;
    const HEAP_SIZE: usize = 0x0400_0000;
    const HEAP_OFFSET: u64 = 0x1000;
    const DEVICE_WORD: u32 = 0x0400_1000;
    const MGMT_ERT_INVALID_PARAM: u32 = 0x0200_0004;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();
    let mut host_memory = crate::device::HostMemory::new();
    host_memory
        .allocate_region("stale context heap", HEAP_BASE, HEAP_SIZE)
        .expect("allocate heap");
    host_memory.write_u32(HEAP_BASE + HEAP_OFFSET, 0xcccc_cccc);

    let boot = proc.boot_to_idle_with_device(&mut device, 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, &mut device);
    let context = management.create_context(&mut proc, &mut device, 1, 1);
    assert_eq!(management.transact(&mut proc, &mut device, 0x03, &[context.context_id]), [0]);

    let destroy = management.transact(&mut proc, &mut device, 0x03, &[context.context_id]);
    assert_eq!(destroy.first().copied(), Some(MGMT_ERT_INVALID_PARAM), "double DESTROY_CONTEXT response");

    // The canonical driver cannot send this ordering: it discards the
    // context channel and ID on destroy, then CREATEs before MAP on restart.
    // Keep the signed firmware's direct-mailbox behavior pinned separately.
    let map = management.transact(
        &mut proc,
        &mut device,
        0x106,
        &[context.context_id, HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
    );
    assert_eq!(map, [0], "stale MAP_HOST_BUFFER response");
    assert_eq!(
        proc.bus
            .with_device_and_host_memory(&mut device, &mut host_memory)
            .data_load32(DEVICE_WORD),
        0xcccc_cccc,
        "successful stale MAP_HOST_BUFFER did not install its translation",
    );

    let replacement = management.create_context(&mut proc, &mut device, 1, 1);
    assert_eq!(replacement.context_id, context.context_id, "stale operations consumed the free slot");
    assert_eq!(
        proc.bus
            .with_device_and_host_memory(&mut device, &mut host_memory)
            .data_load32(DEVICE_WORD),
        0xcccc_cccc,
        "reused slot did not retain its pre-CREATE mapping",
    );
}

#[test]
fn m2c_six_live_contexts_exhaust_firmware_slots() {
    const CONTEXT_LIMIT: u32 = 6;
    const MGMT_ERT_NOAVAIL: u32 = 0x0200_0003;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();

    let boot = proc.boot_to_idle_with_device(&mut device, 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, &mut device);

    // The canonical solver first consumes the four ordinary Phoenix columns,
    // then shares the least-used matching partitions.
    let mut context_ids = Vec::new();
    for requested_col in [1, 2, 3, 4, 1, 2] {
        let context = management.create_context(&mut proc, &mut device, requested_col, 1);
        assert!(context.context_id < CONTEXT_LIMIT, "context ID {} exceeds NPU1 limit", context.context_id);
        assert!(
            !context_ids.contains(&context.context_id),
            "live context ID {} was allocated twice",
            context.context_id
        );
        context_ids.push(context.context_id);
    }
    assert_eq!(
        context_ids,
        [5, 4, 3, 2, 1, 0],
        "pinned firmware changed its driver-visible context allocation order",
    );

    let response = management.transact(
        &mut proc,
        &mut device,
        0x02,
        &[1, u32::from_le_bytes([3, 1, 0, 0]), 1, 0, 0, 0, 2],
    );
    assert_eq!(response.first().copied(), Some(MGMT_ERT_NOAVAIL), "seventh CREATE_CONTEXT response");
}

#[test]
fn m2c_unconfigured_cu_fails_before_pdi_loader() {
    const HEAP_BASE: u64 = 0x0400_0000;
    const HEAP_SIZE: usize = 0x0400_0000;
    const CHAIN_ADDR: u64 = HEAP_BASE;
    const INST_ADDR: u64 = HEAP_BASE + 0x1000;
    const INPUT_A_ADDR: u64 = HEAP_BASE + 0x2000;
    const INPUT_B_ADDR: u64 = HEAP_BASE + 0x3000;
    const OUTPUT_ADDR: u64 = HEAP_BASE + 0x4000;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let Some(mlir_aie) = std::env::var_os("MLIR_AIE_PATH") else {
        eprintln!("skip: MLIR_AIE_PATH is not set");
        return;
    };
    let insts_path =
        std::path::PathBuf::from(mlir_aie).join("build/test/npu-xrt/add_one_using_dma/chess/insts.bin");
    let Ok(insts) = std::fs::read(&insts_path) else {
        eprintln!("skip: frozen instruction stream not built at {}", insts_path.display());
        return;
    };
    assert_eq!(insts.len(), 300, "frozen add_one_using_dma instruction bytes");

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut engine = crate::interpreter::engine::InterpreterEngine::new_npu1();

    let boot = proc.boot_to_idle_with_device(engine.device_mut(), 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, engine.device_mut());
    let mut context = management.create_context(&mut proc, engine.device_mut(), 1, 1);

    engine
        .host_memory_mut()
        .allocate_region("pinned Phoenix context heap", HEAP_BASE, HEAP_SIZE)
        .expect("allocate context heap");
    assert_eq!(
        management.transact(
            &mut proc,
            engine.device_mut(),
            0x106,
            &[context.context_id, HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER",
    );

    let regmap = [
        3, // kernel opcode
        0, // 64-bit alignment
        INST_ADDR as u32,
        (INST_ADDR >> 32) as u32,
        (insts.len() / 4) as u32,
        INPUT_A_ADDR as u32,
        (INPUT_A_ADDR >> 32) as u32,
        INPUT_B_ADDR as u32,
        (INPUT_B_ADDR >> 32) as u32,
        OUTPUT_ADDR as u32,
        (OUTPUT_ADDR >> 32) as u32,
        0,
        0,
        0,
        0,
    ];
    let mut slot_words = vec![
        1, // EXEC_NPU_TYPE_NON_ELF
        0,
        0, // inst_buf_addr
        0,
        0, // save_buf_addr
        0,
        0, // restore_buf_addr
        0, // inst_size
        0, // save_size
        0, // restore_size
        0, // inst_prop_cnt
        0, // cu_idx from ERT cu_mask bit 0
        regmap.len() as u32,
    ];
    slot_words.extend(regmap);
    let slot = slot_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(slot.len(), 112, "pinned driver NON_ELF slot size");

    let input = (1u32..=64).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let host_memory = engine.host_memory_mut();
    host_memory.write_bytes(CHAIN_ADDR, &slot);
    host_memory.write_bytes(INST_ADDR, &insts);
    host_memory.write_bytes(INPUT_A_ADDR, &input);

    let (request_id, x2i_tail, old_i2x_tail) = context.post(
        &mut proc.bus,
        0x18,
        &[0, 0, CHAIN_ADDR as u32, (CHAIN_ADDR >> 32) as u32, slot.len() as u32, 1],
    );
    assert_eq!(request_id, 0x1d00_0000);
    assert_eq!(proc.bus.host_load32(context.x2i.head_addr), 0);
    assert_eq!(proc.bus.host_load32(context.x2i.tail_addr), x2i_tail);
    assert_eq!(proc.bus.host_load32(context.i2x.head_addr), 0);

    // Channel-5 X2I publication raises source 37. Firmware copies the command
    // slot, stages the 16 KiB command window, consumes its shared source-76
    // completion, then rejects CU index 0 because CONFIG_CU was never sent.
    let report = pump_runtime(&mut proc, &mut engine, 4, 200_000, |firmware, _| {
        firmware.bus.host_load32(context.i2x.tail_addr) != old_i2x_tail
    });

    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "{report:?}");
    let idle = report.last_firmware.as_ref().unwrap();
    assert!(idle.reached_idle, "{report:?}");
    assert_eq!(idle.wait_reason, Some(WaitReason::Waiti));
    assert_eq!(idle.unresolved_spin, None);
    assert_eq!(idle.unknown_op, None);
    assert_eq!(
        proc.bus.host_load32(context.x2i.head_addr),
        x2i_tail,
        "firmware did not consume the completed request",
    );
    assert_eq!(
        proc.bus.host_load32(context.i2x.tail_addr),
        old_i2x_tail + 28,
        "firmware did not publish the 28-byte response",
    );
    assert_eq!(
        (0..7)
            .map(|word| proc.bus.host_load32(context.i2x.buf_addr + old_i2x_tail + word * 4))
            .collect::<Vec<_>>(),
        vec![0x0000_000c, 0x0001_000c, 0x1d00_0000, 0x0000_0018, 0x0400_0003, 0x0000_0000, 0x0300_0003,],
        "firmware response",
    );
    assert_eq!(
        (
            proc.bus.data_load32(0x2727_1000),
            proc.bus.data_load32(0x2727_100c),
            proc.bus.data_load32(0x2727_1008),
            proc.bus.data_load32(0x2727_1100),
        ),
        (0x74, 3, 0x0000_f9a0, 0),
        "asynchronous management DMA did not complete successfully",
    );
    assert_ne!(proc.bus.data_load32(0x2720_0308) & (1 << 12), 0, "source 76 was not left enabled",);
    assert_eq!(
        (proc.bus.data_load32(0x2720_03b8), proc.bus.data_load32(0x2720_03c4)),
        (0, 0),
        "completion aperture did not deassert source 76",
    );

    let mut host_staging = vec![0; 0x4000];
    engine.host_memory().read_bytes(HEAP_BASE, &mut host_staging);
    assert_eq!(
        (0..0x4000)
            .map(|offset| proc.bus.load_local8(0x0007_d000 + offset))
            .collect::<Vec<_>>(),
        host_staging,
        "asynchronous management DMA did not stage the complete 16 KiB host range",
    );
    assert_eq!((engine.enabled_cores(), engine.device().tiles_with_code()), (0, 0));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConfiguredCuEnvelope {
    Chained,
    PersistentRepeat,
    PostTdrReplay,
    Direct,
    WithheldTctDestroy,
    ExecDpuNoop,
    ExecDpuElf,
}

fn patch_xrt_shim_dma_48(bytes: &mut [u8], offset: usize, address: u64) {
    let low_offset = offset + 4;
    let high_offset = offset + 8;
    let low = u32::from_le_bytes(bytes[low_offset..low_offset + 4].try_into().unwrap());
    let high = u32::from_le_bytes(bytes[high_offset..high_offset + 4].try_into().unwrap());
    let base = (u64::from(high & 0xffff) << 32) | u64::from(low);
    let patched = base + address + crate::device::dma::DDR_AIE_ADDR_OFFSET;
    bytes[low_offset..low_offset + 4].copy_from_slice(&(patched as u32 & !3).to_le_bytes());
    bytes[high_offset..high_offset + 4]
        .copy_from_slice(&((high & 0xffff_0000) | (patched >> 32) as u32).to_le_bytes());
}

fn load_frozen_chess_context_fixture(
    mlir_aie: &std::path::Path,
    fixture: &str,
    xclbin_name: &str,
    xclbin_size: u64,
    insts_size: usize,
    expected_width: u16,
    expected_starts: &[u16],
) -> (Vec<u8>, Vec<u8>, u32) {
    let fixture_dir = mlir_aie.join(format!("build/test/npu-xrt/{fixture}/chess"));
    let xclbin_path = fixture_dir.join(xclbin_name);
    assert_eq!(std::fs::metadata(&xclbin_path).unwrap().len(), xclbin_size, "frozen {fixture} xclbin size");
    let xclbin = crate::parser::Xclbin::from_file(&xclbin_path).expect("parse frozen xclbin");
    let partition_section = xclbin
        .find_section(crate::parser::xclbin::SectionKind::AiePartition)
        .expect("AIE partition");
    let partition =
        crate::parser::AiePartition::parse(partition_section.data()).expect("parse AIE partition");
    assert_eq!(partition.column_width(), expected_width, "{fixture} column width");
    assert_eq!(partition.start_columns(), expected_starts, "{fixture} valid start columns");
    let pdi = partition.primary_pdi().expect("primary PDI").pdi_image.to_vec();

    let embedded = xclbin
        .find_section(crate::parser::xclbin::SectionKind::EmbeddedMetadata)
        .expect("embedded metadata");
    let metadata = std::str::from_utf8(embedded.data()).expect("UTF-8 embedded metadata");
    let functional = metadata
        .split_once("functional=\"")
        .and_then(|(_, value)| value.split_once('"'))
        .map(|(value, _)| value.parse::<u32>().expect("numeric functional"))
        .expect("kernel functional attribute");
    assert_eq!(functional, 0, "{fixture} kernel functional");

    let insts = std::fs::read(fixture_dir.join("insts.bin")).expect("read frozen instruction stream");
    assert_eq!(insts.len(), insts_size, "frozen {fixture} instruction bytes");
    (pdi, insts, functional)
}

fn pinned_config_cu_body(pdi_device_addr: u64, functional: u32) -> Vec<u32> {
    const NPU1_DEV_MEM_BUF_SHIFT: u32 = 15;
    let alignment = 1u64 << NPU1_DEV_MEM_BUF_SHIFT;
    assert_eq!(pdi_device_addr & (alignment - 1), 0, "PDI address alignment");
    let pdi_units = pdi_device_addr >> NPU1_DEV_MEM_BUF_SHIFT;
    assert!(pdi_units <= 0x1ffff, "PDI address does not fit CONFIG_CU");
    assert!(functional <= 0xff, "kernel functional does not fit CONFIG_CU");
    let mut body = vec![0; 33];
    body[0] = 1;
    body[1] = pdi_units as u32 | functional << 17;
    body
}

#[allow(clippy::too_many_arguments)]
fn pinned_chained_exec_body(
    host_memory: &mut crate::device::HostMemory,
    chain_device_addr: u64,
    chain_host_addr: u64,
    inst_device_addr: u64,
    inst_host_addr: u64,
    insts: &[u8],
    input_a_addr: u64,
    input_b_addr: u64,
    output_addr: u64,
) -> Vec<u32> {
    assert_eq!(insts.len() & 3, 0, "instruction stream word alignment");
    host_memory.write_bytes(inst_host_addr, insts);
    let regmap = [
        3,
        0,
        inst_device_addr as u32,
        (inst_device_addr >> 32) as u32,
        (insts.len() / 4) as u32,
        input_a_addr as u32,
        (input_a_addr >> 32) as u32,
        input_b_addr as u32,
        (input_b_addr >> 32) as u32,
        output_addr as u32,
        (output_addr >> 32) as u32,
        0,
        0,
        0,
        0,
    ];
    let mut slot_words = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, regmap.len() as u32];
    slot_words.extend(regmap);
    let slot = slot_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(slot.len(), 112, "pinned driver NON_ELF slot size");
    host_memory.write_bytes(chain_host_addr, &slot);
    vec![0, 0, chain_device_addr as u32, (chain_device_addr >> 32) as u32, slot.len() as u32, 1]
}

fn pump_pinned_context_command(
    proc: &mut FirmwareProcessor,
    engine: &mut crate::interpreter::engine::InterpreterEngine,
    context: &mut PinnedContextChannel,
    opcode: u32,
    body: &[u32],
    max_iterations: u64,
) -> (u32, u32, u32, RuntimePumpReport, Vec<StubAccess>) {
    proc.bus.arm_probe();
    let (id, x2i_tail, old_i2x_tail) = context.post(&mut proc.bus, opcode, body);
    let i2x_tail_addr = context.i2x.tail_addr;
    let report = pump_runtime(proc, engine, max_iterations, 200_000, |firmware, _| {
        firmware.bus.host_load32(i2x_tail_addr) != old_i2x_tail
    });
    let accesses = proc.bus.take_probe();
    (id, x2i_tail, old_i2x_tail, report, accesses)
}

fn consume_pinned_context_response(
    proc: &mut FirmwareProcessor,
    context: &mut PinnedContextChannel,
    id: u32,
    x2i_tail: u32,
    opcode: u32,
    expected: &[u32],
    report: &RuntimePumpReport,
    label: &str,
) {
    let idle = report
        .last_firmware
        .as_ref()
        .expect("completed response has a firmware boundary");
    assert!(idle.reached_idle, "{label}: {report:?}");
    assert_eq!(idle.wait_reason, Some(WaitReason::Waiti), "{label}: {report:?}");
    assert_eq!(idle.unresolved_spin, None, "{label}: {report:?}");
    assert_eq!(idle.unknown_op, None, "{label}: {report:?}");
    assert_eq!(
        proc.bus.host_load32(context.x2i.head_addr),
        x2i_tail,
        "{label}: firmware did not consume request",
    );
    assert_eq!(proc.bus.host_load32(context.x2i.tail_addr), x2i_tail, "{label}: X2I tail");
    assert_eq!(context.consume_response(&mut proc.bus, id, opcode), expected, "{label}: response");
    assert_ne!(proc.bus.take_pending_msix_mask(), 0, "{label}: missing context MSI-X edge");
}

fn array_write_columns(bus: &Bus, accesses: &[StubAccess]) -> std::collections::BTreeSet<u8> {
    accesses
        .iter()
        .filter(|access| access.region == Region::Array && access.is_write)
        .map(|access| bus.decode_live_array_addr(access.addr).expect("probed array address").0)
        .collect()
}

fn assert_configured_cu_executes_frozen_kernel_through_firmware_response(
    compiler: &str,
    xclbin_size: u64,
    pdi_size: usize,
    envelope: ConfiguredCuEnvelope,
) {
    const DEVICE_HEAP_BASE: u64 = 0x0400_0000;
    const HOST_HEAP_BASE: u64 = 0x6000_0000;
    const HEAP_SIZE: usize = 0x0400_0000;
    const PDI_DEVICE_ADDR: u64 = 0x0402_0000;
    const PDI_HOST_ADDR: u64 = 0x6002_0000;
    const CHAIN_DEVICE_ADDR: u64 = DEVICE_HEAP_BASE;
    const CHAIN_HOST_ADDR: u64 = HOST_HEAP_BASE;
    const INST_DEVICE_ADDR: u64 = 0x0402_8000;
    const INST_HOST_ADDR: u64 = 0x6002_8000;
    const INPUT_A_ADDR: u64 = 0x6400_0000;
    const INPUT_B_ADDR: u64 = 0x6400_1000;
    const OUTPUT_ADDR: u64 = 0x6400_2000;
    const NPU1_DEV_MEM_BUF_SHIFT: u32 = 15;
    const PERSISTENT_INPUT_ELEMENTS: usize = 2048;
    const PERSISTENT_REPEAT_COUNT: usize = 3;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let Some(mlir_aie) = std::env::var_os("MLIR_AIE_PATH") else {
        eprintln!("skip: MLIR_AIE_PATH is not set");
        return;
    };
    let (fixture, xclbin_name) = match envelope {
        ConfiguredCuEnvelope::ExecDpuElf => ("add_one_objFifo_elf", "aie.xclbin"),
        ConfiguredCuEnvelope::PersistentRepeat => ("nd_memcpy_linear_repeat", "final.xclbin"),
        _ => ("add_one_using_dma", "aie.xclbin"),
    };
    let fixture_dir =
        std::path::PathBuf::from(mlir_aie).join(format!("build/test/npu-xrt/{fixture}/{compiler}"));
    let mut xrt_nop_elf = None;
    let xrt_xclbin = if envelope == ConfiguredCuEnvelope::ExecDpuNoop {
        use std::io::Write as _;

        let archive = std::path::Path::new("/opt/xilinx/xrt/share/amdxdna/bins/xrt_smi_phx.a");
        if !archive.exists() {
            eprintln!("skip: pinned Phoenix XRT validation archive is not installed");
            return;
        }
        let output = std::process::Command::new("ar")
            .args(["p", archive.to_str().unwrap(), "validate.xclbin"])
            .output()
            .expect("extract pinned XRT validation xclbin");
        assert!(output.status.success(), "ar failed: {}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(output.stdout.len(), 42_542, "pinned XRT validation xclbin size");
        let nop = std::process::Command::new("ar")
            .args(["p", archive.to_str().unwrap(), "nop.elf"])
            .output()
            .expect("extract pinned XRT no-op ELF");
        assert!(nop.status.success(), "ar failed: {}", String::from_utf8_lossy(&nop.stderr));
        xrt_nop_elf = Some(nop.stdout);
        let mut file = tempfile::NamedTempFile::new().expect("create validation xclbin tempfile");
        file.write_all(&output.stdout).expect("write validation xclbin tempfile");
        Some(file)
    } else {
        None
    };
    let xclbin_path = xrt_xclbin
        .as_ref()
        .map_or_else(|| fixture_dir.join(xclbin_name), |file| file.path().to_path_buf());
    if !xclbin_path.exists() {
        eprintln!("skip: frozen {compiler} xclbin not built at {}", xclbin_path.display());
        return;
    }
    if envelope != ConfiguredCuEnvelope::ExecDpuNoop {
        assert_eq!(
            std::fs::metadata(&xclbin_path).unwrap().len(),
            xclbin_size,
            "frozen {compiler} xclbin size"
        );
    }

    let xclbin = crate::parser::Xclbin::from_file(&xclbin_path).expect("parse frozen xclbin");
    let partition_section = xclbin
        .find_section(crate::parser::xclbin::SectionKind::AiePartition)
        .expect("AIE partition");
    let partition =
        crate::parser::AiePartition::parse(partition_section.data()).expect("parse AIE partition");
    if matches!(envelope, ConfiguredCuEnvelope::ExecDpuNoop | ConfiguredCuEnvelope::PersistentRepeat) {
        assert_eq!(partition.start_columns(), [0]);
    } else {
        assert_eq!(partition.start_columns(), [1, 2, 3, 4]);
    }
    let pdi = partition.primary_pdi().expect("primary PDI").pdi_image.to_vec();
    if envelope == ConfiguredCuEnvelope::ExecDpuNoop {
        assert_eq!(pdi.len(), 8816, "pinned XRT validation primary PDI size");
    } else {
        assert_eq!(pdi.len(), pdi_size, "frozen {compiler} primary PDI size");
    }

    let embedded = xclbin
        .find_section(crate::parser::xclbin::SectionKind::EmbeddedMetadata)
        .expect("embedded metadata");
    let metadata = std::str::from_utf8(embedded.data()).expect("UTF-8 embedded metadata");
    let functional = metadata
        .split_once("functional=\"")
        .and_then(|(_, value)| value.split_once('"'))
        .map(|(value, _)| value.parse::<u32>().expect("numeric functional"))
        .expect("kernel functional attribute");
    assert_eq!(functional, 0, "frozen kernel functional");

    let insts = match envelope {
        ConfiguredCuEnvelope::ExecDpuNoop => {
            let elf = xrt_nop_elf.as_deref().expect("pinned XRT no-op ELF");
            let ctrltext =
                crate::npu::NpuInstructionStream::elf_ctrltext(elf).expect("extract no-op .ctrltext");
            assert_eq!(ctrltext.len(), 20, "pinned no-op control text size");
            ctrltext.to_vec()
        }
        ConfiguredCuEnvelope::ExecDpuElf => {
            let elf = std::fs::read(fixture_dir.join("insts.elf")).expect("read transaction ELF");
            let mut ctrltext = crate::npu::NpuInstructionStream::elf_ctrltext(&elf)
                .expect("extract transaction .ctrltext")
                .to_vec();
            assert_eq!(ctrltext.len(), 0x148, "pinned transaction control text size");
            patch_xrt_shim_dma_48(&mut ctrltext, 0x20, OUTPUT_ADDR);
            patch_xrt_shim_dma_48(&mut ctrltext, 0xb4, INPUT_A_ADDR);
            ctrltext
        }
        _ => {
            let insts = std::fs::read(fixture_dir.join("insts.bin")).expect("read frozen instruction stream");
            assert_eq!(insts.len(), 300, "frozen add_one_using_dma instruction bytes");
            insts
        }
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut engine = crate::interpreter::engine::InterpreterEngine::new_npu1();

    let boot = proc.boot_to_idle_with_device(engine.device_mut(), 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, engine.device_mut());
    let (context_start, context_cols) = if envelope == ConfiguredCuEnvelope::ExecDpuNoop {
        (0, 5)
    } else {
        (1, 1)
    };
    let mut context = management.create_context(&mut proc, engine.device_mut(), context_start, context_cols);

    engine
        .host_memory_mut()
        .allocate_region("pinned Phoenix context heap", HOST_HEAP_BASE, HEAP_SIZE)
        .expect("allocate context heap");
    engine
        .host_memory_mut()
        .allocate_region(
            "pinned Phoenix data BOs",
            INPUT_A_ADDR,
            if envelope == ConfiguredCuEnvelope::PersistentRepeat {
                0x5000
            } else {
                0x3000
            },
        )
        .expect("allocate data BOs");
    assert_eq!(
        management.transact(
            &mut proc,
            engine.device_mut(),
            0x106,
            &[context.context_id, HOST_HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER",
    );
    engine.host_memory_mut().write_bytes(PDI_HOST_ADDR, &pdi);

    // Pinned open-driver wire contract: NPU1 uses 32 KiB device-memory
    // address units; CONFIG_CU stores address bits 16:0 and function bits 24:17.
    let pdi_alignment = 1u64 << NPU1_DEV_MEM_BUF_SHIFT;
    assert_eq!(PDI_DEVICE_ADDR & (pdi_alignment - 1), 0, "PDI address alignment");
    let pdi_units = PDI_DEVICE_ADDR >> NPU1_DEV_MEM_BUF_SHIFT;
    assert!(pdi_units <= 0x1ffff, "PDI address does not fit CONFIG_CU");
    assert!(functional <= 0xff, "kernel functional does not fit CONFIG_CU");
    let mut config_body = vec![0; 33];
    config_body[0] = 1;
    config_body[1] = pdi_units as u32 | functional << 17;

    let (config_id, _, old_config_i2x_tail) = context.post(&mut proc.bus, 0x11, &config_body);
    let config_report = pump_runtime(&mut proc, &mut engine, 4, 200_000, |firmware, _| {
        firmware.bus.host_load32(context.i2x.tail_addr) != old_config_i2x_tail
    });
    assert_eq!(config_report.stop, RuntimePumpStop::ResponseCompleted, "{config_report:?}");
    // Signed firmware 1.5.5.391 produces this exact successful response;
    // mailbox-body tracing on physical NPU1 cross-checked it.
    assert_eq!(context.consume_response(&mut proc.bus, config_id, 0x11), [0], "CONFIG_CU status");
    if matches!(
        envelope,
        ConfiguredCuEnvelope::PersistentRepeat
            | ConfiguredCuEnvelope::PostTdrReplay
            | ConfiguredCuEnvelope::WithheldTctDestroy
    ) {
        assert_eq!(proc.bus.take_pending_msix_mask(), 1 << 5, "CONFIG_CU context MSI-X edge");
    }

    let mut pdi_after = vec![0; pdi.len()];
    engine.host_memory().read_bytes(PDI_HOST_ADDR, &mut pdi_after);
    assert_eq!(pdi_after, pdi, "firmware changed the registered PDI bytes");

    let regmap = [
        3,
        0,
        INST_DEVICE_ADDR as u32,
        (INST_DEVICE_ADDR >> 32) as u32,
        (insts.len() / 4) as u32,
        INPUT_A_ADDR as u32,
        (INPUT_A_ADDR >> 32) as u32,
        INPUT_B_ADDR as u32,
        (INPUT_B_ADDR >> 32) as u32,
        OUTPUT_ADDR as u32,
        (OUTPUT_ADDR >> 32) as u32,
        0,
        0,
        0,
        0,
    ];
    let input = if envelope == ConfiguredCuEnvelope::PersistentRepeat {
        (0..PERSISTENT_INPUT_ELEMENTS)
            .flat_map(|index| (2 * index as i16 + 1).to_le_bytes())
            .collect::<Vec<_>>()
    } else {
        (1u32..=64).flat_map(u32::to_le_bytes).collect::<Vec<_>>()
    };
    let host_memory = engine.host_memory_mut();
    host_memory.write_bytes(INST_HOST_ADDR, &insts);
    host_memory.write_bytes(INPUT_A_ADDR, &input);

    // The same physical trace cross-checked direct [0] and chained [0, 0, 0].
    let (exec_opcode, exec_body, expected_response) = match envelope {
        ConfiguredCuEnvelope::Chained
        | ConfiguredCuEnvelope::PersistentRepeat
        | ConfiguredCuEnvelope::PostTdrReplay
        | ConfiguredCuEnvelope::WithheldTctDestroy => {
            let mut slot_words = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, regmap.len() as u32];
            slot_words.extend(regmap);
            let slot = slot_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
            assert_eq!(slot.len(), 112, "pinned driver NON_ELF slot size");
            host_memory.write_bytes(CHAIN_HOST_ADDR, &slot);
            (
                0x18,
                vec![0, 0, CHAIN_DEVICE_ADDR as u32, (CHAIN_DEVICE_ADDR >> 32) as u32, slot.len() as u32, 1],
                vec![0, 0, 0],
            )
        }
        ConfiguredCuEnvelope::Direct => {
            let mut body = Vec::with_capacity(20);
            body.push(0); // cu_idx from ERT cu_mask bit 0
            body.extend(regmap);
            // The pinned driver leaves this fixed-size tail uninitialized.
            // A nonzero sentinel proves firmware does not consume it.
            body.extend([0xa5a5_a5a5; 4]);
            assert_eq!(body.len(), 20, "pinned driver EXECUTE_BUFFER_CF request words");
            (0x0c, body, vec![0])
        }
        ConfiguredCuEnvelope::ExecDpuNoop | ConfiguredCuEnvelope::ExecDpuElf => {
            let mut body = vec![
                INST_DEVICE_ADDR as u32,
                (INST_DEVICE_ADDR >> 32) as u32,
                insts.len() as u32,
                0, // no instruction properties
                0, // cu_idx from ERT cu_mask bit 0
                3, // pinned XRT latency profile argument
            ];
            body.resize(40, 0);
            (0x10, body, vec![0])
        }
    };

    if envelope != ConfiguredCuEnvelope::WithheldTctDestroy {
        proc.bus.arm_probe();
    }
    let (exec_id, x2i_tail, old_exec_i2x_tail) = context.post(&mut proc.bus, exec_opcode, &exec_body);
    if envelope == ConfiguredCuEnvelope::WithheldTctDestroy {
        assert!(
            !engine
                .device()
                .array
                .dma_engine(1, 0)
                .expect("assigned shim DMA")
                .has_task_token_for_channel(0),
            "shim S2MM0 token predates execution",
        );
        let completion = (1..=100_000).find_map(|cycle| {
            let boundary = proc.run_to_boundary_with_engine(&mut engine, 200_000);
            assert!(boundary.reached_idle, "firmware left its scheduler wait: {boundary:?}");
            assert_eq!(boundary.unresolved_spin, None, "{boundary:?}");
            assert_eq!(boundary.unknown_op, None, "{boundary:?}");
            engine.force_running();
            engine.step();
            engine
                .device()
                .array
                .dma_engine(1, 0)
                .expect("assigned shim DMA")
                .has_task_token_for_channel(0)
                .then_some((cycle, boundary))
        });
        let (cycles, boundary) = completion.expect("execution did not produce shim S2MM0 completion");
        assert_eq!(
            proc.bus.host_load32(context.i2x.tail_addr),
            old_exec_i2x_tail,
            "firmware responded without the withheld TCT after {cycles} cycles: {boundary:?}",
        );
        let destroyed_id = context.context_id;
        // This corrects the earlier hardware-log inference that a missing
        // completion alone forces MGMT_ERT_BUSY. The physical BUSY remains
        // real, but needs a narrower firmware-visible failure state.
        assert_eq!(
            management.transact(&mut proc, engine.device_mut(), 0x03, &[destroyed_id]),
            [0],
            "DESTROY_CONTEXT status",
        );
        assert!(
            engine
                .device()
                .tile(1, 2)
                .unwrap()
                .program_memory()
                .unwrap()
                .iter()
                .all(|&byte| byte == 0),
            "successful destroy did not zeroize program memory",
        );
        assert_eq!(
            management.create_context(&mut proc, engine.device_mut(), 2, 1).context_id,
            destroyed_id,
            "destroyed context slot was not reused",
        );
        return;
    }

    let report = pump_runtime(&mut proc, &mut engine, 100_000, 200_000, |firmware, _| {
        firmware.bus.host_load32(context.i2x.tail_addr) != old_exec_i2x_tail
    });
    let array_accesses = proc.bus.take_probe();

    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "{report:?}");
    let idle = report.last_firmware.as_ref().unwrap();
    assert!(idle.reached_idle, "{report:?}");
    assert_eq!(idle.wait_reason, Some(WaitReason::Waiti));
    assert_eq!(idle.unresolved_spin, None);
    assert_eq!(idle.unknown_op, None);
    assert_eq!(
        proc.bus.host_load32(context.x2i.head_addr),
        x2i_tail,
        "firmware did not consume the completed request",
    );
    assert_eq!(proc.bus.host_load32(context.x2i.tail_addr), x2i_tail);
    assert_eq!(
        context.consume_response(&mut proc.bus, exec_id, exec_opcode),
        expected_response,
        "execution response",
    );
    if matches!(envelope, ConfiguredCuEnvelope::ExecDpuNoop | ConfiguredCuEnvelope::ExecDpuElf) {
        assert!(
            proc.bus.scan_bytes(&insts).iter().any(|&(region, _)| region == "local_data"),
            "EXEC_DPU did not stage the pinned control text in firmware-local memory",
        );
    }

    assert!(
        !array_accesses.iter().any(|access| {
            access.region == Region::System
                && (DEVICE_HEAP_BASE..DEVICE_HEAP_BASE + HEAP_SIZE as u64).contains(&(access.addr as u64))
        }),
        "firmware device-heap access escaped the selected host mapping: {array_accesses:#x?}",
    );
    if envelope == ConfiguredCuEnvelope::PersistentRepeat {
        let mut output = vec![0; input.len() * PERSISTENT_REPEAT_COUNT];
        engine.host_memory().read_bytes(OUTPUT_ADDR, &mut output);
        assert_eq!(output, input.repeat(PERSISTENT_REPEAT_COUNT), "persistent A1 output");
    } else {
        let output = (0..64)
            .map(|index| engine.host_memory().read_u32(OUTPUT_ADDR + index * 4))
            .collect::<Vec<_>>();
        match envelope {
            ConfiguredCuEnvelope::ExecDpuNoop => {
                assert_eq!(output, vec![0; 64], "no-op changed output memory")
            }
            ConfiguredCuEnvelope::ExecDpuElf => {
                assert_eq!(
                    output,
                    (42..=105).collect::<Vec<_>>(),
                    "transaction ELF kernel output; {report:?}"
                )
            }
            _ => assert_eq!(output, (2..=65).collect::<Vec<_>>(), "frozen kernel output"),
        }
    }
    assert!(
        !engine
            .device()
            .array
            .dma_engine(1, 0)
            .expect("assigned shim DMA")
            .has_task_token_for_channel(0),
        "shim S2MM0 completion token was not consumed",
    );

    let pdi_array_writes = array_accesses
        .iter()
        .filter(|access| access.region == Region::Array && access.is_write)
        .collect::<Vec<_>>();
    assert!(!pdi_array_writes.is_empty(), "configured PDI produced no array writes: {report:?}");
    if envelope != ConfiguredCuEnvelope::ExecDpuNoop {
        assert!(
            pdi_array_writes.iter().all(|access| Bus::decode_array_addr(access.addr).0 == 1),
            "PDI wrote outside assigned physical column 1: {pdi_array_writes:#x?}",
        );
    }

    let device = engine.device();
    if envelope == ConfiguredCuEnvelope::ExecDpuNoop {
        assert_eq!(device.tiles_with_code(), 1, "validation PDI program-memory footprint");
        assert_eq!(device.enabled_cores(), 1, "validation PDI core-enable footprint");
        let compute = device.tile(0, 2).expect("reserved DPU compute tile");
        assert!(
            compute.program_memory().unwrap().iter().any(|&byte| byte != 0),
            "validation PDI left DPU program memory empty",
        );
    } else {
        assert_eq!(device.tiles_with_code(), 1, "configured PDI program-memory footprint");
        assert_eq!(device.enabled_cores(), 1, "configured PDI core-enable footprint");
        let compute = device.tile(1, 2).expect("assigned compute tile");
        assert!(
            compute.program_memory().unwrap().iter().any(|&byte| byte != 0),
            "program memory remained empty"
        );
        assert!(compute.data_memory().iter().any(|&byte| byte != 0), "data memory remained empty");
        assert_ne!(compute.core.control & 1, 0, "PDI did not configure Core_Control");
    }
    assert!(device.tile(0, 0).is_none(), "physical column 0 must not expose a shim tile");
    assert!(device.tile(0, 1).is_some(), "physical column 0 must expose its reserved memory tile");
    for row in 2..device.rows() {
        assert!(
            device.tile(0, row).is_some(),
            "physical column 0 must expose reserved compute tile row {row}"
        );
    }

    if envelope == ConfiguredCuEnvelope::PersistentRepeat {
        assert_eq!(proc.bus.take_pending_msix_mask(), 1 << 5, "A1 context MSI-X edge");
        let a2_input = (0..PERSISTENT_INPUT_ELEMENTS)
            .flat_map(|index| (2 * index as i16 + 2).to_le_bytes())
            .collect::<Vec<_>>();
        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(INPUT_A_ADDR, &a2_input);
        host_memory.write_bytes(OUTPUT_ADDR, &vec![0xef; a2_input.len() * PERSISTENT_REPEAT_COUNT]);

        let (id, x2i, _, a2_report, a2_accesses) =
            pump_pinned_context_command(&mut proc, &mut engine, &mut context, 0x18, &exec_body, 100_000);
        assert_eq!(array_write_columns(&proc.bus, &a2_accesses), [1].into_iter().collect());
        assert_eq!(a2_report.stop, RuntimePumpStop::ResponseCompleted, "A2: {a2_report:?}");
        consume_pinned_context_response(
            &mut proc,
            &mut context,
            id,
            x2i,
            0x18,
            &[0, 0, 0],
            &a2_report,
            "A2 CHAIN_EXEC_NPU",
        );
        let mut a2_output = vec![0; a2_input.len() * PERSISTENT_REPEAT_COUNT];
        engine.host_memory().read_bytes(OUTPUT_ADDR, &mut a2_output);
        assert_eq!(a2_output, a2_input.repeat(PERSISTENT_REPEAT_COUNT), "persistent A2 output");
        assert!(
            !engine
                .device()
                .array
                .dma_engine(1, 0)
                .expect("assigned shim DMA")
                .has_task_token_for_channel(0),
            "A2 shim S2MM0 completion token was not consumed",
        );
    }

    if envelope == ConfiguredCuEnvelope::PostTdrReplay {
        assert_eq!(proc.bus.take_pending_msix_mask(), 1 << 5, "A1 context MSI-X edge");
        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(INPUT_A_ADDR, &input);
        host_memory.write_bytes(OUTPUT_ADDR, &vec![0xef; 64 * 4]);

        let old_x2i_head = proc.bus.host_load32(context.x2i.head_addr);
        let (_, a2_x2i_tail, old_a2_i2x_tail, a2_report, a2_accesses) =
            pump_pinned_context_command(&mut proc, &mut engine, &mut context, 0x18, &exec_body, 100_000);
        assert_eq!(array_write_columns(&proc.bus, &a2_accesses), [1].into_iter().collect());
        assert!(
            matches!(
                a2_report.stop,
                RuntimePumpStop::ArrayIdleFirmwareWaiting | RuntimePumpStop::NoProgressExhausted
            ),
            "the frozen one-shot core unexpectedly completed A2: {a2_report:?}",
        );
        assert_eq!(
            proc.bus.host_load32(context.x2i.head_addr),
            old_x2i_head,
            "A2 request was consumed without a response: {a2_report:?}",
        );
        assert_eq!(proc.bus.host_load32(context.x2i.tail_addr), a2_x2i_tail, "A2 X2I tail");
        assert_eq!(
            proc.bus.host_load32(context.i2x.tail_addr),
            old_a2_i2x_tail,
            "A2 changed its response tail without completing",
        );
        assert_eq!(proc.bus.take_pending_msix_mask(), 0, "A2 raised a context MSI-X edge");

        let destroyed_id = context.context_id;
        assert_eq!(
            management.transact(&mut proc, engine.device_mut(), 0x03, &[destroyed_id]),
            [0],
            "DESTROY_CONTEXT during unresolved A2",
        );
        assert!(
            !engine.device().array.dma_engine(1, 0).unwrap().any_channel_active(),
            "DESTROY_CONTEXT left shim DMA active",
        );
        let mut recovered = management.create_context(&mut proc, engine.device_mut(), 1, 1);
        assert_eq!(recovered.context_id, destroyed_id, "recovery did not reuse the firmware context slot");
        assert_eq!(
            management.transact(
                &mut proc,
                engine.device_mut(),
                0x106,
                &[recovered.context_id, HOST_HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
            ),
            [0],
            "recovery MAP_HOST_BUFFER",
        );

        let (id, x2i, _, replay_report, replay_accesses) =
            pump_pinned_context_command(&mut proc, &mut engine, &mut recovered, 0x11, &config_body, 4);
        assert_eq!(
            array_write_columns(&proc.bus, &replay_accesses),
            [1].into_iter().collect(),
            "recovery CONFIG_CU array writes",
        );
        assert_eq!(
            replay_report.stop,
            RuntimePumpStop::ResponseCompleted,
            "recovery CONFIG: {replay_report:?}"
        );
        consume_pinned_context_response(
            &mut proc,
            &mut recovered,
            id,
            x2i,
            0x11,
            &[0],
            &replay_report,
            "recovery CONFIG_CU",
        );

        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(INPUT_A_ADDR, &input);
        host_memory.write_bytes(OUTPUT_ADDR, &vec![0xef; 64 * 4]);
        let (id, x2i, _, a3_report, a3_accesses) =
            pump_pinned_context_command(&mut proc, &mut engine, &mut recovered, 0x18, &exec_body, 100_000);
        assert_eq!(array_write_columns(&proc.bus, &a3_accesses), [1].into_iter().collect());
        assert_eq!(a3_report.stop, RuntimePumpStop::ResponseCompleted, "A3: {a3_report:?}");
        consume_pinned_context_response(
            &mut proc,
            &mut recovered,
            id,
            x2i,
            0x18,
            &[0, 0, 0],
            &a3_report,
            "A3 CHAIN_EXEC_NPU",
        );
        assert_eq!(
            (0..64)
                .map(|index| engine.host_memory().read_u32(OUTPUT_ADDR + index * 4))
                .collect::<Vec<_>>(),
            (2..=65).collect::<Vec<_>>(),
            "A3 output",
        );
        assert!(
            !engine
                .device()
                .array
                .dma_engine(1, 0)
                .expect("assigned shim DMA")
                .has_task_token_for_channel(0),
            "A3 shim S2MM0 completion token was not consumed",
        );
        assert_eq!(
            management.transact(&mut proc, engine.device_mut(), 0x03, &[recovered.context_id]),
            [0],
            "final DESTROY_CONTEXT",
        );
    }
}

#[test]
fn m2c_configured_cu_executes_frozen_chess_kernel_through_firmware_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9671,
        3216,
        ConfiguredCuEnvelope::Chained,
    );
}

#[test]
fn m2c_persistent_kernel_completes_two_same_context_submissions() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9751,
        3296,
        ConfiguredCuEnvelope::PersistentRepeat,
    );
}

#[test]
fn m2c_post_tdr_replay_restores_execution() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9671,
        3216,
        ConfiguredCuEnvelope::PostTdrReplay,
    );
}

#[test]
fn m2c_configured_cu_executes_frozen_peano_kernel_through_firmware_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "peano",
        9062,
        2608,
        ConfiguredCuEnvelope::Chained,
    );
}

#[test]
fn m2c_configured_cu_executes_frozen_chess_kernel_through_direct_firmware_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9671,
        3216,
        ConfiguredCuEnvelope::Direct,
    );
}

#[test]
fn m2c_context_waiting_on_withheld_tct_can_be_destroyed_and_reused() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9671,
        3216,
        ConfiguredCuEnvelope::WithheldTctDestroy,
    );
}

#[test]
fn m2c_configured_cu_executes_pinned_xrt_nop_through_direct_dpu_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9671,
        3216,
        ConfiguredCuEnvelope::ExecDpuNoop,
    );
}

#[test]
fn m2c_configured_cu_executes_pinned_chess_elf_through_direct_dpu_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response(
        "chess",
        9607,
        3152,
        ConfiguredCuEnvelope::ExecDpuElf,
    );
}

#[test]
fn m2c_same_client_a_b_a_matches_observed_placement_and_nonresponse() {
    const DEVICE_HEAP_BASE: u64 = 0x0400_0000;
    const HOST_HEAP_BASE: u64 = 0x6000_0000;
    const HEAP_SIZE: usize = 0x0400_0000;

    const A_CHAIN_DEVICE: u64 = 0x0400_0000;
    const A_CHAIN_HOST: u64 = 0x6000_0000;
    const A_PDI_DEVICE: u64 = 0x0402_0000;
    const A_PDI_HOST: u64 = 0x6002_0000;
    const A_INST_DEVICE: u64 = 0x0402_8000;
    const A_INST_HOST: u64 = 0x6002_8000;
    const A_INPUT: u64 = 0x6400_0000;
    const A_UNUSED: u64 = 0x6400_1000;
    const A_OUTPUT: u64 = 0x6400_2000;

    const B_CHAIN_DEVICE: u64 = 0x0410_0000;
    const B_CHAIN_HOST: u64 = 0x6010_0000;
    const B_PDI_DEVICE: u64 = 0x0412_0000;
    const B_PDI_HOST: u64 = 0x6012_0000;
    const B_INST_DEVICE: u64 = 0x0412_8000;
    const B_INST_HOST: u64 = 0x6012_8000;
    const B_INPUT: u64 = 0x6500_0000;
    const B_UNUSED: u64 = 0x6500_4000;
    const B_OUTPUT: u64 = 0x6500_8000;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let Some(mlir_aie) = std::env::var_os("MLIR_AIE_PATH") else {
        eprintln!("skip: MLIR_AIE_PATH is not set");
        return;
    };
    let mlir_aie = std::path::PathBuf::from(mlir_aie);
    let (a_pdi, a_insts, a_functional) = load_frozen_chess_context_fixture(
        &mlir_aie,
        "add_one_using_dma",
        "aie.xclbin",
        9671,
        300,
        1,
        &[1, 2, 3, 4],
    );
    let (b_pdi, b_insts, b_functional) = load_frozen_chess_context_fixture(
        &mlir_aie,
        "device_width",
        "final.xclbin",
        7362,
        344,
        2,
        &[1, 2, 3],
    );

    let raw = std::fs::read(path).expect("read firmware");
    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let mut proc = FirmwareProcessor::load_m2c(image);
    let mut engine = crate::interpreter::engine::InterpreterEngine::new_npu1();
    let boot = proc.boot_to_idle_with_device(engine.device_mut(), 200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);

    let host_memory = engine.host_memory_mut();
    host_memory
        .allocate_region("shared same-client Phoenix heap", HOST_HEAP_BASE, HEAP_SIZE)
        .expect("allocate shared context heap");
    host_memory
        .allocate_region("context A data BOs", A_INPUT, 0x3000)
        .expect("allocate context A data BOs");
    host_memory
        .allocate_region("context B data BOs", B_INPUT, 0xc000)
        .expect("allocate context B data BOs");

    let mut management = PinnedMgmtChannel::new();
    management.initialize(&mut proc, engine.device_mut());

    let mut context_a = management.create_context(&mut proc, engine.device_mut(), 1, 1);
    assert_eq!(
        management.transact(
            &mut proc,
            engine.device_mut(),
            0x106,
            &[context_a.context_id, HOST_HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER for context A",
    );
    engine.host_memory_mut().write_bytes(A_PDI_HOST, &a_pdi);
    let a_config_body = pinned_config_cu_body(A_PDI_DEVICE, a_functional);
    let (id, x2i, _, report, accesses) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context_a, 0x11, &a_config_body, 4);
    assert_eq!(array_write_columns(&proc.bus, &accesses), [1].into_iter().collect());
    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "A CONFIG: {report:?}");
    consume_pinned_context_response(&mut proc, &mut context_a, id, x2i, 0x11, &[0], &report, "A CONFIG_CU");

    let a_input = (1u32..=64).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let a_exec_body = {
        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(A_INPUT, &a_input);
        host_memory.write_bytes(A_OUTPUT, &vec![0xef; 64 * 4]);
        pinned_chained_exec_body(
            host_memory,
            A_CHAIN_DEVICE,
            A_CHAIN_HOST,
            A_INST_DEVICE,
            A_INST_HOST,
            &a_insts,
            A_INPUT,
            A_UNUSED,
            A_OUTPUT,
        )
    };
    let (id, x2i, _, report, accesses) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context_a, 0x18, &a_exec_body, 100_000);
    assert_eq!(array_write_columns(&proc.bus, &accesses), [1].into_iter().collect());
    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "A1: {report:?}");
    consume_pinned_context_response(
        &mut proc,
        &mut context_a,
        id,
        x2i,
        0x18,
        &[0, 0, 0],
        &report,
        "A1 CHAIN_EXEC_NPU",
    );
    assert_eq!(
        (0..64)
            .map(|index| engine.host_memory().read_u32(A_OUTPUT + index * 4))
            .collect::<Vec<_>>(),
        (2..=65).collect::<Vec<_>>(),
        "A1 output",
    );

    let mut context_b = management.create_context(&mut proc, engine.device_mut(), 2, 2);
    assert_ne!(context_b.context_id, context_a.context_id, "contexts share a firmware ID");
    assert_ne!(context_b.x2i.tail_addr, context_a.x2i.tail_addr, "contexts share an X2I queue");
    assert_ne!(context_b.i2x.tail_addr, context_a.i2x.tail_addr, "contexts share an I2X queue");
    assert_eq!(
        management.transact(
            &mut proc,
            engine.device_mut(),
            0x106,
            &[context_b.context_id, HOST_HEAP_BASE as u32, 0, HEAP_SIZE as u32, 0],
        ),
        [0],
        "MAP_HOST_BUFFER for context B",
    );

    const SHARED_MAPPING_OFFSET: u64 = HEAP_SIZE as u64 - 4;
    const SHARED_MAPPING_SENTINEL: u32 = 0x51a9_c3e7;
    engine
        .host_memory_mut()
        .write_u32(HOST_HEAP_BASE + SHARED_MAPPING_OFFSET, SHARED_MAPPING_SENTINEL);
    let mapped = {
        let (device, host_memory) = engine.device_and_host_memory();
        proc.bus
            .with_device_and_host_memory(device, host_memory)
            .data_load32((DEVICE_HEAP_BASE + SHARED_MAPPING_OFFSET) as u32)
    };
    assert_eq!(mapped, SHARED_MAPPING_SENTINEL, "identical mappings did not select the shared heap");

    engine.host_memory_mut().write_bytes(B_PDI_HOST, &b_pdi);
    let b_config_body = pinned_config_cu_body(B_PDI_DEVICE, b_functional);
    let (id, x2i, _, report, accesses) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context_b, 0x11, &b_config_body, 4);
    let columns = array_write_columns(&proc.bus, &accesses);
    assert!(
        !columns.is_empty() && columns.iter().all(|column| (2..=3).contains(column)) && columns.contains(&3),
        "B CONFIG escaped physical columns 2-3: {columns:?}",
    );
    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "B CONFIG: {report:?}");
    consume_pinned_context_response(&mut proc, &mut context_b, id, x2i, 0x11, &[0], &report, "B CONFIG_CU");

    let b_input = (1u32..=4096).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let b_exec_body = {
        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(B_INPUT, &b_input);
        host_memory.write_bytes(B_OUTPUT, &vec![0xef; 4096 * 4]);
        pinned_chained_exec_body(
            host_memory,
            B_CHAIN_DEVICE,
            B_CHAIN_HOST,
            B_INST_DEVICE,
            B_INST_HOST,
            &b_insts,
            B_INPUT,
            B_UNUSED,
            B_OUTPUT,
        )
    };
    let (id, x2i, _, report, accesses) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context_b, 0x18, &b_exec_body, 100_000);
    let columns = array_write_columns(&proc.bus, &accesses);
    assert!(
        !columns.is_empty() && columns.iter().all(|column| (2..=3).contains(column)) && columns.contains(&3),
        "B execution escaped physical columns 2-3: {columns:?}",
    );
    assert_eq!(report.stop, RuntimePumpStop::ResponseCompleted, "B execution: {report:?}");
    assert_eq!(
        (0..4096)
            .map(|index| engine.host_memory().read_u32(B_OUTPUT + index * 4))
            .collect::<Vec<_>>(),
        (1..=4096).collect::<Vec<_>>(),
        "B output",
    );
    consume_pinned_context_response(
        &mut proc,
        &mut context_b,
        id,
        x2i,
        0x18,
        &[0, 0, 0],
        &report,
        "B CHAIN_EXEC_NPU",
    );

    {
        let host_memory = engine.host_memory_mut();
        host_memory.write_bytes(A_INPUT, &a_input);
        host_memory.write_bytes(A_OUTPUT, &vec![0xef; 64 * 4]);
    }
    let old_x2i_head = proc.bus.host_load32(context_a.x2i.head_addr);
    let (_, x2i, old_i2x_tail, report, accesses) =
        pump_pinned_context_command(&mut proc, &mut engine, &mut context_a, 0x18, &a_exec_body, 100_000);
    let columns = array_write_columns(&proc.bus, &accesses);
    assert!(columns.iter().all(|column| *column == 1), "A2 escaped physical column 1: {columns:?}",);
    // Pin the physical external result, not an internal cause. The finite A
    // program makes this guard unsuitable for distinguishing relaunch failure
    // from a B-induced context interaction without a separate A1 -> A2 control.
    assert!(
        matches!(
            report.stop,
            RuntimePumpStop::ArrayIdleFirmwareWaiting | RuntimePumpStop::NoProgressExhausted
        ),
        "the frozen one-shot A core unexpectedly completed A2: {report:?}",
    );
    assert_eq!(
        proc.bus.host_load32(context_a.x2i.head_addr),
        old_x2i_head,
        "A2 request was consumed without a response: {report:?}",
    );
    assert_eq!(proc.bus.host_load32(context_a.x2i.tail_addr), x2i, "A2 X2I tail");
    assert_eq!(
        proc.bus.host_load32(context_a.i2x.tail_addr),
        old_i2x_tail,
        "A2 changed its response tail without completing",
    );
}
#[test]
fn m2c_first_pinned_startup_command_reaches_firmware_response() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let boot = proc.boot_to_idle(200_000);
    assert!(boot.reached_idle, "firmware did not reach its natural scheduler wait: {boot:?}");

    // Pinned driver 216cefe's first post-alive command is
    // SET_RUNTIME_CONFIG(2, 1). These are its literal little-endian wire
    // words: 16-byte mailbox header followed by the packed 12-byte request.
    let request = [
        0x0000_000c, // body bytes
        0x0001_000c, // protocol 1, body bytes 12
        0x1d00_0000, // first driver message ID
        0x0000_010a, // MSG_OP_SET_RUNTIME_CONFIG
        0x0000_0002, // runtime-config type
        0x0000_0001, // u64 value, low word
        0x0000_0000, // u64 value, high word
    ];
    for (index, word) in request.into_iter().enumerate() {
        proc.bus.host_store32(0x030b_c000 + index as u32 * 4, word);
    }
    proc.bus.host_store32(0x030b_f000, 0);
    proc.bus.host_store32(0x030e_d008, 0);
    proc.bus.host_store32(0x030e_c000, 28);

    let handled = proc.boot_to_idle(200_000);
    assert!(handled.reached_idle, "firmware did not return to idle: {handled:?}");
    assert_eq!(handled.unknown_op, None);
    assert_eq!(handled.unresolved_spin, None);

    assert_eq!(proc.bus.host_load32(0x030e_c004), 28, "firmware must consume the complete request");
    assert_eq!(proc.bus.host_load32(0x030e_d000), 20, "firmware must publish one 20-byte response");

    let response = [
        0x0000_0004, // body bytes
        0x0001_0004, // protocol 1, body bytes 4
        0x1d00_0000, // matching driver message ID
        0x0000_010a, // matching opcode
        0x0000_0000, // AIE2_STATUS_SUCCESS
    ];
    for (index, expected) in response.into_iter().enumerate() {
        assert_eq!(
            proc.bus.host_load32(0x030b_d000 + index as u32 * 4),
            expected,
            "I2X response word {index}",
        );
    }
    assert_eq!(proc.bus.data_load32(0x2720_03b4), 0);
    assert_eq!(proc.bus.data_load32(0x2720_03c4), 0);
    assert_eq!(proc.cpu.interrupt & 1, 0);
}

/// Collapse-to-bit3 characterization (2026-07-08): the ONLY verified external
/// stimulus is the per-column readiness bit3 at `[0xf9e0+col*0x60]` (gated by
/// `FUN_00008c68`'s `Bbci a9,3` at `0x8c8b`). This test runs boot BOTH ways to
/// a bounded n and contrasts them, establishing what bit3 alone unblocks -- with
/// no descriptor/target/done-flag machinery (all deleted as misread TLB/IPC
/// code; see the collapse-to-bit3 audit in the boot-wake finding).
///
/// Arm A: natural boot (no agent). Arm B: bit3 agent. Records for each the
/// waypoints reached (`task_dispatcher` 0xd7f0, the col-poll `0x8c88`,
/// `goalive_runfn` real entry `0x588c`, publisher 0x50e8, idle `waiti` 0x56e6),
/// the current-task path, and the final PC. The OBSERVED result (iter44): bit3 is
/// NOT the boot-progress gate -- with the 0x2450 fix both arms are byte-identical
/// (same final pc 0xb04252, same current-task path init 0x10f10 -> task 0x10dfc),
/// settling in the SAME coherent idle-loop (the first task polling an empty
/// external completion ring). bit3 changes nothing; go-alive/`waiti` never run
/// because progress there is host-triggered. See
/// docs/superpowers/findings/2026-07-10-boot-to-idle-reached.md.
/// Override the horizon with `XDNA_FW_MAX` (default 400k).
#[test]
fn m2c_bit3_advances_boot_past_natural_wall() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");

    const DISPATCH: u32 = 0xd7f0; // task_dispatcher
    const COLSVC: u32 = 0x8c88; // FUN_00008c68's bit3 poll (real exec entry, not 0x8c68 symbol)
    const GOALIVE: u32 = 0x588c; // goalive_runfn real exec Entry (not the 0x55f8 symbol)
    const PUBLISH: u32 = 0x50e8; // publish_chann_info (go-alive)
    const WAITI: u32 = 0x56e6; // idle waiti 0
    const CUR_TASK: u32 = 0x2278; // scheduler current-task pointer
    let waypoints = [DISPATCH, COLSVC, GOALIVE, PUBLISH, WAITI];
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400_000);

    // One boot arm; returns (first-hit map, current-task transitions, final pc,
    // reached_idle, bit3 rising-edges).
    let run_arm = |enable_agent: bool| {
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        if enable_agent {
            proc.enable_host_mailbox();
        }
        let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut cur_tasks: Vec<(u64, u32)> = Vec::new();
        let mut reached_idle = false;
        let mut n = 0u64;
        while n < max {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            for &t in &waypoints {
                if pc == t {
                    first.entry(t).or_insert(n);
                }
            }
            let cur = proc.bus.data_load32(CUR_TASK);
            if cur_tasks.last().map(|&(_, t)| t) != Some(cur) {
                cur_tasks.push((n, cur));
            }
            let step = proc.cpu.step(&mut proc.bus);
            proc.host_mailbox.tick(&mut proc.bus);
            match step {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    reached_idle = true;
                    break;
                }
                Step::Unknown { .. } => break,
            }
        }
        let edges = proc.host_mailbox.column_stats().0;
        let final_pc = proc.cpu.pc & 0x00ff_ffff;
        (first, cur_tasks, final_pc, reached_idle, edges, proc.symbols)
    };

    let (nat_first, nat_cur, nat_pc, nat_idle, _nat_edges, nat_syms) = run_arm(false);
    let (b3_first, b3_cur, b3_pc, b3_idle, b3_edges, b3_syms) = run_arm(true);

    eprintln!("=== collapse-to-bit3: natural vs bit3 boot (max n=400k) ===");
    let report = |label: &str,
                  first: &std::collections::BTreeMap<u32, u64>,
                  cur: &[(u64, u32)],
                  pc: u32,
                  idle: bool,
                  syms: &HashMap<u32, String>| {
        eprintln!("-- {label} --");
        eprintln!("  final pc={pc:#x} {}  reached_idle={idle}", nearest_symbol(syms, pc));
        for &w in &waypoints {
            eprintln!("    waypoint {w:#07x} {:<22} first@ {:?}", nearest_symbol(syms, w), first.get(&w));
        }
        eprintln!("    current-task path: {cur:x?}");
    };
    report("NATURAL (no agent)", &nat_first, &nat_cur, nat_pc, nat_idle, &nat_syms);
    report("BIT3 agent", &b3_first, &b3_cur, b3_pc, b3_idle, &b3_syms);
    eprintln!("  bit3 rising-edges asserted = {b3_edges}");

    assert!(b3_edges > 0, "bit3 agent never asserted a readiness bit");
    // KEY FINDING (collapse-to-bit3, updated 2026-07-09 for the EXCSAVE fix):
    // bit3 is NOT the gate for boot progress. Once EXCSAVE1-7 are modeled
    // (interp/mod.rs), the general-exception handler routes init's
    // cooperative-yield SYSCALL to the service path instead of mis-routing it
    // to the interrupt path, so BOTH arms now service the syscall. The 0xdad2
    // wall that followed was our fetch offset, not an opcode: the syscall-
    // dispatch block is a +0x100 section (SYSCALL_BLOCK overlay, iter19). With
    // it mapped both arms run the syscall jump table, the context-switch
    // chain (0x2630 -> IPC primitive 0xc48c, iter20 overlays), and -- with the
    // exception-restore section (0xe1fc) also mapped -- NO LONGER WALL: both
    // advance into the SAME steady loop inside the exception handler
    // (FUN_0000e098, ~0xe297) and spin there to the horizon. LONG before the
    // go-alive / column-power path (goalive_runfn 0x588c) would run. So the
    // finding stands and is stronger: bit3 (and the whole ColumnPowerAgent) is
    // entirely DOWNSTREAM of a loop the boot enters identically either way; the
    // two arms are byte-identical into it. (That loop is NOT yet proven idle --
    // see m2c_boot_advances_into_c_runtime; this test only checks bit3 does not
    // change the frontier.)
    assert_eq!(
        nat_pc, b3_pc,
        "the bit3 agent changed the boot frontier -- it is not supposed to gate progress \
             (natural arm ends at {nat_pc:#x}, bit3 at {b3_pc:#x})"
    );
    // iter25 (2026-07-10): with the go-alive publish path mapped, both arms boot
    // all the way to alive -- popping the go-alive job, running its run-fn,
    // publishing the mgmt channel, and resting at the same post-alive `waiti`
    // (see m2c_boot_advances_into_c_runtime and the boot-to-idle-reached finding).
    // The bit3-is-downstream finding is unchanged and stronger: both arms are
    // byte-identical (nat_pc == b3_pc above) even through full go-alive, so bit3
    // gates nothing. Here we only assert bit3 does not move the frontier and the
    // boot does not regress into the 0x9040 corruption.
    // The pre-fix 0x9040 cur-task corruption (the livelock's stack-overflow spill into SCHED)
    // is gone: neither arm's current-task ever leaves the legit init task 0x10f10.
    for (label, cur) in [("natural", &nat_cur), ("bit3", &b3_cur)] {
        assert!(
            !cur.iter().any(|&(_, t)| t == 0x9040),
            "{label} boot regressed to the 0x9040 stack-overflow corruption -- the syscall \
                 livelock is back (cur-task path: {cur:x?})"
        );
    }
    // frontier-ext (2026-07-11): with the go-alive TAIL mapped past the 0x5645 gate
    // (see m2c_boot_advances_into_c_runtime and
    // docs/superpowers/findings/2026-07-11-frontier-extension-past-goalive-tail.md),
    // both arms now advance identically PAST go-alive into the mapping-clean periodic
    // dispatch loop -- neither rests at the old 0x5645 waiti within 400k. The
    // bit3-is-downstream finding is unchanged and stronger: the arms are byte-identical
    // (nat_pc == b3_pc above) even through the go-alive tail. Assert idle-PARITY, not
    // idle itself: whatever the endpoint (loop now; a real idle once the loop question
    // is resolved), the two arms must agree -- a divergence means bit3 started gating.
    assert_eq!(
        nat_idle, b3_idle,
        "the bit3 agent changed idle behavior (nat_idle={nat_idle}, b3_idle={b3_idle}) \
             -- it is not supposed to gate the go-alive path",
    );
}

/// M2c Phase 2 iter 2: the PSP load map places segment B (the relocated
/// `.rodata`/`.data`/`.text`-tail) at physical `0x08b00000`, pre-loaded (not
/// copied at runtime). This asserts the load map places segment B's bytes at
/// the right physical addresses.
#[test]
fn m2c_load_map_places_segment_b() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    // Segment B: phys 0x08b041f0 (= file 0x312f0) is the callx8 target that
    // walled iter 1; it must now hold `entry a1,0x60` (36 c1 00 -> 0x4c00c136).
    assert_eq!(proc.bus.inst_load32(0x08b0_41f0), 0x4c00_c136, "segment B callx8 target not placed");
    // And memset's entry at phys 0x08b0e290 (= file 0x3b390): 36 41 00 -> 0x8c004136.
    assert_eq!(proc.bus.inst_load32(0x08b0_e290), 0x8c00_4136, "segment B memset entry not placed");
    // The firmware's task restore maps Segment-B literals through the DTLB as
    // virtual 0x08b00000 -> physical 0x0002d000. L32R must still see the
    // segment's first word at file 0x2d100, not segment A's bytes at 0x2d05c.
    let task_phys = SEG_B_FILE_START - LOW_VMA_FILE_OFFSET;
    assert_eq!(
        proc.bus.inst_load32_overlay(SEG_B_PHYS_BASE, task_phys),
        0x0010_0010,
        "segment B literal view not placed",
    );
}

/// Initialized low D-side data occupies VMA `[0xe740, 0xfefc)` and is stored
/// at file `VMA + 0x100`. The lower bound is the first initializer record; the
/// upper bound is where the startup explicitly begins zeroing BSS.
#[test]
fn m2c_load_map_places_initialized_low_data_section() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const BASE: u32 = 0x0000_e740;
    const END: u32 = 0x0000_fefc;
    for vaddr in BASE..END {
        assert_eq!(
            proc.bus.data_load8(vaddr),
            raw[(vaddr + LOW_VMA_FILE_OFFSET) as usize],
            "initialized D-side byte at {vaddr:#x}",
        );
    }

    // Independent live consumers pin the middle of the section: go-alive
    // metadata, scheduler state, and the device-MMIO base table.
    assert_eq!(proc.bus.data_load32(0xf2f8), 6);
    assert_eq!(proc.bus.data_load32(0xf2fc), 0xb);
    assert_eq!((0..6).map(|i| proc.bus.data_load8(0xf308 + i)).collect::<Vec<_>>(), [1, 1, 1, 1, 1, 0]);
    assert_eq!(proc.bus.data_load32(0xfac0), 0x2728_0000);
    assert_eq!(proc.bus.data_load32(0xfac4), 0x2728_03c0);
    assert_eq!(proc.bus.data_load32(0xfac8), 0x2728_04b0);

    const STRIDE: u32 = 0x1b8;
    for record in 0..6 {
        let base = BASE + record * STRIDE;
        for (count_off, dest_off, first_dest) in
            [(0xc4, 0xc8, 0x0008_5000), (0xd4, 0xd8, 0x0008_b000), (0xe4, 0xe8, 0x0009_1000)]
        {
            assert_eq!(
                proc.bus.data_load32(base + count_off),
                0x1000,
                "record {record} clear count at +{count_off:#x}",
            );
            assert_eq!(
                proc.bus.data_load32(base + dest_off),
                first_dest + record * 0x1000,
                "record {record} clear destination at +{dest_off:#x}",
            );
        }
    }
}

/// The GET_FIRMWARE_VERSION and GET_PROTOCOL_VERSION handlers read the
/// image's first 32-byte body record through D-side VMA zero.
#[test]
fn m2c_load_map_places_low_version_record() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    for vaddr in 0..0x20 {
        assert_eq!(
            proc.bus.data_load8(vaddr),
            raw[(vaddr + LOW_VMA_FILE_OFFSET) as usize],
            "version-record byte at {vaddr:#x}",
        );
    }
}

#[test]
fn m2c_uninitialized_dside_does_not_alias_the_instruction_image() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const FIELD: u32 = 0x0000_8778;
    let image_word = u32::from_le_bytes(
        raw[(FIELD + psp_load_map(raw.len() as u32)[0].rom_load_offset()) as usize..][..4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(image_word, 0x10e0_6010, "firmware tripwire changed");
    assert_eq!(proc.bus.inst_load32(FIELD), image_word, "I-side must still see the instruction image");
    assert_eq!(
        proc.bus.data_load32(FIELD),
        0,
        "uninitialized D-side state must not inherit instruction bytes",
    );
}

#[test]
fn m2c_load_map_places_startup_task_tables() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    for (slot, task) in [0x2000, 0x205c, 0x20b8, 0x2114, 0x2170, 0x21cc].into_iter().enumerate() {
        assert_eq!(proc.bus.data_load32(0xfb70 + slot as u32 * 4), task, "startup task pointer {slot}",);
        assert_eq!(proc.bus.data_load8(0xfb88 + slot as u32), slot as u8, "startup task slot {slot}",);
    }
}

#[test]
fn m2c_task_domain_physical_pages_remain_distinct() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let per_task_page = proc.bus.inst_load32_overlay(0x2540, 0x2540) & !0xfff;
    let fixed_task_page = proc.bus.inst_load32_overlay(0x28ac, 0x28ac) & !0xfff;
    assert_eq!((per_task_page, fixed_task_page), (0x21000, 0x13000), "firmware map geometry changed");

    proc.bus.data_store32(per_task_page, 0x2121_2121);
    proc.bus.data_store32(fixed_task_page, 0x1313_1313);
    assert_eq!(proc.bus.data_load32(per_task_page), 0x2121_2121);
    assert_eq!(proc.bus.data_load32(fixed_task_page), 0x1313_1313);
}

/// M2c Phase 1 coherence gate: with the load-offset, varway56, and the synth
/// PT in place, the real firmware boots from the reset entry past the MMU wall,
/// through the way-5 teardown and data-copy, to the C entry (`call0 0xe080`).
/// This test PINS the load-offset L: it passes iff L makes the continuation
/// coherent. If it fails, `last_pc` / `funcs_entered` localize the correct L.
#[test]
fn m2c_boot_reaches_c_entry() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Record whether the boot reaches the C-entry call (file 0xe080). The C
    // entry is reached via the continuation after the way-5 teardown, so
    // reaching it proves the whole code-region map is coherent.
    let reached = proc.reaches_pc(0xe080, 200_000);
    eprintln!("m2c boot: reached C entry (0xe080) = {reached}, last_pc = {:#x}", proc.cpu.pc);
    assert!(reached, "boot did not reach the C entry; last_pc={:#x} -- L or the map is wrong", proc.cpu.pc);
}
