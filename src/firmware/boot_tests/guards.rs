use super::*;
use crate::firmware::mmio::Region;

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
}

impl PinnedMgmtChannel {
    fn new() -> Self {
        Self { x2i_tail: 0, i2x_head: 0, next_id: 0x1d00_0000 }
    }

    fn deliver(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) -> (u32, u32) {
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

        let report = proc.boot_to_idle_with_device(device, 200_000);
        assert!(report.reached_idle, "opcode {opcode:#x} did not return to idle: {report:?}");
        assert_eq!(report.unknown_op, None, "opcode {opcode:#x}");
        assert_eq!(report.unresolved_spin, None, "opcode {opcode:#x}");
        assert_eq!(
            proc.bus.host_load32(0x030e_c004),
            self.x2i_tail,
            "firmware did not consume opcode {opcode:#x}",
        );

        self.next_id = self.next_id.wrapping_add(1);
        (id, old_i2x_tail)
    }

    fn transact(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) -> Vec<u32> {
        let (id, old_i2x_tail) = self.deliver(proc, device, opcode, body);
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
        proc.bus.host_store32(0x030e_d004, self.i2x_head);
        proc.bus.host_store32(0x030e_d008, 0);
        body
    }

    fn post(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        opcode: u32,
        body: &[u32],
    ) {
        let (_, old_i2x_tail) = self.deliver(proc, device, opcode, body);
        assert_eq!(
            proc.bus.host_load32(0x030e_d000),
            old_i2x_tail,
            "posted opcode {opcode:#x} unexpectedly responded synchronously",
        );
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
            self.post(proc, device, 0x10c, &[address, 0, 0x2000]);
        }
    }

    fn create_context(
        &mut self,
        proc: &mut FirmwareProcessor,
        device: &mut crate::device::DeviceState,
        requested_col: u8,
    ) -> PinnedContextChannel {
        let response = self.transact(
            proc,
            device,
            0x02,
            &[
                1,                                            // AIE2
                u32::from_le_bytes([requested_col, 1, 0, 0]), // one Phoenix column
                1,                                            // one CQ pair, PASID 0
                0,
                0,
                0,
                2, // PRIORITY_HIGH
            ],
        );
        PinnedContextChannel::from_create_response(&response)
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
    let _context = channel.create_context(&mut proc, &mut device, requested_col);
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
    let mut context = management.create_context(&mut proc, engine.device_mut(), 1);

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

fn assert_configured_cu_executes_frozen_kernel_through_firmware_response(
    compiler: &str,
    xclbin_size: u64,
    pdi_size: usize,
) {
    const HEAP_BASE: u64 = 0x0400_0000;
    const HEAP_SIZE: usize = 0x0400_0000;
    const PDI_ADDR: u64 = HEAP_BASE;
    const CHAIN_ADDR: u64 = HEAP_BASE + 0x8000;
    const INST_ADDR: u64 = HEAP_BASE + 0x9000;
    const INPUT_A_ADDR: u64 = HEAP_BASE + 0xa000;
    const INPUT_B_ADDR: u64 = HEAP_BASE + 0xb000;
    const OUTPUT_ADDR: u64 = HEAP_BASE + 0xc000;
    const NPU1_DEV_MEM_BUF_SHIFT: u32 = 15;

    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let Some(mlir_aie) = std::env::var_os("MLIR_AIE_PATH") else {
        eprintln!("skip: MLIR_AIE_PATH is not set");
        return;
    };
    let fixture_dir =
        std::path::PathBuf::from(mlir_aie).join(format!("build/test/npu-xrt/add_one_using_dma/{compiler}"));
    let xclbin_path = fixture_dir.join("aie.xclbin");
    if !xclbin_path.exists() {
        eprintln!("skip: frozen {compiler} xclbin not built at {}", xclbin_path.display());
        return;
    }
    assert_eq!(std::fs::metadata(&xclbin_path).unwrap().len(), xclbin_size, "frozen {compiler} xclbin size");

    let xclbin = crate::parser::Xclbin::from_file(&xclbin_path).expect("parse frozen xclbin");
    let partition_section = xclbin
        .find_section(crate::parser::xclbin::SectionKind::AiePartition)
        .expect("AIE partition");
    let partition =
        crate::parser::AiePartition::parse(partition_section.data()).expect("parse AIE partition");
    assert_eq!(partition.start_columns(), [1, 2, 3, 4]);
    let pdi = partition.primary_pdi().expect("primary PDI").pdi_image.to_vec();
    assert_eq!(pdi.len(), pdi_size, "frozen {compiler} primary PDI size");

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

    let insts = std::fs::read(fixture_dir.join("insts.bin")).expect("read frozen instruction stream");
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
    let mut context = management.create_context(&mut proc, engine.device_mut(), 1);

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
    engine.host_memory_mut().write_bytes(PDI_ADDR, &pdi);

    // Pinned open-driver wire contract: NPU1 uses 32 KiB device-memory
    // address units; CONFIG_CU stores address bits 16:0 and function bits 24:17.
    let pdi_alignment = 1u64 << NPU1_DEV_MEM_BUF_SHIFT;
    assert_eq!(PDI_ADDR & (pdi_alignment - 1), 0, "PDI address alignment");
    let pdi_units = PDI_ADDR >> NPU1_DEV_MEM_BUF_SHIFT;
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
    assert_eq!(context.consume_response(&mut proc.bus, config_id, 0x11), [0], "CONFIG_CU status");

    let mut pdi_after = vec![0; pdi.len()];
    engine.host_memory().read_bytes(PDI_ADDR, &mut pdi_after);
    assert_eq!(pdi_after, pdi, "firmware changed the registered PDI bytes");

    let regmap = [
        3,
        0,
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
    let mut slot_words = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, regmap.len() as u32];
    slot_words.extend(regmap);
    let slot = slot_words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(slot.len(), 112, "pinned driver NON_ELF slot size");

    let input = (1u32..=64).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let host_memory = engine.host_memory_mut();
    host_memory.write_bytes(CHAIN_ADDR, &slot);
    host_memory.write_bytes(INST_ADDR, &insts);
    host_memory.write_bytes(INPUT_A_ADDR, &input);

    proc.bus.arm_probe();
    let (exec_id, x2i_tail, old_exec_i2x_tail) = context.post(
        &mut proc.bus,
        0x18,
        &[0, 0, CHAIN_ADDR as u32, (CHAIN_ADDR >> 32) as u32, slot.len() as u32, 1],
    );
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
    assert_eq!(context.consume_response(&mut proc.bus, exec_id, 0x18), [0, 0, 0], "CHAIN_EXEC_NPU response",);

    let output = (0..64)
        .map(|index| engine.host_memory().read_u32(OUTPUT_ADDR + index * 4))
        .collect::<Vec<_>>();
    assert_eq!(output, (2..=65).collect::<Vec<_>>(), "frozen kernel output");
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
    assert!(
        pdi_array_writes.iter().all(|access| Bus::decode_array_addr(access.addr).0 == 1),
        "PDI wrote outside assigned physical column 1: {pdi_array_writes:#x?}",
    );

    let device = engine.device();
    assert_eq!(device.tiles_with_code(), 1, "configured PDI program-memory footprint");
    assert_eq!(device.enabled_cores(), 1, "configured PDI core-enable footprint");
    let compute = device.tile(1, 2).expect("assigned compute tile");
    assert!(compute.program_memory().unwrap().iter().any(|&byte| byte != 0), "program memory remained empty");
    assert!(compute.data_memory().iter().any(|&byte| byte != 0), "data memory remained empty");
    assert_ne!(compute.core.control & 1, 0, "PDI did not configure Core_Control");
    for row in 0..device.rows() {
        assert!(device.tile(0, row).is_none(), "physical column 0 unexpectedly contains tile row {row}");
    }
}

#[test]
fn m2c_configured_cu_executes_frozen_chess_kernel_through_firmware_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response("chess", 9671, 3216);
}

#[test]
fn m2c_configured_cu_executes_frozen_peano_kernel_through_firmware_response() {
    assert_configured_cu_executes_frozen_kernel_through_firmware_response("peano", 9062, 2608);
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
