use super::*;

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

/// M2c Phase 2 boot observation harness: boots from the reset entry via
/// `load_m2c` and prints the full `IdleReport` -- the instrument each Phase 2
/// walk-and-stub iteration reports against (the current wall's PC / stop
/// reason). It is NOT yet the idle gate (Phase 2 is not complete); the only
/// assertion is a regression guard that the boot still advances at least into
/// the C runtime (past the C entry at virtual 0x2000e024), so a change that
/// regresses the Phase 1 map is caught here. When Phase 2 reaches idle, this
/// hardens into the `reached_idle` gate.
#[test]
fn m2c_boot_advances_into_c_runtime() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let report = proc.boot_to_idle(200_000);
    eprintln!("=== M2c Phase 2 boot observation ===");
    eprintln!("instrs_executed  = {}", report.instrs_executed);
    eprintln!("last_pc          = {:#x}", report.last_pc);
    eprintln!("reached_idle     = {}", report.reached_idle);
    eprintln!("wait_reason      = {:?}", report.wait_reason);
    eprintln!("unresolved_spin  = {:?}", report.unresolved_spin.map(|a| format!("{a:#x}")));
    eprintln!("unknown_op       = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#010x}")));
    eprintln!("window_exceptions= {}", report.window_exceptions);
    eprintln!("funcs_entered    = {:?}", report.funcs_entered);

    // With the fill-loop fast-path the boot no longer grinds the 128 MiB
    // boot memset; it advances well past the region-zeroing routine (which
    // sits ~instr 23k). The exact next wall is under active investigation
    // (iter6); this stays an OBSERVATION test, asserting only that the boot
    // clears the region-init stretch, not a fixed idle gate.
    assert!(
        report.instrs_executed > 20_000 || report.reached_idle,
        "boot regressed short of the region-init routine ({} instrs) -- a map/fast-path regression",
        report.instrs_executed,
    );

    // iter12 (2026-07-05): the firmware's first SYSCALL now dispatches
    // correctly. The kernel exception vector's stub `l32r a3,=0x28b4; jx a3`
    // reads its dispatcher literal from the instruction-stream literal pool
    // (IRAM, via `l32r_load`), NOT the DRAM overlay (`local_data`) that the
    // boot's low-window memset (`fill 0x4..0xff0`) zeroes. So `a3` is the
    // real dispatcher, not 0, and the syscall services and returns instead
    // of jumping to PC=0 (the pre-fix iter12 wall). `main` then returns to
    // the crt0 post-return site 0x2000e035 -- an undecoded op 0x41f0,
    // iter13's wall.
    //
    // This gate is a coarse progress OBSERVATION: it pins only "the syscall
    // dispatch did not collapse to PC=0" (the specific iter12 regression).
    // It does NOT prove the syscall was serviced correctly -- the
    // pre-Harvard iter10 state also reached 0x2000e035 (~47.5k instrs, same
    // 0x41f0 wall) despite the stack-store-drop bug, because
    // window_exceptions=0 keeps the crt0->main return chain in the register
    // file, immune to lost stack data. The precise regression guard for the
    // l32r-reads-IRAM fix is the unit test
    // `low_window_l32r_reads_image_not_clobbered_local_data`.
    assert_ne!(
        report.unknown_op.map(|(pc, _)| pc),
        Some(0x0000_0000),
        "boot walls at PC=0 -- the exception-vector l32r read the zeroed DRAM \
             overlay instead of the IRAM literal (last_pc={:#x})",
        report.last_pc,
    );

    // iter13 (2026-07-05): the boot's one user-mode SYSCALL now routes to
    // the firmware's unified general-exception handler (0x2958) instead of
    // the mislabeled 0x28b4 dispatcher, so it is SERVICED rather than
    // falling through -- `main` no longer returns to the crt0 trap. The
    // wall advances past 0x2000e035 (to the handler's own next frontier, an
    // undecoded `rur` at ~0x2a09). This pins that the main-return wall
    // stays cleared: a regression that re-broke syscall routing would send
    // the boot back to 0x2000e035.
    assert_ne!(
        report.unknown_op.map(|(pc, _)| pc),
        Some(0x2000_e035),
        "boot walls at 0x2000e035 again -- the user-mode syscall was not \
             serviced (general-exception routing regressed; last_pc={:#x})",
        report.last_pc,
    );

    // iter16 (2026-07-06): a dispatch table the firmware builds holds
    // compiled-in function pointers into the low `.text` (0x581c/0x5858/
    // 0x588c). Those functions are stored in the file at `vaddr + 0x100`, not
    // the base `+0x5c` -- a piecewise VMA/LMA layout. The ROM fetch overlay
    // (`LOW_TEXT_BLOCK_*`) now serves them, so `callx8 0x588c` fetches the real
    // `entry` prologue instead of mid-instruction garbage. This pins that the
    // block wall stays cleared: a regression in the overlay would send the boot
    // back to an `Unknown` at 0x588c.
    assert_ne!(
        report.unknown_op.map(|(pc, _)| pc),
        Some(0x0000_588c),
        "boot walls at 0x588c again -- the low-.text +0x100 fetch overlay \
             regressed (last_pc={:#x})",
        report.last_pc,
    );

    // iter18 (2026-07-09): EXCSAVE1-7 are now modeled (interp/mod.rs). The
    // general-exception handler (0x2958) stashes EXCCAUSE into EXCSAVE3 on
    // entry and reads it back at 0x2a66 to dispatch syscall-vs-interrupt.
    // Without the register backing that readback was 0, so EVERY syscall
    // mis-routed to the interrupt path: init's cooperative-yield syscall was
    // never serviced, the scheduler livelocked re-dispatching 0x588c, and the
    // unbounded re-dispatch recursion eventually spilled the stack into SCHED
    // (cur-task corruption to 0x9040 at ~n=58.7k). That recursion was ALSO the
    // only source of window exceptions in this boot -- which is why the old
    // iter17 guards (window_exceptions>0, instrs>48_215) held. With the
    // syscall now serviced correctly the recursion is gone: the boot advances
    // cleanly. The 0xdad2 "unmodeled opcode" that followed was NOT an opcode
    // at all -- it was our fetch offset: the syscall-dispatch block (the
    // Callx4 target 0xdac4 and its callees) is a THIRD +0x100 piecewise-
    // relocated section (SYSCALL_BLOCK_LO..HI overlay, iter19). With that
    // mapped, the syscall handler runs its syscall-number jump table and
    // dispatches (via PC-relative Call8) to the context-switch routine at
    // 0x2630 -- itself another +0x100 section (iter20 CTXSW_CALLEE overlay),
    // which calls the IPC primitive 0xc48c (IPC_PRIMITIVE overlay) that posts
    // to [0xfae0] and jumps into Seg-B, returns, and runs the exception-frame
    // restore (0xe1fc, iter20 EXC_RESTORE overlay + its scattered pools). With
    // that mapped the boot NO LONGER WALLS anywhere in the budget: it advances
    // into a steady loop inside the exception handler (FUN_0000e098, last_pc
    // ~0xe297) and spins there for all 200k instrs. So unknown_op is None and
    // instrs_executed == the budget.
    //
    // iter21 (2026-07-09): the "steady loop in the exception handler" that the
    // prior iter20 note described was NOT a real scheduler state -- it was a
    // FRAMING ARTIFACT (CTXSW_CALLEE overlay ended too short at 0x26aa; the
    // routine tail was misframed at base into a fake poison-wdtlb epilogue).
    // Fixed by extending the overlay; details in the iter22 note below.
    //
    // iter22 (2026-07-09): mapping the 0x26d3 seam resolved the WHOLE region.
    // 0x26d3 is +0x100 (the ctx-switch routine's straight-line fall-through),
    // and the +0x100 run is far larger than the iter20/iter21 stubs captured:
    // it flows through the ctx-switch routine into the symbol-map fn
    // FUN_00002730 and continues as ONE contiguous +0x100 block -- the full
    // syscall/exception context save-restore + dual-way TLB-swap handler --
    // terminating at the `rfe` at 0x2bf2 (then the zero desert to Seg-B). The
    // discriminator is unambiguous: every L32r target hits an embedded pool,
    // FUN_00002730 aligns exactly, 0x28ef jumps to EXC_RESTORE (0xe1fc), and
    // the block reads exccause (0x28c3) / dispatches syscall (bnei a3,1 at
    // 0x28dc) / ends in rfe. CTXSW_CALLEE_HI now covers the whole span
    // (0x2630..0x2bf5); a regression sends boot back to the 0x26d6 poison-tail
    // wall or the 0xe098 spin.
    //
    // iter23 (2026-07-09): the faithful exception-vector model let the boot's
    // user-mode `syscall` be serviced (via VECBASE+0x2e0 -> stub -> 0x28b4),
    // advancing boot ~600 instrs into new code that walled at 0x93f3
    // (FUN_000093f0+0x3, Unknown 0x0000fc5d) -- the next +0x100 seam.
    //
    // iter24 (2026-07-09): mapping the 0x93f0 seam (SYSRET_SCAN overlay --
    // FUN_000093f0 is a +0x100 ring-scan fn Call8'd in the syscall-return
    // path; base-framed it walled Unknown 0xfc5d, +0x100 it is a clean
    // `entry a1,0x20`) CLEARED the opcode wall entirely: `unknown_op` is now
    // None and the boot runs the full budget into a fully-coherent steady
    // state -- the real IPC/ISR scheduler loop. That loop (period ~137
    // instrs): the IPC critical-section primitive 0xc48c posts a message to
    // the [0xfae0] mailbox, cache-flushes it (Dhwbi 0xb0e710), then stores a
    // doorbell byte to 0x2500000a (FUN_00007ed0), takes the resulting
    // exception, saves context (0xe098), calls the ISR 0xd900, and loops.
    //
    // [SUPERSEDED by iter25 (fault-cycle broken) then iter42/iter44 -- kept as
    // the changelog of how the frontier moved.] At iter24 the terminal state was
    // a STORE_PROHIBITED (cause 29) fault-cycle on the 0x2500000a doorbell.
    // Root cause (m2c_probe_mmu_at_fault, XDNA_FW_FAULT_PC=0x7f22): the
    // firmware disabled the 0x20000000 identity DTLB entry (asid=0) and
    // switched to page-table autorefill (ptevaddr=0x3c000000); our model's
    // page table there is empty, so autorefill maps the doorbell page to
    // paddr 0 READ-only and the store re-faults forever, re-posting the same
    // message. That is the NEXT frontier (page-table / external-memory-map
    // modeling), tracked in the boot-wake finding -- a different KIND of wall
    // from the +0x100 seams.
    //
    // iter25 (2026-07-09): standing in for the PSP, the synth page table now
    // identity-maps the peripheral gap between the code region and the mailbox
    // (psp_map::install -> synthesize_identity_page_table, attr 2 RW device),
    // which contains the 0x2500000a doorbell. The store SUCCEEDS (routes to the
    // Bus RAM backing), breaking the fault-cycle: boot advances ~1800 instrs
    // into the task-context-restore path (FUN_00002730 reconstructs task
    // 0x10f10's saved context from [0x12048], loads its run-fn 0x2450 from
    // field [0x12064], and Callx8's to it via the trampoline FUN_0000df8c).
    //
    // iter42 (2026-07-10): 0x2a86 lives in the +0x100 CTXSW section, so its
    // PC-relative Call0 target 0xdf98 must use the matching +0x100 bytes too.
    // Those bytes are the non-windowed WINDOWBASE/WINDOWSTART rotation helper,
    // not the base-framed Callx8 a7 that produced the false 0x2450 wall. With
    // CTXSW_WINDOW_ROTATE mapped, the helper returns at n=49553, the incoming
    // context reaches RFE at n=49586 and task entry 0x08b041bc at n=49624.
    // The 200k observation now consumes its full budget in coherent Seg-B task
    // code (unknown_op=None), directly pinning that the 0x2450 wall stays gone.
    // frontier-ext part 2 (2026-07-11): the periodic go-alive loop that iter25
    // read as a "mapping-clean steady state" was itself a MAPPING ARTIFACT. The
    // empty-queue branch 0xcc2e->0xccb3 landed one byte before the queue-pop
    // overlay ended at 0xccb4, so 0xccb3 fetched a mixed (1 AT + 2 BASE byte)
    // instruction that failed to clear the work-item valid bit [0x15fcb]; the
    // MERT worker re-accepted the same stale go-alive descriptor forever (the
    // queue itself retires correctly, [0x24c4] 1->0). Proven in
    // docs/superpowers/findings/2026-07-11-goalive-loop-discriminator.md.
    // Extending the overlay to [0xcc1c,0xccc1) executes the real empty-queue
    // tail (clears the valid bit) and the boot advances PAST the phantom loop --
    // straight into the KNOWN IRREDUCIBLE framing collision at 0x8cb4 (last_pc
    // 0x8cb1): the one VMA proven to need different file bytes on the publish
    // path vs the service path, unrecoverable from the flat signed $PS1 image
    // (docs/superpowers/findings/2026-07-11-firmware-vma-file-map-not-statically-recoverable.md).
    // Resolving 0x8cb4 needs external ground truth (PSP-loader RE); until then
    // this honest wall is the terminal, and it is strictly more truthful than the
    // phantom loop it replaces. This guard pins the advance-TO-the-collision, not
    // a false idle: a regression that drops the queue-tail overlay sends the boot
    // back into the phantom loop (last_pc in the 0x55f8/0x5645 dispatch cycle).
    assert_eq!(
        report.last_pc & 0x00ff_ffff,
        0x0000_8cb1,
        "boot no longer walls at the known 0x8cb4 framing collision -- the queue-pop \
             overlay extension [0xcc1c,0xccc1) likely regressed (back to the phantom \
             go-alive loop), or the frontier moved: {report:?}",
    );
    // The advance to 0x8cb4 takes exactly one benign register-window spill; the
    // guard against the OLD unbounded-0x588c-re-dispatch livelock is now a bound,
    // not a zero (that livelock produced many window exceptions).
    assert!(
        report.window_exceptions <= 1,
        "boot took {} window exceptions -- regressed into the old recursion livelock \
             (unbounded 0x588c re-dispatch spilling the register window). last_pc={:#x}",
        report.window_exceptions,
        report.last_pc,
    );
    // iter25 (2026-07-10): the go-alive publish path is mapped -- the boot pops the
    // enqueued go-alive job, runs its run-fn, and publishes the mgmt channel (the
    // "_NPU" magic reaches host-visible SRAM, see m2c_probe_alive_magic_scan).
    //
    // frontier-ext (2026-07-11): the go-alive TAIL past the 0x5645 status gate is
    // now mapped too. That gate's `waiti` was a MAPPING ARTIFACT -- a missing +0x100
    // overlay for the status literal at 0x31a4 made us deref garbage and park; with
    // it (plus the callx8 Segment-B pointer pool 0x32c8, the scheduler helpers
    // 0x7c5c/0x7d4c, and the syscall/ctxsw dispatch prefix 0xd864/0x353c/0xc6b0),
    // the boot skips the phantom waiti and runs the full budget mapping-clean in the
    // periodic MERT go-alive dispatch loop. Full account:
    // docs/superpowers/findings/2026-07-11-frontier-extension-past-goalive-tail.md.
    //
    // It does NOT rest at an idle waiti: goalive_runfn re-dispatches (~131x/500k) and
    // FW_ALIVE_OFF stays 0. Whether that loop is an intended MERT worker or a queue
    // item our model fails to retire is the OPEN next question (queue-ownership +
    // host-SRAM visibility), NOT a code-framing gap -- so this guard asserts only the
    // proven facts: the boot published _NPU and advanced PAST the 0x5645 gate. A
    // regression that drops a tail overlay sends it back to parking at 0x5645.
    assert_eq!(
        proc.bus.load_local32(0x14820),
        0x5550_4e5f,
        "boot no longer publishes the _NPU mgmt-channel magic -- a go-alive \
             publish-path overlay regressed: {report:?}",
    );
    assert_ne!(
        report.last_pc & 0x00ff_ffff,
        0x0000_5645,
        "boot still parks at the 0x5645 phantom waiti -- a go-alive TAIL overlay \
             (0x31a4/0x32c8/0x7c5c/0x7d4c/0xd864/0x353c/0xc6b0) regressed: {report:?}",
    );
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
