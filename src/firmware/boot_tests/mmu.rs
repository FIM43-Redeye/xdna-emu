use super::*;

/// iter23 (2026-07-09): capture VECBASE + PS at the first `syscall` so the
/// faithful exception-vector model knows where the CPU actually vectors and in
/// which mode. Steps to the syscall pc (XDNA_FW_STOP_PC, default 0x8b043e1) and
/// dumps vecbase/ps/epc1 + the vector bytes at candidate offsets. Self-skips
/// unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_exc_vector_state() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the exc-vector-state probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let stop_pc = std::env::var("XDNA_FW_STOP_PC")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x08b0_43e1);
    let mut vecbase_writes: Vec<(u64, u32)> = Vec::new();
    let mut last_vb = proc.cpu.vecbase;
    let mut n = 0u64;
    while n < 5_000_000 {
        if proc.cpu.pc == stop_pc {
            break;
        }
        if proc.cpu.vecbase != last_vb {
            vecbase_writes.push((n, proc.cpu.vecbase));
            last_vb = proc.cpu.vecbase;
        }
        match proc.cpu.step(&mut proc.bus) {
            crate::firmware::xtensa::interp::Step::Unknown { .. }
            | crate::firmware::xtensa::interp::Step::Wait(_) => break,
            _ => {}
        }
        n += 1;
    }
    eprintln!("=== M2c exc-vector state @ pc={:#x} (n={n}) ===", proc.cpu.pc);
    eprintln!("vecbase        = {:#x}", proc.cpu.vecbase);
    eprintln!("ps             = {:#010x}", proc.cpu.regs.ps);
    eprintln!(
        "  EXCM={} INTLEVEL={} WOE={} UM(bit5)={}",
        proc.cpu.regs.ps & 0x10 != 0,
        proc.cpu.regs.ps & 0xF,
        proc.cpu.regs.ps & (1 << 18) != 0,
        proc.cpu.regs.ps & 0x20 != 0
    );
    eprintln!("epc1           = {:#x}", proc.cpu.epc1);
    eprintln!("vecbase writes = {:x?}", vecbase_writes);
    // Dump candidate general-exception vector offsets (bytes, both framings).
    for off in [0x300u32, 0x340, 0x2c0, 0x280, 0x31c] {
        let v = proc.cpu.vecbase.wrapping_add(off);
        let base: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(v + k as u32, v + k as u32));
        let hex: String = base.iter().map(|x| format!("{x:02x}")).collect();
        eprintln!("  vecbase+{off:#05x}={v:#06x}: {hex}");
    }
}

/// M2c Phase 2 DIAGNOSTIC (iter13): is the exception dispatcher's entry
/// `l32r a15` value LOAD-BEARING for the "main returns" wall? The dispatcher at
/// runtime 0x28b4 loads a15 from a literal whose PC-relative target wraps to
/// 0xfffe3094 (stubbed to 0, provenance unresolved -- see
/// `exception-dispatch-pc-verdict.md`). This probe FORCES a15 to a chosen value
/// right after that l32r executes and reports where boot then walls, plus where
/// the dispatcher returns. If the wall (instr count / stop / return target) is
/// INVARIANT across forced a15 values, the 0xfffe3094 read is a red herring; if
/// it changes, a15 is load-bearing and worth deriving. Set XDNA_FW_FORCE_A15 to
/// a hex value to force; unset = control (a15 = the stub's 0). Ignored unless
/// XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_a15_loadbearing() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the a15 load-bearing probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let force: Option<u32> = std::env::var("XDNA_FW_FORCE_A15").ok().map(|s| {
        u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect("XDNA_FW_FORCE_A15 must be hex")
    });

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MAX: u64 = 200_000;
    const DISPATCHER_L32R_PC: u32 = 0x28b4; // entry FLIX bundle: l32r a15,<lit>
    const DISPATCHER_RETW_PC: u32 = 0x291c; // retw.n that ends the dispatcher
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    let mut dispatcher_hits = 0u64;
    let mut forced_events = 0u64;
    let mut retw_returns: Vec<(u64, u32)> = Vec::new(); // (n, return-target pc)
    let mut in_dispatcher = false;

    while n < MAX {
        let pc = proc.cpu.pc;
        if pc == DISPATCHER_L32R_PC {
            dispatcher_hits += 1;
            in_dispatcher = true;
        }
        let at_retw = pc == DISPATCHER_RETW_PC && in_dispatcher;

        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                if proc.cpu.pc == pc {
                    stop = format!("idle Wait({reason:?}) at pc={pc:#x}");
                    break;
                }
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown pc={upc:#x} word={word:#010x}");
                break;
            }
        }

        // Override a15 right after the dispatcher's entry l32r executed (pc has
        // advanced past 0x28b4; the l32r set a15=0). a15 is untouched between the
        // l32r and its consumer `bnez.n a15` at 0x28bf, so this override is seen.
        if pc == DISPATCHER_L32R_PC {
            if let Some(v) = force {
                proc.cpu.regs.write_ar(15, v);
                forced_events += 1;
            }
        }
        if at_retw {
            retw_returns.push((n, proc.cpu.pc));
            in_dispatcher = false;
        }

        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }

    eprintln!("=== M2c a15 load-bearing probe ===");
    match force {
        Some(v) => eprintln!("forced a15 = {v:#x} ({forced_events}x)"),
        None => eprintln!("control run (a15 = stub 0, not forced)"),
    }
    eprintln!("instrs executed = {n}");
    eprintln!("stop reason     = {stop}");
    eprintln!("dispatcher hits (pc=0x28b4) = {dispatcher_hits}");
    eprintln!("dispatcher retw returns     = {retw_returns:?}");
}

/// M2c Phase 2 DIAGNOSTIC: stop at the big memset's entry (phys 0x08b0e290)
/// and dump its windowed arguments (a2=dest, a3=fill, a4=count) plus a ring
/// buffer of the recent call8/callx8 history, to see how the boot reaches it
/// and with what count. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_memset_entry() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the memset-entry probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MEMSET_ENTRY: u32 = 0x08b0_e290;
    const MAX: u64 = 300_000;
    // Every memset entry: (instr_n, return_a0, dest, fill, count).
    let mut hits: Vec<(u64, u32, u32, u32, u32)> = Vec::new();
    let mut n = 0u64;
    let mut runaway: Option<(u64, u32, u32, u32, u32)> = None;
    while n < MAX {
        let pc = proc.cpu.pc;
        if pc == MEMSET_ENTRY {
            let rec = (
                n,
                proc.cpu.regs.read_ar(0),
                proc.cpu.regs.read_ar(2),
                proc.cpu.regs.read_ar(3),
                proc.cpu.regs.read_ar(4),
            );
            hits.push(rec);
            // A count over 1 MiB is the runaway; stop so we don't grind it.
            if rec.4 > 0x10_0000 {
                runaway = Some(rec);
                break;
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
    }

    eprintln!("=== M2c memset-entry probe (all invocations) ===");
    eprintln!("total memset entries seen = {}", hits.len());
    eprintln!("{:>8} {:>12} {:>12} {:>12} {:>12}", "instr", "a0(ret)", "dest", "fill", "count");
    for (i, a0, dest, fill, count) in &hits {
        eprintln!("{i:>8} {a0:>#12x} {dest:>#12x} {fill:>#12x} {count:>#12x}");
    }
    if let Some((i, a0, dest, fill, count)) = runaway {
        eprintln!(
                "--- RUNAWAY memset at instr {i}: dest={dest:#x} fill={fill:#x} count={count:#x} (ret a0={a0:#x}) ---"
            );
    } else {
        eprintln!("no runaway (>1MiB) memset within {MAX} instrs; last_pc={:#x}", proc.cpu.pc);
    }
}

/// M2c Phase 2 DIAGNOSTIC: trace the spin loop around instr 23089+ that
/// repeatedly calls memset(0xe740, 0x1b8, 0). Runs to just before the spin,
/// then single-steps ~60 instructions printing pc + decoded op + the
/// current-window a-registers, to reveal what the loop branches on and why
/// it never exits. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_spin_loop() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the spin-loop trace");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Run to just before the spin sets in.
    const WARMUP: u64 = 23_060;
    for _ in 0..WARMUP {
        proc.cpu.step(&mut proc.bus);
    }
    eprintln!("=== M2c spin-loop trace (from instr {WARMUP}) ===");
    for i in 0..70u64 {
        let pc = proc.cpu.pc;
        // Translate the virtual PC to physical so peek reads the real
        // instruction bytes (peek8 on a virtual code address in the System
        // region would otherwise return 0 -> a phantom Unknown{word:0}).
        let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                let bytes = [
                    proc.bus.peek8(phys),
                    proc.bus.peek8(phys.wrapping_add(1)),
                    proc.bus.peek8(phys.wrapping_add(2)),
                ];
                format!("{:?}", decode::decode(&bytes, pc).op)
            }
            Err(_) => "<fetch-fault>".to_string(),
        };
        // Dump the low 8 window registers alongside each instruction.
        let a: Vec<String> = (0..8).map(|r| format!("a{r}={:#x}", proc.cpu.regs.read_ar(r))).collect();
        eprintln!("{:>5} pc={:#x} {:<30} | {}", WARMUP + i, pc, disasm, a.join(" "));
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => {}
            other => {
                eprintln!("stop: {other:?}");
                break;
            }
        }
    }
}

/// M2c MMU-AT-FAULT probe: boot to the steady-state exception livelock, then
/// introspect why the load of the current-task pointer `0x2278` faults with
/// cause 28 (LOAD_PROHIBITED). Stops at the faulting `L32iN` (pc=0xe2a9,
/// a2=0x2278) in the steady loop and dumps the DTLB resolution (hit way/attr
/// or miss), PTEVADDR/DTLBCFG/RASID, the way-6 low-window identity entries,
/// and the full autorefill translate result -- pinning whether the fault is a
/// resident no-read entry or a page-walk to a no-read PTE. Ignored unless
/// XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_mmu_at_fault() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the mmu-at-fault probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    use crate::firmware::xtensa::mmu::attr_to_access;
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let max: u64 = std::env::var("XDNA_FW_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);
    // Default = the legacy 0x2278 load fault (pc 0xe2a9). Override to inspect a
    // different fault site: XDNA_FW_FAULT_PC=<hex> stop pc, XDNA_FW_FAULT_ADDR=<hex>
    // the DTLB vaddr to introspect (e.g. the 0x2500000a doorbell store at 0x7f22).
    let hexenv = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(d)
    };
    let stop_pc = hexenv("XDNA_FW_FAULT_PC", 0xe2a9);
    let fault_addr = hexenv("XDNA_FW_FAULT_ADDR", 0x2278);

    let mut n = 0u64;
    let mut found = false;
    while n < max {
        if proc.cpu.pc == stop_pc {
            let m = &proc.cpu.mmu;
            eprintln!("=== MMU at fault ({fault_addr:#x}) n={n} ===");
            eprintln!(
                "  ptevaddr={:#x} dtlbcfg={:#x} itlbcfg={:#x} rasid={:#x}",
                m.ptevaddr, m.dtlbcfg, m.itlbcfg, m.rasid
            );
            match m.lookup(fault_addr, true) {
                Ok(hit) => {
                    let e = m.dtlb[hit.wi][hit.ei];
                    eprintln!(
                        "  lookup: HIT way={} ei={} ring={} vaddr={:#x} paddr={:#x} attr={} access={:#x}",
                        hit.wi,
                        hit.ei,
                        hit.ring,
                        e.vaddr,
                        e.paddr,
                        e.attr,
                        attr_to_access(e.attr)
                    );
                }
                Err(c) => eprintln!("  lookup: MISS/err cause={c}"),
            }
            eprintln!(
                "  dtlbcfg={:#x} (way5 psz={}, way6 psz={})",
                proc.cpu.mmu.dtlbcfg,
                (proc.cpu.mmu.dtlbcfg >> 20) & 1,
                (proc.cpu.mmu.dtlbcfg >> 24) & 1
            );
            for wi in [4usize, 5, 6] {
                for ei in 0..4usize {
                    let e = proc.cpu.mmu.dtlb[wi][ei];
                    if e.asid == 0 && e.vaddr == 0 && e.paddr == 0 {
                        continue;
                    }
                    eprintln!(
                        "  dtlb[{wi}][{ei}] vaddr={:#x} paddr={:#x} asid={} attr={} var={}",
                        e.vaddr, e.paddr, e.asid, e.attr, e.variable
                    );
                }
            }
            let t = proc.cpu.mmu.translate(&mut proc.bus, fault_addr, 0, 0);
            eprintln!("  translate(load, with autorefill) = {t:?}");
            found = true;
            break;
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            other => {
                eprintln!("stopped early: {other:?} at n={n} pc={:#x}", proc.cpu.pc);
                break;
            }
        }
    }
    if !found {
        eprintln!("did not reach the 0xe2a9 fault within {max} instrs (n={n})");
    }
}

/// M2c iter24 DISCRIMINATOR (2026-07-09, Maya "discriminate first"): the boot's
/// terminal state is a STORE_PROHIBITED fault-cycle on the 0x2500000a doorbell,
/// because the firmware disabled the 0x20000000 identity DTLB entry and switched
/// to page-table autorefill at ptevaddr=0x3c000000 -- which our model reads as
/// empty. This probe answers WHOSE gap that is: over a full boot it (a) counts
/// every store whose EA lands in the page-table region [0x3c000000,0x40000000)
/// and (b) checks, non-perturbingly (mmu.lookup, no autorefill), whether the
/// doorbell vaddr ever becomes a RESIDENT WRITABLE mapping. Writes found or the
/// doorbell going writable => the firmware/our-load owns it (emulator gap to
/// fix). Neither ever => the PT at 0x3c000000 is populated by the PSP/an external
/// agent we don't model. Env: XDNA_FW_MAX (default 400_000),
/// XDNA_FW_DOORBELL (hex, default 0x2500000a). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_pt_writes() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the pt-writes discriminator");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    use crate::firmware::xtensa::mmu::{access_granted, attr_to_access};
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400_000);
    let doorbell = std::env::var("XDNA_FW_DOORBELL")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x2500_000a);
    const PT_LO: u32 = 0x3c00_0000;
    const PT_HI: u32 = 0x4000_0000;

    let mut pt_stores = 0u64;
    let mut first_pt: Option<(u64, u32, u32, u32)> = None; // n, pc, ea, val
    let mut first_writable: Option<(u64, u32, u8, u8)> = None; // n, pc, way, attr
                                                               // Transitions of "doorbell resident-writable": (n, pc, op, now_writable).
    let mut transitions: Vec<(u64, u32, String, bool)> = Vec::new();
    let mut prev_writable = false;
    // Watch dtlb[5][0].asid (the firmware's 0x20000000 128MB RWX install) go 1->0:
    // (n, pc-that-ran, op, old_asid, new_asid). Only invalidate_tlb can zero it.
    let mut way5_events: Vec<(u64, u32, String, u8, u8)> = Vec::new();
    let mut prev_asid5 = proc.cpu.mmu.dtlb[5][0].asid;
    let mut n = 0u64;
    while n < max {
        let pc = proc.cpu.pc;
        // (a) resident-writable check for the doorbell (non-perturbing).
        let cur_writable = match proc.cpu.mmu.lookup(doorbell, true) {
            Ok(hit) => {
                let e = proc.cpu.mmu.dtlb[hit.wi][hit.ei];
                let w = access_granted(attr_to_access(e.attr), 1);
                if w && first_writable.is_none() {
                    first_writable = Some((n, pc & 0x00ff_ffff, hit.wi as u8, e.attr));
                }
                w
            }
            Err(_) => false,
        };
        if cur_writable != prev_writable {
            // The instruction at the PREVIOUS step caused the transition; capture
            // the current pc's op as the site we land on / are about to run.
            let op = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fault>".to_string(),
            };
            transitions.push((n, pc & 0x00ff_ffff, op, cur_writable));
            prev_writable = cur_writable;
        }
        // (b) count stores into the page-table region.
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let d = decode::decode(&b, pc);
            let store = match d.op {
                Op::S32i { s, t, imm } | Op::S8i { s, t, imm } | Op::S16i { s, t, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t)))
                }
                Op::S32iN { s, t, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t)))
                }
                _ => None,
            };
            if let Some((ea, val)) = store {
                if (PT_LO..PT_HI).contains(&ea) {
                    pt_stores += 1;
                    if first_pt.is_none() {
                        first_pt = Some((n, pc & 0x00ff_ffff, ea, val));
                    }
                }
            }
        }
        // Decode the op about to run so we can attribute a way-5 asid change to it.
        let op_here = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                format!("{:?}", decode::decode(&b, pc).op)
            }
            Err(_) => "<fault>".to_string(),
        };
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            eprintln!("halted at n={n} pc={:#x}", proc.cpu.pc);
            break;
        }
        let asid5 = proc.cpu.mmu.dtlb[5][0].asid;
        if asid5 != prev_asid5 {
            way5_events.push((n, pc & 0x00ff_ffff, op_here, prev_asid5, asid5));
            prev_asid5 = asid5;
        }
        n += 1;
    }
    eprintln!("=== pt-writes discriminator (max {max}, doorbell {doorbell:#x}) ===");
    eprintln!("  ran n={n}");
    eprintln!("  dtlb[5][0].asid transitions (n, pc-that-ran, op, old->new):");
    for e in &way5_events {
        eprintln!("    n={:>8} pc={:#08x} {} : {}->{}", e.0, e.1, e.2, e.3, e.4);
    }
    eprintln!("  page-table-region stores [{PT_LO:#x},{PT_HI:#x}) = {pt_stores}");
    eprintln!("    first = {first_pt:x?}");
    eprintln!("  doorbell first resident-writable = {first_writable:x?}");
    eprintln!("  writable transitions (n, pc, op-landed-on, now_writable):");
    for t in &transitions {
        eprintln!("    n={:>8} pc={:#08x} now_writable={} {}", t.0, t.1, t.3, t.2);
    }
    // Also dump the final resident TLB entry that covers the doorbell, if any.
    match proc.cpu.mmu.lookup(doorbell, true) {
        Ok(hit) => {
            let e = proc.cpu.mmu.dtlb[hit.wi][hit.ei];
            eprintln!(
                    "  final lookup({doorbell:#x}): HIT way={} vaddr={:#x} paddr={:#x} asid={} attr={} access={:#x}",
                    hit.wi, e.vaddr, e.paddr, e.asid, e.attr, attr_to_access(e.attr)
                );
        }
        Err(c) => eprintln!("  final lookup({doorbell:#x}): MISS/err cause={c}"),
    }
}

/// M2c POISON-TLB-TRIGGER probe: find the exact instant the low-window no-access
/// TLB entry (a way 0-3 slot with attr>=12) first appears, and what installs it.
/// Logs every way-6 idtlb/wdtlb (pc + operands) and every `dtlb[6][0].asid`
/// liveness transition, and dumps each ctx-switch-tail instruction. Result (iter21):
/// it is NOT autorefill and the region entry is never invalidated -- the firmware
/// itself runs `wdtlb a5(=0xdeadbeef sentinel), a7(=0x2250)` at pc=0x26ac because a
/// task field `[0x121d0]` still holds its create-time uninitialized sentinel.
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_autorefill_trigger() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the autorefill-trigger probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    use crate::firmware::xtensa::mmu::MAX_TLB_WAY_SIZE;
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let max: u64 = std::env::var("XDNA_FW_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);

    let find_poison = |mmu: &crate::firmware::xtensa::mmu::Mmu| -> Option<(usize, usize)> {
        for wi in 0..4usize {
            for ei in 0..MAX_TLB_WAY_SIZE {
                let e = mmu.dtlb[wi][ei];
                if e.asid != 0 && e.attr >= 12 {
                    return Some((wi, ei));
                }
            }
        }
        None
    };

    let mut way6_ops: Vec<String> = Vec::new();
    let mut region_transitions: Vec<String> = Vec::new();
    let mut prev_region_live = proc.cpu.mmu.dtlb[6][0].asid != 0;
    let mut prev_pc = proc.cpu.pc;
    let mut n = 0u64;
    let mut found = false;
    while n < max {
        let pc = proc.cpu.pc;
        // Log way-6 D-side TLB management ops (peek before step; they only read ARs).
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let op = decode::decode(&b, pc).op;
            if (0x2630..0x2700).contains(&pc) {
                // Decode-agnostic dump of each ctx-switch instruction; at the
                // task-struct load (0x2647) also resolve mem[a6] to reconcile the
                // store-watch (0x12048) vs the observed read (0xdeadbeef).
                let a6 = proc.cpu.regs.read_ar(6);
                let mem_a6 = proc
                    .cpu
                    .mmu
                    .translate(&mut proc.bus, a6, 0, 0)
                    .map(|t| proc.bus.data_load32(t.paddr));
                eprintln!(
                    "  [tail] pc={pc:#x} op={op:?} a5={:#x} a6={a6:#x} mem[a6]={:?}",
                    proc.cpu.regs.read_ar(5),
                    mem_a6
                );
            }
            match op {
                Op::Wdtlb { s, .. } => {
                    let as_ = proc.cpu.regs.read_ar(s);
                    if (as_ & 0xf) == 6 {
                        way6_ops
                            .push(format!("n={n} pc={pc:#x} wdtlb as={as_:#x} (way6 vpn={:#x})", as_ & !0xf));
                    }
                }
                Op::Idtlb { s } => {
                    let as_ = proc.cpu.regs.read_ar(s);
                    if (as_ & 0xf) == 6 {
                        way6_ops
                            .push(format!("n={n} pc={pc:#x} idtlb as={as_:#x} (way6 vpn={:#x})", as_ & !0xf));
                    }
                }
                _ => {}
            }
        }

        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            other => {
                eprintln!("stopped early: {other:?} at n={n} pc={:#x}", proc.cpu.pc);
                break;
            }
        }

        let region_live = proc.cpu.mmu.dtlb[6][0].asid != 0;
        if region_live != prev_region_live {
            region_transitions.push(format!(
                "n={n} pc={pc:#x} dtlb[6][0].asid {} -> {}",
                if prev_region_live { "live" } else { "dead" },
                if region_live { "live" } else { "dead" }
            ));
            prev_region_live = region_live;
        }

        if let Some((wi, ei)) = find_poison(&proc.cpu.mmu) {
            let e = proc.cpu.mmu.dtlb[wi][ei];
            let r0 = proc.cpu.mmu.dtlb[6][0];
            eprintln!("=== poison autorefill entry first appeared ===");
            eprintln!("  n={n}  triggering instr (just stepped) = {pc:#x}  (prev instr = {prev_pc:#x})");
            eprintln!(
                "  dtlb[{wi}][{ei}] vaddr={:#x} paddr={:#x} asid={} attr={} var={}",
                e.vaddr, e.paddr, e.asid, e.attr, e.variable
            );
            eprintln!(
                "  region dtlb[6][0] vaddr={:#x} paddr={:#x} asid={} attr={}  (live={})",
                r0.vaddr,
                r0.paddr,
                r0.asid,
                r0.attr,
                r0.asid != 0
            );
            eprintln!("  dtlbcfg={:#x} ptevaddr={:#x}", proc.cpu.mmu.dtlbcfg, proc.cpu.mmu.ptevaddr);
            eprintln!("--- way-6 D-side TLB ops so far ({}) ---", way6_ops.len());
            for l in &way6_ops {
                eprintln!("  {l}");
            }
            eprintln!("--- dtlb[6][0] liveness transitions ({}) ---", region_transitions.len());
            for l in &region_transitions {
                eprintln!("  {l}");
            }
            found = true;
            break;
        }
        prev_pc = pc;
    }
    if !found {
        eprintln!("no poison autorefill entry appeared within {max} instrs (n={n})");
    }
}
