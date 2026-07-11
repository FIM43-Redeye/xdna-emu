use super::*;
use crate::firmware::mmio::Region;

/// Kernel-verification probe (2026-07-08, collapse-to-bit3): what does the
/// firmware ACTUALLY read in the per-column status region during a natural
/// (no-agent) boot?  Records every byte/half/word load whose address falls in
/// `[0xf900, 0xfc00)` (covers 8 columns * 0x60 stride), with the issuing PC,
/// first-seen n, hit count, and last value.  Establishes the verified kernel
/// params -- base, stride, column count, and the polled bit -- WITHOUT trusting
/// the deleted descriptor's `colmask=0xf`.  Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_colstatus_poll() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the col-status poll probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    // NATURAL boot by default; XDNA_FW_AGENT=1 enables the (old) host mailbox so
    // the poll is satisfied and boot advances -- reveals any columns polled only
    // AFTER the first wait phase passes.
    if std::env::var("XDNA_FW_AGENT").is_ok() {
        proc.enable_host_mailbox();
    }
    let stop: u64 = std::env::var("XDNA_FW_DUMP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);
    let lo: u32 = 0xf900;
    let hi: u32 = 0xfc00;

    // addr -> (first_n, count, last_val, width, issuing PCs set)
    let mut hits: std::collections::BTreeMap<u32, (u64, u64, u32, u8, std::collections::BTreeSet<u32>)> =
        std::collections::BTreeMap::new();
    let mut n = 0u64;
    while n < stop {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        // Decode the load about to run; compute its effective address.
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] =
                std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
            let d = decode::decode(&b, proc.cpu.pc);
            let load = match d.op {
                Op::L8ui { s, imm, .. } => Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 1u8)),
                Op::L16ui { s, imm, .. } | Op::L16si { s, imm, .. } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 2))
                }
                Op::L32i { s, imm, .. } | Op::L32iN { s, imm, .. } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 4))
                }
                _ => None,
            };
            if let Some((addr, width)) = load {
                if addr >= lo && addr < hi {
                    let val = match width {
                        1 => proc.bus.data_load8(addr) as u32,
                        2 => proc.bus.data_load32(addr) & 0xffff,
                        _ => proc.bus.data_load32(addr),
                    };
                    let e = hits.entry(addr).or_insert((n, 0, 0, width, std::collections::BTreeSet::new()));
                    e.1 += 1;
                    e.2 = val;
                    e.4.insert(pc);
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
    }

    eprintln!("=== col-status region reads [{lo:#x},{hi:#x}) over natural boot (n={n}) ===");
    eprintln!("(addr, first-n, count, width, last-val, issuing-PCs)");
    let mut prev: Option<u32> = None;
    for (addr, (fn_, cnt, val, w, pcs)) in &hits {
        let stride = prev.map(|p| addr - p).unwrap_or(0);
        let pcstr: Vec<String> = pcs.iter().map(|p| format!("{p:#x}")).collect();
        eprintln!(
                "  {addr:#07x} (+{stride:#x} from prev)  firstN={fn_:>6} count={cnt:>5} w={w} last={val:#x}  pcs=[{}]",
                pcstr.join(",")
            );
        prev = Some(*addr);
    }
    eprintln!("--- distinct addresses: {} ---", hits.len());
}

/// M2c Phase 2 boot-walk DIAGNOSTIC (not a correctness gate): arm the Bus
/// stub-access probe and boot, so every Array/Mailbox/System access the
/// firmware issues is captured with the PC that issued it. The suspected
/// wrong-path source is a peripheral read that returns the stub 0 which the
/// firmware then branches on. Prints a per-site summary (deduped by PC) in
/// access order, plus the raw tail near the boot wall. Ignored unless
/// XDNA_FW_PROBE is set, so the full suite stays fast.
#[test]
fn m2c_probe_peripheral_reads() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the peripheral-read characterization");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    proc.bus.arm_probe();

    // Step the boot manually so we can stamp each access with the issuing
    // PC and record the instruction index at which the run stops. Budget
    // covers past the memset wall (~23105 instrs).
    const MAX: u64 = 40_000;
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    while n < MAX {
        let pc = proc.cpu.pc;
        proc.bus.set_probe_pc(pc);
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                if proc.cpu.pc == pc {
                    stop = format!("idle Wait({reason:?}) at pc={pc:#x}");
                    break;
                }
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#010x}");
                break;
            }
        }
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }
    let log = proc.bus.take_probe();

    eprintln!("=== M2c peripheral-read characterization ===");
    eprintln!("instrs executed = {n}");
    eprintln!("stop reason     = {stop}");
    eprintln!("total stub accesses = {}", log.len());

    // Per-site summary, deduped by (pc, is_write), in first-seen order. For
    // each site: region, the distinct addresses and values seen, access
    // count, and the seq of first occurrence.
    use std::collections::BTreeMap;
    let mut sites: BTreeMap<(u32, bool), (Region, Vec<u32>, Vec<u32>, u64, u64)> = BTreeMap::new();
    let mut order: Vec<(u32, bool)> = Vec::new();
    for a in &log {
        let key = (a.pc, a.is_write);
        let e = sites.entry(key).or_insert_with(|| {
            order.push(key);
            (a.region, Vec::new(), Vec::new(), a.seq, 0)
        });
        if !e.1.contains(&a.addr) {
            e.1.push(a.addr);
        }
        if !e.2.contains(&a.value) {
            e.2.push(a.value);
        }
        e.4 += 1;
    }
    order.sort_by_key(|k| sites[k].3);
    eprintln!("distinct sites = {}", order.len());
    eprintln!(
        "{:>6} {:<5} {:<7} {:<3} {:<28} {:<28} {}",
        "seq", "rd/wr", "region", "cnt", "addrs", "values", "pc"
    );
    for key in &order {
        let (region, addrs, values, first_seq, count) = &sites[key];
        let addr_s = addrs.iter().take(4).map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(",");
        let val_s = values.iter().take(4).map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(",");
        eprintln!(
            "{:>6} {:<5} {:<7} {:<3} {:<28} {:<28} {:#x}",
            first_seq,
            if key.1 { "wr" } else { "rd" },
            format!("{region:?}"),
            count,
            addr_s,
            val_s,
            key.0,
        );
    }

    // Raw tail: the last 40 accesses, to see what immediately precedes the wall.
    eprintln!("--- last 40 raw accesses (seq: pc region rd/wr addr=value) ---");
    for a in log.iter().rev().take(40).rev() {
        eprintln!(
            "{:>6}: pc={:#x} {:?} {} {:#x}={:#x} w{}",
            a.seq,
            a.pc,
            a.region,
            if a.is_write { "wr" } else { "rd" },
            a.addr,
            a.value,
            a.width,
        );
    }
}

/// M2c iter18 Phase 0 DIAGNOSTIC: what interrupt does the firmware actually
/// ARM? The (C) done-flag mechanism delivers a real async event
/// (mailbox doorbell -> level-1 interrupt); to inject the RIGHT interrupt we
/// need the INTENABLE bit(s) the firmware sets during boot. `wsr.intenable`
/// (SR 0xE4) is internal to the CPU -- invisible to the peripheral probe --
/// but `Cpu::intenable` is a public field, so we watch it (and `interrupt`,
/// the pending bits) for changes after each step and log the PC + symbol
/// that caused each. Answers the "INTENABLE bits NOT yet observed" open item
/// in the Phase-0 findings. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_intenable_watch() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the intenable watch");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MAX: u64 = 1_000_000;
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    let mut prev_ie = proc.cpu.intenable;
    let mut prev_ip = proc.cpu.interrupt;
    let mut prev_il = proc.cpu.regs.intlevel();
    // Each transition: (n, causing-pc, symbol, which, old, new).
    let mut changes: Vec<(u64, u32, String, &'static str, u32, u32)> = Vec::new();
    // First instr at which INTLEVEL returns to 0 AFTER intenable is armed --
    // the only window a level-1 doorbell could actually be delivered. If this
    // stays None past the wall, a level-1 IRQ can never set the done-flag
    // (the dispatcher's rsil-2 critical section masks it), which argues the
    // completion is a DMA/DRAM write (shape ii), not a CPU handler (shape i).
    let mut armed_at: Option<u64> = None;
    let mut first_level0_after_arm: Option<(u64, u32)> = None;
    while n < MAX {
        let pc = proc.cpu.pc;
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) at pc={pc:#x} (idle)");
                break;
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                break;
            }
        }
        if proc.cpu.intenable != prev_ie {
            if prev_ie == 0 && proc.cpu.intenable != 0 {
                armed_at = Some(n);
            }
            changes.push((
                n,
                pc,
                nearest_symbol(&proc.symbols, pc),
                "INTENABLE",
                prev_ie,
                proc.cpu.intenable,
            ));
            prev_ie = proc.cpu.intenable;
        }
        if proc.cpu.interrupt != prev_ip {
            changes.push((
                n,
                pc,
                nearest_symbol(&proc.symbols, pc),
                "INTERRUPT",
                prev_ip,
                proc.cpu.interrupt,
            ));
            prev_ip = proc.cpu.interrupt;
        }
        let il = proc.cpu.regs.intlevel();
        if il != prev_il {
            changes.push((n, pc, nearest_symbol(&proc.symbols, pc), "INTLEVEL", prev_il, il));
            prev_il = il;
        }
        if armed_at.is_some() && first_level0_after_arm.is_none() && il == 0 {
            first_level0_after_arm = Some((n, pc));
        }
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }
    eprintln!("=== M2c intenable/interrupt/intlevel watch ===");
    eprintln!("instrs executed = {n}");
    eprintln!("stop reason     = {stop}");
    eprintln!(
        "final INTENABLE = {:#010x}  INTERRUPT = {:#010x}  INTLEVEL = {}",
        proc.cpu.intenable,
        proc.cpu.interrupt,
        proc.cpu.regs.intlevel()
    );
    eprintln!("armed_at instr  = {armed_at:?}");
    eprintln!(
        "first INTLEVEL==0 after arm = {first_level0_after_arm:x?} (level-1 doorbell deliverability window)"
    );
    eprintln!("--- {} SR transition(s) ---", changes.len());
    for (i, pc, sym, which, old, new) in &changes {
        eprintln!("{i:>7} pc={pc:#08x} {sym:<24} {which} {old:#010x} -> {new:#010x}");
    }
}

/// EXTERNAL-REQUEST OBSERVATION (2026-07-08): the causality-respecting first
/// step of the external-agent principle. Boots naturally (array attached) and
/// logs every firmware STORE whose effective address lands in an external
/// aperture -- the per-column HW/SMU pages (`0x2727xxxx`, whole `0x27xxxxxx`
/// peripheral/SMN/mailbox band) and device SRAM (`0x03xxxxxx`). It POKES
/// NOTHING: it only observes whether the firmware ever makes an external
/// request (an ack to `0x2727n114`, a mailbox/SMU write, an alive-struct
/// store) during reachable boot. If it writes nothing external, the completion
/// is not externally-requested in reachable boot -> the divergence is upstream
/// (the request-generating code never runs) and the agent cannot help until
/// that is fixed. If it does, those sites ARE the contract to respond to.
/// Env: XDNA_FW_MAX (budget, default 1_500_000). Ignored unless XDNA_FW_PROBE.
#[test]
fn m2c_probe_external_requests() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the external-request observation");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    proc.bus.attach_device(crate::device::DeviceState::new_npu1());
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);

    // External apertures: the 0x27xxxxxx peripheral/SMN/mailbox/per-column band
    // and device SRAM 0x03xxxxxx. Everything else is firmware-internal RAM.
    let is_external =
        |a: u32| (0x2700_0000..0x2800_0000).contains(&a) || (0x0300_0000..0x0320_0000).contains(&a);

    let syms = load_symbols();
    let mut n = 0u64;
    let mut stop = "budget reached";
    // (pc, addr) -> (count, first_n, last_value)
    let mut sites: std::collections::BTreeMap<(u32, u32), (u64, u64, u32)> =
        std::collections::BTreeMap::new();
    while n < max {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let d = decode::decode(&b, pc);
            let sv = match d.op {
                decode::Op::S32i { t, s, imm }
                | decode::Op::S32iN { t, s, imm }
                | decode::Op::S16i { t, s, imm }
                | decode::Op::S8i { t, s, imm }
                | decode::Op::S32ri { t, s, imm }
                | decode::Op::S32c1i { t, s, imm }
                | decode::Op::S32e { t, s, imm } => {
                    Some((proc.cpu.regs.read_ar(t), proc.cpu.regs.read_ar(s).wrapping_add(imm)))
                }
                _ => None,
            };
            if let Some((val, addr)) = sv {
                if is_external(addr) {
                    let e = sites.entry((pc & 0x00ff_ffff, addr)).or_insert((0, n, val));
                    e.0 += 1;
                    e.2 = val;
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => {
                stop = "waiti";
                break;
            }
            Step::Unknown { .. } => {
                stop = "unknown op";
                break;
            }
        }
    }
    eprintln!("=== external-request observation (array attached) ===");
    eprintln!("instrs = {n}, stop = {stop}");
    eprintln!("distinct external store sites = {}", sites.len());
    eprintln!("--- (pc, addr) <- last_val  count  first_n  sym ---");
    for ((pc, addr), (count, first_n, val)) in &sites {
        eprintln!(
            "  pc={pc:#08x}  [{addr:#010x}] <- {val:#010x}  x{count}  first@{first_n}  {}",
            nearest_symbol(&syms, *pc)
        );
    }
}

/// EXTERNAL CONVERSATION TRACE (2026-07-08): the write->read request/response
/// pairing the external-agent model must reproduce. Boots naturally (array
/// attached, pokes NOTHING) and logs, in temporal order, every firmware load
/// AND store whose effective address lands in the external band
/// (`0x27xxxxxx` peripheral/mailbox/per-column, `0x03xxxxxx` device SRAM).
/// The `m2c_probe_external_requests` sibling captures WRITES only, aggregated;
/// this one keeps loads too and in sequence, so the handshake protocol is
/// visible: request write -> status poll read -> ack write -> re-poll, etc.
/// Window: [XDNA_FW_CONV_START (default 38000), +XDNA_FW_CONV_LEN (default
/// 16000)]. Caps printed lines at XDNA_FW_CONV_CAP (default 500) but always
/// prints the full per-(pc,addr,dir) summary. Ignored unless XDNA_FW_PROBE.
#[test]
fn m2c_probe_external_conversation() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the external-conversation trace");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    if std::env::var("XDNA_FW_CONV_NOATTACH").is_err() {
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
    }
    let start: u64 = std::env::var("XDNA_FW_CONV_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(38_000);
    let len: u64 = std::env::var("XDNA_FW_CONV_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16_000);
    let cap: usize = std::env::var("XDNA_FW_CONV_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let end = start + len;

    let is_external =
        |a: u32| (0x2700_0000..0x2800_0000).contains(&a) || (0x0300_0000..0x0320_0000).contains(&a);

    let syms = load_symbols();
    let mut n = 0u64;
    let mut stop = "budget reached";
    let mut printed = 0usize;
    // (pc, addr, is_write) -> (count, first_n, last_value)
    let mut sites: std::collections::BTreeMap<(u32, u32, bool), (u64, u64, u32)> =
        std::collections::BTreeMap::new();
    // Decoupled PC-visit counters: 0x8c88 is FUN_00008c68's poll L8ui -- in
    // steady state its base is the INTERNAL RAM struct 0xf9e0+col*0x60 (bit3),
    // NOT the external 0x2727n114 page (that base only appears in a rarer
    // iteration); 0xc964 is the sched_ready_popcount rest loop.
    let mut hit_8c88 = 0u64;
    let mut hit_c964 = 0u64;
    eprintln!("=== external conversation (temporal, window [{start},{end})) ===");
    while n < end {
        let pc = proc.cpu.pc;
        match pc & 0x00ff_ffff {
            0x8c88 => hit_8c88 += 1,
            0xc964 => hit_c964 += 1,
            _ => {}
        }
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let d = decode::decode(&b, pc);
            // (is_write, base_reg, imm, store_val_reg)
            let acc: Option<(bool, u8, u32, Option<u8>)> = match d.op {
                decode::Op::L32i { s, imm, .. }
                | decode::Op::L32iN { s, imm, .. }
                | decode::Op::L8ui { s, imm, .. }
                | decode::Op::L16ui { s, imm, .. }
                | decode::Op::L16si { s, imm, .. } => Some((false, s, imm, None)),
                decode::Op::S32i { t, s, imm }
                | decode::Op::S32iN { t, s, imm }
                | decode::Op::S16i { t, s, imm }
                | decode::Op::S8i { t, s, imm } => Some((true, s, imm, Some(t))),
                _ => None,
            };
            if let Some((is_write, base, imm, vreg)) = acc {
                let addr = proc.cpu.regs.read_ar(base).wrapping_add(imm);
                if is_external(addr) && n >= start {
                    let val = match vreg {
                        Some(t) => proc.cpu.regs.read_ar(t),
                        None => proc.cpu.data_read32(&mut proc.bus, addr).unwrap_or(0),
                    };
                    let e = sites.entry((pc & 0x00ff_ffff, addr, is_write)).or_insert((0, n, val));
                    e.0 += 1;
                    e.2 = val;
                    if printed < cap {
                        eprintln!(
                            "  n={n:>7} pc={:#08x} {} [{addr:#010x}] {val:#010x}  {}",
                            pc & 0x00ff_ffff,
                            if is_write { "W" } else { "R" },
                            nearest_symbol(&syms, pc & 0x00ff_ffff)
                        );
                        printed += 1;
                    }
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => {
                stop = "waiti";
                break;
            }
            Step::Unknown { .. } => {
                stop = "unknown op";
                break;
            }
        }
    }
    eprintln!(
        "--- stop={stop}, instrs={n}, final_pc={:#x} {}, printed={printed}/{} distinct ---",
        proc.cpu.pc,
        nearest_symbol(&syms, proc.cpu.pc & 0x00ff_ffff),
        sites.len()
    );
    eprintln!("--- pc-visits: 0x8c88(0x2727-poll)={hit_8c88}  0xc964(sched_ready)={hit_c964} ---");
    eprintln!("--- summary: (pc, addr, dir) <- last_val  count  first_n  sym ---");
    for ((pc, addr, is_write), (count, first_n, val)) in &sites {
        eprintln!(
            "  pc={pc:#08x} {} [{addr:#010x}] {val:#010x}  x{count}  first@{first_n}  {}",
            if *is_write { "W" } else { "R" },
            nearest_symbol(&syms, *pc)
        );
    }
}

/// INTLEVEL SEAM (2026-07-08): the completion the boot waits on is a masked
/// interrupt (the external side is one-directional programming; both live
/// gates -- `[0xf9e0+col*0x60]` bit3 and `[task+0x30]` -- are internal flags
/// an ISR would set). This decides determination (A) vs (B): does INTLEVEL
/// ever drop low enough for an interrupt to deliver? Boots naturally (array
/// attached, pokes nothing) and reports, over the whole boot and over the
/// post-wall tail (n >= XDNA_FW_WALL_N, default 48000): a histogram of
/// PS.INTLEVEL, the min post-wall INTLEVEL, first/last n at which INTLEVEL==0,
/// the distinct INTENABLE values seen (which interrupts the fw enabled), the
/// final INTERRUPT (pending) word, and how many instructions satisfy
/// `interrupt_deliverable()` (our interp models level-1 delivery only: it
/// requires INTLEVEL==0). If INTLEVEL never reaches 0 post-wall, a level-1
/// completion IRQ can never land here -> either the fw should lower it and we
/// diverge (A), or the real completion is a high-level (>2) interrupt this
/// interp does not model (a modeling gap on the B side). XDNA_FW_MAX budget
/// (default 1_500_000). Ignored unless XDNA_FW_PROBE.
#[test]
fn m2c_probe_intlevel_seam() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the intlevel-seam probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    proc.bus.attach_device(crate::device::DeviceState::new_npu1());
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let wall_n: u64 = std::env::var("XDNA_FW_WALL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48_000);

    let syms = load_symbols();
    let mut n = 0u64;
    let mut stop = "budget reached";
    let mut hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut post_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut intenables: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut first_il0: Option<(u64, u32)> = None;
    let mut last_il0: Option<(u64, u32)> = None;
    let mut il0_count = 0u64;
    let mut deliverable = 0u64;
    // INTLEVEL transitions: (n, pc, old, new, disasm-of-the-instr-that-changed-it).
    // The instruction that CHANGED INTLEVEL is the one just stepped, so record
    // the pc/disasm from BEFORE the step and pair it with the post-step level.
    let mut transitions: Vec<(u64, u32, u32, u32, String)> = Vec::new();
    let mut prev_il = proc.cpu.regs.intlevel();
    while n < max {
        let il = proc.cpu.regs.intlevel();
        *hist.entry(il).or_insert(0) += 1;
        if n >= wall_n {
            *post_hist.entry(il).or_insert(0) += 1;
        }
        intenables.insert(proc.cpu.intenable);
        if il == 0 {
            il0_count += 1;
            let here = (n, proc.cpu.pc & 0x00ff_ffff);
            first_il0.get_or_insert(here);
            last_il0 = Some(here);
        }
        if proc.cpu.interrupt_deliverable() {
            deliverable += 1;
        }
        let pc = proc.cpu.pc;
        let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                format!("{:?}", decode::decode(&b, pc).op)
            }
            Err(_) => "<fault>".to_string(),
        };
        let r = proc.cpu.step(&mut proc.bus);
        let new_il = proc.cpu.regs.intlevel();
        if new_il != prev_il {
            transitions.push((n, pc & 0x00ff_ffff, prev_il, new_il, disasm));
            prev_il = new_il;
        }
        match r {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => {
                stop = "waiti";
                break;
            }
            Step::Unknown { .. } => {
                stop = "unknown op";
                break;
            }
        }
    }
    eprintln!("=== intlevel seam (natural boot, array attached) ===");
    eprintln!(
        "instrs={n}, stop={stop}, final_pc={:#x} {}",
        proc.cpu.pc,
        nearest_symbol(&syms, proc.cpu.pc & 0x00ff_ffff)
    );
    eprintln!("--- INTLEVEL histogram (whole boot) ---");
    for (il, c) in &hist {
        eprintln!("  intlevel={il}  x{c}");
    }
    eprintln!("--- INTLEVEL histogram (post-wall, n>={wall_n}) ---");
    for (il, c) in &post_hist {
        eprintln!("  intlevel={il}  x{c}");
    }
    let min_post = post_hist.keys().next().copied();
    eprintln!("min post-wall INTLEVEL = {min_post:?}");
    eprintln!("INTLEVEL==0: count={il0_count}  first={first_il0:?}  last={last_il0:?}");
    eprintln!("interrupt_deliverable() true count = {deliverable}");
    eprintln!(
        "final INTENABLE = {:#010x}  final INTERRUPT(pending) = {:#010x}",
        proc.cpu.intenable, proc.cpu.interrupt
    );
    eprintln!("distinct INTENABLE values seen ({}):", intenables.len());
    for e in &intenables {
        eprintln!("  {e:#010x}");
    }
    eprintln!("--- INTLEVEL transitions ({} total; first 40 + last 20) ---", transitions.len());
    let show: Vec<&(u64, u32, u32, u32, String)> = if transitions.len() <= 60 {
        transitions.iter().collect()
    } else {
        transitions
            .iter()
            .take(40)
            .chain(transitions.iter().skip(transitions.len() - 20))
            .collect()
    };
    for (tn, pc, old, new, dis) in show {
        eprintln!("  n={tn:>8} pc={pc:#08x} {old}->{new}  {:<28} {}", dis, nearest_symbol(&syms, *pc));
    }
    // The pin point: the last transition INTO 2 that is never followed by a drop below 2.
    if let Some((tn, pc, old, new, dis)) =
        transitions.iter().rev().find(|(_, _, _, new, _)| *new <= 2 && *new == 2)
    {
        eprintln!(
            "last transition to INTLEVEL 2: n={tn} pc={pc:#08x} {old}->{new} {dis} {}",
            nearest_symbol(&syms, *pc)
        );
    }
}

/// (A) ISR OBSERVATION (2026-07-09, Maya-approved single controlled delivery).
/// The boot livelock never delivers the armed line-0 IRQ (INTLEVEL pinned at
/// 2), so the ISR's callback-dispatched tail (`Callx4 [lit]` -> trampoline ->
/// `Callx8 [obj+168]` -> ?) can't be read statically (FLIX-heavy, runtime
/// pointers) nor traced under natural boot. This warms to steady state, FORCES
/// ONE faithful delivery (drop PS.INTLEVEL->0 and clear PS.EXCM, set INTERRUPT
/// |= INTENABLE) and TRACES the ISR path through the interp -- the FLIX ground
/// truth (its `step()` executes `Op::Flix1` bundles natively). It records every
/// PC (with nearest symbol), MMIO reads/writes (>= 0x2000_0000 -- the TRUE
/// interrupt-source register the ISR polls), each Call/Callx target (resolving
/// the runtime callback chain), reaching `post_event` (0xcf68) / `wake` (0xd84c)
/// / `deliver_pending_events` (0xcadc), changes to the global event accumulator
/// `[0x22bc]`, and where `rfe` returns. DIAGNOSTIC observation of the ISR
/// mechanism -- NOT a livelock break (no array modelled, one hand-forced
/// delivery). Env: XDNA_FW_ISR_WARMUP (default 300000), XDNA_FW_ISR_STEPS
/// (default 20000), XDNA_FW_ISR_SRC=addr:val (hex, seed an interrupt-source reg
/// before delivery, e.g. 0x272003b8:0x1000). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_isr_observe() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the ISR-observation probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let warmup = env_u64("XDNA_FW_ISR_WARMUP", 300_000);
    let steps = env_u64("XDNA_FW_ISR_STEPS", 20_000);
    // Optional interrupt-source seed: XDNA_FW_ISR_SRC=addr:val (hex).
    let src: Option<(u32, u32)> = std::env::var("XDNA_FW_ISR_SRC").ok().and_then(|s| {
        let (a, v) = s.split_once(':')?;
        Some((
            u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
            u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()?,
        ))
    });
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    for _ in 0..warmup {
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
    }
    const ACCUM: u32 = 0x22bc; // global event accumulator [sched+108]
    let pre_pc = proc.cpu.pc;
    let pre_accum = proc.bus.data_load32(ACCUM);
    eprintln!("=== M2c ISR observation (warmup {warmup}) ===");
    eprintln!(
        "pre: pc={:#x} {} INTENABLE={:#010x} INTLEVEL={} excm={} [0x22bc]={:#x} cur-task[0x2278]={:#x}",
        pre_pc,
        nearest_symbol(&proc.symbols, pre_pc & 0x00ff_ffff),
        proc.cpu.intenable,
        proc.cpu.regs.intlevel(),
        proc.cpu.regs.excm(),
        pre_accum,
        proc.bus.data_load32(0x2278),
    );
    if let Some((a, v)) = src {
        let _ = proc.cpu.data_write32(&mut proc.bus, a, v);
        eprintln!("seeded interrupt-source [{a:#x}] = {v:#x}");
    }
    // FORCE ONE deliverable window: drop INTLEVEL to 0, clear PS.EXCM, assert
    // the armed line(s). The interp's step() then delivers faithfully next tick
    // (EXCCAUSE=4, EPC1<-pc, vector 0x2958).
    proc.cpu.regs.set_intlevel(0);
    proc.cpu.regs.ps &= !0x10; // clear PS.EXCM (bit 4)
    proc.cpu.interrupt |= proc.cpu.intenable;
    eprintln!("forced delivery: INTERRUPT |= {:#010x}", proc.cpu.intenable);

    let dumpreads = std::env::var("XDNA_FW_ISR_DUMPREADS").is_ok();
    let mut prev_accum = pre_accum;
    let mut delivered = false;
    let mut returned_at: Option<u64> = None;
    let mut n = 0u64;
    while n < steps {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        // Decode for EA + call-target annotation.
        let (op_str, note) =
            match proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] =
                        std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                    let d = decode::decode(&b, proc.cpu.pc);
                    let note = match d.op {
                        Op::L32i { s, imm, .. }
                        | Op::L32iN { s, imm, .. }
                        | Op::L8ui { s, imm, .. }
                        | Op::L16ui { s, imm, .. } => {
                            let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                            if ea >= 0x2000_0000 {
                                format!(" MMIO-read ea={ea:#x}")
                            } else if dumpreads {
                                let v = proc.bus.data_load32(ea & 0x00ff_ffff);
                                format!(" read [{:#x}]={:#x}", ea & 0x00ff_ffff, v)
                            } else {
                                String::new()
                            }
                        }
                        Op::Call8 { target }
                        | Op::Call4 { target }
                        | Op::Call12 { target }
                        | Op::Call0 { target } => {
                            format!(" -> call {:#x} {}", target, nearest_symbol(&proc.symbols, target))
                        }
                        Op::Callx8 { s } | Op::Callx4 { s } | Op::Callx12 { s } => {
                            let t = proc.cpu.regs.read_ar(s);
                            format!(" -> callx {:#x} {}", t, nearest_symbol(&proc.symbols, t & 0x00ff_ffff))
                        }
                        _ => String::new(),
                    };
                    (format!("{:?}", d.op), note)
                }
                Err(_) => ("<fault>".into(), String::new()),
            };
        // Log handler entry, event-machinery hits, and all calls/MMIO.
        let key = matches!(pc, 0x2958 | 0xcf68 | 0xcf5c | 0xd84c | 0xcadc | 0x2911);
        if !delivered && pc == 0x2958 {
            delivered = true;
            eprintln!("--- [{n}] ISR ENTRY @0x2958 (EXCCAUSE={}) ---", proc.cpu.regs.exccause);
        }
        if key || !note.is_empty() {
            eprintln!("[{n:>5}] {pc:#08x} {:<22} {op_str}{note}", nearest_symbol(&proc.symbols, pc));
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(r) => {
                eprintln!("[{n}] WAIT({r:?}) @ {pc:#x}");
                n += 1;
                break;
            }
            Step::Unknown { pc: u, word } => {
                eprintln!("[{n}] UNKNOWN @ {u:#x} word={word:#010x}");
                break;
            }
        }
        let accum = proc.bus.data_load32(ACCUM);
        if accum != prev_accum {
            eprintln!("      *** [0x22bc] {prev_accum:#x} -> {accum:#x} (event posted) ***");
            prev_accum = accum;
        }
        if delivered
            && returned_at.is_none()
            && !proc.cpu.regs.excm()
            && (proc.cpu.pc & 0x00ff_ffff) == (pre_pc & 0x00ff_ffff)
        {
            returned_at = Some(n);
            eprintln!("--- [{n}] rfe RETURNED to preempted pc {:#x} ---", pre_pc);
            break;
        }
    }
    eprintln!("--- ISR observation summary ---");
    eprintln!("delivered={delivered} returned_at={returned_at:?} steps_run={n}");
    eprintln!(
        "post: [0x22bc]={:#x} (pre {:#x}) cur-task[0x2278]={:#x} pc={:#x}",
        proc.bus.data_load32(ACCUM),
        pre_accum,
        proc.bus.data_load32(0x2278),
        proc.cpu.pc,
    );
}
