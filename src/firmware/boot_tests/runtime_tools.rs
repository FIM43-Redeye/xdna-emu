use super::*;

/// Waypoint-hit probe (2026-07-09, general). Counts hits + first-hit n for a
/// set of PCs across a full natural boot. `XDNA_FW_WAYPOINTS` = comma-sep hex
/// (default = the scheduler entry points: picker `0xc980` + its two callers
/// `0x42c8`/`0xdd7a`, the command dispatcher `0xdbc4`, early-init `0x41b8`, the
/// linker `0xd4e0`). Answers "does boot ever reach the picker / the early-init
/// scheduler-start call". Ignored unless XDNA_FW_PROBE.
#[test]
fn m2c_probe_waypoint_hits() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the waypoint-hit probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let wps: Vec<u32> = std::env::var("XDNA_FW_WAYPOINTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_else(|| vec![0xc980, 0x42c8, 0xdd7a, 0xdbc4, 0x41b8, 0xd4e0]);
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);

    let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut count: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut n = 0u64;
    while n < max {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        if wps.contains(&pc) {
            first.entry(pc).or_insert(n);
            *count.entry(pc).or_insert(0) += 1;
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
    }
    eprintln!("=== waypoint hits (natural boot, n<{max}) ===");
    for w in &wps {
        match first.get(w) {
            Some(f) => eprintln!(
                "  {w:#08x} ({:<26}) first-hit n={f} (x{})",
                nearest_symbol(&proc.symbols, *w),
                count[w]
            ),
            None => eprintln!("  {w:#08x} ({:<26}) NEVER", nearest_symbol(&proc.symbols, *w)),
        }
    }
}

/// Thread-1 control-flow probe: how does execution ENTER `0x8773` (the
/// `Srli a6,a6,2` that produces the garbage mask `0x9268`)?  If the
/// `Movi a6,-1` at `0x876d` immediately precedes it, a6 should be
/// `0xffffffff` and the mask correct; the probe caught a6=`0x249a0` instead,
/// so `0x876d` must have been skipped.  Dumps a ring of the last N executed
/// (pc, op, a6) before the first arrival at a trigger PC (default `0x8773`,
/// override XDNA_FW_TRIG).  Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_pc_history() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the pc-history probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    proc.enable_host_mailbox();
    let stop: u64 = std::env::var("XDNA_FW_DUMP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);
    let trig: u32 = std::env::var("XDNA_FW_TRIG")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x8773);
    let depth: usize = std::env::var("XDNA_FW_HIST").ok().and_then(|s| s.parse().ok()).unwrap_or(28);

    let mut ring: std::collections::VecDeque<(u64, u32, String, u32, u32)> =
        std::collections::VecDeque::with_capacity(depth + 1);
    let mut n = 0u64;
    let mut fired = false;
    while n < stop {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let op_str =
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                format!("{:?}", decode::decode(&b, proc.cpu.pc).op)
            } else {
                "<xlate-fail>".to_string()
            };
        let a0 = proc.cpu.regs.read_ar(0);
        let a6 = proc.cpu.regs.read_ar(6);
        ring.push_back((n, pc, op_str, a0, a6));
        if ring.len() > depth {
            ring.pop_front();
        }
        if pc == trig {
            fired = true;
            eprintln!("=== pc-history: last {depth} instrs entering {trig:#x} (n={n}) ===");
            for (nn, p, op, a0, a6) in &ring {
                eprintln!(
                    "  n={nn:>7} pc={p:#08x} ({:<26}) a0={a0:#x} a6={a6:#x}  {op}",
                    nearest_symbol(&proc.symbols, *p)
                );
            }
            break;
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        proc.host_mailbox.tick(&mut proc.bus);
    }
    if !fired {
        eprintln!("trigger {trig:#x} never reached in n<{stop}");
    }
}

/// M2c Phase 2 DIAGNOSTIC: run the boot until it walls (unknown-op / spin /
/// idle) and dump a ring buffer of the last N instructions leading up to the
/// stop -- translated disassembly + the full a0..a15 window each step. Finds
/// the exact control-transfer instruction that lands on a bad target.
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_trace_to_wall() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the trace-to-wall probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    const KEEP: usize = 48;
    // Ring buffer of (instr_n, pc, disasm, a0..a15).
    let mut ring: std::collections::VecDeque<(u64, u32, String, [u32; 16], u32, u32)> =
        std::collections::VecDeque::with_capacity(KEEP + 1);
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    // XDNA_FW_CALLS: record ONLY call-family / entry / retw events in the
    // ring, so the tail shows the call/return structure (e.g. an unbounded
    // recursion cycle) instead of the inner-loop or overflow-handler grind.
    let calls_only = std::env::var("XDNA_FW_CALLS").is_ok();
    // XDNA_FW_STOP_PC=<hex>: stop the probe the first time execution reaches
    // this PC, so the ring shows the call-path INTO it (e.g. the top-level
    // caller that first enters a loop we later see recurse).
    let stop_pc = std::env::var("XDNA_FW_STOP_PC")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    while n < max {
        let pc = proc.cpu.pc;
        let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                // Disassemble via `fetch8` (vaddr-aware), NOT `peek8` (phys):
                // low-`.text` overlay PCs (0x581c-0x5d30, 0x800-0x980) execute
                // from file+0x100, so a phys peek shows garbage/misclassified
                // ops there. Fetch8 matches what `step` actually runs. Peek up
                // to 8 bytes so FLIX bundles (op0 0xe/0xf) disassemble fully.
                let b: [u8; 8] = std::array::from_fn(|i| {
                    proc.bus.fetch8(pc.wrapping_add(i as u32), phys.wrapping_add(i as u32))
                });
                format!("{:?}", decode::decode(&b, pc).op)
            }
            Err(_) => "<fetch-fault>".to_string(),
        };
        let mut regs = [0u32; 16];
        for (r, slot) in regs.iter_mut().enumerate() {
            *slot = proc.cpu.regs.read_ar(r as u8);
        }
        let record = !calls_only
            || disasm.starts_with("Call")
            || disasm.starts_with("Entry")
            || disasm.starts_with("Retw")
            || disasm.starts_with("RetN");
        if record {
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm, regs, proc.cpu.regs.windowbase, proc.cpu.regs.windowstart));
        }

        if Some(pc) == stop_pc {
            stop = format!("XDNA_FW_STOP_PC {pc:#x} reached at n={n}");
            break;
        }

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
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }

    eprintln!("=== M2c trace-to-wall (last {KEEP} instrs before the stop) ===");
    eprintln!("instrs executed = {n}");
    eprintln!("stop reason     = {stop}");
    eprintln!("cpu.vecbase     = {:#x}", proc.cpu.vecbase);
    for (i, pc, disasm, regs, wb, ws) in &ring {
        let lo: Vec<String> = (0..8).map(|r| format!("a{r}={:#x}", regs[r])).collect();
        let sym = nearest_symbol(&proc.symbols, *pc);
        eprintln!("{i:>6} pc={pc:#x} {sym:<24} {disasm:<30} wb={wb} ws={ws:#06x} | {}", lo.join(" "));
    }
    // The full a0..a15 window of the last few instructions (call/window state).
    eprintln!("--- a8..a15 of the final {} instrs ---", 6.min(ring.len()));
    for (i, pc, _, regs, _, _) in ring.iter().rev().take(6).rev() {
        let hi: Vec<String> = (8..16).map(|r| format!("a{r}={:#x}", regs[r])).collect();
        eprintln!("{i:>6} pc={pc:#x} | {}", hi.join(" "));
    }
}

/// M2c iter18 DIAGNOSTIC: peek 32-bit words (and one level of deref) at a
/// set of static addresses. Used to resolve literal-pool words (L32r
/// targets) to the pointers/sentinels/bases they hold -- e.g. the scheduler
/// event-source pointer at lit 0x3364. XDNA_FW_PEEK=comma-sep hex addresses.
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_peek() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the peek probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let addrs: Vec<u32> = std::env::var("XDNA_FW_PEEK")
        .expect("set XDNA_FW_PEEK=addr,addr,... (hex)")
        .split(',')
        .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let proc = FirmwareProcessor::load_m2c(img);
    let rd = |a: u32| u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
    eprintln!("=== M2c peek ===");
    for a in addrs {
        let w = rd(a);
        let deref = rd(w);
        eprintln!(
            "  [{a:#010x}] = {w:#010x}  {:<24}  ->deref [{w:#010x}] = {deref:#010x}",
            nearest_symbol(&proc.symbols, w)
        );
    }
}

/// M2c iter18 DIAGNOSTIC: scan mapped memory ranges for a target 32-bit LE
/// word (e.g. a function pointer 0x00005580 to locate the vector/dispatch
/// slot that reaches the event ISR). XDNA_FW_SCAN=targetword (hex),
/// XDNA_FW_SCAN_RANGES=start:end,start:end,... (hex; default = low image +
/// relocated segment). Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_word_scan() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the word-scan probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let target = u32::from_str_radix(
        std::env::var("XDNA_FW_SCAN")
            .expect("set XDNA_FW_SCAN=word (hex)")
            .trim()
            .trim_start_matches("0x"),
        16,
    )
    .expect("target hex");
    let ranges: Vec<(u32, u32)> = std::env::var("XDNA_FW_SCAN_RANGES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|r| {
                    let (a, b) = r.split_once(':')?;
                    Some((
                        u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                        u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?,
                    ))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(0x0, 0x4_0000), (0x0800_0000, 0x0900_0000)]);
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let proc = FirmwareProcessor::load_m2c(img);
    let rd = |a: u32| u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
    eprintln!("=== M2c word-scan for {target:#010x} ===");
    let mut hits = 0u64;
    for (start, end) in ranges {
        let mut a = start & !3;
        while a < end {
            if rd(a) == target {
                eprintln!("  {a:#010x}  {}", nearest_symbol(&proc.symbols, a));
                hits += 1;
            }
            a = a.wrapping_add(4);
        }
    }
    eprintln!("total hits = {hits}");
}

/// M2c iter18 DIAGNOSTIC: runtime store-VALUE watch. Runs boot and records
/// every store (S32i/S32iN/S16i/S8i) whose stored value == XDNA_FW_WATCH_VAL
/// (hex), with the store address + pc. Locates where a function pointer
/// (e.g. the event ISR 0x5580) or a magic gets installed into a table at
/// runtime -- the store address is the dispatch-table slot (which encodes
/// the IRQ number). XDNA_FW_MAX overrides the budget. Ignored unless
/// XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_store_value_watch() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the store-value watch");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let target = u32::from_str_radix(
        std::env::var("XDNA_FW_WATCH_VAL")
            .expect("set XDNA_FW_WATCH_VAL=value (hex)")
            .trim()
            .trim_start_matches("0x"),
        16,
    )
    .expect("value hex");
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let mut n = 0u64;
    let mut stop = "budget reached";
    let mut hits: Vec<(u64, u32, u32)> = Vec::new();
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
                if val == target && hits.len() < 64 {
                    hits.push((n, pc, addr));
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
        if proc.bus.sysstub().spinning().is_some() {
            stop = "spin detected";
            break;
        }
    }
    eprintln!("=== M2c store-value watch for {target:#010x} ===");
    eprintln!("instrs = {n}, stop = {stop}");
    for (at, pc, addr) in &hits {
        eprintln!(
            "  n={at:>8}  pc={pc:#08x} {:<26}  store -> [{addr:#010x}]",
            nearest_symbol(&proc.symbols, *pc)
        );
    }
    eprintln!("total stores of {target:#010x} = {}", hits.len());
}

/// M2c iter18 DIAGNOSTIC: runtime store-to-ADDRESS watch. Runs boot and
/// records every store (S32i/S32iN/S16i/S8i) whose target address is in
/// XDNA_FW_WATCH_ADDR (comma-sep hex), with the stored value + pc + instr.
/// Directly shows whether/where the pending-event words (0x22bc, task+0x30)
/// or the current-task pointer (0x2278) get written -- the event-injection
/// point. XDNA_FW_MAX overrides the budget. Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_addr_store_watch() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the addr store watch");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let watch: Vec<u32> = std::env::var("XDNA_FW_WATCH_ADDR")
        .expect("set XDNA_FW_WATCH_ADDR=addr,addr,... (hex)")
        .split(',')
        .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let mut n = 0u64;
    let mut stop = "budget reached";
    // (n, pc, addr, value); cap per address so a hot store can't flood.
    let mut hits: Vec<(u64, u32, u32, u32)> = Vec::new();
    let mut per_addr: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
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
                if watch.contains(&addr) {
                    let c = per_addr.entry(addr).or_insert(0);
                    *c += 1;
                    if *c <= 12 {
                        hits.push((n, pc, addr, val));
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
        if proc.bus.sysstub().spinning().is_some() {
            stop = "spin detected";
            break;
        }
    }
    eprintln!("=== M2c addr store watch {watch:x?} ===");
    eprintln!("instrs = {n}, stop = {stop}");
    for (at, pc, addr, val) in &hits {
        eprintln!(
            "  n={at:>8}  pc={pc:#010x}  [{addr:#010x}] <- {val:#010x}   {}",
            nearest_symbol(&proc.symbols, *pc & 0x00ff_ffff)
        );
    }
    eprintln!("--- store counts per watched address ---");
    let mut counts: Vec<(u32, u64)> = per_addr.into_iter().collect();
    counts.sort_by_key(|(a, _)| *a);
    for (a, c) in counts {
        eprintln!("  [{a:#010x}]: {c}");
    }
}

/// M2c iter18 DIAGNOSTIC: POLL-based value watch (reliable). Reads each
/// XDNA_FW_POLL_ADDR (comma-sep hex) via `bus.data_load32` every step and
/// records value changes (n, pc, old->new). Unlike the store-EA watches,
/// this catches writes through ANY addressing/alias (the firmware's windowed
/// RAM is written via aliased virtual addresses that exact-EA matching
/// misses -- proven by `0x2278`). Use for the pending-event words / task
/// state. XDNA_FW_MAX overrides budget. Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_poll_watch() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the poll watch");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let addrs: Vec<u32> = std::env::var("XDNA_FW_POLL_ADDR")
        .expect("set XDNA_FW_POLL_ADDR=addr,addr,... (hex)")
        .split(',')
        .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let mut last: Vec<u32> = addrs.iter().map(|a| proc.bus.data_load32(*a)).collect();
    let mut changes: Vec<(u64, u32, u32, u32, u32)> = Vec::new(); // n, pc, addr, old, new
    let mut per: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut n = 0u64;
    let mut stop = "budget reached";
    while n < max {
        let pc = proc.cpu.pc;
        for (i, a) in addrs.iter().enumerate() {
            let v = proc.bus.data_load32(*a);
            if v != last[i] {
                let c = per.entry(*a).or_insert(0);
                *c += 1;
                if *c <= 16 {
                    changes.push((n, pc, *a, last[i], v));
                }
                last[i] = v;
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
        if proc.bus.sysstub().spinning().is_some() {
            stop = "spin detected";
            break;
        }
    }
    eprintln!("=== M2c poll watch {addrs:x?} ===");
    eprintln!("instrs = {n}, stop = {stop}");
    for (at, pc, addr, old, new) in &changes {
        eprintln!("  n={at:>8}  pc={:#08x}  [{addr:#010x}] {old:#010x} -> {new:#010x}", pc & 0x00ff_ffff);
    }
    eprintln!("--- change counts per address ---");
    let mut counts: Vec<(u32, u64)> = per.into_iter().collect();
    counts.sort_by_key(|(a, _)| *a);
    for (a, c) in counts {
        eprintln!("  [{a:#010x}]: {c} change(s)");
    }
}

/// M2c iter18 DIAGNOSTIC: clean EXECUTION trace (follows real PCs, so decode
/// alignment is always correct -- unlike linear disasm). Warms up
/// XDNA_FW_TRACE_WARMUP instrs, then prints the next XDNA_FW_TRACE_COUNT as
/// `n pc symbol op | a2..a7 | ea=<load/store EA & value>`. Reveals the
/// steady-state task work loop and its decision points. Ignored unless
/// XDNA_FW_PROBE set.
#[test]
fn m2c_probe_exec_trace() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the exec trace");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let warmup = env_u64("XDNA_FW_TRACE_WARMUP", 300_000);
    let count = env_u64("XDNA_FW_TRACE_COUNT", 400);
    // Optional: seed the per-column poll-completion flags each traced step,
    // so the trace shows the work-fn's SATISFIED-poll path (what it reads and
    // branches on AFTER the poll succeeds). XDNA_FW_TRACE_SEEDPOLL=<hex bits>
    // (e.g. 0xb = bit0|bit1|bit3).
    let seedpoll: Option<u32> = std::env::var("XDNA_FW_TRACE_SEEDPOLL")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
    for _ in 0..warmup {
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
    }
    eprintln!("=== M2c exec trace (warmup {warmup}, {count} instrs, seedpoll={seedpoll:x?}) ===");
    for i in 0..count {
        if let Some(bits) = seedpoll {
            for k in 0..4u32 {
                let _ = proc.cpu.data_write32(&mut proc.bus, 0x2727_1000 + k * 0x1000, bits);
                let _ = proc.cpu.data_write8(&mut proc.bus, 0xf9e0 + k * 0x60, bits);
            }
        }
        let pc = proc.cpu.pc;
        let (op, ea) = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, pc);
                let ea = match d.op {
                    decode::Op::L32i { s, imm, .. }
                    | decode::Op::L32iN { s, imm, .. }
                    | decode::Op::L8ui { s, imm, .. }
                    | decode::Op::L16ui { s, imm, .. }
                    | decode::Op::S32i { s, imm, .. }
                    | decode::Op::S32iN { s, imm, .. }
                    | decode::Op::S8i { s, imm, .. } => {
                        let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                        format!(" ea={:#x}={:#x}", a, proc.bus.data_load32(a & 0x00ff_ffff))
                    }
                    _ => String::new(),
                };
                (format!("{:?}", d.op), ea)
            }
            Err(_) => ("<fault>".to_string(), String::new()),
        };
        let regs: Vec<String> = (2..8).map(|r| format!("a{r}={:#x}", proc.cpu.regs.read_ar(r))).collect();
        eprintln!(
            "{:>7} {:#08x} {:<22} {:<34}{ea} | {}",
            warmup + i,
            pc & 0x00ff_ffff,
            nearest_symbol(&proc.symbols, pc & 0x00ff_ffff),
            op,
            regs.join(" ")
        );
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            eprintln!("stop");
            break;
        }
    }
}
