use super::*;

/// M2c Phase 0 (iter18) DIAGNOSTIC: image-wide STATIC search for store
/// instructions with a given byte displacement (default 0x30, the task
/// done-flag offset the dispatcher reads via `l32i.n [task+0x30]`).
/// Resolves the shape-(i)-vs-(ii) completion-writer fork: if NO store
/// anywhere in the firmware's code targets `+0x30`, then only an external
/// agent (DMA/peripheral) can set a task's done-flag (shape ii); if some
/// store does, it names the candidate writer for inspection.
///
/// Disassembles every function listed in `symbols.txt` linearly (each entry
/// up to the next symbol). Reads code via `fetch8(vaddr, vaddr)` -- the
/// reset's way-6 identity region makes VMA==phys across all code, and
/// `fetch8` applies the low-window overlays -- so it covers NEVER-EXECUTED
/// functions too, which is the whole point of a static search (vs. iter18's
/// runtime store-watch that only saw executed code). Set XDNA_FW_STORE_DISP
/// to search another offset. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_store_search() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the static store-search");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let disp = std::env::var("XDNA_FW_STORE_DISP")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x30);
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Symbol entries sorted by address; disassemble each [entry, next).
    let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
    syms.sort_by_key(|(a, _)| *a);

    eprintln!("=== M2c static store-search: stores with displacement {disp:#x} ===");
    eprintln!("(each hit: pc  symbol  op -- store address is AR[s]+{disp:#x})");
    let mut hits = 0u32;
    let mut store_disps: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for i in 0..syms.len() {
        let start = syms[i].0;
        let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400);
        // Cap per-function span so a bogus symbol gap can't run away.
        let end = end.min(start + 0x2000);
        let mut pc = start;
        while pc < end {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
            let d = decode::decode(&b, pc);
            let store_imm = match d.op {
                decode::Op::S32i { imm, .. }
                | decode::Op::S32iN { imm, .. }
                | decode::Op::S8i { imm, .. }
                | decode::Op::S16i { imm, .. } => Some(imm),
                _ => None,
            };
            if let Some(imm) = store_imm {
                *store_disps.entry(imm).or_insert(0) += 1;
                if imm == disp {
                    let sym = nearest_symbol(&proc.symbols, pc);
                    eprintln!("  pc={pc:#08x}  {sym:<28}  {:?}", d.op);
                    hits += 1;
                }
            }
            pc += (d.len as u32).max(1);
        }
    }
    eprintln!("--- {hits} store(s) with displacement {disp:#x} across {} functions ---", syms.len());
    eprintln!("--- store-displacement histogram (top offsets) ---");
    let mut by_count: Vec<(u32, u32)> = store_disps.into_iter().collect();
    by_count.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (off, c) in by_count.iter().take(16) {
        eprintln!("  disp {off:#06x}: {c}");
    }
}

/// M2c #140 RE TOOL: static scan of every RSR/WSR/XSR in the firmware,
/// histogrammed by special-register number, with the known names. Answers
/// "does the firmware use the Xtensa TIMER?" -- if it programs CCOMPARE
/// (0xF0/F1/F2) it drives a timer-tick interrupt, and since the interp
/// models CCOUNT/CCOMPARE as no-ops (interp/mod.rs: "every other SR is a
/// logged no-op"), that tick can NEVER fire in EMU -> the event-poll run-fn
/// is starved by construction. The CCOMPARE period would then BE the coarse
/// completion-poll cadence that dominates the fitted 8000-cycle dma_wait.
/// Walks all code via the reset way-6 identity region (covers never-executed
/// code). Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_sr_usage() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the SR-usage scan");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    fn sr_name(sr: u16) -> &'static str {
        match sr {
            0x00 => "LBEG",
            0x01 => "LEND",
            0x02 => "LCOUNT",
            0x03 => "SAR",
            0x48 => "WINDOWBASE",
            0x49 => "WINDOWSTART",
            0x53 => "PTEVADDR",
            0x5A => "ITLBCFG",
            0x5B => "DTLBCFG",
            0x60 => "IBREAKENABLE",
            0x83 => "EPC1",
            0x90 => "EPS2",
            0xB0 => "DEPC",
            0xB1 => "EXCSAVE1",
            0xC0 => "EXCCAUSE",
            0xD1 => "EXCVADDR",
            0xE2 => "INTERRUPT/INTSET",
            0xE3 => "INTCLEAR",
            0xE4 => "INTENABLE",
            0xE6 => "PS",
            0xE7 => "VECBASE",
            0xEA => "CCOUNT (TIMER)",
            0xF0 => "CCOMPARE0 (TIMER)",
            0xF1 => "CCOMPARE1 (TIMER)",
            0xF2 => "CCOMPARE2 (TIMER)",
            _ => "",
        }
    }

    let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
    syms.sort_by_key(|(a, _)| *a);

    // sr -> (rsr, wsr, xsr, first_pc)
    let mut usage: std::collections::BTreeMap<u16, (u32, u32, u32, u32)> = std::collections::BTreeMap::new();
    for i in 0..syms.len() {
        let start = syms[i].0;
        let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
        let mut pc = start;
        while pc < end {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
            let d = decode::decode(&b, pc);
            let hit = match d.op {
                decode::Op::Rsr { sr, .. } => Some((sr as u16, 0)),
                decode::Op::Wsr { sr, .. } => Some((sr as u16, 1)),
                _ => None,
            };
            if let Some((sr, kind)) = hit {
                let e = usage.entry(sr).or_insert((0, 0, 0, pc));
                match kind {
                    0 => e.0 += 1,
                    1 => e.1 += 1,
                    _ => e.2 += 1,
                }
            }
            pc += (d.len as u32).max(1);
        }
    }

    eprintln!("=== M2c static SR-usage scan (#140 timer hunt) ===");
    eprintln!("  sr      rsr wsr xsr  first-pc            name");
    let mut timer_used = false;
    for (&sr, &(r, w, x, first_pc)) in &usage {
        let name = sr_name(sr);
        if matches!(sr, 0xEA | 0xF0 | 0xF1 | 0xF2) {
            timer_used = true;
        }
        eprintln!(
            "  {sr:#04x}   {r:>3} {w:>3} {x:>3}  {first_pc:#08x} {:<20} {name}",
            nearest_symbol(&proc.symbols, first_pc)
        );
    }
    eprintln!(
        "--- TIMER (CCOUNT/CCOMPARE) used by firmware: {} ---",
        if timer_used {
            "YES -- timer-driven tick; interp models it as no-op"
        } else {
            "NO"
        }
    );
}

/// M2c iter18 RE TOOL: static disassembly of an arbitrary VMA range, read
/// via `fetch8` over the reset way-6 identity region (covers never-executed
/// code and branches not taken -- unlike the trace probes, which only show
/// the executed path). Set XDNA_FW_DISASM=<start>:<end> (hex VMAs) to pick
/// the range; each line is `pc symbol op` walked by decoded length. Reading
/// the actual control flow of a function beats theorizing about it.
/// `XDNA_FW_DISASM_FILEOFF=<hex>` bypasses the installed fetch overlays and
/// reads raw image bytes at `(VMA & 0x00ff_ffff) + FILEOFF`; this distinguishes
/// low-window overlay code from the base-framed high code alias at the same VMA.
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_disasm_range() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the range disassembler");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let range = std::env::var("XDNA_FW_DISASM").unwrap_or_else(|_| "0xd7f0:0xd870".to_string());
    let (s, e) = range.split_once(':').expect("XDNA_FW_DISASM must be start:end (hex)");
    let start = u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect("start hex");
    let end = u32::from_str_radix(e.trim().trim_start_matches("0x"), 16).expect("end hex");
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // XDNA_FW_DISASM_OVL=lo:hi -- register a +0x100 fetch overlay over that
    // vaddr range before disassembling, to view an un-mapped region in its
    // piecewise-relocated (LMA=vaddr+0x100) framing. Lets a +0x100-seam
    // candidate be compared against its base framing without editing load_m2c.
    if let Ok(ovl) = std::env::var("XDNA_FW_DISASM_OVL") {
        let (a, b) = ovl.split_once(':').expect("XDNA_FW_DISASM_OVL must be lo:hi (hex)");
        let lo = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).expect("lo hex");
        let hi = u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).expect("hi hex");
        proc.bus.add_rom_overlay(lo, hi, LOW_VMA_FILE_OFFSET);
        eprintln!("(+0x100 overlay registered over {lo:#x}..{hi:#x})");
    }

    let raw_file_offset = std::env::var("XDNA_FW_DISASM_FILEOFF")
        .ok()
        .map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect("file offset hex"));
    if let Some(off) = raw_file_offset {
        eprintln!("(raw image view: file = (VMA & 0x00ff_ffff) + {off:#x})");
    }

    eprintln!("=== M2c static disasm {start:#x}..{end:#x} ===");
    let mut pc = start;
    while pc < end {
        let b: [u8; 8] = std::array::from_fn(|k| {
            if let Some(off) = raw_file_offset {
                raw.get(((pc & 0x00ff_ffff) + off + k as u32) as usize).copied().unwrap_or(0)
            } else {
                proc.bus.fetch8(pc + k as u32, pc + k as u32)
            }
        });
        let d = decode::decode(&b, pc);
        let sym = nearest_symbol(&proc.symbols, pc);
        let raw_hex: String =
            b[..(d.len as usize).max(1).min(8)].iter().map(|x| format!("{x:02x}")).collect();
        eprintln!("  {pc:#08x} {sym:<26} {:<40} [{raw_hex}]", format!("{:?}", d.op));
        pc += (d.len as u32).max(1);
    }
}

/// M2c GOLD LISTING (2026-07-09): overlay-correct recursive-descent
/// disassembly of the whole reachable firmware, on OUR ground-truth decoder.
///
/// Motivation: linear-sweep disasm (`m2c_probe_disasm_range` from 0x0)
/// desyncs the instant it hits a literal pool or the signed header (vaddr <
/// 0x1a4 is the `$PS1` header, NOT code), producing garbage that masks real
/// findings. Ghidra's recursive descent is desync-free but (a) misses
/// indirect/vtable-reached code -- exactly the scheduler core (picker
/// 0xc980, idle 0xc8e0, FUN_000041b8/dbc4, go-alive 0x55f8) -- and (b) uses
/// an Xtensa module that mis-decodes this image's FLIX bundles.
///
/// This walks control flow from every known entry (the full symbol map +
/// the reset vector + the VECBASE=0x800 vector stubs + the indirect
/// scheduler targets we recovered by hand), decoding through `bus.fetch8`
/// (so the `+0x5c` base / `+0x100` low-VMA / Seg-B overlays are ALL applied
/// correctly and identically to execution). Every visited instruction is a
/// real instruction boundary, so there is no desync. Unvisited bytes are
/// data (literal pools / padding / .bss) and are marked as gaps, not forced
/// into instructions.
///
/// THE GATE (the direct answer to "is the overlay faulty anywhere"): any
/// reachable instruction that decodes to `Op::Unknown` is either an
/// overlay/mapping fault (wrong bytes at that vaddr) or a genuinely
/// undecodable op. The probe reports every such site with how it was reached
/// (seed / fall-through / branch-target) so fall-through Unknowns -- the ones
/// that signal a real mid-function mapping problem -- stand out from
/// branch-target ones (which may just be a mis-followed indirect-ish edge).
/// It also cross-checks inst-fetch vs data-load bytes over the visited set
/// (a Harvard/overlay split would diverge). Writes the full listing to
/// `build/experiments/firmware-re/gold-listing.txt`. Extra comma-separated
/// hex seeds via `XDNA_FW_GOLD_SEEDS`. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_gold_disasm() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the gold-listing disassembler");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    use std::collections::{BTreeMap, HashSet};
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Code regions (vaddr) that may hold instructions: the low .text
    // (base-mapped, plus the two low-VMA overlays) and the PSP-relocated
    // Segment B runtime code-tail at 0x08b00000. Everything else (device
    // apertures, .bss zero-desert) is not code; descent never leaves these.
    // Low .text/.rodata ends ~0xe7xx (largest fn tops ~0xc1e5, code pointers
    // to ~0xe594); file 0x10000..0x2d100 is the entropy-scan zero desert
    // (.bss/pad), so the low region stops at 0x10000 -- a branch computed
    // into the desert is data, not code. Seg-B is the relocated runtime tail.
    let regions: [(u32, u32); 2] = [(0x1a4, 0x0001_0000), (0x08b0_0000, 0x08b0_fa10)];
    let in_code = |a: u32| regions.iter().any(|&(lo, hi)| a >= lo && a < hi);

    // Extract every statically-known control-flow target the descent should
    // follow (direct call/branch/jump/loop). Register-indirect callx*/jx and
    // Syscall have no static target. Returns the list of destination vaddrs.
    fn cf_targets(op: &decode::Op) -> Vec<u32> {
        use decode::Op::*;
        match *op {
            Call0 { target, .. }
            | Call4 { target, .. }
            | Call8 { target, .. }
            | Call12 { target, .. }
            | J { target }
            | Beqz { target, .. }
            | Bnez { target, .. }
            | Bltz { target, .. }
            | Bgez { target, .. }
            | BeqzN { target, .. }
            | BnezN { target, .. }
            | Beq { target, .. }
            | Bne { target, .. }
            | Blt { target, .. }
            | Bltu { target, .. }
            | Bge { target, .. }
            | Bgeu { target, .. }
            | Beqi { target, .. }
            | Bnei { target, .. }
            | Blti { target, .. }
            | Bgei { target, .. }
            | Bltui { target, .. }
            | Bgeui { target, .. }
            | Bbci { target, .. }
            | Bbsi { target, .. }
            | Bbc { target, .. }
            | Bbs { target, .. }
            | Bnone { target, .. }
            | Bany { target, .. }
            | Ball { target, .. }
            | Bnall { target, .. } => vec![target],
            Loop { end, .. } | Loopnez { end, .. } => vec![end],
            _ => vec![],
        }
    }
    // Does this instruction end the fall-through run? (control leaves; the
    // next byte is NOT guaranteed to be an instruction). Unconditional J and
    // indirect Jx transfer away; the ret/rf* family returns; Unknown is a
    // wall. Conditional branches, calls (callee returns), Syscall, and
    // Loop all CONTINUE the fall-through.
    fn is_terminator(op: &decode::Op) -> bool {
        use decode::Op::*;
        matches!(op, J { .. } | Jx { .. } | Retw | RetwN | RetN | Rfe | Rfwo | Rfwu | Unknown { .. })
    }

    // How a visited PC was first reached -- to triage the gate.
    #[derive(Clone, Copy, PartialEq)]
    enum Reach {
        Seed,
        Fall,
        Branch,
    }

    // Seeds. XDNA_FW_GOLD_RESETONLY=1 seeds ONLY the reset/vector/verified
    // entries (below), NOT the Ghidra symbol map -- so every Unknown is
    // reached by pure control-flow closure from a known-good boundary. Any
    // Unknown in THAT run is a real overlay/decoder concern, not a
    // misaligned-symbol-seed artifact (Ghidra FUN_ labels are sometimes a
    // byte or two off a real instruction boundary, which seeds garbage).
    let reset_only = std::env::var("XDNA_FW_GOLD_RESETONLY").is_ok();
    let mut work: Vec<(u32, Reach)> = Vec::new();
    let mut skipped_syms = 0u64;
    if !reset_only {
        // Ghidra `FUN_` labels are sometimes a byte or two off a real
        // instruction boundary; seeding descent there decodes garbage.
        // Validate each symbol seed: decode a short run from it and skip it
        // if an Unknown appears within the first few instructions (a
        // misaligned label garbles almost immediately, a real function start
        // does not).
        for (&a, _) in proc.symbols.iter() {
            if !in_code(a) {
                continue;
            }
            let mut p = a;
            let mut aligned = true;
            for _ in 0..4 {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(p + k as u32, p + k as u32));
                let d = decode::decode(&b, p);
                if matches!(d.op, decode::Op::Unknown { .. }) {
                    aligned = false;
                    break;
                }
                p += (d.len as u32).max(1);
            }
            if aligned {
                work.push((a, Reach::Seed));
            } else {
                skipped_syms += 1;
            }
        }
    }
    // Reset vector, the general-exception vector stub (VECBASE=0x800 +
    // 0x2e0 = 0xae0, live-confirmed in exception-dispatch-pc-verdict.md),
    // the whole VECBASE stub table start, and the indirect scheduler core.
    for &s in &[
        0x1a4u32, 0xae0, 0x800, 0xc980, 0xc8e0, 0x41b8, 0xdbc4, 0x55f8, 0xd4e0, 0xd664, 0xd7f0, 0x588c,
        0x50e8, 0x56e6, 0xd84c, 0x2958, 0x28b4,
    ] {
        work.push((s, Reach::Seed));
    }
    if let Ok(extra) = std::env::var("XDNA_FW_GOLD_SEEDS") {
        for t in extra.split(',') {
            if let Ok(a) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) {
                work.push((a, Reach::Seed));
            }
        }
    }

    // Two-pass recursive descent with literal-pool awareness. Xtensa
    // compilers place 4-byte literal words INLINE in the code stream
    // (`l32r` loads from them). Pass 0 collects every `l32r` target as a
    // data word; pass 1 re-descends treating each such word as a hard data
    // boundary, so descent never falls/branches into a pool and mis-decodes
    // it. (Those pool-walks were the bulk of the residual Unknowns; xtdis
    // confirms the bytes are data -- libisa can't decode them either.)
    let seeds = work;
    let mut decoded: BTreeMap<u32, (decode::Op, u8, [u8; 8])> = BTreeMap::new();
    let mut reach: BTreeMap<u32, Reach> = BTreeMap::new();
    let mut data_bytes: HashSet<u32> = HashSet::new();
    for _pass in 0..2 {
        decoded.clear();
        reach.clear();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut work = seeds.clone();
        while let Some((entry, how)) = work.pop() {
            if !in_code(entry) || data_bytes.contains(&entry) {
                continue;
            }
            let mut pc = entry;
            let mut first = true;
            loop {
                if !in_code(pc) || visited.contains(&pc) || data_bytes.contains(&pc) {
                    break;
                }
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                let d = decode::decode(&b, pc);
                visited.insert(pc);
                reach.insert(pc, if first { how } else { Reach::Fall });
                decoded.insert(pc, (d.op.clone(), d.len, b));
                if let decode::Op::L32r { target, .. } = d.op {
                    for k in 0..4 {
                        data_bytes.insert(target.wrapping_add(k));
                    }
                }
                for t in cf_targets(&d.op) {
                    if in_code(t) && !visited.contains(&t) && !data_bytes.contains(&t) {
                        work.push((t, Reach::Branch));
                    }
                }
                if is_terminator(&d.op) {
                    break;
                }
                pc += (d.len as u32).max(1);
                first = false;
            }
        }
    }
    let visited: HashSet<u32> = decoded.keys().copied().collect();

    // Emit the listing + gaps.
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("build/experiments/firmware-re/gold-listing.txt");
    let mut out = String::new();
    out.push_str("=== M2c GOLD LISTING: overlay-correct recursive-descent (our decoder) ===\n");
    let mut unknowns: Vec<(u32, Reach, u32)> = Vec::new();
    let mut prev_end: Option<u32> = None;
    for (&pc, (op, len, b)) in decoded.iter() {
        if let Some(pe) = prev_end {
            if pc > pe && in_code(pe) {
                // Only note gaps inside a single region (skip the big
                // low-text -> Seg-B jump).
                if regions.iter().any(|&(lo, hi)| pe >= lo && pc <= hi) {
                    out.push_str(&format!(
                        "  {:#08x} .. {:#08x}  [gap {} bytes -- data/pool/pad]\n",
                        pe,
                        pc,
                        pc - pe
                    ));
                }
            }
        }
        let sym = nearest_symbol(&proc.symbols, pc);
        let raw_hex: String = b[..(*len as usize).max(1).min(8)].iter().map(|x| format!("{x:02x}")).collect();
        let r = reach[&pc];
        let rc = match r {
            Reach::Seed => 'S',
            Reach::Fall => '.',
            Reach::Branch => 'b',
        };
        out.push_str(&format!("{rc} {pc:#08x} {sym:<26} {:<42} [{raw_hex}]\n", format!("{:?}", op)));
        if let decode::Op::Unknown { word } = op {
            unknowns.push((pc, r, *word));
        }
        prev_end = Some(pc + (*len as u32).max(1));
    }
    std::fs::write(&out_path, &out).expect("write gold listing");

    // Harvard/overlay split check: over the visited set, does inst-fetch
    // agree with data-load byte-for-byte? (A split would diverge.)
    let mut harvard_mismatch = 0u64;
    for &pc in visited.iter() {
        let fetched = proc.bus.fetch8(pc, pc);
        let loaded = proc.cpu.data_read8(&mut proc.bus, pc).unwrap_or(fetched);
        if fetched != loaded {
            harvard_mismatch += 1;
        }
    }

    // Report.
    eprintln!("=== GOLD LISTING SUMMARY ===");
    eprintln!("listing written to {}", out_path.display());
    eprintln!("instructions decoded (reachable) = {}", decoded.len());
    eprintln!("literal-pool data words marked   = {}", data_bytes.len() / 4);
    eprintln!("misaligned symbol seeds skipped  = {skipped_syms}");
    eprintln!("--- THE GATE: Unknown-on-reachable-code = {} ---", unknowns.len());
    let fall_unknowns = unknowns.iter().filter(|(_, r, _)| *r == Reach::Fall).count();
    eprintln!("  of which fall-through (real mid-function mapping suspects) = {}", fall_unknowns);
    for (pc, r, word) in unknowns.iter().take(40) {
        let rc = match r {
            Reach::Seed => "seed",
            Reach::Fall => "FALL",
            Reach::Branch => "bra",
        };
        eprintln!("    {pc:#08x} [{rc}] word={word:#010x}  ({})", nearest_symbol(&proc.symbols, *pc));
    }
    if unknowns.len() > 40 {
        eprintln!("    ... ({} more)", unknowns.len() - 40);
    }
    eprintln!("--- Harvard/overlay split (inst-fetch vs data-load) mismatches = {} ---", harvard_mismatch);
}

/// M2c Phase 0 (iter18) DIAGNOSTIC: static DIRECT-call cross-reference.
/// Scans every symbol function for call-family instructions with an
/// immediate target (call0/call4/call8/call12 -- NOT register-indirect
/// callx*, whose target isn't statically known), and lists the call sites
/// targeting each address in a query set. Traces who reaches the
/// scheduler-region done-flag-setter candidates the store-search surfaced.
/// Query set = XDNA_FW_XREF (comma-separated hex) or a built-in default of
/// those candidates. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_call_xref() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the call-xref probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let targets: Vec<u32> = std::env::var("XDNA_FW_XREF")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_else(|| {
            // The store-search's scheduler-region candidates + the dispatcher.
            vec![0xd7f0, 0xc9dc, 0xd134, 0xd1e8, 0xd53c, 0xd84c, 0xe098]
        });
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
    syms.sort_by_key(|(a, _)| *a);

    // Collect all DIRECT (immediate-target) call edges: (call_site, target).
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..syms.len() {
        let start = syms[i].0;
        let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
        let mut pc = start;
        while pc < end {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
            let d = decode::decode(&b, pc);
            let target = match d.op {
                decode::Op::Call0 { target }
                | decode::Op::Call4 { target }
                | decode::Op::Call8 { target }
                | decode::Op::Call12 { target } => Some(target),
                _ => None,
            };
            if let Some(t) = target {
                edges.push((pc, t));
            }
            pc += (d.len as u32).max(1);
        }
    }

    eprintln!("=== M2c call-xref: DIRECT callers of {} target(s) ===", targets.len());
    eprintln!("(register-indirect callx* callers are NOT shown -- target unknown statically)");
    for tgt in targets {
        let tname = nearest_symbol(&proc.symbols, tgt);
        eprintln!("target {tgt:#08x}  {tname}:");
        let mut any = false;
        for (site, _) in edges.iter().filter(|(_, t)| *t == tgt) {
            eprintln!("    called from {site:#08x}  {}", nearest_symbol(&proc.symbols, *site));
            any = true;
        }
        if !any {
            eprintln!("    (no DIRECT callers -- reached only via callx*/table or fall-through)");
        }
    }
}

/// M2c iter18 (Session-4) DIAGNOSTIC: static L32r-literal cross-reference.
/// The AIE-completion ISR is the last uncharted link in the completion path
/// (it reads the AIE interrupt-status register and posts an event message
/// that `wake_tasks_by_event_mask` turns into a pending-mask bit). It can't
/// be found dynamically -- no interrupt fires in EMU without an array model.
/// So scan every symbol function for `l32r` instructions whose resolved
/// literal VALUE falls in a target range (the AIE mgmt register aperture,
/// default `0x2701_0000..0x2701_2000`), and report each load site + the
/// literal value + the containing function. Whoever loads the int-status
/// constant OUTSIDE `sched_event_poll` is the ISR (or its status helper).
/// Range overridable via `XDNA_FW_LIT_LO`/`XDNA_FW_LIT_HI` (hex). Ignored
/// unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_literal_xref() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the literal-xref probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let parse_hex = |k: &str, d: u32| -> u32 {
        std::env::var(k)
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(d)
    };
    let lo = parse_hex("XDNA_FW_LIT_LO", 0x2701_0000);
    let hi = parse_hex("XDNA_FW_LIT_HI", 0x2701_2000);
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
    syms.sort_by_key(|(a, _)| *a);

    eprintln!("=== M2c L32r-literal xref: value in {lo:#x}..{hi:#x} ===");
    let mut hits = 0u64;
    for i in 0..syms.len() {
        let start = syms[i].0;
        let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
        let fname = syms[i].1.clone();
        let mut pc = start;
        while pc < end {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
            let d = decode::decode(&b, pc);
            if let decode::Op::L32r { target, .. } = d.op {
                let lit: [u8; 4] =
                    std::array::from_fn(|k| proc.bus.fetch8(target + k as u32, target + k as u32));
                let val = u32::from_le_bytes(lit);
                if val >= lo && val < hi {
                    eprintln!("  {pc:#08x} {fname:<26} l32r -> lit@{target:#x} = {val:#010x}");
                    hits += 1;
                }
            }
            pc += (d.len as u32).max(1);
        }
    }
    eprintln!("total hits = {hits}");
}

/// M2c Phase 2 DIAGNOSTIC (iter13): does the firmware COPY code into the low
/// window as DATA (which our Harvard model routes to `local_data`), then FETCH
/// it (which reads the pristine image)? The syscall cause-handler target
/// (0xe1fc) reads as zeros in the image; if `local_data` holds real code there,
/// the low window is unified IRAM/DRAM at that address and our fetch/data split
/// is wrong for it (iter12's deferred "fork (b)"). Boots to the wall (init runs
/// first) and dumps image vs `local_data` at XDNA_FW_DUMP_ADDR (default 0xe1fc).
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_low_window_code() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the low-window code probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MAX: u64 = 200_000;
    let mut n = 0u64;
    let mut syscall_ps: Option<u32> = None;
    while n < MAX {
        let pc = proc.cpu.pc;
        // Capture PS at the boot syscall (the MERT hand-off): PS.UM (bit 5)
        // decides user- vs kernel-mode exception routing.
        if pc == 0x08b0_43e1 && syscall_ps.is_none() {
            syscall_ps = Some(proc.cpu.regs.ps);
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => {
                n += 1;
                if proc.cpu.pc == pc {
                    break;
                }
            }
            Step::Unknown { .. } => break,
        }
        if proc.bus.sysstub().spinning().is_some() {
            break;
        }
    }
    match syscall_ps {
        Some(ps) => eprintln!(
            "syscall PS = {ps:#x}  INTLEVEL={} EXCM={} UM(user)={} RING={} WOE={}",
            ps & 0xf,
            (ps >> 4) & 1,
            (ps >> 5) & 1,
            (ps >> 6) & 3,
            (ps >> 18) & 1
        ),
        None => eprintln!("syscall PS = <syscall pc 0x8b043e1 not reached>"),
    }

    let base = std::env::var("XDNA_FW_DUMP_ADDR")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xe1fc);
    eprintln!("=== low-window handler-region dump @ {base:#x} (after {n} instrs) ===");
    eprintln!("(image = fetch source / IRAM; local_data = data-write overlay / DRAM)");
    let mut any_differ = false;
    for i in 0..16u32 {
        let a = base + i * 4;
        let img_w = proc.bus.inst_load32(a); // physical Rom path == image (fetch source)
        let loc_w = proc.bus.data_load32(a); // local_data overlay (data writes)
        let flag = if img_w != loc_w {
            any_differ = true;
            "   <-- DIFFER"
        } else {
            ""
        };
        eprintln!("  {a:#010x}: image={img_w:#010x}  local_data={loc_w:#010x}{flag}");
    }
    eprintln!(
        "verdict: {}",
        if any_differ {
            "local_data holds code the image lacks -> low window is unified IRAM/DRAM here (fork b)"
        } else {
            "image and local_data agree -> handler genuinely absent; not a fetch/data-split issue"
        }
    );
}

/// M2c Phase 2 DIAGNOSTIC: statically disassemble the low-ROM exception
/// vector entries via our own decoder (which, unlike lx106 objdump, handles
/// the windowed ops these vectors are built from) and resolve each `l32r`
/// literal so `jx`-stub targets are readable. This is the tool that derived
/// the corrected [`KERNEL_EXCEPTION_VECTOR_OFFSET`] (see the finding
/// `docs/superpowers/findings/2026-07-05-iter7-exception-vector-offset.md`).
///
/// The entries below were pinned from the firmware image (vecbase=0x800,
/// confirmed from the prologue's own `wsr.vecbase` literal):
/// - 0xae0 (vecbase+0x2e0): the Kernel/general-exception vector -- a stub
///   `wsr.excsave1 a3; l32r a3,=0x28b4; jx a3` that jumps to the real
///   exception dispatcher at runtime 0x28b4.
/// - 0xb1c (vecbase+0x31c): the DoubleException handler -- inline
///   `wsr.excsave1/2/5/6; rsr.exccause; ...; rfde` (surfaces as `Unknown`
///   at the `rfde`, which our windowed-firmware decoder doesn't carry).
///
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_vector_table() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the vector-table probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let proc = FirmwareProcessor::load_m2c(img);

    const VECBASE: u32 = 0x800;
    const MAX_INSTRS: usize = 24;

    let peek3 = |phys: u32| {
        [proc.bus.peek8(phys), proc.bus.peek8(phys.wrapping_add(1)), proc.bus.peek8(phys.wrapping_add(2))]
    };
    let peek32 = |phys: u32| {
        u32::from_le_bytes([
            proc.bus.peek8(phys),
            proc.bus.peek8(phys.wrapping_add(1)),
            proc.bus.peek8(phys.wrapping_add(2)),
            proc.bus.peek8(phys.wrapping_add(3)),
        ])
    };

    eprintln!("=== M2c vector-table static disasm (vecbase {VECBASE:#x}) ===");
    // (entry, label). Extend when characterizing more of the vector table.
    for &(entry, label) in &[(0xae0u32, "kernel/general exc stub"), (0xb1c, "double exc handler")] {
        eprintln!("--- entry {entry:#x} (vecbase+{:#x}) -- {label} ---", entry - VECBASE);
        let mut pc = entry;
        for _ in 0..MAX_INSTRS {
            let b = peek3(pc);
            let d = decode::decode(&b, pc);
            // Resolve an L32r's literal address + value so `jx`-stub targets
            // are readable (l32r literal = ((pc+3)&~3) + sext(imm16)<<2).
            let extra = if let Op::L32r { target, .. } = d.op {
                format!("  [lit@{target:#x} = {:#x}]", peek32(target))
            } else {
                String::new()
            };
            eprintln!("  {pc:#06x}: {:02x} {:02x} {:02x}   {:?}{extra}", b[0], b[1], b[2], d.op);
            let terminal =
                matches!(d.op, Op::Jx { .. } | Op::RetN | Op::Retw | Op::RetwN | Op::Unknown { .. });
            if terminal {
                break;
            }
            pc = pc.wrapping_add(d.len as u32);
        }
    }
}
