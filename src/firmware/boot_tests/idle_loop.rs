use super::*;

/// TAIL-POLL characterization (iter43/iter44, 2026-07-10): after the iter42
/// 0x2450 fix the boot no longer walls -- it advances into a coherent scheduler
/// steady-loop on the first real task (0x10dfc), dispatcher-dominated, no
/// fault/idle/go-alive. This probe characterizes that loop: over a tail window
/// [lo,hi) it histograms every data-load EA (EXTERNAL HW-aperture reads
/// >=0x2500_0000 vs internal RAM) WITH per-site pc+value, the syscall selector
/// the task re-issues each period, and a dynamic instruction ring of the last K
/// steps. iter44 conclusion: the task loops on void syscall 0x6c, whose kernel
/// dispatch scans an EMPTY external completion ring (head/tail at 0x27200330/
/// 0x2720032c = 0) driven by the configured active-set [0x272003b8]=0x8000 --
/// the firmware is idling on an empty ring, waiting for host/array events we do
/// not supply. Env: XDNA_FW_MAX (default 200000), XDNA_FW_WIN=lo:hi (default
/// 80000:200000), XDNA_FW_RING (default 60). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_tail_poll() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the tail-poll probe");
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
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let (win_lo, win_hi) = std::env::var("XDNA_FW_WIN")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once(':')?;
            Some((a.trim().parse::<u64>().ok()?, b.trim().parse::<u64>().ok()?))
        })
        .unwrap_or((80_000, 200_000));
    let ring_depth: usize = std::env::var("XDNA_FW_RING").ok().and_then(|s| s.parse().ok()).unwrap_or(60);

    const EXT: u32 = 0x2500_0000; // HW-aperture threshold (doorbell/mailbox/columns)
    let mut ext: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    // Per external read site: (pc, addr) -> (count, sample loaded value).
    let mut ext_sites: std::collections::BTreeMap<(u32, u32), (u64, u32)> = std::collections::BTreeMap::new();
    let mut int: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut ring: std::collections::VecDeque<(u64, u32, Op)> = std::collections::VecDeque::new();
    // Syscall dispatcher (FUN_0000dab0): a4 = [a2] is the syscall selector the
    // task issued (the k/m/l/p = 0x6b/0x6d/0x6c/0x70 comparison tree), live at
    // 0xdae4; a2 is the on-stack syscall arg block. Pinned selector => the task
    // re-issues the same call forever (iter44: 0x6c, the empty-ring event scan);
    // cycling => it walks through calls.
    const STATE_PC: u32 = 0xdae4;
    let mut state_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut state_seq: Vec<(u64, u32)> = Vec::new();

    let mut n = 0u64;
    while n < max {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        if pc == STATE_PC {
            let s = proc.cpu.regs.read_ar(4);
            let addr = proc.cpu.regs.read_ar(2); // state var address = [a2]
            *state_hist.entry(s).or_insert(0) += 1;
            if state_seq.last().map(|&(_, v)| v) != Some(s) {
                state_seq.push((n, s));
            }
            if state_seq.len() <= 6 {
                eprintln!("    [syscall@dae4] n={n} argblock={addr:#x} selector={s:#x}");
            }
        }
        let in_win = n >= win_lo && n < win_hi;
        if in_win {
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                let op = decode::decode(&b, proc.cpu.pc).op;
                // Data-load EA = AR[s] + imm (L32r targets are absolute literals).
                let ea = match op {
                    Op::L32i { s, imm, .. }
                    | Op::L32iN { s, imm, .. }
                    | Op::L8ui { s, imm, .. }
                    | Op::L16ui { s, imm, .. }
                    | Op::L16si { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                    _ => None,
                };
                if let Some(addr) = ea {
                    if addr >= EXT {
                        *ext.entry(addr).or_insert(0) += 1;
                        let e = ext_sites.entry((pc, addr)).or_insert((0, 0));
                        e.0 += 1;
                        e.1 = proc.bus.data_load32(addr); // value the fw is about to read
                    } else {
                        *int.entry(addr).or_insert(0) += 1;
                    }
                }
                ring.push_back((n, pc, op));
                if ring.len() > ring_depth {
                    ring.pop_front();
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                eprintln!("  IDLE Wait({reason:?}) at n={n} pc={pc:#x}");
                n += 1;
                break;
            }
            Step::Unknown { pc: upc, word } => {
                eprintln!("  Unknown at pc={upc:#x} word={word:#010x}");
                break;
            }
        }
        proc.host_mailbox.tick(&mut proc.bus);
    }

    eprintln!("=== tail-poll probe (natural+agent, n={n}, win=[{win_lo},{win_hi})) ===");
    let dump = |label: &str, m: &std::collections::BTreeMap<u32, u64>| {
        let mut v: Vec<(u32, u64)> = m.iter().map(|(&a, &c)| (a, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u64 = v.iter().map(|&(_, c)| c).sum();
        eprintln!("--- {label} load EAs (top 16, total {total}) ---");
        for (addr, cnt) in v.into_iter().take(16) {
            eprintln!("  {cnt:>8}  {addr:#010x}");
        }
    };
    dump("EXTERNAL (>=0x25000000)", &ext);
    eprintln!("--- EXTERNAL read SITES (pc, addr) -> count, sample value ---");
    let mut es: Vec<((u32, u32), (u64, u32))> = ext_sites.into_iter().collect();
    es.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for ((pc, addr), (cnt, val)) in es.into_iter().take(20) {
        eprintln!(
            "  {cnt:>8}  pc={pc:#08x} {:<20} [{addr:#010x}] = {val:#010x}",
            nearest_symbol(&proc.symbols, pc)
        );
    }
    dump("INTERNAL", &int);
    eprintln!("--- FUN_0000dab0 state-code histogram ({} distinct) ---", state_hist.len());
    let mut sv: Vec<(u32, u64)> = state_hist.iter().map(|(&a, &c)| (a, c)).collect();
    sv.sort_by(|a, b| b.1.cmp(&a.1));
    for (val, cnt) in sv.into_iter().take(16) {
        eprintln!("  {cnt:>8}  state={val:#x}");
    }
    eprintln!("--- state transition sequence (first 40 distinct-value changes) ---");
    for (nn, v) in state_seq.iter().take(40) {
        eprintln!("  n={nn:>8} state={v:#x}");
    }
    eprintln!("--- dynamic instruction ring (last {} steps) ---", ring.len());
    for (nn, pc, op) in &ring {
        eprintln!("  n={nn:>8} pc={pc:#08x} {:<22} {op:x?}", nearest_symbol(&proc.symbols, *pc));
    }
}

/// task-0x10f10 identity probe (2026-07-09): WHAT is the current task the
/// dispatcher spins on, and what is it blocked waiting for?  Boots to a
/// pre-corruption steady state (default n=50000), dumps the task struct at
/// `XDNA_FW_TASK` (default 0x10f10) word-by-word with pointer annotation, and
/// the go-alive record 0x2320 for comparison. Then, over a full boot, counts how
/// many times the task's run-fn (field[0]) actually EXECUTES, and records the
/// task's state byte (+0x2c), block/wait field (+0x1b), done-flag (+0x30), and
/// link (+0x38) transitions -- so we can see if the task ever runs, and what
/// gates it. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_task_struct() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the task-struct probe");
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
    let task: u32 = std::env::var("XDNA_FW_TASK")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x10f10);
    let snap: u64 = std::env::var("XDNA_FW_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let total: u64 = std::env::var("XDNA_FW_DUMP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);

    let mut n = 0u64;
    while n < snap {
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        proc.host_mailbox.tick(&mut proc.bus);
    }

    let known: [(u32, &str); 3] = [(task, "self"), (0x2320, "goalive-rec"), (0x2250, "SCHED")];
    let annotate = |v: u32| -> String {
        for (a, name) in &known {
            if v >= *a && v < *a + 0x60 {
                return format!(" <- {name}+{:#x}", v - *a);
            }
        }
        // code region?
        if (0x400..0x30000).contains(&v) {
            return format!(" <- code? {}", nearest_symbol(&proc.symbols, v));
        }
        String::new()
    };
    eprintln!("=== task struct dump @ n={snap} (task {task:#x}) ===");
    for i in 0..0x18u32 {
        let a = task + i * 4;
        let v = proc.bus.data_load32(a);
        eprintln!("  [{a:#07x}] +{:#04x} = {v:#010x}{}", i * 4, annotate(v));
    }
    let runfn = proc.bus.data_load32(task) & 0x00ff_ffff;
    eprintln!("--- go-alive record 0x2320 (comparison) ---");
    for i in 0..6u32 {
        let a = 0x2320 + i * 4;
        let v = proc.bus.data_load32(a);
        eprintln!("  [{a:#07x}] +{:#04x} = {v:#010x}{}", i * 4, annotate(v));
    }

    // Continue to `total`, counting run-fn executions + field transitions.
    let mut runfn_hits = 0u64;
    let mut state_hist: Vec<(u64, u8)> = Vec::new();
    let mut done_hist: Vec<(u64, u32)> = Vec::new();
    while n < total {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        if pc == runfn {
            runfn_hits += 1;
        }
        let st = proc.bus.data_load8(task + 0x2c);
        if state_hist.last().map(|&(_, s)| s) != Some(st) {
            state_hist.push((n, st));
        }
        let df = proc.bus.data_load32(task + 0x30);
        if done_hist.last().map(|&(_, v)| v) != Some(df) {
            done_hist.push((n, df));
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        proc.host_mailbox.tick(&mut proc.bus);
    }
    eprintln!("--- over boot to n={total} ---");
    eprintln!("  run-fn {runfn:#x} ({}) executed {runfn_hits} times", nearest_symbol(&proc.symbols, runfn));
    eprintln!("  state byte [+0x2c] transitions (n,val): {state_hist:x?}");
    eprintln!("  done-flag [+0x30] transitions (n,val): {done_hist:x?}");
}

/// goalive-SPIN characterization (2026-07-09, from scratch): what loop is boot
/// actually stuck in, and what does it poll?  Runs to a steady-state `start` n,
/// then records the exact repeating instruction cycle -- traces forward until the
/// PC seen at `start` recurs at the same SP (one period) -- annotating every load
/// with its effective address + value and every conditional branch with
/// taken/not-taken.  Reports the cycle's PC span, its length, the distinct
/// addresses it polls (with values), and the branch that decides the loop. No
/// assumptions about "retire gate" or "worker 0x9040". `XDNA_FW_START` (default
/// 200000) sets the steady-state entry; `XDNA_FW_PERIOD_CAP` (default 4000) caps
/// the traced period. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_goalive_spin() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the goalive-spin probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    proc.enable_host_mailbox(); // bit3 kernel: study the demoted-but-present state
    let start: u64 = std::env::var("XDNA_FW_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let cap: usize = std::env::var("XDNA_FW_PERIOD_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);

    // The recursion never repeats SP (it descends 144B/pass), so anchor on a
    // PC (the dispatcher entry by default) and detect PC-only recurrence.
    let anchor_pc: u32 = std::env::var("XDNA_FW_ANCHOR_PC")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xd7f0);

    // Advance to steady state, then to the first anchor-PC hit at/after `start`.
    let mut n = 0u64;
    while n < start || (proc.cpu.pc & 0x00ff_ffff) != anchor_pc {
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        proc.host_mailbox.tick(&mut proc.bus);
        if n > start + 200_000 {
            break; // anchor PC not reached -- bail
        }
    }
    let anchor_n = n;
    let anchor_sp = proc.cpu.regs.read_ar(1);
    let mut lines: Vec<String> = Vec::new();
    let mut poll: std::collections::BTreeMap<u32, (u64, u32)> = std::collections::BTreeMap::new(); // addr->(count,last)
    let mut pc_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut steps = 0usize;
    loop {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let sp = proc.cpu.regs.read_ar(1);
        *pc_hist.entry(pc).or_insert(0) += 1;
        let mut note = String::new();
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] =
                std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
            let d = decode::decode(&b, proc.cpu.pc);
            match d.op {
                Op::L8ui { s, imm, .. } => {
                    let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    let v = proc.bus.data_load8(a) as u32;
                    note = format!("  L8 [{a:#x}]={v:#x}");
                    let e = poll.entry(a).or_insert((0, 0));
                    e.0 += 1;
                    e.1 = v;
                }
                Op::L32i { s, imm, .. } | Op::L32iN { s, imm, .. } => {
                    let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    let v = proc.bus.data_load32(a);
                    note = format!("  L32 [{a:#x}]={v:#x}");
                    let e = poll.entry(a).or_insert((0, 0));
                    e.0 += 1;
                    e.1 = v;
                }
                _ => {}
            }
            if steps < 400 {
                lines.push(format!("  {pc:#08x} sp={sp:#x} {:<26}{note}", format!("{:?}", d.op)));
            }
        }
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
        proc.host_mailbox.tick(&mut proc.bus);
        steps += 1;
        let now_pc = proc.cpu.pc & 0x00ff_ffff;
        if now_pc == anchor_pc && steps > 1 {
            break; // anchor PC recurred -- one full dispatcher cycle
        }
        if steps >= cap {
            lines.push(format!("  ... period exceeded cap={cap} (anchor PC {anchor_pc:#x} never recurred)"));
            break;
        }
    }
    let end_sp = proc.cpu.regs.read_ar(1);

    eprintln!("=== goalive-spin: dispatcher cycle at n={anchor_n} (anchor pc={anchor_pc:#x}) ===");
    eprintln!(
        "anchor sp={anchor_sp:#x} -> end sp={end_sp:#x} (delta={} bytes)",
        anchor_sp as i64 - end_sp as i64
    );
    eprintln!("period length = {steps} steps; distinct PCs in period = {}", pc_hist.len());
    eprintln!("--- polled addresses in the period (addr -> count, last-val) ---");
    for (a, (c, v)) in &poll {
        eprintln!("  {a:#010x}  count={c:>4} last={v:#x}");
    }
    eprintln!("--- first {} instrs of the period ---", lines.len().min(400));
    for l in &lines {
        eprintln!("{l}");
    }
}

/// M2c #140 CYCLE-ACCOUNTING: bucket every executed instruction by the
/// routine (nearest symbol range) its PC falls in, so every executed cycle
/// lands in exactly one named bucket. This is the "where do the cycles go"
/// instrument for decomposing the fitted 8000-cycle `dma_wait` mailbox
/// constant (`src/npu/executor.rs` `DEFAULT_MAILBOX_CYCLES`) into a real
/// mgmt-firmware instruction budget.
///
/// Grounding: on NPU1 (single Xtensa mgmt processor, NO per-column uC -- the
/// per-column TCT machinery is VE2/Versal), the mgmt firmware is the only
/// processor that consumes a shim-DMA TCT and unblocks a `dma_wait`. So its
/// wait -> notice -> wake -> dispatch instruction cost IS the firmware share
/// of the 8000. The array->completion propagation is the remaining share and
/// is modeled by the emulator's DMA/stream engine, not here.
///
/// Modes (env):
///   (default)              run to the boot wall; report the full per-routine
///                          histogram, and the dispatcher SPIN PERIOD (instrs
///                          per completion re-check) = the firmware's
///                          completion-poll granularity at the scheduler.
///   XDNA_FW_FORCE_DONE=1   force `[task+0x30]=1` at the dispatcher done-flag
///                          check (0xd828) so the completion "fires"; report
///                          where the downstream wake/dispatch/resume cycles
///                          go by routine.
/// XDNA_FW_MAX=<n> overrides the instruction budget (default 120000).
/// Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_cycle_accounting() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the cycle-accounting probe");
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
        .unwrap_or(120_000);
    let force = std::env::var("XDNA_FW_FORCE_DONE").is_ok();
    const DONE_CHECK_PC: u32 = 0xd828; // dispatcher `l32i.n a10,[a4+0x30]`

    // Per-routine instruction counts: bucket entry addr -> (label, count).
    let mut hist: std::collections::HashMap<u32, (String, u64)> = std::collections::HashMap::new();
    // Ring of recent (n, pc) for steady-state loop-period detection.
    const TAIL: usize = 4096;
    let mut tail: std::collections::VecDeque<(u64, u32)> =
        std::collections::VecDeque::with_capacity(TAIL + 1);
    let mut n = 0u64;
    let mut forces = 0u64;
    let mut stop = String::from("budget reached");

    // Full-run completion-check periods: for each anchor, the median
    // instruction gap between successive visits across the WHOLE run (the
    // 4096-instr tail window is too short for the coarse event-dispatcher
    // period). Anchors: done-flag check, event dispatcher, dispatcher entry,
    // pending-event delivery.
    const ANCHORS: [u32; 4] = [0xd828, 0x5580, 0xd7f0, 0xcadc];
    let mut anchor_last: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut anchor_gaps: std::collections::HashMap<u32, Vec<u64>> = std::collections::HashMap::new();

    while n < max {
        let pc = proc.cpu.pc;
        if ANCHORS.contains(&pc) {
            if let Some(&last) = anchor_last.get(&pc) {
                anchor_gaps.entry(pc).or_default().push(n - last);
            }
            anchor_last.insert(pc, n);
        }
        if force && pc == DONE_CHECK_PC {
            let done_addr = proc.cpu.regs.read_ar(4).wrapping_add(0x30);
            let _ = proc.cpu.data_write32(&mut proc.bus, done_addr, 1);
            forces += 1;
        }
        let (bucket, label) = routine_bucket(&proc.symbols, pc);
        hist.entry(bucket).or_insert((label, 0)).1 += 1;
        if tail.len() == TAIL {
            tail.pop_front();
        }
        tail.push_back((n, pc));
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
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }

    eprintln!("=== M2c cycle-accounting (#140 dma_wait 8000cy decomposition) ===");
    eprintln!(
        "mode            = {}",
        if force {
            "FORCE_DONE continuation"
        } else {
            "wall spin"
        }
    );
    eprintln!("instrs executed = {n}");
    if force {
        eprintln!("forced done-flag= {forces} time(s)");
    }
    eprintln!("stop reason     = {stop}");
    eprintln!("last pc         = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));

    // Per-routine histogram, sorted by instruction count descending. Every
    // executed cycle is in exactly one row -> the rows sum to `n`.
    let mut rows: Vec<(u32, String, u64)> = hist.into_iter().map(|(a, (l, c))| (a, l, c)).collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    eprintln!("--- per-routine instruction budget (every cycle accounted) ---");
    let show = 24.min(rows.len());
    let mut shown = 0u64;
    for (addr, label, count) in rows.iter().take(show) {
        let pct = 100.0 * *count as f64 / n.max(1) as f64;
        eprintln!("  {count:>9} ({pct:>5.1}%)  {addr:#08x} {label}");
        shown += *count;
    }
    if rows.len() > show {
        let rest = n - shown;
        eprintln!(
            "  {rest:>9} ({:>5.1}%)  (+{} more routines)",
            100.0 * rest as f64 / n.max(1) as f64,
            rows.len() - show
        );
    }

    // Full-run completion-check periods (median gap between anchor visits).
    eprintln!("--- completion-check periods (full run, median instrs between visits) ---");
    for anchor in ANCHORS {
        match anchor_gaps.get(&anchor) {
            Some(gaps) if gaps.len() >= 2 => {
                let mut g = gaps.clone();
                g.sort_unstable();
                let med = g[g.len() / 2];
                eprintln!(
                    "  {anchor:#08x} {:<24} period ~ {med:>6} instrs  ({} visits)",
                    nearest_symbol(&proc.symbols, anchor),
                    gaps.len() + 1
                );
            }
            _ => eprintln!(
                "  {anchor:#08x} {:<24} <2 visits (essentially never reached)",
                nearest_symbol(&proc.symbols, anchor)
            ),
        }
    }

    // Completion-check granularity: the median instruction gap between
    // successive visits to each anchor over the tail window. The relevant
    // quantum is NOT the scheduler's inner bit-scan loop but the period
    // between successive COMPLETION CHECKS -- the dispatcher's done-flag read
    // (0xd828) and the event dispatcher entry (0x5580). That bounds how fast
    // the firmware can notice a `dma_wait` TCT once the array signals it.
    if tail.len() > 16 {
        let period_of = |anchor: u32| -> Option<(u64, u64)> {
            let visits: Vec<u64> = tail.iter().filter(|(_, pc)| *pc == anchor).map(|(n, _)| *n).collect();
            if visits.len() < 3 {
                return None;
            }
            let mut gaps: Vec<u64> = visits.windows(2).map(|w| w[1] - w[0]).collect();
            gaps.sort_unstable();
            Some((gaps[gaps.len() / 2], visits.len() as u64))
        };
        let mut freq: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        for (_, pc) in &tail {
            *freq.entry(*pc).or_insert(0) += 1;
        }
        let mode_pc = freq.iter().max_by_key(|(_, c)| **c).map(|(&pc, _)| pc).unwrap_or(0);
        let distinct: std::collections::BTreeSet<u32> = tail.iter().map(|(_, pc)| *pc).collect();
        eprintln!(
            "--- completion-check granularity (tail {} instrs, {} distinct PCs) ---",
            tail.len(),
            distinct.len()
        );
        // Named completion-check anchors + the hottest inner PC, plus any
        // caller-supplied anchor via XDNA_FW_ANCHOR_PC=<hex>.
        let extra = std::env::var("XDNA_FW_ANCHOR_PC")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        let anchors: Vec<(u32, &str)> = vec![
            (0xd828, "dispatcher done-flag check l32i.n [task+0x30]"),
            (0x5580, "event dispatcher entry (-> wake_tasks_by_event_mask)"),
            (mode_pc, "hottest inner PC (scheduler bit-scan)"),
            (extra.unwrap_or(0), "XDNA_FW_ANCHOR_PC"),
        ];
        for (anchor, what) in anchors {
            if anchor == 0 {
                continue;
            }
            match period_of(anchor) {
                Some((period, visits)) => eprintln!(
                    "  {anchor:#08x} {:<20} period = {period:>6} instrs  ({visits} visits)  {what}",
                    nearest_symbol(&proc.symbols, anchor)
                ),
                None => eprintln!(
                    "  {anchor:#08x} {:<20} NOT in tail window (<3 visits)  {what}",
                    nearest_symbol(&proc.symbols, anchor)
                ),
            }
        }
    }
}

/// M2c iter18 RE TOOL (per-task completion RE): map each blocked task to the
/// run-function the dispatcher calls for it, and capture what that run-fn
/// TOUCHES (its load/store effective addresses) -- the poll/request sites that
/// reveal what each task is waiting on. The dispatcher (`0xd7f0..0xd848`) does
/// `callx8 a3` (a3 = the current task's run-fn) when the done-flag is 0. This
/// records every distinct `(current_task, call_target, call_pc)` for calls
/// made from inside the dispatcher, plus, for the first execution of each
/// distinct run-fn, the set of distinct data effective-addresses it accesses
/// (region-tagged) over a bounded instruction window. Ignored unless
/// XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_task_runfns() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the task-runfn probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MAX: u64 = 1_500_000;
    const SCHED: u32 = 0x2278;
    const DISP_LO: u32 = 0xd7f0;
    const DISP_HI: u32 = 0xd848;
    let region_name = |a: u32| -> &'static str {
        match a {
            _ if a < 0x0400_0000 => "LOCAL",
            _ if a < 0x0800_0000 => "ARRAY",
            _ if (0x0800_0000..0x08b0_0000).contains(&a) => "GAP",
            _ if (0x08b0_0000..0x2700_0000).contains(&a) => "RAM",
            _ if (0x2700_0000..0x2800_0000).contains(&a) => "MAILBOX",
            _ => "SYSTEM",
        }
    };
    // (task, run_fn target, call_pc) triples observed at dispatcher calls.
    let mut pairs: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
    // For each distinct run_fn target, the distinct data EAs it accessed
    // (only the run-fn currently being traced, bounded).
    let mut runfn_eas: std::collections::BTreeMap<u32, std::collections::BTreeSet<u32>> =
        std::collections::BTreeMap::new();
    // Trace window: when we enter a run_fn we haven't fully profiled, record
    // its data EAs for up to TRACE_WIN instrs (following calls out of it too).
    const TRACE_WIN: u32 = 4000;
    let mut tracing: Option<(u32, u32)> = None; // (run_fn, instrs_left)

    let mut n = 0u64;
    let mut stop = String::from("budget");
    while n < MAX {
        let pc = proc.cpu.pc;
        let cur = proc.cpu.data_read32(&mut proc.bus, SCHED).unwrap_or(0);
        let decoded = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            Ok(phys) => {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                Some(decode::decode(&b, pc))
            }
            Err(_) => None,
        };
        if let Some(d) = &decoded {
            // Dispatcher-made call -> the run-fn (or scheduler helper) target.
            if (DISP_LO..DISP_HI).contains(&pc) {
                let target = match d.op {
                    Op::Call4 { target } | Op::Call8 { target } | Op::Call12 { target } => Some(target),
                    Op::Callx4 { s } | Op::Callx8 { s } | Op::Callx12 { s } => Some(proc.cpu.regs.read_ar(s)),
                    _ => None,
                };
                if let Some(t) = target {
                    pairs.insert((cur, t, pc));
                    // Begin tracing this run-fn's data accesses if new.
                    if !runfn_eas.contains_key(&t) {
                        runfn_eas.insert(t, std::collections::BTreeSet::new());
                        tracing = Some((t, TRACE_WIN));
                    }
                }
            }
            // While tracing, record data EAs of loads/stores.
            if let Some((rf, left)) = tracing {
                let ea = match d.op {
                    Op::L32i { s, imm, .. }
                    | Op::L32iN { s, imm, .. }
                    | Op::L8ui { s, imm, .. }
                    | Op::L16ui { s, imm, .. }
                    | Op::L16si { s, imm, .. }
                    | Op::S32i { s, imm, .. }
                    | Op::S32iN { s, imm, .. }
                    | Op::S8i { s, imm, .. }
                    | Op::S16i { s, imm, .. }
                    | Op::S32ri { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                    _ => None,
                };
                if let Some(ea) = ea {
                    runfn_eas.get_mut(&rf).unwrap().insert(ea);
                }
                tracing = if left <= 1 { None } else { Some((rf, left - 1)) };
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(r) => {
                n += 1;
                stop = format!("Wait({r:?})");
                break;
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown at {upc:#x} word={word:#010x}");
                break;
            }
        }
    }
    eprintln!("=== M2c task run-fn map ===");
    eprintln!("instrs = {n}; stop = {stop}");
    eprintln!("--- (task, run_fn, call_pc) at dispatcher calls ---");
    for (task, rf, cpc) in &pairs {
        eprintln!("  task={task:#x}  run_fn={rf:#x}  call_pc={cpc:#x}");
    }
    eprintln!("--- distinct data EAs each run_fn touched (first {TRACE_WIN} instrs of its first run) ---");
    for (rf, eas) in &runfn_eas {
        let tagged: Vec<String> = eas.iter().map(|a| format!("{a:#x}({})", region_name(*a))).collect();
        eprintln!("  run_fn={rf:#x}: {}", tagged.join(" "));
    }
}

/// M2c iter18 RE TOOL: enumerate every load site the stuck boot spins on.
/// Widen-before-deepen: rather than guess the next gate, run into steady
/// state then, over a window of the recursion cycle, compute the effective
/// address of every load (EA = AR[s] + imm; imm is byte-scaled) and count
/// repetitions. Addresses read many times -- especially in the mailbox
/// (0x2700_0000..0x2800_0000) or system apertures -- are the poll/wait sites.
/// Reports top load addresses by count with region tag and a best-effort
/// peeked value. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_poll_map() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the poll-map");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // WARMUP skips the transient to reach steady state; WINDOW records over
    // many cycles. Override via XDNA_FW_POLL_WARMUP / XDNA_FW_POLL_WINDOW to
    // capture the EARLY phase (e.g. WARMUP=0) instead of the steady spin.
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let warmup = env_u64("XDNA_FW_POLL_WARMUP", 300_000);
    let window = env_u64("XDNA_FW_POLL_WINDOW", 200_000);
    // addr -> (count, causing-pc)
    let mut hits: std::collections::HashMap<u32, (u64, u32)> = std::collections::HashMap::new();
    let mut n = 0u64;
    let region = |a: u32| -> &'static str {
        if (0x2700_0000..0x2800_0000).contains(&a) {
            "MBOX"
        } else if (0x0800_0000..0x2700_0000).contains(&a) {
            "RAM"
        } else if a < 0x0080_0000 {
            "IMG/LO"
        } else if (0x3c00_0000..0x3d00_0000).contains(&a) {
            "PGTBL"
        } else {
            "SYS"
        }
    };
    while n < warmup + window {
        let pc = proc.cpu.pc;
        if n >= warmup {
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let op = decode::decode(&b, pc).op;
                let ea = match op {
                    decode::Op::L32i { s, imm, .. }
                    | decode::Op::L32iN { s, imm, .. }
                    | decode::Op::L8ui { s, imm, .. }
                    | decode::Op::L16ui { s, imm, .. }
                    | decode::Op::L16si { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                    _ => None,
                };
                if let Some(ea) = ea {
                    let e = hits.entry(ea).or_insert((0, pc));
                    e.0 += 1;
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => break,
            Step::Unknown { .. } => break,
        }
        if proc.bus.sysstub().spinning().is_some() {
            break;
        }
    }
    eprintln!("=== M2c poll-map (loads over instrs [{warmup}, {}]) ===", warmup + window);
    let mut v: Vec<(u32, u64, u32)> = hits.iter().map(|(a, (c, pc))| (*a, *c, *pc)).collect();
    v.sort_by_key(|(_, c, _)| std::cmp::Reverse(*c));
    eprintln!("--- top 30 load addresses by repetition ---");
    eprintln!("   addr        region  count   from-pc(symbol)          peeked");
    for (a, c, pc) in v.iter().take(30) {
        let val = u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
        eprintln!(
            "  {a:#010x}  {:<6}  {c:>6}  {pc:#08x} {:<20}  {val:#010x}",
            region(*a),
            nearest_symbol(&proc.symbols, *pc)
        );
    }
    // Focus list: only mailbox/system aperture hits (host/hardware writable).
    eprintln!("--- mailbox/system aperture loads (candidate host handshakes) ---");
    for (a, c, pc) in v.iter().filter(|(a, _, _)| region(*a) == "MBOX" || region(*a) == "SYS") {
        let val = u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
        eprintln!("  {a:#010x}  {:<6}  {c:>6}  {pc:#08x}  {val:#010x}", region(*a));
    }
}

/// FAITHFUL-MODEL step 2 (2026-07-07, Maya "map the seam"): the event-wake
/// path is dormant during boot (all await-masks 0), so boot is POLL-gated. To
/// map the poll seam we need to know, in steady state, (a) where the PC lives
/// (which loop the fw is stuck in) and (b) whether that loop reads ANY external
/// MMIO address (>= 0x2000_0000) -- the register the fw polls for
/// column-power-ready that our shim must answer. This histograms PC by nearest
/// function over a deep window and tallies every distinct external-read EA with
/// hit counts + the reading PC. If the steady loop touches no external address,
/// the gate is internal scheduler state, not a hardware poll. Env:
/// XDNA_FW_HIST_WARMUP (default 200_000), XDNA_FW_HIST_COUNT (default
/// 2_000_000). Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_steady_histogram() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the steady-state histogram");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);
    let mut device = crate::device::DeviceState::new_npu1();
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let warmup = env_u64("XDNA_FW_HIST_WARMUP", 200_000);
    let count = env_u64("XDNA_FW_HIST_COUNT", 2_000_000);

    for _ in 0..warmup {
        if !matches!(
            proc.cpu.step_with_device(&mut proc.bus, &mut device),
            Step::Ran | Step::Exception { .. }
        ) {
            break;
        }
    }

    // PC histogram bucketed by nearest symbol, and distinct external-read EAs.
    let mut pc_hist: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    // ea -> (hits, first reading pc)
    let mut ext_reads: std::collections::BTreeMap<u32, (u64, u32)> = std::collections::BTreeMap::new();
    let mut stopped = String::from("count reached");
    for _ in 0..count {
        let pc = proc.cpu.pc;
        *pc_hist.entry(nearest_symbol(&proc.symbols, pc & 0x00ff_ffff)).or_insert(0) += 1;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let ea = match decode::decode(&b, pc).op {
                Op::L32i { s, imm, .. }
                | Op::L32iN { s, imm, .. }
                | Op::L8ui { s, imm, .. }
                | Op::L16ui { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                _ => None,
            };
            if let Some(ea) = ea {
                if ea >= 0x2000_0000 {
                    let e = ext_reads.entry(ea).or_insert((0, pc & 0x00ff_ffff));
                    e.0 += 1;
                }
            }
        }
        if !matches!(
            proc.cpu.step_with_device(&mut proc.bus, &mut device),
            Step::Ran | Step::Exception { .. }
        ) {
            stopped = format!("halted at pc={:#x}", proc.cpu.pc);
            break;
        }
    }

    eprintln!("=== steady-state histogram (warmup {warmup}, {count} samples) ===");
    eprintln!("stop = {stopped}");
    let mut ranked: Vec<_> = pc_hist.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("--- top PC buckets (by nearest function) ---");
    for (sym, hits) in ranked.iter().take(15) {
        eprintln!("  {hits:>9}  {sym}");
    }
    eprintln!("--- distinct external-read EAs (>= 0x2000_0000) ---");
    if ext_reads.is_empty() {
        eprintln!("  (none -- steady state reads NO external MMIO; gate is internal state)");
    } else {
        for (ea, (hits, pc)) in &ext_reads {
            eprintln!("  {ea:#010x}  hits={hits:>8}  first-read-pc={pc:#x}");
        }
    }
}

/// ARC-2 STEP 1 (2026-07-10): fresh, untainted re-derivation of the go-alive
/// task record and its readiness. The prior (pre-iter42) scheduler map is
/// suspect, so this derives from live behavior only. Anchor: the store of the
/// run-fn pointer 0x55f8 into the task record IS the record's run-fn slot. This
/// probe (1) finds the first store of value 0x55f8, reports its EA and dumps the
/// surrounding 32-word record; (2) for the REST of the boot, watches every store
/// whose EA lands in that record window and reports it -- so we see whether any
/// field (state/ready/wait-mask) is ever mutated after creation. If the record
/// is written once and never touched again, the go-alive task is created-and-
/// abandoned (never readied); the mutations, if any, name the readiness field
/// and who writes it. Env: XDNA_FW_MAX (default 3_000_000), XDNA_FW_RUNFN
/// (default 0x55f8), XDNA_FW_RECWIN (record half-window in bytes, default 0x40).
/// Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_goalive_record() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the go-alive record probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let env_hex = |k: &str, d: u32| -> u32 {
        std::env::var(k)
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(d)
    };
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000_000);
    let runfn = env_hex("XDNA_FW_RUNFN", 0x55f8);
    let win = env_hex("XDNA_FW_RECWIN", 0x40);

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Store decoder shared by both phases.
    let decode_store = |proc: &mut FirmwareProcessor| -> Option<(u32, u32)> {
        let pc = proc.cpu.pc;
        let phys = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch).ok()?;
        let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
        let (t, s, imm) = match decode::decode(&b, pc).op {
            Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => (t, s, imm),
            _ => return None,
        };
        let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
        let val = proc.cpu.regs.read_ar(t);
        Some((ea, val))
    };

    // Phase 1: find the record via the run-fn store.
    let mut rec_base: Option<u32> = None;
    let mut n = 0u64;
    let mut create_n = 0u64;
    let mut create_pc = 0u32;
    while n < max {
        if let Some((ea, val)) = decode_store(&mut proc) {
            if val == runfn && rec_base.is_none() {
                rec_base = Some(ea);
                create_n = n;
                create_pc = proc.cpu.pc;
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        if rec_base.is_some() {
            break;
        }
    }

    eprintln!("=== ARC-2 go-alive record (fresh) ===");
    let Some(runfn_ea) = rec_base else {
        eprintln!("run-fn {runfn:#x} never stored in {n} instrs -- record not built");
        return;
    };
    eprintln!(
        "run-fn {runfn:#x} stored to [{runfn_ea:#x}] at n={create_n} pc={create_pc:#x}[{}]",
        nearest_symbol(&proc.symbols, create_pc)
    );
    let lo = runfn_ea.wrapping_sub(win);
    let hi = runfn_ea.wrapping_add(win);
    eprintln!("--- record window [{lo:#x}..{hi:#x}] just after creation ---");
    let mut a = lo;
    while a < hi {
        let v = proc.cpu.data_read32(&mut proc.bus, a).unwrap_or(0);
        let mark = if a == runfn_ea { "  <- run-fn" } else { "" };
        eprintln!("  [{a:#x}] = {v:#010x}{mark}");
        a = a.wrapping_add(4);
    }

    // Phase 2: watch every store into the record window for the rest of boot.
    eprintln!("--- mutations to the record window over the rest of boot ---");
    let mut mutations = 0u64;
    while n < max {
        if let Some((ea, val)) = decode_store(&mut proc) {
            if (lo..hi).contains(&ea) {
                mutations += 1;
                if mutations <= 60 {
                    let off = ea.wrapping_sub(runfn_ea) as i32;
                    eprintln!(
                        "  n={n:>8} pc={:#x}[{}]  [rec{off:+#x} = {ea:#x}] <- {val:#x}",
                        proc.cpu.pc,
                        nearest_symbol(&proc.symbols, proc.cpu.pc)
                    );
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
    }
    eprintln!("total mutations to record window after creation = {mutations}");
    if mutations == 0 {
        eprintln!("  => go-alive record is WRITE-ONCE: not mutated after creation");
    }
}

/// ARC-2 STEP 2 (2026-07-10): pointer-linked vs fixed-pool dispatch. The
/// go-alive job's record sits at `rec` (run-fn 0x55f8, default rec=0x2320). For a
/// dispatcher to ever run it, EITHER the job POINTER (`rec`) is linked into a
/// pointer queue OR a fixed-pool dispatcher indexes and READS the record. This
/// probe watches a full boot for both: (a) every store of VALUE == rec; (b) every
/// LOAD whose EA is in [rec, rec+0x14). Absence of a pointer store alone does
/// not mean "never enqueued": MERT uses count/index bytes beside its embedded
/// fixed record pool. Env: XDNA_FW_MAX (default
/// 3_000_000), XDNA_FW_REC (default 0x2320). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_goalive_dispatch() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the go-alive dispatch probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let env_hex = |k: &str, d: u32| -> u32 {
        std::env::var(k)
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(d)
    };
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000_000);
    let rec = env_hex("XDNA_FW_REC", 0x2320);

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    let mut enqueues: Vec<(u64, u32, u32)> = Vec::new(); // (n, pc, ea) store of VALUE==rec
    let mut reads: std::collections::BTreeMap<(u32, u32), u64> = std::collections::BTreeMap::new(); // (pc, ea) -> count
    let mut n = 0u64;
    while n < max {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            match decode::decode(&b, pc).op {
                Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => {
                    let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    let val = proc.cpu.regs.read_ar(t);
                    if val == rec {
                        enqueues.push((n, pc, ea));
                    }
                }
                Op::L32i { s, imm, .. }
                | Op::L32iN { s, imm, .. }
                | Op::L8ui { s, imm, .. }
                | Op::L16ui { s, imm, .. }
                | Op::L16si { s, imm, .. } => {
                    let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    if (rec..rec.wrapping_add(0x14)).contains(&ea) {
                        *reads.entry((pc, ea)).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
    }

    eprintln!("=== ARC-2 go-alive dispatch (pointer-linked vs fixed-pool), rec={rec:#x} ===");
    eprintln!("instrs = {n}");
    eprintln!("--- (a) stores of VALUE == {rec:#x} (job pointer enqueued/linked) ---");
    if enqueues.is_empty() {
        eprintln!("  NONE -- no pointer link; fixed-pool enqueue remains possible");
    } else {
        for (n, pc, ea) in enqueues.iter().take(40) {
            eprintln!("  n={n:>8} pc={pc:#x}[{}]  [{ea:#x}] <- {rec:#x}", nearest_symbol(&proc.symbols, *pc));
        }
    }
    eprintln!("--- (b) LOADS from the record [{rec:#x}..{:#x}) (a dispatcher inspecting it) ---", rec + 0x14);
    if reads.is_empty() {
        eprintln!("  NONE -- record not inspected in this observation window");
    } else {
        for ((pc, ea), cnt) in &reads {
            eprintln!("  pc={pc:#x}[{}]  load [{ea:#x}]  x{cnt}", nearest_symbol(&proc.symbols, *pc));
        }
    }
}

/// ARC-2 STEP 3 (2026-07-10): REPRODUCE Codex's +0x100 misframe finding. Claim:
/// the go-alive job IS enqueued (fixed-pool count byte at 0x24c4=1), but the MERT
/// queue-pop (VMA 0xcc1c, reached via 0xc648) and its literal pool 0x3c84 are
/// mis-framed at +0x5c: the pool-base literal reads 0x06194518 (garbage) instead
/// of 0x00002250 (SCHED base), so the count read at [base+0x274] hits unbacked
/// memory (0) and the pop takes the empty exit -- go-alive is never popped. Fix:
/// serve 0xc648/0xcc1c/0x3c84 at +0x100 (same overlay class as the iter42 seams;
/// one add_rom_overlay covers both fetch and the L32r literal read). This probe
/// installs those overlays post-load and checks (1) waypoint first-hit for run-fn
/// 0x55f8 and publisher 0x50e8, (2) a before/after magic scan for a runtime
/// chann_info publish. XDNA_FW_OVL=lo:hi,lo:hi (hex) adds EXTRA +0x100 overlays
/// (e.g. the 0x55f8 section for the full publish path). Env: XDNA_FW_MAX (default
/// 3_000_000). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_goalive_overlay_repro() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the go-alive overlay reproduction");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3_000_000);
    const OVL: u32 = 0x100; // +0x100 file delta (LOW_VMA_FILE_OFFSET)
    const MAGIC: [u8; 4] = [0x5f, 0x4e, 0x50, 0x55];

    // Base overlays proving the queue-pop path: launcher/work-fetch, pop, literal.
    let mut overlays: Vec<(u32, u32)> = vec![(0xc648, 0xc6b0), (0xcc1c, 0xccb4), (0x3c84, 0x3c88)];
    if let Ok(extra) = std::env::var("XDNA_FW_OVL") {
        for part in extra.split(',') {
            if let Some((a, b)) = part.split_once(':') {
                let lo = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).expect("ovl lo hex");
                let hi = u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).expect("ovl hi hex");
                overlays.push((lo, hi));
            }
        }
    }

    let raw = std::fs::read(&path).expect("read firmware");
    let build = |overlays: &[(u32, u32)]| -> FirmwareProcessor {
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        for &(lo, hi) in overlays {
            proc.bus.add_rom_overlay(lo, hi, OVL);
        }
        proc
    };

    // Static magic baseline (no overlays needed -- it's about backing stores).
    let baseline: std::collections::BTreeSet<(&'static str, u32)> =
        build(&[]).bus.scan_bytes(&MAGIC).into_iter().collect();

    let mut proc = build(&overlays);
    eprintln!("=== ARC-2 go-alive overlay reproduction ===");
    eprintln!("overlays (+0x100): {:x?}", overlays);

    // Symbol addresses are not necessarily instruction boundaries.  In this
    // image publish_chann_info is labelled 0x50e8, the second byte of the
    // coherent two-byte instruction at 0x50e7, so record the instruction that
    // covers each waypoint rather than requiring pc == symbol VMA.
    const WP: [u32; 2] = [0x55f8, 0x50e8]; // run-fn, publisher
    let mut first: std::collections::BTreeMap<u32, (u64, u32)> = std::collections::BTreeMap::new();
    let tail_depth: usize = std::env::var("XDNA_FW_TAIL").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let calls_only = std::env::var("XDNA_FW_CALLS").is_ok();
    let audit = std::env::var("XDNA_FW_AUDIT").is_ok();
    let mut entered: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut l32r_targets: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut overlay_stores: Vec<(u64, u32, u32)> = Vec::new();
    let mut tail: std::collections::VecDeque<(u64, u32, String, [u32; 16])> =
        std::collections::VecDeque::with_capacity(tail_depth.saturating_add(1));
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    while n < max {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        for &w in &WP {
            if pc <= w && w < pc.saturating_add(8) {
                let covered = proc
                    .cpu
                    .translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
                    .ok()
                    .map(|phys| {
                        let bytes: [u8; 8] =
                            std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                        w < pc + u32::from(decode::decode(&bytes, proc.cpu.pc).len)
                    })
                    .unwrap_or(false);
                if covered {
                    first.entry(w).or_insert((n, pc));
                }
            }
        }
        if audit {
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                let bytes: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                match decode::decode(&bytes, proc.cpu.pc).op {
                    Op::Entry { .. } => {
                        entered.insert(pc);
                    }
                    Op::L32r { target, .. } => {
                        l32r_targets.insert(target & 0x00ff_ffff);
                    }
                    Op::S32i { s, imm, .. }
                    | Op::S32iN { s, imm, .. }
                    | Op::S16i { s, imm, .. }
                    | Op::S8i { s, imm, .. }
                    | Op::S32ri { s, imm, .. }
                    | Op::S32c1i { s, imm, .. }
                    | Op::S32e { s, imm, .. } => {
                        let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm) & 0x00ff_ffff;
                        if overlays.iter().any(|&(lo, hi)| (lo..hi).contains(&ea)) {
                            overlay_stores.push((n, pc, ea));
                        }
                    }
                    _ => {}
                }
            }
        }
        if tail_depth != 0 {
            let op = match proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let bytes: [u8; 8] =
                        std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&bytes, proc.cpu.pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            if !calls_only
                || op.starts_with("Call")
                || op.starts_with("Entry")
                || op.starts_with("Retw")
                || op.starts_with("RetN")
            {
                let regs = std::array::from_fn(|r| proc.cpu.regs.read_ar(r as u8));
                tail.push_back((n, pc, op, regs));
                if tail.len() > tail_depth {
                    tail.pop_front();
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) at pc={:#x} (idle)", proc.cpu.pc);
                break;
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                break;
            }
        }
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("sysstub spin at {addr:#x}");
            break;
        }
    }

    eprintln!(
        "instrs = {n}, stop = {stop}, last_pc = {:#x} {}",
        proc.cpu.pc,
        nearest_symbol(&proc.symbols, proc.cpu.pc)
    );
    if tail_depth != 0 {
        eprintln!("--- last {tail_depth} instructions ---");
        for (nn, pc, op, regs) in &tail {
            eprintln!(
                "  n={nn:>7} pc={pc:#08x} {:<26} {op:<42} a0={:#x} a2={:#x} a3={:#x} a8={:#x}",
                nearest_symbol(&proc.symbols, *pc),
                regs[0],
                regs[2],
                regs[3],
                regs[8]
            );
        }
    }
    for w in &WP {
        match first.get(w) {
            Some((f, pc)) if pc == w => {
                eprintln!("  {w:#x} ({:<20}) first-hit n={f}", nearest_symbol(&proc.symbols, *w));
            }
            Some((f, pc)) => eprintln!(
                "  {w:#x} ({:<20}) first-hit n={f} (covered by instruction pc={pc:#x})",
                nearest_symbol(&proc.symbols, *w)
            ),
            None => eprintln!("  {w:#x} ({:<20}) NEVER", nearest_symbol(&proc.symbols, *w)),
        }
    }
    if audit {
        let canonical: Vec<u32> = entered
            .iter()
            .copied()
            .filter(|pc| overlays.iter().any(|&(lo, hi)| (lo..hi).contains(pc)))
            .collect();
        let alias_entries: Vec<(u32, u32)> = canonical
            .iter()
            .copied()
            .map(|pc| (pc, pc + (OVL - 0x5c)))
            // If the displaced address is itself overlaid, it does not fetch
            // the canonical function's +0x5c bytes and is not an alias.
            .filter(|&(_, alias)| {
                !overlays.iter().any(|&(lo, hi)| (lo..hi).contains(&alias)) && entered.contains(&alias)
            })
            .collect();
        eprintln!("--- overlay invariants ---");
        eprintln!("  stores into overlay VMAs: {}", overlay_stores.len());
        for (nn, pc, ea) in &overlay_stores {
            eprintln!("    n={nn} pc={pc:#x} -> {ea:#x}");
        }
        eprintln!("  entered +0x5c aliases of canonical entries: {}", alias_entries.len());
        for (pc, alias) in &alias_entries {
            eprintln!("    canonical={pc:#x} alias={alias:#x}");
        }
        let overlaid_l32r: Vec<u32> = l32r_targets
            .iter()
            .copied()
            .filter(|target| overlays.iter().any(|&(lo, hi)| (lo..hi).contains(target)))
            .collect();
        eprintln!("  live overlaid L32r targets: {overlaid_l32r:#x?}");
    }
    let post: Vec<(&'static str, u32)> = proc.bus.scan_bytes(&MAGIC);
    let new: Vec<_> = post.iter().filter(|h| !baseline.contains(*h)).collect();
    eprintln!("--- runtime-new '_NPU' magic (chann_info publish) ---");
    if new.is_empty() {
        eprintln!("  NONE -- still pre-alive (0x55f8 may be reached but publish path needs more framing)");
    } else {
        for (r, a) in &new {
            eprintln!("  {r}@{a:#x}  <-- ALIVE: chann_info published!");
        }
    }
}
