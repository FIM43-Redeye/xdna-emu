use super::*;
use crate::firmware::mmio::Region;

/// M2c iter18 (Session-4) EXPERIMENT: inject the AIE-completion interrupt.
/// The completion path is interrupt-driven: on silicon the AIE array raises
/// an IRQ whose status appears in `0x27010d28`; the level-1 ISR (via the
/// general-exc handler `0x2958`, EXCCAUSE=4) reads it, posts an event
/// message, and `wake_tasks_by_event_mask(1<<id)` sets task 0x10f10's pending
/// mask `[0x10f40]`. Nothing fires that IRQ in EMU (no array model), so boot
/// wedges. This probe FAITHFULLY drives the interrupt: warm up to steady
/// state, seed the AIE status reg, set `cpu.interrupt`, and keep stepping --
/// `step()` delivers the level-1 interrupt as soon as PS.INTLEVEL returns to
/// 0. Watches whether the interrupt is taken, whether the ISR/event path
/// runs, whether `[0x10f40]` gets set, and how far boot advances -- pinning
/// the completion contract end-to-end (the last boundary inch + the first
/// rail of H-b delivery). Env: XDNA_FW_INT_WARMUP (default 60000),
/// XDNA_FW_INT_STATUS (hex, seed of 0x27010d28, default 0xffffffff),
/// XDNA_FW_INT_LINE (hex, cpu.interrupt bits; 0 => use current INTENABLE),
/// XDNA_FW_INT_RUN (steps after inject, default 400000), XDNA_FW_INT_RESEED
/// (presence => re-seed status every step). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_inject_interrupt() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the interrupt-injection experiment");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let env_hex = |k: &str, d: u32| -> u32 {
        std::env::var(k)
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(d)
    };
    let warmup = env_u64("XDNA_FW_INT_WARMUP", 60_000);
    let status_val = env_hex("XDNA_FW_INT_STATUS", 0xffff_ffff);
    let line = env_hex("XDNA_FW_INT_LINE", 0);
    let run = env_u64("XDNA_FW_INT_RUN", 400_000);
    let reseed = std::env::var("XDNA_FW_INT_RESEED").is_ok();

    const STATUS_REG: u32 = 0x2701_0d28;
    const GEN_EXC: u32 = 0x28b4; // general-exception handler entry (iter23; via vector 0xae0)
    const WAKE: u32 = 0xd84c; // wake_tasks_by_event_mask
    const PENDING: u32 = 0x10f40; // task 0x10f10's pending-event mask
    const CUR_TASK: u32 = 0x2278; // scheduler current-task ptr

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Warm up to steady state.
    for _ in 0..warmup {
        if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
            break;
        }
    }
    let intenable = proc.cpu.intenable;
    let fire = if line == 0 { intenable } else { line };
    eprintln!("=== M2c interrupt-injection experiment ===");
    eprintln!("at warmup n={warmup}: pc={:#x} {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
    eprintln!(
        "  INTENABLE={intenable:#010x} INTERRUPT={:#010x} intlevel={} excm={}",
        proc.cpu.interrupt,
        proc.cpu.regs.intlevel(),
        proc.cpu.regs.excm()
    );
    eprintln!(
        "  current-task [0x2278]={:#x}  pending [0x10f40]={:#x}",
        proc.cpu.data_read32(&mut proc.bus, CUR_TASK).unwrap_or(0),
        proc.cpu.data_read32(&mut proc.bus, PENDING).unwrap_or(0)
    );
    eprintln!(
        "  seeding {STATUS_REG:#x}={status_val:#x}, setting INTERRUPT |= {fire:#010x}, reseed={reseed}"
    );

    // Inject.
    let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
    proc.cpu.interrupt |= fire;

    let mut n = warmup;
    let mut min_intlevel = 15u32;
    let mut lvl0_windows = 0u64;
    let mut first_gen_exc: Option<u64> = None;
    let mut first_wake: Option<u64> = None;
    let mut first_pending: Option<(u64, u32)> = None;
    let mut stop = String::from("run budget reached");
    let end = warmup + run;
    while n < end {
        if reseed {
            let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
        }
        let pc = proc.cpu.pc;
        if pc == GEN_EXC && first_gen_exc.is_none() {
            first_gen_exc = Some(n);
        }
        if pc == WAKE && first_wake.is_none() {
            first_wake = Some(n);
        }
        if first_pending.is_none() {
            let p = proc.cpu.data_read32(&mut proc.bus, PENDING).unwrap_or(0);
            if p != 0 {
                first_pending = Some((n, p));
            }
        }
        let lvl = proc.cpu.regs.intlevel();
        if lvl < min_intlevel {
            min_intlevel = lvl;
        }
        if lvl == 0 && !proc.cpu.regs.excm() {
            lvl0_windows += 1;
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) at pc={pc:#x} (idle -- boot reached waiti!)");
                break;
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                break;
            }
        }
    }
    eprintln!("--- results ---");
    eprintln!("ran to n={n}  (advanced {} past warmup)", n - warmup);
    eprintln!("stop reason         = {stop}");
    eprintln!("min intlevel seen   = {min_intlevel}   (level-0 deliverable windows = {lvl0_windows})");
    eprintln!("interrupt taken (@0x2958) first at n = {first_gen_exc:?}");
    eprintln!("wake_tasks (@0xd84c)     first at n = {first_wake:?}");
    eprintln!("[0x10f40] pending set    first at   = {first_pending:x?}");
    eprintln!(
        "final INTERRUPT={:#010x} intlevel={} excm={}",
        proc.cpu.interrupt,
        proc.cpu.regs.intlevel(),
        proc.cpu.regs.excm()
    );
    eprintln!("final pc={:#x} {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
    eprintln!(
        "final current-task [0x2278]={:#x}",
        proc.cpu.data_read32(&mut proc.bus, CUR_TASK).unwrap_or(0)
    );
}

/// M4.1 EXPERIMENT: discover the x2i (host->firmware) mailbox receive
/// address. Boot to idle via the completion-agent bootstrap (`waiti`), then
/// inject the mailbox doorbell (Xtensa INTERRUPT bit0) and capture every
/// Mailbox-aperture access the interrupt handler issues. The FIRST mailbox
/// READ after the doorbell is the firmware's x2i-tail poll -- the address the
/// host producer must write to post a request. Discovered by observation, not
/// guessed (the driver leaves x2i offsets firmware-defined). Ignored unless
/// XDNA_FW_PROBE is set. Env: XDNA_FW_MB_BOOT (idle budget, default 2_000_000),
/// XDNA_FW_MB_STEPS (post-doorbell steps, default 4000).
#[test]
fn m2c_probe_mailbox_receive() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the mailbox-receive discovery probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let boot_budget = env_u64("XDNA_FW_MB_BOOT", 2_000_000);
    let steps = env_u64("XDNA_FW_MB_STEPS", 4000);

    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Boot to the command-loop idle (completion-agent bootstrap gets past the
    // task-B wall to `waiti`).
    proc.enable_host_mailbox();
    let report = proc.boot_to_idle(boot_budget);
    eprintln!("=== M4.1 mailbox-receive discovery ===");
    eprintln!(
        "boot: reached_idle={} instrs={} last_pc={:#x} {} INTENABLE={:#010x}",
        report.reached_idle,
        report.instrs_executed,
        report.last_pc,
        nearest_symbol(&proc.symbols, report.last_pc),
        proc.cpu.intenable,
    );
    if !report.reached_idle {
        eprintln!("did NOT reach idle -- cannot exercise the receive path; aborting");
        return;
    }

    // Inject the mailbox doorbell: ensure bit0 is enabled and raise it.
    proc.cpu.intenable |= 1;
    proc.cpu.interrupt |= 1;
    proc.bus.arm_probe();

    let mut n = 0u64;
    let mut stop = String::from("steps budget reached");
    while n < steps {
        proc.bus.set_probe_pc(proc.cpu.pc);
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) at pc={:#x} (returned to idle)", proc.cpu.pc);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                break;
            }
        }
    }
    let log = proc.bus.take_probe();
    eprintln!("post-doorbell: ran {n} steps, stop={stop}");
    eprintln!("mailbox/array/system accesses after doorbell = {}", log.len());
    eprintln!("--- accesses in order (seq: pc[sym] region rd/wr addr=value wN) ---");
    for a in &log {
        eprintln!(
            "{:>5}: pc={:#x}[{}] {:?} {} {:#x}={:#x} w{}",
            a.seq,
            a.pc,
            nearest_symbol(&proc.symbols, a.pc),
            a.region,
            if a.is_write { "wr" } else { "rd" },
            a.addr,
            a.value,
            a.width,
        );
    }
}

/// M3 EXPERIMENT (the lever M1 unlocked): boot the firmware with a REAL AIE2
/// array attached to its bus, vs the stub, and compare. Until M1 the Array
/// aperture was a discard stub -- every firmware array READ returned 0. With
/// a `DeviceState` attached, array reads return real register values, so if
/// the firmware branches on array state during boot the control flow can
/// diverge. The banked wall was "the completion contract lives in the array,
/// not derivable from firmware alone" -- this is the first test of that
/// hypothesis with an actual array. Runs the same boot twice (stub, attached),
/// reports where each stops and every Array-aperture access. Ignored unless
/// XDNA_FW_PROBE set. Env: XDNA_FW_MAX (budget, default 700_000).
#[test]
fn m2c_probe_boot_with_array() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the array-attached boot experiment");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(700_000);
    let raw = std::fs::read(&path).expect("read firmware");

    // Run one boot; return (instrs, last_pc, stop, array-access log).
    let run = |attach: bool| -> (u64, u32, String, Vec<mmio::StubAccess>) {
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        if attach {
            proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        }
        proc.bus.arm_probe();
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < max {
            proc.bus.set_probe_pc(proc.cpu.pc);
            let pc = proc.cpu.pc;
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE)");
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
        let last_pc = proc.cpu.pc;
        (n, last_pc, stop, proc.bus.take_probe())
    };

    let syms = load_symbols();
    for (label, attach) in [("STUB", false), ("ATTACHED", true)] {
        let (n, last_pc, stop, log) = run(attach);
        let array: Vec<_> = log.iter().filter(|a| a.region == Region::Array).collect();
        let reads = array.iter().filter(|a| !a.is_write).count();
        let writes = array.iter().filter(|a| a.is_write).count();
        eprintln!("=== boot [{label}] ===");
        eprintln!("  instrs={n} last_pc={last_pc:#x} {}", nearest_symbol(&syms, last_pc));
        eprintln!("  stop={stop}");
        eprintln!("  array accesses: {} (rd={reads} wr={writes})", array.len());
        // Distinct array sites (pc, addr, is_write) -> values seen.
        use std::collections::BTreeMap;
        let mut sites: BTreeMap<(u32, u32, bool), Vec<u32>> = BTreeMap::new();
        for a in &array {
            sites.entry((a.pc, a.addr, a.is_write)).or_default().push(a.value);
        }
        for ((pc, addr, is_wr), vals) in sites.iter().take(30) {
            let mut v = vals.clone();
            v.sort_unstable();
            v.dedup();
            let vs = v.iter().take(4).map(|x| format!("{x:#x}")).collect::<Vec<_>>().join(",");
            eprintln!(
                "    pc={pc:#x} {} {addr:#x} vals=[{vs}] (n={})",
                if *is_wr { "wr" } else { "rd" },
                vals.len()
            );
        }
    }
}

/// M2c iter18 RE TOOL (planning discovery): locate the firmware-LOCAL i2x
/// ring buffer where the fw writes the mailbox message header. The driver
/// (xdna-driver) stores HOST-view addresses; the emulator runs the fw's
/// LOCAL view, so the ring base must be found empirically. Store-watches
/// every 32-bit store for the header magic (top byte 0x1D == the `id`
/// field), and every write to the i2x tail reg 0x27200170 (the post).
/// Reports each magic store's EA (= header.id, base = EA-8), the 16-byte
/// header decoded, and the memory region it lands in (backed vs the
/// unmodeled 0x08a00000 gap). Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_i2x_ring_locate() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the i2x-ring locate probe");
        return;
    }
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    const MAX: u64 = 60_000;
    const TAIL: u32 = 0x2720_0170;
    const MAGIC: u32 = 0x1D00_0000;
    const MAGIC_MASK: u32 = 0xFF00_0000;
    let region_name = |a: u32| -> &'static str {
        match a {
            _ if a < 0x0400_0000 => "LOCAL(low)",
            _ if a < 0x0800_0000 => "ARRAY",
            _ if (0x0800_0000..0x08b0_0000).contains(&a) => "GAP(unbacked->System)",
            _ if (0x08b0_0000..0x2700_0000).contains(&a) => "RAM",
            _ if (0x2700_0000..0x2800_0000).contains(&a) => "MAILBOX",
            _ => "SYSTEM",
        }
    };

    let _ = (MAGIC, MAGIC_MASK);
    // First store to each buffer/mailbox address (EA >= 0x08000000): reveals
    // where the fw builds the message. Keyed by EA; keeps (n, pc, val, width).
    let mut buf_stores: std::collections::BTreeMap<u32, (u64, u32, u32, u8)> =
        std::collections::BTreeMap::new();
    let mut tail_writes: Vec<(u64, u32, u32)> = Vec::new(); // (n, pc, val)
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    while n < MAX {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            // (t, s, imm, width_bytes) for every store width.
            let store = match decode::decode(&b, pc).op {
                Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => {
                    Some((t, s, imm, 4u8))
                }
                Op::S16i { t, s, imm } => Some((t, s, imm, 2)),
                Op::S8i { t, s, imm } => Some((t, s, imm, 1)),
                _ => None,
            };
            if let Some((t, s, imm, w)) = store {
                let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                let val = proc.cpu.regs.read_ar(t);
                if ea == TAIL {
                    tail_writes.push((n, pc, val));
                }
                // Buffer/mailbox writes only (skip local/ROM/array scratch).
                if ea >= 0x0800_0000 {
                    buf_stores.entry(ea).or_insert((n, pc, val, w));
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?})");
                break;
            }
            Step::Unknown { pc: upc, word } => {
                stop = format!("Unknown at {upc:#x} word={word:#010x}");
                break;
            }
        }
    }
    eprintln!("=== M2c i2x-ring locate ===");
    eprintln!("instrs executed = {n}; stop = {stop}");
    eprintln!("--- i2x tail (0x27200170) writes (the post) ---");
    for (n, pc, val) in &tail_writes {
        eprintln!("  n={n:>6} pc={pc:#08x} tail <- {val:#x}");
    }
    eprintln!("--- first store to each buffer/mailbox address (EA >= 0x08000000) ---");
    eprintln!("    (contiguous runs in one region == a message/struct the fw wrote)");
    let mut last_ea = 0u32;
    for (ea, (n, pc, val, w)) in &buf_stores {
        let gap = if *ea != last_ea.wrapping_add(4) && last_ea != 0 {
            "  <-- new block"
        } else {
            ""
        };
        eprintln!("  {ea:#010x} ({:<22}) <- {val:#010x} (w{w}) n={n:>6} pc={pc:#08x}{gap}", region_name(*ea));
        last_ea = *ea;
    }
    eprintln!("(total {} distinct buffer/mailbox addresses written)", buf_stores.len());
}

/// M2c iter18 RE TOOL (x2i experiment prep): locate the `mgmt_mbox_chann_info`
/// struct the fw publishes at boot, by catching the store of its magic
/// `0x55504e5f` ("_NPU", struct offset 0x20). Dumps the 64-byte struct
/// (x2i/i2x tail/head/buf/buf_sz device addresses + magic + msi_id +
/// prot_major/minor) from `magic_ea - 0x20`. Also records where the fw wrote
/// the struct pointer (the FW_ALIVE_OFF slot: a store whose VALUE equals the
/// struct base). This gives the x2i register/ring addresses needed to deliver
/// a host->fw message. Ignored unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_alive_struct() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the alive-struct locate");
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
    const MAGIC: u32 = 0x5550_4e5f; // "_NPU"
    let mut magic_ea: Option<(u64, u32, u32)> = None; // (n, pc, ea)
    let mut ptr_stores: Vec<(u64, u32, u32, u32)> = Vec::new(); // (n, pc, ea, val) where val==struct base
    let mut n = 0u64;
    // First pass: find the magic store.
    while n < MAX {
        let pc = proc.cpu.pc;
        if magic_ea.is_none() {
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                if let Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } =
                    decode::decode(&b, pc).op
                {
                    if proc.cpu.regs.read_ar(t) == MAGIC {
                        let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                        magic_ea = Some((n, pc, ea));
                    }
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) | Step::Unknown { .. } => break,
        }
        if magic_ea.is_some() && n > magic_ea.unwrap().0 + 50_000 {
            break; // struct + pointer publish happen close together
        }
    }
    eprintln!("=== M2c alive-struct locate ===");
    match magic_ea {
        None => eprintln!("magic 0x55504e5f store NOT seen in {n} instrs"),
        Some((mn, mpc, mea)) => {
            let base = mea.wrapping_sub(0x20);
            eprintln!("magic store at n={mn} pc={mpc:#x} -> magic@{mea:#x}, struct base {base:#x}");
            let fields = [
                "x2i_tail",
                "x2i_head",
                "x2i_buf",
                "x2i_buf_sz",
                "i2x_tail",
                "i2x_head",
                "i2x_buf",
                "i2x_buf_sz",
                "magic",
                "msi_id",
                "prot_major",
                "prot_minor",
            ];
            for (i, name) in fields.iter().enumerate() {
                let a = base.wrapping_add((i * 4) as u32);
                eprintln!(
                    "  +{:#04x} {name:<11} = {:#x}",
                    i * 4,
                    proc.cpu.data_read32(&mut proc.bus, a).unwrap_or(0)
                );
            }
            // Re-scan for a store whose VALUE == struct base (the FW_ALIVE_OFF publish).
            let mut proc2 = {
                let raw = std::fs::read(&path).expect("read");
                FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse"))
            };
            let mut m = 0u64;
            while m < mn + 50_000 {
                let pc = proc2.cpu.pc;
                if let Ok(phys) = proc2.cpu.translate(&mut proc2.bus, pc, xtensa::interp::Access::Fetch) {
                    let b: [u8; 8] =
                        std::array::from_fn(|k| proc2.bus.fetch8(pc + k as u32, phys + k as u32));
                    if let Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } =
                        decode::decode(&b, pc).op
                    {
                        let v = proc2.cpu.regs.read_ar(t);
                        if v == base {
                            let ea = proc2.cpu.regs.read_ar(s).wrapping_add(imm);
                            ptr_stores.push((m, pc, ea, v));
                        }
                    }
                }
                match proc2.cpu.step(&mut proc2.bus) {
                    Step::Ran | Step::Exception { .. } => m += 1,
                    Step::Wait(_) | Step::Unknown { .. } => break,
                }
            }
            eprintln!(
                "--- stores whose value == struct base {base:#x} (FW_ALIVE_OFF publish candidates) ---"
            );
            for (m, pc, ea, v) in &ptr_stores {
                eprintln!("  n={m} pc={pc:#x}  [{ea:#x}] <- {v:#x}");
            }
        }
    }
}

/// M2c boot-to-idle RE: trace the final alive-publication stores through the
/// firmware's live DTLB. Records every store whose virtual or translated
/// physical address is in the host-SRAM device-address band, every store of a
/// plausible device pointer, and the low-address byte stores used by the
/// publisher. At the terminal waiti, compare translating CPU reads against raw
/// local DRAM for the local struct, its computed device pointer, and
/// FW_ALIVE_OFF. Observation only: no mappings, memory, or interrupts injected.
#[test]
fn m2c_probe_alive_sram_path() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the alive SRAM-path probe");
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
    const SRAM_LO: u32 = 0x0308_0000;
    const SRAM_HI: u32 = 0x0310_0000;
    let mut stores: Vec<(u64, u32, u32, u32, u32, u8)> = Vec::new();
    let mut reads: Vec<(u64, u32, u32, u32, u32)> = Vec::new();
    let mut decoded_stores = 0u64;
    let mut sram_virtual_stores = 0u64;
    let mut sram_physical_stores = 0u64;
    let mut n = 0u64;
    let stop;
    loop {
        let pc = proc.cpu.pc;
        if let Ok(fetch_phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, fetch_phys + k as u32));
            let op = decode::decode(&b, pc).op;
            let (store, width) = match op {
                Op::S32i { t, s, imm }
                | Op::S32iN { t, s, imm }
                | Op::S32ri { t, s, imm }
                | Op::S32c1i { t, s, imm }
                | Op::S32e { t, s, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t), 4))
                }
                Op::Ssi { ft, s, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.fr[ft as usize], 4))
                }
                Op::S16i { t, s, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t) & 0xffff, 2))
                }
                Op::S8i { t, s, imm } => {
                    Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t) & 0xff, 1))
                }
                _ => None,
            }
            .map_or((None, 0), |(ea, value, width)| (Some((ea, value)), width));

            if let Some((ea, value)) = store {
                decoded_stores += 1;
                // Resolve the exact route the impending instruction will use.
                // This is the canonical CPU translator; the real step repeats
                // the same lookup immediately afterward.
                if let Ok(paddr) = proc.cpu.translate(&mut proc.bus, ea, xtensa::interp::Access::Store) {
                    sram_virtual_stores += u64::from((SRAM_LO..SRAM_HI).contains(&ea));
                    sram_physical_stores += u64::from((SRAM_LO..SRAM_HI).contains(&paddr));
                    if (n >= 52_000 && ea < 4)
                        || (0x2721_0000..0x2721_0100).contains(&ea)
                        || (SRAM_LO..SRAM_HI).contains(&ea)
                        || (SRAM_LO..SRAM_HI).contains(&paddr)
                        || (SRAM_LO..SRAM_HI).contains(&value)
                    {
                        stores.push((n, pc, ea, paddr, value, width));
                    }
                }
            }
            if n >= 52_000 {
                if let Op::L32i { s, imm, .. } | Op::L32iN { s, imm, .. } = op {
                    let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    if ea < 4 {
                        if let (Ok(paddr), Ok(value)) = (
                            proc.cpu.translate(&mut proc.bus, ea, xtensa::interp::Access::Load),
                            proc.cpu.data_read32(&mut proc.bus, ea),
                        ) {
                            reads.push((n, pc, ea, paddr, value));
                        }
                    }
                }
            }
        }

        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc & 0x00ff_ffff);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
        if n >= MAX {
            stop = "budget".to_string();
            break;
        }
    }

    eprintln!("=== M2c alive SRAM path ===");
    eprintln!("instrs={n}, stop={stop}");
    eprintln!(
        "decoded store instructions={decoded_stores}, SRAM-band virtual targets={sram_virtual_stores}, \
         SRAM-band translated targets={sram_physical_stores}"
    );
    eprintln!("--- candidate stores: n pc virtual-EA -> physical value width ---");
    for (n, pc, ea, paddr, value, width) in &stores {
        eprintln!(
            "  n={n:>8} pc={:#08x}  EA={ea:#010x} -> PA={paddr:#010x}  value={value:#010x} w{width}",
            pc & 0x00ff_ffff
        );
    }
    eprintln!("--- final low-address loads: n pc virtual-EA -> physical value ---");
    for (n, pc, ea, paddr, value) in &reads {
        eprintln!(
            "  n={n:>8} pc={:#08x}  EA={ea:#010x} -> PA={paddr:#010x}  value={value:#010x}",
            pc & 0x00ff_ffff
        );
    }

    eprintln!("--- no-stimulus steps after the first waiti ---");
    for sample in 1..=4 {
        let pc = proc.cpu.pc;
        let step = proc.cpu.step(&mut proc.bus);
        eprintln!("  n={n} sample={sample} pc={:#08x} -> {step:?}", pc & 0x00ff_ffff);
    }

    eprintln!("--- terminal translating reads ---");
    for addr in [0, 0x0001_4800, 0x030b_b000, 0x030b_b020, 0x030b_f000] {
        let paddr = proc.cpu.translate(&mut proc.bus, addr, xtensa::interp::Access::Load);
        let value = proc.cpu.data_read32(&mut proc.bus, addr);
        eprintln!("  VA={addr:#010x} -> PA={paddr:?}, cpu.data_read32={value:?}");
    }
    eprintln!("--- raw local DRAM comparison ---");
    for addr in [0, 0x0001_4800, 0x030b_b000, 0x030b_b020, 0x030b_f000] {
        eprintln!("  local_data@{addr:#010x} = {:#010x}", proc.bus.load_local32(addr));
    }
}

/// M2c iter18 DIAGNOSTIC: does boot ever enter the event-ISR path
/// (`FUN_00005580`) and read the event-source register `0x27010d28`?
/// Decides the fork: path never entered -> interrupt-vectored, interrupt
/// never fires; path entered with sentinel-value -> value problem. Records
/// pc-hits at the ISR entry / event-loop / dispatch-call / event-dispatcher,
/// every load whose EA == 0x27010d28 (with the value the fw actually read),
/// and the final PS.INTLEVEL. XDNA_FW_MAX overrides the budget. Ignored
/// unless XDNA_FW_PROBE is set.
#[test]
fn m2c_probe_event_source() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the event-source probe");
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
        .unwrap_or(1_500_000);
    const EVENT_SRC: u32 = 0x2701_0d28;
    // pc landmarks in FUN_00005580 / the event dispatcher.
    const ISR_ENTRY: u32 = 0x5580;
    const EVENT_LOOP: u32 = 0x57e0; // Beqi a6,5 -- generic-event decode
    const SRC_LOAD: u32 = 0x57f8; // L32i a2,[a14+0] -- reads the event source
    const DISPATCH_CALL: u32 = 0x5809; // Call8 FUN_0000d84c(mask)
    const DISPATCHER: u32 = 0xd84c; // FUN_0000d84c entry

    let mut pc_hits: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut src_reads: u64 = 0;
    let mut src_samples: Vec<u32> = Vec::new();
    let mut n = 0u64;
    let mut stop = "budget reached";
    while n < max {
        let pc = proc.cpu.pc;
        if matches!(pc, ISR_ENTRY | EVENT_LOOP | SRC_LOAD | DISPATCH_CALL | DISPATCHER) {
            *pc_hits.entry(pc).or_insert(0) += 1;
        }
        // Generic EA watch for loads of the event-source register.
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let op = decode::decode(&b, pc).op;
            let ea = match op {
                decode::Op::L32i { s, imm, .. } | decode::Op::L32iN { s, imm, .. } => {
                    Some(proc.cpu.regs.read_ar(s).wrapping_add(imm))
                }
                _ => None,
            };
            if ea == Some(EVENT_SRC) {
                src_reads += 1;
                // Capture the value the fw reads: step, then read the dest reg.
                if let decode::Op::L32i { t, .. } | decode::Op::L32iN { t, .. } = decode::decode(&b, pc).op {
                    let _ = proc.cpu.step(&mut proc.bus);
                    n += 1;
                    let v = proc.cpu.regs.read_ar(t);
                    if src_samples.len() < 8 {
                        src_samples.push(v);
                    }
                    continue;
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
    eprintln!("=== M2c event-source probe ===");
    eprintln!("instrs = {n}, stop = {stop}, last_pc = {:#x}", proc.cpu.pc);
    eprintln!("PS.INTLEVEL = {}", proc.cpu.regs.intlevel());
    eprintln!("--- pc landmark hits (FUN_00005580 event ISR + dispatcher) ---");
    for (pc, c) in &pc_hits {
        let label = match *pc {
            ISR_ENTRY => "ISR entry (0x5580)",
            EVENT_LOOP => "event-loop decode (0x57e0)",
            SRC_LOAD => "event-source load (0x57f8)",
            DISPATCH_CALL => "dispatch call (0x5809)",
            DISPATCHER => "FUN_0000d84c (0xd84c)",
            _ => "?",
        };
        eprintln!("  {pc:#08x}  {label:<28}  hits={c}");
    }
    if pc_hits.is_empty() {
        eprintln!("  (NONE -- the event ISR path is never entered)");
    }
    eprintln!("--- event-source register 0x27010d28 ---");
    eprintln!("  reads = {src_reads}, sample values = {src_samples:x?}");
}

/// ARC-2 STEP 0 (2026-07-10): pre-alive vs post-alive discriminator. The driver
/// (`aie2_get_mgmt_chann_info`) says the firmware, once alive, writes the
/// `mgmt_mbox_chann_info` struct's *device-BAR address* to `FW_ALIVE_OFF` and the
/// driver polls that slot for a non-zero value. The struct's `magic` is STATIC in
/// the image (file 0x3388), so there is NO runtime magic store to key on -- the
/// only observable "alive" signal is a store whose VALUE is a device-BAR pointer
/// (SRAM/MBOX BAR window 0x0308_0000..0x030D_0000) landing in the mailbox/SRAM
/// aperture. This probe boots NATURALLY (no host model, no injection) to budget
/// and records: (a) every store whose value is such a device pointer -- the
/// alive-publish signature; (b) a deduped map of every store into the firmware's
/// mailbox aperture (0x2700_0000..0x2800_0000) with first/last value and count --
/// the channel-setup writes. If (a) is empty across a full boot, the idle we reach
/// is PRE-alive and go-alive is gated by something upstream (SMU/PSP power-on or a
/// mailbox-init step); if it fires, we have the exact publish site + pointer.
/// Env: XDNA_FW_MAX (default 3_000_000). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_alive_publish() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the alive-publish discriminator");
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
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Device-BAR pointer window (host SRAM+MBOX BARs): a store of a value here is
    // the firmware publishing a device address (chann_info pointer / ring regs).
    const DEVPTR_LO: u32 = 0x0308_0000;
    const DEVPTR_HI: u32 = 0x030D_0000;
    // Firmware-local mailbox aperture.
    const MBOX_LO: u32 = 0x2700_0000;
    const MBOX_HI: u32 = 0x2800_0000;

    // (a) alive-publish signature: stores whose VALUE is a device pointer.
    let mut dev_pub: Vec<(u64, u32, u32, u32)> = Vec::new(); // (n, pc, ea, val)
                                                             // (b) mailbox-aperture store map: EA -> (n_first, pc_first, val_first, count, val_last).
    let mut mbox: std::collections::BTreeMap<u32, (u64, u32, u32, u64, u32)> =
        std::collections::BTreeMap::new();

    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    while n < max {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let store = match decode::decode(&b, pc).op {
                Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => {
                    Some((t, s, imm))
                }
                _ => None,
            };
            if let Some((t, s, imm)) = store {
                let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                let val = proc.cpu.regs.read_ar(t);
                if (DEVPTR_LO..DEVPTR_HI).contains(&val) {
                    dev_pub.push((n, pc, ea, val));
                }
                if (MBOX_LO..MBOX_HI).contains(&ea) {
                    mbox.entry(ea)
                        .and_modify(|e| {
                            e.3 += 1;
                            e.4 = val;
                        })
                        .or_insert((n, pc, val, 1, val));
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

    eprintln!("=== ARC-2 alive-publish discriminator ===");
    eprintln!(
        "instrs = {n}, stop = {stop}, last_pc = {:#x} {}",
        proc.cpu.pc,
        nearest_symbol(&proc.symbols, proc.cpu.pc)
    );
    eprintln!("--- (a) device-pointer publishes (VALUE in {DEVPTR_LO:#x}..{DEVPTR_HI:#x}) ---");
    if dev_pub.is_empty() {
        eprintln!("  NONE -- firmware never published a device pointer => idle is PRE-alive");
    } else {
        for (n, pc, ea, val) in dev_pub.iter().take(40) {
            eprintln!(
                "  n={n:>8} pc={pc:#08x}[{}]  [{ea:#x}] <- {val:#x}",
                nearest_symbol(&proc.symbols, *pc)
            );
        }
        eprintln!("  ({} total device-pointer publishes)", dev_pub.len());
    }
    eprintln!("--- (b) mailbox-aperture stores (EA {MBOX_LO:#x}..{MBOX_HI:#x}) ---");
    eprintln!("    (EA: first_val -> last_val, count, first seen @n/pc)");
    for (ea, (nf, pcf, vf, cnt, vl)) in &mbox {
        let arrow = if vf == vl {
            format!("={vf:#x}")
        } else {
            format!("{vf:#x}->{vl:#x}")
        };
        eprintln!(
            "  {ea:#010x}: {arrow:<24} cnt={cnt:<5} n={nf:>8} pc={pcf:#08x}[{}]",
            nearest_symbol(&proc.symbols, *pcf)
        );
    }
    eprintln!("  ({} distinct mailbox-aperture addresses written)", mbox.len());
}

/// ARC-2 STEP 0b (2026-07-10): the go-alive publish detector, width- and
/// region-independent. The static `chann_info` magic ("_NPU") lives in firmware
/// rodata (a private image copy). For the driver to read `chann_info`, the
/// firmware must place a copy in host-visible memory -- which, by ANY store width
/// (word memcpy or byte memcpy), makes the magic byte pattern appear at a NEW
/// address in a runtime-grown backing store. This probe scans every backing store
/// for the pattern BEFORE any step (static baseline) and AFTER a full boot, and
/// reports occurrences that appeared at runtime.
///
/// Through iter24 this was an AIRTIGHT pre-alive proof (empty runtime-delta =>
/// firmware never published). iter25 (the go-alive publish path mapped) FLIPPED
/// it: a natural boot now copies the built `mgmt_mbox_chann_info` (see
/// m2c_probe_alive_struct for the field dump) to `local_data@0x14820`, so the
/// runtime-delta is non-empty and this probe now witnesses go-alive. An empty
/// delta here is now a REGRESSION signal (a publish-path overlay dropped). Env:
/// XDNA_FW_MAX (default 3_000_000). Ignored unless XDNA_FW_PROBE set.
#[test]
fn m2c_probe_alive_magic_scan() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the alive-magic-scan proof");
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
    const MAGIC: [u8; 4] = [0x5f, 0x4e, 0x50, 0x55]; // "_NPU"
    let raw = std::fs::read(&path).expect("read firmware");

    // Static baseline: scan before stepping.
    let base_proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse"));
    let baseline: std::collections::BTreeSet<(&'static str, u32)> =
        base_proc.bus.scan_bytes(&MAGIC).into_iter().collect();

    // Full boot, then scan.
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse"));
    let mut n = 0u64;
    let mut stop = String::from("budget reached");
    while n < max {
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                n += 1;
                stop = format!("Wait({reason:?}) at pc={:#x} (idle)", proc.cpu.pc);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                break;
            }
        }
        if proc.bus.sysstub().spinning().is_some() {
            stop = format!("sysstub spin");
            break;
        }
    }
    let post: Vec<(&'static str, u32)> = proc.bus.scan_bytes(&MAGIC);
    let new: Vec<_> = post.iter().filter(|h| !baseline.contains(*h)).collect();

    eprintln!("=== ARC-2 alive-magic-scan (go-alive publish detector) ===");
    eprintln!("instrs = {n}, stop = {stop}");
    eprintln!("static baseline '_NPU' occurrences ({}):", baseline.len());
    for (r, a) in &baseline {
        eprintln!("  {r}@{a:#x}");
    }
    eprintln!("RUNTIME-NEW '_NPU' occurrences ({}):", new.len());
    if new.is_empty() {
        eprintln!("  NONE -- REGRESSED to pre-alive: a go-alive publish-path overlay (iter25) dropped");
    } else {
        for (r, a) in &new {
            eprintln!("  {r}@{a:#x}  <-- chann_info published here");
        }
    }
}
