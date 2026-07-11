//! Static two-view coherence mapper for the stripped management firmware.
//!
//! Run the report with:
//! `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_coherence_mapper -- --nocapture`

use super::*;

use std::collections::{BTreeSet, HashSet, VecDeque};

const BASE_DELTA: u32 = 0x5c;
const OVERLAY_DELTA: u32 = 0x100;
const KNOWN_BASE_ROOTS: [u32; 2] = [super::super::RESET_ENTRY, 0x4525];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Range {
    lo: u32,
    hi: u32,
}

#[derive(Debug)]
struct FunctionEvidence {
    range: Range,
    score: i32,
    calls: Vec<u32>,
    literals: Vec<u32>,
}

fn image_bytes(image: &[u8], pc: u32, delta: u32) -> Option<&[u8]> {
    let start = pc.checked_add(delta)? as usize;
    image.get(start..start.checked_add(8)?)
}

fn word_at(image: &[u8], vma: u32, delta: u32) -> Option<u32> {
    let start = vma.checked_add(delta)? as usize;
    Some(u32::from_le_bytes(image.get(start..start.checked_add(4)?)?.try_into().ok()?))
}

fn branch_target(op: &decode::Op) -> Option<u32> {
    use decode::Op::*;
    match *op {
        Beqz { target, .. }
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
        | Bnall { target, .. } => Some(target),
        Loop { end, .. } | Loopnez { end, .. } => Some(end),
        _ => None,
    }
}

fn direct_call_target(op: &decode::Op) -> Option<u32> {
    use decode::Op::*;
    match *op {
        Call0 { target } | Call4 { target } | Call8 { target } | Call12 { target } => Some(target),
        _ => None,
    }
}

fn is_return(op: &decode::Op) -> bool {
    use decode::Op::*;
    matches!(op, Retw | RetwN | RetN | Rfe | Rfwo | Rfwu)
}

fn is_indirect_tail(op: &decode::Op) -> bool {
    matches!(op, decode::Op::Jx { .. })
}

fn literal_is_sane(value: u32) -> bool {
    value <= 0x0010_0000
        || (0x0320_0000..0x0340_0000).contains(&value)
        || (0x08b0_0000..0x08b1_0000).contains(&value)
        || (0x2000_0000..0x2010_0000).contains(&value)
        || (0x2720_0000..0x2730_0000).contains(&value)
        || value.count_ones() <= 4
        || value.count_zeros() <= 4
}

fn analyze_entry(image: &[u8], entry: u32, delta: u32) -> Option<FunctionEvidence> {
    let first = decode::decode(image_bytes(image, entry, delta)?, entry);
    if !matches!(first.op, decode::Op::Entry { .. }) {
        return None;
    }

    let floor = entry;
    let ceiling = entry.saturating_add(0x800).min(0x1_0000);
    let mut blocks = VecDeque::from([entry]);
    let mut visited = HashSet::new();
    let mut calls = BTreeSet::new();
    let mut literals = BTreeSet::new();
    let mut lo = entry;
    let mut hi = entry;
    let mut returns = 0u32;
    let mut tails = 0u32;

    while let Some(mut pc) = blocks.pop_front() {
        loop {
            if pc < floor || pc >= ceiling || !visited.insert(pc) {
                break;
            }
            if visited.len() > 512 {
                return None;
            }
            let d = decode::decode(image_bytes(image, pc, delta)?, pc);
            if d.len == 0 || matches!(d.op, decode::Op::Unknown { .. }) {
                return None;
            }
            let next = pc + u32::from(d.len);
            lo = lo.min(pc);
            hi = hi.max(next);
            if let decode::Op::L32r { target, .. } = d.op {
                literals.insert(target);
            }
            if let Some(target) = direct_call_target(&d.op) {
                calls.insert(target);
            }
            if let Some(target) = branch_target(&d.op) {
                if (floor..ceiling).contains(&target) {
                    blocks.push_back(target);
                }
            }
            if is_return(&d.op) {
                returns += 1;
                break;
            }
            if is_indirect_tail(&d.op) {
                tails += 1;
                break;
            }
            if let decode::Op::J { target } = d.op {
                if (floor..ceiling).contains(&target) {
                    let target_op = decode::decode(image_bytes(image, target, delta)?, target);
                    if !matches!(target_op.op, decode::Op::Entry { .. }) {
                        blocks.push_back(target);
                    } else {
                        tails += 1;
                    }
                } else {
                    tails += 1;
                }
                break;
            }
            pc = next;
        }
    }

    if visited.len() < 2 || returns + tails == 0 {
        return None;
    }
    let mut score = 24 + (visited.len().min(24) as i32) + 5 * returns as i32 + 2 * tails as i32;
    for &target in &literals {
        match word_at(image, target, delta) {
            Some(value) if literal_is_sane(value) => score += 4,
            Some(_) => score -= 7,
            None => score -= 10,
        }
    }
    Some(FunctionEvidence {
        range: Range { lo, hi },
        score,
        calls: calls.into_iter().collect(),
        literals: literals.into_iter().collect(),
    })
}

fn merge_ranges(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort_by_key(|r| (r.lo, r.hi));
    let mut out: Vec<Range> = Vec::new();
    for range in ranges {
        if let Some(last) = out.last_mut() {
            if range.lo <= last.hi + 0x10 {
                last.hi = last.hi.max(range.hi);
                continue;
            }
        }
        out.push(range);
    }
    out
}

fn literal_referenced_low_vmas(image: &[u8]) -> BTreeSet<u32> {
    let mut refs = BTreeSet::new();
    // A shifted alias changes the PC-relative L32R target address but reaches
    // the same physical word. The word's pointer VALUE remains the canonical
    // VMA, so only values loaded by coherent functions are accepted as seeds.
    for entry in 0x1a4..0x1_0000 {
        for delta in [BASE_DELTA, OVERLAY_DELTA] {
            let Some(evidence) = analyze_entry(image, entry, delta) else {
                continue;
            };
            for target in evidence.literals {
                let Some(value) = word_at(image, target, delta) else {
                    continue;
                };
                if (0x1a4..0x1_0000).contains(&value)
                    && (analyze_entry(image, value, BASE_DELTA).is_some()
                        || analyze_entry(image, value, OVERLAY_DELTA).is_some())
                {
                    refs.insert(value);
                }
            }
        }
    }
    refs
}

fn vector_signature_score(image: &[u8], delta: u32) -> (u32, u32) {
    let mut frame_ops = 0;
    let mut returns = 0;
    for pc in 0x800..0x980 {
        let Some(bytes) = image_bytes(image, pc, delta) else {
            continue;
        };
        match decode::decode(bytes, pc).op {
            decode::Op::S32e { .. } | decode::Op::L32e { .. } => frame_ops += 1,
            decode::Op::Rfwo | decode::Op::Rfwu => returns += 1,
            _ => {}
        }
    }
    (frame_ops, returns)
}

fn first_window_vector_is_coherent(image: &[u8], delta: u32) -> bool {
    let mut pc = 0x800;
    let mut frame_ops = 0;
    while pc < 0x840 {
        let Some(bytes) = image_bytes(image, pc, delta) else {
            return false;
        };
        let d = decode::decode(bytes, pc);
        match d.op {
            decode::Op::S32e { .. } | decode::Op::L32e { .. } => frame_ops += 1,
            decode::Op::Rfwo | decode::Op::Rfwu => return frame_ops >= 4,
            decode::Op::Unknown { .. } => return false,
            _ => {}
        }
        pc += u32::from(d.len.max(1));
    }
    false
}

fn derive_overlay_ranges(image: &[u8]) -> Vec<Range> {
    let mut candidates = literal_referenced_low_vmas(image);
    // Reset and Xtensa's VECBASE-relative architectural entry points anchor
    // addresses even when no literal pointer names them.
    candidates.insert(0x1a4);
    for offset in (0..0x300).step_by(0x40) {
        candidates.insert(0x800 + offset);
    }

    let mut overlay = Vec::new();
    let base_vectors = vector_signature_score(image, BASE_DELTA);
    let overlay_vectors = vector_signature_score(image, OVERLAY_DELTA);
    if first_window_vector_is_coherent(image, OVERLAY_DELTA)
        && !first_window_vector_is_coherent(image, BASE_DELTA)
        && overlay_vectors.0 > base_vectors.0
        && overlay_vectors.1 >= 4
    {
        overlay.push(Range { lo: 0x800, hi: 0x980 });
    }
    let mut seen = BTreeSet::new();
    let mut queue: VecDeque<u32> = candidates.into_iter().collect();
    while let Some(entry) = queue.pop_front() {
        if !seen.insert(entry) {
            continue;
        }
        let base = analyze_entry(image, entry, BASE_DELTA);
        let shifted = analyze_entry(image, entry, OVERLAY_DELTA);
        let pick_overlay = match (&base, &shifted) {
            (None, Some(_)) => true,
            (Some(a), Some(b)) => b.score >= a.score + 8,
            _ => false,
        };
        let Some(chosen) = (if pick_overlay {
            shifted.as_ref()
        } else {
            base.as_ref()
        }) else {
            continue;
        };
        for &target in &chosen.calls {
            if (0x1a4..0x1_0000).contains(&target) && !seen.contains(&target) {
                queue.push_back(target);
            }
        }
        for &target in &chosen.literals {
            if let Some(value) = word_at(image, target, if pick_overlay { OVERLAY_DELTA } else { BASE_DELTA })
            {
                if (0x1a4..0x1_0000).contains(&value) && !seen.contains(&value) {
                    queue.push_back(value);
                }
            }
        }
        if pick_overlay {
            overlay.push(chosen.range);
            for &target in &chosen.literals {
                if target < 0x1_0000 {
                    overlay.push(Range { lo: target, hi: target.saturating_add(4) });
                }
            }
        }
    }
    merge_ranges(overlay)
}

fn ground_truth() -> Vec<Range> {
    [
        (0x800, 0x980),
        (0x581c, 0x5d30),
        (0xd8a7, 0xde04),
        (0x2630, 0x2b51),
        (0x2540, 0x2560),
        (0xdf98, 0xe0b1),
        (0xc48c, 0xc4d4),
        (0x3420, 0x3430),
        (0x3c70, 0x3c80),
        (0xe1fc, 0xe334),
        (0xe0e0, 0xe0e4),
        (0x31dc, 0x31e0),
        (0x3cc0, 0x3cc4),
        (0x93f0, 0x9470),
        (0xc648, 0xc6b0),
        (0xcc1c, 0xccb4),
        (0x3c84, 0x3c88),
        (0x55f8, 0x581c),
        (0x501c, 0x518f),
        (0x4a0c, 0x4a37),
        (0x4a5c, 0x4ade),
        (0x7bd0, 0x7c1e),
        (0x7cf0, 0x7d40),
        (0x86f8, 0x8720),
        (0x8970, 0x89d4),
        (0x8c98, 0x8d52),
        (0x8d88, 0x8db4),
        (0x8f44, 0x9065),
        (0x95ec, 0x9704),
        (0x9704, 0x9777),
        (0x9778, 0x978f),
        (0x31ac, 0x31b0),
        (0x325c, 0x3298),
        (0x329c, 0x32a0),
        (0x3364, 0x3368),
        (0x33a8, 0x33ac),
        (0x33f4, 0x33fc),
        (0x3474, 0x347c),
        (0x34a0, 0x34a8),
        (0x34dc, 0x34e8),
        (0x3500, 0x3520),
        (0x3530, 0x3534),
        (0x354c, 0x3564),
    ]
    .into_iter()
    .map(|(lo, hi)| Range { lo, hi })
    .collect()
}

fn known_base_candidates() -> Vec<Range> {
    vec![Range { lo: 0x1a4, hi: 0x400 }, Range { lo: 0x4514, hi: 0x4560 }]
}

fn range_prefers_overlay(image: &[u8], range: Range) -> bool {
    if range.lo == 0x800 {
        return first_window_vector_is_coherent(image, OVERLAY_DELTA)
            && !first_window_vector_is_coherent(image, BASE_DELTA);
    }
    for entry in range.lo..range.hi {
        let base = analyze_entry(image, entry, BASE_DELTA);
        let overlay = analyze_entry(image, entry, OVERLAY_DELTA);
        if matches!((&base, &overlay), (None, Some(_)))
            || matches!((&base, &overlay), (Some(a), Some(b)) if b.score >= a.score + 8)
        {
            return true;
        }
    }
    (range.lo..range.hi.saturating_sub(3)).step_by(4).any(|vma| {
        matches!(
            (word_at(image, vma, BASE_DELTA), word_at(image, vma, OVERLAY_DELTA)),
            (Some(base), Some(overlay)) if !literal_is_sane(base) && literal_is_sane(overlay)
        )
    })
}

fn classify_candidate_ranges(image: &[u8], candidates: &[Range]) -> Vec<Range> {
    let blind = derive_overlay_ranges(image);
    let mut overlay_literals = BTreeSet::new();
    for entry in 0x1a4..0x1_0000 {
        let base = analyze_entry(image, entry, BASE_DELTA);
        let overlay = analyze_entry(image, entry, OVERLAY_DELTA);
        let preferred = match (&base, &overlay) {
            (None, Some(_)) => true,
            (Some(a), Some(b)) => b.score >= a.score + 8,
            _ => false,
        };
        if preferred {
            if let Some(evidence) = overlay {
                overlay_literals.extend(evidence.literals);
            }
        }
    }
    candidates
        .iter()
        .copied()
        .filter(|&range| {
            !KNOWN_BASE_ROOTS.iter().any(|root| (range.lo..range.hi).contains(root))
                && (range_prefers_overlay(image, range)
                    || blind.iter().any(|found| found.lo < range.hi && range.lo < found.hi)
                    || overlay_literals.range(range.lo..range.hi).next().is_some())
        })
        .collect()
}

fn covered_bytes(ranges: &[Range]) -> BTreeSet<u32> {
    ranges.iter().flat_map(|range| range.lo..range.hi).collect()
}

fn assert_calibration_image(raw: &[u8]) {
    assert_eq!(raw.len(), 248_592, "calibration metrics are pinned to Phoenix 1.5.5.391");
    assert_eq!(&raw[0x1d0..0x1e1], b"Release 1.5.5.391");
    assert_eq!(&raw[0x2bc..0x2c4], &[0x9c, 0x02, 0, 0, 0x40, 0x03, 0, 0]);
}

fn load_without_production_overlays(raw: &[u8]) -> FirmwareProcessor {
    let image = FirmwareImage::parse(raw).expect("parse firmware");
    let image_len = image.bytes().len() as u32;
    let segments = super::super::psp_load_map(image_len);
    let mut bus = Bus::new_with_load_offset(image.bytes().to_vec(), segments[0].rom_load_offset());
    let seg_b = &segments[1];
    bus.preload_ram(seg_b.phys_base, &image.bytes()[seg_b.file_range()]);

    let mut cpu = xtensa::interp::Cpu::new(super::super::RESET_ENTRY);
    cpu.mmu = xtensa::mmu::Mmu::new_with_varway56(true);
    cpu.mmu.ptevaddr = 0x3c00_0000;
    cpu.mmu.dtlbcfg = 0x0003_0000;
    super::super::psp_map::install(&mut cpu.mmu, &mut bus, super::super::PSP_LOAD_OFFSET, image_len);
    FirmwareProcessor {
        cpu,
        bus,
        entry: super::super::RESET_ENTRY,
        symbols: super::super::load_symbols(),
        host_mailbox: super::super::host_mailbox::HostMailbox::new(),
    }
}

fn load_with_overlays(raw: &[u8], mut ranges: Vec<Range>) -> FirmwareProcessor {
    ranges.sort_by_key(|range| (range.lo, range.hi));
    for pair in ranges.windows(2) {
        assert!(pair[0].hi <= pair[1].lo, "overlapping overlay assignment: {pair:?}");
    }
    let mut proc = load_without_production_overlays(raw);
    for range in ranges {
        proc.bus.add_rom_overlay(range.lo, range.hi, OVERLAY_DELTA);
    }
    proc
}

fn live_op(proc: &mut FirmwareProcessor) -> decode::Op {
    let pc = proc.cpu.pc;
    let phys = proc
        .cpu
        .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
        .expect("fetch translate");
    let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
    decode::decode(&bytes, pc).op
}

fn image_op(image: &[u8], pc: u32, delta: u32) -> decode::Op {
    decode::decode(image_bytes(image, pc, delta).expect("image bytes"), pc).op
}

#[derive(Debug)]
struct GateOutcome {
    pass: bool,
    stop: String,
}

fn gate_a_publish(raw: &[u8], ranges: Vec<Range>) -> GateOutcome {
    let mut proc = load_with_overlays(raw, ranges);
    let report = proc.boot_to_idle(200_000);
    let magic = proc.bus.load_local32(0x14820);
    GateOutcome {
        pass: report.reached_idle
            && report.wait_reason == Some(WaitReason::Waiti)
            && report.unknown_op.is_none()
            && report.last_pc == 0x5645
            && magic == 0x5550_4e5f,
        stop: format!(
            "idle={} wait={:?} unknown={:?} n={} pc={:#x} magic={magic:#010x}",
            report.reached_idle,
            report.wait_reason,
            report.unknown_op,
            report.instrs_executed,
            report.last_pc,
        ),
    }
}

fn gate_b_line0_processor(mut proc: FirmwareProcessor) -> GateOutcome {
    let report = proc.boot_to_idle(200_000);
    if !report.reached_idle || report.wait_reason != Some(WaitReason::Waiti) || report.last_pc != 0x5645 {
        return GateOutcome { pass: false, stop: format!("precondition failed: {report:?}") };
    }
    if proc.cpu.intenable & 1 == 0 || proc.cpu.regs.intlevel() != 0 || proc.cpu.regs.excm() {
        return GateOutcome {
            pass: false,
            stop: format!(
                "line 0 masked: INTENABLE={:#x} intlevel={} excm={}",
                proc.cpu.intenable,
                proc.cpu.regs.intlevel(),
                proc.cpu.regs.excm()
            ),
        };
    }

    let resume_pc = proc.cpu.pc;
    proc.cpu.interrupt |= 1;
    let mut saw_addmi4 = false;
    let mut saw_addmi5 = false;
    let mut saw_wsr = false;
    let mut saw_retw = false;
    for n in 0..2_000u64 {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let op = live_op(&mut proc);
        let expected = match pc {
            0x8cb1 => {
                saw_addmi4 = matches!(op, decode::Op::Addmi { t: 4, s: 4, imm: 0x1000 });
                saw_addmi4
            }
            0x8cb4 => {
                saw_addmi5 = matches!(op, decode::Op::Addmi { t: 5, s: 5, imm: 0x1000 });
                saw_addmi5
            }
            0x8cb7 => {
                saw_wsr = matches!(op, decode::Op::Wsr { sr: 0xe6, t: 3 });
                saw_wsr
            }
            0x8cba => {
                saw_retw = matches!(op, decode::Op::RetwN);
                saw_retw
            }
            _ => true,
        };
        if !expected {
            return GateOutcome { pass: false, stop: format!("misframed n={n} pc={pc:#x}: {op:?}") };
        }
        let at_retw = pc == 0x8cba;
        let at_rfe = matches!(op, decode::Op::Rfe);
        let step = proc.cpu.step(&mut proc.bus);
        if !matches!(step, Step::Ran | Step::Exception { .. }) {
            return GateOutcome { pass: false, stop: format!("stopped n={n} pc={pc:#x}: {step:?}") };
        }
        if at_retw && proc.cpu.pc & 0x00ff_ffff != 0x7fe4 {
            return GateOutcome {
                pass: false,
                stop: format!("RETW.N returned to {:#x}, not 0x7fe4", proc.cpu.pc),
            };
        }
        if at_rfe {
            let pass = saw_addmi4
                && saw_addmi5
                && saw_wsr
                && saw_retw
                && proc.cpu.pc == resume_pc
                && !proc.cpu.regs.excm();
            return GateOutcome {
                pass,
                stop: format!(
                    "RFE n={n} pc={:#x} addmi4={saw_addmi4} addmi5={saw_addmi5} wsr={saw_wsr} retw={saw_retw}",
                    proc.cpu.pc
                ),
            };
        }
    }
    GateOutcome { pass: false, stop: "post-interrupt budget reached".into() }
}

fn gate_b_line0(raw: &[u8], ranges: Vec<Range>) -> GateOutcome {
    gate_b_line0_processor(load_with_overlays(raw, ranges))
}

fn conflict_assignment(code_overlay: bool, literal_overlay: bool) -> Vec<Range> {
    let mut out = Vec::new();
    for range in ground_truth() {
        match (range.lo, range.hi) {
            (0x8c98, 0x8d52) if !code_overlay => {
                out.push(Range { lo: 0x8c98, hi: 0x8cae });
                out.push(Range { lo: 0x8cbc, hi: 0x8d52 });
            }
            (0x354c, 0x3564) if !literal_overlay => out.push(Range { lo: 0x3550, hi: 0x3564 }),
            _ => out.push(range),
        }
    }
    out
}

fn assert_current_path_landmarks(raw: &[u8]) {
    let mut proc = load_with_overlays(raw, ground_truth());
    let mut fetched = BTreeSet::new();
    let mut candidates = BTreeSet::new();
    let mut calls = BTreeSet::new();
    let mut literals = BTreeSet::new();
    let mut publish_chain = [false; 5];
    let mut service_chain = [false; 4];

    let mut trace_step = |proc: &mut FirmwareProcessor, phase: bool| {
        let pc = proc.cpu.pc;
        let vma = pc & 0x00ff_ffff;
        let phys = proc
            .cpu
            .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
            .expect("fetch translate");
        let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
        let decoded = decode::decode(&bytes, pc);
        let next = pc.wrapping_add(u32::from(decoded.len));
        fetched.insert((vma, vma.wrapping_add(u32::from(decoded.len))));
        candidates.insert(vma);
        candidates.insert(next & 0x00ff_ffff);
        if let Some(target) = direct_call_target(&decoded.op) {
            calls.insert((vma, target & 0x00ff_ffff));
            candidates.insert(target & 0x00ff_ffff);
        }
        if let Some(target) = branch_target(&decoded.op) {
            candidates.insert(target & 0x00ff_ffff);
        }
        if let decode::Op::J { target } = &decoded.op {
            candidates.insert(*target & 0x00ff_ffff);
        }
        if let decode::Op::L32r { target, .. } = &decoded.op {
            literals.insert(*target);
            candidates.insert(*target);
            candidates.insert(target.wrapping_add(4));
        }
        let indirect = match &decoded.op {
            decode::Op::Callx0 { s }
            | decode::Op::Callx4 { s }
            | decode::Op::Callx8 { s }
            | decode::Op::Callx12 { s }
            | decode::Op::Jx { s } => Some(proc.cpu.regs.read_ar(*s)),
            _ => None,
        };
        if !phase {
            publish_chain[0] |= indirect == Some(0x55f8);
            publish_chain[1] |= matches!(&decoded.op, decode::Op::Call8 { target: 0x50d4 }) && vma == 0x55fb;
            publish_chain[2] |= matches!(&decoded.op, decode::Op::Call8 { target: 0x8f44 }) && vma == 0x50f1;
            publish_chain[3] |= matches!(&decoded.op, decode::Op::Call8 { target: 0x8c98 }) && vma == 0x9045;
            publish_chain[4] |= vma == 0x8cb4 && matches!(&decoded.op, decode::Op::MoviN { t: 11, imm: -1 });
        } else {
            service_chain[0] |= matches!(&decoded.op, decode::Op::Call8 { target: 0x8c6c }) && vma == 0x7fe1;
            service_chain[1] |=
                matches!(&decoded.op, decode::Op::L32r { target: 0x354c, .. }) && vma == 0x8c72;
            service_chain[2] |=
                matches!(&decoded.op, decode::Op::Bbci { target: 0x8cae, .. }) && vma == 0x8c8b;
            service_chain[3] |= vma == 0x8cb1 && matches!(&decoded.op, decode::Op::Unknown { .. });
        }
        let step = proc.cpu.step(&mut proc.bus);
        if proc.cpu.pc == next {
            candidates.insert(next & 0x00ff_ffff);
        }
        step
    };

    for _ in 0..200_000u64 {
        match trace_step(&mut proc, false) {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(WaitReason::Waiti) => break,
            other => panic!("pinned publish trace stopped early: {other:?}"),
        }
    }
    assert_eq!(proc.cpu.pc, 0x5645);
    assert_eq!(proc.bus.load_local32(0x14820), 0x5550_4e5f);
    proc.cpu.interrupt |= 1;
    for _ in 0..2_000u64 {
        match trace_step(&mut proc, true) {
            Step::Ran | Step::Exception { .. } => {}
            Step::Unknown { pc: 0x8cb1, .. } => break,
            other => panic!("pinned service trace stopped unexpectedly: {other:?}"),
        }
    }

    drop(trace_step);

    assert!(publish_chain.into_iter().all(|hit| hit), "publish pin chain incomplete: {publish_chain:?}");
    assert!(service_chain.into_iter().all(|hit| hit), "service pin chain incomplete: {service_chain:?}");
    eprintln!(
        "execution candidates: fetched_spans={} boundaries={} direct_calls={} l32r_targets={}",
        fetched.len(),
        candidates.len(),
        calls.len(),
        literals.len()
    );
    let relevant: Vec<u32> = candidates
        .into_iter()
        .filter(|vma| (0x8c68..=0x8cbc).contains(vma) || (0x354c..=0x3550).contains(vma))
        .collect();
    eprintln!("collision-relevant execution boundaries: {relevant:#x?}");
}

#[test]
fn m2c_probe_coherence_mapper() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the coherence mapper");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(&path).expect("read firmware");
    FirmwareImage::parse(&raw).expect("parse firmware");
    assert_calibration_image(&raw);
    let derived = derive_overlay_ranges(&raw);
    let predicted = covered_bytes(&derived);
    let expected = covered_bytes(&ground_truth());
    let true_positive = predicted.intersection(&expected).count();
    let precision = true_positive as f64 / predicted.len().max(1) as f64;
    let recall = true_positive as f64 / expected.len() as f64;

    eprintln!("=== blind derived +0x100 ranges ({}) ===", derived.len());
    for range in &derived {
        eprintln!("{:#06x}-{:#06x} ({} bytes)", range.lo, range.hi, range.hi - range.lo);
    }
    eprintln!(
        "byte precision={precision:.4} recall={recall:.4} tp={true_positive} predicted={} truth={}",
        predicted.len(),
        expected.len()
    );
    let positives = ground_truth();
    let negatives = known_base_candidates();
    let mut calibration_candidates = positives.clone();
    calibration_candidates.extend(&negatives);
    let calibrated = classify_candidate_ranges(&raw, &calibration_candidates);
    eprintln!(
        "candidate-boundary calibration: selected {} positives, {} false positives",
        positives.iter().filter(|range| calibrated.contains(range)).count(),
        negatives.iter().filter(|range| calibrated.contains(range)).count(),
    );
    for range in positives.iter().filter(|range| !calibrated.contains(range)) {
        eprintln!("  unsupported known range: {:#06x}-{:#06x}", range.lo, range.hi);
    }
    assert!(positives.iter().all(|range| calibrated.contains(range)));
    assert!(negatives.iter().all(|range| !calibrated.contains(range)));
}

#[test]
fn m2c_probe_calibrated_map_boots_alive() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to validate the calibrated overlay map");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(path).expect("read firmware");
    assert_calibration_image(&raw);
    let derived = classify_candidate_ranges(&raw, &ground_truth());
    let mut proc = load_without_production_overlays(&raw);
    for range in &derived {
        proc.bus.add_rom_overlay(range.lo, range.hi, OVERLAY_DELTA);
    }
    let report = proc.boot_to_idle(3_000_000);
    let magic = proc.bus.load_local32(0x14820);
    eprintln!(
        "calibrated-map boot: idle={} wait={:?} unknown={:?} instrs={} pc={:#x} local_data[0x14820]={magic:#010x}",
        report.reached_idle,
        report.wait_reason,
        report.unknown_op,
        report.instrs_executed,
        report.last_pc,
    );
    assert!(report.reached_idle, "calibrated map stopped before idle: {report:?}");
    assert_eq!(report.wait_reason, Some(WaitReason::Waiti));
    assert_eq!(report.unknown_op, None);
    assert_eq!(magic, 0x5550_4e5f, "derived map did not publish _NPU");
    assert_eq!(report.last_pc, 0x5645, "derived map did not rest at the expected waiti");
}

#[test]
fn m2c_probe_line0_service_returns() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the line-0 service gate");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(path).expect("read firmware");
    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let outcome = gate_b_line0_processor(FirmwareProcessor::load_m2c(image));
    assert!(outcome.pass, "line-0 service gate failed: {}", outcome.stop);
}

#[test]
fn m2c_probe_execution_guided_framing_search() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the execution-guided framing search");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(path).expect("read firmware");
    assert_calibration_image(&raw);
    assert_current_path_landmarks(&raw);

    // The publish component is rooted by the absolute stored pointer 0x55f8.
    // Its target VMA is fixed: only +0x100 presents an ENTRY there. Every
    // executed PC-relative edge below is then fixed at its encoded target.
    let publish_root_pinned = matches!(image_op(&raw, 0x55f8, OVERLAY_DELTA), decode::Op::Entry { .. })
        && !matches!(image_op(&raw, 0x55f8, BASE_DELTA), decode::Op::Entry { .. });
    assert!(publish_root_pinned);
    let publish_1 = analyze_entry(&raw, 0x50d4, OVERLAY_DELTA).expect("pinned publish entry 0x50d4");
    let publish_1_pinned = publish_root_pinned
        && publish_1.calls.contains(&0x8f44)
        && analyze_entry(&raw, 0x50d4, BASE_DELTA).is_none();
    assert!(publish_1_pinned);
    let publish_2 = analyze_entry(&raw, 0x8f44, OVERLAY_DELTA).expect("pinned publish entry 0x8f44");
    let publish_2_pinned = publish_1_pinned
        && publish_2.calls.contains(&0x8c98)
        && analyze_entry(&raw, 0x8f44, BASE_DELTA).is_none();
    assert!(publish_2_pinned);
    let publish_leaf_pinned = publish_2_pinned
        && analyze_entry(&raw, 0x8c98, OVERLAY_DELTA).is_some()
        && analyze_entry(&raw, 0x8c98, BASE_DELTA).is_none();
    assert!(publish_leaf_pinned);

    // These same physical functions form a self-consistent BASE alias graph,
    // but it starts +0xa4 higher and is unreachable from the pinned 0x55f8
    // absolute root: 0x50d4->0x5178, 0x8f44->0x8fe8, 0x8c98->0x8d3c.
    let alias_1 = analyze_entry(&raw, 0x5178, BASE_DELTA).expect("BASE alias 0x5178");
    assert!(alias_1.calls.contains(&0x8fe8));
    let alias_2 = analyze_entry(&raw, 0x8fe8, BASE_DELTA).expect("BASE alias 0x8fe8");
    assert!(alias_2.calls.contains(&0x8d3c));
    assert!(analyze_entry(&raw, 0x8d3c, BASE_DELTA).is_some());

    // Reset/vector reachability pins the line-0 caller in BASE. Its executed
    // PC-relative target uniquely pins FUN_00008c68 and its tail in BASE.
    let service_caller = analyze_entry(&raw, 0x7fc4, BASE_DELTA).expect("pinned service caller 0x7fc4");
    let service_caller_pinned =
        service_caller.calls.contains(&0x8c6c) && analyze_entry(&raw, 0x7fc4, OVERLAY_DELTA).is_none();
    assert!(service_caller_pinned);
    let service_leaf_pinned = service_caller_pinned
        && analyze_entry(&raw, 0x8c6c, BASE_DELTA).is_some()
        && analyze_entry(&raw, 0x8c6c, OVERLAY_DELTA).is_none();
    assert!(service_leaf_pinned);

    // All caller-framing variables in the backward cones are pinned above.
    // The remaining behavior-equivalent partition has exactly two shared
    // cells, whose boundaries are execution-derived: the taken service-tail
    // target through RETW.N, and the L32R word both paths fetch at 0x354c.
    let candidate_sections = [
        (Range { lo: 0x55f8, hi: 0x581c }, publish_root_pinned, "absolute pointer 0x55f8 -> AT"),
        (publish_1.range, publish_1_pinned, "0x55fb call8 -> 0x50d4 AT"),
        (publish_2.range, publish_2_pinned, "0x50f1 call8 -> 0x8f44 AT"),
        (
            analyze_entry(&raw, 0x8c98, OVERLAY_DELTA).unwrap().range,
            publish_leaf_pinned,
            "0x9045 call8 -> 0x8c98 AT",
        ),
        (service_caller.range, service_caller_pinned, "reset/vector spine -> 0x7fc4 BASE"),
        (
            analyze_entry(&raw, 0x8c6c, BASE_DELTA).unwrap().range,
            service_leaf_pinned,
            "0x7fe1 call8 -> 0x8c6c BASE",
        ),
    ];
    let free_sections: Vec<_> = candidate_sections
        .iter()
        .filter(|(_, pinned, _)| !pinned)
        .map(|(range, _, reason)| (*range, *reason))
        .collect();
    eprintln!("=== execution-guided framing search ===");
    eprintln!("pinned overlay ranges: {}", ground_truth().len());
    for (range, _, reason) in &candidate_sections {
        eprintln!("pinned candidate [{:#x},{:#x}): {reason}", range.lo, range.hi);
    }
    eprintln!("free section variables: {free_sections:?}");
    eprintln!("normalized conflict cells: code [0x8cae,0x8cbc), literal [0x354c,0x3550)");

    let mut solutions = Vec::new();
    for code_overlay in [false, true] {
        for literal_overlay in [false, true] {
            let ranges = conflict_assignment(code_overlay, literal_overlay);
            let gate_a = gate_a_publish(&raw, ranges.clone());
            let gate_b = gate_b_line0(&raw, ranges);
            eprintln!(
                "code={} literal={} | A={} ({}) | B={} ({})",
                if code_overlay { "AT" } else { "BASE" },
                if literal_overlay { "AT" } else { "BASE" },
                gate_a.pass,
                gate_a.stop,
                gate_b.pass,
                gate_b.stop,
            );
            if gate_a.pass && gate_b.pass {
                solutions.push((code_overlay, literal_overlay));
            }
        }
    }
    assert!(solutions.is_empty(), "unexpected single-map solutions: {solutions:?}");
    assert!(free_sections.is_empty(), "unsearched free sections: {free_sections:?}");
}

#[test]
fn m2c_probe_blind_map_boot_frontier() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the blind-map boot frontier");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(path).expect("read firmware");
    assert_calibration_image(&raw);
    let derived = derive_overlay_ranges(&raw);
    let mut proc = load_without_production_overlays(&raw);
    for range in &derived {
        proc.bus.add_rom_overlay(range.lo, range.hi, OVERLAY_DELTA);
    }
    let report = proc.boot_to_idle(200_000);
    eprintln!(
        "blind-map boot: ranges={} idle={} unknown={:?} instrs={} pc={:#x} magic={:#010x}",
        derived.len(),
        report.reached_idle,
        report.unknown_op,
        report.instrs_executed,
        report.last_pc,
        proc.bus.load_local32(0x14820),
    );
}

#[test]
fn m2c_probe_overlay_store_conflicts() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the overlay store audit");
        return;
    }
    let path = std::env::var_os("XDNA_FIRMWARE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin")
        });
    let raw = std::fs::read(path).expect("read firmware");
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    let overlays = ground_truth();
    let mut conflicts = Vec::new();
    let mut executed = 0u64;
    while executed < 200_000 {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let ea = match decode::decode(&bytes, pc).op {
                decode::Op::S32i { s, imm, .. }
                | decode::Op::S32iN { s, imm, .. }
                | decode::Op::S16i { s, imm, .. }
                | decode::Op::S8i { s, imm, .. }
                | decode::Op::S32ri { s, imm, .. }
                | decode::Op::S32c1i { s, imm, .. }
                | decode::Op::S32e { s, imm, .. } => {
                    Some(proc.cpu.regs.read_ar(s).wrapping_add(imm) & 0x00ff_ffff)
                }
                _ => None,
            };
            if let Some(ea) = ea {
                if overlays.iter().any(|range| (range.lo..range.hi).contains(&ea)) {
                    conflicts.push((executed, pc, ea));
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            xtensa::interp::Step::Ran | xtensa::interp::Step::Exception { .. } => executed += 1,
            xtensa::interp::Step::Wait(_) => {
                executed += 1;
                break;
            }
            xtensa::interp::Step::Unknown { .. } => break,
        }
    }
    eprintln!("store audit: instrs={executed} pc={:#x} conflicts={}", proc.cpu.pc, conflicts.len());
    eprintln!("store-conflict VMAs: {conflicts:#x?}");
    assert_eq!(proc.cpu.pc, 0x5645, "audit boot did not reach alive waiti");
}

#[test]
fn synthetic_entry_to_retw_selects_overlay_view() {
    let mut image = vec![0u8; 0x400];
    let pc = 0x1a4u32;
    // entry a1, 0x20; retw.n -- decoder-pinned encodings.
    image[(pc + OVERLAY_DELTA) as usize..(pc + OVERLAY_DELTA + 5) as usize]
        .copy_from_slice(&[0x36, 0x41, 0x00, 0x1d, 0xf0]);

    assert!(analyze_entry(&image, pc, BASE_DELTA).is_none());
    assert_eq!(analyze_entry(&image, pc, OVERLAY_DELTA).map(|e| e.range), Some(Range { lo: pc, hi: pc + 5 }));
}
