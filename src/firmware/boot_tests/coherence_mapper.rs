//! Static two-view coherence mapper for the stripped management firmware.
//!
//! Run the report with:
//! `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_coherence_mapper -- --nocapture`

use super::*;

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

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
fn m2c_probe_isr_remap_hunt() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the ISR remap hunt");
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
    let report = proc.boot_to_idle(200_000);
    assert!(report.reached_idle && report.last_pc == 0x5645, "precondition failed: {report:?}");
    let resume = proc.cpu.pc;
    eprintln!(
        "=== ISR remap hunt === at waiti pc={:#x} intenable={:#x} intlevel={} excm={}",
        resume,
        proc.cpu.intenable,
        proc.cpu.regs.intlevel(),
        proc.cpu.regs.excm()
    );
    proc.cpu.interrupt |= 1;
    let mut trace: Vec<u32> = Vec::new();
    let mut code_stores: Vec<(u64, u32, u32)> = Vec::new(); // (n, store_pc, ea)
    let mut sysops: Vec<(u64, u32, String)> = Vec::new();
    let mut stop = String::from("budget");
    for n in 0..8_000u64 {
        let pc = proc.cpu.pc;
        if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
            let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
            let op = decode::decode(&bytes, pc).op;
            trace.push(pc & 0x00ff_ffff);
            let ea = match op {
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
                // The whole low .text region -- an overlay copy into any code VMA
                // is the signature we are hunting.
                if (0x0400..0x0001_0000).contains(&ea) {
                    code_stores.push((n, pc & 0x00ff_ffff, ea));
                }
            }
            match op {
                decode::Op::Witlb { t, s } | decode::Op::Wdtlb { t, s } => sysops.push((
                    n,
                    pc & 0x00ff_ffff,
                    format!("tlb as={:#x} at={:#x}", proc.cpu.regs.read_ar(s), proc.cpu.regs.read_ar(t)),
                )),
                decode::Op::Iitlb { s } | decode::Op::Idtlb { s } => {
                    sysops.push((n, pc & 0x00ff_ffff, format!("itlb-inv as={:#x}", proc.cpu.regs.read_ar(s))))
                }
                decode::Op::Wsr { sr, t } => sysops.push((
                    n,
                    pc & 0x00ff_ffff,
                    format!("wsr sr={sr:#x} val={:#x}", proc.cpu.regs.read_ar(t)),
                )),
                decode::Op::Unknown { word } => {
                    stop = format!("WALL n={n} pc={:#x} word={word:#x}", pc & 0x00ff_ffff);
                    break;
                }
                _ => {}
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(_) => {
                stop = format!("returned to WAIT n={n}");
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("WALL(step) n={n} pc={pc:#x} word={word:#x}");
                break;
            }
        }
        if proc.cpu.pc == resume {
            stop = format!("returned to resume {resume:#x} n={n}");
            break;
        }
    }
    eprintln!("stop: {stop}");
    eprintln!("ISR path len={} first48={:#x?}", trace.len(), &trace[..trace.len().min(48)]);
    let n_ovl = trace.iter().position(|&p| p == 0x8c6c);
    eprintln!("first hit of 0x8c6c at path index {n_ovl:?}");
    eprintln!("STORES into code region 0x400..0x10000 (overlay-copy signature): {}", code_stores.len());
    for (n, pc, ea) in &code_stores {
        eprintln!("  n={n} store@{pc:#x} -> [{ea:#x}]");
    }
    eprintln!("sysops (witlb/wdtlb/wsr) count={}: {sysops:#x?}", sysops.len());
}

#[test]
fn m2c_probe_natural_isr_chain_reachability() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the natural ISR-chain reachability check");
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
    // The ISR service chain Codex pinned, plus the general-exception plumbing.
    let watch: [u32; 8] = [0x7fc4, 0x7fe1, 0x8c6c, 0x8c72, 0x8c88, 0x8cb1, 0x2958, 0x28b4];
    let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut n = 0u64;
    while n < 200_000 {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        if watch.contains(&pc) {
            first.entry(pc).or_insert(n);
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => break,
            Step::Unknown { .. } => break,
        }
    }
    eprintln!("=== natural-boot reachability of the ISR service chain (n={n}, pc={:#x}) ===", proc.cpu.pc);
    for w in watch {
        eprintln!("  {w:#07x} first@ {:?}", first.get(&w));
    }
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

/// Scratch: does adding the proven-missing AT overlay for the go-alive guard
/// literal at VMA 0x31a4 (BASE=0 / AT=0x27010ac0) move the boot PAST the
/// phantom waiti at 0x5645? Reports the new resting frontier + the distinct
/// pcs visited after 0x5645 so the next mapping gap can be localized.
#[test]
fn m2c_probe_add_31a4_overlay_frontier() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1");
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
    let mut proc = FirmwareProcessor::load_m2c(image);
    // The one proven-missing literal overlay (adjacent to the mapped 0x31ac).
    proc.bus.add_rom_overlay(0x0000_31a4, 0x0000_31a8, OVERLAY_DELTA);
    // Trial: the AT view names the preloaded Segment-B service at 0x08b04428;
    // BASE names the zero-filled phantom at 0x15ff0.
    proc.bus.add_rom_overlay(0x0000_32c8, 0x0000_32cc, OVERLAY_DELTA);
    // Trial: direct call8 target; AT is entry..retw.n, BASE faults immediately.
    proc.bus.add_rom_overlay(0x0000_7c5c, 0x0000_7cee, OVERLAY_DELTA);
    // Trial: direct call8 target from 0x7c62; AT is one function with two
    // return exits before the next entry at 0x7e28. BASE begins mid-body.
    proc.bus.add_rom_overlay(0x0000_7d4c, 0x0000_7e28, OVERLAY_DELTA);
    // Trial: callx4 target from 0x29dd; completes the prefix of the existing
    // syscall-block overlay, which already begins at 0xd8a7.
    proc.bus.add_rom_overlay(0x0000_d864, 0x0000_d8a7, OVERLAY_DELTA);
    // Trial: 0x8952 L32r base pointer. BASE's 0x27200310 makes the bitmap
    // helper store through 0x2a2a1310; AT's 0x000117c0 yields 0x030b27c0.
    proc.bus.add_rom_overlay(0x0000_353c, 0x0000_3540, OVERLAY_DELTA);
    // Trial: direct call8 target from 0xdc25. BASE begins mid-body and later
    // emits the phantom 0xc710 -> 0x26d4 edge; AT begins with entry.
    proc.bus.add_rom_overlay(0x0000_c6b0, 0x0000_c730, OVERLAY_DELTA);
    // Trial: the new c6b0 function's adjacent live pool words. The existing
    // production tuple covers only 0x3c84; these name c6b0 and 0x1186c.
    proc.bus.add_rom_overlay(0x0000_3c88, 0x0000_3c90, OVERLAY_DELTA);

    let mut past_5645 = false;
    let mut seen_after: Vec<u32> = Vec::new();
    let mut seen_set: HashSet<u32> = HashSet::new();
    let mut tail: VecDeque<(u64, u32, String)> = VecDeque::new();
    let mut hits: BTreeMap<u32, (u64, u64, u64)> = BTreeMap::new();
    let mut sram_stores: Vec<(u64, u32, u32, u32)> = Vec::new();
    let mut goalive_visits = 0u32;
    let mut goalive_trace: Vec<String> = Vec::new();
    let mut overlap_trace: Vec<String> = Vec::new();
    let mut exception_trace: Vec<String> = Vec::new();
    let mut key_trace: Vec<String> = Vec::new();
    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let mut n = 0u64;
    let mut prev_pc = 0u32;
    let stop;
    loop {
        if n >= max {
            stop = "budget".to_string();
            break;
        }
        let pc = proc.cpu.pc;
        if matches!(pc, 0x0000_8c98 | 0x0000_8cae | 0x0000_8cb1) {
            let phys = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch);
            overlap_trace.push(format!("n={n} prev={prev_pc:#x} pc={pc:#x} phys={phys:?}"));
        }
        if pc == 0x5645 {
            past_5645 = true;
        }
        if past_5645 && seen_set.insert(pc) && seen_after.len() < 120 {
            seen_after.push(pc);
        }
        if past_5645 {
            if key_trace.len() < 80
                && matches!(pc, 0x8964 | 0xd900 | 0xc48c | 0xc4b5 | 0xc4ca | 0x7f20 | 0x7f22)
            {
                let regs = (0..=15)
                    .map(|r| format!("a{r}={:#010x}", proc.cpu.regs.read_ar(r)))
                    .collect::<Vec<_>>()
                    .join(" ");
                key_trace.push(format!("n={n} pc={pc:#x} {regs}"));
            }
            if pc == 0x5645 {
                goalive_visits += 1;
            }
            if goalive_visits <= 2 && goalive_trace.len() < 160 && (0x5645..0x581c).contains(&pc) {
                let regs = (2..=15)
                    .map(|r| format!("a{r}={:#010x}", proc.cpu.regs.read_ar(r)))
                    .collect::<Vec<_>>()
                    .join(" ");
                goalive_trace.push(format!("n={n} pc={pc:#x} {regs}"));
            }
            let decoded =
                proc.cpu
                    .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
                    .ok()
                    .map(|phys| {
                        let bytes: [u8; 8] =
                            std::array::from_fn(|i| proc.bus.fetch8(pc + i as u32, phys + i as u32));
                        decode::decode(&bytes, pc).op
                    });
            let op = decoded
                .as_ref()
                .map_or_else(|| "<fetch fault>".to_string(), |op| format!("{op:?}"));
            let hit = hits.entry(pc).or_insert((0, n, n));
            hit.0 += 1;
            hit.2 = n;
            if let Some(op) = decoded {
                let store = match op {
                    decode::Op::S32i { t, s, imm }
                    | decode::Op::S32iN { t, s, imm }
                    | decode::Op::S32ri { t, s, imm }
                    | decode::Op::S32c1i { t, s, imm }
                    | decode::Op::S16i { t, s, imm }
                    | decode::Op::S8i { t, s, imm }
                    | decode::Op::S32e { t, s, imm } => {
                        Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t)))
                    }
                    _ => None,
                };
                if let Some((ea, value)) = store {
                    if (0x0308_0000..0x0310_0000).contains(&ea) {
                        sram_stores.push((n, pc, ea, value));
                    }
                }
            }
            if tail.len() == 200 {
                tail.pop_front();
            }
            tail.push_back((n, pc, op));
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran => n += 1,
            Step::Exception { cause, pc: vector } => {
                if past_5645 && exception_trace.len() < 40 {
                    exception_trace.push(format!(
                        "n={n} fault_pc={pc:#x} cause={cause:#x} vector={vector:#x} epc1={:#x} excvaddr={:#x} a2={:#x} a3={:#x} a4={:#x} a5={:#x}",
                        proc.cpu.epc1,
                        proc.cpu.excvaddr,
                        proc.cpu.regs.read_ar(2),
                        proc.cpu.regs.read_ar(3),
                        proc.cpu.regs.read_ar(4),
                        proc.cpu.regs.read_ar(5),
                    ));
                }
                n += 1;
            }
            Step::Wait(r) => {
                stop = format!("Wait({r:?})");
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#010x}");
                break;
            }
        }
        prev_pc = pc;
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("PollSpin {addr:#x}");
            break;
        }
    }
    let pc = proc.cpu.pc;
    let magic = proc.bus.load_local32(0x14820);
    eprintln!("=== +0x31a4 overlay boot ===");
    eprintln!("instrs        = {n}");
    eprintln!("stop          = {stop}");
    eprintln!("last_pc       = {pc:#x} ({})", nearest_symbol(&proc.symbols, pc));
    eprintln!("passed 0x5645 = {past_5645}");
    eprintln!("magic[0x14820]= {magic:#010x}");
    eprintln!("distinct pcs visited after first 0x5645 ({}):", seen_after.len());
    for p in &seen_after {
        eprintln!("  {p:#07x} ({})", nearest_symbol(&proc.symbols, *p));
    }
    eprintln!("last {} executed pcs:", tail.len());
    for (n, p, op) in tail {
        eprintln!("  n={n} pc={p:#010x} {op}");
    }
    let mut hottest: Vec<_> = hits
        .iter()
        .map(|(&pc, &(count, first, last))| (count, pc, first, last))
        .collect();
    hottest.sort_unstable_by(|a, b| b.cmp(a));
    eprintln!("hottest recurring pcs:");
    for (count, pc, first, last) in hottest.into_iter().take(20) {
        eprintln!("  pc={pc:#010x} count={count} first={first} last={last}");
    }
    for pc in [0x26d4, 0x50e8, 0x5127, 0x55f8, 0x5645, 0xc6b0, 0xdc25] {
        eprintln!("watch pc={pc:#x}: {:?}", hits.get(&pc));
    }
    eprintln!("first two goalive-tail visits:");
    for line in goalive_trace {
        eprintln!("  {line}");
    }
    eprintln!("SRAM-band stores after 0x5645: {}", sram_stores.len());
    for event in sram_stores.iter().take(40) {
        eprintln!("  n={} pc={:#x} EA={:#010x} value={:#010x}", event.0, event.1, event.2, event.3);
    }
    let alive = proc.cpu.data_read32(&mut proc.bus, 0x030b_f000);
    eprintln!("terminal FW_ALIVE_OFF read: {alive:?}");
    eprintln!("0x8c98 overlap translations:");
    for line in overlap_trace {
        eprintln!("  {line}");
    }
    eprintln!("first post-goalive exceptions:");
    for line in exception_trace {
        eprintln!("  {line}");
    }
    eprintln!("post-goalive key-pc register trace:");
    for line in key_trace {
        eprintln!("  {line}");
    }
    // Decode goalive tail in AT framing (these VMAs live in the 0x55f8..0x581c
    // +0x100 overlay; peek8 uses BASE, so read raw at vaddr+0x100 directly).
    eprintln!("goalive tail 0x5645..0x5660, AT framing (file = vaddr+0x100):");
    let mut q = 0x5645u32;
    while q < 0x5660 {
        let f = (q + 0x100) as usize;
        let b = raw.get(f..f + 6).unwrap_or(&[]);
        let d = decode::decode(&b[..b.len().min(3)], q);
        let len = (d.len.max(1) as usize).min(b.len());
        eprintln!("  {q:#07x} bytes={:02x?} op={:?} len={}", &b[..len], d.op, d.len);
        q += d.len.max(1) as u32;
    }
    // The callx8 target comes from L32r [0x32c8]. Dump that literal both framings
    // and, for each, decode the entry at the pointer it yields.
    for (name, off) in [("BASE(+0x5c)", 0x5cu32), ("AT(+0x100)", 0x100u32)] {
        let f = (0x32c8u32 + off) as usize;
        let lit = raw.get(f..f + 4).map(|w| u32::from_le_bytes(w.try_into().unwrap()));
        eprint!("  lit[0x32c8] {name} file={f:#x} = {:?}", lit.map(|v| format!("{v:#010x}")));
        // decode entry at that pointer, AT framing
        if let Some(ptr) = lit {
            let tf = (ptr + 0x100) as usize;
            if let Some(tb) = raw.get(tf..tf + 6) {
                eprint!("  -> AT@{:#x} bytes={:02x?} op={:?}", tf, tb, decode::decode(&tb[..3], ptr).op);
            }
        }
        eprintln!();
    }
}
