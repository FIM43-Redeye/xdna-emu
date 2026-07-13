//! Static two-view coherence mapper for the stripped management firmware.
//!
//! Run the report with:
//! `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_coherence_mapper -- --nocapture`

use super::*;

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::Write as _;

const BASE_DELTA: u32 = 0x5c;
const OVERLAY_DELTA: u32 = 0x100;
const KNOWN_BASE_ROOTS: [u32; 2] = [super::super::RESET_ENTRY, 0x4525];
const COLLISION_REGION: Range = Range { lo: 0x8c98, hi: 0x8d52 };
const APPROACH_REGION: Range = Range { lo: 0x8c6c, hi: 0x8d52 };
const ALIVE_DESCRIPTOR: [u32; 16] = [
    0x030e_c000,
    0x030e_c004,
    0x030b_c000,
    0x0000_0400,
    0x030e_d000,
    0x030e_d004,
    0x030b_d000,
    0x0000_0400,
    0x5550_4e5f,
    0x0000_000e,
    0x0000_0005,
    0x0000_0008,
    0,
    0,
    0,
    0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Range {
    lo: u32,
    hi: u32,
}

#[derive(Debug, Clone, Copy)]
struct SearchRegion {
    id: &'static str,
    range: Range,
    entry: u32,
    requires_entry: bool,
}

const SEARCH_REGIONS: [SearchRegion; 8] = [
    SearchRegion { id: "local", range: COLLISION_REGION, entry: 0x8c98, requires_entry: true },
    SearchRegion {
        id: "p55f8",
        range: Range { lo: 0x55f8, hi: 0x581c },
        entry: 0x55f8,
        requires_entry: true,
    },
    // Exact production-overlay bounds keep this candidate from being shadowed.
    SearchRegion {
        id: "p50d4",
        range: Range { lo: 0x501c, hi: 0x518f },
        entry: 0x50d4,
        requires_entry: true,
    },
    SearchRegion {
        id: "p8f44",
        range: Range { lo: 0x8f44, hi: 0x9065 },
        entry: 0x8f44,
        requires_entry: true,
    },
    SearchRegion {
        id: "s8770",
        range: Range { lo: 0x8770, hi: 0x87eb },
        entry: 0x8770,
        requires_entry: false,
    },
    SearchRegion {
        id: "sc530",
        range: Range { lo: 0xc530, hi: 0xc583 },
        entry: 0xc530,
        requires_entry: true,
    },
    SearchRegion {
        id: "s7fc4",
        range: Range { lo: 0x7fc4, hi: 0x801f },
        entry: 0x7fc4,
        requires_entry: true,
    },
    SearchRegion {
        id: "s8c6c",
        range: Range { lo: 0x8c6c, hi: 0x8c98 },
        entry: 0x8c6c,
        requires_entry: true,
    },
];

fn search_region(id: &str) -> SearchRegion {
    SEARCH_REGIONS
        .iter()
        .copied()
        .find(|region| region.id == id)
        .unwrap_or_else(|| panic!("unknown XDNA_FW_REGION {id:?}"))
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

fn live_op(proc: &mut FirmwareProcessor) -> decode::Op {
    let pc = proc.cpu.pc;
    let phys = proc
        .cpu
        .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
        .expect("fetch translate");
    let bytes: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
    decode::decode(&bytes, pc).op
}

#[derive(Debug)]
struct GateOutcome {
    pass: bool,
    stop: String,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SplitCandidate {
    delta_lo: u32,
    delta_hi: u32,
    split: u32,
    literal_delta: u32,
    local_delta: u32,
}

#[derive(Debug)]
struct SplitOutcome {
    publisher_pass: bool,
    service_entered: bool,
    service_pass: bool,
    inconclusive: bool,
    stop_kind: String,
    stop_pc: u32,
    stop_word: u32,
    publisher_boundaries: BTreeSet<(u32, u8)>,
    service_boundaries: BTreeSet<(u32, u8)>,
    last_region: String,
    descriptor_store_mask: u16,
    alive_store: bool,
    final_alive: u32,
    tail: String,
}

fn parse_probe_u32(text: &str) -> u32 {
    let text = text.trim();
    text.strip_prefix("0x").map_or_else(
        || text.parse().expect("decimal probe value"),
        |hex| u32::from_str_radix(hex, 16).expect("hex probe value"),
    )
}

fn parse_split_candidate_spec(spec: &str) -> SplitCandidate {
    let fields: Vec<_> = spec.split(':').map(parse_probe_u32).collect();
    assert!(
        matches!(fields.len(), 4 | 5),
        "XDNA_FW_ONLY entries must be delta_lo:delta_hi:split:literal_delta[:local_delta]"
    );
    SplitCandidate {
        delta_lo: fields[0],
        delta_hi: fields[1],
        split: fields[2],
        literal_delta: fields[3],
        local_delta: fields.get(4).copied().unwrap_or(OVERLAY_DELTA),
    }
}

fn probe_value_tag(values: &[u32]) -> String {
    values.iter().map(|value| format!("{value:x}")).collect::<Vec<_>>().join("-")
}

fn canonical_split_candidate(region: SearchRegion, mut candidate: SplitCandidate) -> SplitCandidate {
    if candidate.delta_lo == candidate.delta_hi || candidate.split == region.range.hi {
        candidate.delta_hi = candidate.delta_lo;
        candidate.split = region.range.hi;
    } else if candidate.split == region.range.lo {
        candidate.delta_lo = candidate.delta_hi;
        candidate.split = region.range.hi;
    }
    candidate
}

fn split_candidates(region: SearchRegion, deltas: &[u32], local_deltas: &[u32]) -> Vec<SplitCandidate> {
    let mut candidates = BTreeSet::new();
    let literal_deltas: &[u32] = &[BASE_DELTA, OVERLAY_DELTA];
    let local_deltas: &[u32] = if region.range == COLLISION_REGION {
        &[OVERLAY_DELTA]
    } else {
        local_deltas
    };
    for &local_delta in local_deltas {
        for &literal_delta in literal_deltas {
            for &delta_lo in deltas {
                for &delta_hi in deltas {
                    for split in region.range.lo..=region.range.hi {
                        candidates.insert(canonical_split_candidate(
                            region,
                            SplitCandidate { delta_lo, delta_hi, split, literal_delta, local_delta },
                        ));
                    }
                }
            }
        }
    }
    candidates.into_iter().collect()
}

fn load_split_candidate(raw: &[u8], region: SearchRegion, candidate: SplitCandidate) -> FirmwareProcessor {
    let fits = |lo: u32, hi: u32, delta: u32| {
        lo == hi || hi.checked_add(delta).is_some_and(|file_hi| file_hi as usize <= raw.len())
    };
    assert!(fits(region.range.lo, candidate.split, candidate.delta_lo));
    assert!(fits(candidate.split, region.range.hi, candidate.delta_hi));

    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(raw).expect("parse firmware"));
    proc.bus.remove_rom_overlay(region.range.lo, region.range.hi);
    if candidate.split != region.range.lo {
        proc.bus.add_rom_overlay(region.range.lo, candidate.split, candidate.delta_lo);
    }
    if candidate.split != region.range.hi {
        proc.bus.add_rom_overlay(candidate.split, region.range.hi, candidate.delta_hi);
    }

    assert!(fits(0x354c, 0x3550, candidate.literal_delta));
    proc.bus.remove_rom_overlay(0x354c, 0x3564);
    proc.bus.add_rom_overlay(0x354c, 0x3550, candidate.literal_delta);
    proc.bus.add_rom_overlay(0x3550, 0x3564, OVERLAY_DELTA);
    if region.range != COLLISION_REGION {
        assert!(fits(COLLISION_REGION.lo, COLLISION_REGION.hi, candidate.local_delta));
        proc.bus.remove_rom_overlay(COLLISION_REGION.lo, COLLISION_REGION.hi);
        proc.bus
            .add_rom_overlay(COLLISION_REGION.lo, COLLISION_REGION.hi, candidate.local_delta);
    }
    proc
}

fn candidate_delta(region: SearchRegion, candidate: SplitCandidate, vma: u32) -> u32 {
    if (region.range.lo..region.range.hi).contains(&vma) && vma < candidate.split {
        candidate.delta_lo
    } else if (region.range.lo..region.range.hi).contains(&vma) {
        candidate.delta_hi
    } else if (COLLISION_REGION.lo..COLLISION_REGION.hi).contains(&vma) {
        candidate.local_delta
    } else {
        BASE_DELTA
    }
}

fn literal_crosses_candidate_seam(region: SearchRegion, candidate: SplitCandidate, target: u32) -> bool {
    let first = candidate_delta(region, candidate, target);
    (1..4).any(|byte| candidate_delta(region, candidate, target.wrapping_add(byte)) != first)
}

fn instruction_crosses_candidate_seam(
    region: SearchRegion,
    candidate: SplitCandidate,
    full_pc: u32,
    len: u8,
) -> bool {
    let pc = full_pc & 0x00ff_ffff;
    if full_pc != pc {
        return false;
    }
    let first = candidate_delta(region, candidate, pc);
    (1..u32::from(len)).any(|byte| candidate_delta(region, candidate, pc.wrapping_add(byte)) != first)
}

fn split_literal_target(region: SearchRegion, candidate: SplitCandidate, op: &decode::Op) -> Option<u32> {
    match op {
        decode::Op::L32r { target, .. } if literal_crosses_candidate_seam(region, candidate, *target) => {
            Some(*target)
        }
        decode::Op::Flix1 { ops } => ops.iter().find_map(|op| split_literal_target(region, candidate, op)),
        _ => None,
    }
}

fn access_overlaps(addr: u32, width: u8, lo: u32, hi: u32) -> bool {
    (0..width).any(|byte| (lo..hi).contains(&addr.wrapping_add(u32::from(byte))))
}

#[test]
fn split_candidates_are_canonical_and_piecewise() {
    assert_eq!(
        parse_split_candidate_spec("0x100:0:0x8cae:0x5c").local_delta,
        OVERLAY_DELTA,
        "legacy four-field XDNA_FW_ONLY specs keep the production local view",
    );
    assert_eq!(parse_split_candidate_spec("0x100:0:0x8f47:0x100:0x5c").local_delta, BASE_DELTA,);
    let local = search_region("local");
    let candidates = split_candidates(local, &[0, BASE_DELTA, OVERLAY_DELTA], &[OVERLAY_DELTA]);
    assert_eq!(candidates.len(), 2_226);
    assert!(candidates.iter().all(|candidate| {
        candidate.split == local.range.hi
            || (local.range.lo < candidate.split
                && candidate.split < local.range.hi
                && candidate.delta_lo != candidate.delta_hi)
    }));

    let candidate = SplitCandidate {
        delta_lo: OVERLAY_DELTA,
        delta_hi: 0,
        split: 0x8cae,
        literal_delta: BASE_DELTA,
        local_delta: OVERLAY_DELTA,
    };
    assert_eq!(candidate_delta(local, candidate, 0x8c97), BASE_DELTA);
    assert_eq!(candidate_delta(local, candidate, 0x8cad), OVERLAY_DELTA);
    assert_eq!(candidate_delta(local, candidate, 0x8cae), 0);
    assert_eq!(candidate_delta(local, candidate, 0x8d52), BASE_DELTA);
    assert!(!literal_crosses_candidate_seam(local, candidate, 0x8ca8));
    assert!(literal_crosses_candidate_seam(local, candidate, 0x8cac));
    assert!(literal_crosses_candidate_seam(local, candidate, 0x8d50));
    let bundle = decode::Op::Flix1 { ops: vec![decode::Op::Nop, decode::Op::L32r { t: 2, target: 0x8cac }] };
    assert_eq!(split_literal_target(local, candidate, &bundle), Some(0x8cac));

    let upstream = search_region("s8c6c");
    assert_eq!(upstream.entry, 0x8c6c);
    assert!(upstream.requires_entry);
    assert!(!search_region("s8770").requires_entry);
    let upstream_candidates =
        split_candidates(upstream, &[0, BASE_DELTA, OVERLAY_DELTA, 0x244], &[BASE_DELTA, OVERLAY_DELTA]);
    assert_eq!(upstream_candidates.len(), 2_080);
    assert!(upstream_candidates
        .iter()
        .any(|candidate| candidate.literal_delta == BASE_DELTA));
    assert!(upstream_candidates
        .iter()
        .any(|candidate| candidate.literal_delta == OVERLAY_DELTA));
    assert!(upstream_candidates.iter().any(|candidate| candidate.local_delta == BASE_DELTA));
    assert!(upstream_candidates
        .iter()
        .any(|candidate| candidate.local_delta == OVERLAY_DELTA));
    let split_inside_entry = SplitCandidate {
        delta_lo: OVERLAY_DELTA,
        delta_hi: 0,
        split: 0x8f46,
        literal_delta: OVERLAY_DELTA,
        local_delta: BASE_DELTA,
    };
    let publisher_helper = search_region("p8f44");
    assert!(instruction_crosses_candidate_seam(publisher_helper, split_inside_entry, 0x8f44, 3));
    assert!(!instruction_crosses_candidate_seam(publisher_helper, split_inside_entry, 0x2000_8f44, 3,));
    assert!(!instruction_crosses_candidate_seam(publisher_helper, split_inside_entry, 0x8f46, 2));
    assert!(instruction_crosses_candidate_seam(
        publisher_helper,
        SplitCandidate {
            delta_lo: OVERLAY_DELTA,
            delta_hi: OVERLAY_DELTA,
            split: publisher_helper.range.hi,
            literal_delta: OVERLAY_DELTA,
            local_delta: BASE_DELTA,
        },
        publisher_helper.range.hi - 1,
        2,
    ));
    assert_eq!(
        candidate_delta(
            upstream,
            SplitCandidate {
                delta_lo: 0,
                delta_hi: 0x244,
                split: 0x8c80,
                literal_delta: OVERLAY_DELTA,
                local_delta: BASE_DELTA,
            },
            0x8c7f,
        ),
        0,
    );
    assert_eq!(
        candidate_delta(
            upstream,
            SplitCandidate {
                delta_lo: 0,
                delta_hi: 0x244,
                split: 0x8c80,
                literal_delta: OVERLAY_DELTA,
                local_delta: BASE_DELTA,
            },
            0x8c98,
        ),
        BASE_DELTA,
    );
    assert!(access_overlaps(0x030b_afff, 2, 0x030b_b000, 0x030b_b004));
    assert!(!access_overlaps(0x030b_affc, 4, 0x030b_b000, 0x030b_b004));
}

fn read_alive_state(proc: &mut FirmwareProcessor) -> ([u32; 16], u32) {
    let descriptor = std::array::from_fn(|i| proc.bus.data_load32(0x030b_b000 + i as u32 * 4));
    let alive = proc.bus.data_load32(0x030b_f000);
    (descriptor, alive)
}

fn execution_fingerprint(
    proc: &FirmwareProcessor,
    opaque_epoch: u64,
    memory_shadow: &BTreeMap<u32, u8>,
) -> Vec<u32> {
    // Diagnostic access-log length is deliberately absent: it cannot affect
    // firmware execution and would make every otherwise-identical state unique.
    // The caller disables recurrence once a store can no longer be mirrored
    // exactly in `memory_shadow`.
    let mut state = Vec::with_capacity(600);
    state.extend([opaque_epoch as u32, (opaque_epoch >> 32) as u32, proc.cpu.pc]);
    state.extend((0..64).map(|reg| proc.cpu.regs.read_ar(reg)));
    state.extend([
        proc.cpu.regs.windowbase,
        proc.cpu.regs.windowstart,
        proc.cpu.regs.sar,
        proc.cpu.regs.ps,
        proc.cpu.regs.lbeg,
        proc.cpu.regs.lend,
        proc.cpu.regs.lcount,
        proc.cpu.regs.exccause,
        proc.cpu.vecbase,
        proc.cpu.epc1,
        proc.cpu.excvaddr,
        proc.cpu.threadptr,
        proc.cpu.interrupt,
        proc.cpu.intenable,
        proc.cpu.scompare1,
        u32::from(proc.cpu.halted),
    ]);
    state.extend(proc.cpu.excsave);
    state.extend(proc.cpu.fr);
    state.extend([
        proc.cpu.mmu.ptevaddr,
        proc.cpu.mmu.rasid,
        proc.cpu.mmu.itlbcfg,
        proc.cpu.mmu.dtlbcfg,
        proc.cpu.mmu.autorefill_idx,
        u32::from(proc.cpu.mmu.varway56),
    ]);
    for entry in proc.cpu.mmu.itlb.iter().flatten().chain(proc.cpu.mmu.dtlb.iter().flatten()) {
        state.extend([
            entry.vaddr,
            entry.paddr,
            u32::from(entry.asid),
            u32::from(entry.attr),
            u32::from(entry.variable),
        ]);
    }
    state.push(memory_shadow.len() as u32);
    for (&addr, &value) in memory_shadow {
        state.extend([addr, u32::from(value)]);
    }
    state
}

fn run_split_candidate(
    raw: &[u8],
    region: SearchRegion,
    candidate: SplitCandidate,
    max: u64,
    trace: bool,
) -> SplitOutcome {
    let mut proc = load_split_candidate(raw, region, candidate);
    let mut publisher_boundaries = BTreeSet::new();
    let mut service_boundaries = BTreeSet::new();
    let mut publisher_pass = false;
    let mut service_entered = false;
    let mut descriptor_store_mask = 0u16;
    let mut alive_store = false;
    let mut last_region = String::from("none");
    let mut n = 0u64;
    let mut publisher_entered = false;
    let mut previous_retired_pc = None;
    let mut opaque_epoch = 0u64;
    let mut memory_shadow = BTreeMap::new();
    let mut recent_states = BTreeMap::<u32, VecDeque<Vec<u32>>>::new();
    let mut pc_visits = BTreeMap::<u32, u16>::new();
    let mut trace_events = 0usize;
    let mut tail = VecDeque::new();
    let (stop_kind, stop_pc, stop_word, inconclusive, service_pass) = loop {
        if n >= max {
            break ("budget".to_string(), proc.cpu.pc, 0, true, false);
        }

        let full_pc = proc.cpu.pc;
        let pc = full_pc & 0x00ff_ffff;
        publisher_entered |= if region.range == COLLISION_REGION {
            pc == COLLISION_REGION.lo && previous_retired_pc == Some(0x9045)
        } else {
            pc == 0x55f8
        };
        if publisher_entered {
            // The fill fastpath retires many stores behind one decoded op,
            // which the recurrence shadow cannot observe individually.
            proc.cpu.fastpath_enabled = false;
        }
        let service_edge = if region.range == COLLISION_REGION {
            pc == 0x8c6c && previous_retired_pc == Some(0x7fe1)
        } else {
            pc == 0x8770 && previous_retired_pc == Some(0x283b)
        };
        if service_edge && !service_entered {
            service_entered = true;
            descriptor_store_mask = 0;
            alive_store = false;
        }
        if publisher_entered {
            let visits = pc_visits.entry(full_pc).or_default();
            *visits = visits.saturating_add(1);
            if *visits > 8 && opaque_epoch == 0 && memory_shadow.len() <= 1024 {
                let state = execution_fingerprint(&proc, opaque_epoch, &memory_shadow);
                let states = recent_states.entry(full_pc).or_default();
                if states.contains(&state) {
                    break ("state-cycle".to_string(), full_pc, 0, false, false);
                }
                if trace && trace_events < 128 {
                    if let Some(previous) = states.back() {
                        let differences: Vec<_> = previous
                            .iter()
                            .zip(&state)
                            .enumerate()
                            .filter_map(|(i, (&old, &new))| (old != new).then_some((i, old, new)))
                            .take(8)
                            .collect();
                        eprintln!("n={n} pc={full_pc:#x} state differences={differences:#x?}");
                    }
                }
                if states.len() == 32 {
                    states.pop_front();
                }
                states.push_back(state);
            }
        }
        if publisher_entered && pc == 0x5645 && proc.bus.load_local32(0x14820) == 0x5550_4e5f {
            publisher_pass = true;
        }

        let fetch = proc.cpu.translate(&mut proc.bus, full_pc, xtensa::interp::Access::Fetch);
        let (step, decoded_store, self_loop, conditional_store) = match fetch {
            Ok(phys) => {
                let bytes: [u8; 8] =
                    std::array::from_fn(|i| proc.bus.fetch8(full_pc + i as u32, phys + i as u32));
                let decoded = decode::decode(&bytes, full_pc);
                if full_pc == pc
                    && pc == region.entry
                    && region.requires_entry
                    && !matches!(decoded.op, decode::Op::Entry { .. })
                {
                    let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
                    break ("root-mismatch".to_string(), full_pc, word, false, false);
                }
                if instruction_crosses_candidate_seam(region, candidate, full_pc, decoded.len) {
                    break ("split-instruction".to_string(), full_pc, candidate.split, false, false);
                }
                if publisher_entered {
                    if tail.len() == 32 {
                        tail.pop_front();
                    }
                    tail.push_back(format!("n={n} pc={full_pc:#x} {:?}", decoded.op));
                }
                if publisher_entered
                    && full_pc == pc
                    && (APPROACH_REGION.lo..APPROACH_REGION.hi).contains(&pc)
                {
                    let boundaries = if service_entered {
                        &mut service_boundaries
                    } else {
                        &mut publisher_boundaries
                    };
                    boundaries.insert((pc, decoded.len));
                    let len = usize::from(decoded.len).min(bytes.len());
                    let files: Vec<_> = (0..len)
                        .map(|i| {
                            let vma = pc + i as u32;
                            vma.wrapping_add(candidate_delta(region, candidate, vma))
                        })
                        .collect();
                    last_region = format!(
                        "{} pc={pc:#x} len={} bytes={:02x?} files={files:x?} op={:?}",
                        if service_entered { "service" } else { "publisher" },
                        decoded.len,
                        &bytes[..len],
                        decoded.op,
                    );
                    if trace && trace_events < 128 {
                        eprintln!("n={n} {last_region}");
                        trace_events += 1;
                    }
                }
                if let Some(target) = split_literal_target(region, candidate, &decoded.op) {
                    break ("split-literal".to_string(), full_pc, target, true, false);
                }
                if publisher_entered
                    && matches!(&decoded.op, decode::Op::Flix1 { ops } if ops.iter().any(|op| decoded_store(&proc, op).is_some()))
                {
                    break ("flix-store".to_string(), full_pc, 0, true, false);
                }
                let store = decoded_store(&proc, &decoded.op);
                let self_loop = matches!(decoded.op, decode::Op::J { target } if target == full_pc);
                let conditional_store = matches!(decoded.op, decode::Op::S32c1i { .. });
                (proc.cpu.step(&mut proc.bus), store, self_loop, conditional_store)
            }
            Err(step) => (step, None, false, false),
        };

        if publisher_entered && matches!(step, Step::Ran) {
            if conditional_store {
                opaque_epoch += 1;
            }
            if let Some((ea, value, width)) = decoded_store {
                if let Ok(pa) = proc.cpu.translate(&mut proc.bus, ea, xtensa::interp::Access::Store) {
                    if !conditional_store {
                        for i in 0..width {
                            let addr = pa.wrapping_add(u32::from(i));
                            if memory_shadow.len() < 1024 || memory_shadow.contains_key(&addr) {
                                memory_shadow.insert(addr, (value >> (8 * i)) as u8);
                            } else {
                                opaque_epoch += 1;
                            }
                        }
                    }
                    for (i, addr) in (0x030b_b000..0x030b_b040).step_by(4).enumerate() {
                        if access_overlaps(ea, width, addr, addr + 4)
                            || access_overlaps(pa, width, addr, addr + 4)
                        {
                            descriptor_store_mask |= 1 << i;
                        }
                    }
                    alive_store |= access_overlaps(ea, width, 0x030b_f000, 0x030b_f004)
                        || access_overlaps(pa, width, 0x030b_f000, 0x030b_f004);
                }
            }
        }

        if matches!(step, Step::Ran) {
            previous_retired_pc = Some(pc);
        }
        n += 1;
        if service_entered && (n & 0x3f == 0 || (descriptor_store_mask != 0 && alive_store)) {
            let (descriptor, alive) = read_alive_state(&mut proc);
            if descriptor == ALIVE_DESCRIPTOR && alive == 0x030b_b000 {
                break ("published".to_string(), proc.cpu.pc, 0, false, true);
            }
        }
        if self_loop && matches!(step, Step::Ran) {
            break ("self-loop".to_string(), full_pc, 0, false, false);
        }
        if let Some(addr) = proc.bus.sysstub().spinning() {
            break (format!("sysstub-spin-{addr:#x}"), proc.cpu.pc, 0, false, false);
        }
        match step {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(reason) => {
                break (format!("wait-{reason:?}"), proc.cpu.pc, 0, false, false);
            }
            Step::Unknown { pc, word } => {
                // `Unknown` includes both invalid encodings and valid Xtensa
                // instructions the interpreter does not implement. Without an
                // independent decode oracle it cannot falsify a candidate.
                break ("unknown".to_string(), pc, word, true, false);
            }
        }
    };

    let (final_descriptor, final_alive) = read_alive_state(&mut proc);
    let service_pass = service_pass
        || (service_entered && final_descriptor == ALIVE_DESCRIPTOR && final_alive == 0x030b_b000);
    SplitOutcome {
        publisher_pass,
        service_entered,
        service_pass,
        inconclusive,
        stop_kind,
        stop_pc,
        stop_word,
        publisher_boundaries,
        service_boundaries,
        last_region,
        descriptor_store_mask,
        alive_store,
        final_alive,
        tail: tail.into_iter().collect::<Vec<_>>().join(" | "),
    }
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
    let region_id = std::env::var("XDNA_FW_REGION").unwrap_or_else(|_| "local".into());
    let region = search_region(&region_id);
    let deltas = std::env::var("XDNA_FW_DELTAS")
        .unwrap_or_else(|_| "0,0x5c,0x100".into())
        .split(',')
        .map(parse_probe_u32)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let local_deltas = std::env::var("XDNA_FW_LOCAL_DELTAS")
        .unwrap_or_else(|_| "0x5c,0x100".into())
        .split(',')
        .map(parse_probe_u32)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut candidates = split_candidates(region, &deltas, &local_deltas);
    let only_spec = std::env::var("XDNA_FW_ONLY").ok();
    if let Some(only) = &only_spec {
        let selected: BTreeSet<_> = only.split(';').map(parse_split_candidate_spec).collect();
        candidates.retain(|candidate| selected.contains(candidate));
        assert_eq!(candidates.len(), selected.len(), "XDNA_FW_ONLY selected a non-canonical candidate");
    }
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(100_000);
    let trace_all = std::env::var("XDNA_FW_TRACE").is_ok();
    let jobs = std::env::var("XDNA_FW_JOBS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let worker_count = jobs.max(1).min(candidates.len().max(1));
    // Runtime is highly skewed: invalid maps fail immediately while coherent
    // loops consume the full budget. Striding avoids assigning one worker a
    // contiguous block of all the slow deltas.
    let mut evaluated = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..worker_count)
            .map(|worker| {
                let candidates = &candidates;
                let raw = &raw;
                scope.spawn(move || {
                    candidates
                        .iter()
                        .copied()
                        .skip(worker)
                        .step_by(worker_count)
                        .map(|candidate| {
                            (candidate, run_split_candidate(raw, region, candidate, max, trace_all))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("delta-split worker panicked"))
            .collect::<Vec<_>>()
    });
    evaluated.sort_by_key(|(candidate, _)| *candidate);

    let mut table = String::from(
        "region\tdelta_lo\tdelta_hi\tsplit\tliteral_delta\tlocal_delta\tpublisher_pass\tservice_entered\tservice_pass\tinconclusive\tstop_kind\tstop_pc\tstop_word\tfailing_cone\tpublisher_boundaries\tservice_boundaries\tdescriptor_store_mask\talive_store\tfinal_alive\tlast_region\ttail\n",
    );
    let mut failures = BTreeMap::<(String, u32), usize>::new();
    let mut solutions = Vec::new();
    let mut inconclusive = 0usize;
    for (candidate, outcome) in evaluated {
        let cone = if !outcome.publisher_pass {
            "publisher"
        } else if !outcome.service_entered {
            "pre-service"
        } else {
            "service"
        };
        *failures
            .entry((format!("{cone}:{}", outcome.stop_kind), outcome.stop_pc))
            .or_default() += 1;
        inconclusive += usize::from(outcome.inconclusive);
        if outcome.publisher_pass && outcome.service_pass {
            solutions.push(candidate);
        }
        writeln!(
            table,
            "{}\t{:#x}\t{:#x}\t{:#x}\t{:#x}\t{:#x}\t{}\t{}\t{}\t{}\t{}\t{:#x}\t{:#x}\t{}\t{:x?}\t{:x?}\t{:#06x}\t{}\t{:#x}\t{}\t{}",
            region.id,
            candidate.delta_lo,
            candidate.delta_hi,
            candidate.split,
            candidate.literal_delta,
            candidate.local_delta,
            outcome.publisher_pass,
            outcome.service_entered,
            outcome.service_pass,
            outcome.inconclusive,
            outcome.stop_kind,
            outcome.stop_pc,
            outcome.stop_word,
            cone,
            outcome.publisher_boundaries,
            outcome.service_boundaries,
            outcome.descriptor_store_mask,
            outcome.alive_store,
            outcome.final_alive,
            outcome.last_region,
            outcome.tail,
        )
        .unwrap();
    }

    let output_name = std::env::var("XDNA_FW_OUTPUT").unwrap_or_else(|_| {
        if region.id == "local" {
            "delta-split-search.tsv".to_string()
        } else {
            let only_suffix = if only_spec.is_some() { "-only" } else { "" };
            format!(
                "delta-split-search-{}-d{}-l{}-m{max}{only_suffix}.tsv",
                region.id,
                probe_value_tag(&deltas),
                probe_value_tag(&local_deltas),
            )
        }
    });
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("build/experiments/firmware-re")
        .join(output_name);
    std::fs::create_dir_all(output.parent().unwrap()).expect("create framing-search output directory");
    std::fs::write(&output, table).expect("write framing-search table");

    eprintln!("=== delta x split execution search ===");
    eprintln!(
        "region={} [{:#x},{:#x}) deltas={deltas:#x?} local_deltas={local_deltas:#x?}",
        region.id, region.range.lo, region.range.hi,
    );
    let literal_deltas = [BASE_DELTA, OVERLAY_DELTA];
    eprintln!(
        "canonical candidates={} literal_deltas={literal_deltas:#x?} solutions={} inconclusive={inconclusive}",
        candidates.len(),
        solutions.len(),
    );
    for ((failure, pc), count) in &failures {
        eprintln!("{count:4} {failure} at {pc:#x}");
    }
    eprintln!("machine-readable table: {}", output.display());
    for &candidate in &solutions {
        eprintln!("=== solution {candidate:#x?} ===");
        let outcome = run_split_candidate(&raw, region, candidate, max, true);
        eprintln!("publisher boundaries: {:#x?}", outcome.publisher_boundaries);
        eprintln!("service boundaries: {:#x?}", outcome.service_boundaries);
    }
    // Keep a bounded run from being reported as an exhaustion proof. The TSV
    // is written first so every unresolved candidate remains inspectable.
    assert_eq!(inconclusive, 0, "candidate executions remain inconclusive; exhaustion is not proven");
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

/// Observation-only hunt for firmware programming of a bulk low-code reload.
/// Records the entire natural boot's MMIO writes and CPU stores whose values
/// look like a low-code destination or a Segment-B/code-alias source.
#[test]
fn m2c_probe_reload_programming_audit() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the reload-programming audit");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    proc.bus.arm_probe();

    let is_low_code = |value: u32| (0x0000_8000..0x0001_0000).contains(&value);
    let is_copy_source = |value: u32| {
        (0x08b0_0000..0x08b0_fa10).contains(&value) || (0x2000_0000..0x2001_0000).contains(&value)
    };
    let mut pointer_stores = Vec::new();
    let mut n = 0u64;
    let stop = loop {
        if n == 100_000 {
            break "budget".to_string();
        }
        let full_pc = proc.cpu.pc;
        proc.bus.set_probe_pc(full_pc);
        let store = proc
            .cpu
            .translate(&mut proc.bus, full_pc, xtensa::interp::Access::Fetch)
            .ok()
            .and_then(|phys| {
                let bytes: [u8; 8] =
                    std::array::from_fn(|i| proc.bus.fetch8(full_pc + i as u32, phys + i as u32));
                decoded_store(&proc, &decode::decode(&bytes, full_pc).op)
            });
        let step = proc.cpu.step(&mut proc.bus);
        if matches!(step, Step::Ran) {
            if let Some((ea, value, width)) =
                store.filter(|(_, value, _)| is_low_code(*value) || is_copy_source(*value))
            {
                pointer_stores.push((n, full_pc, ea, value, width));
            }
        }
        n += 1;
        match step {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(reason) => break format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc),
            Step::Unknown { pc, word } => break format!("Unknown pc={pc:#x} word={word:#x}"),
        }
    };

    let accesses = proc.bus.take_probe();
    let mmio_writes: Vec<_> = accesses.iter().filter(|access| access.is_write).collect();
    let pointer_mmio_writes: Vec<_> = mmio_writes
        .iter()
        .copied()
        .filter(|access| is_low_code(access.value) || is_copy_source(access.value))
        .collect();

    let backing_pairs = |words: &[(u32, u32)]| {
        let sources: Vec<_> = words.iter().copied().filter(|&(_, value)| is_copy_source(value)).collect();
        let destinations: Vec<_> = words.iter().copied().filter(|&(_, value)| is_low_code(value)).collect();
        sources
            .iter()
            .flat_map(|&(source_addr, source)| {
                destinations.iter().filter_map(move |&(dest_addr, dest)| {
                    (source_addr.abs_diff(dest_addr) <= 128).then_some((source_addr, source, dest_addr, dest))
                })
            })
            .collect::<Vec<_>>()
    };
    let local_words: Vec<_> = (0..0x0002_0000u32)
        .step_by(4)
        .map(|addr| (addr, proc.bus.load_local32(addr)))
        .collect();
    let ram_words: Vec<_> = (0x08b0_0000..0x08b0_fa10u32)
        .step_by(4)
        .map(|addr| (addr, proc.bus.data_load32(addr)))
        .collect();
    let local_pairs = backing_pairs(&local_words);
    let ram_pairs = backing_pairs(&ram_words);

    eprintln!("=== low-code reload programming audit ===");
    eprintln!(
        "n={n} stop={stop} mmio_accesses={} mmio_writes={} pointer_mmio_writes={} pointer_cpu_stores={}",
        accesses.len(),
        mmio_writes.len(),
        pointer_mmio_writes.len(),
        pointer_stores.len(),
    );
    for access in &pointer_mmio_writes {
        eprintln!("MMIO {access:#x?}");
    }
    for store in pointer_stores.iter().take(24) {
        eprintln!("CPU_STORE {store:#x?}");
    }
    eprintln!(
        "nearby local source/dest pairs (<=128 bytes): count={} sample={:#x?}",
        local_pairs.len(),
        &local_pairs[..local_pairs.len().min(16)],
    );
    eprintln!(
        "nearby RAM source/dest pairs (<=128 bytes): count={} sample={:#x?}",
        ram_pairs.len(),
        &ram_pairs[..ram_pairs.len().min(16)],
    );

    assert_eq!(stop, "Unknown pc=0x8cb1 word=0x61a800", "audit did not reach the production wall");
    assert_eq!(proc.bus.load_local32(0x14820), 0x5550_4e5f, "audit regressed the publisher landmark");
}

/// Reproducible image-side check for a plain Segment-B copy source or a
/// static `{Segment-B pointer, low-code page}` descriptor pair.
#[test]
fn m2c_probe_reload_source_scan() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1 to run the reload-source scan");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    assert_calibration_image(&raw);
    let segment_b = &raw[0x2d100..0x3cb10];
    let count = |haystack: &[u8], needle: &[u8]| {
        haystack.windows(needle.len()).filter(|window| *window == needle).count()
    };
    let patterns = [
        ("service-addi", &raw[0x8d0a..0x8d0d]),
        ("publisher-bgeu", &raw[0x8dac..0x8daf]),
        ("service-root16", &raw[0x8cc8..0x8cd8]),
        ("publisher-root16", &raw[0x8d98..0x8da8]),
    ];
    for (name, pattern) in patterns {
        let whole = count(&raw, pattern);
        let in_segment_b = count(segment_b, pattern);
        eprintln!("{name}: whole={whole} segment_b={in_segment_b} bytes={pattern:02x?}");
        assert_eq!(whole, 1, "{name} is not unique in the firmware image");
        assert_eq!(in_segment_b, 0, "{name} unexpectedly has a Segment-B copy source");
    }

    let bytewise_low_words = raw
        .windows(4)
        .filter(|bytes| {
            (0x0000_8c00..0x0000_8e00).contains(&u32::from_le_bytes((*bytes).try_into().unwrap()))
        })
        .count();
    let words: Vec<_> = raw
        .chunks_exact(4)
        .enumerate()
        .map(|(index, bytes)| (index * 4, u32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    let sources: Vec<_> = words
        .iter()
        .copied()
        .filter(|&(_, value)| (0x08b0_0000..0x08b0_fa10).contains(&value))
        .collect();
    let pages: Vec<_> = words.iter().copied().filter(|&(_, value)| value == 0x8000).collect();
    let nearby_pairs = sources
        .iter()
        .flat_map(|&(source_offset, source)| {
            pages.iter().filter_map(move |&(page_offset, page)| {
                (source_offset.abs_diff(page_offset) <= 128).then_some((
                    source_offset,
                    source,
                    page_offset,
                    page,
                ))
            })
        })
        .collect::<Vec<_>>();
    eprintln!(
        "bytewise_low_words={bytewise_low_words} aligned_segment_b_pointers={} aligned_0x8000_words={} nearby_pairs={}",
        sources.len(),
        pages.len(),
        nearby_pairs.len(),
    );
    assert_eq!(bytewise_low_words, 0);
    assert_eq!(sources.len(), 379);
    assert_eq!(pages.len(), 8);
    assert!(nearby_pairs.is_empty());
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

/// Distinguish a repeated fixed-pool item, a periodic worker, and a missing
/// external input across executions of `goalive_runfn`. The first local
/// processor reconstructs the old truncated queue-tail overlay; the second uses
/// the production mapping that completed the tail. No firmware memory,
/// interrupt, or scheduler state is changed.
#[test]
fn m2c_probe_goalive_loop_discriminator() {
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
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    proc.bus.remove_rom_overlay(0xcc1c, 0xccc1);
    proc.bus.add_rom_overlay(0xcc1c, 0xccb4, OVERLAY_DELTA);

    const GOALIVE: u32 = 0x55f8;
    const CUR_TASK: u32 = 0x2278;
    const RECORD: std::ops::Range<u32> = 0x2320..0x2334;
    const POOL_CTL: std::ops::Range<u32> = 0x24c4..0x24c7;
    const WORK_ITEM: std::ops::Range<u32> = 0x15fc0..0x15fd4;

    let class = |ea: u32| match ea {
        0x0308_0000..=0x030f_ffff => "host-sram",
        0x2700_0000..=0x27ff_ffff => "mailbox/mmio",
        0x0000_0000..=0x03ff_ffff => "local",
        0x0400_0000..=0x26ff_ffff | 0x2800_0000..=0xffff_ffff => "array/ram/ddr/system",
    };
    let branch_regs = |op: &decode::Op, proc: &FirmwareProcessor| {
        use decode::Op::*;
        match op {
            Beqz { s, .. }
            | Bnez { s, .. }
            | Bltz { s, .. }
            | Bgez { s, .. }
            | BeqzN { s, .. }
            | BnezN { s, .. }
            | Beqi { s, .. }
            | Bnei { s, .. }
            | Blti { s, .. }
            | Bgei { s, .. }
            | Bltui { s, .. }
            | Bgeui { s, .. }
            | Bbci { s, .. }
            | Bbsi { s, .. }
            | Loop { s, .. }
            | Loopnez { s, .. } => format!("a{s}={:#010x}", proc.cpu.regs.read_ar(*s)),
            Beq { s, t, .. }
            | Bne { s, t, .. }
            | Blt { s, t, .. }
            | Bltu { s, t, .. }
            | Bge { s, t, .. }
            | Bgeu { s, t, .. }
            | Bbc { s, t, .. }
            | Bbs { s, t, .. }
            | Bnone { s, t, .. }
            | Bany { s, t, .. }
            | Ball { s, t, .. }
            | Bnall { s, t, .. } => {
                format!("a{s}={:#010x} a{t}={:#010x}", proc.cpu.regs.read_ar(*s), proc.cpu.regs.read_ar(*t))
            }
            _ => String::new(),
        }
    };

    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000u64);
    let mut n = 0u64;
    let mut prev_pc = 0u32;
    let mut entries = 0u32;
    let mut snapshots = Vec::new();
    let mut queue_events = Vec::new();
    let mut current_changes = Vec::new();
    let mut seam_events = Vec::new();
    let mut gate_events = Vec::new();
    let mut alive_events = Vec::new();
    let mut loads = Vec::new();
    let mut branches = Vec::new();
    let mut last_current = proc.bus.load_local32(CUR_TASK);

    while n < max {
        let pc = proc.cpu.pc & 0x00ff_ffff;
        if pc == GOALIVE {
            entries += 1;
            let current = proc.bus.load_local32(CUR_TASK);
            let ctl: Vec<u8> = POOL_CTL.clone().map(|a| proc.bus.load_local8(a)).collect();
            let record: Vec<u32> = RECORD.clone().step_by(4).map(|a| proc.bus.load_local32(a)).collect();
            let work_item: Vec<u32> =
                WORK_ITEM.clone().step_by(4).map(|a| proc.bus.load_local32(a)).collect();
            let task: Vec<u32> = (0..=0x30)
                .step_by(4)
                .map(|off| proc.bus.load_local32(current.wrapping_add(off)))
                .collect();
            snapshots.push(format!(
                "entry={entries} n={n} prev={prev_pc:#x} a0={:#010x} a2={:#010x} \
                 current[{CUR_TASK:#x}]={current:#010x} pool[count,cursor,aux]={ctl:02x?} \
                 record[{:#x}..{:#x}]={record:08x?} work_item[{:#x}..{:#x}]={work_item:08x?} \
                 current_words[+0..+0x30]={task:08x?}",
                proc.cpu.regs.read_ar(0),
                proc.cpu.regs.read_ar(2),
                RECORD.start,
                RECORD.end,
                WORK_ITEM.start,
                WORK_ITEM.end,
            ));
        }

        let phys = proc
            .cpu
            .translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
            .expect("frontier instruction fetch translates");
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(proc.cpu.pc + i as u32, phys + i as u32));
        let op = decode::decode(&bytes, proc.cpu.pc).op;
        if matches!(pc, 0x560d | 0x5044) {
            alive_events.push(format!("n={n} pc={pc:#08x} {op:?}"));
        }

        let load = match &op {
            decode::Op::L32iN { t, s, imm }
            | decode::Op::L32i { t, s, imm }
            | decode::Op::L8ui { t, s, imm }
            | decode::Op::L16ui { t, s, imm }
            | decode::Op::L16si { t, s, imm }
            | decode::Op::L32e { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), *t, None, format!("{op:?}")))
            }
            decode::Op::L32r { t, target } => Some((*target, *t, None, format!("{op:?}"))),
            decode::Op::Lsi { ft, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), 0, Some(*ft), format!("{op:?}")))
            }
            decode::Op::S32c1i { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), *t, None, format!("{op:?} atomic-load")))
            }
            _ => None,
        };
        let branch = branch_target(&op).map(|target| (target, format!("{op:?}"), branch_regs(&op, &proc)));
        let seam = (0xccb3..0xccc1).contains(&pc).then(|| {
            format!(
                "n={n} pc={pc:#08x} a0={:#010x} a2={:#010x} a3={:#010x} {op:?}",
                proc.cpu.regs.read_ar(0),
                proc.cpu.regs.read_ar(2),
                proc.cpu.regs.read_ar(3),
            )
        });

        let watched_store = match &op {
            decode::Op::S32i { t, s, imm }
            | decode::Op::S32iN { t, s, imm }
            | decode::Op::S32ri { t, s, imm }
            | decode::Op::S32c1i { t, s, imm }
            | decode::Op::S32e { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t), 4))
            }
            decode::Op::S16i { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xffff, 2))
            }
            decode::Op::S8i { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xff, 1))
            }
            _ => None,
        };
        if let Some((ea, value, width)) = watched_store {
            if matches!(pc, 0x50c6 | 0x50c9 | 0x50cc | 0x50cf) {
                alive_events
                    .push(format!("n={n} pc={pc:#08x} STORE{width} EA={ea:#x} value={value:#010x} {op:?}"));
            }
            if RECORD.contains(&ea) || POOL_CTL.contains(&ea) || WORK_ITEM.contains(&ea) || ea == CUR_TASK {
                queue_events.push(format!(
                    "n={n} pc={pc:#08x} STORE{width} EA={ea:#010x} value={value:#010x} {op:?}"
                ));
            }
        }

        let step = proc.cpu.step(&mut proc.bus);
        n += 1;
        let next_pc = proc.cpu.pc & 0x00ff_ffff;
        if let Some(seam) = seam {
            seam_events.push(format!("{seam} -> next={next_pc:#08x}"));
        }
        if let Some((ea, t, ft, text)) = load {
            let value = ft.map_or_else(|| proc.cpu.regs.read_ar(t), |f| proc.cpu.fr[f as usize]);
            if matches!(pc, 0x50ba | 0x563e) {
                gate_events
                    .push(format!("n={} pc={pc:#08x} EA={ea:#010x} value={value:#010x} {text}", n - 1,));
            }
            if (1..3).contains(&entries) {
                loads.push(format!(
                    "n={} pc={pc:#08x} EA={ea:#010x} value={value:#010x} class={} {text}",
                    n - 1,
                    class(ea),
                ));
            }
            if RECORD.contains(&ea) || POOL_CTL.contains(&ea) || WORK_ITEM.contains(&ea) || ea == CUR_TASK {
                queue_events
                    .push(format!("n={} pc={pc:#08x} LOAD EA={ea:#010x} value={value:#010x} {text}", n - 1,));
            }
        }
        if (1..3).contains(&entries) {
            if let Some((target, text, regs)) = branch {
                if pc == 0x5640 {
                    gate_events.push(format!(
                        "n={} pc=0x005640 next={next_pc:#08x} target={target:#08x} taken={} {regs} {text}",
                        n - 1,
                        next_pc == target,
                    ));
                }
                branches.push(format!(
                    "n={} pc={pc:#08x} next={next_pc:#08x} target={target:#08x} taken={} {regs} {text}",
                    n - 1,
                    next_pc == target,
                ));
            }
        }
        let current = proc.bus.load_local32(CUR_TASK);
        if current != last_current {
            current_changes.push(format!(
                "n={} after_pc={pc:#08x} current[{CUR_TASK:#x}] {last_current:#010x}->{current:#010x}",
                n - 1,
            ));
            last_current = current;
        }
        match step {
            Step::Ran | Step::Exception { .. } => {}
            other => panic!("loop discriminator stopped at n={n} after {entries} entries: {other:?}"),
        }
        prev_pc = pc;
    }

    eprintln!("=== goalive loop discriminator: truncated queue tail through full horizon ===");
    eprintln!(
        "pre-fix terminal: n={n} goalive_entries={entries} pool_count={:#x} work_flag[0x15fcb]={:#x}",
        proc.bus.load_local8(0x24c4),
        proc.bus.load_local8(0x15fcb),
    );
    eprintln!("-- entry snapshots --");
    for line in &snapshots {
        eprintln!("{line}");
    }
    eprintln!("-- queue/current ownership events through the full horizon --");
    for line in &queue_events {
        eprintln!("{line}");
    }
    for line in &current_changes {
        eprintln!("{line}");
    }
    eprintln!("-- executed queue-overlay seam at 0xccb3 --");
    for line in &seam_events {
        eprintln!("{line}");
    }
    eprintln!("-- 0x27010ac0 gate --");
    for line in &gate_events {
        eprintln!("{line}");
    }
    eprintln!("-- executed alive publisher --");
    for line in &alive_events {
        eprintln!("{line}");
    }
    eprintln!("-- every load between entry 1 and entry 3 --");
    for line in &loads {
        eprintln!("{line}");
    }
    eprintln!("-- conditional edges between entry 1 and entry 3 --");
    for line in &branches {
        eprintln!("{line}");
    }

    assert!(entries >= 3, "did not reach the third goalive entry within XDNA_FW_MAX={max}");

    // Production arm: the empty branch at 0xcc2e targets 0xccb3 inside the
    // completed AT tail, which clears the stale work-item valid bit.
    let mut trial = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse trial firmware"));
    let mut trial_n = 0u64;
    let mut trial_entries = 0u32;
    let mut empty_edges = Vec::new();
    let mut clear_events = Vec::new();
    let mut trial_stop = "budget".to_string();
    while trial_n < max {
        let pc = trial.cpu.pc & 0x00ff_ffff;
        if pc == GOALIVE {
            trial_entries += 1;
        }
        if pc == 0xcc2e {
            empty_edges.push(format!(
                "n={trial_n} pc=0xcc2e count_reg_a6={:#x} flag[0x15fcb]={:#x}",
                trial.cpu.regs.read_ar(6),
                trial.bus.load_local8(0x15fcb),
            ));
        }
        if pc == 0xccbc {
            clear_events.push(format!(
                "n={trial_n} pc=0xccbc S8i a3,[a2+11] EA={:#x} value={:#x} before={:#x}",
                trial.cpu.regs.read_ar(2).wrapping_add(11),
                trial.cpu.regs.read_ar(3) & 0xff,
                trial.bus.load_local8(trial.cpu.regs.read_ar(2).wrapping_add(11)),
            ));
        }
        match trial.cpu.step(&mut trial.bus) {
            Step::Ran | Step::Exception { .. } => trial_n += 1,
            Step::Wait(r) => {
                trial_stop = format!("Wait({r:?})");
                break;
            }
            Step::Unknown { pc, word } => {
                trial_stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
    }
    eprintln!("-- production queue-tail overlay [0xcc1c,0xccc1) --");
    eprintln!(
        "n={trial_n} stop={trial_stop} goalive_entries={trial_entries} pool_count={:#x} work_flag[0x15fcb]={:#x}",
        trial.bus.load_local8(0x24c4),
        trial.bus.load_local8(0x15fcb),
    );
    for line in &empty_edges {
        eprintln!("{line}");
    }
    for line in &clear_events {
        eprintln!("{line}");
    }
    assert_eq!(trial_entries, 1, "completing the queue-pop AT tail did not eliminate repeated dispatch");
    assert_eq!(trial.bus.load_local8(0x15fcb) & 1, 0, "queue-empty tail did not clear work-item valid bit");
}

/// Adjudicate the FW_ALIVE writer on the current post-go-alive frontier.
/// Observation only: the sole test-local mapping is the already-proved queue
/// tail extension that advances the natural boot to the 0x8cb1 frontier.
#[test]
fn m2c_probe_alive_publish_mechanism() {
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
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    proc.bus.add_rom_overlay(0xccb4, 0xccc1, OVERLAY_DELTA);

    const SRAM: std::ops::Range<u32> = 0x030b_0000..0x030c_0000;
    const ALIVE: u32 = 0x030b_f000;
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000u64);
    let literal_hits = |value: u32| {
        let needle = value.to_le_bytes();
        raw.windows(4)
            .enumerate()
            .filter_map(|(off, bytes)| (bytes == needle).then_some(off))
            .collect::<Vec<_>>()
    };

    let mut sram_stores = Vec::new();
    let mut alive_stores = Vec::new();
    let mut publish_trace = Vec::new();
    let mut suspicious_loads = Vec::new();
    let mut dtlb_ops = Vec::new();
    let mut dtlb_at_publish = Vec::new();
    let mut va0_pa = None;
    let mut exception_causes: BTreeMap<u32, u64> = BTreeMap::new();
    let mut controller_enables = Vec::new();
    let mut service_landmarks = Vec::new();
    let mut n = 0u64;
    let stop;

    loop {
        if n >= max {
            stop = format!("budget {max}");
            break;
        }
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let fetch_pa = proc
            .cpu
            .translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
            .expect("frontier fetch translates");
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(proc.cpu.pc + i as u32, fetch_pa + i as u32));
        let op = decode::decode(&bytes, proc.cpu.pc).op;

        if matches!(pc, 0x560d | 0x5044 | 0x50ba | 0x50c6 | 0x50c9 | 0x50cc | 0x50cf) {
            publish_trace.push(format!("n={n} pc={pc:#08x} {op:?}"));
        }
        match &op {
            decode::Op::Wdtlb { t, s } => dtlb_ops.push(format!(
                "n={n} pc={pc:#08x} WDTLB at={:#010x} as={:#010x}",
                proc.cpu.regs.read_ar(*t),
                proc.cpu.regs.read_ar(*s),
            )),
            decode::Op::Idtlb { s } => {
                dtlb_ops.push(format!("n={n} pc={pc:#08x} IDTLB as={:#010x}", proc.cpu.regs.read_ar(*s),))
            }
            _ => {}
        }
        if matches!(pc, 0x86f8 | 0x871c) {
            controller_enables.push(format!(
                "n={n} pc={pc:#08x} a10={:#x} a11={:#x} a12={:#x} a13={:#x} mask_a2={:#010x}",
                proc.cpu.regs.read_ar(10),
                proc.cpu.regs.read_ar(11),
                proc.cpu.regs.read_ar(12),
                proc.cpu.regs.read_ar(13),
                proc.cpu.regs.read_ar(2),
            ));
        }
        if matches!(pc, 0x2958 | 0xd864 | 0x8784 | 0x7fe1 | 0x8c6c | 0x8cb1) {
            service_landmarks.push(format!(
                "n={n} pc={pc:#08x} EXCCAUSE={:#x} EPC1={:#x} INTERRUPT={:#x}",
                proc.cpu.regs.exccause, proc.cpu.epc1, proc.cpu.interrupt,
            ));
        }

        let store = match &op {
            decode::Op::S32i { t, s, imm }
            | decode::Op::S32iN { t, s, imm }
            | decode::Op::S32ri { t, s, imm }
            | decode::Op::S32c1i { t, s, imm }
            | decode::Op::S32e { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t), 4))
            }
            decode::Op::Ssi { ft, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.fr[*ft as usize], 4))
            }
            decode::Op::S16i { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xffff, 2))
            }
            decode::Op::S8i { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xff, 1))
            }
            _ => None,
        };
        if let Some((ea, value, width)) = store {
            let hit = proc.cpu.mmu.lookup(ea, true);
            let pa = proc
                .cpu
                .translate(&mut proc.bus, ea, xtensa::interp::Access::Store)
                .expect("store translates");
            let line = format!(
                "n={n} pc={pc:#08x} STORE{width} EA={ea:#010x} -> PA={pa:#010x} value={value:#010x} hit={hit:?}"
            );
            if SRAM.contains(&ea) || SRAM.contains(&pa) {
                sram_stores.push(line.clone());
            }
            if ea == ALIVE || pa == ALIVE {
                alive_stores.push(line.clone());
            }
            if matches!(pc, 0x50c6 | 0x50c9 | 0x50cc | 0x50cf) {
                publish_trace.push(line);
            }
            if pc == 0x50c6 {
                va0_pa = Some(pa);
                dtlb_at_publish.push(format!(
                    "DTLBCFG={:#010x} PTEVADDR={:#010x} RASID={:#010x} VA0-hit={hit:?} VA0-PA={pa:#010x}",
                    proc.cpu.mmu.dtlbcfg, proc.cpu.mmu.ptevaddr, proc.cpu.mmu.rasid,
                ));
                for (wi, way) in proc.cpu.mmu.dtlb.iter().enumerate() {
                    for (ei, entry) in way.iter().enumerate().filter(|(_, entry)| entry.asid != 0) {
                        dtlb_at_publish.push(format!("dtlb[{wi}][{ei}]={entry:?}"));
                    }
                }
            }
        }

        let load = match &op {
            decode::Op::L32i { t, s, imm } | decode::Op::L32iN { t, s, imm } => {
                Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), *t))
            }
            _ => None,
        };
        let step = proc.cpu.step(&mut proc.bus);
        if let Some((ea, t)) = load {
            if matches!(pc, 0x897d | 0x89b1) {
                suspicious_loads.push(format!(
                    "n={n} pc={pc:#08x} EA={ea:#010x} consumed={:#010x} {op:?}",
                    proc.cpu.regs.read_ar(t),
                ));
            }
        }
        n += 1;
        match step {
            Step::Ran => {}
            Step::Exception { cause, .. } => *exception_causes.entry(cause).or_default() += 1,
            Step::Wait(reason) => {
                stop = format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc & 0x00ff_ffff);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
    }

    eprintln!("=== FW_ALIVE publish mechanism ===");
    eprintln!("n={n} stop={stop} INTENABLE={:#x} INTERRUPT={:#x}", proc.cpu.intenable, proc.cpu.interrupt);
    eprintln!("raw literal 0x030bf000 hits={:?}", literal_hits(ALIVE));
    eprintln!("raw literal 0x030bb000 hits={:?}", literal_hits(0x030b_b000));
    eprintln!(
        "raw 0x31bc BASE={:?} AT={:?}",
        word_at(&raw, 0x31bc, BASE_DELTA),
        word_at(&raw, 0x31bc, OVERLAY_DELTA)
    );
    eprintln!("-- executed 0x560d -> 0x5044 candidate publisher --");
    for line in &publish_trace {
        eprintln!("{line}");
    }
    eprintln!("-- DTLB operations before publication --");
    for line in &dtlb_ops {
        eprintln!("{line}");
    }
    eprintln!("-- DTLB state at pc=0x50c6 --");
    for line in &dtlb_at_publish {
        eprintln!("{line}");
    }
    eprintln!("-- executed 0x897d/0x89b1 loads --");
    for line in &suspicious_loads {
        eprintln!("{line}");
    }
    eprintln!("-- every SRAM-band store (EA or PA in 0x030b0000..0x030c0000) --");
    for line in &sram_stores {
        eprintln!("{line}");
    }
    eprintln!("exact FW_ALIVE stores={}", alive_stores.len());
    eprintln!("natural exception causes={exception_causes:#x?}");
    eprintln!("-- natural syscall/line-0 service landmarks --");
    for line in &service_landmarks {
        eprintln!("{line}");
    }
    eprintln!("-- executed controller enables --");
    for line in &controller_enables {
        eprintln!("{line}");
    }
    for source in [0x2du32, 0x2e] {
        let slot = proc.bus.load_local8(0x11700 + source) as u32;
        let record = 0x110b0 + slot * 0x14;
        eprintln!(
            "source={source:#x} slot={slot} record={record:#x} handler={:#x} arg={:#x}",
            proc.bus.load_local32(record + 0xc),
            proc.bus.load_local32(record + 0x10),
        );
    }

    assert_eq!(va0_pa, Some(0), "VA 0 no longer translates to PA 0 at the candidate publish");
    assert!(alive_stores.is_empty(), "current natural boot unexpectedly wrote FW_ALIVE_OFF");
    assert_eq!(exception_causes.get(&xtensa::interp::EXCCAUSE_LEVEL1_INTERRUPT), None);
}

/// Fork-A investigation (2026-07-11): is the runtime code-view selector at the
/// 0x8cae publisher/service collision an ITLB remap (visible, faithfully
/// modelable) or something the MMU never touches (a true HW bank / external)?
///
/// The publisher (rooted 0x8c98, AT) and the service (rooted 0x8c6c, BASE,
/// entered via syscall) both use the cell at VMA 0x8cae. In hardware the two
/// need DIFFERENT physical bytes there. A page remap CAN supply that (two
/// physical banks, one page each, selected by ITLB) -- but ONLY if the firmware
/// actually reprograms the ITLB for that page between the two executions. This
/// probe answers exactly that, read-only:
///   1. logs every executed Witlb/Iitlb (va, data) -- did any touch the 0x8cxx
///      code page?
///   2. samples the ITLB translation of VMA 0x8cae at the publisher entry and at
///      the service wall -- is the PA the SAME or does it move?
/// SAME PA + no remap => MMU is not the selector => true HW bank / PSP-RE.
/// Different PA / a remap => the selector is in the trace => model it faithfully.
#[test]
fn m2c_probe_itlb_code_view_selector() {
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
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    proc.bus.add_rom_overlay(0xccb4, 0xccc1, OVERLAY_DELTA);
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000u64);

    // The collision cell we care about, and the code page it lives in.
    const CELL: u32 = 0x8cae;
    const CODE_PAGE: std::ops::Range<u32> = 0x0000_8000..0x0000_9000;

    let itlb_pa_of = |proc: &mut FirmwareProcessor, va: u32| -> String {
        let hit = proc.cpu.mmu.lookup(va, false);
        let pa = proc.cpu.translate(&mut proc.bus, va, xtensa::interp::Access::Fetch);
        match hit {
            Ok(h) => {
                let e = proc.cpu.mmu.itlb[h.wi][h.ei];
                format!("PA={pa:?} via itlb[{}][{}] entry={e:?}", h.wi, h.ei)
            }
            Err(c) => format!("PA={pa:?} lookup-miss={c:#x}"),
        }
    };

    let mut itlb_ops: Vec<String> = Vec::new();
    let mut code_page_itlb_ops: Vec<String> = Vec::new();
    let mut cell_samples: Vec<String> = Vec::new();
    let mut n = 0u64;
    let stop;

    loop {
        if n >= max {
            stop = format!("budget {max}");
            break;
        }
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let fetch_pa = match proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
            Ok(pa) => pa,
            Err(c) => {
                stop = format!("fetch xlate fail pc={pc:#x} cause={c:?}");
                break;
            }
        };
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(proc.cpu.pc + i as u32, fetch_pa + i as u32));
        let op = decode::decode(&bytes, proc.cpu.pc).op;

        // Log ITLB modifications (WITLB at,as => va in ar[s], data in ar[t]).
        match &op {
            decode::Op::Witlb { t, s } => {
                let va = proc.cpu.regs.read_ar(*s);
                let data = proc.cpu.regs.read_ar(*t);
                let line = format!("n={n} pc={pc:#08x} WITLB va={va:#010x} data={data:#010x}");
                if CODE_PAGE.contains(&va) {
                    code_page_itlb_ops.push(line.clone());
                }
                itlb_ops.push(line);
            }
            decode::Op::Iitlb { s } => {
                let va = proc.cpu.regs.read_ar(*s);
                let line = format!("n={n} pc={pc:#08x} IITLB va={va:#010x}");
                if CODE_PAGE.contains(&va) {
                    code_page_itlb_ops.push(line.clone());
                }
                itlb_ops.push(line);
            }
            _ => {}
        }

        // Sample the ITLB view of the collision cell at the two contexts.
        if matches!(pc, 0x8c98 | 0x8cac | 0x7fe1 | 0x8c6c | 0x8cae | 0x8cb1) {
            let tag = match pc {
                0x8c98 | 0x8cac => "PUBLISHER",
                0x7fe1 | 0x8c6c => "SERVICE-entry",
                0x8cae | 0x8cb1 => "SERVICE-cell",
                _ => "?",
            };
            let sample = itlb_pa_of(&mut proc, CELL);
            cell_samples.push(format!(
                "n={n} pc={pc:#08x} [{tag}] exccause={:#x} cell {CELL:#x}: {sample}",
                proc.cpu.regs.exccause
            ));
        }

        let step = proc.cpu.step(&mut proc.bus);
        n += 1;
        match step {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(reason) => {
                stop = format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc & 0x00ff_ffff);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
    }

    eprintln!("=== ITLB code-view selector probe ===");
    eprintln!("n={n} stop={stop}");
    eprintln!("-- all executed ITLB modifications ({}) --", itlb_ops.len());
    for l in &itlb_ops {
        eprintln!("{l}");
    }
    eprintln!(
        "-- ITLB modifications touching the 0x8000..0x9000 code page ({}) --",
        code_page_itlb_ops.len()
    );
    for l in &code_page_itlb_ops {
        eprintln!("{l}");
    }
    eprintln!("-- ITLB view of collision cell 0x8cae at publisher vs service --");
    for l in &cell_samples {
        eprintln!("{l}");
    }
}

/// Resolve the effective address, value, and width of a decoded store without
/// executing it. The caller records it only after the step retires.
fn decoded_store(proc: &FirmwareProcessor, op: &decode::Op) -> Option<(u32, u32, u8)> {
    match op {
        decode::Op::S32i { t, s, imm }
        | decode::Op::S32iN { t, s, imm }
        | decode::Op::S32ri { t, s, imm }
        | decode::Op::S32c1i { t, s, imm }
        | decode::Op::S32e { t, s, imm } => {
            Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t), 4))
        }
        decode::Op::Ssi { ft, s, imm } => {
            Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.fr[*ft as usize], 4))
        }
        decode::Op::S16i { t, s, imm } => {
            Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xffff, 2))
        }
        decode::Op::S8i { t, s, imm } => {
            Some((proc.cpu.regs.read_ar(*s).wrapping_add(*imm), proc.cpu.regs.read_ar(*t) & 0xff, 1))
        }
        _ => None,
    }
}

fn decoded_store_events(proc: &FirmwareProcessor, op: &decode::Op) -> Vec<(u32, u32, u8, Option<u8>)> {
    let one = |op: &decode::Op| {
        decoded_store(proc, op).map(|(ea, value, width)| {
            let conditional_t = match op {
                decode::Op::S32c1i { t, .. } => Some(*t),
                _ => None,
            };
            (ea, value, width, conditional_t)
        })
    };
    match op {
        decode::Op::Flix1 { ops } => ops.iter().filter_map(one).collect(),
        _ => one(op).into_iter().collect(),
    }
}

/// Resolve AR-producing loads before execution. The post-step target value is
/// read by the caller, so this stays observational and does not touch the bus.
fn decoded_load_events(proc: &FirmwareProcessor, op: &decode::Op) -> Vec<(u8, u32, u8, bool)> {
    let one = |op: &decode::Op| match op {
        decode::Op::L32iN { t, s, imm } | decode::Op::L32i { t, s, imm } | decode::Op::L32e { t, s, imm } => {
            Some((*t, proc.cpu.regs.read_ar(*s).wrapping_add(*imm), 4, false))
        }
        decode::Op::L8ui { t, s, imm } => Some((*t, proc.cpu.regs.read_ar(*s).wrapping_add(*imm), 1, false)),
        decode::Op::L16ui { t, s, imm } | decode::Op::L16si { t, s, imm } => {
            Some((*t, proc.cpu.regs.read_ar(*s).wrapping_add(*imm), 2, false))
        }
        decode::Op::L32r { t, target } => Some((*t, *target, 4, true)),
        _ => None,
    };
    match op {
        decode::Op::Flix1 { ops } => ops.iter().filter_map(one).collect(),
        _ => one(op).into_iter().collect(),
    }
}

fn decoded_call_target(proc: &FirmwareProcessor, op: &decode::Op) -> Option<u32> {
    match op {
        decode::Op::Call0 { target }
        | decode::Op::Call4 { target }
        | decode::Op::Call8 { target }
        | decode::Op::Call12 { target } => Some(*target),
        decode::Op::Callx0 { s }
        | decode::Op::Callx4 { s }
        | decode::Op::Callx8 { s }
        | decode::Op::Callx12 { s } => Some(proc.cpu.regs.read_ar(*s)),
        decode::Op::Flix1 { ops } => ops.iter().find_map(|op| decoded_call_target(proc, op)),
        _ => None,
    }
}

fn contains_control_op(op: &decode::Op, predicate: fn(&decode::Op) -> bool) -> bool {
    predicate(op) || matches!(op, decode::Op::Flix1 { ops } if ops.iter().any(|slot| predicate(slot)))
}

/// Classify non-local store targets by their firmware-visible effective
/// address. The three device bases are the NPU1 apertures in xdna-driver's
/// npu1_regs.c; they must be checked before Bus::is_local_data because they
/// numerically lie below the emulator's provisional 64 MiB local-data ceiling.
fn nonlocal_store_region(ea: u32) -> Option<&'static str> {
    if (0x0300_0000..0x0308_0000).contains(&ea) {
        return Some("device-bar0-management");
    }
    if (0x0308_0000..0x030c_0000).contains(&ea) {
        return Some("device-bar2-shared-sram");
    }
    if (0x030c_0000..0x0400_0000).contains(&ea) {
        return Some("device-bar4-mailbox-or-reserved");
    }
    if Bus::is_local_data(ea) {
        return None;
    }
    if (0x2000_0000..0x2700_0000).contains(&ea) {
        return Some("high-code-alias");
    }
    if (0x4000_0000..0x8000_0000).contains(&ea) {
        return Some("high-data-alias");
    }
    if (0x8000_0000..0xc000_0000).contains(&ea) {
        return Some("aie-array-noc-mmio");
    }
    Some(match Bus::region(ea) {
        super::super::mmio::Region::Rom => "low-rom",
        super::super::mmio::Region::Ram => "segment-b-or-ram",
        super::super::mmio::Region::Mailbox => "device-mailbox",
        super::super::mmio::Region::Array => "aie-array-mmio",
        super::super::mmio::Region::System => "system-or-vendor-mmio",
        super::super::mmio::Region::PageTable => "page-table",
    })
}

/// Acceptance oracle from the 2026-07-11 BAR2 dump: a natural boot must build
/// the management-channel descriptor in device SRAM and publish its pointer.
#[test]
fn m2c_probe_alive_device_sram_struct() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let mut n = 0;
    let mut stores = Vec::new();
    let stop = loop {
        if n >= max {
            break format!("budget {max}");
        }
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let phys = proc
            .cpu
            .translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
            .expect("fetch translates");
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(proc.cpu.pc + i as u32, phys + i as u32));
        let store = decoded_store(&proc, &decode::decode(&bytes, proc.cpu.pc).op);
        let step = proc.cpu.step(&mut proc.bus);
        if matches!(step, Step::Ran) {
            if let Some((ea, value, width)) =
                store.filter(|(ea, _, _)| (0x030b_0000..0x030c_0000).contains(ea))
            {
                let pa = proc
                    .cpu
                    .translate(&mut proc.bus, ea, xtensa::interp::Access::Store)
                    .expect("observed SRAM store translates");
                stores.push((n, pc, ea, pa, value, width));
            }
        }
        match step {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => break format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc),
            Step::Unknown { pc, word } => break format!("Unknown pc={pc:#x} word={word:#x}"),
        }
    };
    let actual: Vec<u32> = (0..ALIVE_DESCRIPTOR.len())
        .map(|i| {
            proc.cpu
                .data_read32(&mut proc.bus, 0x030b_b000 + i as u32 * 4)
                .expect("SRAM read")
        })
        .collect();
    let alive = proc.cpu.data_read32(&mut proc.bus, 0x030b_f000).expect("FW_ALIVE_OFF read");

    eprintln!("natural boot: n={n} stop={stop}");
    for &(n, pc, ea, pa, value, width) in &stores {
        eprintln!("n={n} pc={pc:#x} STORE{width} EA={ea:#010x} -> PA={pa:#010x} value={value:#010x}");
    }
    assert!(
        stores.iter().any(|&(_, _, ea, pa, _, _)| {
            (0x030b_b000..0x030b_b040).contains(&ea) || (0x030b_b000..0x030b_b040).contains(&pa)
        }),
        "firmware emitted no device-SRAM descriptor stores"
    );
    assert!(
        stores
            .iter()
            .any(|&(_, _, ea, pa, _, _)| ea == 0x030b_f000 || pa == 0x030b_f000),
        "firmware emitted no FW_ALIVE_OFF store"
    );
    assert_eq!(actual, ALIVE_DESCRIPTOR, "firmware did not store the HW-observed mgmt-channel descriptor");
    assert_eq!(alive, 0x030b_b000, "firmware did not store the device-absolute channel pointer");
}

/// Distinguish the real SYSCALL exception path from the later column-service
/// call chain, and preserve the live value that roots that chain. Observation
/// only: no overlays, registers, interrupts, or firmware memory are changed.
#[test]
fn m2c_probe_service_path_provenance() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    let mut syscalls = Vec::new();
    let mut events = Vec::new();
    let mut registration = None;
    let mut service_call = None;
    let mut n = 0u64;

    while n < 100_000 {
        let full_pc = proc.cpu.pc;
        let pc = full_pc & 0x00ff_ffff;
        let phys = proc
            .cpu
            .translate(&mut proc.bus, full_pc, xtensa::interp::Access::Fetch)
            .expect("fetch translates");
        let bytes: [u8; 8] = std::array::from_fn(|i| proc.bus.fetch8(full_pc + i as u32, phys + i as u32));
        let op = decode::decode(&bytes, full_pc).op;

        if matches!(op, decode::Op::Syscall) {
            syscalls.push((n, full_pc, proc.cpu.regs.exccause, proc.cpu.epc1));
        }
        if matches!(
            pc,
            0x0ae0
                | 0x28b4
                | 0x46c4
                | 0xdae2
                | 0x2830
                | 0x283b
                | 0x8770
                | 0x878a
                | 0xc530
                | 0xc56e
                | 0x7fc4
                | 0x7fe1
                | 0x8c6c
                | 0x8c72
                | 0x8c8b
                | 0x8cb1
        ) {
            events.push(format!(
                "n={n} pc={full_pc:#010x} {op:?} a2={:#x} a3={:#x} a4={:#x} a5={:#x} a6={:#x} a7={:#x} EXCCAUSE={:#x} EPC1={:#x}",
                proc.cpu.regs.read_ar(2),
                proc.cpu.regs.read_ar(3),
                proc.cpu.regs.read_ar(4),
                proc.cpu.regs.read_ar(5),
                proc.cpu.regs.read_ar(6),
                proc.cpu.regs.read_ar(7),
                proc.cpu.regs.exccause,
                proc.cpu.epc1,
            ));
        }

        let writes_registration = matches!(op, decode::Op::S32iN { t: 2, s: 3, imm: 16 })
            && proc.cpu.regs.read_ar(3).wrapping_add(16) == 0x1187c;
        let scheduler_load = match op {
            decode::Op::L32iN { t, s, imm } if pc == 0x2830 => {
                Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), t))
            }
            _ => None,
        };
        let step = proc.cpu.step(&mut proc.bus);
        if pc == 0x46c4 {
            registration = Some(proc.cpu.regs.read_ar(10));
        }
        if pc == 0x7fe1 {
            service_call = Some((n, op.clone(), proc.cpu.regs.exccause, proc.cpu.epc1));
        }
        if writes_registration {
            events.push(format!(
                "n={n} pc={full_pc:#010x} registered [0x1187c] <- {:#x}",
                proc.bus.load_local32(0x1187c)
            ));
        }
        if let Some((ea, t)) = scheduler_load {
            events.push(format!(
                "n={n} pc={full_pc:#010x} loaded [{ea:#x}] -> a{t}={:#x}",
                proc.cpu.regs.read_ar(t)
            ));
        }

        match step {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(_) => break,
            Step::Unknown { pc: 0x8cb1, .. } => break,
            Step::Unknown { pc, word } => panic!("unexpected wall pc={pc:#x} word={word:#x}"),
        }
    }

    eprintln!("=== SYSCALL vs column-service provenance ===");
    eprintln!(
        "VMA 0x32f4 words: BASE={:#010x} AT={:#010x}",
        word_at(&raw, 0x32f4, BASE_DELTA).unwrap(),
        word_at(&raw, 0x32f4, OVERLAY_DELTA).unwrap(),
    );
    for &(at, pc, old_cause, old_epc1) in &syscalls {
        eprintln!("SYSCALL n={at} pc={pc:#010x} prior_EXCCAUSE={old_cause:#x} prior_EPC1={old_epc1:#x}");
    }
    for event in &events {
        eprintln!("{event}");
    }

    assert_eq!(registration, Some(0x8770));
    assert_eq!(proc.bus.load_local32(0x1187c), 0x8770);
    assert!(!syscalls.is_empty());
    let (call_n, call_op, cause, epc1) = service_call.expect("0x7fe1 service call did not execute");
    assert!(matches!(call_op, decode::Op::Call8 { target: 0x8c6c }));
    assert!(syscalls.iter().all(|&(sys_n, ..)| sys_n < call_n));
    assert_eq!(cause, 1);
    assert_eq!(epc1, syscalls.last().unwrap().1 + 3);
}

/// Counterfactual discriminator for the proven shared-VMA collision. This is
/// not a production mapping proposal: it asks whether selecting the BASE view
/// for the service-only overlap after the AT publisher has finished advances
/// the natural boot into the device-SRAM writer.
#[test]
fn m2c_probe_runtime_view_discriminator() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000u64);
    let mut n = 0;
    let mut switched = false;
    let mut ctxsw_switched = false;
    let mut returned = false;
    let mut stores = Vec::new();
    let mut tail = VecDeque::new();
    let mut a7_changes = Vec::new();
    let mut reject_inputs = Vec::new();
    let mut last_a7 = proc.cpu.regs.read_ar(7);
    let stop;

    loop {
        if n >= max {
            stop = format!("budget {max}");
            break;
        }
        let pc = proc.cpu.pc & 0x00ff_ffff;
        let a7 = proc.cpu.regs.read_ar(7);
        if ctxsw_switched && a7 != last_a7 {
            a7_changes.push(format!(
                "n={n} pc={pc:#x} a7 {last_a7:#x}->{a7:#x} EXCCAUSE={:#x} EPC1={:#x}",
                proc.cpu.regs.exccause, proc.cpu.epc1,
            ));
        }
        last_a7 = a7;
        if pc == 0x8c6c && !switched {
            proc.bus.remove_rom_overlay(0x8c98, 0x8d52);
            proc.bus.add_rom_overlay(0x8c98, 0x8cae, OVERLAY_DELTA);
            proc.bus.add_rom_overlay(0x8cbc, 0x8d52, OVERLAY_DELTA);
            proc.bus.remove_rom_overlay(0x354c, 0x3564);
            proc.bus.add_rom_overlay(0x3550, 0x3564, OVERLAY_DELTA);
            switched = true;
            eprintln!("n={n} pc={pc:#x}: selected BASE for code [0x8cae,0x8cbc) and literal [0x354c,0x3550)");
        }
        if pc == 0x26d4 && !ctxsw_switched {
            proc.bus.remove_rom_overlay(CTXSW_CALLEE_LO, CTXSW_CALLEE_HI);
            ctxsw_switched = true;
            eprintln!("n={n} pc={pc:#x}: selected BASE for the 0x26d4 context-switch alias");
        }
        if pc == 0x7fec && ctxsw_switched {
            eprintln!(
                "n={n} pc={pc:#x}: service sink a7={:#x} EXCCAUSE={:#x} EPC1={:#x} INTERRUPT={:#x}",
                proc.cpu.regs.read_ar(7),
                proc.cpu.regs.exccause,
                proc.cpu.epc1,
                proc.cpu.interrupt,
            );
            stop = "service sink 0x7fec".into();
            break;
        }
        returned |= pc == 0x7fe4;

        let phys = proc
            .cpu
            .translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
            .expect("fetch translates");
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(proc.cpu.pc + i as u32, phys + i as u32));
        let op = decode::decode(&bytes, proc.cpu.pc).op;
        if ctxsw_switched
            && matches!(pc, 0x26e0 | 0x26e3 | 0x26eb | 0x271e | 0x2720 | 0x2728 | 0x2734 | 0xc533)
        {
            let load = match op {
                decode::Op::L32i { s, imm, .. } | decode::Op::L32iN { s, imm, .. } => {
                    let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    format!(" EA={ea:#x} value={:#x}", proc.bus.data_load32(ea & 0x00ff_ffff))
                }
                _ => String::new(),
            };
            reject_inputs.push(format!(
                "n={n} pc={pc:#x} {op:?}{load} a10={:#x} a11={:#x} a12={:#x} a13={:#x} a14={:#x} a15={:#x}",
                proc.cpu.regs.read_ar(10),
                proc.cpu.regs.read_ar(11),
                proc.cpu.regs.read_ar(12),
                proc.cpu.regs.read_ar(13),
                proc.cpu.regs.read_ar(14),
                proc.cpu.regs.read_ar(15),
            ));
        }
        if tail.len() == 96 {
            tail.pop_front();
        }
        tail.push_back((n, pc, format!("{op:?}")));
        let store = decoded_store(&proc, &op);
        let step = proc.cpu.step(&mut proc.bus);
        if matches!(step, Step::Ran) {
            if let Some((ea, value, width)) =
                store.filter(|(ea, _, _)| (0x030b_0000..0x030c_0000).contains(ea))
            {
                let pa = proc
                    .cpu
                    .translate(&mut proc.bus, ea, xtensa::interp::Access::Store)
                    .expect("observed SRAM store translates");
                stores.push((pc, ea, pa, value, width));
            }
        }
        match step {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                stop = format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc & 0x00ff_ffff);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
    }

    eprintln!(
        "runtime-view discriminator: n={n} stop={stop} switched={switched} ctxsw_switched={ctxsw_switched} returned={returned}"
    );
    for (n, pc, op) in &tail {
        eprintln!("tail n={n} pc={pc:#x} {op}");
    }
    for change in &a7_changes {
        eprintln!("{change}");
    }
    for input in &reject_inputs {
        eprintln!("{input}");
    }
    for &(pc, ea, pa, value, width) in &stores {
        eprintln!("pc={pc:#x} STORE{width} EA={ea:#010x} -> PA={pa:#010x} value={value:#010x}");
    }
    assert!(switched && ctxsw_switched && returned, "BASE service views did not advance coherently: {stop}");
    assert_eq!(proc.cpu.pc & 0x00ff_ffff, 0x7fec);
    assert_eq!(proc.cpu.regs.read_ar(7), 6);
    assert_eq!(proc.cpu.regs.exccause, 1);
    assert_eq!(proc.cpu.epc1, 0x08b0_e713);
    assert_eq!(proc.cpu.interrupt, 0);
    assert_eq!(stores, vec![(0x8964, 0x030b_27c0, 0x030b_27c0, 0, 4)]);
}

fn record_tlb_changes<const N: usize>(
    timeline: &mut Vec<String>,
    n: u64,
    pc: u32,
    phase: &str,
    side: &str,
    before: &[[xtensa::mmu::TlbEntry; xtensa::mmu::MAX_TLB_WAY_SIZE]; N],
    after: &[[xtensa::mmu::TlbEntry; xtensa::mmu::MAX_TLB_WAY_SIZE]; N],
) -> (usize, usize) {
    let mut changed = 0;
    let mut non_autorefill = 0;
    for wi in 0..N {
        for ei in 0..xtensa::mmu::MAX_TLB_WAY_SIZE {
            if before[wi][ei] == after[wi][ei] {
                continue;
            }
            changed += 1;
            non_autorefill += usize::from(wi >= 4);
            let class = if wi < 4 { "autorefill" } else { "non-autorefill" };
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=TLB_CHANGE detail={phase} {side}[{wi}][{ei}] class={class} {{vaddr={:#010x} paddr={:#010x} asid={:#04x} attr={:#x} variable={}}}->{{vaddr={:#010x} paddr={:#010x} asid={:#04x} attr={:#x} variable={}}}",
                before[wi][ei].vaddr,
                before[wi][ei].paddr,
                before[wi][ei].asid,
                before[wi][ei].attr,
                before[wi][ei].variable,
                after[wi][ei].vaddr,
                after[wi][ei].paddr,
                after[wi][ei].asid,
                after[wi][ei].attr,
                after[wi][ei].variable,
            ));
        }
    }
    (changed, non_autorefill)
}

/// Firmware-action timeline across the early AT and later BASE executions of
/// VMA 0x26d4. The two already-established counterfactual view selections are
/// logged as HARNESS events and excluded from the firmware-action verdict.
#[test]
fn m2c_probe_26d4_cache_pageroot_timeline() {
    if std::env::var("XDNA_FW_PROBE").is_err() {
        eprintln!("skip: set XDNA_FW_PROBE=1");
        return;
    }
    let Some(path) = firmware_path() else { return };
    let raw = std::fs::read(path).expect("read firmware");
    assert_calibration_image(&raw);
    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
    let max = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000u64);

    let mut timeline = Vec::new();
    let mut active = false;
    let mut collision_switched = false;
    let mut early_byte_crossed = false;
    let mut later_base_entered = false;
    let mut reached_sink = false;
    let mut n = 0;
    let mut selector_i_cache_ops = 0;
    let mut selector_d_cache_ops = 0;
    let mut selector_root_writes = 0;
    let mut selector_root_changes = 0;
    let mut selector_tlb_ops = 0;
    let mut selector_itlb_ops = 0;
    let mut selector_dtlb_ops = 0;
    let mut selector_tlb_changes = 0;
    let mut selector_itlb_changes = 0;
    let mut selector_dtlb_changes = 0;
    let mut selector_non_autorefill_changes = 0;
    let mut call_chain = Vec::new();
    let mut nonlocal_stores = Vec::new();
    let mut store_region_counts = BTreeMap::new();
    let mut provenance_active = false;
    let mut trace_task_creation = false;
    let mut a7_provenance = Vec::new();
    let mut state_word_timeline =
        vec![format!("n=0 pc=RESET op=INITIAL EA=0x00010e04 value={:#010x}", proc.bus.load_local32(0x10e04))];
    let mut current_task_timeline =
        vec![format!("n=0 pc=RESET op=INITIAL EA=0x00002278 value={:#010x}", proc.bus.load_local32(0x2278))];
    let stop;

    loop {
        if n >= max {
            stop = format!("budget {max}");
            break;
        }
        let full_pc = proc.cpu.pc;
        let pc = full_pc & 0x00ff_ffff;

        if pc == 0xd4e0 && proc.cpu.regs.read_ar(10) == 0x10dfc {
            trace_task_creation = true;
        }

        if !active && pc == CTXSW_CALLEE_LO {
            active = true;
            call_chain.push(full_pc);
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=MARK detail=begin early AT context epoch PTEVADDR={:#010x} RASID={:#010x} ITLBCFG={:#010x} DTLBCFG={:#010x}",
                proc.cpu.mmu.ptevaddr,
                proc.cpu.mmu.rasid,
                proc.cpu.mmu.itlbcfg,
                proc.cpu.mmu.dtlbcfg,
            ));
        }

        if active && pc == 0x8c6c && !collision_switched {
            proc.bus.remove_rom_overlay(0x8c98, 0x8d52);
            proc.bus.add_rom_overlay(0x8c98, 0x8cae, OVERLAY_DELTA);
            proc.bus.add_rom_overlay(0x8cbc, 0x8d52, OVERLAY_DELTA);
            proc.bus.remove_rom_overlay(0x354c, 0x3564);
            proc.bus.add_rom_overlay(0x3550, 0x3564, OVERLAY_DELTA);
            collision_switched = true;
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=HARNESS_VIEW detail=select BASE service overlap; not firmware"
            ));
        }

        if active && early_byte_crossed && !later_base_entered && pc == 0x26d4 {
            let hit = proc.cpu.mmu.lookup(0x26d4, false).expect("0x26d4 ITLB lookup");
            let pa = proc
                .cpu
                .translate(&mut proc.bus, 0x26d4, xtensa::interp::Access::Fetch)
                .expect("0x26d4 fetch translation");
            let entry = proc.cpu.mmu.itlb[hit.wi][hit.ei];
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=VIEW_EPOCH detail=later BASE entry PA={pa:#010x} ITLB[{}][{}]={{vaddr={:#010x} paddr={:#010x} asid={:#04x} attr={:#x}}} PTEVADDR={:#010x} RASID={:#010x} ITLBCFG={:#010x} DTLBCFG={:#010x}",
                hit.wi,
                hit.ei,
                entry.vaddr,
                entry.paddr,
                entry.asid,
                entry.attr,
                proc.cpu.mmu.ptevaddr,
                proc.cpu.mmu.rasid,
                proc.cpu.mmu.itlbcfg,
                proc.cpu.mmu.dtlbcfg,
            ));
            proc.bus.remove_rom_overlay(CTXSW_CALLEE_LO, CTXSW_CALLEE_HI);
            later_base_entered = true;
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=HARNESS_VIEW detail=select BASE 0x26d4 view; not firmware"
            ));
        }

        if active && pc == 0x7fec && later_base_entered {
            reached_sink = true;
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=MARK detail=service sink a7={:#x} EXCCAUSE={:#x} EPC1={:#x}",
                proc.cpu.regs.read_ar(7),
                proc.cpu.regs.exccause,
                proc.cpu.epc1,
            ));
            stop = "service sink 0x7fec".into();
            break;
        }

        if active && early_byte_crossed && pc == 0x8770 {
            provenance_active = true;
        }

        let itlb_before_fetch = proc.cpu.mmu.itlb;
        let dtlb_before_fetch = proc.cpu.mmu.dtlb;
        let fetch_pa = match proc.cpu.translate(&mut proc.bus, full_pc, xtensa::interp::Access::Fetch) {
            Ok(pa) => pa,
            Err(cause) => {
                stop = format!("fetch translation failed pc={full_pc:#x} cause={cause:?}");
                break;
            }
        };
        let bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(full_pc + i as u32, fetch_pa + i as u32));
        let decoded = decode::decode(&bytes, full_pc);
        let op = decoded.op;

        if active && !early_byte_crossed && pc <= 0x26d4 && pc.wrapping_add(decoded.len as u32) > 0x26d4 {
            early_byte_crossed = true;
            proc.cpu.fastpath_enabled = false;
            let hit = proc.cpu.mmu.lookup(0x26d4, false).expect("early 0x26d4 ITLB lookup");
            let pa = proc
                .cpu
                .translate(&mut proc.bus, 0x26d4, xtensa::interp::Access::Fetch)
                .expect("early 0x26d4 fetch translation");
            let entry = proc.cpu.mmu.itlb[hit.wi][hit.ei];
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=VIEW_EPOCH detail=early AT instruction {op:?} len={} spans byte 0x26d4; PA={pa:#010x} ITLB[{}][{}]={{vaddr={:#010x} paddr={:#010x} asid={:#04x} attr={:#x}}} PTEVADDR={:#010x} RASID={:#010x} ITLBCFG={:#010x} DTLBCFG={:#010x}",
                decoded.len,
                hit.wi,
                hit.ei,
                entry.vaddr,
                entry.paddr,
                entry.asid,
                entry.attr,
                proc.cpu.mmu.ptevaddr,
                proc.cpu.mmu.rasid,
                proc.cpu.mmu.itlbcfg,
                proc.cpu.mmu.dtlbcfg,
            ));
            timeline.push(format!(
                "n={n} pc={pc:#08x} op=HARNESS_OBSERVER detail=disable fill-loop fastpath to expose every retired store; architecturally equivalent, not firmware"
            ));
        }

        if active {
            let (all, non_ar) = record_tlb_changes(
                &mut timeline,
                n,
                pc,
                "fetch",
                "ITLB",
                &itlb_before_fetch,
                &proc.cpu.mmu.itlb,
            );
            if early_byte_crossed && !later_base_entered {
                selector_tlb_changes += all;
                selector_itlb_changes += all;
                selector_non_autorefill_changes += non_ar;
            }
            let (all, non_ar) = record_tlb_changes(
                &mut timeline,
                n,
                pc,
                "fetch",
                "DTLB",
                &dtlb_before_fetch,
                &proc.cpu.mmu.dtlb,
            );
            if early_byte_crossed && !later_base_entered {
                selector_tlb_changes += all;
                selector_dtlb_changes += all;
                selector_non_autorefill_changes += non_ar;
            }

            let cache = match &op {
                decode::Op::Dpfr { s, imm }
                | decode::Op::Dpfw { s, imm }
                | decode::Op::Dpfro { s, imm }
                | decode::Op::Dpfwo { s, imm }
                | decode::Op::Dhwb { s, imm }
                | decode::Op::Dhwbi { s, imm }
                | decode::Op::Dhi { s, imm }
                | decode::Op::Dii { s, imm }
                | decode::Op::Dpfl { s, imm }
                | decode::Op::Dhu { s, imm }
                | decode::Op::Diu { s, imm }
                | decode::Op::Diwb { s, imm }
                | decode::Op::Diwbi { s, imm } => Some(("D", *s, *imm)),
                decode::Op::Ipf { s, imm }
                | decode::Op::Ipfl { s, imm }
                | decode::Op::Ihu { s, imm }
                | decode::Op::Iiu { s, imm }
                | decode::Op::Ihi { s, imm }
                | decode::Op::Iii { s, imm } => Some(("I", *s, *imm)),
                _ => None,
            };
            if let Some((side, s, imm)) = cache {
                let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                timeline.push(format!("n={n} pc={pc:#08x} op={op:?} detail=cache side={side} EA={ea:#010x}"));
                if early_byte_crossed && !later_base_entered {
                    if side == "I" {
                        selector_i_cache_ops += 1;
                    } else {
                        selector_d_cache_ops += 1;
                    }
                }
            }

            match &op {
                decode::Op::Wsr { sr, t } if matches!(*sr, 0x53 | 0x5a | 0x5b | 0x5c) => {
                    let old = match *sr {
                        0x53 => proc.cpu.mmu.ptevaddr,
                        0x5a => proc.cpu.mmu.rasid,
                        0x5b => proc.cpu.mmu.itlbcfg,
                        0x5c => proc.cpu.mmu.dtlbcfg,
                        _ => unreachable!(),
                    };
                    timeline.push(format!(
                        "n={n} pc={pc:#08x} op={op:?} detail=page-root/config old={old:#010x} requested={:#010x}",
                        proc.cpu.regs.read_ar(*t),
                    ));
                    if early_byte_crossed && !later_base_entered {
                        selector_root_writes += 1;
                    }
                }
                decode::Op::Witlb { t, s } => {
                    timeline.push(format!(
                        "n={n} pc={pc:#08x} op={op:?} detail=ITLB write AS={:#010x} AT={:#010x}",
                        proc.cpu.regs.read_ar(*s),
                        proc.cpu.regs.read_ar(*t),
                    ));
                    if early_byte_crossed && !later_base_entered {
                        selector_tlb_ops += 1;
                        selector_itlb_ops += 1;
                    }
                }
                decode::Op::Wdtlb { t, s } => {
                    timeline.push(format!(
                        "n={n} pc={pc:#08x} op={op:?} detail=DTLB write AS={:#010x} AT={:#010x}",
                        proc.cpu.regs.read_ar(*s),
                        proc.cpu.regs.read_ar(*t),
                    ));
                    if early_byte_crossed && !later_base_entered {
                        selector_tlb_ops += 1;
                        selector_dtlb_ops += 1;
                    }
                }
                decode::Op::Iitlb { s } => {
                    timeline.push(format!(
                        "n={n} pc={pc:#08x} op={op:?} detail=TLB invalidate AS={:#010x}",
                        proc.cpu.regs.read_ar(*s),
                    ));
                    if early_byte_crossed && !later_base_entered {
                        selector_tlb_ops += 1;
                        selector_itlb_ops += 1;
                    }
                }
                decode::Op::Idtlb { s } => {
                    timeline.push(format!(
                        "n={n} pc={pc:#08x} op={op:?} detail=TLB invalidate AS={:#010x}",
                        proc.cpu.regs.read_ar(*s),
                    ));
                    if early_byte_crossed && !later_base_entered {
                        selector_tlb_ops += 1;
                        selector_dtlb_ops += 1;
                    }
                }
                decode::Op::Call0 { .. }
                | decode::Op::Call4 { .. }
                | decode::Op::Call8 { .. }
                | decode::Op::Call12 { .. }
                | decode::Op::Callx0 { .. }
                | decode::Op::Callx4 { .. }
                | decode::Op::Callx8 { .. }
                | decode::Op::Callx12 { .. }
                | decode::Op::Entry { .. }
                | decode::Op::RetN
                | decode::Op::Retw
                | decode::Op::RetwN => {
                    timeline.push(format!("n={n} pc={pc:#08x} op={op:?} detail=control-boundary"));
                }
                _ => {}
            }
        }

        let roots_before =
            (proc.cpu.mmu.ptevaddr, proc.cpu.mmu.rasid, proc.cpu.mmu.itlbcfg, proc.cpu.mmu.dtlbcfg);
        let itlb_before_step = proc.cpu.mmu.itlb;
        let dtlb_before_step = proc.cpu.mmu.dtlb;
        if active && matches!(op, decode::Op::Entry { .. }) && call_chain.last().copied() != Some(full_pc) {
            call_chain.push(full_pc);
        }
        let stores = if active && early_byte_crossed {
            decoded_store_events(&proc, &op)
        } else {
            Vec::new()
        };
        let trace_provenance = (0x2800..0x2870).contains(&pc)
            || (trace_task_creation && (0xd4e0..0xd620).contains(&pc))
            || (provenance_active
                && ((0x26d4..0x2750).contains(&pc)
                    || (0x7fc4..0x8020).contains(&pc)
                    || (0x8c6c..0x8cbc).contains(&pc)
                    || (0xc530..0xc584).contains(&pc)));
        let loads = trace_provenance.then(|| decoded_load_events(&proc, &op)).unwrap_or_default();
        let provenance_stores =
            trace_provenance.then(|| decoded_store_events(&proc, &op)).unwrap_or_default();
        let state_word_before = proc.bus.load_local32(0x10e04);
        let current_task_before = proc.bus.load_local32(0x2278);
        let pre_wb = proc.cpu.regs.windowbase;
        let pre_ar: [u32; 16] = std::array::from_fn(|i| proc.cpu.regs.read_ar(i as u8));
        let call_target = active.then(|| decoded_call_target(&proc, &op)).flatten();
        let returning = contains_control_op(&op, |op| {
            matches!(op, decode::Op::RetN | decode::Op::Retw | decode::Op::RetwN)
        });
        let exception_returning = contains_control_op(&op, |op| {
            matches!(op, decode::Op::Rfe | decode::Op::Rfwo | decode::Op::Rfwu)
        });
        let scompare1_before = proc.cpu.scompare1;
        let step = proc.cpu.step(&mut proc.bus);

        if trace_provenance {
            let post_wb = proc.cpu.regs.windowbase;
            let post_ar: [u32; 16] = std::array::from_fn(|i| proc.cpu.regs.read_ar(i as u8));
            let view = if (0x26d4..0x2750).contains(&pc) && later_base_entered {
                "HARNESS_VIEW_BASE_26d4"
            } else if (0x8c6c..0x8cbc).contains(&pc) && collision_switched {
                "HARNESS_VIEW_SPLIT_8cxx"
            } else {
                "mapped-firmware"
            };
            let mut changes = String::new();
            for i in 0..16 {
                if pre_ar[i] != post_ar[i] {
                    let _ = write!(changes, " a{i}={:#010x}->{:#010x}", pre_ar[i], post_ar[i]);
                }
            }
            let mut sources = String::new();
            if matches!(step, Step::Ran) {
                for (t, ea, width, literal) in &loads {
                    let region = if *literal {
                        "instruction-literal"
                    } else {
                        nonlocal_store_region(*ea).unwrap_or("ordinary-local-data")
                    };
                    let _ = write!(
                        sources,
                        " LOAD{width}->a{t} EA={ea:#010x} value={:#010x} region={region}",
                        post_ar[*t as usize],
                    );
                }
                for (ea, value, width, _) in &provenance_stores {
                    let region = nonlocal_store_region(*ea).unwrap_or("ordinary-local-data");
                    let _ =
                        write!(sources, " STORE{width} EA={ea:#010x} value={value:#010x} region={region}");
                }
            }
            a7_provenance.push(format!(
                "n={n} pc={pc:#08x} op={op:?} step={step:?} view={view} WB={pre_wb}->{post_wb} a7={:#010x}->{:#010x} a15={:#010x}->{:#010x} changes=[{changes} ] sources=[{sources} ]",
                pre_ar[7], post_ar[7], pre_ar[15], post_ar[15],
            ));
        }

        if trace_task_creation && pc == 0xd611 && matches!(step, Step::Ran) {
            trace_task_creation = false;
        }

        if active {
            if matches!(step, Step::Ran) {
                for (ea, value, width, conditional_t) in stores {
                    if conditional_t.is_some_and(|t| proc.cpu.regs.read_ar(t) != scompare1_before) {
                        continue;
                    }
                    let Some(region) = nonlocal_store_region(ea) else {
                        continue;
                    };
                    let phase = if later_base_entered {
                        "post-view"
                    } else {
                        "between-views"
                    };
                    let chain = call_chain
                        .iter()
                        .map(|addr| format!("{addr:#010x}"))
                        .collect::<Vec<_>>()
                        .join(">");
                    *store_region_counts.entry(region).or_insert(0usize) += 1;
                    nonlocal_stores.push((n, full_pc, width, ea, value, phase, region, chain));
                }
            }

            let roots_after =
                (proc.cpu.mmu.ptevaddr, proc.cpu.mmu.rasid, proc.cpu.mmu.itlbcfg, proc.cpu.mmu.dtlbcfg);
            if roots_before != roots_after {
                timeline.push(format!(
                    "n={n} pc={pc:#08x} op=ROOT_CHANGE detail=PTEVADDR {:#010x}->{:#010x} RASID {:#010x}->{:#010x} ITLBCFG {:#010x}->{:#010x} DTLBCFG {:#010x}->{:#010x}",
                    roots_before.0,
                    roots_after.0,
                    roots_before.1,
                    roots_after.1,
                    roots_before.2,
                    roots_after.2,
                    roots_before.3,
                    roots_after.3,
                ));
                if early_byte_crossed && !later_base_entered {
                    selector_root_changes += 1;
                }
            }
            let (all, non_ar) = record_tlb_changes(
                &mut timeline,
                n,
                pc,
                "step",
                "ITLB",
                &itlb_before_step,
                &proc.cpu.mmu.itlb,
            );
            if early_byte_crossed && !later_base_entered {
                selector_tlb_changes += all;
                selector_itlb_changes += all;
                selector_non_autorefill_changes += non_ar;
            }
            let (all, non_ar) = record_tlb_changes(
                &mut timeline,
                n,
                pc,
                "step",
                "DTLB",
                &dtlb_before_step,
                &proc.cpu.mmu.dtlb,
            );
            if early_byte_crossed && !later_base_entered {
                selector_tlb_changes += all;
                selector_dtlb_changes += all;
                selector_non_autorefill_changes += non_ar;
            }

            match step {
                Step::Ran if call_target.is_some() => call_chain.push(call_target.unwrap()),
                Step::Ran if returning || exception_returning => {
                    call_chain.pop();
                }
                Step::Exception { .. } => call_chain.push(proc.cpu.pc),
                _ => {}
            }
        }

        let state_word_after = proc.bus.load_local32(0x10e04);
        if state_word_before != state_word_after {
            state_word_timeline.push(format!(
                "n={n} pc={pc:#08x} op={op:?} EA=0x00010e04 value={state_word_before:#010x}->{state_word_after:#010x} step={step:?}"
            ));
        }
        let current_task_after = proc.bus.load_local32(0x2278);
        if current_task_before != current_task_after {
            current_task_timeline.push(format!(
                "n={n} pc={pc:#08x} op={op:?} EA=0x00002278 value={current_task_before:#010x}->{current_task_after:#010x} step={step:?}"
            ));
        }

        match step {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(reason) => {
                stop = format!("Wait({reason:?}) pc={:#x}", proc.cpu.pc & 0x00ff_ffff);
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#x}");
                break;
            }
        }
    }

    eprintln!("=== 0x26d4 cache/page-root timeline ===");
    for line in &timeline {
        eprintln!("{line}");
    }
    eprintln!(
        "SUMMARY n={n} stop={stop} early_byte_crossed={early_byte_crossed} later_base_entered={later_base_entered} selector_i_cache_ops={selector_i_cache_ops} selector_d_cache_ops={selector_d_cache_ops} selector_root_writes={selector_root_writes} selector_root_changes={selector_root_changes} selector_tlb_ops={selector_tlb_ops} selector_itlb_ops={selector_itlb_ops} selector_dtlb_ops={selector_dtlb_ops} selector_tlb_changes={selector_tlb_changes} selector_itlb_changes={selector_itlb_changes} selector_dtlb_changes={selector_dtlb_changes} selector_non_autorefill_changes={selector_non_autorefill_changes}"
    );
    eprintln!("=== 0x26d4 non-local store timeline ===");
    for (n, pc, width, ea, value, phase, region, chain) in &nonlocal_stores {
        eprintln!(
            "n={n} pc={pc:#010x} STORE{width} EA={ea:#010x} value={value:#010x} phase={phase} region={region} chain={chain}"
        );
    }
    eprintln!("STORE_SUMMARY count={} regions={store_region_counts:?}", nonlocal_stores.len());
    eprintln!("=== a7 reject provenance ===");
    for line in &a7_provenance {
        eprintln!("{line}");
    }
    eprintln!("=== local state word 0x10e04 timeline ===");
    for line in &state_word_timeline {
        eprintln!("{line}");
    }
    eprintln!("=== scheduler current-task word 0x2278 timeline ===");
    for line in &current_task_timeline {
        eprintln!("{line}");
    }

    assert!(
        active && collision_switched && early_byte_crossed && later_base_entered && reached_sink,
        "timeline did not span both epochs: {stop}"
    );
    assert_eq!(selector_i_cache_ops, 0);
    assert_eq!(selector_d_cache_ops, 18);
    assert_eq!(selector_root_writes, 0);
    assert_eq!(selector_root_changes, 0);
    assert_eq!((selector_tlb_ops, selector_itlb_ops, selector_dtlb_ops), (1, 0, 1));
    assert_eq!((selector_tlb_changes, selector_itlb_changes, selector_dtlb_changes), (4, 0, 4));
    assert_eq!(selector_non_autorefill_changes, 1);
    assert!(
        nonlocal_stores.iter().any(|(_, _, _, ea, _, _, _, _)| *ea == 0x030b_27c0),
        "known device-SRAM publish-path store was not observed"
    );
    assert_eq!(nonlocal_stores.len(), 91);
    assert!(nonlocal_stores
        .iter()
        .all(|(_, _, _, _, _, phase, _, _)| *phase == "between-views"));
    assert_eq!(
        store_region_counts,
        BTreeMap::from([
            ("aie-array-noc-mmio", 68),
            ("device-bar0-management", 3),
            ("device-bar2-shared-sram", 1),
            ("device-mailbox", 18),
            ("high-data-alias", 1),
        ])
    );
    assert_eq!(
        nonlocal_stores
            .iter()
            .filter(|(_, _, _, _, _, _, region, _)| *region == "device-bar0-management")
            .map(|(n, pc, _, ea, value, _, _, _)| (*n, *pc, *ea, *value))
            .collect::<Vec<_>>(),
        vec![
            (50_908, 0x08b0_4229, 0x0301_0d7c, 0x0002_0405),
            (51_762, 0x08b0_4229, 0x0301_0d7c, 0x0204_0506),
            (52_194, 0x08b0_4229, 0x0301_0d7c, 0x0405_0607),
        ]
    );
    assert!(
        nonlocal_stores.iter().all(|(_, pc, _, _, _, _, _, _)| {
            !((0x0000_26d4..0x0000_2750).contains(pc)
                || (0x0000_7fc4..0x0000_8020).contains(pc)
                || (0x0000_8c6c..0x0000_8cbc).contains(pc)
                || (0x0000_c530..0x0000_c584).contains(pc)
                || (0x08b0_e710..0x08b0_e72a).contains(pc))
        }),
        "critical transition path executed a non-local store"
    );
    assert!(!a7_provenance.is_empty(), "a7 provenance observer did not reach the service chain");
    assert_eq!(proc.bus.load_local32(0x10e04), 6);
    assert_eq!(proc.bus.load_local32(0x2278), 0x10dfc);
    assert_eq!(state_word_timeline.len(), 2);
    assert_eq!(current_task_timeline.len(), 3);
    assert!(current_task_timeline[2].contains("n=47985 pc=0x00285d op=S32iN { t: 2, s: 7, imm: 40 }"));
    assert!(state_word_timeline[1].contains("n=39730 pc=0x00d4ef op=S32iN { t: 3, s: 8, imm: 8 }"));
    for needle in [
        "n=39760 pc=0x00d538 op=Addx4 { r: 3, s: 3, t: 15 }",
        "n=39852 pc=0x00d60f op=S32iN { t: 8, s: 3, imm: 56 }",
        "STORE4 EA=0x000022a0 value=0x00010dfc region=ordinary-local-data",
        "n=47969 pc=0x00282e op=L32iN { t: 5, s: 4, imm: 56 }",
        "LOAD4->a5 EA=0x000022a0 value=0x00010dfc region=ordinary-local-data",
        "n=47985 pc=0x00285d op=S32iN { t: 2, s: 7, imm: 40 }",
        "STORE4 EA=0x00002278 value=0x00010dfc region=ordinary-local-data",
        "n=53629 pc=0x00c56c op=MovN { t: 15, s: 7 }",
        "n=53632 pc=0x007fc7 op=Bgeui { s: 7, imm: 6, target: 32748 }",
        "n=53807 pc=0x002728 op=L32iN { t: 15, s: 8, imm: 8 }",
        "EA=0x00010e04 value=0x00000006 region=ordinary-local-data",
        "n=53825 pc=0x00c54d op=S32iN { t: 7, s: 10, imm: 20 }",
        "STORE4 EA=0x0000faf4 value=0x00000006 region=ordinary-local-data",
        "n=53870 pc=0x00c56c op=MovN { t: 15, s: 7 }",
        "n=53873 pc=0x007fc7 op=Bgeui { s: 7, imm: 6, target: 32748 }",
    ] {
        assert!(a7_provenance.iter().any(|line| line.contains(needle)), "missing provenance: {needle}");
    }
}

/// Discriminator (2026-07-11): PSP-patch theory vs local-scratch theory for the
/// alive publish. The `0x5044` publisher stores the exact HW-observed pointer
/// `0x030bb000` bytewise to a destination whose base literal at VMA 0x31bc is 0
/// in the flat signed image. On silicon that slot reaches `FW_ALIVE_OFF`
/// (`0x030bf000`, driver constant `npu1_regs.c`) -- either the literal is
/// PSP-patched at load (Claude's theory) or PA 0 aliases the slot (Sol's).
///
/// Model the PSP patch: 0x31bc is NOT overlaid, so the L32r at 0x50ba reads it
/// BASE-framed at file 0x3218 (= 0x31bc + PSP_LOAD_OFFSET 0x5c). Patch that word
/// to 0x030bf000 and observe:
///   - do the four publisher stores now land at 0x030bf000..3 (FW_ALIVE_OFF gets
///     0x030bb000, matching the 72.8ms HW capture)?
///   - CRITICALLY, do the later readbacks at 0x897d/0x89b1 FOLLOW the patched
///     literal to 0x030bf000 (coherent -> PSP-patch confirmed), or do they stay
///     at address 0 and read empty (the store was local scratch -> refuted)?
///   - does the boot stay coherent to the same 0x8cb1 wall, or diverge earlier?
#[test]
fn m2c_probe_psp_patch_alive_destination() {
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
    let mut raw = std::fs::read(path).expect("read firmware");

    const FW_ALIVE_OFF: u32 = 0x030b_f000;
    const FILE_OFF: usize = 0x3218; // VMA 0x31bc, BASE framing (not overlaid)
    let before = u32::from_le_bytes(raw[FILE_OFF..FILE_OFF + 4].try_into().unwrap());
    raw[FILE_OFF..FILE_OFF + 4].copy_from_slice(&FW_ALIVE_OFF.to_le_bytes());
    eprintln!("=== PSP-patch alive-destination discriminator ===");
    eprintln!("patched file {FILE_OFF:#x} (VMA 0x31bc): {before:#010x} -> {FW_ALIVE_OFF:#010x}");

    let image = FirmwareImage::parse(&raw).expect("parse firmware");
    let mut proc = FirmwareProcessor::load_m2c(image);

    let max: u64 = std::env::var("XDNA_FW_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    // Compute a load/store effective address from the decoded op and regs.
    fn store_ea_val(op: &decode::Op, cpu: &Cpu) -> Option<(u32, u32)> {
        match *op {
            decode::Op::S32i { t, s, imm }
            | decode::Op::S32iN { t, s, imm }
            | decode::Op::S16i { t, s, imm }
            | decode::Op::S8i { t, s, imm } => {
                Some((cpu.regs.read_ar(s).wrapping_add(imm), cpu.regs.read_ar(t)))
            }
            _ => None,
        }
    }
    fn load_ea(op: &decode::Op, cpu: &Cpu) -> Option<u32> {
        match *op {
            decode::Op::L32i { s, imm, .. }
            | decode::Op::L32iN { s, imm, .. }
            | decode::Op::L16ui { s, imm, .. }
            | decode::Op::L8ui { s, imm, .. } => Some(cpu.regs.read_ar(s).wrapping_add(imm)),
            _ => None,
        }
    }

    let mut publish_stores: Vec<(u64, u32, u32, u32)> = Vec::new(); // n, pc, EA, value
    let mut readbacks: Vec<(u64, u32, u32, u32)> = Vec::new(); // n, pc, EA, mem@EA
    let mut alive_writes: Vec<(u64, u32, u32)> = Vec::new(); // n, pc, value
    let stop;
    let mut n = 0u64;
    loop {
        if n >= max {
            stop = "budget".to_string();
            break;
        }
        let pc = proc.cpu.pc;
        let decoded = proc
            .cpu
            .translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
            .ok()
            .map(|phys| {
                let bytes: [u8; 8] = std::array::from_fn(|i| proc.bus.fetch8(pc + i as u32, phys + i as u32));
                decode::decode(&bytes, pc).op
            });
        if let Some(op) = &decoded {
            if matches!(pc, 0x50c6 | 0x50c9 | 0x50cc | 0x50cf) {
                if let Some((ea, val)) = store_ea_val(op, &proc.cpu) {
                    publish_stores.push((n, pc, ea, val));
                }
            }
            if matches!(pc, 0x897d | 0x89b1) {
                if let Some(ea) = load_ea(op, &proc.cpu) {
                    let memv = proc.cpu.data_read32(&mut proc.bus, ea).unwrap_or(0xdead_dead);
                    readbacks.push((n, pc, ea, memv));
                }
            }
            if let Some((ea, val)) = store_ea_val(op, &proc.cpu) {
                if ea == FW_ALIVE_OFF {
                    alive_writes.push((n, pc, val));
                }
            }
        }
        match proc.cpu.step(&mut proc.bus) {
            Step::Ran | Step::Exception { .. } => n += 1,
            Step::Wait(r) => {
                stop = format!("Wait({r:?})");
                break;
            }
            Step::Unknown { pc, word } => {
                stop = format!("Unknown pc={pc:#x} word={word:#010x}");
                break;
            }
        }
        if let Some(addr) = proc.bus.sysstub().spinning() {
            stop = format!("PollSpin {addr:#x}");
            break;
        }
    }

    eprintln!("instrs   = {n}");
    eprintln!("stop     = {stop}");
    eprintln!("last_pc  = {:#x}", proc.cpu.pc);
    eprintln!("-- publisher stores (0x50c6..0x50cf), EA should be 0x030bf000..3 if patch took --");
    for (n, pc, ea, val) in &publish_stores {
        eprintln!("  n={n} pc={pc:#x} EA={ea:#010x} value={:#04x}", val & 0xff);
    }
    eprintln!("-- readbacks (0x897d/0x89b1): EA should FOLLOW to 0x030bf000 if coherent --");
    for (n, pc, ea, memv) in &readbacks {
        eprintln!("  n={n} pc={pc:#x} EA={ea:#010x} mem@EA={memv:#010x}");
    }
    let alive = proc.cpu.data_read32(&mut proc.bus, FW_ALIVE_OFF);
    eprintln!("FW_ALIVE_OFF (0x030bf000) terminal = {alive:?}  (want Ok(0x030bb000))");
    eprintln!("explicit stores to FW_ALIVE_OFF: {}", alive_writes.len());
    for (n, pc, val) in &alive_writes {
        eprintln!("  n={n} pc={pc:#x} value={val:#010x}");
    }
}
