//! Decode sweep: compare the emulator's decoder against Chess listings.
//!
//! Chess (AMD's own compiler) is ground truth for what silicon executes.
//! Every Chess build writes `<prj>/elfs_main_core_*/elfs_main_core_*.elf.lst`,
//! one line per bundle: the decimal program-memory address, the raw bytes and
//! every slot's assembly. This module parses those listings, decodes each
//! bundle's bytes with the emulator's real decoder and classifies the result:
//!
//! | class | meaning |
//! |-------|---------|
//! | `DecodeFailure` | the decoder rejects a slot or the bundle |
//! | `MnemonicMismatch` | both decode, the instruction differs from Chess's text (needs review) |
//! | `OperandMismatch` | same mnemonic, operands differ (best effort, needs review) |
//! | `Match` | everything pairs up |
//!
//! Slot assignment never relies on listing columns: the bundle bytes go to the
//! decoder, and decoded slots are paired with Chess texts by mnemonic.
//! `bin/decode_sweep.rs` is the command-line driver.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::interpreter::bundle::SlotType;
use crate::interpreter::decode::probe::{SlotOutcome, SlotProbe};
use crate::interpreter::decode::InstructionDecoder;

// ---------------------------------------------------------------------------
// Listing parser
// ---------------------------------------------------------------------------

/// One bundle line of a Chess `.elf.lst`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LstBundle {
    /// Program-memory byte address (the listing prints it in decimal).
    pub addr: u32,
    /// Bundle bytes in memory (little-endian) order, as the decoder reads them.
    pub bytes: Vec<u8>,
    /// Per-slot assembly texts in listing order, trimmed, empties dropped.
    pub texts: Vec<String>,
}

/// Parse a Chess listing. Directives (`.label`, `.delay_slot`, ...), comments
/// and data-section lines are skipped; only `<decimal addr> <0xNN bytes> <asm>`
/// lines are bundles.
pub fn parse_lst(text: &str) -> Vec<LstBundle> {
    text.lines().filter_map(parse_lst_line).collect()
}

fn parse_lst_line(line: &str) -> Option<LstBundle> {
    let mut rest = line.trim_start();
    let (addr_tok, after) = split_token(rest);
    if addr_tok.is_empty() || !addr_tok.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let addr: u32 = addr_tok.parse().ok()?;
    rest = after;

    let mut bytes = Vec::new();
    loop {
        let trimmed = rest.trim_start();
        let (tok, after) = split_token(trimmed);
        match tok.strip_prefix("0x").filter(|h| h.len() == 2) {
            Some(hex) => match u8::from_str_radix(hex, 16) {
                Ok(b) => {
                    bytes.push(b);
                    rest = after;
                }
                Err(_) => break,
            },
            None => {
                rest = trimmed;
                break;
            }
        }
    }
    if bytes.is_empty() {
        return None;
    }
    // Chess prints the bundle as one big-endian number (the format marker sits
    // in the low nibble of the LAST byte), so reverse to memory order.
    bytes.reverse();
    let texts = rest
        .split(';')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    Some(LstBundle { addr, bytes, texts })
}

fn split_token(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

// ---------------------------------------------------------------------------
// Chess instruction text
// ---------------------------------------------------------------------------

/// One slot's assembly as Chess printed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChessInstr {
    /// Full text, e.g. `PADDS [sp], #32`.
    pub text: String,
    /// Upper-cased mnemonic including dot suffixes, e.g. `VLDA.CONV.FP32.BF16`.
    pub mnemonic: String,
    /// Register-like tokens, lowercased, in order.
    pub regs: Vec<String>,
    /// Integer literals, in order.
    pub imms: Vec<i64>,
}

impl ChessInstr {
    pub fn parse(text: &str) -> Self {
        let text = text.trim();
        let (mn, ops) = split_token(text);
        let (regs, imms) = scan_operands(ops);
        ChessInstr { text: text.to_string(), mnemonic: mn.to_uppercase(), regs, imms }
    }

    pub fn is_nop(&self) -> bool {
        self.mnemonic.starts_with("NOP")
    }
}

/// Split operand text into register-like identifiers and integer literals.
fn scan_operands(ops: &str) -> (Vec<String>, Vec<i64>) {
    let b = ops.as_bytes();
    let (mut regs, mut imms) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                i += 1;
            }
            regs.push(ops[s..i].to_lowercase());
        } else if c.is_ascii_digit() || (c == b'-' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let s = i;
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            if let Some(v) = parse_int(&ops[s..i]) {
                imms.push(v);
            }
        } else {
            i += 1;
        }
    }
    (regs, imms)
}

fn parse_int(tok: &str) -> Option<i64> {
    let (neg, body) = match tok.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, tok),
    };
    let v = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(h) => i64::from_str_radix(h, 16).ok()?,
        None => body.parse::<i64>().ok()?,
    };
    Some(if neg { -v } else { v })
}

/// True when LLVM instruction `llvm_name` (e.g. `VLDA_CONV_FP32_BF16_ag_idx`)
/// plausibly spells Chess mnemonic `chess` (e.g. `VLDA.CONV.FP32.BF16`).
/// The two naming schemes differ, so this is a token-prefix test either way
/// round; mismatches are reported for review, never asserted as bugs.
pub fn mnemonic_matches(llvm_name: &str, chess: &str) -> bool {
    let llvm: Vec<String> = llvm_name.to_uppercase().split('_').map(str::to_string).collect();
    let ch: Vec<String> = chess.to_uppercase().split(['.', '_']).map(str::to_string).collect();
    let n = llvm.len().min(ch.len());
    n > 0 && llvm[..n] == ch[..n]
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Bundle-level class, ordered worst first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleClass {
    DecodeFailure,
    MnemonicMismatch,
    OperandMismatch,
    Match,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailedSlot {
    pub slot: String,
    pub bits: String,
    pub width: u8,
    /// `ffi_no_match`, `lookup_miss:<llvm name>` or `operand_map_failed:<llvm name>`.
    pub cause: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairNote {
    pub llvm: String,
    pub chess: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleResult {
    pub addr: u32,
    pub bytes: String,
    pub chess: Vec<String>,
    pub class: BundleClass,
    pub failed_slots: Vec<FailedSlot>,
    /// Chess texts no decoded slot accounted for (only filled for failures,
    /// where they are the candidates for the failed slots' text).
    pub unexplained_chess: Vec<String>,
    /// True when exactly one failed slot and one unexplained text pair up.
    pub failure_text_certain: bool,
    pub mnemonic_mismatches: Vec<PairNote>,
    pub operand_mismatches: Vec<PairNote>,
    /// Decoder rejected the whole bundle (size/format) rather than a slot.
    pub bundle_error: Option<String>,
}

fn slot_name(t: SlotType) -> &'static str {
    match t {
        SlotType::Lda => "lda",
        SlotType::Ldb => "ldb",
        SlotType::Alu => "alu",
        SlotType::Mv => "mv",
        SlotType::St => "st",
        SlotType::Vec => "vec",
        SlotType::Lng => "lng",
        SlotType::Nop => "nop",
    }
}

fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ")
}

/// Is `needle` (multiset) contained in `hay` (multiset)?
fn multiset_contains<T: Ord + Clone>(hay: &[T], needle: &[T]) -> bool {
    let mut h: BTreeMap<&T, i32> = BTreeMap::new();
    for x in hay {
        *h.entry(x).or_default() += 1;
    }
    needle.iter().all(|x| match h.get_mut(x) {
        Some(c) if *c > 0 => {
            *c -= 1;
            true
        }
        _ => false,
    })
}

/// Registers Chess spells out that llvm-aie models as implicit operands.
const IMPLICIT_IN_LLVM: [&str; 2] = ["sp", "lr"];

/// LLVM names some register aliases with a dimension suffix (`d0_3d`) where
/// Chess prints the bare name (`d0`).
fn normalize_llvm_reg(r: &str) -> String {
    r.strip_suffix("_3d").unwrap_or(r).to_string()
}

/// Decode one listing bundle with the real decoder and classify it.
pub fn classify_bundle(decoder: &InstructionDecoder, b: &LstBundle) -> BundleResult {
    let chess: Vec<ChessInstr> = b.texts.iter().map(|t| ChessInstr::parse(t)).collect();
    let probes = decoder.probe_bundle(&b.bytes);
    let mut r = classify_probes(b, &chess, &probes);

    // The decoder sizes a bundle from its format marker; the listing's byte
    // count is the truth. Disagreement means the decoder would mis-step.
    let want = crate::interpreter::bundle::detect_format(&b.bytes).size_bytes() as usize;
    if want != b.bytes.len() {
        let cause = format!("size_mismatch:decoder={want},listing={}", b.bytes.len());
        r.bundle_error = Some(cause.clone());
        r.failed_slots.push(FailedSlot {
            slot: "bundle".into(),
            bits: format!("0x{:02x}", b.bytes.first().copied().unwrap_or(0)),
            width: 0,
            cause,
        });
        r.class = BundleClass::DecodeFailure;
    }
    r
}

/// Number of slots the production `Decoder::decode` path leaves as unknown.
/// Must equal the probe's failed-slot count; the sweep reports any bundle
/// where it does not, since that would mean the probe drifted from `decode`.
pub fn production_unknown_slots(decoder: &InstructionDecoder, bytes: &[u8]) -> usize {
    use crate::interpreter::traits::Decoder;
    decoder
        .decode(bytes, 0)
        .map(|b| b.active_slots().filter(|s| s.semantic.is_none()).count())
        .unwrap_or(0)
}

/// Classification core, separated from the decoder so it is unit-testable
/// with synthetic probes.
pub fn classify_probes(b: &LstBundle, chess: &[ChessInstr], probes: &[SlotProbe]) -> BundleResult {
    let mut result = BundleResult {
        addr: b.addr,
        bytes: hex_bytes(&b.bytes),
        chess: b.texts.clone(),
        class: BundleClass::Match,
        failed_slots: Vec::new(),
        unexplained_chess: Vec::new(),
        failure_text_certain: false,
        mnemonic_mismatches: Vec::new(),
        operand_mismatches: Vec::new(),
        bundle_error: None,
    };

    let mut chess_left: Vec<&ChessInstr> = chess.iter().filter(|c| !c.is_nop()).collect();
    let mut decoded = Vec::new();
    for p in probes {
        match &p.outcome {
            SlotOutcome::Decoded(d) => {
                if !d.llvm_name.to_uppercase().starts_with("NOP") {
                    decoded.push(d);
                }
            }
            SlotOutcome::FfiNoMatch => result.failed_slots.push(failed(p, "ffi_no_match".into())),
            SlotOutcome::LookupMiss { llvm_name } => {
                result.failed_slots.push(failed(p, format!("lookup_miss:{llvm_name}")))
            }
            SlotOutcome::OperandMapFailed { llvm_name } => {
                result.failed_slots.push(failed(p, format!("operand_map_failed:{llvm_name}")))
            }
        }
    }

    // Pair decoded slots with Chess texts by mnemonic, preferring a pair whose
    // operands also agree.
    let mut leftover_decoded = Vec::new();
    for d in decoded {
        let candidates: Vec<usize> = chess_left
            .iter()
            .enumerate()
            .filter(|(_, c)| mnemonic_matches(&d.llvm_name, &c.mnemonic))
            .map(|(i, _)| i)
            .collect();
        let d_regs: Vec<String> = d.regs.iter().map(|r| normalize_llvm_reg(r)).collect();
        let operand_ok = |c: &ChessInstr| {
            // sp and lr are implicit operands in llvm-aie (stack-relative
            // memory ops, RET) but explicit in Chess's text.
            let regs: Vec<String> = c
                .regs
                .iter()
                .filter(|r| !(IMPLICIT_IN_LLVM.contains(&r.as_str()) && !d_regs.contains(r)))
                .cloned()
                .collect();
            multiset_contains(&d_regs, &regs) && multiset_contains(&d.imms, &c.imms)
        };
        let pick = candidates
            .iter()
            .copied()
            .find(|&i| operand_ok(chess_left[i]))
            .or_else(|| candidates.first().copied());
        match pick {
            Some(i) => {
                let c = chess_left.remove(i);
                if !operand_ok(c) {
                    result.operand_mismatches.push(PairNote {
                        llvm: d.llvm_name.clone(),
                        chess: c.text.clone(),
                        detail: format!("decoded regs={:?} imms={:?}", d.regs, d.imms),
                    });
                }
            }
            None => leftover_decoded.push(d),
        }
    }

    // Decoded slots with no Chess partner, and Chess texts with no partner:
    // pair them off in order for the report.
    let failures = result.failed_slots.len();
    let mut chess_unpaired = chess_left;
    if failures == 0 {
        let n = leftover_decoded.len().max(chess_unpaired.len());
        for i in 0..n {
            let l = leftover_decoded.get(i).map(|d| d.llvm_name.clone()).unwrap_or_default();
            let c = chess_unpaired.get(i).map(|c| c.text.clone()).unwrap_or_default();
            result
                .mnemonic_mismatches
                .push(PairNote { llvm: l, chess: c, detail: String::new() });
        }
    } else {
        // Decoded-but-unmatched slots are mismatches in their own right; the
        // Chess texts left after removing as many as they could explain are the
        // candidates for the failed slots.
        for (i, d) in leftover_decoded.iter().enumerate() {
            let c = chess_unpaired.get(i).map(|c| c.text.clone()).unwrap_or_default();
            result.mnemonic_mismatches.push(PairNote {
                llvm: d.llvm_name.clone(),
                chess: c,
                detail: String::new(),
            });
        }
        let skip = leftover_decoded.len().min(chess_unpaired.len());
        chess_unpaired.drain(..skip);
        result.unexplained_chess = chess_unpaired.iter().map(|c| c.text.clone()).collect();
        result.failure_text_certain =
            failures == 1 && leftover_decoded.is_empty() && chess_unpaired.len() == 1;
    }

    result.class = if failures > 0 {
        BundleClass::DecodeFailure
    } else if !result.mnemonic_mismatches.is_empty() {
        BundleClass::MnemonicMismatch
    } else if !result.operand_mismatches.is_empty() {
        BundleClass::OperandMismatch
    } else {
        BundleClass::Match
    };
    result
}

fn failed(p: &SlotProbe, cause: String) -> FailedSlot {
    FailedSlot {
        slot: slot_name(p.slot_type).to_string(),
        bits: format!("0x{:x}", p.bits),
        width: p.width,
        cause,
    }
}

// ---------------------------------------------------------------------------
// Corpus-level aggregation
// ---------------------------------------------------------------------------

/// Where a failure was first seen.
#[derive(Debug, Clone, Serialize)]
pub struct Example {
    pub lst: String,
    pub addr: u32,
    pub bundle_bytes: String,
    pub chess_bundle: Vec<String>,
}

/// Failures sharing slot, Chess mnemonic and cause.
#[derive(Debug, Clone, Serialize)]
pub struct FailureGroup {
    pub slot: String,
    /// Chess mnemonic of the failed slot, or `?` when the bundle could not say.
    pub chess_mnemonic: String,
    pub cause: String,
    /// Occurrences across the deduplicated corpus.
    pub count: usize,
    pub distinct_lsts: usize,
    /// Slot bits seen, with counts (most frequent first, capped).
    pub bits: Vec<(String, usize)>,
    /// Chess texts seen for this group, with counts (capped).
    pub chess_texts: Vec<(String, usize)>,
    /// Whether every occurrence paired its failed slot with exactly one text.
    pub text_certain: bool,
    pub example: Example,
}

#[derive(Debug, Default, Serialize)]
pub struct Totals {
    pub decode_failure: usize,
    pub mnemonic_mismatch: usize,
    pub operand_mismatch: usize,
    #[serde(rename = "match")]
    pub matched: usize,
}

impl Totals {
    pub fn total(&self) -> usize {
        self.decode_failure + self.mnemonic_mismatch + self.operand_mismatch + self.matched
    }
    pub fn add(&mut self, c: BundleClass) {
        match c {
            BundleClass::DecodeFailure => self.decode_failure += 1,
            BundleClass::MnemonicMismatch => self.mnemonic_mismatch += 1,
            BundleClass::OperandMismatch => self.operand_mismatch += 1,
            BundleClass::Match => self.matched += 1,
        }
    }
}

/// Per-listing results (non-matching bundles only, to keep the JSON small).
#[derive(Debug, Serialize)]
pub struct LstReport {
    pub lst: String,
    pub totals: Totals,
    /// Sibling ELF `.text` agrees byte for byte with the listing (None: no ELF).
    pub elf_text_agrees: Option<bool>,
    pub flagged: Vec<BundleResult>,
}

pub fn sweep_listing(decoder: &InstructionDecoder, lst: &str, text: &str) -> LstReport {
    let mut totals = Totals::default();
    let mut flagged = Vec::new();
    for b in parse_lst(text) {
        let r = classify_bundle(decoder, &b);
        totals.add(r.class);
        if r.class != BundleClass::Match || r.bundle_error.is_some() {
            flagged.push(r);
        }
    }
    LstReport { lst: lst.to_string(), totals, elf_text_agrees: None, flagged }
}

const TOP_N: usize = 12;

fn top(map: BTreeMap<String, usize>) -> Vec<(String, usize)> {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(TOP_N);
    v
}

/// Group every decode failure across `reports` by (slot, Chess mnemonic, cause).
pub fn group_failures(reports: &[LstReport]) -> Vec<FailureGroup> {
    struct Acc {
        count: usize,
        lsts: BTreeSet<String>,
        bits: BTreeMap<String, usize>,
        texts: BTreeMap<String, usize>,
        certain: bool,
        example: Example,
    }
    let mut groups: BTreeMap<(String, String, String), Acc> = BTreeMap::new();
    for rep in reports {
        for b in rep.flagged.iter().filter(|b| b.class == BundleClass::DecodeFailure) {
            for f in &b.failed_slots {
                let (mn, text) = if b.failure_text_certain {
                    let t = &b.unexplained_chess[0];
                    (ChessInstr::parse(t).mnemonic, t.clone())
                } else {
                    ("?".to_string(), b.unexplained_chess.join(" ; "))
                };
                let key = (f.slot.clone(), mn, f.cause.clone());
                let acc = groups.entry(key).or_insert_with(|| Acc {
                    count: 0,
                    lsts: BTreeSet::new(),
                    bits: BTreeMap::new(),
                    texts: BTreeMap::new(),
                    certain: true,
                    example: Example {
                        lst: rep.lst.clone(),
                        addr: b.addr,
                        bundle_bytes: b.bytes.clone(),
                        chess_bundle: b.chess.clone(),
                    },
                });
                acc.count += 1;
                acc.lsts.insert(rep.lst.clone());
                *acc.bits.entry(f.bits.clone()).or_default() += 1;
                *acc.texts.entry(text).or_default() += 1;
                acc.certain &= b.failure_text_certain;
            }
        }
    }
    let mut out: Vec<FailureGroup> = groups
        .into_iter()
        .map(|((slot, chess_mnemonic, cause), a)| FailureGroup {
            slot,
            chess_mnemonic,
            cause,
            count: a.count,
            distinct_lsts: a.lsts.len(),
            bits: top(a.bits),
            chess_texts: top(a.texts),
            text_certain: a.certain,
            example: a.example,
        })
        .collect();
    out.sort_by(|a, b| b.count.cmp(&a.count));
    out
}

/// A (LLVM name, Chess mnemonic) pair seen in mismatching bundles.
#[derive(Debug, Clone, Serialize)]
pub struct MismatchGroup {
    pub llvm: String,
    pub chess_mnemonic: String,
    pub count: usize,
    pub example_lst: String,
    pub example_addr: u32,
    pub example_chess: String,
    pub example_detail: String,
}

/// Group mnemonic (`operand == false`) or operand (`true`) mismatches.
pub fn group_mismatches(reports: &[LstReport], operand: bool) -> Vec<MismatchGroup> {
    let mut m: BTreeMap<(String, String), MismatchGroup> = BTreeMap::new();
    for rep in reports {
        for b in &rep.flagged {
            let notes = if operand {
                &b.operand_mismatches
            } else {
                &b.mnemonic_mismatches
            };
            for n in notes {
                let mn = ChessInstr::parse(&n.chess).mnemonic;
                let g = m.entry((n.llvm.clone(), mn.clone())).or_insert_with(|| MismatchGroup {
                    llvm: n.llvm.clone(),
                    chess_mnemonic: mn,
                    count: 0,
                    example_lst: rep.lst.clone(),
                    example_addr: b.addr,
                    example_chess: n.chess.clone(),
                    example_detail: n.detail.clone(),
                });
                g.count += 1;
            }
        }
    }
    let mut v: Vec<_> = m.into_values().collect();
    v.sort_by(|a, b| b.count.cmp(&a.count));
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::decode::probe::DecodedSlot;

    /// The bundle Chess emitted for the srs_d spill: `PADDS [sp], #32` in the
    /// store slot, which llvm-aie has no def for. Copied from the listing in
    /// build/experiments/spill-decode (address 3520).
    const SPILL_LINE: &str = "        3520    0x01 0xe7 0x8a 0x0b 0x10 0x00 0x3c 0xb3 0xbe 0x42 0x00 0x0e 0x48 0x7f            MOVA r30, #15;                PADDB [p2], #64;              PADDS [sp], #32;              MOVXM p3, #498660";
    /// Memory order: the listing's bytes reversed.
    const SPILL_BYTES: [u8; 14] =
        [0x7f, 0x48, 0x0e, 0x00, 0x42, 0xbe, 0xb3, 0x3c, 0x00, 0x10, 0x0b, 0x8a, 0xe7, 0x01];

    #[test]
    fn parses_bundle_line() {
        let bs = parse_lst(SPILL_LINE);
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].addr, 3520);
        assert_eq!(bs[0].bytes, SPILL_BYTES);
        assert_eq!(bs[0].texts, ["MOVA r30, #15", "PADDB [p2], #64", "PADDS [sp], #32", "MOVXM p3, #498660"]);
    }

    #[test]
    fn skips_directives_comments_and_data() {
        let text = "// header\n.text_segment PM 0\n.label foo\n.delay_slot\n\t0x0\n\t0x1d\n\
                    \x20         0    0x00 0x01     NOPX\n.function_start\n";
        let bs = parse_lst(text);
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].bytes, [0x01, 0x00]);
        assert_eq!(bs[0].texts, ["NOPX"]);
    }

    #[test]
    fn parses_short_text_gap_after_16_bytes() {
        // 16-byte bundles leave only two spaces before the text.
        let line = "      7024    0x00 0x64 0x56 0x01 0x58 0x00 0x00 0x00 0x00 0x04 0x3d 0xf8 0x10 0x00 0x03 0xc0  MOVA lc, #688;   PADDB [sp], #32;  NOPS";
        let bs = parse_lst(line);
        assert_eq!(bs[0].bytes.len(), 16);
        assert_eq!(bs[0].texts, ["MOVA lc, #688", "PADDB [sp], #32", "NOPS"]);
    }

    #[test]
    fn chess_operands_scanned() {
        let c = ChessInstr::parse("LDA p0, [p6], #-4");
        assert_eq!(c.mnemonic, "LDA");
        assert_eq!(c.regs, ["p0", "p6"]);
        assert_eq!(c.imms, [-4]);
        let c = ChessInstr::parse("MOVXM p3, #0x10");
        assert_eq!(c.imms, [16]);
    }

    #[test]
    fn mnemonic_naming_schemes_meet() {
        assert!(mnemonic_matches("PADDB_sp_imm", "PADDB"));
        assert!(!mnemonic_matches("PADDA_sp_imm", "PADDB"));
        assert!(mnemonic_matches("VLDA_CONV_FP32_BF16_ag_idx", "VLDA.CONV.FP32.BF16"));
        assert!(mnemonic_matches("VLDA_CONV_FP32_BF16_ag_idx", "VLDA.CONV"));
        assert!(mnemonic_matches("MOV_mv_scl", "MOV"));
    }

    fn decoded(name: &str, regs: &[&str], imms: &[i64]) -> SlotProbe {
        SlotProbe {
            slot_type: SlotType::Alu,
            bits: 0,
            width: 20,
            outcome: SlotOutcome::Decoded(DecodedSlot {
                llvm_name: name.into(),
                regs: regs.iter().map(|s| s.to_string()).collect(),
                imms: imms.to_vec(),
            }),
        }
    }

    fn bundle(texts: &[&str]) -> (LstBundle, Vec<ChessInstr>) {
        let b =
            LstBundle { addr: 0, bytes: vec![0; 4], texts: texts.iter().map(|s| s.to_string()).collect() };
        let c = b.texts.iter().map(|t| ChessInstr::parse(t)).collect();
        (b, c)
    }

    #[test]
    fn classifies_match_ignoring_nops_and_order() {
        let (b, c) = bundle(&["NOPA", "MOV r17, r1", "ADD r2, r2, #4"]);
        let probes =
            [decoded("ADD_add_r_ri", &["r2", "r2"], &[4]), decoded("MOV_mv_scl", &["r17", "r1"], &[])];
        assert_eq!(classify_probes(&b, &c, &probes).class, BundleClass::Match);
    }

    #[test]
    fn classifies_operand_mismatch() {
        let (b, c) = bundle(&["MOV r17, r1"]);
        let probes = [decoded("MOV_mv_scl", &["r17", "r2"], &[])];
        let r = classify_probes(&b, &c, &probes);
        assert_eq!(r.class, BundleClass::OperandMismatch);
        assert_eq!(r.operand_mismatches.len(), 1);
    }

    #[test]
    fn implicit_sp_and_lr_and_3d_alias_are_tolerated() {
        let (b, c) = bundle(&["ST lr, [sp, #-4]", "VST.3D.SRS.d8.s32 cm3, s1, [p6], d0"]);
        let probes = [
            decoded("ST_dms_spill", &["lr"], &[-4]),
            decoded("VST_3D_SRS_D8_S32", &["p6", "d0_3d", "cm3", "s1"], &[]),
        ];
        assert_eq!(classify_probes(&b, &c, &probes).class, BundleClass::Match);
    }

    #[test]
    fn classifies_mnemonic_mismatch() {
        let (b, c) = bundle(&["PADDB [sp], #32"]);
        let probes = [decoded("PADDA_sp_imm", &["sp"], &[32])];
        assert_eq!(classify_probes(&b, &c, &probes).class, BundleClass::MnemonicMismatch);
    }

    #[test]
    fn spill_bundle_fixture_pairs_failure_with_chess_text() {
        // Three slots decode; the store slot holds PADDS [sp], #32 and fails.
        let b = parse_lst(SPILL_LINE).remove(0);
        let c: Vec<_> = b.texts.iter().map(|t| ChessInstr::parse(t)).collect();
        let probes = [
            decoded("MOVA_lda_cg", &["r30"], &[15]),
            decoded("PADDB_ptr_imm", &["p2", "p2"], &[64]),
            decoded("MOVXM_lng_cg", &["p3"], &[498660]),
            SlotProbe {
                slot_type: SlotType::St,
                bits: 0x1c9,
                width: 21,
                outcome: SlotOutcome::LookupMiss { llvm_name: "PHI".into() },
            },
        ];
        let r = classify_probes(&b, &c, &probes);
        assert_eq!(r.class, BundleClass::DecodeFailure);
        assert_eq!(r.failed_slots[0].slot, "st");
        assert_eq!(r.failed_slots[0].cause, "lookup_miss:PHI");
        assert_eq!(r.unexplained_chess, ["PADDS [sp], #32"]);
        assert!(r.failure_text_certain);
    }

    #[test]
    fn real_decoder_rejects_spill_bundle_store_slot() {
        let decoder = InstructionDecoder::load_default();
        let b = parse_lst(SPILL_LINE).remove(0);
        let r = classify_bundle(&decoder, &b);
        assert_eq!(r.class, BundleClass::DecodeFailure, "{r:?}");
        assert_eq!(r.failed_slots.len(), 1, "{r:?}");
        assert_eq!(r.failed_slots[0].slot, "st");
        assert_eq!(r.failed_slots[0].bits, "0x1c9");
        assert_eq!(r.unexplained_chess, ["PADDS [sp], #32"]);
    }

    #[test]
    fn groups_failures_by_slot_mnemonic_and_cause() {
        let decoder = InstructionDecoder::load_default();
        let b = parse_lst(SPILL_LINE).remove(0);
        let mk = |lst: &str| LstReport {
            lst: lst.into(),
            totals: Totals::default(),
            elf_text_agrees: None,
            flagged: vec![classify_bundle(&decoder, &b)],
        };
        let groups = group_failures(&[mk("a.lst"), mk("b.lst")]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].count, 2);
        assert_eq!(groups[0].distinct_lsts, 2);
        assert_eq!(groups[0].chess_mnemonic, "PADDS");
    }
}
