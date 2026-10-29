//! Behavioural tests for FLIX bundle decode, through the public `decode()`
//! entry point only.
//!
//! The differential tests are the acceptance gate: they compare `decode()`
//! against `xtdis`, a disassembler over GNU binutils libisa linked with this
//! core's configuration, on tens of thousands of pseudo-random bundles of each
//! format. They run where the `XTDIS` environment variable names the binary and
//! skip otherwise (CI); `XTDIS_REQUIRED=1` turns the skip into a failure.
//!
//! The contract checked for every bundle, from the oracle's view of it:
//! - undecodable bundle, or any slot the oracle cannot decode -> `Op::Unknown`;
//! - inert slots (`excw`, or the slot's designated `nop`) are dropped;
//! - any remaining slot whose mnemonic is outside [`SUPPORTED`], or is a
//!   reserved form (`slli` by 32), -> `Op::Unknown` for the whole bundle;
//! - otherwise 0 real slots -> `Op::Nop`, 1 -> that op, and >= 2 ->
//!   `Op::Flix1 { ops }` in slot order for `xt_format1`, `Op::Unknown` for
//!   `xt_format2` (which supports one real op only);
//! - every decode of an `op0` 0xe/0xf bundle has `len == 8`.
//!
//! Decoding runs at pc 0: the oracle prints `l32r`/`j` operands in their
//! location-independent form, which is what our pc-relative formulas reduce to
//! at pc 0 (`l32r` base `(0+3)&!3 = 0`; the `j` offset as printed).

use super::super::decode;
use super::Op;

#[test]
fn decodes_l32r_bundle_collapsing_excw() {
    // A hand-built xt_format2 bundle, decoded at pc 0x2000:
    // `7f 80 ff 01 00 00 00 00`.
    //   slot0: l32r a7, imm16=0xff80 (op0=1 in slot0 bits[20:23], at=7 in
    //          bits[0:3], imm16 in bits[4:19]): slot0 = 0x1ff807,
    //          insn0 = (slot0 << 4) | 0xf = 0x01ff807f.
    //   slot1: insn1 = 0 -> excw (inert).
    // Target: base = (0x2000+3)&~3 = 0x2000; offset = (0xffff0000|0xff80)
    // << 2 = 0xfffffe00 (-0x200), so target = 0x2000 - 0x200 = 0x1e00.
    let d = decode(&[0x7f, 0x80, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00], 0x2000);
    assert_eq!(d.len, 8, "xt_format2 bundle is 8 bytes");
    assert!(matches!(d.op, Op::L32r { t: 7, target: 0x1e00 }), "got {:?}", d.op);
}

#[test]
fn xt_format1_all_inert_collapses_to_nop() {
    // All three slots excw: a pure 8-byte barrier -> Op::Nop, len 8.
    let d = decode(&[0x0e, 0, 0, 0, 0, 0, 0, 0], 0x3000);
    assert_eq!(d.len, 8);
    assert!(matches!(d.op, Op::Nop), "got {:?}", d.op);
}

#[test]
fn truncated_bundle_does_not_panic() {
    // Fewer than 8 bytes at a 0xf selector -> Unknown, no out-of-bounds.
    let d = decode(&[0xff, 0xf8, 0x81], 0x2000);
    assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
}

// ---- differential tests against the xtdis oracle ----

/// Mnemonics the decoder must map to an `Op` in any FLIX slot where the oracle
/// shows them. Everything else real rejects the bundle.
const SUPPORTED: &[&str] = &[
    "add", "sub", "addx2", "addx4", "addx8", "subx8", "and", "or", "xor", "min", "minu", "maxu", "moveqz",
    "movnez", "mull", "mul16u", "mul16s", "src", "neg", "abs", "ssr", "ssl", "ssai", "nsau", "slli", "srai",
    "srli", "sll", "srl", "sext", "extui", "l8ui", "l16ui", "l16si", "l32i", "s8i", "s16i", "s32i", "movi",
    "movi.n", "addi", "addi.n", "addmi", "l32r", "mov.n", "j", "jx",
];

/// (mnemonic, operand values) for a decoded op, in the oracle's print order.
/// Registers -> index; immediates/targets -> value mod 2^32.
fn op_sig(op: &Op) -> (&'static str, Vec<u32>) {
    use Op::*;
    let rrr = |m, r: &u8, s: &u8, t: &u8| (m, vec![*r as u32, *s as u32, *t as u32]);
    match op {
        Add { r, s, t } => rrr("add", r, s, t),
        Sub { r, s, t } => rrr("sub", r, s, t),
        Addx2 { r, s, t } => rrr("addx2", r, s, t),
        Addx4 { r, s, t } => rrr("addx4", r, s, t),
        Addx8 { r, s, t } => rrr("addx8", r, s, t),
        Subx8 { r, s, t } => rrr("subx8", r, s, t),
        And { r, s, t } => rrr("and", r, s, t),
        Or { r, s, t } => rrr("or", r, s, t),
        Xor { r, s, t } => rrr("xor", r, s, t),
        Min { r, s, t } => rrr("min", r, s, t),
        Minu { r, s, t } => rrr("minu", r, s, t),
        Maxu { r, s, t } => rrr("maxu", r, s, t),
        Moveqz { r, s, t } => rrr("moveqz", r, s, t),
        Movnez { r, s, t } => rrr("movnez", r, s, t),
        Mull { r, s, t } => rrr("mull", r, s, t),
        Mul16u { r, s, t } => rrr("mul16u", r, s, t),
        Mul16s { r, s, t } => rrr("mul16s", r, s, t),
        Src { r, s, t } => rrr("src", r, s, t),
        Neg { r, t } => ("neg", vec![*r as u32, *t as u32]),
        Abs { r, t } => ("abs", vec![*r as u32, *t as u32]),
        Ssr { s } => ("ssr", vec![*s as u32]),
        Ssl { s } => ("ssl", vec![*s as u32]),
        Ssai { imm } => ("ssai", vec![*imm as u32]),
        Nsau { t, s } => ("nsau", vec![*t as u32, *s as u32]),
        Slli { r, s, imm } => ("slli", vec![*r as u32, *s as u32, *imm as u32]),
        Srai { r, t, imm } => ("srai", vec![*r as u32, *t as u32, *imm as u32]),
        Srli { r, t, imm } => ("srli", vec![*r as u32, *t as u32, *imm as u32]),
        Sll { r, s } => ("sll", vec![*r as u32, *s as u32]),
        Srl { r, t } => ("srl", vec![*r as u32, *t as u32]),
        Sext { r, s, imm } => ("sext", vec![*r as u32, *s as u32, *imm as u32]),
        Extui { r, t, shiftimm, maskimm } => {
            ("extui", vec![*r as u32, *t as u32, *shiftimm as u32, *maskimm as u32])
        }
        L8ui { t, s, imm } => ("l8ui", vec![*t as u32, *s as u32, *imm]),
        L16ui { t, s, imm } => ("l16ui", vec![*t as u32, *s as u32, *imm]),
        L16si { t, s, imm } => ("l16si", vec![*t as u32, *s as u32, *imm]),
        L32i { t, s, imm } => ("l32i", vec![*t as u32, *s as u32, *imm]),
        S8i { t, s, imm } => ("s8i", vec![*t as u32, *s as u32, *imm]),
        S16i { t, s, imm } => ("s16i", vec![*t as u32, *s as u32, *imm]),
        S32i { t, s, imm } => ("s32i", vec![*t as u32, *s as u32, *imm]),
        Movi { t, imm } => ("movi", vec![*t as u32, *imm as u32]),
        MoviN { t, imm } => ("movi.n", vec![*t as u32, *imm as u32]),
        Addi { t, s, imm } => ("addi", vec![*t as u32, *s as u32, *imm as u32]),
        AddiN { t, s, imm } => ("addi.n", vec![*t as u32, *s as u32, *imm as u32]),
        Addmi { t, s, imm } => ("addmi", vec![*t as u32, *s as u32, *imm as u32]),
        L32r { t, target } => ("l32r", vec![*t as u32, *target]),
        MovN { t, s } => ("mov.n", vec![*t as u32, *s as u32]),
        J { target } => ("j", vec![*target]),
        Jx { s } => ("jx", vec![*s as u32]),
        other => panic!("op_sig: op {other:?} is not a FLIX slot op this test knows"),
    }
}

/// One slot as the oracle prints it.
#[derive(Debug)]
enum OracleSlot {
    Inert,
    Undefined,
    Real { mnem: String, ops: Vec<u32> },
}

/// One 8-byte position as the oracle prints it: `None` if undecodable.
type OracleBundle = Option<(String, Vec<OracleSlot>)>;

fn xtdis_path() -> Option<std::path::PathBuf> {
    let required = std::env::var_os("XTDIS_REQUIRED").is_some();
    let found = std::env::var_os("XTDIS").map(std::path::PathBuf::from).filter(|p| p.exists());
    if found.is_none() && required {
        panic!("XTDIS_REQUIRED is set but XTDIS does not name an existing xtdis binary");
    }
    found
}

/// Parse one oracle operand token (`a12`, `0x1f`, `18`).
fn parse_operand(tok: &str) -> Option<u32> {
    if let Some(reg) = tok.strip_prefix('a').and_then(|n| n.parse::<u32>().ok()) {
        return Some(reg);
    }
    if let Some(hex) = tok.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).ok();
    }
    tok.parse::<u32>().ok()
}

/// Run the oracle once per 8-byte bundle (`XTDIS_EACH=8` keeps one bad bundle
/// from desynchronising the rest) and parse its output, indexed by bundle.
fn oracle(xtdis: &std::path::Path, bundles: &[[u8; 8]]) -> Vec<OracleBundle> {
    let hex: String = bundles.iter().flatten().map(|b| format!("{b:02x}")).collect();
    let out = std::process::Command::new(xtdis)
        .env("XTDIS_EACH", "8")
        .arg("0x0")
        .arg(&hex)
        .output()
        .expect("run xtdis");
    assert!(out.status.success(), "xtdis failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();

    let mut result: Vec<OracleBundle> = (0..bundles.len()).map(|_| None).collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let (addr, rest) = line.split_once(':').expect("xtdis header line");
        let idx = usize::from_str_radix(addr.trim(), 16).expect("xtdis address") / 8;
        i += 1;
        if rest.contains("<undecodable>") {
            continue;
        }
        let bracket = rest.split('[').nth(1).expect("format bracket");
        let format = bracket.split(',').next().unwrap().trim().to_string();
        let nslots: usize = bracket
            .split(',')
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let mut slots = Vec::with_capacity(nslots);
        for _ in 0..nslots {
            let body = lines[i].split_once(':').expect("slot line").1;
            i += 1;
            let is_nop = body.contains("[NOP]");
            let mut toks = body.split_whitespace().filter(|t| *t != "[NOP]");
            let mnem = toks.next().unwrap_or("").to_string();
            slots.push(if mnem == "<undef>" {
                OracleSlot::Undefined
            } else if is_nop || mnem == "excw" || mnem == "nop" {
                OracleSlot::Inert
            } else {
                OracleSlot::Real { mnem, ops: toks.filter_map(parse_operand).collect() }
            });
        }
        result[idx] = Some((format, slots));
    }
    result
}

/// What `decode()` must return for a bundle the oracle described.
enum Expect<'a> {
    Unknown,
    Nop,
    Ops(Vec<(&'a str, &'a [u32])>),
}

fn expectation<'a>(selector: u8, oracle: &'a OracleBundle) -> Expect<'a> {
    let Some((format, slots)) = oracle else {
        return Expect::Unknown;
    };
    let want_format = if selector == 0xe {
        "xt_format1"
    } else {
        "xt_format2"
    };
    assert_eq!(format, want_format, "op0 {selector:#x} decoded as {format}");
    let mut real = Vec::new();
    for slot in slots {
        match slot {
            OracleSlot::Inert => {}
            OracleSlot::Undefined => return Expect::Unknown,
            OracleSlot::Real { mnem, ops } => {
                let reserved = mnem == "slli" && ops.get(2).is_some_and(|&a| a >= 32);
                if !SUPPORTED.contains(&mnem.as_str()) || reserved {
                    return Expect::Unknown;
                }
                real.push((mnem.as_str(), ops.as_slice()));
            }
        }
    }
    match real.len() {
        0 => Expect::Nop,
        1 => Expect::Ops(real),
        _ if selector == 0xe => Expect::Ops(real),
        _ => Expect::Unknown,
    }
}

/// Sweep `chunks * 1000` pseudo-random bundles with the given `op0` selector
/// nibble and check every one against the oracle. Returns per-mnemonic counts
/// of real slot ops compared.
fn sweep(
    xtdis: &std::path::Path,
    selector: u8,
    seed: u64,
    chunks: usize,
) -> std::collections::BTreeMap<String, usize> {
    // Deterministic LCG (numerical-recipes constants): reproducible sweeps.
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 32) as u32
    };
    let mut counts = std::collections::BTreeMap::new();
    for _ in 0..chunks {
        let bundles: Vec<[u8; 8]> = (0..1000)
            .map(|_| {
                let mut b = [0u8; 8];
                for x in b.iter_mut() {
                    *x = next() as u8;
                }
                b[0] = (b[0] & 0xf0) | selector;
                b
            })
            .collect();
        for (b, ora) in bundles.iter().zip(oracle(xtdis, &bundles)) {
            let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
            let d = decode(b, 0);
            assert_eq!(d.len, 8, "bundle {hex}: len");
            match expectation(selector, &ora) {
                Expect::Unknown => {
                    assert!(
                        matches!(d.op, Op::Unknown { .. }),
                        "bundle {hex}: want Unknown, got {:?}; oracle {ora:?}",
                        d.op
                    )
                }
                Expect::Nop => {
                    assert!(matches!(d.op, Op::Nop), "bundle {hex}: want Nop, got {:?}; oracle {ora:?}", d.op)
                }
                Expect::Ops(want) => {
                    let got: Vec<&Op> = match &d.op {
                        Op::Flix1 { ops } if want.len() >= 2 => ops.iter().collect(),
                        op if want.len() == 1 => vec![op],
                        other => {
                            panic!("bundle {hex}: want {} ops, got {other:?}; oracle {ora:?}", want.len())
                        }
                    };
                    assert_eq!(
                        got.len(),
                        want.len(),
                        "bundle {hex}: op count; got {:?}; oracle {ora:?}",
                        d.op
                    );
                    for (op, (mnem, ops)) in got.iter().zip(&want) {
                        let (my_mnem, my_ops) = op_sig(op);
                        assert_eq!(
                            (my_mnem, my_ops.as_slice()),
                            (*mnem, *ops),
                            "bundle {hex}: decoded {:?}; oracle {ora:?}",
                            d.op
                        );
                        *counts.entry(mnem.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    counts
}

#[test]
fn xt_format1_matches_xtdis_oracle() {
    let Some(xtdis) = xtdis_path() else {
        eprintln!("skip: XTDIS not set or oracle not present");
        return;
    };
    let counts = sweep(&xtdis, 0xe, 0x1234_5678_9abc_def0, 40);
    let total: usize = counts.values().sum();
    eprintln!("xt_format1 vs xtdis: {total} real slot ops compared: {counts:?}");
    assert!(total > 20000, "too few real format1 ops compared ({total}): oracle parse broken?");
}

#[test]
fn xt_format2_matches_xtdis_oracle() {
    let Some(xtdis) = xtdis_path() else {
        eprintln!("skip: XTDIS not set or oracle not present");
        return;
    };
    let counts = sweep(&xtdis, 0xf, 0x0fed_cba9_8765_4321, 100);
    let total: usize = counts.values().sum();
    eprintln!("xt_format2 vs xtdis: {total} real slot ops compared: {counts:?}");
    assert!(total > 2000, "too few real format2 ops compared ({total}): oracle parse broken?");
}
