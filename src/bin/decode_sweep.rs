//! Sweep Chess `.elf.lst` listings for encodings the emulator cannot decode.
//!
//! ```text
//! decode-sweep --out DIR PATH...
//! ```
//! Each PATH is a `.elf.lst` file or a directory searched recursively.
//! Listings with identical content (ignoring the `//` header, which carries a
//! timestamp and the build path) are counted once. Paths containing
//! `xdna-emu-priv` are skipped. Writes `DIR/sweep.json` and `DIR/summary.txt`
//! and prints the summary.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process;

use serde::Serialize;
use xdna_emu::decode_sweep::*;
use xdna_emu::interpreter::decode::InstructionDecoder;
use xdna_emu::parser::elf::AieElf;

#[derive(Serialize)]
struct Corpus {
    lsts_found: usize,
    lsts_unique: usize,
    skipped_priv: Vec<String>,
    skipped_duplicate: Vec<(String, String)>,
    bundles: usize,
    elf_text_checked: usize,
    elf_text_disagree: Vec<String>,
    probe_disagrees_with_decode: usize,
}

#[derive(Serialize)]
struct Sweep {
    corpus: Corpus,
    totals: Totals,
    failure_groups: Vec<FailureGroup>,
    mnemonic_mismatch_groups: Vec<MismatchGroup>,
    operand_mismatch_groups: Vec<MismatchGroup>,
    listings: Vec<LstReport>,
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        let mut ents: Vec<_> = rd.flatten().map(|e| e.path()).collect();
        ents.sort();
        for p in ents {
            collect(&p, out);
        }
    } else if path.to_string_lossy().ends_with(".elf.lst") {
        out.push(path.to_path_buf());
    }
}

fn body_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in text.lines().filter(|l| !l.starts_with("//")) {
        l.hash(&mut h);
    }
    h.finish()
}

/// Does the sibling ELF's .text hold exactly the bytes the listing claims?
fn elf_agrees(lst: &Path, text: &str) -> Option<bool> {
    let elf_path = PathBuf::from(lst.to_string_lossy().trim_end_matches(".lst"));
    let data = std::fs::read(&elf_path).ok()?;
    let elf = AieElf::parse(&data).ok()?;
    let sect = elf.text_section()?;
    Some(parse_lst(text).iter().all(|b| {
        let a = b.addr as usize;
        sect.get(a..a + b.bytes.len()) == Some(&b.bytes[..])
    }))
}

fn main() {
    let mut out_dir = None;
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("usage: decode-sweep --out DIR PATH...");
                process::exit(0);
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    let Some(out_dir) = out_dir else {
        eprintln!("decode-sweep: --out DIR is required");
        process::exit(2);
    };
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let mut found = Vec::new();
    for p in &inputs {
        collect(p, &mut found);
    }
    found.sort();
    found.dedup();

    let decoder = InstructionDecoder::load_default();
    let mut seen: BTreeMap<u64, String> = BTreeMap::new();
    let mut corpus = Corpus {
        lsts_found: found.len(),
        lsts_unique: 0,
        skipped_priv: Vec::new(),
        skipped_duplicate: Vec::new(),
        bundles: 0,
        elf_text_checked: 0,
        elf_text_disagree: Vec::new(),
        probe_disagrees_with_decode: 0,
    };
    let mut reports = Vec::new();
    let mut totals = Totals::default();
    let mut distinct_bundles: HashSet<(String, String)> = HashSet::new();
    for lst in &found {
        let name = lst.display().to_string();
        if name.contains("xdna-emu-priv") {
            corpus.skipped_priv.push(name);
            continue;
        }
        let Ok(text) = std::fs::read_to_string(lst) else {
            eprintln!("decode-sweep: cannot read {name}");
            continue;
        };
        let h = body_hash(&text);
        if let Some(first) = seen.get(&h) {
            corpus.skipped_duplicate.push((name, first.clone()));
            continue;
        }
        seen.insert(h, name.clone());
        let mut rep = sweep_listing(&decoder, &name, &text);
        rep.elf_text_agrees = elf_agrees(lst, &text);
        if let Some(ok) = rep.elf_text_agrees {
            corpus.elf_text_checked += 1;
            if !ok {
                corpus.elf_text_disagree.push(name.clone());
            }
        }
        for b in parse_lst(&text) {
            distinct_bundles.insert((b.bytes.iter().map(|x| format!("{x:02x}")).collect(), String::new()));
        }
        corpus.bundles += rep.totals.total();
        corpus.probe_disagrees_with_decode += rep
            .flagged
            .iter()
            .filter(|b| b.bundle_error.as_deref() == Some("probe_disagrees_with_decode"))
            .count();
        totals.decode_failure += rep.totals.decode_failure;
        totals.mnemonic_mismatch += rep.totals.mnemonic_mismatch;
        totals.operand_mismatch += rep.totals.operand_mismatch;
        totals.matched += rep.totals.matched;
        reports.push(rep);
    }
    corpus.lsts_unique = reports.len();

    let sweep = Sweep {
        failure_groups: group_failures(&reports),
        mnemonic_mismatch_groups: group_mismatches(&reports, false),
        operand_mismatch_groups: group_mismatches(&reports, true),
        corpus,
        totals,
        listings: reports,
    };

    let mut s = String::new();
    let c = &sweep.corpus;
    let _ = writeln!(
        s,
        "listings found {}, unique {}, duplicates skipped {}, priv skipped {}",
        c.lsts_found,
        c.lsts_unique,
        c.skipped_duplicate.len(),
        c.skipped_priv.len()
    );
    let _ = writeln!(s, "bundles {} (distinct byte patterns {})", c.bundles, distinct_bundles.len());
    let _ = writeln!(
        s,
        "sibling ELF .text checked {}, disagree {}; probe/decode disagreement {}",
        c.elf_text_checked,
        c.elf_text_disagree.len(),
        c.probe_disagrees_with_decode
    );
    let t = &sweep.totals;
    let _ = writeln!(
        s,
        "(a) decode failure {}  (b) mnemonic mismatch {}  (c) operand mismatch {}  (d) match {}",
        t.decode_failure, t.mnemonic_mismatch, t.operand_mismatch, t.matched
    );
    let _ = writeln!(s, "\nDECODE-FAILURE GROUPS ({}):", sweep.failure_groups.len());
    for g in &sweep.failure_groups {
        let _ = writeln!(
            s,
            "  {:>5}x {:<6} {:<10} {} bits={:?} text={:?}{} e.g. {}@{}",
            g.count,
            g.slot,
            g.chess_mnemonic,
            g.cause,
            g.bits.iter().take(3).map(|b| b.0.as_str()).collect::<Vec<_>>(),
            g.chess_texts.first().map(|t| t.0.as_str()).unwrap_or(""),
            if g.text_certain { "" } else { " (text uncertain)" },
            g.example.lst,
            g.example.addr
        );
    }
    for (title, groups) in
        [("MNEMONIC", &sweep.mnemonic_mismatch_groups), ("OPERAND", &sweep.operand_mismatch_groups)]
    {
        let _ = writeln!(s, "\n{title} MISMATCH GROUPS ({}), top 40:", groups.len());
        for g in groups.iter().take(40) {
            let _ = writeln!(
                s,
                "  {:>5}x llvm={} chess={} e.g. {:?} {}",
                g.count, g.llvm, g.chess_mnemonic, g.example_chess, g.example_detail
            );
        }
    }
    std::fs::write(out_dir.join("summary.txt"), &s).expect("write summary");
    std::fs::write(out_dir.join("sweep.json"), serde_json::to_string_pretty(&sweep).expect("json"))
        .expect("write json");
    print!("{s}");
}
