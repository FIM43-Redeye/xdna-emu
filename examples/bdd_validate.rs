//! BDD-based exhaustive decoder validation
//!
//! Reads 16-byte raw VLIW bundle patterns from stdin (as produced by
//! `bdd_enum --enumerate-all --format raw --expand --max 1`) and attempts
//! to decode each using our TableGen-based decoder.
//!
//! Each 16-byte record corresponds to one BDD root (in order). The output
//! is a tab-separated report mapping root indices to decoded instructions,
//! plus a summary of coverage gaps.
//!
//! Usage:
//!   bdd_enum --enumerate-all --format raw --expand --max 1 me_das.ena | \
//!       cargo run --release --example bdd_validate
//!
//! Or via the orchestration script:
//!   scripts/bdd-validate.sh

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::time::Instant;
use xdna_emu::interpreter::bundle::{extract_slots, SlotType};
use xdna_emu::interpreter::decode::InstructionDecoder;

fn slot_type_name(st: SlotType) -> &'static str {
    match st {
        SlotType::Lda => "lda",
        SlotType::Ldb => "ldb",
        SlotType::Alu => "alu",
        SlotType::Mv  => "mv",
        SlotType::St  => "st",
        SlotType::Vec => "vec",
        SlotType::Lng => "lng",
        SlotType::Nop => "nop",
    }
}

fn main() {
    let start = Instant::now();

    eprintln!("Loading decoder (TableGen parse)...");
    let decoder = InstructionDecoder::load_cached();
    let load_time = start.elapsed();
    eprintln!("Decoder loaded in {:.1}s", load_time.as_secs_f64());

    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    // Stats
    let mut total: usize = 0;
    let mut bundles_with_slots: usize = 0;
    let mut bundles_no_slots: usize = 0;  // format word yielded no extractable slots
    let mut total_slots: usize = 0;
    let mut slots_decoded: usize = 0;
    let mut slots_failed: usize = 0;
    let mut slots_nop: usize = 0;

    // Root-to-instruction mapping: root_index -> list of (slot, encoding_name, mnemonic)
    let mut encoding_counts: HashMap<String, usize> = HashMap::new();
    let mut mnemonic_counts: HashMap<String, usize> = HashMap::new();
    let mut slot_fail_examples: Vec<(usize, String, String, u64)> = Vec::new(); // (root, slot, hex, bits)

    // Header
    writeln!(out, "# BDD Exhaustive Decoder Validation").unwrap();
    writeln!(out, "# root\tfmt\tslots\tstatus\tdetails").unwrap();

    let mut buf = [0u8; 16];
    loop {
        match handle.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("Read error at root {}: {}", total, e);
                break;
            }
        }

        let root_index = total;
        total += 1;

        // Format word: lower 15 bits
        let format_word = u16::from_le_bytes([buf[0], buf[1]]) & 0x7FFF;

        // Extract slots from the bundle
        let extracted = extract_slots(&buf);

        if extracted.slots.is_empty() {
            bundles_no_slots += 1;
            writeln!(out, "{}\t0x{:04x}\t0\tNO_SLOTS\t-", root_index, format_word).unwrap();
            continue;
        }

        bundles_with_slots += 1;

        let mut slot_results: Vec<String> = Vec::new();
        let mut all_ok = true;

        for slot in &extracted.slots {
            total_slots += 1;

            if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                slots_nop += 1;
                slot_results.push(format!("{}:NOP", slot_type_name(slot.slot_type)));
                continue;
            }

            // Use the production FFI decode path.
            let ffi_slot = match slot.slot_type {
                SlotType::Alu => Some(xdna_emu::tablegen::decoder_ffi::Slot::Alu),
                SlotType::Lda => Some(xdna_emu::tablegen::decoder_ffi::Slot::Lda),
                SlotType::Ldb => Some(xdna_emu::tablegen::decoder_ffi::Slot::Ldb),
                SlotType::Lng => Some(xdna_emu::tablegen::decoder_ffi::Slot::Lng),
                SlotType::Mv  => Some(xdna_emu::tablegen::decoder_ffi::Slot::Mv),
                SlotType::St  => Some(xdna_emu::tablegen::decoder_ffi::Slot::St),
                SlotType::Vec => Some(xdna_emu::tablegen::decoder_ffi::Slot::Vec),
                _ => None,
            };
            match ffi_slot.and_then(|s| xdna_emu::tablegen::decoder_ffi::decode_slot(s, slot.bits)) {
                Some(decoded) => {
                    slots_decoded += 1;
                    let enc_name = &decoded.name;
                    // Derive mnemonic from name (lowercase, strip suffix).
                    let mnemonic = enc_name.split('_').next().unwrap_or(enc_name).to_lowercase();

                    slot_results.push(format!("{}:{}[{}]",
                        slot_type_name(slot.slot_type), mnemonic, enc_name));

                    *encoding_counts.entry(enc_name.clone()).or_insert(0) += 1;
                    *mnemonic_counts.entry(mnemonic).or_insert(0) += 1;
                }
                None => {
                    slots_failed += 1;
                    all_ok = false;

                    slot_results.push(format!("{}:FAIL(0x{:x})",
                        slot_type_name(slot.slot_type), slot.bits));

                    if slot_fail_examples.len() < 200 {
                        let bundle_hex: String = buf.iter()
                            .rev()
                            .map(|b| format!("{:02x}", b))
                            .collect();
                        slot_fail_examples.push((
                            root_index,
                            slot_type_name(slot.slot_type).to_string(),
                            bundle_hex,
                            slot.bits,
                        ));
                    }
                }
            }
        }

        let status = if all_ok { "OK" } else { "PARTIAL" };
        let detail = slot_results.join(" | ");

        writeln!(out, "{}\t0x{:04x}\t{}\t{}\t{}",
            root_index, format_word, extracted.slots.len(), status, detail).unwrap();

        // Progress
        if total % 2000 == 0 {
            eprint!("  {} roots...\r", total);
        }
    }

    // Flush per-root output
    out.flush().unwrap();

    // Summary to stderr
    let elapsed = start.elapsed();
    eprintln!("\n========================================");
    eprintln!("  BDD Decoder Validation Summary");
    eprintln!("========================================\n");

    eprintln!("Bundles processed:   {}", total);
    eprintln!("  With slots:        {} ({:.1}%)", bundles_with_slots,
        pct(bundles_with_slots, total));
    eprintln!("  No slots (bad fmt):{} ({:.1}%)", bundles_no_slots,
        pct(bundles_no_slots, total));
    eprintln!();

    eprintln!("Total slots:         {}", total_slots);
    eprintln!("  Decoded OK:        {} ({:.1}%)", slots_decoded,
        pct(slots_decoded, total_slots));
    eprintln!("  NOP:               {} ({:.1}%)", slots_nop,
        pct(slots_nop, total_slots));
    eprintln!("  FAILED:            {} ({:.1}%)", slots_failed,
        pct(slots_failed, total_slots));
    eprintln!();

    eprintln!("Unique encodings:    {}", encoding_counts.len());
    eprintln!("Unique mnemonics:    {}", mnemonic_counts.len());
    eprintln!();

    eprintln!("Time: {:.1}s ({:.0} roots/sec, excludes {:.1}s decoder load)",
        elapsed.as_secs_f64(),
        total as f64 / (elapsed - load_time).as_secs_f64(),
        load_time.as_secs_f64());

    // Top mnemonics
    eprintln!("\n--- Top 30 mnemonics ---");
    let mut mnemonics: Vec<_> = mnemonic_counts.iter().collect();
    mnemonics.sort_by(|a, b| b.1.cmp(a.1));
    for (mnemonic, count) in mnemonics.iter().take(30) {
        eprintln!("  {:6} {}", count, mnemonic);
    }
    if mnemonics.len() > 30 {
        eprintln!("  ... and {} more", mnemonics.len() - 30);
    }

    // Failed slot examples
    if !slot_fail_examples.is_empty() {
        eprintln!("\n--- Slot decode failures (first 50) ---");
        for (root, slot, _hex, bits) in slot_fail_examples.iter().take(50) {
            eprintln!("  root {:5} slot {} bits=0x{:010x}", root, slot, bits);
        }
        if slot_fail_examples.len() > 50 {
            eprintln!("  ... and {} more", slot_fail_examples.len() - 50);
        }
    }

    eprintln!("\n========================================");
}

fn pct(num: usize, denom: usize) -> f64 {
    if denom > 0 { 100.0 * num as f64 / denom as f64 } else { 0.0 }
}
