//! Itinerary-driven resource collision check over compiled AIE2 code.
//!
//! ```text
//! collision-check resources                 # every unit, the classes that hold it, stage offsets
//! collision-check elf FILE...               # decode .text linearly and report unit collisions
//! collision-check ops FILE...               # print every decoded instruction with its reservations
//! ```
//! `elf` walks `.text` as straight-line code, one bundle per cycle, so a taken branch or a loop
//! back-edge is not modelled; each hit is therefore a candidate to read, not a verdict.

use std::process;

use xdna_archspec::aie2::isa::load_from_generated;
use xdna_archspec::aie2::isa::reservation::{
    find_collisions, reservations_of, stages_from_resolved, unit_users, Issued,
};
use xdna_emu::interpreter::decode::InstructionDecoder;
use xdna_emu::parser::elf::AieElf;

/// Decoded bundles as reservation lists, plus the byte offset decoding stopped at.
fn decode_text(decoder: &InstructionDecoder, units: &[String], text: &[u8]) -> (Vec<Vec<Issued>>, usize) {
    let mut bundles = Vec::new();
    let mut pc = 0usize;
    while pc < text.len() {
        let size = xdna_emu::interpreter::bundle::detect_format(&text[pc..]).size_bytes() as usize;
        if pc + size > text.len() {
            break;
        }
        let ops = decoder
            .resolved_ops(&text[pc..pc + size])
            .into_iter()
            .inspect(|op| {
                if std::env::var_os("COLLISION_CHECK_DEBUG").is_some() {
                    eprintln!("{} {:?}", op.llvm_name, op.stages);
                }
            })
            .map(|op| Issued {
                label: op.llvm_name,
                reservations: reservations_of(&stages_from_resolved(units, &op.stages)),
            })
            .collect();
        bundles.push(ops);
        pc += size.max(2);
    }
    (bundles, pc)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = load_from_generated().itineraries;
    match args.first().map(String::as_str) {
        Some("resources") => {
            for (unit, users) in unit_users(&model) {
                println!("{unit}");
                for (class, offs) in users {
                    println!("  {class}: {offs:?}");
                }
            }
        }
        Some(mode @ ("elf" | "ops")) if args.len() > 1 => {
            let decoder = InstructionDecoder::load_default();
            let units = load_from_generated().functional_units;
            for path in &args[1..] {
                let data = std::fs::read(path).unwrap_or_else(|e| {
                    eprintln!("{path}: {e}");
                    process::exit(2)
                });
                let Ok(elf) = AieElf::parse(&data) else {
                    println!("{path}: NOT-AN-ELF");
                    continue;
                };
                let Some(text) = elf.text_section() else {
                    println!("{path}: NO-TEXT");
                    continue;
                };
                let (bundles, end) = decode_text(&decoder, &units, text);
                if mode == "ops" {
                    println!("{path}: bundles {}", bundles.len());
                    for (b, ops) in bundles.iter().enumerate() {
                        for op in ops {
                            let held: Vec<String> = op
                                .reservations
                                .iter()
                                .map(|r| {
                                    format!("{}:{}{}", r.unit, r.offset, if r.reserved { "r" } else { "" })
                                })
                                .collect();
                            println!("  OP {b} {} {}", op.label, held.join(","));
                        }
                    }
                    continue;
                }
                let collisions = find_collisions(&bundles);
                println!(
                    "{path}: bundles {} ops {} stopped_at {}/{} collisions {}",
                    bundles.len(),
                    bundles.iter().map(Vec::len).sum::<usize>(),
                    end,
                    text.len(),
                    collisions.len()
                );
                for c in &collisions {
                    println!(
                        "  COLLISION {} cycle {} : {}@{}+{} vs {}@{}+{}",
                        c.unit,
                        c.cycle,
                        c.first.label,
                        c.first.bundle,
                        c.first.offset,
                        c.second.label,
                        c.second.bundle,
                        c.second.offset
                    );
                }
            }
        }
        _ => {
            eprintln!("usage: collision-check resources | elf FILE...");
            process::exit(2);
        }
    }
}
