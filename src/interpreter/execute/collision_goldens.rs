//! Silicon replay of the register-port collision probes.
//!
//! Each golden in `collision_goldens_data.rs` is one pair point of `tools/gen_collision_probe.py`
//! run on NPU1 silicon (twice, byte-identical): two instructions, issued `pb - pa` bundles apart
//! (or together), whose resource reservations collide. They are replayed through the
//! cycle-accurate executor from the register state the probe kernel starts the pair from, and the
//! two destination registers must hold exactly the words silicon left in them. One golden per
//! (port group, configuration, outcome signature) the probes showed, so every rule of the
//! collision model (`state::write_collision`, the accumulator read-port steal, the SRS wired OR
//! and its s8 and lane-mode cases) is exercised by a real instruction pair.
//!
//! Regenerate the table with `tools/gen_collision_goldens.py`.

use super::collision_goldens_data::{Golden, Out, Variant, GOLDENS, VARIANTS};
use super::cycle_accurate::CycleAccurateExecutor;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{SlotOp, VliwBundle};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{Decoder, Executor};

/// Tile-local address the probe's input table is placed at; pointer registers address it through
/// the 20-bit space (`0x70000` + local).
const TABLE_AT: u32 = 0x400;
const LOCAL_BASE: u32 = 0x70000;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn op_of(decoder: &InstructionDecoder, out: &Out) -> SlotOp {
    let bundle = decoder
        .decode(&unhex(out.bytes), 0)
        .unwrap_or_else(|e| panic!("{}: {e:?}", out.asm));
    bundle
        .slots()
        .iter()
        .flatten()
        .find(|s| !s.is_nop())
        .unwrap_or_else(|| panic!("{}: no op", out.asm))
        .clone()
}

fn start_state(v: &Variant) -> (ExecutionContext, Tile) {
    let mut ctx = ExecutionContext::new();
    let mut tile = Tile::compute(0, 2);
    let table = unhex(v.table);
    tile.data_memory_mut()[TABLE_AT as usize..TABLE_AT as usize + table.len()].copy_from_slice(&table);
    let at = LOCAL_BASE + TABLE_AT;
    for (p, off) in [(0, 0), (3, 512), (2, 1024), (4, v.newx), (5, v.wnew)] {
        ctx.pointer.write(p, at + off);
    }
    for (k, lanes) in v.acc.iter().enumerate() {
        ctx.accumulator.write(2 * k as u8, lanes[..8].try_into().unwrap());
        ctx.accumulator.write(2 * k as u8 + 1, lanes[8..].try_into().unwrap());
    }
    for (reg, words) in v.vec {
        ctx.vector.write(*reg, *words);
    }
    for (reg, words) in v.q {
        ctx.mask.write(*reg, *words);
    }
    for (reg, value) in v.scalars {
        ctx.scalar.write(*reg, *value);
    }
    (ctx, tile)
}

fn read_out(ctx: &ExecutionContext, out: &Out) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..out.pieces {
        if out.acc {
            let q = out.quarter + i;
            let lanes = ctx.accumulator.read_committed(out.reg + q / 2);
            for lane in &lanes[4 * (q % 2) as usize..][..4] {
                bytes.extend(lane.to_le_bytes());
            }
        } else {
            for w in ctx.vector.read_store(out.reg + i) {
                bytes.extend(w.to_le_bytes());
            }
        }
    }
    bytes
}

/// Replay one golden; the two destination registers' bytes as `(a, b)`.
fn replay(g: &Golden) -> (Vec<u8>, Vec<u8>) {
    let decoder = InstructionDecoder::load_default();
    let (mut ctx, mut tile) = start_state(&VARIANTS[g.variant]);
    let (a, b) = (op_of(&decoder, &g.a), op_of(&decoder, &g.b));
    let mut executor = CycleAccurateExecutor::new();
    let last = g.pa.max(g.pb) as usize;
    for at in 0..=last {
        let mut bundle = VliwBundle::empty();
        if at == g.pa as usize {
            bundle.set_slot(a.clone());
        }
        if at == g.pb as usize {
            bundle.set_slot(b.clone());
        }
        executor.execute(&bundle, &mut ctx, &mut tile);
        if std::env::var("GOLDEN_TRACE").is_ok() {
            eprintln!(
                "bundle {at}: v16 {:08x} v8 {:08x}",
                ctx.vector.read_store(16)[0],
                ctx.vector.read_store(8)[0]
            );
        }
    }
    // let everything land: the longest result latency is a fused load's 9 bundles
    for _ in 0..32 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    (read_out(&ctx, &g.a), read_out(&ctx, &g.b))
}

fn failures(filter: impl Fn(&Golden) -> bool) -> Vec<String> {
    GOLDENS
        .iter()
        .filter(|g| filter(g))
        .filter_map(|g| {
            let (a, b) = replay(g);
            let (want_a, want_b) = (unhex(g.a.silicon), unhex(g.b.silicon));
            (a != want_a || b != want_b).then(|| {
                format!(
                    "{} #{} [{} {}] A `{}` {} B `{}` {}",
                    g.probe,
                    g.id,
                    g.units,
                    g.cfg,
                    g.a.asm,
                    if a == want_a { "ok" } else { "DIFFERS" },
                    g.b.asm,
                    if b == want_b { "ok" } else { "DIFFERS" },
                )
            })
        })
        .collect()
}

/// Every golden: all rules at once.
#[test]
fn every_silicon_collision_point_replays_exactly() {
    let bad = failures(|_| true);
    assert!(
        bad.is_empty(),
        "{} of {} goldens differ from silicon:\n{}",
        bad.len(),
        GOLDENS.len(),
        bad.join("\n")
    );
}

#[test]
#[ignore]
fn show_golden() {
    let id: u32 = std::env::var("GOLDEN_ID").unwrap().parse().unwrap();
    let probe = std::env::var("GOLDEN_PROBE").unwrap();
    let g = GOLDENS.iter().find(|g| g.id == id && g.probe == probe).unwrap();
    let (a, b) = replay(g);
    for (n, got, want) in [("A", a, unhex(g.a.silicon)), ("B", b, unhex(g.b.silicon))] {
        println!(
            "{n} {} {}\n  emu {}\n  hw  {}",
            if got == want { "ok" } else { "DIFF" },
            g.a.asm,
            got.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            want.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
    }
}
