//! Tests of the fused `vst.pack.*` and standalone `vpack.*` instructions against NPU1 silicon
//! (`tools/gen_pack.py`).
//!
//! Four captures, replayed through the real instruction handlers (the fused store through
//! `MemoryUnit`, the standalone form through `VectorAlu`), every result word bit for bit:
//!
//! - `vec_pack_lane_probe`: designed blocks (one-hot lanes, ramps, values on both sides of
//!   every clamp bound at every lane position, every byte value at every position, a dense
//!   run around zero). The rules were read off these.
//! - `vec_pack_lane_rand`: seeded random blocks. Held out: never used to derive a rule.
//! - `vec_pack_addr`: fused-store addressing (index offsets, post-modify steps): where the
//!   data lands and where the pointer ends.
//! - `vec_pack_crtiming`: the sign comes from a run-time table inside a pipelined loop, with
//!   the `crPackSign` write in the same bundle as the `d` form it controls.
//!
//! All four forms, all four `crSat` values and both `crPackSign` values are covered in each
//! lane capture, in the fused store and the standalone instruction.

use super::memory::MemoryUnit;
use super::vector_dispatch::VectorAlu;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{ElementType, Operand, PostModify, SlotIndex, SlotOp};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::{ExecutionContext, MOD_BASE_DJ, MOD_BASE_M};
use crate::interpreter::traits::Decoder;
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::pack::{pack_form, PackAddressing, PackKind, PACK_FORMS};
use xdna_archspec::aie2::sign::SignSource;

/// `crPackSign` (AIE2GenRegisterInfo.td): the id the form table names.
const CR_PACK_SIGN: u8 = 5;

fn load(kernel: &str) -> serde_json::Value {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn words(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn uint(v: &serde_json::Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// Tile-local offset of the store target and the matching core-view address (0x70000 is the
/// local data-memory quadrant).
const TARGET_OFFSET: usize = 0x2000;
const TARGET_BASE: u32 = 0x70000 + TARGET_OFFSET as u32;

/// A context with the live registers the pack reads.
fn context(crsat: u32, sign: u32) -> ExecutionContext {
    let mut ctx = ExecutionContext::new();
    ctx.srs_config.saturation_mode = crsat as u8;
    ctx.set_control_reg_raw(CR_PACK_SIGN, sign);
    ctx.bundle_seq = 0;
    ctx.vector.advance_bundle(0);
    ctx
}

fn standalone_op(form: &str, src: u8, dest: u8) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Pack)
        .as_vector(ElementType::Int8)
        .with_dest(Operand::VectorReg(dest))
        .with_source(Operand::VectorReg(src));
    op.is_wide_vector = true;
    op.encoding_name = Some(format!("vpack.{form}"));
    op
}

fn fused_op(form: &str, src: u8, rest: Vec<Operand>) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Pack).as_vector(ElementType::Int8);
    op.sources = std::iter::once(Operand::VectorReg(src)).chain(rest).collect();
    op.is_wide_vector = true;
    op.encoding_name = Some(format!("vst.pack.{form}"));
    op
}

fn block_words(bytes_words: &[u32]) -> [u32; 16] {
    bytes_words.try_into().unwrap()
}

fn tile_words(tile: &Tile, offset: usize) -> Vec<u32> {
    (0..8).map(|i| tile.read_data_u32(offset + 4 * i).unwrap()).collect()
}

/// Run the standalone `vpack` on a source block; the 256-bit result.
fn run_standalone(ctx: &mut ExecutionContext, form: &str, block: &[u32; 16]) -> Vec<u32> {
    ctx.vector.write_wide(0, *block);
    VectorAlu::execute(&standalone_op(form, 0, 4), ctx);
    ctx.vector.read(4).to_vec()
}

/// Run a fused `vst.pack` at the target address with index offset 0; the stored 256 bits.
fn run_fused(ctx: &mut ExecutionContext, form: &str, block: &[u32; 16]) -> Vec<u32> {
    let mut tile = Tile::compute(0, 2);
    ctx.vector.write_wide(0, *block);
    ctx.pointer.write(0, TARGET_BASE);
    ctx.modifier.write(MOD_BASE_DJ, 0);
    let op = fused_op(form, 0, vec![Operand::PointerReg(0), Operand::ModifierReg(MOD_BASE_DJ)]);
    assert!(MemoryUnit::execute(&op, ctx, &mut tile, None, None));
    tile_words(&tile, TARGET_OFFSET)
}

/// Replay one lane capture; returns (vectors compared, lanes compared).
fn replay_lane_capture(kernel: &str) -> (usize, usize) {
    let g = load(kernel);
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let layout = &g["layout"];
    let header = uint(&layout["header_words"]);
    let per_chunk = uint(&layout["blocks_per_chunk"]);
    let wpb = uint(&layout["words_per_block"]);
    let wpp = uint(&layout["words_per_pass"]);
    let wpv = uint(&layout["words_per_vector"]);
    let blocks = uint(&layout["blocks"]);
    let nin = header + per_chunk * 16;
    let vectors = g["vectors"].as_array().unwrap();
    let passes = vectors.len() / 8;
    assert_eq!(passes, 8);
    let (mut compared_vectors, mut compared_lanes) = (0, 0);

    for b in 0..blocks {
        let chunk = b / per_chunk;
        let at = chunk * nin + header + (b % per_chunk) * 16;
        let block = block_words(&inputs[at..at + 16]);
        for (vi, v) in vectors.iter().enumerate() {
            let (pass, slot) = (vi / 8, vi % 8);
            assert_eq!(uint(&v["slot"]), slot);
            let group = v["group"].as_str().unwrap();
            let form = v["form"].as_str().unwrap();
            let mut ctx = context(uint(&v["crSat"]) as u32, uint(&v["crPackSign"]) as u32);
            let got = match group {
                "standalone" => run_standalone(&mut ctx, form, &block),
                "fused" => run_fused(&mut ctx, form, &block),
                other => panic!("group {other}"),
            };
            let base = b * wpb + pass * wpp + slot * wpv;
            assert_eq!(
                &got[..],
                &silicon[base..base + wpv],
                "{kernel} block {b} crSat {} crPackSign {} {group} {form}: result differs from silicon",
                v["crSat"],
                v["crPackSign"]
            );
            compared_vectors += 1;
            compared_lanes += 256 / uint(&v["bits_out"]);
        }
    }
    (compared_vectors, compared_lanes)
}

#[test]
fn probe_matches_silicon_on_every_form_lane() {
    let (vectors, lanes) = replay_lane_capture("vec_pack_lane_probe");
    eprintln!("pack probe: {vectors} vectors, {lanes} lanes matched silicon");
    assert_eq!(vectors, 504 * 64);
}

#[test]
fn held_out_random_matches_silicon_on_every_form_lane() {
    let (vectors, lanes) = replay_lane_capture("vec_pack_lane_rand");
    eprintln!("pack held-out: {vectors} vectors, {lanes} lanes matched silicon");
    assert_eq!(vectors, 256 * 64);
}

/// The rule the capture implies, per form: the `s` forms are signed whatever the register
/// holds; the `d` forms follow it. The archspec table must say the same.
#[test]
fn form_table_agrees_with_the_silicon_sign_rule() {
    for f in PACK_FORMS {
        for reg in [0u32, 1] {
            let signed = f.sign.is_signed(|id| {
                assert_eq!(id, CR_PACK_SIGN);
                reg
            });
            let expected = f.mnemonic.contains(".s") || reg == 1;
            assert_eq!(signed, expected, "{} with crPackSign={reg}", f.mnemonic);
        }
        if f.mnemonic.contains(".s") {
            assert_eq!(f.sign, SignSource::Fixed(true));
        }
    }
}

/// Fused-store addressing. Chunks hold blocks; per block and (crSat, crPackSign) pass the
/// index group stores four forms through `[p, dj]` with run-time offsets and the post-modify
/// group through `[p], m` with a negative run-time step, then the pointer after the chain is
/// recorded. The replay does the same stores and compares the 4-slot memory images (silicon
/// shape: slot `i` of the index group is form `i`; the post-modify group runs backwards).
#[test]
fn addressing_matches_silicon_for_every_form_and_pass() {
    let g = load("vec_pack_addr");
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let layout = &g["layout"];
    let header = uint(&layout["header_words"]);
    let per_chunk = uint(&layout["blocks_per_chunk"]);
    let wpb = uint(&layout["words_per_block"]);
    let wpp = uint(&layout["words_per_pass"]);
    let blocks = uint(&layout["blocks"]);
    let nin = header + per_chunk * 16;
    let step = layout["step_bytes"].as_i64().unwrap() as i32;
    let idx_offsets: Vec<i32> = layout["idx_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let forms: Vec<&str> = ["s8.s16", "d8.d16", "s4.s8", "d4.d8"].to_vec();
    let vectors = g["vectors"].as_array().unwrap();

    let mut stores = 0usize;
    for b in 0..blocks {
        let at = (b / per_chunk) * nin + header + (b % per_chunk) * 16;
        let block = block_words(&inputs[at..at + 16]);
        for pass in 0..8 {
            let first = &vectors[pass * 8];
            let (crsat, sign) = (uint(&first["crSat"]) as u32, uint(&first["crPackSign"]) as u32);
            let pass_base = b * wpb + pass * wpp;

            // Index group: the pointer is the group's third slot; each form's offset moves it.
            let mut ctx = context(crsat, sign);
            let mut tile = Tile::compute(0, 2);
            ctx.vector.write_wide(0, block);
            for (i, form) in forms.iter().enumerate() {
                ctx.pointer.write(0, TARGET_BASE + 64);
                ctx.modifier.write(MOD_BASE_DJ, idx_offsets[i] as u32);
                let op = fused_op(form, 0, vec![Operand::PointerReg(0), Operand::ModifierReg(MOD_BASE_DJ)]);
                assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
                assert_eq!(ctx.pointer.read(0), TARGET_BASE + 64, "index form must leave the pointer alone");
                stores += 1;
            }
            for slot in 0..4 {
                assert_eq!(
                    tile_words(&tile, TARGET_OFFSET + 32 * slot),
                    &silicon[pass_base + 8 * slot..pass_base + 8 * slot + 8],
                    "block {b} pass {pass} index slot {slot}"
                );
            }

            // Post-modify group: start at the last slot, step backwards.
            let mut ctx = context(crsat, sign);
            let mut tile = Tile::compute(0, 2);
            ctx.vector.write_wide(0, block);
            ctx.pointer.write(0, TARGET_BASE + 96);
            ctx.modifier.write(MOD_BASE_M, step as u32);
            for form in &forms {
                let mut op = fused_op(form, 0, vec![Operand::PointerReg(0)]);
                op.post_modify = PostModify::Register(MOD_BASE_M);
                assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
                stores += 1;
            }
            for slot in 0..4 {
                assert_eq!(
                    tile_words(&tile, TARGET_OFFSET + 32 * slot),
                    &silicon[pass_base + 32 + 8 * slot..pass_base + 32 + 8 * slot + 8],
                    "block {b} pass {pass} post-modify slot {slot}"
                );
            }
            // The kernel records the pointer after the chain relative to the group start.
            let recorded = silicon[pass_base + 64] as i32;
            assert_eq!(
                ctx.pointer.read(0).wrapping_sub(TARGET_BASE) as i32,
                recorded,
                "block {b} pass {pass}: pointer after the post-modify chain"
            );
        }
    }
    eprintln!("pack addressing: {stores} stores matched silicon");
    assert_eq!(stores, blocks * 8 * 8);
}

/// Each block's sign comes from a table; the fused and standalone `d` forms must follow it,
/// including where Chess writes `crPackSign` in the very bundle of the `d` form.
#[test]
fn crtiming_blocks_match_silicon() {
    let g = load("vec_pack_crtiming");
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let layout = &g["layout"];
    let blocks = uint(&layout["blocks"]);
    let chunks = uint(&layout["chunks"]);
    let wpb = uint(&layout["words_per_block"]);
    let control = uint(&layout["control_words"]);
    let nin = blocks + control + blocks * 16;
    let mut matched = 0;
    for c in 0..chunks {
        let crsat = inputs[c * nin + blocks + 1];
        assert_eq!(crsat, 1);
        for b in 0..blocks {
            let sign = inputs[c * nin + b];
            let at = c * nin + blocks + control + b * 16;
            let block = block_words(&inputs[at..at + 16]);
            let out = &silicon[(c * blocks + b) * wpb..(c * blocks + b + 1) * wpb];
            for (vi, (form, fused)) in
                [("d8.d16", true), ("d4.d8", true), ("d8.d16", false), ("d4.d8", false)]
                    .into_iter()
                    .enumerate()
            {
                let mut ctx = context(crsat, sign);
                let got = if fused {
                    run_fused(&mut ctx, form, &block)
                } else {
                    run_standalone(&mut ctx, form, &block)
                };
                assert_eq!(&got[..], &out[vi * 8..(vi + 1) * 8], "chunk {c} block {b} vector {vi}");
                matched += 1;
            }
        }
    }
    eprintln!("pack crtiming: {matched} vectors matched silicon");
    assert_eq!(matched, chunks * blocks * 4);
}

/// The sign register is read when the pack executes, from its live value: the handler keeps
/// no copy. (Chess's own schedule in `vec_pack_crtiming` writes `crPackSign` in the very
/// bundle of the `d` form, `vst.pack.d4.d8 ...; movx crPackSign, r0`, and silicon used the
/// value from before that bundle every time. Same-bundle visibility is the executor's
/// bundle-snapshot rule, checked end to end by the EMU bridge run of that capture.)
#[test]
fn pack_reads_the_sign_register_at_execution() {
    let src = block_words(&[200; 16]);
    let mut ctx = context(1, 0);
    // Unsigned: 200 stays 200. Signed: 200 clamps to 127.
    assert_eq!(run_standalone(&mut ctx, "d8.d16", &src)[0] & 0xFF, 200);
    ctx.set_control_reg_raw(CR_PACK_SIGN, 1);
    assert_eq!(run_standalone(&mut ctx, "d8.d16", &src)[0] & 0xFF, 127);
}

/// The Chess bundles recorded in the captures decode to the pack handlers' shape: a Pack
/// semantic, the form's mnemonic, a 512-bit source and (standalone) a register destination or
/// (fused) a memory operand and a vector register among the sources (the decoder puts the pointer
/// and modifier first). The standalone `vpack` is always wide, so the narrow legacy
/// branch of the old handler could never run.
#[test]
fn recorded_chess_bundles_decode_to_wide_pack_ops() {
    let decoder = InstructionDecoder::load_default();
    let mut seen = std::collections::BTreeSet::new();
    for kernel in ["vec_pack_lane_probe", "vec_pack_addr", "vec_pack_crtiming"] {
        let g = load(kernel);
        for e in g["encodings"].as_array().unwrap() {
            let mnemonic = e["mnemonic"].as_str().unwrap();
            let bytes: Vec<u8> = e["bytes"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(|h| u8::from_str_radix(h, 16).unwrap())
                .collect();
            let bundle = decoder.decode(&bytes, 0).unwrap_or_else(|err| panic!("{mnemonic}: {err:?}"));
            let op = bundle
                .slots()
                .iter()
                .flatten()
                .find(|s| s.encoding_name.as_deref() == Some(mnemonic))
                .unwrap_or_else(|| panic!("{mnemonic}: no slot decodes to it in {bytes:02x?}"));
            let form = pack_form(mnemonic).unwrap();
            assert_eq!(op.semantic, Some(SemanticOp::Pack), "{mnemonic}");
            assert!(op.is_wide_vector, "{mnemonic}: the source is a 512-bit x register");
            match form.kind {
                PackKind::Standalone => {
                    assert_eq!(form.addressing, PackAddressing::Register);
                    assert!(matches!(op.dest, Some(Operand::VectorReg(_))), "{mnemonic}: {:?}", op.dest);
                    assert!(matches!(op.sources.first(), Some(Operand::VectorReg(_))), "{mnemonic}");
                }
                PackKind::FusedStore => {
                    assert!(op.addresses_memory(), "{mnemonic}: a fused store addresses memory");
                    assert!(
                        op.sources.iter().any(|s| matches!(s, Operand::VectorReg(_))),
                        "{mnemonic}: {:?}",
                        op.sources
                    );
                }
            }
            seen.insert((mnemonic.to_string(), e["addressing"].as_str().unwrap().to_string()));
        }
    }
    // Every standalone form, and every fused form at least once.
    for f in PACK_FORMS
        .iter()
        .filter(|f| f.addressing != PackAddressing::TwoDim && f.addressing != PackAddressing::ThreeDim)
    {
        assert!(seen.iter().any(|(m, _)| m == f.mnemonic), "no recorded bundle for {}", f.mnemonic);
    }
}
