//! Tests of the fused `vldb.unpack.*` and standalone `vunpack.*` instructions against
//! NPU1 silicon (`tools/gen_vldb_unpack.py`).
//!
//! Four captures, replayed through the real instruction handlers (the fused load through
//! `MemoryUnit`, the standalone form through `VectorAlu`), every result word bit for bit:
//!
//! - `vec_unpack_lane_probe`: designed blocks (one-hot nibbles and bytes, ramps, every byte
//!   value at every position). The lane map was read off these.
//! - `vec_unpack_lane_rand`: seeded random blocks. Held out: never used to derive the map.
//! - `vec_unpack_addr`: fused-load addressing (index, post-modify, 2D, 3D): the data
//!   loaded, the pointer and the descriptor counts after each access.
//! - `vec_unpack_crtiming`: the sign comes from a run-time table inside a pipelined loop.
//!
//! All four forms and both `crUnpackSign` values are covered in each family.

use super::memory::MemoryUnit;
use super::vector_dispatch::VectorAlu;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{ElementType, Operand, PostModify, SlotIndex, SlotOp};
use crate::interpreter::state::{
    ExecutionContext, MOD_BASE_D2D, MOD_BASE_D3D, MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M,
};
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::sign::SignSource;

/// `crUnpackSign` (AIE2GenRegisterInfo.td): the id the form table names.
const CR_UNPACK_SIGN: u8 = 11;

fn load(kernel: &str) -> serde_json::Value {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn words(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// The four forms in capture order: (suffix, bits_in, bits_out).
const FORMS: [(&str, u32, u32); 4] = [("s16.s8", 8, 16), ("d16.d8", 8, 16), ("s8.s4", 4, 8), ("d8.d4", 4, 8)];

/// Issue an op at bundle 0 and step the bundle clock past the unpack's write-back
/// (sample at issue+6, land at issue+7), the way the cycle-accurate executor does.
fn settle(ctx: &mut ExecutionContext) {
    for b in 1..=8u64 {
        ctx.bundle_seq = b;
        ctx.vector.advance_bundle(b);
        ctx.process_pending_unpacks();
    }
}

fn start(ctx: &mut ExecutionContext) {
    ctx.bundle_seq = 0;
    ctx.vector.advance_bundle(0);
}

fn standalone_op(suffix: &str, src: u8, dest: u8) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Unpack)
        .as_vector(ElementType::Int8)
        .with_dest(Operand::VectorReg(dest))
        .with_source(Operand::VectorReg(src));
    op.is_wide_vector = true;
    op.encoding_name = Some(format!("vunpack.{suffix}"));
    op
}

fn fused_op(mnemonic: &str, dest: u8, sources: Vec<Operand>) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Unpack)
        .as_vector(ElementType::Int8)
        .with_dest(Operand::VectorReg(dest));
    op.sources = sources.into_iter().collect();
    op.is_wide_vector = true;
    op.encoding_name = Some(mnemonic.to_string());
    op
}

/// Tile-local offset of the test data and the matching core-view address (0x70000 is the
/// local data-memory quadrant).
const BLOCK_OFFSET: usize = 0x2000;
const BLOCK_BASE: u32 = 0x70000 + BLOCK_OFFSET as u32;

/// Result of the lane replay of one capture: vectors and lanes compared.
fn replay_lane_capture(kernel: &str) -> (usize, usize) {
    let g = load(kernel);
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let layout = &g["layout"];
    let header = layout["header_words"].as_u64().unwrap() as usize;
    let per_chunk = layout["blocks_per_chunk"].as_u64().unwrap() as usize;
    let wpb = layout["words_per_block"].as_u64().unwrap() as usize;
    let wpv = layout["words_per_vector"].as_u64().unwrap() as usize;
    let blocks = layout["blocks"].as_u64().unwrap() as usize;
    let nin = header + per_chunk * 8;
    let vectors = g["vectors"].as_array().unwrap();
    let (mut compared_vectors, mut compared_lanes) = (0, 0);

    for b in 0..blocks {
        let chunk = b / per_chunk;
        let at = chunk * nin + header + (b % per_chunk) * 8;
        let block: [u32; 8] = inputs[at..at + 8].try_into().unwrap();
        for (vi, v) in vectors.iter().enumerate() {
            let sign = v["crUnpackSign"].as_u64().unwrap() as u32;
            let group = v["group"].as_str().unwrap();
            let form = v["form"].as_str().unwrap();
            let (_, bits_in, bits_out) = *FORMS.iter().find(|f| f.0 == form).unwrap();
            let expected = &silicon[b * wpb + vi * wpv..b * wpb + (vi + 1) * wpv];

            let mut ctx = ExecutionContext::new();
            ctx.set_control_reg_raw(CR_UNPACK_SIGN, sign);
            let mut tile = Tile::compute(0, 2);
            start(&mut ctx);
            if group == "standalone" {
                ctx.vector.write(2, block);
                VectorAlu::execute(&standalone_op(form, 2, 4), &mut ctx);
            } else {
                for (i, w) in block.iter().enumerate() {
                    tile.write_data_u32(BLOCK_OFFSET + 4 * i, *w);
                }
                ctx.pointer.write(0, BLOCK_BASE);
                ctx.modifier.write(MOD_BASE_DJ, 0);
                ctx.modifier.write(MOD_BASE_M, 0);
                let mut op = fused_op(
                    &format!("vldb.unpack.{form}"),
                    4,
                    vec![Operand::PointerReg(0), Operand::ModifierReg(MOD_BASE_DJ)],
                );
                if group == "fused_postmod" {
                    op.sources = [Operand::PointerReg(0)].into_iter().collect();
                    op.post_modify = PostModify::Register(MOD_BASE_M);
                }
                assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
            }
            settle(&mut ctx);
            let got = ctx.vector.read_wide(4);
            assert_eq!(
                &got[..],
                expected,
                "{kernel} block {b} sign {sign} {group} {form}: result differs from silicon"
            );
            let _ = bits_in;
            compared_vectors += 1;
            compared_lanes += (512 / bits_out) as usize;
        }
    }
    (compared_vectors, compared_lanes)
}

#[test]
fn probe_matches_silicon_on_every_form_lane() {
    let (vectors, lanes) = replay_lane_capture("vec_unpack_lane_probe");
    eprintln!("unpack probe: {vectors} vectors, {lanes} lanes matched silicon");
    assert_eq!(vectors, 368 * 24);
}

#[test]
fn held_out_random_matches_silicon_on_every_form_lane() {
    let (vectors, lanes) = replay_lane_capture("vec_unpack_lane_rand");
    eprintln!("unpack held-out: {vectors} vectors, {lanes} lanes matched silicon");
    assert_eq!(vectors, 256 * 24);
}

/// The sign the capture implies per form: `s` forms are signed at either register value,
/// `d` forms follow the register. The archspec table must say the same.
#[test]
fn form_table_agrees_with_the_silicon_sign_rule() {
    use xdna_archspec::aie2::unpack::unpack_form;
    for (suffix, _, _) in FORMS {
        let mnemonic = format!("vunpack.{suffix}");
        let form = unpack_form(&mnemonic).unwrap();
        for reg in [0u32, 1] {
            let signed = form.sign.is_signed(|id| {
                assert_eq!(id, CR_UNPACK_SIGN);
                reg
            });
            let expected = suffix.starts_with('s') || reg == 1;
            assert_eq!(signed, expected, "{mnemonic} with crUnpackSign={reg}");
        }
        if suffix.starts_with('s') {
            assert_eq!(form.sign, SignSource::Fixed(true));
        }
    }
}

/// Addressing replay. Register assignment, as read from the kernel's disassembly: the
/// descriptor is `d0` (`m0` = off, `dn0` = size1, `dj0` = inc1, `dc0` = count1) and for 3D
/// also `d4` (`dn4` = size2, `dj4` = inc2, `dc4` = count2).
#[test]
fn addressing_matches_silicon_for_every_form_and_sign() {
    let g = load("vec_unpack_addr");
    let layout = &g["layout"];
    let silicon = words(&g["silicon"]);
    let region: Vec<u8> = layout["region"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect();
    let region_base = layout["region_base"].as_u64().unwrap() as u32;
    let rec = layout["record_words"].as_u64().unwrap() as usize;
    let list = |key: &str| -> Vec<i64> {
        layout[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
    };
    let sets = |key: &str| -> Vec<Vec<i64>> {
        layout[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect())
            .collect()
    };
    let (idx_offsets, postmods) = (list("idx_offsets"), list("postmod_values"));
    let (sets_2d, sets_3d) = (sets("sets_2d"), sets("sets_3d"));
    let (iter_2d, iter_3d) =
        (layout["iterations_2d"].as_u64().unwrap(), layout["iterations_3d"].as_u64().unwrap());
    let records = idx_offsets.len()
        + postmods.len()
        + sets_2d.len() * iter_2d as usize
        + sets_3d.len() * iter_3d as usize;

    let region_start = BLOCK_BASE;
    let base = region_start + region_base;
    let mut accesses = 0usize;

    for (ci, chunk) in layout["chunks"].as_array().unwrap().iter().enumerate() {
        let sign = chunk["crUnpackSign"].as_u64().unwrap() as u32;
        let form = chunk["form"].as_str().unwrap();
        let mut at = ci * records * rec;
        let next = |at: &mut usize| -> Vec<u32> {
            let r = silicon[*at..*at + rec].to_vec();
            *at += rec;
            r
        };
        let new_ctx = |sign: u32| -> (ExecutionContext, Tile) {
            let mut ctx = ExecutionContext::new();
            ctx.set_control_reg_raw(CR_UNPACK_SIGN, sign);
            let mut tile = Tile::compute(0, 2);
            for (i, b) in region.iter().enumerate() {
                tile.data_memory_mut()[BLOCK_OFFSET + i] = *b;
            }
            (ctx, tile)
        };
        let run = |ctx: &mut ExecutionContext, tile: &mut Tile, op: &SlotOp| -> Vec<u32> {
            start(ctx);
            assert!(MemoryUnit::execute(op, ctx, tile, None, None));
            settle(ctx);
            ctx.vector.read_wide(4).to_vec()
        };
        let check = |r: &[u32], got: &[u32], what: &str| {
            assert_eq!(&got[..], &r[..16], "{form} sign {sign} {what}: data differs from silicon");
        };
        let rel = |r: &[u32]| r[16].wrapping_sub(r[19]);

        for off in &idx_offsets {
            let r = next(&mut at);
            let (mut ctx, mut tile) = new_ctx(sign);
            ctx.pointer.write(0, base);
            ctx.modifier.write(MOD_BASE_DJ, *off as u32);
            let op = fused_op(
                &format!("vldb.unpack.{form}"),
                4,
                vec![Operand::PointerReg(0), Operand::ModifierReg(MOD_BASE_DJ)],
            );
            let got = run(&mut ctx, &mut tile, &op);
            check(&r, &got, &format!("index {off}"));
            assert_eq!(ctx.pointer.read(0), base, "index form must leave the pointer alone");
            assert_eq!(rel(&r), *off as u32);
            accesses += 1;
        }
        for step in &postmods {
            let r = next(&mut at);
            let (mut ctx, mut tile) = new_ctx(sign);
            ctx.pointer.write(0, base);
            ctx.modifier.write(MOD_BASE_M, *step as u32);
            let mut op = fused_op(&format!("vldb.unpack.{form}"), 4, vec![Operand::PointerReg(0)]);
            op.post_modify = PostModify::Register(MOD_BASE_M);
            let got = run(&mut ctx, &mut tile, &op);
            check(&r, &got, &format!("post-modify {step}"));
            assert_eq!(ctx.pointer.read(0).wrapping_sub(base), rel(&r), "post-modify {step}: pointer");
            accesses += 1;
        }
        for set in &sets_2d {
            let (off, size1, inc1, c0) = (set[0], set[1], set[2], set[3]);
            let (mut ctx, mut tile) = new_ctx(sign);
            ctx.pointer.write(0, base);
            ctx.modifier.write(MOD_BASE_M, off as u32);
            ctx.modifier.write(MOD_BASE_DN, size1 as u32);
            ctx.modifier.write(MOD_BASE_DJ, inc1 as u32);
            ctx.modifier.write(MOD_BASE_DC, c0 as u32);
            let mut op = fused_op(&format!("vldb.2d.unpack.{form}"), 4, vec![Operand::PointerReg(0)]);
            op.post_modify = PostModify::Descriptor(MOD_BASE_D2D);
            for it in 0..iter_2d {
                let r = next(&mut at);
                let got = run(&mut ctx, &mut tile, &op);
                check(&r, &got, &format!("2d {set:?} step {it}"));
                assert_eq!(ctx.pointer.read(0).wrapping_sub(base), rel(&r), "2d {set:?} step {it}: pointer");
                assert_eq!(ctx.modifier_read(MOD_BASE_DC), r[17], "2d {set:?} step {it}: count");
                accesses += 1;
            }
        }
        for set in &sets_3d {
            let (off, size1, inc1, size2, inc2, c1, c2) =
                (set[0], set[1], set[2], set[3], set[4], set[5], set[6]);
            let (mut ctx, mut tile) = new_ctx(sign);
            ctx.pointer.write(0, base);
            ctx.modifier.write(MOD_BASE_M, off as u32);
            ctx.modifier.write(MOD_BASE_DN, size1 as u32);
            ctx.modifier.write(MOD_BASE_DJ, inc1 as u32);
            ctx.modifier.write(MOD_BASE_DC, c1 as u32);
            ctx.modifier.write(MOD_BASE_DN + 4, size2 as u32);
            ctx.modifier.write(MOD_BASE_DJ + 4, inc2 as u32);
            ctx.modifier.write(MOD_BASE_DC + 4, c2 as u32);
            let mut op = fused_op(&format!("vldb.3d.unpack.{form}"), 4, vec![Operand::PointerReg(0)]);
            op.post_modify = PostModify::Descriptor(MOD_BASE_D3D);
            for it in 0..iter_3d {
                let r = next(&mut at);
                let got = run(&mut ctx, &mut tile, &op);
                check(&r, &got, &format!("3d {set:?} step {it}"));
                assert_eq!(ctx.pointer.read(0).wrapping_sub(base), rel(&r), "3d {set:?} step {it}: pointer");
                assert_eq!(ctx.modifier_read(MOD_BASE_DC), r[17], "3d {set:?} step {it}: count1");
                assert_eq!(ctx.modifier_read(MOD_BASE_DC + 4), r[18], "3d {set:?} step {it}: count2");
                accesses += 1;
            }
        }
    }
    eprintln!("unpack addressing: {accesses} accesses matched silicon");
    assert_eq!(accesses, 8 * records);
}

/// Each block's sign comes from a table; the fused and standalone `d` forms must follow it.
#[test]
fn crtiming_blocks_match_silicon() {
    let g = load("vec_unpack_crtiming");
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let blocks = g["layout"]["blocks"].as_u64().unwrap() as usize;
    let chunks = g["layout"]["chunks"].as_u64().unwrap() as usize;
    let wpb = g["layout"]["words_per_block"].as_u64().unwrap() as usize;
    let nin = blocks + blocks * 8;
    let mut matched = 0;
    for c in 0..chunks {
        for b in 0..blocks {
            let sign = inputs[c * nin + b];
            let block: [u32; 8] = inputs[c * nin + blocks + b * 8..c * nin + blocks + (b + 1) * 8]
                .try_into()
                .unwrap();
            let out = &silicon[(c * blocks + b) * wpb..(c * blocks + b + 1) * wpb];
            for (vi, (form, fused)) in
                [("d16.d8", true), ("d8.d4", true), ("d16.d8", false), ("d8.d4", false)]
                    .into_iter()
                    .enumerate()
            {
                let mut ctx = ExecutionContext::new();
                ctx.set_control_reg_raw(CR_UNPACK_SIGN, sign);
                let mut tile = Tile::compute(0, 2);
                start(&mut ctx);
                if fused {
                    for (i, w) in block.iter().enumerate() {
                        tile.write_data_u32(BLOCK_OFFSET + 4 * i, *w);
                    }
                    ctx.pointer.write(0, BLOCK_BASE);
                    let op = fused_op(&format!("vldb.unpack.{form}"), 4, vec![Operand::PointerReg(0)]);
                    MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
                } else {
                    ctx.vector.write(2, block);
                    VectorAlu::execute(&standalone_op(form, 2, 4), &mut ctx);
                }
                settle(&mut ctx);
                assert_eq!(
                    &ctx.vector.read_wide(4)[..],
                    &out[vi * 16..(vi + 1) * 16],
                    "chunk {c} block {b} vector {vi}"
                );
                matched += 1;
            }
        }
    }
    eprintln!("unpack crtiming: {matched} vectors matched silicon");
}

/// `crUnpackSign` is read at pipeline stage 7 (issue+6 here), not at issue: a write issued
/// up to five bundles after a register-driven unpack still controls it. Chess emits exactly
/// this schedule in `vec_unpack_crtiming`, whose pipelined loop writes the register after
/// the `vldb.unpack.d*` that it controls; silicon honored the written sign every time.
#[test]
fn sign_register_is_read_late_not_at_issue() {
    for (write_at, expect_signed) in [(1u64, true), (5, true)] {
        let mut ctx = ExecutionContext::new();
        ctx.set_control_reg_raw(CR_UNPACK_SIGN, 0);
        ctx.vector.write(2, [0x8080_8080; 8]);
        start(&mut ctx);
        VectorAlu::execute(&standalone_op("d16.d8", 2, 4), &mut ctx);
        for b in 1..=8u64 {
            ctx.bundle_seq = b;
            ctx.vector.advance_bundle(b);
            ctx.process_pending_unpacks();
            if b == write_at {
                ctx.set_control_reg_raw(CR_UNPACK_SIGN, 1);
            }
        }
        let lane0 = ctx.vector.read_wide(4)[0] & 0xFFFF;
        assert_eq!(lane0, if expect_signed { 0xFF80 } else { 0x0080 }, "sign written at bundle {write_at}");
    }
}
