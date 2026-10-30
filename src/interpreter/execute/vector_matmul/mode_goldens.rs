//! Lane-level tests of the integer MAC config modes with no matmul tile
//! (elementwise, convolution, two-channel) against NPU1 silicon.
//!
//! Each mode has two captures (`tools/gen_mac_modes.py`):
//!
//! - `vec_macmode_<stem>_probe`: one-hot A against B ones and a B ramp, and the
//!   mirror image. The lane map (`xdna_archspec::aie2::matmul::MacShape`) was
//!   read off these.
//! - `vec_macmode_<stem>_rand`: seeded random data and extreme tiles through a
//!   mul -> mac -> msc chain. Held out: never used to derive the map.
//!
//! Both are replayed through `execute_matmul` (config word, instruction
//! semantic, accumulator plumbing), in all four signedness flavours, and every
//! accumulator lane must equal silicon bit for bit.

use super::*;
use crate::interpreter::bundle::SlotIndex;

/// The eight modes captured: (amode, bmode, variant), header family stem.
const MODES: [((u32, u32, u32), &str); 8] = [
    ((0, 1, 1), "elem_32_2"),
    ((0, 1, 2), "conv_4x4_8ch"),
    ((0, 1, 3), "conv_8x8_4ch"),
    ((0, 1, 4), "conv_32x8"),
    ((0, 2, 1), "4x4_4x4_2ch"),
    ((0, 3, 1), "elem_32"),
    ((1, 3, 2), "elem_16_2"),
    ((1, 3, 3), "conv_16x4"),
];

/// Signedness flavours in capture order, as (sgn_x, sgn_y).
const SIGNS: [(u32, u32); 4] = [(1, 1), (1, 0), (0, 1), (0, 0)];

struct Capture {
    a_bits: u32,
    b_bits: u32,
    a_lanes: usize,
    b_lanes: usize,
    acc_bits: u32,
    lanes: usize,
    tiles: usize,
    stages: usize,
    out_words: usize,
    inputs: Vec<Vec<u64>>,
    silicon: Vec<u64>,
}

fn load(kernel: &str) -> Capture {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let num = |x: &serde_json::Value| x.as_u64().unwrap();
    let list = |x: &serde_json::Value| -> Vec<u64> { x.as_array().unwrap().iter().map(num).collect() };
    Capture {
        a_bits: num(&v["mode"]["a_bits"]) as u32,
        b_bits: num(&v["mode"]["b_bits"]) as u32,
        a_lanes: num(&v["mode"]["a_lanes"]) as usize,
        b_lanes: num(&v["mode"]["b_lanes"]) as usize,
        acc_bits: num(&v["mode"]["acc_bits"]) as u32,
        lanes: num(&v["mode"]["acc_lanes"]) as usize,
        tiles: num(&v["tile"]["tiles"]) as usize,
        stages: num(&v["tile"]["stages"]) as usize,
        out_words: num(&v["tile"]["out_words"]) as usize,
        inputs: v["inputs"].as_array().unwrap().iter().map(list).collect(),
        silicon: list(&v["silicon"]),
    }
}

/// Elements of `bits` width, little-endian, into the 512-bit vector register image.
fn pack(values: &[u64], bits: u32) -> Vec512 {
    let mut v = [0u32; 16];
    let per_word = (32 / bits) as usize;
    for (i, &x) in values.iter().enumerate() {
        v[i / per_word] |= (x as u32) << (bits as usize * (i % per_word));
    }
    v
}

/// The config word Chess emits for a mode: key and signedness, no other bits.
fn config_word(key: (u32, u32, u32), (sgn_x, sgn_y): (u32, u32)) -> u32 {
    (key.0 << 1) | (key.1 << 3) | (key.2 << 5) | (sgn_y << 8) | (sgn_x << 9)
}

fn mac_op(semantic: SemanticOp, name: &str) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, semantic);
    op.is_vector = true;
    op.is_wide_vector = true;
    op.sources.push(Operand::VectorReg(0));
    op.sources.push(Operand::VectorReg(2));
    op.sources.push(Operand::ScalarReg(5));
    op.dest = Some(Operand::AccumReg(0));
    op.encoding_name = Some(name.to_string());
    op
}

/// The three instruction forms of a chain, in stage order: vmul, vmac (reads
/// the accumulator), vmsc (reads and subtracts).
fn stage_op(stage: usize) -> SlotOp {
    match stage {
        0 => mac_op(SemanticOp::MatMul, "VMUL_vmac_cm_core_dense"),
        1 => {
            let mut op = mac_op(SemanticOp::Mac, "VMAC_vmac_cm_core_dense");
            op.sources.push(Operand::AccumReg(0));
            op
        }
        2 => {
            let mut op = mac_op(SemanticOp::MatMulSub, "VMSC_vmac_cm_core_dense");
            op.sources.push(Operand::AccumReg(0));
            op
        }
        _ => unreachable!(),
    }
}

/// Lane values (as `acc_bits`-wide bit patterns) of cm0.
fn read_lanes(ctx: &ExecutionContext, lanes: usize, acc_bits: u32) -> Vec<u64> {
    let acc = ctx.accumulator.read_wide(0);
    (0..lanes)
        .map(|l| {
            if acc_bits == 32 {
                (acc[l / 2] >> (32 * (l % 2))) & 0xFFFF_FFFF
            } else {
                acc[l]
            }
        })
        .collect()
}

fn silicon_lanes(c: &Capture, tile: usize, sign: usize, stage: usize) -> Vec<u64> {
    let base = ((tile * 4 + sign) * c.stages + stage) * c.out_words;
    let words = &c.silicon[base..base + c.out_words];
    if c.acc_bits == 32 {
        words.to_vec()
    } else {
        (0..c.lanes).map(|l| words[l] | (words[c.lanes + l] << 32)).collect()
    }
}

/// Replay every tile of a capture through `execute_matmul`.
/// Returns (lanes checked, first mismatches).
fn replay(kernel: &str, key: (u32, u32, u32)) -> (usize, Vec<String>) {
    let c = load(kernel);
    let (mut checked, mut bad) = (0, Vec::new());
    for t in 0..c.tiles {
        for (si, &signs) in SIGNS.iter().enumerate() {
            let mut ctx = ExecutionContext::new();
            // A poisoned accumulator: a lane the executor leaves unwritten shows up as a mismatch.
            ctx.accumulator.write_wide(0, [0xDEAD_BEEF_DEAD_BEEFu64; 16]);
            ctx.scalar.write(5, config_word(key, signs));
            for stage in 0..c.stages {
                let a = &c.inputs[2 * stage][t * c.a_lanes..(t + 1) * c.a_lanes];
                let b = &c.inputs[2 * stage + 1][t * c.b_lanes..(t + 1) * c.b_lanes];
                ctx.vector.write_wide(0, pack(a, c.a_bits));
                ctx.vector.write_wide(2, pack(b, c.b_bits));
                assert!(execute_matmul(&stage_op(stage), &mut ctx), "{kernel}: op must be handled");
                ctx.force_commit_all_pending();
                let got = read_lanes(&ctx, c.lanes, c.acc_bits);
                let want = silicon_lanes(&c, t, si, stage);
                for lane in 0..c.lanes {
                    checked += 1;
                    if got[lane] != want[lane] {
                        if bad.len() < 5 {
                            bad.push(format!(
                                "tile {t} sign {si} stage {stage} lane {lane}: {:#x} != silicon {:#x}",
                                got[lane], want[lane]
                            ));
                        } else {
                            bad.push(String::new());
                        }
                    }
                }
            }
        }
    }
    (checked, bad)
}

fn assert_clean(kernel: &str, checked: usize, bad: &[String]) {
    assert!(
        bad.is_empty(),
        "{kernel}: {} of {checked} lanes differ from silicon, first: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

/// Every probe tile, every lane, all four signedness flavours.
#[test]
fn matches_silicon_on_every_probe_lane() {
    let mut total = 0;
    for (key, stem) in MODES {
        let kernel = format!("vec_macmode_{stem}_probe");
        let (checked, bad) = replay(&kernel, key);
        assert_clean(&kernel, checked, &bad);
        total += checked;
    }
    // 8-bit probes: 256 tiles x 4 flavours x 32 lanes; 16-bit modes 128 x 4 x 16 or 32.
    assert!(total >= 8 * 128 * 4 * 16, "only {total} probe lanes replayed");
}

/// Held-out random and extreme data through mul -> mac -> msc, all four
/// signedness flavours: the accumulation path is the one the matmul modes use.
#[test]
fn matches_silicon_on_every_held_out_lane() {
    let mut total = 0;
    for (key, stem) in MODES {
        let kernel = format!("vec_macmode_{stem}_rand");
        let (checked, bad) = replay(&kernel, key);
        assert_clean(&kernel, checked, &bad);
        total += checked;
    }
    assert!(total >= 8 * 112 * 4 * 3 * 16, "only {total} held-out lanes replayed");
}

/// A config word whose (amode, bmode, variant) names no geometry is reported
/// unhandled, with the destination left exactly as it was, instead of being
/// claimed handled with nothing written.
#[test]
fn a_config_word_with_no_geometry_is_unhandled_and_writes_nothing() {
    let mut ctx = ExecutionContext::new();
    ctx.accumulator.write_wide(0, [0x1111_2222_3333_4444u64; 16]);
    ctx.vector.write_wide(0, [0x0101_0101u32; 16]);
    ctx.vector.write_wide(2, [0x0101_0101u32; 16]);
    // (3, 3, 7) is no mode the header builds.
    ctx.scalar.write(5, (3 << 1) | (3 << 3) | (7 << 5) | (1 << 8) | (1 << 9));
    assert!(!execute_matmul(&stage_op(0), &mut ctx), "an unmodelled config word must report unhandled");
    ctx.force_commit_all_pending();
    assert_eq!(ctx.accumulator.read_wide(0), [0x1111_2222_3333_4444u64; 16]);
}
