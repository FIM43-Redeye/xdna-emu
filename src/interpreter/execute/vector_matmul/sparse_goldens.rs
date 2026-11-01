//! Sparse MAC against NPU1 silicon, lane for lane.
//!
//! Every capture of `tools/gen_mac_sparse.py` (`vec_macsparse_<stem>_{probe,ops,rand,conf,alt}`)
//! is replayed through `execute_matmul` the way the decoder would present it: the sparse
//! operand is a `qx` register (512 data bits from an x register plus a 128-bit q register),
//! A comes from an x register (`_sparse_narrow`) or a y register (`_sparse_wide`), and the
//! config word is the one Chess or Peano built. Every accumulator lane must equal silicon
//! bit for bit.
//!
//! - `probe`: designed tiles that determine the mask rule (one-hot stored data against an A
//!   ramp, every mask shape). The rule in `xdna_archspec::aie2::matmul` was read off these.
//! - `ops`: every MAC-family op with accumulator inputs; `conf`: the config-word variants
//!   (zero_acc1, shift16, sub_mul, sub_acc1, sub_acc2).
//! - `rand`: seeded random tiles through mul -> mac -> msc. Held out: never used to derive the
//!   rule.
//! - `alt`: the other encoding of each mode (wide for the narrow modes, narrow for the wide
//!   ones), built with the toolchain builtins.

use super::*;
use crate::interpreter::bundle::SlotIndex;

/// (stem, (amode, bmode, variant)) of the five sparse modes.
const MODES: [(&str, (u32, u32, u32)); 5] = [
    ("4x32_32x8", (0, 0, 1)),
    ("4x16_16x8", (0, 1, 5)),
    ("2x16_16x8", (1, 2, 2)),
    ("2x8_8x8", (1, 3, 5)),
    ("4x16_16x4", (2, 3, 2)),
];

/// Registers the replay uses: A in x0 (narrow) or y2 (wide), the sparse operand in qx1.
const A_NARROW_REG: u8 = 0;
const A_WIDE_REG: u8 = 8;
const QX_INDEX: u8 = 1;
const QX_VECTOR_REG: u8 = 2;
const ACC1_REG: u8 = 0;
const ACC2_REG: u8 = 2;
const DEST_REG: u8 = 4;

struct Form {
    instr: String,
    sgn_x: u32,
    sgn_y: u32,
    /// zero_acc1, shift16, sub_mul, sub_acc1, sub_acc2 (config-word controls of `*_conf` stages).
    controls: [u32; 5],
}

struct Golden {
    key: (u32, u32, u32),
    a_bits: u32,
    b_bits: u32,
    acc_bits: u32,
    acc_lanes: usize,
    bf16: bool,
    wide: bool,
    kind: String,
    tiles: usize,
    stages: usize,
    a_lanes: usize,
    b_stored: usize,
    out_words: usize,
    flavours: usize,
    forms: Vec<Form>,
    a: Vec<Vec<u64>>,
    d: Vec<Vec<u64>>,
    m: Vec<Vec<u128>>,
    silicon: Vec<u64>,
}

fn load(kernel: &str) -> Golden {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let num = |x: &serde_json::Value| x.as_u64().unwrap();
    let list = |x: &serde_json::Value| -> Vec<u64> { x.as_array().unwrap().iter().map(num).collect() };
    let mode = &v["mode"];
    let tile = &v["tile"];
    let sets = num(&tile["sets"]) as usize;
    let inputs = v["inputs"].as_array().unwrap();
    let forms: Vec<Form> = v["forms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let get = |k: &str| f.get(k).map_or(0, |x| num(x) as u32);
            Form {
                instr: f["instr"].as_str().unwrap().to_string(),
                sgn_x: get("sgn_x"),
                sgn_y: get("sgn_y"),
                controls: [
                    get("zero_acc1"),
                    get("shift16"),
                    get("sub_mul"),
                    get("sub_acc1"),
                    get("sub_acc2"),
                ],
            }
        })
        .collect();
    let stages = num(&tile["stages"]) as usize;
    Golden {
        key: (num(&mode["amode"]) as u32, num(&mode["bmode"]) as u32, num(&mode["variant"]) as u32),
        a_bits: num(&mode["a_bits"]) as u32,
        b_bits: num(&mode["b_bits"]) as u32,
        acc_bits: num(&mode["acc_bits"]) as u32,
        acc_lanes: num(&mode["acc_lanes"]) as usize,
        bf16: mode["bf16"].as_bool().unwrap(),
        wide: mode["wide"].as_bool().unwrap(),
        kind: v["class"].as_str().unwrap().trim_start_matches("mac_sparse_").to_string(),
        tiles: num(&tile["tiles"]) as usize,
        stages,
        a_lanes: num(&tile["a_lanes"]) as usize,
        b_stored: num(&mode["b_stored"]) as usize,
        out_words: num(&tile["out_words"]) as usize,
        flavours: forms.len() / stages,
        forms,
        a: (0..sets).map(|s| list(&inputs[3 * s])).collect(),
        d: (0..sets).map(|s| list(&inputs[3 * s + 1])).collect(),
        m: (0..sets)
            .map(|s| {
                inputs[3 * s + 2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| u128::from_str_radix(x.as_str().unwrap(), 16).unwrap())
                    .collect()
            })
            .collect(),
        silicon: list(&v["silicon"]),
    }
}

/// Elements of `bits` width, little-endian, into `words` 32-bit words.
fn pack(values: &[u64], bits: u32, words: usize) -> Vec<u32> {
    let mut out = vec![0u32; words];
    let per_word = (32 / bits) as usize;
    for (i, &x) in values.iter().enumerate() {
        out[i / per_word] |= (x as u32) << (bits as usize * (i % per_word));
    }
    out
}

fn vec512(words: &[u32]) -> Vec512 {
    let mut v = [0u32; 16];
    v.copy_from_slice(&words[..16]);
    v
}

fn config_word(key: (u32, u32, u32), form: &Form) -> u32 {
    let [zero_acc1, shift16, sub_mul, sub_acc1, sub_acc2] = form.controls;
    zero_acc1
        | (key.0 << 1)
        | (key.1 << 3)
        | (key.2 << 5)
        | (form.sgn_y << 8)
        | (form.sgn_x << 9)
        | (shift16 << 10)
        | (sub_mul << 11)
        | (sub_acc1 << 12)
        | (sub_acc2 << 13)
}

/// The semantic and TableGen record name the decoder gives each op's sparse encoding.
fn semantic_of(instr: &str) -> SemanticOp {
    match instr {
        "mul" => SemanticOp::MatMul,
        "negmul" | "negmsc" => SemanticOp::NegMul,
        "mac" => SemanticOp::Mac,
        "msc" => SemanticOp::MatMulSub,
        "negmac" => SemanticOp::NegMatMul,
        "addmac" | "addmsc" => SemanticOp::AddMac,
        "submac" | "submsc" => SemanticOp::SubMac,
        other => panic!("unknown op {other}"),
    }
}

fn sparse_op(g: &Golden, instr: &str, wide: bool) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, semantic_of(instr));
    op.is_vector = true;
    op.is_wide_vector = true;
    op.is_quad_vector = wide;
    let family = if g.bf16 {
        "_F_vmac_bm_core"
    } else {
        "_vmac_cm_core"
    };
    let enc = if wide { "sparse_wide" } else { "sparse_narrow" };
    op.encoding_name = Some(format!("V{}{family}_{enc}", instr.to_ascii_uppercase()));
    if g.bf16 {
        op.element_type = Some(ElementType::Float32);
    }
    let reg = if wide { A_WIDE_REG } else { A_NARROW_REG };
    op.sources.push(Operand::VectorReg(reg));
    op.sources.push(Operand::SparseQxReg(QX_INDEX));
    op.sources.push(Operand::ScalarReg(5));
    op.dest = Some(Operand::AccumReg(DEST_REG));
    let reads_acc1 = !matches!(instr, "mul" | "negmul");
    if reads_acc1 {
        op.sources.push(Operand::AccumReg(ACC1_REG));
    }
    if matches!(instr, "addmac" | "addmsc" | "submac" | "submsc") {
        op.sources.push(Operand::AccumReg(ACC2_REG));
    }
    op
}

/// An accumulator image from the 32-bit words a capture stored for one stage.
fn acc_from_words(g: &Golden, words: &[u64]) -> Acc1024 {
    let mut acc = [0u64; 16];
    if g.acc_bits == 64 {
        for l in 0..g.acc_lanes {
            acc[l] = words[l] | (words[g.acc_lanes + l] << 32);
        }
    } else {
        for (l, &w) in words.iter().enumerate() {
            acc[l / 2] |= w << (32 * (l % 2));
        }
    }
    acc
}

fn words_from_acc(g: &Golden, acc: &Acc1024) -> Vec<u64> {
    if g.acc_bits == 64 {
        let mut w = vec![0u64; 2 * g.acc_lanes];
        for l in 0..g.acc_lanes {
            w[l] = acc[l] & 0xFFFF_FFFF;
            w[g.acc_lanes + l] = acc[l] >> 32;
        }
        w
    } else {
        (0..g.out_words).map(|l| (acc[l / 2] >> (32 * (l % 2))) & 0xFFFF_FFFF).collect()
    }
}

fn write_acc(ctx: &mut ExecutionContext, g: &Golden, reg: u8, words: &[u64]) {
    let acc = acc_from_words(g, words);
    if g.bf16 {
        ctx.accumulator.write(reg, acc[..8].try_into().unwrap());
    } else {
        ctx.accumulator.write_wide(reg, acc);
    }
}

fn read_acc(ctx: &ExecutionContext, g: &Golden, reg: u8) -> Vec<u64> {
    let mut acc = [0u64; 16];
    if g.bf16 {
        acc[..8].copy_from_slice(&ctx.accumulator.read(reg));
    } else {
        acc = ctx.accumulator.read_wide(reg);
    }
    words_from_acc(g, &acc)
}

/// Load one operand set into the registers an instruction of this encoding reads.
fn load_operands(ctx: &mut ExecutionContext, g: &Golden, wide: bool, tile: usize, set: usize) {
    let a = &g.a[set][tile * g.a_lanes..(tile + 1) * g.a_lanes];
    let a_bits = if g.bf16 { 16 } else { g.a_bits };
    let a_words = pack(a, a_bits, 32);
    if wide {
        ctx.vector.write_wide(A_WIDE_REG, vec512(&a_words[..16]));
        ctx.vector.write_wide(A_WIDE_REG + 2, vec512(&a_words[16..]));
    } else {
        ctx.vector.write_wide(A_NARROW_REG, vec512(&a_words[..16]));
    }
    let d = &g.d[set][tile * g.b_stored..(tile + 1) * g.b_stored];
    let b_bits = if g.bf16 { 16 } else { g.b_bits };
    ctx.vector.write_wide(QX_VECTOR_REG, vec512(&pack(d, b_bits, 16)));
    let m = g.m[set][tile];
    ctx.mask
        .write(QX_INDEX, [m as u32, (m >> 32) as u32, (m >> 64) as u32, (m >> 96) as u32]);
}

/// Replay one capture; returns (lanes checked, first mismatches).
fn replay(kernel: &str) -> (usize, Vec<String>) {
    let g = load(kernel);
    let wide = if g.kind == "alt" { !g.wide } else { g.wide };
    let (mut checked, mut bad) = (0usize, Vec::new());
    let silicon = |tile: usize, flavour: usize, stage: usize| -> &[u64] {
        let base = ((tile * g.flavours + flavour) * g.stages + stage) * g.out_words;
        &g.silicon[base..base + g.out_words]
    };
    for tile in 0..g.tiles {
        for flavour in 0..g.flavours {
            for stage in 0..g.stages {
                let form = &g.forms[flavour * g.stages + stage];
                // Which operand set and accumulator inputs the stage reads.
                let (set, acc1_from, acc2_from) = match g.kind.as_str() {
                    "probe" => (0, None, None),
                    "rand" | "spec" => (stage, stage.checked_sub(1), None),
                    _ => match stage {
                        0 | 1 => (stage, None, None),
                        _ => (2, Some(0), Some(1)),
                    },
                };
                let mut ctx = ExecutionContext::new();
                // A poisoned accumulator: a lane the executor leaves unwritten shows up as a mismatch.
                if g.bf16 {
                    ctx.accumulator.write(DEST_REG, [0xDEAD_BEEF_DEAD_BEEFu64; 8]);
                } else {
                    ctx.accumulator.write_wide(DEST_REG, [0xDEAD_BEEF_DEAD_BEEFu64; 16]);
                }
                load_operands(&mut ctx, &g, wide, tile, set);
                if let Some(s) = acc1_from {
                    write_acc(&mut ctx, &g, ACC1_REG, silicon(tile, flavour, s));
                }
                if let Some(s) = acc2_from {
                    write_acc(&mut ctx, &g, ACC2_REG, silicon(tile, flavour, s));
                }
                ctx.scalar.write(5, config_word(g.key, form));
                let op = sparse_op(&g, &form.instr, wide);
                assert!(execute_matmul(&op, &mut ctx), "{kernel}: {} must be handled", form.instr);
                ctx.force_commit_all_pending();
                let got = read_acc(&ctx, &g, DEST_REG);
                let want = silicon(tile, flavour, stage);
                for lane in 0..g.out_words {
                    checked += 1;
                    if got[lane] != want[lane] {
                        if bad.len() < 5 {
                            bad.push(format!(
                                "tile {tile} flavour {flavour} stage {stage} ({}) word {lane}: {:#x} != silicon {:#x}",
                                form.instr, got[lane], want[lane]
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
        "{kernel}: {} of {checked} words differ from silicon, first: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

fn run_kind(kind: &str, min_words: usize) {
    let mut total = 0;
    for (stem, _) in MODES {
        let kernel = format!("vec_macsparse_{stem}_{kind}");
        let (checked, bad) = replay(&kernel);
        assert_clean(&kernel, checked, &bad);
        total += checked;
    }
    assert!(total >= min_words, "only {total} {kind} words replayed");
}

#[test]
fn matches_silicon_on_every_probe_lane() {
    run_kind("probe", 400_000);
}

#[test]
fn matches_silicon_on_every_op() {
    run_kind("ops", 220_000);
}

#[test]
fn matches_silicon_on_every_config_word_variant() {
    run_kind("conf", 220_000);
}

/// Held-out random tiles through mul -> mac -> msc, all four signedness flavours.
#[test]
fn matches_silicon_on_every_held_out_lane() {
    run_kind("rand", 150_000);
}

/// Held-out bfloat16 tiles with zeros, Inf, NaNs, denormals and overflowing exponents among the
/// normal numbers, against absent and present units of the mask.
#[test]
fn matches_silicon_on_special_values() {
    let kernel = "vec_macsparse_4x16_16x4_spec";
    let (checked, bad) = replay(kernel);
    assert!(checked >= 3_000, "only {checked} special-value words replayed");
    assert_clean(kernel, checked, &bad);
}

/// The other encoding of every mode: the wide form of a narrow mode reads the low 512 bits
/// of its y register, the narrow form of a wide mode zero-extends its x register.
#[test]
fn matches_silicon_on_the_alternative_encodings() {
    run_kind("alt", 220_000);
}

/// A sparse config word whose key has no sparse geometry is reported unhandled.
#[test]
fn a_dense_geometry_with_a_qx_operand_is_unhandled_and_writes_nothing() {
    let mut ctx = ExecutionContext::new();
    ctx.accumulator.write_wide(DEST_REG, [0x1111_2222_3333_4444u64; 16]);
    // (0, 1, 0) is the dense int8 tile, not a sparse mode.
    ctx.scalar.write(5, (1 << 3) | (1 << 8) | (1 << 9));
    let g = load("vec_macsparse_4x16_16x8_probe");
    let op = sparse_op(&g, "mul", false);
    assert!(!execute_matmul(&op, &mut ctx), "a qx operand needs a sparse geometry");
    ctx.force_commit_all_pending();
    assert_eq!(ctx.accumulator.read_wide(DEST_REG), [0x1111_2222_3333_4444u64; 16]);
}
