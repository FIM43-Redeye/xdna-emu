//! The MAC configuration word against NPU1 silicon (`tools/golden/silicon/mac_conf.json`, built by
//! `tools/mac_conf_golden.py` from the probe suites of `tools/gen_mac_conf_probe.py`).
//!
//! Dense tiles run every instruction kind over the modes the word can select: the header's integer
//! modes, the amode-1 bmode alias, the two complex modes, the float-conf multiplier (amode 2/3), the
//! idle codes, with the zero_acc, shift16, sub and `term_neg` fields varied (plans `termneg`, `same`,
//! `flags`, `modes`, `fsingle`, `repro`). Sparse tiles run the Chess `mul_*` sparse intrinsics with one
//! `term_neg` bit at a time. Every one of the 1024 output bits of every tile must equal silicon.

use super::*;
use crate::interpreter::bundle::SlotIndex;
use serde::Deserialize;

#[derive(Deserialize)]
struct Dense {
    plan: String,
    kind: String,
    conf: u32,
    a: String,
    b: String,
    acc1: String,
    acc2: String,
    out: String,
}

#[derive(Deserialize)]
struct Sparse {
    geom: String,
    conf: u32,
    a: String,
    data: String,
    mask: String,
    out: String,
}

#[derive(Deserialize)]
struct Golden {
    dense: Vec<Dense>,
    sparse: Vec<Sparse>,
}

fn golden() -> Golden {
    let path = format!("{}/tools/golden/silicon/mac_conf.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

fn words(hex: &str) -> Vec<u32> {
    (0..hex.len() / 8)
        .map(|i| u32::from_le_bytes(hex_bytes(&hex[8 * i..8 * i + 8]).try_into().unwrap()))
        .collect()
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn acc_from(hex: &str) -> Acc1024 {
    let w = words(hex);
    std::array::from_fn(|i| w[2 * i] as u64 | (w[2 * i + 1] as u64) << 32)
}

fn semantic_of(kind: &str) -> SemanticOp {
    match kind {
        "mul" => SemanticOp::MatMul,
        "negmul" | "negmsc" => SemanticOp::NegMul,
        "mac" => SemanticOp::Mac,
        "msc" => SemanticOp::MatMulSub,
        "negmac" => SemanticOp::NegMatMul,
        "addmac" | "addmsc" => SemanticOp::AddMac,
        "submac" | "submsc" => SemanticOp::SubMac,
        other => panic!("unknown kind {other}"),
    }
}

fn run(op: &SlotOp, ctx: &mut ExecutionContext) -> Acc1024 {
    assert!(execute_matmul(op, ctx), "{:?} must dispatch to the matmul unit", op.encoding_name);
    ctx.force_commit_all_pending();
    ctx.accumulator.read_wide(4)
}

#[test]
fn dense_tiles_match_silicon_for_every_conf_field() {
    let g = golden();
    let mut bad = Vec::new();
    for (n, t) in g.dense.iter().enumerate() {
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, acc_from(&t.acc1));
        ctx.accumulator.write_wide(2, acc_from(&t.acc2));
        ctx.accumulator.write_wide(4, [0xDEAD_BEEF_DEAD_BEEFu64; 16]);
        ctx.vector.write_wide(0, words(&t.a).try_into().unwrap());
        ctx.vector.write_wide(2, words(&t.b).try_into().unwrap());
        ctx.scalar.write(5, t.conf);
        let name = format!("V{}_vmac_cm_core_dense", t.kind.to_ascii_uppercase());
        let mut op = super::tests::make_mac_op(semantic_of(&t.kind), 0, 2, 5, 4, Some(&name));
        if !matches!(t.kind.as_str(), "mul" | "negmul") {
            op.sources.push(Operand::AccumReg(0));
        }
        if matches!(t.kind.as_str(), "addmac" | "addmsc" | "submac" | "submsc") {
            op.sources.push(Operand::AccumReg(2));
        }
        let got = run(&op, &mut ctx);
        if got != acc_from(&t.out) {
            bad.push(format!("#{n} {} {} conf {:#010x}", t.plan, t.kind, t.conf));
        }
    }
    assert!(g.dense.len() > 2000);
    assert!(
        bad.is_empty(),
        "{} of {} tiles differ, first: {:?}",
        bad.len(),
        g.dense.len(),
        &bad[..bad.len().min(8)]
    );
}

#[test]
fn sparse_term_neg_matches_silicon() {
    let g = golden();
    let mut bad = Vec::new();
    for (n, t) in g.sparse.iter().enumerate() {
        let mut ctx = ExecutionContext::new();
        ctx.vector.write_wide(0, words(&t.a).try_into().unwrap());
        ctx.vector.write_wide(2, words(&t.data).try_into().unwrap());
        let m = words(&t.mask);
        ctx.mask.write(1, [m[0], m[1], m[2], m[3]]);
        ctx.scalar.write(5, t.conf);
        let mut op = SlotOp::from_semantic(SlotIndex::Accumulator, SemanticOp::MatMul);
        op.is_vector = true;
        op.is_wide_vector = true;
        op.encoding_name = Some("VMUL_vmac_cm_core_sparse_narrow".to_string());
        op.sources.push(Operand::VectorReg(0));
        op.sources.push(Operand::SparseQxReg(1));
        op.sources.push(Operand::ScalarReg(5));
        op.dest = Some(Operand::AccumReg(4));
        if run(&op, &mut ctx) != acc_from(&t.out) {
            bad.push(format!("#{n} {} conf {:#010x}", t.geom, t.conf));
        }
    }
    assert!(g.sparse.len() > 500);
    assert!(
        bad.is_empty(),
        "{} of {} tiles differ, first: {:?}",
        bad.len(),
        g.sparse.len(),
        &bad[..bad.len().min(8)]
    );
}
