//! Chain -> C kernel lowering.
//!
//! [`lower_chain`] renders a [`ScalarChain`] as a self-contained scalar kernel
//! with the fuzz template's 2-arg I/O contract:
//! `extern "C" void fuzz_kernel(DTYPE* __restrict in, DTYPE* __restrict out)`.
//! The kernel loops over `region_len` element indices; inside the body each
//! stage k computes register `t{k}` and stores it to its own region at
//! `out[k*region_len + i]`. Stage k's operands come from `in[i]`, an earlier
//! register `t{j}` (j<k), or a literal -- the in-body stores crossing the loop
//! back-edge preserve the AIE2 ZOL store-flush / recency catches.
//!
//! Signed-overflow UB (e.g. `mul`/`add`/`shl` on `INT_MIN`, or a negative left
//! shift) is intentionally NOT avoided here: the differential compares two
//! compilations of the *same* generated kernel (EMU vs HW silicon), so both
//! sides execute identical UB and agree. We are not checking the kernel against
//! a defined oracle, so overflow is sound for this purpose -- it is never the
//! cause of a real divergence. (The one way it could bite: a future EMU/HW
//! toolchain split that miscompiles the same UB differently; flagged here so a
//! `mul/I32` divergence investigation starts by ruling that out, not chasing
//! overflow in our model.)

use super::chain::{LoopStyle, Operand, ScalarChain, ScalarStage, StageOp};

/// Render an operand as a C expression.
fn operand_expr(op: Operand) -> String {
    match op {
        Operand::Input => "in[i]".to_string(),
        Operand::Prior(j) => format!("t{j}"),
        Operand::Literal(n) => format!("{n}"),
    }
}

/// Render one stage's assignment to its register `t{k}`.
fn stage_expr(stage: &ScalarStage) -> String {
    let a = operand_expr(stage.a);
    let b = operand_expr(stage.b);
    match stage.op {
        StageOp::Arith(op) => format!("{a} {} {b}", op.c_operator()),
        StageOp::BranchSelect => {
            let cond = operand_expr(stage.cond);
            format!("({cond}) ? {a} : {b}")
        }
    }
}

/// Lower a chain to complete C kernel source.
pub fn lower_chain(chain: &ScalarChain) -> String {
    let ctype = chain.dtype.ctype();
    let mut s = String::new();
    s.push_str(&format!(
        "// Generated scalar fuzz chain -- seed {}, target {}. DO NOT EDIT.\n",
        chain.seed, chain.target_key
    ));
    s.push_str("#include <stdint.h>\n\n");
    s.push_str(&format!(
        "extern \"C\" void fuzz_kernel({ctype}* __restrict in, {ctype}* __restrict out) {{\n"
    ));

    let loop_open = match chain.loop_style {
        LoopStyle::Simple => format!("    for (int i = 0; i < {}; i++) {{\n", chain.region_len),
        LoopStyle::HardwareLoop => {
            format!("    for (int i = 0; i < {}; i++) chess_prepare_for_pipelining {{\n", chain.region_len)
        }
    };
    s.push_str(&loop_open);

    for (k, stage) in chain.stages.iter().enumerate() {
        s.push_str(&format!("        {ctype} t{k} = {};\n", stage_expr(stage)));
        s.push_str(&format!("        out[{} + i] = t{k};\n", k * chain.region_len));
    }

    s.push_str("    }\n"); // close loop
    s.push_str("}\n"); // close function
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::gen::generate;
    use crate::fuzzer::domains::scalar::table::universe_keys;

    #[test]
    fn signature_and_balanced_braces_for_400_cases() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            let src = lower_chain(&c);
            let opens = src.matches('{').count();
            let closes = src.matches('}').count();
            assert_eq!(opens, closes, "seed {seed} key {key}: unbalanced braces in\n{src}");
            let sig = format!(
                "extern \"C\" void fuzz_kernel({0}* __restrict in, {0}* __restrict out)",
                c.dtype.ctype()
            );
            assert!(src.contains(&sig), "seed {seed} key {key}: missing signature");
        }
    }

    #[test]
    fn one_store_per_stage_at_region_stride() {
        let c = generate(1, "add/I32");
        let src = lower_chain(&c);
        assert_eq!(src.matches("out[").count(), c.stages.len(), "one store per stage");
        for k in 0..c.stages.len() {
            assert!(
                src.contains(&format!("out[{} + i] = t{k};", k * c.region_len)),
                "stage {k} store at region stride {}",
                k * c.region_len
            );
        }
    }

    #[test]
    fn hardware_loop_style_is_spelled() {
        let c = generate(3, "loop_hw/I32");
        assert!(lower_chain(&c).contains("chess_prepare_for_pipelining"));
        let cs = generate(3, "loop_simple/I32");
        assert!(!lower_chain(&cs).contains("chess_prepare_for_pipelining"));
    }

    #[test]
    fn branch_select_lowers_to_ternary() {
        let c = generate(1, "branch/I32");
        let src = lower_chain(&c);
        assert!(src.contains(" ? "), "branch-select must lower to a ternary:\n{src}");
        assert!(src.contains(" : "), "branch-select ternary needs else arm");
    }

    #[test]
    fn arith_operators_appear() {
        let c = generate(1, "xor/I32");
        let src = lower_chain(&c);
        assert!(src.contains(" ^ "), "xor operator missing:\n{src}");
    }

    #[test]
    fn golden_header_and_frame() {
        let c = generate(1, "add/I32");
        let src = lower_chain(&c);
        assert!(src.starts_with("// Generated scalar fuzz chain -- seed 1, target add/I32. DO NOT EDIT.\n"));
        assert!(src.ends_with("}\n"));
        assert!(src.contains(&format!("for (int i = 0; i < {}; i++)", c.region_len)));
    }
}
