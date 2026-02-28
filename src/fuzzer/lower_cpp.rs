//! Lower kernel AST to C++ source code.
//!
//! Produces a self-contained kernel function that both Peano and Chess
//! can compile for AIE2.

use crate::fuzzer::ast::*;
use crate::fuzzer::params::*;

/// Lower a FuzzParams to a complete C++ kernel source file.
pub fn lower_to_cpp(params: &FuzzParams) -> String {
    let mut out = String::new();
    let ctype = dtype_to_ctype(params.dtype);

    // Header
    out.push_str("#include <stdint.h>\n\n");
    out.push_str(&format!(
        "extern \"C\" void fuzz_kernel({ctype}* __restrict buf_in, {ctype}* __restrict buf_out) {{\n"
    ));

    // Loop preamble
    let n = params.buffer_size;
    match params.body.loop_style {
        LoopStyle::Simple => {
            out.push_str(&format!("    for (int i = 0; i < {}; i++) {{\n", n));
        }
        LoopStyle::HardwareLoop => {
            // chess_prepare_for_pipelining / chess_loop_range hint
            out.push_str(&format!(
                "    for (int i = 0; i < {}; i++) chess_prepare_for_pipelining {{\n",
                n
            ));
        }
    }

    if params.body.ops.is_empty() {
        // Default passthrough: copy input to output
        out.push_str("        buf_out[i] = buf_in[i];\n");
    } else {
        // Declare temporaries
        let max_var = max_var_id(&params.body.ops);
        for v in 0..=max_var {
            out.push_str(&format!("        {} t{} = 0;\n", ctype, v));
        }
        // Lower ops
        for op in &params.body.ops {
            lower_op(&mut out, op, "        ");
        }
    }

    out.push_str("    }\n"); // close loop
    out.push_str("}\n"); // close function
    out
}

/// Map ScalarType to C type string.
fn dtype_to_ctype(dtype: ScalarType) -> &'static str {
    match dtype {
        ScalarType::I32 => "int32_t",
        ScalarType::I16 => "int16_t",
        ScalarType::I8 => "int8_t",
    }
}

/// Find the highest variable ID used across all ops (recursively).
fn max_var_id(ops: &[KernelOp]) -> u8 {
    let mut max = 0u8;
    for op in ops {
        match op {
            KernelOp::ScalarArith { dst, .. } => max = max.max(dst.0),
            KernelOp::Branch {
                then_ops,
                else_ops,
                ..
            } => {
                max = max.max(max_var_id(then_ops));
                max = max.max(max_var_id(else_ops));
            }
            KernelOp::HwLoop { body, .. } => {
                max = max.max(max_var_id(body));
            }
            _ => {}
        }
    }
    max
}

/// Lower a single KernelOp to C++ text, appending to `out`.
fn lower_op(out: &mut String, op: &KernelOp, indent: &str) {
    match op {
        KernelOp::ScalarArith {
            op: sop,
            dst,
            src1,
            src2,
        } => {
            out.push_str(&format!(
                "{}t{} = {} {} {};\n",
                indent,
                dst.0,
                lower_operand(src1),
                scalar_op_str(*sop),
                lower_operand(src2),
            ));
        }
        KernelOp::Store { buf, idx, val } => {
            let bufname = if buf.0 == 0 { "buf_in" } else { "buf_out" };
            out.push_str(&format!(
                "{}{}[{}] = {};\n",
                indent,
                bufname,
                lower_operand(idx),
                lower_operand(val),
            ));
        }
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            out.push_str(&format!("{}if ({}) {{\n", indent, lower_operand(cond)));
            let inner = format!("{}    ", indent);
            for op in then_ops {
                lower_op(out, op, &inner);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", indent));
                for op in else_ops {
                    lower_op(out, op, &inner);
                }
            }
            out.push_str(&format!("{}}}\n", indent));
        }
        KernelOp::HwLoop { count, body } => {
            out.push_str(&format!(
                "{}for (int _hw = 0; _hw < {}; _hw++) {{\n",
                indent, count
            ));
            let inner = format!("{}    ", indent);
            for op in body {
                lower_op(out, op, &inner);
            }
            out.push_str(&format!("{}}}\n", indent));
        }
    }
}

/// Lower an Operand to a C++ expression string.
fn lower_operand(op: &Operand) -> String {
    match op {
        Operand::Var(v) => format!("t{}", v.0),
        Operand::Literal(n) => format!("{}", n),
        Operand::Load { buf, idx } => {
            let bufname = if buf.0 == 0 { "buf_in" } else { "buf_out" };
            format!("{}[{}]", bufname, lower_operand(idx))
        }
    }
}

/// Map ScalarOp to C++ operator string.
fn scalar_op_str(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "+",
        ScalarOp::Sub => "-",
        ScalarOp::Mul => "*",
        ScalarOp::And => "&",
        ScalarOp::Or => "|",
        ScalarOp::Xor => "^",
        ScalarOp::Shl => "<<",
        ScalarOp::Shr => ">>",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_empty_body_produces_passthrough() {
        let params = FuzzParams {
            seed: 1,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        // Should contain function signature, loop, and buffer copy
        assert!(cpp.contains("void fuzz_kernel("));
        assert!(cpp.contains("int32_t"));
        assert!(cpp.contains("buf_out[i] = buf_in[i]"));
    }

    #[test]
    fn test_lower_add_one() {
        let params = FuzzParams {
            seed: 2,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![
                    KernelOp::ScalarArith {
                        op: ScalarOp::Add,
                        dst: Var(0),
                        src1: Operand::Load {
                            buf: BufRef(0),
                            idx: Box::new(Operand::Var(Var(1))),
                        },
                        src2: Operand::Literal(1),
                    },
                    KernelOp::Store {
                        buf: BufRef(1),
                        idx: Operand::Var(Var(1)),
                        val: Operand::Var(Var(0)),
                    },
                ],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("buf_in["));  // loads from input
        assert!(cpp.contains("+ 1"));      // adds literal
        assert!(cpp.contains("buf_out[")); // stores to output
    }

    #[test]
    fn test_lower_i16_dtype() {
        let params = FuzzParams {
            seed: 3,
            buffer_size: 32,
            dtype: ScalarType::I16,
            body: KernelBody {
                ops: vec![],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("int16_t"));
        assert!(!cpp.contains("int32_t"));
    }

    #[test]
    fn test_lower_i8_dtype() {
        let params = FuzzParams {
            seed: 4,
            buffer_size: 128,
            dtype: ScalarType::I8,
            body: KernelBody {
                ops: vec![],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("int8_t"));
        assert!(cpp.contains("128"));
    }

    #[test]
    fn test_lower_hardware_loop() {
        let params = FuzzParams {
            seed: 5,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![],
                loop_style: LoopStyle::HardwareLoop,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("chess_prepare_for_pipelining"));
    }

    #[test]
    fn test_lower_branch() {
        let params = FuzzParams {
            seed: 6,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![KernelOp::Branch {
                    cond: Operand::Var(Var(0)),
                    then_ops: vec![KernelOp::Store {
                        buf: BufRef(1),
                        idx: Operand::Var(Var(1)),
                        val: Operand::Literal(99),
                    }],
                    else_ops: vec![KernelOp::Store {
                        buf: BufRef(1),
                        idx: Operand::Var(Var(1)),
                        val: Operand::Literal(0),
                    }],
                }],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("if (t0)"));
        assert!(cpp.contains("} else {"));
        assert!(cpp.contains("99"));
    }

    #[test]
    fn test_lower_hwloop_op() {
        let params = FuzzParams {
            seed: 7,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![KernelOp::HwLoop {
                    count: 4,
                    body: vec![KernelOp::ScalarArith {
                        op: ScalarOp::Add,
                        dst: Var(0),
                        src1: Operand::Var(Var(0)),
                        src2: Operand::Literal(1),
                    }],
                }],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("_hw < 4"));
        assert!(cpp.contains("t0 = t0 + 1"));
    }

    #[test]
    fn test_lower_all_scalar_ops() {
        // Verify all operator strings are emitted correctly
        let ops = vec![
            (ScalarOp::Add, "+"),
            (ScalarOp::Sub, "-"),
            (ScalarOp::Mul, "*"),
            (ScalarOp::And, "&"),
            (ScalarOp::Or, "|"),
            (ScalarOp::Xor, "^"),
            (ScalarOp::Shl, "<<"),
            (ScalarOp::Shr, ">>"),
        ];
        for (sop, expected_str) in ops {
            let params = FuzzParams {
                seed: 100,
                buffer_size: 16,
                dtype: ScalarType::I32,
                body: KernelBody {
                    ops: vec![KernelOp::ScalarArith {
                        op: sop,
                        dst: Var(0),
                        src1: Operand::Var(Var(0)),
                        src2: Operand::Literal(2),
                    }],
                    loop_style: LoopStyle::Simple,
                },
            };
            let cpp = lower_to_cpp(&params);
            assert!(
                cpp.contains(expected_str),
                "Missing operator '{}' for {:?}",
                expected_str,
                sop
            );
        }
    }

    #[test]
    fn test_lower_multiple_temporaries() {
        let params = FuzzParams {
            seed: 8,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![
                    KernelOp::ScalarArith {
                        op: ScalarOp::Add,
                        dst: Var(0),
                        src1: Operand::Literal(1),
                        src2: Operand::Literal(2),
                    },
                    KernelOp::ScalarArith {
                        op: ScalarOp::Mul,
                        dst: Var(3),
                        src1: Operand::Var(Var(0)),
                        src2: Operand::Literal(10),
                    },
                ],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        // Should declare t0, t1, t2, t3 (max var is 3)
        assert!(cpp.contains("int32_t t0 = 0;"));
        assert!(cpp.contains("int32_t t1 = 0;"));
        assert!(cpp.contains("int32_t t2 = 0;"));
        assert!(cpp.contains("int32_t t3 = 0;"));
    }

    #[test]
    fn test_lower_nested_load() {
        let params = FuzzParams {
            seed: 9,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![KernelOp::ScalarArith {
                    op: ScalarOp::Add,
                    dst: Var(0),
                    src1: Operand::Load {
                        buf: BufRef(0),
                        idx: Box::new(Operand::Var(Var(1))),
                    },
                    src2: Operand::Load {
                        buf: BufRef(1),
                        idx: Box::new(Operand::Var(Var(1))),
                    },
                }],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("buf_in[t1]"));
        assert!(cpp.contains("buf_out[t1]"));
    }
}
