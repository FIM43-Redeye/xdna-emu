//! Chain AST for the scalar fuzzer.
//!
//! A [`ScalarChain`] is one fuzz case: N elementwise stages over a per-case
//! dtype, run inside a `region_len`-iteration element loop. Stage k writes its
//! result to output region k (`out[k*region_len + i]`); its operands draw from
//! the input buffer, an earlier stage's register (recency chain), or a literal.
//! Coverage localizes per region exactly as the vector tenant localizes per
//! 64-byte slice.

use serde::{Deserialize, Serialize};

/// Scalar element type. The `{:?}` spelling (`I32`/`I16`/`I8`) is embedded in
/// coverage keys; [`template_dtype`](Dtype::template_dtype) gives the lowercase
/// form the compile template expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    I32,
    I16,
    I8,
}

impl Dtype {
    /// C element type as spelled in the kernel.
    pub fn ctype(self) -> &'static str {
        match self {
            Dtype::I32 => "int32_t",
            Dtype::I16 => "int16_t",
            Dtype::I8 => "int8_t",
        }
    }

    /// Lowercase dtype string passed to `compile_kernel_case` / the template.
    pub fn template_dtype(self) -> &'static str {
        match self {
            Dtype::I32 => "i32",
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
        }
    }

    /// Bytes per element.
    pub fn byte_size(self) -> usize {
        match self {
            Dtype::I32 => 4,
            Dtype::I16 => 2,
            Dtype::I8 => 1,
        }
    }
}

/// How the element loop iterates. Case-level coverage dimension (the ZOL boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopStyle {
    Simple,
    HardwareLoop,
}

/// Scalar arithmetic operators (the localizable arith features).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

impl ScalarOp {
    /// Coverage-key feature name.
    pub fn feature(self) -> &'static str {
        match self {
            ScalarOp::Add => "add",
            ScalarOp::Sub => "sub",
            ScalarOp::Mul => "mul",
            ScalarOp::And => "and",
            ScalarOp::Or => "or",
            ScalarOp::Xor => "xor",
            ScalarOp::Shl => "shl",
            ScalarOp::Shr => "shr",
        }
    }

    /// C operator spelling.
    pub fn c_operator(self) -> &'static str {
        match self {
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

    /// The eight arith ops, for table/universe construction.
    pub fn all() -> [ScalarOp; 8] {
        use ScalarOp::*;
        [Add, Sub, Mul, And, Or, Xor, Shl, Shr]
    }
}

/// What a stage does: a binary arith op, or an elementwise branch-select
/// (`out[i] = cond ? a : b`). Both write their own region and localize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageOp {
    Arith(ScalarOp),
    BranchSelect,
}

impl StageOp {
    /// Coverage-key feature: arith op name, or `branch`.
    pub fn feature(self) -> &'static str {
        match self {
            StageOp::Arith(op) => op.feature(),
            StageOp::BranchSelect => "branch",
        }
    }
}

/// A stage operand source. `Prior(j)` references stage j's register `t{j}`;
/// generation guarantees `j` is an earlier stage (`j < k`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operand {
    /// `in[i]` -- the input buffer element at the loop index.
    Input,
    /// `t{j}` -- an earlier stage's result register.
    Prior(usize),
    /// An integer literal.
    Literal(i32),
}

/// One chain stage: an op plus its operands. `cond` is used only by
/// `BranchSelect`; arith stages ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarStage {
    pub op: StageOp,
    pub a: Operand,
    pub b: Operand,
    pub cond: Operand,
}

/// One fuzz case: deterministic stages for `(seed, target_key)` plus the dtype,
/// region length, and loop style.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarChain {
    pub seed: u64,
    pub target_key: String,
    pub dtype: Dtype,
    /// Elements per stage output region (the element-loop trip count).
    pub region_len: usize,
    pub loop_style: LoopStyle,
    pub stages: Vec<ScalarStage>,
}

impl ScalarChain {
    /// The per-stage coverage key (`{feature}/{dtype}`), in region order.
    pub fn stage_keys(&self) -> Vec<String> {
        self.stages
            .iter()
            .map(|s| format!("{}/{:?}", s.op.feature(), self.dtype))
            .collect()
    }

    /// The case-level loop-style coverage key.
    pub fn loop_key(&self) -> String {
        let style = match self.loop_style {
            LoopStyle::Simple => "loop_simple",
            LoopStyle::HardwareLoop => "loop_hw",
        };
        format!("{}/{:?}", style, self.dtype)
    }

    /// All keys a passing case credits: stage keys (region order) then the
    /// loop-style key. The trailing loop key is credited but is never a region
    /// (the comparator filters `loop_`-prefixed keys before localizing).
    pub fn keys(&self) -> Vec<String> {
        let mut keys = self.stage_keys();
        keys.push(self.loop_key());
        keys
    }

    /// Total output elements: one `region_len` region per stage.
    pub fn out_elems(&self) -> usize {
        self.stages.len() * self.region_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ScalarChain {
        ScalarChain {
            seed: 1,
            target_key: "add/I32".into(),
            dtype: Dtype::I32,
            region_len: 32,
            loop_style: LoopStyle::Simple,
            stages: vec![
                ScalarStage {
                    op: StageOp::Arith(ScalarOp::Add),
                    a: Operand::Input,
                    b: Operand::Literal(3),
                    cond: Operand::Input,
                },
                ScalarStage {
                    op: StageOp::BranchSelect,
                    a: Operand::Prior(0),
                    b: Operand::Literal(0),
                    cond: Operand::Input,
                },
            ],
        }
    }

    #[test]
    fn keys_are_stage_keys_then_loop_key() {
        let c = sample();
        assert_eq!(
            c.keys(),
            vec!["add/I32".to_string(), "branch/I32".to_string(), "loop_simple/I32".to_string()]
        );
    }

    #[test]
    fn loop_key_reflects_style_and_dtype() {
        let mut c = sample();
        c.loop_style = LoopStyle::HardwareLoop;
        c.dtype = Dtype::I8;
        assert_eq!(c.loop_key(), "loop_hw/I8");
    }

    #[test]
    fn out_elems_is_stages_times_region_len() {
        assert_eq!(sample().out_elems(), 2 * 32);
    }

    #[test]
    fn dtype_strings_match_template_contract() {
        assert_eq!(Dtype::I32.template_dtype(), "i32");
        assert_eq!(Dtype::I16.template_dtype(), "i16");
        assert_eq!(Dtype::I8.template_dtype(), "i8");
    }

    #[test]
    fn dtype_debug_matches_key_spelling() {
        assert_eq!(format!("{:?}", Dtype::I32), "I32");
    }
}
