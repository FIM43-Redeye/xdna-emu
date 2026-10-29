//! The scalar coverage universe and target-key parsing.
//!
//! Scalar lowering is structural (not table-driven like vector), so the
//! "table" here is just the static feature list. The universe is 33 keys:
//! 8 arith + 1 branch (localizable, per-region) and 2 loop styles (case-level),
//! each crossed with 3 dtypes. Keys are `{feature}/{dtype}` -- no mode dimension.

use super::chain::{Dtype, ScalarOp, StageOp};

/// The three dtypes, in key-sort-friendly order.
const DTYPES: [Dtype; 3] = [Dtype::I32, Dtype::I16, Dtype::I8];

/// Every coverage key, sorted. 24 arith + 3 branch + 6 loop-style = 33.
pub fn universe_keys() -> Vec<String> {
    let mut keys = Vec::with_capacity(33);
    for d in DTYPES {
        for op in ScalarOp::all() {
            keys.push(format!("{}/{:?}", op.feature(), d));
        }
        keys.push(format!("branch/{:?}", d));
        keys.push(format!("loop_simple/{:?}", d));
        keys.push(format!("loop_hw/{:?}", d));
    }
    keys.sort();
    keys
}

/// What a target key asks generation to guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// Force a stage of this op kind into the chain.
    Stage(StageOp),
    /// Force the outer loop to be a simple for-loop.
    LoopSimple,
    /// Force the outer loop to be a hardware/pipelined loop.
    LoopHw,
}

/// Parsed target: dtype + what to guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub dtype: Dtype,
    pub kind: TargetKind,
}

/// Resolve `{feature}/{dtype}` to a [`Target`]. Panics on a malformed/unknown
/// key -- callers feed keys from [`universe_keys`].
pub fn parse_key(key: &str) -> Target {
    let (feature, dtype_s) = key.rsplit_once('/').unwrap_or_else(|| panic!("bad key {key:?}"));
    let dtype = match dtype_s {
        "I32" => Dtype::I32,
        "I16" => Dtype::I16,
        "I8" => Dtype::I8,
        _ => panic!("bad dtype in key {key:?}"),
    };
    let kind = match feature {
        "add" => TargetKind::Stage(StageOp::Arith(ScalarOp::Add)),
        "sub" => TargetKind::Stage(StageOp::Arith(ScalarOp::Sub)),
        "mul" => TargetKind::Stage(StageOp::Arith(ScalarOp::Mul)),
        "and" => TargetKind::Stage(StageOp::Arith(ScalarOp::And)),
        "or" => TargetKind::Stage(StageOp::Arith(ScalarOp::Or)),
        "xor" => TargetKind::Stage(StageOp::Arith(ScalarOp::Xor)),
        "shl" => TargetKind::Stage(StageOp::Arith(ScalarOp::Shl)),
        "shr" => TargetKind::Stage(StageOp::Arith(ScalarOp::Shr)),
        "branch" => TargetKind::Stage(StageOp::BranchSelect),
        "loop_simple" => TargetKind::LoopSimple,
        "loop_hw" => TargetKind::LoopHw,
        _ => panic!("unknown feature in key {key:?}"),
    };
    Target { dtype, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::chain::{Dtype, ScalarOp, StageOp};
    use std::collections::HashSet;

    #[test]
    fn universe_has_33_unique_keys() {
        let u = universe_keys();
        assert_eq!(u.len(), 33, "expected 33 keys, got {}", u.len());
        let set: HashSet<_> = u.iter().collect();
        assert_eq!(set.len(), 33, "duplicate keys");
    }

    #[test]
    fn universe_is_sorted() {
        let u = universe_keys();
        let mut s = u.clone();
        s.sort();
        assert_eq!(u, s);
    }

    #[test]
    fn universe_covers_every_feature_dtype() {
        let u: HashSet<String> = universe_keys().into_iter().collect();
        for d in ["I32", "I16", "I8"] {
            for f in
                ["add", "sub", "mul", "and", "or", "xor", "shl", "shr", "branch", "loop_simple", "loop_hw"]
            {
                assert!(u.contains(&format!("{f}/{d}")), "missing {f}/{d}");
            }
        }
    }

    #[test]
    fn parse_arith_target() {
        let t = parse_key("xor/I16");
        assert_eq!(t.dtype, Dtype::I16);
        assert_eq!(t.kind, TargetKind::Stage(StageOp::Arith(ScalarOp::Xor)));
    }

    #[test]
    fn parse_branch_target() {
        let t = parse_key("branch/I8");
        assert_eq!(t.dtype, Dtype::I8);
        assert_eq!(t.kind, TargetKind::Stage(StageOp::BranchSelect));
    }

    #[test]
    fn parse_loop_targets() {
        assert_eq!(parse_key("loop_simple/I32").kind, TargetKind::LoopSimple);
        assert_eq!(parse_key("loop_hw/I32").kind, TargetKind::LoopHw);
    }
}
