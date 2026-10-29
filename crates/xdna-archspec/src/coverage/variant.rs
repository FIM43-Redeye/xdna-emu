//! Per-variant coverage: the behaviourally distinct forms of one SemanticOp.
//!
//! A SemanticOp is too coarse to verify as a unit: `Mac` was once "verified"
//! by integer fuzzing while every bf16 `Mac` was wrong. Coverage nodes are
//! therefore (op, variant) pairs, and an op's verdict is the worst over its
//! variants. The variant universe is derived, never hand-listed:
//!
//! - encoding-typed ops take the element types TableGen labels each encoding
//!   with (`InstrEncoding.element_type`, plus `from_type` for dual-typed SRS,
//!   UPS, conversions, pack and unpack);
//! - MAC-family and accumulator-ALU ops are typed at runtime by the config
//!   word, so their variants are the modes llvm-aie's `aiev2_vmult.h` emits
//!   (`MAC_CONFIG_MODES`), with operand signedness for the multiplying kinds.
//!
//! SRS/pack saturation and rounding modes are deliberately not an axis: the
//! lane-level silicon tests pin them exhaustively per variant.

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::aie2::isa::{load_from_generated, ElementType, SemanticOp};
use crate::aie2::matmul::{MacOpKind, MAC_CONFIG_MODES};
use crate::coverage::all_semantic_ops;
use crate::types::Architecture;

/// One behaviourally distinct form of a SemanticOp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpVariant {
    /// No type axis: scalar, control and side-effect ops, and vector forms
    /// TableGen leaves untyped.
    Untyped,
    /// Typed by the encoding: the output element type, and the input type for
    /// dual-typed ops.
    Typed {
        out: ElementType,
        from: Option<ElementType>,
    },
    /// A multiplying MAC-family op in config mode (amode, bmode, variant)
    /// with the given operand signedness (true = signed).
    MacMode {
        amode: u8,
        bmode: u8,
        variant: u8,
        sgn_x: bool,
        sgn_y: bool,
    },
    /// An accumulator ALU op (add, sub, neg on accumulators) in config mode
    /// (amode, bmode, variant). No multiply, so no signedness.
    AccMode { amode: u8, bmode: u8, variant: u8 },
}

impl OpVariant {
    /// The (amode, bmode, variant) config key of a config-typed variant.
    pub fn mode_key(&self) -> Option<(u8, u8, u8)> {
        match *self {
            OpVariant::MacMode { amode, bmode, variant, .. }
            | OpVariant::AccMode { amode, bmode, variant } => Some((amode, bmode, variant)),
            OpVariant::Untyped | OpVariant::Typed { .. } => None,
        }
    }

    /// The `aiev2_vmult.h` intrinsic family of a config-typed variant
    /// (`4x16_16x8`, `elem_16_2`, ...), for human-facing docs.
    pub fn mode_family(&self) -> Option<&'static str> {
        let key = self.mode_key()?;
        MAC_CONFIG_MODES
            .iter()
            .find(|m| (m.amode as u8, m.bmode as u8, m.variant as u8) == key)
            .map(|m| m.family)
    }
}

/// Stable text form, shared with the emulator's fuzz recorder keys
/// (`{op:?}/{variant}/m{mode}`): `untyped`, `Int16`, `Int16<-Int32`,
/// `mac(1,3,2,su)` (x then y signedness, s = signed), `acc(1,0,0)`.
impl fmt::Display for OpVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = |s: bool| if s { 's' } else { 'u' };
        match self {
            OpVariant::Untyped => write!(f, "untyped"),
            OpVariant::Typed { out, from: None } => write!(f, "{out:?}"),
            OpVariant::Typed { out, from: Some(from) } => write!(f, "{out:?}<-{from:?}"),
            OpVariant::MacMode { amode, bmode, variant, sgn_x, sgn_y } => {
                write!(f, "mac({amode},{bmode},{variant},{}{})", sign(*sgn_x), sign(*sgn_y))
            }
            OpVariant::AccMode { amode, bmode, variant } => write!(f, "acc({amode},{bmode},{variant})"),
        }
    }
}

/// The variants of `op` on `arch`, sorted. Every op has at least one; an
/// arch without a wired ISA gives every op `[Untyped]`.
pub fn variants_of(arch: Architecture, op: SemanticOp) -> Vec<OpVariant> {
    let op = canonical(op);
    match arch {
        Architecture::Aie2 => aie2_universe().get(&op).cloned().unwrap_or_else(|| vec![OpVariant::Untyped]),
        _ => vec![OpVariant::Untyped],
    }
}

/// The config-mode operation kind of `mnemonic`, if its variant comes from
/// the runtime config word rather than its element-type label: the
/// instruction stems `MAC_CONFIG_MODES` names (`vmac`, `vaddmac`, `vadd` on
/// accumulators, ...) and their bf16 `.f` forms. Shared by the variant
/// universe and the emulator's fuzz recorder so both key variants alike.
pub fn mode_typed_kind(mnemonic: &str) -> Option<MacOpKind> {
    static KINDS: OnceLock<HashMap<String, MacOpKind>> = OnceLock::new();
    KINDS
        .get_or_init(|| {
            let mut m = HashMap::new();
            for mode in MAC_CONFIG_MODES {
                for &kind in mode.ops {
                    let stem = kind.mnemonic_stem();
                    m.insert(
                        if mode.bfloat {
                            format!("{stem}.f")
                        } else {
                            stem.to_string()
                        },
                        kind,
                    );
                }
            }
            m
        })
        .get(mnemonic)
        .copied()
}

/// `Intrinsic(_)` is one node (the catch-all), whatever its table index.
fn canonical(op: SemanticOp) -> SemanticOp {
    match op {
        SemanticOp::Intrinsic(_) => SemanticOp::Intrinsic(0),
        other => other,
    }
}

fn aie2_universe() -> &'static HashMap<SemanticOp, Vec<OpVariant>> {
    static UNIVERSE: OnceLock<HashMap<SemanticOp, Vec<OpVariant>>> = OnceLock::new();
    UNIVERSE.get_or_init(build_aie2_universe)
}

fn build_aie2_universe() -> HashMap<SemanticOp, Vec<OpVariant>> {
    let isa = load_from_generated();
    let encodings: Vec<_> = isa.encodings_by_slot.values().flatten().collect();
    let mut by_op: HashMap<SemanticOp, Vec<OpVariant>> = HashMap::new();

    // Config-typed ops: each mode's operation kinds name an instruction stem;
    // bf16 modes use its `.f` form. Those mnemonics take their variants from
    // the modes, not from their (accumulator-width) element-type label.
    for mode in MAC_CONFIG_MODES {
        let (amode, bmode, variant) = (mode.amode as u8, mode.bmode as u8, mode.variant as u8);
        for kind in mode.ops {
            let mnemonic = if mode.bfloat {
                format!("{}.f", kind.mnemonic_stem())
            } else {
                kind.mnemonic_stem().to_string()
            };
            let ops: Vec<SemanticOp> = encodings
                .iter()
                .filter(|e| e.mnemonic == mnemonic)
                .filter_map(|e| e.semantic)
                .collect();
            for op in ops {
                let entry = by_op.entry(canonical(op)).or_default();
                if kind.is_multiply() {
                    for &(sx, sy) in mode.signs {
                        entry.push(OpVariant::MacMode {
                            amode,
                            bmode,
                            variant,
                            sgn_x: sx == 1,
                            sgn_y: sy == 1,
                        });
                    }
                } else {
                    entry.push(OpVariant::AccMode { amode, bmode, variant });
                }
            }
        }
    }

    for e in &encodings {
        let Some(op) = e.semantic else { continue };
        if mode_typed_kind(&e.mnemonic).is_some() {
            continue;
        }
        let v = match e.element_type {
            Some(out) => OpVariant::Typed { out, from: e.from_type },
            None => OpVariant::Untyped,
        };
        by_op.entry(canonical(op)).or_default().push(v);
    }

    for op in all_semantic_ops() {
        by_op.entry(op).or_insert_with(|| vec![OpVariant::Untyped]);
    }
    for vs in by_op.values_mut() {
        vs.sort();
        vs.dedup();
    }
    by_op
}

#[cfg(test)]
mod tests {
    use super::*;
    use ElementType::*;
    use SemanticOp::*;

    const A: Architecture = Architecture::Aie2;

    fn typed(out: ElementType, from: Option<ElementType>) -> OpVariant {
        OpVariant::Typed { out, from }
    }

    #[test]
    fn every_listed_op_has_a_variant_and_every_encoded_op_is_listed() {
        // The reconciliation all_semantic_ops() has lacked: every SemanticOp a
        // TableGen encoding carries must be in the static universe.
        let listed = all_semantic_ops();
        for op in &listed {
            assert!(!variants_of(A, *op).is_empty(), "{op:?} has no variant");
        }
        for e in load_from_generated().encodings_by_slot.values().flatten() {
            if let Some(op) = e.semantic {
                assert!(listed.contains(&canonical(op)), "{} carries unlisted {op:?}", e.name);
            }
        }
    }

    #[test]
    fn mac_family_variants_are_config_modes_with_signedness() {
        // Mac's universe is every mode whose kinds include Mac, times its
        // sign pairs; the integer and .f forms both come from modes.
        let want: usize = MAC_CONFIG_MODES
            .iter()
            .filter(|m| m.ops.contains(&MacOpKind::Mac))
            .map(|m| m.signs.len())
            .sum();
        let mac = variants_of(A, Mac);
        assert_eq!(mac.len(), want);
        assert!(mac.iter().all(|v| matches!(v, OpVariant::MacMode { .. })), "{mac:?}");
        // bf16 dense 4x8x4 (vmac.f) and the int8 x int4 base mode, all signs.
        assert!(mac.contains(&OpVariant::MacMode {
            amode: 2,
            bmode: 3,
            variant: 0,
            sgn_x: false,
            sgn_y: false
        }));
        for (sx, sy) in [(false, false), (false, true), (true, false), (true, true)] {
            assert!(mac.contains(&OpVariant::MacMode {
                amode: 0,
                bmode: 0,
                variant: 0,
                sgn_x: sx,
                sgn_y: sy
            }));
        }
        for op in [MatMul, MatMulSub, NegMatMul, AddMac, SubMac, NegMul] {
            let vs = variants_of(A, op);
            assert!(vs.iter().all(|v| matches!(v, OpVariant::MacMode { .. })), "{op:?}: {vs:?}");
        }
    }

    #[test]
    fn accumulator_alu_ops_take_acc_modes() {
        // vadd on accumulators (Accumulate) is config-typed: acc32, acc64 and
        // accfloat modes, no signedness.
        let acc = variants_of(A, Accumulate);
        assert!(acc.iter().all(|v| matches!(v, OpVariant::AccMode { .. })), "{acc:?}");
        assert!(acc.contains(&OpVariant::AccMode { amode: 2, bmode: 3, variant: 0 }));
    }

    #[test]
    fn dual_typed_ops_carry_both_types() {
        let srs = variants_of(A, Srs);
        assert!(srs.contains(&typed(Int16, Some(Int32))));
        assert!(srs.contains(&typed(Int16, Some(Int64))));
        assert!(!srs.contains(&OpVariant::Untyped));
        let pack = variants_of(A, Pack);
        assert!(pack.contains(&typed(Int8, Some(Int16))));
        assert!(pack.contains(&typed(UInt4, Some(UInt8))));
        let unpack = variants_of(A, Unpack);
        assert!(unpack.contains(&typed(Int16, Some(Int8))));
        assert!(unpack.iter().all(|v| matches!(v, OpVariant::Typed { from: Some(_), .. })), "{unpack:?}");
    }

    #[test]
    fn mixed_ops_keep_untyped_and_typed_forms() {
        // Scalar add is untyped; vadd.8/16/32 are typed Add forms.
        let add = variants_of(A, Add);
        assert!(add.contains(&OpVariant::Untyped));
        for et in [Int8, Int16, Int32] {
            assert!(add.contains(&typed(et, None)), "{et:?}");
        }
    }

    #[test]
    fn intrinsic_is_one_node_and_other_arches_are_untyped() {
        assert_eq!(variants_of(A, Intrinsic(7)), vec![OpVariant::Untyped]);
        assert_eq!(variants_of(Architecture::Aie2p, Mac), vec![OpVariant::Untyped]);
    }

    #[test]
    fn mode_typed_kinds_cover_int_and_bf16_forms_only() {
        assert_eq!(mode_typed_kind("vmac"), Some(MacOpKind::Mac));
        assert_eq!(mode_typed_kind("vmac.f"), Some(MacOpKind::Mac));
        assert_eq!(mode_typed_kind("vaddmsc.f"), Some(MacOpKind::AddMsc));
        assert_eq!(mode_typed_kind("vadd"), Some(MacOpKind::Add));
        // Element-typed vector adds are not config-typed.
        assert_eq!(mode_typed_kind("vadd.16"), None);
        assert_eq!(mode_typed_kind("vsrs.s16.s32"), None);
    }

    #[test]
    fn display_is_the_recorder_key_form() {
        assert_eq!(OpVariant::Untyped.to_string(), "untyped");
        assert_eq!(typed(Int8, None).to_string(), "Int8");
        assert_eq!(typed(Int16, Some(Int32)).to_string(), "Int16<-Int32");
        let m = OpVariant::MacMode { amode: 1, bmode: 3, variant: 2, sgn_x: true, sgn_y: false };
        assert_eq!(m.to_string(), "mac(1,3,2,su)");
        assert_eq!(OpVariant::AccMode { amode: 1, bmode: 0, variant: 0 }.to_string(), "acc(1,0,0)");
        assert_eq!(m.mode_family(), Some("elem_16_2"));
    }
}
