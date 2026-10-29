//! Thread-local executed-vector-op recorder for the vector fuzzer.
//!
//! The fuzzer emits C++ chains; compilers can fold ops away. Correctness
//! credit for a fuzzed stage comes from its silicon-matched output slice;
//! this recorder is the anti-folding sentinel and audit record: it proves
//! the emulator actually executed vector semantics, and captures the set of
//! (SemanticOp, OpVariant, mode) keys executed per case for coverage
//! banking. The variant is the archspec coverage `OpVariant`, so a key names
//! exactly one node of the per-variant coverage universe.
//!
//! Disarmed by default; the hook in [`super::VectorAlu::execute`] checks
//! [`is_armed`] first, so the variant (a config-word decode and a string) is
//! only computed while a fuzzer campaign is recording.

use std::cell::RefCell;
use std::collections::HashSet;

use super::VectorAlu;
use crate::interpreter::bundle::{ElementType, SlotOp};
use crate::interpreter::state::ExecutionContext;
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::coverage::variant::{mode_typed_kind, OpVariant};

thread_local! {
    static RECORDED: RefCell<Option<HashSet<String>>> = const { RefCell::new(None) };
}

/// Arm the recorder for the current thread, clearing any previous set.
pub fn arm() {
    RECORDED.with(|r| *r.borrow_mut() = Some(HashSet::new()));
}

/// Take the recorded keys (sorted, deduplicated) and disarm.
///
/// Returns `None` if the recorder was never armed on this thread.
pub fn take() -> Option<Vec<String>> {
    RECORDED.with(|r| {
        r.borrow_mut().take().map(|set| {
            let mut keys: Vec<String> = set.into_iter().collect();
            keys.sort();
            keys
        })
    })
}

/// Whether the recorder is armed on this thread.
pub fn is_armed() -> bool {
    RECORDED.with(|r| r.borrow().is_some())
}

/// The coverage variant an executed vector op belongs to.
///
/// Instructions whose variant comes from the runtime MAC config word
/// (`mode_typed_kind`) decode the raw word, laid out as documented on
/// `MatMulConfig::from_config_word`: amode bits 1-2, bmode bits 3-4, variant
/// bits 5-7, sgn_y bit 8, sgn_x bit 9. The decode deliberately bypasses the
/// emulator's geometry lookup so modes the emulator cannot execute are still
/// keyed. A mode-typed mnemonic with no readable config word (no scalar
/// source) falls back to `Untyped`, which names no universe node.
/// Everything else is keyed by its encoding's element types.
pub fn variant_of(
    mnemonic: Option<&str>,
    element_type: Option<ElementType>,
    from_type: Option<ElementType>,
    config_word: Option<u32>,
) -> OpVariant {
    if let Some(kind) = mnemonic.and_then(mode_typed_kind) {
        let Some(conf) = config_word else {
            return OpVariant::Untyped;
        };
        let amode = ((conf >> 1) & 3) as u8;
        let bmode = ((conf >> 3) & 3) as u8;
        let variant = ((conf >> 5) & 7) as u8;
        return if kind.is_multiply() {
            OpVariant::MacMode {
                amode,
                bmode,
                variant,
                sgn_x: (conf >> 9) & 1 != 0,
                sgn_y: (conf >> 8) & 1 != 0,
            }
        } else {
            OpVariant::AccMode { amode, bmode, variant }
        };
    }
    match element_type {
        Some(out) => OpVariant::Typed { out, from: from_type },
        None => OpVariant::Untyped,
    }
}

/// The recorder key parts (variant, mode) of `op`, or `None` when the
/// recorder is unarmed. Reads the op's config register, so call it before the
/// op executes; the config word is a scalar source the op never writes.
///
/// `mode` is `(sat << 4) | rnd` for the ops whose semantics depend on the SRS
/// configuration, 0 otherwise.
pub fn key_parts(semantic: SemanticOp, op: &SlotOp, ctx: &ExecutionContext) -> Option<(OpVariant, u8)> {
    if !is_armed() {
        return None;
    }
    let mode = match semantic {
        SemanticOp::Srs | SemanticOp::Pack | SemanticOp::Convert | SemanticOp::Ups => {
            (ctx.srs_config.saturation_mode << 4) | ctx.srs_config.rounding_mode
        }
        _ => 0,
    };
    let variant = variant_of(
        op.encoding_name.as_deref(),
        op.element_type,
        op.from_type,
        VectorAlu::get_config_register(op, ctx),
    );
    Some((variant, mode))
}

/// Record an executed vector op that always completes (the fused
/// memory-slot forms). No-op unless armed.
pub fn record_op(semantic: SemanticOp, op: &SlotOp, ctx: &ExecutionContext) {
    if let Some((variant, mode)) = key_parts(semantic, op, ctx) {
        record(semantic, &variant, mode);
    }
}

/// Record an executed vector op. No-op unless armed.
///
/// Key format: `{semantic:?}/{variant}/m{mode}` with `variant` the displayed
/// archspec `OpVariant`. `mode` is `(sat << 4) | rnd` for SRS-config-sensitive
/// ops, 0 otherwise.
pub fn record(semantic: SemanticOp, variant: &OpVariant, mode: u8) {
    RECORDED.with(|r| {
        if let Some(set) = r.borrow_mut().as_mut() {
            set.insert(format!("{semantic:?}/{variant}/m{mode}"));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(out: ElementType) -> OpVariant {
        OpVariant::Typed { out, from: None }
    }

    #[test]
    fn test_unarmed_record_noop_take_none() {
        // Never armed on this thread: record is a no-op, take returns None.
        assert!(!is_armed());
        record(SemanticOp::Add, &typed(ElementType::Int32), 0);
        assert_eq!(take(), None);
    }

    #[test]
    fn test_arm_record_take_roundtrip_sorted_dedup() {
        arm();
        assert!(is_armed());
        record(SemanticOp::Sub, &typed(ElementType::Int16), 0);
        record(SemanticOp::Add, &typed(ElementType::Int32), 0);
        record(SemanticOp::Add, &typed(ElementType::Int32), 0); // dup
        record(SemanticOp::Srs, &OpVariant::Untyped, 0x13);

        let keys = take().expect("armed recorder must return Some");
        assert_eq!(
            keys,
            vec!["Add/Int32/m0".to_string(), "Srs/untyped/m19".to_string(), "Sub/Int16/m0".to_string(),]
        );
    }

    #[test]
    fn test_take_disarms() {
        arm();
        record(SemanticOp::Add, &typed(ElementType::Int32), 0);
        assert!(take().is_some());
        assert!(!is_armed());
        record(SemanticOp::Sub, &typed(ElementType::Int32), 0);
        assert_eq!(take(), None);
    }

    fn key(semantic: SemanticOp, v: OpVariant) -> String {
        arm();
        record(semantic, &v, 0);
        take().unwrap().remove(0)
    }

    #[test]
    fn test_key_typed_op() {
        let v = variant_of(Some("vadd.16"), Some(ElementType::Int16), None, None);
        assert_eq!(key(SemanticOp::Add, v), "Add/Int16/m0");
    }

    #[test]
    fn test_key_dual_typed_op() {
        let v = variant_of(Some("vsrs.d16.s32"), Some(ElementType::Int16), Some(ElementType::Int32), None);
        assert_eq!(key(SemanticOp::Srs, v), "Srs/Int16<-Int32/m0");
    }

    #[test]
    fn test_key_integer_mac_decodes_signedness() {
        // amode 1, bmode 3, variant 2 (an unmodelled mode: still keyed),
        // sgn_x = 1 (bit 9), sgn_y = 0 (bit 8).
        let conf = (1 << 1) | (3 << 3) | (2 << 5) | (1 << 9);
        let v = variant_of(Some("vmac"), Some(ElementType::Int32), None, Some(conf));
        assert_eq!(key(SemanticOp::Mac, v), "Mac/mac(1,3,2,su)/m0");
        let conf = (conf ^ (1 << 9)) | (1 << 8);
        let v = variant_of(Some("vmac"), Some(ElementType::Int32), None, Some(conf));
        assert_eq!(key(SemanticOp::Mac, v), "Mac/mac(1,3,2,us)/m0");
    }

    #[test]
    fn test_key_bf16_mac() {
        // aiev2_vmult.h bf16: amode 2, bmode 3, variant 0, unsigned.
        let conf = (2 << 1) | (3 << 3);
        let v = variant_of(Some("vmac.f"), Some(ElementType::Float32), None, Some(conf));
        assert_eq!(key(SemanticOp::Mac, v), "Mac/mac(2,3,0,uu)/m0");
    }

    #[test]
    fn test_key_accumulator_add() {
        // vadd on accumulators: no multiply, so no signedness even if set.
        let conf = (1 << 1) | (1 << 9) | 1;
        let v = variant_of(Some("vadd"), Some(ElementType::Int32), None, Some(conf));
        assert_eq!(key(SemanticOp::Accumulate, v), "Accumulate/acc(1,0,0)/m0");
    }

    #[test]
    fn test_mode_typed_mnemonic_without_config_word_is_untyped() {
        let v = variant_of(Some("vmac"), Some(ElementType::Int32), None, None);
        assert_eq!(v, OpVariant::Untyped);
    }

    #[test]
    fn test_vector_vadd_is_typed_not_accumulator() {
        // The vector (non-accumulator) vadd carries a width suffix and is not
        // mode-typed.
        let v = variant_of(Some("vadd.32"), Some(ElementType::Int32), None, Some(0xffff));
        assert_eq!(v, typed(ElementType::Int32));
    }
}
