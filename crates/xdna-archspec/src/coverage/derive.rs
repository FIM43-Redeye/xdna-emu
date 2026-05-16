//! Derived-default clustering (spec Section 3, Section 5). `category()` is the
//! SemanticOp forcing function: exhaustive, no wildcard. `default_verdict`
//! gives each category its honestly-pessimistic bootstrap verdict.

use crate::aie2::isa::SemanticOp;
use crate::coverage::verdict::{Provenance, Verification, Verdict};

/// Coarse SemanticOp grouping for derived-default provenance (spec Section 3).
/// Finer behavioral seams are the override registry's job (Task 4), not this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Arithmetic,
    Bitwise,
    Comparison,
    Memory,
    ControlFlow,
    Vector,
    Sync,
    SideEffect,
    /// New/unclassified — defaults to Unspecified (default-to-ignorant).
    NeedsTriage,
}

/// SemanticOp -> Category. EXHAUSTIVE WITH NO `_` ARM ON PURPOSE: this is the
/// forcing function (spec Section 1/3). A new variant breaks this build until
/// categorized. Grouping mirrors the enum's own comment sections in
/// `aie2/isa/types.rs:237-376`.
pub fn category(op: &SemanticOp) -> Category {
    match op {
        // This arm deliberately coalesces ALL fully-toolchain-specified scalar
        // machinery into one pessimistic-default bucket: pure arithmetic,
        // div-step, bit-manip, extension, copy/nop, pointer, cycle-counter,
        // and event. They are lumped together NOT because the enum groups
        // them this way (it does not -- this arm spans several distinct enum
        // comment sections) but because they all share the exact same honest
        // default, ToolchainDerived/NotApplicable. Only the Vector/Sync/
        // SideEffect arms below actually mirror the enum's comment sections.
        //
        // `Event` emits a trace packet via a toolchain-specified instruction
        // and `ReadCycleCounter` reads a toolchain-specified counter, so
        // ToolchainDerived is correct at THIS layer; their trace-buffering
        // behavior is a Phase-2 override candidate (a future `Timing`/`trace`
        // override unit), not an optimistic call made here.
        SemanticOp::Add
        | SemanticOp::Sub
        | SemanticOp::Adc
        | SemanticOp::Sbc
        | SemanticOp::Mul
        | SemanticOp::SDiv
        | SemanticOp::UDiv
        | SemanticOp::SRem
        | SemanticOp::URem
        | SemanticOp::Abs
        | SemanticOp::Neg
        | SemanticOp::DivStep
        | SemanticOp::Ctlz
        | SemanticOp::Cttz
        | SemanticOp::Ctpop
        | SemanticOp::Bswap
        | SemanticOp::Clb
        | SemanticOp::SignExtend
        | SemanticOp::ZeroExtend
        | SemanticOp::Truncate
        | SemanticOp::Copy
        | SemanticOp::Nop
        | SemanticOp::Event
        | SemanticOp::ReadCycleCounter
        | SemanticOp::PointerAdd
        | SemanticOp::PointerMov => Category::Arithmetic,

        // Bitwise
        SemanticOp::And
        | SemanticOp::Or
        | SemanticOp::Xor
        | SemanticOp::Not
        | SemanticOp::Shl
        | SemanticOp::Sra
        | SemanticOp::Srl
        | SemanticOp::AshlBidir
        | SemanticOp::LshlBidir
        | SemanticOp::Rotl
        | SemanticOp::Rotr => Category::Bitwise,

        // Comparison
        SemanticOp::SetEq
        | SemanticOp::SetNe
        | SemanticOp::SetLt
        | SemanticOp::SetLe
        | SemanticOp::SetGt
        | SemanticOp::SetGe
        | SemanticOp::SetUlt
        | SemanticOp::SetUle
        | SemanticOp::SetUgt
        | SemanticOp::SetUge
        | SemanticOp::Cmp => Category::Comparison,

        // Memory
        SemanticOp::Load | SemanticOp::Store => Category::Memory,

        // Control flow (Done/Halt are well-understood core termination;
        // Select is the conditional-select/ternary, filed under "Control
        // flow" in the SemanticOp enum's own comment sections)
        SemanticOp::Br
        | SemanticOp::BrCond
        | SemanticOp::Call
        | SemanticOp::Ret
        | SemanticOp::Select
        | SemanticOp::Done
        | SemanticOp::Halt => Category::ControlFlow,

        // Vector / matrix engine — AietoolsModeled, perishable (spec Section 5)
        SemanticOp::Mac
        | SemanticOp::MatMul
        | SemanticOp::MatMulSub
        | SemanticOp::NegMatMul
        | SemanticOp::AddMac
        | SemanticOp::SubMac
        | SemanticOp::NegMul
        | SemanticOp::Srs
        | SemanticOp::Ups
        | SemanticOp::Shuffle
        | SemanticOp::Pack
        | SemanticOp::Unpack
        | SemanticOp::Align
        | SemanticOp::VectorBroadcast
        | SemanticOp::VectorExtract
        | SemanticOp::VectorInsert
        | SemanticOp::VectorPush
        | SemanticOp::VectorPushHi
        | SemanticOp::VectorSelect
        | SemanticOp::VectorClear
        | SemanticOp::Convert
        | SemanticOp::Min
        | SemanticOp::Max
        | SemanticOp::SubLt
        | SemanticOp::SubGe
        | SemanticOp::MaxDiffLt
        | SemanticOp::MaxLt
        | SemanticOp::MinGe
        | SemanticOp::AbsGtz
        | SemanticOp::NegGtz
        | SemanticOp::NegLtz
        | SemanticOp::NegAdd
        | SemanticOp::Accumulate
        | SemanticOp::AccumSub
        | SemanticOp::AccumNegAdd
        | SemanticOp::AccumNegSub => Category::Vector,

        // Synchronization — aie-rt fully specifies lock semantics
        SemanticOp::LockAcquire | SemanticOp::LockRelease => Category::Sync,

        // Side effects — DMA/stream/cascade: timing/semantics only
        // doc-specified, perishable (spec Section 5)
        SemanticOp::CascadeRead
        | SemanticOp::CascadeWrite
        | SemanticOp::StreamRead
        | SemanticOp::StreamWrite
        | SemanticOp::StreamWritePacketHeader
        | SemanticOp::DmaStart
        | SemanticOp::DmaWait => Category::SideEffect,

        // Target-specific intrinsic: a source describes it (intrinsic table)
        // but this layer asserts no model -> default-to-ignorant.
        SemanticOp::Intrinsic(_) => Category::NeedsTriage,
    }
}

/// Honestly-pessimistic bootstrap default per category (spec Section 5).
/// A default is always the weakest plausible provenance for the group;
/// refinement (Phase 2) can only improve a verdict, never hide one.
pub fn default_verdict(cat: Category) -> Verdict {
    match cat {
        Category::Arithmetic
        | Category::Bitwise
        | Category::Comparison
        | Category::Memory
        | Category::ControlFlow
        | Category::Sync => Verdict {
            provenance: Provenance::ToolchainDerived,
            verification: Verification::NotApplicable,
        },
        Category::Vector => {
            Verdict { provenance: Provenance::AietoolsModeled, verification: Verification::Unverified }
        }
        Category::SideEffect => {
            // Spec Section 5 names this tier "timing"; there are no
            // timing-only SemanticOps (timing constants live at the
            // register/spine level, e.g. the `timer` capability domain).
            // DMA/stream/cascade side effects share that pessimistic tier.
            Verdict { provenance: Provenance::DocSpecified, verification: Verification::Unverified }
        }
        Category::NeedsTriage => {
            Verdict { provenance: Provenance::Unspecified, verification: Verification::Unverified }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verification};

    #[test]
    fn scalar_arithmetic_is_toolchain_ground_truth() {
        assert_eq!(category(&SemanticOp::Add), Category::Arithmetic);
        let v = default_verdict(Category::Arithmetic);
        assert_eq!(v.provenance, Provenance::ToolchainDerived);
        assert_eq!(v.verification, Verification::NotApplicable);
    }

    #[test]
    fn vector_is_aietools_modeled_unverified() {
        assert_eq!(category(&SemanticOp::Mac), Category::Vector);
        assert_eq!(category(&SemanticOp::Srs), Category::Vector);
        let v = default_verdict(Category::Vector);
        assert_eq!(v.provenance, Provenance::AietoolsModeled);
        assert_eq!(v.verification, Verification::Unverified);
    }

    #[test]
    fn side_effect_is_doc_specified_unverified() {
        assert_eq!(category(&SemanticOp::DmaStart), Category::SideEffect);
        let v = default_verdict(Category::SideEffect);
        assert_eq!(v.provenance, Provenance::DocSpecified);
        assert_eq!(v.verification, Verification::Unverified);
    }

    #[test]
    fn intrinsic_is_needs_triage_and_thus_a_comprehension_gap() {
        assert_eq!(category(&SemanticOp::Intrinsic(0)), Category::NeedsTriage);
        let v = default_verdict(Category::NeedsTriage);
        assert_eq!(v.provenance, Provenance::Unspecified);
        // Default-to-ignorant (spec Section 1): the lazy path lands loud.
        assert!(v.is_comprehension_gap());
    }

    #[test]
    fn locks_are_toolchain_derived() {
        assert_eq!(category(&SemanticOp::LockAcquire), Category::Sync);
        assert_eq!(default_verdict(Category::Sync).provenance, Provenance::ToolchainDerived);
    }
}
