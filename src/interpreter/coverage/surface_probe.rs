//! Axis-1 evidence: the interpreter's concrete `impl SurfaceProbe`. archspec
//! DEFINES SurfaceClass/SurfaceProbe (single point of definition, spec
//! Section 2); this answers the contract with knowledge that only lives here
//! -- which dispatch unit actually handles each SemanticOp. Compiler-forced
//! exhaustive: a new variant stops THIS build until its true handler is
//! classified (spec Section 1/2/4 Axis-1 forcing function).

use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
use xdna_archspec::coverage::CoverageNode;

/// The interpreter's Axis-1 evidence source. Zero-sized: the answer is a pure
/// function of the SemanticOp variant and the (static) dispatch topology.
pub struct InterpreterSurfaceProbe;

impl SurfaceProbe for InterpreterSurfaceProbe {
    fn surface_class(&self, node: &CoverageNode) -> SurfaceClass {
        match node {
            CoverageNode::Semantic { op, .. } => semantic_surface(op),
            // Register-consumer / capability surface probing is a separate
            // future effort, NOT Plan 2. The reconciliation test feeds only
            // Semantic nodes, so this is structurally unreachable in Plan 2
            // -- a loud `unreachable!`, never a silent false `Absent` (which
            // would be the exact lie this design fights).
            CoverageNode::Register { .. } | CoverageNode::Capability { .. } => {
                unreachable!("register/capability surface probing is out of Plan-2 scope")
            }
        }
    }
}

/// EXHAUSTIVE, NO `_` ARM ON PURPOSE (spec Section 1/2/4 Axis-1 forcing
/// function). Grouped by the REAL dispatch unit that handles each op
/// (`cycle_accurate.rs:127-185`), which is an independent partition from
/// archspec's `category()` -- the reconciliation test (a later task) relies
/// on that independence. A new SemanticOp variant breaks THIS build until its
/// true handler is classified.
fn semantic_surface(op: &SemanticOp) -> SurfaceClass {
    match op {
        // Wired -- explicit scalar handler arms in execute_semantic
        // (semantic.rs:87-181).
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
        | SemanticOp::And
        | SemanticOp::Or
        | SemanticOp::Xor
        | SemanticOp::Not
        | SemanticOp::Shl
        | SemanticOp::Sra
        | SemanticOp::Srl
        | SemanticOp::AshlBidir
        | SemanticOp::LshlBidir
        | SemanticOp::SetEq
        | SemanticOp::SetNe
        | SemanticOp::SetLt
        | SemanticOp::SetLe
        | SemanticOp::SetGt
        | SemanticOp::SetGe
        | SemanticOp::SetUlt
        | SemanticOp::SetUle
        | SemanticOp::SetUgt
        | SemanticOp::SetUge
        | SemanticOp::Cmp
        | SemanticOp::Select
        | SemanticOp::Ctlz
        | SemanticOp::Cttz
        | SemanticOp::Ctpop
        | SemanticOp::Bswap
        | SemanticOp::Clb
        | SemanticOp::Rotl
        | SemanticOp::Rotr
        | SemanticOp::Copy
        | SemanticOp::PointerAdd
        | SemanticOp::PointerMov
        | SemanticOp::SignExtend
        | SemanticOp::ZeroExtend
        | SemanticOp::Truncate
        | SemanticOp::ReadCycleCounter
        | SemanticOp::Nop
        | SemanticOp::Event => SurfaceClass::Wired,

        // Wired via VectorAlu (cycle_accurate.rs:131). These fall THROUGH
        // execute_semantic on the is_vector check (semantic.rs:79-80) but a
        // real handler exists downstream -- Wired, not Fallthrough.
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
        | SemanticOp::AccumNegSub => SurfaceClass::Wired,

        // Wired via MemoryUnit (cycle_accurate.rs:135).
        SemanticOp::Load | SemanticOp::Store => SurfaceClass::Wired,

        // Wired via CascadeOps (cycle_accurate.rs:142).
        SemanticOp::CascadeRead | SemanticOp::CascadeWrite => SurfaceClass::Wired,

        // Wired via StreamOps (cycle_accurate.rs:152).
        SemanticOp::StreamRead | SemanticOp::StreamWrite | SemanticOp::StreamWritePacketHeader => {
            SurfaceClass::Wired
        }

        // Wired via ControlUnit (cycle_accurate.rs:158); LockAcquire/Release
        // drive the device lock model through the same path.
        SemanticOp::Br
        | SemanticOp::BrCond
        | SemanticOp::Call
        | SemanticOp::Ret
        | SemanticOp::Done
        | SemanticOp::Halt
        | SemanticOp::LockAcquire
        | SemanticOp::LockRelease => SurfaceClass::Wired,

        // Wired via ControlUnit (control.rs:372, :379) -- explicit handler
        // arms in execute_with_neighbor_locks, same dispatch path as locks.
        SemanticOp::DmaStart | SemanticOp::DmaWait => SurfaceClass::Wired,

        // Absent: falls through execute_semantic `_ =>` arm (semantic.rs:190),
        // VectorAlu returns false via `_ => false` catch-all
        // (vector_dispatch.rs:120), no other unit claims it, hits hard
        // ExecuteResult::Error (cycle_accurate.rs:178). Labeled but unhandled.
        SemanticOp::Intrinsic(_) => SurfaceClass::Absent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::aie2::isa::SemanticOp;
    use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
    use xdna_archspec::coverage::CoverageNode;
    use xdna_archspec::types::Architecture;

    fn sc(op: SemanticOp) -> SurfaceClass {
        InterpreterSurfaceProbe.surface_class(&CoverageNode::Semantic { arch: Architecture::Aie2, op })
    }

    #[test]
    fn scalar_alu_ops_are_wired_via_execute_semantic() {
        // semantic.rs:87-181 -- explicit handler arms.
        assert_eq!(sc(SemanticOp::Add), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Xor), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::SetLt), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Select), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Copy), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Nop), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Event), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::ReadCycleCounter), SurfaceClass::Wired);
    }

    #[test]
    fn vector_ops_are_wired_via_vector_alu() {
        // Fall through execute_semantic (is_vector) but VectorAlu handles them
        // -- Wired, NOT Fallthrough. This is the subtlety the probe must get
        // right (cycle_accurate.rs:131).
        assert_eq!(sc(SemanticOp::Mac), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::MatMul), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Srs), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Shuffle), SurfaceClass::Wired);
    }

    #[test]
    fn memory_stream_cascade_control_sync_are_wired() {
        assert_eq!(sc(SemanticOp::Load), SurfaceClass::Wired); // MemoryUnit
        assert_eq!(sc(SemanticOp::Store), SurfaceClass::Wired); // MemoryUnit
        assert_eq!(sc(SemanticOp::CascadeRead), SurfaceClass::Wired); // CascadeOps
        assert_eq!(sc(SemanticOp::StreamWrite), SurfaceClass::Wired); // StreamOps
        assert_eq!(sc(SemanticOp::Br), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::Call), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::Halt), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::LockAcquire), SurfaceClass::Wired); // lock model
    }

    #[test]
    fn intrinsic_is_absent_not_silently_wired() {
        // Step-3 investigation determined Intrinsic(_) is Absent: it hits the
        // execute_semantic `_ =>` delegated arm (semantic.rs:190), VectorAlu
        // ends `_ => false` (vector_dispatch.rs:120), no unit claims it, so it
        // reaches the hard ExecuteResult::Error (cycle_accurate.rs:178). This
        // pins that finding -- a future regression silently classifying it
        // Wired (claiming the emulator surfaces arbitrary intrinsics when it
        // does not) fails here. Coheres with Axis-2 Unspecified for Intrinsic.
        assert_eq!(sc(SemanticOp::Intrinsic(0)), SurfaceClass::Absent);
    }

    #[test]
    #[should_panic(expected = "Plan-2 scope")]
    fn non_semantic_nodes_are_out_of_scope_loudly() {
        InterpreterSurfaceProbe
            .surface_class(&CoverageNode::Capability { arch: Architecture::Aie2, domain: "dma".into() });
    }
}
