//! Interpreter-side coverage: the concrete Axis-1 probe and the two-axis
//! reconciliation test. archspec owns both axis DEFINITIONS (spec Section 2);
//! this crate contributes only the one trait impl plus the wiring test.

pub mod surface_probe;

#[cfg(test)]
mod reconciliation {
    use crate::interpreter::coverage::surface_probe::InterpreterSurfaceProbe;
    use xdna_archspec::aie2::isa::SemanticOp;
    use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
    use xdna_archspec::coverage::verdict::Verification;
    use xdna_archspec::coverage::{CoverageModel, CoverageNode};
    use xdna_archspec::types::Architecture;

    /// The SemanticOp universe under reconciliation. Mirrors archspec's
    /// private `all_semantic_ops()` (mod.rs:132-239). The Plan-1
    /// `all_semantic_ops_len_tripwire` test (count == 103) is the maintenance
    /// tripwire on the archspec side; this list is the interpreter-side peer.
    /// If they drift, the count assertion below fails loudly.
    fn semantic_universe() -> Vec<SemanticOp> {
        use SemanticOp::*;
        vec![
            Add,
            Sub,
            Adc,
            Sbc,
            Mul,
            SDiv,
            UDiv,
            SRem,
            URem,
            Abs,
            Neg,
            DivStep,
            Select,
            Ctlz,
            Cttz,
            Ctpop,
            Bswap,
            Clb,
            SignExtend,
            ZeroExtend,
            Truncate,
            Copy,
            Nop,
            Event,
            ReadCycleCounter,
            PointerAdd,
            PointerMov,
            And,
            Or,
            Xor,
            Not,
            Shl,
            Sra,
            Srl,
            AshlBidir,
            LshlBidir,
            Rotl,
            Rotr,
            SetEq,
            SetNe,
            SetLt,
            SetLe,
            SetGt,
            SetGe,
            SetUlt,
            SetUle,
            SetUgt,
            SetUge,
            Cmp,
            Load,
            Store,
            Br,
            BrCond,
            Call,
            Ret,
            Done,
            Halt,
            Mac,
            MatMul,
            MatMulSub,
            NegMatMul,
            AddMac,
            SubMac,
            NegMul,
            Srs,
            Ups,
            Shuffle,
            Pack,
            Unpack,
            Align,
            VectorBroadcast,
            VectorExtract,
            VectorInsert,
            VectorPush,
            VectorPushHi,
            VectorSelect,
            VectorClear,
            Convert,
            Min,
            Max,
            SubLt,
            SubGe,
            MaxDiffLt,
            MaxLt,
            MinGe,
            AbsGtz,
            NegGtz,
            NegLtz,
            NegAdd,
            Accumulate,
            AccumSub,
            AccumNegAdd,
            AccumNegSub,
            LockAcquire,
            LockRelease,
            CascadeRead,
            CascadeWrite,
            StreamRead,
            StreamWrite,
            StreamWritePacketHeader,
            DmaStart,
            DmaWait,
            Intrinsic(0),
        ]
    }

    #[test]
    fn universe_matches_archspec_tripwire_count() {
        // Same 103 the archspec-side tripwire pins; drift here means the two
        // hand-maintained peer lists fell out of sync -- fix both together.
        assert_eq!(semantic_universe().len(), 103);
    }

    #[test]
    fn no_verified_unit_is_entirely_absent() {
        // Spec Section 2 item 1 / Section 4 test-red condition: a verdict
        // claiming silicon-Verified for an op the emulator does not even
        // surface is an incoherent state. At Phase-1 bootstrap the override
        // registry is empty so nothing is Verified yet -- this asserts the
        // invariant holds now and keeps holding as Phase-2 adds overrides.
        let m = CoverageModel::build(Architecture::Aie2);
        let probe = InterpreterSurfaceProbe;
        for op in semantic_universe() {
            let v = m.semantic_verdict(&op);
            if matches!(v.verification, Verification::Verified { .. }) {
                let node = CoverageNode::Semantic { arch: Architecture::Aie2, op };
                assert_ne!(
                    probe.surface_class(&node),
                    SurfaceClass::Absent,
                    "{:?} is Verified but its surface is Absent -- a Verified \
                     verdict for an op the interpreter does not surface is a \
                     contradiction (spec Section 2/4)",
                    node
                );
            }
        }
    }

    #[test]
    fn every_semantic_node_has_a_definite_surface_class() {
        // The probe is total by construction (no `_` arm); this asserts the
        // wired model agrees -- every node in the reconciled universe gets a
        // definite class and the probe never panics on the in-scope set.
        let probe = InterpreterSurfaceProbe;
        for op in semantic_universe() {
            let node = CoverageNode::Semantic { arch: Architecture::Aie2, op };
            let _ = probe.surface_class(&node); // must not panic
        }
    }
}
