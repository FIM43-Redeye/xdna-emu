//! Two-axis coverage provenance infrastructure. Spec:
//! docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod artifacts;
pub mod build_gate;
pub mod derive;
pub mod enforce;
pub mod spine_ids;
pub mod subsystem;
pub mod surface;
pub mod units;
pub mod verdict;

use crate::aie2::isa::SemanticOp;
use crate::coverage::derive::{category, default_verdict, Category};
use crate::coverage::units::{capability_spine, override_registry, BehavioralUnit};
use crate::coverage::verdict::Verdict;
use crate::types::{Architecture, ModuleKind, SubsystemKind, TileKind};
use serde::{Deserialize, Serialize};

/// A fine node, architecture-qualified (spec Section 7: all identity is
/// arch-qualified). Two kinds of node universe today: ISA semantics and
/// registers. Capability-spine domains also get a CoverageNode so they can be
/// claimed by units uniformly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageNode {
    Semantic {
        arch: Architecture,
        op: SemanticOp,
    },
    /// Intentionally finer-grained than the graph-level `types::NodeId::Register`:
    /// it carries `subsystem` for coverage disambiguation, with a conversion
    /// expected in a later task rather than zero-cost aliasing.
    Register {
        arch: Architecture,
        tile: TileKind,
        module: ModuleKind,
        subsystem: SubsystemKind,
        name: String,
    },
    Capability {
        arch: Architecture,
        domain: String,
    },
}

impl CoverageNode {
    pub fn arch(&self) -> Architecture {
        match self {
            CoverageNode::Semantic { arch, .. }
            | CoverageNode::Register { arch, .. }
            | CoverageNode::Capability { arch, .. } => *arch,
        }
    }
}

/// Per-arch coverage model (spec Section 7: a per-ArchModel facet, keyed by
/// arch, built from the arch's node universe). Built-from / keyed-by satisfies
/// the spec intent without forcing a non-serializable field into ArchModel.
#[derive(Debug, Clone)]
pub struct CoverageModel {
    pub arch: Architecture,
    overrides: Vec<BehavioralUnit>,
}

impl CoverageModel {
    /// Build the coverage model for one architecture. Phase 1: overrides are
    /// empty, so verdicts come from coarse category/subsystem defaults.
    pub fn build(arch: Architecture) -> Self {
        Self { arch, overrides: override_registry(arch) }
    }

    /// Verdict for a SemanticOp node: override if one claims it, else the
    /// pessimistic category default (spec Section 3/5).
    pub fn semantic_verdict(&self, op: &SemanticOp) -> Verdict {
        let node = CoverageNode::Semantic { arch: self.arch, op: *op };
        if let Some(u) = self.claiming_unit(&node) {
            return u.verdict.clone();
        }
        default_verdict(category(op))
    }

    /// Worst-wins verdict over every SemanticOp in `cat` (spec Section 3
    /// lattice). The per-category architecture-index row uses this: once a
    /// category is heterogeneous -- a per-op override Verifies some but not all
    /// of its ops (e.g. DMA-ops Verified while stream/cascade are not) -- the
    /// honest category-level verdict is the weakest of its ops, never a lucky
    /// representative.
    pub fn category_verdict(&self, cat: Category) -> Verdict {
        all_semantic_ops()
            .iter()
            .filter(|op| category(op) == cat)
            .map(|op| self.semantic_verdict(op))
            .min_by_key(|v| crate::coverage::subsystem::coverage_rank(v))
            .expect("every category has at least one SemanticOp")
    }

    fn claiming_unit(&self, node: &CoverageNode) -> Option<&BehavioralUnit> {
        self.overrides.iter().find(|u| match &u.claims {
            crate::coverage::units::Claims::Nodes(ns) => ns.contains(node),
        })
    }

    /// All SemanticOp verdicts (the Plan 1 node universe for the gate). Plan 2
    /// adds register/surface joining via the reconciliation test.
    fn all_semantic_verdicts(&self) -> Vec<Verdict> {
        all_semantic_ops().iter().map(|op| self.semantic_verdict(op)).collect()
    }

    /// Perishable queue (spec Section 1): modeled, unverified.
    pub fn perishable_queue(&self) -> Vec<Verdict> {
        self.all_semantic_verdicts().into_iter().filter(|v| v.is_perishable()).collect()
    }

    /// Comprehension gaps (spec Section 1): a source describes it, no model.
    pub fn comprehension_gaps(&self) -> Vec<Verdict> {
        self.all_semantic_verdicts()
            .into_iter()
            .filter(|v| v.is_comprehension_gap())
            .collect()
    }

    /// Per-silicon release gate (spec Section 4/7): both sets empty modulo
    /// Accepted. Green for AIE2 == "safe to retire NPU1".
    pub fn clean_release(&self) -> bool {
        self.perishable_queue().is_empty() && self.comprehension_gaps().is_empty()
    }

    /// The capability spine for this arch (spec Section 6).
    pub fn applicable_capabilities(&self) -> Vec<String> {
        capability_spine()
            .into_iter()
            .filter(|d| d.applies_to(self.arch))
            .map(|d| d.id)
            .collect()
    }
}

/// The static SemanticOp universe. One representative of every enum variant;
/// `Intrinsic` is represented by `Intrinsic(0)` (the table-lookup family).
///
/// NOT compile-time guarded for completeness. `category()` (Task 3) forces a
/// new SemanticOp variant to be *categorized* (exhaustive match, no wildcard)
/// but does NOT force it into this hand-maintained list -- a categorized-but-
/// unlisted variant would silently under-report in the rollups. The
/// authoritative completeness cross-check is Plan 2's reconciliation against
/// the real TableGen-decoded instruction set. The `len()` assertion in the
/// test module is a maintenance tripwire (reminds a maintainer to update this
/// list), NOT a proof of completeness. Do not claim a guard that is not here.
pub(crate) fn all_semantic_ops() -> Vec<SemanticOp> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verification};

    #[test]
    fn semantic_rollup_uses_pessimistic_category_defaults() {
        let m = CoverageModel::build(Architecture::Aie2);
        // Vector ops roll up to AietoolsModeled/Unverified -> perishable.
        let vmac = m.semantic_verdict(&SemanticOp::Mac);
        assert_eq!(vmac.provenance, Provenance::AietoolsModeled);
        assert!(vmac.is_perishable());
        // Scalar add is toolchain ground truth -> in neither set.
        let add = m.semantic_verdict(&SemanticOp::Add);
        assert_eq!(add.verification, Verification::NotApplicable);
        assert!(!add.is_perishable() && !add.is_comprehension_gap());
        // Intrinsic catch-all is Accepted (#104): never constructed, fail-loud,
        // and the concrete ops it resolves to are differentially verified (#103).
        let intr = m.semantic_verdict(&SemanticOp::Intrinsic(0));
        assert!(!intr.is_comprehension_gap(), "Intrinsic must be Accepted, not a gap");
        assert!(matches!(intr.verification, Verification::Accepted { .. }));
    }

    #[test]
    fn category_verdict_is_worst_wins_over_ops() {
        use crate::coverage::derive::Category;
        let m = CoverageModel::build(Architecture::Aie2);
        // SideEffect is heterogeneous now (DMA Verified, stream/cascade not);
        // the category verdict must be the worst-wins floor (Unverified), never
        // the DMA Verified rep -- the per-category architecture-index row must
        // not over-claim.
        let se = m.category_verdict(Category::SideEffect);
        assert!(
            matches!(se.verification, Verification::Unverified),
            "SideEffect category floor stays Unverified while stream/cascade ops are unverified"
        );
        // A homogeneous category is unaffected.
        let arith = m.category_verdict(Category::Arithmetic);
        assert_eq!(arith.verification, Verification::NotApplicable);
    }

    #[test]
    fn dma_ops_verified_but_other_side_effects_still_perishable() {
        // #113 axis-2: the framework's DMA tenant silicon-verified DmaStart +
        // DmaWait (81/81 access-pattern keys vs NPU1, 0 divergent). Those two
        // leave the perishable queue; the rest of the SideEffect category
        // (cascade + core-side stream, tenants 4/5) stays honestly perishable.
        let m = CoverageModel::build(Architecture::Aie2);
        for op in [SemanticOp::DmaStart, SemanticOp::DmaWait] {
            let v = m.semantic_verdict(&op);
            assert!(matches!(v.verification, Verification::Verified { .. }), "{op:?} must be Verified");
            assert!(!v.is_perishable(), "{op:?} must leave the perishable queue");
        }
        for op in [
            SemanticOp::CascadeRead,
            SemanticOp::CascadeWrite,
            SemanticOp::StreamRead,
            SemanticOp::StreamWrite,
            SemanticOp::StreamWritePacketHeader,
        ] {
            assert!(
                m.semantic_verdict(&op).is_perishable(),
                "{op:?} is a tenant-4/5 op, not claimed -- must stay perishable"
            );
        }
    }

    #[test]
    fn clean_release_is_false_via_perishable_not_comprehension() {
        let m = CoverageModel::build(Architecture::Aie2);
        // The intrinsic comprehension gap is closed by Accept (#104), so the
        // comprehension set is empty. But the perishable queue (vector ops,
        // AietoolsModeled/Unverified) keeps the gate honestly red until the
        // silicon Verified flip (Half B) -- Accept does NOT falsely green it.
        assert!(!m.perishable_queue().is_empty(), "vector ops perishable until silicon verification");
        assert!(m.comprehension_gaps().is_empty(), "intrinsic gap Accepted (#104) -- none remain");
        assert!(!m.clean_release(), "gate stays red via the perishable queue (spec S5)");
    }

    #[test]
    fn clean_release_arch_field_is_preserved() {
        // True per-arch isolation (two arches' models being independent) is
        // only testable once a second arch is wired -- TODO then. For now
        // this asserts the model carries the arch it was built for.
        let m = CoverageModel::build(Architecture::Aie2);
        assert_eq!(m.arch, Architecture::Aie2);
    }

    #[test]
    fn clean_release_aie2() {
        // THE per-silicon release gate (spec Section 4/7). Discoverable via
        // `cargo test -p xdna-archspec --lib clean_release`. Green for AIE2 ==
        // "safe to retire NPU1". At bootstrap it is correctly NOT green
        // (vector perishable, Intrinsic a comprehension gap) -- assert the
        // honest negative; Phase 2 closes these and flips this assertion.
        let m = CoverageModel::build(Architecture::Aie2);
        assert!(!m.clean_release(), "bootstrap must not be green -- that is the honest state (spec S5)");
    }

    #[test]
    fn all_semantic_ops_len_tripwire() {
        // MAINTENANCE TRIPWIRE, not a completeness proof. category() forces a
        // new SemanticOp variant to be categorized but NOT added to
        // all_semantic_ops(); this assertion fails loudly if the list length
        // changes, prompting a maintainer to confirm the variant was added
        // here too. Authoritative completeness is Plan 2's reconciliation
        // against the real decoded instruction set. Update the count below
        // ONLY together with a deliberate all_semantic_ops() change.
        assert_eq!(
            all_semantic_ops().len(),
            103,
            "all_semantic_ops() length changed -- confirm every SemanticOp \
             variant is still represented (see the doc comment on the fn)"
        );
    }
}
