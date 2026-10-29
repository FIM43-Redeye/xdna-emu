//! Two-axis coverage provenance infrastructure. Spec:
//! xdna-emu-priv: research/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod artifacts;
pub mod build_gate;
pub mod constants;
pub mod derive;
pub mod enforce;
pub mod spine_ids;
pub mod subsystem;
pub mod surface;
pub mod units;
pub mod variant;
pub mod verdict;

use crate::aie2::isa::SemanticOp;
use crate::coverage::derive::{category, default_verdict, Category};
use crate::coverage::subsystem::coverage_rank;
use crate::coverage::units::{capability_spine, override_registry, BehavioralUnit};
use crate::coverage::variant::{variants_of, OpVariant};
use crate::coverage::verdict::Verdict;
use crate::types::{Architecture, ModuleKind, SubsystemKind, TileKind};
use serde::{Deserialize, Serialize};

/// A fine node, architecture-qualified (spec Section 7: all identity is
/// arch-qualified). Two kinds of node universe today: ISA semantics and
/// registers. Capability-spine domains also get a CoverageNode so they can be
/// claimed by units uniformly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageNode {
    /// One variant of one op (`coverage::variant`): the unit evidence verifies.
    Semantic {
        arch: Architecture,
        op: SemanticOp,
        variant: OpVariant,
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

/// One (op, variant) line in the semantic ledger (`CoverageModel::semantic_ledger`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticLedgerRow {
    pub op: SemanticOp,
    pub variant: OpVariant,
    pub category: Category,
    pub verdict: Verdict,
    /// Id of the override unit that set the verdict; `None` = category default.
    pub claimed_by: Option<String>,
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

    /// Verdict for one (op, variant) node: override if one claims it, else
    /// the pessimistic category default (spec Section 3/5).
    pub fn variant_verdict(&self, op: &SemanticOp, variant: &OpVariant) -> Verdict {
        let node = CoverageNode::Semantic { arch: self.arch, op: *op, variant: *variant };
        if let Some(u) = self.claiming_unit(&node) {
            return u.verdict.clone();
        }
        default_verdict(category(op))
    }

    /// Verdict for a whole SemanticOp: the worst over its variants (spec
    /// Section 3 lattice), so verifying one variant never vouches for another.
    pub fn semantic_verdict(&self, op: &SemanticOp) -> Verdict {
        variants_of(self.arch, *op)
            .iter()
            .map(|v| self.variant_verdict(op, v))
            .min_by_key(coverage_rank)
            .expect("every op has at least one variant")
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
            .min_by_key(coverage_rank)
            .expect("every category has at least one SemanticOp")
    }

    fn claiming_unit(&self, node: &CoverageNode) -> Option<&BehavioralUnit> {
        self.overrides.iter().find(|u| match &u.claims {
            crate::coverage::units::Claims::Nodes(ns) => ns.contains(node),
        })
    }

    /// Every (op, variant) node with its category, verdict and the override
    /// unit that set it (`None` = the category default): ops in
    /// `all_semantic_ops()` order, variants sorted within each. The view the
    /// rendered queues and the op ledger are built from.
    pub fn semantic_ledger(&self) -> Vec<SemanticLedgerRow> {
        all_semantic_ops()
            .into_iter()
            .flat_map(|op| variants_of(self.arch, op).into_iter().map(move |variant| (op, variant)))
            .map(|(op, variant)| {
                let node = CoverageNode::Semantic { arch: self.arch, op, variant };
                let claimed_by = self.claiming_unit(&node).map(|u| u.id.clone());
                SemanticLedgerRow {
                    category: category(&op),
                    verdict: self.variant_verdict(&op, &variant),
                    claimed_by,
                    op,
                    variant,
                }
            })
            .collect()
    }

    /// The override unit `id` (its verdict carries the evidence or rationale).
    pub fn unit(&self, id: &str) -> Option<&BehavioralUnit> {
        self.overrides.iter().find(|u| u.id == id)
    }

    /// Every (op, variant) node's verdict (the node universe for the gate).
    /// Plan 2 adds register/surface joining via the reconciliation test.
    fn all_semantic_verdicts(&self) -> Vec<Verdict> {
        self.semantic_ledger().into_iter().map(|r| r.verdict).collect()
    }

    /// Perishable queue (spec Section 1): modeled, unverified, one entry per
    /// (op, variant) node.
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

    /// Narrow semantic-provenance input to the NPU1 retirement gate: both
    /// semantic sets are empty modulo Accepted.
    pub fn semantic_provenance_clean(&self) -> bool {
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
        // An UNCLAIMED vector op rolls up to ModelInformed/Unverified ->
        // perishable. (Min is dead -- min/max lower to MaxLt/MinGe -- and not
        // claimed by the #126 vector override, so it keeps the category default;
        // Mac et al. are now Verified, so use Min as the perishable exemplar.)
        let vmin = m.semantic_verdict(&SemanticOp::Min);
        assert_eq!(vmin.provenance, Provenance::ModelInformed);
        assert!(vmin.is_perishable());
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
    fn op_verdict_is_the_worst_over_its_variants() {
        // Per-variant coverage (coverage::variant): an op is Verified only when
        // every variant is, so evidence for one variant never vouches for the
        // rest. Each op's verdict is exactly the worst of its variant verdicts.
        use crate::coverage::subsystem::coverage_rank;
        let m = CoverageModel::build(Architecture::Aie2);
        for op in all_semantic_ops() {
            let worst = variants_of(Architecture::Aie2, op)
                .iter()
                .map(|v| coverage_rank(&m.variant_verdict(&op, v)))
                .min()
                .unwrap();
            assert_eq!(coverage_rank(&m.semantic_verdict(&op)), worst, "{op:?}");
        }
        // Mac has unmodelled modes, so it is not Verified whatever evidence its
        // modelled modes carry.
        assert!(!matches!(m.semantic_verdict(&SemanticOp::Mac).verification, Verification::Verified { .. }));
        // Never-executed ops stay honestly perishable -- not claimed.
        use crate::aie2::isa::SemanticOp::*;
        for op in [Min, Max, VectorPush, SubLt] {
            assert!(m.semantic_verdict(&op).is_perishable(), "{op:?} is unclaimed -- must stay perishable");
        }
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
    fn semantic_provenance_clean_is_false_and_gaps_are_only_unmodelled_forms() {
        let m = CoverageModel::build(Architecture::Aie2);
        // The intrinsic comprehension gap is closed by Accept (#104). The
        // perishable queue (vector ops, ModelInformed/Unverified) keeps the
        // gate honestly red until silicon verifies them -- Accept does NOT
        // falsely green it.
        assert!(!m.perishable_queue().is_empty(), "vector ops perishable until silicon verification");
        // The comprehension set is exactly the unmodelled forms: MAC modes the
        // emulator has no geometry for, and variants it declines (gated).
        let gaps: Vec<_> = m
            .semantic_ledger()
            .into_iter()
            .filter(|r| r.verdict.is_comprehension_gap())
            .collect();
        assert!(!gaps.is_empty());
        for r in &gaps {
            assert!(
                matches!(
                    r.claimed_by.as_deref(),
                    Some("aie2.mac_modes.unmodelled" | "aie2.gated.not_modeled")
                ),
                "{:?} {} claimed by {:?}",
                r.op,
                r.variant,
                r.claimed_by
            );
        }
        assert!(!m.semantic_provenance_clean(), "gate stays red via the perishable queue (spec S5)");
    }

    #[test]
    fn semantic_provenance_clean_arch_field_is_preserved() {
        // True per-arch isolation (two arches' models being independent) is
        // only testable once a second arch is wired -- TODO then. For now
        // this asserts the model carries the arch it was built for.
        let m = CoverageModel::build(Architecture::Aie2);
        assert_eq!(m.arch, Architecture::Aie2);
    }

    #[test]
    fn semantic_provenance_clean_aie2() {
        // Discoverable via
        // `cargo test -p xdna-archspec --lib semantic_provenance_clean`.
        // At bootstrap it is correctly NOT green (vector perishable,
        // Intrinsic a comprehension gap) -- assert the honest negative;
        // later evidence work closes these and flips this assertion.
        let m = CoverageModel::build(Architecture::Aie2);
        assert!(
            !m.semantic_provenance_clean(),
            "bootstrap must not be green -- that is the honest state (spec S5)"
        );
    }

    #[test]
    fn semantic_ledger_rows_match_verdicts_and_claims() {
        // One row per (op, variant), ops in all_semantic_ops() order; the
        // verdict is the model's own; every non-default verdict names the unit
        // that set it.
        let m = CoverageModel::build(Architecture::Aie2);
        let ledger = m.semantic_ledger();
        let want: Vec<(SemanticOp, OpVariant)> = all_semantic_ops()
            .into_iter()
            .flat_map(|op| variants_of(Architecture::Aie2, op).into_iter().map(move |v| (op, v)))
            .collect();
        assert_eq!(ledger.iter().map(|r| (r.op, r.variant)).collect::<Vec<_>>(), want);
        for row in &ledger {
            assert_eq!(row.verdict, m.variant_verdict(&row.op, &row.variant));
            assert_eq!(row.category, category(&row.op));
            if row.claimed_by.is_none() {
                assert_eq!(
                    row.verdict,
                    default_verdict(category(&row.op)),
                    "{:?}: unclaimed => default",
                    row.op
                );
            }
        }
        let dma = ledger.iter().find(|r| r.op == SemanticOp::DmaStart).unwrap();
        assert_eq!(dma.claimed_by.as_deref(), Some("aie2.dma_ops.verified"));
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
