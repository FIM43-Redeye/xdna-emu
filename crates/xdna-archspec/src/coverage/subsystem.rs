//! Subsystem-axis weave (spec Section 3): the coarse category->domain link,
//! the worst-wins coverage-strength lattice, and the directional drift
//! cross-check. The link is single-sourced on the category side; the
//! domain->category direction is computed here, never stored.

use crate::coverage::derive::Category;
use crate::coverage::units::capability_spine;
use crate::coverage::verdict::{Completeness, Provenance, Verdict, Verification};

/// Coarse category->domain link (spec Section 2): the single source of truth,
/// stored only on the category side. 9 rows. Category-orphan domains
/// deliberately appear in no row (spec Section 2 -- no fabricated tags).
pub fn category_domains(cat: Category) -> &'static [&'static str] {
    match cat {
        Category::Arithmetic | Category::Bitwise | Category::Comparison | Category::ControlFlow => &["core"],
        Category::Memory => &["data_memory", "program_memory"],
        Category::Vector => &["core"],
        Category::Sync => &["locks"],
        Category::SideEffect => &["dma", "stream_switch", "cascade"],
        Category::NeedsTriage => &["core"],
    }
}

const ALL_CATEGORIES: [Category; 9] = [
    Category::Arithmetic,
    Category::Bitwise,
    Category::Comparison,
    Category::Memory,
    Category::ControlFlow,
    Category::Vector,
    Category::Sync,
    Category::SideEffect,
    Category::NeedsTriage,
];

/// Domains intentionally reachable from NO category (spec Section 2). Their
/// coverage is solely their own seeded verdict; no rollup, no drift check.
pub fn is_category_orphan(domain_id: &str) -> bool {
    !ALL_CATEGORIES.iter().any(|c| category_domains(*c).contains(&domain_id))
}

/// Inverse index, computed (never stored): categories tagged to `domain_id`.
pub fn tagged_categories(domain_id: &str) -> impl Iterator<Item = Category> + '_ {
    ALL_CATEGORIES
        .into_iter()
        .filter(move |c| category_domains(*c).contains(&domain_id))
}

/// Coverage-strength rank (spec Section 3 lattice). Higher == more covered.
/// Closed/terminal states tie at the top; provenance tie-breaks toward weaker.
pub fn coverage_rank(v: &Verdict) -> u32 {
    let base = match &v.verification {
        Verification::Verified { .. } | Verification::NotApplicable | Verification::Accepted { .. } => 50,
        Verification::Modeled { completeness: Completeness::Full } => 40,
        Verification::Modeled { completeness: Completeness::Partial { .. } } => 30,
        Verification::Modeled { completeness: Completeness::Stub } => 20,
        Verification::Unverified => 10,
    };
    let tiebreak = match v.provenance {
        Provenance::Unspecified => 0,
        Provenance::ToolchainDerived
        | Provenance::AietoolsModeled
        | Provenance::DocSpecified
        | Provenance::HardwareObserved => 1,
    };
    base + tiebreak
}

/// Per-op subsystem refinement (spec Section 3): which spine domain(s) an
/// individual SemanticOp belongs to. Defaults to the op's category domains, so
/// for a category that tags a single domain (or whose ops all share a verdict)
/// this is exactly `category_domains(category(op))`. It DIVERGES only where a
/// category spans multiple subsystems with now-distinct verdicts: SideEffect
/// splits into dma (DmaStart/DmaWait), the core-side stream engine
/// (StreamRead/Write/WritePacketHeader -> stream_switch), and cascade
/// (CascadeRead/Write). This refinement is what keeps a per-op Verified
/// override (e.g. the silicon-verified DMA ops, #113) from leaking onto a
/// sibling subsystem whose own ops are still unverified. By construction it is
/// always a subset of the op's category domains (test-gated below).
fn op_domains(op: &crate::aie2::isa::SemanticOp) -> &'static [&'static str] {
    use crate::aie2::isa::SemanticOp::*;
    use crate::coverage::derive::category;
    match op {
        DmaStart | DmaWait => &["dma"],
        StreamRead | StreamWrite | StreamWritePacketHeader => &["stream_switch"],
        CascadeRead | CascadeWrite => &["cascade"],
        other => category_domains(category(other)),
    }
}

/// Worst-wins rollup over the SemanticOps that actually belong to `domain_id`
/// (spec S3), via the per-op `op_domains` refinement. `None` == category-orphan
/// (no op maps; row shows `-`, drift exempt). Computing the floor over the
/// domain's OWN ops -- not a single category representative -- is what makes a
/// heterogeneous category (some ops Verified, some not) roll up honestly per
/// subsystem instead of crediting every tagged domain with a lucky rep's
/// verdict. Phase-1 constraint: AIE2 only (the sole wired arch). A future
/// multi-arch caller must thread an `Architecture` through here rather than
/// relying on this hardcoded `Aie2` (spec Section 8 phasing).
pub fn rolled_up_verdict(domain_id: &str) -> Option<Verdict> {
    use crate::coverage::{all_semantic_ops, CoverageModel};
    use crate::types::Architecture;
    let m = CoverageModel::build(Architecture::Aie2);
    all_semantic_ops()
        .iter()
        .filter(|op| op_domains(op).contains(&domain_id))
        .map(|op| m.semantic_verdict(op))
        .min_by_key(|verd| coverage_rank(verd))
}

/// Directional drift (spec Section 3): material ONLY when the domain's own
/// verdict ranks strictly higher (claims more coverage) than the rollup.
/// Pessimistic divergence is safe over-reporting and is NOT material.
pub fn drift_is_material(own: &Verdict, rolled_up: &Verdict) -> bool {
    coverage_rank(own) > coverage_rank(rolled_up)
}

/// True when a domain has a material optimistic drift AND no
/// `drift_rationale` documenting it (spec Section 3 hard-fail case).
/// Single-domain API (re-scans the spine per call). A batch caller should
/// iterate `capability_spine()` once and use `drift_is_material` directly
/// rather than looping this.
pub fn unannotated_material_drift(domain_id: &str) -> bool {
    let Some(rolled) = rolled_up_verdict(domain_id) else {
        return false;
    };
    let Some(dom) = capability_spine().into_iter().find(|d| d.id == domain_id) else {
        return false;
    };
    drift_is_material(&dom.verdict, &rolled) && dom.drift_rationale.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::derive::Category;
    use crate::coverage::verdict::{Completeness, Provenance, Verdict, Verification};

    fn v(p: Provenance, ver: Verification) -> Verdict {
        Verdict { provenance: p, verification: ver }
    }

    #[test]
    fn lattice_total_order_is_pinned() {
        let verified = v(Provenance::AietoolsModeled, Verification::Verified { evidence: "e".into() });
        let na = v(Provenance::ToolchainDerived, Verification::NotApplicable);
        let accepted = v(Provenance::DocSpecified, Verification::Accepted { rationale: "r".into() });
        let full =
            v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Full });
        let partial = v(
            Provenance::ToolchainDerived,
            Verification::Modeled { completeness: Completeness::Partial { missing: "m".into() } },
        );
        let stub =
            v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Stub });
        let unver = v(Provenance::DocSpecified, Verification::Unverified);
        let unspec = v(Provenance::Unspecified, Verification::Unverified);

        assert_eq!(coverage_rank(&verified), coverage_rank(&na));
        assert_eq!(coverage_rank(&na), coverage_rank(&accepted));
        assert!(coverage_rank(&accepted) > coverage_rank(&full));
        assert!(coverage_rank(&full) > coverage_rank(&partial));
        assert!(coverage_rank(&partial) > coverage_rank(&stub));
        assert!(coverage_rank(&stub) > coverage_rank(&unver));
        assert!(coverage_rank(&unver) > coverage_rank(&unspec));

        // Full monotone chain: first three tie at the closed-state top, the
        // rest strictly decrease. Pins the WHOLE order, not just pairwise.
        let ordered = [&verified, &na, &accepted, &full, &partial, &stub, &unver, &unspec];
        let ranks: Vec<u32> = ordered.iter().map(|v| coverage_rank(v)).collect();
        assert_eq!(ranks[0], ranks[1]);
        assert_eq!(ranks[1], ranks[2]);
        for w in ranks[2..].windows(2) {
            assert!(w[0] > w[1], "lattice not strictly decreasing past the closed tie: {ranks:?}");
        }
    }

    #[test]
    fn every_category_tags_at_least_one_domain() {
        for cat in [
            Category::Arithmetic,
            Category::Bitwise,
            Category::Comparison,
            Category::Memory,
            Category::ControlFlow,
            Category::Vector,
            Category::Sync,
            Category::SideEffect,
            Category::NeedsTriage,
        ] {
            assert!(!category_domains(cat).is_empty(), "category {cat:?} tags no domain");
        }
    }

    #[test]
    fn every_domain_is_tagged_or_explicitly_orphan() {
        use crate::coverage::units::capability_spine;
        for d in capability_spine() {
            let tagged = tagged_categories(&d.id).next().is_some();
            assert!(
                tagged || is_category_orphan(&d.id),
                "domain '{}' is neither category-tagged nor an explicit orphan \
                 (no fabricated tags allowed -- spec Section 2)",
                d.id
            );
        }
    }

    #[test]
    fn drift_is_directional_optimistic_only() {
        let optimistic = drift_is_material(
            &v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Full }),
            &v(Provenance::DocSpecified, Verification::Unverified),
        );
        assert!(optimistic, "optimistic over-claim must be material");
        let pessimistic = drift_is_material(
            &v(Provenance::DocSpecified, Verification::Unverified),
            &v(Provenance::ToolchainDerived, Verification::NotApplicable),
        );
        assert!(!pessimistic, "pessimistic divergence is safe over-reporting");
        assert!(!drift_is_material(
            &v(Provenance::ToolchainDerived, Verification::NotApplicable),
            &v(Provenance::AietoolsModeled, Verification::Verified { evidence: "e".into() }),
        ));
    }

    #[test]
    fn sideeffect_rollup_is_per_subsystem_not_category_wide() {
        // #113: DmaStart/DmaWait are silicon-Verified but the sibling SideEffect
        // ops (core-side stream + cascade) are not. The per-subsystem rollup
        // must credit ONLY dma -- never leak the DMA Verified verdict onto
        // stream_switch/cascade, which is exactly the over-claim a single
        // category-rep rollup produced once the category went heterogeneous.
        let dma = rolled_up_verdict("dma").expect("dma rolls up");
        assert!(
            matches!(dma.verification, Verification::Verified { .. }),
            "dma rolls up Verified from its own ops (DmaStart/DmaWait)"
        );
        for sib in ["stream_switch", "cascade"] {
            let v = rolled_up_verdict(sib).expect("rolls up");
            assert!(
                matches!(v.verification, Verification::Unverified),
                "{sib} must stay Unverified -- the DMA verdict must not leak across the SideEffect category"
            );
        }
    }

    #[test]
    fn op_domains_refines_but_never_escapes_category_domains() {
        // The per-op subsystem refinement sharpens the category->domain link; it
        // must always be a subset of the op's category domains.
        use crate::coverage::all_semantic_ops;
        use crate::coverage::derive::category;
        for op in all_semantic_ops() {
            let cat_doms = category_domains(category(&op));
            for d in op_domains(&op) {
                assert!(
                    cat_doms.contains(d),
                    "op {op:?} maps to domain {d} which is not in its category domains {cat_doms:?}"
                );
            }
        }
    }

    #[test]
    fn no_subsystem_drifts_silently() {
        use crate::coverage::units::capability_spine;
        for d in capability_spine() {
            assert!(
                !unannotated_material_drift(&d.id),
                "domain '{}' claims more coverage than its tagged categories \
                 justify with no drift_rationale (spec Section 3)",
                d.id
            );
        }
    }
}
