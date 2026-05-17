//! Behavioral units: the explicit override registry (spec Section 3) and the
//! single hand-curated CapabilityDomain spine (spec Section 6). Seeded coarse;
//! Phase 2 refines. Co-located on purpose (one location, spec Section 6).

use crate::coverage::verdict::{Completeness, Verdict, Verification};
use crate::coverage::CoverageNode;
use crate::types::Architecture;

/// What fine nodes a behavioral unit claims (spec Section 3). A unit either
/// claims explicit nodes (override) or is the derived bucket for a category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Claims {
    /// Explicit override: this exact set of nodes (spec Section 3).
    Nodes(Vec<CoverageNode>),
}

/// A behavioral unit: a cluster of fine nodes with one verdict (spec Section 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralUnit {
    pub id: String,
    pub arch: Architecture,
    pub claims: Claims,
    pub verdict: Verdict,
    /// Free-text human narrative for the Section-3 no-silent-shadow rule:
    /// when an override pulls a node off the toolchain-derived path, this
    /// records why for a human reader. NEVER substring-matched for
    /// enforcement -- see `shared_from` for the typed soundness gate.
    pub shadows_derived: Option<String>,
    /// Typed cross-arch provenance (spec Section 7). Some(other_arch) means
    /// this verdict was shared from other_arch. enforce_coverage panics on
    /// Verified + shared_from.is_some() -- verification never transfers
    /// across silicon. Typed, so phrasing can neither bypass nor trip it.
    pub shared_from: Option<Architecture>,
}

/// A top-level hardware capability the manual names (spec Section 6), now
/// carrying the retired index's per-subsystem detail as seeded data (spec
/// Section 2). Each must be claimed by >= 1 behavioral unit per applicable
/// arch, or the build panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
    /// Authoritative source (aie-rt path / AM025 / device model). Non-empty,
    /// enforced.
    pub source_ref: String,
    /// Our emulator src/ path(s). Non-empty unless an OOS/MISSING narrative is
    /// present (enforced).
    pub src_locations: Vec<String>,
    /// Human-readable notes and known gaps for this domain.
    pub narrative: String,
    /// The domain's own asserted coverage verdict (spec Section 1/2).
    pub verdict: Verdict,
    /// Documents a deliberate own-vs-rollup divergence (spec Section 3); a
    /// silent material drift is a hard failure, an annotated one is allowed.
    pub drift_rationale: Option<String>,
}

impl CapabilityDomain {
    pub fn applies_to(&self, arch: Architecture) -> bool {
        self.arches.contains(&arch)
    }
}

/// The explicit override registry, per arch. Empty in Phase 1 (spec Section 5):
/// coarse category defaults carry the bootstrap; Phase 2 adds entries here.
pub fn override_registry(_arch: Architecture) -> Vec<BehavioralUnit> {
    Vec::new()
}

/// The single hand-curated capability spine (spec Section 6). Seeded once from
/// the AM020 ToC + aie-rt module tree; maintained ONLY here. Deliberately
/// COARSER than `SubsystemKind` (spec Section 6: architectural domains, not a
/// 1:1 register-taxonomy mirror). Documented folds so a Phase-2 author does
/// not recreate a domain or think one is missing:
///   - `core` covers `SubsystemKind::Processor` (the VLIW compute core)
///   - `program_counter` covers `SubsystemKind::ProgramCounter` (PC sampling)
///   - `events_trace` covers `SubsystemKind::Trace` AND `::Event`
///   - `locks` covers `SubsystemKind::Lock` AND `::LockRequest`
///   - `debug_halt` covers `SubsystemKind::Debug`
///   - `control_packets` covers on-chip control-packet reassembly, packet
///     handler status, and the NPU host instruction stream (spec Appendix)
///   - `clock_control` covers module/column/tile clock + reset control
///   - `tile_isolation` covers Tile_Control isolation bits / N-S-E-W gates
///   - `binary_load` covers CDO/ELF/XCLBIN ingest, SS-routing reconstruction,
///     and array-topology construction (spec Appendix N1 resolutions)
/// `SubsystemKind::Unknown` is a classifier placeholder, NOT a hardware
/// capability -- intentionally excluded. Coarse, arch-invariant; AIE2 is the
/// only wired arch today (Plan 1). Phase 1 auto-claims every domain via the
/// derived shim (Task 6); the SubsystemKind<->spine partition is Phase 2.
pub fn capability_spine() -> Vec<CapabilityDomain> {
    let aie2 = vec![Architecture::Aie2];
    // Task-2 skeleton seed: every domain is honestly Partial pending its
    // curated seed (Tasks 3/4). source_ref/src_locations are non-empty so the
    // enforce loop passes; the Partial{missing} verdict is true (the catalogue
    // entry is genuinely incomplete) and self-corrects out of the
    // implementation-gaps queue as Tasks 3/4 replace it. An accurate coarse
    // bootstrap (the same honest-coarse pattern as Phase-1 category defaults).
    let pending = |missing: &str| Verdict {
        provenance: crate::coverage::verdict::Provenance::ToolchainDerived,
        verification: Verification::Modeled {
            completeness: Completeness::Partial { missing: missing.to_string() },
        },
    };
    crate::coverage::spine_ids::SPINE_DOMAIN_IDS
        .iter()
        .map(|id| CapabilityDomain {
            id: (*id).to_string(),
            arches: aie2.clone(),
            source_ref: "pending (plan Task 3/4)".to_string(),
            src_locations: vec!["src/pending (plan Task 3/4)".to_string()],
            narrative: format!("{id}: curated seed pending (plan Task 3/4)"),
            verdict: pending("curated domain seed pending (plan Task 3/4)"),
            drift_rationale: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Architecture;

    #[test]
    fn override_registry_is_empty_at_phase1() {
        // Phase 1 ships green on coarse category defaults; overrides are
        // Phase 2 refinement work (spec Section 5).
        assert!(override_registry(Architecture::Aie2).is_empty());
    }

    #[test]
    fn capability_spine_seeded_for_aie2() {
        let spine = capability_spine();
        assert!(spine.iter().any(|d| d.id == "dma"));
        assert!(spine.iter().any(|d| d.id == "locks"));
        assert!(spine.iter().any(|d| d.id == "stream_switch"));
        // Every seeded domain applies to AIE2 (the only wired arch today).
        assert!(spine.iter().all(|d| d.applies_to(Architecture::Aie2)));
        // Spine extended to 20 (spec Section 2 hybrid decision).
        assert_eq!(spine.len(), 20);
        for id in ["control_packets", "clock_control", "tile_isolation", "binary_load"] {
            assert!(spine.iter().any(|d| d.id == id), "missing new domain {id}");
        }
        // Every domain carries non-empty source_ref + src_locations.
        assert!(spine.iter().all(|d| !d.source_ref.is_empty()));
        assert!(spine.iter().all(|d| !d.src_locations.is_empty()));
    }

    #[test]
    fn capability_domain_arch_applicability_is_explicit() {
        let dma = capability_spine().into_iter().find(|d| d.id == "dma").unwrap();
        assert!(dma.applies_to(Architecture::Aie2));
    }

    #[test]
    fn capability_spine_matches_the_leaf_id_list() {
        // The leaf list (build.rs-visible) and the rich spine MUST agree --
        // they are the same source of truth, just two views (spec Section 6
        // one location; Plan 2 cycle note).
        use crate::coverage::spine_ids::SPINE_DOMAIN_IDS;
        let rich: Vec<String> = capability_spine().into_iter().map(|d| d.id).collect();
        let leaf: Vec<String> = SPINE_DOMAIN_IDS.iter().map(|s| s.to_string()).collect();
        assert_eq!(rich, leaf);
    }
}
