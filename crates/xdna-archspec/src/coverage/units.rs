//! Behavioral units: the explicit override registry (spec Section 3) and the
//! single hand-curated CapabilityDomain spine (spec Section 6). Seeded coarse;
//! Phase 2 refines. Co-located on purpose (one location, spec Section 6).

use crate::coverage::verdict::Verdict;
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
    /// Set true with a reason when an override pulls a node off the
    /// toolchain-derived path (spec Section 3 no-silent-shadow rule).
    pub shadows_derived: Option<String>,
}

/// A top-level hardware capability the manual names (spec Section 6).
/// Each must be claimed by >= 1 behavioral unit per applicable arch, or the
/// build panics (enforced in Task 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
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
/// the AM020 ToC + aie-rt module tree; maintained ONLY here. Coarse,
/// arch-invariant domains; AIE2 is the only wired arch today (Plan 1).
pub fn capability_spine() -> Vec<CapabilityDomain> {
    let aie2 = vec![Architecture::Aie2];
    [
        "core",
        "program_memory",
        "data_memory",
        "dma",
        "locks",
        "stream_switch",
        "events_trace",
        "performance_counters",
        "timer",
        "watchpoint",
        "debug_halt",
        "cascade",
        "interrupt",
        "noc",
        "shim_mux",
    ]
    .iter()
    .map(|id| CapabilityDomain { id: (*id).to_string(), arches: aie2.clone() })
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
    }

    #[test]
    fn capability_domain_arch_applicability_is_explicit() {
        let dma = capability_spine().into_iter().find(|d| d.id == "dma").unwrap();
        assert!(dma.applies_to(Architecture::Aie2));
    }
}
