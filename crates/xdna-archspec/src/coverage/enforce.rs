//! Build-time Axis-2 enforcement (spec Section 2/4). Panics like
//! Confirmed<T>::confirm: an incomplete coverage model stops the build.

use crate::coverage::units::{capability_spine, BehavioralUnit, CapabilityDomain, Claims};
use crate::coverage::verdict::Verification;
use crate::coverage::CoverageNode;
use crate::types::Architecture;
use std::collections::HashMap;

/// Build-time Axis-2 enforcement for one arch (spec Section 2/4). Panics on:
/// a capability applicable to `arch` with no claiming unit; two units claiming
/// one node; a `Verified` verdict whose unit carries a typed `shared_from`
/// (verification never transfers across silicon, spec Section 7).
///
/// The SemanticOp axis needs no check here -- `category()` is compiler-enforced
/// (Task 3). Register-node partition is checked once the register override
/// registry is non-empty (Phase 2); in Phase 1 it is trivially satisfied
/// because every register rolls up to its subsystem's coarse default.
pub fn enforce_coverage(arch: Architecture, spine: &[CapabilityDomain], units: &[BehavioralUnit]) {
    // 1. Cross-arch transfer rule: verification never transfers (spec S7).
    // Keyed on the TYPED `shared_from` field -- never on prose.
    for u in units {
        if matches!(u.verdict.verification, Verification::Verified { .. }) {
            if let Some(src) = u.shared_from {
                panic!(
                    "COVERAGE: unit '{}' ({arch}) is Verified but shared_from={src:?} \
                     -- verification never transfers across silicon (spec Section 7)",
                    u.id
                );
            }
        }
    }

    // 2. Partition: no node double-claimed by two units (spec Section 3).
    // CoverageNode derives Hash; map node -> first-claiming unit id so the
    // panic names BOTH conflicting units.
    let mut seen: HashMap<&CoverageNode, &str> = HashMap::new();
    for u in units {
        let Claims::Nodes(ns) = &u.claims;
        for n in ns {
            if let Some(first) = seen.insert(n, u.id.as_str()) {
                panic!(
                    "COVERAGE: node {:?} double-claimed by units '{}' and '{}' ({arch}) \
                     -- the registry must partition, not merely cover (spec Section 3)",
                    n, first, u.id
                );
            }
        }
    }

    // 3. Capability spine: every applicable domain claimed by >= 1 unit (S6).
    for dom in spine {
        if !dom.applies_to(arch) {
            continue;
        }
        let claimed = units.iter().any(|u| {
            let Claims::Nodes(ns) = &u.claims;
            ns.iter()
                .any(|n| matches!(n, CoverageNode::Capability { domain, .. } if domain == &dom.id))
        });
        if !claimed {
            panic!(
                "COVERAGE: unclaimed capability '{}' for {arch} -- every hardware capability \
                 the manual names must be modeled by >= 1 behavioral unit (spec Section 6)",
                dom.id
            );
        }
    }
}

/// Phase-1 entry point called from build.rs. Every spine domain is claimed by
/// an implicit derived unit (spec Section 5: coarse defaults carry bootstrap).
/// This keeps the build green and honest while the override registry is empty;
/// Phase 2 replaces derived claims with real override units one at a time.
pub fn enforce_coverage_phase1(arch: Architecture) {
    let spine = capability_spine();
    let derived_units: Vec<BehavioralUnit> = spine
        .iter()
        .filter(|d| d.applies_to(arch))
        .map(|d| BehavioralUnit {
            id: format!("derived.{}", d.id),
            arch,
            claims: Claims::Nodes(vec![CoverageNode::Capability { arch, domain: d.id.clone() }]),
            // Coarse Phase-1 default; refined in Phase 2 (spec Section 5).
            verdict: crate::coverage::derive::default_verdict(crate::coverage::derive::Category::NeedsTriage),
            shadows_derived: None,
            shared_from: None,
        })
        .collect();
    enforce_coverage(arch, &spine, &derived_units);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::units::{BehavioralUnit, CapabilityDomain, Claims};
    use crate::coverage::verdict::{Provenance, Verdict, Verification};
    use crate::coverage::CoverageNode;
    use crate::types::Architecture;

    fn ok_unit(arch: Architecture, id: &str) -> BehavioralUnit {
        BehavioralUnit {
            id: id.into(),
            arch,
            claims: Claims::Nodes(vec![CoverageNode::Capability { arch, domain: id.into() }]),
            verdict: Verdict {
                provenance: Provenance::ToolchainDerived,
                verification: Verification::NotApplicable,
            },
            shadows_derived: None,
            shared_from: None,
        }
    }

    #[test]
    fn passes_when_every_capability_is_claimed() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        let units = vec![ok_unit(arch, "dma")];
        enforce_coverage(arch, &spine, &units); // must not panic
    }

    #[test]
    #[should_panic(expected = "unclaimed capability")]
    fn panics_on_unclaimed_capability() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "locks".into(), arches: vec![arch] }];
        enforce_coverage(arch, &spine, &[]); // nothing claims "locks"
    }

    #[test]
    #[should_panic(expected = "never transfers across silicon")]
    fn panics_on_verified_with_cross_arch_shadow() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        let mut u = ok_unit(arch, "dma");
        u.verdict.verification = Verification::Verified { evidence: "npu4".into() };
        u.shared_from = Some(Architecture::Aie2p); // typed marker, not prose
        enforce_coverage(arch, &spine, &[u]);
    }

    #[test]
    #[should_panic(expected = "double-claimed")]
    fn panics_on_two_units_claiming_one_node() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        enforce_coverage(arch, &spine, &[ok_unit(arch, "dma"), ok_unit(arch, "dma")]);
    }

    #[test]
    fn phase1_entry_point_is_green() {
        // Coarse derived claims satisfy the spine without override entries.
        enforce_coverage_phase1(Architecture::Aie2); // must not panic
    }
}
