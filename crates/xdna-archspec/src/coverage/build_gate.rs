//! Dependency-light build-time gate (spec Section 4 delivery; the "irreducible
//! cycle" note in Plan 2). ZERO `crate::` imports on purpose: this module is
//! `#[path]`-included into build.rs, which cannot see `crate::aie2::isa`
//! (build.rs generates it). Operates only on plain domain-id strings; the deep
//! CoverageNode-dependent checks remain test-gated (permanent, by design).
//! Intent (Maya-approved): build-time enforcement guards against MISSING data
//! the emulator cannot be composed without -- absence aborts the build;
//! subtler structural problems are the test gate's job.

/// Phase-1 spine-existence gate. Panics (stops the build) if the capability
/// spine is empty -- the one Axis-2 invariant expressible without
/// `CoverageNode`. In Phase 1 every domain is auto-claimed by the derived
/// shim, so a non-empty well-formed spine IS the deliverable invariant; the
/// partition / cross-arch / shadow panics live in the test gate (the
/// CoverageNode types cannot cross into build.rs -- see the Plan-2 cycle
/// note). Phase 2 enriches the test gate, never this entry.
pub fn enforce_spine_phase1(domain_ids: &[&str]) {
    if domain_ids.is_empty() {
        panic!(
            "COVERAGE(build): capability spine is empty -- every build must \
             enumerate the hardware capability domains (spec Section 6). This \
             is the dependency-light build-time gate; deep partition/cross-arch \
             checks are test-gated (Plan 2 cycle note)."
        );
    }
    for (i, id) in domain_ids.iter().enumerate() {
        assert!(
            !id.trim().is_empty(),
            "COVERAGE(build): spine domain #{i} is blank -- ids must be \
             non-empty (spec Section 6)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_on_a_normal_spine() {
        enforce_spine_phase1(&["core", "dma", "locks"]); // must not panic
    }

    #[test]
    #[should_panic(expected = "capability spine is empty")]
    fn panics_on_empty_spine() {
        enforce_spine_phase1(&[]);
    }

    #[test]
    #[should_panic(expected = "is blank")]
    fn panics_on_blank_domain_id() {
        enforce_spine_phase1(&["core", "  ", "dma"]);
    }
}
