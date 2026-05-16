//! Behavioral-provenance verdict: where knowledge comes from (Provenance)
//! and whether it is silicon-checked (Verification). See spec Section 1.

use serde::{Deserialize, Serialize};

/// Where the behavioral knowledge for a unit comes from. Spec Section 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provenance {
    /// Fully specified by aie-rt / TableGen / regdb. Toolchain is ground truth.
    ToolchainDerived,
    /// Reimplemented from reading aietools python models / AM020. Not silicon-checked.
    AietoolsModeled,
    /// From AM020/AM025 prose only.
    DocSpecified,
    /// A trusted source describes the node and we assert NO model.
    Unspecified,
}

/// Silicon-verification status, orthogonal to provenance. Spec Section 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verification {
    /// Toolchain-derived; silicon cannot be "more right" than the spec.
    NotApplicable,
    /// Checked against that arch's hardware. `evidence` points at the proof.
    Verified { evidence: String },
    /// Needs that arch's silicon to confirm; not done yet.
    Unverified,
    /// Explicitly signed off as "good enough, will not verify", with reasoning.
    Accepted { rationale: String },
}

/// One behavioral unit's verdict on both axes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub provenance: Provenance,
    pub verification: Verification,
}

impl Verdict {
    /// Perishable queue (spec Section 1): modeled, weak provenance, unverified.
    /// `Unspecified` is deliberately excluded -- it is the comprehension-gap tier.
    pub fn is_perishable(&self) -> bool {
        matches!(self.provenance, Provenance::AietoolsModeled | Provenance::DocSpecified)
            && matches!(self.verification, Verification::Unverified)
    }

    /// Comprehension gap (spec Section 1): a trusted source describes it and we
    /// assert no model. Closed only by understanding it or explicit Accepted.
    pub fn is_comprehension_gap(&self) -> bool {
        matches!(self.provenance, Provenance::Unspecified)
            && !matches!(self.verification, Verification::Accepted { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolchain_derived_is_never_a_gap() {
        let v = Verdict {
            provenance: Provenance::ToolchainDerived,
            verification: Verification::NotApplicable,
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn modeled_unverified_is_perishable_not_comprehension() {
        let v = Verdict { provenance: Provenance::AietoolsModeled, verification: Verification::Unverified };
        assert!(v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn unspecified_is_a_comprehension_gap_not_perishable() {
        let v = Verdict { provenance: Provenance::Unspecified, verification: Verification::Unverified };
        assert!(v.is_comprehension_gap());
        assert!(!v.is_perishable(), "Unspecified is its own tier, not the perishable queue (spec Section 1)");
    }

    #[test]
    fn accepted_closes_both_sets() {
        let v = Verdict {
            provenance: Provenance::Unspecified,
            verification: Verification::Accepted { rationale: "out of scope".into() },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn verified_closes_perishable() {
        let v = Verdict {
            provenance: Provenance::AietoolsModeled,
            verification: Verification::Verified { evidence: "bridge:add_one".into() },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }
}
