//! VLIW slot parallelism and structural hazard detection.
//!
//! AIE2 VLIW bundles can contain up to 7 slots that execute in parallel.
//! However, not all slot combinations are truly parallel - some share
//! resources and must be serialized.
//!
//! # Execution Resources (AM020 Ch4)
//!
//! - **Scalar ALUs**: Two scalar slots share some ALU resources
//! - **Vector Unit**: Single vector execution unit
//! - **Load Ports**: Two independent load ports (A and B)
//! - **Store Port**: Single store port
//! - **AGU**: Address generation unit (shared for post-modify)
//!
//! # Structural Hazards
//!
//! When two slots require the same resource, a structural hazard occurs.
//! The hardware serializes the operations, adding stall cycles.
//!
//! Example: Two stores in the same bundle (if that were possible) would
//! conflict on the store port.

use crate::interpreter::bundle::{Operation, SlotIndex, SlotOp, VliwBundle};

/// Execution resources that can cause structural hazards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionResource {
    /// Scalar ALU (shared by Scalar0/Scalar1 for multiply).
    ScalarAlu,
    /// Vector execution unit.
    VectorUnit,
    /// Load port A (for LDA slot).
    LoadPortA,
    /// Load port B (for LDB slot).
    LoadPortB,
    /// Store port.
    StorePort,
    /// Address generation unit (for post-modify).
    AddressGen,
    /// Accumulator write port.
    AccumulatorWrite,
}

impl ExecutionResource {
    /// Get the serialization penalty for this resource (cycles).
    pub fn serialization_penalty(&self) -> u8 {
        match self {
            // Most conflicts just add 1 cycle
            ExecutionResource::ScalarAlu => 1,
            ExecutionResource::VectorUnit => 1,
            ExecutionResource::LoadPortA => 1,
            ExecutionResource::LoadPortB => 1,
            ExecutionResource::StorePort => 1,
            ExecutionResource::AddressGen => 1,
            ExecutionResource::AccumulatorWrite => 1,
        }
    }

    /// Get a static string name for this resource.
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionResource::ScalarAlu => "ScalarAlu",
            ExecutionResource::VectorUnit => "VectorUnit",
            ExecutionResource::LoadPortA => "LoadPortA",
            ExecutionResource::LoadPortB => "LoadPortB",
            ExecutionResource::StorePort => "StorePort",
            ExecutionResource::AddressGen => "AddressGen",
            ExecutionResource::AccumulatorWrite => "AccumulatorWrite",
        }
    }
}

/// Resources required by a slot operation.
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// Resources needed for this operation.
    pub resources: Vec<ExecutionResource>,
}

impl ResourceRequirements {
    /// Create empty requirements.
    pub fn new() -> Self {
        Self { resources: Vec::new() }
    }

    /// Add a resource requirement.
    pub fn require(&mut self, resource: ExecutionResource) {
        if !self.resources.contains(&resource) {
            self.resources.push(resource);
        }
    }

    /// Check if this requirement conflicts with another.
    pub fn conflicts_with(&self, other: &ResourceRequirements) -> Option<ExecutionResource> {
        for r in &self.resources {
            if other.resources.contains(r) {
                return Some(*r);
            }
        }
        None
    }
}

/// Get resource requirements for a slot operation.
pub fn get_requirements(op: &SlotOp) -> ResourceRequirements {
    let mut reqs = ResourceRequirements::new();

    // Add requirements based on slot
    match op.slot {
        SlotIndex::Scalar0 | SlotIndex::Scalar1 => {
            // Scalar multiply uses a shared resource
            if matches!(op.op, Operation::ScalarMul) {
                reqs.require(ExecutionResource::ScalarAlu);
            }
        }
        SlotIndex::Vector => {
            reqs.require(ExecutionResource::VectorUnit);
        }
        SlotIndex::Accumulator => {
            // Accumulator operations use the vector unit and accumulator port
            reqs.require(ExecutionResource::VectorUnit);
            reqs.require(ExecutionResource::AccumulatorWrite);
        }
        SlotIndex::Load => {
            reqs.require(ExecutionResource::LoadPortA);
        }
        SlotIndex::Store => {
            reqs.require(ExecutionResource::StorePort);
        }
        SlotIndex::Control => {
            // No shared resources
        }
    }

    // Post-modify addressing uses AGU
    match &op.op {
        Operation::Load { post_modify, .. } | Operation::Store { post_modify, .. } => {
            use crate::interpreter::bundle::PostModify;
            if !matches!(post_modify, PostModify::None) {
                reqs.require(ExecutionResource::AddressGen);
            }
        }
        _ => {}
    }

    // Vector MAC uses accumulator write port
    if matches!(op.op, Operation::VectorMac { .. }) {
        reqs.require(ExecutionResource::AccumulatorWrite);
    }

    reqs
}

/// Check if two slots can execute in parallel.
///
/// Returns `None` if compatible, or `Some(resource)` if they conflict.
pub fn check_slot_conflict(slot_a: &SlotOp, slot_b: &SlotOp) -> Option<ExecutionResource> {
    let reqs_a = get_requirements(slot_a);
    let reqs_b = get_requirements(slot_b);
    reqs_a.conflicts_with(&reqs_b)
}

/// Structural hazard information.
#[derive(Debug, Clone)]
pub struct StructuralHazard {
    /// Conflicting resource.
    pub resource: ExecutionResource,
    /// Slots involved.
    pub slots: (SlotIndex, SlotIndex),
    /// Serialization penalty.
    pub penalty_cycles: u8,
}

/// Check a bundle for structural hazards.
///
/// Returns a list of detected conflicts. The total penalty is the
/// maximum of all conflict penalties (since they occur in parallel).
pub fn check_bundle_conflicts(bundle: &VliwBundle) -> Vec<StructuralHazard> {
    let mut hazards = Vec::new();
    let active: Vec<&SlotOp> = bundle.active_slots().collect();

    // Check all pairs
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            if let Some(resource) = check_slot_conflict(active[i], active[j]) {
                hazards.push(StructuralHazard {
                    resource,
                    slots: (active[i].slot, active[j].slot),
                    penalty_cycles: resource.serialization_penalty(),
                });
            }
        }
    }

    hazards
}

/// Calculate the structural hazard penalty for a bundle.
///
/// Returns the additional cycles due to resource conflicts.
pub fn bundle_structural_penalty(bundle: &VliwBundle) -> u8 {
    let hazards = check_bundle_conflicts(bundle);
    hazards.iter().map(|h| h.penalty_cycles).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{MemWidth, PostModify};

    fn make_slot(slot: SlotIndex, op: Operation) -> SlotOp {
        SlotOp::new(slot, op)
    }

    #[test]
    fn test_no_conflict_independent_slots() {
        // Scalar0 and Load use different resources
        let scalar = make_slot(SlotIndex::Scalar0, Operation::ScalarAdd);
        let load = make_slot(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        );

        assert!(check_slot_conflict(&scalar, &load).is_none());
    }

    #[test]
    fn test_conflict_dual_scalar_mul() {
        // Two scalar multiplies share the ALU
        let mul0 = make_slot(SlotIndex::Scalar0, Operation::ScalarMul);
        let mul1 = make_slot(SlotIndex::Scalar1, Operation::ScalarMul);

        let conflict = check_slot_conflict(&mul0, &mul1);
        assert!(conflict.is_some());
        assert_eq!(conflict.unwrap(), ExecutionResource::ScalarAlu);
    }

    #[test]
    fn test_no_conflict_scalar_add_and_mul() {
        // Scalar add doesn't use the shared ALU
        let add = make_slot(SlotIndex::Scalar0, Operation::ScalarAdd);
        let mul = make_slot(SlotIndex::Scalar1, Operation::ScalarMul);

        assert!(check_slot_conflict(&add, &mul).is_none());
    }

    #[test]
    fn test_conflict_dual_post_modify() {
        // Both loads with post-modify share AGU
        let load_a = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::Immediate(4),
            },
        );
        let store = SlotOp::new(
            SlotIndex::Store,
            Operation::Store {
                width: MemWidth::Word,
                post_modify: PostModify::Immediate(4),
            },
        );

        let conflict = check_slot_conflict(&load_a, &store);
        assert!(conflict.is_some());
        assert_eq!(conflict.unwrap(), ExecutionResource::AddressGen);
    }

    #[test]
    fn test_bundle_no_conflicts() {
        // Bundle with independent operations
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(make_slot(SlotIndex::Scalar0, Operation::ScalarAdd));
        bundle.set_slot(make_slot(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        ));

        let hazards = check_bundle_conflicts(&bundle);
        assert!(hazards.is_empty());
        assert_eq!(bundle_structural_penalty(&bundle), 0);
    }

    #[test]
    fn test_bundle_with_conflict() {
        // Bundle with conflicting scalar multiplies
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(make_slot(SlotIndex::Scalar0, Operation::ScalarMul));
        bundle.set_slot(make_slot(SlotIndex::Scalar1, Operation::ScalarMul));

        let hazards = check_bundle_conflicts(&bundle);
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].resource, ExecutionResource::ScalarAlu);
        assert_eq!(bundle_structural_penalty(&bundle), 1);
    }

    #[test]
    fn test_resource_requirements() {
        let load = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::Immediate(4),
            },
        );

        let reqs = get_requirements(&load);
        assert!(reqs.resources.contains(&ExecutionResource::LoadPortA));
        assert!(reqs.resources.contains(&ExecutionResource::AddressGen));
    }

    #[test]
    fn test_serialization_penalty() {
        assert_eq!(ExecutionResource::ScalarAlu.serialization_penalty(), 1);
        assert_eq!(ExecutionResource::VectorUnit.serialization_penalty(), 1);
    }
}
