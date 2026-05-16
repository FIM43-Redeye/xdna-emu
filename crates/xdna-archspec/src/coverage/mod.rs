//! Two-axis coverage provenance infrastructure. Spec:
//! docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod surface;
pub mod verdict;

use crate::aie2::isa::SemanticOp;
use crate::types::{Architecture, ModuleKind, SubsystemKind, TileKind};
use serde::{Deserialize, Serialize};

/// A fine node, architecture-qualified (spec Section 7: all identity is
/// arch-qualified). Two kinds of node universe today: ISA semantics and
/// registers. Capability-spine domains also get a NodeId so they can be
/// claimed by units uniformly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeId {
    Semantic {
        arch: Architecture,
        op: SemanticOp,
    },
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

impl NodeId {
    pub fn arch(&self) -> Architecture {
        match self {
            NodeId::Semantic { arch, .. }
            | NodeId::Register { arch, .. }
            | NodeId::Capability { arch, .. } => *arch,
        }
    }
}
