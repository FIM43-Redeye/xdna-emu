//! Two-axis coverage provenance infrastructure. Spec:
//! docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod derive;
pub mod surface;
pub mod units;
pub mod verdict;

use crate::aie2::isa::SemanticOp;
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
