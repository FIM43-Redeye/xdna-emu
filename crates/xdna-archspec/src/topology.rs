//! Tile topology trait: coordinate-to-kind classification and
//! coordinate-to-neighbor navigation.
//!
//! Subsystem 2 of the device-family refactor introduces this trait as a
//! behavioral seam for per-arch tile layout. AIE2 / AIE2P use the uniform
//! shim-at-row-0 + memtile-at-row-1 + compute-at-row-2+ layout that the
//! concrete `aie2::topology::Aie2Topology` impl captures. AIE1 (Versal)
//! has non-uniform shim columns (ShimNoc + ShimPl interleaved) and
//! alternating-row memory adjacency; its impl will live in
//! `aie1::topology::Aie1Topology` when AIE1 support lands (not this
//! subsystem).
//!
//! Consumers access an impl via `ArchModel::topology()`. Hot-path cost on
//! AIE2 is a single Box allocation on each accessor call; the accessor
//! is called at config time, not per-instruction.

use crate::types::{ArchModel, Architecture, TileKind};

/// Cardinal direction for tile-to-tile navigation.
///
/// `North` increases `row`; `South` decreases it. `East` increases
/// `col`; `West` decreases it. Array coordinates are `(col, row)`
/// pairs, conventional across `xdna-emu` and archspec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

/// Classification and navigation for the tile grid of a given arch family.
///
/// Implementors describe a single arch family's tile layout. Construct a
/// concrete impl from an `ArchModel`'s extents (columns, rows, memtile
/// row count) via `ArchModel::topology()`.
pub trait TileTopology: Send + Sync {
    /// Classify the tile at `(col, row)`.
    ///
    /// AIE2 impls return `TileKind::ShimNoc` for row 0 and ignore `col`.
    /// AIE1 impls consult a per-column shim-kind vector and may return
    /// either `ShimNoc` or `ShimPl` at row 0.
    fn classify(&self, col: u8, row: u8) -> TileKind;

    /// Return the coordinates of the neighbor in the given direction, or
    /// `None` if no neighbor exists in that direction (array edge, or
    /// an arch-specific adjacency absence such as shim rows having no
    /// south neighbor).
    fn neighbor(&self, col: u8, row: u8, dir: Direction) -> Option<(u8, u8)>;
}

impl ArchModel {
    /// Return a topology impl for this arch family.
    ///
    /// Dispatches on `self.arch`. Returns a boxed trait object so
    /// consumers don't need to be generic over the concrete impl type.
    /// Call-site cost: one small allocation per call. `topology()` is a cold
    /// accessor (called at config time, not per-instruction), so the
    /// allocation is acceptable.
    pub fn topology(&self) -> Box<dyn TileTopology + '_> {
        match self.arch {
            Architecture::Aie2 | Architecture::Aie2p => {
                Box::new(crate::aie2::topology::Aie2Topology::from_model(self))
            }
            Architecture::Aie => {
                unimplemented!(
                    "AIE1 topology impl not populated until AIE1 support lands \
                     (tracked as post-Subsystem-2 follow-on work)."
                )
            }
        }
    }
}
