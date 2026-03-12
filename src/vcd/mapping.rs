//! Hierarchical signal mapping tree.
//!
//! Maps between aiesimulator VCD signal hierarchies and [`StatePath`] values.
//! The tree mirrors the VCD scope structure: fixed scopes ("top",
//! "math_engine"), tile groups ("tile_0_1"), and per-tile subsystem mappings
//! (locks, DMA, streams, core, events).
//!
//! # Construction
//!
//! Build the tree declaratively with [`MappingTree::builder`]:
//!
//! ```ignore
//! let tree = MappingTree::builder()
//!     .scope("top")
//!     .scope("math_engine")
//!     .tile_group("mem_row", &[(0, 1), (1, 1)])
//!         .subsystem(lock_mapping)
//!         .done_tile_group()
//!     .build();
//! ```
//!
//! # Resolution
//!
//! Given VCD signal name segments (e.g. `["top", "math_engine", "mem_row",
//! "tile_0_1", "locks", "value_3"]`), [`MappingTree::resolve`] walks the tree
//! and returns the matching [`StatePath`].
//!
//! # Enumeration
//!
//! [`MappingTree::enumerate_all`] produces every possible [`StatePath`] the
//! tree can generate -- used for VCD emission header generation and coverage
//! auditing.

use std::collections::HashMap;

use super::state_path::{StatePath, Subsystem};

// ---------------------------------------------------------------------------
// SignalDef -- a single signal template within a subsystem
// ---------------------------------------------------------------------------

/// Factory function that produces a [`StatePath`] given tile coordinates and
/// an index. Stored as a function pointer for simplicity and `Clone` support.
pub type SignalFactory = fn(col: u8, row: u8, idx: u8) -> StatePath;

/// A single signal definition within a [`SubsystemMapping`].
///
/// Covers both indexed signals (e.g. `value_0` .. `value_63`) and fixed
/// (non-indexed) signals. Fixed signals use `count = 1` and the factory
/// receives `idx = 0`.
#[derive(Clone)]
struct SignalDef {
    /// VCD name prefix. For indexed signals, the full name is `{prefix}_{idx}`.
    /// For fixed signals, the full name is just `prefix`.
    prefix: String,
    /// Number of instances (1 for fixed signals).
    count: u8,
    /// Bit width of the signal in the VCD (informational, used by emit).
    #[allow(dead_code)]
    width: u32,
    /// Whether the signal uses `{prefix}_{idx}` naming (true) or just
    /// `{prefix}` (false).
    indexed: bool,
    /// Factory that builds a [`StatePath`] from (col, row, idx).
    factory: SignalFactory,
}

// ---------------------------------------------------------------------------
// NestedGroupDef -- a group of signals under a named scope
// ---------------------------------------------------------------------------

/// A named scope containing child signals, e.g. `s2mm_state0` containing
/// `cur_bd`, `fsm_state`, etc. Used for DMA channel groupings.
#[derive(Clone)]
struct NestedGroupDef {
    /// Name prefix for the group. The full scope name for instance `i` is
    /// `{prefix}{i}` (e.g. `s2mm_state0`, `s2mm_state1`).
    prefix: String,
    /// Number of group instances.
    count: u8,
    /// Child signal definitions within each group instance.
    children: Vec<NestedSignalDef>,
}

/// Factory for nested signals: receives (col, row, group_idx).
pub type NestedSignalFactory = fn(col: u8, row: u8, group_idx: u8) -> StatePath;

/// A signal definition inside a [`NestedGroupDef`].
#[derive(Clone)]
struct NestedSignalDef {
    /// Leaf signal name (e.g. `"cur_bd"`, `"fsm_state"`).
    name: String,
    /// Bit width (informational).
    #[allow(dead_code)]
    width: u32,
    /// Factory that builds a [`StatePath`] from (col, row, group_idx).
    factory: NestedSignalFactory,
}

// ---------------------------------------------------------------------------
// SubsystemMapping
// ---------------------------------------------------------------------------

/// Mapping definition for one hardware subsystem within a tile.
///
/// Created by Tasks 4-7 (lock, DMA, stream, core, event mappings) and plugged
/// into the [`MappingTree`] via the builder API. Each `SubsystemMapping`
/// knows its VCD scope name (e.g. `"locks"`) and contains signal definitions
/// that can resolve leaf signal names to [`StatePath`] values.
#[derive(Clone)]
pub struct SubsystemMapping {
    /// VCD scope name that identifies this subsystem (e.g. `"locks"`,
    /// `"stream_switch"`, `"cm"`).
    scope_name: String,
    /// Hardware subsystem classification.
    #[allow(dead_code)]
    subsystem: Subsystem,
    /// Flat (non-nested) signal definitions.
    signals: Vec<SignalDef>,
    /// Nested group definitions (e.g. DMA channel groups).
    nested_groups: Vec<NestedGroupDef>,
}

impl SubsystemMapping {
    /// Create a new subsystem mapping with the given VCD scope name and
    /// subsystem classification.
    pub fn new(scope_name: &str, subsystem: Subsystem) -> Self {
        SubsystemMapping {
            scope_name: scope_name.to_string(),
            subsystem,
            signals: Vec::new(),
            nested_groups: Vec::new(),
        }
    }

    /// Add an indexed signal template: matches VCD names `{prefix}_{0}` through
    /// `{prefix}_{count-1}`.
    ///
    /// The `factory` receives `(col, row, idx)` where `idx` is parsed from the
    /// signal name suffix.
    pub fn indexed_signal(
        mut self,
        prefix: &str,
        count: u8,
        width: u32,
        factory: SignalFactory,
    ) -> Self {
        self.signals.push(SignalDef {
            prefix: prefix.to_string(),
            count,
            width,
            indexed: true,
            factory,
        });
        self
    }

    /// Add a single fixed-name signal (no index suffix).
    ///
    /// The `factory` receives `(col, row, 0)`.
    pub fn fixed_signal(
        mut self,
        name: &str,
        width: u32,
        factory: SignalFactory,
    ) -> Self {
        self.signals.push(SignalDef {
            prefix: name.to_string(),
            count: 1,
            width,
            indexed: false,
            factory,
        });
        self
    }

    /// Add a nested group: matches VCD scopes `{prefix}{0}` .. `{prefix}{count-1}`,
    /// each containing child signals. Used for DMA channel groups like
    /// `s2mm_state0.cur_bd`.
    ///
    /// The `children` are `(name, width, factory)` tuples for each leaf signal
    /// within the group.
    pub fn nested_group(
        mut self,
        prefix: &str,
        count: u8,
        children: Vec<(&str, u32, NestedSignalFactory)>,
    ) -> Self {
        self.nested_groups.push(NestedGroupDef {
            prefix: prefix.to_string(),
            count,
            children: children
                .into_iter()
                .map(|(name, width, factory)| NestedSignalDef {
                    name: name.to_string(),
                    width,
                    factory,
                })
                .collect(),
        });
        self
    }

    /// The VCD scope name for this subsystem.
    pub fn scope_name(&self) -> &str {
        &self.scope_name
    }

    /// Attempt to resolve remaining VCD segments to a [`StatePath`].
    ///
    /// `segments` are the VCD name parts after the subsystem scope has been
    /// matched. `col` and `row` are the tile coordinates extracted by the
    /// parent tile group.
    ///
    /// For flat signals, expects exactly one segment like `"value_3"`.
    /// For nested groups, expects two segments like `["s2mm_state0", "cur_bd"]`.
    pub fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        match segments.len() {
            1 => self.resolve_flat(segments[0], col, row),
            2 => self.resolve_nested(segments[0], segments[1], col, row),
            _ => None,
        }
    }

    /// Resolve a flat signal name like `"value_3"` or `"pc"`.
    fn resolve_flat(&self, name: &str, col: u8, row: u8) -> Option<StatePath> {
        for def in &self.signals {
            if def.indexed {
                // Try matching "{prefix}_{idx}"
                if let Some(suffix) = name.strip_prefix(&def.prefix) {
                    if let Some(idx_str) = suffix.strip_prefix('_') {
                        if let Ok(idx) = idx_str.parse::<u8>() {
                            if idx < def.count {
                                return Some((def.factory)(col, row, idx));
                            }
                        }
                    }
                }
            } else if name == def.prefix {
                return Some((def.factory)(col, row, 0));
            }
        }
        None
    }

    /// Resolve a nested signal like `["s2mm_state0", "cur_bd"]`.
    fn resolve_nested(
        &self,
        group_name: &str,
        child_name: &str,
        col: u8,
        row: u8,
    ) -> Option<StatePath> {
        for group in &self.nested_groups {
            // Match "{prefix}{idx}" (no underscore between prefix and index)
            if let Some(idx_str) = group_name.strip_prefix(&group.prefix) {
                if let Ok(idx) = idx_str.parse::<u8>() {
                    if idx < group.count {
                        for child in &group.children {
                            if child.name == child_name {
                                return Some((child.factory)(col, row, idx));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Enumerate all [`StatePath`] values this mapping can produce for the
    /// given tile coordinates.
    pub fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        let mut paths = Vec::new();

        // Flat signals
        for def in &self.signals {
            if def.indexed {
                for idx in 0..def.count {
                    paths.push((def.factory)(col, row, idx));
                }
            } else {
                paths.push((def.factory)(col, row, 0));
            }
        }

        // Nested groups
        for group in &self.nested_groups {
            for idx in 0..group.count {
                for child in &group.children {
                    paths.push((child.factory)(col, row, idx));
                }
            }
        }

        paths
    }
}

// ---------------------------------------------------------------------------
// MappingNode -- internal tree node
// ---------------------------------------------------------------------------

/// Internal node in the mapping tree.
enum MappingNode {
    /// A fixed scope level (e.g. "top", "math_engine"). Matches a single
    /// segment exactly and has one child.
    Scope {
        name: String,
        child: Box<MappingNode>,
    },

    /// A fan-out node containing one or more tile groups and/or scope children.
    /// Used when multiple tile groups (mem_row, array, shim) are siblings.
    FanOut {
        children: Vec<MappingNode>,
    },

    /// A tile group: matches "tile_{col}_{row}" for a known set of tiles,
    /// then delegates to subsystem mappings.
    TileGroup {
        /// VCD scope prefix (e.g. "mem_row", "array", "shim").
        prefix: String,
        /// Set of valid (col, row) pairs, stored as a HashMap for O(1) lookup.
        tiles: HashMap<(u8, u8), ()>,
        /// Subsystem mappings keyed by their scope name.
        subsystems: Vec<SubsystemMapping>,
    },
}

impl MappingNode {
    /// Walk VCD segments through this node, returning a [`StatePath`] if
    /// the full path matches.
    fn resolve(&self, segments: &[&str]) -> Option<StatePath> {
        match self {
            MappingNode::Scope { name, child } => {
                if segments.first() == Some(&name.as_str()) {
                    child.resolve(&segments[1..])
                } else {
                    None
                }
            }

            MappingNode::FanOut { children } => {
                for child in children {
                    if let Some(path) = child.resolve(segments) {
                        return Some(path);
                    }
                }
                None
            }

            MappingNode::TileGroup {
                prefix,
                tiles,
                subsystems,
            } => {
                // Need at least 3 segments: prefix, tile_C_R, subsystem scope
                // (+ at least 1 for the signal leaf).
                if segments.len() < 3 {
                    return None;
                }
                if segments[0] != prefix.as_str() {
                    return None;
                }
                // Parse "tile_{col}_{row}"
                let (col, row) = parse_tile_name(segments[1])?;
                if !tiles.contains_key(&(col, row)) {
                    return None;
                }
                // Find matching subsystem
                let subsystem_scope = segments[2];
                for sub in subsystems {
                    if sub.scope_name() == subsystem_scope {
                        return sub.resolve(&segments[3..], col, row);
                    }
                }
                None
            }
        }
    }

    /// Enumerate all [`StatePath`] values reachable from this node.
    fn enumerate_all(&self) -> Vec<StatePath> {
        match self {
            MappingNode::Scope { child, .. } => child.enumerate_all(),

            MappingNode::FanOut { children } => {
                let mut paths = Vec::new();
                for child in children {
                    paths.extend(child.enumerate_all());
                }
                paths
            }

            MappingNode::TileGroup {
                tiles, subsystems, ..
            } => {
                let mut paths = Vec::new();
                // Enumerate in deterministic order: sort tiles by (col, row).
                let mut tile_list: Vec<(u8, u8)> = tiles.keys().copied().collect();
                tile_list.sort();
                for (col, row) in tile_list {
                    for sub in subsystems {
                        paths.extend(sub.enumerate(col, row));
                    }
                }
                paths
            }
        }
    }
}

/// Parse a VCD tile scope name like `"tile_0_1"` into `(col, row)`.
fn parse_tile_name(name: &str) -> Option<(u8, u8)> {
    let rest = name.strip_prefix("tile_")?;
    let (col_str, row_str) = rest.split_once('_')?;
    let col = col_str.parse::<u8>().ok()?;
    let row = row_str.parse::<u8>().ok()?;
    Some((col, row))
}

// ---------------------------------------------------------------------------
// MappingTree -- the public handle
// ---------------------------------------------------------------------------

/// Immutable mapping tree that resolves VCD signal paths to [`StatePath`]
/// values and enumerates all mapped signals.
pub struct MappingTree {
    root: MappingNode,
}

impl MappingTree {
    /// Start building a new mapping tree.
    pub fn builder() -> TreeBuilder {
        TreeBuilder {
            scopes: Vec::new(),
            tile_groups: Vec::new(),
        }
    }

    /// Resolve a VCD signal path (split into segments) to a [`StatePath`].
    ///
    /// Returns `None` if the path does not match any mapped signal.
    pub fn resolve(&self, segments: &[&str]) -> Option<StatePath> {
        self.root.resolve(segments)
    }

    /// Enumerate every [`StatePath`] the tree can produce.
    ///
    /// Used for VCD emission (generating the header with all possible signals)
    /// and for coverage auditing (checking which signals are mapped).
    pub fn enumerate_all(&self) -> Vec<StatePath> {
        self.root.enumerate_all()
    }
}

// ---------------------------------------------------------------------------
// TreeBuilder -- declarative construction
// ---------------------------------------------------------------------------

/// Builder for constructing a [`MappingTree`] declaratively.
///
/// Scopes accumulate as a prefix chain. Tile groups are collected and wrapped
/// in a [`MappingNode::FanOut`] if there are multiple, or used directly if
/// there is only one.
pub struct TreeBuilder {
    /// Scope names accumulated so far (outermost first).
    scopes: Vec<String>,
    /// Tile groups collected at the current depth.
    tile_groups: Vec<MappingNode>,
}

impl TreeBuilder {
    /// Push a fixed scope level (e.g. `"top"`, `"math_engine"`).
    pub fn scope(mut self, name: &str) -> Self {
        self.scopes.push(name.to_string());
        self
    }

    /// Begin a tile group definition. Returns a [`TileGroupBuilder`] that
    /// collects subsystem mappings. Call [`TileGroupBuilder::done_tile_group`]
    /// to return to this builder.
    pub fn tile_group(self, prefix: &str, tiles: &[(u8, u8)]) -> TileGroupBuilder {
        TileGroupBuilder {
            parent: self,
            prefix: prefix.to_string(),
            tiles: tiles.iter().copied().map(|t| (t, ())).collect(),
            subsystems: Vec::new(),
        }
    }

    /// Consume the builder and produce the immutable [`MappingTree`].
    ///
    /// The tile groups are wrapped in a FanOut node (or used directly if
    /// there is only one), then wrapped in the accumulated scope chain.
    pub fn build(self) -> MappingTree {
        // Build the inner node from tile groups.
        let inner = if self.tile_groups.len() == 1 {
            self.tile_groups.into_iter().next().unwrap()
        } else {
            MappingNode::FanOut {
                children: self.tile_groups,
            }
        };

        // Wrap in scope chain (innermost scope wraps the inner node first).
        let root = self
            .scopes
            .into_iter()
            .rev()
            .fold(inner, |child, name| MappingNode::Scope {
                name,
                child: Box::new(child),
            });

        MappingTree { root }
    }
}

// ---------------------------------------------------------------------------
// TileGroupBuilder
// ---------------------------------------------------------------------------

/// Builder for a single tile group within a [`MappingTree`].
///
/// Collects subsystem mappings, then returns to the parent [`TreeBuilder`]
/// via [`done_tile_group`](Self::done_tile_group).
pub struct TileGroupBuilder {
    parent: TreeBuilder,
    prefix: String,
    tiles: HashMap<(u8, u8), ()>,
    subsystems: Vec<SubsystemMapping>,
}

impl TileGroupBuilder {
    /// Add a subsystem mapping to this tile group.
    pub fn subsystem(mut self, mapping: SubsystemMapping) -> Self {
        self.subsystems.push(mapping);
        self
    }

    /// Finish this tile group and return to the parent tree builder.
    pub fn done_tile_group(mut self) -> TreeBuilder {
        let node = MappingNode::TileGroup {
            prefix: self.prefix,
            tiles: self.tiles,
            subsystems: self.subsystems,
        };
        self.parent.tile_groups.push(node);
        self.parent
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    // -- SubsystemMapping unit tests --

    #[test]
    fn subsystem_mapping_indexed_signal() {
        let mapping = SubsystemMapping::new("locks", Subsystem::Lock).indexed_signal(
            "value",
            16,
            32,
            |col, row, idx| StatePath::LockValue { col, row, idx },
        );
        let result = mapping.resolve(&["value_3"], 2, 1);
        assert_eq!(
            result,
            Some(StatePath::LockValue {
                col: 2,
                row: 1,
                idx: 3,
            })
        );
    }

    #[test]
    fn subsystem_mapping_indexed_signal_out_of_range() {
        let mapping = SubsystemMapping::new("locks", Subsystem::Lock).indexed_signal(
            "value",
            4,
            32,
            |col, row, idx| StatePath::LockValue { col, row, idx },
        );
        // idx=4 is out of range (count=4, valid 0..3)
        assert_eq!(mapping.resolve(&["value_4"], 0, 0), None);
        // idx=3 is the last valid
        assert!(mapping.resolve(&["value_3"], 0, 0).is_some());
    }

    #[test]
    fn subsystem_mapping_fixed_signal() {
        let mapping = SubsystemMapping::new("cm", Subsystem::Core).fixed_signal(
            "pm_address",
            32,
            |col, row, _idx| StatePath::CorePmAddress { col, row },
        );
        let result = mapping.resolve(&["pm_address"], 1, 3);
        assert_eq!(
            result,
            Some(StatePath::CorePmAddress { col: 1, row: 3 })
        );
    }

    #[test]
    fn subsystem_mapping_fixed_signal_no_match() {
        let mapping = SubsystemMapping::new("cm", Subsystem::Core).fixed_signal(
            "pm_address",
            32,
            |col, row, _idx| StatePath::CorePmAddress { col, row },
        );
        assert_eq!(mapping.resolve(&["pm_data"], 0, 0), None);
    }

    #[test]
    fn subsystem_mapping_nested_group() {
        let mapping =
            SubsystemMapping::new("dma", Subsystem::Dma).nested_group(
                "s2mm_state",
                2,
                vec![
                    ("cur_bd", 4, |col, row, ch| StatePath::DmaCurrentBd {
                        col,
                        row,
                        dir: DmaDir::S2mm,
                        ch,
                    }),
                    ("fsm_state", 4, |col, row, ch| StatePath::DmaFsmState {
                        col,
                        row,
                        dir: DmaDir::S2mm,
                        ch,
                    }),
                ],
            );
        let result = mapping.resolve(&["s2mm_state0", "cur_bd"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::DmaCurrentBd {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0,
            })
        );
        let result2 = mapping.resolve(&["s2mm_state1", "fsm_state"], 0, 1);
        assert_eq!(
            result2,
            Some(StatePath::DmaFsmState {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 1,
            })
        );
    }

    #[test]
    fn subsystem_mapping_nested_group_out_of_range() {
        let mapping =
            SubsystemMapping::new("dma", Subsystem::Dma).nested_group(
                "s2mm_state",
                2,
                vec![("cur_bd", 4, |col, row, ch| StatePath::DmaCurrentBd {
                    col,
                    row,
                    dir: DmaDir::S2mm,
                    ch,
                })],
            );
        // idx=2 is out of range
        assert_eq!(mapping.resolve(&["s2mm_state2", "cur_bd"], 0, 0), None);
    }

    #[test]
    fn subsystem_mapping_nested_unknown_child() {
        let mapping =
            SubsystemMapping::new("dma", Subsystem::Dma).nested_group(
                "s2mm_state",
                2,
                vec![("cur_bd", 4, |col, row, ch| StatePath::DmaCurrentBd {
                    col,
                    row,
                    dir: DmaDir::S2mm,
                    ch,
                })],
            );
        assert_eq!(
            mapping.resolve(&["s2mm_state0", "nonexistent"], 0, 0),
            None
        );
    }

    #[test]
    fn subsystem_mapping_enumerate() {
        let mapping = SubsystemMapping::new("locks", Subsystem::Lock)
            .indexed_signal("value", 4, 32, |col, row, idx| StatePath::LockValue {
                col,
                row,
                idx,
            })
            .indexed_signal("lock_op", 4, 32, |col, row, idx| StatePath::LockOp {
                col,
                row,
                idx,
            });
        let paths = mapping.enumerate(0, 1);
        assert_eq!(paths.len(), 8); // 4 values + 4 ops
        assert!(paths.contains(&StatePath::LockValue {
            col: 0,
            row: 1,
            idx: 0,
        }));
        assert!(paths.contains(&StatePath::LockOp {
            col: 0,
            row: 1,
            idx: 3,
        }));
    }

    #[test]
    fn subsystem_mapping_enumerate_mixed() {
        let mapping = SubsystemMapping::new("cm", Subsystem::Core)
            .fixed_signal("pm_address", 32, |col, row, _| {
                StatePath::CorePmAddress { col, row }
            })
            .indexed_signal("pc_E", 2, 32, |col, row, idx| StatePath::CorePc {
                col,
                row,
                stage: idx,
            });
        let paths = mapping.enumerate(1, 3);
        assert_eq!(paths.len(), 3); // 1 fixed + 2 indexed
    }

    #[test]
    fn subsystem_mapping_enumerate_nested() {
        let mapping =
            SubsystemMapping::new("dma", Subsystem::Dma).nested_group(
                "s2mm_state",
                2,
                vec![
                    ("cur_bd", 4, |col, row, ch| StatePath::DmaCurrentBd {
                        col,
                        row,
                        dir: DmaDir::S2mm,
                        ch,
                    }),
                    ("fsm_state", 4, |col, row, ch| StatePath::DmaFsmState {
                        col,
                        row,
                        dir: DmaDir::S2mm,
                        ch,
                    }),
                ],
            );
        let paths = mapping.enumerate(0, 1);
        // 2 channels x 2 signals = 4
        assert_eq!(paths.len(), 4);
    }

    // -- parse_tile_name tests --

    #[test]
    fn parse_tile_name_valid() {
        assert_eq!(parse_tile_name("tile_0_1"), Some((0, 1)));
        assert_eq!(parse_tile_name("tile_4_5"), Some((4, 5)));
        assert_eq!(parse_tile_name("tile_12_0"), Some((12, 0)));
    }

    #[test]
    fn parse_tile_name_invalid() {
        assert_eq!(parse_tile_name("not_a_tile"), None);
        assert_eq!(parse_tile_name("tile_"), None);
        assert_eq!(parse_tile_name("tile_0"), None);
        assert_eq!(parse_tile_name("tile_x_1"), None);
    }

    // -- MappingTree integration tests --

    /// Minimal test tree: locks in two mem_row tiles.
    fn build_test_tree() -> MappingTree {
        let lock_mapping = SubsystemMapping::new("locks", Subsystem::Lock)
            .indexed_signal("value", 64, 32, |col, row, idx| StatePath::LockValue {
                col,
                row,
                idx,
            })
            .indexed_signal("lock_op", 64, 32, |col, row, idx| StatePath::LockOp {
                col,
                row,
                idx,
            });

        MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1), (1, 1)])
            .subsystem(lock_mapping)
            .done_tile_group()
            .build()
    }

    #[test]
    fn resolve_lock_signal() {
        let tree = build_test_tree();
        let segments = ["top", "math_engine", "mem_row", "tile_0_1", "locks", "value_3"];
        let result = tree.resolve(&segments);
        assert_eq!(
            result,
            Some(StatePath::LockValue {
                col: 0,
                row: 1,
                idx: 3,
            })
        );
    }

    #[test]
    fn resolve_lock_op_signal() {
        let tree = build_test_tree();
        let segments = [
            "top",
            "math_engine",
            "mem_row",
            "tile_1_1",
            "locks",
            "lock_op_15",
        ];
        let result = tree.resolve(&segments);
        assert_eq!(
            result,
            Some(StatePath::LockOp {
                col: 1,
                row: 1,
                idx: 15,
            })
        );
    }

    #[test]
    fn resolve_unknown_signal_returns_none() {
        let tree = build_test_tree();
        let segments = ["top", "math_engine", "totally_unknown"];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn resolve_unknown_tile_returns_none() {
        let tree = build_test_tree();
        // Tile (2, 1) is not in the tree.
        let segments = [
            "top",
            "math_engine",
            "mem_row",
            "tile_2_1",
            "locks",
            "value_0",
        ];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn resolve_unknown_subsystem_returns_none() {
        let tree = build_test_tree();
        let segments = [
            "top",
            "math_engine",
            "mem_row",
            "tile_0_1",
            "dma",
            "something",
        ];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn resolve_wrong_root_returns_none() {
        let tree = build_test_tree();
        let segments = ["bottom", "math_engine", "mem_row", "tile_0_1", "locks", "value_0"];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn resolve_too_short_returns_none() {
        let tree = build_test_tree();
        let segments = ["top", "math_engine"];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn resolve_different_tiles() {
        let tree = build_test_tree();
        let seg_a = [
            "top",
            "math_engine",
            "mem_row",
            "tile_0_1",
            "locks",
            "value_0",
        ];
        let seg_b = [
            "top",
            "math_engine",
            "mem_row",
            "tile_1_1",
            "locks",
            "value_0",
        ];
        let a = tree.resolve(&seg_a).unwrap();
        let b = tree.resolve(&seg_b).unwrap();
        assert_eq!(
            a,
            StatePath::LockValue {
                col: 0,
                row: 1,
                idx: 0,
            }
        );
        assert_eq!(
            b,
            StatePath::LockValue {
                col: 1,
                row: 1,
                idx: 0,
            }
        );
    }

    #[test]
    fn enumerate_all_paths() {
        let tree = build_test_tree();
        let paths = tree.enumerate_all();
        // 2 tiles x (64 values + 64 ops) = 256
        assert_eq!(paths.len(), 256);
        assert!(paths.iter().any(|p| matches!(p, StatePath::LockValue { .. })));
        assert!(paths.iter().any(|p| matches!(p, StatePath::LockOp { .. })));
    }

    #[test]
    fn enumerate_all_contains_both_tiles() {
        let tree = build_test_tree();
        let paths = tree.enumerate_all();
        let tile_0_1_count = paths.iter().filter(|p| p.tile() == (0, 1)).count();
        let tile_1_1_count = paths.iter().filter(|p| p.tile() == (1, 1)).count();
        assert_eq!(tile_0_1_count, 128); // 64 + 64
        assert_eq!(tile_1_1_count, 128);
    }

    // -- Multi-tile-group (FanOut) tests --

    fn build_multi_group_tree() -> MappingTree {
        let lock_mapping = SubsystemMapping::new("locks", Subsystem::Lock)
            .indexed_signal("value", 4, 32, |col, row, idx| StatePath::LockValue {
                col,
                row,
                idx,
            });

        let core_mapping = SubsystemMapping::new("cm", Subsystem::Core)
            .fixed_signal("pm_address", 32, |col, row, _| {
                StatePath::CorePmAddress { col, row }
            });

        MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping)
            .done_tile_group()
            .tile_group("array", &[(0, 2), (0, 3)])
            .subsystem(core_mapping)
            .done_tile_group()
            .build()
    }

    #[test]
    fn resolve_across_tile_groups() {
        let tree = build_multi_group_tree();

        // mem_row tile
        let seg_lock = [
            "top",
            "math_engine",
            "mem_row",
            "tile_0_1",
            "locks",
            "value_2",
        ];
        assert_eq!(
            tree.resolve(&seg_lock),
            Some(StatePath::LockValue {
                col: 0,
                row: 1,
                idx: 2,
            })
        );

        // array tile
        let seg_core = [
            "top",
            "math_engine",
            "array",
            "tile_0_3",
            "cm",
            "pm_address",
        ];
        assert_eq!(
            tree.resolve(&seg_core),
            Some(StatePath::CorePmAddress { col: 0, row: 3 })
        );
    }

    #[test]
    fn enumerate_multi_group() {
        let tree = build_multi_group_tree();
        let paths = tree.enumerate_all();
        // mem_row: 1 tile x 4 values = 4
        // array: 2 tiles x 1 fixed = 2
        assert_eq!(paths.len(), 6);
    }

    // -- Nested group in tree --

    #[test]
    fn resolve_nested_in_tree() {
        let dma = SubsystemMapping::new("dma", Subsystem::Dma).nested_group(
            "s2mm_state",
            2,
            vec![("cur_bd", 4, |col, row, ch| StatePath::DmaCurrentBd {
                col,
                row,
                dir: DmaDir::S2mm,
                ch,
            })],
        );

        let tree = MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(dma)
            .done_tile_group()
            .build();

        let segments = [
            "top",
            "math_engine",
            "mem_row",
            "tile_0_1",
            "dma",
            "s2mm_state1",
            "cur_bd",
        ];
        assert_eq!(
            tree.resolve(&segments),
            Some(StatePath::DmaCurrentBd {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 1,
            })
        );
    }

    // -- Multiple subsystems per tile group --

    #[test]
    fn multiple_subsystems_in_one_tile_group() {
        let locks = SubsystemMapping::new("locks", Subsystem::Lock).indexed_signal(
            "value",
            4,
            32,
            |col, row, idx| StatePath::LockValue { col, row, idx },
        );
        let core = SubsystemMapping::new("cm", Subsystem::Core).fixed_signal(
            "pm_address",
            32,
            |col, row, _| StatePath::CorePmAddress { col, row },
        );

        let tree = MappingTree::builder()
            .scope("top")
            .tile_group("array", &[(0, 2)])
            .subsystem(locks)
            .subsystem(core)
            .done_tile_group()
            .build();

        // Lock signal
        let seg_lock = ["top", "array", "tile_0_2", "locks", "value_1"];
        assert_eq!(
            tree.resolve(&seg_lock),
            Some(StatePath::LockValue {
                col: 0,
                row: 2,
                idx: 1,
            })
        );

        // Core signal
        let seg_core = ["top", "array", "tile_0_2", "cm", "pm_address"];
        assert_eq!(
            tree.resolve(&seg_core),
            Some(StatePath::CorePmAddress { col: 0, row: 2 })
        );

        // Enumerate: 4 lock values + 1 core signal = 5
        let paths = tree.enumerate_all();
        assert_eq!(paths.len(), 5);
    }
}
