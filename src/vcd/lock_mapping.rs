//! Lock subsystem signal mapping.
//!
//! VCD hierarchy: `tile_X_Y.locks.value_N` and `tile_X_Y.locks.lock_op_N`
//!
//! Maps to: [`StatePath::LockValue`] and [`StatePath::LockOp`].
//!
//! Lock counts per tile type (from mlir-aie device model):
//! - Compute tile: 16 locks (indices 0-15)
//! - Mem tile:     64 locks (indices 0-63)
//! - Shim tile:    16 locks (indices 0-15)

use crate::vcd::mapping::SubsystemMapping;
use crate::vcd::state_path::{StatePath, Subsystem};

/// Build the lock subsystem mapping for a tile with `num_locks` locks.
///
/// Returns a [`SubsystemMapping`] covering the VCD `locks` scope, mapping:
/// - `value_{idx}`   -> [`StatePath::LockValue`]
/// - `lock_op_{idx}` -> [`StatePath::LockOp`]
///
/// Pass `num_locks = 16` for compute/shim tiles and `num_locks = 64` for
/// mem tiles. The mapping rejects any index outside `[0, num_locks)`.
pub fn lock_mapping(num_locks: u8) -> SubsystemMapping {
    SubsystemMapping::new("locks", Subsystem::Lock)
        .indexed_signal("value", num_locks, 32, |col, row, idx| {
            StatePath::LockValue { col, row, idx }
        })
        .indexed_signal("lock_op", num_locks, 32, |col, row, idx| {
            StatePath::LockOp { col, row, idx }
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::StatePath;

    #[test]
    fn lock_mapping_resolves_value_signal() {
        let mapping = lock_mapping(64);
        let result = mapping.resolve(&["value_3"], 0, 1);
        assert_eq!(result, Some(StatePath::LockValue { col: 0, row: 1, idx: 3 }));
    }

    #[test]
    fn lock_mapping_resolves_op_signal() {
        let mapping = lock_mapping(64);
        let result = mapping.resolve(&["lock_op_7"], 0, 1);
        assert_eq!(result, Some(StatePath::LockOp { col: 0, row: 1, idx: 7 }));
    }

    #[test]
    fn lock_mapping_rejects_out_of_range() {
        let mapping = lock_mapping(16);
        // Index 16 is out of range for 16 locks (0-15).
        assert_eq!(mapping.resolve(&["value_16"], 0, 1), None);
        // Index 15 is the last valid index.
        assert_eq!(
            mapping.resolve(&["value_15"], 0, 1),
            Some(StatePath::LockValue { col: 0, row: 1, idx: 15 })
        );
    }

    #[test]
    fn lock_mapping_enumerates_all() {
        let mapping = lock_mapping(16);
        let paths = mapping.enumerate(0, 3);
        // 16 values + 16 ops = 32
        assert_eq!(paths.len(), 32);
        assert!(paths.contains(&StatePath::LockValue { col: 0, row: 3, idx: 0 }));
        assert!(paths.contains(&StatePath::LockOp { col: 0, row: 3, idx: 15 }));
    }

    #[test]
    fn lock_mapping_64_for_memtile() {
        let mapping = lock_mapping(64);
        let paths = mapping.enumerate(0, 1);
        // 64 values + 64 ops = 128
        assert_eq!(paths.len(), 128);
    }

    #[test]
    fn lock_mapping_scope_name() {
        // The VCD scope name must be "locks" so the tree builder can match it.
        let mapping = lock_mapping(16);
        assert_eq!(mapping.scope_name(), "locks");
    }

    #[test]
    fn lock_mapping_first_and_last_index() {
        // Boundary check: index 0 must resolve for all lock counts.
        let mapping = lock_mapping(16);
        assert_eq!(
            mapping.resolve(&["value_0"], 1, 2),
            Some(StatePath::LockValue { col: 1, row: 2, idx: 0 })
        );
        assert_eq!(
            mapping.resolve(&["lock_op_0"], 1, 2),
            Some(StatePath::LockOp { col: 1, row: 2, idx: 0 })
        );
    }

    #[test]
    fn lock_mapping_tile_coordinates_propagated() {
        // Verify that different (col, row) inputs produce distinct paths.
        let mapping = lock_mapping(16);
        let a = mapping.resolve(&["value_5"], 0, 1).unwrap();
        let b = mapping.resolve(&["value_5"], 2, 3).unwrap();
        assert_ne!(a, b);
        assert_eq!(a, StatePath::LockValue { col: 0, row: 1, idx: 5 });
        assert_eq!(b, StatePath::LockValue { col: 2, row: 3, idx: 5 });
    }

    #[test]
    fn lock_mapping_unknown_signal_returns_none() {
        let mapping = lock_mapping(16);
        assert_eq!(mapping.resolve(&["some_other_signal"], 0, 0), None);
        assert_eq!(mapping.resolve(&["value"], 0, 0), None);
        assert_eq!(mapping.resolve(&["lock_op"], 0, 0), None);
    }
}
