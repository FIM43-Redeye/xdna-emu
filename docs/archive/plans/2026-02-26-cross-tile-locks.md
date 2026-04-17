# Cross-Tile Lock Access Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable MemTile DMA engines to acquire/release locks on neighbor tiles (west col-1, east col+1), matching the real NPU's 192-entry lock address space.

**Architecture:** Add `LockTarget` enum and `NeighborLocks` struct to the DMA engine interface. `resolve_lock_id()` returns `LockTarget` instead of `Option<u8>`. Lock operations route through the target to the correct tile's lock array. The array level constructs `NeighborLocks` using safe disjoint-borrow patterns (`split_at_mut`). Non-MemTile tiles pass `NeighborLocks::empty()` with zero overhead.

**Tech Stack:** Rust, no new dependencies. Safe Rust throughout (no `unsafe`).

**Hardware reference:** mlir-aie `getLockLocalBaseIndex()` in `AIETargetModel.cpp`, aie-rt `_XAieMl_DmaSetLock()` in `xaie_dma_aieml.c`, aie-rt `NumLocks=192` for MemTile in `xaiemlgbl_reginit.c`.

---

### Task 1: Add `LockTarget` enum and `NeighborLocks` struct

**Files:**
- Modify: `src/device/dma/engine.rs` (top of file, near line 1-50 imports/types area)
- Modify: `src/device/dma/mod.rs:75` (pub use exports)

**Step 1: Add `LockTarget` enum to engine.rs**

After the existing `use` block and before the `DmaEngine` struct, add:

```rust
/// Identifies which tile's locks a DMA lock operation targets.
///
/// MemTile DMA BDs use an 8-bit lock ID field addressing 192 entries
/// across three tiles (per mlir-aie getLockLocalBaseIndex):
///   - IDs   0- 63: West column MemTile (col-1) locks
///   - IDs  64-127: Own MemTile locks (local_id = lock_id - 64)
///   - IDs 128-191: East column MemTile (col+1) locks
///
/// Compute/shim tiles use a 4-bit field (0-15), always Own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockTarget {
    /// Lock on own tile (local lock index).
    Own(u8),
    /// Lock on west neighbor MemTile, col-1 (local lock index on that tile).
    West(u8),
    /// Lock on east neighbor MemTile, col+1 (local lock index on that tile).
    East(u8),
}
```

**Step 2: Add `NeighborLocks` struct to engine.rs**

```rust
/// Provides mutable access to neighbor tiles for cross-tile lock operations.
///
/// Models the NPU interconnect that routes MemTile DMA lock accesses to
/// neighbor columns. Constructed by the array level using disjoint borrows.
/// Non-MemTile tiles pass `NeighborLocks::empty()`.
pub struct NeighborLocks<'a> {
    /// West neighbor MemTile (col-1), if it exists.
    pub west: Option<&'a mut Tile>,
    /// East neighbor MemTile (col+1), if it exists.
    pub east: Option<&'a mut Tile>,
}

impl NeighborLocks<'_> {
    /// Create an empty neighbor context (no cross-tile access).
    ///
    /// Used for compute/shim tiles where cross-tile lock access
    /// is not applicable.
    pub fn empty() -> NeighborLocks<'static> {
        NeighborLocks { west: None, east: None }
    }
}
```

**Step 3: Add pub use to mod.rs**

Add `LockTarget` and `NeighborLocks` to the `pub use engine::` line in `src/device/dma/mod.rs:75`.

**Step 4: Run `cargo test --lib` to verify compilation**

Expected: All existing tests pass, new types are just additive.

**Step 5: Commit**

```
feat(dma): add LockTarget enum and NeighborLocks struct for cross-tile lock access
```

---

### Task 2: Change `resolve_lock_id` to return `LockTarget`

**Files:**
- Modify: `src/device/dma/engine.rs:458-496` (resolve_lock_id and resolve_lock_id_static)

**Step 1: Write a unit test for the new resolve behavior**

Add to the `#[cfg(test)] mod tests` block at the bottom of engine.rs:

```rust
#[test]
fn test_resolve_lock_id_memtile() {
    // MemTile: 64 locks, 192-entry address space
    let tile_type = TileType::MemTile;
    let num_locks = 64;

    // West neighbor: IDs 0-63
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 0),
        Some(LockTarget::West(0))
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 63),
        Some(LockTarget::West(63))
    );

    // Own tile: IDs 64-127
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 64),
        Some(LockTarget::Own(0))
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 127),
        Some(LockTarget::Own(63))
    );

    // East neighbor: IDs 128-191
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 128),
        Some(LockTarget::East(0))
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 191),
        Some(LockTarget::East(63))
    );

    // Out of range
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 192),
        None
    );
}

#[test]
fn test_resolve_lock_id_compute() {
    // Compute tiles: 4-bit field, always Own
    let tile_type = TileType::Compute;
    let num_locks = 16;
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_type, 1, 2, num_locks, 5),
        Some(LockTarget::Own(5))
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_resolve_lock_id_memtile`
Expected: Fails because resolve_lock_id_static returns `Option<u8>`, not `Option<LockTarget>`.

**Step 3: Change return type and implementation**

Change both `resolve_lock_id` and `resolve_lock_id_static` from returning `Option<u8>` to `Option<LockTarget>`. Make `resolve_lock_id_static` public (tests need it).

```rust
fn resolve_lock_id(&self, lock_id: u8) -> Option<LockTarget> {
    Self::resolve_lock_id_static(self.tile_type, self.col, self.row, self.num_locks, lock_id)
}

pub fn resolve_lock_id_static(tile_type: TileType, col: u8, row: u8, num_locks: u8, lock_id: u8) -> Option<LockTarget> {
    if !tile_type.is_mem_tile() {
        return Some(LockTarget::Own(lock_id));
    }

    if lock_id < num_locks {
        Some(LockTarget::West(lock_id))
    } else if lock_id < num_locks * 2 {
        Some(LockTarget::Own(lock_id - num_locks))
    } else if lock_id < num_locks * 3 {
        Some(LockTarget::East(lock_id - num_locks * 2))
    } else {
        log::warn!(
            "DMA tile({},{}) lock_id={} out of {}-entry address space",
            col, row, lock_id, num_locks as u16 * 3
        );
        None
    }
}
```

Note: The old `log::warn!` about "not yet implemented" is removed -- cross-tile is now a valid target, not an error.

**Step 4: Fix compilation errors in callers**

Both `try_acquire_lock` and `complete_transfer` currently match `Some(local_id)` from resolve_lock_id. They will fail to compile because the return type changed. DO NOT fix the logic yet -- just make it compile by extracting the local_id from `LockTarget::Own(id)` and treating West/East as `return false` (acquire) or skip (release). This preserves existing behavior while changing the type.

In `try_acquire_lock` (~line 1913):
```rust
let local_id = match self.resolve_lock_id(lock_id) {
    Some(LockTarget::Own(id)) => id,
    Some(_) => return false, // Cross-tile: handled in Task 4
    None => return false,
};
```

In `complete_transfer` (~line 1807):
```rust
if let Some(LockTarget::Own(local_id)) = Self::resolve_lock_id_static(...) {
    // existing release logic unchanged
}
// West/East silently skip release for now -- handled in Task 4
```

**Step 5: Run tests**

Run: `cargo test --lib`
Expected: All tests pass including the new resolve_lock_id tests. Existing behavior preserved.

**Step 6: Commit**

```
refactor(dma): change resolve_lock_id to return LockTarget enum
```

---

### Task 3: Add `NeighborLocks` parameter to `step()` call chain

This is the biggest mechanical task -- threading the `neighbors` parameter through the call chain. No behavioral change yet.

**Files:**
- Modify: `src/device/dma/engine.rs` (5 function signatures + 1 internal call + 7 test call sites)
- Modify: `src/device/array.rs:279-307` (step_dma, step_all_dma)
- Modify: `src/npu/executor.rs:748` (step_dma call site)

**Step 1: Change `step()` signature and propagate through call chain**

Change these 5 signatures in engine.rs (add `neighbors: &mut NeighborLocks<'_>` after `tile: &mut Tile`):

1. `pub fn step(&mut self, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>, host_memory: &mut HostMemory) -> DmaResult` (line 767)
2. `fn step_channel(&mut self, ch_idx: usize, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>, host_memory: &mut HostMemory)` (line 840)
3. `fn step_channel_timed(&mut self, ch_idx: usize, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>, host_memory: &mut HostMemory)` (line 871)
4. `fn complete_transfer(&mut self, ch_idx: usize, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>)` (line 1792)
5. `fn try_acquire_lock(&mut self, ch_idx: usize, lock_id: u8, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>) -> bool` (line 1909)

Also update `execute_1d_transfer` (line 1983) -- add neighbors param and pass through:
```rust
pub fn execute_1d_transfer(
    &mut self, channel: ChannelId, bd_index: u8,
    tile: &mut Tile, neighbors: &mut NeighborLocks<'_>,
    host_memory: &mut HostMemory,
) -> Result<u64, DmaError> {
    // ...
    self.step(tile, neighbors, host_memory);
    // ...
}
```

**Step 2: Update internal call sites within engine.rs**

There are internal calls between these functions. Thread `neighbors` through each:
- `step()` calls `self.try_acquire_lock(ch_idx, lock_id, tile, neighbors)` (line 792)
- `step()` calls `self.step_channel(ch_idx, tile, neighbors, host_memory)` (line 787)
- `step_channel()` calls `self.step_channel_timed(ch_idx, tile, neighbors, host_memory)` (line 863)
- `step_channel_timed()` calls `self.complete_transfer(ch_idx, tile, neighbors)` (all call sites within step_channel_timed)

**Step 3: Update test call sites in engine.rs**

All 7 `engine.step(&mut tile, &mut host_mem)` calls in tests become `engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem)`. The 1 `execute_1d_transfer` call also gets `&mut NeighborLocks::empty()`.

**Step 4: Update array.rs call sites**

For now, pass `NeighborLocks::empty()` at the array level too (actual neighbor construction comes in Task 5).

In `step_dma` (line 283):
```rust
Some(engine.step(tile, &mut NeighborLocks::empty(), host_memory))
```

In `step_all_dma` (line 299):
```rust
let result = self.dma_engines[i].step(&mut self.tiles[i], &mut NeighborLocks::empty(), host_memory);
```

**Step 5: Update executor.rs call site**

In `src/npu/executor.rs:748`, `step_dma` is called on `device.array` which already takes (col, row, host_mem). Since step_dma itself is updated in Step 4, no change needed here -- `step_dma` wraps the call.

**Step 6: Run tests**

Run: `cargo test --lib`
Expected: All tests pass. No behavioral change -- just plumbing.

**Step 7: Commit**

```
refactor(dma): thread NeighborLocks parameter through DMA step call chain
```

---

### Task 4: Implement cross-tile lock acquire and release

This is the core behavioral change. Route lock operations through `LockTarget` to the correct tile.

**Files:**
- Modify: `src/device/dma/engine.rs` (try_acquire_lock ~line 1909, complete_transfer ~line 1792)

**Step 1: Write test for cross-tile lock acquire**

Add to engine.rs tests. This needs a MemTile engine with locks configured on a neighbor tile.

```rust
#[test]
fn test_cross_tile_lock_acquire_west() {
    use crate::device::tile::TileParams;

    // Create MemTile DMA engine at col 1, row 1
    let mut engine = DmaEngine::new(
        TileType::MemTile,
        1, 1,      // col, row
        16,        // num_bds
        12,        // num_channels (6 S2MM + 6 MM2S)
        64,        // num_locks
    );

    // Create own tile (MemTile with 512KB memory)
    let own_params = TileParams {
        tile_type: TileType::MemTile,
        memory_size: 512 * 1024,
        num_locks: 64,
        col: 1, row: 1,
        ..Default::default()
    };
    let mut own_tile = Tile::new(own_params);

    // Create west neighbor tile
    let west_params = TileParams {
        tile_type: TileType::MemTile,
        memory_size: 512 * 1024,
        num_locks: 64,
        col: 0, row: 1,
        ..Default::default()
    };
    let mut west_tile = Tile::new(west_params);

    // Set west tile's lock 5 to value 1 (will be acquired)
    west_tile.locks[5].value = 1;

    // Configure BD with acquire on west neighbor lock 5
    // West locks are IDs 0-63, so lock_id=5 means west lock 5
    let mut bd = BdConfig::simple_1d(0x100, 32);
    bd.acquire_lock = Some(5);   // lock_id 5 = west neighbor lock 5
    bd.acquire_value = 1;        // acq_eq: wait for value == 1
    engine.configure_bd(0, bd).unwrap();

    // Write data to own tile memory
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xAA; 32]);

    // Start MM2S channel (channel 6 for MemTile)
    engine.start_channel(6, 0).unwrap();
    assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)));

    // Take snapshots (required for snapshot-based lock semantics)
    own_tile.begin_lock_cycle();
    west_tile.begin_lock_cycle();

    // Step with neighbor access -- lock should be acquired
    let mut neighbors = NeighborLocks {
        west: Some(&mut west_tile),
        east: None,
    };
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    // Channel should now be active (lock acquired from west neighbor)
    assert_eq!(engine.channel_state(6), ChannelState::Active,
        "Channel should be active after acquiring west neighbor lock");
}

#[test]
fn test_cross_tile_lock_acquire_fails_without_neighbor() {
    // MemTile at col 0 has no west neighbor
    let mut engine = DmaEngine::new(
        TileType::MemTile,
        0, 1, 16, 12, 64,
    );

    let own_params = TileParams {
        tile_type: TileType::MemTile,
        memory_size: 512 * 1024,
        num_locks: 64,
        col: 0, row: 1,
        ..Default::default()
    };
    let mut own_tile = Tile::new(own_params);
    own_tile.begin_lock_cycle();

    let mut bd = BdConfig::simple_1d(0x100, 32);
    bd.acquire_lock = Some(5); // West lock -- but no west neighbor at col 0
    bd.acquire_value = 1;
    engine.configure_bd(0, bd).unwrap();

    engine.start_channel(6, 0).unwrap();

    let mut neighbors = NeighborLocks::empty(); // No neighbors
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    // Should remain waiting -- no neighbor to satisfy lock
    assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)),
        "Should stay waiting when neighbor tile is absent");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_cross_tile_lock_acquire`
Expected: First test FAILS (cross-tile acquire returns false), second may pass (already returns false).

**Step 3: Implement cross-tile lock acquire in `try_acquire_lock`**

Replace the placeholder `Some(_) => return false` from Task 2 with actual cross-tile routing:

```rust
fn try_acquire_lock(&mut self, ch_idx: usize, lock_id: u8, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>) -> bool {
    use crate::device::tile::LockResult;

    let lock_target = match self.resolve_lock_id(lock_id) {
        Some(target) => target,
        None => return false, // Invalid lock ID
    };

    // Route to the correct tile based on lock target
    let (target_tile, local_id): (&mut Tile, u8) = match lock_target {
        LockTarget::Own(id) => (tile, id),
        LockTarget::West(id) => match neighbors.west.as_deref_mut() {
            Some(west) => (west, id),
            None => {
                log::warn!(
                    "DMA tile({},{}) lock_id={} targets west neighbor but none exists (col=0?)",
                    self.col, self.row, lock_id
                );
                return false;
            }
        },
        LockTarget::East(id) => match neighbors.east.as_deref_mut() {
            Some(east) => (east, id),
            None => {
                log::warn!(
                    "DMA tile({},{}) lock_id={} targets east neighbor but none exists",
                    self.col, self.row, lock_id
                );
                return false;
            }
        },
    };

    if (local_id as usize) >= target_tile.locks.len() {
        log::warn!(
            "DMA try_acquire_lock tile({},{}) lock_id={} resolved to {:?} local_id={} \
             but target tile only has {} locks",
            self.col, self.row, lock_id, lock_target, local_id, target_tile.locks.len()
        );
        return false;
    }

    // ... rest of acquire logic (same as current, but using target_tile instead of tile) ...
}
```

The body after the routing is the same as the current implementation, but every `tile.` reference for lock operations becomes `target_tile.`.

**Step 4: Implement cross-tile lock release in `complete_transfer`**

Same pattern. Replace the `if let Some(LockTarget::Own(local_id))` with full routing:

```rust
if let TransferState::ReleasingLock(lock_id) = transfer.state {
    let release_delta = transfer.release_value;
    if let Some(lock_target) = Self::resolve_lock_id_static(self.tile_type, self.col, self.row, self.num_locks, lock_id) {
        let (target_tile, local_id) = match lock_target {
            LockTarget::Own(id) => (&mut *tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref_mut() {
                Some(west) => (west, id),
                None => {
                    log::warn!("DMA tile({},{}) release lock_id={} targets west neighbor but none exists",
                        self.col, self.row, lock_id);
                    // Skip release -- transfer continues
                    transfer.lock_released();
                    // ... (need to move on past the if-let block)
                    // Use a helper or restructure
                }
            },
            LockTarget::East(id) => match neighbors.east.as_deref_mut() {
                Some(east) => (east, id),
                None => { /* same as West None case */ }
            },
        };

        // Release on target tile
        if (local_id as usize) < target_tile.locks.len() {
            target_tile.release_snapshot(local_id as usize, release_delta);
            // ... logging ...
        }
    }
    transfer.lock_released();
}
```

Note: The exact refactoring of complete_transfer will need care with the borrow checker since `transfer` is `&mut self.transfers[ch_idx]` and we also pass `neighbors`. Consider extracting lock release into a helper function.

**Step 5: Run tests**

Run: `cargo test --lib`
Expected: All tests pass including the new cross-tile tests.

**Step 6: Commit**

```
feat(dma): implement cross-tile lock acquire and release for MemTile DMA
```

---

### Task 5: Wire neighbor tiles at the array level

**Files:**
- Modify: `src/device/array.rs:279-307` (step_dma, step_all_dma)

**Step 1: Write test for array-level cross-tile lock routing**

Add to array.rs tests (or a new integration test):

```rust
#[test]
fn test_memtile_cross_tile_lock_via_array() {
    // Create a 2-column array and verify that MemTile DMA at col 1
    // can acquire a lock on the MemTile at col 0.
    // This is an integration test exercised through step_all_dma.
    // (Details depend on array construction helpers available in tests)
}
```

The exact test shape depends on what array construction helpers exist. The key assertion: after `step_all_dma`, a MemTile DMA channel waiting on a cross-tile lock transitions to Active when the neighbor tile's lock has the required value.

**Step 2: Implement neighbor construction in `step_all_dma`**

Replace the current simple loop with MemTile-aware neighbor construction.

The tiles Vec uses `col * rows + row` indexing. For MemTile at index `i` with coordinates `(col, row)`:
- West neighbor index: `(col-1) * rows + row` (exists if `col > 0`)
- East neighbor index: `(col+1) * rows + row` (exists if `col < cols-1`)
- Only MemTile row (row=1 for NPU1) needs neighbor access

Use safe `split_at_mut` to get disjoint mutable borrows from the tiles Vec:

```rust
pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
    let mut any_active = false;
    let rows = self.rows as usize;
    let cols = self.cols as usize;

    for i in 0..self.tiles.len() {
        self.tiles[i].reset_bank_tracking();
        self.dma_engines[i].cycle_dma_banks = 0;

        // Determine if this engine needs cross-tile lock access
        let is_mem_tile = self.dma_engines[i].tile_type().is_mem_tile();

        let result = if is_mem_tile {
            let col = i / rows;
            let row = i % rows;
            // Construct NeighborLocks from disjoint mutable borrows
            let mut neighbors = self.build_neighbor_locks(col, row);
            self.dma_engines[i].step(&mut self.tiles[i], &mut neighbors, host_memory)
        } else {
            self.dma_engines[i].step(&mut self.tiles[i], &mut NeighborLocks::empty(), host_memory)
        };

        if matches!(result, DmaResult::InProgress | DmaResult::WaitingForLock(_)) {
            any_active = true;
        }
        self.tiles[i].cycle_dma_banks |= self.dma_engines[i].cycle_dma_banks;
    }
    any_active
}
```

The `build_neighbor_locks` helper needs to produce `NeighborLocks` with mutable refs into `self.tiles` at disjoint indices. This requires careful borrow management.

**Step 3: Implement `build_neighbor_locks` helper**

This is the trickiest part. We need `&mut tiles[own]`, `&mut tiles[west]`, `&mut tiles[east]` simultaneously. Since they're at different indices (separated by `rows`), we can use `split_at_mut`:

```rust
/// Build NeighborLocks for a MemTile at (col, row).
///
/// Returns NeighborLocks with mutable references to west (col-1) and east (col+1)
/// tiles, using safe split_at_mut for disjoint borrows.
///
/// IMPORTANT: The caller must NOT hold a mutable borrow to self.tiles[own_idx]
/// at the same time. This method is designed to be called where the own tile
/// borrow is handled separately (e.g., via index-based access after this call).
fn build_neighbor_locks(&mut self, col: usize, row: usize) -> NeighborLocks<'_> {
    let rows = self.rows as usize;
    let cols = self.cols as usize;
    let own_idx = col * rows + row;

    // MemTile neighbors are always in the same row, adjacent columns
    let west_idx = if col > 0 { Some((col - 1) * rows + row) } else { None };
    let east_idx = if col + 1 < cols { Some((col + 1) * rows + row) } else { None };

    // Only provide neighbors that are also MemTiles
    let west_idx = west_idx.filter(|&idx| self.tiles[idx].tile_type().is_mem_tile());
    let east_idx = east_idx.filter(|&idx| self.tiles[idx].tile_type().is_mem_tile());

    // ... construct via split_at_mut or raw pointer approach ...
}
```

**Borrow challenge:** `build_neighbor_locks` returns refs into `self.tiles`, but `step_all_dma` also needs `&mut self.tiles[i]` for the engine step. This means we can't use this exact approach.

**Alternative: restructure the loop to avoid the helper.** Destructure `self` to get disjoint field borrows, then use `split_at_mut` inline:

```rust
pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
    let mut any_active = false;
    let rows = self.rows as usize;
    let cols = self.cols as usize;

    // Destructure for disjoint field borrows
    let tiles = &mut self.tiles;
    let engines = &mut self.dma_engines;

    for i in 0..tiles.len() {
        tiles[i].reset_bank_tracking();
        engines[i].cycle_dma_banks = 0;

        let is_mem_tile = engines[i].tile_type().is_mem_tile();

        let result = if is_mem_tile {
            let col = i / rows;
            // Get disjoint mutable borrows for own + west + east
            // Indices are: west = i - rows, own = i, east = i + rows
            // These are guaranteed distinct (separated by rows >= 1)
            let (west_ref, own_ref, east_ref) = get_three_mut(tiles, i, col, rows, cols);
            let mut neighbors = NeighborLocks { west: west_ref, east: east_ref };
            engines[i].step(own_ref, &mut neighbors, host_memory)
        } else {
            engines[i].step(&mut tiles[i], &mut NeighborLocks::empty(), host_memory)
        };

        if matches!(result, DmaResult::InProgress | DmaResult::WaitingForLock(_)) {
            any_active = true;
        }
        tiles[i].cycle_dma_banks |= engines[i].cycle_dma_banks;
    }
    any_active
}
```

Where `get_three_mut` is a safe helper using split_at_mut:

```rust
/// Get mutable references to up to 3 tiles at disjoint indices.
///
/// Returns (west_neighbor, own_tile, east_neighbor) where neighbors
/// are None if they don't exist or aren't MemTiles.
fn get_three_mut(
    tiles: &mut [Tile],
    own_idx: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> (Option<&mut Tile>, &mut Tile, Option<&mut Tile>) {
    let west_idx = if col > 0 { Some(own_idx - rows) } else { None };
    let east_idx = if col + 1 < cols { Some(own_idx + rows) } else { None };

    // Filter to MemTile neighbors only
    let west_idx = west_idx.filter(|&idx| tiles[idx].tile_type().is_mem_tile());
    let east_idx = east_idx.filter(|&idx| tiles[idx].tile_type().is_mem_tile());

    match (west_idx, east_idx) {
        (None, None) => {
            (None, &mut tiles[own_idx], None)
        }
        (Some(w), None) => {
            // w < own_idx guaranteed (west is lower column)
            let (left, right) = tiles.split_at_mut(own_idx);
            (Some(&mut left[w]), &mut right[0], None)
        }
        (None, Some(e)) => {
            // own_idx < e guaranteed (east is higher column)
            let (left, right) = tiles.split_at_mut(e);
            (None, &mut left[own_idx], Some(&mut right[0]))
        }
        (Some(w), Some(e)) => {
            // w < own_idx < e guaranteed
            let (left, rest) = tiles.split_at_mut(own_idx);
            let (mid, right) = rest.split_at_mut(e - own_idx);
            (Some(&mut left[w]), &mut mid[0], Some(&mut right[0]))
        }
    }
}
```

**Step 4: Update `step_dma` similarly**

Apply the same pattern to `step_dma(col, row, host_memory)`.

**Step 5: Run tests**

Run: `cargo test --lib`
Expected: All tests pass. Cross-tile lock warnings should vanish from test output.

**Step 6: Commit**

```
feat(dma): wire NeighborLocks construction at array level with safe disjoint borrows
```

---

### Task 6: Add DmaEngine accessor for tile_type

**Files:**
- Modify: `src/device/dma/engine.rs`

**Step 1: Add `tile_type()` accessor if not already present**

Task 5 uses `engines[i].tile_type()`. Check if this method exists. If not, add:

```rust
/// Get the tile type for this DMA engine.
pub fn tile_type(&self) -> TileType {
    self.tile_type
}
```

**Step 2: Run `cargo test --lib`**

Expected: passes.

**Step 3: Commit** (can be folded into Task 5's commit)

---

### Task 7: Verify with emulator test suite

**Files:** None modified -- this is a validation step.

**Step 1: Run the full emulator test suite**

Run: `cargo run --bin npu-test -- --no-build 2>&1 | tee /tmp/npu-test-results-post-cross-tile.log`

**Step 2: Compare results against baseline**

Baseline (2026-02-26):
- 19 pass, 17 validation fail, 32 timeout
- 300K cross-tile lock warnings

Expected improvements:
- Cross-tile lock warnings should be eliminated (0 warnings)
- Some timeouts should convert to pass or validation-fail (progress!)
- No regressions (anything that passed before should still pass)

**Step 3: Document results**

Update MEMORY.md with new test counts and any interesting findings.

**Step 4: Commit any documentation updates**

```
docs: update test results after cross-tile lock access implementation
```
