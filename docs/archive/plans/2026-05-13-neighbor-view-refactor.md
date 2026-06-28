# NeighborMemory Stage 2 -- TileLookup + NeighborView + lazy access

**Status:** Stage 1 shipped (commit `e85aa72`). Stage 2 (this doc) deferred
to a fresh context window so we can do it carefully.

## Why this exists

Stage 1 fixed the per-step quadratic memcpy in
`src/interpreter/execute/memory/neighbor.rs` by adding a generation counter
on `Tile.data_memory` and making `NeighborMemory::ensure_snapshot` gen-aware,
plus hoisting `NeighborMemory` onto `CoreState` so the cache survives across
steps. Verified: `add_one_ctrl_packet --sweep --chess-only` EMU arm now
completes cleanly (was wedged at 18+ min/test before, with 9 of 16 workers
pegged at 92% CPU in `__memcpy_avx512` under `ensure_snapshot`).

What Stage 1 LEFT in place: coordinator (`src/interpreter/engine/coordinator.rs`
lines ~595-600) still eagerly calls `ensure_snapshot` for South/West/North on
every step. Each call is now O(1) gen-comparison when the neighbor hasn't
changed -- effectively free -- but the read site still doesn't declare its
own data dependency. That's the architectural smell Stage 2 fixes.

The perf delta of Stage 2 over Stage 1 is essentially noise (3 gen-checks/step
vs 0). The motivation is purely architectural: **the read site should ask
for what it needs, not assume the coordinator pre-decided.**

## Stage 2a -- foundation (~50 LoC)

Adds the abstractions; coordinator still does eager refresh, but via the
new clean borrow-safe path. No threading work yet. Existing tests work
unchanged.

### Pieces

1. **`TileLookup` trait** (new). Place in `src/device/array/mod.rs` next to
   `TileArray`, or `src/device/state/mod.rs` next to `DeviceState`:
   ```rust
   pub trait TileLookup {
       fn tile(&self, col: usize, row: usize) -> Option<&Tile>;
   }
   ```
   - `impl TileLookup for DeviceState` -- delegates to existing `tile()`.
   - `impl<'a> TileLookup for NeighborView<'a>` -- defined below.

2. **`NeighborView<'a>` struct** (new). Read-through to non-own tiles via
   split slices. Place it next to `DeviceState::split_tile_mut`:
   ```rust
   pub struct NeighborView<'a> {
       left: &'a [Tile],   // tiles BEFORE own_idx in the linear vec
       right: &'a [Tile],  // tiles AFTER own_idx
       own_idx: usize,     // hole position (for index translation)
       cols: usize,
       rows: usize,
   }
   ```
   `tile(col, row)` returns `None` for out-of-bounds AND for `own_idx`
   (the executing tile is not accessible through the view -- caller has
   `&mut Tile` for that one).

3. **`DeviceState::split_tile_mut`** (new). Safe split using
   `slice::split_at_mut` + `split_first_mut`. Storage is at
   `self.array.tiles: Vec<Tile>` (already `pub(crate)` in
   `src/device/array/mod.rs:118`). Indexing is `col * rows + row`
   (see `TileArray::tile_index` at `src/device/array/mod.rs:290`):
   ```rust
   pub fn split_tile_mut(&mut self, col: usize, row: usize)
       -> Option<(&mut Tile, NeighborView<'_>)>
   {
       let cols = self.cols();
       let rows = self.rows();
       if col >= cols || row >= rows { return None; }
       let idx = col * rows + row;
       let tiles = &mut self.array.tiles;
       let (left, rest) = tiles.split_at_mut(idx);
       let (own, right) = rest.split_first_mut()?;
       Some((own, NeighborView { left, right, own_idx: idx, cols, rows }))
   }
   ```

4. **`NeighborMemory::ensure_snapshot`** generic over `TileLookup`. Currently
   takes `&crate::device::DeviceState`; change to:
   ```rust
   pub fn ensure_snapshot<T: TileLookup>(&mut self, dir: MemoryQuadrant, source: &T) {
       // body unchanged -- still reads source.tile(c, r) and compares gens
   }
   ```
   Existing tests in `src/interpreter/execute/memory/mod.rs` lines ~2305+
   pass `&device` and continue to work because `DeviceState: TileLookup`.

5. **Coordinator refactor** at `src/interpreter/engine/coordinator.rs`
   lines ~561-727 (the per-step body). Replace the manual scaffolding:
   ```rust
   // Old:
   let core = &mut self.cores[idx];
   core.neighbors.ensure_snapshot(MemoryQuadrant::South, &self.device);
   // ... 2 more ...
   if let Some(tile) = self.device.tile_mut(col, row) { ... }

   // New:
   let Some((tile, view)) = self.device.split_tile_mut(col, row) else { continue; };
   if !tile.is_compute() { continue; }
   let core = &mut self.cores[idx];
   core.neighbors.ensure_snapshot(MemoryQuadrant::South, &view);
   core.neighbors.ensure_snapshot(MemoryQuadrant::West, &view);
   core.neighbors.ensure_snapshot(MemoryQuadrant::North, &view);
   // ... step ...
   ```
   Note: drain_writes still needs `&mut self.device` after step; the view
   borrow ends when the if-let body exits.

### Stage 2a tests

- All existing tests must pass (`cargo test --lib`).
- Add a test that `NeighborView::tile(own_col, own_row)` returns `None`
  (the hole). Place in `device/state/mod.rs` or wherever NeighborView lives.
- Add a test that `DeviceState::split_tile_mut` returns `None` for
  out-of-bounds coordinates.

## Stage 2b -- lazy at access (~200 LoC)

Once 2a is in, thread `Option<&NeighborView>` through the read paths so
`ensure_snapshot` happens at access time, not eagerly in coordinator.
Then remove the eager pre-pop entirely.

### Function signatures to update

19 functions take `Option<&[mut] NeighborMemory>` and need a parallel
`Option<&NeighborView>` parameter. Locations as of commit `e85aa72`:

- `src/interpreter/execute/cycle_accurate.rs`: lines 114, 285, 298, 310 (4)
- `src/interpreter/core/interpreter.rs`: lines 117, 129, 141 (3)
- `src/interpreter/execute/memory/mod.rs`: 12 sites at lines 95, 189, 299,
  378, 449, 535, 913, 1006, 1069, 1138, 1441, 1549, 1722, 1758

The change is mechanical: add `view: Option<&NeighborView>` next to every
existing `neighbors: Option<&[mut] NeighborMemory>` parameter, and pass it
through.

### The 3 leaf read sites

These are the only sites that actually consume the view -- they call
`get_memory` today and need to call `ensure_snapshot` lazily first:

- `memory/mod.rs:587` -- vector read in some `*_step` function
- `memory/mod.rs:1461` -- inside `read_memory(tile, addr, width, neighbors)`
- `memory/mod.rs:1728` -- inside `read_vector_from_memory(tile, addr, neighbors)`

Pattern at each leaf:
```rust
// Old:
} else if let Some(nbr_mem) = neighbors.as_ref().and_then(|n| n.get_memory(quadrant)) {
    nbr_mem
}

// New (assumes neighbors changed to &mut and view was threaded through):
} else if let (Some(n), Some(v)) = (neighbors.as_deref_mut(), view) {
    n.ensure_snapshot(quadrant, v);
    match n.get_memory(quadrant) {
        Some(mem) => mem,
        None => continue,  // or return zero, matching existing semantics
    }
}
```

Convenience method worth adding to `NeighborMemory`:
```rust
pub fn get_or_snapshot<T: TileLookup>(
    &mut self, dir: MemoryQuadrant, source: &T
) -> Option<&[u8]> {
    self.ensure_snapshot(dir, source);
    self.get_memory(dir)
}
```

### Coordinator cleanup (after 2b)

Remove the 3 eager `ensure_snapshot` calls added/kept in 2a. Pass
`Some(&view)` to `step_with_neighbor_locks`. The view's lifetime ends
when the if-let body exits, before drain_writes -- which is what we want.

### Stage 2b tests

The big risk is "I broke a read path." Existing integration tests
(`cargo test --lib`) cover the read sites. Specifically watch:
- `test_neighbor_memory_build_and_read` -- exercises all 4 directions
- `test_local_fast_path_with_neighbors` -- exercises Local quadrant skip
- `test_edge_tile_no_west_neighbor` -- exercises None-on-edge case
- DMA cross-tile tests in `device/dma/engine/tests.rs`

Add one new test asserting that `ensure_snapshot` is NOT called for a
direction the kernel doesn't access. Hard to test directly without
instrumentation -- a `cfg(test)` counter on NeighborMemory works.

## Validation

After both stages:
1. `cargo test --lib` -- 2895+ tests, must be green
2. `cargo build -p xdna-emu-ffi` -- rebuild the FFI cdylib
3. `./scripts/emu-bridge-test.sh --chess-only --sweep -v add_one_ctrl_packet`
   -- the sweep that wedged before Stage 1; must complete cleanly. EMU
   sweep duration should be similar to Stage 1 (no perf regression).
4. `./scripts/emu-bridge-test.sh --chess-only` (no filter, no sweep) on a
   broader cross-section -- catches accidental read-path bugs.

## Gotchas / things-to-watch

- **The `@N s` tag in bridge-test output is cumulative seconds, not per-test
  duration** -- duration column was added in commit `78bb547`. Look at the
  duration column for actual test wall time, not the `@Ns` tag.

- **The borrow-checker concern that justified this design**: at the read
  site, the interpreter holds `&mut Tile` for the executing tile (obtained
  from `&mut self.device.tile_mut(col, row)`). To also call
  `ensure_snapshot`, we need to access OTHER tiles -- can't hold both
  `&mut Tile` and `&DeviceState` simultaneously. `NeighborView` solves
  this via `split_at_mut` + `split_first_mut`: the view holds immut
  references to the OTHER tiles, and `&mut Tile` for own. No conflict.

- **`drain_writes` ordering**: takes `&mut self.device`. Must come AFTER
  the if-let body that holds `&mut tile` (and thus `&mut self.device`).
  Stage 1 already has this right; Stage 2 must preserve it.

- **`continue` in the inner loop**: `if !tile.is_compute() { continue; }`
  jumps to the next iteration of `for row`. Make sure drain_writes still
  runs only when step actually buffered writes (currently fine because
  non-compute tiles never step).

- **Test sites in `mod.rs` lines ~2305-2500** call ensure_snapshot with
  `&device`. These continue to work in 2a (DeviceState: TileLookup).

- **The `add_one_ctrl_packet` HW silent-drop** is a separate, pre-existing
  firmware issue documented in
  `docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`.
  It may TDR during the bridge test HW arm; that's not a Stage 2 regression.

## Out of scope (deliberately)

- The same per-step `clone()` pattern at coordinator.rs lines ~575-593 for
  neighbor LOCKS. Locks are tiny (Vec<Lock>), much smaller cost. Same
  architectural smell, separate refactor when the time comes.
- Static prescan of programs to determine which quadrants are accessed
  (would let us skip ensure_snapshot entirely for unaccessed directions
  even more cheaply -- but premature optimization given Stage 2b already
  achieves "snapshot only on actual access").

## When picking this up in a fresh session

Read in order:
1. This doc
2. `src/interpreter/execute/memory/neighbor.rs` -- current state
3. `src/interpreter/engine/coordinator.rs` lines ~561-727 -- the per-step body
4. The 3 leaf read sites in `src/interpreter/execute/memory/mod.rs`
5. Commit `e85aa72` for Stage 1 context

Implement 2a fully (commit), verify tests, then 2b (commit), then run the
sweep verification. Two clean commits, two reviewable diffs.
