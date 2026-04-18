# Subsystem 2 -- Tile Topology Audit

## Baseline (pre-subsystem, at phase1-subsys-isa-decode tag)

- `cargo test --lib`: test result: ok. 2712 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 2.41s
- `cargo test -p xdna-archspec --lib`: test result: FAILED. 220 passed; 1 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.36s

Known pre-existing failures (carry through):
- `test_full_parse_all_devices` (archspec, pre-existing, device count 13 vs expected 12).
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit facts

### TileKind (archspec)
- Defined at `crates/xdna-archspec/src/types.rs:62-77`.
- 4 variants: `Compute`, `Mem`, `ShimNoc`, `ShimPl`.
- `Display` impl emits `compute` / `mem` / `shim_noc` / `shim_pl`.
- Derives `Serialize`, `Deserialize`, `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`.
- No predicate methods today.

### TileType (xdna-emu)
- Defined at `src/device/tile/core_state.rs:49-77`.
- 3 variants: `Shim`, `MemTile`, `Compute`.
- Predicate methods `is_shim()`, `is_mem_tile()`, `is_compute()` at lines 60-78.
- Re-exported publicly at `src/device/tile/mod.rs:38` and `src/device/mod.rs:71`.
- No `Serialize` / `Deserialize`; no FFI exposure.

### Bridge
- `From<TileKind> for TileType` at `src/device/tile/core_state.rs:80-92` (lossy: `ShimNoc | ShimPl -> Shim`).
- `From<TileType> for TileKind` at `src/device/tile/core_state.rs:94-106` (`Shim -> ShimNoc`; cannot recover `ShimPl`).
- Round-trip tests at `src/device/tile/core_state.rs:127-165` (4 tests).

### Bare `row == 0` classification hardcodes (to fix in Task 4)
- `src/npu/executor.rs:895`: `row == 0 && bd_index_for_blockwrite(row, offset).is_some()` in the shim-BD layout dispatch.
- `src/device/array/routing.rs:144`: `if row == 0 { continue; }` in the cascade south-routing block (defensive underflow guard).

### Memory-neighbor `row > 0` sites (to migrate in Task 5)
- `src/interpreter/execute/memory/neighbor.rs:80`: `neighbor_coords(MemoryQuadrant::South)` south-neighbor existence check.
- `src/interpreter/execute/memory/neighbor.rs:135`: same intent inside `apply_writes`.
- `src/interpreter/engine/coordinator.rs:578`: south-lock snapshot guard.
- `src/interpreter/engine/coordinator.rs:705`: south-lock writeback guard.

### Row checks that already route through archspec constants (leave alone)
- `src/device/registers.rs:610-616` (`tile_kind_from_row`).
- `src/device/array/routing.rs:672` (`is_shim` via `SHIM_ROW`).
- `src/interpreter/test_runner.rs:39, 50, 61, 72, 266, 286` (port-range and DMA-channel dispatch).

### `TileType::` references to rewrite in Task 6
- Total: ~182 occurrences across ~13 files. Hotspots:
  - `src/device/tile/mod.rs` (~33)
  - `src/device/dma/bd.rs` (~22)
  - `src/device/dma/transfer/tests.rs` (~20)
  - `src/device/array/routing.rs` (~18)
  - `src/interpreter/engine/coordinator.rs` (~15)
  - `src/device/tile/core_state.rs` (~15)
  - `src/trace/vcd.rs` (~12 uses of ITS OWN `TileType` enum -- unrelated; leave alone)

### Tile-type-specific named tests (smoke list post-rename)
16 tests covering memtile DMA, shim BD parsing, tile classification, and cascade routing. Full list in the spec Section 5.

## Completion

*(To be filled in by Task 8.)*
