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

Landed 2026-04-18. Tag: `phase1-subsys-tile-topo`.

### Commits (Task 1 through tag)

```
d55a2e9 refactor: delete TileType enum, From bridge, and re-exports
64b2740 refactor: tighten predicate discipline after Task 6 rename sweep
9a0ba91 refactor: rename TileType->TileKind across 23 consumer files
d63f734 refactor: route memory-neighbor row>0 guards through archspec SHIM_ROW
93b5757 refactor: migrate bare row==0 tile-classification to archspec SHIM_ROW
eaa169d refactor(archspec): add const-fn is_shim / is_mem / is_compute to TileKind
ac898f7 refactor(archspec): address Task 2 code-review fixups
7a30452 refactor(archspec): add TileTopology trait + Aie2Topology impl
c7e9d98 docs: Subsystem 2 audit + tile-topology design note scaffolds
4e41c3d docs: Subsystem 2 implementation plan -- tile topology
3d76532 docs: pin the two Subsystem 2 deferred-to-plan choices
679af58 docs: Subsystem 2 design spec -- tile topology trait + TileKind rename
```

### Verification (at tag)

- `cargo test --lib`: 2708 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 236 passed; 1 failed (pre-existing `test_full_parse_all_devices`); 2 ignored.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: matches phase1-subsys-isa-decode baseline.
  - Chess: 63 bridge pass, 1 bridge fail; HW 63 pass, 1 fail.
  - Peano: 51 bridge pass, 3 bridge fail (2 pre-existing timeouts + 1 XFAIL); HW 53 pass, 1 fail, 1 XFAIL.
  - All failures are pre-existing: `bd_chain_repeat_on_memtile` EMU deadlock (both compilers), `dma_task_large_linear` + `objectfifo_repeat/init_values_repeat` Peano EMU timeouts, `objectfifo_repeat/distribute_repeat` XFAIL.
- ISA test suite: 4815 / 4815 PASS (100.0%). FAIL: 0.

### Success criteria sweep

- Bare `row == 0` tile-classification hardcodes in src/: 0.
- Bare `row > 0` memory-neighbor hardcodes in src/: 0.
- `TileType::` references in src/ (outside src/trace/vcd.rs): 0.
- `TileType` enum definition in src/device/tile/core_state.rs: deleted.
- `From<TileKind> for TileType` / `From<TileType> for TileKind`: deleted.
- `src/device/tile/mod.rs` / `src/device/mod.rs` re-exports of `TileType`: deleted.
- `xdna_archspec::topology::{TileTopology, Direction}` populated.
- `xdna_archspec::aie2::topology::Aie2Topology` populated.
- `ArchModel::topology()` accessor: returns `Box<dyn TileTopology>` dispatched on `Architecture` (impl block lives in `topology.rs` due to a `#[path]`-include constraint in `build.rs`; see the design note for rationale).
- `TileKind::is_shim()` / `is_mem()` / `is_compute()` const-fn predicates: populated.
- `docs/arch/tile-topology.md` design note: written.

### Net code delta

- New in archspec: ~180 LOC (topology.rs + aie2/topology.rs + TileKind predicates + tests).
- Deleted in xdna-emu: ~200 LOC (TileType enum, bridge, predicate methods, round-trip tests).
- Archspec test count: 220 -> 236 (+16: 11 Aie2Topology tests + 1 ArchModel::topology integration test + 4 TileKind predicate tests).
- xdna-emu test count: 2712 -> 2708 (-4 deleted bridge round-trip tests).
- Call-site rewrites: 2 bare `row == 0` hardcodes + 4 `row > 0` memory-neighbor sites + ~182 `TileType::` rename occurrences across ~13 files (final count after sweep: 23 files touched, because 10 more had field-access-only patterns not caught by the initial `TileType::` grep).

### Follow-ups flagged

Follow-ups that fit naturally in later work, NOT blocking:

- **Subsystem 5 (Stream Switch):** Revisit whether `TileTopology` grows a `shim_mux_kind(col) -> ShimMuxKind` method when stream-switch legality checks need per-column ShimNoc / ShimPl granularity beyond `classify(col, 0)`.
- **Subsystem 7 (ISA Execute):** Thread a full `TileTopology` handle through memory-subsystem code if AIE1's alternating-row adjacency needs dispatch. Today the memory-neighbor sites use the archspec `SHIM_ROW` constant, preserving AIE2 behavior exactly.
- **Subsystem 3 (DMA Engine):** Existing silent DMA channel fallback `(2, 2)` in `from_arch_model()` was flagged in the Phase 1a audit; unchanged in Subsystem 2.
- **Phase 2 hygiene:** The `tile_kind_from_row` helper in `src/device/registers.rs:610-616` could move to `TileTopology::classify` for consistency; deferred because it's currently working correctly.
