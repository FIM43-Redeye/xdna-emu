# Tile Topology -- Design Note

**Subsystem:** 2 (Phase 1b)
**Tag:** `phase1-subsys-tile-topo`
**Spec:** [../superpowers/specs/2026-04-18-subsys2-tile-topology-design.md](../superpowers/specs/2026-04-18-subsys2-tile-topology-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. Unlike Subsystem 6 (which adds no trait),
Subsystem 2 introduces `TileTopology`: this note explains the shape
difference that justifies the trait and what AIE1 / AIE2P impls will
look like.

---

## What lives where

All entries below are in `xdna-archspec` as of the
`phase1-subsys-tile-topo` tag.

| Data/code | Module | Source |
|-----------|--------|--------|
| `TileKind` enum (4 variants + `is_shim` / `is_mem` / `is_compute` const-fn predicates) | `xdna_archspec::types` | aie-rt XAIE_TILE_TYPE constants |
| `TileTopology` trait + `Direction` enum | `xdna_archspec::topology` | Emulator design |
| `Aie2Topology` concrete impl | `xdna_archspec::aie2::topology` | Built from `ArchModel` extents + `SHIM_ROW` / `COMPUTE_ROW_START` constants |
| `ArchModel::topology()` accessor | `xdna_archspec::topology` (additional `impl ArchModel` block -- `types.rs` is `#[path]`-included by build.rs, which prevents cross-module imports from within it) | Dispatches on `ArchModel::arch` |

No runtime-side `TileType` enum remains; consumers import `TileKind`
directly from archspec.

---

## The shape-vs-values principle, applied to tile topology

Subsystem 1 established: lift per-arch differences behind traits only when
they are *shape* differences, not *values* differences. Subsystem 6
applied the principle and chose no-trait for ISA decode (slot counts and
format layouts are data; the walker algorithm is invariant). Subsystem 2
hits the other side of the same principle: tile topology has genuine
shape differences.

Concretely:

- **AIE2 (NPU1/NPU2/NPU4/NPU5/NPU6):** 5x6 or larger grid; row 0 is
  `ShimNoc` (every shim column); row 1 is `Mem`; rows 2-5 are `Compute`.
  Memory adjacency is uniform-direction for all compute rows.
- **AIE1 (Versal, e.g., xcvc1902):** 50x9 grid; row 0 is shim, but some
  columns are `ShimNoc` and others are `ShimPl` (the PL-fabric variant
  is real, unlike AIE2). There is **no memtile row**
  (`num_mem_tile_rows = 0`). Memory adjacency alternates direction by
  row parity -- compute tiles at odd rows share memory with north, even
  rows share with south. `classify` must dispatch on column, not just
  row; `neighbor` must dispatch on row parity.
- **AIE2P (Strix / Strix Halo / Krackan):** similar shape to AIE2 with
  different extents and potentially different memtile placement.
  Expected to share most of `Aie2Topology`'s impl with tweaked
  constants.

A data-only approach would sprinkle `if arch_family == Aie1 { ... }`
conditionals through consumer code. The trait contains that branching
inside one impl per arch family.

---

## The trait surface

```rust
pub enum Direction { North, South, East, West }

pub trait TileTopology: Send + Sync {
    fn classify(&self, col: u8, row: u8) -> TileKind;
    fn neighbor(&self, col: u8, row: u8, dir: Direction) -> Option<(u8, u8)>;
}
```

Two methods, "coarse first":

- `classify` covers every consumer that asks "what kind of tile is at
  this coordinate?" The two AIE2 hardcodes at
  `src/npu/executor.rs:895` and `src/device/array/routing.rs:144`
  replaced by calls to `classify`.
- `neighbor` covers every consumer that does bounds-clamped adjacency
  (the four memory-neighbor `row > 0` sites). On AIE2 the impl is
  uniform; on AIE1 it branches on row parity.

Not on the trait:

- `shim_mux_kind(col) -> ShimMuxKind`: deferred. `classify(col, 0)`
  already returns `ShimNoc` vs `ShimPl`, which is what Subsystem 5
  (Stream Switch) will dispatch on for "can I route to PL from this
  column?". Add a separate mux-kind method there if sub-kind
  granularity materializes.
- `memtile_row_range()`, `shim_row_index()`, etc.: these are pure data
  already served by `ArchModel`'s extents + archspec's generated
  constants (`SHIM_ROW`, `COMPUTE_ROW_START`). Adding trait methods
  for them would duplicate the data layer without adding behavior.

---

## What would AIE1 look like?

- `xdna_archspec::aie1::topology::Aie1Topology` struct carrying
  `ArchModel`-derived extents plus a `shim_columns: Vec<TileKind>`
  field that encodes which shim columns are `ShimNoc` vs `ShimPl` for
  the specific device.
- `Aie1Topology::classify(col, row)`:
  - `row == 0`: return `self.shim_columns[col as usize]`.
  - `row > 0`: return `TileKind::Compute` (no memtile row on AIE1).
- `Aie1Topology::neighbor(col, row, dir)`:
  - East / West / South: same as AIE2.
  - North: clamp at `self.rows - 1`.
  - For memory adjacency (if exposed through the same trait): branch on
    `row & 1` for the AIE1 alternating-row memory scheme. (This might
    prompt an additional method like `memory_neighbor` in a future
    subsystem; Phase 1 keeps the surface minimal.)
- `ArchModel::topology()` adds an arm for `Architecture::Aie` returning
  `Box<dyn TileTopology>::new(Aie1Topology::from_model(self))`.

No changes required in consumer code, provided the consumers are already
calling through the trait.

---

## Where further trait surface could enter

Not introduced this subsystem:

- `memory_neighbor(col, row, dir)`: splits off from `neighbor` if AIE1's
  alternating-row scheme cannot be captured by the same method that
  handles stream-switch adjacency. Most likely surfaces in Subsystem 7
  (ISA Execute) when per-arch memory-access semantics diverge.
- `shim_mux_kind(col) -> ShimMuxKind`: Subsystem 5 (Stream Switch) if
  ShimNoc / ShimPl isn't a granular enough classifier.

Neither adds for AIE2 today.

---

## Completion (2026-04-18)

Landed at `phase1-subsys-tile-topo`. Net effect:

- `xdna_archspec::topology::{TileTopology, Direction}` trait + enum live at the crate root.
- `xdna_archspec::aie2::topology::Aie2Topology` concrete impl for AIE2-family devices (NPU1/NPU4/NPU5/NPU6).
- `xdna_archspec::types::ArchModel::topology()` accessor dispatches on `Architecture` (actual impl block lives in `topology.rs` due to a `#[path]`-include constraint in `build.rs`).
- `xdna_archspec::types::TileKind` gains `const fn` inherent predicates `is_shim` / `is_mem` / `is_compute`.
- `xdna-emu`'s `TileType` enum + `From` bridge deleted; all consumers migrate to `TileKind`.
- Two bare `row == 0` tile-classification hardcodes and four memory-neighbor `row > 0` guards now route through archspec's `SHIM_ROW` constant (the trait exists; full-dispatch migration for those call paths is natural Subsystem 7 work).

Verification: `cargo test --lib` = 2708, archspec = 236, full bridge = phase1-subsys-isa-decode baseline (no new regressions), ISA = 0 fail out of 4815.
