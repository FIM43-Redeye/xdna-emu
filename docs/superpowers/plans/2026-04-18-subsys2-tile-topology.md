# Subsystem 2 -- Tile Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `TileTopology` trait seam in `xdna-archspec`, ship an `Aie2Topology` concrete impl, replace `xdna-emu`'s bespoke `TileType` enum with archspec's canonical `TileKind`, dissolve the lossy `From` bridge at `src/device/tile/core_state.rs`, and route the two remaining bare `row == 0` classification sites plus the four memory-neighbor `row > 0` sites through the new trait.

**Architecture:** Single-part subsystem. Trait + concrete impl land first (Tasks 2-3), then the call-site migrations (Tasks 4-5), then the rename sweep (Task 6), then the enum + bridge deletion (Task 7), then the gate and tag (Task 8). The `TileType <-> TileKind` bridge stays alive through Task 6 so call sites can mix old and new types without breaking compilation; Task 7 deletes it only after every consumer has switched.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, dyn-dispatch trait object for topology lookup, `const fn` inherent methods on `TileKind`.

**Spec:** [docs/superpowers/specs/2026-04-18-subsys2-tile-topology-design.md](../specs/2026-04-18-subsys2-tile-topology-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

**Prior subsystem:** `phase1-subsys-isa-decode` (Subsystem 6, 2026-04-17).

---

> **Sweep-as-of 2026-05-01:** Subsystem 2 completed -- tag `phase1-subsys-tile-topo`. Topology data lifted into archspec via TileTopology; ArrayDimensions consumers migrated. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Scope Note

Single-part subsystem with one tag (`phase1-subsys-tile-topo`) at the end. Unlike Subsystem 6 (9,300 lines relocated, Part A/B) or Subsystem 1 (build-system surgery), Subsystem 2 is:

- ~180 LOC of new code in archspec (trait, `Aie2Topology`, `Direction`, predicate methods, unit tests).
- ~200 LOC deleted (`TileType` enum, bridge, predicate methods, round-trip tests).
- ~6 call-site rewrites in `src/` for the hardcode-and-neighbor fixes.
- ~182 mechanical `TileType::X` -> `TileKind::Y` substitutions across ~13 consumer files.
- No build-system moves. No FFI changes. No second-arch population.

Branch: `dev`. Tag at end: `phase1-subsys-tile-topo`.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green. Baseline at `phase1-subsys-isa-decode`: `2712 passed; 0 failed; 5 ignored`.
- `cargo test -p xdna-archspec --lib` green. Baseline: `220 passed; 1 failed; 2 ignored` (the one failure is the pre-existing `device_model::test_full_parse_all_devices`; leave it alone).
- `cargo build` green. `cargo build --release` clean is required before the tag, not every commit.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green after rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`).
- No commit introduces `TODO`/`FIXME`/`unimplemented!()` without an open-issue reference. (`unimplemented!()` for an AIE1 branch is fine IF the design note documents it; see Task 2.)
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`); no emoji; ends with `Generated using Claude Code.`.
- All work on `dev`. No merges to `master` during this plan.
- **Every `cargo` call** must have `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` prepended (tblgen needs llvm-config 21.x, not mlir-aie's 23.x). The helper `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo ...` is the standing idiom throughout this plan.

---

## File Structure

**Current layout (post-Subsystem 6):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── mod.rs                   # re-exports TileType at line ~71
│   │   ├── tile/
│   │   │   ├── mod.rs               # re-exports TileType at line ~38
│   │   │   └── core_state.rs        # defines TileType + From bridge + round-trip tests
│   │   ├── array/
│   │   │   └── routing.rs           # row == 0 hardcode at line 144
│   │   └── ...
│   ├── interpreter/
│   │   ├── execute/memory/neighbor.rs   # row > 0 at lines 80, 135
│   │   └── engine/coordinator.rs        # row > 0 at lines 578, 705
│   └── npu/
│       └── executor.rs               # row == 0 hardcode at line 895
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                    # declares top-level modules
        ├── types.rs                  # defines TileKind at lines 62-77; ArchModel at line 1447
        └── aie2/
            ├── mod.rs                # aie2 submodule declarations
            └── ...
```

**Target layout (post-Subsystem 2):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── mod.rs                   # no TileType re-export
│   │   ├── tile/
│   │   │   ├── mod.rs               # no TileType re-export
│   │   │   └── core_state.rs        # TileType enum + bridge + round-trip tests ALL DELETED
│   │   ├── array/
│   │   │   └── routing.rs           # row == 0 replaced with topology.neighbor(.., South).is_none()
│   │   └── ...
│   ├── interpreter/
│   │   ├── execute/memory/neighbor.rs   # row > 0 replaced with topology.neighbor(.., South).is_some()
│   │   └── engine/coordinator.rs        # row > 0 replaced
│   └── npu/
│       └── executor.rs               # row == 0 replaced with topology classification
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                    # declares topology as new top-level module
        ├── topology.rs               # NEW: TileTopology trait + Direction enum
        ├── types.rs                  # TileKind gains const-fn predicate methods
        └── aie2/
            ├── mod.rs                # declares topology submodule
            └── topology.rs           # NEW: Aie2Topology struct + TileTopology impl
```

---

## Migration Reference Table

The rename sweep in Task 6 follows this substitution table. Subagents executing Task 6 reference this.

| Old (xdna-emu-local) | New (archspec-canonical) |
|---|---|
| `TileType::Shim` (match arm) | `TileKind::ShimNoc \| TileKind::ShimPl` |
| `TileType::Shim` (construction / comparison) | `TileKind::ShimNoc` |
| `TileType::MemTile` | `TileKind::Mem` |
| `TileType::Compute` | `TileKind::Compute` |
| `tile_type: TileType` (field / param) | `tile_kind: TileKind` |
| `crate::device::tile::TileType` | `xdna_archspec::types::TileKind` |
| `crate::device::TileType` | `xdna_archspec::types::TileKind` |
| `TileType::from(..)` / `.into::<TileKind>()` | delete the conversion; use `TileKind` directly |
| `.is_shim()` / `.is_mem_tile()` / `.is_compute()` (on `TileType`) | `.is_shim()` / `.is_mem()` / `.is_compute()` (on `TileKind`; note `is_mem` not `is_mem_tile`) |

`src/trace/vcd.rs` defines a private enum also named `TileType`. It is unrelated and does not re-export from `core_state.rs`. Leave it alone. After the rename, document the distinction with a one-line comment at the top of `vcd.rs`'s definition (see Task 6 Step 6).

---

## Baseline to Preserve

Before Task 1, capture current numbers so later regression checks have a target:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected current values at `phase1-subsys-isa-decode`:

- xdna-emu library tests: `2712 passed; 0 failed; 5 ignored`
- archspec library tests: `220 passed; 1 failed; 2 ignored` (`test_full_parse_all_devices`, pre-existing, unrelated to Subsystem 2)

Record these in `docs/arch/subsys2-audit.md` (created in Task 1).

---

### Task 1: Audit + design-note scaffolding

**Goal:** Create `docs/arch/subsys2-audit.md` with the baseline numbers + the audit facts. Create the `docs/arch/tile-topology.md` design note with the mandatory per-seam structure, leaving the "Completion" and "Final state" sections as stubs for Task 8 to fill.

**Files:**
- Create: `docs/arch/subsys2-audit.md`
- Create: `docs/arch/tile-topology.md`

- [x] **Step 1: Create the audit doc skeleton**

Write `docs/arch/subsys2-audit.md`:

```markdown
# Subsystem 2 -- Tile Topology Audit

## Baseline (pre-subsystem, at phase1-subsys-isa-decode tag)

- `cargo test --lib`: <paste output>
- `cargo test -p xdna-archspec --lib`: <paste output>

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
```

- [x] **Step 2: Fill in the baseline numbers**

Run:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Paste each output into the two `<paste output>` placeholders in `docs/arch/subsys2-audit.md`.

- [x] **Step 3: Create the design-note skeleton**

Write `docs/arch/tile-topology.md`:

```markdown
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
| `ArchModel::topology()` accessor | `xdna_archspec::types` (ArchModel impl block) | Dispatches on `ArchModel::architecture` |

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

## Completion

*(To be filled in by Task 8 with final LOC counts, test deltas, and
the specific commit shas.)*
```

- [x] **Step 4: Verify no test regression from the new docs**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: 2712 passed / 0 failed / 5 ignored. (Docs don't change test behavior, but verify anyway; a later task may accidentally regress this and bisect points here.)

- [x] **Step 5: Commit**

```bash
git add docs/arch/subsys2-audit.md docs/arch/tile-topology.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 2 audit + tile-topology design note scaffolds

Captures baseline test numbers at phase1-subsys-isa-decode, enumerates
the 2 bare row==0 hardcodes + 4 row>0 memory-neighbor sites + ~182
TileType references the rename must sweep, and sketches the
TileTopology trait surface with its AIE1/AIE2P "what would it look
like?" sections. Completion sections filled by Task 8.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: TileTopology trait + Direction + Aie2Topology impl

**Goal:** Add the `TileTopology` trait + `Direction` enum to `xdna-archspec` at a new top-level `topology.rs` module. Add the `Aie2Topology` struct + impl at a new `aie2/topology.rs` module. Expose `ArchModel::topology()` as a `Box<dyn TileTopology + '_>` accessor that dispatches on `ArchModel::architecture`. Ship unit tests for the AIE2 impl.

**Files:**
- Create: `crates/xdna-archspec/src/topology.rs`
- Create: `crates/xdna-archspec/src/aie2/topology.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod topology;`)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (add `pub mod topology;`)
- Modify: `crates/xdna-archspec/src/types.rs` (add `impl ArchModel { pub fn topology() ... }`)

- [x] **Step 1: Write the trait + Direction enum**

Create `crates/xdna-archspec/src/topology.rs`:

```rust
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

use crate::types::TileKind;

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
```

- [x] **Step 2: Declare the module in archspec's lib.rs**

Edit `crates/xdna-archspec/src/lib.rs`. Add (preserving existing alphabetical order among `pub mod` declarations):

```rust
pub mod topology;
```

- [x] **Step 3: Write the Aie2Topology impl**

Create `crates/xdna-archspec/src/aie2/topology.rs`:

```rust
//! AIE2 tile topology implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). AIE2 and AIE2P share this layout discipline:
//!
//! - row 0 is the shim row (every column is `ShimNoc`; AIE2 has no
//!   software-accessible `ShimPl`).
//! - rows 1 through `num_mem_tile_rows` are memtile rows (`Mem`).
//! - rows above that are compute (`Compute`).
//! - memory adjacency is uniform (no row-parity dependence).
//!
//! Extents (columns, rows, memtile row count) come from
//! `ArchModel::array_topology` and are plumbed into the struct at
//! construction.

use crate::topology::{Direction, TileTopology};
use crate::types::{ArchModel, TileKind};

/// AIE2 tile topology.
///
/// Built via `Aie2Topology::from_model(&arch_model)` from an `ArchModel`
/// whose `architecture` is `Aie2` or `Aie2p`.
#[derive(Debug, Clone, Copy)]
pub struct Aie2Topology {
    pub columns: u8,
    pub rows: u8,
    pub num_mem_tile_rows: u8,
}

impl Aie2Topology {
    /// Build an `Aie2Topology` from an `ArchModel`.
    ///
    /// Caller must ensure the model's architecture is AIE2-family
    /// (`Aie2` or `Aie2p`). `ArchModel::topology()` dispatches this.
    pub fn from_model(model: &ArchModel) -> Self {
        Self {
            columns: model.array_topology.num_columns,
            rows: model.array_topology.total_rows,
            num_mem_tile_rows: model.array_topology.num_mem_tile_rows,
        }
    }

    /// First compute row index (immediately above the last memtile row).
    #[inline]
    const fn compute_row_start(&self) -> u8 {
        1 + self.num_mem_tile_rows
    }
}

impl TileTopology for Aie2Topology {
    fn classify(&self, _col: u8, row: u8) -> TileKind {
        if row == 0 {
            TileKind::ShimNoc
        } else if row < self.compute_row_start() {
            TileKind::Mem
        } else {
            TileKind::Compute
        }
    }

    fn neighbor(&self, col: u8, row: u8, dir: Direction) -> Option<(u8, u8)> {
        match dir {
            Direction::South => {
                if row == 0 { None } else { Some((col, row - 1)) }
            }
            Direction::North => {
                if row + 1 >= self.rows { None } else { Some((col, row + 1)) }
            }
            Direction::East => {
                if col + 1 >= self.columns { None } else { Some((col + 1, row)) }
            }
            Direction::West => {
                if col == 0 { None } else { Some((col - 1, row)) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical NPU1 (Phoenix) topology: 5 columns x 6 rows, 1 memtile row.
    fn npu1() -> Aie2Topology {
        Aie2Topology { columns: 5, rows: 6, num_mem_tile_rows: 1 }
    }

    #[test]
    fn classify_shim_row() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.classify(col, 0), TileKind::ShimNoc);
        }
    }

    #[test]
    fn classify_memtile_row() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.classify(col, 1), TileKind::Mem);
        }
    }

    #[test]
    fn classify_compute_rows() {
        let topo = npu1();
        for col in 0..5 {
            for row in 2..6 {
                assert_eq!(topo.classify(col, row), TileKind::Compute);
            }
        }
    }

    #[test]
    fn neighbor_south_from_shim_is_none() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.neighbor(col, 0, Direction::South), None);
        }
    }

    #[test]
    fn neighbor_south_from_compute_returns_memtile() {
        let topo = npu1();
        assert_eq!(topo.neighbor(2, 2, Direction::South), Some((2, 1)));
    }

    #[test]
    fn neighbor_north_from_top_row_is_none() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.neighbor(col, 5, Direction::North), None);
        }
    }

    #[test]
    fn neighbor_east_from_rightmost_column_is_none() {
        let topo = npu1();
        for row in 0..6 {
            assert_eq!(topo.neighbor(4, row, Direction::East), None);
        }
    }

    #[test]
    fn neighbor_west_from_leftmost_column_is_none() {
        let topo = npu1();
        for row in 0..6 {
            assert_eq!(topo.neighbor(0, row, Direction::West), None);
        }
    }

    #[test]
    fn neighbor_round_trip_north_south() {
        let topo = npu1();
        for col in 0..5 {
            for row in 0..5 {
                let north = topo.neighbor(col, row, Direction::North).unwrap();
                let back = topo.neighbor(north.0, north.1, Direction::South).unwrap();
                assert_eq!(back, (col, row));
            }
        }
    }

    #[test]
    fn neighbor_round_trip_east_west() {
        let topo = npu1();
        for col in 0..4 {
            for row in 0..6 {
                let east = topo.neighbor(col, row, Direction::East).unwrap();
                let back = topo.neighbor(east.0, east.1, Direction::West).unwrap();
                assert_eq!(back, (col, row));
            }
        }
    }

    #[test]
    fn compute_row_start_matches_memtile_count() {
        let t1 = Aie2Topology { columns: 5, rows: 6, num_mem_tile_rows: 1 };
        assert_eq!(t1.compute_row_start(), 2);

        // Hypothetical AIE2-family device with 0 memtile rows:
        let t0 = Aie2Topology { columns: 5, rows: 5, num_mem_tile_rows: 0 };
        assert_eq!(t0.compute_row_start(), 1);
        assert_eq!(t0.classify(0, 1), TileKind::Compute);
    }
}
```

- [x] **Step 4: Declare the aie2 submodule**

Edit `crates/xdna-archspec/src/aie2/mod.rs`. Add (at the top with other `pub mod` declarations):

```rust
/// AIE2-family tile topology impl (`Aie2Topology`). Subsystem 2 seam.
pub mod topology;
```

- [x] **Step 5: Add `ArchModel::topology()` accessor**

Edit `crates/xdna-archspec/src/types.rs`. Find the `impl ArchModel { ... }` block. Add the method at the end of the impl:

```rust
/// Return a topology impl for this arch family.
///
/// Dispatches on `self.architecture`. Returns a boxed trait object so
/// consumers don't need to be generic over the concrete impl type.
/// Call-site cost: one small allocation per call. Topology() is a cold
/// accessor (called at config time, not per-instruction), so the
/// allocation is acceptable.
pub fn topology(&self) -> Box<dyn crate::topology::TileTopology + '_> {
    match self.architecture {
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
```

(If the `impl ArchModel { ... }` block isn't obvious from reading the file, search for `pub struct ArchModel` near line 1447 and find the following impl block; add the method there. If no impl block exists, create one immediately after the struct definition.)

- [x] **Step 6: Verify archspec builds and tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-archspec 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected:
- Clean build.
- Archspec test count increased by 11 (the new tests): `231 passed; 1 failed; 2 ignored`.

If tests fail: check that `crate::topology::{Direction, TileTopology}` resolves inside `aie2/topology.rs` (should, since the new `topology.rs` is at crate root). Check that `ArchModel::array_topology.num_columns` / `total_rows` / `num_mem_tile_rows` field names match the actual struct definition; if the field names differ (e.g., `num_rows` instead of `total_rows`), adjust `from_model` and the test helpers accordingly.

- [x] **Step 7: Verify xdna-emu still builds**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2712 passed / 0 failed / 5 ignored.

- [x] **Step 8: Commit**

```bash
git add crates/xdna-archspec/src/topology.rs crates/xdna-archspec/src/aie2/topology.rs crates/xdna-archspec/src/lib.rs crates/xdna-archspec/src/aie2/mod.rs crates/xdna-archspec/src/types.rs
git commit -m "$(cat <<'EOF'
refactor(archspec): add TileTopology trait + Aie2Topology impl

Subsystem 2 seam lands in archspec: TileTopology trait (classify +
neighbor) at the crate root, Aie2Topology concrete impl under aie2,
ArchModel::topology() accessor returning a boxed trait object.  AIE2
impl covers the NPU1/NPU4/NPU5/NPU6 uniform layout (shim@0, mem@1,
compute@2+, no row-parity quirks).  AIE1 branch is unimplemented!()
until its impl lands in later work.  10 new Aie2Topology unit tests.

Generated using Claude Code.
EOF
)"
```

---

### Task 3: `TileKind` inherent predicate methods

**Goal:** Add `is_shim` / `is_mem` / `is_compute` as `const fn` inherent methods on `TileKind` in archspec. These are the replacement for the soon-to-be-deleted `TileType::is_shim` / `is_mem_tile` / `is_compute` methods (note the rename from `is_mem_tile` to `is_mem`).

**Files:**
- Modify: `crates/xdna-archspec/src/types.rs` (add impl block near TileKind at lines 62-77)

- [x] **Step 1: Add the impl block**

Edit `crates/xdna-archspec/src/types.rs`. After the `impl fmt::Display for TileKind` block (ending around line 78), add:

```rust
impl TileKind {
    /// Is this a shim tile (either `ShimNoc` or `ShimPl`)?
    ///
    /// Both shim variants share the same conceptual "shim tile" role in
    /// the emulator's code paths. AIE2 only ever produces `ShimNoc`;
    /// AIE1 produces both, but code that cares about per-column mux
    /// distinction should match on the variant directly rather than
    /// calling this helper.
    #[inline]
    pub const fn is_shim(self) -> bool {
        matches!(self, TileKind::ShimNoc | TileKind::ShimPl)
    }

    /// Is this a memory tile?
    #[inline]
    pub const fn is_mem(self) -> bool {
        matches!(self, TileKind::Mem)
    }

    /// Is this a compute tile?
    #[inline]
    pub const fn is_compute(self) -> bool {
        matches!(self, TileKind::Compute)
    }
}

#[cfg(test)]
mod tile_kind_predicate_tests {
    use super::TileKind;

    #[test]
    fn is_shim_covers_both_variants() {
        assert!(TileKind::ShimNoc.is_shim());
        assert!(TileKind::ShimPl.is_shim());
        assert!(!TileKind::Mem.is_shim());
        assert!(!TileKind::Compute.is_shim());
    }

    #[test]
    fn is_mem_only_mem() {
        assert!(TileKind::Mem.is_mem());
        assert!(!TileKind::Compute.is_mem());
        assert!(!TileKind::ShimNoc.is_mem());
        assert!(!TileKind::ShimPl.is_mem());
    }

    #[test]
    fn is_compute_only_compute() {
        assert!(TileKind::Compute.is_compute());
        assert!(!TileKind::Mem.is_compute());
        assert!(!TileKind::ShimNoc.is_compute());
        assert!(!TileKind::ShimPl.is_compute());
    }

    #[test]
    fn predicates_are_const_fn() {
        // Verify const-fn status: these can be evaluated at compile time.
        const _SHIM: bool = TileKind::ShimNoc.is_shim();
        const _MEM: bool = TileKind::Mem.is_mem();
        const _COMPUTE: bool = TileKind::Compute.is_compute();
    }
}
```

- [x] **Step 2: Verify archspec builds and tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-archspec 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected:
- Clean build.
- Archspec test count increased by 4: `235 passed; 1 failed; 2 ignored`.

- [x] **Step 3: Verify xdna-emu still builds**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: clean, 2712 passed.

- [x] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/types.rs
git commit -m "$(cat <<'EOF'
refactor(archspec): add const-fn is_shim / is_mem / is_compute to TileKind

Inherent predicate methods matching the discriminant-predicate
convention used elsewhere in the codebase.  is_shim folds ShimNoc
and ShimPl together (the "shim tile" concept); per-variant code paths
match directly on the variants.  Const-fn so they participate in
const-eval contexts.  4 new tests including a compile-time const-fn
verification.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Migrate bare `row == 0` hardcodes through the trait

**Goal:** Replace the two remaining bare `row == 0` classification hardcodes at `src/npu/executor.rs:895` and `src/device/array/routing.rs:144` with topology-aware dispatch. The routing.rs site is a cascade south-routing underflow guard that semantically wants "no south neighbor"; use `topology.neighbor(.., Direction::South).is_none()` there. The executor.rs site is asking "is this a shim-BD register?" -- use a classify check or the archspec constant; pick whichever reads cleanly.

**Files:**
- Modify: `src/npu/executor.rs` (line 895)
- Modify: `src/device/array/routing.rs` (line 144)

- [x] **Step 1: Fix `src/npu/executor.rs:895`**

Read the context at line 895 (already audited: `let is_shim_bd = row == 0 && bd_index_for_blockwrite(row, offset).is_some();`).

The comparison is "is row 0" -- on AIE2 this is equivalent to "is this a shim tile". The simplest rewrite uses the archspec constant directly (no topology object needed at this call site; BD layout is strictly geometric):

Replace line 895:

```rust
let is_shim_bd = row == 0 && bd_index_for_blockwrite(row, offset).is_some();
```

With:

```rust
let is_shim_bd = row == xdna_archspec::aie2::SHIM_ROW
    && bd_index_for_blockwrite(row, offset).is_some();
```

If `xdna_archspec::aie2` is not already imported at the top of the file, add a `use` statement:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -10
```

If the build complains about unresolved `xdna_archspec::aie2::SHIM_ROW`, add to the file's imports:

```rust
use xdna_archspec::aie2::SHIM_ROW;
```

and change the line to `let is_shim_bd = row == SHIM_ROW && ...`.

- [x] **Step 2: Fix `src/device/array/routing.rs:144`**

Read the context at line 144 (audited: `if row == 0 { continue; }` inside the cascade south-routing match arm).

This is a "no south neighbor" check. Rewrite it to use the topology's `neighbor` method. This requires access to the topology; the method lives on `ArchModel`, which the routing code should already have (or can reach via the device state).

First, examine what's in scope at that code path:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib routing 2>&1 | tail -3
```

Read `src/device/array/routing.rs` lines 120-160. Identify whether an `ArchModel` (or something that can produce one) is in scope. If it is, use:

```rust
0 => {
    // South: (col, row - 1)
    let Some((dst_col, dst_row)) = topology.neighbor(col, row, Direction::South) else {
        continue;
    };
    (dst_col, dst_row)
}
```

where `topology` is obtained once at the top of the enclosing function via something like:

```rust
let topology = self.arch_model.topology();
```

or via whatever accessor the device/array context exposes.

**If an `ArchModel` is not directly in scope:** rather than threading one through, use the archspec constant form as a minimal change -- this preserves the current semantic exactly (AIE2 only), documents the intent, and defers the trait-dispatch migration to the cascade routing code's own refactor later:

```rust
0 => {
    // South: (col, row - 1). Shim row has no south neighbor.
    if row == xdna_archspec::aie2::SHIM_ROW { continue; }
    (col, row - 1)
}
```

**Default to the constant form** at this task unless the topology handle is already readily available. The hardcode-to-constant swap is all that's technically required for the spec's success criterion 7; the full trait migration for cascade routing is natural work for Subsystem 5 (Stream Switch) where stream-switch topology deltas surface together.

- [x] **Step 3: Verify**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2712 passed. If `test_cascade_route_*` tests fail, the routing rewrite broke the semantic -- the constant-form rewrite (Step 2's fallback) should produce bit-identical behavior on AIE2.

- [x] **Step 4: Verify zero bare `row == 0` tile-classification sites remain**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -3
# Confirm no raw row==0 classification checks remain in xdna-emu (outside of tests).
# The pattern `row == 0` should only appear in contexts that are geometric
# bounds checks (not tile-type classification).
```

Run a grep to enumerate:

```
rg 'row\s*==\s*0' src/ -g '!**/tests.rs' -g '!**/tests/**'
```

Expected matches: only in code paths that are pure geometric (e.g., loop-index comparisons, not tile-classification).

If any matches are tile-classification, fix them the same way.

- [x] **Step 5: Commit**

```bash
git add src/npu/executor.rs src/device/array/routing.rs
git commit -m "$(cat <<'EOF'
refactor: migrate bare row==0 tile-classification to archspec SHIM_ROW

src/npu/executor.rs:895 (shim-BD layout dispatch) and
src/device/array/routing.rs:144 (cascade south-routing underflow guard)
now use xdna_archspec::aie2::SHIM_ROW instead of hardcoded 0.  Both
sites preserve the current AIE2 behavior exactly; full trait-dispatch
migration for cascade routing is natural Subsystem 5 work.

Generated using Claude Code.
EOF
)"
```

---

### Task 5: Migrate memory-neighbor `row > 0` sites through `neighbor()`

**Goal:** Replace the four memory-neighbor `row > 0` sites with calls to `topology.neighbor(col, row, Direction::South)`. These sites are the per-task concrete justification for the `neighbor()` method on the trait.

**Files:**
- Modify: `src/interpreter/execute/memory/neighbor.rs` (lines 80, 135)
- Modify: `src/interpreter/engine/coordinator.rs` (lines 578, 705)

- [x] **Step 1: Inspect what's in scope at each site**

Read each of the four files around the target lines. For each site, identify:
1. What coordinate types are in scope (likely `usize`, not `u8`).
2. Whether a `TileTopology` / `ArchModel` handle is reachable (likely via `device.arch_model()` or similar).
3. What the current code does with the negative outcome (`continue;` vs. `None` vs. buffering nothing).

```bash
sed -n '60,90p' src/interpreter/execute/memory/neighbor.rs
sed -n '125,145p' src/interpreter/execute/memory/neighbor.rs
sed -n '570,590p' src/interpreter/engine/coordinator.rs
sed -n '695,715p' src/interpreter/engine/coordinator.rs
```

(Use Read with the appropriate offsets rather than sed if you're in the Claude Code interface.)

- [x] **Step 2: Decide on coordinate plumbing**

The topology's `neighbor()` takes and returns `u8`. The neighbor/coordinator code uses `usize`. Two options:

**Option A (minimal):** Leave the existing `usize` coordinates in place and just rewrite the condition using `SHIM_ROW` constant (value `0`):

```rust
MemoryQuadrant::South => {
    if self.row > xdna_archspec::aie2::SHIM_ROW as usize {
        Some((self.col, self.row - 1))
    } else {
        None
    }
}
```

This captures the intent ("south neighbor exists only above the shim row") at the archspec-constant level, preserves the current semantic exactly, and doesn't require threading an `ArchModel` through the memory subsystem.

**Option B (full trait dispatch):** Thread a `TileTopology` / `ArchModel` handle through `NeighborMemory` and `TileCoordinator` state, call `topology.neighbor(col as u8, row as u8, Direction::South)`, convert the `Option<(u8, u8)>` back to `Option<(usize, usize)>`.

**Default to Option A.** Full trait dispatch is appropriate work for Subsystem 7 (ISA Execute) where per-arch memory semantics surface together. For Subsystem 2 the point is to eliminate the `row > 0` hardcode and route it through an archspec constant; the trait exists for future consumers to pick up.

- [x] **Step 3: Rewrite `src/interpreter/execute/memory/neighbor.rs:80`**

Replace lines 78-88 (the `neighbor_coords` match arms):

```rust
fn neighbor_coords(&self, dir: MemoryQuadrant) -> Option<(usize, usize)> {
    match dir {
        MemoryQuadrant::South => {
            if self.row > 0 { Some((self.col, self.row - 1)) } else { None }
        }
        MemoryQuadrant::West => {
            if self.col > 0 { Some((self.col - 1, self.row)) } else { None }
        }
        MemoryQuadrant::North => Some((self.col, self.row + 1)),
        MemoryQuadrant::East => Some((self.col + 1, self.row)),
        MemoryQuadrant::Local => None,
    }
}
```

With:

```rust
fn neighbor_coords(&self, dir: MemoryQuadrant) -> Option<(usize, usize)> {
    // Shim row (row == SHIM_ROW) has no south neighbor; this is the
    // archspec-level invariant that the TileTopology trait encodes.
    // Memory-subsystem call sites use it in constant form today;
    // Subsystem 7 (ISA Execute) threads a full TileTopology handle
    // through if per-arch memory semantics diverge (e.g., AIE1
    // alternating-row adjacency).
    const SHIM_ROW: usize = xdna_archspec::aie2::SHIM_ROW as usize;
    match dir {
        MemoryQuadrant::South => {
            if self.row > SHIM_ROW { Some((self.col, self.row - 1)) } else { None }
        }
        MemoryQuadrant::West => {
            if self.col > 0 { Some((self.col - 1, self.row)) } else { None }
        }
        MemoryQuadrant::North => Some((self.col, self.row + 1)),
        MemoryQuadrant::East => Some((self.col + 1, self.row)),
        MemoryQuadrant::Local => None,
    }
}
```

- [x] **Step 4: Rewrite `src/interpreter/execute/memory/neighbor.rs:135` (the `apply_writes` arm)**

Lines 134-140 currently:

```rust
let coords = match dir {
    MemoryQuadrant::South => if row > 0 { Some((col, row - 1)) } else { None },
    MemoryQuadrant::West => if col > 0 { Some((col - 1, row)) } else { None },
    MemoryQuadrant::North => Some((col, row + 1)),
    MemoryQuadrant::East => Some((col + 1, row)),
    MemoryQuadrant::Local => None,
};
```

Replace with:

```rust
// Same invariant as neighbor_coords above: shim row has no south
// neighbor. Use the archspec constant directly rather than duplicating
// the SHIM_ROW expression inline.
const SHIM_ROW: usize = xdna_archspec::aie2::SHIM_ROW as usize;
let coords = match dir {
    MemoryQuadrant::South => if row > SHIM_ROW { Some((col, row - 1)) } else { None },
    MemoryQuadrant::West => if col > 0 { Some((col - 1, row)) } else { None },
    MemoryQuadrant::North => Some((col, row + 1)),
    MemoryQuadrant::East => Some((col + 1, row)),
    MemoryQuadrant::Local => None,
};
```

(The `const SHIM_ROW: usize = ...` may be redundant if the outer function already declared it. If so, drop the local `const` and reuse the outer one.)

- [x] **Step 5: Rewrite `src/interpreter/engine/coordinator.rs:578`**

Line 578 currently: `if row > 0 {` (south-lock snapshot guard).

Rewrite to:

```rust
if row > xdna_archspec::aie2::SHIM_ROW as usize {
```

(Assumes `row` is `usize` here; if it's `u8`, drop the `as usize`.)

- [x] **Step 6: Rewrite `src/interpreter/engine/coordinator.rs:705`**

Line 705 currently: `if row > 0 {` (south-lock writeback guard).

Same rewrite as Step 5.

- [x] **Step 7: Verify tests pass**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: clean, 2712 passed.

Specifically verify the memtile neighbor tests still pass:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib test_memtile_mm2s_reads_from_east_neighbor 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib test_memtile_mm2s_reads_from_west_neighbor 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib test_memtile_s2mm_writes_to_east_neighbor 2>&1 | tail -3
```

Expected: all pass.

- [x] **Step 8: Verify zero bare `row > 0` memory-neighbor sites remain**

```
rg 'row\s*>\s*0' src/interpreter/
```

Expected: no matches outside `#[cfg(test)]` blocks. (If matches exist in non-memory-neighbor contexts, that's fine -- leave them alone.)

- [x] **Step 9: Commit**

```bash
git add src/interpreter/execute/memory/neighbor.rs src/interpreter/engine/coordinator.rs
git commit -m "$(cat <<'EOF'
refactor: route memory-neighbor row>0 guards through archspec SHIM_ROW

Four sites (neighbor.rs:80 and :135, coordinator.rs:578 and :705) now
reference xdna_archspec::aie2::SHIM_ROW instead of the hardcoded 0.
Semantic preserved exactly on AIE2.  Full TileTopology::neighbor
dispatch in the memory subsystem is Subsystem 7 work; the trait exists
for future consumers when AIE1's alternating-row adjacency lands.

Generated using Claude Code.
EOF
)"
```

---

### Task 6: `TileType` -> `TileKind` rename sweep

**Goal:** Replace every `TileType::` reference outside `src/device/tile/core_state.rs` (which owns the definition; Task 7 handles its deletion) with `TileKind`. Replace every `tile_type: TileType` field/parameter with `tile_kind: TileKind`. Replace predicate method calls (`is_mem_tile` -> `is_mem`). This is mechanical but requires manual match-arm review for `TileType::Shim` (which becomes `TileKind::ShimNoc | TileKind::ShimPl` in match arms).

**Files:**
- Modify: ~13 files across `src/device/`, `src/interpreter/`, `src/npu/`, `src/trace/`. Exact list enumerated in Step 1.

- [x] **Step 1: Enumerate the consumer files**

```bash
rg -l 'TileType' src/ | grep -v 'src/trace/vcd.rs' | sort > /tmp/claude-1000/subsys2-tiletype-consumers.txt
wc -l /tmp/claude-1000/subsys2-tiletype-consumers.txt
cat /tmp/claude-1000/subsys2-tiletype-consumers.txt
```

Expected: ~13 files. `src/trace/vcd.rs` is excluded because it defines an unrelated private `TileType` enum that should stay.

- [x] **Step 2: Per-file sed sweep -- variant names**

Apply the migration reference table's mechanical substitutions. A helper script:

```bash
# /tmp/claude-1000/subsys2-rename-sweep.sh
#!/bin/bash
set -e
FILES=$(cat /tmp/claude-1000/subsys2-tiletype-consumers.txt)

for f in $FILES; do
  # Variant renames (literal -- don't touch match arms yet; see below)
  sed -i 's|TileType::MemTile|TileKind::Mem|g' "$f"
  sed -i 's|TileType::Compute|TileKind::Compute|g' "$f"
  # TileType::Shim is ambiguous (match arm vs construction) -- handled in Step 3
done
```

Run it:

```bash
chmod +x /tmp/claude-1000/subsys2-rename-sweep.sh
/tmp/claude-1000/subsys2-rename-sweep.sh
```

- [x] **Step 3: `TileType::Shim` -- manual review**

`TileType::Shim` has two semantic roles:

1. **Construction / comparison** (e.g., `if tile_kind == TileType::Shim { ... }` or `let t = TileType::Shim;`): replace with `TileKind::ShimNoc` (AIE2 never produces `ShimPl`).
2. **Match arms** (e.g., `match kind { TileType::Shim => ... }`): replace with `TileKind::ShimNoc | TileKind::ShimPl` (to preserve AIE2 behavior AND correctly handle AIE1's `ShimPl` when that lands).

Enumerate every remaining `TileType::Shim` site:

```bash
rg -n 'TileType::Shim' src/ | grep -v 'src/trace/vcd.rs'
```

For each site, inspect the context. The vast majority will be match arms (the `Shim` variant is often checked against tile-type classifications). A handful will be constructions or direct comparisons.

**Rewrite rule:**
- In `match` / `if let` arms: `TileType::Shim` -> `TileKind::ShimNoc | TileKind::ShimPl`
- Elsewhere (construction, comparison, pattern binding like `let TileType::Shim = ...`): `TileType::Shim` -> `TileKind::ShimNoc`

Work through the list manually. Each site should take <30 seconds.

- [x] **Step 4: Remove prefix `TileType::` -> `TileKind::` globally (catch-all)**

After Steps 2-3, any remaining `TileType::X` literals are errors or straggling references. A final sed pass:

```bash
for f in $(cat /tmp/claude-1000/subsys2-tiletype-consumers.txt); do
  sed -i 's|TileType::|TileKind::|g' "$f"
done
```

(This is a safety net for anything Step 2-3 missed. It also unsafely collapses `TileType::Shim` -> `TileKind::Shim`, which doesn't compile -- so if the build fails after this step, Step 3 missed sites.)

- [x] **Step 5: Rename bindings (`tile_type` -> `tile_kind`)**

This is less critical but matches archspec terminology. For each file in the list:

```bash
for f in $(cat /tmp/claude-1000/subsys2-tiletype-consumers.txt); do
  sed -i 's|tile_type:|tile_kind:|g' "$f"   # field declarations
  sed -i 's|\.tile_type|.tile_kind|g' "$f"  # field access
  sed -i 's|tile_type =|tile_kind =|g' "$f" # assignments
  sed -i 's|tile_type,|tile_kind,|g' "$f"   # struct init / destructure
  sed -i 's|tile_type }|tile_kind }|g' "$f" # struct init end
done
```

These are narrow patterns to avoid over-matching. Verify:

```bash
rg 'tile_type' src/ | grep -v 'src/trace/vcd.rs' | grep -v 'tile_kind'
```

Expected: zero or only safe variable-name uses (local `let tile_type = ...` bindings that aren't struct fields). Fix any remaining real field uses by hand.

- [x] **Step 6: Predicate method rename (`is_mem_tile` -> `is_mem`)**

Find every call to `.is_mem_tile()`:

```bash
rg -n '\.is_mem_tile\(\)' src/
```

Replace each with `.is_mem()`:

```bash
for f in $(rg -l '\.is_mem_tile\(\)' src/); do
  sed -i 's|\.is_mem_tile()|.is_mem()|g' "$f"
done
```

The `is_shim()` and `is_compute()` methods keep their names (same on both `TileType` and `TileKind`), so they don't need rewriting.

- [x] **Step 7: Handle imports**

Any file that was importing `TileType` from `crate::device::tile` or `crate::device` now needs to import `TileKind` from `xdna_archspec`. The sed pass in Step 4 turned the type references but not the imports.

```bash
# Catch `use crate::device::{..., TileType, ...};`
for f in $(cat /tmp/claude-1000/subsys2-tiletype-consumers.txt); do
  # Replace standalone TileType import
  sed -i 's|use crate::device::tile::TileType;|use xdna_archspec::types::TileKind;|g' "$f"
  sed -i 's|use crate::device::TileType;|use xdna_archspec::types::TileKind;|g' "$f"
  # Replace TileType inside a brace-group import (simple pattern)
  sed -i 's|TileType,|TileKind,|g' "$f"
  sed -i 's|, TileType|, TileKind|g' "$f"
  sed -i 's|{TileType}|{TileKind}|g' "$f"
  sed -i 's|{TileType |{TileKind |g' "$f"
done
```

The brace-group patterns are approximations; some may require manual fix-up.

- [x] **Step 8: Delete stale `TileType::from(..)` conversions**

With `TileType` gone, any `.into::<TileKind>()` or `TileKind::from(tile_type)` expressions become no-ops or are dead code. Find them:

```bash
rg -n 'TileKind::from|TileType::from' src/
```

Review each site and either delete the conversion (if the source is already `TileKind`) or leave it as a Kind-to-Kind identity `match` if it was doing semantic work. For the sweep, most will be mechanical deletions.

- [x] **Step 9: Build and fix fallout**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -30
```

Expected outcomes:
- **If it builds cleanly:** skip to Step 10.
- **If it fails:** read each error. Common failures:
  - Unresolved `TileType` somewhere the sweep missed: grep and fix.
  - Match-arm exhaustiveness (rustc warns `ShimNoc` missing): Step 3 missed a site; inspect and fix.
  - Import error (`unresolved import xdna_archspec::types::TileKind`): Step 7's brace-group pattern didn't catch a specific import style; add the import manually.

Keep iterating the error list until `cargo build` is clean. This task's "manual review" time-cost is concentrated here.

- [x] **Step 10: Run tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: 2712 passed / 0 failed / 5 ignored (baseline preserved; the 3 bridge round-trip tests in `core_state.rs` haven't deleted yet -- that's Task 7). If tests fail, a match-arm rewrite changed behavior.

- [x] **Step 11: Run the tile-type-specific smoke list**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib \
  test_mem_tile_engine \
  test_memtile_bd_capacity \
  test_memtile_port_mappings \
  test_resolve_lock_id_memtile \
  test_memtile_mm2s_reads_from_east_neighbor \
  test_memtile_mm2s_reads_from_west_neighbor \
  test_memtile_s2mm_writes_to_east_neighbor \
  test_memtile_mm2s_out_of_window_addr_records_fatal_error \
  test_memtile_bd_channel_validity \
  test_memtile_bd_parsing \
  test_shim_bd_parsing \
  test_tile_types \
  test_cascade_route \
  2>&1 | tail -5
```

Expected: all pass.

- [x] **Step 12: FFI smoke**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -5
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -10
```

Expected: FFI builds clean, bridge smoke passes.

- [x] **Step 13: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rename TileType->TileKind across ~13 consumer files

Mechanical sweep of ~182 occurrences: TileType::MemTile -> TileKind::Mem,
TileType::Shim -> TileKind::ShimNoc (construction) or
TileKind::ShimNoc | TileKind::ShimPl (match arms), TileType::Compute ->
TileKind::Compute.  Field bindings tile_type -> tile_kind.  Predicate
rename is_mem_tile -> is_mem (is_shim and is_compute keep their names).
Imports swap from crate::device::*::TileType to
xdna_archspec::types::TileKind.  Bridge and TileType enum definition
itself persist for one more task to keep this diff mechanical.

Generated using Claude Code.
EOF
)"
```

---

### Task 7: Delete `TileType` enum + bridge + update re-exports

**Goal:** With all consumers migrated to `TileKind`, delete the `TileType` enum and the `From` bridge from `src/device/tile/core_state.rs`, remove the public re-exports in `src/device/tile/mod.rs` and `src/device/mod.rs`, and note the unrelated `src/trace/vcd.rs::TileType` distinction with a short comment.

**Files:**
- Modify: `src/device/tile/core_state.rs` (delete the enum, the impl block, the two `From` impls, and the 4 round-trip tests)
- Modify: `src/device/tile/mod.rs` (remove `TileType` from the re-export list)
- Modify: `src/device/mod.rs` (remove `TileType` from the re-export list)
- Modify: `src/trace/vcd.rs` (add one-line note above the private `TileType` enum, if appropriate)

- [x] **Step 1: Verify zero remaining `TileType::` references**

```bash
rg -l 'TileType::' src/
```

Expected: only `src/device/tile/core_state.rs` (the enum definition + impls) and `src/trace/vcd.rs` (the unrelated private enum). If any other file appears, Task 6 missed it; fix before proceeding.

- [x] **Step 2: Delete `TileType` from core_state.rs**

Edit `src/device/tile/core_state.rs`. Delete:

- The `TileType` enum definition (lines 49-58).
- The `impl TileType { is_shim / is_mem_tile / is_compute }` block (lines 60-78).
- The `From<TileKind> for TileType` impl (lines 80-92).
- The `From<TileType> for TileKind` impl (lines 94-106).
- The `use xdna_archspec::types::TileKind;` at the top of the file if it's no longer used anywhere else in the file (grep in the file).
- The 4 tests in the `#[cfg(test)] mod tests` block at lines 127-165.

After the deletion, the file should retain: `CoreState` struct + impl, `LegacyStreamPort` struct, `CtrlPacketAction` enum. That's it.

- [x] **Step 3: Remove `TileType` from `src/device/tile/mod.rs` re-exports**

Edit `src/device/tile/mod.rs`. Find line ~38:

```rust
pub use core_state::{CoreState, LegacyStreamPort, TileType, CtrlPacketAction};
```

Replace with:

```rust
pub use core_state::{CoreState, LegacyStreamPort, CtrlPacketAction};
```

- [x] **Step 4: Remove `TileType` from `src/device/mod.rs` re-exports**

Edit `src/device/mod.rs`. Find line ~71:

```rust
pub use tile::{Tile, TileType, Lock, LockResult, DmaBufferDescriptor, DmaChannel, CoreState};
```

Replace with:

```rust
pub use tile::{Tile, Lock, LockResult, DmaBufferDescriptor, DmaChannel, CoreState};
```

(If any consumer was using `use crate::device::TileType`, Task 6's Step 7 should have already rewritten it. If not, fix at next build error.)

- [x] **Step 5: Add the vcd.rs disambiguation comment**

Edit `src/trace/vcd.rs`. Find the `enum TileType` definition. Immediately above it, add:

```rust
/// VCD-trace-local tile-type enum. NOT the archspec `TileKind` -- this
/// is a private enum used exclusively for VCD signal-name dispatch in
/// trace output. Do not expose it publicly or import `TileKind` in
/// parallel without considering which one the call site wants.
```

If the enum definition already has adjacent docstring lines, fold the disambiguation into the existing comment instead of adding a new one.

- [x] **Step 6: Build and test**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -10
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected:
- Clean build.
- `2709 passed; 0 failed; 5 ignored` (2712 baseline minus the 3 deleted round-trip tests in core_state.rs). The fourth test (`tile_kind_shim_pl_maps_to_shim`) also deletes since `TileType` is gone -- recount is 2712 - 4 = 2708 if all four deleted, or 2709 if three. Either is correct; the spec says "approximately 2712 +/- a small delta."

Actual delta will land as one of those two numbers -- confirm exact count in Step 8 when capturing the audit.

- [x] **Step 7: Run the tile-type-specific smoke list**

Same as Task 6 Step 11. Expected: all 16 tests pass.

- [x] **Step 8: Verify FFI + bridge smoke**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -5
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -10
```

Expected: clean.

- [x] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: delete TileType enum, From bridge, and re-exports

xdna-emu's bespoke TileType enum dissolves into archspec's canonical
TileKind.  src/device/tile/core_state.rs loses the enum definition,
predicate methods, two From impls, and four round-trip tests (which
were testing the now-defunct bridge).  Public re-exports drop from
src/device/tile/mod.rs and src/device/mod.rs.  src/trace/vcd.rs gains
a disambiguation comment explaining that its private TileType is
unrelated.

Test count drops by 3-4 from the baseline (bridge round-trip tests);
archspec's new Aie2Topology + TileKind predicate tests (from Tasks 2+3)
more than offset across the workspace.

Generated using Claude Code.
EOF
)"
```

---

### Task 8: Completion docs, gate, and tag

**Goal:** Fill in the Completion sections of `docs/arch/subsys2-audit.md` and `docs/arch/tile-topology.md`. Run the full HW bridge + ISA gate. Update `NEXT-STEPS.md` to point at Subsystem 3 (DMA Engine) as up next. Tag `phase1-subsys-tile-topo`.

**Files:**
- Modify: `docs/arch/subsys2-audit.md` (fill Completion)
- Modify: `docs/arch/tile-topology.md` (fill Completion)
- Modify: `NEXT-STEPS.md` (update tag + pickup guide)

- [x] **Step 1: Fast verification**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build --release 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -5
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -10
```

Expected:
- Release build clean.
- xdna-emu: 2708-2709 passed; 0 failed; 5 ignored.
- archspec: 235 passed; 1 failed (pre-existing); 2 ignored.
- FFI build clean.
- Bridge smoke: Chess and Peano PASS for `add_one_cpp_aiecc`.

Record the exact xdna-emu and archspec test counts for the audit.

- [x] **Step 2: Verify success criteria sweep**

```bash
# Criterion 6: zero TileType references outside vcd.rs
rg -l 'TileType' src/ | grep -v 'src/trace/vcd.rs'
# Expected: empty output.

# Criterion 7: zero bare row==0 tile-classification hardcodes
rg 'row\s*==\s*0' src/ -g '!**/tests.rs' -g '!**/tests/**' -g '!src/trace/vcd.rs'
# Expected: empty output (or only matches in pure geometric contexts).

# Criterion 8: TileType enum definition gone from core_state.rs
rg -n 'enum TileType' src/device/tile/core_state.rs
# Expected: empty.

# Criterion 8: From bridge gone
rg -n 'From<TileKind> for TileType|From<TileType> for TileKind' src/
# Expected: empty.
```

If any of these fail, a prior task missed a cleanup. Find and fix before proceeding.

- [x] **Step 3: Full HW bridge run**

```bash
nice -n 19 ./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-subsys2.log
```

Expected duration: ~20-30 minutes. Expected outcome: matches the `phase1-subsys-isa-decode` baseline. Known pre-existing failure `bd_chain_repeat_on_memtile` remains (real-HW EMU deadlock; see NEXT-STEPS).

Check the tail for the pass/fail summary:

```
tail -30 /tmp/claude-1000/bridge-subsys2.log
```

- [x] **Step 4: ISA test suite**

```bash
nice -n 19 ./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys2.log
```

Expected duration: ~10 minutes. Expected: `FAIL: 0`.

```
tail -10 /tmp/claude-1000/isa-subsys2.log
```

- [x] **Step 5: Fill in `docs/arch/subsys2-audit.md` Completion section**

Replace the `*(To be filled in by Task 8.)*` line in `docs/arch/subsys2-audit.md` with:

```markdown
## Completion

Landed 2026-MM-DD. Tag: `phase1-subsys-tile-topo`.

### Commits (Task 1 through tag)

<output of `git log --oneline phase1-subsys-isa-decode..HEAD`>

### Verification (at tag)

- `cargo test --lib`: <final count> passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: <final count> passed; 1 failed (pre-existing `test_full_parse_all_devices`); 2 ignored.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: matches phase1-subsys-isa-decode baseline; bd_chain_repeat_on_memtile still fails (known pre-existing EMU deadlock).
- ISA test suite: FAIL: 0.

### Success criteria sweep

- Bare `row == 0` tile-classification hardcodes in src/: 0.
- Bare `row > 0` memory-neighbor hardcodes in src/: 0.
- `TileType::` references in src/ (outside src/trace/vcd.rs): 0.
- `TileType` enum definition in src/device/tile/core_state.rs: deleted.
- `From<TileKind> for TileType` / `From<TileType> for TileKind`: deleted.
- `src/device/tile/mod.rs` / `src/device/mod.rs` re-exports of `TileType`: deleted.
- `xdna_archspec::topology::{TileTopology, Direction}` populated.
- `xdna_archspec::aie2::topology::Aie2Topology` populated.
- `ArchModel::topology()` accessor: returns `Box<dyn TileTopology>` dispatched on `Architecture`.
- `TileKind::is_shim()` / `is_mem()` / `is_compute()` const-fn predicates: populated.
- `docs/arch/tile-topology.md` design note: written.

### Net code delta

- New in archspec: ~180 LOC (topology.rs + aie2/topology.rs + TileKind predicates + tests).
- Deleted in xdna-emu: ~200 LOC (TileType enum, bridge, predicate methods, round-trip tests).
- Call-site rewrites: 2 bare `row == 0` hardcodes + 4 `row > 0` memory-neighbor sites + ~182 `TileType::` rename occurrences across ~13 files.

### Follow-ups flagged

Follow-ups that fit naturally in later work, NOT blocking:

- **Subsystem 5 (Stream Switch):** Revisit whether `TileTopology` grows a `shim_mux_kind(col) -> ShimMuxKind` method when stream-switch legality checks need per-column ShimNoc / ShimPl granularity beyond `classify(col, 0)`.
- **Subsystem 7 (ISA Execute):** Thread a full `TileTopology` handle through memory-subsystem code if AIE1's alternating-row adjacency needs dispatch. Today the memory-neighbor sites use the archspec `SHIM_ROW` constant, preserving AIE2 behavior exactly.
- **Subsystem 3 (DMA Engine):** Existing silent DMA channel fallback `(2, 2)` in `from_arch_model()` was flagged in the Phase 1a audit; unchanged in Subsystem 2.
- **Phase 2 hygiene:** The `tile_kind_from_row` helper in `src/device/registers.rs:610-616` could move to `TileTopology::classify` for consistency; deferred because it's currently working correctly.
```

Then:

- Replace `<final count>` with the exact numbers from Step 1.
- Replace `<output of git log --oneline phase1-subsys-isa-decode..HEAD>` with the actual commit list:

```bash
git log --oneline phase1-subsys-isa-decode..HEAD
```

- Replace `2026-MM-DD` with the actual date (use `date +%Y-%m-%d`).

- [x] **Step 6: Fill in `docs/arch/tile-topology.md` Completion section**

Replace the `*(To be filled in by Task 8...)*` line in `docs/arch/tile-topology.md` with:

```markdown
## Completion (2026-MM-DD)

Landed at `phase1-subsys-tile-topo`. Net effect:

- `xdna_archspec::topology::{TileTopology, Direction}` trait + enum live at the crate root.
- `xdna_archspec::aie2::topology::Aie2Topology` concrete impl for AIE2-family devices (NPU1/NPU4/NPU5/NPU6).
- `xdna_archspec::types::ArchModel::topology()` accessor dispatches on `Architecture`.
- `xdna_archspec::types::TileKind` gains `const fn` inherent predicates `is_shim` / `is_mem` / `is_compute`.
- `xdna-emu`'s `TileType` enum + `From` bridge deleted; all consumers migrate to `TileKind`.
- Two bare `row == 0` tile-classification hardcodes and four memory-neighbor `row > 0` guards now route through archspec's `SHIM_ROW` constant (the trait exists; full-dispatch migration for those call paths is natural Subsystem 7 work).

Verification: `cargo test --lib` = <final count>, archspec = <final count>, full bridge = baseline, ISA = 0 fail.
```

Fill in the numbers and date.

- [x] **Step 7: Update NEXT-STEPS.md**

Edit `NEXT-STEPS.md`. Update the header (around the top):

```markdown
**Last updated:** 2026-MM-DD (Phase 1b Subsystem 2 landed; Subsystem 3 up next)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-tile-topo` (Subsystem 2 completion; Part B commits of prior subsystems are ancestors)
```

Update the subsystem status table:

```markdown
| 2 | Tile Topology | `phase1-subsys-tile-topo` | **Done** | Trait seam (TileTopology) + AIE2 impl. TileType->TileKind deep rename. 2 bare row==0 hardcodes + 4 row>0 neighbor guards routed through archspec constants. See docs/arch/tile-topology.md. |
| 3 | DMA Engine & BD Format | `phase1-subsys-dma` | **Up next** | First behavioral seam. Audit AIE2 vs AIE1 BD layout via aie-rt source. Lift BD parse/encode + channel stepping behind `DmaModel` trait. |
```

Replace the entire `## How to Pick Up Subsystem 2 (Tile Topology)` section with a new `## How to Pick Up Subsystem 3 (DMA Engine)` section:

```markdown
## How to Pick Up Subsystem 3 (DMA Engine & BD Format)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/phase1a-audit.md` -- the DMA-channel fallback `(2, 2)` noted
     under "Follow-ups flagged" is Subsystem 3 scope.
   - `docs/arch/subsys2-audit.md` -- Subsystem 2 completion; gives the
     current tile-topology trait surface that Subsystem 3 consumes.
   - `docs/arch/tile-topology.md` -- the per-seam design note template
     Subsystem 3 should mimic for `docs/arch/dma-model.md`.
   - `docs/arch/subsys6-audit.md` -- Part A + Part B completion logs;
     still the best template for how an audit is structured.

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1-subsys-tile-topo..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Subsystem 2 left off.

   ```bash
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
   ```
   Expect xdna-emu `<final count> passed; 0 failed; 5 ignored` and archspec
   `<final count> passed; 1 failed; 2 ignored` (pre-existing
   `test_full_parse_all_devices`).

3. **Invoke brainstorming** to shape Subsystem 3's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 3: DMA Engine & BD Format."

   **Shape the spec around these questions:**
   - Does AIE2 vs AIE1 BD layout diverge at the field level, or only at
     the per-tile-type level that archspec already parameterizes via
     aiert modules?
   - Which channel-stepping behaviors are actually per-arch vs. just
     per-tile-type?
   - What does the "what would AIE1 look like?" answer for DMA look
     like, and does it justify a trait seam for BD parsing alone, for
     channel stepping alone, or both?
   - Is the silent channel-fallback `(2, 2)` from Phase 1a scope here,
     or Phase 2 hygiene?

4. **Invoke writing-plans** to produce a plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys3-dma.md`.

5. **Invoke subagent-driven-development** to execute.

6. **At end of Subsystem 3:** tag `phase1-subsys-dma`, append a
   completion section to its audit, update this `NEXT-STEPS.md` to
   move Subsystem 4 (Locks) to "up next."
```

Fill in `<final count>` with the actual numbers.

- [x] **Step 8: Commit docs updates**

```bash
git add docs/arch/subsys2-audit.md docs/arch/tile-topology.md NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 2 completion log + NEXT-STEPS points at Subsystem 3

Audit Completion section filled: final test counts, success-criteria
sweep, net code delta, and Subsystem 5/7 follow-up flags.
Tile-topology design note's Completion section filled with the final
trait-surface summary.  NEXT-STEPS pickup guide rewrites to point at
Subsystem 3 (DMA Engine) with the "what would AIE1 look like?"
questions pre-shaped.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 9: Tag**

```bash
git tag phase1-subsys-tile-topo -m "Phase 1b Subsystem 2: TileTopology trait + TileKind deep rename"
```

Verify the tag:

```bash
git log --oneline phase1-subsys-tile-topo -5
```

- [x] **Step 10: Final sanity pass**

```bash
# Confirm we're at the expected commit & tag
git log --oneline -5

# Confirm no stray changes
git status

# Confirm the three success-criteria greps stay clean
rg -l 'TileType' src/ | grep -v 'src/trace/vcd.rs'
rg -n 'enum TileType' src/device/tile/core_state.rs
rg -n 'From<TileKind> for TileType|From<TileType> for TileKind' src/
```

Expected: clean status; three greps all empty.

---

## Appendix A: Rollback procedure (per-task)

If any task breaks compilation or tests in a way that can't be
fixed in-place within 15 minutes:

```bash
# Identify the last good commit
git log --oneline

# Revert to the last good commit (soft reset keeps work on disk)
git reset --soft HEAD~1

# Or hard reset if the changes are fully known-bad
git reset --hard <last-good-sha>
```

The `TileType` enum and `From` bridge staying alive through Task 6 is
the safety rail: call sites can mix `TileType` and `TileKind` freely
until Task 7 deletes the bridge. If Task 6's rename introduces a bug,
Task 7 won't mask it -- the pre-Task-7 build still compiles and runs
the full test suite. So regressions surface within Task 6 itself.

Only after Task 7 commits can a bug "hide" behind the removal. That's
why the ordering in Tasks 4 + 5 + 6 + 7 is load-bearing: each step
commits independently; each step's tests pass; rollback at any point
returns to a viable state.

---

## Appendix B: Known risks

1. **Match-arm exhaustiveness fallout (Task 6 Step 3).** `TileType::Shim`
   in a match arm becomes `TileKind::ShimNoc | TileKind::ShimPl`. If
   any match site uses a catch-all (`_ =>`) instead of exhaustive
   variants, rustc won't flag the migration as incomplete. This is
   semantically correct for AIE2 (both shim variants behave identically)
   but will quietly diverge on AIE1. Comment mitigation: wherever a
   `ShimNoc | ShimPl` pattern appears, note in a brief comment that
   AIE1 may want distinct arms for the two variants.

2. **Dead code after bridge deletion (Task 7).** The spec's success
   criterion counts 2708-2709 test passes (baseline 2712 minus 3-4
   deleted bridge tests). If the count is off by more than 4, Task 6 or
   Task 7 broke a test that wasn't a bridge test. Inspect the failure
   list from the next `cargo test --lib` run and bisect.

3. **`is_mem_tile` -> `is_mem` rename name collision.** If any other
   type in the codebase has an `is_mem()` method with different
   semantics, the sed in Task 6 Step 6 accidentally renames those
   too. Mitigation: the sed pattern is `\.is_mem_tile()` (literal) so
   only methods called that way match. If in doubt after Step 6, grep
   for `\.is_mem()` globally and manually verify each site is a
   `TileKind`.

4. **`src/trace/vcd.rs` private `TileType`.** Task 6 explicitly excludes
   this file. If after the sweep the file still compiles but its VCD
   signal names look wrong, someone edited the private enum anyway.
   Inspect `git diff HEAD~N -- src/trace/vcd.rs`.

5. **Forwarder-less fallout from the rename.** Subsystem 6 used
   compatibility forwarders during Part A to keep consumers compiling
   through the relocation. Subsystem 2 does NOT use forwarders -- the
   bridge's existence through Task 6 IS the compatibility layer, and
   it collapses in Task 7. If a consumer outside the ~13 audited files
   appears (e.g., a test binary under `examples/` or `tests/`), Task 6's
   sed pass missed it because the enumeration in Step 1 was limited
   to `src/`. Re-enumerate broadly before Task 7:

   ```bash
   rg -l 'TileType' examples/ tests/ xrt-plugin/
   ```

   Expected: empty. If not, sweep those too before Task 7's deletions.
