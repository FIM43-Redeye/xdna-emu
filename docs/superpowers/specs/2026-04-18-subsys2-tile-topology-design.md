# Subsystem 2 -- Tile Topology -- Design

**Subsystem:** 2 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-18
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-17-subsys6-isa-decode-design.md](2026-04-17-subsys6-isa-decode-design.md)
**Planned tag:** `phase1-subsys-tile-topo`

---

## Goal

Introduce a `TileTopology` trait seam in `xdna-archspec`, replace the xdna-emu
runtime's bespoke `TileType` enum with archspec's canonical `TileKind`
(eliminating the lossy `From` bridge and the `Shim` / `ShimNoc` collapse),
and route the two remaining hardcoded `row == 0` classification sites plus
the four memory-neighbor `row > 0` sites through the new trait. This
subsystem also consumes Phase 1a's deferred `TileType::MemTile -> Mem`
rename and the `TileType::Shim` split into `ShimNoc` / `ShimPl`.

## Non-goals

- **No AIE1 or AIE2P topology implementation.** Subsystem 2 lands the seam
  and the AIE2 impl. Populating AIE1 (with its non-uniform `ShimNoc` /
  `ShimPl` shim columns and alternating-row memory adjacency) or AIE2P is
  orthogonal future work, identical to how Subsystems 1 and 6 left
  second-arch population unstarted.
- **No `shim_mux_kind` method on the trait.** Versal's per-column shim-mux
  distinction is recoverable from `classify(col, 0)` returning `ShimNoc`
  vs `ShimPl`. A separate mux-kind method earns its keep only if sub-kind
  granularity materializes (two `ShimNoc` columns with different mux
  hookups); neither AM020 nor aie-rt suggests that exists. Add it in
  Subsystem 5 (Stream Switch) if a consumer shows up.
- **No runtime behavior change.** The trait is a dispatch seam; current AIE2
  call sites should produce byte-identical decisions before and after. The
  test suite catches any drift.
- **No serialization / FFI surface change.** `TileType` is not serialized
  directly; the rename does not cross any persisted or wire format.
- **No second-arch implementation during the refactor.** Phase 1 ground rule;
  AIE1 / AIE2P topology impls are Phase 2 or later.

---

## Context

Phase 1a consolidated `xdna-archspec` as the single arch-data authority and
introduced `TileKind` (4 variants: `Compute`, `Mem`, `ShimNoc`, `ShimPl`)
alongside a `From<TileType>` / `From<TileKind>` bridge at
`src/device/tile/core_state.rs:80-106`. The bridge is deliberately lossy in
both directions: `ShimNoc | ShimPl -> Shim` on the way down; `Shim ->
ShimNoc` on the way back. This was intentional for AIE2 (emulator never
produces `ShimPl`) and was flagged as a Subsystem 2 follow-up in the Phase
1a audit (`docs/arch/phase1a-audit.md`, `## Follow-ups flagged`).

Subsystem 1 added arch-topology constants (`SHIM_ROW`, `COMPUTE_ROW_START`)
to `xdna_archspec::aie2` generated at build time from `ArchModel`. Most
row-keyed classification in `src/` already routes through those constants;
the audit surfaced only **two** call sites still using bare `row == 0`:

- `src/npu/executor.rs:895` -- shim BD layout dispatch in block-write path.
- `src/device/array/routing.rs:144` -- stream switch packet route filter.

The audit also surfaced four **memory-neighbor** `row > 0` sites that
encode the hardware invariant "shim tiles have no south data memory":

- `src/interpreter/execute/memory/neighbor.rs:80`
- `src/interpreter/execute/memory/neighbor.rs:135`
- `src/interpreter/engine/coordinator.rs:578`
- `src/interpreter/engine/coordinator.rs:705`

Those are the concrete consumers that justify a `neighbor(col, row, dir) ->
Option<(col, row)>` method on the trait -- AIE1's alternating-row memory
adjacency will require a different impl than AIE2's uniform-direction
adjacency. Without a trait, those four sites would remain correct for AIE2
only and silently diverge on AIE1.

The `TileType` rename surface is ~182 direct `TileType::` references across
~13 files, hot-spots at `src/device/tile/{mod.rs, core_state.rs}`,
`src/device/dma/bd.rs`, `src/device/array/routing.rs`,
`src/interpreter/engine/coordinator.rs`, and `src/trace/vcd.rs` (which keeps
its own independent `TileType` enum -- left alone, different module scope).

---

## Audit (actual state, not narrative)

| Concern | Count | Source |
|---|---|---|
| `TileType::` references in `src/` | ~182 | grep |
| Files touching `TileType` | ~13 | grep |
| Bare `row == 0` classification hardcodes | 2 | `executor.rs:895`, `routing.rs:144` |
| Memory-neighbor `row > 0` sites | 4 | `neighbor.rs:80,135`, `coordinator.rs:578,705` |
| Row checks already routed through archspec constants | 10+ | `test_runner.rs`, `registers.rs`, `routing.rs` |
| Pure bounds checks (`row < self.rows`) | 4+ | Not topology; left as-is |
| Named tests covering tile-type-specific behavior | 16 | `dma/engine/tests.rs`, `dma/bd.rs`, `array/tests.rs` |
| `TileType` serialization / FFI exposure | 0 | No `#[derive(Serialize)]`, no FFI export |
| Public re-exports of `TileType` | 2 | `src/device/tile/mod.rs:38`, `src/device/mod.rs:71` |

Both `TileType` and `TileKind` are plain enums without associated fields,
so the rename is a mechanical variant-name substitution plus one semantic
decision: every `TileType::Shim` match arm must become `TileKind::ShimNoc |
TileKind::ShimPl` (preserving current behavior -- the emulator treats them
identically today and continues to do so under AIE2).

---

## Approach

Single approach (the brainstorm validated a trait seam upfront rather than
deferring like Subsystems 1 and 6 did). Four concerns folded into one
subsystem:

### Section 1: Architecture

Add a `TileTopology` trait to `xdna-archspec` colocated with `ArchConfig`.
Minimal surface: `classify` for tile-kind lookup and `neighbor` for memory
adjacency. `ArchModel` exposes a `topology()` accessor returning a
trait-object reference. The AIE2 impl is a concrete zero-ish-sized struct
(`Aie2Topology`) that reads the existing `SHIM_ROW` / `COMPUTE_ROW_START`
constants and `ArchModel`'s column / row extents; it holds no state beyond
what `ArchModel` already has.

Consumers (xdna-emu `src/`) call `arch_model.topology().classify(col, row)`
or `.neighbor(col, row, Direction::South)` instead of bare row comparisons
or `TileType::from_row(row)`-style helpers.

`TileType` deletes. All sites import `xdna_archspec::TileKind` directly.
The `From` bridge at `src/device/tile/core_state.rs:80-106` deletes.
Predicate helpers (`is_shim`, `is_mem`, `is_compute`) live next to
`TileKind` in archspec so every consumer crate gets them for free.

### Section 2: Components

**New: `crates/xdna-archspec/src/topology.rs`** (or equivalent module path
chosen during implementation):

```rust
pub enum Direction { North, South, East, West }

pub trait TileTopology: Send + Sync {
    fn classify(&self, col: u8, row: u8) -> TileKind;
    fn neighbor(&self, col: u8, row: u8, dir: Direction) -> Option<(u8, u8)>;
}
```

**New: `Aie2Topology` struct + impl** (location likely
`crates/xdna-archspec/src/aie2/topology.rs`):

- `classify`: `row == SHIM_ROW -> ShimNoc`; `SHIM_ROW < row < COMPUTE_ROW_START ->
  Mem`; `row >= COMPUTE_ROW_START -> Compute`. Column index is accepted but
  ignored on AIE2 (every shim column is `ShimNoc`).
- `neighbor`: straightforward grid traversal respecting the `ArchModel`
  column / row extents. North / South / East / West all one step, clamped
  at array boundaries. `South` returns `None` when `row == SHIM_ROW`; this
  is what `coordinator.rs:578, 705` and `neighbor.rs:80, 135` want.

**New: `TileKind` inherent predicate methods** in archspec:

```rust
impl TileKind {
    pub const fn is_shim(self) -> bool {
        matches!(self, TileKind::ShimNoc | TileKind::ShimPl)
    }
    pub const fn is_mem(self) -> bool {
        matches!(self, TileKind::Mem)
    }
    pub const fn is_compute(self) -> bool {
        matches!(self, TileKind::Compute)
    }
}
```

Inherent methods (not free functions) because `kind.is_shim()` reads as
a hardware truth about the discriminant, not an emulator-side convention
-- it matches how every other discriminant-predicate method in the
codebase is shaped. `const fn` lets the predicates participate in
`const`-evaluated contexts if they come up.

**New: `ArchModel::topology(&self) -> &dyn TileTopology`.** Trait-object
return is the committed shape -- it keeps consumer code non-generic and
gives AIE1's eventual `Aie1Topology` impl a drop-in hook without
re-plumbing the accessor. The concrete object lives inside `ArchModel`
(built during `load_from_json`, stored as a `Box<dyn TileTopology>`
field) or as a `&'static` reference to a zero-sized impl -- that storage
choice is a plan-level detail that does not escape archspec's internals.

**Dissolved: `src/device/tile/core_state.rs::TileType`.** The enum
definition, its three predicate methods (`is_shim`, `is_mem_tile`,
`is_compute`), and the two `From` impls all delete. The round-trip tests at
`src/device/tile/core_state.rs:127-163` delete with them.

**Dissolved: public re-exports of `TileType`** at
`src/device/tile/mod.rs:38` and `src/device/mod.rs:71`. Callers that want
the type import `xdna_archspec::TileKind` directly.

### Section 3: Data flow

**Classification (cold path, once per tile at config / init):**

1. `src/npu/executor.rs:895` (shim BD layout dispatch): replace `row == 0`
   with `arch_model.topology().classify(col, row) == TileKind::ShimNoc`
   (or, since col is irrelevant on AIE2, the archspec constant `SHIM_ROW`
   -- plan picks the less-noisy form; both produce the same code on AIE2).
2. `src/device/array/routing.rs:144` (stream switch packet routing, filters
   shim row in route enumeration): same treatment.

**Memory-neighbor navigation (interpreter path, per cross-tile access):**

1. `src/interpreter/execute/memory/neighbor.rs:80`: replace
   `if self.row > 0 { Some((self.col, self.row - 1)) } else { None }` with
   `topology.neighbor(self.col, self.row, Direction::South)`.
2. `src/interpreter/execute/memory/neighbor.rs:135`: same pattern, inside
   the `MemoryQuadrant::South` arm.
3. `src/interpreter/engine/coordinator.rs:578, 705`: same treatment.

These four sites are where the `neighbor()` method earns its keep: the
AIE1 impl will produce a different answer for the same `(col, row,
Direction::South)` on odd-row compute tiles, and the consumers should
pick that up automatically.

**Hot-path cost:** `topology()` returns a reference; `classify` and
`neighbor` are small branch trees. On AIE2 both methods compile to
essentially the same conditionals we replaced. No new allocations, no new
dyn-dispatch fan-out beyond the single `&dyn TileTopology` indirection that
the optimizer devirtualizes when the concrete type is knowable (which it is
for the monomorphic AIE2 build).

### Section 4: Rename scope

**Mechanical substitutions** (sed-style, verified by compilation + tests):

- `TileType::Shim` -> `TileKind::ShimNoc | TileKind::ShimPl` in match arms;
  `TileKind::ShimNoc` in construction sites (AIE2 never produces ShimPl).
- `TileType::MemTile` -> `TileKind::Mem` everywhere.
- `TileType::Compute` -> `TileKind::Compute` everywhere.
- `tile_type: TileType` (field / parameter declarations) -> `tile_kind:
  TileKind` (rename the binding too, per Phase 1a audit follow-up --
  consistency with archspec terminology).
- `TileType` imports replaced with `TileKind` imports.

**Manual review required** for:

- Match arms that previously fell through on `TileType::Shim` -- audit
  each one to confirm `ShimNoc | ShimPl` is the correct collapse (AIE2:
  yes, always; AIE1: will differ, but that impl isn't landing this
  subsystem).
- `TileType::from(_)` / `TileKind::from(_)` call sites: delete the bridge
  call and the wrapper, replace with direct `TileKind`.
- `tile_type.is_shim()` / `is_mem_tile()` / `is_compute()` method calls:
  rewrite to the archspec helper or to inline `matches!`.
- Test assertions referring to `TileType::Shim` etc. -- update to
  `TileKind` variants (many tests in `dma/engine/tests.rs` and
  `array/tests.rs`).

**Scope decisions:**

- `src/trace/vcd.rs::TileType` is a separate private enum used only for
  VCD signal naming; it does not import from `core_state.rs`. Leave it
  alone unless renaming it helps clarity -- the plan can decide. (My
  lean: leave it, document the distinction in a one-line comment.)
- No `pub type TileType = TileKind` deprecation alias. We are on `dev`
  and no master merges happen during Phase 1; a lingering alias only
  delays the cleanup and risks re-introduction elsewhere.

### Section 5: Testing and verification

**Global invariants (every commit):**

- `cargo test --lib` green (baseline: 2712 passed; 0 failed; 5 ignored at
  `phase1-subsys-isa-decode`).
- `cargo test -p xdna-archspec --lib` green (baseline: 220 passed; 1
  pre-existing fail `device_model::test_full_parse_all_devices`; 2 ignored).
- `cargo build` and `cargo build --release` green.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green after
  rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`).

**Smoke-test matrix after each task:**

The 16 tile-type-specific named tests identified in the audit (expected to
all pass before and after):

- `test_mem_tile_engine`
- `test_memtile_bd_capacity`
- `test_memtile_port_mappings`
- `test_resolve_lock_id_memtile`
- `test_memtile_mm2s_reads_from_east_neighbor`
- `test_memtile_mm2s_reads_from_west_neighbor`
- `test_memtile_s2mm_writes_to_east_neighbor`
- `test_memtile_mm2s_out_of_window_addr_records_fatal_error`
- `test_memtile_bd_channel_validity_*` (5 variants)
- `test_memtile_bd_parsing`
- `test_shim_bd_parsing`
- `test_tile_types`
- `test_cascade_route_*` (3 variants)

Ship a `TileTopology` unit test suite in archspec covering:

- `Aie2Topology::classify` at every `(col, row)` in a canonical 5x6 NPU1
  grid: row 0 is `ShimNoc`, row 1 is `Mem`, rows 2-5 are `Compute`.
- `Aie2Topology::neighbor` at the four corners, at shim row
  (`South` returns `None`), at top row (`North` returns `None`), and at
  east / west edges.
- Round-trip consistency: for every `(col, row)` with `North/South/East/
  West` returning `Some`, the reverse direction from the returned cell
  returns the original.

Target: ~10-12 new archspec tests. Global xdna-emu count should match the
baseline minus the 3 deleted `From` round-trip tests (so 2709 after -- plan
confirms the exact number from whatever else shifts).

**Per-subsystem gate (at the tag):**

1. Rebuild the FFI cdylib: `cargo build -p xdna-emu-ffi`.
2. `./scripts/emu-bridge-test.sh 2>&1 | tee
   /tmp/claude-1000/bridge-subsys2.log` -- full HW bridge run, ~30 min.
   Compare pass/fail matrix to the `phase1-subsys-isa-decode` baseline;
   no new regressions. `bd_chain_repeat_on_memtile` remains a known
   pre-existing failure.
3. `./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys2.log`
   -- ISA test suite, ~10 min. Expect `FAIL: 0` across all test points.
4. Bridge and ISA run sequentially (never concurrently).

Expected verification cost: ~45 min at the tag.

### Section 6: Scope gating

Subsystem 2 is a single-part deliverable. Unlike Subsystem 6 (9,300 lines
relocated) or Subsystem 1 (10 tasks across the build-side / runtime
boundary), Subsystem 2 is:

- ~180 LOC of new code in archspec (trait, `Aie2Topology`, helpers, tests).
- ~200 LOC deleted (TileType enum, bridge, tests, predicate methods).
- ~50 call sites changed (the 6 row-keyed hardcodes + a subset of the
  182 `TileType::` refs, many of which sed handles mechanically).
- No build-system moves. No FFI changes. No second-arch population.

One part, one tag. If the rename sprawl is larger than the audit suggests
once we're in the code, a Part A / Part B split can happen opportunistically
(Part A = trait + hardcode fixes; Part B = rename), but the default plan
is one part.

**Task sequence (the plan will flesh these out):**

1. Scaffold design note (`docs/arch/tile-topology.md`) + audit doc
   (`docs/arch/subsys2-audit.md`), template-matched to `isa-decode.md` and
   `subsys6-audit.md`. Completion sections filled later.
2. Add `TileTopology` trait + `Direction` enum + `Aie2Topology` impl to
   archspec. Wire `ArchModel::topology()`. Ship unit tests for
   `Aie2Topology`.
3. Add `TileKind` predicate helpers (`is_shim`, `is_mem`, `is_compute`)
   in archspec.
4. Replace the 2 bare `row == 0` hardcodes with `topology().classify(..)`
   (or the archspec constant, if col is irrelevant at that site).
5. Replace the 4 memory-neighbor `row > 0` sites with
   `topology().neighbor(.., Direction::South)`.
6. `TileType` -> `TileKind` rename pass: sed + manual match-arm review
   across the ~13 consumer files.
7. Delete the `From` bridge, the `TileType` enum, and its predicate
   methods. Update re-exports in `src/device/{mod.rs, tile/mod.rs}`.
8. Fill completion sections in `docs/arch/tile-topology.md` +
   `docs/arch/subsys2-audit.md`. Run full HW + ISA gate. Tag
   `phase1-subsys-tile-topo`. Update `NEXT-STEPS.md` to point at
   Subsystem 3 (DMA) as up next.

---

## Why a trait this time

Subsystem 6 documented "no trait" because ISA decode is values (TableGen
data differs; the walker algorithm is invariant). Subsystem 2 introduces a
trait because **tile topology has genuine shape differences across arch
families**, not just value differences:

- **AIE2 (NPU1/NPU2):** 5x6 (or larger) grid; row 0 is `ShimNoc`; row 1 is
  `Mem`; rows 2-5 are `Compute`. All `ShimNoc`, no `ShimPl`. Memory
  adjacency is uniform-direction for all compute rows.
- **AIE1 (Versal, e.g., xcvc1902):** 50x9 grid; row 0 is shim, but some
  columns are `ShimNoc` and others are `ShimPl` (the PL-fabric variant is
  real, unlike AIE2). There is **no memtile row** (`num_mem_tile_rows = 0`).
  Memory adjacency alternates direction by row parity -- compute tiles at
  odd rows share memory with north, even rows share with south. `classify`
  must dispatch on column, not just row, and `neighbor` must dispatch on
  row parity.
- **AIE2P (NPU4/5/6):** Similar to AIE2 in shape but different extents
  and potentially different memtile placement. Expected to share most of
  AIE2's impl with tweaked constants.

A shared `Aie2Topology` impl that delegates to `ArchModel` data works for
all three AIE2-family devices because their shape is parameterized by the
same constants. AIE1 needs a separate `Aie1Topology` impl with genuinely
different logic -- that is the shape difference that a data-only approach
cannot express without sprinkling `if arch_family == Aie1 { ... }`
conditionals through consumer code.

The "what would AIE1 look like?" exercise from the parent spec resolves
cleanly: `Aie1Topology::classify(col, row)` reads from an
`ArchModel::shim_columns: Vec<ShimColumnKind>` field; `neighbor`
branches on `row & 1`. Both fit the trait shape, neither fits a
data-only model.

---

## Forward pointers

- **Subsystem 3 (DMA Engine & BD Format).** First behavioral seam in the
  DMA space; expected to introduce a `DmaModel` trait. `TileKind` /
  `TileTopology` will appear in the DMA code paths that dispatch on tile
  type (shim DMA vs memtile DMA vs compute DMA have different BD layouts);
  this subsystem makes that dispatch type-safe.
- **Subsystem 5 (Stream Switch).** Revisit whether `shim_mux_kind(col) ->
  ShimMuxKind` belongs on `TileTopology`. If the stream switch legality
  checks want per-column ShimNoc / ShimPl distinction beyond what
  `classify(col, 0)` already returns, add the method there.
- **Phase 2 hygiene.** The audit flagged several tile-topology-adjacent
  items for later cleanup: silent DMA channel fallback `(2, 2)` in
  `from_arch_model()` (Subsystem 3), inner-scope `TileKind` imports
  repeated in `model.rs` test fns (Phase 2), overflow-unguarded
  `columns: topo.columns + 1` (Phase 2).

---

## Deliverables checklist

- [ ] `TileTopology` trait + `Direction` enum in `xdna-archspec`.
- [ ] `Aie2Topology` concrete impl with `classify` and `neighbor`.
- [ ] `ArchModel::topology(&self)` accessor returning the trait.
- [ ] `TileKind` predicate helpers (`is_shim`, `is_mem`, `is_compute`)
      in archspec.
- [ ] The 2 bare `row == 0` hardcodes rewritten to `classify()` /
      archspec constant.
- [ ] The 4 memory-neighbor `row > 0` sites rewritten to
      `neighbor(.., Direction::South)`.
- [ ] `TileType` enum deleted from `src/device/tile/core_state.rs`.
- [ ] `From<TileKind> for TileType` + `From<TileType> for TileKind`
      bridge deleted.
- [ ] Public re-exports of `TileType` removed from
      `src/device/{mod.rs, tile/mod.rs}`.
- [ ] All ~182 `TileType::` references replaced with `TileKind::*`
      across the ~13 consumer files.
- [ ] ~10-12 new `Aie2Topology` unit tests in archspec.
- [ ] `docs/arch/tile-topology.md` written (mandatory per-seam design
      note; template from `isa-decode.md`).
- [ ] `docs/arch/subsys2-audit.md` written (template from
      `subsys6-audit.md`, including a Completion section).
- [ ] Full HW bridge + ISA suite green at the tag.
- [ ] `NEXT-STEPS.md` updated to point at Subsystem 3 (DMA Engine) as
      up next.
- [ ] Tag: `phase1-subsys-tile-topo`.

---

## Success criteria (must all hold at the final tag)

1. `cargo test --lib` passes; count is approximately 2712 +/- a small
   delta (the 3 deleted bridge round-trip tests + whatever else the
   rename drops or adds; plan will nail the exact number).
2. `cargo test -p xdna-archspec --lib` passes with ~10-12 new tests beyond
   the 220-pass baseline, with the same 1 pre-existing failure
   (`device_model::test_full_parse_all_devices`) and 2 ignored.
3. `cargo build --release` clean.
4. Full HW bridge run shows no new regressions vs.
   `phase1-subsys-isa-decode` baseline. `bd_chain_repeat_on_memtile`
   remains a known pre-existing EMU failure.
5. ISA test suite: `FAIL: 0`.
6. Grep across `src/` returns zero occurrences of `TileType::` (modulo
   `src/trace/vcd.rs`'s private enum, which is a distinct type).
7. Grep across `src/` returns zero bare `row == 0` classification sites;
   all row-keyed classification routes through archspec (either a
   constant or `topology().classify`).
8. `src/device/tile/core_state.rs` contains no `TileType` enum definition,
   no predicate methods on it, and no `From` bridge.
9. No `TODO` / `FIXME` / `unimplemented!()` without an open-issue
   reference.
10. All commits land on `dev`; no master merges during the subsystem.

---

## Ground rules (inherited from the parent refactor spec)

- **No master merges during the refactor.** Everything lands on `dev`.
- **`cargo test --lib` green at every commit.** Non-negotiable.
- **Bridge test smoke green at every subsystem tag.** Full HW run before
  the tag.
- **One authoritative source per concept.** Once `TileKind` is the tile
  classifier, `TileType` must not reappear anywhere outside
  `src/trace/vcd.rs`'s private scope.
- **Traits decode / step / check; they do not hold mutable state.**
  `TileTopology` reads from `ArchModel`; it does not mutate anything.
- **Coarse first.** Two methods on `TileTopology`, not five. `classify`
  + `neighbor` cover the identified consumers. `shim_mux_kind` is
  explicitly deferred.
- **"What would AIE1 look like?" design note per seam.** Written into
  `docs/arch/tile-topology.md` as part of Subsystem 2's deliverable.
- **No second-arch implementation during the refactor.** Subsystem 2
  ships the AIE2 impl only. AIE1 / AIE2P impls are follow-on.
