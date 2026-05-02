# Device-Family Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare xdna-emu to support multiple AIE architectures (AIE2P, AIE1/Versal) by consolidating the arch abstraction and adding per-subsystem behavioral seams, without implementing any second architecture.

**Architecture:** Phase 1a collapses three parallel arch shims into a single `xdna-archspec` source-of-truth. Phases 1b/1c walk eight subsystems in order, each plumbing data through archspec and, where behavior genuinely varies across archs, lifting it behind a trait (with AIE2 as the sole implementation). Phase 2 is hygiene driven by what Phase 1 exposes -- large-file splits, module reshapes, possible crate lifts. All work lands on `dev` with per-subsystem tags; no master merges.

**Tech Stack:** Rust 2021, `xdna-archspec` workspace crate, `Confirmed<T>` cross-validation, cargo workspaces, git tags for bisect navigation.

**Spec:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

---

> **Sweep-as-of 2026-05-01:** Phase 1 of this plan completed -- tag `phase1-complete` (commit 0e13979) marks the close. All eight subsystems landed via per-subsystem tags (phase1a-consolidate, phase1-subsys-regs-mem, phase1-subsys-tile-topo, phase1-subsys-dma, phase1-subsys-locks, phase1-subsys-stream-switch, phase1-subsys-isa-decode, phase1-subsys-isa-execute, phase1-subsys-parser-*). Phase 2 hygiene items D.1, D.2, D.4-D.7 also landed (see chore(phase2-hygiene) commits). D.3 (universal register bus) landed separately via tag-less commits (11d2b66 / 61d3a94 / 9e5fecf / d297b98). Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Scope Note

This plan covers **Phase 1a in full bite-sized detail** (the archspec consolidation -- a self-contained, shippable deliverable). Phase 1b/1c subsystem passes and Phase 2 hygiene are sketched at section level; **each subsystem gets its own detailed plan at the moment we start that subsystem**, because the bite-sized task list depends on a fresh audit of then-current code. Writing 8 subsystem plans up-front would produce plausible fiction, not actionable tasks.

When starting any of sub-subsystems 1-8, invoke the `brainstorming` / `writing-plans` skills fresh with the audit findings of that moment.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green.
- `cargo build` green (release optional per commit; required before subsystem tag).
- No commit introduces `TODO` / `FIXME` / `unimplemented!()` without an open-issue reference.
- Commit messages follow repo style (lowercase type prefix: `refactor:`, `docs:`, `test:`; no emoji; ends with "Generated using Claude Code.").
- All work on `dev` branch. No merges to `master` under any circumstance during this plan.

---

## Phase 1a: Consolidate the Arch Abstraction

**Tag at end:** `phase1a-consolidate`

**Current state:**
- `crates/xdna-archspec/` -- full `ArchModel` data type with `Confirmed<T>` cross-validation. Multi-arch aware (AIE, AIE2, AIE2P). 9 files in `src/` import from it.
- `src/archspec/mod.rs` -- 16-line re-export shim: `pub use xdna_archspec::...`. Flattens 4 commonly-used types.
- `src/device/arch_config.rs` -- 729-line `ArchConfig` trait (~20 methods: dimensions, tile classification, memory sizes, DMA, lock counts, stream-switch port layouts, arch name) plus a single `ModelConfig` impl built from `ArchModel` via `from_arch_model()`. Also owns `default_arch() -> Arc<dyn ArchConfig>` and the `ARCHSPEC_MODELS` lazy-static cache.

**Consuming files (8 outside the shims themselves):**
`src/device/model.rs`, `src/device/state/mod.rs`, `src/device/registers.rs`, `src/device/regdb/tests.rs`, `src/device/regdb/mod.rs`, `src/device/regdb/field_layouts.rs`, plus any others picked up during Task 1's audit.

**Target state:**
- Exactly one arch abstraction: `xdna-archspec` crate, with a new `runtime` module containing the `ArchConfig` trait, `ModelConfig` impl, `default_arch()`, and the `ARCHSPEC_MODELS` cache.
- `src/archspec/mod.rs` deleted. `src/device/arch_config.rs` deleted.
- Consuming files import directly from `xdna_archspec::{types, runtime}`.
- All tests green.

### Task 1: Audit Artifact

**Files:**
- Create: `docs/arch/phase1a-audit.md`

- [x] **Step 1: Enumerate ArchConfig trait surface**

Read `src/device/arch_config.rs` end-to-end. For each method on the `ArchConfig` trait, record:
- Method signature (name, params, return type).
- Implementation body in `ModelConfig` (source of the data: `ArchModel` field, `crate::arch` generated table, or hardcoded).
- Any external type dependencies (`TileType`, `Arc`, etc.).

Write results to `docs/arch/phase1a-audit.md` under a `## ArchConfig Trait Surface` heading as a table.

- [x] **Step 2: Enumerate consumers**

Run:
```bash
rg -l 'use (crate::archspec|crate::device::arch_config|xdna_archspec)' src/
```

For each file listed, grep for the specific symbols imported and record them in the audit under `## Consumers and Imports`. Include the exact `use` lines verbatim for the migration step later.

- [x] **Step 3: Identify type dependencies that must cross the crate boundary**

`ArchConfig` references `TileType` from `src/device/tile.rs`. The crate has its own `TileKind` in `xdna_archspec::types`. Record in the audit under `## Type Boundary`:
- Is `TileType` structurally identical to `TileKind`? (Read both.)
- If yes: plan a `From`/`Into` bridge or a rename-and-reuse.
- If not: record the differences, then plan: move `TileType` into the crate (preferred), or adapt `ArchConfig` to use `TileKind` (acceptable).

- [x] **Step 4: Identify stream-switch port data dependency**

`ArchConfig` port-layout methods (`master_ports`, `slave_ports`, `north_master_range`, etc.) use `crate::arch` -- a build.rs-generated module. The audit must answer: does `crate::arch` derive from `xdna-archspec` at build time, or from some other source? Run:
```bash
rg 'crate::arch' src/device/arch_config.rs
cat build.rs | head -80
```

Record under `## Port Data Dependency`: where `crate::arch` comes from, and whether moving `ArchConfig` into the crate requires moving the port-data generation too. If the port tables are generated from archspec itself, the move is straightforward. If they're sourced from a separate AM025 path, either (a) move that path into the crate, or (b) keep port-layout methods as a thin runtime-side extension trait outside the crate.

- [x] **Step 5: Commit the audit**

```bash
mkdir -p docs/arch
git add docs/arch/phase1a-audit.md
git commit -m "$(cat <<'EOF'
docs: audit archspec consolidation surface

Records the ArchConfig trait surface, its consumers, type
boundary with xdna-archspec's TileKind, and the stream-switch
port-data dependency. Feeds the Phase 1a migration tasks.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: Reconcile TileType with TileKind

**Decision gate:** If the audit (Task 1 Step 3) found `TileType` and `TileKind` are structurally identical, this task aligns them. If they differ, this task adapts.

**Files:**
- Read: `src/device/tile.rs`, `crates/xdna-archspec/src/types.rs`
- Modify one of: `crates/xdna-archspec/src/types.rs` (add variants if needed), or `src/device/tile.rs` (re-export `TileKind` as `TileType`).

- [x] **Step 1: Compare the two enums**

Read both files. Write the comparison into the audit doc under `## TileType vs TileKind Resolution`, including:
- Variants in each.
- Any differences in derive traits (`PartialEq`, `Eq`, `Hash`, etc.).
- Any methods on `TileType` not present on `TileKind`.

- [x] **Step 2: Add missing methods/derives to TileKind in the crate if needed**

If `TileType` has methods or derives that `TileKind` lacks but consumers need, add them to `TileKind` in `crates/xdna-archspec/src/types.rs`. Keep `TileType` as a re-export alias in `src/device/tile.rs` for source-compatibility of the rest of the codebase:

```rust
// src/device/tile.rs
pub use xdna_archspec::types::TileKind as TileType;
```

- [x] **Step 3: Run tests**

```bash
cargo test --lib
```
Expected: PASS (this task is source-compatible — `TileType` is the same type alias).

- [x] **Step 4: Commit**

```bash
git add src/device/tile.rs crates/xdna-archspec/src/types.rs
git commit -m "$(cat <<'EOF'
refactor: alias TileType to xdna_archspec::TileKind

Eliminates the parallel enum so ArchConfig can be moved into
the archspec crate without a type-boundary problem. Consumers
still import TileType; it now resolves to the single canonical
TileKind in the archspec crate.

Generated using Claude Code.
EOF
)"
```

---

### Task 3: Move ArchConfig Trait into xdna-archspec::runtime

**Files:**
- Create: `crates/xdna-archspec/src/runtime.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod runtime;`)
- Reference (do not yet delete): `src/device/arch_config.rs`

- [x] **Step 1: Create runtime.rs**

Copy the *entire body* of `src/device/arch_config.rs` into `crates/xdna-archspec/src/runtime.rs`. Then:
- Replace `use super::tile::TileType;` with `use crate::types::TileKind as TileType;` (or keep `TileType` alias).
- Replace `crate::archspec::ArchModel` with `crate::types::ArchModel`.
- Replace `crate::archspec::TileKind` with `crate::types::TileKind`.
- For any reference to `crate::arch` (build.rs-generated), replace with the resolution chosen in Task 1 Step 4. If the port-data path is moving into the crate, import from the new crate-internal location; if it's staying runtime-side, add a `PortDataSource` trait parameter to the functions that need it and leave the callsite changes for later tasks.

- [x] **Step 2: Expose the module**

Add to `crates/xdna-archspec/src/lib.rs`:
```rust
pub mod runtime;
```
Place alphabetically among the other `pub mod` lines.

- [x] **Step 3: Build**

```bash
cargo build
```
Expected: PASS. Fix compile errors in `runtime.rs` until clean. Errors here mean missing imports or a type mismatch the audit didn't catch — resolve before moving on.

- [x] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/runtime.rs crates/xdna-archspec/src/lib.rs
git commit -m "$(cat <<'EOF'
refactor: add xdna_archspec::runtime with ArchConfig trait

Copies the ArchConfig trait, ModelConfig impl, default_arch,
and ARCHSPEC_MODELS cache into the archspec crate. The original
src/device/arch_config.rs still exists; consumers will migrate
in the next task, and the original will be deleted afterward.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Migrate Consumers to xdna_archspec::runtime

**Files (from audit Task 1 Step 2):**
- Modify each file that imports from `crate::device::arch_config` or `crate::archspec`.

**Pattern:** one file per commit. Each commit leaves `cargo test --lib` green.

- [x] **Step 1: Migrate `src/device/model.rs`**

Replace `use crate::device::arch_config::{ArchConfig, ModelConfig, ...};` with `use xdna_archspec::runtime::{ArchConfig, ModelConfig, ...};`. Replace `use crate::archspec::{...};` with `use xdna_archspec::types::{...};`.

Run:
```bash
cargo build
cargo test --lib
```
Expected: PASS.

Commit:
```bash
git add src/device/model.rs
git commit -m "refactor: migrate device::model to xdna_archspec::runtime

Generated using Claude Code."
```

- [x] **Step 2: Migrate `src/device/state/mod.rs`**

Same pattern. Build + test + commit.

- [x] **Step 3: Migrate `src/device/registers.rs`**

Same pattern. Build + test + commit.

- [x] **Step 4: Migrate `src/device/regdb/mod.rs`**

Same pattern. Build + test + commit.

- [x] **Step 5: Migrate `src/device/regdb/field_layouts.rs`**

Same pattern. Build + test + commit.

- [x] **Step 6: Migrate `src/device/regdb/tests.rs`**

Same pattern. Build + test + commit.

- [x] **Step 7: Sweep for any remaining consumers**

```bash
rg 'crate::archspec|crate::device::arch_config' src/
```
Expected: empty output, or only the shim files themselves (`src/archspec/mod.rs` and `src/device/arch_config.rs`).

If any consumer was missed, migrate it following the same pattern and commit.

---

### Task 5: Delete the Parallel Shims

**Files:**
- Delete: `src/archspec/mod.rs`, `src/archspec/` directory
- Delete: `src/device/arch_config.rs`
- Modify: `src/lib.rs` (remove `pub mod archspec;`)
- Modify: `src/device/mod.rs` (remove `pub mod arch_config;`)

- [x] **Step 1: Remove module declarations**

Edit `src/lib.rs`: delete the `pub mod archspec;` line.
Edit `src/device/mod.rs`: delete the `pub mod arch_config;` line.

- [x] **Step 2: Delete the files**

```bash
git rm src/archspec/mod.rs
rmdir src/archspec
git rm src/device/arch_config.rs
```

- [x] **Step 3: Build and test**

```bash
cargo build
cargo test --lib
```
Expected: PASS. Any failures indicate a consumer was missed in Task 4 — go back and migrate it, then return here.

- [x] **Step 4: Verify nothing references the deleted paths**

```bash
rg 'crate::archspec|crate::device::arch_config|src/archspec|src/device/arch_config' .
```
Expected: no matches outside of git history / docs / archive.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: delete parallel arch shims

Removes src/archspec/mod.rs (trivial re-export wrapper) and
src/device/arch_config.rs (729-line trait + impl, now in
xdna_archspec::runtime). Consumers migrated in prior commits.
Single source-of-truth for the arch abstraction: the
xdna-archspec crate.

Generated using Claude Code.
EOF
)"
```

---

### Task 6: Verification and Tag

**Files:** none modified. This is the gate.

- [x] **Step 1: Full library tests**

```bash
cargo test --lib
```
Expected: PASS, all tests.

- [x] **Step 2: Release build**

```bash
cargo build --release
```
Expected: PASS. Release-only compile errors occasionally surface here.

- [x] **Step 3: Bridge test no-hw smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```
Expected: PASS. The `add_one` test is a minimal bridge-path smoke.

- [x] **Step 4: Full bridge test (hardware)**

```bash
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/phase1a-bridge.log
```
Expected: no regressions compared to pre-refactor baseline. If any test that was passing is now failing, stop and debug before tagging.

- [x] **Step 5: Tag**

```bash
git tag phase1a-consolidate
```

- [x] **Step 6: Log completion**

Append to `docs/arch/phase1a-audit.md` a `## Completion` section recording: commit hashes, test counts before/after, any surprises found during migration, and any follow-up items flagged for later subsystems. Commit:

```bash
git add docs/arch/phase1a-audit.md
git commit -m "docs: phase1a completion log

Generated using Claude Code."
```

---

## Phase 1b/1c: Per-Subsystem Plumb + Seam

Each subsystem below gets its own detailed plan written via the `writing-plans` skill at the moment we start it. The sketch here is for awareness only.

### Subsystem 1 — Registers & Memory Map
**Tag:** `phase1-subsys-regs-mem`
**Scope:** plumb all hardcoded register offsets, memory sizes, and per-tile-type counts through `ArchModel` queries. Likely no new trait; mostly data plumbing and consolidation.
**Key files:** `src/device/registers.rs`, `src/device/regdb/*`, `src/device/state/*`, any `const` register offsets scattered across subsystems.

### Subsystem 2 — Tile Topology
**Tag:** `phase1-subsys-tile-topo`
**Scope:** replace any `row == 0` / `row >= N` checks with `ArchModel`-backed classification. Memtile/shim/compute assumptions routed through archspec.
**Key files:** `src/device/tile/*`, `src/device/array/*`, `src/device/model.rs`.

### Subsystem 3 — DMA Engine & BD Format
**Tag:** `phase1-subsys-dma`
**Scope:** first behavioral seam. Audit AIE2 BD layout vs AIE1 BD layout via aie-rt source. Lift BD parse/encode and channel stepping behind a `DmaModel` trait. AIE2 becomes the sole impl.
**Key files:** `src/device/dma/bd.rs`, `src/device/dma/engine/*`.

### Subsystem 4 — Locks
**Tag:** `phase1-subsys-locks`
**Scope:** lock acquire/release/value semantics. Small exercise; good test of trait-sizing discipline. Seam: `LockModel` trait if behavior genuinely differs across AIE1/AIE2 (likely around lock value width).
**Key files:** `src/device/locks/*` (check audit) or wherever locks currently live.

### Subsystem 5 — Stream Switch
**Tag:** `phase1-subsys-stream-switch`
**Scope:** port topology (data, from archspec) plus routing legality rules (behavior → trait). Seam: `StreamSwitchModel` trait.
**Key files:** `src/device/stream_switch/*`, `src/device/array/routing.rs`.

### Subsystem 6 — ISA Decode
**Tag:** `phase1-subsys-isa-decode`
**Scope:** bundle/slot layout, decoder tables. AIE1 (3-slot VLIW, 128-bit vectors) vs AIE2 (6-slot, 256-bit) is the biggest single arch cliff in the codebase. Seam: `IsaDecoder` trait. Decoder tables continue to be generated from llvm-aie TableGen per-arch.
**Key files:** `src/interpreter/decode/*`, `src/interpreter/bundle/*`, `src/tablegen/*`.

### Subsystem 7 — ISA Execute
**Tag:** `phase1-subsys-isa-execute`
**Scope:** semantic ops, intrinsic handlers. The 239KB `vmac_routing.rs`, 124KB `memory/mod.rs`, 102KB `vector_arith.rs` live here. Seam: `IsaExecutor` trait. This is also the subsystem where Phase 2 hygiene will lean hardest — plan it with both concerns in mind.
**Key files:** `src/interpreter/execute/*`, `src/interpreter/engine/coordinator.rs`, `src/interpreter/state/context.rs`.

### Subsystem 8 — Parser (XCLBIN / PDI / ELF)
**Tag:** `phase1-subsys-parser`
**Scope:** container format variance. XDNA uses XCLBIN + PDI sections; Versal has different PDI variants. Seam: `BinaryLoader` trait. At refactor end, XDNA is the sole impl.
**Key files:** `src/parser/xclbin.rs`, `src/parser/cdo.rs`, `src/parser/aie_partition.rs`, `src/parser/elf.rs`.

---

## Phase 2: Hygiene

**Tag(s):** `phase2-hygiene-*` grouped by intent.

Driven by what Phase 1 exposes. Structure decided after Phase 1's seams are in. Visible targets today (to be reconfirmed post-Phase 1):

- Large file splits: `interpreter/execute/vmac_routing.rs` (239KB), `interpreter/execute/memory/mod.rs` (124KB), `vector_arith.rs` (102KB), `fuzzer/trace_sweep.rs` (98KB), `trace/compare.rs` (94KB), `interpreter/decode/decoder.rs` (87KB).
- Module boundary rationalization: `src/config.rs`, `src/build_progress.rs`, other top-level one-offs.
- Naming audit: sweep for obsolete or duplicated-intent names left over from earlier iterations.
- Possible crate lifts: `interpreter/` and/or `parser/` promoted to workspace crates if Phase 1 seams made this clean.

Each Phase 2 deliverable gets a dedicated `writing-plans` session when we reach it.

---

## Self-Review Notes

**Spec coverage:**
- Phase 1a consolidation: Tasks 1-6 cover it.
- Phase 1b/1c subsystems: sketched at section-level, explicit deferral to per-subsystem plans.
- Phase 2 hygiene: sketched with known targets, explicit deferral.
- Testing strategy (cargo test --lib green, bridge test at subsystem tag): enforced by Global Invariants + Task 6.
- Per-subsystem design notes under `docs/arch/<subsystem>.md`: Task 1 establishes the pattern with `phase1a-audit.md`; subsequent subsystem plans will write `docs/arch/<subsystem>.md`.
- Branching (dev only, no master merge): Global Invariants.
- Per-subsystem tags: Task 6 (phase1a), noted per subsystem.
- Trait-design safety mechanic ("what would AIE1 look like?"): noted at each subsystem section; will be enforced in each subsystem's own plan.

**Placeholder scan:** none found. "TBD" and similar phrases only appear in the placeholder section of Phase 2, which is explicitly deferred to a future plan -- acceptable per the scope note.

**Type consistency:** `ArchConfig`, `ModelConfig`, `ArchModel`, `TileKind`, `TileType` used consistently across tasks. `xdna_archspec::runtime` and `xdna_archspec::types` referenced consistently.
