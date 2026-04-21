# Subsystem 3 -- DMA Engine & BD Format -- Design

**Subsystem:** 3 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-21
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-18-subsys2-tile-topology-design.md](2026-04-18-subsys2-tile-topology-design.md)
**Planned tag:** `phase1-subsys-dma`

---

## Goal

Introduce a `DmaModel` trait seam in `xdna-archspec` and migrate the
pure-data BD / register-layout aggregation (`DeviceRegLayout` and its
`BdFieldLayout` family) from `src/device/regdb/` into archspec, so
xdna-emu's DMA code dispatches arch-divergent behavior (task queue,
out-of-order mode, compression, BD iteration, lock-ID-equality, etc.)
through a single trait and pulls arch-keyed layout data from a single
authority. Fold in two follow-ups that live in DMA code today: the
`(2, 2)` silent DMA-channel fallback in
`xdna_archspec::runtime::ModelConfig::from_arch_model` flagged in the
Phase 1a audit, and a short list of small hygiene items ("deodorize
as we go") surfaced by the Subsystem 3 audit in files we will already
be editing.

## Non-goals

- **No AIE1 or AIE2P `DmaModel` implementation.** Subsystem 3 ships the
  trait, the `Aie2DmaModel` impl, and the archspec migration of layout
  data. Populating AIE1 (interleave mode, double-buffer, 2D tensor,
  5-word shim BD, 7-word tile BD, lock-ID-equality enforcement, absent
  memtile) or AIE2P (different extents, likely extra tensor dim) is
  orthogonal future work, identical to how Subsystems 1, 2, and 6
  deferred second-arch population.
- **No interleave / double-buffer FSM phase.** AIE1's `Interleaving`
  phase is a new `ChannelFsm` variant that fires only when
  `DmaModel::supports_interleave_mode()` returns true. The variant
  lands when AIE1 lands. This subsystem codifies the feature flag but
  does not add the phase.
- **No `DmaFifo` activation.** `src/device/dma/fifo.rs:1-16` documents
  that `DmaFifo` is the AIE1-only DMA FIFO mode (disabled on AIE2); the
  code is present but dormant on AIE2 and keeps that state. Activating
  it is an AIE1 task.
- **No lock-value-width plumbing.** Subsystem 4 (Locks) explicitly owns
  that. AIE1 has 6-bit signed lock values; AIE2+ has 7-bit signed. The
  DMA-model flag `supports_independent_lock_ids` (AIE1 requires
  `Acq.LockId == Rel.LockId`; AIE2+ does not) lands here because it is
  a BD-apply-time check, not a lock-subsystem concern -- but the value
  width stays in Subsystem 4.
- **No `ChannelStatusMap` data structure.** Agent 4's audit suggested
  one for the per-channel status-register stride that differs between
  AIE1 and AIE2. For AIE2 the stride is already trivially derivable
  from the register DB, so the data structure buys nothing today. If
  AIE1 lands and the derivation proves insufficient, Subsystem 3's
  audit will capture the need; adding it is straightforward.
- **No runtime behavior change for AIE2.** The trait is a dispatch
  seam; current AIE2 call sites should produce byte-identical
  decisions before and after. The test suite catches any drift.
- **No serialization / FFI surface change.** `DmaModel` is not
  serialized. `DeviceRegLayout` is not exposed over FFI. The
  `BufferDescriptor` struct gains no new public fields (AIE1-only
  fields are added when AIE1 lands, not now).
- **No second-arch implementation during the refactor.** Phase 1
  ground rule.

---

## Context

Subsystem 2 (Tile Topology) landed at `phase1-subsys-tile-topo`
(`6534e28..fe2c08e`). Baselines at that tag: `cargo test --lib` = 2708
pass, 0 fail, 5 ignored; `cargo test -p xdna-archspec --lib` = 236
pass, 1 pre-existing fail (`device_model::test_full_parse_all_devices`),
2 ignored. `TileKind` is the canonical tile classifier; `TileType` is
gone; consumers import from archspec directly.

The DMA subsystem is the first **behavioral** seam in the refactor.
Subsystem 1 (Registers & Memory Map) landed mostly data; Subsystem 2
(Tile Topology) landed a trait whose methods boil down to data lookups
on AIE2. Subsystem 3 is where arch-divergence first becomes FSM
behavior -- task queue existence, interleave mode, BD iteration
support, lock-ID-equality -- not just different constants in the same
equations.

### What the audits found

Four Explore-agent audits ran against the 12,121-LOC DMA module tree
plus the two aie-rt reference implementations (`xaie_dma_aie.c`, 1095
LOC for AIE1; `xaie_dma_aieml.c`, 1526 LOC for AIE2/AIEML). The
subsystem-3 audit doc (`docs/arch/subsys3-audit.md`) is scaffolded in
Task 1 of the implementation plan and completed (with results,
migration totals, and surprises-encountered) in Task 8 at the tag;
it does not exist yet. Synthesized findings from the four agent
reports:

**Arch-divergent behaviors (need trait dispatch):**

| # | Behavior | AIE1 | AIE2+ | Evidence |
|---|---|---|---|---|
| 1 | Task queue (8-deep FIFO per channel) | absent | present | `xaie_dma_aieml.c:1257-1279`, `src/device/dma/engine/task_queue_ops.rs` (entire module AIE2-only) |
| 2 | `WaitForBdTaskQueue` | N/A | yes | `xaie_dma_aieml.h:55-57`; absent in `xaie_dma_aie.h` |
| 3 | Out-of-order BD ID (S2MM) | absent | present | `xaie_dma_aieml.c:313-315, 459-461`; `src/device/dma/engine/types.rs:136` |
| 4 | Compression (MM2S/S2MM) | absent | present | `xaie_dma_aieml.c:360-362, 514-516`; `src/device/dma/compression.rs:1-30` |
| 5 | BD iteration | returns `NOT_SUPPORTED` | full support | `xaie_dma_aie.c:1065-1074` stub; `xaie_dma_aieml.c:1502-1522` |
| 6 | Lock Acq/Rel ID equality | **required** | independent | `xaie_dma_aie.c:113-116` (hard error); `xaie_dma_aieml.c:143-154` (allowed) |
| 7 | Interleave + double-buffer | present (BD word 5 + LockDesc_2) | removed | `xaie_dma_aie.c:189-198, 477-500, 543-545`; absent in AIE2+ |
| 8 | Max tensor dims | 2 (X, Y) | tile=3, memtile=4 | `xaie_dma_aie.c:44`; `xaie_dma_aieml.c:62-95, 112-120` |
| 9 | Timing constants (bus width, latencies) | different | 128-bit bus, 4 w/cyc | `src/device/dma/timing.rs:56-66` (AIE2 hardcoded from `xdna_archspec::aie2::timing::*`) |

**Arch-divergent register-map layouts (need per-arch data in archspec, not trait behavior):**

| # | Area | AIE1 | AIE2+ |
|---|---|---|---|
| 10 | Tile BD register count | 7 words | 6 words |
| 11 | Shim BD register count | 5 words | 8 words |
| 12 | MemTile DMA | absent | 8-word BD |
| 13 | Channel status register shape | `StartQSize + Stalled` | `TaskQSize + multiple stall bits` |
| 14 | Channel status address stride | single | per-channel |

**Tile-type-dispatched, already-fine** (no trait needed; AIE1 plug-in works out-of-the-box once the BD layout is right):

- `src/device/dma/addressing.rs` (1006 LOC) -- 4D `AddressGenerator` with
  `d3_stepsize == 0` guard on compute/shim; AIE1's 2D usage collapses to
  `d2_stepsize == 0` without code change.
- `src/device/dma/transfer/padding.rs` (319 LOC) -- MemTile-MM2S-only,
  guarded by `tile_kind.is_mem()` at the single caller
  (`src/device/dma/transfer/core.rs:232-234`).
- `src/device/dma/fifo.rs` (1007 LOC) -- `DmaFifo` is the AIE1-only
  DMA FIFO mode; `FotCountFifo` is the AIE2 FoT mode. They coexist.
- `src/device/dma/stream_io.rs` (332 LOC) -- TileKind dispatch on port
  mapping.
- `src/device/dma/engine/locks.rs` (167 LOC) -- MemTile cross-tile
  lock window dispatch via `LockTarget` enum, clean.
- `src/device/dma/engine/stream_io.rs` (226 LOC) -- per-direction
  channel mapping, parameterized on `s2mm_count` / `mm2s_count`.

**Migration inventory (pure data to archspec):**

| Symbol | Disposition |
|---|---|
| `DeviceRegLayout` struct (`src/device/regdb/mod.rs:54-...`) | Move to archspec |
| `BdFieldLayout`, `MemTileBdFieldLayout`, `ShimBdFieldLayout` (`src/device/regdb/field_layouts.rs`) | Move to archspec |
| `ChannelFieldLayout`, `StatusFieldLayout`, `StreamSwitchLayout`, `ShimMuxLayout`, `ModuleEventLayout` | Move to archspec |
| `DeviceRegLayout::from_regdb()` constructor | Move to archspec |
| `OnceLock<DeviceRegLayout>` + `device_reg_layout()` accessor | **Stay in xdna-emu** (OnceLock singleton is the emulator-side runtime cache) |
| `DeviceRegLayout::load_for_device()` loader (the one that uses `crate::config::Config::get()` for JSON path resolution) | **Stay in xdna-emu** (config-system coupling) |
| `sign_extend_lock_value` helper (reads `lock_value_mask` + `lock_value_sign_bit` fields on `DeviceRegLayout`) | **Stay in xdna-emu** for Subsystem 3; the method and its backing fields migrate together into Subsystem 4's `LockModel` (see Subsystem 4 forward pointer below). Moving only the helper now would leave a lock-width concept half-migrated across archspec and xdna-emu. |
| `BufferDescriptor` struct + `parse_compute / parse_memtile / parse_shim` | **Stay in xdna-emu** (emulator-internal parsing that reads the archspec-owned layouts) |

**Hygiene items in files we will already be editing** (deodorize as
we go; skip anything in untouched files and punt to Phase 2):

1. `crates/xdna-archspec/src/runtime.rs:278` -- silent `(2, 2)` DMA
   channel fallback. Device-model JSON always has the DMA bundle for
   NPU1 / NPU2 (startup would panic earlier if it did not), so the
   fallback is unreachable for supported devices. Replace with
   `.expect()` mirroring the adjacent line 281.
2. `src/device/dma/compression.rs:39` -- silent `None` on size
   mismatch. Add `log::warn!` citing the expected 32-byte input.
3. `src/device/dma/addressing.rs:217-227` -- magic `63 / 31 / 15`
   padding bit-limits. Name them (`PAD_MAX_D0_WORDS = 63` etc.) and
   cite the AM025 register fields.
4. `src/device/dma/timing.rs:103` -- comment "AIE2: 128-bit bus = 4
   words/cycle" is not cited. Link to
   `xdna_archspec::aie2::timing::DMA_WORDS_PER_CYCLE`.
5. `src/device/dma/token.rs:~100` -- `MAX_TASK_QUEUE_DEPTH = 8`
   comment is self-contradictory about "3 bits / 0-7 / 8 entries".
   Rewrite to state the invariant once.
6. `src/device/dma/engine/{mod.rs,stream_io.rs}` -- stream buffer
   capacity `256` is a magic number. Name it
   `STREAM_BUFFER_CAPACITY_WORDS` and cite its source (empirical;
   verified from bridge traces, not AM025).
7. `src/device/dma/engine/mod.rs:309-327` -- MemTile BD-channel
   validity is currently warn-only. The AM025 invariant
   (`BdNum < 24 <=> ChNum even`) is a hard constraint; upgrade to
   an error. Add the covering test if one does not already exist.

Items deferred to Phase 2 hygiene (not in a file we will already be
touching, or scope-creep risk): `FotCountFifo` hardcoded capacity
citation, trace-event per-arch routing, PadPhase state-machine
diagram.

---

## Audit (actual state, not narrative)

| Concern | Count | Source |
|---|---|---|
| DMA module LOC total | ~12,121 | `wc -l src/device/dma/**/*.rs` |
| Largest files | `engine/stepping.rs` 1517, `bd.rs` 1178, `addressing.rs` 1006, `fifo.rs` 1007, `token.rs` 864, `engine/mod.rs` 672, `engine/tests.rs` 1675, `transfer/core.rs` 594 | same |
| Files to move to archspec (pure data) | 2 + ~3 submodules | `src/device/regdb/{mod.rs, field_layouts.rs, tests.rs}` |
| `DmaModel` feature-flag call sites | ~5 | task queue enqueue/pop, OOO check, compression invoke, status-bits emit, BD-iteration register write |
| `(2, 2)` silent fallback | 1 | `crates/xdna-archspec/src/runtime.rs:278` |
| Hygiene items in edit-zone files | 7 | see list above |
| `DmaModel` trait methods (proposed) | 9 | see Section 2 |
| AIE2-only tests that must stay green | all of `src/device/dma/engine/tests.rs` (1675 LOC) + `src/device/dma/transfer/tests.rs` (575 LOC) + `src/device/regdb/tests.rs` | `cargo test --lib` 2708 baseline |
| Named bridge tests exercising DMA features | ~30 (linear DMA, 2D/3D stride, memtile, shim, task queue, OOO) | `scripts/emu-bridge-test.sh` |

The `DmaEngine` struct (`src/device/dma/engine/mod.rs:60-`) already
carries per-tile-type channel counts (`s2mm_count`, `mm2s_count`),
`num_locks`, and `tile_kind` derived from the archspec `ModelConfig`
at construction. Adding a `DmaModel` reference (or equivalent
feature-flag carrier) at construction time is a natural extension of
that pattern.

---

## Approach

Single approach (Approach 1 from the brainstorm, approved 2026-04-21).
Six concerns folded into one subsystem:

### Section 1: Architecture

Add a `DmaModel` trait to `xdna-archspec` at
`crates/xdna-archspec/src/dma/mod.rs` (new module). Nine methods, all
cold-path: feature flags (booleans), per-tile-type tensor-dim ceilings,
and a `DmaTimingConfig` carrier. One concrete impl for AIE2:
`Aie2DmaModel` (zero-sized struct, const-fn methods where practical).

`ArchModel` exposes `dma_model()` returning `&dyn DmaModel`, matching
the precedent set by `topology()` in Subsystem 2.

`DeviceRegLayout` and its `BdFieldLayout` family move from
`src/device/regdb/` into `crates/xdna-archspec/src/regdb/layouts.rs`
(or equivalent). The xdna-emu side keeps only the `OnceLock` +
config-loading wrapper, now re-exporting the archspec types. Consumers
in `src/device/dma/bd.rs`, `src/device/dma/engine/mod.rs`, and
`src/device/state/*` change one import line each; the aggregation
logic itself moves unchanged.

The **production** `DmaEngine::new()` constructor
(`src/device/dma/engine/mod.rs:147`) gains a `&'static dyn DmaModel`
parameter; its single production caller in `DeviceArray::new()`
(`src/device/array/mod.rs:201`) threads it from `ArchModel::dma_model()`.
`DmaTimingConfig::from_arch()` at `engine/mod.rs:163` becomes
`DmaTimingConfig::from_model(&dyn DmaModel)`, reading the timing values
through the trait instead of the hardcoded
`xdna_archspec::aie2::timing::*` path. The three `#[cfg(test)]` helpers
(`new_compute_tile`, `new_mem_tile`, `new_shim_tile` at lines 188, 194,
200) update their bodies to construct a static `AIE2_DMA_MODEL` reference
and pass it through to `new()` -- test-code-only change.

The ~5 arch-divergent call sites gate on `model.supports_X()`:

1. `src/device/dma/engine/task_queue_ops.rs` -- `enqueue_task`,
   `start_next_queued_task`, `task_queue_size`: guarded by
   `supports_task_queue()`. AIE1 returns `false` here; the methods
   either become no-ops or return a not-supported error (plan
   decides).
2. `src/device/dma/engine/stepping.rs:343-351` -- the task-queue pop
   loop wraps in `if self.dma_model.supports_task_queue()`.
3. `src/device/dma/engine/status.rs:33-37` -- Task_Queue_Size and
   Task_Queue_Overflow status bits emitted only when
   `supports_task_queue()`.
4. `src/device/dma/engine/status.rs:104-109` --
   `is_out_of_order_enabled()` short-circuits to `false` when
   `!supports_ooo_mode()`.
5. `src/device/dma/compression.rs` entry point -- guarded by
   `supports_compression()` from the engine caller; the module itself
   stays untouched (AIE1 never enters it).

BD-apply paths that currently assume AIE2+ semantics (lock-ID
independence, BD iteration support) get early-return / error-path
enforcement gated on the corresponding `supports_*` flags. AIE2
behavior unchanged; AIE1 will hard-error at BD apply time if a
caller violates the invariant, matching `_XAie_DmaSetLock` at
`xaie_dma_aie.c:113-116`.

The FSM itself (`src/device/dma/engine/stepping.rs`, 1517 LOC) is
structurally untouched. AIE1's `Interleaving` / `DoubleBuffer` FSM
phases are new variants on `ChannelFsm` added when AIE1 lands, gated
by `supports_interleave_mode()` / `supports_double_buffer()`.

### Section 2: Components

**New: `crates/xdna-archspec/src/dma/mod.rs`:**

```rust
use crate::types::TileKind;

pub mod timing;

pub use timing::DmaTimingConfig;

pub trait DmaModel: Send + Sync {
    // Feature flags for arch-divergent behaviors.

    /// AIE2+ supports per-channel 8-deep task queues; AIE1 does not.
    fn supports_task_queue(&self) -> bool;

    /// AIE2+ S2MM supports out-of-order completion; AIE1 does not.
    fn supports_ooo_mode(&self) -> bool;

    /// AIE2+ MM2S/S2MM support sparsity compression; AIE1 does not.
    fn supports_compression(&self) -> bool;

    /// AIE2+ supports BD iteration (stepsize + wrap + current counter);
    /// AIE1 returns XAIE_FEATURE_NOT_SUPPORTED.
    fn supports_bd_iteration(&self) -> bool;

    /// AIE2+ allows independent lock IDs for acquire and release;
    /// AIE1 requires acq.lock_id == rel.lock_id.
    fn supports_independent_lock_ids(&self) -> bool;

    /// AIE1 exposes interleave+double-buffer via BD word 5 and
    /// LockDesc_2; AIE2+ removed the mechanism (addressable via
    /// multi-dim iteration instead).
    fn supports_interleave_mode(&self) -> bool;

    /// AIE1-only; paired with supports_interleave_mode.
    fn supports_double_buffer(&self) -> bool;

    // Arch-varying numeric parameters.

    /// Maximum tensor dimensions by tile kind. AIE1: 2 for all.
    /// AIE2 compute/shim: 3. AIE2 mem: 4. AIE2P: possibly higher.
    fn max_tensor_dims(&self, tile: TileKind) -> u8;

    /// The full eight-counter timing config (memory latency, lock
    /// acquire/release cycles, bus words/cycle, etc.).  AIE2 reads
    /// from xdna_archspec::aie2::timing; AIE1 will return AIE1-tuned
    /// values.
    fn timing_config(&self) -> DmaTimingConfig;
}
```

**New: `crates/xdna-archspec/src/aie2/dma.rs`** with a zero-sized
`Aie2DmaModel` struct and a `const fn` impl:

```rust
pub struct Aie2DmaModel;

impl DmaModel for Aie2DmaModel {
    fn supports_task_queue(&self) -> bool { true }
    fn supports_ooo_mode(&self) -> bool { true }
    fn supports_compression(&self) -> bool { true }
    fn supports_bd_iteration(&self) -> bool { true }
    fn supports_independent_lock_ids(&self) -> bool { true }
    fn supports_interleave_mode(&self) -> bool { false }
    fn supports_double_buffer(&self) -> bool { false }

    fn max_tensor_dims(&self, tile: TileKind) -> u8 {
        match tile {
            TileKind::Compute | TileKind::ShimNoc | TileKind::ShimPl => 3,
            TileKind::Mem => 4,
        }
    }

    fn timing_config(&self) -> DmaTimingConfig {
        DmaTimingConfig::aie2_default()
    }
}
```

**Moved: `DeviceRegLayout` family.** Migrated from
`src/device/regdb/` to `crates/xdna-archspec/src/regdb/layouts.rs` (or
sibling file). Public symbols: `DeviceRegLayout`, `BdFieldLayout`,
`MemTileBdFieldLayout`, `ShimBdFieldLayout`, `ChannelFieldLayout`,
`StatusFieldLayout`, `StreamSwitchLayout`, `ShimMuxLayout`,
`ModuleEventLayout`, `DeviceRegLayout::from_regdb()`,
plus the aggregation logic itself. Not moved: `sign_extend_lock_value`
method and its backing `lock_value_mask` / `lock_value_sign_bit` fields
-- these describe the lock-value format, which is Subsystem 4's seam.
Keeping the method in xdna-emu (as a method on the locally-aliased
`DeviceRegLayout` re-export) for Subsystem 3 avoids a half-migrated
lock-width concept straddling the crate boundary. Subsystem 4 migrates
the method and its backing fields together as part of the `LockModel`
work. No API shape change for other migrating symbols.

**Retained: xdna-emu wrapper.** `src/device/regdb/mod.rs` shrinks to
the `OnceLock`, `load_for_device()`, and `device_reg_layout()`
accessor -- all functions whose coupling to `crate::config::Config` or
to the emulator's startup singleton model means they cannot move.
Re-exports `pub use xdna_archspec::regdb::*;` so consumer imports
either stay the same or change one line.

**New: `ArchModel::dma_model(&self) -> &dyn DmaModel`.** Matches the
`topology()` accessor shape from Subsystem 2. Implementation block
location likely mirrors `topology.rs`'s location (an `impl ArchModel`
block in `dma/mod.rs` because `types.rs` is `#[path]`-included by
`build.rs`).

**Modified: `DmaEngine` construction.** The production constructor
`DmaEngine::new(col, row, tile_kind, s2mm_channels, mm2s_channels,
num_bds, num_locks)` at `engine/mod.rs:147` gains a
`dma_model: &'static dyn DmaModel` parameter. The engine stores the
reference for subsequent feature-flag checks during stepping. The
`&'static` shape is the recommended default: `Aie2DmaModel` is a
zero-sized struct, and `ArchModel::dma_model()` can return a reference
to a `static AIE2_DMA_MODEL: Aie2DmaModel = Aie2DmaModel;` singleton.
This avoids `Arc` overhead and a lifetime parameter on `DmaEngine`.
The single production caller at `src/device/array/mod.rs:201` threads
the model through from `ArchModel::dma_model()`. The three
`#[cfg(test)]` helpers (`new_compute_tile`, `new_mem_tile`,
`new_shim_tile` at `engine/mod.rs:188, 194, 200`) call the static
`AIE2_DMA_MODEL` directly.

**Modified: `DmaTimingConfig`.** Gains `from_model(&dyn DmaModel)`
constructor; `from_arch()` either becomes a thin alias or deletes.
The hardcoded pull from `xdna_archspec::aie2::timing::*` moves
inside `Aie2DmaModel::timing_config()`.

**Modified: `(2, 2)` silent fallback.** Closure at
`crates/xdna-archspec/src/runtime.rs:270-279` rewrites:

```rust
let dma_channels = |tile: &crate::types::TileTypeModel| -> (usize, usize) {
    let ports = match tile.kind {
        TileKind::ShimNoc | TileKind::ShimPl => &tile.shim_mux_ports,
        _ => &tile.switchbox_ports,
    };
    let dma = ports.iter()
        .find(|p| p.bundle == "DMA")
        .expect("DMA bundle missing from tile type; regenerate device model with aie-device-dump.py");
    (dma.slaves as usize, dma.masters as usize)
};
```

### Section 3: Data flow

**Archspec to xdna-emu (cold path, construction):**

1. `ModelConfig::from_arch_model()` builds the `ArchModel` once at
   startup (existing behavior).
2. `ArchModel::dma_model()` returns the appropriate `&dyn DmaModel`
   (`&Aie2DmaModel` for NPU1 / NPU2 / NPU4; AIE1 impl added later).
3. `DeviceRegLayout` loads from the archspec-side `from_regdb()`
   builder into the xdna-emu OnceLock (first call to
   `device_reg_layout()`).
4. `DeviceArray::new()` threads `ArchModel::dma_model()` into each
   per-tile `DmaEngine::new()` call at `src/device/array/mod.rs:201`;
   the engine stores the reference alongside existing per-tile params.
5. `DmaTimingConfig::from_model()` reads the eight cycle constants
   through the trait.

**Runtime (hot path, per DMA cycle):**

1. `DmaEngine::step()` iterates channels (unchanged structurally).
2. State-machine transitions consult the stored `DmaModel` reference
   only at the ~5 feature-gated sites identified above. Devirtualizing
   trait dispatch is standard for LLVM when the concrete type is
   knowable at the call site; for the AIE2-only monomorphic build,
   this is nearly free.
3. Per-BD parsing in `bd.rs` reads layouts from `device_reg_layout()`
   (unchanged); the layouts themselves now come from archspec. No
   per-cycle trait calls during parsing.

**Hot-path cost:** one additional `&dyn DmaModel` load per FSM phase
transition that consults a feature flag. Bench-relevant only if AIE2
dispatch proves to be non-devirtualizable; if so, the plan can switch
to storing the flags as a single `u16` bitmask in `DmaEngine` at
construction, recovering zero-cost indirection. Measure before
optimizing.

### Section 4: Scope boundaries

**In scope:**

- `DmaModel` trait + `Aie2DmaModel` impl in archspec.
- `DeviceRegLayout` family migration into archspec.
- Thread `DmaModel` through `DmaEngine` construction and the ~5
  feature-gated call sites.
- Route `DmaTimingConfig` through the trait.
- `(2, 2)` silent-fallback fix.
- Seven hygiene items (compression log, pad constants, timing
  citation, task-queue comment, stream-buffer naming, MemTile BD/ch
  validity error-upgrade, plus the fallback fix).
- `docs/arch/dma-model.md` (design note per parent-spec mandate).
- `docs/arch/subsys3-audit.md` (audit record with completion section).
- Trait unit tests in archspec for `Aie2DmaModel`.
- Baseline verification at the tag (bridge + ISA).

**Out of scope (deferred by sub-subsystem):**

- AIE1 / AIE2P `DmaModel` impls (Phase 2 or second-arch branch).
- `Interleaving` / `DoubleBuffer` `ChannelFsm` variants (AIE1 land).
- `DmaFifo` activation (AIE1 land).
- Lock-value width plumbing (Subsystem 4).
- `ChannelStatusMap` data structure (deferred; no AIE2 consumer).
- Trace-event per-arch routing (Phase 2 hygiene).
- `FotCountFifo` capacity AM025 citation (Phase 2; file not edited
  otherwise).
- `PadPhase` state-machine ASCII diagram (Phase 2; doc-only cosmetic).

### Section 5: Testing and verification

**Global invariants (every commit):**

- `cargo test --lib` green (baseline: 2708 pass, 0 fail, 5 ignored).
- `cargo test -p xdna-archspec --lib` green (baseline: 236 pass, 1
  pre-existing fail, 2 ignored).
- `cargo build` and `cargo build --release` green.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green
  after rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`).

**Smoke-test matrix after each task:**

The DMA-heavy named tests (expected to all pass before and after):

- `test_bd_config_simple`, `test_bd_config_2d`, `test_bd_config_with_locks`
  (in `src/device/dma/mod.rs`).
- All tests under `src/device/dma/engine/tests.rs` (1675 LOC,
  hundreds of individual tests covering task queue, OOO, compression,
  BD chaining, MemTile channels, shim channels, cross-tile locks).
- All tests under `src/device/dma/transfer/tests.rs` (575 LOC).
- All tests under `src/device/regdb/tests.rs` (BD field parsing from
  AM025 JSON for all three tile types).
- `test_memtile_bd_channel_validity_*` (5 variants) -- the MemTile
  BD-channel invariant; ensure upgrading from warn to error does not
  break them.

**New archspec tests (targeted ~12-15):**

- `Aie2DmaModel::supports_task_queue` / `supports_ooo_mode` /
  `supports_compression` / `supports_bd_iteration` /
  `supports_independent_lock_ids` / `supports_interleave_mode` /
  `supports_double_buffer` return the expected booleans.
- `Aie2DmaModel::max_tensor_dims(TileKind::Compute | ShimNoc | ShimPl)`
  returns 3; `TileKind::Mem` returns 4.
- `Aie2DmaModel::timing_config()` returns the eight expected cycle
  values matching `xdna_archspec::aie2::timing::*`.
- `ArchModel::dma_model()` dispatches correctly on `Architecture`.
- `DeviceRegLayout::from_regdb()` for AIE2 still produces the
  expected BD bases, strides, and word counts (migrated-over from
  `src/device/regdb/tests.rs`).

**Per-subsystem gate (at the tag):**

1. Rebuild the FFI cdylib: `cargo build -p xdna-emu-ffi`.
2. `./scripts/emu-bridge-test.sh 2>&1 | tee
   /tmp/claude-1000/bridge-subsys3.log` -- full HW bridge run, ~30
   min. Compare pass/fail matrix to the `phase1-subsys-tile-topo`
   baseline; no new regressions. `bd_chain_repeat_on_memtile` remains
   a known pre-existing failure.
3. `./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys3.log`
   -- ISA test suite, ~10 min. Expect `FAIL: 0`.
4. Bridge and ISA run sequentially (never concurrently, per CLAUDE.md
   operational note).

Expected verification cost: ~45 min at the tag.

### Section 6: Scope gating and part split

Subsystem 3 is larger than Subsystem 2 (DMA is the densest subsystem
in the emulator) but smaller than Subsystem 6 (which relocated
~9,300 LOC across a build-side / runtime boundary). The plan writes
this as a single part with an explicit Part A / Part B split option
if the migration proves sprawlier than the audit estimates:

- **Part A (default plan):** trait scaffold + `Aie2DmaModel` +
  `DeviceRegLayout` migration + `DmaEngine` plumbing +
  feature-gated-site guards + `(2, 2)` fix.
- **Part B (if needed):** hygiene items + audit-doc completion + tag.

The split decision is data-driven: if the migration exceeds ~20
commits or touches ~40+ files, split; otherwise land as one part.
Subsystem 2 landed in 12 commits touching ~23 files with ~200 LOC new
and ~200 deleted. Subsystem 3 is estimated larger in LOC (more moved,
less deleted -- the archspec migration is ~450 LOC of relocation
alone) and likely larger in file-count too: the regdb migration alone
touches `regdb/mod.rs`, `regdb/field_layouts.rs`, `regdb/tests.rs`,
plus consumer imports in `src/device/dma/bd.rs`,
`src/device/dma/engine/mod.rs`, and `src/device/state/{compute,memtile,
dispatch,effects,mod}.rs` -- ~10 files just for the data migration,
before the trait plumbing and hygiene passes. Plan Part A against that
estimate; if realized file count exceeds ~30 or commit count exceeds
~15, split.

**Task sequence (the plan will flesh these out):**

1. Scaffold `docs/arch/dma-model.md` + `docs/arch/subsys3-audit.md`,
   template-matched to `tile-topology.md` and `subsys2-audit.md`.
   Completion sections filled later.
2. Add `DmaModel` trait + `DmaTimingConfig` + `Aie2DmaModel` impl to
   archspec. Wire `ArchModel::dma_model()`. Ship unit tests.
3. Migrate `DeviceRegLayout` family from xdna-emu to archspec.
   Preserve the xdna-emu-side `OnceLock` + `load_for_device()` +
   `device_reg_layout()` accessor. Update all consumer imports.
4. Fix the `(2, 2)` silent fallback in `runtime.rs:278` with
   `.expect()`. Verify AIE2 still loads (it does).
5. Thread `&'static dyn DmaModel` through the production
   `DmaEngine::new()` constructor (`engine/mod.rs:147`) and its single
   caller at `src/device/array/mod.rs:201`; store it on the engine.
   Introduce `DmaTimingConfig::from_model()` at `engine/mod.rs:163`.
   Update the three `#[cfg(test)]` helpers (`new_compute_tile`,
   `new_mem_tile`, `new_shim_tile`) to call through with a static
   `AIE2_DMA_MODEL` singleton -- test-code-only change.
6. Gate the ~5 feature-dispatch call sites
   (`task_queue_ops.rs`, `stepping.rs:343-351`, `status.rs:33-37`,
   `status.rs:104-109`, `compression.rs` entry).
7. Apply the seven deodorize items (compression log, pad constants,
   timing citation, task-queue comment, stream-buffer naming, MemTile
   BD/ch validity error-upgrade, fallback fix from step 4 already
   done).
8. Fill completion sections in `docs/arch/dma-model.md` +
   `docs/arch/subsys3-audit.md`. Run full HW + ISA gate. Tag
   `phase1-subsys-dma`. Update `NEXT-STEPS.md` to point at
   Subsystem 4 (Locks) as up next.

---

## Why a trait this time

Subsystem 6 documented "no trait" because ISA decode is values
(TableGen data differs; the walker algorithm is invariant).
Subsystem 2 introduced a trait because tile topology has genuine
shape differences (AIE1's alternating-row memory adjacency cannot be
expressed by the same constants as AIE2). Subsystem 3 introduces a
trait because **DMA engine behavior has genuine feature differences**
across arch families, not just value differences:

- **AIE2 (NPU1/NPU2/NPU4/NPU5/NPU6):** 8-deep task queue per channel,
  out-of-order S2MM completion, sparsity compression, BD iteration
  with per-BD stepsize + wrap counter, independent acquire / release
  lock IDs, memtile DMA with 4D addressing and zero-padding, tile
  DMA with 3D addressing, 128-bit memory bus (4 words/cycle).
- **AIE1 (Versal, e.g., xcvc1902):** no task queue (direct BD start
  only), no OOO, no compression, no BD iteration, acquire and
  release must use the same lock ID, no memtile, tile DMA is 2D
  (X/Y) with a distinct interleave + double-buffer FSM mode that
  consumes BD word 5 and LockDesc_2, shim DMA is 5 words vs. AIE2's
  8. Memory bus narrower (likely 2 words/cycle; measurement pending
  AIE1 impl).
- **AIE2P (NPU4/5/6):** similar to AIE2 in feature set but possibly
  wider tensor support and different channel status address stride.
  Expected to share most of `Aie2DmaModel`'s impl with tweaked
  constants -- the plan will evaluate sub-variant sharing when AIE2P
  support lands.

A data-only approach cannot express features like task queue
existence or interleave-mode presence without `if arch_family ==
AieX` branches through FSM code. The trait contains the branching
inside one `DmaModel` impl per arch family; the FSM code calls
`model.supports_X()` and remains single-shape.

**Approach 2 (split traits) considered and rejected.** Splitting into
a `BdLayout` trait (register-map data) and a `DmaBehavior` trait
(feature flags + timing) would cleanly separate concerns on paper,
but every real device has these two axes linked -- no practical
device would have AIE2's BD layout with AIE1's behavior or vice
versa. The split is pure ceremony for the devices we actually
target, and adding a second seam for every consumer to thread buys
no current flexibility. If future Versal variants introduce a
BD-layout / behavior cross-product that the single trait cannot
express, promoting to a two-trait shape is a mechanical refactor
with no consumer impact.

**Approach 3 (behavioral hooks + per-tick dispatch) considered and
rejected.** Adding `fn pre_step(&self, ctx: &ChannelContext) ->
Option<FsmPhase>` or similar hot-path dispatch points would
pre-emptively serve AIE1's interleave FSM phase. But that phase is
AIE1-only and fires only in one code path that does not exist on
AIE2; adding the hook now requires AIE2 to pay the hot-path dispatch
cost (even if optimized away in monomorphic builds) and adds trait
surface that no AIE2 consumer calls. The simpler shape is to let
AIE1 add `ChannelFsm::Interleaving` as a new variant when it lands --
AIE2's FSM match never sees it because `supports_interleave_mode()`
returns false.

The "what would AIE1 look like?" exercise from the parent spec
resolves cleanly: `Aie1DmaModel` returns `false` for task_queue /
ooo / compression / bd_iteration / independent_lock_ids and `true`
for interleave / double_buffer, with `max_tensor_dims` = 2 for all
tile kinds and an AIE1-tuned `DmaTimingConfig` (narrower bus,
different memory latency). The lock-ID-equality check is enforced at
BD apply time by reading the flag. When AIE1 lands, xdna-emu adds
`ChannelFsm::Interleaving` and `ChannelFsm::DoubleBuffer` variants
that `supports_interleave_mode()` / `supports_double_buffer()` gate
entry into. AIE2's FSM never sees those variants.

---

## Forward pointers

- **Subsystem 4 (Locks).** Lock value width (AIE1: 6-bit signed;
  AIE2+: 7-bit signed) and the `LockModel` trait live there.
  `DmaModel::supports_independent_lock_ids()` stays in Subsystem 3
  because it is a DMA-side BD-apply check, not a lock-subsystem
  concern. The `sign_extend_lock_value()` helper and its backing
  `lock_value_mask` / `lock_value_sign_bit` fields (currently on
  `DeviceRegLayout`) explicitly stay in xdna-emu for Subsystem 3 and
  migrate to `LockModel` as part of Subsystem 4 -- this avoids a
  half-migrated lock-width concept straddling the crate boundary
  during the refactor.
- **Subsystem 5 (Stream Switch).** If routing legality checks need
  arch-specific DMA-port awareness beyond what the existing
  `TileTopology::classify(col, 0)` returns, revisit whether
  `DmaModel` grows methods for DMA-stream-port enumeration. Not
  anticipated for AIE2.
- **Subsystem 7 (ISA Execute).** DMA intrinsics (used by control
  packets and host-side task launches) consult the `DmaModel` at
  execute time -- expect to touch the ~5 feature-gated sites from
  there too. If the ISA-side dispatch introduces a new gated site,
  add it to `DmaModel` then, not now.
- **AIE1 follow-on.** Adding AIE1 `DmaModel` is bounded: the trait
  is sized to accept it; the `ChannelFsm` variant additions are the
  only non-data work. Expected bounded to ~200-300 LOC of
  `Aie1DmaModel` + ~150 LOC of `Interleaving` / `DoubleBuffer`
  `ChannelFsm` arms + AIE1 `DeviceRegLayout` loading. No refactoring
  of AIE2 code required.
- **Phase 2 hygiene.** `FotCountFifo` capacity AM025 citation
  (`src/device/dma/fifo.rs:297`), `PadPhase` state-machine diagram,
  trace-event per-arch routing, any additional hygiene uncovered
  during Subsystem 3 in files we did not edit.

---

## Deliverables checklist

- [ ] `DmaModel` trait + `DmaTimingConfig` in `xdna-archspec`.
- [ ] `Aie2DmaModel` concrete impl.
- [ ] `ArchModel::dma_model(&self) -> &dyn DmaModel` accessor.
- [ ] `DeviceRegLayout` family migrated from `src/device/regdb/`
      into archspec; xdna-emu-side wrapper (OnceLock +
      `load_for_device()` + `device_reg_layout()`) retained.
- [ ] Production `DmaEngine::new()` (`engine/mod.rs:147`) accepts
      `&'static dyn DmaModel`; threaded through from
      `DeviceArray::new()` at `src/device/array/mod.rs:201`. Test-only
      helpers (`new_compute_tile`, `new_mem_tile`, `new_shim_tile`)
      updated to use a static `AIE2_DMA_MODEL` singleton.
- [ ] `DmaTimingConfig::from_model()` reading from the trait.
- [ ] ~5 feature-gated call sites converted to `model.supports_X()`:
      `task_queue_ops.rs`, `stepping.rs:343-351`, `status.rs:33-37`,
      `status.rs:104-109`, `compression.rs` entry.
- [ ] `(2, 2)` silent fallback in `runtime.rs:278` replaced with
      `.expect()`.
- [ ] Hygiene items 2-7 applied (compression log, pad constants,
      timing citation, task-queue comment rewrite, stream-buffer
      naming, MemTile BD/ch validity error-upgrade with test).
- [ ] ~12-15 new `Aie2DmaModel` unit tests in archspec.
- [ ] `DeviceRegLayout` migrated tests land in archspec test module.
- [ ] `docs/arch/dma-model.md` written (mandatory per-seam design
      note; template from `tile-topology.md`).
- [ ] `docs/arch/subsys3-audit.md` written (template from
      `subsys2-audit.md`, including a Completion section).
- [ ] Full HW bridge + ISA suite green at the tag.
- [ ] `NEXT-STEPS.md` updated to point at Subsystem 4 (Locks) as
      up next.
- [ ] Tag: `phase1-subsys-dma`.

---

## Success criteria (must all hold at the final tag)

1. `cargo test --lib` passes; count is approximately 2708 +/- a
   small delta (the migrated regdb tests land on the archspec side;
   xdna-emu net may decrease; plan nails the exact number).
2. `cargo test -p xdna-archspec --lib` passes with ~12-15 new
   `DmaModel` tests plus the migrated `DeviceRegLayout` tests beyond
   the 236-pass baseline, with the same 1 pre-existing failure
   (`device_model::test_full_parse_all_devices`) and 2 ignored.
3. `cargo build --release` clean.
4. Full HW bridge run shows no new regressions vs.
   `phase1-subsys-tile-topo` baseline. `bd_chain_repeat_on_memtile`
   remains a known pre-existing EMU failure.
5. ISA test suite: `FAIL: 0`.
6. Grep across `src/` returns zero usage of the deleted
   `DmaTimingConfig::from_arch()` entry point (or equivalent) --
   all call sites route through `from_model(&dyn DmaModel)`.
7. Grep across `crates/xdna-archspec/src/` returns zero occurrences
   of `unwrap_or((2, 2))` or equivalent silent fallbacks in the DMA
   path.
8. `src/device/regdb/mod.rs` contains only the OnceLock,
   `load_for_device()`, `device_reg_layout()`, and re-exports --
   no layout struct definitions (moved to archspec).
9. The seven hygiene items applied; their locations contain no
   magic numbers, silent None, or self-contradictory comments.
10. `docs/arch/dma-model.md` includes the mandatory "What would
    AIE1 look like?" paragraph.
11. No `TODO` / `FIXME` / `unimplemented!()` introduced without an
    open-issue reference.
12. All commits land on `dev`; no master merges during the
    subsystem.

---

## Ground rules (inherited from the parent refactor spec)

- **No master merges during the refactor.** Everything lands on `dev`.
- **`cargo test --lib` green at every commit.** Non-negotiable.
- **Bridge test smoke green at every subsystem tag.** Full HW run
  before the tag.
- **One authoritative source per concept.** Once `DeviceRegLayout`
  lives in archspec, no parallel struct reappears in `src/`.
- **Traits decode / step / check; they do not hold mutable state.**
  `DmaModel` reads from `ArchModel`; it does not mutate anything.
  `Aie2DmaModel` is zero-sized.
- **Coarse first.** Nine methods on `DmaModel`, not twenty. The ~5
  feature-gated call sites cover every known arch-divergent
  behavior. Hot-path hooks are explicitly deferred.
- **"What would AIE1 look like?" design note per seam.** Written
  into `docs/arch/dma-model.md` as part of Subsystem 3's deliverable.
- **No second-arch implementation during the refactor.** Subsystem 3
  ships the AIE2 impl only. AIE1 / AIE2P impls are follow-on.
