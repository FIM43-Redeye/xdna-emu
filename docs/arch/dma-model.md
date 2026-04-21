# DMA Model -- Design Note

**Subsystem:** 3 (Phase 1b)
**Tag:** `phase1-subsys-dma`
**Spec:** [../superpowers/specs/2026-04-21-subsys3-dma-engine-design.md](../superpowers/specs/2026-04-21-subsys3-dma-engine-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains the shape difference that
justifies the `DmaModel` trait and what AIE1 / AIE2P impls will look
like.

---

## What lives where

All entries below reflect state as of the `phase1-subsys-dma` tag.

| Data/code | Module | Source |
|-----------|--------|--------|
| `DmaModel` trait (9 methods) + `DmaTimingConfig` | `xdna_archspec::dma` | Emulator design |
| `DeviceRegLayout` + `BdFieldLayout` family | `xdna_archspec::dma::layouts` | AM025 register DB JSON |
| `Aie2DmaModel` concrete impl + `AIE2_DMA_MODEL` static | `xdna_archspec::aie2::dma` | aie-rt xaiemlgbl_params.h + timing constants |
| `ArchModel::dma_model()` accessor | `xdna_archspec::types` (`impl ArchModel` block) | Dispatches on `Architecture` |
| `DmaEngine::new()` threading | `xdna_emu::device::dma::engine` | Carries `&'static dyn DmaModel` |

The `OnceLock<DeviceRegLayout>` + `load_for_device()` + `device_reg_layout()` accessor
stay in xdna-emu (configured-system coupling). The `sign_extend_lock_value()` helper
and its backing `lock_value_mask` / `lock_value_sign_bit` fields also stay in xdna-emu
for this subsystem; they migrate to `LockModel` as part of Subsystem 4 (this avoids
a half-migrated lock-width concept straddling the crate boundary).

---

## The shape-vs-values principle, applied to DMA

Subsystem 2 introduced a trait for tile topology because AIE1's alternating-row
memory adjacency is a *shape* difference. Subsystem 3 introduces a trait because
DMA has multiple *feature* differences: task queue presence/absence, interleave
mode presence/absence, BD iteration support, lock-ID-equality enforcement, and
compression support. None of these can be expressed as "different values for the
same mechanism"; each is a yes/no choice that the FSM and register-map code must
branch on.

Concretely:

- **AIE2 (NPU1/NPU2/NPU4/NPU5/NPU6):** 8-deep task queue per channel, out-of-order
  S2MM completion, sparsity compression, BD iteration, independent acquire / release
  lock IDs, memtile DMA with 4D addressing and zero-padding, tile DMA with 3D
  addressing, 128-bit memory bus (4 words/cycle), 6-word tile BD, 8-word shim BD.
- **AIE1 (Versal, e.g., xcvc1902):** no task queue (direct BD start), no OOO,
  no compression, no BD iteration, acquire and release must use the same lock ID,
  no memtile, tile DMA is 2D (X/Y) with a distinct interleave + double-buffer FSM
  mode that consumes BD word 5 and LockDesc_2, shim DMA is 5 words vs. AIE2's 8.
- **AIE2P (NPU4/5/6):** similar to AIE2 in feature set but different extents and
  potentially different channel status address stride. Expected to share most of
  `Aie2DmaModel`'s impl with tweaked constants.

---

## The trait surface

```rust
pub trait DmaModel: Send + Sync {
    // Feature flags (cold path -- consulted at construction and call-site boundaries)
    fn supports_task_queue(&self) -> bool;
    fn supports_ooo_mode(&self) -> bool;
    fn supports_compression(&self) -> bool;
    fn supports_bd_iteration(&self) -> bool;
    fn supports_independent_lock_ids(&self) -> bool;
    fn supports_interleave_mode(&self) -> bool;
    fn supports_double_buffer(&self) -> bool;

    // Arch-varying numeric parameter (data, per tile)
    fn max_tensor_dims(&self, tile: TileKind) -> u8;

    // Timing carrier (data, read once at DmaEngine construction)
    fn timing_config(&self) -> DmaTimingConfig;
}
```

Nine methods, "coarse first":

- Seven feature flags cover every known AIE1 vs AIE2 behavioral divergence.
- `max_tensor_dims` is data that varies per-tile-kind AND per-arch (AIE1 = 2 for
  all, AIE2 compute/shim = 3, AIE2 mem = 4, AIE2P potentially higher).
- `timing_config` returns the eight per-arch cycle constants (bus width, memory
  latency, lock acquire/release, etc.) as a single carrier struct.

Not on the trait:

- BD-parse / BD-encode functions: stay in xdna-emu as emulator-internal parsing
  over the archspec-owned `DeviceRegLayout`. The packing sequence is already
  per-tile-type in xdna-emu; per-arch BD layouts plug in by adding a new
  `DeviceRegLayout::load_for_device("aie1")` path and adding `parse_aie1_*`
  methods to the same xdna-emu `bd.rs`.
- FSM stepping hooks: the FSM is structurally invariant. AIE1's interleave and
  double-buffer phases are new `ChannelFsm` variants that enter only when
  `supports_interleave_mode()` / `supports_double_buffer()` return `true`.
- Per-BD hot-path dispatch: adding trait calls inside the per-cycle FSM tick was
  rejected (Approach 3 in the spec). Hot paths stay tile-type-dispatched.

---

## What would AIE1 look like?

- `xdna_archspec::aie1::dma::Aie1DmaModel` zero-sized struct + `AIE1_DMA_MODEL`
  static.
- `supports_task_queue`, `supports_ooo_mode`, `supports_compression`,
  `supports_bd_iteration`, `supports_independent_lock_ids` all return `false`.
- `supports_interleave_mode`, `supports_double_buffer` return `true`.
- `max_tensor_dims(_) -> 2` for all tile kinds (AIE1 tiles are 2D-only; no
  memtiles exist).
- `timing_config` returns AIE1-tuned values: narrower memory bus (likely
  2 words/cycle), different lock latencies, different BD setup cycles.
- `ArchModel::dma_model()` adds an arm for `Architecture::Aie` returning
  `&AIE1_DMA_MODEL`.
- AIE1 also requires adding `Aie1RegLayout` to `xdna_archspec::dma::layouts`
  loaded from a new `aie_registers_aie1.json` (5-word shim BD, 7-word tile BD,
  interleave-related fields), plus the `Interleaving` / `DoubleBuffer`
  variants to `ChannelFsm` in xdna-emu.

Call sites in `src/device/dma/engine/` require no changes: task queue,
OOO, compression, BD iteration, and lock-ID-equality paths all short-circuit
on their respective `supports_*()` false returns.

---

## Why not a split trait (rejected Approach 2)

A `BdLayout` trait (data: register-map layouts) + `DmaBehavior` trait (feature
flags + timing) could separate concerns cleanly. Rejected because every real
device has these two axes linked -- no device would have AIE2's BD layout with
AIE1's feature set. The split is pure ceremony for the devices we target, and
adding a second seam for every consumer to thread buys no current flexibility.
If future Versal variants introduce a BD-layout / behavior cross-product the
single trait cannot express, promoting to a two-trait shape is a mechanical
refactor with no consumer impact.

## Why not per-tick behavioral hooks (rejected Approach 3)

Adding `fn pre_step(&self, ctx: &ChannelContext) -> Option<FsmPhase>` (or
similar per-channel-cycle dispatch points) would pre-emptively serve AIE1's
interleave FSM phase. Rejected because that phase is AIE1-only and fires only
in code paths that do not exist on AIE2; adding the hook now requires AIE2 to
pay hot-path dispatch cost and adds trait surface no AIE2 consumer calls. The
simpler shape is to let AIE1 add `ChannelFsm::Interleaving` as a new variant
when it lands -- AIE2's FSM match never sees it because
`supports_interleave_mode()` returns `false`.

---

## Completion (2026-04-21)

Landed at `phase1-subsys-dma`. Net effect:

- `xdna_archspec::dma::DmaModel` trait (9 methods: 7 feature flags +
  `max_tensor_dims` + `timing_config`) + `DmaTimingConfig` struct live
  at the crate root.
- `xdna_archspec::aie2::dma::Aie2DmaModel` concrete impl +
  `AIE2_DMA_MODEL` static singleton for AIE2-family devices
  (NPU1/NPU4/NPU5/NPU6).
- `xdna_archspec::runtime::ArchConfig::dma_model()` accessor returns
  `&'static dyn DmaModel`, dispatching on `ModelConfig::architecture`
  (added to the trait rather than a standalone `ArchModel` method
  because production call sites hold `Arc<dyn ArchConfig>`).
- `xdna_archspec::dma::DeviceRegLayout` + `BdFieldLayout` family
  migrated from xdna-emu (with xdna-emu retaining a thin `Deref`
  wrapper that adds the lock-value-width fields pending Subsystem 4).
- `DmaEngine::new()` at `engine/mod.rs:147` threads `&'static dyn
  DmaModel`; production construction routes through
  `arch.dma_model()` at `src/device/array/mod.rs:205`.
- Five AIE2-only call sites (task-queue enqueue + size/overflow/clear
  accessors, task-queue pop on task complete, task-queue status bits,
  OOO status, compression config setter) gate on `self.dma_model
  .supports_X()`. AIE1's eventual `Aie1DmaModel` returning `false`
  makes those paths inert without touching FSM or status code.
- `(2, 2)` silent DMA-channel fallback in archspec's `runtime.rs`
  replaced with `.expect()` (verified unreachable for supported
  devices).
- Seven hygiene items applied in files we were already editing
  (compression warn messages, named pad constants, AM025 citations,
  rewritten MAX_TASK_QUEUE_DEPTH comment, named STREAM_BUFFER_
  CAPACITY_WORDS, MemTile BD-channel hard error + new covering test,
  plus two doc-comment fold-ins from Task 3/5 code reviews).
- Bonus: the pre-existing `test_full_parse_all_devices` archspec
  failure (carried since Phase 1a, expected 12 devices vs 13 in the
  JSON) was fixed as Task 8 scope creep, giving the tag a clean
  archspec baseline for the first time since the refactor began.

Verification: `cargo test --lib` = 2687 / 0 / 5; archspec = 273 / 0 / 2
(first clean baseline!); release build clean; full HW bridge matches
Subsystem 2 baseline exactly (Chess 63 pass / 1 pre-existing fail,
Peano 51 pass / 3 pre-existing fail, HW 63+53 pass); ISA test suite
4815/4815 PASS (100.0%).
