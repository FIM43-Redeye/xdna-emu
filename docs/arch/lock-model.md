# Lock Model -- Design Note

**Subsystem:** 4 (Phase 1b)
**Tag:** `phase1-subsys-locks`
**Spec:** [../archive/specs/2026-04-21-subsys4-locks-design.md](../archive/specs/2026-04-21-subsys4-locks-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains the shape difference that
justifies the `LockModel` trait and what AIE1 / AIE2P impls will look
like.

---

## What lives where

All entries below reflect state as of the `phase1-subsys-locks` tag.

| Data/code | Module | Source |
|---|---|---|
| `LockModel` trait (3 methods) + `LockValueLayout` carrier | `xdna_archspec::locks` | Emulator design |
| `Aie2LockModel` concrete impl + `AIE2_LOCK_MODEL` / `AIE2_LOCK_VALUE_LAYOUT` statics | `xdna_archspec::aie2::locks` | aie-rt xaiemlgbl_reginit.c + AM025 register DB JSON |
| `ArchConfig::lock_model()` accessor | `xdna_archspec::runtime` | Dispatches on `Architecture` |
| `Lock` struct + `LockArbiter` (runtime state) | `xdna_emu::device::tile::locks` | Unchanged |
| `OnceLock<DeviceRegLayout>` + config-aware loader | `xdna_emu::device::regdb` | Unchanged (xdna-emu owns `Config`) |

xdna-emu's `DeviceRegLayout` wrapper struct (the Subsystem 3 holdover)
collapsed entirely in this subsystem. `pub use xdna_archspec::dma::DeviceRegLayout;`
is now the single source of the type; all consumer imports from
`crate::device::regdb::DeviceRegLayout` resolve through the re-export
unchanged.

---

## The shape-vs-values principle, applied to locks

Subsystem 3 (DMA) introduced a 9-method trait because DMA has many
feature differences: task queue, interleave, compression, BD iteration,
etc. Subsystem 4 (Locks) introduces a 3-method trait because lock
*value width* is genuinely different (AIE1 binary semaphore vs AIE2
7-bit signed) and two features (acq-EQ, dynamic value ops) are
AIE2-only. The narrower trait surface reflects narrower variance.

Concretely:

- **AIE2 (NPU1/NPU4/NPU5/NPU6):** 6-bit register field (mask 0x3F),
  logical 7-bit signed range -64..63, acq-GE + acq-EQ both supported,
  GetValue/SetValue both supported. Values outside the 6-bit field
  alias when read back.
- **AIE1 (Versal, e.g., xcvc1902):** address-encoded lock hardware
  (no read-back value field); software-side bounds -1..1 (binary
  semaphore); single acquire mode (no acq-EQ); GetValue/SetValue
  return `XAIE_FEATURE_NOT_SUPPORTED` at aie-rt level.
- **AIE2P:** expected to match AIE2's trait values 1:1.

---

## The trait surface

```rust
pub trait LockModel: Send + Sync + core::fmt::Debug {
    fn supports_acquire_eq(&self) -> bool;
    fn supports_dynamic_value_ops(&self) -> bool;
    fn value_layout(&self) -> &'static LockValueLayout;
}
```

Three methods, "coarse first":

- Two feature flags cover the two yes/no behavioral divergences
  (acquire-mode support; host-side GetValue/SetValue).
- `value_layout` returns a single data carrier (width, mask, sign_bit,
  min, max) with a `sign_extend(raw: u32) -> i8` method. Bundles the
  layout fields together because they are only meaningful together.

Not on the trait:

- `Lock::MIN_VALUE` / `MAX_VALUE` constants on the `Lock` struct:
  stay hot-path-friendly AIE2-hardcoded i8 constants, validated against
  `value_layout().min / max` by a drift-detection test. Arch-varying
  `Lock` construction is an AIE1-time concern.
- `LockArbiter` arbitration: structurally invariant (round-robin, one
  grant per lock per cycle) regardless of value width.
- Cross-tile lock addressing: already handled by `TileKind::MemTile`
  gating + `TileConfig::num_locks` data.
- Register topology: emulator-internal, per-arch register encoder
  handles it when AIE1 lands.

---

## What would AIE1 look like?

- `xdna_archspec::aie1::locks::Aie1LockModel` zero-sized struct +
  `AIE1_LOCK_MODEL` static.
- `supports_acquire_eq` and `supports_dynamic_value_ops` both return
  `false`.
- `value_layout` returns a `LockValueLayout` with sentinel fields
  (width=0, mask=0, sign_bit=0) reflecting that AIE1 has no read-back
  value field, plus min=-1, max=1 for the software-side bounds. **OR**
  the `LockModel` trait grows an `is_value_readable()` method at that
  point. Decision deferred to AIE1 landing.
- `ArchConfig::lock_model()` adds an arm for `Architecture::Aie`
  returning `&AIE1_LOCK_MODEL`.
- The `Lock` struct on xdna-emu's side specializes: `Lock::MIN_VALUE`
  and `MAX_VALUE` become per-arch (either via a type parameter or via
  separate `Aie1Lock` / `Aie2Lock` structs). The drift-detection test
  added in Subsystem 4 locks the AIE2 values; AIE1's equivalent test
  would lock AIE1's.

Call sites in xdna-emu require no changes: the 6 sites that read
`value_layout()` in this subsystem work the same on both arches
(they read the mask / sign-extend as data, not as behavior).

---

## Why not a split trait (rejected Approach 2 in spec)

A `LockValueModel` (data: value layout) + `LockOpModel` (behavior:
acquire mode, dynamic ops) split would separate concerns cleanly.
Rejected because every real device has both axes linked -- no arch
has AIE2's value width with AIE1's acquire-mode support. The split
is ceremony for a 3-method trait.

## Why not per-tick behavioral hooks (rejected Approach 3 in spec)

Adding per-arbiter-cycle dispatch points (e.g., `pre_acquire` /
`post_release`) would pre-emptively serve AIE1's binary-semaphore
arbitration. Rejected because the round-robin arbitration logic is
structurally invariant; AIE1's differences show up at value-bounds
time, handled by data (the `min` / `max` fields in `LockValueLayout`)
not per-cycle dispatch.

## Why not absorb `supports_independent_lock_ids`

aie-rt enforces the AIE1 `Acq.LockId == Rel.LockId` constraint in its
DMA driver software, not at the lock hardware. BD register fields on
both arches support independent IDs; AIE1's DMA driver just refuses
them. That makes it a DMA-BD-apply-time check, which is why it stayed
on `DmaModel` when Subsystem 3 added it.

---

## Completion (2026-04-21)

Landed at `phase1-subsys-locks`. Net effect:

- `xdna_archspec::locks::LockModel` trait (3 methods:
  `supports_acquire_eq`, `supports_dynamic_value_ops`, `value_layout`)
  + `LockValueLayout` carrier struct (width, mask, sign_bit, min, max)
  with `sign_extend(raw: u32) -> i8` method live at the crate root.
- `xdna_archspec::aie2::locks::Aie2LockModel` concrete impl +
  `AIE2_LOCK_MODEL` static + `AIE2_LOCK_VALUE_LAYOUT` static for
  AIE2-family devices (NPU1/NPU4/NPU5/NPU6).
- `xdna_archspec::runtime::ArchConfig::lock_model()` accessor
  returns `&'static dyn LockModel`, dispatching on
  `ModelConfig::architecture` exactly like `dma_model()`.
- xdna-emu's `DeviceRegLayout` wrapper struct (the Subsystem 3
  holdover) fully collapsed: the three lock fields, `sign_extend_lock_value`
  method, `Deref` impl, and `from_regdb` / `load_for_device`
  method-form constructors all deleted. `src/device/regdb/mod.rs`
  shrunk from ~138 LOC to ~60 LOC (re-exports + OnceLock accessor +
  config-aware loader function).
- Six xdna-emu call sites (`tile/registers.rs:158`,
  `state/compute.rs` write_lock_value + mask_write_lock_value,
  and the deleted `state/mod.rs` wrapper fn) migrate to
  `arch_handle::lock_value_layout()` + `layout.sign_extend()` /
  `layout.mask`.
- New `src/device/arch_handle.rs` module provides a process-global
  cache of `&'static LockValueLayout`, seeded from `default_arch().lock_model()`
  on first access. Bridge pattern until GUI runtime arch-switch
  lands.
- Drift-detection test in `src/device/tile/locks.rs` asserts
  `Lock::MIN_VALUE` / `MAX_VALUE` match
  `AIE2_LOCK_VALUE_LAYOUT.min` / `.max`.
- Regdb drift-detection test in `xdna_archspec::aie2::locks`
  asserts the static constants agree with the
  `memory.Lock0_value.Lock_value` field in the regdb JSON.

Verification: `cargo test --lib` = 2687 passed / 0 failed / 5 ignored;
archspec = 282 passed / 0 failed / 2 ignored; full HW bridge matches
phase1-subsys-dma character (Chess HW 63 pass / 1 fail, Peano HW 53
pass / 1 fail / 1 XFAIL -- the single HW fail on each compiler is the
pre-existing `bd_chain_repeat_on_memtile` EMU deadlock); ISA test
suite 4815/4815 PASS (100.0%).
