# Subsystem 4 -- Locks -- Design

**Subsystem:** 4 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-21
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-21-subsys3-dma-engine-design.md](2026-04-21-subsys3-dma-engine-design.md)
**Planned tag:** `phase1-subsys-locks`

---

## Goal

Introduce a `LockModel` trait seam in `xdna-archspec` and migrate the
lock-value-width metadata and `sign_extend_lock_value` helper -- the
items explicitly pinned to xdna-emu's `DeviceRegLayout` wrapper during
Subsystem 3 -- out of xdna-emu and into archspec. The trait codifies
the arch-divergent pieces that Subsystem 3 left as "Subsystem 4
concerns": lock value width and bounds, acquire-mode support (AIE1 has
only one acquire mode; AIE2 has both acq-GE and acq-EQ), and
dynamic-value-ops support (AIE1 has no GetValue/SetValue on the lock
register; AIE2 does). After this subsystem, xdna-emu's `DeviceRegLayout`
wrapper struct collapses entirely -- only the `OnceLock` accessor +
config-aware `load_for_device()` function remain, because xdna-emu owns
the `Config` system for path resolution.

## Non-goals

- **No AIE1 or AIE2P `LockModel` implementation.** Subsystem 4 ships
  the trait, `LockValueLayout` carrier, `Aie2LockModel` impl, and the
  metadata migration. Populating AIE1 (2-bit binary-semaphore lock
  values, single acquire mode, no GetValue/SetValue) is orthogonal
  future work.
- **No `Lock` struct specialization.** `Lock::MIN_VALUE` / `MAX_VALUE`
  remain AIE2-hardcoded i8 constants. A drift-detection test asserts
  they agree with `AIE2_LOCK_MODEL.value_layout()`. When AIE1 lands,
  per-arch `Lock` bounds become a focused change; the trait surface is
  ready for it today.
- **No `LockArbiter` migration.** The round-robin arbiter holds
  runtime state (pending queue, priority pointer, stats). It stays in
  xdna-emu, same principle as DMA FSM in Subsystem 3: traits describe,
  they do not hold state.
- **No `supports_independent_lock_ids` relocation.** That flag lives on
  `DmaModel` and stays there. aie-rt enforces the AIE1 `Acq.LockId ==
  Rel.LockId` constraint in its DMA driver layer, not at the lock
  register; it is a BD-apply-time concern, not a lock-primitive
  concern. Moving it would leak the seam.
- **No hot-path dispatch change.** The trait is a construction-time
  lookup seam; consumers reading `value_layout()` in loops cache
  `&'static LockValueLayout` at construction rather than re-dispatching
  per call. No per-cycle trait calls.
- **No runtime behavior change for AIE2.** The trait is a dispatch
  seam; current AIE2 call sites should produce byte-identical
  decisions before and after. The test suite catches any drift.
- **No serialization / FFI surface change.** `LockModel` is not
  serialized; `LockValueLayout` is not exposed over FFI.
- **No second-arch implementation during the refactor.** Phase 1
  ground rule.

---

## Context

Subsystem 3 (DMA Engine & BD Format) landed at `phase1-subsys-dma`
(`d420fb7..3296bdd`). Baselines at that tag: `cargo test --lib` =
2687 pass, 0 fail, 5 ignored; `cargo test -p xdna-archspec --lib` =
273 pass, 0 fail, 2 ignored (the first clean archspec baseline since
Phase 1a began, after Task 8 fixed the pre-existing
`test_full_parse_all_devices` failure).

The DMA migration deliberately left three lock-related fields on
xdna-emu's wrapper (`lock_value_width` / `lock_value_mask` /
`lock_value_sign_bit`) plus the `sign_extend_lock_value` helper that
reads them, because moving only the helper without its backing width
metadata would have left a half-migrated lock-width concept straddling
the crate boundary. Subsystem 4 picks them up together.

### What the Explore audit found

A single Explore-agent sweep against the aie-rt lock implementation
(`driver/src/locks/xaie_locks_aie.c` for AIE1, `xaie_locks_aieml.c`
for AIE2) plus the `xaiegbl_reginit.c` / `xaiemlgbl_reginit.c` bounds
tables turned up a richer divergence picture than the NEXT-STEPS.md
pickup guide had staged. The guide assumed AIE1 and AIE2 both used
6-bit signed lock values; they do not.

**Arch-divergent behaviors that justify the trait:**

| # | Behavior | AIE1 | AIE2+ | Evidence |
|---|---|---|---|---|
| 1 | Lock value representation | address-encoded (`Lock.LockVal * LockMod->LockValOff`); software bounds -1..1 | 6-bit register field (bits [5:0]); logical 7-bit signed range -64..63 | `xaie_locks_aie.c` (address-encoded acquire/release) + `xaiegbl_reginit.c:1251-1252` bounds vs `xaiemlgbl_reginit.c:2452-2453` (63 / -64) |
| 2 | Acquire modes | single ACQ only | ACQ_GE + ACQ_EQ | `xaie_locks_aie.c` vs `xaiemlgbl_params.h:8048-8054` (separate event types per mode on AIE2) |
| 3 | Dynamic value ops (GetValue / SetValue) | absent (return XAIE_FEATURE_NOT_SUPPORTED) | present | `xaie_locks_aie.c:169,195` vs AIE2 LockSetValBase registers |
| 4 | Register layout topology | per-lock separate acq/rel registers at 0x1E000, 0x1E040 | unified Lock_Request at 0x40000, lock_id encoded in address | `xaiegbl_reginit.c` vs `xaiemlgbl_params.h:11057-11063` |

**Non-divergences (for the record):**

- Atomic acquire-release: **both arches** implement acquire and release
  as separate register writes. No arch exposes an atomic
  "acquire-then-release-with-different-value" primitive at the HAL.
  `ChannelFsm::AcquiringLock` / `ReleasingLock` separation in xdna-emu
  is correct for both arches.
- Lock count per tile: Compute=16 on both, Shim=16 on both.
  MemTile=64 on AIE2 only (AIE1 has no memtile). Already captured by
  `TileConfig::num_locks` in archspec.
- Cross-tile lock addressing: AIE2 memtile's 3-tile-wide 192-lock
  address space is memtile-specific; AIE1 has no memtile to address.
  Already handled by `TileKind::MemTile` gating + `num_locks` data.
- `Acq.LockId == Rel.LockId` enforcement on AIE1: enforced in aie-rt's
  DMA driver software (`xaie_dma_aie.c`), not at the lock hardware.
  Belongs on `DmaModel::supports_independent_lock_ids`, which it
  already does (shipped in Subsystem 3).

**Register-layout topology** (#4 in the divergences table) is flagged
for the record but does not cross the seam boundary: xdna-emu's `Lock`
struct does not mirror hardware register topology. On AIE2 we read
`Lock_Request[0]` for ACK; we would implement AIE1's per-lock-register
topology separately when AIE1 lands. The trait does not need a method
for register-layout shape.

### Call-site footprint

The pinned-to-xdna-emu fields and helper have a small consumer set:

- `src/device/tile/registers.rs:158` -- reads `reg_layout.lock_value_mask`
  during MMIO lock register read-back. Hot path: MMIO reads occur per
  host request, not per cycle.
- `src/device/state/compute.rs:47` -- reads `reg_layout.lock_value_mask`
  when building a lock register value for tile-state inspection.
- `src/device/state/mod.rs:45-48` -- thin wrapper
  `sign_extend_lock_value(reg_layout, raw)` function that delegates to
  the wrapper method.
- `src/device/state/compute.rs:25, 49` -- two calls to the wrapper
  function. Not on the per-cycle FSM path; these feed tile-state
  reporting for the GUI / trace output.
- `src/device/regdb/tests.rs:28-51` -- 5 assertions on the three
  fields and `sign_extend` behavior. Migrate to archspec.

None of these are on the per-cycle FSM hot path. Caching
`&'static LockValueLayout` at construction on the consumer structs
(`TileRegisters`, `ComputeStateBuilder` or whatever holds the state
path) is still the recommended pattern, but "caching" here means "hoist
the three-hop dispatch outside the loop if a loop exists."

---

## Design

### The shape-vs-values principle, applied to locks

Subsystem 3's DMA seam exists because DMA has many feature differences
the FSM and register-map code must branch on (task queue, interleave,
compression, BD iteration, etc.). Subsystem 4's lock seam exists for a
narrower reason: lock *value width* is genuinely different (binary
semaphore vs 7-bit signed), and a handful of features (acq-EQ, dynamic
value ops) exist on AIE2 and not AIE1. The trait surface reflects
this -- 3 methods, not 9.

The fourth arch-divergent item from the audit (register-layout
topology) is deliberately **not** on the trait: it is emulator-internal.
Our `Lock` struct does not mirror hardware register topology; it holds
a logical value and flags. Whatever register writes we emit on AIE1
(when that arch lands) are the business of the AIE1-specific register
encoder, not a trait consumer's concern.

### What lives where (post-migration)

| Item | Module | Notes |
|---|---|---|
| `LockModel` trait (3 methods) | `xdna_archspec::locks` | New seam |
| `LockValueLayout` carrier struct + `sign_extend(raw)` method | `xdna_archspec::locks` | 5 fields: `width`, `mask`, `sign_bit`, `min`, `max` |
| `Aie2LockModel` + `AIE2_LOCK_MODEL` static + `AIE2_LOCK_VALUE_LAYOUT` static | `xdna_archspec::aie2::locks` | Zero-sized impl, singleton; layout values from aie-rt + AM025 |
| `ArchConfig::lock_model()` accessor | `xdna_archspec::runtime` (added to trait) | Returns `&'static dyn LockModel`; dispatches on `architecture` |
| `Lock` struct + acquire/release ops + `LockArbiter` | `xdna_emu::device::tile::locks` | Stays -- runtime state |
| `OnceLock<DeviceRegLayout>` + `device_reg_layout()` accessor | `xdna_emu::device::regdb` | Stays -- `Config`-aware loader |
| xdna-emu's wrapper struct `DeviceRegLayout` | **deleted** | `pub use xdna_archspec::dma::DeviceRegLayout;` re-export only |

### The trait surface

```rust
// crates/xdna-archspec/src/locks/mod.rs
pub trait LockModel: Send + Sync + core::fmt::Debug {
    // Feature flags -- cold path, consulted at BD-parse / construction boundaries
    fn supports_acquire_eq(&self) -> bool;
    fn supports_dynamic_value_ops(&self) -> bool;

    // Data carrier -- read at construction, cached on consumers if hot
    fn value_layout(&self) -> &'static LockValueLayout;
}
```

Three methods, "coarse first":

- Two feature flags cover the two yes/no behavioral divergences
  (acquire-mode support; host-side GetValue/SetValue).
- `value_layout` returns a single data carrier that bundles width,
  mask, sign-bit, and the signed value bounds. `LockValueLayout` also
  carries the `sign_extend(raw: u32) -> i8` method because the
  formula is identical across arches -- only the inputs differ.

**Debug supertrait:** same reason as DmaModel in Subsystem 3 --
production code holds `Arc<dyn ArchConfig>` and needs `Debug` for
error contexts and structured log output.

**Not on the trait:**

- `Lock::MIN_VALUE` / `MAX_VALUE`: stay as hot-path-friendly compile-time
  i8 constants on the `Lock` struct, validated against
  `value_layout().min` / `max` by a drift-detection test. Arch-varying
  `Lock` construction is an AIE1-time concern.
- `LockArbiter` behavior: arbitration is structurally invariant
  (round-robin, per-cycle resolve). No arch varies it.
- `resolve_lock_id` / cross-tile lock addressing: data-driven by
  `TileConfig::num_locks` and `TileKind::is_mem()` today, no per-arch
  dispatch needed.
- Register topology / per-lock vs unified register: emulator-internal,
  handled by the per-arch register encoder when AIE1 lands.

### `LockValueLayout`

```rust
// crates/xdna-archspec/src/locks/mod.rs
/// Lock-value field layout -- register field shape plus logical value bounds.
///
/// - AIE2: 6-bit register field (bits [5:0], mask 0x3F), min=-64, max=63.
///   Logical range is 7-bit signed; values outside the 6-bit field range
///   alias when read back. aie-rt xaiemlgbl_reginit.c:2452-2453 for the
///   bounds; aie_registers_aie2.json `memory.Lock0_value.Lock_value` for
///   the field shape.
/// - AIE1: no read-back value field. AIE1's lock hardware is
///   address-encoded (`Lock.LockVal * LockMod->LockValOff`); value reads
///   come via the per-lock `LockN` status bit in the all-locks-status
///   register (1 bit per lock, not a per-lock value field). The
///   `width` / `mask` / `sign_bit` fields are not meaningful for AIE1
///   and AIE1's eventual `LockValueLayout` will either use sentinel
///   zeros or the trait will grow an `is_value_readable()` flag at
///   AIE1-implementation time. Bounds (min=-1, max=1) are enforced
///   software-side per aie-rt xaiegbl_reginit.c:1251-1252.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockValueLayout {
    pub width: u8,
    pub mask: u32,
    pub sign_bit: u8,
    pub min: i8,
    pub max: i8,
}

impl LockValueLayout {
    /// Sign-extend a raw register read to a signed 8-bit lock value.
    #[inline]
    pub fn sign_extend(&self, raw: u32) -> i8 {
        let masked = (raw & self.mask) as u8;
        if masked & (1 << self.sign_bit) != 0 {
            masked as i8 | !(self.mask as i8)
        } else {
            masked as i8
        }
    }
}
```

Identical formula to xdna-emu's current helper, moved to archspec with
`min` / `max` added so `Lock::MIN_VALUE` / `MAX_VALUE` have a
data-driven equivalent reachable from anywhere with an `ArchConfig` in
hand.

### AIE2 concrete impl

```rust
// crates/xdna-archspec/src/aie2/locks.rs

// Width and mask derived from mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json:
// memory.Lock0_value.Lock_value (width=6, mask=0x3F).
// Bounds from aie-rt xaiemlgbl_reginit.c:2452-2453
// (LockValUpperBound=63, LockValLowerBound=-64).
pub static AIE2_LOCK_VALUE_LAYOUT: LockValueLayout = LockValueLayout {
    width: 6,
    mask: 0x3F,
    sign_bit: 5,
    min: -64,
    max: 63,
};

#[derive(Debug)]
pub struct Aie2LockModel;

impl LockModel for Aie2LockModel {
    fn supports_acquire_eq(&self) -> bool { true }
    fn supports_dynamic_value_ops(&self) -> bool { true }
    fn value_layout(&self) -> &'static LockValueLayout { &AIE2_LOCK_VALUE_LAYOUT }
}

pub static AIE2_LOCK_MODEL: Aie2LockModel = Aie2LockModel;
```

**Build-time constants, not runtime-derived.** Subsystem 3's
`DmaTimingConfig` pattern -- static constants cited to aie-rt, paired
with a drift-detection test against the register DB. We keep the
runtime-safety property by asserting `width` / `mask` against the JSON
in a test, not by re-deriving every boot.

### Accessor threading

```rust
// crates/xdna-archspec/src/runtime.rs -- added to the ArchConfig trait
fn lock_model(&self) -> &'static dyn LockModel;
```

Implementation dispatches on `ModelConfig::architecture` the same way
`dma_model()` does (added in Subsystem 3):

```rust
fn lock_model(&self) -> &'static dyn LockModel {
    match self.architecture {
        Architecture::Aie2 | Architecture::Aie2p => {
            &crate::aie2::locks::AIE2_LOCK_MODEL
        }
        Architecture::Aie => {
            unimplemented!(
                "AIE1 LockModel not populated until AIE1 support lands \
                 (see docs/arch/lock-model.md for planned Aie1LockModel)"
            )
        }
    }
}
```

Exhaustive match (there is no `Aie2Ps` variant on `Architecture`;
post-stepping silicon revisions live on a separate `DeviceGeneration`
enum). The AIE1 arm is a deliberate `unimplemented!` until AIE1 lands,
mirroring what Subsystem 3's `dma_model()` did.

### Call-site migration

The six xdna-emu sites that read the migrated fields or helper:

1. **`src/device/tile/registers.rs:158`**
   ```rust
   // Before:
   return self.locks[lock_id].value as u32 & reg_layout.lock_value_mask;
   // After:
   let layout = arch.lock_model().value_layout();
   return self.locks[lock_id].value as u32 & layout.mask;
   ```

2. **`src/device/state/compute.rs:47`** -- same `.mask` rewrite.

3. **`src/device/state/mod.rs:45-48`** -- the local wrapper function
   deletes; callers use `value_layout().sign_extend(raw)` directly.

4. **`src/device/state/compute.rs:25, 49`** -- two `sign_extend_lock_value(reg_layout, raw)`
   calls rewrite to `arch.lock_model().value_layout().sign_extend(raw)`.
   For the consumer struct (compute state builder), hoist the
   `&'static LockValueLayout` at construction time and use `layout.sign_extend`
   directly in the body.

5. **`src/device/regdb/tests.rs:28-51`** -- the 5 assertions on
   `lock_value_width` / `lock_value_mask` / `lock_value_sign_bit` and
   `sign_extend_lock_value` migrate to
   `crates/xdna-archspec/src/locks/mod.rs` tests module (alongside
   the new `LockValueLayout` tests).

6. **`docs/archive/accuracy-sweep/verified/locks.md:91`** and
   **`shim-memtile.md:171`** -- comment references to
   `sign_extend_lock_value` update to the new location.

### `DeviceRegLayout` wrapper collapse

After the fields / method / constructors migrate, xdna-emu's
`src/device/regdb/mod.rs` keeps only the `OnceLock` accessor and the
config-aware `load_for_device()` function (because `Config` is an
xdna-emu concern). The wrapper struct, its `Deref` impl, the three
lock fields, `sign_extend_lock_value`, and the `from_regdb` /
`load_for_device` method-form constructors all delete. Net reduction
in this file: ~65 LOC.

**Stays (Subsystem 3 re-exports, unchanged):**

```rust
pub use xdna_archspec::regdb::*;
pub use xdna_archspec::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout,
    MemTileBdFieldLayout, ShimBdFieldLayout,
    ShimMuxField, ShimMuxLayout,
    StreamSwitchLayout, ModuleEventLayout,
};
```

**Changes:** the existing
`pub use xdna_archspec::dma::DeviceRegLayout as ArchDeviceRegLayout;`
aliased re-export becomes a plain
`pub use xdna_archspec::dma::DeviceRegLayout;` once the wrapper struct
of the same name disappears. Plan-writers: do not leave the `as
ArchDeviceRegLayout` alias in place -- it exists today only to
disambiguate the wrapper from the archspec type, and that disambiguation
disappears with the wrapper. Consumer imports from
`crate::device::regdb::DeviceRegLayout` keep working unchanged across
the rename.

---

## Alternatives considered

### Why not a split trait (`LockValueModel` + `LockOpModel`)

Separating the data axis (value layout) from the behavior axis (acquire
mode, dynamic ops) would mirror Subsystem 3's rejected "split
`BdLayout` / `DmaBehavior`" approach. Rejected for the same reason:
every real device has both axes linked -- no arch would have AIE2's
value width with AIE1's acquire-mode support. The split is ceremony
for a 3-method trait and adds a second seam consumers have to thread.
If future arches introduce a value-width / operations cross-product
the single trait cannot express, promoting to two traits is a
mechanical refactor with no consumer impact.

### Why not per-tick behavioral hooks

Adding `fn pre_acquire(&self, ctx: &ArbiterContext) -> Option<LockPhase>`
or similar per-cycle dispatch points would pre-emptively serve AIE1's
potential binary-semaphore arbitration differences. Rejected because
the arbitration logic is structurally invariant (round-robin,
one-grant-per-lock-per-cycle) regardless of value width; AIE1's
differences show up at *value-bound-check* time, which the existing
`Lock::try_apply` / acquire-mode boundary already handles through
data. Adding hooks now costs AIE2 hot-path dispatch for consumers no
AIE2 code calls.

### Why not absorb `supports_independent_lock_ids` from `DmaModel`

aie-rt enforces the AIE1 `Acq.LockId == Rel.LockId` constraint in its
DMA driver software (`xaie_dma_aie.c`), not at the lock hardware
itself. The BD register fields on both arches support independent IDs;
AIE1's driver just refuses them. That makes it a DMA-BD-apply-time
check, not a lock-primitive concern. Moving it to `LockModel` would
leak the seam: the check happens at BD-parse time when neither the
lock nor its arbiter is in the picture yet.

### Why not move `Lock` / `LockArbiter` into archspec

Both hold mutable runtime state: `Lock` has `value` / `overflow` /
`underflow` fields, `LockArbiter` has a priority pointer and pending
queue. Traits describe; state lives in plain structs in runtime. Same
principle as DMA FSM in Subsystem 3.

### Why not runtime-derive `LockValueLayout` from regdb

Keeping `Aie2LockModel::new_from_regdb(db)` + a `OnceLock<Aie2LockModel>`
preserves the current "if the JSON changes, we notice" property.
Rejected: a drift-detection test comparing the static constants to
the JSON at test time serves the same purpose with a simpler trait
(no construction helper, no laziness). This mirrors Subsystem 3's
`DmaTimingConfig` treatment -- static constants with explicit aie-rt
citations and a JSON-drift test.

---

## Testing

### New tests in archspec

**`crates/xdna-archspec/src/locks/mod.rs` (tests module):**

- `lock_value_layout_sign_extend_smoke` -- mirrors the existing
  `regdb/tests.rs:33-36` cases exactly: `sign_extend(0) == 0`,
  `sign_extend(31) == 31`, `sign_extend(0x20) == -32`,
  `sign_extend(0x3F) == -1`, `sign_extend(0xFF) == -1`.
- `aie2_lock_model_feature_flags` -- both `supports_acquire_eq()` and
  `supports_dynamic_value_ops()` return `true`.
- `aie2_lock_value_layout_constants` -- `width=6, mask=0x3F,
  sign_bit=5, min=-64, max=63`.

**`crates/xdna-archspec/src/aie2/locks.rs` (drift-detection):**

- `aie2_lock_layout_matches_regdb` -- load
  `aie_registers_aie2.json`, look up `memory.Lock0_value.Lock_value`,
  assert `width == AIE2_LOCK_VALUE_LAYOUT.width` and
  `mask == AIE2_LOCK_VALUE_LAYOUT.mask`. If the JSON ever changes
  shape, this test fires.

### New tests in xdna-emu

**`src/device/tile/locks.rs` (tests module):**

- `lock_bounds_match_archspec` -- asserts
  `Lock::MIN_VALUE == AIE2_LOCK_MODEL.value_layout().min` and
  `Lock::MAX_VALUE == .max`. Guards the AIE1-specialization follow-up:
  the day `Lock::MIN/MAX` go data-driven, this test still holds (it
  just moves to per-arch `Lock` types).

**Tests that move / update:**

- `src/device/regdb/tests.rs:28-37` -- the 5
  `lock_value_width` / `lock_value_mask` / `lock_value_sign_bit` and
  `sign_extend_lock_value` assertions in
  `test_device_reg_layout_lock_value_extension` move to archspec's
  new tests module alongside `LockValueLayout`. The xdna-emu-side
  test function deletes.
- `src/device/regdb/tests.rs:51` -- `test_load_for_device_uses_config`
  contains a `layout.lock_value_width == 6` assertion that references
  a field that will no longer exist on the post-migration type. Drop
  that one line; the existing `memory_bd_base == 0x1D000` assertion
  at line 48 already covers the "load succeeded and returned a sane
  layout" contract that the test is really checking.
- The `regdb/tests.rs` file **stays** after the updates --
  `test_load_for_device_uses_config` still tests the xdna-emu-side
  `Config`-aware loader, which does not move.

### Gate tests (Global Invariant, green at every commit and at tag)

- `cargo test --lib` green (baseline 2687 / 0 / 5 at
  `phase1-subsys-dma`).
- `cargo test -p xdna-archspec --lib` green (baseline 273 / 0 / 2
  clean).
- Full bridge run matches Subsystem 3 baseline exactly (Chess
  63 pass / 1 pre-existing fail, Peano 51 pass / 3 pre-existing fail,
  HW 63+53 pass).
- ISA test suite 4815/4815 PASS (100%).

---

## Documentation

- `docs/arch/lock-model.md` -- per-seam design note following the
  `dma-model.md` template (what lives where, shape-vs-values applied
  to locks, trait surface, what would AIE1 look like, why not split
  trait, why not hot-path hooks, `supports_independent_lock_ids`
  placement rationale, Completion section filled at tag time).
- `docs/arch/subsys4-audit.md` -- audit document with baseline
  state, decisions, task breakdown, and Completion section filled at
  tag time. Template matches `subsys3-audit.md`.
- Update `NEXT-STEPS.md` -- tag changes from `phase1-subsys-dma` to
  `phase1-subsys-locks`; Subsystem 5 (Stream Switch) moves to
  "up next" with shaping questions.
- Update `docs/archive/accuracy-sweep/verified/locks.md:91` and
  `shim-memtile.md:171` -- comment refs to `sign_extend_lock_value`
  location.

---

## Future direction (recorded, not in scope for Subsystem 4)

The emulator's purpose is cross-arch comparison -- load a binary, see
how it performs on AIE2, AIE2P, AIE1 side by side. That implies a
single binary that can present any supported arch at runtime, which in
turn implies **generic-type-parameter monomorphization on hot types**
is the right target for the post-seam-pass optimization phase:

```rust
// Illustrative; not this subsystem's work.
struct DmaEngine<D: DmaModel, L: LockModel> { ... }
```

After all 8 seam subsystems land, hot paths that today read
`&'static dyn LockModel` / `&'static dyn DmaModel` can switch to a
monomorphized `<L: LockModel>` parameter without re-examining any
seam decision. Per-arch build targets (cargo features) were
considered and rejected -- they would preclude GUI runtime
arch-switching, which is a core product requirement.

Subsystem 4 does nothing that impedes this future direction. The
`AIE2_LOCK_VALUE_LAYOUT` static is `pub` specifically so a future
AIE2-specialized hot path can reach into it directly with zero
dyn-dispatch.

---

## Follow-ups flagged (addressable in later subsystems or Phase 2 hygiene)

- **Per-arch `Lock::MIN_VALUE` / `MAX_VALUE`:** lands with AIE1
  implementation. Drift-detection test in Subsystem 4 locks the AIE2
  values to `AIE2_LOCK_VALUE_LAYOUT` so the follow-up is focused.
- **`docs/archive/accuracy-sweep/verified/locks.md` refresh:** the
  archived accuracy-sweep notes reference `sign_extend_lock_value` by
  its old location. A comment pointer update is in-scope; a full
  refresh of the archived docs is not.
- **Generic monomorphization of hot types:** dedicated post-seam pass
  (see Future Direction).

---

## Subsystem 5 forward pointer

Subsystem 5 (Stream Switch) is next after this ships. Shaping
questions it will need to answer: topology (data, already via archspec
since Phase 1a / Subsystem 1) vs routing legality (behavior), trait
shape for `StreamSwitchModel`, whether packet-switch vs circuit-switch
configuration is arch-divergent (likely: AIE2 has packet IDs, AIE1
circuit-only in most configurations), and whether shim-mux port
configuration shape differs between AIE1 and AIE2+. All
pre-investigation, not this subsystem's work.
