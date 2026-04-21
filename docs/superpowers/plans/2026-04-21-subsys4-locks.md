# Subsystem 4 -- Locks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `LockModel` trait seam in `xdna-archspec` with an `Aie2LockModel` concrete impl, migrate the lock-value-width metadata + `sign_extend_lock_value` helper from xdna-emu's `DeviceRegLayout` wrapper into archspec, thread `&'static dyn LockModel` through `ArchConfig`, update the 6 xdna-emu call sites that read the migrated fields, collapse the xdna-emu wrapper struct, and land a drift-detection test locking AIE2's `Lock::MIN_VALUE` / `MAX_VALUE` to the archspec-side layout.

**Architecture:** Pure seam + metadata migration, no hot-path changes. Trait + impl land first (Task 2), then the accessor on `ArchConfig` (Task 3), then xdna-emu call-site rewrites (Task 4), then the wrapper collapse + test migration (Task 5), then the drift-detection test (Task 6), then the gate + tag (Task 7). Consumers reach the layout via `arch.lock_model().value_layout()` at cold-path call sites; hot-path consumers (none today) would cache `&'static LockValueLayout` at construction.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, `&'static dyn LockModel` trait-object dispatch, AM025 register DB JSON loaded from `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`, aie-rt evidence from `xaiegbl_reginit.c` and `xaiemlgbl_reginit.c`.

**Spec:** [docs/superpowers/specs/2026-04-21-subsys4-locks-design.md](../specs/2026-04-21-subsys4-locks-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

**Prior subsystem:** `phase1-subsys-dma` (Subsystem 3, 2026-04-21).

---

## Scope Note

Single-part subsystem with one tag (`phase1-subsys-locks`) at the end. Scope is intentionally small because Subsystem 3 did the heavy lifting on the `DeviceRegLayout` migration -- Subsystem 4 only needs to finish the lock-specific residue.

- New in archspec: `LockModel` trait + `LockValueLayout` + `Aie2LockModel` impl + `AIE2_LOCK_MODEL` + `AIE2_LOCK_VALUE_LAYOUT` + `ArchConfig::lock_model()` accessor + ~25 LOC of tests (~150 LOC total).
- Modified in xdna-emu: 6 call sites in 3 files (registers.rs, state/compute.rs, state/mod.rs), `DeviceRegLayout` wrapper collapse in `regdb/mod.rs`, test migration in `regdb/tests.rs`, new drift-detection test in `tile/locks.rs`.
- Estimated file-count: ~10 files touched, ~7 commits.
- **No Part A / Part B split** expected. If the file count exceeds 15 or commit count exceeds 10, pause and flag -- but the spec's call-site inventory suggests this is comfortable single-tag scope.

Branch: `dev`. Tag at end: `phase1-subsys-locks`.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green. Baseline at `phase1-subsys-dma`: `2687 passed; 0 failed; 5 ignored`.
- `cargo test -p xdna-archspec --lib` green. Baseline: `273 passed; 0 failed; 2 ignored` (clean -- the pre-existing `test_full_parse_all_devices` was fixed in Subsystem 3's Task 8).
- `cargo build` green. `cargo build --release` clean is required before the tag, not every commit.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green after rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`).
- No commit introduces `TODO` / `FIXME` / `unimplemented!()` without an open-issue reference (the `unimplemented!("AIE1 LockModel ...")` in the accessor is the one sanctioned exception, mirroring `dma_model()`).
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`, `refactor(archspec):`); no emoji; ends with `Generated using Claude Code.`.
- All work on `dev`. No merges to `master` during this plan.
- **Every `cargo` call** must have `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` prepended (tblgen needs llvm-config 21.x, not mlir-aie's 23.x).

---

## File Structure

**Current layout (post-Subsystem 3, at `phase1-subsys-dma`):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── regdb/
│   │   │   ├── mod.rs              # 138 LOC: wrapper struct (lock_value_* fields, Deref)
│   │   │   │                       #          + OnceLock + load_for_device
│   │   │   └── tests.rs            # 57 LOC: 2 tests (lock-value-extension + load_for_device_uses_config)
│   │   ├── state/
│   │   │   ├── mod.rs              # line 47: sign_extend_lock_value wrapper fn
│   │   │   └── compute.rs          # lines 15, 25, 36, 47, 49: reg_layout.lock_value_mask + sign_extend
│   │   ├── tile/
│   │   │   ├── registers.rs        # line 158: reg_layout.lock_value_mask
│   │   │   └── locks.rs            # Lock struct, LockArbiter (stays in place)
│   │   └── ...
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                  # (will gain `pub mod locks;`)
        ├── aie2/
        │   ├── mod.rs              # (will gain `pub mod locks;`)
        │   ├── dma.rs              # (Subsystem 3 reference -- mimic structure)
        │   └── ...
        ├── dma/                    # (Subsystem 3 -- reference for trait+impl shape)
        ├── runtime.rs              # ArchConfig trait + ModelConfig impl
        │                           #   (dma_model() already here; lock_model() joins it)
        └── ...
```

**Target layout (post-Subsystem 4):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── regdb/
│   │   │   ├── mod.rs              # shrunk to ~70 LOC: just OnceLock accessor +
│   │   │   │                       # config-aware load_for_device (function form).
│   │   │   │                       # pub use xdna_archspec::dma::DeviceRegLayout;
│   │   │   │                       # pub use xdna_archspec::dma::field_layouts::{..};
│   │   │   └── tests.rs            # shrunk: lock tests moved to archspec; only
│   │   │                           # load_for_device integration test remains
│   │   ├── state/
│   │   │   ├── mod.rs              # sign_extend_lock_value wrapper fn DELETED
│   │   │   └── compute.rs          # call sites use layout.mask / layout.sign_extend
│   │   ├── tile/
│   │   │   ├── registers.rs        # uses arch.lock_model().value_layout().mask
│   │   │   └── locks.rs            # + drift-detection test against archspec
│   │   └── ...
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                  # + pub mod locks;
        ├── locks/                  # NEW
        │   └── mod.rs              # LockModel trait + LockValueLayout + tests
        ├── aie2/
        │   ├── mod.rs              # + pub mod locks;
        │   └── locks.rs            # NEW: Aie2LockModel + AIE2_LOCK_MODEL +
        │                           #      AIE2_LOCK_VALUE_LAYOUT + drift test
        └── runtime.rs              # ArchConfig gains lock_model() method
```

Archspec new LOC: ~180 (trait + carrier + AIE2 impl + singletons + tests). Archspec migrated LOC: ~25 (5 lock-value tests migrated from xdna-emu/regdb/tests.rs). xdna-emu shrinks by ~75 LOC (wrapper struct collapses). Net workspace LOC: small net increase from the new trait/tests.

---

## Baseline to Preserve

Before Task 1, capture current numbers so later regression checks have a target:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected current values at `phase1-subsys-dma`:

- xdna-emu library tests: `2687 passed; 0 failed; 5 ignored`
- archspec library tests: `273 passed; 0 failed; 2 ignored`

Record these in `docs/arch/subsys4-audit.md` (created in Task 1).

---

### Task 1: Audit + design-note scaffolding

**Goal:** Create `docs/arch/subsys4-audit.md` with the baseline numbers + the audit facts. Create the `docs/arch/lock-model.md` design note with the mandatory per-seam structure, leaving the "Completion" section as a stub for Task 7 to fill.

**Files:**
- Create: `docs/arch/subsys4-audit.md`
- Create: `docs/arch/lock-model.md`

- [ ] **Step 1: Create the audit doc skeleton**

Write `docs/arch/subsys4-audit.md`:

```markdown
# Subsystem 4 -- Locks Audit

## Baseline (pre-subsystem, at phase1-subsys-dma tag)

- `cargo test --lib`: <paste output>
- `cargo test -p xdna-archspec --lib`: <paste output>

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit facts

### Pinned-to-xdna-emu items inherited from Subsystem 3

From `src/device/regdb/mod.rs`:
- Wrapper struct `DeviceRegLayout` (fields `arch`, `lock_value_width`, `lock_value_mask`, `lock_value_sign_bit`).
- Method `sign_extend_lock_value(&self, raw: u32) -> i8`.
- Constructor `from_regdb(db) -> Result<Self>` (derives width/mask/sign_bit from `memory.Lock0_value.Lock_value` field).
- Method `load_for_device(device: &str) -> Result<Self>`.
- `Deref<Target = ArchDeviceRegLayout>` impl (source-compat shim from Subsystem 3).

### Call-site inventory (xdna-emu)

`lock_value_mask` direct reads (2 sites):
- `src/device/tile/registers.rs:158` -- MMIO lock register read-back.
- `src/device/state/compute.rs:47` -- masked lock value write.

`sign_extend_lock_value` calls via wrapper in `state/mod.rs` (2 sites):
- `src/device/state/compute.rs:25` -- write_lock_value sign-extends the input value.
- `src/device/state/compute.rs:49` -- mask_write_lock_value sign-extends after applying the mask.

Wrapper helper fn `sign_extend_lock_value` at `src/device/state/mod.rs:47-49` (1 site, deletes).

Tests in `src/device/regdb/tests.rs` (2 tests):
- `test_device_reg_layout_lock_value_extension` (lines 18-38) -- 3 field assertions + 5 sign_extend assertions. **Migrates to archspec.**
- `test_load_for_device_uses_config` (lines 40-57) -- `memory_bd_base` + `lock_value_width` assertions. **Stays, with the lock_value_width line removed.**

Doc references (2 sites, comment updates only):
- `docs/archive/accuracy-sweep/verified/locks.md:91`.
- `docs/archive/accuracy-sweep/verified/shim-memtile.md:171`.

### aie-rt AIE1 vs AIE2 lock divergence (evidence base for trait)

Confirmed at `../aie-rt/driver/src/`:

| # | Behavior | AIE1 | AIE2 | Source |
|---|---|---|---|---|
| 1 | Lock value representation | address-encoded (Lock.LockVal * LockMod->LockValOff); software bounds -1..1 | 6-bit reg field ([5:0]); logical 7-bit signed -64..63 | `xaiegbl_reginit.c:1251-1252` / `xaiemlgbl_reginit.c:2452-2453` |
| 2 | Acquire modes | single ACQ | ACQ_GE + ACQ_EQ | `xaiemlgbl_params.h:8048-8054` (event type pairs) |
| 3 | Dynamic value ops (Get/Set) | absent (returns XAIE_FEATURE_NOT_SUPPORTED) | present | `xaie_locks_aie.c:169, 195` vs AIE2 LockSetValBase registers |
| 4 | Register layout topology | per-lock separate acq/rel at 0x1E000, 0x1E040 | unified Lock_Request at 0x40000 | emulator-internal; not a trait concern |

Non-divergences (for the record):
- Atomic acquire-release: both arches use separate register writes.
- Per-tile lock counts: Compute=16, Shim=16 on both. MemTile=64 on AIE2 only.
- `Acq.LockId == Rel.LockId` enforcement on AIE1: enforced in aie-rt's DMA driver (`xaie_dma_aie.c`), stays on `DmaModel::supports_independent_lock_ids` (Subsystem 3).

## Completion

*(To be filled in by Task 7.)*
```

- [ ] **Step 2: Fill in the baseline numbers**

Run:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Paste each output into the two `<paste output>` placeholders in `docs/arch/subsys4-audit.md`.

- [ ] **Step 3: Create the design-note skeleton**

Write `docs/arch/lock-model.md`:

```markdown
# Lock Model -- Design Note

**Subsystem:** 4 (Phase 1b)
**Tag:** `phase1-subsys-locks`
**Spec:** [../superpowers/specs/2026-04-21-subsys4-locks-design.md](../superpowers/specs/2026-04-21-subsys4-locks-design.md)

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

## Completion

*(To be filled in by Task 7 with final LOC counts, test deltas, and
specific commit shas.)*
```

- [ ] **Step 4: Verify no test regression from the new docs**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: 2687 passed / 0 failed / 5 ignored. (Docs don't change test behavior, but verify anyway.)

- [ ] **Step 5: Commit**

```bash
git add docs/arch/subsys4-audit.md docs/arch/lock-model.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 4 audit + lock-model design note scaffolds

Captures baseline test numbers at phase1-subsys-dma, enumerates the
6 call sites + Subsystem 3 pinned items, and sketches the LockModel
trait surface with its AIE1 / AIE2P "what would it look like?"
sections. Completion sections filled by Task 7.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: `LockModel` trait + `LockValueLayout` + `Aie2LockModel` impl

**Goal:** Add the `LockModel` trait + `LockValueLayout` carrier struct to `xdna-archspec` at a new top-level `locks/` module. Add the `Aie2LockModel` struct + `AIE2_LOCK_MODEL` / `AIE2_LOCK_VALUE_LAYOUT` statics at a new `aie2/locks.rs` module. Ship unit tests for the AIE2 impl and a drift-detection test against the regdb JSON.

**Files:**
- Create: `crates/xdna-archspec/src/locks/mod.rs`
- Create: `crates/xdna-archspec/src/aie2/locks.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod locks;`)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (add `pub mod locks;`)

- [ ] **Step 1: Write the trait + carrier struct**

Create `crates/xdna-archspec/src/locks/mod.rs`:

```rust
//! Lock model trait: feature flags and value-layout carrier for
//! per-arch lock behavior.
//!
//! Subsystem 4 of the device-family refactor introduces this trait
//! as a behavioral seam. AIE2 / AIE2P use the concrete
//! `aie2::locks::Aie2LockModel` impl (6-bit register field, acq-EQ
//! + dynamic value ops enabled). AIE1's eventual
//! `aie1::locks::Aie1LockModel` will reflect address-encoded locks
//! (no read-back field), binary-semaphore software bounds, and
//! no acq-EQ.
//!
//! Consumers access an impl via `ArchConfig::lock_model()`, which
//! returns a `&'static dyn LockModel`. Because every concrete impl
//! is zero-sized and stateless, the accessor returns a reference to
//! a `static` singleton -- no allocation, no lifetime bookkeeping.
//!
//! The trait is intentionally coarse (3 methods, all cold-path).
//! Hot-path consumers (none today) cache `&'static LockValueLayout`
//! at construction rather than re-dispatching per call.

#[cfg(test)]
mod tests;

/// Lock-value field layout -- register field shape plus logical value bounds.
///
/// Field shape and bounds diverge per-arch:
/// - AIE2: 6-bit register field (bits [5:0], mask 0x3F), min=-64, max=63.
///   Logical range is 7-bit signed; values outside the 6-bit field
///   alias when read back. aie-rt `xaiemlgbl_reginit.c:2452-2453` for
///   the bounds; `aie_registers_aie2.json` `memory.Lock0_value.Lock_value`
///   for the field shape.
/// - AIE1: no read-back value field. AIE1's lock hardware is
///   address-encoded (`Lock.LockVal * LockMod->LockValOff`); value reads
///   come via the per-lock `LockN` status bit in the all-locks-status
///   register (1 bit per lock, not a per-lock value field). The
///   `width` / `mask` / `sign_bit` fields are not meaningful for AIE1;
///   bounds (min=-1, max=1) are enforced software-side per aie-rt
///   `xaiegbl_reginit.c:1251-1252`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockValueLayout {
    /// Width of the Lock_value register field, in bits.
    pub width: u8,
    /// Mask that isolates the field within a raw 32-bit register read.
    pub mask: u32,
    /// Position of the sign bit within the field (always `width - 1`
    /// for two's-complement representations).
    pub sign_bit: u8,
    /// Minimum logical lock value (aie-rt LockValLowerBound).
    pub min: i8,
    /// Maximum logical lock value (aie-rt LockValUpperBound).
    pub max: i8,
}

impl LockValueLayout {
    /// Sign-extend a raw register read to a signed 8-bit lock value.
    ///
    /// Identical formula on every arch; only the field width differs.
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

/// Per-arch lock behavior, consulted at construction / BD-parse
/// boundaries (never in a per-cycle hot path on today's AIE2-only
/// call sites).
pub trait LockModel: Send + Sync + core::fmt::Debug {
    /// Whether the arch supports the acquire-with-equality mode
    /// (lock.value == expected) in addition to the default
    /// acquire-with-greater-or-equal (lock.value >= expected).
    ///
    /// AIE2+: true (both modes). AIE1: false (single ACQ mode only).
    /// Evidence: AIE2 exposes per-lock event types for both ACQ_GE
    /// and ACQ_EQ at `xaiemlgbl_params.h:8048-8054` (offsets);
    /// AIE1's `xaie_locks_aie.c` has no such distinction.
    fn supports_acquire_eq(&self) -> bool;

    /// Whether the arch supports host-side `GetValue` / `SetValue`
    /// of the lock register.
    ///
    /// AIE2+: true. AIE1: false (returns `XAIE_FEATURE_NOT_SUPPORTED`
    /// at `xaie_locks_aie.c:169` for GetValue, `:195` for SetValue).
    fn supports_dynamic_value_ops(&self) -> bool;

    /// The value-field layout for this arch's locks.
    ///
    /// Returns a `'static` reference to a singleton layout so hot-path
    /// consumers can cache the pointer at construction.
    fn value_layout(&self) -> &'static LockValueLayout;
}
```

- [ ] **Step 2: Write the trait / carrier tests**

Create `crates/xdna-archspec/src/locks/tests.rs`:

```rust
//! Tests for the LockModel trait and LockValueLayout carrier.
//!
//! The AIE2 concrete impl tests (and the regdb drift-detection test)
//! live alongside the impl in `aie2/locks.rs`.

use super::LockValueLayout;

/// Fixture matching AIE2's Lock_value field (width=6, mask=0x3F,
/// sign_bit=5, min=-64, max=63). Duplicated here so the trait-side
/// tests are independent of the AIE2 impl.
fn aie2_layout() -> LockValueLayout {
    LockValueLayout {
        width: 6,
        mask: 0x3F,
        sign_bit: 5,
        min: -64,
        max: 63,
    }
}

#[test]
fn sign_extend_zero_is_zero() {
    assert_eq!(aie2_layout().sign_extend(0), 0);
}

#[test]
fn sign_extend_max_positive() {
    // 0x1F = 31 -- max positive for 6-bit signed is 31.
    assert_eq!(aie2_layout().sign_extend(31), 31);
}

#[test]
fn sign_extend_min_negative_for_field() {
    // 0x20 = 0b100000 -- sign bit set, all-zeros payload -> -32.
    assert_eq!(aie2_layout().sign_extend(0x20), -32);
}

#[test]
fn sign_extend_all_bits_set_is_minus_one() {
    // 0x3F = all 6 bits set -> -1.
    assert_eq!(aie2_layout().sign_extend(0x3F), -1);
}

#[test]
fn sign_extend_masks_extra_bits() {
    // 0xFF = all 8 bits set; upper bits outside mask must be ignored
    // before sign-extend.
    assert_eq!(aie2_layout().sign_extend(0xFF), -1);
}
```

- [ ] **Step 3: Declare the module in archspec's lib.rs**

Edit `crates/xdna-archspec/src/lib.rs`. Find the block of `pub mod` declarations (around lines 16-25). Add `pub mod locks;` preserving alphabetical order:

```rust
pub mod aie2;
pub mod device_model;
pub mod dma;
pub mod locks;           // NEW
pub mod model_builder;
pub mod regdb;
pub mod regdb_extractor;
pub mod runtime;
pub mod tablegen;
pub mod topology;
pub mod types;
```

- [ ] **Step 4: Write the Aie2LockModel impl**

Create `crates/xdna-archspec/src/aie2/locks.rs`:

```rust
//! AIE2 lock model implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). All AIE2-family devices share the same lock feature
//! set:
//!
//! - 6-bit register field (mask 0x3F), logical 7-bit signed range
//!   (-64..63); values outside the 6-bit field alias when read back.
//! - Both acquire-GE and acquire-EQ modes supported.
//! - Host-side GetValue / SetValue both supported.
//!
//! Width and mask sourced from
//! `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
//! (`memory.Lock0_value.Lock_value` field). Bounds from aie-rt
//! `xaiemlgbl_reginit.c:2452-2453` (LockValUpperBound=63,
//! LockValLowerBound=-64). A regdb drift-detection test in this
//! module asserts the JSON still agrees with these static constants.

use crate::locks::{LockModel, LockValueLayout};

/// The AIE2 Lock_value field layout.
///
/// Static so hot-path consumers can cache `&'static LockValueLayout`
/// at construction time.
pub static AIE2_LOCK_VALUE_LAYOUT: LockValueLayout = LockValueLayout {
    width: 6,
    mask: 0x3F,
    sign_bit: 5,
    min: -64,
    max: 63,
};

/// AIE2 lock model.
///
/// Zero-sized: a single `AIE2_LOCK_MODEL` static instance serves every
/// tile in every AIE2-family NPU. `ArchConfig::lock_model()` returns a
/// `&'static dyn LockModel` pointing at this singleton.
#[derive(Debug, Clone, Copy)]
pub struct Aie2LockModel;

/// The single `Aie2LockModel` instance used across every AIE2-family
/// consumer. Reference via `ArchConfig::lock_model()`.
pub static AIE2_LOCK_MODEL: Aie2LockModel = Aie2LockModel;

impl LockModel for Aie2LockModel {
    fn supports_acquire_eq(&self) -> bool {
        true
    }

    fn supports_dynamic_value_ops(&self) -> bool {
        true
    }

    fn value_layout(&self) -> &'static LockValueLayout {
        &AIE2_LOCK_VALUE_LAYOUT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regdb::RegisterDb;

    #[test]
    fn aie2_lock_model_feature_flags() {
        assert!(AIE2_LOCK_MODEL.supports_acquire_eq(),
                "AIE2 supports acq-EQ");
        assert!(AIE2_LOCK_MODEL.supports_dynamic_value_ops(),
                "AIE2 supports GetValue/SetValue");
    }

    #[test]
    fn aie2_lock_value_layout_constants() {
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.width, 6);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.mask, 0x3F);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.sign_bit, 5);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.min, -64);
        assert_eq!(AIE2_LOCK_VALUE_LAYOUT.max, 63);
    }

    #[test]
    fn aie2_lock_value_layout_value_layout_accessor() {
        let layout = AIE2_LOCK_MODEL.value_layout();
        assert_eq!(*layout, AIE2_LOCK_VALUE_LAYOUT);
    }

    /// Drift-detection: if the regdb JSON ever changes shape, this
    /// test fires. The static constants above carry the "known-good"
    /// AM025 + aie-rt values; this test asserts the JSON still agrees.
    #[test]
    fn aie2_lock_layout_matches_regdb() {
        let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json"));

        let json_path = match json_path {
            Some(p) if p.exists() => p,
            _ => {
                eprintln!("Skipping drift test: register DB JSON not found");
                return;
            }
        };

        let db = RegisterDb::from_file(&json_path)
            .expect("Failed to load register DB JSON");
        let field = db
            .module("memory")
            .and_then(|m| m.register("Lock0_value"))
            .and_then(|r| r.field("Lock_value"))
            .expect("memory.Lock0_value.Lock_value field not found in JSON");

        assert_eq!(field.width, AIE2_LOCK_VALUE_LAYOUT.width,
                   "Lock_value field width drifted from static constant");
        assert_eq!(field.mask, AIE2_LOCK_VALUE_LAYOUT.mask,
                   "Lock_value field mask drifted from static constant");
    }
}
```

- [ ] **Step 5: Declare the aie2/locks module**

Edit `crates/xdna-archspec/src/aie2/mod.rs`. Find the existing `pub mod dma;` declaration (around line 99). Add `pub mod locks;` after it:

```rust
/// AIE2-family DMA model (`Aie2DmaModel` + `AIE2_DMA_MODEL` static). Subsystem 3 seam.
pub mod dma;

/// AIE2-family lock model (`Aie2LockModel` + `AIE2_LOCK_MODEL` static). Subsystem 4 seam.
pub mod locks;
```

- [ ] **Step 6: Verify compilation + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -5
```

Expected: `273 + 9 = 282 passed; 0 failed; 2 ignored` (9 new tests: 5 in `locks/tests.rs` + 4 in `aie2/locks.rs`). The drift-detection test early-returns if the JSON path cannot be resolved (prints a skip message, counts as pass, not ignored).

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-archspec/src/locks/ \
        crates/xdna-archspec/src/aie2/locks.rs \
        crates/xdna-archspec/src/lib.rs \
        crates/xdna-archspec/src/aie2/mod.rs
git commit -m "$(cat <<'EOF'
feat(archspec): LockModel trait + LockValueLayout carrier + Aie2LockModel

Introduces the Subsystem 4 behavioral seam: a 3-method LockModel trait
(supports_acquire_eq, supports_dynamic_value_ops, value_layout) plus
the LockValueLayout data carrier (width, mask, sign_bit, min, max)
with its sign_extend method. AIE2 impl uses width=6 / mask=0x3F per
AM025's memory.Lock0_value.Lock_value, with bounds -64..63 per aie-rt
xaiemlgbl_reginit.c.

Tests cover the sign-extend formula (5 cases), the AIE2 feature
flags, the static-constant values, the value_layout() accessor, and
a regdb drift-detection test that asserts the JSON agrees with the
static constants.

No consumer uses LockModel yet; the accessor on ArchConfig and the
call-site migration follow in Tasks 3 and 4.

Generated using Claude Code.
EOF
)"
```

---

### Task 3: `ArchConfig::lock_model()` accessor

**Goal:** Add a `lock_model()` method to the `ArchConfig` trait and implement it for `ModelConfig`, following the exact pattern `dma_model()` established in Subsystem 3.

**Files:**
- Modify: `crates/xdna-archspec/src/runtime.rs` (add trait method + impl)

- [ ] **Step 1: Locate the existing dma_model() pattern**

```bash
grep -n 'fn dma_model' crates/xdna-archspec/src/runtime.rs
```

Expected: one match in the trait definition (around line 171), one match in the impl (around line 468). Open the file to line ~160 and line ~465 to see both blocks.

- [ ] **Step 2: Add the trait method**

In `crates/xdna-archspec/src/runtime.rs`, find the `dma_model` trait method (should look like):

```rust
    fn dma_model(&self) -> &'static dyn crate::dma::DmaModel;
}
```

Insert a `lock_model` method immediately before the closing `}`:

```rust
    fn dma_model(&self) -> &'static dyn crate::dma::DmaModel;

    // ========================================================================
    // Lock Model (Subsystem 4)
    // ========================================================================

    /// Return the lock feature-flag + value-layout model for this architecture.
    ///
    /// The returned reference is `'static` because every concrete `LockModel`
    /// impl is a zero-sized, stateless singleton (e.g. `AIE2_LOCK_MODEL`).
    /// Cold-path callers read the returned `value_layout()` when they need
    /// the mask / sign-extend formula; hot-path callers (none today)
    /// should cache `&'static LockValueLayout` at construction.
    fn lock_model(&self) -> &'static dyn crate::locks::LockModel;
}
```

- [ ] **Step 3: Add the ModelConfig impl**

Find the existing `fn dma_model(&self) -> ...` block in the `impl ArchConfig for ModelConfig`:

```rust
    fn dma_model(&self) -> &'static dyn crate::dma::DmaModel {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::dma::AIE2_DMA_MODEL
            }
            Architecture::Aie => {
                unimplemented!(
                    "AIE1 DmaModel not populated until AIE1 support lands \
                     (see docs/arch/dma-model.md for planned Aie1DmaModel)"
                )
            }
        }
    }
```

Add an analogous `lock_model` method immediately after:

```rust
    fn lock_model(&self) -> &'static dyn crate::locks::LockModel {
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

- [ ] **Step 4: Verify compilation + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-archspec 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -5
```

Expected: archspec tests still at ~282 pass (no new tests in this task). xdna-emu tests at 2687 pass (no change). Build clean for both.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/runtime.rs
git commit -m "$(cat <<'EOF'
feat(archspec): ArchConfig::lock_model() accessor

Adds the trait method and ModelConfig impl for reaching the
architecture-specific LockModel. Dispatches on ModelConfig::architecture
exactly like dma_model(): AIE2/AIE2p return &AIE2_LOCK_MODEL, AIE1 arm
is unimplemented! until that arch lands.

No consumer calls this yet; xdna-emu call sites migrate in Task 4.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Migrate xdna-emu call sites

**Goal:** Update the 6 xdna-emu sites that read `reg_layout.lock_value_mask` or call `sign_extend_lock_value` to instead read through `arch.lock_model().value_layout()`. The wrapper functions / fields remain on `src/device/regdb/mod.rs` through this task -- they go away in Task 5.

**Files:**
- Modify: `src/device/tile/registers.rs` (line 158 area; mask read)
- Modify: `src/device/state/compute.rs` (lines 14-52; mask + sign_extend)
- Modify: `src/device/state/mod.rs` (remove the `sign_extend_lock_value` wrapper fn)
- Modify: `docs/archive/accuracy-sweep/verified/locks.md:91`
- Modify: `docs/archive/accuracy-sweep/verified/shim-memtile.md:171`

- [ ] **Step 1: Audit the callers that thread `reg_layout`**

Check where `write_lock_value` / `mask_write_lock_value` are called from, and where `read_register_pure` sits in the call graph:

```bash
grep -n 'write_lock_value\|mask_write_lock_value' src/device/state/
grep -rn 'read_register_pure' src/
```

**Design decision for Task 4:** add a process-global accessor module (`src/device/arch_handle.rs`) that caches `&'static LockValueLayout` via `OnceLock`, seeded from `default_arch().lock_model().value_layout()`. Rationale:

- `read_register_pure` in `src/device/tile/registers.rs` is a `Tile` method; Tiles do not hold an `&dyn ArchConfig`, so threading `arch` through this path would ripple across many callers for a cold-path read.
- The pattern parallels the existing process-global `device_reg_layout()` accessor in `src/device/regdb/mod.rs` (`OnceLock<DeviceRegLayout>`), which is how xdna-emu already shares layout data without arch plumbing.
- The process-global becomes a focused refactor target when GUI runtime arch-switching lands (see the Future Direction section of the spec); until then, xdna-emu is AIE2-only and the cache is effectively a compile-time constant behind one indirection.

Every call site then reads `crate::device::arch_handle::lock_value_layout().mask` etc. `default_arch()` already exists in Subsystem 1 and is the seed.

- [ ] **Step 2: Create `src/device/arch_handle.rs`**

Create the new file:

```rust
//! Process-global cache of the architecture's lock model.
//!
//! Exposes the lock value layout as a fast accessor without forcing
//! every caller to hold a `&dyn ArchConfig` or a `Tile`. Lazy-initialized
//! at first call; the resolved `&'static LockValueLayout` stays cached
//! for the process lifetime.
//!
//! This is a pragmatic bridge for Subsystem 4: the GUI runtime arch-switch
//! direction flagged in the spec will eventually replace the
//! `default_arch()` seed with an explicit init at binary / arch selection
//! time. For now, xdna-emu is single-arch (AIE2), so the cache is
//! effectively a compile-time constant reached through one indirection.

use std::sync::OnceLock;
use xdna_archspec::locks::LockValueLayout;

static LOCK_VALUE_LAYOUT: OnceLock<&'static LockValueLayout> = OnceLock::new();

/// Get the process-wide lock value layout.
///
/// First call lazily resolves through `default_arch().lock_model()`;
/// subsequent calls return the cached pointer directly.
pub fn lock_value_layout() -> &'static LockValueLayout {
    LOCK_VALUE_LAYOUT.get_or_init(|| {
        xdna_archspec::runtime::default_arch().lock_model().value_layout()
    })
}
```

Declare it in `src/device/mod.rs`:

```bash
grep -n 'pub mod ' src/device/mod.rs | head -20
```

Find the appropriate place in the module declarations (alphabetical or by theme; Subsystem 3 added `pub mod regdb;` earlier). Insert:

```rust
pub mod arch_handle;
```

- [ ] **Step 3: Rewrite the three `.lock_value_mask` / `sign_extend_lock_value` call sites**

Open `src/device/tile/registers.rs`. Replace line 158:

Before:
```rust
                return self.locks[lock_id].value as u32 & reg_layout.lock_value_mask;
```

After:
```rust
                return self.locks[lock_id].value as u32 & crate::device::arch_handle::lock_value_layout().mask;
```

Open `src/device/state/compute.rs`. Replace:
- Line 25 (inside `write_lock_value`):
  ```rust
  // Before:
  let signed = sign_extend_lock_value(reg_layout, value);
  // After:
  let signed = crate::device::arch_handle::lock_value_layout().sign_extend(value);
  ```
- Line 47 (inside `mask_write_lock_value`):
  ```rust
  // Before:
  let current_raw = (tile.locks[lock_idx].value as u8 & reg_layout.lock_value_mask as u8) as u32;
  // After:
  let current_raw = (tile.locks[lock_idx].value as u8 & crate::device::arch_handle::lock_value_layout().mask as u8) as u32;
  ```
- Line 49 (same function):
  ```rust
  // Before:
  let signed = sign_extend_lock_value(reg_layout, new_raw);
  // After:
  let signed = crate::device::arch_handle::lock_value_layout().sign_extend(new_raw);
  ```

Since the call sites no longer use `reg_layout` for the lock path, you **may** be able to remove `reg_layout: &regdb::DeviceRegLayout` from the signature if nothing else uses it. Check both `write_lock_value` and `mask_write_lock_value` bodies:

- `write_lock_value`: the new body doesn't use `reg_layout` anywhere. **Delete the parameter and fix callers.**
- `mask_write_lock_value`: same. **Delete.**

Find callers:

```bash
grep -n 'write_lock_value\|mask_write_lock_value' src/device/state/
```

Update each call site to drop the `reg_layout` argument.

- [ ] **Step 4: Delete the `sign_extend_lock_value` wrapper fn in `state/mod.rs`**

Open `src/device/state/mod.rs`. Find lines 43-49:

```rust
/// Sign-extend a lock value from a register u32 to i8.
///
/// Delegates to DeviceRegLayout::sign_extend_lock_value() which derives the
/// field width from the AM025 register database (6 bits for AIE2).
fn sign_extend_lock_value(reg_layout: &regdb::DeviceRegLayout, raw: u32) -> i8 {
    reg_layout.sign_extend_lock_value(raw)
}
```

Delete this block. Also check whether `use super::regdb;` or similar is still needed after the deletion -- if no other code in the file uses `regdb`, remove the `use` too.

- [ ] **Step 5: Update archive docs**

Open `docs/archive/accuracy-sweep/verified/locks.md` at line 91:

```
`sign_extend_lock_value()` correctly handles this 6-bit field, deriving the
```

Replace with:

```
`LockValueLayout::sign_extend()` (on the archspec-side AIE2_LOCK_VALUE_LAYOUT) correctly handles this 6-bit field, deriving the
```

Open `docs/archive/accuracy-sweep/verified/shim-memtile.md` at line 171:

```
`sign_extend_lock_value()` derives field width from regdb (6 bits for AIE2).
```

Replace with:

```
`LockValueLayout::sign_extend()` operates on the archspec-side AIE2_LOCK_VALUE_LAYOUT (width=6 / mask=0x3F for AIE2).
```

- [ ] **Step 6: Verify compilation + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -20
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -5
```

Expected: build clean; `2687 passed; 0 failed; 5 ignored` (no new tests yet; existing ones must still pass).

- [ ] **Step 7: Commit**

```bash
git add src/device/arch_handle.rs \
        src/device/mod.rs \
        src/device/tile/registers.rs \
        src/device/state/compute.rs \
        src/device/state/mod.rs \
        docs/archive/accuracy-sweep/verified/locks.md \
        docs/archive/accuracy-sweep/verified/shim-memtile.md
git commit -m "$(cat <<'EOF'
refactor: migrate lock-value-mask / sign_extend call sites to LockModel

Six xdna-emu sites that previously reached into DeviceRegLayout's
lock_value_* fields now read through a process-global
arch_handle::lock_value_layout() cache of the archspec-side
&'static LockValueLayout.

- src/device/tile/registers.rs:158 -- MMIO lock register read uses
  .mask instead of reg_layout.lock_value_mask.
- src/device/state/compute.rs (2 fns) -- write_lock_value and
  mask_write_lock_value drop their reg_layout parameter; the lock
  path uses layout.sign_extend() / .mask directly.
- src/device/state/mod.rs -- the thin sign_extend_lock_value wrapper
  fn deletes.

The DeviceRegLayout wrapper in src/device/regdb/mod.rs still holds
the lock_value_* fields and sign_extend method for Task 5's collapse;
nothing references them from production code after this commit.

Generated using Claude Code.
EOF
)"
```

---

### Task 5: Collapse the `DeviceRegLayout` wrapper + migrate tests

**Goal:** Delete the xdna-emu-side `DeviceRegLayout` wrapper struct (fields, `Deref`, `from_regdb`, `load_for_device` method, `sign_extend_lock_value` method). `src/device/regdb/mod.rs` shrinks to a re-export module + `OnceLock` accessor + config-aware `load_for_device()` function. Migrate the lock-related tests from `src/device/regdb/tests.rs` to archspec.

**Files:**
- Modify: `src/device/regdb/mod.rs` (significant shrink)
- Modify: `src/device/regdb/tests.rs` (remove migrated tests)

- [ ] **Step 1: Rewrite `src/device/regdb/mod.rs`**

Replace the file contents with the collapsed form:

```rust
//! Register database accessor.
//!
//! After Subsystem 4, DeviceRegLayout lives entirely in xdna_archspec.
//! This module retains only:
//!
//! - Re-exports of the archspec types (so every
//!   `crate::device::regdb::DeviceRegLayout` etc. keeps working).
//! - The `OnceLock`-backed global accessor `device_reg_layout()`.
//! - The config-aware `load_for_device()` loader (xdna-emu owns
//!   `Config` for path resolution).

#[cfg(test)]
mod tests;

pub use xdna_archspec::regdb::*;
pub use xdna_archspec::dma::DeviceRegLayout;
pub use xdna_archspec::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout,
    MemTileBdFieldLayout, ShimBdFieldLayout,
    ShimMuxField, ShimMuxLayout,
    StreamSwitchLayout, ModuleEventLayout,
};

use std::sync::OnceLock;

static DEVICE_REG_LAYOUT: OnceLock<DeviceRegLayout> = OnceLock::new();

/// Get the global register layout, loading from JSON on first access.
///
/// # Panics
///
/// Panics if the register database JSON file cannot be loaded. This
/// requires mlir-aie to be installed and `MLIR_AIE_PATH` configured.
pub fn device_reg_layout() -> &'static DeviceRegLayout {
    DEVICE_REG_LAYOUT.get_or_init(|| {
        load_for_device("aie2").unwrap_or_else(|e| {
            panic!(
                "Failed to load register database: {}.\n\
                 The register database JSON (aie_registers_aie2.json) is required.\n\
                 Ensure mlir-aie is installed and MLIR_AIE_PATH is set.\n\
                 See CLAUDE.md for environment setup instructions.",
                e
            )
        })
    })
}

/// Load a DeviceRegLayout from the mlir-aie install using the emulator's
/// `Config` for path resolution.
pub fn load_for_device(device: &str) -> Result<DeviceRegLayout, String> {
    let config = crate::config::Config::get();
    let json_path = config.mlir_aie_subpath(
        &format!("lib/Dialect/AIE/Util/aie_registers_{}.json", device)
    );
    let db = RegisterDb::from_file(&json_path)?;
    DeviceRegLayout::from_regdb(db)
}
```

Note: the `raw_regdb_load_for_device` helper that used to return `RegisterDb` (not `DeviceRegLayout`) may still be referenced from elsewhere. Check:

```bash
grep -rn 'regdb::load_for_device' src/ crates/
```

If the old `load_for_device` (returning `RegisterDb`) was referenced, either:
- Keep a second function with a distinct name (e.g., `load_raw_regdb_for_device`), or
- Update the one caller to construct a `DeviceRegLayout` via the new signature.

**Expected:** only `src/device/regdb/mod.rs` self-calls and tests reference `load_for_device`. The accessor is the main API.

- [ ] **Step 2: Trim `src/device/regdb/tests.rs`**

Open `src/device/regdb/tests.rs`. Delete `test_device_reg_layout_lock_value_extension` (lines 17-38 in the current file -- the entire function including its doc comment if any). The 5 field/sign-extend assertions there are already covered by archspec-side tests added in Task 2.

In `test_load_for_device_uses_config`, delete the line:

```rust
            assert_eq!(layout.lock_value_width, 6);
```

(The surrounding `match result { Ok(layout) => { ... } Err(e) => { ... } }` structure stays; only this one assertion deletes. The `memory_bd_base` assertion above it remains as the "load succeeded and returned a sane layout" check.)

Update the module doc-comment at the top to reflect the new scope:

```rust
//! Tests for the xdna-emu register database accessor.
//!
//! The `DeviceRegLayout` struct itself lives in `xdna_archspec::dma`.
//! Only xdna-emu-specific integration with `crate::config::Config`
//! is tested here.
```

- [ ] **Step 3: Verify compilation + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -20
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -5
```

Expected: `2687 - 1 = 2686 passed; 0 failed; 5 ignored` (one xdna-emu test deleted; archspec picked up equivalents in Task 2 so the archspec count went up).

If the build fails with "method sign_extend_lock_value not found" or "field lock_value_mask not found", Task 4 missed a call site. Grep for the old API:

```bash
grep -rn 'sign_extend_lock_value\|lock_value_mask\|lock_value_width\|lock_value_sign_bit' src/
```

Expected: all matches are either in comments / doc-strings, or in the (now-deleted) places. If any production code still references them, fix before commit.

- [ ] **Step 4: Commit**

```bash
git add src/device/regdb/mod.rs src/device/regdb/tests.rs
git commit -m "$(cat <<'EOF'
refactor: collapse xdna-emu DeviceRegLayout wrapper

The lock_value_width / lock_value_mask / lock_value_sign_bit fields
and the sign_extend_lock_value method migrate to archspec's
LockValueLayout (Task 2). The xdna-emu wrapper struct, its Deref
impl, and the from_regdb / load_for_device method-form constructors
delete.

src/device/regdb/mod.rs shrinks from ~138 LOC to ~60 LOC: a
re-export of xdna_archspec::dma::DeviceRegLayout plus the
xdna-emu-side OnceLock accessor and config-aware load_for_device
function (which Config is an xdna-emu concern).

Test migration: test_device_reg_layout_lock_value_extension deletes
(its 5 assertions are covered by archspec-side tests);
test_load_for_device_uses_config drops a single lock_value_width
line and keeps its memory_bd_base sanity check.

Generated using Claude Code.
EOF
)"
```

---

### Task 6: Drift-detection test for `Lock::MIN_VALUE` / `MAX_VALUE`

**Goal:** Add a test in `src/device/tile/locks.rs` that asserts the Lock struct's hardcoded `MIN_VALUE` / `MAX_VALUE` constants match the archspec-side `AIE2_LOCK_VALUE_LAYOUT.min` / `.max`. This locks the two together so a future AIE1 migration of `Lock` per-arch is a focused, single-commit change.

**Files:**
- Modify: `src/device/tile/locks.rs` (tests module)

- [ ] **Step 1: Locate or add the tests module**

Open `src/device/tile/locks.rs`. Find the existing `#[cfg(test)] mod tests { ... }` at the end of the file.

If one does not exist, add it at the very bottom of the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
}
```

- [ ] **Step 2: Add the drift-detection test**

Add this test inside the `mod tests` block:

```rust
    #[test]
    fn lock_bounds_match_archspec() {
        use xdna_archspec::aie2::locks::AIE2_LOCK_VALUE_LAYOUT;
        assert_eq!(Lock::MIN_VALUE, AIE2_LOCK_VALUE_LAYOUT.min,
                   "Lock::MIN_VALUE must match AIE2_LOCK_VALUE_LAYOUT.min; \
                    update the constant (or split per-arch) when AIE1 lands");
        assert_eq!(Lock::MAX_VALUE, AIE2_LOCK_VALUE_LAYOUT.max,
                   "Lock::MAX_VALUE must match AIE2_LOCK_VALUE_LAYOUT.max");
    }
```

- [ ] **Step 3: Verify the test passes**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib device::tile::locks::tests::lock_bounds_match_archspec 2>&1 | tail -5
```

Expected: `1 passed`. If it fails, either:
- `Lock::MIN_VALUE` / `MAX_VALUE` differ from `-64` / `63` -- check `src/device/tile/locks.rs` and `crates/xdna-archspec/src/aie2/locks.rs`.
- The `use` path is wrong -- verify the accessor name matches Task 2's static.

- [ ] **Step 4: Full test suite**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -5
```

Expected: `2687 passed; 0 failed; 5 ignored` (previous 2686 + 1 new drift test).

- [ ] **Step 5: Commit**

```bash
git add src/device/tile/locks.rs
git commit -m "$(cat <<'EOF'
test: lock bounds drift-detection against archspec LockValueLayout

Asserts Lock::MIN_VALUE / MAX_VALUE agree with
xdna_archspec::aie2::locks::AIE2_LOCK_VALUE_LAYOUT.min / .max.

The two must match for the AIE2 path; they are allowed to differ
per-arch in the future (AIE1 has binary-semaphore bounds), at which
point this test moves to a per-arch Lock specialization. The test
catches any accidental drift until that refactor.

Generated using Claude Code.
EOF
)"
```

---

### Task 7: Gate + tag

**Goal:** Final verification (unit + bridge smoke + full HW bridge + ISA), fill in the audit and design-note Completion sections, update `NEXT-STEPS.md`, commit, and tag `phase1-subsys-locks`.

**Files:**
- Modify: `docs/arch/subsys4-audit.md` (Completion section)
- Modify: `docs/arch/lock-model.md` (Completion section)
- Modify: `NEXT-STEPS.md`

- [ ] **Step 1: Unit test gate**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build --release 2>&1 | tail -5
```

Expected:
- xdna-emu: `2687 passed; 0 failed; 5 ignored`
- archspec: `~282 passed; 0 failed; 2 ignored`
- Release build: clean.

Record the exact counts for the audit Completion section.

- [ ] **Step 2: Rebuild FFI cdylib**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -3
```

Expected: build clean. This is critical before the bridge runs -- a stale `.so` produces phantom "regressions."

- [ ] **Step 3: Bridge smoke test**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -15
```

Expected: Chess PASS, Peano PASS.

- [ ] **Step 4: Full HW bridge run**

```bash
mkdir -p /tmp/claude-1000
nice -n 19 ./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-subsys4.log
```

Expected duration: ~20-30 minutes. Expected outcome: matches the `phase1-subsys-dma` baseline (Chess 63 pass / 1 pre-existing fail, Peano 51 pass / 3 pre-existing fail, HW 63+53 pass). Known pre-existing failure `bd_chain_repeat_on_memtile` still fails (EMU deadlock, pre-refactor).

Check the tail for the pass/fail summary:

```bash
tail -30 /tmp/claude-1000/bridge-subsys4.log
```

- [ ] **Step 5: ISA test suite**

```bash
nice -n 19 ./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys4.log
```

Expected duration: ~10 minutes. Expected: `FAIL: 0`. Note: bridge and ISA must run sequentially (never concurrent HW; see CLAUDE.md).

```bash
tail -10 /tmp/claude-1000/isa-subsys4.log
```

- [ ] **Step 6: Success-criteria sweep**

```bash
# Criterion 1: no stale references to the deleted xdna-emu fields
grep -rn 'lock_value_width\|lock_value_mask\|lock_value_sign_bit' src/ crates/xdna-archspec/src/ 2>/dev/null \
  | grep -v 'archspec/src/aie2/locks.rs\|archspec/src/locks/\|docs/\|^Binary'

# Expected: empty (all references should be in archspec's locks module or docs).

# Criterion 2: no calls to the deleted wrapper method
grep -rn 'sign_extend_lock_value' src/ crates/ 2>/dev/null | grep -v 'docs/'

# Expected: empty (the method is gone; LockValueLayout::sign_extend replaces it).

# Criterion 3: the xdna-emu DeviceRegLayout wrapper struct is gone
grep -n 'pub struct DeviceRegLayout' src/device/regdb/mod.rs

# Expected: empty (only a re-export remains).

# Criterion 4: archspec has the new module
ls crates/xdna-archspec/src/locks/
ls crates/xdna-archspec/src/aie2/locks.rs

# Expected: both exist.
```

If any criterion fails, a prior task missed cleanup. Find and fix before proceeding.

- [ ] **Step 7: Fill in `docs/arch/subsys4-audit.md` Completion section**

Replace the `*(To be filled in by Task 7.)*` line with:

```markdown
## Completion

Landed 2026-MM-DD. Tag: `phase1-subsys-locks`.

### Commits (Task 1 through tag)

<output of `git log --oneline phase1-subsys-dma..HEAD`>

### Verification (at tag)

- `cargo test --lib`: <final count> passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: <final count> passed; 0 failed; 2 ignored.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: matches phase1-subsys-dma baseline; `bd_chain_repeat_on_memtile` still fails (pre-existing EMU deadlock).
- ISA test suite: FAIL: 0.

### Success criteria sweep

- `LockModel` trait in `xdna_archspec::locks` (3 methods): populated.
- `LockValueLayout` data carrier + `sign_extend()` method: populated.
- `Aie2LockModel` concrete impl + `AIE2_LOCK_MODEL` + `AIE2_LOCK_VALUE_LAYOUT` statics: populated.
- `ArchConfig::lock_model()` accessor: populated.
- xdna-emu wrapper `DeviceRegLayout` struct: **collapsed** (Deref, lock fields, sign_extend_lock_value, from_regdb / load_for_device methods all deleted).
- 6 xdna-emu call sites migrated through `arch_handle::lock_value_layout()`: done.
- `sign_extend_lock_value` wrapper fn in `state/mod.rs`: deleted.
- `Lock::MIN_VALUE` / `MAX_VALUE` drift-detection test against archspec: added.
- `docs/arch/lock-model.md` design note: written.

### Net code delta

- New in archspec: ~180 LOC (LockModel trait + LockValueLayout + Aie2LockModel + tests + drift test).
- Deleted in xdna-emu: ~80 LOC (wrapper struct, sign_extend method, Deref, sign_extend_lock_value wrapper fn, migrated tests).
- Modified in xdna-emu: 6 call sites rewrites (~20 LOC touched), new `arch_handle` module (~30 LOC).
- Net workspace LOC change: +~150 LOC (mostly new trait + tests).

### Follow-ups flagged

Follow-ups that fit naturally in later work, NOT blocking:

- **AIE1 plug-in:** `Aie1LockModel` fills in when AIE1 support starts. At that point, `Lock::MIN_VALUE` / `MAX_VALUE` specialize per-arch (either via a type parameter on `Lock` or via separate `Aie1Lock` / `Aie2Lock` structs). The drift-detection test added here moves along.
- **Generic-type-parameter monomorphization:** post-seam-pass optimization direction (per the spec's "Future direction"). Hot types that today reach `&'static dyn LockModel` / `&'static dyn DmaModel` switch to `<L: LockModel>` / `<D: DmaModel>`.
- **`arch_handle` module generalization:** currently exposes only `lock_value_layout()`. If Subsystem 5 (Stream Switch) or later needs a similar process-global handle, extend `arch_handle` rather than multiplying modules.
- **Subsystem 5 (Stream Switch):** see NEXT-STEPS pickup guide.
```

Then:

- Replace `<final count>` with the exact numbers from Step 1.
- Replace `<output of git log --oneline phase1-subsys-dma..HEAD>` with the actual commit list:

```bash
git log --oneline phase1-subsys-dma..HEAD
```

- Replace `2026-MM-DD` with today's date (`date +%Y-%m-%d`).

- [ ] **Step 8: Fill in `docs/arch/lock-model.md` Completion section**

Replace the `*(To be filled in by Task 7...)*` line with:

```markdown
## Completion (2026-MM-DD)

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

Verification: `cargo test --lib` = <final count>, archspec =
<final count>, full bridge = phase1-subsys-dma baseline, ISA = 0 fail.
```

Fill in the numbers and date.

- [ ] **Step 9: Update NEXT-STEPS.md**

Edit `NEXT-STEPS.md`. Update the header:

```markdown
**Last updated:** 2026-MM-DD (Phase 1b Subsystem 4 landed; Subsystem 5 up next)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-locks` (Subsystem 4 completion)
```

Update the subsystem status table rows for 4 and 5:

```markdown
| 4 | Locks | `phase1-subsys-locks` | **Done** | Small seam: 3-method LockModel trait (supports_acquire_eq, supports_dynamic_value_ops, value_layout) + LockValueLayout carrier + Aie2LockModel. Migrated `sign_extend_lock_value` + lock_value_* fields from xdna-emu's DeviceRegLayout wrapper to archspec; wrapper collapsed. See docs/arch/lock-model.md. |
| 5 | Stream Switch | `phase1-subsys-stream-switch` | **Up next** | Topology (data, already via archspec) + routing legality (behavior). Likely `StreamSwitchModel` trait. Packet ID support (AIE2) vs circuit-only (AIE1) is a candidate feature flag. |
```

Replace the `## How to Pick Up Subsystem 4 (Locks)` section with a new `## How to Pick Up Subsystem 5 (Stream Switch)` section:

```markdown
## How to Pick Up Subsystem 5 (Stream Switch)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/lock-model.md` -- Subsystem 4 completion; and the
     per-seam design note template.
   - `docs/arch/dma-model.md` -- alternate template if the stream
     switch seam ends up closer in shape to Subsystem 3 than 4.
   - Skim `src/device/stream_switch/` and `src/device/port_layout.rs`
     for the current xdna-emu implementation.

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1-subsys-locks..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Subsystem 4 left off.

   ```bash
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
   ```
   Expect xdna-emu `<final count> passed; 0 failed; 5 ignored` and archspec
   `<final count> passed; 0 failed; 2 ignored`.

3. **Invoke brainstorming** to shape Subsystem 5's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 5: Stream Switch."

   **Shape the spec around these questions:**
   - Topology vs routing legality: topology is already data-driven
     via archspec (port counts per bundle, port type assignments).
     What behavior genuinely diverges per-arch?
   - Packet vs circuit switching: AIE2 has packet IDs for multi-source
     routing; AIE1 is typically circuit-only. Is this a clean feature
     flag (`supports_packet_routing`) or does it pervade the FSM?
   - Shim mux configuration: does the port-mapping to NoC / AXI
     differ between AIE1 and AIE2? Does the memtile-less AIE1 change
     the north-side routing constraints?
   - `PortLayout` extension trait in xdna-emu: Phase 1a left this
     runtime-side because its data came from build.rs consts that
     Subsystem 1 migrated to archspec. Does `PortLayout` now move
     behind `StreamSwitchModel::port_layout()` or stay where it is?

4. **Invoke writing-plans** to produce a plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys5-stream-switch.md`.

5. **Invoke subagent-driven-development** to execute.

6. **At end of Subsystem 5:** tag `phase1-subsys-stream-switch`, append a
   completion section to its audit, update this `NEXT-STEPS.md` to
   move Subsystem 7 (ISA Execute) to "up next".
```

Fill in `<final count>` with the actual numbers.

- [ ] **Step 10: Commit docs updates**

```bash
git add docs/arch/subsys4-audit.md docs/arch/lock-model.md NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 4 completion log + NEXT-STEPS points at Subsystem 5

Audit Completion section filled: final test counts, success-criteria
sweep, net code delta, and Subsystem 5 / Phase 2 follow-up flags.
lock-model design note's Completion section filled with the final
trait-surface summary.  NEXT-STEPS pickup guide rewrites to point at
Subsystem 5 (Stream Switch) with the "what would AIE1 look like?"
questions pre-shaped.

Generated using Claude Code.
EOF
)"
```

- [ ] **Step 11: Tag**

```bash
git tag phase1-subsys-locks -m "Phase 1b Subsystem 4: LockModel trait + Aie2LockModel + wrapper collapse"
```

Verify the tag:

```bash
git log --oneline phase1-subsys-locks -5
```

- [ ] **Step 12: Final sanity pass**

```bash
# Confirm we're at the expected commit & tag
git log --oneline -10

# Confirm no stray changes
git status

# Confirm the success-criteria greps stay clean
grep -rn 'lock_value_width\|lock_value_mask\|lock_value_sign_bit' src/ crates/xdna-archspec/src/ 2>/dev/null \
  | grep -v 'archspec/src/aie2/locks.rs\|archspec/src/locks/\|docs/\|^Binary'
grep -rn 'sign_extend_lock_value' src/ crates/ 2>/dev/null | grep -v 'docs/'
grep -n 'pub struct DeviceRegLayout' src/device/regdb/mod.rs
```

Expected: clean status; greps all empty.

---

## Appendix A: Rollback procedure (per-task)

If any task breaks compilation or tests in a way that can't be fixed in-place within 15 minutes:

```bash
# Identify the last good commit
git log --oneline

# Revert to the last good commit (soft reset keeps work on disk)
git reset --soft HEAD~1

# Or hard reset if the changes are fully known-bad
git reset --hard <last-good-sha>
```

Tasks 2-6 commit independently and each pass tests at the commit boundary. Task 4 (the call-site migration) is the biggest single-task risk; if it fails partway through, keep the temporary `arch_handle` module in place and let the wrapper stay in `src/device/regdb/mod.rs` -- the pre-Task-4 state is still functional. Task 5 only runs after Task 4 successfully compiles.

---

## Appendix B: Known risks

1. **`read_register_pure` call-site / `write_lock_value` caller threading (Task 4, Step 1/3).** The plan chose to add a process-global `arch_handle::lock_value_layout()` to avoid threading `&dyn ArchConfig` through every tile-read path. If the review prefers explicit threading, Task 4 can be rewritten: add `arch: &dyn ArchConfig` to `read_register_pure` / `write_lock_value` signatures and update callers. The process-global is the pragmatic choice for Subsystem 4; it becomes a focused refactor when GUI runtime arch-switch lands.

2. **`default_arch()` initialization at first access (Task 4, Step 2).** The `arch_handle::lock_value_layout()` initializer calls `default_arch().lock_model()`. If `default_arch()` has been called before the first lock access, cached state may have drifted. Verify: `default_arch()` is already called eagerly during `DeviceState::new_npu1()` in tests and at startup; the lazy `OnceLock` in `arch_handle` is strictly a caching optimization atop it.

3. **`RegisterDb::module().register().field()` API shape (Task 2, Step 4 drift test).** The archspec-side regdb helper path is what the Subsystem 3 tests use. If the method chain has a different name (e.g., `find_register` / `find_field`), the drift test fails to compile. Cross-reference `crates/xdna-archspec/src/dma/layouts.rs::from_regdb()` for the correct API, and update the test if needed.

4. **`LockValueLayout` PartialEq on static comparison (Task 2, Step 2 and aie2/locks.rs drift test).** The derive `#[derive(PartialEq, Eq)]` must match the field types. All five fields (`u8`, `u32`, `u8`, `i8`, `i8`) implement `Eq` so this is safe. If a future field is added (e.g., an `f32`), `Eq` breaks.

5. **Compilation ordering (Task 3 before Task 4).** Task 4 call sites depend on `arch.lock_model()` existing on `ArchConfig`. Task 3 must land first. If a reviewer reorders these, Task 4 fails to compile.

6. **Test migration boundary (Task 5).** The five lock-value-extension assertions moved to archspec in Task 2 (as separate test functions). If the archspec tests were not added, Task 5 deletes the coverage. Verify the archspec test count went up by at least 5 before deleting the xdna-emu test.

7. **`src/device/state/mod.rs` `use super::regdb;` handling (Task 4, Step 4).** The `regdb` import may still be needed for other type references in `state/mod.rs`. Check before removing.
