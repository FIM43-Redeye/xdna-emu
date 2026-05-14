# Subsystem 4 -- Locks Audit

## Baseline (pre-subsystem, at phase1-subsys-dma tag)

- `cargo test --lib`: test result: ok. 2687 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 2.18s
- `cargo test -p xdna-archspec --lib`: test result: ok. 273 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.35s

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

Landed 2026-04-21. Tag: `phase1-subsys-locks`.

### Commits (Task 1 through tag)

```
8fcdaf5 test: lock bounds drift-detection against archspec LockValueLayout
fbf9347 refactor: collapse xdna-emu DeviceRegLayout wrapper
e044a34 refactor: migrate lock-value-mask / sign_extend call sites to LockModel
b246803 feat(archspec): ArchConfig::lock_model() accessor
7b7d1dd feat(archspec): LockModel trait + LockValueLayout carrier + Aie2LockModel
ce50758 docs: Subsystem 4 audit + lock-model design note scaffolds
7595baf docs: Subsystem 4 (Locks) implementation plan
8166ae0 docs: Subsystem 4 (Locks) design spec
```

### Verification (at tag)

- `cargo test --lib`: 2687 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 282 passed; 0 failed; 2 ignored.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: Chess 62/62 compiled, HW 63 pass / 1 fail; Peano 55/55 compiled, HW 53 pass / 1 fail / 1 XFAIL. Matches phase1-subsys-dma character -- the single HW failure on each compiler is the pre-existing `bd_chain_repeat_on_memtile` EMU deadlock documented in NEXT-STEPS.md. Bridge-side timeouts (11 Chess, 13 Peano) are the pre-existing EMU latency issue, not regressions.
- ISA test suite: 4815/4815 PASS (100.0%); FAIL: 0.

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
- **Phase 2 hygiene carried through from Task 2/4 code-review minor flags:**
  - `OnceLock<&'static LockValueLayout>` double-indirection in `src/device/arch_handle.rs` could be simplified to `OnceLock<LockValueLayout>` (since LockValueLayout is Copy) to store by value.
  - `src/device/tile/registers.rs:158` uses the full `crate::device::arch_handle::lock_value_layout()` path; a module-level `use` import would be more idiomatic.
  - Pre-existing dead-code warnings in `crates/xdna-archspec/src/{regdb_extractor,tablegen,types,regdb}.rs` plus stale imports/errors in `tests/arch_constants.rs`, `examples/run_add_test.rs`, `examples/bdd_validate.rs`, `decoder.rs:1425` (Subsystem 6 rot; not Subsystem 4's scope).
- **Subsystem 5 (Stream Switch):** see NEXT-STEPS pickup guide.
