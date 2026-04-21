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

*(To be filled in by Task 7.)*
