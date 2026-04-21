# Subsystem 3 -- DMA Engine & BD Format Audit

## Baseline (pre-subsystem, at phase1-subsys-tile-topo tag)

- `cargo test --lib`: test result: ok. 2708 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 2.20s
- `cargo test -p xdna-archspec --lib`: test result: FAILED. 236 passed; 1 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.26s

Known pre-existing failures (carry through):
- `test_full_parse_all_devices` (archspec, pre-existing, device count 13 vs expected 12).
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit facts

### DMA module LOC inventory
Total: 12,121 LOC across `src/device/dma/`. Hotspots:
- `engine/stepping.rs` 1517
- `bd.rs` 1178
- `engine/tests.rs` 1675
- `fifo.rs` 1007
- `addressing.rs` 1006
- `token.rs` 864
- `engine/mod.rs` 672
- `transfer/core.rs` 594
- `engine/status.rs` 154
- `engine/task_queue_ops.rs` 161
- `engine/stream_io.rs` 226

### regdb migration inventory (Task 3)
Files moving wholesale into archspec:
- `src/device/regdb/mod.rs` 464 LOC (minus `sign_extend_lock_value` + `lock_value_*` fields, pinned to xdna-emu until Subsystem 4).
- `src/device/regdb/field_layouts.rs` 683 LOC.
- `src/device/regdb/tests.rs` 842 LOC (migration target: archspec-side tests).

Consumer files that import from `crate::device::regdb` (20 files):
src/device/tile/mod.rs, src/device/state/memtile.rs, src/device/state/mod.rs,
src/npu/executor.rs, src/device/state/effects.rs, src/device/state/compute.rs,
src/device/dma/engine/tests.rs, src/device/dma/engine/mod.rs,
src/device/dma/bd.rs, src/device/tile/tests.rs, src/device/tile/registers.rs,
src/interpreter/execute/memory/mod.rs, src/device/state/dispatch.rs,
src/device/registers.rs, src/device/regdb/tests.rs,
src/device/regdb/mod.rs, src/device/aiert_validation.rs,
src/device/tile/dma_legacy.rs, src/device/state/tests.rs,
src/device/regdb/field_layouts.rs.

### DmaEngine construction targets (Task 5)
- Production constructor: `src/device/dma/engine/mod.rs:147` `DmaEngine::new(col, row, tile_kind, s2mm_channels, mm2s_channels, num_bds, num_locks)`.
- Sole production caller: `src/device/array/mod.rs:201`.
- Test-only helpers: `src/device/dma/engine/mod.rs:188` `new_compute_tile`, `:194` `new_mem_tile`, `:200` `new_shim_tile` (all `#[cfg(test)]`, all call through to `new()`).
- DmaTimingConfig::from_arch() call site: `src/device/dma/engine/mod.rs:163`.

### Feature-gated call sites (Task 6)
- `src/device/dma/engine/task_queue_ops.rs` -- entire module is AIE2-only (task queue).
  - `enqueue_task` line 12 (public; gate at entry, return false if !supports_task_queue).
  - `start_next_queued_task` line 51 (private pub(super); guarded implicitly by enqueue_task gate).
  - `task_queue_size` line 96 (public; returns 0 if !supports_task_queue).
  - `task_queue_overflow` line 104 (public; returns false).
  - `clear_task_queue_overflow` line 112 (public; no-op).
- `src/device/dma/engine/stepping.rs:346-351` -- task queue pop on task complete (wrap the `if !task_queue.is_empty()` check).
- `src/device/dma/engine/status.rs:33-39` -- Task_Queue_Size + Task_Queue_Overflow status bits emitted (wrap both insert/set_bit calls).
- `src/device/dma/engine/status.rs:104-110` -- `is_out_of_order_enabled` short-circuits to false when !supports_ooo_mode.
- `src/device/dma/engine/status.rs:136-153` -- `set_channel_compression_config` drops compression flags if !supports_compression (the entry point that the compression paths at stepping.rs:902, 1119, 1247, 1398 transitively depend on).

### Hygiene items (Task 7)
1. `crates/xdna-archspec/src/runtime.rs:278` -- silent `(2, 2)` DMA channel fallback (Phase 1a follow-up).
2. `src/device/dma/compression.rs:39, 78, 113, 129` -- silent `None` / `0` / `false` on length mismatch (4 sites).
3. `src/device/dma/addressing.rs:217-227` -- magic `63 / 31 / 15` padding bit-limits.
4. `src/device/dma/timing.rs:103` -- uncited `"AIE2: 128-bit bus = 4 words/cycle"` comment.
5. `src/device/dma/token.rs:93-100` -- self-contradictory `MAX_TASK_QUEUE_DEPTH` comment.
6. `src/device/dma/engine/stream_io.rs:44, 49, 70, 75, 112` -- magic `256` stream buffer capacity (5 sites for one concept).
7. `src/device/dma/engine/mod.rs:409-419` -- MemTile BD-channel validity is warn-only; AM025 says this is a hard invariant -- upgrade to error.

### aie-rt AIE1 vs AIE2 divergence (evidence base for trait)
Confirmed at `../aie-rt/driver/src/dma/`:
- Task queue: AIE2 only. `xaie_dma_aieml.c:1257-1279` (`_XAieMl_DmaWaitForBdTaskQueue`); absent in AIE1.
- Out-of-order BD ID: AIE2 only. `xaie_dma_aieml.c:313-315, 459-461`.
- Compression: AIE2 only. `xaie_dma_aieml.c:360-362, 514-516`.
- BD iteration: AIE2 only. `xaie_dma_aieml.c:1065-1074` (AIE1 stub returns `NOT_SUPPORTED`).
- Independent lock IDs: AIE2 only. `xaie_dma_aie.c:113-116` enforces `Acq.LockId == Rel.LockId`.
- Interleave + double-buffer: AIE1 only. `xaie_dma_aie.c:189-198, 477-500, 543-545`.
- Tile BD word count: AIE1=7, AIE2=6.
- Shim BD word count: AIE1=5, AIE2=8.
- Max tensor dims: AIE1=2, AIE2 compute=3, AIE2 memtile=4.

## Completion

Landed 2026-04-21. Tag: `phase1-subsys-dma`.

### Commits (Task 1 through tag)

```
82e11bf test(archspec): fix test_full_parse_all_devices device count
f57dbe5 refactor: apply six hygiene items + two code-review fold-ins
2d6838f refactor: gate AIE2-only DMA features on DmaModel feature flags
d420fb7 refactor: thread &'static dyn DmaModel through DmaEngine::new
9b9f2f9 refactor(archspec): replace silent DMA (2,2) fallback with expect()
48181f9 refactor: migrate DeviceRegLayout family from xdna-emu to archspec
9fd16a0 refactor(archspec): add DmaModel trait + Aie2DmaModel impl
3421192 docs: Subsystem 3 audit + dma-model design note scaffolds
```

(Pre-Task-1 spec/plan docs at `8f31641`, `35784f9`, `981a88d` are
ancestors; `fe2c08e` is Subsystem 2 polish that preceded this work.)

### Verification (at tag)

- `cargo test --lib`: **2687 passed; 0 failed; 5 ignored**. Up from
  2686 at Task 7 due to one new test (`memtile_invalid_bd_channel
  _combination_returns_error`) covering the hard-error upgrade.
- `cargo test -p xdna-archspec --lib`: **273 passed; 0 failed; 2 ignored**.
  Up from 236 at Subsystem 2 tag (+13 DmaModel tests from Task 2, +24
  migrated DeviceRegLayout tests from Task 3). **The previously
  pre-existing `test_full_parse_all_devices` failure was fixed in
  Task 8** (device count expectation 12 → 13 to match the JSON's
  actual device list including `xcve2802`) — archspec has a clean
  baseline for the first time since the refactor began.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: matches `phase1-subsys-tile-topo` baseline exactly.
  - Chess: 62/62 compiled, 63 bridge pass, 1 bridge fail; HW 63 pass,
    1 fail (the pre-existing `bd_chain_repeat_on_memtile` EMU
    deadlock).
  - Peano: 55/55 compiled, 51 bridge pass, 3 bridge fail (2 pre-existing
    timeouts: `dma_task_large_linear`, `objectfifo_repeat/init_values_
    repeat`; 1 XFAIL `objectfifo_repeat/distribute_repeat`); HW 53
    pass, 1 fail, 0 skip (plus the 1 XFAIL).
  - All failures pre-existing; no new regressions introduced by
    Subsystem 3.
- ISA test suite: **4815 / 4815 PASS (100.0%). FAIL: 0.** Matches
  the Subsystem 2 baseline exactly.

### Success criteria sweep

- `DmaModel` trait in `xdna_archspec::dma` with 9 methods: populated.
- `Aie2DmaModel` concrete impl + `AIE2_DMA_MODEL` static singleton:
  populated.
- `ArchModel::dma_model()` accessor (added to `ArchConfig` trait per
  Task 5's justified deviation): populated, returns `&'static dyn
  DmaModel`.
- `DeviceRegLayout` + `BdFieldLayout` family migrated to
  `xdna_archspec::dma::{layouts,field_layouts}`: done. xdna-emu's
  `src/device/regdb/mod.rs` shrunk to the OnceLock wrapper + config-
  coupled loader + lock-value-width fields (pinned for Subsystem 4).
- xdna-emu-side `Deref<Target = ArchDeviceRegLayout>`: consumer call
  sites compile unchanged through the migration.
- Production `DmaEngine::new()` at `engine/mod.rs:147` accepts
  `&'static dyn DmaModel`, threaded from `arch.dma_model()` at
  `src/device/array/mod.rs:205`.
- `DmaTimingConfig::from_model(&dyn DmaModel)` replaces `from_arch()`
  at `engine/mod.rs:163`. `from_arch()` retained as `#[deprecated]`
  wrapper for test callers.
- Five AIE2-only feature-gate call sites consult `self.dma_model.
  supports_X()`: `task_queue_ops.rs` (enqueue + 3 accessors),
  `stepping.rs:346` (task-queue pop), `status.rs:32-39` (queue bits),
  `status.rs:104-110` (is_ooo), `status.rs:136-153` (compression
  config). AIE2 returns `true` for every flag so behavior is
  unchanged; AIE1's future `Aie1DmaModel` plugs in by returning
  `false`.
- `(2, 2)` silent fallback in `crates/xdna-archspec/src/runtime.rs:278`
  replaced with `.expect()`.
- Seven hygiene items applied (compression `log::warn!`, named
  `PAD_MAX_D{0,1,2}_*` constants, AM025 citation in timing test
  comment, rewritten `MAX_TASK_QUEUE_DEPTH` comment, named
  `STREAM_BUFFER_CAPACITY_WORDS`, MemTile BD-channel validity hard
  error + test, plus two Task 3/5 code-review fold-ins).
- `docs/arch/dma-model.md` per-seam design note: written, includes
  the mandated "What would AIE1 look like?" section.
- `test_full_parse_all_devices` archspec failure: fixed (Task 8
  scope creep, worth the tiny extra commit to give this tag a clean
  archspec baseline).

### Net code delta

```
 docs/arch/dma-model.md                        |  166 ++
 docs/arch/subsys3-audit.md                    |  165 ++
 crates/xdna-archspec/src/dma/mod.rs           |  146 ++
 crates/xdna-archspec/src/dma/field_layouts.rs |  683 ++ (moved from xdna-emu)
 crates/xdna-archspec/src/dma/layouts.rs       |  674 ++ (moved from xdna-emu)
 crates/xdna-archspec/src/dma/layouts_tests.rs |  ~N migrated
 crates/xdna-archspec/src/aie2/dma.rs          |  139 ++
 crates/xdna-archspec/src/runtime.rs           |   ~60 ~ (ArchConfig::dma_model + Architecture field + (2,2) expect)
 crates/xdna-archspec/src/types.rs             |    - (Task 2's dma_model lived in dma/mod.rs, not types.rs)
 crates/xdna-archspec/src/aie2/mod.rs          |    3 +
 crates/xdna-archspec/src/lib.rs               |    1 +
 crates/xdna-archspec/src/device_model.rs      |    6 ~ (Task 8 test fix)
 src/device/regdb/mod.rs                       | ~120 ~ (shrunk to wrapper + Deref)
 src/device/regdb/field_layouts.rs             |    - (deleted; moved to archspec)
 src/device/regdb/tests.rs                     | ~840 - (most migrated to archspec)
 src/device/dma/engine/mod.rs                  |   ~40 ~ (threading + hygiene 6/7)
 src/device/dma/engine/status.rs               |   ~30 ~ (feature gates)
 src/device/dma/engine/stepping.rs             |    ~6 ~ (task-queue pop gate)
 src/device/dma/engine/task_queue_ops.rs       |   ~20 ~ (feature gates)
 src/device/dma/engine/stream_io.rs            |   ~15 ~ (STREAM_BUFFER_CAPACITY)
 src/device/dma/engine/tests.rs                |   ~20 + (new hard-error test)
 src/device/dma/timing.rs                      |   ~30 ~ (from_model + deprecated wrapper)
 src/device/dma/compression.rs                 |   ~25 ~ (log::warn! sites)
 src/device/dma/addressing.rs                  |   ~15 ~ (named pad constants)
 src/device/dma/token.rs                       |   ~10 ~ (rewritten comment)
 src/device/array/mod.rs                       |    1 ~ (threads dma_model)
```

Net: ~1600 LOC relocated (xdna-emu → archspec), ~200 LOC new (trait +
impl + tests), ~150 LOC modified in xdna-emu for threading + gates +
hygiene. The Deref trick meant zero consumer import rewrites across
the ~20 files that import from `crate::device::regdb`.

### Follow-ups flagged

Follow-ups that fit naturally in later work, NOT blocking:

- **Subsystem 4 (Locks):** migrate `sign_extend_lock_value` + the
  `lock_value_width` / `lock_value_mask` / `lock_value_sign_bit`
  fields from xdna-emu's `DeviceRegLayout` wrapper into `LockModel`.
  The wrapper struct likely collapses entirely after that, replaced
  by `pub use xdna_archspec::dma::DeviceRegLayout;`.
- **AIE1 plug-in:** `Aie1DmaModel` populates when AIE1 support starts.
  Adds `ChannelFsm::Interleaving` and `ChannelFsm::DoubleBuffer`
  variants (gated by `supports_interleave_mode()` /
  `supports_double_buffer()`), adds `Aie1RegLayout` to archspec (5-word
  shim BD, 7-word tile BD, interleave-related fields), and extends
  `ArchConfig::dma_model()` dispatch to return `&AIE1_DMA_MODEL`.
- **Remove deprecated `DmaTimingConfig::from_arch`:** safe once all
  test callers migrate to `from_model(&AIE2_DMA_MODEL)`. Grep-driven
  cleanup.
- **Phase 2 hygiene:** `FotCountFifo` hardcoded capacity AM025
  citation (`src/device/dma/fifo.rs:297`); `PadPhase` state-machine
  ASCII diagram; trace-event per-arch routing; the latent
  `stream_in.len() < 256` channel-count comparison at
  `engine/stream_io.rs:~121` (TODO-commented in Task 7, likely a
  semantic bug, not the buffer-length check it looks like).
