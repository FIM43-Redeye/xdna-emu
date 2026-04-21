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

*(To be filled in by Task 8.)*
