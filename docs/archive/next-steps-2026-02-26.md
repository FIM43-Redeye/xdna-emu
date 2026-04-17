# Next Steps (2026-02-26)

Session accomplished three things on dev branch:

1. **Cross-tile lock access** (commit 7467cf8) - MemTile DMA can now acquire/release locks on west/east neighbor tiles. Eliminated ~300K warnings per run.
2. **DMA task queue backpressure** (commit f127ee3) - NPU instruction executor drains DMA queues before enqueueing when full, matching aie-rt behavior. Reduced 1.4M warnings to 15.
3. **Per-test warning attribution** (commit 4f8a7a8) - Warnings now appear inline per-test instead of flooding the global log. Summary shows "Warnings: 15 (3 tests)".

## Current test results (69 tests, --no-build)

- 19 pass (15 validated, 4 no validation)
- 17 validation fail
- 32 timeout
- 1 skip
- HW: Peano 63/64, Chess 60/61

## What to investigate next

### Remaining 15 warnings: sync_task_complete_token tests

Two tests (`sync_task_complete_token`, `sync_task_complete_token_bd_chaining`) each produce 7 warnings about shim DMA tile(0,0) ch0 queue full. The shim DMA queue can't drain during NPU instruction execution because shim transfers need the full system (cores + stream routing) running. These tests enqueue many BDs to shim before the engine starts stepping.

Possible fix: defer shim DMA enqueue until engine stepping begins, or expand the drain loop to also step stream switches.

### add_one_objFifo_elf: ELF masquerading as insts.bin

This test has an ELF binary where insts.bin would normally be (0x464C457F = ELF magic). The test has "elf" in the name - likely an ELF-only test that doesn't use NPU instructions. Low priority; the warning correctly identifies the situation.

### Timeouts (32 tests)

The biggest category. Many of these are memtile/adjacent tests that might be blocked on:
- Multi-iteration BD chains (add_314 loops with repeat count > 1)
- Token-based synchronization (not yet implemented)
- Control packet reconfiguration (ctrl_packet tests)
- BD chaining across repeated iterations

### Validation failures (17 tests)

Several patterns visible:
- `add_256_using_dma_op_no_double_buffering` - output has 256 where expected 0 (DMA reading uninitialized memory?)
- `cascade_flows` - 59/64 correct (cascade stream data slightly off)
- `matrix_multiplication_using_cascade` - 255/256 correct (single element wrong)
- Token/task tests (`writebd_tokens`, `dma_configure_task_token`) - token mechanism not implemented
- `tile_mapped_read` - gets 42 where expected 0 (core MMIO register access goes to data memory)

### Core MMIO register access

`tile_mapped_read` test reads tile registers via load instructions. Currently, load/store to register addresses goes to data memory instead of register space. This is a known gap (see MEMORY.md "Known Open Issues").

### Token-based DMA synchronization

Multiple tests use BD tokens (`writebd_tokens`, `dma_configure_task_token`, `sync_task_complete_token`). The token mechanism is different from lock-based sync and is not yet implemented.
