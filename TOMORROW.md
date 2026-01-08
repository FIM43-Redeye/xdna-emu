# Tomorrow's Work

## Completed Today (2026-01-06)

- Fixed VLIW execution order bug: Store slots now execute before Scalar slots
- `add_one_using_dma` and `add_314_using_dma_op` both pass (64/64 correct)
- 629 unit tests passing
- Created `run_add_test` example for quick XCLBIN validation

## Next Steps

### 1. Investigate objFifo Test Failures

`add_one_objFifo` produces `input + 41` instead of `input + 1`:
- Different buffer layout convention than `_using_dma` tests
- May use different host memory addresses or buffer organization
- Check how objFifo tests set up their memory regions

### 2. Fix Multi-Stream Routing

`vec_vec_add_tile_init` fails with "write_output failed for (0,0) port 12":
- This test uses 2 input vectors (A + B)
- Shim tile port 12 routing not configured properly
- Need to trace CDO stream switch configuration for multi-input patterns

### 3. Improve Test Runner

The `run_mlir_aie_tests` suite times out because kernels loop forever:
- These kernels use double-buffering and run continuously
- Need output-based validation instead of waiting for halt
- Consider cycle budget + output verification approach

### 4. DMA Refactoring (from plan file)

Low-priority cleanup in `src/device/dma/engine.rs`:
- Replace 8 `unwrap()` calls with `expect()` for better panic messages
- Add `StepResult` struct for transfer error tracking
- See `/home/triple/.claude/plans/greedy-growing-phoenix.md` for full plan

## Test Commands

```bash
# Quick validation
./target/release/examples/run_add_test /path/to/test_dir <add_value>

# Full debug trace
RUST_LOG=warn ./target/release/examples/debug_add_one /path/to/xclbin

# Unit tests
cargo test --lib
```
