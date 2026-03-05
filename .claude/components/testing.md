# Testing Infrastructure

Test harness, suite management, NPU instruction execution, FFI bindings, and external test scripts.

Read this file when working on tests, the test runner, FFI, or NPU instruction handling.

## Files

### Test Runner (`src/interpreter/test_runner.rs`)

`TestRunner` -- end-to-end kernel execution harness. Loads an XCLBIN, configures the device, runs cores, and validates output against expected values.

### Testing Module (`src/testing/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports |
| `xclbin_suite.rs` | `XclbinSuite` -- discovers and runs xclbin test files, reports `TestOutcome` |
| `opcode_collector.rs` | `OpcodeCollector` -- collects unknown opcodes when execution fails (for coverage analysis) |
| `test_cpp_parser.rs` | Parses mlir-aie `test.cpp` files for buffer metadata (`BufferSpec`, sizes, types, input patterns) |
| `npu_test.rs` | `NpuTestSource`, `TestOverrides` -- test discovery, skip/expected-fail gates |
| `npu_runner.rs` | Runs tests on real NPU hardware via npu-runner binary (fallback) |
| `native_hw.rs` | Compiles and runs each test's own test.cpp on real NPU hardware (preferred HW path) |
| `hardware_comparison.rs` | `CrossValidation`, `CompilerComparison` -- emulator vs hardware reference comparison |

### NPU Module (`src/npu/`)

Host-to-NPU communication protocol (XRT instruction stream).

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, `NpuOpcode` enum (Write32, BlockWrite, DdrPatch, etc.) |
| `parser.rs` | `NpuInstructionStream` -- parses binary NPU instruction sequences |
| `executor.rs` | `NpuExecutor` -- executes NPU instructions against device state (configures shim DMA, patches addresses) |

### FFI (`src/ffi/mod.rs`)

`XdnaEmuHandle` -- C-compatible foreign function interface for integrating with C/C++ applications (mock XRT library). Provides `extern "C"` functions for loading xclbins, running emulation, and reading results.

### Configuration (`src/config.rs`)

`Config` -- path management (llvm-aie, mlir-aie, XRT) loaded from `xdna-emu.toml`, environment variables, or defaults.

### External Test Infrastructure (`tests/`)

| Path | Purpose |
|------|---------|
| `tests/test_overrides.toml` | Skip gates and expected-fail annotations (replaces per-test manifest files) |
| `tests/npu-outputs/` | Captured NPU hardware reference outputs (machine-specific, regenerate with `capture_npu_outputs`) |
| `tests/xclbin_integration.rs` | Integration tests loading real xclbin binaries (requires `--features xclbin-tests`) |
| `tests/hardware_comparison.rs` | Cross-validation integration tests (requires `--features hardware-compare`) |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/run-tests.sh` | Main test runner: runs lib tests normally, doc tests with nice -n 19 |
| `scripts/run-tests.ps1` | Windows equivalent |

## Running Tests

```bash
# All tests (recommended)
./scripts/run-tests.sh

# Library tests only (fast)
./scripts/run-tests.sh --lib
cargo test --lib

# Doc tests only (slow, loads TableGen files)
./scripts/run-tests.sh --doc

# Run full mlir-aie test suite (emulator + optional hardware comparison)
cargo run --bin npu-test

# Single test
cargo run --bin npu-test -- add_one_using_dma

# Lit wrapper mode (standard LLVM lit execution)
cargo run --bin npu-test -- --lit

# Trace collection mode
cargo run --bin npu-test -- --trace

# Triple trace comparison (HW + emulator + optional aiesimulator)
cargo run --bin npu-test -- --trace-all

# Capture NPU hardware reference outputs (requires NPU hardware)
cargo run --example capture_npu_outputs

# Cross-validate emulator vs hardware
cargo run --example compare_emu_hw
```

## Test Validation Architecture

Tests are validated against **hardware reference outputs** rather than
computed expected values. The pipeline is:

1. `test_cpp_parser` extracts buffer metadata (sizes, types, input patterns)
   from mlir-aie's test.cpp files
2. `capture_npu_outputs` runs each test on real NPU hardware and saves
   the output to `tests/npu-outputs/<test_name>/output.bin`
3. `xclbin_suite` sets up emulator input buffers from `BufferSpec`, runs
   the emulation, and compares output against the hardware reference
4. `test_overrides.toml` provides skip gates and expected-fail annotations
   for tests that cannot run or have known issues

This replaces the previous manifest system (68 TOML files with hardcoded
expected values) with automatic test discovery and hardware-as-oracle
validation.

## Test Coverage Notes

- Unit tests are concentrated in individual modules; run `cargo test --lib` for the fast suite
- Doc tests spawn separate processes that each load TableGen files -- the script runs them with `nice -n 19` and limited parallelism
- Integration tests (real binary execution) require feature flags and pre-built test artifacts

## Bridge Test Suite (`scripts/emu-bridge-test.sh`)

The primary validation target. Exercises the full XRT bridge path:
`test.exe -> XRT -> plugin -> emulator`. Dual-compiler by default (Chess
ground truth, Peano informational). Five phases: discover, compile, run HW
(parallel -j5), run EMU (parallel -j nproc), report per-compiler matrix.

Key flags: `--chess-only`, `--peano-only`, `--no-hw`, `--compile`,
`--serial-hw`, `--trace=sweep`, `-v`.

## Trace Pipeline

Binary trace comparison between emulator and real NPU hardware:
- `tools/trace-inject.py` -- inject trace routing into MLIR
- `tools/trace-sweep.py` -- multi-batch sweep across event types
- `src/bin/trace_compare.rs` -- Rust binary comparison (replaced Python OOM)
- `src/trace/compare.rs` -- core comparison logic
- `src/trace/vcd.rs` -- aiesimulator VCD parser

Build: `cargo build --release --bin trace-compare`
