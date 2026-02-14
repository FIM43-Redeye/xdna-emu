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
| `manifest_runner.rs` | `ManifestRunner` -- runs tests from TOML manifests (input data, expected output, kernel config) |

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
| `tests/mlir-aie/` | Test discovery and management for mlir-aie test suite |
| `tests/mlir-aie/discover_tests.py` | Discovers available mlir-aie test binaries |
| `tests/mlir-aie/run_tests.py` | Runs mlir-aie tests through the emulator |
| `tests/mlir-aie/manifest.json` | Test manifest with paths and expected results |
| `tests/mlir-aie/schema.json` | JSON schema for manifest validation |
| `tests/mlir-aie/query_tests.py` | Queries test metadata |
| `tests/mlir-aie-extracted/` | Pre-extracted test data (xclbins + TOML manifests) |
| `tests/mlir-aie-extracted/extract_tests.py` | Extracts test data from mlir-aie build |
| `tests/mlir-aie-extracted/manifests/*.toml` | Per-test manifests (input arrays, expected output, tile config) |

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

# Run a specific manifest test
cargo run --example manifest_test -- tests/mlir-aie-extracted/manifests/add_one_using_dma.toml
```

## Test Coverage Notes

- Unit tests are concentrated in individual modules; run `cargo test --lib` for the fast suite
- Doc tests spawn separate processes that each load TableGen files -- the script runs them with `nice -n 19` and limited parallelism
- Integration tests (real binary execution) are sparse; only ~2 mlir-aie binaries have been fully validated
- The `tests/mlir-aie-extracted/manifests/` directory contains TOML manifests for extracted mlir-aie tests
