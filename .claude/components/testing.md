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
| `artifacts.rs` | `BuildArtifact`, `discover_build_artifacts()`, `ExampleSource` -- test artifact and xclbin discovery |
| `hardware_comparison.rs` | `CrossValidation`, `CompilerComparison` -- emulator vs hardware reference comparison |
| `host_defines.rs` | `extra_defines()` -- extra C++ compilation flags injected into test builds |
| `native_hw.rs` | Compiles and runs each test's own test.cpp on real NPU hardware (preferred HW path) |
| `npu_runner.rs` | Runs tests on real NPU hardware via npu-runner binary (fallback) |
| `npu_test.rs` | `NpuTestSource`, `TestOverrides` -- test discovery, skip/expected-fail gates |
| `opcode_collector.rs` | `OpcodeCollector` -- collects unknown opcodes when execution fails (for coverage analysis) |
| `process_control.rs` | `ProcessOutcome`, `spawn_with_timeout()`, `wait_with_timeout()` -- subprocess execution and timeout handling |
| `test_cpp_parser.rs` | Parses mlir-aie `test.cpp` files for buffer metadata (`BufferSpec`, sizes, types, input patterns) |
| `unit_test.rs` | `UnitTest`, `UnitTestBuildResult`, `discover()` -- mlir-aie unit test discovery |
| `xclbin_suite.rs` | `XclbinSuite` -- discovers and runs xclbin test files, reports `TestOutcome` |

### NPU Module (`src/npu/`)

Host-to-NPU communication protocol (XRT instruction stream).

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, `NpuOpcode` enum (Write32, BlockWrite, DdrPatch, etc.) |
| `parser.rs` | `NpuInstructionStream` -- parses binary NPU instruction sequences |
| `executor.rs` | `NpuExecutor` -- executes NPU instructions against device state (configures shim DMA, patches addresses) |
| `classify.rs` | Kernarg-role classifier: recovers semantic slot roles (Ctrlpkt, MM2S, S2MM) from the NPU instruction stream for bridge-trace-runner buffer binding |
| `cycle_cost.rs` | Control-path cycle-cost model: multi-stage latency for host→CMP→NoC→tile control packet delivery |

### FFI (`crates/xdna-emu-ffi/src/lib.rs`)

`XdnaEmuHandle` -- C-compatible foreign function interface for integrating with C/C++ applications (mock XRT library). Provides `extern "C"` functions for loading xclbins, running emulation, and reading results.

### Configuration (`src/config.rs`)

`Config` -- path management (llvm-aie, mlir-aie, XRT) loaded from `xdna-emu.toml`, environment variables, or defaults.

### External Test Infrastructure (`tests/`)

| Path | Purpose |
|------|---------|
| `tests/test_overrides.toml` | Skip gates and expected-fail annotations (replaces per-test manifest files) |
| `tests/npu-outputs/` | Captured NPU hardware reference outputs (machine-specific, gitignored) |
| `tests/mlir-aie-extracted/` | Extracted/generated test files (regenerable from mlir-aie) |
| `tests/mlir-aie/` | Pointers/snapshots into the local mlir-aie build |

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

# Bridge test suite -- the primary validation path
./scripts/emu-bridge-test.sh                    # full HW + EMU dual-compiler run
./scripts/emu-bridge-test.sh --no-hw            # emulator only, no NPU needed
./scripts/emu-bridge-test.sh --no-hw add_one    # single test, EMU only
./scripts/emu-bridge-test.sh --sweep            # add event sweep across all tiles
./scripts/emu-bridge-test.sh --trace=pc-anchored ...  # add PC-anchored cycle trace comparison

# ISA test harness (real NPU only)
./scripts/isa-test.sh

# Standalone Rust binaries
cargo run --release --bin trace-compare -- ...   # HW vs EMU events JSON diff
cargo run --release --bin export-isa             # dump TableGen ISA tables
cargo run --release --bin npu-archspec           # archspec inspector
cargo run --release --bin vcd-compare -- ...     # aiesimulator VCD diff (--features analysis)
```

## Test Validation Architecture

Tests are validated against **hardware reference outputs** rather than
computed expected values. The pipeline is:

1. `test_cpp_parser` extracts buffer metadata (sizes, types, input patterns)
   from mlir-aie's test.cpp files
2. `native_hw` (preferred) compiles each test's own test.cpp on real NPU
   hardware and runs it; `npu_runner` is the fallback path; either way
   the captured output lands under `tests/npu-outputs/<test_name>/`
3. `xclbin_suite` sets up emulator input buffers from `BufferSpec`, runs
   the emulation, and compares output against the hardware reference
4. `test_overrides.toml` provides skip gates and expected-fail annotations
   for tests that cannot run or have known issues
5. The bridge test suite (`scripts/emu-bridge-test.sh`) is a parallel,
   broader validation path that runs each test's own `test.exe` against
   both real NPU and the emulator XRT plugin

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
ground truth, Peano informational). Six phases: discover, compile (parallel),
run HW (parallel -j5), run EMU (parallel -j nproc), trace comparison
(`--trace=pc-anchored`), event sweep (`--sweep`), report per-compiler matrix.

Sweep mode (Phase 5b) is two-phase: HW arm runs serially across tests
(NPU is single-tenant), EMU arm runs serial across tests with parallel
events per test (-j N). The EMU arm uses `bridge-trace-runner --batch-stdin`
with a RESET command so worker processes can re-run multiple tests
without spawning a fresh emulator each time.

`bridge-trace-runner` captures a tile state snapshot (CORE_PC/STATUS,
TIMER_LOW, perf counters, all 16 LOCK*_VALUE, DMA channel CTRL/STATUS)
on `run.wait` timeout when invoked with `--snapshot-on-timeout <dir>`.
Sweep mode wires this automatically into
`<sweep_out_dir>/<test>.<compiler>.lockstep.work/wedge-snapshots/`.

Key flags: `--chess-only`, `--peano-only`, `--no-hw`, `--compile`,
`--serial-hw`, `--sweep`, `--trace=pc-anchored`, `-v <filter>`.

### Known coverage gap: Python/IRON host path

The suite only ever exercises the **C++ host** (`test.exe`, raw XRT API). Some
upstream kernels (e.g. `add_one_objFifo`, `add_one_objFifo_elf`) also ship a
**Python/IRON host** (`test.py`, `aie.iron` + `DefaultNPURuntime.run_test` via
pyxrt) for the *same* xclbin. Both lit RUN lines derive the same variant name
(`""`, from `aie.xclbin`), so `get_run_variants` collapses them and
`get_variant_run_cmd("")` always resolves to the first command -- the C++ one.
The Python host is never run. (Before the variant-dedup fix this surfaced as a
double-enqueued job that ran `test.exe` twice.)

This is fine for *fidelity* -- the kernel compute is identical regardless of
host -- but it leaves the **IRON/pyxrt -> plugin integration** unvalidated.
Making the emulator a true drop-in for the IRON Python flow is a wanted future
feature: it needs a real *host axis* (cpp vs py) distinct from the
xclbin-derived variant, plus confirming `DefaultNPURuntime` actually routes
through `XDNA_EMU` to our plugin. Deliberate feature, not a quick flag.

### Specialized capture: `tools/bug6-trace.sh`

Single-pkexec wrapper that enables the existing `amdxdna_trace`
kernel tracepoints (xdna_job, mbox_*, uc_*), runs one test under
timeout, and snapshots the trace ringbuffer + dmesg. Built for the
bug #6 (`memtile_dmas/writebd` hang) investigation; shelved 2026-05-14
with canonical pass shape captured. Re-run if writebd hangs again --
diff against `build/experiments/bug6/pass-baseline-v2.trace`.

## Trace Pipeline

Binary trace comparison between emulator and real NPU hardware. The legacy
Python tools have been superseded by mlir-aie's declarative trace IRON API
(Phase B of cycle-budget testing). Legacy tools remain in `tools/deprecated/`
for reference.

**Active pipeline:**
- `tools/mlir-trace-inject.py` -- inject trace routing into MLIR (uses mlir-aie API)
- `bridge-runner/bridge-trace-runner` -- multi-batch sweep orchestrator
- `tools/parse-trace.py` -- single-source decoder wrapping mlir-aie's `parse_trace`; emits flat events JSON, cycles scalar, raw Perfetto, or raw command stream
- `src/bin/trace_compare.rs` -- Rust binary comparison (replaced Python OOM)
- `src/trace/compare.rs` -- core comparison logic (mode 0)
- `src/trace/compare_mode2.rs`, `src/trace/mode2_decode.rs` -- mode-2 (INST_EXEC) PC-anchored comparator
- `src/trace/vcd.rs`, `src/vcd/` -- aiesimulator VCD parser and mapper

Build: `cargo build --release --bin trace-compare`

**Deprecated tools** (do not add new callers; v1 suffix avoids confusion with current top-level tools of the same base name):
- `tools/deprecated/trace-inject.py` -- legacy MLIR trace injection
- `tools/deprecated/trace-sweep-v1.py` -- pre-IRON-API multi-batch orchestrator
- `tools/deprecated/trace-trim.py` -- buffer trimming
- `tools/deprecated/trace-merge.py` -- Perfetto JSON merge
- `tools/deprecated/trace-patch-events-v1.py` -- pre-IRON-API event patching
- `tools/deprecated/trace-bridge.sh` -- shell driver superseded by `scripts/emu-bridge-test.sh`
- `tools/deprecated/trace-compare.py` -- Python comparator superseded by `src/bin/trace_compare.rs`

See `tools/deprecated/README.md`.
