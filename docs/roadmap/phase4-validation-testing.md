# Phase 4: Validation & Testing

**Goal**: Ensure correctness and maintain quality through automated testing.

**Status**: A test harness exists and all tests pass. But coverage is
unit-heavy with significant gaps in integration testing, real-binary
validation, and hardware comparison.

For the confidence tier system, see [ROADMAP.md](../../ROADMAP.md).

---

## What Exists Today

### Test Infrastructure

Run `cargo test --lib` to see the current count. The test suite has zero
failures and zero ignored tests.

The test harness (`src/interpreter/test_runner.rs`) supports loading real
xclbin files and running kernels to completion. A test discovery script
(`scripts/run-tests.sh`) manages doc tests separately from library tests
to avoid overwhelming the system with TableGen-loading subprocesses.

### Coverage by Module

Coverage is heavily concentrated in device and interpreter unit tests.
These numbers reflect the approximate distribution, not exact counts
(run `cargo test` for current numbers).

| Area | Approximate Tests | What They Cover |
|------|-------------------|-----------------|
| Device (tile, array, DMA, stream, host memory) | ~130 | Component behavior in isolation |
| Interpreter (decode, execute, timing, state) | ~250 | Instruction execution, hazards, stalls |
| Parser / TableGen | ~85 | Binary parsing, encoding resolution |
| Integration (test_runner, coordinator) | ~20 | Multi-component interaction |
| Other (config, FFI) | ~10 | Utility functions |

### What the Tests Actually Validate

**VERIFIED by tests:**
- Scalar ALU operations produce correct results for representative inputs
- Vector operations produce correct results for basic element types
- Memory loads/stores honor addressing modes and alignment
- DMA transfers move data through the correct state machine transitions
- Lock acquire/release semantics match the AM020 specification
- Stream switch routing forwards data between configured ports
- Hazard detection correctly identifies RAW/WAW/WAR dependencies
- TableGen parser extracts instruction encodings that match manual inspection
- Bundle format detection works for all format sizes (16-128 bit)

**NOT validated by tests:**
- Correctness against real hardware output (no hardware comparison suite)
- Correctness against aiesimulator (no comparison infrastructure)
- Full ISA coverage (only 7 of 40+ SemanticOps have dedicated tests)
- Float32 edge cases (NaN, infinity, denormals)
- Multi-tile data flow under realistic conditions (only 3 multi-tile tests)
- Continuous-loop kernels (test runner times out)
- objFifo buffer convention (known wrong: produces input+41)
- Cycle-count accuracy (no reference to compare against)
- Performance regression (no benchmarks)

---

## Coverage Gaps

These are the areas where more testing would most improve confidence.

### 1. Real-Binary Validation

Currently only 1-2 mlir-aie kernels have been verified to produce correct
output. 30 test binaries are available in the local mlir-aie build directory.

**What "adequate" looks like:**
- All 30 available mlir-aie test binaries load without crashes
- At least 10 produce correct output values
- Failures are triaged and tracked as known issues

### 2. SemanticOp Coverage

Only 7 of 40+ defined SemanticOps have tests. The most impactful gaps:
- No tests for SemanticOps used in real kernels but untested in isolation
- Vector operations tested with ~18 tests, but no float32 edge cases
- No coverage matrix mapping SemanticOps to element types

**What "adequate" looks like:**
- Every SemanticOp that appears in a compiled mlir-aie kernel has at least
  one test
- Vector ops tested with at least int8, int16, int32, bf16

### 3. Multi-Tile Integration

Only 3 tests exercise more than 1 tile, and those are unit tests that
construct tile arrays manually, not tests that load real multi-tile binaries.

**What "adequate" looks like:**
- At least 3 real multi-tile binaries run to completion with correct output
- Cross-tile DMA, stream routing, and lock synchronization all exercised

### 4. Timing Validation

No timing comparisons exist against any reference. The timing model may be
completely wrong.

**What "adequate" looks like:**
- Cycle counts compared against aiesimulator for at least 3 kernels
- Deviations documented and explained (acceptable vs. bugs)

---

## Planned Infrastructure (Not Started)

These items are from the original Phase 4 plan and remain relevant.

### Benchmarks

| Task | Status | Notes |
|------|--------|-------|
| Standard kernel benchmarks (matmul, conv2d) | Not started | |
| Performance regression tracking | Not started | |
| Emulation speed benchmarks (cycles/second) | Not started | |

### Continuous Integration

| Task | Status | Notes |
|------|--------|-------|
| GitHub Actions for build/test | Not started | |
| Coverage reporting | Not started | |
| Benchmark dashboards | Not started | |
| Release automation | Not started | |

### Advanced Testing

| Task | Status | Notes |
|------|--------|-------|
| Automated comparison with aiesimulator | Not started | |
| Hardware comparison tests | Not started | Requires NPU access in CI |
| Fuzzing for decoder robustness | Not started | |
| Property-based testing for DMA addressing | Not started | |
| Differential kernel fuzzer | Not started | See below |

### Differential Kernel Fuzzer (Future)

A tool that randomly generates valid NPU kernels (valid instruction sequences,
valid DMA configurations, valid routing), runs them on both the emulator and
real hardware, and diffs the output buffers. This is the most powerful way to
find subtle emulator/hardware divergence -- it tests the emulator against the
actual silicon rather than against our understanding of the architecture docs.

Key design considerations:
- Must generate *valid* programs (not random bytes) -- valid VLIW bundles,
  legal register usage, coherent DMA descriptors
- Should start simple (single-tile scalar ops) and grow to multi-tile vector
  pipelines
- Needs a harness that packages generated code into minimal xclbin containers
- Shrinking/minimization when a divergence is found (like proptest/hypothesis)
- Could leverage TableGen instruction definitions to know what's legal

---

## Test Binaries

Available from the local mlir-aie build:
```
/home/triple/npu-work/mlir-aie/
├── build/
│   └── ... compiled examples ...
└── programming_examples/
    └── ... source code ...
```

---

## Resources

- **mlir-aie**: `/home/triple/npu-work/mlir-aie`
- **cargo-fuzz**: https://github.com/rust-fuzz/cargo-fuzz
- **criterion.rs**: https://github.com/bheisler/criterion.rs
