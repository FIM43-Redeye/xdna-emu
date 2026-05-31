# Phase 4: Validation & Testing

**Goal**: Ensure correctness and maintain quality through automated testing.

**Status**: Active. Two strong validation paths exist: the unit test suite
(`cargo test --lib`) and the dual-compiler bridge test suite
(`scripts/emu-bridge-test.sh`) which runs mlir-aie xclbins through the full
XRT path with optional real-NPU HW comparison.

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
These numbers reflect the approximate distribution as of 2026-05-14, not
exact counts (run `cargo test --lib` for current numbers).

| Area | Approximate Tests | What They Cover |
|------|-------------------|-----------------|
| Device (tile, array, DMA, stream, host memory, regdb, perf counters, control packets) | ~950 | Component behavior in isolation |
| Interpreter (decode, execute, timing, state, neighbor cache, coordinator) | ~1100 | Instruction execution, hazards, stalls, cross-tile memory |
| Parser / TableGen (xdna-archspec, ELF, XCLBIN, CDO) | ~410 | Binary parsing, encoding resolution |
| Testing infrastructure (xclbin suite, test_cpp_parser, FFI) | ~190 | Test harness, discovery, integration |
| Trace + VCD + visual + NPU instructions | ~440 | Trace pipeline, comparison, GUI components |

### What the Tests Actually Validate

**VERIFIED by unit tests:**
- Scalar ALU operations produce correct results for representative inputs
- Vector operations produce correct results for basic element types
- Memory loads/stores honor addressing modes and alignment
- DMA transfers move data through the correct state machine transitions
- Lock acquire/release semantics match the AM020 specification
- Stream switch routing forwards data between configured ports
- Hazard detection correctly identifies RAW/WAW/WAR dependencies
- TableGen parser extracts instruction encodings that match manual inspection
- Bundle format detection works for all format sizes (16-128 bit)

**VERIFIED by bridge tests against real NPU hardware:**
- Output buffer correctness on ~75 mlir-aie kernels covering objFifo,
  cascade flows, packet flow (incl. fanin/fanout), memtile, multi-column,
  runlist, dynamic objFifo, and control-packet kernels
- Dual-compiler agreement (Chess as ground truth, Peano as second compiler)
- Multi-tile / multi-column data flow

**VERIFIED by ISA harness against real NPU hardware:**
- All 4815 ISA test points across the AIE2 ISA

**NOT validated by tests:**
- Correctness against aiesimulator (no comparison infrastructure)
- Float32 edge cases beyond what the ISA harness covers
- Cycle-count accuracy beyond ~0.6% on clean kernels (broadcast/anchor
  timing tracked under #321/#322/#323)
- Performance regression (no benchmarks)

---

## Coverage Gaps

These are the areas where more testing would most improve confidence.

### 1. SemanticOp Coverage Matrix

Most SemanticOps that appear in compiled mlir-aie kernels are exercised
indirectly through bridge tests, but the dedicated SemanticOp test matrix
is incomplete. Filling it out gives faster failure localization than a
full bridge run.

**What "adequate" looks like:**
- Every SemanticOp that appears in a compiled mlir-aie kernel has at least
  one dedicated unit test
- Vector ops tested with int8, int16, int32, bf16, f32 element types

### 2. Timing Validation Beyond ~0.6%

The trace-sweep pipeline compares EMU vs HW cycle counts on clean
kernels and shows ~0.6% deviation in the easy cases. The harder cases
-- broadcast-event propagation latency (PROPOSED), per-NPU-instruction
cycle cost calibration (Phases 1-3 landed, Phase 4+ ongoing), and
NoC/AXI/DMA pipeline timing (DEFERRED) -- are indexed in
[cycle-accuracy-mission.md](../coverage/cycle-accuracy-mission.md).

**What "adequate" looks like:**
- Cycle counts compared against HW for the full bridge test set
- Anchor window agreement within 1% on every PASSing test
- aiesimulator differential as a stretch goal (lower priority than
  real-HW comparison since real NPU is ground truth)

### 3. Differential Kernel Fuzzing

Bridge tests cover hand-written kernels with hand-checked outputs.
Differential fuzzing against real HW (see Phase 4 "Advanced Testing"
below) is the long-term path to catching subtle EMU/HW divergence.

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
| Hardware comparison tests | VERIFIED | `scripts/emu-bridge-test.sh` runs HW + EMU and diffs outputs; trace-sweep compares cycle counts |
| Fuzzing for decoder robustness | Not started | |
| Property-based testing for DMA addressing | Not started | |
| Differential kernel fuzzer | Revived (scalar), blocked by BUG-A | `cargo run -- fuzz`; EMU+HW differential, Peano, single-tile scalar. In-process EMU path stalls shim DMA (BUG-A) -> not yet a correctness gate. Vector/Chess deferred. See docs/fuzzer-usage.md + findings/2026-05-30-fuzzer-revival-first-batch.md |

### Differential Kernel Fuzzer

The fuzzer is revived and runs end-to-end (`cargo run --release -- fuzz --hw`).
The differential loop (generate -> Peano-compile -> EMU + NPU -> byte-diff)
works and the in-sandbox HW path is proven. See
[docs/fuzzer-usage.md](../fuzzer-usage.md) for full invocation details.

**Current blocker (BUG-A):** the in-process EMU path (`XclbinSuite`) stalls
shim DMA at `BdSetup`, so the emulator returns all-zeros for every fuzz kernel
and every non-trivial seed appears to diverge for that one reason. The XRT
bridge path is unaffected. Until BUG-A is fixed the fuzzer is not a useful
correctness gate. See
`docs/superpowers/findings/2026-05-30-fuzzer-revival-first-batch.md`.

**Scope**: single-tile scalar ops (arith/bitwise/shift/branch/hw-loop),
Peano-compiled, i8/i16/i32. Vector ops and Chess are planned next phases.
Shrinking is out of scope; the deterministic seed is the reproducer.

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
