# Phase 4: Validation & Testing

**Goal**: Ensure correctness and maintain quality.

**Status**: 🟡 267 Tests (real binary test added)

---

## Overview

Validation happens at multiple levels:
- Unit tests for individual components
- Integration tests against real binaries
- Comparison with aiesimulator and hardware
- Fuzzing for robustness

---

## 4.1 Test Infrastructure

| Task | Status | Notes |
|------|--------|-------|
| Import mlir-aie test suite | 🔲 TODO | |
| Automated comparison with aiesimulator | 🔲 TODO | |
| Hardware comparison tests (when available) | 🔲 TODO | |
| Fuzzing for decoder robustness | 🔲 TODO | |

---

## 4.2 Benchmarks

| Task | Status | Notes |
|------|--------|-------|
| Standard kernel benchmarks (matmul, conv2d) | 🔲 TODO | |
| Performance regression tracking | 🔲 TODO | |
| Emulation speed benchmarks (cycles/second) | 🔲 TODO | |

---

## 4.3 Continuous Integration

| Task | Status | Notes |
|------|--------|-------|
| GitHub Actions for build/test | 🔲 TODO | |
| Coverage reporting | 🔲 TODO | |
| Benchmark dashboards | 🔲 TODO | |
| Release automation | 🔲 TODO | |

---

## Current Test Coverage

**Total: 267 tests passing** (262 unit + 5 doc tests)

See [Phase 1](phase1-core-accuracy.md#test-coverage) for detailed breakdown.

### Real Binary Testing

Successfully tested against mlir-aie ELF:
- `add_one_objFifo/main_core_0_2.elf`
- 20% instruction recognition rate (baseline)

---

## Test Binaries

Available from local mlir-aie clone:
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
