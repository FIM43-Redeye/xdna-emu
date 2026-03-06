# Roadmap

A development plan for xdna-emu, an open-source emulator for AMD XDNA NPUs.

## How to Read This Document

Every status claim in this roadmap carries a confidence marker:

| Marker | Meaning | What It Requires |
|--------|---------|------------------|
| **VERIFIED** | Confirmed by automated tests or reproducible evidence | Passing tests exist, or the claim can be demonstrated on demand |
| **OBSERVED** | Worked in a specific session, but not continuously validated | Was seen working; no regression test guards it |
| **CLAIMED** | Stated in docs or believed true, but never systematically verified | Inherited from previous sessions, or based on limited evidence |

This is not pessimism. It is knowing what we can trust when we build on top of it.
Claims marked CLAIMED are not necessarily wrong -- they are untested.

Run `cargo test --lib` to see the current test count. Do not rely on
numbers written in documentation -- they go stale within a session.

---

## Vision

```
Developer writes kernel -> Compiles with Peano/Chess -> Emulates with xdna-emu -> Runs on hardware
                                                               |
                                                     Visual debugging, profiling,
                                                     correctness validation
```

The emulator should be a drop-in component that works with:
- **Peano** (open-source LLVM-based compiler)
- **Chess** (AMD's proprietary compiler, from aietools)
- **mlir-aie** (MLIR-based flow)
- **XRT** (Xilinx Runtime, via driver plugin)

### Target Devices

| Driver ID | Product | Codename | Architecture | Array Size | Status |
|-----------|---------|----------|--------------|------------|--------|
| NPU1 | Ryzen AI | Phoenix/Hawk Point | AIE2 (XDNA) | 5 cols x 6 rows | **Primary target** |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P (XDNA2) | 8+ cols x 6 rows | Planned |
| NPU6 | (TBD) | Krackan | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |

Array sizes include the shim tile row (row 0). Driver IDs NPU2/NPU3 are
prototypes marked for deprecation -- not consumer devices.

We are starting with **Phoenix (NPU1/AIE2)** because it is the hardware we have.
AIE2 and AIE2P share most of the architecture; AIE2P support will be incremental.

---

## Project Status at a Glance

| Phase | Status | Confidence | Summary |
|-------|--------|------------|---------|
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | Functional | Mixed | Extensive unit tests; bridge tests validate real binaries |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | Partial | VERIFIED | XRT plugin works; bridge tests run; Peano compilation not integrated |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | GUI exists | OBSERVED | GUI renders; debugging features not built |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | Active | Mixed | Dual-compiler bridge tests, trace sweep, hardware comparison |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | Not started | N/A | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | Not started | N/A | |

---

## Phase 1: Core Accuracy

Make the emulator faithful to real AIE2 hardware behavior.

See [phase1-core-accuracy.md](docs/roadmap/phase1-core-accuracy.md) for the
detailed breakdown with per-component confidence markers.

**What is VERIFIED:**
- TableGen-driven instruction decoder with O(1) lookup
- Scalar unit (GPRs, pointer/modifier registers, ALU operations)
- Vector unit (W/X/Y registers, accumulators, element types including bf16/f32)
- Memory system (load/store with post-modify, bank conflict detection)
- DMA engine (multi-dimensional addressing, BD chaining, repeat count, zero-padding)
- Synchronization (locks with acquire/release, barriers, deadlock detection)
- Stream switch (circuit routing, packet switching with header insertion)
- Pipeline timing (hazard detection, branch delay slots, event tracing)
- Multi-core coordination (arbitration, cross-tile memory latency, inter-tile stream delays)
- Multiple kernels produce correct outputs via bridge tests (both Chess and Peano compilers)

All of the above have dedicated unit tests that run on every `cargo test`.

**What is OBSERVED (worked in a session, not regression-tested):**
- Multi-tile pipeline data flow completed
- Bidirectional ping-pong DMA transferred correctly
- Trace comparison shows emulator within ~0.6% of hardware cycle counts on clean traces

**What is CLAIMED (untested or based on very limited evidence):**
- Full ISA coverage -- not measured against a complete ISA inventory
- Float32 edge cases (NaN, inf, denorm) -- no dedicated tests
- SIMD shuffle/permute completeness -- basic implementation, mapped generically
- Sparse matrix multiply accuracy -- maps to dense, no true sparse support

**Next:** Expand real-binary testing, close remaining SemanticOp gaps, validate
vector compute config word handling for ML kernels.

### Phase 2: Toolchain Integration

Plug into existing development flows: Peano, Chess, mlir-aie, XRT.

**Status:** Partial. The XRT driver plugin is functional and is the primary
validation path via bridge tests. Direct Peano compilation from the emulator
is not yet implemented. See
[phase2-toolchain-integration.md](docs/roadmap/phase2-toolchain-integration.md).

**What works:**
- XRT driver plugin (`xrt-plugin/`) replaces the kernel driver for emulation
- Bridge test suite runs real XRT test programs against the emulator
- Dual-compiler support: tests compile with both Chess and Peano
- mlir-aie test binaries load and execute

**What does not:**
- Invoking Peano or Chess directly from the emulator
- Source-level debugging (map PC to source line)
- Direct mlir-aie build system integration

### Phase 3: Developer Experience

Visual debugging and profiling: breakpoints, watchpoints, execution timeline.

**Status:** GUI framework (egui) exists and renders the tile grid. No debugging
features have been built. The GUI is disconnected from live emulator state and
needs significant work to become useful. See
[phase3-developer-experience.md](docs/roadmap/phase3-developer-experience.md).

### Phase 4: Validation & Testing

See [phase4-validation-testing.md](docs/roadmap/phase4-validation-testing.md)
for the detailed test audit.

**Status:** Active development. The bridge test suite (`emu-bridge-test.sh`)
is the primary validation path, exercising the full XRT flow. Dual-compiler
testing validates with both Chess (ground truth) and Peano. Trace sweep
infrastructure compares emulator execution traces against real NPU hardware
traces in Perfetto format.

**What exists:**
- Extensive unit test suite (run `cargo test --lib` for current count)
- Bridge test suite with dual-compiler support
- Trace sweep and binary trace comparison (Rust-based, handles large traces)
- Hardware comparison via bridge tests with `--trace=sweep` mode

**Gaps:**
- SemanticOp test coverage is incomplete
- No CI pipeline yet
- No performance benchmarks or regression tracking

### Phase 5: Production Readiness

Performance, multi-device support, APIs, documentation.

**Status:** Not started. See
[phase5-production-readiness.md](docs/roadmap/phase5-production-readiness.md).

### Phase 6: Community & Ecosystem

Open-source hygiene, community building, packaging.

**Status:** Not started. See
[phase6-community-ecosystem.md](docs/roadmap/phase6-community-ecosystem.md).

---

## Appendix: Verification Gaps

These are the things we know we do not know. Each item here is either a CLAIMED
status that needs validation, or a known issue that lacks a regression test.

### ISA Coverage

- SemanticOp coverage is incomplete -- many defined SemanticOps lack dedicated tests
- Vector operations: limited edge case testing, no float32 corner cases
- SIMD shuffle/permute: basic implementation only, mapped generically
- Sparse matrix multiply: maps to dense (no true sparse support)
- Vector compute config word: framework exists but not fully validated for ML kernels

### Timing / Cycle Accuracy

- Trace comparison shows ~0.6% cycle count deviation on clean traces (OBSERVED)
- Remaining timing gaps: stall durations (lock hold times, stream delivery latency)
- Full pipeline model (fetch/decode/execute/writeback stages) not implemented;
  current model uses operation latencies + hazard detection
- Intra-tile and inter-tile latencies have been tuned but not exhaustively validated

### DMA Subsystem

- Zero-padding fields use element counts but state machine may treat as word counts
  (known root cause, see `docs/plans/` for fix plan)
- Token-based DMA synchronization not yet implemented

### Documentation Drift

- Documentation has historically drifted from reality within 1-2 sessions
- Run `cargo test --lib` for authoritative test counts, not documentation
- No automated mechanism exists to detect stale claims

---

## Resources

- **llvm-aie**: https://github.com/Xilinx/llvm-aie (instruction definitions)
- **aie-rt**: https://github.com/Xilinx/aie-rt (register definitions, hardware abstraction)
- **mlir-aie**: https://github.com/Xilinx/mlir-aie (test cases, device models, register database)
- **XRT**: https://github.com/Xilinx/XRT (runtime API)
- **AMD Docs**: AM020 (AIE2 Architecture), AM025 (Register Reference)

---

## Detailed Phase Documentation

| Phase | Document |
|-------|----------|
| Phase 1 | [phase1-core-accuracy.md](docs/roadmap/phase1-core-accuracy.md) |
| Phase 2 | [phase2-toolchain-integration.md](docs/roadmap/phase2-toolchain-integration.md) |
| Phase 3 | [phase3-developer-experience.md](docs/roadmap/phase3-developer-experience.md) |
| Phase 4 | [phase4-validation-testing.md](docs/roadmap/phase4-validation-testing.md) |
| Phase 5 | [phase5-production-readiness.md](docs/roadmap/phase5-production-readiness.md) |
| Phase 6 | [phase6-community-ecosystem.md](docs/roadmap/phase6-community-ecosystem.md) |
