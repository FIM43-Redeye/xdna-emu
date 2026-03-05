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

---

## Vision

```
Developer writes kernel -> Compiles with Peano/Vitis -> Emulates with xdna-emu -> Runs on hardware
                                                               |
                                                     Visual debugging, profiling,
                                                     correctness validation
```

The emulator should be a drop-in component that works with:
- **Peano** (open-source LLVM-based compiler)
- **Vitis** (AMD's full toolchain)
- **mlir-aie** (MLIR-based flow)

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
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | Functional | Mixed | Unit tests pass; real-binary coverage is thin |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | Not started | N/A | |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | GUI exists | OBSERVED | GUI renders; debugging features not built |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | In progress | Mixed | Test harness exists; coverage has major gaps |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | Not started | N/A | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | Not started | N/A | |

Run `cargo test --lib` to see the current test count. Do not rely on
numbers written in documentation -- they go stale within a session.

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
- DMA engine (multi-dimensional addressing, BD chaining, repeat count)
- Synchronization (locks with acquire/release, barriers, deadlock detection)
- Stream switch (circuit routing, packet switching, routing latency)
- Pipeline timing (hazard detection, branch delay slots, event tracing)
- Multi-core coordination (arbitration, cross-tile memory latency)
- Kernel execution: `add_one_using_dma` produces correct outputs

All of the above have dedicated unit tests that run on every `cargo test`.

**What is OBSERVED (worked in a session, not regression-tested):**
- `add_314_using_dma_op` produced correct outputs (64/64)
- Three-tile pipeline data flow completed
- Bidirectional ping-pong DMA transferred correctly
- 24 mlir-aie xclbins loaded without unknown-opcode errors

**What is CLAIMED (untested or based on very limited evidence):**
- "~85% ISA coverage" -- not measured against a complete ISA inventory
- "~95% binary compatibility" -- no systematic binary compatibility suite exists
- "100% instruction recognition" -- tested on very few real binaries
- Component "100% complete" markers -- no acceptance criteria were defined

**Next:** Expand real-binary testing, define verification criteria, validate
against hardware.

### Phase 2: Toolchain Integration

Plug into existing development flows: Peano, Vitis, mlir-aie, XRT.

**Status:** Not started. This is genuine -- no work has been done here.

### Phase 3: Developer Experience

Visual debugging and profiling: breakpoints, watchpoints, execution timeline.

**Status:** GUI framework (egui) exists and renders the tile grid. No debugging
features have been built. The GUI was last exercised several weeks ago and may
need updates to match current data structures.

### Phase 4: Validation & Testing

See [phase4-validation-testing.md](docs/roadmap/phase4-validation-testing.md)
for the detailed test audit.

**Status:** A test harness exists and all unit tests pass. But coverage is
heavily concentrated in unit tests with significant gaps in integration testing,
real-binary validation, and hardware comparison.

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

### Real-Binary Coverage

- Only ~2 mlir-aie binaries have been manually tested to completion
  (`add_one_using_dma`, `add_314_using_dma_op`)
- 30 mlir-aie test binaries are available; most have not been run
- `add_one_objFifo` produces wrong results (input+41 instead of input+1),
  suggesting a buffer layout convention mismatch
- `vec_vec_add_tile_init` fails on multi-stream routing (shim port 12)
- Test runner times out on continuous-loop kernels (double-buffering patterns)

### ISA Coverage

- SemanticOp coverage: only 7 of 40+ defined SemanticOps have dedicated tests
- Vector operations: limited to basic tests, no float32 edge cases
- SIMD shuffle/permute: basic implementation only, mapped generically
- Sparse matrix multiply: maps to dense (no true sparse support)
- ~12 specialized TableGen instructions not yet implemented

### Multi-Tile / Integration

- Only 3 unit tests exercise more than 1 tile
- No integration tests for realistic multi-tile workloads
- Cross-tile memory latency model untested against hardware

### Timing / Cycle Accuracy

- No hardware cycle-count comparisons exist
- No aiesimulator comparisons exist
- Timing model could be completely wrong -- we have no evidence either way
- Full pipeline model (fetch/decode/execute/writeback stages) not implemented;
  current model uses operation latencies + hazard detection

### DMA Subsystem

- Dual-abstraction issue: `channel.rs` and `engine.rs` both define channel state
- 8 `unwrap()` calls in `engine.rs` should be replaced with `expect()`
- BD field parsing correctness depends on AM029 interpretation that has not been
  cross-checked against aie-rt source

### Documentation Drift

- Documentation has historically drifted from reality within 1-2 sessions
- No automated mechanism exists to detect stale claims
- Consider: can any doc claims be replaced with `cargo test` output?

---

## Resources

- **llvm-aie**: https://github.com/Xilinx/llvm-aie (instruction definitions)
- **aie-rt**: https://github.com/Xilinx/aie-rt (register definitions)
- **mlir-aie**: https://github.com/Xilinx/mlir-aie (test cases, examples)
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
