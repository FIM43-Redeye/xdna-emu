# Roadmap

A comprehensive plan to make xdna-emu a production-ready, open-source emulator for AMD XDNA NPUs.

## Vision

```
Developer writes kernel → Compiles with Peano/Vitis → Emulates with xdna-emu → Runs on hardware
                                                              ↓
                                                    Visual debugging, profiling,
                                                    correctness validation
```

The emulator should be a drop-in component that "just works" with:
- **Peano** (open-source LLVM-based compiler)
- **Vitis** (AMD's full toolchain)
- **mlir-aie** (MLIR-based flow)

### Multi-Device Strategy

The project targets all AMD XDNA NPUs (NPU1-4), but we're starting with **AIE2 (Phoenix/NPU1)** because:
1. It's the hardware we have for testing
2. AIE2 and AIE2P share most of the architecture
3. Once AIE2 works, AIE2P support is incremental

The architecture is designed to be generic from day one - device-specific details are isolated so adding new NPU variants is straightforward.

---

## Current Status

| Phase | Status | Progress |
|-------|--------|----------|
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | 🟢 Functional | 587 tests, 100% kernel accuracy, branch delay slots |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | 🔴 Not started | |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | 🟡 GUI exists | Needs debugging features |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | 🟡 587 tests | TestRunner harness added |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | 🔴 Not started | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | 🔴 Not started | |

---

## Phase Summaries

### Phase 1: Core Accuracy 🟢

Make the emulator faithful to real AIE2 hardware behavior.

**Key achievements:**
- TableGen-driven instruction decoder with O(1) lookup
- Scalar unit (32 GPRs, pointer/modifier registers, ALU)
- Vector unit (24x256-bit W registers, 8x256-bit accumulators)
- Memory system (load/store with post-modify addressing, bank conflict detection)
- DMA engine with multi-dimensional addressing (1D-4D patterns)
- Synchronization (locks with timing, barrier coordination, deadlock detection)
- **Stream switch complete**: circuit routing, packet switching, routing latency
- TestRunner harness for kernel execution
- **VLIW slot extraction** for all 16-128 bit bundle formats (100% recognition)
- Hazard detection (RAW/WAW/WAR) and latency tables
- **Pipeline timing complete**: branch penalty, VLIW structural hazards, stall integration
- **Event tracing**: 13 event types for profiling (instructions, stalls, branches, DMA, locks)
- **CycleAccurateExecutor integrated**: `new_cycle_accurate()` mode with full timing
- **Memory tile arbitration**: round-robin arbitration for multi-source access
- **Cross-tile memory latency**: 0/4/8 cycle routing latency based on quadrant
- **DMA-lock timing**: LockTimingState integrated into DMA engine
- **BD chaining fix**: Next_BD/Use_Next_BD correctly parsed from word5 (AM029)
- **DMA repeat count**: Task queue repeat_count support for ping-pong patterns
- **Branch delay slots**: 5-cycle delay slot implementation for correct loop control
- **100% kernel accuracy**: add_one_using_dma produces 32/32 correct outputs

**Next:** Expand to more complex kernels, validate against hardware

### Phase 2: Toolchain Integration 🔴

Plug into existing development flows: Peano, Vitis, mlir-aie, XRT.

### Phase 3: Developer Experience 🟡

Visual debugging and profiling: breakpoints, watchpoints, execution timeline, trace replay.

### Phase 4: Validation & Testing 🟡

Test infrastructure, benchmarks, CI/CD, hardware comparison.

### Phase 5: Production Readiness 🔴

Performance, multi-device support, APIs, documentation.

### Phase 6: Community & Ecosystem 🔴

Open source hygiene, community building, packaging.

---

## Resources

- **llvm-aie**: https://github.com/Xilinx/llvm-aie (instruction definitions)
- **aie-rt**: https://github.com/Xilinx/aie-rt (register definitions)
- **mlir-aie**: https://github.com/Xilinx/mlir-aie (test cases, examples)
- **XRT**: https://github.com/Xilinx/XRT (runtime API)
- **AMD Docs**: AM020 (AIE2 Architecture), AM025 (Register Reference)

---

## Detailed Documentation

| Phase | Document |
|-------|----------|
| Phase 1 | [phase1-core-accuracy.md](docs/roadmap/phase1-core-accuracy.md) |
| Phase 2 | [phase2-toolchain-integration.md](docs/roadmap/phase2-toolchain-integration.md) |
| Phase 3 | [phase3-developer-experience.md](docs/roadmap/phase3-developer-experience.md) |
| Phase 4 | [phase4-validation-testing.md](docs/roadmap/phase4-validation-testing.md) |
| Phase 5 | [phase5-production-readiness.md](docs/roadmap/phase5-production-readiness.md) |
| Phase 6 | [phase6-community-ecosystem.md](docs/roadmap/phase6-community-ecosystem.md) |