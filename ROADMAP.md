/e# Roadmap

A comprehensive plan to make xdna-emu a production-ready, open-source emulator that integrates seamlessly into AMD XDNA NPU development workflows.

## Vision

```
Developer writes kernel → Compiles with Peano/Vitis → Emulates with xdna-emu → Runs on hardware
                                                              ↓
                                                    Visual debugging, profiling,
                                                    correctness validation
```

The emulator should be a drop-in component that "just works" whether you're using:
- **Peano** (open-source LLVM-based compiler)
- **Vitis** (AMD's full toolchain)
- **mlir-aie** (MLIR-based flow)

---

## Phase 1: Core Accuracy

Make the emulator faithful to real hardware behavior.

### 1.1 Instruction Decoder
- [ ] Parse llvm-aie TableGen files programmatically
- [ ] Generate decoder from `AIE2InstrInfo.td`, `AIE2InstrFormats.td`
- [ ] Handle all instruction formats (scalar, vector, memory, control)
- [ ] VLIW bundle decoding (128-bit instruction words)
- [ ] Proper slot assignment (which operations can run in parallel)

### 1.2 Scalar Unit
- [ ] Full GPR file (32 registers)
- [ ] All ALU operations (add, sub, mul, div, shifts, logic)
- [ ] Multiply-accumulate
- [ ] Condition codes and predication
- [ ] Address generation unit

### 1.3 Vector Unit
- [ ] 256-bit vector registers
- [ ] Vector ALU (vadd, vmul, vmac, etc.)
- [ ] Shuffle/permute operations
- [ ] Accumulator registers
- [ ] Fixed-point and floating-point modes

### 1.4 Memory System
- [ ] Accurate memory timing
- [ ] Bank conflicts detection
- [ ] Memory access patterns analysis
- [ ] Stack operations

### 1.5 DMA Engine
- [ ] Multi-dimensional addressing (up to 3D)
- [ ] Stride and wrap configuration
- [ ] BD chaining with proper timing
- [ ] S2MM and MM2S channels
- [ ] Zero-copy between tiles

### 1.6 Synchronization
- [ ] Lock acquire/release with blocking
- [ ] Lock contention tracking
- [ ] Deadlock detection
- [ ] Stream switch packet routing
- [ ] Control packet handling

---

## Phase 2: Toolchain Integration

Plug into existing development flows.

### 2.1 Peano Integration
- [ ] Invoke `peano-clang` to compile kernel source
- [ ] Parse compiler output for debug info
- [ ] Source-level debugging (map PC to source line)
- [ ] Watch variables by name

### 2.2 Vitis Integration
- [ ] Support Vitis-generated xclbin files
- [ ] Parse Vitis metadata sections
- [ ] Compatible with `v++` output
- [ ] aiecompiler output support

### 2.3 mlir-aie Integration
- [ ] Direct integration with mlir-aie build system
- [ ] Run mlir-aie test suite in emulator
- [ ] Support objectFifo patterns
- [ ] Trace comparison with aiesimulator

### 2.4 XRT Integration
- [ ] Link against XRT libraries
- [ ] `xrt::device` emulation backend
- [ ] Run same host code against emulator or hardware
- [ ] Result comparison mode

---

## Phase 3: Developer Experience

Make debugging and profiling excellent.

### 3.1 Debugging
- [ ] Breakpoints (PC, memory access, lock)
- [ ] Watchpoints (memory write triggers)
- [ ] Step into/over/out
- [ ] Call stack visualization
- [ ] Register inspection with symbolic names

### 3.2 Profiling
- [ ] Cycle counts per function
- [ ] IPC (instructions per cycle) analysis
- [ ] Stall analysis (memory, lock, DMA)
- [ ] Hot path identification
- [ ] Vector utilization metrics

### 3.3 Visualization
- [ ] Execution timeline (all cores)
- [ ] DMA transfer visualization
- [ ] Lock contention graph
- [ ] Memory access heatmap
- [ ] Data flow animation

### 3.4 Trace & Replay
- [ ] Record execution trace
- [ ] Replay from trace file
- [ ] Compare traces (emulator vs hardware)
- [ ] Export to VCD/FST for waveform viewers

---

## Phase 4: Validation & Testing

Ensure correctness and maintain quality.

### 4.1 Test Infrastructure
- [ ] Import mlir-aie test suite
- [ ] Automated comparison with aiesimulator
- [ ] Hardware comparison tests (when available)
- [ ] Fuzzing for decoder robustness

### 4.2 Benchmarks
- [ ] Standard kernel benchmarks (matmul, conv2d, etc.)
- [ ] Performance regression tracking
- [ ] Emulation speed benchmarks (cycles/second)

### 4.3 Continuous Integration
- [ ] GitHub Actions for build/test
- [ ] Coverage reporting
- [ ] Benchmark dashboards
- [ ] Release automation

---

## Phase 5: Production Readiness

Polish for real-world use.

### 5.1 Performance
- [ ] JIT-compiled simulation (optional)
- [ ] Parallel core execution
- [ ] Fast mode (skip cycle accuracy for speed)
- [ ] Incremental state updates

### 5.2 Multi-Device Support
- [ ] NPU1 (Phoenix) - AIE2
- [ ] NPU2 (Strix) - AIE2P
- [ ] NPU3 (Strix Halo) - AIE2P, larger array
- [ ] NPU4 (Krackan) - AIE2P
- [ ] Device auto-detection from xclbin

### 5.3 API & Extensibility
- [ ] C API for integration
- [ ] Python bindings
- [ ] Plugin system for custom analysis
- [ ] Scripting support (Lua or Python)

### 5.4 Documentation
- [ ] User guide
- [ ] API reference
- [ ] Architecture deep-dive
- [ ] Tutorial: debugging your first kernel

---

## Phase 6: Community & Ecosystem

Build an open-source community.

### 6.1 Open Source Hygiene
- [ ] Clear contribution guidelines
- [ ] Code of conduct
- [ ] Issue/PR templates
- [ ] Good first issues labeled

### 6.2 Community Building
- [ ] Discord/Matrix for discussion
- [ ] Regular releases with changelogs
- [ ] Blog posts / tutorials
- [ ] Conference talks

### 6.3 Ecosystem Integration
- [ ] Package for major distros
- [ ] Homebrew formula
- [ ] Nix package
- [ ] Docker image

---

## Current Status

| Phase | Status |
|-------|--------|
| 1. Core Accuracy | 🟡 Basic (simplified decoder, instant DMA) |
| 2. Toolchain Integration | 🔴 Not started |
| 3. Developer Experience | 🟡 GUI exists, needs debugging features |
| 4. Validation & Testing | 🟡 86 unit tests, no integration tests |
| 5. Production Readiness | 🔴 Not started |
| 6. Community & Ecosystem | 🔴 Not started |

---

## Resources

- **llvm-aie**: https://github.com/Xilinx/llvm-aie (instruction definitions)
- **aie-rt**: https://github.com/Xilinx/aie-rt (register definitions)
- **mlir-aie**: https://github.com/Xilinx/mlir-aie (test cases, examples)
- **XRT**: https://github.com/Xilinx/XRT (runtime API)
- **AMD Docs**: AM020 (AIE2 Architecture), AM025 (Register Reference)
