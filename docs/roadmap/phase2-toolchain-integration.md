# Phase 2: Toolchain Integration

**Goal**: Plug into existing AMD XDNA development flows.

**Status**: Partial -- XRT plugin works, bridge tests run, direct compilation not integrated

**Compiler strategy**: Both Peano (open-source LLVM-based) and Chess (AMD
proprietary, from aietools) are supported for test compilation via the bridge
test suite. Direct invocation from the emulator is not yet implemented.

---

## Overview

The emulator should be a drop-in component that works with:
- **Peano** (open-source LLVM-based compiler) -- primary
- **mlir-aie** (MLIR-based flow)
- **XRT** (Xilinx Runtime)

---

## 2.1 Peano Integration

| Task | Status | Notes |
|------|--------|-------|
| Invoke `peano-clang` to compile kernel source | TODO | |
| Parse compiler output for debug info | TODO | |
| Source-level debugging (map PC to source line) | TODO | |
| Watch variables by name | TODO | |

---

## 2.2 Vitis Compatibility (Post-1.0, Optional)

Vitis/xchesscc is superseded by Peano for open-source workflows. These tasks
are deferred until after the 1.0 release. Vitis-generated xclbin files may
still work if they use standard formats, but explicit support is not a priority.

| Task | Status | Notes |
|------|--------|-------|
| Support Vitis-generated xclbin files | Deferred | May work already via standard xclbin parsing |
| Parse Vitis metadata sections | Deferred | |
| Compatible with `v++` output | Deferred | |
| aiecompiler output support | Deferred | |

---

## 2.3 mlir-aie Integration

| Task | Status | Notes |
|------|--------|-------|
| Direct integration with mlir-aie build system | TODO | |
| Run mlir-aie test suite in emulator | VERIFIED | `scripts/emu-bridge-test.sh` runs ~75 mlir-aie xclbins through XRT |
| Support objectFifo patterns | VERIFIED | `add_one_objFifo`, `objectfifo_repeat`, `dynamic_object_fifo` PASS bridge tests |
| Trace comparison with aiesimulator | TODO | (HW comparison is the primary path; aiesimulator differential is lower priority) |

---

## 2.4 XRT Integration

| Task | Status | Notes |
|------|--------|-------|
| XRT driver plugin (`xrt-plugin/`) | VERIFIED | Replaces kernel driver, delegates to Rust emulator via FFI |
| Run same host code against emulator or hardware | VERIFIED | Set `XDNA_EMU=1` to use emulator backend |
| Bridge test suite (`emu-bridge-test.sh`) | VERIFIED | Dual-compiler, HW comparison, trace sweep |
| Result comparison mode | VERIFIED | Bridge tests compare EMU vs HW output buffers |
| Device auto-detection from xclbin | TODO | |

---

## Resources

- **Peano**: https://github.com/Xilinx/llvm-aie
- **mlir-aie**: https://github.com/Xilinx/mlir-aie
- **XRT**: https://github.com/Xilinx/XRT (installed at /opt/xilinx/xrt)
- **Local mlir-aie**: `/home/triple/npu-work/mlir-aie`
