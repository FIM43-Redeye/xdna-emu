# Phase 2: Toolchain Integration

**Goal**: Plug into existing AMD XDNA development flows.

**Status**: Not Started

**Compiler strategy**: Peano (open-source LLVM-based) is the primary and only
supported compiler for xdna-emu workflows. Vitis/xchesscc compatibility is
deferred to post-1.0 as an optional target.

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
| Run mlir-aie test suite in emulator | TODO | |
| Support objectFifo patterns | TODO | |
| Trace comparison with aiesimulator | TODO | |

---

## 2.4 XRT Integration

| Task | Status | Notes |
|------|--------|-------|
| Link against XRT libraries | TODO | |
| `xrt::device` emulation backend | TODO | |
| Run same host code against emulator or hardware | TODO | |
| Result comparison mode | TODO | |

---

## Resources

- **Peano**: https://github.com/Xilinx/llvm-aie
- **mlir-aie**: https://github.com/Xilinx/mlir-aie
- **XRT**: https://github.com/Xilinx/XRT (installed at /opt/xilinx/xrt)
- **Local mlir-aie**: `/home/triple/npu-work/mlir-aie`
