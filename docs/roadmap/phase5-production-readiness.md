# Phase 5: Production Readiness

**Goal**: Polish for real-world use.

**Status**: 🔴 Not Started

---

## Overview

Making xdna-emu production-ready means:
- Performance optimization
- Multi-device support
- Clean APIs for integration
- Comprehensive documentation

---

## 5.1 Performance

| Task | Status | Notes |
|------|--------|-------|
| JIT-compiled simulation (optional) | 🔲 TODO | |
| Parallel core execution | 🔲 TODO | |
| Fast mode (skip cycle accuracy for speed) | 🔲 TODO | |
| Incremental state updates | 🔲 TODO | |

---

## 5.2 Multi-Device Support

The project is designed to be generic from the start. We're building on **AIE2 (Phoenix)** first because it's the hardware we have, then extending to AIE2P.

| Task | Status | Notes |
|------|--------|-------|
| NPU1 (Phoenix) - AIE2 | 🟡 Primary target | Current development focus |
| NPU4 (Strix Point) - AIE2P | 🔲 TODO | Incremental from AIE2 |
| NPU5 (Strix Halo) - AIE2P, larger array | 🔲 TODO | Same as NPU4, different array size |
| NPU6 (Krackan) - AIE2P | 🔲 TODO | Same as NPU4 |
| Device auto-detection from xclbin | 🔲 TODO | |

### Device Matrix

| Device | Codename | Architecture | Array Size |
|--------|----------|--------------|------------|
| NPU1 | Phoenix/Hawk Point | AIE2 (XDNA) | 4 cols x 6 rows |
| NPU4 | Strix Point | AIE2P (XDNA2) | 4 cols x 6 rows |
| NPU5 | Strix Halo | AIE2P (XDNA2) | 8 cols x 6 rows |
| NPU6 | Krackan | AIE2P (XDNA2) | 4 cols x 6 rows |

**Note**: Driver IDs NPU2/NPU3 are prototypes marked for deprecation -- not consumer devices.
Array sizes include the shim tile row (row 0).

### Strategy

AIE2 and AIE2P share most of the architecture. Key differences are isolated:
- Additional vector instructions in AIE2P
- Slightly different DMA capabilities
- Same register model, same VLIW structure

---

## 5.3 API & Extensibility

| Task | Status | Notes |
|------|--------|-------|
| C API for integration | 🔲 TODO | |
| Python bindings | 🔲 TODO | |
| Plugin system for custom analysis | 🔲 TODO | |
| Scripting support (Lua or Python) | 🔲 TODO | |

---

## 5.4 Documentation

| Task | Status | Notes |
|------|--------|-------|
| User guide | 🔲 TODO | |
| API reference | 🔲 TODO | |
| Architecture deep-dive | 🔲 TODO | |
| Tutorial: debugging your first kernel | 🔲 TODO | |

---

## Resources

- **PyO3**: https://pyo3.rs/ (Python bindings)
- **cbindgen**: https://github.com/mozilla/cbindgen (C header generation)
