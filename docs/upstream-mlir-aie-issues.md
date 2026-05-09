# mlir-aie Upstream Issues

Issues found during xdna-emu bridge testing (2026-04-10). These are bugs or
limitations in the mlir-aie test suite or Peano codegen, not in the emulator.

## 1. dma_complex_dims: duplicate `-k` cxxopts option (test bug)

**Status:** XFAIL upstream, but trivially fixable
**Severity:** Test crashes before touching NPU

The test at `test/npu-xrt/dma_complex_dims/test.cpp` registers `-k` as a
short option for a matrix dimension parameter:

```cpp
options.add_options()("k", "k, number of columns in the small tile",
    cxxopts::value<int>()->default_value("64"))
```

But `runtime_lib/test_lib/test_utils.cpp` line 40 already registers `-k` as
shorthand for `--kernel`:

```cpp
("kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
 cxxopts::value<std::string>())
```

This causes `cxxopts::option_already_exists` at runtime. The RUN line passes
both `-k MLIR_AIE` (kernel) and `--k 5` (dimension), confirming the conflict.

**Fix:** Change the dimension option in `test.cpp` from `"k"` to something
else (e.g., `"K"` or just remove the short option and use `--k` long form).
Note that `--k 5` in the RUN line already uses the long form.

**Files:**
- `test/npu-xrt/dma_complex_dims/test.cpp` (lines 31-32)
- `runtime_lib/test_lib/test_utils.cpp` (line 40)

## 2. add_one_ctrl_packet (3 variants): Peano codegen failure

**Status:** Not known upstream. Chess passes, Peano fails on real NPU HW.
**Severity:** Peano-compiled kernels produce wrong results

Tests affected:
- `test/npu-xrt/add_one_ctrl_packet/`
- `test/npu-xrt/add_one_ctrl_packet_4_cores/`
- `test/npu-xrt/add_one_ctrl_packet_col_overlay/`

All three show the same pattern on real hardware: first 3 of 8 output values
correct, then wrong. The Peano-compiled ELF is ~2KB vs Chess's ~11KB,
suggesting incomplete code generation for the lock-based multi-stage add
operations with control packet reconfiguration.

The kernel does:
1. Acquire locks via control packets
2. Perform 4 stages of add operations on input buffer
3. Write results to output buffer
4. Release locks

Chess handles this correctly. Peano appears to generate a stub/minimal ELF
that doesn't fully implement the lock acquisition and buffer operation
sequence.

**HW output pattern (Peano):**
```
Correct dma output 7 == 7
Correct dma output 8 == 8
Correct dma output 9 == 9
Error in dma output 8 != 10    (expected 10, got 8 -- 3-element offset)
Error in dma output 9 != 11
...
```

This is likely a Peano backend issue with control packet code generation,
not an mlir-aie IR problem (the MLIR is identical for both compilers).

**Files:**
- `test/npu-xrt/add_one_ctrl_packet/aie.mlir`
- `test/npu-xrt/add_one_ctrl_packet/test.cpp`
- Peano backend (llvm-aie) -- likely the root cause

## 3. objectfifo_repeat/distribute_repeat: known XFAIL

**Status:** Already marked `XFAIL: *` upstream -- acknowledged broken.
**Severity:** Peano-only test, fails on real NPU HW

The test at `test/npu-xrt/objectfifo_repeat/distribute_repeat/aie2.py` has
`# XFAIL: *`. It tests the combination of distribute + repeat + join across
multiple compute tiles.

On real hardware with Peano: first 18 elements correct, then wrong values
(offset by +19), then zeros. 234 of ~252 elements wrong.

No action needed unless actively working on the distribute+repeat pattern.
