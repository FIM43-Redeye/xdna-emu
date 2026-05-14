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

**Status:** Reframed 2026-05-13 -- this is bidirectional flake, not a
peano codegen bug. **Likely a Phoenix firmware silent-drop on
`MSG_OP_CHAIN_EXEC_NPU` (op 0x18)**, see
`docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`.
**Severity:** Probabilistic test failure, both compilers affected.

Tests affected:
- `test/npu-xrt/add_one_ctrl_packet/`
- `test/npu-xrt/add_one_ctrl_packet_4_cores/`
- `test/npu-xrt/add_one_ctrl_packet_col_overlay/`

### Original (2026-04-10) framing -- WRONG

We initially observed peano outputting "first 3 of 8 correct, then
wrong" with a ~2KB ELF, while chess (~11KB ELF) appeared to pass. We
read this as peano emitting a stub kernel.

### Corrected (2026-05-13) framing

HW outcomes across recent bridge runs show the chess-vs-peano
correlation is unstable:

| Date | Chess HW | Peano HW |
|---|---|---|
| 2026-05-11 | TDR | TDR |
| 2026-05-12 | PASS | PASS |
| 2026-05-13 | TDR | PASS |
| 2026-04-10 (the original observation here) | PASS | "first 3 of 8 wrong" |

Both compilers fail on bad runs; both pass on good runs. The original
"peano partial output" pattern is consistent with a probabilistic
firmware silent-drop mid-sequence: kernel runs ~3 lock cycles, then the
next `CHAIN_EXEC_NPU` response is dropped, output DMA partially
completes, downstream BO contains stale data for the remainder.

The chess ELF being ~5x larger than peano's is real but not load-bearing
-- it's because chess fully unrolls the 5 inner loops in this MLIR's
kernel body and pulls in C++ runtime machinery (atexit, cxa_finalize,
ctor table walker). When the test runs successfully, both ELFs produce
correct output.

**Real root cause (current best guess):** Phoenix NPU1 firmware (FW
1.5.5.391, protocol 5.8) probabilistically drops some
`MSG_OP_CHAIN_EXEC_NPU` responses for ctrl_packet workloads. The driver
treats the ensuing 30s timeout as completion (fake-fence-on-teardown
path) and the test PASSes silently if the kernel happened to complete
before the drop, FAILs/TDRs if not. See the finding doc above for the
full evidence chain and probe matrix.

**Files:**
- `test/npu-xrt/add_one_ctrl_packet/aie.mlir`
- `test/npu-xrt/add_one_ctrl_packet/test.cpp`
- `xdna-driver/drivers/accel/amdxdna/aie2_message.c` -- driver-side
  CHAIN_EXEC_NPU dispatch (likely innocent, just delivers the message)
- Phoenix firmware (proprietary, opaque) -- root cause

## 3. objectfifo_repeat/distribute_repeat: known XFAIL

**Status:** Already marked `XFAIL: *` upstream -- acknowledged broken.
**Severity:** Peano-only test, fails on real NPU HW

The test at `test/npu-xrt/objectfifo_repeat/distribute_repeat/aie2.py` has
`# XFAIL: *`. It tests the combination of distribute + repeat + join across
multiple compute tiles.

On real hardware with Peano: first 18 elements correct, then wrong values
(offset by +19), then zeros. 234 of ~252 elements wrong.

No action needed unless actively working on the distribute+repeat pattern.
