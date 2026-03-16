# Instruction-Level Validation Harness

## Purpose

Validate the emulator's instruction decoder and execution engine against real
NPU hardware by auto-generating single-instruction test kernels from the
llvm-aie intrinsic definitions, compiling them with Chess, running on both
HW and EMU, and diffing outputs.

The real NPU is the oracle. We don't compute expected values -- we verify
that HW and EMU agree.

## Motivation

The register class mapping bugs found on 2026-03-15 (commit 402be97, 7484475)
went undetected because:

1. VCD comparison doesn't expose register contents (aiesimulator limitation)
2. Unit tests construct SlotOp directly, bypassing the decoder
3. Bridge tests exercise full programs where multiple bugs can compensate

A single-instruction test catches decoder and execution bugs at the narrowest
possible scope: one instruction, known input, one output to compare.

## Architecture

```
IntrinsicsAIE2.td (317 intrinsics, type signatures)
       |
  tools/instr-test-gen.py (parse + generate)
       |
  build/instr-tests/<name>/
       kernel.cc     -- calls one __builtin_aiev2_* with input data
       aie.mlir      -- single-tile, external func via link_with
       |
  scripts/instr-test.sh (compile + run + compare)
       |
  Per-intrinsic: PASS (HW == EMU) or FAIL (divergence)
```

## Components

### 1. Generator: `tools/instr-test-gen.py`

**Input**: `llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td`

**Parsing**: Extract from each intrinsic definition:
- Intrinsic name (e.g., `int_aie2_vbroadcast32_I512`)
- ClangBuiltin name (e.g., `__builtin_aiev2_vbroadcast32_I512`)
- Return type in LLVM IR (e.g., `llvm_v16i32_ty`)
- Argument types in LLVM IR (e.g., `[llvm_i32_ty]`)
- Attributes (`IntrNoMem`, `IntrReadMem`, etc.)

**Type mapping** (LLVM IR -> Chess C):

| LLVM IR type | Chess C type | Size (bytes) |
|---|---|---|
| `llvm_i32_ty` | `int32_t` | 4 |
| `llvm_i64_ty` / `llvm_v2i32_ty` | `int64_t` | 8 |
| `llvm_bfloat_ty` | `bfloat16` | 2 |
| `llvm_float_ty` | `float` | 4 |
| `llvm_v64i8_ty` | `v64int8` | 64 |
| `llvm_v32i16_ty` | `v32int16` | 64 |
| `llvm_v16i32_ty` | `v16int32` | 64 |
| `llvm_v32bf16_ty` | `v32bfloat16` | 64 |
| `llvm_v16bf16_ty` | `v16bfloat16` | 32 |
| `llvm_v8bf16_ty` | `v8bfloat16` | 16 |
| `llvm_v4i32_ty` | `v4int32` | 16 |
| `llvm_v8i32_ty` | `v8int32` | 32 |
| `llvm_v32i32_ty` | `v32int32` | 128 |
| `llvm_v64i16_ty` | `v64int16` | 128 |
| `llvm_v128i8_ty` | `v128int8` | 128 |
| `llvm_v4i64_ty` | `v4acc64` | 32 |
| `llvm_v8i64_ty` | `v8acc64` | 64 |
| `llvm_v16i64_ty` | `v16acc64` | 128 |
| `llvm_v8f32_ty` | `v8float` | 32 |
| `llvm_v16f32_ty` | `v16float` | 64 |
| `llvm_v32f32_ty` | `v32float` | 128 |
| `llvm_v64bf16_ty` | `v64bfloat16` | 128 |

Types requiring further research or exclusion from initial scope:
- `llvm_i128_ty`: MUL/MAC configuration word. No clean Chess C mapping.
  Exclude initially.

**Filtering**: Skip intrinsics matching any of these criteria:

| Exclusion | Reason | Example |
|---|---|---|
| No `ClangBuiltin` | Cannot call from C | `vabs_gtz8` |
| `IntrHasSideEffects` | Events, locks, done signals | `event0`, `release` |
| `IntrInaccessibleMemOnly` | Reads implicit config registers (SRS/UPS rounding/saturation modes) | BF16 MUL/MAC |
| Cascade/stream/lock | Need hardware infrastructure beyond single tile | `put_mcd`, `get_ss` |
| `UND*` intrinsics | Return undefined values by design | `undef_v16int32` |
| `llvm_i128_ty` args | Type mapping unclear | MUL/MAC config words |

Keep only `IntrNoMem` intrinsics with `ClangBuiltin` and all types in the
mapping table. The generator produces a `manifest.json` listing every
intrinsic with its status (generated / skipped + reason) for coverage
tracking.

**Output per intrinsic**: `build/instr-tests/<short_name>/kernel.cc`

```cpp
// Auto-generated: tests __builtin_aiev2_vbroadcast32_I512
// Signature: v16int32 = f(int32_t)
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20
#include <stdint.h>

extern "C" {
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {
    // Read scalar input
    int32_t arg0 = in[0];

    // Call intrinsic
    v16int32 result = __builtin_aiev2_vbroadcast32_I512(arg0);

    // Store result (copy vector to output buffer)
    v16int32 *out_vec = (v16int32 *)out;
    *out_vec = result;
}
} // extern "C"
```

For multi-argument intrinsics (e.g., VSEL takes two vectors + one scalar),
arguments are read from consecutive regions of the input buffer. Offsets
are computed dynamically from the type sizes -- no fixed layout:

```cpp
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {
    // Read vector inputs from consecutive regions of input buffer
    v16int32 arg0 = *(const v16int32 *)(in + 0);    // offset 0, 64 bytes
    v16int32 arg1 = *(const v16int32 *)(in + 16);   // offset 64, 64 bytes
    int32_t  arg2 = in[32];                          // offset 128, 4 bytes

    v16int32 result = __builtin_aiev2_vsel32(arg0, arg1, arg2);

    v16int32 *out_vec = (v16int32 *)out;
    *out_vec = result;
}
```

Buffer sizes are computed per-intrinsic from the argument type sizes (sum of
all argument sizes for input, return type size for output).

**Output (shared)**: `build/instr-tests/aie_template.mlir`

A parameterized MLIR template for a single compute tile with external kernel
function. Uses the `link_with` mechanism (same pattern as
`add_one_scale_func_link_with_chess`):

- 1 shim tile, 1 memtile, 1 compute tile
- `func.func private @test_kernel(...) attributes {link_with = "kernel.o"}`
- ObjectFIFO: input (host -> compute), output (compute -> host)
- Core body: acquire input fifo, acquire output fifo, call test_kernel
  with memref pointers, release both fifos
- `runtime_sequence` does `dma_memcpy_nd` for input and output
- Template parameters: input buffer size, output buffer size

**Output (shared)**: `build/instr-tests/test_host.cpp`

A single host test binary parameterized by command-line args:
```
./test_host -x test.xclbin -i insts.bin \
    --in-size 256 --out-size 64 --seed 42 --out-file result.bin
```

- Fills input buffer with deterministic pseudo-random data from seed
- Runs kernel
- Writes raw output to `--out-file` (not stdout, to avoid XRT debug noise)
- Exit code 0 on success, 1 on XRT error

### 2. Runner: `scripts/instr-test.sh`

**Phases**:

1. **Generate**: Run `instr-test-gen.py` to produce all kernel.cc files
   and manifest.json (skip if already generated and source hasn't changed)

2. **Compile** (parallel, cached):
   ```bash
   # Compile kernel with Chess
   xchesscc_wrapper aie2 -I ${AIETOOLS}/include -c kernel.cc -o kernel.o

   # Compile MLIR to xclbin + insts.bin (Chess linker for link_with)
   aiecc.py --no-aiesim --xchesscc --xbridge \
     --aie-generate-xclbin --aie-generate-npu-insts \
     --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie.mlir

   # Compile host harness (once, shared)
   clang++ test_host.cpp -o test_host [XRT flags]
   ```

3. **Run HW** (serial):
   ```bash
   ./test_host -x aie.xclbin -i insts.bin --seed 42 --out-file hw_out.bin
   ```

4. **Run EMU** (parallel):
   ```bash
   XDNA_EMU=1 ./test_host -x aie.xclbin -i insts.bin --seed 42 \
       --out-file emu_out.bin
   ```

5. **Compare**:
   ```bash
   cmp hw_out.bin emu_out.bin  # binary exact match
   ```

6. **Report**: Per-intrinsic pass/fail table + summary.

**Flags**:
- `--no-hw`: EMU-only (skip HW phase, no comparison)
- `--no-emu`: HW-only (baseline capture)
- `--filter <pattern>`: Run subset matching pattern
- `--seed <N>`: Override input seed
- `--compile`: Force recompilation
- `-j <N>`: Parallelism for compile + EMU phases

### 3. Input Data Generation

Deterministic PRNG from seed. Same seed -> same input on both HW and EMU.

```python
def gen_input(seed, n_bytes):
    state = seed
    buf = bytearray(n_bytes)
    for i in range(n_bytes):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        buf[i] = (state >> 16) & 0xFF
    return buf
```

The host harness implements the same PRNG in C++ and fills the input BO.
Default seed: 42. Multiple seeds for broader coverage (future work).

### 4. Manifest

The generator outputs `build/instr-tests/manifest.json`:
```json
{
  "generated": [
    {"name": "vbroadcast32", "builtin": "__builtin_aiev2_vbroadcast32_I512",
     "in_size": 4, "out_size": 64},
    ...
  ],
  "skipped": [
    {"name": "put_mcd", "reason": "cascade (needs hardware setup)"},
    {"name": "vabs_gtz8", "reason": "no ClangBuiltin"},
    {"name": "undef_v16int32", "reason": "UND (returns undefined)"},
    ...
  ]
}
```

## Scope

**In scope (initial)**:
- `IntrNoMem` intrinsics with `ClangBuiltin` and supported types
- Chess compiler only
- Single seed, single input per test
- Binary exact comparison (no tolerance)

**Out of scope (future)**:
- Multi-seed / fuzz mode
- Intrinsics with side effects (cascade, stream, lock)
- `IntrInaccessibleMemOnly` intrinsics (need config register preamble)
- `llvm_i128_ty` argument handling (MUL/MAC config words)
- Multi-return intrinsics (struct returns)
- Peano compiler variant
- Floating-point tolerance comparison
- Operation pairs (VBCST -> VEXTRACT round-trip)

## File Layout

```
tools/
  instr-test-gen.py          -- generator script
scripts/
  instr-test.sh              -- runner script
build/instr-tests/           -- generated (gitignored)
  manifest.json              -- coverage inventory
  test_host.cpp              -- shared host harness (generated)
  aie_template.mlir          -- shared MLIR template (generated)
  vbroadcast32/
    kernel.cc                -- generated
    aie.mlir                 -- filled template
  vsel32/
    kernel.cc
    aie.mlir
  ...
```

## Success Criteria

1. Generator produces compilable kernel.cc for all in-scope intrinsics
2. At least the 4 previously-buggy operations (VBCST, VSEL, VEXTRACT, VMOV)
   pass HW == EMU comparison
3. Any failures are real emulator bugs, not test harness issues
4. Adding a new test case requires zero manual work (re-run generator)
5. Manifest tracks coverage: every intrinsic is either generated or has a
   documented skip reason
