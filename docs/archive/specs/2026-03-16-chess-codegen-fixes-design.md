# Chess Kernel Codegen Fixes

## Purpose

Fix the four root causes of xchesscc compile failures in generated Chess
intrinsic test kernels. The goal is 100% compile success for every test
the generator produces -- if a kernel.cc is emitted, it must compile.
Anything genuinely untestable is documented with a reason and a path
toward future testability.

## Context

The Chess intrinsic validation path (spec: `2026-03-15-chess-intrinsic-
validation-design.md`) generates 7,394 test kernels from `me_chess_opns.h`.
Of those, roughly 70% fail to compile with xchesscc. The failures have
four distinct root causes, each with a targeted fix.

## Root Cause 1: Missing Type Stubs from me_iss_types.h

### Problem

200+ low-level types used in `me_chess_opns.h` function signatures are
defined in `me_iss_types.h`, not `me_chess_types.h`. Without stubs,
clang.cindex cannot resolve them, causing functions that use them to
parse with degraded type information.

Key types include: `u1`, `u2`, `u4`, `u8`, `u32`, `w32`, `w64`, `w128`,
`pmode_t` (26 bits), `smode_t` (768 bits), `mmode_t` (8 bits), `v5u1`,
`v8w64`, `v16w32`, `v16w64`, `v16w256`, `v32w32`, `v512w4`, `v512w8`,
plus many `wN_stepM` and `vNuM` variants.

### Source

`aietools/data/aie_ml/lib/isg/me_iss_types.h` -- ISG-generated header.
Every forward declaration has a structured comment:
```cpp
class u2;  // property( 2 bit unsigned )
class v8w64;  // property( vector w64[8] )
```

Full class definitions use `VBit<N, S>` / `VBitVector<elem, count>`:
```cpp
class u2 { public: typedef VBit<2, false> BitType; ... };
class v8w64 { public: typedef VBitVector<w64, 8> BitType; ... };
```

Sizes are deterministic from template parameters:
- `VBit<N, S>` -> N bits
- `VBitVector<ElemType, Count>` -> ElemType.bits * Count

### Fix

Extend `chess_type_stubs.py` to parse `me_iss_types.h`. Two parsing
strategies, in order of preference:

1. **Property comments** (simpler, less fragile): Parse the forward
   declaration comments `// property( N bit [un]signed )` and
   `// property( vector elem[count] )`. This is a one-pass regex that
   gives us all types with minimal parsing.

2. **VBit/VBitVector typedefs** (fallback): Walk class bodies for
   `typedef VBit<N, S> BitType` and `typedef VBitVector<elem, count>
   BitType`. Resolve vector element sizes recursively.

The stub generator accepts multiple input headers and merges them into
a unified type table. The existing `chessTraitsOf<T>` parser handles
`me_chess_types.h`; the new parser handles `me_iss_types.h`. Types
from `chessTraitsOf` take precedence when both sources define the same
type (the traits are the canonical reference).

**Bits-to-bytes conversion**: Use ceiling division for non-byte-aligned
types: `byte_size = max(1, (bits + 7) // 8)`. This correctly handles
`u1` (1 bit -> 1 byte), `v5u1` (5 bits -> 1 byte), and types like
`w9_step4` (9 bits -> 2 bytes). The existing `max(1, bits // 8)`
truncation bug is fixed.

**Input**: `--iss-types-header` flag added to `chess_type_stubs.py` and
`chess-test-gen.py` CLIs.

**Output**: Unified `chess_type_stubs.h` with stubs for all types from
both sources.

## Root Cause 2: Namespace Qualification

### Problem

`me_primitive` namespace functions need `me_primitive::` prefix when
called from user code. Inline wrapper functions (global namespace) do
not. The generator currently emits all calls unqualified.

Evidence: `ext_xl(arg0)` produces error "use of undeclared identifier
'ext_xl'; did you mean 'me_primitive::ext_xl'?" while `broadcast_elem(
arg0, arg1)` compiles fine unqualified.

### Fix

In `generate_chess_kernel_cc`, use namespace-conditional qualification:

```python
qualified_name = f"{namespace}::{func_name}" if namespace else func_name
```

The AST walker already tracks namespace per function (`me_primitive` or
empty string for global scope).

Note: The original spec (2026-03-15) incorrectly stated that xchesscc
resolves me_primitive functions natively without qualification. This was
disproven by actual compile testing. The original spec should be updated.

## Root Cause 3: Reference Output Parameters

### Problem

Functions like `vmac_bf_prom(..., v5u1 &)` have output reference
parameters. The current kernel template tries to cast a buffer pointer
to a reference type, which is nonsensical.

### Detection

Use clang.cindex's `TypeKind.LVALUEREF` to detect reference parameters
rather than string matching on `&` (which is fragile across clang
spelling variants). For `const T &` parameters (const-qualified lvalue
references), treat them as input parameters passed by reference for
efficiency -- NOT output parameters. Only non-const lvalue references
are outputs.

Note: Many reference output parameters in `me_chess_opns.h` are
qualified with `chess_output` (e.g., `chess_output v16int32 &v1`).
The pre-processor already strips `chess_output`, so clang sees
`v16int32 &v1`. If the void-return filter is relaxed in the future,
`chess_output`-qualified functions (like `load_lut_int32`) would need
the reference-output treatment.

### Fix

When a parameter has `TypeKind.LVALUEREF` and is NOT const-qualified:

1. Strip the reference to get the base type
2. Declare a zero-initialized local variable of the base type
3. Pass it by reference to the intrinsic call
4. After the call, write it to the output buffer alongside the return value

**Output buffer layout**: `[return_value @ offset 0][ref_out_0 @ offset
R][ref_out_1 @ offset R+S0][...]` where offsets are byte-aligned.

The `out_size` for a test becomes the sum of the return type size and all
reference output parameter sizes. The manifest records byte offset and
size for each output field so the host-side reader can extract them.

**Input buffer**: Reference output parameters are NOT read from the input
buffer (they are outputs, not inputs). The `in_size` calculation skips
them. `const T &` parameters ARE read from the input buffer.

Example generated kernel for `f(v16int32, v5u1 &) -> int`:
```cpp
#define NOCPP
#include <string.h>

extern "C" {
void test_kernel(const int32_t *restrict in, int32_t *restrict out) {
    v16int32 arg0;
    memcpy(&arg0, (const char *)in + 0, sizeof(v16int32));

    v5u1 arg1_out = {};  // output reference param

    int result = me_primitive::f(arg0, arg1_out);

    memcpy(out, &result, sizeof(int));
    memcpy((char *)out + sizeof(int), &arg1_out, sizeof(v5u1));
}
} // extern "C"
```

## Root Cause 4: Type-Aware Buffer Access

### Problem

The current cast pattern `*(const v16int32 *)(in + offset)` assumes C
pointer casting works for all Chess types. Some types (mode registers,
low-level bit-field wrappers) may not support this pattern because
xchesscc's internal representation doesn't match a simple memory load.

### Fix

Use `memcpy` (from `<string.h>`) for ALL buffer reads and writes:

```cpp
// Read from input buffer
v16int32 arg0;
memcpy(&arg0, (const char *)in + byte_offset, sizeof(v16int32));

// Write return value to output buffer
memcpy(out, &result, sizeof(return_type));
```

`memcpy` is the portable, universally safe choice. xchesscc (which uses
a Clang-based frontend internally) will optimize it away for types that
can be loaded directly. `<string.h>` is unconditionally included since
all kernels now use memcpy.

This replaces both the pointer-cast read pattern and the pointer-cast
write pattern. Byte offsets (via `(const char *)in + N`) are used
instead of int32 offsets to handle types whose sizes aren't multiples
of 4.

Note: `__builtin_memcpy` is a GCC/Clang extension that may not be
recognized by xchesscc. Use standard `memcpy` from `<string.h>`.

## Additional Considerations

### Pointer Parameters

Functions like `load_lut_int32(const void *lut, ...)` take pointer
arguments. These cannot be meaningfully filled from an input buffer
(a pointer value from the host is meaningless on the AIE core). These
functions should be classified as "untestable via this harness" with
reason `pointer_param` and documented as future work requiring a
different test strategy (e.g., allocating a lookup table in tile memory
and passing its address).

### Const Duplicate Qualification

If clang spells a parameter as `const v16int32` and the generated code
reads `const const v16int32`, this is harmless in C++ (duplicate const
is allowed). No special handling needed.

## Changes Summary

| File | Change |
|------|--------|
| `tools/chess_type_stubs.py` | Add property-comment parser for me_iss_types.h; accept multiple headers; merge type tables; fix ceiling division |
| `tools/chess-test-gen.py` | Fix namespace qualification; detect reference params via TypeKind; handle reference outputs; use memcpy for all buffer access; classify pointer params as untestable; add `--iss-types-header` flag |
| `tools/test_chess_test_gen.py` | Tests for ISS type parsing, reference param detection and codegen, memcpy patterns, namespace qualification, pointer param filtering |
| `scripts/instr-test.sh` | Pass `--iss-types-header` to generator |

## Success Criteria

1. Type stubs cover ALL types referenced in `me_chess_opns.h` function
   signatures (zero "unknown type" diagnostics from clang.cindex)
2. Namespace qualification is correct (me_primitive:: where needed)
3. Reference output parameters produce values in the output buffer
4. All generated kernel.cc files compile cleanly with xchesscc
5. Running `--chess --fail-fast` completes without failure
6. Existing tests (51 unit tests) continue to pass
7. Every declaration in me_chess_opns.h is either generated or has a
   documented, actionable skip reason in the manifest
