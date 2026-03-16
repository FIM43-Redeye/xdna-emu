# ISA-Level Validation Harness

## Purpose

Test every compute instruction in the AIE2 ISA by generating assembly
directly from the TableGen instruction set definition, bypassing xchesscc
entirely. Compare hardware and emulator outputs for each instruction to
validate emulator accuracy at the instruction level.

## Context

The Chess intrinsic validation path (specs: 2026-03-15, 2026-03-16) proved
that single-instruction test kernels can validate emulator behavior. But
xchesscc compilation takes ~20 seconds per kernel due to VLIW scheduler
startup costs, making full-suite compilation take hours.

Investigation revealed that each compiled kernel contains exactly ONE
meaningful instruction surrounded by memcpy boilerplate. The rest of the
20 seconds is wasted on VLIW scheduling of trivial code. Meanwhile,
Peano's `llvm-mc` assembler produces identical AIE2 machine code from
assembly text in microseconds.

The emulator already parses the complete AIE2 ISA from llvm-aie TableGen
definitions, including mnemonics, assembly string templates, operand types,
register classes, and scheduling latencies. This data can drive assembly
generation directly, testing the full ~600-instruction ISA rather than
just the ~4,235 Chess intrinsic variants.

## Architecture

```
TableGen ISA (already parsed by emulator)
    |
    v  (one-time Rust export)
Instruction metadata JSON
    |
    v  (Python generator)
.s assembly files (mega-programs, ~200 test points each)
    |
    v  (llvm-mc --triple=aie2, microseconds)
.o object files
    |
    v  (aiecc.py template + link + bootgen + xclbinutil, seconds)
.xclbin test binaries
    |
    v  (run on HW + EMU, diff output buffers)
Per-instruction pass/fail report
```

## Component 1: TableGen Exporter

A Rust binary (or extension to the existing emulator CLI) that dumps
instruction metadata to JSON. Fields per instruction:

- `name`: TableGen record name (e.g., `VMUL_vmul_cm_x_x_r`)
- `mnemonic`: assembly mnemonic (e.g., `vmul`)
- `asm_string`: operand template (e.g., `$cm, $x0, $x1, $r`)
- `operands`: list of `{name, type, register_kind, field_width, signed, scale}`
- `latency`: scheduling latency from itinerary (for NOP insertion)
- `has_complete_decoder`: whether encoding is unambiguous

Source: existing `src/tablegen/` parser, which already extracts all of
this. The exporter is a thin serialization layer.

Output: `tools/aie2-isa.json`, checked into the repo and refreshed when
llvm-aie updates.

## Component 2: Test Generator

Python script (`tools/isa-test-gen.py`) that reads `aie2-isa.json` and
produces assembly test programs.

### Instruction Classification

Not all instructions are testable with this harness:

| Category | Testable | Reason |
|----------|----------|--------|
| Vector ALU (vadd, vmul, vmac, ...) | Yes | Pure compute, register-to-register |
| Scalar ALU (add, sub, and, or, ...) | Yes | Pure compute |
| Shift/compare/select | Yes | Pure compute |
| Type conversion (srs, ups, ...) | Yes | Pure compute |
| Accumulator ops | Yes | Pure compute |
| Load/store | No | Need valid memory addresses (separate harness) |
| Branch/jump | No | Control flow (separate harness) |
| Lock acquire/release | No | Synchronization (separate harness) |
| Stream put/get | No | Communication (separate harness) |
| NOP variants | No | No observable output |
| Control register access | Maybe | Some are testable, some have side effects |

### Operand Generation

For each testable instruction, generate multiple test cases by varying
operands:

**Registers**: 2-3 representative registers per class to catch encoding
bugs. For example, scalar registers: r0, r15, r31. Vector 256-bit:
wl0, wl2. Vector 512-bit: x0, x2. Accumulators: cm0, cm2.

**Immediates**: Boundary values based on field width and signedness:
0, 1, max, and for signed fields, -1 and min.

**Configuration words**: For instructions with scalar config operands
(e.g., vmul's rounding/saturation mode register), test with 0 and a
representative non-zero value.

The generator also filters out pseudo-instructions (`isPseudo = 1` in
TableGen) since `llvm-mc` cannot assemble them. Of the ~600 TableGen
instruction definitions, approximately 150-250 are concrete, testable
compute instructions with fully resolved encodings.

Estimated: ~150-250 instructions x 3-5 operand combos = 450-1,250 test
points.

### Computational Coverage (Axis 2)

Operand variation tests encoding correctness. Computational correctness
requires varying the INPUT DATA. This is free: re-run the same xclbin
with different host-side input buffers. The generator produces a
deterministic PRNG-seeded input buffer; changing the seed tests new
data patterns.

Targeted edge-case data (future enhancement): zero vectors, max-value
vectors, alternating bit patterns, values near saturation thresholds.

### Assembly Template

Each test point in the mega-program follows this pattern:

```asm
// Test point N: vadd.32 x0, x1, x2
// Load input registers from input buffer
vlda wl2, [p0, #OFFSET_N]         // load lower half of x1
vlda wh2, [p0, #OFFSET_N+32]      // load upper half of x1
vlda wl4, [p0, #OFFSET_N+64]      // load x2
vlda wh4, [p0, #OFFSET_N+96]
nop                                 // pipeline latency (no scoreboard)
nop
// Execute instruction under test
vadd.32 x0, x1, x2
nop                                 // result latency
nop
// Store result to output buffer
vst wl0, [p1, #OUT_OFFSET_N]       // store lower half of x0
vst wh0, [p1, #OUT_OFFSET_N+32]    // store upper half of x0
```

The load/store register assignments avoid the destination register to
prevent data hazards. All offsets are computed at generation time and
baked into the assembly.

**VLIW bundles**: AIE2 is a VLIW architecture. Each line of assembly
produces a full 128-bit (16-byte) bundle with unused slots filled with
NOPs by the assembler. Multiple instructions on one line separated by
`;` share a bundle. The template above uses one instruction per line
(each 16 bytes) for clarity. Multi-slot packing (e.g., combining a
`vlda` with a `nop` into one bundle) is a future optimization.

**NOP insertion for pipeline hazards**: AIE2 has no hardware scoreboard.
Data produced by one instruction is not available until a fixed number
of cycles later. The NOP count between load and use (and between
compute and store) comes from the TableGen scheduling model latencies.
If the model is incomplete, a conservative default is used. The exact
latencies should be verified against xchesscc-compiled reference
kernels from the Chess path (count the NOPs xchesscc inserted between
loads and the intrinsic instruction).

**Accumulator inputs**: Instructions like `vmac` that take accumulator
inputs (not just vector inputs) require a preamble: load vectors via
`vlda`, then use `ups` (upshift) to promote into accumulator registers
before the instruction under test. The generator classifies instructions
into simple (vector/scalar inputs only) and accumulator-input (need
`ups` preamble) categories.

### Mega-Program Structure

```asm
    .text
    .globl test_kernel
test_kernel:
    // p0 = input buffer base (set by caller)
    // p1 = output buffer base (set by caller)

    // === Test point 0: vadd.32 x0, x1, x2 ===
    vlda wl2, [p0, #0]
    ...
    vadd.32 x0, x1, x2
    ...
    vst wl0, [p1, #0]
    vst wh0, [p1, #32]

    // === Test point 1: vmul cm0, x0, x1, r24 ===
    vlda wl0, [p0, #128]
    ...
    vmul cm0, x0, x1, r24
    ...
    vst amll0, [p1, #64]
    vst amlh0, [p1, #96]
    vst amhl0, [p1, #128]
    vst amhh0, [p1, #160]

    // ... ~200 more test points ...

    // Return
    ret lr
    nop
    nop
    nop
    nop
```

No function calls, no branches, no loops, no dispatch logic. Pure
straight-line compute.

### Batching

Constraints per mega-program:
- **Program memory**: 16 KB. Each line of assembly produces a 16-byte
  VLIW bundle. A typical test point is ~11 bundles = ~176 bytes
  (4 loads + 2 pre-NOPs + 1 instruction + 2 post-NOPs + 2 stores).
  Accumulator-output instructions are ~13 bundles = ~208 bytes (4 stores).
  With 16 KB, that gives **~75-90 test points** per mega-program.
- **Data memory**: 64 KB total, shared between objectfifo buffers
  (which ARE the input/output data) and stack (~1 KB). Worst case
  ~260 bytes per test point (two 512-bit vector inputs + accumulator
  output) = ~242 max. Average case ~132 bytes = ~480 max.

Program memory is the binding constraint. The generator groups
instructions by code footprint (simple vector ops pack tighter than
accumulator-heavy ops).

Estimated: **~20-40 mega-programs** for the full ISA. At microseconds
per `llvm-mc` invocation and ~20s per aiecc.py link, total compilation
is ~7-15 minutes with `-j16` parallelism on the aiecc.py step.

### Manifest

The generator writes a JSON manifest mapping each test point to its
instruction and byte offsets in the input/output buffers:

```json
{
  "batch": 0,
  "test_points": [
    {
      "instruction": "vadd.32",
      "operands": "x0, x1, x2",
      "in_offset": 0,
      "in_size": 128,
      "out_offset": 0,
      "out_size": 64
    },
    ...
  ]
}
```

This enables per-instruction failure reporting: when the output buffer
diverges, the manifest identifies exactly which instruction(s) failed.

## Component 3: Assembler and Packager

Shell script (`scripts/isa-test.sh`) orchestrating:

1. **Assemble**: `llvm-mc --triple=aie2 --filetype=obj` per mega-program.
   Microseconds each. The `.s` file uses `.text` and `.globl test_kernel`
   so the resulting `.o` has a proper ELF `.text` section with a global
   `test_kernel` symbol -- the same format `aiecc.py` expects from
   `link_with = "kernel.o"`.

2. **Link + package**: `aiecc.py` handles linking the `.o` against the
   MLIR-generated runtime startup code, producing the ELF, CDO, PDI,
   and xclbin. This is invoked once per mega-program (~20-40 times).
   The MLIR template is the same for all mega-programs with the same
   buffer sizes.

   Note: `aiecc.py` uses `--no-xchesscc --no-xbridge` (Peano linker
   mode) since the `.o` is already assembled. The Chess linker is not
   needed and was shown to segfault on multi-function objects. Peano's
   linker (`ld.lld`) handles standard ELF `.o` files produced by
   `llvm-mc`.

Total compilation time: under 15 minutes with `-j16` parallelism on
the aiecc.py link step. Assembly itself is under 1 second total.

### MLIR Template

The MLIR template for the ISA harness is simpler than the Chess path
because we use a single large buffer transfer, not per-test objectfifos:

```mlir
aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 1 : i32)
        : !aie.objectfifo<memref<Nxi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 1 : i32)
        : !aie.objectfifo<memref<Mxi32>>

    func.func private @test_kernel(memref<Nxi32>, memref<Mxi32>)
        attributes {link_with = "kernel.o"}

    aie.core(%tile_0_2) {
        %in = aie.objectfifo.acquire @of_in(Consume, 1)
            : !aie.objectfifosubview<memref<Nxi32>>
        %buf_in = aie.objectfifo.subview.access %in[0]
            : ... -> memref<Nxi32>
        %out = aie.objectfifo.acquire @of_out(Produce, 1)
            : !aie.objectfifosubview<memref<Mxi32>>
        %buf_out = aie.objectfifo.subview.access %out[0]
            : ... -> memref<Mxi32>

        func.call @test_kernel(%buf_in, %buf_out)
            : (memref<Nxi32>, memref<Mxi32>) -> ()

        aie.objectfifo.release @of_in(Consume, 1)
        aie.objectfifo.release @of_out(Produce, 1)
        aie.end
    }

    // runtime_sequence: one DMA in, one DMA out
    aie.runtime_sequence(%in : memref<Nxi32>, %buf : memref<Nxi32>,
                         %out : memref<Mxi32>) {
        ...
    }
}
```

Single acquire, single release, no loops. The simplest possible data
path to isolate compute testing from DMA complexity.

## Component 4: Runner

Extend the existing `scripts/instr-test.sh` or create a new
`scripts/isa-test.sh` with the same 5-phase structure:

1. **Generate**: `python3 tools/isa-test-gen.py`
2. **Assemble + package**: `llvm-mc` + `aiecc.py` per mega-program
3. **Run HW**: send input buffers, collect output buffers
4. **Run EMU**: same, with `XDNA_EMU=1`
5. **Compare**: binary diff per mega-program, report per-instruction
   divergences via manifest

The host binary (`test_host.cpp`) is reused from the existing harness
with minor modifications: it reads a batch manifest instead of running
one test at a time, and reports failures by instruction name.

### Seed-Based Re-Runs

The host generates input data from a PRNG with a configurable seed.
Changing the seed re-tests the full ISA with different data in seconds
(no recompilation needed). This is the computational coverage axis:

```bash
# First run
./scripts/isa-test.sh --seed 42

# Different data, same instructions
./scripts/isa-test.sh --seed 100 --no-compile
```

## Relationship to Chess Path

The Chess intrinsic validation path (`tools/chess-test-gen.py`) becomes
a cross-validation layer:

- **ISA harness** (this spec): primary validation. Tests every
  instruction. Fast compilation. Authoritative for HW vs EMU comparison.
- **Chess path**: confirms that Chess intrinsics map to expected
  instructions. Validates the Chess-specific type system and calling
  conventions. Slower but exercises the real compiler toolchain.

Both paths should eventually agree: if an instruction passes the ISA
harness but its corresponding Chess intrinsic fails, the bug is in the
Chess codegen, not the emulator.

## Success Criteria

1. TableGen exporter produces valid JSON for all ~600 AIE2 instructions
2. Generator classifies each instruction as testable or skipped with reason
3. All generated assembly files assemble without errors via `llvm-mc`
4. All mega-programs link and package into valid xclbins
5. Full ISA assembly completes in under 1 second; full link+package
   completes in under 15 minutes with `-j16`
6. HW + EMU runs complete in under 10 minutes
7. Per-instruction divergence reporting identifies failing instructions
8. Re-run with different seed requires no recompilation
9. Existing Chess path tests continue to pass

## Files

| File | Purpose |
|------|---------|
| `src/bin/export_isa.rs` (or CLI flag) | TableGen -> JSON exporter |
| `tools/aie2-isa.json` | Exported ISA metadata (checked in) |
| `tools/isa-test-gen.py` | Assembly mega-program generator |
| `tools/test_isa_test_gen.py` | Unit tests for generator |
| `scripts/isa-test.sh` | Runner script (generate/assemble/run/compare) |
| `build/isa-tests/` | Generated artifacts (not checked in) |

## Future Extensions

- **Wider encoding coverage**: test all registers in each class, not
  just representatives
- **Targeted edge-case data**: zero vectors, saturation boundaries,
  denormal floats
- **Load/store harness**: separate test suite for memory instructions
  with valid address setup
- **Branch harness**: separate test suite for control flow instructions
- **Config word enumeration**: for instructions like vmul, systematically
  test all rounding/saturation mode combinations
- **Automated regression**: CI integration, run on every emulator change
