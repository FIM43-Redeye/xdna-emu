# Logic Fuzzer Design

Status: APPROVED (2026-02-28)

## Purpose

A differential fuzzer that generates random NPU programs, runs them on the
emulator and real hardware, and compares results. The goal is to prove
emulator correctness across the full space of NPU capabilities, not just
hand-written test cases.

## Architecture

Three-layer fuzzer with multiple oracles, built incrementally. Each layer
stands on its own and delivers value independently.

### Layers

| Layer | Generates | Oracle | Attacks |
|-------|-----------|--------|---------|
| 1. Scalar + Control Flow | C++ kernel: branches, ZLS loops, locks, scalar ALU | Real NPU | Control flow bugs (the historical pain point) |
| 2. Vector Operations | C++ kernel: `__builtin_aiev2_*` with valid configs | Python models (fast) + NPU (ground truth) | Vector semantic gap (Pillar 2) |
| 3. Multi-tile Integration | Parameterized IRON programs (shapes, fifos, DMA) | Real NPU + aiesimulator | DMA/routing/lock interaction bugs |

### Module Structure

```
src/fuzzer/
  mod.rs              // --fuzz flag, iteration loop, seed management
  params.rs           // FuzzParams: shapes, dtypes, ops, patterns
  kernel_gen.rs       // KernelBody AST -> C++ source
  config_space.rs     // Parse constants.py / me_chess.ctt for valid configs
  shrink.rs           // Minimize failing FuzzParams
  oracle.rs           // Compare emulator vs NPU vs Python model outputs

tools/
  fuzz_template.py    // IRON template: FuzzParams -> MLIR -> xclbin
  vector_oracle.py    // Wraps aietools Python models for reference output
```

### Integration

The fuzzer lives as a module within the existing `npu-test` binary, invoked
via `--fuzz`. It generates `FuzzCase` structs (source code + MLIR template +
buffer spec + seed) that the existing build/run/compare pipeline consumes.

```
npu-test --fuzz --iterations 1000 --seed 42
  |
  Rust: fuzzer::params        (randomize parameters)
  Rust: fuzzer::kernel_gen    (AST -> C++ kernel body)
  |
  Python: IRON template       (params -> ObjectFifo + Worker + Runtime)
  Python: aiecc.py            (MLIR -> xclbin)
  |
  Rust: existing pipeline     (emulator run, NPU run, compare)
  Rust: fuzzer::shrink        (minimize failing params on mismatch)
```

## Kernel Body AST

The AST covers only what happens inside a tile. Program structure (DMA,
routing, locks at the array level) is handled by mlir-aie's IRON API.

```
KernelBody
  +-- ops: Vec<KernelOp>
  +-- loop_style: LoopStyle    // for, hardware_loop (ZLS), nested

KernelOp (enum, grows per layer)
  // Layer 1: Scalar + Control
  +-- ScalarArith { op, dst, src1, src2 }
  +-- Branch { cond, then_ops, else_ops }
  +-- HwLoop { count, body }
  +-- LockAcquire { id, value, mode }
  +-- LockRelease { id, value }

  // Layer 2: Vector (valid configs from constants.py)
  +-- VectorArith { op, dtype, mode, dst, src1, src2 }
  +-- VectorMac { dtype, conf: ValidMacConfig }
  +-- Srs { shift, round_mode, sat_mode }
  +-- Ups { shift, sign_mode }

  // Layer 3: multi-tile params live at the IRON level, not in the AST
```

Validity constraints baked into generation:
- Every variable assigned before use
- Buffer indices in bounds (derived from buffer size param)
- Lock IDs match the IRON template's lock allocation
- Hardware loop counts always > 0
- Nesting depth bounded
- Vector configs drawn from enumerated valid values (never random i32)

C++ lowering: each KernelOp maps to 1-3 lines of C++. Layer 1 uses plain
C/C++. Layer 2 uses `__builtin_aiev2_*` builtins (both Peano and Chess
accept these). Future LLVM IR lowering uses `@llvm.aie2.*` intrinsics.

## What We Reuse

### mlir-aie IRON API (program structure)

ObjectFifo, Worker, Runtime, SequentialPlacer, Program.resolve_program().
All DMA/lock/routing handled automatically from Python parameters. The
fuzzer generates a Python script, IRON builds the MLIR, aiecc.py compiles.

### llvm-aie intrinsics (operation vocabulary)

317 AIE2 intrinsics with typed signatures. C++ builtins from
`__builtin_aiev2_*`. Parsed to enumerate valid operations.

### aietools data files (valid configuration space)

| Source | What it provides |
|--------|-----------------|
| `constants.py` (C singleton) | All valid perm_modes, mult_modes, rounding modes, vadd modes |
| `me_chess.ctt` (110K lines) | Machine-parseable enum->value table for every ISA config param |
| `me_chess_opns.h` (me_primitive) | 1,627 primitive signatures with exact bit-width param types |
| `aie_api/detail/aie2/mmul_*.hpp` | Complete valid (M,K,N,TypeA,TypeB) matrix dimension catalog |
| `testfloat/*.txt` | 2,500 pre-made BF16 conversion test vectors |

### aietools Python models (vector oracle)

| Model | Operations |
|-------|-----------|
| `mulmac.py` | vmac, vmul (all integer + bf16 modes) |
| `srs.py` | Shift-right-saturate (10 rounding modes) |
| `ups.py` | Upscale/unpack-shift |
| `pack.py` | Pack/unpack between element widths |
| `bfloat16.py` | BF16 arithmetic |
| `instr_sweep.py` | Existing fuzzer skeleton for vmac (all mode x sign combos) |

These run in pure Python/numpy with no compiled dependencies. For Layer 2,
the Python models are the fast oracle; the NPU is the cross-validation step.

### Existing emulator infrastructure

BuildEnv, XclbinSuite, InterpreterEngine, native_hw runner, comparison
harness, auto-capture of reference outputs. The fuzzer generates tests;
the existing pipeline runs them.

## Oracles

### Real NPU (ground truth for everything)

Same binary runs on emulator and NPU via XRT. Byte-for-byte DDR output
comparison. The NPU has 4 columns -- single-column fuzz tests can pack 4
per submission for 4x throughput.

### Python models (fast oracle for vector ops)

For Layer 2, the aietools Python models compute exact reference outputs
without hardware. Much faster than NPU round-trip. Use for rapid iteration;
NPU for periodic cross-validation.

### aiesimulator (optional third oracle)

Available for unit-test-style programs (needs test.cpp). Currently 6/6 pass.
Useful as a tiebreaker when emulator and NPU disagree (should never happen
but gives confidence).

## Shrinking

When a fuzz case fails (emulator != oracle), minimize the FuzzParams:
1. Remove operations from the kernel body
2. Simplify expressions (replace with literals)
3. Reduce loop counts
4. Simplify branch conditions (replace with true/false)
5. Reduce buffer sizes

At each step, recompile and re-run. If the failure persists, keep the
simplification. The output is a minimal reproducer.

## Phasing

### Phase 1: Scaffold + Scalar (Layer 1)

- Module structure, FuzzParams, seed management
- KernelBody AST with scalar ops + control flow
- C++ lowering
- Single-tile IRON template (hardcoded, not yet parameterized)
- Wire into npu-test --fuzz
- NPU oracle only
- Basic shrinking

### Phase 2: Vector Operations (Layer 2)

- config_space.rs: parse constants.py and me_chess.ctt
- Extend AST with vector ops, valid config generation
- vector_oracle.py: wrap Python models
- Dual oracle (Python models + NPU)

### Phase 3: Multi-tile (Layer 3)

- Parameterized IRON template (shapes, fifo depth, tile count, DMA patterns)
- 4-column packing for NPU throughput
- aiesimulator as optional third oracle

### Phase 4: LLVM IR Backend

- Lower KernelBody to LLVM IR with @llvm.aie2.* intrinsics
- Compile via llc instead of clang (Peano-only path)
- Enables testing instruction patterns the C++ frontend may never produce
