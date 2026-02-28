# Logic Fuzzer Phase 1: Scaffold + Scalar Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the fuzzer scaffold and Layer 1 (scalar + control flow) so
`npu-test --fuzz` generates random single-tile scalar kernels, compiles them,
runs them on the emulator and real NPU, and reports mismatches.

**Architecture:** New `src/fuzzer/` module with kernel body AST, random
generation, C++ lowering, and an IRON-based Python template. Integrated into
npu-test as a new RunMode. Reuses existing BuildEnv for compilation and
XclbinSuite for emulator execution.

**Tech Stack:** Rust (AST, generation, shrinking, integration), Python
(mlir-aie IRON template for program structure), aiecc.py (compilation).

**Design doc:** `docs/plans/2026-02-28-logic-fuzzer-design.md`

---

### Task 1: Create the fuzzer module skeleton

**Files:**
- Create: `src/fuzzer/mod.rs`
- Create: `src/fuzzer/params.rs`
- Create: `src/fuzzer/ast.rs`
- Modify: `src/lib.rs` (add `pub mod fuzzer;`)

**Step 1: Write the test for FuzzParams serialization**

In `src/fuzzer/params.rs`, define FuzzParams and test that it round-trips
through display/debug:

```rust
/// Parameters controlling a single fuzz iteration.
///
/// Every generated test is fully determined by these parameters plus the
/// seed. Shrinking works by simplifying these parameters.
#[derive(Debug, Clone)]
pub struct FuzzParams {
    /// RNG seed that produced this case. Enables reproducibility.
    pub seed: u64,
    /// Number of elements in the input/output buffers.
    pub buffer_size: usize,
    /// Element type for buffers.
    pub dtype: ScalarType,
    /// The kernel body to execute.
    pub body: KernelBody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I16,
    I8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::ast::*;

    #[test]
    fn test_fuzz_params_debug_contains_seed() {
        let params = FuzzParams {
            seed: 42,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody { ops: vec![], loop_style: LoopStyle::Simple },
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("42"));
    }
}
```

**Step 2: Write the AST types**

In `src/fuzzer/ast.rs`:

```rust
/// The body of a kernel function: a sequence of operations over tile data.
///
/// This AST covers only what happens inside a single tile. Program structure
/// (DMA, routing, locks at the array level) is handled by the IRON template.
#[derive(Debug, Clone)]
pub struct KernelBody {
    pub ops: Vec<KernelOp>,
    pub loop_style: LoopStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopStyle {
    /// Simple for loop over buffer elements.
    Simple,
    /// Hardware loop (ZLS/ZLE) over buffer elements.
    HardwareLoop,
}

/// A single operation in the kernel body.
#[derive(Debug, Clone)]
pub enum KernelOp {
    /// dst = src1 op src2
    ScalarArith {
        op: ScalarOp,
        dst: Var,
        src1: Operand,
        src2: Operand,
    },
    /// buf[idx] = val
    Store {
        buf: BufRef,
        idx: Operand,
        val: Operand,
    },
    /// if (cond) { then_ops } else { else_ops }
    Branch {
        cond: Operand,
        then_ops: Vec<KernelOp>,
        else_ops: Vec<KernelOp>,
    },
    /// Hardware loop with fixed count
    HwLoop {
        count: u32,
        body: Vec<KernelOp>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// A reference to a variable (temporary in the kernel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(pub u8);

/// An operand: either a variable, a literal, or a buffer load.
#[derive(Debug, Clone)]
pub enum Operand {
    Var(Var),
    Literal(i32),
    Load { buf: BufRef, idx: Box<Operand> },
}

/// Reference to input (0) or output (1) buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufRef(pub u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_body_empty_is_valid() {
        let body = KernelBody {
            ops: vec![],
            loop_style: LoopStyle::Simple,
        };
        assert!(body.ops.is_empty());
    }

    #[test]
    fn test_scalar_arith_construction() {
        let op = KernelOp::ScalarArith {
            op: ScalarOp::Add,
            dst: Var(0),
            src1: Operand::Load { buf: BufRef(0), idx: Box::new(Operand::Var(Var(1))) },
            src2: Operand::Literal(1),
        };
        // Should be constructible and debuggable
        let _ = format!("{:?}", op);
    }
}
```

**Step 3: Write mod.rs and wire into lib.rs**

`src/fuzzer/mod.rs`:
```rust
//! Differential logic fuzzer for NPU emulator validation.
//!
//! Generates random kernel programs, compiles them via Peano (and optionally
//! Chess), runs them on the emulator and real NPU hardware, and compares
//! outputs. Mismatches indicate emulator bugs.
//!
//! See `docs/plans/2026-02-28-logic-fuzzer-design.md` for the full design.

pub mod ast;
pub mod params;
```

Add to `src/lib.rs`:
```rust
pub mod fuzzer;
```

**Step 4: Run tests**

Run: `cargo test --lib -- fuzzer`
Expected: 3 tests pass (params debug, empty body, arith construction)

**Step 5: Commit**

```
git add src/fuzzer/ src/lib.rs
git commit -m "feat(fuzzer): module skeleton with AST and FuzzParams types"
```

---

### Task 2: C++ kernel lowering

**Files:**
- Create: `src/fuzzer/lower_cpp.rs`
- Modify: `src/fuzzer/mod.rs` (add `pub mod lower_cpp;`)

**Step 1: Write the test for a minimal kernel**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::ast::*;
    use crate::fuzzer::params::*;

    #[test]
    fn test_lower_empty_body_produces_passthrough() {
        let params = FuzzParams {
            seed: 1,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody { ops: vec![], loop_style: LoopStyle::Simple },
        };
        let cpp = lower_to_cpp(&params);
        // Should contain function signature, loop, and buffer copy
        assert!(cpp.contains("void fuzz_kernel("));
        assert!(cpp.contains("int32_t"));
        assert!(cpp.contains("buf_out[i] = buf_in[i]"));
    }

    #[test]
    fn test_lower_add_one() {
        let params = FuzzParams {
            seed: 2,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![
                    KernelOp::ScalarArith {
                        op: ScalarOp::Add,
                        dst: Var(0),
                        src1: Operand::Load { buf: BufRef(0), idx: Box::new(Operand::Var(Var(1))) },
                        src2: Operand::Literal(1),
                    },
                    KernelOp::Store {
                        buf: BufRef(1),
                        idx: Operand::Var(Var(1)),
                        val: Operand::Var(Var(0)),
                    },
                ],
                loop_style: LoopStyle::Simple,
            },
        };
        let cpp = lower_to_cpp(&params);
        assert!(cpp.contains("buf_in["));  // loads from input
        assert!(cpp.contains("+ 1"));      // adds literal
        assert!(cpp.contains("buf_out[")); // stores to output
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib -- fuzzer::lower_cpp`
Expected: FAIL (function not defined)

**Step 3: Implement lower_to_cpp**

```rust
//! Lower kernel AST to C++ source code.
//!
//! Produces a self-contained kernel function that both Peano and Chess
//! can compile for AIE2.

use crate::fuzzer::ast::*;
use crate::fuzzer::params::*;

/// Lower a FuzzParams to a complete C++ kernel source file.
pub fn lower_to_cpp(params: &FuzzParams) -> String {
    let mut out = String::new();
    let ctype = dtype_to_ctype(params.dtype);

    // Header
    out.push_str("#include <stdint.h>\n\n");
    out.push_str(&format!(
        "extern \"C\" void fuzz_kernel({ctype}* __restrict buf_in, {ctype}* __restrict buf_out) {{\n"
    ));

    // Loop preamble
    let n = params.buffer_size;
    match params.body.loop_style {
        LoopStyle::Simple => {
            out.push_str(&format!("    for (int i = 0; i < {}; i++) {{\n", n));
        }
        LoopStyle::HardwareLoop => {
            // chess_prepare_for_pipelining / chess_loop_range hint
            out.push_str(&format!(
                "    for (int i = 0; i < {}; i++) chess_prepare_for_pipelining {{\n", n
            ));
        }
    }

    if params.body.ops.is_empty() {
        // Default passthrough: copy input to output
        out.push_str("        buf_out[i] = buf_in[i];\n");
    } else {
        // Declare temporaries
        let max_var = max_var_id(&params.body.ops);
        for v in 0..=max_var {
            out.push_str(&format!("        {} t{} = 0;\n", ctype, v));
        }
        // Lower ops
        for op in &params.body.ops {
            lower_op(&mut out, op, "        ");
        }
    }

    out.push_str("    }\n"); // close loop
    out.push_str("}\n");     // close function
    out
}

fn dtype_to_ctype(dtype: ScalarType) -> &'static str {
    match dtype {
        ScalarType::I32 => "int32_t",
        ScalarType::I16 => "int16_t",
        ScalarType::I8 => "int8_t",
    }
}

fn max_var_id(ops: &[KernelOp]) -> u8 {
    let mut max = 0u8;
    for op in ops {
        match op {
            KernelOp::ScalarArith { dst, .. } => max = max.max(dst.0),
            KernelOp::Branch { then_ops, else_ops, .. } => {
                max = max.max(max_var_id(then_ops));
                max = max.max(max_var_id(else_ops));
            }
            KernelOp::HwLoop { body, .. } => {
                max = max.max(max_var_id(body));
            }
            _ => {}
        }
    }
    max
}

fn lower_op(out: &mut String, op: &KernelOp, indent: &str) {
    match op {
        KernelOp::ScalarArith { op: sop, dst, src1, src2 } => {
            out.push_str(&format!(
                "{}t{} = {} {} {};\n",
                indent, dst.0, lower_operand(src1), scalar_op_str(*sop), lower_operand(src2),
            ));
        }
        KernelOp::Store { buf, idx, val } => {
            let bufname = if buf.0 == 0 { "buf_in" } else { "buf_out" };
            out.push_str(&format!(
                "{}{}[{}] = {};\n",
                indent, bufname, lower_operand(idx), lower_operand(val),
            ));
        }
        KernelOp::Branch { cond, then_ops, else_ops } => {
            out.push_str(&format!("{}if ({}) {{\n", indent, lower_operand(cond)));
            let inner = format!("{}    ", indent);
            for op in then_ops {
                lower_op(out, op, &inner);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", indent));
                for op in else_ops {
                    lower_op(out, op, &inner);
                }
            }
            out.push_str(&format!("{}}}\n", indent));
        }
        KernelOp::HwLoop { count, body } => {
            out.push_str(&format!(
                "{}for (int _hw = 0; _hw < {}; _hw++) {{\n", indent, count
            ));
            let inner = format!("{}    ", indent);
            for op in body {
                lower_op(out, op, &inner);
            }
            out.push_str(&format!("{}}}\n", indent));
        }
    }
}

fn lower_operand(op: &Operand) -> String {
    match op {
        Operand::Var(v) => format!("t{}", v.0),
        Operand::Literal(n) => format!("{}", n),
        Operand::Load { buf, idx } => {
            let bufname = if buf.0 == 0 { "buf_in" } else { "buf_out" };
            format!("{}[{}]", bufname, lower_operand(idx))
        }
    }
}

fn scalar_op_str(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "+",
        ScalarOp::Sub => "-",
        ScalarOp::Mul => "*",
        ScalarOp::And => "&",
        ScalarOp::Or  => "|",
        ScalarOp::Xor => "^",
        ScalarOp::Shl => "<<",
        ScalarOp::Shr => ">>",
    }
}
```

**Step 4: Run tests**

Run: `cargo test --lib -- fuzzer::lower_cpp`
Expected: 2 tests pass

**Step 5: Commit**

```
git add src/fuzzer/lower_cpp.rs src/fuzzer/mod.rs
git commit -m "feat(fuzzer): C++ lowering from kernel AST"
```

---

### Task 3: Random kernel generation

**Files:**
- Create: `src/fuzzer/gen.rs`
- Modify: `src/fuzzer/mod.rs` (add `pub mod gen;`)

**Step 1: Write test for deterministic generation**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_seed_produces_same_params() {
        let a = generate(42);
        let b = generate(42);
        // Same seed -> same C++ output
        let cpp_a = crate::fuzzer::lower_cpp::lower_to_cpp(&a);
        let cpp_b = crate::fuzzer::lower_cpp::lower_to_cpp(&b);
        assert_eq!(cpp_a, cpp_b);
    }

    #[test]
    fn test_different_seeds_produce_different_params() {
        let a = generate(1);
        let b = generate(2);
        let cpp_a = crate::fuzzer::lower_cpp::lower_to_cpp(&a);
        let cpp_b = crate::fuzzer::lower_cpp::lower_to_cpp(&b);
        assert_ne!(cpp_a, cpp_b);
    }

    #[test]
    fn test_generated_body_has_ops() {
        let params = generate(100);
        assert!(!params.body.ops.is_empty(), "Generated kernel should have operations");
    }
}
```

**Step 2: Implement generate()**

Use a simple xorshift64 RNG (no external dependency) seeded from the input.
Generate 2-8 scalar arithmetic ops with random operations, random literals,
loads from input buffer, and stores to output buffer. The loop variable
(`i` / `Var(1)`) is always available as the index.

Key constraints:
- `Var(0)` is the accumulator (always assigned before use)
- `Var(1)` is the loop index `i` (read-only)
- All buffer accesses use `Var(1)` as index (stays in bounds)
- Literal values range from -128 to 127 (fits i8)
- Shift amounts clamped to 0-7 (no UB)
- At least one Store to buf_out at the end

**Step 3: Run tests**

Run: `cargo test --lib -- fuzzer::gen`
Expected: 3 tests pass

**Step 4: Commit**

```
git add src/fuzzer/gen.rs src/fuzzer/mod.rs
git commit -m "feat(fuzzer): random kernel generation from seed"
```

---

### Task 4: IRON Python template

**Files:**
- Create: `tools/fuzz_template.py`

**Step 1: Write the template**

This Python script takes command-line arguments (kernel .cc path, buffer
size, dtype, output directory) and uses the mlir-aie IRON API to generate
a single-tile program that feeds data in, runs the kernel, and drains
data out.

```python
#!/usr/bin/env python3
"""IRON template for fuzzer-generated kernels.

Usage:
    python fuzz_template.py --kernel kernel.cc --size 64 --dtype i32 --outdir build/fuzz/seed_42

Generates aie.mlir in the output directory, ready for aiecc.py compilation.
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# mlir-aie IRON imports
from aie.iron import Device, Program, Runtime, Worker
from aie.iron import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1

def build_program(kernel_cc: Path, buf_size: int, dtype_str: str):
    """Build a single-tile IRON program around the given kernel."""
    dtype_map = {
        "i32": np.int32,
        "i16": np.int16,
        "i8": np.int8,
    }
    np_dtype = dtype_map[dtype_str]
    tile_ty = np.ndarray[(buf_size,), np.dtype[np_dtype]]

    # Single-tile device, one input fifo, one output fifo
    with Device(NPU1Col1()) as dev:
        of_in = ObjectFifo(tile_ty, name="fuzz_in")
        of_out = ObjectFifo(tile_ty, name="fuzz_out")

        # Worker calls the external kernel function
        @Worker(of_in.cons(), of_out.prod())
        def fuzz_worker(of_in_h, of_out_h):
            elem_in = of_in_h.acquire(1)
            elem_out = of_out_h.acquire(1)
            # Call external kernel (linked from kernel.o)
            call_kernel(elem_in, elem_out)
            of_in_h.release(1)
            of_out_h.release(1)

        rt = Runtime()
        with rt.sequence(tile_ty, tile_ty) as (inp, out):
            rt.start(fuzz_worker)
            rt.fill(of_in.prod(), inp)
            rt.drain(of_out.cons(), out, wait=True)

        return Program(dev, rt)

# ... main() parses args, calls build_program, writes MLIR
```

Note: This is the aspirational template. The exact IRON API for calling
external C functions may require using the lower-level dialect API instead
(with `@core` decorator and explicit `func.call`). The first implementation
should use the simpler approach: a Python script that generates raw MLIR
text (which is what most npu-xrt tests do via their `aie2.py`), not the
full IRON API. We can upgrade later.

The practical first version generates MLIR text directly, mirroring the
pattern from `mlir-aie/test/npu-xrt/add_one_using_dma/aie2.py`. This is
simpler and guaranteed to work.

**Step 2: Test manually**

```bash
python tools/fuzz_template.py --kernel /dev/null --size 64 --dtype i32 --outdir /tmp/fuzz-test
# Should produce /tmp/fuzz-test/aie.mlir
```

**Step 3: Commit**

```
git add tools/fuzz_template.py
git commit -m "feat(fuzzer): IRON-based Python template for single-tile programs"
```

---

### Task 5: Wire --fuzz into npu-test

**Files:**
- Modify: `src/testing/runner_config.rs` (add RunMode::Fuzz, --fuzz flag,
  --iterations, --seed)
- Modify: `src/bin/npu_test.rs` (add fuzz mode dispatch)
- Create: `src/fuzzer/runner.rs` (the fuzz iteration loop)
- Modify: `src/fuzzer/mod.rs` (add `pub mod runner;`)

**Step 1: Add RunMode::Fuzz and CLI args**

In `runner_config.rs`:
- Add `Fuzz` variant to `RunMode`
- Add `fuzz_iterations: usize` and `fuzz_seed: Option<u64>` to `Options`
- Parse `--fuzz`, `--iterations N`, `--seed N` in `parse_args`

**Step 2: Create the fuzz runner**

`src/fuzzer/runner.rs` -- the main loop:

```rust
/// Run the fuzz loop: generate, compile, run on emulator + NPU, compare.
pub fn run_fuzz(opts: &Options) {
    let iterations = opts.fuzz_iterations;
    let base_seed = opts.fuzz_seed.unwrap_or_else(|| {
        // Use wall clock for random seed if none provided
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("Fuzzing {} iterations, base seed {}", iterations, base_seed);

    let build_env = BuildEnv::discover().unwrap_or_else(|e| {
        eprintln!("Build environment: {}", e);
        std::process::exit(1);
    });

    let fuzz_dir = PathBuf::from("build/fuzz");
    std::fs::create_dir_all(&fuzz_dir).ok();

    let mut pass = 0;
    let mut fail = 0;
    let mut error = 0;

    for i in 0..iterations {
        let seed = base_seed.wrapping_add(i as u64);
        let params = gen::generate(seed);
        let case_dir = fuzz_dir.join(format!("seed_{}", seed));
        std::fs::create_dir_all(&case_dir).ok();

        // 1. Lower to C++
        let cpp = lower_cpp::lower_to_cpp(&params);
        let kernel_path = case_dir.join("fuzz_kernel.cc");
        std::fs::write(&kernel_path, &cpp).unwrap();

        // 2. Generate MLIR and compile to xclbin
        //    (calls Python template + aiecc.py via BuildEnv)
        match compile_fuzz_case(&build_env, &params, &case_dir) {
            Ok(xclbin_path) => {
                // 3. Run on emulator
                let emu_output = run_emulator(&xclbin_path, &params);
                // 4. Run on NPU
                let hw_output = run_hardware(&case_dir);
                // 5. Compare
                match compare_outputs(&emu_output, &hw_output) {
                    Comparison::Match => { pass += 1; }
                    Comparison::Mismatch { .. } => {
                        fail += 1;
                        eprintln!("MISMATCH at seed {}", seed);
                        // TODO: shrink
                    }
                    Comparison::Error(e) => {
                        error += 1;
                        if opts.verbose { eprintln!("  error: {}", e); }
                    }
                }
            }
            Err(e) => {
                error += 1;
                if opts.verbose {
                    eprintln!("[{}/{}] seed {} compile error: {}", i+1, iterations, seed, e);
                }
            }
        }

        // Progress
        print!("\r[{}/{}] {} pass, {} fail, {} error", i+1, iterations, pass, fail, error);
        std::io::stdout().flush().ok();
    }
    println!(); // final newline
    println!("Fuzz complete: {} pass, {} fail, {} error", pass, fail, error);

    if fail > 0 {
        std::process::exit(1);
    }
}
```

**Step 3: Wire into npu_test.rs main()**

```rust
RunMode::Fuzz => xdna_emu::fuzzer::runner::run_fuzz(&opts),
```

**Step 4: Test the CLI plumbing**

Run: `cargo run --bin npu-test -- --fuzz --iterations 0`
Expected: "Fuzzing 0 iterations" then exits cleanly.

Run: `cargo run --bin npu-test -- --fuzz --iterations 1 --seed 42 -v`
Expected: Attempts to generate seed 42, likely fails at compile step
(template not yet wired). The scaffold works.

**Step 5: Commit**

```
git add src/fuzzer/runner.rs src/fuzzer/mod.rs src/testing/runner_config.rs src/bin/npu_test.rs
git commit -m "feat(fuzzer): wire --fuzz mode into npu-test with iteration loop"
```

---

### Task 6: End-to-end compile and run

**Files:**
- Modify: `src/fuzzer/runner.rs` (implement compile_fuzz_case,
  run_emulator, run_hardware, compare_outputs)

This task wires the real compilation and execution. It uses:
- `BuildEnv::build_npu_test()` or direct aiecc.py invocation for compilation
- `XclbinSuite::run_single()` for emulator execution
- `native_hw::run_native()` for NPU hardware execution
- Byte-for-byte output buffer comparison

**Step 1: Implement compile_fuzz_case**

Generate the MLIR template by calling the Python script, then invoke
aiecc.py. The exact implementation depends on whether we use the IRON
API or direct MLIR text generation in the Python template (Task 4).

For the first version, the simplest path is:
1. Write `fuzz_kernel.cc` (already done in runner loop)
2. Compile kernel to .o via Peano clang
3. Copy a known-working `aie.mlir` template (e.g. from add_one_using_dma)
   with buffer sizes patched
4. Run `aiecc.py --no-xchesscc --no-xbridge --no-aiesim` on it

This avoids the Python template entirely for the first iteration. We can
upgrade to generated MLIR once the loop works end-to-end.

**Step 2: Implement run_emulator**

Use `XclbinSuite` to load the xclbin, set up input buffers with known
data (e.g. 0, 1, 2, ..., N-1), run the engine, read back output.

**Step 3: Implement run_hardware**

Use `native_hw::run_test()` to execute on the real NPU with the same
input data, read back output.

**Step 4: Implement compare_outputs**

Byte-for-byte comparison of output buffers. Report first mismatch index
and values.

**Step 5: Test end-to-end**

Run (from terminal, NOT sandbox):
```bash
cargo run --release --bin npu-test -- --fuzz --iterations 1 --seed 42 -v
```
Expected: Generates a kernel, compiles, runs on emulator and NPU,
reports MATCH or MISMATCH.

**Step 6: Commit**

```
git add src/fuzzer/runner.rs
git commit -m "feat(fuzzer): end-to-end compile, emulate, and compare"
```

---

### Task 7: First real fuzz run and iteration

**Files:** (no new files -- this is testing and tuning)

**Step 1: Run 10 iterations**

```bash
cargo run --release --bin npu-test -- --fuzz --iterations 10 --seed 1 -v
```

**Step 2: Analyze results**

- How many compiled successfully?
- How many produced matching emulator/NPU output?
- What classes of failures appeared?

**Step 3: Fix any systematic issues**

Common expected issues:
- MLIR template incompatible with generated buffer sizes
- Missing `#include` in generated C++
- Buffer index out of bounds in generated ops
- Emulator timeout on certain control flow patterns

**Step 4: Run 100 iterations**

```bash
cargo run --release --bin npu-test -- --fuzz --iterations 100 -v 2>&1 | tee /tmp/fuzz-100.log
```

**Step 5: Commit any fixes**

```
git commit -am "fix(fuzzer): fixes from initial fuzz runs"
```

---

## Dependencies

```
Task 1 (skeleton)
  |
  +-> Task 2 (C++ lowering)
  |     |
  |     +-> Task 3 (random generation)
  |
  +-> Task 4 (IRON template) [independent of 2-3]
  |
  +-> Task 5 (CLI wiring) [depends on 1, independent of 2-4]
        |
        +-> Task 6 (end-to-end) [depends on 2, 3, 4, 5]
              |
              +-> Task 7 (real fuzz runs) [depends on 6]
```

Tasks 2-3 and Task 4 can proceed in parallel.
Task 5 can proceed once Task 1 is done (just needs the module to exist).

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/bin/npu_test.rs:394-411` | main() with RunMode dispatch |
| `src/testing/runner_config.rs:28-38` | RunMode enum |
| `src/testing/runner_config.rs:464-536` | Options struct |
| `src/testing/runner_config.rs:539+` | parse_args() |
| `src/testing/xclbin_suite.rs:565+` | XclbinSuite::run_single() |
| `src/testing/native_hw.rs` | NPU hardware execution |
| `src/integration/chess_build.rs:70-95` | BuildEnv struct |
| `src/testing/emu_runner.rs` | Existing test runner (reference pattern) |
| `../mlir-aie/test/npu-xrt/add_one_using_dma/` | Reference single-tile test |
| `../mlir-aie/python/iron/` | IRON API source |
