# Framework Step 2: Scalar Coverage-Driven Chain Tenant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-express the scalar fuzzer as a second coverage-driven `core::Domain` tenant with vector-grade per-op localization, while preserving the legacy trace-sweep path untouched and keeping vector fidelity byte-identical.

**Architecture:** A new `ScalarDomain` (in `src/fuzzer/domains/scalar/`) models a fuzz case as a *chain* of elementwise stages, each writing its own output region so a divergence localizes to the exact op -- the direct analogue of the vector tenant's per-slice localization. The shared engine (`core::engine::run_campaign`) drives it. The legacy single-accumulator trace-sweep path (`fuzzer::runner` + `ast`/`gen`/`lower_cpp`/`params`/`trace_sweep`) stays whole and is reached only by `fuzz --trace-sweep`; the non-trace `fuzz` path now routes through the coverage engine.

**Tech Stack:** Rust (the emulator + fuzzer engine), `serde`/`serde_json` (banking), Peano clang + aiecc (kernel compile, already wired in `core::toolchain`), aie2 scalar C kernels.

**Source spec:** `docs/superpowers/specs/2026-06-11-scalar-coverage-domain.md` (confirmed 2026-06-11). Read it before starting -- this plan implements its seven decisions.

**Hard regression bar (every task):** `cargo test --lib` stays green (baseline 3421 tests at plan start -- run it first to confirm the live number), and the vector tenant stays byte-identical: `cargo run --features tooling -- fuzz-vector --report` and a `--replay` of the banked corpus must be unchanged. The `domains/` move (Task 1) and the `dtype(case)` trait change (Task 9) are the two tasks that touch vector code paths; both must leave vector output bit-identical.

---

## Background the implementer needs

This is a from-the-vector-template build. Every new scalar module has a vector counterpart that is the authoritative shape to mirror. Point each implementer at the counterpart:

| New scalar file | Mirror this vector file | What differs |
|-----------------|-------------------------|--------------|
| `domains/scalar/chain.rs` | `domains/vector/chain.rs` | scalar stages (op + 2 operands + cond), per-case dtype/region_len/loop_style; no input pool |
| `domains/scalar/table.rs` | `domains/vector/table.rs` | 33-key universe `{feature}/{dtype}`, no emit fns (lowering is structural, not table-driven) |
| `domains/scalar/gen.rs` | `domains/vector/gen.rs` | targeted stage gen, operands draw from input/prior-stage/literal; Sequential input, no pool |
| `domains/scalar/lower.rs` | `domains/vector/lower.rs` | per-stage scalar store into region k inside an element loop; ternary branch-select; loop styles |
| `domains/scalar/domain.rs` | `domains/vector/domain.rs` | exact-byte per-region comparator (no NaN tolerance), `ScalarChainRecord` bank, vacuity via `warnings()` |
| `domains/scalar/runner.rs` | `domains/vector/runner.rs` | dispatches: `--trace-sweep` -> legacy, else `run_campaign(&ScalarDomain)` |

**Coverage universe (33 keys), spec decision 1:**
- 8 arith features `add sub mul and or xor shl shr` x 3 dtypes `I32 I16 I8` = **24** (localizable, per-region).
- 1 branch-select feature `branch` x 3 dtypes = **3** (localizable, per-region).
- 2 loop styles `loop_simple loop_hw` x 3 dtypes = **6** (case-level, credited but not a region).

Key format is `{feature}/{dtype}`, e.g. `add/I32`, `branch/I16`, `loop_hw/I8`. There is no mode dimension (unlike vector's `/m{n}`).

**Chain execution model:** the kernel loops over `region_len` element indices. Inside the loop body, stage `k` computes register `t{k}` from its operands and stores it to output region `k` at `out[k*region_len + i]`. Stage `k`'s operands draw from `in[i]`, an earlier stage register `t{j}` (`j < k`, the recency-1 structure), or a literal. Because each stage writes its own region and chains from earlier registers, the *first* differing region is the earliest broken op -- exactly vector's `first_divergent_slice` localization. The N in-loop stores crossing the loop back-edge preserve scalar's distinctive AIE2 ZOL store-flush / recency-1 catches (spec "Why a redesign" section).

**Kernel I/O contract** (same as vector, 2 args): `extern "C" void fuzz_kernel(DTYPE* __restrict in, DTYPE* __restrict out)`. The template (`tools/fuzz_template.py`, invoked by `core::toolchain::compile_kernel_case`) wires `in`=`buf_in`, `out`=`buf_out`, sized by `--size <buffer_words> --dtype <dtype>`. For scalar `buffer_words(case) = stages.len() * region_len` (the output is the larger buffer; `in` is the same size, only the first `region_len` elements are read).

---

## Task 1: Regroup `vector/` under `domains/`, create the domains module

**Files:**
- Create: `src/fuzzer/domains/mod.rs`
- Move: `src/fuzzer/vector/` -> `src/fuzzer/domains/vector/` (whole directory, `git mv`)
- Modify: `src/fuzzer/mod.rs` (swap `pub mod vector;` for `pub mod domains;`)
- Modify: `src/fuzzer/cli.rs:10` (`use crate::fuzzer::vector::runner::VecFuzzOptions;` -> `domains::vector`)
- Modify: `src/main.rs:419` (`xdna_emu::fuzzer::vector::runner::run_vector_fuzz` -> `domains::vector`)
- Search-and-fix: every `crate::fuzzer::vector::` and `super::vector::` and `fuzzer::vector::` path across the tree.

This is a pure mechanical move justified by the second tenant landing. The compiler enumerates every broken path; there is no logic change. Vector output must be byte-identical after.

- [ ] **Step 1: Move the directory and create the domains module**

```bash
cd /home/triple/npu-work/xdna-emu
git mv src/fuzzer/vector src/fuzzer/domains-vector-tmp   # two-step avoids a nested name clash
mkdir -p src/fuzzer/domains
git mv src/fuzzer/domains-vector-tmp src/fuzzer/domains/vector
```

Create `src/fuzzer/domains/mod.rs`:

```rust
//! Fuzzer domain tenants: each implements `core::Domain` and is driven by the
//! shared `core::engine::run_campaign`. The vector tenant came first (framework
//! Step 1); the scalar tenant (Step 2) is the second. Timing and trace are
//! planned as *modes* on these tenants, not separate domains.

pub mod vector;
```

- [ ] **Step 2: Update the module declaration in `fuzzer/mod.rs`**

In `src/fuzzer/mod.rs` replace the line `pub mod vector;` with `pub mod domains;`. Leave the legacy scalar modules (`ast`, `gen`, `lower_cpp`, `params`, `runner`, `trace_sweep`) exactly as they are. Result:

```rust
pub mod ast;
pub mod cli;
pub mod core;
pub mod domains;
pub mod gen;
pub mod lower_cpp;
pub mod params;
pub mod runner;
pub mod trace_sweep;
```

- [ ] **Step 3: Fix every broken import path**

Run `cargo build --features tooling 2>&1` and fix each `unresolved import` / `failed to resolve` error by rewriting `fuzzer::vector::` -> `fuzzer::domains::vector::` (and `super::vector::` -> `super::domains::vector::` where it appears from `fuzzer`-level modules). Known sites to start from: `src/fuzzer/cli.rs:10`, `src/main.rs:419`. There may be more in `src/` (tests, FFI, integration) -- let the compiler find them; do not guess. Inside the moved `vector/` files, the `super::` and `crate::fuzzer::core::` paths are unchanged (the new parent `domains` re-exports nothing they reference by `super::vector`).

Run: `cargo build --features tooling`
Expected: clean build.

- [ ] **Step 4: Run the full library test suite**

Run: `cargo test --lib`
Expected: PASS, same count as the plan-start baseline (3421 unless the live number differed). No test moved or changed -- only paths.

- [ ] **Step 5: Confirm vector fidelity is byte-identical**

Run:
```bash
cargo build --features tooling
cargo run --features tooling -- fuzz-vector --report
```
Expected: the coverage report is unchanged from before the move (universe 218, the same covered/divergent/resolved counts). The ledger and bank live under `build/fuzz-vector/` and `~/npu-work/experiments/phoenix-survival/vector/`; the move does not touch them.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fuzzer: regroup vector tenant under domains/ for the second tenant

Pure mechanical move (vector/ -> domains/vector/) ahead of the scalar
chain tenant; no logic change, vector output byte-identical.

Generated using Claude Code."
```

---

## Task 2: Scalar chain AST + coverage keys

**Files:**
- Create: `src/fuzzer/domains/scalar/mod.rs`
- Create: `src/fuzzer/domains/scalar/chain.rs`
- Modify: `src/fuzzer/domains/mod.rs` (add `pub mod scalar;`)
- Test: inline `#[cfg(test)]` in `chain.rs`

Mirror `domains/vector/chain.rs`. The scalar `Chain` carries its own dtype, region length, and loop style (vector's were table/constant-derived). Operands are an enum (vector chained implicitly through `v{k-1}`; scalar must name the source explicitly because arith ops are binary with free operands).

- [ ] **Step 1: Create the scalar module file**

Create `src/fuzzer/domains/scalar/mod.rs`:

```rust
//! The scalar fuzzer as a coverage-driven `core::Domain` tenant (framework
//! Step 2). A case is a chain of elementwise scalar stages, each writing its
//! own output region, so a divergence localizes to the exact op -- the scalar
//! analogue of the vector tenant's per-slice localization. See
//! `docs/superpowers/specs/2026-06-11-scalar-coverage-domain.md`.

pub mod chain;
```

Add `pub mod scalar;` to `src/fuzzer/domains/mod.rs` (after `pub mod vector;`).

- [ ] **Step 2: Write the failing test for the AST + keys**

Add to `src/fuzzer/domains/scalar/chain.rs` (create the file with the types stubbed enough to fail meaningfully, then the test). Write the test first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ScalarChain {
        ScalarChain {
            seed: 1,
            target_key: "add/I32".into(),
            dtype: Dtype::I32,
            region_len: 32,
            loop_style: LoopStyle::Simple,
            stages: vec![
                ScalarStage { op: StageOp::Arith(ScalarOp::Add), a: Operand::Input, b: Operand::Literal(3), cond: Operand::Input },
                ScalarStage { op: StageOp::BranchSelect, a: Operand::Prior(0), b: Operand::Literal(0), cond: Operand::Input },
            ],
        }
    }

    #[test]
    fn keys_are_stage_keys_then_loop_key() {
        let c = sample();
        assert_eq!(c.keys(), vec!["add/I32".to_string(), "branch/I32".to_string(), "loop_simple/I32".to_string()]);
    }

    #[test]
    fn loop_key_reflects_style_and_dtype() {
        let mut c = sample();
        c.loop_style = LoopStyle::HardwareLoop;
        c.dtype = Dtype::I8;
        assert_eq!(c.loop_key(), "loop_hw/I8");
    }

    #[test]
    fn out_elems_is_stages_times_region_len() {
        assert_eq!(sample().out_elems(), 2 * 32);
    }

    #[test]
    fn dtype_strings_match_template_contract() {
        assert_eq!(Dtype::I32.template_dtype(), "i32");
        assert_eq!(Dtype::I16.template_dtype(), "i16");
        assert_eq!(Dtype::I8.template_dtype(), "i8");
    }

    #[test]
    fn dtype_debug_matches_key_spelling() {
        // Keys embed the {:?} spelling; it must be the I32/I16/I8 form.
        assert_eq!(format!("{:?}", Dtype::I32), "I32");
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib domains::scalar::chain`
Expected: FAIL to compile (types not defined).

- [ ] **Step 4: Implement the AST + keys**

Prepend to `src/fuzzer/domains/scalar/chain.rs` (above the test module):

```rust
//! Chain AST for the scalar fuzzer.
//!
//! A [`ScalarChain`] is one fuzz case: N elementwise stages over a per-case
//! dtype, run inside a `region_len`-iteration element loop. Stage k writes its
//! result to output region k (`out[k*region_len + i]`); its operands draw from
//! the input buffer, an earlier stage's register (recency chain), or a literal.
//! Coverage localizes per region exactly as the vector tenant localizes per
//! 64-byte slice.

use serde::{Deserialize, Serialize};

/// Scalar element type. The `{:?}` spelling (`I32`/`I16`/`I8`) is embedded in
/// coverage keys; [`template_dtype`](Dtype::template_dtype) gives the lowercase
/// form the compile template expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    I32,
    I16,
    I8,
}

impl Dtype {
    /// C element type as spelled in the kernel.
    pub fn ctype(self) -> &'static str {
        match self {
            Dtype::I32 => "int32_t",
            Dtype::I16 => "int16_t",
            Dtype::I8 => "int8_t",
        }
    }

    /// Lowercase dtype string passed to `compile_kernel_case` / the template.
    pub fn template_dtype(self) -> &'static str {
        match self {
            Dtype::I32 => "i32",
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
        }
    }

    /// Bytes per element.
    pub fn byte_size(self) -> usize {
        match self {
            Dtype::I32 => 4,
            Dtype::I16 => 2,
            Dtype::I8 => 1,
        }
    }
}

/// How the element loop iterates. Case-level coverage dimension (the ZOL boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopStyle {
    Simple,
    HardwareLoop,
}

/// Scalar arithmetic operators (the localizable arith features).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

impl ScalarOp {
    /// Coverage-key feature name and C operator.
    pub fn feature(self) -> &'static str {
        match self {
            ScalarOp::Add => "add",
            ScalarOp::Sub => "sub",
            ScalarOp::Mul => "mul",
            ScalarOp::And => "and",
            ScalarOp::Or => "or",
            ScalarOp::Xor => "xor",
            ScalarOp::Shl => "shl",
            ScalarOp::Shr => "shr",
        }
    }

    pub fn c_operator(self) -> &'static str {
        match self {
            ScalarOp::Add => "+",
            ScalarOp::Sub => "-",
            ScalarOp::Mul => "*",
            ScalarOp::And => "&",
            ScalarOp::Or => "|",
            ScalarOp::Xor => "^",
            ScalarOp::Shl => "<<",
            ScalarOp::Shr => ">>",
        }
    }

    /// The eight arith ops, for table/universe construction.
    pub fn all() -> [ScalarOp; 8] {
        use ScalarOp::*;
        [Add, Sub, Mul, And, Or, Xor, Shl, Shr]
    }
}

/// What a stage does: a binary arith op, or an elementwise branch-select
/// (`out[i] = cond ? a : b`). Both write their own region and localize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageOp {
    Arith(ScalarOp),
    BranchSelect,
}

impl StageOp {
    /// Coverage-key feature: arith op name, or `branch`.
    pub fn feature(self) -> &'static str {
        match self {
            StageOp::Arith(op) => op.feature(),
            StageOp::BranchSelect => "branch",
        }
    }
}

/// A stage operand source. `Prior(j)` references stage j's register `t{j}`;
/// generation guarantees `j` is an earlier stage (`j < k`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operand {
    /// `in[i]` -- the input buffer element at the loop index.
    Input,
    /// `t{j}` -- an earlier stage's result register.
    Prior(usize),
    /// An integer literal.
    Literal(i32),
}

/// One chain stage: an op plus its operands. `cond` is used only by
/// `BranchSelect`; arith stages ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarStage {
    pub op: StageOp,
    pub a: Operand,
    pub b: Operand,
    pub cond: Operand,
}

/// One fuzz case: deterministic stages for `(seed, target_key)` plus the dtype,
/// region length, and loop style.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarChain {
    pub seed: u64,
    pub target_key: String,
    pub dtype: Dtype,
    /// Elements per stage output region (the element-loop trip count).
    pub region_len: usize,
    pub loop_style: LoopStyle,
    pub stages: Vec<ScalarStage>,
}

impl ScalarChain {
    /// The per-stage coverage key (`{feature}/{dtype}`), in region order.
    pub fn stage_keys(&self) -> Vec<String> {
        self.stages.iter().map(|s| format!("{}/{:?}", s.op.feature(), self.dtype)).collect()
    }

    /// The case-level loop-style coverage key.
    pub fn loop_key(&self) -> String {
        let style = match self.loop_style {
            LoopStyle::Simple => "loop_simple",
            LoopStyle::HardwareLoop => "loop_hw",
        };
        format!("{}/{:?}", style, self.dtype)
    }

    /// All keys a passing case credits: stage keys (region order) then the
    /// loop-style key. The trailing loop key is credited but is never a region
    /// (the comparator filters `loop_`-prefixed keys before localizing).
    pub fn keys(&self) -> Vec<String> {
        let mut keys = self.stage_keys();
        keys.push(self.loop_key());
        keys
    }

    /// Total output elements: one `region_len` region per stage.
    pub fn out_elems(&self) -> usize {
        self.stages.len() * self.region_len
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib domains::scalar::chain`
Expected: PASS.

- [ ] **Step 6: Run the full suite and commit**

Run: `cargo test --lib`
Expected: PASS (baseline + 5 new).

```bash
git add -A
git commit -m "fuzzer/scalar: chain AST + coverage keys for the scalar tenant

Generated using Claude Code."
```

---

## Task 3: Scalar op table + 33-key universe

**Files:**
- Create: `src/fuzzer/domains/scalar/table.rs`
- Modify: `src/fuzzer/domains/scalar/mod.rs` (add `pub mod table;`)
- Test: inline in `table.rs`

Mirror `domains/vector/table.rs::universe_keys` and `parse_key`, but the scalar "table" is the static feature list (no emit fns -- scalar lowering is structural). The universe is the 33 keys; `parse_key` resolves a `{feature}/{dtype}` target into its dtype, op kind, and loop style for generation.

- [ ] **Step 1: Write the failing tests**

Add to `src/fuzzer/domains/scalar/table.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::chain::{Dtype, ScalarOp, StageOp};
    use std::collections::HashSet;

    #[test]
    fn universe_has_33_unique_keys() {
        let u = universe_keys();
        assert_eq!(u.len(), 33, "expected 33 keys, got {}", u.len());
        let set: HashSet<_> = u.iter().collect();
        assert_eq!(set.len(), 33, "duplicate keys");
    }

    #[test]
    fn universe_is_sorted() {
        let u = universe_keys();
        let mut s = u.clone();
        s.sort();
        assert_eq!(u, s);
    }

    #[test]
    fn universe_covers_every_feature_dtype() {
        let u: HashSet<String> = universe_keys().into_iter().collect();
        for d in ["I32", "I16", "I8"] {
            for f in ["add", "sub", "mul", "and", "or", "xor", "shl", "shr", "branch", "loop_simple", "loop_hw"] {
                assert!(u.contains(&format!("{f}/{d}")), "missing {f}/{d}");
            }
        }
    }

    #[test]
    fn parse_arith_target() {
        let t = parse_key("xor/I16");
        assert_eq!(t.dtype, Dtype::I16);
        assert_eq!(t.kind, TargetKind::Stage(StageOp::Arith(ScalarOp::Xor)));
    }

    #[test]
    fn parse_branch_target() {
        let t = parse_key("branch/I8");
        assert_eq!(t.dtype, Dtype::I8);
        assert_eq!(t.kind, TargetKind::Stage(StageOp::BranchSelect));
    }

    #[test]
    fn parse_loop_targets() {
        assert_eq!(parse_key("loop_simple/I32").kind, TargetKind::LoopSimple);
        assert_eq!(parse_key("loop_hw/I32").kind, TargetKind::LoopHw);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::table`
Expected: FAIL to compile.

- [ ] **Step 3: Implement the table + parse**

Prepend to `src/fuzzer/domains/scalar/table.rs`:

```rust
//! The scalar coverage universe and target-key parsing.
//!
//! Scalar lowering is structural (not table-driven like vector), so the
//! "table" here is just the static feature list. The universe is 33 keys:
//! 8 arith + 1 branch (localizable, per-region) and 2 loop styles (case-level),
//! each crossed with 3 dtypes. Keys are `{feature}/{dtype}` -- no mode dimension.

use super::chain::{Dtype, ScalarOp, StageOp};

/// The three dtypes, in key-sort-friendly order.
const DTYPES: [Dtype; 3] = [Dtype::I32, Dtype::I16, Dtype::I8];

/// Every coverage key, sorted. 24 arith + 3 branch + 6 loop-style = 33.
pub fn universe_keys() -> Vec<String> {
    let mut keys = Vec::with_capacity(33);
    for d in DTYPES {
        for op in ScalarOp::all() {
            keys.push(format!("{}/{:?}", op.feature(), d));
        }
        keys.push(format!("branch/{:?}", d));
        keys.push(format!("loop_simple/{:?}", d));
        keys.push(format!("loop_hw/{:?}", d));
    }
    keys.sort();
    keys
}

/// What a target key asks generation to guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// Force a stage of this op kind into the chain.
    Stage(StageOp),
    /// Force the outer loop to be a simple for-loop.
    LoopSimple,
    /// Force the outer loop to be a hardware/pipelined loop.
    LoopHw,
}

/// Parsed target: dtype + what to guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub dtype: Dtype,
    pub kind: TargetKind,
}

/// Resolve `{feature}/{dtype}` to a [`Target`]. Panics on a malformed/unknown
/// key -- callers feed keys from [`universe_keys`].
pub fn parse_key(key: &str) -> Target {
    let (feature, dtype_s) = key.rsplit_once('/').unwrap_or_else(|| panic!("bad key {key:?}"));
    let dtype = match dtype_s {
        "I32" => Dtype::I32,
        "I16" => Dtype::I16,
        "I8" => Dtype::I8,
        _ => panic!("bad dtype in key {key:?}"),
    };
    let kind = match feature {
        "add" => TargetKind::Stage(StageOp::Arith(ScalarOp::Add)),
        "sub" => TargetKind::Stage(StageOp::Arith(ScalarOp::Sub)),
        "mul" => TargetKind::Stage(StageOp::Arith(ScalarOp::Mul)),
        "and" => TargetKind::Stage(StageOp::Arith(ScalarOp::And)),
        "or" => TargetKind::Stage(StageOp::Arith(ScalarOp::Or)),
        "xor" => TargetKind::Stage(StageOp::Arith(ScalarOp::Xor)),
        "shl" => TargetKind::Stage(StageOp::Arith(ScalarOp::Shl)),
        "shr" => TargetKind::Stage(StageOp::Arith(ScalarOp::Shr)),
        "branch" => TargetKind::Stage(StageOp::BranchSelect),
        "loop_simple" => TargetKind::LoopSimple,
        "loop_hw" => TargetKind::LoopHw,
        _ => panic!("unknown feature in key {key:?}"),
    };
    Target { dtype, kind }
}
```

Add `pub mod table;` to `src/fuzzer/domains/scalar/mod.rs`.

- [ ] **Step 4: Run tests, then full suite, then commit**

Run: `cargo test --lib domains::scalar::table` then `cargo test --lib`
Expected: PASS.

```bash
git add -A
git commit -m "fuzzer/scalar: 33-key coverage universe + target-key parsing

Generated using Claude Code."
```

---

## Task 4: Targeted chain generation

**Files:**
- Create: `src/fuzzer/domains/scalar/gen.rs`
- Modify: `src/fuzzer/domains/scalar/mod.rs` (add `pub mod gen;`)
- Test: inline in `gen.rs`

Mirror `domains/vector/gen.rs`. `generate(seed, target_key)` builds a deterministic chain that is guaranteed to exercise the target. Operand legality invariant: every `Prior(j)` has `j < k` (the stage's own index), so the recency chain is always backward-referencing and the lowered C compiles.

Generation recipe:
1. Parse the target. Set `dtype`. Pick `region_len` from `{16, 32, 64}` (xorshift). Set `loop_style`: `LoopSimple`/`LoopHw` target forces it; a `Stage` target picks it randomly.
2. Pick `total` stages in `8..=16` (matches vector's range; sweeps the fetch-packet body-size boundary).
3. Pick a random target slot `0..total`. At that slot emit the target op (for a `Stage` target); every other slot emits a random arith op. (A loop-style target makes every slot a random arith op -- the loop style itself is the guarantee.)
4. For each stage `k`, choose operands: `a` and `b` each from `{Input, Prior(j) for some j<k, Literal}`; `Prior` is only offered when `k>0`. For `BranchSelect`, also choose `cond` the same way. Clamp shift literals (`Shl`/`Shr` second operand) to `0..=7` so shifts are defined across all dtypes.

- [ ] **Step 1: Write the failing tests**

Add to `src/fuzzer/domains/scalar/gen.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::chain::{Dtype, LoopStyle, Operand, StageOp};
    use crate::fuzzer::domains::scalar::table::universe_keys;

    #[test]
    fn same_seed_and_key_identical_chain() {
        for key in universe_keys().iter().step_by(5) {
            assert_eq!(generate(7, key), generate(7, key), "key {key}");
        }
    }

    #[test]
    fn stage_count_in_8_to_16() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            assert!((8..=16).contains(&c.stages.len()), "seed {seed} key {key}: {} stages", c.stages.len());
        }
    }

    #[test]
    fn prior_operands_reference_only_earlier_stages() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            for (k, s) in c.stages.iter().enumerate() {
                for op in [s.a, s.b, s.cond] {
                    if let Operand::Prior(j) = op {
                        assert!(j < k, "seed {seed} key {key}: stage {k} references Prior({j})");
                    }
                }
            }
        }
    }

    #[test]
    fn arith_target_key_is_present_in_keys() {
        for key in universe_keys().iter().filter(|k| !k.starts_with("loop_")) {
            let c = generate(11, key);
            assert!(c.keys().contains(key), "key {key} not in {:?}", c.keys());
        }
    }

    #[test]
    fn loop_target_sets_the_loop_style() {
        for d in ["I32", "I16", "I8"] {
            let cs = generate(3, &format!("loop_simple/{d}"));
            assert_eq!(cs.loop_style, LoopStyle::Simple);
            let ch = generate(3, &format!("loop_hw/{d}"));
            assert_eq!(ch.loop_style, LoopStyle::HardwareLoop);
        }
    }

    #[test]
    fn dtype_follows_target() {
        assert_eq!(generate(1, "add/I8").dtype, Dtype::I8);
        assert_eq!(generate(1, "branch/I16").dtype, Dtype::I16);
    }

    #[test]
    fn shift_literals_are_bounded() {
        // Any Shl/Shr stage with a literal second operand must be in 0..=7.
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            for s in generate(seed, &key).stages {
                if matches!(s.op, StageOp::Arith(op) if matches!(op, crate::fuzzer::domains::scalar::chain::ScalarOp::Shl | crate::fuzzer::domains::scalar::chain::ScalarOp::Shr)) {
                    if let Operand::Literal(n) = s.b {
                        assert!((0..=7).contains(&n), "seed {seed} key {key}: shift literal {n}");
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::gen`
Expected: FAIL to compile.

- [ ] **Step 3: Implement generation**

Prepend to `src/fuzzer/domains/scalar/gen.rs`:

```rust
//! Deterministic, target-driven scalar chain generation.
//!
//! [`generate`] turns `(seed, target_key)` into a [`ScalarChain`] guaranteed to
//! exercise the target: an arith/branch target forces a stage of that op at a
//! random slot; a `loop_*` target forces the outer loop style. Operands draw
//! from the input buffer, an earlier stage's register, or a literal -- the
//! `Prior(j)` references are always backward (`j < k`).

use super::chain::{Dtype, LoopStyle, Operand, ScalarChain, ScalarOp, ScalarStage, StageOp};
use super::table::{parse_key, TargetKind};

/// xorshift64 PRNG (same as the vector/scalar fuzzers'): zero state forbidden.
pub(crate) struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

const REGION_LENS: [usize; 3] = [16, 32, 64];

/// Generate one chain deterministically from `(seed, target_key)`.
pub fn generate(seed: u64, target_key: &str) -> ScalarChain {
    let target = parse_key(target_key);
    let mut rng = Xorshift64(if seed == 0 { 1 } else { seed });

    let region_len = REGION_LENS[(rng.next() % REGION_LENS.len() as u64) as usize];

    let loop_style = match target.kind {
        TargetKind::LoopSimple => LoopStyle::Simple,
        TargetKind::LoopHw => LoopStyle::HardwareLoop,
        TargetKind::Stage(_) => {
            if rng.next() % 2 == 0 {
                LoopStyle::Simple
            } else {
                LoopStyle::HardwareLoop
            }
        }
    };

    let total = 8 + (rng.next() % 9) as usize; // 8-16 stages
    let target_slot = (rng.next() % total as u64) as usize;
    let forced = match target.kind {
        TargetKind::Stage(op) => Some(op),
        _ => None,
    };

    let mut stages = Vec::with_capacity(total);
    for k in 0..total {
        let op = if k == target_slot {
            forced.unwrap_or_else(|| StageOp::Arith(rand_arith(&mut rng)))
        } else {
            StageOp::Arith(rand_arith(&mut rng))
        };
        let a = rand_operand(&mut rng, k, op, false);
        let b = rand_operand(&mut rng, k, op, true);
        let cond = rand_operand(&mut rng, k, op, false);
        stages.push(ScalarStage { op, a, b, cond });
    }

    ScalarChain { seed, target_key: target_key.to_string(), dtype: target.dtype, region_len, loop_style, stages }
}

fn rand_arith(rng: &mut Xorshift64) -> ScalarOp {
    ScalarOp::all()[(rng.next() % 8) as usize]
}

/// Pick an operand for stage `k`. `is_shift_amount` clamps a literal to `0..=7`
/// when the op is a shift and this is the second (amount) operand, so shifts
/// stay defined for every dtype. `Prior` is only offered when `k > 0`.
fn rand_operand(rng: &mut Xorshift64, k: usize, op: StageOp, is_shift_amount: bool) -> Operand {
    let is_shift = matches!(op, StageOp::Arith(ScalarOp::Shl) | StageOp::Arith(ScalarOp::Shr));
    if is_shift_amount && is_shift {
        return Operand::Literal((rng.next() % 8) as i32);
    }
    match rng.next() % 3 {
        0 => Operand::Input,
        1 if k > 0 => Operand::Prior((rng.next() % k as u64) as usize),
        1 => Operand::Input,
        _ => {
            // Small signed literal; fits i8.
            Operand::Literal((rng.next() % 256) as i32 - 128)
        }
    }
}
```

Add `pub mod gen;` to `src/fuzzer/domains/scalar/mod.rs`.

- [ ] **Step 4: Run tests, full suite, commit**

Run: `cargo test --lib domains::scalar::gen` then `cargo test --lib`
Expected: PASS.

```bash
git add -A
git commit -m "fuzzer/scalar: deterministic target-driven chain generation

Generated using Claude Code."
```

---

## Task 5: Chain -> C++ lowering

**Files:**
- Create: `src/fuzzer/domains/scalar/lower.rs`
- Modify: `src/fuzzer/domains/scalar/mod.rs` (add `pub mod lower;`)
- Test: inline in `lower.rs`

Mirror `domains/vector/lower.rs`. Render the chain as a self-contained C kernel: one element loop, N in-body stages each storing to its own region. Arith stage -> `t{k} = a OP b;`; branch-select -> `t{k} = (cond) ? a : b;`. Loop style selects the for-loop spelling. The signature is the 2-arg fuzz contract.

- [ ] **Step 1: Write the failing tests**

Add to `src/fuzzer/domains/scalar/lower.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::gen::generate;
    use crate::fuzzer::domains::scalar::table::universe_keys;

    #[test]
    fn signature_and_balanced_braces_for_400_cases() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            let src = lower_chain(&c);
            let opens = src.matches('{').count();
            let closes = src.matches('}').count();
            assert_eq!(opens, closes, "seed {seed} key {key}: unbalanced braces in\n{src}");
            let sig = format!(
                "extern \"C\" void fuzz_kernel({0}* __restrict in, {0}* __restrict out)",
                c.dtype.ctype()
            );
            assert!(src.contains(&sig), "seed {seed} key {key}: missing signature");
        }
    }

    #[test]
    fn one_store_per_stage_at_region_stride() {
        let c = generate(1, "add/I32");
        let src = lower_chain(&c);
        assert_eq!(src.matches("out[").count(), c.stages.len(), "one store per stage");
        for k in 0..c.stages.len() {
            assert!(
                src.contains(&format!("out[{} + i] = t{k};", k * c.region_len)),
                "stage {k} store at region stride {}",
                k * c.region_len
            );
        }
    }

    #[test]
    fn hardware_loop_style_is_spelled() {
        let c = generate(3, "loop_hw/I32");
        assert!(lower_chain(&c).contains("chess_prepare_for_pipelining"));
        let cs = generate(3, "loop_simple/I32");
        assert!(!lower_chain(&cs).contains("chess_prepare_for_pipelining"));
    }

    #[test]
    fn branch_select_lowers_to_ternary() {
        // Find a seed/key whose chain has a branch-select stage and confirm the
        // ternary spelling appears.
        let c = generate(1, "branch/I32");
        let src = lower_chain(&c);
        assert!(src.contains(" ? "), "branch-select must lower to a ternary:\n{src}");
        assert!(src.contains(" : "), "branch-select ternary needs else arm");
    }

    #[test]
    fn arith_operators_appear() {
        let c = generate(1, "xor/I32");
        let src = lower_chain(&c);
        // The forced xor stage emits a '^'.
        assert!(src.contains(" ^ "), "xor operator missing:\n{src}");
    }

    #[test]
    fn golden_header_and_frame() {
        let c = generate(1, "add/I32");
        let src = lower_chain(&c);
        assert!(src.starts_with("// Generated scalar fuzz chain -- seed 1, target add/I32. DO NOT EDIT.\n"));
        assert!(src.ends_with("}\n"));
        assert!(src.contains(&format!("for (int i = 0; i < {}; i++)", c.region_len)));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::lower`
Expected: FAIL to compile.

- [ ] **Step 3: Implement the lowering**

Prepend to `src/fuzzer/domains/scalar/lower.rs`:

```rust
//! Chain -> C kernel lowering.
//!
//! [`lower_chain`] renders a [`ScalarChain`] as a self-contained scalar kernel
//! with the fuzz template's 2-arg I/O contract:
//! `extern "C" void fuzz_kernel(DTYPE* __restrict in, DTYPE* __restrict out)`.
//! The kernel loops over `region_len` element indices; inside the body each
//! stage k computes register `t{k}` and stores it to its own region at
//! `out[k*region_len + i]`. Stage k's operands come from `in[i]`, an earlier
//! register `t{j}` (j<k), or a literal -- the in-body stores crossing the loop
//! back-edge preserve the AIE2 ZOL store-flush / recency catches.

use super::chain::{LoopStyle, Operand, ScalarChain, ScalarStage, StageOp};

/// Render an operand as a C expression.
fn operand_expr(op: Operand) -> String {
    match op {
        Operand::Input => "in[i]".to_string(),
        Operand::Prior(j) => format!("t{j}"),
        Operand::Literal(n) => format!("{n}"),
    }
}

/// Render one stage's assignment to its register `t{k}`.
fn stage_expr(stage: &ScalarStage) -> String {
    let a = operand_expr(stage.a);
    let b = operand_expr(stage.b);
    match stage.op {
        StageOp::Arith(op) => format!("{a} {} {b}", op.c_operator()),
        StageOp::BranchSelect => {
            let cond = operand_expr(stage.cond);
            format!("({cond}) ? {a} : {b}")
        }
    }
}

/// Lower a chain to complete C kernel source.
pub fn lower_chain(chain: &ScalarChain) -> String {
    let ctype = chain.dtype.ctype();
    let mut s = String::new();
    s.push_str(&format!(
        "// Generated scalar fuzz chain -- seed {}, target {}. DO NOT EDIT.\n",
        chain.seed, chain.target_key
    ));
    s.push_str("#include <stdint.h>\n\n");
    s.push_str(&format!(
        "extern \"C\" void fuzz_kernel({ctype}* __restrict in, {ctype}* __restrict out) {{\n"
    ));

    let loop_open = match chain.loop_style {
        LoopStyle::Simple => format!("    for (int i = 0; i < {}; i++) {{\n", chain.region_len),
        LoopStyle::HardwareLoop => {
            format!("    for (int i = 0; i < {}; i++) chess_prepare_for_pipelining {{\n", chain.region_len)
        }
    };
    s.push_str(&loop_open);

    for (k, stage) in chain.stages.iter().enumerate() {
        s.push_str(&format!("        {ctype} t{k} = {};\n", stage_expr(stage)));
        s.push_str(&format!("        out[{} + i] = t{k};\n", k * chain.region_len));
    }

    s.push_str("    }\n"); // close loop
    s.push_str("}\n"); // close function
    s
}
```

Add `pub mod lower;` to `src/fuzzer/domains/scalar/mod.rs`.

- [ ] **Step 4: Run tests, full suite, commit**

Run: `cargo test --lib domains::scalar::lower` then `cargo test --lib`
Expected: PASS.

```bash
git add -A
git commit -m "fuzzer/scalar: chain -> C kernel lowering (per-stage region stores)

Generated using Claude Code."
```

---

## Task 6: ScalarObs, buffer spec, and `observe`

**Files:**
- Create: `src/fuzzer/domains/scalar/domain.rs`
- Modify: `src/fuzzer/domains/scalar/mod.rs` (add `pub mod domain;`)
- Test: inline in `domain.rs`

This task builds the observation type, the buffer spec, and the `observe` execution path (EMU + HW). It does NOT yet implement the `Domain` trait (that is Task 9, after the comparator and bank exist). Mirror `domains/vector/domain.rs` lines 198-343 (`make_vec_buffer_spec`, the `observe` body) but with scalar dtype and Sequential input.

- [ ] **Step 1: Write the failing tests**

Add to `src/fuzzer/domains/scalar/domain.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::gen::generate;
    use crate::testing::test_cpp_parser::{BufferDir, ElementType, InputPattern};

    #[test]
    fn buffer_spec_has_sequential_input_and_zero_out() {
        let c = generate(1, "add/I8");
        let spec = make_scalar_buffer_spec(&c);
        assert_eq!(spec.buffers.len(), 4); // in, scratch, out, trace
        let buf_in = &spec.buffers[0];
        assert_eq!(buf_in.direction, BufferDir::Input);
        assert_eq!(buf_in.element_type, ElementType::I8);
        assert_eq!(buf_in.input_pattern, InputPattern::Sequential { start: 1, step: 1 });
        assert_eq!(buf_in.size_elements, c.out_elems());
        assert_eq!(spec.buffers[2].direction, BufferDir::Output);
        assert_eq!(spec.buffers[2].input_pattern, InputPattern::Zeros);
    }

    #[test]
    fn buffer_words_is_out_elems() {
        let c = generate(2, "add/I32");
        assert_eq!(buffer_words(&c), c.out_elems());
    }

    #[test]
    fn element_type_maps_each_dtype() {
        use crate::fuzzer::domains::scalar::chain::Dtype;
        assert_eq!(elem_type(Dtype::I32), ElementType::I32);
        assert_eq!(elem_type(Dtype::I16), ElementType::I16);
        assert_eq!(elem_type(Dtype::I8), ElementType::I8);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::domain`
Expected: FAIL to compile.

- [ ] **Step 3: Implement ScalarObs, buffer spec, observe**

Prepend to `src/fuzzer/domains/scalar/domain.rs`:

```rust
//! The scalar fuzzer as a `core::Domain` tenant. Owns chain execution on
//! EMU/HW, the exact-byte per-region comparator, and durable bank/replay.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::toolchain::TRACE_BUFFER_ELEMENTS;
use crate::fuzzer::domains::scalar::chain::{Dtype, ScalarChain};
use crate::fuzzer::domains::scalar::gen::generate;
use crate::fuzzer::domains::scalar::lower::lower_chain;
use crate::fuzzer::domains::scalar::table::universe_keys;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// One execution's result: the full output buffer (N regions concatenated) and
/// an optional binary trace. Scalar has no executed-op recorder (vector's
/// `fuzz_recorder` tracks vector ops); folding/vacuity is detected from the
/// output being all-zero in [`warnings`](ScalarDomain::warnings).
pub struct ScalarObs {
    pub output: Vec<u8>,
    pub trace: Option<Vec<u8>>,
}

/// The scalar fuzzer tenant. Zero-sized: all state is per-case.
pub struct ScalarDomain;

/// Buffer size in dtype elements: the output spans `stages * region_len`; the
/// input is sized the same (only the first `region_len` elements are read).
pub(crate) fn buffer_words(chain: &ScalarChain) -> usize {
    chain.out_elems()
}

/// Map a scalar dtype to the buffer-spec element type.
pub(crate) fn elem_type(dtype: Dtype) -> ElementType {
    match dtype {
        Dtype::I32 => ElementType::I32,
        Dtype::I16 => ElementType::I16,
        Dtype::I8 => ElementType::I8,
    }
}

/// Buffer spec for a scalar chain: Sequential input (1,2,3,...), zero scratch,
/// zero output, standard trace buffer. All data buffers are `out_elems` wide.
pub(crate) fn make_scalar_buffer_spec(chain: &ScalarChain) -> BufferSpec {
    let words = buffer_words(chain);
    let et = elem_type(chain.dtype);
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_scratch".to_string(),
                group_id: 4,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 5,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_trace".to_string(),
                group_id: 6,
                size_elements: TRACE_BUFFER_ELEMENTS,
                element_type: ElementType::I32,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

/// Execute a compiled chain on `backend`, returning a [`ScalarObs`].
pub(crate) fn observe_impl(
    backend: Backend,
    xclbin: &Path,
    insts: &Path,
    chain: &ScalarChain,
    max_cycles: u64,
) -> Result<ScalarObs, String> {
    match backend {
        Backend::Interpreter => {
            let spec = make_scalar_buffer_spec(chain);
            let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
            let suite = XclbinSuite::new().with_max_cycles(max_cycles);
            let (outcome, raw_output, trace) = suite.run_single_with_trace(&test);
            if !outcome.is_pass() {
                return Err(format!("emulator outcome not pass: {outcome:?}"));
            }
            let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
            Ok(ScalarObs { output, trace })
        }
        Backend::Hardware => {
            use crate::testing::npu_runner;
            if !npu_runner::npu_available() {
                return Err("NPU hardware not available".into());
            }
            let spec = make_scalar_buffer_spec(chain);
            let test_name = format!("scalarfuzz_seed_{}", chain.seed);
            match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                Ok(result) => Ok(ScalarObs {
                    output: result.output,
                    trace: result.extra_outputs.get("buf_trace").cloned(),
                }),
                Err(e) => Err(format!("{:?}", e)),
            }
        }
        Backend::Aiesim => Err("aiesim backend not wired for the scalar domain (Step 2)".into()),
    }
}
```

Add `pub mod domain;` to `src/fuzzer/domains/scalar/mod.rs`.

Note: confirm `BufferDir` and `ElementType` derive `PartialEq` (the vector tests compare `input_pattern`, so they do). If a test needs an unimported item, import it.

- [ ] **Step 4: Run tests, full suite, commit**

Run: `cargo test --lib domains::scalar::domain` then `cargo test --lib`
Expected: PASS.

```bash
git add -A
git commit -m "fuzzer/scalar: ScalarObs, buffer spec, EMU/HW observe

Generated using Claude Code."
```

---

## Task 7: Per-region comparator + vacuity warning

**Files:**
- Modify: `src/fuzzer/domains/scalar/domain.rs` (add comparator fns + tests)

The comparator is exact-byte (no NaN tolerance -- scalar is integer). It splits both output buffers into N equal regions where N = the number of region keys (the coverage keys not prefixed `loop_`), and returns the first differing region's key. This filtering is what keeps the case-level loop key out of localization while still being credited (spec decisions 1 + 3). Vacuity (all-zero output) is surfaced via `warnings`, not a trait change (spec decision 4).

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/fuzzer/domains/scalar/domain.rs`:

```rust
    #[test]
    fn region_keys_filter_drops_loop_keys() {
        let keys = vec!["add/I32".to_string(), "branch/I32".to_string(), "loop_hw/I32".to_string()];
        assert_eq!(region_keys(&keys), vec!["add/I32".to_string(), "branch/I32".to_string()]);
    }

    #[test]
    fn first_divergent_region_localizes_to_the_changed_region() {
        // 3 regions of 8 bytes; corrupt a byte in region 1.
        let n = 3;
        let emu = vec![0xAAu8; n * 8];
        let mut hw = emu.clone();
        hw[8 + 3] ^= 0x40;
        assert_eq!(first_divergent_region(&emu, &hw, n), Some(1));
    }

    #[test]
    fn equal_buffers_do_not_diverge() {
        let emu = vec![0x5Au8; 4 * 16];
        assert_eq!(first_divergent_region(&emu, &emu.clone(), 4), None);
    }

    #[test]
    fn length_mismatch_diverges() {
        let emu = vec![0u8; 3 * 8];
        let hw = vec![0u8; 2 * 8];
        assert!(first_divergent_region(&emu, &hw, 3).is_some());
    }

    #[test]
    fn compare_maps_region_to_its_stage_key() {
        let dom = ScalarDomain;
        let keys = vec!["add/I32".to_string(), "sub/I32".to_string(), "loop_simple/I32".to_string()];
        let emu = ScalarObs { output: vec![1u8; 2 * 8], trace: None };
        let mut hwout = emu.output.clone();
        hwout[8] ^= 1; // region 1
        let hw = ScalarObs { output: hwout, trace: None };
        assert_eq!(dom.compare(&emu, &hw, &keys), Some("sub/I32".to_string()));
    }

    #[test]
    fn vacuous_all_zero_output_warns() {
        let dom = ScalarDomain;
        let obs = ScalarObs { output: vec![0u8; 64], trace: None };
        assert!(!dom.warnings(&obs).is_empty(), "all-zero output should warn vacuous");
        let nonzero = ScalarObs { output: vec![1u8; 64], trace: None };
        assert!(dom.warnings(&nonzero).is_empty());
    }
```

Note: `compare` and `warnings` are `Domain` trait methods. To call them before Task 9 wires the trait, add temporary inherent methods now and convert them to the trait impl in Task 9, OR implement the free functions `region_keys` / `first_divergent_region` now and defer the `dom.compare`/`dom.warnings` assertions until Task 9. **Chosen approach:** implement `region_keys` and `first_divergent_region` as free functions in this task with their two tests (`region_keys_filter_drops_loop_keys`, `first_divergent_region_*`); move the `compare_maps_*` and `vacuous_*` tests into Task 9 where `impl Domain` exists. Keep this task's test set to the free-function tests.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::domain`
Expected: FAIL to compile (free fns not defined).

- [ ] **Step 3: Implement the free functions**

Add to `src/fuzzer/domains/scalar/domain.rs` (above the test module):

```rust
/// The region (localizable) keys: coverage keys that are not the case-level
/// loop-style key. Order-preserving, so `region_keys(keys)[k]` is stage k's key.
pub(crate) fn region_keys(keys: &[String]) -> Vec<String> {
    keys.iter().filter(|k| !k.starts_with("loop_")).cloned().collect()
}

/// First differing output region between two buffers split into `n` equal
/// regions, or `None` if equal. Exact-byte (scalar is integer; no tolerance).
/// A length mismatch counts as a divergence at the first region past the
/// common length.
pub(crate) fn first_divergent_region(emu: &[u8], hw: &[u8], n: usize) -> Option<usize> {
    if n == 0 {
        return if emu == hw { None } else { Some(0) };
    }
    let common = emu.len().min(hw.len());
    let region_bytes = (common / n).max(1);
    for r in 0..n {
        let start = r * region_bytes;
        let end = ((r + 1) * region_bytes).min(common);
        if start >= end {
            break;
        }
        if emu[start..end] != hw[start..end] {
            return Some(r);
        }
    }
    if emu.len() != hw.len() {
        return Some((common / region_bytes).min(n.saturating_sub(1)));
    }
    None
}
```

- [ ] **Step 4: Run the free-function tests, full suite, commit**

Run: `cargo test --lib domains::scalar::domain` then `cargo test --lib`
Expected: PASS (the `compare_maps_*` and `vacuous_*` tests are deferred to Task 9).

```bash
git add -A
git commit -m "fuzzer/scalar: per-region exact-byte localization helpers

Generated using Claude Code."
```

---

## Task 8: Banking + replay (`ScalarChainRecord`)

**Files:**
- Modify: `src/fuzzer/domains/scalar/domain.rs` (add record + bank/load free fns + tests)

Mirror `domains/vector/domain.rs` `ChainRecord`/`bank_case`/`load_banked`, but the scalar AST is self-describing (the whole chain serializes), so the record is just the serialized chain plus the banked keys and HW output. No input pool to bank (inputs are formulaic Sequential, spec decision 6). The banked chain is the source of truth; replay deserializes -> lowers -> recompiles -> runs EMU -> compares.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `domain.rs`:

```rust
    #[test]
    fn chain_record_round_trips_through_json() {
        let c = generate(5, "mul/I16");
        let record = ScalarChainRecord::from_chain(&c, &c.keys());
        let json = serde_json::to_string(&record).unwrap();
        let loaded: ScalarChainRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.chain, c);
        assert_eq!(loaded.keys, c.keys());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib domains::scalar::domain::tests::chain_record`
Expected: FAIL to compile.

- [ ] **Step 3: Implement the record + bank/load**

Add to `domain.rs` (above the test module):

```rust
/// Serialized form of a banked scalar chain. The chain AST is self-describing
/// (no live table, no input pool -- inputs are formulaic Sequential), so the
/// record is the chain itself plus the banked coverage keys (for localization)
/// and the silicon output.
#[derive(Serialize, Deserialize)]
pub(crate) struct ScalarChainRecord {
    pub(crate) chain: ScalarChain,
    pub(crate) keys: Vec<String>,
}

impl ScalarChainRecord {
    pub(crate) fn from_chain(chain: &ScalarChain, keys: &[String]) -> Self {
        Self { chain: chain.clone(), keys: keys.to_vec() }
    }
}

/// Bank a divergent/crashed chain under `phoenix-survival/scalar/seed_N/`.
pub(crate) fn bank_case(
    case_dir: &Path,
    chain: &ScalarChain,
    keys: &[String],
    npu_output: Option<&[u8]>,
) -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bank_dir =
        PathBuf::from(home).join(format!("npu-work/experiments/phoenix-survival/scalar/seed_{}", chain.seed));
    std::fs::create_dir_all(&bank_dir).map_err(|e| format!("create {}: {e}", bank_dir.display()))?;

    std::fs::copy(case_dir.join("fuzz_kernel.cc"), bank_dir.join("fuzz_kernel.cc"))
        .map_err(|e| format!("copy fuzz_kernel.cc: {e}"))?;

    let record = ScalarChainRecord::from_chain(chain, keys);
    let json = serde_json::to_string_pretty(&record).map_err(|e| format!("serialize chain: {e}"))?;
    std::fs::write(bank_dir.join("chain.json"), json).map_err(|e| format!("write chain.json: {e}"))?;

    if let Some(out) = npu_output {
        std::fs::write(bank_dir.join("npu_output.bin"), out).map_err(|e| format!("write npu_output: {e}"))?;
    }
    Ok(bank_dir)
}
```

- [ ] **Step 4: Run the test, full suite, commit**

Run: `cargo test --lib domains::scalar::domain` then `cargo test --lib`
Expected: PASS.

```bash
git add -A
git commit -m "fuzzer/scalar: ScalarChainRecord banking (self-describing chain)

Generated using Claude Code."
```

---

## Task 9: `impl Domain for ScalarDomain` + the `dtype(&self, case)` trait change

**Files:**
- Modify: `src/fuzzer/core/domain.rs` (widen `dtype` signature; update `MockDomain`)
- Modify: `src/fuzzer/domains/vector/domain.rs` (`dtype(&self)` -> `dtype(&self, _c)`)
- Modify: `src/fuzzer/core/engine.rs` (call site: `dom.dtype()` -> `dom.dtype(&case.case)`)
- Modify: `src/fuzzer/domains/scalar/domain.rs` (the `impl Domain` block + deferred Task-7 tests)

This is the one trait change the spec calls for (decision 7): scalar's dtype is per-case. Widen `Domain::dtype` to take the case; vector ignores the arg. Then wire the full `ScalarDomain` impl, tying together gen/lower/observe/compare/bank from Tasks 4-8.

- [ ] **Step 1: Widen the trait signature**

In `src/fuzzer/core/domain.rs`, change:

```rust
    /// Element dtype string passed to `compile_kernel_case` (vector: `"i32"`).
    fn dtype(&self) -> &str;
```
to:
```rust
    /// Element dtype string passed to `compile_kernel_case`. Depends on the case
    /// for domains with per-case dtype (scalar I32/I16/I8); vector ignores it.
    fn dtype(&self, case: &Self::Case) -> &str;
```

In the same file's `MockDomain` impl, change `fn dtype(&self) -> &str { "i32" }` to `fn dtype(&self, _c: &u64) -> &str { "i32" }`.

- [ ] **Step 2: Update vector + engine call site**

In `src/fuzzer/domains/vector/domain.rs`, change `fn dtype(&self) -> &str { "i32" }` to `fn dtype(&self, _c: &Chain) -> &str { "i32" }`.

In `src/fuzzer/core/engine.rs`, the two `dom.dtype()` call sites (the compile loop ~line 167 and the replay recompile ~line 405) become `dom.dtype(&case.case)` and `dom.dtype(&case)` respectively. Match the binding name in scope at each site (the compile loop has `case: &Case<D>` so `&case.case`; replay has `case` already the reconstructed `D::Case` so `&case`).

Run: `cargo build --features tooling`
Expected: clean (this confirms the trait change is consistent before the scalar impl lands).

- [ ] **Step 3: Add the deferred Task-7 tests + write the impl-dependent tests**

Add to the `tests` module of `src/fuzzer/domains/scalar/domain.rs` the two tests deferred from Task 7 (`compare_maps_region_to_its_stage_key`, `vacuous_all_zero_output_warns`) plus:

```rust
    #[test]
    fn dtype_is_per_case() {
        let dom = ScalarDomain;
        assert_eq!(Domain::dtype(&dom, &generate(1, "add/I8")), "i8");
        assert_eq!(Domain::dtype(&dom, &generate(1, "add/I32")), "i32");
    }

    #[test]
    fn coverage_keys_match_the_chain_keys() {
        let dom = ScalarDomain;
        let c = generate(2, "add/I16");
        assert_eq!(dom.coverage_keys(&c), c.keys());
        assert_eq!(dom.target_key(&c), "add/I16");
    }

    #[test]
    fn load_banked_reconstructs_replayable_case() {
        // Bank a chain to a temp dir, then load it back as Replayable.
        let c = generate(9, "and/I32");
        let tmp = std::env::temp_dir().join(format!("scalar-bank-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("fuzz_kernel.cc"), lower_chain(&c)).unwrap();
        let record = ScalarChainRecord::from_chain(&c, &c.keys());
        std::fs::write(tmp.join("chain.json"), serde_json::to_string(&record).unwrap()).unwrap();
        std::fs::write(tmp.join("npu_output.bin"), vec![7u8; 16]).unwrap();

        let dom = ScalarDomain;
        match dom.load_banked(&tmp) {
            Ok(Banked::Replayable { case, reference, keys }) => {
                assert_eq!(case, c);
                assert_eq!(reference.output, vec![7u8; 16]);
                assert_eq!(keys, c.keys());
            }
            other => panic!("expected Replayable, got {:?}", other.map(|_| "skip/err")),
        }
        std::fs::remove_dir_all(&tmp).ok();
    }
```

- [ ] **Step 4: Implement `impl Domain for ScalarDomain`**

Add to `src/fuzzer/domains/scalar/domain.rs` (above the test module):

```rust
impl Domain for ScalarDomain {
    type Case = ScalarChain;
    type Obs = ScalarObs;

    fn name(&self) -> &str {
        "scalar"
    }
    fn universe(&self) -> Vec<String> {
        universe_keys()
    }
    fn generate(&self, seed: u64, target: &str) -> ScalarChain {
        generate(seed, target)
    }
    fn coverage_keys(&self, c: &ScalarChain) -> Vec<String> {
        c.keys()
    }
    fn target_key(&self, c: &ScalarChain) -> String {
        c.target_key.clone()
    }
    fn lower(&self, c: &ScalarChain) -> String {
        lower_chain(c)
    }
    fn buffer_words(&self, c: &ScalarChain) -> usize {
        buffer_words(c)
    }
    fn dtype(&self, c: &ScalarChain) -> &str {
        c.dtype.template_dtype()
    }

    fn observe(
        &self,
        backend: Backend,
        xclbin: &Path,
        insts: &Path,
        c: &ScalarChain,
        max_cycles: u64,
    ) -> Result<ScalarObs, String> {
        observe_impl(backend, xclbin, insts, c, max_cycles)
    }

    fn warnings(&self, obs: &ScalarObs) -> Vec<String> {
        if !obs.output.is_empty() && obs.output.iter().all(|&b| b == 0) {
            vec!["vacuous output (all zero) -- chain folded or degenerate".into()]
        } else {
            Vec::new()
        }
    }

    fn compare(&self, emu: &ScalarObs, reference: &ScalarObs, keys: &[String]) -> Option<String> {
        let rkeys = region_keys(keys);
        let n = rkeys.len();
        first_divergent_region(&emu.output, &reference.output, n)
            .map(|r| rkeys[r.min(n.saturating_sub(1))].clone())
    }

    fn bank(
        &self,
        case_dir: &Path,
        c: &ScalarChain,
        reference: Option<&ScalarObs>,
        _emu_obs: Option<&ScalarObs>,
    ) -> Result<PathBuf, String> {
        bank_case(case_dir, c, &c.keys(), reference.map(|o| o.output.as_slice()))
    }

    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<ScalarChain, ScalarObs>, String> {
        let record: ScalarChainRecord = std::fs::read_to_string(seed_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))?;
        let npu_output =
            std::fs::read(seed_dir.join("npu_output.bin")).map_err(|e| format!("npu_output.bin: {e}"))?;
        Ok(Banked::Replayable {
            case: record.chain,
            reference: ScalarObs { output: npu_output, trace: None },
            keys: record.keys,
        })
    }

    fn dump_divergent_observation(&self, case_dir: &Path, emu: &ScalarObs) -> Result<(), String> {
        std::fs::write(case_dir.join("emu_output.bin"), &emu.output)
            .map_err(|e| format!("emu_output.bin write error: {e}"))
    }
}
```

Add any needed imports the tests reference (`lower_chain` is already imported; `Banked` is imported).

- [ ] **Step 5: Run scalar tests, full suite, vector fidelity**

Run: `cargo test --lib domains::scalar` then `cargo test --lib`
Expected: PASS.

Run: `cargo build --features tooling && cargo run --features tooling -- fuzz-vector --report`
Expected: vector report unchanged (the `dtype(case)` change must not perturb vector output).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fuzzer/scalar: impl Domain for ScalarDomain; dtype(&self, case) trait change

Widen Domain::dtype to take the case (scalar dtype is per-case; vector
ignores it) and wire the full scalar tenant: gen, lower, observe, the
per-region comparator, vacuity warning, and ScalarChainRecord bank/replay.

Generated using Claude Code."
```

---

## Task 10: Genericize engine log strings

**Files:**
- Modify: `src/fuzzer/core/engine.rs` (replace hardcoded "Vector fuzz"/"vector ops" with `dom.name()`)

The engine prints "Vector fuzz: ..." and "chains executed no vector ops -- folded" -- vector-specific strings now that scalar shares the engine. Make them `dom.name()`-driven (spec "folded-in deferrals").

- [ ] **Step 1: Replace the strings**

In `src/fuzzer/core/engine.rs`, replace each user-facing literal:
- `"Vector fuzz: coverage complete ..."` -> `format!("{} fuzz: coverage complete ...", dom.name())`
- `"Vector fuzz: 0 iterations requested ..."` -> `format!("{} fuzz: 0 iterations requested ...", dom.name())`
- `"Vector fuzz: {} iterations, base seed ..."` -> prefix with `dom.name()`
- `"Vector fuzz complete: ..."` -> prefix with `dom.name()`
- `"  ({folded} chains executed no vector ops -- folded)"` -> `format!("  ({folded} cases warned -- {})", ...)` or simply `"  ({folded} cases flagged a warning)"` (generic; the per-case warning text already prints the specifics).

Keep the format arguments identical; only the literal prefixes change. Capitalize as `dom.name()` yields lowercase ("vector"/"scalar"); either capitalize the first letter or accept lowercase -- pick lowercase for simplicity and adjust the existing vector-output expectation if any test asserts the exact "Vector fuzz" string (search tests first: `grep -rn "Vector fuzz" src/`).

- [ ] **Step 2: Build and run the suite**

Run: `cargo build --features tooling && cargo test --lib`
Expected: PASS. If a test asserted the literal "Vector fuzz" string, update it to the new generic form.

- [ ] **Step 3: Confirm vector run still reads correctly**

Run: `cargo run --features tooling -- fuzz-vector --report`
Expected: the report still prints (coverage numbers unchanged; only any campaign-progress prefix wording may differ -- `--report` exits before those lines, so output is identical).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fuzzer/engine: genericize campaign log strings via dom.name()

Generated using Claude Code."
```

---

## Task 11: CLI + runner dispatch (scalar coverage path vs legacy trace)

**Files:**
- Create: `src/fuzzer/domains/scalar/runner.rs`
- Modify: `src/fuzzer/domains/scalar/mod.rs` (add `pub mod runner;`)
- Modify: `src/fuzzer/runner.rs` (rename `run_fuzz` -> `run_trace_sweep_legacy`; keep everything else)
- Modify: `src/fuzzer/cli.rs` (`parse_fuzz_args` now builds `ScalarFuzzOptions`)
- Modify: `src/main.rs` (`run_fuzz_command` calls the new dispatcher)

The non-trace `fuzz` path routes through the coverage engine; `fuzz --trace-sweep` keeps the legacy single-accumulator path verbatim (spec decision 5). `ScalarFuzzOptions` carries the generic `CampaignOptions` plus the two trace fields, so the legacy path can be reconstructed when needed.

- [ ] **Step 1: Create the scalar runner + options**

Create `src/fuzzer/domains/scalar/runner.rs`:

```rust
//! Scalar fuzz entry point: dispatches the coverage-driven campaign (the
//! default `fuzz` path) or the legacy trace-sweep path (`--trace-sweep`).
//! Campaign orchestration lives in `core::engine`; scalar specifics in
//! `super::domain::ScalarDomain`. The trace path stays on the legacy
//! single-accumulator kernel in `fuzzer::runner` until trace becomes a
//! framework mode.

use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;
use crate::fuzzer::domains::scalar::domain::ScalarDomain;
use crate::fuzzer::runner::{run_trace_sweep_legacy, FuzzOptions};

/// Scalar `fuzz` knobs: the generic campaign options plus the two legacy
/// trace-sweep fields (used only when `trace_sweep` is set).
pub struct ScalarFuzzOptions {
    pub campaign: CampaignOptions,
    pub trace_sweep: bool,
    pub trace_sweep_reps: usize,
}

/// Run (or report/replay) the scalar fuzz campaign, or the legacy trace sweep.
pub fn run_scalar_fuzz(opts: &ScalarFuzzOptions) {
    if opts.trace_sweep {
        // Reconstruct the legacy FuzzOptions from the campaign knobs and run the
        // untouched accumulator-kernel trace path.
        let legacy = FuzzOptions {
            verbose: opts.campaign.verbose,
            jobs: opts.campaign.jobs,
            hw: opts.campaign.hw,
            max_cycles: opts.campaign.max_cycles,
            fuzz_iterations: opts.campaign.iterations,
            fuzz_seed: opts.campaign.seed,
            trace_sweep: true,
            trace_sweep_reps: opts.trace_sweep_reps,
        };
        run_trace_sweep_legacy(&legacy);
        return;
    }
    run_campaign(&ScalarDomain, &opts.campaign);
}
```

Add `pub mod runner;` to `src/fuzzer/domains/scalar/mod.rs`.

- [ ] **Step 2: Rename the legacy entry point**

In `src/fuzzer/runner.rs`, rename `pub fn run_fuzz(opts: &FuzzOptions)` to `pub fn run_trace_sweep_legacy(opts: &FuzzOptions)`. Leave the body and all helpers exactly as they are -- this path is only reached via `--trace-sweep` now, but the value-diff phases it runs first are harmless. Update the in-file test `test_run_fuzz_zero_iterations` to call `run_trace_sweep_legacy` (keep its assertions). `FuzzOptions` stays public (the scalar runner reconstructs it).

- [ ] **Step 3: Rework `parse_fuzz_args`**

In `src/fuzzer/cli.rs`:
- Change the import: `use crate::fuzzer::domains::scalar::runner::ScalarFuzzOptions;` (drop the `FuzzOptions` import; add `CampaignOptions`).
- Rewrite `parse_fuzz_args` to return `Result<ScalarFuzzOptions, String>`, building a `CampaignOptions` (same defaults as the vector parser: `target_hits: 10`, `report_only: false`, `replay: None`, `reverify: false`) plus `trace_sweep`/`trace_sweep_reps`. Add the coverage flags to the match arms: `--target-hits`, `--report`, `--reverify`, `--replay <dir>` (copy the vector parser's arms). Keep `--trace-sweep`/`--trace-sweep-reps`.

```rust
pub fn parse_fuzz_args(args: &[String]) -> Result<ScalarFuzzOptions, String> {
    let mut campaign = CampaignOptions {
        iterations: 0,
        seed: None,
        jobs: default_jobs(),
        hw: false,
        max_cycles: DEFAULT_MAX_CYCLES,
        target_hits: 10,
        verbose: false,
        report_only: false,
        replay: None,
        reverify: false,
    };
    let mut trace_sweep = false;
    let mut trace_sweep_reps = 5;

    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "fuzz" => {}
            "--iterations" | "-n" => campaign.iterations = parse_next(&mut iter, "--iterations")?,
            "--seed" => campaign.seed = Some(parse_next(&mut iter, "--seed")?),
            "--jobs" | "-j" => campaign.jobs = parse_next(&mut iter, "--jobs")?,
            "--max-cycles" => campaign.max_cycles = parse_next(&mut iter, "--max-cycles")?,
            "--target-hits" => campaign.target_hits = parse_next(&mut iter, "--target-hits")?,
            "--hw" => campaign.hw = true,
            "--no-hw" => campaign.hw = false,
            "--report" => campaign.report_only = true,
            "--reverify" => campaign.reverify = true,
            "--replay" => {
                let dir = iter.next().ok_or("--replay requires a directory")?;
                campaign.replay = Some(PathBuf::from(dir));
            }
            "--trace-sweep" => trace_sweep = true,
            "--trace-sweep-reps" => trace_sweep_reps = parse_next(&mut iter, "--trace-sweep-reps")?,
            "--verbose" | "-v" => campaign.verbose = true,
            other => return Err(format!("unknown fuzz argument: {}", other)),
        }
    }
    Ok(ScalarFuzzOptions { campaign, trace_sweep, trace_sweep_reps })
}
```

Update the existing `parse_fuzz_args` unit tests in `cli.rs` to read through `.campaign` (e.g. `parse_fuzz_args(&argv(&["--iterations","100","--seed","42"])).unwrap().campaign.iterations`), and add coverage of the new flags (`--target-hits`, `--report`, `--reverify`, `--replay`). Keep the trace-sweep test reading `.trace_sweep`/`.trace_sweep_reps`.

- [ ] **Step 4: Update `main.rs`**

In `src/main.rs`, `run_fuzz_command`:
```rust
fn run_fuzz_command(args: &[String]) -> anyhow::Result<()> {
    let opts = xdna_emu::fuzzer::cli::parse_fuzz_args(args).map_err(|e| anyhow::anyhow!("fuzz: {}", e))?;
    xdna_emu::fuzzer::domains::scalar::runner::run_scalar_fuzz(&opts);
    Ok(())
}
```
Update the usage/help text near lines 444-457 to mention scalar coverage flags (`fuzz --report`, `fuzz --replay`) alongside the existing examples, mirroring the `fuzz-vector` help.

- [ ] **Step 5: Build, full suite, commit**

Run: `cargo build --features tooling && cargo test --lib`
Expected: PASS (cli tests updated).

```bash
git add -A
git commit -m "fuzzer/scalar: CLI + runner dispatch (coverage default, --trace-sweep legacy)

Generated using Claude Code."
```

---

## Task 12: EMU smoke campaign + legacy-trace build check

**Files:** none (verification task; may add a doc note)

Confirm the scalar tenant reaches its universe end-to-end on the emulator (no HW needed for credit-less smoke), and that the legacy trace path still builds and runs. This is the acceptance gate's EMU half (spec "Acceptance").

- [ ] **Step 1: Scalar EMU smoke run**

Run:
```bash
cargo build --features tooling
cargo run --features tooling -- fuzz --iterations 40 --seed 1 --jobs 8
```
Expected: cases compile (some compile errors are acceptable and reported), EMU executes without panic, and the run prints a "scalar fuzz" completion line. No credit is earned without `--hw` (silicon-verified credit only) -- that is expected. Capture the output to `build/experiments/scalar-step2-smoke.log` (redirect, then Read it -- do not pipe through tail).

- [ ] **Step 2: Coverage report renders**

Run: `cargo run --features tooling -- fuzz --report`
Expected: a coverage report over the 33-key universe (universe: 33), printed via the genericized engine path.

- [ ] **Step 3: Legacy trace path still builds and runs**

Run:
```bash
cargo run --features tooling -- fuzz --iterations 4 --seed 1 --trace-sweep --no-hw
```
Expected: the legacy accumulator-kernel path compiles 4 cases and runs the emulator-only trace sweep without panic (no HW comparison). This confirms decision 5 -- the legacy path is intact. If it fails to build against anything touched in this plan, that is a regression to fix here.

- [ ] **Step 4: Full suite once more**

Run: `cargo test --lib`
Expected: PASS.

- [ ] **Step 5: Commit any doc/log note**

If you added a smoke-log note or a coverage-doc line, commit it:
```bash
git add -A
git commit -m "docs: scalar tenant EMU smoke + legacy-trace build confirmation

Generated using Claude Code."
```

---

## Task 13: Final review + vector fidelity sign-off

**Files:** none (verification + outcome note)

- [ ] **Step 1: Vector byte-identical sign-off**

Run:
```bash
cargo build --features tooling
cargo run --features tooling -- fuzz-vector --report
cargo run --features tooling -- fuzz-vector --replay ~/npu-work/experiments/phoenix-survival/vector
```
Expected: report unchanged from plan start (218/218 covered, the same divergent/resolved counts); replay shows the same match/divergent/error tallies as before Step 2 began. This is the hard regression bar.

- [ ] **Step 2: Append the Outcome section to this plan**

Add an `## Outcome` section recording: the commit range, the live `cargo test --lib` count before/after, the scalar universe size reached, the vector fidelity result, and any design refinements made during execution (the Step 1 plan's Outcome section is the template).

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "docs: framework Step 2 outcome -- scalar coverage chain tenant

Generated using Claude Code."
```

---

## Self-review notes (for the executor)

- **Spec coverage:** decisions 1 (33-key universe, Task 3) / 2 (targeted gen, Task 4) / 3 (per-region localize, Tasks 7+9) / 4 (vacuity via warnings, Task 9) / 5 (trace stays legacy, Tasks 11-12) / 6 (ScalarChainRecord, Task 8) / 7 (dtype(case), Task 9) are each mapped. The two folded-in deferrals (domains/ regroup Task 1; engine log genericization Task 10) are mapped.
- **The two vector-touching tasks** are Task 1 (move) and Task 9 (trait change). Both gate on vector byte-identical output. If either perturbs vector, stop and fix before proceeding.
- **Type consistency:** `Dtype`/`LoopStyle`/`ScalarOp`/`StageOp`/`Operand`/`ScalarStage`/`ScalarChain` are defined in Task 2 and used unchanged through Tasks 3-11. `ScalarObs` has fields `output`/`trace` throughout. `coverage_keys` = stage keys + loop key everywhere; `region_keys` filters `loop_` everywhere.
- **The loop-key-in-localization trap:** `compare` must filter `loop_`-prefixed keys (Task 7/9). A naive index into the full keys vector would mislabel a length mismatch as the loop key -- the `region_keys` filter plus the `r.min(n-1)` clamp prevent that. This is the scalar analogue of the Step-1 Task-4 compare-on-keys fix; do not regress it.
- **Legacy trace risk:** Task 11 renames but does not gut `fuzzer::runner`. If the rename surfaces a path the legacy path depended on (e.g. an external caller of `run_fuzz`), search `grep -rn "runner::run_fuzz" src/` before renaming and update every caller.
