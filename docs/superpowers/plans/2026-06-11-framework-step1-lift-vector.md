# Framework Step 1: Lift the engine, port vector as tenant 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the vector fuzzer's generic engine (campaign loop, ledger, banking, replay, reporting, compile dispatch) out of `src/fuzzer/vector/` into a shared `src/fuzzer/core/`, and re-express the vector fuzzer as the first `Domain` tenant of that engine — a pure refactor with no change to vector fidelity.

**Architecture:** A `Domain` trait owns the ~30% that is genuinely domain-specific (case AST, generate, lower, observe, compare, bank/replay-load); the engine owns the ~70% that is cross-cutting (the round-robin campaign loop, the coverage `Ledger`, parallel compile, the replay loop, reporting, backend dispatch). The engine is generic over `D: Domain` via static dispatch — `run_campaign::<VectorDomain>` is the only instantiation in this step. Scalar becomes tenant 2 in a later step and is what stress-tests the trait; this step's trait is deliberately vector-shaped.

**Tech Stack:** Rust, `serde`/`serde_json` (bank + ledger persistence), `std::thread::scope` (parallel compile), the existing `XclbinSuite`/`npu_runner` execution paths.

**Hard regression bar (this is a refactor, not a feature):** after every task `cargo test --lib` stays green with no drop in pass count; at the end, `fuzz-vector --replay <bank>` produces **24 match / 0 divergent / 0 error** (bit-identical EMU output to today) and `fuzz-vector --report` reads the existing `build/fuzz-vector/ledger.json` unchanged (218/0/resolved-6). Nothing about vector behavior — paths, ledger format, bank format, comparator — moves.

---

## Scope and deliberate deviations from the framework design doc

The design doc (`2026-06-11-unified-diff-fuzzing-framework.md`) describes the end-state. This step makes two scoped, flagged deviations to keep the PR a clean refactor:

1. **No `domains/` directory yet.** The doc's end-state layout is `core/` + `domains/{vector,scalar,...}/`. This step keeps `src/fuzzer/vector/` where it is (it simply *becomes* the vector tenant) and adds `src/fuzzer/core/`. The `domains/` grouping is introduced in Step 2, when scalar is ported and there are two tenants worth grouping — doing the `git mv vector/ domains/vector/` rename now would add module-path churn across `cli.rs`/`main.rs`/tests to an already-large step for no behavioral gain. **Reconcile in Step 2.**
2. **No `Aiesim` backend wiring.** `Backend` enumerates `{Interpreter, Aiesim, Hardware}` per the design, but the vector domain's `observe` returns an explicit "aiesim backend not wired for the vector domain" error for `Aiesim`. Wiring aiesim as a real backend is a later step; the enum variant is present so the seam is forward-compatible.

Everything else follows the doc: `core/` under the `fuzzer/` roof (fork 1, agreed), one ledger per domain at its existing path (fork 3 namespacing is deferred until two domains share a campaign — out of scope here).

## File structure

**New — `src/fuzzer/core/` (the shared engine):**
- `core/mod.rs` — module declarations + re-exports (`pub use` of the trait, `Backend`, `CampaignOptions`, `Ledger`, toolchain infra).
- `core/toolchain.rs` — build infra lifted verbatim from `fuzzer/runner.rs`: `ToolPaths` (+ `discover`/`aie_api_include`/`apply_env`), `TRACE_BUFFER_ELEMENTS`, `catch_panic`, `compile_kernel_case`.
- `core/ledger.rs` — `Ledger`, lifted from `vector/ledger.rs`, decoupled from `table()` (universe parameterized).
- `core/domain.rs` — the `Domain` trait, `Backend` enum, `CampaignOptions` struct, `Banked<C>` replay-load result.
- `core/engine.rs` — `run_campaign<D: Domain>` and the internal `run_replay<D: Domain>`, lifted from `vector/runner.rs`.

**Modified — `src/fuzzer/vector/` (now the vector tenant):**
- `vector/domain.rs` — **new**: `struct VectorDomain` + `impl Domain for VectorDomain`. Absorbs the vector-specific functions from `runner.rs` (lower, buffer_words, spec, observe, compare/localize, bank, replay-load, `ChainRecord`, `current_table_version`, `out_types_from_keys`).
- `vector/runner.rs` — slims to a thin `pub fn run_vector_fuzz(opts: &CampaignOptions)` that calls `core::engine::run_campaign(&VectorDomain, opts)`. (`VecFuzzOptions` becomes a re-export of `core::CampaignOptions`.)
- `vector/mod.rs` — add `pub mod domain;`.
- `vector/ledger.rs` — **deleted** (moved to `core/ledger.rs`).

**Modified — wiring:**
- `fuzzer/mod.rs` — add `pub mod core;`.
- `fuzzer/runner.rs` (scalar) — drop the lifted infra, import it from `core::toolchain` instead.
- `fuzzer/cli.rs` — unchanged surface; `VecFuzzOptions` still resolves (via re-export).

---

## Task 1: Create `core/` and lift the toolchain infra

Move the shared build infrastructure out of the scalar runner so both the scalar runner and the new engine import it from one home. Pure code-move, no logic change.

**Files:**
- Create: `src/fuzzer/core/mod.rs`
- Create: `src/fuzzer/core/toolchain.rs`
- Modify: `src/fuzzer/mod.rs` (add `pub mod core;`)
- Modify: `src/fuzzer/runner.rs:42-116,364-...,1025-...` (remove lifted items, add `use`)
- Modify: `src/fuzzer/vector/runner.rs:23` (re-point the import)

- [ ] **Step 1: Add the module declaration**

In `src/fuzzer/mod.rs`, add `pub mod core;` (before `pub mod runner;` so the engine module is visible to the runners). Result:

```rust
pub mod ast;
pub mod cli;
pub mod core;
pub mod gen;
pub mod lower_cpp;
pub mod params;
pub mod runner;
pub mod trace_sweep;
pub mod vector;
```

- [ ] **Step 2: Create `core/toolchain.rs` by moving the four shared items**

Cut these from `src/fuzzer/runner.rs` and paste into a new `src/fuzzer/core/toolchain.rs`, **unchanged except visibility**:
- `pub(crate) struct ToolPaths { ... }` and its `impl ToolPaths { discover, aie_api_include, apply_env }` (runner.rs:42-115)
- `pub(crate) const TRACE_BUFFER_ELEMENTS: usize = 262_144;` (runner.rs:116)
- `pub(crate) fn catch_panic<T>(...)` (runner.rs:364-...) and **its three unit tests** (`catch_panic_returns_ok_for_non_panicking_closure`, `catch_panic_converts_str_panic_to_err`, `catch_panic_converts_formatted_panic_to_err`) — move them into a `#[cfg(test)] mod tests` in `toolchain.rs`.
- `pub(crate) fn compile_kernel_case(...)` (runner.rs:1025-...)

Carry over every `use` those items need (`std::path::PathBuf`, `std::process::Command`, etc.) into `toolchain.rs`. Keep all visibilities `pub(crate)`.

- [ ] **Step 3: Create `core/mod.rs` with re-exports**

```rust
//! Domain-agnostic differential-fuzzing engine: the campaign loop, coverage
//! ledger, parallel compile, banking/replay, and reporting that every fuzzer
//! domain (vector, and later scalar/DMA/...) plugs into as a `Domain` tenant.
//!
//! Lifted out of the vector fuzzer in the framework Step 1 refactor
//! (`docs/superpowers/plans/2026-06-11-framework-step1-lift-vector.md`).

pub mod toolchain;

pub(crate) use toolchain::{catch_panic, compile_kernel_case, ToolPaths, TRACE_BUFFER_ELEMENTS};
```

- [ ] **Step 4: Re-point the importers**

In `src/fuzzer/runner.rs`, add `use crate::fuzzer::core::toolchain::{...};` for any of the four items the *remaining* scalar code still references (it uses all four). Remove the now-moved definitions. In `src/fuzzer/vector/runner.rs:23`, change

```rust
use crate::fuzzer::runner::{catch_panic, compile_kernel_case, ToolPaths, TRACE_BUFFER_ELEMENTS};
```

to

```rust
use crate::fuzzer::core::toolchain::{catch_panic, compile_kernel_case, ToolPaths, TRACE_BUFFER_ELEMENTS};
```

- [ ] **Step 5: Build and test**

Run: `cargo test --lib`
Expected: PASS, same count as before the task (the three `catch_panic` tests now live under `core::toolchain::tests` but still run). Note the baseline pass count here — it is the regression bar for the rest of the plan.

- [ ] **Step 6: Commit**

```bash
git add src/fuzzer/mod.rs src/fuzzer/core/ src/fuzzer/runner.rs src/fuzzer/vector/runner.rs
git commit -m "fuzzer: lift shared build infra into core/toolchain

Move ToolPaths, TRACE_BUFFER_ELEMENTS, catch_panic, and compile_kernel_case
out of the scalar runner into a domain-agnostic core/ module so the new
generic engine and both fuzzers share one home. Pure code-move; no logic
change.

Generated using Claude Code."
```

---

## Task 2: Lift the `Ledger` into `core/`, decoupled from the op table

The ledger is already domain-agnostic except `universe()`, which calls `table()`. Make the universe a parameter so the ledger holds only coverage-key strings; the domain supplies its universe.

**Files:**
- Create: `src/fuzzer/core/ledger.rs` (moved from `vector/ledger.rs`)
- Delete: `src/fuzzer/vector/ledger.rs`
- Modify: `src/fuzzer/core/mod.rs` (declare + re-export `Ledger`)
- Modify: `src/fuzzer/vector/mod.rs` (drop `pub mod ledger;`)
- Modify: `src/fuzzer/vector/runner.rs` (re-point `Ledger` import; `current_table_version` no longer uses `Ledger::universe`)

- [ ] **Step 1: Move the file and remove the table coupling**

`git mv src/fuzzer/vector/ledger.rs src/fuzzer/core/ledger.rs`. In `core/ledger.rs`:
- Delete `use super::table::table;` and the `pub fn universe() -> Vec<String>` method (it was the only `table()` user).
- Change the three methods that called `Self::universe()` to take the universe as a parameter:

```rust
/// Universe keys still needing credit: not divergent, not unreachable,
/// fewer than `target` hits. Sorted. `universe` is supplied by the domain.
pub fn uncovered(&self, universe: &[String], target: u32) -> Vec<String> {
    universe
        .iter()
        .filter(|k| !self.divergent.contains(*k) && !self.unreachable.contains_key(*k))
        .filter(|k| self.hits.get(*k).copied().unwrap_or(0) < target)
        .cloned()
        .collect()
}

/// True when every key not divergent/unreachable has >= `target` hits.
pub fn complete(&self, universe: &[String], target: u32) -> bool {
    self.uncovered(universe, target).is_empty()
}

/// Human-readable campaign status over `universe`.
pub fn report(&self, universe: &[String], target: u32) -> String {
    // body identical, but replace the local `let universe = Self::universe();`
    // with the passed-in `universe`, and iterate `universe.iter()`.
    // ... (rest of the existing body unchanged) ...
}
```

(`uncovered` previously consumed `Self::universe()` by value and returned owned `String`s; iterating a borrowed slice now needs `.cloned()`.)

- [ ] **Step 2: Update the ledger's own unit tests**

The tests in `core/ledger.rs` call `Ledger::universe()` and the three methods. Replace with a test-local universe helper so the ledger tests no longer depend on the vector table:

```rust
/// A synthetic coverage universe for ledger unit tests — the ledger is
/// table-agnostic, so tests supply their own key space.
fn test_universe() -> Vec<String> {
    (0..20).map(|i| format!("op{i}/I32x16/m0")).collect()
}
```

Then mechanically: every `Ledger::universe()` -> `test_universe()`; every `ledger.uncovered(t)` -> `ledger.uncovered(&u, t)`, `ledger.complete(t)` -> `ledger.complete(&u, t)`, `ledger.report(t)` -> `ledger.report(&u, t)`, binding `let u = test_universe();` at the top of each test. Delete `fresh_ledger_uncovered_is_full_universe`'s assertion against `table()` modes (it was testing the vector coupling we just removed) — replace it with an assertion that a fresh ledger's `uncovered(&u, 10) == u`.

- [ ] **Step 3: Re-export from core and drop the vector module**

In `src/fuzzer/core/mod.rs` add `pub mod ledger;` and `pub use ledger::Ledger;`. In `src/fuzzer/vector/mod.rs` remove the `pub mod ledger;` line.

- [ ] **Step 4: Re-point the vector runner**

In `vector/runner.rs`: change `use crate::fuzzer::vector::ledger::Ledger;` to `use crate::fuzzer::core::ledger::Ledger;`. Update the three call sites in `run_vector_fuzz`/`run_replay`/`report_only` to pass a universe. Bind it once near the top of `run_vector_fuzz` from the vector table (this is the last `table()`-derived universe in the engine path; it migrates to the domain in Task 4):

```rust
let universe = crate::fuzzer::vector::table::universe_keys();
```

Add a free fn `universe_keys()` to `vector/table.rs` holding exactly the body the old `Ledger::universe()` had (sorted `{name}/{out_type:?}/m{mode}` over `table()`):

```rust
/// Every coverage key derivable from the op table, sorted. Was `Ledger::universe`
/// before the ledger was made domain-agnostic.
pub fn universe_keys() -> Vec<String> {
    let mut keys: Vec<String> = table()
        .iter()
        .flat_map(|e| (0..e.modes).map(move |m| format!("{}/{:?}/m{}", e.name, e.out_type, m)))
        .collect();
    keys.sort();
    keys
}
```

Fix `current_table_version()` in `runner.rs` to hash `universe_keys().join("\n")` instead of `Ledger::universe().join("\n")` — **the hashed bytes are identical**, so banked `table_version` stamps are preserved.

- [ ] **Step 5: Build and test**

Run: `cargo test --lib`
Expected: PASS, no drop from the Task 1 baseline. The ledger tests now run under `core::ledger::tests`.

- [ ] **Step 6: Commit**

```bash
git add -A src/fuzzer/core/ledger.rs src/fuzzer/core/mod.rs src/fuzzer/vector/
git commit -m "fuzzer: lift Ledger into core/, decouple from the op table

The coverage ledger is domain-agnostic except universe(), which called the
vector table directly. Parameterize the universe on uncovered/complete/report
so the ledger holds only key strings; the domain supplies its key space.
table_version hashing is byte-preserved (same keys, same join).

Generated using Claude Code."
```

---

## Task 3: Define the `Domain` trait, `Backend`, `CampaignOptions`, `Banked`

Add the engine's contract. No domain implements it yet; a mock proves it compiles and is object-safe enough for the static-dispatch engine.

**Files:**
- Create: `src/fuzzer/core/domain.rs`
- Modify: `src/fuzzer/core/mod.rs` (declare + re-export)

- [ ] **Step 1: Write the trait and supporting types**

Create `src/fuzzer/core/domain.rs`:

```rust
//! The `Domain` contract: everything a fuzzer subsystem must provide for the
//! shared engine to run a coverage-driven differential campaign over it. The
//! engine owns the campaign loop, ledger, compile dispatch, banking/replay, and
//! reporting; a `Domain` owns the case AST, generation, lowering, execution,
//! comparison, and its own bank/replay persistence (the serialized form is
//! domain-specific — vector banks a `ChainRecord`, DMA would bank a BD program).

use std::path::{Path, PathBuf};

/// Which executor produced an observation. Differential = any two backends; the
/// vector domain compares `Interpreter` vs `Hardware`. `Aiesim` is enumerated
/// for forward-compatibility but not wired for vector in Step 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Interpreter,
    Aiesim,
    Hardware,
}

/// Campaign knobs — domain-agnostic. Lifted from the vector `VecFuzzOptions`.
pub struct CampaignOptions {
    pub iterations: usize,
    /// Base seed (None = wall clock); case i runs seed base+i.
    pub seed: Option<u64>,
    pub jobs: usize,
    pub hw: bool,
    pub max_cycles: u64,
    /// Hits per key needed for completion (ledger target).
    pub target_hits: u32,
    pub verbose: bool,
    /// Print the coverage report and exit.
    pub report_only: bool,
    /// Replay banked divergences from this directory.
    pub replay: Option<PathBuf>,
    /// Clear divergent flags and re-earn them against silicon this run.
    pub reverify: bool,
}

/// Result of loading one banked case for replay.
pub enum Banked<C, O> {
    /// A replayable case: reconstructed case + the banked reference observation
    /// (e.g. the silicon output) + the banked coverage keys for localization.
    Replayable { case: C, reference: O, keys: Vec<String> },
    /// The bank cannot be replayed (e.g. a legacy bank under a changed table);
    /// the engine reports the reason and skips it.
    Skip(String),
}

/// A fuzzer subsystem the engine can drive. `Case` is the domain's AST; `Obs`
/// is whatever an execution yields (output bytes, moved buffers, cycle counts —
/// opaque to the engine, which only ever round-trips it through `compare`/`bank`).
pub trait Domain {
    type Case;
    type Obs;

    /// Stable short name: ledger lives at `build/fuzz-{name}`, banks at
    /// `phoenix-survival/{name}`. For vector this is `"vector"`.
    fn name(&self) -> &str;

    /// The full coverage-key universe (sorted), supplied to the ledger.
    fn universe(&self) -> Vec<String>;

    /// Generate the deterministic case for `(seed, target_key)`.
    fn generate(&self, seed: u64, target: &str) -> Self::Case;

    /// Per-stage coverage keys a passing case credits, in output order.
    fn coverage_keys(&self, case: &Self::Case) -> Vec<String>;

    /// The case's target key (the one generation aimed at).
    fn target_key(&self, case: &Self::Case) -> String;

    /// Lower the case to kernel source text written as `fuzz_kernel.cc`.
    fn lower(&self, case: &Self::Case) -> String;

    /// Buffer size in dtype words for the compile (`buf_in`/`scratch`/`out`).
    fn buffer_words(&self, case: &Self::Case) -> usize;

    /// Element dtype string passed to `compile_kernel_case` (vector: `"i32"`).
    fn dtype(&self) -> &str;

    /// Execute the compiled case on `backend`, returning a domain observation.
    fn observe(
        &self,
        backend: Backend,
        xclbin: &Path,
        insts: &Path,
        case: &Self::Case,
        max_cycles: u64,
    ) -> Result<Self::Obs, String>;

    /// Non-fatal warnings about an observation (e.g. "no vector ops executed —
    /// chain folded"). The engine prints them; default none.
    fn warnings(&self, _obs: &Self::Obs) -> Vec<String> {
        Vec::new()
    }

    /// Differential compare: `None` = match (credit the case); `Some(key)` =
    /// the first divergent coverage key (flag it, bank the case). Localization
    /// lives here because only the domain knows the observation's structure.
    fn compare(&self, emu: &Self::Obs, reference: &Self::Obs, case: &Self::Case) -> Option<String>;

    /// Bank a divergent/crashed case under `phoenix-survival/{name}/seed_N/` for
    /// post-mortem and replay. `reference` is None on an emulator crash (no HW
    /// observation). `case_dir` is the live build dir to copy artifacts from.
    fn bank(
        &self,
        case_dir: &Path,
        case: &Self::Case,
        reference: Option<&Self::Obs>,
        emu_obs: Option<&Self::Obs>,
    ) -> Result<PathBuf, String>;

    /// Load one banked seed dir for replay: reconstruct the case + reference
    /// observation, or report why it cannot be replayed.
    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<Self::Case, Self::Obs>, String>;
}
```

- [ ] **Step 2: Add a compile-only mock test**

In `domain.rs`, prove the trait is implementable and usable behind a generic bound:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockDomain;
    impl Domain for MockDomain {
        type Case = u64;
        type Obs = Vec<u8>;
        fn name(&self) -> &str { "mock" }
        fn universe(&self) -> Vec<String> { vec!["a/I32x16/m0".into()] }
        fn generate(&self, seed: u64, _t: &str) -> u64 { seed }
        fn coverage_keys(&self, _c: &u64) -> Vec<String> { vec!["a/I32x16/m0".into()] }
        fn target_key(&self, _c: &u64) -> String { "a/I32x16/m0".into() }
        fn lower(&self, _c: &u64) -> String { String::new() }
        fn buffer_words(&self, _c: &u64) -> usize { 16 }
        fn dtype(&self) -> &str { "i32" }
        fn observe(&self, _b: Backend, _x: &Path, _i: &Path, _c: &u64, _m: u64) -> Result<Vec<u8>, String> { Ok(vec![]) }
        fn compare(&self, a: &Vec<u8>, b: &Vec<u8>, _c: &u64) -> Option<String> {
            if a == b { None } else { Some("a/I32x16/m0".into()) }
        }
        fn bank(&self, _d: &Path, _c: &u64, _r: Option<&Vec<u8>>, _e: Option<&Vec<u8>>) -> Result<PathBuf, String> { Ok(PathBuf::new()) }
        fn load_banked(&self, _d: &Path) -> Result<Banked<u64, Vec<u8>>, String> { Ok(Banked::Skip("mock".into())) }
    }

    fn drive<D: Domain>(d: &D) -> Vec<String> { d.universe() }

    #[test]
    fn mock_domain_satisfies_the_trait_behind_a_generic_bound() {
        let m = MockDomain;
        assert_eq!(drive(&m), vec!["a/I32x16/m0".to_string()]);
        assert_eq!(m.compare(&vec![1], &vec![2], &0), Some("a/I32x16/m0".into()));
        assert_eq!(m.compare(&vec![1], &vec![1], &0), None);
    }
}
```

- [ ] **Step 3: Re-export from core**

In `src/fuzzer/core/mod.rs` add `pub mod domain;` and `pub use domain::{Backend, Banked, CampaignOptions, Domain};`.

- [ ] **Step 4: Build and test**

Run: `cargo test --lib core::domain`
Expected: PASS (`mock_domain_satisfies_the_trait_behind_a_generic_bound`). Then `cargo test --lib` — no drop from baseline.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/core/domain.rs src/fuzzer/core/mod.rs
git commit -m "fuzzer: add the Domain trait, Backend, CampaignOptions

The engine's contract: a Domain owns case/generate/lower/observe/compare and
its own bank/replay persistence; the engine owns the campaign loop, ledger,
compile, and reporting. Obs is an associated type — opaque to the engine,
which never assumes equal bytes. A mock domain proves it compiles behind the
generic bound the engine will use. No domain implements it yet.

Generated using Claude Code."
```

---

## Task 4: Implement `Domain` for `VectorDomain`

Move the vector-specific functions out of `runner.rs` into a `VectorDomain` impl. The campaign/replay loops stay in `runner.rs` for now (Task 5 lifts them); here they switch from calling free functions to calling `VectorDomain` methods, so behavior is provably identical.

**Files:**
- Create: `src/fuzzer/vector/domain.rs`
- Modify: `src/fuzzer/vector/mod.rs` (add `pub mod domain;`)
- Modify: `src/fuzzer/vector/runner.rs` (remove the moved fns; call `VectorDomain` methods)

- [ ] **Step 1: Define the vector observation and `VectorDomain`**

Create `src/fuzzer/vector/domain.rs`. Define the observation as exactly what `run_emulator_vec`/`run_npu_vec` produced:

```rust
//! The vector fuzzer as a `core::Domain` tenant. Owns the vector-specific half
//! of the engine: chain generation, lowering, execution on EMU/HW, the
//! type-aware NaN-tolerant slice comparator, and durable bank/replay.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::toolchain::TRACE_BUFFER_ELEMENTS;
use crate::fuzzer::vector::chain::{Chain, Stage};
use crate::fuzzer::vector::gen::generate;
use crate::fuzzer::vector::lower::lower_chain;
use crate::fuzzer::vector::table::{table, universe_keys, VecType};
use crate::interpreter::execute::fuzz_recorder;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// One execution's result: output bytes, optional trace, executed coverage keys.
pub struct VecObs {
    pub output: Vec<u8>,
    pub trace: Option<Vec<u8>>,
    pub executed: Vec<String>,
}

/// The vector fuzzer tenant. Zero-sized: all state is per-case.
pub struct VectorDomain;
```

Move into this file, **unchanged in logic**, from `runner.rs`:
- `ChainRecord` + `StageRecord` + their `impl` (`from_chain`, `to_chain`)
- `current_table_version()` (now uses `universe_keys()`)
- `out_types_from_keys()`
- `buffer_words()`, `chain_out_types()`, `bf16_is_nan()`, `slice_equal()`, `first_divergent_slice()`, `slice_to_key()`, `make_vec_buffer_spec()`
- the bodies of `run_emulator_vec`, `run_npu_vec`, `bank_case`
- the replay reconstruction logic from `run_replay` (the `if !record.pool.is_empty() { to_chain } else { regenerate-or-skip }` block) becomes `load_banked`

- [ ] **Step 2: Write the `impl Domain`**

```rust
impl Domain for VectorDomain {
    type Case = Chain;
    type Obs = VecObs;

    fn name(&self) -> &str { "vector" }
    fn universe(&self) -> Vec<String> { universe_keys() }
    fn generate(&self, seed: u64, target: &str) -> Chain { generate(seed, target) }
    fn coverage_keys(&self, c: &Chain) -> Vec<String> { c.keys() }
    fn target_key(&self, c: &Chain) -> String { c.target_key.clone() }
    fn lower(&self, c: &Chain) -> String { lower_chain(c) }
    fn buffer_words(&self, c: &Chain) -> usize { buffer_words(c) }
    fn dtype(&self) -> &str { "i32" }

    fn observe(&self, backend: Backend, xclbin: &Path, insts: &Path, c: &Chain, max_cycles: u64)
        -> Result<VecObs, String>
    {
        match backend {
            Backend::Interpreter => {
                let spec = make_vec_buffer_spec(c, buffer_words(c));
                let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
                let suite = XclbinSuite::new().with_max_cycles(max_cycles);
                fuzz_recorder::arm();
                let (outcome, raw_output, trace) = suite.run_single_with_trace(&test);
                let executed = fuzz_recorder::take().unwrap_or_default();
                if !outcome.is_pass() {
                    return Err(format!("emulator outcome not pass: {outcome:?}"));
                }
                let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
                Ok(VecObs { output, trace, executed })
            }
            Backend::Hardware => {
                use crate::testing::npu_runner;
                if !npu_runner::npu_available() {
                    return Err("NPU hardware not available".into());
                }
                let spec = make_vec_buffer_spec(c, buffer_words(c));
                let test_name = format!("vecfuzz_seed_{}", c.seed);
                match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                    Ok(result) => Ok(VecObs {
                        output: result.output,
                        trace: result.extra_outputs.get("buf_trace").cloned(),
                        executed: Vec::new(),
                    }),
                    Err(e) => Err(format!("{:?}", e)),
                }
            }
            Backend::Aiesim => Err("aiesim backend not wired for the vector domain (Step 1)".into()),
        }
    }

    fn warnings(&self, obs: &VecObs) -> Vec<String> {
        if obs.executed.is_empty() {
            vec!["no vector ops executed (chain folded by compiler)".into()]
        } else {
            Vec::new()
        }
    }

    fn compare(&self, emu: &VecObs, reference: &VecObs, c: &Chain) -> Option<String> {
        first_divergent_slice(&emu.output, &reference.output, &chain_out_types(c))
            .map(|slice| slice_to_key(&c.keys(), slice))
    }

    fn bank(&self, case_dir: &Path, c: &Chain, reference: Option<&VecObs>, _emu: Option<&VecObs>)
        -> Result<PathBuf, String>
    {
        let npu_output = reference.map(|o| o.output.as_slice());
        let npu_trace = reference.and_then(|o| o.trace.as_deref());
        let executed = reference.map(|o| o.executed.as_slice()).unwrap_or(&[]);
        bank_case(case_dir, c, npu_output, npu_trace, executed)
    }

    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<Chain, VecObs>, String> {
        let record: ChainRecord = std::fs::read_to_string(seed_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))?;
        let chain = if !record.pool.is_empty() {
            record.to_chain()
        } else {
            let c = generate(record.seed, &record.target_key);
            if c.keys() != record.keys {
                return Ok(Banked::Skip(
                    "legacy bank under a changed table (no pool) -- re-bank to replay".into(),
                ));
            }
            c
        };
        let npu_output = std::fs::read(seed_dir.join("npu_output.bin"))
            .map_err(|e| format!("npu_output.bin: {e}"))?;
        Ok(Banked::Replayable {
            case: chain,
            reference: VecObs { output: npu_output, trace: None, executed: Vec::new() },
            keys: record.keys,
        })
    }
}
```

Note: `compare`'s localization in replay must use the **banked** keys, not regenerated ones. The engine passes the reconstructed `case`; for durable banks `case.keys()` equals the banked keys, and for the legacy fast-path they were guarded equal — so `compare(... c)` using `c.keys()` is correct in both. (The banked `keys` from `Banked` remain available to the engine for the replay mismatch message.)

- [ ] **Step 3: Move the relevant unit tests into `domain.rs`**

Move from `runner.rs`'s `mod tests` into a `#[cfg(test)] mod tests` in `domain.rs` (they test the moved functions): `slice_localization_maps_to_stage_key`, `equal_buffers_with_zero_padding_do_not_diverge`, `length_mismatch_diverges_at_first_extra_slice`, the `bf16_slice` helper + all five bf16 comparator tests, `slice_past_last_stage_clamps_to_final_key`, `buffer_words_covers_pool_and_output`, `buffer_spec_embeds_pool_bytes_and_zero_out`, `chain_record_round_trip_matches_regeneration`, `durable_bank_reconstructs_chain_without_the_table`, `out_types_parsed_from_keys_match_the_table`, `table_version_is_stable_across_calls`. They reference now-local fns, so only the `use super::*;` changes. Keep `tail_lines` + its test and `vector_compile_clean_200_seeds` + `undefined_symbols` in `runner.rs` for now (Task 5 decides their final home — `tail_lines` is engine-generic).

- [ ] **Step 4: Rewire `runner.rs` to call `VectorDomain`**

In `vector/runner.rs`: delete the moved functions. Replace their call sites in `run_vector_fuzz`/`run_replay` with `let dom = VectorDomain;` and method calls — e.g. `run_emulator_vec(&xclbin, &case.chain, max)` becomes `dom.observe(Backend::Interpreter, &xclbin, &insts, &case.chain, opts.max_cycles)`; `run_npu_vec(...)` becomes `dom.observe(Backend::Hardware, ...)`; `first_divergent_slice(...)`+`slice_to_key(...)` becomes `dom.compare(&emu, &npu, &case.chain)`; `bank_case(...)` becomes `dom.bank(...)`; `lower_chain` becomes `dom.lower`; `buffer_words` becomes `dom.buffer_words`; the replay reconstruction block becomes `dom.load_banked(case_dir)`. The folded warning uses `dom.warnings(&emu)`. Keep the loop structure and counters byte-for-byte otherwise.

- [ ] **Step 5: Build and test**

Run: `cargo test --lib`
Expected: PASS, no drop from baseline. All moved comparator/bank tests run under `vector::domain::tests`.

- [ ] **Step 6: Commit**

```bash
git add src/fuzzer/vector/domain.rs src/fuzzer/vector/mod.rs src/fuzzer/vector/runner.rs src/fuzzer/vector/table.rs
git commit -m "fuzzer: re-express the vector fuzzer as a core::Domain tenant

Move the vector-specific half (lower, observe on EMU/HW, the NaN-tolerant
slice comparator, durable bank/replay-load, ChainRecord) into a VectorDomain
impl of the core Domain trait. The campaign/replay loops still live in
runner.rs but now call VectorDomain methods instead of free functions, so
behavior is provably identical. Comparator and bank unit tests move alongside.

Generated using Claude Code."
```

---

## Task 5: Lift the campaign and replay loops into the generic engine

Move the orchestration out of `vector/runner.rs` into `core/engine.rs` as `run_campaign<D: Domain>`. `run_vector_fuzz` becomes a one-line wrapper.

**Files:**
- Create: `src/fuzzer/core/engine.rs`
- Modify: `src/fuzzer/core/mod.rs` (declare + re-export `run_campaign`)
- Modify: `src/fuzzer/vector/runner.rs` (slim to the wrapper)

- [ ] **Step 1: Create `core/engine.rs` from the lifted loops**

Move `run_vector_fuzz` -> `run_campaign<D: Domain>(dom: &D, opts: &CampaignOptions)` and the private `run_replay` -> `run_replay<D: Domain>(dom: &D, dir: &Path, opts: &CampaignOptions)`, plus `tail_lines` and the `VecCase`-equivalent (now `Case<D>` holding `case: D::Case, case_dir: PathBuf`). Mechanical substitutions throughout:
- `build/fuzz-vector` -> `build/fuzz-{}` with `dom.name()`.
- `Ledger` universe: bind `let universe = dom.universe();` once; pass `&universe` to `ledger.uncovered/report/complete`.
- `generate(seed, target)` -> `dom.generate(seed, target)`.
- `lower_chain(&chain)` -> `dom.lower(case)`; `buffer_words(&chain)` -> `dom.buffer_words(case)`; `"i32"` -> `dom.dtype()`.
- `case.chain.keys()` -> `dom.coverage_keys(case)`; `case.chain.target_key` -> `dom.target_key(case)`.
- `run_emulator_vec`/`run_npu_vec` -> `dom.observe(Backend::Interpreter|Hardware, ...)`.
- divergence check -> `dom.compare(&emu, &reference, case)` (returns `Option<key>` directly; the old `first_divergent_slice` + `slice_to_key` two-step collapses into this one call).
- `bank_case(...)` -> `dom.bank(case_dir, case, Some(&npu_obs), Some(&emu_obs))` (crash path: `dom.bank(case_dir, case, None, None)`).
- folded warning -> iterate `dom.warnings(&emu_obs)`.
- replay reconstruction -> `match dom.load_banked(seed_dir) { Banked::Replayable{case, reference, keys} => ..., Banked::Skip(why) => { errors += 1; println!(...) } }`. The replay compile-if-needed uses `dom.buffer_words(&case)`/`dom.dtype()`; the EMU run uses `dom.observe(Backend::Interpreter, ...)`; the compare uses `dom.compare(&emu, &reference, &case)`, and on mismatch the message localizes via the banked `keys`.

The `reverify` block (clear divergent, re-earn, `mark_resolved`) is fully ledger-side and lifts verbatim. The per-10 `ledger.save` cadence, the unreachable-on-2-distinct-compile-fail logic, the compile thread-scope pool, and all counters lift verbatim.

- [ ] **Step 2: Re-export and write the wrapper**

In `core/mod.rs`: `pub mod engine; pub use engine::run_campaign;`. In `vector/runner.rs`, replace the whole file body with:

```rust
//! Vector fuzz entry point: a thin tenant wrapper over the generic engine.
//! All orchestration lives in `core::engine`; all vector specifics in
//! `super::domain::VectorDomain`.

use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;
use crate::fuzzer::vector::domain::VectorDomain;

/// Run (or report/replay) the vector fuzz campaign.
pub fn run_vector_fuzz(opts: &CampaignOptions) {
    run_campaign(&VectorDomain, opts);
}
```

Keep `vector_compile_clean_200_seeds` + `undefined_symbols` (the ignored Peano-toolchain compile-clean test): move them to `vector/domain.rs`'s test module (they exercise vector lowering, not the engine).

- [ ] **Step 3: Make `VecFuzzOptions` resolve to `CampaignOptions`**

`cli.rs` constructs `VecFuzzOptions { ... }` and returns it. Keep `cli.rs` untouched by re-exporting the type under both names. In `vector/runner.rs` (or `vector/mod.rs`):

```rust
pub use crate::fuzzer::core::domain::CampaignOptions as VecFuzzOptions;
```

`cli.rs`'s `use crate::fuzzer::vector::runner::VecFuzzOptions;` then resolves to the lifted struct, whose fields are identical, so `parse_vector_fuzz_args` compiles unchanged.

- [ ] **Step 4: Build and test**

Run: `cargo test --lib`
Expected: PASS, no drop from baseline. The cli tests (`vector_defaults_when_only_subcommand`, `vector_parses_all_flags`, etc.) still pass against the re-exported options.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/core/engine.rs src/fuzzer/core/mod.rs src/fuzzer/vector/runner.rs
git commit -m "fuzzer: lift the campaign and replay loops into core::engine

run_vector_fuzz is now a one-line wrapper over the generic
run_campaign<D: Domain>(&VectorDomain, opts). The round-robin campaign loop,
parallel compile, ledger bookkeeping, reverify, banking, and replay loop are
domain-agnostic and shared. VecFuzzOptions is now a re-export of the lifted
CampaignOptions; the fuzz-vector CLI is unchanged. Vector is tenant 1.

Generated using Claude Code."
```

---

## Task 6: Regression gate — prove vector fidelity is byte-identical

No new code. Verify the hard regression bar offline (no HW needed: replay is EMU-vs-banked-silicon, report reads the local ledger).

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-framework-step1-lift-vector.md` (add the Outcome section)

- [ ] **Step 1: Full library test sweep**

Run: `cargo test --lib`
Expected: PASS with the same total count noted at Task 1 Step 5 (only test module *paths* changed — `core::toolchain::tests`, `core::ledger::tests`, `vector::domain::tests`, `core::domain::tests` — not the count).

- [ ] **Step 2: Build the binary and the FFI `.so`**

Run: `cargo build --release` then `cargo build -p xdna-emu-ffi`
Expected: clean build (the `.so` is only needed if a later HW spot-check is run; the offline replay below uses the in-process `XclbinSuite`, not the plugin).

- [ ] **Step 3: Coverage report reads the existing ledger unchanged**

Run: `cargo run --release -- fuzz-vector --report`
Expected: `universe: 218`, `covered: 218`, `uncovered: 0`, `divergent: 0`, `resolved: 6` — identical to the pre-refactor ledger (the JSON format and key strings are byte-preserved).

- [ ] **Step 4: Replay the banked corpus — the fidelity bar**

Run: `cargo run --release -- fuzz-vector --replay ~/npu-work/experiments/phoenix-survival/vector`
Expected: `Replay complete: 24 match, 0 divergent, 0 error` — bit-identical to the post-#114 baseline. Any mismatch or error means the refactor changed vector behavior and must be fixed before the step is done.

- [ ] **Step 5: Record the outcome and commit**

Append an `## Outcome` section to this plan with the test count, the report numbers, and the replay result. Commit:

```bash
git add docs/superpowers/plans/2026-06-11-framework-step1-lift-vector.md
git commit -m "docs: framework Step 1 outcome -- vector lifted, fidelity byte-identical

Generated using Claude Code."
```

---

## Self-review

**Spec coverage** (against the framework design doc's Step 1): "Move the engine out of `vector/` into `src/fuzzer/core/`" → Tasks 1, 2, 5. "Re-express vector as `Domain`" → Tasks 3, 4. "the 218/218 ledger, banked corpus, and replay all still pass unchanged" → Task 6. "refactor with a hard regression bar — nothing about vector fidelity moves" → the byte-preservation notes (table_version hash, paths via `name()`, comparator/bank moved unchanged) + Task 6 gate. The `domains/` rename and aiesim backend are explicitly deferred with rationale (Scope section).

**Placeholder scan:** every code step shows complete code or an exact mechanical-substitution list; moved code is identified by symbol + source line, not reproduced where reproduction adds nothing. No "TBD"/"handle edge cases"/"similar to".

**Type consistency:** `CampaignOptions` fields match `VecFuzzOptions` (iterations, seed, jobs, hw, max_cycles, target_hits, verbose, report_only, replay, reverify). `Domain::Obs` = vector's `VecObs { output, trace, executed }`. `compare -> Option<String>` (divergent key) is consumed identically by `run_campaign`. `Banked<C, O>` carries `case`/`reference`/`keys`, all consumed in the replay arm. `dom.name()` = `"vector"` keeps `build/fuzz-vector` and `phoenix-survival/vector` paths.

---

## Outcome (2026-06-11 -- Step 1 complete, fidelity byte-identical)

All six tasks landed subagent-driven (fresh implementer per task; the two
substantive tasks got the full spec + code-quality two-stage review). `cargo
test --lib` held at **3421 passed / 0 failed / 6 ignored** through every task
(3421 = pre-Step-1 3420 + the one `Domain` mock test). Clean build, zero
warnings throughout.

**Commits (in order):**
- `f0569f2d` -- Task 1: lift toolchain infra into `core/toolchain`.
- `80d121e6` -- Task 2: lift `Ledger` into `core/`, decouple from the op table
  (universe parameterized; `table_version` hash byte-preserved).
- `7da58f21` -- Task 3: add the `Domain` trait, `Backend`, `CampaignOptions`,
  `Banked` (mock-tested).
- `b57dc929` -- Task 4: re-express the vector fuzzer as `VectorDomain`.
- `0a875c8b` -- Task 4 correction (from the implementer's own escalation +
  review): `compare()` takes the coverage **keys**, not the case, so the
  durable-bank replay path stays table-independent (it had regressed to
  live-table types); `bank()` draws the executed-op list from the EMU
  observation, not the HW reference. Both restore exact pre-refactor behavior.
- `a27d5dc7` -- Task 4 polish: restored the "stale zeros" rationale comment in
  `observe` (code-quality review nit).
- `90dc2405` -- Task 5: lift the campaign + replay loops into the generic
  `core::engine::run_campaign<D: Domain>`; `run_vector_fuzz` is now a 17-line
  wrapper; `VecFuzzOptions` re-exports `CampaignOptions` (CLI unchanged).
- `9185c14d` -- Task 5 polish: `Domain::name` doc accuracy nit.

**Design refinements made during execution (within the approved framework):**
1. `compare(emu, reference, keys: &[String])` -- keys, not the case. Unifies the
   campaign and replay comparators onto one table-independent path (campaign
   passes live `coverage_keys`, replay passes banked keys). This is strictly
   better than the original plan's `compare(.., case)` and was forced by the
   table-independence requirement the durable-bank format (#118) exists to keep.
2. `dump_divergent_observation(case_dir, &Obs)` -- a **defaulted no-op** trait
   hook (the agreed resolution to the replay seam where the engine must persist
   the EMU output but `Obs` is opaque). Keeps the engine 100% domain-agnostic;
   vector overrides it to write `emu_output.bin`. Chosen over a `-> &[u8]`
   accessor precisely to preserve the opaque-`Obs` invariant (a DMA/timing
   observation has no single natural byte payload).
3. `Sync` bounds (`D: Domain + Sync, D::Case: Sync`) live on `run_campaign`'s
   signature, NOT on the `Domain` trait -- they are a property of the parallel
   compile pool, not of being a domain. `Send` is not needed (the scope threads
   borrow shared refs; no owned `D::Case` crosses the boundary).

**Regression gate (offline, release binary):**
- `fuzz-vector --report`: universe **218**, covered **218**, uncovered **0**,
  divergent **0**, resolved **6** -- identical to the pre-refactor ledger.
- `fuzz-vector --replay ~/npu-work/experiments/phoenix-survival/vector`:
  **24 match, 0 divergent, 0 error** -- bit-identical EMU output to the
  post-#114 baseline.

**Documented cosmetic deltas (no functional impact -- counters, ledger,
banking, replay verdicts, and exit codes are all byte-identical):**
- Verbose `--no-hw` smoke logging lost its `({} bytes, {} vector ops executed)`
  detail (now just `emu ok`), forced by `Obs` opacity. Behind both `-v` and
  `--no-hw`; touches no credited path. Accepted rather than add a trait method
  to restore a debug line (code-quality reviewer concurred).
- The generic engine still prints "Vector"-flavored strings ("Vector fuzz:",
  "chains executed no vector ops"). Faithfully preserved from the original (not
  a defect); a second tenant would mis-label. Genericize via `dom.name()` /
  a domain log hook in **Step 2**, where changing the strings is in-scope.

**Carried to Step 2:** the `domains/{vector,scalar}/` directory regrouping (this
step kept `vector/` in place); porting the scalar fuzzer as tenant 2 (the real
stress test of the trait); and the engine log-string genericization above.
