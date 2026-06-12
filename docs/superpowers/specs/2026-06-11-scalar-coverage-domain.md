# Scalar fuzzer as a coverage-driven Domain tenant (framework Step 2)

**Date:** 2026-06-11
**Status:** design, for review (precedes the Step 2 execution plan).
**Decision taken (Maya, 2026-06-11):** upgrade the scalar fuzzer to coverage-driven
(not a behavior-preserving port). It becomes a full `Domain` tenant matching
vector -- universe, target-driven generation, banking, replay -- validating the
`core::Domain` trait against a second, structurally different tenant.
**Related:** `2026-06-11-unified-diff-fuzzing-framework.md` (the framework),
`2026-06-11-framework-step1-lift-vector.md` (Step 1, the engine + vector tenant).

## Why this is an uplift, not a port

The scalar fuzzer (`src/fuzzer/{ast,gen,lower_cpp,params,runner}.rs`, `fuzz`) is a
**seed-sweep**: `generate(seed)` makes a random program, runs it EMU-vs-HW, diffs
bytes. It has no coverage universe, no target-driven generation, no banking, no
replay -- none of the coverage-driven machinery the `Domain` trait (built against
vector) assumes. Making it a tenant means *adding* that machinery. The trait does
not change; scalar grows into it.

The one structural fact that shapes everything below: **a scalar program writes a
single output buffer** (the final accumulator stored in a loop), whereas a vector
chain writes one 64-byte slice per stage. Vector gets *per-stage* divergence
localization for free; scalar cannot. The design accommodates that.

## The six decisions

### 1. Coverage universe -- `{feature}/{dtype}`

Mirror vector's `{name}/{type}/m{mode}` with a scalar-appropriate key:
`{feature}/{dtype}` where `dtype` is one of `I32`/`I16`/`I8` and `feature` is one of:
- the 8 scalar arithmetic ops: `add`, `sub`, `mul`, `and`, `or`, `xor`, `shl`, `shr`
- the control-flow / structural features the fuzzer exists to stress:
  `branch`, `hwloop`, `loop_simple`, `loop_hw`

That is 12 features x 3 dtypes = **36 keys**. The arithmetic 24 are the core
verification claim ("every scalar op silicon-verified at every width"); the
structural 12 cover the branch / hardware-loop / loop-style paths that the
generator was widened to exercise (the AIE2 ZOL store-flush boundary -- see
`gen.rs`). Operand kinds (Var/Literal/Load) are *not* separate keys -- they are
ubiquitous and not independently interesting.

> **Open choice for review:** granularity. We could drop the structural 12 (keep
> only 24 arith keys), or add operand-kind keys (broader). Proposal: the 36 above
> -- arith is the core, loop/branch coverage is worth it given the fuzzer's
> ZOL-boundary heritage, operand kinds are noise.

### 2. Target-driven generation -- `generate(seed, target)`

Parse `target` = `{feature}/{dtype}`: set the program dtype to `dtype`, then
generate the random (seeded) body as today BUT guarantee at least one instance of
`feature` appears (force a `Mul` op for `mul/*`; force a `HwLoop` for `hwloop/*`;
set `loop_style` for `loop_*`; inject a `Branch` for `branch/*`). The rest stays
random. This is exactly vector's approach (a seeded random case that is
*guaranteed to exercise the target*), so the round-robin campaign over uncovered
keys works unchanged.

### 3. Localization -- coarse by necessity, honest about it

Single output buffer => a byte mismatch cannot be pinned to one op. Resolution,
all expressible within the existing trait:
- **On match:** credit *all* coverage keys the program exercised (like vector
  credits all stage keys). A whole-program silicon match is strong evidence for
  every (op, dtype) it contains.
- **On mismatch:** `compare` returns the **target key** (the one generation aimed
  at), and the engine banks the full program. `coverage_keys(case)` lists the
  target key **first**, so the existing `compare -> keys.first()` shape lands the
  flag on the target meaningfully. Per-*element* mismatch detail (the existing
  `format_mismatch`: "element [i]: emu=X, npu=Y") is preserved in the bank/log for
  human triage.

This is coarser than vector (program-level, not op-level localization), which is
the price of the single-buffer kernel shape. The bank carries the full program,
so triage loses nothing; only the *automatic* attribution is coarse.

> **Flag for review:** this is the real semantic difference from vector. If you
> want op-level localization, scalar kernels would have to be restructured to
> store per-op slices -- a large behavior change I do *not* recommend (it would
> warp scalar into vector). Proposal: accept program-level localization + the
> bank for triage.

### 4. Vacuous (all-zero) matches -- credit, but keep counting them

The existing scalar fuzzer flags all-zero==all-zero as "vacuous" (counted apart
from real passes -- weak evidence, both sides trivially zero). The trait's
`compare` is binary (None=credit / Some=divergent) and has no "match but don't
credit" state. Resolution: a vacuous match credits coverage (it *is* a silicon
match), but the runner still logs the vacuous count for visibility (as today). The
coverage weakening is minor and bounded by `target_hits` (it is unlikely all N
hits for a key are vacuous). Avoiding a trait change is worth that small cost.

> **Alternative if you dislike crediting vacuous:** bias the generator away from
> trivially-zero outputs. Harder to guarantee; proposal is to credit + log.

### 5. trace_sweep -- preserved as-is, NOT ported (it is a future "mode")

`run_fuzz` today bundles value-diff (Phases 1-2) with `--trace-sweep` (Phase 3: a
multi-group trace capture + comparison). The framework treats trace/timing as
**modes**, deferred. Step 2 ports only the **value-diff** to
`run_campaign(&ScalarDomain)`. `--trace-sweep` keeps its existing code path
(`execute_trace_sweep`, `generate_group_insts`, the `trace_sweep` module) intact,
routed separately. So `fuzz` (value-diff) goes through the framework; `fuzz
--trace-sweep` runs the preserved legacy path. trace becomes a real framework mode
in a later step, not now.

### 6. Banking format -- serialize the AST (inputs are formulaic)

Scalar inputs are `Sequential { start: 1, step: 1 }` -- fully determined by
`buffer_size`, so there is no input pool to bank (unlike vector). A `ScalarRecord`
banks: `seed`, `target_key`, `dtype`, `buffer_size`, the `KernelBody` AST, the
coverage `keys`, and (for replay) the HW output. This needs `serde` derives on the
AST types (`KernelBody`/`KernelOp`/`ScalarOp`/`Operand`/`Var`/`BufRef`/`LoopStyle`)
-- the scalar analogue of vector's `StageRecord`. Replay deserializes the AST ->
lowers -> compiles -> runs EMU -> compares vs banked HW. Table-independent by
construction (the AST is the source of truth, not a regeneration).

## What the ScalarDomain looks like (sketch)

```
struct ScalarDomain;
struct ScalarObs { output: Vec<u8>, trace: Option<Vec<u8>>, vacuous: bool }

impl Domain for ScalarDomain {
    type Case = FuzzParams;   // seed + buffer_size + dtype + KernelBody
    type Obs  = ScalarObs;
    fn name(&self) -> &str { "scalar" }                 // build/fuzz-scalar, phoenix-survival/scalar
    fn universe(&self) -> Vec<String> { /* 36 keys */ }
    fn generate(&self, seed, target) -> FuzzParams { /* targeted gen, decision 2 */ }
    fn coverage_keys(&self, c) -> Vec<String> { /* exercised keys, target first */ }
    fn target_key(&self, c) -> String { c.target_key }   // (new field on FuzzParams or carried)
    fn lower(&self, c) -> String { lower_cpp::lower_to_cpp(c) }
    fn buffer_words(&self, c) -> usize { c.buffer_size }
    fn dtype(&self) -> &str { /* per-case: see note */ }
    fn observe(backend, ..) -> ScalarObs { /* existing run_emulator / run_on_npu_raw */ }
    fn compare(emu, ref, keys) -> Option<String> { /* byte-eq; vacuous->credit; mismatch->keys[0] */ }
    fn bank(..) / load_banked(..) { /* ScalarRecord, decision 6 */ }
    fn dump_divergent_observation(..) { /* emu_output.bin, like vector */ }
}
```

**One trait-fit note:** `dtype()` is currently `&self -> &str` (vector is always
`"i32"`), but scalar's dtype is *per-case* (I32/I16/I8). The compile call needs the
case's dtype. Two clean options: (a) widen the trait to `dtype(&self, case:
&Self::Case) -> &str` (vector ignores the arg), or (b) have the engine pass
`dtype` via the case and the domain's `lower`/`buffer_words` already carry it.
Proposal: **(a)** -- `dtype(&self, case)` -- minimal, honest (dtype genuinely
depends on the case for scalar), vector adapts trivially. This is the one small
trait change Step 2 needs; flagging it explicitly.

## Scope and deferrals folded in

Step 2 also clears the two Step-1 deferrals, since this is the natural moment:
- **`domains/` regrouping:** move `vector/` -> `domains/vector/` and the scalar
  files -> `domains/scalar/`, now that there are two tenants to group (per the
  framework doc's end-state layout). Module-path churn, mechanical.
- **Engine log-string genericization:** the engine's hardcoded "Vector fuzz:" /
  "vector ops" strings become `dom.name()`-driven, so the scalar tenant logs
  correctly. (Step 1 deferred this; with a second tenant it now matters.)

## Acceptance

- `cargo test --lib` stays green (no regression; new scalar-domain tests added).
- The vector tenant is **unchanged**: 218/218 report + 24/0/0 replay still
  bit-identical (the `domains/` move and `dtype(case)` change must not touch vector
  fidelity).
- A scalar coverage smoke run reaches its 36-key universe and credits on EMU-only
  (no-HW) without panic; a short `--hw` campaign credits real silicon matches.
- `fuzz --trace-sweep` still works via the preserved legacy path.
