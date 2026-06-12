# Scalar fuzzer as a coverage-driven Domain tenant (framework Step 2)

**Date:** 2026-06-11
**Status:** design, confirmed (Maya, 2026-06-11) -- ready to plan.
**Decision:** upgrade the scalar fuzzer to a **coverage-driven, per-op-localized
chain tenant** matching vector's rigor. Not a behavior-preserving port: scalar's
program model is redesigned from a single-accumulator kernel into a **chain of
elementwise stages**, each writing its own output region, so a divergence
localizes to the exact op -- exactly as vector localizes per slice.
**Related:** `2026-06-11-unified-diff-fuzzing-framework.md` (framework),
`2026-06-11-framework-step1-lift-vector.md` (Step 1: engine + vector tenant).

> **Supersedes** this doc's first draft (the single-output-buffer / coarse
> "program-level localization" model). Maya's call: go all the way to vector-grade
> per-op rigor -- "bringing everything together rigor-wise really matters here."

## Why a redesign, and why it does not lose scalar's value

Today's scalar fuzzer (`src/fuzzer/{ast,gen,lower_cpp,params,runner}.rs`, `fuzz`)
is a seed-sweep with a single accumulator stored once -> one output buffer -> a
mismatch can't be pinned to an op. Vector localizes per-op because each stage
writes its own slice. To match that, scalar adopts the **same chain shape**:

- A case = N **stages** + a dtype + a loop style. Each stage is an elementwise
  scalar op `out_k[i] = op_k(operand_a[i], operand_b[i])` over the buffer, writing
  its **own output region k**. Operands draw from the input buffer, a prior
  stage's output, or a literal.
- The kernel still loops over buffer elements (Simple or HardwareLoop), so the
  loop body still contains stores crossing the AIE2 ZOL back-edge; body size =
  stage count still sweeps the fetch-packet boundary; stage k+1 reading stage k's
  output is exactly the recency-1 structure. **Scalar's distinctive
  ZOL-store-flush / recency-1 catches are preserved** -- we only *add* per-op
  localization on top. (This was the key worry; working it through, the chain
  model keeps every loop-timing property the accumulator model had.)

Both tenants are now "a chain of ops, each writing its own output, localized
per-op" -- distinct op tables and lowering, one coherent shape. That uniformity is
the point.

## The decisions

### 1. Coverage universe -- per-op localizable + a case-level loop dimension

`{feature}/{dtype}`, dtype in `I32`/`I16`/`I8`:
- **Arithmetic stages** (localizable): `add sub mul and or xor shl shr` x 3 = **24**.
- **Branch-select stage** (localizable): `branch` x 3 = **3**. An elementwise
  select `out[i] = cond[i] ? a[i] : b[i]` -- still writes its own region, still
  localizes.
- **Loop style** (case-level, not per-stage): `loop_simple`, `loop_hw` x 3 = **6**.
  This is what exercises the ZOL boundary; credited to the case, not a stage.

= **33 keys**, every *value-producing* key (the 27 arith + branch) per-op
localizable; the 6 loop-style keys are case-level.

**Dropped vs. today:** the nested `HwLoop`-as-an-op variant. A per-element nested
loop has no clean localizable region, and the hardware-loop *execution path* is
still covered by the `loop_hw` outer style. Small, deliberate narrowing for a
fully-localizable model.

### 2. Target-driven generation -- `generate(seed, target)`

Parse `target` = `{feature}/{dtype}`: set the chain's dtype; generate N seeded
random stages; guarantee the target appears (force an `add..shr` stage for an
arith target; force a branch-select stage for `branch/*`; set the outer loop style
for `loop_*`). Round-robin over uncovered keys, exactly as vector.

### 3. Localization -- per-op, matching vector

Per-region comparison (region k <-> stage k). `first_divergent_region` returns the
first differing region; `compare` maps it to that stage's coverage key -- the
direct analogue of vector's `first_divergent_slice` + `slice_to_key`. On match,
credit all stage keys + the case's loop-style key. On mismatch, flag the
localized stage's key and bank the chain. Full op-level rigor.

### 4. Vacuous (all-zero) regions -- credit, but keep counting

A region that is all-zero on both sides is a (weak) silicon match. The trait's
`compare` is binary, so a vacuous region credits, but the runner logs the vacuous
count for visibility (as today). Minor, bounded by `target_hits`; avoids a trait
change. (Per-region now, so vacuity is judged per stage.)

### 5. trace_sweep -- preserved as legacy path, NOT ported (a future "mode")

Step 2 ports only the value-diff to `run_campaign(&ScalarDomain)`. `--trace-sweep`
keeps its existing code (`execute_trace_sweep`, `generate_group_insts`, the
`trace_sweep` module), routed separately. trace becomes a real framework mode in a
later step. (Note: the legacy trace_sweep assumed the old accumulator kernel; it
stays wired to whatever the scalar lowering produces, or is parked behind the flag
until the trace-mode step -- the plan will confirm it still builds against the
chain lowering, else gate it explicitly.)

### 6. Banking -- serialize the chain AST (inputs are formulaic)

Inputs are `Sequential { start: 1, step: 1 }`, determined by buffer_size, so no
input pool to bank (unlike vector). A `ScalarChainRecord` banks: seed, target_key,
dtype, buffer_size, loop_style, the stage list, the coverage keys, and the HW
output. Needs `serde` on the chain AST types. Replay deserializes -> lowers ->
compiles -> runs EMU -> compares per region vs banked HW. Table-independent (the
AST is the source of truth).

### 7. One small trait change -- `dtype(&self, case)`

Vector's dtype is always `"i32"`; scalar's is per-case (I32/I16/I8). Widen the
trait method to `dtype(&self, case: &Self::Case) -> &str` (vector ignores the arg).
Minimal and honest -- dtype genuinely depends on the case for scalar. The only
trait touch Step 2 needs.

## ScalarDomain sketch

```
struct ScalarDomain;
struct ScalarChain { seed, target_key, dtype, buffer_size, loop_style, stages: Vec<ScalarStage> }
struct ScalarStage { op: StageOp, operand_a: Operand, operand_b: Operand }  // StageOp = Arith(ScalarOp) | BranchSelect
struct ScalarObs { regions: Vec<u8>, trace: Option<Vec<u8>> }

impl Domain for ScalarDomain {
    type Case = ScalarChain;   type Obs = ScalarObs;
    fn name(&self) -> &str { "scalar" }                  // build/fuzz-scalar, phoenix-survival/scalar
    fn universe(&self) -> Vec<String>                    // 33 keys (decision 1)
    fn generate(&self, seed, target) -> ScalarChain      // targeted chain gen (decision 2)
    fn coverage_keys(&self, c) -> Vec<String>            // exercised stage keys + loop-style key, target first
    fn target_key(&self, c) -> String { c.target_key }
    fn lower(&self, c) -> String                         // NEW chain lowering: per-stage elementwise store
    fn buffer_words(&self, c) -> usize                   // N stages * buffer_size region layout
    fn dtype(&self, c) -> &str                           // per-case (decision 7)
    fn observe(backend, ..) -> ScalarObs
    fn compare(emu, ref, keys) -> Option<String>         // per-region localize (decision 3)
    fn bank / load_banked                                // ScalarChainRecord (decision 6)
    fn dump_divergent_observation                        // emu regions for diff
}
```

## Folded-in Step-1 deferrals

- **`domains/` regrouping:** `vector/` -> `domains/vector/`, scalar files ->
  `domains/scalar/`. Mechanical module-path churn; now justified by two tenants.
- **Engine log-string genericization:** the engine's hardcoded "Vector fuzz:" /
  "vector ops" strings become `dom.name()`-driven so the scalar tenant logs right.

## Acceptance

- `cargo test --lib` green (new scalar-chain + scalar-domain tests added).
- Vector tenant **unchanged**: 218/218 report + 24/0/0 replay still bit-identical
  (the `domains/` move and `dtype(case)` change must not touch vector fidelity).
- A scalar EMU-only smoke run reaches the 33-key universe and credits without
  panic; a short `--hw` campaign credits real silicon matches with per-op
  localization on any divergence.
- `fuzz --trace-sweep` still builds and runs (or is explicitly gated if the chain
  lowering breaks the legacy trace path -- decided in the plan).
