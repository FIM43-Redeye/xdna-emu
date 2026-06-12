# DMA / data-movement as a coverage-driven Domain tenant (framework Step 3)

**Date:** 2026-06-11
**Status:** design, confirmed (Maya, 2026-06-11) -- ready to plan.
**Decision:** add a third `core::Domain` tenant that fuzzes **data movement**
itself -- buffer-descriptor access patterns (n-dimensional offset/size/stride/wrap,
padding, packet vs circuit, MM2S/S2MM, BD chaining) across the shim and memtile
DMA engines -- with the same per-region localization rigor as the scalar and
vector tenants. This is the **axis-2 Phoenix-retirement payoff**: `clean_release(Aie2)`
axis-2 (DMA/stream side effects) only goes green when the framework drives these
subsystems to verified against silicon.
**Related:** `2026-06-11-unified-diff-fuzzing-framework.md` (framework),
`2026-06-11-framework-step1-lift-vector.md` (Step 1: engine + vector tenant),
`2026-06-11-scalar-coverage-domain.md` (Step 2: scalar chain tenant).

## Guiding principle

> **No signal is marginal when the goal is cycle accuracy.** (Maya, 2026-06-11.)

This is the disposition for the whole tenant. dtype is not deferred as
low-value; padding is not skipped as fiddly; packet routing is not dropped as
rare. Every axis the silicon distinguishes, the fuzzer covers. The only
deferrals here are *separate subsystems* that earn their own later scope
(compute-tile DMA), never a "this signal probably doesn't matter" cut.

## Why DMA inverts the framework, and what that forces

Vector (tenant 1) and scalar (tenant 2) are **compute** tenants. The engine is
built around that shape: `Domain::lower` returns a C++ `fuzz_kernel.cc`, Peano
compiles it, and `fuzz_template.py` wraps it in a **fixed** data-movement harness
(shim 0,0 -> memtile 0,1 -> compute 0,2, linear 1D objectfifos). The *variable*
artifact is the compute kernel; the data path is boilerplate.

A pure data-movement program inverts this exactly. In mlir-aie's real pure-DMA
tests (`test/npu-xrt/shim_dma_bd_reuse/aie.mlir`, `dma_complex_dims/aie2.py`,
`matrix_transpose/aie2.py`, the padding tests under
`add_*_i8_using_*dma_op_with_padding/`) there is **no meaningful compute** -- the
program *is* the data path: an n-dimensional access pattern reshuffling DDR
through the DMA engines. The variable artifact is the **transfer**; the kernel is
a trivial passthrough or absent.

So this tenant cannot reuse the kernel+template compile path. It owns its MLIR
generation and runs `aiecc.py` directly. That is the one framework touch (see
decision 7), additive and default-preserving -- the same minimal spirit as Step
2's `dtype(case)` widening.

The emulator side is already ready: `src/device/dma/` models the full BD
vocabulary data-driven from AM025 -- shim (3D + iteration), memtile (4D +
iteration + per-dimension zero-padding), compute (3D + iteration), MM2S/S2MM,
packet headers, lock acq/rel, BD chaining -- plus an address generator
(`addressing.rs`) that walks all of it. This tenant is about *generating valid
access patterns* and *localizing divergences*, not new emulator capability.

## The decisions

### 1. Program model -- a chain of transfers, per-region localized

A case = a **chain of N >= 1 transfers**. Each transfer reshuffles its own input
region -> its own output region under one generated access pattern, and **exactly
one BD in that transfer carries the fuzzed pattern** -- the BD named by the target
key. Every other BD on the path runs a trivial linear 1D passthrough. Output
region k is a deterministic reshuffle of input region k.

- **N = 1 is the single-transfer mode** (whole-buffer pattern; also the
  minimal-repro shape a banked chain shrinks to). The generator ranges N
  including 1, and forces N = 1 for any feature that wants the whole buffer.
- **Localization** is per-region, byte-identical to scalar: `first_divergent_region`
  returns the first differing output region; `compare` maps it to that transfer's
  coverage key. This reuses the scalar localization machinery wholesale.

### 2. Data path topology

Real pure-DMA tests loop DDR through the array. The minimal faithful path:

```
DDR(in) -> shim MM2S -> memtile S2MM -> memtile MM2S -> shim S2MM -> DDR(out)
```

The fuzzed pattern rides whichever BD the target key names (a shim BD or a
memtile BD, on the MM2S or S2MM side); the other three BDs run linear 1D. The N
transfers issue sequentially (configure -> start -> await, BD-reuse), each with
its own in/out offsets -- exactly the `shim_dma_bd_reuse` structure. No compute
core is involved (the path is pure DMA loopback).

### 3. Coverage universe -- `{feature}/{engine}/{dir}/{dtype}`

Four axes, with capability gating so only valid combinations enter the universe:

- **engine** in {`shim`, `memtile`}. (Compute-tile DMA is deferred -- decision 9.)
- **dir** in {`mm2s`, `s2mm`} -- gather (strided read, linear write) vs scatter
  (linear read, strided write). Distinct silicon paths; a gather-only bug must
  not hide behind scatter coverage (the principle).
- **dtype** in {`i32`, `i16`, `i8`} -- element size changes byte strides and
  padding granularity; first-class per the principle.
- **feature** -- the structural access-pattern class:
  - both engines: `linear` (1D contiguous baseline / known-good anchor),
    `strided2d`, `strided3d`, `transpose`, `overlap` (stride < size), `iter`
    (iteration dimension active), `chain` (use_next_bd, 2+ BDs).
  - shim/memtile MM2S-only: `packet` (packet-switched routing; circuit is the
    default implied by every other feature).
  - memtile-only: `strided4d` (D3).
  - memtile MM2S-only: `padbefore`, `padafter`, `padboth` (per-dimension
    zero-padding -- only expressible at raw MLIR `aie.dma_bd` level, see
    decision 6, and only inserted on memtile MM2S per the emulator's
    `transfer/padding.rs`).

Capability gating (4D + padding -> memtile only; packet + padding -> MM2S only)
is enforced from the AM025 field layouts (`crates/xdna-archspec/src/dma/field_layouts.rs`),
not hardcoded. The valid cross-product lands at roughly 100+ keys -- between
scalar's 33 and vector's 218, finishable, built to 100%.

**Note on packet routing:** the #97-era recon found the emulator currently does
not distinguish packet vs circuit *routing* (it inserts packet headers but treats
both as data movement). This axis may therefore surface a real model gap on
silicon -- which is exactly the fuzzer's job. Expect `packet/*` keys to be the
first divergence candidates.

### 4. Target-driven generation -- `generate(seed, target)`

Parse `target` = `{feature}/{engine}/{dir}/{dtype}`. Set the chain's dtype; choose
N seeded-random transfers; guarantee the target appears (force a transfer on the
named engine/direction carrying the named feature's pattern). The remaining
transfers draw random valid (engine, dir, feature) combinations. Round-robin over
uncovered keys, exactly as scalar/vector. Generation is **deterministic** in
`(seed, target)`.

### 5. Localization, vacuity

- **Localization** (decision 1): per-region `first_divergent_region` -> stage key,
  filtering any case-level keys before indexing region keys (the scalar loop-key
  filter pattern -- avoid a length-mismatch mislabel). For DMA all keys are
  per-transfer, so the filter is a no-op guard, but keep it for parity and safety.
- **Vacuity:** inputs are formulaic `Sequential { start: 1, step: 1 }` (non-zero),
  so a correct transfer yields a non-zero output region. An all-zero output region
  signals a degenerate/empty transfer -> emit a `warnings()` warning (not a trait
  change), exactly as scalar's all-zero hook.

### 6. Lowering -- Rust emits raw MLIR (full BD-field control)

`lower(case)` returns a **complete `aie.mlir`** string built in Rust, in the raw
`aiex.dma_configure_task` / `aie.dma_bd` form (modeled on `shim_dma_bd_reuse` and
the padding tests). This form gives full control of every BD field -- sizes,
strides, wraps, `const_pad_before`/`const_pad_after`, packet info, `next_bd`
chaining -- which the high-level IRON `npu_dma_memcpy_nd` API **cannot express**
(it has no padding arguments). Since padding is in-scope (the principle), MLIR-direct
is required, not optional. The emitter mirrors `lower.rs`'s role but targets MLIR
instead of C++. No `fuzz_template.py`; the generated MLIR is self-contained
(device, tiles, flows/packet_flows, BDs, `aie.runtime_sequence`).

The runtime_sequence declares two DDR memref args (in, out) sized to the chain's
total input/output footprint. `buffer_words(case)` returns that footprint; the
`BufferSpec` built in `observe` (decision 8) must agree with the generated MLIR's
argument order and group ids (in, out; no scratch, no trace -- trace is a future
mode). This MLIR/BufferSpec agreement is the one cross-cutting implementation
invariant.

### 7. The framework touch -- a defaulted `Domain::compile` hook

Today the engine hardwires Phase 2/3: `lower() -> write fuzz_kernel.cc -> Peano ->
fuzz_template.py -> aiecc`. Add one trait method:

```
fn compile(&self, tools: &ToolPaths, case_dir: &Path, case: &Self::Case)
    -> Result<(), String>;
```

with a **default impl equal to today's behavior** (write `self.lower(case)` to
`fuzz_kernel.cc`, call `compile_kernel_case(tools, case_dir, self.buffer_words(case),
self.dtype(case))`). Vector and scalar inherit it unchanged -- zero behavior
change, byte-identical fidelity. The DMA domain **overrides** it: write
`self.lower(case)` to `aie.mlir`, run `aiecc.py` straight through (no Peano kernel
compile, no template). The engine's compile loop changes from inline
lower+write+compile_kernel_case to a single `dom.compile(tools, &case.case_dir,
&case.case)` call. This is the only `core/` change Step 3 needs.

### 8. Execution / observe -- unchanged backends, pure differential

Reuse the existing executors verbatim:
- **Interpreter:** build a `BufferSpec` (in = `Sequential{1,1}`, out = zeros),
  `XclbinSuite::run_single_with_trace`.
- **Hardware:** `npu_runner::run_on_npu(&spec, name, xclbin, insts, timeout)`.

Pure **EMU-vs-HW differential**, no golden oracle (consistent with vector/scalar;
two executions of the same generated transfer agree or they don't). The `Aiesim`
backend **errors out** for now ("not wired for the DMA domain (Step 3)"), matching
scalar Step 2. `Obs` = the output buffer bytes (`DmaObs { output: Vec<u8> }`);
opaque to the engine, round-trips through `compare`/`bank`.

### 9. Safety -- a staged tier, not a wall

A malformed BD can wedge the NPU (TDR or NOAVAIL) on the Phoenix box. Generation
is **staged**, with a safety tier as a knob:

- **Stage A (this build):** the **safe surface** -- patterns valid by construction:
  total footprint <= region size, strides/wraps within AM025 field widths,
  addresses within the allocated buffer, padding consistent with length. The
  generator emits *only* these. Built and verified to 100% against HW. This is the
  defined subsystem Step 3 finishes.
- **Stage B (deliberate, later):** the **wedge-prone edges** -- overflow addresses,
  out-of-bounds footprints, inconsistent lengths -- advanced into *knowingly*,
  accepting wedge risk, **once Stage A is exhausted** and pushing the boundary is
  the only way forward (Maya: "it's OK to wedge the NPU when we've verified
  everything we can with normal stuff and need to advance"). The generator is
  designed so this boundary is *reachable* (a tier flag unlocks it), never
  baked-out-as-impossible -- but Stage B is not built now.

This reconciles the principle (we *will* eventually probe even wedge-capable
patterns) with the hardware cost (safe-exhausted-first, controlled order).

### 10. Banking -- `DmaChainRecord`, table-independent

Inputs are formulaic, so there is no input pool to bank (like scalar). A
`DmaChainRecord` serializes the whole chain AST (seed, target_key, dtype, per-transfer
engine/direction/feature + BD-pattern fields), the banked coverage keys, and the
HW output, under `$HOME/npu-work/experiments/phoenix-survival/dma/seed_N/`.
Replay deserializes -> lowers -> compiles -> runs EMU -> compares per region vs
banked HW. The AST is the source of truth; no live table or generator state in the
serialized form. All chain AST types derive `Serialize`/`Deserialize`.

## DmaDomain sketch

```
struct DmaDomain;
struct DmaChain { seed, target_key, dtype, transfers: Vec<DmaTransfer> }
struct DmaTransfer { engine, dir, feature, bd: BdPattern, in_off, out_off, len }
  // BdPattern = n-D offset/size/stride/wrap + optional padding + packet + chaining,
  //             field-valid by construction (Stage A)
struct DmaObs { output: Vec<u8> }

impl Domain for DmaDomain {
    type Case = DmaChain;   type Obs = DmaObs;
    fn name(&self) -> &str { "dma" }                     // build/fuzz-dma, phoenix-survival/dma
    fn universe(&self) -> Vec<String>                    // ~100+ keys (decision 3)
    fn generate(&self, seed, target) -> DmaChain         // targeted, Stage-A-safe (decisions 4, 9)
    fn coverage_keys(&self, c) -> Vec<String>            // per-transfer keys, target first
    fn target_key(&self, c) -> String { c.target_key }
    fn lower(&self, c) -> String                         // complete aie.mlir (decision 6)
    fn compile(&self, tools, dir, c) -> Result<()>       // OVERRIDE: aiecc on aie.mlir (decision 7)
    fn buffer_words(&self, c) -> usize                   // total in/out footprint
    fn dtype(&self, c) -> &str                           // per-case (i32/i16/i8)
    fn observe(backend, ..) -> DmaObs                    // XclbinSuite / npu_runner (decision 8)
    fn warnings(&self, o) -> Vec<String>                 // all-zero region -> vacuity (decision 5)
    fn compare(emu, ref, keys) -> Option<String>         // per-region localize (decisions 1, 5)
    fn bank / load_banked                                // DmaChainRecord (decision 10)
    fn dump_divergent_observation                        // emu output for diff
}
```

## No legacy path to preserve

Unlike scalar (which carried a legacy `--trace-sweep` accumulator path), the DMA
tenant is new -- there is no prior DMA fuzzer in `src/fuzzer/`. Step 3 is
campaign-only (`run_campaign(&DmaDomain)`); no flag-gated legacy code.

## Acceptance

- `cargo test --lib` green (new DMA chain + DMA domain tests added).
- Vector and scalar tenants **unchanged**: the defaulted `Domain::compile` hook
  must not touch their fidelity -- vector 218/218 report + 24/0/0 replay still
  bit-identical; scalar smoke still clean.
- A DMA EMU-only smoke run reaches the universe and credits without panic across
  shim + memtile, both directions, all three dtypes, including a padding and a
  packet key; a short `--hw` campaign credits real silicon matches with per-op
  localization on any divergence.
- The generator provably emits only Stage-A-safe patterns (a test asserts
  footprint/field-width/address bounds for a large generated sample) -- no NPU
  wedge from a campaign run.

## Deferred (own later scope, never "marginal")

- **Compute-tile DMA** -- core-local (feeding the kernel, not DDR movement), a
  genuinely separate subsystem.
- **Stage-B wedge-prone edges** (decision 9) -- designed-for as a tier, built when
  Stage A is exhausted.
- **Timing mode** on this tenant -- per `2026-06-11-timing-model-derivability.md`,
  downstream of value coverage being solid.
- **Aiesim backend** for DMA -- errors out now; wired when the trace/timing modes land.
