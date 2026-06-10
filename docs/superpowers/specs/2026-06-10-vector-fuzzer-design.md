# Vector Differential Fuzzer (Phase D, #112) — Design

**Date:** 2026-06-10
**Status:** Approved by Maya (sections 1-3 individually)
**Context:** Final phase of the vector-verification campaign
(`docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md`).
Phases A-C silicon-verified SRS/UPS/Pack/Convert/MAC exhaustively. Phase D
covers everything else in the vector op space, and the resulting ledger is the
evidence for flipping `clean_release(Aie2)` green (#113). **Phoenix-gated**:
the silicon leg must run on this box before the board/APU swap kills NPU1.
Completeness and correctness are THE core goal; speed is secondary.

## Goal

Functional + core-timing equivalence between emulator and NPU1 silicon for the
silicon-unverified vector op families: shuffle routing (40+ modes), shifts,
compares/selects, min/max, conditional arithmetic, bitwise, data movement
(align/broadcast/extract/insert/push/clear), unpack -- across element types and
config modes. Strategy 3 of 3 considered: a **typed-pipeline generator with a
coverage ledger** (random AST extension and exhaustive single-op sweep were
rejected for the primary build: random can't prove coverage, sweep gets one op
per ~20s compile). Binary-edit single-op coverage stays available later as
gap-filler.

## Kernel shape and I/O contract

Each fuzz case is one compute kernel: a typed chain of 8-16 vector ops.

```
v0 = load(in slice 0)              // edge-weighted random inputs
v1 = op1(v0[, load(in slice k)])   // op drawn from the typed table
store(out slice 1, v1)             // EVERY stage observed
v2 = op2(v1, ...)
store(out slice 2, v2)
...
```

- **Typed op table**: each entry declares family, `aie::` intrinsic template,
  input/output vector types, and mode dimensions (rounding/saturation/shuffle
  mode). Chains compose only type-legally, so every generated kernel compiles.
- **Input pool**: edge-weighted random vectors (denormals, NaN/Inf, sign
  extremes, zero-runs -- the input classes that found the phase B/C bugs).
  Two-operand stages load the second operand fresh from the pool.
- **Per-stage stores**: one 64-byte output slice per stage. Mismatch
  self-localizes to its stage (no case reduction), defeats DCE, and gives 8-16
  silicon-verified ops per ~20s compile instead of 1.
- **Type couplers**: type changes between stages route through phase-A-verified
  ops (SRS/UPS/Convert) -- verified couplers carrying unverified payload.
- **Mode config**: mode-bearing stages set `aie::set_rounding`/`set_saturation`
  per the ledger's pick.
- **Compiler**: Peano (light, exempt from the Chess -j4 RAM rule). Spike
  confirms intrinsic reach per family; unreachable families get a Chess
  fallback at <=4 jobs.

## Coverage ledger

The ledger turns the fuzzer into a completeness proof.

- **Keys**: `(SemanticOp, element_type, mode)` -- the dimensions the dispatcher
  and `crates/xdna-archspec/src/coverage/` already use. Target N = 10 hits/key
  from distinct chains.
- **Execution-derived credit**: a fuzz-mode hook on `vector_dispatch.rs`
  records executed keys per case. Credit lands only if the whole chain matches
  silicon. Intent can fold under optimization; emulated execution + silicon
  agreement cannot lie.
- **Coverage-first generation**: each chain greedily targets least-covered
  keys; random within a key.
- **Persistent + resumable**: `ledger.json`; `fuzz-vector --report` lists
  uncovered keys. Empty-at-N is the #113 evidence.
- **Failures bank, never block**: divergent case -> inputs + silicon output +
  trace archived; key marked DIVERGENT; campaign continues. Fixes can land
  post-Phoenix against the banked golden via replay.

## Runner, timing, banking

- **Runner**: reuse `src/fuzzer/runner.rs` wholesale (parallel Peano compile,
  serial HW, EMU compare, panic-as-CRASH). Vector path is a new `FuzzParams`
  variant; `fuzz_template.py` unchanged (i32 memref view, byte-sized buffers).
- **Core-timing leg**: trace always on. Outputs compare bit-exact; PC-anchored
  core traces compare via the existing trace machinery. Timing divergence on a
  clean-output case = separate TIMING-FAIL class with its own ledger flag --
  functional and timing claims stay distinguishable.
- **Survival corpus**: per case archive kernel source, seed/spec JSON,
  `npu_output.bin`, `npu_trace.bin` to
  `~/npu-work/experiments/phoenix-survival/vector/` (durable, not `build/`).
  `fuzz-vector --replay <dir>` re-runs EMU against banked silicon -- the
  post-Phoenix regression gate.

## Error handling

- Compile-fail = generator bug -> key marked UNREACHABLE with reason, surfaced
  in `--report`. No silent drops.
- HW timeout/wedge -> existing recovery chain + skip.
- Emulator unimplemented decode -> CRASH bucket (real finding).

## Testing / definition of done

TDD throughout: table well-formedness; 1k-seed compile-clean generation;
ledger arithmetic; localization test against a deliberately broken table
entry. Done = clean `cargo test --lib`, 50-case HW+EMU smoke, then campaign to
ledger-complete (every reachable key at N, divergences banked + triaged).

## Deferred follow-ups (explicitly out of scope)

- Binary-edit single-op coverage for keys the typed pipeline can't reach.
- Chess genuine multi-file/batch compilation investigation (also a
  prerequisite for binary-editing a Chess baseline).
- Scalar-AST x vector interaction mixing (approach 1 thin slice).
- DMA-timing equivalence (stays ballpark-deterministic per prior posture).
