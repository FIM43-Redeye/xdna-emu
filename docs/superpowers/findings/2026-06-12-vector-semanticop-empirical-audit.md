# Vector SemanticOp empirical audit (#126)

**Date:** 2026-06-12
**Purpose:** Establish, for the `clean_release(Aie2)` coverage gate, exactly which
`Category::Vector` SemanticOps the vector differential fuzzer (#112/#114) has
silicon-verified -- so the axis-1 `Verified` override claims real evidence and
nothing more. Over-claim is a correctness bug; under-claim is safe.

## Why empirical, not static

A vector fuzzer family emits an **aie_api C++ expression** (`aie::add`,
`aie::mmul`, `aie::pack`, `aie::min`, ...). Peano compiles that to AIE ISA, and
the emulator's decoder classifies the *compiled* instructions into SemanticOps.
The family name is therefore NOT the SemanticOp -- the mapping runs through
Peano's opaque lowering. Reading it statically is where guessing creeps in (and
it did: see "Corrections" below).

The rigorous mapping is observed, not inferred: the emulator already records
every executed vector SemanticOp via `src/interpreter/execute/fuzz_recorder.rs`
(hook at `vector_dispatch.rs`, key `"{semantic:?}/{et:?}/m{mode}"`), and the
vector domain captures it per case as `VecObs::executed`, banked to
`executed.json`. So for every silicon-matched fuzzer run we have ground truth:
the exact SemanticOps the emulator dispatched while producing an output that
matched real NPU1 byte-for-byte.

## Method

1. **Corpus:** `~/npu-work/experiments/phoenix-survival/vector/` -- the banked
   silicon-matched corpus from the #112/#114 campaign. 24 live (replayable,
   pool-bearing) seeds + 45 archived (pre-218-extension, no pool) seeds; every
   seed carries `executed.json` (the SemanticOps that ran when it matched
   silicon).
2. **Currency:** `xdna-emu fuzz-vector --replay <corpus>` -> **24 match, 0
   divergent, 0 error**. The current emulator still reproduces banked NPU1
   output for all replayable seeds. Because the executed-op set is a
   deterministic function of (xclbin, emulator code), a current output-match on
   the same xclbin locks the banked `executed.json` as a faithful record of
   *current* dispatch.
3. **Verified set:** union of `executed.json` over the live-24 (currently
   replay-confirmable) seeds, intersected with `Category::Vector`.

## Result: 20 Vector-category SemanticOps verified

Executed in current silicon-matched replay (live-24 occurrence count in
parens; the count reflects the curated corpus size, not verification depth --
the full campaign fuzzed each key 10+ times):

    Convert(32) VectorBroadcast(27) VectorSelect(25) MatMul(24) Mac(21)
    VectorClear(15) Accumulate(15) MaxLt(14) AccumSub(13) VectorInsert(12)
    VectorExtract(12) MinGe(12) Srs(11) Ups(10) Align(6) NegLtz(2) NegGtz(2)
    MaxDiffLt(2) Shuffle(1) AbsGtz(1)

## Corrections to the prior static guess

- **`Min`/`Max` never execute.** Integer `aie::min`/`aie::max` lower to the
  hardware's *fused* compare-select instructions (`vmax_lt`/`vmin_ge`), which
  the emulator classifies as **`MaxLt`/`MinGe`**, not `Min`/`Max`. The static
  guess mapped the `min`/`max` families to `Min`/`Max`; empirically those
  SemanticOps are dead for this corpus. Same pattern feeds the `abs`/`neg`/
  `maxdiff` families -> `AbsGtz`/`NegGtz`/`NegLtz`/`MaxDiffLt`.
- **`Align`, `VectorClear`, `Accumulate`, `AccumSub`, `VectorExtract`,
  `VectorInsert`** were flagged "fuzzy/uncovered" statically but genuinely
  execute and match silicon (they are the real lowering of the coupler,
  matrix-accumulator, and movement families).

## Deferred (honest under-claim)

- **`Pack`/`Unpack`** executed+matched only in the 45 **archived** seeds
  (pre-218 table, no banked pool -> `load_banked` returns `Skip`, not currently
  replayable). They were silicon-verified at campaign time but I cannot
  *currently* replay-confirm them, so they stay perishable. Re-flip by
  re-banking pack16/pack8/unpack16/unpack32 seeds with a pool (or a small HW
  re-fuzz). Tracked as a follow-up, NOT claimed.
- **Genuinely uncovered** (never executed; correctly stay perishable):
  `Min`, `Max` (dead -- see above), `MatMulSub`, `NegMatMul`, `AddMac`,
  `SubMac`, `NegMul`, `NegAdd`, `AccumNegAdd`, `AccumNegSub`, `VectorPush`,
  `VectorPushHi`, `SubLt`, `SubGe`.

## Gate impact

The 20 verified ops leave the perishable queue. `clean_release(Aie2)` stays
**red**: the remaining ~14 unclaimed Vector ops + Pack/Unpack + the unclaimed
SideEffect ops (stream/cascade, tenants 4/5) are still perishable. The override
lives in `crates/xdna-archspec/src/coverage/units.rs` (`vector_ops_verified`),
mirroring the DMA-ops override (#113): `AietoolsModeled -> Verified`, evidence =
this audit.
