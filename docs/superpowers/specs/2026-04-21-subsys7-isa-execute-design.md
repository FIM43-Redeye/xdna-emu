# Subsystem 7 -- ISA Execute -- Design

**Subsystem:** 7 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-21
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-21-subsys5-stream-switch-design.md](2026-04-21-subsys5-stream-switch-design.md)
**Planned tag:** `phase1-subsys-isa-execute`

---

## Goal

Land `phase1-subsys-isa-execute`. After this subsystem:

- All arch-specific execute-layer data lives in `xdna_archspec::aie2::*`
  submodules -- VMAC crossbar routing tables, processor-model timing
  values, per-tile-kind memory-size constants, and intrinsic-name
  indexing where the audit confirms they belong there.
- A tight `IsaExecutor` trait lives in `xdna_archspec::isa_execute`.
  The method count is audit-driven, expected 2-5 methods, covering the
  ops where AIE1/AIE2P has *shape* divergence from AIE2 (as opposed to
  *values* divergence, which is data and covered by the prior bullet).
- A concrete `Aie2IsaExecutor` ZST + `&'static dyn IsaExecutor`
  singleton follows the pattern established in Subsystems 3-5
  (`DmaModel`, `LockModel`, `StreamSwitchModel`).
- Execute algorithms in `src/interpreter/execute/*.rs` reach the trait
  via `arch_handle::isa_executor()` (new) and reach arch-specific data
  via the existing `arch_handle::*` accessors extended as needed.
- Adding AIE2P or AIE1 execute support is "implement a sibling
  `IsaExecutor` + populate a sibling submodule of archspec data" -- no
  edits to `src/interpreter/execute/*.rs` files.

Subsystem 7 is the largest of the eight Phase 1b subsystems by source
volume: 20 files under `src/interpreter/execute/` totalling ~1.2 MB of
Rust, plus the 9-file `src/interpreter/timing/` submodule. The subsystem
is executed **audit-first**: Task 1 produces
`docs/arch/subsys7-audit.md`, which catalogues divergence evidence
per-file; the trait surface and data-migration list are then finalized
from the audit's findings rather than decided upfront.

## Non-goals

- **No AIE1 or AIE2P `IsaExecutor` implementation.** Subsystem 7 ships
  the trait, carrier types if any, `Aie2IsaExecutor` impl, and the
  arch-specific data migrations. Populating AIE1/AIE2P is orthogonal
  future work, governed by the Phase 1 ground rule "no second-arch
  implementation during the refactor."
- **No restructuring of `Decoder`/`Executor`/`StateAccess` traits in
  `src/interpreter/traits.rs`.** Those are emulator-internal abstractions
  with the `CycleAccurateExecutor` as their sole implementor. They
  describe *how the emulator dispatches*, not *what the target arch
  does*. Out of scope.
- **No timing-subsystem trait seam.** Per brainstorming question 4,
  timing data migration is in scope (move AIE2-specific latency values
  from `interpreter/timing/latency.rs` into archspec where they aren't
  already, likely piggybacking on `ProcessorModel`), but pipeline
  algorithms (scoreboard, bank-conflict detection, hazard tracking)
  stay in xdna-emu as arch-generic infrastructure. A timing trait
  seam may appear later if AIE1 needs fundamentally different pipeline
  algorithms, but none of the current evidence requires it.
- **No proactive file splits.** `vmac_routing.rs` (234K), `memory/mod.rs`
  (121K), and `vector_arith.rs` (100K) are large enough to raise
  eyebrows, but splitting them is hygiene, not refactor. If data
  migration naturally shrinks a file (e.g., `vmac_routing.rs` moves
  wholesale to archspec and the xdna-emu side disappears), good; if
  not, we leave them at their current size. Standalone splitting goes
  to Subsystem 9 (Phase 2 hygiene) if warranted.
- **No AIE1-semantics backfill in the audit.** The audit reads llvm-aie
  TableGen and aie-rt header files for cheap cross-arch evidence, but
  does **not** spelunk aietools Python models (`mulmac.py`, `srs_ups.py`,
  `permute.py`, `constants.py`) unless a specific question the audit
  must resolve cannot be answered from open-source sources. aietools
  is reading reference, not a primary input here.
- **No hot-path dispatch change for AIE2.** Trait methods resolve to
  `&'static dyn IsaExecutor` with `OnceLock` caching; if any call site
  runs per-instruction, we measure before commit and specialize if
  needed (same escape hatch as DMA/LockModel). Current AIE2 call sites
  should produce byte-identical behavior before and after.
- **No second-arch implementation.** Phase 1 ground rule.
- **No new `SemanticOp` variants.** Scalar and vector dispatch are
  `SemanticOp`-driven and already arch-generic via the `xdna-archspec`
  `isa::SemanticOp` enum shipped in Subsystem 6. If the audit surfaces
  a SemanticOp that *only* makes sense on AIE2, that is a separate
  concern for the AIE1-landing pass.

---

## Context

### Where we are at the starting gate

Subsystem 5 (Stream Switch) landed at `phase1-subsys-stream-switch`
(`48fb17d`). Verified baselines at that tag, re-verified at HEAD
immediately before writing this spec:

- `cargo test --lib` = 2684 pass, 0 fail, 5 ignored
- `cargo test -p xdna-archspec --lib` = 297 pass, 0 fail, 2 ignored
  (the `test_full_parse_all_devices` 13-vs-12 failure fixed in
  Subsystem 3 stayed fixed)
- `cargo build --release` clean; no warnings added since
  `phase1-subsys-stream-switch`
- ISA 4815/4815 at last run; bridge smoke green at last `--no-hw`
  spot check

Stream switch was the fifth Phase 1b subsystem and the third tight
behavioral seam (DMA and Locks being the prior two). ISA Execute is
the sixth, with Parser (Subsystem 8) the final Phase 1b subsystem.

### What the ISA Decode design note said about execute

Subsystem 6 (ISA Decode) landed **without** a trait seam. Its design
note ([docs/arch/isa-decode.md](../../arch/isa-decode.md)) made a
clean separation:

> ISA decode is values. ISA execute is shapes.

And flagged execute specifically:

> Instruction *execution* semantics (Subsystem 7, not yet landed)
> almost certainly warrant an `IsaExecutor` trait: vector rounding,
> saturation, configuration-word interpretation, and accumulator
> precedence genuinely vary between arch families.

That note is a *prediction*, not an audit. The Subsystem 7 audit's
first deliverable is to confirm or refute it -- matching the
discipline Subsystem 5 applied when its audit revealed the presumed
`PortLayout` extension trait was 231 LOC of dead code.

### Current execute-layer surface

Pre-audit inventory of `src/interpreter/execute/`, grouped by
functional area:

| Area | Files | Approximate size |
|---|---|---|
| Dispatcher / orchestration | `mod.rs`, `semantic.rs`, `cycle_accurate.rs`, `vector_dispatch.rs` | 63K + 38K + smaller |
| Scalar / control / stream / cascade | `control.rs`, `stream.rs`, `cascade.rs` | 44K + 16K + 13K |
| Memory | `memory/mod.rs`, `memory/neighbor.rs` | 121K + 6.4K |
| Vector ALU | `vector_arith.rs`, `vector_compare.rs`, `vector_misc.rs`, `vector_pack.rs`, `vector_ups.rs`, `vector_srs.rs`, `vector_helpers.rs`, `vector_semantic.rs`, `vector_permute.rs`, `vector_float.rs`, `vector_config.rs`, `vector_convert.rs`, `vector_validate.rs` | 100K, 40K, 41K, 37K, 34K, 53K, 51K, 50K, 73K, 47K, 41K, 22K, 8.5K |
| VMAC / matmul | `vmac_routing.rs`, `vmac_hw.rs`, `vector_matmul/` | 234K + 69K + subdir |

Adjacent (not under `execute/` but entangled):

| Area | Files | Approximate size |
|---|---|---|
| Timing | `interpreter/timing/arbitration.rs`, `barrier.rs`, `deadlock.rs`, `hazards.rs`, `latency.rs`, `memory.rs`, `mod.rs`, `slots.rs`, `sync.rs` | 9 files; `latency.rs` has ~30 AIE2-specific references |

`vmac_routing.rs` is the single largest file and is **pure generated
data** -- 789 active crossbar m-bits, 15808 route entries, probed
from the AMD C++ ISS, zero algorithmic logic. Semantically it is
already archspec material; only the filesystem location is
xdna-emu-resident.

Scalar and vector dispatch consume `xdna_archspec::aie2::isa::SemanticOp`
directly (`execute/semantic.rs` for scalar, `execute/vector_dispatch.rs`
for vector) -- both already arch-parameterized by archspec data. The
`SemanticOp::Intrinsic(u32)` variant exists (indexing into
`ProcessorModel::intrinsic_names`) but is not currently routed in
either dispatcher; intrinsic dispatch is therefore latent rather than
active. Whether the audit reveals it should become active during
Subsystem 7 depends on evidence.

### Where the expected line sits

Going into the audit neutral per brainstorming question 2, but not
blind: prior subsystem patterns plus a quick pre-audit read suggest
three rough categories:

- **Arch-generic infrastructure.** Dispatcher orchestration
  (`cycle_accurate.rs` pipeline modeling, `vector_dispatch.rs` routing,
  `semantic.rs` scalar dispatch), most scalar/control code,
  scoreboard/hazard/bank-conflict detection in `timing/`. Algorithms
  that work for any VLIW with per-op latency, register hazards, and
  memory banks. Stays in xdna-emu; reads arch-specific constants via
  existing `arch_handle::*` accessors.
- **Arch-specific data.** VMAC crossbar routing tables
  (`vmac_routing.rs`), per-instruction latencies in
  `timing/latency.rs` where not already archspec-resident, memory-size
  constants per tile-kind where not already archspec-resident,
  intrinsic-name indexing where the audit confirms it's load-bearing.
  Moves to archspec. No trait method required.
- **Arch-specific shape.** The ops with genuinely different *algorithms*
  per arch: candidate list is VMAC expansion (different crossbar
  shape, different accumulator paths), SRS / UPS / Pack / Convert
  rounding+saturation (configurable via operation config word, but
  semantics genuinely differ between arches), accumulator-width
  promotion rules in mixed-width chains. Gets trait methods.

The audit refines this three-way split with evidence. Landing Approach
B means the third category is non-empty; landing Approach A (collapsed
down to zero trait methods) is possible if the audit shows every
expected shape divergence is actually data-parameterizable. Either
outcome is acceptable; the spec commits to whichever the audit
produces.

---

## The question to settle

"Where exactly does the trait surface land?" -- and the audit answers
it. The spec commits to the audit *discipline*, not to a trait
surface.

---

## Proposed design

### Audit (Task 1)

Artifact: `docs/arch/subsys7-audit.md`. Per-file deep dive per
brainstorming question 3 option B.

Organization by functional area (not alphabetical), matching the
grouping in the context section:

1. Dispatcher / orchestration
2. Scalar / control / stream / cascade
3. Memory
4. Vector ALU
5. VMAC / matmul
6. Timing

Per-file subsection template:

- **Size + responsibility.** One sentence summarizing what the file
  does in the execute pipeline.
- **AIE2 hardcode count.** Grep-based count of AIE2-specific constants,
  magic numbers, and arch-branded identifiers (literal "AIE2",
  `AIE_ML_*` aie-rt macro mirrors, hardcoded sizes like `512`
  for accumulator width).
- **Divergence risks vs AIE1/AIE2P.** Evidence from file names,
  comments, llvm-aie TableGen `AIE1InstrInfo.td` vs `AIE2InstrInfo.td`
  where cheap, aie-rt per-arch header splits where cheap.
- **Prescribed migration verb.** One of `move-to-archspec`,
  `read-archspec-via-accessor`, `wrap-in-trait`, `leave-alone`.
- **Estimated LOC impact.** Rough count of lines changing on the
  xdna-emu side + lines added on the archspec side.

Two files get ~2 pages each in the audit because their
"move-vs-wrap-vs-split" call is non-obvious:

- **`vmac_routing.rs`**: pure static-data file, 234K. Questions the
  audit resolves: wholesale move to archspec (expected yes) or
  re-generate into archspec's `build.rs` pipeline? Any AIE1 crossbar
  table in the toolchain that would inform data shape? Any
  consumer-side coupling to the file's current crate that makes the
  move costly?
- **`memory/mod.rs`**: 121K of load/store handlers. Questions the
  audit resolves: how much is arch-generic (element-wise dispatch by
  type and size) vs arch-specific (memory-bank topology, row-offset
  arithmetic, tile-kind-specific address layout)? Does the audit
  justify wrapping the load/store dispatch in an `IsaExecutor` method,
  or does the memory hierarchy stay as arch-generic code that reads
  per-tile-kind archspec constants?

The audit's closing section must answer three questions:

1. **Trait method list.** Best read of the evidence: which ops
   genuinely have shape divergence, named with a tentative method
   signature each. Zero methods is a valid answer.
2. **Data-migration list.** What moves to archspec, destination
   module path, and migration pattern (wholesale file move vs
   extract-constants vs extend-existing-module).
3. **AIE1 projection** (~100 words). Per parent-refactor rule.

Audit length target: 8-12 pages (7-10 files at paragraph length plus
two deep-dive files at ~2 pages each plus the closing three-question
section).

### Trait scaffold (Task 2)

Candidate `IsaExecutor` shape, sized by the audit. This is a best-guess
starting point written here to give Task 2 something to confirm or
refute, not a commitment.

```rust
// xdna_archspec::isa_execute
pub trait IsaExecutor: Send + Sync + core::fmt::Debug {
    /// Apply arch-specific SRS rounding + saturation to the accumulator
    /// + shift + configuration-word triple. Returns the narrowed vector
    /// plus any sticky flags the caller propagates.
    fn apply_srs(&self, acc: AccumValue, shift: i8, config: u32) -> SrsResult;

    /// Expand a VMAC routing m-bit + pmode combination into the concrete
    /// input-lane mapping. Backs the former `vmac_routing.rs` data from
    /// inside archspec after wholesale move.
    fn vmac_route(&self, mbit: u16, pmode: u8) -> VmacRoute;

    /// True if accumulator-width promotion follows AIE2 rules in
    /// mixed-width VMAC chains. AIE1 narrower vector unit may differ.
    /// Hedge included to confirm/refute during audit; drops if AIE1
    /// turns out to match.
    fn accumulator_promotion_rule(&self) -> AccumPromotionRule;
}
```

Three methods as the opening bid. Audit-driven outcomes:

- **Shrinks to zero**: if every listed candidate turns out to be
  data-expressible -- rounding modes are already config-word fields
  whose semantics are the same across arches; VMAC routing is data,
  expressible as a `&'static VmacRoutingTable` accessor; accumulator
  promotion rules are the same. Lands Approach A (no trait).
- **Stays at 2-3**: expected outcome if AIE2-vs-AIE1 has the
  handful of shape divergences the ISA Decode note predicted.
- **Grows toward 5+**: if memory-hierarchy load/store dispatch or
  intrinsic dispatch needs trait-level selection; unlikely per
  pre-audit hypothesis, but the audit decides.

### Data migration (Tasks 3-5, exact split TBD from audit)

Expected items, pre-audit:

| Source | Destination | Pattern |
|---|---|---|
| `src/interpreter/execute/vmac_routing.rs` (234K static tables) | `xdna_archspec::aie2::vmac::routing` | Wholesale file move; consumer in `vmac_hw.rs` reads via archspec |
| AIE2-specific constants in `src/interpreter/execute/vmac_hw.rs` | `xdna_archspec::aie2::vmac::hw` | Extract constants |
| AIE2 latency values in `src/interpreter/timing/latency.rs` (~30 refs) | extend `xdna_archspec::aie2::isa::generated` or add `xdna_archspec::aie2::timing` | Probably already partially archspec-driven via `ProcessorModel`; finish the job |
| Memory-size / address-layout constants in `memory/mod.rs` | `xdna_archspec::aie2::memory` (extend if existing) | Verify no duplicates |
| Intrinsic-name indexing if the audit reveals active usage | already in `ProcessorModel` (archspec) | No migration; reach via existing archspec path. If audit reveals `SemanticOp::Intrinsic(u32)` needs active dispatch (currently latent, see Context), Task 6 adds a trait method |

Call-site migrations inside xdna-emu are grep-driven: for each moved
constant, replace the direct-in-file reference with an
`arch_handle::*` accessor. For each wrapped operation, replace the
direct implementation with `arch_handle::isa_executor().method(...)`.

### Runtime wiring (Task 2b)

Following Subsystem 5's pattern:

```rust
// src/device/arch_handle.rs (extending)
static ISA_EXECUTOR: OnceLock<&'static dyn IsaExecutor> = OnceLock::new();

pub fn isa_executor() -> &'static dyn IsaExecutor {
    *ISA_EXECUTOR.get_or_init(|| {
        xdna_archspec::runtime::default_arch().isa_executor()
    })
}
```

And in `xdna-archspec/src/runtime.rs`:

```rust
impl ArchConfig {
    fn isa_executor(&self) -> &'static dyn crate::isa_execute::IsaExecutor {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::isa_execute_model::AIE2_ISA_EXECUTOR
            }
            Architecture::Aie => unimplemented!(
                "AIE1 IsaExecutor not populated; refactor exits at AIE2."
            ),
        }
    }
}
```

A dispatch unit test in `runtime.rs` confirms the accessor resolves
to the AIE2 impl, same shape as
`stream_switch_model_dispatches_to_aie2_for_aie2_family`.

### Testing plan

- **Invariant.** `cargo test --lib` green at every commit. Subsystem-7
  entry baseline: xdna-emu 2684/0/5, archspec 297/0/2. Drift at any
  commit is a regression.
- **Drift tests.** Each new archspec data module gets a
  drift-detection test asserting the hand-written aggregate matches
  generated constants, mirroring
  `aie2_topology_matches_generated_constants` from Subsystem 5.
- **Migration tests.** Each `wrap-in-trait` call-site migration ships
  with a before-after equivalence test in `interpreter::execute::tests`,
  ensuring the trait-dispatched path produces the same result as
  direct invocation. Where a direct invocation already has coverage,
  the existing test suffices.
- **Dispatch test.** `runtime.rs` gets a test asserting
  `default_arch().isa_executor()` dispatches to `Aie2IsaExecutor`
  for AIE2/AIE2P.
- **Bridge smoke after each migration task.**
  `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` catches
  90% of behavioral regressions in ~30s.
- **Full bridge + ISA at the subsystem tag.** Rebuild FFI via
  `cargo build -p xdna-emu-ffi` immediately before the bridge run --
  stale `.so` files produce phantom regressions (documented in
  NEXT-STEPS.md).

### Rollout order

Expected task split (exact numbers per the plan):

1. Audit (`docs/arch/subsys7-audit.md`).
2. Trait scaffold: `xdna_archspec::isa_execute` module + placeholder
   carriers + `Aie2IsaExecutor` ZST + `&'static dyn IsaExecutor`
   singleton + `ArchConfig::isa_executor()` dispatch method +
   `arch_handle::isa_executor()` accessor + dispatch test.
3. Data migration: `vmac_routing.rs` wholesale move to
   `xdna_archspec::aie2::vmac::routing` + consumer update in
   `vmac_hw.rs`.
4. Data migration: AIE2 latency values from `timing/latency.rs` ->
   archspec (extending `ProcessorModel` or new module per audit).
5. Data migration: memory-hierarchy constants if the audit finds any
   outside archspec.
6. Trait-site migrations: one task per method added to `IsaExecutor`.
7. Completion: audit's `## Completion` section + `NEXT-STEPS.md`
   update + tag `phase1-subsys-isa-execute`.

Task 1 (audit) MUST complete before Task 2+ because the audit
determines the trait surface. Tasks 3-5 are independent of each
other and may run in any order (or concurrently if the plan
chooses) once the scaffold is in place. Task 6 is serialized by
the trait surface.

## AIE1 projection

~100 words per parent-refactor rule:

An AIE1 port would implement `Aie1IsaExecutor` as a ZST sibling to
`Aie2IsaExecutor` under `xdna_archspec::aie1::isa_execute_model`. Its
SRS/UPS methods would apply AIE1's narrower rounding semantics (AIE1
vector unit is 128-bit where AIE2 is 256-bit, different saturation
edges, different config-word layout), its VMAC routing methods would
return AIE1's smaller crossbar tables (fewer m-bits, different pmode
layout, probably a smaller generated file), and its accumulator
promotion rule would reflect AIE1's simpler width hierarchy (no
`v8acc64`). Data would live under `xdna_archspec::aie1::{vmac,
timing, memory}`. Execute algorithms in `src/interpreter/execute/*.rs`
would not change -- they dispatch through the trait for the 2-5 seam
methods and read arch-specific data via `arch_handle::*` accessors.
Scalar and vector dispatch in `semantic.rs` and `vector_dispatch.rs`
already work for AIE1 because they are `SemanticOp`-driven and
`SemanticOp` is arch-generic.

## Alternatives considered

### Approach A -- data in archspec, algorithms stay in xdna-emu, no trait

Migrate arch-specific data (VMAC routing, timing, memory-size
constants) to archspec. Execute algorithms stay in xdna-emu and read
arch-specific data via `arch_handle::*` accessors. Follows the ISA
Decode pattern exactly.

**Rejected** for the expected-value path because the ISA Decode
design note flagged specific ops (SRS rounding, VMAC accumulator
rules) as shape-divergent, and our brainstorming agreed that "shape"
does exist here. But Approach A remains the landing if the audit
disproves the flag: a zero-method trait is the degenerate case of
Approach B, and the spec allows the audit to land there if the
evidence supports it.

### Approach C -- full `IsaExecutor` covering the dispatcher layer

Move the dispatch layer (`execute_semantic`, `VectorAlu::execute`,
`MemoryUnit::execute`, etc.) behind trait methods. Per-arch submodule
owns the entire execute pipeline. xdna-emu shrinks; archspec grows.

**Rejected** for three reasons:

1. Most dispatch is already `SemanticOp`-driven and arch-generic. A
   trait layer on top of that adds ceremony without real divergence.
2. Duplication risk: the AIE2 impl would re-export large amounts of
   the current xdna-emu execute code; the AIE1 impl (when eventually
   built) would share most of it; the trait would be a pass-through
   for most methods.
3. Prior subsystems (DMA, Locks, StreamSwitch) all landed at the
   "tight trait for the seams that actually vary" shape, and that
   shape has proven right each time. Approach C goes against that
   pattern without evidence the pattern fails here.

The user's instinct pulled to C during brainstorming, and the
rejection is "if we're duplicating code, it's not worth it"
-- accurate read.

### Pre-audit trait commitment (Approach B without audit-first)

Commit upfront to a specific trait method list (`apply_srs`,
`vmac_route`, `accumulator_promotion_rule`), populate AIE2, migrate
sites, ship. Skip the audit artifact.

**Rejected.** This is the approach that *would have* landed
Subsystem 5 with its presumed-necessary `PortLayout` extension trait
intact, when the audit revealed it was 231 LOC of dead code.
Audit-first is cheap insurance against committing to a trait shape
that doesn't match the ground truth. Subsystem 7 is the largest of
the eight; cheap insurance is especially worth it here.

### Sub-subsystem decomposition (7a/7b/7c/...)

Split Subsystem 7 into 7a = scalar/control, 7b = vector ALU, 7c =
VMAC, etc. Each gets its own tag.

**Rejected** per brainstorming question 1. The `IsaExecutor` trait
is cohesive; splitting would force 7a to predict what 7b-7e need
from the trait, which is exactly the wrong prior. Landing as one
subsystem with multiple tasks (the same pattern as every prior
subsystem) handles the scale without the sub-sub coordination cost.

## Follow-ups to flag (expected out-of-scope items surfaced by audit)

Expected audit surfacings:

- **`memory/mod.rs` internal structure.** If the audit reveals the
  121K file conflates unrelated responsibilities, mark for Subsystem
  9 (Phase 2 hygiene). Not a Subsystem 7 concern.
- **`vmac_hw.rs` algorithmic tangles.** Same call: hygiene, not
  refactor.
- **Pre-existing rot.** rust-analyzer warnings in generated files,
  the `bd_chain_repeat_on_memtile` bridge deadlock, other known
  pre-existing failures listed in NEXT-STEPS.md. Carried forward;
  do not block on them.
- **AIE1-landing-pass prep.** Any data or trait concern that would
  need to be revisited when AIE1 is actually populated. Captured in
  the audit's `## Completion` section at subsystem close.
- **Intrinsic-dispatch activation.** If the audit confirms
  `SemanticOp::Intrinsic(u32)` is load-bearing for some ops that are
  currently falling through to fallback paths, flag for either late
  in Subsystem 7 or a follow-on subsystem depending on size.

## Summary

Subsystem 7 is ISA Execute, audit-first, biased toward Approach B
("tight trait + data in archspec"). The audit is the primary
technical artifact; the trait surface is audit-driven (expected 2-5
methods, but zero or more is accepted). Data migrations include
`vmac_routing.rs` wholesale, timing values, and memory-size
constants where not already archspec-resident. File splits and
other hygiene are deferred to Subsystem 9. The refactor's standing
rule ("no second-arch implementation during the refactor") holds:
we ship the seam, not the AIE1/AIE2P implementation.
