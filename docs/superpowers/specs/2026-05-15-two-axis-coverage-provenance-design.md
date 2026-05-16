# Two-Axis Coverage Provenance Infrastructure

- Date: 2026-05-15
- Status: Approved design, pre-implementation
- Scope: New `coverage` subsystem in `xdna-archspec` + a surface probe and
  reconciliation test in the interpreter crate. Supersedes the
  hand-maintained `docs/coverage/architecture-index.md`.

## Problem

The emulator keeps discovering subsystems it "missed" -- not because the
behavior was absent from the code, but because the coverage *catalogue*
silently failed to track it. Concrete instance caught during this design:
every vector-compute SemanticOp has a real handler in the interpreter, yet
`architecture-index.md` -- the document whose entire job is "do not let us
miss anything" -- has no vector-compute row at all.

Root cause: the node set is **generated** from authoritative sources
(llvm-aie TableGen, the AM025 register JSON, the device-models JSON, aie-rt),
but the coverage verdict is **hand-typed into markdown**, structurally
decoupled from the nodes. Nothing fails when a node has no verdict. The nodes
were never missing; the *binding between node and verdict* was missing, and
its absence was silent.

This matters acutely because NPU1 (Phoenix/AIE2) hardware may be retired when
the developer upgrades the mainboard. There is no time pressure to race
before then -- the requirement is to reach a state where releasing the
hardware is a *provably safe decision* rather than an act of hope: every
behavioral fact either toolchain-ground-truth, silicon-verified with
recorded evidence, or explicitly accepted with stated rationale. Nothing
silently approximated.

## Goals

- Make "we forgot a whole subsystem exists" structurally impossible for
  anything any authoritative source surfaces.
- Bind every coverage verdict to its generated node in the type system, so
  absence is a hard failure (the `Confirmed<T>` philosophy already in the
  crate, extended from "do my sources agree on a constant" to "do we
  understand and have we verified this behavior").
- Produce a mechanical, self-generating "what still needs NPU1" queue, with a
  single-command gate that defines "safe to release the hardware."
- Spend scarce human judgment only on the behaviorally subtle, perishable
  ~10%; auto-cover the toolchain-ground-truth long tail.

## Non-goals

- Discovering hardware behavior present in *no* source (truly undocumented
  silicon). That is empirically discoverable only (trace sweep, future
  differential fuzzer) and is named as an explicit unenforceable tier, not
  solved here.
- Replacing the interpreter's execution handlers or the existing generators.
  This is a coverage-tracking layer over what already exists.
- Performance work. This is a correctness/completeness instrument.

## Section 1 -- The two axes and the verdict vocabulary

### Axis 1: Surface presence (fine node, mechanically derived, compiler-enforced)

Per generated node -- every SemanticOp, register, hardware table -- a
`SurfaceClass`:

- `Wired` -- a real execution handler / register consumer exists.
- `Fallthrough` -- reaches a `_ =>` default; no dedicated handling.
- `Absent` -- decoded but nothing consumes it.

Not hand-assigned. The `SurfaceClass` type and its contract are *defined* in
archspec (archspec is the single point of definition for both axes). The
per-node *evidence* -- "does a handler exist in this build of the
interpreter" -- is an empirical fact about the emulator's implementation
state, not about the AIE2 architecture, so it is supplied by the interpreter
through a `SurfaceProbe` contract archspec declares (Section 2). The concrete
probe is a single exhaustive `match` over `SemanticOp` (and register-group
equivalents) that the Rust compiler forces to be total: a new TableGen
instruction or SemanticOp variant stops the interpreter build until it is
classified.

### Axis 2: Behavioral provenance (behavioral unit, hand-adjudicated, build-enforced)

Two orthogonal fields per behavioral unit.

Provenance -- where the behavioral knowledge comes from:

- `ToolchainDerived` -- fully specified by aie-rt / TableGen / regdb. The
  toolchain is ground truth. No silicon needed, ever.
- `AietoolsModeled` -- reimplemented from reading aietools python models /
  AM020. Not silicon-checked.
- `DocSpecified` -- from AM020/AM025 prose only.
- `Unspecified` -- a trusted source describes the node and we assert *no
  model*. "TableGen says this exists and we do not know what it is."

Verification -- the NPU1-perishable tracker, orthogonal to provenance:

- `NotApplicable` -- toolchain-derived; silicon cannot be "more right" than
  the spec.
- `Verified { evidence }` -- checked against NPU1, with a pointer to the
  bridge test / trace / finding doc that proves it.
- `Unverified` -- needs NPU1 to confirm; not done yet.
- `Accepted { rationale }` -- explicitly signed off as "good enough, will not
  verify," by a human, with reasoning recorded.

### Default to ignorant, not to trusting

The compiler-forced classification makes a human *make a decision*; it cannot
make a tired human *understand*. Therefore there is an explicit
`NeedsTriage` category whose default verdict is `Unspecified`. The only way
out is a positive, reviewed assertion of comprehension. The lazy path does
not land in a quiet `ToolchainDerived`; by construction it lands in the loud
set. Silence equals warned.

### The two honesty-failure sets (mechanical filters)

These are distinct failures, surfaced as two separate committed artifacts:

- Perishable queue -- *modeled, weak provenance, unverified*:
  `provenance in {AietoolsModeled, DocSpecified} AND verification == Unverified`.
  "We think we know; we have not silicon-checked." Artifact:
  `docs/coverage/perishable-queue.md`.
- Comprehension gaps -- *a trusted source describes it and we assert no
  model*: effective verdict `Unspecified`. "TableGen says this exists and we
  do not know what it is." Artifact: `docs/coverage/comprehension-gaps.md`.

`Accepted` is deliberately distinct from `Unverified`: an accepted gap is a
recorded, reasoned decision and never appears in either set. A silent
approximation cannot hide as "probably fine" -- it is verified, explicitly
accepted, or glowing red in one of the two files.

## Section 2 -- Module structure and the ownership seam

Principle: archspec is the single point of definition for *both* axes. The
only thing that cannot live in archspec is the per-node Axis-1 *evidence* --
"does a handler exist in this build of the interpreter" -- because that is a
fact about the emulator's implementation state, not about the AIE2
architecture, and archspec's independence from the emulator's implementation
is load-bearing. That evidence enters through a trait archspec *declares* and
the interpreter *implements*; the definition is never fractured.

xdna-archspec owns the durable spec, both axis definitions, and Axis-2
enforcement (it already holds the generated node universe and a `build.rs`):

```
crates/xdna-archspec/src/coverage/
  mod.rs        - CoverageModel<P: SurfaceProbe>: queryable model + two filter sets
  surface.rs    - SurfaceClass type; the SurfaceProbe trait (Axis-1 contract)
  verdict.rs    - Provenance, Verification enums; BehavioralUnit { id, claims, verdict }
  units.rs      - the explicit OVERRIDE registry + the CapabilityDomain list
  derive.rs     - taxonomy rules: fine node -> default behavioral unit
  enforce.rs    - build-time assertions, called from build.rs
```

`SurfaceProbe` is the Axis-1 contract: `fn surface_class(&self, node:
NodeId) -> SurfaceClass`. `CoverageModel` is generic over a probe. Both axes
are *defined* here; the model simply does not know the answer to Axis-1 until
a probe is injected.

`enforce.rs` runs from the existing `build.rs`, with the full generated node
set in hand. It enforces Axis 2 only: every generated node resolves to
exactly one behavioral unit; every unit carries a non-default verdict; every
`CapabilityDomain` is claimed by at least one unit. It does not know about
handlers -- Axis-2 enforcement needs no probe.

The interpreter supplies the only thing it must: a concrete `impl
SurfaceProbe` (`src/interpreter/coverage/surface_probe.rs`) -- a single
compiler-enforced exhaustive `match` over `SemanticOp` (and register
equivalents). It defines nothing; it answers an archspec-defined contract. A
new variant breaks the interpreter build here, where handler knowledge lives.
This is the Axis-1 forcing function, sited at the only place that can observe
the fact.

One reconciliation test (`src/interpreter/coverage/`, integration test) wires
the concrete probe into archspec's `CoverageModel` and:

1. asserts consistency -- e.g., a unit marked `Verified` whose nodes are all
   `Absent` is a contradiction (test red);
2. regenerates `docs/coverage/architecture-index.md` as a rendered view and
   fails if the committed file is stale (the index becomes generated output,
   never hand-edited again);
3. emits `docs/coverage/perishable-queue.md` and
   `docs/coverage/comprehension-gaps.md` as checked-in, diffable artifacts.

Definitions are single-sourced in archspec. The interpreter contributes one
trait impl plus the wiring test. Two forcing functions -- archspec's build
panic (Axis 2) and the interpreter's compiler-enforced probe (Axis 1) -- each
fire in the crate that owns the corresponding decision.

## Section 3 -- Derived default + override clustering (Approach C)

Rule: every fine node resolves to exactly one behavioral unit, by derivation
unless explicitly overridden.

Derived default (`derive.rs`) -- covers the toolchain-ground-truth ~90%:

- SemanticOps cluster by category. The category comes from a third
  compiler-enforced exhaustive `fn category(SemanticOp) -> Category`
  (arithmetic / bitwise / comparison / memory / control-flow / vector / sync
  / side-effect / NeedsTriage). New variant -> must categorize -> build red.
  Default unit id = `semantic.<category>`.
- Registers cluster by `(subsystem_kind, module_kind)` -> `reg.dma`,
  `reg.lock`, `reg.stream_switch`, etc.

Each category/group carries an honestly-pessimistic default verdict (Section
5). The derived layer needs zero hand-maintenance and is un-missable: a new
instruction or register flows into its bucket automatically.

Override registry (`units.rs`) -- hand-authored, only where taxonomy is the
wrong grain. This is exactly the perishable surface and only it:

```
BehavioralUnit {
  id: "vmac.config_word_rounding",
  claims: Claims::semantic_subset(
            &[Mac, MatMul, MatMulSub, NegMatMul, AddMac, SubMac],
            Aspect::ConfigWordRounding),
  verdict: Verdict { provenance: AietoolsModeled, verification: Unverified },
}
```

Overrides carve sub-aspects a category bucket would blur:
`vmac.config_word_rounding`, `vmac.sparsity_mask`, `srs.guard_sticky`,
`srs.rounding_modes`, `ups.promotion`, `matmul.tile_geometry`, the hardware
data tables, the timing constants. Roughly 80-150 units; the only things a
human writes by hand.

Precedence and the no-silent-shadow guarantee:

- A node touched by any override resolves to that override; the derived
  bucket no longer claims it.
- `enforce.rs` asserts partition, not merely cover: exactly one unit per
  node. Two overrides claiming one node -> build panic. An override claiming
  a vanished node -> build panic.
- The dangerous case -- an override silently pulling a node off the
  toolchain-derived path into a weaker verdict -- requires the override to
  declare `shadows_derived = true` with a one-line reason; the reconciliation
  test panics on an undeclared shadow. Demotion from "toolchain-true" to
  "approximated" is then a conscious, reviewed act.

Human judgment is spent only on the behaviorally subtle, perishable ~10%;
the long tail is auto-covered, un-missable, and defensibly defaulted.

## Section 4 -- Failure modes and the operational meaning of "clean release"

Three places fail, each loud and located where the knowledge is:

- Build panic (archspec `enforce.rs`): a unit with no verdict; two overrides
  claiming one node (partition violation); an override claiming a node the
  toolchain no longer emits (stale claim).
- Compile error: a new `SemanticOp` variant breaks two exhaustive matches
  until resolved -- the interpreter's `impl SurfaceProbe`
  (`surface_probe.rs`, must classify the handler) and archspec's `category()`
  in `derive.rs` (must categorize). Each fires in its own crate. The
  toolchain cannot add an instruction we silently ignore.
- Test red (reconciliation test): a `Verified` unit whose nodes are all
  `Absent`; a stale committed `architecture-index.md`; an undeclared shadow.

The gate is a single command: `cargo test coverage::clean_release`. It
asserts that **both** the perishable queue and the comprehension-gap set are
empty, modulo `Accepted { rationale }`. Green is the literal, checkable
definition of "we can let NPU1 go": every behavioral unit is
toolchain-ground-truth, or `Verified { evidence }`, or
`Accepted { rationale }`; and nothing a trusted source describes is sitting
in "we do not know what this is." Both `perishable-queue.md` and
`comprehension-gaps.md` are committed and diffable, so every closure is one
line removed in git history -- progress is visible and auditable. A
comprehension gap is closed by understanding it (verdict improves) or by
explicit `Accepted` -- never by it quietly aging into the background.

## Section 5 -- Bootstrap without a flag-day

Trap: switching on the build panic with ~100 blank override verdicts bricks
the build until all are filled. Avoided with honestly-pessimistic category
defaults plus monotone refinement.

The derived layer assigns each category its *weakest plausible* provenance,
not a uniform `ToolchainDerived`:

- `scalar-arithmetic / bitwise / comparison / control-flow / memory` ->
  `ToolchainDerived / NotApplicable`
- `vector` -> `AietoolsModeled / Unverified`
- `timing` -> `DocSpecified / Unverified` (coarse on purpose; timing is
  highly heterogeneous and is an explicitly-expected multi-override domain --
  see below)
- `NeedsTriage` -> `Unspecified` (lands in comprehension-gaps by
  construction)

That is roughly 15 default verdicts, not 100. Phase 1 ships green
immediately, and both honesty-failure files are honest from day one -- just
coarse ("all of vector compute: unverified" as one fat line rather than
nothing at all).

Timing in particular does not stay one bucket. It is expected to fan out
into multiple override units in Phase 2, because the hardware itself is not
one clock: the NoC runs in a different clock domain from the compute array
(a non-obvious, perishable hardware fact -- recorded here so it is not lost).
Expected named override units include at least `timing.array_clock`,
`timing.noc_clock_domain` (distinct domain; NoC-to-array crossing latency is
its own behavior), `timing.dma_pipeline_depth`, `timing.lock_latency`, and
`timing.stream_switch_latency` -- each carrying its own honest verdict rather
than inheriting the coarse `timing` default. The coarse default exists only
so Phase 1 is green and honest; it is a placeholder these units retire, not a
claim that timing is monolithic.

Invariant that makes incremental rollout safe: a default is always the
weakest member's provenance, and refinement can only *improve* a verdict,
never hide one. The two sets are therefore always a safe over-approximation
of "what still needs attention" -- they shrink as real work is done and can
never silently under-report.

`architecture-index.md` flips from hand-maintained to generated in the same
PR as Phase 1: delete the old file, the renderer replaces it. No
dual-maintenance window.

Phases:

1. Land infrastructure (types, derive, enforce, surface probe, reconciliation
   test) with ~15 honest coarse category/group defaults. Build green;
   artifacts honest-but-lumpy; index becomes generated.
2. Refine coarse perishable buckets into precise override units as
   verification work proceeds; flip units to `Verified { evidence }` or
   `Accepted { rationale }` one at a time.

## Section 6 -- Capability spine and the completeness tiers

The fine-node enumeration is bottom-up (sources -> nodes). It is un-missable
only relative to the union of its sources, not relative to physical silicon.
A top-down capability spine guards against whole-component blind spots.

Two independent structural enumerations already available, neither
register-level:

- AM020/AM025's own table of contents -- the manual's chapter/section
  structure is itself a hardware-capability enumeration (core, program/data
  memory, DMA, locks, stream switch, events/trace, performance counters,
  error & interrupt, debug/halt, cascade, accumulators, clock/reset).
  Extractable from the committed `docs/xdna/` text headings. ~30-50 domains.
- aie-rt's module tree (`dma/ locks/ stream_switch/ events/ perf/ ...`) -- a
  second, independently authored capability list.

Mechanism: a small hand-curated `CapabilityDomain` list (~40 entries) where
each domain must be claimed by at least one behavioral unit, or the build
panics. The list lives in exactly one location -- `coverage/units.rs`,
co-located with the override registry -- seeded *once* by reading the AM020
ToC and the aie-rt module tree, then maintained only there. It is
deliberately *not* derived from `docs/xdna/` headings or any other source at
build time: a single hand-curated location is the point, the same way the
override registry is one location. Splitting the spine's source of truth
across a doc-scraper would reintroduce exactly the decoupling this whole
design exists to kill. Not "every register has a verdict" but "every hardware
capability the manual names is modeled by something." This catches "we
completely forgot debug-halt is a thing." Cross-referencing two
independently-authored enumerations *at seeding time* is the `Confirmed<T>`
multi-source-agreement trick applied to components instead of constants.
Hand-curated, but ~40 architectural domains that change only when the
architecture does -- a different swamp from 6,412 field verdicts.

Completeness tiers, decreasing strength, each labeled honestly:

| Tier | Strength | Mechanism | Guards against |
|------|----------|-----------|----------------|
| Source union | Strong, enforced | Build panic: every generated node -> unit -> verdict | Losing track of anything a source surfaces |
| Capability spine | Medium, enforced | Build panic: every named capability domain claimed by >=1 unit | Forgetting a whole component the manual names |
| Comprehension | Strong, gated | `clean_release` requires comprehension-gaps empty | A trusted source describing something we do not understand |
| Empirical residual | Weak, process-only | `HardwareObserved` provenance can mint a unit with no originating node | Truly undocumented silicon -- unenforceable; only trace sweep / future fuzzer can find it |

The empirical residual is named, not solved. The design's contribution there
is modest but real: a `Provenance::HardwareObserved` that can mint a
behavioral unit with no originating source node -- a unit born from a
hardware surprise rather than derived top-down -- so the model can *receive*
empirical discoveries instead of structurally denying they can exist. It does
not find them. That is itself an argument for keeping NPU1 until the
empirical channels have had their run, regardless of the static gate.

## Future work

- Wire trace-sweep / differential-fuzzer findings into
  `Provenance::HardwareObserved` unit minting (the empirical intake path).

## Open questions

None blocking. Bootstrap default verdicts (Section 5) will be reviewed
against the real category set during implementation; the invariant
(pessimistic default, monotone refinement) makes any initial mis-estimate
safe -- it can only over-report, never hide.
