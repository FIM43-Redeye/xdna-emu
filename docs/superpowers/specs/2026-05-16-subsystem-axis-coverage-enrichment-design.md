# Subsystem-Axis Coverage Enrichment

- Date: 2026-05-16
- Status: Approved design, pre-implementation (revised after quality review)
- Scope: Enrich the capability-spine (subsystem) axis in `xdna-archspec`
  into a first-class *generated* coverage view, woven bidirectionally to
  the SemanticOp-category axis, using an extended two-axis verdict
  vocabulary. Recovers the retired hand-maintained
  `docs/coverage/architecture-index.md`'s **per-subsystem coverage rows**
  as generated data; its point-in-time audit *meta-sections* are
  explicitly retired (see Section 5 and the Appendix).
- Supersedes: the "Plan 3 = HardwareObserved empirical-intake path"
  framing recorded after Plan 2. See "Regroup reframe" below.

## Regroup reframe

The two-axis coverage Plan 2 shipped and pushed. The regroup before the
next plan produced two outcomes that this spec acts on:

1. **The empirical-intake premise was wrong.** `HardwareObserved` is a
   *hand-loaded* verdict by design -- it encodes facts that *cannot* be
   derived programmatically (a human observes real silicon and records
   it). There is no trace-sweep / differential-fuzzer producer to wire;
   the mint API (`Verdict::hardware_observed(...)`) already shipped in
   Plan 2. Any residual work there is small and is *not* this plan.
2. **The real gap is navigability of the subsystem axis.** Retiring the
   hand-maintained `architecture-index.md` for the generated one lost
   the old index's per-subsystem Source / Coverage / `src/`-location /
   known-gaps content. That content has no structured home: the
   generated artifacts cover only the 9 SemanticOp categories, and
   `CapabilityDomain` is a bare `{ id, arches }` presence checklist with
   no verdict, no source pointer, no narrative. The subsystem axis is
   rendered nowhere. This plan closes that.

## Problem

The coverage model has two axes that do not reference each other:

- **SemanticOp category** (9 buckets): carries verdicts, *is* rendered
  (`docs/coverage/aie2/architecture-index.md`).
- **Spine domain / subsystem** (16 ids in `spine_ids.rs`): a bare
  presence list. No verdict, no source ref, no `src/` location, no
  narrative. The build gate only checks each domain is claimed by >= 1
  unit; Phase 1 auto-claims all via the derived shim.

The retired hand-maintained index's value lived almost entirely on the
subsystem axis. That content now exists only in git history. "Weave the
index to per-subsystem detail" is therefore not a linking task -- there
is no generated subsystem detail to link *to*. The subsystem axis must
first become structured, verdict-bearing, and generated, like the
category axis already is.

Two structural facts discovered during quality review shape the design:

- **The category axis cannot report implementation state.** Every
  category default verdict (verified against `derive.rs::default_verdict`)
  is either `ToolchainDerived/NotApplicable` (Arithmetic, Bitwise,
  Comparison, Memory, ControlFlow, Sync), `AietoolsModeled/Unverified`
  (Vector), `DocSpecified/Unverified` (SideEffect), or
  `Unspecified/Unverified` (NeedsTriage). **No category ever yields a
  "we built / half-built / didn't build it" signal** -- that is
  knowledge-provenance, not emulator build state. Implementation
  completeness can therefore *only* come from a domain's own asserted
  verdict, never from rolling up its tagged categories. The earlier
  draft conflated these; this revision separates them.
- **The 16-id spine cannot home the retired index's content.** Several
  index rows -- control packets (the keystone subsystem the index exists
  *because* we missed), clock control, tile isolation, and the binary
  ingestion path (CDO/ELF/XCLBIN/NPU-instruction-stream) -- have no
  spine domain. Per the regroup decision the spine is extended (Section
  2); the index's non-subsystem *meta-sections* are explicitly retired
  (Section 5).

A second, narrower problem: `Verification` says "how do we know the
model is right" but cannot say "the toolchain fully specifies this, we
just haven't built it" or "we built half of it." The retired index's
`MISSING` / `PARTIAL` / `STUBBED` rows are exactly this state.

## Goals

- Make the subsystem axis a first-class *generated*, staleness-gated
  coverage view -- the retired index's **per-subsystem rows**
  regenerated, never hand-edited.
- Weave the two axes bidirectionally: every category points at its
  relevant subsystem(s); every *category-tagged* subsystem shows its
  curated context next to the rolled-up verdict of its tagged
  categories, with an optimistic-over-claim drift cross-check.
- Give the implementation-completeness state (`PARTIAL` / `STUBBED` /
  absent) a typed home and a self-regenerating queue, alongside the
  perishable and comprehension queues.
- Keep one verdict vocabulary. Extend `Verification` rather than add a
  parallel coverage taxonomy.

## Non-goals

- Per-unit (override-registry) domain tagging. The Phase-1 override
  registry is empty by design; domain tags live on the 9 coarse
  category-default buckets now, per-unit refinement is deferred to
  Phase 2. Consistent with all prior phasing.
- Pulling Phase-2 refinement forward (populating `override_registry`).
- The `HardwareObserved` empirical-intake path (dissolved -- see Regroup
  reframe).
- **Recovering the retired index's meta-sections in generated form.**
  The "Authoritative sources" table, "Gaps summary by triage", the
  AIE-ML constants cross-check, and "Pass 2 deep-dive priorities" were
  point-in-time audit scaffolding, not durable per-subsystem facts.
  Their disposition is specified in the Appendix; they are not carried
  into the generated artifact.
- Performance work. This is a correctness/completeness instrument.

## Section 1 -- Vocabulary: extend `Verification`

```rust
pub enum Completeness {
    Full,                          // built and exercised by tests
    Partial { missing: String },   // layout/some behavior present; named sub-behavior absent
    Stub,                          // placeholder defaults, no real state machine
}

pub enum Verification {
    NotApplicable,
    Verified { evidence: String },
    Unverified,
    Accepted { rationale: String },
    Modeled { completeness: Completeness },   // NEW
}
```

**Two safety mechanisms, named precisely (quality-review C1).** The
no-`_`-arm discipline does **not** hold uniformly across the coverage
crate: `artifacts.rs::render_perishable` carries a documented dead `_`
arm (on `Provenance`, a Phase-1 dead arm -- `artifacts.rs:34`). The
guarantees are therefore split and an implementer must treat them as
two different mechanisms:

1. **Predicate behavior is provably unchanged by `matches!`-arm
   inspection.** `is_perishable` / `is_comprehension_gap` keep their
   exact current arms; neither matches `Modeled`, so their behavior is
   untouched by construction. This is a local proof, not a
   compiler-exhaustiveness argument.
2. **Enforcement / derivation paths are exhaustiveness-protected.**
   `derive.rs` and `enforce.rs` carry no `_` arms on `Verification`;
   adding a variant makes the compiler walk every site.
3. **Renderer paths are *staleness-gated*, not exhaustiveness-protected.**
   The artifact renderers may match `Verification`; the safety net there
   is the regenerate-equals-committed staleness test, not the compiler.

Explicit task constraint (carried into the plan): every new or modified
match on `Verification` **and on `Completeness`** in the renderers must
be written **exhaustive with no `_` arm** (a `_` on the inner
`Completeness` would silently collapse `Partial`'s `missing` string --
quality-review N4), so renderer protection is upgraded to match (1)/(2)
rather than relying on the staleness gate alone. The pre-existing
`render_perishable` `Provenance` `_` arm is left as-is (documented dead
arm, not in scope).

Third gap class:

```rust
pub fn is_implementation_gap(&self) -> bool {
    matches!(
        self.verification,
        Verification::Modeled {
            completeness: Completeness::Partial { .. } | Completeness::Stub
        }
    )
}
```

**`is_implementation_gap` is evaluated over the 20 domains' own seeded
verdicts, never over the SemanticOp universe (quality-review I1/M1).**
No category default is ever `Modeled`, so evaluating this predicate
over `all_semantic_verdicts()` is permanently `false` -- a silent
empty-queue failure of exactly the kind this whole design exists to
kill. The implementation-gaps artifact iterates `capability_spine()`,
reading each domain's `verdict`. `MISSING` is the *absence* of concrete
coverage, represented as the domain's own seeded verdict being
`Modeled{Stub}` (placeholder, no state machine) or, for genuinely
unbuilt subsystems, `Modeled{Partial{missing: "..."}}` /
`Unspecified/Unverified` with a narrative -- never inferred from a
category rollup.

New documented `Verdict` invariant, mirroring the existing "`Unspecified`
never pairs with `Verified`": **`Unspecified` never pairs with
`Modeled`.** Asserting *no model* and asserting *a model exists* are
contradictory. `enforce_coverage` rejects the pairing by inspecting the
**seeded domain `verdict`** (not the synthetic derived units, which are
always `NeedsTriage` default and can never be `Modeled` -- see Section 2
N2 note). This is **test-gated, not a `build.rs` panic** (Section 2) --
stated as plainly as the parent spec's Section-4 phasing note: typedness
prevents phrasing bypass, but the enforcement runs on the
`enforce_coverage_phase1` path (once that path carries seeded domains,
Section 2), not the dependency-light `build_gate.rs` string path.

`Modeled { Full }` is "it exists and is as good as its provenance" --
neither an implementation gap nor (via the unchanged predicates) a
perishable / comprehension entry. The lifecycle/verification merge is a
deliberate, accepted conceptual load on a core type (see Risks).

## Section 2 -- Model: enrich `CapabilityDomain`, extend the spine

```rust
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
    pub source_ref: String,         // authoritative source (aie-rt path / AM025 / device model)
    pub src_locations: Vec<String>, // our emulator src/ path(s); empty allowed only for OOS/MISSING with narrative
    pub narrative: String,          // the old "Notes / known gaps" column
    pub verdict: Verdict,           // the domain's own asserted coverage verdict
    pub drift_rationale: Option<String>, // documents a deliberate own-vs-rollup divergence (Section 3)
}
```

**Spine extension (hybrid decision).** `spine_ids.rs` grows from 16 to
20 ids, adding: `control_packets`, `clock_control`, `tile_isolation`,
`binary_load`. The dependency-light leaf property is preserved -- these
are four `&'static str` literals with zero `crate::` imports, so
`build.rs` can still `#[path]`-include the leaf. `build_gate.rs`'s
`enforce_spine_phase1` logic is unchanged (it iterates the list checking
non-empty/non-blank; the list is merely longer). `units.rs::
capability_spine`'s documented-folds doc comment is extended to define
what each new domain covers (Appendix). The
`capability_spine_matches_the_leaf_id_list` equality test continues to
hold (both sides grow together).

The category->domain link is the **single source of truth, stored only
on the category side**: a coarse hand-curated
`category_domains(Category) -> &'static [&'static str]`, 9 rows. The
domain->category direction is the *inverse index computed at generation
time* -- never stored, so there is no second copy to drift.
**Category-orphan domains are explicit and expected**
(quality-review M2): domains with no natural SemanticOp (e.g. `timer`,
`watchpoint`, `performance_counters`, `interrupt`, `noc`, `shim_mux`,
`clock_control`, `tile_isolation`, `binary_load`) are *not* force-tagged
onto an unrelated category. Their coverage comes solely from their own
seeded verdict; they get no rollup and no drift check (there is nothing
to cross-check against). Fabricating a tag to satisfy a reachability
rule is forbidden -- it produces meaningless rollups.

**Unstated plumbing, named as scope (quality-review N2).** The seeded
fields are *not* reachable on the current enforcement path: today
`units.rs::capability_spine()` builds bare `{ id, arches }` structs, and
`enforce_coverage_phase1` synthesizes `derived.<id>` units hardcoded to
`default_verdict(NeedsTriage)` and reads the `spine` argument only for
`.id` / `.applies_to`. Delivering this section therefore requires two
real code changes, which the seed task (Section 5) owns explicitly --
not a one-line check on existing plumbing:

1. Rewrite `capability_spine()` to construct the 20 **fully-seeded**
   `CapabilityDomain`s (real `source_ref` / `src_locations` /
   `narrative` / `verdict` / `drift_rationale`), not bare id structs.
   `capability_spine_matches_the_leaf_id_list` still holds (it compares
   `.id` only).
2. Add a new per-domain validation loop to `enforce_coverage` keyed on
   the (now-seeded) `spine` argument: non-empty `source_ref`; non-empty
   `src_locations` *unless* the verdict is an explicit OOS/MISSING state
   documented in `narrative`; and the Section-1 `Unspecified + Modeled`
   rejection on `domain.verdict`.

`enforce_coverage_phase1` already calls `capability_spine()`, so once
(1) lands the `spine` argument carries the seeds and the new loop in (2)
exercises them -- the "test-gated via `enforce_coverage_phase1`" framing
is then accurate. This is the old index's "where is the source / where
do we implement it" discipline as a typed invariant. It remains
**test-gated**, not a build.rs panic; `build.rs` and the
`build_gate.rs` string path do not move.

## Section 3 -- Rollup and drift cross-check

The rollup's **only** role is the drift cross-check (it cannot detect
implementation state -- Section 1). For each *category-tagged* domain,
generation computes the rolled-up verdict of its tagged categories via
the inverse index, **worst-wins** (a domain is only as covered as its
weakest tagged category).

Lattice principle: **closed/terminal states rank above open states.**
`Verified == NotApplicable == Accepted` (all "nothing owed": verified,
toolchain-ground-truth, or explicitly signed off) > `Modeled{Full}`
(built, complete, verification still open) > `Modeled{Partial}` >
`Modeled{Stub}` > `Unverified`; provenance tie-breaks toward the weaker
(`Unspecified` weakest). `Accepted` is ranked *with* the closed states
deliberately: in the parent model it closes both honesty queues
(`verdict.rs:64-67`), so ranking it below an open `Modeled{Full}` would
emit confusing drift. The exact total order is fixed in implementation
and **locked by a dedicated test** (Section 7); this list is the
binding shape.

The two sides measure different facets -- the domain's own verdict
encodes implementation completeness (`Modeled{}`); the category rollup
encodes knowledge-provenance and can never be `Modeled{}`. The lattice
is a deliberate unified coverage-strength projection used *only* for
the drift comparison; this asymmetry is intended, not a bug.

**Drift is directional (quality-review I2).** A drift is *material* --
flagged in the generated view and failing `no_subsystem_drifts_silently`
-- **only when the domain's own verdict ranks strictly higher
(more-covered) than the rolled-up category verdict**, i.e. the
subsystem claims better coverage than its constituent categories
justify (optimistic over-claim, the dangerous direction, matching the
parent `shadows_derived` rationale). The reverse (a domain more
pessimistic than its categories) is safe over-reporting and does **not**
trip. A material drift is permitted only if `drift_rationale` is
`Some(...)` -- a deliberate, documented divergence; a silent one is a
hard failure. Category-orphan domains have no rollup and are exempt.

## Section 4 -- Generated artifacts

- **`docs/coverage/aie2/subsystem-index.md`** (new, generated,
  staleness-gated `subsystem_index_is_not_stale`): one row per spine
  domain -- id | source_ref | src_locations | own verdict | rolled-up
  category verdict (or `-` for category-orphan) | drift flag |
  narrative. This *is* the retired index's per-subsystem rows,
  regenerated.
- **`docs/coverage/aie2/implementation-gaps.md`** (new, generated,
  staleness-gated): the domains whose own seeded verdict satisfies
  `is_implementation_gap` (`Modeled{Partial|Stub}`) plus explicit
  MISSING/OOS domains -- the old MISSING / PARTIAL / STUBBED rows, now a
  self-regenerating queue alongside `perishable-queue.md` and
  `comprehension-gaps.md`. Sourced from domain verdicts, never a
  category rollup.
- **`architecture-index.md`** (existing): each category row gains its
  tagged domain id(s); a header line links to `subsystem-index.md`.
  Note (quality-review M3): this is a schema change to a fixed-preamble
  whole-file renderer -- `render_architecture_index`'s preamble prose
  and the `architecture_index_reps_match_category` test are revisited,
  and the regenerated file is committed in the same task that changes
  it (Section 8 ordering rule).
- `crates/xdna-archspec/examples/gen_coverage_artifacts.rs` writes both
  new files; its `eprintln!` summary is updated.

## Section 5 -- Seed data, meta-section retirement, inbound re-linking

Seed all 20 domains' `source_ref` / `src_locations` / `narrative` /
`verdict` by porting the retired index's content (present in git at
`1afdb20^:docs/coverage/architecture-index.md`). The retired index's
~50 tile-type-grouped rows fold onto the 20 domains per the **fold map
in the Appendix** -- this map is part of *this spec*, not deferred to
plan-writing, because it is a judgment fork with no toolchain authority
(quality-review I3). The five `SubsystemKind`->spine folds documented in
`units.rs::capability_spine` are *not* that map (they are a different
relation); the Appendix is authoritative for index-row->domain.

**Meta-section disposition** (the index content the row schema cannot
hold):

- *Authoritative sources* table -> absorbed into per-domain `source_ref`
  values; not a standalone section.
- *AIE-ML constants cross-check* (locks/BD/channels per tile type) ->
  durable verified facts; folded into the relevant domains' `narrative`
  (`locks`: 16/64/16; `dma`: BD 16/48/16, ch 2/6/2). Not a section.
- *Gaps summary by triage* -> resolved items (the `~~FIXED~~` bulk) stay
  in git history only; still-live verification items (#13-17) fold into
  the owning domain's `narrative` as known-gaps; cycle-accuracy items
  already belong to `cycle-accuracy-mission.md` and are not duplicated.
- *Pass 2 deep-dive priorities* -> historical audit state, all closed or
  tracked elsewhere; git history only.

Inbound docs that referenced the old index for *subsystem* detail
(`audit-checklist.md`, `cycle-accuracy-mission.md`, `docs/README.md`,
and any others task scope surfaces) repoint to `subsystem-index.md`;
references genuinely about the *category* matrix stay on
`architecture-index.md`. `audit-checklist.md`'s workflow prose is
extended so verdict flips also feed the implementation-gaps queue and
the subsystem axis, not only perishable / comprehension.

## Section 6 -- Error handling and edge cases

- Adding `Verification::Modeled` makes the compiler walk every
  exhaustive `Verification` match in `derive.rs`/`enforce.rs` (intended).
  Renderer matches are constrained exhaustive-no-`_` by the Section-1
  task constraint; the pre-existing `Provenance` `_` arm in
  `render_perishable` is out of scope and unaffected (it is not a
  `Verification` match).
- A category-orphan domain has no rollup; its row shows `-` for the
  rolled-up column and is exempt from the drift check.
- A category-tagged domain whose own verdict is *more pessimistic* than
  its rollup is safe over-reporting and must not trip the drift test.
- `Unspecified + Modeled` is rejected by `enforce_coverage` (Section 1
  invariant), test-gated.
- `is_implementation_gap` evaluated over the SemanticOp universe is
  permanently false -- the implementation-gaps generator must iterate
  domains. A test asserts the generator's source is the spine, not the
  semantic model, so this silent-empty failure cannot regress.

## Section 7 -- Testing

- Predicate tests: `Modeled` is neither perishable nor a comprehension
  gap; `Modeled{Partial|Stub}` is an implementation gap; `Modeled{Full}`
  is none.
- Invariant test: `Unspecified + Modeled` rejected by `enforce_coverage`
  (test-gated path).
- **Lattice-order test**: pins the full worst-wins total order including
  the `Verified == NotApplicable == Accepted` tie and the `Accepted >
  Modeled{Full}` placement (forces the Section-3 ordering to be a
  conscious, locked decision).
- Staleness gates for `subsystem-index.md` and
  `implementation-gaps.md` (regenerate == committed).
- Drift cross-check: a fixture with a pessimistic divergence (must
  **pass** -- safe over-report), and one with an optimistic over-claim
  (must **fail** unless `drift_rationale` is set).
- Implementation-gaps source test: the generator iterates
  `capability_spine()` domain verdicts, not `all_semantic_verdicts()`
  (guards the M1 silent-empty mode).
- Coverage-link totality: every `Category` tags >= 1 domain; every
  domain is **either** category-tagged **or** explicitly category-orphan
  (no fabricated tags). Same *spirit* as
  `capability_spine_matches_the_leaf_id_list` (a structural totality
  tripwire) but a **distinct mechanism** -- a bipartite cover/partition
  check, not a list-equality `assert_eq!` (quality-review N3).
- `enforce_coverage` extended: every domain has non-empty `source_ref`,
  and non-empty `src_locations` unless an OOS/MISSING narrative is
  present.

## Section 8 -- Scope and phasing

One cohesive plan, **approximately 6-7 tasks** (the spine extension +
struct enrichment is its own task; the seed-fold is plausibly two:
hardware-subsystem domains, then binary/control-plane domains),
executed subagent-driven like Plan 2 (implementer + spec-review +
code-quality-review per task, controller adjudication, plan/spec kept in
lockstep, commit per task, final holistic review). Seed-coarse-now /
per-unit-refine-in-Phase-2.

**Cross-task ordering rule (quality-review M4):** any task that changes
a verdict, a seed value, a category->domain tag, or the spine list must
regenerate *and commit* all affected artifacts (`subsystem-index.md`,
`implementation-gaps.md`, `architecture-index.md`, and the existing
queues) in that same task, or the next task's staleness test fails on
unrelated grounds. `cargo test --lib` is the ground-truth gate; bare
`cargo build` resolves stale-harness-diagnostic ambiguity.

## Risks

- **Conceptual load on a core type.** Folding completeness into
  `Verification` means one enum answers two questions. Accepted
  deliberately: nothing here is load-bearing in production, and
  preserving a suboptimal vocabulary because it has a git SHA would be
  the wrong conservatism. Approach A contains the load (one variant,
  structured sub-enum, predicates provably unchanged by arm inspection).
- **Seed-mapping judgment.** Folding ~50 heterogeneous rows onto 20
  domains is curation, not lookup. Mitigation: the fold map is in the
  Appendix of this spec (a design decision, reviewed), not improvised at
  plan time; per-row narrative *text* is implementation-time authoring
  within that fixed mapping.
- **Spine-extension blast radius.** The four new ids touch the
  dependency-light `spine_ids.rs` leaf. Mitigation: they are bare
  `&'static str` literals (zero imports -- leaf property preserved);
  `build_gate.rs` logic is unchanged; the leaf<->rich equality test and
  `capability_spine` doc-comment are updated in the same task and
  covered by existing tests.

## Appendix -- Retired-index-row -> spine-domain fold map

Authoritative for Section 5. Domains added by the hybrid extension are
marked **(new)**. Per-tile-type duplicate rows (e.g. DMA on
compute/memtile/shim) collapse into one domain; their differing notes
merge into one `narrative` with the AIE-ML constants folded in.

| Retired-index concept | Spine domain |
|---|---|
| VLIW core; Core control (enable/done/reset); Core error halt | `core` |
| Core debug (halt/step/breakpoint) | `debug_halt` |
| Program memory | `program_memory` |
| Data memory (64KB/512KB); bank-conflict events; ECC (OOS narrative) | `data_memory` |
| Watchpoint hardware (compute 2 / memtile 4) | `watchpoint` |
| DMA engine (compute/memtile/shim); Shim DMA | `dma` |
| Locks (16/64/16) | `locks` |
| Stream switch (all tiles) | `stream_switch` |
| Shim Mux/Demux | `shim_mux` |
| Cascade ports | `cascade` |
| Events (all tiles); cross-tile event broadcast; Trace unit | `events_trace` |
| Performance counters | `performance_counters` |
| Timer (per-module); multi-tile timer sync | `timer` |
| L1 / L2 interrupt controller | `interrupt` |
| Direct NoC control; AIE_AXIMM_Config; NoC fabric; NPI (OOS narrative) | `noc` |
| Tile_Control isolation bits; tile isolation gates (N/S/E/W) | `tile_isolation` **(new)** |
| Module/Column/Tile clock control; Reset_Control; Tile column reset | `clock_control` **(new)** |
| Control packet handling; Packet handler status; NPU instruction stream | `control_packets` **(new)** |
| CDO loading; ELF loading; XCLBIN parsing; stream-switch routing reconstruction; tile array topology | `binary_load` **(new)** (see N1 resolutions) |
| PL Interface (Upsizer/Downsizer) | `shim_mux` (OOS narrative) |
| Driver-side surfaces (PSP, SMU, mailbox, partition/ctx, async error, telemetry, TDR, preemption, debug BO) | **Not a spine domain.** Retired with the meta-sections: driver orchestration, not silicon. Git history only; still-live items (async error reporting, debug BO) are named future work, not seeded. |

**N1 resolutions (documented judgment calls, not silent
nearest-neighbor):**

- *Stream-switch routing reconstruction* is the parse-time
  reconstruction of stream-switch state from CDO
  (`src/parser/stream_switch_topology.rs`), distinct from the runtime
  `stream_switch` subsystem. Its known-gaps narrative lives in the
  `binary_load` domain (it is a binary-ingestion concern); the
  `stream_switch` narrative carries a one-line cross-reference to it.
  This split (parse-side vs runtime) is deliberate and stated so an
  implementer does not have to guess where the text belongs.
- *Tile array topology* (`src/device/array/`, the 5x6 grid from the
  device model) has no dedicated domain under the hybrid scope (a
  `device_topology` domain was considered and rejected -- it is
  config-derived, not a coverage gap). It folds into `binary_load` as
  "the array constructed from the loaded binary/device-model", with
  *this* sentence as the recorded rationale -- a documented fold, not a
  reachability-forced tag.

Ambiguity-resolution authority for any row not listed: nearest
hardware-subsystem domain by function, **with the rationale written
into this Appendix** (the anti-fabrication rule of Section 2 applies to
row->domain folds too); if none, retired with the meta-sections and
noted here before implementation.
