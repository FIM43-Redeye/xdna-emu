# NPU1 Phase 2A Research-Reserve Ledger Kernel Implementation Plan

**Goal:** Establish the first versioned NPU1 research ledger, seed one honest
tuple-to-evidence chain, and generate a structured retirement report that
cannot become green through omitted inventory or hand-authored evidence state.

**Architecture:** `xdna-archspec` owns one Serde-defined JSON ledger, structural
validation, fail-closed release evaluation, and deterministic Markdown
rendering. The existing semantic coverage gate becomes one explicitly named
input. External evidence is never trusted in this slice; the following bundle
slice will produce the first production integrity audit.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents, Halo, NPU workload, KVM, vfio-user,
QEMU, corpus mutation, or privileged operation. Every behavioral task records
a focused RED before the smallest GREEN and commits only after its focused
regressions pass.

**Approved design:**
[`2026-07-29-npu1-phase2a-research-reserve-ledger-kernel-design.md`](../specs/2026-07-29-npu1-phase2a-research-reserve-ledger-kernel-design.md)

## Baseline

The approved design is commit:

```text
4faa7026 docs: design NPU1 research-reserve ledger kernel
```

Implementation begins from a clean `investigate/firmware-priors` HEAD that
contains this plan and has `4faa7026` as an ancestor.

At planning time:

- `cargo test --lib --quiet` passes 4,275 tests and ignores 32;
- `xdna-archspec` already owns `CoverageModel`, `SPINE_DOMAIN_IDS`, Serde, and
  the deterministic coverage-artifact generator;
- `CoverageModel::clean_release()` is still semantic-only and red through the
  perishable queue;
- no production evidence-integrity audit exists;
- the Phase 1A Chess witness remains a non-canonical emulator regression
  witness; and
- the same-filesystem Btrfs snapshot has no independent-replica credit.

Live files and test output are authoritative during execution. Counts in this
plan are a checkpoint, not a promised final number.

## Task 0: Preflight and Scope Lock

**Read only:** repository state and existing coverage owner.

- [ ] Require a clean worktree and the approved design in history:

```bash
git status --short
git merge-base --is-ancestor 4faa7026 HEAD
```

- [ ] Re-run the focused and ordinary baselines:

```bash
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
```

- [ ] Re-pin the implementation surface:

```bash
rg -n 'clean_release|SPINE_DOMAIN_IDS|gen_coverage_artifacts' \
    crates/xdna-archspec src docs/coverage
```

- [ ] Confirm no planned implementation or generated file already exists:

```bash
test ! -e crates/xdna-archspec/src/research_reserve.rs
test ! -e crates/xdna-archspec/data/research-reserve/npu1.json
test ! -e docs/coverage/npu1/release-report.md
```

- [ ] Stop if the live tree contradicts the design, contains overlapping
  changes, or the baseline is red.

## Task 1: Rename the Narrow Semantic Gate

**Files:**

- Modify: `crates/xdna-archspec/src/coverage/mod.rs`
- Modify: `crates/xdna-archspec/src/coverage/units.rs`
- Modify: `crates/xdna-archspec/src/coverage/artifacts.rs`
- Regenerate: `docs/coverage/aie2/architecture-index.md`

### RED

- [ ] Rename the three gate-focused test names and their method calls to
  `semantic_provenance_clean()` before changing the implementation.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib semantic_provenance_clean
```

- [ ] Require compilation to fail because `CoverageModel` does not yet expose
  the honestly named method. Record the diagnostic in the execution notes.

### GREEN

- [ ] Rename `CoverageModel::clean_release()` to
  `semantic_provenance_clean()` with no compatibility alias and no behavioral
  change.
- [ ] Update current code comments and generated prose that still assign
  retirement meaning to the narrow gate. Historical specs and findings remain
  historical; do not rewrite them.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib semantic_provenance_clean
cargo run -p xdna-archspec --example gen_coverage_artifacts
cargo test -p xdna-archspec --lib
```

- [ ] Confirm only the expected source files and generated AIE2 index changed,
  and that no current Rust caller retains `.clean_release()`.
- [ ] Run `git diff --check`.
- [ ] Commit:

```text
refactor(coverage): clarify semantic provenance gate
```

## Task 2: Define the Versioned Ledger Schema

**Files:**

- Modify: `crates/xdna-archspec/src/lib.rs`
- Add: `crates/xdna-archspec/src/research_reserve.rs`

The first module remains dependency-free beyond existing `serde` and
`serde_json`.

### Schema

- [ ] Define `SCHEMA_VERSION: u32 = 1` and the root:

```rust
pub struct ReserveLedger {
    pub schema_version: u32,
    pub tuples: Vec<PinnedTuple>,
    pub inventory: Vec<InventoryEntry>,
    pub facts: Vec<HardwareFact>,
    pub evidence: Vec<EvidenceRecord>,
}
```

- [ ] Derive serialization for every persisted type, reject unknown fields,
  and use snake-case tags for new enums.
- [ ] Reuse `types::Architecture`; give the ledger field a small Serde adapter
  so persisted values are `aie`, `aie2`, and `aie2p` without introducing a
  second architecture enum or changing the existing crate-wide serialization
  contract.
- [ ] Define the minimum approved tuple records:
  - `DevicePin`;
  - `ContentPin`;
  - `RevisionPin`;
  - tuple identity `Open { missing_fields }` /
    `Complete { evidence_ids }`;
  - inventory scope `Open { remaining_sources }` /
    `Sealed { evidence_ids }`;
  - kernel-corpus, live-attestation, and offline-rehearsal evidence
    references.
- [ ] Define inventory entries and the tagged dispositions:
  `Applicable`, `ProvenNotApplicable`, `Deferred`, and `Unknown`.
- [ ] Define the complete fact contract and promotion states:
  `Observed`, `Derived`, `Verified`, `Encoded`, `RetirementQualified`, and
  `Contested`.
- [ ] Define evidence kind, stable aliased location, expected digests,
  provenance gaps, retention, redistributability, expected replicas, and
  preservation notes.
- [ ] Keep stable IDs as strings. Do not add ID wrapper types, builders, or a
  record-per-file loader.

### RED

- [ ] Add inline JSON tests for:
  - unsupported `schema_version`;
  - an unknown root field;
  - snake-case AIE2 round-trip; and
  - a minimal structurally representable open ledger.
- [ ] Write tests before the parser/validator entry point and run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::parse_
```

- [ ] Require the focused tests to fail because version and schema validation
  are absent.

### GREEN

- [ ] Implement `ReserveLedger::from_json(&str)` as parse followed by
  structural validation.
- [ ] Use a small local error type carrying a stable field path and message.
  Sort multiple validation issues before returning them so diagnostics do not
  depend on hash iteration.
- [ ] Pass the focused parsing tests.

## Task 3: Enforce Structural Integrity

**File:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`

### RED

- [ ] Add table-driven tests proving rejection of:
  - blank, malformed, duplicate, and wrong-kind stable IDs;
  - empty fact lists for `Applicable` and `ProvenNotApplicable`;
  - dangling tuple, inventory, fact, and evidence references, including the
    tuple's kernel-corpus evidence IDs;
  - an unknown `coverage_domain_id`;
  - malformed lowercase SHA-256 values;
  - absolute, blank, or parent-escaping external paths; and
  - self-referential and cyclic inventory/fact dependency graphs.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::validation_
```

- [ ] Require failures that identify the missing guards rather than parse
  errors in the test inputs.

### GREEN

- [ ] Validate stable IDs with one std-only helper implementing the approved
  dotted ASCII grammar and kind prefix.
- [ ] Build one set per record kind for duplicate and cross-reference checks.
- [ ] Validate `coverage_domain_ids` directly against
  `coverage::spine_ids::SPINE_DOMAIN_IDS`.
- [ ] Validate lowercase SHA-256 as exactly 64 ASCII hexadecimal characters.
- [ ] Validate stable relative paths with `std::path::Component`; reject root,
  prefix, current-directory, parent-directory, and empty forms.
- [ ] Reuse one depth-first graph-cycle helper for both inventory and fact
  dependencies.
- [ ] Pass all Task 2/3 focused tests and:

```bash
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Confirm the diff contains only the new module and `lib.rs` export.
- [ ] Commit:

```text
feat(reserve): define and validate the NPU1 ledger
```

## Task 4: Evaluate the Retirement Contract

**File:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`

### Report types

- [ ] Add typed, serializable:
  - `ReleaseCheckKind`;
  - `BlockerCode`;
  - `ReleaseBlocker`;
  - `ReleaseCheck`; and
  - `ReleaseReport`.
- [ ] Give each blocker a stable code, optional affected record ID, dependency
  path, and concise detail.
- [ ] Derive `is_clean` exclusively from an empty blocker list.

### Trusted inputs

- [ ] Keep the production evidence audit private and empty in Phase 2A. The
  ledger may describe expected evidence, but it cannot mark that evidence
  verified.
- [ ] Route evaluation through a private `EvaluationInputs` containing:
  - the semantic-provenance result; and
  - verified evidence/replica IDs from an internal evidence audit.
- [ ] `ReserveLedger::clean_release(tuple_id)` constructs production inputs
  from `CoverageModel::semantic_provenance_clean()` and the empty Phase 2A
  evidence audit.
- [ ] Tests in the same module may construct synthetic inputs. There is no
  public constructor or serialized field that bypasses future bundle
  validation.

### RED

- [ ] Add focused tests proving:
  - open tuple identity blocks release;
  - open inventory scope blocks release even with zero inventory entries;
  - `Deferred` and `Unknown` entries block;
  - an applicable entry blocks on a non-retirement-qualified fact;
  - legacy or unaudited evidence blocks `Verified` and later promotion;
  - `Contested` facts block each dependent inventory/fact path;
  - missing implementation/test links block an otherwise promoted fact;
  - fewer than two independently verified witness replicas block;
  - missing live attestation and offline rehearsal block;
  - a false semantic-provenance input blocks; and
  - a synthetic fully closed ledger plus test-only trusted inputs produces an
    empty blocker set and `is_clean == true`.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::release_
```

- [ ] Require the tests to fail before evaluator behavior exists.

### GREEN

- [ ] Implement each fixed check independently, then merge and sort blockers
  by check kind, stable code, record ID, and dependency path.
- [ ] Follow validated dependency edges when reporting a contested or
  unqualified upstream fact; do not mutate ledger state during evaluation.
- [ ] Treat a declared promotion as a claim to audit, never as evidence that
  its prerequisites were met.
- [ ] Pass all release tests and:

```bash
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Commit:

```text
feat(reserve): evaluate NPU1 retirement blockers
```

## Task 5: Seed the Primary Phoenix Chain

**Files:**

- Add: `crates/xdna-archspec/data/research-reserve/npu1.json`
- Modify: `crates/xdna-archspec/src/research_reserve.rs`

### Authoritative records

- [ ] Add the primary tuple:

```text
tuple.npu1.phoenix.fw-1_5_5_391
```

  with AIE2, `1022:1502`, the approved firmware logical name/hash, driver
  surface commit, explicit open identity fields, open inventory sources, and
  empty attestation/rehearsal evidence.
- [ ] Add the first inventory entry:

```text
inventory.npu1.firmware.command-list-execution
```

  as `Applicable`, with existing `binary_load`, `dma`, and `interrupt`
  coverage-domain references.
- [ ] Add the derived fact:

```text
fact.npu1.firmware.command-list-lifecycle-candidate
```

  with the query/create/map/configure/CHAIN_EXEC/MSI-X/destroy contract and
  every unknown retained from the Phase 1A intake.
- [ ] Add the historical evidence:

```text
evidence.npu1.legacy-vfio-user-chess-20260729t171244z
```

  pointing through `repo-experiments` to
  `phoenix-vfio-user/20260729T171244Z-3136359`, with:
  - metadata fingerprint
    `4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299`;
  - checksum-index SHA-256
    `e7aaacefa4c8f3606529dd27980397a656b22099a349db59d1c0df84330811e2`;
  - the checked-in intake-report reference;
  - legacy provenance gaps and restricted payload classification;
  - zero expected/verified independent replicas; and
  - the same-disk snapshot as a non-replica preservation note.

### RED

- [ ] Add tests that expect an embedded NPU1 ledger and assert the exact four
  stable IDs and immutable pins before adding the JSON file.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::embedded_
```

- [ ] Require failure because the embedded data file/loader is absent.

### GREEN

- [ ] Load the authoritative document with `include_str!` so validation has no
  checkout-relative runtime path.
- [ ] Validate the embedded ledger in its test.
- [ ] Evaluate its release report and assert the exact expected blocker-code
  set, including open inventory, legacy integrity, replica, semantic,
  attestation, and rehearsal blockers.
- [ ] Assert that no current blocker or fact calls the legacy witness physical
  NPU evidence.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Commit:

```text
feat(reserve): seed the Phoenix research ledger
```

## Task 6: Generate the NPU1 Release Report

**Files:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`
- Modify: `crates/xdna-archspec/examples/gen_coverage_artifacts.rs`
- Add: `docs/coverage/npu1/release-report.md`

### RED

- [ ] Add tests for:
  - deterministic rendering when authored record order changes;
  - all fixed check sections and blocker dependency paths;
  - the primary tuple pins and explicit open fields;
  - the legacy witness non-promotion warning; and
  - committed report staleness.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::render_
cargo test -p xdna-archspec --lib research_reserve::tests::release_report_is_not_stale
```

- [ ] Require rendering/staleness failures before the renderer and artifact
  exist.

### GREEN

- [ ] Render Markdown only from the validated `ReserveLedger` and its
  `ReleaseReport`.
- [ ] Sort cloned views for output; do not reorder or rewrite authoritative
  JSON as a side effect.
- [ ] Extend the existing artifact generator to create `docs/coverage/npu1/`
  and write `release-report.md`.
- [ ] Generate once:

```bash
cargo run -p xdna-archspec --example gen_coverage_artifacts
```

- [ ] Run the focused renderer/staleness tests and full crate tests.
- [ ] Run the generator a second time and require `git diff` to remain
  unchanged, proving deterministic idempotence.
- [ ] Run `git diff --check`.
- [ ] Commit:

```text
docs(reserve): generate the NPU1 release report
```

## Task 7: Phase 2A Regression and Closure

**File:**

- Add:
  `docs/superpowers/findings/2026-07-29-npu1-phase2a-ledger-kernel-closure.md`

### Verification

- [ ] Run formatting and the exact design gates:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib
cargo run -p xdna-archspec --example gen_coverage_artifacts
git diff --exit-code
cargo test --lib --quiet
git diff --check
```

- [ ] Confirm current-code naming:

```bash
rg -n '\\.clean_release\\(' crates/xdna-archspec/src src
rg -n 'semantic_provenance_clean' crates/xdna-archspec/src docs/coverage
```

  Current full `clean_release` calls must resolve to the new reserve gate.
  Historical specs/findings may retain their original terminology.
- [ ] Confirm the worktree contains no corpus changes, bundle implementation,
  external hash sweep, hardware output, database, or new dependency.
- [ ] Confirm every commit is scoped and every intermediate commit passes its
  stated focused tests.

### Closure report

- [ ] Record:
  - implementation commit IDs;
  - witnessed RED and GREEN commands/results;
  - schema version and all seeded stable IDs;
  - generated report result and exact blocker classes;
  - final test counts;
  - explicit nonclaims;
  - the fact that external evidence still has zero production verification
    paths; and
  - the canonical-bundle validator as the next boundary.
- [ ] Run `git diff --check`.
- [ ] Commit:

```text
docs(reserve): close Phase 2A ledger kernel
```

- [ ] Require a clean worktree and report the full Phase 2A commit range.

## Explicit Non-Actions

- Do not launch hardware, firmware, QEMU, KVM, vfio-user, bridge, or ISA test
  suites.
- Do not touch Halo.
- Do not read, hash, rewrite, move, or delete the external research corpus.
- Do not add canonical bundle emission or verification.
- Do not let JSON award verified-evidence or replica credit.
- Do not create a database, service, scanner, watcher, schema generator, ID
  macro, builder hierarchy, or record-per-file loader.
- Do not add a dependency.
- Do not alter emulator or firmware behavior.
- Do not rewrite historical designs merely because their old
  `clean_release(Aie2)` terminology has been superseded.

## Review Checkpoint

Stop after this plan is committed. Maya reviews the task boundaries, RED/GREEN
sequence, seed IDs, and closure gates before implementation begins.
