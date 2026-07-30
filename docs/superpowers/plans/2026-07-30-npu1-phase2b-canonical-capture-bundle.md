# NPU1 Phase 2B Canonical Capture Bundle Implementation Plan

**Goal:** Build the reusable canonical capture-bundle emitter and validator
which can grant the Phase 2A release gate production evidence and replica
credit without trusting authored manifests, cached receipts, filesystem
topology guesses, or machine-specific paths.

**Architecture:** A new device-neutral `capture_bundle` module in
`xdna-archspec` owns the typed emission plan, canonical manifest, SHA-256
identity, sealed-tree emitter, read-only validator, and opaque validated
result. `research_reserve` remains the only owner of `EvidenceAudit`; it
resolves explicit location roots, validates external bundles in-process, and
immediately evaluates the existing release report. The current NPU1 ledger is
the first consumer, not a structural assumption of the bundle format.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents, Halo, NPU workload, KVM, vfio-user,
QEMU, corpus traversal, or privileged operation. Each behavioral task records
a focused RED before the minimum GREEN, runs its focused regression set, and
commits before the next responsibility begins.

**Approved design:**
[Phase 2B canonical capture-bundle design](../specs/2026-07-30-npu1-phase2b-canonical-capture-bundle-design.md)

## Baseline

The approved design is commit:

```text
ec13879a docs(reserve): design Phase 2B canonical capture bundles
```

Implementation begins from a clean `investigate/firmware-priors` HEAD which
contains this plan and has `ec13879a` as an ancestor.

At planning time:

- `cargo test -p xdna-archspec --lib` passes 431 tests and ignores 2;
- `cargo test --lib --quiet` passes 4,275 tests and ignores 32;
- `research_reserve.rs` owns `ReserveLedger`, the private `EvidenceAudit`, and
  `clean_release(tuple_id)`;
- every production `clean_release()` invocation still supplies an empty
  evidence audit;
- `crates/xdna-archspec/src/capture_bundle/` and the `xdna-reserve` binary do
  not exist;
- `xdna-archspec` already depends on Serde/JSON and has `tempfile` as a
  development dependency;
- no SHA-256 implementation is directly available to `xdna-archspec`;
- the embedded reserve-ledger schema is version 1 and has no canonical
  bundle-ID field; and
- the NPU1 report remains intentionally blocked on all nine fixed release
  checks.

Live files and test output are authoritative during execution. Counts in this
plan record the planning checkpoint; they are not promised final totals.

## Task 0: Preflight and Scope Lock

**Read only:** repository state, approved design, and current trust boundary.

- [ ] Require a clean worktree and approved design ancestry:

```bash
git status --short
git merge-base --is-ancestor ec13879a HEAD
```

- [ ] Re-run the focused and root baselines:

```bash
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
```

- [ ] Re-pin every current production release-gate call and trusted input:

```bash
rg -n 'clean_release\(|EvidenceAudit|EvaluationInputs|EvidenceDigests|StableLocation' \
    crates/xdna-archspec/src crates/xdna-archspec/examples src
```

- [ ] Confirm the planned module and command still do not exist:

```bash
test ! -e crates/xdna-archspec/src/capture_bundle
test ! -e crates/xdna-archspec/src/bin/xdna-reserve.rs
```

- [ ] Stop if the live tree contradicts the approved design, contains
  overlapping changes, or either baseline is red.

## Task 1: Define the Device-Neutral Bundle Schema

**Files:**

- Modify: `crates/xdna-archspec/src/lib.rs`
- Add: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Add: `crates/xdna-archspec/src/capture_bundle/tests.rs`

This task defines data and validation only. It performs no filesystem I/O and
does not add hashing yet.

### Module skeleton

- [ ] Export `capture_bundle` from `lib.rs`.
- [ ] Keep responsibilities explicit:

```text
capture_bundle/mod.rs       persisted types, validation, public API
capture_bundle/canonical.rs canonical bytes and SHA-256 identity (Task 2)
capture_bundle/validator.rs read-only filesystem validator (Task 3)
capture_bundle/emitter.rs   sealed-tree emitter (Task 4)
capture_bundle/tests.rs     focused module tests
```

- [ ] Do not introduce traits, builders, storage backends, ID macros, or
  per-device schema modules.

### Versioned persisted types

- [ ] Define independent version constants for the canonical manifest and
  local emission plan, both beginning at version 1.
- [ ] Derive Serde for every persisted type, use snake-case enum tags, and
  reject unknown fields.
- [ ] Reuse the existing `types::Architecture` through the same stable
  lowercase adapter used by the reserve ledger. Do not add another
  architecture enum.
- [ ] Reuse existing persisted pin and redistributability types where their
  meaning is identical. Move a helper/type to a crate-private shared owner only
  when that removes duplication; do not create a general schema-utilities
  module.
- [ ] Define an explicit availability type:

```rust
pub enum Availability<T> {
    Known { value: T },
    Unavailable { reason: String },
    NotApplicable { reason: String },
}
```

- [ ] Define one canonical manifest root:

```rust
pub struct BundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub campaign: Campaign,
    pub artifacts: Vec<ArtifactRecord>,
}
```

- [ ] Define `Campaign` as the shared identity-bearing payload used by both
  emission plan and final manifest:
  - human-readable campaign ID;
  - tuple, inventory, fact, and evidence ID lists;
  - current/legacy provenance;
  - risk class and campaign outcome;
  - platform identity;
  - stimulus; and
  - separate run records.
- [ ] Use closed enums only where the umbrella design already supplies a
  closed vocabulary:
  - provenance is `Current` or `Legacy`;
  - outcome is `Success`, `IntentionalRejection`,
    `InfrastructureFailure`, `ProvenanceFailure`, `SemanticMismatch`, or
    `DeviceFaultOrWedge`.
  Risk class and artifact semantic kind remain validated stable strings until
  real campaign work establishes a closed taxonomy.
- [ ] Keep platform namespaces separate:
  - `architecture`;
  - toolchain device-model key;
  - driver platform ID;
  - PCI vendor/device/subsystem/revision;
  - board identity;
  - firmware identity/hash;
  - host kernel and module identity;
  - driver revision;
  - XRT components;
  - toolchain components; and
  - relevant reset/power/clock/IOMMU/address state.
- [ ] Model exact command stimulus as argv plus a sorted environment map rather
  than one shell-escaped string.
- [ ] Give source/build identity, input identities, initial state, and external
  events explicit fields.
- [ ] Keep run repetitions separate. Each run records stable ID, ordinal,
  repetition, completion, referenced output/observation artifacts, timing,
  errors, recovery actions, teardown, and control-run IDs.
- [ ] Store timing anchors and bounds as exact integers with explicit units;
  identity-bearing persisted fields contain no floating-point values.
- [ ] Define artifact metadata:
  - canonical `raw/...` or `derived/...` relative path;
  - byte size and SHA-256 in the final manifest;
  - semantic-kind stable string;
  - raw/derived class;
  - redistributability;
  - run/observation references; and
  - derivation provenance for derived artifacts.
- [ ] Define the local `EmissionPlan` with the same `Campaign` plus
  `ArtifactSource` records containing a machine-local source path and canonical
  destination. Source paths never appear in `BundleManifest`.

### Structural validation

- [ ] Reuse one stable-ID helper and one lowercase SHA-256 helper. Reuse the
  existing reserve-ledger implementation by moving a genuinely shared helper
  only if doing so reduces code; do not create a generic validation framework.
- [ ] Validate:
  - supported schema versions;
  - nonblank IDs, reasons, argv, component names, and semantic kinds;
  - duplicate run, artifact, and ledger IDs;
  - run/control references and artifact references;
  - lowercase 64-digit hashes;
  - `raw/...` / `derived/...` canonical UTF-8 paths with `/` separators,
    nonempty components, no `.` / `..`, no backslash, and no ASCII control
    characters;
  - derived provenance required only for derived artifacts;
  - raw artifacts forbidden from claiming derivation;
  - fields where `NotApplicable` is permitted; and
  - complete known identity as a separate promotion-eligibility result rather
    than a parse requirement.

### RED

- [ ] Add tests first for:
  - manifest and emission-plan unknown-field rejection;
  - unsupported versions;
  - lowercase AIE2 and AIE2P round trips;
  - Phoenix/NPU1 and AIE2P/NPU5 identities using the same schema;
  - absent required fields failing deserialization;
  - `Unavailable` parsing while producing a promotion blocker;
  - valid and invalid `NotApplicable`;
  - duplicate and dangling IDs;
  - unsafe, duplicate, and wrong-root artifact paths; and
  - raw/derived provenance mismatch.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::schema_
```

- [ ] Require compilation or test failures which identify the absent schema
  and guards.

### GREEN

- [ ] Implement only the types and guards required by the focused tests.
- [ ] Return sorted field-path diagnostics so authored ordering cannot perturb
  errors.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::schema_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Confirm the diff contains only `lib.rs` and the new module/test skeleton.
- [ ] Commit:

```text
feat(reserve): define canonical capture bundle schema
```

## Task 2: Derive Canonical Bytes and Bundle Identity

**Files:**

- Modify: `crates/xdna-archspec/Cargo.toml`
- Modify: `Cargo.lock`
- Add: `crates/xdna-archspec/src/capture_bundle/canonical.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

### Dependency

- [ ] Add one direct `sha2 = "0.10"` dependency compatible with the
  repository's declared Rust baseline. Do not shell out to `sha256sum`, use
  SHA-1, or implement SHA-256 locally.

### Canonical forms

- [ ] Define one private manifest preimage containing:
  - manifest schema version;
  - the complete `Campaign`; and
  - artifact records sorted by canonical path.
- [ ] Exclude only the generated `bundle_id` from its own preimage.
- [ ] Canonically serialize structs and `BTreeMap` values with one prescribed
  `serde_json::to_vec_pretty` representation and one trailing newline. Do not
  add an RFC-8785 or custom JSON serializer.
- [ ] Reject unstable map types in identity-bearing persisted fields.
- [ ] Canonically sort every semantically unordered collection, including
  ledger IDs, component pins, runs, artifacts, observations, and controls.
  Preserve ordering only for genuinely ordered data such as argv and external
  event sequences.
- [ ] Define bundle IDs as:

```text
bundle.sha256.<64 lowercase hexadecimal digits>
```

- [ ] Render `SHA256SUMS` from sorted artifact records in one lowercase,
  version-pinned `<hash><two spaces><path><newline>` format. It covers only
  `raw/` and `derived/` artifacts, never `manifest.json` or itself.
- [ ] Expose ordinary read-only getters for bundle ID, manifest SHA-256, and
  checksum-index SHA-256. Do not serialize those getters as an audit receipt.

### RED

- [ ] Add tests proving:
  - authored artifact order cannot change manifest, checksum, or bundle-ID
    bytes;
  - different source paths with equal canonical data have the same preimage;
  - any identity-bearing metadata change changes the bundle ID;
  - any artifact size/hash/path change changes the bundle ID;
  - the generated ID cannot influence its own hash;
  - canonical JSON is byte-for-byte stable and ends in one newline;
  - checksum entries are sorted and exact; and
  - manifest and checksum-index digests remain distinct from the bundle ID.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::canonical_
```

- [ ] Require failure because canonical emission and SHA-256 identity do not
  exist.

### GREEN

- [ ] Implement the smallest pure functions for SHA-256, canonical JSON,
  manifest construction, and checksum-index rendering.
- [ ] Keep filesystem paths and I/O out of `canonical.rs`.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::canonical_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Confirm only the hash dependency, lockfile resolution, canonical module,
  and focused tests changed.
- [ ] Commit:

```text
feat(reserve): derive canonical capture bundle identity
```

## Task 3: Validate a Sealed Bundle Read-Only

**Files:**

- Add: `crates/xdna-archspec/src/capture_bundle/validator.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

### Result and error boundary

- [ ] Define `ValidatedBundle` as a public opaque type:
  - its fields are private;
  - it implements no deserialization or public constructor;
  - public getters expose informational identity and blockers;
  - crate-private access supplies the facts required by
    `research_reserve`; and
  - only `validate_bundle(root)` can construct it in production.
- [ ] Keep two outcomes distinct:
  - malformed/tampered/unsafe input returns a sorted validation error and no
    `ValidatedBundle`;
  - canonical byte-integrity with explicit unavailable provenance returns a
    `ValidatedBundle` carrying promotion blockers and no release eligibility.
- [ ] Do not add trusted receipt parsing, signature support, or a validation
  cache.

### Exact tree validation

- [ ] Use `symlink_metadata` and a small stdlib recursive walker.
- [ ] Stream artifact hashing through a bounded buffer. Never read a capture
  artifact wholly into memory.
- [ ] Require exactly:

```text
manifest.json
SHA256SUMS
raw/
derived/
```

- [ ] Reject:
  - a symlink at any root or nested entry;
  - non-regular artifact entries;
  - root files other than the two required files;
  - files outside `raw/` and `derived/`;
  - undeclared files;
  - declared but missing files;
  - duplicate or non-canonical paths;
  - size/hash mismatches;
  - a non-canonical manifest;
  - a non-canonical or mismatched checksum index; and
  - a recomputed bundle-ID mismatch.
- [ ] Directory metadata and empty directories do not affect identity.
- [ ] Keep the validator generic. Ledger expected-digest and tuple/evidence
  cross-checks remain in the later research-reserve adapter.

### RED

- [ ] Build one test-only valid bundle writer from the pure canonical helpers,
  then mutate exactly one property per test.
- [ ] Add tests for:
  - valid round-trip validation;
  - missing and extra root entries;
  - missing, extra, altered, truncated, and substituted artifacts;
  - artifact and directory symlinks;
  - unsafe manifest paths;
  - altered and merely reformatted manifest bytes;
  - reordered, malformed, and mismatched `SHA256SUMS`;
  - forged bundle ID;
  - explicit unavailable provenance returning a blocked opaque result.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::validate_
```

- [ ] Require focused failures before the filesystem validator exists.

### GREEN

- [ ] Implement validation in the fail-closed order from the design:
  shape, schema, references, path set, file types, sizes/hashes, checksum
  index, bundle ID, then promotion eligibility.
- [ ] Accumulate and sort independent validation issues where continued
  inspection is safe; stop immediately when traversal would be unsafe.
- [ ] Audit the public API after GREEN: `ValidatedBundle` must expose no public
  constructor, public fields, `Default`, or deserialization implementation.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::validate_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Commit:

```text
feat(reserve): validate sealed capture bundles
```

## Task 4: Emit a Fresh Bundle Safely

**Files:**

- Modify: `crates/xdna-archspec/Cargo.toml`
- Add: `crates/xdna-archspec/src/capture_bundle/emitter.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

### Staging

- [ ] Promote the already-used `tempfile` dependency from dev-only to a normal
  `xdna-archspec` dependency for same-filesystem staging and automatic cleanup.
  Do not add a custom temporary-directory abstraction.
- [ ] Define `emit_bundle(plan, output)`:
  - validate the plan before filesystem writes;
  - require an existing directory parent and an ordinary final path name;
  - reject an existing output path, including an empty directory;
  - create staging beside the requested output;
  - reject symlink and non-regular artifact sources;
  - copy only declared sources into canonical destinations;
  - stream-copy and hash through one bounded buffer;
  - write canonical manifest and checksum index;
  - call the same public validator implemented in Task 3;
  - recheck that the final path remains absent; and
  - rename the validated staging directory to the final path.
- [ ] On any error, leave no final bundle. The staging guard may clean only the
  unique directory it created.
- [ ] Never overwrite, merge, delete, or mutate an existing destination.

### RED

- [ ] Add tests proving:
  - emission followed by validation succeeds;
  - two plans with equal metadata/bytes but different source/output roots emit
    byte-identical bundle trees;
  - source paths are absent from both canonical files;
  - authored source order is irrelevant;
  - an artifact-byte change changes bundle identity;
  - existing output is untouched;
  - symlink and non-regular sources fail;
  - a missing source fails before the final path appears; and
  - a self-validation failure cannot publish the staging directory.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::emit_
```

- [ ] Require failure before the emitter exists.

### GREEN

- [ ] Implement the descriptor-driven emitter with no directory scanning or
  inferred semantic kinds.
- [ ] Compare the two path-independent emitted trees by relative path and
  bytes, not by inode or metadata.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::emit_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Commit:

```text
feat(reserve): emit canonical capture bundles
```

## Task 5: Add the Minimal `xdna-reserve` Command

**Files:**

- Add: `crates/xdna-archspec/src/bin/xdna-reserve.rs`

### Command contract

- [ ] Implement only:

```text
xdna-reserve emit <emission-plan.json> <output-bundle>
xdna-reserve validate <bundle>
```

- [ ] Parse arguments with `std::env`; do not add a CLI framework.
- [ ] `emit` parses the typed plan and calls `emit_bundle`.
- [ ] `validate` calls `validate_bundle` and prints:
  - bundle ID;
  - manifest SHA-256;
  - checksum-index SHA-256;
  - integrity result; and
  - any promotion blockers.
- [ ] The printed output is explicitly informational. Add no option which
  reads it back as trusted state.
- [ ] Return nonzero for malformed/tampered input or promotion-blocked
  validation so shell automation cannot mistake either for release-ready
  evidence.

### RED

- [ ] Keep argument dispatch in a small testable function and add binary tests
  for:
  - missing/unknown arguments;
  - `emit` round trip through a synthetic plan;
  - `validate` on a valid bundle;
  - `validate` on a tampered bundle; and
  - a promotion-blocked bundle returning nonzero.
- [ ] Run:

```bash
cargo test -p xdna-archspec --bin xdna-reserve
```

- [ ] Require failure before the binary exists.

### GREEN

- [ ] Implement only the command glue required by those tests.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --bin xdna-reserve
cargo check -p xdna-archspec --bin xdna-reserve
cargo test -p xdna-archspec --lib capture_bundle
git diff --check
```

- [ ] Commit:

```text
feat(reserve): add canonical bundle command
```

## Task 6: Record Canonical Identity in the Reserve Ledger

**Files:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`
- Modify: `crates/xdna-archspec/data/research-reserve/npu1.json`
- Modify: `docs/coverage/npu1/release-report.md`

This task advances persisted catalogue identity but still grants no production
evidence credit.

### Schema migration

- [ ] Advance reserve-ledger `SCHEMA_VERSION` from 1 to 2.
- [ ] Extend `EvidenceDigests` with:

```rust
pub bundle_id: Option<String>
```

- [ ] Validate the `bundle.sha256.<hex>` grammar when present.
- [ ] Migrate the embedded NPU1 JSON to schema 2 with `"bundle_id": null`.
- [ ] Preserve the existing historical:
  - metadata fingerprint;
  - checksum-index digest;
  - absent manifest digest; and
  - zero replicas.
- [ ] Render expected canonical bundle ID in the evidence section of the
  generated report. `_missing_` remains an explicit blocked state.

### RED

- [ ] Update tests first to require:
  - schema 1 rejection after migration;
  - valid and malformed bundle IDs;
  - the exact NPU1 legacy digests remaining unchanged;
  - missing canonical bundle ID in the seed;
  - report rendering of the bundle-ID field; and
  - committed report staleness before regeneration.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::parse_
cargo test -p xdna-archspec --lib research_reserve::tests::embedded_
cargo test -p xdna-archspec --lib research_reserve::tests::render_
```

- [ ] Require the expected schema, field, and staleness failures.

### GREEN

- [ ] Implement the schema migration and renderer change.
- [ ] Regenerate:

```bash
cargo run -p xdna-archspec --example gen_coverage_artifacts
```

- [ ] Confirm the report remains blocked with the same release-check and
  blocker-code sets.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib research_reserve
cargo run -p xdna-archspec --example gen_coverage_artifacts
git diff --check
```

- [ ] Run the generator twice and require no second diff.
- [ ] Commit:

```text
feat(reserve): record canonical bundle identity
```

## Task 7: Connect Validated Bundles to the Private Audit

**Files:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`

### Concrete root input

- [ ] Add one concrete caller-supplied root type owned by
  `research_reserve`:

```rust
pub struct BundleLocationRoot {
    pub alias: String,
    pub path: PathBuf,
    pub failure_domain_id: String,
}
```

- [ ] Keep paths and failure-domain declarations out of persisted ledger and
  bundle identity.
- [ ] Reject blank or duplicate aliases and blank failure-domain IDs.
- [ ] Resolve each `StableLocation` as
  `root(alias) / location.relative_path`; reuse the existing safe-relative-path
  validation before joining.
- [ ] Add:

```rust
pub fn clean_release_with_bundle_roots(
    &self,
    tuple_id: &str,
    roots: &[BundleLocationRoot],
) -> Result<ReleaseReport, LedgerError>
```

- [ ] Preserve `clean_release(tuple_id)` as the explicit no-roots report path;
  implement it through the same evaluation path with an empty root slice.

### Trusted audit construction

- [ ] For each candidate evidence record with an expected bundle ID:
  - resolve its primary location;
  - call `validate_bundle`;
  - require release eligibility;
  - cross-check every manifest tuple, inventory, fact, and evidence link
    against the ledger and require the current evidence/tuple relationship;
  - cross-check bundle ID, manifest digest, and checksum-index digest; and
  - grant evidence credit only after every check passes.
- [ ] Evidence with no expected bundle ID remains unaudited even if a
  directory happens to exist at its location.
- [ ] For each expected replica:
  - resolve and independently validate its location;
  - require the complete validated identity to match the primary;
  - require the expected replica ID;
  - require a failure-domain ID not already credited to another replica; and
  - add only the verified `(evidence_id, replica_id)` pair to the private
    audit.
- [ ] Extend the private audit with per-evidence and per-replica failure
  details so existing `evidence_unaudited` and `replica_insufficient` blockers
  can explain why credit was withheld.
- [ ] Catch bundle-validation errors as audit failures. A bad external bundle
  blocks release; it does not make the ledger itself unparsable.
- [ ] Do not expose `EvidenceAudit`, `EvaluationInputs`, or any constructor
  from serialized or CLI output.

### RED

- [ ] Replace the top-level synthetic evidence proof with a validator-path
  test which:
  - emits one complete synthetic primary bundle;
  - creates two byte-identical replica bundles;
  - supplies three explicit roots across distinct declared failure domains;
  - maps all expected IDs/digests into a fully closed synthetic ledger; and
  - constructs the private evidence audit through the exact validator/root
    path used by `clean_release_with_bundle_roots`.
- [ ] Feed that validator-produced audit into the existing private
  `EvaluationInputs` with `semantic_provenance_clean: true` and require an
  empty blocker set. This semantic value remains test-only because the real
  AIE2 semantic-provenance gate is intentionally red.
- [ ] Call public `clean_release_with_bundle_roots` over the same synthetic
  inputs and require every evidence/replica blocker to disappear while the
  genuine semantic-provenance blocker remains. Do not add a public semantic
  override merely to make this test green.
- [ ] Retain private evaluator unit tests only where they isolate release-rule
  logic; no test-only audit may stand in for the production-path integration
  proof.
- [ ] Add focused tests for:
  - an authored bundle ID with no validated root receiving no credit;
  - unknown and duplicate root aliases;
  - primary tamper and expected-digest mismatch;
  - manifest tuple/evidence mismatch;
  - explicit unavailable provenance;
  - replica byte mismatch;
  - duplicate replica ID;
  - duplicate failure-domain ID;
  - one valid replica still producing `replica_insufficient`;
  - the real embedded NPU1 seed preserving its exact blocker-code set when no
    roots are supplied.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::bundle_
cargo test -p xdna-archspec --lib research_reserve::tests::release_production_
```

- [ ] Require failures before the production validator handoff exists.

### GREEN

- [ ] Implement the concrete resolver and private audit builder without a
  trait, global registry, environment lookup, or receipt cache.
- [ ] Keep all external validation in the same process as release evaluation.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::bundle_
cargo test -p xdna-archspec --lib research_reserve::tests::release_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
git diff --check
```

- [ ] Confirm `.clean_release()` callers still resolve to the reserve gate and
  no public API can deserialize trusted audit state or consume CLI output as
  evidence.
- [ ] Commit:

```text
feat(reserve): audit canonical evidence bundles
```

## Task 8: Phase 2B Adversarial Regression and Closure

**File:**

- Add:
  `docs/superpowers/findings/2026-07-30-npu1-phase2b-canonical-capture-bundle-closure.md`

### Canonical and tamper matrix

- [ ] Re-run the focused matrix with fresh temporary directories:
  - two source roots and two output roots produce byte-identical bundles;
  - one metadata mutation changes identity;
  - one artifact mutation changes identity;
  - missing, extra, symlinked, and tampered artifacts fail;
  - non-canonical manifest/checksum bytes fail;
  - explicit unavailable provenance remains preservable but uncredited;
  - two distinct replica domains pass;
  - duplicate domains and mismatched replicas fail; and
  - no validation report can be used as trusted input.
- [ ] Confirm all test fixture output remains temporary. No synthetic bundle is
  committed outside Rust test data needed to express the schema.

### Full verification

- [ ] Run:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --bin xdna-reserve
cargo check -p xdna-archspec --bin xdna-reserve
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
cargo run -p xdna-archspec --example gen_coverage_artifacts
git diff --exit-code
cargo test --lib --quiet
git diff --check
```

- [ ] Re-run the artifact generator and require byte-for-byte idempotence.
- [ ] Confirm the committed NPU1 report remains blocked and names missing
  canonical identity rather than promoting the legacy witness.

### Scope audit

- [ ] Audit the implementation range from the approved design:

```bash
git diff --name-status ec13879a..HEAD
git log --oneline --reverse ec13879a..HEAD
```

- [ ] Confirm:
  - no historical corpus file was read, hashed, moved, or changed;
  - no real capture bundle or proprietary payload entered Git;
  - no NPU, firmware, QEMU, KVM, vfio-user, bridge, ISA, or Halo command ran;
  - no emulator, driver, firmware, or array behavior changed;
  - no database, service, watcher, storage backend, signing system, or CLI
    framework was added;
  - the only new cryptographic dependency is `sha2`;
  - the existing `tempfile` dependency was reused for safe staging; and
  - every implementation commit records its focused RED/GREEN evidence.

### Closure report

- [ ] Record:
  - implementation commit IDs;
  - exact RED and GREEN commands/results;
  - manifest, emission-plan, and reserve-ledger schema versions;
  - the canonical bundle-ID and checksum rules;
  - tamper and path-independence matrix results;
  - replica/failure-domain semantics;
  - proof that only the validator populates production audit credit;
  - final test counts;
  - the still-blocked real NPU1 report and exact remaining blockers;
  - explicit nonclaims; and
  - the fresh NPU1 campaign as the next boundary.
- [ ] Run `git diff --check`.
- [ ] Commit:

```text
docs(reserve): close Phase 2B canonical bundles
```

- [ ] Require a clean worktree and report the complete Phase 2B commit range.

## Explicit Non-Actions

- Do not touch Halo.
- Do not launch hardware, firmware, QEMU, KVM, vfio-user, bridge, or ISA test
  suites.
- Do not read, hash, rewrite, move, copy, or delete the historical research
  corpus.
- Do not produce or retrofit a real canonical evidence bundle.
- Do not promote the legacy Phoenix vfio-user witness.
- Do not close real tuple, inventory, fact, replica, live-attestation, or
  offline-rehearsal blockers.
- Do not infer physical storage independence from paths, mounts, devices,
  Btrfs, LVM, RAID, hostnames, or network metadata.
- Do not trust persisted validation output or add receipt parsing.
- Do not add a database, daemon, watcher, backend interface, plugin system,
  signing system, CLI framework, schema generator, builder hierarchy, or
  device-specific bundle type.
- Do not refactor the existing NPU1 report/ledger entry points merely for
  future products.
- Do not change emulator, firmware, driver, or array behavior.

## Review Checkpoint

Stop after this plan is committed. Maya reviews the schema surface,
canonicalization rules, RED/GREEN task order, root-input API, dependency
changes, and closure gates before implementation begins.
