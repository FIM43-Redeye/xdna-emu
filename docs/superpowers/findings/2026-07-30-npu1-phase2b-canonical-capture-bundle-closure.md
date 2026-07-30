# NPU1 Phase 2B Canonical Capture Bundle Closure

**Date:** 2026-07-30

**Status:** Phase 2B complete at the approved synthetic boundary; the real
NPU1 retirement gate remains intentionally **BLOCKED**

## Outcome

Phase 2B now has one device-neutral, machine-checked path from a typed emission
plan to an immutable canonical bundle, an opaque validator result, and the
private research-reserve evidence audit.

A complete synthetic primary bundle and two byte-identical synthetic replicas
can satisfy the evidence and replica checks through the production
`clean_release_with_bundle_roots()` path. The ordinary semantic-provenance
input remains red, so the public synthetic release report still contains that
genuine blocker. A private evaluator test proves the same validator-produced
audit reaches a clean report when the semantic input is explicitly true.

The approved boundary was synthetic only. No historical witness was
retrofitted, no real capture was bundled, no corpus payload was read or
changed, and no NPU or remote machine was used.

## Commits

Approved design and implementation plan:

- `ec13879a7d7c569a2b6445d47ac094b97591f471` --
  `docs(reserve): design Phase 2B canonical capture bundles`
- `2ec8a49e745958b7aa21bf7b5c24baa753336667` --
  `docs(reserve): plan Phase 2B canonical capture bundles`

Implementation checkpoints:

- `bd121ce53a586bf5e084e68c6af0ac014e3c8c75` --
  `feat(reserve): define canonical capture bundle schema`
- `73a2e5d572eb64add84c16ef5edbd73e1bba6235` --
  `feat(reserve): derive canonical capture bundle identity`
- `929a5e8d8093307a4fbf1a261bb5ea07a350623e` --
  `feat(reserve): validate sealed capture bundles`
- `92053f684ae57e37cf08af001734286f0a7643ef` --
  `feat(reserve): emit canonical capture bundles`
- `50a8be5aa69c62f16f46545698021845a2ab0245` --
  `feat(reserve): add canonical bundle command`
- `9f455b81a2d8947d35aeb4a1768104bda4dc2f42` --
  `feat(reserve): record canonical bundle identity`
- `d8b85e4bc07f6675bdb10c24f7c11221f48e2d99` --
  `feat(reserve): audit canonical evidence bundles`

This closure report is committed separately so every implementation checkpoint
remains independently reviewable.

## RED and GREEN Evidence

### Manifest and emission-plan schema

RED:

```text
cargo test -p xdna-archspec --lib capture_bundle::tests::schema_
```

The first run failed to compile because `BundleManifest`, `EmissionPlan`, both
schema-version constants, and `BundleSchemaError` did not exist. A later
adversarial test failed before compiler and execution modes were added to the
platform identity.

GREEN:

- 13 focused schema tests passed;
- `xdna-archspec`: 444 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The schema rejects unknown fields, unsupported versions, absent required
fields, malformed hashes and IDs, unsafe paths, duplicate or dangling
references, invalid raw/derived provenance, and invalid availability states.

### Canonical bytes and identity

RED:

```text
cargo test -p xdna-archspec --lib capture_bundle::tests::canonical_
```

Compilation failed because `build_canonical_bundle()` and
`canonicalize_manifest()` did not exist.

GREEN:

- 6 focused canonical tests passed;
- 19 capture-bundle tests passed;
- `xdna-archspec`: 450 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The tests prove exact canonical bytes, authored-order independence,
source-path exclusion, bundle-ID preimage exclusion, distinct content and file
digests, and identity changes for relevant metadata or artifact changes.

### Sealed-tree validator

RED:

```text
cargo test -p xdna-archspec --lib capture_bundle::tests::validate_
```

Compilation failed because `validate_bundle()` did not exist. An adversarial
substitution test then remained red until the validator compared every
declared artifact's bytes and hash rather than only the tree shape.

GREEN:

- 11 focused validator tests passed;
- 30 capture-bundle tests passed;
- `xdna-archspec`: 461 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

Validation rejects missing, extra, altered, truncated, substituted, and
symlinked artifacts; wrong root entries; unsafe manifest paths; forged bundle
IDs; and non-canonical manifest or checksum-index bytes.

### Atomic emitter

RED:

```text
cargo test -p xdna-archspec --lib capture_bundle::tests::emit_
```

Compilation failed because `emit_bundle()` and the emitter module did not
exist.

GREEN:

- 7 focused emitter tests passed;
- 37 capture-bundle tests passed;
- `xdna-archspec`: 468 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

Emission streams each source through a 64 KiB copy-and-hash buffer, writes a
fresh sibling staging directory, self-validates it, and renames it into a
previously absent destination. Missing, symlinked, and non-regular sources,
existing output paths, and self-validation failure do not publish a final
bundle.

### Minimal command surface

RED:

```text
cargo test -p xdna-archspec --bin xdna-reserve
```

Compilation failed because the command's `run()` entry point did not exist.

GREEN:

- all 5 command tests passed;
- `cargo check -p xdna-archspec --bin xdna-reserve` passed;
- all 37 capture-bundle tests passed;
- `xdna-archspec`: 468 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The command surface is only:

```text
xdna-reserve emit <emission-plan.json> <output-bundle>
xdna-reserve validate <bundle>
```

Validation output labels itself informational and not a trusted receipt.

### Reserve-ledger identity migration

RED:

```text
cargo test -p xdna-archspec --lib research_reserve::tests::parse_
cargo test -p xdna-archspec --lib research_reserve::tests::embedded_
cargo test -p xdna-archspec --lib research_reserve::tests::validation_accepts_and_rejects_canonical_bundle_ids
cargo test -p xdna-archspec --lib research_reserve::tests::release_report_is_not_stale
```

The first compile failed because `EvidenceDigests` had no `bundle_id`. The
canonical-ID validation test then failed until the new field rejected malformed
values. The report staleness test failed until the schema-2 seed and generated
report recorded the missing canonical identity.

GREEN:

- all 35 research-reserve tests passed;
- `xdna-archspec`: 469 passed, 2 ignored;
- root library suite: 4,275 passed, 32 ignored; and
- two generator runs produced no second diff.

The historical metadata fingerprint and checksum-index digest remained
unchanged. The seed gained a distinct, explicitly missing canonical bundle ID
instead of repurposing either legacy digest.

### Private trusted handoff

RED:

```text
cargo test -p xdna-archspec --lib research_reserve::tests::bundle_
```

Compilation produced 16 expected errors because `BundleLocationRoot`,
`build_evidence_audit()`, and `clean_release_with_bundle_roots()` did not
exist.

GREEN:

- 10 focused validator-to-reserve tests passed;
- both production-gate tests passed;
- all 46 research-reserve tests passed;
- all 37 capture-bundle tests passed;
- `xdna-archspec`: 480 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The focused matrix covers no-roots non-credit, blank and duplicate root
aliases, blank failure-domain IDs, unsafe locations, missing aliases, primary
tamper, all three canonical digest mismatches, unknown ledger links,
unavailable provenance, replica tamper, duplicate replica IDs, duplicate
failure domains, and the one-replica insufficiency case.

## Schema and Canonical Identity

Schema versions:

- capture-bundle manifest: `1`;
- emission plan: `1`; and
- research-reserve ledger: `2`.

The exact sealed tree is:

```text
manifest.json
SHA256SUMS
raw/
derived/
```

Artifacts exist only below `raw/` and `derived/`. Root entries outside the two
required files and two required directories are rejected. Symlinks, absolute
paths, parent traversal, empty path components, backslashes, and undeclared
files are rejected.

Canonicalization sorts every order-insensitive ID, component, revision, input,
run, observation, timing, control, artifact-reference, and artifact-path list.
Environment maps are `BTreeMap`s. Canonical JSON is Serde pretty JSON followed
by exactly one newline.

The bundle ID is:

```text
bundle.sha256.<sha256(canonical {
    schema_version,
    campaign,
    artifacts
})>
```

The authored bundle-ID field and local artifact source paths are excluded from
that preimage. Artifact byte sizes and SHA-256 values are included through the
canonical artifact records.

After computing the bundle ID:

- `manifest_sha256` hashes the exact final canonical `manifest.json` bytes,
  including the computed bundle ID; and
- `checksum_index_sha256` hashes the exact canonical `SHA256SUMS` bytes.

Each checksum line is:

```text
<lowercase artifact SHA-256><two spaces><canonical artifact path><newline>
```

Artifact lines are ordered by canonical artifact path. Campaign ID, bundle ID,
manifest digest, checksum-index digest, and the legacy filesystem-metadata
fingerprint remain distinct identities.

## Adversarial Matrix

| Property | Fresh synthetic check | Result |
|----------|-----------------------|--------|
| Path independence | Two source roots and two output roots, with reversed authored artifact order, emitted byte-identical trees | PASS |
| Metadata sensitivity | Relevant campaign metadata mutation changed the bundle ID | PASS |
| Artifact sensitivity | Source artifact byte mutation changed artifact and bundle identity | PASS |
| Closed tree | Missing and extra root entries or artifacts failed validation | PASS |
| Symlink exclusion | Bundle root, required entry, artifact, artifact-directory, and emitter-source symlinks failed | PASS |
| Artifact integrity | Altered, truncated, and substituted artifacts failed | PASS |
| Canonical bytes | Reformatted or altered manifest and reordered, malformed, or mismatched checksum index failed | PASS |
| Explicit unavailable state | Structurally valid unavailable provenance remained inspectable but promotion-ineligible | PASS |
| Valid replicas | Primary plus two byte-identical replicas across three declared failure domains received evidence and replica credit | PASS |
| Replica faults | Content mismatch, duplicate replica ID, duplicate failure domain, and one valid replica withheld sufficient credit | PASS |
| No receipt trust | CLI output has no deserializable trusted form; release evaluation reopens and revalidates every configured bundle | PASS |

Every synthetic filesystem fixture uses `tempfile::TempDir`. No synthetic
bundle, emitted artifact, validation report, or payload entered Git.

## Replica and Failure-Domain Semantics

`BundleLocationRoot` supplies three out-of-band values:

- stable alias;
- local root path; and
- declared failure-domain ID.

Those values do not enter persisted ledger data or bundle identity. Stable
locations are revalidated as safe relative paths before joining them to the
caller-supplied root.

The primary and every expected replica are independently reopened and fully
validated in the same process as release evaluation. A replica receives credit
only when its bundle ID, manifest digest, and checksum-index digest match the
validated primary and its declared failure domain has not already received
credit. The primary domain also cannot be reused by a credited replica.

This proves verified copies across declared failure domains. It does not infer
physical independence from paths, devices, mounts, Btrfs, LVM, RAID,
hostnames, or network metadata.

## Trusted Handoff

`ValidatedBundle` has private fields and no deserializer. Public callers can
inspect only stable identity and promotion diagnostics; the manifest accessor
needed by `research_reserve` is crate-private.

`EvidenceAudit` and `EvaluationInputs` are private and have no serialized
form. The only production insertions into their verified-evidence and
verified-replica sets occur inside `ReserveLedger::build_evidence_audit()`
after live validation, ledger-link checks, promotion eligibility, digest
checks, replica identity checks, and failure-domain checks.

`clean_release(tuple_id)` delegates to the same path with an empty root slice.
Its NPU1 report is byte-for-byte unchanged from the explicit empty-roots call.
Authored ledger identity, CLI text, and persisted validation output cannot
grant credit.

## Current NPU1 Release Result

Result: **BLOCKED**

All nine release checks remain blocked:

- `tuple_identity`;
- `inventory`;
- `fact`;
- `implementation`;
- `evidence`;
- `replica`;
- `semantic_provenance`;
- `live_attestation`; and
- `offline_rehearsal`.

Exact blocker-code set:

- `tuple_identity_open`
- `inventory_scope_open`
- `inventory_fact_unqualified`
- `fact_not_retirement_qualified`
- `fact_unknowns_open`
- `fact_control_evidence_missing`
- `fact_alternatives_missing`
- `implementation_missing`
- `tests_missing`
- `evidence_legacy_incomplete`
- `evidence_provenance_incomplete`
- `evidence_unaudited`
- `replica_insufficient`
- `semantic_provenance_open`
- `live_attestation_missing`
- `offline_rehearsal_missing`

The legacy witness still has:

- no canonical bundle ID;
- no canonical manifest digest;
- seven provenance gaps;
- zero expected independent replicas; and
- only its unchanged metadata fingerprint and checksum-index digest.

The same-filesystem Btrfs preservation copy remains a note with no replica
credit.

## Final Verification

Executed locally from the isolated `investigate/firmware-priors` worktree:

```text
cargo fmt --all -- --check
  PASS

cargo test -p xdna-archspec --lib capture_bundle
  37 passed; 0 failed

cargo test -p xdna-archspec --bin xdna-reserve
  5 passed; 0 failed

cargo check -p xdna-archspec --bin xdna-reserve
  PASS

cargo test -p xdna-archspec --lib research_reserve
  46 passed; 0 failed

cargo test -p xdna-archspec --lib
  480 passed; 0 failed; 2 ignored

cargo run -p xdna-archspec --example gen_coverage_artifacts
  PASS

git diff --exit-code
  PASS before adding this closure report

cargo test --lib --quiet
  4,275 passed; 0 failed; 32 ignored

git diff --check
  PASS
```

The artifact generator was run twice from a clean implementation tree. Both
runs reproduced the committed files byte-for-byte, and the second run left no
diff.

## Scope Audit

The implementation range from approved design through trusted handoff is
`ec13879a..d8b85e4b`.

The range changes only:

- the Phase 2B plan;
- `xdna-archspec` capture-bundle schema, canonicalization, validator, emitter,
  command, and reserve adapter;
- the reserve-ledger schema-2 seed;
- its generated NPU1 release report;
- the crate manifest and lockfile; and
- the crate module export.

The audit confirms:

- no historical corpus file was read, hashed, moved, copied, or changed;
- no real capture bundle or proprietary payload entered Git;
- no NPU, firmware, QEMU, KVM, vfio-user, bridge, ISA, privileged, or Halo
  command ran;
- no emulator, driver, firmware, or array behavior changed;
- no database, daemon, service, watcher, storage backend, signing system,
  plugin system, or CLI framework was added;
- `sha2` is the only new cryptographic dependency;
- the existing `tempfile` dependency moved from dev-only to normal dependency
  scope for safe emitter staging;
- all test output remained temporary; and
- every implementation commit is associated above with its focused RED and
  GREEN evidence.

## Explicit Non-Claims

Phase 2B does not claim:

- that the historical Phoenix witness is canonical, reproducible, or eligible
  for promotion;
- that any real NPU1 evidence or independent physical replica has been
  validated;
- that declared failure domains prove physical storage independence;
- a complete NPU1 tuple, inventory, or firmware command surface;
- a retirement-qualified firmware lifecycle fact;
- live attestation or offline rehearsal;
- warning-clean lifecycle, recovery, repetition, preemption, or cancellation;
- deterministic firmware timing or cycle equivalence;
- direct execution, Peano, other kernels, older firmware, or
  undocumented/development-operation coverage;
- closure of the semantic perishable or comprehension queues; or
- readiness to retire or replace the owned NPU1 hardware.

## Next Boundary

The next slice is a fresh NPU1 evidence campaign, not a retrofit of the legacy
vfio-user witness.

That campaign should begin by pinning the exact Phoenix platform, firmware,
driver, runtime, toolchain, workload, reset/power/clock epoch, stimulus, and
control relationships in an emission plan. It can then run the inexpensive
physical-NPU repetitions, emit a canonical primary bundle, create and validate
two declared replicas, and map the resulting identities into the ledger.

Phase 2B supplies the trustworthy preservation and handoff mechanism for that
work. It intentionally does not supply the evidence itself.
