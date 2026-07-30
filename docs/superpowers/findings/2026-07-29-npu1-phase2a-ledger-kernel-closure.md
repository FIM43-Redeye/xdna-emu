# NPU1 Phase 2A Research-Reserve Ledger Kernel Closure

**Date:** 2026-07-29

**Status:** Phase 2A complete; the NPU1 retirement gate is intentionally
**BLOCKED**

## Outcome

Phase 2A now has one machine-checked path from a pinned Phoenix tuple through a
full-platform inventory entry, a causal fact candidate, and a historical
evidence record to a fail-closed release report.

The authoritative source is the versioned JSON ledger:

`crates/xdna-archspec/data/research-reserve/npu1.json`

The generated human view is:

`docs/coverage/npu1/release-report.md`

The ledger loads successfully, but its primary tuple does not pass release.
All nine fixed checks are red. This is the expected result: Phase 2A makes the
known work mechanically visible without promoting the legacy emulator capture
or allowing authored JSON to certify external evidence.

## Commits

Approved design and plan:

- `4faa70268aafac4da00bcf26e9ee839a7f1ee2ca` --
  `docs: design NPU1 research-reserve ledger kernel`
- `b7a12a43a5c821a4a128094e7a798d60b52e2678` --
  `docs: plan NPU1 research-reserve ledger kernel`

Implementation checkpoints:

- `3e31684611d28e86848d7061208c68d1771962f6` --
  `refactor(coverage): clarify semantic provenance gate`
- `efb5b2c67af26af555f8332a8e5210f0f3bc77c7` --
  `feat(reserve): define and validate the NPU1 ledger`
- `5f41d283adb06e24d4debc578dc28f82a4b04fd4` --
  `feat(reserve): evaluate NPU1 retirement blockers`
- `0c9e68c2c5b677ccccd1f8675ea2f28867e89c9e` --
  `feat(reserve): seed the Phoenix research ledger`
- `e732f5ee092e0b5d2f4599d9a5aeef9fff24c50e` --
  `docs(reserve): generate the NPU1 release report`

This closure report is committed separately so the preceding implementation
checkpoints remain independently reviewable.

## RED and GREEN Evidence

### Semantic-gate rename

RED:

```text
cargo test -p xdna-archspec --lib semantic_provenance_clean
```

Tests and callers were renamed first. Compilation failed because
`CoverageModel::semantic_provenance_clean()` did not yet exist.

GREEN:

- the focused rename tests passed;
- `xdna-archspec`: 397 passed, 2 ignored;
- root library suite: 4,275 passed, 32 ignored; and
- no current Rust caller retained `CoverageModel::clean_release()`.

### Schema and structural validation

RED:

- parsing tests failed to compile before `ReserveLedger` and its persisted
  types existed;
- nine invalid-ledger mutation cases initially failed because validation
  accepted them; and
- an adversarial diagnostic-path test exposed `$.inventorys[...]` instead of
  the real `$.inventory[...]` root.

GREEN:

```text
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
```

At the schema checkpoint:

- 14 research-reserve tests passed;
- `xdna-archspec`: 411 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

Validation rejects unsupported versions, unknown fields, malformed or
duplicate kind-prefixed IDs, dangling references, unknown coverage domains,
unsafe external paths, malformed digests, empty required text, and
inventory/fact dependency cycles.

### Release evaluator

RED:

```text
cargo test -p xdna-archspec --lib research_reserve::tests::release_
```

Compilation produced 32 expected errors for the absent release report,
blocker, trusted-input, and evaluator API.

GREEN:

- 15 focused release tests passed;
- 29 research-reserve tests passed;
- `xdna-archspec`: 426 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The synthetic all-closed ledger reaches `is_clean == true` only with an
in-module trusted evidence audit. The public production gate always constructs
an empty evidence audit in Phase 2A.

### Embedded Phoenix seed

RED:

```text
cargo test -p xdna-archspec --lib research_reserve::tests::embedded_
```

Compilation failed with two expected `E0599` errors because
`ReserveLedger::npu1()` did not exist.

GREEN:

- both embedded-ledger tests passed;
- 31 research-reserve tests passed;
- `xdna-archspec`: 428 passed, 2 ignored; and
- root library suite: 4,275 passed, 32 ignored.

The tests pin the exact IDs, firmware identity, driver commit, external
location, metadata fingerprint, checksum-index digest, historical-emulator
evidence kind, absent canonical manifest, and zero expected replicas.

### Generated report

RED:

- renderer tests produced four expected `E0425` errors before
  `render_release_report()` existed; and
- the staleness test failed with an empty committed value before
  `docs/coverage/npu1/release-report.md` existed.

GREEN:

- deterministic authored-record reordering produced identical Markdown;
- all fixed checks, tuple pins, open fields, dependency paths, integrity
  digests, preservation notes, and the non-promotion warning are present;
- the staleness test passed;
- a second generator run reproduced identical hashes for every generated
  coverage artifact; and
- `xdna-archspec`: 431 passed, 2 ignored before final closure.

A cached `git diff --check` caught one extra generated blank line at EOF that
an unstaged/untracked diff could not see. The renderer was corrected, the
report regenerated, and the cached whitespace gate then passed.

## Schema and Seed Identity

Schema version: `1`

Seeded stable IDs:

- tuple:
  `tuple.npu1.phoenix.fw-1_5_5_391`
- inventory:
  `inventory.npu1.firmware.command-list-execution`
- fact:
  `fact.npu1.firmware.command-list-lifecycle-candidate`
- evidence:
  `evidence.npu1.legacy-vfio-user-chess-20260729t171244z`

Pinned tuple values:

- architecture: AIE2;
- device: `1022:1502`;
- firmware logical name: `amdnpu/1502_00/npu.dev.sbin`;
- firmware SHA-256:
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- driver surface commit:
  `216cefececd74effcd7a88350c71b99f5ef9a215`.

Historical evidence identity:

- stable location:
  `repo-experiments/phoenix-vfio-user/20260729T171244Z-3136359`;
- metadata fingerprint:
  `4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299`;
- checksum-index SHA-256:
  `e7aaacefa4c8f3606529dd27980397a656b22099a349db59d1c0df84330811e2`;
- kind: `historical_emulator_witness`;
- retention: `witness_capture`;
- redistributability: `restricted`;
- canonical manifest: missing; and
- independent replicas: zero.

The same-filesystem Btrfs reflink snapshot remains a preservation note only
and receives no replica credit.

## Current Release Result

Result: **BLOCKED**

Fixed check results:

- `tuple_identity`: blocked;
- `inventory`: blocked;
- `fact`: blocked;
- `implementation`: blocked;
- `evidence`: blocked;
- `replica`: blocked;
- `semantic_provenance`: blocked;
- `live_attestation`: blocked; and
- `offline_rehearsal`: blocked.

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

`is_clean` is derived only from an empty blocker list.

## Trust Boundary

There are zero production paths that grant verified-evidence or
verified-replica credit.

`ReserveLedger::clean_release(tuple_id)` obtains the semantic result from
`CoverageModel::semantic_provenance_clean()` and constructs an empty private
`EvidenceAudit`. The lower-level evaluator and its trusted inputs are private.
Only same-module tests can construct synthetic verified evidence to prove that
the evaluator is capable of reaching green.

The ledger can describe expected evidence, locations, digests, and replicas.
It cannot certify that any external artifact exists, matches those digests, is
independent, or passed intake.

## Final Verification

Executed from the isolated `investigate/firmware-priors` worktree:

```text
cargo fmt --all -- --check
  PASS

cargo test -p xdna-archspec --lib
  431 passed; 0 failed; 2 ignored

cargo run -p xdna-archspec --example gen_coverage_artifacts
  PASS; AIE2 artifacts and docs/coverage/npu1/release-report.md regenerated

git diff --exit-code
  PASS; no generated drift

cargo test --lib --quiet
  4,275 passed; 0 failed; 32 ignored

git diff --check
  PASS
```

Current-code naming audit:

- every `.clean_release()` call under current Rust source resolves to
  `ReserveLedger::clean_release(tuple_id)`;
- the old semantic-only meaning is named
  `CoverageModel::semantic_provenance_clean()`; and
- historical design/findings terminology was not rewritten.

Scope audit from the approved plan base through the generated-report commit
found:

- no Cargo manifest or lockfile change;
- no external corpus change;
- no canonical bundle emitter or validator;
- no external hash sweep;
- no NPU, firmware, QEMU, KVM, vfio-user, bridge, or ISA-suite run;
- no Halo access;
- no hardware output;
- no database, service, scanner, watcher, macro hierarchy, or new dependency;
  and
- no emulator or firmware behavior change.

## Explicit Non-Claims

Phase 2A does not claim:

- full NPU1 retirement readiness;
- a complete platform inventory;
- that the legacy vfio-user pass is physical-NPU evidence;
- a verified NPU1 hardware fact;
- canonical integrity or reproducibility for the historical capture;
- any independently verified replica;
- live attestation or offline rehearsal;
- lifecycle cleanliness, recovery, repetition, preemption, or cancellation;
- deterministic timing or cycle equivalence;
- coverage of direct execution, Peano, other kernels, older firmware, or
  undocumented/development operations; or
- that the current semantic provenance queue is closed.

## Next Boundary

The next Phase 2 slice is the canonical external bundle emitter and validator.
It must define the campaign manifest, hash and validate every declared
artifact, preserve missing fields as failures, establish independent replica
identity, and produce the private trusted audit input consumed by this gate.

Only after that validator exists can real external evidence receive production
verification credit. The current blocked report is therefore the correct
handoff, not unfinished Phase 2A work.
