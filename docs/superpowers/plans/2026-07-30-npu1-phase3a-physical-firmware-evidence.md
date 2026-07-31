# NPU1 Phase 3A Physical Firmware Evidence Implementation Plan

**Goal:** Produce the first fresh canonical physical-NPU1 firmware evidence
pair: one frozen Chess `add_one_using_dma` run through `CHAIN_EXEC_NPU`, one
through `EXEC_DPU`, both under one known module epoch, with exact host output,
firmware-lifecycle evidence, reusable fixture graphs, two complete replicas,
and live reserve-ledger audit credit.

**Architecture:** Extend the existing `xdna-archspec::capture_bundle` format
with version-dispatched v2 fixture and observation documents, explicit fixture
dependencies, and an opaque recursive graph validator. Keep v1 bytes and
identity unchanged. Advance the reserve ledger to schema 3 with a reviewed
expected campaign outcome, and let only in-process graph validation populate
the private `EvidenceAudit`. A campaign-specific Python standard-library tool
owns the frozen NPU1 schedule, oracle, trace parsing, safe privileged
transaction, and deterministic emission-plan generation.

**Execution:** The primary agent works serially in the existing
`firmware-priors` worktree. No subagents. Tasks 1--9 are local and synthetic;
they do not touch NPU hardware or require privilege. Task 10 qualifies a
known-provenance module and stops for review before any load. Task 11 requires
fresh Maya approval for one bounded `pkexec` capture transaction. Task 12
requires joint review of raw and derived evidence before ledger admission.
The 50+50 campaign is implemented but never launched by this plan.

**Approved design:**
[Phase 3A physical firmware evidence design](../specs/2026-07-30-npu1-phase3a-physical-firmware-evidence-design.md)

## Baseline

The approved design and provenance correction are:

```text
1c4f23b2 docs(reserve): design Phase 3A physical evidence
266d7d60 docs(reserve): distinguish executing driver provenance
513f6651 docs(reserve): define pretraffic module rollback
```

Implementation begins from a clean `investigate/firmware-priors` HEAD which
contains this plan and all three commits as ancestors.

At planning time:

- `cargo test -p xdna-archspec --lib` passes 480 tests and ignores 2;
- `cargo test -p xdna-archspec --bin xdna-reserve` passes 5 tests;
- `cargo test --lib --quiet` passes 4,275 tests and ignores 32;
- bundle manifest and emission-plan schemas are version 1;
- the reserve-ledger schema is version 2;
- `capture_bundle` has one self-contained `Campaign` document, an opaque
  `ValidatedBundle`, and no dependency graph;
- `xdna-reserve` implements only `emit` and `validate`;
- no NPU1 firmware-evidence campaign tool exists;
- the loaded module is
  `/lib/modules/7.1.5-custom+/kernel/drivers/accel/amdxdna/amdxdna.ko`;
- that module has SHA-256
  `9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`,
  srcversion `77910A99EDBD0B6C78C8053`, and kernel vermagic
  `7.1.5-custom+`;
- its advertised parameters are `force_iova`, `aie2_max_col`, and
  `force_cmdlist`; it does **not** advertise `tdr_timeout_ms`;
- its exact source-to-module build relationship has not been established;
- the pinned driver-protocol source remains the clean
  `drivers/accel/amdxdna` subtree at
  `amdxdna-driver@216cefececd74effcd7a88350c71b99f5ef9a215`;
- the unrelated dirty `src/driver` and `xrt` state in that sibling repository
  is intentional and must not enter the module build;
- the frozen workload hashes are:

| Artifact | SHA-256 |
|---|---|
| `chess/aie.xclbin` | `c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6` |
| `chess/insts.bin` | `ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896` |
| `chess/test.exe` | `511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806` |
| Phoenix firmware | `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e` |

Live files, module state, and test output are authoritative during execution.
Counts and module facts above record the planning checkpoint, not promised
future state.

## Task 0: Re-pin Scope and Preserve the v1 Oracle

The initial inspection is read-only. The v1 characterization test is the first
write and remains part of the Task 1 commit.

- [ ] Require a clean worktree and approved ancestry:

```bash
git status --short
git merge-base --is-ancestor 1c4f23b2 HEAD
git merge-base --is-ancestor 266d7d60 HEAD
git merge-base --is-ancestor 513f6651 HEAD
```

- [ ] Re-run all three local baselines:

```bash
cargo test -p xdna-archspec --lib
cargo test -p xdna-archspec --bin xdna-reserve
cargo test --lib --quiet
```

- [ ] Re-pin the complete implementation surface:

```bash
rg -n 'MANIFEST_SCHEMA_VERSION|EMISSION_PLAN_SCHEMA_VERSION|BundleManifest|EmissionPlan|ValidatedBundle|validate_bundle|emit_bundle' \
    crates/xdna-archspec/src/capture_bundle crates/xdna-archspec/src/bin
rg -n 'SCHEMA_VERSION|EvidenceRecord|BundleLocationRoot|build_evidence_audit|clean_release_with_bundle_roots' \
    crates/xdna-archspec/src/research_reserve.rs
```

- [ ] Before changing types, add one characterization test which freezes the
  current minimal v1 canonical manifest bytes, checksum bytes, bundle ID,
  manifest hash, and checksum-index hash as literal expected values.
- [ ] Run that characterization test against the unchanged implementation and
  require it to pass. This is the migration oracle, not the RED.
- [ ] Record a read-only fresh `modinfo`, module SHA-256, running-kernel
  identity, and parameter inventory. Do not read privileged parameters, reload
  the module, or submit NPU work.
- [ ] Stop if the live tree contradicts the corrected design or contains
  overlapping changes.

## Task 1: Add Version-Dispatched v2 Fixture and Observation Types

**Files:**

- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

This task defines persisted types and structural validation only. It performs
no filesystem I/O and changes no canonicalization yet.

### Version dispatch

- [ ] Preserve the exact v1 field set and field order in a dedicated v1
  document type.
- [ ] Represent the public manifest and emission-plan documents as explicit
  version-dispatched values. Dispatch by a minimal `schema_version` header
  before deserializing the complete v1 or v2 document.
- [ ] Reject unsupported versions, non-integer versions, duplicate JSON keys,
  and fields unknown to the selected version.
- [ ] Do not deserialize v1 through v2 optional fields, add defaults to v1, or
  round-trip a v1 document through a `serde_json::Value`.
- [ ] Advance both current schema constants to 2 while retaining named v1
  constants for compatibility tests.

### v2 role bodies

- [ ] Define one v2 root with:
  - schema version;
  - generated bundle ID in a manifest only;
  - one role-specific payload;
  - canonical dependency requirements; and
  - local artifact records or sources.
- [ ] Use one adjacently tagged payload:

```text
role: fixture | observation
body: <role-specific typed body>
```

- [ ] Define the fixture body with only:
  - stable fixture ID;
  - stable semantic fixture kind;
  - current/legacy provenance;
  - exact source revisions;
  - explicit construction/acquisition recipe availability; and
  - fixture-specific notes needed to interpret its artifacts.
- [ ] Define the observation body around the existing `Campaign`; do not copy
  that type into a second near-identical campaign schema.
- [ ] For v2 observations, define the existing
  `campaign.platform.driver` field as the executing module's source revision.
  Represent the pinned driver-protocol surface only through its separately
  typed fixture/input reference. Never overload one field with both meanings.
- [ ] Add explicit observation input references which map each existing
  stimulus input ID to one fixture bundle ID and artifact path.
- [ ] Keep fixture artifacts free of run and observation IDs. Observation
  artifacts retain the current run/observation linkage.

### Typed dependency requirements

- [ ] Define one dependency requirement per consumed fixture artifact:

```text
fixture_bundle_id
artifact_path
artifact_sha256
semantic_kind
```

- [ ] Permit multiple distinct artifact requirements from one fixture, such as
  the XCLBIN and instruction stream.
- [ ] Reject duplicate `(fixture_bundle_id, artifact_path)` requirements and
  conflicting requirements for the same artifact.
- [ ] Require every observation input reference to resolve to one declared
  requirement whose path, hash, and semantic kind match the corresponding
  stimulus input.
- [ ] Permit dependency targets only when the target is a v2 fixture.
  Observation bundles can never be dependency targets.
- [ ] Keep machine paths out of canonical v2 documents. Local dependency paths
  exist only in the v2 emission plan and the location plan.

### RED

- [ ] Add tests first for:
  - v2 fixture and observation JSON shape;
  - role-specific required and forbidden fields;
  - unknown role and unknown schema version;
  - duplicate-key rejection during version dispatch;
  - fixture run-link rejection;
  - observation input resolution;
  - missing, duplicate, and conflicting requirements;
  - local paths being absent from canonical manifests; and
  - the Task 0 v1 characterization remaining unchanged.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_schema_
```

- [ ] Require compilation or assertion failures which identify the absent v2
  types and validation.

### GREEN

- [ ] Implement only the version dispatcher, types, and structural guards
  required by the focused tests.
- [ ] Reuse existing `Campaign`, `Availability`, `ArtifactRecord`,
  `ArtifactSource`, and stable-string/hash/path validators.
- [ ] Do not add traits, builders, generic schema utilities, or role-specific
  modules.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v1_
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_schema_
cargo test -p xdna-archspec --lib capture_bundle
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(reserve): version capture bundle roles
```

## Task 2: Canonicalize v2 Identities Without Moving v1

**Files:**

- Modify: `crates/xdna-archspec/src/capture_bundle/canonical.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

### Canonical rules

- [ ] Dispatch canonicalization by document version.
- [ ] Leave the v1 preimage, sort order, JSON bytes, checksum rendering, and
  bundle-ID calculation byte-for-byte unchanged.
- [ ] Define the v2 bundle ID as SHA-256 over the canonical v2 manifest
  preimage with only its generated bundle ID omitted.
- [ ] Include role, role body, dependency requirements, and artifact records in
  the v2 identity.
- [ ] Canonically sort:
  - fixture source revisions;
  - dependency requirements by bundle ID then artifact path;
  - observation input references by input ID;
  - existing semantically unordered campaign fields; and
  - artifact records by path.
- [ ] Keep run order and other semantically ordered sequences ordered.
- [ ] Reuse the existing SHA-256 and canonical JSON functions. Add no
  dependency.

### RED

- [ ] Add tests proving:
  - the frozen v1 bytes and digests remain exact;
  - source ordering does not perturb v2 fixture identity;
  - dependency ordering does not perturb v2 observation identity;
  - changing a required bundle ID, artifact path, hash, or semantic kind
    changes v2 identity;
  - changing a fixture artifact changes fixture identity;
  - changing one observation run changes observation identity; and
  - local dependency source paths do not enter identity.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_canonical_
```

- [ ] Require focused failures before v2 canonicalization exists.

### GREEN

- [ ] Extend the existing canonical helpers rather than creating a second
  hashing subsystem.
- [ ] Keep filesystem I/O out of `canonical.rs`.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v1_
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_canonical_
cargo test -p xdna-archspec --lib capture_bundle
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(reserve): canonicalize fixture dependencies
```

## Task 3: Validate Explicit Bundle Graphs

**Files:**

- Modify: `crates/xdna-archspec/src/capture_bundle/mod.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/validator.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`
- Modify: `crates/xdna-archspec/src/research_reserve.rs`

### Location-plan boundary

- [ ] Extend the existing concrete `BundleLocationRoot` rather than adding a
  storage abstraction. Each root contains:
  - alias;
  - root path;
  - operator-attested failure-domain ID; and
  - explicit bundle-ID-to-relative-path entries.
- [ ] Require the map to contain the root observation or v1 leaf as well as
  every dependency reachable beneath it.
- [ ] Add one typed local `BundleLocationPlan` wrapper shared by the CLI and
  reserve evaluator.
- [ ] Reject blank/duplicate aliases, blank failure domains, unsafe relative
  paths, duplicate bundle IDs, and mappings which escape their declared root.
- [ ] Keep the location plan local and non-canonical. Do not hash it, copy it
  into a bundle, or accept it as evidence.

### Opaque graph result

- [ ] Keep `ValidatedBundle` opaque and version-aware.
- [ ] Add an opaque `ValidatedBundleGraph` which privately retains:
  - the validated root;
  - every validated fixture by bundle ID; and
  - graph-level promotion blockers.
- [ ] Expose informational identity/count accessors only. Keep role bodies and
  graph maps crate-private for immediate ledger audit.
- [ ] Add one graph-validation entry point taking the root bundle path and one
  selected location root. CLI selection from a complete plan must identify
  exactly one root containing the requested root bundle.

### Recursive validation

- [ ] Reuse `validate_bundle` for every node.
- [ ] For each declared dependency:
  1. resolve its bundle ID only through the selected root's explicit map;
  2. validate that bundle canonically;
  3. require v2 fixture role;
  4. require exact declared bundle ID;
  5. locate the required artifact;
  6. compare exact path, SHA-256, and semantic kind; and
  7. recurse through that fixture's own requirements.
- [ ] Use a small depth-first walk with `visiting` and `complete` bundle-ID
  sets. Reject cycles and never scan directories for candidates.
- [ ] Reject a map which sends two required bundle IDs to the same declared
  bundle location.
- [ ] Treat a v1 bundle as a valid self-contained leaf with no dependencies,
  but never as a v2 fixture target.
- [ ] Never fall back to another root.

### RED

- [ ] Add tests for:
  - one valid observation with two fixture artifacts;
  - a fixture depending on a fixture;
  - missing mapping;
  - wrong mapped bundle ID;
  - wrong artifact path, hash, or semantic kind;
  - duplicate dependency and duplicate mapped location;
  - fixture cycle;
  - dependency on an observation;
  - valid v1 self-contained leaf;
  - independent primary and replica graphs;
  - missing replica fixture;
  - cross-root fallback rejection; and
  - all validation output remaining informational.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::graph_
```

- [ ] Require failures before recursive validation exists.

### GREEN

- [ ] Implement the single recursive validator in `validator.rs`; add no
  registry, scanner, cache, trait, or database.
- [ ] Sort and deduplicate independent diagnostics.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::graph_
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib research_reserve
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(reserve): validate canonical bundle graphs
```

## Task 4: Emit and Reuse v2 Fixture Graphs Safely

**Files:**

- Modify: `crates/xdna-archspec/src/capture_bundle/emitter.rs`
- Modify: `crates/xdna-archspec/src/capture_bundle/tests.rs`

### Emission

- [ ] Dispatch emission by plan version.
- [ ] Preserve existing v1 destination and staging behavior.
- [ ] For v2:
  1. validate plan structure;
  2. hash local artifact sources;
  3. build the canonical root document;
  4. validate every dependency through the plan's explicit local paths;
  5. stage only the new bundle's files;
  6. validate the staged bundle and its dependency graph before rename; and
  7. atomically rename into an absent destination.
- [ ] Never copy dependency bytes into the new bundle.
- [ ] If a destination already exists, validate it and return it only when its
  complete canonical identity equals the planned bundle. Otherwise fail
  without mutation. This is the only fixture-reuse mechanism.
- [ ] Never overwrite, merge, delete, repair, or partially update an existing
  bundle.

### RED

- [ ] Add tests proving:
  - v2 fixture emission and validation;
  - v2 observation emission against explicit fixtures;
  - no dependency bytes appear in the observation tree;
  - absent/malformed/substituted dependencies fail before final rename;
  - identical existing fixtures are reused;
  - a mismatched existing destination is untouched;
  - failure leaves no final bundle;
  - two source roots emit identical v2 trees; and
  - v1 emission remains exact.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_emit_
```

- [ ] Require focused failures before v2 emission exists.

### GREEN

- [ ] Extend the current tempfile-backed emitter and graph validator only.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib capture_bundle::tests::v2_emit_
cargo test -p xdna-archspec --lib capture_bundle
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(reserve): emit reusable fixture graphs
```

## Task 5: Audit Reviewed Campaign Outcomes Through Complete Graphs

**Files:**

- Modify: `crates/xdna-archspec/src/research_reserve.rs`
- Modify: `crates/xdna-archspec/data/research-reserve/npu1.json`
- Modify: `crates/xdna-archspec/src/capture_bundle/validator.rs`
- Modify: `docs/coverage/generated/research-reserve.md`

### Ledger schema 3

- [ ] Advance the reserve-ledger schema from 2 to 3.
- [ ] Add `EvidenceRecord.expected_campaign_outcome` as an explicitly tagged:
  - `known`, containing the shared typed `CampaignOutcome`; or
  - `unavailable`, containing a nonblank reason.
- [ ] Do not add `null`, implicit defaults, or stringly typed outcomes.
- [ ] Migrate the historical evidence record to explicit unavailable without
  changing its IDs, digests, disposition, or existing blocker meaning.
- [ ] Render the expected outcome in the reserve report.

### Graph-backed audit

- [ ] Replace leaf-only bundle validation in `build_evidence_audit` with
  complete graph validation under the primary root and under each replica
  root.
- [ ] Require the root to be an observation bundle for graph-backed evidence.
- [ ] Cross-check:
  - root bundle, manifest, and checksum-index digests;
  - tuple, inventory, fact, and evidence links;
  - exact expected campaign outcome;
  - promotion eligibility of root and dependencies;
  - replica root identity;
  - every replica graph's complete byte identity; and
  - distinct operator-attested failure-domain IDs.
- [ ] A structurally valid failure observation receives audit credit only when
  its ledger record explicitly expects that exact failure outcome. It still
  enters supporting, control, or counterevidence lists only through authored
  ledger review.
- [ ] An outcome mismatch, unavailable expected outcome, missing fixture,
  cross-root fallback, or graph blocker yields no evidence credit.
- [ ] Keep `EvidenceAudit` private and ephemeral. Persist no trusted receipt.

### RED

- [ ] Update tests first for:
  - schema-2 rejection and schema-3 acceptance;
  - known/unavailable outcome validation;
  - exact success match;
  - success/failure mismatch;
  - reviewed non-success match;
  - valid primary graph with two complete replica graphs;
  - one missing replica fixture;
  - one substituted replica fixture;
  - duplicate replica failure domain;
  - no roots preserving the current blocked report; and
  - CLI-shaped informational output being unusable as audit input.
- [ ] Run:

```bash
cargo test -p xdna-archspec --lib research_reserve::tests::bundle_
cargo test -p xdna-archspec --lib research_reserve::tests::campaign_outcome_
```

- [ ] Require focused schema and audit failures before implementation.

### GREEN

- [ ] Implement the schema migration and replace only the leaf-validation call
  inside the existing audit builder.
- [ ] Regenerate the report twice and require the second run to leave the
  first run's bytes unchanged:

```bash
cargo run -p xdna-archspec --example gen_coverage_artifacts
cargo run -p xdna-archspec --example gen_coverage_artifacts
```

- [ ] Pass:

```bash
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
git diff --check
```

- [ ] Confirm the real NPU1 report remains blocked with the same unrelated
  release requirements.
- [ ] Commit:

```text
feat(reserve): audit expected campaign outcomes
```

## Task 6: Add the Two Operational CLI Commands

**File:**

- Modify: `crates/xdna-archspec/src/bin/xdna-reserve.rs`

### Command contract

- [ ] Retain existing `emit` and `validate`.
- [ ] Add only:

```text
xdna-reserve validate-graph <bundle> <location-plan.json>
xdna-reserve audit <ledger.json> <tuple-id> <location-plan.json>
```

- [ ] Continue parsing arguments with `std::env`; add no CLI dependency.
- [ ] Parse the location plan into the same Rust type used by the validator and
  reserve evaluator.
- [ ] `validate-graph` first validates the supplied leaf to obtain its bundle
  ID, then selects exactly one root whose map assigns that ID to that exact
  supplied path. It validates the graph and prints an explicitly informational
  graph identity/count report.
- [ ] `audit` parses the supplied ledger, performs fresh in-process graph
  validation, evaluates the tuple, and prints the existing rendered release
  report.
- [ ] Return nonzero for malformed input, validation failure, promotion
  blockers, or a non-clean release report. A blocked report remains useful
  stdout, not a command success.
- [ ] Accept no receipt, cached validation result, override, or
  `--trust-existing` option.

### RED

- [ ] Add binary tests for:
  - exact argument arity;
  - valid graph report;
  - missing fixture failure;
  - valid synthetic audit reaching clean;
  - blocked audit returning nonzero with rendered blockers; and
  - informational output being rejected as a location plan.
- [ ] Run:

```bash
cargo test -p xdna-archspec --bin xdna-reserve
```

- [ ] Require focused failures before command dispatch exists.

### GREEN

- [ ] Add only the two match arms and small shared JSON-read helper needed by
  the tests.
- [ ] Pass:

```bash
cargo test -p xdna-archspec --bin xdna-reserve
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(reserve): add bundle graph audit commands
```

## Task 7: Model the Frozen Campaign and Derive Its Oracle

**Files:**

- Add: `tools/npu1_firmware_evidence.py`
- Add: `tools/test_npu1_firmware_evidence.py`

This task is pure Python. It executes no command and touches no hardware.

### Schema-first campaign model

- [ ] Use frozen dataclasses for:
  - treatment/control arm;
  - schedule entry;
  - command result;
  - raw capture index;
  - lifecycle record;
  - output-oracle result;
  - run classification;
  - campaign classification; and
  - fixture/observation emission-plan inputs.
- [ ] Keep the exact firmware, driver-protocol, XCLBIN, instruction, executable,
  opcode, output, and normal-TDR pins in one campaign-specific `VerticalSpec`.
- [ ] Do not build a generic laboratory framework, plugin API, backend
  interface, or second schema implementation.

### Deterministic pure functions

- [ ] Implement:
  - seeded one-per-arm vertical scheduling;
  - seeded balanced 50+50 scheduling;
  - strict ordered `2..=65` output parsing;
  - strict `PASS!` agreement;
  - trace/log marker scoping;
  - request-opcode extraction;
  - direct and command-list execute-status extraction from their exact
    one-word and three-word response byte layouts;
  - lifecycle ordering for create, map, configure, execute, completion, and
    destroy;
  - TDR and IOMMU delta detection;
  - teardown/restoration accounting;
  - fail-fast campaign classification; and
  - deterministic v2 fixture and observation emission-plan JSON generation.
- [ ] Preserve unknown success response words as explicit unknowns.
- [ ] Classify missing capture data as infrastructure/provenance failure, not
  hardware success or semantic mismatch.
- [ ] Never retry or smooth an anomaly into a distribution.

### RED

- [ ] Add `unittest` cases for all campaign-tool cases in the design:
  - deterministic and balanced schedules;
  - exact output success;
  - bare `PASS!` rejection;
  - missing/wrong/duplicate/out-of-order opcodes;
  - treatment `0x18` and control `0x10`;
  - coherent lifecycle;
  - TDR/IOMMU deltas;
  - timeout and nonzero exit;
  - fail-fast partial sealing;
  - cleanup accounting;
  - fixture reuse; and
  - byte-stable emission plans.
- [ ] Use compact inline trace/log fixtures. Add no captured physical data.
- [ ] Run:

```bash
python3 -m unittest tools/test_npu1_firmware_evidence.py
```

- [ ] Require focused failures before the pure functions exist.

### GREEN

- [ ] Implement the minimum pure code required by the tests using only the
  Python standard library.
- [ ] Pass:

```bash
python3 -m unittest tools/test_npu1_firmware_evidence.py
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(evidence): model NPU1 firmware campaign
```

## Task 8: Add the Safe Capture Transaction and Background Launcher

**Files:**

- Modify: `tools/npu1_firmware_evidence.py`
- Modify: `tools/test_npu1_firmware_evidence.py`

### Ordinary coordinator

- [ ] Add one ordinary-user command for a vertical campaign and one for the
  later 50+50 campaign.
- [ ] Require an explicit campaign ID, deterministic seed, and qualified
  module manifest. Accept a location plan when supplied, but do not require
  storage-attestation choices until sealing and replication.
- [ ] Create all working state beneath:

```text
build/experiments/npu1-firmware-evidence/<campaign-id>/
```

- [ ] Refuse an existing nonempty campaign directory. Never use `/tmp` for
  persistent state.
- [ ] Preflight exact hashes, physical PCI identity, absent emulator
  environment, XRT resolution, tool/source revisions, the recorded pre-load
  system-module identity, candidate module provenance, no active NPU client,
  trace capability, parameter capability, and normal TDR.
- [ ] Invoke only fixed argv subprocesses with `shell=False`.
- [ ] Generate one bounded request for an internal privileged mode. The JSON
  request contains typed data, not arbitrary commands.

### Internal privileged mode

- [ ] Invoke the same script once through `pkexec` for the complete module
  epoch. Reuse the proven `runuser` pattern for each fresh ordinary-user
  `test.exe` process.
- [ ] Resolve the invoking account from `PKEXEC_UID`, require that account to
  own the request and campaign directory, and pass its numeric identity
  explicitly to `runuser`. Never trust root's inherited `USER`.
- [ ] Validate the request again as root, including all frozen hashes, absence
  of active NPU clients, and that every output path remains under the campaign
  directory.
- [ ] In one transaction:
  1. snapshot original mutable state;
  2. unload the current module and `insmod` the reviewed candidate once with
     normal `tdr_timeout_ms=2000`;
  3. re-pin PCI `power/control` to `on` after the candidate's normal runtime-PM
     initialization, record its before/after values, verify the live module
     bytes/build identity, and read back TDR state;
  4. create one dedicated tracefs instance;
  5. enable exactly the available required amdxdna lifecycle events;
  6. enable only source-qualified amdxdna request/response dynamic-debug
     callsites;
  7. emit matching run-boundary markers;
  8. set and read back `force_cmdlist` for each serialized run;
  9. run the frozen executable as Maya's UID with `XDNA_EMU*` absent;
  10. snapshot trace, kernel log, return code, and state after every run;
  11. apply the same pure lifecycle/oracle classifier before starting the next
      run and stop at the first anomaly; and
  12. unconditionally stop tracing and restore debug controls and the original
      writable parameter where safe.
- [ ] If setup fails before the first NPU submission, unload the candidate and
  restore the recorded original system module and its initial `power/control`
  policy, then verify both. This is a pre-traffic rollback, not device
  recovery.
- [ ] After the first NPU submission, never reload the previous module in
  cleanup. A successful campaign leaves the reviewed candidate loaded pending
  review; a failed campaign preserves first-failure device state.
- [ ] Never disable TDR, run concurrent jobs, invoke recovery, PM-cycle,
  suspend, reset, reboot, or call `xrt-smi` during traffic.
- [ ] Use a bounded process timeout which exceeds normal driver TDR but cannot
  hang unattended forever.

### Background launch

- [ ] Render one fixed `systemd-run --user` argv for the separately approved
  50+50 command.
- [ ] Write a terminal status file atomically and rely on the user journal.
- [ ] Add no daemon, watcher, polling loop, or Codex monitoring.
- [ ] Do not launch the service in this implementation slice.

### RED

- [ ] Test with fake command results and temporary directories:
  - preflight pin drift;
  - emulator-environment rejection;
  - pre-load system-module drift;
  - candidate-module provenance rejection;
  - missing normal-TDR capability;
  - unsafe request path;
  - wrong/missing `PKEXEC_UID`;
  - command argv/environment;
  - dedicated trace-instance setup/cleanup plan;
  - exact dynamic-debug selector restoration;
  - serialized runuser commands;
  - pre-traffic module rollback;
  - no post-traffic module rollback;
  - first-failure stop;
  - no recovery command;
  - terminal status write; and
  - deterministic transient-service argv.
- [ ] No test may invoke `pkexec`, `modprobe`, `insmod`, `runuser`, tracefs, or
  the NPU.
- [ ] Run:

```bash
python3 -m unittest tools/test_npu1_firmware_evidence.py
```

- [ ] Require focused failures before orchestration exists.

### GREEN

- [ ] Implement one script with an internal privileged subcommand. Add no shell
  helper.
- [ ] Pass:

```bash
python3 -m unittest tools/test_npu1_firmware_evidence.py
cargo test --lib --quiet
git diff --check
```

- [ ] Commit:

```text
feat(evidence): capture NPU1 firmware lifecycle
```

## Task 9: Prove the Complete Synthetic Vertical Path

**Files:**

- Modify only the tests from Tasks 1--8 if the end-to-end proof exposes a real
  missing assertion.

### Synthetic graph

- [ ] In a temporary directory, create all six canonical fixtures:
  - firmware;
  - driver protocol;
  - executing driver;
  - runtime/toolchain;
  - NPU program; and
  - host oracle.
- [ ] Give the executing-driver fixture an explicit relationship artifact and
  dependency on the driver-protocol fixture.
- [ ] Generate one synthetic successful vertical observation which consumes
  all six fixtures and contains distinct `0x18` and `0x10` runs.
- [ ] Emit the primary and two replica graphs through production emitters.
- [ ] Validate all three graphs through production graph validation.
- [ ] Build a synthetic schema-3 ledger expecting `success`; prove the private
  audit grants evidence and replica credit.
- [ ] Change only the observed outcome and prove credit disappears.
- [ ] Remove one fixture from one replica and prove no cross-root fallback.
- [ ] Run the same emission twice and prove fixture reuse creates no duplicate
  bytes or changed identity.

### Pre-hardware gate

- [ ] Run:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --bin xdna-reserve
python3 -m unittest tools/test_npu1_firmware_evidence.py
cargo test -p xdna-archspec --lib
cargo test --lib --quiet
git diff --check
```

- [ ] Re-run the coverage-artifact generator and require no second diff.
- [ ] Confirm no NPU, privilege, QEMU, KVM, vfio-user, bridge, ISA, or Halo
  command has run.
- [ ] Commit any test-only correction with the component it corrects; do not
  create an empty checkpoint commit.

## Task 10: Qualify a Known Executing Module

**Checkpoint:** Stop and show Maya the pre-hardware gate and proposed module
build before creating a sibling worktree, building, signing, unloading, or
loading anything.

**Durable report after qualification:**

- Add:
  `docs/superpowers/findings/2026-07-30-npu1-phase3a-executing-driver-qualification.md`

### Re-audit the current module

- [ ] Repeat the read-only module, kernel, parameter, tracepoint, and
  dynamic-debug capability inventory.
- [ ] If the current module unexpectedly has exact source/build provenance and
  an equivalent enabled normal-TDR mechanism plus the complete required
  observability surface, document that evidence and ask whether to use it.
- [ ] Otherwise preserve the expected verdict: current module is unqualified;
  do not use it for the physical pair.

### Build candidate without touching intentional dirt

- [ ] Create a clean detached worktree of the sibling `xdna-driver` repository
  at exact commit `216cefececd74effcd7a88350c71b99f5ef9a215`.
- [ ] Verify the clean `drivers/accel/amdxdna` source bytes and record its tree
  identity before building.
- [ ] Build only that out-of-tree module against the exact running-kernel build
  headers. Use the kernel's configured compiler mode and no source patch.
- [ ] Record:
  - source commit and tree hash;
  - kernel release, config hash, `Module.symvers` hash, and header identity;
  - compiler/linker identities;
  - exact build argv and environment;
  - unsigned module hash;
  - signing command and signed module hash when Secure Boot requires it;
  - `modinfo`, srcversion, vermagic, dependencies, and parameters; and
  - source-qualified tracepoint and dynamic-debug callsites.
- [ ] Never copy or record a private signing key.
- [ ] Require the candidate to expose writable `force_cmdlist`, normal
  `tdr_timeout_ms=2000`, and all lifecycle/request evidence required by the
  proof boundary.
- [ ] Do not install or overwrite anything beneath `/lib/modules`. The
  candidate will be loaded transiently by absolute path with `insmod`.
- [ ] If the exact source does not compile or load against the running kernel
  without a source patch, stop. Porting the driver is a new design decision,
  not an implementation workaround.

### Restoration proof

- [ ] Hash and record the untouched system module which normal `modprobe`
  resolves.
- [ ] Record the exact manual success-path restoration command:

```text
rmmod amdxdna
modprobe amdxdna
```

- [ ] Prove offline that normal `modprobe` still resolves the original system
  module. Do not execute restoration or any load yet.
- [ ] Write the qualification report with a direct
  qualified-for-bounded-load/unqualified verdict and `git diff --check`.
- [ ] Commit:

```text
docs(evidence): qualify Phase 3A driver module
```

- [ ] Stop again. Maya reviews the module hashes, debug surface, build
  provenance, and restoration path before the privileged capture.

## Task 11: Capture the One Physical Vertical Pair

**Checkpoint:** Requires Maya's explicit approval of one bounded `pkexec`
transaction. Confirm no other NPU user is running.

### Preflight

- [ ] Create a fresh campaign ID and working directory beneath
  `build/experiments/npu1-firmware-evidence/`.
- [ ] Defer the operator-chosen primary and two replica roots to Task 12, where
  sealing and replication actually use them. Do not infer their failure
  domains.
- [ ] Re-hash all four frozen inputs and the qualified module.
- [ ] Require physical `1022:1502`, exact firmware, exact candidate-module
  manifest, normal-TDR capability, required trace/debug surfaces, the recorded
  pre-load system module, and no active NPU client.
- [ ] Record a deterministic seed and the resulting two-arm order before
  privilege or NPU traffic.

### Capture

- [ ] Run the ordinary vertical command once.
- [ ] Authorize its one `pkexec` request.
- [ ] Require one module epoch and one fresh process/context per arm.
- [ ] Do not retry either arm.
- [ ] On any anomaly, stop, preserve raw state, and ask Maya before recovery.
- [ ] Do not run another NPU command merely to check progress.

### Derive

- [ ] After the command exits, derive both results only from immutable raw
  capture files.
- [ ] Require for each arm:
  - exact command line and environment;
  - context create;
  - host-buffer map;
  - CU configuration;
  - exactly one expected execute opcode;
  - interrupt/response/queue-head/scheduler/fence completion;
  - context destroy;
  - zero process exit;
  - exact ordered output `2..=65`;
  - `PASS!`;
  - no attributable TDR or IOMMU delta; and
  - complete teardown/restoration.
- [ ] Seal a failure classification if any requirement is absent. Do not
  reinterpret missing evidence as success.
- [ ] Leave the qualified module loaded and stop for Maya's raw/derived review.
  Restore the normal system module only through a later explicit successful
  review decision.

## Task 12: Seal, Replicate, Review, and Admit the Evidence

**Files after successful review:**

- Modify: `crates/xdna-archspec/data/research-reserve/npu1.json`
- Modify: `docs/coverage/generated/research-reserve.md`

### Host restoration

- [ ] After Maya confirms that the raw first-state evidence is preserved and
  the device is healthy, request separate approval for one success-path
  restoration transaction:

```text
rmmod amdxdna
modprobe amdxdna
```

- [ ] Verify that the reloaded module exactly matches the recorded original
  system-module bytes and restore the recorded initial `power/control` policy.
  Do not submit NPU traffic merely to test restoration.
- [ ] If the device is faulted or wedged, do not perform this step
  automatically; stop and choose recovery with Maya.

### Canonical graph

- [ ] Emit the six fixtures once under each declared root:
  - exact firmware payload and acquisition identity;
  - pinned driver-protocol source identity and relevant source archive;
  - signed executing module, build manifest, and explicit relationship to the
    driver-protocol fixture without duplicating its source archive;
  - runtime/toolchain identity manifests;
  - XCLBIN, instructions, source MLIR, and build recipe; and
  - host executable, source, and build recipe.
- [ ] Reuse any exact pre-existing fixture only through canonical validation.
- [ ] Emit one observation bundle under each root.
- [ ] Validate the primary and both replica graphs independently with
  `xdna-reserve validate-graph`.
- [ ] Confirm each replica contains its complete graph and no path crosses to
  another root.
- [ ] Keep all payloads external to Git. Commit only schemas, ledger metadata,
  and concise findings.

### Joint review

- [ ] Review:
  - raw stdout/stderr;
  - trace and kernel logs;
  - module/parameter transitions;
  - both lifecycle derivations;
  - exact output oracle;
  - TDR/IOMMU deltas;
  - cleanup state;
  - fixture graph;
  - canonical identities; and
  - redistributability.
- [ ] If the pair failed, do not add it automatically to supporting, control,
  or counterevidence. Decide its scientific disposition separately.
- [ ] If the pair passed, proceed with the scoped successful admission below.

### Successful ledger admission

- [ ] Advance the embedded ledger with:
  - evidence
    `evidence.npu1.firmware.physical-execution-envelope-pair`;
  - fact
    `fact.npu1.firmware.physical-execution-envelope-pair`;
  - exact bundle, manifest, and checksum-index digests;
  - expected campaign outcome `success`;
  - primary and two expected replica locations;
  - `HardwareWitness`, `witness_capture`, and restrictive graph
    redistributability;
  - both run IDs and explicit control-run relationship; and
  - exact protocol-surface and executing-driver source references.
- [ ] Add the evidence ID to the tuple's live-attestation list.
- [ ] Move only
  `inventory.npu1.firmware.command-list-execution` to the new physical closing
  fact. Preserve the historical fact and evidence unchanged.
- [ ] Remove only tuple-identity missing fields directly proven by this graph.
- [ ] Keep the new fact `observed`; do not promote it to verified, encoded, or
  retirement-qualified.
- [ ] Preserve all unrelated inventory, implementation, timing, rehearsal, and
  retirement blockers.

### Live audit

- [ ] Run:

```bash
cargo run -p xdna-archspec --bin xdna-reserve -- \
    audit crates/xdna-archspec/data/research-reserve/npu1.json \
    tuple.npu1.phoenix.fw-1_5_5_391 \
    <local-location-plan.json>
```

- [ ] Require the command to remain nonzero because NPU1 is not retirement
  ready, while the new evidence itself receives live evidence and replica
  credit.
- [ ] Regenerate the tracked report twice and require idempotence.
- [ ] Run the complete pre-hardware test gate again.
- [ ] Commit:

```text
docs(reserve): admit NPU1 physical firmware pair
```

## Task 13: Adversarial Regression and Closure

**File:**

- Add:
  `docs/superpowers/findings/2026-07-30-npu1-phase3a-physical-firmware-evidence-closure.md`

### Regression matrix

- [ ] Re-run with fresh temporary graphs:
  - frozen v1 identity;
  - v2 fixture/observation identity;
  - missing/substituted/duplicate/cyclic dependencies;
  - observation dependency-target rejection;
  - independent replica validation;
  - cross-root fallback rejection;
  - exact outcome match/mismatch;
  - reviewed failure outcome;
  - no trusted receipt;
  - fixture reuse; and
  - complete synthetic ledger credit.

### Full verification

- [ ] Run:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib capture_bundle
cargo test -p xdna-archspec --lib research_reserve
cargo test -p xdna-archspec --bin xdna-reserve
python3 -m unittest tools/test_npu1_firmware_evidence.py
cargo test -p xdna-archspec --lib
cargo run -p xdna-archspec --example gen_coverage_artifacts
git diff --exit-code
cargo test --lib --quiet
git diff --check
```

- [ ] Do not run the full bridge suite, ISA suite, QEMU, KVM, vfio-user,
  emulator, or Halo. This slice changes evidence plumbing and performs only
  the explicitly approved two-run physical campaign.

### Closure report

- [ ] Record:
  - implementation commit range;
  - exact RED and GREEN results;
  - schema versions 1/2/3 and v1 identity proof;
  - graph and replica adversarial matrix;
  - protocol-surface versus executing-driver identities;
  - qualified module build/load/restoration provenance;
  - exact raw and derived vertical-pair outcome;
  - fixture and observation bundle IDs;
  - primary and replica validation results;
  - ledger admission and live audit result;
  - every remaining release blocker;
  - explicit nonclaims; and
  - the separately approved 50+50 campaign as the next checkpoint.
- [ ] Require a clean worktree after the closure commit:

```text
docs(reserve): close Phase 3A physical evidence
```

## Explicit Non-Actions

- Do not use subagents.
- Do not use Halo.
- Do not read, move, rewrite, delete, or retrofit the historical evidence
  corpus.
- Do not overwrite or install a module beneath `/lib/modules`.
- Do not build from the dirty sibling driver tree.
- Do not patch driver source to compile, add instrumentation, or expose
  response words.
- Do not disable TDR.
- Do not add kprobes, eBPF, BPF tracing, a database, daemon, watcher, object
  store, storage backend, receipt format, CLI framework, or Rust campaign
  framework.
- Do not run two NPU jobs concurrently.
- Do not retry an anomalous hardware run.
- Do not automate module reload, PM-cycle, suspend, bus reset, or reboot
  recovery.
- Do not trust filesystem topology as proof of failure-domain independence.
- Do not duplicate shared fixture bytes inside observation bundles.
- Do not change firmware, driver protocol, XRT, compiler, emulator, or array
  behavior.
- Do not claim general firmware equivalence, response-payload completeness,
  determinism, timing equivalence, cycle accuracy, or NPU1 retirement.
- Do not launch the 50+50 campaign.

## Review Checkpoints

1. Stop after this plan is committed. Maya reviews the schema/API changes,
   task order, module boundary, physical proof boundary, and closure gates
   before implementation begins.
2. During implementation, stop after Task 9. Review all synthetic evidence and
   the exact proposed module-build procedure before touching the sibling
   driver repository or requesting privilege.
3. Stop after Task 10. Review the qualified module bytes, debug surface,
   provenance, and restoration path before loading it.
4. Stop after Task 11. Review raw and derived physical evidence before
   canonical admission or restoration.
5. Stop after Phase 3A closure. The 50+50 transient service requires a new
   explicit approval.
