# NPU1 Phase 2B Canonical Capture Bundle -- Design

**Date:** 2026-07-30

**Status:** Approved

**Scope:** A reusable, device-neutral canonical capture-bundle schema,
deterministic emitter, offline validator, replica validation, and the trusted
in-process handoff to the Phase 2A research-reserve release gate

## Purpose

Phase 2A established the catalogue side of the NPU1 research reserve:

- a typed, versioned ledger;
- one Phoenix tuple-to-evidence seed chain;
- a deterministic release report;
- a fail-closed `clean_release()` evaluator; and
- a private `EvidenceAudit` which authored JSON cannot populate.

That release report is intentionally blocked because Phase 2A has no
production path which can establish that an external evidence bundle exists,
is complete, matches its declared identity, or validates any replicas declared
by the ledger.

Phase 2B builds that missing trust bridge:

```text
typed local emission plan
  -> deterministic sealed external bundle
  -> offline integrity and provenance validation
  -> opaque in-memory validated result
  -> private EvidenceAudit
  -> existing release evaluator
```

This slice proves the bridge entirely with small synthetic bundles. It does not
modify the historical corpus or capture new hardware evidence. The first real
canonical bundle will be produced from a fresh NPU campaign in a following
slice; retrofitting the incomplete legacy witness here would conflate format
correctness with evidence quality.

## Selected Approach

The canonical bundle machinery is a new device-neutral module in
`xdna-archspec`. It owns:

- the versioned manifest and emission-plan types;
- canonical serialization and bundle identity;
- SHA-256 calculation;
- bundle emission;
- offline validation;
- replica-content validation; and
- an opaque validated result whose fields cannot be authored by callers.

The existing research-reserve module remains the owner of the private
`EvidenceAudit`. It consumes validated bundle results, cross-checks them against
ledger expectations, and grants evidence or replica credit only after all
required checks pass.

This is preferred over two alternatives:

- Adding the bundle implementation directly to `research_reserve.rs` would mix
  a reusable multi-device format into an already large NPU1-ledger module.
- Creating a new workspace crate or storage service would add dependency and
  trust-boundary plumbing without providing useful separation in this slice.

A small command-line front end exposes `emit` and `validate` over the same
library implementation. It does not contain a second parser or validator.

## Reuse Boundary

The bundle contract is reusable across NPU generations from its first version.
It must not encode Phoenix, NPU1, AIE2, or the current firmware command path as
structural assumptions.

Platform identity keeps distinct namespaces distinct:

- architecture family, using the existing `Architecture` type;
- toolchain device-model key, such as mlir-aie's `npu1`;
- driver-facing platform ID, such as NPU1, NPU4, or NPU5;
- PCI vendor, device, subsystem, and revision identity;
- physical board identity where available; and
- firmware, driver, runtime, and toolchain pins.

These are separate because an architecture family is not a product identity:
multiple products share AIE2P, and mlir-aie model names do not match the
driver's public NPU numbering.

Only the new bundle layer is generalized in Phase 2B. The current NPU1
embedded-ledger constructor, report path, and report title are not refactored
merely to anticipate future ledgers.

## Bundle Unit and Immutability

One bundle represents one capture campaign under one stable platform and
stimulus tuple. Repetitions are separate run records inside that campaign. A
change to behaviorally relevant platform, stimulus, initial-state, or external
event information requires a new bundle.

A canonical bundle is immutable. Adding, removing, or changing an artifact or
identity-bearing field produces a different bundle identity. Corrections to a
derived analysis produce a new derived revision and therefore a new bundle;
raw evidence is never silently rewritten.

## Exact Bundle Tree

The canonical external tree is:

```text
manifest.json
SHA256SUMS
raw/
derived/
```

The tree is closed:

- every regular file beneath `raw/` and `derived/` is declared in the manifest
  and checksum index;
- every declared artifact exists exactly once;
- undeclared regular files are errors;
- symlinks are forbidden;
- absolute paths, parent traversal, and non-canonical relative paths are
  forbidden;
- root files other than `manifest.json` and `SHA256SUMS` are forbidden; and
- validation output is written outside the immutable bundle.

Empty directories carry no identity and need not be preserved. Directory mode,
inode, owner, and modification time do not contribute to canonical identity.

The existing historical `metadata_fingerprint_sha256` field retains its Phase
1A meaning as a filesystem-metadata census fingerprint. It is not silently
redefined as the canonical bundle ID.

## Manifest Model

`manifest.json` is a versioned, deny-unknown-fields document with these logical
sections:

### Identity

- manifest schema version;
- human-readable, operator-assigned campaign ID;
- generated content-derived bundle ID;
- mapped tuple, inventory, fact, and evidence IDs;
- provenance class;
- risk class; and
- campaign outcome.

### Platform

- architecture family;
- toolchain device-model key;
- driver-facing platform ID;
- physical PCI, subsystem, revision, and board identity;
- firmware logical identity and hashes;
- host kernel and module identity;
- driver source and revision;
- XRT packages and relevant library hashes;
- `aie-rt`, `mlir-aie`, `llvm-aie`, register-database, and other relevant
  toolchain revisions;
- compiler and execution modes; and
- relevant reset, power, clock, IOMMU, and address-mode state.

### Stimulus

- exact command and behaviorally relevant environment;
- source and build-recipe identity;
- executable, XCLBIN, ELF, PDI, CDO, configuration, and input hashes as
  applicable;
- known initial state; and
- explicit external-event schedule.

### Runs and observations

- run ordinal and repetition identity;
- output and completion status;
- register and memory observations;
- transaction and trace references;
- timing anchors and bounds;
- errors, wedges, recovery actions, and teardown result; and
- expected control relationships.

Runs remain separate records. Repetition is not collapsed into a distribution
or summary which would hide causal state.

### Artifacts

Each artifact records:

- canonical relative path;
- byte size;
- SHA-256;
- semantic kind;
- raw or derived class;
- redistributability/privacy classification; and
- references from runs or observations.

Derived artifacts additionally identify their source artifact or source bundle,
the exact analysis command, and the analysis-tool revision. A synthetic
implementation fixture is explicitly marked synthetic and cannot masquerade as
raw hardware evidence.

## Missing and Inapplicable Data

Required identity fields do not use a bare nullable value. Their state is
explicit:

```text
known(value)
unavailable(reason)
not_applicable(reason)
```

The outcomes are distinct:

- An absent required field makes the manifest malformed.
- `unavailable` preserves an incomplete capture as canonical evidence but
  blocks release evidence credit.
- `not_applicable` is accepted only for fields whose schema and campaign kind
  permit it.

A well-formed but provenance-incomplete bundle may pass byte-integrity
validation and remain useful for research. It cannot appear in the private
verified-evidence set. The passing synthetic fixture uses complete known
identity throughout.

## Emission Plan

Emission is descriptor-driven. It never infers a campaign or semantic artifact
kind by scanning a working directory.

The typed local emission plan contains:

- the manifest metadata before generated identity fields;
- each source artifact path;
- each artifact's canonical `raw/...` or `derived/...` destination;
- semantic and redistributability metadata.

Source paths are local operational inputs. They may be absolute and
machine-specific, but they never enter `manifest.json`, `SHA256SUMS`, or bundle
identity.

The emitter:

1. validates the plan before copying;
2. refuses an existing final output path;
3. creates a fresh staging directory beside the requested output;
4. rejects symlink sources and copies only explicitly declared regular files;
5. computes sizes and hashes while constructing the artifact inventory;
6. emits canonical `manifest.json` and `SHA256SUMS`;
7. runs the same validator used by the public validation command; and
8. atomically renames the validated staging directory to the final path.

It never overwrites or deletes an existing bundle. A failed emission does not
create a valid-looking final bundle.

## Canonical Identity and Serialization

The campaign ID is readable and human-assigned. It is a reference, not an
integrity claim.

The bundle ID is derived from:

- the canonical identity-bearing manifest payload;
- the sorted artifact inventory; and
- every artifact size and SHA-256.

The generated bundle-ID field is excluded from its own preimage. Filesystem
source paths and output locations are never part of the preimage.

Canonical JSON has one version-defined serialization. Map-like content uses
stable key ordering, artifact and run records use specified stable ordering,
and the file has a single prescribed trailing newline. The validator
re-serializes the parsed manifest and rejects non-canonical bytes rather than
accepting semantically equivalent but differently encoded JSON.

`SHA256SUMS` has one version-defined lowercase format sorted by canonical
artifact path. It covers all files beneath `raw/` and `derived/`; it does not
include `manifest.json` or itself, avoiding a checksum cycle.

Three identities remain distinct:

- bundle ID: stable content-derived identity;
- manifest SHA-256: exact final `manifest.json` bytes; and
- checksum-index SHA-256: exact final `SHA256SUMS` bytes.

The ledger is extended to record an expected bundle ID without repurposing the
legacy metadata fingerprint. That schema change advances the reserve-ledger
schema version; the embedded NPU1 record is migrated with no expected bundle ID
and remains blocked. Any expected digest present in the ledger must match
before evidence receives credit.

SHA-256 is provided by one small Rust cryptographic hash dependency. Phase 2B
does not shell out to `sha256sum` and does not implement SHA-256 locally.

## Validator

Validation is read-only. It checks, in fail-closed order:

1. required root entries and exact directory shape;
2. manifest readability, schema version, unknown fields, and canonical bytes;
3. stable IDs, references, ordering, and explicit availability states;
4. canonical artifact paths and artifact-set equality;
5. regular-file and no-symlink requirements;
6. every artifact size and SHA-256;
7. exact canonical `SHA256SUMS` contents;
8. recomputed bundle ID;
9. manifest and checksum-index digests expected by the ledger;
10. tuple and ledger-ID agreement; and
11. provenance completeness required for release evidence credit.

Malformed input, a missing artifact, an extra artifact, unsafe paths, or any
digest mismatch yields no validated result and no evidence credit.

A structurally valid bundle with explicit unavailable provenance yields an
opaque result carrying its blockers, but cannot populate the verified-evidence
set.

## Replica Semantics

The project normally trusts this machine and its operator. Phase 2B therefore
does not attempt to infer physical storage topology from device numbers,
mounts, Btrfs subvolumes, LVM, RAID, network filesystems, or cloud metadata.

Replica validation proves:

- the expected replica ID was supplied;
- the replica root is separately opened and fully validated;
- its bundle ID and complete byte set match the primary bundle; and
- its declared failure-domain ID is distinct from every replica already
  receiving credit.

Replica and failure-domain identities are supplied out of band with stable
location-root mappings. They do not alter bundle content.

Evidence records may declare zero or more expected replicas. Validation audits
every declared replica, but evidence admission has no universal replica-count
minimum. Backup policy protects against loss; it is separate from canonical
evidence validity.

The release report should describe this honestly as verified copies across
attested failure domains, not as machine-proven physical independence. Two
different paths or two same-filesystem snapshots never receive automatic
independence credit merely because they exist.

## Trusted Handoff

Validator output is opaque: callers can inspect diagnostics and stable public
facts, but cannot construct a value which grants release credit.

The research-reserve integration:

1. resolves each ledger `StableLocation` through caller-supplied roots;
2. validates the primary bundle and every expected replica in-process;
3. cross-checks bundle, tuple, evidence, replica, and expected-digest identity;
4. constructs the private `EvidenceAudit`; and
5. immediately evaluates the release report.

There is no reusable trusted receipt. A CLI may print or save an informational
validation report, but no API accepts that report as proof. Every release
evaluation reopens and revalidates the configured bundles.

This preserves the Phase 2A invariant that authored JSON cannot self-certify
external evidence.

## Command Surface

The Phase 2B command front end is intentionally small:

```text
xdna-reserve emit <emission-plan.json> <output-bundle>
xdna-reserve validate <bundle>
```

It uses the library types and validation path directly. It does not introduce a
daemon, database, storage abstraction, plugin system, or second configuration
language.

Release-root orchestration for real NPU1 evidence belongs to the following
fresh-campaign slice. Phase 2B needs only the concrete library input required
to exercise the trusted handoff with synthetic roots.

## TDD and Verification

Implementation begins with RED tests for:

1. identical plans and bytes emitting identical bundles from different source
   and output paths;
2. any relevant metadata or artifact-byte change producing a different bundle
   ID;
3. deterministic manifest and checksum-index bytes;
4. round-trip emission and validation;
5. missing, extra, altered, truncated, and substituted artifacts;
6. altered or non-canonical manifest and checksum-index bytes;
7. absolute paths, parent traversal, duplicate paths, and symlinks;
8. absent, unavailable, and invalidly not-applicable required fields;
9. mismatched tuple, evidence, ledger digest, and bundle identity;
10. replica content mismatch, duplicate replica ID, and duplicate
    failure-domain ID;
11. an informational report being unusable as trusted validator input;
12. a complete synthetic ledger plus two declared synthetic replicas reaching
    a clean release through the production validator path; and
13. the committed NPU1 seed remaining blocked with no external bundle roots.

The focused crate suite and ordinary repository library suite remain required:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib
cargo test --lib
git diff --check
```

No hardware, QEMU, KVM, vfio-user, bridge, ISA, or Halo run is part of Phase
2B verification.

## Non-Goals

Phase 2B does not:

- modify, hash, move, or repackage the historical 5.2 GiB corpus;
- retrofit the legacy Phoenix vfio-user witness into a canonical bundle;
- run the NPU or capture new hardware evidence;
- promote the Phase 2A seed fact;
- close tuple identity, inventory, live-attestation, or offline-rehearsal
  blockers;
- populate the full NPU1 platform inventory;
- refactor existing NPU1 report ownership for hypothetical future ledgers;
- infer storage hardware or prove physical failure-domain independence;
- trust cached or authored validation receipts;
- introduce a database, service, watcher, storage backend, signing system, or
  cloud integration; or
- change emulator, firmware, driver, or array behavior.

## Exit Criteria

Phase 2B is complete when:

- one generic synthetic emission plan produces a canonical sealed bundle;
- re-emission is byte-for-byte path independent;
- the validator rejects every specified integrity and provenance fault;
- two synthetic replicas can be validated through distinct declared failure
  domains;
- only the production validator path can populate the private evidence audit;
- a fully closed synthetic ledger reaches `is_clean == true` through that path;
- the real NPU1 seed report remains blocked without canonical evidence;
- focused and root library suites pass; and
- the worktree is clean after a closure report records exact RED/GREEN evidence.

## Following Boundary

After Phase 2B closes, the next slice creates the first real canonical bundle
from a fresh, cheap NPU1 campaign. It maps that bundle into the ledger, validates
any replicas deliberately declared for it, and begins replacing the single
historical seed chain with current, promotable evidence.

That following slice is where operational campaign integration begins. Phase
2B stops at the reusable, synthetic, trustworthy apparatus.

## Implementation Authorization

Approval of this design authorizes only this design record. The implementation
plan and code are separate checkpoints. No Phase 2B implementation begins until
Maya reviews the implementation plan and explicitly approves execution.
