# NPU1 Phase 2A Research-Reserve Ledger Kernel -- Design

**Date:** 2026-07-29

**Status:** Approved

**Scope:** Versioned NPU1 ledger schema, one vertical tuple-to-evidence seed,
fail-closed release evaluation, and deterministic generated report

## Purpose

Phase 2A turns the research-reserve rules into the smallest machine-checkable
ledger that can represent the evidence already audited in Phase 1A. It is the
first slice of Phase 2 in
[`2026-07-29-npu1-research-reserve-design.md`](2026-07-29-npu1-research-reserve-design.md).

The slice must prove four things before the catalogue grows:

1. the primary Phoenix tuple can be named without inventing unavailable
   identity;
2. a full-platform inventory entry can reuse the existing AIE2 coverage spine
   without creating another architecture taxonomy;
3. a causal fact and its evidence can remain useful while explicitly failing
   promotion; and
4. the top-level retirement gate cannot become green through missing records,
   prose claims, or hand-authored evidence status.

The canonical capture-bundle emitter and integrity validator are deliberately
the immediately following Phase 2 slice. Phase 2A defines the catalogue side
of that boundary but does not claim that an external bundle has been verified.

## Selected Approach

The ledger is one versioned JSON document whose schema is defined by Rust
types in `xdna-archspec`. Human-readable Markdown is generated from the parsed
and validated records.

This is preferred over two alternatives:

- A record-per-file tree would add directory traversal, ordering, and
  partial-load behavior before the catalogue is large enough to need it.
- Extending `CoverageModel` directly would conflate AIE semantic coverage with
  platform tuples, firmware facts, external evidence, replicas, and offline
  rehearsal.

One document provides atomic review, deterministic loading, and the fewest new
moving parts. It may be split under a later schema version if real merge or
readability pressure appears.

## Ownership and File Layout

The implementation belongs to `xdna-archspec`, which already owns architecture
identity, coverage identity, Serde-backed records, deterministic coverage
artifacts, and the existing semantic release input.

```text
crates/xdna-archspec/src/research_reserve.rs
crates/xdna-archspec/data/research-reserve/npu1.json
docs/coverage/npu1/release-report.md
```

`research_reserve.rs` initially remains one module. The ledger is too small to
justify a module tree.

The JSON file is authoritative. The Markdown report is generated and guarded
by the same staleness-test pattern used by the current files under
`docs/coverage/aie2/`.

## Schema Contract

Rust types are the schema and derive Serde serialization. JSON is the authored
representation, not a second schema. Every persisted record uses
`deny_unknown_fields`, snake-case enum spellings, and an explicit root
`schema_version`.

Version 1 has one root:

```rust
struct ReserveLedger {
    schema_version: u32,
    tuples: Vec<PinnedTuple>,
    inventory: Vec<InventoryEntry>,
    facts: Vec<HardwareFact>,
    evidence: Vec<EvidenceRecord>,
}
```

Stable IDs are plain strings rather than wrapper types. Structural validation
requires an ASCII dotted identifier and the appropriate leading kind:
`tuple.`, `inventory.`, `fact.`, or `evidence.`. Cross-kind wrapper boilerplate
would not improve the JSON trust boundary; reference validation provides the
actual protection.

### Pinned tuple

The first tuple record carries the primary pins already approved by the
umbrella design:

- Phoenix/NPU1 and AIE2 identity;
- PCI device identity `1022:1502`;
- firmware logical name and SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- initial amdxdna driver-surface commit
  `216cefececd74effcd7a88350c71b99f5ef9a215`; and
- references to the independently validated kernel corpus when that corpus is
  frozen for the firmware-array seam.

The record also carries:

- an `Open` or `Complete` tuple-identity state;
- explicit missing identity fields while open;
- an `Open` or `Sealed` inventory-scope state;
- live-attestation evidence references; and
- offline-rehearsal evidence references.

`Open` states are first-class, valid data and always block release. A tuple
becomes `Complete` only by referencing canonical evidence for the exact board,
host kernel/module, XRT, toolchain, workload, relevant parameters, and
reset/power/clock epoch. Those component fields belong in the canonical
campaign manifest designed in the next slice; Phase 2A does not duplicate a
partial manifest inside the ledger.

The inventory scope is explicit because an empty or one-entry inventory cannot
be allowed to look complete. A sealed scope names evidence for every discovery
source used to establish the inventory boundary.

### Inventory entry

Each entry contains:

- stable ID and title;
- discovery/source references;
- applicable tuple IDs;
- zero or more existing coverage-spine domain IDs;
- dependency inventory IDs; and
- a typed disposition.

The disposition is represented as a tagged enum so contradictory combinations
cannot be serialized:

```rust
enum InventoryDisposition {
    Applicable { fact_ids: Vec<String> },
    ProvenNotApplicable { fact_ids: Vec<String> },
    Deferred { reason: String },
    Unknown { reason: String },
}
```

`Applicable` is closed only when every referenced fact is
`RetirementQualified`. `ProvenNotApplicable` requires retirement-qualified
facts proving the exclusion. `Deferred` and `Unknown` always block release.
Both fact-bearing dispositions require at least one fact ID; an entry with no
candidate fact remains `Deferred` or `Unknown`.

Full-platform surfaces receive inventory IDs directly. Optional
`coverage_domain_ids` may name only existing IDs from
`coverage::spine_ids::SPINE_DOMAIN_IDS`. Phase 2A does not create a parallel
subsystem registry. Inventory dependencies express relationships between
full-platform surfaces as those entries are added.

### Hardware fact

A fact contains the complete logical shape required by the umbrella design:

- stable ID and exact causal statement;
- tuple IDs and fact dependencies;
- preconditions and known initial state;
- stimulus and external events;
- expected transition, outputs, ordering, and timing bounds;
- supporting witnesses and controls;
- counterevidence and alternatives ruled out;
- remaining unknowns;
- toolchain or documentation sources;
- emulator implementation and test references; and
- promotion state.

The persisted promotion states are:

```text
Observed -> Derived -> Verified -> Encoded -> RetirementQualified
```

`Contested` is a separate fail-closed state. Promotion is declared in the
ledger for review, but the release evaluator independently checks the
prerequisites for the declared state. Editing the label alone cannot close a
fact.

An applicable fact is retirement-qualified only when it has:

- a non-empty causal contract;
- no remaining unknowns;
- verified supporting witness and control evidence;
- explicit treatment of counterevidence and alternatives;
- implementation and executable test references; and
- no contested dependency.

### Evidence record

The Git-side catalogue record contains:

- stable ID and evidence kind;
- candidate tuple IDs;
- a stable location alias plus relative path;
- checked-in intake or analysis references;
- expected bundle/identity digests when available;
- known provenance gaps;
- retention and redistributability classes;
- expected independent replicas; and
- preservation notes that receive no replica credit.

Absolute paths and parent traversal are rejected. Stable aliases such as
`repo-experiments` and `workspace-experiments` preserve the Phase 1A location
contract without embedding one user's home directory.

Evidence integrity is not trusted merely because JSON says it is valid. The
release evaluator consumes an internal evidence-audit result. Phase 2A has no
production path that can mark an external bundle or replica verified, so the
committed report remains red. The next slice supplies that audit result by
validating canonical manifests, artifacts, and independent replicas.

An in-module test may construct a synthetic audit result to prove that the
release evaluator is capable of reaching green when every prerequisite is
actually supplied. That test-only seam is not a public bypass.

## Initial Vertical Record Chain

Version 1 is seeded with exactly one reviewable chain:

```text
primary Phoenix tuple
  -> firmware command-list execution inventory entry
  -> CHAIN_EXEC lifecycle contract candidate
  -> audited legacy vfio-user Chess witness
```

The inventory entry covers the driver-reachable firmware command-list
lifecycle and references applicable existing domains such as DMA, interrupt,
and binary loading. It remains open because its fact is not
retirement-qualified and because the surrounding platform inventory is not
sealed.

The fact records the candidate contract:

```text
firmware query
  -> context create
  -> host-buffer map
  -> CU configuration
  -> MSG_OP_CHAIN_EXEC_NPU
  -> context MSI-X acknowledgement
  -> context destroy
```

Its state is `Derived`, not `Verified`. Physical-NPU agreement, exact response
payloads, clean lifecycle, timing, repetition, recovery, and complete build
identity remain explicit unknowns.

The evidence record points to:

[`../findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md`](../findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md)

and identifies the external witness through the stable location
`repo-experiments/phoenix-vfio-user/20260729T171244Z-3136359`.
It records the observed metadata fingerprint and checksum-index digest, its
legacy provenance gaps, restricted payload status, and zero verified
independent replicas.

The same-filesystem Btrfs snapshot remains a preservation note only. It is not
serialized as an independent replica.

## Validation and Failure Semantics

Loading and release evaluation are separate operations.

Structural validation rejects:

- unsupported schema versions or unknown fields;
- malformed, duplicate, or wrong-kind IDs;
- dangling tuple, inventory, fact, evidence, or coverage-domain references;
- self-referential or cyclic inventory/fact dependencies;
- absolute or escaping external paths;
- malformed lowercase SHA-256 values;
- empty required text; and
- internally contradictory states.

An incomplete but structurally honest ledger loads successfully. Release
evaluation turns its open work into blockers.

`ReserveLedger::clean_release(tuple_id)` returns a structured
`ReleaseReport`, not a Boolean. The report contains:

- the tuple ID;
- fixed inventory, fact, implementation, evidence, replica, semantic,
  attestation, and rehearsal checks;
- sorted blockers with stable codes, affected record IDs, and dependency
  paths; and
- `is_clean`, derived only from an empty blocker set.

Malformed ledgers and unknown tuple IDs return errors rather than synthetic
blockers. A fact recorded as `Contested` blocks every dependent path.

The current committed report is expected to remain red for at least:

- open tuple identity;
- open inventory scope;
- the derived, non-retirement-qualified command-list fact;
- legacy evidence with no canonical integrity audit;
- no independently verified replicas;
- unresolved implementation, test, and lifecycle requirements;
- the current semantic perishable queue;
- missing live attestation; and
- missing offline rehearsal.

## Existing Semantic Gate

The current `CoverageModel::clean_release()` is renamed
`semantic_provenance_clean()`. Its behavior does not otherwise change: it
continues to report whether the semantic perishable and comprehension queues
are empty.

`ReserveLedger::clean_release(tuple_id)` calls that narrow gate as one input.
Generated AIE2 prose and tests are updated to use the honest name. No
compatibility alias retains the misleading retirement meaning.

## Generated Report

The existing `gen_coverage_artifacts` example also writes
`docs/coverage/npu1/release-report.md`.

The report is rendered solely from the validated ledger and the same
`ReleaseReport` returned by the gate. It includes:

- tuple pins and open identity fields;
- inventory and fact state;
- evidence provenance and replica state;
- each fixed release check;
- every blocker and its dependency path; and
- the final red or green result.

Rendering sorts records and blockers by stable ID/code, so authored JSON order
cannot perturb the output. A unit test compares regeneration with the
committed Markdown and fails on drift.

## TDD and Verification

Implementation begins with focused RED tests for:

1. unsupported schema, unknown fields, duplicate IDs, and dangling references;
2. the explicit open-inventory blocker;
3. a promotion blocker when legacy evidence is cited for `Verified` or later;
4. the exact blocker classes produced by the committed seed;
5. deterministic report rendering and committed-report staleness; and
6. a fully closed synthetic ledger plus test-only verified evidence audit
   producing `is_clean == true`.

The smallest implementation is added behind those tests. No new dependency is
needed: `serde`, `serde_json`, standard collections, and the existing artifact
generator are sufficient.

Required verification is:

```bash
cargo test -p xdna-archspec --lib
cargo run -p xdna-archspec --example gen_coverage_artifacts
git diff --check
cargo test --lib
```

The generated command runs before the final tests so the staleness gate checks
the committed artifact rather than an old report.

## Non-Goals

Phase 2A does not:

- emit or validate canonical capture bundles;
- hash or modify the historical corpus;
- award independent-replica credit;
- run NPU hardware, KVM, vfio-user, QEMU, Halo, or privileged commands;
- fill the complete full-platform inventory;
- promote the legacy Chess witness into NPU1 hardware proof;
- create a database, service, file-watcher, scanner, or storage dependency;
- split the ledger into per-record files;
- add a second AIE2 architecture or subsystem taxonomy; or
- change emulator or firmware behavior.

## Next Boundary

The next Phase 2 slice defines and implements the canonical external bundle
contract:

- deterministic manifest emission;
- worktree- and path-independent identity;
- artifact and manifest checksum validation;
- missing-artifact and tamper rejection;
- independent-replica verification; and
- production evidence-audit results consumed by the release evaluator.

Only that validator may convert an expected catalogue record into verified
reserve state.
