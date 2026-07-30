# NPU1 Phase 1A Historical Corpus Intake -- Design

**Date:** 2026-07-29

**Status:** Approved

**Scope:** Read-only current-tuple freeze, metadata census of the existing
NPU1 research corpus, and deep intake of one legacy firmware witness

## Purpose

Phase 1A establishes what evidence already exists before the research-reserve
program introduces a durable catalogue or resumes firmware implementation.
It is the first slice of Phase 1 in
`2026-07-29-npu1-research-reserve-design.md`.

The work uses a two-pass hybrid:

1. cheaply account for the complete known corpus and freeze the current live
   NPU1 tuple; then
2. deeply intake one high-value firmware capture so the Phase 2 catalogue is
   designed from real evidence rather than an imagined schema.

This phase produces documentation, not inventory software. It does not move,
rewrite, delete, normalize, or otherwise mutate historical evidence.

## Selected Approach

The rejected extremes are:

- building a generic catalogue tool before seeing the legacy data it must
  represent; and
- auditing only firmware captures while leaving preservation hazards elsewhere
  in the corpus invisible.

Instead, Phase 1A traverses both known corpus roots metadata-first, classifies
their campaign families, and then spends content-reading and hashing effort on
one selected witness. Expensive treatment expands to the rest of the corpus in
later Phase 1 work.

No Phase 1A record promotes an old conclusion into a verified hardware fact.
The census and intake report describe evidence quality and missing work; they
do not close ledger entries that do not yet exist.

## Corpus Boundary

Checked-in reports identify source locations through stable aliases:

| Alias | Meaning |
|-------|---------|
| `repo-experiments` | `build/experiments` beneath the xdna-emu checkout used for the scan |
| `workspace-experiments` | `experiments` beneath the containing `npu-work` workspace |

Absolute home-directory paths are operational details and do not appear in the
checked-in identity. The census records the xdna-emu commit and branch used for
the scan so `repo-experiments` has an explicit checkout context.

Both roots are in scope even when a top-level family is ultimately excluded.
For example, BIOS or unrelated DKMS material may receive an
`excluded-with-reason` disposition, but it may not disappear silently.

The first pass does not follow symlinks or cross into mounted targets. It never
awards replica credit: a symlink, hardlink, reflink, or ordinary copy found
during Phase 1A is merely another observed location until the later reserve
process proves independent storage.

## Pass 1 -- Live Tuple and Corpus Census

### Live tuple freeze

The census report begins with a read-only snapshot of the current NPU1
hardware and software tuple. Values come from live files, package metadata,
sysfs, and source-control state rather than from a historical capture.

The snapshot covers the applicable fields already defined by the umbrella
research-reserve design:

- PCI device and subsystem identity;
- firmware name and hash;
- host kernel and loaded amdxdna module identity;
- driver source revision, module hash, and relevant parameters;
- XRT package and library versions;
- aie-rt, mlir-aie, llvm-aie, and register-database revisions;
- xdna-emu branch, commit, and dirty state; and
- relevant IOMMU, address-mode, reset, power, and clock state when exposed
  safely.

Unavailable values remain `unknown` with the attempted source recorded. A live
kernel or package version is not inferred from a nearby source checkout. This
snapshot launches no NPU workload and performs no privileged operation.

### Metadata traversal

Every filesystem entry is enumerated without reading bulk contents. Entries
are aggregated into human-reviewed campaign families. A family normally
corresponds to a top-level directory or an evident timestamped-campaign parent;
heterogeneous material remains `mixed` unless the boundary is clear.

Each family records:

- root alias and relative path;
- disposition and short rationale;
- regular-file, directory, and symlink counts;
- apparent and allocated size, with measurement semantics stated;
- oldest and newest modification times as organizational clues only;
- unreadable entries and broken links;
- link-count or size discrepancies that may indicate shared, sparse, or
  otherwise non-independent storage;
- counts of recognizable provenance markers such as `tuple.txt`,
  `manifest.json`, and checksum files; and
- a short sample of artifact kinds derived from names and small metadata files.

The four dispositions are:

| Disposition | Meaning |
|-------------|---------|
| `npu1-relevant` | The family contains evidence about the target platform or its emulator contract |
| `mixed` | Relevant and unrelated material cannot yet be separated safely |
| `excluded-with-reason` | The family is outside NPU1 preservation scope, with the reason recorded |
| `unknown` | Available metadata is insufficient for a safe classification |

Classification is conservative. A name or timestamp may suggest relevance but
cannot establish provenance.

### Pass 1 output and exit

Pass 1 produces:

`docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md`

The report includes the live tuple, root definitions, scan commands, family
table, preservation hazards, and the selected deep-intake candidate. It does
not include a per-file catalogue.

Pass 1 closes when:

- every top-level child of both roots appears exactly once;
- every child has one of the four dispositions;
- family counts and sizes reconcile with the metadata traversal;
- unreadable, unstable, linked, sparse, or missing material is visible;
- the current tuple distinguishes verified values from unknowns; and
- one high-value firmware witness is nominated for Pass 2.

## Pass 2 -- Legacy Firmware Witness Intake

### Selected exemplar

The first exemplar is:

```text
repo-experiments/phoenix-vfio-user/20260729T171244Z-3136359
```

It is a roughly 187 MiB legacy capture of the successful frozen Chess
command-list path. Its guest log records the complete `2..=65` result sequence
and `PHOENIX_FROZEN_PASS chess`. The tuple identifies Chess compilation and
command-list execution, making it the closest existing single witness to the
normal driver-reachable firmware path.

A current audit also found successful Peano captures, including:

```text
repo-experiments/phoenix-vfio-user/20260729T171042Z-3129577
```

That capture records `PHOENIX_FROZEN_PASS peano` with direct execution. Peano
is therefore not an unrun gate in current local evidence. It is a distinct
stimulus and becomes the next intake candidate rather than being merged with
the Chess command-list bundle.

Both successful paths retain guest-kernel dma-buf and recursive-locking
warnings. Phase 1A reports functional completion and lifecycle cleanliness as
separate claims.

### Immutability and integrity

The legacy directory remains byte-for-byte untouched. Phase 1A does not add a
manifest, checksum file, normalized tuple, or annotation inside it.

The intake:

- verifies every checksum already recorded by the capture;
- hashes each readable regular artifact contained by this exemplar without
  following symlinks;
- records missing or changed external references rather than substituting a
  current file;
- distinguishes duplicate paths from independent evidence; and
- keeps proprietary or non-redistributable payloads outside Git.

New integrity results live in the intake report. They describe the legacy
capture; they do not retrofit it into the canonical bundle contract. The
report embeds the deterministic root-relative checksum listing as an appendix
instead of retaining only an aggregate digest. This preserves per-file
integrity without adding a sidecar to the legacy directory or a fourth Phase
1A artifact.

### Provenance and claim audit

Every material statement is classified as:

- **Observed:** directly present in a named capture artifact;
- **Derived:** a bounded interpretation with its inputs and reasoning named; or
- **Unknown:** not recoverable from the current capture.

The report answers:

- what executable and input artifacts were used;
- what firmware, driver, kernel, runtime, compiler, and execution mode were
  recorded;
- what command launched the run;
- what output, completion, interrupt, and teardown observations exist;
- what warnings, anomalies, and recovery actions occurred;
- which referenced artifacts are self-contained versus externally located;
- what may be redistributed;
- what candidate hardware facts the capture could support; and
- what the capture does not prove.

Each outcome claim links to an exact relative artifact and location. Prior
interpretation is quoted only as prior interpretation; it is not upgraded by
repetition in the new report.

### Pass 2 output and exit

Pass 2 produces:

`docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md`

Pass 2 closes when the report states precisely:

- the stable platform and stimulus fields that can be recovered;
- every unknown or contradiction;
- the integrity status of the contained and referenced artifacts;
- the directly supported functional result;
- why the run is not yet a lifecycle-clean or retirement-qualified proof;
- the candidate facts and emulator contracts it may inform; and
- the exact missing evidence a future canonical rerun must capture.

Phase 1A does not perform that rerun. Exhausting the existing evidence first
keeps the next hardware campaign purposeful and small.

## Data Flow

```text
live read-only platform sources
  -> current tuple section

repo-experiments + workspace-experiments
  -> metadata-only traversal
  -> human-reviewed family classification
  -> corpus census and preservation hazards
  -> selected exemplar

immutable legacy exemplar
  -> integrity verification
  -> observation / derivation / unknown split
  -> claim and lifecycle audit
  -> missing-evidence list
  -> input to the Phase 2 catalogue design
```

No downstream step writes back into either corpus root.

## Failure Semantics

Evidence handling fails closed:

- a missing or unreadable root blocks Pass 1 rather than becoming an empty
  result;
- an ambiguous family is `unknown` or `mixed`, never optimistically relevant
  or excluded;
- two metadata aggregations that disagree mark the affected family unstable
  and require a rescan;
- an existing checksum mismatch is a contradiction and is never repaired by
  updating the recorded hash;
- an absent externally referenced file remains absent even if a similarly
  named current file exists;
- broken links and non-independent storage are preservation hazards;
- capture warnings remain attached to every dependent claim; and
- no old `verified`, `pass`, or equivalent label promotes evidence by itself.

If an uncertainty cannot be resolved read-only, the report identifies the
smallest future NPU rerun or preservation action that can resolve it. Phase 1A
does not silently expand into that work.

## Durable Artifacts

The complete Phase 1A document set is:

```text
docs/superpowers/specs/
  2026-07-29-npu1-phase1a-historical-corpus-intake-design.md

docs/superpowers/findings/
  2026-07-29-npu1-historical-corpus-census.md
  2026-07-29-phoenix-vfio-user-cmdlist-intake.md
```

Only the design exists before implementation. The two findings are created in
order, each as its own reviewable change.

Raw captures, bulk scan output, proprietary binaries, generated catalogues,
and machine-specific absolute paths are not committed. The reports retain the
exact commands and relative aliases needed to audit their conclusions.

## Validation

Phase 1A is documentation-only, but its claims receive executable checks.

### Census checks

- Enumerate top-level children independently and prove that the family table
  accounts for each exactly once.
- Recompute family entry counts and apparent/allocated sizes and reconcile them
  with the report.
- Repeat the metadata aggregation and mark any changing family unstable.
- Verify every live-tuple value against its named current source.
- Search the checked-in report for leaked home-directory paths.

### Exemplar checks

- Verify the legacy tuple's recorded hashes without changing its files.
- Compute a deterministic checksum listing for all readable regular artifacts
  in the exemplar and reproduce the appendix from it.
- Mechanically confirm that the guest log contains exactly the expected
  ordered `2..=65` outputs and the Chess pass marker.
- Account separately for firmware completion, interrupt publication, driver
  teardown, dma-buf warnings, and recursive-locking warnings.
- Prove that each report claim names an observed artifact or is labeled
  `Derived` or `Unknown`.

### Repository checks

- `git diff --check` passes.
- The implementation diff contains only the two approved findings.
- The ordinary required `cargo test --lib` regression gate remains green.
- Neither corpus root contains a Phase 1A-created file or modification.

## Non-Goals

Phase 1A does not:

- define the Phase 2 catalogue schema;
- add an inventory scanner, database, service, or storage dependency;
- reorganize or deduplicate the corpus;
- create or validate independent replicas;
- prune working captures;
- promote facts into the research-reserve ledger;
- change emulator or firmware behavior;
- rerun an NPU campaign; or
- claim that successful guest output proves warning-free lifecycle behavior.

## Next Boundary

After Maya reviews this design, implementation planning divides the work into
two sequential, independently reviewable slices:

1. live tuple plus full shallow census; and
2. deep intake of the selected Chess command-list witness.

Only after both reports are reviewed does Phase 2 design the minimum catalogue
representation around the evidence fields that proved necessary.
