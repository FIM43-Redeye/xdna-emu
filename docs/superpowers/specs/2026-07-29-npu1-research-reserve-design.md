# NPU1 Research Reserve and Retirement Gate -- Design

**Date:** 2026-07-29

**Status:** Approved

**Scope:** Phoenix/Hawk Point NPU1 preservation, characterization, and faithful
emulation

## Purpose

The NPU1 hardware in the current motherboard is a temporary research asset.
Replacing the board is safe only after the behaviors that matter for faithful
emulation have been converted into causal, implementable, independently
auditable facts.

The durable product is not a large collection of captures. It is an executable
description of the NPU:

```text
hardware observation
  -> scoped hardware fact
  -> deterministic contract
  -> emulator behavior and tests
  -> retirement-qualified proof
```

Raw evidence remains necessary provenance, but it becomes archival after its
information has been derived, encoded, and validated.

This design defines the umbrella research program and the meaning of
`clean_release()`. It deliberately does not contain one implementation plan.
Each phase and each coherent subsystem slice receives its own short design and
plan.

## Relationship to Existing Designs

The firmware-equivalence contract in
[`2026-07-26-phoenix-firmware-equivalence-goal-design.md`](2026-07-26-phoenix-firmware-equivalence-goal-design.md)
remains authoritative for the Phoenix management-firmware boundary. In
particular:

- the emulator models the NPU protocol, not a driver implementation;
- the primary target is the unmodified Phoenix firmware image;
- the definitive firmware case is the unmodified driver talking to simulated
  firmware, which drives the shared array emulator and completes an
  independently validated kernel; and
- lifecycle, failure, recovery, packaging, preemption, and timing remain part
  of full equivalence even when implemented as later slices.

This research-reserve design expands the preservation boundary to the complete
NPU1 platform and supplies the evidence, inventory, promotion, retention, and
retirement rules needed before physical hardware is surrendered.

This design also **supersedes the retirement meaning** assigned to the current
semantic-only `CoverageModel::clean_release()` by
[`2026-05-15-two-axis-coverage-provenance-design.md`](2026-05-15-two-axis-coverage-provenance-design.md).
Empty semantic perishable and comprehension queues are necessary, but they are
not proof that NPU1 is safe to retire.

## Terms

**Pinned platform tuple**
: The complete identity under which a fact is valid: physical NPU and board
  identity, firmware, host kernel and driver, XRT, compiler/toolchain inputs,
  workload artifacts, relevant configuration, and reset/power/clock state.

**Capture campaign**
: One stable platform and stimulus tuple with one or more repetitions. A tuple
  or stimulus change starts a new campaign.

**Hardware fact**
: A precise, scoped causal claim derived from authoritative sources and/or
  controlled silicon observation.

**Executable contract**
: A statement of known initial state plus explicit stimulus and external event
  schedule, followed by the expected architectural transition, observables,
  ordering, and timing constraints.

**Research reserve**
: The tracked fact and evidence catalogue plus external witness
  artifacts, source/tool versions, and recipes required to audit or reproduce
  the facts after NPU1 is unavailable.

**Retirement-qualified**
: A fact that is causally understood, implemented, test-covered, differentially
  validated, and backed by intact witness evidence.

## Correctness and Authority

The repository's existing source hierarchy remains in force:

1. Open-source toolchain sources (`aie-rt`, `llvm-aie`, `mlir-aie`, AM025
   register data) are the preferred derivation source.
2. Owned NPU1 silicon is ground truth for what the pinned platform does.
3. `aietools` is a read-only behavioral reference where open sources are
   incomplete.
4. AM020/AM025 prose fills remaining documented gaps.

The toolchain and silicon have different roles. The toolchain supplies the
structure, intended programming model, constants, and stimuli. Silicon decides
the actual behavior of the pinned NPU1 tuple. A disagreement is investigated
and recorded; neither source silently overrides the other. If a toolchain
defect is confirmed, the emulator reproduces observed NPU1 behavior for the
pinned tuple and the contradiction remains explicit.

The driver is a stimulus producer and surface-discovery instrument. It never
defines simulated behavior. A newer or development driver may reveal a missed
surface, but the emulator continues accepting the hardware protocol itself.

Undocumented operations with no legitimate, hardware-verifiable contract are
outside the acceptance surface. Development-only operations are inventoried
because they may become important; before retirement each is either
characterized or proven inapplicable to the declared NPU1 scope.
An undocumented behavior reached by a legitimate stimulus or discovered by a
valid generative campaign becomes an explicit hardware-discovered ledger entry;
excluding blind, unreachable opcode space does not permit ignoring observed
silicon behavior.

## Scope and Tuple Order

The first gate is the pinned Phoenix/NPU1 primary tuple already used by the
firmware program:

- firmware:
  `/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`;
- firmware SHA-256:
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- initial driver-surface corpus:
  `amdxdna-driver` commit
  `216cefececd74effcd7a88350c71b99f5ef9a215`; and
- independently validated NPU1/AIE2 kernel corpus used at the firmware-array
  seam.

The complete tuple is richer than these initial pins. A campaign manifest also
records exact board/device identity, host kernel and module hashes, XRT
packages and library hashes, all compiler/toolchain commits, exact workload
artifacts, relevant driver parameters, and reset/power/clock epoch state.
Unavailable identity fields are recorded as explicitly unavailable rather than
invented.

The primary tuple is closed first. Only then is the authoritative older
Phoenix firmware set frozen and run through its applicable contract. A gate
result belongs to one tuple and never transfers automatically to another
firmware version, board, driver corpus, or NPU generation.

Management firmware remains the current implementation priority because it is
the gating path into the simulated array. Array campaigns may still collect
perishable evidence while the hardware is present, and full NPU1 retirement
eventually requires the declared array contract too. Kernel cycle-accuracy work
does not interrupt the firmware sequence unless it is needed to distinguish a
firmware-array seam behavior.

AIE2P/XDNA2 is not part of this reserve. It receives its own inventory and
evidence when work begins on that architecture.

## Deterministic Modeling Rule

The emulator does not model output distributions.

If repeated hardware runs differ, the difference is evidence of an omitted
initial state, asynchronous input, clock phase, arbitration state, firmware
interaction, or other causal variable. The research task is to expose that
variable and make it an explicit state or external event.

An emulator run is deterministic given:

- the complete modeled initial state;
- the ordered host and device stimuli;
- the explicit external-event schedule; and
- the pinned platform contract.

Random scheduling or fitted outcome probabilities may be used only as
diagnostic tools. They are never accepted as hardware semantics. Unresolved
variation blocks retirement for the affected surface.

Timing follows the same rule. Directly observable hardware timing is compared
directly. Polls, counters, interrupts, and correlated side effects constrain
internal timing where direct observation is impossible. Unobservable
microcycles remain unknown rather than being declared exact.

## Inventory and Ledger Architecture

The existing generated AIE2 coverage infrastructure is the starting spine, not
a taxonomy to replace. The research ledger references its stable architecture,
subsystem, semantic, and register identities and adds the full-platform
surfaces that are not AIE-array semantics:

- firmware boot and execution architecture;
- PSP/SMU envelope behavior required by the NPU;
- BARs, mailboxes, management DMA, and interrupts;
- host-memory and IOMMU/physical-address contracts;
- firmware-array configuration, execution, status, and completion seams;
- reset, power, timeout, teardown, and recovery;
- binary packaging and loading; and
- observable timing and cross-subsystem ordering.

The ledger is schema-first and machine-checkable. Its logical records are:

### Inventory entry

- stable ID;
- generated source or discovery source;
- applicable tuple set;
- subsystem and dependency IDs;
- disposition:
  `Applicable`, `ProvenNotApplicable`, `Deferred`, or `Unknown`; and
- fact IDs that close an applicable entry.

`Deferred` and `Unknown` always block release. `ProvenNotApplicable` requires
evidence; it is not a waiver.

### Hardware fact

- stable ID and exact causal statement;
- tuple scope and preconditions;
- known initial state, stimulus, and external events;
- expected transition, outputs, ordering, and timing bounds;
- supporting witnesses and controls;
- counterevidence and alternatives ruled out;
- remaining unknowns;
- source/toolchain references;
- emulator implementation and test references; and
- promotion status.

The promotion states are:

```text
Observed -> Derived -> Verified -> Encoded -> RetirementQualified
```

`Contested` is a fail-closed state that may replace any promoted state when new
evidence disagrees.

### Release report

- pinned tuple ID;
- inventory, fact, implementation, evidence, replica, and rehearsal checks;
- explicit blockers with dependency paths; and
- an `is_clean` result derived from the absence of blockers.

Human-readable Markdown is generated from the same authoritative records. The
precise code and storage location for the new full-platform ledger belongs to
the Phase 2 design; it must reuse existing coverage identities and must not
fork a second AIE2 taxonomy.

## Existing Semantic Gate

The current `CoverageModel::clean_release()` checks only whether the AIE2
semantic perishable and comprehension queues are empty. It does not include
the subsystem implementation queue, firmware, lifecycle, cross-subsystem
behavior, external evidence integrity, or offline rehearsal.

It will be renamed `semantic_provenance_clean()` and used as one input to the
full-platform report. The top-level
`clean_release(tuple_id)` owns the retirement meaning.

At the time of this design:

- the generated comprehension-gap queue is empty;
- the generated implementation-gap queue is empty; and
- the generated perishable queue is **not** empty: vector behavior remains
  `AietoolsModeled/Unverified`, and side effects remain
  `DocSpecified/Unverified`.

Therefore even the current narrow semantic gate is red. None of the existing
coverage reports establishes retirement readiness.

An existing `Accepted` semantic verdict is not automatically sufficient for
the retirement gate. The full ledger must map it either to a proven
inapplicable entry or to a retirement-qualified fact. “Good enough” is not a
release disposition.

## Program Phases

Each phase has its own design, plan, RED/GREEN evidence, and closure report.
Later phases may add work to earlier ledgers, but they do not weaken their exit
criteria.

### Phase 0 -- Baseline integrity

Make the development and validation environment trustworthy before changing
the research gate.

The current diagnosis is an environment-contract mismatch, not broken
mlir-aie Python bindings or a broken compiler:

- the firmware worktree's build fallback searched sibling paths beneath
  `.worktrees/`, where `mlir-aie` and `llvm-aie` do not exist;
- the Codex shell did not inherit
  `/home/triple/npu-work/mlir-aie/install/python`;
- `AIE_RT_PATH` must resolve to
  `/home/triple/npu-work/aie-rt/driver/src`; and
- Claude's `BASH_ENV` autoactivation is provider-specific and therefore not a
  portable repository contract.

With explicit canonical component roots, PATH, and PYTHONPATH, the targeted
coverage suite passed 55 tests with no failures. Phase 0 turns that successful
manual invocation into a provider-neutral, worktree-independent path and
leaves a failing test for regressions.

Exit requires:

- canonical toolchain roots resolve identically from the main checkout and an
  isolated worktree;
- Python binding imports and generated architecture inputs are checked
  explicitly;
- the exact targeted coverage gate passes without hidden Claude-only shell
  activation; and
- the ordinary required local regression commands remain green.

Phase 0 does not redesign the ledger. It establishes that later failures mean
what they claim to mean.

### Phase 1 -- Historical evidence intake

Inventory the existing approximately 5.2 GiB research corpus without treating
old conclusions as current truth.

For each historical artifact:

- identify its tuple and stimulus as far as the artifact permits;
- preserve missing fields as unknown;
- distinguish raw observation from prior interpretation;
- check hashes, readability, provenance, and redistributability;
- connect it to candidate ledger entries; and
- rerun the corresponding cheap NPU campaign when the current hardware can
  resolve uncertainty.

Historical evidence may seed hypotheses and witnesses. It cannot become
retirement-qualified merely because an old report called it verified.

### Phase 2 -- Research-reserve ledger

Establish the versioned schemas, full-platform inventory, fact records,
evidence catalogue, dependency links, generated reports, and top-level release
report.

The Phase 2 design chooses the smallest code owner and file layout that can
reuse the current coverage spine. It must not introduce a database, service,
or competing architecture taxonomy.

### Phase 3 -- Deterministic subsystem campaigns

Close the structured inventory one coherent subsystem or behavioral edge at a
time. Campaigns prioritize information that cannot be recovered after losing
NPU1, not whichever emulator code is easiest to modify.

Normal NPU runs are cheap and should use full safe matrices with abundant
repetition. Sampling is reserved for probes with demonstrated wedge, recovery,
thermal, or data-volume risk. The expensive emulator and compilation portions
are batched independently from cheap silicon observations.

### Phase 4 -- Cross-subsystem causal campaigns

Exercise interactions that component tests cannot prove: producer/consumer
paths, event and interrupt delivery, firmware-array command completion,
address translation, ordering, reset, power, error propagation, and recovery.

The definitive firmware path remains:

```text
unmodified driver
  -> simulated hardware protocol
  -> unmodified firmware
  -> shared array emulator
  -> independently validated kernel
  -> firmware completion
  -> unmodified driver
```

### Phase 5 -- Generative differential exploration

After structured coverage is near-closed, generate valid programs,
transactions, state combinations, and orderings from authoritative schemas and
run them against hardware and the emulator.

The generator is a gap finder, not a replacement for the structured spine.
Every discovered mismatch becomes a normal ledger item and deterministic
campaign.

### Phase 6 -- Retirement rehearsal

Run the final live NPU1 attestation, seal its witnesses, preserve them through
the ordinary backup policy, validate any replicas explicitly declared by the
ledger, then reproduce the complete gate in an environment with no NPU access.

The board is releasable only after both the live and offline halves pass.

## Capture Bundle Contract

One immutable external bundle represents one capture campaign under a stable
platform and stimulus tuple. Repetitions remain separate run records inside
the campaign. A change to any behaviorally relevant tuple or stimulus field
starts a new bundle.

A bundle contains:

```text
manifest.json
SHA256SUMS
raw/
derived/
```

The versioned `manifest.json` records:

### Identity

- schema version and stable bundle ID;
- campaign and ledger IDs;
- run ordinal and repetition;
- reset, power, and clock epoch;
- current or legacy provenance; and
- risk class and outcome.

### Platform

- physical device, subsystem, board, and available unique identity;
- firmware version and hashes;
- host kernel, driver commit, source, module hash, and parameters;
- XRT package versions and relevant library hashes;
- `aie-rt`, `mlir-aie`, `llvm-aie`, register-database, and other toolchain
  revisions;
- compiler and execution mode; and
- relevant clock, power, reset, IOMMU, and address-mode state.

### Stimulus

- exact command and behaviorally relevant environment;
- source and build-recipe identity;
- executable, XCLBIN, ELF, PDI, CDO, configuration, and input hashes; and
- explicit initial state and external-event schedule where known.

### Observations

- program output and status;
- register and memory observations;
- transaction and trace artifacts;
- timing anchors and bounds;
- errors, wedges, recovery actions, and teardown result; and
- expected controls.

### Artifact inventory

Each artifact has a relative path, byte size, SHA-256, semantic kind, and
redistribution/privacy classification. Absolute machine paths are forbidden
from the stable identity.

### Analysis provenance

Derived files identify the source bundle digest, analysis command, and exact
analysis-tool revision. An analysis correction creates a new derived revision;
it never mutates raw evidence.

The existing Phoenix vfio-user `tuple.txt` files already capture much of the
platform, workload, and artifact identity. Their facts seed the schema, while
their free-form, absolute-path-heavy representation is replaced by relative,
versioned fields.

The canonical bundle and proprietary payloads live outside Git. Git contains a
small catalogue record with the stable bundle ID, manifest digest, mapped
inventory/fact IDs, provenance state, and replica status. The entire sanitized
manifest may be committed only if it contains no private payload or
machine-specific path, but duplicating it is not required.

JSON, SHA-256, and ordinary files are sufficient. Git LFS, DVC, git-annex, a
database, and a custom content-addressed service are out of scope.

The bundle schema tests must prove deterministic manifest emission,
worktree/path independence, checksum tamper rejection, missing-artifact
failure, and that incomplete legacy provenance cannot promote a fact to a
verified state.

## Evidence Promotion and Retention

A capture does not prove a generalized fact by itself. Promotion requires a
precise causal statement, controls, scope, alternatives considered, and an
executable expected transition.

The three retention classes are:

### Working captures

Bulk exploratory repetitions used while deriving a fact. They may be pruned
after campaign closure if:

- every unique outcome and anomaly is preserved;
- per-run manifests, hashes, and structured results remain;
- the fact is retirement-qualified; and
- the retention decision is recorded.

### Witness captures

The minimal sufficient raw evidence for every verified fact, boundary case,
counterexample, anomaly, and relevant toolchain disagreement. Witnesses remain
in the research reserve permanently and are protected by the ordinary backup
policy. Any replicas declared by the ledger must validate independently.

### Implementation fixtures

Small redistributable excerpts or synthesized equivalents committed with tests
where useful. A synthesized fixture is identified as synthetic and links back
to the hardware fact; it never masquerades as raw hardware output.

Evidence showing unresolved variation is never pruned. After hardware
retirement the final witness is the only way to audit a mistaken derivation
without allowing the implementation to prove itself.

## Contradiction and Failure Semantics

Evidence handling fails closed.

A contradiction changes the affected fact to `Contested` and blocks every
dependent fact, implementation claim, and release check. Existing code is not
silently reverted, but the release report identifies that it relies on
contested behavior.

Repeated differing outcomes open a hidden-state investigation. They do not
authorize a probabilistic emulator model.

Experimental failures have distinct meanings:

- **Infrastructure failure:** the experiment did not validly reach the
  intended boundary and contributes no behavioral evidence.
- **Provenance failure:** a result exists, but its tuple or stimulus cannot be
  established; it is a lead, not promotable evidence.
- **Semantic mismatch:** a valid run disagrees with the current fact or
  emulator and opens a fidelity blocker.
- **Device fault or wedge:** behavioral evidence only after the stimulus is
  confirmed valid and device behavior is separated from host instability.
- **Intentional rejection:** a deliberately invalid or unsupported request may
  establish the hardware's rejection contract when the request and response
  are controlled.

An unsuccessful run is never interpreted as absence of behavior merely because
it failed.

## `clean_release()` Contract

`clean_release(tuple_id)` answers one question:

> Can this pinned NPU1 platform be surrendered without losing information
> required for faithful emulation?

It passes only when all of the following are true.

### Inventory closure

Every generated, documented, firmware/driver-exposed, and hardware-discovered
inventory entry has a disposition. Every applicable entry maps to
retirement-qualified facts. `Deferred`, `Unknown`, unexplained variation, and
unresolved contradiction are blockers.

### Fact closure

Every applicable fact is causal, scoped, controlled, witnessed, and reduced to
an executable contract. A report or capture without an implementable
transition is not closed.

### Implementation closure

Every fact maps to deterministic emulator behavior and tests. Required
component, firmware-array seam, lifecycle, recovery, timing, and true
end-to-end differentials pass for the tuple.

### Reserve closure

Every required witness, tool, source, firmware image, manifest, and recipe is
readable, hash-valid, licensed appropriately for its storage location, and
present at its canonical location. Every explicitly declared replica validates
independently.

### Live attestation

Immediately before physical retirement, the complete pinned hardware matrix is
rerun and sealed. It contains no unexplained emulator-versus-silicon
divergence.

### Offline rehearsal

With NPU access unavailable, a clean environment can verify the reserve,
rebuild the required software, run the emulator gates, and reproduce the
validated driver-to-kernel path without relying on unrecorded host paths or
state.

The gate returns a structured blocker report, not merely a Boolean. There are
no discretionary waivers. A genuinely inapplicable entry may close only with
evidence demonstrating why it does not belong to the pinned contract.

“100%” is bounded by the explicit generated inventory and pinned tuple. The
project does not claim knowledge of metaphysically undiscoverable behavior.
It does claim that every surface exposed by the open toolchain, firmware,
driver, documentation, and deliberate hardware exploration is accounted for.

For multiple firmware tuples, each tuple receives its own report. A board-level
retirement decision aggregates the explicitly frozen required tuple set; one
green firmware image does not silently certify another.

## Test and Proof Structure

Every implementation slice follows test-driven development:

1. write the smallest test or probe that fails on the missing fact;
2. derive the behavior from authoritative sources and controlled hardware;
3. implement the smallest shared behavior that explains the evidence;
4. pass focused unit and integration tests;
5. pass the relevant hardware/emulator differential;
6. update the fact, witnesses, generated reports, and fidelity-gap registry;
   and
7. run the required local regression gate.

Proof layers remain complementary:

- **Unit/state-machine tests** make individual transitions deterministic.
- **Component differentials** replay exact stimuli and isolate a subsystem.
- **Firmware-array seam tests** compare configuration, launch, status,
  completion, and interrupt transactions.
- **True end-to-end tests** exercise the unmodified driver, firmware, shared
  array, and validated kernel.
- **Lifecycle campaigns** cover error, timeout, reset, power, teardown,
  preemption, recovery, and repeated operation.
- **Timing campaigns** constrain directly observable and indirectly inferable
  timing.
- **Generative differentials** search for inventory omissions only after the
  structured baseline is nearly closed.

Fixture passes alone do not qualify a fact unless the fixture would fail under
the competing behavior the campaign is intended to distinguish.

## Work-Unit Decomposition

The umbrella design defines the mission, invariants, evidence contract, phase
ordering, and retirement gate. It is not an implementation plan.

Each independently finishable work unit receives a short design and plan
covering one coherent ledger slice, normally one behavioral edge or tightly
coupled subsystem contract. It defines:

- exact inventory and fact IDs;
- toolchain derivation sources and competing hypotheses;
- RED tests or probes;
- the hardware matrix, controls, risk limits, and recovery procedure;
- the deterministic contract and emulator ownership boundary;
- promotion and closure conditions; and
- required witnesses, documentation, and gate updates.

A work unit is not complete with only captures, only analysis, or only code. It
closes when the behavior is derived, encoded, differentially validated,
documented, and represented correctly in the reserve.

If a slice discovers a larger dependency, the dependency is added to the
ledger and its scope is discussed explicitly. The current plan does not
silently grow.

The ledger holds the long horizon. Plans remain small enough to execute,
review, verify, and commit without hiding unrelated correctness decisions.

## Operational Safety

Routine NPU runs are cheap and should not be avoided merely to save time.
Hardware matrices may use many repetitions when they can expose hidden state
or establish a deterministic invariant.

Campaigns still classify risk:

- **Normal:** ordinary valid workloads and read-only observations; run freely.
- **Controlled:** malformed, timeout, reset, power, or concurrency probes with
  bounded execution, retained logs, and an established recovery path.
- **High-risk:** a demonstrated wedge or system-stability hazard; requires an
  approved campaign design and the escalation chain in
  [`docs/operations.md`](../../operations.md).

Risk controls change how a campaign is executed, never what result the
emulator is allowed to invent.

## Non-Goals

- Committing the approximately 5.2 GiB raw corpus to Git in one change.
- Treating raw capture volume as a coverage metric.
- Modeling empirical distributions instead of their causes.
- Allowing the current driver to influence simulated behavior.
- Emulating PSP or SMU behavior unrelated to operating NPU1.
- Replacing the existing generated AIE2 coverage taxonomy.
- Building a database, evidence service, custom object store, or generalized
  workflow engine.
- Claiming direct management-processor cycle equivalence where no observable
  oracle exists.
- AIE2P/XDNA2 preservation within the NPU1 gate.
- One implementation plan for the entire program.

## Immediate Next Design

After written-spec review, the next bounded design is Phase 0 baseline
integrity: a provider-neutral, worktree-independent toolchain and Python
binding contract with an explicit RED/GREEN validation path.

Only after Phase 0 closes should the historical corpus intake and full ledger
implementation begin.
