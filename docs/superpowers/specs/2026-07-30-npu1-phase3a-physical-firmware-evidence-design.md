# NPU1 Phase 3A Physical Firmware Evidence -- Design

**Date:** 2026-07-30

**Status:** Approved

**Scope:** The first fresh canonical physical-NPU1 firmware evidence pair,
fixture-backed bundle graphs, ledger admission, and a separately authorized
50+50 repetition campaign

## Purpose

Phase 2B established a trustworthy, device-neutral path from typed local
capture metadata to immutable external evidence:

```text
typed emission plan
  -> canonical sealed bundle
  -> in-process validation
  -> private EvidenceAudit
  -> research-reserve release report
```

It deliberately stopped at synthetic evidence. Phase 3A takes the first real
step across that boundary. It runs one frozen workload against owned Phoenix
NPU1 silicon through two driver-reachable firmware execution envelopes:

- command-list execution through `CHAIN_EXEC_NPU`; and
- direct execution through `EXEC_DPU`.

The two runs use the same NPU program, host oracle, firmware, driver epoch,
runtime, inputs, and expected array-visible result. Their only intended
treatment difference is the driver's `force_cmdlist` selection.

The first deliverable is one reviewed vertical pair, not a statistical claim.
Only after that pair is accepted may the same machinery launch a bounded,
randomized 50+50 repetition campaign. This order separates basic contract
correctness from later resilience and determinism work.

## Relationship to Existing Designs

This design is subordinate to:

- [`2026-07-26-phoenix-firmware-equivalence-goal-design.md`](2026-07-26-phoenix-firmware-equivalence-goal-design.md),
  which defines the eventual driver-reachable firmware-equivalence contract;
- [`2026-07-29-npu1-research-reserve-design.md`](2026-07-29-npu1-research-reserve-design.md),
  which defines evidence, promotion, and NPU1 retirement; and
- [`2026-07-30-npu1-phase2b-canonical-capture-bundle-design.md`](2026-07-30-npu1-phase2b-canonical-capture-bundle-design.md),
  which defines canonical bundle identity, validation, replicas, and the
  private evidence-audit handoff.

The open-source toolchain still defines structure and intended programming
behavior. Physical NPU1 silicon decides what the pinned tuple actually does.
The driver remains a stimulus producer and surface-discovery instrument; its
implementation does not define emulator semantics.

Phase 3A does not claim general firmware equivalence. It creates the first
fresh physical anchor from which that larger campaign can proceed.

## Selected Approach

Phase 3A uses two existing boundaries and one small campaign-specific wrapper:

1. `xdna-archspec::capture_bundle` remains the schema, emitter, hashing, graph
   validation, and opaque-validation owner.
2. `research_reserve` remains the only owner of the private `EvidenceAudit` and
   ledger-credit decision.
3. A thin Python standard-library campaign tool orchestrates the frozen
   `test.exe`, privileged capture transaction, existing amdxdna tracepoints,
   dynamic-debug evidence, canonical emission, and validation commands.

The wrapper is campaign-specific because the exact workload oracle, firmware
lifecycle, and safe-stop rules are specific. Durable schema and trust logic
remain in Rust. A Rust campaign framework, scheduler, storage service, or
generic hardware laboratory abstraction would add machinery before a second
stable consumer exists.

The existing `tools/multirun-trace-campaign.py` is not reused as the executor.
It captures array trace for K-sweep workloads but does not preserve the exact
host-output or management-firmware lifecycle contract required here.

The event-discovery and cleanup lessons in `tools/amdxdna-trace.sh` are reused,
but Phase 3A uses a dedicated tracefs instance rather than mutating the global
trace ring.

## Pinned Vertical Pair

### Platform tuple

The target remains:

- architecture: AIE2;
- driver platform: NPU1;
- PCI device: AMD `1022:1502`;
- firmware logical path: `amdnpu/1502_00/npu.dev.sbin`;
- firmware SHA-256:
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
  and
- driver protocol surface:
  `amdxdna-driver@216cefececd74effcd7a88350c71b99f5ef9a215`.

The driver protocol surface and the executing driver are distinct identities.
The pinned open-source revision defines the driver-reachable protocol corpus
being studied. It does not assert that its source bytes produced the module
currently loaded by the host.

Every observation records the executing driver's exact module bytes, build
identity, source revision, build recipe, and relevant kernel identity. The
source-to-module relationship must be established for that executing module,
but its source revision need not equal the pinned protocol-surface revision.
If the relationship is ambiguous, or the module lacks the required existing
trace and dynamic-debug surfaces, the physical campaign stops before NPU
submission.

A separately reviewed host-preparation checkpoint may build and load a
known-provenance amdxdna module with the required existing debug capabilities.
It must preserve the previous module bytes and a tested restoration path,
match the running kernel ABI, and make no driver-source or protocol-behavior
change. Building, installing, or loading such a module is never an automatic
preflight action.

### Frozen workload

The selected Chess `add_one_using_dma` artifacts are:

| Artifact | SHA-256 |
|---|---|
| `chess/aie.xclbin` | `c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6` |
| `chess/insts.bin` | `ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896` |
| `test.exe` | `511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806` |

The sources, compiler identities, and build recipe are preserved in fixture
bundles. Preflight re-hashes the files and stops on drift. Rebuilding or
substituting any artifact creates a different fixture and requires an
intentional new campaign.

Both arms execute exactly:

```text
./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
```

`XDNA_EMU` and `XDNA_EMU_RUNTIME` are absent. The selected physical BDF and
behaviorally relevant environment are explicit and recorded.

### Treatment and control

The paired runs are:

| Arm | `force_cmdlist` | Required execute opcode |
|---|---:|---:|
| treatment | `Y` | `CHAIN_EXEC_NPU` (`0x18`) |
| control | `N` | `EXEC_DPU` (`0x10`) |

The run order is generated from a recorded seed. The vertical pair has one run
per arm, so it does not rule out order effects. Later repetitions randomize the
complete 50+50 schedule.

Each run gets a fresh userspace process and hardware context. The prior process
must exit and context destruction must complete before `force_cmdlist` changes.
The parameter is read back after every write.

## Proof Boundary

The vertical pair passes only when both arms satisfy all of these conditions in
one fresh amdxdna module epoch:

1. firmware context creation is observed;
2. the host buffer is mapped;
3. the compute unit is configured;
4. the expected mode-specific execute opcode is submitted;
5. interrupt, mailbox-response, queue-head, scheduler, and fence completion
   activity forms one coherent lifecycle;
6. context destruction completes;
7. the host process exits zero;
8. every output value is exactly the ordered sequence 2 through 65 and the
   host prints `PASS!`;
9. no new TDR or IOMMU fault is attributable to the run; and
10. capture teardown and host-state restoration succeed.

The expected high-level firmware sequence is:

```text
create context (0x02)
  -> map host buffer (0x106)
  -> configure CU (0x11)
  -> execute (0x18 treatment or 0x10 control)
  -> interrupt / response / queue-head / fence completion
  -> destroy context (0x03)
```

The resulting scoped claim is:

> On the pinned physical Phoenix NPU1 tuple, the frozen Chess
> `add_one_using_dma` workload completed both the command-list and direct
> driver-reachable firmware execution envelopes in one module epoch, and both
> produced the same exact ordered 2-through-65 array-visible result without an
> observed TDR or IOMMU fault.

The command-list result also agrees with the array-visible result retained in
the legacy emulator witness. That agreement does not validate every internal
emulator seam.

### Explicit nonclaims

This pair does not establish:

- general firmware equivalence;
- determinism or repeated-run resilience;
- absence of order or warm-state effects;
- firmware, array, or management-processor cycle accuracy;
- a physical timing model;
- complete response-payload equivalence;
- preemption, cancellation, recovery, reset, or fault behavior;
- complete ELF/PDI packaging and load behavior;
- older firmware behavior;
- every driver-reachable command;
- undocumented or development-only operations;
- correctness of every register, interrupt, clock, and firmware-array seam; or
- NPU1 retirement readiness.

The amdxdna asynchronous command-list success handler reads and logs the status
word, but reads `fail_cmd_idx` and `fail_cmd_status` only on the error path.
Ordinary dynamic debug therefore cannot recover the complete 12-byte success
response. Phase 3A leaves those success-path words explicitly unknown rather
than adding a kprobe or driver modification to the first safe capture.

## Capture Architecture

The campaign has four narrow stages:

```text
preflight and fixture resolution
  -> privileged physical capture transaction
  -> deterministic lifecycle/output derivation
  -> canonical graph emission, replication, and audit
```

All campaign working state lives beneath
`build/experiments/npu1-firmware-evidence/<campaign-id>/`. Persistent evidence
never depends on `/tmp`.

### Preflight

Preflight is read-only. It resolves and records:

- physical PCI and board identity;
- firmware bytes and hash;
- running host kernel;
- pinned driver protocol-surface revision;
- executing amdxdna module path, bytes, build identity, source revision, build
  recipe, and relationship to the running kernel;
- XRT package identities and the exact shared objects resolved for the host
  executable;
- toolchain source revisions and binaries which produced the workload;
- workload source, build recipe, and frozen artifact hashes;
- current `force_cmdlist` value;
- `tdr_timeout_ms`;
- IOMMU and address-mode state; and
- the intended tracepoint and dynamic-debug availability.

Any tuple or workload pin mismatch, unavailable required executing-driver
identity, missing trace surface, active NPU client, or ambiguous module
provenance stops before module reload or NPU submission. A stopped preflight
may recommend the separately reviewed known-module preparation checkpoint; it
does not perform that checkpoint itself.

### Privileged transaction

One narrow `pkexec` transaction establishes and owns the capture epoch. The
test process itself runs as Maya's ordinary UID.

The transaction:

1. records the original mutable tracing, dynamic-debug, and `force_cmdlist`
   state;
2. cleanly reloads `amdxdna` once, thereby loading the pinned firmware and
   establishing the module epoch;
3. verifies the normal 2,000 ms TDR remains enabled;
4. creates a dedicated tracefs instance;
5. enables the available amdxdna lifecycle events;
6. enables only the required amdxdna dynamic-debug call sites;
7. places matching run-boundary markers in trace and kernel-log evidence;
8. runs the two serialized user processes with the selected parameter values;
9. snapshots trace and kernel evidence even on failure; and
10. restores the original mutable state in an unconditional cleanup path.

The tracepoint set begins with the existing lifecycle surface:

- `xdna_job`;
- `mbox_set_tail`;
- `mbox_set_head`;
- `mbox_irq_handle`;
- `mbox_rx_worker`;
- `mbox_poll_handle`;
- `uc_irq_handle`; and
- `uc_wakeup`.

Dynamic debug supplies the firmware request bytes and available response/status
information needed to identify the lifecycle and mode-specific opcode.

Normal TDR behavior is part of the experiment. The campaign never disables the
TDR and never treats a timeout-free run under `tdr_timeout_ms=0` as equivalent.

### Derived results

Raw stdout, stderr, exit status, trace, kernel log, state snapshots, and
schedule are immutable inputs. A deterministic parser produces a derived
per-run result containing:

- exact output-oracle disposition;
- observed request opcode sequence;
- lifecycle event ordering;
- command-list or direct mode agreement;
- TDR and IOMMU deltas;
- teardown and restoration result; and
- the bounded claim and remaining unknowns.

Derived files retain exact source-artifact and tool-revision provenance. They
never replace the raw evidence.

## Failure and Safety Semantics

The campaign fails closed and stops on the first:

- nonzero process exit;
- timeout;
- missing, duplicate, unexpected, or wrong-mode execute opcode;
- incomplete required lifecycle;
- output mismatch;
- new TDR;
- new IOMMU fault;
- context or process teardown failure;
- trace or log capture failure;
- state-restoration failure; or
- fixture, bundle, replica, or graph-validation failure.

There is no automatic retry. In particular, an intermittent
`CHAIN_EXEC_NPU` complaint is evidence, not noise to retry away.

After a device fault or wedge, the transaction may perform only non-traffic
cleanup: stop tracing, snapshot evidence, restore debug controls, and restore
the original module parameter where safe. It does not reload the module,
PM-cycle the device, suspend, reset the bus, or reboot automatically. Recovery
is a separate human decision so the first failure state is not destroyed.

The partial campaign is sealed with the most specific applicable outcome,
including `device_fault_or_wedge`, `semantic_mismatch`,
`infrastructure_failure`, or `provenance_failure`. A cleanup failure is
preserved explicitly.

A structurally valid failure bundle remains valuable evidence and may later
become audited counterevidence. It cannot satisfy this vertical pair's
successful supporting/control contract. Phase 3A advances the reserve-ledger
schema with an expected campaign outcome for canonical evidence records. The
trusted audit requires the validated observation outcome to match that
expectation exactly. The new physical witness expects `success`; a reviewed
failure record preserves and expects its actual failure outcome.

If the host itself crashes before sealing, the persistent working directory
remains an incomplete capture, not a valid canonical bundle. It is retained
for manual intake and is never silently presented as complete.

## Canonical Fixture and Observation Graph

Copying `amdxdna.ko`, the firmware, XCLBIN, instructions, host executable, and
runtime identities into every observation would make a large campaign
wasteful. Phase 3A advances the bundle manifest and emission-plan schema to
version 2 with explicit bundle roles and dependencies.

Schema version 1 remains readable and valid as a self-contained campaign
format. Existing v1 bundle bytes and identities do not change. New Phase 3A
fixtures and observations emit v2. Parsing dispatches on the schema version
before deserializing a role-specific document; a v1 manifest is never coerced
through the v2 type.

The reserve ledger advances from schema version 2 to 3 with
`EvidenceRecord.expected_campaign_outcome`, using the existing explicit
known/unavailable representation. A canonical evidence record must declare a
known outcome expected from its observation bundle, and the trusted audit
cross-checks it. The legacy record has no canonical bundle and retains an
explicit unavailable outcome; it remains blocked for its existing provenance
reasons.

### Role-specific bodies

A v2 bundle has one typed role:

- `fixture`: immutable reusable inputs and their construction or acquisition
  provenance; or
- `observation`: a hardware campaign with run and observation records.

The role selects a role-specific body. Fixtures do not fabricate irrelevant
hardware runs or platform state merely to satisfy the v1 `Campaign` shape.
Observations retain the existing campaign model and point stimulus inputs at
typed fixture dependencies.

### Typed dependencies

Each canonical dependency records:

- required fixture bundle ID;
- required artifact path;
- required artifact SHA-256; and
- required semantic kind.

The bundle ID commits the complete fixture. The artifact requirement states
which exact member the dependent bundle consumed and prevents a different file
inside an otherwise valid fixture from being substituted semantically.

Machine-local dependency paths appear only in emission and validation location
plans. They never enter a canonical manifest, checksum index, or bundle ID.

The v2 emission plan supplies both canonical dependency requirements and the
local dependency paths needed for pre-rename graph validation. The emitter
copies no dependency bytes into the new bundle. If the intended fixture path
already exists, the campaign validates and reuses it rather than overwriting or
emitting a duplicate.

Dependencies may target fixture bundles only in Phase 3A. A fixture may depend
on another fixture when its provenance genuinely consumes it, but the initial
graph stays flat unless that relationship exists. Observation bundles are not
dependency targets.

### Fixture boundaries

Fixtures are separated by independent reuse:

1. **Firmware fixture**
   - the exact Phoenix firmware payload;
   - logical firmware identity and acquisition provenance.
2. **Driver-protocol fixture**
   - the pinned open-source driver-surface revision;
   - the exact source identity used to derive the driver-reachable command
     corpus.
3. **Executing-driver fixture**
   - the exact loaded `amdxdna.ko`;
   - source revision, kernel/build identity, build recipe, and an explicit
     same-as/different-from relationship to the driver-protocol fixture.
4. **Runtime/toolchain fixture**
   - XRT package and relevant binary/library identities;
   - `mlir-aie`, `llvm-aie`, `aie-rt`, register database, compiler, and build
     identities needed to reproduce the frozen artifacts.
5. **NPU-program fixture**
   - `aie.xclbin` and matching `insts.bin`;
   - source MLIR and exact build recipe.
6. **Host-oracle fixture**
   - `test.exe`, `test.cpp`, supporting source identity, and build recipe.

Another host test using the same XCLBIN and instructions reuses the
NPU-program fixture. Thousands of observations using the same loaded module
reuse the executing-driver fixture. A new executing module produces a new
fixture without changing the pinned driver-protocol fixture. Each fixture is
stored once per deliberately declared replica root.

### Observation contents

The observation bundle contains only run-specific material:

- seed and exact run schedule;
- platform and epoch state observed for this campaign;
- parameter transitions and readbacks;
- per-run stdout, stderr, exit status, and wall-clock anchors;
- tracefs capture;
- dynamic-debug and kernel-log capture;
- pre/post TDR and IOMMU state;
- teardown and restoration status; and
- derived lifecycle and oracle results.

It references the six fixture bundles rather than copying their bytes.

## Graph Validation and Replica Semantics

Graph validation is explicit. It does not search a reserve, infer nearby
directories, or depend on a database.

A local, typed location plan supplies, for each declared root:

- root alias;
- filesystem path;
- attested failure-domain ID; and
- an explicit bundle-ID-to-relative-path map.

The validator:

1. validates the requested root bundle canonically;
2. resolves each dependency only through the supplied map for that root;
3. validates every dependency bundle canonically;
4. confirms dependency bundle role and bundle ID;
5. confirms every required artifact path, hash, and semantic kind;
6. rejects missing, substituted, duplicate, and cyclic dependencies; and
7. returns an opaque validated graph result.

Validation of a replica repeats the entire graph beneath that replica's own
declared root. A fixture from the primary root cannot silently satisfy a
missing replica dependency. There is no cross-root fallback.

The same trust statement as Phase 2B applies: validation proves complete
matching bytes across operator-attested failure domains. It does not infer or
claim physical storage independence.

There is no object store, catalog, scanner, deduplication daemon, or global
bundle registry. Explicit fixture paths and content identities are sufficient.

## Evidence Admission and Ledger Mapping

Capture and validation never edit the research ledger automatically. Evidence
classification remains a review decision.

### Successful vertical pair

After joint review, one successful paired observation becomes one new
`HardwareWitness` evidence record. The record:

- points to the observation bundle;
- records its bundle, manifest, and checksum-index digests;
- expects the canonical campaign outcome `success`;
- declares two expected replicas;
- has no unresolved provenance gaps;
- uses `witness_capture` retention;
- records the most restrictive redistributability of its graph; and
- is added to the tuple's `live_attestation_evidence_ids`.

The pair becomes one evidence unit because treatment and control were captured
under one epoch and one immutable schedule. Its evidence ID may appear in both
the new fact's supporting and control lists; the run IDs and
`control_run_ids` identify the distinct arms.

A new physical fact is added rather than rewriting the historical emulator
fact. Its stable ID is:

```text
fact.npu1.firmware.physical-execution-envelope-pair
```

Its statement is the scoped claim in the Proof Boundary above. Its promotion
is `observed`, not `verified`, `encoded`, or `retirement_qualified`.

`inventory.npu1.firmware.command-list-execution` changes its active closing
fact reference from the legacy emulator candidate to the new physical fact.
The historical fact and evidence records remain intact as derived regression
history, but no longer gate this inventory item.

Tuple identity remains open. Only missing-field descriptions demonstrably
closed by the validated graph are removed; no field becomes complete by
inference. Inventory discovery remains open, and the evidence earns no
offline-rehearsal credit.

### Failed vertical pair

A failure is sealed but is not inserted automatically into supporting,
control, or counterevidence lists. Review determines whether it is:

- counterevidence to an existing physical claim;
- an infrastructure or provenance failure;
- a fixture defect; or
- a newly discovered behavior requiring its own scoped fact.

This prevents an orchestration classification from silently becoming a
hardware conclusion.

## Operational Audit Surface

The existing command remains the leaf-bundle integrity surface:

```text
xdna-reserve validate <bundle>
```

Phase 3A adds only the operational commands required for explicit graphs and
the existing release evaluator:

```text
xdna-reserve validate-graph <bundle> <location-plan.json>
xdna-reserve audit <ledger.json> <tuple-id> <location-plan.json>
```

`validate-graph` prints an informational graph report.

`audit`:

1. parses and validates the supplied ledger;
2. resolves the tuple's primary and replica locations;
3. validates each complete bundle graph in process;
4. constructs the private `EvidenceAudit`;
5. invokes the existing release evaluator; and
6. prints the current release report.

The location plan is local operational configuration, not evidence. Neither
command emits a trusted receipt. Every audit reopens and revalidates all bytes.

The overall NPU1 report is expected to remain blocked after Phase 3A because
inventory, implementation, timing, offline-rehearsal, and other retirement
requirements remain open. Success means the new evidence itself receives live
audit credit and is not the source of a blocker.

## Repetition Campaign

The background campaign is a second execution checkpoint. It does not begin
merely because the implementation tests pass.

After the vertical pair, canonical graph, ledger mapping, and audit are jointly
reviewed, the campaign may run:

- 50 `CHAIN_EXEC_NPU` treatment repetitions;
- 50 `EXEC_DPU` control repetitions;
- one recorded deterministic random seed;
- one randomized 100-run schedule;
- serial hardware access;
- one process and context per run; and
- the same fixture graph as the vertical pair.

The bounded batch becomes one new observation bundle whose run records remain
separate. Results are not collapsed into a fitted distribution. Any variation
is retained as evidence of omitted causal state.

The first anomaly stops the batch, seals all completed runs and the failing
run, and performs no retry or automatic recovery.

For unattended execution, the ordinary campaign command is launched through a
native transient user service. Its privileged capture transaction requests
authorization once at startup and remains bounded to the batch. The service
writes a terminal status file and journal output, then exits. Codex does not
poll or spin; the result is inspected when Maya requests it.

## TDD and Verification

Implementation begins with RED tests.

### Rust schema and graph tests

Tests cover:

1. existing v1 bundles remaining byte-identical and valid;
2. v2 fixture and observation round trips;
3. role-specific required and forbidden fields;
4. observation inputs resolving to declared fixture artifacts;
5. missing dependency mappings;
6. wrong bundle IDs;
7. substituted paths, hashes, or semantic kinds;
8. duplicate dependencies;
9. dependency cycles;
10. attempts to depend on an observation;
11. graph identity changing when any canonical dependency changes;
12. primary and replica graphs validating independently;
13. cross-root fallback being rejected;
14. expected campaign-outcome mismatches receiving no ledger credit;
15. a complete non-success bundle remaining auditable as reviewed
    counterevidence;
16. saved informational output being unusable as audit input; and
17. a graph-backed synthetic ledger receiving evidence credit only through
    live in-process validation.

### Campaign-tool tests

Python standard-library tests cover:

1. deterministic schedule generation from a seed;
2. exactly balanced 50+50 schedules;
3. exact ordered 2-through-65 output parsing;
4. rejection of a bare `PASS!` with wrong or missing values;
5. expected lifecycle parsing for `0x18` and `0x10`;
6. wrong, missing, duplicate, and out-of-order lifecycle records;
7. TDR and IOMMU delta detection;
8. timeout and nonzero-exit classification;
9. fail-fast partial-campaign sealing;
10. cleanup and original-state restoration accounting;
11. fixture reuse without duplicated fixture artifacts; and
12. deterministic emission-plan generation.

Hardware and privileged operations are represented by recorded command results
in unit tests. Tests never require root or touch the NPU.

### Pre-hardware gate

Before any physical run:

```bash
cargo fmt --all -- --check
cargo test -p xdna-archspec --lib
cargo test -p xdna-archspec --bin xdna-reserve
python3 tools/test_npu1_firmware_evidence.py
cargo test --lib
git diff --check
```

The full bridge suite, ISA suite, QEMU, KVM, vfio-user, emulator, and Halo are
not preconditions. This slice changes evidence plumbing and performs two
targeted physical runs; it does not alter emulator execution.

### Physical acceptance gate

The vertical pair is accepted only after:

- both arms meet the complete proof boundary;
- cleanup restores all mutable host state;
- fixture and observation bundles seal;
- the primary and two replica graphs validate;
- the ledger audit grants the new evidence live credit;
- the remaining report blockers are accurately preserved; and
- Maya reviews the raw and derived result summary.

The 50+50 campaign requires a separate explicit approval after this gate.

## Non-Goals

Phase 3A does not:

- change firmware, driver-source, driver-protocol, XRT, compiler, emulator, or
  array behavior;
- modify the amdxdna driver to expose additional response words;
- add kprobes, eBPF, or BPF tracing;
- claim unknown command-list success-response fields;
- retrofit the legacy witness into a canonical bundle;
- duplicate shared fixtures in every observation;
- build an object store, database, daemon, watcher, or storage service;
- infer physical storage independence;
- retry intermittent hardware failures;
- automate module, bridge, power, suspend, bus-reset, or reboot recovery;
- run two NPU jobs concurrently;
- produce a timing or cycle-accuracy claim;
- promote the new fact beyond `observed`;
- close general firmware equivalence;
- qualify NPU1 for retirement; or
- use Halo.

## Exit Criteria

Phase 3A's implementation slice is complete when:

- schema v2 represents canonical fixtures, observations, and explicit
  dependencies without invalidating v1 bundles;
- graph validation rejects every specified substitution and replica failure;
- canonical evidence receives no audit credit when its observed outcome
  differs from the ledger's reviewed expected outcome;
- the pinned driver-protocol source and exact executing-driver provenance are
  represented independently;
- the campaign tool passes its deterministic synthetic tests;
- the frozen vertical pair runs once on physical NPU1 under normal TDR;
- both arms produce exact output and their required firmware lifecycles;
- no TDR, IOMMU, teardown, capture, or restoration failure occurs;
- shared fixtures exist once per declared replica root;
- the observation and both replica graphs validate;
- the new physical fact and hardware witness are admitted conservatively;
- the live ledger audit recognizes the evidence while retaining unrelated
  release blockers;
- exact RED and GREEN evidence is recorded in a closure report; and
- the worktree is clean.

Launching or completing the 50+50 campaign is not required to call the
vertical implementation slice complete. It is the immediately following,
separately approved execution checkpoint using the same finished machinery.

## Following Boundary

After a successful vertical pair and repetition campaign, the next design
interprets any cross-run variation and chooses the next firmware surface from
the driver-reachable inventory. Likely candidates include repeated lifecycle
resilience, complete response-payload observation, packaging/load variation,
preemption, and recovery. The evidence, rather than a preset implementation
sequence, decides which missing causal variable comes next.

## Implementation Authorization

Approval of this design authorizes this design record only. The implementation
plan and code remain separate checkpoints. No Phase 3A implementation begins
until Maya reviews the committed specification, reviews the implementation
plan, and explicitly approves execution.
