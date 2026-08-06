# Phoenix Real Column-Gate Freeze/Resume Implementation Plan

> **Execution:** Primary Codex agent only. Repository instructions forbid
> subagents. Work task-by-task with an observed RED before each production
> change and a focused GREEN before committing.

**Goal:** Build and safely run the approved one-word control/treatment witness
that proves whether the real Phoenix column clock gate freezes and resumes a
pinned core workload while the shim remains live.

**Architecture:** Extend the existing transaction patcher with one generic
record-insertion seam. Keep the fixture-specific stream builder, manifest, and
classifier in `phoenix-pm-clock-characterize.py`. Add one fail-closed live
placement assertion to the existing bridge runner. Execute the exact generated
streams through the signed-firmware guard before any physical submission.

**Spec:**
[`2026-08-05-phoenix-real-column-gate-resume-design.md`](../specs/2026-08-05-phoenix-real-column-gate-resume-design.md)

## Constraints

- Derive clock register/field data from AM025 and NPI sequence values from the
  resolved aie-rt source; the pinned firmware supplies only the Phoenix NPI base
  and transaction-base formula.
- Admit only physical placement `1:1`, zero register-offset high words, and the
  three targets named in the spec.
- Control and treatment differ in exactly one 32-bit gate-value word.
- Keep context reuse and asynchronous context creation disabled for the run.
- No new dependency, campaign framework, public clock API, scheduler change,
  synthetic marker, or broad test suite.
- KVM/VFIO first. Review its receipt before any host run.

## Task 1: Generic transaction-record insertion

**Files:**

- Modify: `tools/trace-patch-events.py`
- Test: `tools/test_phoenix_pm_clock_characterize.py`

- [ ] Add a failing test that inserts heterogeneous validated records after the
  last TCT and before the existing two trace-stop writes.
- [ ] Run the focused test and confirm the failure is the missing seam.
- [ ] Implement the smallest helper that validates record lengths and updates
  transaction count/size once; make NOOP insertion reuse it.
- [ ] Run the focused file and commit.

## Task 2: Build the paired streams and manifest

**Files:**

- Modify: `tools/phoenix-pm-clock-characterize.py`
- Modify: `tools/test_phoenix_pm_clock_characterize.py`

- [ ] Add failing tests for named aie-rt macro parsing, named AM025 clock-field
  lookup, exact two-envelope ordering, protected closure around both dwells,
  derived addresses, and the one-word control/treatment difference.
- [ ] Add rejection tests for wrong placement, missing source definitions,
  arithmetic/range/alignment failure, nonzero high words, and allowlist drift.
- [ ] Implement a small fixture-specific builder returning control bytes,
  treatment bytes, and their manifest. Use only stdlib plus existing patcher
  helpers.
- [ ] Run the focused file and commit.

## Task 3: Classify freeze and resume

**Files:**

- Modify: `tools/phoenix-pm-clock-characterize.py`
- Modify: `tools/test_phoenix_pm_clock_characterize.py`

- [ ] Add literal positive control/treatment event series and confirm RED.
- [ ] Add one parameterized mutation table covering missing shim liveness,
  insufficient pre/post samples, short/multiple gaps, irregular cadence,
  absent resume, core/broadcast count mismatch, output/clocks/completion failure,
  and failed canary.
- [ ] Implement the exact-cadence classifier by reusing `_constant_cadence`.
- [ ] Run the focused file and commit.

## Task 4: Fail closed on live context placement

**Files:**

- Modify: `bridge-runner/bridge-trace-runner.cpp`

- [ ] Establish RED with the absent `--expect-placement 1:1` CLI contract.
- [ ] Reuse the `DRM_AMDXDNA_HW_CONTEXT_ALL` query pattern from
  `tools/txn-poll-probe/txn-poll-probe.cpp`.
- [ ] Query after final context creation and before BO allocation; require one
  same-PID entry and an exact placement match.
- [ ] Build the runner. In KVM, prove a mismatched value fails before submission
  and exact `1:1` reaches the unmodified control.
- [ ] Commit.

## Task 5: Signed-firmware structural preflight

**Files:**

- Modify: `src/firmware/boot_tests/guards.rs`

- [ ] Add an env-gated focused guard that consumes the exact generated control,
  treatment, and manifest, then confirm RED on the missing access contract.
- [ ] Execute both streams through the existing pinned context command path and
  compare the observed inserted access sequence with the manifest.
- [ ] Require both ordinary final trace-stop writes and successful command
  completion; make no behavioral clock claim.
- [ ] Run the focused guard with both streams and commit.

## Task 6: Software verification and artifact generation

- [ ] Generate the timestamped experiment directory from the pinned full-witness
  input and verify its required SHA-256 before construction.
- [ ] Run:

  ```bash
  nice -n 19 python3 -m pytest tools/test_phoenix_pm_clock_characterize.py
  nice -n 19 cmake --build bridge-runner/build
  nice -n 19 cargo test --lib
  ```

- [ ] Re-run the exact-stream signed-firmware preflight and review the manifest,
  byte diff, and allowlist before physical access.

## Task 7: KVM evidence gate

- [ ] Record KVM firmware, driver, kernel, XRT, placement, and clock identities.
- [ ] Run control, decode `c`/`b`/`h`, require exact output and classification,
  then destroy the context and pass the fresh ordinary canary.
- [ ] Run treatment only after control passes; apply the same checks and canary.
- [ ] Preserve raw evidence plus a compact receipt under the experiment
  directory.
- [ ] Stop and review together. A host run remains unauthorized until that
  receipt passes review.
