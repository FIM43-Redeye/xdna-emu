# Subsystem audit checklist

The questions to ask when adjudicating a subsystem's behavioral coverage
in the two-axis coverage model. Apply before flipping a behavioral unit's
verdict toward `Verified { evidence }` or `Accepted { rationale }` (closing
a line in [aie2/perishable-queue.md](aie2/perishable-queue.md) or
[aie2/comprehension-gaps.md](aie2/comprehension-gaps.md)), or when adding an
override unit. The generated
[aie2/architecture-index.md](aie2/architecture-index.md) is the rolled-up
view of those verdicts; it is regenerated, never hand-edited.

The point of this document is to make the failure modes that have
already burned us once become routine to check for. Each item exists
because we missed it at some point.

## 1. Source-of-truth verification

- [ ] Does an authoritative source (aie-rt, AM025, mlir-aie device
  model) define this subsystem? Cite the file path in the unit's
  `evidence` or `shadows_derived` narrative.
- [ ] If multiple sources define it, do their constants agree? (See the
  lock-count / BD-count cross-check episode, 2026-05-04 — aie-rt's
  `xaiemlgbl_reginit.c` is the most reliable arbiter for AIE-ML.)

## 2. Surface coverage

- [ ] Are all registers in this subsystem either readable, writable, or
  explicitly NOP'd in our model?
- [ ] Are register reset values correct? (Default-zero is wrong for
  several AIE-ML registers — e.g. timer threshold defaults to
  `0xFFFFFFFF_FFFFFFFF`.)
- [ ] Do register-level unit tests exercise every named field?

## 3. Subsystem-integration check

**The check that catches "looks complete but is a no-op".** Added
2026-05-04 after the multi-tile timer-sync gap was found — the timer
module had complete register-level coverage, full unit tests, and a
correct reset method, but no per-cycle code path actually consumed
the configured `Reset_Event` field. The subsystem was functionally
inert despite a clean local API.

For each non-trivial field this subsystem stores (event IDs,
configuration enums, mode bits, threshold registers, ...):

- [ ] **Producer**: can show me the code path that *writes* this field
  during real workloads.
- [ ] **Consumer**: can show me the code path that *reads* this field
  per cycle (or per event, per packet, etc.) and acts on its value.
- [ ] If no consumer exists, the field is a write-only register from
  software's point of view — flag it. Either model the consumer or
  document it explicitly in the unit's verdict (`Accepted { rationale }`)
  or its `shadows_derived` narrative.

A clean subsystem has both producer and consumer for every meaningful
field. A subsystem with producers but no consumers is the shape of
this 2026-05-04 bug class.

## 4. Cross-tile / cross-subsystem effects

- [ ] If this subsystem fires events, do the events reach
  [other subsystems'](https://en.wiktionary.org/wiki/cross-subsystem)
  consumer hooks (event subsystem, trace unit, perf counters, edge
  detectors, **timer reset path**)?
- [ ] If this subsystem participates in broadcast / multi-tile
  protocols (timer sync, broadcast events, lock release across
  tiles, cascade), is the cross-tile case covered?

## 5. Test coverage

- [ ] Unit tests for the subsystem's own state machine.
- [ ] Integration tests for at least one cross-subsystem path
  (e.g. event fires → trace unit captures, timer reset event fires
  → timer resets).
- [ ] Bridge / hardware comparison test where applicable.

## 6. Known-deferred issues

- [ ] Are there test cases that exercise this subsystem but are
  currently disabled (`trace-quarantine.txt`, `test-quarantine.txt`,
  `hw-quarantine.txt`)? If so, link them in the notes.

## How to use this checklist

When you flip a behavioral unit's verdict toward `Verified { evidence }`
or `Accepted { rationale }` — either after modelling a new subsystem or
after closing a perishable / comprehension entry — walk through this
list. Record anything you skipped (and why) in the unit's
`Accepted { rationale }` or its `shadows_derived` narrative; that text
is what a future reader sees, not a hand-edited index cell.

When re-grading an existing `Verified` or toolchain-derived unit, run it
through item 3 specifically: "what consumes this field per cycle?" If the
answer is "nothing", the unit is not actually modelled — demote its
verdict. It then reappears in perishable-queue.md / comprehension-gaps.md
by construction, which is the gap list regenerating itself.

A subsystem that is only partially built or stubbed lands in the
generated [aie2/implementation-gaps.md](aie2/implementation-gaps.md) by
its `CapabilityDomain` verdict (`Modeled{Partial|Stub}`), the same way a
weak-provenance op lands in the perishable queue. Closing it means
raising that domain's seeded verdict in `capability_spine()` to
`Modeled{Full}` (or a closed state) with honest evidence -- the
subsystem-index and implementation-gaps docs then regenerate to match.
