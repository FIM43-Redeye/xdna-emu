# Subsystem audit checklist

The questions to ask when auditing a subsystem listed in
[aie2/architecture-index.md](aie2/architecture-index.md). Apply when adding a row,
and re-apply when the subsystem's coverage state changes.

The point of this document is to make the failure modes that have
already burned us once become routine to check for. Each item exists
because we missed it at some point.

## 1. Source-of-truth verification

- [ ] Does an authoritative source (aie-rt, AM025, mlir-aie device
  model) define this subsystem? Cite the file path in the index row.
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
  document it explicitly in the index row's notes.

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

When you set a row in `aie2/architecture-index.md` to **MODELED** —
either after writing a new subsystem or after closing a gap —
walk through this list. Note any item you skipped (and why) in
the row's notes column.

When grading existing **MODELED** rows, run them through item 3
specifically: "what consumes this field per cycle?" If the answer
is "nothing", drop the row to **PARTIAL** and add it to the gap
list.
