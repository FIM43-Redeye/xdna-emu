# Phoenix BlockWrite Sixteen-Phase Map

**Status:** Approved design. No scheduler or timing-model change is authorized
by this experiment.

## Question

For the pinned Phoenix signed-firmware `BlockWrite(1)` path, what shim timestamp
cost belongs to each word-aligned transaction-record phase modulo 64?

The completed four-phase crossover established transaction phase as a causal
input and falsified ordinal timing for phases `0,20,44,56`. This experiment
extends that proof to the complete word-aligned domain without assigning a
physical mechanism to the resulting map.

## Reused Instrument

Reuse `instrument_firmware_blockwrite_phase_crossover` and its existing
fail-closed classifier. They already accept arbitrary distinct word-aligned
phases, derive BD14 and event fields from AM025 and aie-rt, reject source use of
BD14, clear the full descriptor once, and bracket every measured operation as:

```text
USER_EVENT_1 -> BlockWrite(1 zero word) -> USER_EVENT_0
```

Only authentic four-byte CDO `NOOP` records may adjust the next transaction
phase. They remain outside every measured window. No new timing abstraction or
production scheduler behavior is needed.

## Candidates and Order

Build two candidates from the same pinned source transaction:

- ascending: `0,4,8,...,60`;
- descending: `60,56,52,...,0`.

Run them physically in the order ascending, descending, ascending, descending.
Each dispatch uses a fresh context under the same low-QoS contract as the
four-phase crossover. Reversal changes ordinal, predecessor, and padding
history for every phase while retaining the same sixteen operations.

## Emulator Qualification

Before hardware access, both exact candidate hashes must pass the pinned
signed-firmware guard. Qualification requires:

1. the expected output;
2. exactly sixteen alternating start/stop marker pairs;
3. the same firmware work-attempt count for every measured window; and
4. the same normalized dynamic PC sequence for every measured window within
   and across both candidates.

The qualification assertions are temporary evidence guards and must be removed
afterward. The direct XRT CDO executor's missing `NOOP 0x05` support remains
outside this experiment.

## Physical Admission and Decision

Every physical run must reproduce the expected output, retain the same reported
clock identity before and after dispatch, contain both shim DMA lifecycle
witnesses, and expose exactly 32 alternating measurement markers. Two runs of
the same candidate must repeat digit-for-digit.

The result is a qualified sixteen-phase map only if the ascending and descending
phase-to-cost dictionaries are identical. A disagreement is preserved as
mixed or history-dependent evidence; values are never averaged, fitted, or
used to tune the scheduler. A uniform map is also valid evidence and is
reported as uniform rather than forced into a phase law.

## Provenance and Restoration

The campaign fails closed on any drift in the pinned kernel, loaded driver,
firmware payload, source transaction, XCLBIN, expected output, runner, clock
query, candidate hashes, or clean worktree commit. It preserves candidates,
raw traces, decoded events, outputs, logs, provenance, and the final decision
under `build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output, and requires the reported default clock identity. Postflight records
device ownership and recent kernel warnings/errors. Any failed restoration is
part of the result and blocks a timing conclusion.

## Completion Boundary

The existing focused Python tests and `cargo test --lib` must remain green. Add
one explicit unit case proving placement of all sixteen phases; do not add a new
classifier or generalized campaign framework. Completion produces a committed
finding with the exact phase table and licenses only the next evidence boundary:
a payload-length sweep that holds transaction phase fixed.
