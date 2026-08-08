# Phoenix BlockWrite Phase Crossover

**Status:** Completed on the pinned Phoenix tuple. The exact phase-following
result and receipt are recorded in
[`2026-08-08-phoenix-blockwrite-transaction-phase.md`](../findings/2026-08-08-phoenix-blockwrite-transaction-phase.md).
No scheduler or timing-model change is authorized by this design.

## Motivation

The low-QoS physical capture under
`build/experiments/phoenix-pm-clock-characterization/20260808T201627Z-firmware-blockwrite-timeline/`
was structurally valid, repeated exactly across two runs, and restored the NPU
cleanly. For word counts `1, 2, 1, 4, 1, 8, 1, 4`, both runs produced shim
intervals:

```text
246, 274, 216, 288, 246, 338, 248, 280
```

Equal word counts therefore do not have one context-free cost. The two
one-word records beginning at the same transaction phase (`56 mod 64`) both
cost 246 shim cycles, despite different predecessors. That makes transaction
phase the leading hypothesis, but it is not yet a conclusion: phase, marker
ordinal, padding history, and trace packet state covary in the present stream.

The next experiment isolates those variables before any instruction cost or
firmware/array cadence is inferred.

## Scope and Question

Measure one identical signed-firmware operation -- a one-word zero
`BlockWrite` to unused shim DMA BD14 -- at four transaction phases while
crossing phase against marker ordinal and predecessor history.

The experiment answers only:

> Does the exact physical interval follow transaction byte phase, measurement
> ordinal, or preceding command history when the executed BlockWrite path is
> held fixed?

It does not estimate a general CPI table, assign latency to an Xtensa
instruction, fit a scheduler multiplier, or characterize payload-length
scaling.

## Instrument

Reuse the existing toolchain-derived BlockWrite instrument and its safety
checks:

- derive BD14 width and all register/event identities from AM025 and aie-rt;
- reject a source transaction that writes or queues BD14;
- clear the full descriptor before the first measurement; and
- leave BD14 unqueued throughout the run.

Each measured window is:

```text
USER_EVENT_1 start -> BlockWrite(1 word) -> USER_EVENT_0 stop
```

The source-authored first `USER_EVENT_1` remains the trace lifecycle start.
Later `USER_EVENT_1` writes are measurement starts. Trace configuration records
both user events plus the existing DMA start/finish lifecycle witnesses. The
distinct `PERF_CNT_0` event remains the real trace stop after the final
measurement marker.

Authentic CDO `NOOP` records select the next phase only between a stop marker
and the following start marker, outside every measured window. The
classifier accepts exactly four alternating measurement start/stop pairs. It
may accept one additional leading `USER_EVENT_1` only when emulator
qualification proves that the source-authored trace-start event itself is
recorded; it must never guess around another marker mismatch.

## Crossover Matrix

The measured BlockWrite record starts at these byte phases modulo 64:

```text
0, 20, 44, 56
```

They include the previously observed low-cost phase, the two higher-cost
observed one-word phases, and the cache-line origin. Two instrument streams use
opposite orders:

```text
forward: 0, 20, 44, 56
reverse: 56, 44, 20, 0
```

Reversal changes every phase's measurement ordinal, predecessor, and padding
history. Classification can therefore distinguish a repeatable four-position
trace or pipeline pattern from a byte-phase law.

## Qualification and Capture

Before touching hardware, signed-firmware emulation must prove for both
streams that:

1. all four windows execute the same dynamic PC and operation multiset;
2. each window reports the same corrected firmware work count;
3. marker order and count are exact;
4. the kernel output and lifecycle remain correct; and
5. generated candidates are pinned by SHA-256 in the physical receipt.

The physical campaign uses only the already deterministic Phoenix low-QoS
identity `(gops=1, fps=1000)`, reported as MP-NPU/H `400/800 MHz`:

1. verify the pinned kernel, driver, firmware, XCLBIN, source transaction, and
   default `600/1028` starting state;
2. prime one fresh non-reused context at low QoS with the ordinary source
   transaction;
3. run candidates in the order forward, reverse, forward, reverse, with a
   fresh context for every dispatch;
4. require exact output, clocks, lifecycle events, and marker pairs on every
   run; and
5. in a `finally` path, run the ordinary source transaction without QoS and
   require restoration to the original output and clock identity.

Raw traces, decoded events, hashes, clocks, runner results, and the executable
capture harness remain together under one new `build/experiments/` receipt.

## Decision Rules

Each candidate must first repeat digit-for-digit across its two runs. Failure
stops the experiment as nondeterministic or invalid.

For the four phase costs in each order:

- **Phase-following:** each phase has the same cost in forward and reverse,
  while at least two phases differ. Transaction phase is admitted as a causal
  timing input. Next, map all sixteen four-byte phases before a phase-matched
  payload-length sweep.
- **Ordinal-following:** costs match by measurement position rather than by
  phase across the reversal, while at least two positions differ. The
  trace/marker witness or another ordinal mechanism is implicated; redesign
  the witness before measuring BlockWrite.
- **Mixed or history-dependent:** both streams repeat, but neither pure phase
  nor pure ordinal explains the costs. Phase, predecessor, padding, or their
  interaction remains unresolved. Use a smaller crossover or a drain/fence
  derived from the toolchain next; do not label the mechanism MMIO or NoC
  without further evidence.
- **Uniform:** every qualified window has one cost. The earlier variation was
  removed by the crossover structure, so proceed directly to the
  phase-controlled payload-length probe while preserving this instrument as a
  control.
- **Ambiguous or invalid:** overlapping symmetries, marker anomalies, clock
  changes, output mismatch, lifecycle failure, or restoration failure license
  no timing conclusion.

## Test Boundary

Implementation is complete only when unit tests cover derived phase padding,
both orderings, exact marker pairing, source-BD14 rejection, every decision
rule, and fail-closed malformed traces. The existing Python tool suite and
`cargo test --lib` must pass. Physical evidence is admitted only after the two
emulator-qualified candidate hashes run through the pinned capture harness.
