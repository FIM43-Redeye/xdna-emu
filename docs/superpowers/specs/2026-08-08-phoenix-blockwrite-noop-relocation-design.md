# Phoenix BlockWrite NOOP Relocation Crossover

**Status:** Completed. The physical A/B/A/B crossover qualified as
`history_sensitive`: phase 40 stayed at 246 cycles while phase 44 changed from
232 to 248 cycles. No scheduler or timing-model change is authorized by this
experiment.

## Question

At fixed target phase 44, does the position of a sixteen-record CDO `NOOP`
block relative to the phase-40 predecessor window change signed-firmware
`BlockWrite(1)` timing?

The complete phase crossover made leading `NOOP` history the best current
discriminator, but its ascending and descending candidates also differed in
predecessor and total padding. This experiment isolates recency by relocating,
not adding, one complete 64-byte turn.

## Balanced Candidates

Both candidates use phase order `(40,44)`, contain the same two measured
`BlockWrite(1)` operations, and contain exactly sixteen extra authentic
four-byte CDO `NOOP` records:

- **Control A:** sixteen `NOOP`s before the phase-40 predecessor window, then
  phase 40 followed immediately by target phase 44.
- **Treatment B:** phase 40, then the same sixteen `NOOP`s, then target phase
  44.

One 64-byte turn preserves every transaction phase modulo 64. Relocation also
keeps the transaction header count and size, total bytes, total `NOOP` count,
target absolute byte offset, target instruction ordinal, predecessor phase,
target phase, payload, and measured target path identical. Only whether the
`NOOP` block is immediate to target phase 44 or separated from it by the
phase-40 window changes.

A control that merely omitted the sixteen records is rejected because header
count and total size would differ. Moving the control block after the target is
also rejected because the number of records remaining at target would differ.

## Minimal Instrument Change

Extend `instrument_firmware_blockwrite_phase_crossover` with an optional
per-window count of leading 64-byte turns. Each turn emits exactly sixteen CDO
`NOOP` records before that window's start marker and leaves the requested phase
unchanged. Omission retains the current behavior. The input must fail closed on
length mismatch, booleans, negative values, or non-integers.

Generate A with turns `(1,0)` and B with `(0,1)`. Tests must prove identical
header count, total size, target absolute offset, target ordinal, and exact
record order while showing that the sixteen-record block moved across the
phase-40 window.

## Admission and Classification

Each decoded physical run must reproduce the expected output, retain one clock
identity before and after dispatch, include both shim DMA lifecycle witnesses,
and expose exactly two alternating start/stop marker pairs at phases `(40,44)`.

The crossover classifier requires two digit-for-digit repeats of each arm. It
also requires the phase-40 witness cost to agree across A and B; disagreement
means the predecessor window itself changed and the target comparison is not
admitted. If phase 40 agrees:

- different phase-44 costs qualify as `history_sensitive`;
- equal phase-44 costs qualify as `history_invariant`;
- malformed, non-repeating, phase-mismatched, or contaminated inputs fail
  closed without averaging.

The classifier preserves both exact arm intervals and the signed target delta.

## Signed-Firmware Qualification

Before hardware access, both exact candidate hashes must pass the pinned
signed-firmware `CHAIN_EXEC_NPU` guard. Each must expose one leading source
`USER_EVENT_1` followed by two start/stop measurement pairs. Both measured
windows must consume exactly 47 attempted firmware instructions and execute the
same normalized 47-PC sequence within and across candidates.

Temporary qualification assertions must be removed afterward. The direct XRT
executor's missing CDO `NOOP 0x05` support remains outside this experiment.

## Physical Order, Provenance, and Restoration

Run A, B, A, B with fresh contexts under the same pinned low-QoS `400/800 MHz`
identity used by the preceding campaigns. Fail closed on drift in the kernel,
loaded driver, firmware payload, source transaction, XCLBIN, expected output,
runner, clock query, candidate hashes, or clean worktree commit.

Preserve the harness, candidates, qualification receipt, provenance, raw
traces, decoded events, outputs, logs, and decision under
`build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output, and requires the original reported default clock identity. Postflight
records device ownership and recent kernel warnings/errors. Restoration failure
blocks every timing conclusion.

## Completion Boundary

Use test-first development for the turn-placement knob, its validation, and the
fail-closed A/B classifier. The focused Python suite and `cargo test --lib` must
pass. Completion records the exact target comparison and either licenses a
narrower history-depth experiment or falsifies immediate-versus-separated
sixteen-`NOOP` placement at this target. It does not begin payload scaling or
alter emulator timing.

## Completion Receipt

Implemented at commit `c378190789058fc1cfd2487ace2e9ca7b2bed182` and captured
under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260808T224132Z-firmware-blockwrite-noop-relocation/
```

Both exact candidates passed the signed-firmware guard with two 47-attempt
windows and one identical 47-PC sequence. The physical order A, B, A, B
repeated `246,232` and `246,248` respectively, restored the ordinary source
transaction at reported default `600/1028 MHz`, and left the NPU unowned. Full
interpretation and pins are recorded in
[`2026-08-08-phoenix-blockwrite-noop-relocation.md`](../findings/2026-08-08-phoenix-blockwrite-noop-relocation.md).
