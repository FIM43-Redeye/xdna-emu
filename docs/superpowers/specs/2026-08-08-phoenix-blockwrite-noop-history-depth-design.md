# Phoenix BlockWrite NOOP History-Depth Discriminator

**Status:** Completed. The physical `A/B/C/C/B/A` discriminator returned target
tuple `(232,248,248)`, the preregistered `saturating_hot_cold` outcome. No
scheduler or timing-model change is authorized by this experiment.

## Question

The balanced relocation crossover proved that moving one sixteen-record CDO
`NOOP` turn from before the phase-40 predecessor to immediately before the
fixed phase-44 `BlockWrite(1)` target changes the target from 232 to 248 shim
cycles. It did not distinguish a saturating hot/cold state, a periodic clock
state, or an accumulating per-turn state.

At the same fixed predecessor and target, what happens when two complete
sixteen-record turns are distributed across that boundary?

## Balanced Arms

Reuse `instrument_firmware_blockwrite_phase_crossover` without changing its
record generator. All arms use phases `(40,44)` and contain the same 32 added
four-byte `NOOP` records:

- **A:** `leading_full_turns=(2,0)` -- 32 before phase 40, none between.
- **B:** `leading_full_turns=(1,1)` -- 16 before phase 40, 16 between.
- **C:** `leading_full_turns=(0,2)` -- none before phase 40, 32 between.

Because each arm moves complete 64-byte turns, all three retain the same
transaction header count and size, total bytes, total `NOOP` count, phase-40
predecessor, phase-44 target, target absolute byte offset, target record
ordinal, target bytes, payload, and remaining-record count. Only the split of
the 32 records across the fixed predecessor changes.

The candidate receipt must derive and prove those invariants from the emitted
bytes. Any mismatch stops before firmware or hardware access.

## Admission

Each arm must first pass the pinned signed-firmware `CHAIN_EXEC_NPU` guard. The
three candidates must expose one source start marker followed by two complete
measurement windows; every window must consume exactly 47 attempted firmware
instructions and follow the same normalized 47-PC path within and across arms.

Run fresh physical contexts in order `A/B/C/C/B/A` under the pinned low-QoS
`400/800 MHz` identity. Every run must reproduce expected output, preserve the
clock identity before and after dispatch, include both shim DMA lifecycle
witnesses, and expose exactly two alternating marker pairs at phases `(40,44)`.
Both digit-for-digit repeats of each arm must agree. Phase 40 must agree across
all arms; otherwise the predecessor changed and no target comparison is
admitted.

## Fail-Closed Classifier

Preserve every exact interval and classify the phase-44 target tuple `(A,B,C)`
only as follows:

| Target tuple | Narrow discriminator result |
|---|---|
| `(232,248,248)` | `saturating_hot_cold` |
| `(232,248,232)` | `periodic_temporal` |
| `(232,248,264)` | `accumulating_linear` |

Any other tuple is `unclassified`, not a license to fit a different model.
Malformed runs, drift, nondeterminism, a changed predecessor, or a tuple that
does not match one of the three preregistered outcomes is unqualified. The
names describe the hypotheses discriminated by this one boundary; they do not
identify the physical state carrier or authorize scheduler changes.

## Provenance and Restoration

Pin the clean worktree commit, kernel, loaded driver, firmware payload, source
transaction, XCLBIN, expected output, runner, clock query, exact candidate
hashes, qualification receipt, layout receipt, and reused base harness.
Preserve candidates, raw traces, decoded events, outputs, logs, receipts, and
the decision under `build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies
its output and original default clock identity, records device ownership and
recent kernel warnings/errors, and blocks every timing conclusion if restore
fails.

## Completion Boundary

Use tests first for the byte-derived three-arm balance and classifier. Run the
focused Python suite and `cargo test --lib`. This experiment may license one
narrow next timing boundary; it does not alter firmware, the emulator
scheduler, or any timing cost.

## Completion Receipt

Implemented at commit `b7c097531471b814fb9c1e9f75b172bcfdacea0b` and captured
under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260808T234135Z-firmware-blockwrite-noop-history-depth/
```

All three exact candidates passed the signed-firmware guard with two
47-attempt windows and one identical 47-PC sequence. The six physical runs
repeated `246,232`, `246,248`, and `246,248` by arm, restored the ordinary
source transaction at reported default `600/1028 MHz`, and left the NPU
unowned. Full interpretation and pins are recorded in
[`2026-08-08-phoenix-blockwrite-noop-history-depth.md`](../findings/2026-08-08-phoenix-blockwrite-noop-history-depth.md).
