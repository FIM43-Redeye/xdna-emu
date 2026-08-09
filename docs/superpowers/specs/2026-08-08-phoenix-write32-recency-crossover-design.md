# Phoenix Same-address WRITE32 Recency Crossover

**Status:** Completed and physically qualified; see
[`2026-08-08-phoenix-write32-recency-crossover.md`](../findings/2026-08-08-phoenix-write32-recency-crossover.md).

## Question

At the pinned Phoenix signed-firmware boundary, does a harmless `WRITE32` to
the same unused shim-DMA BD14 word reproduce the recency-sensitive phase-44
target change previously caused by `BlockWrite(1)`?

The completed BlockWrite order discriminator measured 248 cycles when the
re-prime preceded sixteen `NOOP`s and 232 cycles when the same re-prime
followed them. That result changed both the operation nearest the target and
the distance from the re-prime to the target. This crossover keeps the
treatment operation fixed while moving it across the same sixteen-record turn.

## Source-derived Treatment

Derive the six-word `WRITE32` encoding from the resolved mlir-aie
`include/aie/Runtime/TxnEncoding.h`; do not duplicate its opcode or layout as
campaign constants. Candidate generation fails if the toolchain no longer
proves the expected fixed-size encoder contract.

The write targets word zero of the existing AM025-derived unused shim DMA BD14
seam and writes zero. The source transaction must still prove that it neither
writes nor queues BD14. This gives the treatment the same destination, value,
and final register effect as the established one-word BlockWrite re-prime while
changing the firmware opcode and handler path.

The existing trace-start marker is itself a `WRITE32` to `Event_Generate` and
remains byte-identical in every candidate. The experiment therefore does not
ask whether any recent write is sufficient. It asks whether a same-address
BD14 `WRITE32` has the same recency behavior as the prior BD14
`BlockWrite(1)`.

## Balanced Candidates

Reuse the established phase-40 predecessor and phase-44 measured target. After
the phase-40 stop marker, both candidates contain one identical BD14
`WRITE32` and sixteen identical one-word `NOOP`s before the target start
marker:

- **A -- old WRITE32:** `WRITE32`, sixteen `NOOP`s, target;
- **B -- recent WRITE32:** sixteen `NOOP`s, `WRITE32`, target.

The `NOOP` turn is exactly 64 bytes. Moving the 24-byte write across it
preserves the write's transaction phase. Each deliberate inter-window sequence
is 88 bytes and seventeen records.

An emitted-byte receipt must prove:

1. identical candidate lengths, header sizes and operation counts;
2. a byte-identical phase-40 predecessor at the same offset and ordinal;
3. a byte-identical phase-44 target at the same offset and ordinal;
4. identical alignment and marker records;
5. one identical derived BD14 `WRITE32` in each candidate, at offsets separated
   by exactly 64 bytes and at the same phase modulo 64; and
6. sixteen contiguous `NOOP`s whose order relative to that write is the only
   candidate difference.

Any mismatch stops before signed-firmware execution.

## Signed-firmware Admission

Run both exact candidates through the pinned signed-firmware `CHAIN_EXEC_NPU`
guard before physical access. Both must reproduce the command response, output,
marker lifecycle, and established normalized 47-PC paths for the phase-40 and
phase-44 measured BlockWrites. Between the measured windows, the guard must
prove exactly one finite BD14 `WRITE32` handler, sixteen finite `NOOP`
handlers, equal attempted-instruction counts, and no fault, idle wait, context
transfer, scheduler boundary, or other lifecycle change.

Admission establishes candidate integrity only; emulator instruction counts
are not asserted to equal silicon instruction counts.

## Physical Campaign

Run fresh physical contexts in symmetric order `A/B/B/A` under the pinned
low-QoS `400/800 MHz` identity. Every run must reproduce expected output,
preserve the clock identity, contain both shim-DMA lifecycle witnesses, expose
the two phase `(40,44)` marker pairs, and leave the device unowned. Both repeats
of each arm must agree digit-for-digit, every phase-40 predecessor must remain
246 cycles, and restoration plus the campaign-scoped kernel warning/error
check must pass.

## Fail-closed Classifier

After all controls and repeats qualify, classify the exact pair of phase-44
targets:

| A / B target | Narrow result |
|---|---|
| `248 / 232` | `write32_recency_matches_blockwrite` |
| `248 / 248` | `write32_recency_invariant` |
| any other repeatable positive pair | `write32_recency_other` |

The first result means the cold endpoint does not require the BlockWrite
opcode or handler path. The second means this same-address WRITE32 does not
reproduce the BlockWrite recency transition, leaving the BlockWrite path or
state as necessary at this boundary. The third records a new deterministic
state without fitting a mechanism. Nonrepeat, malformed evidence, control
failure, clock drift, or failed restoration is `unclassified`.

## Tests, Provenance, and Restoration

Tests come first for toolchain derivation, changed-source rejection, candidate
byte invariants, unsafe BD14 sources, and every classifier branch. Run the
focused Python characterization suite and `nice -n 19 cargo test --lib` before
hardware access.

Pin the clean worktree commit, kernel, loaded driver, firmware payload,
resolved toolchain commits and source hashes, source transaction, XCLBIN,
expected output, runner, clock query, exact candidates, layout receipt,
signed-firmware qualification, and worktree FFI library. Preserve all
candidates, traces, outputs, logs, receipts, and the decision under
`build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output and original default clock identity, records ownership and recent kernel
warnings/errors, and blocks every timing conclusion if restoration fails.

## Completion Boundary

This experiment may determine whether the known recency transition generalizes
from a same-address `BlockWrite(1)` to `WRITE32`. It does not identify a
physical carrier, distinguish firmware-handler state from transaction/MMIO/NoC
state, test another address, locate the zero-to-sixteen threshold, or license
an emulator timing-model change. Address locality and distance thresholds stay
deferred until this result is reviewed.

## Completion Receipt

The source-derived generator and classifier landed at `54dc71bd`; the exact
signed-firmware qualification guard landed at `6141d32f`. Both physical arms
repeated `246,248` in order `A/B/B/A`, restored the ordinary workload at the
reported default `600/1028 MHz` identity, left the NPU unowned, and produced no
kernel warning or error. Full interpretation and pins are recorded in the
linked finding.
