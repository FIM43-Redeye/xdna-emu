# Phoenix BlockWrite Re-prime Order Discriminator

**Status:** Completed and physically qualified; see
[`2026-08-08-phoenix-blockwrite-reprime-order.md`](../findings/2026-08-08-phoenix-blockwrite-reprime-order.md).

## Question

At the pinned Phoenix signed-firmware boundary, sixteen recent one-word
transaction records raise the phase-44 `BlockWrite(1)` target from the known
232-cycle cold endpoint to 248 shim cycles. Does executing one identical,
harmless `BlockWrite(1)` after that sixteen-record turn change the target state?

This experiment tests operation order. It does not identify a cache, firmware,
transaction-engine, MMIO, or NoC carrier by itself.

## Source-Derived Re-prime

Reuse the existing AM025-derived unused shim DMA BD14 seam. The source
transaction must neither write nor queue BD14. The re-prime is the same
20-byte `BlockWrite(1)` used by the measured target: one zero payload word to
BD14, with its opcode, address, size, and tile fields emitted through the
existing transaction helpers.

No alternate address, invented filler record, or hardcoded register offset is
admitted. If the resolved register database or source transaction cannot prove
BD14 safe, candidate generation fails.

## Balanced Candidates

Reuse the established phase-40 predecessor and phase-44 measured target. After
the phase-40 stop marker, both candidates contain the same re-prime and the
same sixteen-record opcode-5 `NOOP` turn before the target start marker:

- **A -- hot control:** re-prime, sixteen `NOOP`s, target;
- **B -- re-prime treatment:** sixteen `NOOP`s, re-prime, target.

The `NOOP` turn is exactly 64 bytes. Moving the 20-byte re-prime across it
therefore preserves the re-prime's transaction phase. Each deliberate
inter-window sequence is 84 bytes and seventeen records.

An emitted-byte receipt must prove:

1. identical transaction-header operation counts and sizes, candidate byte
   lengths, and total record counts;
2. a byte-identical phase-40 predecessor at the same absolute offset and record
   ordinal in both candidates;
3. a byte-identical phase-44 target at the same absolute offset and record
   ordinal, with the same number of records remaining;
4. identical alignment records and marker records;
5. one identical re-prime record at offsets separated by exactly 64 bytes and
   at the same phase modulo 64; and
6. sixteen identical contiguous `NOOP` records whose order relative to the
   re-prime is the only candidate difference.

Any mismatch stops before signed-firmware execution.

## Signed-Firmware Admission

Run both exact candidates through the pinned signed-firmware `CHAIN_EXEC_NPU`
guard before physical access. Both must:

- complete with the same command response and expected kernel output;
- expose the same two complete measured marker windows;
- follow the established normalized 47-PC path for the phase-40 predecessor
  and phase-44 target;
- consume exactly one finite re-prime `BlockWrite(1)` path and sixteen finite
  `NOOP` paths between the measured windows;
- have equal total inter-window attempted-instruction counts; and
- show no fault, idle wait, task or context transfer, scheduler boundary, or
  other lifecycle change.

The inter-window dynamic order may differ only as required by the candidate
order. Admission does not claim that emulator instruction counts equal silicon
instruction counts.

## Physical Campaign

Run fresh physical contexts in symmetric order `A/B/B/A` under the pinned
low-QoS `400/800 MHz` identity. Every run must:

- reproduce the expected output and ordinary command response;
- preserve the clock identity before and after dispatch;
- contain both shim DMA lifecycle witnesses;
- expose exactly two alternating marker pairs at phases `(40,44)`; and
- leave the device unowned afterward.

Both repeats of each arm must agree digit-for-digit. Every phase-40 predecessor
must be exactly 246 cycles, and A's phase-44 target must reproduce 248 cycles.
A changed control, campaign-scoped kernel warning or error, malformed trace,
or failed restoration makes the campaign unqualified.

## Fail-Closed Classifier

Once the controls and repeats qualify, classify B's exact phase-44 target:

| B target | Narrow result |
|---:|---|
| `248` | `blockwrite_reprime_invariant` |
| `232` | `blockwrite_reprime_matches_cold` |
| any other repeatable positive integer | `blockwrite_reprime_changes_target` |

`blockwrite_reprime_invariant` means the intervening identical BlockWrite did
not change the known higher-cost state at this boundary.
`blockwrite_reprime_matches_cold` means it changed the target to the previously
reproduced cold endpoint. `blockwrite_reprime_changes_target` records a new
exact deterministic state without fitting or naming its mechanism.

Nonrepeat, control failure, malformed inputs, clock drift, or restoration
failure is `unclassified`. No alternate model may be fitted to an unqualified
result.

## Tests, Provenance, and Restoration

Tests come first for candidate generation, byte-level invariants, invalid BD14
sources, and every classifier branch. Run the focused Python suite and
`nice -n 19 cargo test --lib` before hardware access.

Pin the clean worktree commit, kernel, loaded driver, firmware payload, resolved
mlir-aie commit and relevant source hashes, source transaction, XCLBIN,
expected output, runner, clock query, exact candidates, layout receipt,
signed-firmware qualification, and worktree FFI library. Preserve candidates,
raw and decoded traces, outputs, logs, receipts, and the decision under
`build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output and original default clock identity, records device ownership and recent
kernel warnings or errors, and blocks every timing conclusion if restoration
fails.

## Completion Boundary

This experiment may prove whether one recent, identical BlockWrite changes the
saturated record-history state. It does not distinguish operation identity
from the resulting 20-byte change in distance to the sixteen-record turn,
identify a physical carrier, locate the zero-to-sixteen threshold, generalize
to other addresses or opcodes, or license an emulator scheduler or timing-model
change. Address-locality and threshold experiments remain separate decisions.
