# Phoenix BlockWrite Address Crossover

**Status:** Complete; physical result qualified `blockwrite_address_invariant`.

## Question

At the pinned Phoenix signed-firmware boundary, must the immediately recent
one-word `BlockWrite` target the same shim-DMA register as the measured
phase-44 `BlockWrite(1)`, or is a recent equivalent `BlockWrite(1)` to another
unused descriptor sufficient to reproduce the 232-cycle endpoint?

The completed order discriminator measured 232 cycles when a one-word BD14
`BlockWrite` immediately preceded the target and 248 cycles when that same
operation preceded sixteen `NOOP`s. The completed WRITE32 crossover then
showed that an immediately recent same-address `WRITE32` leaves the target at
248 cycles. This boundary retains the BlockWrite command and handler while
changing only its destination address.

## Source-Derived Alternate Address

Use shim DMA BD13 word zero as the alternate and retain BD14 word zero as the
measured target. Derive both addresses and descriptor geometry from the
resolved AM025 register database; do not duplicate either offset in campaign
code.

The safety proof must establish from the resolved toolchain and exact source
transaction that:

1. aie-rt describes the shim DMA as sixteen descriptors with an eight-word,
   `0x20`-byte stride and no descriptor/channel restriction;
2. AM025 gives BD13 and BD14 the same eight-word field schema at adjacent
   descriptor offsets;
3. the source transaction does not read, write, mask-write, patch, or otherwise
   overlap either descriptor;
4. no source task-queue write starts BD13 or BD14; and
5. no source-configured descriptor explicitly reaches BD13 or BD14 through an
   enabled `Next_BD` link.

All register names, bit positions, descriptor counts, and link fields are
resolved from aie-rt or AM025. Unknown or partially analyzable source writes
that could affect the safety proof fail closed. BD15 is not an admissible
alternate: the pinned source transaction writes and queues it.

Strengthen the existing shared unused-BD helper to perform this proof for a
requested descriptor ID while retaining BD14 as its default. Existing callers
must keep their current behavior.

## Two Balanced Arms

Reuse phases `(40,44)` and the exact recent-treatment layout from the qualified
BlockWrite re-prime campaign:

- **A -- same-address control:** sixteen `NOOP`s, one zero-valued payload word
  `BlockWrite(1)` to BD14, then the fixed BD14 target;
- **B -- alternate-address treatment:** the same sixteen `NOOP`s, one
  zero-valued payload word `BlockWrite(1)` to BD13, then the same fixed BD14
  target.

Arm A must be byte-for-byte identical to the previously qualified recent-BD14
candidate. Arm B differs from A only in the treatment record's four-byte
address field. Both treatment records retain the same phase, payload, length,
firmware opcode, target-relative distance, and record ordinal. The phase-40
predecessor and phase-44 target remain byte-identical at the same absolute
offsets and ordinals.

An emitted-byte receipt must prove:

1. identical candidate lengths, header counts and sizes, operation counts, and
   tail-record counts;
2. an exact prior-candidate hash match for Arm A;
3. one treatment `BlockWrite(1)` at the same offset and phase in each arm;
4. sixteen contiguous, byte-identical `NOOP`s immediately before each
   treatment;
5. identical phase-40 and phase-44 measured records, marker adjacency, and
   remaining records; and
6. byte identity outside the treatment address field, whose two values equal
   the independently AM025-derived BD14 and BD13 addresses.

Any mismatch stops before signed-firmware execution.

This direct address substitution is preferred over a symmetric two-address
swap. The latter would add a second BlockWrite handler and either replace part
of the already-qualified sixteen-NOOP history or move the target, introducing
an unnecessary causal variable. A cross-tile address remains a later boundary
because it would also change tile and NoC routing.

## Signed-Firmware Admission

Run both exact candidates through the pinned signed-firmware `CHAIN_EXEC_NPU`
guard before physical access. Both must reproduce the expected command
response, kernel output, marker lifecycle, and established normalized 47-PC
paths for the phase-40 predecessor and phase-44 target.

Between the measured windows, both arms must execute the same finite sequence:
the unchanged source handlers, sixteen `NOOP` handlers, and one one-word
BlockWrite handler. Their attempted-instruction totals and normalized PC paths
must match exactly. Neither arm may enter an idle wait, fault, scheduler,
context-transfer, or other lifecycle boundary. Address-dependent control flow
or instruction count makes the candidates unqualified rather than becoming a
timing result.

Admission proves candidate integrity only. Emulator instruction counts are not
asserted to equal silicon instruction counts.

## Physical Campaign

Run fresh physical contexts in symmetric order `A/B/B/A` under the pinned
low-QoS `400/800 MHz` identity. Every run must:

- reproduce the expected output and ordinary command response;
- preserve the clock identity before and after dispatch;
- contain both shim-DMA lifecycle witnesses;
- expose exactly two alternating marker pairs at phases `(40,44)`; and
- leave the device unowned afterward.

Both repeats of each arm must agree digit-for-digit. Every phase-40 predecessor
must remain exactly 246 cycles, and Arm A's phase-44 target must reproduce the
established 232-cycle same-address control. A changed control, malformed trace,
clock drift, campaign-scoped kernel warning or error, or failed restoration
makes the campaign unqualified.

## Fail-Closed Classifier

After all controls and repeats qualify, classify Arm B's exact phase-44 target:

| B target | Narrow result |
|---:|---|
| `232` | `blockwrite_address_invariant` |
| `248` | `blockwrite_target_address_required` |
| any other repeatable positive integer | `blockwrite_alternate_address_other` |

`blockwrite_address_invariant` means a recent adjacent-descriptor BlockWrite is
sufficient at this boundary, rejecting a state strictly local to BD14 word
zero. `blockwrite_target_address_required` means the known transition requires
the treatment to address BD14 at this boundary. It does not prove where that
address sensitivity resides. The third result records a new deterministic
address-sensitive state without fitting or naming a mechanism.

Nonrepeat, control failure, malformed evidence, address-dependent firmware
paths, clock drift, or failed restoration is `unclassified`. No alternate
model may be fitted to an unqualified result.

## Tests, Provenance, and Restoration

Tests come first for generic unused-BD derivation, direct use, queue use,
enabled `Next_BD` links, unknown link state, byte-level candidate invariants,
the prior Arm-A hash anchor, and every classifier branch. Run the focused
Python characterization suite and `nice -n 19 cargo test --lib` before
hardware access.

Pin the clean worktree commit, kernel, loaded driver, firmware payload, source
transaction, XCLBIN, expected output, runner, clock query, exact candidates,
layout receipt, signed-firmware qualification, resolved aie-rt and mlir-aie
commits and source hashes, and worktree FFI library. Preserve candidates,
traces, outputs, logs, receipts, and the decision under
`build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output and original default clock identity, records ownership and recent kernel
warnings or errors, and blocks every timing conclusion if restoration fails.

## Completion Boundary

This experiment may determine whether the known recent-BlockWrite transition
requires the treatment to address the target register or generalizes to an
adjacent, structurally equivalent unused descriptor. It does not distinguish
firmware-handler state from transaction-engine, MMIO, cache, or NoC state;
generalize beyond this descriptor pair; locate the recency threshold; test
payload length; or license an emulator timing-model change.

## Completion Receipt

The exact `A/B/B/A` physical campaign qualified both repeated endpoints as
`246/232` cycles, restored the original no-QoS state, left the device unowned,
and produced no campaign-scoped kernel warning. The complete evidence and
licensed conclusion are recorded in
[`2026-08-09-phoenix-blockwrite-address-crossover.md`](../findings/2026-08-09-phoenix-blockwrite-address-crossover.md).
