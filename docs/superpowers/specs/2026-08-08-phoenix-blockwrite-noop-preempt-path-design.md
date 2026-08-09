# Phoenix BlockWrite NOOP/PREEMPT(0) Path Discriminator

**Status:** Approved experiment boundary. No implementation or hardware access
has occurred.

## Question

The balanced relocation and depth campaigns proved that the pinned Phoenix
signed-firmware `BlockWrite(1)` target costs 232 shim cycles with no recent CDO
`NOOP` turn and 248 cycles with either sixteen or thirty-two recent `NOOP`
records. They did not distinguish state caused specifically by the firmware's
`NOOP` handler path from state caused by traversing any equal-length recent
transaction records.

At fixed predecessor phase, target phase, record count, byte distance, and
target identity, does replacing the recent `NOOP` turn with sixteen
toolchain-defined `PREEMPT(0)` records preserve the 248-cycle target?

## Source-Derived Primitive

The current mlir-aie transaction encoder defines opcode 6 as `PREEMPT`, emits
it as one 32-bit word, and encodes the level in bits 8 through 15. Its AIEX
operation documentation defines level 0 as `Noop`. Therefore the only admitted
replacement word is:

```text
06 00 00 00
```

The implementation and receipt must derive and pin this encoding from the
resolved mlir-aie tree. It must not copy the literal into the candidate
generator without checking the source definition.

Transaction-stream `PREEMPT(0)` is not the driver command-slot
`EXEC_NPU_TYPE_PREEMPT` feature. Phoenix's lack of the latter neither admits nor
rejects the former. Acceptance and observable no-op behavior of the exact
transaction record must be established by the pinned signed firmware before
physical access.

The repository currently walks transaction opcode 6 as a conservative 16-byte
record in both the Rust parser and trace patcher, which disagrees with the
current toolchain's one-word encoding. Correct that shared format fact, with a
sentinel-following parser test, before generating candidates. Do not broaden
this experiment into modeling nonzero preemption levels.

## Balanced Arms

Reuse the established two-window `BlockWrite(1)` instrument at phases
`(40,44)`. Each arm contains one deliberate sixteen-record, 64-byte history
turn in addition to the ordinary alignment records:

- **A -- cold control:** sixteen opcode-5 `NOOP`s before the phase-40
  predecessor and no deliberate records between phase 40 and phase 44.
- **B -- hot control:** no deliberate records before phase 40 and sixteen
  opcode-5 `NOOP`s immediately before the phase-44 target.
- **C -- path substitution:** no deliberate records before phase 40 and
  sixteen opcode-6 level-0 `PREEMPT` records at the exact offsets occupied by
  B's recent `NOOP` turn.

The emitted-byte receipt must prove:

1. identical transaction header count and total size, candidate byte length,
   and total record count across A, B, and C;
2. identical phase-40 and phase-44 `BlockWrite(1)` bytes, payloads, absolute
   offsets, record ordinals, phases modulo 64, and remaining-record counts;
3. the same sixteen four-byte history-record offsets in B and C;
4. B and C differ at exactly those sixteen opcode bytes, from 5 to 6, with all
   level and reserved bytes zero; and
5. A and B contain the same sixteen deliberate `NOOP` records, relocated only
   across the phase-40 predecessor.

Any mismatch stops before signed-firmware execution.

## Signed-Firmware Admission

Run all three exact candidates through the pinned signed-firmware
`CHAIN_EXEC_NPU` guard before touching hardware. A and B must reproduce the
known command response, expected kernel output, two complete marker windows,
and identical normalized 47-PC `BlockWrite(1)` path in both windows.

Candidate C is admitted only if:

- the command completes normally with the same response and expected output;
- both marker windows complete and the target follows the same normalized
  47-PC `BlockWrite(1)` path as A and B;
- all sixteen `PREEMPT(0)` records are consumed through a finite, repeatable
  firmware path and execution reaches the next record normally; and
- the available trace and lifecycle state show no fault, idle wait, context or
  task transfer, scheduler boundary, or other preemption side effect.

If the pinned firmware rejects opcode 6, treats level 0 as a yield, exposes an
ambiguous path, or changes lifecycle state, record that result and stop. Such a
failure means this equal-size discriminator is unavailable; it is not
permission to send the candidate to physical hardware.

## Physical Campaign

After admission, run fresh physical contexts in symmetric order
`A/B/C/C/B/A` under the pinned low-QoS `400/800 MHz` identity. Every run must:

- reproduce expected output and the ordinary command response;
- preserve the clock identity before and after dispatch;
- contain both shim DMA lifecycle witnesses;
- expose exactly two alternating marker pairs at phases `(40,44)`; and
- leave the device unowned after the run.

Both repeats of each arm must agree digit-for-digit. Every phase-40 predecessor
must be exactly 246 cycles. Arm A's phase-44 target must reproduce 232 cycles,
and arm B's must reproduce 248 cycles. Failure of either control or any recent
kernel warning/error makes the campaign unqualified.

## Fail-Closed Classifier

Preserve every exact interval. Once all controls and repeats qualify, classify
the phase-44 target tuple `(A,B,C)` only as follows:

| Target tuple | Narrow result |
|---|---|
| `(232,248,232)` | `noop_opcode_path_sensitive` |
| `(232,248,248)` | `record_path_invariant` |

`noop_opcode_path_sensitive` means the equal-size level-0 path did not produce
the recent-`NOOP` state at this target. It supports, but does not prove, a
NOOP-handler or code-path carrier. `record_path_invariant` means this boundary
did not distinguish the two recent record paths; it does not prove elapsed
distance is the carrier or that the paths are internally identical.

Any other tuple, changed predecessor, nondeterminism, malformed run, admission
failure, or restore failure is `unclassified`. No alternate model may be fit to
an unregistered result.

## Tests, Provenance, and Restoration

Tests come first for the one-word opcode-6 parser boundary, source-derived
encoding, three-arm byte invariants, and fail-closed classifier. A parser test
must place a known record immediately after `PREEMPT(0)` and prove that it
remains aligned. Run the focused Python suite and `nice -n 19 cargo test --lib`
before any hardware access.

Pin the clean worktree commit, kernel, loaded driver, firmware payload, resolved
mlir-aie commit and relevant source hashes, source transaction, XCLBIN,
expected output, runner, clock query, exact candidates, layout receipt,
signed-firmware qualification, and reused harness. Preserve candidates, raw
and decoded traces, outputs, logs, receipts, and the decision under
`build/experiments/phoenix-pm-clock-characterization/`.

A `finally` path runs the ordinary source transaction without QoS, verifies its
output and original default clock identity, records device ownership and recent
kernel warnings/errors, and blocks every timing conclusion if restoration
fails.

## Completion Boundary

This experiment may distinguish one narrow recent-history mechanism. It does
not locate a threshold, identify a physical cache or state carrier, license a
scheduler or timing-cost change, exercise nonzero preemption, or begin the
deferred `BlockWrite` re-prime experiment. Any next boundary is designed only
after reviewing the complete receipt.
