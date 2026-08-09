# Phoenix BlockWrite NOOP/PREEMPT(0) Path Discriminator

## Verdict

At this pinned Phoenix signed-firmware boundary, the recent-history state that
raises the phase-44 `BlockWrite(1)` target from 232 to 248 shim cycles is
invariant between equal-size transaction opcode-5 `NOOP` and opcode-6
`PREEMPT(0)` records:

| Arm | Sixteen-record placement and type | Phase 40 | Phase 44 |
|---|---|---:|---:|
| A | opcode-5 `NOOP` before phase 40 | 246 | 232 |
| B | opcode-5 `NOOP` before phase 44 | 246 | 248 |
| C | opcode-6 `PREEMPT(0)` before phase 44 | 246 | 248 |

The physical order `A/B/C/C/B/A` repeated every interval digit-for-digit. The
preregistered fail-closed classifier returned qualified
`record_path_invariant` for target tuple `(232,248,248)`.

This label is narrow. It proves that the opcode-5-specific handler path is not
necessary to reproduce the higher-cost state at this target. It does not prove
that all transaction records are equivalent, identify elapsed byte distance as
the carrier, or identify a firmware, cache, transaction-engine, MMIO, or NoC
state.

## Source-derived candidates

The current mlir-aie tree at commit
`e02fb4024a536b51971a9e5cc74ac923ac2f1052` defines opcode 6 as a one-word
`PREEMPT`, encodes its level in bits 8 through 15, and documents level 0 as
`Noop`. The exact derived record was `06 00 00 00`. The pinned source hashes
are:

- `TxnEncoding.h`:
  `b3a50cf8d3a34fe586c8b4bc061fe41abd973deadbe86bd3fbdfcc4d6df4f6c2`;
- `AIEX.td`:
  `217f84653d084e1fea71eaf9e406eda416028dc238bc062eca3f7efdc7fd0cee`.

The byte-derived receipt proves that all three candidates have 1,224 bytes, 67
header instructions, a matching 1,224-byte header size, and 35 records in the
inspected post-TCT region. Both measured `BlockWrite(1)` records retain phases
`(40,44)` and identical bytes and payloads. The target is identical across all
arms at byte offset 1,132, global record ordinal 63, with three records
remaining.

The deliberate A history occupies offsets 976 through 1,036 at four-byte
spacing. B and C use the identical offsets 1,044 through 1,104. B and C differ
at exactly those sixteen opcode bytes, from 5 to 6; all level and reserved bytes
are zero. Removing the respective 64-byte deliberate turns makes A and B
byte-identical. Phase 40 moves from offset 1,064 / ordinal 60 in A to offset
1,000 / ordinal 44 in B and C, as required by that relocation.

Candidate SHA-256 values are:

- A: `7bb32a814d6e4ec3b66738827ddf3cefe0619b56cc9884fcb3ed06cb58b005f7`;
- B: `c1de111d3add28ea5a279754cddc93853b8e87627e84e2a261c1783ba6cb6c20`;
- C: `7958248009754504a84a8944bb9c2eafcd0385d57ae92874263ef94bc44293de`.

The layout receipt SHA-256 is
`9e9521b9f9c425c33d93f58aa73733b2d0998259de40f826d8d723a21bd697b9`.

## Signed-firmware admission

All three exact candidates completed through the pinned signed-firmware
`CHAIN_EXEC_NPU` guard with the expected response, output, marker lifecycle,
and two identical 47-PC `BlockWrite(1)` windows. Their packed little-endian-u32
path SHA-256 remains
`4f9c9139aaa9f3c9ac150e5108f41cd3371f34a76f973a35a7f8633b612a2ef6`.

The inter-window firmware path was exact and finite:

- A: 19 shared envelope instructions;
- B: that envelope plus sixteen identical 14-instruction opcode-5 iterations,
  for 243 instructions;
- C: that envelope plus sixteen identical 15-instruction opcode-6 iterations,
  for 259 instructions.

The sole per-record addition in C was PC `0x08b0f76f`. Every instruction stayed
inside the transaction function, and step/instruction deltas proved that no
idle wait or scheduler boundary occurred. Thus Phoenix firmware accepts
transaction-stream `PREEMPT(0)` and consumes it without an observed fault,
yield, task transfer, or preemption lifecycle effect.

The qualification receipt SHA-256 is
`4a77efed206d3f9bed6bb0a76caeafc4119c138b531ba4ea4a2ade77e6258a1f`.
The exact guard log SHA-256 is
`9e48664d7ed7bc6b1a0c9b10ccb197b6169be6aa248ee778dc5a59a555bcbfc8`.

## Physical capture

Every final run reproduced expected output, retained the low-QoS `400/800 MHz`
identity before and after dispatch, and contained both shim DMA lifecycle
witnesses:

| Ordinal | Arm | Marker timestamps | Intervals |
|---:|---|---|---|
| 1 | A | `365695,365941,366061,366293` | `246,232` |
| 2 | B | `376085,376331,377499,377747` | `246,248` |
| 3 | C | `357783,358029,359229,359477` | `246,248` |
| 4 | C | `374999,375245,376445,376693` | `246,248` |
| 5 | B | `377511,377757,378925,379173` | `246,248` |
| 6 | A | `379013,379259,379379,379611` | `246,232` |

The differing gaps between windows contain the deliberate history records and
are not interpreted as target costs.

The ordinary no-QoS restoration reproduced expected output and the original
reported default `600/1028 MHz` identity. `/dev/accel/accel0` was unowned
afterward, and the campaign-scoped kernel warning/error query was empty.

## Provenance and preserved failures

The physical tuple is Phoenix firmware `1.5.5.391`, kernel `7.1.7-custom+`, and
loaded driver SHA-256
`21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`.
The successful run used mlir-aie's `ironenv` Python and explicitly loaded the
worktree classifier cdylib with SHA-256
`8a115c484af5074b55849391a18dbd8c8708c16bdf64d7591581ef68068193c9`.

Three stopped launches are retained as excluded diagnostics:

1. the first stopped before device access because the new harness referenced
   the ownership helper at the wrong import layer;
2. the second stopped before the campaign when the escalated shell selected a
   Miniforge Python without NumPy; its ordinary restoration passed; and
3. the third completed prime, A, and B but aborted before dispatching C because
   ambient `/opt` resolved a stale main-checkout `libxdna_emu.so`. Its restore
   also passed. The final harness pins the rebuilt worktree cdylib directly.

None of those partial launches contributes to the classified tuple. Their
logs, partial traces, outputs, probes, and restorations remain beside the final
receipt.

The complete harness, candidates, qualification and layout receipts,
provenance, raw and decoded traces, outputs, logs, excluded attempts,
restoration, and decision are preserved under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260809T003243Z-firmware-blockwrite-noop-preempt-path/
```

The final harness and probe receipt SHA-256 values are respectively
`920fec16c5805b25c27014196ac2fbf6223ce9d5cbeb9a0999c6936408a0f411`
and `5203cf18d0426a905756b77b734d1db2dafffafa1c36d04762c52ca46833fc8f`.
The required library verification passed 505 tests with 2 ignored, then 4,343
tests with 33 ignored. Its log SHA-256 is
`cd008110b26112aa6835c38befbef140e4faf0bc29471296ba7afb3dbe0fc6cb`.

## Licensed conclusions

1. Pinned Phoenix firmware accepts a one-word transaction-stream
   `PREEMPT(0)` and completes normally without an externally observed
   preemption side effect.
2. The opcode-5-specific handler path is not necessary for the higher-cost
   recent-history state: an equal-size opcode-6 level-0 turn reproduces the
   same 248-cycle target.
3. The signed-firmware interpreter requires one additional instruction per
   `PREEMPT(0)` record, yet physical C did not change the target relative to B.
   That argues against a simple total-management-instruction-count carrier at
   this boundary, but no physical management-PC trace directly proves the
   instruction count on silicon.
4. The evidence remains compatible with state caused by a shared handler path,
   equal-size record traversal, target-relative byte/record distance, or a
   lower-level transaction/MMIO/NoC mechanism. It does not distinguish them.
5. No general opcode equivalence, nonzero preemption behavior, threshold,
   scheduler cost, or emulator timing-model change is licensed.

The next useful experiment, if pursued, must be designed separately. The
deferred `BlockWrite` re-prime boundary remains one candidate, but this result
does not select it automatically.
