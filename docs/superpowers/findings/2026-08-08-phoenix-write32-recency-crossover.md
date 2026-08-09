# Phoenix Same-address WRITE32 Recency Crossover

## Verdict

At this pinned Phoenix signed-firmware boundary, moving a same-address,
same-value `WRITE32` across sixteen `NOOP`s did not reproduce the cold
phase-44 endpoint caused by an immediately recent `BlockWrite(1)`. Both
WRITE32 arms measured the unchanged 248-cycle endpoint:

| Arm | Deliberate order before target | Phase 40 | Phase 44 |
|---|---|---:|---:|
| A | `WRITE32`, sixteen `NOOP`s | 246 | 248 |
| B | sixteen `NOOP`s, `WRITE32` | 246 | 248 |

Physical order `A/B/B/A` repeated every interval digit-for-digit. The
preregistered classifier returned qualified `write32_recency_invariant`, with
a target delta of zero.

The earlier BlockWrite order discriminator changed the same target from 248 to
232 cycles when its one-word BlockWrite became recent. This crossover proves
that target-relative proximity, destination address, written value, and final
register effect are not sufficient by themselves. The transition requires
something specific to the BlockWrite operation, firmware handler, or
lower-level BlockWrite transaction path at this boundary. It does not identify
which of those carriers is responsible.

## Balanced candidate proof

Both candidates were generated at clean worktree commit
`6141d32f6867e5bafdd1eba5c86642a55d16be0a`. Each is 1,288 bytes with 78
header instructions, a matching 1,288-byte header size, and 46 inspected tail
records. Their phase-40 predecessors are byte-identical at offset 1,000 /
ordinal 44, and their phase-44 targets are byte-identical at offset 1,196 /
ordinal 74.

The only differing region is the contiguous 88-byte, seventeen-record span at
offset 1,084. A contains the derived WRITE32 followed by sixteen `NOOP`s; B
contains the same sixteen `NOOP`s followed by the same WRITE32. The write moves
from offset 1,084 to 1,148, exactly 64 bytes, and remains at phase 60. Bytes
before and after the span are identical. Candidate SHA-256 values are:

- A: `eaf0bd95a02a79c96a52ad4cf4957858dd8bdb7c66b9c93399257ee56bad6341`;
- B: `d459da3bb08b401817d3c2b95507d97769cbcc26128ed999fbfc7737b5a30d96`.

The treatment encoding is derived from mlir-aie commit
`e02fb4024a536b51971a9e5cc74ac923ac2f1052`, whose
`TxnEncoding.h` SHA-256 is
`b3a50cf8d3a34fe586c8b4bc061fe41abd973deadbe86bd3fbdfcc4d6df4f6c2`.
The resolved AM025 database selected unused shim DMA BD14 word zero at
`0x1d1c0`; the source transaction neither writes nor queues it. The layout
receipt SHA-256 is
`4e2326af3c7818191a3cfe9583751c80fda5f524b53adde8e96377c1e8f8e6ca`.

## Signed-firmware admission

Both exact candidates completed through the pinned signed-firmware
`CHAIN_EXEC_NPU` guard with the expected response, output, marker lifecycle,
and two established 47-PC measured BlockWrite windows. Each arm consumed
exactly 402 attempted instructions between windows: 26 identical 14-instruction
`NOOP` handlers and one identical 19-instruction `WRITE32` handler. Every
instruction stayed inside the transaction function, with no idle wait,
context transfer, or scheduler boundary.

The qualification receipt and guard-log SHA-256 values are respectively
`ae253ca5e7e0c47b82a29cb615b4e2925290354f3f6a73b2d6702e78f945d88f`
and `6ad1f340dd5108783d47370af450366d0ed7066bbb85d19bf26c08c606a3dad0`.

## Physical capture

Every run reproduced expected output, retained the low-QoS `400/800 MHz`
identity before and after dispatch, and contained both shim DMA lifecycle
witnesses:

| Ordinal | Arm | Marker timestamps | Intervals |
|---:|---|---|---|
| 1 | A | `357941,358187,360141,360389` | `246,248` |
| 2 | B | `357929,358175,360119,360367` | `246,248` |
| 3 | B | `365297,365543,367487,367735` | `246,248` |
| 4 | A | `378239,378485,380439,380687` | `246,248` |

The ordinary no-QoS restoration reproduced expected output and the original
reported default `600/1028 MHz` identity. `/dev/accel/accel0` was unowned
afterward, and the campaign-scoped kernel warning/error query was empty.

## Provenance

The physical tuple is Phoenix firmware `1.5.5.391`, kernel `7.1.7-custom+`,
and loaded driver SHA-256
`21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`.
The trace-classifier cdylib SHA-256 is
`8a115c484af5074b55849391a18dbd8c8708c16bdf64d7591581ef68068193c9`.
The complete harness, candidates, qualification and layout receipts,
provenance, traces, outputs, logs, restoration, and decision are preserved
under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260809T033021Z-firmware-write32-recency-crossover/
```

The final harness and probe receipt SHA-256 values are respectively
`29c3851d9a26226942f32fc5fa85ef39e2ffeb9babd1e49c0c0f5393cdd15bb4`
and `7fa42bfb50990dd952f7c8553c700fc7d8db8bec92ee8dca566cdc2aa11a2926`.

## Licensed conclusions

1. At fixed target phase, bytes, offset, ordinal, predecessor, address, value,
   and final register effect, relocating a same-address WRITE32 by sixteen
   records does not change the 248-cycle target.
2. The previously observed 232-cycle cold endpoint is not explained by recent
   proximity, an ordinary write to BD14, or address/value state alone.
3. The BlockWrite command class, its firmware handler, or a lower-level
   BlockWrite-specific transaction effect is necessary at this boundary.
4. The result does not distinguish those three possibilities, generalize to a
   second address or payload length, locate the recency threshold, or identify
   a cache, MMIO, or NoC carrier.
5. No emulator timing-model change is licensed.

The next useful experiment, if pursued, should keep the BlockWrite path and
separate address-local state from command-class recency. Payload length and the
zero-to-sixteen threshold remain separate boundaries.
