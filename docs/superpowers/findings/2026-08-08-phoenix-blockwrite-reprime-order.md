# Phoenix BlockWrite Re-prime Order Discriminator

## Verdict

At this pinned Phoenix signed-firmware boundary, an immediately recent,
identical zero-payload `BlockWrite(1)` returns the fixed phase-44 target to its
previously observed 232-cycle cold endpoint. Moving that same re-prime before
sixteen one-word transaction `NOOP`s instead preserves the 248-cycle endpoint:

| Arm | Deliberate order before target | Phase 40 | Phase 44 |
|---|---|---:|---:|
| A | re-prime, sixteen `NOOP`s | 246 | 248 |
| B | sixteen `NOOP`s, re-prime | 246 | 232 |

Physical order `A/B/B/A` repeated every interval digit-for-digit. The
preregistered fail-closed classifier returned qualified
`blockwrite_reprime_matches_cold`, with a signed target delta of `-16` cycles.

This is an operation-order result. It proves that the recent-history state is
updated by the position of an intervening `BlockWrite(1)`; the total record
multiset, bytes, and signed-firmware work count are insufficient. It does not
distinguish operation identity from the resulting target-relative distance or
identify a firmware, cache, transaction-engine, MMIO, or NoC carrier.

## Balanced candidate proof

Both candidates were generated at clean worktree commit
`68d84bec1c717231001144d231186d6a678806b0`. The re-prime uses the existing
AM025-derived unused shim DMA BD14 seam: one zero payload word in the same
20-byte `BlockWrite(1)` form as the measured target. Candidate SHA-256 values
are:

- A: `61c121c2534ce7ca4c54569d36d0b94ccf02a295e2d1cda03f86bab497be55c0`;
- B: `e710a5a499496ddb4dbf041ebe0b3fad7a067b8e13ff51b230410281a57a456b`.

The byte-derived layout receipt proves that both candidates have 1,288 bytes,
79 header instructions, a matching 1,288-byte header size, and 47 records in
the inspected post-TCT region. Their phase-40 predecessors are byte-identical
at offset 1,000 / ordinal 44. Their phase-44 targets are byte-identical at
offset 1,196 / ordinal 75.

The only differing region is the 84-byte, seventeen-record span beginning at
offset 1,088. A contains the re-prime followed by sixteen contiguous `NOOP`s;
B contains the same sixteen `NOOP`s followed by the same re-prime. The re-prime
moves from offset 1,088 to 1,152, exactly 64 bytes, and remains at phase zero.
Bytes before and after that span are identical. The layout receipt SHA-256 is
`62f0e47c691ad4758ed51ebf0d2df3a367317c7e63f780d27feeb9d3f6a5638f`.

## Signed-firmware admission

Both exact candidates completed through the pinned signed-firmware
`CHAIN_EXEC_NPU` guard with the expected response, output, marker lifecycle,
and two established 47-PC measured `BlockWrite(1)` windows. Their packed
little-endian-u32 path SHA-256 remains
`4f9c9139aaa9f3c9ac150e5108f41cd3371f34a76f973a35a7f8633b612a2ef6`.

Each arm consumed exactly 425 attempted instructions between measured windows:

- A: eleven unchanged source `NOOP` handlers, one 28-instruction re-prime
  handler, then sixteen `NOOP` handlers;
- B: the same twenty-seven 14-instruction `NOOP` handlers, then the same
  re-prime handler.

Every instruction stayed inside the transaction function. Step and instruction
deltas prove there was no idle wait or scheduler boundary. The qualification
receipt and guard-log SHA-256 values are respectively
`fd32eef037508bcadbda7fb0a84e43cf3916d3ca8d6505a0439f349b5b07879e`
and `9f4a739699d3814bd4858027821b619f13868e9b9f3b0ce8300669d7ed880066`.

## Physical capture

Every run reproduced expected output, retained the low-QoS `400/800 MHz`
identity before and after dispatch, and contained both shim DMA lifecycle
witnesses:

| Ordinal | Arm | Marker timestamps | Intervals |
|---:|---|---|---|
| 1 | A | `370229,370475,372501,372749` | `246,248` |
| 2 | B | `351161,351407,353441,353673` | `246,232` |
| 3 | B | `352813,353059,355093,355325` | `246,232` |
| 4 | A | `377609,377855,379881,380129` | `246,248` |

The differing gaps between windows contain the deliberate order treatment and
are not interpreted as target costs.

The ordinary no-QoS restoration reproduced expected output and the original
reported default `600/1028 MHz` identity. `/dev/accel/accel0` was unowned
afterward, and the campaign-scoped kernel warning/error query was empty.

## Provenance and preserved preflight

The physical tuple is Phoenix firmware `1.5.5.391`, kernel `7.1.7-custom+`, and
loaded driver SHA-256
`21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`.
The successful run used mlir-aie's `ironenv` Python and the pinned worktree
trace-classifier cdylib SHA-256
`8a115c484af5074b55849391a18dbd8c8708c16bdf64d7591581ef68068193c9`.

The first launch stopped before treatment traffic because the ambient
Miniforge Python lacked NumPy. Its ordinary restore passed, the device was
unowned, and its dmesg window was clean. That receipt and parser log are
preserved under `preflight-failure-miniforge/`; none of it contributes to the
classified tuple.

The complete harness, candidates, qualification and layout receipts,
provenance, raw and decoded traces, outputs, logs, stopped preflight,
restoration, and decision are preserved under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260809T024016Z-firmware-blockwrite-reprime-order/
```

The final harness and probe receipt SHA-256 values are respectively
`0e014fd60295868cc1fd370871a5b9bcab20d6bb80eb47b028509dc93f25f44c`
and `357738131b81483a53213d9ec180babc01957c23c679943a0e18aa1885bfae6d`.
The required library verification passed 505 tests with 2 ignored, then 4,344
tests with 33 ignored. Its log SHA-256 is
`b0655a09a652435915d7092ec8f2912925691d6054c78089af01c00defac6055`.

## Licensed conclusions

1. At fixed target phase, bytes, offset, ordinal, predecessor, total candidate
   size, and operation multiset, the order of an intervening identical
   `BlockWrite(1)` and sixteen `NOOP`s changes the target by exactly 16 shim
   cycles.
2. An immediately preceding re-prime reproduces the known 232-cycle cold
   endpoint; placing sixteen `NOOP`s after it reproduces the known 248-cycle
   endpoint.
3. Both arms execute the same twenty-seven `NOOP` handlers, one re-prime
   handler, and 425 total attempted inter-window firmware instructions. A
   total-work or unordered-record model cannot reproduce both outcomes.
4. The result remains compatible with operation-specific state, distance since
   the re-prime, shared-handler recency, address-local state, or a lower-level
   transaction/MMIO/NoC mechanism. It does not distinguish them.
5. No threshold, address generalization, scheduler cost, or emulator timing
   model is licensed.

The next useful experiment, if pursued, must separately distinguish operation
identity from target-relative distance while retaining the same phase, payload,
and fail-closed controls. Address locality and the zero-to-sixteen threshold
remain separate boundaries.
