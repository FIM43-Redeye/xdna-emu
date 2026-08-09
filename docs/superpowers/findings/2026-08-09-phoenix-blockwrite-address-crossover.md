# Phoenix BlockWrite Address Crossover

## Verdict

At this pinned Phoenix signed-firmware boundary, an immediately recent
one-word `BlockWrite` to adjacent unused shim DMA BD13 reproduces the same
phase-44 endpoint as an otherwise identical `BlockWrite` to the target BD14
word itself:

| Arm | Immediately recent treatment | Phase 40 | Phase 44 |
|---|---|---:|---:|
| A | `BlockWrite(1)` to BD14 | 246 | 232 |
| B | `BlockWrite(1)` to BD13 | 246 | 232 |

Physical order `A/B/B/A` repeated every interval digit-for-digit. The
preregistered fail-closed classifier returned qualified
`blockwrite_address_invariant`, with a target delta of zero.

The known 232-cycle transition therefore does not require the recent
BlockWrite to address the measured BD14 word itself. This rejects state
strictly local to BD14 word zero as a necessary carrier at this boundary. It
does not prove that address is irrelevant across modules or tiles: BD13 and
BD14 are adjacent descriptors in the same shim DMA register bank.

## Balanced candidate proof

Both candidates were generated at clean worktree commit
`2e83f6714ec9416b519271c8edc6d7aa667f6781`. Each is 1,288 bytes with 79
header instructions, a matching 1,288-byte header size, and 47 inspected tail
records. Arm A is byte-for-byte identical to the previously qualified recent
BD14 candidate. Candidate SHA-256 values are:

- A: `e710a5a499496ddb4dbf041ebe0b3fad7a067b8e13ff51b230410281a57a456b`;
- B: `e6c75efaf8a81481c29534c79c3ab911ae00dfa96d8f6d953645143aea471632`.

Both arms contain sixteen contiguous `NOOP`s followed by the treatment at
offset 1,152 / ordinal 73 / phase zero. Their phase-40 predecessors are
byte-identical at offset 1,000 / ordinal 44, and their phase-44 targets are
byte-identical at offset 1,196 / ordinal 75. Bytes are identical everywhere
except offset 1,160, inside the treatment's four-byte address field:

- A addresses AM025-derived BD14 word zero at `0x1d1c0`;
- B addresses AM025-derived BD13 word zero at `0x1d1a0`.

The resolved AM025 database gives both descriptors the same eight-word field
schema. The generalized source-transaction audit proves that neither
descriptor is directly accessed, queued, or reached through an enabled
`Next_BD` link; unknown link state fails closed. The layout receipt SHA-256 is
`7544cc44c8f2b22c894c056387c86e95d66fcf63760060b22dec45f20cc77ab2`.

The independent aie-rt source audit uses commit
`6ee6a4da5f55bd66d278d5032108f0ebe920a501`: its shim DMA module describes 16
descriptors, an `0x20`-byte stride, eight words per descriptor, and no
descriptor/channel restriction. The pinned module-definition and validity
source SHA-256 values are respectively
`f508831031d3550bc88c839744a1170981e0263eaa6f8ded55b9b2aad78e39ff` and
`7b4dbf7022499e895a945391a031d20022df50bcf640e998f62cb76ff0967bd5`.
AM025 came from mlir-aie commit
`e02fb4024a536b51971a9e5cc74ac923ac2f1052`, with register-database SHA-256
`c5a40ea762f70a5d2728d63370a8ad66ae88d3420c5887604491c4cec9b55396`.

## Signed-firmware admission

Both exact candidates completed through the pinned signed-firmware
`CHAIN_EXEC_NPU` guard with the expected response, output, marker lifecycle,
and established 47-PC phase-40 and phase-44 BlockWrite paths. Each arm consumed
exactly 425 attempted instructions between measured windows: twenty-seven
`NOOP` handlers and one BlockWrite handler. Their normalized paths and counts
matched exactly, with no idle wait, context transfer, scheduler, or fault
boundary.

The qualification receipt and successful guard-log SHA-256 values are
respectively
`bb5af1a3535b4f33338a6d9402195edfdada31ddc513b132f095617ec3485ccd` and
`bbb4286f719419c994cd640187d8e88f456e35ed617acc7d8bc83715d884dc43`.
The full library verification passed 505 tests with 2 ignored, then 4,346
tests with 33 ignored; its log SHA-256 is
`652c6f1713b53671a5d2404e2eee04ba1235a96d2f9ea4827719595b09456cdd`.

## Physical capture

Every run reproduced expected output, retained the low-QoS `400/800 MHz`
identity before and after dispatch, and contained both shim DMA lifecycle
witnesses:

| Ordinal | Arm | Marker timestamps | Intervals |
|---:|---|---|---|
| 1 | A | `382763,383009,385043,385275` | `246,232` |
| 2 | B | `359821,360067,362101,362333` | `246,232` |
| 3 | B | `351047,351293,353327,353559` | `246,232` |
| 4 | A | `357589,357835,359869,360101` | `246,232` |

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
  20260809T050616Z-firmware-blockwrite-address-crossover/
```

The final harness and probe receipt SHA-256 values are respectively
`4a3ce7c935e0c1f2f8456c19e609a203d2c2ee6a229d1bfd142310b32dabc506`
and `b6c6cc6941ced53f33152ca282cfaa778198581c9ce48135e58ed19878df0ee7`.

## Licensed conclusions

1. At fixed target phase, bytes, offset, ordinal, predecessor, handler path,
   payload length, and target-relative distance, substituting adjacent unused
   BD13 for target BD14 in the immediately recent BlockWrite leaves the target
   at exactly 232 cycles.
2. A state update strictly local to BD14 word zero is not necessary for the
   known recent-BlockWrite transition.
3. Combined with the WRITE32 crossover, the evidence requires a recent
   BlockWrite-class path but not a recent write to the target word itself.
4. The result remains compatible with shared firmware-handler state and with
   transaction-engine, register-bank, cache-line, MMIO, or NoC state shared by
   the adjacent addresses. It does not generalize across modules or tiles.
5. No payload-length law, recency threshold, physical carrier, or emulator
   timing-model change is licensed.

A later discriminator may change payload length or move the alternate address
outside this shim DMA register bank. Those are separate causal boundaries.
