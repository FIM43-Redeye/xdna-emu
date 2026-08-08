# Phoenix BlockWrite NOOP History Depth

## Verdict

Phoenix's signed-firmware `BlockWrite(1)` target has a saturating recent-history
effect at this pinned boundary. With phase 40 fixed at 246 cycles, the phase-44
target was 232 cycles with no `NOOP` turn between the windows and 248 cycles
with either one or two turns between them:

| Arm | `NOOP`s before phase 40 | `NOOP`s before target | Phase 40 | Phase 44 |
|---|---:|---:|---:|---:|
| A | 32 | 0 | 246 | 232 |
| B | 16 | 16 | 246 | 248 |
| C | 0 | 32 | 246 | 248 |

The physical order `A/B/C/C/B/A` repeated every interval digit-for-digit. The
fail-closed preregistered classifier returned qualified
`saturating_hot_cold` for target tuple `(232,248,248)`.

This label names the discriminator outcome, not a proven physical carrier. The
result is consistent with a binary or threshold state already reached within
the first sixteen recent `NOOP` records. It rejects the preregistered linear
per-turn outcome `(232,248,264)` and the 32-record periodic recurrence outcome
`(232,248,232)` at this target. It does not locate the threshold within the
first turn or distinguish firmware, instruction-cache, transaction-engine,
MMIO, or NoC state.

## Balanced candidate proof

All three candidates were generated at clean worktree commit
`b7c097531471b814fb9c1e9f75b172bcfdacea0b` by relocating two complete
64-byte turns with `leading_full_turns` `(2,0)`, `(1,1)`, and `(0,2)`.

The byte-derived receipt proves that every arm has:

- 1,288 bytes and a matching header size;
- 83 header instructions;
- 44 `NOOP` records in the inspected instrument region;
- measured `BlockWrite(1)` phases `(40,44)`; and
- an identical phase-44 target at byte offset 1,196, record ordinal 49, with
  identical bytes and 33 records remaining.

Candidate SHA-256 values are:

- A: `8dca5834f84d13bb23f065b23e6798f78128bcae13f146fbf12db65d114ff2cc`;
- B: `0663457f0483419e7456f6e5c74c7cf4c75db20c241f7258ec62597fb7eac596`;
- C: `c922646d4cbb1e718aab72a8f011a2fd85b285efbd47ee42cbb2f3323ad92123`.

The layout receipt SHA-256 is
`83effabb54bc35207c55a275347e514b768dc347d5aa7d7df9a1f60567c96a03`.

## Signed-firmware qualification

All three exact candidates passed the pinned signed-firmware
`CHAIN_EXEC_NPU` guard before physical access. Each exposed one source marker
and two complete measurement pairs. Every one of the six windows consumed 47
attempted firmware instructions, and every window followed one identical
47-PC dynamic sequence. Its packed little-endian-u32 SHA-256 is
`4f9c9139aaa9f3c9ac150e5108f41cd3371f34a76f973a35a7f8633b612a2ef6`.

The qualification receipt SHA-256 is
`47b85db901afdb4add1d4ae078207bad81bc2dd9cc2816c5871bb15b16327aaa`.
Temporary guard edits were removed before hardware access, and the worktree
was clean.

## Physical capture

Every run reproduced expected output, retained the low-QoS `400/800 MHz`
identity before and after dispatch, and contained both shim DMA lifecycle
witnesses:

| Ordinal | Arm | Marker timestamps | Intervals |
|---:|---|---|---|
| 1 | A | `360625,360871,360991,361223` | `246,232` |
| 2 | B | `382823,383069,384237,384485` | `246,248` |
| 3 | C | `362397,362643,364867,365115` | `246,248` |
| 4 | C | `359343,359589,361813,362061` | `246,248` |
| 5 | B | `363777,364023,365191,365439` | `246,248` |
| 6 | A | `365277,365523,365643,365875` | `246,232` |

The differing gaps between windows are the deliberate `NOOP` relocation and
are not interpreted as target cost.

The ordinary no-QoS restoration reproduced expected output and the original
reported default `600/1028 MHz` identity. `/dev/accel/accel0` was unowned
afterward, and the recent kernel warning/error query was empty.

The first launch attempt stopped before candidate dispatch because the
escalated shell selected Miniforge Python, which lacked NumPy for the trace
parser. Its ordinary restore succeeded, and its receipt is preserved under
`preflight-failure-miniforge/`. The successful capture used the verified
mlir-aie `ironenv` interpreter.

## Provenance and evidence

The pinned physical tuple remains Phoenix firmware `1.5.5.391`, kernel
`7.1.7-custom+`, loaded driver SHA-256
`21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`,
and the source/XCLBIN/output/runner pins inherited from the preceding balanced
relocation campaign.

The harness, candidates, qualification and layout receipts, provenance, raw
traces, decoded events, outputs, logs, failed parser preflight, restoration,
and decision are preserved under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260808T234135Z-firmware-blockwrite-noop-history-depth/
```

The final harness and probe receipt SHA-256 values are respectively
`1cafc93e6114cde413e2e07aacd028666e0bce4fada874a955996206f422dfb7`
and `278b324b6a308bbb5ad92e4c2f905aeae2b9038c7f64ba9b0273d858c91bbbf9`.
The required `cargo test --lib` verification passed both library suites:
505 tests with 2 ignored, then 4,341 tests with 33 ignored. Its log SHA-256 is
`7e8729e84ab87986a9f8e2542f799cbf3e92df2b6db078d8fd97efc3ef38d575`.

## Licensed conclusions

1. A linear one-cycle-per-recent-`NOOP` rule is falsified: doubling the
   immediate block from 16 to 32 did not change the 248-cycle target.
2. A 32-record periodic return to the no-immediate-`NOOP` state is falsified at
   this target.
3. At the measured depths, the higher-cost response is already present at 16
   immediate `NOOP` records and remains unchanged at 32, adding exactly 16 shim
   cycles relative to zero immediate records.
4. The result strengthens, but does not prove, the instruction-line recency
   hypothesis from the read-only path audit: both B and C interpose the same
   firmware `NOOP` loop path and produce the same target cost.
5. No exact threshold below sixteen, state carrier, scheduler cost, or general
   timing model is licensed yet.

The next useful boundary should distinguish specific firmware-path/cache-line
history from elapsed transaction distance while preserving target phase and
payload. It should be designed separately rather than fitting a scheduler
change to this single saturated boundary.
