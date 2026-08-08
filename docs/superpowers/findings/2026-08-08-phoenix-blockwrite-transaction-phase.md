# Phoenix Signed-Firmware BlockWrite Transaction Phase

## Verdict

Phoenix signed-firmware `BlockWrite(1)` timing follows the transaction record's
byte phase, not its measurement ordinal, in the completed four-phase crossover.
At the deterministic low-QoS identity reported as MP-NPU/H `400/800 MHz`, one
zero word written to unused shim DMA BD14 measured:

| `BlockWrite` start phase mod 64 | Shim timestamp cycles |
|---:|---:|
| 0 | 266 |
| 20 | 216 |
| 44 | 248 |
| 56 | 246 |

The forward order `0,20,44,56` produced `266,216,248,246`. The reverse order
`56,44,20,0` produced `246,248,216,266`. Each order ran twice in the physical
sequence forward, reverse, forward, reverse and repeated digit-for-digit.
Reversal changed every phase's ordinal, predecessor, and padding history while
preserving its exact cost. Transaction byte phase is therefore admitted as a
causal timing input for this path.

This does not identify the mechanism behind the phase law. It does not license
a cache-line, MMIO, NoC, fetch, or firmware-pipeline explanation, and it does
not license a scheduler change. The completed sixteen-phase crossover then
showed that phase alone is insufficient because pre-window history changes six
shared phase costs; see
[`2026-08-08-phoenix-blockwrite-phase-history-interaction.md`](2026-08-08-phoenix-blockwrite-phase-history-interaction.md).

## Pinned tuple and receipt

- Phoenix/NPU1 firmware `1.5.5.391`, payload SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- kernel `7.1.7-custom+`;
- loaded `amdxdna.ko` SHA-256
  `21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`;
- worktree commit `6759b2784d2be7f85c6d25334cd15c9e7b905464`;
- XCLBIN SHA-256
  `d25ab5b8b45a0119c7a62efbe291599020adf86e27609fdc01a6346637ab51b3`;
- source transaction SHA-256
  `25f020f9845a761576205e98db046dd26743d7423595f3c9fe4102e2bb8084ec`;
- expected output SHA-256
  `64ed86b909d6d0502b64b28db0ea1272ffb358e20e9b1d88b63ccb07fa900cf5`;
- forward candidate SHA-256
  `4a3569ea4ce0c780bc1cb859ba90dec01f47523b74404e2bd027eee5337b33c4`;
- reverse candidate SHA-256
  `16e690b6d7df9c86cc6de1d101fbb8fec2a287d4704bec98229a871f77a40388`.

The self-contained harness, provenance, decoded events, raw traces, outputs,
emulator logs, and final classification are preserved under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260808T204937Z-firmware-blockwrite-phase-crossover/
```

Every measured run reproduced the expected output, included both shim DMA
lifecycle witnesses, exposed exactly four alternating `USER_EVENT_1` /
`USER_EVENT_0` measurement pairs, and retained the reported `400/800` clock
identity before and after dispatch. The `finally` restore ran the ordinary
source transaction without QoS, reproduced the output, returned to reported
default `600/1028`, and left `/dev/accel/accel0` unowned. The post-run kernel
warning/error query was empty.

## Derived instrument and emulator qualification

The instrument derives shim BD14 width, register addresses, and event IDs from
the AM025 register database and aie-rt. It rejects a source stream that writes
or queues BD14, clears the complete descriptor once, and never queues it. Each
measured window is exactly:

```text
USER_EVENT_1 -> BlockWrite(1 zero word) -> USER_EVENT_0
```

Authentic four-byte CDO `NOOP` records select the next phase only between the
previous stop and the next start, outside the measured window.

Both candidate hashes first passed the existing pinned signed-firmware
`CHAIN_EXEC_NPU` guard. Each of their four windows consumed exactly 47 firmware
instruction attempts and executed the same 47-PC dynamic sequence. The
sequence was also identical across forward and reverse candidates; its
normalized SHA-256 is
`b483f68d3e7908b25f848efc682a5f9170824930c9d4e54fdf546213fd355659`.
The temporary qualification assertions were removed after both runs.

A separate attempt to qualify through the XRT plugin's direct CDO executor was
rejected as evidence: that path currently reports authentic CDO `NOOP` opcode
`0x05` as unknown. It never reached either candidate's trace admission and did
not touch hardware. The signed-firmware guard is the relevant path for this
experiment; direct-executor NOOP support is separate cleanup.

## Physical crossover

| Physical ordinal | Candidate | Phase order | Intervals |
|---:|---|---|---|
| 1 | forward | `0,20,44,56` | `266,216,248,246` |
| 2 | reverse | `56,44,20,0` | `246,248,216,266` |
| 3 | forward | `0,20,44,56` | `266,216,248,246` |
| 4 | reverse | `56,44,20,0` | `246,248,216,266` |

The raw marker timestamps differ between fresh contexts, as expected, but all
within-window differences are exact. A single four-position ordinal pattern
would have produced the same cost vector in both orders. Instead, the reverse
vector is the exact phase-keyed reversal of the forward vector.

The earlier variable-payload capture is reconciled by this result: its repeated
one-word records at phase 56 both cost 246 cycles, while one-word records at
phases 20 and 44 cost 216 and 248. Payload size alone was never sufficient to
explain those intervals.

## Licensed conclusions and next boundary

This evidence licenses only the following:

1. Transaction record phase modulo 64 is a real timing input to the pinned
   Phoenix signed-firmware `BlockWrite(1)` path.
2. Ordinal alone is falsified for this matrix; predecessor and padding history
   are unnecessary to explain these four repeatable costs.
3. The current emulator's phase-invariant 47-attempt functional path is not a
   timing model for silicon.
4. No scheduler, instruction cost, clock ratio, or fixed delay should change
   from this four-phase result.
5. The completed sixteen-phase map supersedes the payload-length sequence: next
   hold phase 44 and predecessor 40 fixed while adding one 64-byte block of CDO
   `NOOP` history before assigning costs to operations or firmware timing
   classes.

The finding does not establish the physical cause of the phase law, live clock
phase or frequency, a general `BlockWrite` latency, older-firmware behavior,
AIE2P behavior, or firmware cycle accuracy.
