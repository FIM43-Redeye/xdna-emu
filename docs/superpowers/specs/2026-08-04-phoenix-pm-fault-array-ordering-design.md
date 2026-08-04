# Phoenix PM-Fault Array Ordering

**Status:** Diagnostic complete on 2026-08-04. The approved Stage-A premise was
falsified; no production timing change is authorized by this design.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessor:** Native core event 65 reaches the shared event network and the
signed-firmware error lifecycle is already proven. The remaining failing guard
was initially interpreted as two timing defects. The discriminating trace below
shows that only firmware/array scheduling remains demonstrated.

## Decision

Close Stage A without a production change. The discriminating trace found no
array-internal qualitative inversion, while repeated Phoenix runs disproved the
proposed shim-completion ordering invariant. A delay, fault predicate, or DMA
timing change would fit an asynchronous observation rather than reproduce a
hardware rule.

The next design boundary is the already isolated scheduler defect: replace the
run-to-boundary firmware loop with one clock-aware firmware/array scheduler
derived from configured clocks and instruction timing.

## Corrective Evidence

The accepted diagnostic is preserved at
`build/experiments/phoenix-pm-fault-array-ordering/20260804T194245Z/edge-compute-mm2s/receipt.md`.
It reproduces the pinned fault ELF SHA-256
`54aa8187261a592a048ad8f19752802ae331fbb16026a19d0fb8657296383239`
and traces terminal core lock release, compute MM2S completion, shim S2MM, raw
event 65, and group event 46 in one artifact.

Decoded `soc` deltas are `first PM fault - named edge`:

| Run | Core release | Compute MM2S finish | Shim S2MM finish |
|---|---:|---:|---:|
| Phoenix 0 | 106 | 56 | 557 |
| Phoenix 1 | 106 | 61 | 442 |
| Phoenix 2 | 106 | 57 | 552 |
| Phoenix 3 | 106 | 61 | -1712 |
| Phoenix 4 | 106 | 55 | 538 |
| Phoenix 5 | 106 | 55 | 554 |
| Emulator | 113 | 121 | 87 |

The core release-to-fault interval is deterministic on Phoenix and close in the
emulator. The final compute MM2S BD precedes the fault on both sides. Shim
completion crosses the fault in either direction on identical Phoenix runs.
That edge includes the asynchronous shim/NoC/main-memory boundary already
classified as a deliberate timing gap in
[`docs/trace/cross-domain-skew-limit.md`](../../trace/cross-domain-skew-limit.md).
Its raw cross-domain timestamp is not a causal ordering oracle.

All six Phoenix outputs and the emulator output are byte-identical. The richer
trace therefore removes, rather than locates, the claimed Stage-A divergence.

## Consequences

- No DMA, lock, routing, or actor-timing production edit is licensed here.
- `advance_phoenix_tct_publication` remains a pure transport seam.
- Column clock writes remain firmware-owned; they must not be ignored or
  silently reversed.
- The existing signed-firmware guard remains the scheduler RED case. Firmware
  publishes its response and gates the column before the modeled core reaches
  the already-correct native fault transition.
- The scheduler requires its own reviewed design and TDD boundary before
  implementation. This correction does not authorize that implementation.

## Explicit Deferrals

- firmware/array clock ratios and the general clock-aware scheduler;
- absolute producer, propagation, DMA, and firmware-service cycle matching;
- the unexplained multiple physical event-65 pulses;
- broader PM-fault recurrence and recovery behavior;
- finite TCT buffering, AIE2P topology, and unrelated DMA timing calibration;
- removal of dormant experimental timing modes.
