# Phoenix PM-Fault Array Ordering

**Status:** Architecture approved in conversation on 2026-08-04; written
checkpoint awaiting review before implementation.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessor:** Native core event 65 reaches the shared event network and the
signed-firmware error lifecycle is already proven. The remaining failing guard
shows two distinct timing defects: array-internal completion ordering and
firmware/array scheduling. This design covers only the former.

## Decision

Keep one causal implementation path and correct it in two sequential stages:

1. **Stage A (this design):** locate the first array-internal ordering
   divergence and correct that shared mechanism so the PM fault precedes the
   output completion path as it does on Phoenix.
2. **Stage B (separate design):** replace the run-to-boundary firmware loop
   with one clock-aware firmware/array scheduler derived from configured clocks
   and instruction timing.

Stage B is required for final fidelity, but it must not be used to make Stage A
pass. Stage A adds no alternate scheduler, fixed cadence, completion delay, or
workload-specific exception.

## Pinned Evidence

The patched fault ELF has SHA-256
`54aa8187261a592a048ad8f19752802ae331fbb16026a19d0fb8657296383239` in both
the physical receipt and the direct emulator run. The physical receipt is
`build/experiments/phoenix-native-core-pm-address-error/20260804T060601Z/receipt.md`;
the signed-firmware run uses the preserved PDI under
`build/experiments/phoenix-vfio-user/20260804T071731Z-2673818/async-error/`.

For that artifact, the observed order is:

| Boundary | Phoenix cycle | Direct-emulator cycle |
|---|---:|---:|
| PM event 65 and group event 46 | 335521 | 7329 |
| Shim S2MM start | 335965 | 6009 |
| Shim S2MM finish | 336656 | 7213 |

Phoenix therefore raises the fault 444 cycles before shim S2MM starts and
1,135 cycles before it finishes. The emulator raises the fault 1,320 cycles
after S2MM starts and 116 cycles after it finishes. This qualitative inversion
exists without signed firmware and is therefore an array-model defect.

The signed-firmware guard exposes a second defect: firmware publishes its
command response and gates the column at array cycle 2290. If the column is
kept runnable, the modeled event appears 66 array cycles later at cycle 2356.
That starvation is a scheduler problem for Stage B, not evidence for delaying
TCT publication in Stage A.

Existing dormant backpressure modes and coarse startup/cooldown knobs do not
change the failing boundary. They are not production fixes.

## Diagnostic Gate

The measurements identify the first externally visible inversion, not yet its
internal producer. Before changing production behavior, add one discriminating
same-artifact trace that follows both sides of the output handoff:

- terminal core output release and relevant lock transition;
- compute/memory-tile output DMA start and finish, or the nearest existing
  stream-port activity anchors;
- shim S2MM start and finish;
- raw event 65 and promoted group event 46; and
- TCT admission and signed-firmware publication when that layer is present.

Use event IDs and register configuration derived from aie-rt, mlir-aie, and the
existing generated architecture data. Reuse the paired control and fault
artifacts so the terminal jump remains the only producer difference.

The first edge whose relative order differs from hardware selects the fix
site. If the trace does not identify one shared mechanism, stop and revise the
model rather than adding a delay.

## Stage A Correction

Correct the first proven divergence in the existing DMA, lock, routing, or
actor-timing state. Every caller must continue through that shared mechanism;
there is no fault-only completion path.

`advance_phoenix_tct_publication` remains a transport seam: it converts a
completed DMA token, routes it through the configured TileControl fabric, and
publishes only what reaches the shim landing. It must not acquire an arbitrary
delay, inspect core fault state, or become a scheduler.

Column clock writes remain firmware-owned. Stage A must neither ignore them nor
silently re-enable a gated column. The scheduler correction will later ensure
that firmware and array actors advance in their derived clock domains before a
legitimate gate takes effect.

## TDD and Acceptance

The existing signed-firmware guard
`m2c_chained_pm_fault_publishes_native_core_error` is the end-to-end RED case;
it currently ends at `ArrayIdleFirmwareWaiting` with the faulting core at
`0x484` and its column gated.

Before the production edit, add the smallest deterministic in-process RED test
that records the relevant event and completion transitions for the pinned
artifact. Assert causal order, not fitted absolute timestamps.

Stage A is green only when:

1. raw event 65 and group event 46 precede shim S2MM start, finish, and the
   resulting TCT publication for the pinned fault artifact;
2. the paired control and fault outputs remain byte-identical;
3. ordinary command completion remains intact;
4. the correction uses the shared array mechanism identified by the trace,
   with no fault-specific delay or alternate timing path; and
5. focused tests, signed-firmware guards, `nice -n 19 cargo test --lib`,
   `cargo fmt --all --check`, and `git diff --check` pass.

After local green, repeat the paired Phoenix trace. Record absolute deltas for
future calibration, but do not fit them in Stage A. The unchanged-driver KVM
gate remains the final integration proof after Stage B makes the signed path
runnable without manual clock intervention.

## Explicit Deferrals

- firmware/array clock ratios and the general clock-aware scheduler;
- absolute producer, propagation, DMA, and firmware-service cycle matching;
- the unexplained second physical event-65 pulse;
- broader PM-fault recurrence and recovery behavior;
- finite TCT buffering, AIE2P topology, and unrelated DMA timing calibration;
- removal of dormant experimental timing modes unless the diagnostic trace
  proves one is the actual shared mechanism.

Stage B must preserve Stage A's ordering invariant. It may change when
firmware observes an already-causal array transition, but not reorder the
underlying array transition to make the guard pass.
