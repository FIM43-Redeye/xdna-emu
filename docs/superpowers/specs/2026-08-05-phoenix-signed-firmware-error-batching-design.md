# Phoenix Signed-Firmware Error Batching

**Status:** Implemented and verified on 2026-08-05.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

## Boundary

Close the intra-column multiple-pending lifecycle without adding another error
producer. Reuse the native compute-memory invalid-BD producer and native core
PM-address producer in one physical column, assert both before signed firmware
runs, and require one registered `0x10c` buffer to contain both records.

The proof crosses the configured group events, both L1 switches, L2, management
source 56, unmodified signed firmware, and the existing registration response.
It must not synthesize a record, response, ordering decision, or acknowledgement.

## Derived contract

aie-rt backtracks set L2 bits in ascending order. Phoenix maps even L2 bits to
switch A and odd bits to switch B; switch A scans compute-core modules and
switch B scans compute-memory modules. Within a module, set group members are
scanned in ascending event order. The expected batch is therefore:

1. core event 65 (`PM_ADDRESS_OUT_OF_RANGE`);
2. compute-memory event 98 (`DMA_S2MM_1_ERROR`).

Both records must appear in one buffer with `err_cnt = 2` and `ret_code = 0`.

The 8 KiB buffer can hold 681 records. The toolchain default fatal masks expose
14 core and 13 compute-memory members per compute tile, 9 memory-tile members,
and 11 shim members. One Phoenix column has four compute rows, so at most
`4 * (14 + 13) + 9 + 11 = 128` distinct members can be pending. Capacity
overflow is therefore unreachable for the pinned NPU1 topology and is not an
implementation target.

## Implementation and proof

Extend the existing signed-firmware guard rather than adding a second harness.
Write the batch assertion first and record its RED result. If the real path
already batches correctly, no production change is warranted; the guard and
fidelity-ledger correction are the complete slice. Otherwise, correct the
shared event/interrupt path at the first proven loss of state, then rerun the
same guard.

After the local signed-firmware proof, run the unchanged-driver KVM gate with a
two-error batch and require exact backtracking for both records from one async
buffer. Preserve the evidence directory.

## Outcome

The existing firmware path was already correct; no production emulator change
was needed. The signed-firmware guard now asserts the native PM-address and
invalid-BD faults before pumping firmware and receives one ordered two-record
buffer.

The KVM boundary needed its own fresh device lifetime because the existing
lifecycle contains terminal faults. It also cannot use core execution and a
register-write DMA fault as simultaneous stimuli: core execution is deferred,
and the two native faults were consequently serviced in separate buffers. The
isolated gate therefore writes the AM025-derived core `Event_Generate` register
for event 65 immediately before the native invalid-BD write and the terminal
TCT. This preserves the real event network, signed firmware, driver, ordering,
and acknowledgement seams while the local guard retains the native PM-address
producer proof.

Exact unchanged-driver evidence is in
`build/experiments/phoenix-vfio-user/20260805T222620Z-929952`.

## Deferred

- cross-column response arbitration;
- destroy/recreate recovery-visible state;
- exact producer, propagation, firmware-service, and acknowledgement timing;
- event-specific producers beyond the representative native producer in each
  module class;
- broader plugin-to-firmware unification.
