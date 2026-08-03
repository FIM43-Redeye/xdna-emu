# Phoenix Memory-Tile Native DMA Fault Production

**Status:** Approved design; implementation pending.
**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` (version `1.5.5.391`).
**Predecessor:** Controlled memory-tile `DMA_S2MM_ERROR` delivery closed by
commit `42b6905d` and KVM evidence
`build/experiments/phoenix-vfio-user/20260803T205340Z-406956`.

## Purpose and boundary

Replace the final synthetic memory-tile `Event_Generate` in the four-record
async-error proof with a fault raised by the emulated DMA engine itself. The
slice closes one native producer end to end:

```text
invalid memory-tile S2MM descriptor
  -> DMA channel Error state and status
  -> DMA_S2MM_ERROR (event 133)
  -> GROUP_ERRORS (event 128), broadcast, L1/L2
  -> signed firmware async ring
  -> unchanged amdxdna driver cache/ioctl
```

This is a producer correction, not a new error-delivery path. Shim-native
error delivery and the remaining DMA fault classes stay unclaimed.

## Derived facts and current defect

- The aie-rt status contract assigns `Error_BD_Invalid` to bit 11 of the DMA
  channel status.
- The llvm-aie/mlir-aie event data assigns memory-tile
  `DMA_S2MM_ERROR = 133`, `DMA_MM2S_ERROR = 134`, and
  `GROUP_ERRORS = 128`.
- Memory-tile S2MM channel 0 accepts in-range even-direction descriptors
  0 through 23. Descriptor 23 therefore passes index and parity validation.
- An untouched descriptor 23 has `Valid = 0`. Starting it already moves the
  emulator channel to `ChannelFsm::Error` and exposes `Error_BD_Invalid` in
  status, but no hardware error event is published. That missing state-edge
  publication is the root defect.

The controlled `Event_Generate(133)` proof established every downstream
stage. It does not prove that the DMA engine can originate the event.

## Chosen native trigger

On memory tile logical `(0, 1)` / physical `(1, 1)`, write descriptor number
23 to S2MM channel 0 `Start_Queue` without programming that descriptor first.
The trigger is deliberate because it is:

- architecturally valid at the register boundary;
- recoverable and bounded;
- already represented by the emulator's `BdNotValid` and channel-error state;
- independent of uncertain host-address, NoC, and lock timing semantics.

The physical NPU must confirm the tuple before the emulator result is claimed:
the status must report invalid BD and firmware must publish memory-tile event
133 for physical `(1, 1)`.

## Design

### One-shot DMA error transition

Each DMA channel records whether its current `Error` state has already been
reported. A DMA-engine drain returns every newly entered error state once.
Reset, stop, or a new successful start rearms the channel. Repeated scheduler
steps while the channel remains in `Error` must not emit duplicate events.

The drain is intentionally state-based rather than added to each individual
failure branch. It therefore covers synchronous failures during `Start_Queue`
and failures reached while advancing the DMA state machine without duplicating
publication logic.

### Shared semantic event publisher

Extract the existing semantic body of tile `Event_Generate` into one
`DeviceState` helper. The helper performs the real event effects:

1. activate the tile event;
2. notify trace observers;
3. promote configured group events and seed broadcasts;
4. tap the L1 interrupt controller;
5. record categorized Tier-B async errors.

Register-driven `Event_Generate` delegates to this helper. Native DMA faults
call the same helper directly; they do not synthesize an MMIO write. The
coordinator propagates seeded broadcasts to a fixpoint after publishing the
drained DMA errors.

This is the smallest shared correction: downstream behavior remains single
sourced, while register writes and native hardware producers remain distinct
causes.

## Test-first implementation and acceptance

1. Add a focused regression that writes only memory-tile S2MM0 `Start_Queue`
   for untouched BD 23. Before implementation it must fail because event 133
   and the async record are absent.
2. Make that regression pass with the shared publisher and one-shot error
   drain. It must assert status bit 11, event 133, group event 128, the expected
   async record, and no duplicate record after additional cycles.
3. Add a failing then passing trace-patcher test for inserting the derived
   `Start_Queue` register write; reuse the existing terminal-TCT insertion
   machinery rather than create another injector.
4. Confirm the exact trigger on physical Phoenix/NPU1 before treating the
   emulator result as hardware-equivalent.
5. Change the signed-firmware guard's fourth record from
   `Event_Generate(133)` to the native `Start_Queue(BD 23)` trigger. It must
   still produce `[location=0x0101, module=0, event=133]`.
6. Change the KVM unchanged-driver fourth leg the same way. It must preserve
   `err_code=0x2040304000b` and `ex_err_code=0x101` without synthesizing a
   response, record, or completion.
7. Run formatting and focused checks, then `nice -n 19 cargo test --lib`.

## Explicit deferrals

- native MM2S, compute-tile, and shim-tile producer proofs;
- DMA error details other than invalid descriptor;
- address/lock unavailable, FoT, and token-stall behavior;
- multiple pending errors, ordering, overflow, recovery-visible state, and
  exact timing.

Those follow only after this single causal path is hardware-confirmed and
closed end to end.
