# Phoenix Memory-Tile Native DMA Fault Production

**Status:** Implemented and validated on 2026-08-03.
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
- aie-rt's `_XAieMl_MemTileDmaCheckBdChValidity` accepts BDs 0 through 23
  only on even per-direction channels and BDs 24 through 47 only on odd
  channels. S2MM channel 2 with BD 24 is therefore an exact invalid-bank
  pairing.
- Starting that pairing already moves the emulator channel to
  `ChannelFsm::Error` and exposes `Error_BD_Invalid` in status, but no hardware
  error event was published. That missing state-edge publication was the root
  defect.

The controlled `Event_Generate(133)` proof established every downstream
stage. It does not prove that the DMA engine can originate the event.

## Chosen native trigger

On memory tile logical `(0, 1)` / runtime physical `(1, 1)`, write descriptor
number 24 to S2MM channel 2 `Start_Queue` after the workload's last TCT wait.
Channel 2 is unused by the pinned kernel, and BD 24 belongs to the odd-channel
bank, so the write faults without disturbing the completed data path.
The trigger is deliberate because it is:

- accepted at the register boundary and rejected by the derived channel/BD
  validity rule;
- recoverable and bounded;
- already represented by the emulator's `BdNotValid` and channel-error state;
- independent of uncertain host-address, NoC, and lock timing semantics.

The physical NPU must confirm that the tuple itself raises event 133 before the
emulator result is claimed. Firmware delivery is a separate configured-network
proof and must not be inferred from the raw producer observation.

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

1. Add a focused regression that writes only memory-tile S2MM2 `Start_Queue`
   for BD 24. The aie-rt channel/BD validity rule rejects high-bank BDs on
   even channels. Before implementation it must fail because event 133 and the
   async record are absent.
2. Make that regression pass with the shared publisher and one-shot error
   drain. It must assert status bit 11, event 133, group event 128, the expected
   async record, and no duplicate record after additional cycles.
3. Add a failing then passing trace-patcher test for inserting the derived
   `Start_Queue` register write; reuse the existing after-last-TCT insertion
   machinery rather than create another injector.
4. Confirm the exact trigger on physical Phoenix/NPU1 before treating the
   emulator result as hardware-equivalent.
5. Change the signed-firmware guard's fourth record from
   `Event_Generate(133)` to the native `Start_Queue(BD 24)` trigger. It must
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

## Validation evidence

- The focused emulator regression
  `memtile_invalid_bd_bank_start_publishes_dma_error_once` observes status bit
  11, event 133, promoted group event 128, broadcast 0, one Tier-B record, and
  no duplicate record while the channel remains in `Error`.
- A physical Phoenix trace reused the established `a1_solo` memtile route with
  event 133 in slot 0 and the original live anchor events in slots 1 through 4.
  The no-fault control recorded 65 anchor events and no slot-0 event. The
  otherwise identical after-TCT BD-24 run retained its anchors and recorded
  exactly one slot-0 event at runtime tile `(1,1)`. Artifacts are under
  `build/experiments/phoenix-native-memtile-dma-error/20260803T213956Z/`.
- The signed-firmware guard
  `m2c_core_compute_memory_and_memtile_errors_reach_registered_async_buffer_through_signed_firmware`
  delivers `[location=0x0101, module=0, event=133]` from the native start-queue
  cause.
- The unchanged-driver KVM lifecycle passes all four re-registration legs and
  publishes the native fourth record as `err_code=0x2040304000b`,
  `ex_err_code=0x101`. Evidence:
  `build/experiments/phoenix-vfio-user/20260803T221316Z-693969/`.

The physical trace proves the silicon producer. Physical delivery through the
locally generated research error PDI remains unclaimed; the end-to-end delivery
proof is the signed-firmware KVM tuple above.
