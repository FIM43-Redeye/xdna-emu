# Phoenix Compute-Memory Native DMA Fault Production

**Status:** Implemented and validated on 2026-08-03.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` (version `1.5.5.391`).

**Predecessor:** The native memory-tile producer proof in
[`2026-08-03-phoenix-memtile-native-dma-fault-design.md`](2026-08-03-phoenix-memtile-native-dma-fault-design.md).

## Closed boundary

This slice replaces the controlled compute-memory `Event_Generate(97)` leg
with a fault raised by the compute-tile DMA engine:

```text
invalid compute S2MM1 descriptor
  -> DMA channel Error_BD_Invalid
  -> DMA_S2MM_1_ERROR (event 98)
  -> GROUP_ERRORS, broadcast, L1/L2
  -> signed firmware async ring
  -> unchanged amdxdna driver cache/ioctl
```

The implementation generalizes the existing one-shot memory-tile DMA error
drain to compute tiles and selects the toolchain-generated, channel-specific
compute-memory events: S2MM0/1 map to 97/98 and MM2S0/1 map to 99/100. Both
tile types continue through the same semantic event publisher; shim DMA remains
excluded.

## Hardware-derived trigger

The physical control/fault pair uses runtime tile `(1,2)` (xclbin-local
`(0,2)`), unused compute S2MM channel 1, and BD 15. Both streams explicitly
clear `DMA_BD15_5` after the workload's final TCT. Only the fault stream then
writes `DMA_S2MM_1_Start_Queue = 15`.

The control records no event 98. The fault records exactly one event 98 while
retaining its live memory-DMA anchor. Both runs complete normally and produce
byte-identical output. Evidence:
`build/experiments/phoenix-native-compute-dma-error/20260803T224532Z/`.

## Firmware and driver proof

The signed-firmware guard now raises the same native fault and receives one
record `[location=0x0102, module=0, event=98]`, while preserving
`Error_BD_Invalid` status and one-shot publication.

The unchanged-driver KVM lifecycle passes with
`err_code=0x2040304000b`, `ex_err_code=0x201`; dmesg backtracks the record to
row 2, column 1, module 0, event 98, category 8. Evidence:
`build/experiments/phoenix-vfio-user/20260803T225822Z-848074/`.

## Explicit deferrals

- native compute MM2S, shim DMA, and core producer proofs;
- DMA fault details other than invalid descriptor;
- multiple pending faults, ordering, overflow, recovery-visible state, and
  exact timing.
