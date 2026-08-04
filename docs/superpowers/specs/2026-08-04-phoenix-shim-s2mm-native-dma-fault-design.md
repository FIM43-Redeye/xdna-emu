# Phoenix Shim-S2MM Native DMA Fault Proof

**Status:** Approved for implementation on 2026-08-04.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessor:** Native shim-MM2S delivery closed by commit `9d2e3ba7` and
KVM evidence
`build/experiments/phoenix-vfio-user/20260804T040558Z-1922353`.

## Boundary

Close the remaining shim-DMA direction through the complete asynchronous-error
path:

```text
invalid shim S2MM descriptor
  -> DMA channel Error_BD_Invalid
  -> DMA_S2MM_ERROR (PL event 72)
  -> GROUP_ERRORS, L1/L2, Phoenix management source 56
  -> unmodified signed firmware async ring
  -> unchanged amdxdna driver cache/ioctl
```

This is evidence-first characterization. The emulator already maps both shim
directions through the shared native DMA-error publisher, and its focused unit
test covers S2MM event 72. No production-model change is expected. If silicon
disagrees, stop and investigate the mismatch rather than changing the oracle or
forcing the expected result.

## Derived facts

- aie-rt defines AIE2 `DMA_S2MM_ERROR_PL = 72`,
  `DMA_MM2S_ERROR_PL = 73`, and `GROUP_ERRORS_PL = 63`.
- The aie-rt PL group-error layout enables S2MM at bit 8 and MM2S at bit 9.
- AM025 defines `DMA_S2MM_0_Task_Queue` at shim-local offset `0x1d204` and
  derives `Start_BD_ID` from bits 3:0.
- AM025 defines `DMA_BD14_7.Valid_BD` at bit 25. Clearing that word makes BD14
  invalid without relying on prior tile state.
- S2MM and MM2S collapse to the same public driver error class
  `0x2070304000b` with location payload `0x1`; exact dmesg backtracking to event
  72 distinguishes this direction.

## Physical producer proof

Reuse the pinned Chess `add_one_using_dma` trace fixture at physical shim
`(1,0)`. Use S2MM0 for the fault because S2MM1 carries the trace drain.

Construct an otherwise-identical pair immediately before the final TCT:

1. Both control and fault streams clear `DMA_BD14_7`.
2. Only the fault stream writes `DMA_S2MM_0_Task_Queue = 14`.
3. Trace `DMA_S2MM_ERROR` alongside the ordinary live shim MM2S0/S2MM0 task
   anchors.

Acceptance requires exactly one event-72 observation only in the fault run,
normal completion of both trace runs, live anchors in both, and byte-identical
outputs. Preserve the receipt under
`build/experiments/phoenix-native-shim-s2mm-dma-error/`.

## Signed-firmware proof

Extend the existing signed-firmware guard with one more registration cycle.
Clear shim BD14, queue it on S2MM0, assert native `Error_BD_Invalid`, and require
the firmware payload `[location=0x00000100, module=2, event=72]`. The response
must complete through the existing acknowledgement and re-registration
lifecycle; no record or response may be synthesized.

The existing compact coordinator table remains the generic direction test. If
it stays green, do not add another model test or production branch.

## Unchanged-driver KVM proof

Both shim faults leave their XRT command non-completing, so they cannot be
sequenced in one C++ probe process. Preserve the complete A-through-F lifecycle
and its terminal MM2S fault unchanged.

Before A-through-F, the guest launches the existing `--async-error-one` mode in
a separate process with an S2MM0 instruction stream. Process exit closes the
faulted DRM file/context, after which the existing lifecycle runs in the same
QEMU boot. The gate requires:

- a fresh async-error timestamp;
- `err_code=0x2070304000b` and `ex_err_code=0x1`;
- a non-completing S2MM run state;
- exact driver backtracking to row 0, column 1, module 2, event 72;
- all existing A-through-F markers, including event 73, still passing.

If same-boot cleanup does not permit the existing lifecycle to proceed, stop
with the evidence. A separate QEMU boot is the reviewed fallback, not an
automatic workaround.

## Verification

1. Produce and preserve the paired physical control/fault receipt.
2. Run the focused shim direction test and extended signed-firmware guard at
   `nice -n 19`.
3. Compile the C++ probe with `-Wall -Wextra -Werror`; run its CLI-routing test
   and shell syntax checks.
4. Run `./scripts/phoenix-vfio-user-qemu.sh --run-async-error` once and preserve
   its evidence directory.
5. Finish with `nice -n 19 cargo test --lib` and `git diff --check`.
6. Update the firmware fidelity ledger with only the claims demonstrated by
   those receipts.

## Explicit deferrals

- shim DMA error causes other than an invalid descriptor;
- multiple pending errors, ordering, overflow, and recovery-visible state;
- exact producer, interrupt, firmware-service, and acknowledgement timing;
- broader error-network unification.
