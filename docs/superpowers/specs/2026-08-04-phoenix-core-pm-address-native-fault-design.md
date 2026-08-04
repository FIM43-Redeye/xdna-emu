# Phoenix Core PM-Address Native Fault Proof

**Status:** Approved for implementation on 2026-08-04.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessor:** The signed-firmware error lifecycle and native compute,
memtile, and shim DMA producers are already closed. Native core-event
production remains the next missing error-network edge.

## Boundary

Close one unambiguous native core fault through the complete path:

```text
valid jump to out-of-range program address
  -> PM_ADDRESS_OUT_OF_RANGE (core event 65)
  -> GROUP_ERRORS_0 (core event 46)
  -> per-core Error_Halt_Event match
  -> L1/L2 and Phoenix management interrupt
  -> unmodified signed firmware async ring
  -> unchanged amdxdna driver cache/ioctl
```

The correction must halt only the faulting core. Other cores, DMA engines, and
the array coordinator remain runnable. Event delivery must use the existing
shared tile-event publisher rather than a second error path.

## Derived facts

- Phoenix compute program memory is 16 KiB, so address `0x4000` is outside the
  valid range.
- Peano's AIE2 instruction definitions encode `j` with a 20-bit absolute
  program address and delay slots. Current Peano accepts and disassembles
  `j #0x4000` as a valid instruction.
- aie-rt names core event 65 `PM_ADDRESS_OUT_OF_RANGE`, includes it in
  `GROUP_ERRORS_0`, and configures each compute core's `Error_Halt_Event` to
  that group event.
- AM025 defines `Error_Halt_Control` at `0x32030` and `Error_Halt_Event` at
  `0x32034`. The halt state is per-core and clearable through the control
  register's write-one-to-clear bit.
- The unchanged driver classifies event 65 as a core access error. At physical
  tile `(1,2)`, its expected public values are
  `err_code=0x20303040006` and `ex_err_code=0x201`.

## Physical producer proof

Reuse the established traced Chess `add_one_using_dma` fixture and keep the
control artifact unchanged. Create the fault artifact by replacing its final
`done` instruction and the adjacent two-byte nop with the six-byte
`j #0x4000`; retain the following five valid nops as the jump's delay slots.
The fault therefore occurs after the fixture has produced and finalized its
ordinary output.

The patch helper must derive the instruction and ELF file locations with the
resolved Peano LLVM tools. It must verify a unique terminal `done`, the
expected replacement width, and sufficient valid delay-slot instructions, and
fail closed on any mismatch. It must not embed a fixture-specific ELF file
offset.

Trace core events 65 and 46 alongside ordinary live anchors. Acceptance
requires:

- event 65 appears exactly once and only in the fault run;
- group event 46 follows it only in the fault run;
- the control and fault outputs are byte-identical;
- trace/DMA activity after the producer demonstrates that the array was not
  globally frozen.

Normal host-command completion is recorded but not assumed. If the fault run
does not complete, preserve the bounded-timeout evidence and stop rather than
changing the producer. Reload `amdxdna` only if the context remains latched.
Preserve the paired receipt under
`build/experiments/phoenix-native-core-pm-address-error/`.

## Emulator correction

Develop the model change test-first.

1. Replace the existing test that treats an out-of-range PC as an ordinary
   halt with a failing test requiring architectural core event 65.
2. Add focused register tests for `Error_Halt_Event`, the latched
   `Error_Halt_Control` state, core-status error-halt reporting, and W1C clear.
3. Add a coordinator test in which one core faults while another actor makes
   progress; the engine must not transition to a global error state.
4. Require raw event 65 and promoted group event 46 to pass through
   `DeviceState::publish_tile_event`, including the existing propagation,
   trace, and asynchronous-error collection behavior.

The minimum production change is a dedicated architectural core-fault result
carrying the derived event ID. The coordinator publishes it through the shared
event path and continues scheduling unaffected actors. Group-event delivery
applies the configured per-core error-halt action. Do not reinterpret every
generic decode or execution error in this slice.

## Signed-firmware and driver proof

Extend the signed-firmware guard with one native core registration cycle. It
must receive the firmware payload for physical `(1,2)`, core module, event 65,
then complete the existing acknowledgement and re-registration lifecycle. No
record or response may be synthesized.

Run the unchanged-driver KVM gate and require a fresh asynchronous error with
`err_code=0x20303040006`, `ex_err_code=0x201`, and exact dmesg backtracking to
row 2, column 1, core module, event 65. Preserve the evidence even if the
faulting command remains non-completing; do not weaken the gate to infer event
identity from the public error class alone.

## Verification

1. Produce and preserve the paired physical control/fault receipt.
2. Demonstrate each focused emulator test RED before changing production code,
   then GREEN after the minimum correction.
3. Run the signed-firmware guard, C++ probe build with warnings as errors, CLI
   routing checks, and shell syntax checks at `nice -n 19` where applicable.
4. Run the unchanged-driver KVM gate once and preserve its evidence directory.
5. Finish with `nice -n 19 cargo test --lib` and `git diff --check`.
6. Update the fidelity ledger with only claims licensed by the receipts.

## Explicit deferrals

- malformed or unsupported instruction event identity, including the current
  emulator-specific event 69 path;
- unavailable data-memory and lock-access faults;
- multiple pending core errors, ordering, overflow, and recovery;
- exact producer, propagation, firmware-service, and acknowledgement timing;
- broader plugin-to-firmware unification.
