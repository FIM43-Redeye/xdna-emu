# Phoenix Management DMA -- Blocking Descriptor Path Design

**Date:** 2026-07-28

**Status:** Approved

## Purpose

Model the firmware-owned management DMA operation that blocks the pinned
Phoenix `CHAIN_EXEC_NPU` path. The engine copies the firmware-published
descriptor's byte range, publishes a completion result, and clears the command
busy bit last.

This is an off-array firmware peripheral. It is not PSP, SMU, AIE tile DMA, or
per-column readiness.

## Derived Contract

Three global lanes start at `0x27271000` with a `0x1000` stride. Firmware owns
allocation and publishes each lane by writing:

```text
lane + 0x04 = descriptor + 0x20
lane + 0x08 = descriptor
lane + 0x0c = 3                  # asynchronous mode only
lane + 0x00 = 0x75              # command/busy
```

The eight-word descriptor contains flags, byte length, source address and
attributes, destination address and attributes, and two words unused by the
pinned blocking transfer. Blocking firmware waits for command bit 0 to clear,
then treats the low six bits of `lane + 0x100` as the result.

Firmware programs 60 translation slots. An internal address selects its slot
and offset directly:

```text
slot = (address >> 26) - 3
offset = address & ((1 << 26) - 1)
entry = 0x27280000 + slot * 16
```

The translation table, control words, descriptors, and lane registers already
live in `Bus` mailbox backing. Registered `HostMemory` remains the authority for
which reconstructed host ranges exist. Because firmware decorates mapped
low-word addresses with bit 31, resolution tests both the reconstructed address
and its bit-31-cleared form and succeeds only when exactly one complete
registered range matches.

## Implementation

Add one management-DMA tick to `Bus` and invoke it after an attached firmware
instruction. For each busy lane, the tick:

1. reads the firmware-published descriptor;
2. resolves local and translated-host endpoints;
3. copies the descriptor byte length;
4. writes zero result status on success; and
5. clears command bit 0 last.

An invalid descriptor, translation slot, or host range leaves data and lane
state unchanged rather than inventing successful completion or an unsupported
hardware error code.

No new controller type, mapping database, driver-opcode parser, thread, or
clock scheduler is introduced.

## Tests

Tests first cover:

- busy-before-tick and host-to-local publication-after-tick;
- all three lanes and a nonzero translation slot;
- a nonzero 64 MiB-window offset and a high host address;
- local-to-host copying through the same descriptor machinery;
- unmapped and ambiguous host translations failing without copying; and
- the pinned firmware guard advancing naturally beyond the old status poll.

## Deferred

Asynchronous interrupt delivery, `+0x114` acknowledgement/abort semantics,
measured latency, exact hardware error codes, and multi-PASID address-alias
isolation require separate evidence. They are not inferred by this functional
blocking path.
