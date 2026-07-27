# Phoenix Firmware / Array / Driver Wiring

**Target:** Phoenix / NPU1, firmware `1502_00`

**Status:** Active evidence record, updated 2026-07-27

This document supersedes the 2026-07-07 archaeology that treated an opaque
PSP/SMU power handshake as a boot blocker. That interpretation did not survive
the later coherence, visibility, and external-observation probes.

## Goal and Validation Boundary

The end state is the unmodified Phoenix kernel driver talking to unmodified
management firmware, which programs the same `DeviceState` the array emulator
executes:

```text
unmodified driver
  -> virtual Phoenix PCI function
  -> unmodified 1502_00 firmware
  -> shared array DeviceState
  -> independently validated kernel
  -> firmware response
  -> unmodified driver
```

The current XRT plugin is a SHIM replacement above the kernel-driver boundary.
It can exercise the emulator component but cannot prove the flow above. The
driver-equivalence gate needs a virtual Phoenix PCI frontend below the
unmodified driver.

## Proven Pre-CPU Handoff

The emulator does not emulate PSP or SMU internals. It models only the
architecturally required state visible when the Xtensa CPU starts, derived from
coherent execution of the unmodified image:

- PSP ROM-aperture load offset `0x5c`;
- low instruction overlay at file offset `VMA + 0x100`, preserving the
  firmware's Harvard instruction/data views;
- initialized D-side bytes for VMA `[0x0000e740, 0x0000fefc)`;
- segment B preloaded at physical `0x08b00000`;
- reset `varway56` MMU state and a functionally equivalent page table;
- reset-preconfigured, four-byte I2X SRAM slot 15 mapping for `FW_ALIVE_OFF`.

With that handoff, unmodified `1502_00` reaches the scheduler's natural
`waiti`, builds the `_NPU` management-channel descriptor, exposes the
descriptor through firmware-programmed I2X slot 13, and publishes
`0x030bb000` through slot 15 at `0x030bf000`. A driver-style zero write clears
the same local SRAM word. No completion agent or forced CPU progress is
enabled.

The old claim that boot required a new PSP/SMU column-power subsystem is
therefore retired. The task at local `0xfae0` and the earlier internal queue
were real observations, but the inference that they represented an
unrecoverable external boot gate was wrong.

## Firmware / Array State Ownership

`InterpreterEngine` is the sole owner of `DeviceState`.

`FirmwareProcessor` owns only the Xtensa CPU and firmware bus. During a
firmware step, `Cpu::step_with_device` borrows the engine's device and the bus
routes array accesses into that borrow. Standalone research steps retain the
same execution path without a device. There is no cloned array state,
processor-owned `DeviceState`, `Arc`, mutex, or unsafe ownership transfer.

The public component seam lives in `crates/xdna-emu-ffi`:

```text
xdna_emu_load_firmware
xdna_emu_boot_firmware
xdna_emu_firmware_read_host_sram32
xdna_emu_firmware_write_host_sram32
```

It accepts explicit bytes, validates the Phoenix load map, boots against the
existing interpreter device, and accesses the processor bus's genuine SRAM
aliases. It does not parse management commands, translate BAR offsets, enable
the diagnostic `HostMailbox`, or wire the XRT SHIM plugin into this path.

The older `xdna_emu_assign_partition` entry point remains a synthetic SHIM hook
for existing bridge tests. It is not evidence for the real-firmware or
unmodified-driver path.

## Proven Phoenix Host Contract

The open driver's NPU1 register map and mailbox implementation pin these host
resources:

| Resource | Device address | Host operation |
|---|---:|---|
| BAR2 base | `0x03080000` | SRAM windows |
| Descriptor | `0x030bb000` | Read firmware-published `_NPU` structure |
| X2I ring | `0x030bc000` | Write complete host-to-firmware packets |
| I2X ring | `0x030bd000` | Read complete firmware-to-host packets |
| `FW_ALIVE_OFF` | `0x030bf000` | Read descriptor pointer, then clear |
| BAR4 base | `0x030c0000` | Ring indices and IOHUB status |
| X2I tail | `0x030ec000` | Publish after the complete X2I packet |
| X2I head | `0x030ec004` | Firmware consumption index |
| I2X tail | `0x030ed000` | Firmware publication index |
| I2X head | `0x030ed004` | Publish after host consumption |
| IOHUB status/clear | `0x030ed008` | Clear, drain, re-read, repeat |

The channel descriptor reports MSI-X vector 14 and protocol 5.8 for the pinned
image.

Host ordering is part of the contract:

- X2I: read head, copy the full request or tombstone into BAR2, then publish
  tail;
- I2X: read tail, copy the complete response from BAR2, then publish head;
- interrupt worker: clear IOHUB status, drain responses, re-read status, and
  repeat while nonzero.

MSI-X is a wakeup hint. Ring indices and IOHUB status are authoritative, so
coalesced or lost hints must not lose messages.

## Controller Facts and the Remaining Unknown

Firmware and controller-table analysis pins:

- management slot 14 uses firmware event `(6,4)` and aggregate controller
  source 46;
- slot 13 uses event `(6,5)` and aggregate source 45;
- slot-14 subordinate IDs include 14, 38, 108, and 109 on route 3;
- the generic ISR reads the active source, disables it, dispatches the
  registered callback, then acknowledges and re-enables it.

What is not yet pinned is the causal bridge:

```text
BAR4 X2I-tail publication
  -> exact subordinate slot-14 pending source
  -> active controller source 46
  -> Xtensa interrupt bit 0
  -> firmware event (6,4)
```

`0x27200170`, `0x27200174`, and `0x27200178` are an unrelated earlier internal
queue, not the host management mailbox. Writing the published tail must not be
modeled as `cpu.interrupt |= 1` until hardware evidence identifies the
register alias, subordinate source, and arbitration behavior.

## Separate Downstream Completion Contract

DMA task completion tokens are distinct from host X2I publication. A BD with
`Enable_Token_Issue` emits an AIE stream packet toward the management
subsystem; the firmware later consumes that completion state while handling
jobs. The open toolchain pins the AIE-side token format, but the
management-side landing and timing still require a real post-alive job
observation.

The existing `DEFAULT_MAILBOX_CYCLES`, dispatch gates, and forced launch seams
remain fidelity debts. They should disappear only when the real firmware path
drives the same operations, not by tuning a replacement constant.

## Milestones

1. **Natural boot and alive publication -- complete.** The unmodified primary
   image reaches idle and publishes externally visible state.
2. **Shared array ownership -- complete.** Firmware borrows the engine's sole
   `DeviceState`.
3. **Public component FFI -- complete.** Explicit loading, bounded boot, and
   host SRAM access are tested through the C ABI.
4. **Post-alive interrupt capture -- next.** Pin the BAR4-to-slot-14 chain on
   hardware before implementing it.
5. **Virtual PCI driver boundary -- pending.** Present Phoenix BARs, MSI-X, and
   lifecycle below the unmodified driver.
6. **Pinned open-driver command contract -- pending.** Close every legitimate
   normal, error, reset, power, timeout, teardown, and recovery path without a
   driver-specific responder.
7. **Older authoritative Phoenix images -- pending after the primary SHA is
   green.**

## Evidence Entry Points

- PSP handoff and boot loop: `src/firmware/mod.rs`
- Phoenix bus routing and SRAM aliases: `src/firmware/mmio.rs`
- Borrowed device stepping: `src/firmware/xtensa/interp/mod.rs`
- Public component seam: `crates/xdna-emu-ffi/src/firmware.rs`
- Open-driver sources: `../xdna-driver/src/driver/amdxdna/aie2_pci.c`,
  `aie2_pci.h`, `amdxdna_mailbox.c`, and `npu1_regs.c`
- Approved component design:
  `docs/superpowers/specs/2026-07-27-phoenix-firmware-ffi-component-design.md`
- Host/firmware fidelity ledger:
  `docs/fidelity-gaps/host-firmware-dispatch.md`
- Array completion background: `docs/arch/tct-completion-model.md`
