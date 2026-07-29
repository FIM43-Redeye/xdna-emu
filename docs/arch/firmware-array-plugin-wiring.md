# Phoenix Firmware / Array / Driver Wiring

**Target:** Phoenix / NPU1, firmware `1502_00`

**Status:** Active evidence record, updated 2026-07-29

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

The XRT plugin remains a SHIM replacement above the kernel-driver boundary, so
it cannot prove the flow above. The first driver-equivalence slice is instead
proven by the separate vfio-user Phoenix PCI frontend below the unmodified
driver.

## Proven Pre-CPU Handoff

The emulator does not emulate PSP or SMU internals. It models only the
architecturally required state visible when the Xtensa CPU starts, derived from
coherent execution of the unmodified image:

- PSP ROM-aperture load offset `0x5c`;
- `$PS1` signed extent equal to its fixed `0x100`-byte header plus the
  header-declared body size;
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

## Controller and X2I-Publication Facts

Firmware and controller-table analysis pins:

- management slot 14 uses firmware event `(6,4)` and aggregate controller
  source 46;
- the first context CQ returned by the pinned image is channel 5, whose X2I
  tail is `0x030da000` and whose aggregate source is 37;
- slot 13 uses event `(6,5)` and aggregate source 45;
- slot-14 setup assigns packed selector fields 14, 38, 108, and 109 the
  two-bit value 3; open artifacts do not name the fields or the value;
- the generic ISR reads the active source, disables it, dispatches the
  registered callback, then acknowledges and re-enables it.

The host-to-controller connection is externally pinned even though its
internal register-level implementation is not exposed:

- the open driver's send path copies the complete packet and then performs only
  the X2I-tail write;
- Phoenix channel tails follow `0x030d0000 + channel * 0x2000`;
- the unmodified firmware's generic channel handler receives the channel
  number and selects source `channel + 0x20`;
- therefore channel 5 selects source 37 and channel 14 selects source 46.

The emulator models that exact shared seam:

- host and firmware accesses share one state block for the five published
  BAR4 mailbox words;
- a single-source management-controller slice owns the four enable banks, four
  status/acknowledgement banks, and active-source read;
- a host write to any X2I-tail address derives and asserts source
  `channel + 0x20`; adjacent X2I-head and I2X words assert nothing;
- the pinned `1502_00` image handles management source 46, acknowledges
  controller bank 1 bit 14, clears Xtensa pending bit 0, and returns to its
  natural `waiti` without a synthetic source injection;
- a channel-5 context publication reaches APP-ERT through source 37.

This is an externally equivalent publication edge, not a claim about hidden
controller priority, disabled-source latching, or edge/level input semantics.
The deliberately single-active-source controller remains the current model.

The context path now crosses the next concrete platform boundary:

```text
channel-5 X2I-tail publication
  -> controller source 37
  -> APP-ERT event 4
  -> firmware publishes a blocking management-DMA descriptor
  -> translated HostMemory command slot is copied to local 0x00096000
  -> command bit 0 clears after data publication
  -> task_dispatcher consumes the slot
  -> firmware publishes a mode-3 descriptor for the 16 KiB host range
  -> HostMemory is copied to local staging address 0x0007d000
  -> result zero and cleared busy bit are published
  -> shared completion level asserts controller source 76
  -> firmware drains 0xbc000000 and receives the 0xdeadbeef sentinel
  -> the peripheral deasserts source 76 without disabling it
  -> firmware consumes X2I and publishes INVALID_PARAM / APP_LOAD_PDI_FAIL
  -> firmware returns to waiti with the host response complete
```

`0x27271000` is lane 0 of a three-lane, firmware-owned management-DMA
peripheral, not per-column readiness. The functional model derives the host
target from the firmware-programmed 60-slot translation table and uses the same
descriptor/copy path for blocking mode 0 and asynchronous mode 3. Both publish
zero result and clear command bit 0 after the data copy; mode 3 then raises one
shared completion level and asserts source 76 for every lane. Reading the
configured aperture at `0xbc000000` returns the empty `0xdeadbeef` sentinel,
clears that level, and returns the controller to its enabled, inactive state.

`lane + 0x114` is the separate busy-lane drain handshake. A bit-0 write sets
command bit 1 and prevents a late copy or completion interrupt. It is not the
normal completion notification.

The old column-1 L2 wait was a false prior: sources `56..59` and mask `0x3f`
belong to AIE error-network maintenance. The unconfigured negative control
reaches its genuine PDI-load failure response, but no core is enabled and no
program is loaded. Its mode-3 16 KiB copy stages the command chain; firmware
then fails CU lookup and returns `APP_LOAD_PDI_FAIL` before entering the PDI
loader.

The configured path now crosses the PDI handoff:

```text
real xclbin PDI stored in the host context heap at 0x60020000
  -> MAP_HOST_BUFFER selects the 64 MiB device heap at 0x04000000
  -> firmware keeps the PDI device address 0x04020000 unchanged
  -> the selected device-map record routes that offset through the
     firmware-programmed 0x90000000 management-DMA view to host memory
  -> CONFIG_CU registers the NPU1 32 KiB address units and CU function
  -> unmodified firmware reads the PDI through its 0x90000000 translated view
  -> firmware executes direct CDO operations and management-DMA copies
  -> the 0x84000000 transaction and 0x9c000000 management array views converge
     on the engine's sole DeviceState
  -> PDI program/data/register writes land only in assigned physical column 1
  -> one compute core is configured and enabled
  -> firmware returns to waiti while retaining the X2I request until array
     execution completes
  -> the programmed array produces correct output and a shim S2MM0 TCT
  -> firmware drains that TCT and publishes the successful I2X response
```

The harness extracts the PDI and CU metadata from the frozen xclbin and packs
the open driver's wire format. It does not parse or apply the PDI, relocate
tiles, enable the core, or manufacture a response. Both frozen Chess and Peano
artifacts now complete this path through the same observed shim S2MM0 record.
Other TCT actors, measured DMA latency, nonzero hardware result codes, sources
`77..79`, and multi-PASID alias isolation remain unclaimed.

## First Pinned Driver-Boundary Proof

The first complete driver-reachable slice now uses:

```text
frozen test.exe
  -> XRT 2.23
  -> unmodified amdxdna.ko
  -> QEMU vfio-user PCI function
  -> unmodified 1502_00 firmware
  -> shared array emulator
  -> firmware I2X response and MSI-X
  -> unmodified driver and XRT
```

Both frozen `add_one_using_dma` artifacts produce `2..=65`, return `PASS!`,
destroy their context with status zero, and shut down QEMU and the frontend
cleanly. The guest contains the normal XDNA XRT plugin and explicitly rejects
the emulator plugin. `tdr_timeout_ms=0` disables only the driver's four-second
wall-clock timeout because functional emulation takes longer; it does not
replace or synthesize command completion.

The pinned tuple is QEMU `10.2.1`, guest kernel `7.1.5-custom+`, driver commit
`216cefececd74effcd7a88350c71b99f5ef9a215`, libvfio-user commit
`37491ed9af828fc161238dacd82e83ea35a09f87`, firmware SHA-256
`d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`,
and mlir-aie commit `cce2910aadb181d35ddcaa12ace8b9b46082639b`. Evidence:

- Chess: `build/experiments/phoenix-vfio-user/20260729T060334Z-2925605`
- Peano: `build/experiments/phoenix-vfio-user/20260729T161414Z-3021849`

### Direct `EXECUTE_BUFFER_CF` Extension

The same frozen artifacts also pass with the pinned driver's supported
`force_cmdlist=N` path. This changes only the execution envelope:

- Chess: `build/experiments/phoenix-vfio-user/20260729T170819Z-3122555`
- Peano: `build/experiments/phoenix-vfio-user/20260729T171042Z-3129577`

Each driver log contains one 80-byte `EXECUTE_BUFFER_CF` (`0x00c`) request and
the matching four-byte response on the context channel with the same message
ID. Neither contains a 24-byte `CHAIN_EXEC_NPU` (`0x018`) request. Both produce
the same ordered output `2..=65`, receive the genuine context MSI-X completion,
and complete the matched `DESTROY_CONTEXT` request and response.

The in-process guard sends the same direct request with a deliberately nonzero
four-word tail. That tail is unconstrained because the pinned driver does not
initialize unused words in its fixed 80-byte request; the successful response
proves that neither firmware nor the emulator depends on zero padding.

The default path remains intact: frozen Chess passed again with
`force_cmdlist=Y`, `0x018` request/response traffic, and the same output at
`build/experiments/phoenix-vfio-user/20260729T171244Z-3136359`.

This closed `0x00c` for the frozen xclbin-only `ERT_START_CU` flow. At that
stage it did not claim `EXEC_DPU` (`0x010`); the following extension closes
that separately pinned ELF/module-style `ERT_START_NPU` slice.

### Direct `EXEC_DPU` Extension

The installed Phoenix validation archive supplies an authentic normal-XRT
producer: `validate.xclbin`, `nop.elf`, the latency recipe/profile, and
`xrt-runner`. With `force_cmdlist=N`, the pinned driver sends one 160-byte
`EXEC_DPU` request and receives the matching four-byte response without using
`CHAIN_EXEC_NPU`.

The KVM proof passed at:

```text
build/experiments/phoenix-vfio-user/20260729T192543Z-3358656
```

It records the pinned artifact hashes, context channel-5 X2I publication,
management MSI-X completion, matched opcode-`0x10` message ID `0x1d000001`,
and matched `DESTROY_CONTEXT` message ID `0x1d000010`.

The corresponding in-process guard uses those same archive members and
unmodified firmware. It proves that the complete validation PDI configures the
DPU-reserved Phoenix column, stages the exact `nop.elf` control text through
firmware management DMA, loads/enables `(0,2)`, returns success, and leaves
output memory unchanged. This closes the no-op command/control contract, not
observable DPU data-plane work, preemption, relocation-heavy ELFs, resilience,
or timing equivalence.

The lockdep-enabled guest also reports that this pinned driver calls the
reservation-lock-requiring `dma_buf_vmap` / `dma_buf_vunmap` API without the
newer kernel's required lock and nests two BO mutexes of the same lock class.
Those driver/kernel compatibility diagnostics are real and remain visible;
they occur before command completion and do not prevent the successful
firmware response or zero-status context destruction. This proof therefore
closes the frozen normal-command slice, not every driver warning or lifecycle
path.

The host can observe the complete outer transaction envelope -- BAR2 request,
BAR4 X2I-tail publication, X2I-head consumption, I2X response, host IRQ, and
I2X-head publication. It cannot observe the controller registers, Xtensa
special registers, or firmware-local event objects through the Phoenix PCI
apertures. See
[`2026-07-27-phoenix-post-alive-observability.md`](../superpowers/findings/2026-07-27-phoenix-post-alive-observability.md).

## Separate Downstream Completion Contract

DMA task completion tokens are distinct from host X2I publication. A BD with
`Enable_Token_Issue` emits an AIE stream packet toward the management
subsystem; the firmware later consumes that completion state while handling
jobs. The first Phoenix slice now forwards the configured shim's S2MM0 token
through management source 76 and the `0xbc000000` drain aperture. The
unmodified `1502_00` firmware consumes the observed `0x0020600f` record and
publishes the successful response for both frozen Chess and Peano artifacts
through both the default chained and direct `EXECUTE_BUFFER_CF` envelopes.
Other actors and measured management-side timing remain unmodeled.

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
4. **Post-alive host envelope capture -- complete.** One ordinary telemetry
   request produced a matched tail/IRQ/worker/head trace without raw BAR access
   or polling.
5. **Host-tail-to-context negative path -- complete through the real firmware
   response.**
   Address-derived X2I publication reaches sources 37 and 46;
   management traffic completes through unmodified firmware, and context
   traffic crosses both descriptor modes, stages the full 16 KiB host range,
   consumes shared source 76, and publishes the observed PDI-load failure.
6. **Configured PDI handoff -- complete.** Real `CONFIG_CU` state selects the
   frozen xclbin PDI; unmodified firmware loads it into the assigned physical
   column of the shared array.
7. **Array execution and firmware completion -- first slice complete.** The
   frozen Chess and Peano `add_one_using_dma` commands run through the
   configured shim DMA/core state, produce correct output, and reach their
   natural I2X responses through a real shim S2MM0 TCT. Other actors and
   kernels remain pending.
8. **Virtual PCI driver boundary -- first pinned slice complete.** The
   vfio-user Phoenix function presents BARs, MSI-X, guest physical mappings,
   firmware boot, and one complete dual-compiler command lifecycle below the
   unmodified driver.
9. **Pinned open-driver command contract -- first command complete through two
   envelopes.** The frozen Chess and Peano forms pass through both the default
   `CHAIN_EXEC_NPU` and direct `EXECUTE_BUFFER_CF` paths. Every other legitimate
   normal, error, reset, power, timeout, teardown, and recovery path remains to
   be closed without a driver-specific responder.
10. **Older authoritative Phoenix images -- pending after the primary SHA is
   green.**

## Evidence Entry Points

- PSP handoff and boot loop: `src/firmware/mod.rs`
- Phoenix bus routing and SRAM aliases: `src/firmware/mmio.rs`
- Phoenix BAR4 state: `src/firmware/phoenix_mailbox.rs`
- Management interrupt controller:
  `src/firmware/management_controller.rs`
- Borrowed device stepping: `src/firmware/xtensa/interp/mod.rs`
- Pinned lifecycle and PDI-handoff guards:
  `src/firmware/boot_tests/guards.rs`
- Public component seam: `crates/xdna-emu-ffi/src/firmware.rs`
- Virtual PCI frontend and frozen guest gate:
  `tools/phoenix-vfio-user/` and `scripts/phoenix-vfio-user-qemu.sh`
- Open-driver sources: `../xdna-driver/src/driver/amdxdna/aie2_pci.c`,
  `aie2_pci.h`, `amdxdna_mailbox.c`, and `npu1_regs.c`
- Approved component design:
  `docs/superpowers/specs/2026-07-27-phoenix-firmware-ffi-component-design.md`
- Host/firmware fidelity ledger:
  `docs/fidelity-gaps/host-firmware-dispatch.md`
- Array completion background: `docs/arch/tct-completion-model.md`
