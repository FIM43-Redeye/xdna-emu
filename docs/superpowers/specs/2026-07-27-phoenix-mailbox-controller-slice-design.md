# Phoenix Mailbox / Management-Controller Slice -- Design

**Date:** 2026-07-27

**Status:** Approved by Maya

## Purpose

Implement the evidence-backed parts of the selected firmware architecture:

```text
host BAR2/BAR4 mailbox peripheral
        [intentionally unconnected]
management interrupt controller
        -> Xtensa architectural interrupt
```

The missing connection is deliberate. Current evidence proves the host
transaction envelope and the controller-to-firmware handling path separately,
but not the transition by which a BAR4 X2I-tail publication becomes controller
source 46. This slice must not manufacture that transition.

## Existing State to Reuse

- BAR2 X2I/I2X bytes already use the firmware-programmed SRAM aliases in
  `Bus::host_sram_load32` and `Bus::host_sram_store32`.
- Firmware-side `0x27000000..0x28000000` accesses already have one RAM backing
  in `Bus`.
- Xtensa `INTERRUPT`, `INTSET`, `INTCLEAR`, `INTENABLE`, level-1 delivery,
  `waiti`, exception entry, and `rfe` are implemented and tested.
- `FirmwareProcessor` and the interpreter backend already share the one
  `DeviceState`; this slice does not change array ownership.

The XRT SHIM plugin remains outside this path. It currently synthesizes
firmware effects and executes the array directly.

## Host Mailbox Peripheral

Add one concrete, Bus-owned Phoenix mailbox-register type. It stores exactly
the five registers published by the pinned management-channel descriptor:

| Register | Device address |
|---|---:|
| X2I tail | `0x030ec000` |
| X2I head | `0x030ec004` |
| I2X tail | `0x030ed000` |
| I2X head | `0x030ed004` |
| I2X IOHUB interrupt status/clear | `0x030ed008` |

Host and firmware accesses to those addresses use the same state. Existing
BAR2 aliases continue to use the same local-SRAM backing as firmware.

`Bus` gains general internal host-device word accessors:

```rust
pub fn host_load32(&self, device_address: u32) -> u32;
pub fn host_store32(&mut self, device_address: u32, value: u32);
```

They route the five BAR4 registers to the new peripheral and otherwise
delegate to the existing BAR2 SRAM-alias accessors. Unsupported addresses keep
the existing zero-read/dropped-write behavior.

The BAR4 state does not emit an interrupt or controller source in this slice.
A test must lock down that negative contract.

No new C ABI is added. The existing public functions are explicitly
SRAM-named and remain SRAM-only. A later virtual-PCI design will define its
own complete BAR ABI rather than extending a misleading function name.

## Management Interrupt Controller

Add one concrete, Bus-owned management-controller type. Do not reuse the AIE
shim-tile L1/L2 controllers: those model different hardware.

This first controller slice implements only firmware-visible behavior grounded
for the single-source path:

- four 32-bit enable banks at `0x27200300..0x2720030c`;
- four 32-bit status/acknowledgement banks at
  `0x272003b0..0x272003bc`;
- the active-source read at `0x272003c4`;
- one aggregate output to Xtensa interrupt bit 0.

Source-to-bank mapping follows the firmware's own helpers:

```text
bank = source >> 5
bit  = 1 << (source & 31)
```

An internal `assert_source(source)` operation has two preconditions: the
source's enable bit is set and no other source is active. It records the source
in its status bank, exposes it through `0x272003c4`, and queues one aggregate
assertion for the CPU. A caller that violates either precondition gets a
rejection; that is an input-seam guard, not a claim about how silicon latches a
disabled or competing source.

Only one active source is supported. A second assertion is rejected rather
than inventing priority or arbitration. Disabled-source latching, simultaneous
sources, and edge-versus-level peripheral inputs remain undefined until
evidence requires them.

Writing a one bit to the matching acknowledgement bank clears that status bit
and retires the active source. The enable bank remains unchanged. Reads of the
status/acknowledgement bank expose its current status, preserving the known
read/write dual role without treating an acknowledgement register as the
originating peripheral state.

The packed fields at `0x27200904..` remain ordinary backed registers. Their
numeric updates are known, but their semantics are not.

## Controller-to-Xtensa Delivery

Before the existing architectural interrupt-delivery check, `Cpu::step_on`
consumes a queued management-controller assertion through `CpuBus`. That
assertion sets Xtensa pending bit 0:

```text
explicit controller source assertion
  -> active source register
  -> aggregate controller output
  -> Xtensa INTERRUPT bit 0
  -> existing EXCCAUSE 4 delivery
```

The architectural pending bit itself remains set until firmware writes
`INTCLEAR`. This slice does not define whether the controller would reassert
after an out-of-order CPU clear; the pinned handler acknowledges the
controller first and then writes `INTCLEAR=1`, so that unobserved case is not
needed for the proven path.

The source-assertion operation is crate-internal. It is a hardware-boundary
seam for tests and for the future mailbox connector, not a host-visible
firmware-bypass API.

## Pinned Source-46 Path

An explicitly asserted source 46 must exercise this already-recovered path:

```text
0x27200304 bit 14 enabled
  -> 0x272003c4 reads 46
  -> Xtensa interrupt bit 0
  -> level-1 exception
  -> slot 14 / handler 0x5948
  -> firmware event (6,4)
  -> 0x272003b4 bit 14 acknowledgement
  -> WSR INTCLEAR,1
  -> source and CPU pending state retired
```

This test does not claim that a host X2I-tail write is what asserts source 46.

## Tests

Implementation follows RED/GREEN.

1. Host-peripheral unit tests:
   - BAR2 request words reach the firmware local-SRAM backing;
   - all five BAR4 words are independent and shared by host/firmware access;
   - publishing X2I tail does not assert a controller source.
2. Controller unit tests:
   - source 46 is rejected while disabled;
   - enabling source 46, asserting it, reading active source, and
     acknowledging it completes the single-source lifecycle;
   - acknowledgement does not change the enable bank;
   - a second simultaneous source is rejected.
3. CPU integration test:
   - an explicitly asserted enabled source reaches the existing level-1
     exception path without directly mutating `Cpu::interrupt`.
4. Pinned-firmware integration test, gated by `XDNA_FIRMWARE`:
   - boot naturally to idle;
   - explicitly assert source 46;
   - run until firmware returns to idle;
   - verify controller active state and Xtensa pending bit are cleared.

Required gates:

```bash
cargo test --lib
cargo test -p xdna-emu-ffi
cargo fmt --all --check
git diff --check
```

The real-image test uses the pinned primary firmware path. These are brief
local builds; Halo is unnecessary.

## Non-Goals

- BAR4 X2I-tail to source-46 routing.
- Meaning or routing behavior for the packed `0x272009xx` fields.
- Multi-source priority, nesting, or arbitration.
- PSP or SMU internals.
- BAR0 lifecycle registers.
- MSI-X generation.
- XRT SHIM plugin conversion.
- Virtual PCI frontend or unmodified-driver acceptance.
- Management command parsing or synthesized responses.

## Evidence Gate for the Missing Connection

The two halves may be connected only after a non-halting management-Xtensa
trace or authoritative controller specification pins the first internal
transition caused by X2I-tail publication. At minimum it must distinguish the
originating pending state from the acknowledgement bank and explain how the
configured selector fields participate.
