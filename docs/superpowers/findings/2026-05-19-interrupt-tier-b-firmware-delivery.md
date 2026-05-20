# Interrupt Tier B — Firmware Async-Event Host Delivery (tracked follow-up)

**Status:** Spec 1 (plumbing + INSTR_ERROR producer) **shipped** on `dev`.
Spec: [`../specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md`](../specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md);
plan: [`../plans/2026-05-19-interrupt-tier-b-firmware-mailbox.md`](../plans/2026-05-19-interrupt-tier-b-firmware-mailbox.md).
Per-class detection follow-ups (DMA, parity, ECC, lock, stream) remain
tracked as separate not-started specs in that spec's §10. Tier C (TDR /
context-restart) shipped 2026-05-19 -- see
[../findings/2026-05-19-interrupt-tier-c-tdr.md](../findings/2026-05-19-interrupt-tier-c-tdr.md).

## Why this is separate from Tier A

On Phoenix/NPU1 the AIE L1/L2 shim interrupt never reaches the x86 host
directly. It terminates at NPI interrupt lines consumed by on-NPU MGMT/ERT
firmware, which synthesizes a mailbox async-event message; the host only ever
sees the mailbox MSI-X. Tier A (the `interrupt` subsystem) models the AIE path
to that firmware boundary. Tier B is the firmware/mailbox async-event model
the emulator deliberately lacks (it shortcuts MGMT/ERT with a synchronous
completion model).

## Host-boundary contract (derived from xdna-driver)

- Only host IRQ the driver registers: `mailbox_irq_handler` (MSI-X),
  `amdxdna_mailbox.c` ~line 924.
- AIE array errors reach the host via async mailbox messages →
  `aie2_error_async_cb` (`aie2_error.c` ~278-289), queued to a workqueue.
- Mailbox response contract: write response into the I2X ring buffer at the
  firmware-provided base; update the I2X tail pointer register; write the
  IOHUB interrupt status register at `info->i2x.mb_head_ptr_reg + 4`
  (computed by `aie2_calc_intr_reg` in `aie2_pci.h:378`) to fire MSI-X.
- TDR: driven by lack of forward progress on active contexts (periodic timer
  check via `aie2_rq_handle_idle_ctx`/`aie2_rq_is_all_context_stuck`);
  `aie2_tdr.c` ~27-73.

## Scope when picked up

Minimal firmware async-event model: when Tier A asserts an NPI interrupt line
for an error, synthesize the async-event mailbox message so the driver's
`aie2_error_async_cb` path observes it. This is its own brainstorming → spec →
plan cycle.
