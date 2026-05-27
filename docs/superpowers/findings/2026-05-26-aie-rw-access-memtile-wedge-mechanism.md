---
name: AIE_RW_ACCESS memtile wedge mechanism (Phoenix)
description: Empirical characterization of the Phoenix NPU wedge that follows an opcode 0x203 (MSG_OP_AIE_RW_ACCESS) access to row=1 (memtile) from a hwctx that does not claim memtile resources. Wedge is unrecoverable by PCIe-layer resets; only SoC reboot recovers. Includes BAR forensics, SMU state during wedge, and a driver-patch proposal to prevent the footgun.
type: finding
---

# AIE_RW_ACCESS memtile wedge mechanism (Phoenix)

## Context

After patching `drivers/accel/amdxdna/npu1_regs.c` on 2026-05-26 to advertise
`AIE2_RW_ACCESS` in the NPU1 feature table (enabling `xrt::aie::device::read_aie_reg`
end-to-end on Phoenix), we surveyed the reach of opcode 0x203 across tile types to
characterize what's safe to use for dispatch-overhead calibration.

Hardware: Phoenix NPU1 at PCI BDF `0000:c6:00.1`, firmware 1.5.5.391, protocol 5.8.
The probe (`xdna-emu/tools/rw-access-probe/rw-access-probe.cpp`) uses an
`_diag_shim_chain_sweep/k8/chess/aie.xclbin` context — k8 chain occupying 8
compute tiles across rows 2-5, **no memtile** in routing.

## Hypothesis tested (and busted)

The Phoenix FW handler for opcode 0x203 (`FUN_08ad98c4` in Ghidra) does no
row/col/type validation — it forwards the user-supplied tile-local offset to
the AXI fabric and waits for ACK. Predicted: if we pass the *correct*
tile-local offset for the target tile type (per AM025 regdb), the FW would
forward to a valid AXI address and answer cleanly.

This was wrong. Memtile accesses wedge regardless of whether the offset is
"correct" for the memtile, as long as the hwctx doesn't claim memtile resources.

## Survey results

Each scenario: 3 reads of Timer_Low, 50ms spacing, 30s wall-clock timeout.

| # | Tile type | row | reg offset | Result | Notes |
|---|-----------|-----|------------|--------|-------|
| 1 | compute core | 2 | 0x340F8 | clean, 3 reads | roundtrip 69/112/134µs, Δ≈24k cycles/100ms |
| 2 | compute memory module | 2 | 0x140F8 | clean, 3 reads | roundtrip 107/136/196µs, Δ≈24k cycles/100ms |
| 3 | shim | 0 | 0x340F8 | clean, 3 reads | roundtrip 69/121/126µs, Δ≈24k cycles/100ms |
| 4 | memtile (correct offset) | 1 | 0x940F8 | **wedge** | 5.3s mailbox timeout, `ret -62` |
| 5 | memtile (wrong offset) | 1 | 0x340F8 | not run | scenario 4 already established wedge mechanism |

Note: the ~24k cycles/100ms in scenarios 1-3 indicates Timer_Low is largely
paused when no kernel is running and only advances during FW mailbox-processing
windows — not a continuous time source under these conditions. The shim
returning the same delta as compute is curious (shim has no compute core) and
deserves separate investigation.

## Driver and FW symptoms at wedge

dmesg after scenario 4:
```
amdxdna 0000:c6:00.1: xdna_mailbox.145: opcode 0x203 size 24 id 0x1d000010
(5.3 seconds elapse)
amdxdna 0000:c6:00.1: [drm] *ERROR* xdna_send_msg_wait: Wait for completion timeout
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_rw_aie_reg: AIE reg read failed, ret -62
amdxdna 0000:c6:00.1: [drm] *ERROR* amdxdna_aie_tile_read_reg: AIE register read failed, ret -62
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_xrs_unload: destroy context failed, ret -19
```

The destroy-context fails at `-ENODEV` because the FW is no longer answering
any mailbox traffic — the AXI-read wait inside the 0x203 handler has the
Xtensa CPU blocked indefinitely.

Subsequent driver-side cleanup:
```
aie2_set_runtime_cfg: Failed to set runtime config, ret -19  (clock gating off, retry pre-suspend)
aie2_suspend_fw: Failed to suspend fw, ret -19
aie2_mgmt_fw_fini: Suspend_fw failed
smu cmd 4 failed, 0xff                                       (POWER_OFF rejected)
Power off failed, ret -22
```

## Recovery attempt escalation

Each escalation step per CLAUDE.md:

1. **modprobe -r/modprobe reload** — completed in ~2s (didn't hang in
   `synchronize_srcu`; the user-context process had already exited so no
   process held `/dev/accel/accel0`). modprobe re-probe failed with
   `smu cmd 4 failed, 0xff` (defensive POWER_OFF in `aie2_smu_start` rejected
   by SMU; probe returns early before reaching POWER_ON).

2. **Bridge PM-cycle** (D0→D3hot→D0 via bridge's `reset` attribute on
   00:08.2) — completed cleanly at PCIe layer, but re-probe failed identically.
   Confirms CLAUDE.md observation that on-NPU controllers live downstream of
   the PCIe reset domain.

3. **True SBR** (BCR.SBR toggle on 00:08.2) — completed at PCIe layer (BAR
   enable cycles observed). Re-probe failed with `smu cmd 4 failed, 0xffffffff`
   — note the response went from `0xff` to `0xffffffff`, meaning the SMU went
   from "alive, refusing POWER_OFF" to "completely unresponsive". SBR
   made things measurably worse at the SMU level.

4. **suspend/resume** — ruled out per memory [[feedback_no_s3_suspend_on_devbox]]
   (post-resume frequency issues make this devbox unsuspendable).

5. **Reboot** — only remaining path.

## BAR forensics during wedge (P1 + P3)

Before reboot, with the driver unloaded (failed probe), BARs are still mapped
and readable from userspace via `/sys/bus/pci/devices/.../resourceN`. This
gave us an unusual window into the device's internal state.

### BAR2 (SRAM, 256 KiB) — entirely dead

`tools/fw-alive-probe.py` (new) read FW_ALIVE_OFF (BAR2 + 0x3F000):

```
[t=36306.751s] FW_ALIVE = 0xFFFFFFFF
[t=36307.251s] FW_ALIVE = 0xFFFFFFFF
[t=36307.751s] FW_ALIVE = 0xFFFFFFFF
[t=36308.252s] FW_ALIVE = 0xFFFFFFFF
[t=36308.752s] FW_ALIVE = 0xFFFFFFFF
```

Spot-reads of BAR2 at offsets 0x00000, 0x00100, 0x01000, 0x10000, 0x20000,
0x3F000, 0x3FFFC: **all 0xFFFFFFFF**. This isn't FW writing all-ones; it's
the PCIe completer returning "endpoint not responding," with the root complex
synthesizing 0xFFFFFFFF as the read data. **The FW's working memory is
unreachable through the SRAM BAR aperture.** Either:

- The SRAM controller is on the same power/clock domain as the wedged FW
  and goes down with it, or
- The SRAM aperture is FW-mediated (not a flat memory window but a proxy
  routed through the Xtensa) and the proxy is gone with the FW.

Either way, anything FW might have written to SRAM before wedging — log
buffers, panic strings, last-known state — is post-mortem-invisible.

### BAR0 (regs, 512 KiB) — partially alive

`tools/smu-probe.py read` snapshot during wedge:

```
SMU_CMD  (0x100AC) = 0xFFFFFFFF   (scratch reg, last written before wedge — lost)
SMU_RESP (0x100B0) = 0x000000FF   [CMD_FAIL]
SMU_ARG  (0x100B4) = 0xFFFFFFFF
SMU_INTR (0x10094) = 0xFFFFFFFF
PSP_CMD  (0x100A0) = 0xFFFFFFFF
PSP_ARG0 (0x100A4) = 0xFFFFFFFF
PSP_ARG1 (0x100A8) = 0xFFFFFFFF
```

`SMU_RESP = 0x000000FF` is **real device data** — a byte (not 0xFFFFFFFF
synthesized) left over from the SMU's last POWER_OFF rejection. So BAR0 at
offset 0x100B0 is a live MMIO read. The other BAR0 scratch registers reading
0xFFFFFFFF could be either (a) the registers' actual contents post-SBR (some
of them get zeroed/all-onesed on PCIe reset) or (b) the same
endpoint-not-responding synthesis we see on BAR2.

### SMU command path — alive but in fault-state

`tools/smu-probe.py exec 0x1 0xDEADBEE0` (TestMessage — should return arg+1):
```
SMU_RESP = 0x000000FF  [CMD_FAIL]  (after 0.0 ms)
SMU_OUT  = 0xFFFFFFFF
```

`tools/smu-probe.py exec 0x2 0x0` (GetSmuVersion — should return packed
76.101.0):
```
SMU_RESP = 0x000000FF  [CMD_FAIL]  (after 0.0 ms)
SMU_OUT  = 0xFFFFFFFF
```

Both immediate `CMD_FAIL` responses (0.0ms, not timeout). The SMU is alive
enough to receive the cmd write, decode it, and write a response — but it
refuses every command, including the trivial ones that don't require any
NPU coordination. The SMU has entered a fault state where it short-circuits
all commands to CMD_FAIL, and that fault state is not cleared by PCIe SBR.

## Mechanism — the three-stage cascade

The wedge has more depth than "FW Xtensa stuck". It's three interdependent
controllers freezing together:

1. **Xtensa LX7 (FW CPU) blocked.** A synchronous AXI load to a non-decoding
   tile-local address never receives an ACK. The Xtensa's bus interface unit
   has no software-visible AXI timeout in this configuration; the load
   instruction blocks indefinitely. This is the original failure.

2. **SRAM aperture (BAR2) goes dark.** Whether through shared power domain
   or through FW arbitration, BAR2 reads stop returning real data and start
   returning synthesized 0xFFFFFFFF. This destroys any post-mortem visibility
   into FW state — we can't read log buffers, mailbox queues, or working
   memory.

3. **SMU enters fault-refuse mode.** The SMU is on a separate controller
   but its command processing apparently depends on FW being alive. Once FW
   dies, SMU starts failing every command with CMD_FAIL — including
   TestMessage and version queries that should not need FW cooperation.
   This blocks the driver's normal recovery path (which goes
   `POWER_OFF → POWER_ON` via SMU).

Stages 2 and 3 are the key insight: a single FW wedge takes down two
other independent-looking controllers with it. The recovery surface
available to the driver (mbox stop, ring restart, POWER_OFF, POWER_ON, BAR
resets) cannot reach across these boundaries.

## Why PCIe-layer resets don't help

Per CLAUDE.md (and now confirmed empirically): the SMU, the Xtensa FW
controller, and the SRAM controller live downstream of the PCIe domain
that SBR / bridge PM-cycle / BAR reset can affect. Pulsing PERST# on bus c6
resets the c6:00.0/c6:00.1 endpoints' PCIe-side state machines (enabling
BARs, link training, MSI configuration) but does not reset the internal
controllers' fault flags.

Worse, **SBR made the wedge measurably deeper**: SMU went from 0xff
(CMD_FAIL — alive but refusing) to 0xffffffff (completely unresponsive on
subsequent reads). Speculation: SBR perturbed the SMU's internal
state, and without FW to mediate, it's now in a state even further from
serviceable. This is consistent with CLAUDE.md's note that SBR works at
PCIe but not for SMU-level wedges; it's also evidence that SBR is
counterproductive for this failure mode.

The remaining recovery paths per CLAUDE.md:
- **suspend/resume**: drops the SoC to retention voltage, clears
  on-NPU controller state. Not usable on this devbox
  ([[feedback_no_s3_suspend_on_devbox]]).
- **reboot**: required, and we ran the recovery via Maya's `!` after
  this finding doc was written.

## Why the "correct offset" hypothesis was wrong

Pre-survey hypothesis: the FW translates (col, row, addr) into a final
AXI address using the regdb's module-base-formula. If we hand it the
regdb's tile-local Timer_Low offset for the memtile (0x940F8), it should
hit the memtile's actual Timer_Low and answer cleanly.

What actually happens: the hwctx in our probe (`_diag_shim_chain_sweep/k8`)
does not include the memtile in its column/row partition. The Phoenix
firmware's tile-power and AXI-routing setup is per-context — the memtile's
AXI window is **not enabled** in our context's view of the array. Any
opcode 0x203 to row=1 hits a tile whose AXI is gated off (PERST-equivalent
on the AIE array side), which never ACKs. Even a "correct" offset fails
because the *tile* itself isn't decoding within this context.

So the rule is: AIE_RW_ACCESS reaches whatever tiles the hwctx has been
configured for. Tiles outside the hwctx's claim are unreachable, and the
unreachability is enforced by silently-dropping AXI rather than a
graceful FW-side rejection.

We did not validate this hypothesis with a memtile-using hwctx (would
need an xclbin that routes through a memtile, e.g. `memtile_repeat_count`
or similar from mlir-aie tests). Doing so is a future experiment worth
running to confirm the model.

## Implications

### For xdna-emu

The emulator should eventually model:
- Opcode 0x203 dispatch with `aie2_access_type` ∈ {REG_READ, REG_WRITE};
  reject {MEM_READ, MEM_WRITE} (Phoenix FW only implements 0/1, confirmed
  by FW decompile of `FUN_08ad98c4`).
- The per-hwctx "this tile is reachable" check based on the column/row
  partition the hwctx claimed.
- A wedge state when the model is asked for an unreachable tile: subsequent
  mailbox traffic should timeout. The BAR2-goes-dark and SMU-fault-state
  behaviors are out of scope for emulator-as-XRT-target but useful for
  "what does the wedge look like" tests.

### For driver-side defensiveness (companion patch)

The current `aie2_rw_aie_reg` performs only one tile-locality check (in
`amdxdna_aie_tile_read_cb`): `wa->access->col >= hwctx->num_col`. There's
no row check. Adding a row check that matches the hwctx's actual claim
would prevent the footgun we just experienced.

Minimal defensive patch (filed alongside the feature-enable):

```c
/* In drivers/accel/amdxdna/aie.c, amdxdna_aie_tile_read_cb / _write_cb */
if (wa->access->row == 1 /* memtile */ &&
    !hwctx_claims_memtile(hwctx)) {
    XDNA_ERR(xdna,
        "AIE_RW_ACCESS to memtile (row=1) blocked: this hwctx does not "
        "claim memtile resources; access would wedge Phoenix FW "
        "(see docs/superpowers/findings/2026-05-26-aie-rw-access-memtile-wedge-mechanism.md)");
    return -EPERM;
}
```

The `hwctx_claims_memtile()` predicate doesn't exist yet — the simplest
conservative version is "always return false on Phoenix" (rejecting row=1
unconditionally), which prevents the wedge at the cost of disallowing
memtile access entirely. A more correct version inspects the hwctx's
partition descriptor; that's a follow-up.

### For Phoenix FW design

The FW design defect is a synchronous AXI load with no software watchdog.
Any opcode that issues AXI to user-supplied (col, row, addr) is vulnerable.
Comparing to 1.5.6.399 (per [[2026-05-26-phoenix-fw-1.5.6.399-diff]]) shows
this isn't being addressed in newer FW — the AIE_RW_ACCESS handler is
byte-identical between versions. AMD appears to treat this as caller
responsibility.

### Wedge-detection signature (driver-independent)

A fast test for "is the NPU in a 0x203-style wedge":
- BAR2 + 0x3F000 (FW_ALIVE_OFF) reads 0xFFFFFFFF → SRAM unreachable
- BAR0 + 0x100B0 (SMU_RESP) reads 0x000000FF and any SMU exec returns
  CMD_FAIL within <1ms → SMU in fault-refuse state
- /dev/accel/accel0 missing or amdxdna probe failing

This is fully usable from userspace without the driver. Useful for any
future probe that risks wedging — sanity-check first, abort if wedged.

## Tools introduced

- `xdna-emu/tools/fw-alive-probe.py` — direct BAR2 reads of FW_ALIVE_OFF
  with optional dereference into the `mgmt_mbox_chann_info` struct. Works
  even when the driver is unloaded.
- (Existing) `xdna-emu/tools/smu-probe.py` — direct BAR0 SMU command
  probe. Used for P3 in this finding.

## Open questions

1. **Does a memtile-using hwctx make row=1 reachable?** This is the key
   follow-up — confirms or refutes the "per-hwctx tile gating" model.
2. **Does row=0 (shim) write access also wedge under non-shim-using
   hwctx?** We tested only reads in scenario 3. Shim is usually claimed
   by every kernel (DMA interface to host), so this may never happen in
   practice, but worth confirming.
3. **Why does Timer_Low advance by ~24k cycles in 100ms regardless of
   tile type when no kernel is running?** The reads should be from
   independent clock domains; the fact that compute-core, compute-memory
   module, and shim all show the same delta is suspicious. Either the
   timer is gated by FW mailbox processing (which has identical cost per
   read) or the timer values we read are FW-cached rather than fresh
   silicon reads. Worth investigating once the device is recovered.
4. **Was the post-SBR transition from `0xff` (CMD_FAIL) to `0xffffffff`
   (no response) reversible if we hadn't run SBR?** I.e., would a simpler
   recovery have worked before SBR perturbed the SMU? Future wedges
   should test: skip SBR, go straight from modprobe-reload to reboot.

## See also

- [[2026-05-26-phoenix-fw-1.5.6.399-diff]] — confirms no FW-side fix in
  newer firmware.
- [[2026-05-06-npu1-msg-op-capability-survey]] — original survey that
  established opcode 0x203 is implemented on Phoenix.
- [[2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix]] — Phoenix SMU
  command map and the POWER_OFF asymmetry that blocks driver-side
  recovery from FW-hung states.
- CLAUDE.md "NPU recovery" section — the recovery escalation chain.
