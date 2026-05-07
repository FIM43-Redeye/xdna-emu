---
name: NPU1 msg-op capability survey via driver bypass
description: Methodology and first result for probing whether Phoenix firmware actually implements optional MSG_OPs that AMD's NPU1 op-table omits. Uses a module-param-driven driver bypass (xdna-driver `ceecace`) to exercise the production codepath instead of raw mailbox.
type: finding
---

# NPU1 op-table capability survey

## Why

AMD's NPU1 driver-side op-table (`npu1_msg_op_tbl[]` in `npu1_regs.c`)
gates whether each optional `MSG_OP_*` may be sent to firmware. The
table is conservative -- yesterday we confirmed that
`MSG_OP_AIE_RW_ACCESS` (0x203) was implemented in Phoenix firmware
1.5.5.391 but missing from the table, so production XRT calls to
`xrt::hw_context::read_aie_reg` returned EOPNOTSUPP for no good
reason. That entry is now in the table.

NPU4 (Strix) has 9 additional optional ops in its table that NPU1
does not. Some are likely silicon-level features absent on Phoenix;
others may be driver-omissions we can lift, the same as
AIE_RW_ACCESS. Capability-discovery question: which is which?

## Approach

Three probe paths considered; we settled on the third.

1. **Raw mailbox via debugfs `test_case04`** (yesterday's tool).
   Sends arbitrary opcode + payload directly to the management
   firmware. Limitations: manual request struct construction (firmware
   rejects malformed inputs with INVALID_PARAM, indistinguishable from
   "opcode unknown" in our sample); no DMA scratch buffers for ops
   that need them.

2. **Driver bypass via module parameter** (briefly tried today, xdna-driver
   `ceecace`, since reverted). A `unsafe_accept_all_msg=Y` module param
   turned `aie2_is_supported_msg()` into a yes-machine. Problem: the
   bypass leaks into driver-internal flows (PM resume, mgmt_fw_init),
   which then fire gated opcodes the firmware doesn't implement, wedging
   the SMU. Verified failure mode 2026-05-06: with bypass on, runtime
   resume re-fires `MSG_OP_UPDATE_PROPERTY` (0x113), firmware returns
   INVALID_COMMAND, driver destroys mgmt_chann, subsequent IOCTLs hit
   `-ENODEV`. A correctly-scoped bypass (per-fd flag set via custom
   IOCTL, env-var driven on the user side) was sketched but not built
   -- the rebuild-per-opcode path below is faster for the small N of
   opcodes we need to survey.

3. **Per-opcode table entry + driver rebuild** (current approach).
   For each candidate opcode: add a placeholder `{ AIE2_FW_VERSION(5,8),
   MSG_OP_<x> }` entry to `npu1_msg_op_tbl[]`, rebuild + DKMS-install
   the driver (~3 min), reload, run the probe. Firmware response
   decides: SUCCESS = keep the entry, INVALID_COMMAND = revert. Each
   opcode is its own clean commit to xdna-driver. Production code path
   tested exactly as production XRT would call it. No bypass mechanism
   shipped, no internal-leak risk.

## Discriminator

After the bypass fires, the firmware response status word identifies
which class an opcode is in:

- `0x0` (SUCCESS): firmware recognized + handled. **Add to table.**
- `0x4000002` (`AIE2_STATUS_INVALID_COMMAND`): firmware does not
  recognize the opcode. **Table omission is correct.**
- `0x02000004` (`AIE2_STATUS_MGMT_ERT_INVALID_PARAM`): firmware
  recognized but params malformed. Implies the opcode IS implemented
  -- driver constructor got something wrong, or per-NPU request shape
  differs. **Likely an unlock candidate; investigate the request
  shape.**
- Other: more nuanced; treat as "implemented but with caveats."

## Safety lesson learned the hard way

Enabling `unsafe_accept_all_msg=1` (the abandoned bypass approach
above) wedged the SMU twice: once at modprobe time, once again at
runtime when PM resume re-fired `MSG_OP_UPDATE_PROPERTY` after the
device autosuspended. Cause: the bypass affects ALL mailbox sends,
including driver-internal ones (`aie2_mgmt_fw_init`, runtime resume),
which then send opcodes the firmware doesn't implement. Phoenix returns
`INVALID_COMMAND` and the driver bails on the bring-up; in the runtime
case it destroys `mgmt_chann` so subsequent IOCTLs return `-ENODEV`.

This is the structural reason we abandoned the bypass-flag approach.
A correctly-scoped variant (per-fd flag set via custom IOCTL, env-var
driven from userspace, internal flows never opt in) would avoid this,
but for an 8-opcode survey the per-opcode-rebuild path is simpler.

### Why PCIe reset can't recover the SMU wedge

Both the kernel's bridge "reset" (`echo 1 > .../00:08.2/reset`, which is
actually a PM-cycle of the bridge function -- `reset_method = pm`) and a
*real* Secondary Bus Reset (`setpci -s 00:08.2 BRIDGE_CONTROL=0x42`,
hold, then `=0x02`) leave the wedge intact. Verified 2026-05-06 with a
true SBR after a fresh wedge: BCR went 0x02 -> 0x42 -> 0x02, the device
re-enumerated, command register cycled (`enabling device (0000 -> 0002)`
in dmesg), but `smu cmd 4` still returned `0xffffffff` (canonical "no
PCIe completion, all-ones phantom read").

The architecture explains this:

- The driver's "SMU" is an on-NPU controller, not the system MP1.
  Evidence: `void __iomem *smu_base` in `aie2_pci.h:273` is a
  BAR-resident pointer, and the register names use `MP1_C2PMSG_*_ALT_1`
  -- the `_ALT_1` aperture is the NPU-side alias of an SMU-style
  mailbox. NPU4 explicitly maps it to BAR5 (`NPU4_SMU_BAR_INDEX = 5`).
- SBR on the upstream bridge resets the PCIe link and BAR-enable state
  on bus c6 (proven: BAR command register went 0 -> 2). But the SMU
  controller itself runs on memory/state outside the PCIe reset
  domain -- either powered from a different SoC rail, or backed by
  an AIE-array SRAM/microcontroller that survives PCIe-level reset.
- Two separate mailbox paths are involved: the SMU mailbox (driver
  <-> on-NPU SMU controller) and the MGMT_ERT mailbox (driver <->
  NPU firmware running on a control core). The bypass poisons the
  MGMT_ERT path; the wedge propagates to the SMU path because the
  partial bring-up leaves the NPU in a state where the SMU's response
  loop is waiting on something from MGMT_ERT that never arrives.

Recovery escalation, in order:

1. **Driver reload** -- doesn't help here, probe loops on `smu cmd 4`.
2. **Bridge PM-cycle** (`echo 1 > .../00:08.2/reset`) -- doesn't help.
3. **True SBR** (`setpci BRIDGE_CONTROL=0x42`, wait, =0x02) -- doesn't
   help. Useful for normal PCIe-level wedges but not this one.
4. **S3 suspend** (`systemctl suspend`) -- typically works because the
   SoC enters retention voltage, clearing on-NPU controller state.
5. **Reboot** -- always works.

Bottom line: once the wedge is in, it lives downstream of PCIe.
Don't waste time on resets; suspend or reboot.

## Results so far

### MSG_OP_UPDATE_PROPERTY (0x113) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06 via the (now-abandoned) bypass approach. Firmware
returned `status 0x4000002 = AIE2_STATUS_INVALID_COMMAND` when the
driver's bring-up sent it. AMD's table is correctly conservative for
this opcode. Do not add to NPU1 table.

### MSG_OP_GET_COREDUMP (0x119) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06 via the per-opcode-rebuild approach. Probe entry
added to `npu1_msg_op_tbl[]` (xdna-driver `7e641c9`, reverted in
`0c5d393`); validate-readback's M0 fired `DRM_IOCTL_AMDXDNA_GET_ARRAY`
with `param=DRM_AMDXDNA_AIE_COREDUMP`. Mailbox req went out, firmware
sent a 4-byte response, and the hex-dump diagnostic patch (`77e625b`)
revealed the bytes were `04 00 00 02` = `AIE2_STATUS_INVALID_COMMAND`.
AMD's table is correctly conservative; do not add to NPU1 table.

**Side discovery**: Phoenix firmware's INVALID_COMMAND response is
**4 bytes (status only)**, not the 36-byte struct the open-source
driver's `get_coredump_resp` expects. The driver's strict size check
in `xdna_msg_cb` discards the response and surfaces `-EINVAL` instead
of the cleaner `-EOPNOTSUPP` we'd see if the driver could read the
status. Worth fixing upstream: relax the check to accept smaller
responses, copy what fits, and let callers act on the actual status.

### Other candidates -- not yet probed

NPU4 table entries absent from NPU1's table that we have not yet
exercised on Phoenix:

| Opcode | Name                          | How to trigger from userspace |
|--------|-------------------------------|-------------------------------|
| 0x114  | MSG_OP_GET_APP_HEALTH         | DRM_AMDXDNA_FW_LOG via GET_INFO IOCTL or context creation/destroy |
| 0x116  | MSG_OP_CONFIG_FW_LOG          | Setting fw log buffer params (unclear which IOCTL) |
| 0x117  | MSG_OP_GET_DEV_REVISION       | Auto-fired during init -- needs separate hook to retrigger post-init |
| 0x10F  | MSG_OP_START_FW_TRACE         | DRM_AMDXDNA_SET_FW_TRACE_STATE via SET_STATE IOCTL |
| 0x110  | MSG_OP_STOP_FW_TRACE          | same path |
| 0x111  | MSG_OP_SET_FW_TRACE_CATEGORIES| same path (with categories arg) |
| 0x11C  | MSG_OP_CALIBRATE_TIME         | Auto-fired during init only -- not safely re-triggerable |

`GET_DEV_REVISION` and `CALIBRATE_TIME` are init-only paths; the only
way to probe them is to intentionally re-init, which historically
wedged the SMU. With autosuspend pinned off
(`/etc/modprobe.d/amdxdna.conf`) the resume-fail path no longer hits,
so an init re-trigger may now be tractable.

## Recommended next moves

1. **MSG_OP_START_FW_TRACE (0x10F)** -- next probe. If firmware
   implements it, we get firmware-level event tracing, which complements
   the AIE-level trace pipeline. Add table entry, rebuild, fire via
   `DRM_AMDXDNA_SET_FW_TRACE_STATE` from a probe extension to
   validate-readback (M1 test).
2. **Upstream a `xdna_msg_cb` size-check relaxation patch.** The
   one-shot diagnostic patch (`77e625b`) hex-dumps for visibility;
   the principled fix is to copy `min(got, want)`, surface the
   actual status, and translate AIE2_STATUS_INVALID_COMMAND to
   `-EOPNOTSUPP` rather than `-EINVAL`. Cleaner errno semantics.
3. **MSG_OP_GET_APP_HEALTH (0x114) and MSG_OP_CONFIG_FW_LOG (0x116).**
   Both touch firmware-level logging; if either implements, useful for
   debugging.
4. **For each opcode that returns SUCCESS**: keep its table entry as a
   commit, run shim_test #64/#65/#66 (filter expanded in `92ed2fa`) to
   validate not just acknowledgement but actual functional behavior --
   the half-implementation concern we flagged at survey start.

## See also

- `xdna-driver` `289c207` -- `MSG_OP_AIE_RW_ACCESS` table entry (the
  SUCCESS template for ops that probe positively).
- `xdna-driver` `7e641c9` + `0c5d393` -- probe and revert for COREDUMP
  (the INVALID_COMMAND template).
- `xdna-driver` `77e625b` -- hex-dump diagnostic patch in xdna_msg_cb
  (the visibility tool that turned `0x400000d` from a mystery into the
  4-byte answer).
- `xdna-driver` `25b3f51` -- raw-mailbox debugfs (`test_case04`),
  fallback for opcodes without a pre-built request constructor.
- `xdna-driver` `cd6bf13` -- the abandoned bypass module parameter
  (revert), retained in history for context.
- `tools/validate-readback/` -- M0 probe + AIE_RW_ACCESS validation.
