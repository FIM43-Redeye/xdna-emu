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

### MSG_OP_START_FW_TRACE (0x10F) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06 via the per-opcode-rebuild approach. Probe entry
added to `npu1_msg_op_tbl[]` (paired with `MSG_OP_STOP_FW_TRACE` so the
teardown path would be safe if START succeeded), driver rebuilt + DKMS
reinstalled. **No userspace probe extension was needed**: `amdxdna_probe`
auto-fires `aie2_start_fw_trace` during DPT init when a default trace
size is configured. dmesg captured the full handshake:

```
xdna_mailbox.145: req opcode 0x10f size 24 id 0x1d00000f
xdna_mailbox.145: resp opcode 0x10f size 4 id 0x1d00000f
aie2_send_mgmt_msg_wait: command opcode 0x10f failed, status 0x4000002
aie2_start_fw_trace: start fw trace failed, ret 0x4000002
```

The 4-byte response carrying `AIE2_STATUS_INVALID_COMMAND` came through
cleanly thanks to the size-check relaxation in `a155466`; pre-relaxation
this would have surfaced as a generic `-EMSGSIZE`. AMD's table is
correctly conservative for this opcode. Do not add to NPU1 table.

`MSG_OP_STOP_FW_TRACE` (0x110) is presumed unimplemented as well: it's
paired with START in NPU4's table and Phoenix has no other path that
would fire it. Not directly proven (firing STOP requires an active
session, which START can't establish) but not worth a separate probe.

**Methodology refinement**: for opcodes that the driver auto-fires
during init or some other predictable internal path, just add the table
entry -- no userspace probe extension needed. The driver's existing
codepath becomes the trigger; dmesg captures the response. This applies
to START_FW_TRACE (DPT init), GET_DEV_REVISION (mgmt_fw_init), and
CALIBRATE_TIME (init).

### MSG_OP_GET_APP_HEALTH (0x114) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06. Probe entry added to `npu1_msg_op_tbl[]`,
triggered manually by reading the debugfs `get_app_health` file at
`/sys/kernel/debug/accel/0000:c6:00.1/get_app_health`. dmesg:

```
req opcode 0x114 size 16 id 0x1d000010
resp opcode 0x114 size 4 id 0x1d000010
aie2_send_mgmt_msg_wait: command opcode 0x114 failed, status 0x4000002
```

4-byte short reply, INVALID_COMMAND. AMD's table is correctly
conservative.

### MSG_OP_GET_DEV_REVISION (0x117) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06. Probe entry added; auto-fired during driver
bring-up via `amdxdna_vbnv_init -> get_dev_revision`. dmesg:

```
req opcode 0x117 size 4 id 0x1d00000f
resp opcode 0x117 size 4 id 0x1d00000f
aie2_send_mgmt_msg_wait: command opcode 0x117 failed, status 0x4000002
```

4-byte short reply, INVALID_COMMAND. Failure was non-fatal (vbnv
fell back to `default_vbnv`). AMD's table is correctly conservative.

### MSG_OP_CONFIG_FW_LOG (0x116) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06. Probe entry added; auto-fired during driver
bring-up via `amdxdna_dpt_init -> amdxdna_fw_log_init` (the FW log
init runs *before* the FW trace init we previously saw fail). dmesg:

```
req opcode 0x116 size 32 id 0x1d000010
resp opcode 0x116 size 4 id 0x1d000010
aie2_send_mgmt_msg_wait: command opcode 0x116 failed, status 0x4000002
aie2_config_fw_log: Config fw log failed, ret 0x4000002
amdxdna_fw_log_init: Failed to configure FW logging: -22
```

4-byte short reply, INVALID_COMMAND. Failure was non-fatal (dpt_init
emits WARN and the rest of bring-up continues). AMD's table is
correctly conservative.

### Other candidates -- inferred or deferred

| Opcode | Name                          | Status |
|--------|-------------------------------|--------|
| 0x110  | MSG_OP_STOP_FW_TRACE          | Inferred NOT IMPLEMENTED (paired with 0x10F in NPU4 table; can't be tested without an active session, which 0x10F can't establish). |
| 0x111  | MSG_OP_SET_FW_TRACE_CATEGORIES| Moot -- only fires when a START_FW_TRACE session is active. |
| 0x11C  | MSG_OP_CALIBRATE_TIME         | Deferred. Failure mode is fatal-on-init (`aie2_mgmt_fw_init` propagates the error and `amdxdna_probe` bails), and the prior bypass attempt showed mgmt_fw_init failure can wedge the SMU. Recovery is reboot-only. The pattern across all 5/5 directly-tested ops is so consistent that the marginal information from confirming this one isn't worth the risk. |

## Survey-wide conclusion

5/5 directly-tested NPU4-only opcodes return the same fingerprint on
Phoenix: 4-byte short response carrying `0x4000002`
(`AIE2_STATUS_INVALID_COMMAND`).

| Opcode | Name | Verdict |
|--------|------|---------|
| 0x113 | UPDATE_PROPERTY | NOT IMPLEMENTED |
| 0x114 | GET_APP_HEALTH | NOT IMPLEMENTED |
| 0x116 | CONFIG_FW_LOG | NOT IMPLEMENTED |
| 0x117 | GET_DEV_REVISION | NOT IMPLEMENTED |
| 0x119 | GET_COREDUMP | NOT IMPLEMENTED |
| 0x10F | START_FW_TRACE | NOT IMPLEMENTED |

The headline finding is that AMD's NPU1 op-table is correctly
conservative. The only NPU4-vs-NPU1 gap that turned out to be a
driver-omission (rather than a silicon-level absence) was
`MSG_OP_AIE_RW_ACCESS` (0x203), already added in xdna-driver
`289c207`. The firmware-level
logging/tracing/health/coredump/property/revision family is genuinely
absent on Phoenix -- not a driver oversight.

**No half-implementations were found**: every probe either succeeded
cleanly (AIE_RW_ACCESS) or returned the same canonical
INVALID_COMMAND fingerprint. The half-implementation concern that
motivated the conservative dual-validation methodology (verdict +
shim_test #64/#65/#66) turned out to be unwarranted for this opcode
family on this firmware version (Phoenix 1.5.5.391, protocol 5.8).
Future surveys on different firmware versions may behave differently.

## Recommended next moves

1. **Possibly upstreamable: `a155466` xdna_msg_cb size-check
   relaxation.** The patch is upstream-quality, and the principled
   rationale (firmware error replies are short, driver should not hide
   their status behind EMSGSIZE) is now proven by 5 independent
   verdicts. Could be paired with errno translation
   (`AIE2_STATUS_INVALID_COMMAND` -> `-EOPNOTSUPP`) for cleaner UAPI.
2. **Re-survey on next firmware update.** When Phoenix firmware advances
   (current `1.5.5.391`, protocol `5.8`), re-run the auto-fire batch
   (add 0x113 + 0x10F + 0x114 + 0x116 + 0x117 + 0x119 simultaneously,
   modprobe, observe). Each opcode is independent -- adding more table
   entries simultaneously is safe as long as none have a fatal-on-init
   path. CALIBRATE_TIME (0x11C) remains the only deferred opcode.
3. **Methodology to remember**: for opcodes the driver auto-fires
   during a non-fatal init path or via debugfs, just adding the
   table entry is the entire probe -- no userspace extension needed.
   The xdna-driver build cycle (~30s) is fast enough that batching
   N opcodes per rebuild is the cheapest path. Record verdicts in
   commit messages on the xdna-driver branch, revert the entries
   in a single follow-up commit.

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
