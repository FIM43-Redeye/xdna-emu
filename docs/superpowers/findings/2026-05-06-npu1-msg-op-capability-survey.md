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

Two probe paths, increasing in fidelity:

1. **Raw mailbox via debugfs `test_case04`** (yesterday's tool).
   Sends arbitrary opcode + payload directly to the management
   firmware, captures response. Limitations: we have to manually
   reconstruct the request struct shape (firmware rejects malformed
   inputs with INVALID_PARAM, indistinguishable in our sample run
   from "opcode unknown"); cannot allocate DMA scratch buffers for
   ops that need them.

2. **Driver bypass via module parameter** (today's tool, xdna-driver
   `ceecace`). New `unsafe_accept_all_msg` module parameter turns
   `aie2_is_supported_msg()` into a yes-machine. The driver's
   existing per-IOCTL request constructors fire and use the production
   codepath (proper struct layout, DMA buffers if needed). The
   firmware's response is what production XRT would see if we'd
   simply added the opcode to the table.

Path 2 is the trustworthy one. Raw-mailbox is useful for an opcode
without a pre-built request constructor in the driver, but for ops
the driver already knows how to format, the bypass produces cleaner
signal.

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

Enabling `unsafe_accept_all_msg=1` at modprobe time on NPU1 wedges
the SMU and only a full reboot recovers (SBR on the upstream PCIe
bridge does NOT clear it). Cause: the driver's bring-up path
(`aie2_hw_start` -> `aie2_mgmt_fw_init` -> `aie2_runtime_update_prop`)
calls `MSG_OP_UPDATE_PROPERTY` (0x113), which the bypass lets through.
Phoenix firmware returns `INVALID_COMMAND`, the bring-up code treats
that as fatal `-EINVAL`, and the partial-init state leaves the SMU
unable to take subsequent commands.

**Safe pattern**: load the module with the default off, wait for
`/dev/accel/accel0` to appear, THEN toggle the param via sysfs:

```bash
pkexec modprobe amdxdna   # default: bypass off
# ...wait for /dev/accel/accel0...
pkexec sh -c 'echo Y > /sys/module/amdxdna/parameters/unsafe_accept_all_msg'
# trigger the IOCTL of interest
pkexec sh -c 'echo N > /sys/module/amdxdna/parameters/unsafe_accept_all_msg'
```

After a wedge, recover with a reboot, not with PCIe reset.

## Results so far

### MSG_OP_UPDATE_PROPERTY (0x113) -- NOT IMPLEMENTED on Phoenix

Verified 2026-05-06. Bypassed at bring-up; firmware returned
`status 0x4000002 = AIE2_STATUS_INVALID_COMMAND`. AMD's table is
correctly conservative for this opcode. Do not add to NPU1 table.

### Other candidates -- not yet probed

The following are NPU4 table entries absent from NPU1's table; each
needs probing via the bypass-then-trigger pattern, NOT at modprobe
time.

| Opcode | Name                          | How to trigger from userspace |
|--------|-------------------------------|-------------------------------|
| 0x114  | MSG_OP_GET_APP_HEALTH         | DRM_AMDXDNA_FW_LOG via GET_INFO IOCTL or context creation/destroy |
| 0x116  | MSG_OP_CONFIG_FW_LOG          | Setting fw log buffer params (unclear which IOCTL) |
| 0x117  | MSG_OP_GET_DEV_REVISION       | Auto-fired during init -- needs separate hook to retrigger post-init |
| 0x119  | MSG_OP_GET_COREDUMP           | DRM_AMDXDNA_AIE_COREDUMP via GET_INFO IOCTL |
| 0x10F  | MSG_OP_START_FW_TRACE         | DRM_AMDXDNA_SET_FW_TRACE_STATE via SET_STATE IOCTL |
| 0x110  | MSG_OP_STOP_FW_TRACE          | same path |
| 0x111  | MSG_OP_SET_FW_TRACE_CATEGORIES| same path (with categories arg) |
| 0x11C  | MSG_OP_CALIBRATE_TIME         | Auto-fired during init only -- not safely re-triggerable |

`GET_DEV_REVISION` and `CALIBRATE_TIME` are tricky because they are
init-only paths; the only way to probe them via bypass is to
intentionally re-init, which is the unsafe path that wedges the SMU.
For these, the raw-mailbox path remains useful (with the caveat that
INVALID_PARAM from raw-mailbox is not a definitive signal).

## Recommended next moves

1. After NPU recovery (reboot): redo `MSG_OP_GET_DEV_REVISION` via the
   raw-mailbox path with a properly-formatted request (the
   `place_holder` field shape may differ from NPU4).
2. With bypass enabled post-bring-up, trigger `MSG_OP_GET_COREDUMP`
   via the existing `DRM_AMDXDNA_AIE_COREDUMP` IOCTL path -- this
   would be hugely useful for the multi-run-on-same-hwctx hang we hit
   in validate-readback (#355a-related work).
3. Trigger `MSG_OP_START_FW_TRACE` via `DRM_AMDXDNA_SET_FW_TRACE_STATE`
   to see if firmware-level event tracing works on Phoenix.
4. For each opcode that returns SUCCESS: write a minimal patch adding
   it to `npu1_msg_op_tbl[]` and verify the production XRT path now
   works without the bypass. Each becomes its own commit.

## See also

- `xdna-driver` `ceecace` -- the bypass module parameter (this work)
- `xdna-driver` `289c207` -- yesterday's `MSG_OP_AIE_RW_ACCESS` table
  entry (the success template for ops that probe to SUCCESS)
- `xdna-driver` `25b3f51` -- `test_case04` raw-mailbox debugfs path
- `tools/validate-readback/` -- end-to-end probe of the AIE_RW_ACCESS
  path, demonstrates the post-table-fix workflow
