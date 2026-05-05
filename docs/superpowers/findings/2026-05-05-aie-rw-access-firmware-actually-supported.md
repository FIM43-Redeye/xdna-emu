---
name: AIE_RW_ACCESS is firmware-supported on Phoenix; driver op-table was wrong
description: Phoenix NPU1 firmware (1.5.5.391) does implement MSG_OP_AIE_RW_ACCESS. The previous "firmware never implemented" diagnosis was wrong — the only obstacle was a missing entry in the driver's npu1_regs.c op-table. Fix and verification documented here. Supersedes the 2026-05-04 read_aie_reg conclusion.
type: project
---

# AIE_RW_ACCESS works on Phoenix — driver op-table was the bug

## TL;DR

`xrt::hw_context::read_aie_reg` was failing on NPU1 because the driver's
`npu1_regs.c` op-table omits `MSG_OP_AIE_RW_ACCESS`. Phoenix firmware
1.5.5.391 (mailbox protocol 5.8) implements the opcode just fine. Adding
one entry to the op-table unblocks the production XRT path entirely.

This supersedes both the original "no open-source definition" theory and
the subsequent "Phoenix firmware never implemented opcode 0x203" theory
in `2026-05-04-control-path-cycle-calibration.md`. Both were wrong.

**Why:** the diagnosis-ladder collapsed on the very last rung. We
correctly traced the gate to `aie2_is_supported_msg(MSG_OP_AIE_RW_ACCESS)`
returning false because the op-table on NPU1 lacked the entry, and we
correctly noted that NPU4's op-table has it at FW 6.24+. The wrong
inference was assuming the table absence meant "firmware never shipped
support." It only meant "AMD never updated this driver table." The
firmware itself was not consulted.

**How to apply:** when an opcode is gated by `aie2_is_supported_msg`,
test it with raw mailbox before assuming firmware support is missing.
Op-tables are documentation that drifts.

## Evidence

### Test infrastructure

Added `test_case04` to `xdna-driver/src/driver/amdxdna/aie2_debugfs.c`.
It extends the existing `nputest` debugfs file: takes `(opcode,
req_size_words, resp_cap_bytes, [request_words...])` from a userspace
write and sends the bytes raw via `xdna_mailbox_send_msg` on the mgmt
channel, dumping firmware's response to dmesg. Bypasses
`aie2_is_supported_msg` entirely. Production paths still respect their
gates; only this debugfs side-channel is unguarded.

Required root (debugfs file mode 0600). After signing the rebuilt
module with the local MOK key, no other steps.

### Sanity check — echo opcode 0x101010

```bash
echo "4 0x101010 4 16 4 0xdeadbeef 0xcafebabe 0xfeedface" > nputest
```

Firmware ack'd with a 4-byte response. Confirms our debug path works
end-to-end.

### The real test — opcode 0x203, REG_READ TIMER_LOW

Request layout (`aie_rw_access_req`, 24 bytes packed):

| word | bytes (LE)   | field                                   |
|------|--------------|------------------------------------------|
| 0    | `00000002`   | type = AIE2_ACCESS_TYPE_REG_READ         |
| 1    | `00000200`   | ctx_id=0, row=2, col=0, reserved=0       |
| 2    | `000340f8`   | aie_offset = TIMER_LOW                   |
| 3    | `00000000`   | reg.write_value (unused for read)        |
| 4-5  | `0,0`        | union tail (mem variant pad)             |

Five sequential reads:

```
test4 resp: 00000000  00000000 00054174   ; status=SUCCESS, val=344948
test4 resp: 00000000  00000000 000575ec   ; +13432
test4 resp: 00000000  00000000 0005aa1e   ; +13362
test4 resp: 00000000  00000000 0005de50   ; +13362
test4 resp: 00000000  00000000 00061282   ; +13362
```

Diff between reads ≈ 13.4k cycles at 400 MHz MP-NPU = 33.5 μs of NPU
time per read, matching the shell-loop wall time of ~13 ms per
iteration (kernel mailbox round-trip is fast; rest is shell + driver
overhead between writes). Status field is `AIE2_STATUS_SUCCESS = 0x0`
on every call. Firmware understood the opcode, dereferenced the
register, returned a valid monotonic counter.

### Driver-side fix

```c
/* xdna-driver/src/driver/amdxdna/npu1_regs.c */
const struct msg_op_ver npu1_msg_op_tbl[] = {
    { AIE2_FW_VERSION(5, 8), MSG_OP_CHAIN_EXEC_NPU },
    { AIE2_FW_VERSION(5, 8), MSG_OP_AIE_RW_ACCESS },  /* added 2026-05-05 */
    { 0 },
};
```

After rebuild, sign, reload, the production XRT path
(`xrt::hw_context::read_aie_reg`) goes through `aie2_aie_tile_read` →
`aie2_rw_aie_reg`, where `aie2_is_supported_msg(MSG_OP_AIE_RW_ACCESS)`
now returns true. End-to-end smoke test:
`bridge-trace-runner --read-perf-counter` returns
`perf_ok:true, core_cycles:0` (no longer EOPNOTSUPP).

The zero comes from a *separate* bug — see "Remaining lifecycle bug"
below.

## Remaining lifecycle bug in bridge-trace-runner

`hwctx->num_col` is set inside the runqueue connect path, which fires
on first kernel launch (`aie2_ctx_runqueue.c:316`, in
`aie2_ctx_connect`-adjacent code). Until `run.start()` triggers
that connect, `num_col == 0`. The partition-range check at
`aie2_pci.c:1635` (`access.col >= hwctx->num_col` with num_col=0
becomes `0 >= 0`) fails with EINVAL.

`bridge-trace-runner.cpp:1715-1731` does perf-counter setup
(`read_aie_reg(PERF_CTRL0)` + `write_aie_reg(PERF_CTRL0, ...)` +
`write_aie_reg(PERF_COUNTER0, 0)`) BEFORE `run.start()`. All three
hit EINVAL. Catch block silently logs in verbose mode.

The post-wait read at line 1740 succeeds because by then the partition
has been allocated. But it reads whatever uninitialized value PERF_COUNTER0
had (the firmware default is 0 since we never configured anything).

Three workable fixes (none implemented yet):
1. Pre-warm: do an empty kernel launch (or just access the context in a
   way that triggers connect) before doing the perf setup.
2. Stateful: read PERF_COUNTER0 *both* before and after the run; treat
   delta as the cycle count. The pre-run read also fails on the first
   call, but for runs 2..N we have a baseline from run N-1's post-read.
3. Driver patch: allocate partition at hwctx creation rather than at
   first kernel launch. Larger blast radius; risks regressing real
   workloads that count on lazy allocation.

## A bug we encountered along the way

Initial test_case04 implementation stack-allocated the response struct
(`struct aie2_dbgfs_raw_resp rbuf;`) and passed `&rbuf` as the mailbox
message handle. When firmware stopped responding to a malformed
register-write attempt, our `wait_for_completion_timeout` returned 0
and we vfree'd + returned. The message remained on the mailbox channel
queue. Later, `aie2_pm_fini` triggered a runtime suspend whose own
mailbox call also timed out, leading to `xdna_mailbox_release_channel`
sweeping the queue and invoking each pending message's `notify_cb` —
including ours, on the long-since-freed stack frame:

```
BUG: unable to handle page fault for address: ffffd184a8387cc8
RIP: 0010:aie2_dbgfs_raw_resp_cb+0x9 [amdxdna]
Call Trace:
  xdna_mailbox_release_channel
  aie2_send_mgmt_msg_wait
  aie2_set_runtime_cfg
  aie2_runtime_cfg
  aie2_pm_fini
  aie2_hw_stop
  aie2_hw_suspend
  amdxdna_pmops_suspend
  pci_pm_runtime_suspend
```

The instruction at `aie2_dbgfs_raw_resp_cb+0x9` is `mov %rdx, 0x10(%rdi)`
— `r->actual = size`, where r was a freed stack pointer.

Fix: heap-allocate `aie2_dbgfs_raw_resp` and `req_words`, free only on
the wait-success path, intentionally leak on timeout. Loss is bounded
to ~1KB resp + 1KB req per timed-out call, in a debugfs-only
diagnostic path. See `aie2_debugfs.c` struct and function comments.

The same lifetime bug almost certainly exists in pre-existing
`test_case02` — `DECLARE_COMPLETION(comp)` puts the completion struct
on the stack, and a timed-out echo would leave a dangling pointer in
the same way. We did not fix it (out of scope; test_case02 is AMD's
code; firmware echo rarely hangs in practice).

A second lesson from the same incident: writing to per-tile registers
(PERF_CTRL0) via the raw mailbox path without an allocated hardware
context wedges the firmware. Reads are safe (TIMER_LOW worked fine
without context), but writes that target tile state need a partition.

## Updated picture of the read_aie_reg path

```
xrt::hw_context::read_aie_reg
  └─ xrt_core::query::aie_read       [open shim, xdna-driver/src/shim/]
      └─ DRM_AMDXDNA_AIE_TILE_READ   [open driver, aie2_pci.c:1564]
          └─ aie2_aie_tile_read
              └─ partition-range check (NEEDS hwctx->num_col != 0)
                  └─ aie2_rw_aie_reg
                      └─ aie2_is_supported_msg(0x203) ◄── op-table check
                          └─ npu1_msg_op_tbl              ◄── WAS missing entry
                              [now: { 5.8, MSG_OP_AIE_RW_ACCESS }]
                          └─ true → proceed
                      └─ xdna_send_msg_wait
                          └─ firmware returns reg value
```

Both gates have understandable purposes:
- The op-table gate prevents the driver from sending opcodes the
  firmware can't handle (avoids hangs).
- The partition-range gate prevents accidental access to other
  contexts' partitions.

Neither was wrong in design — but the op-table fell out of sync with
the firmware capability set on NPU1.

## Followups

- **Submit driver patch upstream**: one-line addition to `npu1_regs.c`,
  plus the test_case04 debugfs feature. AMD will probably accept
  either both or just the npu1_regs.c entry.
- **Fix the bridge-trace-runner lifecycle bug** so cycle measurement
  actually returns nonzero counts. Approach 2 (stateful baseline)
  looks cheapest.
- **Re-test on NPU4 when hardware arrives**: should work without any
  driver changes since `npu4_regs.c` already has the entry. Confirms
  cross-architecture and gives us a comparison baseline.
- **Mine `optional_msg` tables for other missing entries**: if the
  Phoenix table is incomplete for AIE_RW_ACCESS, it's plausibly
  incomplete for other opcodes. Worth a sweep with raw mailbox at
  some idle moment.

## See also

- `2026-05-04-control-path-cycle-calibration.md` — corrected (the
  read_aie_reg analysis at the end of that doc is now stale; this
  doc supersedes it)
- `xdna-driver/src/driver/amdxdna/aie2_debugfs.c` — test_case04 source
- `xdna-driver/src/driver/amdxdna/npu1_regs.c` — op-table entry
- task #356 (read_aie_reg verification, in-progress)
