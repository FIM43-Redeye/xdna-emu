---
name: 'CHAIN_EXEC_NPU silent-drop captured directly on Phoenix -- first two-sided trace of the dropped op-0x18 message, plus the column-leak wedge cascade decoded'
description: A traced ctrl_packet sweep on the drivers/accel tree (kernel 7.0.9-custom, FW 1.5.5.391, TDR recovery active) reproduced the add_one_ctrl_packet wedge with working mailbox tracepoints for the first time. Prior findings (2026-05-13) inferred the silent-drop from 32s latency clustering; this run caught the dropped message itself. The firmware received MSG_OP_CHAIN_EXEC_NPU (op 0x18, id 0x1d000001) on the per-hwctx channel and never raised the i2x completion interrupt -- confirmed independently by the driver verbose mailbox log (TX present, zero resp data) and the kernel tracepoints (mbox_set_tail present, no mbox_irq_handle/rx_worker/set_head). TDR named the hung message explicitly. The wedge cascade was decoded with firmware status codes: the dropped exec leaves a compute column whose job is hung; DESTROY_CONTEXT then fails AIE2_STATUS_MGMT_ERT_BUSY (0x2000006) because the management firmware cannot reclaim that column; the column leaks; once the pool is exhausted every CREATE_CONTEXT fails AIE2_STATUS_MGMT_ERT_NOAVAIL (0x2000003). The management firmware itself stays alive throughout -- it is a compute-column job hang, not a mailbox-transport death. Cross-references the reverse-engineered LX7 firmware (event 0xf mailbox transport FUN_08ad8480, event-driven per-column teardown FUN_08ad70b8).
type: project
---

# CHAIN_EXEC_NPU silent-drop captured -- 2026-05-22

## TL;DR

The op-0x18 (`MSG_OP_CHAIN_EXEC_NPU`) firmware silent-drop -- hypothesized
since [`2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`](2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md)
from 32s latency clustering -- has now been **captured directly**, on both
the driver verbose mailbox log and the kernel tracepoints, in the same run.

Prior runs could only infer the drop ("a job took 32s, so its response
must have been missing"). The `amdxdna-trace` daemon that should have shown
the mailbox traffic was silently broken (it hardcoded the obsolete
`src/driver` trace subsystem name; fixed today, commit `a78d1fa`). With it
working, a traced `ctrl_packet` sweep reproduced the wedge and recorded:

1. **The drop.** Firmware received op-0x18 exec `id 0x1d000001` and never
   raised the completion interrupt. No response message, no IRQ.
2. **The cascade, decoded.** One dropped exec leaves a compute column whose
   job is hung. `DESTROY_CONTEXT` then fails `MGMT_ERT_BUSY` -- the
   management firmware cannot reclaim a column with a hung job. The column
   leaks. Once the pool is exhausted, every `CREATE_CONTEXT` fails
   `MGMT_ERT_NOAVAIL`. A single probabilistic drop wedges the whole device.

## Run

| Axis | Value |
|---|---|
| Command | `./scripts/emu-bridge-test.sh --sweep --no-emu ctrl_packet` |
| Driver tree | `drivers/accel/amdxdna` (canonical), module `b1d58df` |
| Kernel | `7.0.9-custom` |
| Firmware | `1.5.5.391` (protocol 5.8) |
| TDR | recovery active (`tdr_dump_only=false`, default) |
| Trace daemon | `amdxdna-trace.sh` post-fix -- `subsystem=amdxdna events=8` |
| Result | `add_one_ctrl_packet` sweep FAIL both compilers; wedge reproduced |

All 7 ctrl_packet tests passed the single-shot bridge run (Phase 3). The
wedge hit the Phase-5b event sweep on `add_one_ctrl_packet`, HW batch 1 --
same test, same failure mode as the prior two ctrl_packet sweeps.

## The drop -- captured two ways

### Driver verbose mailbox log (`dyndbg=+p`)

A healthy iteration creates a context, configures it, runs one op-0x18
exec, destroys the context. The wedging iteration got as far as the exec
(`amdxdna.dmesg` lines 786-791):

```
786  xdna_mailbox.136: opcode 0x18 size 24 id 0x1d000001    <- exec TX (request)
787  req data: 00000018 00010018 1d000001 00000018
788  req data: 00000000 00000000 04000000 00000000
789  req data: 00000070 00000001
       <-- no `resp data` line for id 0x1d000001 ever appears -->
790  *ERROR* aie2_tdr_detect: TDR timeout detected            (+4 s)
791  xdna_mailbox.136: msg_id 0x1d000001 msg opcode 0x18      <- TDR names the hung msg
```

Every other mailbox message in the run -- on the mgmt channel `.145`
(`CREATE_CONTEXT` 0x2, `DESTROY_CONTEXT` 0x3, config 0x106) and the healthy
op-0x18 execs on `.136` -- has a paired TX `req data` + RX `resp data`.
Only `id 0x1d000001` of this iteration has a TX and no RX.

### Kernel tracepoints (`xdna_mailbox.136`, op 0x18)

Every op-0x18 exec is a `mbox_set_tail` (host enqueues) followed by
`mbox_irq_handle` -> `mbox_rx_worker` -> `mbox_set_head` (host consumes the
response). 14-15 healthy execs on `.136` show that clean quartet. The last
`.136` op-0x18:

```
2655.732434  mbox_set_tail: xdna_mailbox.136 id 0x1d000001 opcode 0x18
       <-- no mbox_irq_handle, no mbox_rx_worker, no mbox_set_head -->
2660.267814  mbox_set_tail: xdna_mailbox.135 id 0x1d000001 opcode 0x18   (post-TDR ctx)
```

Event tally invariant for the whole capture: `mbox_irq_handle ==
mbox_rx_worker == mbox_set_head == 194`, against `mbox_set_tail == 200`.
Six submissions never completed; the dropped op-0x18 is one of them.

**Conclusion (observed, not inferred):** the firmware received the op-0x18
`CHAIN_EXEC_NPU` request and never raised the i2x completion interrupt.
The drop is at the firmware completion-IRQ level -- not a lost host write
(the `set_tail` happened), not a PCIe fault.

## The wedge cascade -- decoded

After the drop, `amdxdna.dmesg` shows (timestamps 12:41:25 -> 12:41:34):

```
12:41:25  TDR #1 -- stops channel .136, names msg 0x1d000001
12:41:25  DESTROY_CONTEXT (0x3) -> resp 02000006 -> *ERROR* opcode 0x3 failed, status 0x2000006
12:41:25  CREATE_CONTEXT (0x2) id 0x1d000041 -> resp word0 00000000 -> succeeds (pool not yet empty)
12:41:29  TDR #2
12:41:30  CREATE_CONTEXT (0x2) -> resp 02000003 -> *ERROR* opcode 0x2 failed, status 0x2000003
   ... every subsequent CREATE_CONTEXT fails 0x2000003 ...
```

Status codes from `enum aie2_msg_status` (`aie2_msg_priv.h`), MGMT ERT
block (sequential from `0x2000001`):

| Code | Enum | Where |
|---|---|---|
| `0x2000003` | `AIE2_STATUS_MGMT_ERT_NOAVAIL` | every post-wedge `CREATE_CONTEXT` |
| `0x2000006` | `AIE2_STATUS_MGMT_ERT_BUSY` | post-TDR `DESTROY_CONTEXT` |

Both are **MGMT ERT** codes -- the management embedded runtime, not the
AIE array (`0x1000xxx`) and not the app ERT (`0x3000xxx`). The mechanism:

1. The dropped op-0x18 exec leaves a compute column with a job that never
   completed (or never reported completion).
2. `DESTROY_CONTEXT` asks the MGMT ERT to tear that context down. It
   answers `MGMT_ERT_BUSY` -- it cannot reclaim a column whose job is
   hung. The teardown does not happen.
3. The column is **leaked** -- still marked allocated, never freed.
4. The sweep keeps creating/destroying contexts. One fresh create still
   succeeds right after TDR #1 (free columns remain), but each
   failed-teardown leaks another column.
5. Once the column pool is exhausted, `CREATE_CONTEXT` returns
   `MGMT_ERT_NOAVAIL`. The device is wedged for all new work.

Key point: **the management firmware stays alive the whole time.** It
answers `BUSY`/`NOAVAIL` with correct status codes -- it does not go
silent. This is a *compute-column job hang* that the management firmware
correctly reports it cannot recover from -- not a death of the mailbox
transport. That refines the "harder mode / mgmt mailbox responsive but
refuses contexts" observation in
[`2026-05-22-ctrl-packet-wedge-drivers-accel.md`](2026-05-22-ctrl-packet-wedge-drivers-accel.md):
the refusal is a leaked-resource symptom, not a mailbox failure.

## Frequency

15-ish op-0x18 execs completed cleanly before one dropped (~6%).
Consistent with the 2026-05-13 estimate of ~4% (24/585). The drop is
probabilistic and count/state-dependent -- not tied to a specific input
(2026-05-13 already killed the chess-codegen and slot-type hypotheses;
the wedge is bidirectional flake across both compilers).

## Cross-reference: the firmware is reverse-engineered

The Phoenix LX7 management firmware has been statically reverse-engineered
-- see [`2026-05-20-npu-firmware-format.md`](2026-05-20-npu-firmware-format.md)
(format, Xtensa LX7, load base `0x08ad3000`, the Ghidra pipeline) and
[`2026-05-20-npu-fw-suspend-waitmode-path.md`](2026-05-20-npu-fw-suspend-waitmode-path.md)
(main event loop + dispatch table). That gives the silent-drop a concrete
place to live:

- The FW main loop `FUN_08ad85f8` dispatches on internal event codes.
  Event **0xf -> `FUN_08ad8480`** is the "mailbox transport processor" --
  every mailbox message, including the op-0x18 exec request, is processed
  through there.
- Per-column teardown is **`FUN_08ad70b8`** -- "called from a per-context
  destroy path, not only from suspend." This is the routine `DESTROY_CONTEXT`
  drives. Its inability to complete on a hung column is exactly the
  `MGMT_ERT_BUSY` we observed.

So the silent-drop sits in the op-0x18 exec path: `FUN_08ad8480` accepts
the request and hands it to a compute column, and either the column job
hangs or its completion is never posted back. Reading the decompiled
op-0x18 / `CHAIN_EXEC_NPU` handler reachable from `FUN_08ad8480` -- and the
exec-completion path that should re-arm the i2x interrupt -- is now a
viable investigation route, not just a host-side black box. The
`SeedFunctions` pass (623 functions recovered) means the indirectly-reached
exec handlers are disassembled.

## Side observation: `Get bo 4 failed` every iteration

`[drm] *ERROR* aie2_hwctx_cfg_debug_bo: Get bo 4 failed` fires on **every**
sweep iteration, healthy ones included. It is the trace sweep's debug-BO
config path failing to bind BO index 4. It is **not** the wedge trigger --
healthy iterations log it and still complete the op-0x18 exec -- but it is
a separate `drivers/accel` defect (the trace-injection debug BO is not
wired up correctly on this tree) and deserves its own finding.

## What is known vs. still unknown

**Known (observed):** the drop is real, op-0x18-specific, at the firmware
completion-IRQ level; the mgmt transport stays alive; the wedge is a
column-leak cascade with named MGMT ERT status codes; rate ~6%.

**Unknown:** whether the firmware *hung mid-exec* (the AIE job itself
deadlocked on the column) or *finished the job but failed to post the
response / re-arm the i2x interrupt*. The `DESTROY -> BUSY` strongly favors
an actual column-job hang -- a cleanly finished job's column would be
reclaimable. No firmware-internal runtime visibility: Phoenix's
`npu1_fw_feature_table` carries no `AIE2_FW_TRACE` bit, so the DPT
firmware-trace path is not available here. The decompiled firmware is the
remaining lever.

## Recovery

Driver reload (`modprobe -r amdxdna && modprobe amdxdna`) recovered the
device cleanly -- the HW runner had exited, no D-state process pinning
`/dev/accel/accel0`, mgmt mailbox responsive. `xrt-smi validate` afterward:
129 us latency, PASSED. This is the benign wedge class (see
[`2026-05-22-ctrl-packet-wedge-drivers-accel.md`](2026-05-22-ctrl-packet-wedge-drivers-accel.md)).

## Next

- **Legacy-path probe.** Patch `npu1_regs.c` so `AIE2_NPU_COMMAND` never
  matches -> `aie2_msg_init` selects `legacy_exec_message_ops` ->
  chained execs go out as `MSG_OP_CHAIN_EXEC_BUFFER_CF` (op 0x12). Re-sweep
  traced. If the wedge disappears, the bug is isolated to the op-0x18
  firmware handler (actionable upstream: gate `AIE2_NPU_COMMAND` off for
  NPU1). If it persists, the hang is below the message-opcode layer.
- **Read the decompiled op-0x18 handler** reachable from `FUN_08ad8480`,
  and the exec-completion / i2x-interrupt re-arm path.

## Artifacts

`build/experiments/2026-05-22-ctrl-packet-wedge-drivers-accel/traced-rerun/`:
- `amdxdna.trace` -- kernel tracepoints, 846 entries (the wedge tail)
- `amdxdna.dmesg` -- driver verbose mailbox log across the run
- `add_one_ctrl_packet.{chess,peano}.sweep.log` -- the failed sweep logs
- `sweep-stdout.log` -- full bridge sweep stdout
