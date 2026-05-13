---
name: 'ctrl_packet_reconfig_4x1_cores peano emits NULL-relative addresses, NPU IOMMU page-faults on submission'
description: Peano-compiled ctrl_packet_reconfig_4x1_cores triggers 7 sequential AMD-Vi IO_PAGE_FAULT events at NPU DMA addresses 0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0 in IOMMU domain 0x0001 (flags=0x0007 = read+write+I/O), followed by firmware stuck in DMA-rejected state for 4 seconds, followed by clean TDR recovery (count=1, dump_only=N as of 2026-05-13). Chess-compiled version of the same test passes normally. Other ctrl_packet_reconfig variants (`reconfig`, `_1x4_cores`, `_elf`) pass with both compilers. So the bug is specific to peano + 4x1_cores layout. Tight cluster of zero-page-relative addresses (0x0 + N*0x20 for N=0..6) strongly suggests peano emitted zeroed memory where it should have placed real buffer base addresses. Reportable upstream (llvm-aie / mlir-aie / peano backend).
type: project
---

# ctrl_packet_reconfig_4x1_cores peano IOMMU fault -- 2026-05-13

## TL;DR

When the bridge test runs `ctrl_packet_reconfig_4x1_cores` compiled
with peano, the NPU DMA engine attempts reads at addresses that the
AMD IOMMU has not mapped:

```
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x0  flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x0  flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x20 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x40 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x60 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x60 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x80 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0xa0 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0xc0 flags=0x0007]
[02:26:05] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0xc0 flags=0x0007]
```

The addresses form an obvious pattern: `0x0 + N * 0x20` for N=0..6.
That's not random firmware corruption; that's seven buffer-base
pointers that were never filled in -- staying at their zero
initial values. peano emitted relative or null addresses where the
runtime should have seen real DMA-mapped buffer bases.

After the faults, the firmware is stuck (can't service its DMAs)
and never sends a response on the per-ctx mailbox. 4 seconds later
TDR fires (count=1), dumps ctx state, then with `tdr_dump_ctx=N`
(post-2026-05-13) actually executes `aie2_rq_stop_all/restart_all`,
which cleanly destroys the ctx, signals the pending fence with
`ERT_CMD_STATE_TIMEOUT`, and frees the partition. The test fails
correctly with `xrt::runlist::aie_error: runlist failed execution
(ERT_CMD_STATE_TIMEOUT)`. NPU stays healthy and ready for next test.

## Differential evidence

| Test | Chess HW | Chess EMU | Peano HW | Peano EMU |
|---|---|---|---|---|
| `ctrl_packet_reconfig` | PASS | PASS | PASS | PASS |
| `ctrl_packet_reconfig_1x4_cores` | PASS | PASS | PASS | PASS |
| `ctrl_packet_reconfig_elf` | PASS | PASS | PASS | PASS |
| **`ctrl_packet_reconfig_4x1_cores`** | **PASS** | **PASS** | **TDR** | **BUDGET** |

Same MLIR, same test, only the compiler differs. Chess gets it
right; peano emits zeros. The bug is in the peano backend's
handling of the `_4x1_cores` layout specifically -- the `_1x4_cores`
layout works fine for both compilers.

## What's at addresses 0x0..0xc0?

`flags=0x0007` per AMD IOMMU = read + write + I/O attributes set;
the device asserted a fault on a DMA transaction.

`domain=0x0001` is the NPU's IOMMU domain. Address 0x0 isn't a
valid IOVA in that domain.

Seven addresses `0x0, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0` likely
correspond to seven buffer slots/operands in the workload. Each
slot is 32 bytes = 0x20 -- consistent with a buffer descriptor
table where each entry holds (at least) a 64-bit base address +
metadata. The base addresses in those entries are all zero.

To pin down exactly which buffer descriptors got zeroed, the
compiled artifact needs inspection:

```
build dir: /home/triple/npu-work/mlir-aie/build/test/npu-xrt/ctrl_packet_reconfig_4x1_cores/peano/
```

Useful next steps:
- `xclbinutil --info --input aie.xclbin` to dump section list
- `xclbinutil --dump-section CONTROL_PACKET-...` to extract the
  ctrl_packet payload that gets DMA'd to the NPU; check for zero
  base addresses where instr_buffer / save_buffer / arg pointers
  should be
- Compare with chess-compiled artifact for the same test

## Recovery is clean (no reboot needed)

Pre-2026-05-13 with `tdr_dump_ctx=1`, this would have wedged
permanently via `aie2_hmm_invalidate` waiting forever (see
`2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`). With
`tdr_dump_ctx=N`, recovery is fast and clean:

```
[02:26:09] aie2_tdr_work: Device isn't making progress... Count 1 timeout 2 dump_only 0
[02:26:09] aie2_dump_ctx: Dumping ctx ctx.16647.1, hwctx 5, sub=1, comp=0
[02:26:09] aie2_dump_ctx: JOB[0]: seq: 0, op: 0x13, msg: 0x1d000001, fence: unsignaled
[02:26:09] ctx.16647.1 @[1, 4] stop
[02:26:09] xdna_mailbox.136: IRQ disabled and RX work cancelled
[02:26:09] xdna_mailbox.145: req opcode 0x3 size 4 id 0x1d000026
[02:26:09] xdna_mailbox.145: resp opcode 0x3 size 4 id 0x1d000026
[02:26:09] xdna_mailbox.136: msg_id 0x1d000001 msg opcode 0x18
[02:26:09] xdna_mailbox.136: Mailbox channel released type 1 irq: 136
[02:26:09] ctx.16647.1 yield work
[02:26:09] partition [1, 4] max_hwctx 6 hwctx 0 cnt 0
[02:26:10] PID 16647 close context 1
```

The bridge runner's userspace catches the timeout cleanly:
```
terminate called after throwing an instance of 'xrt::runlist::aie_error'
  what():  runlist failed execution (ERT_CMD_STATE_TIMEOUT)
```

`runlist::aie_error` triggers SIGABRT (uncaught), the runner core-
dumps, and bash reports "Aborted (core dumped)". Cosmetically loud
but functionally correct -- the test is reported FAIL/BUDGET.

## What this is NOT

Distinct from `2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`
(mode A), which was a `add_one_ctrl_packet (chess)` wedge with:
- Zero IOMMU faults
- Pure firmware non-response (32s timing was hardcoded userspace timeout)
- Worked with one compiler (chess), not the other (peano untested in sweep)
- Different opcode in the dump (`op: 0x0` vs `op: 0x13` here)

That mode is a real firmware-side mystery. This one is a peano-
backend bug, and a much more tractable target.

## Reportable upstream

llvm-aie (peano backend) or mlir-aie:

- Project: <https://github.com/Xilinx/llvm-aie> or <https://github.com/Xilinx/mlir-aie>
- Failing test: `mlir-aie/test/npu-xrt/ctrl_packet_reconfig_4x1_cores`
- Symptom: NPU IOMMU page-faults at 0x0, 0x20, ..., 0xc0 when running
  the peano-compiled xclbin on real hardware; chess-compiled xclbin
  works. Test would PASS in EMU regardless (peano EMU also passed).
- Repro:
  ```bash
  # In mlir-aie/build/test/npu-xrt/ctrl_packet_reconfig_4x1_cores/peano:
  ./test.exe   # on real NPU1 (Phoenix); see TDR + IOMMU faults
  ```
- Recovery: TDR with `tdr_dump_ctx=N` cleans up cleanly; no reboot needed.

Before filing, complete the binary diff: which payload field is
the actual zero source?
