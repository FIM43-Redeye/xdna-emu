---
name: 'CHAIN_EXEC_NPU silent-drop captured directly on Phoenix -- first two-sided trace of the dropped op-0x18 message, plus the column-leak wedge cascade decoded'
description: A traced ctrl_packet sweep on the drivers/accel tree (kernel 7.0.9-custom, FW 1.5.5.391, TDR recovery active) reproduced the add_one_ctrl_packet wedge with working mailbox tracepoints for the first time. Prior findings (2026-05-13) inferred the silent-drop from 32s latency clustering; this run caught the dropped message itself. The firmware received MSG_OP_CHAIN_EXEC_NPU (op 0x18, id 0x1d000001) on the per-hwctx channel and never raised the i2x completion interrupt -- confirmed independently by the driver verbose mailbox log (TX present, zero resp data) and the kernel tracepoints (mbox_set_tail present, no mbox_irq_handle/rx_worker/set_head). TDR named the hung message explicitly. The wedge cascade was decoded with firmware status codes: the dropped exec leaves a compute column whose job is hung; DESTROY_CONTEXT then fails AIE2_STATUS_MGMT_ERT_BUSY (0x2000006) because the management firmware cannot reclaim that column; the column leaks; once the pool is exhausted every CREATE_CONTEXT fails AIE2_STATUS_MGMT_ERT_NOAVAIL (0x2000003). The management firmware itself stays alive throughout -- it is a compute-column job hang, not a mailbox-transport death. The op-0x18 firmware path was then reverse-engineered end to end from the decompiled LX7 image: exec ops are served by a per-hwctx APP-ERT RTOS task (FUN_08b04554 -> FUN_08b05194), the op-0x18 handler unconditionally sends a response unless its task blocks, and the task blocks on an RTOS array-completion event-wait (FUN_08b04428(0x10000,0)) that carries no firmware-side timeout -- a single missed completion event wedges the task permanently.
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

## The op-0x18 firmware path -- reverse-engineered

The Phoenix LX7 firmware is statically reverse-engineered -- see
[`2026-05-20-npu-firmware-format.md`](2026-05-20-npu-firmware-format.md)
(format, Xtensa LX7, load base `0x08ad3000`, the Ghidra pipeline). The
pipeline now also emits decompiled pseudo-C for every function
(`DecompileNpuFw.java` postScript -> `analysis-xtensa/decompiled.c`,
623/623 functions decompiled). With that, the op-0x18 path was traced end
to end.

### Two mailbox servers, not one

A correction to
[`2026-05-20-npu-fw-suspend-waitmode-path.md`](2026-05-20-npu-fw-suspend-waitmode-path.md):
`FUN_08ad8480` (main-loop event 0xf) is **not** the host-message processor
for exec ops -- it dispatches firmware-*internal* inter-column events (a
16-bit type field, values `{0x10,0x14..0x17}`). The host mailbox is served
two different ways:

| Host channel | Firmware server | Opcodes |
|---|---|---|
| Management `.145` | `FUN_08ad9ff4` (main-loop event 4 drain) | CREATE/DESTROY ctx (2/3), config, SUSPEND (0x101), `GET_PROTOCOL_VERSION` (0x301) ... |
| Per-hwctx exec `.135/.136` | **APP-ERT task** `FUN_08b04554` -> `FUN_08b05194` | exec ops 0xc, 0x10-0x14, 0x17, **0x18** |

`CREATE_CONTEXT` (`FUN_08ad8dc0`) spawns a per-hwctx **APP-ERT task**: it
calls a task-spawn primitive `FUN_08ae05c0(entry=FUN_08b04554, type, col,
..., channel=col+0x10)`. That task is a separate RTOS thread; it owns the
hwctx's exec mailbox channel. op-0x18 `CHAIN_EXEC_NPU` is serviced there,
not by the management server. The opcode dispatch in `FUN_08b05194` matches
the driver's `aie2_msg_priv.h` enum exactly (0xc, 0x10-0x14, 0x17, 0x18).

This matches the driver's `MGMT_ERT` vs `APP_ERT` status namespaces: the
management server is the MGMT ERT, each per-hwctx APP-ERT task is an APP ERT.

### The op-0x18 handler always sends a response -- unless it blocks

The op-0x18 handler is inline in `FUN_08b05194`. It validates the
`cmd_chain_npu_req` (body size 0x18, `buf_size <= 0x1000` =
`MAX_CHAIN_CMDBUF_SIZE`), copies the chain buffer from host DDR, then loops
over `count` slots of `cmd_chain_slot_npu` (slot size `(arg_cnt + 13) * 4`
-- the exact packed layout from the driver header, cross-checked field by
field).

**Every path through the handler -- success, every validation failure,
every `break` out of the slot loop -- falls through to the response call
`FUN_08b0586c(...)`.** There is no early return and no skipped-response
branch. So a *silent* drop is not a dispatch bug: the only way no response
is sent is for the APP-ERT task to **block inside the slot loop and never
return**.

Per-slot exec descends:

```
FUN_08b05194  (op 0x18, slot loop)
  -> FUN_08b04280                  RTOS syscall  (in-loop)
  -> FUN_08b04638                  per-slot exec
       -> FUN_08b04428(0x10000,0)  RTOS syscall: WAIT for event 0x10000
       -> FUN_08b0a544 -> FUN_08b0a260 -> FUN_08b0aa68 -> FUN_08b0a9e8
            -> FUN_08b0e9a0        NPU command-stream interpreter
```

`FUN_08b0e9a0` is the command-stream interpreter -- a **bounded** loop over
a buffer of register-write / block-write / descriptor-enqueue ops, decoding
AIE-array MMIO addresses (`(word >> 0x10) * 0x2000000 + base + ...`). It
does **not** spin: the op handlers (`FUN_08b0ecf0` writes a DMA BD at array
offset `0x1d000`, `FUN_08b0ed6c` at `0xa0000`, ...) just write descriptors
and advance the cursor. The interpreter programs the array and returns.

The blocking points are RTOS **syscalls**. Each `syscall`-wrapper builds a
request struct on the stack -- `{u32 number, u32 retval, args...}` -- and
passes its pointer in a user register before the `syscall` instruction
traps (confirmed in `disasm.txt`; the syscall number is a `movi` constant
in each wrapper). The recovered number map: `FUN_08b04280/042c8/04354` =
0x64 (7-arg; `042c8` is the host-DDR chain-buffer copy), `FUN_08b04334` =
0x65, `FUN_08b0430c` = 0x66, `FUN_08b04398` = 0x68, `FUN_08b04404` = 0x6a,
`FUN_08b04428` = **0x6b**.

The decisive one is **`FUN_08b04428(0x10000, 0)`** inside `FUN_08b04638` --
syscall 0x6b, the RTOS event-wait. Its wrapper marshals **exactly two
arguments**: a 32-bit event mask (struct +0x08) and a one-byte mode flag
(struct +0x0c). There is no third slot -- the wrapper structurally *cannot*
carry a timeout. The mode byte is block-vs-poll, not a timeout: the only
non-blocking caller is the task's message-dequeue in `FUN_08b04554`,
`while (FUN_08b04428(0xffffffff,1) == 5)` -- mode 1, spin while status is 5.
Every exec/config caller uses mode 0.

And in the blocking form the syscall **return value is discarded** -- the
exec path calls `FUN_08b04428(0x10000,0);` as a bare statement. The result
is delivered through a *shared-memory slot* instead: `FUN_08b04638` puts
`&iStack_28` (a pre-zeroed local) into the request struct it submits via
`FUN_08b044b8`, waits, then reads `iStack_28`. If that wait ever returned
*without* event 0x10000 having fired, the caller would read `iStack_28 == 0`
and treat a non-completion as "success, result 0." There is no code path
that observes or recovers from a wait that does not complete -- the firmware
exec path has **no timeout handling at all**, independent of whatever the
kernel-side syscall does.

### The kernel-side dispatcher -- not decoded, and why it does not matter

Confirming whether syscall 0x6b *itself* (kernel side) arms an internal
timer would mean decoding the RTOS syscall-exception handler. That code is
not among the 623 call-graph-reachable functions -- vector handlers have no
`entry` prologue for `SeedFunctions.java` to key off. A raw disassembly of
the pre-code region (`tools/ghidra-scripts/DisasmRange.java` ->
`analysis-xtensa/vectors.txt`) found it is ~90% zero-fill, with the Xtensa
vector table at `VECBASE ~= 0x08ad3800` (six window over/underflow vectors,
unmistakable 0x40-spaced) and a real exception handler near `0x08ad3a78`
(`rsr EXCCAUSE` / `wsr EXCSAVE3` / `rsr DEPC` + the `addi 3`
skip-past-`syscall` idiom). Decoding the per-number dispatch precisely is a
multi-hour exercise -- it needs correct per-vector instruction alignment,
and Ghidra's Xtensa module never decoded a single `rfe`/`rur` anywhere, so
its exception-return modelling is suspect.

It is also **moot for this bug**. Even if syscall 0x6b kernel-side had an
internal timeout, the caller discards the return and reads the shared slot
-- an internal timeout would surface as a silent wrong result, not as
recovery. With no internal timeout it blocks forever. Both outcomes produce
exactly the observed wedge. The reportable claim -- *the firmware exec path
has no timeout handling* -- is established from the caller side and does not
depend on the dispatcher.

### Conclusion

The silent drop is the **APP-ERT task for the hwctx blocking forever on an
RTOS completion-event wait** (event 0x10000) -- or on one of the other
in-loop syscalls -- because a prior AIE-array operation (DMA, lock, kernel)
never signalled completion. The task never returns to `FUN_08b0586c`, so
no mailbox response and no i2x interrupt are produced. The host sees a
clean `set_tail` and nothing else; TDR fires at 5 s.

This is a firmware **robustness gap**, not a dispatch bug and not
necessarily broken silicon: the firmware waits on an array-completion event
with no firmware-side timeout, so a single missed completion wedges the
APP-ERT task permanently. The ~6% is whatever upstream condition
occasionally drops that completion event.

It also explains the cascade precisely. The wedged APP-ERT task still owns
its column. `DESTROY_CONTEXT` (mgmt op 3, `FUN_08ad9344`) asks the MGMT ERT
to tear that context down; the MGMT ERT cannot reclaim a column whose
APP-ERT task is blocked in an RTOS wait -> `MGMT_ERT_BUSY` -> column leak ->
eventual `MGMT_ERT_NOAVAIL`. The MGMT ERT (`FUN_08ad9ff4`) stays alive and
keeps answering correctly throughout -- exactly as observed.

## Side observation: `Get bo 4 failed` every iteration

`[drm] *ERROR* aie2_hwctx_cfg_debug_bo: Get bo 4 failed` fires on **every**
sweep iteration, healthy ones included. It is the trace sweep's debug-BO
config path failing to bind BO index 4. It is **not** the wedge trigger --
healthy iterations log it and still complete the op-0x18 exec -- but it is
a separate `drivers/accel` defect (the trace-injection debug BO is not
wired up correctly on this tree) and deserves its own finding.

## What is known vs. still unknown

**Known (observed + reverse-engineered):** the drop is real, on op-0x18
execs (the only working exec opcode for this driver/FW 5.8 combination --
see the legacy-path probe above), at the firmware completion-IRQ level;
the mgmt transport stays alive; the wedge is a column-leak cascade with
named MGMT ERT status codes; rate ~6%.
The op-0x18 handler `FUN_08b05194` unconditionally sends a response unless
its APP-ERT task blocks; the task blocks on RTOS syscalls -- notably the
`FUN_08b04428(0x10000,0)` array-completion event-wait, which carries no
firmware-side timeout. The `DESTROY -> BUSY` confirms the column's job
genuinely did not finish: this is a real mid-exec hang, not a finished job
with a lost response.

**Unknown:** *which* in-loop syscall the task is parked in -- the event-wait
`FUN_08b04428(0x10000,0)`, the host-DDR copy `FUN_08b042c8`, or
`FUN_08b04280` -- and *what* upstream array condition drops the completion
event (a DMA that never reports done, a lock never released, a kernel
deadlock). Pinning the exact park point needs firmware/array register state
at wedge time (the `bridge-trace-runner --snapshot-on-timeout` path). No
firmware-trace visibility: Phoenix's `npu1_fw_feature_table` carries no
`AIE2_FW_TRACE` bit, so the DPT firmware-trace path is unavailable here.

## Recovery

Driver reload (`modprobe -r amdxdna && modprobe amdxdna`) recovered the
device cleanly -- the HW runner had exited, no D-state process pinning
`/dev/accel/accel0`, mgmt mailbox responsive. `xrt-smi validate` afterward:
129 us latency, PASSED. This is the benign wedge class (see
[`2026-05-22-ctrl-packet-wedge-drivers-accel.md`](2026-05-22-ctrl-packet-wedge-drivers-accel.md)).

## Legacy-path probe -- attempted, closed off (2026-05-22)

The plan was to force `aie2_msg_init` onto `legacy_exec_message_ops` by
patching `npu1_fw_feature_table` (`npu1_regs.c:70`, `min_minor` 8 -> 99 so
`AIE2_NPU_COMMAND` never matches FW 5.8), re-sweep, and see whether the
wedge is op-0x18-specific. It does not work as a control -- **the legacy
path is broken at the driver level on this tree.**

With `AIE2_NPU_COMMAND` off, a single `xrt-smi validate` execbuf went out
as **opcode 0x13** (`MSG_OP_CHAIN_EXEC_DPU`). The firmware *responded*
correctly (`0x13` TX 24 B -> RX 12 B -- the firmware op-0x13 handler
`FUN_08b04980` works), but the **driver's response callback returned
`-EINVAL`** (`xdna_mailbox.136: Message callback ret -22` ->
`Unexpected ret -22, disable irq` -> `Channel in bad state` ->
`aie2_cmdlist_single_execbuf: Send message failed`). `xrt-smi validate`
FAILED on the first exec; no test ran. Deterministic, immediate, 100% --
the opposite of the probabilistic ~6% op-0x18 silent-drop.

So `legacy_exec_message_ops` has bit-rotted on `drivers/accel` -- which is
*why* `AIE2_NPU_COMMAND`/op-0x18 was added: the op-0x18 path is the only
working exec path for FW 5.8. The legacy path is dead as both a workaround
and a diagnostic. The probe left the device healthy (clean TDR recovery,
no hardware wedge); the driver was reverted to the op-0x18 baseline and
`xrt-smi validate` passed.

The op-0x18-specificity question therefore stays open -- but note the
suspect `FUN_08b04428(0x10000,0)` event-wait lives in `FUN_08b04638`,
shared by *all* exec opcodes (0xc, 0x10-0x14, 0x18), so the hang is most
likely below the per-opcode layer regardless.

Probe mechanics (for repeat): the baseline driver is the DKMS tree
`/usr/src/xrt-amdxdna-2.23.0/` (= branch `fix/mailbox-timeout-msg-uaf`
@ `b1d58df`, the capture-run module). `build.sh -release` fails on an
unrelated XRT/boost configure error and `-refresh_dkms` is not on
`emu-shim-base`; the working path is to patch `npu1_regs.c` in the DKMS
tree directly, `dkms build/install --force -k $(uname -r)` (DKMS
auto-signs with the MOK -- `kmodsign` + `MOK.priv`/`MOK.der`), then
`modprobe -r/modprobe`.

## Next

- **Snapshot on timeout.** Re-run with `bridge-trace-runner
  --snapshot-on-timeout <dir>` to capture CORE/DMA/lock register state on
  the `run.wait` timeout, before TDR wipes it -- identifies which column
  and which DMA/lock the array job is stuck on. This is the remaining route
  to *what* upstream condition drops the completion event.
- **Report it.** The firmware exec path has no timeout handling (above);
  the driver already relies on TDR as the sole backstop (`aie2_ctx.c`
  comment: TDR terminates the ctx "if firmware doesn't respond"). Worth an
  `xdna-driver` issue framed as the driver-interface gap -- op-0x18
  `CHAIN_EXEC_NPU` can silently drop with no firmware-side timeout.

(The "decode the RTOS syscall dispatcher" route is closed -- see *The
kernel-side dispatcher* above: moot, the caller-side proof is decisive.)

## Artifacts

`build/experiments/2026-05-22-ctrl-packet-wedge-drivers-accel/traced-rerun/`:
- `amdxdna.trace` -- kernel tracepoints, 846 entries (the wedge tail)
- `amdxdna.dmesg` -- driver verbose mailbox log across the run
- `add_one_ctrl_packet.{chess,peano}.sweep.log` -- the failed sweep logs
- `sweep-stdout.log` -- full bridge sweep stdout

Firmware analysis (regenerate with `tools/ghidra-npu-fw.sh analyze`):
`ghidra-projects/npu-fw/analysis-xtensa/`:
- `decompiled.c` -- Ghidra pseudo-C, 623/623 functions (`DecompileNpuFw.java`)
- `disasm.txt` -- raw Xtensa disassembly (exact; cross-check load-bearing claims)
- `functions.tsv`, `strings.tsv` -- function + string tables
- `vectors.txt` -- raw disassembly of the pre-code region `08ad3000-08ad5568`
  (Xtensa vector table; `tools/ghidra-scripts/DisasmRange.java`, run
  separately against the existing project with `-process -noanalysis`)
