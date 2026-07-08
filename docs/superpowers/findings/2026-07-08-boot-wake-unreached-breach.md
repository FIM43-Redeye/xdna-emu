# Boot-to-idle: the wake mechanism is provably UNREACHED (breach reopened)

**Date:** 2026-07-08
**Issue:** #140 firmware-emulation dream / boot-to-idle.
**Branch:** `feat/m2c-mapping-boot-to-idle`.
**Status:** Active. Reopens the boot wall that was banked 2026-07-08 as
"array-gated, not derivable from firmware alone" -- that banking was *imprecise*
and abandoned a live thread. This finding replaces the vague wall with a sharp,
binary, answerable question.

## Why this was reopened

Every recent emulator arc (timer-sync cold-start, send-cadence, the 8000 mailbox
constant, firmware runtime wiring) dead-ends at the same boot-completion wall. We
kept working *around* it. The banked conclusion ("the completion contract lives
in the array's behavior, not derivable from firmware alone") turned out to
contradict the firmware arc's OWN annotation map, which flags the gate-flag
setter as **"still-unidentified (14 disp-0x6c store sites)"** -- i.e. the trace
was abandoned mid-way, then rationalized into a wall. Decision (Maya): drill
*through* it.

## The measured result (decisive)

`m2c_probe_addr_store_watch` on the two wake-state words across **1.5M boot
instructions** (normal boot, no forcing):

| watched addr | meaning | writes in boot |
|--------------|---------|----------------|
| `0x22bc` (`[SCHED 0x2250 + 0x6c]`) | scheduler event-pending bitmask | **0** |
| `0x10f40` (`[task 0x10f10 + 0x30]`) | task 0x10f10's pending flag | **0** |

**Neither is ever written.** The wake mechanism is not mis-sequenced or
mis-timed -- the code that completes task `0x10f10` simply never runs in
reachable boot. `force_done` (a direct write of `[0x10f40]=1`) advances boot
58k -> 623k, proving the entire downstream is intact; only the *origin* is
missing.

## The structural lock

Task `0x10f10`'s run-fn is `0x588c` = `sched_event_poll` (`FUN_00005580+0x30c`;
polls event source `0x27010d28`, builds a mask, calls `wake_tasks_by_event_mask`).
Its await-mask `[task+0x38]` is **0**. The waker skips any task where
`delivered & [task+0x38] == 0`, so `0x10f10` is **structurally un-wakeable by the
event system**. Its flag must be set by a *direct* write -- exactly what
`force_done` does and what nothing in boot does.

## Candidate setters (static, disp-0x6c = writes to `[base+0x6c]`)

14 sites total. Known consumers: `deliver_pending_events+0x19` (`0xcaf5`),
`wake_tasks_by_event_mask+0x2b` (`0xd877`). Scheduler-cluster candidates: `0xc9dc`,
`0xcb38`, `0xcf5c`, `0xd134`, `0xd21c`, `0xd53c`. Stack-local (base a1, likely not
`0x22bc`): `0x5e30`, `0x34ef8+0x246`, `0x37218`. Init-region: `0x3fb8`, `0x9ce0`,
`0xc3f8`. **Two sites (`0x34ef8`, `0x37218`) sit in the dormant AIE2 DMA-HAL
region** (`_XAieMl_Dma*`, ops table `0x35004..0x35048` built by `FUN_0000b5d4`),
independently corroborating Session-9's "the real fw installs+calls the DMA HAL
in its own bring-up; EMU diverges by not doing so."

## Sibling pin: the Wall-C column index

Past the 58k wall (via forced completion), the 623k abort is `FUN_00007fa0`'s
`Bgeui a7,6 -> j.`. Traced (`m2c_probe_col_cmd_trace`): the column index is read
from `[0x10f18]` = `[task 0x10f10 + 8]` and holds `0xff` (the unassigned sentinel
set at task init by `FUN_00004570@0x46a9`). It flows into the descriptor at
`0xfae0` (`{opcode=0xf, target-task=0x9140, col-idx=0xff}`) -> the gate's `a7` ->
abort. **Caveat:** the `0xff` is partly a *forcing artifact* -- the probe writes
`[task+0x30]` but not `[task+8]`, whereas the real completion agent plausibly
writes both together (matches Session-7 "event supplied without the powered-column
STATE").

## The question that remains (binary, answerable)

Is task `0x10f10`'s completion produced by:

- **(A) Divergence** -- firmware setup code (arming its await-mask, or the DMA-HAL
  bring-up that would generate the event) that our interp fails to reach due to a
  wrong branch / missing side-effect. Outcome: **fix the interp, boot advances,
  zero new modeling.** Strongest lead, corroborated twice (DMA-HAL dormant).
- **(B) External** -- a hardware agent (SMU/PSP/array) writes `[task+0x30]`
  (and `[task+8]`) directly. Outcome: **model the exact minimal contract**, now
  fully pinned by the above.

## Next pull

Trace `0x10f10`'s full lifecycle: who spawns it, and what code *should* arm its
await-mask `[task+0x38]` (or directly complete it) -- and why boot never reaches
that code. That settles (A) vs (B).

## BREACH RESULT (same session): the contract lever works

Determination is **(B) external**, contract pinned. Evidence the completion is
handed to an external consumer: the descriptor at `0xfae0` is **cache-flushed
(`Dhwbi`)** to shared memory by the builder before the gate `FUN_00007fa0`
submits it; the gate's submit path calls only firmware-internal functions (no
`0x27xxxxxx` MMIO write). `colmask=0xf` = ungate all 4 Phoenix columns
(SMU/PSP, per Session-6).

**Decisive experiment (`m2c_probe_colassign_boot`, new):** every prior force-done
set only `[task+0x30]` and hit Wall C because `[task+8]` stayed `0xff`. Setting
**both** `[task+0x30]=1` AND `[task+8]=col 0` for the worker at the dispatcher
done-check (`0xd828`, fires once at n=47896) **cleared Wall C**: boot then ran the
full 3M-instruction budget with **no abort, no `Unknown`, no spin** (vs the old
58k wedge / 623k abort). The pinned contract -- flag + valid column, together --
is the lever nobody had pulled.

Trajectory note: `max_pc=0x2000e6f2` and a `0x2000_0000` region in the PC
histogram are a **benign low-memory alias** (Harvard overlay), reached at n=1025
during normal early boot -- not a crash.

**Alive/idle drill (same session): NOT idle yet -- one more gate.** After the
worker completes, over the full 3M budget: **min INTLEVEL = 2** (never 0, i.e.
never receive-ready idle) and the **FW_ALIVE magic `0x55504e5f` is never stored**.
So the breach clears Wall C but boot settles into a *further* INTLEVEL-2 stall
short of alive+idle. The multi-wall chain is real: **two walls down (58k
pending-flag, 623k Wall-C column), >=1 more to go.** Likely-next lead: the
descriptor's `target-task=0x9140` and whitelist task `0x9040` (which never parked
this run) -- a second worker completion. **Next drill:** find the third gate the
same way -- what state the post-Wall-C boot now waits on (which task parks, at
what flag), and whether the same flag+column contract clears it.

## Third-gate drill RESULT: the "gate" is the idle scheduler loop, not a wall

Extended `m2c_probe_colassign_boot` to bucket the *post-breach tail* by nearest
symbol and record the load addresses it polls. Over the ~2.95M instructions
after the 0x10f10 completion, boot rests in a **steady, cycling scheduler tick**
(~6000 cycles), not a stuck spin:

| rest function | hits | what it does |
|---------------|------|--------------|
| `FUN_0000c928+0x39` | 194k | HW `Loop` (32-iter) `Extui`+`AddN` = **popcount over a ready-mask** |
| `FUN_00007bf0+0x32` | 37k | **task-table scan**: `L8ui [a4+8]`/`L8ui [a4+4]` (SCHED `0x2250`+4/+8), OR + `Bbci bit0` |
| `FUN_0000c96c+0x17` | -- | `Rsil imm:2` -- **the SOURCE of the INTLEVEL=2 floor** (level-2 critical section) |

Top polled loads: `0x2254`/`0x2258` (SCHED+4/+8, 37524 each), `0xf9e0` (pending
struct, 6065). No `WAITI` executes (the probe's `Step::Wait` idle-break never
fires) -- so it is a **busy-poll idle**, not a halt.

**Reframe.** The `min INTLEVEL = 2` is NOT a third wall -- it is merely the
scheduler's `Rsil(2)` critical-section level. Past the two real walls, boot runs
its scheduler ready-loop, which matches the long-standing hypothesis
(`m2c_probe_faithful_smu_boot` comment) that the fw idles waiting for a host
mailbox command and programs the array only in response. Two walls were the
whole boot obstruction.

Two items remain, different in kind:
- **FW_ALIVE false-negative risk.** The detector only catches an `S32i` of the
  literal `0x55504e5f` ("_NPU") *from a register*. A byte-wise or MMIO write of
  the alive marker would be missed -- absence is not proof boot isn't alive.
- **Idle vs. dead busy-loop (decisive test).** Inject the host mailbox
  **doorbell** from this post-breach state; if boot leaves the scheduler loop
  into the mailbox-receive path, it was idle-waiting (= effectively booted to
  idle). If nothing changes, it is a real busy sub-wall. This is the next pull.

## Doorbell + bit3-shim tests: BOTH negative -- wrong wake mechanism

Built `m2c_probe_breach_doorbell` (breach to the real post-Wall-C loop, THEN
inject the mailbox doorbell -- unlike `m2c_probe_mailbox_wake`, whose warmup
wedges pre-Wall-C and tested the doorbell from the wrong state):

- **Doorbell (bit0 IRQ): negative.** INTLEVEL is pinned at 2 the whole
  post-breach run, so a level-1 IRQ is masked and never taken (pending bit0
  never clears). Boot never leaves the scheduler cluster.
- **+ continuous DMA-HAL bit3 shim: negative.** The column IS serviced
  (`FUN_00008c68` runs its full `0x2727_col000` handshake) but INTLEVEL still
  never drops below 2. bit3 is not the gate that lowers the interrupt level.

The reason both fail: reading `src/firmware/host_mailbox.rs` + the xdna-driver
protocol shows the **RX path is not this interrupt**. See below.

## x2i RECEIVE PATH mapped (the real receive-ready gate)

**Authoritative protocol (xdna-driver `src/driver/amdxdna/`):**
- The host submits a job by `memcpy`-ing the packet into the **SRAM ring** then
  a **single `writel` to the x2i tail-pointer register** (`amdxdna_mailbox.c`
  `mailbox_send_msg`:271-312). That tail write is the *only* notification --
  there is **no separate x2i doorbell/interrupt register** (contrast i2x, which
  has `intr = head+4`). The fw side either polls the x2i tail reg or the mailbox
  block raises an Xtensa IRQ on that write; the driver can't distinguish.
- The x2i register addresses are **firmware-defined, not driver-fixed**: at boot
  the fw publishes `struct mgmt_mbox_chann_info { x2i_tail,x2i_head,x2i_buf,
  x2i_buf_sz, i2x_tail,i2x_head,i2x_buf,i2x_buf_sz, magic(0x55504e5f), msi_id,
  ... }` and the host reads it from SRAM at the FW_ALIVE pointer
  (`aie2_pci.c:65-79,229-263`).

**Located in OUR firmware image.** The `_NPU` magic `0x55504e5f` occurs exactly
once, at file `0x3388` = struct base `0x3368 + 0x20` (exact field alignment).
The full block is present in the image:

| off | field | value | | off | field | value |
|-----|-------|-------|-|-----|-------|-------|
| 0x3368 | x2i_tail | 0x0000f190 | | 0x3378 | i2x_tail | 0x030ec000 |
| 0x336c | x2i_head | 0x0000f1a0 | | 0x337c | i2x_head | 0x030ec004 |
| 0x3370 | x2i_buf | 0x00010f6c | | 0x3380 | i2x_buf | 0x030ed000 |
| 0x3374 | x2i_buf_sz | 0x00014800 | | 0x3384 | i2x_buf_sz | 0x030ed004 |
| 0x3388 | **magic** | **0x55504e5f** | | 0x338c | msi_id | 0x00014000 |

**Correction to the alive/idle drill above:** the magic is *pre-initialized data*
in the image (not L32r-referenced by any code, not S32i-stored in reachable
boot), so "magic never stored" was the wrong alive-signal to watch. The real
receive-ready signal is the fw writing **FW_ALIVE_OFF** (the flag the host polls)
and entering the command loop -- the `waiti 0` at `0x56e6` inside
`sched_event_poll` (0x5524; task 0x10f10's run-fn 0x588c lives inside it). Boot
never reaches that `waiti`. **The final gate = whatever transition takes boot
from the INTLEVEL-2 bring-up scheduler into the go-alive/command-loop path.**

**Open reconciliations (deferred):** (1) the struct's i2x addresses (`0x030ec000`)
vs the emulator's chosen i2x regs (`0x27200170` in `host_mailbox.rs`) -- device-
address vs BAR-offset translation, or a wrong chosen address; the wiring doc's
"x2i offsets are firmware-defined, DISCOVER not choose" applies. (2) whether RX
is fw-polled or mailbox-IRQ. Both settle once the go-alive routine is found.

**Next pull:** find FW_ALIVE_OFF and the routine that writes it (the "declare
alive + publish channel info" code), then why boot's INTLEVEL-2 scheduler never
calls it. That routine is the receive-ready transition.

## GO-ALIVE ROUTINE FOUND: it's a task boot creates but never dispatches

Resolved the host-side contract from the xdna-driver, then found the fw code
statically and proved its reachability dynamically. This reframes the whole wall
saga: **the task we spent it forcing (`0x10f10`) is not the one that publishes
alive.**

**Host contract (NPU1/Phoenix, `npu1_regs.c` + `aie2_pci.c:202-282`).**
`FW_ALIVE_OFF` = `MPNPU_SRAM_I2X_MAILBOX_15` = device SRAM `0x030BF000`.
Go-alive = fw writes the channel-info struct into SRAM, then writes that struct's
address (non-zero) into `FW_ALIVE_OFF`. Host polls `FW_ALIVE_OFF` (gapped ~20ms,
`readx_poll_timeout`), reads the 16-word struct at `*addr`, validates
`magic==0x55504e5f`, then clears `FW_ALIVE_OFF`. SRAM device base is `0x03000000`
(so `FW_ALIVE_OFF = base+0xBF000`, the struct's own i2x fields `0x030ec000` =
`base+0xEC000` -- **reconciliation (1) resolved: those are device-SRAM addresses,
NOT the emulator's guessed `0x27200170` MMIO reg**).

**The publisher (static, `m2c_probe_literal_xref` over device-SRAM range +
`m2c_probe_disasm_range`).** `FUN_000050d4` (entry `0x50e8`) builds the 48-byte
`mgmt_mbox_chann_info` field-by-field, loading the i2x ring addresses
(`0x030ec000/4`, `0x030ed000/4`) as literals and storing the magic literal
(`@0x332c` = `0x55504e5f`) into `[buf+0x20]` (the `magic` field offset). It is
called from **`0x56b1`**, inside routine `0x5524`, ~53 bytes before the idle
`waiti 0` at `0x56e6`. Sequence is straight-line: publish (`0x56b1`) -> idle
(`0x56e6`), guarded only by a trivial `BeqzN` skip.

**It's a task, not the one we forced (static call/lit xref).** `0x5524`'s only
direct caller is its own internal loop (`0x5773`). The publish+idle path is
entered via run-fn **`0x55f8`** (a mid-`0x5524` entry that flows down through
publish->waiti). `0x55f8` is the run-fn of a task **created at `0x3de9`**
(`Call8 0xd664`, args run-fn=`0x55f8`, `a11`=4, col-sentinel=`0xff`) -- a
DIFFERENT task from worker `0x10f10`, whose run-fn `0x588c` is a short routine
that stores a few bytes and `RetwN`s (never reaches the publish). We spent the
wall saga force-completing `0x10f10`; that was never going to publish alive.

**Proven never-dispatched (`m2c_probe_goalive_lifecycle`, new).** With the
flag+column breach applied at `0xd828` (so boot clears both walls), over the full
3M-instruction budget:

| PC | go-alive chain step | first-hit |
|----|---------------------|-----------|
| `0x3de9` | task-create(run-fn=`0x55f8`) | **n=47335 (x1)** |
| `0x55f8` | go-alive run-fn entry | **NEVER** |
| `0x56b1` | call publisher `0x50e8` | NEVER |
| `0x50e8` | publisher body | NEVER |
| `0x56e6` | `waiti 0` (idle) | NEVER |

The go-alive task IS created during boot init (just before the `0x10f10` breach
at n=47896) but its run-fn is **never entered**. So the final gate is precisely:
**what readies/dispatches the `0x55f8` task?** -- not anything about `0x10f10`.

**Next pull (self-contained).** Find the `0x55f8` task's handle (the create at
`0x3de9`/`0xd664` returns a status bool in `a2`, not the handle -- the handle is
stored elsewhere; locate it by scanning task structs for run-fn field ==
`0x55f8`, or trace `0xd664`'s side effects). Then read its await-mask
`[handle+0x38]` and state: is it un-wakeable-needs-direct-write (like `0x10f10`,
mask 0 -> we can breach it too and reach `waiti` THIS way), or does it await a
specific event/flag that boot should raise? That answers boot-to-idle.

## Scheduler model mapped + the go-alive task's record is LOST

Maya's call: map the full scheduler before intervening. Progress:

**Task creation is two distinct paths.**
- `task_init` (`0x4570`) registers the kernel task family via
  `0xdb28(0x581c, 0x5858, 0x588c, 8, 7)` into the TCB at **SCHED2 `0x1186c`**
  (dump: run-fns `0x581c`/`0x5858`/`0x588c` at `+0x1c/+0x20/+0x24`, args `8`/`7`
  at `+4/+8`, state `6` at `+0x28`). Both dispatched tasks `0x9040` and `0x10f10`
  share run-fn `0x588c` (a short returner) + scheduler helper `0xc938`.
- `task_create` (`0xd664`) registers the go-alive task (run-fn `0x55f8`, prio/idx
  `a3=4`, col `a4=0xff`), a SEPARATE path.

**Scheduler globals (resolved via literal peeks).** SCHED=`0x2250`,
SCHED2=`0x1186c`, per-index state table=`0xf308`, ready-mask base=`0x11098`.
The scheduler picks tasks by popcount over a 32-bit ready-mask (`FUN_0000c928`)
+ a task-table scan reading ready-bits at `[entry+4]`/`[entry+8]`
(`FUN_00007bf0`), dispatching via `[sched+36]`.

**`task_create` SUCCEEDS (traced n=47336-47405).** The "table full" early-exit
(`0xd67c`, returns 2) is NOT taken: task count `[0x24c4]`=0 < 15, so it runs the
create path, writes the record (run-fn `0x55f8`->`[0x2320]`=SCHED+0xd0, col
`0xff`->`[0x2324]`, counters at SCHED+512=`0x2450`, count `[0x24c4]` 0->1) and
returns **1 (success)**. So the go-alive task is genuinely registered.

**But the record does not survive.** Post-breach (3M) `[0x2320]` holds a
**dispatcher stack frame** (ret `0x7fe7`, run-fn `0x588c`), not `0x55f8`; and a
scan of all low RAM (`0x1000..0x30000`, where every scheduler struct lives)
finds `0x55f8` **nowhere**. At create SP=`0x12180` (high), but the post-breach
dispatch loop runs on a stack at `0x22a0..0x2350` -- **overlapping SCHED and the
go-alive record at `0x2320`**. So the go-alive task's run-fn is written into a
region the scheduler's own dispatch stack later overwrites.

**The reframed gate (open fork).** Is this (A) a faithful firmware transient --
the record at SCHED+0xd0 is a staging slot meant to be copied into a persistent
task array we haven't located (the persistent copy stores an index/pointer, not
`0x55f8` by value, which is why the scan misses it) -- or (B) corruption: the
dispatch stack at `0x22xx` overlaps the SCHED task-table, plausibly an artifact
of our *artificial* breach (forcing `0x10f10` drops SP into the low `0x22xx`
region vs the `0x12180` seen at create). Either way it reinforces "forcing
`0x10f10` is the wrong lever." **Next:** dump `[0x2320]` right after create
(n~47410) vs at the natural-boot wedge (58k, no breach) to see whether the
record is clobbered by normal flow or only under the breach; and find the
persistent task array the `SCHED+512` counters index.

## Probes used

`m2c_probe_addr_store_watch` (`XDNA_FW_WATCH_ADDR=0x22bc,0x10f40`),
`m2c_probe_store_search` (`XDNA_FW_STORE_DISP=0x6c`),
`m2c_probe_col_cmd_trace`, `m2c_probe_disasm_range`,
`m2c_probe_colassign_boot` (tail histogram), `m2c_probe_breach_doorbell`
(+bit3 shim), `m2c_probe_literal_xref` (`XDNA_FW_LIT_LO/HI` over device-SRAM +
code-pointer ranges), `m2c_probe_call_xref` (`XDNA_FW_XREF`, publisher/run-fn
caller chains), `m2c_probe_disasm_range`, `m2c_probe_goalive_lifecycle` (new:
flag+column breach + first-hit of the go-alive chain), `m2c_probe_mailbox_receive`
(canonical `boot_to_idle`).
