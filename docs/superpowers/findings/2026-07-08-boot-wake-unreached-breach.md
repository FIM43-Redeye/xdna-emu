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

## RESOLVED: the breach corrupts the record; natural boot preserves it

Three-way dump of `[0x2320]` (probe `m2c_probe_goalive_lifecycle`, new
`XDNA_FW_NOBREACH=1`):

| run | `[0x2320..0x232c]` | verdict |
|-----|--------------------|---------|
| natural, just after create (n=47600) | `55f8 00ff 04000000 0` | **intact** |
| natural, at wedge (n=60000) | `55f8 00ff 04000000 060122` | **intact** |
| breached, 3M | `80007fe7 2350 060722 588c` | **clobbered (stack frame)** |

**Determination: (B) corruption -- and it is a BREACH ARTIFACT.** The go-alive
record survives natural boot (through the 58k wedge) and is destroyed only under
the forced-completion breach, which drives the dispatcher stack up over the SCHED
task table (`0x22a0..0x2350`). So:
- `SCHED+0xd0` (`0x2320`) IS the real persistent go-alive record -- there is no
  hidden array to hunt. The earlier "`0x55f8` nowhere in low RAM" scan was on the
  already-corrupted *breached* run.
- The whole "force `0x10f10`" lever is not just wrong, it actively **destroys the
  task it's meant to let run**. Every breach-based result past the 58k wall is
  suspect.

**But even in natural boot the go-alive run-fn `0x55f8` is never dispatched** --
boot wedges at 58k (`0x10f10` parks) before the scheduler reaches it. So the
gate is unchanged in KIND but corrected in APPROACH: **`0x10f10` must complete
NATURALLY** (the real external-agent/array completion, determination B of the
original wall), which both (a) avoids the stack corruption and (b) lets the
scheduler advance to the intact go-alive record. Note the go-alive record's own
col is `0xff` (unassigned) -- the same sentinel as Wall C -- so its readiness may
also depend on a column assignment.

**Pivot:** stop breaching. Pursue the natural completion event for `0x10f10`
(array/SMU/mailbox), which is the real receive-ready path.

## CORRECTION + sharpening (2026-07-08, post-compaction): livelock, not crash; the contract is SYNCHRONOUS per-column

Fresh natural-boot probes (no forcing) settle the shape of the wall and correct
the "breach corrupts the record" pivot above.

- **Boot is a LIVELOCK, not a crash.** `m2c_probe_boot_with_array` runs the full
  700k budget with `stop=budget reached`, `last_pc` in `sched_ready_popcount` --
  and **zero array accesses** (STUB and ATTACHED identical). The firmware never
  touches the AIE array during boot, so the array is *not* the completion agent.
- **The steady-state loop (trace_to_wall tail).** `FUN_00007fa0` -> `FUN_00008c68`
  (polls 4 per-column pages `0x2727_1000/_2000/_3000/_4000`, byte at `+0x114`,
  **bit 3**; all clear) -> `task_dispatcher` (`0xd7f0`, `rsil 2`) -> done-check
  `0xd828` reads `[current-task 0x9040 + 0x30]` == 0 -> loop. **INTLEVEL held at 2
  the whole time**, so the level-1 completion IRQ (INTENABLE=0x1) can *never* be
  delivered here (`m2c_probe_inject_interrupt`: min intlevel 2, handler `0x2958`
  never fires). Current task is `0x9040`, not `0x10f10` -- both share run-fn
  `0x588c` and both are column-power workers.
- **The breach-corruption pivot was imprecise.** `m2c_probe_external_complete`
  delivers the completion the faithful way -- an external write of `[task+0x30]=1`
  **and** `[task+8]=col` ONCE, at the idle loop, for `{0x9040,0x10f10}`. Result:
  the go-alive record `[0x2320]` is **byte-identical before and after**
  (`000055f8 000000ff 04000000 00060122`). So completing the task does NOT corrupt
  `0x2320`; the earlier corruption was the *repeated* force at `0xd828` over 3M
  instructions growing the dispatch stack into the SCHED table. Faithful delivery
  preserves the record.
- **But one completion does NOT advance boot** (min intlevel stays 2; publisher
  `0x50e8` / go-alive `0x55f8` / waiti `0x56e6` never reached). Cause: the
  done-path (`deliver_pending_events`) **consumes and clears** `[task+0x30]`, so a
  single flag is eaten and the worker re-waits. That is exactly why force-*every*-
  cycle brute-forced past the wall (and grew the stack), while one real completion
  cannot.

**The corrected crux.** The completion contract is **synchronous and per-column**
(matches Session-4 of the completion-causality RE): to retire the column-power
worker, the poll state (`0x2727_(col)000` bit0/1 + local `0xf9e0+col*0x60` bit3)
AND the task flag must be satisfied *together, per column*, so `0x8c68` records
each column powered and the worker finishes -- after which the scheduler advances
to the intact go-alive record `0x2320`. Neither the poll alone (Session-3 negative)
nor the task flag alone (this session) advances boot.

**Fork for the next step:**
- **(B) Model the contract** -- a minimal faithful per-column SMU/PSP completion
  agent: when the firmware posts the colmask-`0xf` descriptor, synchronously set
  the poll bits + task flag per column so the worker retires naturally. This is
  the emergent-timing dream's actual seam. (Requires the alias-correct write path
  the RE flagged for `0x2727_n000`.)
- **(A) Chase the divergence** -- the firmware's own DMA-HAL bring-up
  (`FUN_0000b5d4` ops table; disp-0x6c setters `0x34ef8`/`0x37218` in the dormant
  `_XAieMl_Dma*` region) may be what generates these completions internally; our
  interp never installs/calls it. Pure interp fix, no modeling -- but a longer
  hunt, and prior sessions leaned away from it.

## VALIDATED (2026-07-08): the synchronous poll+flag contract advances boot, non-corruptingly

Per Maya's "validate the contract, then model" call, `m2c_probe_bit3_shim` was
extended with an env-gated task-flag delivery (`XDNA_FW_SHIM_TASKS`,
`XDNA_FW_SHIM_COL`) so the two halves can be satisfied *together*. Three-way
result (warmup 200k, 400k steps, mode=once):

| what is satisfied | outcome |
|-------------------|---------|
| poll only (bit3 `0xf9e0+k*0x60` + `0x2727_(col)000` bit0/1) | column SERVICED (bit3 cleared) but boot does NOT advance -- back to the poll spin |
| task flag only (`m2c_probe_external_complete`) | flag consumed once, worker re-waits -- no advance |
| **poll + flag together** (tasks `0x9040,0x10f10`, col 0) | **ADVANCES**: `deliver_pending_events` runs, boot leaves the poll/done-check spin and enters NEW scheduler code (`FUN_00007c38` + `sched_task_scan`) |

- **Non-corrupting.** `[0x2320]` is byte-identical before and after
  (`000055f8 000000ff 04000000 00060122`) -- the go-alive record survives. This
  closes the pivot: completion does not corrupt; only *repeated* forcing at
  `0xd828` did.
- **A next gate (gate 3) appears.** After the workers retire, boot spins in
  `sched_task_scan` (entry `0x7c10`), which loops over the task table reading each
  entry's `[entry+8]` (**column**) and `[entry+4]` and building readiness masks.
  So the scheduler now gates on per-task **column assignment** -- and both workers
  were given col 0, while the go-alive record carries col `0xff`. The next lever
  is very likely the *right* column value(s), not another flag. Reconnects to the
  "col=0xff sentinel = readiness depends on a real column assignment" thread.

**Where this leaves the model.** The completion contract is confirmed and
tractable: per-column {poll bits + task flag + column id}, delivered synchronously.
Modeling it = a small external agent that supplies exactly this when the firmware
posts the colmask-`0xf` descriptor. Open sub-question before/with modeling: what
column id each worker's `[task+8]` must hold to satisfy `sched_task_scan` (gate 3).

### Gate 3 characterized: it is a scheduler ready-mask, NOT the worker column

Column sweep (`XDNA_FW_SHIM_COL` in {0,1,2,3,0xff} on both workers): **no effect**
-- every value lands in the identical `sched_task_scan` spin (~26.6k iters). So the
worker `[task+8]` is not gate 3's lever. Instrumenting the scan (capture entry base
`a4` + `[a4+4]`/`[a4+8]` at `sched_task_scan+0x32`) shows it evaluates a **single**
base = `0x2250` = **SCHED itself**, reading `[SCHED+4]=0x00000000` (ready mask) vs
`[SCHED+8]=0xc0000000` (pending: **bits 30/31**). The scheduler is waiting for two
more slots to go ready and our shim never produces that.

**The meta-finding: shimming completions is whack-a-mole.** Each gate cleared by
short-circuiting a completion reveals a next gate the *real* firmware would have
satisfied as a side-effect of running the actual completion path. Gate 1 (worker
done-flag) -> gate 3 (SCHED ready bits 30/31) is the same shape one level up. This
argues the faithful fix is to make the firmware's OWN completion/ready-propagation
code run -- i.e. re-weight toward determination (A) divergence (why the interp
never reaches the code that readies these tasks), rather than shimming each gate.

## RETHINK (2026-07-08): the shim is structurally corrupting; adopt an external-agent principle

Chasing gate 3 (`sched_task_scan`) with the decode-correct ring
(`XDNA_FW_SHIM_RING`) showed the executed decode is CLEAN -- the static-disasm
`Unknown` ops at `0x7bfe..0x7c0d` are a literal pool, never executed. The real
anomaly is data: the scan runs with `a5=0x7c20` (a CODE address as its per-task
array base) and `a2` climbing unboundedly from `0x67f5`, sweeping a bogus range
forever. Root cause: **the completion shim pokes `[task+0x30]` (the one flag the
dispatcher polls) but skips the firmware's own completion code, which on silicon
also updates ready-masks / task-table entries / indices.** We advance the PC past
a gate while leaving scheduler DATA half-updated, so the next routine computes
garbage. Shimming is not just whack-a-mole -- it actively corrupts.

**Principle adopted (Maya):** supply ONLY the external stimulus the hardware
physically provides, at the point and in the form the firmware expects, then let
the firmware run ALL its own code. Never write firmware-internal state directly.
This is also the only version that serves the emergent-timing dream.

### Causality-respecting observation: the firmware DOES make external requests

`m2c_probe_external_requests` (pokes nothing; logs firmware stores whose EA lands
in an external aperture `0x27xxxxxx` / device SRAM `0x03xxxxxx`) over 1.5M
natural-boot instructions found **39 distinct external store sites** -- the
firmware's own bring-up IS running and programming hardware:
- **`0x27200xxx` control/DMA aperture.** `FUN_0000d4a0` writes `0x27200170..190`
  with `0x08a00ff0` / `0x08b041bc` (pointers into firmware Segment-B data,
  `0x08b00000` base) + sizes -- **DMA descriptor programming**. `FUN_0000893c`
  writes the `0x27200300..3bc` block **repeatedly** (x22/x32) starting right at
  the wall (`first@41559..44657`) -- a write/poll/retry loop.
- **`0x27270000 <- 0x1b6`** (`FUN_00008ad4`, `first@2188`): per-column HW page 0
  initialized early (the poll reads pages `0x27271000..0x27274000`).
- **`0x27010d6c`** (`task_init` `FUN_00004570`): event/interrupt config.

**Conclusion.** The completion IS externally-mediated and the firmware's own HW
code runs -- so the external-agent principle is viable and the contract surface is
the `0x27200xxx` control/DMA aperture (plus the `0x2727n000` per-column reads the
poll gates on). Our sysstub answers all these reads with 0, so the firmware's
programming never "completes." NEXT (faithful): find the write->read request/
response pair that gates the wall -- prime suspect the `0x27200300` block written
x32 at the wall, and the `0x2727n000` poll reads -- and model that responder in
the bus, letting the firmware consume it through its own code.

## DECISIVE CORRECTION (2026-07-08, post-compaction): there is NO external polled gate; the wall is a masked completion INTERRUPT

Set out to find the write->read request/response pair to model an external
responder. Full-boot observation (`m2c_probe_external_conversation`, new: logs
every external load AND store in temporal order, pokes nothing) falsifies the
premise -- **the pair does not exist in natural boot.**

- **The `0x27200300` block (`FUN_0000893c`) is one-directional config
  programming, not a handshake.** It is a bitmask walk: `0x272003b4` gets
  1,2,4,...,0x80000000 while `0x27200304` OR-accumulates to `0x0fc06000`;
  `0x272003bc`->`0x2720030c` fills to `0xffffffff`. Every "read" in the block is
  a read-modify-write of a register the firmware itself just wrote. **Nothing
  branches on a hardware response.** These are the DMA-descriptor / event-enable
  writes (Segment-B pointers `0x08a00ff0`/`0x08b041bc` at `0x27200170..190`).
- **After n~47500 the boot makes ZERO external accesses for the remaining
  ~1.45M instructions.** Purely internal from there.
- **Both gates in the livelock are INTERNAL.** The steady-state poll
  `FUN_00008c68` (`0x8c88`, hit 11115x over 1.5M) reads the internal RAM struct
  `[0xf9e0+col*0x60]` bit3 -- NOT the external `0x2727n114` page. (First-hit
  n=47866: base `a8=0xf9e0`, an internal address. The `0x2727n114` base that the
  earlier finding logged appears only in a rarer iteration of the same function;
  the "0x2727n000 poll is the steady-state gate" claim above was that mislabel.)
  The dispatcher done-check reads the internal flag `[task+0x30]`. The boot rests
  in `sched_ready_popcount` (`0xc964`, hit 118577x) -- the `rsil 2` INTLEVEL-2
  critical section (`FUN_0000c96c`).
- **This matches the codebase's own prior conclusion** (Session-3, comment at
  `mod.rs` `m2c_probe_faithful_smu_boot`): the poll path was "DEFINITIVELY
  falsified as the gate"; the completion is "the bit the (unmodeled)
  interrupt/ISR would have produced" via `wake_tasks_by_event_mask`.

**Conclusion.** The faithful external stimulus is NOT a polled-register responder
in the bus -- it is a **completion INTERRUPT** the firmware's own ISR consumes to
set the internal flag, and that interrupt is **masked at INTLEVEL 2** in the
livelock. This retires the "model the responder in the bus" plan. The fork
(Maya's call: characterize the IRQ seam first):
- **(A) divergence** -- the firmware's own code should lower INTLEVEL / reach the
  flag-setter and our interp diverges (wrong branch, missing side-effect, or
  never restoring INTLEVEL). Pure interp fix, no modeling.
- **(B) model the IRQ** -- raise the completion interrupt (level > 2, so
  unmasked) after the firmware programs the op, let the firmware's own ISR run.
- Deciding step: trace INTLEVEL across boot + find the completion IRQ's level.
  If level > 2 it is not masked by `rsil 2` -> (B). If level 1 and INTLEVEL never
  drops -> (A).

### IRQ-seam characterized: level-1 completion IRQ, permanently masked by rsil-2 busy-wait

`m2c_probe_intlevel_seam` (new; natural boot, array attached, pokes nothing)
over 1.5M instructions:

| metric | value |
|--------|-------|
| INTLEVEL histogram (whole boot) | 0: x125, 1: x2004, **2: x1497871** |
| INTLEVEL post-wall (n>=48000) | **2: x1452000 (ONLY)** -- never drops below 2 |
| INTLEVEL==0 | only n=0..2173 (125x), never after early boot |
| `interrupt_deliverable()` true | **0 times** (needs INTLEVEL==0) |
| final INTENABLE | **0x00000001** (only line 0 = a level-1 IRQ) |
| final INTERRUPT (pending) | 0x0 (nothing sources it -- no HW model) |

**The interp models level-1 delivery only** (`interrupt_deliverable` requires
`intlevel()==0`, `mod.rs:475`). The firmware enables exactly one IRQ (line 0,
level 1) and then holds INTLEVEL=2 for the entire post-wall boot. So a level-1
completion IRQ is **permanently masked** by the scheduler's own `rsil 2`
busy-wait (`task_dispatcher` 0xd7f3, re-entered every loop). Circular: the flag
needs the IRQ; the IRQ needs INTLEVEL 0; INTLEVEL only drops at the go-alive
`waiti 0` (0x56e6), which boot never reaches because the flag isn't set.

**Refined fork (this is NOT "pend the IRQ and it fires"):**
- **(A) interp divergence** -- silicon's dispatcher would restore PS / drop to a
  lower-level `waiti` to receive line 0, and our interp instead busy-polls at
  INTLEVEL 2 forever (a PS/critical-section handling divergence). Pure interp
  fix.
- **(B'') memory-writeback completion** -- the completion is NOT an interrupt at
  all: the DMA/HW engine writes the completion flag (`[0xf9e0+col*0x60]` bit3, or
  `[task+0x30]`) directly into firmware RAM at an address the firmware programmed
  as a writeback target. Then the "external agent" is a DMA completion-writeback,
  and no INTLEVEL change is needed.
- Deciding step: trace the dispatcher's PS/INTLEVEL management -- does it ever
  execute a path that restores INTLEVEL below 2 (and why is it never reached),
  and did the firmware's DMA programming (`FUN_0000d4a0`/`FUN_0000893c`) hand a
  RAM writeback address that points at `0xf9e0`/the task struct?

### FORK RESOLVED -> (B'') memory-writeback completion: the INTLEVEL-2 hold is BY DESIGN

`m2c_probe_intlevel_seam` extended to log every INTLEVEL transition. Over the
full 1.5M boot there are only **9 transitions**; the last is **n=2218**, after
which INTLEVEL is 2 forever:

```
n=  11  0x0001d8  0->1  Rsil imm:1
n=2011  0x00e02c  1->0  Wsr PS         FUN_0000e01c+0x10
n=2120  0x008901  0->2  Rsil imm:2     FUN_00008884+0x7d
n=2160  0x008993  2->0  Wsr PS         FUN_0000893c+0x57
n=2162  0x008999  0->2  Rsil imm:2     FUN_0000893c+0x5d
n=2171  0x0089b2  2->0  Wsr PS         FUN_0000893c+0x76
n=2173  0x0089b8  0->2  Rsil imm:2     FUN_0000893c+0x7c   <- ambient scheduler level
n=2214  0x0088cc  2->1  Rsil imm:1     FUN_00008884+0x48   <- brief critical section
n=2218  0x0088d8  1->2  Wsr PS         FUN_00008884+0x54   <- restores to 2
```

**The firmware's OWN code deliberately raises and holds INTLEVEL 2** for the
whole scheduler bring-up phase; every critical section dips to 1 and restores to
2. The interp reflects this faithfully -- so **(A) "interp wrongly stays at 2" is
ruled out**. And INTENABLE=0x1 (only the level-1 line 0 enabled) rules out a
high-level (>2) interrupt. The only completion mechanism that works at a pinned
INTLEVEL 2 is a **memory writeback**: hardware writes the completion flag
(`[0xf9e0+col*0x60]` bit3, which `FUN_00008c68` polls, and/or `[task+0x30]`,
which the dispatcher polls) into firmware RAM, and the scheduler **busy-polls**
it. No interrupt is involved in the wall.

**Determination: (B'') HW memory-writeback completion.** This is the external
agent in its truest form -- the firmware hands the HW (DMA/array/SMU) a RAM
writeback target during its bring-up programming, and the HW sets the completion
word there when the operation finishes. **Next: find the exact writeback target
address** (what sets `[0xf9e0]` bit3 / `[task+0x30]` on silicon) -- trace the
firmware's DMA/HW programming (`FUN_0000d4a0` descriptors, `FUN_0000893c`) for a
RAM writeback/completion-address field pointing at the `0xf9e0` struct or the
task struct -- then model that HW writeback as the external stimulus and let the
firmware's own busy-poll consume it.

### Writeback target confirmed never-set; run-fn trace is the next pull

`m2c_probe_poll_watch` on the four per-column struct bytes
(`0xf9e0`/`0xfa40`/`0xfaa0`/`0xfb00`, stride `0x60`) over 1.5M: **zero changes**.
The completion flag `[0xf9e0+col*0x60]` bit3 is never written in reachable boot
-- the only firmware writer (`FUN_00008c68`'s RMW `S8i [a8]`) sits on the
bit3-SET branch (never taken) and *clears* the bit. So the writeback is
genuinely external/missing, as (B'') predicts.

`FUN_00008c68` (executed trace authoritative; static disasm is overlay-
misaligned): the loop polls internal `[0xf9e0+col*0x60]` byte bit3; only when
bit3 is SET does it touch the external `0x2727n000` (bit0/bit1) + `0x2727n114`
pages with `Memw` fences (the ack handshake). So `[0xf9e0]` bit3 is the PRIMARY
trigger; the `0x2727` MMIO is the secondary ack. Open: does the HW write
`[0xf9e0]` bit3 directly (mgmt-RAM writeback), or does the worker run-fn
(`0x588c`, `Callx8`'d each dispatch) read an external status (`0x2727n000`) and
propagate it to `[0xf9e0]`? **Next pull: trace the worker run-fn `0x588c`** --
what it reads and the condition under which it would set the per-column flag --
to pin the exact external stimulus to model (the former fork's option 3).

### Worker run-fn `0x588c` traced: it SUBMITS a shared-memory column-power command (no doorbell)

`m2c_probe_runfn_trace` (new; executed/overlay-aware, captures a full pass of the
`Callx8`'d run-fn). The worker run-fn is NOT a no-op -- it builds and submits the
column-power command:

- `0x588c` stores per-tile bytes, then `Call8 FUN_00008620` -> `Call8 FUN_0000c530`.
- **`FUN_0000c530` builds the descriptor at `0xfae0`**: `[0xfae0]=1`, `[0xfae4]=1`,
  `[0xfae8]=0xf` (**colmask = all 4 columns**), `[0xfaf0]=0x9040` (**target task**),
  rest 0. `Memw`-fences, then `Callx8 0xb0e710`.
- **`0xb0e710` is a `Dhwbi` (dcache writeback-invalidate) loop + `Dsync`** -- it
  **flushes the descriptor to shared/physical memory** so an external agent can
  read it. This is the RE's long-known `0xfae0` colmask-0xf descriptor.
- Then `FUN_0000c530` falls straight into `Call8 FUN_00007fa0` -> `FUN_00008c68`
  (polls `[0xf9e0]`/`[0xfa40]`/`[0xfaa0]` bit3 -- all 0) -> `task_dispatcher`
  (polls `[0x9070]` = `[task 0x9040+0x30]` -- 0).

**There is NO external MMIO doorbell in the submission path** (no `0x27xxxxxx`
write between build and poll). The command is handed off purely by **cache-
flushing the `0xfae0` descriptor to shared memory**; the SMU/PSP is expected to
poll that shared region, execute the column power-up, and write the completion
**back into mgmt RAM** -- the per-column flag `[0xf9e0+col*0x60]` bit3 and/or the
task done-flag `[0x9070]`. That is the textbook (B'') memory-writeback contract,
and it is a genuine HW-writeback target (the descriptor is deliberately flushed
FOR an external reader), NOT firmware-internal state we would be poking.

**The model (to confirm with Maya before building).** An SMU/PSP agent in the
firmware bus that: (1) triggers on the `0xfae0` descriptor flush (the `Dhwbi`/
`Dsync` of a valid colmask descriptor), (2) reads the colmask + target task from
the flushed descriptor, (3) writes the completion back into mgmt RAM for each
column in the mask. **Open sub-question:** the earlier VALIDATED experiment
showed setting `[0xf9e0]` bit3 ALONE services the column (FUN_00008c68 clears
bit3, does the `0x2727n114` ack) but does NOT advance boot -- only bit3 + the
task flag `[0x9070]` together advance. So either the SMU writeback sets BOTH, or
the firmware's own bit3->task-flag propagation (via its event system /
`wake_tasks_by_event_mask`) has a further gap. Resolve by finding what the
firmware does with a SET `[0xf9e0]` bit3 (does servicing it raise the event that
sets `[0x9070]`?) -- that decides whether the faithful writeback is bit3-only
(pure) or bit3+task-flag.

### SUB-QUESTION RESOLVED: the completion must write the task flag DIRECTLY (await-mask 0)

Two tests settle the writeback semantics:

- **Poll-bits do NOT propagate** (`m2c_probe_force_event`, bits=0xb = bit0|bit1|
  bit3, seeded at n=60000 past the wall entry). Even satisfying BOTH the
  `[0xf9e0+col*0x60]` bit3 poll AND the `0x2727n000` bit0/bit1 checks, the
  firmware services the column (active path hit 6591x with reseed) but
  `[0x9070]` (task 0x9040's flag) stays 0 and boot never advances.
- **The task is un-wakeable by events** (`m2c_probe_event_propagation`, new). At
  n=60000 task 0x9040's await-mask `[0x9078] = 0`. Seeding the event-pending
  bitmask `[0x22bc]` = 0xffffffff (one-shot or reseed) does NOT set `[0x9070]`
  and boot stays in `FUN_0000c96c`. `wake_tasks_by_event_mask` skips any task
  with `delivered & await-mask == 0`, so no event can wake it. (Same as 0x10f10;
  both column-power workers have await-mask 0. Note `[0x22bc]` held `0x588c` at
  warmup, not a clean event bitmask -- but the await-mask=0 fact is decisive
  independent of that.)

**Determination.** The completion MUST set the task done-flag `[task+0x30]`
(`[0x9070]`) **directly** -- neither the poll bits nor the event system reaches
it. And this REFRAMES the earlier "direct write is corrupting" worry: these
column-power tasks carry await-mask 0 *precisely because* they are designed to be
completed by a **direct external write** (the SMU/PSP), not by an event. The
earlier corruption came from REPEATED forcing at `0xd828`; a faithful ONE-SHOT
write (the VALIDATED "poll+flag together, once") advances non-corruptingly. So
the direct task-flag write IS the faithful seam.

### FULLY-SPECIFIED MODEL (ready to build, pending Maya's go-ahead)

An SMU/PSP completion agent in the firmware bus:
1. **Trigger:** the worker flushes the `0xfae0` command descriptor
   (`Dhwbi`/`Dsync` of a valid descriptor: `[0xfae8]` = colmask nonzero,
   `[0xfaf0]` = target task).
2. **Read** colmask (`[0xfae8]`) and target task (`[0xfaf0]`) from the flushed
   descriptor.
3. **Write completion back into mgmt RAM** (the genuine HW-writeback):
   - `[0xf9e0+col*0x60]` bit3 = 1 for each column in the colmask (the per-column
     status FUN_00008c68 polls), and
   - `[target_task+0x30]` = 1 once (the done-flag the dispatcher polls;
     `[0x9040+0x30]` = `[0x9070]`).
   Then let the firmware's own dispatcher retire the worker and advance to the
   go-alive record. Non-corrupting per the VALIDATED one-shot result.

Open build questions: exact trigger detection (watch the `0xfae0` write +
`Dsync`, vs a dedicated submit register), per-command idempotency (fire once per
distinct descriptor, not every poll), and whether the `0xf9e0` bit3 write is
even needed or the task flag alone suffices (test at build time).

## BUILT & VALIDATED (2026-07-08): the agent breaks the task_dispatcher wall

The `ColumnPowerAgent` is implemented in `src/firmware/host_mailbox.rs` and wired
into the existing per-step `HostMailbox::tick` seam (enabled by
`enable_host_mailbox`). It reuses that scaffold rather than a new Dsync hook: the
tick already runs once per instruction in `boot_to_idle`, which models the SMU as
an independent concurrent poller. Integration gate:
`m2c_boot_completion_advances_past_recursion` (un-`ignore`d; was parked for
exactly "the faithful per-task model"). 5 new unit tests + the boot gate; full
suite 4064 pass.

**Trigger + target (as specified):** poll `[0xfae0]` valid; on a valid descriptor
read colmask `[0xfae8]` + target `[0xfaf0]`; write bit3 into `[0xf9e0+col*0x60]`
per masked column AND `[target+0x30]=1`. Both writes are kept (the VALIDATED
"poll+flag together" combination); dropping bit3 was not needed to advance, so it
stays for fidelity.

**Idempotency resolved -- the completion is LEVEL-based, not one-shot.** The first
build used a content-keyed one-shot latch. It failed: the agent writes `[0x9070]=1`
at n~47809 (when the descriptor is flushed), but task 0x9040 only becomes
*current* at n~58775 and its dispatch setup ZEROES `[task+0x30]` before polling it
(observed: `[0x9070]` 1->0 at n~58929). The standing descriptor never changed
content, so the one-shot refused to re-fire and boot stayed livelocked. Fix: the
agent re-asserts bit3 + the done-flag on EVERY tick the descriptor stands valid
(real SMU status is "column powered", a held level), and re-arms only when the
firmware tears the descriptor down (`[0xfae0]` valid -> 0, its own completion ack).
This is faithful, NOT the old corrupting done-check forcing: the firmware runs all
its own retire code; the descriptor tear-down is the handshake that stops the
re-assert.

**Result -- primary wall broken.** With the agent enabled, boot leaves the
multi-session `task_dispatcher` (0xd7f0) livelock: final pc moves 0xd80b ->
`FUN_00007c38` (0x7c40), the descriptor cycles valid 1->0->1 (worker retires and
re-dispatches, 4 completions in 2M), `[0x9070]` holds at 1, and the go-alive task
record `[0x2320]` is BUILT with the correct run-fn pointer 0x55f8 (not corrupted).

**NEXT WALL (sched_task_scan retire gate).** Boot does not yet reach the go-alive
chain (0x50e8 / 0x56e6). Worker 0x9040 re-runs through `sched_task_scan` (0x7bf0)
in a larger loop instead of PERMANENTLY retiring -- `current-task` stays 0x9040
and the descriptor keeps recycling. Completing the inner poll lets the dispatcher
proceed, but the worker is not removed from the scheduler ready-set, so it is
re-picked. This is the old finding's "gate 3 = a SCHED ready-mask" observation,
now reached faithfully (record intact, unlike the corrupting poke). Resolving it
is the next RE step: what marks the worker complete at the scheduler level once
its column-power command is acknowledged (a second writeback target, or a firmware
path that clears the ready-bit that our completion isn't yet satisfying).

## RETIRE-GATE CHARACTERIZED (2026-07-08, faithful): the go-alive waker never fires

`m2c_probe_retire_gate` (new; agent enabled, no forcing, 2M) pins the gate
precisely and reconnects it to the 2026-07-06 completion-causality RE.

**It is INTERNAL, not an external poll.** The post-completion tail rests entirely
in `sched_task_scan`/`FUN_00007c38` hammering `[0x2254]`/`[0x2258]` (128690 reads
each); external `0x2727n000` reads are negligible (4 hits). So the "firmware polls
the per-column HW page and stalls on our zero sysstub" hypothesis is FALSIFIED.

**The scheduler is waiting on a slot mask that never goes ready.**
`[SCHED+8]=[0x2258]` (pending slots) is set to `0xc0000000` (bits 30/31) once at
n=59411 and never changes; `[SCHED+4]=[0x2254]` (ready slots) stays 0 the whole
run (ZERO stores to either word after the initial set). The dispatcher hardcodes
the same `0xc0000000` mask (`0xd805 Slli -1,30`) -- bits 30/31 are two special
scheduler slots. The `a3=0x9040` trip count / `a5=0x7c20` (code) base the RE saw
under the corrupting shim reappear here faithfully; they are a large safety bound
that DEGENERATES to a 36928-iter sweep precisely because ready stays 0 -- the
missing readiness is the cause, not the count.

**Causal chain (executed trace, `a0=return`):** dispatcher `0xd7f0` reads
current-task `[SCHED+40]=0x9040`, sees done-flag `[0x9070]=1` (our agent) at
`0xd828`, so it takes the delivery path: `0xd82c Call8 0xcadc`
(`deliver_pending_events`) -> returns to `0xd82f` -> `S8i 6,[task+0x2c]` (state) ->
`0xd836 Call8 0xc938` (scheduler helper) -> the `0x7c20` slot scan. Loops, finds
nothing ready, returns, re-dispatches the worker.

**The decisive count: `deliver_pending_events` (0xcadc) runs 4x but
`wake_tasks_by_event_mask` (0xd84c) runs ZERO times.** Per the 2026-07-06 RE, the
go-alive task's slot is readied ONLY by `wake_tasks_by_event_mask`, driven by an
event mask that `sched_event_poll` (0x5524) derives from the HW event source
`0x27010d28`. Our completion delivers event bit 0 (from `[0x9040+0x30]=1`), but it
matches no waiter's `[entry+0x38]` wait-mask, so `deliver_pending_events` clears
the bit and returns WITHOUT waking anyone. The only state=6 "mark ready" stores in
the run are the dispatcher re-marking the two column-power workers (`[0x10f3c]`,
`[0x2099]`) -- never the go-alive task, never through the event waker.

**Determination: this IS the 2026-07-06 RE's open "event source never sampled"
knot, now REACHED FAITHFULLY (record intact) instead of via the corrupting
force-done.** The retire gate is not a second writeback target -- it is that the
firmware's own event-driven readiness path (`sched_event_poll` 0x5524 reads
`0x27010d28` -> mask -> `wake_tasks_by_event_mask` readies slots 30/31) never runs:
`sched_event_poll`'s run-fn is never dispatched (its pointer is nowhere in
`[SCHED2+36]`), so the early-HW event that readies the go-alive slots is never
produced.

**The fork (Maya's call before building):**
- **(A) faithful/deep.** Model the HW event source `0x27010d28` (return a
  non-sentinel event = "columns powered", which we already signal via bit3) AND
  resolve why `sched_event_poll` (0x5524) is never dispatched, so the firmware's
  own waker readies the go-alive slots. Most faithful to the emergent-timing
  dream; larger, and re-opens the "why is the poll run-fn never registered"
  sub-question.
- **(B) targeted event delivery.** Find the go-alive waiter's `[entry+0x38]`
  wait-mask (one of the 9 entries at `[SCHED+56]=[0x2288]`) and have the agent set
  the pending-event bit(s) in `[SCHED+108]=[0x22bc]` / the waiter's `[task+0x30]`
  that it matches, so the already-running `deliver_pending_events` wakes it.
  Smaller, but risks the RE's "shimming is whack-a-mole" trap (injecting the
  translated event skips the firmware's own columns-powered -> event translation).
- **Cheap decisive next step (settles A vs B):** dump the 9 waiter entries at
  `[SCHED+56]=[0x2288]` (each `[+0x38]` wait-mask, `[+0x2c]` state) and trace what
  the go-alive waiter waits on -- that names the exact event bit, and whether it is
  producible by delivery (B) or requires the poll (A).

**Sharpening (same probe, waiter-table dump): the waiter table is EMPTY.**
`[SCHED+56]=[0x2288] = 0` -- the 9-entry waiter-table base the event-waker walks is
null, and the `0x2288..0x2320` region holds register-window spill / task save-area
data (return addrs `0x80007fe7`/`0x8000878d`, saved SPs `0x2320`/`0x2350`, the
dispatcher's own `a6=0xc0000000`/`a7=0x3fffffff`). So the scheduler stack/save
areas abut the SCHED table and the waiter table is unusable. The event-driven
readiness path is therefore STRUCTURALLY unavailable, not merely un-triggered:
(1) delivery gets an unmatched bit 0, (2) the waker is never called, (3) the
waiter table is empty, (4) the event-poll run-fn is never dispatched. This tilts
the determination toward (A) DIVERGENCE -- the firmware's own event/scheduler flow
is degenerate here -- rather than a missing external writeback. The go-alive
record at `0x2320` is nonetheless intact (0x55f8), so this is NOT the old
force-done stack corruption; it is the scheduler's normal-but-diverged state under
faithful completion.

## ROOT CAUSE (2026-07-08, Maya: "chase divergence, scoped"): the scheduler stack overlaps the SCHED table

Chasing "why is `sched_event_poll` never dispatched" collapsed the whole
event-system framing. Robust post-step change detection on the mask words
(`m2c_probe_retire_gate`, extended) shows the retire gate is a **stack-vs-data
memory-overlap divergence**, not anything about events or writebacks.

**`[0x2258]=0xc0000000` is spill garbage, not a pending mask.** The one write to
it happens at n=59411 by **pc=0x000895 -- an `S32e` register-window OVERFLOW
SPILL**, storing the dispatcher's `a6` mask register (`0xc0000000`) onto SCHED+8.
So "the scheduler waits for slots 30/31" was a RED HERRING; the scan even reads
the LOW byte of `[0x2258]` (=0), so it never depended on that value. `[0x2254]`
(ready) is genuinely never written by anything.

**The causal chain (SP trace).** `sched_ready_popcount` (`0xc938`, entered from
the dispatcher's `0xd836 Call8`) switches SP from the high stack `0x121d0` to a
LOW task stack `~0x30d0` (n=47930) and never resets it again. From there SP
**leaks monotonically downward** under the repeated worker dispatch -- the deep
scheduler call chain (dispatcher -> `0xc938` -> `FUN_00007fa0` -> `FUN_00008c68` /
`FUN_0000c530` -> the `0x7c20` scan) is never fully unwound, so window-overflow
spills (`0x880` handler) march SP through `0x23e0 -> 0x2250 (SCHED) -> 0x2200`,
clobbering the SCHED table on the way down. The go-alive record at `0x2320`
survives only because the descent stops above it.

**Consequence.** The `0x7c20` slot scan then runs on a CORRUPTED SCHED: its base
`a5=0x7c20` comes from `[0x11868]=SCHED2-4` (spill-clobbered), its trip count
`a3=0x9040` is garbage, the ready byte is 0 -- so it never finds a ready task,
`current-task` never leaves `0x9040`, and the go-alive task (run-fn `0x55f8`,
inside the same big `0x5524` function) is never picked. `wake_tasks_by_event_mask`
never runs and the waiter table `[SCHED+56]` is null because that region IS the
clobbered stack. Every downstream symptom is the one stack overlap.

**The divergence is either:** (a) the task stack is MIS-BASED at `~0x30d0` (only
~3.7 KB above SCHED at 0x2250) when it should live in the high region (a
context-switch / task-creation SP bug), or (b) the worker's repeated dispatch is a
NON-RETURNING recursion that should unwind back to the high-stack scheduler loop
each cycle (a control-flow divergence) -- SP only ever decreases, never restores,
which favors (b). Deciding step: check whether, on a correct cooperate-yield, the
worker run-fn `Retw`s back to the `0x121d0` scheduler loop between dispatches (SP
restored) vs. re-enters deeper; and confirm where the `~0x30d0` task-stack SP is
first established (task create vs. context switch) and whether that base is
faithful. This is a pure interp/firmware-control-flow question -- NO external
modeling -- consistent with the emergent-timing dream.

### DECIDING STEP (2026-07-08): it is (a) a stack that overlaps SCHED, base firmware-chosen

Two corrections to the intermediate read, then the answer.

**Correction 1 -- not a leak.** Tail (n>=60000) SP up/down step counts are ~balanced
(up=119, down=131) and there are only ~250 changes across 1.9M tail instructions
(SP is constant during the 36928-iter scan spin, oscillating only across the 4
dispatches). Dispatches DO return. So **(b) non-returning recursion is FALSIFIED**.
Most of the earlier "SP switches" (0x800/0x880) were window under/overflow handler
artifacts (read_ar(1) reads a rotated window), NOT real stack switches.

**Correction 2 -- the real switch, and the base is firmware-chosen.** The one
genuine high->low switch is in the context-switch routine `FUN_00002730` (~0x2a44):
```
0x2a44 Addi a2, a1, -16      ; save slot on the OLD high stack (a1=0x12120)
0x2a47 L32r a1, [pool]       ; a1 <- 0x3170  -- SP LOADED FROM A FIRMWARE LITERAL
0x2a4a..0x2a58               ; copy 4 words old-save-area -> [0x3170]  (frame restore)
0x2a60 Wsr PS ...            ; context-switch epilogue
```
So the `0x3170` stack base is a firmware constant, NOT an interp miscompute.

**Determination: (a).** The scheduler context runs on the firmware's `0x3170`
stack, which sits only ~3.8 KB above the SCHED table (0x2250) and its waiter table
(`[SCHED+56]=0x2288`). The dispatch call chain (worker run-fn 0x588c -> descriptor
build/flush -> `FUN_00007fa0` -> `FUN_00008c68` poll -> ... , many nested Call8s)
descends far enough that **window-overflow spills (`0x880` handler, `S32e`)
overwrite the waiter table with garbage** (confirmed: the one write to SCHED+8 at
n=59411 is exactly such a spill; the waiter table dump reads null/stack data). A
corrupt waiter table -> `sched_ready_popcount` (0xc938) walks null entries -> no
ready mask -> no task ready -> the go-alive run-fn 0x55f8 is never picked. Every
symptom is downstream of the stack descending into the SCHED table.

**The uncovered next question.** Since `0x3170` is firmware-faithful and SCHED at
`0x2250` is firmware-faithful, on silicon this call chain must NOT descend ~3.8 KB
into SCHED. So either (a1) our emulation drives an ABNORMALLY DEEP call chain
(e.g. our completion keeps the worker in the deep `FUN_00007fa0`/`FUN_00008c68`
poll path that silicon exits early once the column is truly powered), or (a2) some
frame allocates far more stack in our interp than on silicon, or (a3) the SCHED
base / stack base literal is version-mismatched to this firmware image. Deciding
step: measure the true call-nesting depth of ONE dispatch (count Entry-without-Retw
minus the window-handler noise) and the largest single-frame allocation on the
descent; compare against the 3.8 KB budget. If the chain is only reached because
the worker re-polls a column silicon would have finished, the fix is on the
completion side (make the column "powered" fully, so the worker exits the deep
poll shallow) -- which loops back to fidelity of the `FUN_00008c68` ack, not the
event system.

### PRIME SUSPECT TESTED (2026-07-08): stack corruption is a SYMPTOM; go-alive readiness is NOT the only gate

Maya: "push on the prime suspect." Three measurements settle it.

1. **The bit3 re-service "fight" is falsified.** `FUN_00008c68` entries = 0 in the
   tail (it is not busy-polling); `[0xf9e0]` bit3 was SET once by the agent and
   CLEARED zero times by firmware. The level-based bit3 re-assertion is inert here,
   not driving a re-service loop.
2. **The corruption is a recursion symptom, not an independent bug.** Whole-boot
   `Entry`=629 vs `Retw`=497 -> **net +132 frames never returned**. The one-cycle
   call structure shows why: the worker run-fn `0x588c` -> `FUN_0000c530`
   (build+flush, returns) -> `FUN_00007fa0` -> `FUN_00008c68` (poll, **returns
   cleanly**) -> then `FUN_00007fa0` **CALLs `task_dispatcher`** (0xd7f0), which
   `Callx8 [SCHED2+36]=0x588c` re-dispatches the SAME worker -- a cooperative-yield
   recursion. This is real firmware code (runs on silicon too); it terminates on
   silicon within a few cycles when a task becomes ready and the scheduler reaches
   the idle `waiti`. On EMU nothing becomes ready, so it recurses unbounded (+132)
   until the stack descends into SCHED and corrupts it. So the poll is NOT stuck on
   column fidelity; the stack overlap is downstream of "no task ever becomes ready".
3. **Distance-to-finish: go-alive readiness is NOT the only gate.** With
   `XDNA_FW_FORCE_GOALIVE=1` (point `[SCHED2+36]` at the go-alive run-fn `0x55f8`),
   boot DOES enter `0x55f8` (n=616224) but STILL never reaches publish `0x50e8` or
   `waiti 0x56e6` -- it rests at `0x7fec`. Even dispatched, the go-alive run-fn hits
   its own internal precondition and does not publish. (The force is crude -- wrong
   task context -- so not conclusive on how close go-alive is, but it definitively
   kills "flip one pointer and boot comes alive".)

**Conclusion.** The prime suspect (a quick completion-side fidelity fix) does NOT
hold. The worker's poll returns fine; the deep-stack corruption is a symptom of the
worker being re-dispatched forever because no task becomes ready; and even
force-dispatching go-alive does not publish. Reaching boot-to-idle requires
faithfully reproducing the firmware's cooperative readiness/event flow (the (A)
path: what readies the go-alive/event-poll task and satisfies the go-alive run-fn's
own precondition), not a one-line completion tweak. The completion agent remains a
genuine milestone (dispatcher wall broken); the remaining distance is the
scheduler event system, now precisely bounded and no longer a mystery.

Probe: `m2c_probe_retire_gate` (faithful boot; tail PC/read split ext-vs-int,
SCHED-mask transitions, `sched_task_scan`/dispatcher/`wake` fresh-entry register
bursts, delivery/wake entry counts, state=6 stores). Ignored unless `XDNA_FW_PROBE`.

## REFRAME (2026-07-08, post-compaction): the event system is a RED HERRING; go-alive is never LINKED into the schedule

Two static disasms (go-alive routine `0x5524..0x56ec`, wakers `0xcadc`/`0xd84c`)
plus a clean pre-corruption waiter-table dump (`m2c_probe_waiter_table`, new)
overturn the "model the event source" framing from the prior section.

**The event/wake machinery is real but UNUSED for these tasks.** `deliver_pending_events`
(0xcadc) and `wake_tasks_by_event_mask` (0xd84c) both walk the same 9-pointer array
at `SCHED+56=0x2288`, matching each entry's wait-mask `[+0x38]` against an event
mask and setting state `[+0x2c]=6` on a hit. In clean natural boot (agent enabled,
n<58000) the array holds **exactly 3 registered entries** (slots 6/7/8 =
`0x10dfc`/`0x10e58`/`0x10eb4`, the column-power worker family, states 0/1/6) and
**every one has wait-mask `[+0x38]=0`** with callback `[+0x24]=0`. Nothing waits on
an event; `deliver` runs 4x and matches nobody because there is nothing to match.
So the HW event source `0x27010d28` / `sched_event_poll` dispatch question does not
gate go-alive.

**The go-alive record is created but never linked into the schedule.**
`FUN_0000d6c0` (task_create) writes the go-alive record to the FIXED slot
`SCHED+208=0x2320` at n=47360 (runfn `0x55f8`, col `0xff`, prio 4, state
`[0x232c]=0`). The runnable/waiter array at `0x2288` is populated by a SEPARATE
function `FUN_0000d53c` (a ~200-byte struct initializer whose tail store
`[a3+56] <- ptr` at pc `0xd60f` links the entry), called only ~n=39852..40152 for
the 3 workers -- i.e. BEFORE go-alive is created. **No slot ever points at
`0x2320`.** So go-alive is registered-as-a-record but never made runnable; the
"link into schedule" step (a start/resume/enqueue call after task_create) is never
reached.

**The `[0x232c]=0x060122` write I chased last session is CORRUPTION, not a state
transition.** It is `FUN_0000c530+0x6` (a stack store at pc `0xc536`) landing in
SCHED at n=58106 once the recursion has descended the stack into the table; the
same store hits `[0x229c]<-0x060922` (n=58519), and `deliver` writes the code
address `0x588c` into pending `[0x22bc]` (n=59499) -- all post-58k spill garbage.
Clean state: pending `[0x22bc]=0` throughout; go-alive `[0x232c]=0` through n=55000.

**Current-task is worker `0x10f10` from n<48000 through n=59000**, only becoming
`0x9040` after corruption. So the boot sits on a column-power worker for ~10k
instructions after go-alive is created, never advancing to link/start it, then the
unbounded worker recursion corrupts SCHED. The gate is therefore in the
**task-lifecycle / boot-init sequencing** (what makes go-alive runnable, and why
boot stays on the worker instead of proceeding to that step), NOT in the event
system. Two threads:
- **(T1) lifecycle:** find the "start/make-runnable" call that should follow
  task_create for go-alive and why boot never reaches it (pure firmware
  control-flow; aligns with emergent dream).
- **(T2) completion sufficiency at array scale:** the boot stays on `0x10f10`
  because the column-power workers aren't all reaching done -- check whether the
  `ColumnPowerAgent` completes ALL columns (5 for Phoenix), not just the flag the
  dispatcher polls; the 3 registered workers' states (0/1/6) at n=48000 differ, so
  at least one is not done. Boot-init likely gates go-alive-start on all-columns-done.

T1 and T2 are connected: the most likely silicon story is a boot-init loop
"dispatch a power worker per column, wait all done, THEN start the publish task."
If completion is incomplete, boot stalls on the worker and never links go-alive.

Probe: `m2c_probe_waiter_table` (natural boot; chronological stores into the
`0x2288` array / pending `0x22bc` / go-alive record `0x2320`, first-non-null per
slot, checkpoint dumps with waiter derefs at n=48k/55k/59k/end). Ignored unless
`XDNA_FW_PROBE`.

## T2 RESULT (2026-07-08, Maya: "completion at array scale"): the completion sets a done-flag but NOT the scheduler state; the completion-target task is never made ready

Two probes (`m2c_probe_worker_wait`, `m2c_probe_state_machine`) + three static
disasms (`sched_ready_popcount` 0xc938, build/flush `FUN_0000c530` 0xc530,
`task_dispatcher` 0xd7f0) pin the park precisely.

**The scheduler dispatches on the STATE BYTE, not the done-flag.**
`sched_ready_popcount` (0xc938) loops the 6-slot array at `SCHED+56`, counting a
task ready iff `[task+0x2c] == 1` (`Bnei a6,1,skip`). The dispatcher's completion
path (`FUN_0000d828` from 0xd7f0) reads the CURRENT task's done-flag `[cur+0x30]`;
whether set or not it then writes `[cur+0x2c] = 6` ("serviced") and reschedules.
So `state 1 = ready-to-dispatch`, `state 6 = serviced/running`. Empirically the
initial states are set ONCE at registration (`FUN_0000d4a0+0x76`, n≈39.7k):
`0x10dfc→0`, `0x10e58→1`, `0x10eb4→6`, and never change.

**The park (measured, `m2c_probe_worker_wait` / `m2c_probe_state_machine`).**
Current-task is worker `0x10f10` from n=41464 straight through to the n≈59k
corruption. At n=47809 its run-fn flushes ONE column-power descriptor
(`FUN_0000c530`): `valid=1, colmask=0xf, target=0x9040`. The `ColumnPowerAgent`
completes it faithfully -- bit3 for cols 0..3 (`0xf9e0+col*0x60`) and done-flag
`[0x9040+0x30]=1`. From n=47920 the dispatcher then re-marks `0x10f10.state=6`
every ~413 instructions (a periodic scheduler tick) and re-dispatches it; the
parked window's PC histogram is dominated by `sched_ready_popcount` /
`FUN_0000c96c` (the worker spins in the scheduler, NOT in the column bit3 poll --
`FUN_00008c68` barely runs). `0x10f10`'s own done-flag `[0x10f40]` is never set.

**The gap (decisive).** The completion's target is `0x9040`, but **`0x9040.state`
is NEVER written -- it stays 0 the whole boot, and `0x9040` is not even in the
runnable array** (the array holds `0x10dfc/e58/eb4`). So the completion sets
`0x9040`'s done-flag + bit3 but drives NO scheduler transition: nothing makes
`0x9040` `state=1` / dispatchable. The running worker `0x10f10` is waiting for the
completion handler to ready `0x9040`; that never happens, so `0x10f10` never
yields, the scheduler never advances, and eventually the unbounded re-dispatch
recursion corrupts SCHED (~58k). This is why the prior "the agent broke the
dispatcher wall" milestone was necessary but not sufficient: it satisfied the
done-flag the dispatcher polls, but the *scheduler* keys on the state byte, and
the completion->ready transition is unmodeled.

**Convergence with the earlier IRQ finding.** Every scheduler routine here runs at
`Rsil imm:2` (INTLEVEL 2): `sched_ready_popcount` (0xc93b), `deliver_pending_events`
(0xcadf), `wake_tasks_by_event_mask` (0xd84f), `task_dispatcher` (0xd7f3),
`FUN_0000c530` (0xc533). On silicon the column-power completion is a level-1 IRQ
whose handler runs `deliver`/`wake` to set the target task `state=1`; the firmware
spends the park at INTLEVEL 2 so the memory writeback alone can't substitute --
the STATE TRANSITION the handler performs is the missing piece. This is the same
"masked completion interrupt" seam from the DECISIVE CORRECTION section above, now
localized to a concrete effect: **the completion must drive the target task
(`0x9040`) to scheduler-ready (`state[+0x2c]=1`, and present in the runnable set),
the way the firmware's own completion-IRQ handler would.**

**Next (the fix-shape fork, Maya's call).** (i) FAITHFUL: drive the firmware's own
completion path -- inject the level-1 completion IRQ (or synchronously invoke the
handler that calls `deliver`/`wake`) so the firmware itself sets `0x9040` ready;
most faithful, re-opens "what readies 0x9040 into the array / what event bit". (ii)
SURGICAL MODEL: extend the agent to also write the target task's `state[+0x2c]=1`
(and enqueue it) on completion -- smaller, tests the model immediately (does the
scheduler then dispatch `0x9040` and advance boot?), but risks skipping the
firmware's own translation. Cheap decisive experiment before either: with the
agent standing, poke `0x9040.state=1` (and add it to the array if popcount still
misses it) and observe whether current-task advances off `0x10f10` and boot
reaches the go-alive link/publish.

Probes: `m2c_probe_worker_wait` (current-task transitions vs descriptor
flush/colmask/target, per-column bit3 first-set, worker done-flags, parked-window
PC histogram), `m2c_probe_state_machine` (all `[task+0x2c]`/`[+0x30]` writes with
pc+value, checkpoint state bytes). Ignored unless `XDNA_FW_PROBE`.

### RECONCILIATION: the faithful "IRQ handler" path is BLOCKED by the INTLEVEL-2-by-design masking; the real seam is the writeback->ready translation

Cross-checking T2 against the earlier "IRQ-seam characterized" / "FORK RESOLVED
-> (B'')" sections resolves an apparent contradiction. Injecting the level-1
completion IRQ cannot work as the completion channel: INTLEVEL is pinned at 2 by
the firmware's OWN code (9 transitions total, last at n=2218), `interrupt_deliverable()`
(needs INTLEVEL 0) is true 0 times post-wall, INTENABLE=0x1 (only the level-1
line). The ONLY `waiti` that drops INTLEVEL is inside the go-alive task (0x56e6),
which never runs -- so the IRQ is circular (IRQ needs the idle waiti; the idle
waiti needs go-alive; go-alive needs the completion). The prior session already
retired the IRQ mechanism for exactly this reason and pivoted to memory-writeback.

So the faithful completion at a pinned INTLEVEL 2 MUST be a **memory writeback the
firmware's own busy-poll consumes** -- which is what the `ColumnPowerAgent` does.
T2's contribution is the precise insufficiency: the writeback sets the done-flag
`[+0x30]` and bit3, but the scheduler advances on the READY-BIT the selection scan
reads (`sched_task_scan` 0x7c10 reads `[entry+4]/[entry+8]` + a status byte;
`sched_ready_popcount` reads `state[+0x2c]==1`). **Nothing translates "column done"
into "target task ready-bit set."** The exact next question -- for either a
divergence fix or a model extension -- is: **what code path is supposed to set the
ready-bit / `state=1` for the completion target (`0x9040`) once bit3/done-flag are
set, and why is it not reached?** The worker run-fn `0x588c` returns cleanly each
dispatch (it just zeroes status bytes), so the worker is not failing to yield --
the dispatcher re-selects it because the completion never flips a ready-bit the
selection scan would pick up. Tracing `sched_task_scan` (0x7c10) + `FUN_00007c38`
(the ready-bit writer at `S8i [a8],a9` 0x7c54, and its `Call8 0xafec`) against the
completion signal is the pull that names the translation.

### DIVERGENCE HUNT step 1 (2026-07-08): `0x9040` is NOT a task; the scheduler is a bigger RE than a quick lever

`m2c_probe_task9040_wiring` (new) forks the hunt and shows the translation is not a
one-line fix:
- **`0x9040` is not a task struct.** At clean n=50000 its whole 64-byte struct is
  zero except the done-flag `[0x9040+0x30]=1` (our agent). No run-fn, no state
  byte, no fields. So "make `0x9040` a ready task" is ill-posed -- it is a bare
  completion-target *address* the HW writeback lands on, not a schedulable entity.
- **`FUN_0000c530` is a GENERIC descriptor builder the dispatcher calls every
  ~413-instruction cycle** (caller `a0=0xd845` = the dispatcher's own
  `Callx8 [sched+36]` return; args `target=0x245a0, colmask=0`), distinct from the
  ONE-shot column-power flush (`colmask=0xf, target=0x9040`, [0xfaf0]) at n=47809.
  The per-cycle dispatch itself flushes a descriptor, so the "worker issues one
  column-power command and waits" picture is too simple.
- **The `sched_task_scan` pool literals I read (`[0x349c]`/`[0x3498]`) are null** --
  either the wrong literals for the table bases or an empty selection class in this
  phase; the active selection structure is not yet pinned.
- The ~32 `0x9040` values in `[0x28a8,0x3148]` sit in the SCHED/stack region
  (stack base 0x3170, descending) -- consistent with register spill of a
  0x9040-holding register, not a task table.

**Assessment.** T2 precisely characterized the GAP (completion sets done-flag/bit3
but not the scheduler-ready transition) and ruled out the easy levers (IRQ masked
by design; current-task done-flag insufficient -- the dispatcher marks state 6 and
reschedules on both branches; `deliver` matches no 0-mask waiter). But pinning the
exact faithful translation requires MAPPING THE SCHEDULER: the two dispatch
structures (current-task at SCHED+40 vs the `[sched+36]` run-fn pointer that
resolves to `FUN_0000c530` per cycle), the real selection tables (not the null
`[0x349c]`/`[0x3498]`), and what sets a task's ready-bit / `state=1` from a
completion. That is a bounded but multi-session RE sub-project, not a quick
divergence fix -- flagged so the next session starts with the scheduler-mapping
framing, not another single-probe lever hunt.

Probe: `m2c_probe_task9040_wiring` (0x9040 struct dump, FUN_0000c530 caller+args
per flush, sched-table literal read, low-RAM search for the value 0x9040). Ignored
unless `XDNA_FW_PROBE`.

## SCHEDULER-MAPPING SESSION (2026-07-08, post-compact): re-derived + QUANTIFIED the stack-overlap root; an interp window-bug lead RAISED and FALSIFIED

Took the "map the scheduler" framing. Five new probes (`m2c_probe_ready_mask`,
`m2c_probe_current_task`, `m2c_probe_select_trace`, `m2c_probe_stack_range`,
`m2c_probe_recursion`, `m2c_probe_stack_leak`). Net: this session **confirmed and
sharpened the ROOT CAUSE section above (lines 849-975)** with cleaner numbers and
corrected two details -- it did NOT overturn it. Honest accounting of new vs.
re-derived:

**NEW / corrected (solid):**
- **The `sched_task_scan` literals are NOT null -- they resolve in FETCH space.**
  `[0x349c]` fetch=`0x11098` (ready-mask base), `[0x3498]` fetch=`0xf308` (state
  table); both read `0x0` via `data_load32` (Harvard overlay: literals live in
  instruction space). This CLOSES the "null pool literals" open item from the
  DIVERGENCE HUNT section -- it was a data-vs-fetch artifact, not empty tables.
- **The ready-mask/scan scheduler tier is DORMANT in boot.** `sched_task_scan`
  entry `0x7c10`, its ready-bit write `S8i` `0x7c54`, and its notify helper
  `0xafec` all have **0 hits** across the whole boot (`m2c_probe_ready_mask`). The
  scan reads a ready-mask (`0x11098`, gate bytes +4/+8) + state table (`0xf308`)
  that are never driven. What DOES run is the popcount routine via its INTERIOR
  entry `0xc938` (the dispatcher's `0xd836 Call8`), not the dormant `0xc928` entry
  (0 exact hits, but the `0xc928` routine bucket is the hottest thing in the parked
  window: 3809). So the active selection uses the `0x2288` runnable array + state
  bytes (`FUN_0000c984` @ `0xc980` indexes `[SCHED+idx*4+56]`, checks `[task+0x2c]`),
  NOT the ready-mask scan.
- **Quantified the descent.** After the syscall context-switch to the firmware's
  `0x3170` supervisor stack (confirmed: `FUN_00002730`, `L32r a1,[pool]`), SP leaks
  **exactly 144 bytes (0x90) per dispatcher pass, dead constant** across 29 nested
  `0xd7f0` entries, `0x3110 -> 0x2150`. Window vectors over the boot:
  **114 OVERFLOW (`0x880`) : 0 UNDERFLOW** (`m2c_probe_stack_leak`).
- **`current-task` `[0x2278]` has only 2 transitions ever:** `->0x10f10` at init
  (`task_init` 0x4570, n~41k) and `->0x9040` at ~n=58.7k. The second is a
  **register-window SPILL artifact**, not a scheduler pick: `m2c_probe_select_trace`
  rings the transition and shows it happening inside the `0x880` overflow handler
  with `a6=0x9040` being spilled onto `[0x22a8]`/`[0x2278]`; the runnable array is
  already clobbered (slot8=`0x9040`, pending `[0x22bc]`=`0x588c` = a code ptr). So
  "the scheduler selects 0x9040" is FALSE -- 0x9040-as-current is spill garbage,
  matching the DIVERGENCE HUNT read that 0x9040 is not a task.

**Interp window-underflow bug: RAISED then FALSIFIED.** The 114:0 overflow:underflow
imbalance + the dead-constant 144 B/pass leak looked like a smoking gun for a
missing window-underflow refill in our Xtensa interp (`interp/control.rs` Retw path).
It is NOT: 114:0 is *exactly what genuine non-returning recursion produces* (you only
underflow when you RETURN far enough to refill a spilled frame -- recurse forever,
never underflow), and the **29 dispatcher entries at monotonically DECREASING SP**
prove each dispatch is nested inside the previous, not returning. The Retw underflow
path (control.rs:98-103, raises on `!frame_live(caller)`) is correct; there simply is
no return. This reconfirms the ROOT-CAUSE section's cooperative-yield recursion
(`FUN_00007fa0 -> task_dispatcher -> re-dispatch same worker 0x588c`), which is
FAITHFUL firmware control flow that terminates on silicon once a task becomes ready
and the scheduler reaches idle `waiti`. The 144 B/pass is the firmware's own
per-recursion-level frame cost, not an interp leak.

**Net conclusion (unchanged gate, sharper target).** The boot wall is NOT a quick
fix and NOT an interp bug. It is the same gate the ROOT-CAUSE and REFRAME sections
name: **no task is ever LINKED into the active runnable array (`0x2288`, SCHED+56)
with `state=1`**, so the cooperative-yield recursion never terminates and the
firmware stack descends into SCHED. The concrete, now-pinned RE question for the
next session: **what firmware path inserts a task (specifically the go-alive task,
record at `0x2320`, run-fn `0x55f8`) into the `0x2288` runnable array and sets its
`[task+0x2c]` state byte to 1?** The active selector is `FUN_0000c984` (`0xc980`)
reading `[SCHED+idx*4+56]`; find its producer -- the enqueue/link primitive -- not
another readiness lever. (The go-alive task IS created -- `task_create` 0xd664 writes
its record to `0x2320` -- but "never LINKED into the runnable set" per the REFRAME
section; the link primitive is the missing piece.)

Probes (all self-skip unless `XDNA_FW_PROBE`; run
`cargo test --lib firmware::boot_tests::<name> -- --nocapture --test-threads=1`):
`m2c_probe_ready_mask` (waypoint hits + runtime ready-mask/state-table bases +
fetch-vs-data literal peeks + region store watch), `m2c_probe_current_task`
(current-task writes + parked-window symbol histogram), `m2c_probe_select_trace`
(ring buffer on `[0x2278]` change; runnable-array dump at trigger),
`m2c_probe_stack_range` (SP trajectory + SCHED-band entry), `m2c_probe_recursion`
(ring + call/ret balance + call-target histogram on SP-descent trigger),
`m2c_probe_stack_leak` (SP at each dispatcher entry + window over/underflow vector
tallies).

## LINK-PRIMITIVE STRIKE (2026-07-08, Maya: "one focused strike, stop-gate"): go-alive is CREATED-BUT-NEVER-PROMOTED into the runnable array

Focused strike on the pinned gate (what links a task into `0x2288` with `state=1`,
and why go-alive misses it). ANSWERED, with a clean structural map:

- **The link primitive is `FUN@0xd4e0`** (real entry; `0xd4a0` is a different small
  fn). It fills the TCB, writes the state byte (`0xd516 S8i a12,[task+0x2c]`), and
  indexes SCHED (`0xd531 L32r a15,[0x3d28]=SCHED`; `0xd538 Addx4`); its tail
  (`0xd53c`, pc `0xd60f`) writes the runnable slot `[SCHED+idx*4+56]`. **Called only
  from `task_init` `FUN_00004570`** (4 sites: +0x19/+0x9f/+0xc4/+0xed) -- i.e. for the
  static KERNEL WORKERS.
- **go-alive takes a DIFFERENT path: `task_create` `0xd664`** (called from `0x3de9`).
  task_create is gated on create-count `[0x24c4]<15`, builds a record in the
  `SCHED+512` (`0x2450`) create-registry, and stores the run-fn/arg
  (`0xd6e6 S32i a2,[reg+208]`). Its only `Call8`s are `0xc530` (descriptor build),
  `0x2694`, `0xcadc` (deliver) -- **it never calls `0xd4e0`, never writes the `0x2288`
  runnable array, never sets state=1.**
- **Direct measurement (`m2c_probe_runnable_writes`, clean boot).** The runnable
  array ends with slots 6/7/8 = `0x10dfc`/`0x10e58`/`0x10eb4` (the kernel workers,
  linked at n=39.8-40.1k via `0xd60f`). go-alive: at n=47363 its record fills
  (`[0x2320]=0x55f8` run-fn, `[0x2324]=0xff` col), `[0x24c4]` count 0->1 -- **and the
  runnable array is never touched again.** go-alive is created-but-never-promoted.

**The gate, exactly.** The deferred-created go-alive task is parked in the `0x2450`
create-registry (count `[0x24c4]=1`) and the promoter that would move a created task
into the `0x2288` runnable array (via `0xd4e0` or equivalent) never runs in our boot.
That is the precise missing mechanism.

**Sharp secondary observation (possibly the deeper crux).** Runnable slot 7 =
`0x10e58` sits at **state=1 (ready) the ENTIRE boot yet is never dispatched** --
current-task stays `0x10f10` until corruption. So even a LINKED, READY worker in the
array is not being picked. This says the dispatcher's parked loop re-services
current-task (`0x10f10` -> run-fn `0x588c`) and never invokes the array picker
(`FUN_0000c984` @ `0xc980`, reads `[SCHED+idx*4+56]`, processes state!=0) to switch.
So "promote go-alive" may be necessary-but-not-sufficient; the picker not switching to
the ready slot-7 worker is a parallel gate.

**Stop-gate reached.** The strike ANSWERED its question (link primitive found;
go-alive misses it via the deferred-create path; promoter absent) but did NOT surface
a single flippable cause -- it is a genuine multi-mechanism promotion/dispatch gap,
now precisely mapped. Two concrete next questions for a future push:
1. What PROMOTES a `0x2450`-registry task into the `0x2288` runnable array (find the
   consumer of `[0x24c4]`/the registry), and why doesn't it run for go-alive?
2. Why does the dispatcher's parked loop never invoke the picker (`0xc980`) to switch
   to the ready slot-7 worker (`0x10e58`, state=1)?

Probes (self-skip unless `XDNA_FW_PROBE`): `m2c_probe_runnable_writes` (poll the 9
runnable slots + create-count + go-alive record across clean boot, PC per change);
plus the disasm/xref of `0xd4e0`/`0xd664` via `m2c_probe_disasm_range` /
`m2c_probe_call_xref`.

## PICKER-GATE STRIKE (2026-07-08, follow-on) -- question #2 ANSWERED

The prior section left two next-questions. Question #2 ("why does the parked loop
never invoke the picker `0xc980` to switch to the ready slot-7 worker") is now
**decisively answered, and it merges #1 and #2 into a single gap.**

**The picker `0xc980` (`FUN_0000c984`) is reached ZERO times across the entire boot**
(`m2c_probe_picker_gate`, clean boot n=0..58500: picker ENTRY total=0, DISPATCH
`0xc9b9` total=0). Task selection never runs, not once. So the slot-7 worker isn't
"skipped by a bailing picker" -- the picker is simply never called.

**Why: selection is architecturally decoupled from the dispatch loop.**
- The dispatch loop is `FUN_00007fa0` (`0x7fe4 Call8 0xd7f0` = task_dispatcher,
  bracketed by `0x8c6c` and the `0x26d4` context-switch). `task_dispatcher` `0xd7f0`
  reads current-task `[SCHED+40]`, gates on byte `[+0x1b]==1`, builds a descriptor
  (`0xc530`), and on the done-flag `[+0x30]` sets `[+0x2c]=6` and calls
  `sched_ready_popcount` `0xc938`. **It never iterates the runnable array and never
  calls the picker.** Every pass it merely *counts* ready tasks (`0xc938`, hot loop),
  never *selects* one. Counting is not selecting.
- The picker `0xc980` has exactly two direct callers -- `FUN_000041b8+0x110` (early
  init) and `FUN_0000dbc4+0x1b6`. `FUN_0000dbc4` is a **syscall/command dispatcher**
  (a chain of tail-call arms); the picker is ONE arm (`0xdd78 L32iN a10,[a2+8];
  0xdd7a Call8 0xc980`). So a reschedule only happens when the running task issues
  that specific yield/schedule syscall.

**Worker state-byte `[+0x2c]` timeline (`m2c_probe_picker_gate`, clean boot):**

| n | 0x10dfc (slot6) | 0x10e58 (slot7) | 0x10eb4 (slot8) | 0x10f10 (cur) | cur-task |
|------|------|------|------|------|----------|
| 40000 | 0x00 | **0x01** | 0x00 | 0x00 | 0x0 |
| 42000 | 0x00 | **0x01** | 0x06 | 0x00 | 0x10f10 |
| 48000 | 0x00 | **0x01** | 0x06 | 0x06 | 0x10f10 |
| 58000 | 0x00 | **0x01** | 0x06 | 0x06 | 0x10f10 |

Reading: current-task is set ONCE to `0x10f10` by `task_init` at n=41464 (NOT by the
picker). `0x10f10` runs, reaches state=6 (**done**) by n~48000, and is then **never
replaced** for ~10k steps until the recursion corrupts SCHED (~58.7k). Slot-7 worker
`0x10e58` sits at state=1 (**ready**) the entire boot, never selected. So the wall is:
**a cooperative scheduler whose running task never issues the reschedule syscall, so
no successor is ever picked -- even though a ready one is sitting in slot 7 and a done
one is sitting in the current slot.**

**The merged gap (single next crux).** #1 (go-alive never promoted) and #2 (picker
never called) are the same failure at two altitudes: nothing triggers a
reschedule/select. The remaining question for the next push is now singular and
sharp: **what is current-task `0x10f10` doing in the parked hot loop that prevents it
from ever issuing the yield/schedule syscall (the `FUN_0000dbc4` arm at `0xdd7a`)?**
Candidates: (a) it is busy-waiting on a completion/event that never arrives (ties to
the array-completion contract), so it never reaches its own yield point; (b) the
recursion (144 B/pass stack leak, syscall context-switch to `0x2730`/`0x26d4`)
re-enters before the yield point. Next probe: over the parked window, does
`0x10f10`'s run ever reach `FUN_0000dbc4` at all, and if so which arm selector -- i.e.
is it making syscalls-but-not-yield, or making no syscall at all (pure busy-wait)?

Probe added: `m2c_probe_picker_gate` (count picker `0xc980`/`0xc986`/`0xc9b9` hits +
index + `[+0x2c]` state across whole boot; periodic 4-worker state-byte timeline;
`XDNA_FW_WIN=lo:hi`, `XDNA_FW_MAX`).

## YIELD-SYSCALL STRIKE (2026-07-08, follow-on) -- the crux REFRAMED

Chasing the merged crux ("what keeps current-task `0x10f10` from ever issuing the
reschedule syscall?"), `m2c_probe_yield_syscall` (clean boot, parked window
[44000,58000)) returned two decisive facts that **overturn the "go-alive never runs"
framing from the two sections above.**

**1. ZERO `Syscall` instructions execute across the entire boot.** The syscall handler
`FUN_0000dbc4` (`0xdbc4`) is entered 0 times; the picker arm `0xdd7a` 0 times; the
context-switch stubs `0x2730`/`0x26d4` 0 times; and a decode-level count of `Op::Syscall`
is **0 total**. So the yield/reschedule-via-`Syscall` path is not "not taken by this
task" -- it is **never taken by any code in this firmware boot.** The scheduler here is
a **direct-call cooperative loop**, not a trap-driven one. (The earlier banked note of a
"syscall context-switch `FUN_00002730`" during the recursion refers to activity at/after
corruption >58k, outside this window -- the parked window has none.)

**2. `goalive_runfn` (`0x55f8`) IS executing every dispatch cycle** -- 474 steps in the
parked window. **Go-alive is NOT starved.** The "created-but-never-promoted / never runs"
conclusion was wrong: go-alive's run-fn runs on every loop; it simply **never completes.**

**The actual parked-window hot loop (routine histogram, top rows):**

| steps | routine | role |
|------:|---------|------|
| 3513 | `sched_ready_popcount` (0xc928) | counts ready tasks every pass |
| 1723 | `FUN_0000c96c` (0xc96c) | popcount helper (adjacent to picker) |
| 1716 | `FUN_0000893c` | (loop helper) |
| 1224 | `FUN_00008c68` | (loop helper) |
|  674 | `FUN_0000c530` (0xc530) | descriptor builder |
|  474 | **`goalive_runfn` (0x55f8)** | **go-alive's run-fn -- runs, loops, never finishes** |
|  300 | `task_dispatcher` (0xd7f0) | ~25 entries, each nesting 144 B deeper (the leak) |
|  274 | `FUN_00007fa0` | dispatch driver |

So the loop is: `FUN_00007fa0 -> task_dispatcher -> FUN_0000c530 -> ... -> goalive_runfn
-> sched_ready_popcount -> back`, re-entered ~25x with a 144 B/pass stack leak, no
syscall, no picker, no reschedule. Go-alive runs inside this and loops.

**`goalive_runfn` shape (disasm 0x55f8..0x56b0).** A multi-branch state machine whose
arms nearly all `J 0x555c` (the real dispatch/loop head is ~`0x555c`, below the symbol).
It calls a helper chain (`0x4300`, `0x531c`, `0x3f4c`, `0xe674`=`FUN_0000e674`, `0x9414`,
`0x4a00`, `0x5178`, `0x981c`, `0x97a8`) with `Beqz`-on-return early-exits back to the
loop head -- i.e. it polls a set of conditions each cycle and loops when they aren't
satisfied.

**Reframed crux (the real next question).** Boot-to-idle is NOT gated by a missing
promoter or a dormant picker (both true but downstream). It is gated by **`goalive_runfn`
polling a condition that never becomes true** -- it runs every cycle and re-loops. This
reconnects the wall to the **array/column-power completion contract** (the `goalive`
task is the alive/publish worker; its poll is almost certainly waiting on a completion
signal the `ColumnPowerAgent` model doesn't deliver in the exact form/place it reads).
Next push (bounded RE): trace `goalive_runfn`'s branch decisions over ~3 consecutive
cycles -- which `Beqz`-guarded helper is the one that keeps returning "not ready," and
what memory/device word that helper reads. That word is the missing completion.

Probe added: `m2c_probe_yield_syscall` (over the parked window: `Op::Syscall` count
bucketed by a2/a4; waypoint hits for `0xdbc4`/`0xdd7a`/`0xd7f0`/`0x2730`/`0x26d4`;
handler-entry register snapshots; routine histogram; `XDNA_FW_WIN`, `XDNA_FW_MAX`).

## POLL-LOAD STRIKE (2026-07-08, follow-on) -- the completion-TARGET MISMATCH

Tracing "what does the loop poll that never flips," `m2c_probe_poll_loads`
(histogram every non-stack DATA load over the parked window, addr -> count/last-val/PCs)
resolved the wall to a **completion-target mismatch**, and cleanly separated the
working half of the contract from the broken half.

**The agent's column bit3 IS correctly read (SRAM half works).** `FUN_00008c68`
(the per-column poll) reads `[0xf9e0+col*0x60]` as a byte and gets **`0x08` = bit3
set** (cols 0/1/2 at `0xf9e0`/`0xfa40`/`0xfaa0`, 50 reads each). The device-aperture
handshake in that function (`0x8c93 L32iN a9,[a5]; 0x8c95 Bbci a9,bit0,skip`; and the
`0x8ca0/0x8ca2` spin-until-bit1) is **bit0-guarded and safely SKIPPED** when the
aperture reads stub 0 -- so the `0x271000`/`0x272000` stub-0 reads are NOT a hard gate.

**The broken half: the done-flag target mismatch.** Poll-load counts over the window:

| addr | = | reader | value | meaning |
|------|---|--------|------:|---------|
| `[0x010f40]` | `0x10f10+0x30` | dispatcher `0xd828` (25x) | **`0x0`** | current-task done-flag -- what the dispatcher WAITS ON |
| `[0x00fae0]` | DESC_VALID | builder `0xc530` (25x) | `0x1` | column-power descriptor stands valid |
| `[0x009070]` | `0x9040+0x30` | -- | -- | **NEVER READ by any firmware code** |

The dispatcher retires the current task on `[current+0x30]` = `[0x10f10+0x30]`
(`0xd828 L32iN a10,[a4+48]`, `a4`=current=`0x10f10`). The `ColumnPowerAgent`
(`host_mailbox.rs`) completes `[descriptor.target+0x30]` where `target`=`[0xfaf0]`=
`0x9040` -- i.e. it sets `[0x9070]=1`. **But `[0x9070]` is never read by anything.**
So the agent's task-done-flag write is **inert**: the only done-flag the firmware
polls is the current task's `[0x10f40]`, and nothing ever sets it. `0x10f10` spins in
the dispatcher forever, stack leaks, SCHED corrupts.

**The agent's own rationale is built on the falsified spill artifact.** `host_mailbox.rs`
lines 117-126 justify the LEVEL re-assert by "the firmware makes the target `0x9040`
current and zeroes its flag at ~n=58929." But the picker-gate strike proved the
`->0x9040` current-task flip at ~n=58.7k is a **register-spill artifact during the
corruption window**, not a real schedule. `0x9040` never legitimately becomes current
(the picker never runs; `0x9040` isn't in the runnable array). So the "handshake" the
agent models never actually happens.

**Unification.** The three strikes are one wall: (picker never runs) => (no reschedule)
=> (`0x10f10` stays current) => (dispatcher waits on `[0x10f40]`) => (agent completes
`[0x9070]` instead, which nobody reads) => (spin -> stack overflow -> SCHED corruption).

**The fork (why this needs a decision, not a code reflex).** On silicon the descriptor
target IS `0x9040` (firmware-built at `0xfaf0`), so real SMU/PSP would complete
`0x9040` too -- yet silicon boots. So one of these is true and RE must decide which:
(a) our scheduler diverges earlier and the real current-task at flush time is NOT
`0x10f10`; (b) `0x9040`'s completion propagates via a mechanism we don't model
(a firmware path that reads `[0x9070]` under some condition, or an event/IRQ) rather
than the current-task done-flag; (c) the `[0x10f10+0x30]` wait is satisfied by
`0x10f10` finishing its OWN (non-column-power) work, and go-alive's stall is a separate
downstream step. **Forcing `[0x10f10+0x30]=1` from the agent would be UNFAITHFUL** --
silicon writes the descriptor's target, not the current task -- so that is the old
corrupting forcing, NOT the fix. The next step is RE: what does `0x10f10`'s run-fn
actually wait for, and how is `0x9040`'s completion meant to reach it.

Probe added: `m2c_probe_poll_loads` (non-stack DATA load histogram over the parked
window: addr -> count/last-value/issuing-PCs; `XDNA_FW_WIN`, `XDNA_FW_MAX`,
`XDNA_FW_TOPN`).

## STRUCT-DUMP STRIKE (2026-07-08, follow-on) -- hypothesis (c)->(a): the target is disjoint

Pulling hypothesis (c) ("does `goalive_runfn` loop forever on a poll?") then (a)
("did the agent complete the WRONG object?") with `m2c_probe_goalive_cycle` (full
instruction trace of the go-alive block) and `m2c_probe_desc_dump` (struct dump in
steady state) resolved the completion mechanism to bedrock.

**(c) answered: go-alive is NOT an infinite poll -- it's a one-time array-config
burst.** The trace shows the go-alive path runs once (~n=44000) through
`FUN_0000893c`/`FUN_00008910`/`FUN_000091a8`, scanning and bit-masking the array/NoC
apertures `[0x200314]`, `[0x200324]`, `[0x200400..0x200408]`, `[0x20040c]` -- **all
stub 0** -- and reading a dispatch-function pointer `[0x12158]=0x5a3c` (into the
`[0x581c,0x5d30)` low-text block). After that burst the go-alive block does NOT
re-execute; the steady state is purely the dispatcher re-running current-task
`0x10f10`. So go-alive submits its command once; the stall is downstream.

**The dispatcher re-runs `0x10f10` forever pending its own done-flag.** With current
= `0x10f10`: gate byte `[+0x1b]=0` (so `task_dispatcher` `0xd7f0` takes the `Bnei a5,1`
-> `0xd828` branch), done-flag `[+0x30]=0` (so `0xd828 BeqzN` skips retire), and it
falls to `0xd83b`: `Callx8 [[0x3d30]+36]` = re-invoke `0x588c` (the run-fn). It will
keep re-running `0x10f10` until `[0x10f40]` (its done-flag) or `[+0x1b]` becomes 1.

**(a) answered by the struct dump (steady state):**

| struct | key fields |
|--------|-----------|
| descriptor `[0xfae0]` | `+0=1` valid, `+8=0xf` colmask (4 cols), **`+0x10 (0xfaf0)=0x9040` target**, rest 0 |
| target `0x9040` | **every word 0 except `[0x9070] (+0x30)=1`** (the agent's own write) |
| current task `0x10f10` | `+0=0x12048`, `+4/+0x10/+0x14=0x121d0`, `+8=0xff`, `+0x2c=6` (state done), **`+0x30=0`** (done-flag NEVER set) |

**`0x9040` is a fully empty struct** -- not a task, no back-pointer to the submitter
`0x10f10`, nothing but the agent's inert `+0x30=1`. The descriptor's only task-pointer
is `[0xfaf0]=0x9040`. So the completion target (`0x9040`) and the dispatcher's wait
target (`0x10f10`) are **structurally DISJOINT**: completing `0x9040` can never set
`0x10f10`'s done-flag, even in principle, on EMU or HW. `[0x9070]` is never read; the
write is truly inert.

**`0x9040` is caller-supplied via a generic primitive.** The `[0xfaf0]<-0x9040` write
is `FUN_0000c530+0x1b` (`0xc54b S32iN a6,[a10+16]`), and `FUN_0000c530` is a GENERIC
6-word "write record + Memw-flush" helper with 19 callers (incl. `task_dispatcher+0x33`).
`0x9040` arrives in `a6` from whichever caller builds the column-power command -- it is
not a constant of the builder. Writes repeat every ~392 instrs (the rebuild cycle).

**Two concrete threads for the next push (RE, stop-gated).**
1. **Is `0x9040` MISCOMPUTED?** Trace the `a6`/target origin at the column-power
   caller of `0xc530` in the agent-enabled boot: if `0x9040` is `base + off` where
   `base` came from a stubbed-0 array/aperture read, the target is a divergence
   artifact and the real target should be a live task (plausibly `0x10f10`). `0x9040`
   is in the `0x9xxx` region -- unlike any known task slot (`0x10dfc..0x10f10`,
   go-alive `0x2320`, create-registry `0x2450`) -- which is consistent with a
   scratch/mis-based pointer.
2. **Is the completion signaled by IRQ, not memory?** Since the write to `0x9040` is
   structurally inert, HW's "SMU completed `0x9040`" may reach the firmware via the
   completion IRQ (known masked at INTLEVEL-2) rather than a memory poll -- reconnecting
   to the intlevel-seam finding. If so, the model owes an IRQ, not a done-flag write.

Probes added: `m2c_probe_goalive_cycle` (full instruction trace of the go-alive block
with load/call annotation; `XDNA_FW_LO`/`XDNA_FW_HI`/`XDNA_FW_TRACE_START`/`_CAP`) and
`m2c_probe_desc_dump` (word dump of descriptor/target/current-task/go-alive structs
with pointer annotation; `XDNA_FW_DUMP_N`).

## TARGET-ORIGIN STRIKE (2026-07-08, follow-on) -- thread 1 narrowed + recursion link

Chasing where `0x9040` is born (`m2c_probe_target_origin`).

**The write site.** `[0xfaf0]<-0x9040` is `FUN_0000c530+0x1b` (generic 6-word writer),
called from a SINGLE site: `ret=0x878d` (the `Call8 0xc530` at `0x878a`, inside the
block the disassembler labels `FUN_00008620`). Args: `a2=1` valid, `a3=0xf` colmask,
`a6(target)=0x9040`. So the column-power descriptor is built here every rebuild cycle
(~every 392 instrs).

**Full register file at the call (`0x878a`, when `a14`=target=`0x9040`):**
`a0=0x8000d845` (ret->`0xd845`), `a1(SP)=0x3160`, `a5=0x1eb00`, `a6=0x9268`,
`a7=0x245a0`, `a8=0x800058cc`, `a9=0x581c`, `a10=1`, `a11=0xf`, **`a14=0x9040`**, rest 0.

**Two structural results:**
1. **`0x8620` (the labeled entry) is NEVER executed** -- the entry-capture is empty,
   yet `0x878a` runs. So execution JUMPS INTO the `0x87xx` block mid-stream from another
   function, in that function's window; the `FUN_00008620` symbol boundary is misleading.
   Consequence: `a14=0x9040` is set UPSTREAM of this block, so the local `a0`-tagged-
   pointer idiom (`a14 = (a0 & 0x3FFFFFFF)+tag`, which with `a0=0x8000d845` computes
   `0xd845`, NOT `0x9040`) is NOT the source. `0x9040`'s birth is one hop further up.
2. **This dispatch path IS the stack-leaking recursion.** `a1` (SP) at the call
   decreases exactly 0x90 (144 bytes) per rebuild cycle: `0x3160 -> 0x30d0 -> 0x3040 ->
   0x2fb0 -> 0x2f20 -> 0x2e90`. That is the 144 B/pass leak measured earlier, now tied
   to the column-power rebuild loop -- and SP (~0x2e90) is closing on SCHED (0x2250).

**Thread 1 status: narrowed, not closed.** `0x9040` is NOT the `a0`-return-address
idiom; it is preloaded into `a14` by whatever jumps into the `0x87xx` block. Next hop:
find the set-site of `a14=0x9040` upstream (backward from the jump into `~0x876d`) and
whether it derives from a stubbed-0 array/aperture base. Register hints to chase:
`a5=0x1eb00`, `a6=0x9268`, `a7=0x245a0`, `a9=0x581c` (the low-text block base).

Probe added: `m2c_probe_target_origin` (callers of `0xc530` by (ret,a6); full reg file
at `0x878a` for the `a14==0x9040` calls; entry-`a14` at `0x8620`; `XDNA_FW_DUMP_N`).

## HW-OBSERVABILITY VERDICT (2026-07-08) -- the values thread 1 needs are NOT host-readable on NPU1

Checked (Explore over xdna-driver + XRT, full citations) whether a HW read could
shortcut thread 1 by supplying the REAL value of a stubbed aperture (what feeds
`a14=0x9040`) or the firmware's computed target. **Verdict: NO for the data we need.**

- **Mgmt Xtensa SRAM** (`0xfaf0` target, `0x9040`, `0xf9e0`, SCHED `0x2250`, `0x10f10`):
  not host-readable as such. BAR2 maps the SRAM window (APERTURE1 base `0x3080000`, full
  length) but the driver interprets only two offsets -- X2I mailbox `0x30A0000` and
  `FW_ALIVE 0x30BF000` (`npu1_regs.c:113-116`); no path returns "the value at firmware
  `0xfaf0`." A raw BAR2 `readl` could alias it ONLY IF the firmware link map places that
  `.data` at a fixed APERTURE1 offset (unknown, driver never uses it) -- and even then it
  would only re-confirm the descriptor target we ALREADY read in EMU, not the stubbed
  aperture value thread 1 needs.
- **Firmware device apertures** (`0x271000`, `0x272000`, `0x200300-0x200410`): **NO.**
  Mgmt-core-private (behind the uc local-bus/MMU), in no BAR. Not expressible via the
  array-register mailbox `MSG_OP_AIE_RW_ACCESS=0x203` (`aie2_message.c:1421`), which
  addresses `(col,row,20-bit tile offset)`: these values are `> 0xFFFFF`. Reinterpreting
  `0x271000` as a full array address decodes to col0 (shim, below `first_col=1`
  `npu1_regs.c:150`) -- unreachable. Firmware logging/telemetry is **BROKEN on NPU1**
  (user-confirmed) -- no log/crash-dump path either. BAR0 (PSP/SMU, `npu1_regs.c:17-27`)
  exposes only scratch/handshake registers, none aliasing firmware data.
- These stubbed-aperture values ARE exactly what thread 1 needs (the base that computes
  `0x9040`). So **HW cannot shortcut it -- the emulator backward-trace is the only path.**

**Useful capability for ARRAY-fidelity work (NOT this thread):** AIE array tile/DMA/lock
registers ARE host-readable via `xrt::aie::device::read_aie_reg` / `DRM_AMDXDNA_AIE_TILE_READ`
(one mailbox round-trip each). Requires the `AIE2_RW_ACCESS` feature bit (our xdna-driver
tree adds it experimentally on NPU1, `npu1_regs.c:73-80`, opcode 0x203) and an owned,
NON-memtile partition column (Phoenix memtile reads `-EPERM`, wedge firmware until reboot
-- `aie.c:499-517`). Filed for later array validation; irrelevant to the mgmt-firmware boot.

## 0x9040-BIRTH STRIKE (2026-07-08, follow-on) -- thread 1 CLOSED; corrects the TARGET-ORIGIN read; the reframe is dead

Traced `0x9040` to the exact producing instruction (`m2c_probe_reg_9040_origin`
catches the AR that first transitions to `0x9040`; `m2c_probe_pc_history` rings the
28 instrs before it). **Result: `0x9040` is NOT a stubbed-aperture value (thread-1's
hypothesis is FALSIFIED) and NOT "set upstream" (the TARGET-ORIGIN strike above was
WRONG on this). It is the local `a0`-return-address idiom, fed a garbage mask because
a windowed call lands one instruction PAST the mask-init.**

**Birth site.** `0x8779 FUN_00008620+0x159 : And a14,a0,a6` -> `a14 = a0 & a6 =
0x8000d845 & 0x9268 = 0x9040`, at n=47788 (long before the n=58.7k spill; that spill
is this same value re-surfacing, per the SCHEDULER-MAPPING section). `a6=0x9268` is
itself `0x8773 Srli a6,a6,2` = `0x249a0 >> 2` (n=47786).

**Why the mask is garbage -- the mechanism.** `FUN_00008620` holds THREE identical
inline "create-descriptor" blocks (`0x8720`, `0x874a`, `0x876d`). Each builds the
low-30 mask INLINE right before the `And`: `Movi a6,-1 ; Srli a6,a6,2` = `0x3FFFFFFF`,
then `And a14,a0,a6` = `a0 & 0x3FFFFFFF` -- the standard reconstruct-a-code-pointer-
from-the-windowed-return-address idiom (`a14 = (a0 & 0x3FFFFFFF) + region`). The block
that runs is entered by `Call8` from `goalive_runfn` at `0x58c9`, whose target decodes
to **`0x8770`** (verified against the known-good `0x878a->0xc530` call; `Call8` targets
are 4-aligned and `0x876d` is unaligned, so `0x876d` is *unreachable by call*). `0x8770`
is ONE instruction past the block's own `Movi a6,-1` at `0x876d`, so the mask-init is
skipped; `a6` keeps `goalive_runfn`'s value `0x249a0` (a table pointer it built as
`a3+offset` and used as an `S8i` store target -- genuinely a data pointer, not a mask).
`Call8` does not rotate the window (only `Entry` does; `0x8770` has none), so the helper
runs in `goalive_runfn`'s window: `a0=0x8000d845` is *goalive_runfn's own* return addr
(it was `Callx8`-called from `0xd842`), and `a8=0x800058cc` is the return `Call8` just
wrote. **Had the mask-init run: `a14 = 0xd845 & 0x3FFFFFFF = 0xd845` -- a live code
pointer beside `task_dispatcher` (`0xd7f0`). Skipped: garbage `0x9040`.**

**So the completion-target mismatch is born HERE.** The column-power descriptor target
`[0xfaf0]=0x9040` should plausibly be `~0xd845`; the ColumnPowerAgent then completes
`[0x9040+0x30]` (inert) instead of a live task, and the dispatcher's wait on
`[0x10f10+0x30]` never satisfies -> the 144 B/pass recursion. The remaining ambiguity
is NOT "what aperture feeds 0x9040" (none does) but **whether `goalive_runfn` reaching
`0x58c9` and Call8-ing into the MIDDLE of `FUN_00008620` (past its `0x8620` Entry, past
the mask-init) is faithful firmware or an upstream control-flow/register divergence.**
Three ways it could be a divergence, all needing the next pull: (a) `goalive_runfn` is
supposed to pass `a6`=mask and its `a6=0x249a0` is wrong; (b) the intended callee is the
`0x8620` Entry and control reached `0x58c9`/this Call8 corrupted; (c) an overlay/symbol
mapping issue means the bytes at `0x8770` differ on HW. It could also be faithful (the
firmware genuinely computes `0x9040`) and the whole descriptor/completion framing is
misread -- but a code-pointer reconstruction fed a data pointer as its mask is a strong
divergence smell.

**Strategic consequence:** the "model the array's aperture config-response" reframe is
NOT the path to the wall -- `0x9040` has no aperture input. The wall is a windowed-call /
register-state divergence at the `goalive_runfn -> FUN_00008620` seam.

Probes added: `m2c_probe_reg_9040_origin` (`XDNA_FW_TARGET` overrides `0x9040`; dumps
distinct producing PCs + pre-step reg files, filtering window-rotation artifacts by
first-transition), `m2c_probe_pc_history` (`XDNA_FW_TRIG` default `0x8773`, `XDNA_FW_HIST`
depth; rings the last N executed `(pc,op,a0,a6)` into a trigger PC).

## COLLAPSE-TO-BIT3 (2026-07-08, Maya: "pull out everything not fully verified, characterize from that kernel"). The descriptor/completion framing was a MISREAD; only the bit3 poll survives -- and bit3 is NOT the boot-progress gate.

Re-derived the whole "column-power completion" model from the code (not the store
pattern), audited its provenance, then stripped the emulator to the verified kernel.

**RE-DERIVATION (disassembly, cold, no descriptor lens).**
- `FUN_00008620` is a **data-TLB / cache remap routine** -- eight `Wdtlb`, an `Idtlb`
  invalidate loop, a `Dii` cache-invalidate loop, `Dsync` barriers, page-descriptor
  loads. NOT a "create-descriptor" fn. Its three tail blocks reconstruct an address
  and post a message. The `Movi a6,-1 ; Srli a6,a6,2 ; And a14,a0,a6` we'd read as
  "build-descriptor mask" is the windowed return-address / page-address masking idiom.
- `0xc530` is a **generic critical-section IPC message-post primitive** (`Rsil 2` ..
  `Wsr PS` bracket; writes 6 words `a2..a7` into `[0xfae0+4..+0x18]`, sets bit0
  "pending" at `[0xfae0]`, `Memw`, calls a notify fn). **28 callers** = the firmware's
  enqueue primitive, proven `a10=0xfae0` (`S32i.n a6,[a10+16]` -> `[0xfaf0]`).
- The `0xfae0` "descriptor" is therefore a **6-word IPC message record + control
  word**; `colmask@8`/`target@10` are our INFERRED names for `a3`/`a6`. `[0xfaf0]=0x9040`
  is the garbage-masked reconstructed address (`0x8000d845 & 0x9268`), not a task ptr.

**PROVENANCE AUDIT (how the claim was originally derived).** All five constants
entered in ONE commit `eece411a`, lifted from this doc, never independently verified.
Sub-claim verdicts: `0xfae0`=descriptor -> INFERRED (store-trace + a colmask/4-col
coincidence; no consumer struct ever matched); field names -> ASSUMED; `[0xfaf0]=0x9040`
target-task + `[+0x30]` done-flag -> ASSUMED then **self-REFUTED** by this doc's own
later sections (`0x9040` not a task, `[0x9070]` never read); `[0xf9e0+col*0x60]` **bit3
poll -> the ONE VERIFIED piece** (executed-trace: `FUN_00008c68` `L8ui a9,[a8]; Bbci
a9,3,<skip>` at `0x8c88`/`0x8c8b`; bit3 CLEAR skips the column, SET services it).

**COLUMN COUNT re-derived (not the descriptor's `0xf`=4).** A natural boot reads
exactly THREE per-column status bytes -- `0xf9e0`, `0xfa40`, `0xfaa0` (stride `0x60`),
all from poll PC `0x8c88` (`m2c_probe_colstatus_poll`). Not four.

**THE STRIP.** `src/firmware/host_mailbox.rs` reduced to the bit3 kernel: assert bit3
on the 3 polled columns each tick, nothing else. Deleted: the `0xfae0` descriptor read,
`colmask`/`target`, the `[target+0x30]` done-flag, the i2x mailbox `CompletionAgent` +
`HostMailboxConsumer`. Full lib suite green (4076).

**CHARACTERIZATION -- bit3 is NOT the boot-progress gate (corrects the old claim).**
`m2c_bit3_advances_boot_past_natural_wall` runs boot BOTH ways to 2M and contrasts.
Natural and bit3 reach the SAME waypoints at the SAME n: task_dispatcher `0xd7f0`
(~47.9k), the col-poll `0x8c88` (47866), goalive_runfn real entry `0x588c` (47761);
current-task `0->0x10f10->0x9040` both; NEITHER reaches publish/idle. bit3's ONLY
effect is the late spin location (@2M: goalive region `0x58b3` with bit3 vs the `~0x880`
window-overflow handler without). So the prior "bit3 broke the livelock and advanced to
the go-alive chain" measured **descriptor-completion count** (fictional), not boot
progress -- which is identical with or without bit3. (Note: `0x55f8`/`0x8c68` are SYMBOL
boundaries, never the exec entry; real entries are `0x588c`/`0x8c88`, same mid-function
entry pattern as `FUN_00008620`.)

**BIT3, DEMOTED (Maya).** Kept as the one verified external stimulus (harmless, gives a
cleaner spin to study), but explicitly on-notice: we do not yet KNOW what bit3 MEANS.
Removal plan gates on characterizing (a) what the bit3-SET branch of `FUN_00008c68`
actually does (the `0x2727x114` handshake it enters) and (b) whether any boot progress
ever depends on it. Until then it stays, demoted, non-load-bearing.

**NEXT (clean eyes -- everything unknown unless proven).** The real wall was never
"column-power completion." It is: **boot reaches `goalive_runfn` (`0x588c`), builds the
go-alive record at `0x2320`, current-task becomes `0x9040`, and spins -- regardless of
any external stimulus we model.** Characterize that spin from scratch: what loop is boot
stuck in at `0x58b3`/`~0x880`, and what memory/condition breaks it -- NOT assuming it is
"the retire gate for worker 0x9040."

Probes added this arc: `m2c_probe_reg_9040_origin`, `m2c_probe_pc_history`,
`m2c_probe_colstatus_poll` (`XDNA_FW_AGENT` toggles the kernel). Suite test replaced:
`m2c_boot_completion_advances_past_recursion` (old-model asserts) ->
`m2c_bit3_advances_boot_past_natural_wall` (verified natural-vs-bit3 invariant).

## THE WALL, RE-CHARACTERIZED FROM SCRATCH (2026-07-09): it is an EVENT-WAIT deadlock, not a column/completion wall

With the misframing cleared, the actual boot wall, derived fresh from the scheduler code
(`m2c_probe_goalive_spin`, `m2c_probe_task_struct`, `m2c_probe_bit3_meaning`; commit
`ed52f906`). Everything below is executed-code-verified.

**The dispatcher cycle** (`task_dispatcher` 0xd7f0 -> body 0xd828), captured pre-corruption
at n=50000 (SP `0x2db0`, leaking exactly 144 B/pass):
- reads current-task `[0x2278]=0x10f10`; its done-flag `[0x10f40]=0` (not done);
- `S8i` state `[0x10f3c]=6`; calls `sched_ready_popcount` (0xc938) which scans the 6-slot
  runnable array `[0x2288..0x229c]` -- ALL EMPTY -> ready-count 0;
- `BnezN a10` (ready!=0) falls through -> `Callx8 [0x11890]=0x588c` = the idle handler
  `goalive_runfn`, which posts the "go-alive" IPC message (via 0x8770->0xc530->0xfae0) and
  returns; its notify path re-enters the dispatcher -> **recursion, 144 B/pass**.

**The three dispatcher exits** (0xd828-0xd848), only the third ever taken:
1. done-flag `[current+0x30]!=0` -> `Call 0xcadc` (retire/deliver-events);
2. ready-count `!=0` -> run the ready task;
3. nothing ready -> call `goalive_runfn`, return. <- the spin.

**Tasks are made ready ONLY by `wake_tasks_by_event_mask`** (0xd84c, immediately below the
dispatcher): it takes an event mask, scans the task table (stride 0x38), and sets
`[task+0x2c]` ready for matches. **No event ever fires**, so the runnable array stays empty
forever.

**Task `0x10f10` is a degenerate current slot:** `+0x00`=`0x12048` is NOT code and executes
0 times; state `+0x2c` stuck at 6, done-flag `+0x30` stuck at 0. The dispatcher never
switches to it -- it is "what is current while the scheduler waits", not a runnable task.

**`0x9040` is confirmed DOWNSTREAM corruption:** the 144 B/pass leak drives SP below zero
(wraps to `0xffff_xxxx` by n=200k) and spills `0x9040` into current-task. A symptom of the
leak, never a cause. The many sessions spent chasing `0x9040`-as-target were chasing a
corruption artifact.

**bit3, closed out:** `m2c_probe_bit3_meaning` shows the bit3-SET branch reads the per-column
MMIO aperture `0x2727(N+1)000` and acts on ITS bit0 (stub=0 -> inert). So bit3 is a "service
column N now" doorbell whose real payload is the aperture; with the aperture stubbed 0 the
body does nothing -- exactly why bit3 is non-load-bearing. Kept demoted; removable once the
aperture seam is (or isn't) modeled.

**THE REAL SEAM (next, #1/#3 together).** Boot needs an **event/IRQ** that
`wake_tasks_by_event_mask` turns into a ready task -- NOT bit3, a done-flag, or a column
aperture. This re-derives the earlier "level-1 completion IRQ masked at INTLEVEL-2" finding
cleanly from the scheduler side (convergence, not proof). NEXT: trace the event PRODUCER
side -- who calls `wake_tasks_by_event_mask` and with what event; is there an ISR that posts
events; is the awaited event an IRQ the firmware expects HW to raise (-> the faithful
external stimulus to inject, as an interrupt not a memory poke). Re-derive the "event system"
/ "AIE-completion ISR" from code; do NOT trust the prior mapping.

## EVENT-PRODUCER STRIKE (2026-07-09, Maya: "#1 and #3 together"): the stimulus IS an interrupt, but it is UNDELIVERABLE; and the demoted aperture is a workload-time handshake, correctly dormant at idle

Three probes, all executed-code-verified, settle the event-producer question and rule out the
aperture as the idle escape.

**#1 -- the producer is an ISR, never reached by fall-through** (`m2c_probe_event_source`,
1.5M-instr natural boot): the event-poll region (`FUN_00005580` / `sched_event_poll` 0x5524)
is **never entered** and the event-source register `0x27010d28` is **read 0 times**.
`wake_tasks_by_event_mask` (0xd84c) has a single caller inside that poll region, so the only
way a task becomes ready is via **interrupt delivery**. The awaited event is therefore a
**hardware interrupt**, not a memory poke -- vindicating THE PRINCIPLE. Final PS.INTLEVEL at
the spin = **2**.

**#3 -- the interrupt is UNDELIVERABLE** (`m2c_probe_inject_interrupt`, warmup 100k + 400k
run): the dispatcher's `rsil 2` at entry (0xd7f3) holds **INTLEVEL pinned at 2 for the entire
spin -- zero level-0 delivery windows**. Only `INTENABLE=0x1` (one level-1 line) is enabled,
which `rsil 2` masks. Faithfully asserting the line + seeding `0x27010d28` -> the interrupt is
**never taken, stays pending forever**. This re-derives the "level-1 completion IRQ masked at
INTLEVEL-2" finding from the scheduler side, now PROVEN not merely convergent.

**The wall is a self-sustaining deadlock CYCLE** (`m2c_probe_goalive_spin`, one dispatcher
cycle at n=200k, 413 steps): the idle handler has **no `waiti` on its path**. One pass:
`0xd7f0 dispatcher (rsil 2; nothing ready)` -> `0xc530 post go-alive IPC to [0xfae0]` (pending
ALREADY 1, no consumer) -> `0xb0e710 cache-flush the message (Dhwbi loop + Dsync)` -> `0x7fc4
-> 0x8c6c col-service` (aperture bit0=0 -> skip) -> **re-enters 0xd7f0 with SP 144 B lower
(recursion, not a loop)**. Closed cycle: no `waiti` -> no IRQ delivery -> no wake -> nothing
ready -> idle recurses -> leaks 144 B/pass -> corrupts current-task to `0x9040`. The masking
and the leak are the SAME wall from two angles.

**The demoted aperture, fully decoded and RULED OUT** (`m2c_probe_apertureset_branch`, forces
the never-taken branch): with bit3 on, at the col-service aperture load `0x8c93`
(`L32iN a9,[0x2727(N+1)000]`) we forced `a9|=1` so the `Bbci a9,0` at 0x8c95 falls through.
The never-executed body is a **column power-up hardware handshake**:
```
0x8c98  Memw
0x8c9b  S32iN [0x2727(N+1)114] <= 1     ; ring doorbell: firmware acks / proceed
0x8c9d  Memw
0x8ca0  L32iN a9, [0x2727(N+1)000]      ; poll status
0x8ca2  Bbci  a9, 1 -> 0x8ca0           ; spin until bit1 (completion ack) set
```
The trace after forcing = one store `[0x27271114]<=1` then a **tight infinite spin** waiting
for bit1 (stubbed 0, never set); no calls, no ready-array writes, no wake. Crucially this body
is **gated by bit0 ("power request pending")**, which a natural idle boot leaves 0 -> the
firmware correctly **SKIPS the handshake**. So the stub-0 aperture is **faithful for
boot-to-idle** (no column-power request is pending at idle); bit3 + this aperture are
**workload-time column power management, not the idle stimulus**. Forcing bit0 only manufactured
an artificial handshake wall. This **rules the aperture OUT** as the idle escape and vindicates
bit3's demotion.

**Where this leaves the idle escape.** Interrupts are masked until a `waiti` that never comes,
so **no interrupt of any level can break the idle recursion** -- the completion IRQ we were
about to inject included. The escape must be whatever, on silicon, terminates the go-alive
recursion BEFORE stack exhaustion and lets boot reach a `waiti` (where the IRQ then becomes
deliverable). Two live hypotheses for next session: **(H1)** the go-alive IPC at `[0xfae0]`
needs an external consumer (host/SMU/PSP) that clears pending and responds -- missing consumer
= the divergence; **(H2)** the idle run-fn ptr `[0x11890]=0x588c` (goalive, no waiti) should be
updated to `0x5524` (`sched_event_poll`, which HAS the `waiti` + the `0x27010d28` poll) by a
boot step we diverge before reaching. Both are memory/handshake seams, not the interrupt.

## H1 AND H2 BOTH FALSIFIED (2026-07-09): the divergence is a create-but-never-LINKED task; the promote-call site is pinned

Two cheap probes settle both banked hypotheses as NEGATIVE and collapse the wall onto a pure
firmware task-lifecycle divergence -- no external stimulus, no interrupt, no run-fn repoint.

**H1 FALSIFIED -- the go-alive IPC is fire-and-forget** (`m2c_probe_goalive_spin`, one full
413-step dispatcher cycle at n=60301). The ONLY access to `[0xfae0]` in the entire cycle is a
read-modify-write inside the post primitive `0xc530` that OR-s in the pending bit:
`0xc53f L32i a9,[0xfae0]=0x1` -> `0xc546 Or a8,a9,1` -> `0xc551 S32iN a8,[0xfae0]`. The firmware
**never reads a reply field and never branches on `[0xfae0]` being cleared**; post -> `0xb0e710`
cache-flush -> falls straight into `0x7fc4`/`0x8c6c` col-service -> re-enters the dispatcher. So an
external consumer that clears pending / writes a response would change NOTHING -- there is no
request/response transaction to complete. The "model an SMU/PSP/host consumer of `[0xfae0]`"
direction is retired.

**H2 FALSIFIED -- the idle run-fn ptr is write-once** (`m2c_probe_addr_store_watch`
`XDNA_FW_WATCH_ADDR=0x11890`, 1.5M natural boot). `[0x11890]` is stored **exactly once** --
n=41480, `pc=FUN_0000daf0+0x2`, value `0x588c` -- and never rewritten. Nothing is supposed to
repoint it to `0x5524`; `0x588c` is genuinely the kernel idle handler. H2's mechanism has no
trigger.

**What survives: create-but-never-LINKED, and the promote site is now pinned.** The same
store-watch on the runnable array `[0x2288..0x22b0)` (`m2c_probe_taskstart_calls`, new) shows the
array is written ONLY during early init by the link primitive `FUN_0000d53c+0xd3` (`0xd60f`):
kernel workers land in slots 6/7/8 (`0x10dfc`/`0x10e58`/`0x10eb4`) at n=39852..40152. **After
n=40152 nothing legitimately writes the runnable array** (the lone n=57973 `slot5 <- 0x60922` by
`0xc530+0x6` is post-wall spill corruption). The go-alive task, created at n=47335, is never
linked into a slot.

**The concrete promote-call site (`m2c_probe_taskstart_calls`, natural boot).** Right after
`task_create(run-fn=0x55f8, idx=4, col=0xff)` at `0x3de9` sit two indirect calls -- the prime
"start/enqueue task #4" candidates -- and boot DOES reach and execute both:
```
n=47335  0x3de9  Call8 0xd664     task_create(a10=run-fn=0x55f8, idx=4, col=0xff)
n=47405  0x3dec  (returned, a2=1 success)
n=47407  0x3df1  Callx8 a8        a10=4  -> target a8=0x08b041f0   (Segment-B, no symbol)
n=47422  0x3df4  (returned)
n=47424  0x3df9  Callx8 a8        a10=0  -> target a8=0x08b043cc   (Segment-B, no symbol)
```
Both call targets are inside **Segment-B (`0x08b00000`)** -- the same region whose pointers the
firmware writes into the `0x27200170..190` aperture (`0x08b041bc` etc.). Both calls execute and
return cleanly, but **neither writes the runnable array**: the promote does not happen.

**The fork (next pull).** Either (a) those Segment-B calls ARE the real promoter and they diverge
internally -- read stubbed-0 state and bail before the enqueue -- or (b) the call target is
mis-resolved: `a8` is loaded from a literal pool (`0x3dee`/`0x3df6 L32r`), and if that is a
Harvard fetch-vs-data overlay artifact (the class of bug that made `[0x349c]` read 0), we are
jumping into Segment-B DATA instead of the intended code. **Deciding step: trace into the `a10=4`
call (`0x08b041f0`)** -- what it reads and where it decides not to enqueue. That one trace
distinguishes (a) from (b) and is the most direct line to the promote step. Interrupts, `[0xfae0]`
consumers, and `[0x11890]` repoints are all ruled out.

## SEGMENT-B START-CALL DEEP TRACE (2026-07-09): fork (b) dead; the boot->scheduler handoff is a real Syscall context-switch; the missing promote is a write to runnable SLOT 4

Traced INTO both post-`task_create` calls (`m2c_probe_segb_startcall`, natural boot, full
annotation). Fork (b) is FALSIFIED and the actual boot->scheduler mechanism is now mapped.

**Fork (b) DEAD -- the targets are real code.** Raw-byte dump + decode: both `0x08b041f0` and
`0x08b043cc` begin `Entry {s:1,imm:96}` and decode as clean Xtensa. Segment-B (`0x08b00000`)
holds executable code and the `Callx8` targets (loaded via `0x3dee`/`0x3df6 L32r`) resolve
correctly. NOT a Harvard-overlay artifact.

**Call #1 (`0x08b041f0`, arg a2=index=4) = a device-SRAM notify, not a link.** RMW on device
SRAM `0x3010d7c`: `L32iN`=`0x2` -> `(0x2<<8)|4` -> `S32iN 0x204`. Posts the task index into a
host-visible SRAM slot (bumps a generation byte). No scheduler-array touch. `RetwN`.

**Call #2 (`0x08b043cc`, arg=0) = the boot->scheduler cooperative yield (a real `Syscall`).** It
sets a couple of local words then `Wur ur231` + **`Syscall` at `0x08b043e1`**, trapping to the
handler `0x2958`. The handler builds the INIT thread's context frame at `0x12048` -- saves the
continuation `0xa0003dfc` (return to `0x3dfc`) and the full register window, links it via
`[0x10f10]<=0x12048`, increments `[0x2284]` (SCHED+0x34) 0->1 -- then switches to the supervisor
stack `0x3170` and enters the scheduler. So the init/boot thread SUSPENDS itself as a task
(current-task `0x10f10`, continuation `0x3dfc`) and hands off to the scheduler. **This overturns
the YIELD-SYSCALL STRIKE's "zero `Syscall` in boot" claim** -- that count was under the discarded
agent-enabled model; natural boot DOES execute this one `Syscall` at n~47430, and it is the
handoff, not a yield-inside-the-park.

**On scheduler entry: no registry drain, immediate recursion.** On the `0x3170` stack
`sched_ready_popcount` (`0xc938`) scans exactly **6 runnable slots** `[0x2288..0x229c]` (hardcoded
`MoviN a5,6`), all empty, and **never reads the create-registry `0x2450` or the go-alive record
`0x2320`**. SP descends `0x3160 -> 0x30d0 -> 0x3040 -> 0x2fb0` -- the 144 B/pass recursion begins
right here. There is genuinely NO "process pending creates / link go-alive" step between
`task_create` and the spin.

**The missing promote, pinned to a SLOT.** `task_create` was called with **idx = 4** (`a11=4`,
echoed as call #1's `a10=4`). Runnable slot 4 = `0x2288 + 4*4 = 0x2298` -- which is INSIDE the
6-slot scan range. The static kernel workers sit in slots 6/7/8 (`0x22a0/a4/a8`, OUTSIDE the
6-slot scan -- a parked/system category, which explains PICKER-GATE's "ready slot-7 worker never
picked": it is simply out of scan range). So go-alive's target slot (**4 = `0x2298`**) is exactly
the region the scheduler DOES scan, and nothing ever writes it. **The missing promote step is
precisely: `[0x2298] <- go-alive-task` with `state[+0x2c]=1`; the very next 6-slot scan would then
dispatch go-alive** (-> its run-fn -> publisher `0x50e8` -> `waiti 0x56e6` = idle).

**Refined next question (the promoter, target now exact).** What firmware primitive writes
runnable-slot[idx] = task-ptr with state=1 for a `task_create`'d task (idx in `a11`), and why does
it not run for go-alive (idx 4 -> slot `0x2298`)? Candidates, in order: (1) `task_create` (`0xd664`)
SHOULD perform the slot write and a branch inside it diverges (re-read `0xd664` for a
`[0x2288+idx*4]` store gated on a condition our boot fails); (2) a separate "start/admit" primitive
should run after the SRAM-notify + yield and is skipped. The `idx=4 -> slot 0x2298` mapping is the
lens for both -- watch who writes `0x2298`.

## TASK_CREATE DRILL (2026-07-09): task_create only STAGES; init yields via the Syscall and is NEVER RESUMED; the schedulable band (slots 0-5) is always empty

Traced `task_create` (`0xd664`) EXECUTED during the go-alive create (`m2c_probe_segb_startcall`
with `XDNA_FW_TRACE_FROM=0x3de9`). Resolves the promoter question and reframes the wall.

**`task_create` STAGES, does not LINK.** Over its 70-instr body it: checks create-count
`[0x24c4]=0 < 15` (no early-exit); builds the go-alive record at SCHED+208 (`0x2320`=run-fn
`0x55f8`, `0x2324`=col `0xff`, idx byte `0x232b`=4, state byte `0x2330`=1); bumps counters
(`[0x24c4]` 0->1, `[0x24b4]`, `[0x24c8]`). It writes NO runnable slot and calls NO link primitive.

**The only immediate post-create action is a conditional PREEMPT, correctly NOT taken.** Tail:
`0xd7b9 Bgeu a13,a4 -> 0xd7c4` where `a13`=new-idx=4, `a4`=`[0x10f3d]`=current-task priority byte=0.
The fall-through (skipped) block is `0xd7bf Call8 0x2694` (the reschedule/preempt primitive; static
disasm overlay-garbled, never executes in our boot). Gate: preempt iff new-idx < cur-priority. With
`4 >= 0`, go-alive does NOT preempt init -- CORRECT for a low-priority background task. So this is
not the divergence; `0x2694` is just the preempt path.

**REFRAME -- init yields via the Syscall and is NEVER RESUMED.** The init thread (task `0x10f10`,
priority 0) creates go-alive, notifies SRAM (call#1), then `Syscall`-yields (call#2, `0x08b043e1`
-> handler `0x2958` -> context-switch to scheduler stack `0x3170`), its continuation saved at frame
`0x12048` (return to `0x3dfc`, a `MoviN a2,0; RetwN` that would return UP into more boot init). But
boot **never returns to `0x3dfc`** (verified: 0 hits post-Syscall): the scheduler's 6-slot popcount
(`0xc938`, hardcoded `MoviN a5,6`) scans slots 0-5 `[0x2288..0x229c]`, finds them all empty, and
the dispatcher falls to the idle handler `0x588c` and recurses. Nothing context-switches back to
init.

**The core anomaly: the schedulable band (slots 0-5) is ALWAYS EMPTY.** Every task lives OUTSIDE
the scanned band: the 3 kernel workers were linked into slots 6/7/8 (`0x22a0/a4/a8`, past the
6-slot window); go-alive is staged in the registry (`0x2320`); init `0x10f10` is current-task but
in no slot. So NO task is ever schedulable via the primary dispatcher's scan, which is exactly why
it always idles. (This subsumes the earlier "ready slot-7 worker never picked" -- slot 7 is out of
scan range.)

**The fork (next, a genuine scheduler-mapping choice).** (A) init `0x10f10` should be in a runnable
slot (0-5) so the scheduler RESUMES it after the yield -> init continues (`0x3dfc` -> caller -> more
boot -> eventually admits go-alive / enters the command loop); divergence = init's missing
runnable-slot link or its state byte (currently 6 "serviced", not 1 "ready"). (B) the workers'
idx->slot mapping put them in 6/7/8 but the scan covers 0-5 -- an idx-computation divergence (trace
the link primitive `FUN_0000d53c`/`0xd4e0` EXECUTED during the worker links n~39.8-40.1k to see how
slot 6/7/8 is derived and whether it should be 0-2). (C) the fuller picker `0xc980` (never called,
per PICKER-GATE) is the real selector and is gated behind a command-syscall that never fires.
Sharpest single next drill: pin the idx->slot mapping in the link primitive -- it decides whether
"band 0-5 empty" is a link-index bug (B) or a two-band design where the primary scan is not the
selector for these tasks (C).

## SCHEDULER MAP (2026-07-09): fork (B) dead -- the array is priority-indexed; the ADMIT for created tasks is structurally absent; the real PICKER never runs

Three drills (link-primitive executed trace, registry-access watch, waypoint-hit + call-xref) map
the scheduler and localize the divergence to a missing task-admission path.

**The runnable array is PRIORITY-INDEXED (fork B dead).** Link primitive `0xd4e0` (executed trace
of a worker link, n~39.9k): it builds a full TCB for the task then `0xd538 Addx4 a3,a3,a15`
(`a15`=SCHED) + `0xd60f S32iN task,[a3+56]` -> writes `[SCHED + 56 + priority*4]`. The worker
`0x10e58` has priority **7** -> slot 7 (`0x22a4`). So slots 6/7/8 holding the workers is NOT a
link-index bug -- they are low-priority daemons deliberately outside the 6-slot fast scan (0-5).
`0xd4e0` is called ONLY from `task_init` `FUN_00004570` (4 sites) -- exclusively for kernel tasks.

**The ADMIT for created tasks is STRUCTURALLY ABSENT (`m2c_probe_registry_access`, full boot).**
Every access to the create-registry control block (`[0x24c4]` count, `[0x24b4]`/`[0x24c8]` flags)
and the go-alive record (`[0x2320]` run-fn, `[0x2330]` state) comes from `task_create` ITSELF
(all PCs `0xd6xx`, n~47342-47396). After staging, **the go-alive record is NEVER read again**
(`reads=0`) and the count `[0x24c4]` is never re-read across 1.5M instructions. So nothing ever
drains the registry / converts the staged record into a TCB / links it into a priority slot. The
created go-alive task is written once and orphaned. (Note: the registry record layout != a TCB --
state byte at `+0x10`, not `+0x2c` -- so admit must BUILD a TCB like `0xd4e0` does, then link;
it is not a one-word slot poke.)

**The real PICKER never runs, because neither entry is reached (`m2c_probe_waypoint_hits`,
`m2c_probe_call_xref`).** Two selectors exist: the IDLE dispatcher `task_dispatcher 0xd7f0` (the
6-slot popcount `0xc938`; idles via `0x588c` when the band is empty; runs constantly) and the real
PICKER `0xc980` (scans/selects properly). The picker has exactly two callers -- `FUN_000041b8+0x110`
(early-init) and `FUN_0000dbc4+0x1b6` (command dispatcher). Both are **NEVER** reached in boot, and
so are the functions containing them (`FUN_000041b8`, `FUN_0000dbc4` -- both entry PCs never hit).
The command dispatcher not running is expected (no host commands at boot); but `FUN_000041b8`, the
**boot-time scheduler-start / early-init picker call, is structurally unreached**. Both
`FUN_000041b8` and the scheduler loop `FUN_00007fa0` have NO direct callers -- they are
indirect/table (run-fn) targets; `FUN_00007fa0` IS reached that way (it is the loop init recurses
in), `FUN_000041b8` is NOT.

**Unified picture.** Boot runs the IDLE dispatcher forever but never (1) admits the created
go-alive task into a priority slot, nor (2) invokes the real picker. The kernel workers ARE linked
(priorities 6/7/8) but sit outside the 6-slot fast scan; go-alive (priority 4, would land in the
scanned slot 4) is never admitted; init yields via the `Syscall` (-> `0x2958` -> the idle
dispatcher, NOT the picker) and is never resumed. The single missing action is the task-admission
path that (a) builds+links created tasks into priority slots AND/OR (b) routes boot to
`FUN_000041b8`/the picker. **Next drill: disasm/understand `FUN_000041b8`** (the boot picker-caller)
-- what it is, how it is meant to be invoked (which run-fn table / dispatch reaches it), and why
that indirect call never fires. That names the exact boot-scheduler-start divergence.

## SLOT-SUFFICIENCY TEST -> MODEL CORRECTED (2026-07-09): readiness is EVENT-GATED via `[task+0x38]`; the scheduler map CONVERGES back to the event-wait deadlock

A decisive diagnostic experiment (`m2c_probe_slot_sufficiency`) FALSIFIED this session's own
"slots 0-5 empty is the sole gate" hypothesis and corrected the dispatch model.

**The experiment.** At a dispatcher entry (n=49066, SP=`0x2f60`, pre-corruption), relocate an
existing READY worker TCB pointer (slot 7 = `0x10e58`, `state[+0x2c]=1`) into the scanned slot 4
(`0x2298`) -- a minimal faithful poke (a real TCB, not fabricated state) -- then continue.
**Result: NO advance.** The ready-return `0xd845` was hit **0** times; the idle handler `0x588c`
ran 509x; boot recursed to SP wrap (`0xffff10f0`). Putting a state=1 task in a scanned slot did
NOT make popcount count it.

**Why -- the real popcount semantics (disasm `0xc94c..0xc979`).** For each slot with a task,
`sched_ready_popcount` reads `state[+0x2c]` AND `await[+0x38]`, and only if `state==1` does it
`Or a3, [task+0x38], a3` -- i.e. it accumulates the task's **`[+0x38]` await-mask** into a
ready-bitmap, then popcounts it and returns nonzero only when >= 2 bits are set. So a task counts
as dispatchable iff `state==1` AND `[+0x38]` contributes bits. **A state=1 task with `[+0x38]=0`
contributes NOTHING** -- exactly the relocated worker's case.

**Verified scheduler state at the poke (dump):**
```
slot6 [0x22a0]=0x10dfc  state=0  await[+0x38]=0
slot7 [0x22a4]=0x10e58  state=1  await[+0x38]=0   <- ready state, ZERO await-mask
slot8 [0x22a8]=0x10eb4  state=6  await[+0x38]=0
slots 0-5: empty
```
All three workers have `await[+0x38]=0` (the link primitive `0xd4e0` never writes `+0x38`). So
**none is dispatchable even though one is state=1**, and the empty scanned band is a red herring:
the true gate is the await-mask.

**CONVERGENCE.** `[+0x38]` is the event await-mask that `wake_tasks_by_event_mask` (`0xd84c`)
operates on. So the popcount-based idle dispatcher is **event-gated**: a task becomes dispatchable
only when the event system sets its await-mask, and **no event fires** (EVENT-PRODUCER STRIKE:
the producer is a masked ISR). This session's independent scheduler mapping thus arrives at the
SAME wall as the event analysis, now proven from the DISPATCH side: no event -> all await-masks 0
-> popcount always 0 -> the dispatcher always idles + recurses. It also corrects the earlier
"admit go-alive into slot 4" idea -- admission alone is insufficient; a dispatched task needs a
nonzero `[+0x38]`, which only an event supplies. The non-event path (the real picker `0xc980`,
which may select without the await gate) never runs.

**Net for the arc.** The wall is doubly confirmed as event-gated: (event producer = masked ISR
that never fires) AND (dispatch readiness = event await-mask `[+0x38]` that only that ISR sets).
The two remaining escapes are the two we have NOT closed: (1) the real picker `0xc980` / its
boot-time caller `FUN_000041b8` (an unreached indirect-dispatch target) that would select tasks
WITHOUT the event gate; (2) whatever, on silicon, fires the first event/IRQ that sets an
await-mask -- reconnecting to the INTLEVEL-2 masking (no `waiti` -> no delivery). Both were already
on the table; the scheduler map did not add a third escape, it unified the two.

## PICKER / IDLE / INTLEVEL DRILL (2026-07-09) -- the wall is an INTLEVEL trap

Maya: "follow the real picker and boot caller, next." Done. It sharpened the wall
from "event-gated" to a precise **INTLEVEL-2 trap**, and tied the bootstrap to a
single armed interrupt line.

**1. Picker `0xc980` disassembled.** Windowed (`Entry`), `Rsil imm:2` (runs at
INTLEVEL 2), indexes the runnable array at `SCHED + idx*4 + 56` (confirms
`SCHED=0x2250`, array `0x2288`), reads the state byte `[task+44]`, calls popcount
`0xc938`, and bookkeeps the `+92`/`+96` counters. It is the real dispatcher.

**2. Picker callers are table-only.** Direct callers: `0x42c8` (inside
`FUN_000041b8`) and `0xdd7a` (inside `FUN_0000dbc4`). Both *containing* functions
have **no direct callers** -- reached only via `callx*`/table. `FUN_000041b8` is a
heavy (re)config routine: `Idtlb` x8 (TLB reset), per-block init (stride 440),
byte-field descriptor parse, ending in the picker call -- not a hot path.

**3. Sole event producer is inside go-alive.** `wake_tasks_by_event_mask`
(`0xd84c`) has exactly ONE caller: `goalive_runfn+0x211` (`0x5809`), i.e. go-alive
itself (base `0x55f8`). The bootstrap circle: go-alive must run to wake the
workers, but go-alive is staged-not-linked.

**4. Dispatch is COMPUTED, not a table.** Raw-image scan for the LE address words
of `{0xc980, 0x41b8, 0xdbc4, 0x42c8, 0xdd7a, 0xd84c}` -> **zero matches**. None is
stored as data. The scheduler core is a **function-pointer/vtable OS**: picker,
idle-wait (`FUN_0000c8e0`), and both picker-callers are reached via `Callx8` off
scheduler-ops pointers -- which is why none has direct callers.

**5. Timer FALSIFIED.** No `CCOMPARE` (SR 240-242) writes and no `CCOUNT` (SR 234)
reads anywhere in the image. The firmware does not self-tick; the bootstrap is not
a core-timer interrupt. (The interp models neither SR either, but that is moot.)

**6. High-level interrupts FALSIFIED.** **Zero `Rfi` instructions** in the whole
image (only one `Rfe`, plus window over/underflow returns). `Rfi` = return from
high-level interrupt; its total absence means **no level-2+ interrupt handlers
exist**. So line 0 is a **level-1** interrupt (returns via the single `Rfe` /
general-exc path). "The interp can't deliver a high-level IRQ" is therefore NOT
the gap.

**7. Exactly one armed line.** Steady-state `INTENABLE = 0x00000001` -- interrupt
**line 0** only. This is THE bootstrap interrupt. (Armed via a generic
`irq_enable(line)` helper `FUN_00008884`: raise INTLEVEL, OR the line bit into
INTENABLE, restore PS.)

**8. The CPU is pinned at INTLEVEL 2.** `m2c_probe_inject_interrupt`: at warmup the
CPU sits at INTLEVEL 2 / EXCM, current-task `0x9040` (a worker); across 400k steps
`min_intlevel = 2`, `level-0 windows = 0` -- it **never** drops to level 0.
Injecting line 0 does nothing (masked: `interrupt_deliverable` needs
`intlevel==0 && !excm`). It busy-spins in `sched_ready_popcount` (final pc
`0xc967`) at level 2.

**9. Every DESIGNED idle drops to level 0.** All `Waiti` immediates in the image
are **0**. `FUN_0000c8e0` is the proper idle: set PS (drop level) -> call a hook ->
`waiti 0` (`0xc8eb`) -> loop. But the dispatcher busy-spins popcount at level 2 and
**never enters `FUN_0000c8e0`**, so no deliverable window ever opens.

**SHARP CHARACTERIZATION.** The wall is an **INTLEVEL trap**. Worker `0x9040` ran,
blocked, and returned to the dispatcher, which busy-spins popcount at INTLEVEL 2.
Only line 0 (level-1) is armed. Level-1 delivery requires INTLEVEL 0, reached only
via `waiti 0` (or a running task). The dispatcher-spin never reaches `waiti 0` ->
no deliverable window -> the armed line-0 interrupt can never fire -> no await-mask
is ever set -> popcount stays 0 -> the dispatcher spins. The INTLEVEL-2 pinning and
the "no ready task" deadlock are the **same** wall from two sides; this drill did
not add a third escape, it mechanized the interrupt half.

**WHY IT MATTERS.** Line 0 is the single bootstrap interrupt, and the blocked
current-task `0x9040` is an array/column-init worker -- so line 0 is almost
certainly the **AIE-array completion IRQ** (status reg `0x27010d28`, iter18). This
ties the boot wall directly to the firmware-dream premise: *the emulated array must
raise line-0 completion*. But the current state cannot take it (trapped at level
2). The open fidelity question is whether, on silicon, worker `0x9040` -- after
arming line 0 -- **`waiti 0`s** (a deliverable window) rather than returning to the
level-2 dispatcher spin. If our model routes the block to the dispatcher instead of
to a `waiti 0`, that skipped window is the divergence to fix.

**NEXT.** Trace worker `0x9040`'s block path: where/how it blocks after arming line
0, whether it reaches a `waiti 0`, and its `await[+0x38]` / `state[+0x2c]`. That
decides the fix: (a) model the array's line-0 completion IRQ + ensure the worker
reaches a `waiti 0` window, or (b) fix a block-path divergence that skips `waiti 0`.

## STATIC-ANALYSIS SUBSTRATE VALIDATED (2026-07-09) -- the overlay is sound

Maya flagged that garbled static disasm anywhere is a problem: if the PSP
load-overlay is faulty anywhere, the executed traces and every static claim built
on them are suspect. Settled it by building an overlay-correct recursive-descent
disassembler (`m2c_probe_gold_disasm`, `XDNA_FW_PROBE=1`) on OUR ground-truth
decoder, reading via `bus.fetch8` so the base `+0x5c` / two `+0x100` low-VMA /
Seg-B overlays apply identically to execution. Gate = every reachable
`Op::Unknown` + an inst-fetch-vs-data-load Harvard check. Full verdict:
`build/experiments/firmware-re/overlay-completeness-verdict.md`.

**VERDICT: the overlay is NOT faulty.** Descent from the reset vector decodes
coherent Xtensa throughout, including coherently ACROSS the one boundary a fault
could hide behind (the `LOW_TEXT_BLOCK` `+0x100` edge at `0x581c`, inside
go-alive's tail -- clean `RetwN`+`Entry` boundaries there). The "garbage" has
three benign causes: (1) linear-sweep desync on the `$PS1` header (vaddr `<0x1a4`
is header, not code) and inline literal pools -- descent marks these as data gaps
and the boot-region garbage vanishes; (2) ~150 of 163 raw Unknowns are misaligned
Ghidra `FUN_` labels seeding descent mid-instruction (reset-only descent, no
symbol seeds, drops Unknowns 163 -> 13); (3) the residual are DATA descent walked
into (literal pools, jump tables, `.bss`) -- NOT unimplemented opcodes (an earlier
draft said so; that was a misdiagnosis). `xtdis` (AMD-config libisa) confirms:
its "`excw`" output is a fallback-for-unknown (it prints `excw` for `0xffffff`
too), so neither our decoder nor libisa recognizes these bytes because they are
not code. DECISIVE: the boot executes 2,000,000 instructions with ZERO Unknowns
-> the decoder is complete for all executed code; there is no decoder chore. The
descent was then hardened (two-pass `l32r`-literal-pool awareness; symbol-seed
validation, 127 misaligned Ghidra labels skipped; low region bounded to
`0x10000`), dropping full-seed Unknowns 163 -> 37 (reset-only holds at 13, all
data / the `0x581c` overlay-straddle). The 214-359 Harvard mismatches are exactly
the two `+0x100` overlay regions, by design.

**Cross-validation of the picker-drill claims.** None of the 13 gaps is
`rfi`/`rfe`/`ccompare`/`ccount`; Ghidra's recursive-descent listing (`listing.txt`)
independently reports `rfi:0 rfe:0 ccompare:0 ccount:0`, and INFODUMP.md's Ghidra
sweep already stated "ZERO CCOUNT/CCOMPARE." So no-timer, no-high-level-interrupt
(line 0 is level-1), VECBASE=0x800, and the INTLEVEL-2 dispatcher pin all hold on
trustworthy ground (linear sweep -> Ghidra + xtdis + this descent, three ways).
No open decoder chore (the boot runs 2M instrs with zero Unknowns); the residual
Unknowns are jump-tables/computed-branches/`.bss` -- a cosmetic disassembler
polish only, not a mapping or decoder fault.

## WORKER-BLOCK-PATH DRILL (2026-07-09, Maya: "trace worker 0x9040 reaching waiti 0 or not after arming line 0") -- answered, and it is a THIRD option

The picker drill left a binary open question: after arming line 0, does worker
`0x9040`'s block path reach a `waiti 0` (a deliverable window) or return to the
INTLEVEL-2 dispatcher spin? Ran `m2c_probe_intenable_watch` + `m2c_probe_steady_histogram`
+ the gold listing. The answer is neither (a) nor (b) as framed -- it is a third
shape, and it is cleaner.

**1. INTENABLE line 0 is armed ONCE in early boot, not by `0x9040`.** `intenable_watch`:
INTENABLE goes `0 -> 0x1` at **instr 2218** (pc `0x200088d5`), i.e. symbol offset
`0x88d5` -- deep in early scheduler init, ~56k instructions BEFORE worker `0x9040`
is even dispatched (~n=58754). At the very next instruction (2219, pc `0x88d8`)
INTLEVEL is raised `0x1 -> 0x2`, and it **never returns to 0** for the entire
1,000,000-instruction run. Final state: `INTENABLE=0x1 INTERRUPT=0x0 INTLEVEL=2`,
stop = "budget reached" -- **NOT** `Wait`. So the firmware executes **no `waiti` at
all** in ~998k instructions after arming. Worker `0x9040`'s block path does NOT
reach `waiti 0`; it returns into the level-2 spin. That much confirms the picker
drill.

**2. But the scheduler never even APPROACHES idle.** `steady_histogram` (warmup 200k,
2M samples): the entire steady state sits in `sched_ready_popcount` (`0xc938`), the
picker tail `FUN_0000c96c`, and `FUN_00008c68` -- plus a large no-symbol bucket.
The idle function `FUN_0000c8e0` (which holds the only `waiti 0`, at `0xc8eb`) is
**entirely absent** from the histogram. So the fix is NOT "reaches `waiti 0` at the
wrong INTLEVEL" (a) and NOT "a block-path divergence skips `waiti 0`" (b). It is:
**the scheduler never becomes idle**, because a task stays perpetually runnable, so
`sched_ready_popcount` always reports work and the dispatcher always has something
to pick -- the idle path (with its `waiti 0`) is never entered. Note `FUN_0000c8e0`
does `Wsr PS` (lowers INTLEVEL) at `0xc8e5` immediately before `waiti 0` -- so IF it
were reached, the window WOULD be deliverable. It never is.

**3. `sched_ready_popcount` disassembled (`0xc938`).** `Rsil imm:2`; scan a 6-entry
runnable array at `base+56` (`base` = lit `0x3d28`); for each entry whose state byte
`[task+44]==1`, `OR` its `[task+56]` mask into an accumulator; popcount the
accumulator over 32 bits; **return whether count >= 2**. Restores PS, `RetwN`. This
is the pure-internal-state gate: `steady_histogram` reports **zero external MMIO
reads** (`>= 0x2000_0000`) in the whole 2M-sample steady state -- the firmware is
NOT polling any hardware register for completion. It expects an interrupt or a
memory flag set by an external agent.

**4. The state-machine CONSUMER found: `FUN_00008c68` is a per-column completion-drain.**
Its sole caller is `FUN_00007fa0+0x41` (the scheduler tick, which then calls
`0xd7f0` and `0x26d4`). `FUN_00008c68` loops over **3 columns** (struct stride 96,
MMIO stride **4096** = the AIE per-column register stride); for each column whose
struct byte has **bit 3 set** and `[+8]==a2`, it does a `memw`-fenced store of `1`
to a column MMIO register and then **clears bit 3**. In steady state every column
has bit 3 clear, so it runs its 3 iterations and returns having done nothing
(hot spots `+0x20`/`+0x23` = load-byte/test-bit-3, `+0x46`/`+0x49` = advance). This
is the routine that would advance the boot state machine the moment an external
completion SETS bit 3 on a column -- and "bit3" is exactly the verified stimulus
this arc collapsed to (`ed52f906`), here shown to be a per-column completion-ack
consumer, not the boot-progress gate directly.

**SHARP RESULT.** The wall is a single external-signal starvation with two coupled
consumers, both dormant because the signal never arrives: (i) `sched_ready_popcount`
never sees a task go non-runnable (nothing clears the perpetual state==1), so the
scheduler never idles to a `waiti 0`; (ii) `FUN_00008c68` never sees a column's bit
3 set, so it never acks a completion. The firmware reads NO MMIO in the spin -- it is
waiting to be *interrupted* (line 0) or to have a *memory flag written* by the array,
not polling. This is THE PRINCIPLE's contract surface made concrete: the emulated
AIE array must produce the completion signal (line-0 IRQ and/or the bit-3 column
flag). The INTLEVEL-2 pin means even a modeled line-0 IRQ can't be delivered until
the scheduler reaches an idle `waiti 0` -- which it only does once a task retires --
so the completion likely arrives as a **memory write** (the bit-3 column flag set by
a DMA/agent), not purely as a masked CPU interrupt. That memory-flag path is
deliverable at INTLEVEL 2 (it is a plain load in `FUN_00008c68`), which resolves the
INTLEVEL paradox: the drain does not need the interrupt, only the flag.

**NEXT (fix-design fork, for Maya).** Two coupled producers to locate/model: (1) who
SETS bit 3 on a column struct (the completion-ack producer `FUN_00008c68` consumes)
-- static hunt for `S8i`/`Or ...,8` into the `base = lit 0x3d28`-ish column structs,
then decide if the emulated array writes it; (2) who clears a task's state byte
`[+44]` from 1 (the retire that lets popcount fall) -- likely the same completion.
Deferred pending Maya's call on whether to model the array completion now or map the
producers first.

## PRODUCER MAP (2026-07-09, Maya: "map the producers, do it RIGHT not fast") -- the full event pipeline, end to end

Static map of every producer/consumer in the boot livelock, resolved from the gold
listing + `peek`/`literal_xref`/`disasm_range` + a deep `exec_trace` of one full
outer cycle. Two independent pipelines, both present and correctly wired, both
dormant because their single external trigger never arrives.

**The actual steady-state outer loop (deep exec_trace @ warmup 305k).** Not a tight
popcount spin -- a full scheduler tick that RUNS go-alive every iteration:

    task_dispatcher (0xd7f0)
      cur-task = [sched+40] = [0x2278] = 0x9040
      read state[0x9040+27]=0 (Bnei !=1 -> not the dispatch path)
      -> FUN_0000d828: read pending[0x9040+48]=0 (BeqzN ==0)
           set state[0x9040+44]=6; call sched_ready_popcount (=0)
           load run-fn [0x11890]=0x588c; Callx8 ->
             goalive_runfn+0x294 (0x588c):   <-- go-alive IS running, every tick
               reset a batch of state bytes (0x123d0/0x1eb00/0x249a0/0x245a0/...)
               -> FUN_00008620 (TLB/cache) -> FUN_0000c530: post IPC msg to [0xfae0]
               return
      loop back to 0xd7f0

So "go-alive is never dispatched" (an earlier framing) is now superseded: go-alive's
run-fn `0x588c` is invoked by the dispatcher every tick. It re-posts the `[0xfae0]`
IPC and resets state each pass -- a retry loop, not fire-and-forget. The two gate
values that would break it are **`state[0x9040+27]`** (dispatcher `Bnei ==1` at
`0xd811`) and **`pending[0x9040+48]`** (dispatcher `BeqzN ==0` at `0xd82a`); both are
0 forever.

**PIPELINE 1 -- the event/readiness pipeline (gates on the global accumulator
`[sched+108] = [0x22bc]`).** All wired, all dormant:

- `post_event(mask)` = **`FUN_0000cf5c`** (`0xcf5c`): if `mask` not already in
  `[0x22bc]`, `[0x22bc] |= mask`, call a notify hook `[lit 15668]`, then walk the
  runnable array to deliver. Returns 254 if already pending. **Reached ONLY via
  vtable** (zero direct callers) -- the completion ISR would call it through a
  function pointer. THIS is the missing setter of `[0x22bc]`.
- `deliver_pending_events(mask)` = **`0xcadc`**: `pending[cur+48] = [0x22bc] & mask`,
  then per-task callbacks. Callers: picker `FUN_0000c994+0x3f` (`0xc9d3`) and the
  dispatcher-done path `FUN_0000d828+0x4` (`0xd82c`, taken only when pending!=0 --
  never, in steady state).
- `wake_tasks_by_event_mask(mask)` = **`0xd84c`**: for each runnable-array task whose
  await-mask `[+0x38]` intersects `mask`, set `state[+44]=6` and clear the matched
  bits from `pending[+48]`/`[task+108]`. **Sole caller: go-alive `+0x211` (`0x5809`)**
  -- but that is BEFORE the `+0x294` continuation the dispatcher re-enters, so wake is
  not even called in steady state.
- The `[0x22bc]` writers that are NOT the setter, for the record: `FUN_00003fb8`/
  `FUN_0000c9dc` (bulk struct-init, offsets 96..108 in sequence), `FUN_0000d53c`
  (zeroes every field 4..108 -- a `memset`-style SCHED initializer), `FUN_0000cb38`
  (rolls a task's own `pending[+48]` into `[0x22bc]` during delivery). Only
  `FUN_0000cf5c` ORs an external `mask` in.

**PIPELINE 2 -- the per-column completion pipeline (gates on bit 3 of
`[0xf9e0 + col*0x60]`).** Independent of pipeline 1, also dormant:

- setter = **`FUN_00008c14`** (`0x8c59`: `[0xf9e0+col*0x60] |= 8`) -- claims the first
  column whose bit 3 is clear. Bases resolved by `peek`: struct `0xf9a0` (+64 =
  `0xf9e0`). Caller NOT in the descent; per prior mapping it is the array-init HAL
  `FUN_00035444`, dormant at idle.
- consumer/ack = **`FUN_00008c68`** (sole caller `FUN_00007fa0+0x41`): for each of 3
  columns (struct stride 96, MMIO stride 4096) with bit 3 set, `memw`-store 1 to a
  column MMIO reg `[0x27271114 + col*0x1000]` and clear bit 3 -- but the store is
  gated on **bit 0 of the aperture `[0x27271000]`** (peek-resolved), which idle boot
  leaves 0 (matches the demoted-aperture finding: correctly dormant at idle).

**ROOT.** Both pipelines wait on the same thing: an external completion that either
(a) calls `post_event` (setting `[0x22bc]`, waking a task whose await-mask matches),
or (b) sets bit 3 on a column / bit 0 on the `0x27271000` aperture. `post_event` is
vtable-reached, so its caller is the completion ISR -- which needs the line-0 IRQ
that INTLEVEL-2 masks (the interrupt half of this same wall). The `exec_trace`
confirms the firmware reads NO external MMIO in the loop: it is not polling; it is
waiting to be signalled. **THE PRINCIPLE's contract surface is now exact:** the
emulated array must, on completing its programmed work, drive ONE of -- the line-0
completion IRQ (whose ISR calls `post_event`), a `post_event`-equivalent write to
`[0x22bc]`, or the bit-3/`0x27271000` column-completion flags. Which one silicon
uses is the next empirical question; the memory-flag paths (bit 3, `0x22bc`) are
deliverable at INTLEVEL 2, the IRQ is not.

**NEXT (producer root -- to close, then decide model vs. more static).** Find where
`post_event`'s address (`0xcf5c`/`0xcf68`) is stored as a data pointer (the ops-struct
/ vtable slot the ISR loads it from), and the ISR that owns that slot -- that pins the
exact completion-ISR entry and confirms the line-0 binding. Then Maya's model-vs-map
call: model the array completion (drive the chosen signal) or first pin the ISR's
event-id (which `mask` bit the completion posts, so `wake` matches an await-mask).

## ISR INCH (2026-07-09, Maya: "close the ISR inch, break the livelock only AFTER full understanding") -- solid parts + the wall static analysis hits

Traced the event-post primitive back toward its interrupt trigger. Nailed the top
and the primitive; the middle is callback-dispatched at every level, which is where
pure static reading runs out.

**SOLID (base-ISA, cross-checked):**
- `post_event(mask)` = `FUN_0000cf5c` has exactly ONE call in the descent:
  **`FUN_00002730+0x1e1` (`0x2911`)**, `Call8 0xcf68`. It passes `mask=[obj+56]`,
  posted when `[obj+56] != [obj+48]` (a head!=tail "item available" test). So the
  thing that sets `[0x22bc]` is reached from the `0x2730` region.
- **General exception/interrupt handler = `0x2958`** (entered by hardware vectoring
  from VECBASE=0x800; the gold descent never reaches it, only the vector does). It
  saves full context (EXCSAVE2-6, EPC1, EXCVADDR, EXCCAUSE, all a0-a15, SAR, PS,
  LBEG/LEND/LCOUNT, the User/FLIX regs 231-233), then **dispatches on EXCCAUSE**:
  `Beqi EXCCAUSE,1 -> 0x2a88` splits **Syscall (cause 1)** from **everything else
  (the interrupt path)**.
- **Interrupt path** (EXCCAUSE != 1, fall-through at `0x2a6c`): sets up args
  (`[0x2278]` cur-task, a hook object), does **`Callx4 [literal]`** at `0x2a81`, then
  `J 0x2ac0` (context-restore + return). So the real ISR body is behind a function
  pointer in a literal.
- **Correction:** the `0x272003b8` MMIO read (and the bits-12-15 source scan +
  per-index `Call8 0x5a18` loop) is in the **Syscall path `0x2a88+`, NOT the
  interrupt path** -- and that region is FLIX and mis-decodes under linear disasm. So
  `0x272003b8` is a syscall-time register, NOT confirmed as the completion source.
  (Earlier-in-this-drill claim retracted before it was committed.)

**THE WALL static analysis hits -- the ISR is callback-dispatched all the way down.**
The interrupt-path `Callx4 [literal]` target resolves ambiguously (`[0x293c]=0xd864`
lands mid-instruction = wrong; `[0x2940]=0xdac4 = FUN_0000dab0+0x14` is the clean
entry, so the handler is almost certainly `FUN_0000dab0`). And `FUN_0000dab0+0x14`
is itself a **trampoline**: `L32i a2=[obj+168]; Beqz; Callx8 a2` -- it calls a
**runtime-installed callback pointer** at `[obj+168]`. So the chain is
`vector -> 0x2958 -> Callx4 [literal] -> trampoline -> Callx8 [obj+168] -> ?`, every
hop through a pointer that is WRITTEN at init, not encoded in the instruction stream.
Pure static reading cannot resolve the tail (source-bit -> handler -> `post_event` +
which `mask`/event-id) without also tracing the pointer installs, and the region is
FLIX-heavy (linear disasm desyncs; the gold descent never vectors in). The interp is
the FLIX ground truth but it NEVER delivers the interrupt (INTLEVEL pinned at 2), so
it cannot trace the ISR under natural boot either.

**FORK (for Maya -- bears on "understanding before breaking").** To read the ISR tail
reliably, two options: (A) a **controlled single-delivery inject** -- momentarily drop
INTLEVEL/excm for ONE line-0 delivery purely so the interp executes the ISR and
reveals the resolved callback chain + the event-id it posts (diagnostic OBSERVATION
of the ISR mechanism, distinct from modeling the array and declaring boot fixed); or
(B) **trace the pointer installs statically** -- `store_value_watch`/`store_search`
for who writes the `[lit]`/`[obj+168]` callback slots at init, then hand-resolve. (A)
is the FLIX ground truth and far cheaper; (B) stays fully static but slow and still
FLIX-blocked in spots. Recommend (A), framed strictly as ISR-path observation.

## (A) ISR OBSERVATION RESULT (2026-07-09, Maya: "try (A) once") -- delivery is faithful; the ISR chain resolves; completion is a MEMORY flag; the FLIX "faults" are linear-sweep, not a decoder bug

Built `m2c_probe_isr_observe` (`XDNA_FW_PROBE=1`): warm to steady state, FORCE ONE
faithful level-1 delivery (drop PS.INTLEVEL->0, clear PS.EXCM, `INTERRUPT |=
INTENABLE`), and trace the ISR through the interp -- the FLIX ground truth (its
`step()` runs `Op::Flix1` bundles natively). One hand-forced delivery, no array
modelled: DIAGNOSTIC, not a livelock break.

**RESULT (unseeded -- no completion source):**
- **Delivery is faithful end to end.** `interrupt_deliverable` -> `step()` raised
  EXCCAUSE=4, EPC1<-pc, vectored to `0x2958`; the interp executed the ENTIRE ISR
  (282 steps) with ZERO Unknowns; `rfe` returned cleanly to the preempted pc
  (`0xc95b`). The interp's interrupt model is sound top to bottom.
- **The runtime ISR chain (pointers resolved LIVE, which static could not do):**
  `0x2958` (ctx save) -> `0xe354 Callx4 -> 0xd900` (`FUN_0000d8a8+0x58`, the handler)
  -> `0xd95b Call8 -> 0xc530` (the `[0xfae0]` IPC post) -> `0xc55c Callx8 ->
  0x08b0e710` (a Seg-B callback) -> `0x7fc4` scheduler tick -> `0x8c68` column-drain
  -> `0xd7f0` dispatcher -> popcount -> `rfe`. My two static guesses for the Callx4
  target (`0xd864`/`0xdac4`) were BOTH wrong; the live pointer is `0xd900`.
- **The handler `FUN_0000d904` (entry `0xd908`) is an EXCCAUSE sub-dispatcher:** reads
  a cause code from the EXCSAVE regs / `[sched+176]` and branches on its value
  (`Beqi 4`, `Bne 13`, `Beqi 16`, `Bne 24` -- these are EXCCAUSE codes: 4=Level1Int,
  16=InstTLBMiss, 24=LoadStoreTLBMiss). For our source-less cause-4 it does IPC +
  reschedule.
- **The ISR reads NO MMIO** (zero loads `>= 0x2000_0000` across all 282 steps). So
  the completion source the ISR would act on is a **memory flag**, NOT an MMIO status
  register. Corroborates "the firmware waits to be signalled" and BURIES the
  `0x272003b8` lead for good.
- **With no completion present it posts no event:** `[0x22bc]` unchanged
  (`0x588c` throughout -- note that value is a code pointer `goalive_runfn+0x294`, so
  `[sched+108]`'s role needs a re-read; it is NOT a plain event bitmask here),
  `wake`/`post_event` never reached, cur-task stays `0x9040`. The ISR is a generic
  service-and-reschedule until a completion flag is set.

**The FLIX question, answered concretely (Maya: "if disasm is faulty, WHY?").** It is
NOT a decoder bug. The interp (which shares the same decoder) executed the whole ISR
-- including whatever FLIX bundles it traverses -- with zero Unknowns. The "garbled"
static output came from **linear-sweep desync**: in a region reached only by hardware
vectoring (never control-flow-descended), a byte-walk cannot know instruction
boundaries where code, inline literal pools, and 8-byte FLIX bundles interleave, so
it mis-splits and prints Unknowns. The fix is not decoder work -- it is to disassemble
these regions by DESCENT from a real entry (or, as here, just let the interp execute
them). The gold descent already does this for control-reachable code; the ISR tail is
simply vector-reached, so neither the gold descent nor a linear sweep covers it, but
the interp does.

**NEXT (extend A, targeted).** To surface the event-id / wake, discover the completion
MEMORY flag the ISR gates on: trace the ISR's data READS (not just MMIO) to find what
`FUN_0000d904` / the Seg-B callback `0x08b0e710` check, then seed that flag (via
`XDNA_FW_ISR_SRC` or a memory write) and re-run `m2c_probe_isr_observe` to watch it
post an event and wake go-alive's await-mask. This is A+B combined, now cheap because
the ISR path is known.

## A' MODEL CORRECTION (2026-07-09, Maya: "fix the [sched+108] model first") -- the "event accumulator" was a MISREAD; the bitmask event system is DORMANT and my done-flag experiment corrupted a pointer

Went to resolve whether `[sched+108] = [0x22bc]` is an event bitmask accumulator or a
pointer. It is a **POINTER**, and the correction cascades.

**Evidence:**
- `deliver_pending_events` (`0xcadc`) and `wake_tasks` (`0xd84c`) both do bitmask ops
  on `[sched+108]`: `deliver` computes `a6 = accum & mask; a5 = accum ^ a6` (=
  `accum & ~mask`) and stores it back; `wake` does `[+108] &= ~mask`. `post_event`
  (`0xcf5c`) does `[+108] |= mask`. Read in isolation this LOOKS like a pending-event
  bitmask -- which is what I committed.
- BUT: static (load-time) `[0x22bc] = 0`; at steady state `[0x22bc] = 0x588c`;
  `addr_store_watch` on `0x22bc` caught **zero** stores across 1.5M boot instrs; and
  `store_value_watch` for `0x588c` found it stored **once** (n=41480, `FUN_0000daf0`)
  to `[0x11890]` -- the confirmed go-alive run-fn pointer -- NOT to `[0x22bc]`.
  `[0x22bc]` gets `0x588c` via a watch-invisible fill/copy path.
- `0x588c = goalive_runfn+0x294`, a CODE ADDRESS, identical to the run-fn pointer at
  `[0x11890]`. Steady-state sched context: `[sched+108]=0x588c`, `[sched+120]=0x9268`
  -- pointer-valued neighbours. A stable value equal to a code address, never
  accumulated/cleared, is a POINTER, not a live bitmask.

**Correction:**
- `[sched+108]` is a **pointer field** (holds the go-alive/current run-fn), NOT the
  "global event accumulator" the earlier PRODUCER-MAP / pipeline-1 sections claimed.
  Those sections' "event-id / [0x22bc] accumulator" framing is **demoted** -- treat as
  a misread pending re-derivation.
- The bitmask event-delivery functions (`deliver`/`wake`/`post_event` operating on
  `[sched+108]`) **do not run during real boot**: `deliver`/`wake` are gated behind
  `[task+0x30] != 0` (never set), and `post_event` is reached only via the dormant
  producer `FUN_00002730`. They are latent code. If run, their `& ~mask` / `|= mask`
  would CORRUPT the `[sched+108]` pointer.
- **My seeded done-flag experiment was therefore NON-faithful and corrupting.** Forcing
  `[0x9040+0x30]` nonzero drove the dispatcher into `deliver`, which did
  `[sched+108] & ~mask` (`0x588c -> 0`), clobbering the run-fn pointer; the "562k
  steps of progress then return to idle" was boot running on corrupted state, not a
  real advance. Retract the "done-flag is the completion consumer that advances boot"
  reading.

**What still holds:** the faithful-delivery mechanism (interp raises EXCCAUSE=4, vectors
`0x2958`, runs the ISR clean, `rfe` returns) and the runtime ISR chain
(`0x2958 -> 0xd900 -> 0xc530 -> Seg-B -> tick -> dispatcher`). The ISR reads NO MMIO.
And the dispatcher gates are unchanged: cur-task `0x9040` loops because
`state[+27]=0` / `pending[+48]=0` forever.

**OPEN (the real question, reframed).** With the bitmask-event-system demoted, what
LEGITIMATELY sets `state[0x9040+27]=1` or `pending[0x9040+48]!=0` in a real boot? The
`[sched+108]` event path is dormant, so it is not that. Candidates to chase next:
`[task+0x30]`'s only non-init setter is `FUN_00002730` (the dormant producer, which
ALSO touches `[sched+108]`), and `[task+27]`'s setters (`FUN_000044d4`, `FUN_000061a8`,
`FUN_00006374`, `FUN_0000d4a0+0x7f`, `FUN_0000e750`). Decide with Maya whether the
completion mechanism is one of these or something not yet located.

## GATE-SETTER ANNOTATION (2026-07-09, Maya: "chase the gate-setters, annotate them anyway no matter what") -- ALL dormant during boot; the dispatcher gates are frozen at creation-time

Annotated every setter of the two dispatcher gates. The headline: **none of them
runs during boot**, so cur-task `0x9040`'s gates are frozen at their creation-time
values (0) and nothing in the executing code can flip them.

**The dispatcher gates (task_dispatcher `0xd7f0`, cur-task `[sched+40]=[0x2278]=0x9040`):**
- `state[task+27]` -- `L8ui` at `0xd808`, `Bnei ...,1` at `0xd811`. `[task+24..27]` is
  a 32-bit descriptor stored little-endian, so `[+27]` is its MSB; the check is
  `(descriptor >> 24) == 1`. If ==1 -> dispatch path `0xd814` (`Call8 0xc530` IPC post,
  args `1,21`, then done-check). Else -> done-check `0xd828`.
- `pending[task+48]` -- `L32iN` at `0xd828`, `BeqzN` at `0xd82a`. If !=0 ->
  `deliver_pending_events` (the now-known-dormant, `[sched+108]`-corrupting path). If
  ==0 -> set `[task+44]=6`, popcount, `Callx8` the go-alive run-fn (the loop).

**`state[task+27]` setters -- all 5 are bulk struct-init / serializers, NOT targeted
"make runnable" transitions, and all are NEVER reached in boot (1.5M instrs):**
| Setter | What it is | `[+27]` value |
|---|---|---|
| `FUN_000044d4+0x31` (`0x4505`) | writes a run of byte fields +12,13,14,20..27 from one reg | fill (`a2`) |
| `FUN_000061a8+0xec` (`0x6294`) | `Srai/Srli`-unpacks a 32-bit word `a10` (= `FUN_00004b00` ret) into bytes +24,25,26,27 | `a10>>24` (descriptor MSB) |
| `FUN_00006374+0x4e` (`0x63c2`) | same unpack of `a10` into +24..27 | `a10>>24` |
| `FUN_0000d4a0+0x7f` (`0xd51f`) | struct init; writes +12,44,45,47,27 | `0` (`a9=MoviN 0`) |
| `FUN_0000e750+0x2a` (`0xe77a`) | writes a run of byte fields +18,20..29 | fill (`a4`) |

So there is **no code anywhere that sets `state[task+27] := 1` as a runnable
transition**; `+27` is only ever written as the high byte of a serialized 32-bit
descriptor at `[task+24]`, built by `FUN_00004b00` (packs a type tag `389<<17` into the
high bits). The descriptor -- hence whether a task is the dispatchable class -- is fixed
at task construction, in a routine that does not execute during boot.

**`pending[task+48]` setter (the only OR-in, non-init one):** `FUN_00002730+0x1d9`
(`0x2909`) -- the event PRODUCER (sets `[task+48]=[obj+56]` then `post_event`). Also
**NEVER reached in boot.** The remaining `+48` writers are struct-init (`FUN_0000c9dc`,
`FUN_0000d53c`) or the dormant `deliver`/`wake` clears.

**Reachability (waypoint_hits, natural boot n<1.5M):** `FUN_000044d4`, `FUN_000061a8`,
`FUN_00006374`, `FUN_0000d4a0`, `FUN_0000e750`, `FUN_00004b00` (the descriptor builder),
and `FUN_00002730` (the pending producer) -- **every one NEVER**.

**IMPLICATION.** The livelock is structural at the gate level: for cur-task `0x9040`,
`state[+27]` and `pending[+48]` are whatever they were at creation (both read 0
throughout), and **no executing code can change them**. So a real completion can only
reach the dispatcher by (a) an EXTERNAL write to `state[0x9040+27]` / `pending[0x9040+48]`
(the array/DMA setting a task field directly -- the one path that doesn't need the
dormant setters), or (b) the framing is upstream-wrong: `0x9040` should not be cur-task,
and the bug is in task creation/linking (ties to the earlier "go-alive created but never
linked" thread, and to the standing question of whether `0x9040` is even a real task vs a
garbage-masked address). This annotation does not decide (a) vs (b) -- it removes every
in-boot code path as the setter, which is the useful negative result.

## CUR-TASK VERIFICATION (2026-07-09, Maya: "verify -- is 0x9040 a real cur-task or is that the corruption?") -- CONFIRMED corruption; the livelock PREDATES it; converges with the 2026-07-08 stack-overlap root

Verified from task-creation forward. **`cur-task = 0x9040` is corruption, not a
context-switch** -- independently re-derived, and it re-confirms + sharpens the
2026-07-08 "stack overlaps SCHED" root cause above.

**The cur-task pointer `[sched+40] = [0x2278]` has exactly TWO writes in all of boot**
(`addr_store_watch`, now `S32e`-aware):
- n=41463, `FUN_00004570+0x13c` (`pc=0x46ac`) writes **`0x10f10`** -- the legitimate
  scheduler context-switch. `0x10f10` is the real cur-task.
- n=58753, **`pc=0x000895` -- a window-overflow-spill `S32e`** -- writes **`0x9040`**.
  This is NOT scheduler code; it is the register-window overflow handler (`0x880`
  vector) spilling `a6` (which happened to hold `0x9040`) to `[a0-24]`, and `a0-24`
  equals `0x2278`. So the SCHED cur-task pointer is clobbered by a stack spill. (The
  decode-based watch missed this until `S32e` was added to its store match -- the same
  blind spot that hid the `[0x22bc]` writes.)

`0x9040` is therefore garbage (a spilled register value), exactly as the "0x9040 is
downstream corruption, not a task" thread long suspected -- now nailed to the precise
instruction.

**The livelock PREDATES the corruption.** The dispatcher's first done-flag check
(`0xd828`) fires at **n=47896 with `a4 = 0x10f10`, pending `[0x10f10+0x30] = 0`** --
i.e. while cur-task is still the LEGIT `0x10f10`, the dispatcher already takes the
`pending==0` arm and calls the go-alive run-fn, looping. That cooperative-yield
recursion (no task ever becomes ready) is what marches the stack (firmware base
`0x3170`, only ~3.8 KB above SCHED `0x2250`) down through the SCHED table until the
n=58753 spill clobbers cur-task. **The corruption is a SYMPTOM of the readiness
livelock, not its cause** -- fixing the pointer would not help; the loop was already
running on the valid task.

**Convergence (this whole session).** Three threads meet on one root:
1. *Stack-overlap (2026-07-08):* the dispatch recursion descends into SCHED because no
   task becomes ready.
2. *Gate-setter annotation (this session):* every setter of the readiness gates
   (`state[task+27]`, `pending[task+48]`) is DORMANT in boot -- no executing code can
   ready a task.
3. *`[sched+108]` correction (this session):* the bitmask "event accumulator" was a
   misread pointer; that event path is dormant too.
   
So the single root, now triangulated: **nothing readies a task during boot because the
readiness-producing code (`FUN_00002730` producer, the `[task+24]` descriptor builder,
the event system) never runs** -- and it never runs because its trigger is absent. The
trigger is either an EXTERNAL write (array/DMA sets a task's readiness field directly)
or an upstream task-creation/linking step that this boot never performs. The
verification does not decide which, but it converts "0x9040 won't complete" into the
correct question: **what, in a real boot, first readies a task (the legit `0x10f10` or
the go-alive task) -- external stimulus or an unperformed creation/link step -- BEFORE
n~47896 when the recursion begins?**

## FIRST-DISPATCH TRACE (2026-07-09, Maya: "trace 0x10f10's dispatch n=41464..47896, find the precondition its run-fn checks that never gets satisfied") -- the healthy first pass is byte-identical to steady state; the run-fn has NO path to task-readiness

Traced the legit worker's FIRST run-fn dispatch (`m2c_probe_runfn_trace`, `XDNA_FW_RUNFN=0x588c`,
warmup 41000) -- the healthy window BEFORE any recursion/corruption. Result: **the first pass is
byte-for-byte the steady-state loop.** There is no special first-dispatch logic that links go-alive
or seeds an await-mask. Full mechanized path:

- **n=41464..47760 (the ~6.3k-instr window) is boot-INIT, not the run-fn.** cur-task = `0x10f10`
  is set at n=41463, but its run-fn `0x588c` does not ENTER until **n=47761**. The intervening
  instructions are boot-init (task_create writes the go-alive record at n=47360). So "trace
  0x10f10's dispatch" = trace from n=47761; the earlier window is init staging tasks.
- **The run-fn `0x588c` (n=47761..47882) does exactly two externally-relevant things, then returns:**
  1. **Flushes a column-power command descriptor** via `FUN_0000c530` (`[0xfae0]` build:
     valid=1, colmask=`0xf`, target field; `Memw` + `Callx8 0xb0e710` `Dhwbi`/`Dsync` cache-flush).
     Fire-and-forget to the external SMU/array.
  2. **Polls per-column completion bit 3** via `FUN_00008c68` @ `0x8c88`:
     `L8ui [0xf9e0+col*0x60]; Bbci bit3` for cols 0/1/2 (`0xf9e0`/`0xfa40`/`0xfaa0`) -- **all read 0**,
     so all three take the skip branch. No completion present.
- **The bit3-SET body sets NO task state (gold listing `0x8c8e..0x8cba`, the branch never taken
  here).** When bit3 IS set it reads `[col+8]`, compares to an expected word, touches the external
  MMIO ack (`L32iN [0x27271000]` bit0 -> `S32iN [0x27271114]=1`, `Memw` fences), then `And ~0x08` +
  `S8i [col]` to **clear bit 3**. It sets no await-mask `[+0x38]`, no task done-flag `[+0x30]`,
  links nothing into the runnable array. **Servicing a completed column does not ready any task** --
  independently re-confirming the prior VALIDATED "bit3 alone doesn't advance boot".
- **The dispatcher's no-ready path is real firmware code that RE-DISPATCHES, with no `waiti`.**
  Run-fn returns -> `FUN_00007fa0` `Call8 task_dispatcher` (`0xd7f0`). Dispatcher reads cur-task
  `[0x2278]=0x10f10`, its state `[0x10f2b]=0` (`!=1` -> not "ready-to-switch"), falls to `0xd828`:
  reads pending `[0x10f40]=0` (`BeqzN` taken), marks cur-task `state[+44]=6` ("serviced"), calls
  `sched_ready_popcount` (`0xc938`). **popcount scans priority-band slots 0-5 (`0x2288..0x229c`) --
  all EMPTY** (the 3 workers sit at slots 6-8 = `0x22a0/a4/a8`, OUTSIDE this 6-slot scan; go-alive,
  priority 4, WOULD be in-band but is never linked), ORs each `state==1` slot's await-mask `[+0x38]`,
  popcounts -> **0**. Back at `0xd839`: count 0 -> `BnezN` not taken -> `L32iN t3,[0x11890]=0x588c;
  Callx8 t3` -> **re-enters `0x588c`** (n=48115). No `waiti`, INTLEVEL still 2. The window-overflow
  spill fires immediately after (n=48122, `0x880` handler) -- the stack has begun its descent.

**So the precondition that never gets satisfied = `sched_ready_popcount > 0`**, which requires a
task in the scanned band (priorities 0-5) with `state==1` AND a nonzero await-mask `[+0x38]`.
Nothing in the run-fn OR the column-completion path ever sets an await-mask; only
`wake_tasks_by_event_mask` does, whose only callers are go-alive (never dispatched) and the line-0
IRQ handler (masked at INTLEVEL 2; this idle loop never reaches `waiti 0` to unmask it). `[0x11890]`
holds `0x588c` by design (H2: written once, never repointed) -- the firmware INTENDS to re-run the
idle keepalive run-fn until a task becomes ready. It never does.

**Answers Maya's reframed fork: it is BOTH, one knot.** (a) The creation/link step IS unperformed --
go-alive (prio 4, in-band) is staged at `0x2320` but never linked into `0x2288`; even the linked
workers sit at prio 6-8 outside the scan with await-mask 0. (b) But linking alone is INSUFFICIENT
(SLOT-SUFFICIENCY proved it -- a state=1 in-band task with `[+0x38]=0` still popcounts 0); the
in-band task needs a nonzero await-mask, which ONLY an event/IRQ supplies. So the true bootstrap is
the **external line-0 completion IRQ** -- blocked by the INTLEVEL-2 pin + the waiti-less idle loop.

**The sharpest remaining lead (unchanged gate, now mechanized): what, on silicon, sets the FIRST
await-mask given INTLEVEL is pinned at 2 and the idle loop has no `waiti`?** The strongest candidate
is the **boot-init Syscall that is never resumed** (TASK_CREATE DRILL: init yields via the Syscall
context-switch to the `0x3170` scheduler and is NEVER resumed). Whatever boot-init would do AFTER
that yield -- link go-alive into the ready band, and/or arm the first await-mask, and/or route the
idle to `waiti 0` -- never runs, because the scheduler it yielded to idles on `0x588c` forever.
Next pull: trace the Syscall handler's resume/return path -- what SHOULD switch back to the init
task, and why the scheduler never does. (Fork is Maya's; presented, not taken.)

Probe: `m2c_probe_runfn_trace` (`XDNA_FW_RUNFN`/`_WARMUP`/`_INSTRS`; linear instr trace from the
first run-fn entry after warmup, with load/store EA+value annotation). Log:
`build/experiments/firmware-re/first_dispatch_588c.log`.

## THE BREACH (2026-07-09, Maya: "pursue the Syscall handler's resume path"): the general-exception handler mis-routes EVERY syscall because EXCSAVE1-7 were unmodeled -- FIXED; the livelock is broken, the 0x9040 corruption is gone

Traced init's yield Syscall (`m2c_probe_segb_startcall`, `FROM=0x2958 WARMUP=47000`) and found the
root divergence -- an **interpreter bug**, not a firmware-state gap.

**What the handler does.** At `0x2958` the general-exception handler stashes live state into the
EXCSAVE scratch SRs on entry (before it has a stack/free ARs): `EXCSAVE2<=a5`, `EXCSAVE5<=a2`,
`EXCSAVE4<=EXCVADDR`, `EXCSAVE6<=a4`, **`EXCSAVE3<=EXCCAUSE`** (`Rsr sr232; Wsr sr211` at `0x2967`/
`0x296a`). It then saves init's full context to a frame at `0x12048` (resume PC `0x3dfc`), writes
init's TCB `[0x10f10]<=0x12048` (init is saved + resumable), links the frame via `[0x2b60]`/`[0x2b64]`/
seg-B `[0x8b04404]`, switches to the `0x3170` supervisor stack, and at `0x2a66` reads **`EXCSAVE3`**
back and `Beqi ...,1 -> 0x2a88` to dispatch: EXCCAUSE==1 (SYSCALL) takes the syscall-service path
`0x2a88`; anything else falls through to the interrupt/reschedule path `0x2a6c`.

**The bug.** `Cpu::read_sr`/`write_sr` (`interp/mod.rs`) modeled SAR/WINDOW*/EPC1/PS/VECBASE/PTE*/
INT*/SCOMPARE1/EXCCAUSE/EXCVADDR but **NOT EXCSAVE1-7** (`sr209..215`): `Wsr` to them was a logged
no-op and `Rsr` returned 0. So the handler's `Wsr EXCSAVE3<=EXCCAUSE` was dropped and the `Rsr
EXCSAVE3` at `0x2a66` read **0** -> `Beqi 0,1` NOT taken -> **every syscall is mis-routed to the
interrupt/reschedule path.** Init's cooperative yield was never serviced, so init was never re-queued/
resumed; the scheduler idled on `0x588c` forever (the livelock), and the unbounded re-dispatch
recursion eventually walked the stack into SCHED and produced the `0x9040` cur-task spill. (The bug
also zeroed the `a2/a4/a5` saved into init's frame, via the same dropped EXCSAVE2/5/6.)

**The fix (faithful).** Modeled EXCSAVE1-7 as a real 7-entry register file (`Cpu::excsave`, indexed
`sr - 0xD1`), routed in `read_sr`/`write_sr`. EXCSAVE registers are architectural Xtensa state that
every exception handler uses as entry scratch -- modeling them is squarely faithful, not a workaround.

**Verified effect.** After the fix: (1) the dispatch at `0x2a69` now branches to the syscall path
`0x2a88` (trace `returned=true`); (2) a 4M-instr boot no longer recurses -- **cur-task becomes the
legit `0x10f10` and STAYS there; the `0x9040` spill/corruption is GONE**; (3) `cargo test --lib` =
4085 pass, and the only 2 failures are `m2c_bit3_advances_boot_past_natural_wall` /
`m2c_boot_advances_into_c_runtime`, which pin the OLD intermediate-wall trajectory (they assert boot
reaches `goalive_runfn 0x588c` / exercises window exceptions -- both now happen AFTER a new, more
advanced wall, so they trip). Not real regressions; superseded by a correct fix.

**The new frontier (n=47551).** Boot now advances ~11k instructions further and walls at a genuine
**unmodeled opcode**: `pc=0xdad2` word `0x00983100` in `FUN_0000dab0` (the syscall/ISR trampoline the
syscall path `Callx4`s into). op0=0 (NOT a FLIX bundle), LSC4 group (decoder `op1=9`), `op2=8` --
base Xtensa LSC4 defines only L32E (op2=0) and S32E (op2=4), so op2=8 is either a custom AMD/AIE
load-store variant or (caveat) an earlier mis-sized op in that trampoline drifting the PC (we enter
at the mid-function offset `+0x14` via `Callx4`). Next: disassemble `FUN_0000dab0` from its real
entry to confirm alignment, then identify/model the `op1=9/op2=8` op. This is the first wall PAST the
scheduler livelock -- the multi-session boot-to-idle blocker is broken.

Probe: `m2c_probe_segb_startcall` (`FROM=0x2958`/`0x2a88`) traced the handler + syscall path;
`m2c_probe_poll_watch` (`MAX=4000000`) confirmed no-corruption + the new stop. Logs:
`build/experiments/firmware-re/syscall_resume.log`, `syscall_fixed.log`.

### FORK-2 CHARACTERIZED (2026-07-09): the 0xdad2 wall is NOT a missing opcode -- it is a BAD Callx4 TARGET in the syscall dispatch (oracle-confirmed)

Chased the new frontier and the "unmodeled opcode" dissolved. Ground truth from the Ghidra Xtensa
oracle (`ghidra-projects/npu-fw/analysis-xtensa/disasm.txt`; mapping Ghidra = our_VMA + `0x08ad2f5c`,
anchored on the dispatcher's `l8ui a5,a4,0x1b` = our `0xd808` = Ghidra `0x08ae0764`):

- The syscall handler's service path (`0x2a88`) does `L32r a4,[lit]; ...; Callx4 a4`, and `a4`
  resolves to **`0xdac4`** (`m2c_probe_exec_trace`, register dump at the call). It jumps there.
- **`0xdac4` is mid-instruction.** The Ghidra true-file decode of this region (our `0xdad0` =
  Ghidra `0x08ae0a2c`) is a clean run of tiny syscall-setter functions, each a real `Entry`:
  `FUN@0xdad0 = entry a1,0x20; l32r a3,=0x11868; s32i.n a2,a3,0x0; retw.n` (store a2 -> scheduler
  global `0x11868`); siblings at `0xdadc`/`0xdae8`/`0xdafc` store to `0x1186c+0x10` etc. These are
  the SCHED2/TCB-field setters -- exactly the "resume/re-queue" primitives init's yield needs.
- Our FETCHED BYTES are CORRECT (they match Ghidra byte-for-byte: our `l32i` at `0xdac5` = Ghidra
  `0x08ae0a21` `l32i a2,a3,0xa8`, etc.). Executing from the bad entry `0xdac4` drifts the decode 2
  bytes into the `Entry`/`l32r` boundary at `0xdad0`, and the straddle `[00 31 98]` is the bogus
  `0x00983100` "opcode." So the wall is a MISALIGNED ENTRY, not a decoder gap -- no opcode to add.
- The real target is a proper `Entry` ~12 bytes ahead (`0xdad0`); `a4=0xdac4` is one setter-slot
  early. The setters sit ~0xc apart, so this smells like an off-by-one/-0xc in the syscall
  function-pointer resolution (the `L32r a4,[lit]` literal, or a table index), plausibly tied to the
  low-VMA load-offset/Harvard scheme that already needed the iter16/iter17 `+0x100` fetch overlays.

**NEXT (Fork-2 root):** pin why `a4` = `0xdac4` and not the real `Entry` -- read the handler's
function-pointer table / the `L32r` literal's true value (find the handler `0x2958`'s segment mapping
in the Ghidra oracle; it is NOT the dispatcher's `0x08ad2f5c` segment) and compare to our resolved
value. A focused literal/pointer-resolution bug in the exception path, analogous to the EXCSAVE find --
NOT a decoder chore. Probe: `m2c_probe_exec_trace` (`WARMUP=47535`) for the register dump at the call;
`m2c_probe_disasm_range` for the byte-level alignment cross-check against Ghidra.

### FORK-2 RESOLVED (2026-07-09, -> commit `3a91e1e9`): the wall was a THIRD +0x100 fetch-overlay seam, not a bad target

**Both prior Fork-2 characterizations above are WRONG** (as was the iter18 "unmodeled opcode
`0x00983100`"). `a4` = `0xdac4` is the CORRECT, never-patched pointer. The wall was OUR FETCH OFFSET:
the section containing `0xdac4` is a THIRD piecewise-relocated `+0x100` block (same species as the
iter16 `LOW_TEXT_BLOCK` `[0x581c,0x5d30)` and the iter17 window-vector table `[0x800,0x980)`), which
we were fetching at the base `+0x5c` -- mid-instruction garbage. The `0xdad2 0x00983100` "opcode" was
just where the mis-fetched byte stream desynced.

**How it was proven (all in the raw image + our decoder, no Ghidra needed):**
- The exception handler `0x2958` is fetched correctly at `+0x5c` (handler bytes match the file at
  `+0x5c`); it loads `a4` from pool literal `[0x2940]` = `0xdac4` via a PC-relative `L32r` whose target
  and value are both correct; a `m2c_probe_addr_store_watch` on the pool slots showed ZERO writes in
  the whole boot -> `0xdac4` is static, not runtime-relocated.
- The syscall is REAL: `m2c_probe_trace_to_wall` (`XDNA_FW_STOP_PC=0x2958`) caught init executing a
  genuine `Syscall` at `pc=0x08b043e1` (sets `UR231=0x12130` as the arg, `a2=0` as the number), so
  `EXCCAUSE=1` is legit and the `Beqi a10,1,0x2a88` syscall branch is the right path.
- Offset test: every pool code-pointer in this block -- `0xdac4` (syscall callee), `0xd900` (ISR),
  `0xd9f0` (sched-fn) -- decodes as a clean `entry a1,X` prologue at `+0x100` (file `VMA+0x100`) and as
  mid-instruction at `+0x5c`. Clincher: pool value `0xd9f0` -> file `0xdaf0`, the EXACT file offset
  Ghidra labels `FUN_08ae09f0`, differing by `0xa4` = `0x100-0x5c` (the pool literals are ground truth;
  Ghidra's `+0x5c` labels are off by the overlay delta because Ghidra doesn't model the overlay either).
- A `+0x100` length-walk of the block decodes as a continuous run of clean `entry`/`retw.n` functions.

**The fix (commit `3a91e1e9`):** `add_rom_overlay(SYSCALL_BLOCK_LO=0xd8a7, HI=0xde04, 0x100)` in
`FirmwareProcessor::load_m2c`. Bounds by walk-and-stub against the `+0x5c` anchors that bracket the
block: `wake_tasks_by_event_mask` (reachable `+0x5c`) ends at `0xd8a5` (LO must be `>= 0xd8a7` so its
final 2-byte `retw.n` at `0xd8a5` is not split -- an `LO=0xd8a6` first try DID split it, caught by the
gold-disasm gate), and `FUN_0000dea0` (reachable `+0x5c`) resumes at `0xdea8`. The only "code" the
`+0x5c` descent found between them was the mislabeled `FUN_0000dbc4` seed -- really this section's
`0xdac4`+`0x100`.

**Verified effect:** boot advances PAST `0xdad2` (the multi-session wall). The syscall handler runs its
syscall-number jump table (compares `a5` against `107/112/102/99`) and walls at a NEW frontier
`0x44a34` (`n~47562`), reached via `Call0` from `0x2630` -- itself the head of yet ANOTHER `+0x100`
section (`entry a1,0x20` at `+0x100`), so `0x44a34` is an out-of-image fetch (`word=0`). `cargo test
--lib` = 4087 pass (the two frontier-guard boot tests updated to `0x44a34`).

**The generalized situation (the FIXME(iter16) "reconstruct every seam"):** the low `.text` has
SEVERAL scattered `+0x100` sections, revealed one call-hop at a time as boot walks into each. An
automated coherence classifier (`+0x5c` vs `+0x100` framing score) is UNRELIABLE -- dense Xtensa
decodes plausibly at both offsets, so it false-flags known-good `+0x5c` regions (the dispatcher, the
executed `0x8xxx` code). The robust method is boot-driven walk-and-stub: let boot wall, identify the
section (its pool pointers are clean `entry` at `+0x100`), overlay it, confirm boot advances. Only the
sections ON THE BOOT PATH matter (four mapped: window vectors, `LOW_TEXT_BLOCK`, this syscall block,
and next `0x2630`). **NEXT: map the `0x2630` section, repeat.**

### DUAL-MAPPING RESOLVED + LITERAL-POOL FIX (2026-07-09, iter20): the `0x2630` chain runs; `+0x100` is a load-layout property, not a runtime one

The `0x2630` seam broke, and along the way a wrong turn was corrected and the whole `+0x100` mechanism
was characterized to ground truth. Sequence (all verified by coherent execution, the strongest oracle):

**A false-start worth recording (the framing trap).** First read of `0x2630` concluded it was a
*misframe artifact* -- that the callee's true VMA was `0x26d4` (`+0x5c`, already fetched), because the
callee's internal `Call8` (encoded offset `0x9E20`) reaches the IPC primitive `0xc530` only if the
function is based at `0x26d4`. An adversarial Opus reviewer CONFIRMED this. **Both were wrong, sharing
one framing bug:** they evaluated the callee's `+0x100` call target `0xc48c` at `+0x5c` (file `0xc4e8` =
garbage) and declared it a non-entry. Read at the callee's own `+0x100` framing, `0xc48c` -> file
`0xc58c` = a clean `entry`. The trace settles it directly: the handler `0xdac4` is byte-provably
`+0x100`-only (`+0x5c` = file `0xdb20` = `f0 22 23`, not an entry), it is entered via the ABSOLUTE
pointer `0xdac4` (from pool `[0x2940]`, delta-independent), and it flows to a byte-verified `Call8` at
`0xdd2d` (`+0x100`-only; `+0x5c` = `4d 0a`, a narrow op) whose target is `0x2630`. So `0x2630` is a
REAL `+0x100` callee, not an artifact. Lesson: when a section is `+0x100`, evaluate its call/literal
targets at `+0x100` too -- mixing framings manufactures false discriminators.

**The chain, once served (`XDNA_FW_TEST_DUALMAP` -> permanent iter20 overlays).** `0x2630` runs a
context-switch routine (`Rsil 2`, bit-manip, arg setup) and `Call8`s the IPC critical-section primitive
`0xc48c`, which builds a message struct and posts to the `[0xfae0]` mailbox, then jumps into Seg-B at
`0x08b0e710`, returns, and runs the exception-frame restore (`0x2958` + `0xb1c`). Boot advances 122
instructions past the old wall to a NEW frontier `0xe1fc` (another `+0x100` seam: code at file+0x100,
zeros at +0x5c).

**Literal-pool fix (permanent, `mem.rs` + `mmio.rs`).** The `+0x100` window covers L32r literal pools,
not just instruction fetch. `l32r_load` read pools via `inst_load32` (base `+0x5c`), so `0xc48c`'s
literal at VMA `0x3424` came back `0x08a8000c` (garbage, `+0x5c` file `0x3480`) instead of `0x08b0e710`
(Seg-B, `+0x100` file `0x3524`); `Callx8` walled. New `Bus::inst_load32_overlay(vaddr, paddr)` honors
the overlays by vaddr; `l32r` routes through it. With the callee/primitive code AND their pools
(`0x254c`, `0x3424`, `0x3c74`) overlaid, the primitive reads `a10=0xfae0` and `a8=0x08b0e710` correctly.
Suite stays green (4087) -- no `+0x5c` code shares those pools.

**Two structural facts nailed (they reshape the whole model):**
- **No firmware relocation.** A full-boot store-watch on every `+0x100` region head (`0x2630`, `0xc48c`,
  `0x581c`, `0xd8a7`, `0x800`, `0x900`) recorded ZERO stores. The firmware CPU does not copy itself; the
  on-chip PSP does (`PSP_START_COPY_FW`, `aie2_psp.c`), opaquely, per a `$PS1` segment table we cannot
  read (the container header is hashes + `"Release 1.5.5.391"`, no load descriptors). So there is no
  startup relocation loop/table to derive the layout from -- the PSP's real segment map is inaccessible.
- **No dual-execution.** `0xc530` (the `+0x5c` alias of the primitive) is NEVER executed in a full boot
  (default boot walls at `0x44a34` at `n=47562` without reaching it). Each function has ONE canonical
  VMA fixed by its section's file offset; cross-section transfers go through absolute pointers,
  intra-section calls are PC-relative (same offset). The finding's older "`0xc530` builder 25x" reflected
  a different overlay condition; the primitive's canonical VMA is `0xc48c` (`+0x100`).

**Method going forward (locked in): execution-verified walk-and-stub.** Since the PSP segment table is
unreachable and no static coherence classifier is trustworthy (dense Xtensa aliases both framings),
letting the real firmware execute and walling is the ground-truth oracle. Map each boot-path `+0x100`
section as boot reaches it, verify by coherent execution, overlay, repeat.

**Boot now reaches a STEADY LOOP -- not yet proven idle (2026-07-09, iter20 cont.).** The next seam was
the exception-frame RESTORE routine at `0xe1fc` (file `0xe2fc`; `+0x5c` is all zeros, so unambiguously
`+0x100`), `Jx`-ed to from the syscall-return path. Its scattered `+0x100` literal pools (`0xe0e0`,
`0x31dc`, `0x3cc0` -> values `0xe108`/`0x2278`/`0xd900`, all garbage at `+0x5c`) needed serving too; once
they were, its `Callx4` resolves to the real ISR `0xd900` (already in `SYSCALL_BLOCK`). With that
overlaid the boot **no longer walls anywhere in a 200k-instr budget** -- it advances into a steady loop
inside the exception handler (`FUN_0000e098`, `last_pc ~0xe297`) and spins there. `unknown_op=None`,
`reached_idle=false`, `window_exceptions=0`, `unresolved_spin=None`. Suite 4087 pass; the two
frontier-guard boot tests now assert the mechanical facts (no wall, ran the budget, spinning in
`[0xe098,0xe340)`) and make NO idle claim.

**LOOP CHARACTERIZED -> it is a LIVELOCK, root-caused to a page-table modeling gap (2026-07-09, iter21).**
Answered both of Maya's questions decisively:

- **Idle vs livelock: LIVELOCK.** The loop has an exact **62-instruction period** and makes **zero forward
  progress**. Traced one full cycle: a load at `0xe2a9` (`L32iN a7 = [0x2278]`, the scheduler's current-task
  pointer) raises an exception; the next PC is the vector `0xb1c` (the load never completes). The vector
  prologue reads `EXCCAUSE` (`Rsr sr232` at `0xb2a`) = **`0x1c` = 28 (LOAD_PROHIBITED)**, branches to the
  save block at `0xe1fc`, saves a full exception frame into the **fixed** struct `@0xe108`, returns to
  `0xe2a9`, and re-faults -- forever. The store-watch on the frame slots shows every field written
  **byte-identical 2456 times** from the 2nd period on (`[0xe148]=0`, `[0xe150]=1`, `[0xe154]=0x590349d0`,
  `[0xe158]=0xe2a9` = the faulting PC, `[0xe15c]=0`). The loop even **feeds itself its own constant**:
  `0xe29d Wsr sr230 <- 0x590349d0`, read back into the frame next period. `a5=0xdeadbeef` (poison) sits in
  a register throughout. Not idle -- a fixed-point fault loop.

- **Leak: NONE.** `SP (a1) = 0x30d0` invariant, register window (`wb=8 ws=0x016b`) invariant, all frame
  stores hit the single fixed struct `@0xe108` (overwrite, not growth). Bounded footprint -- as expected
  for a fixed-point loop. (Per Maya's "any leak implies broken": there is no leak, but the clean-footprint
  verdict is not a clean bill of health -- the "broken" signal is the livelock itself.)

**Root cause (probe `m2c_probe_mmu_at_fault`): autorefill reads a POISON page-table entry.** At the fault
the DTLB lookup of `0x2278` returns `HIT way=0 ei=2 vaddr=0x2000 paddr=0xdeadb000 attr=15 access=0x0`.
`PTEVADDR=0x3c000000`; the PTE address `(0x3c000000 | (0x2278>>10)) & ~3 = 0x3c000008` reads **`0xdeadbeef`**
(our page-table poison fill), decoded as `paddr=0xdeadb000, attr=0xf=15` (no access) -> cause 28. That
paged way-0 entry **shadows** the still-live low-window identity region (`dtlb[6][0]` vaddr 0, attr 3),
which is why the load fails now but succeeded pre-context-switch (the passing
`low_window_dram_is_translation_covered_from_reset` samples at `pc==0x2630`, before the switch).

The gap is in the **synthesized PSP page table** (`src/firmware/psp_map.rs::install`): it populates PTEs
for the code region (`0x20000000+`) and the mailbox aperture (`0x27000000+`) but **NOT the low DRAM
window** (`0x0..0x1fffffff`, scheduler/data). Everything else in the page table stays `0xdeadbeef` poison.
When autorefill walks for a low-window VPN it installs garbage. This is the **exact same failure mode the
code already handles for the mailbox aperture** (`psp_map.rs:41-53` comment: firmware invalidates the
region entry, accesses "fall to the autorefill walk and need a writable PTE here or they fault") -- the low
window has the same need but was never mapped.

**Recommended fix (design fork, awaiting Maya):** extend `psp_map::install` to also lay identity PTEs for
the low DRAM window (attr 3/RW, as the mailbox aperture does), so autorefill of the low window returns a
valid identity mapping instead of poison. This is consistent with the sanctioned "reconstruct the PSP
table's observed effect by coherence" approach and would let `[0x2278]` load succeed and break the
livelock, advancing boot to the next frontier. Open sub-question to settle alongside the fix: *why does
autorefill fire for the low window at all* when the region identity `dtlb[6][0]` is live -- i.e. is the
region entry transiently invalidated by the context switch (mailbox precedent) or is there a lookup-order
subtlety. **NEXT: decide the fix with Maya, then implement + validate that the livelock breaks.**

## Probes used

`m2c_probe_current_task_timeline` (2026-07-09: cur-task 0x10f10@n41464 -> 0x9040@n58754, 2 transitions),
`m2c_probe_addr_store_watch` (2026-07-09, S32e-aware: [0x2278] writes = 0x10f10 by FUN_00004570, then 0x9040 by the 0x895 spill),
`m2c_probe_waypoint_hits` (2026-07-09: all 5 state[+27] setters + FUN_00004b00 + FUN_00002730 NEVER reached in boot),
`m2c_probe_addr_store_watch` (2026-07-09: zero stores to 0x22bc in 1.5M instrs),
`m2c_probe_store_value_watch` (2026-07-09: 0x588c stored once to 0x11890, the run-fn ptr, not to 0x22bc),
`m2c_probe_isr_observe` (2026-07-09: forced ONE faithful level-1 delivery at steady state;
ISR runs clean via the interp, chain 0x2958->0xd900->0xc530->Seg-B->tick->dispatcher, NO MMIO,
no event posted absent a completion source),
`m2c_probe_peek` (2026-07-09: resolved FUN_00008c68 bases -- col struct 0xf9a0/+64=0xf9e0,
aperture 0x27271000, store 0x27271114), `m2c_probe_literal_xref` (2026-07-09: 9 referencers of
the 0xf9a0 column struct; FUN_00008c14 = the bit-3 setter), `m2c_probe_disasm_range`
(2026-07-09: FUN_00008c14 bit-3 set path 0x8c59), `m2c_probe_exec_trace` (2026-07-09: one full
outer cycle @ warmup 305k -- dispatcher->d828->go-alive 0x588c->7fa0->8c68, gates [0x9040+27]/[+48]=0),
`m2c_probe_intenable_watch` (2026-07-09: line 0 armed at instr 2218/pc 0x88d5, INTLEVEL
pinned at 2 thereafter, no `waiti` in 1M instrs), `m2c_probe_steady_histogram` (2026-07-09:
2M-sample spin = `sched_ready_popcount`+`FUN_0000c96c`+`FUN_00008c68`, idle `FUN_0000c8e0`
absent, ZERO external MMIO reads),
`m2c_probe_slot_sufficiency` (2026-07-09: relocating a state=1 worker into the scanned band does
NOT advance boot -- readiness needs a nonzero await-mask `[+0x38]`; popcount ORs `[+0x38]` over
state==1 slots), `m2c_probe_registry_access` (2026-07-09: create-registry + go-alive record are
write-only after staging -- admit structurally absent), `m2c_probe_waypoint_hits` (picker `0xc980`
+ both callers +
`FUN_000041b8`/`FUN_0000dbc4` all NEVER reached; linker `0xd4e0` runs x9),
`m2c_probe_segb_startcall` (2026-07-09: also `XDNA_FW_TRACE_FROM/TO`/`WARMUP` -- traced the link
primitive `0xd4e0` = priority-indexed slot write `[SCHED+56+prio*4]`; and `task_create`
`0xd664` = stage-not-link + the preempt gate `0xd7b9`; and the two post-create calls -- Seg-B code
is real; call#1 = SRAM notify `0x3010d7c`; call#2 = `Syscall` context-switch to the `0x3170`
scheduler; init never resumed; schedulable band 0-5 always empty),

`m2c_probe_taskstart_calls` (2026-07-09: the two post-`task_create` calls at `0x3df1`/`0x3df9`
execute but call into Segment-B and never link go-alive into the runnable array),
`m2c_probe_goalive_spin` (H1: `[0xfae0]` is fire-and-forget, RMW-only, no response read),
`m2c_probe_addr_store_watch` (H2: `[0x11890]` written once = `0x588c`, never repointed),
`m2c_probe_event_source` (#1: event ISR/`0x27010d28` never reached in natural boot),
`m2c_probe_apertureset_branch` (the never-taken bit0=1 body = a column power-up doorbell+ack
handshake, dormant at idle),
`m2c_probe_addr_store_watch` (`XDNA_FW_WATCH_ADDR=0x22bc,0x10f40`),
`m2c_probe_store_search` (`XDNA_FW_STORE_DISP=0x6c`),
`m2c_probe_col_cmd_trace`, `m2c_probe_disasm_range`,
`m2c_probe_colassign_boot` (tail histogram), `m2c_probe_breach_doorbell`
(+bit3 shim), `m2c_probe_literal_xref` (`XDNA_FW_LIT_LO/HI` over device-SRAM +
code-pointer ranges), `m2c_probe_call_xref` (`XDNA_FW_XREF`, publisher/run-fn
caller chains), `m2c_probe_disasm_range`, `m2c_probe_goalive_lifecycle` (new:
flag+column breach + first-hit of the go-alive chain), `m2c_probe_mailbox_receive`
(canonical `boot_to_idle`), `m2c_probe_boot_with_array` (STUB vs ATTACHED,
zero array accesses), `m2c_probe_inject_interrupt` (INTLEVEL-2 masks the IRQ; 2026-07-09: steady-state
`INTENABLE=0x1` = one armed line (line 0); across 400k steps `min_intlevel=2`,
zero level-0 windows -> the dispatcher busy-spins popcount and never opens a
deliverable window),
`m2c_probe_trace_to_wall` (steady-loop tail), `m2c_probe_external_complete` (new:
faithful once-at-idle completion -- record intact, boot un-advanced),
`m2c_probe_disasm_range`/`m2c_probe_call_xref` (2026-07-09: picker `0xc980` =
windowed `Rsil 2` runnable-array dispatcher; callers `0x42c8`/`0xdd7a` in
table-only `FUN_000041b8`/`FUN_0000dbc4`; `wake_tasks` `0xd84c` <- only go-alive
`0x5809`; idle-wait `FUN_0000c8e0` = `waiti 0`; no `CCOMPARE`/`CCOUNT`, no `Rfi`
-> line 0 is level-1, timer + high-level-IRQ both falsified; raw-scan: scheduler
core is computed-`callx`/vtable, not an address table).
