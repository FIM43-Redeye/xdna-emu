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

## Probes used

`m2c_probe_addr_store_watch` (`XDNA_FW_WATCH_ADDR=0x22bc,0x10f40`),
`m2c_probe_store_search` (`XDNA_FW_STORE_DISP=0x6c`),
`m2c_probe_col_cmd_trace`, `m2c_probe_disasm_range`,
`m2c_probe_colassign_boot` (tail histogram), `m2c_probe_breach_doorbell`
(+bit3 shim), `m2c_probe_literal_xref` (`XDNA_FW_LIT_LO/HI` over device-SRAM +
code-pointer ranges), `m2c_probe_call_xref` (`XDNA_FW_XREF`, publisher/run-fn
caller chains), `m2c_probe_disasm_range`, `m2c_probe_goalive_lifecycle` (new:
flag+column breach + first-hit of the go-alive chain), `m2c_probe_mailbox_receive`
(canonical `boot_to_idle`), `m2c_probe_boot_with_array` (STUB vs ATTACHED,
zero array accesses), `m2c_probe_inject_interrupt` (INTLEVEL-2 masks the IRQ),
`m2c_probe_trace_to_wall` (steady-loop tail), `m2c_probe_external_complete` (new:
faithful once-at-idle completion -- record intact, boot un-advanced).
