# iter18: completion-causality RE -- shape (i) vs shape (ii) reopened

> Branch `feat/m2c-mapping-boot-to-idle`, #140. Follows
> `2026-07-06-iter18-phase0-interrupt-wiring.md`. Deep RE requested by Maya:
> match completions to specific tasks/requests rather than complete
> indiscriminately. This doc records what the RE found and the model reframe it
> forces. NOT yet resolved -- ends with the decisive experiment to run.

## The question

The boot `task_dispatcher` (`0xd7f0`) recursion blocks on task done-flags
`[task+0x30]` that nothing in our runs sets. What real event sets each, and
which task does each completion target?

## Firmware-side findings (probes in `mod boot_tests`, XDNA_FW_PROBE-gated)

**`m2c_probe_current_task_timeline`** -- two tasks block, in order:

```
post (i2x tail 0x27200170 -> 0xf18)   instr ~6973
[0x2278] (current-task) <- 0x10f10    instr  41464   (task A; done-flag 0x10f40)
first done-flag check @0xd828 a4=0x10f10  instr 47896 (flag 0)
[0x2278] <- 0x9040                     instr  58754   (task B; done-flag 0x9070)
```

**`m2c_probe_task_runfns`** -- both tasks are dispatched through the SAME two
shared routines, not per-task pollers:

```
task=0x9040/0x10f10  run_fn=0xc938  call_pc=0xd836
task=0x9040/0x10f10  run_fn=0x588c  call_pc=0xd842
```

The run-fns access ONLY local memory (scheduler table `0x2278/0x2288..`, a
strided task/timer array `0x2b9c + k*0x90`, the task done-flags, ...). No
mailbox/register/DMA poll. So the per-task wait condition lives in the task
struct; the shared dispatcher reads it; completion is an external local write.

**`m2c_probe_completion_binding`** -- the done-flag addresses are NEVER stored
as values. Across 1.5M instrs the only stores of a tracked pointer were:

```
[0x2278] <- 0x10f10   (scheduler sets current-task; expected)
[0xfaf0] <- 0x9040    (task base into a table)
[0x12060] <- 0x10f10  (task base into a table)
```

So `0x9070`/`0x10f40` are never registered anywhere: the completion target
`+0x30` is IMPLICIT in the task-struct layout, not bound per-request in a
descriptor. An agent that writes it must already know the task layout.

## Host-side findings (xdna-driver Explore, `src/driver/amdxdna/`)

- **No host->fw doorbell in aie2.** The ONLY host->fw signal for a new request
  is the **x2i tail-pointer register write** (`mailbox_send_msg`,
  `amdxdna_mailbox.c:310-312`; `mailbox_set_tailptr` -> `writel` into the MBOX
  BAR at the offset the fw published in `mgmt_mbox_chann_info.x2i_tail`).
- Boot config sequence the host SENDS over x2i (all blocking request/response
  via `aie2_send_mgmt_msg_wait`), in order (`aie2_mgmt_fw_init`/`_query`,
  `aie2_pci.c`): `SET_RUNTIME_CONFIG(0x10A)`xN -> `ASSIGN_MGMT_PASID(0x103)` ->
  [`UPDATE_PROPERTY`] -> `SUSPEND(0x101)` -> `RESUME(0x102)` ->
  [`CALIBRATE_TIME`] -> `GET_FIRMWARE_VERSION(0x108)` -> `QUERY_AIE_VERSION` ->
  `QUERY_AIE_TILE_INFO` -> `REGISTER_ASYNC_EVENT(0x10C)`xcols -> `GET_FW_VER`
  (barrier). (`GET_PROTOCOL_VERSION`/`GET_RUNTIME_CONFIG` are NOT sent at boot;
  protocol version comes from the alive struct.)
- Alive handshake: host polls `FW_ALIVE_OFF` non-zero, reads
  `mgmt_mbox_chann_info`, then **zeros `FW_ALIVE_OFF`** (`aie2_pci.c:187`) -- the
  only host->fw write in the alive step.
- **Driver engineer's conclusion:** if the fw tasks block on distinct local
  flags, those flags "must be set by the firmware's own message-dispatch loop
  after it decodes each opcode from the x2i ring." The driver has no
  host-visible per-flag write beyond the ordered opcodes + tail-pointer writes.

## The reframe: shape (i) is reopened

Earlier (phase-0 findings) we concluded **shape (ii)**: an external hardware
agent writes `[task+0x30]`; force-ack ruled out register/interrupt handshakes;
the store-search found no fw write to `+0x30`. BUT:

- force-ack only wrote the **i2x** (fw->host) registers + the doorbell. It never
  **delivered an x2i (host->fw) message** (a real config request into the x2i
  ring + an x2i tail advance).
- The store-search found no `+0x30` write because **the fw's x2i-RX handler
  never ran** -- we never delivered a host message for it to process.

So the store-search's "no fw write" is consistent with **shape (i)**: the fw's
OWN x2i-message handler sets the done-flags, and it simply never executes
because we never play the host's side of the conversation. The boot wedges
because the host->fw config dialogue never happens, not because a magic agent
write is missing.

This inverts the model. If shape (i) holds, the faithful thing to build is
**host x2i message delivery** (the emulator, as the host, sends the real config
sequence over the x2i ring), and the firmware completes its own tasks. The
`CompletionAgent` (an agent writing `[task+0x30]`) built in Tasks 1-3 would be
the WRONG shape -- reworked, not extended.

## Open tension / caveats

- 2 tasks vs many config messages: the 2 blocked tasks are not obviously 1:1
  with the ~10 config opcodes. The tasks may be early init/timer tasks that
  gate on the first exchange(s); more tasks may appear once earlier ones clear.
- The interrupt/poll question (unresolved): the recursion polls only local
  memory (no x2i poll), so for the fw to process an x2i message it must take a
  HW interrupt from the mailbox IP on the x2i tail write. INTLEVEL is locked at
  2 in the scheduler critical section (masks level-1). Whether the x2i interrupt
  is deliverable here is the same chicken/egg the phase-0 doc raised -- but now
  with a concrete new thing to test.

## UPDATE: the boot never reaches the alive handshake -- the wall is PRE-mailbox

The x2i experiment prep (`m2c_probe_alive_struct`) tried to locate the
`mgmt_mbox_chann_info` struct by catching the fw's store of its magic
`0x55504e5f` ("_NPU"). **The magic store never executes in 1.5M instrs.** The
firmware never publishes the alive struct -- it wedges in the early-task
recursion (tasks block at ~41k/59k) LONG before the mgmt mailbox is brought up.

This redirects the model materially:

- The two blocked tasks (`0x10f10`, `0x9040`) are **pre-mailbox init tasks**.
  They cannot be waiting on the host config sequence (`SET_RUNTIME_CONFIG`, etc.)
  -- that whole dialogue happens AFTER the alive handshake, which we never reach.
- So **shape (i) (fw processes host x2i config messages) is a LATER-stage
  concern**, not the current wall. The x2i-delivery experiment is premature: it
  would exercise a stage the boot hasn't gotten to.
- The early-task completion is therefore either shape (ii) (a low-level init /
  DMA / hardware agent whose completion the tasks wait on) or an internal
  cooperative signal our emulation mis-handles. force-done unblocking these
  tasks -> boot proceeds to ~575k (the atomic helper) confirms completing them
  is what advances boot.
- The i2x post at ~6972 (tail 0xf18 + payload ptr 0x08a00ff0) is thus an EARLY
  op (a boot-status/log post, or an early request), NOT the mgmt alive handshake.

**Revised next step:** RE what the two EARLY tasks actually wait on -- trace each
task's woken continuation after force-done sets its flag (what the completion
enables reveals what it was waiting for), and/or trace what operation each task
initiated before blocking. Match each early-task completion to its real trigger
before choosing a model. The host-side x2i findings remain valid but apply to the
post-alive stage.

## RESOLUTION: shape (i) -- a firmware EVENT DISPATCHER wakes the tasks

Static disassembly + xref cracked the wake mechanism, overturning the phase-0
"shape (ii)" conclusion.

- **`FUN_0000d84c`** (named a shape-(i) candidate in phase-0, now confirmed) is
  the firmware's **event-dispatch / wake** function. Signature: called with an
  event mask in `a2`. It walks 9 registered entries at `[base+0x38]` and, for
  each entry whose `[entry+0x38]` mask intersects `a2` (`bnone a2,[entry+0x38]`):
  sets the state byte `[entry+0x2c]=6`, clears the matched bit(s) from
  `[entry+0x30]` (`and [entry+0x30], ~a2`), and calls the scheduler helper
  `0xc938`. So `[task+0x30]` is a **pending-event mask**; an event CLEARS its
  bit; when satisfied the task is marked runnable (state 6). This is the fw
  setting its own task flags -- NOT an external agent.
- **Caller:** the static xref shows exactly one direct caller of `FUN_0000d84c`:
  `FUN_00005580+0x289` (`0x5809`). And **`FUN_00005580` IS the scheduler core**:
  the dispatcher's run-fn `0x588c` is `FUN_00005580+0x30c` (a mid-function
  entry). `FUN_00005580` has no static callers -> it is entered indirectly
  (function-pointer / vector). Around `+0x270..+0x289` it reads state (e.g.
  `l8ui a4,[a13+34]`) and branches before calling the event dispatcher with the
  derived mask.

**So the faithful completion is shape (i):** the scheduler core reads an
**event source**, derives an event mask, and calls `FUN_0000d84c` to wake the
tasks whose pending-event bits the event clears. The boot wedges because the
**event source produces no event** in our emulation -- not because a magic
external write to `[task+0x30]` is missing. force-done "worked" only by brute-
forcing the flag past this whole mechanism.

Why phase-0 mis-concluded shape (ii): it tested force-event (seeding status-page
bits at `0x2727n000`) which failed, and force-ack (i2x/doorbell registers) which
were inert -- but it never identified `FUN_00005580`'s actual event source, and
it did not yet know the wall is pre-mailbox. The 63 static `+0x30` writers it
found were real (this dispatcher among them), not "coincidental."

### The remaining link (next): the EVENT SOURCE

Find what feeds `FUN_00005580`'s event mask -- what state at/around `[a13+0x22]`
(and the literals it reads) is the event, and what sets it. Candidates: a
hardware interrupt status the scheduler polls, a timer tick, or an early-init
completion register. That source, modeled faithfully, is the completion: deliver
the event -> the fw's own dispatcher wakes the right task. This is the per-task
match Maya asked for -- each event maps to the tasks whose `[entry+0x38]` mask it
intersects.

## UPDATE 2026-07-06 (session 2): EVENT SOURCE RESOLVED = register 0x27010d28

The event-source hunt landed. New probes (`m2c_probe_peek`, `m2c_probe_event_source`,
`m2c_probe_word_scan`, `m2c_probe_store_value_watch` in `mod boot_tests`) plus
static disasm of the actual function pinned it. Several earlier framings were
corrected in the process.

### The event source (the answer)

Disassembling the generic-event decode path (label `0x5580`, interior to a
larger function -- see below), instructions `+0x260..+0x289`:

```
+0x263  a2 = a6 << 4
+0x266  a2 = a9 + a2                 ; a9 = *lit(0x3310) = 0xf1a0, event-descriptor table (stride 16)
+0x269  a7 = *(a2+13)               ; per-event "already-pending" byte
+0x26c  a14 = *lit(0x3364) = 0x27010d28   ; <-- EVENT-SOURCE POINTER
+0x272  if a7 != 0 skip              ; already pending
+0x275  *(a2+13) = 1                 ; mark pending
+0x278  a2 = *(a14+0)               ; <-- READ THE EVENT SOURCE  [0x27010d28]
+0x27b  a15 = *lit(0x3368) = 0x53494d4e   ; sentinel ("no event")
+0x280  if [0x27010d28] == 0x53494d4e skip ; source says nothing -> no wake
+0x283  SAR = shift by a6
+0x286  a10 = 1 << a6               ; event mask
+0x289  Call8 FUN_0000d84c(mask)     ; wake tasks whose [entry+0x38] mask intersects
```

Literals resolved by `m2c_probe_peek`:
- `[0x3364] = 0x27010d28` -- the event-source register pointer.
- `[0x3368] = 0x53494d4e` -- the "no-event" sentinel.
- `[0x3310] = 0x0000f1a0` -- event-descriptor table base (sane RAM neighbor of the task structs; cross-checks the literal addressing).

**Register identity (Explore of xdna-driver/aie-rt/RyzenAI-SW):** FW `0x27010d28`
== driver `0x3010d28` == `MPNPU_APERTURE0_BASE (0x3000000) + 0x10d28`. Base
mapping confirmed via PWAITMODE (FW `0x27010034` == driver `0x3010034`,
`npu1_regs.c:11`). APERTURE0 is the MP_NPU "public" register block (BAR0/REG,
doubles as PSP/SMU BAR). **No named register at `0x10d28`** -- the documented
APERTURE0 registers stop at `0x100BC` (PUB_SCRATCH9). The neighbourhood is
PWAITMODE / PSP-interrupt / SMU-interrupt / scratch -- i.e. **early
hardware-init / PSP / SMU / power territory, NOT the host mailbox** (the mailbox
ring lives in APERTURE2 `0x30C0000`, located dynamically post-alive). So the two
blocked tasks are **pre-mailbox early-init tasks waiting on early hardware
events** surfaced through this APERTURE0 event register.

**Sentinel `0x53494d4e` ("NMIS"/"SIMN"):** found NOWHERE in driver/aie-rt/
RyzenAI-SW -- a firmware-internal constant. The tempting "SIM = aiesimulator
detection" reading is **dead**: verified from the aietools install that
aiesimulator/aiesimmsm/mesimulator/iss model the AIE **array/ME compute cores**,
with NO Xtensa mgmt-core model (only a stray GCC `xtensa-config.h` and an LLVM
reloc `.def`). aiesimulator never executes this firmware, so the firmware can't
be detecting it. Do not anchor on simulator-detection.

### Mechanism refined -- and two earlier claims corrected

- **`0x5580` is NOT a function** (Ghidra mislabel). It is an interior label of a
  larger windowed function whose real entry is **`0x5524`** (`entry a1,80`).
  That function is the scheduler **event-poll / idle run-fn**: big switch on
  event index `a6=0..18`, contains `waiti 0` (`0x56e6`), and reaches the
  event-source read via internal jumps (`j 0x5581` at `0x5556`).
- **It is NOT interrupt-vectored.** It uses `entry`/`retw` (a normal windowed
  call), has no direct external callers, and is reached only via the
  dispatcher's indirect run-fn pointer (`callx8 [sched+36]` at
  `task_dispatcher+0x1a` = `0xd842`) -- the SAME mechanism that runs `0x588c`
  and `0xc938`. There is no interrupt in this path. (The Phase-0
  "interrupt-vectored" hypothesis does not hold for THIS wall.)
- **`task_dispatcher` (`0xd7f0`) is NOT self-recursive.** It is a straight-line
  critical section: `rsil a2,2` (enter, INTLEVEL=2) -> read current task
  `[sched+40]` -> check `[task+0x30]` at `0xd828` (`beqz` gates a `Call8 0xd608`
  = the pending-event handler) -> set task state `[task+0x2c]=6` -> run-fns
  (`0xc938`, indirect `[sched+36]`) -> `wsr PS` (exit) -> `retw`. The "recursion"
  is a HIGHER-level loop re-dispatching the same not-ready task.
- **The waker `FUN_0000d84c` itself calls the scheduler helper `0xc938`**
  (from `0xd881`), confirming this is one cooperative scheduler, not two layers.

### Why boot wedges (root cause, this wall)

`m2c_probe_event_source` (1.5M instrs): the event-poll block `0x5580` is **never
entered**, `0x27010d28` is **never read**, PS.INTLEVEL=2, last_pc `0xc964`. The
dispatcher only ever runs the blocked tasks' run-fns (`0x588c`/`0xc938`, per
`m2c_probe_task_runfns`); it **never selects the event-poll run-fn `0x5524`**.
So the fw's own event dispatch never gets a turn -> the event source is never
sampled -> no event mask -> the waker never fires -> the tasks never unblock.
force-done "worked" only by brute-forcing `[task+0x30]` past this whole path.

### The design fork (next -- Maya-sequenced BUILD)

Two shapes, to decide before building:
- **(A) Reach the poll.** The event-poll (`0x5524`) is an idle/every-cycle run-fn
  the real scheduler reaches when a task is not ready; our dispatcher instead
  treats the blocked task as ready (`state=6` at `0xd831`) and re-runs it
  forever. If so, the gap is in the readiness/run-fn-selection logic (why the
  blocked task is re-selected instead of yielding to the event-poll), and once
  the poll runs, modeling `0x27010d28` to signal the right early event
  (`!= 0x53494d4e`) wakes the task via `FUN_0000d84c`.
- **(B) External trigger.** A genuine hardware completion (PSP/SMU/power init)
  both makes `0x27010d28` signal AND drives the poll. Less likely given the
  cooperative-scheduler evidence, but not excluded.

Open sub-questions for the build: (1) what SELECTS the event-poll run-fn vs the
blocked task's run-fn (disasm the higher-level dispatch loop + `0xcadc` the
pending-event handler); (2) reconcile `[task+0x30]` semantics -- `0xd828` treats
non-zero as "has pending work" (force-done set it to 1 and progressed), while
`FUN_0000d84c` CLEARS bits from `[entry+0x30]`; confirm whether `task` (dispatcher,
`[sched+40]`) and `entry` (waker, `[base+0x38]` table) are the same struct/offset
or different. Resolve these before committing to a delivery model.

### Session-2b: the mechanism mapped, and the remaining knot

Followed both sub-questions into the scheduler internals. Result: the delivery
side is fully mapped; the injection side is the one knot left.

Scheduler layout (literals resolved via `m2c_probe_peek`):
- `SCHED = *lit(0x3d28) = 0x2250`; current task = `[SCHED+40] = [0x2278]`;
  global pending-event word = `[SCHED+108] = [0x22bc]`; 9-entry waiter table at
  `[SCHED+56] = [0x2288]`.
- `SCHED2 = *lit(0x3d30) = 0x1186c`; the dispatcher's indirect run-fn pointer is
  `[SCHED2+36] = [0x11890]`.

`task_dispatcher` (`0xd7f0`), corrected call targets:
- `0xd82c Call8 0xCADC` (NOT `0xd608` -- earlier arithmetic slip; `51932=0xCADC`).
  Reached only when `[task+0x30] != 0`; arg `a2 = [task+0x30]` (call8 window:
  callee a2 = caller a10, loaded at `0xd828`). `0xCADC` = `deliver_pending_events`.
- `0xd836 Call8 0xC938` (scheduler helper) and `0xd842 Callx8 [0x11890]` (run-fn).

Event delivery (the CONSUMER side, fully mapped):
- `wake_tasks_by_event_mask` (`0xd84c`), arg mask `a2`: walk 9 entries at
  `[SCHED+56]`; for each whose `[entry+0x38]` wait-mask intersects `a2`, set
  `[entry+0x2c]=6` (ready), then CLEAR the matched bits from `[SCHED+108]` and
  from `[entry+0x30]`, call helper `0xc938`, and call the run-fn `[SCHED2+36]`.
- `deliver_pending_events` (`0xCADC`, = `FUN_0000c9dc+0x100`) is the same shape,
  driven by the dispatcher with `a2 = [task+0x30]`.
- So `[task+0x30]` / `[SCHED+108]` are **pending-event bitmasks**; both handlers
  CONSUME (clear) them and mark waiters ready. force-done set `[task+0x30]=1` ->
  `deliver_pending_events(1)` ran -> progress. The dispatcher check reconciles:
  non-zero = "events to deliver", not "still blocked".

The run-fn selection (sub-Q1) -- ANSWERED:
- `[0x11890]` is set to `0x588c` exactly once (instr 41480) by the generic
  registrar `FUN_0000daf0` (`set_runfn(struct, fnptr,...) -> [struct+36]=fnptr`),
  called from init. `sched_event_poll` (`0x5524`) is **never** registered:
  0 runtime stores of `0x5524`, 0 static pointers to it anywhere in the image,
  no direct callers (only its own recursion at `0x5773`). So the HW event
  register `0x27010d28` is read from NOWHERE reachable in this boot.

The remaining knot (the BUILD pivot):
- Two intertwined unknowns: (a) how is `sched_event_poll` (`0x5524`, the only
  reader of `0x27010d28`) supposed to be entered, given it has no pointer
  anywhere? and (b) who SETS the pending-event bit (`[SCHED+108]`/`[task+0x30]`)
  -- neither consumer sets it; a static disp-0x6c search found 14 store sites
  (most in the `c9dc`/`cb38`/`d84c` scheduler family, base not necessarily
  SCHED), unresolved. One of these, or an as-yet-unfound path, is the true
  event-injection point.
- Hypothesis to test next: `sched_event_poll` is the IDLE-task body, reached via
  a TCB run-fn pointer COMPUTED at runtime (base+offset, so no literal `0x5524`
  appears), entered only when the ready-list drains -- which never happens
  because the blocked tasks are (wrongly, in emu) kept selectable. If so the fix
  is upstream of the event register: why the blocked task stays selected instead
  of the system going idle. Alternatively the injection is a distinct
  set-the-bit path a hardware condition drives, independent of `sched_event_poll`.
  Decide by tracing (1) how the idle/ready-list transition works and (2) which
  of the 14 disp-0x6c sites ORs bits into `[0x22bc]`.

## Session-2c: the tasks are NOT event-mask-blocked (correction), + a methodology fix

A dynamic pass with reliable polling CORRECTS the event-flags framing above.

**Methodology fix (important):** store-EA watches that match `read_ar(s)+imm`
against a target address are UNRELIABLE in this firmware -- the windowed/relocated
RAM is written through aliased virtual addresses (e.g. a store to `0x2278` goes
via `0x2000_2278`-class aliases), so exact-EA matching misses real writes. Proven:
`m2c_probe_addr_store_watch` saw only ONE write to `0x2278`, but polling
`bus.load_local32(0x2278)` every step (`m2c_probe_current_task_timeline`,
`m2c_probe_poll_watch`) shows TWO transitions. **Always poll via `load_local32`
for RAM state; do not trust store-EA/value watches for aliased RAM.**

**Reliable dynamic facts (poll-based, 1.5M instrs):**
- Current task: `0x10f10` (instr 41464) -> `0x9040` (instr 58754), then FROZEN on
  task B (`0x9040`) for the remaining ~1.44M instrs.
- Task B state byte `[0x906c]` (= `0x9040+0x2c`) set to `6` (ready) at instr
  58882 (pc `0xd836`, dispatcher) and stays 6 -- perpetually ready.
- **Task B pending mask `[0x9070]`(+0x30) = 0 and wait-mask `[0x9078]`(+0x38) = 0
  throughout** (only the init-zero at 7280). Task A pending `[0x10f40]` = 0 too.
- Steady-state hot loop (`m2c_probe_poll_map`, warmup 300k): a ~390-instr macro
  cycle in the scheduler helper `FUN_0000c928` walking the 9-entry waiter table
  (`0x2288..0x229c`) and task fields, almost all reading **0**. Hottest reads are
  `[base+0x2c]`/`[base+0x38]` with a base that is sometimes NULL.

**What this overturns:** the tasks are NOT blocked on the pending-event bitmask
mechanism -- those masks are never set because the tasks are not waiting on them.
Task B is simply perpetually READY, and the scheduler re-runs it forever without
it completing/yielding. So `[task+0x30]` is not "the awaited completion":
force-done set it to 1, which merely made the dispatcher take the
`deliver_pending_events` (`0xcadc`) side-path that happens to advance the
scheduler -- it does NOT correspond to the real completion. Earlier sections that
frame the wall as "an event bit that never gets set" are superseded by this.

**Where the real wait lives (next):** inside task B's own work, not the
scheduler's event-flags. The perpetually-ready task's run loop reads scheduler/
task fields that stay 0; in a healthy boot something makes one non-zero
(a task becomes ready, a timer field advances, an init step completes). The
event source register `0x27010d28` + `sched_event_poll` remain unreached and may
be a LATER-stage concern, not this wall. Pinning THIS wall needs: trace task B's
actual work function (what it computes/polls each cycle and what field it needs
non-zero), ideally diffed against a healthy reference. This is where a HW-
differential (observe a real Phoenix mgmt-fw boot, or read APERTURE0 `0x3010d28`
live) would beat more solo static RE. Reassessing approach with Maya here.

## Session-2d: THE POLL FOUND -- HW event-status pages 0x2727_n000 bit3

A clean execution trace (`m2c_probe_exec_trace`, follows real PCs so decode
alignment is correct -- linear disasm misaligns here) of one steady-state
macro-cycle (~390 instrs, warmup 300k) resolved what task B actually waits on.

The macro-cycle: popcount loop (`0xc964`) -> run-fn `0x588c` -> `0x8770` ->
`0x8620` -> `0xc530` -> a memcpy (`0xb0e71d`, relocated seg) -> `0x7fc4` ->
`FUN_00007fa0` -> **`FUN_00008c68` (the HW poll)** -> `task_dispatcher` ->
scheduler helper `0xc938` -> repeat.

**The wait (decisive):** inside `FUN_00008c68`, a loop (`0x8c85`) reads the
hardware event-status pages and checks a bit:

```
a5 = 0x2727_1000, then += 0x1000 each iter -> 0x2727_2000, _3000, _4000
0x8c8b  Bbci a9, bit3, 0x8cae     ; a9 = [a5]; if bit3 CLEAR, skip to next page
```

So every macro-cycle the firmware polls **`0x2727_1000` / `0x2727_2000` /
`0x2727_3000` / `0x2727_4000`** (four pages, stride `0x1000` -- looks per-column)
for **bit3**. All read 0 in emulation -> the poll always falls through -> the
boot never advances. (`FUN_00008c68` also has bit0/bit1 checks at `0x8c95`/
`0x8ca2` -- the prior-session lead -- but the steady-state gate here is bit3.)
This is the concrete hardware event the mgmt firmware waits on, and it is the
answer to "what does the perpetually-ready task poll".

**Seed experiment INCONCLUSIVE (translation gap):** seeding bit3 (`0x8`) onto
`0x2727_n000` via `bus.store32` and re-running did NOT advance the boot, and the
bit3-set path (`0x8c8e`) was hit 0 times -- i.e. the CPU's translated read of
virtual `0x2727_n000` never saw the seed. Same MMU-aliasing class as the RAM
writes: the firmware reads these pages through its DTLB to a physical the raw
`store32` didn't hit. So we cannot yet confirm bit3 is THE gate; seeding must go
through the correct translation (find the DTLB mapping for `0x2727_xxxx`, or
store via the CPU's translated data path).

**Relationship to `0x27010d28`:** that register is read only by
`sched_event_poll`, which is never reached -- a LATER-stage / different concern.
The ACTUAL wall is this `0x2727_n000` bit3 poll in `FUN_00008c68`.

**Build pivot (concrete now):** (1) seed bit3 through the right translation to
CONFIRM it unblocks (find the DTLB phys for `0x2727_n000`); (2) if confirmed,
determine what sets bit3 on real HW (an early-init hardware-event/interrupt
status -- these pages are per-column event-status in the mailbox aperture) and
model it faithfully; (3) characterize `FUN_00008c68`'s full bit semantics
(bit0/1/3) and the ack it does when a bit is set (`0x8c9b` writes `[a4]` at
`0x2727_n114`).

## Session-2d-confirm: poll gate corrected + MMU-seed cracked, but not the completion

Re-reading the trace's load EA corrected the poll source, and the confirming
experiment cracked the MMU-seed issue -- but revealed another layer beneath.

**Poll source corrected:** the load at `0x8c88` is `L8ui a9,[a8+0]` with base
`a8`, EA = **`0xf9e0`** (then `0xfa40`, `0xfaa0`, stride `0x60`) -- a RAM struct
BYTE, NOT the HW page. The `0x2727_n000`/`0x2727_n114` values in `a5`/`a4` are
the ACK TARGETS (written at `0x8c9b` only after a bit is seen), not the poll
source. So the firmware polls per-struct RAM pending bytes at `0xf9e0 + k*0x60`
for bit3, and acks to the associated HW page `0x2727_n114`.

**MMU-seed cracked:** seeding `0xf9e0 + k*0x60` bit3 via `store_local8` (the
alias-correct local path) DID reach the poll -- the bit3-set path (`0x8c8e`) was
hit 6918 times (vs 0 with the raw `store32`/HW-page seed). This confirms the
recurring "seed never lands" failures all session were the MMU-alias tax: raw
`bus.store32` to a virtual RAM/peripheral address does not hit what the CPU's
DTLB-translated read reads. **Use `store_local8`/`load_local32` for firmware RAM
state.** (A proper DTLB-backed data path would retire this whole probe-bug class.)

**But bit3 is NOT the completion gate:** with bit3 driven 6918x, the boot still
does not advance -- last_pc `0xc969`, done-flag `0x9070` still 0, steady state
unchanged (the popcount loop `0xc964` + scheduler). The bit3 handler runs each
cycle and is absorbed into the same loop. So this poll is a routine per-cycle
status check, not the thing task B is ultimately blocked on.

**Assessment:** the boot is a DEEP multi-gate dependency chain -- this session
corrected the model 5-6 times as each layer resolved (external-agent ->
event-dispatcher -> interrupt -> event-mask -> HW-page poll -> RAM-byte poll),
and the completion still recedes. The hottest steady-state work is the popcount
loop at `0xc964` (counts bits in a mask `a3`, exits at >=2) -- the scheduler's
core readiness/priority decision, whose input `a3` is the next thing to
understand if we keep peeling. Strategic reassessment with Maya before more RE:
whether to (a) keep peeling gates, (b) implement a proper MMU/DTLB data path
first (removes the alias tax that slows every experiment), or (c) reconsider
whether full-boot-to-idle is the right depth for the timing-emergence goal.

## (Deferred) x2i experiment -- for the post-alive stage

Once boot reaches the alive handshake, deliver a real x2i host->fw message and
observe whether the fw's RX handler runs and sets a done-flag:
1. Recover the fw-local x2i ring base + x2i tail register offset from the
   `mgmt_mbox_chann_info` struct the fw wrote (find via the alive handshake:
   the fw writes the struct addr to `FW_ALIVE_OFF`).
2. Write a `SET_RUNTIME_CONFIG(0x10A)` message (16-byte header + payload) into
   the x2i ring at its tail, advance the x2i tail register, and (if needed)
   raise the mailbox interrupt.
3. Watch: does the fw decode it (RX handler executes), and does any
   `[task+0x30]` get set by fw code? 
   - YES -> shape (i) confirmed; model = host x2i message delivery; rework the
     completion mechanism accordingly.
   - NO (fw never polls x2i, interrupt undeliverable) -> shape (ii) stands;
     external-agent local write is the model, and we need the per-task binding
     from elsewhere.

## Session-3 (2026-07-07): re-validation on the trustworthy data path + the wall, freshly mapped

Context: the MMU/DTLB data-path refactor landed (translation-authoritative; the
alias tax that corrupted probes all along is retired; canonical `Cpu::data_*` /
`Bus::data_*`/`inst_*` API, bare bus load/store gone). All boot-RE probes were
migrated to that API. This session re-validated the iter18 conclusions under the
now-trustworthy lens, then advanced the frontier.

### Re-validation: the wall is BYTE-IDENTICAL to iter18

`m2c_probe_current_task_timeline` on the canonical path:

```
instrs = 1500000; stop = budget
post (tail!=0) at instr = 6973
first done-flag check (pc=0xd828): (0x10f10, flag 0)
[0x2278] <- 0x10f10  at n=41464
[0x2278] <- 0x9040   at n=58754
then frozen on task B (0x9040)
```

Every value matches iter18 exactly (post ~6972, tasks at 41464/58754). So making
the executor translation-authoritative changed NOTHING about firmware execution
-- **confirming empirically that the low window is identity-mapped from reset**
(the way-6 ei0 characterization). No iter18 conclusion is overturned; they were
all run on an execution identical to what we have now. The payoff is friction
removal: every probe from here is alias-correct by construction (including the
store-watches that were untrustworthy).

Note: the doc's LATE conclusions (session-2c/2d-confirm) were already made with
`load_local32`/`store_local8`, which for the identity low window are byte-
identical to `cpu.data_read32` -- so they were already trustworthy. The alias tax
had been progressively worked around during iter18; the refactor systematizes it.

### The wall, freshly and reliably mapped (exec_trace, warmup 300k)

Task B (`0x9040`) is perpetually selected. Its macro-cycle:

1. **Scheduler readiness decision** `FUN_0000c928` (entered `0xc938` from
   dispatcher `0xd836`): tail at `0xc95e..0xc972` is a **popcount-ge-2 test** --
   count set bits in mask `a3` over 32 shifts, return 1 if >=2 else 0. Runtime:
   **`a3 = 0`** (empty mask), derived from **`a6 = NULL`** (`0xc948 a3=[a6+28]`).
   So it returns 0 every cycle. (a6=NULL matches session-2c's "base sometimes
   NULL".) This is the "nothing ready" state; the real scheduler would idle here
   (-> `sched_event_poll` `0x5524`), ours re-runs task B instead -- but that is a
   SYMPTOM of the missing completion, not the root gate.
2. **Task B run-fn** `0x588c` (=`FUN_00005580+0x30c`) -> `0x8620`/`0x8770` ->
   `FUN_0000c530`: **builds a 7-word descriptor at `0xfae0`**:
   `{+0:1, +4:1, +8:0xf, +0xc:0, +0x10:0x9040(task ptr), +0x14:0, +0x18:0}`.
   `+8 = 0xf` is a **4-bit mask == 4 columns** (Phoenix NPU1 = 4 usable AIE cols).
3. **Cache-flush** the descriptor: `Callx8 -> 0xb0e710` (relocated seg) runs a
   `Dhwbi` (data-cache writeback-invalidate) loop over `0xfae0..0xfb00` --
   publishes it to a coherent/DMA consumer.
4. **Poll** `FUN_00008c68`: reads 4 per-column pages `0x2727_1000/_2000/_3000/
   _4000` (stride 0x1000) for bit3, acks to `0x2727_n114`. All read 0 in EMU.

The descriptor's **colmask 0xf lines up with the 4 per-column poll pages**: task B
posts a **4-column hardware operation and waits for per-column completion**.

### H1 vs H2: leans H1 (faithful firmware waiting on hardware)

The firmware is doing legitimate work -- publish a per-column descriptor, poll for
per-column hardware completion. Nothing in EMU consumes the descriptor or sets the
per-column done bits, so task B re-posts/re-polls forever. This is NOT an emu
scheduler bug (the NULL-mask readiness is the correct "waiting on HW" state); it is
exactly the "hardcoded HW timing that must EMERGE" target of the firmware-emulation
dream -- here, per-column completion latency.

**Caveat (do not overclaim):** session-2d seeded bit3 on the poll pages (6918
hits) and boot did NOT advance -> the completion is the FULL per-column ack
protocol (the consumer writing a multi-field response), not a single bit. Nailing
it requires decoding what a real per-column agent writes back.

### Open (Explore dispatched 2026-07-07): identify the per-column consumer

Explore over xdna-driver + aie-rt to identify: (1) the hardware identity of the
`0x2727_1000..4000`/`+0x114` block (per-column event/status; FW 0x27xxxxxx ->
driver 0x03xxxxxx); (2) the driver/HAL struct matching the 7-word descriptor;
(3) the per-column agent + its full ack protocol; (4) whether there is a
documented pre-mailbox per-column bring-up handshake matching "post a 4-col-masked
descriptor, wait per-column done". Probes/disasm used this session:
`m2c_probe_current_task_timeline`, `m2c_probe_disasm_range` (XDNA_FW_DISASM),
`m2c_probe_exec_trace` (XDNA_FW_TRACE_WARMUP/COUNT).

### Session-3 cont.: FUN_00008c68 poll DEFINITIVELY FALSIFIED as task B's gate

With the alias tax retired, the poll handler could finally be tested cleanly.

**Poll handler decoded correctly (PC-following trace; static disasm is FLIX-
misaligned here).** `FUN_00008c68` is a TWO-level per-column protocol, not one bit:
```
loop 4 cols (a8=0xf9e0+k*0x60 RAM byte; a5=0x2727n000 HW page; a4=0x2727n114 ack):
  a9=[a8]; Bbci a9,bit3 -> next col            ; RAM byte bit3 = "work pending"
    a9=[a5]; Bbci a9,bit0 -> next col          ; HW page bit0 = "hw ready" (NO ack if clear)
    [a4]=a7                                     ; ack -> 0x2727n114
    a9=[a5]; Bbci a9,bit1 -> spin              ; wait until HW page bit1 SET
    [a8] &= 0xf7  (clear RAM bit3)              ; consume
```
Session-2d's seed set ONLY the RAM byte bit3 (and via a raw physical `store32`),
so the handler's HW-page bit0 check always failed -> it never acked/consumed.
That was an alias artifact + an incomplete-protocol seed, not evidence about the gate.

**The clean experiment (`m2c_probe_force_event`, now alias-correct).** Fixed the
probe: HW pages seeded via `cpu.data_write32` (translation-aware, matches the
handler's DTLB read -- a raw `bus.data_store32` lands on a different backing);
added `XDNA_FW_EVENT_SEED_AT` to deliver the completion at steady state (seeding
at n=0 is wiped by early-boot memset before the poll runs). Results:
- bits=0xf, reseed, n=0, 8 pages -> **fault-spin** at 0xb1f (`wsr EXCSAVE2` loop).
  Over-seed (held-forever + bit2 + extra pages) corrupts boot. Lesson: the
  completion is a DISCRETE consumable event, not a permanently-held state.
- bits=0xb (bit0|1|3), seed once at n=100000 -> poll saw bit3 (`active_hits=3`),
  handler acked+consumed, but `last_pc=0xc55c` (task B's loop), `[0x9070]=0`.
- bits=0xb, reseed from n=100000 (persistent, no early corruption) ->
  `active_hits=6309` (handler acks EVERY cycle), yet `last_pc=0xc533`,
  `[0x9070]=0`. **Task B never advances.**

**Conclusion (definitive, trustworthy):** fully satisfying `FUN_00008c68` --
correct two-level bits, alias-landed, timed, persistent -- does NOT advance task B.
The `0xf9e0`/`0x2727n000` poll is a routine per-cycle status sweep, NOT task B's
completion gate. This hypothesis (dominant in sessions 2c/2d) is KILLED. Note this
falsification was impossible pre-alias-fix: session-2d couldn't distinguish "not
the gate" from "seed didn't land." Now the seed provably lands and it's still not
the gate.

**Next thread:** find task B's ACTUAL gate. force-done (set `[0x9070]`) advances to
~575k, so the real gate is whatever naturally completes task B's run-fn / sets its
done-flag. Trace task B's run-fn (`0x588c` -> `0xc530` build+flush -> `0x7fa0`) for
the complete-vs-loop decision and the state it needs -- NOT the 0x8c68 poll.

### Session-3 cont. (2): dispatcher decoded -- force-done is a SKIP, not a completion

Clean disasm of `task_dispatcher` (0xd7f0) resolves the control flow and explains
why force-done "advances" without being the gate:
```
a4 = [SCHED+40]                 ; current task (0x9040)
a5 = [task+0x1b] (byte)
0xd811 Bnei a5,1 -> 0xd828      ; if task[0x1b]==1: Call 0xc530 (work-A) first
0xd828 a10 = [task+0x30]        ; pending-event mask (=[0x9070])
0xd82a Beqz a10 -> skip
0xd82c Call 0xcadc(a10)         ; deliver_pending_events
0xd833 task[0x2c]=6             ; mark ready
0xd836 Call 0xc938              ; scheduler helper (popcount-ge-2 on a3)
0xd839 Bnez a10 -> 0xd845(retw) ; *** if pending!=0, SKIP the work run-fn ***
0xd842 Callx8 [SCHED2+36]=0x588c; else run task B's work
```
So with `[task+0x30]==0` (task B's real state) the dispatcher calls `0x588c`
(task B's work) EVERY dispatch. force-done set `[task+0x30]=1`, which at 0xd839
makes the dispatcher SKIP `0x588c` and just deliver+return -- confirming (as
session-2c said) it advances by bypassing the stuck work loop, NOT by completing it.

**Task B's work `0x588c` re-executes fully each dispatch** (not a tight spin): it
clears byte flags (0x123d0/0x1eb00/0x249a0/0x9268...), then via `0x8770`
(FUN_00008620+0x150, reads [0x1eb08]=0) rebuilds the colmask-0xf descriptor at
0xfae0, cache-flushes, and polls. Every field it reads stays 0, so it re-does the
same work forever. The steady state never enters FUN_00008620's ENTRY (only the
+0x150 tail) -- the entry's decision logic is skipped.

### The real gate -- two live hypotheses (need to distinguish)

- **(H-a) Task B is a WORKER** that completes when its per-column operation's
  result field goes non-zero (a status distinct from the falsified 0x8c68 poll --
  candidates: [0x1eb08] or the descriptor's own response fields). It re-issues
  each tick because the result never appears in EMU.
- **(H-b) Task B is PERIODIC/IDLE** (runs forever by design) and boot advances
  when ANOTHER task is readied -- which needs the dormant event path
  (0x27010d28 via sched_event_poll 0x5524, never reached) to fire.

Both ultimately require an external event that never occurs in EMU. Next decisive
experiments: (1) trace the force-done run FORWARD (advances to ~575k) to see what
task B's bypass enables and where boot wedges next -- triangulates the gate from
the destination; (2) determine worker-vs-idle by disassembling FUN_00008620's
entry + what 0x588c does with [0x1eb08]/the descriptor response.

---

## Session-4 (2026-07-07, post-compact): the round -- worker-vs-idle RESOLVED

Ran the two decisive experiments on the alias-correct trunk. Result is unambiguous
and it converges H-a and H-b: **task 0x10f10 is a WORKER, and its completion is an
AIE-array event.** The firmware RE has bottomed out at the firmware<->array boundary.

### Correction to Session-3 labeling

`m2c_probe_force_done` forces `[a4+0x30]` at the FIRST `0xd828` hit. The current-task
timeline (`m2c_probe_current_task_timeline`) shows current task = **0x10f10** first
(n=41464), only later 0x9040 (n=58754); the first `0xd828` is at n=47896 with
`a4=0x10f10`. So force-done forces **task 0x10f10's** mask `[0x10f40]`, NOT task B's
`[0x9070]`. `forces=1` (the whole 623k run hits `0xd828` exactly once) -> that single
force cascades boot from the ~58k wall to **n=623097** (~565k advance), never reaching
task B's spin at all. Task B's spin was a *downstream symptom* of 0x10f10's work-fn
running instead of being skipped.

### Experiment (a): force-done forward -> new frontier at ~623k

Stop: `Unknown at pc=0xd903 word=0x1d020cfe`, reached via `Callx4`->`0xd900`. The 48
instrs before it are all `FUN_0000e098` doing `Rotw` + a full spill of the register
windows + the special-register file -- **the exception entry vector**. So ~623k took
an exception whose C-dispatch hit a decode gap (FLIX-dense region; likely a bundle
misdecode or an unimplemented opcode -- next frontier, not chased here).

Reading: forcing the event is a crude stand-in for "columns finished." It advances
boot far (proving the event is the gate) but eventually wedges because the columns
never actually produced their results -- downstream code reads a result that stays 0
and faults. This is exactly what H-b predicts.

### Experiment (b): task 0x10f10's work-fn (0x588c) does REAL work

`m2c_probe_exec_trace WARMUP=48000` traces 0x10f10's dispatch (a4=0x10f10, `[0x10f40]=0`
so 0xd839 does NOT skip -> runs 0x588c). The run-fn is SHARED (`[SCHED2+36]=0x588c` for
whichever task is current; 0x10f10 and 0x9040 use the same executor). For 0x10f10 it:
1. `0x8770` (FUN_00008620+0x150) reads `[0x1eb08]=0` (descriptor response).
2. `0xc530` builds a **colmask-0xf descriptor** at `0xfae0` = `{1,1,0xf,0,...,0x9040,...}`,
   `Memw`, then cache-flushes `0xfae0..0xfb70` (the `Dhwbi` loop at 0xb0e710).
3. `0x7fa0`->`0x8c68`: the two-level poll over 4 columns (`0xf9e0+k*0x60` bit3 ->
   HW page `0x2727n000` bit0). All bytes 0 -> nothing ready.
4. `0x7fe4`->`task_dispatcher`: loops back; `[0x10f40]` still 0.

So 0x10f10 posts a per-column (colmask 0xf) descriptor, flushes it, and polls for the
columns' completion. It is a **worker**, not idle.

### Why the poll (0x8c68) was falsifiable yet the event (pending mask) is the gate

Two distinct mechanisms, and they are NOT equivalent:
- **Poll path** (0x8c68, inside the work-fn): reads the per-column RAM byte / HW page.
  Session-3 fully satisfied it (6309 alias-correct acks) -> boot did NOT advance.
- **Event path** (pending mask `[0x10f40]`, via `deliver_pending_events`/
  `wake_tasks_by_event_mask`): forcing it -> dispatcher SKIPS the work-fn, marks the
  task ready, delivers the event -> boot cascades 565k. THIS is the real completion.

The event source is the hardware event register **0x27010d28** (driver `0x03010d28`),
read by `sched_event_poll` (0x5524) -- which is NEVER reached naturally because the
cooperative scheduler stays inside 0x10f10's post-poll loop and never yields to the
event/idle path. On silicon the AIE array raises that event (a task-complete token)
when the posted colmask-0xf work finishes; the fw event dispatcher then sets 0x10f10's
pending mask and the task completes.

### Verdict and fork resolution

- **Worker-vs-idle: WORKER (task 0x10f10), definitively.**
- **Completion contract fully characterized:** AIE-array task-complete event ->
  HW event reg 0x27010d28 -> fw pending mask `[0x10f40]` -> task 0x10f10 done -> boot
  proceeds. Forcing the mask proves the gate; boot then wedges at ~623k for lack of
  *real* column results (the array never executed the posted descriptor).
- **Direction: H-b (array integration), NOT H-a (more firmware peeling).** The RE has
  reached the firmware<->array boundary. The emulator already has an AIE-array model;
  the path forward is wiring it to the firmware's descriptor-post + event-delivery path
  (consume the colmask-0xf descriptor at 0xfae0, execute the per-column init, raise the
  completion event / set the pending mask). That is a design-fork decision -- parked for
  Maya.

### Session-4 cont'd: the event-delivery boundary, charted end-to-end

Having resolved WORKER, chased the actual delivery machinery (H-a, "chart to 100%").
Direct-caller xref (`m2c_probe_call_xref`) + static disasm of the scheduler core:

**The event path is all rooted in `FUN_00005580` (the scheduler/message dispatcher):**
- `+0x1f3` (0x5773) -> `Call8 sched_event_poll` (0x5524)
- `+0x286..0x289`: `Ssl a6; Sll a10 = a7<<a6` (a7=1) => **a10 = 1<<event_id**, then
  `Call8 wake_tasks_by_event_mask` (0xd84c). This is the event-id -> bitmask -> pending-
  mask conversion. `wake_tasks_by_event_mask` walks tasks and ORs `1<<id` into their
  pending masks `[task+0x30]`.
- `+0x30c` (0x588c) -> the shared WORK run-fn traced in exp (b). So the dispatcher and
  the work-executor are the same code region, entered at different offsets via fn ptrs.

**`deliver_pending_events` (0xcadc)** callers: 0xc9d3, 0xd07b, 0xd7e3, and **0xd82c**
(the task_dispatcher itself, right after the pending-mask read at 0xd828). So the
dispatcher both consumes (0xd82c) and the wake path produces (0x5809) pending-mask bits.

**Delivery is INTERRUPT-driven, not polled:**
- `FUN_00005580` has **no direct callers** -- reached only via a fn-ptr/handler table.
- `sched_event_poll` (which would poll HW event reg `0x27010d28`) is **never reached
  naturally**. So on silicon the AIE completion does NOT arrive by the fw polling
  0x27010d28; it arrives as an **interrupt** -> ISR posts an event message -> the
  message dispatcher (FUN_00005580) runs `wake_tasks_by_event_mask(1<<id)` -> sets
  task 0x10f10's pending mask `[0x10f40]` -> next dispatch completes the task.
- Consistent with the ~623k stop landing in the exception/interrupt context-save vector
  (FUN_0000e098): exceptions/interrupts route through there.

**The complete silicon completion flow (now fully mapped except the IRQ entry):**
1. AIE array finishes the colmask-0xf work task 0x10f10 posted (descriptor @ 0xfae0).
2. Array raises an interrupt (status visible at/around `0x27010d28` = driver `0x03010d28`).
3. Level-1 int vector -> ISR reads AIE int-status, builds an event message.
4. Message dispatcher `FUN_00005580` -> `wake_tasks_by_event_mask(1<<event_id)`.
5. Task 0x10f10's pending mask `[0x10f40]` set -> dispatcher (0xd828) sees pending!=0
   -> `deliver_pending_events` + mark ready + skip the (now-satisfied) work-fn -> boot
   proceeds. (Exactly the path `m2c_probe_force_done` short-circuits.)

**The ONE remaining uncharted link (next stretch):** step 3 -- the level-1 interrupt
vector and the specific ISR that reads the AIE interrupt status and posts the event
message. Can only be found by static RE (no interrupt fires in EMU without an array
model). Entry: disasm the int/exc vector (general-exc stub ~0xae0, VECBASE TBD) ->
the interrupt dispatcher -> the handler that maps the AIE IRQ -> event message ->
FUN_00005580. This is the last piece before the boundary is 100% charted and the
array-integration contract (H-b) is fully specified.

### The A-frontier (623k decode gap) -- classified as a forcing artifact

`0xd903 Unknown` is NOT a clean opcode frontier. The region decodes cleanly when
aligned from 0xd8f0 (Call8->0xc530, MoviN, RetwN, Entry@0xd908). The executor reached
0xd900 (1 byte off the real boundary) via a `Callx4` whose literal target is garbage --
consistent with boot running on CORRUPT state after force-done skipped 0x10f10's real
work (the columns never produced results, so a later fn-ptr/result read is bad). It is
downstream corruption from forcing, not a decoder gap worth fixing. Set aside.

### Session-4 cont'd: the ISR hunt -- interrupt subsystem mapped, exact bit is a dynamic question

Built `m2c_probe_literal_xref` (static L32r-literal xref over a value range) to find who
loads the AIE interrupt-status constants. The AIE int-controller register cluster:

```
0x27010d00/d04/d08/d0c/d10/d14/d18/d1c   status/enable/clear register bank
0x27010d28                                main status/pending register
0x27010ac0, 0x27010554/558               related control regs
```

Loaders of the pending reg `0x27010d28`: `sched_event_poll`'s path (FUN_00005580 +0x26c,
the poll case) AND handler funcs `FUN_00007df0`, `FUN_00007e4c`, `FUN_00009804`,
`FUN_00009958`. The d00..d1c bank is driven by `FUN_00007824/7880/9c80/9ce0`.

Findings on the interrupt subsystem:
- `FUN_00007df0`/`FUN_00007e4c` are enable/mask CONFIG funcs -- read-modify-write specific
  IRQ-enable bits (bit4, 0x100, 0x700, ...) via a shared MMIO accessor `0x89d8`.
- `FUN_00009804`/`FUN_00009958` are per-bit enable/status helpers (masks like `63<<23`,
  bit25) -- also via `0x89d8`.
- `FUN_00007e4c`, `FUN_00009804`, `FUN_00009958` have **no direct callers** -- registered
  as fn-ptrs and dispatched via `callx*` (an IRQ-handler table). The static direct-call
  xref cannot follow `callx*`, and these handlers sit in FLIX-bundle regions where linear
  static disasm misaligns.

**Conclusion / seam.** The interrupt subsystem is table-driven with registered handlers;
the AIE column-completion is one specific bit in `0x27010d28` whose handler posts the
event message that `wake_tasks_by_event_mask(1<<id)` turns into task 0x10f10's pending
bit. Pinning the EXACT bit + handler + event-id is best done DYNAMICALLY -- inject the
interrupt (set the Xtensa int line + seed 0x27010d28 with a candidate bit), run, and watch
whether the ISR fires and `[0x10f40]` gets set. That injection experiment IS the first
concrete step of H-b (array-integration delivery): it both closes the last boundary inch
and stands up the completion-delivery path. So the boundary is charted to the
interrupt-subsystem level; the final pin overlaps the start of delivery work.

### Session-4 cont'd: interrupt INJECTION -- falsifies interrupt-driven, reframes to SYNCHRONOUS poll

Built `m2c_probe_inject_interrupt` to faithfully drive the AIE-completion IRQ: warm to
steady state, seed the status reg `0x27010d28`, set `cpu.interrupt`, keep stepping (the
interp's `step()` delivers a level-1 interrupt via the general-exc handler `0x2958`,
EXCCAUSE=4, as soon as PS.INTLEVEL returns to 0). Result (warmup 60k, status=0xffffffff,
fire=INTENABLE, run 400k):

```
INTENABLE = 0x00000001        <- only ONE line armed: bit 0 = the mailbox doorbell
                                 (NOT an AIE line)
min intlevel seen = 2         <- the busy dispatch loop holds PS.INTLEVEL=2 the WHOLE time
level-0 deliverable windows = 0  <- over 400k steps, NEVER a moment an interrupt could fire
interrupt taken (@0x2958) = None
[0x10f40] pending set     = None
```

**This FALSIFIES the "interrupt-driven" conclusion from earlier this session.** A level-1
(or level-2) interrupt is masked the entire time boot is stuck, and the AIE completion
line isn't even enabled in INTENABLE. The completion CANNOT be delivered by interrupt in
this state.

**Corrected model -- TWO separate subsystems, earlier conflated:**
- **Boot per-column completion (THE boot gate): SYNCHRONOUS.** Task 0x10f10 posts the
  colmask-0xf descriptor and *synchronously polls* `0x8c68` for per-column completion
  (RAM byte `0xf9e0+k*0x60` bit3 -> HW page `0x2727n000` bit0/bit1), holding INTLEVEL=2 in
  a critical section. No interrupt involved. The gate is the ARRAY writing those per-column
  completion flags (and the descriptor response the post-poll code reads, e.g. `[0x1eb08]`).
- **General async events (NOT the boot gate): interrupt-driven.** `0x27010d28` +
  `sched_event_poll` + `FUN_00005580` message dispatcher + `wake_tasks_by_event_mask` are a
  *different* event class. That machinery is real but is not what gates boot here. My
  earlier inference ("sched_event_poll never reached => interrupt-driven completion") was
  wrong: it's never reached because the boot completion doesn't route through it.
- **force_done (pending mask) is a BYPASS, not the completion.** Setting `[0x10f40]` makes
  the dispatcher SKIP task 0x10f10's work-fn (0xd839) -- it never runs the post/poll, which
  is why it advances (on absent results) and faults at 623k. It does NOT model the real
  synchronous handshake.

**Why Session-3's poll "falsification" was incomplete:** it seeded the per-column poll
flags but not the descriptor RESPONSE the work-fn reads after the poll succeeds. The full
synchronous completion needs BOTH: (1) poll flags so `0x8c68` consumes a completion, and
(2) the descriptor response data so the post-poll code sees a result instead of re-looping.

**Reframed H-b contract (the real one):** when task 0x10f10 posts the colmask-0xf
descriptor at `0xfae0`, the array model must write, per column: the completion poll flags
(`0x2727n000` bit0/bit1 + RAM `0xf9e0+k*0x60` bit3) AND the descriptor response fields the
work-fn reads. Next experiment: seed BOTH together (poll flags + response) at steady state
and confirm boot advances synchronously -- that both validates the corrected model and is
the first real rail of the array-response model.

### Session-4 cont'd: synchronous-completion experiment -- poll is satisfiable but NOT the gate

Added an EA-resolving trace of the SATISFIED-poll path (`m2c_probe_exec_trace` +
`XDNA_FW_TRACE_SEEDPOLL=<bits>`, seeds `0x2727n000` + `0xf9e0+k*0x60` each step). Also ran
`force_event` with `EVENT_BITS=0xb SEED_AT=50000 RESEED=1`.

**Result: the poll can be fully satisfied alias-correctly, but boot still does NOT advance.**
- With bit0|bit1|bit3 seeded, `FUN_00008c68`'s active path (`0x8c8e`) is taken 6660 times;
  the poll runs its full ack (`S32i -> 0x2727n114`) + consume (`clear RAM bit3`) sequence.
  (Trace-display caveat: the `ea=...=0x0` shown for `0x2727n000` reads is an alias artifact
  -- the display read is physical/24-bit-masked, the CPU's actual read is DTLB-translated
  and DOES see the `0xb` seed, which is why it takes the ack path instead of skipping.)
- Yet `[0x10f40]=0`, `[0x9070]=0`, boot stuck in the poll loop at the 1M budget.

So **satisfying the poll is necessary-looking but not sufficient** -- the poll-consume is
NOT the completion gate. Two candidate deeper gates examined:
- The poll's `0x8c8e`: `L32iN a9,[0xf9e8]; Bne a9,a2` -- a per-column TAG/sequence match
  (RAM struct +8 must equal a2). Trivially matches at 0==0 today; a REAL completion likely
  needs the array to write a specific tag here that the poll returns as "descriptor N done."
- `FUN_00008620` entry (`0x8620`, the steady-state-skipped decision logic) -- turns out to
  be TLB/cache management (`Wdtlb`/`Dii`/`Dsync`), NOT a completion decision. Ruled out.

**Where this leaves the boundary (two concrete mechanisms now EXCLUDED by experiment):**
1. Interrupt-driven delivery -- excluded (INTLEVEL held at 2, AIE line not armed).
2. Simple synchronous poll-consume -- excluded (satisfiable but boot doesn't advance).

The work-fn `0x588c` RE-POSTS the colmask-0xf descriptor every dispatch and re-dispatches;
boot doesn't advance because no OTHER task is readied. Task-readying goes through the
event/wake path (`wake_tasks_by_event_mask` sets `[task+0x30]`), whose trigger is hardware
(array) activity -- specifically a TAG-MATCHED descriptor completion the array must write
(the `[0xf9e8]` tag + likely the `0x1eb00` descriptor-response struct), not a bare poll bit.
No single firmware-side seed satisfies it because the completion is a *tagged* response tied
to the specific descriptor posted.

**Assessment:** the boundary's MECHANISMS are now fully charted -- descriptor post
(0xfae0, colmask 0xf), the per-column poll handshake (0x8c68: RAM bit3 -> HW page bit0/1 ->
ack 0x2727n114 -> tag match [0xf9e8]), and the task-readying event path. What remains is not
more firmware RE but building the ARRAY-SIDE RESPONDER (H-b) that consumes the posted
descriptor and writes the correctly-tagged per-column completion + response the firmware
polls for. That crosses from charting into delivery -- a design-fork checkpoint for Maya.

### Session-4 cont'd: THE PIN -- exact tag/response layout (and it's degenerate)

Mapped every field of the descriptor + per-column + ring structs from the EA-resolved
satisfied-poll trace:

```
Descriptor @ 0xfae0 (built by 0xc530, cache-flushed):
  [0x00]=1   [0x04]=1   [0x08]=0xf (colmask)   [0x0c]=0
  [0x10]=0x9040 (task ptr)   [0x14]=0   [0x18]=0
  -- no sequence/tag field; all fields are small constants or the task ptr.

Per-column poll struct @ 0xf9e0 + k*0x60 (polled by FUN_00008c68):
  [0x00] byte : bit3 = "work pending" (poll skips col if clear)
  [0x08]      : TAG, matched via `Bne a9,a2` against the poll's arg a2

Descriptor ring @ 0x1eb00 (read by the work-fn at 0x877c):
  [0x00] byte : per-iteration status flag (cleared each loop at 0x58b6)
  [0x08]      : ring WRITE pointer -- `Add a14,a14,[0x1eb08]` = addressing only, NOT a gate
```

**The tag is DEGENERATE.** The poll's tag arg comes from `a7` (`0x7fde Or a10,a7,a7` ->
callee a2), and `a7=0` at this boot stage. So the poll matches TAG 0, and `[0xf9e8]=0`
satisfies it trivially -- there is NO nonzero tag or response value the array must echo that
we are missing. The descriptor carries no sequence field; `[0x1eb08]` is a write pointer.

**Conclusion (airtight):** the descriptor/poll/tag machinery is fully mapped and is a
trivially-satisfiable completion-DRAIN, not the boot gate. This CONFIRMS by exhaustion that
the boot completion is NOT any pollable memory value the array writes -- it is the
task-readying event path (`wake_tasks_by_event_mask` -> `[task+0x30]`), whose trigger is
masked (interrupt, INTLEVEL=2) in the busy loop.

**Sharpened H-b problem statement (for the regroup):** the array-side responder's job is NOT
to produce tag-matched poll completions (those drain as no-ops here). The open question the
H-b design must answer is what makes the busy dispatch loop EXIT / lower INTLEVEL so the
next task can be readied -- i.e. what the array's activity must change in firmware-visible
state to let boot progress past the descriptor-pump loop. Every LOCAL mechanism (interrupt,
poll, tag, ring) is now mapped and individually excluded; the gate is a loop-exit /
task-readying condition that spans the scheduler, and pinning it is the first task of the
H-b design (not more single-field seeding).

## Session-5 (2026-07-07): closing the firmware-side ledger before H-b

Goal: before pivoting to array integration, confirm from OBSERVED evidence (not
inference) everything around the array -- so the array is the sole remaining
unknown. Two decidable scans, both reusing existing probes. Prior spec/plan for
this went through three adversarial-review rounds (the array-responder "A2"
design was killed as theater; the delivery-probe "classify the exit" GATE was
killed as undecidable; the reframe below is what survived). Design record:
`docs/superpowers/specs/2026-07-07-hb-array-responder-slice-a-design.md`; plan:
`docs/superpowers/plans/2026-07-07-hb-delivery-scheduler-gate-probe.md`.

### (ii-b): INTENABLE / INTLEVEL audit -- GATE verdict = NO ATTEMPT (observed)

`m2c_probe_intenable_watch` over a full 1,000,000-instr boot. Result: the
firmware NEVER attempts an interrupt-delivery window anywhere in the reachable
trajectory.

- INTENABLE reaches `0x1` once at instr 2218 (`pc=0x200088d5`, the mailbox
  doorbell) and never changes again for the remaining ~1M instrs. No non-mailbox
  (AIE) bit is ever armed.
- INTLEVEL settles to 2 by instr 2219 and never returns to 0:
  `first INTLEVEL==0 after arm = None`. No delivery window ever opens.
- All 10 SR transitions occur by instr ~2219 (early boot setup); SR state is
  frozen (`INTENABLE=0x1`, `INTLEVEL=2`) for the entire rest of the run,
  including across the wall (~58754) to the budget.

This upgrades the Session-4 injection *inference* ("INTENABLE=0x1, no AIE line,
INTLEVEL held 2") to an *observed full-boot transition log*. The delivery-probe
"(i)" was conditional on an attempt; there is none, so (i) is correctly dead --
not run. Residual caveat (structural, not a gap here): "no attempt" covers the
entire reachable EMU trajectory, but if silicon arms an AIE line only AFTER
per-column init (which never completes in EMU), that is invisible to a
firmware-only trace -- exactly the array-gated hypothesis, and precisely what
H-b must resolve.

### (ii-a): readiness-field `[task+0x30]` writer map -- no non-event writer

Question: are there writers of the task-readiness field (`[0x10f10+0x30] =
0x10f40` for the stuck first task) beyond the known event path -- a non-event
writer would break the paradox.

**Runtime (airtight):** `m2c_probe_poll_watch XDNA_FW_POLL_ADDR=0x10f40` over
1,500,000 instrs -- `[0x10f40]` is NEVER written. Zero changes. No readiness
writer fires in the reachable trajectory, regardless of which function it lives
in. This is the strongest statement and it holds independent of any static
blind spot.

**Static (`m2c_probe_store_search XDNA_FW_STORE_DISP=0x30`):** 63 immediate-disp
`0x30` stores across 592 functions. The task/scheduler-pointer-based ones that
target a readiness field are the event-delivery path:
- `wake_tasks_by_event_mask+0x33` (`0xd87f`, base a6 = walked task ptr) and
  `+0x4b` (`0xd897`, base a4): the disasm confirms it reads `[entry+0x30]`,
  clears matched bits from both `[SCHED+108]` global-pending and `[entry+0x30]`,
  and invokes each woken task's run-fn via `Callx8 [runfn+36]`.
- `deliver_pending_events+0x28` (`0xcb04`): the companion RMW; clears delivered
  bits. A no-op when the global-pending mask is empty.
The large `s=1` block at `0x031xxx..0x03cxxx` (e.g. `_XAie_TileCtrlSetIsolation`,
`FUN_000387e8`) is the AIE-array HAL with struct-arg bases, not task pointers.

Sound bound (per review): "no writer fires" is airtight (runtime); statically,
the readiness writers are the event path. A writer via non-immediate-`0x30`
addressing cannot be excluded by `store_search` alone -- but the runtime watch
excludes ANY writer firing, so the conclusion stands: no non-event writer breaks
the paradox.

### Call-graph: the entire downstream chain is charted; only the origin is missing

`m2c_probe_call_xref XDNA_FW_XREF=0xd84c,0xcadc`:
- `wake_tasks_by_event_mask` (`0xd84c`) has exactly ONE direct caller:
  `FUN_00005580+0x289` (`0x5809`) -- the table-dispatched message dispatcher
  (Session-4). The readiness *setter* is reachable only through the event path.
- `deliver_pending_events` (`0xcadc`) has four direct callers, including
  `0xd82c` -- immediately after the dispatcher's done-flag check at `0xd828`. So
  the dispatcher runs `deliver_pending_events` EVERY cycle; it is a no-op only
  because the pending mask is never set.

**Synthesis -- the firmware ledger is closed.** The complete chain from "event
arrives" to "boot advances" is now charted and proven functional downstream
(force_done confirmed setting `[0x10f40]` cascades boot 58k->623k): event ->
`FUN_00005580` (message dispatcher) -> `wake_tasks_by_event_mask(1<<id)` -> sets
pending mask `[0x10f40]` + global-pending `[SCHED+108]` -> next dispatcher cycle
`deliver_pending_events` delivers -> task readied -> loop exits. The delivery
machinery runs constantly with nothing to deliver. The firmware provably cannot
originate the event: it never self-generates it and never arms an interrupt to
receive one (ii-b). **The event must come from outside the firmware's
self-driven trajectory -- from the array's per-column completion, delivered via
the message-dispatcher table (mailbox/doorbell), NOT via an interrupt the
firmware sets up.** This confirms the pivot to H-b array integration and hands
us the integration seam: the array must ultimately drive `FUN_00005580` (or the
mailbox/table that dispatches it) with the completion event id. The open H-b
design question is now precise -- what does the array write, into what
mailbox/doorbell, that dispatches `FUN_00005580` with which event id -- rather
than the diffuse "what makes the loop exit."
