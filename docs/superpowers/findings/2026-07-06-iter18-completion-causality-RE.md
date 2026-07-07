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
