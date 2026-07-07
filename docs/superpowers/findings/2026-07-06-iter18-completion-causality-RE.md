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
