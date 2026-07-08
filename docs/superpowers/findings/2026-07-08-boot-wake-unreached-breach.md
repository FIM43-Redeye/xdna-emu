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

## Probes used

`m2c_probe_addr_store_watch` (`XDNA_FW_WATCH_ADDR=0x22bc,0x10f40`),
`m2c_probe_store_search` (`XDNA_FW_STORE_DISP=0x6c`),
`m2c_probe_col_cmd_trace`, `m2c_probe_disasm_range`,
`m2c_probe_colassign_boot` (tail histogram), `m2c_probe_breach_doorbell`
(+bit3 shim), `m2c_probe_literal_xref`, `m2c_probe_mailbox_receive`
(canonical `boot_to_idle`).
