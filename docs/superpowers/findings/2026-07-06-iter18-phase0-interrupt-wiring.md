# iter18 Phase 0 — firmware interrupt/mailbox wiring (RE findings)

**Date:** 2026-07-06
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Plan:** `docs/superpowers/plans/2026-07-06-firmware-interrupt-mailbox-delivery.md` (Task 0)
**Spec:** `docs/superpowers/specs/2026-07-06-firmware-interrupt-mailbox-delivery-design.md`

Reverse-engineering the async-event wiring the (C) interrupt+mailbox milestone
needs. Gathered by running the real firmware (`npu.dev.sbin`) on the in-tree
interpreter, post-Phase-1 (the level-1 interrupt mechanism is landed but inert —
nothing arms or pends an interrupt yet). All addresses are low-window VMAs ==
the PCs the trace probes print.

## How this was gathered (tooling built this iteration)

Three `XDNA_FW_PROBE`-gated probes in `src/firmware/mod.rs`, plus a durable
symbol overlay:

- **`m2c_probe_peripheral_reads`** — every MMIO/stub access deduped by issuing
  PC (region, addrs, values). Catches the mailbox block and doorbell (they are
  bus accesses); does NOT catch `wsr.intenable` (an SR write, internal to the
  CPU).
- **`m2c_probe_trace_to_wall`** — disassembly ring buffer + full register window
  at the wall. Now prints `symbol+0xNN` per PC via `nearest_symbol()`.
- **`m2c_probe_store_search`** (`XDNA_FW_STORE_DISP`) — image-wide static
  disassembly hunting stores at a displacement (default `0x30`). Reads via
  `fetch8(vaddr,vaddr)` over the way-6 identity region, so it covers
  never-executed code.
- **`m2c_probe_call_xref`** (`XDNA_FW_XREF`) — static direct-call cross-reference
  (immediate-target `call0/4/8/12`; not register-indirect `callx*`).
- **Symbol overlay**: `load_symbols()` layers a git-tracked
  `src/firmware/firmware-symbols.txt` on top of the gitignored Ghidra export, so
  semantic RE names persist. Add names there as RE proceeds.

## The wall (unchanged from iter18, now named)

Boot reaches the firmware's **function-pointer-driven cooperative task
scheduler** and spins in unbounded recursion. Named the dispatcher
`task_dispatcher` (`0xd7f0`). The recursion:

- `task_dispatcher` (`0xd7f0`): `entry`; `rsil a2, 2` (**raises INTLEVEL to 2** —
  a critical section that masks level-1 interrupts); loads the scheduler table
  `struct2 = 0x2250`, reads the current task `struct2[0x28] = 0x9040`.
- `0xd828`: `l32i.n a10, [a4+0x30]` — reads the task **done-flag** at
  `0x9070`; `0xd82a`: `beqz a10` — done-flag is 0, so it re-dispatches forever.
- No CPU store writes the done-flag during boot (iter18 store-watch; corroborated
  below).

## Phase-0 fields

| Field | Value | Evidence |
|---|---|---|
| Mailbox message block | `0x272001xx` (fields `170/174/178/17c/180/184/18b/190/194/19c/19d/19f`) | peripheral probe seq 924–935, PCs `0x2000d4ef`–`0x2000d51f` |
| Ring head/tail | `0x27010dxx` (reads `0d14`/`0d18`; write `0d6c=0x13bf8`) | peripheral probe seq 26/27/5524 |
| Event/status pages (doorbell candidate) | `0x2727n000` (n=0..4); early write `0x27270000=0x1b6` | peripheral probe seq 18; the `0x8c72` routine polls status bytes at `0x27271000/2000/3000/4000` bit-testing bit 3 |
| Idle `waiti` level | **`waiti 0`** (INTLEVEL 0) at the idle loop `0xc8cc` | iter18; consistent with `rsil 2` critical section elsewhere |
| Dispatcher masking | **INTLEVEL 2** (`rsil 2` at `0xd7f3`) | trace-to-wall |
| Boot mode at syscall | **UM=1** (user), INTLEVEL=2, EXCM=0, WOE=1 | `m2c_probe_low_window_code` syscall-PS capture |
| Current task struct `0x9040` | bare TCB: `[0x2c]=6` (state), `[0x30]=0` (done); everything else 0 | `XDNA_FW_DUMP_ADDR=0x9040` |
| Scheduler table `struct2` `0x2250` | `[0x00]=0x10f10` (other task), `[0x28]=0x9040` (current), return-addr/ptr fields | `XDNA_FW_DUMP_ADDR=0x2250` |
| INTENABLE bits | **NOT yet observed** — needs SR-write logging for `wsr.intenable` (0xE4) across the boot | (open) |
| `0x2958` cause-4 arm post-init | **NOT yet confirmed** — the shared exception handler's `EXCCAUSE==4` path | (open, Phase 1/2 gate) |

## The mailbox handshake (a distinct earlier phase)

Before the dispatcher recursion, the firmware builds a structured mailbox message
in `0x272001xx` (notably `0x27200180=0x8a00ff0` and `0x27200190=0x8b041bc` —
pointers; `0x27200170=0xf18` — a ring offset), then **polls `0x27200170`
hundreds of times, always reading back `0xf18`** (the value it wrote), across
many PCs (`0x2000d5xx`, `0x20004xxx`). That is the firmware waiting for the
other side (host) to advance the ring — which never happens in our emulation.

Open question whether this polled handshake is a separate gate *before* the
interrupt-driven idle loop, or part of the same scheduler cycle. (The 200k-instr
trace ends in the dispatcher recursion, not this poll, so the poll is either
bounded/abandoned or a different phase.)

## The completion-writer fork — UNRESOLVED (static hit the indirect-dispatch wall)

The pivotal question for Phase 2: **who sets the task done-flag `[task+0x30]`?**
Two faithful shapes (per the spec):
- **(i)** a CPU interrupt/event handler stores it;
- **(ii)** a DMA/peripheral writes DRAM directly, the interrupt only signalling.

Static store-search (`disp=0x30`): **63 stores** to `+0x30` across 590 functions.
Most are stack-frame stores (`s:1`). ~8 non-stack candidates sit in the
scheduler region: `FUN_0000c9dc`, `FUN_0000d134`, `FUN_0000d1e8`,
`FUN_0000d53c`, **`FUN_0000d84c`** (adjacent to the dispatcher), `FUN_0000e098`.
So the clean "nothing writes `+0x30`, therefore only DMA can" kill is **off the
table** — `+0x30` is a common offset and it is written in the scheduler region.

Call-xref of those candidates: most have **no direct callers** — reached via
`callx*` (function pointers), because this is a function-pointer-driven
scheduler. Only `FUN_0000d84c` has a traceable direct caller (`FUN_00005580` at
`0x5809`). Static direct-call analysis structurally can't trace the rest.

**Conclusion:** the fork is not resolvable by static analysis against this
indirect-dispatch design. It is now shape-(i)-with-a-named-candidate
(`FUN_0000d84c`) vs shape-(ii)-with-coincidental-`+0x30`-stores.

## The force-done-flag experiment — RESULT (model CONFIRMED)

`m2c_probe_force_done` force-writes `[current_task + 0x30] = 1` at the
dispatcher's check (`0xd828`) and observes. Result:

1. **Setting the done-flag unwinds the recursion.** A single force (on task
   `0x10f10`'s flag `0x10f40`) and `task_dispatcher` stops spinning. **The
   done-flag IS the causal blocker** — pulling it makes the scheduler proceed.
2. The firmware then runs a **full context-save / task-switch routine**
   (`FUN_0000e098`): saves EPC1 (sr 177), PS (sr 230), SAR, loop regs, user
   regs to a save area — textbook RTOS context switch. Healthy progress.
3. It advances **~575k more instructions** (to 623,096) then hits a **new wall:
   an undecoded op at `0xd900`**. xtdis identifies it: **`s32c1i a0, a5, 0x308`
   = S32C1I, the atomic store-conditional (compare-and-swap)** — a decode gap,
   not another event-wait. (Used for lock-free queue manipulation; plausibly on
   the genuine scheduler path.)

**Verdict:** the (C) event/interrupt model is on track — the done-flag is exactly
the lever, and the scheduler behaves like a real RTOS when it's set. The next
obstacle on the path to idle is mundane (implement S32C1I, then re-observe).

Caveats: the force is artificial (set at a point the real firmware may reach
differently), so the S32C1I wall *could* be partly a force artifact — but S32C1I
is a real instruction the scheduler likely uses regardless. We did NOT reach
`waiti 0` idle (hit the decode gap first), so the true path to idle is: the real
done-flag mechanism + clearing S32C1I (and any further gaps).

The completion-writer fork (shape i vs ii) remains open but is now LOWER stakes:
we know setting the flag works; the real writer can be settled during Phase 2
implementation, testing both.

## Still-open Phase-0 items (fold into the experiment / Phase 1)

- Observe `wsr.intenable` (SR 0xE4) writes across the boot → the actual INTENABLE
  bit(s).
- Confirm the `0x2958` handler's `EXCCAUSE==4` arm reaches real interrupt
  servicing post-init.
- Doorbell int-bit number + edge-vs-level trigger — likely not firmware-
  observable; assume a single dedicated bit, edge-triggered (spec calibration
  knob), validate against HW.
