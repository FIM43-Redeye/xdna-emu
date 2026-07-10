# Codex adversarial brief -- the "every step is faithful yet it walls" paradox (iter40)

**Your job:** find which of my structural PREMISES is wrong. Default to skepticism.
I have traced a firmware boot wall to a point where every individual step checks
out as faithful to the firmware code, yet the result walls -- while the same
firmware boots on real silicon. That is a contradiction, so I am wrong about
*something*. Try to break each claim below. I care more about which premise is
false than about confirmation.

## Context (one paragraph)

xdna-emu runs the AMD XDNA NPU management firmware (Zephyr v3.7.1 RTOS + AMD's
"MERT" run-to-completion dispatcher) on an in-tree Xtensa interpreter against the
emulated AIE2 array. Boot livelocks: init (the bootstrap/idle context, TCB
`0x10f10`, prio `0xff`) yields to hand off to the first real task (TCB `0x10dfc`,
prio 6, entry `0x08b041bc`). During that ONE context switch the dispatcher's
epilogue executes `Callx8 a7` where `a7 = 0x2450`, which is a DATA address (not
code), and the CPU faults on an all-zeros "instruction" at pc `0x2450`. n=49473.

## The switch mechanism (all verified by running/observing)

- Dispatcher = `FUN_00002730`. Its epilogue, VMA `0x2a7b..0x2a86`:
  - `0x2a7b L32iN a0,a2,0` -> a0 = head = `[0x2b60]` = `0x12048` (init's frame)
  - `0x2a7d L32iN a1,a1,0` -> a1 = tail = `[0x2b64]` = `0x15f18` (incoming frame)
  - `0x2a7f Beq a0,a1,0x2ae0` -> if head==tail, skip the hook
  - `0x2a82 S32iN a1,a2,0` -> head := tail (advance head to incoming)
  - `0x2a84 L32iN a1,a3,4`
  - `0x2a86 Call0 0xdf98` -> the hook
- Trampoline `FUN_0000df8c` (target `0xdf98` enters PAST the guard):
  - `0xdf94 BeqzN a7,0xdf9f` -> **guard: if a7==0 skip the call, return 0**
  - `0xdf98 Callx8 a7` -> **call the outgoing task's saved a7 as a function**
  - `0xdf9d RetwN`
- So the design calls the OUTGOING task's saved a7 once per switch (a "switch-out
  callback"). Contract: outgoing a7 must be `0` (guard skips) OR a valid callable.

## The five faithful steps (each independently verified)

1. **a7 = 0x2450 is faithfully COMPUTED.** `FUN_0000d6c0` (the priority-indexed
   ready-queue manager) writes `0x2450` into a7 at `+0xe1` (`0xd7a1`), where
   `0x2450 = SCHED(0x2250) + 0x200` = the ready-queue array base (`a3+512`). It
   stays live in a7 through the yield.
2. **init yields with a7 = 0x2450.** At the yield-syscall (`0x8b043e1`,
   WINDOWBASE=5, WINDOWSTART=`0b101011`, PS=`0x60022`), a7 = `0x2450`. The yield
   wrapper `FUN_0x8b043cc` (`entry a1,96`) touches only a1/a2/a3, so it never
   clears a7.
3. **The dispatcher spills a7 -> frame+0x1c faithfully.** `[0x12064]` (init
   frame `0x12048` + `0x1c`) is written EXACTLY ONCE, at `FUN_00002730+0x1f5`
   (`0x2925 S32iN a7,a3,28`), a plain a2..a12 register-spill loop. `[frame+0x1c]`
   = a7 (offset 7*4). `[frame+0x50]` = saved PC (init's = `0x08b043e4`).
4. **The guard passes.** `0x2450 != 0`, so `BeqzN a7` does not skip -> `Callx8`.
5. **0x2450 is data, not code.** Disasm of `0x2440..0x2478` raw AND with a
   `+0x100` overlay is all zeros. `0x2450` is the ready-queue array base (RAM).

Also verified: the syscall vectors to REAL firmware handler code
(`raise_general_exception -> VECBASE + offset`), NOT an emulator shim. Real task
frames (`0x15f18`, `0x15e78`) have `[+0x1c] = 0` -> guard skips them cleanly.

**Single-variable proof the wall is exactly this (iter39).** Overwrite ONLY
`[0x12064]` from `0x2450` to the address of a verified ABI-correct clean-leaf
thunk (`entry a1,32; movi.n a2,0; retw.n` at `FUN_0003b4cc`), change nothing
else. Result: the hook calls+returns through the thunk, reaches the no-hook rfe
`0x2ae0`, and rfe's into `0x08b041bc` (the incoming task entry, exactly), then
runs real task code. Thunk-independent. So the wall is PURELY init's a7 value.

## The paradox

Every step above is faithful to the firmware, yet `Callx8 0x2450` walls -- and
the same firmware boots on silicon. One premise must be false. Candidates I rank:

- **(i) window/SR emulation divergence in the exception save/restore.** The
  spill reads a2/a4/a5 from EXCSAVE SRs (`Rsr sr213/214/210`) but a6/a7/a8..
  directly from the window. If our windowbase/windowstart handling or the
  vector's register preservation diverges from HW, the "a7" spilled/called could
  differ. (But I confirmed a7=0x2450 in init's OWN syscall window, so the spill
  matches init's live a7.)
- **(ii) head/tail semantics.** The restored/called frame is the OUTGOING (init)
  frame -- structurally odd for a "switch-out callback." Maybe head should equal
  tail here (hook skipped), and our head-advance timing is wrong. (iter37b's
  "pre-commit current upstream" version of this was REFUTED by you via the
  `+0x25b` store `current->frame := outgoing`; but the weaker "head should be
  incoming at the Beq" may still hold via a different mechanism.)
- **(iii) init is special.** init is the scheduler/main/bootstrap context, not a
  normal task. Maybe its a7 at this yield SHOULD be a valid continuation/callback
  that the code we execute never installs -- i.e. a missing bootstrap-frame
  construction step, and on HW init's a7 here is NOT `0x2450`.

## What I want from you

1. Which premise is false? Attack each. In particular: is it PLAUSIBLE that on
   real HW init's a7 at this exact yield is NOT `0x2450` -- and if so, by what
   mechanism (window preservation? a different init code path? a bootstrap step)?
2. Is the "switch calls the OUTGOING task's saved a7" reading even correct, or is
   there a saner interpretation of `0xdf98 Callx8 a7` I'm missing (e.g. a7 is not
   the outgoing frame's spilled a7 but something the restore loads from the
   INCOMING frame / a TCB callback field)?
3. If (i): name the specific Xtensa window/SR mechanism most likely mis-modeled
   in a syscall-exception register save (WINDOWSTART spill semantics, EXCSAVE
   usage, `entry`/`rotw` in the handler, PS.EXCM/WOE interactions).

## Repro (all gated `XDNA_FW_PROBE=1`, from repo root)

- Window + a7 provenance at the yield:
  `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_yield_window -- --nocapture`
- Single-variable thunk inject (control vs inject):
  `XDNA_FW_PROBE=1 [XDNA_FW_INJECT=1] cargo test --lib m2c_probe_thunk_inject -- --nocapture`
- Disasm any VMA range (add `XDNA_FW_DISASM_OVL=lo:hi` for a +0x100 overlay view):
  `XDNA_FW_PROBE=1 XDNA_FW_DISASM=0x2a70:0x2ae6 cargo test --lib m2c_probe_disasm_range -- --nocapture`
- Watch stores to an address:
  `XDNA_FW_PROBE=1 XDNA_FW_WATCH_ADDR=0x12064 XDNA_FW_MAX=80000 cargo test --lib m2c_probe_addr_store_watch -- --nocapture`

Full narrative: `docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md`
(sections iter35-iter40). Do NOT trust my conclusions -- verify against the probes.
