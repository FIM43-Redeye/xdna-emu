# M2c iter17: the windowed-register ABI needs a general `window_check`, not just an entry-time overflow

**Date:** 2026-07-06
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Status:** RESOLVED -- PC=0 window-ABI wall cleared; boot 48144 -> full 200k budget (no wall), 1553 window exceptions.

## The wall

After iter16, boot walled at `Unknown pc=0x0 word=0x00000000`, reached by a
`retw.n` returning to address 0. The reaching trace (WINDOWBASE `wb`,
WINDOWSTART `ws` annotated):

```
48176 pc=0xc559 L32r  a8, [0x34c8]   wb=1 ws=0xaaaa   ; a8 = call target 0x8b0e710
48177 pc=0xc55c Callx8 a8             wb=1 ws=0xaaaa   ; a8 = ret 0x8000c55f, jump
48178 pc=0x8b0e710 Entry s=1,imm=16   wb=1 ws=0xaaaa   ; OVERFLOW -> 0x880 (Overflow8)
...   handler spills, rfwo, re-enters entry at 48189...
48214 pc=0x8b0e729 RetwN              wb=3 ws=0xaaaa   ; a0=0  -> return to PC=0
```

`ws=0xaaaa` with `wb=1` is a **completely full** register window: eight packed
`call8` frames, one WINDOWSTART bit at every odd quad (frames based at quads
1,3,5,...,15, each occupying two of the sixteen 4-register quads). At `wb=1`,
logical `a8` maps to physical register 12 = quad 3's `a0` -- which is the
**oldest live frame's saved a0** (the frame the overflow will spill).

## Root cause

My model checked window overflow **only inside `entry`**. That is too late.
The Xtensa windowed ABI requires the overflow to fire *before any instruction
writes a register in the a8..a15 range that overlaps a live older frame*.

In the trace:

1. `L32r a8` (48176) writes the call target into physical reg 12 -- the oldest
   frame's `a0` slot. No overflow fired, so the oldest frame's saved `a0` was
   already clobbered here.
2. `callx8 a8` (48177) then writes the return address `0x8000c55f` into the same
   slot and jumps.
3. `entry` (48178) finally detects the full window and raises Overflow8. The
   handler spills quad 3 -- but quad 3 now holds the return address, not the
   oldest frame's real `a0`. The handler's `l32e a0` scratches the register to
   0, then `rfwo` returns.
4. `entry` re-executes (48189) and rotates the callee's `a0` from that
   now-zeroed physical register. The callee runs and `retw.n` (48214) reads
   `a0=0` -> returns to PC=0.

The return address was written into a slot that was immediately spilled and
scratched, so it was lost.

## The fix: `gen_window_check` before every high-register access

Real Xtensa (and QEMU's `gen_window_check` -> `HELPER(window_check)`) run a
window check *before* an instruction whose highest register operand reaches
beyond the current frame. If a live older frame sits within `max_ar/4` quads
ahead of WINDOWBASE, it is spilled first; the faulting instruction re-executes
after `rfwo`. This makes the spill happen at the `L32r a8` (48176), preserving
the oldest frame's `a0`, and by the time `callx8`/`entry` run the window has
room -- no return-address clobber.

Implemented as:

- **`Op::max_ar()`** (`decode/mod.rs`): the highest AR (`a0..a15`) register
  operand each instruction touches. `r`/`s`/`t` are AR registers; `ft` (FP),
  `sr`/`ur` (special/user register *indices*), and all immediates are not. The
  windowed CALL family folds in its implicit `a[callinc*4]` return-address write
  (`a4`/`a8`/`a12`). `entry`/`retw`/`retw.n` return `None` (they derive their
  over/underflow from the *runtime* call size, PS.CALLINC / `a0[31:30]`, which a
  static field can't see); `rfwo`/`rfwu` return `None` (they only run with
  PS.EXCM set, where the check is suppressed). A wildcard-free `match` makes the
  compiler enforce coverage of all 127 variants.

- **`Cpu::window_check()`** (`interp/mod.rs`): given `max_ar`, computes
  `w = max_ar/4` and reuses the QEMU-faithful `RegFile::overflow_check(w)`
  (`n = ctz(replicate(ws) >> (wb+1)) + 1`; vector by `ctz(norm >> n)` ->
  Overflow4/8/12). Raises via `raise_window_exception`. **Suppressed unless
  window exceptions are enabled** (PS.WOE set, PS.EXCM clear) -- the spill
  handler itself uses `s32e` on a8..a15 and must not re-fault.

- **`Cpu::step()`**: after decode, before exec, runs `window_check(max_ar)` and
  returns straight through on a raise, so `pc` does not advance and the
  instruction re-executes after `rfwo`.

The pre-existing entry-time overflow check is kept: it uses the runtime
`callinc` that `max_ar` can't express, and in practice is pre-empted (the call's
own implicit `a[callinc*4]` write triggers `window_check` first), so it acts as
a runtime-size-aware backstop rather than the primary trigger.

## Verification

- `Op::max_ar`: unit tests for r/s/t / ft-excluded / sr-ur-excluded / call-family
  implicit register / self-windowing `None`.
- `Cpu::window_check`: full window spills before a `max_ar=8` access
  (Overflow8, `wb` 1->3, OWB=1, EXCM set, EPC1=faulting pc); suppressed under
  PS.EXCM; no-op for `max_ar<4` (current-frame quad).
- Boot (`m2c_boot_advances_into_c_runtime`): PC=0 wall gone; boot runs the full
  200,000-instruction budget with **no** `Unknown` wall and **1553** window
  exceptions -- the overflow/underflow handlers (`s32e`/`l32e` + `rfwo`/`rfwu`)
  round-trip correctly. Two regression guards added: `window_exceptions > 0`
  (ABI exercised) and `instrs_executed > 48_215` (past the old wall).
- Full suite: `cargo test --lib` 3978 -> (iter17) green.

## Next frontier (iter18)

Boot no longer walls but reaches the 200k budget without idling. The tail is a
repetitive init loop (`FUN_0000c530` called hundreds of times; `a4`/`a5` step
through memory `0x27271000 -> 0x27274000+` -- a region-init/copy loop making
forward progress). iter18 = determine whether this loop terminates (and where it
is going) or is a poll on peripheral/mailbox state the stub doesn't yet model.
Use `m2c_probe_peripheral_reads` to see what the loop reads.
