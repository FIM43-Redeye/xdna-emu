# Codex brief round 2 -- your premise-#5 verdict was REFUTED; what are we missing? (iter41)

You did the round-1 review (brief: `CODEX-BRIEF-boot-wall-iter40.md`, your verdict is
recorded in `docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md` at iter40's
tail / the iter41 section). You concluded the false premise was #5: "`0x2450` is data therefore
not code" -- that on this Harvard Xtensa, I-side `0x2450` holds a PSP-loaded executable switch
thunk we fail to map. **We verified that and it does not hold.** Default to skepticism again,
including at your own prior verdict. We care about finding the TRUE missing premise.

## Why premise-#5 (I-side code at 0x2450) is refuted -- evidence

Firmware image: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin` (size `0x3cb10`).
This image uses exactly TWO relocation deltas for low `.text`: base `+0x5c` (file = VMA + 0x5c)
and scattered `+0x100` overlay sections (file = VMA + 0x100), all documented in
`src/firmware/mod.rs` around lines 330-420 (LOW_TEXT_BLOCK, WINDOW_VECTOR, SYSCALL_BLOCK,
CTXSW_CALLEE etc.).

1. **No code backs VMA 0x2450 under either delta.** File `0x24ac` (=0x2450+0x5c) and file
   `0x2550` (=0x2450+0x100) are both zero-deserts (2 nonzero bytes in 32, sparse noise, not a
   function). Verified by reading the raw image bytes directly, not through the emulator overlay.
2. **`0x00002450` appears ZERO times as a 32-bit literal anywhere in the image.** Every real
   `+0x100` code section we have ever mapped was discovered because a compiled-in literal-pool
   pointer referenced its entry (`0xdac4`, `0x581c`, `0x588c`, ...). `0x2450` is referenced by
   nothing. It is never a designed code entry.
3. **`0x2450` is provably a DATA structure.** `FUN_0000d6c0+0xde` (`0xd79e Addmi a7,a3,512`)
   computes `a7 = SCHED(0x2250) + 0x200 = 0x2450`, then `0xd7a1 L8ui a8,a7,120` and
   `0xd7a7 S8i a4,a7,120` load/store a BYTE at `[0x2450+120]=[0x24c8]`. It is the per-priority
   ready-queue array base. `a7=0x2450` is data-pointer RESIDUE left in the physical register.

So there is no I-side thunk to recover. (Caveat we hold honestly: this assumes no THIRD,
unprecedented relocation delta exists for this VMA. If you think a third delta is plausible,
say how it would be evidenced.)

## What IS established (do not re-litigate unless you find a concrete error)

- The switch is the dispatcher `FUN_00002730` epilogue (a `+0x100` CTXSW section, VMA
  `0x2630..0x2bf5`). `0x2a5d..0x2a73` is a bulk register RESTORE: `a4..a15 := [a3+16..60]`, so
  `a7 := [a3+0x1c]` is just register 7 of the restored (outgoing) frame -- NOT a dedicated
  callback slot. `a3` = the OUTGOING (init) frame `0x12048` on pass 1.
- `0x2a7f Beq head,tail` (head=`[0x2b60]`=`0x12048` outgoing, tail=`[0x2b64]`=`0x15f18`
  incoming). head!=tail -> `0x2a82 head:=tail`, then `0x2a86 Call0 0xdf98` -> `Callx8 a7`.
- The trampoline `FUN_0000df8c` has a guard `0xdf94 BeqzN a7,0xdf9f` (a7==0 -> return, skip),
  but `Call0 0xdf98` enters PAST it, so the guard does not execute. Even if it did, init's
  `a7=0x2450 != 0` would fail it. So the guard is irrelevant to init's wall either way.
- `a7=0x2450` is faithfully computed (see #3) and faithfully spilled (`0x2925 S32iN a7,a3,28`,
  a plain a2..a12 save loop) and faithfully restored. The syscall vectors to REAL firmware
  handler code (`raise_general_exception -> VECBASE`), not an emulator shim.
- iter39 single-variable proof: overwrite ONLY `[0x12064]` (init frame+0x1c) from `0x2450` to
  a verified clean-leaf thunk address, change nothing else -> the hook calls+returns, reaches
  the no-hook rfe `0x2ae0`, rfe's into `0x08b041bc` (incoming entry), runs real task code.
  So the wall is PURELY that `a7` is a non-callable value.
- Real task frames (`0x15f18`, `0x15e78`) have `[+0x1c]=0`. We never observe a SECOND
  (task->task) switch in the whole boot -- only init->first-task.

## The reframed paradox

`a7=0x2450` is faithful data residue; there is no code at `0x2450`; the `Callx8 a7` genuinely
targets a non-code address; yet the identical firmware boots on real silicon. So on HW,
`Callx8 a7` must NOT execute with `a7=0x2450` at this point. By elimination, one of:

- **(A) a7 should be 0 (or callable) here -- a window/register-RESIDUE divergence.** Our boot
  takes ~114 register-window OVERFLOWS (the `0x880` handler; the finding doc reports
  "114 OVERFLOW : 0 UNDERFLOW"). If, between `FUN_0000d6c0` leaving `0x2450` in physical AR27
  (n=47394) and the hook (n=49472), a real HW window overflow/underflow spill+fill would
  overwrite that physical AR with a different value (or the handler clears it) but OUR model
  does not, HW's `a7` at the hook would differ from ours. This is your round-1 suspect (i),
  which you ruled out ASSUMING no overflow -- re-examine WITH the overflows.
- **(B) the hook should not fire -- head/tail or frame selection.** head is seeded from the
  outgoing frame before the pick; on HW maybe head==tail here (hook skipped) via a mechanism
  we mis-time. (Note: iter37b's "pre-commit current upstream" version of this was refuted by
  you via the `+0x25b` store `current->frame := outgoing`; a DIFFERENT mechanism may still hold.)
- **(C) the Call0 target / trampoline entry is wrong** -- we compute `0xdf98` (unguarded); maybe
  a `+0x100`-framing or target-computation error should land it at the guarded entry. (But even
  guarded, init's `a7=0x2450 != 0` walls, so this alone is insufficient unless combined with A.)
- **(D) something we have not modeled.** You are free to find a premise we did not enumerate.

## What we want

Investigate freely and tell us what we are missing. Concretely, the highest-value threads:
1. Trace physical AR27 (and the `a7` logical view) across init's boot from n~47394 to n~49472.
   Does a window overflow/underflow spill/fill touch it? Is our overflow (`0x880`) / underflow
   handler faithful to Xtensa `s32e`/`l32e`/`rfwo`/`rfwu` semantics and to WINDOWSTART updates?
2. If a7 is genuinely `0x2450` on HW too, then what makes the `Callx8` safe on HW that differs
   in our model? Attack the frame-selection (a3) and head/tail wiring.
3. Give a single decisive experiment to confirm the true premise, and if it is an emulator bug,
   name the specific instruction/mechanism.

## Tools (all gated `XDNA_FW_PROBE=1`, run via cargo from repo root)

- `m2c_probe_yield_window` -- window state + all 16 ARs at init's yield syscall; traces writes
  of `0x2450` into any logical AR (with the writing pc + windowbase).
- `m2c_probe_thunk_inject` (`XDNA_FW_INJECT=1`) -- the single-variable a7 experiment.
- `m2c_probe_disasm_range` (`XDNA_FW_DISASM=lo:hi`, optional `XDNA_FW_DISASM_OVL=lo:hi` for a
  +0x100 overlay view) -- static disasm of any VMA range.
- `m2c_probe_addr_store_watch` (`XDNA_FW_WATCH_ADDR=hex,hex`, `XDNA_FW_MAX=N`) -- watch stores.
- Interp: `src/firmware/xtensa/` (`decode/`, `interp/` incl. `control.rs` for entry/retw/rotw
  and window exceptions, `regfile.rs` for the 64-AR window model, `system.rs` for exceptions).
- Full narrative: `docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md`
  (iter35-iter41). Do NOT trust our conclusions -- verify against the probes and the image.

Deliverable: the TRUE missing premise (or the best-supported hypothesis + the experiment that
would confirm it), with concrete addresses and Xtensa semantics. Ruling a suspect out firmly is
as valuable as confirming one.
