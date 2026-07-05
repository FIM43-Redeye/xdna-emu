# M2c iter9: FLIX xt_format2 decode -- reversed encoding + collapse model

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2. Implements the design from
`docs/superpowers/specs/2026-07-05-m2c-flix-bundle-decode-design.md` (approved:
collapse-to-single-op). Clears the wall at runtime `0x28b4` (the exception
dispatcher's entry FLIX bundle). Commit `ba0b8203`.

## Result

The boot now decodes the `xt_format2` bundle at `0x28b4`, executes the whole
exception dispatcher (`0x28b4`..`0x291c`), and returns through the `retw.n`
chain into the C runtime -- ~40 instructions further. Next wall: base-ISA
`break 0x1, 0xf` at `0x2000e035` (iter10). Suite 3942/0.

## The reversed encoding (all derived + byte-verified)

Source: this core's real ISA config (`binutils-2.37/bfd/xtensa-modules.c`,
`xt_format2` / `xt_flix64_slot0` / `xt_flix64_slot3`), read as reference;
every value byte-verified against the `xtdis` oracle
(`build/experiments/firmware-re/xtdis/`, `XTDIS_RAW=1` dumps slot bits +
per-operand raw/decoded/`undo_reloc` target). Nothing copied into the repo.

**Bundle -> two LE words** (libisa `insnbuf` packing): `insn0 = LE(bytes[0:4])`,
`insn1 = LE(bytes[4:8])`.

**slot0 (general slot, 24 bits)** = `(insn0 >> 4) & 0xffffff`. Its own opcode is
`op0 = (slot0 >> 20) & 0xf`. `op0 == 1` -> `l32r`: register `at = slot0 & 0xf`,
`imm16 = (slot0 >> 4) & 0xffff`. The literal-offset formula is IDENTICAL to
base-ISA `l32r`: `((pc+3)&~3) + ((0xFFFF0000|imm16)<<2)`. Verified: bundle
`ff f8 81 71 4e ff 68 51 @ 0x28b4` -> `l32r a15, [0xfffe3094]` (`imm16=0x81f8`,
`undo_reloc` target `0xfffe3094`).

**slot3 (branch/`excw` slot)** classification -- the useful simplification:
`op0_s6 = (insn1 >> 23) & 0x1f` (== `(slotbuf[0] >> 27) & 0x1f`) selects the op.
The 24 `.w18` branch cases (`op0_s6` 1..=24) ALL share the identical guard
`(slotbuf[1] & 7) == 0` where `slotbuf[1] = (insn1 >> 28) & 0x7`. `op0_s6 == 25`
is `nop`. So:

```
real .w18 branch  <=>  op0_s6 in 1..=24  AND  (insn1 >> 28) & 7 == 0
nop               <=>  op0_s6 == 25
excw (default)    <=>  otherwise   (incl. guard-fail)
```

`nop` and `excw` are both inert. The dispatcher bundle's slot3 has `op0_s6 == 2`
(would be `bbsi.w18`) but `slotbuf[1] == 5` so the guard fails -> `excw`. That
guard-fail-to-excw is exactly why a naive `op0_s6`-only check would have
false-walled the `l32r` bundle.

## Collapse model (why no new execute code)

Every observed bundle is one real op + one inert slot, so bundle execution is
never actually parallel. Decode collapses `(Slot0::Real(op), Slot3::Inert)` to
`op` with `len = 8`; the existing per-op `interp` runs it unchanged
(`mem::exec` advances `pc += len`). Anything else -- unrecognized slot0, a real
`.w18` branch in slot3 (not yet implemented; also, with a real slot0, a
two-real-op bundle) -- walls as `Op::Unknown` rather than silently dropping a
slot. `op0 == 0xe` (`xt_format1`, 3-slot) never occurs and walls too.

Length rule updated at all three fetch sites (`decode::decode`, `interp::step`,
`fastpath::decode_at`): `op0` 0xe/0xf -> 8 bytes.

## Next (iter10)

`break 0x1, 0xf` @ `0x2000e035` -- a base-ISA debug-breakpoint op the C runtime
reaches after the exception return. Decode it (and decide its execute semantics:
likely a `Debug`-cause exception or a benign no-op in our model) next.

## Oracle additions this iteration

`xtdis` gained an `XTDIS_RAW=1` mode (slot bits + per-operand raw/decoded/
`undo_reloc` target) -- the reversing view that pinned the field positions and
the `l32r` target base. `undo_reloc` (disassembler direction) is the correct
call for absolute targets; `do_reloc` is the assembler direction and gives the
wrong value.
