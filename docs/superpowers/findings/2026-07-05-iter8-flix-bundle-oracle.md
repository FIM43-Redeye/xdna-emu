# M2c iter8: the custom opcode is a FLIX bundle; oracle built, config confirmed

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2. After iter7 corrected the
kernel exception vector offset (0x300 -> 0x2e0), the boot re-vectors to the real
dispatcher at runtime 0x28b4 and walls on its first instruction `ff f8 81`
(op0=0xf), which our decoder rejects. This finding is the de-risking pass.

## TL;DR

`ff f8 81 71 4e ff 68 51` is not a mystery opcode -- it is a **FLIX (VLIW)
bundle**. This core is a custom AMD Xtensa FLIX config: op0 nibble `0xe` ->
`xt_format1` (8-byte, 3-slot), `0xf` -> `xt_format2` (8-byte, 2-slot). We built a
ground-truth disassembler for the exact config, confirmed it matches the firmware,
and censused the FLIX footprint. It is tiny and simple.

## The oracle (de-risking done)

`build/experiments/firmware-re/xtdis/` (gitignored recon scratch): a ~90-line
libisa harness linking the firmware core's real ISA config
(`amd-unified-software/.../binutils-2.37/bfd/xtensa-modules.c`, 870 `flix64` refs)
through `bfd/xtensa-isa.c`. See its README for build/validation detail.

**Config match confirmed EXACT:** disassembling a known-good boot-trace region
(runtime 0x2000d79c) reproduces our own interpreter's decode operand-for-operand.
So the AIE-tools binutils-2.37 config *is* the mgmt core's config, FLIX slots
included -- we can *read* the bundle format from it rather than reverse-engineer it.

This also re-confirms the iter7 fix: 0x28b4 decodes as a fully coherent exception
dispatcher (reads a cause-indexed struct, `callx8`s a handler, `rsil`s, restores
PS). Ghidra placed it "mid-function" only because Ghidra can't decode FLIX.

## Census (whole firmware)

- **~4-6 FLIX bundles total.** 3 in the 589 Ghidra-identified functions + the
  dispatcher entry bundle + possibly a couple more in un-identified handlers.
- **All `xt_format2`** (2-slot). **Zero** `xt_format1` (3-slot).
- **Every bundle = one real op + one `excw`**: `l32r+excw`, `l32i+excw`,
  `excw+bbci.w18` (`excw` appears in either slot).
- The dispatcher at 0x28b4 has **exactly one** bundle (its entry); the rest is
  base-ISA our decoder already handles.

## `excw` is a functional nop

`excw` = Exception Wait, a pipeline/exception synchronization barrier: no register
write, no memory access, no PC change. In our functional (non-pipelined)
interpreter it is a **no-op**. Because the firmware uses `excw` as the bundle
filler, **bundle execution is never actually parallel here** -- one slot is always
the `excw` barrier, the other does the real work. (It is *not* the config's
designated `[NOP]` opcode; the firmware just uses it as filler.)

## Implementation plan (next: brainstorm the design)

Small and bounded:
1. Decode the 8-byte `xt_format2` bundle: extract slot0/slot1 (each slot has its
   own opcode subset -- e.g. slot0: l32r/l32i; slot1: excw/bbci.w18) from the config.
2. Execute the real slot op; treat `excw` as nop. Implement read-inputs-then-commit
   parallel semantics for safety even though the `excw`-filler pattern never
   exercises it.
3. Ops likely needed: `l32r`, `l32i` (slot-encoded variants of ops we already
   have), possibly `bbci.w18` (a custom branch) if the walk hits that bundle.
   Walk-and-stub surfaces exactly which -- FLIX bundles are rare, so the walk hits
   only a handful.

Open design question for the brainstorm: how faithfully to decode the slots --
full config-derived slot-field extraction vs. a minimal walk-and-stub decoder that
handles just the ops we encounter (faithfully decoding the bundle bytes either way).
