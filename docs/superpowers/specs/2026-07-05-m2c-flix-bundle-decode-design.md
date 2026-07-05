# M2c FLIX bundle decode/execute -- design

**Date:** 2026-07-05
**Status:** design approved (representation fork decided: collapse-to-single-op)
**Context:** Firmware-emulation dream, M2c Phase 2 (walk-and-stub boot-to-idle),
branch `feat/m2c-mapping-boot-to-idle`. Follows iter8's de-risking finding
(`docs/superpowers/findings/2026-07-05-iter8-flix-bundle-oracle.md`).

## Goal

Teach the Xtensa firmware interpreter to decode and execute the custom AMD FLIX
(VLIW) bundles the management core uses, so the boot walk gets past the
exception dispatcher's entry at runtime `0x28b4` (`ff f8 81 71 4e ff 68 51`) and
onward toward the mailbox idle loop. One wall per iteration, as with iter5-8.

## Established facts (oracle-confirmed, not assumed)

All from `build/experiments/firmware-re/xtdis/` (the FLIX-aware disassembler
built from this core's real ISA config, `binutils-2.37/bfd/xtensa-modules.c`;
config match confirmed exact against a known-good boot-trace region -- see the
iter8 finding).

- **Two bundle formats.** `op0` nibble (`byte0 & 0xf`) `0xe` -> `xt_format1`
  (8 bytes, 3 slots); `0xf` -> `xt_format2` (8 bytes, 2 slots). binutils'
  `length_table` maps both to 8-byte length.
- **Only `xt_format2` occurs.** Census: ~4-6 bundles in the whole firmware, all
  `xt_format2`, zero `xt_format1`.
- **Every bundle is one real op + one inert slot.** Inert = `excw` (Exception
  Wait: a pipeline/exception barrier with no register/memory/PC effect -> a
  functional nop in a non-pipelined interpreter) or the slot's designated nop.
  Real ops seen: `l32r`, `l32i`, `bbci.w18`. Because one slot is always inert,
  bundle execution is never actually parallel here.
- **Slots are bit-packed, not byte-aligned.** libisa extracts each slot's bits
  into a separate `slotbuf` and decodes with a slot-specific opcode table
  (`xtensa_format_get_slot` -> `xtensa_opcode_decode`). A slot is NOT a
  base-ISA byte-aligned instruction; it needs slot-specific field extraction,
  not the `op0` dispatch our base decoder uses.
- **Concrete vector, fully reversed (`XTDIS_RAW=1 xtdis 0x28b4 fff881714eff6851`):**
  `slot0 = l32r a15, <lit>`, `slot1 = excw`. In slot0's canonical bit
  serialization (`8f 1f 18`): register `at` = bits[0:3] (raw `0xf` -> a15);
  `imm16` = bits[4:19] (raw `0x81f8`); `decoded offset = sign_extend16(0x81f8)
  << 2 = 0xfffe07e0` -- the SAME offset formula as base-ISA `l32r`. So FLIX
  `l32r` reuses our existing `L32r` target arithmetic; only the field positions
  differ.

## Architecture: collapse to the single real op

Chosen over a faithful 2-slot representation (`Op::FlixBundle{slot0,slot1}`
with read-then-commit). Rationale: faithful across the entire observed surface
(a `{realop || inert}` bundle genuinely *is* its real op), it sidesteps
PC-ordering (exactly one op drives the PC), it reuses every existing per-op
`exec` and the loop-back machinery untouched, and the one dangerous case (two
real ops) is walled loudly rather than silently mishandled. YAGNI on
read-then-commit parallelism the firmware never exercises.

### Decode (`src/firmware/xtensa/decode/`)

1. **Length + fetch.** `op0 == 0xe || 0xf` -> an 8-byte bundle. Today both the
   `step()` fetch length rule and `decode::decode`'s length rule treat `0xe`/
   `0xf` as a single undecodable byte (len 1); both change to len 8 for these
   selectors. `step()`'s 3-byte `buf` grows to 8; the `fastpath::decode_at`
   helper's length rule updates in lockstep.
2. **Format split.** `op0 == 0xf` -> `xt_format2`, split the 64-bit bundle into
   slot0 and slot1 per the format's fixed slot boundaries (a small, derivable
   fact -- pulled from the config via the `XTDIS_RAW` slot-bit dump, not
   guessed). `op0 == 0xe` (`xt_format1`) -> `Op::Unknown` (never occurs;
   documented, not implemented -- if the boot ever hits one it walls loudly).
3. **Per-slot opcode recognition (walk-and-stub).** Decode each slot's opcode
   from its slot-specific encoding. Implement only the slot ops encountered
   (initially `l32r` + `excw`); an unrecognized slot opcode -> the whole bundle
   is `Op::Unknown` (wall, extend next iteration). This mirrors exactly how the
   ~150 base-ISA ops were built: one test vector per op, from the real firmware,
   verified against the oracle.
4. **Collapse rule.** Classify each slot as *inert* (`excw` or the slot's
   designated nop) or *real*.
   - exactly one real -> emit that slot's `Op` (e.g. `Op::L32r{t,target}`) with
     `len = 8`.
   - both inert -> emit `Op::Nop` with `len = 8` (a do-nothing bundle).
   - both real -> `Op::Unknown` (loud wall; never observed).

### Slot field extraction

Per real slot op, extract its operands from the slot's bit layout, derived via
the `XTDIS_RAW` dump and cross-checked against xtdis's decoded output. Where the
FLIX op's value semantics match a base-ISA op we already implement (confirmed
true for `l32r`: identical `sign_extend16(imm16) << 2` offset), reuse the
existing computation and only supply the FLIX field positions. This keeps the
FLIX decoder thin -- it is a bit-position remap over semantics we already own.

### Execute -- no new path

The collapsed op flows through the existing dispatch chain unchanged
(`mem::exec` / `arith::exec` / ... in `interp/mod.rs::step`), now with
`len = 8`:

- Non-branch real op (`l32r`/`l32i`): its `exec` tail sets `pc = pc + len =
  pc + 8`. Correct.
- Branch real op (`bbci.w18`): computes its target from `pc` via its own
  (FLIX-specific, derived-when-hit) offset formula; fall-through is `pc + len =
  pc + 8`. Correct, because it is the sole PC-driving op in the bundle.

The zero-overhead-loop back-edge check in `step()` already keys off
`self.pc == pc + len`, so it works with `len = 8` with no change.

## Derivation method + provenance

The slot layouts and per-op field positions are derived from
`binutils-2.37/bfd/xtensa-modules.c` (this core's auto-generated ISA config,
GPL) *as a reading reference only*, via the `xtdis` oracle's `XTDIS_RAW` dump
(`getenv("XTDIS_RAW")`: prints slot length, extracted slot bits, and per-operand
raw / decoded / pc-reloc'd values). The oracle links the GPL config at build
time and is gitignored recon scratch; **no GPL code or table is copied into the
emulator.** The emulator's FLIX decode is written from the hardware facts (bit
positions, offset formulas), exactly as every base-ISA op in `decode/` was.

## Testing

- **Decode unit test per slot op**, with the real firmware vector as the oracle
  (initially `l32r`: `fff881714eff6851 @ 0x28b4` -> `L32r{t:15, target:<lit>}`,
  `len:8`), asserting the collapse produced the right single `Op`.
- **Both-inert bundle** -> `Op::Nop`, `len:8`.
- **Both-real bundle** -> `Op::Unknown` (guard test; synthesize a two-real-op
  `xt_format2` word).
- **`xt_format1` (`op0 == 0xe`)** -> `Op::Unknown`, len 8 (documented wall).
- **`step()` fetch length**: an `op0==0xf` bundle straddling a page boundary
  fetches all 8 bytes / faults correctly (extends the existing per-byte
  translate-on-fetch test).
- **Integration**: the boot diagnostic (`m2c_probe_trace_to_wall`) walks PAST
  `0x28b4` into the dispatcher body (next wall becomes the next iteration's
  target).
- `cargo test --lib` stays green (no regression to the 3939 base-ISA points).

## Scope boundaries (YAGNI)

- No `xt_format1` (3-slot) support -- zero occurrences; walled.
- No `Op::FlixBundle` / no read-then-commit parallel-execute path -- one slot is
  always inert.
- No general FLIX slot-table port -- walk-and-stub only the ops hit.

## Walls surfaced iteratively

Walk-and-stub, one op per iteration like iter5-8:
1. `l32r` + bundle framing (this spec's Task 1) -> past `0x28b4`.
2. Next wall surfaces `l32i` / `bbci.w18` / other -> reversed via `XTDIS_RAW`
   when hit, added the same way.

## Open items / risks

- **`bbci.w18` target formula** is unreversed until its wall is hit (its `.w18`
  suffix implies an 18-bit offset; formula derived via `XTDIS_RAW` at that
  point). Does not block Task 1.
- **`l32r` target base** (`pc & ~3` vs the reloc direction) is pinned during
  implementation by cross-checking the FLIX decode against our existing
  base-ISA `L32r` and the real literal read; the offset *formula* is already
  confirmed identical to base ISA.
