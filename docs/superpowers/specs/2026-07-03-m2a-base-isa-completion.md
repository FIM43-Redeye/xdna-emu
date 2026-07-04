# M2a: Xtensa Base-ISA Completion -- Design Spec

**Date:** 2026-07-03
**Status:** Approved (brainstorm), pending plan
**Milestone:** M2a -- first of three sub-projects splitting the original M2
("boot past the MMU wall to command-loop idle").

## Context

The firmware interpreter (`src/firmware/`, merged to master at `1bca6444`)
decodes and executes 21 Xtensa opcodes -- enough to run the real management
firmware from its reset entry (`0x320`) through the 42-instruction MMU-init
prologue to the `jx` into virtual space. Reaching the command-loop idle is
blocked by two independent things: the Xtensa MMU is unmodeled, and most of the
base ISA the command-loop code uses is undecoded.

The original M2 was split into three sequential sub-projects so the
deterministic, derive-from-toolchain work is isolated from the speculative RE:

| # | Sub-project | This spec |
|---|-------------|-----------|
| **M2a** | Base-ISA completion (this doc) | full base-ISA opcode coverage |
| M2b | Xtensa MMU mechanism (`mmu.rs`) | faithful TLB + autorefill walk |
| M2c | Mapping reconstruction & boot-to-idle | reconstruct PSP's V->P map, reach idle |

**Dependency:** M2a -> M2b -> M2c. M2a is independent of the MMU (loads/stores
hit the flat `Bus` directly) and is the prerequisite for *validating* M2c
end-to-end (you cannot tell translation works without running real code through
it). Isolating M2a means that in M2c any stall is unambiguously a mapping bug,
never a missing opcode.

## Goal

The interpreter decodes and executes the **full set of base-ISA opcodes that
appear in the firmware image**, so that -- given correct memory -- real firmware
code runs coherently with no `Unknown` on any real instruction.

## Completeness bar: full-image coverage

The bar is **every distinct base-ISA mnemonic that appears in the firmware's
identified-code disassembly**, not just those on the boot path. The mnemonic set
is finite and derived: extracting mnemonics from the Ghidra listing
(`build/experiments/firmware-re/listing.txt`, Ghidra-identified code) yields
**108 distinct mnemonics**. 21 are already implemented; the remaining **~87 are
this milestone's scope**.

Rationale (chosen over boot-path-driven or full-documented-ISA): it front-loads
the mechanical work so M2c reconstruction never detours into opcode gaps, and it
stays finite/testable (unlike implementing undocumented-here ISA ops we have no
oracle vector for). Matches the "finish what you start / 100% coverage"
principle.

### The gap (~87 mnemonics), by category

Counts are occurrences in the identified code; they indicate importance, not
difficulty.

**Memory (8):** `s32i.n` (2403), `l8ui` (1503), `s8i` (658), `s32i` (532),
`l16ui` (35), `s16i` (33), `l16si` (4), `s32ri` (1).

**Arithmetic / logic (~37):** `add.n` (2299), `slli`, `addi`, `addi.n`, `and`,
`add`, `sub`, `addx4`, `addx2`, `addx8`, `subx8`, `srli`, `srai`, `srl`, `sll`,
`src`, `ssl`, `ssr`, `ssai`, `addmi`, `mov` (wide; canonical `or as,as`),
`movnez`, `moveqz`, `mull`, `mul16s`, `mul16u`, `rems`, `remu`, `quou`, `xor`,
`sext`, `nsau`, `neg`, `abs`, `min`, `minu`, `maxu`.

**Branches (27):** `j`, `beqz.n`, `bltu`, `beqz`, `bnei`, `bnez`, `beq`, `bne`,
`bgeu`, `beqi`, `bnez.n`, `bgeui`, `bltui`, `bbci`, `bnone`, `bbsi`, `bltz`,
`bany`, `bgei`, `blti`, `blt`, `bge`, `bbc`, `bgez`, `bbs`, `bnall`, `ball`.

**Control / window (~7):** `loop` (178), `loopnez` (3), `rotw` (6), `waiti` (4),
`ret.n` (3), `call0` (1). (`j` counted under branches.)

**System / sync / cache (~10):** `memw` (115), `rsil` (94), `wur` (22),
`syscall` (22), `rsr` (14), `nop` (13), `rsync` (6), `nop.n` (2), `dii` (3),
`dhi` (1), `dhwbi` (1), `ihi` (1).

Although ~87 mnemonics, there are roughly a dozen real implementation patterns:
the 27 branches are variants of compare-and-PC-relative-jump; the arithmetic
family shares structure; the cache ops are logged no-ops like the existing
`isync`/`dsync`.

### Already implemented (21, out of scope)

`l32i.n`, `movi.n`, `mov.n`, `movi`, `l32i`, `l32r`, `or`, `extui`, `entry`,
`call8`, `callx8`, `retw`, `retw.n`, `witlb`, `wdtlb`, `iitlb`, `idtlb`, `wsr`,
`isync`, `dsync`, `jx`.

## Derivation sources

Two distinct oracles, kept distinct:

- **Encoding vectors** (which bits -> which op/operands): `xtensa-lx106-elf-objdump
  -D -b binary -m xtensa` for base ISA; the captured Ghidra `listing.txt` for
  anything objdump mis-decodes. Same method M1 used. Every op gets a real
  firmware byte vector.
- **Runtime semantics** (what the op *does*): QEMU `target/xtensa/translate.c`
  -- the authoritative open-source Xtensa semantics reference, read-and-
  reimplement (same well M2b draws the MMU from). This matters for the ops M1
  didn't need: the `addx`/`subx` shift-adds, the multiply/divide family, the
  shift ops' `SAR` interaction, `nsau`, `sext`, conditional moves, and the
  zero-overhead loop.

Every opcode carries a comment naming its encoding vector and (where non-obvious)
its semantics source, per the repo's source-derivation policy.

## Architecture

### Module layout ("split by category")

`decode.rs` is already 523 lines; adding ~87 ops to it and to `interp.rs` would
push both past what is comfortable to hold in context. Split both by category.
The refactor is behavior-preserving and done first (existing ops move into the
new modules, all tests stay green) before any new op is added.

```
src/firmware/xtensa/
  mod.rs            submodule wiring (updated for the new tree)
  regfile.rs        + LBEG / LEND / LCOUNT loop registers
  decode/
    mod.rs          Op enum (central), Decoded, decode() dispatcher
                    (op0 selection, narrow/wide length, dispatch to category)
    arith.rs        add/sub/addi/addx/subx/shifts/mul/div/logical/moves/misc
    mem.rs          l8ui/s8i/s32i/s32i.n/l16ui/s16i/l16si/s32ri (+ existing l32i/l32i.n/l32r)
    branch.rs       the 27 branch variants
    control.rs      j/call0/ret.n/loop/loopnez/waiti/rotw (+ existing call8/callx8/entry/retw/jx)
    system.rs       rsr/wsr/wur/rsil/syscall/nop/memw/cache-noops (+ existing wsr/sync/tlb)
  interp/
    mod.rs          Cpu, step() core, Step enum, exception raise (from M1.5)
    arith.rs mem.rs branch.rs control.rs system.rs   per-category exec fns
```

The `Op` enum stays in `decode/mod.rs` (a Rust enum cannot span files); every
decode/execute *arm* lives in its category file. `decode()` dispatches by `op0`
(and format) into the category decoders; `step()`'s match dispatches each `Op`
into the category exec functions.

### Interpreter-core changes

Only one structural change; everything else is per-op arms.

1. **Zero-overhead loop.** Add `LBEG`/`LEND`/`LCOUNT` to the register file.
   After each instruction retire that did **not** take a branch/jump, if
   `PC == LEND` and `LCOUNT != 0`, decrement `LCOUNT` and set `PC = LBEG`.
   `loop as,label` sets `LCOUNT = AR[s] - 1`, `LBEG = pc + insn_len`,
   `LEND = pc + insn_len + imm8`; `loopnez` adds a not-taken guard that skips
   past `LEND` when `AR[s] == 0` (`loopgtz` is not present in this firmware, so
   it is out of scope). Exact edge semantics (the `-1`, the `LEND` boundary, the
   interaction with a branch whose target is `LEND`) come from QEMU
   `translate.c`. This is the one op that touches the fetch/retire loop rather
   than just adding a match arm.

2. **No other structural change.** `waiti` returns the existing `Step::Wait`
   (the command loop likely parks here -- relevant to M2c's `reached_idle`).
   `syscall` raises through the M1.5 exception machinery (EXCCAUSE + EPC1 +
   vector to `vecbase`), currently dormant. `rotw` rotates `WINDOWBASE` via the
   existing `regfile` rotate. Loads/stores call the flat `Bus` directly -- no
   translation yet (M2b wraps a `translate()` in front).

### Per-category semantics notes

- **Memory:** byte/half/word loads and stores, little-endian, offset scaling per
  the format (word ops scale the imm by 4, half by 2). `l8ui`/`l16ui` zero-
  extend; `l16si` sign-extends. `s32ri` is a store-release (same store, memory
  ordering is a no-op in a single-threaded model but decoded/executed distinctly
  and commented as such).
- **Arithmetic:** two's-complement wrap (no traps). `addx2/4/8` = `(AR[s] << k)
  + AR[t]`; `subx` analogous. Shifts: `slli`/`srli`/`srai` take an immediate;
  `sll`/`srl`/`sra`/`src` take the shift amount from `SAR` (set by `ssl`/`ssr`/
  `ssai`/`ssa8*`). `mull` = low 32 bits of the product; `mul16s`/`mul16u` =
  16x16 with sign/zero extension; `rems`/`remu`/`quou` = signed/unsigned
  remainder and unsigned quotient (division-by-zero behavior per QEMU). `nsau`
  = normalize-shift-amount-unsigned (leading-zero count, 32 for zero). `sext` =
  sign-extend from a given bit. `moveqz`/`movnez` = conditional move on a third
  register. `min`/`minu`/`maxu` = signed/unsigned min/max.
- **Branches:** all PC-relative from the instruction, target = `pc + len +
  sign_extend(offset)`. Families: register-vs-register compare (`beq`/`bne`/
  `blt`/`bltu`/`bge`/`bgeu`), compare-vs-immediate (`beqi`/`bnei`/`blti`/`bgei`/
  `bltui`/`bgeui`), compare-vs-zero (`beqz`/`bnez`/`bltz`/`bgez` and the `.n`
  narrow forms), and bit-test (`bbci`/`bbsi`/`bbc`/`bbs`) / mask-test (`bnone`/
  `bany`/`ball`/`bnall`). Immediate branch tables (`beqi` etc.) use the Xtensa
  `B4CONST`/`B4CONSTU` encoded-constant tables -- transcribe from QEMU.
- **System:** `rsr`/`wsr`/`wur`/`rur` read/write special/user registers -- model
  the ones the firmware relies on, log-and-no-op the rest (extending the
  existing `wsr` router; pair `rsr` to return modeled values, 0 for unmodeled).
  `rsil` sets the interrupt level in `PS` and returns the old level. `memw`,
  `nop`, `nop.n`, and the cache ops (`dii`/`dhi`/`dhwbi`/`ihi`) are logged
  no-ops. `rsync` joins the existing sync group.

## Testing

Per-op TDD, same discipline as M1:

- **Decode test** per op: a real firmware byte vector (objdump/Ghidra) ->
  expected `Op` + length.
- **Execute test** per op: minimal register/memory setup -> expected result,
  semantics matched to QEMU `translate.c`. Branches tested **taken and
  not-taken**. Conditional moves tested both conditions. Multiply/divide tested
  with sign-boundary and (for div/rem) zero-divisor cases.
- **Zero-overhead loop**: a dedicated test that a short `loop` body iterates the
  right number of times and the loopback fires exactly at `LEND`, including the
  `LCOUNT == 0` fall-through.
- **Disambiguation regressions** where encodings are close (the branch sub-op
  selectors; `mov` vs `or`; the shift family's SAR vs immediate forms), in the
  spirit of M1's `jx`-vs-`callx8` and `ret.n`-vs-`retw.n` guards.

**Exit gate (firmware-gated, like the boot test):** a decode-coverage scan over
Ghidra's identified code ranges (`funcs.txt`) asserts **zero `Unknown` on real
instructions** -- empirical proof of full-image coverage -- modulo a documented
allowlist of data-in-code islands Ghidra mis-identified. Skips cleanly when the
firmware binary is absent.

## Scope boundaries

M2a explicitly does **not**:

- model the MMU or translate addresses (M2b; loads/stores use the flat `Bus`);
- reconstruct the V->P mapping or reach command-loop idle (M2c);
- execute exception *handlers* (the raise mechanism exists from M1.5; running a
  handler needs mapped vectors, which is M2b/M2c);
- implement Xtensa ops that do not appear in this firmware image (rejected
  "full documented ISA" option -- no oracle vector, untestable).

## Success criterion

All 108 firmware mnemonics decode and execute with QEMU-faithful semantics, each
covered by a hermetic decode + execute test; the zero-overhead loop works; the
coverage-scan exit gate is clean. Entirely independent of the MMU -- M2a stands
alone as "given correct memory, real firmware code runs coherently."
