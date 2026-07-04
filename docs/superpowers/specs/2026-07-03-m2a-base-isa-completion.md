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

## Completeness bar: identified-code coverage (+ boot region audited)

The bar is **every distinct base-ISA mnemonic that appears in the firmware's
executed code**. In practice this is derived from two regions:

- **The Ghidra-identified body (`0x2730..0x3ca0e`)** -- `listing.txt` yields
  **108 distinct mnemonics**.
- **The boot/reset region (`0x0..0x2730`)** -- Ghidra did NOT disassemble this,
  so it is not in `listing.txt`. It is audited separately by objdump. The
  executed reset/MMU prologue (`0x320..0x399`, and the reset head at `~0x200`)
  uses only `movi.n, wsr, isync, l32r, dsync, witlb, wdtlb, or, iitlb, idtlb,
  jx` -- **all already implemented** -- so it adds no new opcode. This is a real
  coverage obligation, not an assumption: the exit-gate scan (below) covers this
  range explicitly rather than trusting inspection.

**The count.** Of the 21 already-implemented ops, only 18 appear in
`listing.txt` -- `witlb`, `iitlb`, `jx` are boot-only (below `0x2730`). So
`108 - 18 = 90` listing mnemonics are unimplemented. One of those, wide `mov`, is
the canonical assembler form of `or as,at,at` and is **already decoded/executed**
by the existing `Or` arm (verified: `mov a4,a2` = bytes `20 42 20` ->
`Op::Or { r:4, s:2, t:2 }`). Dropping it leaves **89 opcodes needing new
decode/execute arms** -- this milestone's scope -- plus a disambiguation test
that `or` with `s == t` is what Ghidra prints as `mov`.

Rationale (chosen over boot-path-driven or full-documented-ISA): it front-loads
the mechanical work so M2c reconstruction never detours into opcode gaps, and it
stays finite/testable (unlike implementing undocumented-here ISA ops we have no
oracle vector for). Matches the "finish what you start / 100% coverage"
principle.

### The gap (89 opcodes), by category

Counts are occurrences in the identified code; they indicate importance, not
difficulty.

**Memory (8):** `s32i.n` (2403), `l8ui` (1503), `s8i` (658), `s32i` (532),
`l16ui` (35), `s16i` (33), `l16si` (4), `s32ri` (1).

**Arithmetic / logic (36):** `add.n` (2299), `slli`, `addi`, `addi.n`, `and`,
`add`, `sub`, `addx4`, `addx2`, `addx8`, `subx8`, `srli`, `srai`, `srl`, `sll`,
`src`, `ssl`, `ssr`, `ssai`, `addmi`, `movnez`, `moveqz`, `mull`, `mul16s`,
`mul16u`, `rems`, `remu`, `quou`, `xor`, `sext`, `nsau`, `neg`, `abs`, `min`,
`minu`, `maxu`. (Wide `mov` is excluded -- already covered by `Or`, see above.)

**Branches (27):** `j`, `beqz.n`, `bltu`, `beqz`, `bnei`, `bnez`, `beq`, `bne`,
`bgeu`, `beqi`, `bnez.n`, `bgeui`, `bltui`, `bbci`, `bnone`, `bbsi`, `bltz`,
`bany`, `bgei`, `blti`, `blt`, `bge`, `bbc`, `bgez`, `bbs`, `bnall`, `ball`.

**Control / window (6):** `loop` (178), `loopnez` (3), `rotw` (6), `waiti` (4),
`ret.n` (3), `call0` (1). (`j` counted under branches.)

**System / sync / cache (12):** `memw` (115), `rsil` (94), `wur` (22),
`syscall` (22), `rsr` (14), `nop` (13), `rsync` (6), `nop.n` (2), `dii` (3),
`dhi` (1), `dhwbi` (1), `ihi` (1).

Totals: 8 + 36 + 27 + 6 + 12 = **89** opcodes. Although sizeable, there are
roughly a dozen real implementation patterns: the 27 branches are variants of
compare-and-PC-relative-jump; the arithmetic family shares structure; the cache
ops are logged no-ops like the existing `isync`/`dsync`.

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

`decode.rs` is already 523 lines; adding 89 ops to it and to `interp.rs` would
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
    mod.rs          Cpu, step() core, Step enum, window-exc raise (M1.5)
                    + new general-exception raise (EXCCAUSE/EPC1/PS.EXCM)
    arith.rs mem.rs branch.rs control.rs system.rs   per-category exec fns
```

The `Op` enum stays in `decode/mod.rs` (a Rust enum cannot span files); every
decode/execute *arm* lives in its category file. `decode()` dispatches by `op0`
(and format) into the category decoders; `step()`'s match dispatches each `Op`
into the category exec functions.

### Interpreter-core changes

Two structural additions (the loop machinery and a general-exception raise
path); everything else is per-op arms.

1. **Zero-overhead loop.** Add `LBEG`/`LEND`/`LCOUNT` to the register file.
   After each instruction retire that did **not** take a branch/jump, if
   `PC == LEND` and `LCOUNT != 0`, decrement `LCOUNT` and set `PC = LBEG`.
   `loop as,label` sets `LCOUNT = AR[s] - 1`, `LBEG = pc + 3` (the `loop`
   instruction's length), and **`LEND = pc + 4 + imm8`**. Note the deliberate
   Xtensa asymmetry: `LBEG` is `pc + 3` but `LEND` is `pc + 4 + imm8`, **not**
   `pc + 3 + imm8`. This is verified against the real firmware: all 181 `loop`
   instructions in the image match `LEND = pc + 4 + imm8` and zero match
   `pc + 3 + imm8` (e.g. `loop` at `0x3f8d`, bytes `76 84 07`, imm8=`0x07` ->
   Ghidra `LEND = 0x3f98 = 0x3f8d + 4 + 7`). Getting this wrong makes every
   loop body one byte short and breaks all 178 firmware loops. `loopnez` adds a
   not-taken guard that skips past `LEND` when `AR[s] == 0` (`loopgtz` is not
   present in this firmware, so it is out of scope). Cross-check the remaining
   edge semantics (the `-1`, the `LEND` retirement check, a branch whose target
   is `LEND`) against QEMU `translate.c`. This is the one op that touches the
   fetch/retire loop rather than just adding a match arm.

2. **General-exception raise.** `syscall` (22 uses) needs a raise path that does
   NOT exist yet. M1.5 built only `raise_window_exception`, which vectors
   through the window-vector table for overflow/underflow and uses a synthetic
   internal cause, not an architectural `EXCCAUSE` -- there is no `EXCCAUSE` SR
   modeled at all. M2a adds: an `EXCCAUSE` special register, and a
   general-exception raise (`EXCCAUSE <- 1` for `syscall`, `EPC1 <- pc`,
   `PS.EXCM <- 1`, `PC <- vecbase + user-exception-vector-offset`). In M2a the
   vector target is unmapped (pre-MMU), so the unit test asserts the **raised
   machine state** (EXCCAUSE/EPC1/PS.EXCM/PC), not execution past the vector.
   `rsr`/`wsr` gain `EXCCAUSE` as a modeled SR.

3. **No other structural change.** `waiti` returns the existing `Step::Wait`
   (the command loop likely parks here -- relevant to M2c's `reached_idle`).
   `rotw` rotates `WINDOWBASE` via the existing `regfile` rotate. Loads/stores
   call the flat `Bus` directly -- no translation yet (M2b wraps a `translate()`
   in front).

### Per-category semantics notes

- **Memory:** byte/half/word loads and stores, little-endian, offset scaling per
  the format (word ops scale the imm by 4, half by 2). `l8ui`/`l16ui` zero-
  extend; `l16si` sign-extends. `s32ri` is a store-release (same store, memory
  ordering is a no-op in a single-threaded model but decoded/executed distinctly
  and commented as such).
- **Arithmetic:** two's-complement wrap (no traps). `addx2/4/8` = `(AR[s] << k)
  + AR[t]`; `subx8` = `(AR[s] << 3) - AR[t]` (minuend is the shifted `s`, not
  `t`; only `subx8` is present in the image). Shifts: `slli`/`srli`/`srai` take an immediate;
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
  `rsil at,imm` writes the **full old `PS`** into `AR[t]` (`AR[t] <- PS`), then
  sets `PS.INTLEVEL <- imm` -- it returns all of `PS`, not just the level, since
  firmware saves and later restores the whole register. `memw`,
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

**Exit gate (firmware-gated, like the boot test):** a decode-coverage scan that
asserts **zero `Unknown` on real instructions** across **both** regions -- the
Ghidra-identified ranges (`funcs.txt`, `0x2730+`) **and** the boot/reset region
(`0x0..0x2730`, which Ghidra skipped). The boot-region scan follows the actual
executed path (entry at `~0x200` through the `jx` at `0x399`) rather than a blind
linear sweep, so it audits real instructions, not the interleaved literal pools.
Modulo a documented allowlist of data-in-code islands. Skips cleanly when the
firmware binary is absent. This gate is what upgrades "identified-code coverage"
to genuine executed-code coverage.

## Scope boundaries

M2a explicitly does **not**:

- model the MMU or translate addresses (M2b; loads/stores use the flat `Bus`);
- reconstruct the V->P mapping or reach command-loop idle (M2c);
- execute exception *handlers* -- M2a adds the general-exception *raise*
  (EXCCAUSE/EPC1/PS.EXCM), but running a handler needs mapped vectors, which is
  M2b/M2c;
- run real firmware end-to-end -- there is no MMU, so execution cannot reach
  idle; execution correctness in M2a is validated at the **unit level** (per-op
  vectors), not by an integration run;
- implement Xtensa ops that do not appear in this firmware image (rejected
  "full documented ISA" option -- no oracle vector, untestable).

## Success criterion

Every base-ISA opcode present in the firmware (the 89 new arms plus the 21
already implemented) decodes and executes with QEMU-faithful semantics, each
covered by a hermetic decode + execute unit test; the zero-overhead loop works
(with the verified `LEND = pc + 4 + imm8` boundary); the coverage-scan exit gate
is clean across both the identified body and the boot region. Execution
correctness is validated at the unit level (no end-to-end run pre-MMU). Entirely
independent of the MMU -- M2a stands alone as "given correct memory, every
firmware instruction decodes and each executes correctly in isolation."
