# M2a: Xtensa Base-ISA Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the in-tree Xtensa interpreter (`src/firmware/`) to decode and execute every base-ISA opcode present in the real management firmware -- the 89 opcodes not yet implemented -- so that, given correct memory, real firmware code executes coherently.

**Architecture:** Refactor the two growing files (`decode.rs`, `interp.rs`) into per-category modules, then add opcodes family by family, each with a real-firmware oracle vector and a hermetic test. Two non-mechanical additions: the zero-overhead loop machinery (`LBEG`/`LEND`/`LCOUNT` + a per-retire loopback check) and a general-exception raise path (`EXCCAUSE`/`EPC1`/`PS.EXCM`). No MMU -- loads/stores hit the flat `Bus` directly.

**Tech Stack:** Rust, the existing `src/firmware/xtensa/` interpreter, `xtensa-lx106-elf-objdump` (encoding oracle for base ISA), the captured Ghidra `listing.txt` (encoding oracle where objdump fails), QEMU `target/xtensa/translate.c` (runtime-semantics reference).

**Spec:** `docs/superpowers/specs/2026-07-03-m2a-base-isa-completion.md` (read it; this plan implements it).

## Global Constraints

- **Derive from the toolchain.** Every opcode's encoding comes from a real-firmware oracle vector (objdump for base ISA; Ghidra `listing.txt` where objdump prints `excw`). Every opcode's runtime semantics comes from QEMU `target/xtensa/translate.c` or the Xtensa ISA reference. Comment each op with its source, as the existing code does.
- **The firmware binary is not in the repo.** It is at `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin` (248592 bytes; file offset == firmware address; base-0 image including a 0x18-byte `$PS1` header). Unit tests are **hermetic** (byte-literal vectors, no file dependency). The exit-gate coverage scan is **firmware-gated** (skips cleanly when the binary is absent), like the existing `boots_real_firmware_from_pinned_entry` test.
- **No MMU in M2a.** Loads/stores call `Bus::load32/store32/load8/store8` directly on the flat address space. Address translation is M2b.
- **`LEND = pc + 4 + imm8`** for the zero-overhead loop -- NOT `pc + 3 + imm8`. This is the documented Xtensa asymmetry (`LBEG = pc + 3`), verified against all 181 `loop` instructions in the firmware. Getting it wrong breaks every loop.
- **Test discipline.** Per opcode: one decode test (bytes -> `Op` + length) and one execute test (state in -> state out). Branches tested taken AND not-taken. Conditional moves tested both conditions. Multiply/divide tested at sign boundaries and (div/rem) zero divisor. Run `cargo test --lib` after each task; a regression is a blocker.
- **No emoji. Commit after each task.** Commit messages end with the two-line footer used across this repo (`Generated using Claude Code.` / `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`).

## The oracle vector table (all 89 opcodes)

Every opcode below has a representative instruction taken verbatim from the firmware (`listing.txt`), giving real bytes, operands, and the address (for PC-relative ops). Use these as the decode-test vectors. For any op whose exact field layout is unclear, disassemble the bytes with `xtensa-lx106-elf-objdump -D -b binary -m xtensa` on a scratch file to confirm before writing the arm.

```
# op        bytes (LE)      disassembly (Ghidra)                @addr
s32i.n      69 c7           s32i.n a6,a7,0x30                    0x27b6
l8ui        22 02 00        l8ui   a2,a2,0x0                     0x279a
s8i         82 44 2c        s8i    a8,a4,0x2c                    0x2847
s32i        22 61 07        s32i   a2,a1,0x1c                    0x2736
l16ui       32 13 02        l16ui  a3,a3,0x4                     0x4089
s16i        42 57 02        s16i   a4,a7,0x4                     0x45bc
l16si       22 92 00        l16si  a2,a2,0x0                     0x9f65
s32ri       a2 ff 86        s32ri  a10,a15,0x218                 0x73e0
add.n       7a e8           add.n  a14,a8,a7                     0x2761
slli        20 33 01        slli   a3,a3,0x1e                    0x2750
addi        62 cf 27        addi   a6,a15,0x27                   0x3f55
addi.n      1b 22           addi.n a2,a2,0x1                     0x2820
and         30 72 10        and    a7,a2,a3                      0x2759
add         20 26 80        add    a2,a6,a2                      0x3f4f
sub         90 44 c0        sub    a4,a4,a9                      0x28b1
addx4       20 22 a0        addx4  a2,a2,a2                      0x27e4
addx2       20 32 90        addx2  a3,a2,a2                      0x42f5
addx8       50 52 b0        addx8  a5,a2,a5                      0x59e5
subx8       80 98 f0        subx8  a9,a8,a8                      0x4271
srli        40 42 41        srli   a4,a4,0x2                     0x2753
srai        a0 38 31        srai   a3,a10,0x18                   0x51bd
srl         a0 d0 91        srl    a13,a10                       0x90b5
sll         00 33 a1        sll    a3,a3                         0x5ace
src         80 28 81        src    a2,a8,a8                      0x8976
ssl         00 14 40        ssl    a4                            0x5acb
ssr         00 06 40        ssr    a6                            0x90b0
ssai        10 40 40        ssai   0x10                          0x3b8ed
addmi       22 d7 02        addmi  a2,a7,0x200                   0x27c5
movnez      50 a3 93        movnez a10,a3,a5                     0x2918
moveqz      40 f2 83        moveqz a15,a2,a4                     0x28b7
mull        50 73 82        mull   a7,a3,a5                      0x4914
mul16s      30 22 d1        mul16s a2,a2,a3                      0x3fb1
mul16u      d0 23 c1        mul16u a2,a3,a13                     0x4a67
rems        50 6a f2        rems   a6,a10,a5                     0x37bc9
remu        70 52 e2        remu   a5,a2,a7                      0xe524
quou        f0 2e c2        quou   a2,a14,a15                    0xa7d0
xor         80 85 30        xor    a8,a5,a8                      0x2887
sext        00 98 23        sext   a9,a8,7                       0x426b
nsau        20 f2 40        nsau   a2,a2                         0xe0bd
neg         70 80 60        neg    a8,a7                         0xa433
abs         20 21 60        abs    a2,a2                         0x3b8a9
min         20 2a 43        min    a2,a10,a2                     0x82f9
minu        90 94 63        minu   a9,a4,a9                      0x2819
maxu        40 57 73        maxu   a5,a7,a4                      0x9d91
j           46 02 00        j      0x27c5                        0x27b8
beqz.n      8c 52           beqz.n a2,0x27aa                     0x27a1
bltu        97 34 01        bltu   a4,a9,0x2819                  0x2814
beqz        16 88 04        beqz   a8,0x28c1                     0x2875
bnei        66 63 04        bnei   a3,0x6,0x27bb                 0x27b3
bnez        56 c3 fe        bnez   a3,0x42b6                     0x42c6
beq         57 17 14        beq    a7,a5,0x276e                  0x2756
bne         87 94 0f        bne    a4,a8,0x2820                  0x280d
bgeu        37 b2 09        bgeu   a2,a3,0x3ecd                  0x3ec0
beqi        26 66 02        beqi   a6,0x6,0x4619                 0x4613
bnez.n      dc 02           bnez.n a2,0x2911                     0x28fd
bgeui       f6 3a 02        bgeui  a10,0x3,0x42c0                0x42ba
bltui       b6 62 02        bltui  a2,0x6,0x3fc1                 0x3fbb
bbci        37 64 05        bbci   a4,0x3,0x3f9c                 0x3f93
bnone       77 0f 0f        bnone  a15,a7,0x4edf                 0x4ecc
bbsi        07 e8 8f        bbsi   a8,0x0,0x2820                 0x288d
bltz        96 19 03        bltz   a9,0x42a3                     0x426e
bany        a7 88 bc        bany   a8,a10,0x2820                 0x2860
bgei        e6 32 10        bgei   a2,0x3,0x60c5                 0x60b1
blti        a6 12 26        blti   a2,0x1,0xd7bc                 0xd792
blt         47 26 02        blt    a6,a4,0xdbf3                  0xdbed
bge         57 a4 3c        bge    a4,a5,0x5624                  0x55e4
bbc         47 5a 0e        bbc    a10,a4,0x5aaa                 0x5a98
bgez        d6 c5 00        bgez   a5,0x55d1                     0x55c1
bbs         37 d2 05        bbs    a2,a3,0x803a                  0x8031
bnall       67 c7 13        bnall  a7,a6,0x36b49                 0x36b32
ball        37 44 02        ball   a4,a3,0xdc2b                  0xdc25
loop        76 84 07        loop   a4,0x3f98                     0x3f8d
loopnez     76 93 05        loopnez a3,0x3b826                   0x3b81d
rotw        10 80 40        rotw   0x1                           0xe0fe
waiti       00 70 00        waiti  0x0                           0x52db
ret.n       0d f0           ret.n                                0xe173
call0       85 ec ff        call0  0xe098                        0xe1ce
memw        c0 20 00        memw                                 0x2741
rsil        20 62 00        rsil   a2,0x2                        0x2733
wur         30 e7 f3        wur    a3,VECBASE                    0xdbab
syscall     00 50 00        syscall                              0xdbae
rsr         30 e4 03        rsr    a3,INTENABLE                  0x892b
nop         f0 20 00        nop                                  0x4228
rsync       10 20 00        rsync                                0x8937
nop.n       3d f0           nop.n                                0x457c
dii         72 72 00        dii    a2,0x0                        0x42dd
dhi         62 72 00        dhi    a2,0x0                        0x3b83c
dhwbi       52 72 00        dhwbi  a2,0x0                        0x3b820
ihi         e2 72 00        ihi    a2,0x0                        0x3b859
```

## Xtensa instruction formats (reference for all decode tasks)

The existing decoder (`decode.rs`) already extracts these nibbles from the little-endian 24-bit `word` (`b0 | b1<<8 | b2<<16`): `op0 = b0 & 0xF`, `n1 = b0>>4`, `n2 = b1 & 0xF` ("s"), `n3 = b1>>4` ("r"), `n4 = b2 & 0xF` ("op1"), `n5 = b2>>4` ("op2"). Narrow ops (2 bytes) use `op0` in `0x8..=0xD`.

Formats you will touch:
- **RRR** (`op0=0`): `r`(n3)=dest, `s`(n2), `t`(n1); `op1`(n4)/`op2`(n5) select the op. Arith/logic/shift/mul, and the RSR/sync/TLB group.
- **RRI8** (`op0=2`): `r`(n3) selects sub-op; `t`(n1)=data reg, `s`(n2)=base reg, `imm8`=b2. Loads/stores and `addi`/`addmi`.
- **RRI4** / **RSR** (`op0=0`, `op1=3`): `rsr`/`wsr`/`xsr` -- SR number = b1, reg = n1. `wur`/`rur` -- UR number, `op1`/`op2` distinguish.
- **CALLN** (`op0=5`): `n`(bits5:4 of b0) selects call size; `imm18` PC-relative. `call0` is `n=0`.
- **BRI12** (`op0=6`): `j` (`m=n1&0x3 == 0`? -- confirm), `beqz`/`bnez`/`bltz`/`bgez` (`n=n1&0x3`), and `loop`/`loopnez`.
- **BRI8** (`op0=7`): register/immediate conditional branches (`beq`/`bne`/`blt`/`bltu`/`bge`/`bgeu`/`beqi`/`bnei`/`blti`/`bgei`/`bltui`/`bgeui`/`bbci`/`bbsi`/`bnone`/`bany`/`bnall`/`ball`/`bbc`/`bbs`), selected by `r`(n3).
- **Narrow** (`op0` in `0x8..=0xD`): `l32i.n`/`s32i.n` (0x8/0x9), `add.n`/`addi.n` (0xA/0xB), `movi.n`/`beqz.n`/`bnez.n` (0xC), `mov.n`/`ret.n`/`nop.n` (0xD ST3 group).

Do not trust this list's bit-level claims blindly -- each is a hint. Confirm every field against the op's oracle vector with objdump before committing its arm. The `word >> N` extraction the existing `l32r`/`call8`/`entry` arms use is the pattern to follow.

---

## Task 1: Module-split refactor (behavior-preserving scaffold)

**Files:**
- Create: `src/firmware/xtensa/decode/mod.rs`, `decode/arith.rs`, `decode/mem.rs`, `decode/branch.rs`, `decode/control.rs`, `decode/system.rs`
- Create: `src/firmware/xtensa/interp/mod.rs`, `interp/arith.rs`, `interp/mem.rs`, `interp/branch.rs`, `interp/control.rs`, `interp/system.rs`
- Delete: `src/firmware/xtensa/decode.rs`, `src/firmware/xtensa/interp.rs` (content moves into the new trees)
- Modify: `src/firmware/xtensa/mod.rs` (module declarations)

**Interfaces:**
- Produces: `decode::decode(bytes: &[u8], pc: u32) -> decode::Decoded` (unchanged signature); `decode::Op` (the central enum, unchanged variants); `decode::Decoded { op: Op, len: u8 }`. Category decode helpers, e.g. `arith::decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, word: u32) -> Option<Op>`, returning `None` when the sub-op is not in that category so `decode()` can try the next category or fall to `Op::Unknown`.
- Produces: `interp::Cpu`, `interp::Step`, and category exec fns, e.g. `arith::exec(cpu: &mut Cpu, op: &Op) -> Option<Step>` (returns `None` if `op` isn't this category's, so `step()` can dispatch to the next). Keep whatever the current `interp.rs` exposes to `mod.rs`/`FirmwareProcessor` identical.

This task is a pure refactor: no opcode is added, no behavior changes, every existing test passes unchanged.

- [ ] **Step 1: Read the current files.** Read `src/firmware/xtensa/decode.rs` and `src/firmware/xtensa/interp.rs` in full, plus `src/firmware/xtensa/mod.rs` and `regfile.rs`, to inventory the 21 existing ops and the current `step()`/`decode()` structure.

- [ ] **Step 2: Create the `decode/` tree.** Move `Op`, `Decoded`, `sign_extend`, and the top-level `decode()` skeleton (op0 dispatch, narrow/wide length, field extraction) into `decode/mod.rs`. Move the existing per-op arms into category files by this mapping:
  - `mem.rs`: `L32iN`, `L32i`, `L32r`
  - `arith.rs`: `MovN`, `MoviN`, `Movi`, `Or`, `Extui`
  - `control.rs`: `Entry`, `Call8`, `Callx8`, `Retw`, `RetwN`, `Jx`
  - `system.rs`: `Wsr`, `Isync`, `Dsync`, `Witlb`, `Wdtlb`, `Iitlb`, `Idtlb`
  - `branch.rs`: (empty for now; create with a `//! Branch-family decode.` header so the module exists)

  `decode/mod.rs` keeps the format skeleton and delegates the leaf construction to `arith::decode_*`/`mem::decode_*`/etc. helpers. Keep the `Op` enum and all its variants in `decode/mod.rs` (a Rust enum cannot span files); only the match *arms* move.

- [ ] **Step 3: Create the `interp/` tree.** Move `Cpu`, `Step`, `step()`, and the M1.5 window-exception raise into `interp/mod.rs`. Move the per-`Op` execute arms into the category files with the same mapping as Step 2. `step()`'s dispatch calls `mem::exec`, `arith::exec`, `control::exec`, `system::exec` in turn (each returns `Option<Step>`), preserving the exact current behavior including the `Op::Unknown` freeze.

- [ ] **Step 4: Move the tests with their code.** The `#[cfg(test)] mod tests` blocks in the old files move alongside the code they test (decode tests into the relevant `decode/*.rs`; any interp tests into `interp/*.rs`). Do not change any assertion.

- [ ] **Step 5: Wire `mod.rs`.** Update `src/firmware/xtensa/mod.rs` so `decode` and `interp` are directory modules (`pub mod decode;` / `pub mod interp;` with their own `mod.rs`), and re-export whatever `FirmwareProcessor` (in `src/firmware/mod.rs`) currently imports (`decode::{decode, Op, Decoded}`, `interp::{Cpu, Step}`, etc.) so no consumer outside `xtensa/` changes.

- [ ] **Step 6: Build and test.** Run `cargo build --lib > /tmp/m2a-t1-build.txt 2>&1` then `cargo test --lib firmware > /tmp/m2a-t1-test.txt 2>&1`. Read both files. Expected: clean build, and the same firmware test count as before the refactor (71) all passing. Zero behavior change.

- [ ] **Step 7: Commit.**
```bash
git add -A src/firmware/xtensa/
git commit -m "refactor(#140): split xtensa decode/interp into per-category modules

Behavior-preserving scaffold for M2a. No opcode added; all 71 firmware
tests pass unchanged.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 2: Memory opcodes (8)

**Files:**
- Modify: `src/firmware/xtensa/decode/mem.rs`, `src/firmware/xtensa/interp/mem.rs`
- Modify: `src/firmware/xtensa/decode/mod.rs` (add the new `Op` variants)

**Interfaces:**
- Consumes: `Bus::load8/load16?/load32/store8/store32` on the `Cpu`'s bus. NOTE: check `mmio.rs` for a `load16`/`store16`; if absent, compose from two `load8`/`store8` (little-endian) rather than adding to `Bus` (keep `Bus` changes out of M2a).
- Produces: `Op::{S32iN, L8ui, S8i, S32i, L16ui, S16i, L16si, S32ri}` and their exec arms.

Opcodes and vectors (from the table): `s32i.n` `69 c7`, `l8ui` `22 02 00`, `s8i` `82 44 2c`, `s32i` `22 61 07`, `l16ui` `32 13 02`, `s16i` `42 57 02`, `l16si` `22 92 00`, `s32ri` `a2 ff 86`.

Semantics (all little-endian; effective address = `AR[s] + (imm << scale)`):
- `l8ui at,as,imm8`: `AR[t] = zero_extend(mem8[AR[s]+imm8])`. scale 0.
- `s8i at,as,imm8`: `mem8[AR[s]+imm8] = AR[t] & 0xFF`. scale 0.
- `l16ui at,as,imm8`: `AR[t] = zero_extend(mem16[AR[s]+imm8*2])`. scale 1.
- `l16si at,as,imm8`: `AR[t] = sign_extend16(mem16[AR[s]+imm8*2])`. scale 1.
- `s16i at,as,imm8`: `mem16[AR[s]+imm8*2] = AR[t] & 0xFFFF`. scale 1.
- `s32i at,as,imm8`: `mem32[AR[s]+imm8*4] = AR[t]`. scale 2. (Sibling of the existing `l32i`.)
- `s32i.n at,as,imm4`: narrow store, `mem32[AR[s]+imm4*4] = AR[t]`. (Sibling of the existing `l32i.n`; imm4 pre-scaled by 4 like `l32i.n`.)
- `s32ri at,as,imm8`: store-release; identical store to `s32i` in a single-threaded model. Decode/execute as a distinct `Op` and comment that the release ordering is a no-op here.

Decode: `l8ui/s8i/s32i/l16ui/s16i/l16si/s32ri` are RRI8 (`op0=2`), selected by `r`(n3); confirm each `r` value from its vector (e.g. `22 61 07` -> n3 = `(0x61>>4)=... ` compute it and match to `s32i`). `s32i.n` is narrow `op0=9`.

- [ ] **Step 1: Write the failing decode tests** in `decode/mem.rs` `tests`, one per op, e.g.:
```rust
#[test]
fn decodes_s32i() {
    // listing.txt: `22 61 07` @0x2736 -> s32i a2,a1,0x1c
    let d = decode(&[0x22, 0x61, 0x07], 0x2736);
    assert_eq!(d.len, 3);
    assert!(matches!(d.op, Op::S32i { t: 2, s: 1, imm: 0x1c }), "got {:?}", d.op);
}
```
Write the analogous test for each of the 8 ops using its vector. For `imm`, store the **pre-scaled byte offset** if that matches the existing `l32i` convention (`l32i` stores `imm: b2*4`), so `s32i`'s `imm` field holds `0x1c` only if you keep it unscaled -- match whatever `l32i` does and assert accordingly. Decide once, be consistent, assert the concrete value.

- [ ] **Step 2: Run the decode tests to confirm they fail.** `cargo test --lib firmware::xtensa::decode::mem > /tmp/m2a-t2-dec.txt 2>&1`; read it. Expected: fail to compile (variants undefined) or assertion failures.

- [ ] **Step 3: Add the `Op` variants** to `decode/mod.rs` (`S32iN{t,s,imm}`, `L8ui{t,s,imm}`, `S8i{t,s,imm}`, `S32i{t,s,imm}`, `L16ui{t,s,imm}`, `S16i{t,s,imm}`, `L16si{t,s,imm}`, `S32ri{t,s,imm}`), each with a `///` doc naming the vector. Add the decode arms in `decode/mem.rs` (RRI8 `r`-selector + narrow `op0=9`).

- [ ] **Step 4: Run decode tests to pass.** Same command as Step 2. Expected: all 8 decode tests pass.

- [ ] **Step 5: Write the failing execute tests** in `interp/mem.rs` `tests`. Set up a `Cpu` with a small backing memory (follow the pattern the existing `interp` tests use for `l32i`), write known bytes, run the op, assert the register/memory result. Example:
```rust
#[test]
fn executes_l8ui_zero_extends() {
    let mut cpu = /* test Cpu with bus */;
    cpu.regs.set_ar(3, 0x100);              // base
    cpu.bus.store8(0x104, 0xF7);            // byte to load
    step_op(&mut cpu, Op::L8ui { t: 2, s: 3, imm: 4 });
    assert_eq!(cpu.regs.ar(2), 0x0000_00F7); // zero-extended, not sign
}
```
Include a `l16si` test whose byte has bit 15 set (assert sign extension) and a `s32i`/`l32i` round-trip test.

- [ ] **Step 6: Run execute tests to confirm they fail**, then implement the exec arms in `interp/mem.rs` (`mem::exec`), then run to pass. `cargo test --lib firmware::xtensa::interp::mem > /tmp/m2a-t2-exec.txt 2>&1`.

- [ ] **Step 7: Full suite + commit.** `cargo test --lib > /tmp/m2a-t2-all.txt 2>&1`; read it; expect no regressions. Commit:
```bash
git add -A src/firmware/xtensa/
git commit -m "feat(#140): M2a memory opcodes (l8ui/s8i/s32i/s32i.n/l16ui/s16i/l16si/s32ri)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 3: Arithmetic -- integer, logical, conditional-move, min/max (19)

**Files:** Modify `decode/arith.rs`, `interp/arith.rs`, `decode/mod.rs` (variants).

**Opcodes (vectors in the table):** `add` `20 26 80`, `add.n` `7a e8`, `addi` `62 cf 27`, `addi.n` `1b 22`, `addmi` `22 d7 02`, `sub` `90 44 c0`, `addx2` `20 32 90`, `addx4` `20 22 a0`, `addx8` `50 52 b0`, `subx8` `80 98 f0`, `and` `30 72 10`, `xor` `80 85 30`, `neg` `70 80 60`, `abs` `20 21 60`, `movnez` `50 a3 93`, `moveqz` `40 f2 83`, `min` `20 2a 43`, `minu` `90 94 63`, `maxu` `40 57 73`.

**Semantics** (two's-complement wrap; no traps; verified against the Xtensa ISA):
- `add ar,as,at`: `AR[r] = AR[s] + AR[t]`. `add.n` same, narrow.
- `sub ar,as,at`: `AR[r] = AR[s] - AR[t]`.
- `addi at,as,imm8`: `AR[t] = AR[s] + sign_extend8(imm8)`. `addi.n` narrow, imm is a small encoded constant (confirm from vector).
- `addmi at,as,imm8`: `AR[t] = AR[s] + (sign_extend8(imm8) << 8)`.
- `addx2/4/8 ar,as,at`: `AR[r] = (AR[s] << k) + AR[t]`, k = 1/2/3.
- `subx8 ar,as,at`: `AR[r] = (AR[s] << 3) - AR[t]` (minuend is the shifted `s`; only `subx8` appears in the firmware).
- `and ar,as,at`: bitwise AND. `xor ar,as,at`: bitwise XOR. (`or` already exists.)
- `neg ar,at`: `AR[r] = -AR[t]` (two's complement). `abs ar,at`: `AR[r] = |AR[t]|` as signed.
- `moveqz ar,as,at`: `if AR[t] == 0 { AR[r] = AR[s] }` else unchanged. `movnez`: `if AR[t] != 0 { AR[r] = AR[s] }`.
- `min ar,as,at`: signed min. `minu`: unsigned min. `maxu`: unsigned max. (`max` signed is absent.)

Decode: all RRR (`op0=0`) except `add.n` (`op0=0xA`), `addi.n` (`op0=0xB`), selected by `(op1,op2)`; `addi`/`addmi` are RRI8 (`op0=2`, `r`-selector). Confirm each selector from its vector.

- [ ] **Step 1: Decode tests** (one per op, `decode/arith.rs`), using the vectors; assert exact `Op` + len. Example:
```rust
#[test]
fn decodes_addx4() {
    // listing.txt: `20 22 a0` @0x27e4 -> addx4 a2,a2,a2
    let d = decode(&[0x20, 0x22, 0xa0], 0x27e4);
    assert!(matches!(d.op, Op::Addx4 { r: 2, s: 2, t: 2 }), "got {:?}", d.op);
}
```
- [ ] **Step 2: Run to fail.** `cargo test --lib firmware::xtensa::decode::arith > /tmp/m2a-t3-dec.txt 2>&1`.
- [ ] **Step 3: Add variants + decode arms** (`Op::{Add,AddN,Addi,AddiN,Addmi,Sub,Addx2,Addx4,Addx8,Subx8,And,Xor,Neg,Abs,Moveqz,Movnez,Min,Minu,Maxu}`), each `///`-documented with its vector.
- [ ] **Step 4: Run decode to pass.**
- [ ] **Step 5: Execute tests** (`interp/arith.rs`), each asserting the semantic. Include: `addx4` result, `subx8` operand order (distinct `s`/`t` values so the order is pinned), `moveqz` both conditions, `movnez` both conditions, `abs` on a negative, `neg`, `addmi` shift, `minu` vs `min` on a value with the high bit set (so signed/unsigned differ).

- [ ] **Step 5b: `mov`-vs-`or` disambiguation test.** Ghidra prints `or as,at,at` (an `or` with `s == t`) as the pseudo-op `mov`. Wide `mov` is therefore NOT a separate opcode -- it is already covered by the existing `Or` arm. Add a decode test in `decode/arith.rs` asserting that `mov a4,a2` (bytes `20 42 20`) decodes to `Op::Or { r: 4, s: 2, t: 2 }` and executes to `AR[4] = AR[2]`, documenting that this is what the disassembler calls `mov`. (This closes the spec's note that wide `mov` is excluded from the gap.)
- [ ] **Step 6: Run to fail, implement `arith::exec` arms, run to pass.** `cargo test --lib firmware::xtensa::interp::arith`.
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a arithmetic/logical/cmov/minmax opcodes`).

---

## Task 4: Arithmetic -- shift and extract family (11)

**Files:** Modify `decode/arith.rs`, `interp/arith.rs`, `decode/mod.rs`. Uses the existing `SAR` register in `regfile.rs`.

**Opcodes (vectors):** `slli` `20 33 01`, `srli` `40 42 41`, `srai` `a0 38 31`, `sll` `00 33 a1`, `srl` `a0 d0 91`, `src` `80 28 81`, `ssl` `00 14 40`, `ssr` `00 06 40`, `ssai` `10 40 40`, `sext` `00 98 23`, `nsau` `20 f2 40`.

**Semantics** (verified: `sll` uses `SAR` set by `ssl`; `srl` uses `SAR` set by `ssr`):
- `slli ar,as,imm`: `AR[r] = AR[s] << imm` (imm 1..31; note the encoded imm layout -- confirm from vector, the shift amount is split across fields).
- `srli ar,as,imm4`: `AR[r] = AR[s] >> imm4` (logical, 0..15).
- `srai ar,as,imm`: `AR[r] = (AR[s] as i32) >> imm` (arithmetic).
- `ssl as`: `SAR = 32 - (AR[s] & 31)` (sets up a **left** shift by `AR[s]`). `sll ar,as`: `AR[r] = AR[s] << (32 - SAR)` -- i.e. left-shift by the amount `ssl` encoded. Cross-check the exact `SAR`/`sll` relation in QEMU `translate.c`; the invariant to preserve is `ssl(n); sll` == `AR[s] << n`.
- `ssr as`: `SAR = AR[s] & 31` (right shift). `srl ar,at`: `AR[r] = AR[t] >> SAR` (logical). Invariant: `ssr(n); srl` == `AR[t] >> n`.
- `ssai imm`: `SAR = imm` (set SAR to an immediate 0..31).
- `src ar,as,at`: funnel shift right -- `AR[r] = ((AR[s]:AR[t]) as u64 >> SAR) & 0xFFFFFFFF` (concatenate `s` high, `t` low, shift right by SAR, take low 32).
- `sext ar,as,imm`: sign-extend `AR[s]` from bit `imm` (imm 7..22): `AR[r] = sign_extend(AR[s], imm+1)`. Vector `00 98 23` = `sext a9,a8,7`.
- `nsau ar,as`: normalize shift amount unsigned = count of leading zero bits of `AR[s]`; **result is 32 when `AR[s] == 0`**.

- [ ] **Step 1: Decode tests** using the vectors (assert exact `Op` + fields; for `slli`/`srli`/`srai` assert the decoded shift amount, e.g. `slli a3,a3,0x1e` -> shift 30).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Variants + decode arms** (`Op::{Slli,Srli,Srai,Sll,Srl,Src,Ssl,Ssr,Ssai,Sext,Nsau}`), `///`-documented.
- [ ] **Step 4: Run decode to pass.**
- [ ] **Step 5: Execute tests.** Critically include the SAR round-trips: `ssl(5); sll a_r,a_s` yields `AR[s]<<5`; `ssr(5); srl a_r,a_t` yields `AR[t]>>5`. Plus `srai` on a negative (sign fill), `src` funnel across the boundary, `sext` from bit 7, `nsau(0)==32` and `nsau(0x0000_0001)==31`.
- [ ] **Step 6: Run to fail, implement `arith::exec` arms, run to pass.**
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a shift/extract opcodes (SAR family)`).

---

## Task 5: Arithmetic -- multiply and divide (6)

**Files:** Modify `decode/arith.rs`, `interp/arith.rs`, `decode/mod.rs`.

**Opcodes (vectors):** `mull` `50 73 82`, `mul16s` `30 22 d1`, `mul16u` `d0 23 c1`, `rems` `50 6a f2`, `remu` `70 52 e2`, `quou` `f0 2e c2`.

**Semantics:**
- `mull ar,as,at`: `AR[r] = (AR[s] * AR[t]) & 0xFFFFFFFF` (low 32 of the product; sign-agnostic for the low word).
- `mul16s ar,as,at`: `AR[r] = (sign_extend16(AR[s]) * sign_extend16(AR[t]))` (16x16 signed, full 32-bit result).
- `mul16u ar,as,at`: `AR[r] = ((AR[s] & 0xFFFF) * (AR[t] & 0xFFFF))` (16x16 unsigned).
- `quou ar,as,at`: `AR[r] = AR[s] / AR[t]` unsigned. (`quos` is absent from the firmware.)
- `remu ar,as,at`: `AR[r] = AR[s] % AR[t]` unsigned.
- `rems ar,as,at`: `AR[r] = (AR[s] as i32) % (AR[t] as i32)` signed remainder.
- **Division by zero:** match QEMU `translate.c` -- Xtensa leaves the result unspecified/does not trap; return `AR[s]` (or 0) and comment the choice. Do NOT panic (a `/ 0` in Rust panics -- guard it explicitly).

- [ ] **Step 1: Decode tests** (vectors). - [ ] **Step 2: Run to fail.** - [ ] **Step 3: Variants + decode arms** (`Op::{Mull,Mul16s,Mul16u,Rems,Remu,Quou}`). - [ ] **Step 4: Decode to pass.**
- [ ] **Step 5: Execute tests**, including: `mul16s` with two negative 16-bit inputs (assert positive full result), `mul16u` with high-bit-set 16-bit inputs, `mull` low-32 truncation of a product that overflows 32 bits, `rems` with a negative dividend (sign of remainder follows dividend), `quou`/`remu` on large unsigned values, and a **zero-divisor test** asserting no panic and the documented result.
- [ ] **Step 6: Run to fail, implement (guard the zero divisor), run to pass.**
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a multiply/divide opcodes`).

---

## Task 6: Branch family (27)

**Files:** Modify `decode/branch.rs`, `interp/branch.rs`, `decode/mod.rs`.

**Opcodes (vectors in the table):** `j`, `beqz`, `bnez`, `beqz.n`, `bnez.n`, `bltz`, `bgez`, `beq`, `bne`, `blt`, `bltu`, `bge`, `bgeu`, `beqi`, `bnei`, `blti`, `bgei`, `bltui`, `bgeui`, `bbci`, `bbsi`, `bbc`, `bbs`, `bnone`, `bany`, `bnall`, `ball`.

**Target computation** (all PC-relative): compute the absolute target and store it in the `Op`, mirroring how `call8` already stores an absolute `target`. Use each op's vector to pin the offset field width and the base:
- `j`: `op0=6`, `imm18` (word>>6), target = `pc + 4 + sign_extend(imm18,18)`. Vector `46 02 00` @0x27b8 -> `0x27c5`.
- `beqz/bnez/bltz/bgez`: `op0=6`, `imm12` (word>>12), 12-bit signed, target = `pc + 4 + sign_extend(imm12,12)`; the compare reg is `s`(n2), the condition is in `n1` low bits. Vector `16 88 04` (`beqz a8,0x28c1`).
- `beqz.n/bnez.n`: narrow `op0=0xC`, `imm6` from `n3`/`b1`, target = `pc + 4 + imm6` (unsigned 6-bit, forward-only). Vector `8c 52` (`beqz.n a2,0x27aa`).
- register compares `beq/bne/blt/bltu/bge/bgeu`: `op0=7` (BRI8), `imm8` = b2, target = `pc + 4 + sign_extend8(imm8)`; regs `s`(n2), `t`(n1); `r`(n3) selects which compare. Vector `57 17 14` (`beq a7,a5,0x276e`).
- immediate compares `beqi/bnei/blti/bgei`: `op0=7`, compare `AR[s]` against **B4CONST[t]** (the encoded-constant table, indexed by `n1`): `B4CONST = [-1,1,2,3,4,5,6,7,8,10,12,16,32,64,128,256]`. `bltui/bgeui` use **B4CONSTU** = `[32768,65536,2,3,4,5,6,7,8,10,12,16,32,64,128,256]`. Vector `26 66 02` (`beqi a6,0x6,0x4619`) -> index giving constant 6.
- bit-test `bbci/bbsi`: `op0=7`, test bit `imm5` (from `t`(n1) and part of b2) of `AR[s]`; `bbc/bbs` take the bit index from `AR[t]`. Vectors `37 64 05` (`bbci a4,0x3,...`), `47 5a 0e` (`bbc a10,a4,...`).
- mask-test `bnone/bany/ball/bnall`: `op0=7`, test `AR[s] & AR[t]`: `bnone` (none set) / `bany` (any set) / `ball` (all bits of `t` set in `s`) / `bnall` (not all). Vectors `77 0f 0f` (`bnone`), `a7 88 bc` (`bany`), `37 44 02` (`ball`), `67 c7 13` (`bnall`).

Confirm every `r`(n3) selector value from the vectors before writing the arms; the BRI8 sub-op map is the crux of this task.

- [ ] **Step 1: Decode tests** -- one per op, asserting the exact absolute `target` from the vector's disassembly (e.g. `beq` -> `target: 0x276e`). This directly validates the offset math.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Variants + decode arms.** Add `Op` variants; a compact shape is one variant per structural family carrying a comparison selector, or one variant per mnemonic -- choose per the existing code's granularity (the existing decoder uses one variant per mnemonic; follow that). Store absolute `target: u32`, plus the operand regs/immediate/constant.
- [ ] **Step 4: Decode to pass.**
- [ ] **Step 5: Execute tests** -- for EACH op, one taken and one not-taken case, asserting `cpu.pc` becomes the target (taken) or `pc + len` (not-taken). For `beqi`/`bltui` include a case pinning the B4CONST/B4CONSTU table lookup. For `bbci`/`bbc` pin the bit index source (immediate vs register). For `blt` vs `bltu` use a value with the high bit set so signed/unsigned diverge.
- [ ] **Step 6: Run to fail, implement `branch::exec`, run to pass.**
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a branch family (27 opcodes)`).

---

## Task 7: Zero-overhead loop (loop/loopnez + interpreter-core machinery)

**Files:** Modify `regfile.rs` (add `LBEG`/`LEND`/`LCOUNT`), `interp/mod.rs` (the per-retire loopback in `step()`), `decode/control.rs`, `interp/control.rs`, `decode/mod.rs`.

**Interfaces:**
- Produces: `regfile` fields `lbeg`/`lend`/`lcount: u32` with getters/setters; `Op::{Loop, Loopnez}`; loopback logic in `step()`.

**Vectors:** `loop` `76 84 07` @0x3f8d -> `loop a4,0x3f98`; `loopnez` `76 93 05` @0x3b81d -> `loopnez a3,0x3b826`. Both are `op0=6`; `imm8` = b2; the `r`(n3) field distinguishes `loop`(0x8)/`loopnez`(0x9)/`loopgtz`(0xA) -- confirm from the two vectors.

**Semantics (CRITICAL -- the LEND asymmetry):**
- `loop as,label`: `LCOUNT = AR[s] - 1`; `LBEG = pc + 3`; `LEND = pc + 4 + imm8`. Execution falls through into the body. (Verify: `0x3f8d + 4 + 0x07 = 0x3f98`, matching the disassembly.)
- `loopnez as,label`: if `AR[s] == 0`, branch past the loop: `pc = pc + 4 + imm8` (== `LEND`), do not set the loop registers active. Else identical to `loop`.
- **Loopback (in `step()`):** after an instruction retires and advances PC **sequentially** (not via a taken branch/jump/call/ret), if the new `pc == LEND` and `LCOUNT != 0`, then `LCOUNT -= 1` and `pc = LBEG`. If `pc == LEND` and `LCOUNT == 0`, fall through (loop complete). Cross-check the branch-target-equals-LEND edge against QEMU `translate.c`; the sequential case above is the load-bearing one for the firmware.

- [ ] **Step 1: Add loop registers.** Add `lbeg`/`lend`/`lcount` to `regfile.rs` with `///` docs and getters/setters. Write a `regfile` unit test that they default to 0 and round-trip.
- [ ] **Step 2: Decode tests** for `loop`/`loopnez` in `decode/control.rs`, asserting the decoded absolute `end` address (`LEND`) computes to the vector's target (`0x3f98`, `0x3b826`) -- this is the regression guard for the `pc+4+imm8` formula. Include a negative check that the formula is NOT `pc+3+imm8` (assert `end != pc + 3 + imm8` for a case where they differ; here imm8>0 so they always differ).
- [ ] **Step 3: Run to fail.**
- [ ] **Step 4: Add `Op::Loop { s, end }` / `Op::Loopnez { s, end }`** (store the absolute `end`, and `lbeg = pc + 3` can be recomputed at exec or stored) and decode arms. Run decode to pass.
- [ ] **Step 5: Execute test for the loopback.** Construct a `Cpu` with a tiny program in memory: a `loop a4, +N` with `AR[4] = 3`, a body of two trivial ops (e.g. `addi.n a5,a5,1`), and a marker instruction at `LEND`. Run `step()` in a loop and assert the body executes exactly 3 times (e.g. `AR[5]` incremented 3 times) and PC then proceeds past `LEND`. Add a `loopnez` test with `AR[s] == 0` asserting the body is skipped entirely (PC jumps straight to `LEND`).
- [ ] **Step 6: Run to fail, implement `loop`/`loopnez` in `interp/control.rs` and the loopback in `step()`, run to pass.**
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a zero-overhead loop (loop/loopnez, LEND=pc+4+imm8)`).

---

## Task 8: Control -- rotw, waiti, call0, ret.n (4)

**Files:** Modify `decode/control.rs`, `interp/control.rs`, `decode/mod.rs`.

**Vectors:** `rotw` `10 80 40` -> `rotw 0x1`; `waiti` `00 70 00` -> `waiti 0x0`; `ret.n` `0d f0`; `call0` `85 ec ff` @0xe1ce -> `call0 0xe098`.

**Semantics:**
- `rotw imm4`: rotate the window base by a signed 4-bit immediate: `WINDOWBASE = (WINDOWBASE + sign_extend4(imm4)) mod NAREG/4`. Use the existing `regfile` rotate helper (the one `retw` uses). Vector `rotw 0x1` -> +1.
- `waiti imm4`: set `PS.INTLEVEL = imm4` and enter wait state -- return the existing `Step::Wait`. (This is likely the command-loop idle instruction.)
- `ret.n`: narrow return, `pc = AR[0]` (non-windowed return; the existing `ret_n_is_not_misdecoded_as_retw_n` test proves it currently decodes to `Unknown` -- this task makes it a real op). Note: `ret.n` restores from `a0` WITHOUT the window rotation `retw.n` does.
- `call0 label`: CALLN, like the existing `call8` but non-windowed. The target is computed with `call8`'s exact formula -- `((pc + 4) & !3) + (sign_extend18(imm18) << 2)` -- but `call0` writes the return address to `a0` (`AR[0] = pc + 3`) and leaves `PS.CALLINC` unchanged (no window increment). Vector `85 ec ff` @0xe1ce -> target `0xe098`.

- [ ] **Step 1: Decode tests** (vectors; `ret.n` must now be `Op::RetN`, no longer `Unknown` -- and update the existing `ret_n_is_not_misdecoded_as_retw_n` test to assert `Op::RetN` and that it's still distinct from `Op::RetwN`).
- [ ] **Step 2: Run to fail.** - [ ] **Step 3: Variants + decode arms** (`Op::{Rotw, Waiti, RetN, Call0}`). - [ ] **Step 4: Decode to pass.**
- [ ] **Step 5: Execute tests:** `rotw` moves WINDOWBASE by the signed immediate; `waiti` returns `Step::Wait`; `ret.n` sets `pc = AR[0]`; `call0` sets `AR[0] = pc + 3` and `pc = target`.
- [ ] **Step 6: Run to fail, implement, run to pass.**
- [ ] **Step 7: Full suite + commit** (`feat(#140): M2a control opcodes (rotw/waiti/call0/ret.n)`).

---

## Task 9: System -- SR/UR access, rsil, general-exception raise, sync/cache no-ops (12)

**Files:** Modify `regfile.rs` (add `EXCCAUSE`, and the `EPC1`/`PS.EXCM` already exist from M1.5 -- confirm), `interp/mod.rs` (general-exception raise), `decode/system.rs`, `interp/system.rs`, `decode/mod.rs`.

**Vectors:** `rsr` `30 e4 03` -> `rsr a3,INTENABLE`; `rsil` `20 62 00` -> `rsil a2,0x2`; `wur` `30 e7 f3` -> `wur a3,VECBASE`; `syscall` `00 50 00`; `memw` `c0 20 00`; `nop` `f0 20 00`; `rsync` `10 20 00`; `nop.n` `3d f0`; `dii` `72 72 00`; `dhi` `62 72 00`; `dhwbi` `52 72 00`; `ihi` `e2 72 00`.

**Semantics:**
- `rsr at,sr`: `AR[t] = SR[sr]` -- read the modeled SR (extend the existing `wsr` router: SAR/PS/WINDOWBASE/WINDOWSTART/VECBASE/EPC1/EXCCAUSE and any already modeled); return 0 for unmodeled SRs and log, mirroring `wsr`. SR number = b1 (`(r<<4)|s`), reg = `t`(n1). (`wsr` already exists; add `rsr` as its read sibling. `xsr` is absent from the firmware.)
- `wur at,ur` / `rur at,ur`: write/read a user (TIE) register. The firmware uses `wur` with UR names (e.g. `VECBASE` shown by Ghidra is a UR alias here). Model the URs the firmware writes; log-and-no-op the rest. UR number is encoded in `op1`/`op2` (`rur`/`wur` share `op0=0`, `op2=3`; confirm `op1` split and the UR-number field from the vector `30 e7 f3`).
- `rsil at,imm4`: `AR[t] = PS` (the FULL old PS, not just the level), then `PS.INTLEVEL = imm4`.
- `syscall`: raise a general exception -- `EXCCAUSE = 1` (SYSCALL cause), `EPC1 = pc`, `PS.EXCM = 1`, `pc = VECBASE + user_exception_offset`. The user-exception vector offset for the Xtensa exception architecture is a fixed offset from `VECBASE` (confirm the value from QEMU/the ISA -- commonly the "UserExceptionVector"); since the vector is unmapped pre-MMU, the test asserts the raised machine state, not execution past the vector.
- `memw`, `nop`, `nop.n`, `rsync`, and the cache ops `dii`/`dhi`/`dhwbi`/`ihi`: logged no-ops that advance PC, exactly like the existing `isync`/`dsync`.

- [ ] **Step 1: Add `EXCCAUSE` to `regfile.rs`** (getter/setter, `///` doc). Confirm `EPC1` and `PS.EXCM` handling already exist from M1.5 (`interp/mod.rs`); if `PS.EXCM` set/clear helpers are missing, add them. Write a regfile test for `EXCCAUSE` round-trip.
- [ ] **Step 2: Decode tests** for all 12 ops (vectors). For the no-ops assert the `Op` variant and len; for `rsr`/`wur`/`rsil` assert the decoded SR/UR number and reg.
- [ ] **Step 3: Run to fail.** - [ ] **Step 4: Variants + decode arms** (`Op::{Rsr, Wur, Rur?, Rsil, Syscall, Memw, Nop, NopN, Rsync, Dii, Dhi, Dhwbi, Ihi}` -- include `Rur` only if it appears; the table shows only `wur`, so add `Rur` only if you find it). - [ ] **Step 5: Decode to pass.**
- [ ] **Step 6: Execute tests:**
  - `rsr` reads a modeled SR (set `EXCCAUSE`, read it back via `rsr`); reads 0 for an unmodeled SR without panic.
  - `rsil` writes the full old `PS` into `AR[t]` and sets `PS.INTLEVEL` (use a `PS` value with bits outside INTLEVEL set, and assert all of them come back in `AR[t]`).
  - `syscall` sets `EXCCAUSE=1`, `EPC1=pc`, `PS.EXCM=1`, and `pc = VECBASE + user_exception_offset`.
  - `memw`/`nop`/`dii`/etc. advance PC by their length and change nothing else.
- [ ] **Step 7: Run to fail, implement (`system::exec` + the `raise_general_exception` in `interp/mod.rs`), run to pass.**
- [ ] **Step 8: Full suite + commit** (`feat(#140): M2a system opcodes + general-exception raise`).

---

## Task 10: Exit-gate coverage scan (firmware-gated)

**Files:** Create `src/firmware/xtensa/coverage_scan.rs` (or a `#[cfg(test)]` module in `interp/mod.rs`); Modify `xtensa/mod.rs` if a new file.

**Goal:** prove full executed-code coverage -- zero `Op::Unknown` on real instructions across both the Ghidra-identified body and the boot region.

**Approach:** a firmware-gated test (skips when `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin` is absent, like the existing boot test) that:
1. Loads the firmware image (via the existing `FirmwareImage` loader).
2. Decodes forward from a set of known code entry points, following the linear instruction stream within each identified function range from `build/experiments/firmware-re/funcs.txt` -- OR, more simply, decode every instruction in the Ghidra `listing.txt` by re-deriving its bytes from the image at each listed offset (the listing gives the authoritative instruction boundaries, avoiding data-in-code mis-alignment).
3. Also decode the boot path `0x320..0x399` (the MMU prologue) and the reset head from `~0x200` -- the region Ghidra skipped -- following the real executed path.
4. Assert every decode is NOT `Op::Unknown`, collecting any exceptions into a documented allowlist of data-in-code islands (there should be none for real instructions; if any appear, investigate whether it is a genuine missing op or a listing mis-identification, and document).

- [ ] **Step 1:** Write the coverage-scan test skeleton, firmware-gated (early-return with a log line if the binary is absent). Use `build/experiments/firmware-re/listing.txt` line offsets as the instruction-boundary oracle: for each `OFFSET: HEXBYTES MNEMONIC` line, read `HEXBYTES.len()/2` bytes from the image at `OFFSET`, `decode()` them, and assert non-`Unknown`. (The listing path is `build/experiments/firmware-re/listing.txt`; it is git-ignored but present on disk -- gate on its existence too.)
- [ ] **Step 2: Run it.** `cargo test --lib firmware::xtensa::coverage > /tmp/m2a-t10.txt 2>&1`; read it. Expected: PASS (zero Unknown) if Tasks 2-9 are complete, or a precise list of the mnemonics still decoding to Unknown (which points at a bug in the corresponding task).
- [ ] **Step 3:** For the boot region, add the `0x200..0x399` executed-path decode assertions (these ops are all pre-existing, so this mainly guards against a refactor regression).
- [ ] **Step 4: Full suite + commit** (`feat(#140): M2a exit-gate coverage scan (executed-code, zero Unknown)`).

---

## Notes for the executor

- **Field layouts are hints, not gospel.** Every "op0=X, r selects Y" claim in this plan is a starting point. The authority is the oracle vector: decode the real bytes with objdump on a scratch file, match the disassembly, and derive the field extraction from that. This is the derive-from-toolchain loop; it is how M1 was built and how the adversarial reviewer verified it.
- **Semantics authority is QEMU `target/xtensa/translate.c`.** Where this plan states a semantic (especially the SAR/shift relations, B4CONST tables, division-by-zero, the loop `LEND` formula, `rsil` returning full PS), it has been checked, but confirm against QEMU when the arm is non-obvious, and comment the source.
- **Keep `Bus` unchanged.** M2a adds no MMU and no new `Bus` methods beyond what already exists (compose 16-bit access from `load8`/`store8` if needed).
- **The `Op` enum grows by ~89 variants.** That is inherent to the ISA; keep it in `decode/mod.rs`. If a single category file exceeds ~500 lines, that is expected and fine -- the split already keeps each below the monolith it replaced.
