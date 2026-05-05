# Mode-2 LC frame semantics: bit-28 flag is always 0

**Status:** Phase 0 of A.2b complete. Placeholder hypothesis refuted.
**Captured:** 2026-04-30 on Phoenix (NPU1, AIE2), Chess-built fixture.

## Hypothesis under test

The mode-2 (Execution / `inst_exec`) trace's LC frame has a 1-bit flag at
position 28 of its 32-bit word. The placeholder rule shipped in
`src/device/trace_unit/mod.rs::compute_lc_flag` predicted:

> `flag = 1 iff lc_after == 0` -- i.e., flag=1 on the iteration where the
> loop count register reaches 0 (the last iteration of the ZOL).

This implied LC frames were emitted **per-iteration** with a 28-bit running
count, terminated by one flag=1 frame at exhaustion.

## Method

Built `tools/mode2_capture_fixtures/heavy_zol/` with Chess (kernel.cc +
inline aie.mlir wrapper). The wrapper calls `k_pass` 64 times; each pass
runs a ZOL of trip count N read from `in[0]`. Disassembly confirmed
Chess emits a hardware ZOL via `movxm ls, ...; movxm le, ...; mov lc, r25`
around a `ZLS_F.../ZLE_F...` body. Captured mode-2 traces for
N ∈ {1, 2, 4, 8, 16, 64, 256, 1024, 16384} via `bridge-trace-runner`,
decoded with `tools/parse-trace.py --decoder ours --trace-mode inst_exec`.

**Capture artifacts:** `build/experiments/mode2_phase0/captures/heavy_zol_n*.trace.bin`
**Decoded:** `build/experiments/mode2_phase0/decoded/heavy_zol_n*.cmd.json`

## Result

```
N=    1: 64 LC frames  flag=0 always  count=1 always
N=    2: 64 LC frames  flag=0 always  count=2 always
N=    4: 64 LC frames  flag=0 always  count=4 always
N=    8: 64 LC frames  flag=0 always  count=8 always
N=   16: 64 LC frames  flag=0 always  count=16 always
N=   64: 64 LC frames  flag=0 always  count=64 always
N=  256: 64 LC frames  flag=0 always  count=256 always
N= 1024: 64 LC frames  flag=0 always  count=1024 always
N=16384: 64 LC frames  flag=0 always  count=16384 always
```

In every capture: **exactly 64 LC frames** (one per outer wrapper pass,
i.e. one per ZOL invocation), **count = N** (the value loaded into LC at
loop start), **flag = 0 always**.

## Findings

1. **LC frames are emitted once per ZOL invocation, not per iteration.**
   The atoms inside the body (E_atom / N_atom / Repeat0 / Repeat1) carry
   the per-cycle execution information; the LC frame is a one-shot
   structural marker tied to the ZOL boundary.

2. **The 28-bit count field stores the initial trip count loaded into LC
   at loop start, not the running LC register value.** This matches what
   you'd get by reading the LC SR once at ZLS, not by decrementing.

3. **Bit-28 (the `flag`) is 0 for all observed normal ZOL executions.**
   We never reproduced flag=1 in controlled fixtures. The historical
   sighting (`count=232774656`, ~232M, in an exploratory `/tmp` capture)
   was almost certainly a decoder false-positive on misaligned bytes:
   232M is far outside any reasonable trip count, and the byte pattern
   that would decode as flag=1 here (`010 1` at bits 31..28 = nibble
   `0x5`) collides with several legitimate padding patterns when the
   decoder's bit pointer lands wrong.

4. **The placeholder rule in `compute_lc_flag` is wrong on two fronts:**
   it emits a frame per iteration instead of per ZOL, and it computes a
   flag that never actually fires.

## Phase 0.5: documentation search and nested case (2026-04-30 follow-up)

### AM020 / AM025 / AM027 / AM029 search: bit-28 not documented

Two parallel Explore agents combed the public AMD architecture and
register references:

- **Architecture (AM020 ch. 2, AM027 equivalents):** mention only that
  mode-2 trace records "Conditional and unconditional direct branches,
  All indirect branches, **Zero-overhead-loop LC**." No bit layout, no
  flag-bit semantics, no frame-format spec.
- **Register reference (AM025: ~1800 registers, ~6400 bit fields; AM029
  for v2):** documents `Trace_Control0.Mode [1:0]`, `Trace_Status.Mode
  [2:0]`, and `Core_LC` as a flat 32-bit register with no subfields.
  No register-level definition of the LC trace frame format. `Trace_
  Control1.Packet_Type [14:12]` is a 3-bit field with no enumeration
  table publicly disclosed.

The bit-by-bit packet structure (4-bit prefix, 1-bit flag, 28-bit count)
is **only** in `libxv_trace_decoder_opt.so`'s reverse-engineered symbol
table (`Execution_LC` class). AMD does not publish it.

**Conclusion:** the docs cannot tell us what flag=1 means. Empirical
probing is the only path.

### Nested-loop capture (outer software / inner ZOL)

`tools/mode2_capture_fixtures/nested_loop/` built with Chess (kernel.cc
+ inline aie.mlir wrapper). Outer-loop trip count = `in[0]`, inner =
`in[1]`. Disassembly confirms Chess emits ZOL only for the inner loop
(`movxm ls, ...; movxm le, ...; mov lc, r24` around `ZLS_F.../ZLE_F...`)
and a software `jnz` for the outer.

| outer × inner | LC frames | counts        | flags |
|---------------|-----------|---------------|-------|
| 64 × 8        | 64        | {8: 64}       | {0: 64} |
| 32 × 4        | 32        | {4: 32}       | {0: 32} |
| 16 × 2        | 16        | {2: 16}       | {0: 16} |
| 4  × 64       | 4         | {64: 4}       | {0: 4}  |
| 8  × 16       | 8         | {16: 8}       | {0: 8}  |

**Result:** 124 LC frames in the nested case, all flag=0. The
"reactivate same ZOL via software outer loop" pattern does not trigger
bit-28.

**Caveat:** this is **not** a true ZOL-in-ZOL save/restore. AIE2 has one
LC register; only handwritten assembly can express outer-ZOL → save
LS/LE/LC → inner-ZOL → restore. The achievable Peano/Chess pattern
(outer-software + inner-ZOL) is what we tested; the explicit-save/
restore variant is deferred under the "high bar / no handwritten
assembly" guidance.

## Phase 0.6: AIE2P/AIE2PS docs + handwritten kernel attempts (2026-04-30)

### AIE2P / AIE2PS register search: also negative

A focused parallel agent surveyed:
- AM027 (AIE-ML v2 architecture), AM029 (AIE-ML v2 register reference)
- aie-rt headers `xaie2psgbl_params.h` / `xaie2psgbl_reginit.c` (Strix-class)
- mlir-aie JSON DBs (no `aie_registers_aie2p.json` exists yet)
- `me_chess_llvm.h` (Chess's special-register accessor list)

Result: `Core_LC` is a flat 32-bit register on AIE2P with no subfields,
identical to AIE2. Trace_Control0 bit layout is identical. No bit-28
field is named in any v2 source. The "reserved for later NPUs" hypothesis
**cannot be confirmed from any public AMD/Xilinx documentation across
all generations we have access to**.

### Branch-out via Chess C: refused

`tools/mode2_capture_fixtures/branchout_loop` (now in
`build/experiments/mode2_phase0/branchout_loop/`) — Chess C kernel with
an explicit `break` inside a runtime-count loop. Chess **refuses to
emit a ZOL** for any loop containing a conditional break: disassembly
shows zero `ZLS_F` / `ZLE_F` / `mov lc` instructions, only software
`jnz`. Sensible compiler behavior, but it means C-level break-out
tells us nothing about the LC frame -- there is no LC frame to inspect.

### Handwritten Chess inline asm: blocked by `darts`

Chess's VLIW assembler (`darts`) rejected every non-empty inline-asm
content we tried: `nop`, `NOP`, `nopa;`, `.inst NOP`. Empty `asm("")`
parses, anything substantive fails with "Could not assemble." The
underlying syntax appears to be a Chess-specific bundle format that
isn't publicly documented. `me_chess_llvm.h` exposes `register ... asm
("crFPMask")`-style accessors for control / status registers, but
**no LS/LE/LC accessor** -- those remain compiler-managed and not
user-writable from C.

### Peano `.S` linked into chess xclbin: blocked at link time

Wrote `kernel.S` in peano AIE2 assembly (which assembles cleanly with
`clang -x assembler --target=aie2-none-unknown-elf`), tried to feed
the resulting `kernel.o` into a chess-aiecc build via the existing
heavy-zol wrapper. `xbridge` linker fails:
```
undefined symbol "llvm___aie2___acquire"
undefined symbol "llvm___aie2___release"
```
These are intrinsic names emitted by chess's MLIR-to-AIE-ELF lowering
of `aie.objectfifo.acquire/release`. Chess only pulls in the runtime
that defines them when it compiles a `.cc` file; a linked-in peano
`.o` doesn't carry chess's runtime. There is no obvious way to merge
the two.

### Current empirical totals

- 576 LC frames from `heavy_zol` (single ZOL, N from 1 to 16384)
- 124 LC frames from `nested_loop` (5 ratios, software-outer / ZOL-inner)
- = **700 LC frames total, all flag=0**
- AM020/AM025/AM027/AM029 + AIE2P/AIE2PS sources: 0 mentions of bit-28
  semantics
- Chess C with break: refuses to emit ZOL (no test possible)
- Chess inline asm: undocumented syntax, every attempt rejected
- Peano `.S` linked into chess xclbin: blocked by intrinsic mismatch

## Conclusion (closing Phase 0)

Bit-28 of the mode-2 LC frame **is empirically inert across every code
path reachable through the public Peano + Chess + mlir-aie toolchain
on Phoenix (NPU1, AIE2)**. The flag is undocumented in every public
AMD/Xilinx reference manual. We cannot construct a hypothesis-isolating
test for "abnormal exit" or "true nested ZOL save/restore" without
either (a) reverse-engineering the Chess inline-asm bundle syntax, or
(b) finding a way to interleave a peano-compiled kernel with chess's
objfifo runtime intrinsics. Both are open-ended toolchain investments.

The emulator's `compute_lc_flag` is correctly held at 0 for the
reachable codebase. If a future bridge test surfaces a divergence on
bit-28, that would itself be a reproducer worth investigating; until
then, treat the flag as functionally reserved.

## Future probes (low-priority; pursue only if a divergence appears)

- **Real-world ML kernel sweep.** Mass-capture mode-2 traces from
  large ML kernels (matmul, conv, softmax) and check for any flag=1
  in the wild. If any production kernel triggers it, we have a
  natural reproducer.
- **Chess inline-asm syntax.** Worth a focused investigation when
  there's a clear payoff. Likely sources: `me_chess_llvm.h` (full),
  Chess release notes if shipped with aietools, AMD developer forums.
- **Aietools `libxv_trace_decoder_opt.so` deeper dive.** The decoder
  knows the bit-28 field exists; if its source were available it
  might reveal what it decodes flag=1 *as*, even if the trigger is
  still HW-internal.

## Action items

### Required: fix `compute_lc_flag`

Current implementation:
```rust
fn compute_lc_flag(_lc_before: u32, lc_after: u32) -> u8 {
    if lc_after == 0 { 1 } else { 0 }
}
```

Replace with:
```rust
fn compute_lc_flag() -> u8 {
    // Bit-28 of the LC frame is 0 for all normal ZOL executions on
    // Phoenix (NPU1, AIE2). See docs/superpowers/findings/
    // 2026-04-30-mode2-lc-flag-semantics.md for the empirical evidence.
    // The flag's exact trigger is unknown; we have not reproduced flag=1
    // in any controlled fixture. If a future bridge test exposes a
    // divergence on this bit, follow the methodology in the findings
    // doc and update accordingly.
    0
}
```

### Required: fix LC frame emission frequency

The encoder must emit **one LC frame per ZOL invocation**, with
`count = initial trip count`, not one per LC-register-decrement. The
emit point is the ZOL start (when the core executes `mov lc, <reg>` or
`add.nc lc, <reg>, ...`), not the per-iteration boundary.

Entry points to update:
- `src/device/trace_unit/mod.rs` -- the `notify_loop_boundary` call
  site and `compute_lc_flag` itself.
- The interpreter side (Phase 2 drain in
  `src/interpreter/engine/coordinator.rs`) -- needs to surface "ZOL
  starting with trip count N" rather than "ZOL iterated".

### Optional: capture nested_loop later

If divergence appears in bridge tests once the emitter is fixed,
revisit this finding doc and capture `nested_loop` to probe the flag
under LC save/restore. The fixture is built (see
`tools/mode2_capture_fixtures/nested_loop/`).

## Why our `runtime_loop` fixture was useless

The `runtime_loop` kernel runs once per host invocation with a tiny
body (~30 instructions for N=4). The trace shim DMA needs enough atom
data to flush its internal buffer to DDR before kernel exit; tiny
single-pass kernels never flush. `heavy_zol`'s 64-pass wrapper was
designed exactly for this -- 64 ZOL completions guarantee enough atom
volume to surface the LC frame pattern reliably. **Future Phase-0-style
hypothesis tests should default to wrapper-amplified fixtures, not
single-pass tiny kernels.**

## Why Peano builds didn't trace

A side discovery: building our fixture with Peano (the calibration
choice in the original `tools/mode2_capture_fixtures/README.md`)
produces an empty trace BO regardless of mode. The full bridge test
suite happens to use Chess for traced builds; Peano-traced builds are
silently broken in the current toolchain wiring. Not on the critical
path for Phase 0 -- Chess builds ZOL identically (`mov lc, r25` /
`movxm ls / le` / ZLS_F.../ZLE_F...) and produce trace data, so we
switched to Chess for the captures here.

If Peano-traced builds matter for other workflows, the broken path is
between aiecc.py's lowering and the trace shim DMA setup -- the
runtime_sequence's Trace_Control0 register write looks correct, but
no atom data ever reaches the BO. Investigation deferred.

## Public discussion

Posted upstream as [mlir-aie #3047](https://github.com/Xilinx/mlir-aie/issues/3047)
on 2026-05-05 — finding 1 in that issue is the LC bit-28-inert
observation captured here. Update this section if maintainers respond
with the trigger condition for flag=1, fixtures that reproduce it, or
confirmation that the bit is reserved/unused on AIE2.
