# Vector Fuzzer Spike: Peano Intrinsic Reach (Task 1)

Date: 2026-06-10. Feasibility probe for the vector differential fuzzer (#112):
which `aie_api` intrinsics does Peano compile to native AIE2 vector
instructions (no scalarized fallback)?

**Bottom line: every family probed is REACHABLE. Zero compile failures, zero
scalarized loops.** Some ops lower to vector composites rather than a single
opcode (noted below); none fall back to scalar loops, so all are valid
fuzzer ops.

## Setup

- Probe: `build/experiments/vector-fuzzer-spike/probe.cc` (one extern "C"
  function per family; every result stored via `aie::store_v`).
- Compile: `llvm-aie/install/bin/clang++ -O2 -std=c++20
  --target=aie2-none-unknown-elf -I mlir-aie/install/include -c probe.cc`
  -- clean on first attempt, no diagnostics.
- Disassembly: Peano `llvm-objdump -d probe.o` -> `probe.dis` (659 lines,
  all functions straight-line vector code, no loops anywhere).

## int32 x 16 (`aie::vector<int32_t,16>`)

| Family | Intrinsic spelling | Emitted opcode(s) | Verdict |
|---|---|---|---|
| add | `aie::add(a,b)` | `vadd.32` | REACHABLE |
| sub | `aie::sub(a,b)` | `vsub.32` | REACHABLE |
| min | `aie::min(a,b)` | `vmin_ge.s32` | REACHABLE |
| max | `aie::max(a,b)` | `vmax_lt.s32` | REACHABLE |
| neg | `aie::neg(a)` | `vneg_gtz32` | REACHABLE |
| abs | `aie::abs(a)` | `vabs_gtz.s32` | REACHABLE |
| band | `aie::bit_and(a,b)` | `vband` | REACHABLE |
| bor | `aie::bit_or(a,b)` | `vbor` | REACHABLE |
| bxor | `aie::bit_xor(a,b)` | 2x `vbneg_ltz.s32` + 2x `vband` + `vbor` | REACHABLE (composite -- no native VBXOR on AIE2; lowered as `(a&~b)\|(~a&b)`, all-vector) |
| bneg | `aie::bit_not(a)` | `vbneg_ltz.s32` | REACHABLE |
| downshift | `aie::downshift(a,3)` | `vlda.ups.s64.s32` (s0=0) + `vst.srs.s32.s64` (s0=3) | REACHABLE (via UPS->accum->SRS pipeline; shift folded into SRS shift reg, not a lane-arith opcode) |
| upshift | `aie::upshift(a,2)` | `vlda.ups.s64.s32` (s0=2) + `vst.srs.s32.s64` (s0=0) | REACHABLE (shift folded into UPS) |
| lt -> select | `aie::lt` + `aie::select` | `vlt.s32` + `vsel.32` | REACHABLE |
| ge -> select | `aie::ge` + `aie::select` | `vge.s32` + `vsel.32` | REACHABLE |
| eq -> select | `aie::eq` + `aie::select` | `vsub.32` + `veqz.32` + `vsel.32` | REACHABLE (composite -- no native VEQ; eq = eqz(a-b)) |
| broadcast | `aie::broadcast<int32_t,16>(s)` | `vbcst.32` | REACHABLE |
| shuffle_up | `aie::shuffle_up(a,1)` | `vshift` (shift amt 0x3c bytes in reg) | REACHABLE |
| shuffle_down | `aie::shuffle_down(b,2)` | `vshift` (shift amt 0x8 bytes in reg) | REACHABLE |
| raw shuffle | `::shuffle(a,b,28)` (global-ns builtin; compiled with no include tricks) | `vshuffle x0,x0,x2,r0` (mode 28 in `r0`) | REACHABLE |
| max reduce | `aie::max_reduce(a)` (note: `max_red` does not exist in this aie_api) | log2 tree: 4x (`vshift` + `vmax_lt.s32`) + `vextract.s32` | REACHABLE (composite) |
| pack narrow | `aie::concat(a,b)` -> `aie::pack` -> int16x32 | `vlda.ups.s64.s32` + `vups.s64.s32` + 2x `vst.srs.s16.s64` | REACHABLE (lowers through accum SRS, not VPACK, for 32->16) |
| unpack widen | `aie::unpack(int16x32)` -> int32x32 | `vlt.s16` + `vbcst.16` + `vsel.16` + 2x `vshuffle` (modes 0x12/0x13) | REACHABLE (composite: sign-mask + interleave-shuffle) |

## int16 x 32

| Family | Spelling | Opcode(s) | Verdict |
|---|---|---|---|
| add | `aie::add` | `vadd.16` | REACHABLE |
| min | `aie::min` | `vmin_ge.s16` | REACHABLE |
| select | `aie::lt` + `aie::select` | `vlt.s16` + `vsel.16` | REACHABLE |

## int8 x 64

| Family | Spelling | Opcode(s) | Verdict |
|---|---|---|---|
| add | `aie::add` | `vadd.8` | REACHABLE |
| min | `aie::min` | `vmin_ge.s8` (mask pair `r25:r24`) | REACHABLE |
| select | `aie::lt` + `aie::select` | `vlt.s8` + `vsel.8` | REACHABLE |

## bfloat16 x 32

| Family | Spelling | Opcode(s) | Verdict |
|---|---|---|---|
| add | `aie::add` | `vconv.fp32.bf16` x3 + 2x `vadd.f` (bmh accums) + `vst.conv.bf16.fp32` | REACHABLE (fp32 accumulator pipeline) |
| sub | `aie::sub` | same with `vsub.f` | REACHABLE |
| min | `aie::min` | `vmin_ge.bf16` (native, single op) | REACHABLE |
| max | `aie::max` | `vmax_lt.bf16` (native) | REACHABLE |
| neg | `aie::neg` | `vbcst.16 0x8000` + `vadd.16` | REACHABLE (integer sign-bit flip, no FP op) |

## Notes for the fuzzer op table

- **Shuffle mode is a register operand**: Peano emits `mova r0, #mode;
  vshuffle xd, xa, xb, r0`. The mode is not an instruction immediate, so the
  fuzzer can sweep modes at runtime through one binary, or as compile-time
  constants per kernel -- either works. Mode 28 (T32 family) confirmed; range
  not constrained by encoding.
- `vshift` shift amounts are byte counts in a register (shuffle_up 1 lane of
  int32 -> 0x3c = shift right by 60 bytes within the 64B vector; shuffle_down
  2 lanes -> 0x8).
- min/max/abs/neg/compare ops also write a lane-mask GPR (`r16`, or `r25:r24`
  for 8-bit) -- the emulator must get those masks right, not just the vector
  lanes; the fuzzer should compare destination vectors AND mask-consuming
  select results.
- Composite lowerings (bxor, eq, max_reduce, unpack, bf16 add/sub/neg) are
  still fuzzer-valid: every step is a vector op, so wrong emulation of any
  constituent surfaces in the output.
- Spellings that do NOT exist (would fail compile): `aie::band/bor/bxor/bneg`
  (use `bit_and/bit_or/bit_xor/bit_not`), `aie::max_red` (use `max_reduce`).
  No other surprises; the whole probe compiled first try.
- pack(int32->int16) and down/upshift route through the UPS/SRS accumulator
  pipeline -- they exercise SRS saturation/shift semantics, overlapping the
  Half-A SRS coverage rather than adding a distinct VPACK form (VPACK appears
  only for narrower types, not probed here).

Artifacts: `probe.cc`, `compile.log` (clean), `probe.dis` under
`build/experiments/vector-fuzzer-spike/` (build/ is gitignored; this doc is
the committed record).
