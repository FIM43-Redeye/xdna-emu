# MAC configuration word

The MAC instructions (`vmul`, `vmac`, `vmsc`, `vnegmul`, `vnegmac`, `vnegmsc`, `vaddmac`, `vaddmsc`,
`vsubmac`, `vsubmsc`, the accumulator-only `vadd`, `vsub`, `vnegadd`, `vnegsub`, `vneg`, and their `.f`
forms) take a 32-bit configuration register. Layout: llvm-aie `VecConf`
(`llvm/lib/Target/AIE/aie2/AIE2InstrPatterns.td`) and `aiev2_compute_control` (`aiev2_vmult.h`); every
field below was exercised on NPU1 silicon by `tools/gen_mac_conf_probe.py` and replayed by
`src/interpreter/execute/vector_matmul/conf_goldens.rs` (`tools/golden/silicon/mac_conf.json`).

## Field table

| Field | Bits | Meaning | Source | Silicon-confirmed |
|---|---|---|---|---|
| `zero_acc` (dynZeroAccum) | 0 | acc1 is replaced by zero | VecConf | yes (`flags`, `modes`, `accops`) |
| `amode` | 2:1 | 0: 32-bit lanes, 1: 64-bit lanes, 2 and 3: float multiplier ("float conf") | VecConf, probes | yes (`codes`) |
| `bmode` | 4:3 | multiplier precision (see geometry table); ignored when `amode >= 2`; `bmode` 1 is `bmode` 3 when `amode` is 1 | VecConf, `aiev2_vmult.h`, probes | yes (`codes`, `same`) |
| `cmode` (variant) | 7:5 | multiplication mode (lane map) | `aiev2_vmult.h`, probes | yes |
| `sgn_y` | 8 | B operand signed | VecConf | yes. Ignored by the float multiplier |
| `sgn_x` | 9 | A operand signed | VecConf | yes. Ignored by the float multiplier |
| `shift16` (accShift) | 10 | acc1 is shifted left by 16 before its negation and the add (acc2 and the product are not) | VecConf, sparse probes | yes (`flags`, `modes`, `accops`) |
| `sub_mul` (dynMulNeg) | 11 | inverts the instruction's negation of the product sum | VecConf | yes |
| `sub_acc1` (dynAcc0Neg) | 12 | inverts the instruction's negation of acc1 (also on integer `vneg`) | VecConf | yes (`flags`, `accops`) |
| `sub_acc2` (dynAcc1Neg) | 13 | inverts the instruction's negation of acc2 | VecConf | yes |
| reserved | 15:14, 31:24 | no effect | VecConf | yes (`upper`: every bit alone, every kind, two geometries) |
| `term_neg` (dynTermNeg) | 23:16 | one bit per post-adder term slot, negates the partial products of the slot | VecConf, probes | yes, see below |

Result = `(P) + acc1 + acc2` with signs: the instruction kind's (`MacOpKind::term_signs`: `mul = P`,
`negmul = -P`, `mac = acc1 + P`, `msc = acc1 - P`, `negmac = -acc1 - P`, `negmsc = -acc1 + P`,
`addmac = acc1 + acc2 + P`, `addmsc = acc1 + acc2 - P`, `submac = acc1 - acc2 + P`,
`submsc = acc1 - acc2 - P`) each flipped by its `sub_*` bit; wrapping at the lane width. The dense and
sparse integer forms share this adder (`sparse::combine_integer`). `vadd`/`vsub`/`vnegadd`/`vnegsub`
use the same adder without a product.

## Multiplier modes (`amode`, `bmode`, `cmode`)

All 128 keys were run on silicon with identical operands (`same`) and decoded (`codes`, `onehot`):

- **The header's integer modes** (`aiev2_vmult.h`, generated into `CONFIG_GEOMETRY_TABLE`): as modelled.
  With `amode` 1, `bmode` 1 is the same mode as `bmode` 3 (`canonical_key`).
- **Complex elementwise modes** (not built by the header, used by Chess `mul_elem_8*` /
  `mul_elem_8_2*`): `(1, 0, 1)` cint32 x cint16 over 8 complex lanes, `(1, 1|3, 4)` cint16 x cint16 over
  two sets of 8. Accumulator lane `2n + p` sums `A[16t + 2n + q] * B[16t + 2n + (q ^ p)]` over
  `(t, q)`; the minus sign of `re*re - im*im` and the conjugations are `term_neg` bits (Chess's
  `0x0a0322` negates the im*im slots of the real lanes). `MacShape::ComplexElem`,
  `SILICON_GEOMETRY_TABLE`.
- **Float conf** (`amode` 2 or 3, any `bmode`, on an integer instruction): `cmode` 0 is the 4x8x4 tile,
  `cmode` 1 the 16-lane pair form. Each product is the unsigned product of the **low bytes** of the
  16-bit elements, taken to 14 signed bits, **after** the instruction's, `sub_mul`'s and `term_neg`'s
  negation; lanes 0-15 sum them (32-bit), lanes 16-31 see only the accumulator inputs. `Multiplier::FloatConf`.
- **Every other key** (sparse keys on a dense instruction, `cmode >= 5`, `amode` 0/1 keys no mode uses,
  `amode >= 2` with `cmode >= 2`): the product is zero; the accumulator inputs still pass through the adder.
  The lanes are 64-bit under `amode` 1, else 32-bit. `Multiplier::Idle`. No key faults or hangs.

## `term_neg`

Bit `b` negates every partial product whose slot is `b`, with
`b = (lane % lane_classes) * (8 / lane_classes) + slot(term)` (`TermNegRule`, `term_neg_rule`). Read off
silicon with each bit alone on shared operands (`termneg`, 21 dense modes; `sparse`, 3 sparse modes, Chess):

| Mode | classes | slot |
|---|---|---|
| int8x4 4x16x8 | 2 | `2*(t>>1&1) + (t>>3&1)` |
| 8x8: 4x8x8, conv 8x8, conv 32x8 | 2 | `(t>>2&1) + 2*(t&1)` |
| 8x8 elementwise k=2 | 2 | 0 |
| conv 4x4 8ch | 2 | `2*(t&1)` |
| 16x8: 4x4x8, 2x4x8ch | 4 | `t>>1` |
| 16x16 acc32: 4x2x8 | 4 | `t` |
| 16x16 elementwise k=1 | 4 | 0 |
| 32x16 and complex 32x16 | 2 | `2*half + t` (a 32-bit A element is a low and a high 16-bit half product) |
| 16x8 acc64 | 2 | `t>>1` |
| 16x16 acc64, complex 16x16 | 2 | `t` |
| float conf cmode 0 / cmode 1 | 2 | `t & 3` / `t` |
| sparse 16-bit 2x8x8 / 8-bit 2x16x8 / 4x16x8 | 2 | stored B row `s` of the term: `s` / `s>>1` / `2*(s&1) + (s>>2&1)` |

## Evidence

Every suite below has silicon run r1 == r2 byte for byte and EMU == HW (`build/experiments/harness/macconf/<plan>`,
regenerate with `tools/gen_mac_conf_probe.py <plan>`):

| Plan | Content | Cases / tiles |
|---|---|---|
| `flags` | every kind x {zero, shift16, sub_mul, sub_acc1, sub_acc2, sgn_x, sgn_y} on int8 and int16 geometries | 60 / 2880 |
| `upper` | each reserved and `term_neg` bit alone | 20 / 740 |
| `termneg` | each `term_neg` bit alone on every valid dense mode | 21 / 189 |
| `codes`, `same` | all 128 keys, mul and mac / identical operands for all keys | 32 / 1536, 11 / 512 |
| `onehot`, `fprobe`, `fsingle` | lane maps of the complex and float-conf modes | 12, 12, 5 |
| `modes`, `random`, `repro` | every kind x random flags on the complex, float-conf and idle keys; random conf words | 80, 40, 4 |
| `accops` | `vadd`/`vsub`/`vnegadd`/`vnegsub`/`vneg` x every field | 24 / 1152 |
| `sparse` | sparse term_neg, Chess `mul_*` sparse intrinsics | 48 / 2304 |

## Open

- Sparse 4-bit mode `(0, 0, 1)`: `term_neg` not probed (no effect modelled).
- Sparse instructions with a key that is not a sparse mode: no probe (needs hand assembly; reported unhandled).
- `term_neg` on the bf16 `.f` instructions: not probed; `vnegmac.f`, `vsubmac.f`, `vsubmsc.f` still unmodelled
  (no intrinsic emits them).
