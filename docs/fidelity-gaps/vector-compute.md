---
class: vector-compute
subsystem: AIE2 vector unit -- result forwarding/bypass network + special-value semantics
posture: mostly-resolved -- the forwarding model is HW-verified for the bf16 matmul class; the residuals are scoped deferrals and one uncharacterized HW-state regime, not unvalidated model
status: 2 deferred (bypass-matrix fold, benign operand alignment); NaN payload resolved (datapath regime)
---

# Vector Compute Gaps

The AIE2 per-operand bypass/forwarding matrix is implemented and validated
against the chess bridge sweep (89/89 PASS, 0 regressions) **and against real
Phoenix (NPU1) silicon**: `vec_mac_bf16`, `two_col`, and all three
`matrix_multiplication_using_cascade` variants (plain/buffer/cascade) PASS
HW==EMU on the bridge (2026-06-09), with `vec_mac_bf16` and `two_col` also
diffing CLEAN on the trace comparison. The W/X-file forwarding model is therefore
HW-verified for the bf16 matmul class; the gaps below are scoped deferrals, not
unvalidated model.

## Result forwarding / bypass network

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Accumulator/CM-domain (`VEC_Bypass`) result visibility | not in the per-operand bypass matrix; modeled via the separate MAC-pipeline-latency path (`ExecutionContext::queue_matmul_accum_write`). The matrix resolves W/X-file results; `VEC_Bypass` (MAC/MUL results into the accumulator file) is deferred. | `src/interpreter/state/registers.rs` (`VectorRegisterFile::visible_at`); `src/interpreter/state/context.rs` (`queue_matmul_accum_write`, carries a `FIXME(bypass-model)`) | **OPEN / deferred.** The MAC-latency path is HW-validated *functionally* -- all three `matrix_multiplication_using_cascade` variants PASS HW==EMU on the bridge (2026-06-09), which exercises the accumulator-write path -- but not yet at trace granularity (those kernels' TRACE PREP fails for an orthogonal trace-injection reason). Accumulator folding into the per-operand bypass matrix is a noted future extension (see `xdna-emu-priv: research/plans/2026-06-09-vector-write-result-latency.md` Scope). **Partially advanced 2026-06-10 (`220d685`):** the `VMOV bml,x` (wide vector->accumulator move) write is now deferred by its def latency via the same `queue_matmul_accum_write` path, after the phase-B `vec_conv_bf16_edge` silicon kernels proved an immediate write corrupts delay-slot `VST.CONV` stores (read freshly-overwritten accumulator -> Inf/NaN garbage). All 10 convert modes EMU==silicon. The full bypass-matrix fold (distinguishing `VEC_Bypass` MAC->MAC from `NoBypass` store consumers) is still the open extension. |
| `source_forward` positional alignment for merged memory operands | `SlotOp::source_forward` aligns exactly for compute ops; may drift for fused memory operands that merge address/data into a single TableGen operand list position | `src/interpreter/bundle/slot.rs` (`FIXME(source-forward-memory-alignment)`); `src/interpreter/decode/operand_extraction.rs` | **DOCUMENTED / benign.** The bypass model only reads vector-register compute sources; memory operands use `NoBypass` or are not vector-reg reads, so this positional drift does not affect result visibility decisions. |

## Special-value NaN payload (`vadd.f`/`vsub.f`, bf16 + fp32)

> Superseded for the accumulator-to-accumulator `vadd.f`/`vsub.f`/`vnegadd.f`/
> `vnegsub.f`/`vneg.f` by the fp32 accumulator section at the end of this file:
> a fresh 2026-10-29 NPU1 capture (`vec_acc_addsub_f32`, three bit-identical
> runs) follows the MAC datapath's NaN rule on every lane, with no second
> regime observed. The text below is the older two-regime reading.

**RESOLVED 2026-06-11** -- full mechanism and evidence in
`xdna-emu-priv: research/specs/2026-06-11-nan-payload-datapath-regime.md`.

The result NaN payload when an operand is a NaN has **two real silicon regimes**:
a **datapath regime** (dominant) where the payload is the actual exp-255
mantissa-datapath sum `r`, and a **canonical regime** (rare) that forces payload
1. The regime is residual float-unit HW state -- global across lanes,
deterministic within a session, but it varies across sessions and survives a
driver reload. This is **not** bf16-specific, **not** two-NaN-operand-only, and
**not** in the bf16 narrow -- it is the shared `aie2_acc_fp32_add` datapath
(fp32 shows it identically), and `+Inf + -NaN` is exactly the trigger.

The model now produces the **datapath regime** (FIM43-Redeye's call: it is what the ALU
computes; canonical is a residual-state suppression). One change in
`aie2_acc_fp32_add`: `use_r = (a_nan || b_nan) && !overflow` lets the datapath
sum `r` through instead of zeroing it. Validated **8160/8160** against fresh
dominant-regime NPU1 captures (bf16+fp32, add+sub); `cargo test --lib`
3408/3408; seed 6159 via EMU now reproduces `0xFF8C`.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| special-value NaN payload: datapath vs canonical regime | HW has two residual-state regimes for `vadd.f`/`vsub.f` NaN-input payloads; we now model the dominant **datapath** regime (payload = exp-255 mantissa sum `r`). HW in the rare **canonical** regime forces payload 1 and would mismatch the payload bits (a HW-state artifact, not a model bug). | `src/interpreter/execute/vector_float.rs` (`aie2_acc_fp32_add`, `use_r` gate). | **RESOLVED (datapath regime) 2026-06-11.** Validated 8160/8160 vs dominant-regime HW. Residual open: *what* flips the regime is uncharacterized (not driver reload); low priority since the payload is functionally dead. See the design note. |


## SRS rounding/saturation and f32->bf16 conversion

**HW-verified** against NPU1 (Phoenix) silicon captures of 2026-10-28
(`tools/golden/silicon/vec_srs_i32*.json`, 77 kernels; `vec_conv_bf16*.json`,
27 kernels). `srs_lane` / `shift_round` (`src/interpreter/execute/vector_srs.rs`)
and `f32_to_bf16` (`vector_float.rs`) are checked lane by lane against every one
of those files by `matches_every_silicon_srs_capture` and
`matches_every_silicon_bf16_capture`. The kernel parameters come from
`tests/vector-verify/<kernel>/`; the captures cover every `crRnd` value 0-15,
`crSat` none/saturate/symmetric, and SRS shifts 0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 31.
Mode names follow llvm-aie `aiev2_defines.h`.

**Shift.** The effective shift equals the configured shift at every shift tried:
the unrounded result is exactly `clamp16(v >> s)`. There is no hidden bias, so the
former `srs_shift_bias` architecture parameter was removed.

**Rounding.** Let `q = floor(v / 2^s)` and "dropped" the low `s` bits of `v`
(two's complement, so for negative `v` the dropped bits are those of `v - 2^s*q`).

| crRnd | name | rule |
|-------|------|------|
| 0 | floor | `q` |
| 1 | ceil | `q+1` if any dropped bit set |
| 2 | sym_floor | toward zero: `q+1` if inexact and `v < 0` |
| 3 | sym_ceil | away from zero: `q+1` if inexact and `v >= 0` |
| 8 | neg_inf | nearest; tie goes to `q` |
| 9 | pos_inf | nearest; tie goes to `q+1` |
| 10 | sym_zero | nearest; tie toward zero |
| 11 | sym_inf | nearest; tie away from zero |
| 12 | conv_even | nearest; tie to the even result |
| 13 | conv_odd | nearest; tie to the odd result |
| 4, 6 | undefined; `RoundingMode::InexactEven` | if inexact, the even one of `q`, `q+1`; exact values unchanged |
| 5, 7 | undefined; `RoundingMode::InexactOdd` | if inexact, the odd one of `q`, `q+1`; exact values unchanged |
| 14 | undefined; alias of 12 | |
| 15 | undefined; alias of 13 | |

The undefined modes are real and reachable (`RoundingMode::from_raw` maps all 16
`crRnd` values; only values above 15 give `None`). 4 and 6 are indistinguishable
on silicon, as are 5 and 7, and 14/15 equal 12/13. The data separates 4/6 from
conv_even (13 of 64 lanes differ) and 5/7 from conv_odd (11 of 64).

**Saturation.** Rounding happens first, then: `crSat` none wraps to the output
width (low bits kept); saturate clamps to `[-2^(n-1), 2^(n-1)-1]`; symmetric
clamps to `[-(2^(n-1)-1), 2^(n-1)-1]`. A rounding carry is saturated, not wrapped.
Unsigned outputs clamp to `[0, 2^n-1]` (symmetric makes no difference). Int16
output is what the captures exercise; the unsigned and 8/32-bit paths follow the
same rule but have no dedicated silicon capture.

**f32 -> bf16 (`to_v16bfloat16`, governed by `crRnd`).** The modes act as an
SRS shift of 16 on the signed f32 bit pattern (sign-magnitude, rounded by value):

- Normal values, **denormals** and zeros are rounded; denormals are not flushed.
  A rounding carry bumps the exponent, so rounding the largest finite value up
  gives infinity (verified at `0x7F7FFFFF` and the exp-254 neighbours).
- Sign is always preserved: a negative value rounded to zero yields `-0`.
- Infinities pass through.
- **NaN is never rounded**: the high 16 bits are kept in every mode (sign,
  exponent, top 7 payload bits). If those leave a zero payload (NaN payload only
  in the low 16 bits, e.g. `0x7F800001`, `0x7F808000`), payload bit 0 is set so
  the result stays a NaN (`0x7F81` / `0xFF81`). The captures contain 27 distinct
  NaN inputs, all consistent with this.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| SRS paths other than int32 -> int16, signed | unsigned, 8-bit, 32-bit and 64-bit-accumulator outputs reuse the same lane function but have no dedicated silicon capture beyond the fuzzer's 28/28 match | `srs_lane` callers in `vector_srs.rs` | open: extend `vector_kernel_specs.py` |
| `crSat` value 2 | not captured; modelled as symmetric-without-saturate (bit 1 only), which clamps nothing | `SrsConfig::saturate` / `symmetric_saturate` | open |

## bf16 multiply-accumulate datapath (`vmul.f`, `vmac.f`, `vmsc.f`, `vaddmac.f`, ...)

Modeled in `vector_matmul/bf16.rs` (`mac_lane`), derived from NPU1 silicon only
(no model, no vendor source). A lane is the sum of its bf16 products and its
accumulator inputs; the same datapath serves the dense 4x8x4 tile (K = 8) and the
16-lane elementwise form (lane L = A[L]*B[L] + A[L+16]*B[L+16], K = 2).

**Datapath (every item pinned by a silicon capture, see Evidence):**

1. Terms are exact: product `(1.ma)(1.mb) * 2^(ea+eb-254)` (16-bit significand,
   so no product rounding exists), accumulators as stored. Exponent fields 255
   (Inf/NaN) and 0 (denormal, read as field 1) are ordinary numbers. Zero terms
   do not take part in the reference exponent.
2. Reference exponent `E` = largest IEEE exponent of any term (acc1 and acc2
   included). Products use `ea+eb-127`, i.e. a product in [2, 4) has its leading
   bit at `E+1`.
3. Every term is aligned to `E` and rounded **to nearest-even onto a grid of
   weight `2^(E-23)`**. The largest term is exact; a term below half a grid step
   vanishes (this is why `2^30 + 1 - 2^30` gives 0 and why K position is
   irrelevant).
4. The grid integers are summed exactly. Summation order, tree shape and K slot
   have no effect (random wide-spread tiles and permuted multisets all match).
5. The sum is rounded once to fp32 (RNE). A sum with magnitude below 2^-126
   flushes to a **signed** zero **before** rounding (2^-126 - 2^-150, a tie that
   RNE would round up to the smallest normal, reads 0); an exact zero is +0;
   overflow after rounding is Inf. Denormal inputs and accumulators are used as
   is; only results flush.
6. Sign controls: product negated for `negmul`, `msc`, `addmsc`, XOR the config
   word `sub_mul` bit; acc1 negated by `sub_acc1`, acc2 by `sub_acc2`; `zero_acc`
   drops acc1. acc1 and acc2 join the **same** aligned sum (one rounding), they
   are not added afterwards.
7. Specials: any NaN input or Inf*0 -> NaN; Inf terms of one sign -> that Inf;
   Inf terms of both signs -> NaN. A NaN lane is the ordinary numeric result of
   step 5 computed with Inf/NaN exponent fields taken at face value, then
   exponent forced to 255 and mantissa bit 0 set. When that numeric result
   overflows or flushes its mantissa is 0, so nearly all NaNs read
   `0x7f800001` / `0xff800001` (sign = sign of the numeric result; 0*Inf and
   0*NaN with nothing else give +). Only cancelling special terms (e.g. Inf
   against NaN of opposite sign) leave a payload, the normalized difference of
   the two significands (`0xff820001`, `0xff940001`, `0x7ffc0001`, ...). No NaN
   input payload is otherwise propagated.

**Model fit** (silicon lanes reproduced bit-exactly; candidate models scored on
65,536 two-term elementwise lanes with gaps 0..41, random signs and slots, plus
the older K = 8 captures):

| Model | Exact lanes |
|-------|-------------|
| exact sum, one RNE (previous emulator model) | 62,489 (95.35%); 299/384 `vec_mac_bf16`, 772/848 finite `_ovf` |
| fixed-point aligned to the largest leading bit, truncate, RNE, best F=25 | 63,063 (96.23%) |
| aligned to max E, per-term RNE onto `2^(E-22)` / `2^(E-24)` | 87.65% / 94.69% |
| aligned to max E, per-term truncate onto `2^(E-23)` | 88.65% |
| aligned to max E, per-term round-half-away onto `2^(E-23)` | 99.15% |
| **aligned to max E, per-term RNE onto `2^(E-23)`, one RNE (adopted)** | **65,536 (100%)** |

The decisive probe is the decision boundary of the small term: with a large
product in [2, 4) (result ulp 2 grid steps) the up/down decision on the fraction
of the small term was exactly "xi even: up iff fraction >= 0.75; xi odd: up iff
> 0.25", which is RNE onto the half-ulp grid followed by RNE onto the ulp
(double rounding). Large products in [1, 2) show plain correct rounding because
grid and ulp coincide. Cutoff: terms 25 or more exponent steps below the largest
product are dropped.

**Evidence** (captures in `tools/golden/silicon/`, kernels in
`tests/vector-verify/`):

- `vec_mac_bf16_forms` (this work): 23 forms x 336 designed tiles = 123,648
  lanes, bit-exact in the emulator both through `mac_lane` and end to end through
  `execute_matmul`. Sections: gap sweep 0..47 with power-of-two, all-ones, tie and
  sparse small mantissas in both slots and sign relations; permuted multisets;
  accumulators at -34..+33 exponent offsets with tie/all-ones mantissas; edges
  (zero products with huge partner exponent, denormal inputs/accumulators, the
  2^-126 boundary, overflow/carry, signed zeros); sampled Inf/NaN space;
  random K = 8 tiles with exponent spreads 2..60.
- `vec_mac_bf16`, `vec_mac_bf16_ovf`: now exact on all lanes including the NaN/Inf
  tiles that were previously skipped (the earlier gaps are closed).
- Exploration beyond the committed goldens (runtime-input probe, ~700k lanes, all
  reproduced with zero mismatches; data under `build/experiments/r4-d6/`): 52,488
  special-value lanes (9^4 product-pair operands x 8 accumulator operands, four
  elementwise forms), 49,152-lane wild/tiny/huge K = 8 tiles with denormals and
  specials, and the 23-form accumulator sweeps. Discriminating cases: denormal
  accumulator reference exponent -126 (E=-127 or -125 lose 6 and 24 lanes), denormal
  bf16 inputs use field 1 not a normalized exponent (10 lanes), flush before vs
  after rounding (2^-126 boundary tiles), zero products not setting `E`.
- Silicon is deterministic: the forms kernel gave bit-identical output on two
  consecutive HW runs, and every exploration batch run twice agreed on all tiles.

**Timing note.** Float add* forms (`vaddmac.f`, `vaddmsc.f`) retire one cycle
after the other float MACs (7, not the 6 in II_VADDMACf). Chess's schedule for
the forms kernel reads a destination of a `VADDMAC.f` as the accumulator source
of a `VMSC.f` three bundles later and silicon shows the old value, which requires
the write to land after the +3 source read. 7 is the smallest latency consistent
with that; no kernel pins it exactly.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| `vnegmac.f`, `vsubmac.f`, `vsubmsc.f` | no silicon capture (Chess/Peano emit no bf16 intrinsic for them) | `execute_matmul` bf16 gate | gated "[NOT MODELED]" |
| sparse bf16 4x16x4 | no capture | `execute_matmul` sparse gate | gated |
| `vaddmac.f` / `vaddmsc.f` latency | 7 inferred from one write-after-read hazard, not pinned | `execute_matmul` latency | open: needs a dependent-chain kernel at controlled distances |
| fp32-input MAC (emulated through bf16 splits in aie_api) | not examined; the per-term rounding grid is derived for bf16 products | n/a | open |
| other NaN payload sources | payload from a cancelling special pair derived for products and accumulators; NaN produced by an fp32 accumulator whose mantissa has bits beyond bf16 precision is covered only by the sampled capture | `bf16::mac_lane` | closed for the sampled space |

## fp32 accumulator add/sub/neg (`vadd.f`, `vsub.f`, `vnegadd.f`, `vnegsub.f`, `vneg.f`)

**HW-verified** against `tools/golden/silicon/vec_acc_addsub_f32.json`
(`tests/vector-verify/vec_acc_addsub_f32/`, generated by
`tools/gen_acc_addsub_f32.py`): 20 forms x 704 designed tiles x 16 lanes =
225,280 words, bit-identical across three hardware runs. The two identity forms
return the inputs unchanged (no input-path transformation), and the other 18
forms (the five instructions plus their `zero_acc1` / `sub_acc1` / `sub_acc2`
config variants) are reproduced by the emulator on all 202,752 lanes
(`acc_add_matches_silicon_on_every_form_lane`, `vector_matmul/bf16.rs`).

**Rule.** The accumulator adders are the bf16 MAC lane (`mac_lane`) with no
product terms: the two operands are accumulator terms, each with its effective
negation, and everything else is the MAC datapath unchanged -- align to the
larger exponent with per-term round-to-nearest-even onto the `2^(E-23)` grid,
exact integer sum, one RNE to fp32, a sum below `2^-126` flushes to a signed
zero before rounding, an exact zero is `+0`, overflow gives Inf, denormal
operands enter with exponent field 1, and NaN/Inf follow the MAC rules
(NaN encoded `?f800001`, payload only from cancelling specials).

| Instruction | acc1 sign | acc2 sign |
|-------------|-----------|-----------|
| `vadd.f` | + | + |
| `vsub.f` | + | - |
| `vnegadd.f` | - | - |
| `vnegsub.f` | - | + |
| `vneg.f` | - | (unused) |

The config word negates an operand again by XOR: bit 12 (`dynAcc0Neg`) acc1,
bit 13 (`dynAcc1Neg`) acc2; bit 0 (`dynZeroAccum`) drops acc1 (a `vneg.f`
with it set, or any sum with no terms, gives `+0`). The positions come from
llvm-aie's `VecConf` and the constants Chess emits for the `_conf` intrinsics
(`0x101c`, `0x201c`, `0x301c`, `0x1d`); the fp32 mode bits `0x1c` are
irrelevant to the result.

**Hypothesis tested first, and held:** the D6 lane (`mac_lane(&[], accs, false)`)
with the accumulator terms and their negations reproduced 202,752 / 202,752
lanes with no change.

**Rejected alternatives** (on the 10,725 finite-operand `vadd.f` lanes with no
Inf/NaN):

| Model | Exact lanes |
|-------|-------------|
| plain IEEE-754 fp32 add, round-to-nearest-even | 9,658 (90.0%) |
| IEEE add with denormal operands flushed | 9,492 (88.5%) |
| IEEE add with denormal operands and result flushed | 9,670 (90.2%) |
| **MAC-datapath lane (adopted)** | **10,725 (100%)** |

The deciding lanes are the cancellation and tie cases: the larger operand is
exact and the smaller one is rounded to the grid before the sum, so the result
differs from a single IEEE rounding (double rounding), and a nonzero sum below
`2^-126` is `+-0` rather than the smallest normal or a denormal.

**`vneg.f` is not a sign flip.** It is the adder with one operand: a denormal
becomes `-0` (`0x007fffff` -> `0x80000000`), `+0` stays `+0` (not `-0`), and a
NaN is re-encoded with its sign inverted (`0x7fc00000` -> `0xff800001`).
The integer-style path `execute_acc_negate` used before did flush the denormal
but turned `+0` into `-0`, kept a NaN positive and ignored `sub_acc1`; it is
replaced by the shared lane (`float_vneg_is_not_a_bare_sign_flip`).

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| fp32 add/neg result latency | sources are read at issue+2 bundles (operand cycle 3, as for the integer add) and the result lands 6 cycles after issue (`II_VACCf` / `II_VNEGf` operand cycle 0, `FLOAT_ACC_RESULT_LATENCY`). Chess reuses destinations while results are in flight, so the bridge kernel only passes with it (a write at the source read fails 164,717 words). The exact figure is the itinerary value, not pinned by a dependent-chain capture | `PendingAccAdd` | open: no dependent-chain kernel pins 6 against 5 or 7 |
| integer accumulator add/sub config bits | `execute_acc_add_sub` decoded integer `sub_acc1`/`sub_acc2` from bits 11/12 (bit 11 is `dynMulNeg`); TableGen's `VecConf` and the constants Chess emits for `add_conf` (`0x1`, `0x400`, `0x1000`, `0x2000`) put them at 12/13 | `execute_acc_add_sub` | fixed from TableGen (one layout for integer and float, silicon-confirmed on the float forms); integer variants not captured on silicon |
| float vector-register add/sub gate | `float_addsub_not_modeled` (x/w float `Add`/`Sub`/`SubLt`/`SubGe`) is unreachable: no TableGen encoding pairs those semantics with a float element type | `vector_arith.rs` | guard only; `float_add_sub_is_reachable_only_through_the_accumulator_family` asserts it |

## Wide VSHIFT / VSHIFT_ALIGN (512-bit x registers)

**HW-verified** against `tools/golden/silicon/vec_vshift_wide.json`
(`tests/vector-verify/vec_vshift_wide/`, `tools/gen_vshift_wide.py`): operands
A = bytes 1..64 and B = bytes 65..128 (every byte distinct and non-zero, so each
output byte names its source and a zero is a fill), 38 chunks x 64 entries = 2,432
(step, shift) combinations, bit-identical across three hardware runs, all
reproduced by `wide_shift_matches_silicon_on_every_entry` (`vector_misc.rs`).

`vshift d, s1, s2, r`: the 128-byte concatenation `s2:s1` (s1 low) is shifted
down by the shift register and the low 64 bytes kept; `out[k] = cat[shift + k]`,
zero past the end. The register is read as a whole unsigned 32-bit value: 64
returns s2, and 128 or any larger value (0x100, 0x10005, 0x80000000, 0xffffffff,
a "negative" amount) returns zeros, so there is no wrap and no masking.

`vshift.align d, s1, s, s2, r` reads the step register `s0..s3` as `s & 7`
(0x21, 0x41 behave as 1; 0x20, 0x100 as 0). Steps 1..4 select a pre-offset `p` of
4, 8, 16, 32 bytes; steps 0 and 5..7 select none and the result is the plain
shift. With `p` and `shift < 64`, bytes below `64 - shift` come from `s1[p + k]`
(zero past its end) independent of `shift`, and `s2` is laid over from byte
`64 - shift` on (it wins any overlap). For `shift >= 64` the pre-offset has no
effect.

**Rejected:** the narrow `vector_align` model (an unsigned 6-bit shift, 256-bit
operands) is wrong for these forms in three ways: it masks the amount to six bits,
it has no pre-offset, and it concatenates 256-bit halves.

**`vmov w, am*` quarter moves.** Chess loads fp32 operands with `vlda amll/amlh`
and reaches a bm register through `vmov wh, amlh` then `vmov bm, x`. The narrow
accumulator-to-w move took one 32-bit lane from each u64 word (and the reverse
zero-extended), which duplicated lanes and failed `vec_acc_addsub_f32` on the
emulator. A quarter (`QuarterLow` / `QuarterHigh`) is now moved raw as four u64
words of two 32-bit lanes, matching the quarter loads and stores
(`test_vmov_w_acc_quarter_is_a_raw_256_bit_move`). The remaining narrow
`vmov w, am` widths (Half) keep the legacy one-lane-per-word behavior,
unverified.

## Integer MAC modes without a matmul tile (elementwise, convolution, two-channel)

**Silicon-derived** (NPU1, 2026-10-29) from `tools/golden/silicon/vec_macmode_*.json`
(`tests/vector-verify/vec_macmode_*/`, generated by `tools/gen_mac_modes.py`;
decoder `tools/analyze_mac_mode_probe.py`; config words `tools/mac_mode_config_words.py`).
Eight modes `aiev2_vmult.h` builds (the intrinsic stems below) had no geometry: the
emulator rejected their config word and dropped the op with its destination
unwritten. Each now has a closed-form lane map, replayed lane for lane through
`execute_matmul` (`vector_matmul/mode_goldens.rs`) and against the table alone
(`aie2::matmul::tests`): 188,416 probe lanes and 301,056 held-out lanes, all four
signedness flavours, every lane bit-exact; the 16 kernels are bit-identical across
three hardware runs.

**Method.** Per mode, a PROBE kernel and a HELD-OUT kernel. The probe uses that the
multiply is bilinear: with A = e_i (one-hot) lane l reads the sum of B[j] over the
partners j of i, so against B = ones it counts the partners and against the ramp
B[j] = j + 1 it names the partner when the count is one (and the mirror image with
B one-hot). Every (A index, B index) pair of every lane falls out, with its
coefficient; the A-side and B-side decodings agree, no cell had more than one
partner, and all coefficients are 1. A first probe design (B bit planes, 896 tiles)
decoded to the same pair sets before it was replaced by the smaller ramp design.
The held-out kernel (seeded random data over the full element range, 25 constant
and 32 per-element extreme tiles of 0, 1, max, min and -1, then uniform) runs
`mul` -> `mac` -> `msc` chained on the previous result, in ss/su/us/uu, and was
never used to derive a map.

**Lane maps** (element indices; `MacShape::operand_indices`). Parameters come from
the intrinsic family name and are cross-checked against the vector types at build time.

| Mode (amode,bmode,variant) | Family | Kind and parameters | Lane `l` sums |
|---|---|---|---|
| (0,1,1) i8 x i8 -> acc32 | `elem_32_2` | `Elementwise { lanes: 32, k: 2 }` | `A[l + 32t] * B[l + 32t]`, t < 2 |
| (0,3,1) i16 x i16 -> acc32 | `elem_32` | `Elementwise { lanes: 32, k: 1 }` | `A[l] * B[l]` |
| (1,3,2) i16 x i16 -> acc64 | `elem_16_2` | `Elementwise { lanes: 16, k: 2 }` | `A[l + 16t] * B[l + 16t]`, t < 2 |
| (2,3,1) bf16 -> accfloat | `elem_16_2` | `Elementwise { lanes: 16, k: 2 }` | same map (folded in, see below) |
| (0,1,2) i8 x i8 -> acc32 | `conv_4x4_8ch` | `Conv { outputs: 4, taps: 4, channels: 8 }` | `A[(p+t)*8 + c] * B[t*8 + c]`, `l = p*8 + c` |
| (0,1,3) i8 x i8 -> acc32 | `conv_8x8_4ch` | `Conv { outputs: 8, taps: 8, channels: 4 }` | `A[(p+t)*4 + c] * B[t*4 + c]`, `l = p*4 + c` |
| (0,1,4) i8 x i8 -> acc32 | `conv_32x8` | `Conv { outputs: 32, taps: 8, channels: 1 }` | `A[l + t] * B[t]`, t < 8 |
| (1,3,3) i16 x i16 -> acc64 | `conv_16x4` | `Conv { outputs: 16, taps: 4, channels: 1 }` | `A[l + t] * B[t]`, t < 4 |
| (0,2,1) i16 x i8 -> acc32 | `4x4_4x4_2ch` | `MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 }` | `A[(r*4 + k)*2 + ch] * B[(k*4 + c)*2 + ch]`, `l = (r*4 + c)*2 + ch` |

A convolution is a one-dimensional sliding window over channel-interleaved data: A is
the signal (`outputs + taps - 1` positions are read), B the filter taps, and the filter
is per channel. The convolution B operand is read only up to `taps * channels` (conv_16x4
reads B[0..4] of 32; conv_32x8 B[0..8] of 64) and the two-channel mode reads B[0..32] of
its 64 int8 elements; the remaining B elements are never touched. The two-channel mode is
two independent 4x4 matmuls with the channel as the innermost index of A, B and the result.

**Bf16 elementwise folded in.** The hand-coded `lookup_bf16_geometry` variant 1 (16
lanes, `A[L]*B[L] + A[L+16]*B[L+16]`) is the same `Elementwise { lanes: 16, k: 2 }`
kind and now comes from the extracted table; its D6 goldens
(`vec_mac_bf16_forms.json`) still pass bit for bit.

**Signedness and accumulation.** Config bit 9 is X (A) signedness and bit 8 is Y (B),
as for the matmul modes; a signed operand is sign-extended per element, an unsigned
one zero-extended, and the product is exact before summation. All four flavours share
one lane map (the probe relations are identical). Accumulators wrap modulo 2^32 (acc32)
or 2^64 (acc64), no saturation; `mac` adds and `msc` subtracts the product to the
accumulator the previous instruction left, exactly as the matmul modes do. Reading an
acc64 lane back as two int32 halves needs `crSat = none` (`srs` by 0 and by 32).

**Config words Chess emits** (zero_acc = 0, no sub bits; `+ 0x100` sets sgn_y, `+ 0x200`
sgn_x, so `uu` is the base word and `ss` adds `0x300`):

| Mode | `uu` word | decodes to |
|---|---|---|
| elem_32_2 | `0x0028` | (0,1,1) |
| conv_4x4_8ch | `0x0048` | (0,1,2) |
| conv_8x8_4ch | `0x0068` | (0,1,3) |
| conv_32x8 | `0x0088` | (0,1,4) |
| 4x4_4x4_2ch | `0x0030` | (0,2,1) |
| elem_32 | `0x0038` | (0,3,1) |
| elem_16_2 (int) | `0x005a` | (1,3,2) |
| conv_16x4 | `0x007a` | (1,3,3) |

**Rejected alternatives** (exact lanes on the uniform-random held-out tiles, ss, `mul`):

| Mode | Hypothesis | Exact lanes |
|---|---|---|
| elem_32_2 | adjacent pairs `A[2l]B[2l] + A[2l+1]B[2l+1]` | 0 / 1,760 |
| elem_16_2 | adjacent pairs | 0 / 880 |
| conv_4x4_8ch | channel-major planes | 0 / 1,760 |
| conv_4x4_8ch | plain 1-D window, one filter for all lanes | 0 / 1,760 |
| 4x4_4x4_2ch | two stacked matmuls, channel outermost | 0 / 1,760 |
| all eight | adopted maps | 100% |

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| silent drop of a config word with no geometry | `execute_matmul` logged an error and returned "handled" with the destination unwritten | `vector_matmul/mod.rs` | fixed: reports the op unhandled with a "[NOT MODELED]" warning (`a_config_word_with_no_geometry_is_unhandled_and_writes_nothing`); today every mode the header builds has a geometry, so only a config word outside the header can reach it |
| mode forms beyond mul/mac/msc | only `vmul`, `vmac`, `vmsc` were captured per mode; `vnegmul`/`vnegmac`/`vaddmac`/`vsubmac`... run the same lane map with the instruction-level sign and second-accumulator handling the matmul modes already have, but no capture of these modes pins them | `execute_matmul` | open: no silicon evidence beyond mul/mac/msc |
| mode result latency | the new modes use the matmul MAC latency (5); not pinned by a dependent-chain capture | `execute_matmul` | open |

## Unpack: fused `vldb.unpack.*` and standalone `vunpack.*`

**RESOLVED 2026-10-30.** Pinned on NPU1 silicon with four capture kernels from
`tools/gen_vldb_unpack.py` (goldens `tools/golden/silicon/vec_unpack_*.json`, each
bit-identical across three runs and passing against its own capture), replayed lane for lane
by `execute/unpack_goldens.rs`. Both families have four forms, `s8.s4`, `s16.s8`, `d8.d4` and
`d16.d8`; the fused load also has the index, post-modify, 2D and 3D addressing flavours (16
distinct mnemonics plus the shared index/post-modify mnemonic, 20 TableGen records). The
disassembly of the compiled kernels contains every one of them: `vldb.unpack.*` with
`[p, dj]` and `[p], m`, `vldb.2d.unpack.*` and `vldb.3d.unpack.*` with `[p], d0`, and
`vunpack.*` with a w-register source, for all four forms. No TableGen form was unreachable.

**Lane map (all eight instructions, all four forms).** The 256-bit source is `256/bits_in`
packed elements, element 0 in the least-significant bits (low nibble first for the 4-bit
forms). Result lane `i` is element `i`, extended to `bits_out`; the 512-bit result holds
`512/bits_out` lanes, equal in number, so the high half of the result continues the order.
The fused load always consumes 32 bytes and fills the whole X register; the old model read 8
to 16 bytes and wrote 256 bits.

| Form | Source elements | Result lanes |
|---|---|---|
| `s8.s4`, `d8.d4` | 64 nibbles | 64 x 8-bit |
| `s16.s8`, `d16.d8` | 32 bytes | 32 x 16-bit |

**Signedness.** The `s` forms (encoding `sgn = 1`) always sign-extend, with `crUnpackSign`
at 0 or at 1: the register is not read. The `d` forms (`sgn = 0`, `Uses = [crUnpackSign]`)
sign-extend when the register is 1 and zero-extend when it is 0. Only the values 0 and 1 are
observable from C++ (`unpack(v, bool)` writes a bool); bit 0 is what the emulator tests.

**`crUnpackSign` timing.** The register is read late, at pipeline stage 7 as the itineraries
say (operand cycle 7 of II_VUNPACK and II_VLDB_UNPACK*). `vec_unpack_crtiming` drives the sign
from a run-time table inside a software-pipelined loop; Chess schedules the `movx
crUnpackSign` after the `vldb.unpack.d*` and `vunpack.d*` it controls (one bundle after for
the first of an iteration's pair, in the same bundle for the standalone form), and silicon
used the written sign in all 1,536 vectors (6 sign patterns x 64 blocks x 4 vectors). The
emulator samples the sign with the source at issue+6 (`PendingUnpack`), for the fused load as
well as the standalone form. Not pinned: how late the write can be before it misses the read
(only Chess's own schedules, writes up to five bundles after the issue, were observed).

**Fused-load pointer behavior** (`vec_unpack_addr`: 1,888 accesses, 4 forms x 2 sign values):

| Flavour | Load address | Pointer afterwards |
|---|---|---|
| index `[p, dj]` | `p + dj` (bytes) | unchanged |
| post-modify `[p], m` | `p` | `p + m`, the whole modifier in bytes, for `m` in 0, 32..1024, -32..-512; independent of the 32 bytes consumed |
| 2D `[p], d` | `p` | `if dc < dn { p += dj; dc += 1 } else { p += m; dc = 0 }` |
| 3D `[p], d` | `p` | `if dc1 < dn1 { p += dj1; dc1 += 1 } else if dc2 < dn2 { p += dj2; dc1 = 0; dc2 += 1 } else { p += m; dc1 = dc2 = 0 }` |

A descriptor `d<k>` is the register group `m<k>, dn<k>, dj<k>, dc<k>`; the 3D operand `d<k>_3d`
adds `d<k+4>` as the outer dimension (TableGen: `d0 = [m0, dn0, dj0, dc0]`, `d0_3d = [d0,
d4]`). Counts that start at or above their size take the wrap branch (4 sets). The data loaded
and every count matched in all 1,888 records. The rule is in `execute/memory/agu.rs`.

**Replay counts.** Probe: 8,832 vectors / 423,936 lanes; held-out random set: 6,144 vectors /
294,912 lanes; 1,888 addressing accesses; 1,536 timing vectors; all equal to silicon.

**Rejected alternatives** (exact vectors out of 8,832 on the probe, 6,144 on the held-out set,
all forms and both sign values):

| Hypothesis | Probe | Held-out |
|---|---|---|
| adopted: lane `i` = element `i` | 8,832 | 6,144 |
| the high half of the result restarts at element 0 | 336 | 0 |
| the two halves interleave | 336 | 0 |
| adjacent elements swapped | 240 | 0 |
| reversed order | 240 | 0 |
| `d` forms always unsigned (the old `unpack_widths_from_name`) | 6,759 | 4,608 |
| `d` forms always signed | 6,759 | 4,608 |
| `s` forms follow the register | 6,759 | 4,608 |

(The surviving matches of the wrong maps are blocks whose lanes are zero, uniform or
symmetric, and for the sign rules the vectors the hypothesis happens to agree on.)

**What was wrong in the emulator** (fixed): the fused load built a 256-bit result from 8 to 16
bytes; its 4-bit forms fell through to a raw load; it read signedness from the element types,
and the standalone path from the `D` in the name, so `crUnpackSign` was never consulted; the
`BFloat16 -> Float32` arm was reached by no encoding; and two separate lane maps existed. The
control register write to `crUnpackSign` was discarded. One lane map (`unpack_lanes`) now
serves both paths, per-form widths and the sign source come from the archspec form table
(`aie2::unpack`, extracted from TableGen), and the descriptor update of 2D/3D loads exists for
the first time (before, the descriptor operand was added to the address as an offset).

**Open questions.**

| Gap | Model vs hardware | Status |
|---|---|---|
| unaligned fused loads | all captured addresses are 32-byte multiples; the emulator reads from any byte address | open; not captured, to avoid an alignment fault on the shared box |
| `crUnpackSign` values other than 0 and 1 | bit 0 decides; higher bits ignored | open; unobservable from `unpack(v, bool)`, needs a raw register write |
| how late a `crUnpackSign` write can be | writes up to five bundles after the issue are honored | open; only Chess's own schedules observed |
| descriptors other than `d0` / `d0_3d` | register mapping `d<k>` = `m<k>, dn<k>, dj<k>, dc<k>` taken from TableGen; only `d0` appears in the captures | open |
| negative or very large counts and sizes | compared unsigned | open; only 0..7 observed |
| cross-tile fused unpack loads | go through the shared vector read, not captured | open |
| AIE2P | `unpackSign0` / `unpackSign1`: the form table names the register and `PendingUnpack` resolves it by id, so no single register is assumed; not implemented | deferred |
