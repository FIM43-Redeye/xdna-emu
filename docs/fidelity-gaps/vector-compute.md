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

## bf16 fresh matrix multiply (`vmul.f`)

`vmul.f` (fresh 4x8x4 bf16 x bf16 -> fp32, no accumulator input) is modeled in
`vector_matmul::matmul_bf16_fresh` as an exactly-accumulated dot product rounded
once to fp32 (RNE). Result latency is the float MAC latency (6). Accumulating
bf16 forms (`vmac.f`, `vmsc.f`, ...) remain unmodeled.

Against the NPU1 captures (`tools/golden/silicon/vec_mac_bf16{,_ovf}.json`,
EMU bridge run 2026-10-28):

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| fp32 accumulation of wide-exponent-spread tiles | silicon differs from the correctly rounded dot product in 85 of 384 `vec_mac_bf16` lanes (79 by 1 ulp, 6 by up to 8 ulp; first at idx 99) and 7 of 224 finite `_ovf` lanes (up to 3 ulp); tiles 0-5 are bit-exact. Per-product fp32 rounding with sequential or pairwise sums, fixed-point alignment to the max exponent (any guard width, truncate or floor), and grouped-then-summed forms were all tried and fit worse | `matmul_bf16_fresh` | open: internal alignment/truncation not derived; pinned by `test_vmul_f_known_gap_wide_exponent_tiles_within_few_ulps` |
| NaN/Inf-input encoding in the matmul | 640 of 864 `_ovf` lanes (tiles with NaN inputs) differ: silicon returns NaNs with payload 1 or a datapath payload (`0x7f800001`, `0xff800001`, `0xff905901`) whose sign follows the product signs; we return the host-canonical NaN | `matmul_bf16_fresh` | open: same two-regime question as the `vadd.f` NaN payload above |
