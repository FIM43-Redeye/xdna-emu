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
| Accumulator/CM-domain (`VEC_Bypass`) result visibility | not in the per-operand bypass matrix; modeled via the separate MAC-pipeline-latency path (`ExecutionContext::queue_matmul_accum_write`). The matrix resolves W/X-file results; `VEC_Bypass` (MAC/MUL results into the accumulator file) is deferred. | `src/interpreter/state/registers.rs` (`VectorRegisterFile::visible_at`); `src/interpreter/state/context.rs` (`queue_matmul_accum_write`, carries a `FIXME(bypass-model)`) | **OPEN / deferred.** The MAC-latency path is HW-validated *functionally* -- all three `matrix_multiplication_using_cascade` variants PASS HW==EMU on the bridge (2026-06-09), which exercises the accumulator-write path -- but not yet at trace granularity (those kernels' TRACE PREP fails for an orthogonal trace-injection reason). Accumulator folding into the per-operand bypass matrix is a noted future extension (see [`2026-06-09-vector-write-result-latency.md`](../superpowers/plans/2026-06-09-vector-write-result-latency.md) Scope). **Partially advanced 2026-06-10 (`430d841`):** the `VMOV bml,x` (wide vector->accumulator move) write is now deferred by its def latency via the same `queue_matmul_accum_write` path, after the phase-B `vec_conv_bf16_edge` silicon kernels proved an immediate write corrupts delay-slot `VST.CONV` stores (read freshly-overwritten accumulator -> Inf/NaN garbage). All 10 convert modes EMU==silicon. The full bypass-matrix fold (distinguishing `VEC_Bypass` MAC->MAC from `NoBypass` store consumers) is still the open extension. |
| `source_forward` positional alignment for merged memory operands | `SlotOp::source_forward` aligns exactly for compute ops; may drift for fused memory operands that merge address/data into a single TableGen operand list position | `src/interpreter/bundle/slot.rs` (`FIXME(source-forward-memory-alignment)`); `src/interpreter/decode/operand_extraction.rs` | **DOCUMENTED / benign.** The bypass model only reads vector-register compute sources; memory operands use `NoBypass` or are not vector-reg reads, so this positional drift does not affect result visibility decisions. |

## Special-value NaN payload (`vadd.f`/`vsub.f`, bf16 + fp32)

**RESOLVED 2026-06-11** -- full mechanism and evidence in
[`2026-06-11-nan-payload-datapath-regime.md`](../superpowers/specs/2026-06-11-nan-payload-datapath-regime.md).

The result NaN payload when an operand is a NaN has **two real silicon regimes**:
a **datapath regime** (dominant) where the payload is the actual exp-255
mantissa-datapath sum `r`, and a **canonical regime** (rare) that forces payload
1. The regime is residual float-unit HW state -- global across lanes,
deterministic within a session, but it varies across sessions and survives a
driver reload. This is **not** bf16-specific, **not** two-NaN-operand-only, and
**not** in the bf16 narrow -- it is the shared `aie2_acc_fp32_add` datapath
(fp32 shows it identically), and `+Inf + -NaN` is exactly the trigger.

The model now produces the **datapath regime** (Maya's call: it is what the ALU
computes; canonical is a residual-state suppression). One change in
`aie2_acc_fp32_add`: `use_r = (a_nan || b_nan) && !overflow` lets the datapath
sum `r` through instead of zeroing it. Validated **8160/8160** against fresh
dominant-regime NPU1 captures (bf16+fp32, add+sub); `cargo test --lib`
3408/3408; seed 6159 via EMU now reproduces `0xFF8C`.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| special-value NaN payload: datapath vs canonical regime | HW has two residual-state regimes for `vadd.f`/`vsub.f` NaN-input payloads; we now model the dominant **datapath** regime (payload = exp-255 mantissa sum `r`). HW in the rare **canonical** regime forces payload 1 and would mismatch the payload bits (a HW-state artifact, not a model bug). | `src/interpreter/execute/vector_float.rs` (`aie2_acc_fp32_add`, `use_r` gate). | **RESOLVED (datapath regime) 2026-06-11.** Validated 8160/8160 vs dominant-regime HW. Residual open: *what* flips the regime is uncharacterized (not driver reload); low priority since the payload is functionally dead. See the design note. |
