# Half-B vector-compute capture kernels

Targeted AIE2 kernels that exercise the **actual vector intrinsics** (not scalar
loops) so a bridge run can confirm the emulator matches real NPU1 silicon on the
vector-compute classes Half A verified against the aietools model. See
`docs/superpowers/plans/2026-06-08-vector-compute-halfB-silicon.md`.

Most kernels are **generated** from a spec by `tools/gen_vector_kernel.py`
(registry: `tools/vector_kernel_specs.py`). A spec is the config slice + the
intrinsic body (the IP); the generator derives run.lit/aie.mlir/test.cpp/kernel.cc
and bakes the golden input/expected arrays from the Half-A corpus
(`tools/golden/vector_ops.json`) so no expected value is hand-transcribed. To
(re)generate one durably here: `python3 tools/gen_vector_kernel.py <name>`
(or `all`). The early kernels (`vec_eltwise_add`, `vec_srs_i32`) were authored by
hand before the generator and remain the template it was modeled on.

These are the durable, committed source of truth. The bridge harness
(`scripts/emu-bridge-test.sh`) discovers tests only under
`$MLIR_AIE/test/npu-xrt/`, so to build/run a kernel, stage it there first:

```bash
# Stage into the mlir-aie clone (the harness's discovery root)
cp -r tests/vector-verify/vec_eltwise_add \
      ../mlir-aie/test/npu-xrt/vec_eltwise_add

# EMU-only smoke (no hardware): compile (Chess) + run on the emulator
./scripts/emu-bridge-test.sh --no-hw vec_eltwise_add

# Full HW vs EMU comparison (HW-gated, real-NPU session):
./scripts/emu-bridge-test.sh vec_eltwise_add
```

A passing HW+EMU bridge result (`build/bridge-test-results/<date>/
vec_eltwise_add.<compiler>.{hw,bridge}.result`) is the citable silicon evidence
for the corresponding vector class (plan Task 3 -> Task 5 Verified flip).

> A small harness change to add `tests/vector-verify/` as a second discovery
> root (so staging into the mlir-aie clone is unnecessary) is a tidy follow-up;
> until then the `cp` step above stages each kernel.

## Suite (plan Task 1)

| Dir | Class | Status |
|-----|-------|--------|
| `vec_eltwise_add` | element-wise (vadd_Int32) | **EMU-smoke PASS** (chess, 18.5s) |
| `vec_srs_i32` | SRS | **EMU-smoke PASS** (chess) -- first compiled vector kernel through the decode->execute interpreter. Surfaced + fixed a four-bug cascade (wide-cm SRS panic 4472bfb; VMOV-q 5570f5d; fused post-inc dispatch + get_address 960975a; VLIW vector/accum/mask snapshot 13534bc). See `docs/superpowers/plans/2026-06-08-q-register-vector-modeling.md`. |
| `vec_ups_i32` | UPS | **EMU-smoke PASS** (chess) -- first GENERATED kernel (`tools/gen_vector_kernel.py`). Surfaced + fixed two interpreter gaps: wide acc32->x VMOV as a raw 512-bit reinterpret (80e7ba7), and same-bundle scalar->UPS-shift forwarding (724548e, E1->E7 per llvm-aie AIE2Schedule.td). |
| `vec_pack_i16` | Pack (truncating) | **HW==EMU PASS** (chess) -- generated; `set_saturation(none)` int16->int8 narrow takes the low byte. |
| `vec_pack_i16_sat` | Pack (saturating) | **HW==EMU PASS** (chess) -- hand-authored companion; `set_saturation(saturate)`. Confirmed on silicon that AIE2 narrowing pack honors `crSat` (`VPACK/vst.pack Uses=[crSat]`): HW saturates int16->int8 to [-128,127]. Surfaced + fixed an interpreter gap -- both `execute_pack` (standalone) and the fused `vst.pack` path (`memory/mod.rs`) hardcoded `PackMode::Truncate`; now derive the mode from `ctx.srs_config` via `PackMode::from_sat_flags`. |
| `vec_conv_bf16` | Convert (f32->bf16) | **EMU-smoke PASS** (chess) -- generated; round-narrow via accfloat accumulator, `VST.CONV.bf16.fp32`, ConvEven. Surfaced + fixed an interpreter gap: the convert path hardcoded bit-truncation, ignoring the configured rounding mode (47% of the 438-record golden slice are round-ups it would have failed). Fix threads `ctx.srs_config.rounding_mode` into `vector_convert` / the vst.conv fast path. Needed a generator extension (`Buf.ktype` host-vs-kernel type split for bit-pattern staging; `select_records` predicate for normal-finite-f32 filtering). Denormal/NaN/Inf inputs deferred (HW-gated FTZ + canonicalization edges). |
| `vec_mac_i8` | MatMul int (i8) | **EMU-smoke PASS** (chess) -- first matmul-variant kernel. Drove the MAC-tier cascade below (bugs 1-3). |
| `vec_mac_i16` | MatMul int (i16) | **EMU-smoke PASS** (chess) -- int16, 4x2x8. |
| `vec_mac_bf16` | MatMul bf16 | **EMU-smoke PASS** (chess) -- bf16, 4x8x4, fp32-bit compare, all-finite slice. Drove bugs 4-5 (float MAC latency + the general vector-write-latency model). |

### Shared MAC-tier gap (a five-bug cascade)

All three MAC kernels share the same compiled-mmul datapath. Pinned by
memory-watch runs on `vec_mac_i16` (bugs 1-3) and instrumented bf16 bridge
runs (bugs 4-5), a cascade of five bugs gated the tier; all now fixed.

**Bug 1 -- matrix-VMUL routed to elementwise Mul (FIXED).** Chess lowers
`aie::mmul` to **`VMUL cm0, x0, x1, r24`**: a matrix-multiply VMUL with a `cm`
(matrix accumulator) destination and the matmul shape/mode in the `r24` config
word (read back via `VST amhl0`). In AIE2 there is no elementwise VMUL, but the
Pat<>-based inference assigned `SemanticOp::Mul`, so it never reached
`vector_matmul::execute_matmul` and the `cm` accumulator was never written --
the core stored the poison init (`0xDEADBEEF`). Fix (two parts):
  - `resolver/semantic_inference.rs::refine_matmul_semantic` upgrades the
    matrix-VMUL name (`VMUL_..._cm_core_...` / `..._bm_core_...`) from `Mul` to
    `MatMul`, so it routes to `execute_matmul` (which already decodes the `r24`
    config word via `MatMulConfig::from_config_word`).
  - `execute_matmul` force-zeros the accumulator for the fresh-multiply forms
    (`MatMul`/`NegMul`): VMUL encodes `acc1 = 0b1111` (no accumulator input), so
    it zeroes structurally -- the compiled config word has `zero_acc=0`
    (accumulate), so honoring only the config word would leak the destination
    accumulator's stale contents. VMAC-family forms keep config-driven
    accumulate.

  After Bug 1, the arithmetic is correct (every tile's `A.B` matches the golden)
  but the result is shifted by exactly one tile (see Bug 2).

**Bug 2 -- MAC pipeline latency not modeled (FIXED).** Chess software-pipelines
the batch loop: a 2-deep `VMUL cm0` prologue, then a zero-overhead loop whose
body bundles the tile store with the *next* tile's `VMUL cm0` (`VST amhl0,
[p2], #64; VMUL cm0, x0, x1, r24`). Hardware relies on the multi-cycle MAC
latency -- the VMUL result is not visible in `cm0` for several cycles, so the
in-flight stores read the *older* tile's result. The emulator applied the VMUL
write immediately, so every tile was stored one ahead: emu tile N == golden tile
N+1 (verified: emu tiles 5/6/7 == golden tiles 6/7/8). Fixed by deferring the
matmul accumulator write by the MAC result latency (`LATENCY_VECTOR_MAC` = 5,
llvm-aie II_VMAC operand_cycles[0]) through the `PendingWrite` /
`commit_pending_writes` machinery, so pipelined-loop stores read the correct
prior tile. `commit_pending_writes` runs at cycle start with `<=` ready-cycle
semantics -- the hardware def-use model.

**Bug 3 -- golden slice included unsigned-input records (FIXED, golden-only).**
With bugs 1-2 fixed the emulator computed every tile's textbook product, but the
i8/i16 bridge still failed on a few tiles. Root cause was the GOLDEN slice, not
the emulator: the Half-A `matmul` corpus sweeps `x_signed`/`y_signed` in {True,
False} per geometry while storing `a_type:"Int8"` regardless, so the filter
`{a_type:Int8, ...}` also selected unsigned-sampled records. The signed `int8`
kernel reads a high-bit byte (0xFF) as -1, but those records' `expected` was
computed from 255 -> mismatch on high-bit values. Fixed by adding
`x_signed:True, y_signed:True` to the i8/i16 golden filters (batch 64 -> 48,
since only 50 signed records exist). A generator test
(`test_integer_matmul_specs_golden_matches_textbook`) now asserts each integer
matmul spec's baked C equals the textbook product of its baked row-major A/B.
**After bugs 1-3: vec_mac_i8 and vec_mac_i16 EMU-smoke PASS.**

**Bug 4 -- float MAC result latency is 6, not 5 (FIXED).** `vec_mac_bf16` runs
the same compiled path; the matmul value is correct (the uniform-1.0 tile
computes 8.0), so the original "element type not inferred" hypothesis was wrong
-- `VMUL_F_vmac_bm_core_dense` already resolves to `sem=MatMul, et=Float32`
(verified against the real TableGen), and `execute_matmul` takes the bf16
datapath. The real timing finding: float vector MAC/MUL has result latency **6**
(llvm-aie `II_VMULf`/`II_VMACf` operand_cycles[0]=6, the extra `EmptyCycles<4>`
vs the integer classes' `<3>` is the float-normalization stage) vs integer 5.
Fixed by deriving `VECTOR_MAC_F=6` in archspec and selecting it in
`execute_matmul` when `config.bfloat`. (Correct, but not sufficient on its own --
see bug 5.)

**Bug 5 -- vector-register result visibility not modeled (FIXED, AIE2
bypass/forwarding network).** The bf16 accumulator-drain idiom is two bundles:
`VST wh2; VMOV x2,bml0; VMUL.f bml0` then `VST wl2`. `VMOV_mv_x` (BM->X) has
result latency 2 (`II_VMOV_X_BM_XM`) with `NoBypass`, but the emulator applied
the `VMOV x2` write immediately. So the same-bundle `wh2` store read x2 via the
per-bundle snapshot (old tile, correct), but the next-bundle `wl2` store read
x2 *after* the VMOV overwrote it (next tile) -- each output tile-slot came out
split: high half from tile K, low half from tile K+1.

The fix is the **AIE2 per-operand bypass/forwarding network** derived from
llvm-aie TableGen itineraries. Every vector-register write records
`(reg, value, issue_bundle, l_def, def_bypass)` as an in-flight `PendingVecWrite`.
Each read resolves its own visibility via:
```
match      = (def_bypass == use_bypass) && def_bypass != NoBypass
eff        = max(1, l_def - use_cycle + 1 - (match ? 1 : 0))
visible_at = issue_bundle + eff
```
Producer `def_bypass`/`l_def` and consumer `(use_cycle, use_bypass)` come from
the LLVM itinerary, resolved per-instruction (not per-opcode-class) by the FFI
decode path so that register-pair-variant opcodes like `VMOV_mv_x` are handled
correctly (e.g., X<-BM carries `MOV_Bypass`; the static base class does not).
Store-data reads use `NoBypass`; vector ALU reads use `MOV_Bypass`. This subsumes
the per-bundle snapshot hazard and models cross-bundle forwarding. Validated
against the chess bridge sweep (89/89 PASS, 0 regressions) and against real
Phoenix (NPU1) silicon: `vec_mac_bf16`, `two_col`, and all three
`matrix_multiplication_using_cascade` variants PASS HW==EMU on the bridge
(2026-06-09); `vec_mac_bf16` and `two_col` also diff CLEAN at trace granularity.
See `docs/superpowers/plans/2026-06-09-vector-write-result-latency.md`.

The accumulator/CM-domain (`VEC_Bypass`) file is NOT in this matrix -- those
MAC/MUL results use the separate MAC-pipeline-latency path
(`queue_matmul_accum_write`). The integer MAC kernels never hit bug 5 -- they
store straight from the (already deferred) accumulator with no intermediate
`VMOV -> x -> store`.
**After bugs 4-5: vec_mac_bf16 EMU-smoke PASS.**

`vec_eltwise_add` is the proof-of-pattern: once its author -> stage -> compile ->
EMU-smoke loop is green, the rest follow the same template (`runtime_cumsum`-
derived: kernel `.cc` with `aie::` intrinsics, single-tile `aie.mlir` with
shim<->core objectfifos, `test.cpp` host check, `run.lit` build recipe).
