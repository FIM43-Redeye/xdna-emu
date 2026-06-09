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
| `vec_pack_i16` | Pack | **EMU-smoke PASS** (chess) -- generated; native VPACK int16->int8 (truncating, not saturating). No new interpreter gap. |
| `vec_conv_bf16` | Convert (f32->bf16) | **EMU-smoke PASS** (chess) -- generated; round-narrow via accfloat accumulator, `VST.CONV.bf16.fp32`, ConvEven. Surfaced + fixed an interpreter gap: the convert path hardcoded bit-truncation, ignoring the configured rounding mode (47% of the 438-record golden slice are round-ups it would have failed). Fix threads `ctx.srs_config.rounding_mode` into `vector_convert` / the vst.conv fast path. Needed a generator extension (`Buf.ktype` host-vs-kernel type split for bit-pattern staging; `select_records` predicate for normal-finite-f32 filtering). Denormal/NaN/Inf inputs deferred (HW-gated FTZ + canonicalization edges). |
| `vec_mac_i8` | MatMul int (i8) | **EMU-blocked** (chess) -- generated (first matmul-variant kernel). See the shared MAC-tier gap below. |
| `vec_mac_i16` | MatMul int (i16) | **EMU-blocked** (chess) -- generated (int16, 4x2x8). Same shared gap. |
| `vec_mac_bf16` | MatMul bf16 | **EMU-blocked** (chess) -- generated (bf16, 4x8x4, fp32-bit compare, all-finite slice). Same shared gap. |

### Shared MAC-tier gap (a two-bug cascade)

All three MAC kernels share the same compiled-mmul datapath. Pinned by
memory-watch runs on `vec_mac_i16`, two distinct interpreter bugs gate it.

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

**Bug 2 -- MAC pipeline latency not modeled (OPEN).** Chess software-pipelines
the batch loop: a 2-deep `VMUL cm0` prologue, then a zero-overhead loop whose
body bundles the tile store with the *next* tile's `VMUL cm0` (`VST amhl0,
[p2], #64; VMUL cm0, x0, x1, r24`). Hardware relies on the multi-cycle MAC
latency -- the VMUL result is not visible in `cm0` for several cycles, so the
in-flight stores read the *older* tile's result. The emulator applies the VMUL
write immediately, so every tile is stored one ahead: emu tile N == golden tile
N+1 (verified: emu tiles 5/6/7 == golden tiles 6/7/8). Fix direction: defer the
matmul accumulator write by the MAC latency using the interpreter's existing
`PendingWrite` / `commit_pending_writes` machinery (the latency table already
lists "Vector MAC | 4+ cycles"), so pipelined-loop stores read the correct
prior tile. This is a timing-model change, tracked separately.

`vec_eltwise_add` is the proof-of-pattern: once its author -> stage -> compile ->
EMU-smoke loop is green, the rest follow the same template (`runtime_cumsum`-
derived: kernel `.cc` with `aie::` intrinsics, single-tile `aie.mlir` with
shim<->core objectfifos, `test.cpp` host check, `run.lit` build recipe).
