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
| `vec_mac_i8` | MatMul int (i8) | todo |
| `vec_mac_i16` | MatMul int (i16) | todo |
| `vec_mac_bf16` | MatMul bf16 | todo |
| `vec_conv_bf16` | Convert (bf16<->f32) | todo |

`vec_eltwise_add` is the proof-of-pattern: once its author -> stage -> compile ->
EMU-smoke loop is green, the rest follow the same template (`runtime_cumsum`-
derived: kernel `.cc` with `aie::` intrinsics, single-tile `aie.mlir` with
shim<->core objectfifos, `test.cpp` host check, `run.lit` build recipe).
