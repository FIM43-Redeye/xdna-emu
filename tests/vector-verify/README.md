# Half-B vector-compute capture kernels

Targeted AIE2 kernels that exercise the **actual vector intrinsics** (not scalar
loops) so a bridge run can confirm the emulator matches real NPU1 silicon on the
vector-compute classes Half A verified against the aietools model. See
`docs/superpowers/plans/2026-06-08-vector-compute-halfB-silicon.md`.

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
| `vec_srs_i32` | SRS | authored; **EMU-blocked on q-register modeling** (see `docs/superpowers/plans/2026-06-08-q-register-vector-modeling.md`). Surfaced + fixed the wide-cm SRS panic (commit 4472bfb). |
| `vec_ups_i32` | UPS | todo |
| `vec_pack_i16` | Pack | todo |
| `vec_mac_i8` | MatMul int (i8) | todo |
| `vec_mac_i16` | MatMul int (i16) | todo |
| `vec_mac_bf16` | MatMul bf16 | todo |
| `vec_conv_bf16` | Convert (bf16<->f32) | todo |

`vec_eltwise_add` is the proof-of-pattern: once its author -> stage -> compile ->
EMU-smoke loop is green, the rest follow the same template (`runtime_cumsum`-
derived: kernel `.cc` with `aie::` intrinsics, single-tile `aie.mlir` with
shim<->core objectfifos, `test.cpp` host check, `run.lit` build recipe).
