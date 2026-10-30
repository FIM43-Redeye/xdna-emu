# Vector-compute capture kernels

Targeted AIE2 kernels that exercise the **actual vector intrinsics** (not scalar
loops) so a bridge run can confirm the emulator matches real NPU1 silicon on the
vector-compute classes.

Most kernels are **generated** from a spec by `tools/gen_vector_kernel.py`
(registry: `tools/vector_kernel_specs.py`). A spec is the config + the
intrinsic body (the IP); the generator derives run.lit/aie.mlir/kernel.cc.

## Data convention

A committed kernel directory holds **no data and no host program**: `run.lit`,
`aie.mlir` and the kernel `.cc` only. The host program is the one shared
`common/vv_host.cpp`, and the data is staged from the golden JSON at the moment
a kernel is built. The golden JSON in `tools/golden/silicon/` is the single
source of truth.

Provenance of the data:

- **Inputs are designed**, not borrowed: `tools/vector_inputs.py` builds each
  array from an edge set derived from the operation's definition (rounding ties
  and near-ties, saturation bounds, sign and byte boundaries, extreme and
  exactly-cancelling matmul tiles) filled up to the buffer size from a seeded
  PRNG. Reproducible from the seeds. The 13 `*_edge_*` / `denorm` / `ovf`
  kernels keep their originally captured inputs in the golden JSON instead.
- **Expected values are observed on NPU1 (Phoenix) silicon**: a HW run's
  `out.txt` is captured by `tools/capture_silicon.py` (or a generator's own
  `capture` subcommand) into `tools/golden/silicon/<kernel>.json` (`kernel,
  class, n, inputs, silicon, provenance`). No expected value comes from a model
  or is transcribed by hand.
- **Packing is each generator's `kernel_data()`**: it returns the input buffers
  and the expected buffer as little-endian bytes in the buffers' host C types.
  The generators' own code paths and the stager call the same function.

Staged files (written into the staged copy, never committed; `*.bin` is
gitignored under this directory in case one lands here):

| File | Content |
|------|---------|
| `test.cpp` | a copy of `common/vv_host.cpp` |
| `in0.bin`, `in1.bin` | raw input buffers, exactly the bytes put into the input BOs |
| `exp.bin` | the expected output (the silicon golden); absent for an uncaptured kernel |

`vv_host.cpp` reads its per-kernel knobs from `-D` flags on the clang line of
the kernel's `run.lit` (written by the generator): `VV_DATA_DIR` (the staged
directory, via `%S`), `VV_NIN` (1 or 2 input buffers), `VV_OUT_T` (output
element C type, used for printing and the compare width) and `VV_NOUT` (output
element count). Input sizes are the `.bin` file lengths. It keeps the
established behavior: the same XRT setup (instruction BO group 1, data buffers
from group 3: inputs then output), a zeroed output buffer, an `out.txt` dump
(one signed or unsigned integer per line, parsed by `capture_silicon.py` and the
generators' capture subcommands), up to ten `Error [i]: got=... != exp=...`
lines, and `PASS!` / `failed (N errors).` with the same exit codes. With no
`exp.bin` it still runs and writes `out.txt`, then fails: the bootstrap run.

Capture flow for a new or changed kernel: generate, stage (no `exp.bin` yet),
run once on HW (the uncaptured run fails by design but `out.txt` is written
first), `capture_silicon.py <kernel> --out-txt <build dir>/out.txt --date
YYYY-MM-DD`, re-stage (now `exp.bin` = the capture), and re-run HW (must pass
against its own capture) and EMU (a mismatch is an emulator finding).

To (re)generate: `python3 tools/gen_vector_kernel.py <name>` (a spec name, a
sweep name, `all`, or `all-sweeps`). In the sweeps, the `_r<N>` suffix is the
crRnd rounding mode and `_s<N>` the crSat saturation mode (the SRS/UPS shift
is fixed at 4). The early kernels (`vec_eltwise_add`) were authored by hand
before the generator.

The kernel directories are the durable, committed source. The bridge harness
(`scripts/emu-bridge-test.sh`) discovers tests only under
`$MLIR_AIE/test/npu-xrt/` and needs a `test.cpp` there, so a kernel is staged
before it is built or run:

```bash
# Stage into the mlir-aie clone (the harness's discovery root): kernel dir,
# shared host as test.cpp, and the data files from the golden JSON
python3 tools/stage_vector_kernel.py vec_srs_i32 vec_mac_i8   # or --all

# EMU-only smoke (no hardware): compile (Chess) + run on the emulator
./scripts/emu-bridge-test.sh --no-hw vec_srs_i32

# Full HW vs EMU comparison (HW-gated, real-NPU session):
./scripts/emu-bridge-test.sh vec_srs_i32

# Remove every staged copy and its mlir-aie/build/test/npu-xrt output
python3 tools/stage_vector_kernel.py --clean
```

MLIR_AIE resolves like the bridge script: `$MLIR_AIE`, else
`$NPU_WORK_DIR/mlir-aie`, else the nearest ancestor of the checkout holding an
`mlir-aie` directory. Re-stage after a recapture or a regeneration; staging
replaces the previous copy. `--clean` touches only the kernels the stager can
stage.

Hand-written kernels (`vec_eltwise_add`, `vec_chain_state_probe`,
`vec_seed6159_clean`, `vec_nan_bf16_{add,sub,dense_add,dense_sub}`,
`vec_nan_fp32_{add,sub,dense_add,dense_sub}`) have no silicon golden baked into
an `EXP`, still carry their own `test.cpp`, and are copied into the mlir-aie
clone by hand (`cp -r`).

A passing HW+EMU bridge result (`build/bridge-test-results/<date>/
vec_eltwise_add.<compiler>.{hw,bridge}.result`) is the citable silicon evidence
for the corresponding vector class (plan Task 3 -> Task 5 Verified flip).

> A small harness change to add `tests/vector-verify/` as a second discovery
> root (so staging into the mlir-aie clone is unnecessary) is a tidy follow-up;
> until then the stager does that step.

## Suite

| Dir(s) | Class | Notes |
|--------|-------|-------|
| `vec_eltwise_add` | element-wise (vadd_Int32) | hand-authored template |
| `vec_srs_i32`, `vec_srs_i32_r{0-3,8-13}_s{0,1,3}` | SRS acc64 -> int16, shift 4 | anchor (floor, saturate) plus 30-point crRnd x crSat sweep. Surfaced a four-bug interpreter cascade early on (wide-cm SRS panic; VMOV-q; fused post-inc dispatch; VLIW vector/accum/mask snapshot). |
| `vec_ups_i32`, `vec_ups_i32_s{0,1}` | UPS int16 -> acc32, shift 4 | anchor plus crSat sweep. |
| `vec_pack_i16`, `vec_pack_i16_sat`, `vec_pack_i16_s{0,1,3}` | Pack int16 -> int8 | crSat none / saturate / sweep. Silicon honors `crSat` for the narrowing pack (VPACK / `vst.pack` use `crSat`). |
| `vec_conv_bf16`, `vec_conv_bf16_r{0-3,8-13}` | Convert f32 -> bf16 | normal finite inputs; crRnd sweep. The convert path honors the configured rounding mode. |
| `vec_conv_bf16_edge_r*`, `vec_convexp_bf16_denorm`, `vec_vfloor_bf16_denorm` | convert edges | denormal / NaN / Inf f32, exhaustive bf16 denormals. |
| `vec_mac_bf16_forms` | bf16 MAC forms (23 forms x 336 tiles) | own generator `tools/gen_mac_bf16_forms.py` (not a spec): every vmul/vmac/vmsc/vaddmac form, dense 4x8x4 and elementwise, accumulators, config sub/zero controls, denormals, flush, Inf/NaN. Golden holds all four inputs and a `forms` table. |
| `vec_mac_i8`, `vec_mac_i16`, `vec_mac_bf16`, `vec_mac_bf16_ovf` | MatMul (mmul tiles) | drove the MAC-tier cascade below; `_ovf` carries bf16 overflow tiles. |

Further kernels: `vec_srs_i32_sh<S>_r<R>[_s0]` (SRS shift sweep: shifts
0,1,2,3,5,8,12,16,24,31 at floor / symmetric_ceil / conv_even with saturation
on, plus conv_even with saturation off), `vec_srs_i32_r{4,5,6,7,14,15}_s1` and
`vec_conv_bf16_r{4,5,6,7,14,15}` (crRnd values the aie_api enum does not name,
driven by a raw cast), and `vec_shuffle_mode` (VSHUFFLE under a runtime table of
mode numbers: named controls and 44..63; operand bytes are 1..128 so each output
byte names its source and zero is distinguishable).

Validation state (2026-10-28, chess): all 119 generated kernels pass on NPU1
hardware against their own silicon capture, bit-identical across two
consecutive full runs. On the emulator, `vec_mac_bf16`, `vec_mac_bf16_ovf` and
`vec_mac_bf16_forms` now pass (2026-10-28, datapath derived from these captures); the twelve
undefined-crRnd kernels (4-7, 14, 15; SRS and convert) mismatch. Re-run the
bridge for current numbers: the emulator moves.

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

**Bug 3 -- integer matmul inputs mixed signedness (FIXED).** With bugs 1-2
fixed the emulator computed every tile's textbook product, but some i8/i16
tiles still failed: the signed `int8` kernel reads a high-bit byte (0xFF) as -1
while the reference for those tiles had treated the same bytes as unsigned. The
integer matmul inputs are now designed as signed values over the full range
(`vector_inputs.matmul_int_inputs`), so the kernel and the harness agree on
signedness by construction.
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
See `xdna-emu-priv: research/plans/2026-06-09-vector-write-result-latency.md`.

The accumulator/CM-domain (`VEC_Bypass`) file is NOT in this matrix -- those
MAC/MUL results use the separate MAC-pipeline-latency path
(`queue_matmul_accum_write`). The integer MAC kernels never hit bug 5 -- they
store straight from the (already deferred) accumulator with no intermediate
`VMOV -> x -> store`.
**After bugs 4-5: vec_mac_bf16 EMU-smoke PASS.**

`vec_eltwise_add` is the proof-of-pattern: once its author -> stage -> compile ->
EMU-smoke loop is green, the rest follow the same template (`runtime_cumsum`-
derived: kernel `.cc` with `aie::` intrinsics, single-tile `aie.mlir` with
shim<->core objectfifos, `test.cpp` host check, `run.lit` build recipe; the
generated kernels use the shared `common/vv_host.cpp` as that host check).
