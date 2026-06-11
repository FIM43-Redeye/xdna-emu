# NaN/Inf-input sweep: bf16 & fp32 elementwise V+V add/sub datapath

**Date:** 2026-06-10
**Status:** Authored offline; silicon capture pending (HW-gated, human-run).
**Scope:** The **elementwise** vector-vector add/sub path
(`src/interpreter/execute/vector_arith.rs` `vector_add`/`vector_sub`,
`BFloat16` and `Float32` lanes). This is **separate** from the fp32
accumulator ALU path (`aie2_acc_fp32_add`, VADD.f/VSUB.f), which is already
silicon-verified (commit 526e0204).

## Why

The emulator's bf16 elementwise add computes naive host `a+b` then a **raw
16-bit truncate** (`f32_to_bf16` = `>>16`, no rounding, no NaN preservation),
collapsing every NaN-input case to a host-FPU-canonical value. The fp32
elementwise path is plain host IEEE `a+b`. Differential fuzzing (seeds
6159/6258, banked `~/npu-work/experiments/phoenix-survival/vector/`) proved
silicon instead treats Inf/NaN operands as **exp-255 normal magnitudes**,
computes the sum, forces the result exponent to `0xFF`, and **preserves the
computed sign + mantissa** (e.g. bf16 `Inf + NaN -> FF8C`, `NaN + Inf ->
FFF8`). That formula is verified **only for Inf+NaN** and is provably
incomplete for other special-operand cases.

We need silicon ground truth across the **full operand-class matrix** before
touching the interpreter, so the model is *reconciled* against the two existing
contradictory fp-add helpers rather than becoming a third contradictory guess:

- `aie2_fp32_add` (vector_float.rs) -- single NaN short-circuits to `+NaN`
  (`0x00000001`, sign forced positive); flushes denormal inputs. **No live
  callers.**
- `aie2_acc_fp32_add` (vector_float.rs) -- exp-255 datapath, keeps the sum's
  sign, sticky mantissa bit 0, no input FTZ. **Silicon-verified** (the acc
  path). This sweep's results should be reconciled toward this one's structure
  where the datapaths agree.

The interpreter fix is **out of scope here** -- modeling waits on the capture.

## Matrix size & shape

- **Operand classes (16):** both signs of {zero, denormal, small-normal,
  large-normal, Inf, qNaN (mantissa MSB set), sNaN (MSB clear + low bits set),
  NaN-with-distinct-mantissa-payload}. The distinct-payload NaN (bf16 `0x7FD5`
  / fp32 `0x7FB55555`) exists to read off which operand's payload propagates.
- **Per (type, op):** full cross product A-class x B-class = **16 x 16 = 256
  lanes**. Lane index `= a_class * 16 + b_class`, so every lane is exactly one
  ordered pair and is self-localizing.
- **(type, op) combinations (4):** {bf16, fp32} x {add, sub}. Four kernel dirs:
  `vec_nan_bf16_add`, `vec_nan_bf16_sub`, `vec_nan_fp32_add`,
  `vec_nan_fp32_sub`.
- **Kernel shape:** single op (one `aie::add`/`aie::sub`) looped across the 256
  inputs (bf16: 32 lanes/vector x 8 iters; fp32: 8 lanes/vector x 32 iters).
  One op family per kernel -- zero pipeline-adjacency confound. bf16 routes the
  op through a `noinline` helper (the fuzzer's GlobalISel workaround); both
  emit native `vadd.f`/`vsub.f` (bf16 via `vconv.fp32.bf16` + `vadd.f`, the
  real datapath).

Total: **1024 captured lanes** (256 x 4), each yielding a `{A_bits, B_bits,
hw_out_bits}` triple.

## Discriminator cases & what each tells us

Every pair below is present in all four matrices (sub flips the operand roles,
exposing sign/payload asymmetry). Bit values shown are the **current
emulator** output (the baked `EMU[]` column); silicon is expected to diverge,
and the divergence *is* the finding.

| Pair (add) | bf16 emu | fp32 emu | What silicon answers |
|---|---|---|---|
| `+Inf + -Inf` | `0xFFC0` | `0xFFC00000` | **Force-0xFF + sign/mantissa rule for opposing Inf.** Acc path gives `0x7F800001` (positive, sticky); does elementwise match? Emu gives host `-qNaN`. |
| `+Inf + +0`, `+Inf + +nlarge` | `0x7F80` | `0x7F800000` | **Inf-vs-NaN boundary / treat-as-normal discriminator.** If silicon treats Inf as exp-255 *normal magnitude*, Inf+finite would NOT stay clean Inf -- it would carry the finite into the mantissa and force exp 0xFF. Emu keeps clean Inf. **The key discriminator** for the treat-as-normal hypothesis. |
| `+nlarge + +nlarge` | `0x7F80` | `0x7F800000` | **Overflow -> Inf vs forced-NaN.** Finite overflow: does silicon produce clean Inf, or force-0xFF with sticky like the NaN path? |
| `+qNaN + +qNaN2` vs `+qNaN2 + +qNaN` | `0x7FD5` vs `0x7FC0` | `0x7FF55555` vs `0x7FC00000` | **Mantissa-propagation rule.** Emu's host FPU keeps the *second* operand's payload on add (first on sub). Silicon's exp-255-datapath *sums* mantissas -- which operand's payload survives, and is it order-dependent? |
| `+qNaN + +0`, `+NaN + finite` | `0x7FC0` | `0x7FC00000` | **NaN absorbs finite?** Confirms whether a single NaN operand forces NaN regardless of the other, and with which sign. |
| `+sNaN + +nsmall`, `+sNaN + +0` | `0x7FE0` | `0x7FE00000` | **sNaN handling.** Host FPU quietens sNaN (sets mantissa MSB). Does silicon's datapath preserve sNaN encoding or quieten? |
| `+denorm + +denorm` | `0x0002` | (normal) | **Denormal input handling.** Acc path does NOT flush denormal inputs; `aie2_fp32_add` does. Which does the elementwise path do? |
| `-0 + +0` | `0x0000` | `0x00000000` | **Signed-zero rule** under round-to-nearest (IEEE: `+0`). |

The **sign rule**, **mantissa-propagation rule**, **force-0xFF condition**, and
**Inf-vs-NaN boundary** are each pinned by a distinct column above:
- *Sign rule* -> opposing-Inf and signed-NaN lanes (`-qnan`, `-snan`, `-inf`
  crosses).
- *Mantissa propagation* -> the `qNaN x qNaN2` ordered pair (and its sub flip).
- *Force-0xFF condition* -> `Inf + finite`, `nlarge + nlarge`, `Inf + Inf`
  (same-sign overflow), contrasted against clean-Inf emu output.
- *Inf-vs-NaN boundary* -> `Inf + finite` staying Inf vs going NaN is the
  single sharpest discriminator for the treat-as-exp-255-normal hypothesis.

## Deliverables / file map

- **Generator (source of truth):** `tools/gen_nan_inf_sweep.py`. Defines the
  class reps, builds the matrix, computes the offline expected column (numpy
  `float32`, bit-exact with Rust `f32`), and emits all four kernel dirs + the
  offline dumps. `--check` flags drift (CI-style). Regenerate with
  `python3 tools/gen_nan_inf_sweep.py`.
- **Kernels:** `tests/vector-verify/vec_nan_{bf16,fp32}_{add,sub}/` --
  `kernel.cc`, `aie.mlir`, `test.cpp`, `run.lit` each. Peano flow
  (`--no-xchesscc`), compile-clean verified offline.
- **Offline expected dumps:**
  `build/experiments/nan-inf-add-sub-sweep/vec_nan_*.expected.txt` -- per-lane
  `lane A_class B_class A_bits B_bits emu_out_bits`, the "what the present
  emulator computes" reference to diff against silicon.

## Silicon capture (human-run, HW-gated)

The `test.cpp` dumps per-lane `A_bits B_bits hw_out_bits` (decimal) to
`out.txt`. Stage the kernels into the mlir-aie discovery root and run the
bridge harness against hardware. **Single command** (after a clean NPU /
sourced env):

```bash
cd /home/triple/npu-work/xdna-emu && \
for k in vec_nan_bf16_add vec_nan_bf16_sub vec_nan_fp32_add vec_nan_fp32_sub; do
  cp -r tests/vector-verify/$k ../mlir-aie/test/npu-xrt/$k
done && \
./scripts/emu-bridge-test.sh --peano-only \
  vec_nan_bf16_add vec_nan_bf16_sub vec_nan_fp32_add vec_nan_fp32_sub
```

The harness compiles (Peano) and runs each on hardware; each run writes its
`out.txt` (the silicon ground truth) into the per-test build dir under
`../mlir-aie/build/test/npu-xrt/<name>/peano/`. The bridge "PASS/FAIL" verdict
is informational only -- divergence from the baked `EMU[]` column is the
*expected, desired* outcome (it is the datapath signal). **Collect the four
`out.txt` files** -- those are the capture.

Post-capture: diff each silicon `out.txt` against the matching
`build/experiments/nan-inf-add-sub-sweep/vec_nan_*.expected.txt` per lane, read
off the datapath rule from the discriminator columns, and reconcile a single
elementwise NaN/Inf model against `aie2_acc_fp32_add` before implementing.

## Out of scope (explicit)

- The interpreter fix (waits on silicon data).
- The fp32 accumulator ALU path (already verified, 526e0204).
- Mixed-op chains / pipeline interactions (single-op by design).
