# Vector-Compute Half B — Silicon Verification (capture-harness prep)

- Date: 2026-06-08. Status: plan for review (no-HW prep; the capture itself is HW-gated).
- Task: #103 Half B. Depends on: Half A (#103, complete — `docs/superpowers/findings/2026-06-08-vector-compute-audit-half-a-rollup.md`) and #104 (complete — the override_registry Accept pattern).

## Goal

Flip the emulator's AIE2 vector-compute verdicts from `AietoolsModeled /
Unverified` (perishable) to `Verified { evidence }` by confirming the emulator
matches **real NPU1 silicon** on each vector class (SRS, UPS, Pack/Unpack,
MatMul/MAC int, MatMul/MAC bf16, bf16 conversion, element-wise). That empties
the perishable queue and turns `clean_release(Aie2)` **green** — the
machine-checked "safe to retire NPU1" signal, the whole point before the
one-way Strix swap.

Half A proved emulator == genuine aietools model across the full discrete mode
space (bit-exact, modulo the HW-correct bf16-NaN payload). Half B is the other
half: emulator == silicon. Together: emulator is faithful to both the modeled
spec and the hardware.

## The blocker this plan exists to solve

**No existing artifact is silicon evidence for the vector datapath.** Inventory
(2026-06-08):

- The ~75-kernel bridge corpus covers scalar arithmetic + DMA + ObjectFIFO well,
  but exercises **no** vector intrinsics. Even `matrix_multiplication_using_cascade`
  computes via **scalar loops**, not `vmac`/`vmul`. `vec_mul_*` kernels call a
  scalar-multiply helper.
- The logic fuzzer (`src/fuzzer/ast.rs`) generates **scalar ops only**
  (ScalarArith / Store / Branch / HwLoop) — no vector ops. Unsuitable.
- The Phoenix-survival corpus (`2026-06-01-phoenix-survival-corpus.md`) is the
  fuzzer-driven scalar regression freeze; it does not cover vector compute.

So Half B must **author a small suite of targeted vector-compute capture
kernels** that exercise each class's actual intrinsic, run them on real NPU1,
and compare. This plan designs that suite + the capture/compare/flip flow.

## Architecture

Three pieces, mirroring the existing infrastructure rather than duplicating it:

1. **Targeted vector kernels** (`mlir-aie/test/npu-xrt/vec_verify_*/` or a local
   `tests/vector-verify/` dir): one kernel per (class, representative mode). Each
   reads an input buffer from DDR, applies the *actual vector intrinsic* (aie::
   API: `aie::mac`, `aie::mul`, `srs`, `ups`, `aie::pack`/`unpack`,
   bf16 paths), and writes an output buffer to DDR. Inputs are drawn from the
   **Half-A golden** so the expected values already exist.
2. **Capture + compare** via the existing bridge harness (`emu-bridge-test.sh`):
   the kernels join the npu-xrt corpus, so the dual-compiler / HW / EMU / compare
   machinery and the `build/bridge-test-results/<date>/<kernel>.<compiler>.{hw,bridge}.result`
   recording come for free. A passing HW+EMU comparison per kernel is the
   citable silicon evidence.
3. **Verified flip** in the coverage model: after the HW run confirms, add
   `override_registry` entries (same mechanism as #104's Accept, but
   `Verification::Verified { evidence }`) for the Vector-category SemanticOps,
   citing the bridge-result path + date. `enforce_coverage` requires
   `shared_from: None` for Verified (silicon never transfers) — satisfied, this
   is direct AIE2 silicon.

## Coverage strategy (honest sampling)

HW runs are expensive; we do NOT re-cover Half A's full mode space on silicon.
The argument for `Verified` from a representative subset:

- Half A: emulator == model across the **full** discrete mode space (bit-exact).
- Half B: emulator == silicon across a **representative** subset per class
  (each class's distinct compute paths + edge inputs).
- Therefore emulator == silicon across the space, to the confidence the
  representative subset affords. The evidence string records exactly which modes
  were silicon-checked, so the claim is auditable, not blanket.

Each kernel's input batch is pulled from the golden (so HW output is compared
against the **emulator** for the verdict, and cross-checked against the **model
golden** to flag any model-vs-silicon divergence like the known bf16-NaN one).

## The capture kernels (the suite to author)

One per row; each loops over a DDR input batch and writes outputs. Modes chosen
as the distinct compute paths, not the full Half-A grid.

| Kernel | Class | Intrinsic path | Representative modes | Inputs from |
|--------|-------|----------------|----------------------|-------------|
| `vec_srs_i32` | SRS | acc->vec `srs` | rounding {floor, even, inf} x {sat,no-sat} x signed | golden `srs` |
| `vec_ups_i32` | UPS | vec->acc `ups` | the 4 size/acc tuples x sat | golden `ups` |
| `vec_pack_i16` | Pack | `aie::pack` 32->16, 16->8 | Trunc + Saturate + SymSat | golden `pack` |
| `vec_mac_i8` | MatMul int | `aie::mac` i8xi8 | 4x8x8 default geom, signed+unsigned | golden `matmul` (Int8) |
| `vec_mac_i16` | MatMul int | `aie::mac` i16xi16 | 4x2x8 default geom | golden `matmul` (Int16) |
| `vec_mac_bf16` | MatMul bf16 | `aie::mac` bf16 | 4x8x4 default geom | golden `matmul` (BFloat16) |
| `vec_conv_bf16` | Convert | bf16<->f32 + `f32_to_bf16` rounding | RNE + a couple modes | golden `bf16_srs` |
| `vec_eltwise` | Element-wise | `aie::add/sub/mul/min/max` | int8/16/32 | golden `vadd_*` etc |

Notes:
- `vec_conv_bf16` doubles as the Convert-class silicon check (Half A treated
  Convert as composite; this gives it a direct datapath touch).
- Sparse MatMul, the bf16 element-wise (16x2x1) geometry, and the acc_cmb=2
  variants are **explicit non-goals** for the first pass (named follow-ups) —
  they are extra geometries, not new compute kinds.

## Verified-flip mechanism (the coverage edit, post-HW)

After the bridge run records PASS for the suite, add to
`crates/xdna-archspec/src/coverage/units.rs::override_registry(Aie2)`:

```
BehavioralUnit {
  id: "aie2.vector_compute.verified",
  claims: Nodes([Semantic{Mac}, Semantic{MatMul}, Semantic{Srs}, Semantic{Ups},
                 Semantic{Pack}, Semantic{Unpack}, Semantic{Convert}, ...]),
  verdict: Verdict { provenance: AietoolsModeled,   // modeled-from + now silicon-checked
                     verification: Verified { evidence:
                       "Half A model-diff (bit-exact, full mode space) + Half B
                        silicon bridge run <date>: vec_{srs,ups,pack,mac_i8,
                        mac_i16,mac_bf16,conv_bf16,eltwise} HW==EMU PASS, results
                        build/bridge-test-results/<date>/. bf16 NaN payload is the
                        documented HW-correct divergence (vector_float.rs)." } },
  shadows_derived: Some("flips the Vector category off its AietoolsModeled/
                         Unverified perishable default to silicon-Verified"),
  shared_from: None,   // AIE2 silicon, not transferred
}
```

Then regenerate artifacts (`gen_coverage_artifacts`); `perishable-queue.md` should
shrink to empty for the vector items and `clean_release(Aie2)` flips green.
Update the mod.rs tests that currently assert the perishable queue is non-empty
(they encode the pre-Half-B state, just like #104 updated the comprehension
ones) — TDD: invert the assertion first (RED), then add the override (GREEN).

## Tasks

1. **Author the capture kernels** (no-HW: write + compile via Peano/Chess, EMU
   smoke each against its golden inputs in-process via `XclbinSuite`). Each
   kernel's EMU output must already match the golden (it's the same emulator
   Half A verified) — a failing EMU smoke means the kernel doesn't exercise the
   path we think. TDD per kernel.
2. **Wire into the bridge corpus** (no-HW): drop into the npu-xrt test dir with
   the host harness that loads golden inputs and checks outputs; confirm
   `emu-bridge-test.sh --no-hw <name>` runs each EMU-side clean.
3. **HW capture run** (HW-gated, Maya/real-NPU): `emu-bridge-test.sh` over the
   suite; collect `*.hw.result` / `*.bridge.result`. Watch for wedging.
4. **Cross-check vs model golden** (post-HW): diff HW outputs against the model
   golden too; any divergence beyond the known bf16-NaN is a finding (could be a
   new fidelity gap or a model artifact — triage).
5. **Flip the verdicts** (no-HW, after evidence exists): add the override,
   regenerate artifacts, update tests, confirm `clean_release` green.

Tasks 1-2 and the Task-5 scaffolding are **no-HW and buildable now**; Tasks 3-4
and the final flip are HW-gated. This plan is the prep; execution of 1-2 can
start immediately, 3-5 wait for a HW session.

## Risks

- **Kernel authoring is the real cost.** Writing IRON/mlir-aie kernels that
  exercise specific vector intrinsics (and compile under Peano *and* Chess) is
  the bulk of the work; the aie::API surface + objectfifo plumbing is
  non-trivial. Mitigation: start from the closest existing programming_example
  (matmul) and specialize; lean on Chess (ground truth) if Peano can't express a
  path.
- **Representative-subset honesty.** The Verified evidence string must name the
  exact modes silicon-checked so `Verified` is not overclaimed. The full-space
  confidence rests on Half A's model agreement, stated explicitly.
- **bf16 NaN (known).** Expect HW == EMU, HW != model on NaN payload. Pre-known,
  not a new finding; the cross-check (Task 4) filters it.
- **HW cost / wedging.** The suite is small (8 kernels) to keep the HW run cheap
  and recoverable.
