# bf16/fp32 special-value NaN-payload: the two silicon regimes

**Date:** 2026-06-11
**Status:** model fix landed (interpreter now reproduces the dominant regime);
this note records the mechanism and the evidence.
**Related:** #115 (bf16/fp32 elementwise add/sub NaN-input silicon model),
#114 (vector fuzzer Tier 1), `2026-06-10-nan-inf-add-sub-sweep-design.md`,
`docs/known-fidelity-gaps.md`.

## TL;DR

The NaN **payload** bits produced by `vadd.f` / `vsub.f` (the bf16/fp32
accumulator add/sub datapath) when an operand is a NaN are **not** a single
fixed value on silicon. NPU1 exhibits two distinct, real regimes:

- **Datapath regime (dominant):** the result NaN payload is the actual
  exp-255 mantissa-datapath sum -- the same `r` our `aie2_acc_fp32_add`
  already computes. E.g. `+Inf + (-NaN 0x46)` -> bf16 `0xFF8C`
  (`0x800000 - 0xC60000` normalized = mantissa `0x0C0000`). On mantissa
  **overflow** (same-sign Inf+NaN, carry-out) silicon canonicalizes the
  payload to 1.
- **Canonical regime (rare):** every such payload is forced to 1
  (`0x..81`-class), regardless of the datapath sum.

Both are real silicon output of the **same binary**. The regime is selected by
**residual float-unit hardware state** that is global (all lanes agree),
deterministic within a session, and **survives a driver reload** (`modprobe -r
amdxdna && modprobe amdxdna` did not flip it). The payload is functionally
irrelevant (always a NaN); only the 7 payload bits differ.

**Decision (Maya, 2026-06-11): model the datapath regime.** It is what the ALU
actually computes; the canonical regime is a residual-state suppression. The fix
is a two-line change in `aie2_acc_fp32_add`.

## How it surfaced

Vector-fuzzer seed 6159, lane 29: `h4 = aie::add(v3, p3)` where
`v3[29] = 0x7F80 (+Inf)`, `p3[29] = 0xFFC6 (-NaN payload 0x46)`. NPU1 produced
`0xFF8C`; the interpreter produced `0xFF81`. The committed "silicon-verified"
dense add/sub sweep, by contrast, recorded canonical `0xFF81` for that exact
pair -- a direct contradiction.

## Investigation (what each step ruled out)

1. **Operands / order / config.** Disassembly: seed `h4`, the dense `vop`, and a
   purpose-built isolated `add32` kernel are byte-identical
   (`vconv.fp32.bf16; vadd.f ...,r0=0x1c; vst.conv.bf16.fp32`). `r0 = 0x1c` is
   set fresh before every float op. Not the cause.
2. **Cross-lane / position / neighbors.** A new isolated single-add oracle
   (`tools/gen_add32_oracle.py`, reads any 32+32-lane input from `input.bin`)
   fed the pair with zeroed neighbors, at all 32 lanes, and with the seed's
   exact vectors -- HW gave `0xFF8C` in every case. Not cross-lane.
3. **aiesim is not a faithful oracle here.** AMD's cluster model returns
   `0xFF8C` *unconditionally* (lane/neighbor independent) -- a fixed rule. It
   matched the seed by always returning the datapath value; it would **not**
   match the canonical-regime HW. (Corrects the earlier
   `reference_aiesim_bridge_introspection` claim that the VCD exposes vector
   registers -- it does not; only I/O ports + an undriven debug bus. The closed
   `MathEngineBase` ABI exposes no AIE-core register peek; the only gdbserver is
   shim-microblaze-only.)
4. **The regime is residual HW state.** The committed golden (captured 12:40,
   real HW per `capture.log`) is canonical for all 127 Inf+NaN payloads. A fresh
   run of the **identical binary** at 15:33 -- and three reruns, and a run after
   a driver reload -- all gave the datapath payloads, differing from the golden
   in 120/127 rows. Same silicon, same binary, different session, different
   regime. Driver reload did not restore canonical, so the state is deeper than
   the driver.

## The mechanism (and why our model already had the answer)

`aie2_acc_fp32_add` (`src/interpreter/execute/vector_float.rs`) computes the
exp-255 mantissa-datapath sum `r` for every case, then **discards it** for
inf-involved results: `rr = (if out_zeros { r } else { 0 }) | nan`. For a NaN
input `out_zeros` is false (an Inf flag or the overflow flag is set), so the
mantissa is zeroed and only the `| nan` sticky bit remains -> canonical
`0x..81`. That is exactly the **canonical regime**. The datapath sum `r` -- the
value silicon exposes in the **dominant regime** -- was sitting right there,
thrown away.

Hand-derivation, confirmed numerically:
`+Inf + (-NaN 0x46)` -> `0x800000 - 0xC60000 = -0x460000` -> normalize ->
mantissa `0x0C0000` -> bf16 `0xFF8C`. `+Inf + (+NaN 0x04)` (same sign) ->
`0x800000 + 0x840000` overflows -> canonical `0x7F81`.

### Architecture corroboration (AM020 -- mechanism read, not just inferred)

The datapath our model uses is documented verbatim in AM020 ch.4
("Floating-Point Vector Unit", line 290-291):

> "The accumulator unit supports addition/subtract/negate of accumulator
> registers in a single-precision FP32 format. All floating-point additions are
> done in one go, by aligning all mantissas to the one with the largest exponent
> and with 23 bits of fractional bits. The FP normalization unit handles the
> cases where the mantissa coming from the post-adder is negative and if the
> mantissa is outside the acceptable range."

That maps onto `aie2_acc_fp32_add` term for term: one FP32 post-adder all adds
share (hence bf16 and fp32 take the *same* path), align-to-largest-exponent,
23-bit fractional, normalize (handles negative mantissa = our sign logic,
out-of-range = our `overflow` branch). Line 297: "Denormalized numbers are not
supported by the AIE-ML floating-point data path" = our FTZ. Critically, the
datapath is described as **uniform** ("all floating-point additions are done in
one go") with NaN/`invalid` handling in a *separate* exception-event path (line
295) -- the architecture documents **no canonicalization step in the adder**. So
the post-adder mantissa simply *is* the result: that is why the datapath regime
exposes `r`, and the canonical regime (payload forced to 1) is the residual-state
*deviation* from the documented default, not the default. AM027 ch.4 carries the
identical datapath to AIE-ML v2 (AIE2P / Strix).

The mechanism is thus triangulated three ways -- AM020 (datapath structure, read),
AMD's cluster model (computes the same `r`), and silicon (8160/8160 in the
datapath regime). The "inferred vs read" gap is closed; only AMD's exact RTL for
payload *exposure* remains unread (in the protected model + silicon), and it is
empirically pinned. Disassembling the cluster `.so` would add a fourth
confirmation of an already-triangulated fact -- not pursued.

## The rule and its validation

```
use_r = (a_nan || b_nan) && !overflow
rr    = (if out_zeros || use_r { r } else { 0 }) | u32::from(nan)
```

Validated **8160/8160** against fresh dominant-regime NPU1 captures, all lanes,
all four sweeps:

| sweep | datapath-`r` model | canonical (old) |
|-------|--------------------|-----------------|
| bf16 add | 2048/2048 | 1688/2048 |
| bf16 sub | 2048/2048 | 1688/2048 |
| fp32 add | 2032/2032 | 1672/2032 |
| fp32 sub | 2032/2032 | 1672/2032 |

Post-fix: `cargo test --lib` 3408/3408; seed 6159 via EMU -> `0xFF8C`
(REPRODUCES); dense bf16 add EMU vs dominant HW golden 2033/2033.

## Caveats / open

- **HW validation is regime-dependent.** The model `r` is deterministic, but any
  *committed HW golden* of NaN payloads is only valid in the regime it was
  captured in. Goldens used to validate this change were captured in the
  dominant regime (`build/experiments/special-value-dense/*.dominant.hw.txt`).
  A bridge run while HW is in the canonical regime would mismatch the payload
  bits -- not a model bug.
- **What flips the regime is unknown.** Not driver reload. Candidates: array
  reset / CDO init path (the 12:40 capture went through the bridge harness;
  the datapath-regime runs were bare `test.exe`), cold-NPU / power state,
  a specific prior kernel. Worth a follow-up if regime control is ever needed;
  the payload is functionally dead, so it is low priority.

## Tools added

- `tools/gen_add32_oracle.py` + `tools/run_add32_oracle.sh` -- reusable
  single-add oracle: one `test.exe` reading an arbitrary 32+32-lane `input.bin`,
  runnable on HW / interpreter / aiesim. The instrument that isolated the
  datapath behavior.
- `tools/gen_h4_isolated_probe.py` -- re-authors seed 6159's `h4` as a clean
  single-add fed its exact vectors (the cross-lane classifier).
- `crates/xdna-emu-ffi/tests/aiesim_h4_probe.rs` -- in-process classifier probe.
