# Vector Verify B: Edge Inputs on Silicon -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the emulator's bf16/float **edge** behavior (denormal FTZ, NaN/Inf, overflow) against real NPU1 silicon across the full rounding-mode space, using silicon-captured output as the oracle (the aietools model provably diverges from silicon on exactly these inputs).

**Architecture:** Phase 0 regrows the golden corpus (`tools/golden/vector_ops.json`) with rich edge inputs -- dense denormal mantissas spanning the FTZ boundary, more NaN payloads, more bf16 overflow matmul tiles -- via the standing out-of-repo aietools oracle, without perturbing the committed phase-A slices. Phases 1-6 then add a HW-observed golden tier (`tools/golden/silicon_edge/`): each edge kernel runs on real silicon once, its output becomes `EXP`, and the model-vs-silicon divergences are recorded as the phase-B finding. The kernel generator gains edge predicates, an always-on `out.txt` dump, and a self-bootstrapping golden source (bake `EXP` from the silicon JSON when present, else the model corpus). The bridge asserts EMU == silicon (HW leg re-confirms stability; EMU leg is the test).

**Tech Stack:** Python 3.13 (generators + capture tool, `unittest`), the de-circularized aietools Python model at `~/npu-work/experiments/vector-oracle/model` (oracle for the regrow), Chess/xchesscc, the XRT bridge (`scripts/emu-bridge-test.sh`), real Phoenix NPU1 hardware.

---

## Why the model can't be the oracle here (the motivating facts)

Established 2026-06-10 (`vector_convert.rs:280`, `vector_float.rs:243`, oracle smoke test):

- **Convert denormals:** the execute path does `fp32_flush_to_zero` (FTZ) before
  `f32_to_bf16`; the aietools model does **not** FTZ -- it *rounds* the denormal,
  which is **mode-dependent**. Smoke test, input `0x00008000`:
  `model per-mode = 0x0,0x1,0x0,0x1,...` (round-down vs round-up) while emu FTZs to
  `0x0` for **every** mode. Critically rnd=0 AND rnd=12 both give `0x0`, so a
  2-point sample misses the divergence -- the full 10-mode sweep is required.
- **Convert NaN/Inf:** model **==** emu (both via `f32_to_bf16` NaN-preservation,
  `0x7f800001`->`0x7f81`, mode-independent). Expected to confirm.
- **MAC-bf16 overflow:** the overflow result lane's canonical NaN is mantissa=`1`
  on silicon+emu (HW-verified for `vadd.f`, `vector_float.rs:241`) vs the model's
  mantissa=`0x7F`. Emu matches silicon; the model is the outlier.

So baking the model as `EXP` would make **both** HW and EMU "fail" the self-check
on these inputs while actually agreeing -- the phase-A transitive-HW==EMU trick does
not carry over. Silicon must be the oracle.

## Classes NOT in scope (verified covered or unreachable)

- **Integer SRS:** phase A drove the int16 saturation boundary (278/478 saturate-mode
  inputs overflow, expecteds `+/-32767/32768`) across the full rnd x sat sweep. Covered.
- **Pack:** 52/66 inputs beyond int8 range across all 3 sat modes in phase A. Covered.
- **UPS saturation:** **unreachable** -- widening N->>=2N bits with shift<=12 never
  overflows; the corpus has **zero** sat-clamped UPS records across all widths/shifts.
  Phase A's "saturate == none identical output" is the correct, complete behavior,
  not a gap. No UPS edge kernel.
- **MAC int8/int16:** int32 accumulator wrap is in the phase-A golden expected and was
  HW-verified in the all-9. Covered.

The bf16/float path is the entire frontier: **convert** (denormal/NaN/Inf x 10 modes)
and **mac-bf16 overflow**.

## File structure

- `tools/gen_vector_golden.py` (MODIFY) -- enrich `bf16_srs_input_patterns` (denormal
  + NaN edges) and `gen_matmul_golden` bf16 (overflow tiles), RNG-stable.
- `tools/golden/vector_ops.json` (REGEN) -- regrown corpus.
- `tools/golden/silicon_edge/` (NEW dir) -- per-kernel HW-observed golden JSON
  (provenance tier: hardware observation, CLAUDE.md source #2).
- `tools/gen_vector_kernel.py` (MODIFY) -- `out.txt` dump; `silicon_golden` field on
  `KernelSpec` + `SweepSpec`; self-bootstrapping `EXP` resolution; edge-sweep support.
- `tools/vector_kernel_specs.py` (MODIFY) -- edge predicates + the convert-edge
  `SweepSpec` and the mac-overflow `KernelSpec`.
- `tools/capture_silicon_edge.py` (NEW) -- read `out.txt`, diff model, write silicon golden.
- `tools/test_gen_vector_kernel.py`, `tools/test_capture_silicon_edge.py` (NEW/MODIFY) -- tests.
- `tools/test_gen_vector_golden.py` (NEW or MODIFY if exists) -- regrow safety tests.
- `docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md` (MODIFY) -- close B.

## Edge kernel set

| Kernel(s) | Class | Slice | Oracle |
|-----------|-------|-------|--------|
| `vec_conv_bf16_edge_r{0,1,2,3,8,9,10,11,12,13}` (10, via SweepSpec) | bf16_srs | denormal+NaN+Inf inputs | silicon |
| `vec_mac_bf16_ovf` | matmul bf16 | the overflow (Inf/NaN-result) tiles | silicon |

---

# PHASE 0: Regrow the golden corpus (aietools-gated)

**Oracle:** `VECTOR_ORACLE_MODEL=~/npu-work/experiments/vector-oracle/model` (standing,
smoke-tested 2026-06-10). The regrow MUST NOT perturb the committed phase-A slices --
Task 0.4 is a hard safety gate that regenerates the phase-A kernels and asserts a clean
`git diff`. If it isn't clean, the enrichment disturbed corpus order or the RNG stream;
fix before proceeding.

## Task 0.1: Enrich bf16 edge inputs (denormal + NaN)

**Files:**
- Modify: `tools/gen_vector_golden.py` (`bf16_srs_input_patterns` ~:426-465)
- Test: `tools/test_gen_vector_golden.py`

- [ ] **Step 1: Write the failing test**

`tools/test_gen_vector_golden.py` (new file; imports the generator without the oracle
-- `bf16_srs_input_patterns` is pure, no model dependency):

```python
import importlib.util, os, unittest
_p = os.path.join(os.path.dirname(__file__), "gen_vector_golden.py")
_spec = importlib.util.spec_from_file_location("gvg", _p)
gvg = importlib.util.module_from_spec(_spec)
# bf16_srs_input_patterns is defined before any oracle use; exec the module body
# guarded so load_oracle() at import time does not run. The module only calls
# load_oracle() inside main(), so import is safe without VECTOR_ORACLE_MODEL.
_spec.loader.exec_module(gvg)

def _cls(v):
    exp = (v >> 23) & 0xFF; frac = v & 0x7FFFFF
    if exp == 0: return "denorm" if frac else "zero"
    if exp == 0xFF: return "nan" if frac else "inf"
    return "normal"

class TestBf16EdgeEnrichment(unittest.TestCase):
    def setUp(self):
        self.pats = gvg.bf16_srs_input_patterns()

    def test_dense_denormals_span_ftz_boundary(self):
        # denormals with set bits straddling bit15 (guard) and bit16 (lsb): these
        # are where the model rounds (mode-dependent) but the execute path FTZs.
        dens = [p for p in self.pats if _cls(p) == "denorm"]
        mans = {p & 0x7FFFFF for p in dens}
        for m in (0x004000, 0x008000, 0x00C000, 0x010000, 0x018000, 0x020000):
            self.assertIn(m, mans, f"denormal mantissa {m:#x} missing")
        self.assertGreaterEqual(len(dens), 40, "want a dense denormal sweep, both signs")

    def test_more_nan_payloads_both_signs(self):
        nans = [p for p in self.pats if _cls(p) == "nan"]
        pos = [p for p in nans if not (p >> 31)]
        neg = [p for p in nans if (p >> 31)]
        self.assertGreaterEqual(len(pos), 6)
        self.assertGreaterEqual(len(neg), 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_gen_vector_golden.TestBf16EdgeEnrichment -v`
Expected: FAIL (`denormal mantissa 0xc000 missing` / not enough denormals).

- [ ] **Step 3: Implement the enrichment**

In `bf16_srs_input_patterns`, after the existing explicit NaN/Inf block (~:459) and
before the random block (~:461), insert:

```python
    # Dense denormal sweep (exp=0) straddling the guard (bit15) and lsb (bit16)
    # boundaries -- the FTZ diagnostic region. The model ROUNDS these (mode-
    # dependent: round-up modes give a nonzero bf16, round-down give 0), while the
    # execute path FTZs every denormal to signed zero. Silicon adjudicates per mode.
    denorm_mans = [0x000001, 0x002000, 0x004000, 0x006000, 0x007FFF,
                   0x008000, 0x008001, 0x00A000, 0x00C000, 0x00FFFF,
                   0x010000, 0x014000, 0x018000, 0x01C000, 0x020000,
                   0x030000, 0x040000, 0x07FFFF]
    for sgn in (0, 1):
        for man in denorm_mans:
            pats.add(fp32_bits(sgn, 0, man))   # exp=0 -> denormal

    # Richer NaN payloads (exp=255, man!=0), both signs: probe whether silicon
    # preserves the payload (like f32_to_bf16) or canonicalizes it.
    nan_mans = [0x000001, 0x000040, 0x004000, 0x008000, 0x080000,
                0x200000, 0x400000, 0x7FFFFF]
    for sgn in (0, 1):
        for man in nan_mans:
            pats.add(fp32_bits(sgn, 255, man))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m unittest test_gen_vector_golden.TestBf16EdgeEnrichment -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gen_vector_golden.py tools/test_gen_vector_golden.py
git commit -m "vector golden: enrich bf16 denormal + NaN edge inputs

Generated using Claude Code."
```

## Task 0.2: Enrich bf16 matmul overflow tiles (RNG-stable)

**Files:**
- Modify: `tools/gen_vector_golden.py` (`gen_matmul_golden` bf16 branch ~:565-598)
- Test: `tools/test_gen_vector_golden.py`

- [ ] **Step 1: Write the failing test**

The test needs the oracle (bf16_mac_hw), so guard it to skip when the model is absent:

```python
import os
@unittest.skipUnless(os.environ.get("VECTOR_ORACLE_MODEL"),
                     "needs VECTOR_ORACLE_MODEL (aietools model)")
class TestMatmulOverflowEnrichment(unittest.TestCase):
    def test_many_bf16_overflow_tiles(self):
        cases = gvg.gen_matmul_golden()
        bf = [c for c in cases if c["a_type"] == "BFloat16"]
        def has_ovf(c):
            return any(((b >> 23) & 0xFF) == 0xFF for b in c["expected"])
        ovf = [c for c in bf if has_ovf(c)]
        self.assertGreaterEqual(len(ovf), 30,
            f"want >=30 bf16 overflow tiles for the edge kernel, got {len(ovf)}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && VECTOR_ORACLE_MODEL=~/npu-work/experiments/vector-oracle/model python3 -m unittest test_gen_vector_golden.TestMatmulOverflowEnrichment -v`
Expected: FAIL (got 12, want >=30).

- [ ] **Step 3: Implement the enrichment (RNG-stable)**

In `gen_matmul_golden`, inside `if bfloat:`, AFTER the existing `mats` population
(the `nice`, 24 random-nice, 12 random blocks, ~:577) and BEFORE `for (av, bv) in mats:`,
append deliberate-overflow matrices using a **separate** RNG so the existing global
`rng` stream is untouched (preserving every phase-A finite tile byte-for-byte):

```python
            # Deliberate overflow/NaN tiles for the bf16 edge kernel. A separate
            # RNG keeps the global `rng` stream (and thus the phase-A finite tiles)
            # byte-identical. Large-exponent bf16 values (~1e38) overflow fp32 on
            # accumulate -> Inf/NaN result lanes; explicit NaN inputs propagate.
            import random as _random
            ovf_rng = _random.Random(0xB16ED6E)   # distinct, fixed seed
            big = [0x7F00, 0x7F40, 0x7F7F, 0xFF00, 0xFF40,  # +/- ~1e38
                   0x7E80, 0xFE80, 0x7F00, 0x7EC0, 0xFEC0]
            nan_in = [0x7FC0, 0xFFC0, 0x7F81, 0x7FFF]        # bf16 NaN bit patterns
            ovf_mats = []
            ovf_mats.append(([0x7F7F] * na, [0x7F7F] * nb))  # max-magnitude product
            ovf_mats.append(([0x7FC0] + [0x3F80] * (na - 1),
                             [0x3F80] * nb))                  # NaN input propagates
            for _ in range(40):
                ovf_mats.append(([ovf_rng.choice(big) for _ in range(na)],
                                 [ovf_rng.choice(big + nan_in) for _ in range(nb)]))
            mats.extend(ovf_mats)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && VECTOR_ORACLE_MODEL=~/npu-work/experiments/vector-oracle/model python3 -m unittest test_gen_vector_golden.TestMatmulOverflowEnrichment -v`
Expected: PASS (>=30 overflow tiles).

- [ ] **Step 5: Commit**

```bash
git add tools/gen_vector_golden.py tools/test_gen_vector_golden.py
git commit -m "vector golden: deliberate bf16 matmul overflow tiles (separate RNG)

Generated using Claude Code."
```

## Task 0.3: Regenerate the corpus

**Files:** Modify: `tools/golden/vector_ops.json` (regen artifact)

- [ ] **Step 1: Regenerate against the oracle**

```bash
cd tools && VECTOR_ORACLE_MODEL=~/npu-work/experiments/vector-oracle/model \
  python3 gen_vector_golden.py
```
Expected: prints per-class counts; `bf16_srs` and `matmul` counts rise, others
unchanged. The script fails loud if the oracle dir is absent.

- [ ] **Step 2: Confirm only intended classes grew**

```bash
cd tools/golden && python3 -c "
import json; d=json.load(open('vector_ops.json'))
print({k: len(v) for k,v in d.items() if k in ('srs','ups','pack','bf16_srs','matmul')})
"
```
Expected: `srs`=32400, `ups`=2840, `pack`=1890 **unchanged**; `bf16_srs` and
`matmul` larger. If srs/ups/pack changed, the enrichment leaked into the wrong
class -- STOP and investigate.

- [ ] **Step 3: Commit the regrown corpus** (after Task 0.4 passes -- do not commit a
  corpus that fails the safety gate). Hold this commit; see Task 0.4 Step 4.

## Task 0.4: SAFETY GATE -- phase-A slices unperturbed

**Files:** none (verification only)

- [ ] **Step 1: cargo test --lib (Half-A validation still bit-exact)**

Run: `cargo test --lib`
Expected: PASS. `validate_bf16_srs_golden` now asserts over MORE cases (new edges)
but the emulator is still bit-exact against the model on the existing ones; the new
denormal/NaN cases are validated against `f32_to_bf16` (the unit path, which matches
the model -- it's the *execute* path that FTZs, not this unit). If a NEW bf16_srs case
fails here, it means `f32_to_bf16` diverges from the model on an enriched input --
investigate (could be a real `f32_to_bf16` gap) before proceeding.

- [ ] **Step 2: Regenerate ALL phase-A kernels**

```bash
cd tools && python3 gen_vector_kernel.py all && python3 gen_vector_kernel.py all-sweeps
```

- [ ] **Step 3: Assert the committed phase-A test.cpp arrays did not move**

```bash
git diff --stat tests/vector-verify/
```
Expected: **EMPTY** (no changes). The enrichment added only edge inputs (filtered
out of every phase-A normal/finite slice) and used a separate RNG (phase-A finite
matmul tiles unchanged), so every baked `IN`/`EXP` array is byte-identical.

**If the diff is non-empty:** the regrow perturbed a phase-A slice. Inspect which
kernel/array moved. Likely cause: a new input landed inside a phase-A filter, or the
global RNG stream shifted. Fix the enrichment (tighten the predicate / move RNG
consumption) and re-regen until this diff is clean. Do NOT proceed past this gate.

- [ ] **Step 4: Commit the corpus once the gate is clean**

```bash
git add tools/golden/vector_ops.json
git commit -m "vector golden: regrow corpus with rich bf16 edge inputs

bf16_srs + matmul grow with dense denormal/NaN/overflow inputs for phase-B edge
verification; phase-A slices verified byte-identical (separate RNG, edge-only
inputs filtered out of normal/finite slices). cargo test --lib green.

Generated using Claude Code."
```

---

# PHASE 1: Harness + generator plumbing

## Task 1: Always-on output dump in the host harness

**Files:**
- Modify: `tools/gen_vector_kernel.py` (`_TEST_CPP_TMPL` ~:413-430, `_TEST_CPP_MATMUL_TMPL` ~:530-547)
- Test: `tools/test_gen_vector_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
import gen_vector_kernel as gen
from vector_kernel_specs import SPECS
import json, os

class TestOutputDump(unittest.TestCase):
    def setUp(self):
        gp = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
        self.golden = json.loads(open(gp).read())

    def test_elementwise_harness_dumps_out_txt(self):
        cpp = gen.render_test_cpp(SPECS["vec_srs_i32"], self.golden)
        self.assertIn('std::ofstream', cpp)
        self.assertIn('"out.txt"', cpp)
        self.assertIn('for (int i = 0; i < N; i++)', cpp)

    def test_matmul_harness_dumps_out_txt(self):
        cpp = gen.render_test_cpp(SPECS["vec_mac_i8"], self.golden)
        self.assertIn('std::ofstream', cpp)
        self.assertIn('"out.txt"', cpp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestOutputDump -v`
Expected: FAIL (`'std::ofstream' not found`).

- [ ] **Step 3: Implement the dump**

Add `#include <fstream>` to both templates' include blocks. In `_TEST_CPP_TMPL`,
immediately after `bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);` and before the error
loop, insert:

```cpp
  // Always dump the raw output buffer (one value per line) so a HW run can be
  // captured as a silicon golden. Harmless for model-golden kernels.
  {
    std::ofstream dump("out.txt");
    for (int i = 0; i < N; i++)
      dump << (int64_t)bufOut[i] << "\\n";
  }
```

Same in `_TEST_CPP_MATMUL_TMPL` but `for (int i = 0; i < NC; i++)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestOutputDump -v`
Expected: PASS.

- [ ] **Step 5: Regenerate + confirm no golden drift**

Run: `cd tools && python3 gen_vector_kernel.py all && python3 gen_vector_kernel.py all-sweeps`
Then: `git diff --stat tests/vector-verify/` -- every `test.cpp` gains the dump +
`<fstream>`; no `IN`/`EXP` array changes.

- [ ] **Step 6: Commit**

```bash
git add tools/gen_vector_kernel.py tools/test_gen_vector_kernel.py tests/vector-verify/
git commit -m "vector gen: always dump out.txt for silicon-golden capture

Generated using Claude Code."
```

## Task 2: Edge-slice predicates

**Files:**
- Modify: `tools/vector_kernel_specs.py` (near `_is_normal_f32` ~:42)
- Test: `tools/test_gen_vector_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
class TestEdgePredicates(unittest.TestCase):
    def test_is_edge_f32_selects_nonnormal(self):
        from vector_kernel_specs import _is_edge_f32
        for v in (0x00010000, 0x7F800001, 0x7F800000, 0x00000000):
            self.assertTrue(_is_edge_f32({"value": v}))
        self.assertFalse(_is_edge_f32({"value": 0x3F800000}))

    def test_is_edge_is_complement_of_normal(self):
        from vector_kernel_specs import _is_edge_f32, _is_normal_f32
        for v in [0x0, 0x1, 0x00010000, 0x3F800000, 0x7F800000, 0x7F800001, 0xFF800000]:
            self.assertNotEqual(_is_edge_f32({"value": v}), _is_normal_f32({"value": v}))

    def test_has_overflow_expected(self):
        from vector_kernel_specs import _has_overflow_expected, _all_expected_finite_f32
        ovf = {"expected": [0x3F800000, 0x7F800000]}
        fin = {"expected": [0x3F800000, 0x40000000]}
        self.assertTrue(_has_overflow_expected(ovf))
        self.assertFalse(_has_overflow_expected(fin))
        self.assertFalse(_all_expected_finite_f32(ovf))
        self.assertTrue(_all_expected_finite_f32(fin))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestEdgePredicates -v`
Expected: FAIL (`cannot import name '_is_edge_f32'`).

- [ ] **Step 3: Implement the predicates**

After `_is_normal_f32` in `tools/vector_kernel_specs.py`:

```python
def _is_edge_f32(rec):
    """The record's f32 bit pattern is NOT a normal finite number.

    Exact complement of `_is_normal_f32`: zero/denormal (exp=0) and Inf/NaN
    (exp=255) -- the inputs the phase-A conv kernel excluded. Where the execute
    path's input FTZ and NaN handling can diverge from the aietools model.
    """
    return not _is_normal_f32(rec)


def _has_overflow_expected(rec):
    """At least one lane of `expected` (fp32 bit patterns) is Inf/NaN.

    Complement of `_all_expected_finite_f32`: the bf16 matmul tiles whose result
    overflows. The overflow lane's canonical NaN is mantissa=1 on silicon+emu but
    mantissa=0x7F in the model, so these tiles need a silicon oracle.
    """
    return not _all_expected_finite_f32(rec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestEdgePredicates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/vector_kernel_specs.py tools/test_gen_vector_kernel.py
git commit -m "vector gen: edge-slice predicates (non-normal f32, overflow tiles)

Generated using Claude Code."
```

## Task 3: Self-bootstrapping silicon golden source (KernelSpec + SweepSpec)

**Files:**
- Modify: `tools/gen_vector_kernel.py` (`KernelSpec` ~:83, `SweepSpec` ~:163, `_bake_io` ~:435, `bake_matmul` ~:313, `render_test_cpp` ~:552)
- Test: `tools/test_gen_vector_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
class TestSiliconGoldenSource(unittest.TestCase):
    def setUp(self):
        gp = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
        self.golden = json.loads(open(gp).read())

    def test_silicon_field_defaults_none(self):
        spec = gen.KernelSpec(name="x", func="x", doc="",
                              inputs=[gen.Buf("in","int16_t","i16")],
                              output=gen.Buf("out","int16_t","i16"), n=4,
                              golden={"class":"srs","filt":{}}, body="")
        self.assertIsNone(spec.silicon_golden)

    def test_bootstrap_bakes_model_when_no_silicon_file(self):
        # a convert-edge sweep point with a missing silicon path -> model EXP
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]   # first mode point
        miss = gen.replace_silicon(pt, "/nonexistent.json")
        in_vals, exp_vals = gen._bake_io(miss, self.golden)
        self.assertEqual(len(exp_vals), pt.n)

    def test_silicon_file_overrides_exp_keeps_inputs(self):
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        in_model, _ = gen._bake_io(gen.replace_silicon(pt, None), self.golden)
        import tempfile, json as _json
        sj = {"kernel": pt.name, "n": pt.n, "input": in_model,
              "silicon": [0xDEAD]*pt.n, "model": [0]*pt.n, "divergences": []}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            _json.dump(sj, f); path = f.name
        in_vals, exp_vals = gen._bake_io(gen.replace_silicon(pt, path), self.golden)
        self.assertEqual(exp_vals, [0xDEAD]*pt.n)
        self.assertEqual(in_vals, in_model)

    def test_sweep_assigns_silicon_path_per_point(self):
        from vector_kernel_specs import SWEEPS
        pts = SWEEPS["vec_conv_bf16_edge_sweep"].expand()
        for pt in pts:
            self.assertIsNotNone(pt.silicon_golden)
            self.assertIn(pt.name, pt.silicon_golden)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestSiliconGoldenSource -v`
Expected: FAIL (`KernelSpec has no field 'silicon_golden'`).

- [ ] **Step 3: Implement**

(a) `KernelSpec`: add field after `matmul`:
```python
    silicon_golden: Optional[str] = None
```

(b) Helpers near `_bake_io`:
```python
def replace_silicon(spec, path):
    """Return a copy of `spec` with `silicon_golden` set (test helper)."""
    import dataclasses
    return dataclasses.replace(spec, silicon_golden=path)


def _load_silicon(spec):
    """Load the spec's silicon golden JSON if set and present, else None."""
    import os, json
    if not spec.silicon_golden or not os.path.exists(spec.silicon_golden):
        return None
    return json.loads(open(spec.silicon_golden).read())
```

(c) `_bake_io` -- override `exp_vals` from silicon when present:
```python
def _bake_io(spec, golden):
    g = spec.golden
    recs = select_records(golden[g["class"]], g["filt"], g.get("value_range"),
                          predicate=g.get("predicate"))
    in_vals = bake_array(recs, g.get("value_field", "value"), spec.n)
    exp_vals = bake_array(recs, g.get("expected_field", "expected"), spec.n)
    sj = _load_silicon(spec)
    if sj is not None:
        assert sj["input"] == in_vals, \
            f"{spec.name}: silicon golden inputs != corpus slice (regenerate capture)"
        exp_vals = sj["silicon"]
    return in_vals, exp_vals
```

(d) `bake_matmul` -- optional silicon override; `render_test_cpp` passes it:
```python
def bake_matmul(records, filt, mm, predicate=None, silicon=None):
    recs = select_records(records, filt, predicate=predicate)
    assert len(recs) >= mm.batch, f"{len(recs)} matmul records < batch {mm.batch}"
    signed = not mm.bfloat
    a_out, b_out, c_out = [], [], []
    for r in recs[:mm.batch]:
        a_out += unpack_vec512(r["a"], mm.size_a, mm.a_bytes, signed=signed)
        b_out += unpack_vec512(r["b"], mm.size_b, mm.b_bytes, signed=signed)
        c_out += r["expected"][:mm.size_c]
    if silicon is not None:
        assert silicon["input_a"] == a_out and silicon["input_b"] == b_out, \
            "silicon golden inputs != corpus slice (regenerate capture)"
        c_out = silicon["silicon"]
    return a_out, b_out, c_out
```
In `render_test_cpp` matmul branch:
```python
        a_vals, b_vals, c_vals = bake_matmul(
            golden[spec.golden["class"]], spec.golden["filt"], mm,
            predicate=spec.golden.get("predicate"), silicon=_load_silicon(spec))
```

(e) `SweepSpec`: add a `silicon: bool = False` field; in `expand()`, when building each
`KernelSpec`, set `silicon_golden=_silicon_path(self.prefix + suffix) if self.silicon
else None`. Define a module-level helper the spec file can also use:
```python
def _silicon_path(name):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "silicon_edge", name + ".json")
```
and pass `silicon_golden=...` into the `KernelSpec(...)` constructed in `expand()`.

- [ ] **Step 4: Run test to verify it passes**

Run after Task 4 registers the sweep (see note). For now, run:
`cd tools && python3 -m unittest test_gen_vector_kernel.TestSiliconGoldenSource.test_silicon_field_defaults_none -v`
Expected: PASS. Full class passes once Task 4's sweep exists.

- [ ] **Step 5: Commit**

```bash
git add tools/gen_vector_kernel.py tools/test_gen_vector_kernel.py
git commit -m "vector gen: self-bootstrapping silicon golden source (KernelSpec+SweepSpec)

Generated using Claude Code."
```

---

# PHASE 2: Edge kernel specs

## Task 4: Convert-edge sweep + mac-overflow spec

**Files:**
- Modify: `tools/vector_kernel_specs.py` (append)
- Test: `tools/test_gen_vector_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
class TestEdgeSpecs(unittest.TestCase):
    def setUp(self):
        gp = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
        self.golden = json.loads(open(gp).read())

    def test_convert_edge_sweep_is_ten_points(self):
        from vector_kernel_specs import SWEEPS
        pts = SWEEPS["vec_conv_bf16_edge_sweep"].expand()
        self.assertEqual(len(pts), 10)
        self.assertEqual({p.name for p in pts},
                         {f"vec_conv_bf16_edge_r{r}" for r in (0,1,2,3,8,9,10,11,12,13)})

    def test_convert_edge_inputs_all_nonnormal(self):
        from vector_kernel_specs import SWEEPS, _is_normal_f32
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        in_vals, _ = gen._bake_io(gen.replace_silicon(pt, None), self.golden)
        for v in in_vals:
            self.assertFalse(_is_normal_f32({"value": v}), f"{v:#x} normal")

    def test_convert_edge_slice_fits_buffer(self):
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        recs = gen.select_records(self.golden["bf16_srs"], pt.golden["filt"],
                                  predicate=pt.golden["predicate"])
        self.assertLessEqual(len(recs), pt.n)
        self.assertGreater(len(recs), 0)

    def test_mac_ovf_selects_overflow_tiles(self):
        from vector_kernel_specs import SPECS, _has_overflow_expected
        spec = SPECS["vec_mac_bf16_ovf"]
        recs = gen.select_records(self.golden["matmul"], spec.golden["filt"],
                                  predicate=spec.golden["predicate"])
        self.assertGreaterEqual(len(recs), spec.matmul.batch)
        for r in recs[:spec.matmul.batch]:
            self.assertTrue(_has_overflow_expected(r))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel.TestEdgeSpecs -v`
Expected: FAIL (`vec_conv_bf16_edge_sweep not in SWEEPS`).

NOTE the buffer size: after Task 0.1, the rnd=0 non-normal slice grows (was 64).
Set the sweep's `n` to the next multiple of 16 >= the new edge count. Compute it
first: `cd tools/golden && python3 -c "import json;b=json.load(open('vector_ops.json'))['bf16_srs'];print(sum(1 for r in b if r['rnd']==0 and (((r['value']>>23)&0xFF) in (0,0xFF))))"`
and round up to a multiple of 16. The spec below uses `EDGE_N` -- set it to that value.

- [ ] **Step 3: Implement the specs**

Append to `tools/vector_kernel_specs.py`. Use the `_silicon_path` helper from
gen_vector_kernel (import it). Set `EDGE_N` from the Step-2 computation (a multiple
of 16; e.g. 112 if the slice is ~100). The mac batch = the overflow-tile count from
Task 0.2 (>=30); set `MAC_OVF_BATCH` to a multiple that fits and that the corpus
provides -- compute and use the floor.

```python
from gen_vector_kernel import (Buf, KernelSpec, Matmul, SweepSpec, _silicon_path)

EDGE_N = 112          # set to ceil(non-normal-count/16)*16 from Step 2
MAC_OVF_BATCH = 30    # set to the actual overflow-tile count from Task 0.2

# --- Convert edge sweep: denormal + NaN + Inf f32 -> bf16 across all 10 rounding
# modes. The model ROUNDS denormals (mode-dependent) while the execute path FTZs;
# rnd=0 and rnd=12 alone miss it, so the full mode space is swept. EXP = silicon.
_reg_sweep(SweepSpec(
    prefix="vec_conv_bf16_edge",
    func="conv_bf16",
    doc="Convert edge (round-narrow), f32 -> bf16, denormal/NaN/Inf inputs.",
    inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
    output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
    n=EDGE_N,
    gclass="bf16_srs",
    base_filt={},
    rnds=_ALL_RND,
    predicate=_is_edge_f32,
    silicon=True,
    defines=[("CONV_N", EDGE_N)],
    body_template="""  event0();
$mode

  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<bfloat16, 16> o = acc.to_vector<bfloat16>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))


# --- MAC bf16 overflow: the bf16 matmul tiles whose result overflows to Inf/NaN.
# The overflow lane's canonical NaN is mantissa=1 on silicon+emu vs 0x7F in the
# model, so EXP = silicon. Same 4x8x4 mmul datapath as vec_mac_bf16.
_reg(KernelSpec(
    name="vec_mac_bf16_ovf",
    func="mac_bf16",
    doc="MatMul bf16 overflow tiles (4x8x4): result overflows to Inf/NaN. EXP is "
        "HW-captured silicon (model canonical NaN mantissa 0x7F vs silicon 1). "
        "Host stages bf16/fp32 bit patterns.",
    inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
            Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=0,
    golden={"class": "matmul",
            "filt": {"a_type": "BFloat16", "b_type": "BFloat16", "rows": 4,
                     "inner": 8, "cols": 4, "subtract": False},
            "predicate": _has_overflow_expected},
    matmul=Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2, batch=MAC_OVF_BATCH, bfloat=True),
    silicon_golden=_silicon_path("vec_mac_bf16_ovf"),
    defines=[("MAC_BATCH", MAC_OVF_BATCH)],
    body="""  event0();
  using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto>;
  for (int n = 0; n < MAC_BATCH; n++) {
    aie::vector<bfloat16, MMUL::size_A> a = aie::load_v<MMUL::size_A>(inA + n * MMUL::size_A);
    aie::vector<bfloat16, MMUL::size_B> b = aie::load_v<MMUL::size_B>(inB + n * MMUL::size_B);
    MMUL m;
    m.mul(a, b);
    aie::store_v(out + n * MMUL::size_C, m.to_vector<float>());
  }
  event1();
""",
))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m unittest test_gen_vector_kernel -v`
Expected: PASS (TestEdgeSpecs + TestSiliconGoldenSource full).

- [ ] **Step 5: Generate the edge kernels (bootstrap model EXP)**

```bash
cd tools && python3 gen_vector_kernel.py vec_conv_bf16_edge_sweep \
  && python3 gen_vector_kernel.py vec_mac_bf16_ovf
```
Expected: 10 `vec_conv_bf16_edge_r*` dirs + `vec_mac_bf16_ovf` under
`tests/vector-verify/`. Inspect one: `EXP` = model (bootstrap), dump block present.

- [ ] **Step 6: Commit**

```bash
git add tools/vector_kernel_specs.py tools/test_gen_vector_kernel.py tests/vector-verify/
git commit -m "vector gen: bf16 edge kernels (convert 10-mode sweep, mac overflow)

Generated using Claude Code."
```

---

# PHASE 3: Capture tool

## Task 5: capture_silicon_edge.py

**Files:**
- Create: `tools/capture_silicon_edge.py`
- Test: `tools/test_capture_silicon_edge.py`

- [ ] **Step 1: Write the failing test**

```python
import json, os, tempfile, unittest
import capture_silicon_edge as cap
import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS

GP = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
GOLDEN = json.loads(open(GP).read())

class TestCapture(unittest.TestCase):
    def test_parse_out_txt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1\\n-2\\n65535\\n"); p = f.name
        self.assertEqual(cap.parse_out_txt(p), [1, -2, 65535])

    def test_build_record_computes_divergences(self):
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        in_vals, model_exp = gen._bake_io(gen.replace_silicon(pt, None), GOLDEN)
        silicon = list(model_exp); silicon[0] = model_exp[0] ^ 0x1
        rec = cap.build_record(pt, GOLDEN, silicon)
        self.assertEqual(rec["input"], in_vals)
        self.assertEqual(rec["model"], model_exp)
        self.assertEqual(rec["silicon"], silicon)
        self.assertEqual(len(rec["divergences"]), 1)
        self.assertEqual(rec["divergences"][0]["i"], 0)
        self.assertIn("provenance", rec)

    def test_build_record_rejects_wrong_length(self):
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        with self.assertRaises(AssertionError):
            cap.build_record(pt, GOLDEN, [0, 1, 2])

    def test_matmul_record_has_split_inputs(self):
        spec = SPECS["vec_mac_bf16_ovf"]
        a, b, c = gen.bake_matmul(GOLDEN["matmul"], spec.golden["filt"], spec.matmul,
                                  predicate=spec.golden["predicate"])
        rec = cap.build_record(spec, GOLDEN, list(c))
        self.assertEqual(rec["input_a"], a)
        self.assertEqual(rec["input_b"], b)
        self.assertEqual(rec["silicon"], list(c))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m unittest test_capture_silicon_edge -v`
Expected: FAIL (`No module named 'capture_silicon_edge'`).

- [ ] **Step 3: Implement the tool**

`tools/capture_silicon_edge.py`:

```python
#!/usr/bin/env python3
"""Capture real-silicon output as the golden oracle for a bf16 edge kernel.

Phase-B edge inputs (denormal FTZ, NaN/Inf, overflow) are where the aietools
model diverges from NPU1 silicon, so the model cannot be the oracle. This reads
the raw output a HW run leaves in out.txt, pairs it with the corpus edge-slice
inputs + the model prediction, records every model-vs-silicon divergence (the
phase-B finding), and writes a provenance-stamped silicon golden to
tools/golden/silicon_edge/<kernel>.json. The generator then bakes EXP from it.

Flow per kernel:
  1. Generate (bootstrap model EXP) + compile via the bridge.
  2. Run test.exe on real HW once -> out.txt in the build dir.
  3. python3 capture_silicon_edge.py <kernel> --out-txt <build>/out.txt --date YYYY-MM-DD
  4. Regenerate (now EXP = silicon) and bridge-verify.
"""

import argparse
import json
import os

import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS


def parse_out_txt(path):
    """Read out.txt (one integer per line) into a list of ints."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip() != ""]


def _golden_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "vector_ops.json")


def resolve_spec(name):
    """Find a kernel spec by name across SPECS and expanded SWEEPS."""
    if name in SPECS:
        return SPECS[name]
    for sw in SWEEPS.values():
        for pt in sw.expand():
            if pt.name == name:
                return pt
    raise KeyError(f"unknown kernel '{name}'")


def build_record(spec, golden, silicon, date="UNDATED"):
    """Assemble the silicon golden: inputs, model, silicon, divergences.

    `silicon` is the captured HW output (length must equal the output element
    count). Divergences are indices where silicon != model -- the phase-B record.
    """
    prov = (f"HW-observed: NPU1 Phoenix (real silicon), captured {date} via "
            "tools/capture_silicon_edge.py. Oracle tier: hardware observation "
            "(CLAUDE.md source #2). EXP for this edge kernel = `silicon`; `model` "
            "is the aietools prediction, kept for the divergence record only.")
    if spec.matmul is not None:
        a, b, c = gen.bake_matmul(golden[spec.golden["class"]], spec.golden["filt"],
                                  spec.matmul, predicate=spec.golden.get("predicate"))
        model = list(c); n = len(model)
        assert len(silicon) == n, f"silicon len {len(silicon)} != output count {n}"
        div = [{"i": i, "model": model[i], "silicon": silicon[i]}
               for i in range(n) if model[i] != silicon[i]]
        return {"kernel": spec.name, "class": spec.golden["class"], "n": n,
                "input_a": a, "input_b": b, "model": model,
                "silicon": list(silicon), "divergences": div, "provenance": prov}
    in_vals, model = gen._bake_io(gen.replace_silicon(spec, None), golden)
    n = spec.n
    assert len(silicon) == n, f"silicon len {len(silicon)} != n {n}"
    div = [{"i": i, "input": in_vals[i], "model": model[i], "silicon": silicon[i]}
           for i in range(n) if model[i] != silicon[i]]
    return {"kernel": spec.name, "class": spec.golden["class"], "n": n,
            "input": in_vals, "model": model, "silicon": list(silicon),
            "divergences": div, "provenance": prov}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture silicon output as an edge golden.")
    ap.add_argument("kernel")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--date", default="UNDATED")
    args = ap.parse_args(argv)

    spec = resolve_spec(args.kernel)
    golden = json.loads(open(_golden_path()).read())
    silicon = parse_out_txt(args.out_txt)
    rec = build_record(spec, golden, silicon, date=args.date)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "golden", "silicon_edge")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, args.kernel + ".json")
    with open(dest, "w") as f:
        json.dump(rec, f, indent=1); f.write("\\n")
    print(f"wrote {dest}: {len(rec['divergences'])} model-vs-silicon divergences")
    for d in rec["divergences"][:20]:
        line = f"  [{d['i']}] model={d['model']:#x} silicon={d['silicon']:#x}"
        if "input" in d:
            line += f" input={d['input']:#x}"
        print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m unittest test_capture_silicon_edge -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_silicon_edge.py tools/test_capture_silicon_edge.py
git commit -m "vector verify: silicon-edge capture tool (out.txt -> golden + divergences)

Generated using Claude Code."
```

---

# PHASE 4: HW capture (real silicon -- the oracle)

## Task 6: Capture silicon output for all 11 edge kernels

**Hardware step.** No unit test; Task 7's bridge run is the verification.

- [ ] **Step 1: Stage + compile the edge kernels via the bridge (Chess)**

Stage the generated kernels into the mlir-aie clone (the same re-sync phase A used:
copy `tests/vector-verify/<name>` -> `../mlir-aie/test/npu-xrt/<name>`). Then:

```bash
mkdir -p build/experiments/vector-verify-B
./scripts/emu-bridge-test.sh --chess-only --compile \
  -v 'vec_conv_bf16_edge_r|vec_mac_bf16_ovf'
```
Expected: 11 xclbins + test.exe under `../mlir-aie/build/test/npu-xrt/<name>/chess/`.

- [ ] **Step 2: Run each on real HW (capture out.txt), twice, confirm stable**

```bash
for k in vec_conv_bf16_edge_r0 vec_conv_bf16_edge_r1 vec_conv_bf16_edge_r2 \
         vec_conv_bf16_edge_r3 vec_conv_bf16_edge_r8 vec_conv_bf16_edge_r9 \
         vec_conv_bf16_edge_r10 vec_conv_bf16_edge_r11 vec_conv_bf16_edge_r12 \
         vec_conv_bf16_edge_r13 vec_mac_bf16_ovf; do
  d="../mlir-aie/build/test/npu-xrt/$k/chess"
  ( cd "$d" && env -u XDNA_EMU ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin \
      && cp out.txt out.run1.txt \
      && env -u XDNA_EMU ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin )
  diff "$d/out.run1.txt" "$d/out.txt" && echo "$k STABLE" || echo "$k UNSTABLE -- STOP"
done
```
Expected: every kernel STABLE (deterministic datapath). The self-check may print
"failed (N errors)" -- EXPECTED (bootstrap EXP is the model, which diverges); we read
out.txt. An UNSTABLE kernel means HW flakiness -- investigate before committing a golden.

- [ ] **Step 3: Bake the silicon goldens + read the divergence reports**

```bash
cd tools
for k in vec_conv_bf16_edge_r0 vec_conv_bf16_edge_r1 vec_conv_bf16_edge_r2 \
         vec_conv_bf16_edge_r3 vec_conv_bf16_edge_r8 vec_conv_bf16_edge_r9 \
         vec_conv_bf16_edge_r10 vec_conv_bf16_edge_r11 vec_conv_bf16_edge_r12 \
         vec_conv_bf16_edge_r13 vec_mac_bf16_ovf; do
  python3 capture_silicon_edge.py $k \
    --out-txt ../../mlir-aie/build/test/npu-xrt/$k/chess/out.txt --date 2026-06-10
done
```
**Sanity-check divergences against predictions:**
- conv denormals: divergences on round-up modes (model rounds non-zero, silicon `0`
  IF it FTZs). The divergence SET should differ across modes (round-up vs round-down)
  -- that mode-dependence is the finding.
- conv NaN/Inf: 0 divergences expected (model == emu == silicon).
- mac overflow: divergences on Inf/NaN result lanes (model mantissa region vs silicon 1).

If silicon contradicts a prediction (e.g. does NOT FTZ a denormal, or a NaN lane
differs unexpectedly), STOP and investigate via systematic-debugging -- a real new
finding, possibly an emulator bug. This is a design checkpoint; surface it.

- [ ] **Step 4: Regenerate the edge kernels (now EXP = silicon)**

```bash
cd tools && python3 gen_vector_kernel.py vec_conv_bf16_edge_sweep \
  && python3 gen_vector_kernel.py vec_mac_bf16_ovf
git diff tests/vector-verify/vec_conv_bf16_edge_r9/test.cpp   # a round-up mode: EXP moved
```

- [ ] **Step 5: Commit the silicon goldens + regenerated kernels**

```bash
git add tools/golden/silicon_edge/ tests/vector-verify/
git commit -m "vector verify B: capture bf16 edge silicon goldens (HW oracle)

HW-observed NPU1 Phoenix output for the bf16 edge kernels (convert
denormal/NaN/Inf x 10 modes, mac overflow). Divergence records vs the aietools
model: <summarize: denormal FTZ vs mode-dependent round on convert; canonical-NaN
mantissa on mac>.

Generated using Claude Code."
```

---

# PHASE 5: Bridge verification + emulator fixes

## Task 7: HW==EMU on edges

**Hardware step.** Rebuild the `.so` first.

- [ ] **Step 1: Rebuild the emulator .so**

Run: `cargo build && cargo build -p xdna-emu-ffi`
Expected: clean.

- [ ] **Step 2: Full bridge run on the edge kernels**

```bash
./scripts/emu-bridge-test.sh --chess-only \
  -v 'vec_conv_bf16_edge_r|vec_mac_bf16_ovf' \
  2>&1 | tee build/experiments/vector-verify-B/bridge-edge.log
```
Expected: HW leg PASS all (EXP == silicon == HW, stability re-confirm); EMU leg is
the real test.

- [ ] **Step 3: Triage EMU results**

- **All EMU PASS** -> the emulator already matches silicon on these edges (the
  predicted outcome: FTZ + canonical-NaN already implemented). Phase B confirms the
  emulator and records the model's divergences. Proceed to Task 8.
- **An EMU leg FAILS** -> systematic-debugging: map the failing index via the silicon
  golden's `divergences`/`input`, trace the execute path (`vector_convert.rs` FTZ for
  convert, `vector_float.rs` / `vector_matmul/bf16_pipeline.rs` for mac). Fix the
  emulator to match silicon (derive from toolchain/HW observation, never hardcode a
  single value). Rebuild `.so`, re-run Step 2. Repeat until green.

- [ ] **Step 4: cargo test --lib (no regression)**

Run: `cargo test --lib`
Expected: green; `validate_bf16_srs_golden` + mac/convert unit tests still pass.

- [ ] **Step 5: Commit any emulator fix** (only if Step 3 required one)

```bash
cargo build -p xdna-emu-ffi
git add -A
git commit -m "<class> edge: <what diverged> -- match silicon (HW-validated)

Generated using Claude Code."
```

---

# PHASE 6: Document + close

## Task 8: Close phase B

**Files:**
- Modify: `docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md` (section B)
- Modify: `docs/known-fidelity-gaps.md` (only if a divergence is a genuine deferred gap)
- Modify: memory `project_vector_verification_depth_inflight.md` (resume -> phase C)

- [ ] **Step 1: Mark section B done in the AtoD plan**

Replace `## B. Edge inputs on silicon` with a result block (mirror section A's
`[DONE ...]` style): the 11 kernels, the silicon-golden oracle tier + corpus regrow,
the confirmed model-vs-silicon divergences (denormal FTZ-vs-mode-dependent-round on
convert; canonical-NaN mantissa on mac; NaN/Inf-convert confirmed identical;
rounding-mode dependence of the denormal divergence), and the HW==EMU verdict (+ any
emulator fix). Note UPS proven complete (saturation unreachable) and integer classes
covered by phase A.

- [ ] **Step 2: known-fidelity-gaps -- only if warranted**

If every edge is EMU==silicon, add NOTHING (that doc is for confirmed HW-disagreement
gaps). If an edge revealed a deferred emulator gap, add a row.

- [ ] **Step 3: Update the resume memory anchor**

Point the memory file's resume at phase C (op breadth); summarize phase B
(silicon-golden tier added; corpus regrown; bf16 edges HW==EMU; divergences recorded).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md docs/known-fidelity-gaps.md
git commit -m "plan: phase B done -- bf16 edge inputs HW==EMU via silicon-golden oracle

Generated using Claude Code."
```

---

## Self-review notes

- **Provenance honesty:** the silicon golden is a distinct oracle tier (HW
  observation) from the model corpus -- the `provenance` field + the retained `model`
  column make that explicit. Not copying aietools data; recording our silicon.
- **No circularity:** EXP for edges comes from HW; the emulator's edge behavior was
  derived independently (FTZ from the model's `fp32_denorm_to_0` shape; canonical NaN
  from prior `vadd.f` HW observation). Capture CONFIRMS an independent prediction.
- **Regrow safety is a hard gate:** Task 0.4 blocks until the phase-A `git diff` is
  empty -- edge-only inputs (filtered out of normal/finite slices) + a separate RNG
  for new matmul tiles guarantee it, and the gate proves it empirically.
- **Coverage is now airtight per class:** Convert mode x edge (full 10-mode sweep,
  catching the per-mode denormal divergence a 2-point sample misses); MAC-bf16 overflow
  (enriched tile count); SRS/Pack covered by phase A; UPS proven complete (saturation
  unreachable). This is the evidence the gate flip (phase 113) needs.
- **Input diversity is corpus-bound, now grown:** the regrow is the one lever that
  needed the aietools oracle; kernel count is free via SweepSpec.
