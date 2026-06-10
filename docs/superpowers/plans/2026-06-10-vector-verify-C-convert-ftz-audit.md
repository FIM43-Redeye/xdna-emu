# Phase C: Convert FTZ-Path Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine on NPU1 Phoenix silicon whether the standalone `VCONV` (bf16->f32 expand) and `VFLOOR` (bf16->int32 floor) datapaths flush denormals to zero, and either fix the emulator's `vector_convert` FTZ to match silicon or confirm-and-document it.

**Architecture:** Reuse phase B's silicon-golden tier. Add a localized direct-input mode to the kernel generator (bf16 denormal inputs no model corpus produces; silicon is the oracle). Author two capture kernels driving the exact standalone intrinsics, verify the emitted opcode by disassembly, capture silicon, compare against the emulator, and fix `fp32_flush_to_zero` where silicon proves no-flush. Disposition the f32->int FTZ paths analytically.

**Tech Stack:** Python generator (`tools/gen_vector_kernel.py`, `tools/vector_kernel_specs.py`), Chess/aie_api kernels, the bridge harness (`scripts/emu-bridge-test.sh`), `tools/capture_silicon_edge.py`, Rust emulator (`src/interpreter/execute/vector_convert.rs`).

---

## Background facts (established before this plan)

- The FTZ paths all route through `VectorAlu::vector_convert`
  (`src/interpreter/execute/vector_convert.rs`), blanket-added in `ef77756`:
  - bf16->f32 expand: `:273` `fp32_flush_to_zero(bf16_to_f32(bf16))`
  - f32->bf16 contract: `:281`
  - f32->int32 / ->uint32: `:307`, `:315`
  - bf16->int32 vfloor: `:343`
- Phase B's `VST.CONV` was the fused store-**from-accumulator** path
  (`memory/mod.rs:1110`) which calls `f32_to_bf16` directly and never flushes --
  so B never tested any of the above.
- Intrinsics (from llvm-aie TableGen / aie_api), confirmed by Explore:
  - bf16->f32 expand: `int_aie2_v16bf16_to_v16accfloat` -> `VCONV_FP32_BF16`.
    aie_api form: `aie::accum<accfloat,16> acc(bf16_vec)`.
  - bf16->int32 floor: `int_aie2_v16bf16_to_v16i32` -> `VFLOOR_S32_BF16`.
    aie_api form: `aie::to_fixed<int32>(bf16_vec, 0)`.
- Generator bakes **inputs** from a model corpus class; no class produces bf16
  denormal inputs. We add a direct-input mode rather than a model class.
- The no-FTZ mathematical reference is exact and non-circular:
  - expand: `bf16_to_f32(bits) == (bits << 16)` (bf16 is the high 16 bits of f32).
    A bf16 denormal `0x0001` widens to f32 `0x0001_0000`, itself an f32 denormal.
  - vfloor: `floor(bf16_value)`; a tiny negative denormal floors to `-1`, a tiny
    positive to `0`. FTZ would give `0`/`0`.

## File structure

- **Create** `tests/` unit coverage inside existing Python test file
  `tools/test_gen_vector_kernel.py` (direct-input mode tests). Check the actual
  test filename first (Task 1 Step 0).
- **Modify** `tools/gen_vector_kernel.py` -- add `DirectIO` dataclass + direct
  branch in `_bake_io`. ~20 lines, one responsibility (input sourcing).
- **Modify** `tools/vector_kernel_specs.py` -- add a bf16-denormal input helper
  and two `KernelSpec`s (`vec_convexp_bf16_denorm`, `vec_vfloor_bf16_denorm`).
- **Create** `tools/golden/silicon_edge/vec_convexp_bf16_denorm.json`,
  `tools/golden/silicon_edge/vec_vfloor_bf16_denorm.json` -- HW captures.
- **Modify** `src/interpreter/execute/vector_convert.rs` -- remove/guard FTZ in
  the silicon-disproven branches + per-branch regression tests; analytic comment
  + test for the f32->int no-op.
- **Modify** docs: `docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md`
  (section C -> DONE), `docs/known-fidelity-gaps.md`, the memory anchor.
- `tools/capture_silicon_edge.py` -- **no change** (its `_bake_io` call inherits
  the direct branch; "model" column becomes the no-FTZ reference automatically).

---

## Task 1: Generator direct-input mode

**Files:**
- Modify: `tools/gen_vector_kernel.py`
- Test: `tools/test_gen_vector_kernel.py` (confirm exact name in Step 0)

- [ ] **Step 0: Locate the generator test file**

Run: `ls tools/test_*vector* tools/*test*.py 2>/dev/null; grep -rl "gen_vector_kernel\|_bake_io\|render_test_cpp" tools/ --include=*test*.py`
Expected: a test module (e.g. `tools/test_gen_vector_kernel.py`). Use that path
below. If none exists, create `tools/test_gen_vector_kernel.py` with
`import sys, os; sys.path.insert(0, os.path.dirname(__file__))` then
`import gen_vector_kernel as gen` at the top.

- [ ] **Step 1: Write the failing test**

Add to the generator test module:

```python
def test_direct_io_bakes_inputs_and_reference_without_corpus():
    spec = gen.KernelSpec(
        name="vec_direct_probe", func="probe_bf16",
        doc="direct-input probe.",
        inputs=[gen.Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
        output=gen.Buf("out", "uint32_t", "f32", ktype="float"),
        n=4,
        golden={"class": "direct",
                "direct": gen.DirectIO(inputs=(1, 2, 0x8001, 0),
                                       reference=(0x10000, 0x20000, 0x80010000, 0))},
        body="  // body\n",
    )
    # An empty corpus must not be consulted for a direct spec.
    in_vals, exp_vals = gen._bake_io(spec, {})
    assert in_vals == [1, 2, 0x8001, 0]
    assert exp_vals == [0x10000, 0x20000, 0x80010000, 0]


def test_direct_io_silicon_override_with_input_alignment(tmp_path):
    import json
    sj = tmp_path / "vec_direct_probe.json"
    sj.write_text(json.dumps({"input": [1, 2, 0x8001, 0],
                              "silicon": [0, 0, 0x80000000, 0]}))
    spec = gen.replace_silicon(gen.KernelSpec(
        name="vec_direct_probe", func="probe_bf16", doc="x",
        inputs=[gen.Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
        output=gen.Buf("out", "uint32_t", "f32", ktype="float"),
        n=4,
        golden={"class": "direct",
                "direct": gen.DirectIO(inputs=(1, 2, 0x8001, 0),
                                       reference=(0x10000, 0x20000, 0x80010000, 0))},
        body="  // body\n"), str(sj))
    in_vals, exp_vals = gen._bake_io(spec, {})
    assert in_vals == [1, 2, 0x8001, 0]
    assert exp_vals == [0, 0, 0x80000000, 0]   # silicon wins
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python3 -m pytest test_gen_vector_kernel.py -k direct_io -v`
Expected: FAIL -- `AttributeError: module 'gen_vector_kernel' has no attribute 'DirectIO'`.

- [ ] **Step 3: Implement `DirectIO` + the `_bake_io` direct branch**

In `tools/gen_vector_kernel.py`, after the `Matmul` dataclass, add:

```python
@dataclass(frozen=True)
class DirectIO:
    """Inputs supplied directly by the spec, bypassing the model corpus.

    For kernels whose oracle is silicon (not the aietools model) and whose input
    space (e.g. bf16 denormals) no corpus class produces. `inputs` are the host-
    staged bit patterns; `reference` is a simple mathematical expectation (no-FTZ
    widen/floor) used ONLY as a bootstrap EXP and a divergence baseline -- never
    an oracle. The binding comparison is emulator-output vs captured silicon.
    """

    inputs: tuple
    reference: tuple
```

Replace the body of `_bake_io` (currently starting `g = spec.golden`) so the
direct branch precedes the corpus path:

```python
def _bake_io(spec, golden):
    """Select the spec's golden slice and bake (input, expected) arrays to N.

    A spec whose `golden` carries a `direct` DirectIO sources inputs and the
    bootstrap reference straight from it (no corpus class). Otherwise the model
    corpus drives both arrays. In both cases, a present silicon capture overrides
    the expected array, with an input-alignment cross-check.
    """
    g = spec.golden
    direct = g.get("direct")
    if direct is not None:
        in_vals = list(direct.inputs)
        exp_vals = list(direct.reference)
    else:
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

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python3 -m pytest test_gen_vector_kernel.py -k direct_io -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full generator test module (no regressions)**

Run: `cd tools && python3 -m pytest test_gen_vector_kernel.py -v`
Expected: all PASS (the existing corpus-path tests still pass; the direct branch
is additive).

- [ ] **Step 6: Commit**

```bash
git add tools/gen_vector_kernel.py tools/test_gen_vector_kernel.py
git commit -m "vector gen: direct-input mode for silicon-oracle kernels

DirectIO lets a KernelSpec source bf16 denormal inputs + a no-FTZ reference
straight from the spec, bypassing the model corpus (no class produces bf16
denormal inputs, and silicon is the oracle for the convert FTZ audit). The
silicon override + input-alignment assert are unchanged. capture_silicon_edge
inherits this through its _bake_io call -- the model column becomes the no-FTZ
reference automatically.

Generated using Claude Code.

Co-Authored-By: <leave as configured>"
```
(Drop the Co-Authored-By line if not the repo convention; keep the trailer
`Generated using Claude Code.`)

---

## Task 2: Spike -- confirm the kernel bodies emit standalone VCONV / VFLOOR

This is a compiler-in-the-loop spike (the A/B compile lottery). Output: two
confirmed `body` strings whose `.lst` shows `VCONV_FP32_BF16` and
`VFLOOR_S32_BF16`. No final code is committed here; the confirmed bodies feed
Task 3.

**Files:** scratch under `build/experiments/vector-verify-C/` (regenerable).

- [ ] **Step 1: Register two throwaway probe specs**

Temporarily add to `tools/vector_kernel_specs.py` (these become the real specs in
Task 3 -- here they exist only to compile):

```python
_DENORM_N = 256  # all 254 bf16 denormals + 0x0000 + 0x8000


def _bf16_denorm_inputs():
    """All 254 bf16 denormals (exp=0, mantissa 1..127, both signs) + +/-0.

    Exhaustive denormal enumeration: the FTZ discriminator lives entirely here.
    Positive then negative then the two zeros, padded to _DENORM_N (already 256).
    """
    pos = [m for m in range(1, 128)]              # 0x0001..0x007F
    neg = [0x8000 | m for m in range(1, 128)]     # 0x8001..0x807F
    vals = pos + neg + [0x0000, 0x8000]
    assert len(vals) == _DENORM_N
    return tuple(vals)


_reg(KernelSpec(
    name="vec_convexp_bf16_denorm", func="convexp_bf16", stem="convexp",
    doc="probe: bf16->f32 expand (VCONV), denormal inputs.",
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "uint32_t", "f32", ktype="float"),
    n=_DENORM_N,
    golden={"class": "direct",
            "direct": __import__("gen_vector_kernel").DirectIO(
                inputs=_bf16_denorm_inputs(),
                reference=tuple(v << 16 for v in _bf16_denorm_inputs()))},
    defines=[("CONV_N", _DENORM_N)],
    body="""  event0();
  for (int i = 0; i < CONV_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    aie::accum<accfloat, 16> acc(v);
    aie::vector<float, 16> o = acc.to_vector<float>();
    aie::store_v(out + i, o);
  }
  event1();
""",
))

_reg(KernelSpec(
    name="vec_vfloor_bf16_denorm", func="vfloor_bf16", stem="vfloor",
    doc="probe: bf16->int32 floor (VFLOOR), denormal inputs.",
    inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
    output=Buf("out", "int32_t", "i32"),
    n=_DENORM_N,
    golden={"class": "direct",
            "direct": __import__("gen_vector_kernel").DirectIO(
                inputs=_bf16_denorm_inputs(),
                reference=_bf16_floor_reference())},
    defines=[("VFL_N", _DENORM_N)],
    body="""  event0();
  ::aie::set_rounding(aie::rounding_mode::floor);
  for (int i = 0; i < VFL_N; i += 16) {
    aie::vector<bfloat16, 16> v = aie::load_v<16>(in + i);
    aie::vector<int32, 16> o = aie::to_fixed<int32>(v, 0);
    aie::store_v(out + i, o);
  }
  event1();
""",
))
```

Add the floor-reference helper near `_bf16_denorm_inputs`:

```python
def _bf16_to_f32_bits(b):
    """Exact bf16->f32 widen: bf16 is the high 16 bits of the f32 word."""
    return (b & 0xFFFF) << 16


def _bf16_floor_reference():
    """No-FTZ floor(bf16) per lane as int32 two's-complement bit patterns."""
    import math
    import struct
    out = []
    for b in _bf16_denorm_inputs():
        f = struct.unpack("<f", struct.pack("<I", _bf16_to_f32_bits(b)))[0]
        out.append(int(math.floor(f)) & 0xFFFFFFFF)
    return tuple(out)
```

(Clean up the throwaway `_reg(...)` calls at the end of the task if Task 3
restructures them; the helpers stay.)

- [ ] **Step 2: Generate + Chess-compile both probes via the bridge**

Stage and compile (Chess only, no HW, no run -- we only want the `.o`/`.lst`):

```bash
cd /home/triple/npu-work/xdna-emu
python3 tools/gen_vector_kernel.py vec_convexp_bf16_denorm
python3 tools/gen_vector_kernel.py vec_vfloor_bf16_denorm
cp -r tests/vector-verify/vec_convexp_bf16_denorm ../mlir-aie/test/npu-xrt/
cp -r tests/vector-verify/vec_vfloor_bf16_denorm  ../mlir-aie/test/npu-xrt/
./scripts/emu-bridge-test.sh --chess-only --no-hw --compile \
  vec_convexp_bf16_denorm > build/experiments/vector-verify-C/compile-convexp.log 2>&1
./scripts/emu-bridge-test.sh --chess-only --no-hw --compile \
  vec_vfloor_bf16_denorm  > build/experiments/vector-verify-C/compile-vfloor.log 2>&1
```

(Confirm the bridge filter/flag syntax against phase B: positional filter, `-v`
is verbose not filter. `--no-trace` if available. Read each log -- do NOT pipe
through tail/grep.)

Expected: both compile (Chess emits a `.o`). If a body fails to compile, iterate
the aie_api call (alternatives below) until it does.

- [ ] **Step 3: Disassemble and confirm the emitted opcode**

Find each build's `.o` and disassemble:

```bash
find ../mlir-aie/build/test/npu-xrt/vec_convexp_bf16_denorm/chess -name '*.o' -o -name '*.lst' | head
# Use the chess-produced .lst if present, else:
llvm-objdump -d <path>/convexp.o > build/experiments/vector-verify-C/convexp.lst 2>&1
llvm-objdump -d <path>/vfloor.o  > build/experiments/vector-verify-C/vfloor.lst  2>&1
```

Read the `.lst` and confirm:
- convexp body emits **VCONV** (bf16->fp32 / accfloat expand), NOT a MAC and NOT
  a fused store-convert.
- vfloor body emits **VFLOOR** (s32 <- bf16 floor), NOT a round-to-nearest
  convert.

Expected: the target opcode appears in the loop body. Record the exact mnemonic.

- [ ] **Step 4: Decision point -- opcode confirmed or fall back**

- Both opcodes confirmed -> proceed to Task 3 with these exact bodies.
- convexp emits a MAC/fused form instead -> try, in order: (a)
  `aie::vector<float,16> o = aie::to_float(v);`, (b) the builtin
  `__builtin_aiev2_v16bf16_to_v16accfloat(v)` wrapped in a v16accfloat, (c)
  inspect `mlir-aie/aie_runtime_lib/AIE2/lut_based_ops.h` for the expand idiom.
- vfloor emits a non-floor convert -> try `::bfloat16_to_int(v, 0)` (the lower-
  level intrinsic from `elementary.hpp`), and confirm `set_rounding(floor)`
  actually selects VFLOOR vs a rounding variant.
- If, after reasonable iteration, Chess refuses to emit a *standalone* VCONV/
  VFLOOR for a path, STOP and report to Maya: that path may be compiler-
  unreachable, which changes its disposition (document-by-inference rather than
  silicon-test). Do not fabricate a passing kernel.

- [ ] **Step 5: Note findings (no commit)**

Write the confirmed bodies + opcode evidence into
`build/experiments/vector-verify-C/SPIKE-FINDINGS.md` (regenerable scratch). Task
3 commits the real specs.

---

## Task 3: Register the two audit KernelSpecs

**Files:**
- Modify: `tools/vector_kernel_specs.py`
- Test: `tools/test_gen_vector_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_audit_specs_generate_with_direct_denorm_inputs():
    import vector_kernel_specs as vks
    for name in ("vec_convexp_bf16_denorm", "vec_vfloor_bf16_denorm"):
        spec = vks.SPECS[name]
        # Direct inputs: 254 denormals + 2 zeros, exhaustive, length 256.
        d = spec.golden["direct"]
        assert len(d.inputs) == 256 and len(d.reference) == 256
        # A negative bf16 denormal must be present (the FTZ discriminator).
        assert 0x8001 in d.inputs
        # Bootstrap-mode bake (no silicon json yet) returns the direct arrays.
        in_vals, exp_vals = gen._bake_io(gen.replace_silicon(spec, None), {})
        assert in_vals == list(d.inputs)
        assert exp_vals == list(d.reference)


def test_vfloor_reference_floors_negative_denormal_to_minus_one():
    import vector_kernel_specs as vks
    spec = vks.SPECS["vec_vfloor_bf16_denorm"]
    d = spec.golden["direct"]
    idx = d.inputs.index(0x8001)          # tiny negative denormal
    assert d.reference[idx] == 0xFFFFFFFF  # floor(-tiny) = -1 (no FTZ)
    idx0 = d.inputs.index(0x0001)          # tiny positive denormal
    assert d.reference[idx0] == 0          # floor(+tiny) = 0


def test_convexp_reference_widens_denormal_without_flush():
    import vector_kernel_specs as vks
    spec = vks.SPECS["vec_convexp_bf16_denorm"]
    d = spec.golden["direct"]
    idx = d.inputs.index(0x0001)
    assert d.reference[idx] == 0x0001_0000  # widened f32 denormal, not flushed
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python3 -m pytest test_gen_vector_kernel.py -k "audit_specs or vfloor_reference or convexp_reference" -v`
Expected: FAIL (specs not yet finalized / helpers absent).

- [ ] **Step 3: Finalize the two specs in `vector_kernel_specs.py`**

Promote the Task 2 probe specs to their committed form: keep `_DENORM_N`,
`_bf16_denorm_inputs`, `_bf16_to_f32_bits`, `_bf16_floor_reference`, and the two
`_reg(KernelSpec(...))` blocks with the **opcode-confirmed bodies from Task 2**.
Replace the `__import__("gen_vector_kernel").DirectIO` inline hack with a top-of-
file import: add `DirectIO` to the existing
`from gen_vector_kernel import Buf, KernelSpec, Matmul, SweepSpec, _silicon_path`
line. Set `silicon_golden=_silicon_path("<name>")` on each spec so the captured
JSON is consumed once present. Replace each spec's `doc` with a real description
(drop "probe:"). Add a header comment block explaining the FTZ audit (mirror the
phase-B edge-spec header).

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python3 -m pytest test_gen_vector_kernel.py -v`
Expected: all PASS.

- [ ] **Step 5: Generate the kernel files**

```bash
cd /home/triple/npu-work/xdna-emu
python3 tools/gen_vector_kernel.py vec_convexp_bf16_denorm
python3 tools/gen_vector_kernel.py vec_vfloor_bf16_denorm
```

Expected: `generated vec_convexp_bf16_denorm -> tests/vector-verify/...` (and
vfloor). Confirm `tests/vector-verify/vec_*_bf16_denorm/{run.lit,aie.mlir,test.cpp,<stem>.cc}`
exist; spot-read the `.cc` to confirm the confirmed body baked in, and `test.cpp`
to confirm `IN`/`EXP` (256 each).

- [ ] **Step 6: Commit**

```bash
git add tools/vector_kernel_specs.py tools/test_gen_vector_kernel.py
git commit -m "vector gen: bf16 convert FTZ-audit specs (VCONV expand, VFLOOR)

Two silicon-oracle capture kernels with exhaustive bf16-denormal inputs (all
254 denormals + +/-0), driving the standalone VCONV_FP32_BF16 expand and
VFLOOR_S32_BF16 paths that route through vector_convert's fp32_flush_to_zero --
the paths phase B never exercised (B only hit the fused store-from-accumulator
path, which never flushed). EXP comes from silicon; the no-FTZ widen/floor
reference is the bootstrap + divergence baseline. Opcodes .lst-confirmed.

Generated using Claude Code."
```

(The generated `tests/vector-verify/vec_*_bf16_denorm/` dirs stay untracked, per
the phase-B convention -- they are regenerable artifacts.)

---

## Task 4: Capture silicon (HW, deterministic)

**Files:** Create `tools/golden/silicon_edge/vec_*_bf16_denorm.json`.

> HARDWARE TASK. Costs a bridge run. Do NOT run concurrently with any other HW
> suite. Rebuild the FFI `.so` is NOT needed here (capture is HW-only, emulator
> not involved). If the NPU wedges: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`.

- [ ] **Step 1: Bootstrap-compile + run both kernels on HW**

The specs reference `silicon_golden` paths that do not exist yet -> `_bake_io`
bakes the no-FTZ reference as bootstrap EXP. Stage and run on real NPU1:

```bash
cd /home/triple/npu-work/xdna-emu
cp -r tests/vector-verify/vec_convexp_bf16_denorm ../mlir-aie/test/npu-xrt/
cp -r tests/vector-verify/vec_vfloor_bf16_denorm  ../mlir-aie/test/npu-xrt/
./scripts/emu-bridge-test.sh --chess-only \
  vec_convexp_bf16_denorm vec_vfloor_bf16_denorm \
  > build/experiments/vector-verify-C/hw-capture-pass1.log 2>&1
```

Expected: kernels run on HW. They may REPORT failures (bootstrap EXP = no-FTZ
reference; if silicon FTZs, the negative-denormal lanes mismatch) -- that is
fine, `out.txt` is dumped regardless. Read the log; confirm each build dir has an
`out.txt`. Find them:

```bash
find ../mlir-aie/build/test/npu-xrt/vec_convexp_bf16_denorm/chess -name out.txt
find ../mlir-aie/build/test/npu-xrt/vec_vfloor_bf16_denorm/chess  -name out.txt
```

- [ ] **Step 2: Capture each silicon golden**

```bash
cd /home/triple/npu-work/xdna-emu/tools
python3 capture_silicon_edge.py vec_convexp_bf16_denorm \
  --out-txt <convexp-build>/out.txt --date 2026-06-10
python3 capture_silicon_edge.py vec_vfloor_bf16_denorm \
  --out-txt <vfloor-build>/out.txt  --date 2026-06-10
```

Expected: `wrote .../vec_convexp_bf16_denorm.json: <K> model-vs-silicon divergences`
(K = number of lanes where the no-FTZ reference differs from silicon). The
printed divergences ARE the finding: zero -> silicon does not FTZ (reference
correct); nonzero at negative-denormal lanes -> silicon FTZs there. Record the
verdict per kernel.

- [ ] **Step 3: Determinism pass 2**

Re-run Step 1's bridge command into `hw-capture-pass2.log`, re-capture to temp
files, and diff the `silicon` arrays against the committed JSONs:

```bash
python3 capture_silicon_edge.py vec_convexp_bf16_denorm --out-txt <build>/out.txt --date 2026-06-10
# (writes over the json; git diff must show no change to the `silicon` array)
cd /home/triple/npu-work/xdna-emu && git diff --stat tools/golden/silicon_edge/
```

Expected: identical capture (silicon is deterministic, per phase B). If the two
passes differ, STOP and investigate before trusting the golden.

- [ ] **Step 4: Commit the silicon goldens**

```bash
git add tools/golden/silicon_edge/vec_convexp_bf16_denorm.json \
        tools/golden/silicon_edge/vec_vfloor_bf16_denorm.json
git commit -m "vector verify C: silicon goldens for VCONV expand + VFLOOR denormals

HW-captured on real NPU1 Phoenix (2 deterministic passes). Records the bf16
denormal -> {f32 expand, int32 floor} silicon outputs as the oracle, with the
no-FTZ reference kept for the divergence record. Verdict: <FTZ | no-FTZ> per
path (see divergence counts).

Generated using Claude Code."
```

---

## Task 5: Emulator execution + silicon comparison (the verdict)

**Files:** none yet (read-only investigation); produces the fix decision.

- [ ] **Step 1: Rebuild the FFI `.so` and regenerate with silicon EXP**

Now the silicon JSONs exist, so `_bake_io` bakes EXP = silicon:

```bash
cd /home/triple/npu-work/xdna-emu
cargo build -p xdna-emu-ffi
python3 tools/gen_vector_kernel.py vec_convexp_bf16_denorm
python3 tools/gen_vector_kernel.py vec_vfloor_bf16_denorm
cp -r tests/vector-verify/vec_convexp_bf16_denorm ../mlir-aie/test/npu-xrt/
cp -r tests/vector-verify/vec_vfloor_bf16_denorm  ../mlir-aie/test/npu-xrt/
```

- [ ] **Step 2: Run both kernels through the EMULATOR (no HW)**

```bash
./scripts/emu-bridge-test.sh --chess-only --no-hw \
  vec_convexp_bf16_denorm vec_vfloor_bf16_denorm \
  > build/experiments/vector-verify-C/emu-verdict.log 2>&1
```

Read the log. Two checkpoints:

1. **Decode/execute checkpoint:** did the emulator EXECUTE the VCONV/VFLOOR ops
   without erroring (unknown opcode, panic, wrong-result-shape)? If an op is not
   decoded or `vector_convert` panics on the conversion, that is a SEPARATE gap
   to fix first (decode/semantics), surfaced before the FTZ question. Note it.
2. **FTZ verdict:** does EMU output == silicon EXP?
   - PASS -> emulator already matches silicon for that path. If silicon was
     no-FTZ, that means the FTZ branch was never reached for this opcode (record
     why -- e.g. the expand writes via a path that doesn't flush) OR silicon DOES
     FTZ and the emulator's flush is correct. Cross-check against Task 4's
     divergence verdict to disambiguate.
   - FAIL at the negative-denormal lanes -> the emulator FTZs where silicon does
     not (the expected bug). Proceed to Task 6.

- [ ] **Step 3: Record the verdict**

Append to `build/experiments/vector-verify-C/SPIKE-FINDINGS.md`: per path, the
silicon verdict (FTZ / no-FTZ), the emulator result (match / mismatch), and the
exact `vector_convert.rs` line(s) implicated. This drives Task 6 and Task 8 docs.

---

## Task 6: Fix `vector_convert` FTZ where silicon disproves it

**Conditional:** only the branches Task 5 showed EMU != silicon (silicon no-FTZ).
If silicon FTZs everywhere tested, SKIP to Task 7 and instead add a comment
documenting the silicon-confirmed flush.

**Files:**
- Modify: `src/interpreter/execute/vector_convert.rs`
- Test: same file's `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing regression test (per disproven branch)**

Example for the bf16->f32 expand branch (adapt the conversion + expected to the
actual disproven branch from Task 5). Add to the `tests` module:

```rust
#[test]
fn bf16_to_f32_expand_does_not_flush_denormals() {
    // Silicon (NPU1 Phoenix, 2026-06-10) expands bf16 denormals to f32 denormals
    // rather than flushing them. bf16 0x0001 -> f32 0x0001_0000 (an f32 denormal).
    let mut src = [0u32; 8];
    // Two bf16 lanes per word; lane0 = 0x0001 (tiny +denormal), lane1 = 0x8001 (-).
    src[0] = 0x8001_0001;
    let out = VectorAlu::vector_convert(
        &src, ElementType::BFloat16, ElementType::Float32, RoundingMode::Floor);
    assert_eq!(out[0], 0x0001_0000, "positive bf16 denormal must widen, not flush");
    assert_eq!(out[1], 0x8001_0000, "negative bf16 denormal must widen with sign");
}
```

For the vfloor branch, if disproven:

```rust
#[test]
fn vfloor_bf16_to_i32_floors_negative_denormal_to_minus_one() {
    // Silicon does not flush: floor(tiny negative bf16) = -1, not 0.
    let mut src = [0u32; 8];
    src[0] = 0x8001_0001;  // lane0 +denormal, lane1 -denormal
    let out = VectorAlu::vector_convert(
        &src, ElementType::BFloat16, ElementType::Int32, RoundingMode::Floor);
    assert_eq!(out[0] as i32, 0,  "floor(tiny +denormal) = 0");
    assert_eq!(out[1] as i32, -1, "floor(tiny -denormal) = -1 (no FTZ)");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib bf16_to_f32_expand_does_not_flush_denormals vfloor_bf16_to_i32_floors_negative_denormal_to_minus_one`
Expected: FAIL (current code flushes -> got 0x0 / got 0).

- [ ] **Step 3: Remove the FTZ in the disproven branch(es)**

For bf16->f32 expand (`:267-275`), drop the flush:

```rust
            // BFloat16 -> Float32 (expand: 16 bf16 -> 8 f32, use lower half)
            (ElementType::BFloat16, ElementType::Float32) => {
                for i in 0..8 {
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // Silicon (NPU1 Phoenix, verified 2026-06-10) expands bf16
                    // denormals to f32 denormals -- it does NOT flush to zero.
                    result[i] = Self::bf16_to_f32(bf16).to_bits();
                }
            }
```

For bf16->int32 vfloor (`:336-346`):

```rust
            (ElementType::BFloat16, ElementType::Int32) => {
                for i in 0..8 {
                    let bf16 = (src[i / 2] >> ((i % 2) * 16)) as u16;
                    // Silicon does NOT flush bf16 denormals before floor; a tiny
                    // negative denormal floors to -1 (verified 2026-06-10).
                    let f = Self::bf16_to_f32(bf16);
                    result[i] = f.floor() as i32 as u32;
                }
            }
```

(Remove the now-unused `use super::vector_float::fp32_flush_to_zero;` lines in
those branches if they become dead. Leave the f32->bf16 / f32->int branches
exactly as-is unless Task 5 disproved them too.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib bf16_to_f32_expand_does_not_flush_denormals vfloor_bf16_to_i32_floors_negative_denormal_to_minus_one`
Expected: PASS.

- [ ] **Step 5: Full library test (no regressions)**

Run: `cargo test --lib`
Expected: all PASS (the count should be the prior baseline + the new tests).
Watch specifically for any existing convert test that asserted the OLD flushing
behavior -- if one fails, it encoded the unvalidated `ef77756` assumption; update
it to the silicon-verified behavior and note the change in the commit.

- [ ] **Step 6: Rebuild the `.so` and re-verify EMU == silicon on the kernels**

```bash
cargo build -p xdna-emu-ffi
cp -r tests/vector-verify/vec_convexp_bf16_denorm ../mlir-aie/test/npu-xrt/
cp -r tests/vector-verify/vec_vfloor_bf16_denorm  ../mlir-aie/test/npu-xrt/
./scripts/emu-bridge-test.sh --chess-only --no-hw \
  vec_convexp_bf16_denorm vec_vfloor_bf16_denorm \
  > build/experiments/vector-verify-C/emu-postfix.log 2>&1
```

Expected: both kernels now EMU == silicon (PASS).

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/execute/vector_convert.rs
git commit -m "vector convert: stop flushing bf16 denormals (silicon-verified)

NPU1 Phoenix expands bf16 denormals to f32 denormals and floors tiny negative
denormals to -1 -- it does NOT flush-to-zero in the standalone VCONV/VFLOOR
datapaths. The blanket fp32_flush_to_zero added in ef77756 was an unvalidated
accuracy-era generalization; silicon (vec_convexp_bf16_denorm,
vec_vfloor_bf16_denorm, captured 2026-06-10) disproves it. Per-branch regression
tests assert the no-flush behavior.

Generated using Claude Code."
```

---

## Task 7: Analytic disposition of the f32->int FTZ no-op

The f32->int32 / ->uint32 FTZ (`:307`, `:315`) is observationally a no-op (an f32
denormal lies in (-1, 1), truncating to 0 with or without the flush). Document
and pin it with a test rather than a HW run.

**Files:**
- Modify: `src/interpreter/execute/vector_convert.rs`
- Test: same file.

- [ ] **Step 1: Write the test**

```rust
#[test]
fn f32_to_int_denormal_truncates_to_zero_ftz_irrelevant() {
    // An f32 denormal is in (-1, 1) -> truncates to 0 whether or not it is first
    // flushed. The FTZ on this branch is a provable no-op (not silicon-verified
    // because it is unobservable). Guards against a future "fix" that changes it.
    let mut src = [0u32; 8];
    src[0] = 0x0000_0001; // smallest +f32 denormal
    src[1] = 0x8000_0001; // smallest -f32 denormal
    src[2] = 0x007F_FFFF; // largest +f32 denormal
    let out = VectorAlu::vector_convert(
        &src, ElementType::Float32, ElementType::Int32, RoundingMode::Floor);
    assert_eq!(out[0] as i32, 0);
    assert_eq!(out[1] as i32, 0);
    assert_eq!(out[2] as i32, 0);
}
```

- [ ] **Step 2: Run to verify it passes immediately**

Run: `cargo test --lib f32_to_int_denormal_truncates_to_zero_ftz_irrelevant`
Expected: PASS (documents existing behavior; no code change needed).

- [ ] **Step 3: Add the explanatory comment**

In the f32->int32 and f32->uint32 branches, replace the bare
`// AIE2 FTZ ...`-style line with:

```rust
                    // FTZ here is a provable no-op: an f32 denormal is in (-1, 1)
                    // and truncates to 0 either way. Kept for symmetry with the
                    // other convert branches; not silicon-observable. See the
                    // f32_to_int_denormal_truncates_to_zero test.
```

- [ ] **Step 4: Commit**

```bash
git add src/interpreter/execute/vector_convert.rs
git commit -m "vector convert: document f32->int FTZ as a provable no-op

An f32 denormal truncates to 0 with or without flushing, so the FTZ on the
f32->int32/uint32 branches is unobservable -- dispositioned analytically (no HW
run) and pinned with a regression test.

Generated using Claude Code."
```

---

## Task 8: Close-out (regression, docs, memory, gate readiness)

**Files:** docs + memory.

- [ ] **Step 1: Full regression sweep**

```bash
cargo test --lib
```
Expected: all PASS. Then the vector kernel regression (EMU-only is sufficient for
the convert audit; full HW only if a fix could affect other kernels):

```bash
./scripts/emu-bridge-test.sh --chess-only --no-hw \
  > build/experiments/vector-verify-C/regression-emu.log 2>&1
```
Expected: no NEW compute failures vs the phase-B baseline (the known pre-existing
`vec_mul_trace_distribute_lateral` Chess-compile FAIL may persist -- confirm it is
the same one, not a regression).

- [ ] **Step 2: Update the campaign plan doc**

In `docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md`, change
section C's heading to `## C. Op breadth on silicon  [DONE 2026-06-10 -- convert
FTZ audit]` and write the result: which paths were silicon-tested, the verdict
(FTZ / no-FTZ per path), the fix commit, and that op breadth (shuffle/vsel/vcmp/
vshift/min-max/reductions) folded into phase D (#112) by Maya's call.

- [ ] **Step 3: Update known-fidelity-gaps**

In `docs/known-fidelity-gaps.md`, update the convert/FTZ-related row(s): the
standalone VCONV/VFLOOR denormal behavior is now silicon-verified; note the
f32->int no-op disposition. If a row claimed FTZ-everywhere, correct it.

- [ ] **Step 4: Update the memory anchor**

Edit `/home/triple/.claude/projects/-home-triple-npu-work-xdna-emu/memory/project_vector_verification_depth_inflight.md`:
mark phase C DONE with the verdict + fix commit; set resume to phase D (#112,
vector differential fuzzer) then the gate flip (#113). Update the one-line
pointer in that dir's `MEMORY.md`.

- [ ] **Step 5: Mark task #111 complete**

Use TaskUpdate to set #111 (Vector verify C) to completed. Confirm the tracked
working tree is clean (untracked `tests/vector-verify/vec_*_bf16_denorm/` dirs
stay, per convention).

- [ ] **Step 6: Final commit (docs + memory are separate; memory is outside the repo)**

```bash
git add docs/superpowers/plans/2026-06-10-vector-verification-depth-AtoD.md \
        docs/known-fidelity-gaps.md
git commit -m "vector verify C: convert FTZ audit done (docs + gaps)

Standalone VCONV (bf16->f32 expand) and VFLOOR (bf16->int32 floor) silicon-
verified on NPU1 Phoenix; emulator FTZ corrected where silicon disproved it.
f32->int FTZ dispositioned as a provable no-op. Op breadth folds into phase D.

Generated using Claude Code."
```

---

## Self-review notes

- **Spec coverage:** spec's two kernels -> Tasks 2-5; the fix -> Task 6; f32->int
  analytic disposition -> Task 7; .lst opcode discipline -> Task 2 Step 3; silicon
  determinism (2 passes) -> Task 4 Step 3; docs/gate -> Task 8. The "standalone
  f32->bf16 VCONV, only if Chess cooperates" item from the spec is folded into
  Task 2's fall-back decision (Step 4) -- if a third standalone path proves
  emittable it gets the same kernel+capture treatment; if not, it is
  dispositioned by inference. Flagged here so it is not silently dropped.
- **Conditional tasks:** Task 6 is explicitly conditional on the Task 5 verdict;
  if silicon FTZs, the fix becomes a documenting comment instead. This is the
  honest both-outcomes shape from the spec.
- **No fabricated HW results:** Tasks 2/4/5 are discovery; their "expected"
  outcomes name decision points, not predetermined answers.
- **Type consistency:** `DirectIO(inputs, reference)` is used identically in
  Tasks 1/2/3; `_bake_io` signature unchanged; `_bf16_denorm_inputs`/
  `_bf16_floor_reference`/`_bf16_to_f32_bits` defined once (Task 2) and reused.
