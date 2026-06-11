"""Unit tests for gen_vector_kernel.py (Half-B vector-kernel generator).

The generator turns a per-kernel spec + the Half-A golden corpus
(tools/golden/vector_ops.json) into the four files a bridge kernel needs
(run.lit, aie.mlir, test.cpp, kernel.cc). The correctness anchor is that
regenerating a known-good kernel (vec_srs_i32) reproduces its committed
golden arrays exactly.
"""

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

import gen_vector_kernel as gen
from gen_vector_kernel import (
    Buf,
    DirectIO,
    KernelSpec,
    bake_array,
    render_test_cpp,
    select_records,
)

GOLDEN = json.loads((Path(__file__).parent / "golden" / "vector_ops.json").read_text())


def srs_spec():
    """The committed vec_srs_i32 kernel, expressed as a spec (validation anchor)."""
    return KernelSpec(
        name="vec_srs_i32",
        func="srs_i32",
        doc="SRS capture kernel.",
        inputs=[Buf("in", "int32_t", "i32")],
        output=Buf("out", "int16_t", "i16"),
        n=48,
        golden={
            "class": "srs",
            "filt": {"bits_o": 16, "signed": True, "sat": True,
                     "sym_sat": False, "rnd": 0, "shift": 4},
            "value_range": (-(2 ** 31), 2 ** 31 - 1),
        },
        defines=[("SRS_N", 48), ("SRS_SHIFT", 4)],
        body="  // body\n",
    )


class TestGoldenBaking:
    def test_srs_slice_reproduces_committed_input(self):
        # The committed vec_srs_i32 golden is this config slice of the srs
        # corpus, int32-representable accumulator values, in corpus order.
        recs = select_records(
            GOLDEN["srs"],
            filt={"bits_o": 16, "signed": True, "sat": True,
                  "sym_sat": False, "rnd": 0, "shift": 4},
            value_range=(-(2 ** 31), 2 ** 31 - 1),
        )
        vals = bake_array(recs, "value", n=48)
        exps = bake_array(recs, "expected", n=48)

        assert len(vals) == 48 and len(exps) == 48
        # Exact head match against tests/vector-verify/vec_srs_i32/test.cpp.
        assert vals[:11] == [0, 1, -255, 256, -256, -524288, -524287,
                             8388352, -524280, 8388353, 255]
        assert exps[:11] == [0, 0, -16, 16, -16, -32768, -32768,
                             32767, -32768, 32767, 15]
        # 43 real records, padded to 48 with zeros.
        assert vals[43:] == [0, 0, 0, 0, 0]
        assert exps[43:] == [0, 0, 0, 0, 0]


class TestRenderTestCpp:
    def test_bakes_golden_input_and_expected_arrays(self):
        cpp = render_test_cpp(srs_spec(), GOLDEN)
        # N and buffer element types come from the spec.
        assert "static constexpr int N = 48;" in cpp
        assert "int32_t *bufIn" in cpp
        assert "int16_t *bufOut" in cpp
        # Golden arrays are baked from the corpus (head match, corpus order).
        assert "0, 1, -255, 256, -256, -524288, -524287," in cpp
        assert "0, 0, -16, 16, -16, -32768, -32768," in cpp
        # Exactly one input buffer => single dma input arg in the kernel call.
        assert "bo_in" in cpp and "bo_out" in cpp
        # int8 outputs must print as integers, not raw chars (readable errors).
        assert "(int)bufOut[i]" in cpp and "(int)IN[i]" in cpp and "(int)EXP[i]" in cpp


class TestRenderMlir:
    def test_emits_func_sig_objectfifos_and_dma(self):
        from gen_vector_kernel import render_mlir
        mlir = render_mlir(srs_spec())
        # private func decl with N-shaped memrefs and the linked object file.
        assert "func.func private @srs_i32(memref<48xi32>, memref<48xi16>)" in mlir
        assert 'link_with = "srs.o"' in mlir
        # one input + one output objectfifo, correctly shaped/directional.
        assert "@inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<48xi32>>" in mlir
        assert "@outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<48xi16>>" in mlir
        # shim DMA moves N elements in and N out.
        assert "[1, 1, 1, 48][0, 0, 0, 1]" in mlir
        assert "npu1_1col" in mlir


class TestRenderKernel:
    def test_wraps_body_with_signature_defines_and_includes(self):
        from gen_vector_kernel import render_kernel
        spec = srs_spec()
        spec.body = "  event0();\n  // intrinsic IP here\n  event1();\n"
        cc = render_kernel(spec)
        assert "#include <aie_api/aie.hpp>" in cc
        assert "#define SRS_N 48" in cc
        assert "#define SRS_SHIFT 4" in cc
        # signature derived from inputs+output, each restrict-qualified.
        assert 'extern "C" {' in cc
        assert "void srs_i32(int32_t *restrict in, int16_t *restrict out) {" in cc
        assert "  event0();" in cc and "  event1();" in cc


class TestRenderRunLit:
    def test_emits_chess_compile_and_run_recipe(self):
        from gen_vector_kernel import render_run_lit
        lit = render_run_lit(srs_spec())
        assert "REQUIRES: ryzen_ai_npu1, chess" in lit
        assert "xchesscc_wrapper aie2" in lit and "%S/srs.cc -o ./srs.o" in lit
        assert "%S/aie.mlir" in lit
        assert "%S/test.cpp -o test.exe" in lit
        assert "%run_on_npu1% ./test.exe" in lit


REPO = Path(__file__).resolve().parent.parent


class TestGenerate:
    def test_writes_four_files(self, tmp_path):
        from gen_vector_kernel import generate
        outdir = generate(srs_spec(), GOLDEN, tmp_path)
        for fn in ["run.lit", "aie.mlir", "test.cpp", "srs.cc"]:
            assert (Path(outdir) / fn).is_file(), f"missing {fn}"

    def test_regenerated_srs_golden_matches_committed(self, tmp_path):
        """The generator's baked arrays must equal the hand-authored committed ones."""
        import re
        from gen_vector_kernel import generate

        def nums(text, name):
            m = re.search(name + r"\[\w+\]\s*=\s*\{([^}]*)\}", text, re.S)
            return [int(x) for x in re.findall(r"-?\d+", m.group(1))]

        outdir = generate(srs_spec(), GOLDEN, tmp_path)
        gen = (Path(outdir) / "test.cpp").read_text()
        committed = (REPO / "tests/vector-verify/vec_srs_i32/test.cpp").read_text()

        assert nums(gen, "IN") == nums(committed, "IN")
        assert nums(gen, "EXP") == nums(committed, "EXP")


class TestRegistry:
    def test_shipped_srs_spec_regenerates_committed_golden(self, tmp_path):
        """The spec we actually ship (registry) -- not a test fixture -- must
        reproduce the committed vec_srs_i32 golden arrays."""
        import re
        from gen_vector_kernel import generate
        from vector_kernel_specs import SPECS

        def nums(text, name):
            m = re.search(name + r"\[\w+\]\s*=\s*\{([^}]*)\}", text, re.S)
            return [int(x) for x in re.findall(r"-?\d+", m.group(1))]

        outdir = generate(SPECS["vec_srs_i32"], GOLDEN, tmp_path)
        gen = (Path(outdir) / "test.cpp").read_text()
        committed = (REPO / "tests/vector-verify/vec_srs_i32/test.cpp").read_text()
        assert nums(gen, "IN") == nums(committed, "IN")
        assert nums(gen, "EXP") == nums(committed, "EXP")

    def test_every_spec_has_a_resolvable_golden_slice(self):
        """No spec names a golden config that yields zero records (silent empty)."""
        from gen_vector_kernel import select_records
        from vector_kernel_specs import SPECS
        for name, spec in SPECS.items():
            g = spec.golden
            if g["class"] == "direct":
                # Direct specs carry their own input/reference arrays instead of
                # slicing a corpus class; the anti-silent-empty / fits-N guarantee
                # still holds, just against the DirectIO tuples.
                d = g["direct"]
                assert d.inputs, f"{name}: direct inputs empty"
                assert len(d.inputs) == len(d.reference) == spec.n, \
                    f"{name}: direct arrays len {len(d.inputs)}/{len(d.reference)} != N={spec.n}"
                continue
            recs = select_records(GOLDEN[g["class"]], g["filt"],
                                  g.get("value_range"), predicate=g.get("predicate"))
            assert recs, f"{name}: golden slice is empty"
            if spec.matmul is not None:
                # Matmul specs consume `batch` records (each a full tile), not
                # one element per record; the slice must hold at least a batch.
                assert len(recs) >= spec.matmul.batch, \
                    f"{name}: {len(recs)} records < batch {spec.matmul.batch}"
            else:
                assert len(recs) <= spec.n, f"{name}: {len(recs)} records > N={spec.n}"


class TestKernelTypeSplit:
    """A Buf's host-staging C type (for baking golden bit patterns) can differ
    from the kernel-signature type. The f32->bf16 conv kernel stages uint32/
    uint16 bit patterns on the host but the kernel reads float/bfloat16."""

    def test_buf_kernel_ctype_defaults_to_host_ctype(self):
        assert Buf("x", "int32_t", "i32").kernel_ctype == "int32_t"

    def test_buf_kernel_ctype_overrides_when_set(self):
        assert Buf("x", "uint32_t", "f32", ktype="float").kernel_ctype == "float"

    def test_signature_uses_kernel_ctype_not_host_ctype(self):
        from gen_vector_kernel import _signature
        spec = KernelSpec(
            name="vec_conv_bf16", func="conv_bf16", doc="conv.",
            inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
            output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
            n=16, golden={"class": "bf16_srs", "filt": {"rnd": 12}},
            body="  // body\n",
        )
        assert _signature(spec) == "float *restrict in, bfloat16 *restrict out"

    def test_test_cpp_buffers_use_host_staging_ctype(self):
        # Host buffers + baked arrays use the staging type (uint32/uint16),
        # so exact golden bit patterns are moved byte-for-byte over DMA.
        spec = KernelSpec(
            name="vec_conv_bf16", func="conv_bf16", doc="conv.",
            inputs=[Buf("in", "uint32_t", "f32", ktype="float")],
            output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"),
            n=600, golden={"class": "bf16_srs", "filt": {"rnd": 12}},
            body="  // body\n",
        )
        cpp = render_test_cpp(spec, GOLDEN)
        assert "uint32_t *bufIn" in cpp
        assert "uint16_t *bufOut" in cpp


class TestEdgePredicates:
    def test_is_edge_f32_selects_nonnormal(self):
        from vector_kernel_specs import _is_edge_f32
        for v in (0x00010000, 0x7F800001, 0x7F800000, 0x00000000):
            assert _is_edge_f32({"value": v})
        assert not _is_edge_f32({"value": 0x3F800000})

    def test_is_edge_is_complement_of_normal(self):
        from vector_kernel_specs import _is_edge_f32, _is_normal_f32
        for v in [0x0, 0x1, 0x00010000, 0x3F800000, 0x7F800000, 0x7F800001, 0xFF800000]:
            assert _is_edge_f32({"value": v}) != _is_normal_f32({"value": v})

    def test_has_overflow_expected(self):
        from vector_kernel_specs import _has_overflow_expected, _all_expected_finite_f32
        ovf = {"expected": [0x3F800000, 0x7F800000]}
        fin = {"expected": [0x3F800000, 0x40000000]}
        assert _has_overflow_expected(ovf)
        assert not _has_overflow_expected(fin)
        assert not _all_expected_finite_f32(ovf)
        assert _all_expected_finite_f32(fin)


class TestSelectRecordsPredicate:
    def test_predicate_filters_records_in_order(self):
        recs = [{"value": 1, "rnd": 12}, {"value": 2, "rnd": 12},
                {"value": 3, "rnd": 12}, {"value": 4, "rnd": 8}]
        out = select_records(recs, {"rnd": 12},
                             predicate=lambda r: r["value"] % 2 == 1)
        assert [r["value"] for r in out] == [1, 3]


class TestMatmulSupport:
    """The matmul-shape generator variant: two row-major input buffers (A, B)
    and one output (C), baked by unpacking the matmul golden's row-major-packed
    vec512 words and concatenating a batch of independent tile multiplies."""

    def test_unpack_vec512_int8_little_endian(self):
        from gen_vector_kernel import unpack_vec512
        assert unpack_vec512([0x04030201, 0x08070605], 8, 1) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_unpack_vec512_int16_signed(self):
        from gen_vector_kernel import unpack_vec512
        # word low half 0xFFFF = -1 (signed int16), high half 0x0002 = 2
        assert unpack_vec512([0x0002FFFF], 2, 2) == [-1, 2]

    def test_unpack_vec512_bf16_unsigned_bits(self):
        from gen_vector_kernel import unpack_vec512
        # raw bf16 bit patterns, not sign-interpreted
        assert unpack_vec512([0xBF803F80], 2, 2, signed=False) == [0x3F80, 0xBF80]

    def test_bake_matmul_concatenates_batch_row_major(self):
        from gen_vector_kernel import Matmul, bake_matmul
        mm = Matmul(M=4, K=8, N=8, a_bytes=1, b_bytes=1, batch=2)
        filt = {"a_type": "Int8", "b_type": "Int8", "rows": 4, "inner": 8,
                "cols": 8, "subtract": False}
        A, B, C = bake_matmul(GOLDEN["matmul"], filt, mm)
        assert len(A) == 2 * 32 and len(B) == 2 * 64 and len(C) == 2 * 32
        # The first tile's row-major A.B must equal its baked C (oracle anchor).
        for r in range(4):
            for c in range(8):
                s = sum(A[r * 8 + k] * B[k * 8 + c] for k in range(8))
                got = C[r * 8 + c]
                # both wrapped to i32
                assert (s & 0xFFFFFFFF) == (got & 0xFFFFFFFF), f"C[{r},{c}]"

    def test_render_mlir_two_inputs_emits_both_objectfifos(self):
        from gen_vector_kernel import Matmul, render_mlir
        spec = _mac_i8_spec()
        mlir = render_mlir(spec)
        # func decl carries three memrefs (A=8*32, B=8*64, C=8*32).
        assert "memref<256xi8>, memref<512xi8>, memref<256xi32>" in mlir
        # two input objectfifos + one output, correctly shaped.
        assert "@inA(%shim, {%core}" in mlir and "@inB(%shim, {%core}" in mlir
        assert "@outC(%core, {%shim}" in mlir

    def test_render_test_cpp_two_inputs_two_buffers(self):
        from gen_vector_kernel import render_test_cpp
        cpp = render_test_cpp(_mac_i8_spec(), GOLDEN)
        assert "int8_t *bufInA" in cpp and "int8_t *bufInB" in cpp
        assert "int32_t *bufOut" in cpp
        assert "INA" in cpp and "INB" in cpp and "EXP" in cpp

    def test_integer_matmul_specs_golden_matches_textbook(self):
        """Each shipped integer matmul spec's baked C must equal the textbook
        product of its baked row-major A/B. The kernel feeds SIGNED inputs, so
        the golden slice must select the signed-input records (x_signed/y_signed
        True) -- otherwise unsigned-sampled records (whose high-bit bytes the
        signed kernel reads as negative) diverge from their stored expected."""
        from gen_vector_kernel import bake_matmul
        from vector_kernel_specs import SPECS
        for name in ("vec_mac_i8", "vec_mac_i16"):
            spec = SPECS[name]
            mm = spec.matmul
            a_vals, b_vals, c_vals = bake_matmul(
                GOLDEN["matmul"], spec.golden["filt"], mm,
                predicate=spec.golden.get("predicate"))
            sa, sb, sc = mm.size_a, mm.size_b, mm.size_c
            for t in range(mm.batch):
                a = a_vals[t * sa:(t + 1) * sa]
                b = b_vals[t * sb:(t + 1) * sb]
                c = c_vals[t * sc:(t + 1) * sc]
                for i in range(mm.M):
                    for j in range(mm.N):
                        s = sum(a[i * mm.K + k] * b[k * mm.N + j] for k in range(mm.K))
                        s = ((s + 2 ** 31) % 2 ** 32) - 2 ** 31  # int32 wrap
                        assert s == c[i * mm.N + j], \
                            f"{name} tile {t} [{i},{j}]: textbook {s} != golden {c[i * mm.N + j]}"

    def test_bake_matmul_honors_predicate(self):
        """bake_matmul forwards a record predicate to select_records, so a
        matmul spec can exclude records whose expected output has NaN/Inf lanes
        (the bf16 overflow edge that breaks an exact bit-compare)."""
        from gen_vector_kernel import Matmul, bake_matmul, select_records
        filt = {"a_type": "BFloat16", "b_type": "BFloat16", "rows": 4,
                "inner": 8, "cols": 4, "subtract": False}
        finite = lambda r: all(((b >> 23) & 0xFF) != 0xFF for b in r["expected"])
        full = select_records(GOLDEN["matmul"], filt)
        clean = select_records(GOLDEN["matmul"], filt, predicate=finite)
        # The predicate must actually drop records (bf16 slice has overflow).
        assert 0 < len(clean) < len(full)
        mm = Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2,
                    batch=len(clean), bfloat=True)
        A, B, C = bake_matmul(GOLDEN["matmul"], filt, mm, predicate=finite)
        # No baked C lane is NaN/Inf (exp field 0xFF) -> exact bit-compare safe.
        assert all(((c >> 23) & 0xFF) != 0xFF for c in C)
        assert len(C) == len(clean) * mm.size_c

    def test_render_test_cpp_passes_spec_predicate_to_bake(self):
        """A matmul spec's golden predicate reaches bake_matmul through
        render_test_cpp (not just direct bake_matmul calls)."""
        from gen_vector_kernel import Buf, KernelSpec, Matmul, render_test_cpp
        finite = lambda r: all(((b >> 23) & 0xFF) != 0xFF for b in r["expected"])
        spec = KernelSpec(
            name="vec_mac_bf16", func="mac_bf16", doc="bf16 matmul.",
            inputs=[Buf("inA", "uint16_t", "bf16", ktype="bfloat16"),
                    Buf("inB", "uint16_t", "bf16", ktype="bfloat16")],
            output=Buf("out", "uint32_t", "f32", ktype="float"),
            n=0,
            golden={"class": "matmul",
                    "filt": {"a_type": "BFloat16", "b_type": "BFloat16",
                             "rows": 4, "inner": 8, "cols": 4,
                             "subtract": False},
                    "predicate": finite},
            matmul=Matmul(M=4, K=8, N=4, a_bytes=2, b_bytes=2,
                          batch=24, bfloat=True),
            body="  // bf16 mmul body\n",
        )
        cpp = render_test_cpp(spec, GOLDEN)
        import re
        m = re.search(r"EXP\[\w+\]\s*=\s*\{([^}]*)\}", cpp, re.S)
        exp = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
        # Predicate applied: no NaN/Inf lanes in the baked expected.
        assert exp and all(((c & 0xFFFFFFFF) >> 23) & 0xFF != 0xFF for c in exp)


def _mac_i8_spec():
    from gen_vector_kernel import Buf, KernelSpec, Matmul
    # batch of 8 native 4x8x8 i8 tiles: A=8*32, B=8*64, C=8*32.
    return KernelSpec(
        name="vec_mac_i8", func="mac_i8", doc="i8 matmul.",
        inputs=[Buf("inA", "int8_t", "i8"), Buf("inB", "int8_t", "i8")],
        output=Buf("out", "int32_t", "i32"),
        n=0,
        golden={"class": "matmul",
                "filt": {"a_type": "Int8", "b_type": "Int8", "rows": 4,
                         "inner": 8, "cols": 8, "subtract": False}},
        matmul=__import__("gen_vector_kernel").Matmul(
            M=4, K=8, N=8, a_bytes=1, b_bytes=1, batch=8),
        body="  // mmul body\n",
    )


class TestModeSweep:
    """SweepSpec.expand()s a class's full crRnd/crSat space into one KernelSpec
    per mode-point, keeping the swept body (set_rounding/set_saturation) and the
    baked golden slice in lockstep -- the move that makes the silicon check
    mode-exhaustive rather than one-representative-per-class."""

    def test_rounding_and_sat_enum_maps_match_toolchain(self):
        # crRnd index -> aie::rounding_mode name, per aietools me_defines.h.
        from gen_vector_kernel import ROUNDING_ENUM, SAT_ENUM
        assert ROUNDING_ENUM == {
            0: "floor", 1: "ceil", 2: "symmetric_floor", 3: "symmetric_ceil",
            8: "negative_inf", 9: "positive_inf", 10: "symmetric_zero",
            11: "symmetric_inf", 12: "conv_even", 13: "conv_odd",
        }
        # crSat index -> saturation_mode name (none=0, saturate=1, symmetric=3).
        assert SAT_ENUM == {0: "none", 1: "saturate", 3: "symmetric"}

    def test_mode_lines_emit_only_present_axes(self):
        from gen_vector_kernel import mode_lines
        # both axes
        ml = mode_lines(12, 1)
        assert "aie::rounding_mode::conv_even" in ml
        assert "aie::saturation_mode::saturate" in ml
        # rounding-only class (no crsat): no set_saturation line
        assert "set_saturation" not in mode_lines(0, None)
        assert "aie::rounding_mode::floor" in mode_lines(0, None)
        # sat-only class (no crrnd): no set_rounding line
        assert "set_rounding" not in mode_lines(None, 3)
        assert "aie::saturation_mode::symmetric" in mode_lines(None, 3)

    def test_srs_sweep_expands_to_full_cross_product(self):
        from vector_kernel_specs import SWEEPS
        specs = SWEEPS["vec_srs_i32_sweep"].expand()
        # 10 rounding modes x 3 saturation modes.
        assert len(specs) == 30
        names = {s.name for s in specs}
        assert "vec_srs_i32_r0_s0" in names
        assert "vec_srs_i32_r12_s3" in names
        assert "vec_srs_i32_r13_s1" in names

    def test_sweep_point_golden_filter_matches_its_body_mode(self):
        """The baked golden slice's mode fields must equal the crRnd/crSat the
        kernel body sets -- otherwise the kernel drives one mode while the
        expected values came from another (the lockstep guarantee)."""
        from vector_kernel_specs import SWEEPS
        by_name = {s.name: s for s in SWEEPS["vec_srs_i32_sweep"].expand()}
        # rnd=12 (conv_even), crsat=3 (symmetric) -> filt {rnd:12, sat:True, sym_sat:True}
        s = by_name["vec_srs_i32_r12_s3"]
        assert s.golden["filt"]["rnd"] == 12
        assert s.golden["filt"]["sat"] is True
        assert s.golden["filt"]["sym_sat"] is True
        assert "aie::rounding_mode::conv_even" in s.body
        assert "aie::saturation_mode::symmetric" in s.body
        # crsat=0 (none) -> filt {sat:False, sym_sat:False}
        s0 = by_name["vec_srs_i32_r0_s0"]
        assert s0.golden["filt"]["sat"] is False
        assert s0.golden["filt"]["sym_sat"] is False
        assert "aie::saturation_mode::none" in s0.body

    def test_pack_sweep_is_saturation_only(self):
        from vector_kernel_specs import SWEEPS
        specs = SWEEPS["vec_pack_i16_sweep"].expand()
        assert {s.name for s in specs} == {
            "vec_pack_i16_s0", "vec_pack_i16_s1", "vec_pack_i16_s3"}
        for s in specs:
            assert "set_rounding" not in s.body  # pack does not round
            assert "rnd" not in s.golden["filt"]

    def test_convert_sweep_is_rounding_only_normals(self):
        from vector_kernel_specs import SWEEPS
        specs = SWEEPS["vec_conv_bf16_sweep"].expand()
        assert len(specs) == 10  # 10 rounding modes, no saturation axis
        for s in specs:
            assert "set_saturation" not in s.body
            assert s.golden.get("predicate") is not None  # normals-only

    def test_ups_sweep_two_sat_modes_no_symmetric(self):
        from vector_kernel_specs import SWEEPS
        specs = SWEEPS["vec_ups_i32_sweep"].expand()
        assert {s.name for s in specs} == {"vec_ups_i32_s0", "vec_ups_i32_s1"}
        for s in specs:
            assert "sym_sat" not in s.golden["filt"]  # ups corpus has no such column

    def test_every_sweep_point_has_a_resolvable_golden_slice_that_fits(self):
        """No mode-point names an empty golden slice, and every slice fits its
        buffer N (so bake_array never silently truncates a mode's records)."""
        from gen_vector_kernel import select_records
        from vector_kernel_specs import SWEEPS
        for sname, sweep in SWEEPS.items():
            for spec in sweep.expand():
                g = spec.golden
                recs = select_records(GOLDEN[g["class"]], g["filt"],
                                      g.get("value_range"),
                                      predicate=g.get("predicate"))
                assert recs, f"{spec.name}: golden slice is empty"
                assert len(recs) <= spec.n, \
                    f"{spec.name}: {len(recs)} records > N={spec.n}"

    def test_sweep_kernel_generates_four_files(self, tmp_path):
        from gen_vector_kernel import generate
        from vector_kernel_specs import SWEEPS
        spec = SWEEPS["vec_srs_i32_sweep"].expand()[0]
        outdir = generate(spec, GOLDEN, tmp_path)
        for fn in ["run.lit", "aie.mlir", "test.cpp", "srs.cc"]:
            assert (Path(outdir) / fn).is_file(), f"missing {fn}"


class TestOutputDump(unittest.TestCase):
    def setUp(self):
        import os
        gp = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
        self.golden = json.loads(open(gp).read())

    def test_elementwise_harness_dumps_out_txt(self):
        from vector_kernel_specs import SPECS
        cpp = gen.render_test_cpp(SPECS["vec_srs_i32"], self.golden)
        self.assertIn('std::ofstream', cpp)
        self.assertIn('"out.txt"', cpp)
        self.assertIn('for (int i = 0; i < N; i++)', cpp)

    def test_matmul_harness_dumps_out_txt(self):
        from vector_kernel_specs import SPECS
        cpp = gen.render_test_cpp(SPECS["vec_mac_i8"], self.golden)
        self.assertIn('std::ofstream', cpp)
        self.assertIn('"out.txt"', cpp)


class TestSiliconGoldenSource:
    def test_silicon_field_defaults_none(self):
        spec = gen.KernelSpec(name="x", func="x", doc="",
                              inputs=[gen.Buf("in","int16_t","i16")],
                              output=gen.Buf("out","int16_t","i16"), n=4,
                              golden={"class":"srs","filt":{}}, body="")
        assert spec.silicon_golden is None

    def test_bootstrap_bakes_model_when_no_silicon_file(self):
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        miss = gen.replace_silicon(pt, "/nonexistent.json")
        in_vals, exp_vals = gen._bake_io(miss, GOLDEN)
        assert len(exp_vals) == pt.n

    def test_silicon_file_overrides_exp_keeps_inputs(self, tmp_path):
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        in_model, _ = gen._bake_io(gen.replace_silicon(pt, None), GOLDEN)
        sj = {"kernel": pt.name, "n": pt.n, "input": in_model,
              "silicon": [0xDEAD]*pt.n, "model": [0]*pt.n, "divergences": []}
        path = tmp_path / "sj.json"
        path.write_text(json.dumps(sj))
        in_vals, exp_vals = gen._bake_io(gen.replace_silicon(pt, str(path)), GOLDEN)
        assert exp_vals == [0xDEAD]*pt.n
        assert in_vals == in_model

    def test_sweep_assigns_silicon_path_per_point(self):
        from vector_kernel_specs import SWEEPS
        for pt in SWEEPS["vec_conv_bf16_edge_sweep"].expand():
            assert pt.silicon_golden is not None
            assert pt.name in pt.silicon_golden


class TestEdgeSpecs:
    def test_convert_edge_sweep_is_ten_points(self):
        from vector_kernel_specs import SWEEPS
        pts = SWEEPS["vec_conv_bf16_edge_sweep"].expand()
        assert len(pts) == 10
        assert {p.name for p in pts} == {f"vec_conv_bf16_edge_r{r}" for r in (0,1,2,3,8,9,10,11,12,13)}

    def test_convert_edge_inputs_all_nonnormal(self):
        from vector_kernel_specs import SWEEPS, _is_normal_f32
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        in_vals, _ = gen._bake_io(gen.replace_silicon(pt, None), GOLDEN)
        for v in in_vals:
            assert not _is_normal_f32({"value": v}), f"{v:#x} normal"

    def test_convert_edge_slice_fits_buffer(self):
        from vector_kernel_specs import SWEEPS
        pt = SWEEPS["vec_conv_bf16_edge_sweep"].expand()[0]
        recs = gen.select_records(GOLDEN["bf16_srs"], pt.golden["filt"],
                                  predicate=pt.golden["predicate"])
        assert len(recs) <= pt.n
        assert len(recs) > 0

    def test_mac_ovf_selects_overflow_tiles(self):
        from vector_kernel_specs import SPECS, _has_overflow_expected
        spec = SPECS["vec_mac_bf16_ovf"]
        recs = gen.select_records(GOLDEN["matmul"], spec.golden["filt"],
                                  predicate=spec.golden["predicate"])
        assert len(recs) >= spec.matmul.batch
        for r in recs[:spec.matmul.batch]:
            assert _has_overflow_expected(r)


class TestDirectInput(unittest.TestCase):
    """A spec whose `golden` carries a DirectIO sources its inputs and bootstrap
    reference straight from the spec, bypassing the model corpus -- for kernels
    (bf16 denormal convert) whose oracle is silicon and whose input space no
    corpus class produces. The silicon-override block runs on top of the direct
    path unchanged: a present capture overrides the expected array, gated by an
    input-alignment cross-check."""

    @staticmethod
    def _direct_spec():
        return KernelSpec(
            name="vec_conv_bf16_direct",
            func="conv_bf16",
            doc="direct-input convert.",
            inputs=[Buf("in", "uint16_t", "bf16", ktype="bfloat16")],
            output=Buf("out", "uint32_t", "f32", ktype="float"),
            n=4,
            golden={"class": "direct",
                    "direct": DirectIO(
                        inputs=(1, 2, 0x8001, 0),
                        reference=(0x10000, 0x20000, 0x80010000, 0))},
            body="  // body\n",
        )

    def test_direct_bypasses_corpus(self):
        # Empty corpus proves the class is never consulted on the direct path.
        in_vals, exp_vals = gen._bake_io(self._direct_spec(), {})
        self.assertEqual(in_vals, [1, 2, 0x8001, 0])
        self.assertEqual(exp_vals, [0x10000, 0x20000, 0x80010000, 0])

    def test_direct_silicon_overrides_reference(self):
        spec = self._direct_spec()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sj.json")
            with open(path, "w") as f:
                json.dump({"input": [1, 2, 0x8001, 0],
                           "silicon": [0, 0, 0x80000000, 0]}, f)
            in_vals, exp_vals = gen._bake_io(gen.replace_silicon(spec, path), {})
        self.assertEqual(in_vals, [1, 2, 0x8001, 0])
        self.assertEqual(exp_vals, [0, 0, 0x80000000, 0])

    def test_direct_wrong_length_raises(self):
        # A direct spec whose tuples don't match n must fail loudly, the same
        # anti-silent-truncation guard bake_array gives the corpus path.
        spec = dataclasses.replace(
            self._direct_spec(),
            golden={"class": "direct",
                    "direct": DirectIO(inputs=(1, 2, 3),
                                       reference=(1, 2, 3))})
        with self.assertRaises(AssertionError):
            gen._bake_io(spec, {})


class TestKernelHeaderFormatting:
    def test_ruler_line_is_eighty_columns(self):
        """The kernel.cc header ruler matches the 80-col LLVM convention
        regardless of stem length (the other files' rulers are fixed-width)."""
        from gen_vector_kernel import render_kernel
        first = render_kernel(srs_spec()).splitlines()[0]
        assert len(first) == 80, f"ruler is {len(first)} cols: {first!r}"
        assert first.startswith("//===- srs.cc ")
        assert first.endswith("*- C++ -*-===//")


class TestConvertFtzAuditSpecs(unittest.TestCase):
    """The two bf16 convert FTZ-audit capture kernels (Half-B phase C). Both
    sweep the exhaustive bf16-denormal input space (all 254 denormals + +/-0)
    via DirectIO, so their oracle is silicon and their bootstrap reference is
    the no-FTZ widen/floor. vec_vfloor_bf16_denorm is the real audit test: a
    standalone VFLOOR routes through vector_convert's bf16->int32 FTZ, so a
    negative denormal floors to -1 without FTZ (vs 0 with it)."""

    def test_audit_specs_generate_with_direct_denorm_inputs(self):
        import vector_kernel_specs as vks
        for name in ("vec_convexp_bf16_denorm", "vec_vfloor_bf16_denorm"):
            spec = vks.SPECS[name]
            d = spec.golden["direct"]
            # 254 denormals + 2 zeros, exhaustive, length 256.
            self.assertEqual(len(d.inputs), 256)
            self.assertEqual(len(d.reference), 256)
            # negative bf16 denormal present (the FTZ discriminator).
            self.assertIn(0x8001, d.inputs)
            # bootstrap bake (no silicon json) returns the direct arrays.
            in_vals, exp_vals = gen._bake_io(gen.replace_silicon(spec, None), {})
            self.assertEqual(in_vals, list(d.inputs))
            self.assertEqual(exp_vals, list(d.reference))

    def test_vfloor_reference_floors_negative_denormal_to_minus_one(self):
        import vector_kernel_specs as vks
        spec = vks.SPECS["vec_vfloor_bf16_denorm"]
        d = spec.golden["direct"]
        idx = d.inputs.index(0x8001)            # tiny negative denormal
        self.assertEqual(d.reference[idx], -1)  # floor(-tiny) = -1, signed (no FTZ)
        idx0 = d.inputs.index(0x0001)           # tiny positive denormal
        self.assertEqual(d.reference[idx0], 0)  # floor(+tiny) = 0

    def test_convexp_reference_widens_denormal_without_flush(self):
        import vector_kernel_specs as vks
        spec = vks.SPECS["vec_convexp_bf16_denorm"]
        d = spec.golden["direct"]
        idx = d.inputs.index(0x0001)
        self.assertEqual(d.reference[idx], 0x00010000)  # widened f32 denormal, not flushed
