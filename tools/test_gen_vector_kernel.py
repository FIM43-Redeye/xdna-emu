"""Unit tests for gen_vector_kernel.py (Half-B vector-kernel generator).

The generator turns a per-kernel spec + the Half-A golden corpus
(tools/golden/vector_ops.json) into the four files a bridge kernel needs
(run.lit, aie.mlir, test.cpp, kernel.cc). The correctness anchor is that
regenerating a known-good kernel (vec_srs_i32) reproduces its committed
golden arrays exactly.
"""

import json
from pathlib import Path

from gen_vector_kernel import (
    Buf,
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

        assert nums(gen, "IN") == nums(committed, "SRS_IN")
        assert nums(gen, "EXP") == nums(committed, "SRS_EXP")


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
        assert nums(gen, "IN") == nums(committed, "SRS_IN")
        assert nums(gen, "EXP") == nums(committed, "SRS_EXP")

    def test_every_spec_has_a_resolvable_golden_slice(self):
        """No spec names a golden config that yields zero records (silent empty)."""
        from gen_vector_kernel import select_records
        from vector_kernel_specs import SPECS
        for name, spec in SPECS.items():
            g = spec.golden
            recs = select_records(GOLDEN[g["class"]], g["filt"], g.get("value_range"))
            assert recs, f"{name}: golden slice is empty"
            assert len(recs) <= spec.n, f"{name}: {len(recs)} records > N={spec.n}"


class TestKernelHeaderFormatting:
    def test_ruler_line_is_eighty_columns(self):
        """The kernel.cc header ruler matches the 80-col LLVM convention
        regardless of stem length (the other files' rulers are fixed-width)."""
        from gen_vector_kernel import render_kernel
        first = render_kernel(srs_spec()).splitlines()[0]
        assert len(first) == 80, f"ruler is {len(first)} cols: {first!r}"
        assert first.startswith("//===- srs.cc ")
        assert first.endswith("*- C++ -*-===//")
