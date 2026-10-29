"""Unit tests for gen_vector_kernel.py and the designed inputs (vector_inputs.py).

The generator turns a per-kernel spec (designed inputs + intrinsic body) and an
optional silicon capture into the four files a bridge kernel needs. No external
model or corpus is involved: inputs come from vector_inputs.py, expected values
from tools/golden/silicon/<kernel>.json (bootstrap: zeros).
"""

import dataclasses
import json
import re
from pathlib import Path

import pytest

import gen_vector_kernel as gen
import vector_inputs as vi
from gen_vector_kernel import Buf, DirectIO, KernelSpec, render_test_cpp
from vector_kernel_specs import SPECS, SWEEPS

TOOLS = Path(__file__).resolve().parent


def all_specs():
    out = dict(SPECS)
    for sw in SWEEPS.values():
        for pt in sw.expand():
            out[pt.name] = pt
    return out


def srs_spec(silicon=None):
    return KernelSpec(
        name="vec_srs_t", func="srs_i32", doc="SRS capture kernel.",
        inputs=[Buf("in", "int32_t", "i32")], output=Buf("out", "int16_t", "i16"),
        n=4, io_class="srs", direct=DirectIO(inputs=((1, -2, 3, 40),)),
        defines=[("SRS_N", 4), ("SRS_SHIFT", 4)], body="  // body\n",
        silicon_golden=silicon or "/nonexistent/none.json")


def nums(text, name):
    m = re.search(name + r"\[\w+\]\s*=\s*\{([^}]*)\}", text, re.S)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


class TestRegistry:
    def test_sixty_six_kernels(self):
        assert len(all_specs()) == 119

    def test_every_spec_has_inputs_that_fit(self):
        for name, spec in all_specs().items():
            ins, exp = gen._bake_io(spec)   # asserts input/EXP lengths
            assert len(exp) == spec.n_out, name

    def test_every_spec_has_designed_inputs(self):
        for name, spec in all_specs().items():
            assert spec.direct is not None, name

    def test_every_kernel_has_a_silicon_capture(self):
        for name, spec in all_specs().items():
            assert Path(spec.silicon_path).is_file(), f"{name}: not captured"

    def test_silicon_records_follow_schema(self):
        for name, spec in all_specs().items():
            rec = json.loads(Path(spec.silicon_path).read_text())
            assert set(rec) == {"kernel", "class", "n", "inputs", "silicon",
                                "provenance"}, name
            assert rec["kernel"] == name and rec["class"] == spec.io_class
            assert rec["provenance"].startswith("HW-observed: NPU1 Phoenix silicon")
            assert len(rec["silicon"]) == spec.n_out == rec["n"]


class TestNoModelDependence:
    def test_generator_and_specs_do_not_reference_a_corpus(self):
        for fn in ("gen_vector_kernel.py", "vector_kernel_specs.py",
                   "vector_inputs.py", "capture_silicon.py"):
            text = (TOOLS / fn).read_text()
            assert "vector_ops" not in text and "golden_corpus" not in text, fn
            assert "select_records" not in text, fn

    def test_generated_headers_claim_silicon_not_model(self, tmp_path):
        for name in ("vec_srs_i32", "vec_mac_i8", "vec_conv_bf16_r12"):
            d = gen.generate(all_specs()[name], tmp_path)
            cpp = (Path(d) / "test.cpp").read_text()
            assert "aietools" not in cpp and "Half-A" not in cpp
            assert "silicon" in cpp


class TestRenderTestCpp:
    def test_bakes_designed_input_and_bootstrap_zero_exp(self):
        cpp = render_test_cpp(srs_spec())
        assert "static constexpr int N = 4;" in cpp
        assert nums(cpp, "IN") == [1, -2, 3, 40]
        assert nums(cpp, "EXP") == [0, 0, 0, 0]
        assert "int32_t *bufIn" in cpp and "int16_t *bufOut" in cpp
        assert "(int)bufOut[i]" in cpp and "(int)IN[i]" in cpp

    def test_silicon_capture_supplies_exp(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"inputs": [[1, -2, 3, 40]], "silicon": [9, 8, 7, 6]}))
        cpp = render_test_cpp(srs_spec(str(p)))
        assert nums(cpp, "IN") == [1, -2, 3, 40]
        assert nums(cpp, "EXP") == [9, 8, 7, 6]

    def test_stale_capture_inputs_are_rejected(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"inputs": [[1, 2, 3, 4]], "silicon": [0, 0, 0, 0]}))
        with pytest.raises(AssertionError):
            render_test_cpp(srs_spec(str(p)))

    def test_wrong_silicon_length_is_rejected(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"inputs": [[1, -2, 3, 40]], "silicon": [0]}))
        with pytest.raises(AssertionError):
            render_test_cpp(srs_spec(str(p)))

    def test_no_direct_inputs_and_no_capture_fails_loudly(self):
        spec = dataclasses.replace(srs_spec(), direct=None)
        with pytest.raises(AssertionError):
            render_test_cpp(spec)

    def test_no_direct_inputs_are_read_from_capture(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"inputs": [[5, 6, 7, 8]], "silicon": [1, 1, 1, 1]}))
        spec = dataclasses.replace(srs_spec(str(p)), direct=None)
        assert nums(render_test_cpp(spec), "IN") == [5, 6, 7, 8]

    def test_harness_dumps_out_txt(self):
        for name in ("vec_srs_i32", "vec_mac_i8"):
            cpp = render_test_cpp(SPECS[name])
            assert "std::ofstream" in cpp and '"out.txt"' in cpp

    def test_matmul_two_inputs_two_buffers(self):
        cpp = render_test_cpp(SPECS["vec_mac_i8"])
        assert "INA[" in cpp and "INB[" in cpp and "bo_inA" in cpp and "bo_inB" in cpp
        spec = SPECS["vec_mac_i8"]
        assert len(nums(cpp, "INA")) == 48 * 4 * 8
        assert len(nums(cpp, "INB")) == 48 * 8 * 8
        assert len(nums(cpp, "EXP")) == 48 * 4 * 8


class TestRenderOthers:
    def test_mlir(self):
        mlir = gen.render_mlir(srs_spec())
        assert "func.func private @srs_i32(memref<4xi32>, memref<4xi16>)" in mlir
        assert 'link_with = "srs.o"' in mlir and "npu1_1col" in mlir

    def test_mlir_matmul_has_both_objectfifos(self):
        mlir = gen.render_mlir(SPECS["vec_mac_i8"])
        assert "@inA" in mlir and "@inB" in mlir and "@outC" in mlir

    def test_kernel_wraps_body(self):
        cc = gen.render_kernel(srs_spec())
        assert "#define SRS_N 4" in cc
        assert "void srs_i32(int32_t *restrict in, int16_t *restrict out) {" in cc
        assert len(cc.splitlines()[0]) == 80

    def test_run_lit(self):
        lit = gen.render_run_lit(srs_spec())
        assert "REQUIRES: ryzen_ai_npu1, chess" in lit
        assert "xchesscc_wrapper aie2" in lit and "%run_on_npu1% ./test.exe" in lit

    def test_buf_kernel_type_split(self):
        assert Buf("x", "int32_t", "i32").kernel_ctype == "int32_t"
        b = Buf("in", "uint32_t", "f32", ktype="float")
        assert b.kernel_ctype == "float"
        spec = dataclasses.replace(
            srs_spec(), inputs=[b], output=Buf("out", "uint16_t", "bf16", ktype="bfloat16"))
        assert gen._signature(spec) == "float *restrict in, bfloat16 *restrict out"

    def test_generate_writes_four_files(self, tmp_path):
        d = gen.generate(srs_spec(), tmp_path)
        for fn in ("run.lit", "aie.mlir", "test.cpp", "srs.cc"):
            assert (Path(d) / fn).is_file()


class TestModeSweeps:
    def test_enum_maps(self):
        assert gen.ROUNDING_ENUM[12] == "conv_even" and gen.SAT_ENUM[3] == "symmetric"

    def test_mode_lines_only_present_axes(self):
        assert "set_saturation" not in gen.mode_lines(12, None)
        assert "set_rounding" not in gen.mode_lines(None, 1)

    def test_sweep_sizes(self):
        counts = {k: len(v.expand()) for k, v in SWEEPS.items()}
        assert counts == {"vec_srs_i32_sweep": 30, "vec_pack_i16_sweep": 3,
                          "vec_conv_bf16_sweep": 10, "vec_ups_i32_sweep": 2,
                          "vec_conv_bf16_edge_sweep": 10,
                          "vec_srs_i32_undef_rnd_sweep": 6,
                          "vec_conv_bf16_undef_rnd_sweep": 6}

    def test_point_body_matches_name(self):
        pt = {p.name: p for p in SWEEPS["vec_srs_i32_sweep"].expand()}["vec_srs_i32_r9_s3"]
        assert "rounding_mode::positive_inf" in pt.body
        assert "saturation_mode::symmetric" in pt.body

    def test_points_share_inputs(self):
        pts = SWEEPS["vec_srs_i32_sweep"].expand()
        assert all(p.direct is pts[0].direct for p in pts)


class TestDesignedInputs:
    def test_deterministic(self):
        assert vi.srs_inputs(64) == vi.srs_inputs(64)
        assert vi.conv_inputs(448, seed=1) != vi.conv_inputs(448, seed=2)

    def test_fill_rejects_oversized_edge_set(self):
        with pytest.raises(ValueError):
            vi.fill(list(range(5)), 4, lambda: 0)

    def test_srs_edges(self):
        for n in (48, 64):
            v = vi.srs_inputs(n, shift=4)
            assert len(v) == n
            assert all(vi.INT32_MIN <= x <= vi.INT32_MAX for x in v)
            for k in (0, 1, 2, -1, -2):          # tie, just below, just above
                for low in (7, 8, 9):
                    assert (k << 4 | low) in v
            for x in (0, -1, vi.INT32_MIN, vi.INT32_MAX):
                assert x in v
            assert (32767 << 4 | 8) in v and (-32769 << 4) in v   # beyond int16

    def test_conv_edges_are_normal_finite_with_ties_at_both_parities(self):
        v = vi.conv_inputs(448)
        assert len(v) == 448
        for x in v:
            if x not in (0, 0x80000000):
                assert 1 <= (x >> 23) & 0xFF <= 254, hex(x)
        for sign in (0, 0x80000000):
            for hi in (0x3F80, 0x3F81, 0x7F7F, 0x0081):
                assert (sign | hi << 16 | 0x8000) in v
                assert (sign | hi << 16 | 0x7FFF) in v
                assert (sign | hi << 16 | 0x8001) in v

    def test_pack_edges(self):
        v = vi.pack_inputs(32)
        assert len(v) == 32
        for x in (127, 128, -128, -129, 255, 256, 32767, -32768, 0):
            assert x in v
        assert all(vi.INT16_MIN <= x <= vi.INT16_MAX for x in vi.pack_inputs(96))

    def test_ups_edges(self):
        v = vi.ups_inputs(48)
        assert len(v) == 48
        for x in (vi.INT16_MAX, vi.INT16_MIN, 0, -1, 0x80, -0x81):
            assert x in v

    def test_int_matmul_ranges_and_extreme_tiles(self):
        a, b = vi.matmul_int_inputs(48, 4, 2, 8, bits=16, seed=1)
        assert len(a) == 48 * 8 and len(b) == 48 * 16
        assert set(a[:8]) == {-32768} and set(b[:16]) == {-32768}
        assert all(-32768 <= x <= 32767 for x in a + b)

    def test_bf16_matmul_finite_and_cancels(self):
        a, b = vi.matmul_bf16_inputs(24, 4, 8, 4)
        assert len(a) == 24 * 32 and len(b) == 24 * 32
        assert all(((x >> 7) & 0xFF) != 0xFF for x in a + b)
        # Tile 3: every output is a sum of x*y + (-x)*y pairs => exactly zero.
        ta, tb = a[3 * 32:4 * 32], b[3 * 32:4 * 32]
        for i in range(4):
            for j in range(4):
                for p in range(4):
                    assert ta[i * 8 + 2 * p] ^ 0x8000 == ta[i * 8 + 2 * p + 1]
                    assert tb[(2 * p) * 4 + j] == tb[(2 * p + 1) * 4 + j]


class TestDenormSpecs:
    def test_exhaustive_bf16_denormals(self):
        for name in ("vec_convexp_bf16_denorm", "vec_vfloor_bf16_denorm"):
            (ins,) = SPECS[name].direct.inputs
            assert len(ins) == 256 and 0x8001 in ins and 0x0001 in ins


class TestNewDesigns:
    def test_srs_shift_inputs_keep_ties_for_every_shift(self):
        for shift in (1, 2, 3, 5, 8, 12, 16, 24, 31):
            v = vi.srs_inputs(64, shift)
            assert len(v) == 64
            assert all(vi.INT32_MIN <= x <= vi.INT32_MAX for x in v)
            half = 1 << (shift - 1)
            for kept in (1, 2, -1, -2):
                want = kept << shift | half
                if vi.INT32_MIN <= want <= vi.INT32_MAX:
                    assert want in v, (shift, kept)

    def test_srs_shift_zero_has_no_fractional_cases(self):
        v = vi.srs_inputs(64, 0)
        assert 32767 in v and 32768 in v and -32768 in v and -32769 in v

    def test_srs_shift_sweep_names(self):
        names = set(all_specs())
        for sh in (0, 1, 2, 3, 5, 8, 12, 16, 24, 31):
            for r in (0, 3, 12):
                assert f"vec_srs_i32_sh{sh}_r{r}" in names
            assert f"vec_srs_i32_sh{sh}_r12_s0" in names
        spec = all_specs()["vec_srs_i32_sh16_r3"]
        assert "#define SRS_SHIFT 16" in gen.render_kernel(spec)
        assert "symmetric_ceil" in spec.body

    def test_undefined_modes_are_raw_casts(self):
        spec = all_specs()["vec_srs_i32_r14_s1"]
        assert "static_cast<aie::rounding_mode>(14)" in spec.body
        assert "static_cast" not in all_specs()["vec_srs_i32_r12_s1"].body

    def test_conv_edge_inputs_are_all_nonnormal(self):
        for x in vi.conv_edge_inputs(96):
            assert (x >> 23) & 0xFF in (0, 255), hex(x)
        v = vi.conv_edge_inputs(96)
        assert 0x7F80FFFF in v and 0x007FFFFF in v and 0x00018000 in v

    def test_mac_bf16_edge_tiles(self):
        a, b = vi.matmul_bf16_edge_inputs(54, 4, 8, 4)
        assert len(a) == 54 * 32 and len(b) == 54 * 32
        assert all(0 <= x <= 0xFFFF for x in a + b)
        assert set(a[:32]) == {0x7F7F}                      # max-finite tile first
        assert 0x8000 in a and 0x7F80 in a                  # -0 and +Inf present

    def test_shuffle_inputs_label_every_source_byte(self):
        w = vi.shuffle_inputs(400)
        data = b"".join((x & 0xFFFFFFFF).to_bytes(4, "little") for x in w[:32])
        assert list(data) == list(range(1, 129))
        assert not any(w[32:])

    def test_shuffle_spec_mode_table(self):
        spec = all_specs()["vec_shuffle_mode"]
        assert spec.n == 16 * 25
        for m in (48, 63, 44):
            assert str(m) in spec.body
