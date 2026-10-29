import json, os, tempfile, unittest

import capture_silicon as cap
import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS


class TestCapture(unittest.TestCase):
    def test_parse_out_txt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1\n-2\n65535\n"); p = f.name
        self.assertEqual(cap.parse_out_txt(p), [1, -2, 65535])

    def test_resolve_finds_specs_and_sweep_points(self):
        self.assertIs(cap.resolve_spec("vec_mac_i8"), SPECS["vec_mac_i8"])
        self.assertEqual(cap.resolve_spec("vec_srs_i32_r3_s1").name, "vec_srs_i32_r3_s1")
        with self.assertRaises(KeyError):
            cap.resolve_spec("nope")

    def test_record_schema_has_no_model_or_divergences(self):
        spec = cap.resolve_spec("vec_srs_i32_r0_s0")
        rec = cap.build_record(spec, [0] * spec.n_out, date="2026-01-02")
        self.assertEqual(set(rec), {"kernel", "class", "n", "inputs", "silicon", "provenance"})
        self.assertEqual(rec["inputs"], [list(spec.direct.inputs[0])])
        self.assertEqual(rec["provenance"],
                         "HW-observed: NPU1 Phoenix silicon, captured 2026-01-02 "
                         "via tools/capture_silicon.py")

    def test_rejects_wrong_length(self):
        with self.assertRaises(AssertionError):
            cap.build_record(cap.resolve_spec("vec_srs_i32_r0_s0"), [0, 1, 2])

    def test_rejects_value_outside_output_type(self):
        spec = cap.resolve_spec("vec_srs_i32_r0_s0")   # int16 output
        with self.assertRaises(AssertionError):
            cap.build_record(spec, [70000] + [0] * (spec.n_out - 1))

    def test_matmul_record_has_two_input_arrays(self):
        spec = SPECS["vec_mac_i8"]
        rec = cap.build_record(spec, [0] * spec.n_out)
        self.assertEqual(len(rec["inputs"]), 2)
        self.assertEqual(rec["n"], 48 * 32)

    def test_write_record_round_trips(self):
        spec = cap.resolve_spec("vec_srs_i32_r0_s0")
        rec = cap.build_record(spec, [1] * spec.n_out)
        with tempfile.TemporaryDirectory() as td:
            dest = cap.write_record(rec, td)
            self.assertEqual(json.load(open(dest)), rec)
