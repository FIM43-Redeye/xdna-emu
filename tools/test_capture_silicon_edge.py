import json, os, tempfile, unittest
import capture_silicon_edge as cap
import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS

GP = os.path.join(os.path.dirname(gen.__file__), "golden", "vector_ops.json")
GOLDEN = json.loads(open(GP).read())

class TestCapture(unittest.TestCase):
    def test_parse_out_txt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1\n-2\n65535\n"); p = f.name
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
