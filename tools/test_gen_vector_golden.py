import importlib.util, os, unittest
_p = os.path.join(os.path.dirname(__file__), "gen_vector_golden.py")
_spec = importlib.util.spec_from_file_location("gvg", _p)
gvg = importlib.util.module_from_spec(_spec)
# bf16_srs_input_patterns is pure; the module only calls load_oracle() inside
# main(), so importing it is safe without VECTOR_ORACLE_MODEL.
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
        self.assertGreaterEqual(len(dens), 36, "want a dense denormal sweep, both signs")

    def test_more_nan_payloads_both_signs(self):
        nans = [p for p in self.pats if _cls(p) == "nan"]
        pos = [p for p in nans if not (p >> 31)]
        neg = [p for p in nans if (p >> 31)]
        self.assertGreaterEqual(len(pos), 8)
        self.assertGreaterEqual(len(neg), 8)
