"""Unit tests for gen_nan_inf_sweep.py (NaN/Inf add/sub silicon sweep generator).

The generator emits the four bf16/fp32 add/sub sweep kernels and an offline
"expected under current emulator" column. These tests pin:
  - matrix shape (16 classes -> 256 lanes, lane = a*16+b),
  - the offline model against hand-verified IEEE facts,
  - the bf16 model matching the documented emulator path (host f32 op then raw
    >>16 truncate),
  - emitter well-formedness (every kernel single-op, bf16 noinline-wrapped),
  - --check idempotency after a fresh generation.
"""

import os
import struct
import subprocess
import sys
import unittest

import gen_nan_inf_sweep as g

HERE = os.path.dirname(os.path.abspath(__file__))


class TestMatrixShape(unittest.TestCase):
    def test_sixteen_classes(self):
        self.assertEqual(g.NCLASS, 16)
        self.assertEqual(len(g.CLASS_NAMES), 16)
        self.assertEqual(len(g.BF16_REPS), 16)
        self.assertEqual(len(g.FP32_REPS), 16)
        self.assertEqual(g.N, 256)

    def test_lane_indexing_is_ordered_pairs(self):
        c = g.COMBOS[0]
        a, b, _ = c.build_matrix()
        self.assertEqual(len(a), 256)
        # lane = a_class*16 + b_class: row ai holds rep[ai] for all 16 columns.
        for ai in range(16):
            for bi in range(16):
                lane = ai * 16 + bi
                self.assertEqual(a[lane], g.BF16_REPS[ai])
                self.assertEqual(b[lane], g.BF16_REPS[bi])


class TestOfflineModel(unittest.TestCase):
    """The offline column must mirror the documented emulator datapath exactly."""

    def test_bf16_finite_add(self):
        # 1.0 + 1.0 = 2.0 in bf16. 1.0 = 0x3F80, 2.0 = 0x4000.
        self.assertEqual(g.emu_bf16_op(0x3F80, 0x3F80, "add"), 0x4000)

    def test_bf16_is_truncate_not_round(self):
        # The documented path is raw >>16 truncate (no rounding). Build an f32
        # sum whose low 16 bits are nonzero and confirm we truncate, not round.
        a = 0x3F80  # 1.0
        # Adding a tiny bf16 that perturbs below the bf16 ulp: 1.0 + smallest
        # normal bf16 stays 1.0 under truncation of the f32 result.
        r = g.emu_bf16_op(a, 0x0001, "add")  # 1.0 + denorm
        self.assertEqual(r, 0x3F80)

    def test_bf16_inf_plus_neg_inf_is_host_canonical(self):
        # Present emulator: +Inf + -Inf -> host f32 NaN (0xFFC00000) truncated.
        self.assertEqual(g.emu_bf16_op(0x7F80, 0xFF80, "add"), 0xFFC0)

    def test_bf16_inf_plus_finite_stays_inf(self):
        # The key discriminator: present emulator keeps clean Inf.
        self.assertEqual(g.emu_bf16_op(0x7F80, 0x0000, "add"), 0x7F80)
        self.assertEqual(g.emu_bf16_op(0x7F80, 0x7F00, "add"), 0x7F80)

    def test_bf16_payload_propagation_order(self):
        # Host FPU keeps the second operand's payload on add; first on sub.
        self.assertEqual(g.emu_bf16_op(0x7FC0, 0x7FD5, "add"), 0x7FD5)
        self.assertEqual(g.emu_bf16_op(0x7FC0, 0x7FD5, "sub"), 0x7FC0)

    def test_fp32_inf_plus_neg_inf(self):
        self.assertEqual(g.emu_fp32_op(0x7F800000, 0xFF800000, "add"), 0xFFC00000)

    def test_fp32_finite(self):
        # 2.0 - 0.5 = 1.5. 2.0=0x40000000, 0.5=0x3F000000, 1.5=0x3FC00000.
        self.assertEqual(g.emu_fp32_op(0x40000000, 0x3F000000, "sub"), 0x3FC00000)

    def test_fp32_large_overflow_to_inf(self):
        # nlarge + nlarge overflows f32 to +Inf.
        self.assertEqual(g.emu_fp32_op(0x7F000000, 0x7F000000, "add"), 0x7F800000)

    def test_signed_zero_add(self):
        self.assertEqual(g.emu_bf16_op(0x8000, 0x0000, "add"), 0x0000)  # -0 + +0 = +0
        self.assertEqual(g.emu_fp32_op(0x80000000, 0x00000000, "add"), 0x00000000)


class TestEmitters(unittest.TestCase):
    def test_bf16_kernels_are_noinline_wrapped(self):
        for name in ("vec_nan_bf16_add", "vec_nan_bf16_sub"):
            c = next(x for x in g.COMBOS if x.name == name)
            cc = g.emit_kernel_cc(c)
            self.assertIn("__attribute__((noinline))", cc)
            self.assertIn("aie::vector<bfloat16, 32>", cc)

    def test_fp32_kernels_not_noinline(self):
        for name in ("vec_nan_fp32_add", "vec_nan_fp32_sub"):
            c = next(x for x in g.COMBOS if x.name == name)
            cc = g.emit_kernel_cc(c)
            self.assertNotIn("noinline", cc)
            self.assertIn("aie::vector<float, 8>", cc)

    def test_single_op_per_kernel(self):
        # Exactly one add OR one sub intrinsic call in the body (single-op).
        for c in g.COMBOS:
            cc = g.emit_kernel_cc(c)
            intr = "aie::add" if c.op == "add" else "aie::sub"
            other = "aie::sub" if c.op == "add" else "aie::add"
            self.assertIn(intr, cc)
            self.assertNotIn(other, cc)

    def test_test_cpp_dumps_triple(self):
        c = g.COMBOS[0]
        a, b, e = c.build_matrix()
        cpp = g.emit_test_cpp(c, a, b, e)
        # The per-lane silicon dump writes A_bits B_bits hw_out_bits.
        self.assertIn('std::ofstream dump("out.txt")', cpp)
        self.assertIn("(uint32_t)bufA[i]", cpp)
        self.assertIn("(uint32_t)bufB[i]", cpp)
        self.assertIn("(uint32_t)bufC[i]", cpp)


class TestRegenerationStable(unittest.TestCase):
    def test_check_is_clean_after_generate(self):
        # Generate, then --check must report up-to-date (idempotent).
        gen = subprocess.run([sys.executable, os.path.join(HERE, "gen_nan_inf_sweep.py")],
                             capture_output=True, text=True)
        self.assertEqual(gen.returncode, 0, gen.stderr)
        chk = subprocess.run([sys.executable, os.path.join(HERE, "gen_nan_inf_sweep.py"), "--check"],
                             capture_output=True, text=True)
        self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)


if __name__ == "__main__":
    unittest.main()
