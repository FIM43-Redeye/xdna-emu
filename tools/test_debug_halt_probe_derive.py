#!/usr/bin/env python3
"""Unit tests for debug-halt-probe-derive.py parsers (real-fixture oracle)."""
import importlib.util
import pathlib
import unittest

_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "dhpd", _HERE / "debug-halt-probe-derive.py")
dhpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dhpd)


class ParseOutbufAddr(unittest.TestCase):
    def test_known_symbol(self):
        nm = (_HERE / "fixtures" / "debug_halt_probe_nm.txt").read_text()
        self.assertEqual(dhpd.parse_outbuf_addr(nm), 0x0400)

    def test_missing_symbol_is_hard_error(self):
        with self.assertRaises(dhpd.DeriveError):
            dhpd.parse_outbuf_addr("0000abcd A something_else\n")


class ParseTrapPc(unittest.TestCase):
    def test_known_bundle(self):
        objd = (_HERE / "fixtures" / "debug_halt_probe_objdump.txt").read_text()
        self.assertEqual(dhpd.parse_trap_pc(objd, outbuf_full=0x70400), 0x184)

    def test_unlocatable_bundle_is_hard_error(self):
        with self.assertRaises(dhpd.DeriveError):
            dhpd.parse_trap_pc("nothing relevant here\n", outbuf_full=0x70400)


class PcEvent0(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(dhpd.pc_event0_value(0x184), 0x80000184)


if __name__ == "__main__":
    unittest.main()
