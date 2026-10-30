"""Tests for vv_data.py: packing, run.lit host knobs, and the staged data files."""

import struct

import pytest

import vv_data


def test_pack_is_little_endian_and_typed():
    assert vv_data.pack([1, -2], "int16_t") == struct.pack("<hh", 1, -2)
    assert vv_data.pack([0xDEADBEEF], "uint32_t") == bytes.fromhex("efbeadde")
    assert vv_data.pack([-1, 127], "int8_t") == bytes([0xFF, 0x7F])
    assert vv_data.pack([], "int32_t") == b""


@pytest.mark.parametrize("ctype", sorted(vv_data.CTYPES))
def test_unpack_inverts_pack_at_the_type_limits(ctype):
    bits = 8 * vv_data.CTYPES[ctype][1]
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if ctype.startswith("int") \
        else (0, (1 << bits) - 1)
    vals = [lo, -1 if lo < 0 else 0, 0, 1, hi]
    assert vv_data.unpack(vv_data.pack(vals, ctype), ctype) == vals


def test_pack_rejects_values_outside_the_type():
    with pytest.raises(struct.error):
        vv_data.pack([1 << 16], "uint16_t")
    with pytest.raises(struct.error):
        vv_data.pack([-1], "uint32_t")


def test_unpack_rejects_partial_elements():
    with pytest.raises(AssertionError):
        vv_data.unpack(b"\x00\x00\x00", "int16_t")


def test_host_flags_quote_the_data_dir_for_bash():
    flags = vv_data.host_flags(2, "int32_t", 1536)
    assert flags == "-DVV_DATA_DIR='\"%S\"' -DVV_NIN=2 -DVV_OUT_T=int32_t -DVV_NOUT=1536"


def test_host_flags_reject_unknown_output_type():
    with pytest.raises(AssertionError):
        vv_data.host_flags(1, "float", 4)


def test_write_data_writes_inputs_and_expected(tmp_path):
    vv_data.write_data(tmp_path, vv_data.KernelData([b"ab", b"cd"], b"xyz"))
    assert (tmp_path / "in0.bin").read_bytes() == b"ab"
    assert (tmp_path / "in1.bin").read_bytes() == b"cd"
    assert (tmp_path / "exp.bin").read_bytes() == b"xyz"


def test_uncaptured_kernel_gets_no_expected_and_loses_a_stale_one(tmp_path):
    (tmp_path / "exp.bin").write_bytes(b"stale")
    vv_data.write_data(tmp_path, vv_data.KernelData([b"ab"], None))
    assert (tmp_path / "in0.bin").exists()
    assert not (tmp_path / "exp.bin").exists()
