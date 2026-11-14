"""Tests for trace_stream.py: the whole trace stream from the FFI's S2MM stream sidecar."""

import json

import pytest

import trace_stream as ts

SIZE = 4096


def _stream(n: int, seed: int = 1) -> bytes:
    """n bytes of non-zero stream words."""
    return b"".join(((seed << 24) | (i + 1)).to_bytes(4, "little") for i in range(n // 4))


def _sidecar(*buffers):
    return {"schema": 1, "buffers": [
        {"base": base, "size": size, "runs": [{"offset": o, "bytes": b.hex()} for o, b in runs]}
        for base, size, runs in buffers]}


def _raw(*parts):
    out = bytearray(SIZE)
    for o, b in parts:
        out[o:o + len(b)] = b
    return bytes(out)


class TestBuild:
    def test_a_cut_capture_gives_the_whole_stream(self):
        s = _stream(100)                       # 100 bytes staged, 64 written
        res = ts.build(_raw((0, s[:64])), _sidecar((0x1000, SIZE, [(0, s)])))
        assert res.ok, res.message
        assert res.image[:100] == s and not any(res.image[100:])

    def test_two_channels_each_cut_to_whole_bursts(self):
        a, b = _stream(96, 1), _stream(40, 2)  # 96 -> 64 written, 40 -> nothing written
        raw = _raw((0, a[:64]))
        res = ts.build(raw, _sidecar((0x1000, SIZE, [(0, a), (2048, b)])))
        assert res.ok, res.message
        assert res.image[2048:2088] == b

    def test_a_completed_run_may_end_off_a_burst_boundary(self):
        s = _stream(100)
        assert ts.build(_raw((0, s)), _sidecar((0x1000, SIZE, [(0, s)]))).ok

    def test_a_partial_burst_in_the_capture_fails(self):
        s = _stream(100)
        res = ts.build(_raw((0, s[:72])), _sidecar((0x1000, SIZE, [(0, s)])))
        assert not res.ok and "whole 64-byte bursts" in res.message

    def test_a_run_names_its_own_burst(self):
        s = _stream(200)
        side = _sidecar((0x1000, SIZE, [(0, s)]))
        side["buffers"][0]["runs"][0]["burst"] = 128
        assert ts.build(_raw((0, s[:128])), side).ok
        res = ts.build(_raw((0, s[:64])), side)
        assert not res.ok and "whole 128-byte bursts" in res.message

    def test_a_buffer_rewritten_by_a_later_run_keeps_the_earlier_runs_bytes(self):
        """A traced kernel dispatched again writes its trace buffer from the start; past the
        later run's written bursts the buffer still holds the earlier run's."""
        early, late = _stream(128, 1), _stream(96, 2)      # late: 64 written, 32 staged
        raw = _raw((0, late[:64]), (64, early[64:128]))
        side = _sidecar((0x1000, SIZE, [(0, late), (96, early[96:128])]))
        side["buffers"][0]["runs"][0]["pass"] = 1
        side["buffers"][0]["runs"][1]["pass"] = 0
        res = ts.build(raw, side)
        assert res.ok, res.message
        assert res.image[:96] == late and res.image[96:128] == early[96:128]

    def test_bytes_past_the_prefix_of_a_single_run_fail(self):
        s = _stream(128)
        raw = _raw((0, s[:64]), (100, b"\x07"))
        res = ts.build(raw, _sidecar((0x1000, SIZE, [(0, s)])))
        assert not res.ok and "not a prefix" in res.message

    def test_a_capture_that_is_not_a_prefix_fails(self):
        s = _stream(128)
        bad = bytearray(s[:64]); bad[8] ^= 0xFF
        res = ts.build(_raw((0, bytes(bad))), _sidecar((0x1000, SIZE, [(0, s)])))
        assert not res.ok and "not a prefix" in res.message

    def test_bytes_outside_every_run_fail(self):
        s = _stream(64)
        res = ts.build(_raw((0, s), (3000, b"\x07")), _sidecar((0x1000, SIZE, [(0, s)])))
        assert not res.ok and "outside every stream run" in res.message

    def test_the_trace_buffer_is_the_one_the_capture_fits(self):
        """Two buffers of the capture's size (a further kernel call's trace buffer): the capture
        picks its own."""
        mine, other = _stream(128, 1), _stream(128, 2)
        side = _sidecar((0x9000, SIZE, [(0, other)]), (0x1000, SIZE, [(0, mine)]))
        res = ts.build(_raw((0, mine[:128])), side)
        assert res.ok, res.message
        assert res.image[:128] == mine

    def test_an_empty_capture_two_different_streams_is_ambiguous(self):
        side = _sidecar((0x9000, SIZE, [(0, _stream(32, 2))]), (0x1000, SIZE, [(0, _stream(32, 1))]))
        res = ts.build(bytes(SIZE), side)
        assert not res.ok and "different streams" in res.message

    def test_no_buffer_of_the_capture_size_fails(self):
        res = ts.build(bytes(SIZE), _sidecar((0x1000, 256, [(0, _stream(32))])))
        assert not res.ok


class TestMain:
    def test_writes_the_image_and_the_check(self, tmp_path):
        s = _stream(100)
        (tmp_path / "trace_raw.bin").write_bytes(_raw((0, s[:64])))
        (tmp_path / "trace_stream.json").write_text(json.dumps(_sidecar((0x1000, SIZE, [(0, s)]))))
        assert ts.main(["trace_stream.py", str(tmp_path)]) == 0
        assert (tmp_path / "trace_stream.bin").read_bytes()[:100] == s
        assert (tmp_path / "trace_stream.check").read_text().startswith("OK ")

    def test_a_failed_check_writes_no_image(self, tmp_path):
        s = _stream(100)
        (tmp_path / "trace_raw.bin").write_bytes(_raw((0, s[:72])))
        (tmp_path / "trace_stream.json").write_text(json.dumps(_sidecar((0x1000, SIZE, [(0, s)]))))
        assert ts.main(["trace_stream.py", str(tmp_path)]) == 1
        assert not (tmp_path / "trace_stream.bin").exists()
        assert (tmp_path / "trace_stream.check").read_text().startswith("FAIL ")

    def test_missing_inputs(self, tmp_path):
        assert ts.main(["trace_stream.py", str(tmp_path)]) == 2
