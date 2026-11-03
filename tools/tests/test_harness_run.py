"""Pure parts of tools/harness/run.py: input packing, output unpacking, run record.

No hardware, no toolchain: the plan comes from the real planner on small
topologies, the bytes are checked against the test_host LCG written out by hand.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from harness.plan import plan_channels
from harness.run import (RunError, RunRecord, PhaseResult, buffer_input, input_sources,
                         pack_phase_inputs, prng_bytes, unpack_phase_outputs, write_run_record,
                         read_run_record)
from harness.topology import Buffer, Core, TileRef, Topology


def lcg(n, seed):
    # test_host fill_prng: state = (state*1103515245+12345) & 0x7FFFFFFF; byte = (state>>16)&0xFF
    out, s = bytearray(), seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((s >> 16) & 0xFF)
    return bytes(out)


def core(name, col, row, *bufs):
    return Core(name, TileRef(col, row), Path(f"{name}.s"), args=tuple(bufs))


def phase_dicts(topology):
    plan = plan_channels(topology)
    assert not hasattr(plan, "code"), plan
    return dataclasses.asdict(plan)["phases"]


# -- PRNG --------------------------------------------------------------------

def test_prng_matches_test_host_lcg():
    assert prng_bytes(37, 42) == lcg(37, 42)
    assert prng_bytes(0, 1) == b""


def test_prng_seed_changes_stream():
    assert prng_bytes(16, 1) != prng_bytes(16, 2)


# -- buffer input definition -------------------------------------------------

def test_explicit_data_is_used_verbatim():
    t = Topology("t", (core("a", 0, 2, Buffer("i", "in", 8, data=b"\x01" * 8)),), seed=5)
    data, source = buffer_input(t, t.cores[0], t.cores[0].args[0])
    assert data == b"\x01" * 8 and source == {"kind": "data"}


def test_missing_data_is_seeded_per_buffer():
    t = Topology("t", (core("a", 0, 2, Buffer("i", "in", 16), Buffer("j", "in", 16)),
                       core("b", 0, 3, Buffer("i", "in", 16))), seed=5)
    got = {}
    for c in t.cores:
        for b in c.args:
            data, source = buffer_input(t, c, b)
            assert source["kind"] == "prng" and data == prng_bytes(16, source["seed"])
            got[(c.name, b.name)] = (data, source["seed"])
    assert len({v[1] for v in got.values()}) == 3        # distinct streams
    again, src = buffer_input(t, t.cores[0], t.cores[0].args[0])
    assert (again, src["seed"]) == got[("a", "i")]       # deterministic
    other = dataclasses.replace(t, seed=6)
    assert buffer_input(other, other.cores[0], other.cores[0].args[0])[1]["seed"] != src["seed"]


def test_wrong_data_length_rejected():
    t = Topology("t", (core("a", 0, 2, Buffer("i", "in", 8, data=b"\x01" * 4)),))
    with pytest.raises(RunError, match="4 bytes.*8"):
        buffer_input(t, t.cores[0], t.cores[0].args[0])


def test_input_sources_lists_inputs_only():
    t = Topology("t", (core("a", 0, 2, Buffer("i", "in", 8, data=b"\x01" * 8),
                            Buffer("o", "out", 8)),))
    s = input_sources(t)
    assert [(e["core"], e["buffer"], e["source"]["kind"]) for e in s] == [("a", "i", "data")]


# -- pack / unpack -----------------------------------------------------------

def two_in_two_out():
    return Topology("t", (
        core("a", 0, 2, Buffer("i", "in", 16, data=bytes(range(16))), Buffer("o", "out", 32)),
        core("b", 1, 2, Buffer("i", "in", 8, data=b"\xaa" * 8), Buffer("o", "out", 4))),
        device="auto", seed=3)


def test_pack_places_each_input_at_its_route_offset():
    t = two_in_two_out()
    ph = phase_dicts(t)[0]
    packed = pack_phase_inputs(t, ph)
    assert len(packed) == ph["host_in_bytes"]
    for r in ph["routes"]:
        if r["direction"] == "in":
            want = bytes(range(16)) if r["core"] == "a" else b"\xaa" * 8
            assert packed[r["host_offset"]:r["host_offset"] + r["size_bytes"]] == want


def test_pack_pads_empty_phase_input_to_one_word():
    t = Topology("t", (core("a", 0, 2, Buffer("o", "out", 8)),))
    ph = phase_dicts(t)[0]
    assert ph["host_in_bytes"] == 0
    assert pack_phase_inputs(t, ph) == b"\0" * 4


def test_unpack_inverts_offsets():
    t = two_in_two_out()
    ph = phase_dicts(t)[0]
    out = bytearray(max(ph["host_out_bytes"], 4))
    for r in ph["routes"]:
        if r["direction"] == "out":
            out[r["host_offset"]:r["host_offset"] + r["size_bytes"]] = \
                bytes([0x10 + len(r["core"])]) * r["size_bytes"]
    got = unpack_phase_outputs(ph, bytes(out))
    assert set(got) == {("a", "o"), ("b", "o")}
    assert got[("a", "o")] == b"\x11" * 32 and got[("b", "o")] == b"\x11" * 4


def test_unpack_rejects_short_output():
    t = two_in_two_out()
    ph = phase_dicts(t)[0]
    with pytest.raises(RunError, match="short"):
        unpack_phase_outputs(ph, b"\0" * (ph["host_out_bytes"] - 4))


# -- run record --------------------------------------------------------------

def test_run_record_roundtrip(tmp_path):
    rec = RunRecord(tag="r1", backend="emu", topology="t", git_rev="abc", ffi_so_mtime=12.5,
                    phases=(PhaseResult(0, 0, 1.25, "phase_0.log"),
                            PhaseResult(1, 1, 2.0, "phase_1.log")),
                    outputs=("a.o.bin",), inputs=tuple(input_sources(two_in_two_out())))
    write_run_record(tmp_path, rec)
    assert read_run_record(tmp_path) == rec
    assert json.loads((tmp_path / "run.json").read_text())["backend"] == "emu"
    assert not rec.ok


def test_run_record_ok_when_all_phases_return_zero():
    rec = RunRecord("r", "hw", "t", "g", None, (PhaseResult(0, 0, 0.1, "l"),), (), ())
    assert rec.ok


def test_run_record_timed_out_is_a_timeout_phase_not_any_failure():
    ok = PhaseResult(0, 0, 0.1, "l")
    assert not RunRecord("r", "hw", "t", "g", None, (ok,), (), ()).timed_out
    assert not RunRecord("r", "hw", "t", "g", None, (PhaseResult(0, 3, 0.1, "l"),), (), ()).timed_out
    assert RunRecord("r", "hw", "t", "g", None, (PhaseResult(0, -1, 60.0, "l"),), (), ()).timed_out


def test_run_record_package_hash_roundtrips_and_old_records_still_read(tmp_path):
    rec = RunRecord("r", "hw", "t", "g", None, (PhaseResult(0, 0, 0.1, "l"),), ("a.b.bin",), (),
                    package_hash="abc")
    write_run_record(tmp_path, rec)
    assert read_run_record(tmp_path).package_hash == "abc"
    d = json.loads((tmp_path / "run.json").read_text())
    del d["package_hash"]
    (tmp_path / "run.json").write_text(json.dumps(d))
    assert read_run_record(tmp_path).package_hash == ""
