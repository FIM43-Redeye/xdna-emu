"""Tests for tools/harness/suite.py: format, build/run orchestration, compare.

No toolchain, no hardware: build_package, run_package, build_ffi and build_test_host are
replaced by fakes, so these pin the runner's own rules (reuse, serial HW, abort on the
first timeout, tags never overwritten, byte-exact compare).
"""
import json
from pathlib import Path

import pytest

from harness import suite as S
from harness.build import BuildError, Package
from harness.run import PhaseResult, RunRecord, write_run_record
from harness.topology import Buffer, Core, TileRef, Topology


def make_suite(tmp_path, names=("c0", "c1", "c2"), **kw):
    for n in names:
        t = Topology(f"t_{n}", (Core("core", TileRef(0, 2), Path("kernel.s"),
                                     args=(Buffer("out", "out", 8),)),))
        S.write_case(tmp_path, n, t, files={"kernel.s": f"; {n}\n"}, meta={"tag": n})
    S.write_suite_json(tmp_path, "demo", "test", names, generator_rev="abc", **kw)
    return S.load_suite(tmp_path)


# -- format ------------------------------------------------------------------

def test_roundtrip_order_meta_and_relative_sources(tmp_path):
    suite = make_suite(tmp_path, meta={"record_bytes": 128})
    assert suite.cases == ("c0", "c1", "c2") and suite.meta == {"record_bytes": 128}
    assert suite.git_rev == "abc"
    assert suite.case_entries()[1] == {"name": "c1", "tag": "c1"}
    topo = suite.topology("c1")
    assert topo.cores[0].source == tmp_path / "cases" / "c1" / "kernel.s"
    assert topo.cores[0].source.read_text() == "; c1\n"


def test_write_case_rejects_sources_outside_the_case(tmp_path):
    t = Topology("t", (Core("core", TileRef(0, 2), Path("/abs/k.s")),))
    with pytest.raises(S.SuiteError, match="relative"):
        S.write_case(tmp_path, "x", t, files={})
    t = Topology("t", (Core("core", TileRef(0, 2), Path("k.s")),))
    with pytest.raises(S.SuiteError, match="relative"):
        S.write_case(tmp_path, "x", t, files={"other.s": ""})


def test_load_suite_rejects_malformed_dirs(tmp_path):
    with pytest.raises(S.SuiteError, match="not a suite"):
        S.load_suite(tmp_path)
    make_suite(tmp_path)
    (tmp_path / "cases" / "c1" / "topology.json").unlink()
    with pytest.raises(S.SuiteError, match="c1"):
        S.load_suite(tmp_path)


def test_unknown_hw_check_is_refused_at_write_time(tmp_path):
    with pytest.raises(S.SuiteError, match="hw_checks"):
        S.write_suite_json(tmp_path, "d", "g", [], hw_checks=["nope"])


def test_result_file_follows_case_json_result(tmp_path):
    t = Topology("t", (Core("consumer", TileRef(0, 2), Path("k.s")),))
    S.write_case(tmp_path, "a", t, files={"k.s": ""}, meta={"result": {"core": "consumer", "buffer": "out"}})
    S.write_case(tmp_path, "b", t, files={"k.s": ""})
    S.write_suite_json(tmp_path, "d", "g", ["a", "b"])
    s = S.load_suite(tmp_path)
    assert s.output("r1", "a") == tmp_path / "runs" / "r1" / "a" / "consumer.out.bin"
    assert s.output("r1", "b") == tmp_path / "runs" / "r1" / "b" / "core.out.bin"


def test_select_filters_in_suite_order_and_rejects_unknown(tmp_path):
    suite = make_suite(tmp_path)
    assert S.select(suite, ["c2", "c0"]) == ("c0", "c2")
    with pytest.raises(S.SuiteError, match="zz"):
        S.select(suite, ["zz"])


# -- build -------------------------------------------------------------------

def test_build_reuses_and_reports_failures_per_case(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    calls = []

    def fake_build(topology, out_dir, **kw):
        calls.append((topology.name, Path(out_dir), kw))
        if topology.name == "t_c1":
            raise BuildError("aiecc blew up")
        return Package(Path(out_dir), None, (), "h", reused=topology.name == "t_c2")

    monkeypatch.setattr(S, "build_package", fake_build)
    out = S.build_suite(suite, jobs=2)
    assert [(o.case, o.status) for o in out] == [("c0", "built"), ("c1", "failed"), ("c2", "reused")]
    assert "aiecc blew up" in out[1].detail
    assert all(kw == {"reuse": True} for _, _, kw in calls)
    assert {d for _, d, _ in calls} == {tmp_path / "pkgs" / c for c in ("c0", "c1", "c2")}
    assert [o.case for o in S.build_suite(suite, only=["c1"])] == ["c1"]


# -- run ---------------------------------------------------------------------

class FakeRuns:
    """run_package stand-in: writes one output per case, scripted per-case return codes."""

    def __init__(self, monkeypatch, rcs=None, fill=b"\x01" * 8):
        self.rcs, self.fill, self.calls = rcs or {}, fill, []
        monkeypatch.setattr(S, "run_package", self.run)
        monkeypatch.setattr(S, "build_test_host", lambda d: Path(d) / "test_host")
        monkeypatch.setattr(S, "build_ffi", lambda runtime: 123.0)

    def run(self, pkg, backend, tag, *, out_dir, host, **kw):
        case = Path(pkg).name
        self.calls.append((case, backend, kw))
        rc = self.rcs.get(case, 0)
        out_dir.mkdir(parents=True)
        outputs = ()
        if rc == 0:
            (out_dir / "core.out.bin").write_bytes(self.fill if not callable(self.fill) else self.fill(case))
            outputs = ("core.out.bin",)
        rec = RunRecord(tag, backend, case, "g", None, (PhaseResult(0, rc, 0.1, "log"),), outputs, ())
        write_run_record(out_dir, rec)
        return rec


def built(tmp_path, suite):
    for c in suite.cases:
        (tmp_path / "pkgs" / c).mkdir(parents=True)
        (tmp_path / "pkgs" / c / "manifest.json").write_text("{}")


def test_hw_runs_serially_in_order_and_summarises(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    fake = FakeRuns(monkeypatch)
    out = S.run_hw(suite, "r1")
    assert [c for c, *_ in fake.calls] == ["c0", "c1", "c2"] and all(o.status == "ok" for o in out)
    summary = json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())
    assert summary["backend"] == "hw" and summary["counts"] == {"ok": 3} and summary["aborted"] is None


def test_hw_first_timeout_stops_the_whole_run(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, names=("c0", "c1", "c2", "c3"))
    built(tmp_path, suite)
    fake = FakeRuns(monkeypatch, rcs={"c1": -1})
    out = S.run_hw(suite, "r1")
    assert [c for c, *_ in fake.calls] == ["c0", "c1"]
    assert [o.status for o in out] == ["ok", "timeout", "skipped", "skipped"]
    summary = json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())
    assert summary["aborted"]["case"] == "c1" and "wedged" in summary["aborted"]["reason"]
    assert summary["counts"] == {"ok": 1, "timeout": 1, "skipped": 2}


def test_hw_other_failures_are_recorded_and_the_run_continues(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    fake = FakeRuns(monkeypatch, rcs={"c0": 3})
    out = S.run_hw(suite, "r1")
    assert [o.status for o in out] == ["failed", "ok", "ok"] and len(fake.calls) == 3
    assert "3" in out[0].detail


def test_unbuilt_case_is_a_failure_not_a_crash(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, names=("c0",))
    FakeRuns(monkeypatch)
    assert S.run_hw(suite, "r1")[0].status == "failed"


def test_tags_are_never_overwritten(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    FakeRuns(monkeypatch)
    S.run_hw(suite, "r1")
    with pytest.raises(S.SuiteError, match="already holds a run"):
        S.run_hw(suite, "r1")
    with pytest.raises(S.SuiteError, match="already holds a run"):
        S.run_emu(suite, "r1")


def test_hw_check_refusal_skips_that_case_only(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, hw_checks=["aligned_labels"])
    built(tmp_path, suite)
    fake = FakeRuns(monkeypatch)
    monkeypatch.setitem(S.HW_CHECKS, "aligned_labels",
                        lambda pkg: ["label .Lx not aligned"] if pkg.name == "c1" else [])
    out = S.run_hw(suite, "r1")
    assert [o.status for o in out] == ["ok", "refused", "ok"]
    assert "c1" not in [c for c, *_ in fake.calls] and "aligned" in out[1].detail


def test_emu_rebuilds_ffi_once_then_runs_every_case_without_rebuild(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    fake = FakeRuns(monkeypatch)
    builds = []
    monkeypatch.setattr(S, "build_ffi", lambda runtime: builds.append(runtime) or 5.0)
    out = S.run_emu(suite, "e1", jobs=3, runtime="release")
    assert builds == ["release"] and len(out) == 3
    assert all(kw["rebuild_ffi"] is False and kw["runtime"] == "release" for _, _, kw in fake.calls)
    summary = json.loads((tmp_path / "runs" / "e1" / "summary.json").read_text())
    assert summary["ffi_so_mtime"] == 5.0 and summary["runtime"] == "release"


def test_emu_no_rebuild_skips_cargo(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    FakeRuns(monkeypatch)
    monkeypatch.setattr(S, "build_ffi", lambda runtime: pytest.fail("cargo must not run"))
    S.run_emu(suite, "e1", rebuild_ffi=False)


# -- repeat (determinism) ----------------------------------------------------

class RepeatRuns(FakeRuns):
    """FakeRuns whose output depends on the repeat index: `script[(case, k)]` bytes, else `fill`."""

    def __init__(self, monkeypatch, script=None, **kw):
        super().__init__(monkeypatch, **kw)
        self.script = script or {}

    def run(self, pkg, backend, tag, *, out_dir, host, **kw):
        case = Path(pkg).name
        k = int(out_dir.name.split("_")[1]) if out_dir.name.startswith("repeat_") else 1
        self.fill = self.script.get((case, k), b"\x01" * 8)
        return super().run(pkg, backend, tag, out_dir=out_dir, host=host, **kw)


def test_repeat_runs_each_case_back_to_back_and_keeps_every_repeat(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    fake = RepeatRuns(monkeypatch)
    out = S.run_hw(suite, "r1", repeat=3)
    assert [c for c, *_ in fake.calls] == ["c0"] * 3 + ["c1"] * 3 + ["c2"] * 3
    for c in suite.cases:
        d = tmp_path / "runs" / "r1" / c
        assert (d / "core.out.bin").is_file() and (d / "run.json").is_file()      # repeat 1: the plain layout
        for k in (2, 3):
            assert (d / f"repeat_{k}" / "core.out.bin").is_file()
    assert [(o.status, o.agree) for o in out] == [("ok", True)] * 3
    summary = json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())
    assert summary["repeat"] == 3 and summary["cases"]["c1"]["agree"] is True


def test_no_repeat_leaves_the_layout_and_agreement_unset(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    fake = RepeatRuns(monkeypatch)
    out = S.run_hw(suite, "r1")
    assert len(fake.calls) == 3 and all(o.agree is None for o in out)
    assert not list((tmp_path / "runs" / "r1").glob("*/repeat_*"))
    summary = json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())
    assert summary["repeat"] == 1 and "agree" not in summary["cases"]["c0"]


def test_repeat_flags_the_case_whose_repeats_differ_and_still_runs_the_rest(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    RepeatRuns(monkeypatch, script={("c1", 2): b"\x01\x01\x07\x01\x01\x01\x01\x01"})
    out = S.run_hw(suite, "r1", repeat=3)
    assert [(o.case, o.status, o.agree) for o in out] == [
        ("c0", "ok", True), ("c1", "ok", False), ("c2", "ok", True)]
    assert "c1: repeats DIFFER" in S.format_outcomes(out, "hw r1")


def test_repeat_report_names_the_differing_bytes_and_the_settling_pattern(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    # c1: the first run differs, runs 2-3 agree (state left by the previous kernel)
    RepeatRuns(monkeypatch, script={("c1", 1): b"\x09\x01\x01\x01\x01\x01\x01\x08"})
    S.run_hw(suite, "r1", repeat=3)
    rep = {r.case: r for r in S.repeat_report(suite, "r1")}
    assert rep["c0"].agree and rep["c0"].repeats == 3 and not rep["c0"].diffs
    r = rep["c1"]
    assert not r.agree and r.first_run_only
    assert {(d.repeat, d.file, d.offset, d.ref, d.got) for d in r.diffs} == {
        (k, "core.out.bin", off, ref, 1) for k in (2, 3) for off, ref in ((0, 9), (7, 8))}
    text = S.format_repeat_report(list(rep.values()), "r1")
    assert "c1" in text and "first run only" in text and "0x00: 09 -> 01" in text
    assert "2/3 cases agree" in text or "2 of 3" in text


def test_repeat_report_distinguishes_a_drifting_case(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, names=("c0",))
    built(tmp_path, suite)
    RepeatRuns(monkeypatch, script={("c0", 2): b"\x02" * 8, ("c0", 3): b"\x03" * 8})
    S.run_hw(suite, "r1", repeat=3)
    (r,) = S.repeat_report(suite, "r1")
    assert not r.agree and not r.first_run_only


def test_repeat_timeout_aborts_the_run(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, names=("c0", "c1", "c2"))
    built(tmp_path, suite)

    class Timeout(RepeatRuns):
        def run(self, pkg, backend, tag, *, out_dir, host, **kw):
            if Path(pkg).name == "c1" and out_dir.name == "repeat_2":
                self.rcs = {"c1": -1}
            return super().run(pkg, backend, tag, out_dir=out_dir, host=host, **kw)

    Timeout(monkeypatch)
    out = S.run_hw(suite, "r1", repeat=3)
    assert [o.status for o in out] == ["ok", "timeout", "skipped"]
    assert "repeat 2" in out[1].detail


def test_repeat_failure_other_than_timeout_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, names=("c0", "c1"))
    built(tmp_path, suite)

    class Fail(RepeatRuns):
        def run(self, pkg, backend, tag, *, out_dir, host, **kw):
            self.rcs = {"c0": 2} if out_dir.name == "repeat_2" else {}
            return super().run(pkg, backend, tag, out_dir=out_dir, host=host, **kw)

    Fail(monkeypatch)
    out = S.run_hw(suite, "r1", repeat=2)
    assert [o.status for o in out] == ["failed", "ok"] and "repeat 2" in out[0].detail


def test_repeat_must_be_positive(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    with pytest.raises(S.SuiteError, match="repeat"):
        S.run_hw(suite, "r1", repeat=0)


# -- compare -----------------------------------------------------------------

def run_two(tmp_path, monkeypatch, fill_b, rcs_b=None):
    suite = make_suite(tmp_path)
    built(tmp_path, suite)
    FakeRuns(monkeypatch)
    S.run_hw(suite, "r1")
    FakeRuns(monkeypatch, rcs=rcs_b, fill=fill_b)
    S.run_emu(suite, "e1", rebuild_ffi=False)
    return suite


def test_compare_identical(tmp_path, monkeypatch):
    suite = run_two(tmp_path, monkeypatch, b"\x01" * 8)
    res = S.compare_runs(suite, "r1", "e1")
    assert all(r.identical for r in res)
    assert "3/3 identical" in S.format_comparison(res, "r1", "e1")


def test_compare_reports_byte_counts_and_first_offset(tmp_path, monkeypatch):
    suite = run_two(tmp_path, monkeypatch,
                    lambda case: b"\x01" * 3 + b"\x02" + b"\x01" * 2 + b"\x09\x09" if case == "c1" else b"\x01" * 8)
    res = {r.case: r for r in S.compare_runs(suite, "r1", "e1")}
    assert res["c0"].identical and res["c2"].identical
    assert res["c1"].status == "differs" and "3 byte(s) differ, first at offset 3" in res["c1"].detail


def test_compare_never_calls_a_failed_case_identical(tmp_path, monkeypatch):
    suite = run_two(tmp_path, monkeypatch, b"\x01" * 8, rcs_b={"c2": 1})
    res = {r.case: r for r in S.compare_runs(suite, "r1", "e1")}
    assert res["c2"].status == "unusable" and "e1" in res["c2"].detail


def test_compare_unknown_tag(tmp_path, monkeypatch):
    suite = run_two(tmp_path, monkeypatch, b"\x01" * 8)
    with pytest.raises(S.SuiteError, match="zz"):
        S.compare_runs(suite, "r1", "zz")


# -- hardware pre-flight -------------------------------------------------------

def test_unaligned_labels_found_in_a_disassembly():
    dump = ("00000000 <test_kernel>:\n   0: nop\n00000030 <.Lgood>:\n   30: nop\n"
            "00000044 <.Lbad>:\n   44: nop\n")
    assert S.unaligned_labels(dump) == [".Lbad@0x44"]
    assert S.unaligned_labels("") == []


def test_single_core_case_is_in_out_with_core_out_result(tmp_path):
    S.write_single_core_case(tmp_path, "k0", "; asm\n", input_bytes=b"\x07" * 16, out_bytes=32,
                             meta={"points": [1, 2]})
    S.write_suite_json(tmp_path, "d", "g", ["k0"])
    s = S.load_suite(tmp_path)
    t = s.topology("k0")
    (core,) = t.cores
    assert (core.name, core.tile) == ("core", TileRef(0, 2))
    assert [(b.name, b.direction, b.size_bytes, b.data) for b in core.args] == [
        ("in", "in", 16, b"\x07" * 16), ("out", "out", 32, None)]
    assert core.source.read_text() == "; asm\n"
    assert s.case_entries() == [{"name": "k0", "points": [1, 2]}]


def test_meta_entries_and_core_object(tmp_path):
    suite = make_suite(tmp_path, meta={"record_bytes": 128})
    assert suite.meta["record_bytes"] == 128
    assert [k["name"] for k in suite.case_entries()] == ["c0", "c1", "c2"]
    core = suite.topology("c1").cores[0]
    assert core.source == tmp_path / "cases" / "c1" / "kernel.s"
    assert suite.core_object("c1", core) == tmp_path / "pkgs" / "c1" / "phase_0" / "core.o"
