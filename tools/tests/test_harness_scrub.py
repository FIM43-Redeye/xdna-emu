"""The scrub seam of the harness (tools/harness/scrub.py, run.py, suite.py).

No hardware and no aiecc: subprocess, build_package and run_package are faked where a test
would otherwise launch something, so these pin the rules: the scrub runs before every phase
on silicon, a missing marker stops the run, the emulator gets XDNA_EMU_CORE_INIT, records and
summaries name the pattern, and --no-scrub turns all of it off.
"""
import json
import struct
import subprocess
from pathlib import Path

import pytest

from harness import run as R
from harness import scrub as SC
from harness import suite as S
from harness.build import build_package
from harness.topology import Buffer, Core, TileRef, Topology, to_dict


KERNEL = SC.load_scrub_kernel()


# -- the artifact ----------------------------------------------------------------

def test_artifact_names_the_pattern_and_carries_table_and_kernel():
    assert KERNEL.name == f"scrub-v{KERNEL.version}"
    assert len(KERNEL.table) % 32 == 0 and len(set(KERNEL.table)) > 200     # not a constant
    assert "test_kernel:" in KERNEL.kernel_s and "ret lr" in KERNEL.kernel_s
    assert KERNEL.out_marker & 0xFF == KERNEL.version


# -- topology ----------------------------------------------------------------

def test_case_tiles_are_unique_and_sorted():
    t = Topology("t", tuple(Core(n, TileRef(c, r), Path("k.s")) for n, c, r in
                            (("a", 1, 3), ("b", 0, 2), ("c", 1, 3), ("d", 0, 5))))
    assert SC.case_tiles(t) == (TileRef(0, 2), TileRef(0, 5), TileRef(1, 3))


def test_scrub_topology_has_one_core_per_tile_with_the_table_in_and_a_marker_out():
    tiles = (TileRef(0, 2), TileRef(0, 3))
    t = SC.scrub_topology(KERNEL, "npu1", tiles)
    assert [c.tile for c in t.cores] == list(tiles)
    for c in t.cores:
        (i, o) = c.args
        assert (i.direction, i.data) == ("in", KERNEL.table)
        assert (o.direction, o.size_bytes) == ("out", SC.MARKER_BYTES)
    with pytest.raises(ValueError):
        SC.scrub_topology(KERNEL, "npu1", ())


@pytest.mark.parametrize("tiles", [
    [(0, 2)],
    [(0, 2), (0, 3), (0, 4), (0, 5)],
    [(c, r) for c in range(4) for r in range(2, 6)],      # every compute tile of npu1
])
def test_scrub_topologies_plan_and_emit_as_one_launch(tmp_path, tiles):
    t = SC.write_scrub_case(KERNEL, tmp_path / "case", "npu1", [TileRef(*x) for x in tiles])
    pkg = build_package(t, tmp_path / "pkg", run_aiecc=False)
    assert len(pkg.phase_dirs) == 1


def test_marker_problems_names_cores_without_the_marker():
    good = struct.pack("<I", KERNEL.out_marker)
    outs = {("scrub_c0r2", "out"): good, ("scrub_c0r3", "out"): b"\0\0\0\0"}
    (p,) = SC.marker_problems(outs, KERNEL.out_marker)
    assert p.startswith("scrub_c0r3")


# -- run.py --------------------------------------------------------------------

def test_emu_env_carries_the_pattern_only_when_scrubbing(monkeypatch):
    monkeypatch.setenv("XDNA_EMU_CORE_INIT", "stale")
    sc = SC.Scrub(KERNEL.name, KERNEL.out_marker)
    assert R._env("emu", "debug", sc)["XDNA_EMU_CORE_INIT"] == KERNEL.name
    assert "XDNA_EMU_CORE_INIT" not in R._env("emu", "debug", None)
    assert "XDNA_EMU_CORE_INIT" not in R._env("hw", "debug", sc)


def fake_package(d: Path, topology: Topology, phases: int) -> Path:
    """A manifest run_package accepts: `phases` phases, one 4-byte output per core each."""
    d.mkdir(parents=True)
    plan = {"phases": [{"index": i, "host_in_bytes": 0, "host_out_bytes": 4 * len(topology.cores),
                        "routes": [{"core": c.name, "buffer": "out", "direction": "out",
                                    "host_offset": 4 * k, "size_bytes": 4}
                                   for k, c in enumerate(topology.cores)]}
                       for i in range(phases)]}
    (d / "manifest.json").write_text(json.dumps({"topology": to_dict(topology), "plan": plan}))
    return d


class FakeHost:
    """subprocess.run stand-in: records which xclbin ran, writes `out` bytes to --out-file."""

    def __init__(self, monkeypatch, scrub_out=None):
        self.calls, self.scrub_out = [], scrub_out
        monkeypatch.setattr(R.subprocess, "run", self.run)

    def run(self, cmd, env=None, **kw):
        if "-x" not in cmd:                      # git rev-parse and friends
            return subprocess.CompletedProcess(cmd, 0, "rev\n", "")
        xclbin = Path(cmd[cmd.index("-x") + 1])
        out = Path(cmd[cmd.index("--out-file") + 1])
        size = int(cmd[cmd.index("--out-size") + 1])
        is_scrub = "scrubpkg" in xclbin.parts
        self.calls.append(("scrub" if is_scrub else xclbin.parent.name, env.get("XDNA_EMU_CORE_INIT")))
        if is_scrub:
            out.write_bytes(self.scrub_out if self.scrub_out is not None
                            else struct.pack("<I", KERNEL.out_marker) * (size // 4))
        else:
            out.write_bytes(b"\x07" * size)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")


def case_topology():
    return Topology("case", (Core("core", TileRef(0, 2), Path("k.s"), args=(Buffer("out", "out", 4),)),))


def scrub_pkg(tmp_path):
    return fake_package(tmp_path / "scrubpkg", SC.scrub_topology(KERNEL, "npu1", (TileRef(0, 2),)), 1)


def test_hw_scrub_runs_before_every_phase_and_is_recorded(tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg", case_topology(), 2)
    host = FakeHost(monkeypatch)
    sc = SC.Scrub(KERNEL.name, KERNEL.out_marker, scrub_pkg(tmp_path))
    rec = R.run_package(pkg, "hw", "r1", host=tmp_path / "host", scrub=sc)
    assert [c for c, _ in host.calls] == ["scrub", "phase_0", "scrub", "phase_1"]
    assert rec.ok and rec.scrub == KERNEL.name
    assert (pkg / "r1" / "phase_1.scrub.log").read_text().startswith(KERNEL.name)
    assert R.read_run_record(pkg / "r1").scrub == KERNEL.name


def test_hw_scrub_without_its_marker_stops_the_run(tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg", case_topology(), 2)
    host = FakeHost(monkeypatch, scrub_out=b"\0\0\0\0")
    sc = SC.Scrub(KERNEL.name, KERNEL.out_marker, scrub_pkg(tmp_path))
    rec = R.run_package(pkg, "hw", "r1", host=tmp_path / "host", scrub=sc)
    assert [c for c, _ in host.calls] == ["scrub"]
    assert not rec.ok and rec.phases[0].returncode == R.SCRUB_MARKER_RC
    assert "marker" in (pkg / "r1" / rec.phases[0].log).read_text()


def test_hw_scrub_needs_its_package(tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg", case_topology(), 1)
    FakeHost(monkeypatch)
    with pytest.raises(R.RunError, match="scrub package"):
        R.run_package(pkg, "hw", "r1", host=tmp_path / "host", scrub=SC.Scrub(KERNEL.name, 1))


def test_emu_scrub_sets_core_init_and_launches_no_scrub_kernel(tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg", case_topology(), 2)
    host = FakeHost(monkeypatch)
    monkeypatch.setattr(R, "ffi_so_path", lambda runtime: tmp_path)
    rec = R.run_package(pkg, "emu", "e1", host=tmp_path / "host", rebuild_ffi=False,
                        scrub=SC.Scrub(KERNEL.name, KERNEL.out_marker))
    assert host.calls == [("phase_0", KERNEL.name), ("phase_1", KERNEL.name)]
    assert rec.scrub == KERNEL.name


def test_unscrubbed_run_records_no_pattern(tmp_path, monkeypatch):
    pkg = fake_package(tmp_path / "pkg", case_topology(), 1)
    host = FakeHost(monkeypatch)
    rec = R.run_package(pkg, "hw", "r1", host=tmp_path / "host")
    assert host.calls == [("phase_0", None)] and rec.scrub == ""


def test_old_run_records_without_a_scrub_field_still_read(tmp_path):
    rec = R.RunRecord("r1", "hw", "t", "g", None, (R.PhaseResult(0, 0, 0.1, "log"),), (), ())
    d = json.loads(json.dumps(__import__("dataclasses").asdict(rec)))
    del d["scrub"]
    (tmp_path / "run.json").write_text(json.dumps(d))
    assert R.read_run_record(tmp_path).scrub == ""


# -- suite.py ------------------------------------------------------------------

def make_suite(tmp_path, tiles_per_case):
    names = []
    for n, tiles in tiles_per_case.items():
        t = Topology(n, tuple(Core(f"k{i}", TileRef(*x), Path("kernel.s"), args=(Buffer("out", "out", 8),))
                              for i, x in enumerate(tiles)))
        S.write_case(tmp_path, n, t, files={"kernel.s": "; x\n"})
        (tmp_path / "pkgs" / n).mkdir(parents=True)
        (tmp_path / "pkgs" / n / "manifest.json").write_text("{}")
        names.append(n)
    S.write_suite_json(tmp_path, "demo", "test", names, generator_rev="abc")
    return S.load_suite(tmp_path)


class SuiteFakes:
    def __init__(self, monkeypatch):
        self.runs, self.builds = [], []
        monkeypatch.setattr(S, "build_test_host", lambda d: Path(d) / "test_host")
        monkeypatch.setattr(S, "build_ffi", lambda runtime: 1.0)
        monkeypatch.setattr(S, "build_package", self.build)
        monkeypatch.setattr(S, "run_package", self.run)

    def build(self, topology, out_dir, **kw):
        self.builds.append((tuple((c.tile.col, c.tile.row) for c in topology.cores), kw))
        return type("P", (), {"out_dir": Path(out_dir), "reused": False})()

    def run(self, pkg, backend, tag, *, out_dir, host, scrub=None, **kw):
        self.runs.append((Path(pkg).name, backend, scrub))
        out_dir.mkdir(parents=True)
        rec = R.RunRecord(tag, backend, "t", "g", None, (R.PhaseResult(0, 0, 0.1, "log"),), (), (),
                          scrub=scrub.name if scrub else "")
        R.write_run_record(out_dir, rec)
        return rec


def test_hw_builds_one_scrub_package_per_tile_set_and_passes_it_to_each_case(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, {"a": [(0, 2)], "b": [(0, 2)], "c": [(0, 3), (0, 2)]})
    f = SuiteFakes(monkeypatch)
    out = S.run_hw(suite, "r1")
    assert all(o.status == "ok" for o in out)
    assert [b[0] for b in f.builds] == [((0, 2),), ((0, 2), (0, 3))]
    assert all(kw == {"reuse": True} for _, kw in f.builds)
    pkgs = [s.pkg_dir.parent.name for _, _, s in f.runs]
    assert pkgs == ["npu1_c0r2", "npu1_c0r2", "npu1_c0r2_c0r3"]
    assert json.loads((tmp_path / "runs" / "r1" / "summary.json").read_text())["scrub"] == KERNEL.name


def test_no_scrub_hw_and_emu_pass_none_and_say_so(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, {"a": [(0, 2)]})
    f = SuiteFakes(monkeypatch)
    S.run_hw(suite, "r1", scrub=False)
    S.run_emu(suite, "e1", scrub=False)
    assert [s for *_, s in f.runs] == [None, None] and not f.builds
    for tag in ("r1", "e1"):
        assert S.run_scrub(suite, tag) is None


def test_emu_scrub_passes_the_pattern_without_a_package(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, {"a": [(0, 2)], "b": [(1, 4)]})
    f = SuiteFakes(monkeypatch)
    S.run_emu(suite, "e1", jobs=1)
    assert {(s.name, s.pkg_dir) for *_, s in f.runs} == {(KERNEL.name, None)} and not f.builds
    assert S.run_scrub(suite, "e1") == KERNEL.name


def test_compare_warns_when_runs_started_from_different_states(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, {"a": [(0, 2)]})
    SuiteFakes(monkeypatch)
    S.run_hw(suite, "r1")
    S.run_emu(suite, "e1")
    S.run_emu(suite, "e0", scrub=False)
    assert S.scrub_warning(suite, "r1", "e1") is None
    assert "scrub=none" in S.scrub_warning(suite, "r1", "e0")


def test_scrub_build_failure_fails_that_case_only(tmp_path, monkeypatch):
    suite = make_suite(tmp_path, {"a": [(0, 2)], "b": [(0, 3)]})
    f = SuiteFakes(monkeypatch)
    real = f.build

    def build(topology, out_dir, **kw):
        if topology.cores[0].tile == TileRef(0, 3):
            raise S.BuildError("aiecc failed")
        return real(topology, out_dir, **kw)
    monkeypatch.setattr(S, "build_package", build)
    out = S.run_hw(suite, "r1")
    assert [(o.case, o.status) for o in out] == [("a", "ok"), ("b", "failed")]
    assert "scrub package" in out[1].detail
