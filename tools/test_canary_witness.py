import glob
import json
import os
import pytest
from canary_witness import witness_clean, WitnessResult, SENTINEL_ACQ, SENTINEL_REL

_EXP = "/home/triple/npu-work/xdna-emu/build/experiments"


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    """rows: list over runs of {full_key: anchored_offset}. Writes the
    run_NN/batch_00/hw/trace.events.json layout the engine reads."""
    dirs = []
    for i, row in enumerate(rows):
        rd = tmp_path / f"run_{i:02d}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_witness_clean_on_cycle_exact_span(tmp_path):
    # core-lock span exactly 24 every run -> Segment -> clean.
    dirs = _runs(tmp_path, [{SENTINEL_ACQ: 0, SENTINEL_REL: 24},
                            {SENTINEL_ACQ: 50, SENTINEL_REL: 74}])
    res = witness_clean(dirs)
    assert isinstance(res, WitnessResult)
    assert res.clean is True
    assert res.offset == 24
    assert res.reason is None


def test_witness_dirty_on_flickered_span(tmp_path):
    # span 24 then 23 -> within_domain_nonexact -> dirty.
    dirs = _runs(tmp_path, [{SENTINEL_ACQ: 0, SENTINEL_REL: 24},
                            {SENTINEL_ACQ: 50, SENTINEL_REL: 73}])
    res = witness_clean(dirs)
    assert res.clean is False
    assert res.reason == "within_domain_nonexact"
    assert res.offset is None


@pytest.mark.skipif(not os.path.isdir(f"{_EXP}/lock-jitter-clean"),
                    reason="persisted HW fixtures absent on this machine")
def test_witness_certifies_real_clean_fixture():
    runs = sorted(glob.glob(f"{_EXP}/lock-jitter-clean/capture_00/run_*"))
    res = witness_clean(runs)
    assert res.clean is True and res.offset == 24


@pytest.mark.skipif(not os.path.isdir(f"{_EXP}/lock-jitter-loaded"),
                    reason="persisted HW fixtures absent on this machine")
def test_witness_flags_real_loaded_fixture():
    runs = sorted(glob.glob(f"{_EXP}/lock-jitter-loaded/capture_00/run_*"))
    res = witness_clean(runs)
    assert res.clean is False and res.reason == "within_domain_nonexact"


def test_capture_and_witness_applies_verdict_to_captured_dirs(tmp_path, monkeypatch):
    # Stub the HW capture: write a clean 2-run sentinel capture, return its dirs.
    import canary_witness as cw
    dirs = _runs(tmp_path, [{SENTINEL_ACQ: 0, SENTINEL_REL: 24},
                            {SENTINEL_ACQ: 9, SENTINEL_REL: 33}])

    def fake_capture(out_root, test, compiler, n_runs):
        return dirs  # pretend the NPU produced these run dirs

    monkeypatch.setattr(cw, "_capture_sentinel_runs", fake_capture)
    res = cw.capture_and_witness(str(tmp_path), n_runs=2)
    assert res.clean is True and res.offset == 24


def test_capture_sentinel_runs_passes_real_dump_fixture(tmp_path, monkeypatch):
    # The live capture must configure the sentinel's lock events -- which requires
    # the add_one dump fixture, NOT dump_path=None. Verify the real fixture path
    # is passed through to run_experiment (and that the fixture actually exists).
    import canary_witness as cw
    import inference.run_experiment as rexp
    captured = {}

    def fake_run(cfg, *a, **k):
        captured["cfg"] = cfg
        (tmp_path / "capture_00" / "run_00").mkdir(parents=True)
        return {}

    monkeypatch.setattr(rexp, "run_experiment", fake_run)
    cw._capture_sentinel_runs(str(tmp_path), "add_one_using_dma", "chess", 2)
    assert captured["cfg"].dump_path == cw.SENTINEL_DUMP
    assert os.path.isfile(cw.SENTINEL_DUMP), "sentinel dump fixture must exist"
