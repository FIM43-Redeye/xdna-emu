# tools/test_inference_sp4b_e2e.py
"""End-to-end (SP-4b, #140): the whole inference-engine pipeline with an
origin_D sidecar attached.

Two properties, one test each:

  1. An UNCALIBRATED sidecar is a pure no-op on the report: `causal` stays
     empty and every other field is byte-identical to a run with no sidecar
     at all (`model_path=None`).
  2. A CALIBRATED sidecar surfaces `causal` facts -- but only where there is
     a cross-domain gap to decompose, and only with the model's origin_D
     leaves cited so `provenance_ok` stays True.

Fixtures: the byte-identity test (the load-bearing one) runs against REAL
captured silicon data -- the same `build/experiments/infer-smoke/run_*` +
`add_one_using_dma.ledger.json` + 5-pair candidate set that
test_inference_hw_smoke.py uses (reused verbatim, no env-var HW gate needed
since the data is already captured on disk).

That real fixture's 5 candidate pairs are all WITHIN-domain (same
col|row|pkt_type prefix on both endpoints -- "1|0|2" or "1|1|3", see
inference.grounding.same_domain), so calibrating it can never produce a
causal fact: there is no cross-domain gap for try_causal to decompose. The
calibrated test checks that honestly against the real fixture first
(provenance_ok stays True, causal stays empty), then additionally exercises
the causal-emission path itself against a minimal synthetic cross-domain
capture (mirroring test_inference_engine.py::
test_engine_gap_carries_reproduction_offset) -- constructing a synthetic
RUNS fixture for this purpose is the documented fallback the task brief
sanctions when a real fixture cannot reach the code path under test.
"""
import json
from pathlib import Path

from inference.engine import run_engine

_HERE = Path(__file__).resolve().parent
_INFER_SMOKE = _HERE.parent / "build" / "experiments" / "infer-smoke"
_LEDGER = _HERE / "inference" / "fixtures" / "add_one_using_dma.ledger.json"

# Reused verbatim from test_inference_hw_smoke.py::_CANDIDATE_PAIRS -- the
# five config-oriented (child, parent) pairs the sound engine derives from
# the add_one_using_dma capture. All five are within-domain (see module
# docstring), which is exactly why the calibrated test needs the synthetic
# cross-domain fixture below to reach try_causal's decomposition path.
_CANDIDATE_PAIRS = [
    ("1|0|2|DMA_S2MM_0_START_TASK", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|0|2|DMA_S2MM_0_STREAM_STARVATION", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|1|3|PORT_RUNNING_1", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_0"),
]


def _real_run_dirs():
    dirs = sorted(str(p) for p in _INFER_SMOKE.glob("run_*") if p.is_dir())
    assert dirs, (
        f"no run_* dirs under {_INFER_SMOKE} -- expected the captured "
        f"infer-smoke fixture (also used by test_inference_hw_smoke.py) "
        f"to be present on disk")
    return dirs


def _without(report: dict, *keys: str) -> dict:
    return {k: v for k, v in report.items() if k not in keys}


def test_uncalibrated_sidecar_is_byte_identical(tmp_path):
    dirs = _real_run_dirs()
    base = run_engine(dirs, str(_LEDGER), _CANDIDATE_PAIRS)

    sidecar = tmp_path / "origin_d.json"
    sidecar.write_text(json.dumps({
        "calibrated": False,
        "flood_source": "0|0",
        "modules": {"1|2|core": 0},
    }))
    withmodel = run_engine(dirs, str(_LEDGER), _CANDIDATE_PAIRS,
                           model_path=str(sidecar))

    assert withmodel["causal"] == []
    assert withmodel["provenance_ok"] is True
    # Every field OTHER than `causal` is untouched by attaching an
    # uncalibrated sidecar: causal_offset() (inference.grounding) returns
    # None unconditionally when model["calibrated"] is False, before it ever
    # looks at the modules table, so derives/segments/gaps/timeline/etc. take
    # the identical code path whether or not a sidecar is attached at all.
    assert _without(base, "causal") == _without(withmodel, "causal")


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _synthetic_cross_domain_fixture(tmp_path):
    """A minimal hand-built 2-run capture carrying one cross-domain pair
    (shim "1|0|2" MM2S -> core "1|2|0" CORE) with an EXACT raw offset (40 in
    both runs), so ground_edge() lands a Gap(reason="cross_domain") with
    reproduction_offset=40 -- the only shape try_causal ever decomposes.
    Same construction as
    test_inference_engine.py::test_engine_gap_carries_reproduction_offset;
    rebuilt locally (not imported) to keep this e2e file self-contained.
    """
    dirs = []
    for i, row in enumerate([
        {"1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|2|MM2S": 9, "1|2|0|CORE": 49},
    ]):
        rd = tmp_path / f"syn_run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    ledger_path = tmp_path / "syn_led.json"
    ledger_path.write_text(json.dumps({"entries": [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE",
         "kind": "program"}]}))
    return dirs, str(ledger_path)


def test_calibrated_sidecar_emits_causal_with_provenance(tmp_path):
    # --- Real fixture: honest single-domain result ---------------------
    # The captured add_one_using_dma fixture's 5 candidate pairs are all
    # within-domain (module docstring); calibrating it can never produce a
    # causal fact because no cross-domain gap exists for try_causal to act
    # on. Note that honestly rather than forcing a synthetic pair onto real
    # capture data that structurally has none.
    dirs = _real_run_dirs()
    real_sidecar = tmp_path / "origin_d_real.json"
    real_sidecar.write_text(json.dumps({
        "calibrated": True,
        "flood_source": "0|0",
        "modules": {"1|0|shim": 0, "1|1|memtile": 0},
    }))
    real_rep = run_engine(dirs, str(_LEDGER), _CANDIDATE_PAIRS,
                          model_path=str(real_sidecar))
    assert real_rep["provenance_ok"] is True
    assert real_rep["causal"] == []

    # --- Synthetic cross-domain fixture: causal emission with provenance ---
    # Covers the path the real fixture structurally cannot reach: a
    # calibrated sidecar whose `modules` table covers BOTH endpoint domains
    # of a cross-domain gap turns it into a causal fact (design Sec.5a-5d),
    # citing the origin_D ModelDerived leaves so provenance_ok still holds.
    syn_dirs, syn_ledger = _synthetic_cross_domain_fixture(tmp_path)
    # domain_of("1|2|0|CORE") == "1|2|0" -> module kind "core" (pkt_type 0)
    # domain_of("1|0|2|MM2S") == "1|0|2" -> module kind "shim" (pkt_type 2)
    # (inference.model_io.MODULE_PKT_TYPE: core=0, mem=1, shim=2, memtile=3)
    syn_sidecar = tmp_path / "origin_d_synthetic.json"
    syn_sidecar.write_text(json.dumps({
        "calibrated": True,
        "flood_source": "0|0",
        "modules": {"1|2|core": 5, "1|0|shim": 2},
    }))
    syn_rep = run_engine(syn_dirs, syn_ledger, [("1|2|0|CORE", "1|0|2|MM2S")],
                         model_path=str(syn_sidecar))
    assert syn_rep["provenance_ok"] is True
    # raw reproduction_offset (40) - skew(shim(2) - core(5) = -3) = 43.
    assert ("1|2|0|CORE", "1|0|2|MM2S", 43) in syn_rep["causal"]
