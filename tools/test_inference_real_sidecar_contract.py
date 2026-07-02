# tools/test_inference_real_sidecar_contract.py
"""Cross-language contract (SP-5a, #140): a REAL Rust-produced origin_d.json
round-trips through the Python inference engine.

SP-4b only ever fed hand-written synthetic sidecars to run_engine. This test
closes that gap by consuming the committed real export
(tools/tests/fixtures/origin_d_real_export.json, produced by the Rust test
export_origin_d_sidecar_matches_committed_fixture). load_model re-keys the
real "col|row|kind" module strings via to_domain_key on load -- so loading the
real fixture at all exercises the real key/translation path and catches
schema/key/type drift between the two language sides.

The fixture is committed (not a gitignored capture), so it is always present --
no skipif gate needed.

Two properties:
  (a) The real fixture carries calibrated:false (SP-5a does not flip it), so it
      is a clean no-op end-to-end even where a cross-domain gap exists.
  (b) The real fixture's SCHEMA can drive a causal emission once calibrated:
      flip calibrated:true and populate modules for a cross-domain pair's two
      domains. Note: the cross-domain emission gate (inference.grounding)
      checks both domains are present in `modules`, NOT flood_source
      reachability, so this test asserts nothing about flood_source.
"""
import json
from pathlib import Path

from inference.engine import run_engine

_HERE = Path(__file__).resolve().parent
_FIXTURE = _HERE / "tests" / "fixtures" / "origin_d_real_export.json"


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _synthetic_cross_domain_fixture(tmp_path):
    """A minimal 2-run capture with one cross-domain pair (shim "1|0|2" MM2S ->
    core "1|2|0" CORE), exact raw offset 40 -- the shape try_causal decomposes.
    Self-contained (mirrors test_inference_sp4b_e2e.py)."""
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


def test_real_fixture_is_calibrated_and_drives_causal(tmp_path):
    # SP-5c flip (2026-07-02): the real exported fixture is now CALIBRATED, so
    # feeding it DRIVES causal emission (the inverse of the SP-5a-era no-op).
    real = json.loads(_FIXTURE.read_text())
    assert real["calibrated"] is True, "SP-5c flipped calibrated"
    syn_dirs, syn_ledger = _synthetic_cross_domain_fixture(tmp_path)
    sidecar = tmp_path / "origin_d_real.json"
    sidecar.write_text(json.dumps(real))
    # The pair re-keys to shim "1|0|shim" and core "1|2|core"; the real fixture's
    # own origin_D for both is 8 (shim(1,0) = the d_h+2*d_v E/W detour; core(1,2)
    # = Manhattan d_h+2*d_v), so skew = 8-8 = 0 and causal = raw 40 - 0 = 40.
    # Exercises load_model's re-keying of the REAL "col|row|kind" module strings
    # end-to-end through the calibrated path.
    assert real["modules"]["1|0|shim"] == 8 and real["modules"]["1|2|core"] == 8
    rep = run_engine(syn_dirs, syn_ledger, [("1|2|0|CORE", "1|0|2|MM2S")],
                     model_path=str(sidecar))
    assert rep["provenance_ok"] is True
    assert ("1|2|0|CORE", "1|0|2|MM2S", 40) in rep["causal"]


def test_real_fixture_schema_drives_causal_when_calibrated(tmp_path):
    real = json.loads(_FIXTURE.read_text())
    syn_dirs, syn_ledger = _synthetic_cross_domain_fixture(tmp_path)
    # Start from the real fixture's schema; flip calibrated and set modules to
    # cover the synthetic cross-domain pair's two domains
    # (model_io.MODULE_PKT_TYPE: core=0, shim=2). skew(shim 2 - core 5 = -3),
    # raw 40 - (-3) = 43.
    calibrated = dict(real)
    calibrated["calibrated"] = True
    calibrated["modules"] = {"1|2|core": 5, "1|0|shim": 2}
    sidecar = tmp_path / "origin_d_calibrated.json"
    sidecar.write_text(json.dumps(calibrated))
    rep = run_engine(syn_dirs, syn_ledger, [("1|2|0|CORE", "1|0|2|MM2S")],
                     model_path=str(sidecar))
    assert rep["provenance_ok"] is True
    assert ("1|2|0|CORE", "1|0|2|MM2S", 43) in rep["causal"]
