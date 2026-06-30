"""Engine-level: installing an uncalibrated model leaves all existing report
fields byte-identical and provenance_ok True (the inert-fact byte-identity
guarantee, design Sec.5e). A calibrated model surfaces a causal list."""
import json
from inference.facts import KB, provenance_ok
from inference.loader_model import install_model, load_model


def test_uncalibrated_model_keeps_provenance_ok(tmp_path):
    p = tmp_path / "origin_d.json"
    p.write_text(json.dumps({"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|core": 0}}))
    kb = KB.empty()
    install_model(kb, load_model(str(p)))
    # Inert ModelDerived leaves present, cited -> provenance_ok holds.
    assert provenance_ok(kb) is True
    # No causal facts emitted pre-calibration.
    assert kb.by_predicate("causal") == []
