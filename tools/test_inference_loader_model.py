import json
import pytest
from inference.facts import KB, provenance_ok
from inference.loader_model import load_model, install_model
from inference.model_io import SidecarError


def _write(tmp_path, obj):
    p = tmp_path / "origin_d.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_load_model_rekeys_modules_to_pkt_type(tmp_path):
    path = _write(tmp_path, {"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|core": 0, "0|0|shim": 0}})
    m = load_model(path)
    assert m["modules"] == {"1|2|0": 0, "0|0|2": 0}
    assert m["calibrated"] is False
    assert m["flood_source"] == "0|0"


def test_install_model_adds_cited_facts_and_passes_provenance(tmp_path):
    path = _write(tmp_path, {"calibrated": True, "flood_source": "0|0",
                             "modules": {"1|2|core": 4, "1|2|mem": 6}})
    kb = KB.empty()
    install_model(kb, load_model(path), cite="origin_d.json")
    assert kb.get("origin_d", ("1|2|0", 4)) is not None
    assert kb.get("skew_calibrated", (True,)) is not None
    assert kb.get("flood_source", ("0|0",)) is not None
    assert provenance_ok(kb) is True


def test_load_model_rejects_unknown_module_kind(tmp_path):
    path = _write(tmp_path, {"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|bogus": 0}})
    with pytest.raises(SidecarError):
        load_model(path)
