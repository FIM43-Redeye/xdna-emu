import pytest
from inference.grounding import (domain_of, causal_offset, CrossDomainModelError,
                                 ground_edge, Gap, GAP_CROSS_DOMAIN)


def test_domain_of_strips_event_name():
    assert domain_of("1|2|0|DMA_S2MM_0_START_TASK") == "1|2|0"


def test_causal_offset_pins_known_delta_wall():
    # Construct a known physical situation, asymmetric origin_D + known Δwall.
    # child in domain A=1|2|0 (origin_D=10), parent in domain B=0|0|2 (origin_D=4).
    # raw = Δwall − (origin_D[A] − origin_D[B]). Choose Δwall = 7:
    #   raw = 7 − (10 − 4) = 1.  Then causal must recover Δwall = 7.
    model = {"calibrated": True, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    child = "1|2|0|EVT_X"     # domain A
    parent = "0|0|2|EVT_Y"    # domain B
    raw = 1
    assert causal_offset(model, child, parent, raw) == 7


def test_causal_offset_withheld_when_uncalibrated():
    model = {"calibrated": False, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    assert causal_offset(model, "1|2|0|X", "0|0|2|Y", 1) is None


def test_causal_offset_fails_loud_on_missing_domain():
    model = {"calibrated": True, "flood_source": "0|0", "modules": {"1|2|0": 10}}
    with pytest.raises(CrossDomainModelError):
        causal_offset(model, "1|2|0|X", "0|0|2|Y", 1)  # parent domain absent


def test_ground_edge_attaches_causal_offset_to_cross_domain_gap(monkeypatch):
    import inference.grounding as g
    monkeypatch.setattr(g, "offset_exact", lambda *a, **k: 1)
    model = {"calibrated": True, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    res = ground_edge([], "1|2|0|X", "0|0|2|Y", model=model)
    assert isinstance(res, Gap) and res.reason == GAP_CROSS_DOMAIN
    assert res.reproduction_offset == 1 and res.causal_offset == 7


def test_ground_edge_causal_none_without_model(monkeypatch):
    import inference.grounding as g
    monkeypatch.setattr(g, "offset_exact", lambda *a, **k: 1)
    res = ground_edge([], "1|2|0|X", "0|0|2|Y")
    assert res.causal_offset is None and res.reproduction_offset == 1
