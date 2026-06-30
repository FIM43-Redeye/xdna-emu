"""Model-derived leaves: load the emulator's origin_D sidecar (origin_d.json).

The sidecar is the emulator's modeled per-module timer-reset arrival. It is
neither measured (loader.py) nor structural/binary (ledger.py) -- it is
model-derived, loaded under facts.ModelDerived and cited to the artifact so a
causal fact built on it stays traceable to the model. Module-kind keys are
translated to numeric pkt_type domain keys here (model_io)."""
from __future__ import annotations
import json
from typing import Dict, List
from inference.facts import Fact, ModelDerived, KB
from inference.model_io import to_domain_key, SidecarError


def load_model(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    for k in ("calibrated", "modules"):
        if k not in raw:
            raise SidecarError(f"sidecar missing {k!r}: {path}")
    rekeyed: Dict[str, int] = {}
    for mk, d in raw["modules"].items():
        parts = mk.split("|")
        if len(parts) != 3:
            raise SidecarError(f"bad module key {mk!r}")
        col, row, kind = parts
        rekeyed[to_domain_key(int(col), int(row), kind)] = d
    return {"calibrated": bool(raw["calibrated"]),
            "flood_source": raw.get("flood_source"),
            "modules": rekeyed}


def model_facts(model: dict, cite: str) -> List[Fact]:
    facts: List[Fact] = [
        Fact("skew_calibrated", (model["calibrated"],), ModelDerived(cite)),
    ]
    if model.get("flood_source") is not None:
        facts.append(Fact("flood_source", (model["flood_source"],), ModelDerived(cite)))
    for dom, d in model["modules"].items():
        facts.append(Fact("origin_d", (dom, d), ModelDerived(cite)))
    return facts


def install_model(kb: KB, model: dict, cite: str = "origin_d.json") -> None:
    kb.model[cite] = model
    for f in model_facts(model, cite):
        kb.add(f)
