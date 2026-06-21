"""Structural ledger: config_path / identity facts read from an input JSON file.

In Plan 1 the ledger is an INPUT -- a quote of the loaded configuration provided as
JSON (hand-authored for the add_one_using_dma HW smoke test). Plan 2 replaces the
loader with an automated extractor (a Rust examples/dump_config_json.rs over the
parsed CDO: stream-switch routes, BD chains, lock pairings) and citation_resolves
with resolution against the real binary.

Schema:
  {"entries": [
    {"cite": str, "a": event_key, "b": event_key,
     "kind": "route"|"bd"|"lock"|"identity"}, ...]}

route/bd/lock -> config_path(a, b, cite): the config routes a's producer to b's
consumer (orientation premise for `derives`).
identity      -> identity(a, b, cite): a and b are the same physical event at two
                 trace units (premise for `same_source`).
"""
from __future__ import annotations
import json
from typing import Dict, List
from inference.facts import Fact, Structural, KB

_KINDS = {"route", "bd", "lock", "identity"}


def load_ledger(path: str) -> Dict[str, dict]:
    with open(path) as fh:
        raw = json.load(fh)
    out: Dict[str, dict] = {}
    for e in raw.get("entries", []):
        for k in ("cite", "a", "b", "kind"):
            if k not in e:
                raise ValueError(f"ledger entry missing {k!r}: {e}")
        if e["kind"] not in _KINDS:
            raise ValueError(f"unknown kind {e['kind']!r} in {e}")
        if e["cite"] in out:
            raise ValueError(f"duplicate cite {e['cite']!r}")
        out[e["cite"]] = e
    return out


def ledger_facts(ledger: Dict[str, dict]) -> List[Fact]:
    facts: List[Fact] = []
    for cite, e in ledger.items():
        pred = "identity" if e["kind"] == "identity" else "config_path"
        facts.append(Fact(pred, (e["a"], e["b"], cite), Structural(cite)))
    return facts


def install_ledger(kb: KB, ledger: Dict[str, dict]) -> None:
    kb.ledger = ledger
    for f in ledger_facts(ledger):
        kb.add(f)


def citation_resolves(ledger: Dict[str, dict], cite: str) -> bool:
    return cite in ledger
