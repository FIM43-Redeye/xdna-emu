"""Forward chainer: apply verified rules to fixpoint.

Marks determinism for every fired event, then repeatedly attempts derives/same_source
over the candidate pairs until no new fact is added. Every fact it adds carries a
Derived provenance whose leaves are measured or ledgered-structural -- chaining_sound
(== provenance_ok at fixpoint) pins the no-unaudited-axiom keystone.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
from inference.facts import KB, provenance_ok
from inference.rules import (mark_determinism, try_derives, try_same_source,
                             try_causal, is_stochastic_root)
from inference.verifier import ANCHOR


def _fired_event_keys(kb: KB) -> List[str]:
    return sorted({f.args[0] for f in kb.by_predicate("fired")})


def chain(run_dirs: List[str], kb: KB,
          candidate_pairs: Iterable[Tuple[str, str]],
          anchor_key: str = ANCHOR) -> KB:
    pairs = list(candidate_pairs)
    keys = _fired_event_keys(kb)
    undetermined = [k for k in keys
                    if not (kb.has("deterministic", (k,)) or kb.has("stochastic", (k,)))]
    if undetermined:
        mark_determinism(run_dirs, kb, undetermined, anchor_key)

    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            if not _has_derive(kb, a, b):
                d = try_derives(run_dirs, kb, a, b, anchor_key)
                if d is not None and not kb.has(d.predicate, d.args):
                    kb.add(d); changed = True
            if not _has_same_source(kb, a, b):
                s = try_same_source(run_dirs, kb, a, b, anchor_key)
                if s is not None and not kb.has(s.predicate, s.args):
                    kb.add(s); changed = True
            if not _has_causal(kb, a, b):
                c = try_causal(run_dirs, kb, a, b, anchor_key)
                if c is not None and not kb.has(c.predicate, c.args):
                    kb.add(c); changed = True
    return kb


def _has_derive(kb: KB, child: str, parent: str) -> bool:
    return any(f.args[0] == child and f.args[1] == parent
               for f in kb.by_predicate("derives"))


def _has_same_source(kb: KB, a: str, b: str) -> bool:
    return any({f.args[0], f.args[1]} == {a, b}
               for f in kb.by_predicate("same_source"))


def _has_causal(kb: KB, child: str, parent: str) -> bool:
    return any(f.args[0] == child and f.args[1] == parent
               for f in kb.by_predicate("causal"))


def classify_events(kb: KB, event_keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    derived_children = {f.args[0] for f in kb.by_predicate("derives")}
    for ek in event_keys:
        if ek in derived_children:
            out[ek] = "derived"
        elif is_stochastic_root(kb, ek):
            out[ek] = "stochastic_root"
        elif kb.has("deterministic", (ek,)):
            out[ek] = "deterministic"
        else:
            out[ek] = "unresolved"
    return out


def chaining_sound(kb: KB) -> bool:
    return provenance_ok(kb)
