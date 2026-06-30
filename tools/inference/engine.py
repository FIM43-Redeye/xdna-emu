"""Top-level engine: static placement reconstruction over captured data + a ledger.

Loads fired (measured) + ledger (structural), checks replication, chains to fixpoint
under the verified rules, condenses placement cycles into irreducible groups, and
classifies degeneracy for coincident root pairs. Returns a placement report. The CLI
runs it over already-captured batch dirs; the closed loop (loop.py) drives the
actuator for MEASURE-NEXT.
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Dict, List, Tuple
from inference.facts import (KB, provenance_ok, derive_kind, derive_offset,
                             derive_reproduction_offset, derive_gap_reason)
from inference.grounding import gap_accounted, same_domain
from inference.timeline import assemble_timeline
from inference.ledger import load_ledger, install_ledger
from inference.loader_model import load_model, install_model
from inference.loader import load_fired, replication_violations
from inference.chainer import chain, classify_events
from inference.degeneracy import IdentityClasses, condense
from inference.classify import classify_pair
from inference.reachability import ReachabilityModel
from inference.verifier import ANCHOR, coincident


def run_engine(run_dirs: List[str], ledger_path: str,
               candidate_pairs: List[Tuple[str, str]],
               anchor_key: str = ANCHOR, dump=None, start_col: int = 1,
               model_path: str = None) -> dict:
    # run_engine stays IO-free: it ACCEPTS an already-loaded config dump (so the
    # connectivity oracle + count-truncation ceiling run in production), it never
    # loads one. The caller (run_experiment) owns dump loading. dump=None keeps
    # backward compatibility for callers that pre-date the connectivity oracle.
    kb = KB.empty()
    install_ledger(kb, load_ledger(ledger_path))
    if model_path is not None:
        install_model(kb, load_model(model_path))
    for f in load_fired(run_dirs, anchor_key):
        kb.add(f)

    reps = replication_violations(run_dirs, anchor_key)
    kb = chain(run_dirs, kb, candidate_pairs, anchor_key)

    fired = sorted({f.args[0] for f in kb.by_predicate("fired")})
    cls = classify_events(kb, fired)
    derives = [f.args for f in kb.by_predicate("derives")]
    roots = [e for e in fired if cls.get(e) == "stochastic_root"]

    _comp, groups = condense(kb)
    irreducible = list(groups)

    # Degeneracy classification for coincident root pairs.
    identity = IdentityClasses.from_kb(kb)
    model = ReachabilityModel()
    verdicts = []
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            a, b = roots[i], roots[j]
            if coincident(run_dirs, a, b, anchor_key):
                verdicts.append(classify_pair(a, b, identity, model).__dict__)

    derives_facts = kb.by_predicate("derives")
    segments = [(f.args[0], f.args[1], derive_offset(f))
                for f in derives_facts if derive_kind(f) == "segment"]
    gaps = [(f.args[0], f.args[1], derive_reproduction_offset(f), derive_gap_reason(f))
            for f in derives_facts if derive_kind(f) == "gap"]
    # Every unaccounted gap (a within-domain span that should be cycle-exact but
    # ranges) is surfaced loudly -- never silently accepted. Accounted-for gaps
    # (cross-domain / async-CDC) stay quiet. Verification is manual (re-capture on
    # a quiet host): see docs/trace/capture-load-sensitivity.md.
    warnings = [{"child": child, "parent": parent, "reason": reason}
                for (child, parent, _repro, reason) in gaps
                if not gap_accounted(reason)]
    rejected = [{"name": r.name, "reason": r.reason, "evidence": r.evidence}
                for r in kb.rejected_rules]

    derives_pairs = {(f.args[0], f.args[1]) for f in kb.by_predicate("derives")}
    cross_domain_pairs = [(c, p) for (c, p) in candidate_pairs if not same_domain(c, p)]
    model = next(iter(kb.model.values()), None)
    timeline = assemble_timeline(run_dirs, fired, derives_pairs, cross_domain_pairs,
                                 dump=dump, start_col=start_col, anchor_key=anchor_key,
                                 model=model)

    causal = [(f.args[0], f.args[1], f.args[2]) for f in kb.by_predicate("causal")]

    return {"replication_violations": reps,
            "classification": cls,
            "derives": derives,
            "segments": segments,
            "gaps": gaps,
            "causal": causal,
            "warnings": warnings,
            "rejected_rules": rejected,
            "stochastic_roots": roots,
            "irreducible_groups": irreducible,
            "degeneracy": verdicts,
            "provenance_ok": provenance_ok(kb),
            "timeline": timeline}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Trace inference engine (static report).")
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs")
    ap.add_argument("--ledger", required=True, help="structural ledger JSON")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="candidate child:parent pairs")
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args(argv)
    pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    rep = run_engine(args.runs, args.ledger, pairs, args.anchor)
    # frozensets are not JSON-serializable -> list them
    rep["irreducible_groups"] = [sorted(g) for g in rep["irreducible_groups"]]
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
