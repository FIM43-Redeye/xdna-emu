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
from inference.facts import KB, provenance_ok
from inference.ledger import load_ledger, install_ledger
from inference.loader import load_fired, replication_violations
from inference.chainer import chain, classify_events
from inference.degeneracy import IdentityClasses, condense
from inference.classify import classify_pair
from inference.reachability import ReachabilityModel
from inference.verifier import ANCHOR, coincident


def run_engine(run_dirs: List[str], ledger_path: str,
               candidate_pairs: List[Tuple[str, str]],
               anchor_key: str = ANCHOR) -> dict:
    kb = KB.empty()
    install_ledger(kb, load_ledger(ledger_path))
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

    return {"replication_violations": reps,
            "classification": cls,
            "derives": derives,
            "stochastic_roots": roots,
            "irreducible_groups": irreducible,
            "degeneracy": verdicts,
            "provenance_ok": provenance_ok(kb)}


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
