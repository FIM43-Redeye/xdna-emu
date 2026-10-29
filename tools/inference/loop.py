"""The closed loop: chain to fixpoint, act on the stall, converge or halt.

Termination rests on a three-component lexicographic measure
(# configured-but-unfired events, # unresolved fired events, # untested candidate
edges); the top component's domain is the static configured-event set from the
xclbin. The seed sweeps the whole configured set; events that don't fire in the
seed are constrained never_fired (with provenance) and excluded from the unfired
count, so the loop converges without a discovery/re-seed path. No livelock
branch exists -- "ambiguous and no batch separates" is degeneracy (it halts).

MockInstrument provides synthetic ground truth for Axis-1 convergence tests; the real
instrument (engine.py) drives the actuator.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
from inference.facts import KB
from inference.loader import load_fired
from inference.ledger import install_ledger
from inference.chainer import chain, classify_events
from inference.planner import propose_next, seed_plan, NO_GAIN, Batch
from inference.reachability import ReachabilityModel, Constraint
from inference.verifier import ANCHOR, Q
import trace_join


class MockInstrument:
    """Synthetic instrument over a known ground-truth model.

    gt = {"events": {event_key: {"base": int, "jitter": int}},
          "routes": [(producer_key, consumer_key), ...],
          "reveal_on_iter": {iter_index: event_key},   # optional discovery
          "workdir": str}
    Each .capture(batch) writes a new run-dir set (n_runs) tracing the requested
    events, sampling jitter deterministically per run index.
    """

    def __init__(self, gt: dict, n_runs: int = 6):
        self.gt = gt
        self.n_runs = n_runs
        self._iter = 0
        self._revealed = set()

    def ledger_entries(self) -> List[dict]:
        return [{"cite": f"route#{i}", "a": a, "b": b, "kind": "route"}
                for i, (a, b) in enumerate(self.gt.get("routes", []))]

    def capture(self, batch: Batch) -> List[str]:
        reveal = self.gt.get("reveal_on_iter", {})
        if self._iter in reveal:
            self._revealed.add(reveal[self._iter])
        requested = {f"{tile}|{name}"
                     for tile, names in batch.tiles.items() for name in names}
        # Always include the anchor so anchoring works.
        requested.add(ANCHOR)
        # Withhold not-yet-revealed events.
        hidden = {k for k in self.gt.get("reveal_on_iter", {}).values()
                  if k not in self._revealed}
        visible = [k for k in requested
                   if k in self.gt["events"] and k not in hidden]
        run_dirs = []
        base_dir = Path(self.gt["workdir"]) / f"capture_{self._iter:02d}"
        for run in range(self.n_runs):
            evs = []
            for ek in visible:
                col, row, pkt, name = ek.split("|")
                spec = self.gt["events"][ek]
                # Deterministic per-run jitter; co-derived events share the run's draw.
                draw = (run * 37) % 100 if spec["jitter"] else 0
                soc = 1000 + spec["base"] + (draw if spec["jitter"] else 0)
                evs.append({"col": int(col), "row": int(row), "pkt_type": int(pkt),
                            "slot": 0, "name": name, "ts": soc, "soc": soc, "mode": 0})
            rd = base_dir / f"run{run}"
            (rd / "batch_00" / "hw").mkdir(parents=True, exist_ok=True)
            (rd / "batch_00" / "hw" / "trace.events.json").write_text(
                json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
            run_dirs.append(str(rd))
        self._iter += 1
        return run_dirs


def ranking(kb: KB, configured_events: List[str],
            candidate_pairs: List[Tuple[str, str]]) -> Tuple[int, int, int]:
    fired = {f.args[0] for f in kb.by_predicate("fired")}
    unfired = sum(1 for e in configured_events if e not in fired)
    cls = classify_events(kb, sorted(fired))
    unresolved = sum(1 for e, c in cls.items() if c == "unresolved")
    derived_pairs = {(f.args[0], f.args[1]) for f in kb.by_predicate("derives")}
    untested = sum(1 for p in candidate_pairs if p not in derived_pairs)
    return (unfired, unresolved, untested)


def run_loop_until_converged(instrument, configured_events: List[str],
                             candidate_pairs: List[Tuple[str, str]],
                             *, anchor_key: str = ANCHOR,
                             max_iters: int = 50) -> dict:
    kb = KB.empty()
    install_ledger(kb, {e["cite"]: e for e in instrument.ledger_entries()})
    model = ReachabilityModel()
    all_run_dirs: List[str] = []
    rankings: List[Tuple[int, int, int]] = []

    # Phase 0: seed sweep over the static configured-event set.
    seed = seed_plan(configured_events)
    seed_dirs = instrument.capture(seed)
    all_run_dirs += seed_dirs

    # Empirical limit (never-fired): the seed traced everything once; anything
    # that did not fire is unfirable -- constrain it with the seed as provenance.
    seeded_fired = {f.args[0] for f in load_fired(all_run_dirs, anchor_key)}
    for ev in configured_events:
        if ev not in seeded_fired and ev not in model.unfirable_events():
            model.add_constraint(Constraint(
                name=f"never_fired:{ev}", predicate="never_fired",
                args=(ev,), provenance_batch=seed_dirs[0]))

    # Empirical limit (uncorrelated): the seed may already co-trace candidate
    # pairs. Check each now -- if co-traced but offset not exact (range > Q),
    # record the constraint so the planner won't re-propose and the loop won't spin.
    for a, b in candidate_pairs:
        name = f"uncorrelated:{a}:{b}"
        if not any(c.name == name for c in model.constraints()):
            st = trace_join.pair_derivability(all_run_dirs, a, b, anchor_key)
            if st is not None and st.range > Q:
                model.add_constraint(Constraint(
                    name=name, predicate="cannot_correlate",
                    args=(a, b), provenance_batch=seed_dirs[0]))

    def live_unfired(fired_set):
        unfirable = model.unfirable_events()
        return [e for e in configured_events
                if e not in fired_set and e not in unfirable]

    prev = None
    for _ in range(max_iters):
        kb_iter = KB.empty()
        install_ledger(kb_iter, {e["cite"]: e for e in instrument.ledger_entries()})
        for f in load_fired(all_run_dirs, anchor_key):
            kb_iter.add(f)
        kb_iter = chain(all_run_dirs, kb_iter, candidate_pairs)
        kb = kb_iter

        r = ranking(kb, configured_events, candidate_pairs)
        rankings.append(r)
        if prev is not None and r > prev:
            raise RuntimeError(f"ranking increased {prev} -> {r} (livelock)")
        prev = r

        fired = {f.args[0] for f in kb.by_predicate("fired")}
        cls = classify_events(kb, sorted(fired))
        unresolved = [e for e, c in cls.items() if c == "unresolved"]
        unfired = live_unfired(fired)
        if not unresolved and not unfired:
            return {"converged": True, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls,
                    "run_dirs": all_run_dirs, "model": model,
                    "terminal_state": "placed"}

        # Act on the stall: propose the next measurement (proven gain only).
        progressed = False
        for pair in candidate_pairs:
            batch = propose_next(kb, all_run_dirs, pair, model, anchor_key)
            if batch is not NO_GAIN:
                new_dirs = instrument.capture(batch)
                all_run_dirs += new_dirs
                # Empirical limit (uncorrelated): offset not exact (range > Q) ->
                # no stable offset, no derivation. Constrain so we don't re-propose.
                a, b = pair
                st = trace_join.pair_derivability(all_run_dirs, a, b, anchor_key)
                if st is not None and st.range > Q:
                    model.add_constraint(Constraint(
                        name=f"uncorrelated:{a}:{b}", predicate="cannot_correlate",
                        args=(a, b), provenance_batch=new_dirs[0]))
                progressed = True
                break
        if not progressed:
            # Halt: distinguish a falsifiable halt (we recorded WHY) from an
            # unexplained one (a bug signal -- unresolved with no constraint).
            state = "halted_falsifiable" if model.constraints() else "halted_unexplained"
            return {"converged": False, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls,
                    "run_dirs": all_run_dirs, "model": model,
                    "terminal_state": state}

    return {"converged": False, "iterations": len(rankings),
            "rankings": rankings, "classification": cls,
            "run_dirs": all_run_dirs, "model": model,
            "terminal_state": "halted_unexplained"}
