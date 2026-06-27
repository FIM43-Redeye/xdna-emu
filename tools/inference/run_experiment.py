"""Entry point: run a kernel through the active loop to a terminal state and
write a provenance-complete convergence report.

CLI:
    cd tools && env -u XDNA_EMU python -m inference.run_experiment \\
        --test add_one_using_dma --dump config_extract/fixtures/add_one_using_dma.config.json \\
        --start-col 1 --out ../build/experiments/exp-loop/add_one
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class KernelConfig:
    test: str
    compiler: str
    dump_path: Optional[str]
    start_col: int
    anchor_tile_abs: str
    anchor_event: str
    n_runs: int
    out_root: str


def run_experiment(cfg: KernelConfig, instrument=None,
                   configured: Optional[List[str]] = None,
                   candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> dict:
    from config_extract.dump_model import load_dump
    from inference.selfmodel import (enumerate_configured_events,
                                     candidate_pairs_from_dump)
    from inference.loop import run_loop_until_converged
    from inference.hw_instrument import HwInstrument

    anchor_key = f"{cfg.anchor_tile_abs}|{cfg.anchor_event}"

    dump = load_dump(cfg.dump_path) if cfg.dump_path else None
    if configured is None:
        configured = enumerate_configured_events(dump, cfg.start_col)
    if candidate_pairs is None:
        candidate_pairs = candidate_pairs_from_dump(dump, configured, cfg.start_col)

    if instrument is None:
        instrument = HwInstrument(
            cfg.test, dump, configured, start_col=cfg.start_col,
            anchor_tile_abs=cfg.anchor_tile_abs, anchor_event=cfg.anchor_event,
            n_runs=cfg.n_runs, out_root=cfg.out_root, compiler=cfg.compiler)

    res = run_loop_until_converged(instrument, configured, candidate_pairs,
                                   anchor_key=anchor_key)

    # Rich placement backbone from the engine over the final run dirs.
    derives, roots, provenance_ok, engine_ok = [], [], None, False
    segments, gaps, rejected_rules, warnings = [], [], [], []
    timeline = None
    try:
        from inference.engine import run_engine
        led = {"entries": instrument.ledger_entries()}
        ledger_path = Path(cfg.out_root) / "ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(led))
        # Thread the config dump into the engine so the independent connectivity
        # oracle + count-truncation ceiling actually run in production. run_engine
        # is IO-free (it accepts a dump object), so the load happens HERE; guard it
        # so a missing/unparseable dump degrades to dump=None rather than failing.
        engine_dump = None
        if cfg.dump_path:
            try:
                from config_extract.dump_model import load_dump
                engine_dump = load_dump(cfg.dump_path)
            except Exception:
                engine_dump = None
        rep = run_engine(res["run_dirs"], str(ledger_path), candidate_pairs,
                         dump=engine_dump, start_col=cfg.start_col)
        derives = rep.get("derives", [])
        segments = rep.get("segments", [])
        gaps = rep.get("gaps", [])
        warnings = rep.get("warnings", [])
        rejected_rules = rep.get("rejected_rules", [])
        roots = rep.get("stochastic_roots", [])
        provenance_ok = rep.get("provenance_ok")
        timeline = rep.get("timeline")
        engine_ok = True
    except Exception as exc:  # engine report is best-effort; loop result stands.
        provenance_ok = f"engine_report_error: {exc}"

    return {
        "kernel": cfg.test,
        "converged": res["converged"],
        "terminal_state": res["terminal_state"],
        "iterations": res["iterations"],
        "classification": res["classification"],
        "derives": derives,
        "segments": segments,
        "gaps": gaps,
        "warnings": warnings,
        "rejected_rules": rejected_rules,
        "stochastic_roots": roots,
        "provenance_ok": provenance_ok,
        "engine_ok": engine_ok,
        "timeline": timeline,
        "constraints": [
            {"name": c.name, "predicate": c.predicate, "args": list(c.args),
             "provenance_batch": c.provenance_batch}
            for c in res["model"].constraints()],
        "config": asdict(cfg),
    }


def write_report(report: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # "timeline" is a rich Python object (IntegratedTimeline) -- not JSON-serializable;
    # omit it from the persisted JSON file (consumed programmatically, not via files).
    p.write_text(json.dumps({k: v for k, v in report.items() if k != "timeline"}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--compiler", default="chess")
    ap.add_argument("--start-col", type=int, default=1)
    ap.add_argument("--anchor-tile", default="1|2|0")
    ap.add_argument("--anchor-event", default="PERF_CNT_2")
    ap.add_argument("--n-runs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = KernelConfig(test=a.test, compiler=a.compiler, dump_path=a.dump,
                       start_col=a.start_col, anchor_tile_abs=a.anchor_tile,
                       anchor_event=a.anchor_event,
                       n_runs=a.n_runs, out_root=a.out)
    report = run_experiment(cfg)
    out_path = str(Path(a.out) / "convergence_report.json")
    write_report(report, out_path)
    print(f"[run_experiment] {a.test}: {report['terminal_state']} "
          f"({report['iterations']} iters); report -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
