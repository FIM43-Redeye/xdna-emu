"""Harvest a bridge --trace results dir into a fresh accuracy baseline: per
(kernel, compiler) CLEAN/DIVERGE verdict, a divergence score for ranking, and a
best-effort cross-reference against the known-fidelity-gaps registry. Emits a
dated markdown baseline. See docs/superpowers/specs/2026-06-26-trace-accuracy-
corpus-remeasure-design.md.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

_DIVERGE_RE = re.compile(
    r"edge:\s*(\d+)\s*diverged,\s*(\d+)\s*mismatch;\s*"
    r"level:\s*(\d+)\s*diverged,\s*(\d+)\s*mismatch")


@dataclass
class Entry:
    kernel: str
    compiler: str
    verdict: str
    score: int
    documented: bool = False


def parse_divergence_score(trace_log_text: str) -> int:
    """Sum the four edge/level diverged+mismatch counts on the TRACE_VERDICT
    line. 0 if the line is CLEAN or absent."""
    m = _DIVERGE_RE.search(trace_log_text)
    if not m:
        return 0
    return sum(int(g) for g in m.groups())


def harvest(results_dir: str, known_gaps_path: Optional[str] = None) -> List[Entry]:
    known_text = ""
    if known_gaps_path and os.path.isfile(known_gaps_path):
        with open(known_gaps_path) as fh:
            known_text = fh.read()
    entries: List[Entry] = []
    for summ in sorted(glob.glob(os.path.join(results_dir, "*.trace.summary"))):
        base = os.path.basename(summ)[:-len(".trace.summary")]
        kernel, _, compiler = base.rpartition(".")
        with open(summ) as fh:
            verdict = fh.read().strip()
        log_path = os.path.join(results_dir, f"{base}.trace.log")
        score = 0
        if verdict == "DIVERGE" and os.path.isfile(log_path):
            with open(log_path) as fh:
                score = parse_divergence_score(fh.read())
        documented = bool(known_text) and kernel in known_text
        entries.append(Entry(kernel, compiler, verdict, score, documented))
    return entries


def render_markdown(entries: List[Entry], date_str: str) -> str:
    n_clean = sum(1 for e in entries if e.verdict == "CLEAN")
    n_div = sum(1 for e in entries if e.verdict == "DIVERGE")
    n_err = sum(1 for e in entries if e.verdict not in ("CLEAN", "DIVERGE"))
    lines = [f"# Trace-Accuracy Baseline {date_str}", "",
             f"**Tally:** {n_clean} CLEAN / {n_div} DIVERGE / {n_err} ERROR "
             f"(of {len(entries)} kernel-compiler points)", "",
             "## Ranked divergences (worst first)", "",
             "| Kernel | Compiler | Score | Documented gap |",
             "|--------|----------|-------|----------------|"]
    diverging = sorted((e for e in entries if e.verdict == "DIVERGE"),
                       key=lambda e: e.score, reverse=True)
    for e in diverging:
        lines.append(f"| {e.kernel} | {e.compiler} | {e.score} | "
                     f"{'yes' if e.documented else 'NEW'} |")
    lines += ["", "## CLEAN", ""]
    for e in sorted((e for e in entries if e.verdict == "CLEAN"),
                    key=lambda e: (e.kernel, e.compiler)):
        lines.append(f"- {e.kernel} ({e.compiler})")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Harvest a trace accuracy baseline.")
    ap.add_argument("--results", required=True, help="bridge --trace results dir")
    ap.add_argument("--out", required=True, help="output markdown path")
    ap.add_argument("--known-gaps",
                    default="docs/known-fidelity-gaps.md")
    ap.add_argument("--date", default="UNDATED")
    args = ap.parse_args(argv)
    entries = harvest(args.results, args.known_gaps)
    with open(args.out, "w") as fh:
        fh.write(render_markdown(entries, args.date))
    print(f"baseline written: {args.out} ({len(entries)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
