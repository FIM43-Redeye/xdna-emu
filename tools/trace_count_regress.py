#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bridge-suite regression gate: compare two result dirs on trace counts AND timing.

``scripts/emu-bridge-test.sh`` writes ``<test>.<compiler>.{hw,emu}/events.json``
into ``build/bridge-test-results/YYYYMMDD/``.  Its CLEAN/DIVERGE verdict
anchors each tile on one event and compares pairs, so it misses a change that
adds phantom intervals on an otherwise quiet row (EMU STREAM_STARVATION went
32 -> 62 against silicon's 32).  This tool scores every (test, compiler, tile,
event) row of a baseline run against a candidate run, always measured against
the silicon samples of both days.

Rows
----
A row is (test, compiler, tile col/row, tile module, event name); columns are
shifted by the file's placement origin so HW and EMU line up.  Its *series* is
the list of event timestamps; for level events (stalls, starvation, port
running ... one event per cycle while the level is high) consecutive cycles
collapse into an interval and the series is the interval start times, the same
convention as trace-compare's ``events_to_intervals``.  A row's count is the
length of its series (intervals for level rows).  A side that never fires a
row has count 0, so a phantom row (EMU fires, silicon does not) is a plain
0 -> N count move.

Scores (both are distances to silicon, lower is better)
-------------------------------------------------------
* Count score: distance of the EMU count from the silicon band, 0 inside: the
  closed range [min, max] of the HW counts over the (up to) two HW samples,
  widened on each side by its own spread (``silicon_band``), since two samples
  underestimate silicon's run-to-run spread and a 1-event EMU move inside it is
  not a regression. A repeatable row (both samples equal) stays exact. The band
  is the same for baseline and candidate scoring, so only the EMU change moves
  the score.  ``hw_repeatable`` is True when both
  HW samples have the same count.
* Pattern score: 1-Wasserstein distance (cycles) between the EMU's and HW's
  inter-event gap distributions.  Gaps are shift invariant (no anchoring),
  and W1 is in cycles, so "gap 128 vs silicon 140" scores 12 and "131" scores
  9: a closer pattern is a smaller number, and extra or missing events change
  the score only through the gaps they add.  Distributions of different
  length are compared as quantile functions (exact integral of
  |Q_emu - Q_hw| over [0, 1]), which is the textbook W1 and needs no
  resampling.  Ordering of gaps is deliberately ignored: a systematic shift of
  the cadence is what we care about.  With two HW samples the score is the
  minimum over them (consistent with the range distance: "close to any
  silicon observation").  Undefined (None) when either side has no gap, i.e.
  fewer than two events.

Classification of a changed row
-------------------------------
A row is *changed* when its EMU count moved or its pattern score moved by
more than PATTERN_EPS cycles.  Let "count worse/better" mean the count score
went up/down, "pattern better/worse" a move beyond PATTERN_EPS, and "pattern
not better" include equal and undefined.

* count worse, pattern better, and the new count score is at most
  max(PHASE_MIN_COUNT, PHASE_FRAC * HW max): "phase shift (pattern closer)",
  verdict neutral.  This is the objectfifo_turns_walkers trap: a count equal
  to silicon by lucky phase alignment moved by a few while the cadence got
  closer to silicon.
* count worse otherwise: regressed.  **Gate failure** when the pattern is
  not unchanged (farther, undefined because silicon has no gaps, or closer
  but beyond the phase tolerance), or when a repeatable exact match broke.
  Count worse with an unchanged pattern and no broken exact match is a soft
  regression (listed, exit 0).
* count better, pattern worse: neutral ("mixed"); count better otherwise:
  improved.
* count equal: pattern better -> improved, pattern worse -> regressed (soft,
  does not fail the gate), else neutral.

Exit status: 1 when any row is a gate failure: a row regressing on both
scores (an undefined pattern counts as not improving, so phantom rows on
quiet channels fail), or a repeatable-HW exact count match breaking without
the pattern improving by enough to be a phase shift.  Otherwise 0; 2 for
usage errors.  A "repeatable" match needs two HW samples that agree.  Tests present in only one run, or lacking either side's
events.json, are listed but not scored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

PATTERN_EPS = 0.5      # cycles of W1 movement below which the pattern is "unchanged"
PHASE_FRAC = 0.10      # phase shift allowed to move the count by this fraction of HW max
PHASE_MIN_COUNT = 2    # ... but at least this many events/intervals

# Level events: held while true, one trace event per cycle.  This mirrors
# crates/xdna-archspec trace_events::is_level_event by name family; validated
# against the [level]/[edge] tags of every trace.log in the real results.
_LEVEL_RE = re.compile(
    r"(STALL|STARVATION|BACKPRESSURE|PORT_RUNNING|CONFLICT_DM_BANK|^ACTIVE$|^DISABLED$)"
)


def is_level_name(name: str) -> bool:
    return bool(_LEVEL_RE.search(name))


# --- series and distances -----------------------------------------------

def series_from_events(events: list[dict], name: str) -> list[int]:
    """Sorted timestamps; level events collapse to interval start times."""
    ts = sorted(e["ts"] for e in events)
    if not is_level_name(name) or not ts:
        return ts
    starts = [ts[0]]
    for prev, cur in zip(ts, ts[1:]):
        if cur - prev > 1:
            starts.append(cur)
    return starts


def gaps(series: list[int]) -> list[int]:
    return [b - a for a, b in zip(series, series[1:])]


def wasserstein1(a: list[float], b: list[float]) -> float:
    """Exact 1-Wasserstein between two empirical samples of any lengths."""
    if not a or not b:
        raise ValueError("wasserstein1 needs non-empty samples")
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    # Quantile functions are step functions on [0,1]; walk merged breakpoints
    # in integer units of 1/(na*nb) to stay exact.
    total = 0.0
    i = j = 0
    pos = 0
    end = na * nb
    while pos < end:
        nxt = min((i + 1) * nb, (j + 1) * na)
        total += abs(a[i] - b[j]) * (nxt - pos)
        pos = nxt
        if pos == (i + 1) * nb:
            i += 1
        if pos == (j + 1) * na:
            j += 1
    return total / end


def range_distance(count: int, counts: list[int]) -> int:
    lo, hi = min(counts), max(counts)
    return lo - count if count < lo else count - hi if count > hi else 0


def silicon_band(counts: list[int]) -> tuple[int, int]:
    """The count band silicon is taken to cover: [min, max] widened by its own spread.

    Two samples underestimate how far silicon wanders, so a row whose samples disagree by d is
    given d of slack on each side (a crude prediction interval: the next sample is about as
    likely to land a spread beyond the observed range as inside it). A repeatable row (spread 0)
    stays exact, so a phantom row on a quiet channel or a broken exact match still scores.
    """
    lo, hi = min(counts), max(counts)
    d = hi - lo
    return lo - d, hi + d


def band_distance(count: int, counts: list[int]) -> int:
    lo, hi = silicon_band(counts)
    return lo - count if count < lo else count - hi if count > hi else 0


def pattern_distance(emu: list[int], hw_samples: list[list[int]]) -> float | None:
    eg = gaps(emu)
    if not eg:
        return None
    dists = [wasserstein1(eg, hg) for hg in map(gaps, hw_samples) if hg]
    return min(dists) if dists else None


# --- loading -------------------------------------------------------------

@dataclass(frozen=True, order=True)
class RowKey:
    test: str
    compiler: str
    col: int
    row: int
    module: int
    event: str

    def tile(self) -> str:
        return f"({self.col},{self.row})/m{self.module}"


def _load_side(path: Path) -> dict[tuple, list[int]]:
    doc = json.loads(path.read_text())
    pl = doc.get("placement") or {}
    oc, orow = pl.get("origin_col", 0), pl.get("origin_row", 0)
    groups: dict[tuple, list[dict]] = {}
    for e in doc["events"]:
        k = (max(e["col"] - oc, 0), max(e["row"] - orow, 0), e["pkt_type"], e["name"])
        groups.setdefault(k, []).append(e)
    return {k: series_from_events(v, k[3]) for k, v in groups.items()}


def load_run(root: Path) -> dict[tuple[str, str], dict[str, dict[tuple, list[int]]]]:
    """{(test, compiler): {"hw": {rowkey: series}, "emu": {...}}} (sides may be absent)."""
    out: dict = {}
    for ev in sorted(Path(root).glob("*.*.*/events.json")):
        d = ev.parent.name
        stem, _, side = d.rpartition(".")
        if side not in ("hw", "emu"):
            continue
        test, _, comp = stem.rpartition(".")
        if not test:
            continue
        out.setdefault((test, comp), {})[side] = _load_side(ev)
    return out


# --- scoring -------------------------------------------------------------

@dataclass
class RowResult:
    key: RowKey
    hw_counts: list[int]
    hw_repeatable: bool
    emu_count_before: int
    emu_count_after: int
    count_before: int
    count_after: int
    pattern_before: float | None
    pattern_after: float | None
    exact_match_broken: bool
    verdict: str            # improved | regressed | neutral
    label: str
    gate_fail: bool

    @property
    def severity(self) -> tuple:
        pb = self.pattern_before if self.pattern_before is not None else 0
        pa = self.pattern_after if self.pattern_after is not None else 0
        return (self.gate_fail, self.count_after - self.count_before, pa - pb)


@dataclass
class Report:
    rows: list[RowResult] = field(default_factory=list)       # changed rows only
    rows_compared: int = 0
    only_in_baseline: list[tuple] = field(default_factory=list)
    only_in_candidate: list[tuple] = field(default_factory=list)
    skipped: list[tuple] = field(default_factory=list)

    @property
    def changed(self):
        return self.rows

    @property
    def exit_code(self) -> int:
        return 1 if any(r.gate_fail for r in self.rows) else 0

    def of(self, verdict):
        return sorted((r for r in self.rows if r.verdict == verdict),
                      key=lambda r: r.severity,
                      reverse=(verdict == "regressed"))

    def summary(self) -> dict:
        def tot(f):
            return sum(f(r) for r in self.rows)
        pat = [r for r in self.rows if r.pattern_before is not None and r.pattern_after is not None]
        return {
            "rows_compared": self.rows_compared,
            "rows_changed": len(self.rows),
            "improved": len(self.of("improved")),
            "regressed": len(self.of("regressed")),
            "neutral": len(self.of("neutral")),
            "gate_failures": sum(r.gate_fail for r in self.rows),
            "count_distance_before": tot(lambda r: r.count_before),
            "count_distance_after": tot(lambda r: r.count_after),
            "pattern_distance_before": round(sum(r.pattern_before for r in pat), 3),
            "pattern_distance_after": round(sum(r.pattern_after for r in pat), 3),
            "pattern_rows": len(pat),
        }


def _classify(cb, ca, pb, pa, hw_max, exact_broken):
    """-> (verdict, label, gate_fail) from count/pattern scores before/after."""
    if pb is None or pa is None:
        pat = "undefined"
    elif pa < pb - PATTERN_EPS:
        pat = "better"
    elif pa > pb + PATTERN_EPS:
        pat = "worse"
    else:
        pat = "same"
    if ca > cb:
        tol = max(PHASE_MIN_COUNT, PHASE_FRAC * hw_max)
        if pat == "better" and ca <= tol:
            return "neutral", "phase shift (pattern closer)", False
        why = {"better": "pattern closer but count off by more than the phase tolerance",
               "worse": "count and pattern both farther",
               "same": "count farther, pattern unchanged",
               "undefined": "count farther, no silicon gaps to compare"}[pat]
        # Count worse with an unchanged pattern is a soft regression unless it
        # broke a repeatable exact match; every other branch fails the gate.
        return "regressed", f"count regressed ({why})", pat != "same" or exact_broken
    if ca < cb:
        if pat == "worse":
            return "neutral", "mixed (count closer, pattern farther)", False
        return "improved", "count closer", False
    if pat == "better":
        return "improved", "pattern closer", False
    if pat == "worse":
        return "regressed", "pattern farther (count unchanged)", False
    return "neutral", "no score change", False


def compare_runs(baseline: Path, candidate: Path, only: list[str] | None = None) -> Report:
    base, cand = load_run(baseline), load_run(candidate)
    rep = Report()
    keep = (lambda t: True) if not only else (lambda t: t in only)
    rep.only_in_baseline = sorted(k for k in base.keys() - cand.keys() if keep(k[0]))
    rep.only_in_candidate = sorted(k for k in cand.keys() - base.keys() if keep(k[0]))
    for tc in sorted(base.keys() & cand.keys()):
        if not keep(tc[0]):
            continue
        b, c = base[tc], cand[tc]
        hw_runs = [r["hw"] for r in (b, c) if "hw" in r]
        if "emu" not in b or "emu" not in c or not hw_runs:
            rep.skipped.append(tc)
            continue
        keys = set(b["emu"]) | set(c["emu"])
        for r in hw_runs:
            keys |= set(r)
        for k in sorted(keys):
            hw = [r.get(k, []) for r in hw_runs]
            eb, ec = b["emu"].get(k, []), c["emu"].get(k, [])
            hw_counts = [len(s) for s in hw]
            rep.rows_compared += 1
            cb = band_distance(len(eb), hw_counts)
            ca = band_distance(len(ec), hw_counts)
            pb, pa = pattern_distance(eb, hw), pattern_distance(ec, hw)
            moved = (len(eb) != len(ec)
                     or (pb is not None and pa is not None and abs(pa - pb) > PATTERN_EPS))
            if not moved:
                continue
            exact = len(set(hw_counts)) == 1 and len(hw_counts) > 1 and cb == 0 and ca > 0
            verdict, label, gate = _classify(cb, ca, pb, pa, max(hw_counts), exact)
            rep.rows.append(RowResult(
                key=RowKey(tc[0], tc[1], k[0], k[1], k[2], k[3]),
                hw_counts=hw_counts,
                hw_repeatable=len(set(hw_counts)) == 1 and len(hw_counts) > 1,
                emu_count_before=len(eb), emu_count_after=len(ec),
                count_before=cb, count_after=ca,
                pattern_before=pb, pattern_after=pa,
                exact_match_broken=exact,
                verdict=verdict, label=label, gate_fail=gate))
    return rep


# --- output ----------------------------------------------------------------

def _fmt_p(p):
    return "n/a" if p is None else f"{p:.1f}"


def _line(r: RowResult) -> str:
    rep = "repeatable" if r.hw_repeatable else "HW varies" if len(r.hw_counts) > 1 else "1 HW sample"
    return (f"  {r.key.test}.{r.key.compiler} {r.key.tile()} {r.key.event}: "
            f"EMU {r.emu_count_before}->{r.emu_count_after} vs HW {r.hw_counts} ({rep}); "
            f"count dist {r.count_before}->{r.count_after}, "
            f"pattern W1 {_fmt_p(r.pattern_before)}->{_fmt_p(r.pattern_after)}; {r.label}"
            + ("  [GATE]" if r.gate_fail else ""))


def format_text(rep: Report) -> str:
    s = rep.summary()
    out = [
        f"rows compared: {s['rows_compared']}",
        f"rows changed: {s['rows_changed']}  improved: {s['improved']}  "
        f"regressed: {s['regressed']}  neutral: {s['neutral']}  gate failures: {s['gate_failures']}",
        f"count distance (sum):   {s['count_distance_before']} -> {s['count_distance_after']}",
        f"pattern distance (sum, {s['pattern_rows']} rows, cycles): "
        f"{s['pattern_distance_before']} -> {s['pattern_distance_after']}",
    ]
    for title, lst in (("only in baseline", rep.only_in_baseline),
                       ("only in candidate", rep.only_in_candidate),
                       ("skipped (missing HW or EMU events.json)", rep.skipped)):
        if lst:
            out.append(f"{title}: " + ", ".join(f"{t}.{c}" for t, c in lst))
    for verdict, title in (("regressed", "Regressions"), ("improved", "Improvements"),
                           ("neutral", "Neutral changes")):
        rows = rep.of(verdict)
        if rows:
            out += ["", f"{title} ({len(rows)}):"] + [_line(r) for r in rows]
    out.append("")
    out.append("RESULT: " + ("FAIL" if rep.exit_code else "ok"))
    return "\n".join(out)


def to_json(rep: Report) -> dict:
    def row(r):
        d = asdict(r)
        d["key"] = asdict(r.key)
        return d
    return {
        "summary": rep.summary(),
        "exit_code": rep.exit_code,
        "regressed": [row(r) for r in rep.of("regressed")],
        "improved": [row(r) for r in rep.of("improved")],
        "neutral": [row(r) for r in rep.of("neutral")],
        "only_in_baseline": [list(t) for t in rep.only_in_baseline],
        "only_in_candidate": [list(t) for t in rep.only_in_candidate],
        "skipped": [list(t) for t in rep.skipped],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--only", nargs="+", metavar="TEST", help="restrict to these tests")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    for d in (args.baseline, args.candidate):
        if not Path(d).is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2
    rep = compare_runs(Path(args.baseline), Path(args.candidate), only=args.only)
    print(json.dumps(to_json(rep), indent=1) if args.json else format_text(rep))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
