#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bridge-suite regression gate: compare two result dirs on trace counts AND timing.

``scripts/emu-bridge-test.sh`` writes ``<test>.<compiler>.{hw,emu}/{events.json,trace_raw.bin}``
into ``build/bridge-test-results/YYYYMMDD/``.  Its CLEAN/DIVERGE verdict
anchors each tile on one event and compares pairs, so it misses a change that
adds phantom intervals on an otherwise quiet row (EMU STREAM_STARVATION went
32 -> 62 against silicon's 32).  This tool scores every (test, compiler, tile,
event) row of a baseline run against a candidate run, always measured against
the silicon samples of both days (plus any given with ``--hw-samples``).

Rows
----
A row is (test, compiler, tile col/row, tile module, event name); columns are
shifted by the file's placement origin so HW and EMU line up.  Its *series* is
the list of event timestamps; a row's count is the length of its series.  A
side that never fires a row has count 0, so a phantom row (EMU fires, silicon
does not) is a plain 0 -> N count move.

Level rows are scored on true assertion intervals, not on events.json
-------------------------------------------------------------------
Level events (stalls, starvation, backpressure, port running, bank conflicts: high for as
many cycles as the condition holds) are scored as intervals, the series being the interval
start times.  events.json cannot give those: the decoder follows the mlir-aie convention, in
which a Repeat after a zero-cycle frame only advances the timer and emits no event, and that
is exactly how silicon and the emulator write a held level (open frame, one (mask, 0) frame,
Repeat).  A held level therefore appears in events.json only at the frames where the tile's
8-bit event mask changes, so the "interval count" read from it is the number of mask
changes caused by *other* traced events while the level was high: it moves when an unrelated
event shifts by a cycle, and not at all when the level itself does.  So the level rows are
rebuilt from ``trace_raw.bin`` (``trace_decoder.rebuild_timeline_mode0_levels``, the same
walk as the events.json rebuild plus one event per held cycle) and consecutive cycles
collapse into one interval; a one-cycle drop makes two.  Edge rows (one event per
occurrence) keep reading events.json, and tiles in EVENT_PC mode keep their events.json
events (their timestamps are program counters; there is no held level to expand).  A level
row whose side lacks a usable ``trace_raw.bin`` (missing, or one that does not reproduce
that side's events.json) is *not scored*: it is listed as unscored with the reason, never
counted from events.json.

Silicon samples
---------------
The silicon of a (test, compiler) is every HW sample of the baseline dir, the candidate dir
and each ``--hw-samples`` dir.  Samples with identical content (same ``trace_raw.bin``, or
same events.json when there is none) are one sample: a result dir that was copied forward
must not count twice.  A sample counts for a row only when its traced event set for the row's
tile module (events.json ``slot_names``) equals the candidate EMU's: a different event set
is a different measurement.  Dropped samples are listed with the reason.

Scores (both are distances to silicon, lower is better)
-------------------------------------------------------
* Count score: distance of the EMU count from the silicon band, 0 inside: the
  closed range [min, max] of the HW counts over the usable HW samples,
  widened on each side by its own spread (``silicon_band``), since a few samples
  underestimate silicon's run-to-run spread and a 1-event EMU move inside it is
  not a regression. A repeatable row (all samples equal) stays exact. The band
  is the same for baseline and candidate scoring, so only the EMU change moves
  the score.  ``hw_repeatable`` is True when two or more HW samples have the
  same count.
* Pattern score: 1-Wasserstein distance (cycles) between the EMU's and HW's
  inter-event gap distributions.  Gaps are shift invariant (no anchoring),
  and W1 is in cycles, so "gap 128 vs silicon 140" scores 12 and "131" scores
  9: a closer pattern is a smaller number, and extra or missing events change
  the score only through the gaps they add.  Distributions of different
  length are compared as quantile functions (exact integral of
  |Q_emu - Q_hw| over [0, 1]), which is the textbook W1 and needs no
  resampling.  Ordering of gaps is deliberately ignored: a systematic shift of
  the cadence is what we care about.  With several HW samples the score is the
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
usage errors.  A "repeatable" match needs two HW samples that agree.  Unscored rows never
fail the gate, but are counted in the summary.  Tests present in only one run, or lacking
either side's events.json, are listed but not scored.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_decoder import (  # noqa: E402
    TraceMode,
    decode_words,
    detect_per_tile_modes,
    rebuild_timeline_mode0,
    rebuild_timeline_mode0_levels,
)
from trace_decoder.decode import _PT_CODE_TO_NAME  # noqa: E402

PATTERN_EPS = 0.5      # cycles of W1 movement below which the pattern is "unchanged"
PHASE_FRAC = 0.10      # phase shift allowed to move the count by this fraction of HW max
PHASE_MIN_COUNT = 2    # ... but at least this many events/intervals

# Level events: high for as many cycles as the condition holds.  This mirrors
# crates/xdna-archspec trace_events::is_level_event by name family; validated
# against the [level]/[edge] tags of every trace.log in the real results.
_LEVEL_RE = re.compile(
    r"(STALL|STARVATION|BACKPRESSURE|PORT_RUNNING|CONFLICT_DM_BANK|^ACTIVE$|^DISABLED$)"
)


def is_level_name(name: str) -> bool:
    return bool(_LEVEL_RE.search(name))


# --- series and distances -----------------------------------------------

def series_from_stamps(stamps: list[int], name: str) -> list[int]:
    """Sorted timestamps; level events collapse to interval start times.

    For a level the stamps must be one per asserted cycle (see the module docstring), or the
    "intervals" are not intervals."""
    ts = sorted(stamps)
    if not is_level_name(name) or not ts:
        return ts
    starts = [ts[0]]
    for prev, cur in zip(ts, ts[1:]):
        if cur - prev > 1:
            starts.append(cur)
    return starts


def series_from_events(events: list[dict], name: str) -> list[int]:
    return series_from_stamps([e["ts"] for e in events], name)


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


class Side:
    """One ``<test>.<compiler>.<hw|emu>`` result dir, decoded on first use.

    ``series``: {(col, row, pkt_type, event): [timestamps]} with level rows rebuilt from
    ``trace_raw.bin`` into one event per asserted cycle (so the series of a level is its interval
    start times).  ``level_problem`` is None when that worked, else why level rows of this side
    cannot be scored.  ``event_sets``: {pkt_type: slot names} as traced.  ``digest`` identifies
    the capture by content.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._loaded = False

    @property
    def digest(self) -> str:
        raw = self.path / "trace_raw.bin"
        f = raw if raw.exists() else self.path / "events.json"
        return hashlib.sha256(f.read_bytes()).hexdigest()

    def _load(self):
        if self._loaded:
            return
        doc = json.loads((self.path / "events.json").read_text())
        pl = doc.get("placement") or {}
        self._origin = (pl.get("origin_col", 0), pl.get("origin_row", 0))
        sn = doc.get("slot_names") or {}
        self._event_sets = {pt: tuple(sn[kind]) for pt, kind in _PT_CODE_TO_NAME.items()
                            if kind in sn}
        # (col, row, pkt_type, name, ts) of every event, level rows expanded where possible.
        rows, self._level_problem = _event_rows(self.path, doc)
        groups: dict[tuple, list[int]] = {}
        oc, orow = self._origin
        for col, row, pt, name, ts in rows:
            groups.setdefault((max(col - oc, 0), max(row - orow, 0), pt, name), []).append(ts)
        self._series = {k: series_from_stamps(v, k[3]) for k, v in groups.items()}
        self._loaded = True

    @property
    def series(self) -> dict[tuple, list[int]]:
        self._load()
        return self._series

    @property
    def level_problem(self) -> str | None:
        self._load()
        return self._level_problem

    @property
    def event_sets(self) -> dict[int, tuple]:
        self._load()
        return self._event_sets


def _event_rows(path: Path, doc: dict):
    """-> ([(col, row, pkt_type, name, ts)], level_problem) for one side.

    Mode-0 tiles come from ``trace_raw.bin`` with held levels expanded; every other tile (and
    every tile, when the raw trace cannot be used) comes from events.json.
    """
    original = [(e["col"], e["row"], e["pkt_type"], e["name"], e["ts"]) for e in doc["events"]]
    raw = path / "trace_raw.bin"
    if not raw.exists():
        return original, "no trace_raw.bin"
    data = raw.read_bytes()
    words = list(struct.unpack(f"<{len(data) // 4}I", data[: len(data) // 4 * 4]))
    modes = detect_per_tile_modes(words)
    cmds = {k: v for k, v in decode_words(words, mode=None).items()
            if modes.get(k) == TraceMode.EVENT_TIME}
    sn = doc.get("slot_names") or {}
    names = {pt: {f"{r},{c}": sn.get(_PT_CODE_TO_NAME[pt], [""] * 8) for (p, r, c) in cmds
                  if p == pt} for pt in _PT_CODE_TO_NAME}
    # The raw trace must reproduce this side's events.json, or it is a different capture.
    plain = collections.Counter((e.col, e.row, e.pkt_type, e.slot, e.ts)
                                for e in rebuild_timeline_mode0(cmds, names))
    want = collections.Counter((e["col"], e["row"], e["pkt_type"], e["slot"], e["ts"])
                               for e in doc["events"]
                               if (e["pkt_type"], e["row"], e["col"]) in cmds)
    if plain != want:
        return original, "trace_raw.bin does not reproduce events.json"
    expanded = [(e.col, e.row, e.pkt_type, e.name, e.ts)
                for e in rebuild_timeline_mode0_levels(cmds, names)]
    rest = [r for r in original if (r[2], r[1], r[0]) not in cmds]
    return expanded + rest, None


def load_run(root: Path) -> dict[tuple[str, str], dict[str, Side]]:
    """{(test, compiler): {"hw": Side, "emu": Side}} (sides may be absent)."""
    out: dict = {}
    for ev in sorted(Path(root).glob("*.*.*/events.json")):
        d = ev.parent.name
        stem, _, side = d.rpartition(".")
        if side not in ("hw", "emu"):
            continue
        test, _, comp = stem.rpartition(".")
        if not test:
            continue
        out.setdefault((test, comp), {})[side] = Side(ev.parent)
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
    # (RowKey, reason): level rows that could not be scored.
    unscored: list[tuple] = field(default_factory=list)
    # (test, compiler, sample label, reason): silicon samples not used.
    dropped_samples: list[tuple] = field(default_factory=list)

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
            "rows_unscored": len(self.unscored),
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


def _usable_hw_samples(tc, b, c, extra, rep):
    """The distinct HW samples of one (test, compiler): [(label, Side)] in day order."""
    cands = [(str(d), r[tc]["hw"]) for d, r in extra if tc in r and "hw" in r[tc]]
    cands = [(lbl, r["hw"]) for lbl, r in (("baseline", b), ("candidate", c)) if "hw" in r] + cands
    seen: dict[str, str] = {}
    out = []
    for label, side in cands:
        dig = side.digest
        if dig in seen:
            rep.dropped_samples.append((*tc, label, f"identical to {seen[dig]}"))
            continue
        seen[dig] = label
        out.append((label, side))
    return out


def compare_runs(baseline: Path, candidate: Path, only: list[str] | None = None,
                 hw_samples: list[Path] | None = None) -> Report:
    base, cand = load_run(baseline), load_run(candidate)
    extra = [(d, load_run(d)) for d in (hw_samples or [])]
    rep = Report()
    keep = (lambda t: True) if not only else (lambda t: t in only)
    rep.only_in_baseline = sorted(k for k in base.keys() - cand.keys() if keep(k[0]))
    rep.only_in_candidate = sorted(k for k in cand.keys() - base.keys() if keep(k[0]))
    for tc in sorted(base.keys() & cand.keys()):
        if not keep(tc[0]):
            continue
        b, c = base[tc], cand[tc]
        samples = _usable_hw_samples(tc, b, c, extra, rep)
        if "emu" not in b or "emu" not in c or not samples:
            rep.skipped.append(tc)
            continue
        eb_side, ec_side = b["emu"], c["emu"]
        # A sample measures a tile module only with the event set the candidate EMU traced.
        usable = {pt: [] for pt in _PT_CODE_TO_NAME}
        for label, side in samples:
            for pt in usable:
                if side.event_sets.get(pt) == ec_side.event_sets.get(pt):
                    usable[pt].append((label, side))
                elif pt in side.event_sets or pt in ec_side.event_sets:
                    rep.dropped_samples.append(
                        (*tc, label, f"event set of module {pt} differs from the candidate EMU's"))
        keys = set(eb_side.series) | set(ec_side.series)
        for pt, lst in usable.items():
            for _, side in lst:
                keys |= {k for k in side.series if k[2] == pt}
        for k in sorted(keys):
            rk = RowKey(tc[0], tc[1], k[0], k[1], k[2], k[3])
            hw_sides = usable[k[2]]
            if not hw_sides:
                rep.unscored.append((rk, "no silicon sample with the candidate EMU's event set"))
                continue
            if eb_side.event_sets.get(k[2]) != ec_side.event_sets.get(k[2]):
                rep.unscored.append((rk, "baseline and candidate EMU traced different event sets"))
                continue
            if is_level_name(k[3]):
                bad = [f"{lbl}: {side.level_problem}"
                       for lbl, side in [("baseline emu", eb_side), ("candidate emu", ec_side)]
                       + hw_sides if side.level_problem]
                if bad:
                    rep.unscored.append((rk, "level not reconstructable (" + "; ".join(bad) + ")"))
                    continue
            hw = [side.series.get(k, []) for _, side in hw_sides]
            eb, ec = eb_side.series.get(k, []), ec_side.series.get(k, [])
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
                key=rk,
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
    n = len(r.hw_counts)
    rep = (f"{n} HW samples, repeatable" if r.hw_repeatable
           else f"{n} HW samples, HW varies" if n > 1 else "1 HW sample")
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
        f"unscored rows: {s['rows_unscored']}",
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
    if rep.dropped_samples:
        out += ["", f"Silicon samples not used ({len(rep.dropped_samples)}):"] + [
            f"  {t}.{c} {lbl}: {why}" for t, c, lbl, why in rep.dropped_samples]
    if rep.unscored:
        out += ["", f"Unscored rows ({len(rep.unscored)}):"] + [
            f"  {k.test}.{k.compiler} {k.tile()} {k.event}: {why}" for k, why in rep.unscored]
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
        "unscored": [{"key": asdict(k), "reason": why} for k, why in rep.unscored],
        "dropped_samples": [list(t) for t in rep.dropped_samples],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--only", nargs="+", metavar="TEST", help="restrict to these tests")
    ap.add_argument("--hw-samples", nargs="+", metavar="DIR", default=[],
                    help="further result dirs whose HW samples count as silicon samples "
                         "(identical captures and other event sets are dropped)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    for d in (args.baseline, args.candidate, *args.hw_samples):
        if not Path(d).is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2
    rep = compare_runs(Path(args.baseline), Path(args.candidate), only=args.only,
                       hw_samples=[Path(d) for d in args.hw_samples])
    print(json.dumps(to_json(rep), indent=1) if args.json else format_text(rep))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
