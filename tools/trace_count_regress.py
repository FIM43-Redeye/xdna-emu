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
shifted by the file's placement origin column so HW and EMU line up, rows are
physical and never shifted.  Its *series* is
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

Absent units and cut tails
--------------------------
A trace unit (tile, module) streams to the host in packets; on silicon the last, partial packet
is flushed by the stop event and is often not delivered.  A unit whose whole stream fits in that
one packet is then *absent* from the capture: no Start frame, no events.  That is not "recorded
zero events", and the gate keeps the two apart using the raw trace (a unit is carried when its
packets hold a Start frame; ``Side.units``):

* A silicon sample that lacks a row's unit contributes no count to that row's band.  Each row
  line says how many of the usable samples carry the unit ("unit carried by 2 of 3 HW samples").
  A unit that is carried and recorded zero events is a measured zero, as before.
* A row whose unit no usable silicon sample carries is *not measured on silicon*: listed with the
  emulator's counts, neither pass nor fail (a gap that stays visible, not a row that vanishes).
* A carried unit whose stream has no End token lost its tail (the events after the last full
  packet): "tail not delivered".  Reported per row and in the summary; counts of such samples
  are lower bounds and bias a band low.  They stay in the band unless ``--exclude-cut-tails`` is
  given, which leaves them out (a row with none left is then not measured).  Excluding is
  opt-in: on the banked suite it moved 43 bands (mostly the free-running PERF_CNT_2 of the core
  unit) but no gate verdict, and it would leave single-sample bands and unmeasured rows.
* The emulator side is not excused: a unit the candidate emulator trace lacks while silicon
  carries it fails the gate when the baseline emulator trace carried it; when both emulator
  traces lack it, it is listed as standing (a gap, not a regression).
  Without a usable ``trace_raw.bin`` a side's units cannot be told: all count as carried.
* The cross-module offset section already skips a sample that lacks either unit (no rows, no
  offset); it is unchanged.

Cross-module offset
-------------------
The compute tile's memory and core trace units are timed from different broadcast paths, so
the two disagree with silicon on time by a small fixed number of cycles, and a change to one
unit's timer moves every one of its rows by the same amount, which no per-row score above
sees.  ``mem_core_offset`` measures it (memory-unit minus core-unit stamp difference of
causally tied events, emulator minus silicon; its docstring has the method).  The gate
reports one row per (test, tile) with that offset for baseline and candidate over the same
silicon samples (the baseline and candidate dirs' own plus ``--hw-samples``), and the spread
of the silicon samples' own offsets.  The distance of a side from silicon is the distance of
zero to the band [min, max] of its per-sample offsets, so silicon's own spread is absorbed
the way the count band absorbs it.  A row is a *gate failure* when the candidate's distance
exceeds the baseline's by more than OFFSET_EPS (half a cycle: medians of even sample counts
land on half cycles); a smaller candidate distance is "better".  A timer offset is a few
cycles; a tile whose baseline or candidate offset exceeds OFFSET_GATE_MAX is one where the
tied events are separated by the workload's own run-to-run timing (silicon samples of such
tiles differ by hundreds to thousands of cycles), so a one-cycle move there is noise: it is
reported as "workload-dominated" and never fails.  A tile measurable on only
one side is "unmeasured" and never fails.  ``--no-offset`` leaves the section out.

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
import statistics
import struct
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_decoder import (  # noqa: E402
    StopCmd,
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
OFFSET_EPS = 0.5       # cycles a tile's cross-module offset may move away from silicon
OFFSET_GATE_MAX = 16   # beyond this many cycles an offset is workload, not timer: reported, not gated

import mem_core_offset  # noqa: E402  (the measurement the offset section is built on)

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

@dataclass(frozen=True)
class UnitState:
    """A trace unit that a capture carries (its stream has a Start frame).

    ``tail_cut`` is True when the stream has no End token (its last packet was not delivered, so
    the events after the last full packet are missing), False when it ends with one, and None
    when the unit's mode has no End token in the decoder (event-PC streams).
    """
    tail_cut: bool | None


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

    ``units``: {(col, row, pkt_type): UnitState} of the units the capture carries, or None when
    that cannot be told (no usable ``trace_raw.bin``: events.json cannot separate a unit that
    recorded zero events from one that is missing).
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
        # Only the column needs an origin: the run may be placed at a different start column, but
        # rows are physical (0 shim, 1 memtile, 2+ compute) and never shift.  The capture's
        # ``origin_row`` is the lowest traced row, which moves when one capture carries a shim or
        # memtile unit the other lacks, so it is not used.
        self._origin_col = pl.get("origin_col", 0)
        sn = doc.get("slot_names") or {}
        self._event_sets = {pt: tuple(sn[kind]) for pt, kind in _PT_CODE_TO_NAME.items()
                            if kind in sn}
        # (col, row, pkt_type, name, ts) of every event, level rows expanded where possible.
        rows, self._level_problem, units = _event_rows(self.path, doc)
        oc = self._origin_col
        self._units = (None if units is None else
                       {(max(c - oc, 0), r, pt): st for (c, r, pt), st in units.items()})
        groups: dict[tuple, list[int]] = {}
        for col, row, pt, name, ts in rows:
            groups.setdefault((max(col - oc, 0), row, pt, name), []).append(ts)
        self._series = {k: series_from_stamps(v, k[3]) for k, v in groups.items()}
        self._loaded = True

    @property
    def series(self) -> dict[tuple, list[int]]:
        self._load()
        return self._series

    @property
    def units(self) -> dict[tuple, UnitState] | None:
        self._load()
        return self._units

    def carries(self, unit: tuple) -> bool | None:
        """Whether the capture carries ``unit`` (col, row, pkt_type); None when it cannot be told."""
        units = self.units
        return None if units is None else unit in units

    def tail_cut(self, unit: tuple) -> bool | None:
        units = self.units
        return units[unit].tail_cut if units is not None and unit in units else None

    @property
    def level_problem(self) -> str | None:
        self._load()
        return self._level_problem

    @property
    def event_sets(self) -> dict[int, tuple]:
        self._load()
        return self._event_sets


def _unit_states(words: list[int], modes: dict, all_cmds: dict) -> dict:
    """{(pkt_type, row, col): UnitState} of the units with a Start frame in the raw words.

    A unit with no packet at all, or whose packets carry no Start, is absent and not listed.
    """
    out = {}
    for key, mode in modes.items():
        if mode is None:
            continue
        cmds = all_cmds.get(key) or []
        cut = None if not cmds else not any(isinstance(c, StopCmd) for c in cmds)
        out[key] = UnitState(tail_cut=cut if mode == TraceMode.EVENT_TIME else None)
    return out


def _event_rows(path: Path, doc: dict):
    """-> ([(col, row, pkt_type, name, ts)], level_problem, units) for one side.

    Mode-0 tiles come from ``trace_raw.bin`` with held levels expanded; every other tile (and
    every tile, when the raw trace cannot be used) comes from events.json.  ``units`` is
    {(col, row, pkt_type): UnitState} from the raw trace, None without a usable one.
    """
    original = [(e["col"], e["row"], e["pkt_type"], e["name"], e["ts"]) for e in doc["events"]]
    raw = path / "trace_raw.bin"
    if not raw.exists():
        return original, "no trace_raw.bin", None
    data = raw.read_bytes()
    words = list(struct.unpack(f"<{len(data) // 4}I", data[: len(data) // 4 * 4]))
    modes = detect_per_tile_modes(words)
    all_cmds = decode_words(words, mode=None)
    cmds = {k: v for k, v in all_cmds.items() if modes.get(k) == TraceMode.EVENT_TIME}
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
        return original, "trace_raw.bin does not reproduce events.json", None
    expanded = [(e.col, e.row, e.pkt_type, e.name, e.ts)
                for e in rebuild_timeline_mode0_levels(cmds, names)]
    rest = [r for r in original if (r[2], r[1], r[0]) not in cmds]
    units = {(c, r, pt): st for (pt, r, c), st in _unit_states(words, modes, all_cmds).items()}
    return expanded + rest, None, units


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
    hw_carrying: int        # silicon samples that carry the row's trace unit (the band's samples)
    hw_total: int           # usable silicon samples (same event set), carrying the unit or not
    hw_tail_cut: int        # of the carrying ones, those whose stream lost its tail

    @property
    def severity(self) -> tuple:
        pb = self.pattern_before if self.pattern_before is not None else 0
        pa = self.pattern_after if self.pattern_after is not None else 0
        return (self.gate_fail, self.count_after - self.count_before, pa - pb)


@dataclass
class NotMeasured:
    """A row no silicon sample measured: none of them carries the row's trace unit."""
    key: RowKey
    hw_total: int
    emu_before: int
    emu_after: int
    reason: str = "unit absent from every HW sample"


@dataclass
class UnitCoverage:
    """One trace unit (tile, module) of a test: who carries it.

    ``hw_*`` count the usable silicon samples; ``emu_baseline`` / ``emu_candidate`` say whether
    that emulator trace carries the unit (None: cannot be told, no raw trace).
    """
    test: str
    compiler: str
    col: int
    row: int
    module: int
    hw_total: int
    hw_carrying: int
    hw_tail_cut: int
    emu_baseline: bool | None
    emu_candidate: bool | None

    def tile(self) -> str:
        return f"({self.col},{self.row})/m{self.module}"


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

    # Cross-module offset rows, one per (test, tile); empty when switched off.
    offsets: list = field(default_factory=list)

    # Rows no silicon sample measured (the unit is absent from all of them): neither pass nor fail.
    not_measured: list = field(default_factory=list)
    # Per trace unit: how many silicon samples carry it, and how many of those lost their tail.
    units: list = field(default_factory=list)

    @property
    def emu_absent(self) -> list:
        """Units the candidate emulator trace lacks although silicon carries them."""
        return [u for u in self.units if u.emu_candidate is False and u.hw_carrying > 0]

    @property
    def emu_lost(self) -> list:
        """The emu_absent units the baseline emulator trace did carry (or could not be told): gate
        failures, since an emulator that stops tracing a unit would otherwise read as a silent
        zero.  A unit both emulator traces lack is a standing gap: listed, not a regression."""
        return [u for u in self.emu_absent if u.emu_baseline is not False]

    @property
    def changed(self):
        return self.rows

    @property
    def exit_code(self) -> int:
        return 1 if (any(r.gate_fail for r in self.rows) or any(o.gate_fail for o in self.offsets)
                     or self.emu_lost) else 0

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
            "rows_not_measured": len(self.not_measured),
            "units_absent_from_hw_samples": sum(1 for u in self.units if u.hw_carrying < u.hw_total),
            "hw_samples_without_unit": sum(u.hw_total - u.hw_carrying for u in self.units),
            "hw_samples_tail_cut": sum(u.hw_tail_cut for u in self.units),
            "emu_units_absent": len(self.emu_absent),
            "emu_units_lost": len(self.emu_lost),
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


@dataclass
class OffsetRow:
    """One (test, tile): memory-minus-core offset, emulator against silicon, in cycles."""
    test: str
    compiler: str
    tile: tuple
    spread: float | None           # max - min of the silicon samples' offsets (candidate's set)
    baseline: float | None         # median over silicon samples; None when unmeasurable
    candidate: float | None
    distance_baseline: float | None
    distance_candidate: float | None
    verdict: str                   # better / worse / unchanged / unmeasured
    gate_fail: bool


def band_offset_distance(offsets: list[float]) -> float:
    """Distance of zero (the emulator agreeing with silicon) from the band the per-sample offsets span."""
    lo, hi = min(offsets), max(offsets)
    return 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))


def compare_offsets(baseline: Path, candidate: Path, silicon_dirs: list[Path],
                    only: list[str] | None = None) -> list[OffsetRow]:
    dirs = [baseline, candidate, *silicon_dirs]
    b = mem_core_offset.measure_samples(baseline, dirs, only or ())
    c = mem_core_offset.measure_samples(candidate, dirs, only or ())
    rows = []
    for key in sorted(b.keys() | c.keys(), key=lambda k: (k[0], k[1])):
        test, _, comp = key[0].rpartition(".")
        bs, cs = b.get(key), c.get(key)
        if not bs or not cs:
            rows.append(OffsetRow(test, comp, key[1], None, statistics_median(bs), statistics_median(cs),
                                  None, None, "unmeasured", False))
            continue
        db, dc = band_offset_distance(bs), band_offset_distance(cs)
        if max(abs(statistics.median(bs)), abs(statistics.median(cs))) > OFFSET_GATE_MAX:
            rows.append(OffsetRow(test, comp, key[1], max(cs) - min(cs), statistics.median(bs),
                                  statistics.median(cs), db, dc, "workload-dominated", False))
            continue
        verdict = ("worse" if dc - db > OFFSET_EPS else "better" if db - dc > OFFSET_EPS
                   else "unchanged")
        rows.append(OffsetRow(test, comp, key[1], max(cs) - min(cs), statistics_median(bs),
                              statistics_median(cs), db, dc, verdict, verdict == "worse"))
    return rows


def statistics_median(v):
    return statistics.median(v) if v else None


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
                 hw_samples: list[Path] | None = None, offsets: bool = True,
                 exclude_cut_tails: bool = False) -> Report:
    base, cand = load_run(baseline), load_run(candidate)
    extra = [(d, load_run(d)) for d in (hw_samples or [])]
    rep = Report()
    if offsets:
        rep.offsets = compare_offsets(baseline, candidate, hw_samples or [], only)
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
        _unit_coverage(rep, tc, usable, eb_side, ec_side, keys)
        for k in sorted(keys):
            rk = RowKey(tc[0], tc[1], k[0], k[1], k[2], k[3])
            hw_sides = usable[k[2]]
            if not hw_sides:
                rep.unscored.append((rk, "no silicon sample with the candidate EMU's event set"))
                continue
            if eb_side.event_sets.get(k[2]) != ec_side.event_sets.get(k[2]):
                rep.unscored.append((rk, "baseline and candidate EMU traced different event sets"))
                continue
            # Only samples that carry the unit measured anything: an absent unit is not a zero.
            carrying = [(lbl, side) for lbl, side in hw_sides if side.carries(k[:3]) is not False]
            if not carrying:
                rep.not_measured.append(NotMeasured(
                    rk, len(hw_sides), len(eb_side.series.get(k, [])), len(ec_side.series.get(k, []))))
                continue
            hw_sides_all = hw_sides
            hw_sides = [(lbl, side) for lbl, side in carrying
                        if not (exclude_cut_tails and side.tail_cut(k[:3]) is True)]
            if not hw_sides:
                rep.not_measured.append(NotMeasured(
                    rk, len(hw_sides_all), len(eb_side.series.get(k, [])), len(ec_side.series.get(k, [])),
                    "tail not delivered in every sample that carries the unit"))
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
                verdict=verdict, label=label, gate_fail=gate,
                hw_carrying=len(carrying), hw_total=len(hw_sides_all),
                hw_tail_cut=sum(side.tail_cut(k[:3]) is True for _, side in carrying)))
    return rep


def _unit_coverage(rep: Report, tc: tuple, usable: dict, eb_side: Side, ec_side: Side, keys: set):
    """Record, per trace unit of one (test, compiler), who carries it (see UnitCoverage)."""
    units = {k[:3] for k in keys}
    for side in (eb_side, ec_side, *[s for lst in usable.values() for _, s in lst]):
        units |= set(side.units or ())
    for unit in sorted(units):
        samples = usable[unit[2]]
        if not samples:
            continue
        carrying = [s for _, s in samples if s.carries(unit) is not False]
        rep.units.append(UnitCoverage(
            test=tc[0], compiler=tc[1], col=unit[0], row=unit[1], module=unit[2],
            hw_total=len(samples), hw_carrying=len(carrying),
            hw_tail_cut=sum(s.tail_cut(unit) is True for s in carrying),
            emu_baseline=eb_side.carries(unit), emu_candidate=ec_side.carries(unit)))


# --- output ----------------------------------------------------------------

def _fmt_p(p):
    return "n/a" if p is None else f"{p:.1f}"


def _unit_note(carrying: int, total: int, cut: int, used: int) -> str:
    """The suffix of a row line when its unit is missing from some samples or lost a tail."""
    if carrying == total and not cut:
        return ""
    bits = [f"carried by {carrying} of {total} HW samples"]
    if cut:
        bits.append(f"tail not delivered in {cut}")
    if used < carrying:
        bits.append(f"{carrying - used} of them left out of the band")
    return "; unit " + ", ".join(bits)


def _line(r: RowResult) -> str:
    n = len(r.hw_counts)
    rep = (f"{n} HW samples, repeatable" if r.hw_repeatable
           else f"{n} HW samples, HW varies" if n > 1 else "1 HW sample")
    return (f"  {r.key.test}.{r.key.compiler} {r.key.tile()} {r.key.event}: "
            f"EMU {r.emu_count_before}->{r.emu_count_after} vs HW {r.hw_counts} ({rep}); "
            f"count dist {r.count_before}->{r.count_after}, "
            f"pattern W1 {_fmt_p(r.pattern_before)}->{_fmt_p(r.pattern_after)}; {r.label}"
            + _unit_note(r.hw_carrying, r.hw_total, r.hw_tail_cut, len(r.hw_counts))
            + ("  [GATE]" if r.gate_fail else ""))


def _g(v):
    return "-" if v is None else f"{v:g}"


def offset_section(rows: list[OffsetRow]) -> list[str]:
    ok = [r for r in rows if r.verdict != "unmeasured"]
    out = ["", f"Cross-module offset, memory minus core, emulator vs silicon, cycles ({len(rows)} test-tiles):"]
    for r in rows:
        out.append(f"  {r.test}.{r.compiler} tile{r.tile}: silicon spread {_g(r.spread)}; "
                   f"baseline {_g(r.baseline)}, candidate {_g(r.candidate)}; {r.verdict}"
                   + ("  [GATE]" if r.gate_fail else ""))
    if ok:
        near = lambda f: sum(1 for r in ok if abs(f(r)) < 0.5)
        out.append(f"cross-module offset: {len(ok)} tiles measured; median candidate offset "
                   f"{statistics.median(r.candidate for r in ok):g}; within 0.5 of silicon: "
                   f"baseline {near(lambda r: r.baseline)}, candidate {near(lambda r: r.candidate)}; "
                   f"{sum(r.gate_fail for r in ok)} gate failures"
                   f"; {len(rows) - len(ok)} unmeasured; "
                   f"{sum(r.verdict == 'workload-dominated' for r in ok)} workload-dominated (not gated)")
    else:
        out.append(f"cross-module offset: no tile measurable ({len(rows)} unmeasured)")
    return out


def format_text(rep: Report) -> str:
    s = rep.summary()
    out = [
        f"rows compared: {s['rows_compared']}",
        f"rows changed: {s['rows_changed']}  improved: {s['improved']}  "
        f"regressed: {s['regressed']}  neutral: {s['neutral']}  gate failures: {s['gate_failures']}",
        f"unscored rows: {s['rows_unscored']}  not measured on silicon: {s['rows_not_measured']}",
        f"trace units: {len(rep.units)}; absent from some HW samples: {s['units_absent_from_hw_samples']} "
        f"({s['hw_samples_without_unit']} unit-samples); HW streams with the tail not delivered: "
        f"{s['hw_samples_tail_cut']}; absent from the candidate emulator but carried by silicon: "
        f"{s['emu_units_absent']} ({s['emu_units_lost']} newly, gated)",
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
    if rep.emu_absent:
        out += ["", f"Trace units absent from the candidate emulator trace although silicon carries "
                    f"them ({len(rep.emu_absent)}):"] + [
            f"  {u.test}.{u.compiler} {u.tile()}: silicon carries it in {u.hw_carrying} of "
            f"{u.hw_total} samples; baseline emulator "
            f"{'carries it' if u.emu_baseline else 'lacks it' if u.emu_baseline is False else '?'}"
            + ("  [GATE]" if u in rep.emu_lost else "  (standing, not gated)")
            for u in rep.emu_absent]
    if rep.not_measured:
        out += ["", f"Rows not measured on silicon ({len(rep.not_measured)}): no HW sample carries "
                    f"the unit; neither pass nor fail"] + [
            f"  {n.key.test}.{n.key.compiler} {n.key.tile()} {n.key.event}: EMU "
            f"{n.emu_before}->{n.emu_after}, {n.reason} ({n.hw_total} HW samples)"
            for n in rep.not_measured]
    if rep.unscored:
        out += ["", f"Unscored rows ({len(rep.unscored)}):"] + [
            f"  {k.test}.{k.compiler} {k.tile()} {k.event}: {why}" for k, why in rep.unscored]
    if rep.offsets:
        out += offset_section(rep.offsets)
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
        "offsets": [asdict(o) for o in rep.offsets],
        "not_measured": [dict(asdict(n), key=asdict(n.key)) for n in rep.not_measured],
        "units": [asdict(u) for u in rep.units],
        "emu_absent": [asdict(u) for u in rep.emu_absent],
        "emu_lost": [asdict(u) for u in rep.emu_lost],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--only", nargs="+", metavar="TEST", help="restrict to these tests")
    ap.add_argument("--hw-samples", nargs="+", metavar="DIR", default=[],
                    help="further result dirs whose HW samples count as silicon samples "
                         "(identical captures and other event sets are dropped)")
    ap.add_argument("--no-offset", action="store_true",
                    help="leave out the cross-module (memory minus core) offset section")
    ap.add_argument("--exclude-cut-tails", action="store_true",
                    help="leave HW samples whose stream lost its tail out of the count band "
                         "(see 'Absent units and cut tails' in the module docstring)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    for d in (args.baseline, args.candidate, *args.hw_samples):
        if not Path(d).is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2
    rep = compare_runs(Path(args.baseline), Path(args.candidate), only=args.only,
                       hw_samples=[Path(d) for d in args.hw_samples],
                       offsets=not args.no_offset, exclude_cut_tails=args.exclude_cut_tails)
    print(json.dumps(to_json(rep), indent=1) if args.json else format_text(rep))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
