#!/usr/bin/env python3
"""Three-way timing-calibration comparator (Stage 2, task #72).

Joins per-kernel timing records from three sources -- real HW, the interpreter
emulator, and AMD's standalone aiesimulator -- and reports cycle drift with HW
as ground truth. See docs/coverage/three-way-timing-calibration.md.

Each input is a "timing record" JSON file following the data contract:

    {
      "kernel": "add_one_using_dma",
      "compiler": "chess",
      "source": "hw" | "interp" | "aiesim",
      "total_cycles": 12345,
      "anchors": [ {"col":1,"row":2,"kind":"dma_s2mm0_start","cycle":1000}, ... ]
    }

`anchors` is optional and unused by the Option-C (total-cycle) report; the
Option-B per-anchor report consumes it. Records are matched on (kernel,
compiler); the `hw` record is the ground truth each row is measured against.

Usage:
    timing-three-way.py --records <dir-or-glob> [--json] [-o out.txt]
    timing-three-way.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

SOURCES = ("hw", "interp", "aiesim")


@dataclass
class TimingRecord:
    kernel: str
    compiler: str
    source: str
    total_cycles: Optional[int]
    anchors: list = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict, origin: str = "<dict>") -> "TimingRecord":
        for key in ("kernel", "compiler", "source"):
            if key not in d:
                raise ValueError(f"{origin}: timing record missing '{key}'")
        src = d["source"]
        if src not in SOURCES:
            raise ValueError(f"{origin}: unknown source '{src}' (want one of {SOURCES})")
        tc = d.get("total_cycles", None)
        if tc is not None:
            tc = int(tc)
        return TimingRecord(
            kernel=str(d["kernel"]),
            compiler=str(d["compiler"]),
            source=src,
            total_cycles=tc,
            anchors=list(d.get("anchors", [])),
        )


@dataclass
class DriftRow:
    kernel: str
    compiler: str
    hw: Optional[int]
    interp: Optional[int]
    aiesim: Optional[int]

    def drift(self, source_cycles: Optional[int]) -> Optional[float]:
        """Percent drift of `source_cycles` vs HW ground truth, or None."""
        if self.hw is None or self.hw == 0 or source_cycles is None:
            return None
        return (source_cycles - self.hw) / self.hw * 100.0


def load_records(paths: list[str]) -> list[TimingRecord]:
    records: list[TimingRecord] = []
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        # A file may hold a single record or a list of records.
        items = data if isinstance(data, list) else [data]
        for item in items:
            records.append(TimingRecord.from_dict(item, origin=path))
    return records


def expand_record_paths(spec: str) -> list[str]:
    """Resolve a directory or glob into a sorted list of *.timing.json files."""
    if os.path.isdir(spec):
        spec = os.path.join(spec, "*.timing.json")
    return sorted(glob.glob(spec))


def build_drift_rows(records: list[TimingRecord]) -> list[DriftRow]:
    """Group records by (kernel, compiler) into one DriftRow each."""
    grouped: dict[tuple[str, str], dict[str, Optional[int]]] = {}
    for r in records:
        key = (r.kernel, r.compiler)
        grouped.setdefault(key, {})[r.source] = r.total_cycles
    rows = []
    for (kernel, compiler), by_source in sorted(grouped.items()):
        rows.append(
            DriftRow(
                kernel=kernel,
                compiler=compiler,
                hw=by_source.get("hw"),
                interp=by_source.get("interp"),
                aiesim=by_source.get("aiesim"),
            )
        )
    return rows


def _fmt_cycles(v: Optional[int]) -> str:
    return "-" if v is None else str(v)


def _fmt_drift(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:+.1f}%"


def text_report(rows: list[DriftRow]) -> str:
    lines = []
    lines.append("Three-way timing calibration (total cycles, HW = ground truth)")
    lines.append("")
    header = f"{'kernel':<48} {'comp':<6} {'HW':>9} {'interp':>9} {'aiesim':>9} {'iΔ':>8} {'aΔ':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row.kernel:<48} {row.compiler:<6} "
            f"{_fmt_cycles(row.hw):>9} {_fmt_cycles(row.interp):>9} {_fmt_cycles(row.aiesim):>9} "
            f"{_fmt_drift(row.drift(row.interp)):>8} {_fmt_drift(row.drift(row.aiesim)):>8}"
        )

    # Summary: mean absolute drift over rows where the source and HW both exist.
    def mean_abs(selector) -> Optional[float]:
        vals = [abs(d) for r in rows for d in [r.drift(selector(r))] if d is not None]
        return sum(vals) / len(vals) if vals else None

    lines.append("")
    interp_mad = mean_abs(lambda r: r.interp)
    aiesim_mad = mean_abs(lambda r: r.aiesim)
    lines.append(f"mean |drift| vs HW:  interp={_fmt_drift(interp_mad)}  aiesim={_fmt_drift(aiesim_mad)}")
    n_hw = sum(1 for r in rows if r.hw is not None)
    n_ai = sum(1 for r in rows if r.aiesim is not None)
    lines.append(f"rows={len(rows)}  with_hw={n_hw}  with_aiesim={n_ai}")
    return "\n".join(lines) + "\n"


def json_report(rows: list[DriftRow]) -> str:
    out = []
    for row in rows:
        out.append(
            {
                "kernel": row.kernel,
                "compiler": row.compiler,
                "hw": row.hw,
                "interp": row.interp,
                "aiesim": row.aiesim,
                "interp_drift_pct": row.drift(row.interp),
                "aiesim_drift_pct": row.drift(row.aiesim),
            }
        )
    return json.dumps(out, indent=2) + "\n"


# ---------------------------------------------------------------------------
# Option B: per-anchor report
# ---------------------------------------------------------------------------
#
# Each source counts cycles from a different origin (sim power-on vs trace
# enable), so absolute anchor cycles are not comparable. We normalize each
# source to ITS earliest anchor (the calibration-doc decision) -- setting that
# source's first activity to cycle 0 -- then compare anchors on (col, row, kind).
# This assumes the earliest anchor is the same physical event across sources,
# which holds when the first DMA to arm is the same on both; the per-anchor
# deltas are therefore relative to each source's first activity.

AnchorKey = tuple  # (col:int, row:int, kind:str)


def normalize_anchors(record: TimingRecord) -> dict:
    """Return {(col,row,kind): relative_cycle}, normalized to the record's
    earliest anchor. Empty if the record carries no anchors."""
    if not record.anchors:
        return {}
    cycles = [int(a["cycle"]) for a in record.anchors]
    base = min(cycles)
    out = {}
    for a in record.anchors:
        key = (int(a["col"]), int(a["row"]), str(a["kind"]))
        out[key] = int(a["cycle"]) - base
    return out


@dataclass
class AnchorRow:
    kernel: str
    compiler: str
    col: int
    row: int
    kind: str
    hw: Optional[int]
    interp: Optional[int]
    aiesim: Optional[int]

    def delta(self, rel: Optional[int]) -> Optional[int]:
        if self.hw is None or rel is None:
            return None
        return rel - self.hw


def build_anchor_rows(records: list[TimingRecord]) -> list[AnchorRow]:
    """Group records by (kernel, compiler) and align their anchors on
    (col, row, kind), each source normalized to its own earliest anchor."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for r in records:
        grouped.setdefault((r.kernel, r.compiler), {})[r.source] = normalize_anchors(r)

    rows: list[AnchorRow] = []
    for (kernel, compiler), by_source in sorted(grouped.items()):
        hw = by_source.get("hw", {})
        interp = by_source.get("interp", {})
        aiesim = by_source.get("aiesim", {})
        keys = set(hw) | set(interp) | set(aiesim)
        for key in sorted(keys):
            col, row, kind = key
            rows.append(
                AnchorRow(
                    kernel=kernel,
                    compiler=compiler,
                    col=col,
                    row=row,
                    kind=kind,
                    hw=hw.get(key),
                    interp=interp.get(key),
                    aiesim=aiesim.get(key),
                )
            )
    return rows


def _fmt_int(v: Optional[int]) -> str:
    return "-" if v is None else str(v)


def _fmt_delta_int(v: Optional[int]) -> str:
    return "-" if v is None else f"{v:+d}"


def text_anchor_report(rows: list[AnchorRow]) -> str:
    lines = []
    lines.append("Three-way per-anchor timing (each source normalized to its earliest anchor; HW = ground truth)")
    lines.append("Cycles are relative to each source's first activity; iΔ/aΔ are vs HW.")
    lines.append("")

    # Group rows by (kernel, compiler) for readable sectioning.
    from itertools import groupby

    header = f"  {'tile':<7} {'kind':<20} {'HW':>9} {'interp':>9} {'aiesim':>9} {'iΔ':>8} {'aΔ':>8}"
    all_i_abs: list[int] = []
    all_a_abs: list[int] = []
    for (kernel, compiler), group in groupby(rows, key=lambda r: (r.kernel, r.compiler)):
        group = list(group)
        lines.append(f"{kernel}  [{compiler}]")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        i_abs: list[int] = []
        a_abs: list[int] = []
        for r in group:
            di = r.delta(r.interp)
            da = r.delta(r.aiesim)
            if di is not None:
                i_abs.append(abs(di))
            if da is not None:
                a_abs.append(abs(da))
            tile = f"({r.col},{r.row})"
            lines.append(
                f"  {tile:<7} {r.kind:<20} "
                f"{_fmt_int(r.hw):>9} {_fmt_int(r.interp):>9} {_fmt_int(r.aiesim):>9} "
                f"{_fmt_delta_int(di):>8} {_fmt_delta_int(da):>8}"
            )
        all_i_abs.extend(i_abs)
        all_a_abs.extend(a_abs)

        def mad(vals):
            return f"{sum(vals) / len(vals):.1f}" if vals else "-"

        lines.append(f"  mean |Δ| vs HW:  interp={mad(i_abs)}  aiesim={mad(a_abs)}  (anchors: i={len(i_abs)} a={len(a_abs)})")
        lines.append("")

    def mad_all(vals):
        return f"{sum(vals) / len(vals):.1f}" if vals else "-"

    lines.append(
        f"overall mean |Δ| vs HW:  interp={mad_all(all_i_abs)}  aiesim={mad_all(all_a_abs)}  "
        f"(anchors: i={len(all_i_abs)} a={len(all_a_abs)})"
    )
    return "\n".join(lines) + "\n"


def json_anchor_report(rows: list[AnchorRow]) -> str:
    out = []
    for r in rows:
        out.append(
            {
                "kernel": r.kernel,
                "compiler": r.compiler,
                "col": r.col,
                "row": r.row,
                "kind": r.kind,
                "hw": r.hw,
                "interp": r.interp,
                "aiesim": r.aiesim,
                "interp_delta": r.delta(r.interp),
                "aiesim_delta": r.delta(r.aiesim),
            }
        )
    return json.dumps(out, indent=2) + "\n"


def selftest() -> int:
    recs = [
        TimingRecord.from_dict({"kernel": "k1", "compiler": "chess", "source": "hw", "total_cycles": 1000}),
        TimingRecord.from_dict({"kernel": "k1", "compiler": "chess", "source": "interp", "total_cycles": 1050}),
        TimingRecord.from_dict({"kernel": "k1", "compiler": "chess", "source": "aiesim", "total_cycles": 900}),
        TimingRecord.from_dict({"kernel": "k2", "compiler": "chess", "source": "hw", "total_cycles": 2000}),
        TimingRecord.from_dict({"kernel": "k2", "compiler": "chess", "source": "aiesim", "total_cycles": 2000}),
    ]
    rows = build_drift_rows(recs)
    assert len(rows) == 2, rows
    k1 = rows[0]
    assert k1.kernel == "k1" and k1.hw == 1000 and k1.interp == 1050 and k1.aiesim == 900
    assert abs(k1.drift(k1.interp) - 5.0) < 1e-9, k1.drift(k1.interp)
    assert abs(k1.drift(k1.aiesim) - (-10.0)) < 1e-9, k1.drift(k1.aiesim)
    k2 = rows[1]
    assert k2.interp is None and k2.drift(k2.interp) is None
    assert k2.drift(k2.aiesim) == 0.0
    # Division-by-zero / missing-HW guard.
    z = DriftRow("z", "chess", hw=0, interp=5, aiesim=None)
    assert z.drift(z.interp) is None

    # -- Per-anchor (Option B) --
    # HW and aiesim share two anchors; each source has a different absolute
    # origin (HW base 1000, aiesim base 5000). After per-source normalization to
    # the earliest anchor, the shared spacing (0 and 200) must match -> delta 0.
    arecs = [
        TimingRecord.from_dict({
            "kernel": "ak", "compiler": "chess", "source": "hw", "total_cycles": 200,
            "anchors": [
                {"col": 1, "row": 2, "kind": "dma_s2mm0_start", "cycle": 1000},
                {"col": 1, "row": 2, "kind": "dma_s2mm0_done", "cycle": 1200},
            ],
        }),
        TimingRecord.from_dict({
            "kernel": "ak", "compiler": "chess", "source": "aiesim", "total_cycles": 210,
            "anchors": [
                {"col": 1, "row": 2, "kind": "dma_s2mm0_start", "cycle": 5000},
                {"col": 1, "row": 2, "kind": "dma_s2mm0_done", "cycle": 5210},  # +10 drift
            ],
        }),
    ]
    arows = build_anchor_rows(arecs)
    assert len(arows) == 2, arows
    start = next(r for r in arows if r.kind == "dma_s2mm0_start")
    done = next(r for r in arows if r.kind == "dma_s2mm0_done")
    # Normalized: HW start=0 done=200; aiesim start=0 done=210.
    assert start.hw == 0 and start.aiesim == 0 and start.delta(start.aiesim) == 0, start
    assert done.hw == 200 and done.aiesim == 210 and done.delta(done.aiesim) == 10, done
    assert done.interp is None and done.delta(done.interp) is None, done
    # Reports render without error.
    assert "per-anchor" in text_anchor_report(arows)
    assert json.loads(json_anchor_report(arows))[0]["kind"] in ("dma_s2mm0_done", "dma_s2mm0_start")
    # Empty anchors -> empty normalization.
    assert normalize_anchors(TimingRecord.from_dict(
        {"kernel": "x", "compiler": "c", "source": "hw", "total_cycles": 1})) == {}

    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Three-way timing calibration comparator")
    ap.add_argument("--records", help="directory of *.timing.json, or a glob")
    ap.add_argument("--per-anchor", action="store_true",
                    help="emit the Option-B per-anchor report instead of total-cycle drift")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("-o", "--output", help="write report to file instead of stdout")
    ap.add_argument("--selftest", action="store_true", help="run internal self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.records:
        ap.error("--records is required (or use --selftest)")

    paths = expand_record_paths(args.records)
    if not paths:
        print(f"No timing records matched: {args.records}", file=sys.stderr)
        return 1
    records = load_records(paths)
    if args.per_anchor:
        arows = build_anchor_rows(records)
        report = json_anchor_report(arows) if args.json else text_anchor_report(arows)
    else:
        rows = build_drift_rows(records)
        report = json_report(rows) if args.json else text_report(rows)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
