#!/usr/bin/env python3
"""Render and diff trace-sweep result matrices.

Three modes:

1. **Single file**: `show-sweep-matrix.py path/to/test.chess.core_c0r2.json`
   Prints a per-event table for that one (test, compiler, tile).

2. **Directory**: `show-sweep-matrix.py build/trace-sweep-results/latest`
   Prints every combo in the directory.

3. **Diff**: `show-sweep-matrix.py --diff BASELINE_DIR NEW_DIR`
   Prints only the (combo, event) cells whose classification changed.
   This is the regression-verification view: cells flipping MATCH ->
   DRIFT/MISS are events we used to emit correctly and now don't.

Classification per row (the `status` column):
  MATCH(N)     -- HW and EMU fired N times each
  DRIFT(H/E)   -- both fired but counts differ
  HW-ONLY(N)   -- HW fired N times, EMU 0 (real EMU gap)
  EMU-ONLY(N)  -- EMU fired N times, HW 0 (spurious emulator emission)
  ZERO         -- neither fired (filtered out of single-combo view)
  HW-ERR/EMU-ERR -- the corresponding side errored; data untrustworthy
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class ComboMatrix:
    """A single (test, compiler, tile) result loaded from one JSON file."""
    path: Path
    test: str
    compiler: str
    tile_label: str  # "core_c0r2"
    rows: List[Dict]  # from the "events" array in the JSON


def _tile_label(tile: Dict) -> str:
    return f"{tile['type']}_c{tile['col']}r{tile['row']}"


def load_combo(path: Path) -> ComboMatrix:
    doc = json.loads(path.read_text())
    return ComboMatrix(
        path=path,
        test=doc["test"],
        compiler=doc["compiler"],
        tile_label=_tile_label(doc["tile"]),
        rows=doc.get("events", []),
    )


def load_dir(d: Path) -> List[ComboMatrix]:
    """Load every sweep JSON in a directory (skipping .merged.json).

    Sorted by (test, compiler, tile_label) so output order is
    deterministic across runs.
    """
    combos: List[ComboMatrix] = []
    for p in sorted(d.glob("*.json")):
        # Skip our own non-combo outputs: merged timelines from
        # trace-sweep.py and any summary.*.json (harness writes
        # summary.json, merge-sweep-results.py writes summary.hw.json
        # and summary.emu.json).
        if p.name.endswith(".merged.json") or p.name.startswith("summary"):
            continue
        try:
            combos.append(load_combo(p))
        except Exception as e:
            print(f"warn: could not load {p}: {e}", file=sys.stderr)
    return combos


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_row(row: Dict) -> Tuple[str, Optional[int], Optional[int]]:
    """Return (status, hw_fires, emu_fires) for one event row.

    A side is 'error' when trace-sweep recorded a non-None `error` string
    on it -- meaning the runner or parser failed. That distinguishes
    "ran fine, event never fired" (fired=0, error=None) from "couldn't
    tell because the run itself blew up" (fired=None, error=...).
    """
    hw = row.get("hw", {})
    emu = row.get("emu", {})
    hf = hw.get("fired")
    ef = emu.get("fired")
    hw_err = hw.get("error") is not None and hf is None
    emu_err = emu.get("error") is not None and ef is None
    if hw_err and emu_err:
        return "BOTH-ERR", None, None
    if hw_err:
        return "HW-ERR", None, ef
    if emu_err:
        return "EMU-ERR", hf, None
    if (hf or 0) == 0 and (ef or 0) == 0:
        return "ZERO", 0, 0
    if (hf or 0) > 0 and (ef or 0) == 0:
        return f"HW-ONLY({hf})", hf, 0
    if (hf or 0) == 0 and (ef or 0) > 0:
        return f"EMU-ONLY({ef})", 0, ef
    if hf == ef:
        return f"MATCH({hf})", hf, ef
    return f"DRIFT(H{hf}/E{ef})", hf, ef


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _print_combo(combo: ComboMatrix, hide_zero: bool) -> None:
    print(f"\n== {combo.test} / {combo.compiler} / {combo.tile_label} "
          f"({combo.path.name}) ==")
    print(f"{'EVENT':32s} {'STATUS':20s}  {'HW':>6s}  {'EMU':>6s}")
    nonzero = 0
    for row in combo.rows:
        status, hf, ef = classify_row(row)
        if hide_zero and status == "ZERO":
            continue
        nonzero += 1
        hf_s = "-" if hf is None else str(hf)
        ef_s = "-" if ef is None else str(ef)
        print(f"{row['name']:32s} {status:20s}  {hf_s:>6s}  {ef_s:>6s}")
    if hide_zero:
        total = len(combo.rows)
        print(f"  [{nonzero}/{total} events had nonzero or errored rows]")


def show_single(combo: ComboMatrix, show_zero: bool) -> None:
    _print_combo(combo, hide_zero=not show_zero)


def show_directory(combos: List[ComboMatrix], show_zero: bool) -> None:
    if not combos:
        print("no sweep matrices found")
        return
    for c in combos:
        _print_combo(c, hide_zero=not show_zero)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _combo_key(c: ComboMatrix) -> str:
    return f"{c.test}.{c.compiler}.{c.tile_label}"


def _row_key(row: Dict) -> Tuple[str, int, int]:
    return row["name"], row.get("slot", -1), row.get("batch", -1)


def show_diff(baseline: List[ComboMatrix], new: List[ComboMatrix]) -> int:
    """Print cells whose classification changed. Returns number of diffs."""
    b_map = {_combo_key(c): c for c in baseline}
    n_map = {_combo_key(c): c for c in new}

    only_b = set(b_map) - set(n_map)
    only_n = set(n_map) - set(b_map)
    if only_b:
        print("[missing in NEW]: " + ", ".join(sorted(only_b)))
    if only_n:
        print("[missing in BASELINE]: " + ", ".join(sorted(only_n)))

    diffs = 0
    for key in sorted(set(b_map) & set(n_map)):
        b = b_map[key]
        n = n_map[key]
        # Index rows by (name, slot, batch) so sweeps with different
        # batch grouping still line up event-by-event.
        b_rows = {_row_key(r): r for r in b.rows}
        n_rows = {_row_key(r): r for r in n.rows}
        changed_rows = []
        for rk in sorted(set(b_rows) | set(n_rows)):
            b_row = b_rows.get(rk)
            n_row = n_rows.get(rk)
            b_cls = classify_row(b_row)[0] if b_row else "MISSING"
            n_cls = classify_row(n_row)[0] if n_row else "MISSING"
            if b_cls == n_cls:
                continue
            # Suppress noise: both sides zero/zero isn't a meaningful diff
            # even if wording differs. But ZERO -> anything-nonzero or
            # anything -> ZERO *is* a diff (that's a real event count
            # change).
            if b_cls == "ZERO" and n_cls == "ZERO":
                continue
            changed_rows.append((rk[0], b_cls, n_cls))
        if not changed_rows:
            continue
        print(f"\n== {key} ==")
        print(f"{'EVENT':32s} {'BASELINE':20s} -> {'NEW':20s}")
        for name, b_cls, n_cls in changed_rows:
            print(f"{name:32s} {b_cls:20s} -> {n_cls:20s}")
            diffs += 1
    shared = len(set(b_map) & set(n_map))
    print(f"\n[{diffs} changed cell(s) across {shared} shared combo(s)]")
    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_input(p: Path) -> Path:
    """Resolve `latest` symlink under build/trace-sweep-results if present."""
    if not p.exists():
        latest = Path(__file__).resolve().parent.parent / "build" / "trace-sweep-results" / p.name
        if latest.exists():
            return latest
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", nargs="?",
                    help="sweep JSON file or results directory "
                         "(default: build/trace-sweep-results/latest)")
    ap.add_argument("--diff", nargs=2, metavar=("BASELINE", "NEW"),
                    help="compare two results directories, show only changed cells")
    ap.add_argument("--show-zero", action="store_true",
                    help="include events that fired 0 times on both sides")
    args = ap.parse_args()

    if args.diff:
        b_dir = _resolve_input(Path(args.diff[0]))
        n_dir = _resolve_input(Path(args.diff[1]))
        if not b_dir.is_dir() or not n_dir.is_dir():
            print("--diff requires two directories", file=sys.stderr)
            return 2
        diffs = show_diff(load_dir(b_dir), load_dir(n_dir))
        return 0 if diffs == 0 else 1

    default = Path(__file__).resolve().parent.parent / "build" / "trace-sweep-results" / "latest"
    p = _resolve_input(Path(args.path)) if args.path else default
    if p.is_file():
        show_single(load_combo(p), show_zero=args.show_zero)
    elif p.is_dir():
        show_directory(load_dir(p), show_zero=args.show_zero)
    else:
        print(f"not a file or directory: {p}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
