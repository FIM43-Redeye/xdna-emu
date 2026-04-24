#!/usr/bin/env python3
"""Merge an HW-only sweep directory with an EMU-only sweep directory.

Rationale: full-sweep validation runs HW and EMU in two passes --
HW serially (NPU contention) and EMU at N-way parallelism. Each pass
writes its own results directory. This script stitches them into a
single matrix directory that scripts/show-sweep-matrix.py can diff.

Matching: for every JSON in the HW dir, we look for a same-named JSON in
the EMU dir. If both are present, we emit a merged JSON to the output
dir where each event row takes:
  - hw.*  from the HW JSON  (emu.* fields discarded)
  - emu.* from the EMU JSON (hw.*  fields discarded)

If one side is missing, we emit the present side's JSON as-is (still
useful -- show-sweep-matrix.py will classify missing side as *-ERR or
skipped).

The merged JSON's top-level metadata (test, compiler, tile) is taken
from the HW side (EMU should match; if it doesn't, that's surfaced as a
warning).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def _index_events(doc: Dict) -> Dict[Tuple[str, int, int], Dict]:
    """Key events by (name, slot, batch) so cross-file lookup doesn't
    depend on list ordering."""
    idx = {}
    for row in doc.get("events", []):
        key = (row.get("name"), row.get("slot", -1), row.get("batch", -1))
        idx[key] = row
    return idx


def merge_docs(hw_doc: Dict, emu_doc: Dict) -> Dict:
    """Row-by-row merge. HW doc provides the row set and metadata; the
    EMU doc contributes its emu.* block per matched row. Rows only in
    one side keep whatever they had."""
    out = {
        "test": hw_doc.get("test") or emu_doc.get("test"),
        "compiler": hw_doc.get("compiler") or emu_doc.get("compiler"),
        "tile": hw_doc.get("tile") or emu_doc.get("tile"),
        "grounding_event": hw_doc.get("grounding_event")
                         or emu_doc.get("grounding_event"),
        "events": [],
        "elapsed_sec": round(
            (hw_doc.get("elapsed_sec") or 0.0)
            + (emu_doc.get("elapsed_sec") or 0.0), 2
        ),
    }
    emu_idx = _index_events(emu_doc)
    hw_idx = _index_events(hw_doc)
    all_keys = sorted(set(hw_idx) | set(emu_idx),
                      key=lambda k: (k[2], k[1], k[0]))
    for key in all_keys:
        name, slot, batch = key
        hw_row = hw_idx.get(key, {})
        emu_row = emu_idx.get(key, {})
        template = hw_row or emu_row
        merged = {
            "id": template.get("id"),
            "name": name,
            "slot": slot,
            "batch": batch,
            "hw":  hw_row.get("hw")  if hw_row  else {"fired": None, "error": "missing from HW pass"},
            "emu": emu_row.get("emu") if emu_row else {"fired": None, "error": "missing from EMU pass"},
        }
        out["events"].append(merged)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("hw_dir", type=Path, help="HW-only sweep directory")
    ap.add_argument("emu_dir", type=Path, help="EMU-only sweep directory")
    ap.add_argument("out_dir", type=Path, help="merged output directory (created)")
    args = ap.parse_args()

    for d in (args.hw_dir, args.emu_dir):
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hw_jsons = {p.name: p for p in args.hw_dir.glob("*.json")
                if not p.name.endswith(".merged.json") and p.name != "summary.json"}
    emu_jsons = {p.name: p for p in args.emu_dir.glob("*.json")
                 if not p.name.endswith(".merged.json") and p.name != "summary.json"}
    all_names = sorted(set(hw_jsons) | set(emu_jsons))

    merged_count = 0
    for name in all_names:
        hw_doc = json.loads(hw_jsons[name].read_text()) if name in hw_jsons else {}
        emu_doc = json.loads(emu_jsons[name].read_text()) if name in emu_jsons else {}
        merged = merge_docs(hw_doc, emu_doc)
        # Sanity: metadata should match. Surface divergence but keep going.
        for key in ("test", "compiler", "tile"):
            hv = hw_doc.get(key)
            ev = emu_doc.get(key)
            if hv and ev and hv != ev:
                print(f"warn: {name} metadata mismatch on {key!r}: "
                      f"HW={hv} EMU={ev}", file=sys.stderr)
        (args.out_dir / name).write_text(json.dumps(merged, indent=2) + "\n")
        merged_count += 1

    # Copy summary.json from each side if present so post-hoc log
    # inspection doesn't need to cross-reference the source dirs.
    for side_name, side_dir in (("hw", args.hw_dir), ("emu", args.emu_dir)):
        sj = side_dir / "summary.json"
        if sj.is_file():
            (args.out_dir / f"summary.{side_name}.json").write_text(sj.read_text())
    print(f"[merge] wrote {merged_count} merged matrices to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
