#!/usr/bin/env python3
"""Convert a recorded single-core probe directory into a suite (harness/suite.py).

For evidence directories written before the suite format, whose generator no longer
reproduces them (the collision catalogue has grown since `collisions/probe`). Reads the
flat layout `manifest.json` + `kNNN.s` + `input.bin` (gen_acc_timing_probe /
gen_collision_probe) and writes cases whose kernels are byte-copies of the recorded `.s`,
so a suite run measures exactly the recorded kernels.

  python3 tools/probe_to_suite.py <probe-dir> <suite-dir> [--name NAME]

Recorded results (`<probe-dir>/<tag>/kNNN.bin`) are not imported; compare them with
`legacy_bin_matches` / by hand against `<suite-dir>/runs/<tag>/kNNN/core.out.bin`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness import suite   # noqa: E402


def convert(probe_dir: str | Path, suite_dir: str | Path, name: str | None = None) -> int:
    src, dst = Path(probe_dir), Path(suite_dir)
    manifest = json.loads((src / "manifest.json").read_text())
    table = (src / "input.bin").read_bytes()
    meta = {k: v for k, v in manifest.items() if k != "kernels"}
    for entry in manifest["kernels"]:
        if len(table) != entry["in_size"]:
            raise SystemExit(f"{entry['name']}: input.bin is {len(table)} bytes, in_size {entry['in_size']}")
        suite.write_single_core_case(dst, entry["name"], (src / f"{entry['name']}.s").read_text(),
                                     input_bytes=table, out_bytes=entry["out_size"], meta=entry)
    suite.write_suite_json(dst, name or src.name, f"probe_to_suite.py (from {src.name})",
                           [e["name"] for e in manifest["kernels"]], meta=meta)
    return len(manifest["kernels"])


def legacy_bin_matches(probe_dir: str | Path, legacy_tag: str, suite_dir: str | Path,
                       tag: str) -> tuple[int, list[str]]:
    """(cases compared, names that differ) between a legacy `<tag>/kNNN.bin` set and a suite run."""
    s = suite.load_suite(suite_dir)
    bad = [c for c in s.cases
           if (Path(probe_dir) / legacy_tag / f"{c}.bin").read_bytes() != s.output(tag, c).read_bytes()]
    return len(s.cases), bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe_dir")
    ap.add_argument("suite_dir")
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    print(f"{convert(a.probe_dir, a.suite_dir, a.name)} case(s) written to {a.suite_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
