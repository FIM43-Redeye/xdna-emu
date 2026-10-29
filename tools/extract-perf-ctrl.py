#!/usr/bin/env python3
"""Extract per-tile PERF_CTRL configuration from post-injection MLIR.

Feeds item #9 (perf-counter-driven LOCK_STALL emission) analysis in the
HW measurement campaign.

For IRON-style tests, perf counters are NOT configured in the xclbin's
CDO -- they're written at runtime by the NPU instruction stream. The
authoritative source is the post-injection MLIR (e.g.,
`build/test/npu-xrt/<test>/traced/aie_traced.mlir`), which carries
`aie.trace.reg register = "Performance_..." value = N` ops inside
`aie.trace.config @perf_<scope>_<col>_<row>(%tile_<col>_<row>)` blocks.

This script scans those config blocks and emits per-tile JSON:

    {
        "mlir": "path/to/aie_traced.mlir",
        "tiles": [
            {"col": 1, "row": 2, "scope": "core", "registers": {
                "Performance_Control1": 28,
                "Performance_Control2": 458752,
                "Performance_Counter2_Event_Value": 1024
            }},
            ...
        ]
    }

The companion Rust binary `extract-perf-ctrl` reads CDO writes from
the xclbin directly; use that only when a test configures perf
counters at xclbin load.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Matches:
#   aie.trace.config @perf_<scope>_<col>_<row>(%tile_<col>_<row>) {
# Scope is something like "core" or "memtile" (used for grouping; the
# (col, row) tuple is the canonical identifier).
CONFIG_HEADER_RE = re.compile(
    r"aie\.trace\.config\s+@(?P<name>perf_[A-Za-z0-9_]+)"
    r"\(%tile_(?P<col>\d+)_(?P<row>\d+)\)\s*\{"
)

# Matches:
#   aie.trace.reg register = "<NAME>" value = <N> [mask = <M>] [comment = "..."]
REG_WRITE_RE = re.compile(
    r'aie\.trace\.reg\s+register\s*=\s*"(?P<name>[^"]+)"\s+value\s*=\s*(?P<value>\d+)'
)


def parse_mlir(path: Path) -> list[dict]:
    text = path.read_text()
    tiles: list[dict] = []
    pos = 0
    while True:
        m = CONFIG_HEADER_RE.search(text, pos)
        if not m:
            break
        # Find the matching closing brace.
        block_start = m.end()
        depth = 1
        i = block_start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        block = text[block_start : i - 1]
        pos = i

        # Filter Performance_* registers; trace setup also writes
        # Trace_Control / Trace_Event regs in the same block, but
        # those aren't perf-counter-related.
        regs: dict[str, int] = {}
        for rm in REG_WRITE_RE.finditer(block):
            name = rm.group("name")
            if name.startswith("Performance_"):
                regs[name] = int(rm.group("value"))

        if regs:
            tiles.append(
                {
                    "col": int(m.group("col")),
                    "row": int(m.group("row")),
                    "scope": m.group("name").removeprefix("perf_"),
                    "registers": regs,
                }
            )
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract PERF_CTRL config from post-injection MLIR.")
    ap.add_argument("mlir", help="Path to aie_traced.mlir")
    ap.add_argument("-o", "--output", help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    mlir_path = Path(args.mlir)
    if not mlir_path.exists():
        print(f"error: {mlir_path} does not exist", file=sys.stderr)
        sys.exit(1)

    tiles = parse_mlir(mlir_path)
    report = {"mlir": str(mlir_path), "tiles": tiles}
    out = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(out + "\n")
        print(
            f"Wrote {args.output} ({len(tiles)} tile(s) with Performance_* writes)",
            file=sys.stderr,
        )
    else:
        print(out)


if __name__ == "__main__":
    main()
