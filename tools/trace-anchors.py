#!/usr/bin/env python3
"""Reduce a parse-trace events.json to canonical DMA timing anchors (task #84).

This is the HW / interpreter side of Option B (per-anchor) three-way timing
calibration (docs/coverage/three-way-timing-calibration.md). It turns the flat
trace-BO event list emitted by `parse-trace.py --out-events` into the same
canonical anchor vocabulary the aiesim side produces from the in-process NPU1
VCD (`vcd-compare --anchors`), so all three sources align on (col, row, kind).

Anchor vocabulary (v1), grounded on the mlir-aie event enum
(python/utils/trace/setup.py, the authoritative event-name source):

    DMA_S2MM_{ch}_START_TASK     -> kind dma_s2mm{ch}_start
    DMA_S2MM_{ch}_FINISHED_TASK  -> kind dma_s2mm{ch}_done
    DMA_MM2S_{ch}_START_TASK     -> kind dma_mm2s{ch}_start
    DMA_MM2S_{ch}_FINISHED_TASK  -> kind dma_mm2s{ch}_done

A channel may fire its BD multiple times, producing several START/FINISHED
events. To match the VCD side (first leave-idle / last change), per
(col, row, kind) we keep the EARLIEST `soc` for *_start anchors and the LATEST
`soc` for *_done anchors. `soc` (start-of-cycle, cycle-corrected) is the precise
cycle field -- preferred over `ts` per the calibration doc.

Coordinates are NPU1 geometry on both the trace BO and the in-process VCD, so no
geometry normalization is applied here.

Usage:
    trace-anchors.py <events.json> [--source hw|interp] [--kernel K] \
        [--compiler C] [--total-cycles N] [--json] [-o out.json]
    trace-anchors.py --selftest

With --kernel/--compiler/--source a full timing record is emitted (matching the
data contract in timing-three-way.py); otherwise a bare {"anchors":[...]}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# DMA task events -> (direction, channel, phase). Names per mlir-aie event enum.
_DMA_EVENT_RE = re.compile(r"^DMA_(S2MM|MM2S)_(\d+)_(START|FINISHED)_TASK$")


def load_events(fh) -> list[dict]:
    """Load a parse-trace events file, accepting both schemas it emits:
    a bare flat list, or the wrapped `{"schema_version":N, "events":[...]}`.
    Returns the flat event list (empty if neither shape is present)."""
    data = json.load(fh)
    if isinstance(data, dict):
        return list(data.get("events", []))
    if isinstance(data, list):
        return data
    return []


def canonical_kind(event_name: str) -> str | None:
    """Map a trace-BO event name to a canonical anchor kind, or None."""
    m = _DMA_EVENT_RE.match(event_name or "")
    if not m:
        return None
    direction = m.group(1).lower()
    ch = int(m.group(2))
    phase = "start" if m.group(3) == "START" else "done"
    return f"dma_{direction}{ch}_{phase}"


def reduce_anchors(events: list[dict]) -> list[dict]:
    """Reduce a flat event list to canonical anchors.

    For each (col, row, kind): *_start keeps the minimum `soc`, *_done keeps the
    maximum `soc`. Returns anchors sorted by (col, row, kind, cycle).
    """
    # key -> chosen cycle
    best: dict[tuple[int, int, str], int] = {}
    for e in events:
        kind = canonical_kind(e.get("name", ""))
        if kind is None:
            continue
        # `soc` is the precise start-of-cycle; fall back to `ts` if absent.
        cycle = e.get("soc")
        if cycle is None:
            cycle = e.get("ts")
        if cycle is None:
            continue
        cycle = int(cycle)
        col = int(e["col"])
        row = int(e["row"])
        key = (col, row, kind)
        if key not in best:
            best[key] = cycle
        elif kind.endswith("_start"):
            best[key] = min(best[key], cycle)
        else:  # _done
            best[key] = max(best[key], cycle)

    anchors = [
        {"col": col, "row": row, "kind": kind, "cycle": cycle}
        for (col, row, kind), cycle in best.items()
    ]
    anchors.sort(key=lambda a: (a["col"], a["row"], a["kind"], a["cycle"]))
    return anchors


def build_output(
    anchors: list[dict],
    kernel: str | None,
    compiler: str | None,
    source: str | None,
    total_cycles: int | None,
) -> dict:
    """Bare {anchors} measurement, or a full timing record if identity given."""
    if kernel or compiler or source:
        return {
            "kernel": kernel or "",
            "compiler": compiler or "",
            "source": source or "",
            "total_cycles": total_cycles,
            "anchors": anchors,
        }
    out: dict = {"anchors": anchors}
    if total_cycles is not None:
        out["total_cycles"] = total_cycles
    return out


def selftest() -> int:
    events = [
        # Two BD iterations on compute s2mm0: start keeps min soc, done keeps max.
        {"col": 1, "row": 2, "name": "DMA_S2MM_0_START_TASK", "soc": 395, "ts": 9},
        {"col": 1, "row": 2, "name": "DMA_S2MM_0_START_TASK", "soc": 800, "ts": 9},
        {"col": 1, "row": 2, "name": "DMA_S2MM_0_FINISHED_TASK", "soc": 1200, "ts": 9},
        {"col": 1, "row": 2, "name": "DMA_S2MM_0_FINISHED_TASK", "soc": 2577, "ts": 9},
        # Different channel / direction.
        {"col": 1, "row": 2, "name": "DMA_MM2S_0_START_TASK", "soc": 398, "ts": 9},
        # Shim channel.
        {"col": 1, "row": 0, "name": "DMA_S2MM_0_START_TASK", "soc": 2093, "ts": 9},
        # Non-DMA event ignored.
        {"col": 1, "row": 2, "name": "LOCK_STALL", "soc": 10, "ts": 9},
        # Empty name ignored.
        {"col": 1, "row": 2, "name": "", "soc": 11, "ts": 9},
        # soc absent -> falls back to ts.
        {"col": 1, "row": 1, "name": "DMA_MM2S_1_START_TASK", "ts": 470},
    ]
    anchors = reduce_anchors(events)
    by = {(a["col"], a["row"], a["kind"]): a["cycle"] for a in anchors}
    assert by[(1, 2, "dma_s2mm0_start")] == 395, by  # min of 395, 800
    assert by[(1, 2, "dma_s2mm0_done")] == 2577, by  # max of 1200, 2577
    assert by[(1, 2, "dma_mm2s0_start")] == 398, by
    assert by[(1, 0, "dma_s2mm0_start")] == 2093, by
    assert by[(1, 1, "dma_mm2s1_start")] == 470, by  # ts fallback
    # No spurious anchors from LOCK_STALL / empty name.
    assert len(anchors) == 5, anchors
    # canonical_kind edge cases.
    assert canonical_kind("DMA_MM2S_3_FINISHED_TASK") == "dma_mm2s3_done"
    assert canonical_kind("PORT_RUNNING_0") is None
    assert canonical_kind("") is None
    # Record wrapping.
    rec = build_output(anchors, "k", "chess", "hw", 2182)
    assert rec["source"] == "hw" and rec["total_cycles"] == 2182
    assert rec["anchors"] == anchors
    bare = build_output(anchors, None, None, None, None)
    assert bare == {"anchors": anchors}
    # load_events accepts both the flat-list and {events:[...]} schemas.
    import io
    flat = [{"col": 0, "row": 0, "name": "X", "soc": 1}]
    assert load_events(io.StringIO(json.dumps(flat))) == flat
    assert load_events(io.StringIO(json.dumps({"schema_version": 1, "events": flat}))) == flat
    assert load_events(io.StringIO(json.dumps({"schema_version": 1}))) == []
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Reduce parse-trace events.json to DMA timing anchors")
    ap.add_argument("events", nargs="?", help="path to parse-trace --out-events JSON")
    ap.add_argument("--source", help="timing-record source tag (hw|interp); enables record output")
    ap.add_argument("--kernel", help="kernel name for the timing record")
    ap.add_argument("--compiler", help="compiler tag for the timing record")
    ap.add_argument("--total-cycles", type=int, help="total-cycle scalar to embed (from --out-cycles)")
    ap.add_argument("-o", "--output", help="write to file instead of stdout")
    ap.add_argument("--selftest", action="store_true", help="run internal self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.events:
        ap.error("events.json path is required (or use --selftest)")

    with open(args.events) as fh:
        events = load_events(fh)

    anchors = reduce_anchors(events)
    out = build_output(anchors, args.kernel, args.compiler, args.source, args.total_cycles)
    report = json.dumps(out, indent=2) + "\n"

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        print(f"Anchors written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
