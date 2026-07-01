#!/usr/bin/env python3
"""Normalize decoded trace events.json back to the MLIR-declared placement
(#140 SP-5b Task 6 glue).

tools/parse-trace.py's ``--out-events`` output leaves each event's raw,
physical ``(col, row)`` untouched and separately reports the observed
runtime placement in a top-level ``"placement": {"origin_col", "origin_row"}``
field (the smallest observed col/row corner -- see parse-trace.py's own
comment: "trace-compare uses this to normalize HW vs EMU when one side runs
at start_col != 0"). bridge-trace-runner places sp5_skew_r1 at its real
allocation-scheme column rather than the MLIR-declared col=0 (observed
origin_col=1 for the sibling SP-3 kernel on the same 2-column-device build,
confirmed on real NPU1 by
build/experiments/sp3-spike-trace/task3_tally.py's EXPECTED_TILES, which
hardcodes the shifted col=1 coordinates it actually saw). geometry.json (this
kernel's pair definitions for tools/calibration/skew/r1_observe.py) declares
every anchor at the MLIR-authored col=0, and observe_r1 does exact
(col,row,pkt_type,name) dict lookups -- so without this normalization step,
every anchor lookup against a real decoded trace raises KeyError.

This fix intentionally lives here, NOT in r1_tally.py: the task brief
requires r1_tally.py's body be used byte-for-byte verbatim. The same
origin_col/origin_row subtraction is already an established contract
elsewhere in this repo -- see tools/shim-chain-fit.py,tools/shim-
throughput-fit.py, and src/trace/compare.rs's placement-normalization path.

Idempotent: a no-op when origin_col == origin_row == 0 (already in the
declared frame -- e.g. a future build that isn't column-shifted), and safe
to call twice (normalizes once, then reports {0,0} and no-ops thereafter).
"""
import json
import sys


def normalize(path):
    with open(path) as f:
        doc = json.load(f)
    placement = doc.get("placement", {})
    origin_col = placement.get("origin_col", 0)
    origin_row = placement.get("origin_row", 0)
    if origin_col == 0 and origin_row == 0:
        return doc  # already in the declared frame -- nothing to do
    for e in doc["events"]:
        e["col"] -= origin_col
        e["row"] -= origin_row
    doc["placement"] = {"origin_col": 0, "origin_row": 0}
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: normalize_placement.py <events.json> [<events.json> ...]")
    for p in sys.argv[1:]:
        normalize(p)
