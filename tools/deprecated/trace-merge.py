#!/usr/bin/env python3
"""Merge multiple trace runs into a unified Perfetto JSON trace.

Uses TRUE events (slot 0 metronome) as alignment anchors to stitch
traces from different sweep batches onto a common timeline.

Each batch has TRUE in slot 0, firing every cycle.  The TRUE events
provide a continuous timestamp reference.  This tool:

1. Loads all batch traces
2. Extracts TRUE events from each to build a timestamp mapping
3. Finds the common time origin (first TRUE in each batch)
4. Remaps all non-TRUE events to the unified timeline
5. Deduplicates metadata events
6. Writes a single merged Perfetto JSON

Usage:
    trace-merge.py batch_00/trace.json batch_01/trace.json -o merged.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_trace(path: str) -> list[dict]:
    """Load a Perfetto JSON trace file."""
    with open(path) as f:
        return json.load(f)


def extract_true_events(events: list[dict]) -> list[dict]:
    """Extract TRUE events from a trace.

    TRUE events are identified by name -- mlir-aie's parse.py labels
    them based on the event code in the configured slot.
    """
    true_names = {"TRUE", "true", "Event TRUE"}
    return [e for e in events if e.get("name", "") in true_names]


def find_first_timestamp(events: list[dict]) -> int | None:
    """Find the earliest timestamp in a list of events."""
    timestamps = [e["ts"] for e in events if "ts" in e]
    return min(timestamps) if timestamps else None


def classify_events(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split events into (metadata, true_events, other_events).

    Metadata events have ph="M" (process/thread names).
    TRUE events are the metronome.
    Other events are the payload to merge.
    """
    metadata = []
    true_evts = []
    others = []
    true_names = {"TRUE", "true", "Event TRUE"}

    for e in events:
        ph = e.get("ph", "")
        name = e.get("name", "")
        if ph == "M":
            metadata.append(e)
        elif name in true_names:
            true_evts.append(e)
        else:
            others.append(e)

    return metadata, true_evts, others


def build_time_mapping(true_events: list[dict]) -> dict[int, int]:
    """Build a mapping from TRUE event timestamps to cycle indices.

    Since TRUE fires every cycle, the i-th TRUE event corresponds to
    cycle i relative to the first TRUE.  Returns {timestamp: cycle_index}.
    """
    if not true_events:
        return {}

    sorted_events = sorted(true_events, key=lambda e: e.get("ts", 0))
    base_ts = sorted_events[0].get("ts", 0)
    mapping = {}
    for i, e in enumerate(sorted_events):
        mapping[e.get("ts", 0)] = i
    return mapping


def remap_timestamp(ts: int, src_mapping: dict[int, int], dst_base: int) -> int:
    """Remap a timestamp from source run to unified timeline.

    Finds the nearest TRUE cycle in the source mapping and applies the
    same offset to the destination base timestamp.
    """
    if ts in src_mapping:
        cycle_idx = src_mapping[ts]
        return dst_base + cycle_idx

    # Find nearest TRUE timestamp (interpolate)
    if not src_mapping:
        return ts

    sorted_ts = sorted(src_mapping.keys())

    # Binary search for nearest
    lo, hi = 0, len(sorted_ts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] < ts:
            lo = mid + 1
        else:
            hi = mid

    # Interpolate: find nearest TRUE and compute fractional position
    nearest_ts = sorted_ts[lo]
    nearest_cycle = src_mapping[nearest_ts]
    offset = ts - nearest_ts
    return dst_base + nearest_cycle + offset


def merge_traces(trace_paths: list[str]) -> list[dict]:
    """Merge multiple batch traces into a unified trace."""
    if not trace_paths:
        return []

    all_metadata = {}  # Dedup by (pid, tid, name)
    all_events = []
    unified_base = 0  # Base timestamp for unified timeline

    # First pass: find the global time origin from the first batch
    first_trace = load_trace(trace_paths[0])
    first_meta, first_true, first_others = classify_events(first_trace)
    first_mapping = build_time_mapping(first_true)

    if first_mapping:
        unified_base = min(first_mapping.keys())

    # Collect metadata from all batches (dedup)
    for path in trace_paths:
        trace = load_trace(path)
        for e in trace:
            if e.get("ph") == "M":
                key = (e.get("pid", 0), e.get("tid", 0), e.get("name", ""))
                if key not in all_metadata:
                    all_metadata[key] = e

    # Add deduplicated metadata
    all_events.extend(all_metadata.values())

    # Process each batch
    for batch_idx, path in enumerate(trace_paths):
        trace = load_trace(path)
        metadata, true_evts, others = classify_events(trace)
        src_mapping = build_time_mapping(true_evts)

        if not src_mapping:
            # No TRUE events -- just add events with original timestamps
            print(f"  Warning: batch {batch_idx} ({path}) has no TRUE events, "
                  "adding with original timestamps", file=sys.stderr)
            all_events.extend(others)
            continue

        # Remap non-TRUE events to unified timeline
        src_base = min(src_mapping.keys())
        for e in others:
            if "ts" in e:
                new_e = dict(e)
                new_e["ts"] = remap_timestamp(e["ts"], src_mapping, unified_base)
                # Tag the source batch for debugging
                if "args" not in new_e:
                    new_e["args"] = {}
                new_e["args"]["sweep_batch"] = batch_idx
                all_events.append(new_e)
            else:
                all_events.append(e)

        print(f"  Batch {batch_idx}: {len(others)} events remapped "
              f"(TRUE anchor: {len(true_evts)} points)")

    # Sort by timestamp for clean output
    all_events.sort(key=lambda e: (e.get("ts", 0), e.get("pid", 0), e.get("tid", 0)))

    return all_events


def normalize_to_first_work_event(events: list[dict]) -> list[dict]:
    """Rebase all timestamps so the first work event is at t=0.

    "Work events" are non-metadata, non-TRUE payload events.  The boot
    phase (CDO application, ELF loading, timer init) produces the first
    TRUE and possibly some early events, but the first instruction-level
    or DMA event marks when real computation starts.

    This removes the platform-specific boot offset so HW and EMU traces
    can be compared on a common timeline.
    """
    # Find the first work event timestamp
    true_names = {"TRUE", "true", "Event TRUE"}
    first_work_ts = None
    for e in events:
        if e.get("ph") == "M":
            continue
        if e.get("name", "") in true_names:
            continue
        ts = e.get("ts")
        if ts is not None and ts > 0:
            if first_work_ts is None or ts < first_work_ts:
                first_work_ts = ts

    if first_work_ts is None or first_work_ts == 0:
        return events

    # Shift all timestamps
    result = []
    for e in events:
        if "ts" in e:
            new_e = dict(e)
            new_e["ts"] = max(0, e["ts"] - first_work_ts)
            result.append(new_e)
        else:
            result.append(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-batch trace sweep results using TRUE alignment",
    )
    parser.add_argument(
        "traces",
        nargs="+",
        help="Perfetto JSON trace files to merge",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for merged Perfetto JSON",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip boot-offset normalization (default: normalize to first work event)",
    )
    args = parser.parse_args()

    # Validate inputs
    for path in args.traces:
        if not Path(path).exists():
            print(f"Error: trace file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Merging {len(args.traces)} trace files...")
    merged = merge_traces(args.traces)

    if not args.no_normalize:
        merged = normalize_to_first_work_event(merged)
        print("Normalized: timestamps rebased to first work event (t=0)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    # Count event types in merged output
    event_counts = {}
    for e in merged:
        if e.get("ph") == "M":
            continue
        name = e.get("name", "unknown")
        event_counts[name] = event_counts.get(name, 0) + 1

    print(f"Merged trace: {len(merged)} events ({len(event_counts)} types)")
    print(f"Output: {args.output}")
    for name, count in sorted(event_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
