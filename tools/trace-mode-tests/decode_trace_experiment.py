#!/usr/bin/env python3
"""Decode and analyze trace mode experiment results.

Reads raw trace captures from run_trace_experiment.sh output directory
and produces a run-to-run consistency and data-volume analysis.
Execution mode (2) frames are not decoded: mode 2 is not modeled.
"""
import os
import struct
import sys


def load_trace(path):
    with open(path, "rb") as f:
        return f.read()


def find_data_end(data):
    for i in range(len(data) - 1, -1, -1):
        if data[i] != 0:
            return ((i + 32) // 32) * 32
    return 0


def extract_packets_by_source(data):
    """Extract packet data bytes grouped by source (packet ID)."""
    end = find_data_end(data)
    sources = {}
    for off in range(0, end, 32):
        pkt_id = data[off] & 0x1F
        if pkt_id not in sources:
            sources[pkt_id] = bytearray()
        sources[pkt_id].extend(data[off + 4 : off + 32])
    return sources


def main():
    results_dir = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    modes = [
        ("mode0_event_time", "Event-Time"),
        ("mode1_event_pc", "Event-PC"),
        ("mode2_execution", "Execution"),
        ("mode3_reserved_11", "Reserved(11)"),
    ]

    # Consistency check
    print("=== Run-to-run consistency (core data) ===")
    for dirname, label in modes:
        runs_core = []
        for r in range(1, num_runs + 1):
            path = os.path.join(results_dir, dirname, f"run{r}", "trace_raw.bin")
            if not os.path.exists(path):
                continue
            data = load_trace(path)
            sources = extract_packets_by_source(data)
            runs_core.append(bytes(sources.get(0x1E, b"")))

        if len(runs_core) < 2:
            print(f"  {label:12s}: insufficient runs")
            continue

        if all(r == runs_core[0] for r in runs_core[1:]):
            print(f"  {label:12s}: ALL {num_runs} RUNS IDENTICAL ({len(runs_core[0])} core bytes)")
        else:
            diffs = sum(1 for a, b in zip(runs_core[0], runs_core[1]) if a != b)
            print(f"  {label:12s}: VARIES ({diffs} differing bytes in core data)")

    # Cross-mode data volumes
    print()
    print("=== Data volumes (run1) ===")
    for dirname, label in modes:
        path = os.path.join(results_dir, dirname, "run1", "trace_raw.bin")
        if not os.path.exists(path):
            print(f"  {label:12s}: NOT FOUND")
            continue
        data = load_trace(path)
        sources = extract_packets_by_source(data)
        parts = []
        for pkt_id in sorted(sources):
            name = {0x1D: "memtile", 0x1E: "core", 0x1F: "mem"}.get(pkt_id, f"0x{pkt_id:02x}")
            parts.append(f"{name}={len(sources[pkt_id])}")
        total = find_data_end(data)
        print(f"  {label:12s}: {total:4d} total  [{', '.join(parts)}]")


if __name__ == "__main__":
    main()
