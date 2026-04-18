#!/usr/bin/env python3
"""Decode and analyze trace mode experiment results.

Reads raw trace captures from run_trace_experiment.sh output directory
and produces consistency analysis + execution mode decoding.
"""
import os
import struct
import sys
from collections import Counter


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


def decode_execution_core(core_data):
    """Decode execution mode core data into atom sequence + events."""
    if len(core_data) < 4:
        return {"sync_pc": 0, "atoms": "", "atom_packets": 0, "events": 0,
                "atom_detail": [], "event_detail": []}

    sync_pc = core_data[0] | (core_data[1] << 8)
    atoms = ""
    atom_packets = 0
    events = 0
    atom_detail = []
    event_detail = []

    i = 4  # skip SYNC block
    while i < len(core_data):
        b = core_data[i]

        if b == 0xFE:
            i += 1
            continue
        if b == 0xCC or (b & 0xFC) == 0xDC:
            i += 1
            continue

        # 2-byte atom packet: next byte is 0x80-0x87
        if i + 1 < len(core_data) and (core_data[i + 1] & 0xF8) == 0x80:
            high_nib = (b >> 4) & 0xF
            type_low = core_data[i + 1] & 0x7
            packet_atoms = ""
            for bit in range(3, -1, -1):
                packet_atoms += "E" if high_nib & (1 << bit) else "N"
            atoms += packet_atoms
            atom_packets += 1
            atom_detail.append((b, core_data[i + 1], packet_atoms, type_low))
            i += 2
            continue

        # 1-byte event (Single0): MSB=0
        if (b & 0x80) == 0x00:
            slot = (b >> 4) & 0x7
            delta = b & 0xF
            events += 1
            event_detail.append((slot, delta))
            i += 1
            continue

        i += 1

    return {
        "sync_pc": sync_pc,
        "atoms": atoms,
        "atom_packets": atom_packets,
        "events": events,
        "atom_detail": atom_detail,
        "event_detail": event_detail,
    }


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

    # Execution mode decode
    print()
    print("=== Execution mode decode ===")
    path = os.path.join(results_dir, "mode2_execution", "run1", "trace_raw.bin")
    if not os.path.exists(path):
        print("  No execution mode data found")
        return

    data = load_trace(path)
    sources = extract_packets_by_source(data)
    core_data = bytes(sources.get(0x1E, b""))

    result = decode_execution_core(core_data)
    print(f"  Sync PC: 0x{result['sync_pc']:04x}")
    print(f"  Atom packets: {result['atom_packets']} ({len(result['atoms'])} atom bits)")
    print(f"  Events: {result['events']}")
    print(f"  E count: {result['atoms'].count('E')}, N count: {result['atoms'].count('N')}")
    print()

    # Show atom detail
    print("  Atom packets:")
    for i, (data_byte, type_byte, packet_atoms, tlow) in enumerate(result["atom_detail"]):
        print(f"    [{i:2d}] 0x{data_byte:02x} 0x{type_byte:02x}  {packet_atoms}  tlow={tlow}")

    print()
    print(f"  Full atom sequence:")
    # Print in groups of 4 for readability
    seq = result["atoms"]
    for i in range(0, len(seq), 40):
        chunk = seq[i : i + 40]
        print(f"    {chunk}")

    # Show event summary
    print()
    slot_names = {
        0: "INSTR_EVENT_0", 1: "INSTR_EVENT_1", 2: "INSTR_VECTOR",
        3: "LOCK_ACQ_INSTR", 5: "LOCK_ACQ_REQ", 7: "LOCK_STALL",
    }
    event_counts = Counter(slot for slot, _ in result["event_detail"])
    print(f"  Events by slot:")
    for slot, count in sorted(event_counts.items()):
        sname = slot_names.get(slot, f"slot{slot}")
        print(f"    slot {slot} ({sname}): {count}")

    # type_low distribution
    tlow_dist = Counter(tlow for _, _, _, tlow in result["atom_detail"])
    print(f"\n  type_low distribution: {dict(sorted(tlow_dist.items()))}")


if __name__ == "__main__":
    main()
