#!/usr/bin/env python3
"""Compare hardware vs emulator trace binaries at the raw event level.

Works directly on the binary trace buffers (trace_raw.bin) using the
mlir-aie low-level decoder.  No Perfetto conversion -- zero information
loss.

Supports two modes:

1. **Single-batch**: Compare one HW trace_raw.bin against one EMU trace_raw.bin
   with a shared MLIR source for event name resolution.

2. **Sweep directory**: Compare all batches from a trace-sweep.py run,
   normalizing each batch pair independently to remove boot offset.

Usage:
    # Single batch (from trace-bridge.sh output)
    trace-compare.py --hw results/hw/trace_raw.bin \\
                     --emu results/emu/trace_raw.bin \\
                     --mlir results/traced/aie_traced.mlir

    # Sweep directory (from trace-sweep.py output)
    trace-compare.py --sweep /tmp/trace-sweep-add-one
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Raw binary trace decoder (derived from mlir-aie utils.py logic)
# ---------------------------------------------------------------------------

def decode_packets(raw: np.ndarray) -> list[dict]:
    """Split raw u32 trace buffer into packets with decoded headers.

    Each packet is 8 words.  Word 0 is the header:
      col[27:21], row[20:16], pkt_type[12:11], pkt_id[4:0]
    Words 1-7 are 28 bytes of compressed event data.
    """
    packets = []
    for i in range(0, len(raw), 8):
        pkt = raw[i:i+8]
        if not np.any(pkt):
            continue
        hdr = int(pkt[0])
        col = (hdr >> 21) & 0x7F
        row = (hdr >> 16) & 0x1F
        pkt_type = (hdr >> 11) & 0x3
        pkt_id = hdr & 0x1F

        # Extract 28 payload bytes (big-endian from words 1-7)
        payload = []
        for w in range(1, 8):
            word = int(pkt[w])
            payload.extend([
                (word >> 24) & 0xFF,
                (word >> 16) & 0xFF,
                (word >> 8) & 0xFF,
                word & 0xFF,
            ])

        packets.append({
            "col": col, "row": row,
            "pkt_type": pkt_type, "pkt_id": pkt_id,
            "payload": payload,
        })
    return packets


def decode_byte_stream(payload: list[int]) -> list[dict]:
    """Decode a 28-byte payload into a list of trace commands.

    Returns dicts with keys: type, event (slot index), cycles (delta),
    and for Multiple types: event0..event7 (bitmask of active slots).
    """
    commands = []
    cursor = 0

    try:
        while cursor < len(payload):
            b = payload[cursor]

            # Pad (0xFE) or end (0xFF sync)
            if b == 0xFE:
                cursor += 1
                continue
            if b == 0xFF:
                commands.append({"type": "Event_Sync"})
                cursor += 1
                continue

            # Start: 0b1111000x + 7 bytes timer
            if (b & 0b11111011) == 0b11110000:
                timer = 0
                for i in range(7):
                    if cursor + i + 1 < len(payload):
                        timer += payload[cursor + i + 1] * (256 ** (6 - i))
                commands.append({"type": "Start", "timer_value": timer})
                cursor += 8
                continue

            # Skip: 0b110111xx + 3 bytes
            if (b & 0b11111100) == 0b11011100:
                cursor += 4
                continue

            # Single0: 0b0EEETTTT (1 byte)
            if (b & 0b10000000) == 0:
                commands.append({
                    "type": "Single0",
                    "event": (b >> 4) & 0x7,
                    "cycles": b & 0xF,
                })
                cursor += 1
                continue

            # Single1: 0b100EEETTT TTTTTTTT (2 bytes)
            if (b & 0b11100000) == 0b10000000:
                slot = (b >> 2) & 0x7
                delta = ((b & 0x3) << 8) | payload[cursor + 1]
                commands.append({"type": "Single1", "event": slot, "cycles": delta})
                cursor += 2
                continue

            # Single2: 0b101EEETTT TTTTTTTT TTTTTTTT (3 bytes)
            if (b & 0b11100000) == 0b10100000:
                slot = (b >> 2) & 0x7
                delta = ((b & 0x3) << 16) | (payload[cursor+1] << 8) | payload[cursor+2]
                commands.append({"type": "Single2", "event": slot, "cycles": delta})
                cursor += 3
                continue

            # Multiple0: 0b1100EEEE EEEETTTT (2 bytes)
            if (b & 0b11110000) == 0b11000000:
                events_bits = ((b & 0xF) << 4) | (payload[cursor+1] >> 4)
                delta = payload[cursor+1] & 0xF
                cmd = {"type": "Multiple0", "cycles": delta}
                for i in range(8):
                    if (events_bits >> i) & 1:
                        cmd[f"event{i}"] = i
                commands.append(cmd)
                cursor += 2
                continue

            # Multiple1: 0b110100EE EEEEEETTT TTTTTTTT (3 bytes)
            if (b & 0b11111100) == 0b11010000:
                events_bits = ((b & 0x3) << 6) | (payload[cursor+1] >> 2)
                delta = ((payload[cursor+1] & 0x3) << 8) | payload[cursor+2]
                cmd = {"type": "Multiple1", "cycles": delta}
                for i in range(8):
                    if (events_bits >> i) & 1:
                        cmd[f"event{i}"] = i
                commands.append(cmd)
                cursor += 3
                continue

            # Multiple2: 0b110101EE EEEEEETTT TTTTTTTT TTTTTTTT (4 bytes)
            if (b & 0b11111100) == 0b11010100:
                events_bits = ((b & 0x3) << 6) | (payload[cursor+1] >> 2)
                delta = ((payload[cursor+1] & 0x3) << 16) | (payload[cursor+2] << 8) | payload[cursor+3]
                cmd = {"type": "Multiple2", "cycles": delta}
                for i in range(8):
                    if (events_bits >> i) & 1:
                        cmd[f"event{i}"] = i
                commands.append(cmd)
                cursor += 4
                continue

            # Repeat0: 0b1110RRRR (1 byte)
            if (b & 0b11110000) == 0b11100000:
                commands.append({"type": "Repeat0", "repeats": b & 0xF})
                cursor += 1
                continue

            # Repeat1: 0b110110RR RRRRRRRR (2 bytes)
            if (b & 0b11111100) == 0b11011000:
                repeats = ((b & 0x3) << 8) | payload[cursor+1]
                commands.append({"type": "Repeat1", "repeats": repeats})
                cursor += 2
                continue

            # Unknown -- skip
            cursor += 1

    except IndexError:
        pass

    return commands


def commands_to_events(commands: list[dict]) -> list[tuple[int, int]]:
    """Convert decoded commands into a flat list of (absolute_cycle, slot).

    Handles Single (one slot per command), Multiple (bitmap of slots),
    and Repeat (replays the previous command's events).
    """
    events = []
    cycle = 0
    last_slots = []

    for cmd in commands:
        t = cmd["type"]

        if t == "Start":
            cycle = cmd.get("timer_value", 0)
            continue
        if t in ("Event_Sync",):
            continue

        if t in ("Single0", "Single1", "Single2"):
            cycle += cmd["cycles"]
            slot = cmd["event"]
            events.append((cycle, slot))
            last_slots = [slot]

        elif t in ("Multiple0", "Multiple1", "Multiple2"):
            cycle += cmd["cycles"]
            slots = [cmd[k] for k in sorted(cmd) if k.startswith("event")]
            for s in slots:
                events.append((cycle, s))
            last_slots = slots

        elif t in ("Repeat0", "Repeat1"):
            repeats = cmd["repeats"]
            for _ in range(repeats):
                cycle += 1  # Repeat implies delta=1
                for s in last_slots:
                    events.append((cycle, s))

    return events


def decode_trace_binary(
    raw: np.ndarray,
) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
    """Decode a raw trace binary into per-tile event lists.

    Returns: {(col, row, pkt_type): [(absolute_cycle, slot_index), ...]}
    """
    packets = decode_packets(raw)
    tiles = {}

    for pkt in packets:
        key = (pkt["col"], pkt["row"], pkt["pkt_type"])
        commands = decode_byte_stream(pkt["payload"])
        tile_events = commands_to_events(commands)
        tiles.setdefault(key, []).extend(tile_events)

    # Sort each tile's events by cycle
    for key in tiles:
        tiles[key].sort()

    return tiles


def slot_name(slot: int, slot_names: list[str]) -> str:
    """Map a slot index to its configured event name."""
    if slot < len(slot_names):
        return slot_names[slot]
    return f"slot{slot}"


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

# Level events fire every cycle a condition holds.  They produce huge
# runs of repeated events whose raw counts depend on execution duration,
# not on how many times the condition *transitioned*.  Comparing them by
# occurrence index is meaningless -- compare intervals instead.
LEVEL_EVENTS = {
    "TRUE", "ACTIVE", "DISABLED",
    "LOCK_STALL", "MEMORY_STALL", "STREAM_STALL", "CASCADE_STALL",
    "PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2", "PORT_RUNNING_3",
    "PORT_RUNNING_4", "PORT_RUNNING_5", "PORT_RUNNING_6", "PORT_RUNNING_7",
    "PORT_IDLE_0", "PORT_IDLE_1", "PORT_IDLE_2", "PORT_IDLE_3",
    "PORT_IDLE_4", "PORT_IDLE_5", "PORT_IDLE_6", "PORT_IDLE_7",
    "PORT_STALLED_0", "PORT_STALLED_1", "PORT_STALLED_2", "PORT_STALLED_3",
    "PORT_STALLED_4", "PORT_STALLED_5", "PORT_STALLED_6", "PORT_STALLED_7",
    # DMA stalls are also level events (fire every stalled cycle)
    "DMA_S2MM_0_STALLED_LOCK", "DMA_S2MM_1_STALLED_LOCK",
    "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_1_STALLED_LOCK",
    "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_1_STREAM_STARVATION",
    "DMA_MM2S_0_STREAM_BACKPRESSURE", "DMA_MM2S_1_STREAM_BACKPRESSURE",
    "DMA_S2MM_0_MEMORY_BACKPRESSURE", "DMA_S2MM_1_MEMORY_BACKPRESSURE",
    "DMA_MM2S_0_MEMORY_STARVATION", "DMA_MM2S_1_MEMORY_STARVATION",
    "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
    "CONFLICT_DM_BANK_2", "CONFLICT_DM_BANK_3",
}


def events_to_intervals(
    cycles: list[int],
) -> list[tuple[int, int]]:
    """Convert a sorted list of event cycles into contiguous intervals.

    Groups consecutive cycles (delta <= 1) into (start, end) intervals.
    Non-consecutive firings produce separate intervals.

    Example: [10, 11, 12, 20, 21] -> [(10, 12), (20, 21)]
    """
    if not cycles:
        return []
    intervals = []
    start = cycles[0]
    prev = cycles[0]
    for c in cycles[1:]:
        if c - prev > 1:
            intervals.append((start, prev))
            start = c
        prev = c
    intervals.append((start, prev))
    return intervals


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def find_edge_anchor(
    hw_events: list[tuple[int, int]],
    emu_events: list[tuple[int, int]],
    slot_names: list[str],
) -> tuple[int, int]:
    """Find the first shared edge event to use as the alignment anchor.

    Returns (hw_t0, emu_t0) such that subtracting these from their
    respective timelines aligns the first shared edge event at t=0.
    """
    # Group by slot, collect first occurrence cycle
    hw_first = {}
    for c, s in hw_events:
        if s not in hw_first:
            hw_first[s] = c
    emu_first = {}
    for c, s in emu_events:
        if s not in emu_first:
            emu_first[s] = c

    # Find first slot that's an edge event in both
    for slot in sorted(set(hw_first) & set(emu_first)):
        name = slot_name(slot, slot_names)
        if name not in LEVEL_EVENTS and name != "TRUE":
            return hw_first[slot], emu_first[slot]

    # Fallback: first non-TRUE event in each
    hw_t0 = next((c for c, s in hw_events if s != 0 and c > 0), 0)
    emu_t0 = next((c for c, s in emu_events if s != 0 and c > 0), 0)
    return hw_t0, emu_t0


def compare_tile_events(
    hw_events: list[tuple[int, int]],
    emu_events: list[tuple[int, int]],
    slot_names: list[str],
) -> dict:
    """Compare one tile's HW vs EMU event streams.

    Uses edge-event alignment and per-event-name occurrence matching.
    Level events are compared by interval structure.
    """
    # Find alignment anchor (first shared edge event)
    hw_t0, emu_t0 = find_edge_anchor(hw_events, emu_events, slot_names)

    # Rebase both to the anchor
    hw_rebased = [(c - hw_t0, s) for c, s in hw_events]
    emu_rebased = [(c - emu_t0, s) for c, s in emu_events]

    # Group events by slot (= event name)
    hw_by_slot: dict[int, list[int]] = {}
    emu_by_slot: dict[int, list[int]] = {}
    for c, s in hw_rebased:
        hw_by_slot.setdefault(s, []).append(c)
    for c, s in emu_rebased:
        emu_by_slot.setdefault(s, []).append(c)

    # Compare per event name
    all_slots = sorted(set(hw_by_slot) | set(emu_by_slot))
    edge_results = []
    level_results = []

    for slot in all_slots:
        name = slot_name(slot, slot_names)
        hw_cycles = sorted(hw_by_slot.get(slot, []))
        emu_cycles = sorted(emu_by_slot.get(slot, []))

        if name in LEVEL_EVENTS:
            hw_ivs = events_to_intervals(hw_cycles)
            emu_ivs = events_to_intervals(emu_cycles)
            level_results.append(analyze_level_event(name, hw_ivs, emu_ivs))
        else:
            edge_results.append(
                analyze_edge_event(name, hw_cycles, emu_cycles)
            )

    return {
        "hw_t0": hw_t0,
        "emu_t0": emu_t0,
        "edge_results": edge_results,
        "level_results": level_results,
    }


# Divergence threshold: deltas above this are considered a real divergence,
# not just micro-timing jitter.  DMA transfers on real hardware have
# inherent cycle-level variation, so small deltas are expected.
DIVERGE_THRESHOLD = 10


def analyze_edge_event(
    name: str,
    hw_cycles: list[int],
    emu_cycles: list[int],
) -> dict:
    """Analyze one edge event type across HW and EMU.

    Pairs by occurrence index, computes per-pair delta, and finds
    the divergence point (first occurrence where |delta| > threshold).
    Also tracks whether drift is gradual (accumulating) or sudden (jump).
    """
    paired_count = min(len(hw_cycles), len(emu_cycles))
    deltas = [hw_cycles[i] - emu_cycles[i] for i in range(paired_count)]

    # Find divergence point: first index where |delta| > threshold
    diverge_idx = None
    for i, d in enumerate(deltas):
        if abs(d) > DIVERGE_THRESHOLD:
            diverge_idx = i
            break

    # Classify drift pattern in the divergent tail
    drift_type = "none"
    if diverge_idx is not None and diverge_idx < len(deltas) - 1:
        tail = deltas[diverge_idx:]
        # Check if deltas are monotonically increasing in magnitude (gradual)
        # or jump suddenly then stay flat (sudden)
        diffs = [abs(tail[i+1]) - abs(tail[i]) for i in range(len(tail)-1)]
        growing = sum(1 for d in diffs if d > 0)
        if growing > len(diffs) * 0.6:
            drift_type = "accumulating"
        elif all(abs(d) < DIVERGE_THRESHOLD for d in diffs):
            drift_type = "constant_offset"
        else:
            drift_type = "irregular"

    # Samples around the divergence point (context window)
    if diverge_idx is not None:
        ctx_start = max(0, diverge_idx - 2)
        ctx_end = min(paired_count, diverge_idx + 5)
    else:
        ctx_start = 0
        ctx_end = min(6, paired_count)

    samples = [
        (i, hw_cycles[i], emu_cycles[i], deltas[i])
        for i in range(ctx_start, ctx_end)
    ]

    return {
        "name": name,
        "hw_count": len(hw_cycles),
        "emu_count": len(emu_cycles),
        "paired": paired_count,
        "deltas": deltas,
        "diverge_idx": diverge_idx,
        "drift_type": drift_type,
        "samples": samples,
    }


def analyze_level_event(
    name: str,
    hw_ivs: list[tuple[int, int]],
    emu_ivs: list[tuple[int, int]],
) -> dict:
    """Analyze one level event type by interval structure.

    Compares interval count, start offsets, and durations.  Finds the
    first interval where duration diverges significantly.
    """
    paired_count = min(len(hw_ivs), len(emu_ivs))

    start_deltas = []
    dur_deltas = []
    diverge_idx = None

    for i in range(paired_count):
        hw_dur = hw_ivs[i][1] - hw_ivs[i][0] + 1
        emu_dur = emu_ivs[i][1] - emu_ivs[i][0] + 1
        sd = hw_ivs[i][0] - emu_ivs[i][0]
        dd = hw_dur - emu_dur
        start_deltas.append(sd)
        dur_deltas.append(dd)
        if diverge_idx is None and abs(dd) > DIVERGE_THRESHOLD:
            diverge_idx = i

    # Collect interval samples around divergence (or first few)
    if diverge_idx is not None:
        ctx_start = max(0, diverge_idx - 1)
        ctx_end = min(paired_count, diverge_idx + 4)
    else:
        ctx_start = 0
        ctx_end = min(4, paired_count)

    samples = []
    for i in range(ctx_start, ctx_end):
        hw_s, hw_e = hw_ivs[i]
        emu_s, emu_e = emu_ivs[i]
        samples.append((i, hw_s, hw_e, emu_s, emu_e))

    return {
        "name": name,
        "hw_intervals": len(hw_ivs),
        "emu_intervals": len(emu_ivs),
        "paired": paired_count,
        "start_deltas": start_deltas,
        "dur_deltas": dur_deltas,
        "diverge_idx": diverge_idx,
        "samples": samples,
    }


def compare_batch(
    hw_raw_path: Path,
    emu_raw_path: Path,
    events_config: dict,
) -> dict:
    """Compare one batch: decode, align by edge events, compare per-tile."""
    hw_raw = np.fromfile(str(hw_raw_path), dtype=np.uint32)
    emu_raw = np.fromfile(str(emu_raw_path), dtype=np.uint32)

    hw_tiles = decode_trace_binary(hw_raw)
    emu_tiles = decode_trace_binary(emu_raw)

    core_names = events_config.get("core_events", [])
    mem_names = events_config.get("mem_events", [])

    results = {}
    all_keys = sorted(set(hw_tiles) | set(emu_tiles))

    for key in all_keys:
        col, row, pkt_type = key
        hw_ev = hw_tiles.get(key, [])
        emu_ev = emu_tiles.get(key, [])
        names = core_names if pkt_type == 0 else mem_names
        results[key] = compare_tile_events(hw_ev, emu_ev, names)

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    batch_results: list[tuple[int, dict, dict]],
) -> str:
    """Format comparison results into a readable report."""
    lines = []
    lines.append("=" * 76)
    lines.append("Raw Binary Trace Comparison")
    lines.append(f"  Alignment:  first shared edge event per tile")
    lines.append(f"  Edge:       paired by occurrence index, divergence at |dt|>{DIVERGE_THRESHOLD}")
    lines.append(f"  Level:      compared by interval structure")
    lines.append("=" * 76)
    lines.append("")

    all_edge_deltas_clean = []   # Deltas before divergence
    all_edge_deltas_total = []   # All deltas
    total_edge_clean = 0         # Event types with no divergence
    total_edge_diverged = 0      # Event types that diverge
    total_edge_count_mismatch = 0
    total_level_clean = 0
    total_level_diverged = 0
    total_level_count_mismatch = 0
    divergence_details = []      # (batch, tile, name, idx, context) for summary

    for batch_idx, batch_events_config, batch_tiles in batch_results:
        core_names = batch_events_config.get("core_events", [])
        mem_names = batch_events_config.get("mem_events", [])
        active_core = [n for n in core_names if n not in ("TRUE", "NONE")]
        active_mem = [n for n in mem_names if n not in ("TRUE", "NONE")]

        lines.append(f"--- Batch {batch_idx} ---")
        lines.append(f"  Core: {', '.join(active_core) if active_core else '(none)'}")
        lines.append(f"  Mem:  {', '.join(active_mem) if active_mem else '(none)'}")
        lines.append("")

        for key in sorted(batch_tiles):
            col, row, pkt_type = key
            r = batch_tiles[key]
            module = "Core" if pkt_type == 0 else "Mem"

            lines.append(
                f"  Tile ({col},{row}) {module}  "
                f"(anchor: HW cy {r['hw_t0']}, EMU cy {r['emu_t0']})"
            )

            # -- Edge events --
            for er in r["edge_results"]:
                if er["name"] == "TRUE":
                    continue

                name = er["name"]
                hw_n, emu_n = er["hw_count"], er["emu_count"]
                deltas = er["deltas"]
                div_idx = er["diverge_idx"]

                all_edge_deltas_total.extend(deltas)

                # Count match?
                count_ok = hw_n == emu_n
                if not count_ok:
                    total_edge_count_mismatch += 1

                # Clean deltas = everything before the divergence point
                if div_idx is not None:
                    clean = deltas[:div_idx]
                    total_edge_diverged += 1
                else:
                    clean = deltas
                    total_edge_clean += 1
                all_edge_deltas_clean.extend(clean)

                # Format header
                count_str = f"{hw_n}/{emu_n}"
                if not count_ok:
                    count_str += " COUNTS DIFFER"
                status = "OK" if div_idx is None else f"DIVERGES at #{div_idx}"
                lines.append(f"    [edge] {name:<32s} {count_str:<20s} {status}")

                # Show timing for clean portion
                if clean:
                    lines.append(
                        f"           Clean ({len(clean)} pairs): "
                        f"min={min(clean):+d} max={max(clean):+d} "
                        f"mean={sum(clean)/len(clean):+.1f}"
                    )

                # Show context around divergence or first few events
                for idx, hw_c, emu_c, dt in er["samples"]:
                    marker = ""
                    if div_idx is not None and idx == div_idx:
                        marker = "  <<< DIVERGENCE"
                    lines.append(
                        f"           [{idx:>4d}] HW={hw_c:<8d} EMU={emu_c:<8d} "
                        f"dt={dt:+d}{marker}"
                    )

                # Divergence annotation
                if div_idx is not None:
                    drift = er["drift_type"]
                    lines.append(f"           Drift pattern: {drift}")
                    divergence_details.append(
                        (batch_idx, f"({col},{row}) {module}", name, div_idx, drift)
                    )

            # -- Level events --
            for lr in r["level_results"]:
                if lr["name"] == "TRUE":
                    continue

                name = lr["name"]
                hw_n, emu_n = lr["hw_intervals"], lr["emu_intervals"]
                div_idx = lr["diverge_idx"]
                dur_deltas = lr["dur_deltas"]

                count_ok = hw_n == emu_n
                if not count_ok:
                    total_level_count_mismatch += 1

                if div_idx is not None:
                    total_level_diverged += 1
                else:
                    total_level_clean += 1

                count_str = f"{hw_n}/{emu_n} intervals"
                if not count_ok:
                    count_str += " DIFFER"
                status = "OK" if div_idx is None else f"DURATION DIVERGES at interval #{div_idx}"
                lines.append(f"    [level] {name:<31s} {count_str:<20s} {status}")

                # Duration summary for clean intervals
                if dur_deltas:
                    clean_dur = dur_deltas[:div_idx] if div_idx is not None else dur_deltas
                    if clean_dur:
                        lines.append(
                            f"            Clean durations ({len(clean_dur)}): "
                            f"min={min(clean_dur):+d} max={max(clean_dur):+d} "
                            f"mean={sum(clean_dur)/len(clean_dur):+.1f}"
                        )

                # Show interval samples
                for idx, hw_s, hw_e, emu_s, emu_e in lr["samples"]:
                    hw_d = hw_e - hw_s + 1
                    emu_d = emu_e - emu_s + 1
                    dd = hw_d - emu_d
                    marker = ""
                    if div_idx is not None and idx == div_idx:
                        marker = "  <<< DIVERGENCE"
                    lines.append(
                        f"            [{idx:>2d}] HW={hw_s}-{hw_e} ({hw_d}cy)  "
                        f"EMU={emu_s}-{emu_e} ({emu_d}cy)  "
                        f"dt_dur={dd:+d}{marker}"
                    )

                if div_idx is not None:
                    divergence_details.append(
                        (batch_idx, f"({col},{row}) {module}", name, div_idx, "duration")
                    )

            lines.append("")

    # ---- Divergence summary ----
    if divergence_details:
        lines.append("=" * 76)
        lines.append("Divergence Points (where HW and EMU first disagree)")
        lines.append("=" * 76)
        for batch, tile, name, idx, drift in divergence_details:
            lines.append(f"  Batch {batch}  {tile:<16s} {name:<34s} #{idx:<5d} ({drift})")
        lines.append("")

    # ---- Overall summary ----
    lines.append("=" * 76)
    lines.append("Summary")
    lines.append("=" * 76)
    lines.append(f"Batches:             {len(batch_results)}")
    lines.append("")
    lines.append(f"Edge event types:    {total_edge_clean} clean, {total_edge_diverged} diverged, {total_edge_count_mismatch} count mismatch")
    lines.append(f"Level event types:   {total_level_clean} clean, {total_level_diverged} diverged, {total_level_count_mismatch} count mismatch")

    # Timing stats for clean edge events (before any divergence)
    if all_edge_deltas_clean:
        sorted_c = sorted(all_edge_deltas_clean)
        n = len(sorted_c)
        lines.append("")
        lines.append(f"Edge timing (CLEAN pairs only -- before divergence):")
        lines.append(f"  Pairs:           {n}")
        lines.append(f"  Min:             {sorted_c[0]:+d}")
        lines.append(f"  Max:             {sorted_c[-1]:+d}")
        lines.append(f"  Mean:            {sum(sorted_c)/n:+.1f}")
        lines.append(f"  Median:          {sorted_c[n//2]:+d}")
        lines.append(f"  Spread:          {sorted_c[-1]-sorted_c[0]}")
        lines.append(f"")
        for threshold in (0, 1, 2, 5, 10):
            within = sum(1 for d in sorted_c if abs(d) <= threshold)
            pct = within / n * 100
            lines.append(f"  Within +/-{threshold:>3d}:    {within:>4d}/{n} ({pct:.1f}%)")

    # Also show total (including post-divergence) for completeness
    if all_edge_deltas_total and len(all_edge_deltas_total) != len(all_edge_deltas_clean):
        sorted_t = sorted(all_edge_deltas_total)
        n = len(sorted_t)
        lines.append(f"")
        lines.append(f"Edge timing (ALL pairs including post-divergence):")
        lines.append(f"  Pairs:           {n}")
        lines.append(f"  Spread:          {sorted_t[-1]-sorted_t[0]}")
        for threshold in (0, 1, 2, 5, 10, 50):
            within = sum(1 for d in sorted_t if abs(d) <= threshold)
            pct = within / n * 100
            lines.append(f"  Within +/-{threshold:>3d}:    {within:>4d}/{n} ({pct:.1f}%)")

    lines.append("=" * 76)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def compare_sweep_dir(sweep_dir: Path) -> str:
    """Compare all batches in a sweep directory."""
    manifest_path = sweep_dir / "sweep-manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    batch_results = []

    for batch_info in manifest["batches"]:
        idx = batch_info["batch"]
        if batch_info.get("status") != "ok":
            continue
        if batch_info.get("hw_status") != "ok" or batch_info.get("emu_status") != "ok":
            continue

        batch_dir = sweep_dir / f"batch_{idx:02d}"
        hw_raw = batch_dir / "hw" / "trace_raw.bin"
        emu_raw = batch_dir / "emu" / "trace_raw.bin"
        events_json = batch_dir / "events.json"

        if not hw_raw.exists() or not emu_raw.exists():
            continue

        events_config = {}
        if events_json.exists():
            events_config = json.loads(events_json.read_text())

        tiles = compare_batch(hw_raw, emu_raw, events_config)
        batch_results.append((idx, events_config, tiles))

    return format_report(batch_results)


def compare_single(hw_path: Path, emu_path: Path, events_config: dict) -> str:
    """Compare a single HW vs EMU trace binary pair."""
    tiles = compare_batch(hw_path, emu_path, events_config)
    return format_report([(0, events_config, tiles)])


def main():
    parser = argparse.ArgumentParser(
        description="Compare HW vs EMU traces at the raw binary level",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sweep", type=Path,
        help="Sweep directory from trace-sweep.py",
    )
    group.add_argument(
        "--hw", type=Path,
        help="Hardware trace_raw.bin (use with --emu)",
    )
    parser.add_argument("--emu", type=Path, help="Emulator trace_raw.bin")
    parser.add_argument(
        "--events-json", type=Path,
        help="Events config JSON (for --hw/--emu mode)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write report to file",
    )
    args = parser.parse_args()

    if args.sweep:
        report = compare_sweep_dir(args.sweep)
    else:
        if not args.emu:
            parser.error("--emu required when using --hw")
        events_config = {}
        if args.events_json and args.events_json.exists():
            events_config = json.loads(args.events_json.read_text())
        report = compare_single(args.hw, args.emu, events_config)

    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
