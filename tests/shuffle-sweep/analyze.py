#!/usr/bin/env python3
"""Analyze shuffle sweep results.

Reads the 3072-byte output from the shuffle sweep kernel and builds
a routing table: for each mode and output position, which input byte
was selected.

Usage:
    python3 analyze.py <hw_output.bin> [--json routing.json]
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Analyze shuffle sweep output")
    parser.add_argument("input", help="Path to HW output binary (3072 bytes)")
    parser.add_argument("--json", help="Write routing table as JSON", default=None)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()

    expected = 48 * 64
    if len(data) < expected:
        print(f"ERROR: expected {expected} bytes, got {len(data)}", file=sys.stderr)
        sys.exit(1)

    # Build routing table: routing[mode][out_pos] = input_byte_index
    routing = {}
    for mode in range(48):
        offset = mode * 64
        chunk = data[offset:offset + 64]
        route = [int(b) for b in chunk]
        routing[mode] = route

    # Print human-readable summary.
    print("=== Shuffle Network Routing Table ===")
    print(f"Modes: 48, Output positions: 64, Input range: 0-127")
    print()

    for mode in range(48):
        route = routing[mode]
        # Classify the routing pattern.
        from_a = sum(1 for b in route if b < 64)
        from_b = sum(1 for b in route if b >= 64)
        unique = len(set(route))

        # Check if it's a clean stride pattern.
        diffs = [route[i+1] - route[i] for i in range(len(route)-1) if route[i+1] > route[i]]
        stride = diffs[0] if diffs and all(d == diffs[0] for d in diffs) else None

        desc = f"from_A={from_a} from_B={from_b} unique={unique}"
        if stride:
            desc += f" stride={stride}"

        print(f"Mode {mode:2d}: {desc}")
        # Print first 16 bytes for quick visual check.
        print(f"         [{', '.join(f'{b:3d}' for b in route[:16])} ...]")

    # Identify which modes produce the same permutation on repeated data.
    print()
    print("=== Unique permutations ===")
    seen = {}
    for mode in range(48):
        key = tuple(routing[mode])
        if key in seen:
            print(f"  Mode {mode:2d} == Mode {seen[key]}")
        else:
            seen[key] = mode

    print(f"  {len(seen)} unique permutations out of 48 modes")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(routing, f, indent=2)
        print(f"\nRouting table written to {args.json}")


if __name__ == "__main__":
    main()
