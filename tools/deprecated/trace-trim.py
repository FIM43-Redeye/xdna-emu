#!/usr/bin/env python3
"""Trim trace_raw.bin files to actual data length.

NPU trace buffers are typically 1MB but contain real data only at the front.
The end-of-data boundary is a 0xFEFEFEFE sentinel word followed by two zero
words (matching mlir-aie's trim_trace_pkts heuristic). Everything from the
sentinel onward is padding.

Usage:
    trace-trim.py <trace_raw.bin> [...]       # in-place trim
    trace-trim.py --dir /tmp/sweep-results    # trim all trace_raw.bin under dir
    trace-trim.py --dry-run <file>            # show savings without writing

Exits 0 on success. Reports total savings to stderr.
"""

import argparse
import struct
import sys
from pathlib import Path


def find_trim_point(data: bytes) -> int:
    """Find the byte offset where real trace data ends.

    Scans for 0xFEFEFEFE followed by two 0x00000000 words.  Returns byte
    length of the valid data prefix (includes the sentinel itself, matching
    mlir-aie behavior).  If no trim point is found, returns len(data).
    """
    word_count = len(data) // 4
    for i in range(word_count):
        off = i * 4
        word = struct.unpack_from("<I", data, off)[0]
        if word == 0xFEFEFEFE and i + 2 < word_count:
            w1 = struct.unpack_from("<I", data, off + 4)[0]
            w2 = struct.unpack_from("<I", data, off + 8)[0]
            if w1 == 0 and w2 == 0:
                return (i + 1) * 4
    return len(data)


def trim_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Trim a trace_raw.bin file in place.

    Returns (original_size, trimmed_size).
    """
    data = path.read_bytes()
    original = len(data)
    trim_len = find_trim_point(data)

    if trim_len >= original:
        return (original, original)

    if not dry_run:
        path.write_bytes(data[:trim_len])

    return (original, trim_len)


def main():
    parser = argparse.ArgumentParser(
        description="Trim trace_raw.bin files to actual data length",
    )
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="trace_raw.bin files to trim in place",
    )
    parser.add_argument(
        "--dir", "-d", type=Path, default=None,
        help="Recursively find and trim all trace_raw.bin under this directory",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Report savings without modifying files",
    )
    args = parser.parse_args()

    targets = list(args.files)
    if args.dir:
        targets.extend(sorted(args.dir.rglob("trace_raw.bin")))

    if not targets:
        parser.error("No files specified (use positional args or --dir)")

    total_before = 0
    total_after = 0
    trimmed_count = 0

    for path in targets:
        if not path.exists():
            print(f"  SKIP {path} (not found)", file=sys.stderr)
            continue

        before, after = trim_file(path, dry_run=args.dry_run)
        total_before += before
        total_after += after

        if after < before:
            trimmed_count += 1
            pct = (1 - after / before) * 100 if before > 0 else 0
            label = "would trim" if args.dry_run else "trimmed"
            print(f"  {label} {path}: {before:,} -> {after:,} ({pct:.0f}% saved)")

    if total_before > 0:
        total_pct = (1 - total_after / total_before) * 100
        action = "Would save" if args.dry_run else "Saved"
        print(
            f"\n{action}: {total_before - total_after:,} bytes "
            f"({total_pct:.0f}%) across {trimmed_count}/{len(targets)} files",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
