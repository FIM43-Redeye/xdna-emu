# SPDX-License-Identifier: MIT
"""Mode-1 wire-format exploration helper.

Read raw mode-0 and mode-1 trace BO dumps captured from the same kernel,
trim padding, de-interleave by tile, and dump the byte stream for each
tile alongside a tentative mode-0 decode of the same bytes.  This is
research code -- not part of the public package -- but lives in-tree so
it survives reboots.

Usage::

    python3 -m tools.trace_decoder.explore_mode1 <m0.bin> <m1.bin>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_decoder.modes.mode0 import decode as decode_mode0
from trace_decoder.packet import (
    PACKET_WORDS,
    deinterleave_packets,
    trim_trailing_padding,
    words_to_bytes,
)


def _load_words(path: Path) -> list[int]:
    raw = np.fromfile(path, dtype=np.uint32).tolist()
    raw = trim_trailing_padding(raw)
    return raw


def _hex_bytes(bs: list[int], per_line: int = 32) -> str:
    out = []
    for i in range(0, len(bs), per_line):
        chunk = bs[i : i + per_line]
        out.append(f"{i:04x}  " + " ".join(f"{b:02x}" for b in chunk))
    return "\n".join(out)


def _summarize_tile(label: str, words: list[int]) -> None:
    print(f"\n=== {label} ({len(words)} words) ===")
    by_tile = deinterleave_packets(words)
    for key, payload in by_tile.items():
        pkt_type, row, col = key
        bs = words_to_bytes(payload)
        # strip trailing 0xfe filler so we don't drown in skip bytes
        while bs and bs[-1] in (0xFE, 0x00):
            bs.pop()
        print(f"\n-- tile pkt_type={pkt_type} row={row} col={col} "
              f"({len(payload)} payload words / {len(bs)} sig bytes)")
        print(_hex_bytes(bs))
        # naive mode-0 decode for first ~30 commands
        try:
            cmds = list(decode_mode0(bs))
        except Exception as e:  # pragma: no cover - debug helper
            print(f"  decode error: {e}")
            cmds = []
        for i, cmd in enumerate(cmds[:40]):
            print(f"  [{i:3d}] {cmd}")
        if len(cmds) > 40:
            print(f"  ... {len(cmds) - 40} more")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("m0", type=Path)
    ap.add_argument("m1", type=Path)
    args = ap.parse_args()

    m0_words = _load_words(args.m0)
    m1_words = _load_words(args.m1)

    _summarize_tile(f"MODE 0 ({args.m0.name})", m0_words)
    _summarize_tile(f"MODE 1 ({args.m1.name})", m1_words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
