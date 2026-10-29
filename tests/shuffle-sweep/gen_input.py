#!/usr/bin/env python3
"""Generate identity input pattern for shuffle sweep.

128 bytes: byte[i] = i for i in 0..127.
This maps directly to the two 512-bit vectors A (0-63) and B (64-127).
When a shuffle output byte contains value N, it came from input byte N.
"""

import struct
import sys

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "input.bin"
    data = bytes(range(128))
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"Wrote {len(data)} bytes to {out_path}")

if __name__ == "__main__":
    main()
