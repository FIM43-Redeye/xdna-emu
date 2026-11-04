#!/usr/bin/env python3
"""Generate identity input pattern for shuffle sweep.

128 bytes: byte[i] = i for i in 0..127.
This maps directly to the two 512-bit vectors A (0-63) and B (64-127).
When a shuffle output byte contains value N, it came from input byte N.
"""


def identity_input() -> bytes:
    return bytes(range(128))
