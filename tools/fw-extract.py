#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# fw-extract.py -- strip the 256-byte signed header from a Phoenix NPU
# firmware `.sbin` blob and emit the raw Xtensa body that loads at
# 0x08ad3000. The body is suitable for feeding into Ghidra headless
# analysis with the existing tools/ghidra-scripts/ pipeline.
#
# The `.sbin` format observed on Phoenix (1.5.5.391, 1.5.6.399):
#
#   bytes 0..255     : 256-byte signed header. Contains a `$PS1` magic
#                      at offset 0x10, a 32-bit body size at 0x14
#                      (little-endian), a hash region, and assorted
#                      version/build metadata. Signature lives in the
#                      first 16 bytes (binary, non-printable).
#   bytes 256..N-257 : 248080-byte Xtensa code+data body. Starts with a
#                      build-info struct (zero-filled words then the hex
#                      digest "7bd12109..." followed by "Release X.Y.Z.W").
#                      Loads at absolute address 0x08ad3000 (recovered via
#                      tools/fw-find-base.py).
#   bytes N-256..N-1 : 256-byte trailer (signature?). Not loaded.
#
# This is the same extraction the existing project at
# ghidra-projects/npu-fw/npu-fw-body.bin used; cmp-verified against
# 1.5.5.391.

import argparse
import os
import sys

HEADER_SIZE = 256
TRAILER_SIZE = 256
EXPECTED_BODY_SIZE = 248080
EXPECTED_TOTAL_SIZE = HEADER_SIZE + EXPECTED_BODY_SIZE + TRAILER_SIZE  # 248592
PS1_MAGIC = b"$PS1"
PS1_MAGIC_OFFSET = 0x10
LOAD_BASE = 0x08ad3000  # for downstream consumers; not used here


def extract(sbin_path: str, out_path: str) -> int:
    with open(sbin_path, "rb") as f:
        data = f.read()

    if len(data) != EXPECTED_TOTAL_SIZE:
        print(
            f"warning: {sbin_path} is {len(data)} bytes, "
            f"expected {EXPECTED_TOTAL_SIZE}",
            file=sys.stderr,
        )
    if data[PS1_MAGIC_OFFSET : PS1_MAGIC_OFFSET + 4] != PS1_MAGIC:
        print(
            f"error: {sbin_path} does not have '$PS1' magic at offset 0x10 "
            f"(found {data[PS1_MAGIC_OFFSET:PS1_MAGIC_OFFSET+4]!r})",
            file=sys.stderr,
        )
        return 2

    body = data[HEADER_SIZE : HEADER_SIZE + EXPECTED_BODY_SIZE]
    with open(out_path, "wb") as f:
        f.write(body)

    # Surface the release string so callers can confirm which version they got.
    rel = b""
    for tag in (b"Release ", b"release "):
        i = body.find(tag)
        if i >= 0:
            end = body.find(b"\x00", i)
            rel = body[i:end] if end > 0 else body[i:i+32]
            break
    print(
        f"extracted {len(body)} bytes -> {out_path}"
        + (f"  ({rel.decode(errors='replace')})" if rel else "")
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strip the 256-byte signed header from a Phoenix NPU "
        "firmware .sbin and emit the Xtensa body (loads at 0x08ad3000)."
    )
    ap.add_argument("sbin", help="Phoenix NPU firmware .sbin file")
    ap.add_argument(
        "-o",
        "--output",
        help="output body path (default: <sbin>.body.bin in same dir)",
    )
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.sbin)[0] + ".body.bin"
    return extract(args.sbin, out)


if __name__ == "__main__":
    sys.exit(main())
