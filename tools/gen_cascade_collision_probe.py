#!/usr/bin/env python3
"""VMOV_CASCADE_READ collision probe: gen_collision_probe pairs with a cascade read, on two cores.

The single-tile collision probe (gen_collision_probe.py) derives every (A, B, gap) whose
TableGen reservations put two instructions on one resource in one cycle, runs each pair
bundle-exact with controls and solo references, and records both destinations. This tool
runs the same machinery with the cascade reads (`vmov x/bml/bmh, SCD`) in the catalogue
(gen_collision_probe.enable_cascade()) and keeps the pairs that contain one:

  * consumer core (0,2): the generated kernel, unchanged, plus a cascade prologue that enables
    SCD and waits for the producer to fill the cascade;
  * producer core (0,3): enables MCD and writes exactly as many 512-bit words as the kernel
    has SCD reads (a core that writes more than the cascade can hold hangs), all the same word.

The output is a suite (harness/suite.py): each case's `consumer.out.bin` is the record file, so the
record tools (diff_probe_points.py, analyze_collision_probe.py) read it as they read the
single-core probes. The suite carries the `aligned_labels` pre-flight: silicon fetches a taken
branch's target from its 16-byte line start, so a package with an unaligned `.L` label is refused
on hardware (an endless loop, then a TDR per kernel).

  python3 tools/gen_cascade_collision_probe.py <dir> [--limit N]
  python3 -m harness suite build|hw|emu|compare <dir> ...      (PYTHONPATH=tools)
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_cascade_probe as casc                                   # noqa: E402
import gen_collision_probe as cp                                   # noqa: E402
from harness import suite                                           # noqa: E402
from harness.topology import Buffer, Cascade, Core, TileRef, Topology   # noqa: E402

CASCADE_KEYS = [i.key for i in cp.CATALOGUE_CASCADE]
PRODUCER_TILE, CONSUMER_TILE = (0, 3), (0, 2)       # South: producer directly north of the consumer
WORD_LANES = 16


def cascade_word() -> bytes:
    """The word every SCD read returns: 16 distinct lanes, none equal to the probe's table values."""
    return struct.pack(f"<{WORD_LANES}I", *[0x5C0D0000 + k for k in range(WORD_LANES)])


def scd_reads(kernel_asm: str) -> int:
    """SCD reads in a straight-line kernel (each executes once)."""
    n = 0
    for line in kernel_asm.splitlines():
        code = line.split("//")[0]
        n += len(re.findall(r"\bSCD\b", code))
    return n


def producer_asm(words: int) -> str:
    body = casc._enable("crMCDEn")
    body += ["  vlda wl0, [p0, #0]", "  vlda wh0, [p0, #32]", *casc._nops(casc.LOAD_NOPS)]
    body += ["  vmov MCD, x0"] * words
    return casc._frame(body)


def case_topology(name: str, kernel: dict, consumer_asm: str, input_bytes: bytes) -> tuple[Topology, str]:
    """Two-core topology for one collision kernel, and the producer's asm."""
    words = scd_reads(consumer_asm)
    (pc, pr), (cc, cr) = PRODUCER_TILE, CONSUMER_TILE
    topo = Topology(
        name=f"cascol_{name}",
        cores=(Core("producer", TileRef(pc, pr), Path("producer.s"),
                    args=(Buffer("word", "in", 64, data=cascade_word()),)),
               Core("consumer", TileRef(cc, cr), Path("consumer.s"),
                    args=(Buffer("in", "in", kernel["in_size"], data=input_bytes),
                          Buffer("out", "out", kernel["out_size"])))),
        links=(Cascade("producer", "consumer"),))
    return topo, producer_asm(words)


def cmd_generate(a) -> int:
    """Write a suite: the single-core collision kernels that read SCD, each paired with a producer."""
    d = Path(a.dir).resolve()
    cp.enable_cascade()
    cp.ANY_KEYS.update(CASCADE_KEYS)
    kernels, meta, _ = cp.plan_kernels(a.limit)
    table = cp.TABLE_BYTES.ljust(cp.IN_SIZE, b"\0")
    names = []
    for entry, asm in kernels:
        topo, prod = case_topology(entry["name"], entry, asm, table)
        suite.write_case(d, entry["name"], topo, files={"consumer.s": asm, "producer.s": prod},
                         meta={**entry, "scd_reads": scd_reads(asm),
                               "result": {"core": "consumer", "buffer": "out"}})
        names.append(entry["name"])
    suite.write_suite_json(d, "cascade-collisions", "gen_cascade_collision_probe.py", names,
                           meta={"in_size": cp.IN_SIZE, "record_bytes": cp.RECORD_BYTES, **meta},
                           hw_checks=["aligned_labels"])
    pts = [p for e, _ in kernels for p in e["points"]]
    print(f"{len(kernels)} kernels, {len(pts)} points")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--limit", type=int)
    ap.set_defaults(fn=cmd_generate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
