#!/usr/bin/env python3
"""VMOV_CASCADE_READ collision probe: gen_collision_probe pairs with a cascade read, on two cores.

The single-tile collision probe (gen_collision_probe.py) derives every (A, B, gap) whose
TableGen reservations put two instructions on one resource in one cycle, runs each pair
bundle-exact with controls and solo references, and records both destinations. This tool
runs the same machinery with the cascade reads (`vmov x/bml/bmh, SCD`) in the catalogue
(COLLISION_CASCADE=1) and keeps the pairs that contain one:

  * consumer core (0,2): the generated kernel, unchanged, plus a cascade prologue that enables
    SCD and waits for the producer to fill the cascade;
  * producer core (0,3): enables MCD and writes exactly as many 512-bit words as the kernel
    has SCD reads (a core that writes more than the cascade can hold hangs), all the same word.

The topology is run through the harness; outputs are copied to `<dir>/<tag>/<kernel>.bin` so
the existing record tools (diff_probe_points.py, analyze_collision_probe.py) apply unchanged.

  python3 tools/gen_cascade_collision_probe.py generate <dir> [--limit N]
  python3 tools/gen_cascade_collision_probe.py build    <dir>
  python3 tools/gen_cascade_collision_probe.py hw       <dir> <tag>     # serial, one NPU job at a time
  python3 tools/gen_cascade_collision_probe.py emu      <dir> <tag>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

os.environ["COLLISION_CASCADE"] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_cascade_probe as casc                                   # noqa: E402
import gen_collision_probe as cp                                   # noqa: E402
from harness.build import build_package                            # noqa: E402
from harness.run import run_package                                # noqa: E402
from harness.topology import Buffer, Cascade, Core, TileRef, Topology, load_topology, to_dict   # noqa: E402

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


def topology_for(d: Path, kernel: dict) -> Topology:
    name = kernel["name"]
    asm = (d / f"{name}.s").read_text()
    words = scd_reads(asm)
    psrc = d / f"{name}.producer.s"
    psrc.write_text(producer_asm(words))
    (pc, pr), (cc, cr) = PRODUCER_TILE, CONSUMER_TILE
    return Topology(
        name=f"cascol_{name}",
        cores=(Core("producer", TileRef(pc, pr), psrc,
                    args=(Buffer("word", "in", 64, data=cascade_word()),)),
               Core("consumer", TileRef(cc, cr), d / f"{name}.s",
                    args=(Buffer("in", "in", kernel["in_size"], data=(d / "input.bin").read_bytes()),
                          Buffer("out", "out", kernel["out_size"])))),
        links=(Cascade("producer", "consumer"),))


def cmd_generate(a) -> int:
    d = Path(a.dir).resolve()
    cp.ANY_KEYS.update(CASCADE_KEYS)
    m = cp.generate(str(d), a.limit)
    for k in m["kernels"]:
        t = topology_for(d, k)
        (d / f"{k['name']}.topology.json").write_text(json.dumps(to_dict(t), indent=2) + "\n")
        k["scd_reads"] = scd_reads((d / f"{k['name']}.s").read_text())
    (d / "manifest.json").write_text(json.dumps(m, indent=1))
    pts = [p for k in m["kernels"] for p in k["points"]]
    print(f"{len(m['kernels'])} kernels, {len(pts)} points")
    return 0


def kernels(d: Path) -> list[dict]:
    return json.loads((d / "manifest.json").read_text())["kernels"]


def label_addresses(pkg: Path) -> dict[str, int]:
    """Addresses of the consumer's `.L` labels in the linked ELF."""
    import subprocess
    elf = next((pkg / "phase_0" / "aie.mlir.prj").glob("elfs_main_core_0_2/*.elf"))
    dump = subprocess.run([casc_tool("llvm-objdump"), "-d", str(elf)], capture_output=True, text=True,
                          check=True).stdout
    return {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+) <(\.L\w+)>:", dump, re.M)}


def casc_tool(name: str) -> str:
    from harness.build import _peano_tool
    return _peano_tool(name)


def assert_aligned(pkg: Path) -> None:
    """Silicon fetches a taken branch's target from its 16-byte line start, so an unaligned loop
    label re-runs the code before it (an endless loop, then a TDR per kernel). Never run an
    unaligned package on hardware."""
    for name, addr in label_addresses(pkg).items():
        if addr % 16:
            raise SystemExit(f"{pkg}: label {name} at {addr:#x} is not 16-byte aligned; not running")


def cmd_build(a) -> int:
    d = Path(a.dir).resolve()
    for k in kernels(d):
        pkg = d / "pkgs" / k["name"]
        if not (pkg / "manifest.json").exists():
            build_package(load_topology(d / f"{k['name']}.topology.json"), pkg)
        assert_aligned(pkg)
    return 0


def cmd_run(a, backend: str) -> int:
    d = Path(a.dir).resolve()
    out = d / a.tag
    out.mkdir(exist_ok=True)
    bad = 0
    for k in kernels(d):
        pkg = d / "pkgs" / k["name"]
        assert_aligned(pkg)
        kw = {"runtime": "release", "rebuild_ffi": False} if backend == "emu" else {}
        try:
            rec = run_package(pkg, backend, a.tag, **kw)
        except Exception as e:                                 # noqa: BLE001
            print(f"{k['name']}: {e}")
            bad += 1
            continue
        if not rec.ok:
            print(f"{k['name']}: run failed")
            bad += 1
            continue
        shutil.copyfile(pkg / a.tag / "consumer.out.bin", out / f"{k['name']}.bin")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("dir")
    g.add_argument("--limit", type=int)
    g.set_defaults(fn=cmd_generate)
    b = sub.add_parser("build")
    b.add_argument("dir")
    b.set_defaults(fn=cmd_build)
    for name in ("hw", "emu"):
        r = sub.add_parser(name)
        r.add_argument("dir")
        r.add_argument("tag")
        r.set_defaults(fn=lambda a, n=name: cmd_run(a, n))
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
