"""Silicon regression gates for the harness (step 4).

Gate A: every kernel of the recorded collision probes (probe, probe12) as a
one-core topology; the harness' EMU output must equal the recorded EMU tag and
its HW output the recorded silicon r1, run twice (r1 == r2).

Gate B: several recorded kernels packed onto one array, each core with its own
input; every core's output must equal that kernel's recorded single-core r1.

  python3 -m harness.gate a build
  python3 -m harness.gate a hw  r1        # serial, one NPU job at a time
  python3 -m harness.gate a emu e1
  python3 -m harness.gate a compare
  python3 -m harness.gate b build|hw|emu|compare   (same shape)

Everything lands under <main checkout>/build/experiments/harness/gate/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .build import build_package
from .run import REPO, read_run_record, run_package
from .topology import Buffer, Core, TileRef, Topology, to_dict

MAIN = REPO.parents[1] if REPO.parent.name == ".worktrees" else REPO
RECORDED = MAIN / "build" / "experiments" / "collisions"
GATE = MAIN / "build" / "experiments" / "harness" / "gate"
PROBES = ("probe", "probe12")
RECORDED_EMU_TAG = "m4"          # last full sweep, the current model version
RECORDED_HW_TAG = "r1"


@dataclass(frozen=True)
class Kernel:
    probe: str
    name: str
    in_size: int
    out_size: int

    @property
    def key(self) -> str:
        return f"{self.probe}_{self.name}"

    @property
    def source(self) -> Path:
        return RECORDED / self.probe / f"{self.name}.s"

    @property
    def input(self) -> bytes:
        return (RECORDED / self.probe / "input.bin").read_bytes()

    def recorded(self, tag: str) -> bytes:
        return (RECORDED / self.probe / tag / f"{self.name}.bin").read_bytes()


def kernels() -> list[Kernel]:
    out = []
    for probe in PROBES:
        m = json.loads((RECORDED / probe / "manifest.json").read_text())
        out += [Kernel(probe, k["name"], k["in_size"], k["out_size"]) for k in m["kernels"]]
    return out


def kernel_core(k: Kernel, name: str, col: int, row: int) -> Core:
    """`test_kernel(in, out)` of a recorded probe kernel as a core."""
    return Core(name, TileRef(col, row), k.source, args=(
        Buffer("in", "in", k.in_size, data=k.input), Buffer("out", "out", k.out_size)))


def single_topology(k: Kernel) -> Topology:
    return Topology(f"gateA_{k.key}", (kernel_core(k, k.name, 0, 2),), device="auto")


# -- gate B shapes -----------------------------------------------------------

def _pick(ks: list[Kernel], n: int, start: int = 0) -> list[Kernel]:
    return [ks[(start + i) % len(ks)] for i in range(n)]


def gate_b_topologies() -> list[tuple[Topology, dict[str, Kernel]]]:
    """(topology, core name -> recorded kernel) for each multi-core shape."""
    ks = kernels()
    shapes: list[tuple[str, list[tuple[int, int]], int]] = [
        ("cols4_direct", [(c, 2) for c in range(4)], 0),
        ("col0_stack4", [(0, r) for r in range(2, 6)], 4),
        ("full16", [(c, r) for c in range(4) for r in range(2, 6)], 8),
        ("cols4x2_reserve1", [(c, r) for c in range(4) for r in (2, 4)], 12),
        # same array, different kernel-to-tile assignment: position independence
        ("full16_shifted", [(c, r) for c in range(4) for r in range(2, 6)], 21),
    ]
    out = []
    for name, tiles, start in shapes:
        picked = _pick(ks, len(tiles), start)
        cores, mapping = [], {}
        for k, (c, r) in zip(picked, tiles):
            cn = f"c{c}_{r}_{k.key}"
            cores.append(kernel_core(k, cn, c, r))
            mapping[cn] = k
        out.append((Topology(f"gateB_{name}", tuple(cores), device="auto",
                             reserve_shim_s2mm=1 if "reserve" in name else 0), mapping))
    return out


# -- building / running ------------------------------------------------------

def units(gate: str) -> list[tuple[str, Topology, dict[str, Kernel]]]:
    if gate == "a":
        return [(single_topology(k).name, single_topology(k), {k.name: k}) for k in kernels()]
    return [(t.name, t, m) for t, m in gate_b_topologies()]


def pkg_dir(gate: str, name: str) -> Path:
    return GATE / gate.upper() / name


def cmd_build(gate: str, only: list[str]) -> int:
    rc = 0
    for name, topo, _ in units(gate):
        if only and name not in only:
            continue
        t0 = time.monotonic()
        try:
            build_package(topo, pkg_dir(gate, name))
            print(f"{name}: built {time.monotonic() - t0:.1f}s", flush=True)
        except Exception as e:      # report and continue: one bad shape must not hide the rest
            print(f"{name}: BUILD FAIL {e}", flush=True)
            rc = 1
    return rc


def cmd_run(gate: str, backend: str, tag: str, only: list[str], jobs: int = 1) -> int:
    chosen = [n for n, _, _ in units(gate) if not only or n in only]

    def one(name: str) -> tuple[str, bool, str]:
        rec = run_package(pkg_dir(gate, name), backend, tag, rebuild_ffi=False)
        wall = sum(p.wall_seconds for p in rec.phases)
        return name, rec.ok, (f"{name}: {'OK' if rec.ok else 'FAIL'} {wall:.2f}s "
                              f"rc={[p.returncode for p in rec.phases]}")

    rc = 0
    if backend == "hw" or jobs <= 1:        # silicon is one job at a time
        for name in chosen:
            _, ok, line = one(name)
            print(line, flush=True)
            if not ok:
                rc = 1
                if backend == "hw":
                    print("stopping at the first hardware failure", file=sys.stderr)
                    return rc
        return rc
    with cf.ThreadPoolExecutor(jobs) as ex:
        for _, ok, line in ex.map(one, chosen):
            print(line, flush=True)
            rc |= 0 if ok else 1
    return rc


# -- comparison --------------------------------------------------------------

def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _first_diff(a: bytes, b: bytes) -> str:
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    i = next((j for j in range(len(a)) if a[j] != b[j]), None)
    n = sum(x != y for x, y in zip(a, b))
    return "identical" if i is None else f"{n} bytes differ, first at {i}"


def compare(gate: str, hw_tags=("r1", "r2"), emu_tag="e1") -> list[dict]:
    """One row per (topology, core): EMU vs recorded EMU, HW r1 vs recorded r1, r1 == r2."""
    rows = []
    for name, topo, mapping in units(gate):
        pdir = pkg_dir(gate, name)
        for core in topo.cores:
            k = mapping[core.name]
            fname = f"{core.name}.out.bin"
            hw1, hw2 = (_read(pdir / t / fname) for t in hw_tags)
            emu = _read(pdir / emu_tag / fname)
            rec_hw, rec_emu = k.recorded(RECORDED_HW_TAG), k.recorded(RECORDED_EMU_TAG)
            rows.append({
                "topology": name, "core": core.name, "kernel": k.key,
                "emu_vs_recorded_emu": "missing" if emu is None else _first_diff(emu, rec_emu),
                "emu_vs_recorded_r1": "missing" if emu is None else _first_diff(emu, rec_hw),
                "hw_r1_vs_recorded_r1": "missing" if hw1 is None else _first_diff(hw1, rec_hw),
                "hw_r1_vs_hw_r2": "missing" if hw1 is None or hw2 is None else _first_diff(hw1, hw2),
            })
    return rows


def table(rows: list[dict]) -> str:
    cols = ["topology", "core", "kernel", "emu_vs_recorded_emu", "emu_vs_recorded_r1",
            "hw_r1_vs_recorded_r1", "hw_r1_vs_hw_r2"]
    lines = ["\t".join(cols)] + ["\t".join(r[c] for c in cols) for r in rows]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["a", "b"])
    ap.add_argument("cmd", choices=["build", "hw", "emu", "compare"])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("-j", type=int, default=1, help="parallel EMU runs")
    ap.add_argument("--only", nargs="*", default=[], help="topology names")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return cmd_build(args.gate, args.only)
    if args.cmd in ("hw", "emu"):
        if not args.tag:
            ap.error("tag required")
        return cmd_run(args.gate, args.cmd, args.tag, args.only, args.j)
    rows = compare(args.gate)
    text = table(rows)
    (GATE / f"results_{args.gate}.tsv").write_text(text)
    sys.stdout.write(text)
    bad = [r for r in rows if any(r[c] not in ("identical",) for c in
           ("emu_vs_recorded_emu", "hw_r1_vs_recorded_r1", "hw_r1_vs_hw_r2"))]
    print(f"{len(rows) - len(bad)}/{len(rows)} rows fully identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
