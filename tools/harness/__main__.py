"""Command line: build a topology, run the package on silicon or the emulator.

  python3 -m tools.harness build <topology.json> <pkg_dir>
  python3 -m tools.harness hw    <pkg_dir> <tag>
  python3 -m tools.harness emu   <pkg_dir> <tag> [--release]

(or `PYTHONPATH=tools python3 -m harness ...`). Never run two hardware jobs at
once. Outputs land in <pkg_dir>/<tag>/<core>.<buffer>.bin plus run.json.
"""
from __future__ import annotations

import argparse
import sys

from .build import BuildError, build_package
from .run import RunError, run_package
from .topology import TopologyFormatError, load_topology


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="validate, plan, emit and build a package")
    b.add_argument("topology")
    b.add_argument("pkg_dir")
    for name in ("hw", "emu"):
        p = sub.add_parser(name, help=f"run a built package on {'silicon' if name == 'hw' else 'the emulator'}")
        p.add_argument("pkg_dir")
        p.add_argument("tag")
        p.add_argument("--timeout", type=float, default=None, help="seconds per phase")
        if name == "emu":
            p.add_argument("--release", action="store_true", help="use the release FFI .so")
            p.add_argument("--no-rebuild", action="store_true", help="skip cargo build -p xdna-emu-ffi")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            pkg = build_package(load_topology(args.topology), args.pkg_dir)
            print(f"built {len(pkg.phase_dirs)} phase(s) in {pkg.out_dir}")
            return 0
        kw = {"timeout": args.timeout}
        if args.cmd == "emu":
            kw.update(runtime="release" if args.release else None, rebuild_ffi=not args.no_rebuild)
        rec = run_package(args.pkg_dir, args.cmd, args.tag, **kw)
    except (BuildError, RunError, TopologyFormatError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for p in rec.phases:
        print(f"phase {p.index}: rc={p.returncode} {p.wall_seconds:.2f}s")
    print(f"{len(rec.outputs)} output file(s) in {args.pkg_dir}/{args.tag}")
    return 0 if rec.ok else 1


if __name__ == "__main__":
    sys.exit(main())
