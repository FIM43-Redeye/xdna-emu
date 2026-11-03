"""Command line: build a topology, run the package on silicon or the emulator.

  python3 -m tools.harness build <topology.json> <pkg_dir>
  python3 -m tools.harness hw    <pkg_dir> <tag>
  python3 -m tools.harness emu   <pkg_dir> <tag> [--release]

  python3 -m harness suite build   <suite_dir> [--only CASE ...] [-j N]
  python3 -m harness suite hw      <suite_dir> <tag> [--only ...] [--timeout S] [--repeat N]
  python3 -m harness suite repeats <suite_dir> <tag> [--only ...]
  python3 -m harness suite emu     <suite_dir> <tag> [--only ...] [-j N] [--release] [--no-rebuild]
  python3 -m harness suite compare <suite_dir> <tag_a> <tag_b> [--only ...]

(or `PYTHONPATH=tools python3 -m harness ...`). A suite (suite.py) is a directory of cases;
results land in <suite_dir>/runs/<tag>/<case>/. `suite compare` exits 1 on any mismatch;
`suite hw --repeat N` runs every case N times (repeats 2..N in <case>/repeat_<k>/) and exits 1 if
any case's repeats disagree; `suite repeats` lists the differing bytes of such a run. Never run two hardware jobs at
once. Outputs land in <pkg_dir>/<tag>/<core>.<buffer>.bin plus run.json.
"""
from __future__ import annotations

import argparse
import sys

from .build import BuildError, build_package
from . import suite as S
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
    st = sub.add_parser("suite", help="build, run and compare a suite directory")
    ssub = st.add_subparsers(dest="suite_cmd", required=True)
    for name in ("build", "hw", "emu", "compare", "repeats"):
        p = ssub.add_parser(name)
        p.add_argument("suite_dir")
        if name == "compare":
            p.add_argument("tag_a")
            p.add_argument("tag_b")
        elif name != "build":
            p.add_argument("tag")
        p.add_argument("--only", nargs="+", metavar="CASE", help="restrict to these cases")
        if name in ("build", "emu"):
            p.add_argument("-j", type=int, default=None, help="parallel jobs (default: all cores)")
        if name in ("hw", "emu"):
            p.add_argument("--timeout", type=float, default=None, help="seconds per phase")
        if name == "hw":
            p.add_argument("--repeat", type=int, default=1, metavar="N",
                           help="run every case N times and report whether all repeats agree")
        if name == "emu":
            p.add_argument("--release", action="store_true", help="use the release FFI .so")
            p.add_argument("--no-rebuild", action="store_true", help="skip cargo build -p xdna-emu-ffi")
    args = ap.parse_args(argv)
    if args.cmd == "suite":
        return suite_main(args)
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


def suite_main(args) -> int:
    try:
        suite = S.load_suite(args.suite_dir)
        if args.suite_cmd == "compare":
            res = S.compare_runs(suite, args.tag_a, args.tag_b, only=args.only)
            print(S.format_comparison(res, args.tag_a, args.tag_b))
            return 0 if all(r.identical for r in res) else 1
        if args.suite_cmd == "build":
            out = S.build_suite(suite, only=args.only, jobs=args.j)
            print(S.format_outcomes(out, "build"))
        elif args.suite_cmd == "repeats":
            rows = S.repeat_report(suite, args.tag, only=args.only)
            print(S.format_repeat_report(rows, args.tag))
            return 0 if all(r.agree for r in rows) else 1
        elif args.suite_cmd == "hw":
            out = S.run_hw(suite, args.tag, only=args.only, timeout=args.timeout, repeat=args.repeat)
            print(S.format_outcomes(out, f"hw {args.tag}"))
            if any(o.agree is False for o in out):
                return 1
        else:
            out = S.run_emu(suite, args.tag, only=args.only, jobs=args.j,
                            runtime="release" if args.release else None,
                            rebuild_ffi=not args.no_rebuild, timeout=args.timeout)
            print(S.format_outcomes(out, f"emu {args.tag}"))
    except (S.SuiteError, BuildError, RunError, TopologyFormatError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0 if all(o.good for o in out) else 1


if __name__ == "__main__":
    sys.exit(main())
