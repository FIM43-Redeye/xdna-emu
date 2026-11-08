"""Suites: a directory of cases, built and run as one unit.

Layout (generators write the first part, the runner everything else)

    <suite>/suite.json                       name, generator, git rev, case order, meta
    <suite>/cases/<case>/topology.json       what to build (sources relative to the case dir)
    <suite>/cases/<case>/<kernel sources>
    <suite>/cases/<case>/case.json           free-form metadata owned by the case's analyzer
    <suite>/pkgs/<case>/                     built package (build.py), reused while inputs hash the same
    <suite>/host/                            the shared test_host
    <suite>/runs/<tag>/summary.json          run-level summary
    <suite>/runs/<tag>/<case>/<core>.<buffer>.bin, run.json, phase logs

Tags are the history: a tag is never overwritten, so `r1`, `r2`, `e1`... sit side by
side and `compare` diffs any two of them. Generators never call llvm-mc, aiecc or
XRT; this module owns those mechanics once.

Hardware runs are serial (the NPU is one resource) and the first timeout stops the
whole run: a wedged NPU poisons every later result, so the remaining cases are
recorded as skipped, with the reason, instead of being run. Any other failure is
recorded and the run continues. EMU runs rebuild the FFI `.so` once, then go in
parallel.

Both run kinds scrub by default (scrub.py): on silicon the scrub package runs on the case's
tiles before each phase, in the emulator every core starts from the same values, so a case
that reads a register it never wrote gives the same output on both and in any order.
`scrub=False` (`--no-scrub`) turns it off, for persistence experiments. The run summary and
every run record name the pattern (`scrub`); `compare` warns when two tags differ in it.

`suite.json` `meta` is the analyzer's suite-level data (record sizes, catalogues...).
`hw_checks` names pre-flight checks that refuse a package on silicon (see `HW_CHECKS`).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .emit import kernel_object
from .build import BuildError, _peano_tool, build_package
from .run import (RunError, build_ffi, build_test_host, git_rev, output_file_name,
                  read_run_record, run_package)
from .scrub import Scrub, ScrubKernel, case_tiles, load_scrub_kernel, scrub_key, write_scrub_case
from .topology import Buffer, Core, TileRef, Topology, load_topology, to_dict

SUITE_FORMAT = 1
DEFAULT_CORE, DEFAULT_BUFFER = "core", "out"


class SuiteError(RuntimeError):
    """A suite directory is malformed, or a run could not be set up."""


# -- layout ------------------------------------------------------------------

def case_dir(suite_dir: Path, case: str) -> Path:
    return Path(suite_dir) / "cases" / case


def pkg_dir(suite_dir: Path, case: str) -> Path:
    return Path(suite_dir) / "pkgs" / case


def runs_dir(suite_dir: Path, tag: str) -> Path:
    return Path(suite_dir) / "runs" / tag


def output_path(suite_dir: Path, tag: str, case: str,
                core: str = DEFAULT_CORE, buffer: str = DEFAULT_BUFFER) -> Path:
    """Where `run` leaves one output buffer of one case."""
    return runs_dir(suite_dir, tag) / case / output_file_name(core, buffer)


# -- writing (generators) ----------------------------------------------------

def write_case(suite_dir: str | Path, name: str, topology: Topology, *,
               files: Mapping[str, str | bytes], meta: dict[str, Any] | None = None) -> None:
    """Write one case: the topology, the files its sources name, and `case.json`.

    Every core source must be a relative path naming a key of `files`, so the case
    directory is self-contained.
    """
    for c in topology.cores:
        if Path(c.source).is_absolute() or str(c.source) not in files:
            raise SuiteError(f"case {name}: core {c.name!r} source {c.source} is not a "
                             f"relative path among the case's files")
    d = case_dir(Path(suite_dir), name)
    d.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (d / fname).write_bytes(content if isinstance(content, bytes) else content.encode())
    (d / "topology.json").write_text(json.dumps(to_dict(topology), indent=1) + "\n")
    (d / "case.json").write_text(json.dumps(meta or {}, indent=1) + "\n")


def write_single_core_case(suite_dir: str | Path, name: str, asm: str, *, input_bytes: bytes,
                           out_bytes: int, meta: dict[str, Any] | None = None,
                           tile: tuple[int, int] = (0, 2)) -> None:
    """The common probe case: one core running `asm` on one input, one output buffer.

    The kernel ABI is `(in, out)`; the result is `core.out.bin`, the default
    `Suite.output` looks for.
    """
    topo = Topology(name, (Core(DEFAULT_CORE, TileRef(*tile), Path("kernel.s"), args=(
        Buffer("in", "in", len(input_bytes), data=input_bytes),
        Buffer(DEFAULT_BUFFER, "out", out_bytes))),))
    write_case(suite_dir, name, topo, files={"kernel.s": asm}, meta=meta)


def write_suite_json(suite_dir: str | Path, name: str, generator: str, cases: Iterable[str], *,
                     meta: dict[str, Any] | None = None, hw_checks: Iterable[str] = (),
                     generator_rev: str | None = None) -> None:
    unknown = set(hw_checks) - set(HW_CHECKS)
    if unknown:
        raise SuiteError(f"unknown hw_checks {sorted(unknown)} (known: {sorted(HW_CHECKS)})")
    doc = {"format": SUITE_FORMAT, "name": name, "generator": generator,
           "git_rev": generator_rev if generator_rev is not None else git_rev(),
           "cases": list(cases), "hw_checks": list(hw_checks), "meta": meta or {}}
    (Path(suite_dir) / "suite.json").write_text(json.dumps(doc, indent=1) + "\n")


# -- reading (runner and analyzers) -----------------------------------------

@dataclass(frozen=True)
class Suite:
    path: Path
    name: str
    generator: str
    git_rev: str
    cases: tuple[str, ...]
    hw_checks: tuple[str, ...]
    meta: dict[str, Any]

    def case_meta(self, case: str) -> dict[str, Any]:
        p = case_dir(self.path, case) / "case.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def topology(self, case: str) -> Topology:
        return load_topology(case_dir(self.path, case) / "topology.json")

    def case_entries(self) -> list[dict[str, Any]]:
        """Each case's `case.json` with its `name`, in suite order (what a probe analyzer iterates)."""
        return [{"name": c, **self.case_meta(c)} for c in self.cases]

    def core_object(self, case: str, core: Core) -> Path:
        """The core's built object, once `suite build` has run."""
        return pkg_dir(self.path, case) / "phase_0" / kernel_object(core)

    def output(self, tag: str, case: str) -> Path:
        """The case's result file: `case.json` `result` ({core, buffer}) or core.out."""
        r = self.case_meta(case).get("result", {})
        return output_path(self.path, tag, case, r.get("core", DEFAULT_CORE),
                           r.get("buffer", DEFAULT_BUFFER))


def load_suite(suite_dir: str | Path) -> Suite:
    d = Path(suite_dir).resolve()
    p = d / "suite.json"
    if not p.is_file():
        raise SuiteError(f"{d} is not a suite (no suite.json)")
    doc = json.loads(p.read_text())
    if doc.get("format") != SUITE_FORMAT:
        raise SuiteError(f"{p}: format {doc.get('format')!r}, expected {SUITE_FORMAT}")
    cases = tuple(doc["cases"])
    if len(set(cases)) != len(cases):
        raise SuiteError(f"{p}: duplicate case names")
    for c in cases:
        if not (case_dir(d, c) / "topology.json").is_file():
            raise SuiteError(f"{p}: case {c!r} has no cases/{c}/topology.json")
    return Suite(d, doc["name"], doc["generator"], doc.get("git_rev", ""), cases,
                 tuple(doc.get("hw_checks", ())), doc.get("meta", {}))


def select(suite: Suite, only: Iterable[str] | None) -> tuple[str, ...]:
    if not only:
        return suite.cases
    only = list(only)
    missing = [c for c in only if c not in suite.cases]
    if missing:
        raise SuiteError(f"no such case(s): {', '.join(missing)}")
    return tuple(c for c in suite.cases if c in only)


# -- outcomes ----------------------------------------------------------------

@dataclass(frozen=True)
class CaseOutcome:
    case: str
    status: str            # built | reused | ok | failed | timeout | refused | skipped
    detail: str = ""
    wall_seconds: float = 0.0
    agree: bool | None = None   # hw --repeat: did every repeat produce the same bytes (None: not repeated)

    @property
    def good(self) -> bool:
        return self.status in ("built", "reused", "ok")


# -- build -------------------------------------------------------------------

def build_suite(suite: Suite, *, only: Iterable[str] | None = None,
                jobs: int | None = None) -> list[CaseOutcome]:
    """Build every selected case's package in parallel, reusing unchanged ones."""
    cases = select(suite, only)

    def one(case: str) -> CaseOutcome:
        t0 = time.monotonic()
        try:
            pkg = build_package(suite.topology(case), pkg_dir(suite.path, case), reuse=True)
        except (BuildError, SuiteError, OSError, ValueError) as e:
            return CaseOutcome(case, "failed", str(e), time.monotonic() - t0)
        return CaseOutcome(case, "reused" if pkg.reused else "built", "", time.monotonic() - t0)

    with cf.ThreadPoolExecutor(jobs or os.cpu_count() or 1) as ex:
        return list(ex.map(one, cases))


# -- hardware pre-flight checks -----------------------------------------------

_LABEL = re.compile(r"^([0-9a-f]+) <(\.L\w+)>:", re.M)


def unaligned_labels(objdump_text: str) -> list[str]:
    """`.L` labels of a disassembly that do not sit on a 16-byte fetch line.

    Silicon fetches a taken branch's target from its 16-byte line start, so an
    unaligned loop label re-runs the code before it (an endless loop, then a TDR per
    kernel). The toolchain pads every jump target; hand-written asm must `.p2align 4`.
    """
    return [f"{name}@{int(addr, 16):#x}" for addr, name in _LABEL.findall(objdump_text)
            if int(addr, 16) % 16]


def check_aligned_labels(package: Path) -> list[str]:
    """Problems with the linked ELFs of a package (empty when it is safe to run on silicon)."""
    problems: list[str] = []
    elfs = sorted(package.glob("phase_*/aie.mlir.prj/elfs_main_core_*/*.elf"))
    if not elfs:
        return [f"no core ELFs found under {package}"]
    for elf in elfs:
        dump = subprocess.run([_peano_tool("llvm-objdump"), "-d", str(elf)],
                              capture_output=True, text=True)
        if dump.returncode:
            problems.append(f"{elf.name}: llvm-objdump failed")
        problems += [f"{elf.parent.name}: label {x} not 16-byte aligned"
                     for x in unaligned_labels(dump.stdout)]
    return problems


HW_CHECKS: dict[str, Callable[[Path], list[str]]] = {"aligned_labels": check_aligned_labels}


# -- run ---------------------------------------------------------------------

def _prepare_tag(suite: Suite, tag: str) -> Path:
    out = runs_dir(suite.path, tag)
    if (out / "summary.json").exists():
        raise SuiteError(f"{out} already holds a run; tags are never overwritten, pick another")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_summary(out: Path, suite: Suite, backend: str, tag: str, outcomes: list[CaseOutcome],
                   aborted: dict[str, str] | None, extra: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    doc = {"suite": suite.name, "tag": tag, "backend": backend, "git_rev": git_rev(),
           "counts": counts, "aborted": aborted, **extra,
           "cases": {o.case: {"status": o.status, "detail": o.detail,
                              "wall_seconds": round(o.wall_seconds, 3),
                              **({} if o.agree is None else {"agree": o.agree})}
                     for o in outcomes}}
    (out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")


def _run_case(suite: Suite, case: str, backend: str, tag: str, host: Path,
              out_dir: Path | None = None, **kw: Any) -> CaseOutcome:
    t0 = time.monotonic()
    pkg = pkg_dir(suite.path, case)
    if not (pkg / "manifest.json").is_file():
        return CaseOutcome(case, "failed", "not built (run `suite build`)")
    try:
        rec = run_package(pkg, backend, tag, out_dir=out_dir or runs_dir(suite.path, tag) / case,
                          host=host, **kw)
    except RunError as e:
        return CaseOutcome(case, "failed", str(e), time.monotonic() - t0)
    wall = time.monotonic() - t0
    if rec.timed_out:
        return CaseOutcome(case, "timeout", "phase timed out", wall)
    if not rec.ok:
        rcs = ",".join(str(p.returncode) for p in rec.phases) or "none"
        return CaseOutcome(case, "failed", f"phase return code(s) {rcs}", wall)
    return CaseOutcome(case, "ok", "", wall)


class _HwScrubs:
    """The suite's scrub packages, one per device and tile set, built on first use."""

    def __init__(self, suite: Suite, kernel: ScrubKernel):
        self.suite, self.kernel, self.built = suite, kernel, {}

    def for_case(self, case: str) -> Scrub:
        topo = self.suite.topology(case)
        tiles = case_tiles(topo)
        key = scrub_key(topo.device, tiles)
        if key not in self.built:
            base = self.suite.path / "scrub" / key
            t = write_scrub_case(self.kernel, base / "case", topo.device, tiles)
            pkg = build_package(t, base / "pkg", reuse=True)
            self.built[key] = Path(pkg.out_dir)
        return Scrub(self.kernel.name, self.kernel.out_marker, self.built[key])


def run_hw(suite: Suite, tag: str, *, only: Iterable[str] | None = None,
           timeout: float | None = None, repeat: int = 1, scrub: bool = True) -> list[CaseOutcome]:
    """Run the selected cases on silicon, one at a time; the first timeout stops the run.

    `repeat` N > 1 runs each case N times back to back: repeat 1 lands in the plain
    layout (`runs/<tag>/<case>/`, so every reader works unchanged), repeats 2..N in
    `runs/<tag>/<case>/repeat_<k>/`. A case's `agree` says whether all N produced the
    same bytes; `repeat_report` names the differing bytes.
    """
    if repeat < 1:
        raise SuiteError(f"repeat must be >= 1, got {repeat}")
    cases = select(suite, only)
    out = _prepare_tag(suite, tag)
    host = build_test_host(_host_dir(suite))
    kernel = load_scrub_kernel() if scrub else None
    scrubs = _HwScrubs(suite, kernel) if kernel else None
    outcomes: list[CaseOutcome] = []
    aborted = None
    try:
        for i, case in enumerate(cases):
            problems = [p for chk in suite.hw_checks for p in HW_CHECKS[chk](pkg_dir(suite.path, case))]
            if problems:
                outcomes.append(CaseOutcome(case, "refused", "; ".join(problems)))
                continue
            sc = None
            if scrubs is not None:
                try:
                    sc = scrubs.for_case(case)
                except (BuildError, OSError, ValueError) as e:
                    outcomes.append(CaseOutcome(case, "failed", f"scrub package: {e}"))
                    continue
            o = _run_case(suite, case, "hw", tag, host, timeout=timeout, scrub=sc)
            if o.good and repeat > 1:
                o = _repeat_case(suite, case, tag, host, timeout, repeat, o, sc)
            outcomes.append(o)
            if o.status == "timeout":
                aborted = {"case": case, "reason": "hardware timeout; the NPU may be wedged, "
                           "later results would be unreliable"}
                outcomes += [CaseOutcome(c, "skipped", f"run aborted at {case}") for c in cases[i + 1:]]
                break
    finally:
        _write_summary(out, suite, "hw", tag, outcomes, aborted,
                       {"repeat": repeat, "scrub": kernel.name if kernel else None})
    return outcomes


def repeat_dir(suite_dir: Path, tag: str, case: str, k: int) -> Path:
    """Where repeat `k` (>= 2) of a case lands; repeat 1 is the case's plain run directory."""
    return runs_dir(suite_dir, tag) / case / f"repeat_{k}"


def _repeat_case(suite: Suite, case: str, tag: str, host: Path, timeout: float | None,
                 repeat: int, first: CaseOutcome, scrub: Scrub | None = None) -> CaseOutcome:
    wall, agree = first.wall_seconds, True
    base = runs_dir(suite.path, tag) / case
    for k in range(2, repeat + 1):
        o = _run_case(suite, case, "hw", tag, host, out_dir=repeat_dir(suite.path, tag, case, k),
                      timeout=timeout, scrub=scrub)
        wall += o.wall_seconds
        if not o.good:
            return CaseOutcome(case, o.status, f"repeat {k}: {o.detail}", wall)
        agree = agree and not _file_diffs(base, repeat_dir(suite.path, tag, case, k), k)
    return CaseOutcome(case, "ok", "", wall, agree=agree)


def run_emu(suite: Suite, tag: str, *, only: Iterable[str] | None = None, jobs: int | None = None,
            runtime: str | None = None, rebuild_ffi: bool = True,
            timeout: float | None = None, scrub: bool = True) -> list[CaseOutcome]:
    """Run the selected cases on the emulator in parallel (FFI `.so` rebuilt once first)."""
    kernel = load_scrub_kernel() if scrub else None
    sc = Scrub(kernel.name, kernel.out_marker) if kernel else None
    cases = select(suite, only)
    out = _prepare_tag(suite, tag)
    runtime = runtime or os.environ.get("XDNA_EMU_RUNTIME", "debug")
    so_mtime = build_ffi(runtime) if rebuild_ffi else None
    host = build_test_host(_host_dir(suite))
    outcomes: list[CaseOutcome] = []
    try:
        with cf.ThreadPoolExecutor(jobs or os.cpu_count() or 1) as ex:
            outcomes = list(ex.map(
                lambda c: _run_case(suite, c, "emu", tag, host, runtime=runtime,
                                    rebuild_ffi=False, timeout=timeout, scrub=sc), cases))
    finally:
        _write_summary(out, suite, "emu", tag, outcomes, None,
                       {"runtime": runtime, "ffi_rebuilt": rebuild_ffi, "ffi_so_mtime": so_mtime,
                        "scrub": kernel.name if kernel else None})
    return outcomes


def _host_dir(suite: Suite) -> Path:
    d = suite.path / "host"
    d.mkdir(exist_ok=True)
    return d


# -- compare -----------------------------------------------------------------

@dataclass(frozen=True)
class CaseComparison:
    case: str
    status: str            # identical | differs | unusable
    detail: str = ""

    @property
    def identical(self) -> bool:
        return self.status == "identical"


def _first_diff(a: bytes, b: bytes) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare_runs(suite: Suite, tag_a: str, tag_b: str, *,
                 only: Iterable[str] | None = None) -> list[CaseComparison]:
    """Per case and per output buffer, byte-exact. A case that did not run cleanly in
    either tag is `unusable`, never silently identical."""
    for t in (tag_a, tag_b):
        if not runs_dir(suite.path, t).is_dir():
            raise SuiteError(f"no run tagged {t!r} in {suite.path}")
    results: list[CaseComparison] = []
    for case in select(suite, only):
        recs = {}
        for t in (tag_a, tag_b):
            d = runs_dir(suite.path, t) / case
            try:
                recs[t] = read_run_record(d)
            except (OSError, json.JSONDecodeError, TypeError):
                recs[t] = None
        bad = [f"{t}: " + ("no run record" if r is None else "run failed")
               for t, r in recs.items() if r is None or not r.ok]
        if bad:
            results.append(CaseComparison(case, "unusable", "; ".join(bad)))
            continue
        names_a, names_b = set(recs[tag_a].outputs), set(recs[tag_b].outputs)
        problems = [f"{n} only in {tag_a if n in names_a else tag_b}" for n in sorted(names_a ^ names_b)]
        for n in sorted(names_a & names_b):
            a = (runs_dir(suite.path, tag_a) / case / n).read_bytes()
            b = (runs_dir(suite.path, tag_b) / case / n).read_bytes()
            if a != b:
                nd = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
                problems.append(f"{n}: {nd} byte(s) differ, first at offset {_first_diff(a, b)}"
                                + (f" (sizes {len(a)} vs {len(b)})" if len(a) != len(b) else ""))
        results.append(CaseComparison(case, "differs" if problems else "identical", "; ".join(problems)))
    return results


def run_scrub(suite: Suite, tag: str) -> str | None:
    """The scrub pattern a run started from (None: no scrub, or a run older than the scrub)."""
    p = runs_dir(suite.path, tag) / "summary.json"
    return json.loads(p.read_text()).get("scrub") if p.is_file() else None


def scrub_warning(suite: Suite, tag_a: str, tag_b: str) -> str | None:
    """A line to print when two runs did not start from the same core state."""
    a, b = run_scrub(suite, tag_a), run_scrub(suite, tag_b)
    if a == b:
        return None
    return (f"warning: {tag_a} scrub={a or 'none'}, {tag_b} scrub={b or 'none'}: lanes a case "
            f"never writes are not comparable")


# -- repeat report -------------------------------------------------------------

@dataclass(frozen=True)
class ByteDiff:
    repeat: int            # the repeat that differs from repeat 1
    file: str
    offset: int
    ref: int               # repeat 1's byte
    got: int


@dataclass(frozen=True)
class CaseRepeats:
    case: str
    repeats: int
    diffs: tuple[ByteDiff, ...]
    first_run_only: bool   # repeats 2..N agree with each other and only repeat 1 stands apart

    @property
    def agree(self) -> bool:
        return not self.diffs


def _output_names(case_run: Path) -> tuple[str, ...]:
    return read_run_record(case_run).outputs


def _file_diffs(ref_dir: Path, other_dir: Path, k: int) -> list[ByteDiff]:
    """Byte differences of repeat `k` against repeat 1, over repeat 1's output files."""
    diffs: list[ByteDiff] = []
    for name in _output_names(ref_dir):
        a = (ref_dir / name).read_bytes()
        p = other_dir / name
        b = p.read_bytes() if p.is_file() else b""
        diffs += [ByteDiff(k, name, i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(a) != len(b):
            diffs.append(ByteDiff(k, name, min(len(a), len(b)), -1, -1))   # size mismatch marker
    return diffs


def repeat_report(suite: Suite, tag: str, *, only: Iterable[str] | None = None) -> list[CaseRepeats]:
    """Per case of a `hw --repeat` run: do all repeats agree, and if not, which bytes differ."""
    if not runs_dir(suite.path, tag).is_dir():
        raise SuiteError(f"no run tagged {tag!r} in {suite.path}")
    rows: list[CaseRepeats] = []
    for case in select(suite, only):
        base = runs_dir(suite.path, tag) / case
        ks = sorted(int(d.name.split("_")[1]) for d in base.glob("repeat_*") if d.is_dir())
        if not (base / "run.json").is_file():
            continue
        diffs: list[ByteDiff] = []
        for k in ks:
            diffs += _file_diffs(base, repeat_dir(suite.path, tag, case, k), k)
        later_agree = all(not _file_diffs(repeat_dir(suite.path, tag, case, ks[0]),
                                          repeat_dir(suite.path, tag, case, k), k) for k in ks[1:]) if ks else True
        differing = {d.repeat for d in diffs}
        rows.append(CaseRepeats(case, 1 + len(ks), tuple(diffs),
                                bool(diffs) and len(ks) >= 2 and differing == set(ks) and later_agree))
    return rows


# -- reports -----------------------------------------------------------------

def format_outcomes(outcomes: list[CaseOutcome], verb: str) -> str:
    lines = [f"{o.case}: {o.status}" + (f" ({o.detail})" if o.detail else "") for o in outcomes
             if not o.good or o.status == "failed"]
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    lines += [f"{o.case}: repeats DIFFER (see `suite repeats`)" for o in outcomes if o.agree is False]
    lines.append(f"{verb}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) +
                 f" of {len(outcomes)}")
    return "\n".join(lines)


def format_comparison(results: list[CaseComparison], tag_a: str, tag_b: str) -> str:
    lines = [f"{r.case}: {r.status.upper()} {r.detail}" for r in results if not r.identical]
    n_id = sum(r.identical for r in results)
    lines.append(f"compare {tag_a} vs {tag_b}: {n_id}/{len(results)} identical, "
                 f"{sum(r.status == 'differs' for r in results)} differ, "
                 f"{sum(r.status == 'unusable' for r in results)} unusable")
    return "\n".join(lines)


def format_repeat_report(rows: list[CaseRepeats], tag: str, *, max_bytes: int = 16) -> str:
    lines: list[str] = []
    for r in rows:
        if r.agree:
            continue
        pattern = ("first run only: repeats 2.." f"{r.repeats} agree with each other"
                   if r.first_run_only else "repeats disagree among themselves")
        lines.append(f"{r.case}: NONDETERMINISTIC, {len({d.repeat for d in r.diffs})} of "
                     f"{r.repeats - 1} repeat(s) differ from repeat 1 ({pattern})")
        for d in r.diffs[:max_bytes]:
            lines.append(f"    repeat {d.repeat} {d.file} 0x{d.offset:02x}: "
                         + (f"{d.ref:02x} -> {d.got:02x}" if d.ref >= 0 else "size differs"))
        if len(r.diffs) > max_bytes:
            lines.append(f"    ... {len(r.diffs) - max_bytes} more byte(s)")
    n_agree = sum(r.agree for r in rows)
    lines.append(f"repeat {tag}: {n_agree}/{len(rows)} cases agree across all repeats, "
                 f"{len(rows) - n_agree} nondeterministic")
    return "\n".join(lines)
