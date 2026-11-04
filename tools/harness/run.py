"""Runner: execute a built package on silicon or the emulator and unpack results.

`run_package(pkg_dir, backend, tag)` takes the directory `build_package` wrote
and, for each phase in order, runs the shared instr-test-gen `test_host`
(unchanged, three BOs) with the phase's packed input, then splits the phase's
`out` BO into one file per core output:

    <pkg_dir>/<tag>/<core>.<buffer>.bin      one per output buffer
    <pkg_dir>/<tag>/phase_<i>.in.bin|.out.bin|.log
    <pkg_dir>/<tag>/run.json                 the run record

Input definition (recorded per buffer in the package manifest and run record)
-----------------------------------------------------------------------------
- `Buffer.data` given: those bytes verbatim (length must equal `size_bytes`).
- `Buffer.data` None: `size_bytes` bytes of the test_host LCG (a=1103515245,
  c=12345, m=2^31, byte = state >> 16 & 0xFF) seeded with
  `crc32("<topology.seed>:<core>:<buffer>")`, so every buffer has its own
  stream and the same topology always gets the same bytes.
Inputs are placed at the planner's `host_offset`; a phase with no inputs (or
outputs) still gets one zero word, because the shared host needs a non-empty BO.

Backends
--------
- `hw`: `XDNA_EMU` / `XDNA_EMU_RUNTIME` removed from the environment. One job
  at a time is the caller's contract (the NPU is a single resource); phases run
  serially and the first failing phase stops the run.
- `emu`: `XDNA_EMU=1`, `XDNA_EMU_RUNTIME` (default debug), `XDNA_EMU_DIR` = this
  checkout. The FFI `.so` is rebuilt first (a stale `.so` causes phantom bugs)
  and its mtime goes in the run record.

Scrub (scrub.py)
----------------
With a `Scrub`, a hardware run launches the scrub package before EVERY phase (each phase is
its own launch, and silicon carries core registers from one launch into the next), checks its
marker outputs and logs it as `phase_<i>.scrub.log`; an emulator run sets
`XDNA_EMU_CORE_INIT` so every core starts from the same values. The record's `scrub` names
the pattern ("" = no scrub).
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .scrub import Scrub, marker_problems
from .topology import Buffer, Core, Topology, topology_from_dict

REPO = Path(__file__).resolve().parents[2]
XRT = "/opt/xilinx/xrt"
KERNEL_NAME = "MLIR_AIE"
MIN_BO_BYTES = 4
TIMEOUT_RC = -1                  # PhaseResult.returncode of a phase that hit its timeout
SCRUB_MARKER_RC = -2             # the scrub before the phase ran but left no marker


class RunError(RuntimeError):
    """A run could not be set up or a result could not be interpreted."""


# -- input definition --------------------------------------------------------

def prng_bytes(n: int, seed: int) -> bytes:
    """The test_host `fill_prng` stream."""
    out = bytearray(n)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = (state >> 16) & 0xFF
    return bytes(out)


def buffer_seed(topology: Topology, core: Core, buf: Buffer) -> int:
    return zlib.crc32(f"{topology.seed}:{core.name}:{buf.name}".encode())


def buffer_input(topology: Topology, core: Core, buf: Buffer) -> tuple[bytes, dict[str, Any]]:
    """The bytes of an input buffer and a description of where they came from."""
    if buf.data is not None:
        if len(buf.data) != buf.size_bytes:
            raise RunError(f"{core.name}.{buf.name}: data is {len(buf.data)} bytes, "
                           f"size_bytes is {buf.size_bytes}")
        return buf.data, {"kind": "data"}
    seed = buffer_seed(topology, core, buf)
    return prng_bytes(buf.size_bytes, seed), {"kind": "prng", "seed": seed}


def input_sources(topology: Topology) -> list[dict[str, Any]]:
    """Per input buffer: core, buffer, size and source (`data` or `prng` + seed)."""
    return [{"core": c.name, "buffer": b.name, "size_bytes": b.size_bytes,
             "source": buffer_input(topology, c, b)[1]}
            for c in topology.cores for b in c.args if b.direction == "in"]


# -- pack / unpack -----------------------------------------------------------
# Phases are the plain dicts stored in the manifest (`dataclasses.asdict` of
# plan.Phase), so a run needs nothing but the package directory.

def pack_phase_inputs(topology: Topology, phase: dict) -> bytes:
    by_name = {c.name: c for c in topology.cores}
    packed = bytearray(max(phase["host_in_bytes"], MIN_BO_BYTES))
    for r in phase["routes"]:
        if r["direction"] != "in":
            continue
        core = by_name[r["core"]]
        buf = next(b for b in core.args if b.name == r["buffer"])
        data, _ = buffer_input(topology, core, buf)
        packed[r["host_offset"]:r["host_offset"] + len(data)] = data
    return bytes(packed)


def unpack_phase_outputs(phase: dict, out: bytes) -> dict[tuple[str, str], bytes]:
    if len(out) < phase["host_out_bytes"]:
        raise RunError(f"phase {phase['index']}: short output, {len(out)} bytes "
                       f"< {phase['host_out_bytes']}")
    return {(r["core"], r["buffer"]): out[r["host_offset"]:r["host_offset"] + r["size_bytes"]]
            for r in phase["routes"] if r["direction"] == "out"}


def output_file_name(core: str, buffer: str) -> str:
    return f"{core}.{buffer}.bin"


# -- run record --------------------------------------------------------------

@dataclass(frozen=True)
class PhaseResult:
    index: int
    returncode: int          # -1 = timeout
    wall_seconds: float
    log: str


@dataclass(frozen=True)
class RunRecord:
    tag: str
    backend: str
    topology: str
    git_rev: str
    ffi_so_mtime: float | None       # emu only: mtime of the .so that ran
    phases: tuple[PhaseResult, ...]
    outputs: tuple[str, ...]
    inputs: tuple[dict, ...]
    package_hash: str = ""           # the build's `inputs_hash`: which package produced this
    scrub: str = ""                  # scrub pattern the run started from ("" = none)

    @property
    def ok(self) -> bool:
        return bool(self.phases) and all(p.returncode == 0 for p in self.phases)

    @property
    def timed_out(self) -> bool:
        return any(p.returncode == TIMEOUT_RC for p in self.phases)


def write_run_record(out_dir: Path, rec: RunRecord) -> None:
    (Path(out_dir) / "run.json").write_text(
        json.dumps(asdict(rec), indent=2, sort_keys=True) + "\n")


def read_run_record(out_dir: Path) -> RunRecord:
    d = json.loads((Path(out_dir) / "run.json").read_text())
    d["phases"] = tuple(PhaseResult(**p) for p in d["phases"])
    d["outputs"] = tuple(d["outputs"])
    d["inputs"] = tuple(d["inputs"])
    return RunRecord(**d)


# -- host and emulator plumbing ----------------------------------------------

def _instr_gen():
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("instr-test-gen")


def workspace() -> Path:
    """Directory holding mlir-aie/ and llvm-aie/: NPU_WORK_DIR, else the nearest ancestor."""
    env = os.environ.get("NPU_WORK_DIR")
    if env:
        return Path(env)
    for anc in Path(__file__).resolve().parents:
        if (anc / "mlir-aie").is_dir() and (anc / "llvm-aie").is_dir():
            return anc
    raise RunError("cannot find the workspace (set NPU_WORK_DIR)")


def build_test_host(pkg_dir: Path) -> Path:
    """Compile the shared test_host into `pkg_dir` (the shared host, built once per suite)."""
    mlir_aie = Path(os.environ.get("MLIR_AIE") or workspace() / "mlir-aie")
    test_lib = mlir_aie / "build" / "runtime_lib" / "x86_64" / "test_lib"
    cpp, exe = pkg_dir / "test_host.cpp", pkg_dir / "test_host"
    # Rewrite (and so rebuild) only on a change: runs on one suite share this host, and a
    # rebuild under a running suite makes its next launch fail (EACCES on a half-written exe).
    src = _instr_gen().generate_test_host_cpp()
    if not cpp.is_file() or cpp.read_text() != src:
        cpp.write_text(src)
    if not exe.exists() or exe.stat().st_mtime < cpp.stat().st_mtime:
        tmp = exe.with_name(f".test_host.{os.getpid()}")
        r = subprocess.run(
            ["clang++", str(cpp), "-o", str(tmp), "-std=c++17", "-Wall",
             "-I", f"{test_lib}/include", "-I", f"{XRT}/include",
             "-L", f"{test_lib}/lib", "-L", f"{XRT}/lib",
             "-ltest_utils", "-lxrt_coreutil", "-lrt", "-lstdc++"],
            capture_output=True, text=True)
        if r.returncode:
            tmp.unlink(missing_ok=True)
            raise RunError(f"building test_host failed:\n{r.stderr[-1500:]}")
        tmp.replace(exe)        # atomic: a concurrent launch sees the old exe or the new one
    return exe


def git_rev(repo: Path = REPO) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def ffi_so_path(runtime: str) -> Path:
    return REPO / "target" / runtime / "libxdna_emu.so"


def build_ffi(runtime: str = "debug") -> float:
    """Rebuild the FFI cdylib for `runtime` and return the .so mtime."""
    cmd = ["cargo", "build", "-p", "xdna-emu-ffi"] + (["--release"] if runtime == "release" else [])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        raise RunError(f"cargo build -p xdna-emu-ffi failed:\n{r.stderr[-1500:]}")
    return ffi_so_path(runtime).stat().st_mtime


def _env(backend: str, runtime: str, scrub: Scrub | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("XDNA_EMU_CORE_INIT", None)
    if backend == "emu":
        env.update(XDNA_EMU="1", XDNA_EMU_RUNTIME=runtime, XDNA_EMU_DIR=str(REPO))
        if scrub is not None:
            env["XDNA_EMU_CORE_INIT"] = scrub.name
    else:
        env.pop("XDNA_EMU", None)
        env.pop("XDNA_EMU_RUNTIME", None)
        env.pop("XDNA_EMU_DIR", None)
    return env


def run_package(pkg_dir: str | Path, backend: str, tag: str, *, runtime: str | None = None,
                timeout: float | None = None, rebuild_ffi: bool = True,
                out_dir: str | Path | None = None, host: str | Path | None = None,
                scrub: Scrub | None = None) -> RunRecord:
    """Run every phase of the package at `pkg_dir`.

    Outputs land in `out_dir`, default `<pkg_dir>/<tag>/`. `host` is a prebuilt
    test_host (a suite builds one for all its packages); default builds one inside
    `pkg_dir`. `scrub`: see the module docstring (hardware needs `scrub.pkg_dir`).
    """
    if backend not in ("hw", "emu"):
        raise RunError(f"backend must be 'hw' or 'emu', not {backend!r}")
    pkg_dir = Path(pkg_dir).resolve()
    manifest_path = pkg_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RunError(f"{pkg_dir} is not a built package (no manifest.json)")
    out_dir = Path(out_dir) if out_dir is not None else pkg_dir / tag
    if (out_dir / "run.json").exists():
        raise RunError(f"{out_dir} already holds a run; recorded evidence is never overwritten, "
                       f"pick another tag")
    manifest = json.loads(manifest_path.read_text())
    topology = topology_from_dict(manifest["topology"])
    phases = manifest["plan"]["phases"]
    runtime = runtime or os.environ.get("XDNA_EMU_RUNTIME", "debug")
    timeout = timeout if timeout is not None else (1800 if backend == "emu" else 60)

    so_mtime = build_ffi(runtime) if backend == "emu" and rebuild_ffi else (
        ffi_so_path(runtime).stat().st_mtime if backend == "emu" else None)
    host = Path(host) if host is not None else build_test_host(pkg_dir)
    if scrub is not None and backend == "hw" and scrub.pkg_dir is None:
        raise RunError("a hardware scrub needs the built scrub package")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _env(backend, runtime, scrub)

    results: list[PhaseResult] = []
    outputs: list[str] = []
    for ph in phases:
        i = ph["index"]
        pdir = pkg_dir / f"phase_{i}"
        in_bin, out_bin = out_dir / f"phase_{i}.in.bin", out_dir / f"phase_{i}.out.bin"
        in_bin.write_bytes(pack_phase_inputs(topology, ph))
        in_size = max(ph["host_in_bytes"], MIN_BO_BYTES)
        out_size = max(ph["host_out_bytes"], MIN_BO_BYTES)
        cmd = [str(host), "-x", str(pdir / "aie.xclbin"), "-k", KERNEL_NAME,
               "-i", str(pdir / "insts.bin"), "--in-size", str(in_size),
               "--out-size", str(out_size), "--in-file", str(in_bin), "--out-file", str(out_bin)]
        log = out_dir / f"phase_{i}.log"
        if scrub is not None and backend == "hw":
            failed = _run_scrub(scrub, host, env, out_dir, i, timeout)
            if failed is not None:
                results.append(failed)
                break
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
            rc, text = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            rc, text = TIMEOUT_RC, f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"
        wall = time.monotonic() - t0
        log.write_text(text)
        results.append(PhaseResult(i, rc, round(wall, 3), log.name))
        if rc != 0:
            break                  # a failed phase may have wedged the device: go no further
        for (core, buf), data in unpack_phase_outputs(ph, out_bin.read_bytes()).items():
            name = output_file_name(core, buf)
            (out_dir / name).write_bytes(data)
            outputs.append(name)

    rec = RunRecord(tag, backend, topology.name, git_rev(), so_mtime, tuple(results),
                    tuple(outputs), tuple(input_sources(topology)),
                    manifest.get("inputs_hash", ""), scrub.name if scrub else "")
    write_run_record(out_dir, rec)
    return rec


def _run_scrub(scrub: Scrub, host: Path, env: dict[str, str], out_dir: Path, phase: int,
               timeout: float) -> PhaseResult | None:
    """Launch the scrub package before phase `phase`; None when it ran and every scrub core
    stored its marker, else the PhaseResult that stops the run (log `phase_<i>.scrub.log`)."""
    manifest = json.loads((scrub.pkg_dir / "manifest.json").read_text())
    topology = topology_from_dict(manifest["topology"])
    (ph,) = manifest["plan"]["phases"]           # a scrub is one launch
    tmp_in, tmp_out = out_dir / f"phase_{phase}.scrub.in.bin", out_dir / f"phase_{phase}.scrub.out.bin"
    tmp_in.write_bytes(pack_phase_inputs(topology, ph))
    pdir = scrub.pkg_dir / "phase_0"
    cmd = [str(host), "-x", str(pdir / "aie.xclbin"), "-k", KERNEL_NAME, "-i", str(pdir / "insts.bin"),
           "--in-size", str(max(ph["host_in_bytes"], MIN_BO_BYTES)),
           "--out-size", str(max(ph["host_out_bytes"], MIN_BO_BYTES)),
           "--in-file", str(tmp_in), "--out-file", str(tmp_out)]
    log = out_dir / f"phase_{phase}.scrub.log"
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        rc, text = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = TIMEOUT_RC, f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"
    wall = round(time.monotonic() - t0, 3)
    if rc == 0:
        problems = marker_problems(unpack_phase_outputs(ph, tmp_out.read_bytes()), scrub.out_marker)
        if problems:
            rc, text = SCRUB_MARKER_RC, text + "\nscrub marker check failed: " + "; ".join(problems)
    log.write_text(f"{scrub.name} before phase {phase}\n{text}")
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return None if rc == 0 else PhaseResult(phase, rc, wall, log.name)
