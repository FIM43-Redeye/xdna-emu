"""Package builder: topology -> per-phase directory with xclbin + insts.

`build_package(topology, out_dir)` runs validate -> plan -> emit -> build.
Any validation or planning error stops before aiecc. Per phase it writes

    <out_dir>/phase_<i>/aie.mlir, <core>.o ..., aie.xclbin, insts.bin, aiecc.log
    <out_dir>/manifest.json

Kernel objects, by source suffix (`SOURCE_KINDS`): `.s` is assembled with Peano
`llvm-mc --triple=aie2 --filetype=obj`; `.ll` is compiled with Peano
`llc -mtriple=aie2 --issue-limit=1` (one instruction per bundle, as scripts/isa-test.sh
does for single-instruction tests); `.cc` is compiled by the package's compiler (Peano
`clang++`, or `xchesscc_wrapper aie2`; see `cxx_command`); `.o` is copied. Each object's
`core.symbol` is then renamed to the core's unique linker symbol with
`llvm-objcopy --redefine-sym` (the idiom the retired multi-tile ISA generator used), so
several cores can share one source symbol.

The aiecc invocation is the one scripts/isa-test.sh uses; the compiler picks the linker
(`aiecc_flags`): Peano passes `--xchesscc=false --xbridge=false`, Chess `--xchesscc --xbridge`.
Tool paths come from the environment the workspace activates
(`PEANO_INSTALL_DIR`, `PATH`); the manifest records them workspace-relative.

Reuse: with `reuse=True` a package whose inputs hash the same as the one on disk is
not rebuilt. The hash covers the topology (sources by content), the emitted MLIR of
every phase, the aiecc flags and the identity (version, size, mtime) of every tool,
and is stored in the manifest as `inputs_hash`.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .device_model import DeviceModel
from .emit import emit_mlir, kernel_object, kernel_symbol
from .run import RunError, input_sources
from .plan import ChannelPlan, PlanError, plan_channels, resolve_device
from .topology import Topology, to_dict
from .validate import validate, validate_topology

_AIECC_COMMON = ("--get-xclbin", "--xclbin-name=aie.xclbin",
                 "--get-npu-insts", "--npu-insts-name=insts.bin")
_LINKER_FLAGS = {"peano": ("--xchesscc=false", "--xbridge=false"),
                 "chess": ("--xchesscc", "--xbridge")}
PEANO_CXX_FLAGS = ("--target=aie2-none-unknown-elf", "-O2")


def aiecc_flags(compiler: str) -> tuple[str, ...]:
    return _LINKER_FLAGS[compiler] + _AIECC_COMMON


AIECC_FLAGS = aiecc_flags("peano")


class BuildError(RuntimeError):
    """A pipeline stage failed; the message names the log to read."""


@dataclass(frozen=True)
class Package:
    out_dir: Path
    plan: ChannelPlan
    phase_dirs: tuple[Path, ...]
    inputs_hash: str = ""
    reused: bool = False         # True when an up-to-date build was found and left alone


def _workspace() -> Path:
    # Same discovery rule as the rest of the tools: env first, then ancestry.
    env = os.environ.get("NPU_WORK_DIR")
    if env:
        return Path(env)
    for anc in Path(__file__).resolve().parents:
        if (anc / "llvm-aie").is_dir() and (anc / "mlir-aie").is_dir():
            return anc
    return Path(__file__).resolve().parents[2]


def _rel(path: str | Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _peano_tool(name: str) -> str:
    peano = os.environ.get("PEANO_INSTALL_DIR")
    if peano and (Path(peano) / "bin" / name).exists():
        return str(Path(peano) / "bin" / name)
    found = shutil.which(name)
    if found:
        return found
    raise BuildError(f"{name} not found (PEANO_INSTALL_DIR={peano!r}, not on PATH)")


def aiecc_path() -> str | None:
    return shutil.which("aiecc")


def _run(cmd: list[str], cwd: Path | None, log: Path, what: str) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
    if r.returncode:
        raise BuildError(f"{what} failed (exit {r.returncode}); log: {log}")


def _assemble(src: Path, dest: Path, log: Path, compiler: str) -> None:
    _run([_peano_tool("llvm-mc"), "--triple=aie2", "--filetype=obj",
          "-o", str(dest), str(src)], None, log, f"assembling {src.name}")


def _llc(src: Path, dest: Path, log: Path, compiler: str) -> None:
    _run([_peano_tool("llc"), "-mtriple=aie2", "--issue-limit=1", "-filetype=obj",
          "-o", str(dest), str(src)], None, log, f"compiling {src.name}")


def cxx_command(compiler: str, src: Path, dest: Path) -> list[str]:
    """The command that compiles a C++ kernel with the package's compiler."""
    if compiler == "peano":
        return [_peano_tool("clang++"), *PEANO_CXX_FLAGS, "-c", str(src), "-o", str(dest)]
    if compiler == "chess":
        wrapper = shutil.which("xchesscc_wrapper")
        if wrapper is None:
            raise BuildError("xchesscc_wrapper not found on PATH (Chess needs the aietools env)")
        return [wrapper, "aie2", "-c", str(src), "-o", str(dest)]
    raise BuildError(f"no C++ compiler named {compiler!r}")


def _compile_cxx(src: Path, dest: Path, log: Path, compiler: str) -> None:
    # Chess writes listings and scratch beside its cwd; give every kernel a private one so
    # parallel builds never collide and nothing leaks into the package.
    scratch = Path(tempfile.mkdtemp(prefix="cxx-", dir=dest.parent))
    try:
        built = scratch / "kernel.o"       # Chess drops its listing next to the output
        _run(cxx_command(compiler, src.resolve(), built), scratch, log,
             f"compiling {src.name} ({compiler})")
        shutil.move(built, dest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _copy_object(src: Path, dest: Path, log: Path, compiler: str) -> None:
    shutil.copyfile(src, dest)


# source suffix -> function(src, dest object, log, compiler) producing a relocatable AIE2 object
SOURCE_KINDS = {".s": _assemble, ".ll": _llc, ".cc": _compile_cxx, ".o": _copy_object}


def _build_object(core, phase_dir: Path, compiler: str = "peano") -> None:
    dest = phase_dir / kernel_object(core)
    log = phase_dir / f"{kernel_object(core)}.log"
    src = Path(core.source)
    if not src.is_file():
        raise BuildError(f"core {core.name!r}: source {src} does not exist")
    make = SOURCE_KINDS.get(src.suffix)
    if make is None:
        raise BuildError(f"core {core.name!r}: unsupported source type {src.suffix!r} "
                         f"(one of {', '.join(SOURCE_KINDS)})")
    make(src, dest, log, compiler)
    _run([_peano_tool("llvm-objcopy"), "--redefine-sym",
          f"{core.symbol}={kernel_symbol(core)}", str(dest)],
         None, log.with_suffix(".objcopy.log"), f"renaming symbol in {dest.name}")


def _git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


@functools.lru_cache(maxsize=None)
def _tool_versions(root: Path) -> dict:
    """Path, version line and file stamp (size, mtime) of every tool a build runs."""
    def first_line(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def entry(p: str) -> dict:
        st = Path(p).stat()
        return {"path": _rel(p, root), "version": first_line([p, "--version"]),
                "stamp": f"{st.st_size}:{st.st_mtime_ns}"}

    tools = {"aiecc": entry(aiecc_path()) if aiecc_path() else {"path": "", "version": "", "stamp": ""}}
    for name in ("llvm-mc", "llc", "llvm-objcopy"):
        try:
            tools[name] = entry(_peano_tool(name))
        except BuildError:
            tools[name] = {"path": "", "version": "", "stamp": ""}
    return tools


def _compiler_tools(topology: Topology, root: Path) -> dict:
    """Identity of the compiler tools beyond the base set: Peano `clang++` when a core is C++,
    and Chess (wrapper and driver) for every Chess package, since aiecc runs Chess on the cores.
    Empty for the plain Peano assembly packages, so their hashes do not move."""
    def stamp(path: str) -> dict:
        st = Path(path).stat()
        return {"path": _rel(path, root), "stamp": f"{st.st_size}:{st.st_mtime_ns}"}

    tools: dict = {}
    if any(Path(c.source).suffix == ".cc" for c in topology.cores):
        tools["cxx_flags"] = list(PEANO_CXX_FLAGS) if topology.compiler == "peano" else []
        if topology.compiler == "peano":
            tools["clang++"] = stamp(_peano_tool("clang++"))
    if topology.compiler == "chess":
        for name in ("xchesscc_wrapper", "xchesscc"):
            found = shutil.which(name)
            tools[name] = stamp(found) if found else {"path": "", "stamp": ""}
    return tools


def compute_inputs_hash(topology: Topology, mlir: list[str], root: Path) -> str:
    """Hash of everything a build depends on; sources count by content, not path."""
    sources = {}
    for c in topology.cores:
        src = Path(c.source)
        if not src.is_file():
            raise BuildError(f"core {c.name!r}: source {src} does not exist")
        sources[c.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    doc = {"topology": {**to_dict(topology), "cores": [
               {**c, "source": sources[c["name"]]} for c in to_dict(topology)["cores"]]},
           "mlir": mlir, "aiecc_flags": list(aiecc_flags(topology.compiler)),
           "tools": _tool_versions(root)}
    extra = _compiler_tools(topology, root)
    if extra:
        doc["compiler_tools"] = extra
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _up_to_date(out_dir: Path, inputs_hash: str, phases: int) -> bool:
    try:
        m = json.loads((out_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return m.get("inputs_hash") == inputs_hash and all(
        (out_dir / f"phase_{i}" / n).is_file()
        for i in range(phases) for n in ("aie.xclbin", "insts.bin"))


def build_package(topology: Topology, out_dir: str | Path, *, run_aiecc: bool = True,
                  device: DeviceModel | None = None, reuse: bool = False) -> Package:
    """Validate, plan, emit and build every phase of `topology` into `out_dir`.

    `device` overrides the planning model (tests shrink the memtile to reach
    plan shapes the real limits never produce); the emitted MLIR still targets
    the real device of that name. `reuse` leaves an up-to-date build in place (see the
    module docstring); it needs `run_aiecc`, since a package without xclbins is never current.
    """
    out_dir = Path(out_dir)
    if device is None:
        device = resolve_device(topology)
        if isinstance(device, PlanError):
            raise BuildError(f"{topology.name}: {device.code}: {device.message}")
        errors = validate_topology(topology)
    else:
        errors = validate(dataclasses.replace(topology, device=device.name), device)
    if errors:
        raise BuildError(f"{topology.name}: validation failed:\n" +
                         "\n".join(f"  {e.code}: {e.message}" for e in errors))
    plan = plan_channels(topology, device)
    if isinstance(plan, PlanError):
        raise BuildError(f"{topology.name}: planning failed: {plan.code}: {plan.message}")

    try:
        inputs = input_sources(topology)     # fails early on mis-sized Buffer.data
    except RunError as e:
        raise BuildError(f"{topology.name}: {e}") from None

    out_dir.mkdir(parents=True, exist_ok=True)
    root = _workspace()
    mlir = [emit_mlir(topology, plan, ph.index, device) for ph in plan.phases]
    inputs_hash = compute_inputs_hash(topology, mlir, root)
    if reuse and run_aiecc and _up_to_date(out_dir, inputs_hash, len(plan.phases)):
        return Package(out_dir, plan, tuple(out_dir / f"phase_{p.index}" for p in plan.phases),
                       inputs_hash, reused=True)

    (out_dir / "manifest.json").unlink(missing_ok=True)   # a half-built package is never current
    by_name = {c.name: c for c in topology.cores}
    phase_dirs: list[Path] = []
    for ph, text in zip(plan.phases, mlir):
        pdir = out_dir / f"phase_{ph.index}"
        pdir.mkdir(exist_ok=True)
        (pdir / "aie.mlir").write_text(text)
        for name in ph.cores:
            _build_object(by_name[name], pdir, topology.compiler)
        if run_aiecc:
            exe = aiecc_path()
            if exe is None:
                raise BuildError("aiecc not found on PATH")
            _run([exe, *aiecc_flags(topology.compiler), "aie.mlir"], pdir, pdir / "aiecc.log",
                 f"aiecc for {topology.name} phase {ph.index}")
        phase_dirs.append(pdir)

    manifest = {
        "topology": to_dict(topology),
        "plan": dataclasses.asdict(plan),
        "git_rev": _git_rev(),
        "tools": {**_tool_versions(root), **_compiler_tools(topology, root)},
        "aiecc_flags": list(aiecc_flags(topology.compiler)),
        "inputs": inputs,
        "inputs_hash": inputs_hash,
        "phases": [f"phase_{p.index}" for p in plan.phases],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                      default=str) + "\n")
    return Package(out_dir, plan, tuple(phase_dirs), inputs_hash)
