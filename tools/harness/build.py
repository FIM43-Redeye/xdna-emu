"""Package builder: topology -> per-phase directory with xclbin + insts.

`build_package(topology, out_dir)` runs validate -> plan -> emit -> build.
Any validation or planning error stops before aiecc. Per phase it writes

    <out_dir>/phase_<i>/aie.mlir, <core>.o ..., aie.xclbin, insts.bin, aiecc.log
    <out_dir>/manifest.json

Kernel objects: a `.s` source is assembled with Peano `llvm-mc
--triple=aie2 --filetype=obj` (as tools/run_acc_timing_probe.py does); a `.o`
is copied. Each object's `core.symbol` is then renamed to the core's unique
linker symbol with `llvm-objcopy --redefine-sym` (as tools/isa-multi-tile-gen.py
does), so several cores can share one source symbol.

The aiecc invocation is the one scripts/isa-test.sh uses (Peano, no Chess).
Tool paths come from the environment the workspace activates
(`PEANO_INSTALL_DIR`, `PATH`); the manifest records them workspace-relative.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .device_model import DeviceModel
from .emit import emit_mlir, kernel_object, kernel_symbol
from .run import RunError, input_sources
from .plan import ChannelPlan, PlanError, plan_channels, resolve_device
from .topology import Topology, to_dict
from .validate import validate, validate_topology

AIECC_FLAGS = ("--xchesscc=false", "--xbridge=false",
               "--get-xclbin", "--xclbin-name=aie.xclbin",
               "--get-npu-insts", "--npu-insts-name=insts.bin")


class BuildError(RuntimeError):
    """A pipeline stage failed; the message names the log to read."""


@dataclass(frozen=True)
class Package:
    out_dir: Path
    plan: ChannelPlan
    phase_dirs: tuple[Path, ...]


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


def _build_object(core, phase_dir: Path) -> None:
    dest = phase_dir / kernel_object(core)
    log = phase_dir / f"{kernel_object(core)}.log"
    src = Path(core.source)
    if not src.is_file():
        raise BuildError(f"core {core.name!r}: source {src} does not exist")
    if src.suffix == ".s":
        _run([_peano_tool("llvm-mc"), "--triple=aie2", "--filetype=obj",
              "-o", str(dest), str(src)], None, log, f"assembling {src.name}")
    elif src.suffix == ".o":
        shutil.copyfile(src, dest)
    else:
        raise BuildError(f"core {core.name!r}: unsupported source type {src.suffix!r}")
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


def _tool_versions(root: Path) -> dict:
    def first_line(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""
    aiecc = aiecc_path() or ""
    tools = {"aiecc": {"path": _rel(aiecc, root), "version": first_line([aiecc, "--version"]) if aiecc else ""}}
    for name in ("llvm-mc", "llvm-objcopy"):
        try:
            p = _peano_tool(name)
            tools[name] = {"path": _rel(p, root), "version": first_line([p, "--version"])}
        except BuildError:
            tools[name] = {"path": "", "version": ""}
    return tools


def build_package(topology: Topology, out_dir: str | Path, *, run_aiecc: bool = True,
                  device: DeviceModel | None = None) -> Package:
    """Validate, plan, emit and build every phase of `topology` into `out_dir`.

    `device` overrides the planning model (tests shrink the memtile to reach
    plan shapes the real limits never produce); the emitted MLIR still targets
    the real device of that name.
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
    by_name = {c.name: c for c in topology.cores}
    phase_dirs: list[Path] = []
    for ph in plan.phases:
        pdir = out_dir / f"phase_{ph.index}"
        pdir.mkdir(exist_ok=True)
        (pdir / "aie.mlir").write_text(emit_mlir(topology, plan, ph.index, device))
        for name in ph.cores:
            _build_object(by_name[name], pdir)
        if run_aiecc:
            exe = aiecc_path()
            if exe is None:
                raise BuildError("aiecc not found on PATH")
            _run([exe, *AIECC_FLAGS, "aie.mlir"], pdir, pdir / "aiecc.log",
                 f"aiecc for {topology.name} phase {ph.index}")
        phase_dirs.append(pdir)

    root = _workspace()
    manifest = {
        "topology": to_dict(topology),
        "plan": dataclasses.asdict(plan),
        "git_rev": _git_rev(),
        "tools": _tool_versions(root),
        "aiecc_flags": list(AIECC_FLAGS),
        "inputs": inputs,
        "phases": [f"phase_{p.index}" for p in plan.phases],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                      default=str) + "\n")
    return Package(out_dir, plan, tuple(phase_dirs))
