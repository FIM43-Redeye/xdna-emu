"""Several launches in one process: `multi_launch_host.cpp` and how to drive it.

The shared test_host runs one launch per process, so every phase of a suite is a new
emulator instance (and a new process on silicon). State that silicon carries from one
launch to the next within one process (core registers, load-port rings) needs one process
running several launches on one `xrt::device`: this host. Each launch is a
`xclbin,insts,in,out,in_size,out_size[,ctx]` argument (see the C++ header).

`run_launches(host, launches, backend)` runs it on `hw` (XDNA_EMU removed, the caller holds
the NPU) or `emu` (XDNA_EMU=1, this checkout's FFI library).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .run import REPO, XRT, RunError, _env

SOURCE = Path(__file__).with_name("multi_launch_host.cpp")


@dataclass(frozen=True)
class Launch:
    xclbin: Path
    insts: Path
    in_file: Path
    out_file: Path
    in_size: int
    out_size: int
    same_context: bool = False     # a second run in the previous launch's hw context

    def arg(self) -> str:
        for p in (self.xclbin, self.insts, self.in_file, self.out_file):
            if "," in str(p):
                raise RunError(f"a launch path cannot contain ',': {p}")
        ctx = "same" if self.same_context else "new"
        return (f"{self.xclbin},{self.insts},{self.in_file},{self.out_file},"
                f"{self.in_size},{self.out_size},{ctx}")


def build_multi_launch_host(dest_dir: Path) -> Path:
    """Compile the host into `dest_dir` (only when the source is newer)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    exe = dest_dir / "multi_launch_host"
    if exe.exists() and exe.stat().st_mtime >= SOURCE.stat().st_mtime:
        return exe
    tmp = exe.with_name(f".multi_launch_host.{os.getpid()}")
    r = subprocess.run(["clang++", str(SOURCE), "-o", str(tmp), "-std=c++17", "-Wall",
                        "-I", f"{XRT}/include", "-L", f"{XRT}/lib", "-lxrt_coreutil"],
                       capture_output=True, text=True)
    if r.returncode:
        tmp.unlink(missing_ok=True)
        raise RunError(f"building multi_launch_host failed:\n{r.stderr[-1500:]}")
    tmp.replace(exe)
    return exe


def launch_command(host: Path, launches: list[Launch]) -> list[str]:
    if not launches:
        raise RunError("no launches")
    if launches[0].same_context:
        raise RunError("the first launch has no previous context to share")
    return [str(host)] + [l.arg() for l in launches]


def run_launches(host: Path, launches: list[Launch], backend: str, *, runtime: str = "debug",
                 timeout: float = 1800, extra_env: dict[str, str] | None = None
                 ) -> subprocess.CompletedProcess:
    if backend not in ("hw", "emu"):
        raise RunError(f"backend must be 'hw' or 'emu', not {backend!r}")
    env = _env(backend, runtime)
    env.update(extra_env or {})
    return subprocess.run(launch_command(host, launches), env=env, capture_output=True,
                          text=True, timeout=timeout, cwd=REPO)
