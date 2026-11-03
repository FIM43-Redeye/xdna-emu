"""Suite writer for single-core C++ kernels (Peano or Chess).

The instr-test, chess-test and shuffle-sweep generators decide what to test; this turns
their kernels into the suite layout of harness/suite.py, so the shared runner owns
compiling, packaging, running and comparing. Every kernel is one case: one core `core`
at (0,2), ABI `(in, out)` as `extern "C" void test_kernel(const int32_t *in, int32_t *out)`,
result in `core.out.bin`. The compiler is the suite's (aiecc links per package).

Inputs default to the `test_host --seed` PRNG stream of the suite seed, spelled out as
buffer data, so a rerun feeds the kernels what the retired scripts/instr-test.sh did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import suite as S
from harness.run import prng_bytes
from harness.topology import Buffer, Core, TileRef, Topology

DEFAULT_SEED = 42
KERNEL_FILE = "kernel.cc"
WORD = 4


@dataclass(frozen=True)
class CxxKernel:
    name: str                       # case name
    source: str                     # C++ text
    in_size: int                    # bytes the kernel reads from `in`
    out_size: int                   # bytes it writes to `out`
    input_bytes: bytes | None = None    # None: PRNG of the suite seed
    meta: dict[str, Any] = field(default_factory=dict)


def _words(n: int) -> int:
    return -(-n // WORD) * WORD


def write_cxx_suite(suite_dir: str | Path, name: str, generator: str, compiler: str,
                    kernels: list[CxxKernel], *, seed: int = DEFAULT_SEED,
                    skipped: list[dict[str, str]] | None = None,
                    meta: dict[str, Any] | None = None) -> None:
    names = [k.name for k in kernels]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate case names: {dup}")
    Path(suite_dir).mkdir(parents=True, exist_ok=True)
    for k in kernels:
        in_bytes = _words(k.in_size)
        if k.input_bytes is not None and len(k.input_bytes) != k.in_size:
            raise ValueError(f"{k.name}: input is {len(k.input_bytes)} bytes, in_size is {k.in_size}")
        data = (k.input_bytes if k.input_bytes is not None else prng_bytes(k.in_size, seed))
        data = data.ljust(in_bytes, b"\0")
        topo = Topology(k.name, (Core("core", TileRef(0, 2), Path(KERNEL_FILE), args=(
            Buffer("in", "in", in_bytes, data=data),
            Buffer("out", "out", _words(k.out_size)))),), device="npu1", compiler=compiler, seed=seed)
        S.write_case(suite_dir, k.name, topo, files={KERNEL_FILE: k.source},
                     meta={**k.meta, "in_size": k.in_size, "out_size": k.out_size})
    S.write_suite_json(suite_dir, name, generator, names,
                       meta={**(meta or {}), "seed": seed, "compiler": compiler,
                             "skipped": skipped or []})
