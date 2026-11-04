"""Scrub kernel: put every core register of a case's tiles into a known state before the case.

Silicon keeps a core's data registers across kernel launches (separate processes, separate
xclbins; build/experiments/harness/scrub/step1), so a case that reads a register it never
wrote sees whatever ran before it, and its output depends on suite order. Before each phase of
a hardware run the runner therefore launches the scrub package on the phase's tiles: one core
per tile, each loading a table into every register (`test_kernel(in, out)`, out = a marker
word). An emulator run starts every core from the same values (`XDNA_EMU_CORE_INIT`).

Pattern, table, register list and kernel source come from one place, the emulator's
`src/interpreter/state/scrub.rs`, via the checked-in `scrub_kernel.json` (regenerate with
`cargo run --bin scrub-kernel > tools/harness/scrub_kernel.json`; a Rust test fails when it is
stale). The scrub package is built once per suite and tile set under `<suite>/scrub/<key>/`.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .topology import Buffer, Core, TileRef, Topology, to_dict

ARTIFACT = Path(__file__).with_name("scrub_kernel.json")
MARKER_BYTES = 4


@dataclass(frozen=True)
class ScrubKernel:
    name: str              # "scrub-v<version>": what run records and XDNA_EMU_CORE_INIT carry
    version: int
    out_marker: int        # the word each scrub core stores to its output
    table: bytes           # each scrub core's input
    kernel_s: str


def load_scrub_kernel(path: str | Path = ARTIFACT) -> ScrubKernel:
    d = json.loads(Path(path).read_text())
    return ScrubKernel(d["name"], d["version"], d["out_marker"], bytes.fromhex(d["table_hex"]),
                       d["kernel_s"])


@dataclass(frozen=True)
class Scrub:
    """What `run_package` needs: the pattern name, and on hardware the built scrub package."""
    name: str
    out_marker: int
    pkg_dir: Path | None = None      # hardware only


def case_tiles(topology: Topology) -> tuple[TileRef, ...]:
    """Every core tile the topology uses, sorted (all phases: a topology's cores are its tiles)."""
    return tuple(sorted({c.tile for c in topology.cores}, key=lambda t: (t.col, t.row)))


def scrub_key(device: str, tiles: Iterable[TileRef]) -> str:
    return device + "_" + "_".join(f"c{t.col}r{t.row}" for t in tiles)


def scrub_topology(kernel: ScrubKernel, device: str, tiles: Iterable[TileRef]) -> Topology:
    """One scrub core per tile; the table is each core's input, the marker its output."""
    tiles = tuple(tiles)
    if not tiles:
        raise ValueError("a scrub needs at least one tile")
    cores = tuple(Core(f"scrub_c{t.col}r{t.row}", t, Path("scrub.s"), args=(
        Buffer("in", "in", len(kernel.table), data=kernel.table),
        Buffer("out", "out", MARKER_BYTES))) for t in tiles)
    return Topology(f"{kernel.name}_{scrub_key(device, tiles)}", cores, device=device)


def write_scrub_case(kernel: ScrubKernel, case_dir: Path, device: str,
                     tiles: Iterable[TileRef]) -> Topology:
    """Write the scrub case (topology.json + scrub.s) and return its topology, sources resolved."""
    from dataclasses import replace
    topo = scrub_topology(kernel, device, tiles)
    case_dir.mkdir(parents=True, exist_ok=True)
    src = case_dir / "scrub.s"
    if not src.is_file() or src.read_text() != kernel.kernel_s:
        src.write_text(kernel.kernel_s)
    (case_dir / "topology.json").write_text(json.dumps(to_dict(topo), indent=1) + "\n")
    return replace(topo, cores=tuple(replace(c, source=src.resolve()) for c in topo.cores))


def marker_problems(scrub_outputs: dict[tuple[str, str], bytes], marker: int) -> list[str]:
    """Scrub cores whose output is not the marker (the scrub did not run to its end)."""
    want = struct.pack("<I", marker)
    return [f"{core}: output {data.hex()} is not the marker {want.hex()}"
            for (core, _), data in sorted(scrub_outputs.items()) if data[:MARKER_BYTES] != want]
