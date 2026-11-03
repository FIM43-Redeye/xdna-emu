"""The ISA test generator's manifest as a harness suite (one case per batch).

`isa-test-gen.py` decides what to test and writes the batches' sources and a manifest;
this module turns that into the suite layout of harness/suite.py, so the shared runner
owns assembling, packaging, running and comparing:

- `assembly` / `llvm_ir` batch: one core `core` at (0,2), kernel ABI `(in, out)`.
- `cascade_pair`: `producer` at (0,3) directly north of `consumer` at (0,2), joined by
  a `Cascade`. The producer takes `(in, out)`; the consumer's kernel takes only its
  output pointer (the generated assembly addresses it as p0).
- `stream_pair`: the same two tiles joined by a `CoreStream` (`aie.flow` Core:0 -> Core:0);
  both kernels take only their output pointer.

Inputs are the shared test_host PRNG stream of the suite seed, spelled out as buffer
data, so a rerun with the same seed feeds the kernels exactly what `test_host --seed`
did. `case.json` carries what the analyzer needs: the batch's identity and its test
points, each with `slices` naming the bytes of which core's output buffer it owns (a
pair's test point owns both cores' outputs).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness import suite as S
from harness.run import prng_bytes
from harness.topology import Buffer, Cascade, Core, CoreStream, TileRef, Topology

SUITE_NAME = "isa"
PAIR_TYPES = ("cascade_pair", "stream_pair")
CONSUMER_ROW, PRODUCER_ROW = 2, 3


class IsaSuiteError(RuntimeError):
    """The manifest or its sources cannot be rendered as a suite."""


@dataclass(frozen=True)
class IsaCase:
    name: str
    topology: Topology
    files: dict[str, str]
    meta: dict[str, Any]


def case_name(batch: dict[str, Any]) -> str:
    return f"batch_{batch['batch_index']:03d}"


def _source(sources: dict[str, str], filename: str, batch: dict[str, Any]) -> str:
    if filename not in sources:
        raise IsaSuiteError(f"{case_name(batch)}: source {filename} not found")
    return sources[filename]


def _suffix(filename: str) -> str:
    return Path(filename).suffix


def _buffers(in_size: int, out_size: int, seed: int) -> tuple[Buffer, ...]:
    bufs = []
    if in_size:
        bufs.append(Buffer("in", "in", in_size, data=prng_bytes(in_size, seed)))
    bufs.append(Buffer("out", "out", out_size))
    return tuple(bufs)


def _slice(core: str, offset: int, size: int) -> dict[str, Any]:
    return {"core": core, "buffer": "out", "offset": offset, "size": size}


def build_case(batch: dict[str, Any], sources: dict[str, str], seed: int) -> IsaCase:
    """One manifest batch as a case. `sources` maps source filename to its text."""
    name, kind = case_name(batch), batch.get("source_type", "assembly")
    base_meta = {"batch_index": batch["batch_index"], "source_type": kind,
                 "in_size": batch["in_size"], "out_size": batch["out_size"]}
    if kind in ("assembly", "llvm_ir"):
        src = f"kernel{_suffix(batch['filename'])}"
        core = Core("core", TileRef(0, CONSUMER_ROW), Path(src),
                    args=_buffers(batch["in_size"], batch["out_size"], seed))
        topo = Topology(name, (core,))
        files = {src: _source(sources, batch["filename"], batch)}
        tests = [{**t, "slices": [_slice("core", t["out_offset"], t["out_size"])]}
                 for t in batch["tests"]]
        meta = {**base_meta, "tests": tests}
    elif kind in PAIR_TYPES:
        if batch["consumer_in_size"]:
            raise IsaSuiteError(f"{name}: a pair consumer with input is not supported")
        psrc = f"producer{_suffix(batch['producer_filename'])}"
        csrc = f"consumer{_suffix(batch['consumer_filename'])}"
        prod = Core("producer", TileRef(0, PRODUCER_ROW), Path(psrc),
                    args=_buffers(batch["producer_in_size"], batch["producer_out_size"], seed))
        cons = Core("consumer", TileRef(0, CONSUMER_ROW), Path(csrc),
                    args=_buffers(0, batch["consumer_out_size"], seed))
        link = (Cascade("producer", "consumer") if kind == "cascade_pair"
                else CoreStream("producer", "consumer"))
        topo = Topology(name, (prod, cons), links=(link,))
        files = {psrc: _source(sources, batch["producer_filename"], batch),
                 csrc: _source(sources, batch["consumer_filename"], batch)}
        slices = [_slice("producer", 0, batch["producer_out_size"]),
                  _slice("consumer", 0, batch["consumer_out_size"])]
        meta = {**base_meta, "instruction": batch["instruction"], "slot": batch["slot"],
                "tests": [{**t, "slices": slices} for t in batch["tests"]]}
    else:
        raise IsaSuiteError(f"{name}: unknown source_type {kind!r}")
    return IsaCase(name, topo, files, meta)


def write_isa_suite(manifest: dict[str, Any], src_dir: str | Path, suite_dir: str | Path,
                    seed: int) -> None:
    """Write the suite for `manifest`, reading each batch's sources from `src_dir`.

    Cases of an earlier generation are removed (built packages under pkgs/ stay: the
    builder reuses them by content hash).
    """
    src_dir, suite_dir = Path(src_dir), Path(suite_dir)
    names = {f for b in manifest["batches"]
             for f in (b.get("filename"), b.get("producer_filename"), b.get("consumer_filename")) if f}
    sources = {}
    for f in sorted(names):
        p = src_dir / f
        if not p.is_file():
            raise IsaSuiteError(f"source {p} not found")
        sources[f] = p.read_text()
    shutil.rmtree(suite_dir / "cases", ignore_errors=True)
    suite_dir.mkdir(parents=True, exist_ok=True)
    cases = [build_case(b, sources, seed) for b in manifest["batches"]]
    for c in cases:
        S.write_case(suite_dir, c.name, c.topology, files=c.files, meta=c.meta)
    meta = {k: v for k, v in manifest.items() if k != "batches"}
    S.write_suite_json(suite_dir, SUITE_NAME, "isa-test-gen.py", [c.name for c in cases],
                       meta={**meta, "seed": seed})


def filter_cases(suite: S.Suite, pattern: str) -> list[str]:
    """Cases whose name, source type or any instruction name matches the regex `pattern`."""
    rx = re.compile(pattern)
    out = []
    for case in suite.cases:
        m = suite.case_meta(case)
        haystack = [case, m.get("source_type", ""), *(t["instruction"] for t in m.get("tests", []))]
        if any(rx.search(h) for h in haystack):
            out.append(case)
    return out
