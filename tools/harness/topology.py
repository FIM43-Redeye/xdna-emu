"""Topology schema: what a multi-core probe is, as frozen data.

Hand-written fields only. Everything derived (channel plan, objectfifo names,
packed host layout, MLIR) is computed by later pipeline steps from this plus
the device model. Topologies can be built in Python or read from a JSON file
with `load_topology`; `to_dict` is the inverse.

Field order note: `Topology` lists `cores` before the defaulted fields
(dataclasses require non-default fields first).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal


class TopologyFormatError(ValueError):
    """A topology file or dict does not match the schema."""


@dataclass(frozen=True)
class TileRef:
    """Logical coordinates, as mlir-aie spells them (logical col 0 = physical col 1)."""
    col: int
    row: int


@dataclass(frozen=True)
class Buffer:
    """One kernel argument backed by host memory."""
    name: str                       # unique within the core
    direction: Literal["in", "out"]
    size_bytes: int                 # multiple of 4
    data: bytes | None = None       # inputs only; None = harness PRNG from Topology.seed


@dataclass(frozen=True)
class Core:
    name: str
    tile: TileRef
    source: Path                    # .s (llvm-mc) or .o (prebuilt)
    symbol: str = "test_kernel"
    args: tuple[Buffer, ...] = ()   # kernel ABI = this order, all memref<Nxi32>


@dataclass(frozen=True)
class Cascade:
    """Accumulator cascade, producer -> consumer. Endpoints are core names."""
    src: str
    dst: str


@dataclass(frozen=True)
class CoreStream:
    """aie.flow Core:src_port -> Core:dst_port between two cores."""
    src: str
    dst: str
    src_port: int = 0
    dst_port: int = 0
    # Route through the source tile's stream-switch FIFO 0 (AM020 ch.2:74): the flow is
    # `src Core:src_port -> src FIFO:0` and `src FIFO:0 -> dst Core:dst_port`.
    via_fifo: bool = False


@dataclass(frozen=True)
class SharedLock:
    """`aie.lock(<src tile>, lock_id) {init}`: a lock in the memory module of core `src`'s
    tile that core `dst` also uses (`dst == src` for a lock only its own core touches).

    The lock has a fixed ID, so objectfifo locks are assigned around it, and an init, so
    every load of the design starts it from the same count. Kernels address it by the raw
    ID of the quadrant it sits in, as seen from each core."""
    src: str
    dst: str
    lock_id: int
    init: int = 0


Link = Cascade | CoreStream | SharedLock


@dataclass(frozen=True)
class Topology:
    name: str
    cores: tuple[Core, ...]
    device: str = "npu1"
    links: tuple[Link, ...] = ()
    reserve_shim_s2mm: int = 0      # shim S2MM channels kept free (trace uses them)
    seed: int = 1
    # aiecc picks the linker per package, so the compiler belongs to the package, not to a
    # core. Only "peano" is accepted until the C++ source kind lands (validate.py).
    compiler: str = "peano"


# -- serialization -----------------------------------------------------------

_LINK_KINDS: dict[str, type] = {"cascade": Cascade, "stream": CoreStream, "lock": SharedLock}


def to_dict(t: Topology) -> dict[str, Any]:
    """JSON-ready dict; `topology_from_dict(to_dict(t)) == t`."""
    def buffer(b: Buffer) -> dict[str, Any]:
        d: dict[str, Any] = {"name": b.name, "direction": b.direction,
                             "size_bytes": b.size_bytes}
        if b.data is not None:
            d["data"] = b.data.hex()
        return d

    def core(c: Core) -> dict[str, Any]:
        return {"name": c.name, "tile": {"col": c.tile.col, "row": c.tile.row},
                "source": str(c.source), "symbol": c.symbol,
                "args": [buffer(b) for b in c.args]}

    def link(l: Link) -> dict[str, Any]:
        kind = next(k for k, t in _LINK_KINDS.items() if isinstance(l, t))
        d: dict[str, Any] = {"kind": kind, "src": l.src, "dst": l.dst}
        if isinstance(l, SharedLock):
            d.update(lock_id=l.lock_id, init=l.init)
        if isinstance(l, CoreStream):
            d.update(src_port=l.src_port, dst_port=l.dst_port)
            if l.via_fifo:
                d["via_fifo"] = True
        return d

    return {"name": t.name, "device": t.device,
            "cores": [core(c) for c in t.cores],
            "links": [link(l) for l in t.links],
            "reserve_shim_s2mm": t.reserve_shim_s2mm, "seed": t.seed,
            "compiler": t.compiler}


def _fields(d: Any, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise TopologyFormatError(f"{where}: expected an object, got {type(d).__name__}")
    missing = required - d.keys()
    if missing:
        raise TopologyFormatError(f"{where}: missing key(s) {sorted(missing)}")
    unknown = d.keys() - required - optional
    if unknown:
        raise TopologyFormatError(f"{where}: unknown key(s) {sorted(unknown)}")
    return d


def topology_from_dict(d: dict[str, Any]) -> Topology:
    """Strict inverse of `to_dict`: unknown or missing keys raise."""
    d = _fields(d, {"name", "cores"},
                {"device", "links", "reserve_shim_s2mm", "seed", "compiler"}, "topology")

    def buffer(b: Any, where: str) -> Buffer:
        b = _fields(b, {"name", "direction", "size_bytes"}, {"data"}, where)
        data = None
        if "data" in b:
            try:
                data = bytes.fromhex(b["data"])
            except (ValueError, TypeError):
                raise TopologyFormatError(f"{where}: data is not a hex string") from None
        return Buffer(b["name"], b["direction"], b["size_bytes"], data)

    def core(c: Any, i: int) -> Core:
        where = f"cores[{i}]"
        c = _fields(c, {"name", "tile", "source"}, {"symbol", "args"}, where)
        tile = _fields(c["tile"], {"col", "row"}, set(), f"{where}.tile")
        kwargs: dict[str, Any] = {}
        if "symbol" in c:
            kwargs["symbol"] = c["symbol"]
        args = tuple(buffer(b, f"{where}.args[{j}]") for j, b in enumerate(c.get("args", [])))
        return Core(c["name"], TileRef(tile["col"], tile["row"]), Path(c["source"]),
                    args=args, **kwargs)

    def link(l: Any, i: int) -> Link:
        where = f"links[{i}]"
        if isinstance(l, dict) and l.get("kind") not in _LINK_KINDS:
            raise TopologyFormatError(f"{where}: unknown link kind {l.get('kind')!r}")
        if isinstance(l, dict) and l["kind"] == "cascade":
            l = _fields(l, {"kind", "src", "dst"}, set(), where)
            return Cascade(l["src"], l["dst"])
        if isinstance(l, dict) and l["kind"] == "lock":
            l = _fields(l, {"kind", "src", "dst", "lock_id"}, {"init"}, where)
            return SharedLock(l["src"], l["dst"], l["lock_id"], l.get("init", 0))
        l = _fields(l, {"kind", "src", "dst"}, {"src_port", "dst_port", "via_fifo"}, where)
        return CoreStream(l["src"], l["dst"], l.get("src_port", 0), l.get("dst_port", 0),
                          bool(l.get("via_fifo", False)))

    return Topology(
        name=d["name"],
        cores=tuple(core(c, i) for i, c in enumerate(d["cores"])),
        device=d.get("device", "npu1"),
        links=tuple(link(l, i) for i, l in enumerate(d.get("links", []))),
        reserve_shim_s2mm=d.get("reserve_shim_s2mm", 0),
        seed=d.get("seed", 1),
        compiler=d.get("compiler", "peano"),
    )


def load_topology(path: str | Path) -> Topology:
    """Read a topology file. A relative `Core.source` is relative to the file's directory,
    so a case directory (topology plus its kernels) can be moved as a unit."""
    path = Path(path)
    try:
        t = topology_from_dict(json.loads(path.read_text()))
    except json.JSONDecodeError as e:
        raise TopologyFormatError(f"{path}: invalid JSON: {e}") from None
    base = path.resolve().parent
    return replace(t, cores=tuple(
        c if c.source.is_absolute() else replace(c, source=base / c.source) for c in t.cores))
