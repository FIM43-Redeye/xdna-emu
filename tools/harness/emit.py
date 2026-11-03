"""MLIR emitter: one `aie.device` module per plan phase.

`emit_mlir(topology, plan, phase)` renders the phase's cores, objectfifos,
cascade/stream links and the host runtime sequence as `aie.mlir` text. Every
choice (which shim serves which buffer, which buffers share a memtile object,
host offsets) was made by the planner; this module only spells it. The text is
a pure function of its inputs, so the same topology gives byte-identical MLIR.

Idioms reused from the probes that already run on this machine
(tools/instr-test-gen.py and the retired multi-tile ISA generator): objectfifos of
`memref<Nxi32>`, `link_with` kernels, a `(%in, %buf, %out)` runtime sequence
of `dma_memcpy_nd` ops addressed through the fourth offset/size dimension,
output DMAs issued before inputs with `issue_token` and one `dma_wait` each.

Naming (shared with build.py so object files and symbols agree)
---------------------------------------------------------------
- core-side objectfifo of buffer `b` of core `c`: `of_<c>_<b>`
- staged group `g` of phase `p`: `of_grp<p>_<g>_<in|out>` (the shim-side fifo)
- kernel function and renamed linker symbol of core `c`: `k_<c>`
- kernel object of core `c`: `<c>.o`
Core and buffer names are reduced to `[A-Za-z0-9_]`; two names that collide
after reduction raise `EmitError`.

Host-side pairing: the runtime sequence signature stays `(%in, %buf, %out)`
of the shared test_host. `%in` and `%buf` have the phase's input size and
`%out` its output size, each at least one element.

Shim BD budget
--------------
Each `dma_memcpy_nd` consumes one shim BD. The planner holds one fifo per shim
channel, so a shim never carries more transfers than it has channels, far under
the BD count; the emitter still checks each shim against
`DeviceModel.num_bds("shim_noc")` and raises `EmitError` rather than let aiecc
fail later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .device_model import DeviceModel, load_device_model
from .plan import BufferRoute, ChannelPlan, Phase, StagingGroup
from .topology import Cascade, Core, CoreStream, Topology


class EmitError(ValueError):
    """The plan cannot be spelled as MLIR (naming collision, BD budget, bad phase)."""


# -- naming ------------------------------------------------------------------

def _ident(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def kernel_symbol(core: Core) -> str:
    """The unique linker symbol the core's `core.symbol` is renamed to."""
    return f"k_{_ident(core.name)}"


def kernel_object(core: Core) -> str:
    """File name of the core's renamed object, as referenced by `link_with`."""
    return f"{_ident(core.name)}.o"


def _core_fifo(core: Core, buffer: str) -> str:
    return f"of_{_ident(core.name)}_{_ident(buffer)}"


def _group_fifo(phase: int, group: StagingGroup) -> str:
    direction = "in" if group.kind == "memtile_distribute" else "out"
    return f"of_grp{phase}_{group.index}_{direction}"


def _elems(nbytes: int) -> int:
    return max(1, nbytes // 4)


def _memref(nbytes: int) -> str:
    return f"memref<{_elems(nbytes)}xi32>"


# -- emission ----------------------------------------------------------------

@dataclass(frozen=True)
class _Transfer:
    direction: str      # "in" | "out"
    fifo: str           # shim-side objectfifo named by the DMA op
    host_elems: int     # element offset into %in / %out
    length_elems: int
    shim_col: int


def emit_mlir(topology: Topology, plan: ChannelPlan, phase: int = 0,
              device: DeviceModel | None = None) -> str:
    if not 0 <= phase < len(plan.phases):
        raise EmitError(f"phase {phase} out of range: plan has {len(plan.phases)}")
    ph = plan.phases[phase]
    if device is None:
        device = load_device_model(plan.device)
    by_name = {c.name: c for c in topology.cores}
    cores = [by_name[n] for n in ph.cores]
    _check_names(cores, ph, phase, plan)
    route = {(r.core, r.buffer): r for r in ph.routes}
    group_of = {g.index: g for g in ph.groups}

    tiles = _tiles(cores, ph, device)
    transfers = _transfers(ph, plan, phase, by_name, group_of)
    _check_shim_bds(transfers, device)

    out: list[str] = ["module {", f"  aie.device({plan.device}) {{"]
    for (col, row) in sorted(tiles):
        out.append(f"    %tile_{col}_{row} = aie.tile({col}, {row})")
    out.append("")
    _emit_fifos(out, cores, ph, phase, plan, route, group_of, device)
    _emit_links(out, topology, ph, by_name, device)
    _emit_kernels(out, cores)
    for core in cores:
        _emit_core(out, core, plan)
    _emit_sequence(out, ph, transfers)
    out += ["  }", "}", ""]
    return "\n".join(out)


def _check_names(cores: list[Core], ph: Phase, phase: int, plan: ChannelPlan) -> None:
    seen: dict[str, str] = {}
    def claim(ident: str, what: str) -> None:
        if ident in seen:
            raise EmitError(f"{what} and {seen[ident]} both reduce to {ident!r}")
        seen[ident] = what
    for core in cores:
        claim(f"core {_ident(core.name)}", f"core {core.name!r}")
        for buf in core.args:
            claim(_core_fifo(core, buf.name), f"buffer {core.name}.{buf.name}")


def _tiles(cores: list[Core], ph: Phase, device: DeviceModel) -> set[tuple[int, int]]:
    shim_row, mem_row = _shim_row_of(device), _mem_row_of(device)
    tiles = {(c.tile.col, c.tile.row) for c in cores}
    for r in ph.routes:
        tiles.add((r.shim_col, shim_row))
        if r.memtile_col is not None:
            tiles.add((r.memtile_col, mem_row))
    return tiles


def _shim_row_of(device: DeviceModel) -> int:
    return next(r for r in range(device.rows) if device.tile_type(0, r) == "shim_noc")


def _mem_row_of(device: DeviceModel) -> int:
    return next(r for r in range(device.rows) if device.tile_type(0, r) == "mem_tile")


def _transfers(ph: Phase, plan: ChannelPlan, phase: int, by_name: dict[str, Core],
               group_of: dict[int, StagingGroup]) -> list[_Transfer]:
    """Host transfers in plan order; a staged group is one transfer at its first member."""
    out: list[_Transfer] = []
    seen_groups: set[int] = set()
    for r in ph.routes:
        if r.group is None:
            out.append(_Transfer(r.direction, _core_fifo(by_name[r.core], r.buffer),
                                 r.host_offset // 4, _elems(r.size_bytes), r.shim_col))
        elif r.group not in seen_groups:
            seen_groups.add(r.group)
            g = group_of[r.group]
            out.append(_Transfer(r.direction, _group_fifo(phase, g),
                                 g.host_offset // 4, _elems(g.object_bytes), g.column))
    return out


def _check_shim_bds(transfers: list[_Transfer], device: DeviceModel) -> None:
    limit = device.num_bds("shim_noc")
    per_shim: dict[int, int] = {}
    for t in transfers:
        per_shim[t.shim_col] = per_shim.get(t.shim_col, 0) + 1
    for col, n in sorted(per_shim.items()):
        if n > limit:
            raise EmitError(
                f"shim of column {col} would need {n} BDs for host transfers; "
                f"{device.name} shim has {limit}")


def _emit_fifos(out: list[str], cores: list[Core], ph: Phase, phase: int,
                plan: ChannelPlan, route: dict[tuple[str, str], BufferRoute],
                group_of: dict[int, StagingGroup], device: DeviceModel) -> None:
    depth = plan.objectfifo_depth
    by_name = {c.name: c for c in cores}
    shim_row = _shim_row_of(device)
    mem_row = _mem_row_of(device)

    def tile(col: int, row: int) -> str:
        return f"%tile_{col}_{row}"

    def fifo(name: str, src: str, dst: str, nbytes: int) -> None:
        out.append(f"    aie.objectfifo @{name}({src}, {{{dst}}}, {depth} : i32) "
                   f": !aie.objectfifo<{_memref(nbytes)}>")

    for core in cores:
        ctile = tile(core.tile.col, core.tile.row)
        for buf in core.args:
            r = route[(core.name, buf.name)]
            name = _core_fifo(core, buf.name)
            if r.group is None:
                shim = tile(r.shim_col, shim_row)
                if buf.direction == "in":
                    fifo(name, shim, ctile, buf.size_bytes)
                else:
                    fifo(name, ctile, shim, buf.size_bytes)
            else:
                mem = tile(r.memtile_col, mem_row)
                if buf.direction == "in":
                    fifo(name, mem, ctile, buf.size_bytes)
                else:
                    fifo(name, ctile, mem, buf.size_bytes)

    for g in ph.groups:
        shim = tile(g.column, shim_row)
        mem = tile(g.column, mem_row)
        name = _group_fifo(phase, g)
        members = [_core_fifo(by_name[c], b) for c, b in g.members]
        offsets = ", ".join(str(route[m].link_offset // 4) for m in g.members)
        if g.kind == "memtile_join":
            fifo(name, mem, shim, g.object_bytes)
            out.append(f"    aie.objectfifo.link [{', '.join('@' + m for m in members)}]"
                       f" -> [@{name}]([{offsets}] [])")
        else:
            fifo(name, shim, mem, g.object_bytes)
            out.append(f"    aie.objectfifo.link [@{name}] -> "
                       f"[{', '.join('@' + m for m in members)}]([] [{offsets}])")
    out.append("")


def _emit_links(out: list[str], topology: Topology, ph: Phase,
                by_name: dict[str, Core], device: DeviceModel) -> None:
    in_phase = set(ph.cores)
    lines: list[str] = []
    for link in topology.links:
        if link.src not in in_phase or link.dst not in in_phase:
            continue
        s, d = by_name[link.src].tile, by_name[link.dst].tile
        if isinstance(link, Cascade):
            lines.append(f"    aie.cascade_flow(%tile_{s.col}_{s.row}, %tile_{d.col}_{d.row})")
        elif isinstance(link, CoreStream) and link.via_fifo:
            lines.append(f"    aie.flow(%tile_{s.col}_{s.row}, Core : {link.src_port}, "
                         f"%tile_{s.col}_{s.row}, FIFO : 0)")
            lines.append(f"    aie.flow(%tile_{s.col}_{s.row}, FIFO : 0, "
                         f"%tile_{d.col}_{d.row}, Core : {link.dst_port})")
        elif isinstance(link, CoreStream):
            lines.append(f"    aie.flow(%tile_{s.col}_{s.row}, Core : {link.src_port}, "
                         f"%tile_{d.col}_{d.row}, Core : {link.dst_port})")
    if lines:
        out += lines + [""]


def _emit_kernels(out: list[str], cores: list[Core]) -> None:
    for core in cores:
        sig = ", ".join(_memref(b.size_bytes) for b in core.args)
        out.append(f'    func.func private @{kernel_symbol(core)}({sig}) '
                   f'attributes {{link_with = "{kernel_object(core)}"}}')
    out.append("")


def _emit_core(out: list[str], core: Core, plan: ChannelPlan) -> None:
    out.append(f"    aie.core(%tile_{core.tile.col}_{core.tile.row}) {{")
    names = []
    for i, buf in enumerate(core.args):
        mode = "Consume" if buf.direction == "in" else "Produce"
        out.append(f"      %a{i} = aie.objectfifo.acquire @{_core_fifo(core, buf.name)}"
                   f"({mode}, 1) : {_memref(buf.size_bytes)}")
        names.append(f"%a{i}")
    sig = ", ".join(_memref(b.size_bytes) for b in core.args)
    out.append(f"      func.call @{kernel_symbol(core)}({', '.join(names)}) : ({sig}) -> ()")
    for buf in core.args:
        mode = "Consume" if buf.direction == "in" else "Produce"
        out.append(f"      aie.objectfifo.release @{_core_fifo(core, buf.name)}({mode}, 1)")
    out += ["      aie.end", "    }", ""]


def _emit_sequence(out: list[str], ph: Phase, transfers: list[_Transfer]) -> None:
    n_in, n_out = _elems(ph.host_in_bytes), _elems(ph.host_out_bytes)
    out.append(f"    aie.runtime_sequence(%in : memref<{n_in}xi32>, "
               f"%buf : memref<{n_in}xi32>, %out : memref<{n_out}xi32>) {{")
    out += ["      %c0 = arith.constant 0 : i64", "      %c1 = arith.constant 1 : i64"]
    consts: dict[int, str] = {}

    def const(v: int) -> str:
        if v in (0, 1):
            return f"%c{v}"
        if v not in consts:
            consts[v] = f"%c{v}"
            out.append(f"      %c{v} = arith.constant {v} : i64")
        return consts[v]

    inputs = [t for t in transfers if t.direction == "in"]
    outputs = [t for t in transfers if t.direction == "out"]
    # Constants first (SSA order), then outputs before inputs: receive buffers
    # are armed before any data is sent.
    for t in transfers:
        const(t.host_elems), const(t.length_elems)
    next_id = 0
    ids: dict[str, int] = {}
    for t in inputs:
        ids[t.fifo] = next_id
        next_id += 1
    for t in outputs:
        ids[t.fifo] = next_id
        next_id += 1

    def op(t: _Transfer, host: str, size: int, token: bool) -> str:
        extra = ", issue_token = true" if token else ""
        return (f"      aiex.npu.dma_memcpy_nd({host}[%c0, %c0, %c0, {const(t.host_elems)}]"
                f"[%c1, %c1, %c1, {const(t.length_elems)}][%c0, %c0, %c0, %c1]) "
                f"{{metadata = @{t.fifo}, id = {ids[t.fifo]} : i64{extra}}} "
                f": memref<{size}xi32>")

    for t in outputs:
        out.append(op(t, "%out", n_out, True))
    for t in inputs:
        out.append(op(t, "%in", n_in, False))
    for t in outputs:
        out.append(f"      aiex.npu.dma_wait {{symbol = @{t.fifo}}}")
    out.append("    }")
