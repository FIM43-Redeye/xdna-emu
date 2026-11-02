"""DMA channel planner: how every buffer of a topology reaches host memory.

`plan_channels(topology, device)` returns a `ChannelPlan` (plain frozen data,
printed into run manifests and consumed by the emitter) or a `PlanError`. It
is pure and deterministic: the same input gives an identical plan, and every
ordering follows core order, then buffer order. The topology is assumed to have
passed `validate`; nothing here re-checks tiles or memory budgets.

What it decides
---------------
Every `in` buffer needs a path shim MM2S -> core S2MM and every `out` buffer a
path core MM2S -> shim S2MM. Shim channels are scarce (2 per direction on
Phoenix; `Topology.reserve_shim_s2mm` of the S2MM ones are held back for
trace), so per phase and per column the planner applies, in this order:

1. direct: shim and core in the same column, one shim channel per buffer.
2. staged: when a column's buffers in one direction exceed its free shim
   channels, as many as the memtile can take go through that column's memtile
   with `aie.objectfifo.link`. Outputs JOIN (k core MM2S -> k memtile S2MM ->
   1 memtile MM2S -> 1 shim S2MM), inputs DISTRIBUTE (mirror image). The free
   shim channels are split between direct routes (the first buffers in core
   order) and staged groups, picking the split that places the most buffers,
   then uses the fewest shim channels, then the fewest groups; memtile channels are shared by the column's joins
   and distributes, plus locks, BDs and bytes. A group of one buffer
   degenerates to a direct route (a link of one is a pointless hop). Staging
   in the core's own column is preferred to borrowing because cross-column
   flows load the shim row's East/West stream ports.
3. borrowed: the remainder a column could not place on its own shim (neither
   direct nor staged) take the free shim channels of
   the nearest other column (|dcol|, then lower column), as a direct route
   that crosses columns. Borrowing only uses channels left over after every
   column has placed its own buffers; any device column may lend, used or not.
4. phases: if the whole array cannot hold the topology, cores are split into
   sequential phases (first-fit over units, in core order). Cores connected by a `Cascade`
   or `CoreStream` link form one unit that always lands in a single phase; a
   unit that cannot fit even alone is a `PlanError("unplaceable")`.

Facts relied on (mlir-aie; verified against the installed `aie-opt`)
--------------------------------------------------------------------
- `aie.objectfifo.link [@a, @b] -> [@o] ([0, 16] [])` is a join and
  `[@i] -> [@a, @b] ([] [0, 16])` a distribute. The offsets are ELEMENT
  offsets (not bytes) into the shared object, must strictly increase, and each
  slot must hold a whole number of its participant's objects
  (AIEOps.td ObjectFifoLinkOp docs; AIEDialect.cpp ObjectFifoLinkOp::verify;
  test/objectFifo-stateful-transform/data_movement_patterns/link/
  link_test_join_offsets.mlir uses [0,16,36] for objects of 16, 20 and 12
  elements into a 48-element object). Join and distribute are illegal on
  shim tiles and need semaphore locks, which AIE2 has. A fifo may belong to
  at most one link.
- Each participant gets its own memtile DMA channel: a join of k uses k S2MM
  + 1 MM2S, a distribute of k uses 1 S2MM + k MM2S, and 2k memtile locks
  (one prod/cons pair per participant).
- The memtile holds `depth` copies of the shared (joined) object, where depth
  is the depth of the memtile-side objectfifo (checked by running
  `aie-opt --aie-objectFifo-stateful-transform` with depths 1 and 2 on a 2-way
  join and a 2-way distribute). The core-side fifos have their own, independent
  depth.
- The planner uses depth 1 on every fifo: validate budgets core memory at one
  buffer per argument, the harness has no throughput goal, and depth 1 is
  accepted by the transform. Hardware behaviour of a depth-1 link is not
  exercised until the emitter/runner steps.

Host layout
-----------
Per phase, `in` and `out` are packed in core order then buffer order, each
buffer's offset aligned to `DeviceModel.dma_alignment_bytes` (the shim DMA's
granule: mlir-aie reports "Transfer sizes must be multiples of
addressGranularity/8 bytes" and requires dma_memcpy_nd offsets to be 4-byte
aligned, AIEXDialect.cpp; the JSON's `address_gen_granularity` is in bits).
One exception: the members of a staged group are laid out back to back, at
the position of the group's first member, because the shim moves the whole
joined object as one contiguous transfer. Hence a non-last member of a group
must be a multiple of the alignment (`join_member_misaligned` otherwise).

Not modelled
------------
Stream-switch routing congestion. Cascade, CoreStream and DMA flows all need
switch ports and `aiecc` owns that routing; failures surface in the aiecc
sweep of the emitter step. Cascade links consume no DMA channels.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .device_model import (
    DEFAULT_MODELS_PATH, DeviceModel, DeviceModelError, available_devices,
    load_device_model,
)
from .topology import Buffer, Cascade, Core, CoreStream, Topology

OBJECTFIFO_DEPTH = 1
DEFAULT_DEVICE_FAMILY = "npu1"

RouteKind = Literal["direct", "memtile_join", "memtile_distribute"]


# -- plan data ---------------------------------------------------------------

@dataclass(frozen=True)
class PlanError:
    code: str                  # stable identifier, asserted by tests
    message: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class BufferRoute:
    core: str
    buffer: str
    direction: Literal["in", "out"]
    size_bytes: int
    kind: RouteKind
    shim_col: int                  # logical column of the serving shim
    borrowed: bool                 # shim_col differs from the core's column
    memtile_col: int | None        # memtile used when staged
    group: int | None              # index into Phase.groups when staged
    host_offset: int               # byte offset inside the phase's `in`/`out`
    link_offset: int | None        # byte offset inside the staged object


@dataclass(frozen=True)
class StagingGroup:
    index: int
    kind: Literal["memtile_join", "memtile_distribute"]
    column: int                    # shim and memtile column (never borrowed)
    members: tuple[tuple[str, str], ...]   # (core, buffer), link order
    object_bytes: int              # the shared object: sum of member sizes
    depth: int
    memtile_bytes: int             # object_bytes * depth
    host_offset: int               # start of the contiguous host region


@dataclass(frozen=True)
class TileUsage:
    col: int
    row: int
    tile_type: str
    s2mm_used: int
    s2mm_available: int            # channels minus s2mm_reserved
    mm2s_used: int
    mm2s_available: int
    s2mm_reserved: int = 0
    bytes_used: int = 0            # memtile staging bytes
    bytes_available: int = 0


@dataclass(frozen=True)
class Phase:
    index: int
    cores: tuple[str, ...]
    routes: tuple[BufferRoute, ...]   # core order, then buffer order
    groups: tuple[StagingGroup, ...]
    usage: tuple[TileUsage, ...]      # sorted by (col, row)
    host_in_bytes: int
    host_out_bytes: int


@dataclass(frozen=True)
class ChannelPlan:
    topology: str
    device: str                       # resolved name, never "auto"
    alignment_bytes: int
    objectfifo_depth: int
    reserve_shim_s2mm: int
    phases: tuple[Phase, ...]


# -- device resolution -------------------------------------------------------

def resolve_device(topology: Topology, models_path: Path = DEFAULT_MODELS_PATH,
                   family: str = DEFAULT_DEVICE_FAMILY) -> DeviceModel | PlanError:
    """The topology's device model; `device="auto"` is resolved here.

    Rule for "auto": among the devices of `family` (the name stem, e.g. "npu1"
    and "npu1_Ncol"), take the one with the fewest columns that still covers
    the highest used column; ties break on name. Column counts come from the
    device model, never from the names.
    """
    if topology.device != "auto":
        try:
            return load_device_model(topology.device, models_path)
        except DeviceModelError as e:
            return PlanError("no_device", str(e), (topology.device,))
    need = max((c.tile.col for c in topology.cores), default=0) + 1
    covering = []
    for name in available_devices(models_path):
        if name == family or name.startswith(family + "_"):
            model = load_device_model(name, models_path)
            if model.columns >= need:
                covering.append(model)
    if not covering:
        return PlanError("no_device",
                         f"no {family} device has {need} columns", (family,))
    return min(covering, key=lambda m: (m.columns, m.name))


def plan_channels(topology: Topology, device: DeviceModel | None = None, *,
                  models_path: Path = DEFAULT_MODELS_PATH) -> ChannelPlan | PlanError:
    if not topology.cores:
        return PlanError("no_cores", f"topology {topology.name!r} has no cores")
    if device is None:
        resolved = resolve_device(topology, models_path)
        if isinstance(resolved, PlanError):
            return resolved
        device = resolved

    units = _linked_units(topology)
    cores = list(topology.cores)
    attempt = _plan_phase(0, cores, topology, device)
    phases: list[Phase] = []
    if isinstance(attempt, Phase):
        phases = [attempt]
    else:
        if attempt.fatal:
            return _as_error(attempt, [c.name for c in cores])
        phases_or_err = _split_phases(units, cores, topology, device)
        if isinstance(phases_or_err, PlanError):
            return phases_or_err
        phases = phases_or_err
    bad = next((p for p in phases if isinstance(p, _Fail)), None)
    if bad:     # a phase that fit alone failed once merged: cannot happen, but never hide it
        return _as_error(bad, [c.name for c in cores])
    return ChannelPlan(topology.name, device.name, device.dma_alignment_bytes,
                       OBJECTFIFO_DEPTH, topology.reserve_shim_s2mm, tuple(phases))


# -- phase splitting ---------------------------------------------------------

def _linked_units(topology: Topology) -> list[list[int]]:
    """Core indices grouped by Cascade/CoreStream connectivity, in core order."""
    index = {c.name: i for i, c in enumerate(topology.cores)}
    parent = list(range(len(topology.cores)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for link in topology.links:
        if isinstance(link, (Cascade, CoreStream)) and link.src in index and link.dst in index:
            a, b = find(index[link.src]), find(index[link.dst])
            if a != b:
                parent[max(a, b)] = min(a, b)
    members: dict[int, list[int]] = {}
    for i in range(len(parent)):
        members.setdefault(find(i), []).append(i)
    return [members[k] for k in sorted(members)]


def _split_phases(units: list[list[int]], cores: list[Core], topology: Topology,
                  device: DeviceModel) -> list[Phase] | PlanError:
    """First-fit over linked units in core order: each unit goes into the
    first open phase that still plans, else opens a new one. Phases are
    re-planned whole each time, and keep their cores in topology order."""
    members: list[list[int]] = []

    def attempt(indices: list[int], index: int) -> Phase | _Fail:
        return _plan_phase(index, [cores[i] for i in sorted(indices)], topology, device)

    for unit in units:
        for i, open_phase in enumerate(members):
            trial = attempt(open_phase + unit, i)
            if isinstance(trial, Phase):
                members[i] = sorted(open_phase + unit)
                break
            if trial.fatal:
                return _as_error(trial, [cores[j].name for j in open_phase + unit])
        else:
            alone = attempt(unit, len(members))
            if isinstance(alone, _Fail):
                return _as_error(alone, [cores[i].name for i in unit])
            members.append(list(unit))
    phases = [attempt(m, i) for i, m in enumerate(members)]
    bad = next((p for p in phases if isinstance(p, _Fail)), None)
    if bad:     # each phase planned before, so this cannot happen; never hide it
        return _as_error(bad, [c.name for c in cores])
    return phases


def _as_error(fail: "_Fail", names: list[str]) -> PlanError:
    if fail.fatal:
        return PlanError(fail.code, fail.message, tuple(names))
    return PlanError(fail.code,
                     f"cores {names} cannot be placed in one phase: {fail.message}",
                     tuple(names))


# -- one phase ---------------------------------------------------------------

@dataclass(frozen=True)
class _Fail:
    """Why a phase could not be planned. `fatal` failures are properties of
    the buffers themselves, so splitting into more phases cannot help."""
    message: str
    code: str = "unplaceable"
    fatal: bool = False


@dataclass(frozen=True)
class _Item:
    pos: tuple[int, int]      # (core index in the phase, buffer index): global order
    core: Core
    buf: Buffer


@dataclass
class _Memtile:
    s2mm: int = 0
    mm2s: int = 0
    bytes: int = 0
    locks: int = 0
    bds: int = 0


@dataclass
class _Placement:
    item: _Item
    kind: RouteKind
    shim_col: int
    group: "_Group | None" = None


@dataclass
class _Group:
    kind: Literal["memtile_join", "memtile_distribute"]
    column: int
    items: list[_Item]


def _split_even(items: list[_Item], groups: int) -> list[list[_Item]]:
    base, extra = divmod(len(items), groups)
    out, at = [], 0
    for g in range(groups):
        size = base + (1 if g < extra else 0)
        out.append(items[at:at + size])
        at += size
    return out


def _plan_phase(index: int, cores: list[Core], topology: Topology,
                device: DeviceModel) -> Phase | _Fail:
    """Plan one phase, or return the reason it cannot be placed."""
    shim = device.dma_channels("shim_noc")
    memtile = device.dma_channels("mem_tile")
    reserve = topology.reserve_shim_s2mm
    free = {"in": shim.mm2s, "out": shim.s2mm - reserve}
    depth = OBJECTFIFO_DEPTH
    align = device.dma_alignment_bytes

    by_col: dict[int, dict[str, list[_Item]]] = {}
    for ci, core in enumerate(cores):
        for bi, buf in enumerate(core.args):
            by_col.setdefault(core.tile.col, {"in": [], "out": []})[buf.direction].append(
                _Item((ci, bi), core, buf))

    shim_used = {d: {c: 0 for c in range(device.columns)} for d in ("in", "out")}
    mt = {c: _Memtile() for c in range(device.columns)}
    placed: list[_Placement] = []
    leftover: list[_Item] = []

    def cost(direction: str, chunks: list[list[_Item]]) -> _Memtile:
        add = _Memtile()
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            k = len(chunk)
            if direction == "out":      # join
                add.s2mm += k
                add.mm2s += 1
            else:                       # distribute
                add.s2mm += 1
                add.mm2s += k
            add.bytes += sum(i.buf.size_bytes for i in chunk) * depth
            add.locks += 2 * k
            add.bds += (k + 1) * depth
        return add

    def options(items: list[_Item], direction: str, f: int) -> list[tuple]:
        """Ways to place a column's buffers of one direction on its own shim:
        (placed, channels, groups, d, staged_count, chunks, memtile_cost).
        d buffers go direct (the first ones), the next staged_count are split
        over g staged groups; the rest is left for borrowing."""
        n = len(items)
        if n <= f:
            return [(n, n, 0, n, 0, [], _Memtile())]
        out = [(f, f, 0, f, 0, [], _Memtile())]
        for g in range(1, f + 1):
            for d in range(0, f - g + 1):
                for s_count in range(2, n - d + 1):
                    chunks = [c for c in _split_even(items[d:d + s_count], g) if c]
                    out.append((d + s_count, d + g, g, d, s_count, chunks,
                                cost(direction, chunks)))
        return out

    def joint_fit(a: _Memtile, b: _Memtile) -> bool:
        return (a.s2mm + b.s2mm <= memtile.s2mm and a.mm2s + b.mm2s <= memtile.mm2s
                and (a.bytes + b.bytes) <= device.data_memory_bytes("mem_tile")
                and a.locks + b.locks <= device.num_locks("mem_tile")
                and a.bds + b.bds <= device.num_bds("mem_tile"))

    for col in sorted(by_col):
        # The memtile's channels, locks and bytes are shared by the column's
        # distribute (inputs) and join (outputs), so choose both together:
        # most buffers placed, then fewest shim channels (the rest stay free
        # for borrowers), then fewest groups; first option wins remaining ties.
        in_opts = options(by_col[col]["in"], "in", free["in"])
        out_opts = options(by_col[col]["out"], "out", free["out"])
        best = None
        for oi in in_opts:
            for oo in out_opts:
                if not joint_fit(oi[6], oo[6]):
                    continue
                key = (oi[0] + oo[0], -(oi[1] + oo[1]), -(oi[2] + oo[2]))
                if best is None or key > best[0]:
                    best = (key, oi, oo)
        # Every direction has the direct-only option, which costs no memtile.
        assert best is not None
        for direction, opt in (("in", best[1]), ("out", best[2])):
            items = by_col[col][direction]
            _, _, _, d, s_count, chunks, add = opt
            m = mt[col]
            m.s2mm += add.s2mm; m.mm2s += add.mm2s; m.bytes += add.bytes
            m.locks += add.locks; m.bds += add.bds
            kind = "memtile_join" if direction == "out" else "memtile_distribute"
            for it in items[:d]:
                placed.append(_Placement(it, "direct", col))
            shim_used[direction][col] += d
            for chunk in chunks:
                shim_used[direction][col] += 1
                if len(chunk) < 2:
                    placed.append(_Placement(chunk[0], "direct", col))
                    continue
                group = _Group(kind, col, list(chunk))
                placed.extend(_Placement(it, kind, col, group) for it in chunk)
            leftover.extend(items[d + s_count:])

    for it in leftover:           # borrow, after every column placed its own
        direction, home = it.buf.direction, it.core.tile.col
        lenders = sorted((c for c in range(device.columns)
                          if c != home and shim_used[direction][c] < free[direction]),
                         key=lambda c: (abs(c - home), c))
        if not lenders:
            channel = "MM2S" if direction == "in" else "S2MM"
            return _Fail(f"no free shim {channel} channel for {it.core.name}.{it.buf.name} "
                    f"(column {home}; {free[direction]} usable per shim, "
                    f"reserve_shim_s2mm={reserve}) after staging and borrowing")
        shim_used[direction][lenders[0]] += 1
        placed.append(_Placement(it, "direct", lenders[0]))

    placed.sort(key=lambda p: p.item.pos)
    return _assemble(index, cores, placed, shim_used, mt, device, reserve, depth, align)


def _assemble(index: int, cores: list[Core], placed: list[_Placement],
              shim_used: dict[str, dict[int, int]], mt: dict[int, _Memtile],
              device: DeviceModel, reserve: int, depth: int, align: int) -> Phase | _Fail:
    # Groups get indices in order of their first member, then direction.
    group_list: list[_Group] = []
    for p in placed:
        if p.group is not None and p.group not in group_list:
            group_list.append(p.group)
    group_list.sort(key=lambda g: (g.items[0].pos, g.kind))
    group_index = {id(g): i for i, g in enumerate(group_list)}

    # Host layout, per direction. A group is laid out whole at its first member.
    host_offset: dict[tuple[int, int], int] = {}
    link_offset: dict[tuple[int, int], int] = {}
    group_host: dict[int, int] = {}
    totals = {"in": 0, "out": 0}
    for direction in ("in", "out"):
        cursor = 0
        done: set[tuple[int, int]] = set()
        for p in placed:
            if p.item.buf.direction != direction or p.item.pos in done:
                continue
            cursor = -(-cursor // align) * align
            members = p.group.items if p.group is not None else [p.item]
            if p.group is not None:
                group_host[id(p.group)] = cursor
            at = 0
            for k, m in enumerate(members):
                if k < len(members) - 1 and m.buf.size_bytes % align:
                    return _Fail(
                        f"{m.core.name}.{m.buf.name} ({m.buf.size_bytes} B) is not a "
                        f"multiple of the {align} B DMA alignment but is followed by "
                        f"another member of its staged group",
                        "join_member_misaligned", fatal=True)
                host_offset[m.pos] = cursor + at
                link_offset[m.pos] = at
                at += m.buf.size_bytes
                done.add(m.pos)
            cursor += at
        totals[direction] = cursor

    routes = tuple(BufferRoute(
        core=p.item.core.name, buffer=p.item.buf.name, direction=p.item.buf.direction,
        size_bytes=p.item.buf.size_bytes, kind=p.kind, shim_col=p.shim_col,
        borrowed=p.shim_col != p.item.core.tile.col,
        memtile_col=p.group.column if p.group is not None else None,
        group=group_index[id(p.group)] if p.group is not None else None,
        host_offset=host_offset[p.item.pos],
        link_offset=link_offset[p.item.pos] if p.group is not None else None,
    ) for p in placed)

    groups = tuple(StagingGroup(
        index=i, kind=g.kind, column=g.column,
        members=tuple((m.core.name, m.buf.name) for m in g.items),
        object_bytes=sum(m.buf.size_bytes for m in g.items), depth=depth,
        memtile_bytes=sum(m.buf.size_bytes for m in g.items) * depth,
        host_offset=group_host[id(g)],
    ) for i, g in enumerate(group_list))

    return Phase(index=index, cores=tuple(c.name for c in cores), routes=routes,
                 groups=groups, usage=_usage(cores, shim_used, mt, device, reserve),
                 host_in_bytes=totals["in"], host_out_bytes=totals["out"])


def _tile_row(device: DeviceModel, col: int, tile_type: str) -> int:
    return next(r for r in range(device.rows) if device.tile_type(col, r) == tile_type)


def _usage(cores: list[Core], shim_used: dict[str, dict[int, int]],
           mt: dict[int, _Memtile], device: DeviceModel, reserve: int) -> tuple[TileUsage, ...]:
    shim = device.dma_channels("shim_noc")
    memtile = device.dma_channels("mem_tile")
    core_ch = device.dma_channels("core")
    out: list[TileUsage] = []
    for col in range(device.columns):
        s_in, s_out = shim_used["in"][col], shim_used["out"][col]
        if s_in or s_out:
            out.append(TileUsage(col, _tile_row(device, col, "shim_noc"), "shim_noc",
                                 s2mm_used=s_out, s2mm_available=shim.s2mm - reserve,
                                 mm2s_used=s_in, mm2s_available=shim.mm2s,
                                 s2mm_reserved=reserve))
        m = mt[col]
        if m.s2mm or m.mm2s:
            out.append(TileUsage(col, _tile_row(device, col, "mem_tile"), "mem_tile",
                                 s2mm_used=m.s2mm, s2mm_available=memtile.s2mm,
                                 mm2s_used=m.mm2s, mm2s_available=memtile.mm2s,
                                 bytes_used=m.bytes,
                                 bytes_available=device.data_memory_bytes("mem_tile")))
    for core in cores:
        n_in = sum(1 for b in core.args if b.direction == "in")
        n_out = len(core.args) - n_in
        out.append(TileUsage(core.tile.col, core.tile.row, "core",
                             s2mm_used=n_in, s2mm_available=core_ch.s2mm,
                             mm2s_used=n_out, mm2s_available=core_ch.mm2s))
    return tuple(sorted(out, key=lambda u: (u.col, u.row)))
