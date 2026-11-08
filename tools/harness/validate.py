"""Topology validator: structural checks against the device model.

`validate` returns every problem it finds as structured `ValidationError`s and
never raises for user mistakes. It deliberately does not look at the files
named by `Core.source` (the build step owns that) and does not plan DMA
channels (the planner owns that); it only rejects topologies that cannot be
realised on the device at all.

Memory budget: a core's data memory minus the compiler's default stack. When
`aie.core` has no `stack_size`, mlir-aie reserves
`AIETargetModel::getDefaultCoreStackSize()` (AIEOps.td, aie.core docs), which
the device-model export carries as `default_core_stack_size`. The harness emits
no `stack_size` and its kernels are leaf assembly, so the default is the
reservation that applies. Buffers are modelled at depth 1 (one buffer per
kernel argument, as in the design note).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from .device_model import (
    DEFAULT_MODELS_PATH, DeviceModel, DeviceModelError, load_device_model,
)
from .topology import Cascade, Core, CoreStream, SharedLock, Topology


# aiecc picks the linker per package (`--xchesscc --xbridge` or not), hence one compiler
# per topology. The Chess install here covers AIE2 only (no AIE2P); mlir-aie names AIE2
# devices `npu1*` (AGENTS.md "Target facts").
SUPPORTED_COMPILERS = ("peano", "chess")
CHESS_DEVICE_PREFIXES = ("npu1",)


@dataclass(frozen=True)
class ValidationError:
    code: str                 # stable identifier, asserted by tests
    message: str
    names: tuple[str, ...]    # offending core/buffer/device names


def validate_topology(topology: Topology,
                      models_path: Path = DEFAULT_MODELS_PATH) -> list[ValidationError]:
    """Load the topology's device (resolving "auto" from the used columns, via
    `plan.resolve_device`), then validate against it."""
    if topology.device == "auto":
        from .plan import PlanError, resolve_device    # plan imports topology only
        resolved = resolve_device(topology, models_path)
        if isinstance(resolved, PlanError):
            return [ValidationError(resolved.code, resolved.message, resolved.names)]
        return validate(dataclasses.replace(topology, device=resolved.name), resolved)
    try:
        device = load_device_model(topology.device, models_path)
    except DeviceModelError as e:
        return [ValidationError("unknown_device", str(e), (topology.device,))]
    return validate(topology, device)


def validate(topology: Topology, device: DeviceModel) -> list[ValidationError]:
    errors: list[ValidationError] = []

    def err(code: str, message: str, *names: str) -> None:
        errors.append(ValidationError(code, message, tuple(names)))

    if topology.device != device.name:
        err("device_mismatch",
            f"topology targets {topology.device!r} but was validated against "
            f"{device.name!r}", topology.device)

    if topology.compiler not in SUPPORTED_COMPILERS:
        err("unsupported_compiler",
            f"compiler {topology.compiler!r} is not supported (one of "
            f"{', '.join(SUPPORTED_COMPILERS)})", topology.compiler)
    elif topology.compiler == "chess" and not device.name.startswith(CHESS_DEVICE_PREFIXES):
        err("chess_aie2_only",
            f"Chess is available for AIE2 only ({', '.join(p + '*' for p in CHESS_DEVICE_PREFIXES)}), "
            f"not {device.name!r}", device.name)

    placed = _check_cores(topology, device, err)
    for core in placed:
        _check_buffers(core, device, err)
    _check_links(topology, device, err)

    shim = device.dma_channels("shim_noc").s2mm
    if not 0 <= topology.reserve_shim_s2mm <= shim:
        err("shim_reserve_range",
            f"reserve_shim_s2mm={topology.reserve_shim_s2mm} outside 0..{shim} "
            f"({device.name} shim S2MM channels)", topology.name)
    return errors


def _check_cores(topology: Topology, device: DeviceModel, err) -> list[Core]:
    """Name/tile checks. Returns cores that sit on a real core tile."""
    placed: list[Core] = []
    seen_names: set[str] = set()
    by_tile: dict[tuple[int, int], list[str]] = {}
    for core in topology.cores:
        if core.name in seen_names:
            err("core_name_duplicate", f"core name {core.name!r} used twice", core.name)
        seen_names.add(core.name)

        col, row = core.tile.col, core.tile.row
        if not 0 <= col < device.columns:
            err("column_out_of_range",
                f"core {core.name!r} at column {col}; {device.name} has "
                f"columns 0..{device.columns - 1}", core.name)
            continue
        tile_type = device.tile_type(col, row)
        if tile_type is None:
            err("tile_missing", f"core {core.name!r}: no tile at ({col},{row})", core.name)
            continue
        if tile_type != "core":
            err("tile_not_core",
                f"core {core.name!r} at ({col},{row}) is a {tile_type} tile", core.name)
            continue
        by_tile.setdefault((col, row), []).append(core.name)
        placed.append(core)

    for (col, row), names in by_tile.items():
        if len(names) > 1:
            err("tile_shared", f"cores {names} share tile ({col},{row})", *names)
    return placed


def _check_buffers(core: Core, device: DeviceModel, err) -> None:
    channels = device.dma_channels("core")
    names: set[str] = set()
    total = 0
    n_in = n_out = 0
    for buf in core.args:
        where = f"{core.name}.{buf.name}"
        if buf.name in names:
            err("buffer_name_duplicate",
                f"core {core.name!r} has two buffers named {buf.name!r}", where)
        names.add(buf.name)

        if buf.direction == "in":
            n_in += 1
        elif buf.direction == "out":
            n_out += 1
        else:
            err("buffer_direction_invalid",
                f"{where}: direction {buf.direction!r} is not 'in' or 'out'", where)

        if not isinstance(buf.size_bytes, int) or buf.size_bytes <= 0 or buf.size_bytes % 4:
            err("buffer_size_invalid",
                f"{where}: size {buf.size_bytes!r} is not a positive multiple of 4", where)
        else:
            total += buf.size_bytes

        if buf.data is not None:
            if buf.direction == "out":
                err("buffer_data_on_out", f"{where}: out buffers cannot carry data", where)
            elif len(buf.data) != buf.size_bytes:
                err("buffer_data_length",
                    f"{where}: data is {len(buf.data)} bytes, size_bytes is "
                    f"{buf.size_bytes}", where)

    budget = device.local_memory_size - device.default_core_stack_size
    if total > budget:
        err("core_memory_overflow",
            f"core {core.name!r}: buffers total {total} bytes, budget {budget} "
            f"({device.local_memory_size} data memory - "
            f"{device.default_core_stack_size} default stack)", core.name)
    if n_in > channels.s2mm:
        err("too_many_in_buffers",
            f"core {core.name!r}: {n_in} in buffers, {channels.s2mm} S2MM channels",
            core.name)
    if n_out > channels.mm2s:
        err("too_many_out_buffers",
            f"core {core.name!r}: {n_out} out buffers, {channels.mm2s} MM2S channels",
            core.name)


def _check_links(topology: Topology, device: DeviceModel, err) -> None:
    cores = {c.name: c for c in topology.cores}
    cascade_out: dict[str, int] = {}
    cascade_in: dict[str, int] = {}
    core_ports = device.stream_ports("core", "Core")
    locks_seen: set[tuple[int, int, int]] = set()

    for link in topology.links:
        missing = [n for n in (link.src, link.dst) if n not in cores]
        for n in missing:
            err("link_endpoint_unknown", f"link names unknown core {n!r}", n)
        if missing:
            continue
        src, dst = cores[link.src], cores[link.dst]

        if isinstance(link, Cascade):
            cascade_out[link.src] = cascade_out.get(link.src, 0) + 1
            cascade_in[link.dst] = cascade_in.get(link.dst, 0) + 1
            hop = device.cascade.direction((src.tile.col, src.tile.row),
                                           (dst.tile.col, dst.tile.row))
            if hop is None:
                legal = ", ".join(f"{d} (src = dst + {o})"
                                  for d, o in device.cascade.adjacency)
                err("cascade_adjacency",
                    f"cascade {link.src!r} -> {link.dst!r}: "
                    f"({src.tile.col},{src.tile.row}) -> "
                    f"({dst.tile.col},{dst.tile.row}) is not a legal hop; "
                    f"legal: {legal}", link.src, link.dst)
        elif isinstance(link, SharedLock):
            _check_lock(link, src, dst, device, err)
            key = (src.tile.col, src.tile.row, link.lock_id)
            if key in locks_seen:
                err("lock_duplicate",
                    f"lock {link.lock_id} of core {link.src!r} is declared twice", link.src)
            locks_seen.add(key)
        elif isinstance(link, CoreStream):
            if link.src == link.dst:
                err("stream_endpoints_same",
                    f"core stream from {link.src!r} to itself", link.src)
            if not 0 <= link.src_port < core_ports.slave:
                err("stream_port_range",
                    f"stream {link.src!r}->{link.dst!r}: src_port {link.src_port} "
                    f"outside 0..{core_ports.slave - 1}", link.src)
            if not 0 <= link.dst_port < core_ports.master:
                err("stream_port_range",
                    f"stream {link.src!r}->{link.dst!r}: dst_port {link.dst_port} "
                    f"outside 0..{core_ports.master - 1}", link.dst)

    for name, n in cascade_out.items():
        if n > 1:
            err("cascade_multiple_out", f"core {name!r} has {n} cascade outputs", name)
    for name, n in cascade_in.items():
        if n > 1:
            err("cascade_multiple_in", f"core {name!r} has {n} cascade inputs", name)


# Memory modules an AIE2 core reaches, as (owner - user) tile offsets: its own (the
# "east" quadrant, raw lock IDs 48..63), south (row - 1, IDs 0..15), west (col - 1,
# 16..31) and north (row + 1, 32..47). mlir-aie AIE2TargetModel::isLegalMemAffinity
# and getLockLocalBaseIndex.
_LOCK_REACH = ((0, 0), (0, -1), (-1, 0), (0, 1))


def _check_lock(link: SharedLock, owner: Core, user: Core, device: DeviceModel, err) -> None:
    n = device.num_locks("core")
    if not 0 <= link.lock_id < n:
        err("lock_id_range", f"lock {link.lock_id} of core {link.src!r}: outside 0..{n - 1}",
            link.src)
    if not 0 <= link.init <= device.max_lock_value:
        err("lock_init_range", f"lock {link.lock_id} of core {link.src!r}: init {link.init} "
            f"outside 0..{device.max_lock_value}", link.src)
    offset = (owner.tile.col - user.tile.col, owner.tile.row - user.tile.row)
    if offset not in _LOCK_REACH:
        err("lock_out_of_reach",
            f"core {link.dst!r} at ({user.tile.col},{user.tile.row}) cannot reach the locks of "
            f"({owner.tile.col},{owner.tile.row})", link.src, link.dst)
