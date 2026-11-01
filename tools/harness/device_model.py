"""Device facts for the harness, read from tools/aie-device-models.json.

The JSON is generated from mlir-aie's target models by
`tools/mlir-aie-bridge.py device-model`. This module only reshapes it into
typed lookups so that no hardware number is ever written in harness code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_PATH = Path(__file__).resolve().parents[1] / "aie-device-models.json"

TileCoord = tuple[int, int]


class DeviceModelError(Exception):
    """The requested device or fact is not in the device-model JSON."""


@dataclass(frozen=True)
class DmaChannels:
    s2mm: int   # stream -> memory: receives data (a kernel `in` buffer)
    mm2s: int   # memory -> stream: sends data (a kernel `out` buffer)


@dataclass(frozen=True)
class StreamPorts:
    master: int   # switchbox destination ports of this bundle
    slave: int    # switchbox source ports of this bundle


@dataclass(frozen=True)
class CascadeRule:
    width_bits: int
    core_tiles_only: bool
    # direction -> (dcol, drow), meaning src = dst + offset
    adjacency: tuple[tuple[str, TileCoord], ...]

    def direction(self, src: TileCoord, dst: TileCoord) -> str | None:
        """Flow direction if src -> dst is a legal cascade hop, else None."""
        for direction, (dc, dr) in self.adjacency:
            if src == (dst[0] + dc, dst[1] + dr):
                return direction
        return None


@dataclass(frozen=True)
class DeviceModel:
    name: str
    columns: int
    rows: int
    local_memory_size: int
    mem_tile_size: int
    default_core_stack_size: int
    address_gen_granularity_bits: int   # mlir-aie getAddressGenGranularity(): BITS
    cascade: CascadeRule
    _tile_types: dict[TileCoord, str]
    _type_configs: dict[str, dict]

    def tile_type(self, col: int, row: int) -> str | None:
        """Tile type string ("core", "mem_tile", "shim_noc") or None if absent."""
        return self._tile_types.get((col, row))

    def data_memory_bytes(self, tile_type: str) -> int:
        return self.mem_tile_size if tile_type == "mem_tile" else self.local_memory_size

    def num_locks(self, tile_type: str) -> int:
        return self._type_config(tile_type)["num_locks"]

    def num_bds(self, tile_type: str) -> int:
        return self._type_config(tile_type)["num_bds"]

    @property
    def dma_alignment_bytes(self) -> int:
        """Granule of DMA addressing and transfer length, in bytes.

        mlir-aie's AIEXDialect.cpp reports "Transfer sizes must be multiples
        of (addressGranularity / 8) bytes", so the stored bit width is divided
        by 8 (32 bits -> 4 bytes on AIE2).
        """
        return self.address_gen_granularity_bits // 8

    @property
    def program_memory_bytes(self) -> int:
        return self._type_config("core")["program_memory_size"]

    def stream_ports(self, tile_type: str, bundle: str) -> StreamPorts:
        """Switchbox port counts for one bundle ("Core", "DMA", "North", ...)."""
        ports = self._type_config(tile_type)["switchbox_ports"].get(bundle)
        if ports is None:
            return StreamPorts(0, 0)
        return StreamPorts(ports["master"], ports["slave"])

    def dma_channels(self, tile_type: str) -> DmaChannels:
        """DMA channels per direction. Shim DMA ports live in the shim mux."""
        cfg = self._type_config(tile_type)
        ports = cfg["switchbox_ports"].get("DMA") or cfg.get("shim_mux_ports", {}).get("DMA")
        if ports is None:
            raise DeviceModelError(f"{self.name}: tile type {tile_type!r} has no DMA ports")
        # A switchbox master is a destination (into the DMA = S2MM); a slave
        # is a source (out of the DMA = MM2S).
        return DmaChannels(s2mm=ports["master"], mm2s=ports["slave"])

    def _type_config(self, tile_type: str) -> dict:
        try:
            return self._type_configs[tile_type]
        except KeyError:
            raise DeviceModelError(f"{self.name}: no tile type {tile_type!r}") from None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())["devices"]


def available_devices(path: Path = DEFAULT_MODELS_PATH) -> list[str]:
    return sorted(_load_json(path))


def load_device_model(name: str, path: Path = DEFAULT_MODELS_PATH) -> DeviceModel:
    devices = _load_json(path)
    if name not in devices:
        raise DeviceModelError(
            f"unknown device {name!r}; available: {', '.join(sorted(devices))}")
    d = devices[name]
    try:
        cascade = d["cascade"]
        rule = CascadeRule(
            width_bits=cascade["width_bits"],
            core_tiles_only=cascade["core_tiles_only"],
            adjacency=tuple((a["direction"], (a["src_offset"][0], a["src_offset"][1]))
                            for a in cascade["adjacency"]),
        )
        return DeviceModel(
            name=name,
            columns=d["columns"],
            rows=d["rows"],
            local_memory_size=d["local_memory_size"],
            mem_tile_size=d["mem_tile_size"],
            default_core_stack_size=d["default_core_stack_size"],
            address_gen_granularity_bits=d["address_gen_granularity"],
            cascade=rule,
            _tile_types={(t["col"], t["row"]): t["type"] for t in d["tile_map"]},
            _type_configs=d["tile_types"],
        )
    except KeyError as e:
        raise DeviceModelError(
            f"{name}: device-model JSON lacks {e}; regenerate with "
            "tools/mlir-aie-bridge.py device-model") from None
