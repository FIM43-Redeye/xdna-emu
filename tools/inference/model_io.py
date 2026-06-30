"""Sidecar schema constants + module-kind -> pkt_type translation.

The emulator writes origin_d.json keyed by semantic module kind (core/mem/
memtile/shim). The inference engine keys timer domains by NUMERIC pkt_type
(col|row|pkt_type), the field the trace decoder stamps on each event. This
module is the single place that bridges the two conventions, so the decoder's
pkt_type assignment is never duplicated into Rust.
"""
from __future__ import annotations
from typing import Dict


class SidecarError(Exception):
    """Malformed origin_d.json sidecar or unknown module kind."""


# Decoder convention (col|row|pkt_type). Confirmed: spike.events.json shows
# row0->2, row1->3, compute->0; dma-fill-measure.py shows core->0, mem->1.
MODULE_PKT_TYPE: Dict[str, int] = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def to_domain_key(col: int, row: int, module_kind: str) -> str:
    if module_kind not in MODULE_PKT_TYPE:
        raise SidecarError(f"unknown module_kind {module_kind!r}")
    return f"{col}|{row}|{MODULE_PKT_TYPE[module_kind]}"
