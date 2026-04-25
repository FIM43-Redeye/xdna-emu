# SPDX-License-Identifier: MIT
"""Top-level trace-decode orchestration.

Pipeline:

  raw uint32 word stream
        |
        v
  deinterleave_packets        (packet.py: group payloads by tile)
        |
        v
  words_to_bytes              (packet.py: MSB-first byte stream per tile)
        |
        v
  decode (per-mode)           (modes/mode{0,1,2}.py: opcode stream -> commands)
        |
        v
  rebuild_timeline            (this module: commands -> Event records)

The timeline rebuild is the mode-0 algorithm documented in mlir-aie's
``convert_commands_to_json`` (Apache 2.0); we re-implement it here on
our typed command stream.  Mode-1 and mode-2 timelines will hook in
through alternative rebuild functions on the same EventCmd schema.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .frame import (
    Event,
    EventCmd,
    PacketType,
    RepeatCmd,
    StartCmd,
    StopCmd,
    SyncCmd,
    TraceCommand,
    TraceMode,
)
from .modes import mode0
from .packet import deinterleave_packets, trim_trailing_padding, words_to_bytes


# Sync advances the timer by a fixed value documented in mlir-aie's
# convert_commands_to_json with a "Typo in spec" annotation -- the value
# is what the on-tile encoder actually emits, so we match it for
# bit-equivalence.
_SYNC_CYCLE_ADVANCE = 0x3FFFF


def decode_words(words: Iterable[int], mode: TraceMode = TraceMode.EVENT_TIME):
    """De-interleave a raw word stream and decode per-tile command lists.

    Returns a dict ``{(pkt_type, row, col): [TraceCommand, ...]}``.

    Modes 1 and 2 are not yet implemented; passing them raises
    NotImplementedError so callers fail fast rather than silently
    decoding with the wrong opcode table.
    """
    if mode != TraceMode.EVENT_TIME:
        raise NotImplementedError(
            f"trace_decoder: mode {mode!r} not yet implemented "
            "(only EVENT_TIME / mode 0 is supported in this iteration)"
        )

    word_list = list(words)
    # Drop the trailing 0xFEFEFEFE + zero-fill that the encoder appends
    # past the actual payload.
    word_list = trim_trailing_padding(word_list)
    by_tile_words = deinterleave_packets(word_list)

    commands: dict[tuple[int, int, int], list[TraceCommand]] = {}
    for key, payload_words in by_tile_words.items():
        bytes_ = words_to_bytes(payload_words)
        commands[key] = list(mode0.decode(bytes_))
    return commands


def _emit_event(
    out: list[Event],
    *,
    col: int,
    row: int,
    pkt_type: int,
    slot: int,
    name: str,
    ts: int,
) -> None:
    out.append(
        Event(col=col, row=row, pkt_type=pkt_type, slot=slot, name=name, ts=ts)
    )


def rebuild_timeline_mode0(
    commands_per_tile: dict[tuple[int, int, int], list[TraceCommand]],
    slot_names: dict[int, dict[str, list[str]]],
) -> list[Event]:
    """Walk per-tile command lists and emit ``Event`` records.

    ``slot_names[pkt_type][f"{row},{col}"]`` is a list of 8 slot-name
    strings (or empty strings when a slot is unused), keyed the same way
    mlir-aie's parse_mlir_trace_events emits them.

    The timer increments by 1 cycle implicitly per non-Start command,
    then by the command's ``cycles`` value (matches mlir-aie's
    convert_commands_to_json so cycle counts agree exactly).
    """
    events: list[Event] = []
    for (pkt_type, row, col), cmds in commands_per_tile.items():
        timer = 0
        prev_event: EventCmd | None = None
        # Keys are stringified row,col (matching mlir-aie's convention)
        names_for_pt = slot_names.get(pkt_type, {})
        slot_table = names_for_pt.get(f"{row},{col}", [""] * 8)

        for cmd in cmds:
            if isinstance(cmd, StartCmd):
                timer = cmd.timer_value
                prev_event = None
                continue
            if isinstance(cmd, SyncCmd):
                timer += _SYNC_CYCLE_ADVANCE
                continue
            if isinstance(cmd, StopCmd):
                # No timeline contribution; mode 0 doesn't actually emit
                # a Stop opcode in the byte stream (it terminates with
                # padding / end-of-buffer).
                continue
            if isinstance(cmd, EventCmd):
                timer += 1  # implicit per-command increment
                timer += cmd.cycles
                for slot in range(8):
                    if cmd.event_bits & (1 << slot):
                        name = slot_table[slot] if slot < len(slot_table) else ""
                        _emit_event(
                            events,
                            col=col,
                            row=row,
                            pkt_type=pkt_type,
                            slot=slot,
                            name=name,
                            ts=timer,
                        )
                prev_event = cmd
                continue
            if isinstance(cmd, RepeatCmd):
                if prev_event is None:
                    # Repeat with no preceding event is malformed; skip
                    # rather than crash.
                    continue
                cycles = prev_event.cycles
                if cycles == 0:
                    # Zero-cycle repeats just extend the timer linearly.
                    timer += cmd.count
                else:
                    for _ in range(cmd.count):
                        timer += 1
                        timer += cycles
                        for slot in range(8):
                            if prev_event.event_bits & (1 << slot):
                                name = (
                                    slot_table[slot]
                                    if slot < len(slot_table)
                                    else ""
                                )
                                _emit_event(
                                    events,
                                    col=col,
                                    row=row,
                                    pkt_type=pkt_type,
                                    slot=slot,
                                    name=name,
                                    ts=timer,
                                )
                continue

    events.sort(key=lambda e: (e.col, e.row, e.pkt_type, e.ts))
    return events


def parse_trace(
    trace_buffer,
    slot_names: dict[int, dict[str, list[str]]] | None = None,
    mode: TraceMode = TraceMode.EVENT_TIME,
) -> list[Event]:
    """Decode a raw trace buffer into a flat ``Event`` list.

    Mirrors the public surface of mlir-aie's ``parse_trace`` so this
    can drop into the same call sites once validated.

    ``trace_buffer`` may be a numpy uint32 array, a list of ints, or
    bytes (which will be reinterpreted as little-endian uint32).
    ``slot_names`` is an optional 2-level dict mapping ``pkt_type ->
    "row,col" -> list[str]`` of length 8; missing entries default to
    empty strings so downstream tools still see a well-formed record.
    """
    if isinstance(trace_buffer, (bytes, bytearray)):
        words = list(np.frombuffer(bytes(trace_buffer), dtype=np.uint32))
    elif hasattr(trace_buffer, "tolist"):
        words = list(trace_buffer.tolist())
    else:
        words = list(trace_buffer)

    commands = decode_words(words, mode=mode)
    return rebuild_timeline_mode0(commands, slot_names or {})


__all__ = [
    "decode_words",
    "rebuild_timeline_mode0",
    "parse_trace",
    "PacketType",
    "TraceMode",
]
