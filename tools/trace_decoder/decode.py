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

NOT THE REGRESSION ORACLE.  The emulator-vs-HW trace comparison decodes with
**upstream** ``aie.utils.trace.parse_trace`` (the skip-token model: a held
level is ``Single(cycles=0)`` + ``Repeat`` tokens, closed by the next
``cycles>0`` frame).  This local decoder is a development/inspection aid only;
the held-level *emitter* (``src/device/trace_unit``) is tuned to the upstream
decoder, not to this one.  If the two ever disagree on a held span, upstream is
authoritative -- do not "fix" the emitter to satisfy this module.
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
from .modes import mode0, mode1, mode2
from .packet import deinterleave_packets, trim_trailing_padding, words_to_bytes


# Sync advances the timer by a fixed value documented in mlir-aie's
# convert_commands_to_json with a "Typo in spec" annotation -- the value
# is what the on-tile encoder actually emits, so we match it for
# bit-equivalence.
_SYNC_CYCLE_ADVANCE = 0x3FFFF


_MODE_DECODER = {
    TraceMode.EVENT_TIME: mode0.decode,
    TraceMode.EVENT_PC: mode1.decode,
    TraceMode.INST_EXEC: mode2.decode,
}


def detect_per_tile_modes(
    words: Iterable[int],
) -> dict[tuple[int, int, int], TraceMode | None]:
    """Return ``{(pkt_type, row, col): TraceMode}`` from the Start opcode
    of each per-tile payload.

    A value of ``None`` means the tile has no recognisable Start opcode
    or uses the reserved mode 3 -- both treated as "skip" by
    ``decode_words(mode=None)``.  Useful when downstream code needs to
    know *which* mode each tile was configured for (e.g. to pick a
    timeline-rebuild strategy per tile).
    """
    word_list = trim_trailing_padding(list(words))
    by_tile_words = deinterleave_packets(word_list)
    return {
        key: _detect_tile_mode(words_to_bytes(payload))
        for key, payload in by_tile_words.items()
    }


def _detect_tile_mode(byte_stream: list[int]) -> TraceMode | None:
    """Read the trace-mode discriminator from the first Start opcode.

    Every per-tile payload begins with a Start opcode in one of the
    forms ``1111 0XXM`` where ``M`` (low 2 bits) encodes the trace
    mode (0=EVENT_TIME, 1=EVENT_PC, 2=INST_EXEC, 3=reserved) and bit 2
    is the segment-start vs re-anchor flag.  Skip-filler bytes
    (``0xFE``) may precede the Start opcode and are tolerated.
    Returns ``None`` if no Start is found in the stream -- the caller
    should then fall back to a default or raise.
    """
    for b in byte_stream:
        if b == 0xFE:
            continue
        if (b & 0xF8) == 0xF0:
            try:
                return TraceMode(b & 0x3)
            except ValueError:
                return None
        return None
    return None


def decode_words(
    words: Iterable[int],
    mode: TraceMode | None = TraceMode.EVENT_TIME,
):
    """De-interleave a raw word stream and decode per-tile command lists.

    Returns a dict ``{(pkt_type, row, col): [TraceCommand, ...]}``.

    All three implemented modes (EVENT_TIME, EVENT_PC, INST_EXEC) are
    supported; passing ``mode=None`` enables per-tile auto-detection
    via the Start opcode's low-2-bit discriminator (the encoder always
    starts each tile's stream with that opcode).  Mode 3 is reserved
    and raises ``NotImplementedError`` if encountered.
    """
    if mode is not None and mode not in _MODE_DECODER:
        raise NotImplementedError(
            f"trace_decoder: mode {mode!r} not yet implemented "
            "(EVENT_TIME, EVENT_PC, INST_EXEC supported in this iteration)"
        )

    word_list = list(words)
    # Drop the trailing 0xFEFEFEFE + zero-fill that the encoder appends
    # past the actual payload.
    word_list = trim_trailing_padding(word_list)
    by_tile_words = deinterleave_packets(word_list)

    commands: dict[tuple[int, int, int], list[TraceCommand]] = {}
    for key, payload_words in by_tile_words.items():
        bytes_ = words_to_bytes(payload_words)
        if mode is None:
            tile_mode = _detect_tile_mode(bytes_)
            if tile_mode is None or tile_mode not in _MODE_DECODER:
                # Tile has no recognisable Start (or mode 3): skip
                # rather than poison downstream consumers.
                commands[key] = []
                continue
            per_mode_decode = _MODE_DECODER[tile_mode]
        else:
            per_mode_decode = _MODE_DECODER[mode]
        commands[key] = list(per_mode_decode(bytes_))
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
    soc: int,
) -> None:
    out.append(
        Event(col=col, row=row, pkt_type=pkt_type, slot=slot, name=name, ts=ts, soc=soc)
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
        # cmd_index counts EventCmds (and Repeat iterations) since the
        # most recent Start; soc = ts - cmd_index strips the +1-per-cmd
        # implicit increment that the mlir-aie convention bakes into ts.
        # Reset on every Start (mid-stream re-anchor in modes 1/2).
        cmd_index = 0
        prev_event: EventCmd | None = None
        # Keys are stringified row,col (matching mlir-aie's convention)
        names_for_pt = slot_names.get(pkt_type, {})
        slot_table = names_for_pt.get(f"{row},{col}", [""] * 8)

        for cmd in cmds:
            if isinstance(cmd, StartCmd):
                timer = cmd.timer_value
                cmd_index = 0
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
                cmd_index += 1
                soc = timer - cmd_index
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
                            soc=soc,
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
                    # Zero-cycle repeats just extend the timer linearly:
                    # the decoder advances timer by `count` without firing
                    # the implicit per-EventCmd +1, so cmd_index must not
                    # advance either -- the timer increase in this region
                    # represents real SoC cycles passing with no events,
                    # which is exactly the case where ts == SoC.
                    timer += cmd.count
                else:
                    for _ in range(cmd.count):
                        timer += 1
                        timer += cycles
                        cmd_index += 1
                        soc = timer - cmd_index
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
                                    soc=soc,
                                )
                continue

    events.sort(key=lambda e: (e.col, e.row, e.pkt_type, e.ts))
    return events


_PT_CODE_TO_NAME = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}


def rebuild_perfetto_mode0(
    commands_per_tile: dict[tuple[int, int, int], list[TraceCommand]],
    slot_names: dict[int, dict[str, list[str]]],
) -> list[dict]:
    """Walk per-tile mode-0 commands and emit Chrome-trace B/E pairs.

    Output shape matches mlir-aie's ``parse_trace`` Perfetto output so
    ``--decoder=ours --out-perfetto`` is a drop-in replacement:

      * one ``M`` ``process_name`` event per tile, ``pid`` opaque,
        ``args.name = "<pkt_type_name>(row,col)"``;
      * one ``M`` ``thread_name`` event per (tile, slot), ``tid =
        slot``, ``args.name`` = the slot's event name (or "");
      * a ``B``/``E`` pair per event activation: ``B`` at the cycle the
        event becomes active, ``E`` at the cycle it deactivates (or end
        of trace if it's still asserted at the segment end).

    Mode-0 semantics: each EventCmd is a snapshot of the *currently
    asserted* slot mask after a cycle delta, so we diff against the
    previous mask to find activations / deactivations.  Repeat extends
    the most-recent transition pattern by ``count``; a zero-cycle prior
    just lengthens the timer linearly without changing the mask, while
    a non-zero prior replays the pattern N times.  Sync nudges the
    timer by the documented spec-typo constant; Start anchors it.
    """
    out: list[dict] = []
    pid_counter = 0
    for (pkt_type, row, col), cmds in commands_per_tile.items():
        pid = pid_counter
        pid_counter += 1
        pt_name = _PT_CODE_TO_NAME.get(pkt_type, f"pt{pkt_type}")
        out.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "args": {"name": f"{pt_name}({row},{col})"},
            }
        )
        slot_table = slot_names.get(pkt_type, {}).get(f"{row},{col}", [""] * 8)
        for slot in range(8):
            name = slot_table[slot] if slot < len(slot_table) else ""
            out.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": slot,
                    "args": {"name": name},
                }
            )

        timer = 0
        active: dict[int, int] = {}  # slot -> activation_ts
        prev_mask = 0
        prev_event: EventCmd | None = None

        def _emit_be(new_mask: int, ts: int) -> None:
            nonlocal active
            for slot in range(8):
                bit = 1 << slot
                was_on = bool(prev_mask & bit)
                now_on = bool(new_mask & bit)
                if now_on and not was_on:
                    out.append(
                        {
                            "ph": "B",
                            "name": slot_table[slot] if slot < len(slot_table) else "",
                            "pid": pid,
                            "tid": slot,
                            "ts": ts,
                        }
                    )
                    active[slot] = ts
                elif was_on and not now_on:
                    out.append(
                        {
                            "ph": "E",
                            "name": slot_table[slot] if slot < len(slot_table) else "",
                            "pid": pid,
                            "tid": slot,
                            "ts": ts,
                        }
                    )
                    active.pop(slot, None)

        for cmd in cmds:
            if isinstance(cmd, StartCmd):
                # Close out any still-active events at the previous
                # timer, then anchor.
                _emit_be(0, timer)
                prev_mask = 0
                timer = cmd.timer_value
                prev_event = None
                continue
            if isinstance(cmd, SyncCmd):
                timer += _SYNC_CYCLE_ADVANCE
                continue
            if isinstance(cmd, StopCmd):
                continue
            if isinstance(cmd, EventCmd):
                timer += 1 + cmd.cycles
                _emit_be(cmd.event_bits, timer)
                prev_mask = cmd.event_bits
                prev_event = cmd
                continue
            if isinstance(cmd, RepeatCmd):
                if prev_event is None:
                    continue
                if prev_event.cycles == 0:
                    # Linear timer extension; mask is unchanged across
                    # the repeat, so no transitions to emit.
                    timer += cmd.count
                else:
                    for _ in range(cmd.count):
                        timer += 1 + prev_event.cycles
                        # The pattern that gets repeated is the
                        # previous transition (mask stays constant
                        # across replays in the mlir-aie algorithm),
                        # so no B/E during the body -- only the timer
                        # advances.  Active events get their dur
                        # extended naturally by the next transition's
                        # E timestamp.
                continue

        # Close out any events still asserted at end-of-segment.
        _emit_be(0, timer)

    return out


def rebuild_timeline_mode1(
    commands_per_tile: dict[tuple[int, int, int], list[TraceCommand]],
    slot_names: dict[int, dict[str, list[str]]],
) -> list[Event]:
    """Walk per-tile mode-1 command lists and emit ``Event`` records.

    EventCmd's ``cycles`` field carries the program counter rather than
    a cycle delta; we surface it as the Event's ``ts`` so existing
    consumers can still sort/group on a single scalar.  Sync and Start
    have no PC contribution and are skipped.  Repeat expands the most
    recent fire ``count`` times at the same PC -- in mode-1 traces this
    corresponds to the same event firing for N cycles in a tight loop
    that doesn't change the PC.
    """
    events: list[Event] = []
    for (pkt_type, row, col), cmds in commands_per_tile.items():
        prev_event: EventCmd | None = None
        names_for_pt = slot_names.get(pkt_type, {})
        slot_table = names_for_pt.get(f"{row},{col}", [""] * 8)

        def _emit(ev: EventCmd) -> None:
            for slot in range(8):
                if ev.event_bits & (1 << slot):
                    name = slot_table[slot] if slot < len(slot_table) else ""
                    _emit_event(
                        events,
                        col=col,
                        row=row,
                        pkt_type=pkt_type,
                        slot=slot,
                        name=name,
                        ts=ev.cycles,  # mode 1: ts carries PC
                        soc=ev.cycles,  # mode 1 has no cycle-domain soc; mirror ts
                    )

        for cmd in cmds:
            if isinstance(cmd, EventCmd):
                _emit(cmd)
                prev_event = cmd
                continue
            if isinstance(cmd, RepeatCmd):
                if prev_event is None:
                    continue
                for _ in range(cmd.count):
                    _emit(prev_event)
                continue
            # Start, Sync, Stop carry no PC contribution in mode 1.

    events.sort(key=lambda e: (e.col, e.row, e.pkt_type, e.ts, e.slot))
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
    if mode == TraceMode.EVENT_PC:
        return rebuild_timeline_mode1(commands, slot_names or {})
    return rebuild_timeline_mode0(commands, slot_names or {})


def parse_trace_auto(
    trace_buffer,
    slot_names: dict[int, dict[str, list[str]]] | None = None,
) -> list[Event]:
    """Decode a mixed-mode trace buffer.

    Real captures interleave packets from several tiles, each programmed
    with whatever ``Trace_Control0.MODE`` makes sense for its module:

    * compute cores typically run in ``EVENT_PC`` (mode 1) so each event
      carries the PC of the firing instruction;
    * shim PL and memtile modules have no PC and run in ``EVENT_TIME``
      (mode 0) so each event carries a cycle delta;
    * mode 2 (``INST_EXEC``) emits a different command vocabulary (atom
      / New_PC / LC) that doesn't fit the timeline ``Event`` schema
      and is skipped here -- callers that need it should use
      ``decode_words(mode=None)`` and dispatch to mode-2 rebuilders
      directly.

    ``parse_trace_auto`` reads each per-tile payload's Start opcode
    discriminator, decodes that tile with its own mode, and rebuilds
    its timeline with the matching ``rebuild_timeline_mode{0,1}``.
    The returned ``Event`` list combines both mode-0 (``ts`` is a cycle)
    and mode-1 (``ts`` is the PC of the firing instruction) records;
    consumers disambiguate via ``pkt_type`` (cores in mode 1, shim/
    memtile in mode 0).

    Use ``parse_trace`` (single-mode) for fixtures whose configuration
    is known up-front and uniform; use ``parse_trace_auto`` for whole
    BOs from real or emulator captures where modes are mixed.
    """
    if isinstance(trace_buffer, (bytes, bytearray)):
        words = list(np.frombuffer(bytes(trace_buffer), dtype=np.uint32))
    elif hasattr(trace_buffer, "tolist"):
        words = list(trace_buffer.tolist())
    else:
        words = list(trace_buffer)

    word_list = trim_trailing_padding(words)
    by_tile_words = deinterleave_packets(word_list)
    names = slot_names or {}

    all_events: list[Event] = []
    for key, payload_words in by_tile_words.items():
        bytes_ = words_to_bytes(payload_words)
        tile_mode = _detect_tile_mode(bytes_)
        if tile_mode is None or tile_mode not in _MODE_DECODER:
            # No recognisable Start (or reserved mode 3): skip this
            # tile.  Modes 0/1 are timeline-shaped; mode 2 has its own
            # output shape (atoms, New_PC, LC) so it gets the same
            # treatment as the unrecognised cases here -- callers who
            # need mode-2 should drive it through decode_words/decode
            # directly instead of expecting flat Event records.
            continue
        if tile_mode == TraceMode.INST_EXEC:
            continue
        commands = list(_MODE_DECODER[tile_mode](bytes_))
        if tile_mode == TraceMode.EVENT_PC:
            tile_events = rebuild_timeline_mode1({key: commands}, names)
        else:
            tile_events = rebuild_timeline_mode0({key: commands}, names)
        all_events.extend(tile_events)

    all_events.sort(key=lambda e: (e.col, e.row, e.pkt_type, e.ts, e.slot))
    return all_events


__all__ = [
    "decode_words",
    "rebuild_timeline_mode0",
    "rebuild_timeline_mode1",
    "parse_trace",
    "parse_trace_auto",
    "PacketType",
    "TraceMode",
]
