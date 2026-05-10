# SPDX-License-Identifier: MIT
"""Frame and command schema for the trace decoder.

All trace-mode decoders (EVENT_TIME, EVENT_PC, INST_EXEC) emit a
sequence of ``TraceCommand`` values.  Higher-level conversions
(timeline, Perfetto, scalar cycles) are computed on top of this stream.

Naming follows the public ``adf::Trace`` API surface so the
correspondence to the hardware reference is direct:

* ``StartCmd``  <-> ``processStart``        (timer anchor for the segment)
* ``StopCmd``   <-> ``processStop``         (segment terminator)
* ``SyncCmd``   <-> ``processSync``         (decoder resync, no payload)
* ``RepeatCmd`` <-> ``processRepeat``       (run-length compression)
* ``EventCmd``  <-> ``processAssertedEvents`` (mode 0)
                 / ``processEventPC``        (mode 1)

EventCmd carries an 8-bit event mask plus a single quantity that means
*cycles since previous event* in mode 0 and *PC value* in mode 1; the
TraceMode field on the parent segment disambiguates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TraceMode(IntEnum):
    """Trace control register MODE field values.

    Values are stable across AIE2/AIE2P/AIE2PS and match the public
    ``XAie_TraceMode`` enum in aie-rt's ``xaie_trace.h`` (MIT).
    """

    EVENT_TIME = 0
    EVENT_PC = 1
    INST_EXEC = 2
    RESERVED = 3


class PacketType(IntEnum):
    """Stream-switch packet type field for trace traffic.

    Values match mlir-aie's enumeration (PacketType in
    ``aie.utils.trace.events``).  The bit layout in the packet header
    is documented in ``packet.py``.
    """

    CORE = 0
    MEM = 1
    SHIMTILE = 2
    MEMTILE = 3


@dataclass(frozen=True)
class TraceCommand:
    """Marker base for all decoded commands."""


@dataclass(frozen=True)
class StartCmd(TraceCommand):
    """Segment-start anchor.  Carries the 56-bit timer value at trace start.

    All subsequent timestamps in mode 0 are deltas from this anchor.
    """

    timer_value: int


@dataclass(frozen=True)
class StopCmd(TraceCommand):
    """Segment-end marker.  No payload required by the byte stream itself;
    higher-level decoders may attach a timestamp from a preceding
    EventCmd."""


@dataclass(frozen=True)
class SyncCmd(TraceCommand):
    """Decoder resync marker (mlir-aie historically calls this Event_Sync).

    Carries no payload but advances the timer by a fixed amount (mlir-aie
    uses 0x3FFFF and notes the value as a spec typo retained for
    bit-compatibility).  The cycles advance is applied during timeline
    conversion, not stored on the command itself.
    """


@dataclass(frozen=True)
class RepeatCmd(TraceCommand):
    """Run-length compression: repeat the most recent EventCmd N times.

    The semantics interact with the prior event's cycles field: if the
    prior event had cycles=0 (i.e. simultaneous repeats) the timer
    advances by N; otherwise the prior pattern (deactivate + delta +
    activate) is replayed N times.
    """

    count: int


@dataclass(frozen=True)
class EventCmd(TraceCommand):
    """One or more events firing.

    ``event_bits``: 8-bit mask, bit *i* set <-> trace slot *i* fired.
    ``cycles``: cycle delta from the previous event (mode 0).  In modes
    that anchor on PC instead of cycles this field is reused for the PC
    value (the parent segment's TraceMode disambiguates).
    """

    event_bits: int
    cycles: int


@dataclass(frozen=True)
class Event:
    """Flat per-tile event record produced by timeline conversion.

    Mirrors the schema emitted by ``parse-trace.py --out-events`` so
    downstream consumers (trace-compare, Perfetto export) work
    unchanged.

    ``ts`` matches the mlir-aie ``convert_commands_to_json`` convention:
    each event in a tile's encoded stream advances the timer by
    ``1 + cmd.cycles``, where the ``1`` is the implicit per-command
    increment.  This means ``ts`` is *not* the SoC cycle of the event;
    it inflates by +1 cyc per preceding event in the same tile stream.

    ``soc`` removes the implicit per-event drift: ``soc = ts - cmd_index``
    where ``cmd_index`` is the 1-based count of EventCmds (not slots --
    a Multiple frame with N slots advances ``cmd_index`` by 1) preceding
    this event in the tile.  For sparse-event tiles the two are nearly
    equal; for dense-event tiles (shim with STREAM_STARVATION every
    cycle) they diverge by hundreds of cycles.  Use ``soc`` for any
    cross-tile timing comparison; ``ts`` is for compatibility with the
    upstream mlir-aie tooling.

    See `docs/superpowers/findings/2026-05-10-trace-decoder-event-density-drift.md`.
    """

    col: int
    row: int
    pkt_type: int
    slot: int
    name: str
    ts: int
    soc: int
