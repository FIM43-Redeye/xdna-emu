# SPDX-License-Identifier: MIT
"""xdna-emu trace decoder.

Independent re-implementation of the AIE2 on-tile trace decoder.  This
package replaces the runtime dependency on mlir-aie's ``parse_trace`` and
on aietools' shared libraries with a portable, MIT-licensed decoder we
can ship and evolve.

Reference sources used to design this module (read-only -- no code or
data copied):

* ``mlir-aie/python/utils/trace/{parse,utils}.py`` (Apache 2.0) -- the
  mode-0 (EVENT_TIME) byte-level opcode format is fully specified there.
  We re-implement the same algorithm, validate bit-perfect agreement on
  fixtures, and treat that match as the correctness contract.

* ``aie-rt/driver/src/trace/xaie_trace.h`` (open aie-rt) -- the
  ``XAie_TraceMode`` enum (EVENT_TIME / EVENT_PC / INST_EXEC) and the
  module types (core / memory / shim) it is used with.

Per-mode derivation basis (these differ, and the difference matters):

* Mode 0 (EVENT_TIME) re-implements the openly-documented mlir-aie
  table above.

* Mode 1 (EVENT_PC) was derived black-box: mode-0 and mode-1 captures
  of an identical kernel were diffed against that table, and every field
  falls out of the observed bytes (see ``modes/mode1.py`` and the mode-1
  derivation note under ``docs/``).  It depends on no vendor binary and
  is clean-room -- suitable for upstream contribution.

* Mode 2 (INST_EXEC) is not modeled in this tree.  An earlier decoder
  was derived from a vendor binary rather than from observation, so it
  was removed; mode 2 returns once it is re-derived black-box from
  silicon captures.  Decoding it raises ``NotImplementedError``.

Mode 3 is reserved and undocumented.  All implementations here are
original.
"""

from .frame import (
    StartCmd,
    StopCmd,
    SyncCmd,
    RepeatCmd,
    EventCmd,
    Event,
    TraceMode,
    PacketType,
)
from .packet import StreamPacketHeader, parse_packet_header, deinterleave_packets
from .decode import (
    MODE2_NOT_MODELED,
    decode_words,
    detect_per_tile_modes,
    parse_trace,
    parse_trace_auto,
    rebuild_timeline_mode0,
    rebuild_timeline_mode1,
    rebuild_perfetto_mode0,
)

__all__ = [
    "MODE2_NOT_MODELED",
    "StartCmd",
    "StopCmd",
    "SyncCmd",
    "RepeatCmd",
    "EventCmd",
    "Event",
    "TraceMode",
    "PacketType",
    "StreamPacketHeader",
    "parse_packet_header",
    "deinterleave_packets",
    "decode_words",
    "detect_per_tile_modes",
    "parse_trace",
    "parse_trace_auto",
    "rebuild_timeline_mode0",
    "rebuild_timeline_mode1",
    "rebuild_perfetto_mode0",
]
