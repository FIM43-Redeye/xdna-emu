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

* ``aietools/include/drivers/aiengine/xaiengine/xaie_trace.h`` (MIT) and
  ``aietools/include/adf/adf_api/BaseImpl.h`` (MIT) -- public enum
  definitions for trace mode (EVENT_TIME / EVENT_PC / INST_EXEC) and
  module type (core / memory / shim).

* AM020 architecture reference -- mode descriptions for INST_EXEC
  ("branches and ZOL LC").

Per-mode derivation basis (these differ, and the difference matters):

* Mode 0 (EVENT_TIME) re-implements the openly-documented mlir-aie
  table above.

* Mode 1 (EVENT_PC) was derived black-box: mode-0 and mode-1 captures
  of an identical kernel were diffed against that table, and every field
  falls out of the observed bytes (see ``modes/mode1.py`` and the mode-1
  derivation note under ``docs/``).  It depends on no vendor binary and
  is clean-room -- suitable for upstream contribution.

* Mode 2 (INST_EXEC): the frame tree was recovered by inspecting
  ``cardano::Trace::TraceDecoder::initializeExecutionTraceFrameTree`` in
  ``libxv_trace_decoder_opt.so`` (read-only objdump; never linked or
  shipped).  This is disassembly-derived, NOT clean-room, and is
  retained for the emulator only -- it must not be contributed upstream
  or shipped publicly without an independent black-box re-derivation.

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
    decode_words,
    detect_per_tile_modes,
    parse_trace,
    parse_trace_auto,
    rebuild_timeline_mode0,
    rebuild_timeline_mode1,
    rebuild_perfetto_mode0,
)

__all__ = [
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
