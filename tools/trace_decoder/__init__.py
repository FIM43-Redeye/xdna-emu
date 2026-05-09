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

* ``aietools/lib/lnx64.o/libevent_trace_decoder.so`` -- read-only
  symbol-table inspection of ``adf::Trace::TraceDecoder`` to map the
  mode-0 / mode-1 frame surface (``processAssertedEvents`` vs
  ``processEventPC``, etc.).  Library is never linked or distributed
  with this code.

* AM020 architecture reference -- mode descriptions for INST_EXEC
  ("branches and ZOL LC").

Modes 0 (EVENT_TIME), 1 (EVENT_PC), and 2 (INST_EXEC) are implemented;
mode 3 is reserved and not documented in any source we have access to.
The mode-1 byte format and mode-2 frame tree were reverse-engineered
from captured traces and confirmed against the dispatch in
``adf::Trace::TraceDecoder::decodePacket`` (mode 1) and
``cardano::Trace::TraceDecoder::initializeExecutionTraceFrameTree``
(mode 2) -- read-only objdump inspection only; the implementations
here are original.
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
