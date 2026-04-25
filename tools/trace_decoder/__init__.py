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

Modes 0 (EVENT_TIME) and 1 (EVENT_PC) are implemented; mode 2 (INST_EXEC)
and 3 (reserved) are work-in-progress.  Mode-1 wire format was reverse-
engineered from captured traces and confirmed against the per-word
dispatch in ``adf::Trace::TraceDecoder::decodePacket`` (read-only objdump
inspection); the implementation here is original.
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
from .decode import decode_words, parse_trace

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
    "parse_trace",
]
