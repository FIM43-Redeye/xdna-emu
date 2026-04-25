# SPDX-License-Identifier: MIT
"""Mode 1 (EVENT_PC) byte-stream decoder.

Mode 1 traces *Program Counter* of every traced event instead of cycle
deltas.  The opcode skeleton is mostly shared with mode 0 (Start, DC,
Repeat0, Repeat1, Sync, Skip), but the per-event Single*/Multiple*
encodings are replaced by a single 4-byte ``EventPC`` opcode.

Bit-level encoding (MSB-first within each byte):

  +-----------+--------------------------+-----------------------------+
  | Opcode    | Pattern                  | Meaning                     |
  +-----------+--------------------------+-----------------------------+
  | Start     | 1111 0XX1 + 7B timer     | Segment-start anchor (mode 1)|
  | EventPC   | 1100 01EE EEEEEERR RRPPPPPP PPPPPPPP | 8b mask + 14b PC|
  | DC        | 1101 11XX + 3B padding   | Don't-care, advance 4 bytes |
  | Repeat0   | 1110 xxxx                | 4-bit repeat count          |
  | Repeat1   | 110110xx + 1B            | 10-bit repeat count         |
  | Sync      | 1111 1111                | Decoder resync              |
  | Skip      | 1111 1110                | Idle filler, advance 1 byte |
  +-----------+--------------------------+-----------------------------+

The Start opcode reuses the mode-0 ``1111 0X00`` family, but with bit 0
(the trace-mode discriminator) set, giving 0xF1 (segment start) and
0xF5 (mid-stream re-anchor).  Bit 2 of the opcode byte selects the
"anchor type" the same way it does in mode 0.

EventPC layout (32-bit word, MSB-first):

  bits 31..26: opcode discriminator (= 0b110001, i.e. 0xC4..0xC7 prefix)
  bits 25..18: 8-bit event mask (which of the 8 trace slots fired)
  bits 17..14: reserved (zero in every observed capture)
  bits 13..0:  14-bit PC value

Source of behavioral knowledge: byte-level skeleton matches the mode-0
opcode table that is documented openly by mlir-aie's ``convert_to_commands``
(Apache 2.0).  The EventPC bit layout was reverse-engineered from
captured traces and confirmed against the dispatch in
``adf::Trace::TraceDecoder::decodePacket`` (read-only inspection of
libevent_trace_decoder.so symbols and an objdump of the same function);
the implementation here is original.
"""

from __future__ import annotations

from typing import Generator

from ..frame import EventCmd, RepeatCmd, StartCmd, SyncCmd, TraceCommand


def decode(byte_stream: list[int]) -> Generator[TraceCommand, None, None]:
    """Yield ``TraceCommand``s parsed from a mode-1 payload byte stream.

    EventPC opcodes are emitted as :class:`EventCmd` with the ``cycles``
    field repurposed to carry the PC.  Downstream tools that distinguish
    modes are responsible for interpreting the field accordingly.
    """
    n = len(byte_stream)
    cursor = 0
    while cursor < n:
        b = byte_stream[cursor]

        # --- Start: 1111 0XX1 + 7 timer bytes --------------------------
        # Mask 0b11110011 == 0b11110001 catches 0xF1 (segment start) and
        # 0xF5 (mid-stream re-anchor).  Mode-0 Start uses the bit-0 = 0
        # variants (0xF0 / 0xF4); the bit-0 discriminator picks the
        # trace-mode the encoder is operating in.
        if (b & 0b11110011) == 0b11110001:
            if cursor + 7 >= n:
                return
            timer_value = 0
            for i in range(7):
                timer_value = (timer_value << 8) | byte_stream[cursor + 1 + i]
            yield StartCmd(timer_value=timer_value)
            cursor += 8
            continue

        # --- Sync: 1111 1111 -------------------------------------------
        if b == 0xFF:
            yield SyncCmd()
            cursor += 1
            continue

        # --- Skip (idle filler): 1111 1110 -----------------------------
        if b == 0xFE:
            cursor += 1
            continue

        # --- DC (don't-care padding): 1101 11XX + 3B -------------------
        if (b & 0b11111100) == 0b11011100:
            cursor += 4
            continue

        # --- Repeat0: 1110 xxxx ----------------------------------------
        if (b & 0b11110000) == 0b11100000:
            yield RepeatCmd(count=b & 0x0F)
            cursor += 1
            continue

        # --- Repeat1: 1101 10xx + 1B -----------------------------------
        if (b & 0b11111100) == 0b11011000:
            if cursor + 1 >= n:
                return
            count = ((b & 0x03) << 8) | byte_stream[cursor + 1]
            yield RepeatCmd(count=count)
            cursor += 2
            continue

        # --- EventPC: 1100 01XX + 3B (8b event mask + 14b PC) ---------
        # Layout (4-byte word, MSB-first):
        #   byte0 = 1100 01ee     (top 6 bits opcode; low 2 bits = mask high 2)
        #   byte1 = eeeeee rr     (mask low 6 bits + 2 reserved bits)
        #   byte2 = rr pppppp     (2 reserved bits + PC high 6 bits)
        #   byte3 = pppppppp      (PC low 8 bits)
        if (b & 0b11111100) == 0b11000100:
            if cursor + 3 >= n:
                return
            b1 = byte_stream[cursor + 1]
            b2 = byte_stream[cursor + 2]
            b3 = byte_stream[cursor + 3]
            mask = ((b & 0b11) << 6) | (b1 >> 2)
            pc = ((b2 & 0b00111111) << 8) | b3
            yield EventCmd(event_bits=mask, cycles=pc)
            cursor += 4
            continue

        # No opcode matched.  Skip and continue rather than raise -- on
        # well-formed traces this branch is unreachable; on malformed
        # captures we want a partial decode rather than a crash.
        cursor += 1
