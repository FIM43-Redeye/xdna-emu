# SPDX-License-Identifier: MIT
"""Mode 0 (EVENT_TIME) byte-stream decoder.

The opcode table below is documented in the public ``convert_to_commands``
function in mlir-aie's ``aie.utils.trace.utils`` (Apache 2.0); we
re-implement the same encoding here.  Each opcode emits exactly one
``TraceCommand``.

Bit-level encoding (MSB-first within each byte):

  +-----------+------------------------+-----------------------------+
  | Opcode    | Pattern                | Meaning                     |
  +-----------+------------------------+-----------------------------+
  | Start     | 1111 0X00 + 7B timer   | Segment-start anchor        |
  | DC        | 1101 11XX + 3B padding | Don't-care, advance 4 bytes |
  | Single0   | 0xxxxxxx               | 1 event, 4-bit cycle delta  |
  | Single1   | 100xxxxx + 1B          | 1 event, 10-bit cycle delta |
  | Single2   | 101xxxxx + 2B          | 1 event, 18-bit cycle delta |
  | Multiple0 | 1100xxxx + 1B          | Up to 8 events, 4-bit delta |
  | Multiple1 | 110100xx + 2B          | Up to 8 events, 10-bit delta|
  | Multiple2 | 110101xx + 3B          | Up to 8 events, 18-bit delta|
  | Repeat0   | 1110xxxx               | 4-bit repeat count          |
  | Repeat1   | 110110xx + 1B          | 10-bit repeat count         |
  | Skip      | 1111 1110              | Idle filler, advance 1 byte |
  | Sync      | 1111 1111              | Decoder resync (Event_Sync) |
  +-----------+------------------------+-----------------------------+

The Single* / Multiple* split is an encoder optimization: a Single*
opcode embeds *one* event index in its slot field (3 bits, 0..7), while
a Multiple* opcode carries an 8-bit event mask split across two bytes.
Both decode to the same logical EventCmd in our schema -- we expand the
Single* slot index back into the corresponding bitmask bit so the
downstream timeline conversion only has to handle one shape.
"""

from __future__ import annotations

from typing import Generator

from ..frame import EventCmd, RepeatCmd, StartCmd, SyncCmd, TraceCommand


def decode(byte_stream: list[int]) -> Generator[TraceCommand, None, None]:
    """Yield ``TraceCommand``s parsed from a mode-0 payload byte stream.

    The stream is expected to begin with a Start opcode.  Any bytes that
    do not match a known opcode advance the cursor by 1 (this matches
    mlir-aie's behaviour when the encoder inserts unexpected bytes; in
    practice every well-formed stream we have observed is fully
    consumed).
    """
    n = len(byte_stream)
    cursor = 0
    while cursor < n:
        b = byte_stream[cursor]

        # --- Start: 1111 0X00 + 7 timer bytes --------------------------
        # Mask 0b11111011 catches both 0xF0 and 0xF4 (bit 2 differs).
        if (b & 0b11111011) == 0b11110000:
            if cursor + 7 >= n:
                return
            timer_value = 0
            for i in range(7):
                timer_value = (timer_value << 8) | byte_stream[cursor + 1 + i]
            yield StartCmd(timer_value=timer_value)
            cursor += 8
            continue

        # --- Sync (Event_Sync): 1111 1111 ------------------------------
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

        # --- Repeat0: 1110 xxxx ---------------------------------------
        if (b & 0b11110000) == 0b11100000:
            yield RepeatCmd(count=b & 0x0F)
            cursor += 1
            continue

        # --- Repeat1: 1101 10xx + 1B ----------------------------------
        if (b & 0b11111100) == 0b11011000:
            if cursor + 1 >= n:
                return
            count = ((b & 0x03) << 8) | byte_stream[cursor + 1]
            yield RepeatCmd(count=count)
            cursor += 2
            continue

        # --- Multiple2: 1101 01xx + 3B (events + 18-bit cycles) -------
        if (b & 0b11111100) == 0b11010100:
            if cursor + 3 >= n:
                return
            b1 = byte_stream[cursor + 1]
            b2 = byte_stream[cursor + 2]
            b3 = byte_stream[cursor + 3]
            events = ((b & 0b11) << 6) | (b1 >> 2)
            cycles = ((b1 & 0b11) << 16) | (b2 << 8) | b3
            yield EventCmd(event_bits=events, cycles=cycles)
            cursor += 4
            continue

        # --- Multiple1: 1101 00xx + 2B (events + 10-bit cycles) -------
        if (b & 0b11111100) == 0b11010000:
            if cursor + 2 >= n:
                return
            b1 = byte_stream[cursor + 1]
            b2 = byte_stream[cursor + 2]
            events = ((b & 0b11) << 6) | (b1 >> 2)
            cycles = ((b1 & 0b11) << 8) | b2
            yield EventCmd(event_bits=events, cycles=cycles)
            cursor += 3
            continue

        # --- Multiple0: 1100 xxxx + 1B (events + 4-bit cycles) --------
        if (b & 0b11110000) == 0b11000000:
            if cursor + 1 >= n:
                return
            b1 = byte_stream[cursor + 1]
            events = ((b & 0x0F) << 4) | (b1 >> 4)
            cycles = b1 & 0x0F
            yield EventCmd(event_bits=events, cycles=cycles)
            cursor += 2
            continue

        # --- Single2: 101x xxxx + 2B (1 event + 18-bit cycles) --------
        if (b & 0b11100000) == 0b10100000:
            if cursor + 2 >= n:
                return
            slot = (b >> 2) & 0b111
            cycles = (
                ((b & 0b11) << 16)
                | (byte_stream[cursor + 1] << 8)
                | byte_stream[cursor + 2]
            )
            yield EventCmd(event_bits=1 << slot, cycles=cycles)
            cursor += 3
            continue

        # --- Single1: 100x xxxx + 1B (1 event + 10-bit cycles) --------
        if (b & 0b11100000) == 0b10000000:
            if cursor + 1 >= n:
                return
            slot = (b >> 2) & 0b111
            cycles = ((b & 0b11) << 8) | byte_stream[cursor + 1]
            yield EventCmd(event_bits=1 << slot, cycles=cycles)
            cursor += 2
            continue

        # --- Single0: 0xxx xxxx (1 event + 4-bit cycles) --------------
        if (b & 0b10000000) == 0b00000000:
            slot = (b >> 4) & 0b111
            cycles = b & 0x0F
            yield EventCmd(event_bits=1 << slot, cycles=cycles)
            cursor += 1
            continue

        # No opcode matched.  Skip and continue rather than raise -- on
        # well-formed traces this branch is unreachable; on malformed
        # captures we want a partial decode rather than a crash.
        cursor += 1
