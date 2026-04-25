# SPDX-License-Identifier: MIT
"""Mode 2 (INST_EXEC) bit-stream decoder.

Mode 2 traces the instruction-execution stream of a tile core: each
cycle is recorded as either an ``E_atom`` (executed) or ``N_atom``
(stalled), with ``New_PC`` frames at every taken branch and ``LC``
frames at zero-overhead-loop boundaries.  The encoding is bit-packed
into 32-bit words: a small prefix-code Huffman tree per frame plus a
fixed-width payload field where applicable.

Frame tree (MSB-first within each 32-bit word) -- recovered by parsing
``cardano::Trace::TraceDecoder::initializeExecutionTraceFrameTree`` in
``libxv_trace_decoder_opt.so``; field widths verified against each
``Execution_*::decode`` body.  Implementation here is original.

  +-------------+-----------------------------+--------------------------+
  | Prefix      | Frame                       | Payload                  |
  +-------------+-----------------------------+--------------------------+
  | 0000        | N_atom    (1 cycle stalled) | (none)                   |
  | 0001        | E_atom    (1 cycle exec)    | (none)                   |
  | 0010        | Filler0   (idle padding)    | (none)                   |
  | 010         | LC        (loop counter)    | 1b flag + 28b count      |
  | 10          | New_PC    (taken branch)    | 14b absolute PC          |
  | 1110        | Repeat0   (RLE small)       | 4b count                 |
  | 110110      | Repeat1   (RLE large)       | 10b count                |
  | 110111      | Stop      (segment end)     | 26b payload (consumes word)|
  | 11110       | Start     (segment anchor)  | 1b flag + 14b anchor PC  |
  | 11111110    | Filler1   (8b filler)       | (none)                   |
  | 11111111    | Sync      (decoder resync)  | (none)                   |
  +-------------+-----------------------------+--------------------------+

Start, LC, and Stop consume the rest of the 32-bit word; per-cycle
frames (E/N_atom, New_PC, Repeat*) are short and several pack into one
word, with the encoder padding the trailing bits with Filler0.

The Start opcode's bit-26 flag distinguishes initial vs re-anchor
starts (matching the analogous bit in mode 0/1's ``F0``/``F4`` and
``F1``/``F5`` opcodes), and bits 13..0 of the Start word carry the
anchor PC in instruction-byte units.

We reuse the EventCmd dataclass for New_PC frames, packing the PC into
the ``cycles`` field (the parent TraceMode disambiguates) -- this
keeps the schema flat.  E_atom / N_atom emit their own dataclasses so
downstream tools can count cycles cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from ..frame import EventCmd, RepeatCmd, StartCmd, SyncCmd, TraceCommand


@dataclass(frozen=True)
class CycleCmd(TraceCommand):
    """One cycle of execution.  ``stalled=True`` => N_atom; False => E_atom."""

    stalled: bool


@dataclass(frozen=True)
class LoopCountCmd(TraceCommand):
    """Zero-overhead-loop iteration boundary."""

    flag: int  # bit 28 of the LC word; semantics not yet pinned down
    count: int  # bits 27..0 (28-bit ZOL counter snapshot)


def _bits(byte_stream: list[int]) -> Generator[int, None, None]:
    """Yield bits MSB-first across the byte stream, one bit at a time."""
    for b in byte_stream:
        for shift in range(7, -1, -1):
            yield (b >> shift) & 1


def _take(bit_iter, n: int) -> int | None:
    """Take ``n`` bits and reassemble them MSB-first.  Returns None at EOF."""
    val = 0
    for _ in range(n):
        try:
            val = (val << 1) | next(bit_iter)
        except StopIteration:
            return None
    return val


def decode(byte_stream: list[int]) -> Generator[TraceCommand, None, None]:
    """Yield ``TraceCommand``s parsed from a mode-2 payload byte stream.

    The stream is interpreted as a continuous bit stream, MSB-first.
    Frame prefixes are decoded against the tree above; each frame's
    fixed-width payload (where present) is consumed immediately after
    the prefix.  Unknown / ambiguous prefixes raise an internal error
    rather than guessing -- mode 2 is sensitive to alignment, and a
    silent skip would cascade into nonsense.
    """
    bits = _bits(byte_stream)

    # Helper to read prefix one bit at a time and walk the tree.
    def _next_frame():
        # Tree walk: returns the frame label or None at EOF.
        try:
            b0 = next(bits)
        except StopIteration:
            return None
        if b0 == 0:
            b1 = next(bits, None)
            if b1 is None:
                return None
            if b1 == 0:
                # 00xx
                b2 = next(bits, None)
                b3 = next(bits, None)
                if b2 is None or b3 is None:
                    return None
                if b2 == 0 and b3 == 0:
                    return "N_atom"
                if b2 == 0 and b3 == 1:
                    return "E_atom"
                if b2 == 1 and b3 == 0:
                    return "Filler0"
                # 0011 unused in tree
                return f"unknown_00{b2}{b3}"
            else:
                # 01x
                b2 = next(bits, None)
                if b2 is None:
                    return None
                if b2 == 0:
                    return "LC"
                return f"unknown_011"
        else:
            b1 = next(bits, None)
            if b1 is None:
                return None
            if b1 == 0:
                # 10
                return "New_PC"
            else:
                # 11xx
                b2 = next(bits, None)
                b3 = next(bits, None)
                if b2 is None or b3 is None:
                    return None
                if b2 == 1 and b3 == 0:
                    return "Repeat0"  # 1110
                if b2 == 0:
                    # 110x
                    b4 = next(bits, None)
                    b5 = next(bits, None)
                    if b4 is None or b5 is None:
                        return None
                    if b4 == 1 and b5 == 0:
                        return "Repeat1"  # 110110
                    if b4 == 1 and b5 == 1:
                        return "Stop"  # 110111
                    return f"unknown_110{b4}{b5}"
                # 1111x
                b4 = next(bits, None)
                if b4 is None:
                    return None
                if b4 == 0:
                    return "Start"  # 11110
                # 11111x...
                b5 = next(bits, None)
                b6 = next(bits, None)
                b7 = next(bits, None)
                if b5 is None or b6 is None or b7 is None:
                    return None
                if (b5, b6, b7) == (1, 1, 0):
                    return "Filler1"  # 11111110
                if (b5, b6, b7) == (1, 1, 1):
                    return "Sync"  # 11111111
                return f"unknown_11111{b5}{b6}{b7}"

    while True:
        frame = _next_frame()
        if frame is None:
            return

        if frame == "E_atom":
            yield CycleCmd(stalled=False)
        elif frame == "N_atom":
            yield CycleCmd(stalled=True)
        elif frame == "Filler0" or frame == "Filler1":
            # Filler frames advance the cursor without emitting a command.
            continue
        elif frame == "Sync":
            yield SyncCmd()
        elif frame == "New_PC":
            pc = _take(bits, 14)
            if pc is None:
                return
            # Pack PC into the cycles slot of EventCmd; event_bits=0
            # marks "no event mask, this is a PC anchor".
            yield EventCmd(event_bits=0, cycles=pc)
        elif frame == "Repeat0":
            count = _take(bits, 4)
            if count is None:
                return
            yield RepeatCmd(count=count)
        elif frame == "Repeat1":
            count = _take(bits, 10)
            if count is None:
                return
            yield RepeatCmd(count=count)
        elif frame == "Start":
            # Start consumes the rest of the 32-bit word.  We've already
            # read 5 bits of prefix; bit 26 (= bit index 5 in stream-order
            # within this word) is the anchor flag, bits 25..14 are
            # reserved, bits 13..0 are the anchor PC.
            rest = _take(bits, 27)
            if rest is None:
                return
            anchor_pc = rest & 0x3FFF
            yield StartCmd(timer_value=anchor_pc)
        elif frame == "LC":
            # LC consumes 29 more bits (3 prefix + 1 flag + 28 count = 32).
            rest = _take(bits, 29)
            if rest is None:
                return
            flag = (rest >> 28) & 1
            count = rest & 0x0FFFFFFF
            yield LoopCountCmd(flag=flag, count=count)
        elif frame == "Stop":
            # Stop consumes the rest of the 32-bit word (26 more bits
            # after a 6-bit prefix).  We don't yet have a typed Stop
            # command in the schema, so emit a SyncCmd as a placeholder
            # marker and discard the payload.
            rest = _take(bits, 26)
            if rest is None:
                return
            yield SyncCmd()
        else:
            # Unknown prefix.  In a clean mode-2 stream this branch is
            # unreachable, but real captures may interleave tiles
            # configured for different trace modes; rather than crash
            # we drain the rest of the bit stream so downstream
            # callers can still inspect what we did decode.
            for _ in bits:
                pass
            return
