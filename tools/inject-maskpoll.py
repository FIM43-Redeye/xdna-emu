#!/usr/bin/env python3
"""inject-maskpoll.py -- splice a halt-synchronization MASKPOLL into insts.bin.

The debug_halt_probe (mlir-aie test/npu-xrt/debug_halt_probe) arms either a
PC-event breakpoint (Exp1) or a Debug_Control0 count-step register write (Exp2)
on a compute core, then reads back output_buffer + Core_Status via control-
packet OP_READ.  Without synchronization the OP_READ has no happens-after
against the core terminal state -- it is pure relative latency (spec
docs/superpowers/specs/2026-05-18-debug-halt-design.md, sec 4.2
"Halt-synchronization").  No MLIR dialect op emits a poll into the runtime
sequence (NpuMaskPollOp does not exist), so the robust fix is a post-compile
binary patch: insert a firmware XAIE_IO_MASKPOLL (opcode 0x04) that blocks the
instruction stream until the chosen Core_Status bit is set, positioned
immediately before the first ctrl-in OP_READ push.  EMU and HW then run the
*identical* patched binary (byte-parity).

Two witnesses are supported via --witness {done,halt}:

  --witness halt  (DEFAULT, Exp1)
      Polls Core_Status[16] (DEBUG_HALT, mask/value 0x00010000).
      Used for Exp1 (PC_Event0 breakpoint): the core halts at the trap
      bundle; the poll satisfies when DEBUG_HALT becomes set.
      Byte-for-byte identical to the original behavior; all existing tests
      cover this path.

  --witness done  (Exp2)
      Polls Core_Status[20] (CORE_DONE, mask/value 0x00100000).
      Used for Exp2 (Debug_Control0 count-step): the expected outcome is
      that the count-step is inert and the core runs to completion, latching
      CORE_DONE.  If the DONE-MASKPOLL wedges (core never reached DONE
      because count-step halted it), recover and re-run with --witness halt.

Only the value+mask words ([16..20] and [20..24] in the 28-byte record) differ
between witnesses; reg_off (Core_Status, tile (0,2)) and everything else are
identical.

This tool is deliberately separate from trace-patch-events.py: that tool is
rewrite-only (it never resizes the stream and never touches the header), and
mixing a size-changing splice into it would risk silently desynchronising the
sweep walkers.

Byte-exact MASKPOLL form (28 bytes, little-endian).  Same record shape as
MaskWrite (opcode 0x03), confirmed against:
  - xdna-emu/src/npu/parser.rs NpuOpcode::MaskPoll decode (8 hdr + 8 reg_off
    u64 + 4 value + 4 mask + 4 size)
  - xdna-emu/tools/trace-patch-events.py _STANDARD_OP_SIZES[0x04] == 28
  - spec sec 4.2 grounded byte layout

  [0]      opcode = 0x04
  [1..4]   pad   = 0
  [4..8]   zero word (standard-op second header word) = 0
  [8..12]  reg_off lo = 0x00232004  = (col0<<25)|(row2<<20)|0x32004
                                       (Core_Status, NPU absolute address)
  [12..16] reg_off hi = 0  (reg_off is a u64; high half always 0 here)
  [16..20] value = 0x00010000 (--witness halt, DEBUG_HALT bit 16)
                or 0x00100000 (--witness done, CORE_DONE  bit 20)
  [20..24] mask  = same as value (poll the single witness bit)
  [24..28] size  = 28

insts.bin header (16 bytes; xdna-emu/src/npu/parser.rs:115-123):
  word[0] magic = 0x06030100
  word[1] flags
  word[2] num_ops    -- bumped +1 by this injector
  word[3] total_size -- bumped +28 by this injector

Anchor: the first ctrl-in MaskWrite32 (opcode 0x03) whose reg_off tile-local
offset is 0x1d218 (the ch1 channel-type setup that precedes the first OP_READ
push).  Channel-robust, NOT instruction-count-based: the byte offset of that
MaskWrite varies with the compiled stream, but the 0x1d218 target does not.

Idempotent: refuses (no-op) if a MASKPOLL on reg_off 0x00232004 already exists
in the stream.
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import List, Tuple

# -- Constants derived from spec sec 4.2 + parser.rs (do NOT invent) ----------

_INSTS_HEADER_LEN = 16
_INSTS_MAGIC = 0x06030100

# NPU address packing: (col << 25) | (row << 20) | tile_offset.
# Mirrors xdna-emu/src/npu/executor.rs decode_npu_address and
# trace-patch-events.py _npu_address.
_COL0 = 0
_ROW2 = 2
_CORE_STATUS_TILE_OFF = 0x32004
_MASKPOLL_REG_OFF = (_COL0 << 25) | (_ROW2 << 20) | _CORE_STATUS_TILE_OFF  # 0x00232004

# Witness bit masks (Core_Status register, aie-rt xaiemlgbl_params.h:2347-2365).
# DEBUG_HALT = bit 16 (XAIEMLGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALTED_LSB=16,
#   width=1, line 2363-2365).  Used by Exp1 (PC_Event0 breakpoint).
# CORE_DONE  = bit 20 (XAIEMLGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_LSB=20,
#   width=1, line 2347-2349).  Used by Exp2 (count-step inert path).
_DEBUG_HALT_BIT = 0x0001_0000  # Core_Status[16]
_CORE_DONE_BIT  = 0x0010_0000  # Core_Status[20]

_OPC_WRITE32 = 0x00
_OPC_BLOCKWRITE = 0x01
_OPC_MASKWRITE = 0x03
_OPC_MASKPOLL = 0x04

_MASKPOLL_LEN = 28
_MASKWRITE_LEN = 28

# Anchor: tile-local offset of the ctrl-in MM2S ch1 CTRL register that the
# probe's hand-rolled OP_READ push configures before the first read.
# 0x1d218 = XAIEMLGBL_NOC_MODULE_DMA_MM2S_1_CTRL (ch1; ch0 = 0x1d210, stride
# 0x8) -- see debug_halt_probe/aie.mlir and aie-rt xaiemlgbl_params.h:18775.
# The MaskWrite32's reg_off encodes col0/row0/offset; we match on the
# tile-local offset so partition relocation does not break the anchor.
_ANCHOR_TILE_OFF = 0x1D218

# Fixed-size standard-op byte lengths.  Mirrors trace-patch-events.py
# _STANDARD_OP_SIZES (kept consistent with parser.rs cursor reads).
_STANDARD_OP_SIZES = {
    0x00: 24,  # Write32
    0x03: 28,  # MaskWrite
    0x04: 28,  # MaskPoll
    0x05: 8,   # Noop
    0x06: 16,  # Preempt
    0x08: 16,  # LoadPdi
    0x09: 16,  # LoadPmStart
    0x0A: 16,  # CreateScratchpad
    0x0B: 16,  # UpdateStateTable
    0x0C: 16,  # UpdateReg
    0x0D: 16,  # UpdateScratch
    0xC8: 16,  # LoadPmEndInternal (= 200)
}


def _instruction_length(buf: bytes, off: int) -> int:
    """Total byte length of the instruction starting at `off`.

    Same logic as trace-patch-events.py._instruction_length; kept independent
    so the rewrite-only patcher and this size-changing injector evolve
    separately.
    """
    if off + 8 > len(buf):
        raise ValueError(f"truncated instruction at {off:#x}")
    opcode = buf[off]
    if opcode >= 128:
        size = struct.unpack_from("<I", buf, off + 4)[0]
        return max(size, 8)
    if opcode == _OPC_BLOCKWRITE:
        size = struct.unpack_from("<I", buf, off + 12)[0]
        return max(size, 16)
    length = _STANDARD_OP_SIZES.get(opcode)
    if length is None:
        raise ValueError(
            f"unknown standard opcode {opcode:#04x} at {off:#x} -- "
            f"add to _STANDARD_OP_SIZES or extend parser"
        )
    return length


def _walk(buf: bytes) -> List[Tuple[int, int]]:
    """Yield (record_offset, opcode) for every instruction in stream order."""
    results: List[Tuple[int, int]] = []
    off = _INSTS_HEADER_LEN
    while off < len(buf):
        opcode = buf[off]
        length = _instruction_length(buf, off)
        results.append((off, opcode))
        off += length
    return results


def _maskwrite_maskpoll_reg_off(buf: bytes, off: int) -> int:
    """reg_off (u64, low 32 bits) of a MaskWrite/MaskPoll record at `off`.

    Layout (parser.rs): [0..8] header, [8..16] reg_off u64, [16..20] value,
    [20..24] mask, [24..28] size.
    """
    return struct.unpack_from("<Q", buf, off + 8)[0] & 0xFFFFFFFF


def _tile_offset(reg_off: int) -> int:
    """Tile-local offset = low 20 bits of the NPU address (executor.rs
    TILE_OFFSET_MASK)."""
    return reg_off & 0xFFFFF


def build_maskpoll_bytes(witness: str = "halt") -> bytes:
    """The 28-byte XAIE_IO_MASKPOLL instruction (spec sec 4.2, byte-exact).

    witness="halt" (default): polls Core_Status[16] DEBUG_HALT (0x00010000).
      Byte-for-byte identical to the original behavior -- passing no --witness
      flag produces the same 28-byte record as before this parameterization.
    witness="done": polls Core_Status[20] CORE_DONE (0x00100000).
      Used for Exp2 (count-step inert path).
    """
    if witness == "halt":
        bit = _DEBUG_HALT_BIT
    elif witness == "done":
        bit = _CORE_DONE_BIT
    else:
        raise ValueError(f"unknown witness {witness!r} -- expected 'halt' or 'done'")
    rec = bytearray(_MASKPOLL_LEN)
    rec[0] = _OPC_MASKPOLL  # [0] opcode; [1..4] pad stays 0
    # [4..8] zero word -- already 0
    struct.pack_into("<Q", rec, 8, _MASKPOLL_REG_OFF)  # [8..16] reg_off u64
    struct.pack_into("<I", rec, 16, bit)               # [16..20] value
    struct.pack_into("<I", rec, 20, bit)               # [20..24] mask
    struct.pack_into("<I", rec, 24, _MASKPOLL_LEN)     # [24..28] size
    return bytes(rec)


def _existing_maskpoll_value(buf: bytes):
    """The `value` word ([16..20]) of the first Core_Status MASKPOLL already
    in `buf`, or None if none is present.  The value word IS the witness bit
    (build_maskpoll_bytes sets value == mask == the witness bit), so this
    doubles as the existing-witness probe."""
    for off, opcode in _walk(buf):
        if opcode == _OPC_MASKPOLL:
            if _maskwrite_maskpoll_reg_off(buf, off) == _MASKPOLL_REG_OFF:
                return struct.unpack_from("<I", buf, off + 16)[0]  # [16..20]
    return None


def already_injected(buf: bytes) -> bool:
    """True if a MASKPOLL on reg_off 0x00232004 is already present (any
    witness)."""
    return _existing_maskpoll_value(buf) is not None


def find_anchor_offset(buf: bytes) -> int:
    """Byte offset of the first ctrl-in MaskWrite32 at tile-local 0x1d218.

    Raises ValueError if no such MaskWrite32 exists (caller must STOP and
    report -- never guess an alternate anchor).
    """
    for off, opcode in _walk(buf):
        if opcode == _OPC_MASKWRITE:
            reg_off = _maskwrite_maskpoll_reg_off(buf, off)
            if _tile_offset(reg_off) == _ANCHOR_TILE_OFF:
                return off
    raise ValueError(
        f"no ctrl-in MaskWrite32 at tile-local {_ANCHOR_TILE_OFF:#x} found in "
        f"the instruction stream -- the probe's ch1 OP_READ setup is absent or "
        f"changed.  STOP: do not guess an alternate anchor (spec sec 4.2)."
    )


def inject(buf: bytes, witness: str = "halt") -> Tuple[bytes, bool]:
    """Return (patched_bytes, injected).

    injected == False means a MASKPOLL was already present (idempotent no-op).
    witness: "halt" (default, DEBUG_HALT bit 16) or "done" (CORE_DONE bit 20).
    """
    if len(buf) < _INSTS_HEADER_LEN:
        raise ValueError(f"insts.bin too short ({len(buf)} bytes)")
    magic = struct.unpack_from("<I", buf, 0)[0]
    if magic != _INSTS_MAGIC:
        raise ValueError(
            f"bad insts.bin magic {magic:#010x} (want {_INSTS_MAGIC:#010x})"
        )

    want = _DEBUG_HALT_BIT if witness == "halt" else _CORE_DONE_BIT
    existing = _existing_maskpoll_value(buf)
    if existing is not None:
        if existing == want:
            return bytes(buf), False  # idempotent: same witness already there
        # A MASKPOLL with a DIFFERENT witness is present.  The injector never
        # rewrites an existing poll (a size-changing excise+re-splice would
        # risk desynchronising the stream -- same rationale as the no-op
        # contract).  This is the Task 7 wedge-re-run footgun: switching
        # DEBUG_HALT_PROBE_WITNESS on a warm compile cache.  Fail loud
        # (surfaces as a bridge COMPILE FAIL) rather than silently run the
        # wrong witness and mis-derive G2.
        raise ValueError(
            f"insts.bin already has a Core_Status MASKPOLL with a DIFFERENT "
            f"witness (existing value {existing:#010x}; requested "
            f"{witness!r} = {want:#010x}).  The injector never rewrites an "
            f"existing poll.  Recompile to a fresh insts.bin before switching "
            f"the witness: emu-bridge-test.sh --compile (or edit aie.mlir)."
        )

    anchor = find_anchor_offset(buf)
    poll = build_maskpoll_bytes(witness)

    out = bytearray(buf)
    # Splice the MASKPOLL immediately before the anchor MaskWrite32.
    out[anchor:anchor] = poll

    # Bump header: word[2] num_ops += 1, word[3] total_size += 28.
    num_ops = struct.unpack_from("<I", out, 8)[0]
    total_size = struct.unpack_from("<I", out, 12)[0]
    struct.pack_into("<I", out, 8, num_ops + 1)
    struct.pack_into("<I", out, 12, total_size + _MASKPOLL_LEN)

    return bytes(out), True


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insts", help="path to insts.bin (patched in place)")
    ap.add_argument("-o", "--output",
                    help="write to this path instead of in-place")
    ap.add_argument(
        "--witness", choices=["halt", "done"], default="halt",
        help=(
            "which Core_Status bit to poll (default: halt). "
            "'halt' = bit 16 DEBUG_HALT (0x00010000), used for Exp1 "
            "(PC_Event0 breakpoint); byte-for-byte identical to original "
            "behavior when omitted. "
            "'done' = bit 20 CORE_DONE (0x00100000), used for Exp2 "
            "(count-step inert path -- DONE-MASKPOLL satisfies when the "
            "core runs to completion)."
        ),
    )
    args = ap.parse_args(argv)

    with open(args.insts, "rb") as f:
        data = f.read()

    try:
        patched, injected = inject(data, witness=args.witness)
    except ValueError as e:
        print(f"inject-maskpoll: ERROR: {e}", file=sys.stderr)
        return 2

    dst = args.output or args.insts
    bit = _DEBUG_HALT_BIT if args.witness == "halt" else _CORE_DONE_BIT
    if not injected:
        print(f"inject-maskpoll: MASKPOLL on {_MASKPOLL_REG_OFF:#010x} already "
              f"present in {args.insts} -- no-op (idempotent)", file=sys.stderr)
        if args.output:
            with open(dst, "wb") as f:
                f.write(patched)
        return 0

    with open(dst, "wb") as f:
        f.write(patched)
    print(f"inject-maskpoll: spliced 28-byte MASKPOLL (witness={args.witness}, "
          f"reg_off={_MASKPOLL_REG_OFF:#010x}, value=mask={bit:#010x}) "
          f"before anchor MaskWrite32@tile-local {_ANCHOR_TILE_OFF:#x}; "
          f"header op-count +1, byte-size +{_MASKPOLL_LEN} -> {dst}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
