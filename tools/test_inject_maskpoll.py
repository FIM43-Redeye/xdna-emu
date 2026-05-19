"""Tests for tools/inject-maskpoll.py.

Covers (plan Step 3 TDD requirements):
  - headers correct after inject (op-count +1, byte-size +28)
  - the spliced op decodes back as MaskPoll{reg_off=0x00232004,
    value=0x10000, mask=0x10000} with the byte-exact 28-byte layout
  - the anchor (first ctrl-in MaskWrite32 at tile-local 0x1d218) is located
    and the MASKPOLL is spliced immediately before it
  - idempotent: re-injecting a stream that already has the MASKPOLL is a no-op
  - missing anchor -> ValueError (caller must STOP, not guess)
  - --witness done -> value=mask=0x00100000 (CORE_DONE bit 20)
  - --witness halt (default) -> value=mask=0x00010000 (DEBUG_HALT bit 16)
  - default (no --witness) == halt (byte-for-byte identical)

A representative insts.bin is built in memory: 16-byte header + a Write32 +
a decoy MaskWrite32 (different tile-local offset) + the anchor MaskWrite32 at
tile-local 0x1d218 + a trailing Write32.  This mirrors the probe's ctrl-in
push shape (channel-type MaskWrite then address_patch/write32/sync).

Byte layout reference (parser.rs cursor reads):
  MaskWrite/MaskPoll (28 bytes):
    [0]      opcode
    [1..4]   pad
    [4..8]   zero word
    [8..16]  reg_off u64 LE
    [16..20] value u32 LE
    [20..24] mask u32 LE
    [24..28] size u32 LE (= 28)
  Write32 (24 bytes): [8..16] reg_off u64, [16..20] value, [20..24] size=24
"""

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).parent / "inject-maskpoll.py"
_spec = importlib.util.spec_from_file_location("inject_maskpoll", _TOOL_PATH)
inj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inj)

_INSTS_MAGIC = 0x06030100
_HDR = 16


def _npu_addr(col: int, row: int, off: int) -> int:
    return (col << 25) | (row << 20) | off


def _write32(reg_off: int, value: int) -> bytes:
    rec = bytearray(24)
    rec[0] = 0x00
    struct.pack_into("<Q", rec, 8, reg_off)
    struct.pack_into("<I", rec, 16, value)
    struct.pack_into("<I", rec, 20, 24)
    return bytes(rec)


def _maskwrite(reg_off: int, value: int, mask: int) -> bytes:
    rec = bytearray(28)
    rec[0] = 0x03
    struct.pack_into("<Q", rec, 8, reg_off)
    struct.pack_into("<I", rec, 16, value)
    struct.pack_into("<I", rec, 20, mask)
    struct.pack_into("<I", rec, 24, 28)
    return bytes(rec)


def _header(num_ops: int, total_size: int) -> bytes:
    hdr = bytearray(16)
    struct.pack_into("<I", hdr, 0, _INSTS_MAGIC)
    struct.pack_into("<I", hdr, 4, 0)            # flags
    struct.pack_into("<I", hdr, 8, num_ops)      # word[2]
    struct.pack_into("<I", hdr, 12, total_size)  # word[3]
    return bytes(hdr)


def _build_insts(with_anchor: bool = True) -> bytes:
    """Representative stream.  The anchor is a MaskWrite32 at tile-local
    0x1d218 (ch1 CTRL on shim col0/row0)."""
    ops = []
    # Arming write32 (PC_Event0 on compute (0,2)).
    ops.append(_write32(_npu_addr(0, 2, 0x32020), 0x80000184))
    # Decoy MaskWrite32 at a *different* tile-local offset (must be skipped).
    ops.append(_maskwrite(_npu_addr(0, 0, 0x1D210), 0x400, 0x00000F00))
    if with_anchor:
        # The anchor: ctrl-in ch1 CTRL setup at tile-local 0x1d218.
        ops.append(_maskwrite(_npu_addr(0, 0, 0x1D218), 0x400, 0x00000F00))
    # Trailing write32 (address_patch stand-in).
    ops.append(_write32(_npu_addr(0, 0, 0x1D21C), 0x80000000))

    body = b"".join(ops)
    return _header(len(ops), _HDR + len(body)) + body


def _walk(buf: bytes):
    return inj._walk(buf)


def test_inject_bumps_header():
    data = _build_insts()
    orig_ops = struct.unpack_from("<I", data, 8)[0]
    orig_size = struct.unpack_from("<I", data, 12)[0]

    patched, injected = inj.inject(data)
    assert injected is True

    new_ops = struct.unpack_from("<I", patched, 8)[0]
    new_size = struct.unpack_from("<I", patched, 12)[0]
    assert new_ops == orig_ops + 1, "op-count must bump +1"
    assert new_size == orig_size + 28, "byte-size must bump +28"
    assert len(patched) == len(data) + 28, "stream grows by exactly 28 bytes"


def test_spliced_op_decodes_as_maskpoll():
    data = _build_insts()
    patched, _ = inj.inject(data)

    # Find the MASKPOLL in the patched stream and decode it byte-exactly.
    maskpolls = [
        off for off, opc in _walk(patched) if opc == inj._OPC_MASKPOLL
    ]
    assert len(maskpolls) == 1, "exactly one MASKPOLL must be present"
    off = maskpolls[0]

    opcode = patched[off]
    pad = patched[off + 1:off + 4]
    zero_word = struct.unpack_from("<I", patched, off + 4)[0]
    reg_off = struct.unpack_from("<Q", patched, off + 8)[0]
    value = struct.unpack_from("<I", patched, off + 16)[0]
    mask = struct.unpack_from("<I", patched, off + 20)[0]
    size = struct.unpack_from("<I", patched, off + 24)[0]

    assert opcode == 0x04, "opcode must be MaskPoll (0x04)"
    assert pad == b"\x00\x00\x00", "3 header pad bytes must be zero"
    assert zero_word == 0, "standard-op second header word must be zero"
    assert reg_off == 0x00232004, (
        "reg_off must be (col0<<25)|(row2<<20)|0x32004 = Core_Status"
    )
    assert value == 0x00010000, "value must set DEBUG_HALT bit 16"
    assert mask == 0x00010000, "mask must select only DEBUG_HALT"
    assert size == 28, "size field must be 28"


def test_maskpoll_spliced_immediately_before_anchor():
    data = _build_insts()
    anchor_before = inj.find_anchor_offset(data)
    # In the original stream the anchor is a MaskWrite32 at tile-local 0x1d218.
    assert data[anchor_before] == inj._OPC_MASKWRITE
    assert inj._tile_offset(
        inj._maskwrite_maskpoll_reg_off(data, anchor_before)
    ) == 0x1D218

    patched, _ = inj.inject(data)

    # The MASKPOLL must sit immediately before the anchor MaskWrite32.
    seq = _walk(patched)
    poll_idx = next(i for i, (_, opc) in enumerate(seq)
                    if opc == inj._OPC_MASKPOLL)
    nxt_off, nxt_opc = seq[poll_idx + 1]
    assert nxt_opc == inj._OPC_MASKWRITE, "op after MASKPOLL must be MaskWrite"
    assert inj._tile_offset(
        inj._maskwrite_maskpoll_reg_off(patched, nxt_off)
    ) == 0x1D218, "op after MASKPOLL must be the 0x1d218 anchor"


def test_idempotent_double_inject():
    data = _build_insts()
    once, injected1 = inj.inject(data)
    assert injected1 is True

    twice, injected2 = inj.inject(once)
    assert injected2 is False, "second inject must be a no-op"
    assert twice == once, "double-inject must not change bytes"


def test_idempotent_same_witness_done():
    """Re-injecting the SAME witness on an already-injected binary is a
    no-op (not just for the default halt witness)."""
    data = _build_insts()
    once, injected1 = inj.inject(data, witness="done")
    assert injected1 is True
    twice, injected2 = inj.inject(once, witness="done")
    assert injected2 is False and twice == once


def test_witness_mismatch_raises():
    """Re-injecting a DIFFERENT witness on an already-injected binary must
    fail loud (the Task 7 wedge-re-run footgun: switching
    DEBUG_HALT_PROBE_WITNESS on a warm compile cache).  The injector never
    silently keeps the stale witness."""
    data = _build_insts()
    halted, _ = inj.inject(data, witness="halt")
    with pytest.raises(ValueError, match="DIFFERENT witness"):
        inj.inject(halted, witness="done")
    # And the symmetric direction.
    done, _ = inj.inject(data, witness="done")
    with pytest.raises(ValueError, match="DIFFERENT witness"):
        inj.inject(done, witness="halt")


def test_missing_anchor_raises():
    data = _build_insts(with_anchor=False)
    with pytest.raises(ValueError, match="no ctrl-in MaskWrite32"):
        inj.inject(data)


def test_bad_magic_raises():
    bad = bytearray(_build_insts())
    struct.pack_into("<I", bad, 0, 0xDEADBEEF)
    with pytest.raises(ValueError, match="bad insts.bin magic"):
        inj.inject(bytes(bad))


def test_cli_in_place(tmp_path):
    data = _build_insts()
    p = tmp_path / "insts.bin"
    p.write_bytes(data)

    r = subprocess.run(
        [sys.executable, str(_TOOL_PATH), str(p)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    patched = p.read_bytes()
    assert len(patched) == len(data) + 28
    assert struct.unpack_from("<I", patched, 8)[0] == \
        struct.unpack_from("<I", data, 8)[0] + 1

    # Re-running is idempotent (no-op, exit 0, bytes unchanged).
    r2 = subprocess.run(
        [sys.executable, str(_TOOL_PATH), str(p)],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert "no-op" in r2.stderr
    assert p.read_bytes() == patched


# ---------------------------------------------------------------------------
# Witness parameterization tests (Task 6 -- --witness done|halt)
# ---------------------------------------------------------------------------

def _extract_maskpoll_value_mask(patched: bytes):
    """Return (value, mask) of the single MASKPOLL in the patched stream."""
    maskpolls = [off for off, opc in _walk(patched) if opc == inj._OPC_MASKPOLL]
    assert len(maskpolls) == 1, "exactly one MASKPOLL expected"
    off = maskpolls[0]
    value = struct.unpack_from("<I", patched, off + 16)[0]
    mask  = struct.unpack_from("<I", patched, off + 20)[0]
    return value, mask


def test_witness_halt_default():
    """Default (no witness arg) produces DEBUG_HALT value=mask=0x00010000."""
    data = _build_insts()
    patched, injected = inj.inject(data)  # no witness arg -- defaults to "halt"
    assert injected is True
    value, mask = _extract_maskpoll_value_mask(patched)
    assert value == 0x00010000, f"halt witness value wrong: {value:#010x}"
    assert mask  == 0x00010000, f"halt witness mask  wrong: {mask:#010x}"


def test_witness_halt_explicit():
    """Explicit --witness halt produces the same bytes as the default."""
    data = _build_insts()
    default_patched, _ = inj.inject(data)
    explicit_patched, _ = inj.inject(data, witness="halt")
    assert default_patched == explicit_patched, (
        "explicit witness=halt must be byte-for-byte identical to default"
    )


def test_witness_done():
    """--witness done produces CORE_DONE value=mask=0x00100000."""
    data = _build_insts()
    patched, injected = inj.inject(data, witness="done")
    assert injected is True
    value, mask = _extract_maskpoll_value_mask(patched)
    assert value == 0x00100000, f"done witness value wrong: {value:#010x}"
    assert mask  == 0x00100000, f"done witness mask  wrong: {mask:#010x}"


def test_witness_done_header_bump():
    """--witness done still bumps the header correctly (op-count +1, size +28)."""
    data = _build_insts()
    orig_ops  = struct.unpack_from("<I", data, 8)[0]
    orig_size = struct.unpack_from("<I", data, 12)[0]
    patched, _ = inj.inject(data, witness="done")
    assert struct.unpack_from("<I", patched, 8)[0]  == orig_ops  + 1
    assert struct.unpack_from("<I", patched, 12)[0] == orig_size + 28


def test_witness_done_cli(tmp_path):
    """CLI --witness done writes CORE_DONE into the patched stream."""
    data = _build_insts()
    p = tmp_path / "insts.bin"
    p.write_bytes(data)

    r = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--witness", "done", str(p)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    patched = p.read_bytes()
    value, mask = _extract_maskpoll_value_mask(patched)
    assert value == 0x00100000
    assert mask  == 0x00100000
    # Stderr must mention the witness and the CORE_DONE mask.
    assert "done" in r.stderr
    assert "0x00100000" in r.stderr


def test_witness_invalid():
    """An invalid witness name raises ValueError."""
    data = _build_insts()
    with pytest.raises(ValueError, match="unknown witness"):
        inj.inject(data, witness="bogus")
