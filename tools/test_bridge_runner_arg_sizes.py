#!/usr/bin/env python3
"""Regression guard for bridge-trace-runner's toolchain-derived output-BO sizing.

Mirrors the C++ `discover_arg_sizes_from_insts` in
`bridge-runner/bridge-trace-runner.cpp`: walk insts.bin, and for each DDR-patched
data buffer recover its DMA transfer length from the shim-DMA BD config
(BLOCKWRITE payload[0] = Buffer_Length in 32-bit words) that immediately
precedes the DdrPatch carrying that buffer's arg_idx. required_bytes = words * 4.

Why this exists: XRT kernel-arg metadata reports an NPU buffer arg as its 8-byte
pointer size, not its extent. Output-only buffers (no --input to infer from) were
allocated at 8 bytes, so the kernel's output DMA overran them -- intermittent
IOMMU IO_PAGE_FAULT / silent DDR corruption (root-caused 2026-07-01 on the SP-5b
R1 gate: output BO 8 B, drain 8192 B). This test guards the extraction algorithm
against both regressions in our parser and drift in mlir-aie's NPU lowering (the
BLOCKWRITE-then-DdrPatch emission order + BD-word-0 = length layout).
"""
import os
import struct
import subprocess

import pytest

INSTS_MAGIC = 0x06030100
OP_WRITE32 = 0x00
OP_BLOCKWRITE = 0x01
OP_MASKWRITE = 0x03
OP_DDR_PATCH = 0x81
# mlir-aie include/aie/Runtime/TxnEncoding.h: patches of args at index >= 5 may carry the DDR
# aperture offset folded into arg_plus.
DDR_AIE_ADDR_OFFSET = 0x80000000
NUM_FIRMWARE_TRANSLATED_ARGS = 5


def discover_arg_sizes(data: bytes) -> dict:
    """Return {data_buffer_index: required_bytes}. Mirror of the C++ walker."""
    if len(data) < 16 or struct.unpack_from("<I", data, 0)[0] != INSTS_MAGIC:
        return {}
    end = min(len(data), struct.unpack_from("<I", data, 12)[0])
    off = 16
    sizes = {}
    last_len_words = 0
    have_len = False
    while off + 4 <= end:
        opcode = data[off]
        if opcode == OP_WRITE32:
            op_size = 24
        elif opcode == OP_BLOCKWRITE:
            if off + 16 > end:
                break
            op_size = struct.unpack_from("<I", data, off + 12)[0]
            if op_size < 20 or off + 20 > end:
                have_len = False
            else:
                last_len_words = struct.unpack_from("<I", data, off + 16)[0]
                have_len = True
        elif opcode == OP_MASKWRITE:
            op_size = 28
        elif opcode == OP_DDR_PATCH:
            op_size = 48
            if off + 48 <= end:
                arg_idx = data[off + 8 + 24]
                arg_plus = struct.unpack_from("<Q", data, off + 40)[0]
                if arg_idx >= NUM_FIRMWARE_TRANSLATED_ARGS and arg_plus >= DDR_AIE_ADDR_OFFSET:
                    arg_plus -= DDR_AIE_ADDR_OFFSET
                if have_len and last_len_words > 0:
                    need = arg_plus + last_len_words * 4
                    sizes[arg_idx] = max(sizes.get(arg_idx, 0), need)
            have_len = False
        else:
            break
        off += op_size
    return sizes


def _blockwrite(len_words: int) -> bytes:
    # 16-byte header [opcode, _, _, size=48] + 32-byte payload; payload[0]=len.
    hdr = struct.pack("<IIII", OP_BLOCKWRITE, 0, 0, 48)
    payload = struct.pack("<I", len_words) + b"\x00" * 28
    return hdr + payload


def _ddr_patch(arg_idx: int, arg_plus: int = 0) -> bytes:
    op = bytearray(48)
    op[0] = OP_DDR_PATCH
    op[8 + 24] = arg_idx  # arg_idx at payload[24] (word 8)
    struct.pack_into("<Q", op, 40, arg_plus)  # words 10..11: byte offset into the buffer
    return bytes(op)


def _synthetic_insts(pairs) -> bytes:
    """pairs: (arg_idx, length in words) or (arg_idx, length in words, byte offset)."""
    body = b"".join(_blockwrite(p[1]) + _ddr_patch(p[0], p[2] if len(p) > 2 else 0) for p in pairs)
    total = 16 + len(body)
    header = struct.pack("<IIII", INSTS_MAGIC, 0, 0, total)
    return header + body


def test_synthetic_two_buffers():
    # arg 0 = output (2048 words = 8192 B), arg 1 = trace (4096 words = 16384 B),
    # matching sp5_skew_r1's real layout.
    data = _synthetic_insts([(0, 2048), (1, 4096)])
    assert discover_arg_sizes(data) == {0: 8192, 1: 16384}


def test_a_transfer_at_an_offset_needs_offset_plus_length():
    # 16 words written 4096 bytes into the buffer: the buffer must reach 4096 + 64 bytes.
    data = _synthetic_insts([(0, 16, 4096)])
    assert discover_arg_sizes(data) == {0: 4096 + 64}


def test_the_largest_transfer_of_a_buffer_sizes_it_whatever_the_order():
    # three BDs on one buffer: 8192 words at 0, then 16 at 0, then 16 at 64 bytes
    data = _synthetic_insts([(0, 8192), (0, 16), (0, 16, 64)])
    assert discover_arg_sizes(data) == {0: 8192 * 4}
    data = _synthetic_insts([(0, 16), (0, 8192)])
    assert discover_arg_sizes(data) == {0: 8192 * 4}


def test_a_folded_aperture_offset_is_not_counted_as_extent():
    data = _synthetic_insts([(5, 16, DDR_AIE_ADDR_OFFSET + 128)])
    assert discover_arg_sizes(data) == {5: 128 + 64}


def _runner():
    path = os.environ.get("BRIDGE_TRACE_RUNNER", os.path.join(
        os.path.dirname(__file__), "..", "bridge-runner", "build", "bridge-trace-runner"))
    return path if os.path.exists(path) else None


@pytest.mark.parametrize("pairs", [
    [(0, 2048), (1, 4096)],
    [(0, 16, 4096)],
    [(0, 8192), (0, 16), (0, 16, 64)],
    [(0, 16), (0, 8192), (1, 16, 32768)],
    [(5, 16, DDR_AIE_ADDR_OFFSET + 128)],
])
def test_the_runner_sizes_buffers_as_the_mirror_does(pairs, tmp_path):
    exe = _runner()
    if exe is None:
        pytest.skip("bridge-trace-runner not built (set BRIDGE_TRACE_RUNNER)")
    data = _synthetic_insts(pairs)
    f = tmp_path / "insts.bin"
    f.write_bytes(data)
    out = subprocess.run([exe, "--print-arg-sizes", str(f)], capture_output=True, text=True, check=True).stdout
    got = {int(a): int(b) for a, b in (ln.split() for ln in out.splitlines() if ln.strip())}
    assert got == discover_arg_sizes(data)


def test_empty_on_bad_magic():
    assert discover_arg_sizes(b"\x00" * 64) == {}


def test_empty_on_truncated():
    assert discover_arg_sizes(b"") == {}


def test_real_sp5_skew_r1_insts_if_present():
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "mlir-aie", "build", "test", "npu-xrt",
        "sp5_skew_r1", "chess", "insts.bin",
    )
    if not os.path.exists(path):
        return  # build artifact not present -- skip (not a failure)
    with open(path, "rb") as f:
        data = f.read()
    sizes = discover_arg_sizes(data)
    # arg 0 = the 8192-byte output drain (the buffer the bug under-allocated),
    # arg 1 = the 16384-byte trace buffer.
    assert sizes.get(0) == 8192, sizes
    assert sizes.get(1) == 16384, sizes


if __name__ == "__main__":
    test_synthetic_two_buffers()
    test_empty_on_bad_magic()
    test_empty_on_truncated()
    test_real_sp5_skew_r1_insts_if_present()
    print("all bridge-runner arg-size tests passed")
