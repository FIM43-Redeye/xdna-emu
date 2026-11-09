"""Tests for tools/ctrlpkt_readback.py: control-packet encoding and the runtime sequence."""
import pytest

import ctrlpkt_readback as cp


def test_header_fields_and_odd_parity():
    h = cp.header(cp.LOCKS_OVERFLOW, 1, response_id=2, beats=1)
    assert h & 0xFFFFF == cp.LOCKS_OVERFLOW
    assert (h >> 22) & 3 == 1 and (h >> 24) & 0xF == 2 and (h >> 20) & 3 == 0
    assert bin(h).count("1") % 2 == 1


def test_header_matches_the_mlir_aie_test_packet():
    # mlir-aie test/npu-xrt/add_one_ctrl_packet/test.cpp: write to Lock0_value (0x1F000), stream 0
    h = cp.header(0x1F000, 0)
    assert h == 0x1F000 | (0 << 31)          # five set bits: already odd, so no parity bit
    # its reads: stream id 2, opcode 1, four beats, address 0x440
    r = cp.header(0x440, 1, response_id=2, beats=4)
    assert r & 0x7FFFFFFF == 2 << 24 | 1 << 22 | 3 << 20 | 0x440


@pytest.mark.parametrize("addr, opcode, beats", [(1 << 20, 0, 1), (0, 2, 1), (0, 0, 0), (0, 0, 5)])
def test_header_rejects_out_of_range_fields(addr, opcode, beats):
    with pytest.raises(ValueError):
        cp.header(addr, opcode, beats=beats)


def test_lock_value_registers_are_16_bytes_apart():
    assert cp.lock_value_addr(3) - cp.lock_value_addr(2) == 0x10
    assert cp.lock_value_addr(0) == 0x1F000


def test_encode_lays_packets_out_in_step_order():
    steps = [cp.Read(cp.LOCKS_OVERFLOW, "o"), cp.Write(cp.lock_value_addr(1), 7), cp.Wait("x"),
             cp.Read(cp.LOCKS_UNDERFLOW, "u")]
    lay = cp.encode(steps)
    assert lay.packets == ((0, 0, 1), (1, 1, 2), (3, 3, 1))
    assert lay.labels == ("o", "u") and len(lay.words) == 4 and lay.words[2] == 7


def test_sequence_waits_for_the_transfer_between_packets():
    steps = [cp.Write(0x1F000, 1), cp.Wait("stamp0"), cp.Read(0x1F120, "o")]
    lay = cp.encode(steps)
    ops = cp.sequence_lines(steps, lay, "out", "in")
    assert "metadata = @ctrl0" in ops[0] and "[1, 1, 1, 1]" in ops[0]
    waits = [i for i, l in enumerate(ops) if "dma_wait" in l]
    assert "@ctrlin0" in ops[waits[0]] and "@stamp0" in ops[waits[1]]
    assert "@ctrl0" in ops[waits[-1]], "the host ends by waiting for the last response"
