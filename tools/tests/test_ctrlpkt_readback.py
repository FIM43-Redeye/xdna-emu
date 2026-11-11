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


def test_register_offsets_come_from_the_am025_database():
    assert (cp.LOCKS_OVERFLOW, cp.LOCKS_UNDERFLOW) == (0x1F120, 0x1F128)
    assert cp.MEM_EVENT_STATUS3 == 0x1420C


def test_each_target_tile_has_its_own_packet_and_response_ids():
    # Distinct pairs: probe generators register aliases of a base tile (pipeline-fill "prod" is "core").
    ids = list({(t.pkt_id, t.response_id) for t in cp.TARGETS.values()})
    assert len({p for p, _ in ids}) == len(ids) and len({r for _, r in ids}) == len(ids)
    steps = [cp.Read(0x14120, "shim", tile="shim"), cp.Read(0xC0420, "mem", tile="mem")]
    lay = cp.encode(steps)
    assert (lay.words[0] >> 24) & 0xF == cp.TARGETS["shim"].response_id
    assert (lay.words[1] >> 24) & 0xF == cp.TARGETS["mem"].response_id
    ops = cp.sequence_lines(steps, lay, "o", "i")
    assert f"pkt_id = {cp.TARGETS['shim'].pkt_id}," in ops[1]
    assert f"pkt_id = {cp.TARGETS['mem'].pkt_id}," in ops[3]


def test_a_raw_step_is_emitted_where_it_stands_and_sends_no_packet():
    steps = [cp.Read(0x1F120, "a"), cp.Raw(("aiex.dma_start_task(%t)",)), cp.Read(0x1F128, "b")]
    lay = cp.encode(steps)
    assert lay.packets == ((0, 0, 1), (2, 1, 1))
    ops = cp.sequence_lines(steps, lay, "o", "i")
    at = [i for i, l in enumerate(ops) if "dma_start_task" in l]
    assert at and "ctrlin" in ops[at[0] - 1] and "ctrlin" in ops[at[0] + 1]


@pytest.mark.parametrize("target, flags, locks", [("shim", 1, 16), ("mem", 2, 64), ("core", 1, 16)])
def test_tile_registers_come_from_the_am025_database_per_tile_kind(target, flags, locks):
    r = cp.TileRegs(target)
    assert len(r.overflow) == flags == len(r.underflow)
    assert r.flag_bits * len(r.overflow) == locks
    assert r.value(1) - r.value(0) == 0x10
    # Lock_Id sits at bit 10, Acq_Rel at 9, the 7-bit change at 2
    assert r.request(3, True, -1) == r.request_base + (3 << 10) + 0x200 + (0x7F << 2)
    assert r.request(3, False, 1) == r.request_base + (3 << 10) + 4
    assert (r.flag_reg(r.overflow, locks - 1), r.flag_bit(locks - 1)) == (r.overflow[-1], 1 << (r.flag_bits - 1))
