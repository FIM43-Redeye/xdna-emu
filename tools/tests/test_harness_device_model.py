"""Tests for tools/harness/device_model.py (device facts read from the JSON)."""
import pytest

from harness.device_model import (
    DeviceModelError,
    available_devices,
    load_device_model,
)


@pytest.fixture(scope="module")
def npu1():
    return load_device_model("npu1")


def test_available_devices_lists_npu_family():
    names = available_devices()
    assert "npu1" in names and "npu2" in names


def test_unknown_device_raises():
    with pytest.raises(DeviceModelError, match="no_such_device"):
        load_device_model("no_such_device")


def test_array_shape(npu1):
    assert (npu1.columns, npu1.rows) == (4, 6)


def test_tile_types_by_row(npu1):
    assert npu1.tile_type(0, 0) == "shim_noc"
    assert npu1.tile_type(0, 1) == "mem_tile"
    assert npu1.tile_type(3, 2) == "core"
    assert npu1.tile_type(3, 5) == "core"


def test_tile_type_out_of_range_is_none(npu1):
    assert npu1.tile_type(4, 2) is None
    assert npu1.tile_type(0, 6) is None
    assert npu1.tile_type(-1, 2) is None


def test_memory_sizes(npu1):
    assert npu1.data_memory_bytes("core") == npu1.local_memory_size == 65536
    assert npu1.data_memory_bytes("mem_tile") == npu1.mem_tile_size == 524288
    assert npu1.program_memory_bytes == 16384


def test_dma_channel_counts_per_direction(npu1):
    core = npu1.dma_channels("core")
    assert (core.s2mm, core.mm2s) == (2, 2)
    mem = npu1.dma_channels("mem_tile")
    assert (mem.s2mm, mem.mm2s) == (6, 6)
    # Shim DMA ports live in shim_mux_ports, not switchbox_ports.
    shim = npu1.dma_channels("shim_noc")
    assert (shim.s2mm, shim.mm2s) == (2, 2)


def test_core_stream_port_counts(npu1):
    ports = npu1.stream_ports("core", "Core")
    assert (ports.master, ports.slave) == (1, 1)


def test_default_core_stack_and_cascade(npu1):
    assert npu1.default_core_stack_size == 1024
    assert npu1.cascade.width_bits == 512
    assert npu1.cascade.core_tiles_only


def test_cascade_direction_rule(npu1):
    c = npu1.cascade
    # src directly North of dst (row + 1): flow leaves South.
    assert c.direction((1, 3), (1, 2)) == "South"
    # src directly West of dst (col - 1): flow leaves East.
    assert c.direction((0, 2), (1, 2)) == "East"
    assert c.direction((1, 2), (1, 3)) is None   # wrong way
    assert c.direction((1, 2), (0, 2)) is None
    assert c.direction((0, 3), (1, 2)) is None   # diagonal
    assert c.direction((1, 4), (1, 2)) is None   # not adjacent
    assert c.direction((1, 2), (1, 2)) is None


# -- facts the channel planner needs ----------------------------------------

def test_tile_resource_counts(npu1):
    assert (npu1.num_locks("mem_tile"), npu1.num_bds("mem_tile")) == (64, 48)
    assert (npu1.num_locks("core"), npu1.num_bds("core")) == (16, 16)


def test_dma_alignment_is_granularity_in_bytes(npu1):
    # The JSON carries mlir-aie getAddressGenGranularity(), which is in BITS
    # (32 on AIE2); the 4-byte alignment is derived, not stored.
    assert npu1.address_gen_granularity_bits == 32
    assert npu1.dma_alignment_bytes == 4
