"""Tests for tools/trace_selectors.py: the memtile DMA-channel and lock
selection that gives the SELn trace events their meaning.

Run with:
    PYTHONPATH=tools python3 -m pytest tools/test_trace_selectors.py -v

Everything here is checked against the AM025 register database that ships
with mlir-aie, because that database (not a literal in the tools) is the
source of the register names, field widths, reset values and the channel and
lock counts.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import toolchain_paths  # noqa: E402
import trace_selectors as ts  # noqa: E402

pytestmark = pytest.mark.skipif(
    toolchain_paths.try_resolve_mlir_aie() is None,
    reason="mlir-aie register database not found",
)


@pytest.fixture(scope="module")
def regdb():
    return ts.load_memtile_registers()


def _writes(sel):
    return {(w.register, w.field): w.value for w in sel.writes}


class TestNoSelection:
    @pytest.mark.parametrize("spec", [None, "", "  ", " , "])
    def test_empty_spec_requests_nothing(self, spec, regdb):
        assert ts.parse_selection(spec, regdb) is None


class TestDmaChannels:
    def test_slot_maps_to_regdb_field(self, regdb):
        sel = ts.parse_selection("S2MM_SEL0:1", regdb)
        assert _writes(sel)[("DMA_Event_Channel_Selection", "S2MM_Sel0_Channel")] == 1

    def test_unrequested_dma_slots_are_written_as_reset(self, regdb):
        """One register write replaces the whole register, so the three slots
        the caller did not name must carry their reset value explicitly."""
        sel = ts.parse_selection("MM2S_SEL1:3", regdb)
        w = _writes(sel)
        assert w[("DMA_Event_Channel_Selection", "MM2S_Sel1_Channel")] == 3
        for field in ("S2MM_Sel0_Channel", "S2MM_Sel1_Channel", "MM2S_Sel0_Channel"):
            assert w[("DMA_Event_Channel_Selection", field)] == 0
        assert sel.dma == {"S2MM_SEL0": 0, "S2MM_SEL1": 0, "MM2S_SEL0": 0, "MM2S_SEL1": 3}

    def test_lock_only_request_writes_no_dma_register(self, regdb):
        sel = ts.parse_selection("LOCK_SEL0:5", regdb)
        assert sel.dma == {}
        assert not any(w.register == "DMA_Event_Channel_Selection" for w in sel.writes)

    @pytest.mark.parametrize("slot,bad", [("S2MM_SEL0", 6), ("MM2S_SEL1", 7), ("S2MM_SEL1", -1)])
    def test_channel_beyond_the_channels_the_tile_has_is_rejected(self, slot, bad, regdb):
        # Channel 6 and 7 fit the 3-bit field but name no channel (the
        # database lists DMA_S2MM_0..5 / DMA_MM2S_0..5 control registers).
        with pytest.raises(ValueError, match="out of range"):
            ts.parse_selection(f"{slot}:{bad}", regdb)

    def test_channel_count_comes_from_the_database(self, regdb):
        assert ts.channel_count(regdb, "S2MM") == 6
        assert ts.channel_count(regdb, "MM2S") == 6


class TestLockSelectors:
    def test_lock_select_and_reset_value_are_both_written(self, regdb):
        """The register is written whole: naming only the lock would clear
        Lock_Value (reset 0x3F) and change what EQUAL_TO_VALUE means."""
        sel = ts.parse_selection("LOCK_SEL1:33", regdb)
        w = _writes(sel)
        assert w[("Locks_Event_Selection_1", "Lock_Select")] == 33
        assert w[("Locks_Event_Selection_1", "Lock_Value")] == 0x3F
        assert sel.locks == {"LOCK_SEL1": (33, 0x3F)}

    def test_explicit_trigger_value(self, regdb):
        sel = ts.parse_selection("LOCK_SEL0:5@2", regdb)
        assert _writes(sel)[("Locks_Event_Selection_0", "Lock_Value")] == 2
        assert sel.locks == {"LOCK_SEL0": (5, 2)}

    def test_only_named_lock_selectors_are_written(self, regdb):
        sel = ts.parse_selection("LOCK_SEL0:5,LOCK_SEL7:63", regdb)
        regs = {w.register for w in sel.writes}
        assert regs == {"Locks_Event_Selection_0", "Locks_Event_Selection_7"}

    def test_lock_id_up_to_the_last_memtile_lock(self, regdb):
        # Memtile locks are 0..63; the 6-bit Lock_Select reaches all of them.
        assert ts.lock_count(regdb) == 64
        ts.parse_selection("LOCK_SEL3:63", regdb)
        with pytest.raises(ValueError, match="out of range"):
            ts.parse_selection("LOCK_SEL3:64", regdb)

    def test_trigger_value_must_fit_the_field(self, regdb):
        ts.parse_selection("LOCK_SEL0:1@63", regdb)
        with pytest.raises(ValueError, match="out of range"):
            ts.parse_selection("LOCK_SEL0:1@64", regdb)


class TestMixedAndRejects:
    def test_dma_and_locks_in_one_spec(self, regdb):
        sel = ts.parse_selection(
            "S2MM_SEL0:1, MM2S_SEL0:0, LOCK_SEL0:4, LOCK_SEL1:5@1", regdb)
        assert sel.dma["S2MM_SEL0"] == 1
        assert sel.locks == {"LOCK_SEL0": (4, 0x3F), "LOCK_SEL1": (5, 1)}

    @pytest.mark.parametrize("spec,needle", [
        ("S2MM_SEL2:0", "unknown slot"),
        ("LOCK_SEL8:0", "unknown slot"),
        ("S2MM_SEL0", "missing ':'"),
        ("S2MM_SEL0:x", "invalid"),
        ("S2MM_SEL0:1@2", "applies only to LOCK_SEL"),
        ("S2MM_SEL0:1,S2MM_SEL0:2", "twice"),
    ])
    def test_malformed_specs_are_rejected_by_name(self, spec, needle, regdb):
        with pytest.raises(ValueError, match=needle):
            ts.parse_selection(spec, regdb)

    def test_slot_names_are_case_insensitive(self, regdb):
        assert ts.parse_selection("s2mm_sel1:2", regdb).dma["S2MM_SEL1"] == 2


class TestRecord:
    def test_json_record_names_what_each_selector_meant(self, regdb):
        sel = ts.parse_selection("S2MM_SEL0:1,LOCK_SEL0:4@1", regdb)
        assert sel.as_json() == {
            "dma_channels": {"S2MM_SEL0": 1, "S2MM_SEL1": 0,
                             "MM2S_SEL0": 0, "MM2S_SEL1": 0},
            "locks": {"LOCK_SEL0": {"lock": 4, "value": 1}},
        }

    def test_json_record_omits_groups_not_requested(self, regdb):
        assert "locks" not in ts.parse_selection("S2MM_SEL0:1", regdb).as_json()
        assert "dma_channels" not in ts.parse_selection("LOCK_SEL2:9", regdb).as_json()
