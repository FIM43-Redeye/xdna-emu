"""Tests for trace-inject.py shim DMA channel conflict guard.

Validates that find_occupied_shim_dma_channels() correctly identifies
S2MM-occupied channels on shim tiles, preventing trace injection from
reusing them.  Trace collection uses S2MM (stream-to-memory) channels;
MM2S (memory-to-stream) channels are physically independent and do not
conflict with trace.

Also validates has_control_overlay() detection for control packet tests.
"""

import sys
from pathlib import Path

import pytest

# Import trace-inject.py (filename has a hyphen, so importlib is needed).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "trace_inject", Path(__file__).parent / "trace-inject.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["trace_inject"] = _mod
_spec.loader.exec_module(_mod)

find_occupied_shim_dma_channels = _mod.find_occupied_shim_dma_channels
has_control_overlay = _mod.has_control_overlay


# ---------------------------------------------------------------------------
# Test fixtures -- minimal MLIR snippets for regex matching
# ---------------------------------------------------------------------------

# packet_flow dest on shim DMA channel 1 (col 0, row 0).
# From packet_flow_fanout: output data routed to shim via packet_flow.
# packet_dest on shim = S2MM (data flows into shim from tiles).
MLIR_DEST_SHIM_DMA_1 = """\
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
"""

# packet_flow source on shim DMA channel 0 (col 0, row 0).
# From add_one_ctrl_packet: control packets sent FROM shim to tiles.
# packet_source on shim = MM2S (data flows out of shim to tiles).
MLIR_SOURCE_SHIM_DMA_0 = """\
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
"""

# Both S2MM channels on col 0 occupied (dest on ch 0 and ch 1).
MLIR_BOTH_S2MM_COL0 = """\
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
"""

# Source (MM2S) and dest (S2MM) on same channel number (col 0, ch 0).
# Only the dest (S2MM) counts as occupied for trace.
MLIR_SOURCE_AND_DEST_SAME_CHANNEL = """\
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
"""

# No shim DMA in packet_flow -- only aie.flow (circuit-switched).
# From add_one_using_dma: standard flow, no packet routing to shim.
MLIR_NO_PACKET_FLOW_DMA = """\
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
"""

# Multi-column: col 0 both S2MM channels occupied, col 1 free.
MLIR_COL0_OCCUPIED_COL1_FREE = """\
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
"""

# Tile name variant: %shim_noc_tile_C_0 (used in some traced MLIR output).
MLIR_SHIM_NOC_TILE_NAME = """\
    aie.packet_flow(10) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    }
"""

# Non-shim tile DMA (row != 0) -- should NOT be detected as occupied.
MLIR_NON_SHIM_DMA = """\
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
"""

# shim_dma_allocation with S2MM direction -- should be detected.
MLIR_ALLOC_S2MM = """\
    aie.shim_dma_allocation @objFifo_out0 (%tile_0_0, S2MM, 0)
"""

# shim_dma_allocation with MM2S direction -- should NOT be detected.
MLIR_ALLOC_MM2S = """\
    aie.shim_dma_allocation @objFifo_in0 (%tile_0_0, MM2S, 0)
"""

# shim_dma_allocation with shim_noc_tile name variant.
MLIR_ALLOC_NOC_TILE = """\
    aie.shim_dma_allocation @out0 (%shim_noc_tile_0_0, S2MM, 1)
"""

# Combined: packet_dest + shim_dma_allocation both contribute.
# From add_one_ctrl_packet: packet_dest S2MM:0 + alloc S2MM:0 + alloc S2MM:1.
MLIR_COMBINED_DEST_AND_ALLOC = """\
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)
"""

# Control overlay: @base device block present.
MLIR_WITH_OVERLAY = """\
module {
  aie.device(npu1_1col) @base {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
  }
  aie.device(npu1_1col) @main {
    %tile_0_0 = aie.tile(0, 0)
  }
}
"""

# No control overlay: single device, no @base.
MLIR_WITHOUT_OVERLAY = """\
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
  }
}
"""


# ---------------------------------------------------------------------------
# Tests -- S2MM occupancy detection
# ---------------------------------------------------------------------------

class TestFindOccupiedShimDmaChannels:
    """Tests for find_occupied_shim_dma_channels().

    The guard returns S2MM-occupied channels only, since trace uses S2MM.
    MM2S channels (packet_source on shim, shim_dma_allocation with MM2S)
    are physically independent and not detected.
    """

    def test_dest_channel_detected(self):
        """packet_dest on shim DMA = S2MM, detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_DEST_SHIM_DMA_1)
        assert (0, 1) in occupied

    def test_source_channel_not_detected(self):
        """packet_source on shim DMA = MM2S, NOT detected (independent)."""
        occupied = find_occupied_shim_dma_channels(MLIR_SOURCE_SHIM_DMA_0)
        assert len(occupied) == 0

    def test_both_s2mm_channels_occupied(self):
        """Both S2MM channels on a column can be detected as occupied."""
        occupied = find_occupied_shim_dma_channels(MLIR_BOTH_S2MM_COL0)
        assert occupied == {(0, 0), (0, 1)}

    def test_only_dest_detected_when_both_directions(self):
        """With source (MM2S) + dest (S2MM) on same channel, only S2MM counts."""
        occupied = find_occupied_shim_dma_channels(
            MLIR_SOURCE_AND_DEST_SAME_CHANNEL,
        )
        assert (0, 0) in occupied
        assert len(occupied) == 1

    def test_no_packet_flow_dma(self):
        """Standard aie.flow (circuit-switched) has no exclusions."""
        occupied = find_occupied_shim_dma_channels(MLIR_NO_PACKET_FLOW_DMA)
        assert len(occupied) == 0

    def test_non_shim_dma_not_detected(self):
        """packet_flow on non-shim tiles (row != 0) is not detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_NON_SHIM_DMA)
        assert len(occupied) == 0

    def test_col0_occupied_col1_free(self):
        """Col 0 S2MM channels occupied, col 1 channels are free."""
        occupied = find_occupied_shim_dma_channels(
            MLIR_COL0_OCCUPIED_COL1_FREE,
        )
        assert (0, 0) in occupied
        assert (0, 1) in occupied
        assert (1, 0) not in occupied
        assert (1, 1) not in occupied

    def test_shim_noc_tile_name_variant(self):
        """Detects channels on %shim_noc_tile_C_0 name variant."""
        occupied = find_occupied_shim_dma_channels(MLIR_SHIM_NOC_TILE_NAME)
        assert (0, 1) in occupied


class TestShimDmaAllocation:
    """Tests for shim_dma_allocation parsing in find_occupied_shim_dma_channels()."""

    def test_alloc_s2mm_detected(self):
        """shim_dma_allocation with S2MM direction is detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_ALLOC_S2MM)
        assert (0, 0) in occupied

    def test_alloc_mm2s_not_detected(self):
        """shim_dma_allocation with MM2S direction is NOT detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_ALLOC_MM2S)
        assert len(occupied) == 0

    def test_alloc_noc_tile_name(self):
        """shim_dma_allocation with %shim_noc_tile name variant is detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_ALLOC_NOC_TILE)
        assert (0, 1) in occupied

    def test_combined_dest_and_alloc(self):
        """packet_dest + shim_dma_allocation both contribute to occupied set."""
        occupied = find_occupied_shim_dma_channels(MLIR_COMBINED_DEST_AND_ALLOC)
        assert (0, 0) in occupied  # from packet_dest AND alloc S2MM:0
        assert (0, 1) in occupied  # from alloc S2MM:1 only
        assert len(occupied) == 2


class TestControlOverlayDetection:
    """Tests for has_control_overlay()."""

    def test_overlay_detected(self):
        """@base device block detected as control overlay."""
        assert has_control_overlay(MLIR_WITH_OVERLAY)

    def test_no_overlay(self):
        """Single device without @base is not an overlay."""
        assert not has_control_overlay(MLIR_WITHOUT_OVERLAY)


class TestCandidateFiltering:
    """Integration-level tests for candidate filtering logic.

    These test the filtering pattern used in plan_trace_route() without
    invoking the full planner (which requires mlir-aie Python API).
    """

    def test_occupied_excluded_from_candidates(self):
        """Occupied S2MM channels are excluded from candidate list."""
        occupied = {(0, 0), (0, 1)}
        num_cols = 2
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert (0, 0) not in candidates
        assert (0, 1) not in candidates
        assert (1, 0) in candidates
        assert (1, 1) in candidates

    def test_all_occupied_yields_empty(self):
        """When all S2MM channels on all columns are occupied, no candidates."""
        occupied = {(0, 0), (0, 1)}
        num_cols = 1
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert len(candidates) == 0

    def test_no_occupied_preserves_all(self):
        """When no S2MM channels are occupied, all candidates remain."""
        occupied = set()
        num_cols = 2
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert len(candidates) == 4

    def test_mm2s_only_does_not_filter(self):
        """MM2S-only occupancy (e.g., ctrl overlay) leaves all candidates."""
        # Simulate: source MLIR has packet_source on shim DMA:0 (MM2S)
        # Guard should return empty set, so all candidates survive.
        occupied = find_occupied_shim_dma_channels(MLIR_SOURCE_SHIM_DMA_0)
        num_cols = 1
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert len(candidates) == 2  # both (0,0) and (0,1) are free


assign_targets_to_shims = _mod.assign_targets_to_shims
TraceSlot = _mod.TraceSlot


class TestAssignTargetsToShims:
    """Tests for assign_targets_to_shims() -- pure logic, no mlir-aie needed.

    The function assigns trace targets to (shim_col, channel) slots,
    preferring same-column shims and load-balancing across free slots.
    """

    def test_single_col_same_shim(self):
        """Single-column targets route to same-column shim."""
        targets = [(0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1)]
        result = assign_targets_to_shims(targets, set(), num_cols=1)
        # All targets should go to col 0 (only column available)
        col0_targets = []
        for slot, tgts in result.items():
            assert slot[0] == 0
            col0_targets.extend(tgts)
        assert set(map(tuple, col0_targets)) == set(targets)

    def test_multi_col_local_routing(self):
        """Multi-column targets prefer their own column's shim."""
        targets = [
            (0, 2, 0), (0, 2, 1),  # col 0 tiles
            (1, 2, 0), (1, 2, 1),  # col 1 tiles
        ]
        result = assign_targets_to_shims(targets, set(), num_cols=2)
        # Col 0 targets should go to col 0 shim
        # Col 1 targets should go to col 1 shim
        for slot, tgts in result.items():
            for col, row, port in tgts:
                assert col == slot[0], (
                    f"Target col {col} assigned to shim col {slot[0]}"
                )

    def test_occupied_skipped(self):
        """Occupied S2MM channels are never assigned."""
        targets = [(0, 2, 0)]
        occupied = {(0, 0), (0, 1)}  # both col 0 channels occupied
        result = assign_targets_to_shims(targets, occupied, num_cols=2)
        for slot in result:
            assert slot[0] != 0 or slot not in occupied

    def test_all_occupied_empty(self):
        """All channels occupied -> empty assignment."""
        targets = [(0, 2, 0)]
        occupied = {(0, 0), (0, 1)}
        result = assign_targets_to_shims(targets, occupied, num_cols=1)
        assert result == {}

    def test_nearest_fallback(self):
        """When same-column shim is occupied, falls back to nearest."""
        targets = [(0, 2, 0), (0, 2, 1)]
        occupied = {(0, 0), (0, 1)}  # col 0 full
        result = assign_targets_to_shims(targets, occupied, num_cols=2)
        # Should fall back to col 1
        for slot in result:
            assert slot[0] == 1

    def test_load_balance(self):
        """Targets spread across channels when same distance."""
        # 4 targets on col 0, 2 channels available on col 0
        targets = [
            (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1),
        ]
        result = assign_targets_to_shims(targets, set(), num_cols=1)
        # With 2 channels on col 0, load should be balanced
        for slot, tgts in result.items():
            assert len(tgts) <= 3  # no slot gets all 4

    def test_three_col_distribution(self):
        """Three-column design distributes to per-column shims."""
        targets = [
            (0, 2, 0), (0, 2, 1),
            (1, 2, 0), (1, 2, 1),
            (2, 2, 0), (2, 2, 1),
        ]
        result = assign_targets_to_shims(targets, set(), num_cols=3)
        # Each column's targets should prefer their own shim
        for slot, tgts in result.items():
            for col, row, port in tgts:
                # Distance should be 0 (same col) since all shims are free
                assert col == slot[0]

    def test_partial_occupied_reroutes(self):
        """Col 1 both occupied -> col 1 targets go to col 0 or col 2."""
        targets = [
            (0, 2, 0),
            (1, 2, 0),
            (2, 2, 0),
        ]
        occupied = {(1, 0), (1, 1)}
        result = assign_targets_to_shims(targets, occupied, num_cols=3)
        # Col 1 target should go to col 0 or col 2 (both distance=1)
        for slot, tgts in result.items():
            assert slot not in occupied
            for col, row, port in tgts:
                if col == 1:
                    assert abs(slot[0] - 1) == 1

    def test_empty_targets(self):
        """No targets -> empty assignment."""
        result = assign_targets_to_shims([], set(), num_cols=4)
        assert result == {}

    def test_preserves_all_targets(self):
        """Every input target appears exactly once in the output."""
        targets = [
            (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1),
            (1, 2, 0), (1, 2, 1),
        ]
        result = assign_targets_to_shims(targets, set(), num_cols=2)
        all_assigned = []
        for tgts in result.values():
            all_assigned.extend(tgts)
        assert sorted(all_assigned) == sorted(targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
