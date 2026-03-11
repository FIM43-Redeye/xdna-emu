"""Tests for trace-inject.py shim DMA channel conflict guard.

Validates that find_occupied_shim_dma_channels() correctly identifies
application packet_flow ops that occupy shim DMA channels, preventing
trace injection from reusing them and causing data corruption.
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


# ---------------------------------------------------------------------------
# Test fixtures -- minimal MLIR snippets for regex matching
# ---------------------------------------------------------------------------

# packet_flow dest on shim DMA channel 1 (col 0, row 0).
# From packet_flow_fanout: output data routed to shim via packet_flow.
MLIR_DEST_SHIM_DMA_1 = """\
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
"""

# packet_flow source on shim DMA channel 0 (col 0, row 0).
# From add_one_ctrl_packet: control packets sent FROM shim to tiles.
MLIR_SOURCE_SHIM_DMA_0 = """\
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
"""

# Both channels on col 0 occupied (dest on ch 0 and ch 1).
MLIR_BOTH_CHANNELS_COL0 = """\
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(6) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
"""

# Source and dest on same channel (col 0, ch 0) -- from add_one_ctrl_packet.
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

# Multi-column: col 0 both channels occupied, col 1 free.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindOccupiedShimDmaChannels:
    """Tests for find_occupied_shim_dma_channels()."""

    def test_dest_channel_detected(self):
        """DMA channel used by packet_flow dest on shim is detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_DEST_SHIM_DMA_1)
        assert (0, 1) in occupied

    def test_source_channel_detected(self):
        """DMA channel used by packet_flow source on shim is detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_SOURCE_SHIM_DMA_0)
        assert (0, 0) in occupied

    def test_both_channels_occupied(self):
        """Both channels on a column can be detected as occupied."""
        occupied = find_occupied_shim_dma_channels(MLIR_BOTH_CHANNELS_COL0)
        assert occupied == {(0, 0), (0, 1)}

    def test_source_and_dest_same_channel(self):
        """Source and dest on same channel produce a single entry."""
        occupied = find_occupied_shim_dma_channels(
            MLIR_SOURCE_AND_DEST_SAME_CHANNEL,
        )
        assert (0, 0) in occupied
        # Only one entry for (0, 0) even though both source and dest use it.
        assert len([p for p in occupied if p == (0, 0)]) == 1

    def test_no_packet_flow_dma(self):
        """Standard aie.flow (circuit-switched) has no exclusions."""
        occupied = find_occupied_shim_dma_channels(MLIR_NO_PACKET_FLOW_DMA)
        assert len(occupied) == 0

    def test_non_shim_dma_not_detected(self):
        """packet_flow on non-shim tiles (row != 0) is not detected."""
        occupied = find_occupied_shim_dma_channels(MLIR_NON_SHIM_DMA)
        assert len(occupied) == 0

    def test_col0_occupied_col1_free(self):
        """Col 0 channels occupied, col 1 channels are free."""
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


class TestCandidateFiltering:
    """Integration-level tests for candidate filtering logic.

    These test the filtering pattern used in plan_trace_route() without
    invoking the full planner (which requires mlir-aie Python API).
    """

    def test_occupied_excluded_from_candidates(self):
        """Occupied channels are excluded from candidate list."""
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
        """When all channels on all columns are occupied, no candidates."""
        occupied = {(0, 0), (0, 1)}
        num_cols = 1
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert len(candidates) == 0

    def test_no_occupied_preserves_all(self):
        """When no channels are occupied, all candidates remain."""
        occupied = set()
        num_cols = 2
        candidates = [
            (col, ch) for col in range(num_cols) for ch in range(2)
            if (col, ch) not in occupied
        ]
        assert len(candidates) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
