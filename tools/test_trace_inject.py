"""Tests for trace injection tooling.

Two separate test sections:

1. Tests for the deprecated trace-inject.py at tools/deprecated/trace-inject.py
   (shim DMA channel conflict guard, target assignment, control overlay
   detection).  These functions are pure Python and do not require the
   mlir-aie environment.  The deprecated tool is still the authoritative
   implementation of these helpers until they're ported to the new injector.

2. Tests for mlir-trace-inject.py (Task 4 of the A.2 PC-anchored validation plan):
   --trace-mode, per-module grounding/sweep flags, and perfcnt config block emission.
   These require the mlir-aie Python environment (aie.dialects.aie / aiex).  Tests
   invoke mlir-trace-inject.py as a subprocess via run_inject() and inspect the MLIR
   output with string assertions.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Section 1: deprecated trace-inject.py imports
# ---------------------------------------------------------------------------
import importlib.util

# The original trace-inject.py was moved to tools/deprecated/ when the new
# mlir-trace-inject.py replaced it as the active injector.  The helper
# functions tested here (find_occupied_shim_dma_channels, has_control_overlay,
# assign_targets_to_shims) are still authoritative for the regex-based shim
# DMA channel conflict guard and have not been ported to the new injector;
# we keep them under test from the deprecated location.
_TRACE_INJECT_PATH = Path(__file__).parent / "deprecated" / "trace-inject.py"
_trace_inject_missing = not _TRACE_INJECT_PATH.exists()

if not _trace_inject_missing:
    _spec = importlib.util.spec_from_file_location(
        "trace_inject", _TRACE_INJECT_PATH,
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["trace_inject"] = _mod
    _spec.loader.exec_module(_mod)

    find_occupied_shim_dma_channels = _mod.find_occupied_shim_dma_channels
    has_control_overlay = _mod.has_control_overlay
else:
    # Provide stubs so the name resolution at module level succeeds.
    # Tests in Section 1 are skipped via _skip_section1 below.
    def find_occupied_shim_dma_channels(text):  # noqa: F811
        raise NotImplementedError("deprecated/trace-inject.py not present")

    def has_control_overlay(text):  # noqa: F811
        raise NotImplementedError("deprecated/trace-inject.py not present")

_skip_section1 = pytest.mark.skipif(
    _trace_inject_missing,
    reason="tools/deprecated/trace-inject.py not present",
)


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

@_skip_section1
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


@_skip_section1
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


@_skip_section1
class TestControlOverlayDetection:
    """Tests for has_control_overlay()."""

    def test_overlay_detected(self):
        """@base device block detected as control overlay."""
        assert has_control_overlay(MLIR_WITH_OVERLAY)

    def test_no_overlay(self):
        """Single device without @base is not an overlay."""
        assert not has_control_overlay(MLIR_WITHOUT_OVERLAY)


@_skip_section1
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


if not _trace_inject_missing:
    assign_targets_to_shims = _mod.assign_targets_to_shims
    TraceSlot = _mod.TraceSlot
else:
    def assign_targets_to_shims(*a, **kw):  # noqa: F811
        raise NotImplementedError("trace-inject.py not present")
    TraceSlot = None


@_skip_section1
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


# ---------------------------------------------------------------------------
# Section 2: mlir-trace-inject.py tests (Task 4 of A.2 PC-anchored plan)
#
# These tests invoke mlir-trace-inject.py as a subprocess and inspect the
# generated MLIR output with string assertions.  They require the mlir-aie
# Python environment (aie.dialects.aie / aiex).
#
# Fixture: simple_design_mlir -- a minimal well-formed aie.device MLIR that
#   has exactly one compute tile at (0,2) and a non-empty runtime_sequence.
#   "Non-empty" is required because the current injector accesses blocks[0]
#   on the runtime_sequence region -- an empty {} body has 0 blocks.
#
# Helper: run_inject(argv) -- invokes mlir-trace-inject.py as a subprocess
#   and returns its exit code.  Stderr is passed through so capsys can
#   capture it.
# ---------------------------------------------------------------------------

_MLIR_INJECT = Path(__file__).parent / "mlir-trace-inject.py"

# Import the helper functions directly (not the whole module) for pure-Python
# unit testing of _resolve_events.  The hyphen in the filename means we need
# importlib here too.  Importing the module's top level does NOT import the
# heavy mlir-aie deps -- those are imported lazily inside main().
_mti_spec = importlib.util.spec_from_file_location(
    "mlir_trace_inject", _MLIR_INJECT,
)
_mti_mod = importlib.util.module_from_spec(_mti_spec)
sys.modules["mlir_trace_inject"] = _mti_mod
_mti_spec.loader.exec_module(_mti_mod)

_resolve_events = _mti_mod._resolve_events

# Minimal MLIR with one compute tile (col=0, row=2) and a non-empty
# runtime_sequence so the injector can walk both the device body and
# the runtime sequence block.
_SIMPLE_DESIGN_MLIR = """\
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c64] [%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @dummy} : memref<64xi32>
    }
  }
}
"""


@pytest.fixture
def simple_design_mlir():
    """Minimal MLIR string for mlir-trace-inject tests."""
    return _SIMPLE_DESIGN_MLIR


def run_inject(argv: list) -> int:
    """Run mlir-trace-inject.py with the given argument list.

    Returns the process exit code.  stdout/stderr are passed through to the
    test process so pytest's capsys can capture them.

    Auto-supplies --trace-config-out next to --out when injecting; the
    injector errors out without it, and every test here writes to a
    tmp_path so the auto-derivation is correct by construction.
    """
    argv = list(argv)
    if (
        "--no-op" not in argv
        and "--help" not in argv
        and "--trace-config-out" not in argv
        and "--out" in argv
    ):
        out_path = Path(argv[argv.index("--out") + 1])
        argv += ["--trace-config-out",
                 str(out_path.with_name("trace_config.json"))]
    result = subprocess.run(
        [sys.executable, str(_MLIR_INJECT)] + argv,
        # Let both streams flow to the test process stdout/stderr so
        # capsys.readouterr() can capture them.
    )
    return result.returncode


class TestResolveEvents:
    """Pure-Python unit tests for _resolve_events() (no mlir-aie env needed).

    Validates the dedup + cap-at-8 contract that drives event-slot assignment
    in the injector.
    """

    def test_grounding_only(self):
        """No sweep, no defaults -> just grounding."""
        result = _resolve_events("PERF_CNT_0,INSTR_EVENT_0", None, None)
        assert result == ["PERF_CNT_0", "INSTR_EVENT_0"]

    def test_grounding_plus_sweep(self):
        """Explicit sweep is appended after grounding."""
        result = _resolve_events(
            "PERF_CNT_0",
            "INSTR_VECTOR,MEMORY_STALL",
            None,
        )
        assert result == ["PERF_CNT_0", "INSTR_VECTOR", "MEMORY_STALL"]

    def test_dedup_sweep_against_grounding(self):
        """Sweep events that already appear in grounding are deduplicated."""
        result = _resolve_events(
            grounding="PERF_CNT_0,INSTR_VECTOR",
            sweep="INSTR_VECTOR,MEMORY_STALL,LOCK_STALL",
            defaults=None,
        )
        assert result == [
            "PERF_CNT_0", "INSTR_VECTOR", "MEMORY_STALL", "LOCK_STALL",
        ]
        # INSTR_VECTOR must appear exactly once (in its grounding slot).
        assert result.count("INSTR_VECTOR") == 1

    def test_defaults_used_when_sweep_none(self):
        """sweep=None falls back to defaults."""
        result = _resolve_events(
            "PERF_CNT_0", None, ("INSTR_VECTOR", "MEMORY_STALL"),
        )
        assert result == ["PERF_CNT_0", "INSTR_VECTOR", "MEMORY_STALL"]

    def test_all_treated_as_default(self):
        """sweep='all' is reserved; today it behaves like sweep=None."""
        result = _resolve_events(
            "PERF_CNT_0", "all", ("INSTR_VECTOR", "MEMORY_STALL"),
        )
        assert result == ["PERF_CNT_0", "INSTR_VECTOR", "MEMORY_STALL"]

    def test_caps_at_eight(self):
        """Result is capped at 8 slots (hardware trace-unit limit)."""
        sweep = ",".join(f"E{i}" for i in range(20))
        result = _resolve_events("G0,G1", sweep, None)
        assert len(result) == 8
        assert result[:2] == ["G0", "G1"]

    def test_empty_grounding(self):
        """Empty grounding -> sweep fills all slots."""
        result = _resolve_events("", "A,B,C", None)
        assert result == ["A", "B", "C"]


class TestMlirTraceInjectMode:
    """Tests for --trace-mode CLI flag (Task 4.1 / 4.3)."""

    def test_inject_mode_event_pc_default(self, tmp_path, simple_design_mlir):
        """Default (no --trace-mode) emits EventPC mode.

        Production default is event_pc (mode 1) -- it records PC alongside
        each event, which is the easiest mode to ground in disassembly.
        The mlir-aie custom printer serialises TraceMode.EventPC as the
        string literal "Event-PC" (with quotes and a hyphen).
        """
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject(["--input", str(inp), "--out", str(out)])
        assert rc == 0
        text = out.read_text()
        # Actual MLIR text: aie.trace.mode "Event-PC"
        assert 'aie.trace.mode "Event-PC"' in text

    def test_inject_mode_event_pc_round_trips(self, tmp_path, simple_design_mlir):
        """--trace-mode event_pc emits EventPC mode and refuses double-injection.

        The mlir-aie custom printer serialises TraceMode.EventPC as the
        string literal "Event-PC" (with quotes and a hyphen).
        """
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--trace-mode", "event_pc",
        ])
        assert rc == 0
        text = out.read_text()
        # Actual MLIR text: aie.trace.mode "Event-PC"
        assert 'aie.trace.mode "Event-PC"' in text
        # Re-injection must be refused (idempotency guard):
        rc2 = run_inject([
            "--input", str(out), "--out", str(tmp_path / "out2.mlir"),
            "--trace-mode", "event_pc",
        ])
        assert rc2 == 2  # already-traced -> exit 2


class TestMlirTraceInjectPerfcnt:
    """Tests for perfcnt config block emission (Task 4.5 / 4.6)."""

    def test_inject_perfcnt_config_emitted(self, tmp_path, simple_design_mlir):
        """When PERF_CNT_2 is in grounding, a perf_core_<col>_<row> config block
        is emitted with Performance_Counter2_Event_Value set to --perfcnt-period.

        Counter 2 (not counter 0) is the trace anchor since the cnt2 move (#377);
        userspace tools rely on counters 0/1 staying untouched."""
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--trace-mode", "event_pc",
            "--core-grounding", "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1",
            "--perfcnt-period", "1024",
        ])
        assert rc == 0
        text = out.read_text()
        # At least one perf_core_<col>_<row> trace.config block:
        assert "@perf_core_" in text, f"No perf_core_ block found:\n{text[:2000]}"
        # Performance_Counter2_Event_Value with value=1024:
        assert "Performance_Counter2_Event_Value" in text
        assert "1024" in text
        # Referenced in start_config list in runtime_sequence:
        assert "aie.trace.start_config @perf_core_" in text

    def test_inject_perfcnt_not_emitted_without_perf_cnt_2(
        self, tmp_path, simple_design_mlir
    ):
        """When PERF_CNT_2 is absent from grounding, no perf_core_ block is emitted."""
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--core-grounding", "INSTR_EVENT_0,INSTR_EVENT_1",
        ])
        assert rc == 0
        text = out.read_text()
        assert "@perf_core_" not in text

    def test_inject_perfcnt_default_grounding_emits_block(
        self, tmp_path, simple_design_mlir
    ):
        """Default --core-grounding includes PERF_CNT_2, so perf block is emitted
        even with no explicit grounding flags."""
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject(["--input", str(inp), "--out", str(out)])
        assert rc == 0
        text = out.read_text()
        assert "@perf_core_" in text, f"No perf_core_ block found:\n{text[:2000]}"

    def test_inject_perfcnt_uses_correct_registers(
        self, tmp_path, simple_design_mlir
    ):
        """The perfcnt block must write the cnt2-side registers and must NOT use
        field= on Performance_Control{1,2}.

        Per aie-rt xaiemlgbl_params.h:
          - Performance_Control1 holds Cnt2_Start_Event at bits[6:0]   (event 28).
          - Performance_Control2 holds Cnt2_Reset_Event at bits[22:16] (event 7).
        We do NOT touch Performance_Control0 -- that holds cnt0 start/stop
        fields, which we deliberately leave alone so userspace can use cnt0/cnt1.
        The aie_registers_aie2.json regdb has NO named bitfields for these
        registers, so `aie.trace.reg ... field="..."` would fail at lower-time
        with "Field not found in register: ..." -- this pins the regression
        fix from gate-A.2 Task 9.
        """
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--trace-mode", "event_pc",
            "--core-grounding", "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1",
            "--perfcnt-period", "2048",
        ])
        assert rc == 0
        text = out.read_text()

        # Cnt2_Start_Event lives in Performance_Control1 on the core module.
        assert "Performance_Control1" in text, (
            "Performance_Control1 must be written for Cnt2_Start_Event"
        )
        # Cnt2_Reset_Event lives in Performance_Control2 on the core module.
        assert "Performance_Control2" in text, (
            "Performance_Control2 must be written for Cnt2_Reset_Event"
        )
        # Performance_Control0 must NOT be written -- that holds cnt0 fields,
        # which we leave alone so userspace can use cnt0 freely.
        assert "Performance_Control0" not in text, (
            "Performance_Control0 must not be written; cnt0 must remain "
            "available to userspace tools after the cnt2 move (#377)"
        )
        # Performance_Counter2_Event_Value must reflect --perfcnt-period.
        assert "Performance_Counter2_Event_Value" in text
        assert "2048" in text
        # The perf trace.reg ops must NOT carry field= on Performance_Control{0,2}
        # -- the regdb has no named fields for those registers, and field=
        # would fail in AIETraceRegPackWritesPass.  Look for the substring on
        # the same line as Performance_Control to be safe.
        for line in text.splitlines():
            if "Performance_Control" in line and "aie.trace.reg" in line:
                assert "field" not in line, (
                    f"aie.trace.reg on Performance_Control* must not use field=: "
                    f"{line!r}"
                )


class TestMlirTraceInjectWarnings:
    """Tests for --trace-mode event_pc + non-core sweep warning (Task 4.7)."""

    def test_inject_event_pc_with_non_core_sweep_warns(
        self, tmp_path, simple_design_mlir, capsys
    ):
        """--trace-mode event_pc with --memmod-sweep-events warns but succeeds."""
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--trace-mode", "event_pc",
            "--memmod-sweep-events", "DMA_S2MM_0_FINISHED_BD,LOCK_0_ACQUIRED",
        ])
        # Must succeed (exit 0), not refuse (exit 2):
        assert rc == 0
        # Warning must appear -- but we invoked as a subprocess so capsys
        # won't capture its stderr.  Instead re-invoke and capture via
        # subprocess.run with stderr=PIPE. This call doesn't go through
        # run_inject's auto --trace-config-out wiring, so pass it
        # explicitly here -- the injector requires it on injection paths.
        out_warn = tmp_path / "out_warn.mlir"
        result = subprocess.run(
            [sys.executable, str(_MLIR_INJECT),
             "--input", str(inp), "--out", str(out_warn),
             "--trace-config-out", str(tmp_path / "trace_config_warn.json"),
             "--trace-mode", "event_pc",
             "--memmod-sweep-events", "DMA_S2MM_0_FINISHED_BD,LOCK_0_ACQUIRED"],
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0
        stderr = result.stderr.lower()
        assert "warning" in stderr, f"No warning in stderr: {result.stderr!r}"
        assert "memmod" in stderr, f"'memmod' not in warning: {result.stderr!r}"
        # Output MLIR must still contain the EventPC mode:
        text = (tmp_path / "out_warn.mlir").read_text()
        # Actual MLIR text: aie.trace.mode "Event-PC"
        assert 'aie.trace.mode "Event-PC"' in text


# Multi-tile design: two compute tiles plus the implicit shim (row 0) and
# memtile (row 1). Exercises every trace-source path (core, memmod, memtile,
# shim) so the per-source packet-ID assignment can be checked for uniqueness.
_MULTI_TILE_MLIR = """\
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c64] [%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @dummy} : memref<64xi32>
    }
  }
}
"""


@pytest.fixture
def multi_tile_mlir():
    """MLIR string with two compute tiles for packet-ID uniqueness tests."""
    return _MULTI_TILE_MLIR


def _injected_packet_ids(text: str) -> list[int]:
    """Extract the packet IDs from every aie.trace.packet op in injected MLIR.

    The injector serialises packet ops as ``aie.trace.packet id = N type = ...``;
    one appears per trace source (core / memmod / memtile / shim).
    """
    return [int(m) for m in re.findall(r"aie\.trace\.packet id = (\d+)", text)]


class TestMlirTraceInjectPacketIds:
    """Each trace source multiplexed onto the shim's packet-switched S2MM
    channel must carry a UNIQUE packet id -- the shim stream-switch packet
    router arbitrates flows BY id, so two sources sharing an id collide at the
    router and one flow's trace is dropped/mangled (decode then sees only the
    survivor). This regressed silently for any design tracing more than one
    tile of the same packet type; single-tile-per-type captures never hit it.
    Mirrors mlir-aie's configure_trace, which increments packet_id per tile.
    """

    def test_single_compute_tile_starts_at_one(
        self, tmp_path, simple_design_mlir
    ):
        """One compute tile still gets id 1 (byte-identical to the old path)."""
        inp = tmp_path / "in.mlir"
        inp.write_text(simple_design_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject(["--input", str(inp), "--out", str(out)])
        assert rc == 0
        ids = _injected_packet_ids(out.read_text())
        assert ids == [1], f"expected single id [1], got {ids}"

    def test_multiple_sources_get_distinct_ids(
        self, tmp_path, multi_tile_mlir
    ):
        """All trace sources (2 core + 2 memmod + memtile + shim) get distinct
        packet ids -- no two flows collide on the shared shim channel."""
        inp = tmp_path / "in.mlir"
        inp.write_text(multi_tile_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject([
            "--input", str(inp), "--out", str(out),
            "--memtile-sweep-events", "all",
            "--shim-sweep-events", "all",
            "--memmod-sweep-events", "all",
        ])
        assert rc == 0
        ids = _injected_packet_ids(out.read_text())
        # 2 compute cores + 2 memmod + 1 memtile + 1 shim = 6 trace sources.
        assert len(ids) == 6, f"expected 6 trace sources, got {len(ids)}: {ids}"
        assert len(set(ids)) == len(ids), (
            f"packet ids must be unique per source; got duplicates: {ids}"
        )

    def test_two_compute_cores_distinct_without_extra_sources(
        self, tmp_path, multi_tile_mlir
    ):
        """The minimal regression: two same-type (core) tiles alone must not
        share id 1 -- the exact collision that masked the second core's trace."""
        inp = tmp_path / "in.mlir"
        inp.write_text(multi_tile_mlir)
        out = tmp_path / "out.mlir"
        rc = run_inject(["--input", str(inp), "--out", str(out)])
        assert rc == 0
        ids = _injected_packet_ids(out.read_text())
        assert ids == [1, 2], f"two cores must get ids [1, 2], got {ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
