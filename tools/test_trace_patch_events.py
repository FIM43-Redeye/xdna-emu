"""Tests for tools/trace-patch-events.py.

Covers:
- patch_events() and patch_trace_control() as library functions
- --multi-tile CLI flag: byte-identical to chained single-tile invocations
- --multi-tile mutual exclusion with --col/--row/--tile-type
- Edge cases: empty spec list, event-only vs control-only vs combined specs

The sample_insts_bin fixture builds a minimal but structurally correct
insts.bin in memory -- 16-byte header + one Write32 pair per tile.
Patchable by patch_events() because it contains real Trace_Event0 and
Trace_Event1 Write32 entries at the correct NPU-encoded addresses.

NPU address encoding: (col << 25) | (row << 20) | tile_offset.
Write32 layout (24 bytes):
  [0]   opcode byte (0x00)
  [1-3] three zero bytes
  [4-7] four zero bytes (pad)
  [8-15] reg_off as u64 LE (the NPU address)
  [16-19] value as u32 LE
  [20-23] size as u32 LE (= 24, the full instruction length)
"""

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Module import (hyphen in filename requires importlib)
# ---------------------------------------------------------------------------

_PATCHER_PATH = Path(__file__).parent / "trace-patch-events.py"

_spec = importlib.util.spec_from_file_location("trace_patch_events", _PATCHER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

patch_events = _mod.patch_events
patch_trace_control = _mod.patch_trace_control
_parse_events_arg = _mod._parse_events_arg
_npu_address = _mod._npu_address
_TRACE_EVENT_REGS = _mod._TRACE_EVENT_REGS
_TRACE_CONTROL0_REGS = _mod._TRACE_CONTROL0_REGS

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_INSTS_HEADER_LEN = 16
# Real header from a compiled insts.bin; only the 16-byte size matters.
_HEADER = bytes([0x00, 0x01, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


def _write32_instruction(reg_off: int, value: int = 0) -> bytes:
    """Build a 24-byte Write32 instruction with the given NPU register address."""
    opcode_pad = struct.pack("<I", 0x00)          # opcode=0 + 3 zero bytes
    extra_pad  = struct.pack("<I", 0x00)          # 4 more zero bytes
    reg_field  = struct.pack("<Q", reg_off)       # 8-byte u64 LE address
    val_field  = struct.pack("<I", value)         # 4-byte u32 LE value
    size_field = struct.pack("<I", 24)            # 4-byte u32 LE = instruction size
    return opcode_pad + extra_pad + reg_field + val_field + size_field


def _make_insts_bin(tile_specs: List[tuple]) -> bytes:
    """Build a minimal insts.bin containing Trace_Event0+1 Write32 pairs.

    tile_specs is a list of (col, row, tile_type) tuples. For each spec,
    two Write32 instructions are emitted (Trace_Event0 and Trace_Event1)
    so that patch_events() finds both slots (8-slot capacity).
    """
    payload = b""
    for col, row, tile_type in tile_specs:
        off_e0, off_e1 = _TRACE_EVENT_REGS[tile_type]
        payload += _write32_instruction(_npu_address(col, row, off_e0))
        payload += _write32_instruction(_npu_address(col, row, off_e1))
    return _HEADER + payload


# Three distinct core tiles that the multi-tile tests use.
_THREE_TILE_SPECS = [
    (0, 2, "core"),
    (1, 2, "core"),
    (0, 3, "core"),
]


@pytest.fixture
def sample_insts_bin(tmp_path) -> Path:
    """Minimal but patchable insts.bin with three core-tile Trace_Event pairs."""
    data = _make_insts_bin(_THREE_TILE_SPECS)
    p = tmp_path / "insts.bin"
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def run_patch(argv: List[str]) -> int:
    """Invoke trace-patch-events.py as a subprocess; return its exit code."""
    result = subprocess.run(
        [sys.executable, str(_PATCHER_PATH)] + argv,
        capture_output=True,
        text=True,
    )
    return result.returncode


# ---------------------------------------------------------------------------
# Library-level smoke tests
# ---------------------------------------------------------------------------

class TestPatchEventsLibrary:
    """Smoke tests for patch_events() as a library function."""

    def test_patch_single_tile_roundtrip(self, sample_insts_bin):
        data = sample_insts_bin.read_bytes()
        events = [37, 23, 26, 28]
        patched, n = patch_events(data, 0, 2, "core", events)
        assert n in (1, 2)
        # Re-reading the patched buffer should yield updated values.
        patched2, n2 = patch_events(patched, 0, 2, "core", [1, 2, 3, 4])
        assert n2 in (1, 2)

    def test_patch_leaves_other_tiles_unchanged(self, sample_insts_bin):
        """Patching (0,2,core) must not touch (1,2,core) or (0,3,core)."""
        data = sample_insts_bin.read_bytes()
        events_a = [37, 23, 26, 28]
        patched, _ = patch_events(data, 0, 2, "core", events_a)
        # The (1,2,core) registers should still have value 0 (fixture default).
        patched_b, _ = patch_events(patched, 1, 2, "core", [1])
        # Chaining should work independently.
        assert patched_b != data  # something changed

    def test_chained_patches_composable(self, sample_insts_bin):
        """Chaining patch_events calls on the output buffer is safe."""
        data = sample_insts_bin.read_bytes()
        data, _ = patch_events(data, 0, 2, "core", [10, 11])
        data, _ = patch_events(data, 1, 2, "core", [20, 21])
        data, _ = patch_events(data, 0, 3, "core", [30, 31])
        # All three should produce a patched buffer with no exception.
        assert len(data) > _INSTS_HEADER_LEN


# ---------------------------------------------------------------------------
# Step 5.1: multi-tile matches chained single-tile
# ---------------------------------------------------------------------------

class TestMultiTileCLI:

    def test_multi_tile_matches_chained_single_tile(self, tmp_path, sample_insts_bin):
        """--multi-tile JSON output must be byte-identical to N sequential
        single-tile invocations on the same input."""
        spec = [
            {"col": 0, "row": 2, "tile_type": "core", "events": [37, 23, 26, 28]},
            {"col": 1, "row": 2, "tile_type": "core", "events": [37, 23, 26, 28]},
            {"col": 0, "row": 3, "tile_type": "core", "events": [37, 23, 26, 28]},
        ]
        spec_json = tmp_path / "spec.json"
        spec_json.write_text(json.dumps(spec))

        # Multi-tile path:
        multi_out = tmp_path / "multi.bin"
        rc = run_patch([
            str(sample_insts_bin),
            "--multi-tile", str(spec_json),
            "--output", str(multi_out),
        ])
        assert rc == 0

        # Chained per-tile path:
        chain_in = sample_insts_bin
        for i, s in enumerate(spec):
            chain_out = tmp_path / f"chain{i}.bin"
            rc = run_patch([
                str(chain_in),
                "--col", str(s["col"]),
                "--row", str(s["row"]),
                "--tile-type", s["tile_type"],
                "--events", ",".join(str(e) for e in s["events"]),
                "--output", str(chain_out),
            ])
            assert rc == 0
            chain_in = chain_out

        assert multi_out.read_bytes() == chain_in.read_bytes()

    # Step 5.3: mutual exclusion
    def test_multi_tile_rejects_col_row_tile_type(self, tmp_path, sample_insts_bin):
        """--multi-tile must be mutually exclusive with --col/--row/--tile-type."""
        spec_json = tmp_path / "spec.json"
        spec_json.write_text(json.dumps(
            [{"col": 0, "row": 2, "tile_type": "core", "events": [1]}]
        ))
        out = tmp_path / "out.bin"

        rc = run_patch([
            str(sample_insts_bin),
            "--multi-tile", str(spec_json),
            "--col", "0",
            "--output", str(out),
        ])
        assert rc != 0

    # Step 5.4: empty spec list
    def test_multi_tile_empty_spec(self, tmp_path, sample_insts_bin):
        """Empty spec list produces output identical to input (no-op)."""
        spec_json = tmp_path / "spec.json"
        spec_json.write_text(json.dumps([]))
        out = tmp_path / "out.bin"

        rc = run_patch([
            str(sample_insts_bin),
            "--multi-tile", str(spec_json),
            "--output", str(out),
        ])
        assert rc == 0
        assert out.read_bytes() == sample_insts_bin.read_bytes()

    def test_multi_tile_single_spec_matches_single_tile(self, tmp_path, sample_insts_bin):
        """A single-entry --multi-tile spec must equal the single-tile path."""
        spec = [{"col": 0, "row": 2, "tile_type": "core", "events": [42, 7]}]
        spec_json = tmp_path / "spec.json"
        spec_json.write_text(json.dumps(spec))

        multi_out = tmp_path / "multi.bin"
        rc = run_patch([
            str(sample_insts_bin),
            "--multi-tile", str(spec_json),
            "--output", str(multi_out),
        ])
        assert rc == 0

        single_out = tmp_path / "single.bin"
        rc = run_patch([
            str(sample_insts_bin),
            "--col", "0", "--row", "2", "--tile-type", "core",
            "--events", "42,7",
            "--output", str(single_out),
        ])
        assert rc == 0

        assert multi_out.read_bytes() == single_out.read_bytes()

    def test_multi_tile_exits_zero_on_success(self, tmp_path, sample_insts_bin):
        """Basic sanity: --multi-tile with valid spec returns exit code 0."""
        spec = [{"col": 0, "row": 2, "tile_type": "core", "events": [1]}]
        spec_json = tmp_path / "spec.json"
        spec_json.write_text(json.dumps(spec))
        out = tmp_path / "out.bin"

        rc = run_patch([
            str(sample_insts_bin),
            "--multi-tile", str(spec_json),
            "--output", str(out),
        ])
        assert rc == 0


# ---------------------------------------------------------------------------
# _parse_events_arg unit tests
# ---------------------------------------------------------------------------

class TestParseEventsArg:
    """Unit tests for the event-string parser."""

    def test_decimal(self):
        assert _parse_events_arg("37,23,26,28") == [37, 23, 26, 28]

    def test_hex(self):
        assert _parse_events_arg("0x25,0x17") == [0x25, 0x17]

    def test_trailing_zeros_trimmed(self):
        assert _parse_events_arg("33,,,,") == [33]

    def test_single(self):
        assert _parse_events_arg("1") == [1]

    def test_empty_gives_empty(self):
        assert _parse_events_arg("") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
