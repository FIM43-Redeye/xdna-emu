"""Tests for tools/parse-trace.py.

Unit tests exercise the pure-Python event-restructuring helpers without
requiring mlir-aie. Integration tests feed a real HW/EMU binary + MLIR pair
through the full pipeline and are skipped if mlir-aie isn't importable.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "parse-trace.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures" / "vector_scalar"

# Load parse-trace as a module so we can unit-test its helpers. The script's
# filename has a hyphen, so we can't `import parse-trace` directly.
_spec = importlib.util.spec_from_file_location("parse_trace", TOOL)
parse_trace_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse_trace_mod)


def _has_mlir_aie():
    return importlib.util.find_spec("aie.utils.trace.parse") is not None


requires_mlir_aie = pytest.mark.skipif(
    not _has_mlir_aie(),
    reason="mlir-aie not importable; activate ironenv + PYTHONPATH",
)


# ---------------------------------------------------------------------------
# Unit tests: pid map recovery and flattening
# ---------------------------------------------------------------------------

def test_build_pid_map_core_tile():
    perfetto = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "core_trace for tile2,1"}},
    ]
    m = parse_trace_mod.build_pid_map(perfetto)
    assert m == {0: (0, 2, 1)}


def test_build_pid_map_all_tile_types():
    perfetto = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "core_trace for tile2,1"}},
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "mem_trace for tile2,1"}},
        {"name": "process_name", "ph": "M", "pid": 2,
         "args": {"name": "shim_trace for tile0,0"}},
        {"name": "process_name", "ph": "M", "pid": 3,
         "args": {"name": "memtile_trace for tile1,0"}},
    ]
    m = parse_trace_mod.build_pid_map(perfetto)
    assert m[0] == (0, 2, 1)
    assert m[1] == (1, 2, 1)
    assert m[2] == (2, 0, 0)
    assert m[3] == (3, 1, 0)


def test_build_pid_map_ignores_non_metadata():
    perfetto = [
        {"name": "INSTR_EVENT_0", "ph": "B", "pid": 0, "tid": 1, "ts": 100},
        {"name": "thread_name", "ph": "M", "pid": 0, "tid": 1,
         "args": {"name": "INSTR_EVENT_0"}},
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "unrelated"}},
    ]
    # No valid process_name → empty map.
    assert parse_trace_mod.build_pid_map(perfetto) == {}


def test_perfetto_to_events_keeps_only_begin():
    perfetto = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "core_trace for tile2,1"}},
        {"name": "INSTR_EVENT_0", "ph": "B", "pid": 0, "tid": 1, "ts": 100},
        {"name": "INSTR_EVENT_0", "ph": "E", "pid": 0, "tid": 1, "ts": 101},
        {"name": "INSTR_EVENT_1", "ph": "B", "pid": 0, "tid": 2, "ts": 200},
    ]
    flat = parse_trace_mod.perfetto_to_events(perfetto)
    assert len(flat) == 2
    assert flat[0] == {
        "col": 1, "row": 2, "pkt_type": 0, "slot": 1,
        "name": "INSTR_EVENT_0", "ts": 100,
    }
    assert flat[1]["ts"] == 200
    assert flat[1]["slot"] == 2


def test_perfetto_to_events_sorted_by_tile_then_ts():
    perfetto = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "core_trace for tile2,0"}},
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "core_trace for tile2,1"}},
        # Interleaved, out of order
        {"name": "E0", "ph": "B", "pid": 1, "tid": 1, "ts": 300},
        {"name": "E0", "ph": "B", "pid": 0, "tid": 1, "ts": 400},
        {"name": "E0", "ph": "B", "pid": 1, "tid": 1, "ts": 100},
        {"name": "E0", "ph": "B", "pid": 0, "tid": 1, "ts": 200},
    ]
    flat = parse_trace_mod.perfetto_to_events(perfetto)
    # Col 0 (tile2,0) first with its ts in order, then col 1 (tile2,1).
    assert [(e["col"], e["ts"]) for e in flat] == [
        (0, 200), (0, 400), (1, 100), (1, 300),
    ]


def test_perfetto_to_slot_names_extracts_per_type():
    perfetto = [
        {"name": "process_name", "ph": "M", "pid": 0,
         "args": {"name": "core_trace for tile2,1"}},
        {"name": "process_name", "ph": "M", "pid": 1,
         "args": {"name": "mem_trace for tile2,1"}},
        {"name": "thread_name", "ph": "M", "pid": 0, "tid": 0,
         "args": {"name": "INSTR_VECTOR"}},
        {"name": "thread_name", "ph": "M", "pid": 0, "tid": 1,
         "args": {"name": "INSTR_EVENT_0"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3,
         "args": {"name": "DMA_S2MM_0"}},
    ]
    names = parse_trace_mod.perfetto_to_slot_names(perfetto)
    assert names["core"][0] == "INSTR_VECTOR"
    assert names["core"][1] == "INSTR_EVENT_0"
    assert names["core"][2] == ""  # unset slot stays empty
    assert names["mem"][3] == "DMA_S2MM_0"
    assert names["shim"] == [""] * 8
    assert names["memtile"] == [""] * 8


def test_perfetto_to_events_drops_events_with_unknown_pid():
    # A B-event with a pid that has no process_name metadata is unroutable
    # and should be dropped rather than silently placed at col=0,row=0.
    perfetto = [
        {"name": "E0", "ph": "B", "pid": 42, "tid": 1, "ts": 100},
    ]
    assert parse_trace_mod.perfetto_to_events(perfetto) == []


# ---------------------------------------------------------------------------
# CLI/integration tests: require mlir-aie + real fixtures
# ---------------------------------------------------------------------------

def _run_cli(args, check=True):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, check=check,
    )


def test_cli_help():
    r = _run_cli(["--help"])
    assert "parse-trace.py" in r.stdout or "parse-trace.py" in r.stderr \
        or "usage" in (r.stdout + r.stderr).lower()


def test_cli_requires_at_least_one_output(tmp_path):
    fake = tmp_path / "t.bin"
    fake.write_bytes(b"\x00" * 32)
    fake_mlir = tmp_path / "m.mlir"
    fake_mlir.write_text("module {}")
    r = _run_cli(
        ["--trace-bin", str(fake), "--xclbin-mlir", str(fake_mlir)],
        check=False,
    )
    assert r.returncode != 0
    assert "out-" in r.stderr or "required" in r.stderr.lower()


@requires_mlir_aie
def test_cli_end_to_end_hw_fixture(tmp_path):
    """Full pipeline on the real vector_scalar HW trace: all four outputs."""
    events = tmp_path / "events.json"
    cycles = tmp_path / "cycles.txt"
    perfetto = tmp_path / "perfetto.json"
    commands = tmp_path / "commands.json"

    r = _run_cli([
        "--trace-bin", str(FIXTURES / "trace_hw.bin"),
        "--xclbin-mlir", str(FIXTURES / "input_with_addresses.mlir"),
        "--out-events", str(events),
        "--out-cycles", str(cycles),
        "--out-perfetto", str(perfetto),
        "--out-commands", str(commands),
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"

    # Cycles matches the known-good span for this fixture.
    assert cycles.read_text().strip() == "41181"

    # Events: 8 core events on tile (col=1, row=2), alternating slot 1/2.
    data = json.loads(events.read_text())
    assert data["schema_version"] == 1
    flat = data["events"]
    assert len(flat) == 8
    # slot_names present; core slot 1 maps to INSTR_EVENT_0 for this kernel.
    assert data["slot_names"]["core"][1] == "INSTR_EVENT_0"
    assert data["slot_names"]["core"][2] == "INSTR_EVENT_1"
    assert all(e["col"] == 1 and e["row"] == 2 and e["pkt_type"] == 0
               for e in flat)
    slots = [e["slot"] for e in flat]
    assert set(slots) == {1, 2}
    # Inner-loop period is ~10312 cycles (HW has micro-drift of a few cycles).
    slot1_ts = [e["ts"] for e in flat if e["slot"] == 1]
    periods = [b - a for a, b in zip(slot1_ts, slot1_ts[1:])]
    assert all(10300 <= p <= 10320 for p in periods), periods

    # Perfetto and commands outputs are valid JSON; we don't pin their shape
    # because it's mlir-aie's contract, but they must be non-trivial.
    pjson = json.loads(perfetto.read_text())
    assert isinstance(pjson, list) and len(pjson) > 0
    cjson = json.loads(commands.read_text())
    assert "trace_types" in cjson


@requires_mlir_aie
def test_cli_end_to_end_emu_fixture_matches_hw_structure(tmp_path):
    """EMU trace has the same event structure (8 events, slot 1/2 alternating)
    as the HW trace; this is the invariant trace-compare relies on."""
    events_hw = tmp_path / "hw.events.json"
    events_emu = tmp_path / "emu.events.json"
    for side, bin_name, out in [
        ("hw", "trace_hw.bin", events_hw),
        ("emu", "trace_emu.bin", events_emu),
    ]:
        _run_cli([
            "--trace-bin", str(FIXTURES / bin_name),
            "--xclbin-mlir", str(FIXTURES / "input_with_addresses.mlir"),
            "--out-events", str(out),
        ])
    hw = json.loads(events_hw.read_text())["events"]
    emu = json.loads(events_emu.read_text())["events"]
    assert len(hw) == len(emu) == 8
    # Same slot sequence on both sides.
    assert [e["slot"] for e in hw] == [e["slot"] for e in emu]
    # Note: col differs (HW=1, EMU=0) -- that's the start_col offset
    # trace-compare handles with --remap-columns.
