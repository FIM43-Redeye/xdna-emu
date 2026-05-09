# SPDX-License-Identifier: MIT
"""Tests for tools/trace_decoder/.

The correctness contract for the byte-level decoder is bit-equivalent
agreement with the public mlir-aie ``convert_to_commands`` decoder
(Apache 2.0).  We freeze that oracle's output on a captured fixture and
diff our typed commands against it.

Fixtures live in ``tools/trace_decoder/fixtures/``:

* ``mode0_add_one_objfifo_r0.bin`` -- 87-word trimmed mode-0 capture
  from add_one_objFifo on NPU1, run 0.
* ``mode0_add_one_objfifo_r0.expected.json`` -- JSON dump of mlir-aie's
  ``convert_to_commands`` on the same buffer (regenerable, see
  ``regenerate-fixtures.sh`` in this directory).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sys

# Add tools/ to the import path so test_trace_decoder.py can `import trace_decoder`
sys.path.insert(0, str(Path(__file__).parent))

from trace_decoder import (
    EventCmd,
    PacketType,
    RepeatCmd,
    StartCmd,
    SyncCmd,
    TraceMode,
    decode_words,
    parse_packet_header,
)
from trace_decoder.packet import deinterleave_packets, words_to_bytes

FIXTURE_DIR = Path(__file__).parent / "trace_decoder" / "fixtures"


# ---------------------------------------------------------------------------
# packet header parsing
# ---------------------------------------------------------------------------


def test_parse_packet_header_valid_core_tile():
    # First word of mode0 fixture: pkt_type=0 (CORE), col=1, row=2.
    raw = np.fromfile(FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin", dtype=np.uint32)
    hdr = parse_packet_header(int(raw[0]))
    assert hdr is not None
    assert hdr.pkt_type == PacketType.CORE
    assert hdr.col == 1
    assert hdr.row == 2


def test_parse_packet_header_rejects_zero_word():
    assert parse_packet_header(0) is None


def test_parse_packet_header_rejects_bad_parity():
    # 0x00220003 has even parity (one extra bit set vs the valid 0x00220002).
    assert parse_packet_header(0x00220003) is None


# ---------------------------------------------------------------------------
# de-interleaving
# ---------------------------------------------------------------------------


def test_deinterleave_groups_by_tile():
    raw = np.fromfile(FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin", dtype=np.uint32)
    by_tile = deinterleave_packets(raw.tolist())
    # The fixture has CORE traffic at (col=1, row=2) and MEMTILE traffic
    # at (col=1, row=1); both must appear and only those two.
    assert (PacketType.CORE, 2, 1) in by_tile
    assert (PacketType.MEMTILE, 1, 1) in by_tile
    assert len(by_tile) == 2


# ---------------------------------------------------------------------------
# mode-0 byte-level decode -- bit-equivalence with mlir-aie oracle
# ---------------------------------------------------------------------------


def _cmd_to_oracle_dict(cmd) -> dict:
    """Convert one of our typed commands to mlir-aie's dict form.

    The oracle distinguishes Single0/1/2 and Multiple0/1/2 by the size
    of the cycles field; we collapse those into a single EventCmd shape
    in our schema, but for comparison we re-derive the oracle's tag
    from the cycles range and the pop-count of the event mask.
    """
    if isinstance(cmd, StartCmd):
        return {"type": "Start", "timer_value": cmd.timer_value}
    if isinstance(cmd, SyncCmd):
        return {"type": "Event_Sync"}
    if isinstance(cmd, RepeatCmd):
        # The oracle picks Repeat0 (4-bit) or Repeat1 (10-bit) based on
        # encoded width.  Our decoder doesn't preserve that distinction;
        # we test that *any* Repeat with the same count matches.
        return {"type": "Repeat", "repeats": cmd.count}
    if isinstance(cmd, EventCmd):
        bits_set = bin(cmd.event_bits).count("1")
        if bits_set == 1:
            slot = (cmd.event_bits & -cmd.event_bits).bit_length() - 1
            return {"type": "Single", "event": slot, "cycles": cmd.cycles}
        out = {"type": "Multiple", "cycles": cmd.cycles}
        for i in range(8):
            if cmd.event_bits & (1 << i):
                out[f"event{i}"] = i
        return out
    raise AssertionError(f"unhandled command: {cmd!r}")


def _oracle_to_canonical(d: dict) -> dict:
    """Collapse the oracle's Single0/1/2 and Repeat0/1 size variants
    into the same canonical shape ``_cmd_to_oracle_dict`` produces.

    Single* -> {"type": "Single", "event", "cycles"}
    Multiple* -> {"type": "Multiple", "cycles", "event0".."event7"}
    Repeat* -> {"type": "Repeat", "repeats"}
    Start -> unchanged
    Event_Sync -> unchanged
    """
    t = d["type"]
    if t.startswith("Single"):
        return {"type": "Single", "event": d["event"], "cycles": d["cycles"]}
    if t.startswith("Multiple"):
        out = {"type": "Multiple", "cycles": d["cycles"]}
        for i in range(8):
            k = f"event{i}"
            if k in d:
                out[k] = d[k]
        return out
    if t.startswith("Repeat"):
        return {"type": "Repeat", "repeats": d["repeats"]}
    return dict(d)


@pytest.fixture(scope="module")
def oracle_commands():
    path = FIXTURE_DIR / "mode0_add_one_objfifo_r0.expected.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def fixture_words():
    raw = np.fromfile(FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin", dtype=np.uint32)
    return raw.tolist()


def test_mode0_decode_matches_oracle_byte_for_byte(fixture_words, oracle_commands):
    """Our mode-0 decoder must produce the same command sequence per
    tile as mlir-aie's ``convert_to_commands``.  This is the hard
    correctness contract for the package."""
    ours_per_tile = decode_words(fixture_words, mode=TraceMode.EVENT_TIME)

    # Oracle is keyed as: trace_types[pkt_type_int][f"{row},{col}"] = [cmd, ...]
    oracle = oracle_commands["trace_types"]

    # Same set of (pkt_type, row, col) keys on both sides.
    ours_keys = {(pt, r, c) for (pt, r, c) in ours_per_tile.keys()}
    oracle_keys = set()
    for pt, by_loc in enumerate(oracle):
        for loc in by_loc:
            r, c = (int(x) for x in loc.split(","))
            oracle_keys.add((pt, r, c))
    assert ours_keys == oracle_keys, (ours_keys, oracle_keys)

    # Per-tile command sequences must agree after canonicalization.
    for (pt, r, c), our_cmds in ours_per_tile.items():
        oracle_cmds_raw = oracle[pt][f"{r},{c}"]
        oracle_canonical = [_oracle_to_canonical(d) for d in oracle_cmds_raw]
        ours_canonical = [_cmd_to_oracle_dict(cmd) for cmd in our_cmds]
        assert ours_canonical == oracle_canonical, (
            f"mismatch at tile pkt_type={pt} row={r} col={c}\n"
            f"ours[:5]={ours_canonical[:5]}\n"
            f"oracle[:5]={oracle_canonical[:5]}\n"
            f"len ours={len(ours_canonical)} oracle={len(oracle_canonical)}"
        )


def test_mode0_command_count_per_tile(fixture_words, oracle_commands):
    """Sanity: total command count per tile matches the oracle.

    Cheaper than the full byte-for-byte test; useful as a fast
    early-warning signal during decoder development.
    """
    ours = decode_words(fixture_words, mode=TraceMode.EVENT_TIME)
    oracle = oracle_commands["trace_types"]
    for (pt, r, c), our_cmds in ours.items():
        oracle_n = len(oracle[pt][f"{r},{c}"])
        assert len(our_cmds) == oracle_n, (pt, r, c, len(our_cmds), oracle_n)


# ---------------------------------------------------------------------------
# Mode 1 (EVENT_PC) decode -- frozen-fixture regression
# ---------------------------------------------------------------------------
#
# There is no public oracle for mode-1: mlir-aie's decoder is mode-0 only,
# and the AMD libevent_trace_decoder.so we used as a structural reference
# is read-only / not redistributable.  We instead freeze a known-good
# decode of a captured trace and assert against it; the freeze can be
# regenerated by ``regenerate-fixtures.sh`` whenever the schema changes.
# The fixture is a real on-tile capture (CORE tile in mode 1, MEMTILE in
# mode 0) so the test exercises the per-tile mode mismatch path too.
#
# CORE tile sanity anchors:
#   - First command is StartCmd(timer_value=0x4e5f7).
#   - First EventCmd is event_bits=0x80, pc=0x330 (event slot 7 at PC 816).
#   - Final commands trail off into PC=758,762,766,770,774 (sequential
#     fetch through a basic block on a 4-byte instruction stride).


@pytest.fixture(scope="module")
def mode1_fixture_words():
    raw = np.fromfile(FIXTURE_DIR / "mode1_mixed_r0.bin", dtype=np.uint32)
    return raw.tolist()


@pytest.fixture(scope="module")
def mode1_core_expected():
    path = FIXTURE_DIR / "mode1_mixed_r0_core_expected.json"
    return json.loads(path.read_text())


def _mode1_cmd_to_dict(cmd) -> dict:
    if isinstance(cmd, StartCmd):
        return {"type": "Start", "timer_value": cmd.timer_value}
    if isinstance(cmd, SyncCmd):
        return {"type": "Sync"}
    if isinstance(cmd, RepeatCmd):
        return {"type": "Repeat", "count": cmd.count}
    if isinstance(cmd, EventCmd):
        return {"type": "EventPC", "event_bits": cmd.event_bits, "pc": cmd.cycles}
    raise AssertionError(f"unhandled command: {cmd!r}")


def test_mode1_core_tile_decode_matches_freeze(
    mode1_fixture_words, mode1_core_expected
):
    """Mode-1 CORE tile decode must agree with the frozen JSON fixture."""
    by_tile = decode_words(mode1_fixture_words, mode=TraceMode.EVENT_PC)
    core_key = (PacketType.CORE, 2, 1)
    assert core_key in by_tile, by_tile.keys()
    ours = [_mode1_cmd_to_dict(c) for c in by_tile[core_key]]
    assert ours == mode1_core_expected, (
        f"len ours={len(ours)} expected={len(mode1_core_expected)}\n"
        f"first 5 ours={ours[:5]}\nfirst 5 expected={mode1_core_expected[:5]}"
    )


def test_mode1_eventpc_bit_layout():
    """Synthetic single-word check on the EventPC bit layout.

    Word 0xC4200150 should decode to mask=0x08, pc=0x0150 -- worked out
    by hand from the bit pattern documented in modes/mode1.py.
    """
    from trace_decoder.modes.mode1 import decode

    bytes_ = [0xC4, 0x20, 0x01, 0x50]
    cmds = list(decode(bytes_))
    assert len(cmds) == 1
    assert isinstance(cmds[0], EventCmd)
    assert cmds[0].event_bits == 0x08
    assert cmds[0].cycles == 0x0150  # PC stored in the cycles slot


def test_mode1_start_opcode_distinguishes_from_mode0():
    """Mode-1 Start (0xF1) must decode where mode-0 Start (0xF0) doesn't.

    The 0xF0 vs 0xF1 distinction is the trace-mode discriminator bit
    (bit 0 of the opcode byte); a clean mode-1 stream must accept the
    F1-prefixed Start and not silently fall through to a no-op.
    """
    from trace_decoder.modes.mode1 import decode

    bytes_ = [0xF1, 0, 0, 0, 0, 0, 0, 42]  # timer = 42
    cmds = list(decode(bytes_))
    assert len(cmds) == 1
    assert isinstance(cmds[0], StartCmd)
    assert cmds[0].timer_value == 42


# ---------------------------------------------------------------------------
# Mode 2 (INST_EXEC) decode -- frozen-fixture regression
# ---------------------------------------------------------------------------
#
# Mode 2 has no public oracle either; the freeze captures the current
# decoder output of a real on-tile capture.  CORE tile sanity anchors:
#   - StartCmd anchor_pc = 0x330 (= 816, matches Start word's low-14
#     bits in the capture).
#   - Two repeating sequences of New_PC at 0x150, 0x170, 0x1E0, 0x200,
#     0x220, 0x240, 0x2B0, 0x2D0 -- a tight loop body with 7-8 taken
#     branches.
#   - A few E_atom (executed) and N_atom (stalled) cycles trailing.


@pytest.fixture(scope="module")
def mode2_fixture_words():
    raw = np.fromfile(FIXTURE_DIR / "mode2_mixed_r0.bin", dtype=np.uint32)
    return raw.tolist()


@pytest.fixture(scope="module")
def mode2_core_expected():
    path = FIXTURE_DIR / "mode2_mixed_r0_core_expected.json"
    return json.loads(path.read_text())


def _mode2_cmd_to_dict(cmd) -> dict:
    from trace_decoder.modes.mode2 import CycleCmd, LoopCountCmd

    if isinstance(cmd, StartCmd):
        return {"type": "Start", "anchor_pc": cmd.timer_value}
    if isinstance(cmd, CycleCmd):
        return {"type": "N_atom" if cmd.stalled else "E_atom"}
    if isinstance(cmd, EventCmd):
        return {"type": "New_PC", "pc": cmd.cycles}
    if isinstance(cmd, RepeatCmd):
        return {"type": "Repeat", "count": cmd.count}
    if isinstance(cmd, LoopCountCmd):
        return {"type": "LC", "flag": cmd.flag, "count": cmd.count}
    if isinstance(cmd, SyncCmd):
        return {"type": "Sync"}
    raise AssertionError(f"unhandled mode2 cmd: {cmd!r}")


def test_mode2_core_tile_decode_matches_freeze(
    mode2_fixture_words, mode2_core_expected
):
    """Mode-2 CORE tile decode must agree with the frozen JSON fixture."""
    by_tile = decode_words(mode2_fixture_words, mode=TraceMode.INST_EXEC)
    core_key = (PacketType.CORE, 2, 1)
    assert core_key in by_tile, by_tile.keys()
    ours = [_mode2_cmd_to_dict(c) for c in by_tile[core_key]]
    assert ours == mode2_core_expected, (
        f"len ours={len(ours)} expected={len(mode2_core_expected)}\n"
        f"first 5 ours={ours[:5]}\nfirst 5 expected={mode2_core_expected[:5]}"
    )


# ---------------------------------------------------------------------------
# Per-tile auto-detect: real captures mix tile modes (CORE in mode 1/2,
# MEMTILE in mode 0).  decode_words(mode=None) must dispatch per tile
# based on each Start opcode's low-2-bit discriminator.
# ---------------------------------------------------------------------------


def test_detect_per_tile_modes_mixed_capture():
    """The mode-1 fixture has CORE in EVENT_PC and MEMTILE in EVENT_TIME."""
    from trace_decoder import detect_per_tile_modes

    raw = np.fromfile(FIXTURE_DIR / "mode1_mixed_r0.bin", dtype=np.uint32)
    modes = detect_per_tile_modes(raw.tolist())
    assert modes[(PacketType.CORE, 2, 1)] == TraceMode.EVENT_PC
    assert modes[(PacketType.MEMTILE, 1, 1)] == TraceMode.EVENT_TIME


def test_decode_words_auto_picks_per_tile_mode():
    """decode_words(mode=None) must decode each tile with its own mode.

    The CORE tile's first command should be a StartCmd whose timer
    matches the mode-1 fixture's known anchor (0x4e5f7), AND the
    MEMTILE tile (mode 0) must produce an EventCmd-shaped command
    sequence (rather than nonsense).
    """
    from trace_decoder.modes.mode2 import CycleCmd  # noqa: F401 (sentinel)

    raw = np.fromfile(FIXTURE_DIR / "mode1_mixed_r0.bin", dtype=np.uint32)
    by_tile = decode_words(raw.tolist(), mode=None)

    core_cmds = by_tile[(PacketType.CORE, 2, 1)]
    assert isinstance(core_cmds[0], StartCmd)
    assert core_cmds[0].timer_value == 0x4E5F7

    memtile_cmds = by_tile[(PacketType.MEMTILE, 1, 1)]
    assert isinstance(memtile_cmds[0], StartCmd)
    # Mode-0 timer is the 56-bit anchor read off the F0 prefix.
    assert memtile_cmds[0].timer_value > 0
    # Mode-0 has cycle-delta EventCmds; mode-1 EventCmds reuse the
    # cycles slot for PCs (max 0x3FFF = 16383).  A non-trivial mode-0
    # timer will exceed 16383, so this gives us a coarse "decoded as
    # mode 0, not mode 1" sanity check.
    assert memtile_cmds[0].timer_value > 16383


def test_parse_trace_auto_emits_events_for_both_modes():
    """parse_trace_auto on the mixed mode-1 fixture must surface events
    from BOTH the EVENT_PC core tile and the EVENT_TIME memtile tile.

    Single-mode parse_trace cannot do this: forcing mode=EVENT_PC
    misdecodes the memtile, and mode=EVENT_TIME misdecodes the core.
    parse_trace_auto picks each tile's mode from its Start opcode and
    rebuilds the timeline accordingly, so the returned Event list
    covers both.
    """
    from trace_decoder import parse_trace_auto

    raw = np.fromfile(FIXTURE_DIR / "mode1_mixed_r0.bin", dtype=np.uint32)
    events = parse_trace_auto(raw.tolist())

    pkt_types_with_events = {e.pkt_type for e in events}
    # CORE = 0 (EVENT_PC) and MEMTILE = 3 (EVENT_TIME) must both
    # contribute events.  Earlier per-tile-mode tests confirm the
    # fixture has those two tiles configured that way.
    assert PacketType.CORE in pkt_types_with_events, (
        f"expected CORE events, got pkt_types={pkt_types_with_events}"
    )
    assert PacketType.MEMTILE in pkt_types_with_events, (
        f"expected MEMTILE events, got pkt_types={pkt_types_with_events}"
    )


def test_decode_words_auto_skips_no_start():
    """Tiles with no recognisable Start opcode auto-detect to skip
    rather than crash the whole decode."""
    # Construct a fake one-tile word stream with no Start byte:
    # pkt header at word[0], all-FE filler in words[1..7].
    # Header: pkt_type=CORE, col=1, row=2 -- valid parity.  We borrow
    # the value from the mode-0 fixture so we don't recompute parity.
    raw = np.fromfile(FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin", dtype=np.uint32)
    header = int(raw[0])  # known-good CORE header
    bogus = [header] + [0xFEFEFEFE] * 7
    by_tile = decode_words(bogus, mode=None)
    # Either the tile silently empties out (no Start, skip) or the
    # decoder finds no tile at all -- both are acceptable; the key
    # assertion is that no exception bubbled up.
    for cmds in by_tile.values():
        assert cmds == []


def test_mode2_frame_tree_synthetic():
    """Hand-built words exercise each frame's bit pattern.

    Word 0xF2003030: Start prefix 11110 + bit26 flag (0) + ...
                     anchor PC = low 14 bits = 0x0330 = 816.
    Word 0x81708160: New_PC at 0x0170 (368), then New_PC at 0x0160 (352).
    """
    from trace_decoder.modes.mode2 import CycleCmd
    from trace_decoder.modes.mode2 import decode

    # 0xF2003030 0x81708160 packed MSB-first.
    bytes_ = [0xF2, 0x00, 0x03, 0x30, 0x81, 0x70, 0x81, 0x60]
    cmds = list(decode(bytes_))
    assert isinstance(cmds[0], StartCmd) and cmds[0].timer_value == 0x330
    assert isinstance(cmds[1], EventCmd) and cmds[1].cycles == 0x170
    assert isinstance(cmds[2], EventCmd) and cmds[2].cycles == 0x160


# ---------------------------------------------------------------------------
# parse-trace.py integration: --decoder=ours produces oracle-equivalent
# --out-commands JSON
# ---------------------------------------------------------------------------


def test_parse_trace_cli_out_commands_matches_oracle(tmp_path):
    """End-to-end: parse-trace.py --decoder=ours --out-commands must
    produce a file byte-identical to the regenerated oracle JSON.

    Skipped when the mlir-aie ironenv Python is not available since
    parse-trace.py imports aie.utils for slot-name lookup; the byte
    decoder itself does not need it.
    """
    import os
    import shutil
    import subprocess

    iron_py = "/home/triple/npu-work/mlir-aie/ironenv/bin/python3"
    if not os.path.isfile(iron_py):
        pytest.skip("ironenv python not available")
    script = Path(__file__).parent / "parse-trace.py"
    fixture = FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin"
    expected = FIXTURE_DIR / "mode0_add_one_objfifo_r0.expected.json"
    out = tmp_path / "ours_commands.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        "/home/triple/npu-work/mlir-aie/install/python:" + env.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [
            iron_py,
            str(script),
            "--trace-bin",
            str(fixture),
            "--xclbin-mlir",
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/peano/aie_arch.mlir.prj/input_with_addresses.mlir",
            "--decoder",
            "ours",
            "--out-commands",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_text() == expected.read_text()


def test_parse_trace_cli_out_perfetto_mode0_emits_be_pairs(tmp_path):
    """--decoder=ours --out-perfetto on a mode-0 capture must emit the
    Chrome-trace JSON shape mlir-aie's path produces: M/process_name +
    M/thread_name metadata, then B/E pairs whose timestamps lie in a
    sane range and whose pids match the metadata.
    """
    import os
    import subprocess

    iron_py = "/home/triple/npu-work/mlir-aie/ironenv/bin/python3"
    if not os.path.isfile(iron_py):
        pytest.skip("ironenv python not available")
    script = Path(__file__).parent / "parse-trace.py"
    fixture = FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin"
    out = tmp_path / "ours_perfetto.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        "/home/triple/npu-work/mlir-aie/install/python:" + env.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [
            iron_py,
            str(script),
            "--trace-bin",
            str(fixture),
            "--xclbin-mlir",
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/peano/aie_arch.mlir.prj/input_with_addresses.mlir",
            "--decoder",
            "ours",
            "--out-perfetto",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    events = json.loads(out.read_text())
    assert isinstance(events, list) and len(events) > 0
    # Must contain at least one process_name M event and one B-phase event.
    has_proc = any(e.get("ph") == "M" and e.get("name") == "process_name" for e in events)
    has_b = any(e.get("ph") == "B" for e in events)
    has_e = any(e.get("ph") == "E" for e in events)
    assert has_proc and has_b and has_e, [e.get("ph") for e in events[:5]]
    # B/E pairs should balance per (pid, tid).
    from collections import Counter
    open_count = Counter()
    for e in events:
        if e.get("ph") == "B":
            open_count[(e["pid"], e["tid"])] += 1
        elif e.get("ph") == "E":
            open_count[(e["pid"], e["tid"])] -= 1
    assert all(v == 0 for v in open_count.values()), open_count


def test_parse_trace_cli_out_perfetto_rejects_non_mode0(tmp_path):
    """--decoder=ours --out-perfetto for mode 1 or 2 must fail loudly
    until those timeline rebuilds land, so callers don't silently get a
    malformed timeline."""
    import os
    import subprocess

    iron_py = "/home/triple/npu-work/mlir-aie/ironenv/bin/python3"
    if not os.path.isfile(iron_py):
        pytest.skip("ironenv python not available")
    script = Path(__file__).parent / "parse-trace.py"
    fixture = FIXTURE_DIR / "mode1_mixed_r0.bin"
    out = tmp_path / "should_not_exist.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        "/home/triple/npu-work/mlir-aie/install/python:" + env.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [
            iron_py,
            str(script),
            "--trace-bin",
            str(fixture),
            "--xclbin-mlir",
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/peano/aie_arch.mlir.prj/input_with_addresses.mlir",
            "--decoder",
            "ours",
            "--trace-mode",
            "event_pc",
            "--out-perfetto",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode != 0
    assert "event_pc" in proc.stderr.lower() or "perfetto" in proc.stderr.lower()
    assert not out.exists()
