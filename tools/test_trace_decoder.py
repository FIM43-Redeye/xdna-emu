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
# Mode 1 / 2 stubs are not yet implemented -- guard against silent decode
# ---------------------------------------------------------------------------


def test_mode1_raises_not_implemented(fixture_words):
    with pytest.raises(NotImplementedError):
        decode_words(fixture_words, mode=TraceMode.EVENT_PC)


def test_mode2_raises_not_implemented(fixture_words):
    with pytest.raises(NotImplementedError):
        decode_words(fixture_words, mode=TraceMode.INST_EXEC)


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


def test_parse_trace_cli_out_perfetto_rejects_ours(tmp_path):
    """--decoder=ours --out-perfetto must fail loudly until we port
    the B/E pair generation, so callers don't silently get a missing
    output file."""
    import os
    import subprocess

    iron_py = "/home/triple/npu-work/mlir-aie/ironenv/bin/python3"
    if not os.path.isfile(iron_py):
        pytest.skip("ironenv python not available")
    script = Path(__file__).parent / "parse-trace.py"
    fixture = FIXTURE_DIR / "mode0_add_one_objfifo_r0.bin"
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
            "--out-perfetto",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode != 0
    assert "perfetto" in proc.stderr.lower()
    assert not out.exists()
