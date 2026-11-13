# SPDX-License-Identifier: MIT
"""Tests for tools/mem_core_offset.py: the memory-minus-core offset, emulator against silicon."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mem_core_offset as M  # noqa: E402


def write_side(root, name, events, origin=(1, 0)):
    d = root / name
    d.mkdir(parents=True)
    doc = {"placement": {"origin_col": origin[0], "origin_row": origin[1]},
           "events": [{"col": c + origin[0], "row": r + origin[1], "pkt_type": k, "name": n, "ts": t}
                      for (c, r, k, n, t) in events]}
    (d / "events.json").write_text(json.dumps(doc))


def rows(core_shift, mem_shift, base=1000):
    # core rows and memory rows each seen 3 times; timestamps shifted per unit
    out = []
    for i in range(3):
        out.append((0, 2, M.CORE, "INSTR_EVENT_0", base + 10 * i + core_shift))
        out.append((0, 2, M.MEM, "DMA_S2MM_0_START_TASK", base + 5 + 10 * i + mem_shift))
    return out


def test_offset_cancels_common_host_shift_and_reads_unit_difference(tmp_path):
    emu, hw = tmp_path / "emu", tmp_path / "hw"
    # silicon starts 300000 cycles later than the emulator on both units
    write_side(emu, "t.chess.emu", rows(0, 1))
    write_side(hw, "t.chess.hw", rows(300000, 300000))
    res = M.measure(emu, [hw])
    assert res == {("t.chess", (0, 2)): 1}  # emulator memory unit 1 cycle later than core, vs silicon


def test_median_over_silicon_samples(tmp_path):
    emu = tmp_path / "emu"
    write_side(emu, "t.chess.emu", rows(0, 2))
    hws = []
    for i, mem_hw in enumerate((0, 1, 1)):
        d = tmp_path / f"hw{i}"
        write_side(d, "t.chess.hw", rows(0, mem_hw))
        hws.append(d)
    assert M.measure(emu, hws)[("t.chess", (0, 2))] == 1


def test_level_rows_and_unequal_counts_are_ignored(tmp_path):
    emu, hw = tmp_path / "emu", tmp_path / "hw"
    ev = rows(0, 0) + [(0, 2, M.MEM, "DMA_S2MM_1_STREAM_STARVATION", 5000 + 40 * i) for i in range(3)]
    write_side(emu, "t.chess.emu", ev)
    hev = rows(0, 0) + [(0, 2, M.MEM, "DMA_S2MM_1_STREAM_STARVATION", 9000 + 40 * i) for i in range(3)]
    hev += [(0, 2, M.CORE, "INSTR_EVENT_1", 7000), (0, 2, M.CORE, "INSTR_EVENT_1", 7010)]
    write_side(hw, "t.chess.hw", hev)
    emu_extra = [(0, 2, M.CORE, "INSTR_EVENT_1", 7000)]  # one occurrence on the emulator side only
    write_side(tmp_path / "emu2", "t.chess.emu", rows(0, 0) + emu_extra)
    assert M.measure(emu, [hw])[("t.chess", (0, 2))] == 0
    assert M.measure(tmp_path / "emu2", [hw])[("t.chess", (0, 2))] == 0


def test_tile_needs_usable_rows_on_both_units(tmp_path):
    emu, hw = tmp_path / "emu", tmp_path / "hw"
    only_core = [e for e in rows(0, 0) if e[2] == M.CORE]
    write_side(emu, "t.chess.emu", only_core)
    write_side(hw, "t.chess.hw", only_core)
    assert M.measure(emu, [hw]) == {}


def test_cli_prints_per_tile_and_summary(tmp_path, capsys):
    emu, hw = tmp_path / "emu", tmp_path / "hw"
    write_side(emu, "t.chess.emu", rows(0, -1))
    write_side(hw, "t.chess.hw", rows(0, 0))
    assert M.main(["--emu", str(emu), "--silicon", str(hw)]) == 0
    out = capsys.readouterr().out
    assert "t.chess tile(0, 2)" in out and "-1" in out and "1 test-tiles; median -1" in out
