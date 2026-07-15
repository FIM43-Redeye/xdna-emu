#!/usr/bin/env python3
"""HW-free fixture tests for the memtile_bankwidth generator + analyzer.

Two independent things are checked, neither of which needs a compiler or
hardware:

  1. STRUCTURAL: the emitted MLIR pins a1_collide's two buffers to the SAME
     physical bank, a1_apart to DIFFERENT banks, a1_idle omits the fill
     channel entirely, and each a2_stride_S BD encodes stride S (in words).

  2. ANALYZER FIXTURE: a hand-built synthetic decoded-events capture (the
     same perfetto B/E + trace_config.json shape `bankdisc_measure.
     load_intervals` consumes -- see its docstring) with a KNOWN conflict
     area and KNOWN fill cadence that inverts to a KNOWN width, and KNOWN
     A2 spans that produce KNOWN ratios; asserts memtile_bankwidth_analyze's
     `analyze()` recovers exactly those planted numbers.

Run: python3 tools/experiments/test_memtile_bankwidth_analyze.py
"""
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memtile_bankwidth import (  # noqa: E402
    A1_VARIANTS, A2_VARIANTS, emit, physical_bank,
)
from memtile_bankwidth_measure import TRANSFER_BYTES, TRANSFER_WORDS, measure  # noqa: E402
from memtile_bankwidth_analyze import analyze  # noqa: E402

_ADDR_RE = r'sym_name = "{}", address = (\d+)'


def _buf_addr(mlir_text: str, buf_name: str) -> int:
    m = re.search(_ADDR_RE.format(buf_name), mlir_text)
    assert m, f"{buf_name} address not found in emitted MLIR"
    return int(m.group(1))


# ---------------------------------------------------------------- Step 2 ---

def test_a1_collide_same_bank():
    text = emit("a1_collide")
    fill_bank = physical_bank(_buf_addr(text, "fill_buf"))
    drain_bank = physical_bank(_buf_addr(text, "drain_buf"))
    assert fill_bank == drain_bank, (
        f"a1_collide must pin both buffers to the same physical bank, "
        f"got fill={fill_bank} drain={drain_bank}")


def test_a1_apart_different_bank():
    text = emit("a1_apart")
    fill_bank = physical_bank(_buf_addr(text, "fill_buf"))
    drain_bank = physical_bank(_buf_addr(text, "drain_buf"))
    assert fill_bank != drain_bank, (
        f"a1_apart must pin buffers to DIFFERENT physical banks, "
        f"got fill={fill_bank} drain={drain_bank} (both {fill_bank})")


def test_a1_idle_omits_fill_channel():
    text = emit("a1_idle")
    assert "fill_buf" not in text, "a1_idle must not declare a fill buffer"
    assert "aie.dma_start(S2MM" not in text, (
        "a1_idle must not open a memtile S2MM (fill) channel -- the shim's "
        "OWN S2MM receive allocation for the drain path is expected and fine")
    assert "drain_buf" in text, "a1_idle must still drain (the floor channel)"


def test_a1_variants_cover_collide_apart_idle():
    assert set(A1_VARIANTS) == {"a1_collide", "a1_apart", "a1_idle"}


def test_a2_stride_encodes_stride_words():
    for name, stride_bytes in A2_VARIANTS.items():
        stride_words = stride_bytes // 4
        text = emit(name)
        assert f"stride = {stride_words}" in text, (
            f"{name}: expected 'stride = {stride_words}' (from "
            f"{stride_bytes} B) in emitted BD, got:\n{text}")
        # Fixed word count per transfer, regardless of stride.
        assert re.search(r"dma_bd\(%stride_buf.*?, 0, 256,", text), (
            f"{name}: expected a fixed 256-word transfer length")


# ---------------------------------------------------------------- Step 4 ---

def _write_synthetic_capture(build_dir: Path, slot_table: list, events: list):
    """events: [(ph, tid, ts), ...] on a single synthetic 'memtile' tile.

    Mirrors exactly the shape `bankdisc_measure.load_intervals` consumes:
    a bare list of Chrome-trace-style dicts (one 'M' process_name naming the
    module, then 'B'/'E' pairs keyed by tid) plus a trace_config.json whose
    tiles_traced[i]['events'] is the slot-index -> event-name table.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    perfetto = [{"ph": "M", "name": "process_name", "pid": 0,
                 "args": {"name": "memtile(1,0)"}}]
    for ph, tid, ts in events:
        perfetto.append({"ph": ph, "pid": 0, "tid": tid, "ts": ts})
    (build_dir / "perfetto_r1.json").write_text(json.dumps(perfetto))
    config = {"tiles_traced": [{"module": "memtile", "events": slot_table}]}
    (build_dir / "trace_config.json").write_text(json.dumps(config))


# Slot layout for the a1_collide fixture (fill + drain + one contending bank).
_A1_SLOTS = [
    "MEM_TILE_DMA_MM2S_SEL0_FINISHED_BD",   # 0
    "MEM_TILE_DMA_MM2S_SEL0_STALLED_LOCK",  # 1
    "CONFLICT_DM_BANK_0",                  # 2
    "MEM_TILE_DMA_S2MM_SEL0_FINISHED_BD",   # 3
    "MEM_TILE_DMA_S2MM_SEL0_STALLED_LOCK",  # 4
]

# Planted so the recovered width is an exact, round number:
#   fill:  STALLED_LOCK [0,5) -> FINISHED_BD@69   => T_fill=64, f=256/64=4.0
#   drain: STALLED_LOCK [0,8) -> FINISHED_BD@100  => T_drain=92 (unused by width)
#   bank0 conflict: [10,74) => area=64
#   accesses_per_transfer = 64 / 4.0 = 16
#   width_bytes = TRANSFER_BYTES(1024) / 16 = 64.0
_A1_EVENTS = [
    ("B", 4, 0), ("E", 4, 5),        # fill STALLED_LOCK
    ("B", 3, 69), ("E", 3, 70),      # fill FINISHED_BD
    ("B", 1, 0), ("E", 1, 8),        # drain STALLED_LOCK
    ("B", 0, 100), ("E", 0, 101),    # drain FINISHED_BD
    ("B", 2, 10), ("E", 2, 74),      # CONFLICT_DM_BANK_0
]
_EXPECTED_WIDTH_BYTES = 64.0

# A2 slot layout: just the drain channel's bracket pair.
_A2_SLOTS = [
    "MEM_TILE_DMA_MM2S_SEL0_FINISHED_BD",   # 0
    "MEM_TILE_DMA_MM2S_SEL0_STALLED_LOCK",  # 1
]
# Planted spans: contiguous(4B)=20, stride16=40 (ratio 2.0), stride64=80 (ratio 4.0)
_A2_PLANTED = {
    4:  20,
    16: 40,
    64: 80,
}
_EXPECTED_RATIOS = {16: 2.0, 64: 4.0}


def _build_synthetic_stats(root: Path) -> dict:
    _write_synthetic_capture(
        root / "build_memtile_bankwidth_a1_collide", _A1_SLOTS, _A1_EVENTS)
    for stride, dur in _A2_PLANTED.items():
        events = [("B", 1, 0), ("E", 1, 5), ("B", 0, 5 + dur), ("E", 0, 6 + dur)]
        _write_synthetic_capture(
            root / f"build_memtile_bankwidth_a2_stride_{stride}", _A2_SLOTS, events)

    stats = {}
    collide_dir = root / "build_memtile_bankwidth_a1_collide"
    drain = measure(collide_dir, 1, channel="MM2S")
    fill = measure(collide_dir, 1, channel="S2MM")
    stats["a1_collide"] = {
        "conflict_area": drain["conflict_area"],
        "fill_median": fill["median"],
    }
    for stride in _A2_PLANTED:
        d = root / f"build_memtile_bankwidth_a2_stride_{stride}"
        stats[f"a2_stride_{stride}"] = {"median": measure(d, 1)["median"]}
    return stats


def test_analyzer_recovers_planted_width_and_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        stats = _build_synthetic_stats(Path(tmp))
        result = analyze(stats)

    assert result["width_bytes"] == _EXPECTED_WIDTH_BYTES, (
        f"expected width_bytes={_EXPECTED_WIDTH_BYTES}, got {result['width_bytes']}")
    for stride, expected_ratio in _EXPECTED_RATIOS.items():
        got = result["strided_ratio"].get(stride)
        assert got == expected_ratio, (
            f"stride={stride}: expected ratio {expected_ratio}, got {got}")


def test_transfer_bytes_matches_word_count():
    assert TRANSFER_BYTES == TRANSFER_WORDS * 4


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
