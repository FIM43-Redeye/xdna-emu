#!/usr/bin/env python3
"""HW-free fixture tests for the memtile_bankwidth generator + analyzer.

Two independent things are checked, neither of which needs a compiler or
hardware:

  1. STRUCTURAL: the emitted MLIR pins a1_collide's two buffers to the SAME
     physical bank (0), a1_apart to DIFFERENT banks (drain 0, fill 8),
     a1_solo omits the fill channel entirely (drain still on bank 0, for an
     apples-to-apples floor), every A1 channel's dma_bd is the strided
     single-bank geometry (not the old flat, all-16-banks descriptor a real
     HW capture read CONFLICT_DM_BANK_0 = 0 against), and each a2_stride_S BD
     encodes stride S (in words).

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
    # Exact bank INDEX, not just equality -- a regression that pins both
    # buffers to the same WRONG bank (e.g. both accidentally on bank 5) would
    # pass a bare equality check but must still fail here.
    assert fill_bank == 0 and drain_bank == 0, (
        f"a1_collide must pin BOTH buffers to physical bank 0, "
        f"got fill={fill_bank} drain={drain_bank}")


def test_a1_apart_different_bank():
    text = emit("a1_apart")
    fill_bank = physical_bank(_buf_addr(text, "fill_buf"))
    drain_bank = physical_bank(_buf_addr(text, "drain_buf"))
    assert fill_bank == 8 and drain_bank == 0, (
        f"a1_apart must pin fill_buf to bank 8 and drain_buf to bank 0, "
        f"got fill={fill_bank} drain={drain_bank}")


def test_a1_solo_omits_fill_channel():
    text = emit("a1_solo")
    assert "fill_buf" not in text, "a1_solo must not declare a fill buffer"
    assert "aie.dma_start(S2MM" not in text, (
        "a1_solo must not open a memtile S2MM (fill) channel -- the shim's "
        "OWN S2MM receive allocation for the drain path is expected and fine")
    assert "drain_buf" in text, "a1_solo must still drain (the floor channel)"
    drain_bank = physical_bank(_buf_addr(text, "drain_buf"))
    assert drain_bank == 0, (
        f"a1_solo's drain_buf must stay on bank 0 (same as collide/apart's "
        f"drain, for an apples-to-apples mutual-slowdown comparison), got "
        f"bank {drain_bank}")


def test_a1_drain_addr_consistent_across_variants():
    """The mutual-slowdown comparison (collide vs solo) is only
    apples-to-apples if drain_buf's address -- hence its physical bank AND
    exact byte offset -- is IDENTICAL across all three A1 variants, so the
    ONLY thing that changes between them is the fill channel's presence/bank."""
    addrs = {_buf_addr(emit(v), "drain_buf") for v in A1_VARIANTS}
    assert len(addrs) == 1, (
        f"drain_buf address must be identical across a1_collide/apart/solo, "
        f"got {addrs}")


def test_a1_variants_cover_collide_apart_solo():
    assert set(A1_VARIANTS) == {"a1_collide", "a1_apart", "a1_solo"}


def test_a1_strided_to_single_bank():
    """Every A1 channel's dma_bd must use the strided-to-single-bank wrap
    (256 B/64-word stride, 16 B/4-word granule) -- not the old flat,
    contiguous descriptor that spanned all 16 banks and read
    CONFLICT_DM_BANK_0 = 0 on a real HW capture."""
    for variant in A1_VARIANTS:
        text = emit(variant)
        assert "stride = 64" in text and "stride = 1" in text, (
            f"{variant}: expected the strided-to-single-bank dims "
            f"(<size=64, stride=64>, <size=4, stride=1>), got:\n{text}")


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
    # Match the REAL injector: memtile entries carry "kind", not "module"
    # (mlir-trace-inject writes "module" only for compute tiles). load_intervals
    # falls back to "kind" when "module" is absent.
    config = {"tiles_traced": [{"kind": "memtile", "events": slot_table}]}
    (build_dir / "trace_config.json").write_text(json.dumps(config))


# Slot layout for the a1_collide fixture (fill + drain + one contending bank).
# Bare event names as the decoder actually emits them (no MEM_TILE_ prefix).
_A1_SLOTS = [
    "DMA_MM2S_SEL0_FINISHED_BD",   # 0
    "DMA_MM2S_SEL0_STALLED_LOCK",  # 1
    "CONFLICT_DM_BANK_0",          # 2
    "DMA_S2MM_SEL0_FINISHED_BD",   # 3
    "DMA_S2MM_SEL0_STALLED_LOCK",  # 4
]

# Planted with TWO drain (MM2S) transfers, each gated by its own STALLED_LOCK
# interval, and two DISTINCT bank-0 conflict intervals -- so capture-total
# conflict area != per-transfer conflict area, and the n=1 degeneracy that
# let the un-normalized formula hide inside the old single-transfer fixture
# can no longer occur:
#   fill:  STALLED_LOCK [0,5)   -> FINISHED_BD@69   => T_fill=64, f=256/64=4.0
#   drain xfer 1: STALLED_LOCK [0,8)    -> FINISHED_BD@100  (bracket start=8)
#   drain xfer 2: STALLED_LOCK [150,180) -> FINISHED_BD@200 (bracket start=180)
#   => n_bracketed = 2 (T_drain unused by width either way)
#   bank0 conflict interval 1: [10,106)  => area=96
#   bank0 conflict interval 2: [120,280) => area=160
#   capture-total conflict area = 96 + 160 = 256
#   per_xfer_conflict = 256 / n_bracketed(2) = 128        <- the fix
#   accesses_per_transfer = 128 / f_contender(4.0) = 32
#   width_bytes = TRANSFER_BYTES(1024) / 32 = 32.0
#
# Hand check that this fixture DISCRIMINATES the bug: the old, un-normalized
# formula fed the capture-total (256) straight into accesses_per_transfer =
# 256 / 4.0 = 64, giving width = 1024 / 64 = 16.0 -- exactly 2x off (matching
# n_bracketed=2), so a regression to the old formula fails this assertion.
_A1_EVENTS = [
    ("B", 4, 0), ("E", 4, 5),        # fill STALLED_LOCK
    ("B", 3, 69), ("E", 3, 70),      # fill FINISHED_BD
    ("B", 1, 0), ("E", 1, 8),        # drain STALLED_LOCK #1
    ("B", 0, 100), ("E", 0, 101),    # drain FINISHED_BD #1
    ("B", 1, 150), ("E", 1, 180),    # drain STALLED_LOCK #2
    ("B", 0, 200), ("E", 0, 201),    # drain FINISHED_BD #2
    ("B", 2, 10), ("E", 2, 106),     # CONFLICT_DM_BANK_0 interval 1 (area 96)
    ("B", 2, 120), ("E", 2, 280),    # CONFLICT_DM_BANK_0 interval 2 (area 160)
]
_EXPECTED_WIDTH_BYTES = 32.0


def test_a1_collide_fixture_reads_nonzero_conflict():
    """Sanity companion to test_analyzer_recovers_planted_width_and_ratio: the
    collide fixture's planted CONFLICT_DM_BANK_0 intervals must yield a
    nonzero conflict_area -- the width inversion is meaningless at 0, which is
    exactly what a broken, non-strided A1 reads on real HW."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "build_memtile_bankwidth_a1_collide"
        _write_synthetic_capture(d, _A1_SLOTS, _A1_EVENTS)
        drain = measure(d, 1, channel="MM2S")
    assert drain["conflict_area"] > 0, (
        f"a1_collide fixture must read conflict_area > 0, "
        f"got {drain['conflict_area']}")


# Slot layout + planted events for the a1_apart validity-control fixture:
# BOTH channels active and bracketed, but on DIFFERENT banks -- no
# CONFLICT_DM_BANK_0 interval is planted at all, since apart's two channels
# never share a bank. This is the null result the apart variant exists to
# produce (a regression that lands both channels on the same bank by
# accident would show up here as a nonzero conflict_area).
_A1_APART_SLOTS = [
    "DMA_MM2S_SEL0_FINISHED_BD",   # 0
    "DMA_MM2S_SEL0_STALLED_LOCK",  # 1
    "DMA_S2MM_SEL0_FINISHED_BD",   # 2
    "DMA_S2MM_SEL0_STALLED_LOCK",  # 3
]
_A1_APART_EVENTS = [
    ("B", 1, 0), ("E", 1, 8),        # drain STALLED_LOCK
    ("B", 0, 100), ("E", 0, 101),    # drain FINISHED_BD
    ("B", 3, 0), ("E", 3, 5),        # fill STALLED_LOCK
    ("B", 2, 69), ("E", 2, 70),      # fill FINISHED_BD
]


def test_a1_apart_fixture_reads_zero_conflict():
    """Analyzer-fixture companion to test_a1_apart_different_bank: even with
    BOTH channels active and bracketed, a capture with no planted
    CONFLICT_DM_BANK_n interval must measure conflict_area == 0 -- the
    validity-control null result a1_apart exists to produce."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "build_memtile_bankwidth_a1_apart"
        _write_synthetic_capture(d, _A1_APART_SLOTS, _A1_APART_EVENTS)
        drain = measure(d, 1, channel="MM2S")
    assert drain["conflict_area"] == 0, (
        f"a1_apart fixture must read conflict_area == 0 (no shared bank), "
        f"got {drain['conflict_area']}")
    assert drain["n_bracketed"] == 1, "sanity: exactly one drain transfer planted"


# A2 slot layout: just the drain channel's bracket pair.
_A2_SLOTS = [
    "DMA_MM2S_SEL0_FINISHED_BD",   # 0
    "DMA_MM2S_SEL0_STALLED_LOCK",  # 1
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
        "n_bracketed": drain["n_bracketed"],
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
