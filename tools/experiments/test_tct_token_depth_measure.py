#!/usr/bin/env python3
"""HW-free fixture tests for tct_token_depth.py + tct_token_depth_measure.py.

Per the task brief's mandatory-discrimination requirement (Task 2 and Task 3
both shipped a Critical silent-wrong-constant bug because their first-pass
fixtures were too weak -- a single planted instance, or a tile-blind loader
nothing exercised): every scenario below is HAND-DERIVED from the planted
timestamps (never computed by calling the code under test), and each one is
built to make a SPECIFIC wrong implementation disagree with the correct
answer, not just to exercise the happy path.

Scenarios (module="shim", tile (0,0) unless noted):

  SMALL (expected outstanding = 4). Two FINISHED_TASK slots (MM2S_0, S2MM_0)
  contribute 2 edges each before the TOKEN_STALL onset -- exercises that the
  measure SUMS across both slots, not just one hardcoded name. A FIFTH edge
  planted EXACTLY AT the onset timestamp must be EXCLUDED (onset is a
  strict-before boundary, not <=) -- catches an off-by-one. A SIXTH edge
  planted well AFTER onset must also be excluded -- catches "count every
  FINISHED_TASK in the capture, ignore the stall" entirely.

  MEDIUM (expected outstanding = 8). 7 edges clearly before the true onset,
  plus a STRADDLING edge whose B(=1620)-phase is before onset(1650) but whose
  E(=1700)-phase is after it -- FINISHED_TASK is a discrete rising edge, so
  the measure must key off its B-ts; keying off E would wrongly exclude it
  (7, not 8). A further probe edge sits AFTER the true onset (1800) but
  BEFORE the stall interval's END (2000) -- if the measure mistakenly used
  the stall interval's END as "onset" instead of its START, this edge would
  be wrongly INCLUDED (9, not 8). Both bugs are independently observable from
  this one scenario's expected value.

  LARGE (expected outstanding = 11). A plain, larger count with no
  boundary tricks -- guards against a formula that happens to coincide with
  4 or 8 (the Task 2/3 lesson: distinct-enough numbers actually
  discriminate).

  NO_STALL. FINISHED_TASK edges are present but DMA_TASK_TOKEN_STALL never
  appears at all -- the measure must report `stall_observed=False` and
  `tasks_outstanding=None`, not a bogus integer (0 included -- a naive
  `max(..., default=0)` style implementation would silently return 0 and
  look plausible).

  MULTI-TILE. A SECOND "shim" tile (row=0, col=1 -- a hypothetical second
  shim column) shares the exact same module string prefix and event names as
  the tile under test (row=0, col=0). The decoy plants 20 edges before its
  OWN earlier onset (80); the real tile plants 5 edges before its own later
  onset (350). Under a tile-BLIND loader (keyed by (module_type, event_name)
  alone, e.g. bankdisc_measure.load_intervals), sorting the merged
  TOKEN_STALL intervals gives onset = min(80, 350) = 80, and counting merged
  FINISHED_TASK edges before ts=80 catches all 20 decoy edges and none of
  the real tile's (all >= 100) -- reporting 20 instead of the correct 5.
  This is verified out-of-band below (not part of the checked-in pass/fail
  suite) by literally reproducing that tile-blind computation over the same
  planted fixture and asserting it disagrees with the tile-aware answer.

Run: python3 tools/experiments/test_tct_token_depth_measure.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tct_token_depth import VARIANTS, emit  # noqa: E402
from tct_token_depth_measure import measure  # noqa: E402

_SLOTS = [
    "DMA_MM2S_0_FINISHED_TASK",   # tid 0
    "DMA_S2MM_0_FINISHED_TASK",   # tid 1
    "DMA_TASK_TOKEN_STALL",       # tid 2
]


def _write_capture(build_dir: Path, tiles: list):
    """tiles: [(module, row, col, events), ...] where events is
    [(phase, tid, ts), ...] ('B'/'E' pairs). Every tile uses the shared
    _SLOTS table (matches the real single-trace-unit-config shape).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    perfetto = []
    config_tiles = []
    for pid, (mod, row, col, events) in enumerate(tiles):
        perfetto.append({"ph": "M", "name": "process_name", "pid": pid,
                          "args": {"name": f"{mod}({row},{col})"}})
        for ph, tid, ts in events:
            perfetto.append({"ph": ph, "pid": pid, "tid": tid, "ts": ts})
        config_tiles.append({"module": mod, "events": _SLOTS})
    (build_dir / "perfetto_r1.json").write_text(json.dumps(perfetto))
    (build_dir / "trace_config.json").write_text(
        json.dumps({"tiles_traced": config_tiles}))


# ---- SMALL: expected outstanding = 4 ----
_SMALL_EVENTS = [
    ("B", 0, 10), ("E", 0, 11),      # MM2S FINISHED_TASK edge (counted)
    ("B", 0, 20), ("E", 0, 21),      # MM2S FINISHED_TASK edge (counted)
    ("B", 1, 15), ("E", 1, 16),      # S2MM FINISHED_TASK edge (counted)
    ("B", 1, 25), ("E", 1, 26),      # S2MM FINISHED_TASK edge (counted)
    ("B", 0, 30), ("E", 0, 31),      # AT the onset ts -- must be EXCLUDED
    ("B", 0, 100), ("E", 0, 101),    # well after onset -- must be EXCLUDED
    ("B", 2, 30), ("E", 2, 500),     # TOKEN_STALL onset=30
]
_SMALL_EXPECTED = 4

# ---- MEDIUM: expected outstanding = 8 ----
_MEDIUM_EVENTS = [
    ("B", 0, 1000), ("E", 0, 1005),
    ("B", 1, 1100), ("E", 1, 1105),
    ("B", 0, 1200), ("E", 0, 1205),
    ("B", 1, 1300), ("E", 1, 1305),
    ("B", 0, 1400), ("E", 0, 1405),
    ("B", 1, 1500), ("E", 1, 1505),
    ("B", 0, 1600), ("E", 0, 1605),   # 7 plain edges before onset (1650)
    ("B", 0, 1620), ("E", 0, 1700),   # straddling: B<onset<E -- must count (B-keyed)
    ("B", 0, 1800), ("E", 0, 1810),   # after TRUE onset, before stall's END(2000)
                                       # -- must be excluded (guards against
                                       # using the stall interval's END as onset)
    ("B", 2, 1650), ("E", 2, 2000),   # TOKEN_STALL onset=1650, end=2000
]
_MEDIUM_EXPECTED = 8

# ---- LARGE: expected outstanding = 11 ----
_LARGE_EVENTS = []
for i in range(11):
    tid = 0 if i % 2 == 0 else 1
    ts = 5000 + 5 * i
    _LARGE_EVENTS.append(("B", tid, ts))
    _LARGE_EVENTS.append(("E", tid, ts + 1))
_LARGE_EVENTS.append(("B", 2, 6000))
_LARGE_EVENTS.append(("E", 2, 9000))
_LARGE_EXPECTED = 11

# ---- NO_STALL: FINISHED_TASK present, TOKEN_STALL absent entirely ----
_NO_STALL_EVENTS = [
    ("B", 0, 10), ("E", 0, 11),
    ("B", 0, 20), ("E", 0, 21),
    ("B", 0, 30), ("E", 0, 31),
    ("B", 1, 15), ("E", 1, 16),
    ("B", 1, 25), ("E", 1, 26),
]

# ---- MULTI-TILE ----
_MULTI_SOURCE_EVENTS = [
    ("B", 0, 100), ("E", 0, 101),
    ("B", 0, 150), ("E", 0, 151),
    ("B", 0, 200), ("E", 0, 201),
    ("B", 0, 250), ("E", 0, 251),
    ("B", 0, 300), ("E", 0, 301),
    ("B", 2, 350), ("E", 2, 900),   # source's own onset = 350
]
_MULTI_SOURCE_EXPECTED = 5

_MULTI_DECOY_EVENTS = []
for i, ts in enumerate(range(10, 78, 4)):  # 10,14,...,74 -> 17 edges, all < 80
    _MULTI_DECOY_EVENTS.append(("B", 0, ts))
    _MULTI_DECOY_EVENTS.append(("E", 0, ts + 1))
# pad to exactly 20 decoy edges before its own onset (80)
for ts in (75, 76, 77):
    _MULTI_DECOY_EVENTS.append(("B", 0, ts))
    _MULTI_DECOY_EVENTS.append(("E", 0, ts + 1))
_MULTI_DECOY_EVENTS.append(("B", 2, 80))
_MULTI_DECOY_EVENTS.append(("E", 2, 120))   # decoy's own onset = 80 (earlier!)
_MULTI_DECOY_EDGE_COUNT = 20


def _measure_fixture(events) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        _write_capture(Path(tmp), [("shim", 0, 0, events)])
        return measure(Path(tmp), 1)


def test_small_outstanding_is_four_excludes_onset_boundary_and_after():
    m = _measure_fixture(_SMALL_EVENTS)
    assert m["stall_observed"] is True
    assert m["onset"] == 30, m["onset"]
    assert m["tasks_outstanding"] == _SMALL_EXPECTED, m["tasks_outstanding"]


def test_medium_outstanding_is_eight_uses_b_edge_for_both_events():
    m = _measure_fixture(_MEDIUM_EVENTS)
    assert m["stall_observed"] is True
    assert m["onset"] == 1650, m["onset"]
    assert m["tasks_outstanding"] == _MEDIUM_EXPECTED, m["tasks_outstanding"]


def test_large_outstanding_is_eleven():
    m = _measure_fixture(_LARGE_EVENTS)
    assert m["tasks_outstanding"] == _LARGE_EXPECTED, m["tasks_outstanding"]


def test_no_stall_reports_none_not_a_bogus_count():
    m = _measure_fixture(_NO_STALL_EVENTS)
    assert m["stall_observed"] is False
    assert m["onset"] is None
    assert m["tasks_outstanding"] is None
    assert m["n_finished_task"] == 5


def test_multitile_measure_is_not_corrupted_by_decoy_shim_tile():
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        _write_capture(build_dir, [
            ("shim", 0, 0, _MULTI_SOURCE_EVENTS),   # tile under test
            ("shim", 0, 1, _MULTI_DECOY_EVENTS),    # decoy second shim column
        ])
        m = measure(build_dir, 1)
    assert m["stall_observed"] is True
    assert m["onset"] == 350, m["onset"]
    assert m["tasks_outstanding"] == _MULTI_SOURCE_EXPECTED, m["tasks_outstanding"]


def test_multitile_fixture_fails_under_a_tile_blind_loader():
    """Out-of-band check (not testing our shipped code -- reproducing the
    OLD tile-blind merge by hand) proving the multi-tile fixture actually
    discriminates: a loader keyed by (module_type, event_name) alone, with
    no (row, col) filter, merges both tiles' intervals into one pool and
    gets the WRONG answer (the decoy's 20, not the source's 5).
    """
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        _write_capture(build_dir, [
            ("shim", 0, 0, _MULTI_SOURCE_EVENTS),
            ("shim", 0, 1, _MULTI_DECOY_EVENTS),
        ])
        ev = json.loads((build_dir / "perfetto_r1.json").read_text())

    # Tile-blind rebuild: same B/E pairing, but keyed by event name alone
    # (module type recovered but (row, col) discarded), exactly bankdisc_
    # measure.load_intervals's pre-Task-3-fix shape.
    from collections import defaultdict
    open_b, blind = {}, defaultdict(list)
    for e in ev:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        key = _SLOTS[e["tid"]]
        if ph == "B":
            open_b[(key, e["pid"])] = e["ts"]
        else:
            k = (key, e["pid"])
            if k in open_b:
                blind[key].append((open_b.pop(k), e["ts"]))

    edges = sorted(
        start for name, ivs in blind.items()
        if name.endswith("FINISHED_TASK") for start, _ in ivs
    )
    stalls = sorted(blind.get("DMA_TASK_TOKEN_STALL", []))
    blind_onset = stalls[0][0]
    blind_outstanding = sum(1 for ts in edges if ts < blind_onset)

    assert blind_onset == 80, blind_onset
    assert blind_outstanding == _MULTI_DECOY_EDGE_COUNT, blind_outstanding
    assert blind_outstanding != _MULTI_SOURCE_EXPECTED, (
        "tile-blind fixture failed to discriminate: it coincidentally "
        "matched the correct tile-aware answer"
    )


def test_variants_cover_control_and_throttle_sweep():
    assert "control" in VARIANTS
    assert VARIANTS["control"]["tokens_per_wave"] == 1
    assert VARIANTS["control"]["reclaim_token"] is True
    for v in ("all_small", "all_large", "half_small", "half_large"):
        assert v in VARIANTS
    assert VARIANTS["all_small_safe_reclaim"]["reclaim_token"] is False


def test_emit_does_not_crash_for_every_variant():
    for v in VARIANTS:
        text = emit(v)
        assert "aie.device(npu1_2col)" in text, v
        assert "aiex.dma_await_task" in text, v
        assert "issue_token = true" in text, v


def test_control_variant_only_tags_terminal_task_with_issue_token():
    from tct_token_depth import WAVE_SIZE
    text = emit("control")
    # tokens_per_wave=1 -> exactly one {issue_token = true} per wave.
    assert text.count("issue_token = true") == text.count("aiex.dma_await_task")


def test_all_variant_tags_every_task_in_wave_with_issue_token():
    from tct_token_depth import NUM_WAVES, WAVE_SIZE
    text = emit("all_small")
    # +1 for the long-lived S2MM %recv task, which always carries
    # issue_token=true regardless of variant.
    assert text.count("issue_token = true") == NUM_WAVES * WAVE_SIZE + 1


def test_safe_reclaim_variant_omits_token_on_terminal_task_only():
    from tct_token_depth import NUM_WAVES, WAVE_SIZE
    text = emit("all_small_safe_reclaim")
    # WAVE_SIZE-1 tagged per wave (every task except the terminal), +1 for
    # the long-lived S2MM %recv task (always issue_token=true).
    assert text.count("issue_token = true") == NUM_WAVES * (WAVE_SIZE - 1) + 1


def test_finished_task_matcher_sums_across_channels_not_one_hardcoded_name():
    # Regression guard for the "hardcoded to one slot name" failure mode:
    # the SMALL fixture's 4 counted edges are split 2-and-2 across
    # DMA_MM2S_0_FINISHED_TASK and DMA_S2MM_0_FINISHED_TASK.
    from tct_token_depth_measure import _finished_task_edges, _load_tile_intervals
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        _write_capture(build_dir, [("shim", 0, 0, _SMALL_EVENTS)])
        iv = _load_tile_intervals(build_dir / "perfetto_r1.json",
                                   build_dir / "trace_config.json")
    assert len(iv["DMA_MM2S_0_FINISHED_TASK"]) == 4  # 2 counted + 2 excluded
    assert len(iv["DMA_S2MM_0_FINISHED_TASK"]) == 2
    edges = _finished_task_edges(iv)
    assert len(edges) == 6  # all edges pooled, onset-filtering happens in measure()


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
