"""Non-vacuousness proof for build/experiments/sp5-skew/r1_tally.py (#140
SP-5b Task 6).

Task 6 is a HW-gated runnability check: the real gate (r1_gate.sh) needs
Phoenix, which this authoring pass deliberately does not touch (silicon runs
are held for a human -- see task-6-report.md). Without a real 20-run capture
to exercise it against, an untested tally script is just prose that happens
to parse. This test substitutes synthetic per-run events.json fixtures (the
exact schema tools/parse-trace.py's --out-events produces, after this task's
normalize_placement.py step) and invokes r1_tally.py's actual CLI entry
point via subprocess -- the same way r1_gate.sh calls it -- to prove the
range-0 + non-degeneracy logic actually fires both ways:

  a) identical runs -> every pair range 0 -> gate passes (rc 0).
  b) one run's d_v-pair anchor perturbed -> that pair's range > 0 -> the
     "d_v not range-0" assertion fires (rc 1).
  c) geometry with only 2 distinct core dn_v among its pairs' endpoints ->
     the "degenerate" assertion fires (rc 1), before any range is even
     computed.

r1_tally.py is used byte-for-byte verbatim from the task-6 brief (not
modified by this task), so this test exercises it as an external CLI rather
than importing it as a module -- no test-only seams were added to the
verbatim file.
"""
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_R1_TALLY = _REPO_ROOT / "build" / "experiments" / "sp5-skew" / "r1_tally.py"

_CORE = 0
_MEM = 1


def _ev(col, row, pkt_type, name, soc):
    return {"col": col, "row": row, "pkt_type": pkt_type, "name": name, "soc": soc}


def _ep(col, row, pkt_type, name, dn_v):
    return {"col": col, "row": row, "pkt_type": pkt_type, "name": name, "dn_v": dn_v}


# Full, non-degenerate geometry: 3 within-column core-core d_v pairs spanning
# dn_v {2,3,4} (matching the real sp5_skew_r1 geometry.json's shape) plus one
# same-tile core<->mem intra-contrast pair (the last entry, per r1_tally.py's
# `ranges[:-1]` / `ranges[-1]` convention).
_GEOM_OK = {
    "pairs": [
        {"a": _ep(0, 2, _CORE, "LOCK_STALL", 2), "b": _ep(0, 3, _CORE, "LOCK_STALL", 3)},
        {"a": _ep(0, 3, _CORE, "LOCK_STALL", 3), "b": _ep(0, 4, _CORE, "LOCK_STALL", 4)},
        {"a": _ep(0, 2, _CORE, "LOCK_STALL", 2), "b": _ep(0, 4, _CORE, "LOCK_STALL", 4)},
        {"a": _ep(0, 2, _CORE, "LOCK_STALL", 2), "b": _ep(0, 2, _MEM, "MEM_EVT", 2)},
    ]
}

# Degenerate geometry: only 2 distinct core dn_v ({2,3}) among the pairs'
# endpoints -- drops the dn_v=4 pair entirely.
_GEOM_DEGENERATE = {
    "pairs": [
        {"a": _ep(0, 2, _CORE, "LOCK_STALL", 2), "b": _ep(0, 3, _CORE, "LOCK_STALL", 3)},
        {"a": _ep(0, 2, _CORE, "LOCK_STALL", 2), "b": _ep(0, 2, _MEM, "MEM_EVT", 2)},
    ]
}

_DWALL_EVENTS = [
    _ev(0, 2, _CORE, "LOCK_STALL", 100),
    _ev(0, 3, _CORE, "LOCK_STALL", 90),
    _ev(0, 4, _CORE, "LOCK_STALL", 80),
    _ev(0, 2, _MEM, "MEM_EVT", 100),
]

_MEASURED_EVENTS = [
    _ev(0, 2, _CORE, "LOCK_STALL", 100),
    _ev(0, 3, _CORE, "LOCK_STALL", 93),
    _ev(0, 4, _CORE, "LOCK_STALL", 86),
    _ev(0, 2, _MEM, "MEM_EVT", 102),
]


def _write(path, obj):
    path.write_text(json.dumps(obj))


def _run_tally(*args, cwd):
    return subprocess.run(
        [sys.executable, str(_R1_TALLY), *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_r1_tally_exists_and_is_the_verbatim_brief_file():
    assert _R1_TALLY.is_file(), f"r1_tally.py not found at {_R1_TALLY}"


def test_identical_runs_pass_with_all_ranges_zero(tmp_path):
    _write(tmp_path / "geom.json", _GEOM_OK)
    _write(tmp_path / "dwall.events.json", {"events": _DWALL_EVENTS})
    for n in (1, 2, 3):
        _write(tmp_path / f"run_{n:02d}.events.json", {"events": _MEASURED_EVENTS})

    result = _run_tally(
        str(tmp_path / "run_*.events.json"),
        str(tmp_path / "dwall.events.json"),
        str(tmp_path / "geom.json"),
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["n_runs"] == 3
    assert summary["ranges"] == [0.0, 0.0, 0.0, 0.0]
    assert summary["dn_core"] == [2, 3, 4]
    assert "contrast: range-0" in result.stdout


def test_perturbed_d_v_pair_fails_range_zero_check(tmp_path):
    _write(tmp_path / "geom.json", _GEOM_OK)
    _write(tmp_path / "dwall.events.json", {"events": _DWALL_EVENTS})
    _write(tmp_path / "run_01.events.json", {"events": _MEASURED_EVENTS})
    _write(tmp_path / "run_02.events.json", {"events": _MEASURED_EVENTS})
    # Perturb run 3's row-3 core anchor by +2 soc. Row 3 is an endpoint of
    # two within-column d_v pairs (dn_v 2-3 and dn_v 3-4), so both those
    # pairs' skew shift relative to runs 1/2; the dn_v 2-4 pair and the
    # same-tile intra-contrast pair share no endpoint with row 3 and stay
    # untouched -- a real, non-cherry-picked consequence of the geometry's
    # shared endpoints, not a test artifact.
    perturbed = [dict(e) for e in _MEASURED_EVENTS]
    for e in perturbed:
        if e["row"] == 3:
            e["soc"] += 2
    _write(tmp_path / "run_03.events.json", {"events": perturbed})

    result = _run_tally(
        str(tmp_path / "run_*.events.json"),
        str(tmp_path / "dwall.events.json"),
        str(tmp_path / "geom.json"),
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 1
    assert "d_v not range-0" in result.stderr
    # The diagnostic JSON line still prints before the assertion fires,
    # showing the two row-3-touching pairs (dn_v 2-3, dn_v 3-4) as the
    # nonzero offenders, while the dn_v 2-4 pair and the intra-contrast
    # pair (neither touching row 3) stay at range 0.
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["ranges"][0] != 0.0
    assert summary["ranges"][1] != 0.0
    assert summary["ranges"][2] == 0.0
    assert summary["ranges"][3] == 0.0


def test_degenerate_geometry_fails_before_computing_ranges(tmp_path):
    _write(tmp_path / "geom.json", _GEOM_DEGENERATE)
    _write(tmp_path / "dwall.events.json", {"events": _DWALL_EVENTS})
    for n in (1, 2, 3):
        _write(tmp_path / f"run_{n:02d}.events.json", {"events": _MEASURED_EVENTS})

    result = _run_tally(
        str(tmp_path / "run_*.events.json"),
        str(tmp_path / "dwall.events.json"),
        str(tmp_path / "geom.json"),
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 1
    assert "degenerate: only 2 core dn_v" in result.stderr
    # Degeneracy is checked before the range-0 print/assert, so no JSON
    # summary line is ever emitted on this path.
    assert result.stdout == ""
