"""R3b-PC shipped-geometry rank-sufficiency guard (#140 SP-5b).

Loads the kernel dir's geometry.json and asserts the {d_h, d_v} extractor is
rank-2 non-degenerate on it, with >=3 collinear same-kind tiles per axis. Guards
against a geometry edit silently making the instrument unidentifiable before it
ever reaches hardware."""
import json
import os
import struct
import pytest
from calibration.skew.r3b_observe import observe_r3b
from calibration.skew.r3b_extract import extract_r3b

GEOM_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "mlir-aie", "test", "npu-xrt", "sp5_skew_r3b_pc", "geometry.json"))


def _load():
    with open(GEOM_PATH) as f:
        return json.load(f)


def test_geometry_file_exists():
    assert os.path.exists(GEOM_PATH), GEOM_PATH


def test_target_is_npu1_3col():
    assert _load()["target"] == "npu1_3col"  # npu1_4col does not exist


def test_counter_indices_are_dense_and_unique():
    tiles = _load()["tiles"]
    idx = sorted(t["counter_index"] for t in tiles)
    assert idx == list(range(len(tiles))), idx


def test_extract_is_rank_two_on_synthetic_reading():
    geom = _load()
    n = max(t["counter_index"] for t in geom["tiles"]) + 1
    # Synthesize a readback from a known truth via the bridge's own coefficients.
    d_h, d_v, const = 2.0, 3.0, 500.0
    obs0 = observe_r3b(struct.pack("<%dI" % n, *([0] * n)), geom)
    vals = [0] * n
    for t, o in zip(geom["tiles"], obs0):
        vals[t["counter_index"]] = int(round(const + o["dn_h"] * d_h + o["dn_v"] * d_v))
    obs = observe_r3b(struct.pack("<%dI" % n, *vals), geom)
    r = extract_r3b(obs)  # min_rank=2; raises RankDeficientError if under-spanned
    assert r["fit_residual"] < 1e-6


def test_three_collinear_per_axis():
    tiles = _load()["tiles"]
    # >=3 tiles sharing a row (horizontal axis) and >=3 sharing a col (vertical).
    from collections import Counter
    rows = Counter(t["row"] for t in tiles)
    cols = Counter(t["col"] for t in tiles)
    assert max(rows.values()) >= 3, rows
    assert max(cols.values()) >= 3, cols
