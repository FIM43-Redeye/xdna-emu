"""R3b two-source identifiability guard (#140 SP-5b).

Executable encoding of the theorem in
docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md:
a two-source two-flood interval CANNOT identify within-axis direction anisotropy
(d_hE vs d_hW, d_vN vs d_vS). This is why extract_r3b fits only {d_h, d_v}. If a
future change tries to "enrich the geometry" to split the axes, these tests fail
and point back at the finding, rather than the failure surfacing on hardware.
"""
import numpy as np


def _hops(src, tile):
    c, r = tile
    sc, sr = src
    return (max(c - sc, 0), max(sc - c, 0), max(r - sr, 0), max(sr - r, 0))


def _enriched_rows(s1, s2, tiles):
    """Signed per-direction design rows [a_hE, a_hW, a_vN, a_vS, a_turn]."""
    rows = []
    for t in tiles:
        e1, w1, n1, s1h = _hops(s1, t)
        e2, w2, n2, s2h = _hops(s2, t)
        rows.append([e2 - e1, w2 - w1, n2 - n1, s2h - s1h,
                     (e2 + w2) * (n2 + s2h) - (e1 + w1) * (n1 + s1h)])
    return np.array(rows, dtype=float)


def test_east_west_and_north_south_columns_are_constant_offsets():
    # a_hE - a_hW = s1.col - s2.col and a_vN - a_vS = s1.row - s2.row for EVERY
    # tile at ANY placement -- the algebraic identity max(dx,0)-max(-dx,0)=dx.
    for s1, s2 in [((0, 0), (2, 4)), ((1, 3), (0, 0)), ((2, 5), (0, 1))]:
        tiles = [(c, r) for c in range(3) for r in range(6)]
        A = _enriched_rows(s1, s2, tiles)
        assert np.allclose(A[:, 0] - A[:, 1], s1[0] - s2[0])  # a_hE - a_hW
        assert np.allclose(A[:, 2] - A[:, 3], s1[1] - s2[1])  # a_vN - a_vS


def test_corner_source_enriched_design_is_rank_two():
    # Corner sources on npu1_3col (cols 0..2, rows 0..5): the enriched 5-column
    # design collapses to rank 2 after referencing -- no tile addition can raise
    # it, because the array IS the full grid. Only {d_h, d_v} are identifiable.
    s1, s2 = (0, 0), (2, 5)
    tiles = [(c, r) for c in range(3) for r in range(6)]
    A = _enriched_rows(s1, s2, tiles)
    D = A[1:] - A[0]  # difference against a reference tile to drop the const
    assert np.linalg.matrix_rank(D) == 2


def test_cross_axis_dh_dv_remain_identifiable():
    # The two axis columns {|dcol|-diff, |drow|-diff} DO span rank 2 -- so the
    # cross-axis asymmetry d_h vs d_v (what extract_r3b fits) is identifiable.
    s1, s2 = (0, 0), (2, 5)
    tiles = [(c, r) for c in range(3) for r in range(6)]
    rows = []
    for (c, r) in tiles:
        e1, w1, n1, s1h = _hops(s1, (c, r))
        e2, w2, n2, s2h = _hops(s2, (c, r))
        rows.append([(e2 + w2) - (e1 + w1), (n2 + s2h) - (n1 + s1h)])
    A = np.array(rows, dtype=float)
    D = A[1:] - A[0]
    assert np.linalg.matrix_rank(D) == 2
