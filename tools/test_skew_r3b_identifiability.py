"""R3b two-source identifiability guard (#140 SP-5b).

Executable encoding of the theorem in
docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md:
a two-source two-flood interval CANNOT identify within-axis direction anisotropy
(d_hE vs d_hW, d_vN vs d_vS). This is why extract_r3b fits only {d_h, d_v}. If a
future change tries to "enrich the geometry" to split the axes, these tests fail
and point back at the finding, rather than the failure surfacing on hardware.
"""
import numpy as np

from calibration.skew.r3b_observe import reset_routed_coeffs


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


def test_reset_routed_vertical_term_is_constant_dv_collapses():
    """Under reset routing (shim-row horizontal, then column climb), both floods
    share each tile's vertical climb, so the interval's vertical coefficient is
    (s2.row - s1.row) for EVERY tile -- a constant. d_v is therefore unidentifiable
    from a block-replicated capture (design Sec.1 pt2)."""
    s1 = {"col": 0, "row": 0}
    s2 = {"col": 2, "row": 5}
    tiles = [{"col": 1, "row": r} for r in (2, 3, 4, 5)]  # the bring-up vertical spine
    dn_v = [reset_routed_coeffs(s1, s2, t)[1] for t in tiles]
    assert dn_v == [5, 5, 5, 5], dn_v  # constant -> zero signal after referencing
    # and horizontal DOES vary across columns, so d_h is still identifiable
    dn_h_row3 = [reset_routed_coeffs(s1, s2, {"col": c, "row": 3})[0] for c in (0, 1, 2)]
    assert dn_h_row3 == [2, 0, -2], dn_h_row3


def test_block_routed_capture_is_rank_deficient_for_dh_dv():
    """A single block-replicated capture: the [dn_h, dn_v] design matrix has a
    constant dn_v column, so after referencing the dn_v column is all-zero and the
    rank is 1 -- a {d_h,d_v} fit is rank-deficient. This is WHY the captures are
    decoupled (design Sec.2)."""
    s1 = {"col": 0, "row": 0}
    s2 = {"col": 2, "row": 5}
    tiles = ([{"col": c, "row": 3} for c in range(3)] +
             [{"col": 1, "row": r} for r in (2, 4, 5)])
    A = np.array([reset_routed_coeffs(s1, s2, t) for t in tiles], dtype=float)
    D = A[1:] - A[0]  # reference-difference, same convention as the rank-2 guard
    assert np.linalg.matrix_rank(D) == 1, np.linalg.matrix_rank(D)


def test_free_flood_straddling_capture_identifies_dv():
    """A free-flood capture with sources straddling the measured tiles vertically
    (s1 below, s2 above) recovers d_v with >=3 collinear-spine leverage: only the
    vertical axis is exercised here (dn_h is identically 0 on a single column), so
    the referenced design matrix is rank 1 with a non-degenerate dn_v column --
    d_v is identifiable."""
    s1, s2 = (1, 0), (1, 5)  # same column, straddling -> pure vertical signal
    tiles = [(1, r) for r in (1, 2, 3, 4)]
    rows = []
    for c, r in tiles:
        e1, w1, n1, s1h = _hops(s1, (c, r))
        e2, w2, n2, s2h = _hops(s2, (c, r))
        rows.append([(e2 + w2) - (e1 + w1), (n2 + s2h) - (n1 + s1h)])
    A = np.array(rows, dtype=float)
    D = A[1:] - A[0]
    # dn_v varies across the spine (not constant) -> d_v is identifiable
    assert not np.allclose(A[:, 1], A[0, 1]), A[:, 1]
    # single axis exercised -> rank 1; and the vertical column survives referencing
    assert np.linalg.matrix_rank(D) == 1
    assert not np.allclose(D[:, 1], 0.0)
