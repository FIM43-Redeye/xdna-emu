"""R3b interval-difference skew extraction: solve {d_h, d_v} from per-tile
two-flood interval readings r_X = const + dn_h*d_h + dn_v*d_v (#140 SP-5b).

const (= T0_2 - T0_1) is removed by differencing against a reference tile.
Because differencing removes one degree of freedom, >=3 collinear tiles per axis
are needed to detect per-hop non-uniformity via the fit residual.

SP-5c Phase 2 (docs/superpowers/plans/2026-07-02-sp5c-phase2-dh-capture.md
Sec.1c) adds extract_r3b_dh, a single-column {d_h} extractor for
block_shim_row captures (tools/calibration/skew/r3b_observe.py's
reset_routed_coeffs), where dn_v is constant across tiles by construction --
see test_block_routed_capture_is_rank_deficient_for_dh_dv in
test_skew_r3b_identifiability.py. extract_r3b (the rank-2 {d_h,d_v} fit) must
keep raising RankDeficientError on such a capture; that failure is a
deliberate guard against fitting a coefficient that routing cannot identify,
not a bug to route around.
"""
from ._solve import solve_design_matrix


def extract_r3b(observations, reference=0):
    """observations: list of {"dn_h": int, "dn_v": int, "r": float}.
    reference: index differenced against to drop const.
    Returns {"d_h", "d_v", "fit_residual"}."""
    ref = observations[reference]
    A, b = [], []
    for i, o in enumerate(observations):
        if i == reference:
            continue
        A.append([float(o["dn_h"] - ref["dn_h"]), float(o["dn_v"] - ref["dn_v"])])
        b.append(float(o["r"] - ref["r"]))
    x, resid = solve_design_matrix(A, b, min_rank=2)
    return {"d_h": float(x[0]), "d_v": float(x[1]), "fit_residual": resid}


def extract_r3b_dh(observations, reference=0):
    """observations: list of {"dn_h": int, "dn_v": int, "r": float} from a
    block_shim_row (block-mask-replicated) capture, where dn_v is constant
    across tiles and therefore carries no signal after reference-differencing.
    reference: index differenced against to drop const.
    Returns {"d_h", "fit_residual"}."""
    ref = observations[reference]
    A, b = [], []
    for i, o in enumerate(observations):
        if i == reference:
            continue
        A.append([float(o["dn_h"] - ref["dn_h"])])
        b.append(float(o["r"] - ref["r"]))
    x, resid = solve_design_matrix(A, b, min_rank=1)
    return {"d_h": float(x[0]), "fit_residual": resid}
