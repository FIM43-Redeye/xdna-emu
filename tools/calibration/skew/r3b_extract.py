"""R3b interval-difference skew extraction: solve {d_h, d_v} from per-tile
two-flood interval readings r_X = const + dn_h*d_h + dn_v*d_v (#140 SP-5b).

const (= T0_2 - T0_1) is removed by differencing against a reference tile.
Because differencing removes one degree of freedom, >=3 collinear tiles per axis
are needed to detect per-hop non-uniformity via the fit residual.
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
