"""R3b enriched interval-difference skew extraction (#140 SP-5b, rev3).

Fits per-direction hop coefficients (d_hE, d_hW, d_vN, d_vS) plus a turn term
(d_turn) from per-tile two-flood interval readings, so direction anisotropy and
cross-axis non-additivity are identifiable — the falsification apparatus the
soundness audit (2026-07-01) requires. See kernel spec rev3 Sec.5.1/5.2.

The const (= T0_2 - T0_1) is removed by differencing against a reference tile.
Each observation carries the SIGNED design-row coefficients the interval
accumulates per direction; the observation bridge (r3b_observe) computes them
from geometry.json, so this extractor is geometry-agnostic.
"""
from ._solve import solve_design_matrix

ALL_PARAMS = ("d_hE", "d_hW", "d_vN", "d_vS", "d_turn")
_COEF = {"d_hE": "a_hE", "d_hW": "a_hW", "d_vN": "a_vN", "d_vS": "a_vS",
         "d_turn": "a_turn"}


def extract_r3b(observations, reference=0, params=ALL_PARAMS):
    """observations: list of {a_hE, a_hW, a_vN, a_vS, a_turn, r}.
    reference: index differenced against to drop const.
    params: which per-direction columns to identify (subset of ALL_PARAMS,
        in ALL_PARAMS order). min_rank = len(params); a column the geometry
        leaves all-zero makes the matrix rank-deficient -> RankDeficientError.
    Returns {each requested param, plus d_h, d_v, aniso_h, aniso_v, fit_residual}."""
    for p in params:
        if p not in ALL_PARAMS:
            raise ValueError(f"unknown param {p!r}")
    cols = [_COEF[p] for p in params]
    ref = observations[reference]
    A, b = [], []
    for i, o in enumerate(observations):
        if i == reference:
            continue
        A.append([float(o[c] - ref[c]) for c in cols])
        b.append(float(o["r"] - ref["r"]))
    x, resid = solve_design_matrix(A, b, min_rank=len(params))
    out = {p: float(xi) for p, xi in zip(params, x)}
    out["fit_residual"] = resid
    # Convenience aggregates (present only when both halves were fit).
    if "d_hE" in out and "d_hW" in out:
        out["d_h"] = 0.5 * (out["d_hE"] + out["d_hW"])
        out["aniso_h"] = out["d_hE"] - out["d_hW"]
    if "d_vN" in out and "d_vS" in out:
        out["d_v"] = 0.5 * (out["d_vN"] + out["d_vS"])
        out["aniso_v"] = out["d_vN"] - out["d_vS"]
    return out
