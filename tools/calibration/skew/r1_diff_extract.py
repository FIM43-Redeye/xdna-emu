"""R1 differencing skew extraction: solve {d_v, intra_contrast} from within-column
cross-domain pair-difference observations (#140 SP-5b).

Model: module_delay(M) = dn_v(M)*d_v + intra_off(kind_M), intra_off = core_off for
{core,shim}, mem_off for {mem,memtile}. Only the gauge-invariant contrast
(core_off - mem_off) is observable -- adding a constant to both offsets leaves every
reset target (max_delay - module_delay) unchanged -- so this recovers exactly
{d_v, intra_contrast} from pair differences (spec Sec.4.2). Structurally identical
to r3b_extract's reference-differencing. Falsifying per-hop uniformity needs >=3
collinear observations per axis (2 points fit any line with zero residual).
"""
from ._solve import solve_design_matrix

_CORE_GROUP = {"core", "shim"}
_MEM_GROUP = {"mem", "memtile"}


def _core_ind(kind):
    if kind in _CORE_GROUP:
        return 1.0
    if kind in _MEM_GROUP:
        return 0.0
    raise ValueError(f"unknown module kind: {kind!r}")


def extract_r1_diff(pairs):
    """pairs: list of {"a": {"dn_v": int, "kind": str},
                       "b": {"dn_v": int, "kind": str},
                       "skew": float}  where skew = module_delay(b) - module_delay(a).
    Returns {"d_v", "intra_contrast", "fit_residual"} where
    intra_contrast = (core_off - mem_off)."""
    A, bvec = [], []
    for p in pairs:
        a, b = p["a"], p["b"]
        A.append([float(b["dn_v"] - a["dn_v"]),
                  _core_ind(b["kind"]) - _core_ind(a["kind"])])
        bvec.append(float(p["skew"]))
    x, resid = solve_design_matrix(A, bvec, min_rank=2)
    return {"d_v": float(x[0]), "intra_contrast": float(x[1]),
            "fit_residual": resid}
