"""R1 within-column skew extraction: solve {d_v, intra_core, intra_mem} from
within-column cross-domain trace residuals (#140 SP-5b).

Sign convention: the decoded trace carries the REFLECTED origin offset
(max_delay - module_delay); origin_D (BroadcastTiming) is module_delay directly.
Cross-domain differences cancel max_delay but INVERT the sign, so with
reflected=True we negate each origin before solving to report in origin_D form.
Falsification of per-hop uniformity requires >=3 collinear observations per axis
(2 points fit any line with zero residual).

Kind grouping mirrors the emulator's origin_d_table (effects.rs:527-532): the
intra offset is core_off for {core, shim} and mem_off for {mem, memtile}, so the
extractor fits real emu/silicon data directly. (Provisional emulator model; SP-5c
may revise the grouping.)
"""
from ._solve import solve_design_matrix

# params order: [d_v, intra_core, intra_mem].
_CORE_GROUP = {"core", "shim"}
_MEM_GROUP = {"mem", "memtile"}


def _design_row(dn_v, kind):
    return [float(dn_v),
            1.0 if kind in _CORE_GROUP else 0.0,
            1.0 if kind in _MEM_GROUP else 0.0]


def extract_r1(observations, reflected=True):
    """observations: list of {"dn_v": int, "kind": str, "origin": float}
    (origin = module origin relative to the source, source = 0).
    Returns {"d_v", "intra_core", "intra_mem", "fit_residual"}."""
    A, b = [], []
    for o in observations:
        A.append(_design_row(o["dn_v"], o["kind"]))
        b.append(-o["origin"] if reflected else o["origin"])
    x, resid = solve_design_matrix(A, b, min_rank=2)
    return {"d_v": float(x[0]), "intra_core": float(x[1]),
            "intra_mem": float(x[2]), "fit_residual": resid}
