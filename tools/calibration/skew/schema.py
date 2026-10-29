"""Read/write of skew_constants.json -- the SP-5b -> SP-5c handoff schema (#140)."""
import json


def empty_constants():
    return {
        "d_h": None, "d_v": None,
        "intra": {"core": None, "mem": None},
        "fit_residual": None, "source_route": None, "provenance": None,
        # SP-5c honest-provenance fields (design Sec.7):
        "per_channel": None, "b_vector_range": None, "jitter_range": None,
        "assumptions": {}, "d_h_path": None,
    }


def write_constants(path, *, d_h=None, d_v=None, intra_core=None, intra_mem=None,
                    fit_residual=None, source_route=None, provenance=None,
                    per_channel=None, b_vector_range=None, jitter_range=None,
                    assumptions=None, d_h_path=None):
    c = empty_constants()
    c.update({
        "d_h": d_h, "d_v": d_v,
        "intra": {"core": intra_core, "mem": intra_mem},
        "fit_residual": fit_residual, "source_route": source_route,
        "provenance": provenance, "per_channel": per_channel,
        "b_vector_range": b_vector_range, "jitter_range": jitter_range,
        "assumptions": assumptions if assumptions is not None else {},
        "d_h_path": d_h_path,
    })
    with open(path, "w") as f:
        json.dump(c, f, indent=2)
    return c


def read_constants(path):
    with open(path) as f:
        return json.load(f)
