"""skew_constants.json read/write -- the SP-5b -> SP-5c handoff schema (#140)."""
import json


def empty_constants():
    return {"d_h": None, "d_v": None,
            "intra": {"core": None, "mem": None},
            "fit_residual": None, "source_route": None, "provenance": None}


def write_constants(path, *, d_h=None, d_v=None, intra_core=None, intra_mem=None,
                    fit_residual=None, source_route=None, provenance=None):
    obj = {"d_h": d_h, "d_v": d_v,
           "intra": {"core": intra_core, "mem": intra_mem},
           "fit_residual": fit_residual,
           "source_route": source_route, "provenance": provenance}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return obj


def read_constants(path):
    with open(path) as f:
        return json.load(f)
