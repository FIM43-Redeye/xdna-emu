"""Emu inject-and-recover check for R1 (#140 SP-5b): decode injected + zero
trace bins, run observe -> differencing extract, assert recovered {d_v,
intra_contrast} == injected. Plumbing/regression -- validates the seam ->
timer-reset -> in-process-run -> decode -> observe -> extract pipeline, NOT
silicon correctness (spec Sec.4.3)."""
import json
import math
import sys
from pathlib import Path

from calibration.skew.r1_observe import observe_r1
from calibration.skew.r1_diff_extract import extract_r1_diff

# PacketType numeric convention shared with tools/trace_decoder (frame.py):
# CORE=0, MEM=1, SHIMTILE=2, MEMTILE=3. A compute tile carries two trace
# units multiplexed onto the same packet stream -- a "core" module (kind and
# module both "core") and a "mem" module (kind "core", module "mem") -- while
# memtile/shim entries have no "module" key and pkt_type follows "kind".
_PKT_TYPE_BY_MODULE = {"core": 0, "mem": 1}
_PKT_TYPE_BY_KIND = {"core": 0, "memtile": 3, "shim": 2}


def recover_and_check(measured_events, dwall_events, geometry,
                      *, expect_d_v, expect_contrast, abs_tol=1e-6):
    obs = observe_r1(measured_events, dwall_events, geometry)
    r = extract_r1_diff(obs)
    ok = (math.isclose(r["d_v"], expect_d_v, abs_tol=abs_tol)
          and math.isclose(r["intra_contrast"], expect_contrast, abs_tol=abs_tol))
    return ok, r


def _slot_names_from_trace_config(trace_config):
    """trace_config.json's ``tiles_traced`` list -> the ``slot_names`` dict
    ``parse_trace`` expects (``{pkt_type: {"row,col": [name, ...]}}``).

    The xclbin's trace units are configured statically at kernel-build time
    (baked into the CDO) from exactly this list, so it is the authoritative
    slot->name mapping for a trace-only kernel with no HW register-patch
    step (unlike the dynamic-batch flow in tools/trace_capture.py, which
    builds its label_map from the batch it patches at runtime instead)."""
    out = {}
    for tile in trace_config["tiles_traced"]:
        module = tile.get("module")
        pkt_type = _PKT_TYPE_BY_MODULE.get(module)
        if pkt_type is None:
            pkt_type = _PKT_TYPE_BY_KIND[tile["kind"]]
        key = f"{tile['row']},{tile['col']}"
        out.setdefault(pkt_type, {})[key] = tile["events"]
    return out


def _events_from_bin(path, slot_names):
    # Decode a raw trace.bin (as produced by the in-process runner) to flat
    # dicts. slot_names resolves each trace slot to the event name it was
    # statically configured for (see _slot_names_from_trace_config) --
    # without it every decoded event's name is "" and geometry.json's
    # named anchors can never match (#140 SP-5b Task-5 integration finding).
    from trace_decoder import parse_trace  # tools/ on sys.path
    with open(path, "rb") as f:
        raw = f.read()
    events = parse_trace(raw, slot_names=slot_names)
    return [{"col": e.col, "row": e.row, "pkt_type": e.pkt_type,
             "name": e.name, "soc": e.soc} for e in events]


def main(argv):
    # argv: injected.bin zero.bin geometry.json expect_d_v expect_contrast
    inj, zero, geom_path, d_v, contrast = argv[1:6]
    with open(geom_path) as f:
        geom = json.load(f)
    # trace_config.json is the static per-kernel companion to geometry.json
    # (both authored/generated alongside the source .py under the same
    # mlir-aie test directory) -- it carries the slot->event-name mapping
    # baked into this xclbin's trace units.
    trace_config_path = Path(geom_path).parent / "trace_config.json"
    with open(trace_config_path) as f:
        trace_config = json.load(f)
    slot_names = _slot_names_from_trace_config(trace_config)

    ok, r = recover_and_check(_events_from_bin(inj, slot_names),
                              _events_from_bin(zero, slot_names),
                              geom, expect_d_v=float(d_v),
                              expect_contrast=float(contrast))
    print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
