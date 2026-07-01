"""R1 observation bridge: decoded trace(s) -> pair-difference observations for
r1_diff_extract (#140 SP-5b).

For each deterministic cross-domain pair (a, b) named in geometry.json:
  skew = (soc_meas(a) - soc_meas(b)) - (soc_dwall(a) - soc_dwall(b))
       = module_delay(b) - module_delay(a)
The Delta_wall = soc_dwall(a) - soc_dwall(b) cancels the (deterministic) wall-clock
separation of the two events, leaving the skew. Works identically on emu and
silicon traces (spec Sec.4.2): on emu, dwall = a zero-constants run; on silicon,
an emulator run at zero constants. Anchor = first (min-soc) occurrence of the named
event on that (col,row,pkt_type), mirroring tools/trace_join.anchored_firsts.

Pairs need not literally co-fire (any deterministic pair works), but small
|Delta_wall| keeps emulator Delta_wall-prediction error from swamping the
single-digit skew -- so the kernel picks near-co-firing anchors.
"""

_PKT_KIND = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}


def _kind(pkt_type):
    if pkt_type not in _PKT_KIND:
        raise ValueError(f"unknown pkt_type: {pkt_type!r}")
    return _PKT_KIND[pkt_type]


def _first_soc(events, col, row, pkt_type, name):
    best = None
    for e in events:
        if (e["col"] == col and e["row"] == row
                and e["pkt_type"] == pkt_type and e["name"] == name):
            if best is None or e["soc"] < best:
                best = e["soc"]
    if best is None:
        raise KeyError(f"anchor event not found: {col}|{row}|{pkt_type}|{name}")
    return best


def observe_r1(measured_events, dwall_events, geometry):
    """measured_events, dwall_events: list of flat event dicts
    (trace.events.json schema: col,row,pkt_type,name,soc,...).
    geometry: parsed geometry.json with 'pairs', each = {"a": {...}, "b": {...}}
    where each endpoint carries col,row,pkt_type,name,dn_v.
    Returns list of {"a": {dn_v,kind}, "b": {dn_v,kind}, "skew": float}."""
    out = []
    for p in geometry["pairs"]:
        a, b = p["a"], p["b"]
        sa_m = _first_soc(measured_events, a["col"], a["row"], a["pkt_type"], a["name"])
        sb_m = _first_soc(measured_events, b["col"], b["row"], b["pkt_type"], b["name"])
        sa_d = _first_soc(dwall_events, a["col"], a["row"], a["pkt_type"], a["name"])
        sb_d = _first_soc(dwall_events, b["col"], b["row"], b["pkt_type"], b["name"])
        skew = (sa_m - sb_m) - (sa_d - sb_d)
        out.append({"a": {"dn_v": a["dn_v"], "kind": _kind(a["pkt_type"])},
                    "b": {"dn_v": b["dn_v"], "kind": _kind(b["pkt_type"])},
                    "skew": float(skew)})
    return out
