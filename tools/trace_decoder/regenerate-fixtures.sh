#!/usr/bin/env bash
# Regenerate the *.expected.json oracle outputs for every .bin fixture
# in this directory.
#
# Mode 0 fixtures (named ``mode0_*.bin``) are checked against
# mlir-aie's ``convert_to_commands`` (Apache 2.0 oracle).  Mode 1
# fixtures (named ``mode1_*.bin``) have no public oracle and are
# instead frozen by replaying our own decoder; this is a regression
# test, not a cross-validation, but it lets us spot drift when we
# refactor the decoder.
#
# Run from anywhere; resolves paths relative to the script location.
# Mode-0 regen requires the mlir-aie ironenv Python (it imports
# aie.utils.trace).  Mode-1 regen uses the project's own decoder via
# system Python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NPU_WORK="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IRONPYTHON="${NPU_WORK}/mlir-aie/ironenv/bin/python3"
MLIR_AIE_PY="${NPU_WORK}/mlir-aie/install/python"

shopt -s nullglob

# --- mode 0 -- mlir-aie oracle ---------------------------------------
if [[ ! -x "${IRONPYTHON}" ]]; then
    echo "warning: ironenv python not found at ${IRONPYTHON}" >&2
    echo "         skipping mode-0 oracle regen" >&2
else
  for bin in "${FIXTURE_DIR}"/mode0_*.bin; do
    name="${bin%.bin}"
    out="${name}.expected.json"
    echo "regenerating $(basename "${out}") from $(basename "${bin}")..."
    PYTHONPATH="${MLIR_AIE_PY}" "${IRONPYTHON}" - "${bin}" "${out}" <<'PY'
import sys
import numpy as np
import json
from aie.utils.trace.utils import (
    split_trace_segments,
    trim_trace_pkts,
    trace_pkts_de_interleave,
    convert_to_byte_stream,
    convert_to_commands,
)
from aie.utils.trace.parse import check_for_valid_trace
from aie.utils.trace.events import NUM_TRACE_TYPES

bin_path, out_path = sys.argv[1], sys.argv[2]
raw = np.fromfile(bin_path, dtype=np.uint32)
# Trim trailing zero padding.
last = len(raw) - 1
while last >= 0 and raw[last] == 0:
    last -= 1
trimmed = raw[: last + 8]

trace_pkts = [f"{int(w):08x}" for w in trimmed]
segments = split_trace_segments(trace_pkts)

merged = [dict() for _ in range(NUM_TRACE_TYPES)]
for seg in segments:
    if not check_for_valid_trace("<regen>", seg):
        continue
    seg_trimmed = trim_trace_pkts(seg)
    sp = trace_pkts_de_interleave(seg_trimmed)
    for t in range(NUM_TRACE_TYPES):
        for loc, data in sp[t].items():
            merged[t].setdefault(loc, []).extend(data)

bs = convert_to_byte_stream(merged)
cmds = convert_to_commands(bs, False)
with open(out_path, "w") as f:
    json.dump({"trace_types": cmds}, f, indent=2)
PY
  done

  # --- mode 0 perfetto B/E oracle --------------------------------------
  # Freeze mlir-aie convert_commands_to_json's B/E timeline (reduced to
  # per-slot span durations) so rebuild_perfetto_mode0 is cross-validated,
  # not just the byte->command layer. Upstream omits the trailing E for a
  # still-asserted level; we close it at the true end-of-trace timer (which
  # is what our decoder does and is the correct span end), so the oracle and
  # our output are apples-to-apples.
  for bin in "${FIXTURE_DIR}"/mode0_*.bin; do
    name="${bin%.bin}"
    out="${name}.perfetto_expected.json"
    echo "regenerating $(basename "${out}") from $(basename "${bin}")..."
    PYTHONPATH="${MLIR_AIE_PY}:${TOOLS_DIR}" "${IRONPYTHON}" - "${bin}" "${out}" <<'PY'
import sys, json
import numpy as np
from collections import defaultdict
from trace_decoder import decode_words, TraceMode
from trace_decoder.frame import StartCmd, EventCmd, RepeatCmd
from aie.utils.trace.parse import convert_commands_to_json, NUM_EVENTS
import aie.utils.trace.parse as _p
_p.lookup_event_name_by_type = lambda tt, code, mod: f"slot{code}"

bin_path, out_path = sys.argv[1], sys.argv[2]
raw = np.fromfile(bin_path, dtype=np.uint32).tolist()
per_tile = decode_words(raw, mode=TraceMode.EVENT_TIME)

def final_timer(cmds):
    # convert_commands_to_json starts its timer at 0 and ignores Start.
    t = 0; prev = None
    for c in cmds:
        if isinstance(c, StartCmd): prev = None
        elif isinstance(c, EventCmd): t += 1 + c.cycles; prev = c
        elif isinstance(c, RepeatCmd):
            if prev is None: continue
            t += c.count if prev.cycles == 0 else c.count * (1 + prev.cycles)
    return t

def cmd_to_d(c):
    if isinstance(c, EventCmd):
        d = {"type": "Multiple", "cycles": c.cycles}
        for i, s in enumerate([s for s in range(8) if c.event_bits & (1 << s)]):
            d[f"event{i if i else ''}"] = s
        return d
    if isinstance(c, RepeatCmd): return {"type": "Repeat", "repeats": c.count}
    return None

out = {}
for (pt, row, col), cmds in per_tile.items():
    dcmds = [cmd_to_d(c) for c in cmds if cmd_to_d(c)]
    commands = [{f"{row},{col}": dcmds}]
    pe = [{f"{row},{col}": {k: k for k in range(8)}}]
    pe[0][f"{row},{col}"][NUM_EVENTS] = pt
    te = []
    convert_commands_to_json(te, commands, pe, events_module=None)
    end = final_timer(cmds)
    stacks = defaultdict(list); spans = defaultdict(list)
    for e in sorted(te, key=lambda x: (x["ts"], 0 if x["ph"] == "E" else 1)):
        if e["ph"] == "B": stacks[e["tid"]].append(e["ts"])
        elif stacks[e["tid"]]: b = stacks[e["tid"]].pop(); spans[e["tid"]].append(e["ts"] - b)
    for tid, opens in stacks.items():
        for b in opens: spans[tid].append(end - b)
    out[f"{pt},{row},{col}"] = {str(s): sorted(spans[s]) for s in sorted(spans) if spans[s]}
with open(out_path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
PY
  done
fi

# --- mode 1 -- frozen replay of our own decoder ----------------------
# These have no public oracle; a freeze drift means we changed
# decoder behaviour (which may or may not be intentional -- review
# carefully when this regenerates).
for bin in "${FIXTURE_DIR}"/mode1_*.bin; do
    [[ "${bin}" == *_expected.json ]] && continue
    name="${bin%.bin}"
    out="${name}_core_expected.json"
    echo "regenerating $(basename "${out}") from $(basename "${bin}")..."
    PYTHONPATH="${TOOLS_DIR}" python3 - "${bin}" "${out}" <<'PY'
import json
import sys
import numpy as np
from trace_decoder import decode_words, TraceMode
from trace_decoder.frame import StartCmd, EventCmd, RepeatCmd, SyncCmd

bin_path, out_path = sys.argv[1], sys.argv[2]
raw = np.fromfile(bin_path, dtype=np.uint32).tolist()
cmds = decode_words(raw, mode=TraceMode.EVENT_PC)
core_key = (0, 2, 1)  # CORE tile, row=2, col=1 -- the convention for our captures
core_cmds = cmds.get(core_key, [])

def to_dict(cmd):
    if isinstance(cmd, StartCmd):
        return {"type": "Start", "timer_value": cmd.timer_value}
    if isinstance(cmd, EventCmd):
        return {"type": "EventPC", "event_bits": cmd.event_bits, "pc": cmd.cycles}
    if isinstance(cmd, RepeatCmd):
        return {"type": "Repeat", "count": cmd.count}
    if isinstance(cmd, SyncCmd):
        return {"type": "Sync"}
    raise AssertionError(cmd)

with open(out_path, "w") as f:
    json.dump([to_dict(c) for c in core_cmds], f, indent=2)
PY
done

echo "done."
