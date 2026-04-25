#!/usr/bin/env bash
# Regenerate the *.expected.json oracle outputs for every .bin fixture
# in this directory.
#
# Run from anywhere; resolves paths relative to the script location.
# Requires the mlir-aie ironenv Python (it imports aie.utils.trace).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures"
NPU_WORK="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IRONPYTHON="${NPU_WORK}/mlir-aie/ironenv/bin/python3"
MLIR_AIE_PY="${NPU_WORK}/mlir-aie/install/python"

if [[ ! -x "${IRONPYTHON}" ]]; then
    echo "error: ironenv python not found at ${IRONPYTHON}" >&2
    echo "       run 'source mlir-aie/ironenv/bin/activate' once to bootstrap" >&2
    exit 1
fi

shopt -s nullglob
for bin in "${FIXTURE_DIR}"/*.bin; do
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

echo "done."
