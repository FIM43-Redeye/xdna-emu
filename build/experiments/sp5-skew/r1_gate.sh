#!/usr/bin/env bash
# SP-5b R1 HW runnability gate (#140, Task 6 -- final task of Phase 1).
#
# Loops sp5_skew_r1 (chess-compiled) N times on real NPU1: bridge-trace-runner
# -> decode event_time -> per-run events.json. Once, runs the SAME xclbin
# through the emulator plugin (zero broadcast-timing constants, the compiled-in
# default -- see src/device/state/effects.rs's
# broadcast_timing_consts_default_to_zero invariant) to produce the
# Delta_wall baseline dwall.events.json that tools/calibration/skew/r1_observe.py
# subtracts out. r1_tally.py then checks: all runs clean (rc-0, non-empty
# trace, zero dmesg TDR/IOMMU delta) + per-pair skew range-0 across runs +
# non-degeneracy (>=3 distinct core dn_v). NO skew value is ever asserted --
# this is a reproducibility/runnability gate only, mirroring SP-3 Task-3's
# 20-run evidence (build/experiments/sp3-spike-trace/task3_gate.sh, the
# template this script is cloned from).
#
# Serial only -- never run concurrently with another HW suite (bridge tests,
# isa-test.sh, another gate, etc. -- they all fight over the one NPU). No
# xrt-smi calls anywhere in this script (segfaults mid-HW-run on this devbox
# per project memory) -- use dmesg-based TDR/IOMMU delta scanning instead,
# the same mechanism scripts/emu-bridge-test.sh uses (tdr_count/
# iommu_fault_count below are the same dmesg grep patterns, reproduced here
# so this script stays self-contained rather than sourcing that much larger
# multi-purpose harness).
set -u

N="${1:-20}"

EMU=/home/triple/npu-work/xdna-emu
BU=/home/triple/npu-work/mlir-aie/build/test/npu-xrt/sp5_skew_r1/chess
XCLBIN="$BU/aie.xclbin"
INSTS="$BU/insts.bin"
PRJ_MLIR="$BU/aie_traced.mlir.prj/input_with_addresses.mlir"
GEOM=/home/triple/npu-work/mlir-aie/test/npu-xrt/sp5_skew_r1/geometry.json
RUNNER="$EMU/bridge-runner/build/bridge-trace-runner"
DECODE="$EMU/tools/parse-trace.py"
NORMALIZE="$EMU/build/experiments/sp5-skew/normalize_placement.py"
TALLY="$EMU/build/experiments/sp5-skew/r1_tally.py"
OUT="$EMU/build/experiments/sp5-skew/task6"
TRACE_SIZE=16384

# ---------------------------------------------------------------------------
# Preflight: fail loudly (never silently skip) if a toolchain input the rest
# of this script depends on is missing. Better a clear message here than a
# confusing failure 15 runs into a 20-run HW loop.
# ---------------------------------------------------------------------------
missing=0
for f in "$XCLBIN" "$INSTS" "$PRJ_MLIR" "$GEOM" "$RUNNER" "$DECODE" "$NORMALIZE" "$TALLY"; do
  if [ ! -e "$f" ]; then
    echo "MISSING REQUIRED INPUT: $f" >&2
    missing=1
  fi
done
if [ -e "$RUNNER" ] && [ ! -x "$RUNNER" ]; then
  echo "NOT EXECUTABLE: $RUNNER" >&2
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "r1_gate.sh: preflight failed -- fix the paths above before running." >&2
  echo "  (see xdna-emu/.superpowers/sdd/task-6-brief.md for where each comes from)" >&2
  exit 1
fi

# Count TDR events in dmesg. Same grep pattern as scripts/emu-bridge-test.sh's
# tdr_count(). "TDR" (Timeout Detection & Recovery) *is* the NPU reset event
# on Phoenix, so this single counter covers both "TDR" and "reset" scanning.
tdr_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'aie2_tdr_work') || true
  echo "$n"
}

# Count IOMMU page faults in dmesg. Same grep pattern as emu-bridge-test.sh's
# iommu_fault_count(). Not explicitly required by the brief, but a page
# fault during this apparatus is just as disqualifying as a TDR -- gate on
# it too rather than silently accepting a corrupted run.
iommu_fault_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'IO_PAGE_FAULT') || true
  echo "$n"
}

mkdir -p "$OUT"
echo "SP-5b R1 HW runnability gate (#140 Task 6): $N runs of sp5_skew_r1 (chess) on real NPU1"
echo "  runner : $RUNNER"
echo "  xclbin : $XCLBIN"
echo "  geom   : $GEOM"
echo "  out    : $OUT"
echo
echo "Serial only -- do not run alongside another HW suite. No xrt-smi calls here."
echo

clean=1

for i in $(seq 1 "$N"); do
  rd="$OUT/run_$(printf '%02d' "$i")"
  mkdir -p "$rd"

  tdr_before=$(tdr_count)
  iommu_before=$(iommu_fault_count)

  # Real HW: XDNA_EMU unset. XDNA_EMU_RUNTIME=release is inert here (only
  # consulted by the plugin when XDNA_EMU is set) -- kept for byte-identical
  # symmetry with the dwall invocation below and with task3_gate.sh's own
  # proven real-NPU1 invocation.
  env -u XDNA_EMU XDNA_EMU_RUNTIME=release \
    "$RUNNER" \
    --xclbin "$XCLBIN" --instr "$INSTS" \
    --output "$rd/out.bin" --trace-out "$rd/trace.bin" \
    --trace-size "$TRACE_SIZE" >"$rd/runner.log" 2>&1
  rc=$?
  echo "$rc" >"$rd/runner.rc"

  tdr_after=$(tdr_count)
  iommu_after=$(iommu_fault_count)
  tdr_delta=$((tdr_after - tdr_before))
  iommu_delta=$((iommu_after - iommu_before))
  echo "$tdr_delta" >"$rd/tdr_delta.txt"
  echo "$iommu_delta" >"$rd/iommu_delta.txt"

  if [ "$tdr_delta" -ne 0 ] || [ "$iommu_delta" -ne 0 ]; then
    echo "run $i: DMESG ALARM tdr_delta=$tdr_delta iommu_delta=$iommu_delta -- NOT clean"
    clean=0
  fi

  if [ $rc -ne 0 ] || [ ! -s "$rd/trace.bin" ]; then
    echo "run $i: RUNNER FAIL rc=$rc trace=$( [ -s "$rd/trace.bin" ] && echo present || echo empty)"
    clean=0
    continue
  fi

  python3 "$DECODE" --trace-bin "$rd/trace.bin" \
    --xclbin-mlir "$PRJ_MLIR" \
    --trace-mode event_time \
    --out-events "$rd/events.json" \
    --out-perfetto "$rd/perfetto.json" >"$rd/decode.log" 2>&1
  drc=$?
  if [ $drc -ne 0 ] || [ ! -s "$rd/events.json" ]; then
    echo "run $i: DECODE FAIL rc=$drc (see $rd/decode.log)"
    clean=0
    continue
  fi

  # Normalize the decoded events' (col,row) back to the MLIR-declared frame
  # geometry.json uses (col=0). bridge-trace-runner places this kernel at
  # its real allocation-scheme column (observed origin_col=1 for sibling
  # SP-3/SP-5b kernels on this 2-column-device build -- see
  # mlir-aie/test/npu-xrt/sp5_skew_r1/README.md and
  # build/experiments/sp3-spike-trace/task3_tally.py's independent real-NPU1
  # confirmation of the same col0->col1 shift), while parse-trace.py's
  # --out-events output leaves raw/physical (col,row) untouched and instead
  # reports the shift in a top-level "placement" field (see
  # tools/parse-trace.py's payload["placement"] = {origin_col, origin_row}).
  # Without this step, r1_observe.observe_r1's exact-match anchor lookups
  # raise KeyError against every real decoded trace. This normalization is
  # NOT part of r1_tally.py (kept byte-for-byte verbatim per the task brief)
  # -- it lives here, in glue this script owns, mirroring the same
  # origin_col/origin_row contract already used by tools/shim-chain-fit.py,
  # tools/shim-throughput-fit.py, and src/trace/compare.rs.
  python3 "$NORMALIZE" "$rd/events.json"

  nev=$(python3 -c "import json;print(len(json.load(open('$rd/events.json'))['events']))" 2>/dev/null)
  echo "run $i: ok  rc=$rc tdr_delta=$tdr_delta iommu_delta=$iommu_delta events=$nev"
done

echo
echo "--- Delta_wall (emu, zero broadcast-timing constants) ---"
echo "Zero is the compiled-in plugin default (broadcast_timing_consts_default_to_zero"
echo "in src/device/state/effects.rs), so no override injection is needed here -- the"
echo "Task-3 override seam is in-process-only and does not apply on this plugin path"
echo "anyway. NOTE: if the plugin default is ever changed to a calibrated (non-zero)"
echo "value, this step will need explicit XDNA_EMU_* env vars to force zero."
dwd="$OUT/dwall"
mkdir -p "$dwd"
env XDNA_EMU=1 XDNA_EMU_RUNTIME=release \
  "$RUNNER" \
  --xclbin "$XCLBIN" --instr "$INSTS" \
  --output "$dwd/out.bin" --trace-out "$dwd/trace.bin" \
  --trace-size "$TRACE_SIZE" >"$dwd/runner.log" 2>&1
dwrc=$?
echo "$dwrc" >"$dwd/runner.rc"
if [ $dwrc -ne 0 ] || [ ! -s "$dwd/trace.bin" ]; then
  echo "dwall: RUNNER FAIL rc=$dwrc (see $dwd/runner.log)"
  clean=0
else
  python3 "$DECODE" --trace-bin "$dwd/trace.bin" \
    --xclbin-mlir "$PRJ_MLIR" \
    --trace-mode event_time \
    --out-events "$dwd/events.json" \
    --out-perfetto "$dwd/perfetto.json" >"$dwd/decode.log" 2>&1
  dwdrc=$?
  if [ $dwdrc -ne 0 ] || [ ! -s "$dwd/events.json" ]; then
    echo "dwall: DECODE FAIL rc=$dwdrc (see $dwd/decode.log)"
    clean=0
  else
    python3 "$NORMALIZE" "$dwd/events.json"
    echo "dwall: ok (zero-constants emu baseline for Delta_wall subtraction)"
  fi
fi

echo
if [ "$clean" -ne 1 ]; then
  echo "GATE: FAIL -- not all $N HW runs + dwall were clean (see per-run RUNNER/DECODE/DMESG lines above)."
  echo "Not running the tally -- fix runnability first (no point range-checking incomplete data)."
  exit 1
fi

echo "All $N HW runs + 1 emu dwall run completed clean (rc-0, non-empty trace, zero dmesg TDR/IOMMU delta)."
echo
echo "Tally (range-0 + non-degeneracy, NO value asserted):"
# Three positional args only -- deliberately never pass a 4th (max_range).
# r1_tally.py's argv-driven `main(*sys.argv[1:])` would hand max_range through
# as a *string*, and its own comparisons (`r <= max_range`) are float<=str,
# which raises TypeError rather than the intended AssertionError. Omitting it
# lets Python's own keyword default (max_range=0, an int) apply -- which is
# also exactly the literal "range-0" bar the Phase-1 done criteria require,
# not a configurable slop. See task-6-report.md for the full writeup.
(cd "$EMU" && python3 "$TALLY" "$OUT/run_*/events.json" "$dwd/events.json" "$GEOM")
exit $?
