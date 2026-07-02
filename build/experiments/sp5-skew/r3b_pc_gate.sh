#!/usr/bin/env bash
# SP-5b R3b-PC HW runnability gate (#140, Task 4 -- final task of the R3b-PC
# kernel plan).
#
# AUTHORED NOW; RUNS GREEN ONLY ON PHOENIX AT SP-5c. This script is written
# and reviewed here but is not executed against hardware as part of this
# task -- the HW loop is gated on the SP-5c calibration pass. Do not run this
# against real NPU1 until that gate opens.
#
# Loops the hand-authored sp5_skew_r3b_pc kernel (Task 3, mlir-aie commit
# 128e8f3f4f6) N times on real NPU1: bridge-trace-runner drives the two
# broadcast floods + six per-tile control-packet OP_READ requests, and the
# 24-byte readback (6 x u32 Performance_Counter0 values, counter_index order)
# is recovered from the runner's --trace-out sink -- NOT --output, which only
# ever writes the XRT-declared 8-byte pointer size (Task-3 finding, see
# docs/superpowers/plans/2026-07-01-sp5b-r3b-pc-kernel.md "Host binding" and
# .superpowers/sdd/task-3-report.md Concern #2). r3b_pc_tally.py then checks:
# all runs clean (rc-0, non-empty 24-byte counters, zero dmesg TDR/IOMMU
# delta) + non-inversion (every raw counter nonzero and sane) + range-0
# reproducibility of the per-tile b-vector across runs + rank-sufficiency of
# the {d_h, d_v} extraction. NO skew value is ever asserted -- this is a
# reproducibility/runnability gate only, mirroring r1_gate.sh's proven
# 20-run template (build/experiments/sp5-skew/r1_gate.sh, Task 6 of the R1
# plan) and, further back, SP-3 Task-3's 20-run evidence
# (build/experiments/sp3-spike-trace/task3_gate.sh).
#
# Serial only -- never run concurrently with another HW suite (bridge tests,
# isa-test.sh, another gate, etc. -- they all fight over the one NPU). No
# xrt-smi calls anywhere in this script (segfaults mid-HW-run on this devbox
# per project memory) -- use dmesg-based TDR/IOMMU delta scanning instead, the
# same mechanism scripts/emu-bridge-test.sh and r1_gate.sh use.
set -u

N="${1:-20}"

EMU=/home/triple/npu-work/xdna-emu
BU=/home/triple/npu-work/mlir-aie/test/npu-xrt/sp5_skew_r3b_pc
XCLBIN="$BU/aie.xclbin"
INSTS="$BU/insts.bin"
GEOM="$BU/geometry.json"
RUNNER="$EMU/bridge-runner/build/bridge-trace-runner"
TALLY="$EMU/build/experiments/sp5-skew/r3b_pc_tally.py"
OUT="$EMU/build/experiments/sp5-skew/task4"
TRACE_SIZE=256
COUNTERS_BYTES=24   # 6 tiles x u32, counter_index order (r3b_observe.py)

# ---------------------------------------------------------------------------
# Preflight: fail loudly (never silently skip) if a toolchain input the rest
# of this script depends on is missing. Better a clear message here than a
# confusing failure partway into an N-run HW loop. Note: the --ctrlpkt binary
# itself is NOT checked here -- it is generated fresh per run below (from
# geometry.json via r3b_ctrlpkt.py, which IS checked here as an importable
# source file) so a corrupt/stale ctrlpkt.bin from a prior invocation can
# never leak into a run.
# ---------------------------------------------------------------------------
CTRLPKT_GEN="$EMU/tools/calibration/skew/r3b_ctrlpkt.py"
missing=0
for f in "$XCLBIN" "$INSTS" "$GEOM" "$RUNNER" "$CTRLPKT_GEN" "$TALLY"; do
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
  echo "r3b_pc_gate.sh: preflight failed -- fix the paths above before running." >&2
  echo "  (see xdna-emu/.superpowers/sdd/task-4-brief.md and task-3-report.md for provenance)" >&2
  exit 1
fi

# Count TDR events in dmesg. Same grep pattern as scripts/emu-bridge-test.sh's
# tdr_count() and r1_gate.sh's -- copied verbatim so this script stays
# self-contained rather than sourcing that much larger multi-purpose harness.
tdr_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'aie2_tdr_work') || true
  echo "$n"
}

# Count IOMMU page faults in dmesg. Same grep pattern as emu-bridge-test.sh's
# iommu_fault_count() and r1_gate.sh's. A page fault during this apparatus is
# just as disqualifying as a TDR -- gate on it too rather than silently
# accepting a corrupted run.
iommu_fault_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'IO_PAGE_FAULT') || true
  echo "$n"
}

mkdir -p "$OUT"
# Clear stale per-run output from any prior invocation before starting.
# r3b_pc_tally.py globs run_*/counters.bin, so leftover run_NN dirs from an
# earlier invocation (e.g. a larger N, or a partial failure) would be
# silently mixed into the range-0 check -- a wrong verdict on real hardware,
# exactly when it is most expensive to notice. Every gate run starts clean.
rm -rf "$OUT"/run_*
echo "SP-5b R3b-PC HW runnability gate (#140 Task 4): $N runs on real NPU1"
echo "  runner : $RUNNER"
echo "  xclbin : $XCLBIN"
echo "  geom   : $GEOM"
echo "  out    : $OUT"
echo
echo "AUTHORED NOW, HW-GATED AT SP-5c -- do not invoke against real hardware"
echo "before that gate opens. Serial only -- do not run alongside another HW"
echo "suite. No xrt-smi calls here."
echo

clean=1

for i in $(seq 1 "$N"); do
  rd="$OUT/run_$(printf '%02d' "$i")"
  mkdir -p "$rd"
  CTRLPKT="$rd/ctrlpkt.bin"

  # Generate the --ctrlpkt request binary fresh for this run (deterministic
  # from geometry.json, but regenerated every run rather than shared/cached
  # so a corrupted file from a prior run can never silently carry forward).
  python3 -c "
import json, sys
sys.path.insert(0, '$EMU/tools')
from calibration.skew.r3b_ctrlpkt import build_ctrlpkt
geom = json.load(open('$GEOM'))
open('$CTRLPKT', 'wb').write(build_ctrlpkt(geom))
" >"$rd/ctrlpkt_gen.log" 2>&1
  grc=$?
  if [ $grc -ne 0 ] || [ ! -s "$CTRLPKT" ]; then
    echo "run $i: CTRLPKT GENERATION FAIL rc=$grc (see $rd/ctrlpkt_gen.log)"
    clean=0
    continue
  fi

  tdr_before=$(tdr_count)
  iommu_before=$(iommu_fault_count)

  # Real HW: XDNA_EMU unset. XDNA_EMU_RUNTIME=release is inert here (only
  # consulted by the plugin when XDNA_EMU is set) -- kept for symmetry with
  # r1_gate.sh's proven real-NPU1 invocation.
  env -u XDNA_EMU XDNA_EMU_RUNTIME=release \
    "$RUNNER" \
    --xclbin "$XCLBIN" --instr "$INSTS" \
    --ctrlpkt "$CTRLPKT" --trace-out "$rd/trace.bin" \
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

  # Readback = first 24 bytes of the --trace-out sink (6 x u32 counters, in
  # counter_index order). NOT --output -- that path only ever dumps the
  # XRT-declared 8-byte pointer size and cannot carry the 24-byte readback
  # (Task-3 finding; see the header comment above and task-3-report.md
  # Concern #2). Extracted here so the tally only ever deals in fixed-size
  # counters.bin files, independent of --trace-size.
  head -c "$COUNTERS_BYTES" "$rd/trace.bin" >"$rd/counters.bin"
  csz=$(wc -c <"$rd/counters.bin" 2>/dev/null || echo 0)
  if [ "$csz" -ne "$COUNTERS_BYTES" ]; then
    echo "run $i: SHORT READBACK got=$csz want=$COUNTERS_BYTES bytes (trace.bin truncated?)"
    clean=0
    continue
  fi

  echo "run $i: ok  rc=$rc tdr_delta=$tdr_delta iommu_delta=$iommu_delta counters=${csz}B"
done

echo
if [ "$clean" -ne 1 ]; then
  echo "GATE: FAIL -- not all $N HW runs were clean (see per-run RUNNER/DMESG/READBACK lines above)."
  echo "Not running the tally -- fix runnability first (no point range-checking incomplete data)."
  exit 1
fi

echo "All $N HW runs completed clean (rc-0, 24-byte readback, zero dmesg TDR/IOMMU delta)."
echo
echo "Tally (non-inversion + range-0 + rank-sufficiency; NO value asserted):"
# Two positional args only -- deliberately never pass a 3rd (max_range).
# r3b_pc_tally.py's argv-driven `main(*sys.argv[1:])` would hand max_range
# through as a *string*, and its own comparisons (`r <= max_range`) are
# float<=str, which raises TypeError rather than the intended AssertionError.
# Omitting it lets Python's own keyword default (max_range=0, an int) apply --
# which is also exactly the literal "range-0" bar this gate requires, not a
# configurable slop. Mirrors r1_gate.sh's identical convention (see its
# comment at the equivalent call site).
(cd "$EMU" && python3 "$TALLY" "$OUT/run_*/counters.bin" "$GEOM")
exit $?
