#!/usr/bin/env bash
# SP-3 Task-3 reproduction gate (#140 timer-sync arc).
#
# Loops the FROZEN compiled spike (build_q0_rich_trace) N times on real NPU1:
# run bridge-trace-runner -> decode event_time -> per-run events.json. The
# tally step (task3_tally.py) then checks the gate criteria across all runs:
# stable structure (5 tiles, ~1250 events, no TDR) + PERF_CNT_2 anchor on
# every core every run.
#
# Serial only -- never concurrent with any other HW suite. No xrt-smi here.
set -u

N="${1:-20}"
EMU=/home/triple/npu-work/xdna-emu
SPIKE=/home/triple/npu-work/mlir-aie/test/npu-xrt/spike_bringup
BU="$SPIKE/build_q0_rich_trace"
RUNNER="$EMU/bridge-runner/build/bridge-trace-runner"
DECODE="$EMU/tools/parse-trace.py"
PRJ_MLIR="$BU/aie_traced.mlir.prj/input_with_addresses.mlir"
OUT="$EMU/build/experiments/sp3-spike-trace/task3"
TRACE_SIZE=16384

# Count NPU reset (TDR) events in dmesg. Same grep pattern as
# scripts/emu-bridge-test.sh's tdr_count(). No xrt-smi here (segfaults
# mid-HW-run on this devbox); dmesg delta scanning is the safe mechanism.
tdr_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'aie2_tdr_work') || true
  echo "${n:-0}"
}
# Count IOMMU page faults in dmesg. A fault during a trace capture means a BD
# drove a DMA past its buffer (the bridge-runner output-BO under-allocation
# bug, #140) -- just as disqualifying as a TDR: it silently corrupts DDR when
# the overrun lands on a mapped neighbour, so "clean rc-0" runs can be lies.
iommu_fault_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'IO_PAGE_FAULT') || true
  echo "${n:-0}"
}

mkdir -p "$OUT"
echo "Task-3 gate: $N runs of frozen spike on real NPU1"
echo "  runner : $RUNNER"
echo "  xclbin : $BU/aie.xclbin"
echo "  out    : $OUT"
echo

for i in $(seq 1 "$N"); do
  rd="$OUT/run_$(printf '%02d' "$i")"
  mkdir -p "$rd"
  # Snapshot dmesg TDR/IOMMU counts to attribute any fault to this run.
  tdr_before=$(tdr_count)
  iommu_before=$(iommu_fault_count)
  # HW run (real NPU: XDNA_EMU unset). release .so only matters if emu; harmless here.
  env -u XDNA_EMU XDNA_EMU_RUNTIME=release \
    "$RUNNER" \
    --xclbin "$BU/aie.xclbin" --instr "$BU/insts.bin" \
    --output "$rd/out.bin" --trace-out "$rd/trace.bin" \
    --trace-size "$TRACE_SIZE" >"$rd/runner.log" 2>&1
  rc=$?
  tdr_after=$(tdr_count)
  iommu_after=$(iommu_fault_count)
  tdr_delta=$((tdr_after - tdr_before))
  iommu_delta=$((iommu_after - iommu_before))
  echo "$tdr_delta" >"$rd/tdr_delta.txt"
  echo "$iommu_delta" >"$rd/iommu_delta.txt"
  if [ "$tdr_delta" -ne 0 ] || [ "$iommu_delta" -ne 0 ]; then
    echo "run $i: DMESG ALARM tdr_delta=$tdr_delta iommu_delta=$iommu_delta -- NOT clean (data suspect)"
    echo "$rc" >"$rd/runner.rc"
    continue
  fi
  if [ $rc -ne 0 ] || [ ! -s "$rd/trace.bin" ]; then
    echo "run $i: RUNNER FAIL rc=$rc trace=$( [ -s "$rd/trace.bin" ] && echo present || echo empty)"
    echo "$rc" >"$rd/runner.rc"
    continue
  fi
  echo "$rc" >"$rd/runner.rc"
  # Decode event_time -> Perfetto + events.json
  python3 "$DECODE" --trace-bin "$rd/trace.bin" \
    --xclbin-mlir "$PRJ_MLIR" \
    --trace-mode event_time \
    --out-events "$rd/events.json" \
    --out-perfetto "$rd/perfetto.json" >"$rd/decode.log" 2>&1
  drc=$?
  if [ $drc -ne 0 ]; then
    echo "run $i: DECODE FAIL rc=$drc (see $rd/decode.log)"
    continue
  fi
  nev=$(python3 -c "import json;print(len(json.load(open('$rd/events.json'))['events']))" 2>/dev/null)
  echo "run $i: ok  rc=$rc tdr_delta=$tdr_delta iommu_delta=$iommu_delta events=$nev"
done

echo
echo "Done. Tally with: python3 $EMU/build/experiments/sp3-spike-trace/task3_tally.py"
