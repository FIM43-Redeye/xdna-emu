#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# trace-bridge.sh -- End-to-end trace comparison: emulator vs hardware.
#
# Injects hardware tracing into an mlir-aie test, runs it on both the real
# NPU and the emulator (via XRT bridge), decodes both trace outputs with
# mlir-aie's parse_trace(), and produces a comparison report.
#
# Prerequisites:
#   - mlir-aie Python environment active (source activate-npu-env.sh)
#   - XRT installed at /opt/xilinx/xrt
#   - Emulator plugin built: xrt-plugin/build/libxrt_driver_emu.so.2
#   - For hardware runs: NPU device accessible
#
# Usage:
#   ./tools/trace-bridge.sh add_one_using_dma            # full pipeline
#   ./tools/trace-bridge.sh --no-hw add_one_using_dma     # emulator only
#   ./tools/trace-bridge.sh --no-emu vec_vec_add           # hardware only
#   ./tools/trace-bridge.sh --compile add_one_using_dma   # force recompile

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLIR_AIE="${EMU_ROOT}/../mlir-aie"
TEST_SRC="${MLIR_AIE}/test/npu-xrt"
BUILD_BASE="${MLIR_AIE}/build/test/npu-xrt"
RESULTS_BASE="/tmp/trace-bridge-results"
TRACE_SIZE=1048576  # 1MB default

# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

FILTER=""
RUN_HW=true
RUN_EMU=true
FORCE_COMPILE=false
DEBUG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-hw)       RUN_HW=false; shift ;;
    --no-emu)      RUN_EMU=false; shift ;;
    --compile)     FORCE_COMPILE=true; shift ;;
    --debug)       DEBUG=true; shift ;;
    --trace-size)  TRACE_SIZE="$2"; shift 2 ;;
    --help|-h)
      cat <<'USAGE'
Usage: trace-bridge.sh [options] <test-name>

Compare emulator vs hardware trace output for an mlir-aie npu-xrt test.

Options:
  --no-hw         Skip hardware run (emulator only)
  --no-emu        Skip emulator run (hardware only)
  --compile       Force recompile traced xclbin
  --debug         Enable debug output
  --trace-size N  Trace buffer size in bytes (default: 1048576)

The test name is a substring match against test directories in
mlir-aie/test/npu-xrt/ (e.g., "add_one" matches "add_one_using_dma").
USAGE
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      FILTER="$1"; shift ;;
  esac
done

if [[ -z "$FILTER" ]]; then
  echo "Error: test name required" >&2
  echo "Usage: trace-bridge.sh [options] <test-name>" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

info()  { echo ">>> $*"; }
warn()  { echo "WARNING: $*" >&2; }
die()   { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Locate test
# ---------------------------------------------------------------------------

info "Locating test matching '$FILTER'..."
TEST_DIR=""
for dir in "$TEST_SRC"/*/; do
  name="$(basename "$dir")"
  if [[ "$name" == *"$FILTER"* ]]; then
    if [[ -n "$TEST_DIR" ]]; then
      die "Ambiguous filter '$FILTER': matches both $(basename "$TEST_DIR") and $name"
    fi
    TEST_DIR="$dir"
  fi
done

if [[ -z "$TEST_DIR" ]]; then
  die "No test found matching '$FILTER' in $TEST_SRC"
fi

TEST_NAME="$(basename "$TEST_DIR")"
info "Test: $TEST_NAME ($TEST_DIR)"

# Results directory for this run
RESULTS_DIR="${RESULTS_BASE}/${TEST_NAME}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"
info "Results: $RESULTS_DIR"

# ---------------------------------------------------------------------------
# Phase 1: Trace injection
# ---------------------------------------------------------------------------

TRACED_DIR="$RESULTS_DIR/traced"
MANIFEST="$TRACED_DIR/manifest.json"

if [[ "$FORCE_COMPILE" == true ]] || [[ ! -f "$MANIFEST" ]]; then
  info "Phase 1: Injecting trace into $TEST_NAME..."
  python3 "$SCRIPT_DIR/trace-inject.py" "$TEST_DIR" \
    --output "$TRACED_DIR" \
    --trace-size "$TRACE_SIZE" \
    || die "Trace injection failed"
else
  info "Phase 1: Using cached trace injection"
fi

# Check if test was skipped (already traced)
if python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
if m.get('skipped'):
    print(f'Test skipped: {m.get(\"reason\", \"unknown\")}')
    sys.exit(2)
" 2>/dev/null; then
  : # not skipped, continue
elif [[ $? -eq 2 ]]; then
  die "Test already has trace configuration"
fi

# ---------------------------------------------------------------------------
# Phase 2: Compile traced xclbin
# ---------------------------------------------------------------------------

XCLBIN="$TRACED_DIR/aie.xclbin"
INSTS="$TRACED_DIR/insts.bin"

if [[ "$FORCE_COMPILE" == true ]] || [[ ! -f "$XCLBIN" ]]; then
  info "Phase 2: Compiling traced xclbin..."

  MLIR_FILE="$TRACED_DIR/aie_traced.mlir"
  [[ -f "$MLIR_FILE" ]] || die "Traced MLIR not found: $MLIR_FILE"

  # Read extra aiecc flags if present
  EXTRA_FLAGS=""
  if [[ -f "$TRACED_DIR/.aiecc-extra-flags" ]]; then
    EXTRA_FLAGS="$(cat "$TRACED_DIR/.aiecc-extra-flags" | tr '\n' ' ')"
  fi

  # Compile with aiecc.py (same flags as run.lit, plus traced MLIR)
  # shellcheck disable=SC2086
  ( cd "$TRACED_DIR" && nice -n 19 aiecc.py \
    --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
    --no-compile-host --alloc-scheme=basic-sequential \
    --no-xchesscc \
    --xclbin-name=aie.xclbin --npu-insts-name=insts.bin \
    $EXTRA_FLAGS \
    ./aie_traced.mlir \
  ) 2>&1 | tee "$RESULTS_DIR/compile.log" \
    || die "Compilation failed (see $RESULTS_DIR/compile.log)"

  [[ -f "$XCLBIN" ]] || die "xclbin not produced by aiecc.py"
  [[ -f "$INSTS" ]] || die "insts.bin not produced by aiecc.py"
  info "  xclbin: $XCLBIN"
  info "  insts:  $INSTS"
else
  info "Phase 2: Using cached xclbin"
fi

# ---------------------------------------------------------------------------
# Phase 3: Run on hardware
# ---------------------------------------------------------------------------

HW_DIR="$RESULTS_DIR/hw"
mkdir -p "$HW_DIR"

if [[ "$RUN_HW" == true ]]; then
  info "Phase 3: Running on hardware..."
  python3 "$SCRIPT_DIR/trace-run.py" \
    "$MANIFEST" \
    --output-dir "$HW_DIR" \
    2>&1 | tee "$RESULTS_DIR/hw-run.log" \
    || warn "Hardware run failed (see $RESULTS_DIR/hw-run.log)"
else
  info "Phase 3: Skipped (--no-hw)"
fi

# ---------------------------------------------------------------------------
# Phase 4: Run on emulator
# ---------------------------------------------------------------------------

EMU_DIR="$RESULTS_DIR/emu"
mkdir -p "$EMU_DIR"

if [[ "$RUN_EMU" == true ]]; then
  info "Phase 4: Running on emulator..."

  # The emulator runs through the XRT bridge: the same test.exe with
  # XDNA_EMU=1 routes through our plugin instead of the real driver.
  # But trace-run.py uses pyxrt directly, so we need to set XDNA_EMU=1.
  XDNA_EMU=1 python3 "$SCRIPT_DIR/trace-run.py" \
    "$MANIFEST" \
    --output-dir "$EMU_DIR" \
    2>&1 | tee "$RESULTS_DIR/emu-run.log" \
    || warn "Emulator run failed (see $RESULTS_DIR/emu-run.log)"
else
  info "Phase 4: Skipped (--no-emu)"
fi

# ---------------------------------------------------------------------------
# Phase 5: Compare traces
# ---------------------------------------------------------------------------

info "Phase 5: Comparing traces..."

HW_TRACE="$HW_DIR/trace.json"
EMU_TRACE="$EMU_DIR/trace.json"

compare_traces() {
  # Compare two Perfetto JSON trace files.
  # Extract event sequences and report differences.
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json
import sys
from pathlib import Path

def load_trace(path):
    """Load Perfetto JSON and extract duration events."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        events = json.load(f)
    # Filter to duration events (ph=X: complete, ph=B/E: begin/end)
    durations = [e for e in events if e.get("ph") in ("X", "B", "E")]
    return durations

def event_key(e):
    """Create a comparable key for an event."""
    return (e.get("pid", 0), e.get("tid", 0), e.get("name", ""), e.get("ph", ""))

def summarize_events(events):
    """Count events by name."""
    counts = {}
    for e in events:
        name = e.get("name", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts

hw_path, emu_path, report_path = sys.argv[1], sys.argv[2], sys.argv[3]

hw_events = load_trace(hw_path)
emu_events = load_trace(emu_path)

report_lines = []
report_lines.append("=" * 72)
report_lines.append("Trace Comparison Report")
report_lines.append("=" * 72)
report_lines.append("")

if hw_events is None and emu_events is None:
    report_lines.append("No trace data available from either source.")
    report = "\n".join(report_lines)
    print(report)
    Path(report_path).write_text(report + "\n")
    sys.exit(0)

if hw_events is None:
    report_lines.append("Hardware trace: NOT AVAILABLE")
    report_lines.append(f"Emulator trace: {len(emu_events)} events")
    report_lines.append("")
    emu_counts = summarize_events(emu_events)
    report_lines.append("Emulator event summary:")
    for name, count in sorted(emu_counts.items()):
        report_lines.append(f"  {name}: {count}")
    report = "\n".join(report_lines)
    print(report)
    Path(report_path).write_text(report + "\n")
    sys.exit(0)

if emu_events is None:
    report_lines.append(f"Hardware trace: {len(hw_events)} events")
    report_lines.append("Emulator trace: NOT AVAILABLE")
    report_lines.append("")
    hw_counts = summarize_events(hw_events)
    report_lines.append("Hardware event summary:")
    for name, count in sorted(hw_counts.items()):
        report_lines.append(f"  {name}: {count}")
    report = "\n".join(report_lines)
    print(report)
    Path(report_path).write_text(report + "\n")
    sys.exit(0)

# Both traces available -- compare
report_lines.append(f"Hardware trace: {len(hw_events)} events")
report_lines.append(f"Emulator trace: {len(emu_events)} events")
report_lines.append("")

hw_counts = summarize_events(hw_events)
emu_counts = summarize_events(emu_events)
all_names = sorted(set(hw_counts.keys()) | set(emu_counts.keys()))

report_lines.append(f"{'Event Type':<40} {'HW':>6} {'EMU':>6} {'Delta':>6}")
report_lines.append("-" * 60)

mismatches = 0
for name in all_names:
    hw_n = hw_counts.get(name, 0)
    emu_n = emu_counts.get(name, 0)
    delta = emu_n - hw_n
    marker = ""
    if delta != 0:
        marker = " <--"
        mismatches += 1
    report_lines.append(f"{name:<40} {hw_n:>6} {emu_n:>6} {delta:>+6}{marker}")

report_lines.append("")

# Timing comparison: for events present in both, compare timestamps
hw_by_pid_tid = {}
for e in hw_events:
    key = (e.get("pid", 0), e.get("tid", 0))
    hw_by_pid_tid.setdefault(key, []).append(e)

emu_by_pid_tid = {}
for e in emu_events:
    key = (e.get("pid", 0), e.get("tid", 0))
    emu_by_pid_tid.setdefault(key, []).append(e)

common_keys = set(hw_by_pid_tid.keys()) & set(emu_by_pid_tid.keys())
if common_keys:
    report_lines.append("Timing comparison (first 10 events per tile):")
    report_lines.append("")
    for key in sorted(common_keys):
        hw_tile = hw_by_pid_tid[key][:10]
        emu_tile = emu_by_pid_tid[key][:10]
        report_lines.append(f"  Tile pid={key[0]} tid={key[1]}:")
        for i, (h, e) in enumerate(zip(hw_tile, emu_tile)):
            h_ts = h.get("ts", 0)
            e_ts = e.get("ts", 0)
            h_name = h.get("name", "?")
            e_name = e.get("name", "?")
            match = "OK" if h_name == e_name else "MISMATCH"
            report_lines.append(
                f"    [{i:2d}] hw={h_name:<20s} ts={h_ts:<8} "
                f"emu={e_name:<20s} ts={e_ts:<8} {match}"
            )
        report_lines.append("")

# Summary
report_lines.append("=" * 72)
if mismatches == 0:
    report_lines.append("RESULT: All event types match between hardware and emulator.")
else:
    report_lines.append(f"RESULT: {mismatches} event type(s) differ between hardware and emulator.")
report_lines.append("=" * 72)

report = "\n".join(report_lines)
print(report)
Path(report_path).write_text(report + "\n")
PYEOF
}

REPORT="$RESULTS_DIR/comparison-report.txt"

if [[ "$RUN_HW" == true ]] || [[ "$RUN_EMU" == true ]]; then
  compare_traces "$HW_TRACE" "$EMU_TRACE" "$REPORT"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

info ""
info "Trace bridge complete for $TEST_NAME"
info "Results:    $RESULTS_DIR/"
[[ -f "$HW_TRACE" ]]  && info "  HW trace:  $HW_TRACE"
[[ -f "$EMU_TRACE" ]] && info "  EMU trace: $EMU_TRACE"
[[ -f "$REPORT" ]]    && info "  Report:    $REPORT"
info ""
info "View traces in Perfetto: https://ui.perfetto.dev/"
info "  Drag and drop the trace.json files to compare visually."
