#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# emu-bridge-test.sh -- Phased parallel comparison of emulator vs hardware.
#
# Runs mlir-aie NPU integration tests through two execution paths and
# produces a comparison matrix:
#
#   1. XRT bridge + emulator  (XDNA_EMU=1 ./test.exe)
#   2. XRT bridge + real hardware  (./test.exe with real BDF)
#
# Five-phase architecture:
#   Phase 1: Discover     -- find tests, filter, skip npu2-only
#   Phase 2: Compile      -- parallel xclbin + test.exe builds
#   Phase 3: Run hardware -- serial (NPU jams under parallel load)
#   Phase 4: Run emulator -- parallel -j$(nproc)
#   Phase 5: Report       -- comparison matrix + summary
#
# Usage:
#   ./scripts/emu-bridge-test.sh                    # all tests, emulator + hardware
#   ./scripts/emu-bridge-test.sh add_one_using_dma  # single test
#   ./scripts/emu-bridge-test.sh --no-hw            # emulator only, skip hardware
#   ./scripts/emu-bridge-test.sh --compile          # force recompile xclbins
#   ./scripts/emu-bridge-test.sh --list             # list available tests
#   ./scripts/emu-bridge-test.sh -j4 add_one        # limit parallelism

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLIR_AIE="${EMU_ROOT}/../mlir-aie"
TEST_SRC="${MLIR_AIE}/test/npu-xrt"
BUILD_BASE="${MLIR_AIE}/build/test/npu-xrt"

# XRT paths
XRT_DIR="/opt/xilinx/xrt"
XRT_INCLUDE="${XRT_DIR}/include"
XRT_LIB="${XRT_DIR}/lib"

# test_utils from mlir-aie build
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
TEST_UTILS_INCLUDE="${TEST_LIB_DIR}/include"
TEST_UTILS_LIB="${TEST_LIB_DIR}/lib"

# aietools
AIETOOLS_DIR="${AIETOOLS_DIR:-/home/triple/npu-work/aietools}"

# Results directory -- one per day, phases append into it
RESULTS_DIR="/tmp/emu-bridge-results-$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

# Default parallelism
JOBS="$(nproc)"

# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

FILTER=""
FORCE_COMPILE=false
RUN_HW=true
LIST_ONLY=false
VERBOSE=false
TRACE_MODE=""  # "", "default", "all", "sweep", "sweep-all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)     FORCE_COMPILE=true; shift ;;
    --no-hw)       RUN_HW=false; shift ;;
    --list)        LIST_ONLY=true; shift ;;
    -v|--verbose)  VERBOSE=true; shift ;;
    --trace)       TRACE_MODE="default"; shift ;;
    --trace=*)
      TRACE_MODE="${1#--trace=}"
      case "$TRACE_MODE" in
        all|sweep|sweep-all) ;;
        *) echo "Unknown --trace mode: $TRACE_MODE (use: all, sweep, sweep-all)" >&2; exit 1 ;;
      esac
      shift ;;
    -j*)
      JOBS="${1#-j}"
      if [[ -z "$JOBS" ]] || ! [[ "$JOBS" =~ ^[0-9]+$ ]]; then
        echo "Invalid -j value: $1" >&2; exit 1
      fi
      shift ;;
    --help|-h)
      cat <<'USAGE'
Usage: emu-bridge-test.sh [options] [test-name-filter]

Options:
  --compile       Force recompile all xclbins (default: use cached)
  --no-hw         Skip real hardware runs (default: hardware enabled)
  --list          List available tests and exit
  --trace         Run trace comparison (default events, passing tests only)
  --trace=all     Trace all tests (pass + fail)
  --trace=sweep   Full event sweep (passing tests only)
  --trace=sweep-all  Full sweep, all tests
  -jN             Override parallelism (default: nproc)
  -v, --verbose   Show log snippets on failure

Filter:
  Substring match on test directory name.
  e.g., 'add_one' matches add_one_using_dma, add_one_two, etc.
USAGE
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      FILTER="$1"; shift ;;
  esac
done

# Export variables that parallel jobs need.
export RESULTS_DIR FORCE_COMPILE VERBOSE TRACE_MODE
export MLIR_AIE TEST_SRC BUILD_BASE EMU_ROOT
export XRT_DIR XRT_INCLUDE XRT_LIB
export TEST_LIB_DIR TEST_UTILS_INCLUDE TEST_UTILS_LIB
export AIETOOLS_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

info() { echo ">>> $*"; }
err()  { echo "ERROR: $*" >&2; }

# ---------------------------------------------------------------------------
# Helpers (shared with parallel jobs via export -f)
# ---------------------------------------------------------------------------

# Check if a test directory has a standard test.cpp + aie.mlir/run.lit.
is_standard_test() {
  local dir="$1"
  [[ -f "$dir/test.cpp" ]] && [[ -f "$dir/aie.mlir" || -f "$dir/run.lit" ]]
}

# Check if a test requires npu2 (AIE2P -- skip for now).
requires_npu2() {
  local dir="$1"
  local lit="$dir/run.lit"
  [[ -f "$lit" ]] && grep -q 'ryzen_ai_npu2' "$lit" && ! grep -q 'ryzen_ai_npu1' "$lit" && return 0
  return 1
}

# Extract NPUDEVICE substitution from run.lit.
get_npu_device() {
  local lit="$1/run.lit"
  if [[ -f "$lit" ]] && grep -q 'npu1_1col' "$lit"; then
    echo "npu1_1col"
  elif [[ -f "$lit" ]] && grep -q 'npu1_4col' "$lit"; then
    echo "npu1_4col"
  else
    echo "npu1_1col"
  fi
}

# Apply lit-style substitutions to a RUN line.
apply_lit_subs() {
  local src_dir="$1"
  local cmd="$2"
  cmd="${cmd#*RUN: }"
  cmd="${cmd//'%S'/$src_dir}"
  cmd="${cmd//'%aietools'/$AIETOOLS_DIR}"
  cmd="${cmd//'%python '/}"
  cmd="${cmd//'%python'/python3}"
  cmd="${cmd//'%xrt_flags'/-I$XRT_INCLUDE -L$XRT_LIB -luuid -lxrt_coreutil}"
  cmd="${cmd//'%test_utils_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_utils}"
  cmd="${cmd//'%test_lib_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_lib}"
  cmd="${cmd//'%run_on_npu1%'/}"
  cmd="${cmd//'%run_on_npu2%'/}"
  cmd="${cmd#"${cmd%%[![:space:]]*}"}"
  cmd="${cmd%"${cmd##*[![:space:]]}"}"
  if [[ "$cmd" == clang\ * ]]; then
    cmd="/usr/bin/clang++ ${cmd#clang }"
  elif [[ "$cmd" == g++\ * ]]; then
    cmd="/usr/bin/g++ ${cmd#g++ }"
  fi
  echo "$cmd"
}

# Parse run.lit and extract build commands (everything except execution lines).
extract_build_commands() {
  local lit_file="$1"
  local src_dir="$2"
  [[ -f "$lit_file" ]] || return 1

  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    local cmd
    cmd="$(apply_lit_subs "$src_dir" "$line")"
    [[ "$cmd" == *"./test.exe"* ]] && continue
    [[ "$cmd" == *"run_on_npu"* ]] && continue
    [[ "$cmd" == *"NPUDEVICE"* ]] && continue
    [[ "$cmd" == "cp "* ]] && continue
    echo "$cmd"
  done < "$lit_file"
}

# Extract the test execution command from run.lit.
get_run_cmd() {
  local src_dir="$1"
  local lit_file="$src_dir/run.lit"
  [[ -f "$lit_file" ]] || return 1

  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    if [[ "$line" == *"./test.exe"* ]] && [[ "$line" == *"npu1"* || "$line" != *"npu2"* ]]; then
      local cmd
      cmd="$(apply_lit_subs "$src_dir" "$line")"
      cmd="${cmd#"${cmd%%[![:space:]]*}"}"
      echo "$cmd"
      return 0
    fi
  done < "$lit_file"

  echo "./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin"
}

# Sanitize test name for use as filename (replace / with _).
sanitize_name() {
  echo "${1//\//_}"
}

# Export all helpers for xargs subshells.
export -f is_standard_test requires_npu2 get_npu_device apply_lit_subs
export -f extract_build_commands get_run_cmd sanitize_name

# ---------------------------------------------------------------------------
# Phase 1: Discover tests
# ---------------------------------------------------------------------------

discover_tests() {
  local tests=()

  for dir in "$TEST_SRC"/*/; do
    local name
    name="$(basename "$dir")"

    [[ "$name" == "makefile-common" ]] && continue
    [[ "$name" == "lit.local.cfg" ]] && continue
    [[ "$name" == "core_dmas" ]] && continue

    if [[ -n "$FILTER" ]] && [[ "$name" != *"$FILTER"* ]]; then
      continue
    fi

    if ! is_standard_test "$dir"; then
      continue
    fi

    tests+=("$name")
  done

  # Also check subdirectories (e.g., core_dmas/*).
  for subdir in "$TEST_SRC"/*/*/; do
    [[ -d "$subdir" ]] || continue
    local parent child name
    parent="$(basename "$(dirname "$subdir")")"
    child="$(basename "$subdir")"
    name="${parent}/${child}"

    if [[ -n "$FILTER" ]] && [[ "$name" != *"$FILTER"* ]]; then
      continue
    fi

    if is_standard_test "$subdir"; then
      tests+=("$name")
    fi
  done

  printf '%s\n' "${tests[@]}" | sort
}

# ---------------------------------------------------------------------------
# Phase 2: Compile (parallel xclbin + test.exe)
# ---------------------------------------------------------------------------

# Compile one test's xclbin and test.exe. Called via xargs.
# Writes $RESULTS_DIR/$safe.compile.result and $RESULTS_DIR/$safe.compile.log.
compile_one() {
  local name="$1"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local build_dir="$BUILD_BASE/$name"
  local log_file="$RESULTS_DIR/${safe}.compile.log"
  local result_file="$RESULTS_DIR/${safe}.compile.result"
  local lit_file="$src_dir/run.lit"

  mkdir -p "$build_dir"
  : > "$log_file"

  if [[ ! -f "$lit_file" ]]; then
    echo "FAIL" > "$result_file"
    echo "No run.lit found" >> "$log_file"
    echo "  COMPILE $name: FAIL (no run.lit)"
    return 0
  fi

  # --- 2a: xclbin build (expensive, skip if cached) -----------------------

  local have_xclbin=false
  if [[ -f "$build_dir/aie.xclbin" ]] || ls "$build_dir"/*.xclbin &>/dev/null; then
    have_xclbin=true
  fi

  local cached=false
  if $have_xclbin && [[ "$FORCE_COMPILE" != "true" ]]; then
    cached=true
  else
    # Prepare architecture MLIR (NPUDEVICE substitution).
    local npu_dev
    npu_dev="$(get_npu_device "$src_dir")"
    if [[ -f "$src_dir/aie.mlir" ]]; then
      cp "$src_dir/aie.mlir" "$build_dir/aie_arch.mlir"
      sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch.mlir"
    fi

    local failed=false
    while IFS= read -r cmd; do
      [[ -z "$cmd" ]] && continue
      # Skip host compilation -- handled in 2b.
      [[ "$cmd" == *clang*test.cpp* ]] && continue
      [[ "$cmd" == *g++*test.cpp* ]] && continue
      # Fix aie.mlir references in aiecc.py commands.
      if [[ "$cmd" == *aiecc.py* ]]; then
        cmd="${cmd//$src_dir\/aie.mlir/./aie_arch.mlir}"
        cmd="${cmd//\.\/aie.mlir/./aie_arch.mlir}"
      fi
      if ! ( cd "$build_dir" && nice -n 19 bash -c "$cmd" ) >> "$log_file" 2>&1; then
        failed=true
        break
      fi
    done < <(extract_build_commands "$lit_file" "$src_dir")

    if $failed; then
      echo "FAIL" > "$result_file"
      echo "  COMPILE $name: FAIL"
      return 0
    fi

    # Verify xclbin was produced.
    if [[ ! -f "$build_dir/aie.xclbin" ]]; then
      local any_xclbin
      any_xclbin=$(find "$build_dir" -name "*.xclbin" -print -quit 2>/dev/null || true)
      if [[ -z "$any_xclbin" ]]; then
        echo "FAIL" > "$result_file"
        echo "  COMPILE $name: FAIL (no xclbin produced)"
        return 0
      fi
    fi
  fi

  # --- 2b: test.exe build (always, BDF patch must be present) --------------

  if [[ -f "$src_dir/test.cpp" ]]; then
    sed \
      -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
      -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
      "$src_dir/test.cpp" > "$build_dir/test.cpp"
  fi

  # Find the clang/g++ line from run.lit for correct flags.
  local clang_cmd=""
  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    local cmd
    cmd="$(apply_lit_subs "$src_dir" "$line")"
    if [[ "$cmd" == *clang*test.cpp* ]] || [[ "$cmd" == *g++*test.cpp* ]]; then
      cmd="${cmd//$src_dir\/test.cpp/./test.cpp}"
      clang_cmd="$cmd"
      break
    fi
  done < "$lit_file"

  if [[ -n "$clang_cmd" ]]; then
    if ! ( cd "$build_dir" && bash -c "$clang_cmd" ) >> "$log_file" 2>&1; then
      echo "FAIL" > "$result_file"
      echo "  COMPILE $name: FAIL (test.exe)"
      return 0
    fi
  else
    if ! /usr/bin/clang++ "$build_dir/test.cpp" -o "$build_dir/test.exe" \
        -std=c++17 -Wall \
        -I"$XRT_INCLUDE" -L"$XRT_LIB" \
        -I"$TEST_UTILS_INCLUDE" -L"$TEST_UTILS_LIB" \
        -luuid -lxrt_coreutil -ltest_utils -lrt -lstdc++ \
        >> "$log_file" 2>&1; then
      echo "FAIL" > "$result_file"
      echo "  COMPILE $name: FAIL (test.exe fallback)"
      return 0
    fi
  fi

  echo "OK" > "$result_file"
  if $cached; then
    echo "  COMPILE $name: OK (cached)"
  else
    echo "  COMPILE $name: OK"
  fi
}
export -f compile_one

# ---------------------------------------------------------------------------
# Phase 3: Hardware runs (serial)
# ---------------------------------------------------------------------------

run_one_hardware() {
  local name="$1"
  local bdf="$2"
  local safe
  safe="$(sanitize_name "$name")"
  local build_dir="$BUILD_BASE/$name"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${safe}.hw.log"
  local result_file="$RESULTS_DIR/${safe}.hw.result"

  if [[ ! -f "$build_dir/test.exe" ]]; then
    echo "SKIP" > "$result_file"
    return
  fi

  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP" > "$result_file"
    return
  fi

  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  local rc=0
  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$bdf"
    timeout 30 bash -c "$run_cmd"
  ) > "$log_file" 2>&1 || rc=$?

  if [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    echo "PASS" > "$result_file"
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT" > "$result_file"
  else
    echo "FAIL" > "$result_file"
  fi
}

# ---------------------------------------------------------------------------
# Phase 4: Emulator runs (parallel)
# ---------------------------------------------------------------------------

run_one_bridge() {
  local name="$1"
  local safe
  safe="$(sanitize_name "$name")"
  local build_dir="$BUILD_BASE/$name"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${safe}.bridge.log"
  local result_file="$RESULTS_DIR/${safe}.bridge.result"

  if [[ ! -f "$build_dir/test.exe" ]]; then
    echo "SKIP" > "$result_file"
    echo "  BRIDGE $name: SKIP (no test.exe)"
    return
  fi

  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP" > "$result_file"
    echo "  BRIDGE $name: SKIP (no xclbin)"
    return
  fi

  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  local rc=0
  (
    cd "$build_dir"
    export XDNA_EMU=1
    export XDNA_EMU_LOG_LEVEL="${XDNA_EMU_LOG_LEVEL:-info}"
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    timeout 120 bash -c "$run_cmd"
  ) > "$log_file" 2>&1 || rc=$?
  local result
  if [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    result="PASS"
  elif [[ $rc -eq 124 ]]; then
    result="TIMEOUT"
  else
    result="FAIL"
  fi

  # Verify emulator actually ran (catch silent fallthrough to real NPU).
  if [[ "$result" == "PASS" ]]; then
    if ! grep -qE '(Loaded PDI|xdna_emu|XDNA emulator)' "$log_file"; then
      result="EMU_MISS"
    fi
  fi

  echo "$result" > "$result_file"
  echo "  BRIDGE $name: $result"
}
export -f run_one_bridge

# ---------------------------------------------------------------------------
# Phase 4b: Trace comparison
# ---------------------------------------------------------------------------

# Run trace comparison for a single test.
# Uses trace-sweep.py for sweep mode, or trace-inject + trace-run +
# trace-compare for default (8-event) mode.
#
# Writes $RESULTS_DIR/${safe}.trace.summary (one line) and full report
# to $RESULTS_DIR/${safe}.trace.log.
trace_one_test() {
  local name="$1"
  local mode="$2"  # "default" or "sweep"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local trace_dir="$RESULTS_DIR/${safe}.trace"
  local summary_file="$RESULTS_DIR/${safe}.trace.summary"
  local log_file="$RESULTS_DIR/${safe}.trace.log"
  local tools_dir="$EMU_ROOT/tools"

  mkdir -p "$trace_dir"
  : > "$log_file"

  if [[ "$mode" == "sweep" || "$mode" == "sweep-all" ]]; then
    # Full sweep: delegate to trace-sweep.py
    local sweep_args=("$src_dir" -o "$trace_dir/sweep")
    if [[ "$RUN_HW" != "true" ]]; then
      sweep_args+=(--no-hw)
    fi
    if ! python3 "$tools_dir/trace-sweep.py" "${sweep_args[@]}" >> "$log_file" 2>&1; then
      echo "ERROR sweep_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (sweep failed)"
      return
    fi

    # Trim trace buffers to actual data length
    python3 "$tools_dir/trace-trim.py" --dir "$trace_dir/sweep" >> "$log_file" 2>&1 || true

    # Compare
    if ! python3 "$tools_dir/trace-compare.py" --sweep "$trace_dir/sweep" \
        -o "$trace_dir/report.txt" >> "$log_file" 2>&1; then
      echo "ERROR compare_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (compare failed)"
      return
    fi
  else
    # Default 8-event mode: inject, compile, run HW+EMU, compare.

    # Step 1: Inject tracing into MLIR
    local traced_dir="$trace_dir/traced"
    if ! python3 "$tools_dir/trace-inject.py" "$src_dir" -o "$traced_dir" \
        >> "$log_file" 2>&1; then
      echo "ERROR injection_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (injection failed)"
      return
    fi

    local manifest="$traced_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
      echo "ERROR no_manifest" > "$summary_file"
      echo "  TRACE $name: ERROR (no manifest)"
      return
    fi

    # Check if injection was skipped (unsupported test)
    if python3 -c "import json,sys; m=json.load(open('$manifest')); sys.exit(0 if m.get('skipped') else 1)" 2>/dev/null; then
      echo "SKIP unsupported" > "$summary_file"
      echo "  TRACE $name: SKIP (unsupported)"
      return
    fi

    # Step 2: Compile traced xclbin
    if ! ( cd "$traced_dir" && nice -n 19 aiecc.py \
        --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
        --no-compile-host --alloc-scheme=basic-sequential --no-xchesscc \
        --xclbin-name=aie.xclbin --npu-insts-name=insts.bin \
        ./aie_traced.mlir ) >> "$log_file" 2>&1; then
      echo "ERROR compile_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (compile failed)"
      return
    fi

    # Step 3+4: Run HW and EMU concurrently.
    # HW is I/O-bound (NPU), EMU is CPU-bound -- no contention.
    local hw_dir="$trace_dir/hw"
    local emu_dir="$trace_dir/emu"
    local hw_ok=true emu_ok=true

    # Launch EMU in background
    (
      XDNA_EMU=1 XRT_DEVICE_BDF="ffff:ff:1f.0" \
        python3 "$tools_dir/trace-run.py" "$manifest" -o "$emu_dir" \
        >> "$log_file.emu" 2>&1
    ) &
    local emu_pid=$!

    # Run HW in foreground (with cooldown after)
    if [[ "$RUN_HW" == "true" ]]; then
      if ! python3 "$tools_dir/trace-run.py" "$manifest" -o "$hw_dir" \
          >> "$log_file.hw" 2>&1; then
        hw_ok=false
      fi
      # Cooldown: let driver settle before next test's HW run
      sleep 2
    fi

    # Wait for EMU
    if ! wait "$emu_pid"; then
      emu_ok=false
    fi

    # Merge sub-logs
    cat "$log_file.hw" >> "$log_file" 2>/dev/null || true
    cat "$log_file.emu" >> "$log_file" 2>/dev/null || true
    rm -f "$log_file.hw" "$log_file.emu"

    if ! $hw_ok && [[ "$RUN_HW" == "true" ]]; then
      echo "ERROR hw_run_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (hw run failed)"
      return
    fi
    if ! $emu_ok; then
      echo "ERROR emu_run_failed" > "$summary_file"
      echo "  TRACE $name: ERROR (emu run failed)"
      return
    fi

    # Trim trace buffers to actual data length
    python3 "$tools_dir/trace-trim.py" --dir "$trace_dir" >> "$log_file" 2>&1 || true

    # Step 5: Compare (only if both traces exist)
    if [[ "$RUN_HW" == "true" ]] && [[ -f "$trace_dir/hw/trace_raw.bin" ]] \
        && [[ -f "$trace_dir/emu/trace_raw.bin" ]]; then
      if ! python3 "$tools_dir/trace-compare.py" \
          --hw "$trace_dir/hw/trace_raw.bin" \
          --emu "$trace_dir/emu/trace_raw.bin" \
          -o "$trace_dir/report.txt" >> "$log_file" 2>&1; then
        echo "ERROR compare_failed" > "$summary_file"
        echo "  TRACE $name: ERROR (compare failed)"
        return
      fi
    else
      # EMU-only: no comparison possible, just record that trace was collected
      echo "EMU_ONLY collected" > "$summary_file"
      echo "  TRACE $name: EMU_ONLY (trace collected, no HW to compare)"
      return
    fi
  fi

  # Parse the report to produce a one-line summary.
  local report="$trace_dir/report.txt"
  if [[ ! -f "$report" ]]; then
    echo "ERROR no_report" > "$summary_file"
    echo "  TRACE $name: ERROR (no report)"
    return
  fi

  # Extract key stats from the report's summary section.
  local edge_line level_line pairs_line
  edge_line="$(grep '^Edge event types:' "$report" 2>/dev/null || true)"
  pairs_line="$(grep '^  Pairs:' "$report" 2>/dev/null || true)"

  if [[ -n "$edge_line" ]]; then
    # Parse: "Edge event types:    N clean, M diverged, K count mismatch"
    local clean diverged
    clean="$(echo "$edge_line" | grep -oP '\d+ clean' | grep -oP '\d+')"
    diverged="$(echo "$edge_line" | grep -oP '\d+ diverged' | grep -oP '\d+')"
    local pairs="0"
    if [[ -n "$pairs_line" ]]; then
      pairs="$(echo "$pairs_line" | grep -oP '\d+')"
    fi

    if [[ "${diverged:-0}" -eq 0 ]]; then
      echo "CLEAN ${clean:-0} event types, ${pairs} pairs" > "$summary_file"
      echo "  TRACE $name: CLEAN (${clean:-0} event types, ${pairs} pairs)"
    else
      echo "DIVERGE ${diverged} of $((${clean:-0}+${diverged})) event types" > "$summary_file"
      echo "  TRACE $name: DIVERGE (${diverged} of $((${clean:-0}+${diverged})) event types)"
    fi
  else
    echo "UNKNOWN parse_error" > "$summary_file"
    echo "  TRACE $name: UNKNOWN (could not parse report)"
  fi
}
export -f trace_one_test

# ---------------------------------------------------------------------------
# Phase 5: Report
# ---------------------------------------------------------------------------

print_report() {
  local -n test_list=$1
  local run_hw=$2
  local has_trace=false
  [[ -n "$TRACE_MODE" ]] && has_trace=true

  echo ""
  echo "==========================================================================="
  echo "  RESULTS"
  echo "==========================================================================="
  echo ""

  # Header row adapts to which columns are active.
  if [[ "$run_hw" == "true" ]] && $has_trace; then
    printf "%-40s  %-10s  %-10s  %-5s  %s\n" "TEST" "BRIDGE" "HARDWARE" "MATCH" "TRACE"
    printf "%-40s  %-10s  %-10s  %-5s  %s\n" \
      "----------------------------------------" \
      "----------" "----------" "-----" "------------------------------"
  elif [[ "$run_hw" == "true" ]]; then
    printf "%-45s  %-10s  %-10s  %-5s\n" "TEST" "BRIDGE" "HARDWARE" "MATCH"
    printf "%-45s  %-10s  %-10s  %-5s\n" \
      "---------------------------------------------" \
      "----------" "----------" "-----"
  elif $has_trace; then
    printf "%-40s  %-10s  %s\n" "TEST" "BRIDGE" "TRACE"
    printf "%-40s  %-10s  %s\n" \
      "----------------------------------------" "----------" "------------------------------"
  else
    printf "%-45s  %-10s\n" "TEST" "BRIDGE"
    printf "%-45s  %-10s\n" \
      "---------------------------------------------" "----------"
  fi

  local pass_bridge=0 fail_bridge=0 skip_bridge=0 timeout_bridge=0 emumiss_bridge=0
  local pass_hw=0 fail_hw=0 skip_hw=0 timeout_hw=0
  local match_count=0 mismatch_count=0
  local trace_clean=0 trace_diverge=0 trace_error=0 trace_skip=0

  for name in "${test_list[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"

    # Read compile result -- skip tests that failed to compile.
    local compile_result="OK"
    if [[ -f "$RESULTS_DIR/${safe}.compile.result" ]]; then
      compile_result="$(< "$RESULTS_DIR/${safe}.compile.result")"
    fi
    if [[ "$compile_result" != "OK" ]]; then
      if [[ "$run_hw" == "true" ]] && $has_trace; then
        printf "%-40s  %-10s  %-10s  %-5s  %s\n" "$name" "COMP_FAIL" "COMP_FAIL" "-" "-"
      elif [[ "$run_hw" == "true" ]]; then
        printf "%-45s  %-10s  %-10s  %-5s\n" "$name" "COMP_FAIL" "COMP_FAIL" "-"
      elif $has_trace; then
        printf "%-40s  %-10s  %s\n" "$name" "COMP_FAIL" "-"
      else
        printf "%-45s  %-10s\n" "$name" "COMP_FAIL"
      fi
      ((skip_bridge++)) || true
      ((skip_hw++)) || true
      continue
    fi

    # Read bridge result.
    local bridge_result="SKIP"
    if [[ -f "$RESULTS_DIR/${safe}.bridge.result" ]]; then
      bridge_result="$(< "$RESULTS_DIR/${safe}.bridge.result")"
    fi
    case "$bridge_result" in
      PASS)     ((pass_bridge++)) || true ;;
      TIMEOUT)  ((timeout_bridge++)); ((fail_bridge++)) || true ;;
      EMU_MISS) ((emumiss_bridge++)); ((fail_bridge++)) || true ;;
      SKIP*)    ((skip_bridge++)) || true ;;
      *)        ((fail_bridge++)) || true ;;
    esac

    # Read hardware result.
    local hw_result="SKIP"
    if [[ "$run_hw" == "true" ]] && [[ -f "$RESULTS_DIR/${safe}.hw.result" ]]; then
      hw_result="$(< "$RESULTS_DIR/${safe}.hw.result")"
    fi
    if [[ "$run_hw" == "true" ]]; then
      case "$hw_result" in
        PASS)    ((pass_hw++)) || true ;;
        TIMEOUT) ((timeout_hw++)); ((fail_hw++)) || true ;;
        SKIP*)   ((skip_hw++)) || true ;;
        *)       ((fail_hw++)) || true ;;
      esac
    fi

    # Read trace summary (if tracing was enabled).
    local trace_summary="-"
    if $has_trace && [[ -f "$RESULTS_DIR/${safe}.trace.summary" ]]; then
      trace_summary="$(< "$RESULTS_DIR/${safe}.trace.summary")"
      case "$trace_summary" in
        CLEAN*)   ((trace_clean++)) || true ;;
        DIVERGE*) ((trace_diverge++)) || true ;;
        ERROR*)   ((trace_error++)) || true ;;
        SKIP*|EMU_ONLY*) ((trace_skip++)) || true ;;
      esac
    elif $has_trace; then
      ((trace_skip++)) || true
    fi

    # Compute match column.
    local match="-"
    if [[ "$run_hw" == "true" ]]; then
      if [[ "$bridge_result" == "SKIP"* ]] || [[ "$hw_result" == "SKIP"* ]]; then
        match="-"
      elif [[ "$bridge_result" == "$hw_result" ]]; then
        match="yes"
        ((match_count++)) || true
      else
        match="NO"
        ((mismatch_count++)) || true
      fi
    fi

    # Print row.
    if [[ "$run_hw" == "true" ]] && $has_trace; then
      printf "%-40s  %-10s  %-10s  %-5s  %s\n" "$name" "$bridge_result" "$hw_result" "$match" "$trace_summary"
    elif [[ "$run_hw" == "true" ]]; then
      printf "%-45s  %-10s  %-10s  %-5s\n" "$name" "$bridge_result" "$hw_result" "$match"
    elif $has_trace; then
      printf "%-40s  %-10s  %s\n" "$name" "$bridge_result" "$trace_summary"
    else
      printf "%-45s  %-10s\n" "$name" "$bridge_result"
    fi

    # Verbose: show log tail on failure.
    if [[ "$VERBOSE" == "true" ]] && [[ "$bridge_result" != "PASS" ]] && [[ "$bridge_result" != "SKIP"* ]]; then
      local logf="$RESULTS_DIR/${safe}.bridge.log"
      if [[ -f "$logf" ]]; then
        echo "    --- bridge log tail ---"
        tail -5 "$logf" | sed 's/^/    /'
      fi
    fi
    if [[ "$VERBOSE" == "true" ]] && [[ "$run_hw" == "true" ]] && \
       [[ "$hw_result" != "PASS" ]] && [[ "$hw_result" != "SKIP"* ]]; then
      local logf="$RESULTS_DIR/${safe}.hw.log"
      if [[ -f "$logf" ]]; then
        echo "    --- hw log tail ---"
        tail -5 "$logf" | sed 's/^/    /'
      fi
    fi
  done

  # Summary.
  echo ""
  echo "=== Summary ==="
  echo "Bridge:   $pass_bridge pass, $fail_bridge fail, $skip_bridge skip"
  if [[ $timeout_bridge -gt 0 ]]; then
    echo "          ($timeout_bridge timeout)"
  fi
  if [[ $emumiss_bridge -gt 0 ]]; then
    echo "          ($emumiss_bridge EMU_MISS -- emulator not detected in log!)"
  fi
  if [[ "$run_hw" == "true" ]]; then
    echo "Hardware: $pass_hw pass, $fail_hw fail, $skip_hw skip"
    if [[ $timeout_hw -gt 0 ]]; then
      echo "          ($timeout_hw timeout)"
    fi
    echo "Match:    $match_count match, $mismatch_count mismatch"
  fi
  if $has_trace; then
    echo "Trace:    $trace_clean clean, $trace_diverge diverge, $trace_error error, $trace_skip skip"
  fi
  echo "Logs:     $RESULTS_DIR/"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  # ---- Phase 1: Discover -------------------------------------------------

  mapfile -t tests < <(discover_tests)

  if [[ ${#tests[@]} -eq 0 ]]; then
    err "No tests found${FILTER:+ matching '$FILTER'}"
    exit 1
  fi

  if $LIST_ONLY; then
    info "Available tests (${#tests[@]}):"
    printf '  %s\n' "${tests[@]}"
    exit 0
  fi

  info "Found ${#tests[@]} test(s), parallelism: -j${JOBS}"
  info "Results in: $RESULTS_DIR"
  echo ""

  # Filter out npu2-only tests before running phases.
  local runnable=()
  local skipped_npu2=0
  for name in "${tests[@]}"; do
    if requires_npu2 "$TEST_SRC/$name"; then
      local safe
      safe="$(sanitize_name "$name")"
      echo "SKIP_NPU2" > "$RESULTS_DIR/${safe}.bridge.result"
      echo "SKIP_NPU2" > "$RESULTS_DIR/${safe}.compile.result"
      ((skipped_npu2++)) || true
    else
      runnable+=("$name")
    fi
  done

  if [[ $skipped_npu2 -gt 0 ]]; then
    info "Skipped $skipped_npu2 npu2-only test(s)"
  fi

  # ---- Phase 2: Compile --------------------------------------------------

  info "Phase 2: Compiling ${#runnable[@]} test(s) (-j${JOBS})"

  printf '%s\n' "${runnable[@]}" | xargs -P"$JOBS" -I{} bash -c 'compile_one "$@"' _ {}

  # Count compile results.
  local compile_ok=0 compile_fail=0
  for name in "${runnable[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"
    local cr="OK"
    [[ -f "$RESULTS_DIR/${safe}.compile.result" ]] && cr="$(< "$RESULTS_DIR/${safe}.compile.result")"
    if [[ "$cr" == "OK" ]]; then
      ((compile_ok++)) || true
    else
      ((compile_fail++)) || true
    fi
  done
  info "Phase 2 done: $compile_ok OK, $compile_fail failed"
  echo ""

  # Build list of tests that compiled successfully.
  local compiled=()
  for name in "${runnable[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"
    local cr="FAIL"
    [[ -f "$RESULTS_DIR/${safe}.compile.result" ]] && cr="$(< "$RESULTS_DIR/${safe}.compile.result")"
    if [[ "$cr" == "OK" ]]; then
      compiled+=("$name")
    fi
  done

  # ---- Phase 3: Hardware runs (serial, opt-in) ---------------------------

  if $RUN_HW; then
    local real_bdf
    real_bdf="$(xrt-smi examine 2>/dev/null | grep -oP '\[0000:[0-9a-f:\.]+\]' | head -1 | tr -d '[]')" || true

    if [[ -z "$real_bdf" ]]; then
      info "Phase 3: SKIPPED -- no NPU hardware detected"
      RUN_HW=false
    else
      info "Phase 3: Running ${#compiled[@]} test(s) on hardware (serial, BDF=$real_bdf)"
      local hw_done=0
      for name in "${compiled[@]}"; do
        ((hw_done++)) || true
        run_one_hardware "$name" "$real_bdf"
        local safe
        safe="$(sanitize_name "$name")"
        local hr="SKIP"
        [[ -f "$RESULTS_DIR/${safe}.hw.result" ]] && hr="$(< "$RESULTS_DIR/${safe}.hw.result")"
        echo "  [${hw_done}/${#compiled[@]}] HW $name: $hr"
      done
      info "Phase 3 done"
      echo ""
    fi
  fi

  # ---- Phase 4: Emulator runs (parallel) ---------------------------------

  info "Phase 4: Running ${#compiled[@]} test(s) on emulator (-j${JOBS})"

  printf '%s\n' "${compiled[@]}" | xargs -P"$JOBS" -I{} bash -c 'run_one_bridge "$@"' _ {}

  info "Phase 4 done"
  echo ""

  # ---- Phase 4b: Trace comparison (optional) ------------------------------

  if [[ -n "$TRACE_MODE" ]]; then
    # Determine which tests to trace.
    local trace_targets=()
    local trace_include_failing=false
    local trace_sweep=false

    case "$TRACE_MODE" in
      default) ;;
      all)       trace_include_failing=true ;;
      sweep)     trace_sweep=true ;;
      sweep-all) trace_sweep=true; trace_include_failing=true ;;
    esac

    for name in "${compiled[@]}"; do
      if ! $trace_include_failing; then
        # Only trace tests that passed the bridge run.
        local safe
        safe="$(sanitize_name "$name")"
        local br="FAIL"
        [[ -f "$RESULTS_DIR/${safe}.bridge.result" ]] && br="$(< "$RESULTS_DIR/${safe}.bridge.result")"
        if [[ "$br" != "PASS" ]]; then
          continue
        fi
      fi
      trace_targets+=("$name")
    done

    local trace_mode_arg="default"
    if $trace_sweep; then
      trace_mode_arg="sweep"
    fi

    if [[ ${#trace_targets[@]} -gt 0 ]]; then
      info "Phase 4b: Trace comparison for ${#trace_targets[@]} test(s) (mode=$trace_mode_arg)"
      # Each trace_one_test internally runs HW serial + EMU parallel via
      # trace-sweep.py. We run tests serially here because each test's HW
      # runs need the NPU exclusively. The EMU runs within each test are
      # already parallelized.
      for name in "${trace_targets[@]}"; do
        trace_one_test "$name" "$trace_mode_arg"
      done
      info "Phase 4b done"
    else
      info "Phase 4b: No tests eligible for tracing"
    fi
    echo ""
  fi

  # ---- Phase 5: Report ---------------------------------------------------

  info "Phase 5: Report"
  print_report tests "$RUN_HW"
}

main
