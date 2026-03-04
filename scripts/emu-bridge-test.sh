#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# emu-bridge-test.sh -- Automated comparison of emulator execution paths.
#
# Runs mlir-aie NPU integration tests through multiple execution paths
# and produces a comparison matrix:
#
#   1. XRT bridge + emulator  (XDNA_EMU=1 ./test.exe)
#   2. npu-test runner + emulator  (cargo run --bin npu-test)
#   3. XRT bridge + real hardware  (./test.exe, no XDNA_EMU)
#
# Usage:
#   ./scripts/emu-bridge-test.sh                    # all tests
#   ./scripts/emu-bridge-test.sh add_one_using_dma  # single test
#   ./scripts/emu-bridge-test.sh --bridge-only       # skip npu-test & hw
#   ./scripts/emu-bridge-test.sh --compile           # force recompile
#   ./scripts/emu-bridge-test.sh --hw                # include hardware run
#   ./scripts/emu-bridge-test.sh --list              # list available tests

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

# Results directory
RESULTS_DIR="/tmp/emu-bridge-results-$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

FILTER=""
BRIDGE_ONLY=false
FORCE_COMPILE=false
RUN_HW=false
LIST_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bridge-only) BRIDGE_ONLY=true; shift ;;
    --compile)     FORCE_COMPILE=true; shift ;;
    --hw)          RUN_HW=true; shift ;;
    --list)        LIST_ONLY=true; shift ;;
    -v|--verbose)  VERBOSE=true; shift ;;
    --help|-h)
      echo "Usage: $0 [options] [test-name-filter]"
      echo ""
      echo "Options:"
      echo "  --bridge-only   Only run XRT bridge path (skip npu-test)"
      echo "  --compile       Force recompile all tests"
      echo "  --hw            Also run on real hardware"
      echo "  --list          List available tests and exit"
      echo "  -v, --verbose   Show test output"
      echo ""
      echo "Filter:"
      echo "  Substring match on test directory name."
      echo "  e.g., 'add_one' matches add_one_using_dma, add_one_two, etc."
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      FILTER="$1"; shift ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "  $*"; }
info() { echo ">>> $*"; }
err()  { echo "ERROR: $*" >&2; }

# Check if a test directory has a standard test.cpp + aie.mlir.
is_standard_test() {
  local dir="$1"
  [[ -f "$dir/test.cpp" ]] && [[ -f "$dir/aie.mlir" || -f "$dir/run.lit" ]]
}

# Check if a test requires Chess compiler.
# We have Chess (aietools) installed, so this always returns false.
requires_chess() {
  return 1
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
    echo "npu1_1col"  # default
  fi
}

# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------

discover_tests() {
  local tests=()
  for dir in "$TEST_SRC"/*/; do
    local name
    name="$(basename "$dir")"

    # Skip non-test directories.
    [[ "$name" == "makefile-common" ]] && continue
    [[ "$name" == "lit.local.cfg" ]] && continue
    [[ "$name" == "core_dmas" ]] && continue  # subdirectory container

    # Apply filter.
    if [[ -n "$FILTER" ]] && [[ "$name" != *"$FILTER"* ]]; then
      continue
    fi

    # Must have test.cpp and either aie.mlir or run.lit.
    if ! is_standard_test "$dir"; then
      continue
    fi

    tests+=("$name")
  done

  # Also check subdirectories (e.g., core_dmas/*)
  for subdir in "$TEST_SRC"/*/*/; do
    [[ -d "$subdir" ]] || continue
    local parent
    parent="$(basename "$(dirname "$subdir")")"
    local child
    child="$(basename "$subdir")"
    local name="${parent}/${child}"

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
# Compilation -- driven by run.lit RUN lines
# ---------------------------------------------------------------------------

# Substitutions for lit RUN lines.
AIETOOLS_DIR="${AIETOOLS_DIR:-/home/triple/npu-work/aietools}"

apply_lit_subs() {
  local src_dir="$1"
  local cmd="$2"
  # Strip the "// RUN: " prefix.
  cmd="${cmd#*RUN: }"
  # Lit substitutions.
  cmd="${cmd//'%S'/$src_dir}"
  cmd="${cmd//'%aietools'/$AIETOOLS_DIR}"
  # %python is used as "python3 script.py" but scripts are on PATH
  # with shebangs, so just strip the python3 prefix entirely.
  cmd="${cmd//'%python '/}"
  cmd="${cmd//'%python'/python3}"
  cmd="${cmd//'%xrt_flags'/-I$XRT_INCLUDE -L$XRT_LIB -luuid -lxrt_coreutil}"
  cmd="${cmd//'%test_utils_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_utils}"
  cmd="${cmd//'%test_lib_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_lib}"
  # Strip run_on_npu wrappers (we handle execution separately).
  cmd="${cmd//'%run_on_npu1%'/}"
  cmd="${cmd//'%run_on_npu2%'/}"
  # Trim leading/trailing whitespace.
  cmd="${cmd#"${cmd%%[![:space:]]*}"}"
  cmd="${cmd%"${cmd##*[![:space:]]}"}"
  # Replace bare 'clang' with system clang++ (avoid aietools wrapper).
  if [[ "$cmd" == clang\ * ]]; then
    cmd="/usr/bin/clang++ ${cmd#clang }"
  elif [[ "$cmd" == g++\ * ]]; then
    cmd="/usr/bin/g++ ${cmd#g++ }"
  fi
  echo "$cmd"
}

# Parse run.lit and extract build commands (everything before the
# run_on_npu execution line).  Returns commands one per line.
extract_build_commands() {
  local lit_file="$1"
  local src_dir="$2"
  [[ -f "$lit_file" ]] || return 1

  while IFS= read -r line; do
    # Only process RUN lines.
    [[ "$line" == *"RUN:"* ]] || continue
    local cmd
    cmd="$(apply_lit_subs "$src_dir" "$line")"
    # Skip execution lines (the actual test run).
    [[ "$cmd" == *"./test.exe"* ]] && continue
    [[ "$cmd" == *"run_on_npu"* ]] && continue
    # Skip sed for NPUDEVICE -- we handle that separately.
    [[ "$cmd" == *"NPUDEVICE"* ]] && continue
    # Skip cp of aie.mlir -- we handle that.
    [[ "$cmd" == "cp "* ]] && continue
    echo "$cmd"
  done < "$lit_file"
}

compile_test() {
  local name="$1"
  local src_dir="$TEST_SRC/$name"
  local build_dir="$BUILD_BASE/$name"

  mkdir -p "$build_dir"

  # Check if already compiled (xclbin + test.exe both present).
  if [[ -f "$build_dir/aie.xclbin" ]] && [[ -f "$build_dir/test.exe" ]] && ! $FORCE_COMPILE; then
    return 0
  fi

  info "Compiling: $name"

  local log_file="$RESULTS_DIR/${name//\//_}_compile.log"

  # Prepare architecture MLIR (NPUDEVICE substitution).
  local npu_dev
  npu_dev="$(get_npu_device "$src_dir")"
  if [[ -f "$src_dir/aie.mlir" ]]; then
    cp "$src_dir/aie.mlir" "$build_dir/aie_arch.mlir"
    sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch.mlir"
  fi

  # Extract and execute all build commands from run.lit.
  local lit_file="$src_dir/run.lit"
  if [[ ! -f "$lit_file" ]]; then
    err "No run.lit found for $name"
    return 1
  fi

  # Patch test.cpp: read XRT_DEVICE_BDF env var for device selection.
  # This lets us compile once and run against either the real NPU or
  # the emulated device by setting the BDF at runtime.
  if [[ -f "$src_dir/test.cpp" ]]; then
    sed \
      -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
      -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
      "$src_dir/test.cpp" > "$build_dir/test.cpp"
  fi

  local failed=false
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    # Replace references to ./aie_arch.mlir or aie.mlir in aiecc.py
    # commands to use our prepared copy.
    if [[ "$cmd" == *aiecc.py* ]]; then
      cmd="${cmd//$src_dir\/aie.mlir/./aie_arch.mlir}"
      cmd="${cmd//\.\/aie.mlir/./aie_arch.mlir}"
    fi
    # Point clang at our patched test.cpp in the build dir.
    cmd="${cmd//$src_dir\/test.cpp/./test.cpp}"
    if ! ( cd "$build_dir" && nice -n 19 bash -c "$cmd" ) >> "$log_file" 2>&1; then
      err "Build step failed for $name: $cmd"
      failed=true
      break
    fi
  done < <(extract_build_commands "$lit_file" "$src_dir")

  if $failed; then
    return 1
  fi

  # If run.lit didn't produce test.exe (some tests have non-standard
  # build), try a fallback compilation using the patched copy.
  if [[ ! -f "$build_dir/test.exe" ]] && [[ -f "$build_dir/test.cpp" ]]; then
    /usr/bin/clang++ "$build_dir/test.cpp" -o "$build_dir/test.exe" \
      -std=c++17 -Wall \
      -I"$XRT_INCLUDE" -L"$XRT_LIB" \
      -I"$TEST_UTILS_INCLUDE" -L"$TEST_UTILS_LIB" \
      -luuid -lxrt_coreutil -ltest_utils -lrt -lstdc++ \
      >> "$log_file" 2>&1 || true
  fi

  # Check that we got the expected artifacts.
  if [[ ! -f "$build_dir/aie.xclbin" ]]; then
    # Some tests produce differently-named xclbins.
    local any_xclbin
    any_xclbin=$(find "$build_dir" -name "*.xclbin" -print -quit 2>/dev/null)
    if [[ -z "$any_xclbin" ]]; then
      err "No xclbin produced for $name"
      return 1
    fi
  fi

  return 0
}

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# Extract the test execution command from run.lit to get correct args.
get_run_cmd() {
  local src_dir="$1"
  local lit_file="$src_dir/run.lit"
  [[ -f "$lit_file" ]] || return 1

  # Find the first npu1 execution line (prefer npu1, fall back to any).
  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    if [[ "$line" == *"./test.exe"* ]] && [[ "$line" == *"npu1"* || "$line" != *"npu2"* ]]; then
      local cmd
      cmd="$(apply_lit_subs "$src_dir" "$line")"
      # Trim leading whitespace from stripping run_on_npu wrappers.
      cmd="${cmd#"${cmd%%[![:space:]]*}"}"
      echo "$cmd"
      return 0
    fi
  done < "$lit_file"

  # Fallback: standard invocation.
  echo "./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin"
}

# Run test.exe through XRT bridge (XDNA_EMU=1).
run_bridge() {
  local name="$1"
  local build_dir="$BUILD_BASE/$name"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${name//\//_}_bridge.log"

  if [[ ! -f "$build_dir/test.exe" ]]; then
    echo "SKIP_NOEXE"
    return
  fi

  # Find any xclbin in the build dir.
  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP_NOXCLBIN"
    return
  fi

  # Extract the run command from run.lit for correct args.
  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  (
    cd "$build_dir"
    export XDNA_EMU=1
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    timeout 60 bash -c "$run_cmd"
  ) > "$log_file" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    echo "PASS"
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT"
  else
    echo "FAIL"
  fi
}

# Run through npu-test runner (emulator direct API).
run_nputest() {
  local name="$1"
  local log_file="$RESULTS_DIR/${name//\//_}_nputest.log"

  (
    cd "$EMU_ROOT"
    timeout 120 cargo run --release --bin npu-test -- "$name" --no-build 2>&1
  ) > "$log_file" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]] && grep -qE 'Passed: *[1-9]' "$log_file"; then
    echo "PASS"
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT"
  elif grep -qE 'Skipped: *[1-9]' "$log_file"; then
    echo "SKIP"
  elif grep -qE 'Unknown: *[1-9]' "$log_file"; then
    echo "UNKNOWN"
  else
    echo "FAIL"
  fi
}

# Real NPU BDF -- detected once at startup.
REAL_NPU_BDF="$(xrt-smi examine 2>/dev/null | grep -oP '\[0000:[0-9a-f:\.]+\]' | head -1 | tr -d '[]')"

# Run test.exe on real hardware via BDF selection.
run_hardware() {
  local name="$1"
  local build_dir="$BUILD_BASE/$name"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${name//\//_}_hw.log"

  if [[ ! -f "$build_dir/test.exe" ]]; then
    echo "SKIP_NOEXE"
    return
  fi

  if [[ -z "$REAL_NPU_BDF" ]]; then
    echo "SKIP_NOHW"
    return
  fi

  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$REAL_NPU_BDF"
    timeout 30 bash -c "$run_cmd"
  ) > "$log_file" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    echo "PASS"
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT"
  else
    echo "FAIL"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
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

  info "Found ${#tests[@]} test(s) to run"
  info "Results in: $RESULTS_DIR"
  echo ""

  # Header
  if $BRIDGE_ONLY; then
    printf "%-45s  %-10s  %-10s\n" "TEST" "BRIDGE" "NOTES"
    printf "%-45s  %-10s  %-10s\n" "----" "------" "-----"
  elif $RUN_HW; then
    printf "%-45s  %-10s  %-10s  %-10s  %-10s\n" "TEST" "BRIDGE" "NPU-TEST" "HARDWARE" "MATCH"
    printf "%-45s  %-10s  %-10s  %-10s  %-10s\n" "----" "------" "--------" "--------" "-----"
  else
    printf "%-45s  %-10s  %-10s  %-10s\n" "TEST" "BRIDGE" "NPU-TEST" "MATCH"
    printf "%-45s  %-10s  %-10s  %-10s\n" "----" "------" "--------" "-----"
  fi

  local pass_bridge=0 fail_bridge=0 skip_bridge=0
  local pass_nputest=0 fail_nputest=0 skip_nputest=0
  local pass_hw=0 fail_hw=0 skip_hw=0
  local match_count=0 mismatch_count=0

  for name in "${tests[@]}"; do
    local notes=""
    local bridge_result="SKIP"
    local nputest_result="SKIP"
    local hw_result="SKIP"

    # Skip Chess-only and NPU2-only tests.
    if requires_chess "$TEST_SRC/$name"; then
      notes="chess"
      printf "%-45s  %-10s" "$name" "SKIP_CHESS"
      if ! $BRIDGE_ONLY; then
        printf "  %-10s" "SKIP_CHESS"
      fi
      if $RUN_HW; then
        printf "  %-10s" "SKIP_CHESS"
      fi
      printf "  %-10s\n" "$notes"
      ((skip_bridge++)) || true
      ((skip_nputest++)) || true
      continue
    fi

    if requires_npu2 "$TEST_SRC/$name"; then
      notes="npu2"
      printf "%-45s  %-10s" "$name" "SKIP_NPU2"
      if ! $BRIDGE_ONLY; then
        printf "  %-10s" "SKIP_NPU2"
      fi
      if $RUN_HW; then
        printf "  %-10s" "SKIP_NPU2"
      fi
      printf "  %-10s\n" "$notes"
      ((skip_bridge++)) || true
      ((skip_nputest++)) || true
      continue
    fi

    # Compile everything (kernel objects, xclbin, host test.exe).
    if ! compile_test "$name" 2>/dev/null; then
      notes="compile_fail"
      printf "%-45s  %-10s" "$name" "COMP_FAIL"
      if ! $BRIDGE_ONLY; then
        printf "  %-10s" "COMP_FAIL"
      fi
      if $RUN_HW; then
        printf "  %-10s" "COMP_FAIL"
      fi
      printf "  %-10s\n" "$notes"
      ((skip_bridge++)) || true
      ((skip_nputest++)) || true
      continue
    fi

    # Run XRT bridge path.
    bridge_result=$(run_bridge "$name")

    case "$bridge_result" in
      PASS) ((pass_bridge++)) || true ;;
      SKIP*) ((skip_bridge++)) || true ;;
      *) ((fail_bridge++)) || true ;;
    esac

    # Run npu-test runner path.
    if ! $BRIDGE_ONLY; then
      nputest_result=$(run_nputest "$name")
      case "$nputest_result" in
        PASS) ((pass_nputest++)) || true ;;
        SKIP*) ((skip_nputest++)) || true ;;
        *) ((fail_nputest++)) || true ;;
      esac
    fi

    # Run on hardware.
    if $RUN_HW; then
      hw_result=$(run_hardware "$name")
      case "$hw_result" in
        PASS) ((pass_hw++)) || true ;;
        SKIP*) ((skip_hw++)) || true ;;
        *) ((fail_hw++)) || true ;;
      esac
    fi

    # Determine match.
    local match=""
    if ! $BRIDGE_ONLY; then
      if [[ "$bridge_result" == "$nputest_result" ]]; then
        match="yes"
        ((match_count++)) || true
      elif [[ "$bridge_result" == "SKIP"* ]] || [[ "$nputest_result" == "SKIP"* ]]; then
        match="-"
      else
        match="NO"
        ((mismatch_count++)) || true
      fi
    fi

    # Print row.
    printf "%-45s  %-10s" "$name" "$bridge_result"
    if ! $BRIDGE_ONLY; then
      printf "  %-10s" "$nputest_result"
    fi
    if $RUN_HW; then
      printf "  %-10s" "$hw_result"
    fi
    if [[ -n "$match" ]]; then
      printf "  %-10s" "$match"
    fi
    if [[ -n "$notes" ]]; then
      printf "  %s" "$notes"
    fi
    printf "\n"

    # Show verbose output on failure.
    if $VERBOSE && [[ "$bridge_result" != "PASS" ]] && [[ "$bridge_result" != "SKIP"* ]]; then
      local logf="$RESULTS_DIR/${name//\//_}_bridge.log"
      [[ -f "$logf" ]] && tail -5 "$logf" | sed 's/^/    /'
    fi
  done

  # Summary
  echo ""
  echo "=== Summary ==="
  echo "Bridge:   $pass_bridge pass, $fail_bridge fail, $skip_bridge skip"
  if ! $BRIDGE_ONLY; then
    echo "npu-test: $pass_nputest pass, $fail_nputest fail, $skip_nputest skip"
  fi
  if $RUN_HW; then
    echo "Hardware: $pass_hw pass, $fail_hw fail, $skip_hw skip"
  fi
  if ! $BRIDGE_ONLY; then
    echo "Match:    $match_count match, $mismatch_count mismatch"
  fi
  echo "Logs:     $RESULTS_DIR/"
}

main
