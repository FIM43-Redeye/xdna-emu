#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# emu-bridge-test.sh -- Dual-compiler phased comparison of emulator vs hardware.
#
# Compiles and runs mlir-aie NPU integration tests with BOTH compilers
# (Chess and Peano) through two execution paths, producing a per-compiler
# comparison matrix:
#
#   1. XRT bridge + emulator  (XDNA_EMU=1 ./test.exe)
#   2. XRT bridge + real hardware  (./test.exe with real BDF)
#
# Chess is the ground truth compiler. Peano results are informational --
# compile failures with Peano are expected for some Chess-specific tests.
#
# Six-phase pipelined architecture:
#   Phase 1: Discover        -- find tests, filter, skip npu2-only
#   Phase 2: Compile         -- parallel xclbin builds (both compilers in parallel) + test.exe
#   Phase 3+4: Run HW+EMU   -- HW (-j5) and EMU (-j$JOBS) run concurrently
#   Phase 5: Trace compare   -- automatic trace comparison (HW vs EMU)
#   Phase 6: Report          -- per-compiler comparison matrix + summary
#
# Usage:
#   ./scripts/emu-bridge-test.sh                    # all tests, both compilers
#   ./scripts/emu-bridge-test.sh add_one_using_dma  # single test, both compilers
#   ./scripts/emu-bridge-test.sh --chess-only       # Chess only (ground truth)
#   ./scripts/emu-bridge-test.sh --peano-only       # Peano only
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

# Compiler paths (auto-detect from environment)
PEANO_CLANG="${PEANO_INSTALL_DIR:-${MLIR_AIE}/../llvm-aie/install}/bin/clang++"
PEANO_INCLUDE="${MLIR_AIE_INSTALL_DIR:-${MLIR_AIE}/install}/include"
CHESS_INCLUDE="${AIETOOLS_DIR}/include"

# Peano kernel compilation flags (from mlir-aie makefile-common)
PEANO_KERNEL_FLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG"
PEANO_KERNEL_FLAGS+=" -Wno-parentheses -Wno-attributes -Wno-macro-redefined"
PEANO_KERNEL_FLAGS+=" -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw"

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
RUN_EMU=true
LIST_ONLY=false
VERBOSE=false
COMPILER_MODE="both"  # "both", "chess", "peano"
NO_TRACE="${NO_TRACE:-false}"
SWEEP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)     FORCE_COMPILE=true; shift ;;
    --no-hw)       RUN_HW=false; shift ;;
    --no-emu)      RUN_EMU=false; shift ;;
    --list)        LIST_ONLY=true; shift ;;
    -v|--verbose)  VERBOSE=true; shift ;;
    -j*)
      JOBS="${1#-j}"
      if [[ -z "$JOBS" ]] || ! [[ "$JOBS" =~ ^[0-9]+$ ]]; then
        echo "Invalid -j value: $1" >&2; exit 1
      fi
      shift ;;
    --chess-only|--chess)  COMPILER_MODE="chess"; shift ;;
    --peano-only|--peano)  COMPILER_MODE="peano"; shift ;;
    --no-trace)            NO_TRACE=true; shift ;;
    --sweep)               SWEEP=true; shift ;;
    --serial-hw)           NPU_HW_JOBS=1; shift ;;
    --help|-h)
      cat <<'USAGE'
Usage: emu-bridge-test.sh [options] [test-name-filter]

Options:
  --compile       Force recompile all xclbins (default: use cached)
  --no-hw         Skip real hardware runs (default: hardware enabled)
  --no-emu        Skip emulator runs (default: emulator enabled)
  --no-trace      Disable always-on trace preparation (default: traces enabled)
  --sweep         Run full event sweep (trace-sweep.py) on passing tests after runs
  --list          List available tests and exit
  -jN             Override parallelism (default: nproc)
  -v, --verbose   Show log snippets on failure
  --chess-only    Only compile/run with Chess compiler (ground truth)
  --peano-only    Only compile/run with Peano compiler
  --serial-hw     Run hardware tests sequentially (for crash classification)
  (default: both compilers, Chess is ground truth, HW parallel -j5)

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

# Build compiler list from mode
case "$COMPILER_MODE" in
  chess) COMPILERS=("chess") ;;
  peano) COMPILERS=("peano") ;;
  both)  COMPILERS=("chess" "peano") ;;
esac

# Validate compiler availability
for _c in "${COMPILERS[@]}"; do
  if [[ "$_c" == "chess" ]] && ! command -v xchesscc_wrapper &>/dev/null; then
    echo "Warning: xchesscc_wrapper not found, Chess builds will fail" >&2
  fi
  if [[ "$_c" == "peano" ]] && [[ ! -x "$PEANO_CLANG" ]]; then
    echo "Warning: Peano clang not found at $PEANO_CLANG" >&2
  fi
done

# Export variables that parallel jobs need.
export RESULTS_DIR FORCE_COMPILE VERBOSE RUN_EMU NO_TRACE SWEEP
export MLIR_AIE TEST_SRC BUILD_BASE EMU_ROOT SCRIPT_DIR TRACE_QUARANTINE_FILE
export XRT_DIR XRT_INCLUDE XRT_LIB
export TEST_LIB_DIR TEST_UTILS_INCLUDE TEST_UTILS_LIB
export AIETOOLS_DIR
export COMPILER_MODE PEANO_CLANG PEANO_INCLUDE CHESS_INCLUDE PEANO_KERNEL_FLAGS
export COMPILERS_STR="${COMPILERS[*]}"

# ---------------------------------------------------------------------------
# Hardware quarantine (tests that cause TDRs, run last + isolated)
# ---------------------------------------------------------------------------

QUARANTINE_FILE="${SCRIPT_DIR}/hw-quarantine.txt"
declare -A QUARANTINE=()   # "name:compiler" -> 1
if [[ -f "$QUARANTINE_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"        # strip comments
    line="${line// /}"        # strip spaces
    [[ -z "$line" ]] && continue
    local_name="${line%%:*}"
    local_comp="${line##*:}"
    if [[ "$local_comp" == "*" ]]; then
      QUARANTINE["${local_name}:chess"]=1
      QUARANTINE["${local_name}:peano"]=1
    else
      QUARANTINE["${line}"]=1
    fi
  done < "$QUARANTINE_FILE"
  if [[ ${#QUARANTINE[@]} -gt 0 ]]; then
    echo ">>> Quarantine: ${#QUARANTINE[@]} test:compiler pair(s) will run last, isolated"
  fi
fi
export QUARANTINE_FILE

# Check if a job is quarantined.
is_quarantined() {
  [[ -n "${QUARANTINE[${1}]+x}" ]]
}

# ---------------------------------------------------------------------------
# Trace quarantine (tests that are dangerous when trace-injected)
# ---------------------------------------------------------------------------

TRACE_QUARANTINE_FILE="${SCRIPT_DIR}/trace-quarantine.txt"
declare -A TRACE_QUARANTINE=()   # "test_name" -> 1
if [[ -f "$TRACE_QUARANTINE_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"        # strip comments
    line="${line// /}"        # strip spaces
    [[ -z "$line" ]] && continue
    TRACE_QUARANTINE["$line"]=1
  done < "$TRACE_QUARANTINE_FILE"
  if [[ ${#TRACE_QUARANTINE[@]} -gt 0 ]]; then
    echo ">>> Trace quarantine: ${#TRACE_QUARANTINE[@]} test(s) excluded from tracing"
  fi
fi

is_trace_quarantined() {
  # Fast path: associative array (main process only -- not exported).
  if [[ ${#TRACE_QUARANTINE[@]} -gt 0 ]] 2>/dev/null; then
    [[ -n "${TRACE_QUARANTINE[${1}]+x}" ]]
    return
  fi
  # Slow path: file read (subshells via xargs).
  local name="$1"
  [[ -f "$TRACE_QUARANTINE_FILE" ]] || return 1
  while IFS= read -r line; do
    local entry="${line%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    [[ "$entry" == "$name" ]] && return 0
  done < "$TRACE_QUARANTINE_FILE"
  return 1
}
export -f is_trace_quarantined

# ---------------------------------------------------------------------------
# Test quarantine (fundamentally broken tests -- skip entirely)
# ---------------------------------------------------------------------------

is_test_quarantined() {
  local name="${1%%:*}"  # Strip compiler suffix if present.
  local quarantine_file="$SCRIPT_DIR/test-quarantine.txt"
  [[ -f "$quarantine_file" ]] || return 1
  while IFS= read -r line; do
    local entry="${line%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    [[ "$entry" == "$name" ]] && return 0
  done < "$quarantine_file"
  return 1
}
export -f is_test_quarantined

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

info() { echo ">>> $*"; }
err()  { echo "ERROR: $*" >&2; }

# Count TDR events in dmesg.  Returns the number of "aie2_tdr_work" lines.
# Usage: before=$(tdr_count); run test; after=$(tdr_count); new=$((after-before))
tdr_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'aie2_tdr_work') || true
  echo "$n"
}
export -f tdr_count

# Count IOMMU page faults in dmesg.
iommu_fault_count() {
  local n
  n=$(dmesg 2>/dev/null | grep -c 'IO_PAGE_FAULT') || true
  echo "$n"
}
export -f iommu_fault_count

# Current system uptime in seconds (integer), for dmesg correlation.
uptime_sec() {
  awk '{printf "%.0f", $1}' /proc/uptime
}
export -f uptime_sec

# Check NPU health: can it create contexts?  Returns 0 if healthy.
npu_health_check() {
  xrt-smi examine -r aie-partitions &>/dev/null
}
export -f npu_health_check

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
  if [[ ! -f "$lit" ]]; then
    echo "npu1_1col"
    return
  fi
  # Check for specific device variants in the run.lit sed commands.
  # The pattern is: sed 's/NPUDEVICE/<device>/g'
  local device
  device="$(grep -oP "NPUDEVICE/\K[a-z0-9_]+" "$lit" | head -1)" || true
  if [[ -n "$device" ]]; then
    echo "$device"
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

# Wait for NPU to have zero active hardware contexts.
# Polls xrt-smi at 100ms intervals, times out after 10s, falls back to 0.5s sleep.
wait_npu_idle() {
  local deadline=$((SECONDS + 10))
  while [[ $SECONDS -lt $deadline ]]; do
    if xrt-smi examine -r aie-partitions 2>/dev/null \
        | grep -q 'No hardware contexts running'; then
      return 0
    fi
    sleep 0.1
  done
  # Fallback: xrt-smi unavailable or contexts stuck
  sleep 0.5
}

# Transform a build command for Chess compilation.
# - xchesscc_wrapper commands: pass through unchanged
# - aiecc.py commands: ensure --xchesscc and --xbridge are present
# - Other commands: pass through unchanged
transform_for_chess() {
  local cmd="$1"

  if [[ "$cmd" == *xchesscc_wrapper* ]]; then
    echo "$cmd"
    return
  fi

  if [[ "$cmd" == *aiecc.py* ]]; then
    # Remove any --no-xchesscc or --no-xbridge
    cmd="${cmd//--no-xchesscc/}"
    cmd="${cmd//--no-xbridge/}"
    # Add --xchesscc if not present
    if [[ "$cmd" != *"--xchesscc"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --xchesscc}"
    fi
    # Add --xbridge if not present
    if [[ "$cmd" != *"--xbridge"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --xbridge}"
    fi
    echo "$cmd"
    return
  fi

  echo "$cmd"
}

# Transform a build command for Peano compilation.
# - xchesscc_wrapper commands: replace with Peano clang++ equivalent
# - aiecc.py commands: strip --xchesscc and --xbridge, add --no-xchesscc
# - Other commands: pass through unchanged
transform_for_peano() {
  local cmd="$1"
  local src_dir="$2"

  if [[ "$cmd" == *xchesscc_wrapper* ]]; then
    # Extract -c source and -o output arguments.
    # Pattern: xchesscc_wrapper aie2 [flags...] -c source.cc -o output.o
    local source output
    # Use grep -oP for argument extraction
    source="$(echo "$cmd" | grep -oP '(?<=-c\s)\S+' || true)"
    output="$(echo "$cmd" | grep -oP '(?<=-o\s)\S+' || true)"

    if [[ -n "$source" ]] && [[ -n "$output" ]]; then
      # Resolve source path relative to src_dir if not absolute
      local resolved_source="$source"
      if [[ "$source" != /* ]]; then
        resolved_source="$src_dir/$source"
      fi
      # Extract any -D defines and extra -I paths from original command
      local extra_flags=""
      while read -r flag; do
        [[ -n "$flag" ]] && extra_flags+=" $flag"
      done < <(echo "$cmd" | grep -oP '\-D\S+' || true)
      while read -r flag; do
        [[ -n "$flag" ]] && extra_flags+=" $flag"
      done < <(echo "$cmd" | grep -oP '\-I\s*\S+' || true)
      # Use Peano clang with correct include path
      echo "$PEANO_CLANG $PEANO_KERNEL_FLAGS -I$PEANO_INCLUDE${extra_flags} -c $resolved_source -o $output"
    else
      echo "# SKIP (unparseable xchesscc): $cmd"
    fi
    return
  fi

  if [[ "$cmd" == *aiecc.py* ]]; then
    # Strip Chess flags
    cmd="${cmd//--xchesscc/}"
    cmd="${cmd//--xbridge/}"
    # Ensure --no-xchesscc is present
    if [[ "$cmd" != *"--no-xchesscc"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --no-xchesscc}"
    fi
    echo "$cmd"
    return
  fi

  echo "$cmd"
}

# Export all helpers for xargs subshells.
export -f is_standard_test requires_npu2 get_npu_device apply_lit_subs
export -f extract_build_commands get_run_cmd sanitize_name wait_npu_idle
export -f transform_for_chess transform_for_peano

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

# Compile a test's xclbin with a specific compiler.
# Args: $1=test_name  $2=compiler ("chess"|"peano")
compile_one_compiler() {
  local name="$1"
  local compiler="$2"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local build_dir="$BUILD_BASE/$name/$compiler"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.compile.log"
  local result_file="$RESULTS_DIR/${safe}.${compiler}.compile.result"
  local lit_file="$src_dir/run.lit"

  mkdir -p "$build_dir"
  : > "$log_file"

  if [[ ! -f "$lit_file" ]]; then
    echo "FAIL" > "$result_file"
    echo "No run.lit found" >> "$log_file"
    echo "  COMPILE $name ($compiler): FAIL (no run.lit)"
    return 0
  fi

  # Check cache
  local have_xclbin=false
  if [[ -f "$build_dir/aie.xclbin" ]] || ls "$build_dir"/*.xclbin &>/dev/null; then
    have_xclbin=true
  fi

  if $have_xclbin && [[ "$FORCE_COMPILE" != "true" ]]; then
    echo "  COMPILE $name ($compiler): cached"
    echo "OK" > "$result_file"
    return 0
  fi

  # Prepare architecture MLIR (NPUDEVICE substitution)
  local npu_dev
  npu_dev="$(get_npu_device "$src_dir")"
  if [[ -f "$src_dir/aie.mlir" ]]; then
    cp "$src_dir/aie.mlir" "$build_dir/aie_arch.mlir"
    sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch.mlir"
  fi

  # Use traced MLIR if trace preparation succeeded.
  if [[ "$TRACE_OK" == "true" ]] && [[ -f "$TRACED_DIR/aie_traced.mlir" ]]; then
    cp "$TRACED_DIR/aie_traced.mlir" "$build_dir/aie_arch.mlir"
  fi

  local failed=false
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    # Skip host compilation -- handled separately
    [[ "$cmd" == *clang*test.cpp* ]] && continue
    [[ "$cmd" == *g++*test.cpp* ]] && continue

    # Fix MLIR path references
    if [[ "$cmd" == *aiecc.py* ]]; then
      cmd="${cmd//$src_dir\/aie.mlir/./aie_arch.mlir}"
      cmd="${cmd//\.\/aie.mlir/./aie_arch.mlir}"
    fi

    # Transform command for this compiler
    if [[ "$compiler" == "chess" ]]; then
      cmd="$(transform_for_chess "$cmd")"
    else
      cmd="$(transform_for_peano "$cmd" "$src_dir")"
    fi

    # Skip commands that couldn't be transformed
    [[ "$cmd" == "# SKIP"* ]] && { echo "$cmd" >> "$log_file"; continue; }

    if ! ( cd "$build_dir" && nice -n 19 bash -c "$cmd" ) >> "$log_file" 2>&1; then
      failed=true
      break
    fi
  done < <(extract_build_commands "$lit_file" "$src_dir")

  if $failed; then
    echo "FAIL" > "$result_file"
    echo "  COMPILE $name ($compiler): FAIL"
    return 0
  fi

  # Verify xclbin was produced
  if [[ ! -f "$build_dir/aie.xclbin" ]]; then
    local any_xclbin
    any_xclbin=$(find "$build_dir" -name "*.xclbin" -print -quit 2>/dev/null || true)
    if [[ -z "$any_xclbin" ]]; then
      echo "FAIL" > "$result_file"
      echo "  COMPILE $name ($compiler): FAIL (no xclbin produced)"
      return 0
    fi
  fi

  echo "OK" > "$result_file"
  echo "  COMPILE $name ($compiler): OK"
}

# Compile one test for all active compilers + build shared test.exe.
# Called via xargs. Writes per-compiler compile results and shared test.exe.
compile_one() {
  local name="$1"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local build_dir="$BUILD_BASE/$name"
  local lit_file="$src_dir/run.lit"

  # Check test quarantine (fundamentally broken tests -- skip entirely).
  if is_test_quarantined "$name"; then
    local compilers
    read -ra compilers <<< "$COMPILERS_STR"
    for compiler in "${compilers[@]}"; do
      echo "SKIP_QUARANTINED" > "$RESULTS_DIR/${safe}.${compiler}.compile.result"
    done
    echo "  COMPILE $name: SKIP (test-quarantined)"
    return 0
  fi

  # ---- Trace preparation (always-on unless --no-trace) ----
  local traced_dir="$build_dir/traced"
  local trace_ok=false

  if [[ "$NO_TRACE" != "true" ]] && ! is_trace_quarantined "$name"; then
    local trace_log="$RESULTS_DIR/${safe}.trace-prepare.log"
    if nice -n 19 python3 "$EMU_ROOT/tools/trace-prepare.py" "$src_dir" \
        -o "$traced_dir" > "$trace_log" 2>&1; then
      if [[ -f "$traced_dir/prepare-status.txt" ]] \
          && grep -q '^OK' "$traced_dir/prepare-status.txt"; then
        trace_ok=true
        echo "  TRACE PREP $name: OK"
      else
        echo "  TRACE PREP $name: FAIL (status)"
      fi
    else
      echo "  TRACE PREP $name: FAIL (exit code)"
    fi
  fi
  export TRACE_OK="$trace_ok"
  export TRACED_DIR="$traced_dir"

  # Reconstruct COMPILERS array from serialized string
  local compilers
  read -ra compilers <<< "$COMPILERS_STR"

  # Compile xclbin for each compiler IN PARALLEL (independent build dirs)
  local pids=()
  for compiler in "${compilers[@]}"; do
    compile_one_compiler "$name" "$compiler" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done

  # Build shared test.exe (compiler-agnostic, only needs XRT)
  local log_file="$RESULTS_DIR/${safe}.testexe.log"
  mkdir -p "$build_dir"
  : > "$log_file"

  if [[ -f "$src_dir/test.cpp" ]]; then
    if [[ "$TRACE_OK" == "true" ]] && [[ -f "$traced_dir/test_traced.cpp" ]]; then
      # Use tree-sitter-patched version (includes BDF + trace transforms).
      cp "$traced_dir/test_traced.cpp" "$build_dir/test.cpp"
    else
      # No tracing -- just apply BDF patch.
      sed \
        -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
        -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
        "$src_dir/test.cpp" > "$build_dir/test.cpp"
    fi
  fi

  # Find the clang/g++ line from run.lit for correct flags
  local clang_cmd=""
  if [[ -f "$lit_file" ]]; then
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
  fi

  if [[ -n "$clang_cmd" ]]; then
    if ! ( cd "$build_dir" && bash -c "$clang_cmd" ) >> "$log_file" 2>&1; then
      echo "  COMPILE $name: FAIL (test.exe)"
      return 0
    fi
  elif [[ -f "$build_dir/test.cpp" ]]; then
    if ! /usr/bin/clang++ "$build_dir/test.cpp" -o "$build_dir/test.exe" \
        -std=c++17 -Wall \
        -I"$XRT_INCLUDE" -L"$XRT_LIB" \
        -I"$TEST_UTILS_INCLUDE" -L"$TEST_UTILS_LIB" \
        -luuid -lxrt_coreutil -ltest_utils -lrt -lstdc++ \
        >> "$log_file" 2>&1; then
      echo "  COMPILE $name: FAIL (test.exe fallback)"
      return 0
    fi
  fi
}
export -f compile_one_compiler compile_one

# ---------------------------------------------------------------------------
# Phase 3+4: Hardware + Emulator runs (concurrent)
# ---------------------------------------------------------------------------

# Maximum concurrent NPU hardware contexts.  NPU1 supports 6, but we
# use 5 to leave headroom for the driver and avoid context-exhaustion
# crashes observed at 10+ concurrent jobs.
NPU_HW_JOBS="${NPU_HW_JOBS:-5}"
export NPU_HW_JOBS

run_one_hardware() {
  local name="$1"
  local bdf="$2"
  local compiler="$3"
  local safe
  safe="$(sanitize_name "$name")"
  local build_dir="$BUILD_BASE/$name/$compiler"
  local test_exe="$BUILD_BASE/$name/test.exe"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.hw.log"
  local result_file="$RESULTS_DIR/${safe}.${compiler}.hw.result"

  # Symlink shared test.exe into per-compiler build dir
  if [[ -f "$test_exe" ]] && [[ ! -f "$build_dir/test.exe" ]]; then
    ln -sf "$test_exe" "$build_dir/test.exe"
  fi

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

  local trace_out_dir="$RESULTS_DIR/${safe}.${compiler}.hw"
  mkdir -p "$trace_out_dir"

  # Snapshot TDR count and uptime for post-test attribution.
  local tdr_before
  tdr_before="$(tdr_count)"
  local t_start
  t_start="$(uptime_sec)"

  local rc=0
  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$bdf"
    export XDNA_TRACE_DIR="$trace_out_dir"
    timeout 30 bash -c "$run_cmd"
  ) > "$log_file" 2>&1 || rc=$?

  local tdr_after
  tdr_after="$(tdr_count)"
  local tdr_new=$(( tdr_after - tdr_before ))

  if [[ $tdr_new -gt 0 ]]; then
    echo "TDR" > "$result_file"
    echo "TDR detected: $tdr_new new aie2_tdr_work events (uptime ${t_start}s)" >> "$log_file"
  elif [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    echo "PASS" > "$result_file"
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT" > "$result_file"
  else
    echo "FAIL" > "$result_file"
  fi

  # Copy events.json for trace decoding.
  local build_traced="$BUILD_BASE/$name/traced"
  if [[ -f "$build_traced/events.json" ]]; then
    cp "$build_traced/events.json" "$trace_out_dir/"
  fi

  # Trim trace buffer to actual data length.
  if [[ -f "$trace_out_dir/trace_raw.bin" ]]; then
    python3 "$EMU_ROOT/tools/trace-trim.py" "$trace_out_dir/trace_raw.bin" 2>/dev/null || true
  fi
}
export -f run_one_hardware

# ---------------------------------------------------------------------------
# Emulator bridge runner (used by Phase 3+4)
# ---------------------------------------------------------------------------

run_one_bridge() {
  local name="$1"
  local compiler="${2:-}"
  local safe
  safe="$(sanitize_name "$name")"

  # If no compiler specified, deserialize from COMPILERS_STR and run all.
  if [[ -z "$compiler" ]]; then
    local compilers
    read -ra compilers <<< "$COMPILERS_STR"
    for c in "${compilers[@]}"; do
      run_one_bridge "$name" "$c"
    done
    return
  fi

  local build_dir="$BUILD_BASE/$name/$compiler"
  local test_exe="$BUILD_BASE/$name/test.exe"
  local src_dir="$TEST_SRC/$name"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.bridge.log"
  local result_file="$RESULTS_DIR/${safe}.${compiler}.bridge.result"

  # Symlink shared test.exe into per-compiler build dir.
  if [[ -f "$test_exe" ]] && [[ ! -f "$build_dir/test.exe" ]]; then
    ln -sf "$test_exe" "$build_dir/test.exe"
  fi

  if [[ ! -f "$build_dir/test.exe" ]]; then
    echo "SKIP" > "$result_file"
    echo "  BRIDGE $name ($compiler): SKIP (no test.exe)"
    return
  fi

  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP" > "$result_file"
    echo "  BRIDGE $name ($compiler): SKIP (no xclbin)"
    return
  fi

  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  local trace_out_dir="$RESULTS_DIR/${safe}.${compiler}.emu"
  mkdir -p "$trace_out_dir"

  local rc=0
  (
    cd "$build_dir"
    export XDNA_EMU=1
    export XDNA_EMU_LOG_LEVEL="${XDNA_EMU_LOG_LEVEL:-info}"
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    export XDNA_TRACE_DIR="$trace_out_dir"
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

  # Copy events.json for trace decoding.
  local build_traced="$BUILD_BASE/$name/traced"
  if [[ -f "$build_traced/events.json" ]]; then
    cp "$build_traced/events.json" "$trace_out_dir/"
  fi

  # Trim trace buffer to actual data length.
  if [[ -f "$trace_out_dir/trace_raw.bin" ]]; then
    python3 "$EMU_ROOT/tools/trace-trim.py" "$trace_out_dir/trace_raw.bin" 2>/dev/null || true
  fi

  echo "  BRIDGE $name ($compiler): $result"
}
export -f run_one_bridge

# ---------------------------------------------------------------------------
# Trace comparison helper
# ---------------------------------------------------------------------------

# Run trace-compare: prefer Rust binary, fall back to Python.
# Uses the same CLI interface for both.
run_trace_compare() {
  local rust_bin="$EMU_ROOT/target/release/trace-compare"
  if [[ -x "$rust_bin" ]]; then
    "$rust_bin" "$@"
  else
    python3 "$EMU_ROOT/tools/trace-compare.py" "$@"
  fi
}
export -f run_trace_compare

# ---------------------------------------------------------------------------
# Phase 6: Report
# ---------------------------------------------------------------------------

print_report() {
  local -n test_list=$1
  local run_hw=$2
  local has_trace=false
  [[ "$NO_TRACE" != "true" ]] && has_trace=true

  local compilers
  read -ra compilers <<< "$COMPILERS_STR"
  local num_compilers=${#compilers[@]}

  echo ""
  echo "==========================================================================="
  echo "  RESULTS"
  echo "==========================================================================="
  echo ""

  # --- Header ---
  # Build dynamic header based on active compilers and modes.
  local name_width=40
  local col_width=10

  # Print header.
  printf "%-${name_width}s" "TEST"
  for compiler in "${compilers[@]}"; do
    local label
    label="$(echo "$compiler" | sed 's/./\U&/')"  # Capitalize first letter
    if [[ "$run_hw" == "true" ]]; then
      printf "  %-${col_width}s" "${label}/HW"
    fi
    printf "  %-${col_width}s" "${label}/EMU"
  done
  if $has_trace; then
    for compiler in "${compilers[@]}"; do
      local label
      label="$(echo "$compiler" | sed 's/./\U&/')"
      printf "  %-20s" "${label}/TRACE"
    done
  fi
  echo ""

  # Print separator.
  printf "%-${name_width}s" "$(printf '%0.s-' $(seq 1 $name_width))"
  for compiler in "${compilers[@]}"; do
    if [[ "$run_hw" == "true" ]]; then
      printf "  %-${col_width}s" "$(printf '%0.s-' $(seq 1 $col_width))"
    fi
    printf "  %-${col_width}s" "$(printf '%0.s-' $(seq 1 $col_width))"
  done
  if $has_trace; then
    for _ in "${compilers[@]}"; do
      printf "  %-20s" "$(printf '%0.s-' $(seq 1 20))"
    done
  fi
  echo ""

  # --- Per-compiler counters ---
  # Use associative arrays keyed by compiler.
  declare -A compile_ok compile_fail
  declare -A bridge_pass bridge_fail bridge_skip bridge_timeout bridge_emumiss
  declare -A hw_pass hw_fail hw_skip hw_timeout hw_tdr
  for compiler in "${compilers[@]}"; do
    compile_ok[$compiler]=0
    compile_fail[$compiler]=0
    bridge_pass[$compiler]=0
    bridge_fail[$compiler]=0
    bridge_skip[$compiler]=0
    bridge_timeout[$compiler]=0
    bridge_emumiss[$compiler]=0
    hw_pass[$compiler]=0
    hw_fail[$compiler]=0
    hw_skip[$compiler]=0
    hw_timeout[$compiler]=0
    hw_tdr[$compiler]=0
  done
  declare -A trace_clean trace_diverge trace_error trace_skip
  for compiler in "${compilers[@]}"; do
    trace_clean[$compiler]=0
    trace_diverge[$compiler]=0
    trace_error[$compiler]=0
    trace_skip[$compiler]=0
  done

  local has_compile_fail=false

  # --- Data rows ---
  for name in "${test_list[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"

    printf "%-${name_width}s" "$name"

    for compiler in "${compilers[@]}"; do
      # Read compile result.
      local cr="FAIL"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && \
        cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"

      if [[ "$cr" != "OK" ]]; then
        compile_fail[$compiler]=$(( ${compile_fail[$compiler]} + 1 ))
        has_compile_fail=true
        if [[ "$run_hw" == "true" ]]; then
          printf "  %-${col_width}s" "FAIL*"
        fi
        printf "  %-${col_width}s" "FAIL*"
        continue
      fi

      compile_ok[$compiler]=$(( ${compile_ok[$compiler]} + 1 ))

      # Read HW result.
      if [[ "$run_hw" == "true" ]]; then
        local hr="SKIP"
        [[ -f "$RESULTS_DIR/${safe}.${compiler}.hw.result" ]] && \
          hr="$(< "$RESULTS_DIR/${safe}.${compiler}.hw.result")"
        printf "  %-${col_width}s" "$hr"
        case "$hr" in
          PASS)    hw_pass[$compiler]=$(( ${hw_pass[$compiler]} + 1 )) ;;
          TDR)     hw_tdr[$compiler]=$(( ${hw_tdr[$compiler]} + 1 ))
                   hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
          TIMEOUT) hw_timeout[$compiler]=$(( ${hw_timeout[$compiler]} + 1 ))
                   hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
          SKIP*)   hw_skip[$compiler]=$(( ${hw_skip[$compiler]} + 1 )) ;;
          *)       hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
        esac
      fi

      # Read bridge (EMU) result.
      local br="SKIP"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.bridge.result" ]] && \
        br="$(< "$RESULTS_DIR/${safe}.${compiler}.bridge.result")"
      printf "  %-${col_width}s" "$br"
      case "$br" in
        PASS)     bridge_pass[$compiler]=$(( ${bridge_pass[$compiler]} + 1 )) ;;
        TIMEOUT)  bridge_timeout[$compiler]=$(( ${bridge_timeout[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        EMU_MISS) bridge_emumiss[$compiler]=$(( ${bridge_emumiss[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        SKIP*)    bridge_skip[$compiler]=$(( ${bridge_skip[$compiler]} + 1 )) ;;
        *)        bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
      esac
    done

    # Trace columns (per-compiler).
    if $has_trace; then
      for compiler in "${compilers[@]}"; do
        local trace_summary="-"
        if [[ -f "$RESULTS_DIR/${safe}.${compiler}.trace.summary" ]]; then
          trace_summary="$(< "$RESULTS_DIR/${safe}.${compiler}.trace.summary")"
          case "$trace_summary" in
            CLEAN*)   trace_clean[$compiler]=$(( ${trace_clean[$compiler]} + 1 )) ;;
            DIVERGE*) trace_diverge[$compiler]=$(( ${trace_diverge[$compiler]} + 1 )) ;;
            ERROR*)   trace_error[$compiler]=$(( ${trace_error[$compiler]} + 1 )) ;;
            SKIP*|EMU_ONLY*|HW_ONLY*|NONE*) trace_skip[$compiler]=$(( ${trace_skip[$compiler]} + 1 )) ;;
          esac
        else
          trace_skip[$compiler]=$(( ${trace_skip[$compiler]} + 1 ))
        fi
        printf "  %-20s" "$trace_summary"
      done
    fi

    echo ""

    # Verbose: show log tail on failure.
    if [[ "$VERBOSE" == "true" ]]; then
      for compiler in "${compilers[@]}"; do
        local br="SKIP"
        [[ -f "$RESULTS_DIR/${safe}.${compiler}.bridge.result" ]] && \
          br="$(< "$RESULTS_DIR/${safe}.${compiler}.bridge.result")"
        if [[ "$br" != "PASS" ]] && [[ "$br" != "SKIP"* ]]; then
          local logf="$RESULTS_DIR/${safe}.${compiler}.bridge.log"
          if [[ -f "$logf" ]]; then
            echo "    --- $compiler bridge log tail ---"
            tail -5 "$logf" | sed 's/^/    /'
          fi
        fi
        if [[ "$run_hw" == "true" ]]; then
          local hr="SKIP"
          [[ -f "$RESULTS_DIR/${safe}.${compiler}.hw.result" ]] && \
            hr="$(< "$RESULTS_DIR/${safe}.${compiler}.hw.result")"
          if [[ "$hr" != "PASS" ]] && [[ "$hr" != "SKIP"* ]]; then
            local logf="$RESULTS_DIR/${safe}.${compiler}.hw.log"
            if [[ -f "$logf" ]]; then
              echo "    --- $compiler hw log tail ---"
              tail -5 "$logf" | sed 's/^/    /'
            fi
          fi
        fi
      done
    fi
  done

  # Footnote.
  local has_tdr=false
  for compiler in "${compilers[@]}"; do
    [[ ${hw_tdr[$compiler]} -gt 0 ]] && has_tdr=true
  done
  if $has_compile_fail || $has_tdr; then
    echo ""
    $has_compile_fail && echo "* = compile failed"
    $has_tdr && echo "TDR = hardware timeout detection and recovery (NPU hung)"
  fi

  # Show TDR suspect groups (from parallel runs).
  if [[ -s "$RESULTS_DIR/tdr_suspects.log" ]]; then
    echo ""
    echo "=== TDR Suspect Groups ==="
    echo "(Tests running concurrently when TDR was detected)"
    while IFS= read -r line; do
      echo "  $line"
    done < "$RESULTS_DIR/tdr_suspects.log"
  fi

  # --- Summary ---
  echo ""
  echo "=== Summary ==="
  for compiler in "${compilers[@]}"; do
    local label
    label="$(echo "$compiler" | sed 's/./\U&/')"
    local total=$(( ${compile_ok[$compiler]} + ${compile_fail[$compiler]} ))
    echo "${label}: ${compile_ok[$compiler]}/${total} compiled, ${bridge_pass[$compiler]} bridge pass, ${bridge_fail[$compiler]} bridge fail"
    if [[ ${bridge_timeout[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_timeout[$compiler]} timeout)"
    fi
    if [[ ${bridge_emumiss[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_emumiss[$compiler]} EMU_MISS)"
    fi
    if [[ "$run_hw" == "true" ]]; then
      local hw_extra=""
      [[ ${hw_tdr[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_tdr[$compiler]} TDR)"
      [[ ${hw_timeout[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_timeout[$compiler]} timeout)"
      echo "  HW: ${hw_pass[$compiler]} pass, ${hw_fail[$compiler]} fail, ${hw_skip[$compiler]} skip${hw_extra}"
    fi
  done
  if $has_trace; then
    for compiler in "${compilers[@]}"; do
      local label
      label="$(echo "$compiler" | sed 's/./\U&/')"
      echo "${label} trace: ${trace_clean[$compiler]} clean, ${trace_diverge[$compiler]} diverge, ${trace_error[$compiler]} error, ${trace_skip[$compiler]} skip"
    done
  fi
  echo "Logs: $RESULTS_DIR/"
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
      local compilers
      read -ra compilers <<< "$COMPILERS_STR"
      for _c in "${compilers[@]}"; do
        echo "SKIP_NPU2" > "$RESULTS_DIR/${safe}.${_c}.bridge.result"
        echo "SKIP_NPU2" > "$RESULTS_DIR/${safe}.${_c}.compile.result"
      done
      ((skipped_npu2++)) || true
    else
      runnable+=("$name")
    fi
  done

  if [[ $skipped_npu2 -gt 0 ]]; then
    info "Skipped $skipped_npu2 npu2-only test(s)"
  fi

  # ---- Phase 1b: Auto-rebuild plugin if Rust lib is newer ----------------

  local rust_lib="$EMU_ROOT/target/release/libxdna_emu.so"
  local installed_lib="$XRT_LIB/libxdna_emu.so"

  if [[ -f "$rust_lib" ]]; then
    if [[ ! -f "$installed_lib" ]] || [[ "$rust_lib" -nt "$installed_lib" ]]; then
      info "Plugin outdated -- rebuilding from $rust_lib"
      "$SCRIPT_DIR/rebuild-plugin.sh" 2>&1 | sed 's/^/  /'
    fi
  else
    warn "No release build found -- run 'cargo build --release' first"
  fi

  # ---- Phase 2: Compile --------------------------------------------------

  info "Phase 2: Compiling ${#runnable[@]} test(s) (-j${JOBS})"

  printf '%s\n' "${runnable[@]}" | xargs -P"$JOBS" -I{} bash -c 'compile_one "$@"' _ {}

  # Count compile results (per-compiler).
  local compile_ok=0 compile_fail=0
  local compilers
  read -ra compilers <<< "$COMPILERS_STR"
  for name in "${runnable[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"
    for compiler in "${compilers[@]}"; do
      local cr="FAIL"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"
      if [[ "$cr" == "OK" ]]; then
        ((compile_ok++)) || true
      else
        ((compile_fail++)) || true
      fi
    done
  done
  info "Phase 2 done: $compile_ok OK, $compile_fail failed (across ${#compilers[@]} compiler(s))"
  echo ""

  # Build list of tests where at least one compiler succeeded.
  local compiled=()
  for name in "${runnable[@]}"; do
    local safe
    safe="$(sanitize_name "$name")"
    local any_ok=false
    for compiler in "${compilers[@]}"; do
      local cr="FAIL"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"
      if [[ "$cr" == "OK" ]]; then
        any_ok=true
        break
      fi
    done
    if $any_ok; then
      compiled+=("$name")
    fi
  done

  # ---- Phase 3+4: Run HW+EMU concurrently --------------------------------

  # Build job list: (name:compiler) pairs that compiled successfully.
  # Split into parallel (safe) and quarantine (known TDR) pools.
  local all_jobs=()
  local hw_parallel_jobs=()
  local hw_quarantine_jobs=()
  for name in "${compiled[@]}"; do
    for compiler in "${compilers[@]}"; do
      local safe
      safe="$(sanitize_name "$name")"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] || continue
      [[ "$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")" == "OK" ]] || continue
      all_jobs+=("$name:$compiler")
      if is_quarantined "$name:$compiler"; then
        hw_quarantine_jobs+=("$name:$compiler")
      else
        hw_parallel_jobs+=("$name:$compiler")
      fi
    done
  done

  # Detect NPU hardware for HW runs.
  local real_bdf=""
  if $RUN_HW; then
    real_bdf="$(xrt-smi examine 2>/dev/null | grep -oP '\[0000:[0-9a-f:\.]+\]' | head -1 | tr -d '[]')" || true
    if [[ -z "$real_bdf" ]]; then
      info "HW: SKIPPED -- no NPU hardware detected"
      RUN_HW=false
    fi
  fi

  local hw_label="" emu_label=""
  $RUN_HW && hw_label="HW -j${NPU_HW_JOBS}"
  $RUN_EMU && emu_label="EMU -j${JOBS}"
  local q_label=""
  [[ ${#hw_quarantine_jobs[@]} -gt 0 ]] && q_label=", ${#hw_quarantine_jobs[@]} quarantined"
  info "Phase 3+4: Running ${#all_jobs[@]} job(s) (${hw_label:+$hw_label }${emu_label:+$emu_label}${q_label})"

  # Launch EMU for all jobs (parallel, no NPU constraint).
  local emu_pool_pid=""
  if $RUN_EMU; then
    for entry in "${all_jobs[@]}"; do
      local name="${entry%%:*}" compiler="${entry##*:}"
      echo "$name $compiler"
    done | xargs -P"$JOBS" -n2 bash -c 'run_one_bridge "$@"' _ &
    emu_pool_pid=$!
  fi

  # Launch HW with NPU job pool (if enabled), concurrently with EMU.
  if $RUN_HW; then
    local tdr_suspect_file="$RESULTS_DIR/tdr_suspects.log"
    : > "$tdr_suspect_file"
    local hw_total=$(( ${#hw_parallel_jobs[@]} + ${#hw_quarantine_jobs[@]} ))
    local hw_done=0

    # --- Parallel pool: safe tests at -j$NPU_HW_JOBS ---
    if [[ ${#hw_parallel_jobs[@]} -gt 0 ]]; then
      declare -A hw_pids=()   # pid -> "name:compiler"
      local hw_idx=0

      while [[ $hw_idx -lt ${#hw_parallel_jobs[@]} ]] || [[ ${#hw_pids[@]} -gt 0 ]]; do
        # Launch jobs while slots available and queue not empty.
        while [[ ${#hw_pids[@]} -lt $NPU_HW_JOBS ]] && [[ $hw_idx -lt ${#hw_parallel_jobs[@]} ]]; do
          local entry="${hw_parallel_jobs[$hw_idx]}"
          local hw_name="${entry%%:*}"
          local hw_compiler="${entry##*:}"
          ((hw_idx++)) || true

          (
            run_one_hardware "$hw_name" "$real_bdf" "$hw_compiler"
          ) &
          hw_pids[$!]="$entry"
        done

        # Wait for any one job to finish.
        if [[ ${#hw_pids[@]} -gt 0 ]]; then
          local done_pid=0
          wait -n -p done_pid "${!hw_pids[@]}" 2>/dev/null || true

          if [[ $done_pid -ne 0 ]] && [[ -n "${hw_pids[$done_pid]+x}" ]]; then
            local finished="${hw_pids[$done_pid]}"
            unset 'hw_pids[$done_pid]'
            local fin_name="${finished%%:*}"
            local fin_compiler="${finished##*:}"
            local fin_safe
            fin_safe="$(sanitize_name "$fin_name")"
            local hr="SKIP"
            [[ -f "$RESULTS_DIR/${fin_safe}.${fin_compiler}.hw.result" ]] && \
              hr="$(< "$RESULTS_DIR/${fin_safe}.${fin_compiler}.hw.result")"
            ((hw_done++)) || true
            local tdr_tag=""
            if [[ "$hr" == "TDR" ]]; then
              tdr_tag=" *** TDR DETECTED ***"
              local suspects="$finished"
              for spid in "${!hw_pids[@]}"; do
                suspects+=", ${hw_pids[$spid]}"
              done
              echo "TDR @$(uptime_sec)s -- concurrent: $suspects" >> "$tdr_suspect_file"
            fi
            echo "  [${hw_done}/${hw_total}] HW $fin_name ($fin_compiler): $hr  @$(uptime_sec)s${tdr_tag}"
          fi
        fi
      done
    fi

    # --- Retry: rerun non-quarantine TDR results serially ---
    # TDRs in the parallel pool are often collateral damage from a stuck
    # context left by another test.  Rerunning in isolation distinguishes
    # real failures from contention artifacts.
    local retry_jobs=()
    for entry in "${hw_parallel_jobs[@]}"; do
      local r_name="${entry%%:*}" r_compiler="${entry##*:}"
      local r_safe
      r_safe="$(sanitize_name "$r_name")"
      local r_result=""
      [[ -f "$RESULTS_DIR/${r_safe}.${r_compiler}.hw.result" ]] && \
        r_result="$(< "$RESULTS_DIR/${r_safe}.${r_compiler}.hw.result")"
      [[ "$r_result" == "TDR" ]] && retry_jobs+=("$entry")
    done

    if [[ ${#retry_jobs[@]} -gt 0 ]]; then
      info "HW retry: ${#retry_jobs[@]} TDR result(s) rerunning serially"
      for entry in "${retry_jobs[@]}"; do
        local r_name="${entry%%:*}" r_compiler="${entry##*:}"
        local r_safe
        r_safe="$(sanitize_name "$r_name")"
        # Save original log, then rerun.
        local orig_log="$RESULTS_DIR/${r_safe}.${r_compiler}.hw.log"
        [[ -f "$orig_log" ]] && cp "$orig_log" "${orig_log%.log}.tdr-orig.log"
        run_one_hardware "$r_name" "$real_bdf" "$r_compiler"
        local rr="SKIP"
        [[ -f "$RESULTS_DIR/${r_safe}.${r_compiler}.hw.result" ]] && \
          rr="$(< "$RESULTS_DIR/${r_safe}.${r_compiler}.hw.result")"
        if [[ "$rr" == "PASS" ]]; then
          echo "  RETRY $r_name ($r_compiler): PASS (was TDR collateral)"
        else
          echo "  RETRY $r_name ($r_compiler): $rr (confirmed failure)"
        fi
      done
    fi

    # --- Quarantine pool: known TDR tests, run last, serially ---
    if [[ ${#hw_quarantine_jobs[@]} -gt 0 ]]; then
      info "HW quarantine: running ${#hw_quarantine_jobs[@]} isolated test(s)"
      for entry in "${hw_quarantine_jobs[@]}"; do
        local q_name="${entry%%:*}"
        local q_compiler="${entry##*:}"
        run_one_hardware "$q_name" "$real_bdf" "$q_compiler"
        local q_safe
        q_safe="$(sanitize_name "$q_name")"
        local qr="SKIP"
        [[ -f "$RESULTS_DIR/${q_safe}.${q_compiler}.hw.result" ]] && \
          qr="$(< "$RESULTS_DIR/${q_safe}.${q_compiler}.hw.result")"
        ((hw_done++)) || true
        local q_tag=""
        [[ "$qr" == "TDR" ]] && q_tag=" *** TDR ***"
        echo "  [${hw_done}/${hw_total}] HW $q_name ($q_compiler): $qr  @$(uptime_sec)s [QUARANTINE]${q_tag}"
      done
    fi

    info "HW runs done"
  fi

  # Wait for EMU pool to finish.
  if [[ -n "$emu_pool_pid" ]]; then
    wait "$emu_pool_pid" 2>/dev/null || true
  fi
  info "EMU runs done"
  echo ""

  # ---- Phase 5: Automatic trace comparison --------------------------------

  if [[ "$NO_TRACE" != "true" ]]; then
    info "Phase 5: Comparing traces"
    for name in "${compiled[@]}"; do
      local safe
      safe="$(sanitize_name "$name")"
      for compiler in "${compilers[@]}"; do
        local hw_trace="$RESULTS_DIR/${safe}.${compiler}.hw/trace_raw.bin"
        local emu_trace="$RESULTS_DIR/${safe}.${compiler}.emu/trace_raw.bin"
        local events_file="$RESULTS_DIR/${safe}.${compiler}.hw/events.json"
        [[ ! -f "$events_file" ]] && events_file="$RESULTS_DIR/${safe}.${compiler}.emu/events.json"
        local summary_file="$RESULTS_DIR/${safe}.${compiler}.trace.summary"

        if [[ -f "$hw_trace" ]] && [[ -f "$emu_trace" ]]; then
          local cmp_log="$RESULTS_DIR/${safe}.${compiler}.trace.log"
          local cmp_out
          cmp_out="$(run_trace_compare --hw "$hw_trace" --emu "$emu_trace" 2>&1)" || true
          echo "$cmp_out" > "$cmp_log"

          if echo "$cmp_out" | grep -q "CLEAN"; then
            echo "CLEAN" > "$summary_file"
          elif echo "$cmp_out" | grep -q "DIVERGE"; then
            echo "DIVERGE" > "$summary_file"
          else
            echo "ERROR" > "$summary_file"
          fi
        elif [[ -f "$emu_trace" ]]; then
          echo "EMU_ONLY" > "$summary_file"
        elif [[ -f "$hw_trace" ]]; then
          echo "HW_ONLY" > "$summary_file"
        else
          echo "NONE" > "$summary_file"
        fi
      done
    done
  fi

  # ---- Phase 5b: Event sweep (optional) -----------------------------------

  if [[ "$SWEEP" == "true" ]]; then
    # Build list of tests eligible for sweep: must have passed on at least
    # one platform (HW or EMU) and not be trace-quarantined.
    local sweep_targets=()
    for name in "${compiled[@]}"; do
      is_trace_quarantined "$name" && continue
      local safe
      safe="$(sanitize_name "$name")"
      local any_pass=false
      for compiler in "${compilers[@]}"; do
        for suffix in hw bridge; do
          local rf="$RESULTS_DIR/${safe}.${compiler}.${suffix}.result"
          [[ -f "$rf" ]] && [[ "$(< "$rf")" == "PASS" ]] && any_pass=true
        done
        $any_pass && break
      done
      $any_pass && sweep_targets+=("$name")
    done

    if [[ ${#sweep_targets[@]} -gt 0 ]]; then
      info "Phase 5b: Event sweep for ${#sweep_targets[@]} test(s)"
      for name in "${sweep_targets[@]}"; do
        local safe
        safe="$(sanitize_name "$name")"
        local src_dir="$TEST_SRC/$name"
        local sweep_dir="$RESULTS_DIR/${safe}.sweep"
        local sweep_log="$RESULTS_DIR/${safe}.sweep.log"
        local sweep_args=("$src_dir" -o "$sweep_dir")

        # Pass through HW/EMU flags.
        $RUN_HW || sweep_args+=(--no-hw)
        $RUN_EMU || sweep_args+=(--no-emu)

        echo "  SWEEP $name ..."
        if python3 "$EMU_ROOT/tools/trace-sweep.py" "${sweep_args[@]}" \
            > "$sweep_log" 2>&1; then
          echo "  SWEEP $name: OK (see $sweep_dir/)"
        else
          echo "  SWEEP $name: FAIL (see $sweep_log)"
        fi
      done
    else
      info "Phase 5b: no tests eligible for sweep"
    fi
  fi

  # ---- Phase 6: Report ---------------------------------------------------

  info "Phase 6: Report"
  print_report tests "$RUN_HW"
}

main
