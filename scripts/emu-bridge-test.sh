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
# Four-phase pipelined architecture:
#   Phase 1: Discover       -- find tests, filter, skip npu2-only
#   Phase 2: Compile        -- parallel xclbin builds (both compilers in parallel) + test.exe
#   Phase 3+4: Run HW+EMU  -- HW (-j5) and EMU (-j$JOBS) run concurrently
#   Phase 4b: Trace         -- trace comparison (needs both HW+EMU results)
#   Phase 5: Report         -- per-compiler comparison matrix + summary
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
TRACE_MODE=""  # "", "default", "all", "sweep", "sweep-all"
COMPILER_MODE="both"  # "both", "chess", "peano"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)     FORCE_COMPILE=true; shift ;;
    --no-hw)       RUN_HW=false; shift ;;
    --no-emu)      RUN_EMU=false; shift ;;
    --list)        LIST_ONLY=true; shift ;;
    -v|--verbose)  VERBOSE=true; shift ;;
    --trace)       TRACE_MODE="default"; shift ;;
    --trace=*)
      TRACE_MODE="${1#--trace=}"
      case "$TRACE_MODE" in
        all|sweep|sweep-all|compare|compare-all) ;;
        *) echo "Unknown --trace mode: $TRACE_MODE (use: all, sweep, sweep-all, compare, compare-all)" >&2; exit 1 ;;
      esac
      shift ;;
    -j*)
      JOBS="${1#-j}"
      if [[ -z "$JOBS" ]] || ! [[ "$JOBS" =~ ^[0-9]+$ ]]; then
        echo "Invalid -j value: $1" >&2; exit 1
      fi
      shift ;;
    --chess-only|--chess)  COMPILER_MODE="chess"; shift ;;
    --peano-only|--peano)  COMPILER_MODE="peano"; shift ;;
    --serial-hw)           NPU_HW_JOBS=1; shift ;;
    --help|-h)
      cat <<'USAGE'
Usage: emu-bridge-test.sh [options] [test-name-filter]

Options:
  --compile       Force recompile all xclbins (default: use cached)
  --no-hw         Skip real hardware runs (default: hardware enabled)
  --no-emu        Skip emulator runs (default: emulator enabled)
  --list          List available tests and exit
  --trace         Run trace comparison (default events, passing tests only)
  --trace=all     Trace all tests (pass + fail)
  --trace=sweep   Full event sweep (passing tests only)
  --trace=sweep-all  Full sweep, all tests
  --trace=compare    Sweep + serial-vs-parallel determinism check (HW-passing)
  --trace=compare-all  Determinism check, all tests
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
export RESULTS_DIR FORCE_COMPILE VERBOSE TRACE_MODE RUN_EMU
export MLIR_AIE TEST_SRC BUILD_BASE EMU_ROOT
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
  [[ -n "${TRACE_QUARANTINE[${1}]+x}" ]]
}

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
    sed \
      -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
      -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
      "$src_dir/test.cpp" > "$build_dir/test.cpp"
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

  # Snapshot TDR count and uptime for post-test attribution.
  local tdr_before
  tdr_before="$(tdr_count)"
  local t_start
  t_start="$(uptime_sec)"

  local rc=0
  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$bdf"
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
  echo "  BRIDGE $name ($compiler): $result"
}
export -f run_one_bridge

# ---------------------------------------------------------------------------
# Phase 4b: Trace comparison
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

# Compile traced xclbin base for a single test (used by compile-ahead).
# Delegates to trace-sweep.py --compile-only, writes status to
# $RESULTS_DIR/${safe}.trace-compile/compile-status.txt.
compile_trace_base_one() {
  local name="$1"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local compile_dir="$RESULTS_DIR/${safe}.trace-compile"

  python3 "$EMU_ROOT/tools/trace-sweep.py" "$src_dir" \
    -o "$compile_dir" --compile-only \
    > "$RESULTS_DIR/${safe}.trace-compile.log" 2>&1
  local rc=$?

  if [[ $rc -eq 0 ]] && [[ -f "$compile_dir/compile-status.txt" ]] \
      && grep -q '^OK' "$compile_dir/compile-status.txt"; then
    echo "  TRACE COMPILE $name: OK"
  else
    echo "  TRACE COMPILE $name: FAIL"
  fi
}
export -f compile_trace_base_one

# Run trace comparison for a single test.
# Uses trace-sweep.py for sweep mode, or trace-inject + trace-run +
# trace-compare for default (8-event) mode.
#
# Writes $RESULTS_DIR/${safe}.trace.summary (one line) and full report
# to $RESULTS_DIR/${safe}.trace.log.
trace_one_test() {
  local name="$1"
  local mode="$2"  # "default" or "sweep"
  local compiler="${3:-peano}"  # defaults to peano for backward compat
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local trace_dir="$RESULTS_DIR/${safe}.${compiler}.trace"
  local summary_file="$RESULTS_DIR/${safe}.${compiler}.trace.summary"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.trace.log"
  local tools_dir="$EMU_ROOT/tools"

  mkdir -p "$trace_dir"
  : > "$log_file"

  if [[ "$mode" == "sweep" || "$mode" == "sweep-all" ]]; then
    # Full sweep: delegate to trace-sweep.py
    # TODO: pass compiler flags to trace-sweep.py when it supports them.
    # For now, sweep always uses the default (peano) compiler.
    local sweep_args=("$src_dir" -o "$trace_dir/sweep")
    if [[ "$RUN_HW" != "true" ]]; then
      sweep_args+=(--no-hw)
    fi
    if ! python3 "$tools_dir/trace-sweep.py" "${sweep_args[@]}" >> "$log_file" 2>&1; then
      echo "ERROR sweep_failed" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (sweep failed)"
      return
    fi

    # Check if sweep was skipped (exit 0 but with skipped manifest)
    local sweep_manifest="$trace_dir/sweep/sweep-manifest.json"
    if [[ -f "$sweep_manifest" ]]; then
      local skip_reason
      skip_reason="$(python3 -c "import json,sys; m=json.load(open('$sweep_manifest')); print(m.get('reason','')) if m.get('skipped') else sys.exit(1)" 2>/dev/null)" && {
        echo "SKIP $skip_reason" > "$summary_file"
        echo "  TRACE $name ($compiler): SKIP ($skip_reason)"
        return
      }
    fi

    # Trim trace buffers to actual data length
    python3 "$tools_dir/trace-trim.py" --dir "$trace_dir/sweep" >> "$log_file" 2>&1 || true

    # Compare
    if ! run_trace_compare --sweep "$trace_dir/sweep" \
        -o "$trace_dir/report.txt" >> "$log_file" 2>&1; then
      echo "ERROR compare_failed" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (compare failed)"
      return
    fi
  else
    # Default 8-event mode: inject, compile, run HW+EMU, compare.

    # Step 1: Inject tracing into MLIR
    local traced_dir="$trace_dir/traced"
    if ! python3 "$tools_dir/trace-inject.py" "$src_dir" -o "$traced_dir" \
        >> "$log_file" 2>&1; then
      echo "ERROR injection_failed" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (injection failed)"
      return
    fi

    local manifest="$traced_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
      echo "ERROR no_manifest" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (no manifest)"
      return
    fi

    # Check if injection was skipped (unsupported test)
    if python3 -c "import json,sys; m=json.load(open('$manifest')); sys.exit(0 if m.get('skipped') else 1)" 2>/dev/null; then
      echo "SKIP unsupported" > "$summary_file"
      echo "  TRACE $name ($compiler): SKIP (unsupported)"
      return
    fi

    # Step 2: Compile traced xclbin
    local aiecc_compiler_flags="--no-xchesscc"
    if [[ "$compiler" == "chess" ]]; then
      aiecc_compiler_flags="--xchesscc --xbridge"
    fi
    if ! ( cd "$traced_dir" && nice -n 19 aiecc.py \
        --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts \
        --no-compile-host --alloc-scheme=basic-sequential $aiecc_compiler_flags \
        --xclbin-name=aie.xclbin --npu-insts-name=insts.bin \
        ./aie_traced.mlir ) >> "$log_file" 2>&1; then
      echo "ERROR compile_failed" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (compile failed)"
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

    # Run HW in foreground, then wait for NPU idle before next test
    if [[ "$RUN_HW" == "true" ]]; then
      if ! python3 "$tools_dir/trace-run.py" "$manifest" -o "$hw_dir" \
          >> "$log_file.hw" 2>&1; then
        hw_ok=false
      fi
      # Wait for NPU to release hardware contexts (fast poll, not fixed sleep)
      wait_npu_idle
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
      echo "  TRACE $name ($compiler): ERROR (hw run failed)"
      return
    fi
    if ! $emu_ok; then
      echo "ERROR emu_run_failed" > "$summary_file"
      echo "  TRACE $name ($compiler): ERROR (emu run failed)"
      return
    fi

    # Trim trace buffers to actual data length
    python3 "$tools_dir/trace-trim.py" --dir "$trace_dir" >> "$log_file" 2>&1 || true

    # Step 5: Compare (only if both traces exist)
    if [[ "$RUN_HW" == "true" ]] && [[ -f "$trace_dir/hw/trace_raw.bin" ]] \
        && [[ -f "$trace_dir/emu/trace_raw.bin" ]]; then
      if ! run_trace_compare \
          --hw "$trace_dir/hw/trace_raw.bin" \
          --emu "$trace_dir/emu/trace_raw.bin" \
          -o "$trace_dir/report.txt" >> "$log_file" 2>&1; then
        echo "ERROR compare_failed" > "$summary_file"
        echo "  TRACE $name ($compiler): ERROR (compare failed)"
        return
      fi
    else
      # EMU-only: no comparison possible, just record that trace was collected
      echo "EMU_ONLY collected" > "$summary_file"
      echo "  TRACE $name ($compiler): EMU_ONLY (trace collected, no HW to compare)"
      return
    fi
  fi

  # Parse the report to produce a one-line summary.
  local report="$trace_dir/report.txt"
  if [[ ! -f "$report" ]]; then
    echo "ERROR no_report" > "$summary_file"
    echo "  TRACE $name ($compiler): ERROR (no report)"
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
      echo "  TRACE $name ($compiler): CLEAN (${clean:-0} event types, ${pairs} pairs)"
    else
      echo "DIVERGE ${diverged} of $((${clean:-0}+${diverged})) event types" > "$summary_file"
      echo "  TRACE $name ($compiler): DIVERGE (${diverged} of $((${clean:-0}+${diverged})) event types)"
    fi
  else
    echo "UNKNOWN parse_error" > "$summary_file"
    echo "  TRACE $name ($compiler): UNKNOWN (could not parse report)"
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
            SKIP*|EMU_ONLY*) trace_skip[$compiler]=$(( ${trace_skip[$compiler]} + 1 )) ;;
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

  # Launch background trace compile-ahead (overlaps with HW+EMU runs).
  local trace_compile_pid=""
  if [[ -n "$TRACE_MODE" ]] && [[ "$TRACE_MODE" == "compare" || "$TRACE_MODE" == "compare-all" ]]; then
    local trace_compile_targets=()
    for name in "${compiled[@]}"; do
      is_trace_quarantined "$name" && continue
      trace_compile_targets+=("$name")
    done

    if [[ ${#trace_compile_targets[@]} -gt 0 ]]; then
      info "Background: compiling ${#trace_compile_targets[@]} traced xclbin(s) (-j${JOBS})"
      (
        printf '%s\n' "${trace_compile_targets[@]}" | \
          xargs -P"$JOBS" -I{} bash -c 'compile_trace_base_one "$@"' _ {}
      ) &
      trace_compile_pid=$!
    fi
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

  # ---- Phase 4b: Trace comparison (optional) ------------------------------

  if [[ -n "$TRACE_MODE" ]]; then
    # Determine which tests to trace.
    local trace_targets=()
    local trace_include_failing=false
    local trace_sweep=false
    local trace_compare=false

    case "$TRACE_MODE" in
      default)     ;;
      all)         trace_include_failing=true ;;
      sweep)       trace_sweep=true ;;
      sweep-all)   trace_sweep=true; trace_include_failing=true ;;
      compare)     trace_sweep=true; trace_compare=true ;;
      compare-all) trace_sweep=true; trace_compare=true; trace_include_failing=true ;;
    esac

    # For compare mode, filter on HW pass (not bridge pass) since we're
    # measuring NPU determinism, not emulator accuracy.
    for name in "${compiled[@]}"; do
      if ! $trace_include_failing; then
        local safe
        safe="$(sanitize_name "$name")"
        local any_pass=false
        for compiler in "${compilers[@]}"; do
          if $trace_compare; then
            # Compare mode: require HW pass.
            local hr="FAIL"
            [[ -f "$RESULTS_DIR/${safe}.${compiler}.hw.result" ]] && hr="$(< "$RESULTS_DIR/${safe}.${compiler}.hw.result")"
            [[ "$hr" == "PASS" ]] && any_pass=true
          else
            # Normal trace mode: require bridge (EMU) pass.
            local br="FAIL"
            [[ -f "$RESULTS_DIR/${safe}.${compiler}.bridge.result" ]] && br="$(< "$RESULTS_DIR/${safe}.${compiler}.bridge.result")"
            [[ "$br" == "PASS" ]] && any_pass=true
          fi
          $any_pass && break
        done
        if ! $any_pass; then
          continue
        fi
      fi
      # Skip trace-quarantined tests (IOMMU faults, NPU wedge).
      if $trace_compare && is_trace_quarantined "$name"; then
        continue
      fi
      # Skip HW-quarantined tests in compare mode (they TDR).
      if $trace_compare && is_quarantined "$name:chess" && is_quarantined "$name:peano"; then
        continue
      fi
      trace_targets+=("$name")
    done

    if $trace_compare; then
      # --- Compare mode: serial-vs-parallel determinism check ---
      # Wait for background trace compile-ahead to finish.
      if [[ -n "$trace_compile_pid" ]]; then
        info "Phase 4b: waiting for trace compile-ahead to finish..."
        wait "$trace_compile_pid" 2>/dev/null || true
        trace_compile_pid=""
        info "Phase 4b: trace compile-ahead done"
      fi

      if [[ ${#trace_targets[@]} -gt 0 ]]; then
        info "Phase 4b: Serial vs Parallel trace comparison for ${#trace_targets[@]} test(s)"
        local cmp_pass=0 cmp_fail=0 cmp_err=0 cmp_skip=0
        local cmp_results_file="$RESULTS_DIR/trace-compare-results.txt"
        : > "$cmp_results_file"

        # Snapshot IOMMU fault count before starting.
        local iommu_before
        iommu_before="$(iommu_fault_count)"

        for name in "${trace_targets[@]}"; do
          local safe
          safe="$(sanitize_name "$name")"
          local src_dir="$TEST_SRC/$name"
          local cmp_dir="$RESULTS_DIR/${safe}.trace-compare"
          local cmp_log="$RESULTS_DIR/${safe}.trace-compare.log"

          local test_idx=$((cmp_pass + cmp_fail + cmp_err + cmp_skip + 1))

          # Pre-flight: wait for NPU health (may need recovery from prior fault).
          local wait_start
          wait_start="$(uptime_sec)"
          local npu_ok=false
          for _w in $(seq 1 30); do
            if npu_health_check; then
              npu_ok=true
              break
            fi
            echo "    waiting for NPU recovery... ($(( $(uptime_sec) - wait_start ))s)"
            sleep 2
          done
          if ! $npu_ok; then
            err "NPU did not recover after 60s -- aborting remaining tests"
            echo "ABORT npu_wedged" > "$RESULTS_DIR/${safe}.trace-compare.result"
            ((cmp_err++)) || true
            break
          fi

          echo "  [${test_idx}/${#trace_targets[@]}] COMPARE $name  @$(uptime_sec)s"

          # Run trace-sweep.py with --compare-parallel.
          # Use pre-compiled base if compile-ahead succeeded.
          local compile_dir="$RESULTS_DIR/${safe}.trace-compile"
          local sweep_args=("$src_dir" -o "$cmp_dir" --compare-parallel --hw-jobs "$NPU_HW_JOBS" --no-emu)
          if [[ -f "$compile_dir/compile-status.txt" ]] \
              && grep -q '^OK' "$compile_dir/compile-status.txt"; then
            # Single-pass: base/ dir directly under compile_dir.
            # Multi-pass: pass_NN/base/ dirs under compile_dir.
            if [[ -d "$compile_dir/base" ]]; then
              sweep_args+=(--use-base "$compile_dir/base")
            elif [[ -d "$compile_dir/pass_00" ]]; then
              sweep_args+=(--use-base "$compile_dir")
            fi
          fi

          local iommu_pre_test
          iommu_pre_test="$(iommu_fault_count)"
          local sweep_result
          sweep_result=$(
            python3 "$EMU_ROOT/tools/trace-sweep.py" "${sweep_args[@]}" \
              2>&1
          ) || true
          echo "$sweep_result" > "$cmp_log"

          # Post-flight: check for IOMMU faults from THIS test.
          local iommu_post_test
          iommu_post_test="$(iommu_fault_count)"
          if [[ $iommu_post_test -gt $iommu_pre_test ]]; then
            local new_faults=$((iommu_post_test - iommu_pre_test))
            err "IOMMU FAULT: $name caused $new_faults page fault(s)"
            echo "IOMMU_FAULT $new_faults faults" > "$RESULTS_DIR/${safe}.trace-compare.result"
            echo "    -> IOMMU_FAULT ($new_faults page faults -- waiting for NPU recovery)"
            ((cmp_err++)) || true
            # Don't break -- wait for recovery and continue with next test.
            continue
          fi

          # Parse the serial-vs-parallel report.
          local report="$cmp_dir/serial-vs-parallel-report.txt"
          if [[ -f "$report" ]]; then
            if grep -q 'DETERMINISTIC' "$report"; then
              local batches diverged
              batches="$(grep 'batches compared' "$report" | grep -oP '\d+')"
              diverged="$(grep 'Diverged:' "$report" | grep -oP '\d+')"
              if [[ "${diverged:-0}" -eq 0 ]]; then
                echo "DETERMINISTIC ${batches:-?} batches" > "$RESULTS_DIR/${safe}.trace-compare.result"
                echo "    -> DETERMINISTIC (${batches:-?} batches)"
                ((cmp_pass++)) || true
              else
                echo "NON-DETERMINISTIC ${diverged}/${batches} diverged" > "$RESULTS_DIR/${safe}.trace-compare.result"
                echo "    -> NON-DETERMINISTIC (${diverged}/${batches} diverged)"
                ((cmp_fail++)) || true
              fi
            else
              echo "ERROR no_verdict" > "$RESULTS_DIR/${safe}.trace-compare.result"
              echo "    -> ERROR (no verdict in report)"
              ((cmp_err++)) || true
            fi
          elif grep -q 'skipped' "$cmp_log" 2>/dev/null; then
            echo "SKIP" > "$RESULTS_DIR/${safe}.trace-compare.result"
            echo "    -> SKIP (trace injection unsupported)"
            ((cmp_skip++)) || true
          else
            echo "ERROR no_report" > "$RESULTS_DIR/${safe}.trace-compare.result"
            echo "    -> ERROR (no report generated)"
            ((cmp_err++)) || true
          fi
        done

        echo ""
        echo "=== Determinism Summary ==="
        echo "  Tests:             ${#trace_targets[@]}"
        echo "  Deterministic:     $cmp_pass"
        echo "  Non-deterministic: $cmp_fail"
        echo "  Skipped:           $cmp_skip"
        echo "  Errors/aborts:     $cmp_err"
        # Save summary
        {
          echo "tests=${#trace_targets[@]}"
          echo "deterministic=$cmp_pass"
          echo "non_deterministic=$cmp_fail"
          echo "skipped=$cmp_skip"
          echo "errors=$cmp_err"
        } > "$RESULTS_DIR/trace-compare-summary.txt"

        info "Phase 4b done"
      else
        info "Phase 4b: No tests eligible for determinism comparison"
      fi
    else
      # --- Normal trace modes: sweep or default ---
      local trace_mode_arg="default"
      if $trace_sweep; then
        trace_mode_arg="sweep"
      fi

      if [[ ${#trace_targets[@]} -gt 0 ]]; then
        info "Phase 4b: Trace comparison for ${#trace_targets[@]} test(s) (mode=$trace_mode_arg)"
        for name in "${trace_targets[@]}"; do
          for compiler in "${compilers[@]}"; do
            local safe
            safe="$(sanitize_name "$name")"
            [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] || continue
            [[ "$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")" == "OK" ]] || continue
            trace_one_test "$name" "$trace_mode_arg" "$compiler"
          done
        done
        info "Phase 4b done"
      else
        info "Phase 4b: No tests eligible for tracing"
      fi
    fi
    echo ""
  fi

  # Clean up any remaining background trace compile.
  if [[ -n "$trace_compile_pid" ]]; then
    wait "$trace_compile_pid" 2>/dev/null || true
  fi

  # ---- Phase 5: Report ---------------------------------------------------

  info "Phase 5: Report"
  print_report tests "$RUN_HW"
}

main
