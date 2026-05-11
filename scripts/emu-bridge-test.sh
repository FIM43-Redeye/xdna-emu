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
#   Phase 3+4: Run HW+EMU   -- HW (serial) and EMU (-j$JOBS) run concurrently
#   Phase 5: Trace compare   -- automatic trace comparison (HW vs EMU)
#   Phase 5c: aiesim        -- run aiesimulator on Chess builds + VCD coverage
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
#   ./scripts/emu-bridge-test.sh --aiesim           # also run aiesimulator + VCD audit

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
AIETOOLS_DIR="${AIETOOLS_DIR:-/home/triple/npu-work/amd-unified-software/aietools}"

# Compiler paths (auto-detect from environment)
PEANO_CLANG="${PEANO_INSTALL_DIR:-${MLIR_AIE}/../llvm-aie/install}/bin/clang++"
PEANO_INCLUDE="${MLIR_AIE_INSTALL_DIR:-${MLIR_AIE}/install}/include"
CHESS_INCLUDE="${AIETOOLS_DIR}/include"

# Peano kernel compilation flags (from mlir-aie makefile-common)
PEANO_KERNEL_FLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG"
PEANO_KERNEL_FLAGS+=" -Wno-parentheses -Wno-attributes -Wno-macro-redefined"
PEANO_KERNEL_FLAGS+=" -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw"

# Results directory -- under build/ so they survive reboots (unlike /tmp).
# Override with BRIDGE_TEST_RESULTS env var if needed.
RESULTS_DIR="${BRIDGE_TEST_RESULTS:-${EMU_ROOT}/build/bridge-test-results/$(date +%Y%m%d)}"
mkdir -p "$RESULTS_DIR"
# Maintain a 'latest' symlink for easy access.
ln -sfn "$RESULTS_DIR" "${RESULTS_DIR%/*}/latest"

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
PC_ANCHORED=false
MODE2=false
RUN_AIESIM=false
NO_TIMEOUT=false
WITH_HW_CYCLES=${WITH_HW_CYCLES:-false}
WITH_CYCLE_DIFF=${WITH_CYCLE_DIFF:-false}

# Phase E dual-bound EMU timing constants.
# EMU_SECONDS_PER_CYCLE: wall-clock seconds per simulated cycle. Emulator's
# reported simulation rate is ~800 MHz (pessimistic) to ~1 GHz; 2e-9 s/cycle
# = 500 M sim-cycles/sec is a conservative starting value that gives headroom
# on both figures. In practice the 600 s wall-clock floor (below) dominates
# for any test under ~300 G cycles, so this mostly matters as a sanity cap
# for pathologically long runs.
EMU_SECONDS_PER_CYCLE=${EMU_SECONDS_PER_CYCLE:-2e-9}
EMU_CYCLE_BUDGET_MULTIPLIER=${EMU_CYCLE_BUDGET_MULTIPLIER:-2.0}
export EMU_SECONDS_PER_CYCLE EMU_CYCLE_BUDGET_MULTIPLIER

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
    --trace)               NO_TRACE=false; shift ;;
    --sweep)               SWEEP=true; shift ;;
    --trace=pc-anchored)   PC_ANCHORED=true; NO_TRACE=false; shift ;;
    --mode2)               MODE2=true; PC_ANCHORED=true; NO_TRACE=false; shift ;;
    --aiesim)              RUN_AIESIM=true; shift ;;
    --serial-hw)           NPU_HW_JOBS=1; shift ;;
    --parallel-hw)         NPU_HW_JOBS="${NPU_HW_JOBS_PARALLEL:-5}"; shift ;;
    --no-timeout)          NO_TIMEOUT=true; shift ;;
    --with-hw-cycles)      WITH_HW_CYCLES=true; shift ;;
    --with-cycle-diff)     WITH_CYCLE_DIFF=true; WITH_HW_CYCLES=true; shift ;;
    --help|-h)
      cat <<'USAGE'
Usage: emu-bridge-test.sh [options] [test-name-regex]

Options:
  --compile       Force recompile all xclbins (default: use cached)
  --no-hw         Skip real hardware runs (default: hardware enabled)
  --no-emu        Skip emulator runs (default: emulator enabled)
  --trace         Enable trace injection and comparison (default: on)
  --no-trace      Disable trace preparation (e.g., when only validating
                  functional correctness)
  --sweep         Run full event sweep (trace-sweep.py) on passing tests after runs
  --trace=pc-anchored
                  Run mode-1 (event_pc) lockstep sweep on passing tests and
                  produce a PC-anchored HW/EMU comparison report per test.
                  Auto-detects compute tiles from the MLIR source. Outputs go
                  to RESULTS_DIR/<test>.<compiler>.pc-anchored/.
  --mode2         Implies --trace=pc-anchored, additionally captures a
                  mode-2 (inst_exec) HW baseline per test (already the
                  default behavior of trace-sweep.py, made explicit here).
                  Mode-2 capture currently relies on the test/CDO writing
                  Trace_Control0 mode=2; see TODOs in this script for the
                  Phase 0 kernel work and the EMU-side raw-stream output
                  path needed to fully wire HW vs EMU mode-2 comparison.
  --aiesim        Run aiesimulator on Chess builds + VCD coverage audit
  --list          List available tests and exit
  -jN             Override parallelism (default: nproc)
  -v, --verbose   Show log snippets on failure
  --chess-only    Only compile/run with Chess compiler (ground truth)
  --peano-only    Only compile/run with Peano compiler
  --serial-hw     Run hardware tests sequentially (explicit, same as default)
  --parallel-hw   Run hardware tests in parallel (-j5) for speed
  --no-timeout    Run EMU without wall-clock timeout (use for very long runs)
  --with-hw-cycles  Run trace-based HW cycle capture pipeline per HW test;
                    emits cycles.HW.<variant>.txt beside each result.
  --with-cycle-diff Additionally run EMU through the trace pipeline and
                    compare against HW via trace-compare. Implies
                    --with-hw-cycles.
  (default: both compilers, Chess is ground truth, HW serial)

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
export RESULTS_DIR FORCE_COMPILE VERBOSE RUN_EMU NO_TRACE SWEEP PC_ANCHORED MODE2 RUN_AIESIM NO_TIMEOUT WITH_HW_CYCLES WITH_CYCLE_DIFF
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
# Trace-injection incompatibility list (Phase E Task 10)
# ---------------------------------------------------------------------------
# Tests whose MLIR fails to compile after Phase B trace injection. Known
# trace-incompat tests still have a useful HW-only signal; skipping injection
# up-front saves a noisy compile failure every --with-hw-cycles run.

TRACE_INCOMPAT_FILE="${SCRIPT_DIR}/trace-incompat-tests.txt"
declare -A TRACE_INCOMPAT=()
if [[ -f "$TRACE_INCOMPAT_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line// /}"
    [[ -z "$line" ]] && continue
    TRACE_INCOMPAT["$line"]=1
  done < "$TRACE_INCOMPAT_FILE"
  if [[ ${#TRACE_INCOMPAT[@]} -gt 0 ]]; then
    echo ">>> Trace-injection incompat: ${#TRACE_INCOMPAT[@]} test(s) will have injection skipped"
  fi
fi
export TRACE_INCOMPAT_FILE

is_trace_incompat() {
  # Fast path: associative array (main process only -- not exported).
  if [[ ${#TRACE_INCOMPAT[@]} -gt 0 ]] 2>/dev/null; then
    [[ -n "${TRACE_INCOMPAT[${1}]+x}" ]]
    return
  fi
  # Slow path: file read (subshells via xargs).
  local name="$1"
  [[ -f "$TRACE_INCOMPAT_FILE" ]] || return 1
  while IFS= read -r line; do
    local entry="${line%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    [[ "$entry" == "$name" ]] && return 0
  done < "$TRACE_INCOMPAT_FILE"
  return 1
}
export -f is_trace_incompat

# ---------------------------------------------------------------------------
# Cycle-drift overrides (Phase E Task 11)
# ---------------------------------------------------------------------------
# Per-test EMU/HW ratio bounds for MATCH/DRIFT classification. Defaults to
# [0.5, 2.0]; overridden per-test via scripts/cycle-drift-overrides.txt.

DRIFT_OVERRIDES_FILE="${SCRIPT_DIR}/cycle-drift-overrides.txt"
declare -A DRIFT_LOWER=() DRIFT_UPPER=()
if [[ -f "$DRIFT_OVERRIDES_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | awk '{$1=$1; print}')"  # trim
    [[ -z "$line" ]] && continue
    read -r _t _l _u <<< "$line"
    [[ -z "$_t" || -z "$_l" || -z "$_u" ]] && continue
    DRIFT_LOWER["$_t"]="$_l"
    DRIFT_UPPER["$_t"]="$_u"
  done < "$DRIFT_OVERRIDES_FILE"
  if [[ ${#DRIFT_LOWER[@]} -gt 0 ]]; then
    echo ">>> Cycle-drift overrides: ${#DRIFT_LOWER[@]} test(s) with custom bounds"
  fi
fi
export DRIFT_OVERRIDES_FILE

# _lookup_drift_bounds <test_name> _out_lower_var _out_upper_var
# Sets two caller-named variables to the accepted EMU/HW ratio bounds.
# Returns per-test overrides if listed in DRIFT_OVERRIDES_FILE, else [0.5, 2.0].
_lookup_drift_bounds() {
    local t="$1"
    local _out_l="$2"
    local _out_u="$3"
    local l="0.5" u="2.0"
    # Fast path: in-memory array (main process).
    if [[ ${#DRIFT_LOWER[@]} -gt 0 ]] 2>/dev/null; then
        if [[ -n "${DRIFT_LOWER[$t]+x}" ]]; then
            l="${DRIFT_LOWER[$t]}"
            u="${DRIFT_UPPER[$t]}"
        fi
    elif [[ -f "$DRIFT_OVERRIDES_FILE" ]]; then
        # Subshell path: file read.
        while IFS= read -r line; do
            line="${line%%#*}"
            line="$(echo "$line" | awk '{$1=$1; print}')"
            [[ -z "$line" ]] && continue
            read -r _t _l _u <<< "$line"
            if [[ "$_t" == "$t" ]]; then
                l="$_l"; u="$_u"; break
            fi
        done < "$DRIFT_OVERRIDES_FILE"
    fi
    printf -v "$_out_l" '%s' "$l"
    printf -v "$_out_u" '%s' "$u"
}
export -f _lookup_drift_bounds

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

# Wall-clock seconds since this script started, for user-facing progress.
SCRIPT_START_EPOCH=${SCRIPT_START_EPOCH:-$(date +%s)}
export SCRIPT_START_EPOCH
elapsed_sec() {
  echo $(( $(date +%s) - SCRIPT_START_EPOCH ))
}
export -f elapsed_sec

# Check NPU health: can it create contexts?  Returns 0 if healthy.
npu_health_check() {
  xrt-smi examine -r aie-partitions &>/dev/null
}
export -f npu_health_check

# ---------------------------------------------------------------------------
# Helpers (shared with parallel jobs via export -f)
# ---------------------------------------------------------------------------

# Find the best .lit/.py file with RUN lines in a test directory.
# Prefers run.lit, then *.lit, then aie2.py (Python kernel generators),
# then test.py (Python-driven host harnesses).
# Returns empty string if no suitable file found.
find_lit_file() {
  local dir="$1"
  if [[ -f "$dir/run.lit" ]]; then
    echo "$dir/run.lit"
    return
  fi
  # Fall back to first *.lit file (alphabetically).
  local first
  first="$(find "$dir" -maxdepth 1 -name '*.lit' -print 2>/dev/null | sort | head -1)"
  if [[ -n "$first" ]]; then
    echo "$first"
    return
  fi
  # Python-generated kernel tests: aie2.py emits aie.mlir and carries RUN lines.
  if [[ -f "$dir/aie2.py" ]] && grep -q 'RUN:' "$dir/aie2.py" 2>/dev/null; then
    echo "$dir/aie2.py"
    return
  fi
  # Python-driven host harnesses: test.py IS the test executable; its RUN
  # lines drive aie-opt/aiecc plus the python3 test.py invocation itself.
  # Example: vec_mul_trace_distribute_lateral.
  if [[ -f "$dir/test.py" ]] && grep -q 'RUN:' "$dir/test.py" 2>/dev/null; then
    echo "$dir/test.py"
    return
  fi
}

# Check if a test directory has a runnable test.
#
# Standard tests have: test.cpp + (aie.mlir or run.lit)
# Python-generated kernel tests have: test.cpp + aie2.py (generates aie2.mlir)
# Python-driven host tests have: test.py + aie.mlir (test.py is the harness;
#   has no test.cpp; the bridge runs it via `python3 test.py` instead of
#   building a test.exe). Example: vec_mul_trace_distribute_lateral.
# Nested tests: handled by subdirectory scan in discover_tests().
is_standard_test() {
  local dir="$1"
  local lit
  lit="$(find_lit_file "$dir")"
  if [[ -f "$dir/test.cpp" ]]; then
    [[ -f "$dir/aie.mlir" || -n "$lit" || -f "$dir/aie2.py" ]]
    return $?
  fi
  # Python-driven host: test.py + aie.mlir (no test.cpp).
  [[ -f "$dir/test.py" && -f "$dir/aie.mlir" ]]
}

# True if the test directory uses test.py as its host harness (no test.cpp).
# Discovery / compile / run paths branch on this to skip test.exe machinery.
is_python_host_test() {
  local dir="$1"
  [[ ! -f "$dir/test.cpp" ]] && [[ -f "$dir/test.py" ]] \
    && grep -q 'RUN:' "$dir/test.py" 2>/dev/null
}

# Check if a test requires npu2 (AIE2P -- skip for now).
# A test is npu2-only if its REQUIRES line mentions ryzen_ai_npu2 but NOT
# ryzen_ai_npu1.  The REQUIRES directive is authoritative.
#
# NOTE: %run_on_npu1% commands in the lit file are conditional execution
# prefixes, NOT compatibility indicators.  A test with REQUIRES: npu2 and
# %run_on_npu1% commands is still npu2-only -- the conditional commands
# exist for the lit infrastructure, not to signal NPU1 support.
# (add_one_two_txn was incorrectly included on NPU1 because of this.)
# Checks ALL .lit files in the directory.
requires_npu2() {
  local dir="$1"
  local any_lit=false
  local any_npu2=false
  for lit in "$dir"/*.lit; do
    [[ -f "$lit" ]] || continue
    any_lit=true
    if grep -q 'REQUIRES:.*ryzen_ai_npu2' "$lit"; then
      any_npu2=true
      # Only NPU1-compatible if REQUIRES explicitly includes npu1.
      grep -q 'REQUIRES:.*ryzen_ai_npu1' "$lit" && return 1
    fi
  done
  $any_lit && $any_npu2 && return 0
  return 1
}

# Check if a test requires a specific compiler (chess or peano).
# Returns 0 if the lit file has a REQUIRES line naming only the other compiler.
requires_only_compiler() {
  local dir="$1"
  local required_compiler="$2"  # "chess" or "peano"
  local lit
  lit="$(find_lit_file "$dir")"
  [[ -n "$lit" ]] || return 1
  # Look for REQUIRES line with the other compiler explicitly listed.
  # Pattern: "REQUIRES: ..., chess" or "REQUIRES: chess, ..."
  local other
  [[ "$required_compiler" == "chess" ]] && other="peano" || other="chess"
  # If the lit file requires the other compiler and does NOT mention ours,
  # then this test requires only the other compiler.
  grep -qE "REQUIRES:.*\\b${other}\\b" "$lit" || return 1
  grep -qE "REQUIRES:.*\\b${required_compiler}\\b" "$lit" && return 1
  return 0
}

# Check whether a test is expected to fail.
#
# Parses the `XFAIL:` directive in the lit file's header.  Accepted forms:
#   XFAIL: *              -- expected to fail unconditionally
#   XFAIL: <feature,...>  -- expected to fail when any listed feature matches
#
# We treat the compiler name (chess/peano) as a feature.  Only the first ~40
# lines are scanned (lit itself only honors directives near the top).
#
# Returns 0 if the test is expected to fail for this compiler, 1 otherwise.
is_xfail() {
  local src_dir="$1"
  local compiler="$2"
  local lit
  lit="$(find_lit_file "$src_dir")"
  [[ -n "$lit" ]] || return 1
  local xfail_line
  xfail_line="$(head -n 40 "$lit" | grep -oE 'XFAIL:[[:space:]]*[^[:space:]].*' | head -1)" || return 1
  [[ -n "$xfail_line" ]] || return 1
  # Strip "XFAIL:" prefix and surrounding whitespace.
  local spec="${xfail_line#XFAIL:}"
  spec="${spec#"${spec%%[![:space:]]*}"}"
  spec="${spec%"${spec##*[![:space:]]}"}"
  # Strip trailing comment closers (e.g., from C-style `// XFAIL: *`).
  spec="${spec%%[[:space:]]//*}"
  # Unconditional XFAIL.
  [[ "$spec" == "*" ]] && return 0
  # Feature list -- match on compiler name.
  local IFS=', '
  local feat
  for feat in $spec; do
    [[ "$feat" == "$compiler" ]] && return 0
  done
  return 1
}

# Extract NPUDEVICE substitution from the test's .lit file.
get_npu_device() {
  local lit
  lit="$(find_lit_file "$1")"
  if [[ -z "$lit" ]]; then
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
  # %s is lit's "current test file" macro.  Compute it from src_dir so
  # `FileCheck %s` picks up CHECK: directives in the lit source itself.
  local lit_file
  lit_file="$(find_lit_file "$src_dir")"
  cmd="${cmd#*RUN: }"
  cmd="${cmd//'%S'/$src_dir}"
  [[ -n "$lit_file" ]] && cmd="${cmd//'%s'/$lit_file}"
  cmd="${cmd//'%aietools'/$AIETOOLS_DIR}"
  # %PEANO_INSTALL_DIR is set by mlir-aie's lit.site.cfg from cmake's
  # PEANO_INSTALL_DIR. Tests that compile their own kernel via Peano clang
  # (func-link-with-peano, matrix_transpose, sync_task_complete_token, etc.)
  # use it to find clang. Without this substitution, the literal %PEANO_...
  # leaks into the shell and bash interprets the leading % as a job spec.
  local _peano_install_dir
  _peano_install_dir="${PEANO_INSTALL_DIR:-${MLIR_AIE}/../llvm-aie/install}"
  cmd="${cmd//'%PEANO_INSTALL_DIR'/$_peano_install_dir}"
  # %python aiecc.py: aiecc.py has a python3 shebang and lives on PATH after
  # env activation, but `python3 aiecc.py` makes python3 search CWD only and
  # fails. Strip the python3 prefix and let the shebang handle it.
  cmd="${cmd//'%python aiecc.py'/aiecc.py}"
  cmd="${cmd//'%python '/python3 }"
  cmd="${cmd//'%python'/python3}"
  cmd="${cmd//'%xrt_flags'/-I$src_dir -I$XRT_INCLUDE -L$XRT_LIB -luuid -lxrt_coreutil}"
  cmd="${cmd//'%test_utils_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_utils}"
  cmd="${cmd//'%test_lib_flags'/-I$TEST_UTILS_INCLUDE -L$TEST_UTILS_LIB -ltest_lib}"
  cmd="${cmd//'%run_on_npu1%'/}"
  cmd="${cmd//'%run_on_npu2%'/}"
  cmd="${cmd#"${cmd%%[![:space:]]*}"}"
  cmd="${cmd%"${cmd##*[![:space:]]}"}"
  if [[ "$cmd" == clang\ * ]]; then
    cmd="/usr/bin/clang++ ${cmd#clang }"
  elif [[ "$cmd" == g++-[0-9]* ]]; then
    # Normalize versioned g++ (e.g. g++-13) to system g++ to avoid ABI
    # mismatches with XRT libs built by the system compiler.
    cmd="/usr/bin/g++ ${cmd#g++-[0-9]* }"
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

  # Determine target NPU so we can drop the wrong-target lit RUN lines.
  # %run_on_npu1% and %run_on_npu2% are conditional-execution markers that
  # apply_lit_subs strips to empty -- without filtering on the raw line
  # first, both branches fire and the npu2 step clobbers the npu1 output
  # (e.g., aie2p kernel.o overwriting aie2 kernel.o for func-link tests).
  local target_npu1=true target_npu2=false
  if requires_npu2 "$src_dir"; then
    target_npu1=false
    target_npu2=true
  fi

  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    if ! $target_npu2 && [[ "$line" == *"%run_on_npu2%"* ]]; then
      continue
    fi
    if ! $target_npu1 && [[ "$line" == *"%run_on_npu1%"* ]]; then
      continue
    fi
    local cmd
    cmd="$(apply_lit_subs "$src_dir" "$line")"
    [[ "$cmd" == *"./test.exe"* ]] && continue
    # Some newer tests build to bare `./test` rather than `./test.exe`
    # (matrix_transpose, sync_task_complete_token, ...). Skip those run
    # lines too -- bridge always rebuilds the host binary as test.exe.
    [[ "$cmd" == "./test" ]] && continue
    [[ "$cmd" == "./test "* ]] && continue
    [[ "$cmd" == *"test.py"* ]] && continue
    [[ "$cmd" == *"run_on_npu"* ]] && continue
    [[ "$cmd" == *"NPUDEVICE"* ]] && continue
    [[ "$cmd" == "cp "* ]] && continue
    echo "$cmd"
  done < "$lit_file"
}

# ---------------------------------------------------------------------------
# Test-artifact name discovery (Phase E post-validation)
# ---------------------------------------------------------------------------
# aiecc.py emits NPU instruction + control-packet binaries whose filenames
# are specified by the test's run.lit via `--npu-insts-name=` and
# `--ctrlpkt-name=` flags. Historically we hardcoded "insts.bin"; that
# breaks for control-packet tests (aie_run_seq.bin) and any future test
# that picks its own name. Parse the lit authoritatively instead.
#
# Each helper echoes the discovered filename (no path) and defaults to the
# aiecc.py default when the flag isn't present. Both are pure text
# operations, safe to call from subshells via export -f below.

# _discover_aiecc_name <src_dir> <flag_suffix> <default>
# Extracts --<flag_suffix>=<value> from the test's run.lit and echoes the value.
# Echoes <default> if no matching flag is found or the lit is missing.
_discover_aiecc_name() {
  local src_dir="$1"
  local flag="$2"
  local fallback="$3"
  local lit
  lit="$(find_lit_file "$src_dir")"
  if [[ -z "$lit" || ! -f "$lit" ]]; then
    echo "$fallback"
    return
  fi
  local val
  val="$(grep -oE -- "--${flag}=[^[:space:]]+" "$lit" | tail -1 | sed "s/^--${flag}=//")"
  if [[ -z "$val" ]]; then
    echo "$fallback"
  else
    echo "$val"
  fi
}

# _discover_instr_binary <src_dir>
# Primary NPU instruction binary (aiecc --npu-insts-name). Default: insts.bin.
_discover_instr_binary() {
  _discover_aiecc_name "$1" "npu-insts-name" "insts.bin"
}

# _discover_ctrlpkt_binary <src_dir>
# Optional control-packet blob (aiecc --ctrlpkt-name). Default: "" (not used).
# Callers should test the returned path for existence; the default is empty
# so builds without ctrlpkt generation produce no phantom --input path.
_discover_ctrlpkt_binary() {
  _discover_aiecc_name "$1" "ctrlpkt-name" ""
}

export -f _discover_aiecc_name _discover_instr_binary _discover_ctrlpkt_binary

# Strip legacy trace flags from a run command.
# Used when our trace injection is active (NO_TRACE != "true") so the test's
# built-in tracing allocates nothing (trace_size defaults to 0).
_strip_trace_flags() {
  local cmd="$1"
  if [[ "${NO_TRACE:-false}" != "true" ]]; then
    cmd="$(echo "$cmd" | sed 's/--trace_sz[[:space:]]\+[0-9]\+//g')"
    cmd="$(echo "$cmd" | sed 's/--trace_file[[:space:]]\+[^[:space:]]\+//g')"
    cmd="$(echo "$cmd" | sed 's/[[:space:]]\+/ /g' | sed 's/[[:space:]]*$//')"
  fi
  # Drop `| FileCheck ...` tails: the bridge captures test.exe output and
  # grep's the log for "PASS" itself, so piping through FileCheck just
  # swallows the stdout we need to inspect (bd_chain_repeat_on_memtile is
  # currently the only test in the suite that uses this pattern).
  cmd="$(echo "$cmd" | sed 's/[[:space:]]*|[[:space:]]*FileCheck[[:space:]][^|]*$//')"
  echo "$cmd"
}

# _run_trace_cycles_pipeline <side> <build_dir> <xclbin> <kernel> <instr> <variant>
#   side ∈ {HW, EMU}.
# Runs bridge-trace-runner against a traced xclbin on the requested side,
# then runs parse-trace.py (in-tree tools/trace_decoder backend by default;
# mlir-aie still used for slot-name lookup from MLIR) to emit both the flat
# events JSON (for trace-compare) and the cycles scalar (for the CYCLES
# column) in a single pass. Writes:
#   RESULTS_DIR/<safe>.hw-cycles/trace_{hw,emu}.<variant>.bin
#   RESULTS_DIR/<safe>.hw-cycles/trace_{hw,emu}.<variant>.events.json
#   RESULTS_DIR/<safe>.<variant>.cycles.{HW,EMU}.txt
# Best-effort: any failure logs and returns non-zero, but callers pass || true
# so the core bridge/hw verdicts are never gated on the cycle-capture side.
_run_trace_cycles_pipeline() {
    local side="$1"
    local build_dir="$2"
    local xclbin="$3"
    local kernel="$4"      # currently unused; runner auto-detects single kernel
    local instr="$5"
    local variant="$6"     # already includes compiler + any variant suffix

    local test_name
    test_name="$(basename "$(dirname "$build_dir")")"

    local runner="$EMU_ROOT/bridge-runner/build/bridge-trace-runner"
    if [[ ! -x "$runner" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): runner not built at $runner; skipping" >&2
        return 0
    fi

    # parse_trace needs the POST-LOWERING MLIR (input_with_addresses.mlir)
    # produced by aiecc.py inside its .prj scratch dir. The .prj dir is
    # named after whatever MLIR aiecc was invoked on -- aie_arch.mlir.prj
    # for most tests, aie_overlay.mlir.prj for control-packet tests, etc.
    # Discover it by searching for the unique input_with_addresses.mlir
    # rather than hardcoding the source-filename assumption.
    #
    # The trace-presence check uses a source MLIR with high-level aie.trace
    # ops. We pick any MLIR in build_dir with aie.trace (typically
    # aie_arch.mlir, written by our injector); if none, no tracing happened.
    local src_mlir=""
    for _cand in "$build_dir"/*.mlir; do
        [[ -f "$_cand" ]] || continue
        if grep -q "aie.trace " "$_cand" 2>/dev/null; then
            src_mlir="$_cand"
            break
        fi
    done
    if [[ -z "$src_mlir" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): no MLIR with aie.trace ops in $build_dir; skipping" >&2
        return 0
    fi
    local mlir_path
    mlir_path="$(find "$build_dir" -mindepth 2 -maxdepth 2 -name 'input_with_addresses.mlir' -print -quit 2>/dev/null || true)"
    if [[ -z "$mlir_path" || ! -f "$mlir_path" ]]; then
        echo "[trace-cycles:$side] $test_name ($variant): no aiecc-lowered MLIR (input_with_addresses.mlir) under $build_dir/*.prj/; cannot extract cycles" >&2
        return 0
    fi

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    mkdir -p "$work_dir"

    local bin_label cycles_label
    case "$side" in
        HW)  bin_label="trace_hw";  cycles_label="HW" ;;
        EMU) bin_label="trace_emu"; cycles_label="EMU" ;;
        *)   echo "[trace-cycles] unknown side: $side" >&2; return 1 ;;
    esac

    local trace_bin="$work_dir/${bin_label}.${variant}.bin"
    local events_json="$work_dir/${bin_label}.${variant}.events.json"
    local cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.${cycles_label}.txt"
    local runner_log="$work_dir/runner.${bin_label}.${variant}.log"
    local extract_log="$work_dir/extract.${bin_label}.${variant}.log"

    # Side-specific env: EMU routes through the XRT plugin + xdna-emu. HW runs
    # on real silicon, so nothing extra is needed.
    local -a env_prefix=()
    if [[ "$side" == "EMU" ]]; then
        env_prefix+=("XDNA_EMU=${XDNA_EMU:-debug}")
        env_prefix+=("XDNA_EMU_LOG_LEVEL=${XDNA_EMU_LOG_LEVEL:-info}")
        env_prefix+=("XRT_DEVICE_BDF=ffff:ff:1f.0")
        [[ -n "${XDNA_EMU_DIR:-}" ]] && env_prefix+=("XDNA_EMU_DIR=$XDNA_EMU_DIR")
    fi

    # Optional control-packet blob gets fed as --ctrlpkt when the test's
    # lit specifies --ctrlpkt-name. The runner uses libxdna_emu.so's
    # kernarg classifier to identify the Ctrlpkt-role arg_idx and bind
    # the blob there; if the classifier is unavailable the flag falls
    # back to --input semantics (legacy positional).
    local -a extra_args=()
    local _ctrlpkt_name
    _ctrlpkt_name="$(_discover_ctrlpkt_binary "$TEST_SRC/$test_name")"
    if [[ -n "$_ctrlpkt_name" ]] && [[ -f "$build_dir/$_ctrlpkt_name" ]]; then
        extra_args+=(--ctrlpkt "$build_dir/$_ctrlpkt_name")
    fi

    if ! env "${env_prefix[@]}" "$runner" \
        --xclbin "$xclbin" \
        --instr "$instr" \
        --trace-out "$trace_bin" \
        --trace-size 1048576 \
        "${extra_args[@]}" \
        2>"$runner_log"; then
        echo "[trace-cycles:$side] $test_name ($variant): runner failed; see $runner_log" >&2
        return 1
    fi

    # Single-pass decode: parse-trace.py wraps mlir-aie's parser and emits
    # both the flat events JSON (for trace-compare) and the cycles scalar
    # (for the CYCLES column). One invocation, derived from shared state,
    # guarantees the two outputs never disagree.
    if ! PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
        python3 "$EMU_ROOT/tools/parse-trace.py" \
        --trace-bin "$trace_bin" \
        --xclbin-mlir "$mlir_path" \
        --out-events "$events_json" \
        --out-cycles "$cycles_txt" \
        2>"$extract_log"; then
        echo "[trace-cycles:$side] $test_name ($variant): parse-trace failed; see $extract_log" >&2
        return 1
    fi

    echo "[trace-cycles:$side] $test_name ($variant): cycles=$(cat "$cycles_txt")" >&2
    return 0
}

# _run_trace_compare <test_name> <variant>
# Runs trace-compare on the HW and EMU events JSONs produced by
# _run_trace_cycles_pipeline (via parse-trace.py). Writes the report to
# RESULTS_DIR/<safe>.<variant>.cycles.compare.txt.
# Returns 0 on success, 1 if either events JSON is missing, 2 on compare error.
_run_trace_compare() {
    local test_name="$1"
    local variant="$2"

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    local hw_events="$work_dir/trace_hw.${variant}.events.json"
    local emu_events="$work_dir/trace_emu.${variant}.events.json"
    local report="$RESULTS_DIR/${safe}.${variant}.cycles.compare.txt"

    if [[ ! -f "$hw_events" || ! -f "$emu_events" ]]; then
        return 1
    fi

    local rust_bin="$EMU_ROOT/target/release/trace-compare"
    if [[ ! -x "$rust_bin" ]]; then
        echo "[trace-compare] trace-compare binary not built at $rust_bin" >&2
        return 2
    fi

    # --remap-columns normalizes each side's physical column numbers to
    # 0-indexed logical columns before pairing. HW schedulers pick
    # start_col=1 (or similar) whereas the emulator always places kernels
    # at col=0; without remapping, identical events on row=2 appear on
    # tile (1,2) in HW vs (0,2) in EMU and fall out as count mismatches.
    if ! "$rust_bin" \
        --hw "$hw_events" \
        --emu "$emu_events" \
        --remap-columns \
        --stalls \
        --extended \
        -o "$report" 2>/dev/null; then
        return 2
    fi
    return 0
}

# _classify_cycle_diff <test_name> <variant>
# Reads the HW/EMU cycles files and the compare report, writes one of
# MATCH(<ratio>) / DRIFT(ratio=<r>,diverge=<n>) / EMPTY / NO_DATA /
# EMU_TRACE_BUG / HW_TRACE_BUG / COMPARE-ERR to
# RESULTS_DIR/<safe>.<variant>.cycle.result. Always returns 0 unless
# called with bad args; the classification itself is advisory.
_classify_cycle_diff() {
    local test_name="$1"
    local variant="$2"

    local safe
    safe="$(sanitize_name "$test_name")"
    local work_dir="$RESULTS_DIR/${safe}.hw-cycles"
    local hw_bin="$work_dir/trace_hw.${variant}.bin"
    local emu_bin="$work_dir/trace_emu.${variant}.bin"
    local hw_cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.HW.txt"
    local emu_cycles_txt="$RESULTS_DIR/${safe}.${variant}.cycles.EMU.txt"
    local report="$RESULTS_DIR/${safe}.${variant}.cycles.compare.txt"
    local out_file="$RESULTS_DIR/${safe}.${variant}.cycle.result"

    local hw_have_bin=false; [[ -f "$hw_bin"  ]] && hw_have_bin=true
    local emu_have_bin=false; [[ -f "$emu_bin" ]] && emu_have_bin=true

    if ! $hw_have_bin && ! $emu_have_bin; then
        echo "NO_DATA" > "$out_file"
        return 0
    fi

    local hw_cycles=0 emu_cycles=0
    [[ -f "$hw_cycles_txt"  ]] && hw_cycles="$(tr -d '[:space:]' < "$hw_cycles_txt")"
    [[ -f "$emu_cycles_txt" ]] && emu_cycles="$(tr -d '[:space:]' < "$emu_cycles_txt")"
    [[ -z "$hw_cycles"  ]] && hw_cycles=0
    [[ -z "$emu_cycles" ]] && emu_cycles=0

    # Asymmetry cases: one side traced, the other didn't.
    if $hw_have_bin && ! $emu_have_bin; then
        echo "EMU_TRACE_BUG" > "$out_file"
        return 0
    fi
    if $emu_have_bin && ! $hw_have_bin; then
        echo "HW_TRACE_BUG" > "$out_file"
        return 0
    fi
    if [[ "$hw_cycles" -gt 0 && "$emu_cycles" -eq 0 ]]; then
        echo "EMU_TRACE_BUG" > "$out_file"
        return 0
    fi
    if [[ "$emu_cycles" -gt 0 && "$hw_cycles" -eq 0 ]]; then
        echo "HW_TRACE_BUG" > "$out_file"
        return 0
    fi

    # Both sides captured zero events. Two legitimate shapes:
    #   NO_CORE  -- traced MLIR has no aie.core (DMA-only passthrough like
    #               column_specific). Core-side trace events cannot fire
    #               without a core, so HW and EMU both produce zeros.
    #   EMPTY    -- has core, but default event set did not fire (Phase B
    #               Limitation 1 -- scalar kernels). Broadening the event
    #               set would turn these into usable data points.
    if [[ "$hw_cycles" -eq 0 && "$emu_cycles" -eq 0 ]]; then
        local traced_mlir="$BUILD_BASE/$test_name/aie-hw-cycles-traced.mlir"
        if [[ -f "$traced_mlir" ]] && ! grep -q 'aie\.core' "$traced_mlir"; then
            echo "NO_CORE" > "$out_file"
        else
            echo "EMPTY" > "$out_file"
        fi
        return 0
    fi

    # Both non-zero: parse the compare report for divergence counts.
    if [[ ! -f "$report" ]]; then
        echo "COMPARE-ERR" > "$out_file"
        return 0
    fi

    local edge_line level_line
    edge_line="$(grep -E '^Edge event types:'  "$report" || true)"
    level_line="$(grep -E '^Level event types:' "$report" || true)"
    if [[ -z "$edge_line" || -z "$level_line" ]]; then
        echo "COMPARE-ERR" > "$out_file"
        return 0
    fi

    # Summary line format: "Edge event types:    N clean, M diverged, K count mismatch"
    local e_div e_cmm l_div l_cmm
    e_div="$(echo "$edge_line"  | grep -oE '[0-9]+ diverged'       | awk '{print $1}')"
    e_cmm="$(echo "$edge_line"  | grep -oE '[0-9]+ count mismatch' | awk '{print $1}')"
    l_div="$(echo "$level_line" | grep -oE '[0-9]+ diverged'       | awk '{print $1}')"
    l_cmm="$(echo "$level_line" | grep -oE '[0-9]+ count mismatch' | awk '{print $1}')"
    [[ -z "$e_div" ]] && e_div=0
    [[ -z "$e_cmm" ]] && e_cmm=0
    [[ -z "$l_div" ]] && l_div=0
    [[ -z "$l_cmm" ]] && l_cmm=0
    local total_diverge=$(( e_div + e_cmm + l_div + l_cmm ))

    local lower upper
    _lookup_drift_bounds "$test_name" lower upper

    local ratio
    ratio="$(awk -v e="$emu_cycles" -v h="$hw_cycles" \
        'BEGIN{ if(h==0){print "0.00"} else{printf "%.2f", e/h} }')"
    local in_bounds
    in_bounds="$(awk -v r="$ratio" -v l="$lower" -v u="$upper" \
        'BEGIN{ if(r>=l && r<=u) print 1; else print 0 }')"

    if [[ "$total_diverge" -eq 0 && "$in_bounds" == "1" ]]; then
        echo "MATCH($ratio)" > "$out_file"
    else
        echo "DRIFT(ratio=$ratio,diverge=$total_diverge)" > "$out_file"
    fi
    return 0
}

# Classify the mode-2 comparison block of a PC-anchored report into a
# single PASS / FAIL / SKIP / ERROR token.
#
#   $1 -- result file (single-token output)
#   $2 -- summary file (one-line human-readable detail)
#   $3 -- PC-anchored report.txt produced by trace-compare
#
# Tokens:
#   PASS   -- one or more tile pairs and zero FAIL tiles
#   FAIL   -- one or more tile pairs with at least one FAIL tile
#   SKIP   -- mode-2 block present but no per-tile pairs (HW-only or EMU-only,
#             or "no tiles common"), or the block was missing entirely
#   ERROR  -- comparator could not load events JSON
#
# Always returns 0; the classification itself is advisory.
_classify_mode2() {
    local result_file="$1"
    local summary_file="$2"
    local report="$3"

    if [[ ! -f "$report" ]]; then
        echo "ERROR" > "$result_file"
        echo "ERROR: report file missing ($report)" > "$summary_file"
        return 0
    fi

    # Extract the Mode-2 comparison block (from the heading to EOF or the
    # next top-level section heading).
    local block
    block="$(awk '
        /^Mode-2 comparison:/ { in_block = 1; next }
        in_block && /^[A-Z]/   { in_block = 0 }
        in_block               { print }
    ' "$report")"

    if [[ -z "$block" ]]; then
        echo "SKIP" > "$result_file"
        echo "SKIP: no mode-2 baseline captured" > "$summary_file"
        return 0
    fi

    # ERROR sentinels written by compare.rs.
    if echo "$block" | grep -qE '^[[:space:]]+ERROR loading events:'; then
        local errline
        errline="$(echo "$block" | grep -m1 -oE 'ERROR loading events:.*')"
        echo "ERROR" > "$result_file"
        echo "${errline}" > "$summary_file"
        return 0
    fi

    # grep -c returns exit 1 when there are zero matches; under `set -e`
    # this would abort the parent script even though zero is the answer
    # we want, so we trap the rc with `|| true` and split declaration
    # from assignment so `local` doesn't mask the substitution failure.
    local pass_count fail_count hw_only_count emu_only_count
    pass_count="$(echo "$block" | grep -cE '^[[:space:]]+tile pt=.*\[PASS\]' || true)"
    fail_count="$(echo "$block" | grep -cE '^[[:space:]]+tile pt=.*\[FAIL\]' || true)"
    hw_only_count="$(echo "$block" | grep -cE 'HW only -- no EMU events' || true)"
    emu_only_count="$(echo "$block" | grep -cE 'EMU only -- no HW events' || true)"

    local total=$((pass_count + fail_count))
    if [[ $total -eq 0 ]]; then
        # Either "no tiles common" or every tile is one-sided. The
        # comparator emitted "SKIP (EMU events JSON ... not present)" in
        # the lone-side case; treat both as SKIP.
        echo "SKIP" > "$result_file"
        echo "SKIP: 0 tile pairs (hw_only=${hw_only_count} emu_only=${emu_only_count})" \
            > "$summary_file"
        return 0
    fi

    if [[ $fail_count -eq 0 ]]; then
        echo "PASS" > "$result_file"
        echo "PASS: ${pass_count}/${total} tiles" > "$summary_file"
    else
        echo "FAIL" > "$result_file"
        echo "FAIL: ${fail_count}/${total} tiles diverged" > "$summary_file"
    fi
    return 0
}

# Derive a variant name from a run command's xclbin filename.
# e.g., "aie2_cascade.xclbin" -> "cascade", "aie.xclbin" -> ""
# Returns empty string for the standard "aie.xclbin" case.
_variant_from_cmd() {
  local cmd="$1"
  local xclbin
  xclbin="$(echo "$cmd" | grep -oP '(?<=-x\s)\S+\.xclbin' || true)"
  if [[ -z "$xclbin" ]]; then
    # Try positional xclbin (e.g., "./test.exe aie.xclbin")
    xclbin="$(echo "$cmd" | grep -oP '\S+\.xclbin' || true)"
  fi
  # Standard single-variant names -> empty variant
  case "$xclbin" in
    aie.xclbin|"") echo ""; return ;;
  esac
  # Strip directory prefix, .xclbin suffix, and common "aie2_" prefix.
  local base="${xclbin##*/}"
  base="${base%.xclbin}"
  base="${base#aie2_}"
  base="${base#aie_}"
  echo "$base"
}

# List all run variants for a test.  Outputs one line per variant:
#   <variant_name>
# For single-variant tests (the common case), outputs a single empty line.
# For multi-variant tests (e.g., matrix_multiplication_using_cascade), outputs
# one line per variant (e.g., "plain", "buffer", "cascade").
get_run_variants() {
  local src_dir="$1"
  local lit_file
  lit_file="$(find_lit_file "$src_dir")"
  if [[ -z "$lit_file" ]]; then
    echo ""
    return
  fi

  local variants=()
  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    # Host-side run line: test.exe (cpp tests) or test.py (python tests).
    if [[ "$line" == *"./test.exe"* || "$line" == *"test.py"* ]] \
       && [[ "$line" == *"npu1"* || "$line" != *"npu2"* ]]; then
      local cmd
      cmd="$(apply_lit_subs "$src_dir" "$line")"
      cmd="${cmd#"${cmd%%[![:space:]]*}"}"
      local v
      v="$(_variant_from_cmd "$cmd")"
      variants+=("$v")
    fi
  done < "$lit_file"

  if [[ ${#variants[@]} -le 1 ]]; then
    # Zero or one run command: single-variant test (empty variant name).
    echo ""
    return
  fi

  # Multiple run commands: multi-variant test.  Output variant names.
  printf '%s\n' "${variants[@]}"
}

# Extract the run command for a specific variant.
# $1 = src_dir, $2 = variant name (empty string = first/only npu1 command)
get_variant_run_cmd() {
  local src_dir="$1"
  local target_variant="${2:-}"
  local lit_file
  lit_file="$(find_lit_file "$src_dir")"
  [[ -n "$lit_file" ]] || return 1

  # Collect all npu1-matching run commands (test.exe or test.py).
  local cmds=()
  while IFS= read -r line; do
    [[ "$line" == *"RUN:"* ]] || continue
    if [[ "$line" == *"./test.exe"* || "$line" == *"test.py"* ]] \
       && [[ "$line" == *"npu1"* || "$line" != *"npu2"* ]]; then
      local cmd
      cmd="$(apply_lit_subs "$src_dir" "$line")"
      cmd="${cmd#"${cmd%%[![:space:]]*}"}"
      cmds+=("$cmd")
    fi
  done < "$lit_file"

  # Empty variant with 0-1 commands: return the first (or fallback).
  if [[ -z "$target_variant" ]]; then
    if [[ ${#cmds[@]} -ge 1 ]]; then
      echo "$(_strip_trace_flags "${cmds[0]}")"
      return 0
    fi
    echo "./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin"
    return 0
  fi

  # Non-empty variant: match by derived variant name.
  for cmd in "${cmds[@]}"; do
    local v
    v="$(_variant_from_cmd "$cmd")"
    if [[ "$v" == "$target_variant" ]]; then
      echo "$(_strip_trace_flags "$cmd")"
      return 0
    fi
  done

  return 1
}

# Legacy wrapper: returns the first run command (backward compat for sweep etc.).
get_run_cmd() {
  local src_dir="$1"
  get_variant_run_cmd "$src_dir" ""
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
    # Add --unified if not present. compileCoresUnified merges all cores into a
    # single xchesscc invocation, paying the ~21 s Synopsys pipeline startup
    # once per device instead of once per core. Multi-core tests
    # (cascade matmul, etc.) drop from minutes to ~30 s.
    if [[ "$cmd" != *"--unified"* ]] && [[ "$cmd" != *"--no-unified"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --unified}"
    fi
    # Add --aiesim if requested (requires --xbridge, which Chess builds have)
    if [[ "${RUN_AIESIM:-false}" == "true" ]] && [[ "$cmd" != *"--aiesim"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --aiesim}"
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
    # Ensure --no-xchesscc and --no-xbridge are present (the new C++ aiecc
    # defaults to xbridge when Chess is available; we must opt out explicitly).
    if [[ "$cmd" != *"--no-xchesscc"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --no-xchesscc}"
    fi
    if [[ "$cmd" != *"--no-xbridge"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --no-xbridge}"
    fi
    echo "$cmd"
    return
  fi

  echo "$cmd"
}

# Export all helpers for xargs subshells.
export -f find_lit_file is_standard_test is_python_host_test requires_npu2 requires_only_compiler is_xfail
export -f get_npu_device apply_lit_subs
export -f extract_build_commands get_run_cmd get_run_variants get_variant_run_cmd
export -f _strip_trace_flags _variant_from_cmd _run_trace_cycles_pipeline _run_trace_compare _classify_cycle_diff
export -f sanitize_name wait_npu_idle
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

    if [[ -n "$FILTER" ]] && ! echo "$name" | grep -qE "$FILTER"; then
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

    if [[ -n "$FILTER" ]] && ! echo "$name" | grep -qE "$FILTER"; then
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
  local lit_file
  lit_file="$(find_lit_file "$src_dir")"

  mkdir -p "$build_dir"
  : > "$log_file"

  if [[ -z "$lit_file" ]]; then
    echo "FAIL" > "$result_file"
    echo "No .lit file found in $src_dir" >> "$log_file"
    echo "  COMPILE $name ($compiler): FAIL (no .lit file)"
    return 0
  fi

  # Skip if the lit file requires only the other compiler.
  if requires_only_compiler "$src_dir" "$compiler"; then
    local other
    [[ "$compiler" == "chess" ]] && other="peano" || other="chess"
    echo "SKIP_COMPILER" > "$result_file"
    echo "Test requires $other (REQUIRES line in $(basename "$lit_file"))" >> "$log_file"
    echo "  COMPILE $name ($compiler): SKIP ($other-only)"
    return 0
  fi

  # Prepare architecture MLIR (NPUDEVICE substitution).
  # Only copy/modify if the cached copy is missing, stale, or --compile was passed,
  # so that timestamps are preserved and the xclbin cache check below works.
  local npu_dev
  npu_dev="$(get_npu_device "$src_dir")"

  # Determine which source MLIR to use.
  local src_mlir=""
  if [[ -n "$HW_CYCLES_TRACED_MLIR" ]] && [[ -f "$HW_CYCLES_TRACED_MLIR" ]]; then
    src_mlir="$HW_CYCLES_TRACED_MLIR"
  elif [[ "$TRACE_OK" == "true" ]] && [[ -f "$TRACED_DIR/aie_traced.mlir" ]]; then
    src_mlir="$TRACED_DIR/aie_traced.mlir"
  elif [[ -f "$src_dir/aie.mlir" ]]; then
    src_mlir="$src_dir/aie.mlir"
  fi

  if [[ -n "$src_mlir" ]]; then
    local need_copy=false
    if [[ "$FORCE_COMPILE" == "true" ]]; then
      need_copy=true
    elif [[ ! -f "$build_dir/aie_arch.mlir" ]]; then
      need_copy=true
    elif [[ "$src_mlir" -nt "$build_dir/aie_arch.mlir" ]]; then
      need_copy=true
    fi

    if $need_copy; then
      cp "$src_mlir" "$build_dir/aie_arch.mlir"
      # Traced MLIR already has the device substituted by trace-inject.py.
      if [[ "$src_mlir" == "$src_dir/aie.mlir" ]]; then
        sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch.mlir"
      fi
    fi
  fi

  # For multi-step builds with tracing active, secondary aiecc commands
  # (those that generate auxiliary artifacts such as control packets and
  # transaction binaries rather than the xclbin) should use the original
  # untraced MLIR so that trace instrumentation does not contaminate them.
  # Count aiecc steps first to determine whether this applies.
  local aiecc_count=0
  while IFS= read -r _cmd; do
    [[ "$_cmd" == *aiecc.py* ]] && aiecc_count=$((aiecc_count + 1)) || true
  done < <(extract_build_commands "$lit_file" "$src_dir")

  # When tracing is active, multiple aiecc steps exist, and the original
  # MLIR is available in the source directory, copy it (with NPUDEVICE
  # substituted) as aie_arch_orig.mlir for use by secondary aiecc steps.
  local orig_mlir_available=false
  if [[ "$TRACE_OK" == "true" ]] && [[ $aiecc_count -gt 1 ]] \
      && [[ -f "$src_dir/aie.mlir" ]]; then
    local need_orig_copy=false
    if [[ "$FORCE_COMPILE" == "true" ]]; then
      need_orig_copy=true
    elif [[ ! -f "$build_dir/aie_arch_orig.mlir" ]]; then
      need_orig_copy=true
    elif [[ "$src_dir/aie.mlir" -nt "$build_dir/aie_arch_orig.mlir" ]]; then
      need_orig_copy=true
    fi
    if $need_orig_copy; then
      cp "$src_dir/aie.mlir" "$build_dir/aie_arch_orig.mlir"
      sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch_orig.mlir"
    fi
    orig_mlir_available=true
  fi

  # Check cache -- must come AFTER traced MLIR substitution so that
  # the cache is invalidated when the MLIR has changed (e.g., tracing
  # was enabled on a previously-compiled test).
  local have_xclbin=false
  if [[ -f "$build_dir/aie.xclbin" ]] || ls "$build_dir"/*.xclbin &>/dev/null; then
    have_xclbin=true
  fi

  # Invalidate cache if MLIR changed since last compile.
  if $have_xclbin && [[ -f "$build_dir/aie_arch.mlir" ]] \
      && [[ -f "$build_dir/aie.xclbin" ]]; then
    if [[ "$build_dir/aie_arch.mlir" -nt "$build_dir/aie.xclbin" ]]; then
      have_xclbin=false
    fi
  fi

  # Invalidate cache if --aiesim requested but sim/ artifacts are missing.
  if $have_xclbin && [[ "${RUN_AIESIM:-false}" == "true" ]] \
      && [[ "$compiler" == "chess" ]]; then
    local prj_check
    prj_check="$(find "$build_dir" -maxdepth 1 -name '*.prj' -type d -print -quit 2>/dev/null || true)"
    if [[ -z "$prj_check" ]] || [[ ! -d "$prj_check/sim" ]]; then
      have_xclbin=false
    fi
  fi

  if $have_xclbin && [[ "$FORCE_COMPILE" != "true" ]]; then
    echo "  COMPILE $name ($compiler): cached"
    echo "OK" > "$result_file"
    return 0
  fi

  # Run all build steps in order: aie-opt, aiecc (one or more), aie-translate.
  # The first aiecc command (which produces the xclbin) uses the traced MLIR
  # (aie_arch.mlir).  Subsequent aiecc commands use the original untraced MLIR
  # (aie_arch_orig.mlir, if available) so that trace instrumentation does not
  # contaminate auxiliary artifacts such as control packets and txn binaries.
  local first_aiecc_done=false
  local failed=false
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    # Skip host compilation -- handled separately
    [[ "$cmd" == *clang*test.cpp* ]] && continue
    [[ "$cmd" == *g++*test.cpp* ]] && continue

    # When HW_CYCLES is active and this RUN line is the .py MLIR generator
    # (column_specific-style tests: `python3 .../aie*.py > ./foo.mlir`),
    # substitute `cp <traced_mlir> ./foo.mlir` so the traced MLIR our
    # injector produced ends up where aiecc expects it. Without this swap
    # the .py would regenerate the MLIR clean, clobbering our injection.
    if [[ -n "$HW_CYCLES_TRACED_MLIR" ]] \
       && [[ "$cmd" =~ ^python3[[:space:]]+[^[:space:]]+/aie[^[:space:]]*\.py[[:space:]]+\>[[:space:]]+([^[:space:]]+\.mlir)$ ]]; then
      local _py_out="${BASH_REMATCH[1]}"
      cmd="cp $HW_CYCLES_TRACED_MLIR $_py_out"
    fi

    # Fix MLIR path references for aiecc commands.
    #
    # The rewrite only applies when we actually produced an aie_arch.mlir
    # (i.e., src_mlir was non-empty: either trace-prepare output, HW_CYCLES
    # traced MLIR, or a source aie.mlir to copy). Tests that generate
    # ./aie.mlir locally from a python script and don't have a source
    # aie.mlir (e.g. bd_chain_repeat_on_memtile) need the aiecc command to
    # reference that generated file as-is.
    if [[ "$cmd" == *aiecc.py* ]] && [[ -n "$src_mlir" ]]; then
      cmd="${cmd//$src_dir\/aie.mlir/./aie_arch.mlir}"
      cmd="${cmd//\.\/aie.mlir/./aie_arch.mlir}"
      # Secondary aiecc steps (post-xclbin) use original MLIR when available.
      if $first_aiecc_done && $orig_mlir_available; then
        cmd="${cmd//aie_arch.mlir/aie_arch_orig.mlir}"
      fi
      first_aiecc_done=true
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
    local fail_tag="FAIL"
    local label="FAIL"
    if [[ -n "$HW_CYCLES_TRACED_MLIR" ]] && [[ "$src_mlir" == "$HW_CYCLES_TRACED_MLIR" ]]; then
      fail_tag="FAIL_TRACED"
      label="FAIL(traced)"
    fi
    echo "$fail_tag" > "$result_file"
    echo "  COMPILE $name ($compiler): $label"
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
  local lit_file
  lit_file="$(find_lit_file "$src_dir")"

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

  if [[ "$NO_TRACE" != "true" ]] && ! is_trace_quarantined "$name" \
       && ! is_python_host_test "$src_dir"; then
    local trace_log="$RESULTS_DIR/${safe}.trace-prepare.log"
    # --shim-sweep-events all (#372 / stage 1), --memtile-sweep-events all
    # (#373 / stage 2), and --memmod-sweep-events all (#374 / stage 3) all
    # activate inject-time trace decls on the corresponding tile types.
    # Phase 5b's lockstep sweep then rotates events on every type that has
    # decls, giving full-array coverage. Pre-stage-1 behavior (core-only
    # trace) is preserved by omitting these flags.
    # Optional override: set XDNA_TRACE_MODE={event_time,event_pc,inst_exec}
    # to override mlir-trace-inject's default (event_pc / mode 1). Used by
    # #355a cycle-delta calibration which needs event_time (mode 0)
    # so tools/dma-fill-measure.py can extract per-event cycle counts.
    local _trace_mode_args=()
    if [[ -n "${XDNA_TRACE_MODE:-}" ]]; then
      _trace_mode_args=(--trace-mode "$XDNA_TRACE_MODE")
    fi
    # Optional override: explicit memtile event slot list, replacing the
    # default PORT_RUNNING_* sweep with caller-chosen names. Used by #355a
    # to get DMA_S2MM_SEL{0,1}_*_TASK boundary events on memtile so we
    # can attribute per-stage propagation cycles.
    local _memtile_args=("--memtile-sweep-events" "all")
    if [[ -n "${XDNA_TRACE_MEMTILE_EVENTS:-}" ]]; then
      _memtile_args=("--memtile-sweep-events" "$XDNA_TRACE_MEMTILE_EVENTS")
    fi
    # Optional override: memtile DMA Event Channel Selection register
    # programming (offset 0xA06A0). Without this, memtile DMA SEL events
    # only fire for physical channel 0 because every SEL slot points
    # there at reset. Format: 'S2MM_SEL0:N,S2MM_SEL1:N,MM2S_SEL0:N,MM2S_SEL1:N'
    # (any subset; unset slots default to channel 0). Used by #355a
    # multi-channel memtile attribution.
    local _memtile_sel_args=()
    if [[ -n "${XDNA_TRACE_MEMTILE_SEL_CHANNELS:-}" ]]; then
      _memtile_sel_args=("--memtile-sel-channels" "$XDNA_TRACE_MEMTILE_SEL_CHANNELS")
    fi
    # Optional override: explicit memmod (compute-tile memory module) event
    # slot list. Default 'all' uses the upstream defaults (DMA START_TASK +
    # CONFLICT_DM_BANK + EDGE). Used by #355a to swap DMA_S2MM_0_FINISHED_BD
    # in for stage-4 anchoring (memtile->compute chain → data in compute LM).
    local _memmod_args=("--memmod-sweep-events" "all")
    if [[ -n "${XDNA_TRACE_MEMMOD_EVENTS:-}" ]]; then
      _memmod_args=("--memmod-sweep-events" "$XDNA_TRACE_MEMMOD_EVENTS")
    fi
    if nice -n 19 python3 "$EMU_ROOT/tools/trace-prepare.py" "$src_dir" \
        -o "$traced_dir" \
        --shim-sweep-events all \
        "${_memtile_args[@]}" \
        "${_memtile_sel_args[@]}" \
        "${_memmod_args[@]}" \
        "${_trace_mode_args[@]}" \
        > "$trace_log" 2>&1; then
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

  # ---- Phase B: HW cycle capture trace injection (separate from --trace) ----
  # When WITH_HW_CYCLES=true, inject declarative aie.trace ops into the
  # source MLIR via mlir-trace-inject.py. The injected output gets fed to
  # aiecc.py in compile_one_compiler via HW_CYCLES_TRACED_MLIR. Independent
  # of NO_TRACE / trace-prepare.py -- the two paths produce different artifacts
  # and target different downstream consumers.
  #
  # The injector requires a valid device keyword (npu1, npu1_1col, etc.);
  # mlir-aie test sources use the literal "NPUDEVICE" placeholder which the
  # test's lit file normally resolves via sed. We do the substitution into a
  # temporary pre-substituted MLIR before handing it to the injector.
  local hw_cycles_traced_mlir=""
  if [[ "$WITH_HW_CYCLES" == "true" ]] && is_trace_incompat "$name"; then
      echo "  HW-CYCLES INJECT $name: SKIP (in trace-incompat-tests.txt)"
  fi
  if [[ "$WITH_HW_CYCLES" == "true" ]] && ! is_trace_incompat "$name"; then
      local hw_cycles_inject_log="$RESULTS_DIR/${safe}.hw-cycles-inject.log"
      local hw_cycles_target="$build_dir/aie-hw-cycles-traced.mlir"
      local hw_cycles_src="$build_dir/aie-hw-cycles-src.mlir"
      local hw_cycles_config="$build_dir/aie-hw-cycles-trace-config.json"
      mkdir -p "$build_dir"
      : > "$hw_cycles_inject_log"
      # Resolve the source MLIR. Two cases:
      #   - Static:  $src_dir/aie.mlir exists verbatim. Apply the test's
      #              NPUDEVICE sed substitution, write to hw_cycles_src.
      #   - Python:  No aie.mlir, but an aie*.py generator lives in the
      #              source dir (column_specific-style tests). Run it
      #              with PYTHONPATH to materialize the MLIR, then
      #              inject on that. The .py is expected to emit a
      #              complete device-bound MLIR -- no NPUDEVICE sub
      #              needed.
      local have_src=false
      if [[ -f "$src_dir/aie.mlir" ]]; then
          local hw_cycles_dev
          hw_cycles_dev="$(get_npu_device "$src_dir")"
          sed "s/NPUDEVICE/${hw_cycles_dev}/g" "$src_dir/aie.mlir" > "$hw_cycles_src"
          have_src=true
      else
          local py_gen=""
          for _cand in "$src_dir"/aie*.py; do
              [[ -f "$_cand" ]] || continue
              grep -q 'RUN:' "$_cand" 2>/dev/null || continue
              py_gen="$_cand"; break
          done
          if [[ -n "$py_gen" ]]; then
              echo "  HW-CYCLES INJECT $name: generating MLIR from $(basename "$py_gen")" >> "$hw_cycles_inject_log"
              if PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
                  python3 "$py_gen" > "$hw_cycles_src" 2>> "$hw_cycles_inject_log"; then
                  have_src=true
              else
                  echo "  HW-CYCLES INJECT $name: FAIL (running $(basename "$py_gen") -- see $hw_cycles_inject_log)"
              fi
          fi
      fi
      if $have_src; then
          PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
              python3 "$EMU_ROOT/tools/mlir-trace-inject.py" \
              --input "$hw_cycles_src" \
              --out "$hw_cycles_target" \
              --buffer-size 8192 \
              --trace-config-out "$hw_cycles_config" \
              --config-test-name "$name" \
              --config-src-mlir "$hw_cycles_src" \
              >> "$hw_cycles_inject_log" 2>&1
          case $? in
              0) hw_cycles_traced_mlir="$hw_cycles_target"
                 echo "  HW-CYCLES INJECT $name: OK" ;;
              2) echo "  HW-CYCLES INJECT $name: SKIP (already traced)" ;;
              *) echo "  HW-CYCLES INJECT $name: FAIL (see $hw_cycles_inject_log)" ;;
          esac
      fi
  fi
  export HW_CYCLES_TRACED_MLIR="$hw_cycles_traced_mlir"

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
      # Some newer tests use explicit `xrt::device device = xrt::device(device_index);`
      # instead of `auto`; cover both forms or device_index leaks unresolved.
      sed \
        -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
        -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
        -e 's/xrt::device device = xrt::device(device_index);/xrt::device device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
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
    # Force build output to test.exe -- some newer tests use `-o test`
    # (matrix_transpose, sync_task_complete_token, ...) but run_one_hardware
    # only invokes ./test.exe. Rewriting here is simpler than handling both
    # filenames downstream.
    clang_cmd="${clang_cmd//-o test /-o test.exe }"
    clang_cmd="${clang_cmd%-o test}"
    if [[ "$clang_cmd" != *"-o test.exe"* ]]; then
      clang_cmd+=" -o test.exe"
    fi
    if ! ( cd "$build_dir" && bash -c "$clang_cmd" ) >> "$log_file" 2>&1; then
      echo "  COMPILE $name: FAIL (test.exe)"
      return 0
    fi
  elif [[ -f "$build_dir/test.cpp" ]]; then
    if ! /usr/bin/clang++ "$build_dir/test.cpp" -o "$build_dir/test.exe" \
        -std=c++17 -Wall \
        -I"$src_dir" -I"$XRT_INCLUDE" -L"$XRT_LIB" \
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

# Maximum concurrent NPU hardware contexts.  Default serial (1) to avoid
# TDR cascade -- a single bad test can poison all concurrent tests.
# Use --parallel-hw for speed (5 jobs, leaving headroom from NPU1's 6
# context limit).
NPU_HW_JOBS="${NPU_HW_JOBS:-1}"
NPU_HW_JOBS_PARALLEL="${NPU_HW_JOBS_PARALLEL:-5}"
export NPU_HW_JOBS

run_one_hardware() {
  local name="$1"
  local bdf="$2"
  local compiler="$3"
  local variant="${4:-}"   # Empty for single-variant tests.
  local safe
  safe="$(sanitize_name "$name")"
  local build_dir="$BUILD_BASE/$name/$compiler"
  local test_exe="$BUILD_BASE/$name/test.exe"
  local src_dir="$TEST_SRC/$name"

  # Variant suffix for result/log filenames (empty for single-variant tests).
  local vsuffix=""
  [[ -n "$variant" ]] && vsuffix=".${variant}"
  local log_file="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.log"
  local result_file="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.result"

  # Python-driven host tests have no test.exe; the run command invokes
  # `python3 .../test.py ...` directly. Skip the test.exe symlink+check.
  local _py_host=false
  is_python_host_test "$src_dir" && _py_host=true

  if ! $_py_host; then
    # Symlink shared test.exe into per-compiler build dir. Always relink so
    # stale per-compiler test.exes (left over from older builds when test.exe
    # was compiled per compiler) get replaced.
    if [[ -f "$test_exe" ]]; then
      rm -f "$build_dir/test.exe"
      ln -sf "$test_exe" "$build_dir/test.exe"
    fi

    if [[ ! -f "$build_dir/test.exe" ]]; then
      echo "SKIP" > "$result_file"
      return
    fi
  fi

  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP" > "$result_file"
    return
  fi

  local run_cmd
  run_cmd="$(get_variant_run_cmd "$src_dir" "$variant")"

  local trace_out_dir="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw"
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

  local result
  if [[ $tdr_new -gt 0 ]]; then
    result="TDR"
    echo "TDR detected: $tdr_new new aie2_tdr_work events (uptime ${t_start}s)" >> "$log_file"
  elif [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    result="PASS"
  elif [[ $rc -eq 124 ]]; then
    result="TIMEOUT"
  else
    result="FAIL"
  fi

  # Honor XFAIL annotations: a failing xfail is XFAIL, a passing xfail is XPASS.
  if is_xfail "$src_dir" "$compiler"; then
    case "$result" in
      PASS)          result="XPASS" ;;
      FAIL|TIMEOUT)  result="XFAIL" ;;
    esac
  fi

  echo "$result" > "$result_file"

  # Phase B: capture HW cycle count via trace pipeline (best-effort).
  if [[ "$WITH_HW_CYCLES" == "true" && "$result" == "PASS" ]]; then
      local _hw_xclbin
      _hw_xclbin="$(find "$build_dir" -maxdepth 1 -name '*.xclbin' -print -quit 2>/dev/null || true)"
      # Discover the test's instruction binary name from its run.lit file
      # (--npu-insts-name=<foo>). Falls back to the aiecc default insts.bin
      # when the test relies on the default. This avoids hardcoding the
      # set of known filenames.
      local _src_dir="$TEST_SRC/$name"
      local _hw_instr_name
      _hw_instr_name="$(_discover_instr_binary "$_src_dir")"
      local _hw_instr=""
      [[ -f "$build_dir/$_hw_instr_name" ]] && _hw_instr="$build_dir/$_hw_instr_name"
      if [[ -n "$_hw_xclbin" && -n "$_hw_instr" ]]; then
          _run_trace_cycles_pipeline HW "$build_dir" "$_hw_xclbin" "" "$_hw_instr" "${compiler}${vsuffix}" || true
      else
          echo "[trace-cycles:HW] $name (${compiler}${vsuffix}): missing xclbin or instruction binary ($_hw_instr_name) in $build_dir; skipping" >&2
      fi
  fi

  # Mirror trace_config.json into the trace output dir so downstream
  # tools see one self-contained directory per (test, compiler, side).
  local build_traced="$BUILD_BASE/$name/traced"
  if [[ -f "$build_traced/trace_config.json" ]]; then
    cp "$build_traced/trace_config.json" "$trace_out_dir/"
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
  local variant="${3:-}"   # Empty for single-variant tests.
  local safe
  safe="$(sanitize_name "$name")"

  # If no compiler specified, deserialize from COMPILERS_STR and run all.
  if [[ -z "$compiler" ]]; then
    local compilers
    read -ra compilers <<< "$COMPILERS_STR"
    for c in "${compilers[@]}"; do
      run_one_bridge "$name" "$c" "$variant"
    done
    return
  fi

  local build_dir="$BUILD_BASE/$name/$compiler"
  local test_exe="$BUILD_BASE/$name/test.exe"
  local src_dir="$TEST_SRC/$name"

  # Variant suffix for result/log filenames (empty for single-variant tests).
  local vsuffix=""
  [[ -n "$variant" ]] && vsuffix=".${variant}"
  # Display name includes variant for multi-variant tests.
  local display_name="$name"
  [[ -n "$variant" ]] && display_name="${name}/${variant}"

  local log_file="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.log"
  local result_file="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.result"

  # Python-driven host tests have no test.exe; the run command invokes
  # `python3 .../test.py ...` directly. Skip the test.exe symlink+check.
  local _py_host=false
  is_python_host_test "$src_dir" && _py_host=true

  if ! $_py_host; then
    # Symlink shared test.exe into per-compiler build dir. Always relink so
    # stale per-compiler test.exes (left over from older builds when test.exe
    # was compiled per compiler) get replaced.
    if [[ -f "$test_exe" ]]; then
      rm -f "$build_dir/test.exe"
      ln -sf "$test_exe" "$build_dir/test.exe"
    fi

    if [[ ! -f "$build_dir/test.exe" ]]; then
      echo "SKIP" > "$result_file"
      echo "  BRIDGE $display_name ($compiler): SKIP (no test.exe)"
      return
    fi
  fi

  if ! ls "$build_dir"/*.xclbin &>/dev/null; then
    echo "SKIP" > "$result_file"
    echo "  BRIDGE $display_name ($compiler): SKIP (no xclbin)"
    return
  fi

  local run_cmd
  run_cmd="$(get_variant_run_cmd "$src_dir" "$variant")"

  local trace_out_dir="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.emu"
  mkdir -p "$trace_out_dir"

  # Phase E dual-bound timing: if a HW cycles file exists for this
  # test+compiler+variant, derive a tighter cycle budget and a scaled
  # wall-clock timeout. Otherwise fall back to today's 600 s.
  # File naming matches _run_trace_cycles_pipeline (variant = compiler+vsuffix).
  local _hw_cycles_file="$RESULTS_DIR/${safe}.${compiler}${vsuffix}.cycles.HW.txt"
  local _hw_cycles=0
  if [[ -f "$_hw_cycles_file" ]]; then
      _hw_cycles="$(tr -d '[:space:]' < "$_hw_cycles_file")"
      [[ -z "$_hw_cycles" ]] && _hw_cycles=0
  fi

  local _cycle_budget=""
  local _timeout_s=600
  if [[ "$_hw_cycles" -gt 0 ]]; then
      _cycle_budget="$(awk -v c="$_hw_cycles" -v m="$EMU_CYCLE_BUDGET_MULTIPLIER" \
          'BEGIN{ printf "%d", c*m + 0.5 }')"
      _timeout_s="$(awk -v c="$_hw_cycles" -v m="$EMU_CYCLE_BUDGET_MULTIPLIER" -v s="$EMU_SECONDS_PER_CYCLE" \
          'BEGIN{ t=c*m*s; if (t<600) t=600; printf "%d", t + 0.5 }')"
  fi

  local rc=0
  (
    cd "$build_dir"
    export XDNA_EMU="${XDNA_EMU:-debug}"
    export XDNA_EMU_LOG_LEVEL="${XDNA_EMU_LOG_LEVEL:-info}"
    # Pass through XDNA_EMU_LIB if set (explicit override).
    [[ -n "${XDNA_EMU_LIB:-}" ]] && export XDNA_EMU_LIB
    export XRT_DEVICE_BDF="ffff:ff:1f.0"
    export XDNA_TRACE_DIR="$trace_out_dir"
    if [[ -n "$_cycle_budget" ]]; then
      export XDNA_EMU_MAX_CYCLES="$_cycle_budget"
    fi
    if [[ "${NO_TIMEOUT:-false}" == "true" ]]; then
      bash -c "$run_cmd"
    else
      timeout "${_timeout_s}" bash -c "$run_cmd"
    fi
  ) > "$log_file" 2>&1 || rc=$?
  local result
  if [[ $rc -eq 0 ]] && grep -q "PASS" "$log_file"; then
    result="PASS"
  elif [[ $rc -eq 124 ]]; then
    result="TIMEOUT"
  else
    result="FAIL"
  fi

  # Override with BUDGET if the plugin reported budget-exceeded.
  # BUDGET overrides FAIL/TIMEOUT but not EMU_MISS (infrastructure failure takes priority).
  local status_line
  status_line="$(grep 'XDNA_EMU_STATUS:' "$log_file" | tail -1 || true)"
  if [[ -n "$status_line" ]]; then
    local hr
    hr="$(echo "$status_line" | grep -oP 'halt_reason=\K\w+' || true)"
    if [[ "$hr" == "budget" ]]; then
      result="BUDGET"
    fi
  fi

  # Verify emulator actually ran (catch silent fallthrough to real NPU).
  # Runs on PASS or BUDGET -- both imply sym_run_ was called; if the plugin
  # still didn't load, the EMU markers will be missing and this flips to
  # EMU_MISS (infrastructure failure takes precedence over budget signal).
  if [[ "$result" == "PASS" || "$result" == "BUDGET" ]]; then
    if ! grep -qE '(Loaded PDI|xdna_emu|XDNA emulator)' "$log_file"; then
      result="EMU_MISS"
    fi
  fi

  # Honor XFAIL annotations: a failing xfail is XFAIL, a passing xfail is XPASS.
  # EMU_MISS is an infrastructure bug -- never mask it with XFAIL.
  if [[ "$result" != "EMU_MISS" ]] && is_xfail "$src_dir" "$compiler"; then
    case "$result" in
      PASS)                 result="XPASS" ;;
      FAIL|TIMEOUT|BUDGET)  result="XFAIL" ;;
    esac
  fi

  echo "$result" > "$result_file"

  # Phase E: capture EMU cycle count via trace pipeline (best-effort).
  # Mirrors run_one_hardware's WITH_HW_CYCLES hook; only fires under the
  # cycle-diff superset flag, and only for tests that passed (a FAILed
  # bridge run's cycle count is meaningless for drift comparison).
  if [[ "$WITH_CYCLE_DIFF" == "true" && "$result" == "PASS" ]]; then
      local _emu_xclbin
      _emu_xclbin="$(find "$build_dir" -maxdepth 1 -name '*.xclbin' -print -quit 2>/dev/null || true)"
      # Discover the test's instruction binary name from run.lit (see HW path).
      local _src_dir="$TEST_SRC/$name"
      local _emu_instr_name
      _emu_instr_name="$(_discover_instr_binary "$_src_dir")"
      local _emu_instr=""
      [[ -f "$build_dir/$_emu_instr_name" ]] && _emu_instr="$build_dir/$_emu_instr_name"
      if [[ -n "$_emu_xclbin" && -n "$_emu_instr" ]]; then
          _run_trace_cycles_pipeline EMU "$build_dir" "$_emu_xclbin" "" "$_emu_instr" "${compiler}${vsuffix}" || true
      else
          echo "[trace-cycles:EMU] $name (${compiler}${vsuffix}): missing xclbin or instruction binary ($_emu_instr_name) in $build_dir; skipping" >&2
      fi
  fi

  # Phase E: compare HW vs EMU traces when both bins exist, then classify
  # the result into a persistent .cycle.result (MATCH / DRIFT / EMPTY /
  # *_TRACE_BUG / COMPARE-ERR / NO_DATA).
  if [[ "$WITH_CYCLE_DIFF" == "true" && "$result" == "PASS" ]]; then
      _run_trace_compare "$name" "${compiler}${vsuffix}" || true
      _classify_cycle_diff "$name" "${compiler}${vsuffix}" || true
  fi

  # Mirror trace_config.json into the trace output dir so downstream
  # tools see one self-contained directory per (test, compiler, side).
  local build_traced="$BUILD_BASE/$name/traced"
  if [[ -f "$build_traced/trace_config.json" ]]; then
    cp "$build_traced/trace_config.json" "$trace_out_dir/"
  fi

  # Trim trace buffer to actual data length.
  if [[ -f "$trace_out_dir/trace_raw.bin" ]]; then
    python3 "$EMU_ROOT/tools/trace-trim.py" "$trace_out_dir/trace_raw.bin" 2>/dev/null || true
  fi

  echo "  BRIDGE $display_name ($compiler): $result"
}
export -f run_one_bridge

# ---------------------------------------------------------------------------
# Trace comparison helpers (Phase 5: run_one_bridge trace_raw.bin flow)
# ---------------------------------------------------------------------------

# Decode a raw trace bin to our events-JSON format via mlir-aie's parser.
# Usage: _bin_to_events_json <trace.bin> <xclbin-mlir> <out.events.json>
# Returns 0 on success, non-zero if mlir-aie parsing fails (unwritten output).
_bin_to_events_json() {
  local bin="$1" mlir="$2" out="$3"
  PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
    python3 "$EMU_ROOT/tools/parse-trace.py" \
      --trace-bin "$bin" \
      --xclbin-mlir "$mlir" \
      --out-events "$out" 2>/dev/null
}
export -f _bin_to_events_json

# Run trace-compare on a bin pair. Requires the Rust binary and a matching
# xclbin-mlir (for parse-trace.py). Caller supplies --hw <bin> --emu <bin>
# --xclbin-mlir <mlir> [extra trace-compare args]; we convert to events
# JSONs in a temp dir, then invoke the Rust tool.
run_trace_compare() {
  local rust_bin="$EMU_ROOT/target/release/trace-compare"
  if [[ ! -x "$rust_bin" ]]; then
    echo "error: trace-compare not built at $rust_bin" >&2
    return 127
  fi

  local hw_bin="" emu_bin="" mlir="" passthrough=()
  while (( $# > 0 )); do
    case "$1" in
      --hw) hw_bin="$2"; shift 2 ;;
      --emu) emu_bin="$2"; shift 2 ;;
      --xclbin-mlir) mlir="$2"; shift 2 ;;
      *) passthrough+=("$1"); shift ;;
    esac
  done

  if [[ -z "$hw_bin" || -z "$emu_bin" || -z "$mlir" ]]; then
    echo "run_trace_compare: --hw, --emu, --xclbin-mlir required" >&2
    return 2
  fi

  local tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp/claude-1000}/trace-compare.XXXXXX")"
  local hw_events="$tmp/hw.events.json"
  local emu_events="$tmp/emu.events.json"

  if ! _bin_to_events_json "$hw_bin" "$mlir" "$hw_events" \
       || ! _bin_to_events_json "$emu_bin" "$mlir" "$emu_events"; then
    rm -rf "$tmp"
    return 3
  fi

  local rc=0
  "$rust_bin" --hw "$hw_events" --emu "$emu_events" "${passthrough[@]}" || rc=$?
  rm -rf "$tmp"
  return $rc
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
  local has_cycle=false
  [[ "$WITH_CYCLE_DIFF" == "true" ]] && has_cycle=true
  local has_mode2=false
  [[ "$MODE2" == "true" ]] && has_mode2=true

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
  local name_width=50
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
  if $has_cycle; then
    for compiler in "${compilers[@]}"; do
      local label
      label="$(echo "$compiler" | sed 's/./\U&/')"
      printf "  %-24s" "${label}/CYCLES"
    done
  fi
  if $has_mode2; then
    for compiler in "${compilers[@]}"; do
      local label
      label="$(echo "$compiler" | sed 's/./\U&/')"
      printf "  %-${col_width}s" "${label}/MODE2"
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
  if $has_cycle; then
    for _ in "${compilers[@]}"; do
      printf "  %-24s" "$(printf '%0.s-' $(seq 1 24))"
    done
  fi
  if $has_mode2; then
    for _ in "${compilers[@]}"; do
      printf "  %-${col_width}s" "$(printf '%0.s-' $(seq 1 $col_width))"
    done
  fi
  echo ""

  # --- Per-compiler counters ---
  # Use associative arrays keyed by compiler.
  declare -A compile_ok compile_fail
  declare -A bridge_pass bridge_fail bridge_skip bridge_timeout bridge_emumiss
  declare -A bridge_xfail bridge_xpass bridge_budget
  declare -A hw_pass hw_fail hw_skip hw_timeout hw_tdr hw_xfail hw_xpass
  for compiler in "${compilers[@]}"; do
    compile_ok[$compiler]=0
    compile_fail[$compiler]=0
    bridge_pass[$compiler]=0
    bridge_fail[$compiler]=0
    bridge_skip[$compiler]=0
    bridge_timeout[$compiler]=0
    bridge_emumiss[$compiler]=0
    bridge_xfail[$compiler]=0
    bridge_xpass[$compiler]=0
    bridge_budget[$compiler]=0
    hw_pass[$compiler]=0
    hw_fail[$compiler]=0
    hw_skip[$compiler]=0
    hw_timeout[$compiler]=0
    hw_tdr[$compiler]=0
    hw_xfail[$compiler]=0
    hw_xpass[$compiler]=0
  done
  declare -A trace_clean trace_diverge trace_error trace_skip
  for compiler in "${compilers[@]}"; do
    trace_clean[$compiler]=0
    trace_diverge[$compiler]=0
    trace_error[$compiler]=0
    trace_skip[$compiler]=0
  done

  # Cycle-drift counters (Phase E Task 12).
  declare -A cycle_match cycle_drift cycle_empty cycle_no_core cycle_emu_bug cycle_hw_bug cycle_compare_err cycle_no_data
  for compiler in "${compilers[@]}"; do
    cycle_match[$compiler]=0
    cycle_drift[$compiler]=0
    cycle_empty[$compiler]=0
    cycle_no_core[$compiler]=0
    cycle_emu_bug[$compiler]=0
    cycle_hw_bug[$compiler]=0
    cycle_compare_err[$compiler]=0
    cycle_no_data[$compiler]=0
  done
  declare -a cycle_offenders=()   # lines for the summary block
  declare -a cycle_empty_list=()
  declare -a cycle_no_core_list=()

  # Mode-2 counters (Task 305).
  declare -A mode2_pass mode2_fail mode2_skip mode2_error
  for compiler in "${compilers[@]}"; do
    mode2_pass[$compiler]=0
    mode2_fail[$compiler]=0
    mode2_skip[$compiler]=0
    mode2_error[$compiler]=0
  done
  declare -a mode2_offenders=()

  local has_compile_fail=false

  # Track which test names have already been counted for compile stats
  # (compile is per-test, not per-variant).
  declare -A _compile_counted  # "name:compiler" -> 1
  declare -A _mode2_counted    # "name:compiler" -> 1 (mode-2 is per-test, not per-variant)

  # --- Data rows ---
  # Each entry in test_list is "name:variant" where variant may be empty.
  for row in "${test_list[@]}"; do
    local name="${row%%:*}"
    local variant="${row#*:}"
    local safe
    safe="$(sanitize_name "$name")"

    # Variant suffix for result file lookups.
    local vsuffix=""
    [[ -n "$variant" ]] && vsuffix=".${variant}"

    # Display name: "name/variant" for multi-variant, "name" for single.
    local display_name="$name"
    [[ -n "$variant" ]] && display_name="${name}/${variant}"

    printf "%-${name_width}s" "$display_name"

    for compiler in "${compilers[@]}"; do
      # Read compile result (compile is per-test, not per-variant).
      local cr="FAIL"
      [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && \
        cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"

      if [[ "$cr" == SKIP_* ]]; then
        local skip_label="SKIP"
        [[ "$cr" == "SKIP_NPU2" ]] && skip_label="SKIP(npu2)"
        [[ "$cr" == "SKIP_COMPILER" ]] && skip_label="SKIP(compiler)"
        [[ "$cr" == "SKIP_QUARANTINED" ]] && skip_label="SKIP(quarantine)"
        if [[ "$run_hw" == "true" ]]; then
          printf "  %-${col_width}s" "$skip_label"
        fi
        printf "  %-${col_width}s" "$skip_label"
        continue
      fi

      if [[ "$cr" != "OK" ]]; then
        # Only count compile failure once per test:compiler pair.
        local ck="${name}:${compiler}"
        if [[ -z "${_compile_counted[$ck]+x}" ]]; then
          compile_fail[$compiler]=$(( ${compile_fail[$compiler]} + 1 ))
          _compile_counted["$ck"]=1
        fi
        has_compile_fail=true
        local _fail_label="FAIL*"
        [[ "$cr" == "FAIL_TRACED" ]] && _fail_label="FAILt*"
        if [[ "$run_hw" == "true" ]]; then
          printf "  %-${col_width}s" "$_fail_label"
        fi
        printf "  %-${col_width}s" "$_fail_label"
        continue
      fi

      # Only count compile success once per test:compiler pair.
      local ck="${name}:${compiler}"
      if [[ -z "${_compile_counted[$ck]+x}" ]]; then
        compile_ok[$compiler]=$(( ${compile_ok[$compiler]} + 1 ))
        _compile_counted["$ck"]=1
      fi

      # Read HW result (variant-aware).
      if [[ "$run_hw" == "true" ]]; then
        local hr="SKIP"
        [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.result" ]] && \
          hr="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.result")"
        printf "  %-${col_width}s" "$hr"
        case "$hr" in
          PASS)    hw_pass[$compiler]=$(( ${hw_pass[$compiler]} + 1 )) ;;
          TDR)     hw_tdr[$compiler]=$(( ${hw_tdr[$compiler]} + 1 ))
                   hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
          TIMEOUT) hw_timeout[$compiler]=$(( ${hw_timeout[$compiler]} + 1 ))
                   hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
          XFAIL)   hw_xfail[$compiler]=$(( ${hw_xfail[$compiler]} + 1 )) ;;
          XPASS)   hw_xpass[$compiler]=$(( ${hw_xpass[$compiler]} + 1 ))
                   hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
          SKIP*)   hw_skip[$compiler]=$(( ${hw_skip[$compiler]} + 1 )) ;;
          *)       hw_fail[$compiler]=$(( ${hw_fail[$compiler]} + 1 )) ;;
        esac
      fi

      # Read bridge (EMU) result (variant-aware).
      local br="SKIP"
      [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.result" ]] && \
        br="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.result")"
      printf "  %-${col_width}s" "$br"
      case "$br" in
        PASS)     bridge_pass[$compiler]=$(( ${bridge_pass[$compiler]} + 1 )) ;;
        BUDGET)   bridge_budget[$compiler]=$(( ${bridge_budget[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        TIMEOUT)  bridge_timeout[$compiler]=$(( ${bridge_timeout[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        EMU_MISS) bridge_emumiss[$compiler]=$(( ${bridge_emumiss[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        XFAIL)    bridge_xfail[$compiler]=$(( ${bridge_xfail[$compiler]} + 1 )) ;;
        XPASS)    bridge_xpass[$compiler]=$(( ${bridge_xpass[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
        SKIP*)    bridge_skip[$compiler]=$(( ${bridge_skip[$compiler]} + 1 )) ;;
        *)        bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
      esac
    done

    # Trace columns (per-compiler, variant-aware).
    if $has_trace; then
      for compiler in "${compilers[@]}"; do
        local trace_summary="-"
        if [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.trace.summary" ]]; then
          trace_summary="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.trace.summary")"
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

    # Cycle-drift column (Phase E Task 12).
    if $has_cycle; then
      for compiler in "${compilers[@]}"; do
        local cyc="-"
        local tag=""
        if [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.cycle.result" ]]; then
          cyc="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.cycle.result")"
          case "$cyc" in
            MATCH*)          tag="match";         cycle_match[$compiler]=$(( ${cycle_match[$compiler]} + 1 )) ;;
            DRIFT*)          tag="drift";         cycle_drift[$compiler]=$(( ${cycle_drift[$compiler]} + 1 )) ;;
            EMPTY)           tag="empty";         cycle_empty[$compiler]=$(( ${cycle_empty[$compiler]} + 1 )) ;;
            NO_CORE)         tag="no-core";       cycle_no_core[$compiler]=$(( ${cycle_no_core[$compiler]} + 1 )) ;;
            EMU_TRACE_BUG)   tag="emu-trace-bug"; cycle_emu_bug[$compiler]=$(( ${cycle_emu_bug[$compiler]} + 1 )) ;;
            HW_TRACE_BUG)    tag="hw-trace-bug";  cycle_hw_bug[$compiler]=$(( ${cycle_hw_bug[$compiler]} + 1 )) ;;
            COMPARE-ERR)     tag="compare-err";   cycle_compare_err[$compiler]=$(( ${cycle_compare_err[$compiler]} + 1 )) ;;
            NO_DATA|-)       tag="no-data";       cycle_no_data[$compiler]=$(( ${cycle_no_data[$compiler]} + 1 )) ;;
          esac
          if [[ "$tag" == "drift" || "$tag" == "emu-trace-bug" || "$tag" == "hw-trace-bug" || "$tag" == "compare-err" ]]; then
            cycle_offenders+=("  $display_name ($compiler): $cyc")
          elif [[ "$tag" == "empty" ]]; then
            cycle_empty_list+=("  $display_name ($compiler)")
          elif [[ "$tag" == "no-core" ]]; then
            cycle_no_core_list+=("  $display_name ($compiler)")
          fi
        else
          cyc="-"
          cycle_no_data[$compiler]=$(( ${cycle_no_data[$compiler]} + 1 ))
        fi
        printf "  %-24s" "$cyc"
      done
    fi

    # Mode-2 column (Task 305).
    # Mode-2 baselines are captured per (name, compiler), not per variant,
    # so we look up using $safe (no vsuffix) and only count once per
    # name:compiler pair across variant rows.
    if $has_mode2; then
      for compiler in "${compilers[@]}"; do
        local m2="-"
        if [[ -f "$RESULTS_DIR/${safe}.${compiler}.mode2.result" ]]; then
          m2="$(< "$RESULTS_DIR/${safe}.${compiler}.mode2.result")"
        fi
        printf "  %-${col_width}s" "$m2"
        local m2k="${name}:${compiler}"
        if [[ -z "${_mode2_counted[$m2k]+x}" ]]; then
          _mode2_counted["$m2k"]=1
          case "$m2" in
            PASS)  mode2_pass[$compiler]=$(( ${mode2_pass[$compiler]} + 1 )) ;;
            FAIL)  mode2_fail[$compiler]=$(( ${mode2_fail[$compiler]} + 1 ))
                   local m2detail="-"
                   [[ -f "$RESULTS_DIR/${safe}.${compiler}.mode2.summary" ]] && \
                     m2detail="$(< "$RESULTS_DIR/${safe}.${compiler}.mode2.summary")"
                   mode2_offenders+=("  $name ($compiler): $m2detail") ;;
            ERROR) mode2_error[$compiler]=$(( ${mode2_error[$compiler]} + 1 ))
                   local m2detail="-"
                   [[ -f "$RESULTS_DIR/${safe}.${compiler}.mode2.summary" ]] && \
                     m2detail="$(< "$RESULTS_DIR/${safe}.${compiler}.mode2.summary")"
                   mode2_offenders+=("  $name ($compiler): $m2detail") ;;
            SKIP|-) mode2_skip[$compiler]=$(( ${mode2_skip[$compiler]} + 1 )) ;;
            *)      mode2_skip[$compiler]=$(( ${mode2_skip[$compiler]} + 1 )) ;;
          esac
        fi
      done
    fi

    echo ""

    # Verbose: show log tail on failure.
    if [[ "$VERBOSE" == "true" ]]; then
      for compiler in "${compilers[@]}"; do
        local br="SKIP"
        [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.result" ]] && \
          br="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.result")"
        if [[ "$br" != "PASS" ]] && [[ "$br" != "SKIP"* ]] && [[ "$br" != "XFAIL" ]]; then
          local logf="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.log"
          if [[ -f "$logf" ]]; then
            echo "    --- $compiler bridge log tail ---"
            tail -5 "$logf" | sed 's/^/    /'
          fi
        fi
        if [[ "$run_hw" == "true" ]]; then
          local hr="SKIP"
          [[ -f "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.result" ]] && \
            hr="$(< "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.result")"
          if [[ "$hr" != "PASS" ]] && [[ "$hr" != "SKIP"* ]] && [[ "$hr" != "XFAIL" ]]; then
            local logf="$RESULTS_DIR/${safe}${vsuffix}.${compiler}.hw.log"
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
  # Detect whether any FAIL_TRACED exists to tailor the footnote.
  local has_traced_fail=false
  for row in "${test_list[@]}"; do
    local _n="${row%%:*}"
    local _safe
    _safe="$(sanitize_name "$_n")"
    for compiler in "${compilers[@]}"; do
      if [[ -f "$RESULTS_DIR/${_safe}.${compiler}.compile.result" ]]; then
        local _cr
        _cr="$(< "$RESULTS_DIR/${_safe}.${compiler}.compile.result")"
        [[ "$_cr" == "FAIL_TRACED" ]] && has_traced_fail=true
      fi
    done
  done
  if $has_compile_fail || $has_tdr; then
    echo ""
    $has_compile_fail && echo "*  = compile failed"
    $has_traced_fail  && echo "t* = compile failed on trace-injected MLIR (retry without --with-hw-cycles)"
    $has_tdr          && echo "TDR = hardware timeout detection and recovery (NPU hung)"
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
    if [[ ${bridge_budget[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_budget[$compiler]} BUDGET)"
    fi
    if [[ ${bridge_timeout[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_timeout[$compiler]} timeout)"
    fi
    if [[ ${bridge_emumiss[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_emumiss[$compiler]} EMU_MISS)"
    fi
    if [[ ${bridge_xfail[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_xfail[$compiler]} XFAIL)"
    fi
    if [[ ${bridge_xpass[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_xpass[$compiler]} XPASS -- unexpected pass)"
    fi
    if [[ "$run_hw" == "true" ]]; then
      local hw_extra=""
      [[ ${hw_tdr[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_tdr[$compiler]} TDR)"
      [[ ${hw_timeout[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_timeout[$compiler]} timeout)"
      [[ ${hw_xfail[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_xfail[$compiler]} XFAIL)"
      [[ ${hw_xpass[$compiler]} -gt 0 ]] && hw_extra+=" (${hw_xpass[$compiler]} XPASS)"
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

  if $has_cycle; then
    echo ""
    echo "==========================================================================="
    echo "  CYCLE DRIFT (Phase E)"
    echo "==========================================================================="
    for compiler in "${compilers[@]}"; do
      printf "  %-8s  %d MATCH  %d DRIFT  %d EMPTY  %d NO_CORE  %d EMU_TRACE_BUG  %d HW_TRACE_BUG  %d COMPARE-ERR  %d skipped\n" \
        "$compiler" \
        "${cycle_match[$compiler]}" \
        "${cycle_drift[$compiler]}" \
        "${cycle_empty[$compiler]}" \
        "${cycle_no_core[$compiler]}" \
        "${cycle_emu_bug[$compiler]}" \
        "${cycle_hw_bug[$compiler]}" \
        "${cycle_compare_err[$compiler]}" \
        "${cycle_no_data[$compiler]}"
    done

    if [[ ${#cycle_offenders[@]} -gt 0 ]]; then
      echo ""
      echo "  Offenders (DRIFT / *_TRACE_BUG / COMPARE-ERR):"
      for line in "${cycle_offenders[@]}"; do
        echo "$line"
      done
    fi

    if [[ ${#cycle_empty_list[@]} -gt 0 ]]; then
      echo ""
      echo "  Empty-trace tests (default event set insufficient -- see Phase B Limitation 1):"
      for line in "${cycle_empty_list[@]}"; do
        echo "$line"
      done
    fi

    if [[ ${#cycle_no_core_list[@]} -gt 0 ]]; then
      echo ""
      echo "  No-core tests (DMA-only / passthrough; core trace events cannot fire by design):"
      for line in "${cycle_no_core_list[@]}"; do
        echo "$line"
      done
    fi
  fi

  if $has_mode2; then
    echo ""
    echo "==========================================================================="
    echo "  MODE-2 (per-tile PC + LC sequence comparison)"
    echo "==========================================================================="
    for compiler in "${compilers[@]}"; do
      printf "  %-8s  %d PASS  %d FAIL  %d SKIP  %d ERROR\n" \
        "$compiler" \
        "${mode2_pass[$compiler]}" \
        "${mode2_fail[$compiler]}" \
        "${mode2_skip[$compiler]}" \
        "${mode2_error[$compiler]}"
    done

    if [[ ${#mode2_offenders[@]} -gt 0 ]]; then
      echo ""
      echo "  Offenders (FAIL / ERROR):"
      for line in "${mode2_offenders[@]}"; do
        echo "$line"
      done
    fi
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

  # Determine profile from XDNA_EMU env (matches runtime selection).
  local emu_profile="${XDNA_EMU:-debug}"
  # XDNA_EMU=1 is legacy shorthand for "debug".
  [[ "$emu_profile" == "1" ]] && emu_profile="debug"
  local rust_lib="$EMU_ROOT/target/$emu_profile/libxdna_emu.so"
  local installed_plugin="$XRT_LIB/libxrt_driver_emu.so.2.21.0"

  local rebuild_flags=""
  [[ "$emu_profile" == "release" ]] && rebuild_flags="--release"

  if [[ -f "$rust_lib" ]]; then
    if [[ ! -f "$installed_plugin" ]] || [[ "$rust_lib" -nt "$installed_plugin" ]]; then
      info "Plugin outdated -- rebuilding ($emu_profile profile)"
      "$SCRIPT_DIR/rebuild-plugin.sh" $rebuild_flags 2>&1 | sed 's/^/  /'
    fi
  else
    warn "No $emu_profile build found -- run 'cargo build${rebuild_flags:+ $rebuild_flags}' first"
  fi

  # ---- Phase 2: Compile --------------------------------------------------

  info "Phase 2: Compiling ${#runnable[@]} test(s) (-j${JOBS})"

  printf '%s\n' "${runnable[@]}" | xargs -P"$JOBS" -I{} bash -c 'compile_one "$@"' _ {}

  # Count compile results (per-compiler).
  local compile_ok=0 compile_fail=0 compile_skip=0
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
      elif [[ "$cr" == SKIP_* ]]; then
        ((compile_skip++)) || true
      else
        ((compile_fail++)) || true
      fi
    done
  done
  local skip_msg=""
  [[ $compile_skip -gt 0 ]] && skip_msg=", $compile_skip skipped"
  info "Phase 2 done: $compile_ok OK, $compile_fail failed${skip_msg} (across ${#compilers[@]} compiler(s))"
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

  # Build job list: (name:compiler:variant) tuples that compiled successfully.
  # Variant is empty for single-variant tests, non-empty for multi-variant.
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
      # Expand into per-variant jobs.
      local src_dir="$TEST_SRC/$name"
      while IFS= read -r variant; do
        local job_key="${name}:${compiler}:${variant}"
        all_jobs+=("$job_key")
        # Quarantine check uses name:compiler (variant-agnostic).
        if is_quarantined "$name:$compiler"; then
          hw_quarantine_jobs+=("$job_key")
        else
          hw_parallel_jobs+=("$job_key")
        fi
      done < <(get_run_variants "$src_dir")
    done
  done

  # Helper to parse job tuples.  Format: "name:compiler:variant"
  # Variant may be empty (single-variant test).
  _job_name()     { echo "${1%%:*}"; }
  _job_compiler() { local tmp="${1#*:}"; echo "${tmp%%:*}"; }
  _job_variant()  { local tmp="${1#*:}"; echo "${tmp#*:}"; }
  # Result file suffix: ".variant" for multi-variant, empty for single.
  _job_vsuffix()  { local v; v="$(_job_variant "$1")"; [[ -n "$v" ]] && echo ".$v" || echo ""; }
  # Display name: "name/variant" for multi-variant, "name" for single.
  _job_display()  { local n; n="$(_job_name "$1")"; local v; v="$(_job_variant "$1")"; [[ -n "$v" ]] && echo "${n}/${v}" || echo "$n"; }

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
  info "Phase 3+4: Running ${#all_jobs[@]} job(s) (${hw_label:+$hw_label first}${emu_label:+, then $emu_label}${q_label})"

  # Stale-artifact guard: nuke per-test .hw/ or .emu/ dirs (and their
  # paired .log/.result and trace.summary files) for any side we are NOT
  # regenerating this run, so a prior run's data can't leak into Phase 5
  # comparison or the final report. Tests not in $all_jobs (filtered out
  # by the test-name regex) are left alone.
  local _stale_hw=0 _stale_emu=0
  for entry in "${all_jobs[@]}"; do
    local _name _safe _compiler
    _name="$(_job_name "$entry")"
    _safe="$(sanitize_name "$_name")"
    _compiler="$(_job_compiler "$entry")"
    if ! $RUN_HW; then
      [[ -e "$RESULTS_DIR/${_safe}.${_compiler}.hw" ]] && _stale_hw=$((_stale_hw + 1))
      rm -rf "$RESULTS_DIR/${_safe}.${_compiler}.hw"
      rm -f  "$RESULTS_DIR/${_safe}.${_compiler}.hw.log" \
             "$RESULTS_DIR/${_safe}.${_compiler}.hw.result"
    fi
    if ! $RUN_EMU; then
      [[ -e "$RESULTS_DIR/${_safe}.${_compiler}.emu" ]] && _stale_emu=$((_stale_emu + 1))
      rm -rf "$RESULTS_DIR/${_safe}.${_compiler}.emu"
      rm -f  "$RESULTS_DIR/${_safe}.${_compiler}.emu.log" \
             "$RESULTS_DIR/${_safe}.${_compiler}.emu.result"
    fi
    # trace.summary is always regenerated by Phase 5 (or skipped); clear
    # so a stale CLEAN/ERROR doesn't survive into a one-sided report.
    rm -f "$RESULTS_DIR/${_safe}.${_compiler}.trace.summary"
  done
  if [[ $_stale_hw -gt 0 || $_stale_emu -gt 0 ]]; then
    info "Cleared stale artifacts: $_stale_hw .hw/ + $_stale_emu .emu/ dirs (sides not in this run)"
  fi

  # Sequential: run HW first (serial, NPU-bound), then EMU (parallel).
  # Concurrent execution starved HW's host-side dispatch when EMU ran at
  # -j$(nproc); doing HW solo makes its already-fast tests finish quickly
  # while EMU then has the box to itself.
  local emu_pool_pid=""

  # Launch HW with NPU job pool (if enabled).
  if $RUN_HW; then
    local tdr_suspect_file="$RESULTS_DIR/tdr_suspects.log"
    : > "$tdr_suspect_file"
    local hw_total=$(( ${#hw_parallel_jobs[@]} + ${#hw_quarantine_jobs[@]} ))
    local hw_done=0

    # --- Parallel pool: safe tests at -j$NPU_HW_JOBS ---
    if [[ ${#hw_parallel_jobs[@]} -gt 0 ]]; then
      declare -A hw_pids=()   # pid -> "name:compiler:variant"
      local hw_idx=0

      while [[ $hw_idx -lt ${#hw_parallel_jobs[@]} ]] || [[ ${#hw_pids[@]} -gt 0 ]]; do
        # Launch jobs while slots available and queue not empty.
        while [[ ${#hw_pids[@]} -lt $NPU_HW_JOBS ]] && [[ $hw_idx -lt ${#hw_parallel_jobs[@]} ]]; do
          local entry="${hw_parallel_jobs[$hw_idx]}"
          local hw_name hw_compiler hw_variant
          hw_name="$(_job_name "$entry")"
          hw_compiler="$(_job_compiler "$entry")"
          hw_variant="$(_job_variant "$entry")"
          ((hw_idx++)) || true

          (
            run_one_hardware "$hw_name" "$real_bdf" "$hw_compiler" "$hw_variant"
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
            local fin_name fin_compiler fin_vsuffix fin_display
            fin_name="$(_job_name "$finished")"
            fin_compiler="$(_job_compiler "$finished")"
            fin_vsuffix="$(_job_vsuffix "$finished")"
            fin_display="$(_job_display "$finished")"
            local fin_safe
            fin_safe="$(sanitize_name "$fin_name")"
            local hr="SKIP"
            [[ -f "$RESULTS_DIR/${fin_safe}${fin_vsuffix}.${fin_compiler}.hw.result" ]] && \
              hr="$(< "$RESULTS_DIR/${fin_safe}${fin_vsuffix}.${fin_compiler}.hw.result")"
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
            echo "  [${hw_done}/${hw_total}] HW $fin_display ($fin_compiler): $hr  @$(elapsed_sec)s${tdr_tag}"
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
      local r_name r_compiler r_vsuffix
      r_name="$(_job_name "$entry")"
      r_compiler="$(_job_compiler "$entry")"
      r_vsuffix="$(_job_vsuffix "$entry")"
      local r_safe
      r_safe="$(sanitize_name "$r_name")"
      local r_result=""
      [[ -f "$RESULTS_DIR/${r_safe}${r_vsuffix}.${r_compiler}.hw.result" ]] && \
        r_result="$(< "$RESULTS_DIR/${r_safe}${r_vsuffix}.${r_compiler}.hw.result")"
      [[ "$r_result" == "TDR" ]] && retry_jobs+=("$entry")
    done

    if [[ ${#retry_jobs[@]} -gt 0 ]]; then
      info "HW retry: ${#retry_jobs[@]} TDR result(s) rerunning serially"
      for entry in "${retry_jobs[@]}"; do
        local r_name r_compiler r_variant r_vsuffix r_display
        r_name="$(_job_name "$entry")"
        r_compiler="$(_job_compiler "$entry")"
        r_variant="$(_job_variant "$entry")"
        r_vsuffix="$(_job_vsuffix "$entry")"
        r_display="$(_job_display "$entry")"
        local r_safe
        r_safe="$(sanitize_name "$r_name")"
        # Save original log, then rerun.
        local orig_log="$RESULTS_DIR/${r_safe}${r_vsuffix}.${r_compiler}.hw.log"
        [[ -f "$orig_log" ]] && cp "$orig_log" "${orig_log%.log}.tdr-orig.log"
        run_one_hardware "$r_name" "$real_bdf" "$r_compiler" "$r_variant"
        local rr="SKIP"
        [[ -f "$RESULTS_DIR/${r_safe}${r_vsuffix}.${r_compiler}.hw.result" ]] && \
          rr="$(< "$RESULTS_DIR/${r_safe}${r_vsuffix}.${r_compiler}.hw.result")"
        if [[ "$rr" == "PASS" ]]; then
          echo "  RETRY $r_display ($r_compiler): PASS (was TDR collateral)"
        else
          echo "  RETRY $r_display ($r_compiler): $rr (confirmed failure)"
        fi
      done
    fi

    # --- Quarantine pool: known TDR tests, run last, serially ---
    if [[ ${#hw_quarantine_jobs[@]} -gt 0 ]]; then
      info "HW quarantine: running ${#hw_quarantine_jobs[@]} isolated test(s)"
      for entry in "${hw_quarantine_jobs[@]}"; do
        local q_name q_compiler q_variant q_vsuffix q_display
        q_name="$(_job_name "$entry")"
        q_compiler="$(_job_compiler "$entry")"
        q_variant="$(_job_variant "$entry")"
        q_vsuffix="$(_job_vsuffix "$entry")"
        q_display="$(_job_display "$entry")"
        run_one_hardware "$q_name" "$real_bdf" "$q_compiler" "$q_variant"
        local q_safe
        q_safe="$(sanitize_name "$q_name")"
        local qr="SKIP"
        [[ -f "$RESULTS_DIR/${q_safe}${q_vsuffix}.${q_compiler}.hw.result" ]] && \
          qr="$(< "$RESULTS_DIR/${q_safe}${q_vsuffix}.${q_compiler}.hw.result")"
        ((hw_done++)) || true
        local q_tag=""
        [[ "$qr" == "TDR" ]] && q_tag=" *** TDR ***"
        echo "  [${hw_done}/${hw_total}] HW $q_display ($q_compiler): $qr  @$(elapsed_sec)s [QUARANTINE]${q_tag}"
      done
    fi

    info "HW runs done"
  fi

  # Now launch EMU for all jobs (parallel, no NPU constraint).
  # Each job is passed as a single "name:compiler:variant" tuple to avoid
  # xargs token-splitting issues with empty variants.
  if $RUN_EMU; then
    printf '%s\n' "${all_jobs[@]}" \
      | xargs -P"$JOBS" -I{} bash -c '
          entry="$1"
          j_name="${entry%%:*}"
          tmp="${entry#*:}"
          j_compiler="${tmp%%:*}"
          j_variant="${tmp#*:}"
          run_one_bridge "$j_name" "$j_compiler" "$j_variant"
        ' _ {}
  fi
  info "EMU runs done"
  echo ""

  # ---- Phase 5: Automatic trace comparison --------------------------------
  # Comparison requires both HW and EMU captures; skip when either is off.

  if [[ "$NO_TRACE" != "true" ]] && $RUN_HW && $RUN_EMU; then
    info "Phase 5: Comparing traces"
    for entry in "${all_jobs[@]}"; do
      local t5_name t5_compiler t5_vsuffix
      t5_name="$(_job_name "$entry")"
      t5_compiler="$(_job_compiler "$entry")"
      t5_vsuffix="$(_job_vsuffix "$entry")"
      local t5_safe
      t5_safe="$(sanitize_name "$t5_name")"

      local hw_trace="$RESULTS_DIR/${t5_safe}${t5_vsuffix}.${t5_compiler}.hw/trace_raw.bin"
      local emu_trace="$RESULTS_DIR/${t5_safe}${t5_vsuffix}.${t5_compiler}.emu/trace_raw.bin"
      local summary_file="$RESULTS_DIR/${t5_safe}${t5_vsuffix}.${t5_compiler}.trace.summary"
      # parse-trace.py needs the post-lowering MLIR to resolve event slots.
      local t5_mlir="$BUILD_BASE/$t5_name/${t5_compiler}/aie_arch.mlir.prj/input_with_addresses.mlir"

      if [[ -f "$hw_trace" ]] && [[ -f "$emu_trace" ]] && [[ -f "$t5_mlir" ]]; then
        local cmp_log="$RESULTS_DIR/${t5_safe}${t5_vsuffix}.${t5_compiler}.trace.log"
        local cmp_out
        # parse-trace.py emits per-side `placement.origin_col/row` in the
        # events.json; trace-compare uses it automatically to align HW's
        # physical start_col with EMU's always-col-0 placement. The
        # --remap-columns flag is a backstop for any events.json files
        # that predate placement (older sweep artifacts, hand-rolled JSON);
        # when placement is present it's a no-op.
        cmp_out="$(run_trace_compare --hw "$hw_trace" --emu "$emu_trace" --xclbin-mlir "$t5_mlir" --remap-columns 2>&1)" || true
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
  elif [[ "$NO_TRACE" != "true" ]]; then
    local _t5_skip_reason=""
    $RUN_HW || _t5_skip_reason="HW disabled"
    $RUN_EMU || _t5_skip_reason="${_t5_skip_reason:+$_t5_skip_reason, }EMU disabled"
    info "Phase 5: Skipping trace comparison ($_t5_skip_reason)"
  fi

  # ---- Phase 5b: Event sweep (optional) -----------------------------------

  if [[ "$SWEEP" == "true" ]]; then
    # Collect unique test names that have at least one passing variant.
    local sweep_targets=()
    declare -A _sweep_seen=()
    for entry in "${all_jobs[@]}"; do
      local sw_name sw_compiler sw_vsuffix
      sw_name="$(_job_name "$entry")"
      sw_compiler="$(_job_compiler "$entry")"
      sw_vsuffix="$(_job_vsuffix "$entry")"
      [[ -n "${_sweep_seen[$sw_name]+x}" ]] && continue
      is_trace_quarantined "$sw_name" && continue
      local sw_safe
      sw_safe="$(sanitize_name "$sw_name")"
      local sw_found=false
      for suffix in hw bridge; do
        local rf="$RESULTS_DIR/${sw_safe}${sw_vsuffix}.${sw_compiler}.${suffix}.result"
        if [[ -f "$rf" ]] && [[ "$(< "$rf")" == "PASS" ]]; then
          sw_found=true
          break
        fi
      done
      if $sw_found; then
        sweep_targets+=("$sw_name")
        _sweep_seen["$sw_name"]=1
      fi
    done

    if [[ ${#sweep_targets[@]} -gt 0 ]]; then
      info "Phase 5b: Event sweep for ${#sweep_targets[@]} test(s) (HW serial, EMU -j${JOBS})"

      # Build the eligible (name, compiler) pair list once. Each pair is
      # encoded as "name|compiler|tile_spec" so the per-arm helper does
      # not need to re-parse the MLIR.
      local sweep_pairs=()
      for name in "${sweep_targets[@]}"; do
        local src_dir="$TEST_SRC/$name"

        # Discover tiles from aie.mlir. Row >= 2 emits both core and
        # memmod entries (the compute tile has two trace units, sweep
        # rotates events on each independently); row == 1 emits memtile
        # (stage 2 / #373); row == 0 emits shim (stage 1 / #372). Memmod
        # co-locates with the core tile, hence two entries per row >= 2
        # line. Mixed-type tiles route trace-sweep.py to its lockstep
        # path automatically.
        local mlir_src="$src_dir/aie.mlir"
        local tile_spec=""
        if [[ -f "$mlir_src" ]]; then
          tile_spec="$(grep -v '^[[:space:]]*//' "$mlir_src" \
            | grep -oP 'aie\.tile\(\K[0-9]+,\s*[0-9]+(?=\))' \
            | awk -F',' '{ col=$1; row=$2+0; if(row>=2) { printf "%s%d:%d:core,%d:%d:memmod", sep, col, row, col, row; sep="," } else if(row==1) { printf "%s%d:%d:memtile", sep, col, row; sep="," } else if(row==0) { printf "%s%d:%d:shim", sep, col, row; sep="," } }' \
            | sed 's/ //g')"
        fi
        if [[ -z "$tile_spec" ]]; then
          echo "  SWEEP $name: SKIP -- no traceable tiles found in $mlir_src"
          continue
        fi

        local safe
        safe="$(sanitize_name "$name")"
        for compiler in "${compilers[@]}"; do
          local build_dir="$BUILD_BASE/$name/$compiler"

          # Skip if this compiler didn't compile successfully.
          local cr="FAIL"
          [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && \
            cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"
          [[ "$cr" != "OK" ]] && continue
          [[ ! -f "$build_dir/insts.bin" ]] && continue

          sweep_pairs+=("${name}|${compiler}|${tile_spec}")
        done
      done

      if [[ ${#sweep_pairs[@]} -eq 0 ]]; then
        info "Phase 5b: no tests eligible for sweep"
      else
        # Per-arm helper. Runs trace-sweep.py for one (name, compiler)
        # pair with `arm` set to "hw" or "emu". Output is appended to a
        # single per-pair log so humans can grep one file.
        _phase5b_run_arm() {
          local arm="$1" entry="$2"
          local name compiler tile_spec
          IFS='|' read -r name compiler tile_spec <<<"$entry"

          local safe; safe="$(sanitize_name "$name")"
          local build_dir="$BUILD_BASE/$name/$compiler"
          local sweep_dir="$RESULTS_DIR/${safe}.${compiler}.sweep"
          local sweep_log="$RESULTS_DIR/${safe}.${compiler}.sweep.log"

          # trace-sweep.py auto-discovers build_dir from --test/--compiler
          # under MLIR_AIE_ROOT/build/test/npu-xrt/. Pass --build-dir
          # explicitly for parity with our caching layout.
          local sweep_args=(
            --test "$name"
            --compiler "$compiler"
            --build-dir "$build_dir"
            --tiles "$tile_spec"
            --out-dir "$sweep_dir"
            --core-sweep all
            --reuse-ctx
          )
          if [[ "$arm" == "hw" ]]; then
            sweep_args+=(--no-emu)
          else
            sweep_args+=(--no-hw)
          fi

          local label="SWEEP[${arm}] $name ($compiler)"
          {
            echo
            echo "==== $label ===="
            python3 "$EMU_ROOT/tools/trace-sweep.py" "${sweep_args[@]}"
          } >>"$sweep_log" 2>&1
          local rc=$?
          if [[ $rc -eq 0 ]]; then
            echo "  ${label}: OK (see $sweep_dir/)"
          else
            echo "  ${label}: FAIL (see $sweep_log)"
          fi
          return $rc
        }
        export -f _phase5b_run_arm

        # HW arm: serial (only one NPU device on the host).
        if $RUN_HW; then
          info "Phase 5b: HW arm -- ${#sweep_pairs[@]} sweep(s), serial"
          for entry in "${sweep_pairs[@]}"; do
            _phase5b_run_arm hw "$entry" || true
          done
        fi

        # EMU arm: parallel via xargs -P "$JOBS" (no device contention).
        if $RUN_EMU; then
          info "Phase 5b: EMU arm -- ${#sweep_pairs[@]} sweep(s), -j${JOBS}"
          printf '%s\n' "${sweep_pairs[@]}" | \
            xargs -d '\n' -I{} -P"$JOBS" bash -c '_phase5b_run_arm emu "$@"' _ {} || true
        fi
      fi
    else
      info "Phase 5b: no tests eligible for sweep"
    fi
  fi

  # ---- Phase 5b': PC-anchored sweep (optional) ----------------------------
  #
  # Runs a mode-1 (event_pc) lockstep sweep on passing tests and produces a
  # PC-anchored HW/EMU coverage report.  Requires --trace=pc-anchored
  # (or its superset --mode2, which additionally enables mode-2 baseline
  # capture documented below).
  #
  # Tile discovery: greps the test's aie.mlir for aie.tile(col, row) lines
  # where row >= 2 (compute rows) and formats them as col:row:core.  If no
  # compute tiles are found the test is skipped with a warning.
  #
  # Grounding: PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1 (default for mode-1).
  # The sweep uses --reuse-ctx to cut per-batch latency on Phoenix.
  #
  # Mode-2 baseline (Phase 6 / Task 6.2):
  #   trace-sweep.py's --with-mode2-baseline is on by default, so each
  #   sweep drops mode-2 (inst_exec) baselines into
  #   <sweep_dir>/mode2-baseline/{hw,emu}/trace.events.json after the
  #   mode-1 sweep. The trace-compare aggregator pairs them via
  #   find_mode2_baseline_pair() and runs the three-layer comparator
  #   per tile (PC sequence + LC sequence gate; atom windows are
  #   informational). One-sided baselines (HW or EMU only) get a SKIP
  #   line, which is harmless rather than a regression.

  if [[ "$PC_ANCHORED" == "true" ]]; then
    local tc_bin=""
    if [[ -x "$EMU_ROOT/target/release/trace-compare" ]]; then
      tc_bin="$EMU_ROOT/target/release/trace-compare"
    elif [[ -x "$EMU_ROOT/target/debug/trace-compare" ]]; then
      tc_bin="$EMU_ROOT/target/debug/trace-compare"
    fi

    if [[ -z "$tc_bin" ]]; then
      info "Phase 5b': SKIP -- trace-compare binary not found (cargo build --release --bin trace-compare)"
    else
      # Collect passing tests (same eligibility as Phase 5b sweep).
      local pa_targets=()
      declare -A _pa_seen=()
      for entry in "${all_jobs[@]}"; do
        local pa_name pa_compiler pa_vsuffix
        pa_name="$(_job_name "$entry")"
        pa_compiler="$(_job_compiler "$entry")"
        pa_vsuffix="$(_job_vsuffix "$entry")"
        [[ -n "${_pa_seen[$pa_name]+x}" ]] && continue
        is_trace_quarantined "$pa_name" && continue
        local pa_safe
        pa_safe="$(sanitize_name "$pa_name")"
        local pa_found=false
        for suffix in hw bridge; do
          local rf="$RESULTS_DIR/${pa_safe}${pa_vsuffix}.${pa_compiler}.${suffix}.result"
          if [[ -f "$rf" ]] && [[ "$(< "$rf")" == "PASS" ]]; then
            pa_found=true
            break
          fi
        done
        $pa_found && { pa_targets+=("$pa_name"); _pa_seen["$pa_name"]=1; }
      done

      if [[ ${#pa_targets[@]} -gt 0 ]]; then
        info "Phase 5b': PC-anchored sweep for ${#pa_targets[@]} test(s)"
        for name in "${pa_targets[@]}"; do
          local safe
          safe="$(sanitize_name "$name")"
          local src_dir="$TEST_SRC/$name"

          # Discover tiles from aie.mlir:
          #   strip line comments (//), then grep for aie.tile(col, row),
          #   then classify: row >= 2 emits core+memmod (compute tile has
          #   two trace units, stage 3 / #374); row == 1 emits memtile
          #   (stage 2 / #373); row == 0 emits shim (stage 1 / #372). A
          #   long-term improvement would be to invoke aie-translate for
          #   an authoritative tile list, but the regex covers every test
          #   currently in the suite. Tracked in docs/superpowers/findings/
          #   2026-04-28-aie-translate-tile-discovery-followup.md
          local mlir_src="$src_dir/aie.mlir"
          local tile_spec=""
          if [[ -f "$mlir_src" ]]; then
            tile_spec="$(grep -v '^[[:space:]]*//' "$mlir_src" \
              | grep -oP 'aie\.tile\(\K[0-9]+,\s*[0-9]+(?=\))' \
              | awk -F',' '{ col=$1; row=$2+0; if(row>=2) { printf "%s%d:%d:core,%d:%d:memmod", sep, col, row, col, row; sep="," } else if(row==1) { printf "%s%d:%d:memtile", sep, col, row; sep="," } else if(row==0) { printf "%s%d:%d:shim", sep, col, row; sep="," } }' \
              | sed 's/ //g')"
          fi
          if [[ -z "$tile_spec" ]]; then
            echo "  PC-ANCHORED $name: SKIP -- no traceable tiles found in $mlir_src"
            continue
          fi

          for compiler in "${compilers[@]}"; do
            local build_dir="$BUILD_BASE/$name/$compiler"
            local sweep_dir="$RESULTS_DIR/${safe}.${compiler}.pc-anchored"
            local sweep_log="$RESULTS_DIR/${safe}.${compiler}.pc-anchored.sweep.log"
            local report="$RESULTS_DIR/${safe}.${compiler}.pc-anchored.report.txt"

            # Skip if this compiler didn't compile successfully.
            local cr="FAIL"
            [[ -f "$RESULTS_DIR/${safe}.${compiler}.compile.result" ]] && \
              cr="$(< "$RESULTS_DIR/${safe}.${compiler}.compile.result")"
            [[ "$cr" != "OK" ]] && continue

            [[ -f "$build_dir/insts.bin" ]] || continue

            local sweep_args=(
              --test "$name"
              --compiler "$compiler"
              --tiles "$tile_spec"
              --out-dir "$sweep_dir"
              --mode event_pc
              --core-grounding "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1"
              --reuse-ctx
            )
            # Mode-2 baseline capture: trace-sweep.py defaults this on, but
            # we pass it explicitly when --mode2 is set so a future default
            # flip upstream cannot silently turn it off here. Without
            # --mode2 we inherit whatever the Python tool's default is.
            if [[ "$MODE2" == "true" ]]; then
              sweep_args+=(--with-mode2-baseline)
            fi
            $RUN_HW || sweep_args+=(--no-hw)
            $RUN_EMU || sweep_args+=(--no-emu)

            echo "  PC-ANCHORED sweep $name ($compiler) ..."
            if python3 "$EMU_ROOT/tools/trace-sweep.py" "${sweep_args[@]}" \
                > "$sweep_log" 2>&1; then
              echo "  PC-ANCHORED sweep $name ($compiler): OK"
              # Run trace-compare --pc-anchored on the sweep output.
              # --remap-columns dense-remaps each side's columns to 0..N-1
              # before pairing tiles. HW packet headers carry the
              # absolute placement column (e.g., col=1 for a kernel placed
              # starting at col 1), while the emulator's trace_unit emits
              # the array-local column (always starting at 0). Without
              # remap, mode-2 baselines log "HW only" + "EMU only" for
              # the same logical tile; the mode-1 PC-anchored aggregator
              # honours the same flag.
              if "$tc_bin" --sweep "$sweep_dir" --pc-anchored --remap-columns -o "$report" 2>&1; then
                # `grep set_diff` returns 1 when the report has zero divergence
                # lines (e.g. Batches: 0). Under `set -e -o pipefail` that
                # would abort the whole script, so guard the pipe with
                # `|| true` and let the empty fallback below handle the
                # display.
                local top_event
                top_event="$( { grep 'set_diff' "$report" || true; } | sort -t '=' -k2 -rn | head -1 | awk '{print $1}')"
                echo "  PC-ANCHORED compare $name ($compiler): OK (top event: ${top_event:-none}; report: $report)"
                if [[ "$MODE2" == "true" ]]; then
                  _classify_mode2 "$RESULTS_DIR/${safe}.${compiler}.mode2.result" \
                                  "$RESULTS_DIR/${safe}.${compiler}.mode2.summary" \
                                  "$report"
                  local m2r
                  m2r="$(< "$RESULTS_DIR/${safe}.${compiler}.mode2.result")"
                  echo "  MODE2 $name ($compiler): $m2r"
                fi
              else
                echo "  PC-ANCHORED compare $name ($compiler): FAIL (see $report)"
                if [[ "$MODE2" == "true" ]]; then
                  echo "ERROR" > "$RESULTS_DIR/${safe}.${compiler}.mode2.result"
                  echo "ERROR: trace-compare failed -- see $report" \
                    > "$RESULTS_DIR/${safe}.${compiler}.mode2.summary"
                fi
              fi
            else
              echo "  PC-ANCHORED sweep $name ($compiler): FAIL (see $sweep_log)"
              if [[ "$MODE2" == "true" ]]; then
                echo "ERROR" > "$RESULTS_DIR/${safe}.${compiler}.mode2.result"
                echo "ERROR: trace-sweep.py failed -- see $sweep_log" \
                  > "$RESULTS_DIR/${safe}.${compiler}.mode2.summary"
              fi
            fi
          done
        done
      else
        info "Phase 5b': no tests eligible for PC-anchored sweep"
      fi
    fi
  fi

  # ---- Phase 5c: aiesimulator VCD coverage audit -------------------------

  if [[ "$RUN_AIESIM" == "true" ]]; then
    # Find the vcd_compare binary (prefer release, fall back to debug).
    local vcd_compare_bin=""
    if [[ -x "$EMU_ROOT/target/release/vcd_compare" ]]; then
      vcd_compare_bin="$EMU_ROOT/target/release/vcd_compare"
    elif [[ -x "$EMU_ROOT/target/debug/vcd_compare" ]]; then
      vcd_compare_bin="$EMU_ROOT/target/debug/vcd_compare"
    fi

    if [[ -z "$vcd_compare_bin" ]]; then
      info "Phase 5c: SKIP -- vcd_compare binary not found (cargo build --bin vcd_compare)"
    elif ! command -v aiesimulator &>/dev/null; then
      info "Phase 5c: SKIP -- aiesimulator not in PATH"
    else
      # Collect Chess tests that compiled successfully and have sim artifacts.
      local aiesim_targets=()
      for name in "${compiled[@]}"; do
        local safe
        safe="$(sanitize_name "$name")"
        local cr="FAIL"
        [[ -f "$RESULTS_DIR/${safe}.chess.compile.result" ]] && \
          cr="$(< "$RESULTS_DIR/${safe}.chess.compile.result")"
        [[ "$cr" != "OK" ]] && continue
        # Check for the sim/ directory (produced by --aiesim flag on aiecc).
        local build_dir="$BUILD_BASE/$name/chess"
        local prj_dir
        prj_dir="$(find "$build_dir" -maxdepth 1 -name '*.prj' -type d -print -quit 2>/dev/null || true)"
        [[ -z "$prj_dir" ]] && continue
        [[ -d "$prj_dir/sim" ]] || continue
        aiesim_targets+=("$name")
      done

      if [[ ${#aiesim_targets[@]} -eq 0 ]]; then
        info "Phase 5c: no Chess builds with sim/ artifacts found"
      else
        info "Phase 5c: Running aiesimulator on ${#aiesim_targets[@]} Chess build(s)"

        local aiesim_pass=0 aiesim_fail=0 aiesim_skip=0
        for name in "${aiesim_targets[@]}"; do
          local safe
          safe="$(sanitize_name "$name")"
          local build_dir="$BUILD_BASE/$name/chess"
          local prj_dir
          prj_dir="$(find "$build_dir" -maxdepth 1 -name '*.prj' -type d -print -quit 2>/dev/null)"
          local sim_log="$RESULTS_DIR/${safe}.chess.aiesim.log"
          local sim_result_file="$RESULTS_DIR/${safe}.chess.aiesim.result"
          local sim_out_dir="$build_dir/aiesimulator_output"

          # Run aiesimulator from the build directory so output lands there.
          local sim_rc=0
          (
            cd "$build_dir"
            nice -n 19 timeout 120 aiesimulator \
              --pkg-dir="$prj_dir/sim" \
              --dump-vcd=aiesim_trace
          ) > "$sim_log" 2>&1 || sim_rc=$?

          if [[ $sim_rc -ne 0 ]]; then
            echo "FAIL" > "$sim_result_file"
            echo "  AIESIM $name: FAIL (exit $sim_rc)"
            ((aiesim_fail++)) || true
            continue
          fi

          # Find the VCD file. aiesimulator creates it as
          # <build_dir>/aiesimulator_output/aiesim_trace.vcd (or similar).
          local vcd_file=""
          if [[ -d "$sim_out_dir" ]]; then
            vcd_file="$(find "$sim_out_dir" -name '*.vcd' -print -quit 2>/dev/null || true)"
          fi
          # Also check build_dir directly (some versions place it there).
          if [[ -z "$vcd_file" ]]; then
            vcd_file="$(find "$build_dir" -maxdepth 1 -name '*.vcd' -print -quit 2>/dev/null || true)"
          fi

          if [[ -z "$vcd_file" ]]; then
            echo "FAIL" > "$sim_result_file"
            echo "  AIESIM $name: FAIL (no VCD produced)"
            ((aiesim_fail++)) || true
            continue
          fi

          # Run VCD coverage audit.
          local coverage_log="$RESULTS_DIR/${safe}.chess.aiesim-coverage.log"
          local cov_rc=0
          "$vcd_compare_bin" --coverage "$vcd_file" --device vc2802 > "$coverage_log" 2>&1 || cov_rc=$?

          if [[ $cov_rc -eq 0 ]]; then
            echo "PASS" > "$sim_result_file"
            # Extract mapped percentage from coverage output for summary.
            local mapped_pct
            mapped_pct="$(grep -oP '\(\K[0-9.]+%' "$coverage_log" | head -1 || true)"
            echo "  AIESIM $name: PASS${mapped_pct:+ ($mapped_pct mapped)}"
            ((aiesim_pass++)) || true
          else
            echo "FAIL" > "$sim_result_file"
            echo "  AIESIM $name: VCD coverage FAIL (see $coverage_log)"
            ((aiesim_fail++)) || true
          fi
        done

        info "Phase 5c done: $aiesim_pass pass, $aiesim_fail fail"
      fi
    fi
  fi

  # ---- Phase 6: Report ---------------------------------------------------

  # Build variant-expanded report rows.  Each row is "name:variant" where
  # variant is empty for single-variant tests.  Tests that were skipped
  # (npu2-only, not compiled) still appear with an empty variant.
  local report_rows=()
  declare -A _reported_tests=()
  # First, add all variant-expanded entries from the run phase.
  for entry in "${all_jobs[@]}"; do
    local rr_name rr_variant
    rr_name="$(_job_name "$entry")"
    rr_variant="$(_job_variant "$entry")"
    local rr_key="${rr_name}:${rr_variant}"
    if [[ -z "${_reported_tests[$rr_key]+x}" ]]; then
      report_rows+=("$rr_key")
      _reported_tests["$rr_key"]=1
      _reported_tests["$rr_name"]=1  # Mark test name as seen.
    fi
  done
  # Add tests that were skipped or failed compile (not in all_jobs).
  for name in "${tests[@]}"; do
    if [[ -z "${_reported_tests[$name]+x}" ]]; then
      report_rows+=("${name}:")
      _reported_tests["$name"]=1
    fi
  done

  # Re-classify cycle diffs from final on-disk state. The in-Phase-3+4
  # call inside run_one_bridge runs on EMU's timeline; HW's parallel job
  # may not have written cycles.HW.txt yet, so an early classification
  # can read an EMU_TRACE_BUG / HW_TRACE_BUG asymmetry that resolves
  # once both sides finish. Re-running here with the complete state
  # produces stable classifications.
  if [[ "$WITH_CYCLE_DIFF" == "true" ]]; then
    local _reclass_compilers
    read -ra _reclass_compilers <<< "$COMPILERS_STR"
    for _row in "${report_rows[@]}"; do
      local _rn="${_row%%:*}"
      local _rv="${_row#*:}"
      local _rsuffix=""
      [[ -n "$_rv" ]] && _rsuffix=".${_rv}"
      for _rc in "${_reclass_compilers[@]}"; do
        local _br="SKIP"
        [[ -f "$RESULTS_DIR/$(sanitize_name "$_rn")${_rsuffix}.${_rc}.bridge.result" ]] && \
          _br="$(< "$RESULTS_DIR/$(sanitize_name "$_rn")${_rsuffix}.${_rc}.bridge.result")"
        [[ "$_br" == "PASS" ]] || continue
        _classify_cycle_diff "$_rn" "${_rc}${_rsuffix}" || true
      done
    done
  fi

  info "Phase 6: Report"
  print_report report_rows "$RUN_HW"
}

main
