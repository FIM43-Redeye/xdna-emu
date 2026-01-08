#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Test harness for running mlir-aie tests against mock XRT + xdna-emu
#
# This script discovers built tests in mlir-aie, compiles them against
# our mock XRT library, and runs them to verify emulator correctness.

set -e

# Configuration - adjust these paths as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK_XRT_ROOT="${SCRIPT_DIR}/.."
XDNA_EMU_ROOT="${MOCK_XRT_ROOT}/.."

# Default paths (can be overridden via environment)
MLIR_AIE_ROOT="${MLIR_AIE_ROOT:-/home/triple/npu-work/mlir-aie}"
MLIR_AIE_BUILD="${MLIR_AIE_BUILD:-${MLIR_AIE_ROOT}/build}"

# Build paths
MOCK_XRT_BUILD="${MOCK_XRT_ROOT}/build"
MOCK_XRT_INCLUDE="${MOCK_XRT_ROOT}/include"
XDNA_EMU_LIB="${XDNA_EMU_ROOT}/target/release"

# We use our own test_utils.h (drop-in replacement for mlir-aie's)
# No dependency on mlir-aie's test_utils library

# Output directory for compiled tests
TEST_BUILD_DIR="${MOCK_XRT_BUILD}/mlir_aie_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
COMPILE_ERRORS=0

usage() {
    echo "Usage: $0 [OPTIONS] [TEST_PATTERN]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -l, --list       List available tests without running"
    echo "  -v, --verbose    Verbose output (show test output)"
    echo "  -c, --compile    Only compile tests, don't run"
    echo "  -k, --keep       Keep compiled test binaries"
    echo "  -j N             Parallel compilation (default: 4)"
    echo ""
    echo "TEST_PATTERN:"
    echo "  Optional glob pattern to filter tests (e.g., 'add_one*')"
    echo ""
    echo "Environment variables:"
    echo "  MLIR_AIE_ROOT    Path to mlir-aie source (default: /home/triple/npu-work/mlir-aie)"
    echo "  MLIR_AIE_BUILD   Path to mlir-aie build (default: \$MLIR_AIE_ROOT/build)"
    echo "  RUST_LOG         Set to 'info' or 'debug' for emulator logging"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    local missing=0

    if [[ ! -f "${MOCK_XRT_BUILD}/libxrt_mock.so" ]]; then
        log_error "Mock XRT library not found. Build it first:"
        echo "  cd ${MOCK_XRT_ROOT}/build && cmake .. && make"
        missing=1
    fi

    if [[ ! -f "${XDNA_EMU_LIB}/libxdna_emu.so" ]]; then
        log_error "xdna-emu library not found. Build it first:"
        echo "  cd ${XDNA_EMU_ROOT} && cargo build --release"
        missing=1
    fi

    if [[ ! -d "${MLIR_AIE_BUILD}/test/npu-xrt" ]]; then
        log_error "mlir-aie test build directory not found"
        echo "  Build mlir-aie tests first"
        missing=1
    fi

    if [[ $missing -eq 1 ]]; then
        exit 1
    fi
}

# Discover tests with built xclbin files
discover_tests() {
    local pattern="${1:-*}"
    local tests=()

    # Find all directories with aie.xclbin in the build tree
    while IFS= read -r xclbin; do
        local test_dir="$(dirname "$xclbin")"
        local test_name="$(basename "$test_dir")"

        # Check if pattern matches
        if [[ "$test_name" == $pattern ]]; then
            # Check if test.cpp exists in source
            local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}"
            if [[ -f "${source_dir}/test.cpp" ]]; then
                tests+=("$test_name")
            fi
        fi
    done < <(find "${MLIR_AIE_BUILD}/test/npu-xrt" -name "aie.xclbin" 2>/dev/null | sort)

    printf '%s\n' "${tests[@]}"
}

# Compile a single test
compile_test() {
    local test_name="$1"
    local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}"
    local build_dir="${MLIR_AIE_BUILD}/test/npu-xrt/${test_name}"
    local output_bin="${TEST_BUILD_DIR}/${test_name}"

    # Ensure output directory exists
    mkdir -p "${TEST_BUILD_DIR}"

    # Compile command
    # Uses our mock XRT headers and test_utils.h (drop-in replacement)
    g++ -std=c++17 -O2 \
        "${source_dir}/test.cpp" \
        -o "${output_bin}" \
        -I"${MOCK_XRT_INCLUDE}" \
        -L"${MOCK_XRT_BUILD}" -lxrt_mock \
        -L"${XDNA_EMU_LIB}" -lxdna_emu \
        -Wl,-rpath,"${MOCK_XRT_BUILD}" \
        -Wl,-rpath,"${XDNA_EMU_LIB}" \
        2>&1

    return $?
}

# Run a single test
run_test() {
    local test_name="$1"
    local verbose="$2"
    local build_dir="${MLIR_AIE_BUILD}/test/npu-xrt/${test_name}"
    local test_bin="${TEST_BUILD_DIR}/${test_name}"
    local xclbin="${build_dir}/aie.xclbin"
    local insts="${build_dir}/insts.bin"

    if [[ ! -f "$test_bin" ]]; then
        log_error "Test binary not found: $test_bin"
        return 1
    fi

    if [[ ! -f "$xclbin" ]]; then
        log_error "xclbin not found: $xclbin"
        return 1
    fi

    if [[ ! -f "$insts" ]]; then
        log_error "insts.bin not found: $insts"
        return 1
    fi

    # Run the test
    local output
    local exit_code

    if [[ "$verbose" == "1" ]]; then
        "$test_bin" --xclbin "$xclbin" --instr "$insts" --kernel MLIR_AIE --verbosity 1
        exit_code=$?
    else
        output=$("$test_bin" --xclbin "$xclbin" --instr "$insts" --kernel MLIR_AIE 2>&1)
        exit_code=$?
    fi

    if [[ $exit_code -eq 0 ]]; then
        return 0
    else
        if [[ "$verbose" != "1" ]]; then
            echo "$output" | tail -20
        fi
        return 1
    fi
}

# Main execution
main() {
    local list_only=0
    local verbose=0
    local compile_only=0
    local keep_binaries=0
    local parallel=4
    local pattern="*"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -l|--list)
                list_only=1
                shift
                ;;
            -v|--verbose)
                verbose=1
                shift
                ;;
            -c|--compile)
                compile_only=1
                shift
                ;;
            -k|--keep)
                keep_binaries=1
                shift
                ;;
            -j)
                parallel="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                pattern="$1"
                shift
                ;;
        esac
    done

    echo "=========================================="
    echo "  mlir-aie Test Harness (Mock XRT)"
    echo "=========================================="
    echo ""

    # Check prerequisites
    check_prerequisites

    # Discover tests
    log_info "Discovering tests matching pattern: $pattern"
    mapfile -t tests < <(discover_tests "$pattern")

    if [[ ${#tests[@]} -eq 0 ]]; then
        log_error "No tests found matching pattern: $pattern"
        exit 1
    fi

    log_info "Found ${#tests[@]} test(s)"
    echo ""

    # List only mode
    if [[ $list_only -eq 1 ]]; then
        for test_name in "${tests[@]}"; do
            echo "  $test_name"
        done
        exit 0
    fi

    # Compile tests
    log_info "Compiling tests..."
    for test_name in "${tests[@]}"; do
        TOTAL=$((TOTAL + 1))
        printf "  Compiling %-40s " "$test_name..."

        if compile_output=$(compile_test "$test_name" 2>&1); then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
            echo "$compile_output" | head -10
            COMPILE_ERRORS=$((COMPILE_ERRORS + 1))
            SKIPPED=$((SKIPPED + 1))
        fi
    done
    echo ""

    if [[ $COMPILE_ERRORS -gt 0 ]]; then
        log_error "$COMPILE_ERRORS test(s) failed to compile"
    fi

    # Run tests (unless compile-only)
    if [[ $compile_only -eq 0 ]]; then
        log_info "Running tests..."
        echo ""

        for test_name in "${tests[@]}"; do
            # Skip if compilation failed
            if [[ ! -f "${TEST_BUILD_DIR}/${test_name}" ]]; then
                continue
            fi

            printf "  Running %-42s " "$test_name..."

            if run_test "$test_name" "$verbose"; then
                log_success ""
                PASSED=$((PASSED + 1))
            else
                log_fail ""
                FAILED=$((FAILED + 1))
            fi
        done
    fi

    # Clean up unless keeping binaries
    if [[ $keep_binaries -eq 0 && $compile_only -eq 0 ]]; then
        rm -rf "${TEST_BUILD_DIR}"
    fi

    # Summary
    echo ""
    echo "=========================================="
    echo "  Summary"
    echo "=========================================="
    echo "  Total:    $TOTAL"
    echo "  Passed:   $PASSED"
    echo "  Failed:   $FAILED"
    echo "  Skipped:  $SKIPPED"
    echo "=========================================="

    if [[ $FAILED -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
