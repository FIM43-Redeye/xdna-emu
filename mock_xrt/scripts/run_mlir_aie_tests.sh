#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Test harness for running mlir-aie tests against mock XRT + xdna-emu
#
# This script discovers built tests in mlir-aie, compiles them against
# our mock XRT library, and runs them to verify emulator correctness.

# Note: We intentionally do NOT use set -e here because we want to
# continue running tests even when some fail, and collect results.

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

# mlir-aie toolchain paths (for --prepare)
MLIR_AIE_BIN="${MLIR_AIE_BUILD}/bin"
MLIR_AIE_PYTHON="${MLIR_AIE_ROOT}/install/python:${MLIR_AIE_ROOT}/build/python"
MLIR_AIE_VENV="${MLIR_AIE_ROOT}/ironenv"
PEANO_INSTALL_DIR="${MLIR_AIE_VENV}/lib/python3.13/site-packages/llvm-aie"
AIECC="${MLIR_AIE_BIN}/aiecc.py"
AIE_OPT="${MLIR_AIE_BIN}/aie-opt"

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
    echo "  -p, --prepare    Generate missing instruction files using mlir-aie toolchain"
    echo "  -f, --force      Force recompilation even if up-to-date"
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

    # Find all directories with xclbin files (aie.xclbin or final.xclbin) in the build tree
    while IFS= read -r xclbin; do
        local test_dir="$(dirname "$xclbin")"

        # Get relative path from npu-xrt directory
        # This handles nested tests like adjacent_memtile_access/two_memtiles
        local rel_path="${test_dir#${MLIR_AIE_BUILD}/test/npu-xrt/}"
        local test_name="$rel_path"

        # For pattern matching, use the leaf directory name
        local leaf_name="$(basename "$test_dir")"

        # Check if pattern matches (match against leaf name for simplicity)
        if [[ "$leaf_name" == $pattern ]] || [[ "$test_name" == $pattern ]]; then
            # Check if test.cpp exists in source
            local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${rel_path}"
            if [[ -f "${source_dir}/test.cpp" ]]; then
                tests+=("$rel_path")
            fi
        fi
    done < <(find "${MLIR_AIE_BUILD}/test/npu-xrt" \( -name "aie.xclbin" -o -name "final.xclbin" \) 2>/dev/null | sort)

    printf '%s\n' "${tests[@]}"
}

# Prepare a test by generating missing instruction files
# Uses mlir-aie toolchain (aie-opt, aiecc.py)
prepare_test() {
    local test_name="$1"
    local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}"
    local build_dir="${MLIR_AIE_BUILD}/test/npu-xrt/${test_name}"

    # Check if instruction files already exist
    if [[ -f "${build_dir}/insts.bin" ]] || [[ -f "${build_dir}/aie_run_seq.bin" ]]; then
        echo "EXISTS"
        return 0
    fi

    # Check what kind of test this is by parsing run.lit
    local run_lit="${source_dir}/run.lit"
    if [[ ! -f "$run_lit" ]]; then
        echo "NO_LIT"
        return 1
    fi

    # Detect ctrl_packet tests (need overlay generation)
    if grep -q "aie-generate-column-control-overlay" "$run_lit"; then
        # This is a ctrl_packet test - needs special handling
        pushd "$build_dir" > /dev/null

        # Step 1: Generate overlay if not exists
        if [[ ! -f "aie_overlay.mlir" ]]; then
            "$AIE_OPT" -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" \
                aie_arch.mlir -o aie_overlay.mlir 2>&1 || { popd > /dev/null; return 1; }
        fi

        # Step 2: Generate ctrlpkt.bin and aie_run_seq.bin
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" --no-xchesscc --no-xbridge \
            --device-name=main \
            --aie-generate-ctrlpkt --ctrlpkt-name=ctrlpkt.bin \
            --aie-generate-npu-insts --npu-insts-name=aie_run_seq.bin \
            aie_overlay.mlir 2>&1

        local result=$?
        popd > /dev/null
        return $result

    # Detect ELF-based tests that might need bin conversion
    elif grep -q "aie-generate-elf" "$run_lit" && [[ -f "${build_dir}/insts.elf" ]]; then
        # ELF exists, test harness can use it directly
        echo "HAS_ELF"
        return 0

    # Standard test - try to generate insts.bin
    elif grep -q "aie-generate-npu-insts\|npu-insts-name" "$run_lit"; then
        pushd "$build_dir" > /dev/null

        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" --no-xchesscc --no-xbridge \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie_arch.mlir 2>&1

        local result=$?
        popd > /dev/null
        return $result
    fi

    echo "UNKNOWN"
    return 1
}

# Check if test needs recompilation
# Returns 0 if recompilation needed, 1 if up-to-date
needs_recompile() {
    local test_name="$1"
    local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}"
    local output_bin="${TEST_BUILD_DIR}/${test_name}"

    # Need to compile if binary doesn't exist
    [[ ! -f "$output_bin" ]] && return 0

    # Need to compile if source is newer than binary
    [[ "${source_dir}/test.cpp" -nt "$output_bin" ]] && return 0

    # Need to compile if any header changed (check mock XRT headers)
    local newest_header
    newest_header=$(find "${MOCK_XRT_INCLUDE}" -name "*.h" -newer "$output_bin" 2>/dev/null | head -1)
    [[ -n "$newest_header" ]] && return 0

    # Up-to-date
    return 1
}

# Compile a single test
compile_test() {
    local test_name="$1"
    local force="$2"
    local source_dir="${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}"
    local build_dir="${MLIR_AIE_BUILD}/test/npu-xrt/${test_name}"
    local output_bin="${TEST_BUILD_DIR}/${test_name}"

    # Ensure output directory exists (handles nested paths like adjacent_memtile_access/two_memtiles)
    mkdir -p "$(dirname "${output_bin}")"

    # Skip if up-to-date (unless forced)
    if [[ "$force" != "1" ]] && ! needs_recompile "$test_name"; then
        echo "UP-TO-DATE"
        return 0
    fi

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
    local insts="${build_dir}/insts.bin"

    # Find xclbin file (may be aie.xclbin or final.xclbin)
    local xclbin="${build_dir}/aie.xclbin"
    if [[ ! -f "$xclbin" ]]; then
        xclbin="${build_dir}/final.xclbin"
    fi

    if [[ ! -f "$test_bin" ]]; then
        log_error "Test binary not found: $test_bin"
        return 1
    fi

    if [[ ! -f "$xclbin" ]]; then
        log_error "xclbin not found in: $build_dir"
        return 1
    fi

    # Check for instruction file - different tests use different names/formats
    # Priority: insts.bin > aie_run_seq.bin > insts.elf
    if [[ ! -f "$insts" ]]; then
        if [[ -f "${build_dir}/aie_run_seq.bin" ]]; then
            insts="${build_dir}/aie_run_seq.bin"
        elif [[ -f "${build_dir}/insts.elf" ]]; then
            insts="${build_dir}/insts.elf"
        else
            # List what instruction files ARE needed for this test
            local needed=""
            if grep -q "aie_run_seq.bin\|ctrlpkt" "${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}/run.lit" 2>/dev/null; then
                needed="(needs: aiecc.py --aie-generate-ctrlpkt --aie-generate-npu-insts)"
            elif grep -q "insts.elf" "${MLIR_AIE_ROOT}/test/npu-xrt/${test_name}/run.lit" 2>/dev/null; then
                needed="(has insts.elf but may need conversion)"
            fi
            log_skip "No instruction file found $needed"
            return 2  # Return 2 for "skipped"
        fi
    fi

    # Run the test from the build directory so relative paths work
    # Many mlir-aie tests use hardcoded relative paths like "insts.bin"
    local output
    local exit_code
    local abs_test_bin
    abs_test_bin="$(cd "$(dirname "$test_bin")" && pwd)/$(basename "$test_bin")"

    pushd "$build_dir" > /dev/null

    if [[ "$verbose" == "1" ]]; then
        "$abs_test_bin" --xclbin "$xclbin" --instr "$insts" --kernel MLIR_AIE --verbosity 1
        exit_code=$?
    else
        output=$("$abs_test_bin" --xclbin "$xclbin" --instr "$insts" --kernel MLIR_AIE 2>&1)
        exit_code=$?
    fi

    popd > /dev/null

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
    local prepare_tests=0
    local force_compile=0
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
            -p|--prepare)
                prepare_tests=1
                shift
                ;;
            -f|--force)
                force_compile=1
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

    # Prepare tests (generate missing instruction files)
    if [[ $prepare_tests -eq 1 ]]; then
        log_info "Preparing tests (generating missing instruction files)..."
        local prepared=0
        local already_exist=0
        local prep_failed=0

        for test_name in "${tests[@]}"; do
            printf "  Preparing %-40s " "$test_name..."

            prep_output=$(prepare_test "$test_name" 2>&1)
            prep_result=$?

            case "$prep_output" in
                EXISTS)
                    echo -e "${BLUE}EXISTS${NC}"
                    already_exist=$((already_exist + 1))
                    ;;
                HAS_ELF)
                    echo -e "${BLUE}HAS_ELF${NC}"
                    already_exist=$((already_exist + 1))
                    ;;
                NO_LIT|UNKNOWN)
                    echo -e "${YELLOW}SKIP${NC}"
                    ;;
                *)
                    if [[ $prep_result -eq 0 ]]; then
                        echo -e "${GREEN}OK${NC}"
                        prepared=$((prepared + 1))
                    else
                        echo -e "${RED}FAILED${NC}"
                        echo "$prep_output" | tail -5
                        prep_failed=$((prep_failed + 1))
                    fi
                    ;;
            esac
        done

        log_info "Prepared: $prepared, Already exist: $already_exist, Failed: $prep_failed"
        echo ""
    fi

    # Compile tests
    log_info "Compiling tests..."
    local compiled=0
    local up_to_date=0
    for test_name in "${tests[@]}"; do
        TOTAL=$((TOTAL + 1))
        printf "  Compiling %-40s " "$test_name..."

        compile_output=$(compile_test "$test_name" "$force_compile" 2>&1)
        compile_result=$?

        if [[ "$compile_output" == "UP-TO-DATE" ]]; then
            echo -e "${BLUE}UP-TO-DATE${NC}"
            up_to_date=$((up_to_date + 1))
        elif [[ $compile_result -eq 0 ]]; then
            echo -e "${GREEN}OK${NC}"
            compiled=$((compiled + 1))
        else
            echo -e "${RED}FAILED${NC}"
            echo "$compile_output" | head -10
            COMPILE_ERRORS=$((COMPILE_ERRORS + 1))
            SKIPPED=$((SKIPPED + 1))
        fi
    done
    if [[ $compiled -gt 0 || $up_to_date -gt 0 ]]; then
        log_info "Compiled: $compiled, Up-to-date: $up_to_date"
    fi
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

            run_test "$test_name" "$verbose"
            local result=$?
            if [[ $result -eq 0 ]]; then
                log_success ""
                PASSED=$((PASSED + 1))
            elif [[ $result -eq 2 ]]; then
                # Already logged skip message
                SKIPPED=$((SKIPPED + 1))
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
