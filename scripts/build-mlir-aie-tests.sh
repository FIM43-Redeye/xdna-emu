#!/bin/bash
# Build mlir-aie test binaries (xclbin + insts.bin) using the Peano toolchain.
#
# This script compiles Peano-compatible mlir-aie tests so the emulator
# can run them for output verification. Tests requiring xchesscc are
# skipped since we use the open-source toolchain only.
#
# Usage:
#   ./scripts/build-mlir-aie-tests.sh           # Build all compatible tests
#   ./scripts/build-mlir-aie-tests.sh --list     # List test categories
#   ./scripts/build-mlir-aie-tests.sh --force    # Rebuild even if artifacts exist
#   ./scripts/build-mlir-aie-tests.sh add_one*   # Build matching tests only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XDNA_EMU_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths -- override via environment if needed
MLIR_AIE_ROOT="${MLIR_AIE_ROOT:-${XDNA_EMU_ROOT}/../mlir-aie}"
MLIR_AIE_BUILD="${MLIR_AIE_BUILD:-${MLIR_AIE_ROOT}/build}"
# Prefer install/bin over build/bin when available (matches activate-npu-env.sh)
if [[ -d "${MLIR_AIE_ROOT}/install/bin" ]]; then
    MLIR_AIE_BIN="${MLIR_AIE_ROOT}/install/bin"
else
    MLIR_AIE_BIN="${MLIR_AIE_BUILD}/bin"
fi
MLIR_AIE_VENV="${MLIR_AIE_ROOT}/ironenv"
# Peano (llvm-aie) location: prefer top-level llvm-aie/install (has runtime
# libs like crt0.o), fall back to ironenv package, allow environment override.
if [[ -z "${PEANO_INSTALL_DIR:-}" ]]; then
    if [[ -d "${XDNA_EMU_ROOT}/../llvm-aie/install/bin" ]]; then
        PEANO_INSTALL_DIR="${XDNA_EMU_ROOT}/../llvm-aie/install"
    else
        PEANO_INSTALL_DIR="${MLIR_AIE_VENV}/lib/python3.13/site-packages/llvm-aie"
    fi
fi
MLIR_AIE_PYTHON="${MLIR_AIE_ROOT}/install/python:${MLIR_AIE_ROOT}/build/python:${MLIR_AIE_ROOT}/my_install/python"

AIECC="${MLIR_AIE_BIN}/aiecc.py"
AIE_OPT="${MLIR_AIE_BIN}/aie-opt"
AIE_TRANSLATE="${MLIR_AIE_BIN}/aie-translate"

# aiecc.py internally calls aie-translate, aie-opt, etc. via subprocess.
# Ensure they are on PATH regardless of whether the environment was activated.
export PATH="${MLIR_AIE_BIN}:${PATH}"

# Source test directory
TEST_SRC="${MLIR_AIE_ROOT}/test/npu-xrt"
# Output directory for built artifacts
BUILD_DIR="${MLIR_AIE_BUILD}/test/npu-xrt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
BUILT=0
SKIPPED=0
FAILED=0
CACHED=0

# Tests that require xchesscc (not buildable with Peano).
#
# Detected by:
#   - Presence of kernel.cc (also caught by auto-detection below)
#   - REQUIRES: valid_xchess_license in run.lit
#   - i8 vector kernels that hit Peano llc legalization gap
#     (G_ADD <4 x s8> -- xchesscc handles this, Peano does not yet)
#
# Note: mlir-aie is built with aie_compile_with_xchesscc=ON by default
# (see aiecc/configure.py). Tests that don't say --no-xchesscc implicitly
# assume xchesscc is available. Our build script forces --no-xchesscc.
XCHESSCC_TESTS=(
    # Explicit kernel.cc (need xchesscc to compile C++ to AIE ELF)
    bd_chain_repeat_on_memtile
    cascade_flows
    loadpdi
    matrix_multiplication_using_cascade
    matrix_transpose
    nd_memcpy_transforms
    runtime_cumsum
    tile_mapped_read
    two_col
    vector_scalar_using_dma
    # i8 vector kernels (Peano llc cannot legalize <4 x s8> G_ADD)
    add_12_i8_using_2d_dma_op_with_padding
    add_21_i8_using_dma_op_with_padding
    ctrl_packet_reconfig
    ctrl_packet_reconfig_1x4_cores
    ctrl_packet_reconfig_4x1_cores
    ctrl_packet_reconfig_elf
    packet_flow
    packet_flow_fanin
    packet_flow_fanout
)

# Tests with no run.lit and no clear build method.
# Parent directories (adjacent_memtile_access, core_dmas, etc.) are not
# listed here -- they are structural containers and are never visited by
# find_test_dirs().
NO_BUILD_TESTS=(
    reconfigure_loadpdi
    reconfigure_loadpdi_persistent_memtile
    vec_mul_event_trace
)

usage() {
    echo "Usage: $0 [OPTIONS] [PATTERN]"
    echo ""
    echo "Build mlir-aie test binaries using Peano (open-source) toolchain."
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help"
    echo "  -l, --list    List tests by category (STD/CTRL/ELF/PYMLIR/CHESS/SKIP)"
    echo "  -f, --force   Force rebuild even if artifacts exist"
    echo "  -v, --verbose Show aiecc.py output"
    echo ""
    echo "PATTERN:"
    echo "  Optional glob pattern to filter tests (e.g., 'add_one*')"
}

is_xchesscc_test() {
    local name="$1"
    for chess in "${XCHESSCC_TESTS[@]}"; do
        if [[ "$name" == "$chess" ]]; then
            return 0
        fi
    done
    return 1
}

is_no_build_test() {
    local name="$1"
    for nb in "${NO_BUILD_TESTS[@]}"; do
        if [[ "$name" == "$nb" ]]; then
            return 0
        fi
    done
    return 1
}

# Find all test directories containing buildable files.
# Returns paths relative to TEST_SRC, sorted alphabetically. Handles both
# flat tests (e.g., "add_one_using_dma") and nested tests inside parent
# directories (e.g., "core_dmas/writebd", "objectfifo_repeat/simple_repeat").
#
# A "test directory" is any directory containing at least one of:
#   aie.mlir, run.lit, kernel.cc, or aie*.py
# Parent-only directories (those with children but no buildable files
# themselves) are naturally excluded.
find_test_dirs() {
    local base="$1"
    find "$base" -type f \( \
        -name "aie.mlir" -o \
        -name "run.lit" -o \
        -name "kernel.cc" -o \
        -name "aie.py" -o \
        -name "aie2.py" -o \
        -name "aie2p.py" \
    \) -exec dirname {} \; | sort -u | while read -r dir; do
        echo "${dir#${base}/}"
    done
}

# Check if a test already has built artifacts
has_artifacts() {
    local build_dir="$1"
    [[ -f "${build_dir}/aie.xclbin" ]] && \
        { [[ -f "${build_dir}/insts.bin" ]] || \
          [[ -f "${build_dir}/aie_run_seq.bin" ]] || \
          [[ -f "${build_dir}/insts.elf" ]] || \
          ls "${build_dir}"/*_insts.bin > /dev/null 2>&1; }
}

# Get the MLIR device name for NPUDEVICE substitution.
#
# mlir-aie tests use NPUDEVICE as a placeholder. The actual device target
# is specified in run.lit via sed commands like:
#   sed 's/NPUDEVICE/npu1_1col/'     (single-column tests)
#   sed 's/NPUDEVICE/npu1/'          (multi-column tests)
#
# We parse run.lit for the NPU1 target. Falls back to npu1_1col if
# no run.lit exists or the pattern is not found.
detect_device_name() {
    local src_dir="$1"
    local run_lit="${src_dir}/run.lit"

    if [[ -f "$run_lit" ]]; then
        # Extract the NPU1 device name from sed substitution.
        # Pattern: sed 's/NPUDEVICE/npu1_1col/' or sed "s/NPUDEVICE/npu1/"
        local device
        device=$(grep -oP "s[/|]NPUDEVICE[/|]\Knpu1[a-z0-9_]*" "$run_lit" | head -1)
        if [[ -n "$device" ]]; then
            echo "$device"
            return
        fi
    fi

    # Default for tests without explicit device in run.lit
    echo "npu1_1col"
}

# Build a standard test (aiecc.py with --no-xchesscc)
build_standard() {
    local name="$1"
    local src_dir="$2"
    local bld_dir="$3"
    local verbose="$4"

    mkdir -p "$bld_dir"

    # Copy and substitute MLIR file
    local mlir_src="${src_dir}/aie.mlir"
    if [[ ! -f "$mlir_src" ]]; then
        echo "NO_MLIR"
        return 1
    fi

    # Substitute device placeholder and write to build dir
    sed "s/NPUDEVICE/$(detect_device_name "$src_dir")/g" "$mlir_src" > "${bld_dir}/aie_arch.mlir"

    # Run aiecc.py
    local aiecc_args=(
        --no-xchesscc --no-xbridge
        --no-aiesim
        --aie-generate-xclbin --xclbin-name=aie.xclbin
        --aie-generate-npu-insts --npu-insts-name=insts.bin
        --no-compile-host
        aie_arch.mlir
    )

    pushd "$bld_dir" > /dev/null
    if [[ "$verbose" == "1" ]]; then
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" 2>&1
    else
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" > build.log 2>&1
    fi
    local result=$?
    popd > /dev/null
    return $result
}

# Build a ctrl_packet test (needs aie-opt overlay generation first)
build_ctrl_packet() {
    local name="$1"
    local src_dir="$2"
    local bld_dir="$3"
    local verbose="$4"

    mkdir -p "$bld_dir"

    local mlir_src="${src_dir}/aie.mlir"
    if [[ ! -f "$mlir_src" ]]; then
        echo "NO_MLIR"
        return 1
    fi

    sed "s/NPUDEVICE/$(detect_device_name "$src_dir")/g" "$mlir_src" > "${bld_dir}/aie_arch.mlir"

    pushd "$bld_dir" > /dev/null

    # Step 1: Generate column control overlay
    "$AIE_OPT" -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" \
        aie_arch.mlir -o aie_overlay.mlir > overlay.log 2>&1
    if [[ $? -ne 0 ]]; then
        popd > /dev/null
        return 1
    fi

    # Step 2: Generate xclbin, ctrlpkt.bin, and aie_run_seq.bin
    local aiecc_args=(
        --no-xchesscc --no-xbridge
        --no-aiesim
        --device-name=main
        --aie-generate-xclbin --xclbin-name=aie.xclbin
        --aie-generate-ctrlpkt --ctrlpkt-name=ctrlpkt.bin
        --aie-generate-npu-insts --npu-insts-name=aie_run_seq.bin
        --no-compile-host
        aie_overlay.mlir
    )

    if [[ "$verbose" == "1" ]]; then
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" 2>&1
    else
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" > build.log 2>&1
    fi
    local result=$?
    popd > /dev/null
    return $result
}

# Build a Python-generated MLIR test
build_pymlir() {
    local name="$1"
    local src_dir="$2"
    local bld_dir="$3"
    local verbose="$4"

    mkdir -p "$bld_dir"

    # Find the Python MLIR generator (typically aie2.py)
    local py_src
    py_src=$(find "$src_dir" -maxdepth 1 -name "aie*.py" | head -1)
    if [[ -z "$py_src" ]]; then
        echo "NO_PY"
        return 1
    fi

    pushd "$bld_dir" > /dev/null

    # Step 1: Generate MLIR from Python.
    # Many aie2.py scripts require command-line arguments (device name,
    # dimensions, etc.). These are documented in # RUN: comments inside
    # the Python file itself (PYMLIR tests typically have no run.lit).
    # Pattern: # RUN: %python %S/aie2.py <args> > ./aie2.mlir
    local py_args=()
    local args_str
    args_str=$(grep -oP '^#\s*RUN:\s*%python\s+%S/aie\d*\.py\s+\K[^>|]+' "$py_src" | head -1 | xargs)
    if [[ -n "$args_str" ]]; then
        read -ra py_args <<< "$args_str"
    fi

    PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
    "${MLIR_AIE_VENV}/bin/python3" "$py_src" "${py_args[@]}" > aie_arch.mlir 2> pymlir.log
    if [[ $? -ne 0 ]] || [[ ! -s aie_arch.mlir ]]; then
        popd > /dev/null
        return 1
    fi

    # Step 2: Build with aiecc.py
    local aiecc_args=(
        --no-xchesscc --no-xbridge
        --no-aiesim
        --aie-generate-xclbin --xclbin-name=aie.xclbin
        --aie-generate-npu-insts --npu-insts-name=insts.bin
        --no-compile-host
        aie_arch.mlir
    )

    if [[ "$verbose" == "1" ]]; then
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" 2>&1
    else
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" > build.log 2>&1
    fi
    local result=$?
    popd > /dev/null
    return $result
}

# Build an ELF-format test
build_elf() {
    local name="$1"
    local src_dir="$2"
    local bld_dir="$3"
    local verbose="$4"

    mkdir -p "$bld_dir"

    local mlir_src="${src_dir}/aie.mlir"
    if [[ ! -f "$mlir_src" ]]; then
        echo "NO_MLIR"
        return 1
    fi

    sed "s/NPUDEVICE/$(detect_device_name "$src_dir")/g" "$mlir_src" > "${bld_dir}/aie_arch.mlir"

    local aiecc_args=(
        --no-xchesscc --no-xbridge
        --no-aiesim
        --aie-generate-xclbin --xclbin-name=aie.xclbin
        --aie-generate-elf --npu-insts-name=insts.elf
        --no-compile-host
        aie_arch.mlir
    )

    pushd "$bld_dir" > /dev/null
    if [[ "$verbose" == "1" ]]; then
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" 2>&1
    else
        PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
        PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
        "${MLIR_AIE_VENV}/bin/python3" "$AIECC" "${aiecc_args[@]}" > build.log 2>&1
    fi
    local result=$?
    popd > /dev/null
    return $result
}

# Build a multi-kernel test by parsing and replaying aiecc.py commands from
# run.lit. Multi-kernel tests have 2+ aiecc.py invocations where later
# passes use --xclbin-input to merge kernels. May also include aie-translate
# for transaction-based reconfiguration (txn tests).
#
# Output: aie.xclbin (renamed from final xclbin), insts.bin, and any extra
# instruction/config files referenced in the build.
build_multi_kernel() {
    local name="$1"
    local src_dir="$2"
    local bld_dir="$3"
    local verbose="$4"

    mkdir -p "$bld_dir"

    local mlir_src="${src_dir}/aie.mlir"
    if [[ ! -f "$mlir_src" ]]; then
        echo "NO_MLIR"
        return 1
    fi

    sed "s/NPUDEVICE/$(detect_device_name "$src_dir")/g" "$mlir_src" > "${bld_dir}/aie_arch.mlir"

    local run_lit="${src_dir}/run.lit"
    pushd "$bld_dir" > /dev/null

    local step=0
    local last_xclbin=""

    # Execute each aiecc.py command from run.lit in sequence
    while IFS= read -r line; do
        # Strip comment prefix and leading whitespace
        line="${line#//}"
        line="${line#"${line%%[![:space:]]*}"}"

        # Skip RUN: prefix and any npu1/npu2 selectors
        line="${line#RUN:}"
        line="${line#"${line%%[![:space:]]*}"}"

        # Handle aiecc.py commands
        if [[ "$line" =~ aiecc\.py ]]; then
            # Extract everything after "aiecc.py "
            local args_str="${line#*aiecc.py }"

            # Substitute %S/ with source dir path
            args_str="${args_str//%S\//${src_dir}/}"

            # Add --no-xchesscc --no-xbridge if not already present
            if [[ ! "$args_str" =~ --no-xchesscc ]]; then
                args_str="--no-xchesscc --no-xbridge ${args_str}"
            fi

            # Track the last xclbin name produced
            if [[ "$args_str" =~ --xclbin-name=([^ ]+) ]]; then
                last_xclbin="${BASH_REMATCH[1]}"
            fi

            step=$((step + 1))
            if [[ "$verbose" == "1" ]]; then
                echo "  [step $step] aiecc.py ${args_str}"
                PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
                PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
                "${MLIR_AIE_VENV}/bin/python3" "$AIECC" ${args_str} 2>&1
            else
                PYTHONPATH="${MLIR_AIE_PYTHON}:${PYTHONPATH:-}" \
                PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" \
                "${MLIR_AIE_VENV}/bin/python3" "$AIECC" ${args_str} >> build.log 2>&1
            fi
            if [[ $? -ne 0 ]]; then
                popd > /dev/null
                return 1
            fi
        fi

        # Handle aie-translate commands (used by txn tests)
        if [[ "$line" =~ aie-translate ]]; then
            local translate_args="${line#*aie-translate }"

            step=$((step + 1))
            if [[ "$verbose" == "1" ]]; then
                echo "  [step $step] aie-translate ${translate_args}"
                "$AIE_TRANSLATE" ${translate_args} 2>&1
            else
                "$AIE_TRANSLATE" ${translate_args} >> build.log 2>&1
            fi
            if [[ $? -ne 0 ]]; then
                popd > /dev/null
                return 1
            fi
        fi
    done < <(grep -E "(aiecc\.py|aie-translate)" "$run_lit" | grep -v "run_on_npu2")

    # Rename the final xclbin to aie.xclbin for consistency
    if [[ -n "$last_xclbin" ]] && [[ "$last_xclbin" != "aie.xclbin" ]] && [[ -f "$last_xclbin" ]]; then
        mv "$last_xclbin" aie.xclbin
    fi

    popd > /dev/null
    return 0
}

# Categorize a test directory
categorize_test() {
    local name="$1"
    local src_dir="$2"

    if is_xchesscc_test "$name"; then
        echo "CHESS"
    elif [[ -f "${src_dir}/kernel.cc" ]]; then
        # Tests with kernel.cc require xchesscc to compile the C++ kernel
        echo "CHESS"
    elif is_no_build_test "$name"; then
        echo "NOBUILD"
    elif [[ -f "${src_dir}/run.lit" ]]; then
        # Multi-kernel: 2+ aiecc.py invocations in run.lit
        local aiecc_count
        aiecc_count=$(grep -c "aiecc\.py" "${src_dir}/run.lit" 2>/dev/null || echo "0")
        if [[ "$aiecc_count" -ge 2 ]]; then
            echo "MULTI"
        elif grep -q "aie-generate-column-control-overlay\|aie-generate-ctrlpkt" "${src_dir}/run.lit"; then
            echo "CTRL"
        elif grep -q "aie-generate-elf" "${src_dir}/run.lit"; then
            echo "ELF"
        else
            echo "STD"
        fi
    elif ls "${src_dir}"/aie*.py > /dev/null 2>&1; then
        echo "PYMLIR"
    else
        echo "UNKNOWN"
    fi
}

# Main
main() {
    local force=0
    local list_only=0
    local verbose=0
    local pattern="*"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            -l|--list) list_only=1; shift ;;
            -f|--force) force=1; shift ;;
            -v|--verbose) verbose=1; shift ;;
            -*) echo "Unknown option: $1"; usage; exit 1 ;;
            *) pattern="$1"; shift ;;
        esac
    done

    # Validate toolchain
    if [[ ! -f "$AIECC" ]]; then
        echo -e "${RED}ERROR:${NC} aiecc.py not found at $AIECC"
        echo "  Build mlir-aie first, or set MLIR_AIE_BUILD."
        exit 1
    fi

    echo "=========================================="
    echo "  mlir-aie Test Binary Builder"
    echo "=========================================="
    echo ""
    echo "Source:    $TEST_SRC"
    echo "Output:   $BUILD_DIR"
    echo "Toolchain: $MLIR_AIE_BIN"
    echo ""

    # Process each test directory.
    # find_test_dirs discovers both flat tests (add_one_using_dma) and nested
    # tests inside parent directories (core_dmas/writebd, objectfifo_repeat/
    # simple_repeat). Names are relative paths from TEST_SRC.
    while IFS= read -r name; do
        local test_dir="${TEST_SRC}/${name}"

        # Pattern filtering (matches against full relative path, so both
        # "add_one*" and "core_dmas/*" work as expected)
        if [[ "$pattern" != "*" ]] && [[ "$name" != $pattern ]]; then
            continue
        fi

        local category
        category=$(categorize_test "$name" "$test_dir")
        local bld_dir="${BUILD_DIR}/${name}"

        if [[ $list_only -eq 1 ]]; then
            printf "  %-6s %s\n" "$category" "$name"
            continue
        fi

        printf "  %-50s " "$name..."

        case "$category" in
            CHESS)
                echo -e "${YELLOW}SKIP${NC} (requires xchesscc)"
                SKIPPED=$((SKIPPED + 1))
                ;;
            NOBUILD)
                echo -e "${YELLOW}SKIP${NC} (no build method)"
                SKIPPED=$((SKIPPED + 1))
                ;;
            UNKNOWN)
                echo -e "${YELLOW}SKIP${NC} (unknown type)"
                SKIPPED=$((SKIPPED + 1))
                ;;
            *)
                # Check cache
                if [[ $force -eq 0 ]] && has_artifacts "$bld_dir"; then
                    echo -e "${BLUE}CACHED${NC}"
                    CACHED=$((CACHED + 1))
                    continue
                fi

                # Build based on category
                local build_fn
                case "$category" in
                    STD) build_fn=build_standard ;;
                    CTRL) build_fn=build_ctrl_packet ;;
                    ELF) build_fn=build_elf ;;
                    PYMLIR) build_fn=build_pymlir ;;
                    MULTI) build_fn=build_multi_kernel ;;
                esac

                if $build_fn "$name" "$test_dir" "$bld_dir" "$verbose"; then
                    echo -e "${GREEN}OK${NC}"
                    BUILT=$((BUILT + 1))
                else
                    echo -e "${RED}FAILED${NC}"
                    if [[ -f "${bld_dir}/build.log" ]]; then
                        tail -3 "${bld_dir}/build.log" | sed 's/^/    /'
                    fi
                    FAILED=$((FAILED + 1))
                fi
                ;;
        esac
    done < <(find_test_dirs "$TEST_SRC")

    if [[ $list_only -eq 1 ]]; then
        exit 0
    fi

    # Summary
    echo ""
    echo "=========================================="
    echo "  Summary"
    echo "=========================================="
    echo "  Built:   $BUILT"
    echo "  Cached:  $CACHED"
    echo "  Skipped: $SKIPPED"
    echo "  Failed:  $FAILED"
    echo "=========================================="

    if [[ $FAILED -gt 0 ]]; then
        echo ""
        echo "Failed test build logs are in: $BUILD_DIR/<test_name>/build.log"
        exit 1
    fi
}

main "$@"
