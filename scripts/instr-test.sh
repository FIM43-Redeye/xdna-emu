#!/usr/bin/env bash
# scripts/instr-test.sh -- Instruction-level validation harness runner.
#
# Generates single-instruction test kernels, compiles them, runs on real
# NPU and emulator, diffs outputs.
#
# Usage:
#   scripts/instr-test.sh [options]
#
# Compiler selection:
#   --peano          Use Peano compiler (default)
#   --chess          Use Chess compiler (xchesscc)
#   --both           Run both compilers sequentially
#
# Options:
#   --no-hw          Skip hardware runs (EMU-only, no comparison)
#   --no-emu         Skip emulator runs (HW-only baseline)
#   --filter PAT     Only run tests matching PAT (grep -E)
#   --seed N         PRNG seed (default: 42)
#   --compile        Force recompilation
#   -j N             Parallelism for compile + EMU (default: nproc)
#   --generate-only  Only run the generator, skip compile/run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TD_FILE="${PROJECT_DIR}/../llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td"
AIETOOLS_DIR="${PROJECT_DIR}/../aietools"

# mlir-aie paths for host compilation
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
XRT_DIR="/opt/xilinx/xrt"
PEANO_INSTALL_DIR="${PROJECT_DIR}/../llvm-aie/install"

# Defaults
COMPILER="peano"
RUN_HW=true
RUN_EMU=true
FILTER=""
SEED=42
FORCE_COMPILE=false
JOBS=$(nproc)
GENERATE_ONLY=false

# Collect pass-through args for --both mode
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --peano)       COMPILER="peano"; shift ;;
        --chess)       COMPILER="chess"; shift ;;
        --both)        COMPILER="both"; shift ;;
        --no-hw)       RUN_HW=false; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --no-emu)      RUN_EMU=false; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --filter)      FILTER="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --seed)        SEED="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --compile)     FORCE_COMPILE=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        -j)            JOBS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --generate-only) GENERATE_ONLY=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --both: re-invoke self for each compiler, then exit
if [[ "$COMPILER" == "both" ]]; then
    echo "=== Running Peano path ==="
    "$0" --peano "${PASSTHROUGH_ARGS[@]}"
    echo ""
    echo "=== Running Chess path ==="
    "$0" --chess "${PASSTHROUGH_ARGS[@]}"
    exit $?
fi

# Set paths based on compiler choice
OUT_DIR="${PROJECT_DIR}/build/instr-tests-${COMPILER}"
RESULTS_DIR="/tmp/instr-test-results-$(date +%Y%m%d)-${COMPILER}"

echo "=== Instruction-Level Validation Harness ==="
echo "Compiler: $COMPILER"
echo "Out dir:  $OUT_DIR"
echo "Results:  $RESULTS_DIR"
echo ""

# ---- Phase 1: Generate ----
echo "--- Phase 1: Generate ---"
if [[ "$COMPILER" == "chess" ]]; then
    python3 "${PROJECT_DIR}/tools/chess-test-gen.py" \
        --opns-header "${AIETOOLS_DIR}/data/aie_ml/lib/isg/me_chess_opns.h" \
        --types-header "${AIETOOLS_DIR}/data/aie_ml/lib/isg/me_chess_types.h" \
        --out-dir "$OUT_DIR"
else
    python3 "${PROJECT_DIR}/tools/instr-test-gen.py" --td "$TD_FILE" --out-dir "$OUT_DIR"
fi
echo ""

if $GENERATE_ONLY; then
    echo "Generate-only mode. Done."
    exit 0
fi

# Read manifest to get list of generated tests
TESTS=$(python3 -c "
import json, sys
m = json.load(open('${OUT_DIR}/manifest.json'))
for t in m['generated']:
    print(t['name'], t['in_size'], t['out_size'])
")

# Apply filter
if [[ -n "$FILTER" ]]; then
    TESTS=$(echo "$TESTS" | grep -E "$FILTER" || true)
fi

TOTAL=$(echo "$TESTS" | grep -c . || echo 0)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No tests match filter. Done."
    exit 0
fi
echo "Tests to run: $TOTAL"
echo ""

# ---- Phase 2: Compile ----
echo "--- Phase 2: Compile (j=$JOBS) ---"

# Compile shared host harness (once)
HOST_BIN="${OUT_DIR}/test_host"
if $FORCE_COMPILE || [[ ! -f "$HOST_BIN" ]] || [[ "${OUT_DIR}/test_host.cpp" -nt "$HOST_BIN" ]]; then
    echo "Compiling test_host..."
    clang++ "${OUT_DIR}/test_host.cpp" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${TEST_LIB_DIR}/include" \
        -I "${XRT_DIR}/include" \
        -L "${TEST_LIB_DIR}/lib" \
        -L "${XRT_DIR}/lib" \
        -ltest_utils -lxrt_coreutil \
        -lrt -lstdc++
fi

compile_one() {
    local name="$1"
    local test_dir="${OUT_DIR}/${name}"

    # Skip if already compiled (unless --compile)
    if ! $FORCE_COMPILE && [[ -f "${test_dir}/aie.xclbin" ]] && [[ -f "${test_dir}/insts.bin" ]]; then
        return 0
    fi

    echo "  Compiling ${name} [${COMPILER}]..."

    if [[ "$COMPILER" == "chess" ]]; then
        # Compile kernel with Chess
        (cd "$test_dir" && \
            nice -n 19 xchesscc_wrapper aie2 \
                -c kernel.cc -o kernel.o 2>"${test_dir}/chess.log") || {
            echo "  FAIL compile chess: ${name}"
            return 1
        }

        # Compile MLIR -> xclbin + insts.bin (Chess linker)
        (cd "$test_dir" && \
            nice -n 19 aiecc.py --no-aiesim --xchesscc --xbridge \
                --aie-generate-xclbin --xclbin-name=aie.xclbin \
                --aie-generate-npu-insts --npu-insts-name=insts.bin \
                aie.mlir 2>"${test_dir}/aiecc.log") || {
            echo "  FAIL compile aiecc: ${name}"
            return 1
        }
    else
        # Compile kernel with Peano
        (cd "$test_dir" && \
            nice -n 19 "${PEANO_INSTALL_DIR}/bin/clang++" \
                --target=aie2-none-unknown-elf -O2 \
                -c kernel.cc -o kernel.o 2>"${test_dir}/peano.log") || {
            echo "  FAIL compile peano: ${name}"
            return 1
        }

        # Compile MLIR -> xclbin + insts.bin (Peano linker)
        (cd "$test_dir" && \
            nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
                --aie-generate-xclbin --xclbin-name=aie.xclbin \
                --aie-generate-npu-insts --npu-insts-name=insts.bin \
                aie.mlir 2>"${test_dir}/aiecc.log") || {
            echo "  FAIL compile aiecc: ${name}"
            return 1
        }
    fi
}
export -f compile_one
export OUT_DIR FORCE_COMPILE PEANO_INSTALL_DIR COMPILER

echo "$TESTS" | awk '{print $1}' | xargs -P "$JOBS" -I{} bash -c 'compile_one "$1"' _ {}
echo ""

# ---- Phase 3: Run HW ----
mkdir -p "$RESULTS_DIR"

if $RUN_HW; then
    echo "--- Phase 3: Run HW (serial) ---"
    while IFS=' ' read -r name in_size out_size; do
        test_dir="${OUT_DIR}/${name}"
        hw_out="${RESULTS_DIR}/${name}_hw.bin"

        if [[ ! -f "${test_dir}/aie.xclbin" ]]; then
            echo "  SKIP ${name}: not compiled"
            continue
        fi

        "$HOST_BIN" \
            -x "${test_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${test_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$hw_out" 2>/dev/null && \
            echo "  HW OK: ${name}" || \
            echo "  HW FAIL: ${name}"
    done <<< "$TESTS"
    echo ""
fi

# ---- Phase 4: Run EMU ----
if $RUN_EMU; then
    echo "--- Phase 4: Run EMU (j=$JOBS) ---"

    run_emu_one() {
        local name="$1"
        local in_size="$2"
        local out_size="$3"
        local test_dir="${OUT_DIR}/${name}"
        local emu_out="${RESULTS_DIR}/${name}_emu.bin"

        if [[ ! -f "${test_dir}/aie.xclbin" ]]; then
            return 0
        fi

        XDNA_EMU=1 "$HOST_BIN" \
            -x "${test_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${test_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$emu_out" 2>/dev/null && \
            echo "  EMU OK: ${name}" || \
            echo "  EMU FAIL: ${name}"
    }
    export -f run_emu_one
    export HOST_BIN OUT_DIR RESULTS_DIR SEED

    echo "$TESTS" | while IFS=' ' read -r name in_size out_size; do
        printf '%s\0%s\0%s\0' "$name" "$in_size" "$out_size"
    done | xargs -0 -n3 -P "$JOBS" bash -c 'run_emu_one "$1" "$2" "$3"' _
    echo ""
fi

# ---- Phase 5: Compare ----
if $RUN_HW && $RUN_EMU; then
    echo "--- Phase 5: Compare ---"
    PASS=0
    FAIL=0
    SKIP=0
    FAIL_LIST=""

    while IFS=' ' read -r name in_size out_size; do
        hw_out="${RESULTS_DIR}/${name}_hw.bin"
        emu_out="${RESULTS_DIR}/${name}_emu.bin"

        if [[ ! -f "$hw_out" ]] || [[ ! -f "$emu_out" ]]; then
            SKIP=$((SKIP + 1))
            continue
        fi

        if cmp -s "$hw_out" "$emu_out"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            FAIL_LIST="${FAIL_LIST}  DIVERGE: ${name}\n"
            echo "  DIVERGE: ${name}"
        fi
    done <<< "$TESTS"

    echo ""
    echo "=== Results ==="
    echo "PASS: $PASS"
    echo "FAIL: $FAIL"
    echo "SKIP: $SKIP"
    if [[ $FAIL -gt 0 ]]; then
        echo ""
        echo "Divergences:"
        printf '%b' "$FAIL_LIST"
    fi
else
    echo "=== Comparison skipped (need both --hw and --emu) ==="
fi
