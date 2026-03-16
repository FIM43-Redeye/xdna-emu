#!/usr/bin/env bash
# scripts/isa-test.sh -- ISA-level validation harness runner.
#
# Generates assembly test batches from aie2-isa.json, assembles with llvm-mc,
# packages with aiecc.py, runs on real NPU and emulator, diffs outputs.
#
# This tests raw ISA instruction behavior (assembly level), complementing
# scripts/instr-test.sh which tests intrinsic-level behavior (C++ level).
#
# Usage:
#   scripts/isa-test.sh [options]
#
# Options:
#   --no-hw          Skip hardware runs (EMU-only, no comparison)
#   --no-emu         Skip emulator runs (HW-only baseline)
#   --seed N         PRNG seed (default: 42)
#   --compile        Force recompilation
#   -j N             Parallelism for compile + EMU (default: nproc)
#   --generate-only  Only run the generator, skip compile/run
#   --filter PAT     Only run batches matching PAT (grep -E on batch filename)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISA_JSON="${PROJECT_DIR}/tools/aie2-isa.json"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"
PEANO_INSTALL_DIR="${HOME}/npu-work/llvm-aie/install"

# mlir-aie paths for host compilation and aiecc.py
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
XRT_DIR="/opt/xilinx/xrt"

# Defaults
RUN_HW=true
RUN_EMU=true
SEED=42
FORCE_COMPILE=false
JOBS=$(nproc)
GENERATE_ONLY=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-hw)         RUN_HW=false; shift ;;
        --no-emu)        RUN_EMU=false; shift ;;
        --seed)          SEED="$2"; shift 2 ;;
        --compile)       FORCE_COMPILE=true; shift ;;
        -j)              JOBS="$2"; shift 2 ;;
        --generate-only) GENERATE_ONLY=true; shift ;;
        --filter)        FILTER="$2"; shift 2 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUT_DIR="${PROJECT_DIR}/build/isa-tests"
RESULTS_DIR="/tmp/isa-test-results-$(date +%Y%m%d)"

echo "=== ISA-Level Validation Harness ==="
echo "ISA JSON: $ISA_JSON"
echo "Out dir:  $OUT_DIR"
echo "Results:  $RESULTS_DIR"
echo ""

# ---- Phase 1: Generate ----
echo "--- Phase 1: Generate ---"
python3 "${PROJECT_DIR}/tools/isa-test-gen.py" \
    --isa-json "$ISA_JSON" \
    --out-dir "$OUT_DIR"
echo ""

if $GENERATE_ONLY; then
    echo "Generate-only mode. Done."
    exit 0
fi

# Read manifest to get batch list.
MANIFEST="${OUT_DIR}/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest.json not found at $MANIFEST"
    exit 1
fi

BATCH_INFO=$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
for b in m['batches']:
    print(b['batch_index'], b['filename'], b['in_size'], b['out_size'])
")

# Apply filter if specified.
if [[ -n "$FILTER" ]]; then
    BATCH_INFO=$(echo "$BATCH_INFO" | grep -E "$FILTER" || true)
fi

TOTAL=$(echo "$BATCH_INFO" | grep -c . || echo 0)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No batches match filter. Done."
    exit 0
fi
echo "Batches to process: $TOTAL"
echo ""

# ---- Phase 2: Assemble ----
echo "--- Phase 2: Assemble (llvm-mc) ---"

if [[ ! -x "$LLVM_MC" ]]; then
    echo "ERROR: llvm-mc not found at $LLVM_MC"
    exit 1
fi

assemble_one() {
    local batch_idx="$1"
    local filename="$2"
    local s_path="${OUT_DIR}/${filename}"
    local o_path="${s_path%.s}.o"

    # Skip if already assembled (unless --compile).
    if ! $FORCE_COMPILE && [[ -f "$o_path" ]] && [[ "$o_path" -nt "$s_path" ]]; then
        return 0
    fi

    nice -n 19 "$LLVM_MC" --triple=aie2 --filetype=obj -o "$o_path" "$s_path" 2>"${s_path%.s}.mc.log" && \
        echo "  ASM OK: ${filename}" || \
        echo "  ASM FAIL: ${filename} (see ${s_path%.s}.mc.log)"
}
export -f assemble_one
export OUT_DIR FORCE_COMPILE LLVM_MC

echo "$BATCH_INFO" | awk '{print $1, $2}' | \
    xargs -P "$JOBS" -n2 bash -c 'assemble_one "$1" "$2"' _
echo ""

# ---- Phase 3: Link + Package ----
echo "--- Phase 3: Link + Package (aiecc.py) ---"

# Generate shared test_host.cpp from instr-test-gen.py.
HOST_CPP="${OUT_DIR}/test_host.cpp"
if [[ ! -f "$HOST_CPP" ]]; then
    python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_test_host_cpp())
" > "$HOST_CPP"
fi

# Compile shared host binary (once).
HOST_BIN="${OUT_DIR}/test_host"
if $FORCE_COMPILE || [[ ! -f "$HOST_BIN" ]] || [[ "$HOST_CPP" -nt "$HOST_BIN" ]]; then
    echo "Compiling test_host..."
    clang++ "$HOST_CPP" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${TEST_LIB_DIR}/include" \
        -I "${XRT_DIR}/include" \
        -L "${TEST_LIB_DIR}/lib" \
        -L "${XRT_DIR}/lib" \
        -ltest_utils -lxrt_coreutil \
        -lrt -lstdc++
fi

package_one() {
    local batch_idx="$1"
    local filename="$2"
    local in_size="$3"
    local out_size="$4"
    local batch_dir="${OUT_DIR}/batch_${batch_idx}"
    local o_path="${OUT_DIR}/${filename%.s}.o"

    if [[ ! -f "$o_path" ]]; then
        echo "  SKIP batch_${batch_idx}: assembly failed"
        return 0
    fi

    # Skip if already packaged (unless --compile).
    if ! $FORCE_COMPILE && [[ -f "${batch_dir}/aie.xclbin" ]] && [[ -f "${batch_dir}/insts.bin" ]]; then
        return 0
    fi

    mkdir -p "$batch_dir"
    cp "$o_path" "${batch_dir}/kernel.o"

    # Generate aie.mlir for this batch's buffer sizes.
    python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_aie_mlir(${in_size}, ${out_size}))
" > "${batch_dir}/aie.mlir"

    # Run aiecc.py (Peano mode, no Chess).
    (cd "$batch_dir" && \
        nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
            --aie-generate-xclbin --xclbin-name=aie.xclbin \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie.mlir 2>"${batch_dir}/aiecc.log") && \
        echo "  PKG OK: batch_${batch_idx}" || \
        echo "  PKG FAIL: batch_${batch_idx} (see ${batch_dir}/aiecc.log)"
}
export -f package_one
export HOST_BIN PROJECT_DIR

echo "$BATCH_INFO" | while IFS=' ' read -r idx filename in_size out_size; do
    package_one "$idx" "$filename" "$in_size" "$out_size"
done
echo ""

# ---- Phase 4: Run HW ----
mkdir -p "$RESULTS_DIR"

if $RUN_HW; then
    echo "--- Phase 4: Run HW (serial) ---"
    while IFS=' ' read -r idx filename in_size out_size; do
        batch_dir="${OUT_DIR}/batch_${idx}"
        hw_out="${RESULTS_DIR}/batch_${idx}_hw.bin"

        if [[ ! -f "${batch_dir}/aie.xclbin" ]]; then
            echo "  SKIP batch_${idx}: not packaged"
            continue
        fi

        "$HOST_BIN" \
            -x "${batch_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${batch_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$hw_out" 2>/dev/null && \
            echo "  HW OK: batch_${idx}" || \
            echo "  HW FAIL: batch_${idx}"
    done <<< "$BATCH_INFO"
    echo ""
fi

# ---- Phase 5: Run EMU ----
if $RUN_EMU; then
    echo "--- Phase 5: Run EMU (j=$JOBS) ---"

    run_emu_one() {
        local idx="$1"
        local in_size="$2"
        local out_size="$3"
        local batch_dir="${OUT_DIR}/batch_${idx}"
        local emu_out="${RESULTS_DIR}/batch_${idx}_emu.bin"

        if [[ ! -f "${batch_dir}/aie.xclbin" ]]; then
            return 0
        fi

        XDNA_EMU=1 "$HOST_BIN" \
            -x "${batch_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${batch_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$emu_out" 2>/dev/null && \
            echo "  EMU OK: batch_${idx}" || \
            echo "  EMU FAIL: batch_${idx}"
    }
    export -f run_emu_one
    export HOST_BIN OUT_DIR RESULTS_DIR SEED

    echo "$BATCH_INFO" | while IFS=' ' read -r idx filename in_size out_size; do
        printf '%s\0%s\0%s\0' "$idx" "$in_size" "$out_size"
    done | xargs -0 -n3 -P "$JOBS" bash -c 'run_emu_one "$1" "$2" "$3"' _
    echo ""
fi

# ---- Phase 6: Compare ----
if $RUN_HW && $RUN_EMU; then
    echo "--- Phase 6: Compare ---"
    PASS=0
    FAIL=0
    SKIP=0
    FAIL_LIST=""

    while IFS=' ' read -r idx filename in_size out_size; do
        hw_out="${RESULTS_DIR}/batch_${idx}_hw.bin"
        emu_out="${RESULTS_DIR}/batch_${idx}_emu.bin"

        if [[ ! -f "$hw_out" ]] || [[ ! -f "$emu_out" ]]; then
            SKIP=$((SKIP + 1))
            continue
        fi

        if cmp -s "$hw_out" "$emu_out"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            FAIL_LIST="${FAIL_LIST}  DIVERGE: batch_${idx}\n"
            echo "  DIVERGE: batch_${idx}"
        fi
    done <<< "$BATCH_INFO"

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
    echo "=== Comparison skipped (need both HW and EMU) ==="
fi
