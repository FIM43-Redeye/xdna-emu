#!/bin/bash
# Run a traced test with all 4 trace modes, capture raw traces, and decode.
# Usage: ./run_trace_experiment.sh <test_name> [num_runs]
#
# Prerequisites:
#   - Chess build exists: mlir-aie/build/test/npu-xrt/<test>/chess/{insts.bin,aie.xclbin}
#   - Traced MLIR exists: mlir-aie/build/test/npu-xrt/<test>/traced/aie_traced.mlir
#   - test_traced binary compiled (auto-compiled if missing)
#
# Results are stored persistently under xdna-emu/build/experiments/trace-modes/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_NAME="${1:-add_one_using_dma}"
NUM_RUNS="${2:-2}"

MLIR_AIE="/home/triple/npu-work/mlir-aie"
CHESS_DIR="$MLIR_AIE/build/test/npu-xrt/$TEST_NAME/chess"
TRACED_DIR="$MLIR_AIE/build/test/npu-xrt/$TEST_NAME/traced"
RESULTS_BASE="/home/triple/npu-work/xdna-emu/build/experiments/trace-modes"
RESULTS="$RESULTS_BASE/$TEST_NAME"

# Verify prerequisites
for f in "$CHESS_DIR/insts.bin" "$CHESS_DIR/aie.xclbin" "$TRACED_DIR/aie_traced.mlir"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

echo "=== Trace Mode Experiment: $TEST_NAME ==="
echo "  Runs per mode: $NUM_RUNS"
echo "  Results: $RESULTS/"
echo ""

# Step 1: Compile traced MLIR if needed
if [ ! -f "$TRACED_DIR/insts.bin" ] || [ ! -f "$TRACED_DIR/traced.xclbin" ]; then
    echo "--- Compiling traced MLIR ---"
    cd "$TRACED_DIR"
    nice -n 19 aiecc.py --xchesscc --no-aiesim \
        --aie-generate-npu-insts --aie-generate-xclbin \
        --no-compile-host \
        --xclbin-name="traced.xclbin" \
        --npu-insts-name="insts.bin" \
        aie_traced.mlir 2>&1 | tail -5
    echo ""
fi

# Step 2: Compile test binary if needed
if [ ! -f "$TRACED_DIR/test_traced" ]; then
    echo "--- Compiling test binary ---"
    # Use the test_traced.cpp from add_one_using_dma as template
    TEMPLATE="$MLIR_AIE/build/test/npu-xrt/add_one_using_dma/traced/test_traced.cpp"
    if [ -f "$TEMPLATE" ]; then
        cp "$TEMPLATE" "$TRACED_DIR/test_traced.cpp"
    else
        echo "ERROR: No test_traced.cpp template found"
        exit 1
    fi
    g++ -O2 -o "$TRACED_DIR/test_traced" "$TRACED_DIR/test_traced.cpp" \
        -I"$MLIR_AIE/install/runtime_lib/x86_64/test_lib/include" \
        -I/opt/xilinx/xrt/include \
        -L/opt/xilinx/xrt/lib \
        -L"$MLIR_AIE/install/runtime_lib/x86_64/test_lib/lib" \
        -lxrt_coreutil -ltest_utils -lpthread 2>&1
    echo "  Compiled test binary"
    echo ""
fi

# Step 3: Disassemble ELF
ELF=$(find "$CHESS_DIR" -name "main_core_0_2.elf" 2>/dev/null | head -1)
if [ -n "$ELF" ]; then
    mkdir -p "$RESULTS"
    /home/triple/npu-work/llvm-aie/install/bin/llvm-objdump -d "$ELF" > "$RESULTS/disasm.txt" 2>&1
    branches=$(grep -cE "\bj[a-z]*\b|\bret\b" "$RESULTS/disasm.txt" || true)
    echo "ELF: $ELF"
    echo "Branch instructions: $branches"
    echo ""
fi

# Step 4: Patch trace modes
mkdir -p "$RESULTS/patched"
python3 "$SCRIPT_DIR/patch_trace_mode.py" \
    "$TRACED_DIR/insts.bin" "$RESULTS/patched"
echo ""

# Step 5: Run all 4 modes
for mode in 0 1 2 3; do
    case $mode in
        0) name="event_time" ;;
        1) name="event_pc" ;;
        2) name="execution" ;;
        3) name="reserved_11" ;;
    esac

    insts="$RESULTS/patched/insts_mode${mode}_${name}.bin"

    for run in $(seq 1 "$NUM_RUNS"); do
        outdir="$RESULTS/mode${mode}_${name}/run${run}"
        mkdir -p "$outdir"

        XDNA_TRACE_DIR="$outdir" "$TRACED_DIR/test_traced" \
            -x "$TRACED_DIR/traced.xclbin" \
            -i "$insts" \
            -k MLIR_AIE \
            -v 0 > "$outdir/stdout.txt" 2>&1

        result=$(grep -c "PASS" "$outdir/stdout.txt" || true)
        trace_bytes=$(python3 -c "
d=open('$outdir/trace_raw.bin','rb').read()
for i in range(len(d)-1,-1,-1):
    if d[i]!=0: print(i+1); break
else: print(0)
" 2>/dev/null || echo "?")
        status=$([ "$result" -gt 0 ] && echo "PASS" || echo "FAIL")
        echo "  Mode $mode ($name) run$run: $status  trace=$trace_bytes bytes"
    done
done

# Step 6: Consistency and decode
echo ""
python3 "$SCRIPT_DIR/decode_trace_experiment.py" "$RESULTS" "$NUM_RUNS"
