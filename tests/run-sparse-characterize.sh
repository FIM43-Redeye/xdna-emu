#!/usr/bin/env bash
# Run the sparse mask characterization test on hardware and emulator.
#
# Usage:
#   tests/run-sparse-characterize.sh [--hw-only] [--emu-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PROJECT_DIR/build/sparse-characterize"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"
# Input: we use p0 as scratch (kernel writes its own data).
# Minimum 16 bytes for q0 mask scratch. Round up to 32.
IN_SIZE=32
OUT_SIZE=1024

RUN_HW=true
RUN_EMU=true
for arg in "$@"; do
    case "$arg" in
        --hw-only) RUN_EMU=false ;;
        --emu-only) RUN_HW=false ;;
    esac
done

mkdir -p "$WORK_DIR"

# Step 1: Assemble
echo "=== Assembling ==="
"$LLVM_MC" -triple=aie2 -filetype=obj \
    "$SCRIPT_DIR/sparse-mask-characterize.s" \
    -o "$WORK_DIR/kernel.o" 2>&1
echo "  OK"

# Step 2: Generate MLIR and package with aiecc.py
echo "=== Packaging ==="
python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_aie_mlir(${IN_SIZE}, ${OUT_SIZE}))
" > "$WORK_DIR/aie.mlir"

(cd "$WORK_DIR" && \
    nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --xclbin-name=aie.xclbin \
        --aie-generate-npu-insts --npu-insts-name=insts.bin \
        aie.mlir 2>"$WORK_DIR/aiecc.log") || {
    echo "FAIL: aiecc.py packaging failed. See $WORK_DIR/aiecc.log"
    tail -20 "$WORK_DIR/aiecc.log"
    exit 1
}
echo "  OK"

# Step 3: Find host binary
ISA_OUT="$PROJECT_DIR/build/isa-tests"
HOST_BIN="${ISA_OUT}/test_host"
if [[ ! -f "$HOST_BIN" ]]; then
    echo "Host binary not found at $HOST_BIN"
    echo "Run scripts/isa-test.sh once first to build it."
    exit 1
fi

# Step 4: Run
if $RUN_HW; then
    echo "=== Running on Hardware ==="
    rc=0
    env -u XDNA_EMU timeout 30 "$HOST_BIN" \
        -x "$WORK_DIR/aie.xclbin" \
        -k MLIR_AIE \
        -i "$WORK_DIR/insts.bin" \
        --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
        --seed 42 \
        --out-file "$WORK_DIR/hw.bin" \
        2>"$WORK_DIR/hw.log" || rc=$?
    if [[ $rc -eq 0 ]] && [[ -f "$WORK_DIR/hw.bin" ]]; then
        echo "  HW OK"
        python3 "$SCRIPT_DIR/sparse-mask-characterize.py" --analyze-hw "$WORK_DIR/hw.bin"
    else
        echo "  HW FAIL (rc=$rc)"
        cat "$WORK_DIR/hw.log"
    fi
fi

if $RUN_EMU; then
    echo "=== Running on Emulator ==="
    EMU_PROFILE="${XDNA_EMU:-release}"
    rc=0
    XDNA_EMU="$EMU_PROFILE" timeout 30 "$HOST_BIN" \
        -x "$WORK_DIR/aie.xclbin" \
        -k MLIR_AIE \
        -i "$WORK_DIR/insts.bin" \
        --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
        --seed 42 \
        --out-file "$WORK_DIR/emu.bin" \
        2>"$WORK_DIR/emu.log" || rc=$?
    if [[ $rc -eq 0 ]] && [[ -f "$WORK_DIR/emu.bin" ]]; then
        echo "  EMU OK"
        python3 "$SCRIPT_DIR/sparse-mask-characterize.py" --analyze-emu "$WORK_DIR/emu.bin"
    else
        echo "  EMU FAIL (rc=$rc)"
        cat "$WORK_DIR/emu.log"
    fi
fi

echo ""
echo "=== Done ==="
