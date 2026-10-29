#!/usr/bin/env bash
# Run the bf16 sub-precision characterizer on real NPU hardware and emulator.
# Extracts full fp32 accumulator via dual SRS to characterize sub-bf16 rounding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"

WORKDIR="${PROJECT_DIR}/build/bf16_sub_precision"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

SRC="${SCRIPT_DIR}/bf16_sub_precision_characterizer.s"
GEN="${SCRIPT_DIR}/bf16_sub_precision_gen.py"
ANALYZE="${SCRIPT_DIR}/bf16_sub_precision_analyze.py"

# 3 test vectors x 128 bytes each = 384 bytes input
# 4 tests x 64 bytes each = 256 bytes output (tests 1&2 share first input)
IN_SIZE=384
OUT_SIZE=512

echo "=== bf16 Sub-Precision Rounding Characterizer ==="

# Step 1: Assemble
echo "Assembling..."
"$LLVM_MC" -triple aie2 -filetype=obj -o "$WORKDIR/kernel.o" "$SRC"

# Step 2: Generate input data
echo "Generating input data..."
python3 "$GEN" "$WORKDIR/input.bin"

# Step 3: Generate aie.mlir
echo "Generating aie.mlir..."
python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_aie_mlir(${IN_SIZE}, ${OUT_SIZE}))
" > "$WORKDIR/aie.mlir"

# Step 4: Package with aiecc.py
echo "Packaging..."
(cd "$WORKDIR" && \
 nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
     --aie-generate-xclbin --xclbin-name=aie.xclbin \
     --aie-generate-npu-insts --npu-insts-name=insts.bin \
     aie.mlir 2>"$WORKDIR/aiecc.log") || {
    echo "aiecc.py failed:"
    cat "$WORKDIR/aiecc.log"
    exit 1
}

XCLBIN="$WORKDIR/aie.xclbin"
INSTS="$WORKDIR/insts.bin"

# Step 5: Get test_host
HOST_BIN="${PROJECT_DIR}/build/isa-tests/test_host"
if [[ ! -x "$HOST_BIN" ]]; then
    echo "test_host not found at $HOST_BIN -- run scripts/isa-test.sh first"
    exit 1
fi

# Step 6: Run on HW
echo "Running on hardware..."
env -u XDNA_EMU "$HOST_BIN" \
    -x "$XCLBIN" -k MLIR_AIE -i "$INSTS" \
    --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
    --in-file "$WORKDIR/input.bin" \
    --out-file "$WORKDIR/hw_output.bin" \
    2>"$WORKDIR/hw.log" || {
    echo "HW run failed:"; cat "$WORKDIR/hw.log"; exit 1
}
echo "HW output: $(wc -c < "$WORKDIR/hw_output.bin") bytes"

# Step 7: Run on emulator
echo "Running on emulator..."
XDNA_EMU=release "$HOST_BIN" \
    -x "$XCLBIN" -k MLIR_AIE -i "$INSTS" \
    --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
    --in-file "$WORKDIR/input.bin" \
    --out-file "$WORKDIR/emu_output.bin" \
    2>"$WORKDIR/emu.log" || {
    echo "EMU run failed:"; cat "$WORKDIR/emu.log"; exit 1
}
echo "EMU output: $(wc -c < "$WORKDIR/emu_output.bin") bytes"

# Step 8: Analyze
echo ""
echo "=== Analysis ==="
python3 "$ANALYZE" "$WORKDIR/input.bin" "$WORKDIR/hw_output.bin" "$WORKDIR/emu_output.bin"
