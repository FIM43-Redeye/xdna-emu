#!/usr/bin/env bash
# Run the VLDB_4x characterizer on real NPU hardware and emulator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"
MLIR_AIE="${HOME}/npu-work/mlir-aie"

WORKDIR="${PROJECT_DIR}/build/vldb_4x_char"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

SRC="${SCRIPT_DIR}/vldb_4x_characterizer.s"
ANALYZE="${SCRIPT_DIR}/vldb_4x_analyze.py"

IN_SIZE=256
OUT_SIZE=256
SEED=42

echo "=== VLDB_4x Characterizer ==="

# Step 1: Assemble
echo "Assembling..."
"$LLVM_MC" -triple aie2 -filetype=obj -o "$WORKDIR/kernel.o" "$SRC"

# Step 2: Generate aie.mlir
echo "Generating aie.mlir..."
python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_aie_mlir(${IN_SIZE}, ${OUT_SIZE}))
" > "$WORKDIR/aie.mlir"

# Step 3: Package with aiecc.py
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

# Step 4: Get test_host
HOST_BIN="${PROJECT_DIR}/build/isa-tests/test_host"
if [[ ! -x "$HOST_BIN" ]]; then
    echo "test_host not found at $HOST_BIN -- run scripts/isa-test.sh first"
    exit 1
fi

# Step 5: Run on HW
echo "Running on hardware..."
env -u XDNA_EMU "$HOST_BIN" \
    -x "$XCLBIN" -k MLIR_AIE -i "$INSTS" \
    --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
    --seed "$SEED" --out-file "$WORKDIR/hw_output.bin" \
    2>"$WORKDIR/hw.log" || {
    echo "HW run failed:"; cat "$WORKDIR/hw.log"; exit 1
}
echo "HW output: $(wc -c < "$WORKDIR/hw_output.bin") bytes"

# Step 6: Run on emulator
echo "Running on emulator..."
XDNA_EMU=debug "$HOST_BIN" \
    -x "$XCLBIN" -k MLIR_AIE -i "$INSTS" \
    --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
    --seed "$SEED" --out-file "$WORKDIR/emu_output.bin" \
    2>"$WORKDIR/emu.log" || {
    echo "EMU run failed:"; cat "$WORKDIR/emu.log"; exit 1
}
echo "EMU output: $(wc -c < "$WORKDIR/emu_output.bin") bytes"

# Step 7: Analyze
echo ""
echo "=== Analysis ==="
python3 "$ANALYZE" "$WORKDIR/hw_output.bin" "$WORKDIR/emu_output.bin"
