#!/usr/bin/env bash
# Run the bf16 element-wise characterizer on real NPU hardware and emulator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"

WORKDIR="${PROJECT_DIR}/build/bf16_elemwise_char"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

SRC="${SCRIPT_DIR}/bf16_elemwise_characterizer.s"
ANALYZE="${SCRIPT_DIR}/bf16_elemwise_analyze.py"

# Input: 256 bytes (4 x 64-byte patterns)
# Output: 192 bytes (3 tests x 2 SRS passes x 32 bytes)
IN_SIZE=256
OUT_SIZE=192

echo "=== bf16 Element-Wise Characterizer ==="

# Step 1: Assemble
echo "Assembling..."
"$LLVM_MC" -triple aie2 -filetype=obj -o "$WORKDIR/kernel.o" "$SRC"

# Step 2: Generate input data
echo "Generating input data..."
python3 -c "
import struct, sys

# Sidon set: all pairwise sums unique, all values bf16-exact.
SIDON = [1, 2, 4, 8, 13, 21, 31, 45, 66, 81, 97, 123, 148, 182,
         204, 252, 290, 364, 410, 482, 536, 636, 788, 876, 916,
         1080, 1288, 1456, 1640, 1896, 2032, 2400]
assert len(SIDON) == 32

def f32_to_bf16(val):
    bits = struct.unpack('>I', struct.pack('>f', float(val)))[0]
    return bits >> 16

def bf16_pack(values):
    return b''.join(struct.pack('<H', f32_to_bf16(v)) for v in values)

# Layout: [Sidon][ones][ones][Sidon]
pattern_sidon = bf16_pack(SIDON)
pattern_ones = bf16_pack([1.0] * 32)
data = pattern_sidon + pattern_ones + pattern_ones + pattern_sidon
assert len(data) == 256

sys.stdout.buffer.write(data)
" > "$WORKDIR/input.bin"
echo "Input: $(wc -c < "$WORKDIR/input.bin") bytes"

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
XDNA_EMU=debug "$HOST_BIN" \
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
python3 "$ANALYZE" "$WORKDIR/hw_output.bin" "$WORKDIR/emu_output.bin"
