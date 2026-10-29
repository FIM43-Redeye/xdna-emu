#!/usr/bin/env bash
# Run the sparse MAC probe test on hardware and emulator.
# Usage: tests/run-sparse-mac-probe.sh [--hw-only] [--emu-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PROJECT_DIR/build/sparse-mac-probe"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"
IN_SIZE=128
OUT_SIZE=128

RUN_HW=true
RUN_EMU=true
for arg in "$@"; do
    case "$arg" in
        --hw-only) RUN_EMU=false ;;
        --emu-only) RUN_HW=false ;;
    esac
done

mkdir -p "$WORK_DIR"

echo "=== Assembling ==="
"$LLVM_MC" -triple=aie2 -filetype=obj \
    "$SCRIPT_DIR/sparse-mac-probe.s" \
    -o "$WORK_DIR/kernel.o" 2>&1
echo "  OK"

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

HOST_BIN="$PROJECT_DIR/build/isa-tests/test_host"
if [[ ! -f "$HOST_BIN" ]]; then
    echo "Host binary not found at $HOST_BIN"
    exit 1
fi

analyze() {
    local label="$1" binfile="$2"
    python3 -c "
import struct, sys
with open('$binfile', 'rb') as f:
    data = f.read()
print(f'$label output ({len(data)} bytes):')
for test in range(2):
    off = test * 64
    vals = struct.unpack_from('<32h', data, off)
    label = ['Identity (A=1,B=1)', 'Counting (B=1..8)'][test]
    print(f'  Test {test+1} ({label}):')
    print(f'    Even lanes: {[vals[i] for i in range(0,32,2)]}')
    print(f'    Odd lanes:  {[vals[i] for i in range(1,32,2)]}')
"
}

if $RUN_HW; then
    echo "=== Running on Hardware ==="
    rc=0
    env -u XDNA_EMU timeout 30 "$HOST_BIN" \
        -x "$WORK_DIR/aie.xclbin" -k MLIR_AIE \
        -i "$WORK_DIR/insts.bin" \
        --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
        --seed 42 --out-file "$WORK_DIR/hw.bin" \
        2>"$WORK_DIR/hw.log" || rc=$?
    if [[ $rc -eq 0 ]] && [[ -f "$WORK_DIR/hw.bin" ]]; then
        echo "  HW OK"
        analyze "HW" "$WORK_DIR/hw.bin"
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
        -x "$WORK_DIR/aie.xclbin" -k MLIR_AIE \
        -i "$WORK_DIR/insts.bin" \
        --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
        --seed 42 --out-file "$WORK_DIR/emu.bin" \
        2>"$WORK_DIR/emu.log" || rc=$?
    if [[ $rc -eq 0 ]] && [[ -f "$WORK_DIR/emu.bin" ]]; then
        echo "  EMU OK"
        analyze "EMU" "$WORK_DIR/emu.bin"
    else
        echo "  EMU FAIL (rc=$rc)"
        cat "$WORK_DIR/emu.log"
    fi
fi

# Compare if both exist
if [[ -f "$WORK_DIR/hw.bin" ]] && [[ -f "$WORK_DIR/emu.bin" ]]; then
    echo ""
    if cmp -s "$WORK_DIR/hw.bin" "$WORK_DIR/emu.bin"; then
        echo "=== MATCH: HW and EMU outputs are identical ==="
    else
        echo "=== MISMATCH: HW and EMU outputs differ ==="
        python3 -c "
import struct
with open('$WORK_DIR/hw.bin', 'rb') as f: hw = f.read()
with open('$WORK_DIR/emu.bin', 'rb') as f: emu = f.read()
for test in range(2):
    off = test * 64
    hw_vals = struct.unpack_from('<32h', hw, off)
    emu_vals = struct.unpack_from('<32h', emu, off)
    label = ['Identity', 'Counting'][test]
    diffs = [(i, emu_vals[i], hw_vals[i]) for i in range(32) if emu_vals[i] != hw_vals[i]]
    if diffs:
        print(f'Test {test+1} ({label}): {len(diffs)} lanes differ')
        for lane, e, h in diffs[:8]:
            print(f'  lane {lane:2d}: EMU={e:7d}  HW={h:7d}')
    else:
        print(f'Test {test+1} ({label}): MATCH')
"
    fi
fi

echo ""
echo "=== Done ==="
