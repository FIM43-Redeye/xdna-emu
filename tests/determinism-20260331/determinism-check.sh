#!/bin/bash
set -e
ISA_DIR="/home/triple/npu-work/xdna-emu/build/isa-tests"
TEST_HOST="$ISA_DIR/test_host"
TMPDIR="/tmp/claude-1000/determinism"
mkdir -p "$TMPDIR"

TOTAL=0; NONDET=0; ERRORS=0

for batch_dir in "$ISA_DIR"/batch_*/; do
    batch=$(basename "$batch_dir")
    xclbin="$batch_dir/aie.xclbin"
    insts="$batch_dir/insts.bin"
    mlir="$batch_dir/aie.mlir"
    [ -f "$xclbin" ] && [ -f "$insts" ] && [ -f "$mlir" ] || continue

    # Extract sizes from objectfifo declarations
    in_i32=$(grep 'objectfifo @of_in' "$mlir" | grep -oP 'memref<\K[0-9]+' | head -1)
    out_i32=$(grep 'objectfifo @of_out' "$mlir" | grep -oP 'memref<\K[0-9]+' | head -1)
    [ -z "$in_i32" ] || [ -z "$out_i32" ] && continue
    in_bytes=$((in_i32 * 4))
    out_bytes=$((out_i32 * 4))

    TOTAL=$((TOTAL + 1))
    ok=true
    for i in 1 2 3 4 5; do
        if ! env -u XDNA_EMU timeout 30 "$TEST_HOST" \
            -x "$xclbin" -k MLIR_AIE -i "$insts" \
            --in-size "$in_bytes" --out-size "$out_bytes" \
            --out-file "$TMPDIR/${batch}_run${i}.bin" 2>/dev/null; then
            echo "ERROR: $batch run $i failed"
            ERRORS=$((ERRORS + 1))
            ok=false
            break
        fi
    done
    $ok || continue

    for i in 2 3 4 5; do
        if ! diff -q "$TMPDIR/${batch}_run1.bin" "$TMPDIR/${batch}_run${i}.bin" > /dev/null 2>&1; then
            echo "NONDET: $batch (run 1 vs $i differ)"
            NONDET=$((NONDET + 1))
            ok=false
            break
        fi
    done
    $ok && echo "OK: $batch"
done

echo ""
echo "=== SUMMARY ==="
echo "Total batches: $TOTAL"
echo "Deterministic: $((TOTAL - NONDET - ERRORS))"
echo "Nondeterministic: $NONDET"
echo "Errors: $ERRORS"
