#!/usr/bin/env bash
# Materialize shim-throughput calibration kernel variants from the n64 template.
#
# Usage: ./scripts/gen-shim-throughput-sweep.sh
#
# Creates mlir-aie/test/npu-xrt/_diag_shim_throughput_sweep/n{N}/ for
# N in {8, 16, 32, 128, 256, 512, 1024}; the n64 source variant must
# already exist and is left untouched.  Substitutions are scoped to
# patterns that contain the BD size as a literal -- "memref<64xi32>",
# "constant 64", "0, 64)", "= 64", and the n64 directory tag in
# headers / cxxopts strings.  memref<32xi32> (the unused dummy arg)
# is left alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_DIR="$EMU_ROOT/../mlir-aie/test/npu-xrt/_diag_shim_throughput_sweep"
SRC="$SWEEP_DIR/n64"

if [[ ! -d "$SRC" ]]; then
  echo "error: source variant $SRC missing; nothing to copy from" >&2
  exit 1
fi

SIZES=(8 16 32 128 256 512 1024)

for N in "${SIZES[@]}"; do
  DST="$SWEEP_DIR/n${N}"
  rm -rf "$DST"
  mkdir -p "$DST"
  for f in aie.mlir run.lit test.cpp; do
    sed -e "s/memref<64xi32>/memref<${N}xi32>/g" \
        -e "s/arith.constant 64 : i64/arith.constant ${N} : i64/g" \
        -e "s/0, 64)/0, ${N})/g" \
        -e "s/BD size N = 64 i32 words/BD size N = ${N} i32 words/g" \
        -e "s/BD size 64 i32 words/BD size ${N} i32 words/g" \
        -e "s/N = 64;/N = ${N};/g" \
        -e "s/shim_throughput_n64/shim_throughput_n${N}/g" \
        "$SRC/$f" > "$DST/$f"
  done
done

echo "materialized:"
for N in "${SIZES[@]}"; do
  echo "  $SWEEP_DIR/n${N}/"
done
