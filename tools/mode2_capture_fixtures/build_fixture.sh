#!/usr/bin/env bash
# Build one mode-2 capture fixture end to end.
#
# For a given fixture directory (containing kernel.cc + aie.mlir):
#   1. Compile kernel.cc with peano clang to kernel.o (target aie2)
#   2. Materialize aie_arch.mlir from aie.mlir with NPUDEVICE -> npu1_1col
#   3. Run aiecc to produce baseline build/baseline/{aie.xclbin, insts.bin}
#   4. Inject mode-2 trace ops -> build/aie_traced.mlir
#   5. Run aiecc on traced MLIR -> build/traced/{aie.xclbin, insts.bin}
#
# The two xclbins are independent artifacts. The baseline pair is useful
# for sanity-running the kernel without trace overhead; the traced pair
# is what gets handed to bridge-trace-runner for HW capture.
#
# Usage:
#   build_fixture.sh <fixture_dir>
#
# Environment overrides (rare):
#   PEANO_INSTALL_DIR (default: /home/triple/npu-work/llvm-aie/install)
#   AIECC             (default: aiecc.py from the activated environment)

set -euo pipefail

PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-/home/triple/npu-work/llvm-aie/install}"
AIECC="${AIECC:-aiecc.py}"
INJECT="$(dirname "$(realpath "$0")")/../mlir-trace-inject.py"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <fixture_dir>" >&2
  exit 64
fi

FIXTURE_DIR="$(realpath "$1")"
if [[ ! -f "$FIXTURE_DIR/kernel.cc" || ! -f "$FIXTURE_DIR/aie.mlir" ]]; then
  echo "error: $FIXTURE_DIR is missing kernel.cc or aie.mlir" >&2
  exit 65
fi

BUILD_DIR="$FIXTURE_DIR/build"
BASELINE_DIR="$BUILD_DIR/baseline"
TRACED_DIR="$BUILD_DIR/traced"

rm -rf "$BUILD_DIR"
mkdir -p "$BASELINE_DIR" "$TRACED_DIR"

echo "[1/5] peano clang $FIXTURE_DIR/kernel.cc -> $BUILD_DIR/kernel.o"
"$PEANO_INSTALL_DIR/bin/clang" \
  --target=aie2-none-unknown-elf -O2 \
  -c "$FIXTURE_DIR/kernel.cc" \
  -o "$BUILD_DIR/kernel.o"

# aiecc looks for kernel.o relative to its working directory (because the
# link_with attribute in the MLIR is a bare filename). Stage a copy in
# each per-aiecc workdir so both invocations can find it.
cp "$BUILD_DIR/kernel.o" "$BASELINE_DIR/kernel.o"
cp "$BUILD_DIR/kernel.o" "$TRACED_DIR/kernel.o"

echo "[2/5] materialize $BASELINE_DIR/aie_arch.mlir (NPUDEVICE -> npu1_1col)"
sed 's/NPUDEVICE/npu1_1col/g' "$FIXTURE_DIR/aie.mlir" \
  > "$BASELINE_DIR/aie_arch.mlir"

echo "[3/5] aiecc baseline -> $BASELINE_DIR/{aie.xclbin, insts.bin}"
( cd "$BASELINE_DIR" && \
  "$AIECC" \
    --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --xclbin-name=aie.xclbin \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    ./aie_arch.mlir )

echo "[4/5] mode-2 trace inject -> $TRACED_DIR/aie_arch.mlir"
python3 "$INJECT" \
  --trace-mode inst_exec \
  --input "$BASELINE_DIR/aie_arch.mlir" \
  --out   "$TRACED_DIR/aie_arch.mlir"

echo "[5/5] aiecc traced -> $TRACED_DIR/{aie.xclbin, insts.bin}"
( cd "$TRACED_DIR" && \
  "$AIECC" \
    --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --xclbin-name=aie.xclbin \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    ./aie_arch.mlir )

echo
echo "done. artifacts:"
echo "  baseline xclbin: $BASELINE_DIR/aie.xclbin"
echo "  baseline insts:  $BASELINE_DIR/insts.bin"
echo "  traced xclbin:   $TRACED_DIR/aie.xclbin"
echo "  traced insts:    $TRACED_DIR/insts.bin"
echo "  kernel.o:        $BUILD_DIR/kernel.o (disassemble with llvm-objdump -d)"
