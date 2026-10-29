#!/usr/bin/env bash
# Run the add32 single-add oracle for a given input.bin and backend.
# Usage: run_add32_oracle.sh <input.bin> <interp|aiesim> [out.txt]
# Prints the 32-lane output (decimal u16, one per line) to stdout via out.txt.
set -euo pipefail

INPUT="${1:?input.bin path}"
BACKEND="${2:-aiesim}"
OUTTXT="${3:-/tmp/add32_out.txt}"

EMU_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRIV_DIR=${NPU_WORK_DIR:-$(dirname "$EMU_ROOT")}/xdna-emu-priv
BRIDGE_DIR=$PRIV_DIR/aiesim-bridge/build
NPU_WORK=${NPU_WORK_DIR:-$(dirname "$EMU_ROOT")}
D=$NPU_WORK/mlir-aie/build/test/npu-xrt/vec_add32_oracle/peano
source "$NPU_WORK/toolchain-build/activate-npu-env.sh" >/dev/null 2>&1

cp -f "$INPUT" "$D/input.bin"
cd "$D"
rm -f out.txt

if [[ "$BACKEND" == "aiesim" ]]; then
  env XDNA_EMU=1 XDNA_EMU_RUNTIME=debug \
      XDNA_BACKEND=aiesim \
      ADD32_INPUT="$D/input.bin" \
      XDNA_AIESIM_BRIDGE=$BRIDGE_DIR/libxdna_aiesim_bridge.so \
      XDNA_AIESIM_DEVICE_JSON=$PRIV_DIR/aiesim-bridge/device/NPU1.json \
      XDNA_AIESIM_NATIVE_GEOMETRY=1 \
      LD_LIBRARY_PATH="$XILINX_VITIS_AIETOOLS/lib/lnx64.o:$BRIDGE_DIR:${LD_LIBRARY_PATH:-}" \
      ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin >/tmp/add32_run.log 2>&1 || { tail -5 /tmp/add32_run.log; exit 1; }
else
  env XDNA_EMU=1 XDNA_EMU_RUNTIME=debug ADD32_INPUT="$D/input.bin" \
      ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin >/tmp/add32_run.log 2>&1 || { tail -5 /tmp/add32_run.log; exit 1; }
fi

cp -f out.txt "$OUTTXT"
cat "$OUTTXT"
