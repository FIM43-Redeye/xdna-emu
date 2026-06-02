#!/usr/bin/env bash
# build-hal-validate.sh -- Build the HAL-driven validation harness: run a real
# xaiengine config+run+check sequence (libxaienginecdo SIM backend) against our
# in-process bridge, proving it drives the live aiesim cluster end to end.
#
# This is a VALIDATION harness, separate from the bridge .so: it links the HAL
# (libxaienginecdo) + our SystemC bridge components into one executable with its
# own sc_main. Requires aietools + the mlir-aie xaiengine runtime lib.
#
#   source toolchain-build/activate-npu-env.sh
#   ./scripts/build-hal-validate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$EMU_DIR/aiesim-bridge/src"
VAL="$EMU_DIR/aiesim-bridge/validation"
OUT="$EMU_DIR/aiesim-bridge/build/hal_validate"
NPU_WORK="$(dirname "$EMU_DIR")"

AT="${XILINX_VITIS_AIETOOLS:?source toolchain-build/activate-npu-env.sh first}"
SC_INC="$AT/data/osci_systemc/include"
SC_MAIN="$AT/data/osci_systemc/sc_main"
SC_LIB="$AT/lib/lnx64.o"
XTLM_INC1="$AT/include/xtlm/include"
XTLM_INC2="$AT/include/common_cpp/common_cpp_v1_0/include"
ME_INC="$(ls -d "$AT"/../data/simmodels/osci/*/lnx64/*/systemc/protected/aie_cluster_v1_0_0/include 2>/dev/null | head -1)"

# mlir-aie xaiengine runtime (the HAL: libxaienginecdo + headers).
XAIE_BASE="$NPU_WORK/mlir-aie/install/runtime_lib/x86_64/xaiengine"
XAIE_INC="$XAIE_BASE/include"
XAIE_LIB="$XAIE_BASE/lib"
[[ -f "$XAIE_LIB/libxaienginecdo.so" ]] || { echo "FATAL: libxaienginecdo.so not found at $XAIE_LIB" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
echo ">>> Compiling + linking hal_validate ..."
nice -n 19 g++ -std=c++17 -O0 -g -fPIC -DSC_INCLUDE_DYNAMIC_PROCESSES \
  -I"$SRC" -I"$SC_INC" -I"$XTLM_INC1" -I"$XTLM_INC2" -I"$ME_INC" -I"$XAIE_INC" \
  "$VAL/hal_validate.cpp" \
  "$SRC/aiesim_top.cpp" "$SRC/ps_bridge.cpp" "$SRC/ddr_target.cpp" \
  "$SC_MAIN/sc_main.cpp" "$SC_MAIN/sc_main_main.cpp" \
  -L"$SC_LIB" -lsystemc -lxtlm \
  -L"$XAIE_LIB" -lxaienginecdo \
  -ldl -lpthread \
  -rdynamic -Wl,--disable-new-dtags -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$XAIE_LIB" \
  -o "$OUT"

# Drop the marker-cleared libsystemc next to the harness (same exec-stack fix as
# the bridge build; $ORIGIN rpath resolves it ahead of the aietools one).
cp -f "$SC_LIB/libsystemc.so" "$(dirname "$OUT")/libsystemc.so"
python3 "$SCRIPT_DIR/clear-execstack.py" "$(dirname "$OUT")/libsystemc.so"

echo ">>> Built: $OUT"
echo ">>> Run:   XDNA_AIESIM_BRIDGE unused here; needs aietools libs on LD_LIBRARY_PATH"
