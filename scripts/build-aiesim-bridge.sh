#!/usr/bin/env bash
# build-aiesim-bridge.sh -- Build libxdna_aiesim_bridge.so (the in-process
# aiesim backend's C++ side). Unlike the XRT plugin, the bridge is not installed
# system-wide: the Rust DlopenBridge dlopens it by path (XDNA_AIESIM_BRIDGE env)
# or by name via LD_LIBRARY_PATH. This script just configures + builds + reports.
#
# Requires aietools (for SystemC + the cluster libs at runtime). Source
# toolchain-build/activate-npu-env.sh first so XILINX_VITIS_AIETOOLS is set.
#
# Usage:
#   source toolchain-build/activate-npu-env.sh
#   ./scripts/build-aiesim-bridge.sh              # configure (if needed) + build
#   ./scripts/build-aiesim-bridge.sh --reconfigure
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$EMU_DIR/aiesim-bridge"
BUILD_DIR="$SRC_DIR/build"

if [[ -z "${XILINX_VITIS_AIETOOLS:-}" ]]; then
  # Not fatal -- CMake can still fall back to FindAIETools -- but warn loudly.
  echo ">>> WARNING: XILINX_VITIS_AIETOOLS is unset." >&2
  echo ">>>          Run: source toolchain-build/activate-npu-env.sh" >&2
fi

for arg in "$@"; do
  case "$arg" in
    --reconfigure)
      echo ">>> Reconfiguring cmake..."
      rm -rf "$BUILD_DIR/CMakeCache.txt"
      ;;
  esac
done

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo ">>> First-time cmake configure..."
  mkdir -p "$BUILD_DIR"
  ( cd "$BUILD_DIR" && cmake .. )
fi

echo ">>> Building aiesim bridge..."
nice -n 19 make -C "$BUILD_DIR" -j"$(nproc)"

SO="$BUILD_DIR/libxdna_aiesim_bridge.so"
if [[ ! -f "$SO" ]]; then
  echo "FATAL: $SO not found after build!" >&2
  exit 1
fi

# Drop a marker-cleared copy of libsystemc next to the bridge .so. aietools'
# libsystemc ships with a spurious executable-stack marker that makes dlopen
# fail on hardened kernels; the bridge carries DT_RPATH=$ORIGIN so this
# co-located copy resolves ahead of the aietools one. Regenerated each build
# from the local aietools (nothing aietools is modified in place or committed).
# See docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
AIETOOLS="${XILINX_VITIS_AIETOOLS:-}"
if [[ -z "$AIETOOLS" ]]; then
  # Fall back to the path CMake resolved (cached) so this works even if the env
  # var is unset but FindAIETools found it.
  AIETOOLS="$(grep -m1 '^AIETOOLS_DIR' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2-)"
fi
SRC_SYSTEMC="$AIETOOLS/lib/lnx64.o/libsystemc.so"
if [[ ! -f "$SRC_SYSTEMC" ]]; then
  echo "FATAL: libsystemc not found at $SRC_SYSTEMC" >&2
  exit 1
fi
cp -f "$SRC_SYSTEMC" "$BUILD_DIR/libsystemc.so"
python3 "$SCRIPT_DIR/clear-execstack.py" "$BUILD_DIR/libsystemc.so"

echo ">>> Built: $SO"
echo ">>> Exported aiesim_* symbols:"
nm -D --defined-only "$SO" | grep -E ' T aiesim_' || {
  echo "FATAL: no aiesim_* symbols exported -- check the version script / extern \"C\"." >&2
  exit 1
}
echo ">>> To use from the FFI: export XDNA_AIESIM_BRIDGE=$SO"
