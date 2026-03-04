#!/bin/bash
# build-all.sh -- Build XRT, xdna plugin, emulator, and emulator plugin.
#
# Builds everything from the xdna-emu submodule tree so there is one
# source of truth for xdna-driver and XRT.
#
# Usage:
#   ./scripts/build-all.sh           # full build + install
#   ./scripts/build-all.sh --no-xrt  # skip XRT (fast: plugin + emulator only)
#   ./scripts/build-all.sh --dry-run # show what would be done
#
# Prerequisites:
#   - GCC 14 or 15 (NOT 13 -- aietools must not shadow it)
#   - cmake, ninja or make, boost headers
#   - sudo for deb installation (will prompt for fingerprint once)

set -euo pipefail

EMU_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XDNA_DRIVER="$EMU_ROOT/xdna-driver"
XRT_SRC="$XDNA_DRIVER/xrt"
XRT_BUILD="$XRT_SRC/build/Release"
XDNA_BUILD="$XDNA_DRIVER/build/Release"
PLUGIN_BUILD="$EMU_ROOT/xrt-plugin/build"
JOBS="$(nproc)"

BUILD_XRT=true
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --no-xrt)  BUILD_XRT=false ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--no-xrt] [--dry-run]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

# -- Sanity checks -----------------------------------------------------------

gcc_ver=$(gcc -dumpversion 2>/dev/null || echo "none")
case "$gcc_ver" in
  13.*) echo "ERROR: GCC 13 detected -- likely aietools contamination." >&2
        echo "Source activate-npu-env.sh and verify 'gcc --version'." >&2
        exit 1 ;;
  none) echo "ERROR: gcc not found." >&2; exit 1 ;;
esac

if [ ! -d "$XDNA_DRIVER" ]; then
  echo "ERROR: xdna-driver submodule not found at $XDNA_DRIVER" >&2
  echo "Run: git submodule update --init" >&2
  exit 1
fi

echo "=== build-all.sh ==="
echo "  GCC:          $(gcc --version | head -1)"
echo "  EMU_ROOT:     $EMU_ROOT"
echo "  XDNA_DRIVER:  $XDNA_DRIVER"
echo "  JOBS:         $JOBS"
echo "  BUILD_XRT:    $BUILD_XRT"
echo ""

if $DRY_RUN; then
  echo "(dry run -- exiting)"
  exit 0
fi

# -- 1. Apply XRT patches ----------------------------------------------------

echo "--- Applying XRT patches ---"
for patch in "$EMU_ROOT"/docs/patches/xrt-*.patch; do
  [ -f "$patch" ] || continue
  name="$(basename "$patch")"
  if git -C "$XRT_SRC" apply --check "$patch" 2>/dev/null; then
    echo "  Applying: $name"
    git -C "$XRT_SRC" apply "$patch"
  else
    echo "  Already applied or N/A: $name"
  fi
done
echo ""

# -- 2. Build XRT ------------------------------------------------------------

if $BUILD_XRT; then
  echo "--- Building XRT (Release) ---"
  mkdir -p "$XRT_BUILD"
  if [ ! -f "$XRT_BUILD/Makefile" ]; then
    echo "  Configuring..."
    (cd "$XRT_BUILD" && cmake "$XRT_SRC" -DCMAKE_BUILD_TYPE=Release)
  fi
  nice -n 19 make -C "$XRT_BUILD" -j"$JOBS"
  echo ""

  echo "--- Packaging XRT debs ---"
  nice -n 19 make -C "$XRT_BUILD" -j"$JOBS" package
  echo ""

  echo "--- Installing XRT debs (sudo required) ---"
  debs=("$XRT_BUILD"/xrt_*-amd64-*.deb)
  if [ ${#debs[@]} -gt 0 ]; then
    sudo dpkg -i "${debs[@]}"
  else
    echo "  WARNING: No debs found, skipping install"
  fi
  echo ""
fi

# -- 3. Build xdna driver plugin ---------------------------------------------

echo "--- Building xdna driver plugin (Release) ---"
mkdir -p "$XDNA_BUILD"
if [ ! -f "$XDNA_BUILD/Makefile" ]; then
  echo "  Configuring..."
  (cd "$XDNA_BUILD" && cmake "$XDNA_DRIVER" -DCMAKE_BUILD_TYPE=Release)
fi
nice -n 19 make -C "$XDNA_BUILD" -j"$JOBS"
echo ""

echo "--- Packaging xdna plugin deb ---"
nice -n 19 make -C "$XDNA_BUILD" -j"$JOBS" package
echo ""

echo "--- Installing xdna plugin (sudo required) ---"
xdna_debs=("$XDNA_BUILD"/xrt_plugin*.deb)
if [ ${#xdna_debs[@]} -gt 0 ]; then
  sudo dpkg -i "${xdna_debs[@]}"
else
  echo "  WARNING: No xdna plugin debs found, skipping install"
fi
echo ""

# -- 4. Build emulator shared library ----------------------------------------

echo "--- Building emulator (cargo, release) ---"
TMPDIR=/tmp/claude-1000 nice -n 19 cargo build --release --lib --manifest-path "$EMU_ROOT/Cargo.toml"
echo ""

# -- 5. Build emulator XRT plugin --------------------------------------------

echo "--- Building emulator plugin ---"
mkdir -p "$PLUGIN_BUILD"
(cd "$PLUGIN_BUILD" && cmake "$EMU_ROOT/xrt-plugin" -DCMAKE_BUILD_TYPE=Debug)
nice -n 19 make -C "$PLUGIN_BUILD" -j"$JOBS"
echo ""

echo "--- Installing emulator plugin (sudo required) ---"
emu_so="$PLUGIN_BUILD/libxrt_driver_emu.so.2.21.0"
if [ -f "$emu_so" ]; then
  sudo cp "$emu_so" /opt/xilinx/xrt/lib/
  sudo ldconfig /opt/xilinx/xrt/lib/
else
  echo "  WARNING: emulator plugin not found at $emu_so"
fi
echo ""

echo "=== All done ==="
echo "  Test: xrt-smi examine                     (real NPU only)"
echo "  Test: XDNA_EMU=1 xrt-smi examine          (real + emulated)"
