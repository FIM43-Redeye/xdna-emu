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
#   - pkexec for privileged installation steps
#
# Install prefix:
#   Everything installs to /opt/xilinx/xrt.  XRT's build.sh sets this
#   automatically; the xdna plugin and emulator plugin must be told
#   explicitly via CMAKE_INSTALL_PREFIX.

set -euo pipefail

XRT_PREFIX="/opt/xilinx/xrt"

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
echo "  XRT_PREFIX:   $XRT_PREFIX"
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
  echo "--- Building XRT (Release, NPU split packages) ---"
  # Use XRT's own build.sh which sets CMAKE_INSTALL_PREFIX=/opt/xilinx/xrt,
  # enables the NPU component split (xrt-base, xrt-base-dev, xrt-npu),
  # and handles all the cmake flags correctly.
  (cd "$XRT_SRC/build" && nice -n 19 bash build.sh -npu -noctest -j "$JOBS")
  echo ""

  echo "--- Installing XRT debs ---"
  debs=("$XRT_BUILD"/xrt_*-amd64-base.deb "$XRT_BUILD"/xrt_*-amd64-base-dev.deb "$XRT_BUILD"/xrt_*-amd64-npu.deb)
  if [ ${#debs[@]} -gt 0 ]; then
    pkexec dpkg -i "${debs[@]}"
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
  (cd "$XDNA_BUILD" && cmake "$XDNA_DRIVER" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$XRT_PREFIX")
fi
nice -n 19 make -C "$XDNA_BUILD" -j"$JOBS"
echo ""

echo "--- Packaging xdna plugin deb ---"
nice -n 19 make -C "$XDNA_BUILD" -j"$JOBS" package
echo ""

echo "--- Installing xdna plugin ---"
xdna_debs=("$XDNA_BUILD"/xrt_plugin*.deb)
if [ ${#xdna_debs[@]} -gt 0 ]; then
  pkexec dpkg -i "${xdna_debs[@]}"
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

echo "--- Installing emulator libraries ---"
emu_plugin="$PLUGIN_BUILD/libxrt_driver_emu.so.2.21.0"
emu_lib="$EMU_ROOT/target/release/libxdna_emu.so"
install_files=""
[ -f "$emu_plugin" ] && install_files="$install_files '$emu_plugin'"
[ -f "$emu_lib" ]    && install_files="$install_files '$emu_lib'"

if [ -n "$install_files" ]; then
  eval pkexec sh -c "\"cp $install_files '$XRT_PREFIX/lib/' && ldconfig '$XRT_PREFIX/lib/'\""
else
  echo "  WARNING: no emulator libraries found to install"
fi
echo ""

echo "=== All done ==="
echo "  Test: xrt-smi examine                     (real NPU only)"
echo "  Test: XDNA_EMU=1 xrt-smi examine          (emulator at xrt::device(0))"
