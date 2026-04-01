#!/usr/bin/env bash
# rebuild-plugin.sh -- Build emulator + XRT plugin.
#
# This is a convenience wrapper around cargo build. The build.rs script
# handles cmake and plugin installation automatically.
#
# Usage:
#   ./scripts/rebuild-plugin.sh             # Debug build (default)
#   ./scripts/rebuild-plugin.sh --release   # Release build
#   ./scripts/rebuild-plugin.sh --reconfigure  # Force cmake reconfigure
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$EMU_DIR/xrt-plugin/build"

CARGO_FLAGS=""
PROFILE="debug"
for arg in "$@"; do
  case "$arg" in
    --release)
      CARGO_FLAGS="--release"
      PROFILE="release"
      ;;
    --reconfigure)
      echo ">>> Reconfiguring cmake..."
      rm -rf "$BUILD_DIR/CMakeCache.txt"
      mkdir -p "$BUILD_DIR"
      ( cd "$BUILD_DIR" && cmake .. )
      ;;
  esac
done

# Ensure cmake is configured (first-time setup)
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo ">>> First-time cmake configure..."
  mkdir -p "$BUILD_DIR"
  ( cd "$BUILD_DIR" && cmake .. )
fi

echo ">>> Building (cargo build handles plugin automatically)..."
nice -n 19 cargo build $CARGO_FLAGS

RUST_LIB="$EMU_DIR/target/$PROFILE/libxdna_emu.so"
# The C++ plugin is the XRT driver shim that dlopen's the Rust lib.
# This is what goes into /opt/xilinx/xrt/lib/ -- NOT the Rust .so.
CPP_PLUGIN="$BUILD_DIR/libxrt_driver_emu.so.2.21.0"
XRT_PLUGIN="/opt/xilinx/xrt/lib/libxrt_driver_emu.so.2.21.0"

if [[ ! -f "$RUST_LIB" ]]; then
  echo "FATAL: Rust lib $RUST_LIB not found after build!" >&2
  exit 1
fi

if [[ ! -f "$CPP_PLUGIN" ]]; then
  echo "FATAL: C++ plugin $CPP_PLUGIN not found!" >&2
  echo "  The XRT driver plugin must be built separately (cmake in xrt-plugin/)." >&2
  echo "  Run: ./scripts/rebuild-plugin.sh --reconfigure" >&2
  exit 1
fi

# Install the C++ plugin to XRT. The Rust .so stays in target/ and is
# loaded by the plugin at runtime via dlopen (path resolved from XDNA_EMU).
echo ">>> Installing C++ plugin to $XRT_PLUGIN ..."
pkexec cp "$CPP_PLUGIN" "$XRT_PLUGIN"

# Verify the copy actually took effect (pkexec can silently fail if
# the user dismisses the auth dialog).
HASH_SRC="$(md5sum "$CPP_PLUGIN" | cut -d' ' -f1)"
HASH_DST="$(md5sum "$XRT_PLUGIN" | cut -d' ' -f1)"
if [[ "$HASH_SRC" != "$HASH_DST" ]]; then
  echo "FATAL: Plugin install failed -- hashes differ!" >&2
  echo "  Built:     $HASH_SRC  $CPP_PLUGIN" >&2
  echo "  Installed: $HASH_DST  $XRT_PLUGIN" >&2
  exit 1
fi

echo ">>> Done. C++ plugin installed and verified ($HASH_SRC)."
echo ">>> Rust lib: $RUST_LIB"
echo ">>> EMU test usage: XDNA_EMU=$PROFILE ./test.exe"
