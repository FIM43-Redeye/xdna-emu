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
echo ">>> Done. Rust lib: $RUST_LIB"

# Verify the Rust lib exists and report which profile test scripts will use.
if [[ -f "$RUST_LIB" ]]; then
  echo ">>> EMU test usage: XDNA_EMU=$PROFILE ./test.exe"
else
  echo ">>> WARNING: $RUST_LIB not found!"
fi
