#!/bin/bash
# rebuild-plugin.sh -- Build the XRT emulator plugin and install it.
#
# Usage: ./scripts/rebuild-plugin.sh [--reconfigure] [--rust]
#
# --reconfigure  Force cmake reconfigure
# --rust         Also rebuild the Rust emulator library (cargo build --release)

set -euo pipefail

EMU_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_DIR="$EMU_DIR/xrt-plugin"
BUILD_DIR="$PLUGIN_DIR/build"
INSTALL_DIR="/opt/xilinx/xrt/lib"
SONAME="libxrt_driver_emu.so.2.21.0"

BUILD_RUST=false
RECONFIGURE=false
for arg in "$@"; do
    case "$arg" in
        --rust) BUILD_RUST=true ;;
        --reconfigure) RECONFIGURE=true ;;
    esac
done

# Optionally rebuild the Rust emulator library.
if $BUILD_RUST; then
    echo ">>> Building Rust emulator library..."
    (cd "$EMU_DIR" && nice -n 19 cargo build --release 2>&1)
fi

# Reconfigure if requested or if cache is missing.
if $RECONFIGURE || [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo ">>> Configuring..."
    mkdir -p "$BUILD_DIR"
    nice -n 19 cmake -S "$PLUGIN_DIR" -B "$BUILD_DIR" 2>&1
fi

# Build plugin.
echo ">>> Building plugin..."
nice -n 19 cmake --build "$BUILD_DIR" 2>&1

# Find the Rust emulator library: prefer release (matches bridge test
# workflow), fall back to debug.  Always install it alongside the C++
# plugin so
# the dlopen at runtime picks up the matching build.
RUST_LIB=""
if [[ -f "$EMU_DIR/target/release/libxdna_emu.so" ]]; then
    RUST_LIB="$EMU_DIR/target/release/libxdna_emu.so"
elif [[ -f "$EMU_DIR/target/debug/libxdna_emu.so" ]]; then
    RUST_LIB="$EMU_DIR/target/debug/libxdna_emu.so"
fi

# Install (requires polkit auth).
echo ">>> Installing to $INSTALL_DIR..."
pkexec bash -c "
    cp '$BUILD_DIR/$SONAME' '$INSTALL_DIR/$SONAME' && \
    cp '$BUILD_DIR/$SONAME' '$INSTALL_DIR/libxrt_driver_emu.so.2' && \
    if [[ -n '$RUST_LIB' ]]; then
        cp '$RUST_LIB' '$INSTALL_DIR/libxdna_emu.so'
    fi && \
    echo '>>> Installed OK'
"

if [[ -n "$RUST_LIB" ]]; then
    echo ">>> Rust lib: $RUST_LIB"
else
    echo ">>> WARNING: No Rust library found -- run 'cargo build' first"
fi
