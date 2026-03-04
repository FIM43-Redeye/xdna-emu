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

# Install (requires polkit auth).
echo ">>> Installing to $INSTALL_DIR..."
RUST_LIB="$EMU_DIR/target/release/libxdna_emu.so"
pkexec bash -c "
    cp '$BUILD_DIR/$SONAME' '$INSTALL_DIR/$SONAME' && \
    cp '$BUILD_DIR/$SONAME' '$INSTALL_DIR/libxrt_driver_emu.so.2' && \
    if [[ -f '$RUST_LIB' ]]; then
        cp '$RUST_LIB' '$INSTALL_DIR/libxdna_emu.so'
    fi && \
    echo '>>> Installed OK'
"
