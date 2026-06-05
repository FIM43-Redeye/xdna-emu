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
#   ./scripts/rebuild-plugin.sh --aiesim    # Also compile in the aiesim backend
#
# --aiesim builds the Rust FFI .so with `--features aiesim`, which gates in the
# AiesimBackend code (selected at runtime via XDNA_BACKEND=aiesim). The feature
# is purely additive -- there is NO build-time aietools dependency (the bridge
# .so is dlopened at runtime), so the resulting .so still runs the interpreter
# normally when XDNA_BACKEND is unset. Off by default so the standard plugin .so
# stays interpreter-only. The aiesim bridge .so itself is built separately (see
# aiesim-bridge/); this flag only affects the Rust feature gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$EMU_DIR/xrt-plugin/build"

# The C++ plugin compiles SHIM sources from the sibling xdna-driver tree.
# It must be on the emu-shim-base branch (protected m_dev_fd + start_col
# plumbing, xrt submodule pinned to emu-xrt-base). Warn if it is not.
XDNA_DRIVER_DIR="${XDNA_DRIVER_DIR:-$(dirname "$EMU_DIR")/xdna-driver}"
if [[ -d "$XDNA_DRIVER_DIR/.git" ]]; then
  drv_branch="$(git -C "$XDNA_DRIVER_DIR" branch --show-current 2>/dev/null || true)"
  if [[ "$drv_branch" != "emu-shim-base" ]]; then
    echo ">>> WARNING: xdna-driver is on '${drv_branch:-?}', expected 'emu-shim-base'." >&2
    echo ">>>          The plugin build needs that branch's SHIM hooks. Run:" >&2
    echo ">>>          git -C $XDNA_DRIVER_DIR checkout emu-shim-base && git -C $XDNA_DRIVER_DIR submodule update xrt" >&2
  fi
fi

CARGO_FLAGS=""
CARGO_FEATURES=""
PROFILE="debug"
for arg in "$@"; do
  case "$arg" in
    --release)
      CARGO_FLAGS="--release"
      PROFILE="release"
      ;;
    --aiesim)
      CARGO_FEATURES="--features aiesim"
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

echo ">>> Building Rust FFI lib...${CARGO_FEATURES:+ ($CARGO_FEATURES)}"
nice -n 19 cargo build -p xdna-emu-ffi $CARGO_FLAGS $CARGO_FEATURES

echo ">>> Building C++ plugin..."
( cd "$BUILD_DIR" && make -j$(nproc) )

RUST_LIB="$EMU_DIR/target/$PROFILE/libxdna_emu.so"
# The C++ plugin is the XRT driver shim that dlopen's the Rust lib.
# This is what goes into /opt/xilinx/xrt/lib/ -- NOT the Rust .so.
CPP_PLUGIN="$BUILD_DIR/libxrt_driver_emu.so.2.21.0"
XRT_PLUGIN="/opt/xilinx/xrt/lib/libxrt_driver_emu.so.2.21.0"

# Profile-named symlinks installed in /opt/xilinx/xrt/lib/:
#   libxdna_emu_debug.so   -> target/debug/libxdna_emu.so
#   libxdna_emu_release.so -> target/release/libxdna_emu.so
# The plugin dlopens libxdna_emu_<profile>.so first (from pdev_emu.cpp) so
# debug/release selection works via XDNA_EMU_RUNTIME env var without XDNA_EMU_DIR.
XRT_LIB_DIR="/opt/xilinx/xrt/lib"
PROFILE_LINK="$XRT_LIB_DIR/libxdna_emu_${PROFILE}.so"

if [[ ! -f "$RUST_LIB" ]]; then
  echo "FATAL: Rust lib $RUST_LIB not found after build!" >&2
  exit 1
fi

if [[ ! -f "$CPP_PLUGIN" ]] || [[ ! -s "$CPP_PLUGIN" ]]; then
  echo "FATAL: C++ plugin $CPP_PLUGIN not found or empty!" >&2
  echo "  Run: ./scripts/rebuild-plugin.sh --reconfigure" >&2
  exit 1
fi

# Install the C++ plugin to XRT. The Rust .so stays in target/ and is
# loaded by the plugin at runtime via dlopen (path resolved from XDNA_EMU_RUNTIME).
#
# If $XRT_PLUGIN is a symlink pointing at our build output, no install
# is needed -- XRT picks up the build output directly.
if [[ -L "$XRT_PLUGIN" ]]; then
  LINK_TARGET="$(readlink -f "$XRT_PLUGIN")"
  PLUGIN_REAL="$(readlink -f "$CPP_PLUGIN")"
  if [[ "$LINK_TARGET" == "$PLUGIN_REAL" ]]; then
    echo ">>> C++ plugin symlinked to build output, no install needed."
  else
    echo ">>> Symlink exists but points elsewhere ($LINK_TARGET)."
    echo ">>> Re-linking to $CPP_PLUGIN ..."
    ln -sf "$CPP_PLUGIN" "$XRT_PLUGIN"
  fi
else
  # Not a symlink -- check content and copy if needed.
  HASH_SRC="$(md5sum "$CPP_PLUGIN" | cut -d' ' -f1)"
  HASH_DST=""
  if [[ -f "$XRT_PLUGIN" ]]; then
    HASH_DST="$(md5sum "$XRT_PLUGIN" | cut -d' ' -f1)"
  fi

  if [[ "$HASH_SRC" == "$HASH_DST" ]]; then
    echo ">>> C++ plugin already up-to-date ($HASH_SRC), skipping install."
  else
    echo ">>> Installing C++ plugin to $XRT_PLUGIN ..."
    # Try unprivileged copy first (works if dir is user-writable).
    if cp "$CPP_PLUGIN" "$XRT_PLUGIN" 2>/dev/null; then
      echo ">>> Installed (no elevation needed)."
    else
      pkexec cp "$CPP_PLUGIN" "$XRT_PLUGIN"
    fi

    # Verify the copy took effect.
    HASH_DST="$(md5sum "$XRT_PLUGIN" | cut -d' ' -f1)"
    if [[ "$HASH_SRC" != "$HASH_DST" ]]; then
      echo "FATAL: Plugin install failed -- hashes differ!" >&2
      echo "  Built:     $HASH_SRC  $CPP_PLUGIN" >&2
      echo "  Installed: $HASH_DST  $XRT_PLUGIN" >&2
      exit 1
    fi
    echo ">>> Done. C++ plugin installed and verified ($HASH_SRC)."
  fi
fi
# Ensure the profile-suffixed symlink in /opt/xilinx/xrt/lib/ points at the
# Rust .so we just built.  The C++ plugin (pdev_emu.cpp) dlopens
# libxdna_emu_<profile>.so by name; this is what makes that resolution work.
install_profile_symlink() {
  local target_profile="$1"
  local target_lib="$EMU_DIR/target/$target_profile/libxdna_emu.so"
  local link="$XRT_LIB_DIR/libxdna_emu_${target_profile}.so"

  # If the Rust lib for the OTHER profile doesn't exist yet, skip silently
  # (we only install the one we built).
  if [[ "$target_profile" != "$PROFILE" && ! -f "$target_lib" ]]; then
    return
  fi
  if [[ ! -f "$target_lib" ]]; then
    return
  fi

  # Desired: symlink pointing at the built Rust .so.
  local want="$target_lib"
  if [[ -L "$link" ]]; then
    local have="$(readlink "$link")"
    if [[ "$have" == "$want" ]]; then
      return
    fi
  fi

  echo ">>> Linking $link -> $want"
  if ln -sfn "$want" "$link" 2>/dev/null; then
    :
  else
    pkexec ln -sfn "$want" "$link"
  fi
  NEEDS_LDCONFIG=1
}
NEEDS_LDCONFIG=0

install_profile_symlink "$PROFILE"
# Also refresh the OTHER profile's symlink if its build exists, so toggling
# XDNA_EMU_RUNTIME between debug/release just works without re-running this script.
OTHER_PROFILE="debug"
[[ "$PROFILE" == "debug" ]] && OTHER_PROFILE="release"
install_profile_symlink "$OTHER_PROFILE"

# Refresh ldconfig if we created or updated any symlinks, so the newly
# installed names are visible to dlopen via the standard search path.
if [[ "$NEEDS_LDCONFIG" == "1" ]]; then
  echo ">>> Refreshing ldconfig..."
  if ldconfig 2>/dev/null; then
    :
  else
    pkexec ldconfig
  fi
fi

echo ">>> Rust lib: $RUST_LIB"
echo ">>> Profile symlink: $PROFILE_LINK -> $RUST_LIB"
if [[ "$PROFILE" == "debug" ]]; then
  echo ">>> EMU test usage: XDNA_EMU=1 ./test.exe"
else
  echo ">>> EMU test usage: XDNA_EMU=1 XDNA_EMU_RUNTIME=$PROFILE ./test.exe"
fi
