#!/usr/bin/env bash
set -euo pipefail

readonly PIN=37491ed9af828fc161238dacd82e83ea35a09f87
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
readonly ROOT
readonly SOURCE="$ROOT/build/deps/libvfio-user"
readonly VFIO_BUILD="$SOURCE/build"
readonly OUT="$ROOT/build/tools/phoenix-vfio-user"

for tool in git meson ninja pkg-config cargo "${CC:-cc}"; do
    command -v "$tool" >/dev/null ||
        { echo "missing required tool: $tool" >&2; exit 1; }
done
for package in json-c cmocka; do
    pkg-config --exists "$package" ||
        { echo "missing development package: $package" >&2; exit 1; }
done

# Meson may auto-wrap the compiler with a host ccache whose cache directory is
# outside this worktree. This dependency is small, so caching is not useful.
export CCACHE_DISABLE=1

if [[ ! -e "$SOURCE/.git" ]]; then
    [[ ! -e "$SOURCE" ]] ||
        { echo "$SOURCE exists but is not a git checkout" >&2; exit 1; }
    mkdir -p "$(dirname "$SOURCE")"
    git clone https://gitlab.com/qemu-project/libvfio-user.git "$SOURCE"
fi

[[ -z "$(git -C "$SOURCE" status --porcelain)" ]] ||
    { echo "$SOURCE has local changes" >&2; exit 1; }
if ! git -C "$SOURCE" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$SOURCE" fetch --depth=1 origin "$PIN"
fi
git -C "$SOURCE" checkout --detach "$PIN"
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$PIN" ]] ||
    { echo "libvfio-user checkout did not match $PIN" >&2; exit 1; }

if [[ -f "$VFIO_BUILD/build.ninja" ]]; then
    meson setup --reconfigure "$VFIO_BUILD" "$SOURCE"
else
    meson setup "$VFIO_BUILD" "$SOURCE"
fi
ninja -C "$VFIO_BUILD" lib/libvfio-user.so.0.0.1

COMMON_GIT="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)"
readonly COMMON_GIT
NPU_WORK="$(dirname "$(dirname "$COMMON_GIT")")"
readonly NPU_WORK
export PATH="$NPU_WORK/mlir-aie/ironenv/bin:$PATH"
export PYTHONPATH="$NPU_WORK/mlir-aie/install/python"
export MLIR_AIE_PATH="$NPU_WORK/mlir-aie"
export LLVM_AIE_PATH="$NPU_WORK/llvm-aie"
export AIE_RT_PATH="$NPU_WORK/aie-rt/driver/src"
cargo build --manifest-path "$ROOT/Cargo.toml" -p xdna-emu-ffi

mkdir -p "$OUT"
"${CC:-cc}" \
    -std=gnu11 -D_GNU_SOURCE -Wall -Wextra -Werror \
    -I"$SOURCE/include" -I"$ROOT/include" \
    "$ROOT/tools/phoenix-vfio-user/phoenix_vfio_user.c" \
    -L"$VFIO_BUILD/lib" -Wl,-rpath,"$VFIO_BUILD/lib" -lvfio-user \
    -L"$ROOT/target/debug" -Wl,-rpath,"$ROOT/target/debug" -lxdna_emu \
    -pthread -ldl \
    -o "$OUT/phoenix-vfio-user"
