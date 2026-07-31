#!/usr/bin/env bash

set -euo pipefail

ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

XRT="$ROOT/xrt"
DRIVER="$ROOT/driver"

git init -q "$XRT"
git -C "$XRT" config user.name test
git -C "$XRT" config user.email test@example.com
touch "$XRT/source"
git -C "$XRT" add source
git -C "$XRT" commit -qm xrt
PIN="$(git -C "$XRT" rev-parse HEAD)"

git init -q "$DRIVER"
git -C "$DRIVER" config user.name test
git -C "$DRIVER" config user.email test@example.com
git -C "$DRIVER" -c protocol.file.allow=always submodule add -q "$XRT" xrt
git -C "$DRIVER" commit -qam driver

OUTPUT="$(XDNA_DRIVER_DIR="$DRIVER" scripts/halo-xrt-build.sh plan)"
grep -F "XRT source: $PIN" <<<"$OUTPUT"
grep -F 'XRT build: build/build.sh -npu -opt -noctest -j 32' <<<"$OUTPUT"
grep -F 'Plugin build: build/build.sh -release -nokmod -j 32' <<<"$OUTPUT"
grep -F 'Packages: xrt-base xrt-base-dev xrt-npu xrt-plugin-amdxdna' <<<"$OUTPUT"

mkdir -p "$ROOT/remote/build"
cp scripts/halo-xrt-build.sh "$ROOT/remote/build/"
if "$ROOT/remote/build/halo-xrt-build.sh" __remote-build >"$ROOT/out" 2>"$ROOT/err"; then
  echo 'invalid remote invocation unexpectedly succeeded' >&2
  exit 1
fi
grep -F 'Invalid remote build invocation' "$ROOT/err"

touch "$XRT/new-source"
git -C "$XRT" add new-source
git -C "$XRT" commit -qm newer-xrt
git -C "$DRIVER/xrt" fetch -q origin
git -C "$DRIVER/xrt" checkout -q FETCH_HEAD

if XDNA_DRIVER_DIR="$DRIVER" scripts/halo-xrt-build.sh plan >"$ROOT/out" 2>"$ROOT/err"; then
  echo 'plan accepted an XRT checkout that differs from the driver pin' >&2
  exit 1
fi
if ! grep -F 'XRT checkout does not match the driver pin' "$ROOT/err"; then
  cat "$ROOT/err" >&2
  exit 1
fi
