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

FAKE_BIN="$ROOT/fake-bin"
mkdir -p "$FAKE_BIN"
# Variables expand when the generated fake runs.
# shellcheck disable=SC2016
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "$FAKE_RESULT"' >"$FAKE_BIN/ssh"
# Variables expand when the generated fake runs.
# shellcheck disable=SC2016
printf '%s\n' '#!/bin/sh' 'printf "%s\n" "$@" >"$NOTIFY_LOG"' >"$FAKE_BIN/notify-send"
chmod +x "$FAKE_BIN/ssh" "$FAKE_BIN/notify-send"

for scenario in 'success normal succeeded' 'failed critical failed'; do
  read -r result urgency summary <<<"$scenario"
  notify_log="$ROOT/notify-$result"
  if ! PATH="$FAKE_BIN:$PATH" FAKE_RESULT="status=$result" NOTIFY_LOG="$notify_log" \
    "$ROOT/remote/build/halo-xrt-build.sh" __watch "$result-build"; then
    echo "watcher did not handle status=$result" >&2
    exit 1
  fi
  printf '%s\n' \
    '--app-name=Halo XRT' \
    "--urgency=$urgency" \
    "Halo XRT build $summary" \
    "$result-build" >"$ROOT/want-notify"
  cmp "$ROOT/want-notify" "$notify_log"
done

make_deb() {
  local package=$1 output=$2 tree="$ROOT/deb-$1"
  mkdir -p "$tree/DEBIAN"
  printf 'Package: %s\nVersion: 2.26.0\nArchitecture: amd64\nMaintainer: test <test@example.com>\nDescription: test\n' \
    "$package" >"$tree/DEBIAN/control"
  dpkg-deb --root-owner-group --build "$tree" "$output" >/dev/null
}

ARTIFACT_ROOT="$ROOT/artifacts"
XRT_RELEASE="$ARTIFACT_ROOT/src/driver/xrt/build/Release"
DRIVER_RELEASE="$ARTIFACT_ROOT/src/driver/Release"
XRT_STAGING="$XRT_RELEASE/_CPack_Packages/Linux/DEB"
DRIVER_STAGING="$DRIVER_RELEASE/_CPack_Packages/Linux/DEB"
mkdir -p "$XRT_STAGING" "$DRIVER_STAGING"
for package in xrt-base xrt-base-dev xrt-npu; do
  make_deb "$package" "$XRT_RELEASE/$package.deb"
  cp "$XRT_RELEASE/$package.deb" "$XRT_STAGING/"
done
make_deb xrt-plugin-amdxdna "$DRIVER_RELEASE/xrt-plugin-amdxdna.deb"
cp "$DRIVER_RELEASE/xrt-plugin-amdxdna.deb" "$DRIVER_STAGING/"

if ! "$ROOT/remote/build/halo-xrt-build.sh" __collect-packages "$ARTIFACT_ROOT"; then
  echo 'collector rejected normal CPack staging duplicates' >&2
  exit 1
fi
(cd "$ARTIFACT_ROOT/packages" && sha256sum -c SHA256SUMS >/dev/null)
find "$ARTIFACT_ROOT/packages" -maxdepth 1 -type f -name '*.deb' -printf '%f\n' \
  | sort >"$ROOT/collected"
printf '%s\n' xrt-base-dev.deb xrt-base.deb xrt-npu.deb xrt-plugin-amdxdna.deb \
  >"$ROOT/want-collected"
cmp "$ROOT/want-collected" "$ROOT/collected"

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
