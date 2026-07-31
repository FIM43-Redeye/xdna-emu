#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if COMMON_GIT_DIR="$(git -C "$EMU_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
  PROJECT_ROOT="$(dirname "$COMMON_GIT_DIR")"
else
  PROJECT_ROOT="$EMU_ROOT"
fi
NPU_WORK_ROOT="$(dirname "$PROJECT_ROOT")"

XDNA_DRIVER_DIR="${XDNA_DRIVER_DIR:-$NPU_WORK_ROOT/xdna-driver-xrt-2.26}"
DRIVER_BASE_REF="${DRIVER_BASE_REF:-upstream/main}"
XRT_BASE_REF="${XRT_BASE_REF:-origin/master}"
HALO_HOST="${HALO_HOST:-halo}"
HALO_HOSTNAME="${HALO_HOSTNAME:-triple-RAH-001}"
HALO_SSH_CONFIG="${HALO_SSH_CONFIG:-$HOME/.ssh/config}"
JOBS="${HALO_JOBS:-32}"
LOCAL_BUILD_ROOT="$PROJECT_ROOT/build/halo-xrt"
REMOTE_BUILD_ROOT="${HALO_XRT_ROOT:-/home/triple/npu-work/xrt-halo}"
REMOTE_CURRENT="$REMOTE_BUILD_ROOT/current"
EXPECTED_PACKAGES=(xrt-base xrt-base-dev xrt-npu xrt-plugin-amdxdna)

fail() {
  echo "$*" >&2
  return 1
}

load_source_contract() {
  [[ -e "$XDNA_DRIVER_DIR/.git" && -d "$XDNA_DRIVER_DIR/xrt" ]] ||
    fail "Missing isolated driver source: $XDNA_DRIVER_DIR" || return

  DRIVER_TIP="$(git -C "$XDNA_DRIVER_DIR" rev-parse HEAD)"
  DRIVER_REF="$(git -C "$XDNA_DRIVER_DIR" branch --show-current)"
  XRT_PIN="$(git -C "$XDNA_DRIVER_DIR" ls-tree HEAD xrt | awk '{print $3}')"
  XRT_TIP="$(git -C "$XDNA_DRIVER_DIR/xrt" rev-parse HEAD)"
  XRT_REF="$(git -C "$XDNA_DRIVER_DIR/xrt" branch --show-current)"

  [[ -n "$XRT_PIN" && "$XRT_TIP" == "$XRT_PIN" ]] ||
    fail 'XRT checkout does not match the driver pin' || return
  [[ -z "$(git -C "$XDNA_DRIVER_DIR" status --porcelain)" ]] ||
    fail 'Driver source worktree is dirty' || return
  [[ -z "$(git -C "$XDNA_DRIVER_DIR/xrt" status --porcelain)" ]] ||
    fail 'XRT source worktree is dirty' || return
  [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "Invalid HALO_JOBS: $JOBS"
}

load_bundle_contract() {
  load_source_contract
  [[ -n "$DRIVER_REF" && -n "$XRT_REF" ]] ||
    fail 'Source worktrees must be on named branches' || return
  DRIVER_BASE="$(git -C "$XDNA_DRIVER_DIR" merge-base HEAD "$DRIVER_BASE_REF")"
  DRIVER_URL="$(git -C "$XDNA_DRIVER_DIR" remote get-url upstream)"
  XRT_BASE="$(git -C "$XDNA_DRIVER_DIR/xrt" merge-base HEAD "$XRT_BASE_REF")"
  XRT_URL="$(git -C "$XDNA_DRIVER_DIR/xrt" remote get-url origin)"
}

source_contract() {
  load_source_contract
  printf 'Driver source: %s\n' "$DRIVER_TIP"
  printf 'XRT source: %s\n' "$XRT_TIP"
  printf 'XRT build: build/build.sh -npu -opt -noctest -j %s\n' "$JOBS"
  printf 'Plugin build: build/build.sh -release -nokmod -j %s\n' "$JOBS"
  printf 'Packages: %s\n' "${EXPECTED_PACKAGES[*]}"
}

ssh_halo() {
  command ssh -F "$HALO_SSH_CONFIG" -o ForwardAgent=no \
    -o BatchMode=yes "$HALO_HOST" "$@"
}

check_halo() {
  local actual
  actual="$(ssh_halo hostname)"
  [[ "$actual" == "$HALO_HOSTNAME" ]] ||
    fail "Refusing remote operation: expected $HALO_HOSTNAME, got $actual"
}

make_bundles() {
  local destination=$1
  mkdir -p "$destination"
  git -C "$XDNA_DRIVER_DIR" bundle create "$destination/driver.bundle" \
    "$DRIVER_REF" "^$DRIVER_BASE"
  git -C "$XDNA_DRIVER_DIR/xrt" bundle create "$destination/xrt.bundle" \
    "$XRT_REF" "^$XRT_BASE"
  cp "${BASH_SOURCE[0]}" "$destination/halo-xrt-build.sh"
}

command_start() {
  local build_id local_dir remote_dir transport unit
  [[ $# -eq 0 ]] || fail "Usage: $0 start" || return
  load_bundle_contract
  check_halo

  build_id="$(date -u +%Y%m%dT%H%M%SZ)-${DRIVER_TIP:0:12}-${XRT_TIP:0:12}"
  local_dir="$LOCAL_BUILD_ROOT/$build_id"
  remote_dir="$REMOTE_BUILD_ROOT/$build_id"
  unit="halo-xrt-$build_id"
  make_bundles "$local_dir"

  ssh_halo mkdir -p "$remote_dir"
  transport="ssh -F $HALO_SSH_CONFIG -o ForwardAgent=no -o BatchMode=yes"
  rsync --archive -e "$transport" "$local_dir/" "$HALO_HOST:$remote_dir/"
  ssh_halo systemd-run --user --collect "--unit=$unit" \
    "$remote_dir/halo-xrt-build.sh" __remote-build \
    "$remote_dir" "$DRIVER_URL" "$DRIVER_BASE" "$DRIVER_TIP" "$DRIVER_REF" \
    "$XRT_URL" "$XRT_BASE" "$XRT_TIP" "$XRT_REF" "$JOBS"
  ssh_halo ln -sfn "$build_id" "$REMOTE_CURRENT"
  printf 'Halo XRT build started: %s\n' "$build_id"
}

checkout_bundle() {
  local destination=$1 url=$2 base=$3 tip=$4 ref=$5 bundle=$6
  git init -q "$destination"
  git -C "$destination" remote add origin "$url"
  git -C "$destination" fetch -q --depth=1 origin "$base"
  git -C "$destination" checkout -q --detach FETCH_HEAD
  git -C "$destination" fetch -q "$bundle" "$ref"
  git -C "$destination" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$tip" ]] ||
    fail "Reconstructed source does not match $tip"
}

collect_packages() {
  local root=$1 package_dir="$1/packages" deb package expected
  local -a candidates
  declare -A found=()

  mkdir -p "$package_dir"
  mapfile -d '' candidates < <(
    find "$root/src/driver/xrt/build/Release" "$root/src/driver/build/Release" \
      -type f -name '*.deb' -print0
  )
  for deb in "${candidates[@]}"; do
    package="$(dpkg-deb -f "$deb" Package)"
    for expected in "${EXPECTED_PACKAGES[@]}"; do
      [[ "$package" == "$expected" ]] || continue
      [[ -z "${found[$package]:-}" ]] || fail "Duplicate package: $package" || return
      cp "$deb" "$package_dir/"
      found[$package]="$deb"
    done
  done
  for expected in "${EXPECTED_PACKAGES[@]}"; do
    [[ -n "${found[$expected]:-}" ]] || fail "Missing package: $expected" || return
  done
  (cd "$package_dir" && sha256sum ./*.deb >SHA256SUMS)
}

remote_build() {
  [[ $# -eq 10 ]] || fail 'Invalid remote build invocation' || return
  local root=$1 driver_url=$2 driver_base=$3 driver_tip=$4 driver_ref=$5
  local xrt_url=$6 xrt_base=$7 xrt_tip=$8 xrt_ref=$9 jobs=${10}
  local driver="$root/src/driver"

  exec >"$root/build.log" 2>&1
  trap 'printf "status=failed\n" >"$root/result"' ERR
  printf 'status=running\n' >"$root/result"
  printf 'driver=%s\nxrt=%s\njobs=%s\n' "$driver_tip" "$xrt_tip" "$jobs"

  mkdir -p "$root/src"
  checkout_bundle "$driver" "$driver_url" "$driver_base" "$driver_tip" \
    "$driver_ref" "$root/driver.bundle"
  checkout_bundle "$driver/xrt" "$xrt_url" "$xrt_base" "$xrt_tip" \
    "$xrt_ref" "$root/xrt.bundle"
  git -C "$driver/xrt" submodule update --init --recursive

  (cd "$driver/xrt" && build/build.sh -npu -opt -noctest -j "$jobs")
  (cd "$driver" && build/build.sh -release -nokmod -j "$jobs")
  collect_packages "$root"

  printf 'status=success\n' >"$root/result"
  trap - ERR
}

current_build() {
  local target
  target="$(ssh_halo readlink "$REMOTE_CURRENT" 2>/dev/null)" ||
    fail 'No Halo XRT build has been recorded' || return
  basename "$target"
}

command_status() {
  local build_id unit active result
  [[ $# -eq 0 ]] || fail "Usage: $0 status" || return
  check_halo
  build_id="$(current_build)"
  unit="halo-xrt-$build_id"
  active="$(ssh_halo systemctl --user is-active "$unit" 2>/dev/null || true)"
  result="$(ssh_halo cat "$REMOTE_BUILD_ROOT/$build_id/result" 2>/dev/null || true)"
  if [[ "$active" == active || "$active" == activating ]]; then
    printf 'build_id=%s\nstate=running\n' "$build_id"
  elif [[ "$result" == status=success ]]; then
    printf 'build_id=%s\nstate=succeeded\n' "$build_id"
  else
    printf 'build_id=%s\nstate=failed\n' "$build_id"
  fi
  ssh_halo tail -n 30 "$REMOTE_BUILD_ROOT/$build_id/build.log" 2>/dev/null || true
}

verify_packages() {
  local directory=$1 deb package expected count
  local -a debs
  mapfile -t debs < <(find "$directory" -maxdepth 1 -type f -name '*.deb' -print | sort)
  [[ ${#debs[@]} -eq ${#EXPECTED_PACKAGES[@]} ]] ||
    fail "Expected ${#EXPECTED_PACKAGES[@]} packages, found ${#debs[@]}" || return
  for expected in "${EXPECTED_PACKAGES[@]}"; do
    count=0
    for deb in "${debs[@]}"; do
      package="$(dpkg-deb -f "$deb" Package)"
      [[ "$package" == "$expected" ]] && count=$((count + 1))
    done
    [[ $count -eq 1 ]] || fail "Expected one $expected package, found $count" || return
  done
}

command_fetch() {
  local build_id remote_dir local_dir transport result
  [[ $# -eq 0 ]] || fail "Usage: $0 fetch" || return
  check_halo
  build_id="$(current_build)"
  remote_dir="$REMOTE_BUILD_ROOT/$build_id"
  local_dir="$LOCAL_BUILD_ROOT/$build_id/packages"
  result="$(ssh_halo cat "$remote_dir/result" 2>/dev/null || true)"
  [[ "$result" == status=success ]] || fail "Build $build_id has not succeeded" || return

  mkdir -p "$local_dir"
  transport="ssh -F $HALO_SSH_CONFIG -o ForwardAgent=no -o BatchMode=yes"
  rsync --archive -e "$transport" "$HALO_HOST:$remote_dir/packages/" "$local_dir/"
  (cd "$local_dir" && sha256sum -c SHA256SUMS)
  verify_packages "$local_dir"
  printf 'Fetched packages: %s\n' "$local_dir"
}

usage() {
  printf 'Usage: %s plan|start|status|fetch\n' "$0" >&2
}

case "${1:-}" in
  plan) shift; source_contract "$@" ;;
  start) shift; command_start "$@" ;;
  status) shift; command_status "$@" ;;
  fetch) shift; command_fetch "$@" ;;
  __remote-build) shift; remote_build "$@" ;;
  *) usage; exit 2 ;;
esac
