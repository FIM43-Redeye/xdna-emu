#!/usr/bin/env bash
# Re-run the EMU on every collision probe directory under a new tag and print the number of points
# that differ from silicon run r1. Run `cargo build -p xdna-emu-ffi` first: the runner loads the
# debug libxdna_emu.so, which `cargo build` does not refresh.
#   tools/collision_emu_sweep.sh <emu-tag> [dir ...]
set -euo pipefail
tag=${1:?emu tag}; shift
root=${COLLISION_EVIDENCE:-$HOME/npu-work/xdna-emu/build/experiments}
dirs=("$@")
[ ${#dirs[@]} -eq 0 ] && dirs=(collisions/probe collisions/probe2 collisions/probe3 collisions/probe4 \
  collisions/probe5 collisions/probe6 collisions/probe7 collisions/probe8 collisions/probe9 \
  collisions/probe10 collisions/probe11 collisions/neg sparse/acc-timing/probe-v2)
here=$(cd "$(dirname "$0")" && pwd)
for d in "${dirs[@]}"; do
  python3 "$here/run_acc_timing_probe.py" emu "$root/$d" "$tag" -j "${JOBS:-12}" >"$root/$d/emu_$tag.log" 2>&1 || true
  python3 "$here/diff_probe_points.py" "$root/$d" r1 "$tag"
done
