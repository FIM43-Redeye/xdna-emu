#!/usr/bin/env bash
# Grant AppArmor's `hostname` profile read access to the FlexLM shim.
#
# Why this exists
# ---------------
# activate-npu-env.sh sets LD_PRELOAD to a small shim
# (toolchain-build/flexlm-shim/libflexlm_shim.so) that intercepts
# ioctl(SIOCGIFHWADDR) so xchesscc's FlexLM license check works inside
# sandboxed/namespaced environments where no Wi-Fi adapter is visible.
#
# On Ubuntu 25, Canonical ships an AppArmor profile for `hostname`
# (xchesscc invokes it during license setup) that only allows reading a
# small whitelist of paths. Our shim isn't on the list, so ld.so refuses
# to preload it into hostname, the FlexLM check falls through to the
# real ioctl, and xchesscc segfaults with `Failed No such device`.
#
# The profile defines an explicit extension hook for exactly this case:
#   include if exists <local/hostname>
#
# This script drops a one-line `local/hostname` rule that grants read
# access to our shim, then reloads the profile via apparmor_parser.
# The change is reversible (delete the file + reload), system-local,
# and surgical -- no profile is disabled, no broad permissions granted.
#
# Idempotent: running twice is a no-op.
#
# Usage:
#   sudo ./tools/setup/apparmor-flexlm-shim.sh        # via root
#   pkexec ./tools/setup/apparmor-flexlm-shim.sh      # via polkit

set -euo pipefail

LOCAL_DIR=/etc/apparmor.d/local
LOCAL_FILE="$LOCAL_DIR/hostname"
PROFILE_FILE=/etc/apparmor.d/hostname
SHIM_PATH=/home/triple/npu-work/toolchain-build/flexlm-shim/libflexlm_shim.so
RULE="$SHIM_PATH rm,"
MARKER="# xdna-emu flexlm shim"

if [[ "$EUID" -ne 0 ]]; then
  echo "error: must run as root (use sudo or pkexec)" >&2
  exit 64
fi

if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "error: $PROFILE_FILE does not exist; this system does not appear to ship the hostname AppArmor profile, no fix needed" >&2
  exit 0
fi

mkdir -p "$LOCAL_DIR"

if [[ -f "$LOCAL_FILE" ]] && grep -qF "$RULE" "$LOCAL_FILE"; then
  echo "[apparmor-flexlm-shim] rule already present in $LOCAL_FILE -- nothing to do"
else
  # Strip any prior stale variants (e.g. older 'r,' rule from before we
  # learned mmap was also needed). Match by marker line + the next line.
  if [[ -f "$LOCAL_FILE" ]] && grep -qF "$MARKER" "$LOCAL_FILE"; then
    sed -i "/$(printf '%s' "$MARKER" | sed 's/[\/&]/\\&/g')/,+1d" "$LOCAL_FILE"
    echo "[apparmor-flexlm-shim] stripped stale rule from $LOCAL_FILE"
  fi
  {
    echo "$MARKER"
    echo "$RULE"
  } >> "$LOCAL_FILE"
  echo "[apparmor-flexlm-shim] appended rule to $LOCAL_FILE: $RULE"
fi

echo "[apparmor-flexlm-shim] reloading $PROFILE_FILE"
apparmor_parser -r "$PROFILE_FILE"

echo "[apparmor-flexlm-shim] done"
