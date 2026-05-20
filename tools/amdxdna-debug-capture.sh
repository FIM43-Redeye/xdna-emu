#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# amdxdna-debug-capture.sh -- snapshot amdxdna debugfs + sysfs state
# in a single pkexec invocation.  Two modes:
#
#   baseline OUT_DIR   -- clean device, no submissions yet.  Reads
#                         tdr_control, ctx_rq, ringbuf, FW log/trace
#                         buffers (empty unless fw_log_level / fw_trace_size
#                         module params are set), telemetry, sysfs IDs.
#   wedged OUT_DIR     -- device is wedged, TDR has fired.  Same set
#                         plus dmesg since 5 min ago, and a mailbox
#                         liveness probe (echo 1 > nputest).  Does NOT
#                         attempt recovery -- that's a separate step.
#
# OUT_DIR is created if absent; a timestamp suffix is appended unless
# the path already exists.  Final ownership is chowned back to the
# calling user.
#
# See docs/superpowers/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md
# for the wider procedure and what to do with the captured forensics.

set -euo pipefail

DBG=/sys/kernel/debug/accel/0000:c6:00.1
SYSFS=/sys/class/accel/accel0/device

_calling_user() {
    if [[ -n "${PKEXEC_UID:-}" ]]; then
        getent passwd "$PKEXEC_UID" | cut -d: -f1
    else
        echo "${USER:-$(id -un)}"
    fi
}

_resolve_dbg() {
    # /sys/kernel/debug is mode 0700 root, so we can't stat or list it
    # as the calling user.  Always emit $DBG and let the pkexec block
    # validate it (it'll error visibly if the path is wrong).  Override
    # with AMDXDNA_DEBUGFS_DIR=/sys/kernel/debug/accel/<bdf> if the BDF
    # moves after a hardware swap.
    echo "${AMDXDNA_DEBUGFS_DIR:-$DBG}"
}

_capture_common() {
    # Privileged half -- runs under pkexec.  Args: $1=OUT_DIR, $2=DBG path.
    local out=$1
    local dbg=$2
    mkdir -p "$out"
    cp "$dbg/tdr_control" "$out/tdr_control.txt"
    cp "$dbg/ctx_rq" "$out/ctx_rq.txt"
    cp "$dbg/ringbuf" "$out/ringbuf.txt"
    # FW log/trace come back empty (Invalid input) unless fw_log_level /
    # fw_trace_size are set -- don't fail the capture.
    dd if="$dbg/dump_fw_log_buffer" of="$out/fw_log.bin" status=none 2>/dev/null || \
        : > "$out/fw_log.bin"
    dd if="$dbg/dump_fw_trace_buffer" of="$out/fw_trace.bin" status=none 2>/dev/null || \
        : > "$out/fw_trace.bin"
    cat "$dbg/telemetry_health"      > "$out/telemetry_health.bin"      2>/dev/null || true
    cat "$dbg/telemetry_error_info"  > "$out/telemetry_error_info.bin"  2>/dev/null || true
    cat "$dbg/telemetry_disabled"    > "$out/telemetry_disabled.txt"    2>/dev/null || true
}

cmd_baseline() {
    local out=${1:?usage: baseline OUT_DIR}
    [[ -d "$out" ]] || out="${out}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"
    local dbg
    dbg=$(_resolve_dbg) || exit 1

    {
        echo "===date==="
        date
        echo "===uname==="
        uname -a
        echo "===amdxdna module==="
        cat /sys/module/amdxdna/version    2>/dev/null
        cat /sys/module/amdxdna/srcversion 2>/dev/null
        echo "===module params==="
        for f in /sys/module/amdxdna/parameters/*; do
            printf "%s = %s\n" "$(basename "$f")" "$(cat "$f" 2>/dev/null)"
        done
        echo "===accel0 sysfs==="
        for f in vbnv device_type fw_version power_state; do
            printf "%s: %s\n" "$f" "$(cat "$SYSFS/$f" 2>/dev/null)"
        done
    } > "$out/userland.txt"

    local user
    user=$(id -un)
    pkexec sh -c "
        $(declare -f _capture_common)
        _capture_common '$out' '$dbg'
        chown -R '$user:$user' '$out'
    "
    echo "baseline captured: $out"
}

cmd_wedged() {
    local out=${1:?usage: wedged OUT_DIR}
    [[ -d "$out" ]] || out="${out}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"
    local dbg
    dbg=$(_resolve_dbg) || exit 1

    {
        echo "===date==="
        date
        echo "===fw_version (post-wedge)==="
        cat "$SYSFS/fw_version" 2>/dev/null
        echo "===power_state==="
        cat "$SYSFS/power_state" 2>/dev/null
    } > "$out/userland.txt"

    dmesg --since "5 min ago" > "$out/dmesg-pre-probe.txt"

    local user
    user=$(id -un)
    pkexec sh -c "
        $(declare -f _capture_common)
        _capture_common '$out' '$dbg'
        # Mailbox liveness probe (writes to nputest).  Result goes to dmesg.
        echo 1 > '$dbg/nputest' 2>'$out/nputest.err' || true
        # app_health requires a live mailbox -- if it hangs, the timeout will
        # surface as 'pkexec ... took too long' upstream.
        timeout 5 cat '$dbg/get_app_health' > '$out/app_health.txt' 2>&1 || \
            echo '(timed out or failed)' > '$out/app_health.txt'
        chown -R '$user:$user' '$out'
    "
    # Capture the nputest verdict from dmesg post-probe
    sleep 1
    dmesg --since "30 sec ago" | grep -iE 'nputest|NPU health|protocol_version' \
        > "$out/dmesg-nputest.txt" || true

    echo "wedged forensics captured: $out"
    echo "  Check $out/dmesg-nputest.txt for mailbox liveness verdict."
    echo "  Compare $out/{tdr_control,ctx_rq}.txt against baseline."
}

case "${1:-}" in
    baseline) shift; cmd_baseline "$@" ;;
    wedged)   shift; cmd_wedged   "$@" ;;
    *)
        echo "usage: $(basename "$0") baseline OUT_DIR" >&2
        echo "       $(basename "$0") wedged   OUT_DIR" >&2
        exit 2
        ;;
esac
