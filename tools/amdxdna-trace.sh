#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# amdxdna-trace.sh -- enable/snapshot/disable amdxdna kernel tracepoints
# around a long-running test suite (bridge sweep, ISA test).
#
# Wraps the same /sys/kernel/tracing/events/amdxdna_trace/* events that
# tools/bug6-trace.sh uses, but designed for SUITE-wide capture under a
# SINGLE pkexec auth.  The suite spawns a backgrounded `daemon` instance
# which enables tracing, polls a sentinel file, and snapshots+disables
# when the parent removes the sentinel.  Single auth covers the whole
# suite -- no second prompt mid-run, no auth-gap delays.
#
# Subcommands:
#   daemon SENTINEL PARENT_PID OUT_DIR [LABEL] [BUFFER_KB]
#                                -- pkexec target.  Enables events, sizes
#                                   buffer (default 16384 = 16MB/CPU),
#                                   starts tracing, then polls SENTINEL.
#                                   When SENTINEL is removed (or PARENT_PID
#                                   dies), snapshots OUT_DIR/<LABEL>.{trace,
#                                   dmesg}, disables events, chowns back to
#                                   the calling user, exits.  Default
#                                   LABEL=amdxdna.
#   enable [BUFFER_KB]           -- manual: pkexec, enable events + start
#                                   tracing, leave running.  No daemon.
#                                   Pair with `snapshot-disable` later.
#   snapshot OUT_DIR [LABEL]     -- pkexec: snapshot trace + dmesg only,
#                                   leave tracing on.
#   snapshot-disable OUT_DIR [LABEL]
#                                -- pkexec: snapshot, then stop tracing.
#   disable                      -- pkexec: stop tracing without snapshot.
#                                   Idempotent.
#
# State (start timestamp) lives in $AMDXDNA_TRACE_STATE (default
# /tmp/claude-1000/amdxdna-trace.state) for the manual subcommands so
# `snapshot` / `snapshot-disable` can derive `dmesg --since`.  Daemon
# mode keeps its own state in process memory.

set -euo pipefail

TRACEFS=/sys/kernel/tracing
EVENTS=(
    amdxdna_trace/xdna_job
    amdxdna_trace/mbox_set_tail
    amdxdna_trace/mbox_set_head
    amdxdna_trace/mbox_irq_handle
    amdxdna_trace/mbox_rx_worker
    amdxdna_trace/uc_irq_handle
    amdxdna_trace/uc_wakeup
)
STATE_FILE=${AMDXDNA_TRACE_STATE:-/tmp/claude-1000/amdxdna-trace.state}

# Build "enable / disable each event if its tracepoint dir exists" snippet
# for inlining inside heredocs.  Each event is gated because amdxdna may
# or may not export every event in older srcversions; missing events
# shouldn't fail the whole setup.
_event_cmds() {
    local action=$1   # 1 (enable) or 0 (disable)
    for e in "${EVENTS[@]}"; do
        printf '[[ -d %s/events/%s ]] && echo %s > %s/events/%s/enable || true\n' \
            "$TRACEFS" "$e" "$action" "$TRACEFS" "$e"
    done
}

# Resolve the username to chown captured files back to.  Under pkexec,
# PKEXEC_UID points at the calling user's UID; outside pkexec, fall
# back to USER (we're not root anyway).
_calling_user() {
    if [[ -n "${PKEXEC_UID:-}" ]]; then
        getent passwd "$PKEXEC_UID" | cut -d: -f1
    else
        echo "${USER:-$(id -un)}"
    fi
}

cmd_enable() {
    local buffer_kb=${1:-16384}
    local start_ts
    start_ts=$(date '+%Y-%m-%d %H:%M:%S')

    local enable_cmds
    enable_cmds=$(_event_cmds 1)

    pkexec bash -c "
        set -e
        echo 0 > $TRACEFS/tracing_on
        echo > $TRACEFS/trace
        echo $buffer_kb > $TRACEFS/buffer_size_kb
        $enable_cmds
        echo 1 > $TRACEFS/tracing_on
    "
    mkdir -p "$(dirname "$STATE_FILE")"
    echo "$start_ts" > "$STATE_FILE"
    echo "[amdxdna-trace] enabled (buffer=${buffer_kb}KB, start=$start_ts)"
}

cmd_snapshot() {
    local out_dir=${1:?out dir required}
    local label=${2:-amdxdna}
    local start_ts
    if [[ -f "$STATE_FILE" ]]; then
        start_ts=$(cat "$STATE_FILE")
    else
        echo "WARN: $STATE_FILE missing -- using last hour for dmesg" >&2
        start_ts=$(date '+%Y-%m-%d %H:%M:%S' --date='1 hour ago')
    fi
    mkdir -p "$out_dir"
    local trace_file="$out_dir/${label}.trace"
    local dmesg_file="$out_dir/${label}.dmesg"
    pkexec bash -c "
        cat $TRACEFS/trace > '$trace_file'
        dmesg -T --since '$start_ts' > '$dmesg_file'
        chown $USER:$USER '$trace_file' '$dmesg_file'
    "
    local lines
    lines=$(wc -l < "$trace_file" 2>/dev/null || echo 0)
    echo "[amdxdna-trace] snapshot $label: ${lines} trace lines -> $out_dir/"
}

cmd_snapshot_disable() {
    local out_dir=${1:?out dir required}
    local label=${2:-amdxdna}
    local start_ts
    if [[ -f "$STATE_FILE" ]]; then
        start_ts=$(cat "$STATE_FILE")
    else
        echo "WARN: $STATE_FILE missing -- using last hour for dmesg" >&2
        start_ts=$(date '+%Y-%m-%d %H:%M:%S' --date='1 hour ago')
    fi
    mkdir -p "$out_dir"
    local trace_file="$out_dir/${label}.trace"
    local dmesg_file="$out_dir/${label}.dmesg"

    local disable_cmds
    disable_cmds=$(_event_cmds 0)

    pkexec bash -c "
        cat $TRACEFS/trace > '$trace_file'
        dmesg -T --since '$start_ts' > '$dmesg_file'
        chown $USER:$USER '$trace_file' '$dmesg_file'
        echo 0 > $TRACEFS/tracing_on
        $disable_cmds
        echo > $TRACEFS/trace
    "
    rm -f "$STATE_FILE"
    local lines
    lines=$(wc -l < "$trace_file" 2>/dev/null || echo 0)
    echo "[amdxdna-trace] snapshot+disable $label: ${lines} trace lines -> $out_dir/"
}

cmd_disable() {
    local disable_cmds
    disable_cmds=$(_event_cmds 0)
    pkexec bash -c "
        echo 0 > $TRACEFS/tracing_on
        $disable_cmds
        echo > $TRACEFS/trace
    " || true   # idempotent: already-disabled state is fine
    rm -f "$STATE_FILE"
    echo "[amdxdna-trace] disabled"
}

# Daemon: enable + poll-sentinel + snapshot+disable on exit.  Designed
# to be invoked under pkexec by the suite scripts -- single auth covers
# the whole suite lifetime.  Self-cleaning: trap snapshots even if
# killed via signal or if parent dies.
cmd_daemon() {
    local sentinel=${1:?sentinel file required}
    local parent_pid=${2:?parent pid required}
    local out_dir=${3:?out dir required}
    local label=${4:-amdxdna}
    local buffer_kb=${5:-16384}

    if [[ ! -e "$sentinel" ]]; then
        echo "ERROR: sentinel $sentinel does not exist at daemon start" >&2
        exit 1
    fi

    local user
    user=$(_calling_user)
    if [[ -z "$user" ]]; then
        echo "ERROR: could not resolve calling user" >&2
        exit 1
    fi

    mkdir -p "$out_dir"
    local trace_file="$out_dir/${label}.trace"
    local dmesg_file="$out_dir/${label}.dmesg"

    local start_ts
    start_ts=$(date '+%Y-%m-%d %H:%M:%S')

    local enable_cmds disable_cmds
    enable_cmds=$(_event_cmds 1)
    disable_cmds=$(_event_cmds 0)

    # EXIT trap snapshots + disables + chowns regardless of how we exit
    # (sentinel removed, parent died, signal received, error).  Inline
    # the variables that need outer-scope expansion; everything else
    # runs in the trap's eval context.
    # shellcheck disable=SC2064
    trap "
        cat $TRACEFS/trace > '$trace_file' 2>/dev/null || true
        dmesg -T --since '$start_ts' > '$dmesg_file' 2>/dev/null || true
        chown $user:$user '$trace_file' '$dmesg_file' 2>/dev/null || true
        echo 0 > $TRACEFS/tracing_on 2>/dev/null || true
        $disable_cmds
        echo > $TRACEFS/trace 2>/dev/null || true
        echo '[amdxdna-trace daemon] snapshot+disabled at exit -> $out_dir/${label}.{trace,dmesg}'
    " EXIT TERM INT

    # Set up tracing.  set -e is on (script-wide); failures here abort
    # before the loop, and the EXIT trap still tries snapshot+disable.
    echo 0 > $TRACEFS/tracing_on
    echo > $TRACEFS/trace
    echo "$buffer_kb" > $TRACEFS/buffer_size_kb
    eval "$enable_cmds"
    echo 1 > $TRACEFS/tracing_on

    echo "[amdxdna-trace daemon] enabled (buffer=${buffer_kb}KB, start=$start_ts, sentinel=$sentinel, parent=$parent_pid)"

    # Poll sentinel file + parent liveness.  When sentinel disappears
    # (parent signals "we're done") OR parent dies (orphaned daemon),
    # exit and let the trap fire.
    while [[ -e "$sentinel" ]] && kill -0 "$parent_pid" 2>/dev/null; do
        sleep 2
    done
}

case "${1:-}" in
    daemon)             shift; cmd_daemon "$@" ;;
    enable)             shift; cmd_enable "$@" ;;
    snapshot)           shift; cmd_snapshot "$@" ;;
    snapshot-disable)   shift; cmd_snapshot_disable "$@" ;;
    disable)            shift; cmd_disable ;;
    *)
        sed -n '1,38p' "$0" >&2
        exit 1
        ;;
esac
