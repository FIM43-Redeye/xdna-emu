#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# bug6-trace.sh -- capture amdxdna_trace events around a single test run.
#
# Bug #6 (memtile_dmas hang at run.wait, host never observes completion)
# investigation aid. Enables the existing kernel tracepoints, runs the
# target test under a timeout, and snapshots the trace ringbuffer + dmesg
# so we can compare hang vs pass states side by side.
#
# Tracepoints used (all already in amdxdna driver):
#   xdna_job          - job submit ("job run") / fence signal / free
#   mbox_set_tail     - host posts message to firmware (msg_id + opcode)
#   mbox_set_head     - host consumes response from firmware (msg_id + opcode)
#   mbox_irq_handle   - mailbox IRQ fired
#   mbox_rx_worker    - rx worker dispatching responses
#   uc_irq_handle     - microcontroller-channel IRQ
#   uc_wakeup         - microcontroller-channel wakeup
#
# Usage:
#   bug6-trace.sh <label> <test-dir> [timeout-sec]
#
# Example:
#   bug6-trace.sh hang  ~/npu-work/mlir-aie/build/test/npu-xrt/memtile_dmas/writebd/chess 30
#   bug6-trace.sh pass  ~/npu-work/mlir-aie/build/test/npu-xrt/memtile_dmas/writebd/chess 30
#
# Outputs (under $OUT_DIR, default xdna-emu/build/experiments/bug6/):
#   <label>.trace            -- /sys/kernel/tracing/trace snapshot
#   <label>.dmesg            -- dmesg -T --since=<start> output
#   <label>.test.log         -- test.exe stdout/stderr
#   <label>.meta             -- timing, exit code, kernel/srcversion, args
#
# All privileged operations (trace setup, test invocation via runuser,
# snapshot, chown) are combined into a SINGLE pkexec call so the user
# only authenticates once and there are no timing gaps between auth
# prompts. Capturing trace+test+snapshot in one root context also
# eliminates the risk of background events firing between two pkexec
# windows.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '1,40p' "$0"
    exit 1
fi

LABEL=$1
TEST_DIR=$2
TIMEOUT_SEC=${3:-30}

OUT_DIR=${OUT_DIR:-/home/triple/npu-work/xdna-emu/build/experiments/bug6}
mkdir -p "$OUT_DIR"

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

if [[ ! -x "$TEST_DIR/test.exe" ]]; then
    echo "ERROR: $TEST_DIR/test.exe not found or not executable" >&2
    exit 2
fi
if [[ ! -f "$TEST_DIR/aie.xclbin" || ! -f "$TEST_DIR/insts.bin" ]]; then
    echo "ERROR: $TEST_DIR missing aie.xclbin or insts.bin" >&2
    exit 2
fi

START_TS=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)
SRCVER=$(cat /sys/module/amdxdna/srcversion 2>/dev/null || echo "module-not-loaded")
KVER=$(uname -r)

RUN_LOG="$OUT_DIR/${LABEL}.test.log"
RC_FILE="$OUT_DIR/${LABEL}.rc"
TRACE_FILE="$OUT_DIR/${LABEL}.trace"
DMESG_FILE="$OUT_DIR/${LABEL}.dmesg"

echo "=== bug6-trace: $LABEL ==="
echo "Test dir : $TEST_DIR"
echo "Timeout  : ${TIMEOUT_SEC}s"
echo "Start    : $START_TS"
echo "Kernel   : $KVER"
echo "Module sv: $SRCVER"
echo "Out dir  : $OUT_DIR"

# Pre-create writable output files so root chown is the only ownership op.
: > "$RUN_LOG"
rm -f "$RC_FILE" "$TRACE_FILE" "$DMESG_FILE"

# Build the events-enable loop as plain commands (avoid bash array expansion
# headaches inside the heredoc).
EVENT_ENABLE_CMDS=$(for e in "${EVENTS[@]}"; do
    printf '[[ -d %s/events/%s ]] && echo 1 > %s/events/%s/enable || echo "WARN: missing event %s" >&2\n' \
        "$TRACEFS" "$e" "$TRACEFS" "$e" "$e"
done)

# Single pkexec: setup trace, drop to user for test, snapshot trace + dmesg.
# All variables expanded by the *outer* shell -- inside the heredoc, escape
# anything we want the inner shell to evaluate (mostly nothing here; we
# capture the test's exit code directly via a marker file).
pkexec bash -c "
set -e
echo 0 > $TRACEFS/tracing_on
echo > $TRACEFS/trace
echo 16384 > $TRACEFS/buffer_size_kb
$EVENT_ENABLE_CMDS
echo 1 > $TRACEFS/tracing_on

# Drop to user for the test. test.exe is a self-contained mlir-aie binary
# with rpath; no env beyond cwd is needed.
runuser -u $USER -- bash -c \"cd '$TEST_DIR' && timeout '${TIMEOUT_SEC}s' ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin\" > '$RUN_LOG' 2>&1
echo \$? > '$RC_FILE'

echo 0 > $TRACEFS/tracing_on
cat $TRACEFS/trace > '$TRACE_FILE'
dmesg -T --since '$START_TS' > '$DMESG_FILE'
chown $USER:$USER '$TRACE_FILE' '$DMESG_FILE' '$RUN_LOG' '$RC_FILE'
" || { echo "ERROR: privileged block failed (exit $?)"; exit 3; }

RC=$(cat "$RC_FILE" 2>/dev/null || echo "999")
END_TS=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(( $(date +%s) - START_EPOCH ))

cat > "$OUT_DIR/${LABEL}.meta" <<EOF
label:        $LABEL
test_dir:     $TEST_DIR
timeout_sec:  $TIMEOUT_SEC
start:        $START_TS
end:          $END_TS
elapsed_sec:  $ELAPSED
exit_code:    $RC
kernel:       $KVER
srcversion:   $SRCVER
trace_events:
$(printf '  - %s\n' "${EVENTS[@]}")
EOF

TRACE_LINES=$(wc -l < "$TRACE_FILE")
DMESG_LINES=$(wc -l < "$DMESG_FILE")
echo "[done] $LABEL: rc=$RC elapsed=${ELAPSED}s trace=${TRACE_LINES}lines dmesg=${DMESG_LINES}lines"
echo "       outputs in $OUT_DIR/${LABEL}.{trace,dmesg,test.log,meta,rc}"
