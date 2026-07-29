#!/usr/bin/env bash
set -euo pipefail

[[ "${1:-}" == "--map-smoke" && $# == 1 ]] || {
    echo "usage: $0 --map-smoke" >&2
    exit 2
}

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
readonly ROOT
readonly SERVER="$ROOT/build/tools/phoenix-vfio-user/phoenix-vfio-user"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_ID
readonly RUN_DIR="$ROOT/build/experiments/phoenix-vfio-user/$RUN_ID"
readonly VFIO_SOCKET="/tmp/xdna-emu-vfio-$$.sock"
readonly MONITOR_SOCKET="/tmp/xdna-emu-monitor-$$.sock"
readonly RESPONSE="$RUN_DIR/server-nonce.bin"

for tool in qemu-system-x86_64 socat xxd; do
    command -v "$tool" >/dev/null || {
        echo "missing required tool: $tool" >&2
        exit 1
    }
done

qemu-system-x86_64 --version | head -n 1 |
    grep -Fqx "QEMU emulator version 10.2.1 (Debian 1:10.2.1+ds-1ubuntu3.1)" || {
    echo "map smoke requires the pinned QEMU 10.2.1 package" >&2
    exit 1
}

mkdir -p "$RUN_DIR"
"$ROOT/tools/phoenix-vfio-user/build.sh" >"$RUN_DIR/build.log" 2>&1

server_pid=
qemu_pid=
cleanup() {
    set +e
    [[ -z "$qemu_pid" ]] || kill "$qemu_pid" 2>/dev/null
    [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null
    [[ -z "$qemu_pid" ]] || wait "$qemu_pid" 2>/dev/null
    [[ -z "$server_pid" ]] || wait "$server_pid" 2>/dev/null
    rm -f "$VFIO_SOCKET" "$MONITOR_SOCKET"
}
trap cleanup EXIT INT TERM

wait_for_socket() {
    local path=$1
    local pid=$2
    for ((attempt = 0; attempt < 200; ++attempt)); do
        [[ -S "$path" ]] && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 0.05
    done
    return 1
}

wait_for_log() {
    local pattern=$1
    local file=$2
    local pid=$3
    for ((attempt = 0; attempt < 200; ++attempt)); do
        grep -Fq "$pattern" "$file" && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 0.05
    done
    return 1
}

wait_for_exit() {
    local pid=$1
    for ((attempt = 0; attempt < 200; ++attempt)); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.05
    done
    return 1
}

"$SERVER" --map-smoke "$VFIO_SOCKET" \
    >"$RUN_DIR/server.log" 2>&1 &
server_pid=$!
wait_for_socket "$VFIO_SOCKET" "$server_pid" || {
    echo "vfio-user server did not open its socket; see $RUN_DIR/server.log" >&2
    exit 1
}

readonly VFIO_DEVICE="{\"driver\":\"vfio-user-pci\",\"socket\":{\"path\":\"$VFIO_SOCKET\",\"type\":\"unix\"}}"
qemu=(
    qemu-system-x86_64
    -no-user-config
    -nodefaults
    -machine "q35,accel=tcg,memory-backend=ram"
    -m 2G
    -object "memory-backend-memfd,id=ram,size=2G,share=on"
    -S
    -display none
    -serial none
    -monitor "unix:$MONITOR_SOCKET,server=on,wait=off"
    -device "loader,addr=0x60001000,data=0x47554553544e5055,data-len=8"
    -device "$VFIO_DEVICE"
)
printf '%q ' "${qemu[@]}" >"$RUN_DIR/qemu-command.txt"
printf '\n' >>"$RUN_DIR/qemu-command.txt"

"${qemu[@]}" >"$RUN_DIR/qemu.log" 2>&1 &
qemu_pid=$!
wait_for_socket "$MONITOR_SOCKET" "$qemu_pid" || {
    echo "QEMU did not open its monitor; see $RUN_DIR/qemu.log" >&2
    exit 1
}
wait_for_log "map-smoke: guest nonce observed; server nonce published" \
    "$RUN_DIR/server.log" "$server_pid" || {
    echo "server did not publish the nonce; logs retained in $RUN_DIR" >&2
    exit 1
}

printf 'pmemsave 0x60001008 8 "%s"\n' "$RESPONSE" |
    socat -T 5 - "UNIX-CONNECT:$MONITOR_SOCKET" \
        >"$RUN_DIR/monitor-pmemsave.log"
[[ -f "$RESPONSE" ]] || {
    echo "QEMU did not save the server nonce; logs retained in $RUN_DIR" >&2
    exit 1
}
actual="$(xxd -p -c 8 "$RESPONSE")"
[[ "$actual" == "504e524556524553" ]] || {
    echo "QEMU observed server nonce $actual, expected 504e524556524553" >&2
    exit 1
}

printf 'quit\n' |
    socat -T 5 - "UNIX-CONNECT:$MONITOR_SOCKET" \
        >"$RUN_DIR/monitor-quit.log"
wait_for_exit "$qemu_pid" || {
    echo "QEMU did not exit after quit; logs retained in $RUN_DIR" >&2
    exit 1
}
if ! wait "$qemu_pid"; then
    echo "QEMU exited unsuccessfully; logs retained in $RUN_DIR" >&2
    exit 1
fi
qemu_pid=

wait_for_exit "$server_pid" || {
    echo "vfio-user server did not exit after QEMU; logs retained in $RUN_DIR" >&2
    exit 1
}
if ! wait "$server_pid"; then
    echo "vfio-user server rejected the map contract; logs retained in $RUN_DIR" >&2
    exit 1
fi
server_pid=
grep -Fq "map-smoke: PASS" "$RUN_DIR/server.log"

echo "phoenix vfio-user QEMU map smoke: PASS"
echo "evidence: $RUN_DIR"
