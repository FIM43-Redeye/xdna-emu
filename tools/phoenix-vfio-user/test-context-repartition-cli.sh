#!/bin/bash
set -u -o pipefail

readonly PRODUCER=${1:?usage: test-context-repartition-cli.sh PRODUCER}

check_mode() {
    local marker=$1
    local output
    local status
    shift

    output=$("$PRODUCER" "$@" 2>&1)
    status=$?
    if ((status != 1)); then
        printf 'expected exit 1 for %s, got %d\n%s\n' "$marker" "$status" "$output" >&2
        exit 1
    fi
    if [[ "$output" != *"$marker"* ]]; then
        printf 'missing marker %s\n%s\n' "$marker" "$output" >&2
        exit 1
    fi
}

check_mode PHOENIX_REPARTITION_FAIL: \
    missing-a.xclbin missing-a.insts missing-b.xclbin missing-b.insts
check_mode PHOENIX_CONTEXT_REPEAT_FAIL: \
    --same-context-repeat missing-a.xclbin missing-a.insts
check_mode PHOENIX_TDR_RETRY_FAIL: \
    --immediate-post-tdr-retry missing-a.xclbin missing-a.insts
check_mode PHOENIX_POST_REPLAY_RETRY_FAIL: \
    --post-replay-tdr-retry /dev/null missing-a.xclbin missing-a.insts

printf 'context-repartition CLI routing: PASS\n'
