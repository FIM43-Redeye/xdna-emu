#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# ghidra-npu-fw.sh -- one-shot prep for opening the Phoenix NPU FW in
# Ghidra.  Extracts the plaintext body from the PSP-signed .sbin,
# launches Ghidra, and points the user at the project location.
#
# Per docs/superpowers/findings/2026-05-20-npu-firmware-format.md, the
# body is plaintext ARM Thumb-2 code+data, two sections separated by a
# 120KB virtual-address gap.  This script extracts the body once and
# kicks Ghidra; the user then needs to:
#
#   1. Create a new project (or open existing).
#   2. Import npu-fw-body.bin.
#   3. Pick processor: ARM Cortex (try v7-M or v7-A Thumb first).
#   4. Set load address to 0 (or the recovered load address, see doc).
#   5. Analyze.
#   6. After auto-analysis: define memory regions for the two
#      sections, mark string regions as ASCII, follow PC-relative refs.

set -euo pipefail

# Paths
GHIDRA_DIR=${GHIDRA_DIR:-/home/triple/npu-work/ghidra}
FW_SBIN=${FW_SBIN:-/lib/firmware/amdnpu/1502_00/npu.dev.sbin}
WORK_DIR=${WORK_DIR:-/home/triple/npu-work/ghidra-projects/npu-fw}
BODY_BIN="$WORK_DIR/npu-fw-body.bin"

# Body layout per the finding doc:
#   header: 0x000 - 0x100 (256 bytes, $PS1 signed header)
#   body:   0x100 - 0x3CA10 (248080 bytes, plaintext code+data)
#   signature: last 256 bytes
HEADER_SIZE=256
SIG_SIZE=256

cmd_prepare() {
    [[ -d "$GHIDRA_DIR" ]] || {
        echo "FATAL: Ghidra not found at $GHIDRA_DIR" >&2
        echo "       Set GHIDRA_DIR to override, or extract to /home/triple/npu-work/ghidra/" >&2
        exit 1
    }
    [[ -r "$FW_SBIN" ]] || {
        echo "FATAL: NPU FW not readable at $FW_SBIN" >&2
        exit 1
    }

    mkdir -p "$WORK_DIR"

    local file_size
    file_size=$(stat -c %s "$FW_SBIN")
    local body_size=$((file_size - HEADER_SIZE - SIG_SIZE))

    echo "FW file:    $FW_SBIN  ($file_size bytes)"
    echo "Body size:  $body_size bytes (file - header - signature)"
    echo "Extracting body to: $BODY_BIN"

    dd if="$FW_SBIN" of="$BODY_BIN" bs=1 skip=$HEADER_SIZE count=$body_size status=none

    # Also extract just section 1 and section 2 as separate files
    # so they can be loaded at their respective virtual addresses.
    # Per the finding doc, these are at body offsets:
    #   Section 1: body 0x3F00 (file 0x4000) - body 0xEF00 (file 0xF000)  ~44KB
    #   Section 2: body 0x2C700 (file 0x2D000) - body 0x3C700 (file 0x3CA00) ~64KB
    local sec1_start=$((0x4000 - HEADER_SIZE))   # 0x3F00 in body coords
    local sec1_end=$((0xF000 - HEADER_SIZE))
    local sec1_size=$((sec1_end - sec1_start))
    local sec2_start=$((0x2D000 - HEADER_SIZE))  # 0x2CF00 in body coords
    local sec2_end=$((0x3CA00 - HEADER_SIZE))
    local sec2_size=$((sec2_end - sec2_start))

    dd if="$BODY_BIN" of="$WORK_DIR/section1.bin" \
        bs=1 skip=$sec1_start count=$sec1_size status=none
    dd if="$BODY_BIN" of="$WORK_DIR/section2.bin" \
        bs=1 skip=$sec2_start count=$sec2_size status=none

    echo
    echo "Prepared inputs:"
    ls -la "$WORK_DIR/"*.bin
    echo
    echo "Next: run '$(basename "$0") launch' to start Ghidra."
}

cmd_launch() {
    [[ -d "$GHIDRA_DIR" ]] || {
        echo "FATAL: Ghidra not found at $GHIDRA_DIR" >&2
        exit 1
    }
    [[ -r "$BODY_BIN" ]] || {
        echo "FATAL: body not extracted yet; run '$(basename "$0") prepare' first." >&2
        exit 1
    }
    echo "Launching Ghidra (project files will live in $WORK_DIR/)..."
    echo
    echo "In Ghidra:"
    echo "  File -> New Project -> Non-Shared, location: $WORK_DIR"
    echo "  File -> Import File... -> $BODY_BIN"
    echo "  Processor: ARM Cortex (try v7-M Little Endian first; switch to v7-A if vector table doesn't fit Cortex-M shape)"
    echo "  Block Name: BODY, Base Address: 0x00000000 (refine after vector-table inspection)"
    echo "  Analyze -> ARM Aggressive Instruction Finder + standard analyzers"
    echo
    exec "$GHIDRA_DIR/ghidraRun"
}

case "${1:-help}" in
    prepare)  cmd_prepare ;;
    launch)   cmd_launch ;;
    both)     cmd_prepare && cmd_launch ;;
    help|*)
        echo "usage: $(basename "$0") {prepare | launch | both}"
        echo
        echo "  prepare  -- extract body + sections from npu.dev.sbin into $WORK_DIR"
        echo "  launch   -- start Ghidra (run 'prepare' first if you haven't)"
        echo "  both     -- prepare then launch"
        echo
        echo "Environment overrides: GHIDRA_DIR, FW_SBIN, WORK_DIR"
        ;;
esac
