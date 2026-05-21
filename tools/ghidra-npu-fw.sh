#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# ghidra-npu-fw.sh -- headless Ghidra analysis pipeline for the Phoenix
# NPU firmware.
#
# The firmware (npu.dev.sbin) is a PSP $PS1-signed blob wrapping a raw
# Xtensa LX7 little-endian image with no ELF headers.  See
# docs/superpowers/findings/2026-05-20-npu-firmware-format.md for the
# format and architecture characterization.
#
# This script:
#   1. prepare -- extracts the plaintext body from the signed .sbin.
#   2. analyze -- runs Ghidra's headless analyzer on the body with the
#      Xtensa language, rebases it to the recovered load address so
#      l32r literal pools resolve, runs full auto-analysis, and dumps
#      text artifacts (functions.tsv, strings.tsv, disasm.txt).
#
# Ghidra's GUI is not scriptable for this workflow; everything is
# CLI-driven and reproducible.  Outputs land in <WORK_DIR>/analysis-xtensa/.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override via environment) ---------------------------
GHIDRA_DIR=${GHIDRA_DIR:-/home/triple/npu-work/ghidra}
FW_SBIN=${FW_SBIN:-/lib/firmware/amdnpu/1502_00/npu.dev.sbin}
WORK_DIR=${WORK_DIR:-/home/triple/npu-work/ghidra-projects/npu-fw}
GHIDRA_SCRIPTS=${GHIDRA_SCRIPTS:-$SCRIPT_DIR/ghidra-scripts}

# Xtensa little-endian; load base recovered by tools/fw-find-base.py.
LANGUAGE=${LANGUAGE:-Xtensa:LE:32:default}
LOAD_BASE=${LOAD_BASE:-08ad3000}

# PSP $PS1 container: 256-byte header, trailing 256-byte signature.
HEADER_SIZE=256
SIG_SIZE=256

BODY_BIN="$WORK_DIR/npu-fw-body.bin"
ANALYSIS_DIR="$WORK_DIR/analysis-xtensa"
PROJ_LOC=$(dirname "$WORK_DIR")
PROJ_NAME="npu-fw-xtensa"

cmd_prepare() {
    [[ -r "$FW_SBIN" ]] || {
        echo "FATAL: NPU FW not readable at $FW_SBIN" >&2
        exit 1
    }
    mkdir -p "$WORK_DIR"

    local file_size body_size
    file_size=$(stat -c %s "$FW_SBIN")
    body_size=$((file_size - HEADER_SIZE - SIG_SIZE))

    echo "FW file:    $FW_SBIN  ($file_size bytes)"
    echo "Body:       $body_size bytes  (file - 0x100 header - 0x100 signature)"
    echo "Extracting: $BODY_BIN"
    dd if="$FW_SBIN" of="$BODY_BIN" bs=1 skip=$HEADER_SIZE count=$body_size status=none
    echo "Done."
}

cmd_analyze() {
    [[ -d "$GHIDRA_DIR" ]] || {
        echo "FATAL: Ghidra not found at $GHIDRA_DIR (set GHIDRA_DIR)" >&2
        exit 1
    }
    [[ -r "$BODY_BIN" ]] || {
        echo "FATAL: body not extracted; run '$(basename "$0") prepare' first." >&2
        exit 1
    }
    mkdir -p "$ANALYSIS_DIR"

    echo "Headless analysis:"
    echo "  language:   $LANGUAGE"
    echo "  load base:  0x$LOAD_BASE"
    echo "  project:    $PROJ_LOC/$PROJ_NAME"
    echo "  output:     $ANALYSIS_DIR"
    echo

    nice -n 19 "$GHIDRA_DIR/support/analyzeHeadless" \
        "$PROJ_LOC" "$PROJ_NAME" \
        -import "$BODY_BIN" \
        -processor "$LANGUAGE" \
        -scriptPath "$GHIDRA_SCRIPTS" \
        -preScript SetImageBase.java "$LOAD_BASE" \
        -postScript DumpNpuFw.java "$ANALYSIS_DIR" \
        -overwrite

    echo
    echo "Artifacts:"
    ls -la "$ANALYSIS_DIR"
}

case "${1:-help}" in
    prepare)  cmd_prepare ;;
    analyze)  cmd_analyze ;;
    all)      cmd_prepare && cmd_analyze ;;
    help|*)
        echo "usage: $(basename "$0") {prepare | analyze | all}"
        echo
        echo "  prepare  -- extract the plaintext body from npu.dev.sbin"
        echo "  analyze  -- run headless Ghidra (Xtensa) + dump text artifacts"
        echo "  all      -- prepare then analyze"
        echo
        echo "Environment overrides: GHIDRA_DIR, FW_SBIN, WORK_DIR,"
        echo "  GHIDRA_SCRIPTS, LANGUAGE, LOAD_BASE"
        ;;
esac
