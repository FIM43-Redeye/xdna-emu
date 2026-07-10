#!/usr/bin/env bash
# Claude Code SessionStart hook: activate the NPU toolchain environment.
#
# Sources activate-npu-env.sh and exports key variables to CLAUDE_ENV_FILE
# so they persist across all Bash tool calls in the session.

set -euo pipefail

export NPU_WORK_DIR="/home/triple/npu-work"
ACTIVATE_SCRIPT="$NPU_WORK_DIR/toolchain-build/activate-npu-env.sh"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "Warning: $ACTIVATE_SCRIPT not found, skipping NPU env activation" >&2
    exit 0
fi

# The activation script guards against being executed (vs sourced) via
# BASH_SOURCE[0] == $0. In a hook we're executing a wrapper that sources
# the inner script, so this is fine. But the inner script uses nounset,
# so pre-export the variables it checks.
source "$ACTIVATE_SCRIPT"

# Write key environment variables to CLAUDE_ENV_FILE so they persist
# across all subsequent Bash tool calls in this session.
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    {
        echo "PATH=$PATH"
        echo "PYTHONPATH=${PYTHONPATH:-}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        # mlir-aie / Peano
        [[ -n "${MLIR_AIE_DIR:-}" ]] && echo "MLIR_AIE_DIR=$MLIR_AIE_DIR"
        [[ -n "${PEANO_INSTALL_DIR:-}" ]] && echo "PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR"
        # aietools / Chess
        [[ -n "${XILINX_VITIS_AIETOOLS:-}" ]] && echo "XILINX_VITIS_AIETOOLS=$XILINX_VITIS_AIETOOLS"
        [[ -n "${XILINX_VITIS:-}" ]] && echo "XILINX_VITIS=$XILINX_VITIS"
        [[ -n "${AIETOOLS_DIR:-}" ]] && echo "AIETOOLS_DIR=$AIETOOLS_DIR"
        # XRT
        [[ -n "${XILINX_XRT:-}" ]] && echo "XILINX_XRT=$XILINX_XRT"
        # NPU work
        [[ -n "${NPU_WORK_DIR:-}" ]] && echo "NPU_WORK_DIR=$NPU_WORK_DIR"
    } >> "$CLAUDE_ENV_FILE"
fi
