#!/usr/bin/env bash
# Run the known-RED Phoenix signed-firmware PM-fault scheduler gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
readonly SCRIPT_DIR ROOT

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_file() {
    local path="$1"
    local description="$2"
    [[ -f "$path" ]] || die "$description is missing: $path"
}

discover_npu_work() {
    if [[ -n "${NPU_WORK_DIR:-}" ]]; then
        printf '%s\n' "$NPU_WORK_DIR"
        return
    fi

    local common_git_dir
    common_git_dir="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)"
    dirname "$(dirname "$common_git_dir")"
}

NPU_WORK="$(discover_npu_work)"
readonly NPU_WORK
readonly FIRMWARE="${XDNA_FIRMWARE-/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin}"
readonly MLIR_SOURCE="${MLIR_AIE_PATH-$NPU_WORK/mlir-aie}"
readonly MLIR_BUILD="${MLIR_AIE_BUILD-$MLIR_SOURCE/build}"
readonly FIXTURE="$MLIR_BUILD/test/npu-xrt/add_one_using_dma/chess"
readonly FAULT_PDI="${XDNA_PM_ERROR_PDI:-}"
readonly NICE_LEVEL="${NICE_LEVEL:-19}"

[[ -n "$FAULT_PDI" ]] || die "XDNA_PM_ERROR_PDI must name the native PM-fault PDI"
[[ -n "$MLIR_SOURCE" ]] || die "MLIR_AIE_PATH is set but blank"
[[ -n "$MLIR_BUILD" ]] || die "MLIR_AIE_BUILD is set but blank"
require_file "$FIRMWARE" "Phoenix firmware"
require_file \
    "$MLIR_SOURCE/lib/Dialect/AIE/Util/aie_registers_aie2.json" \
    "mlir-aie AIE2 register database"
require_file "$FIXTURE/aie.xclbin" "Chess add_one_using_dma XCLBIN"
require_file "$FIXTURE/insts.bin" "Chess add_one_using_dma instruction stream"
require_file "$FAULT_PDI" "native PM-fault PDI"

cd "$ROOT"
exec env \
    XDNA_FIRMWARE="$FIRMWARE" \
    MLIR_AIE_PATH="$MLIR_SOURCE" \
    MLIR_AIE_BUILD="$MLIR_BUILD" \
    XDNA_PM_ERROR_PDI="$FAULT_PDI" \
    nice -n "$NICE_LEVEL" \
    cargo test -p xdna-emu --lib \
    firmware::boot_tests::guards::m2c_chained_pm_fault_publishes_native_core_error \
    -- --ignored --exact --nocapture
