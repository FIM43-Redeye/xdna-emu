#!/usr/bin/env bash
set -euo pipefail

case "${1:-}:$#" in
    --map-smoke:1 | --driver-probe:1 | --run-npu-direct:1 | \
        --run-context-repartition:1 | --run-async-error:1 | \
        --run-async-error-batch:1 | --probe-phoenix-npi-read:1) ;;
    --run-real-column-gate:3)
        case "$2" in
            control | treatment) ;;
            *)
                echo "real column-gate arm must be control or treatment" >&2
                exit 2
                ;;
        esac
        ;;
    --run-frozen:2 | --run-frozen-direct:2 | --run-pinned-elf:2)
        case "$2" in
            chess | peano) ;;
            *)
                echo "compiler must be chess or peano" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        echo "usage: $0 --map-smoke | --driver-probe | --probe-phoenix-npi-read | --run-frozen chess|peano | --run-frozen-direct chess|peano | --run-npu-direct | --run-pinned-elf chess|peano | --run-context-repartition | --run-async-error | --run-async-error-batch | --run-real-column-gate control|treatment <pair-dir>" >&2
        exit 2
        ;;
esac

MODE=$1
ASYNC_BATCH_ONLY=false
if [[ "$MODE" == "--run-async-error-batch" ]]; then
    MODE=--run-async-error
    ASYNC_BATCH_ONLY=true
fi
readonly MODE ASYNC_BATCH_ONLY
FROZEN_COMPILER=
ELF_COMPILER=
GATE_ARM=
GATE_ROOT_INPUT=
case "$MODE" in
    --run-frozen | --run-frozen-direct) FROZEN_COMPILER=$2 ;;
    --run-pinned-elf) ELF_COMPILER=$2 ;;
    --run-real-column-gate)
        GATE_ARM=$2
        GATE_ROOT_INPUT=$3
        ;;
esac
readonly FROZEN_COMPILER ELF_COMPILER GATE_ARM GATE_ROOT_INPUT
FROZEN_EXECUTION=
case "$MODE" in
    --run-frozen) FROZEN_EXECUTION=cmdlist ;;
    --run-frozen-direct) FROZEN_EXECUTION=direct ;;
esac
readonly FROZEN_EXECUTION
NEEDS_XRT=false
if [[ -n "$FROZEN_COMPILER$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ||
    "$MODE" == "--run-context-repartition" || "$MODE" == "--run-async-error" ||
    -n "$GATE_ARM" ]]; then
    NEEDS_XRT=true
fi
readonly NEEDS_XRT
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
readonly ROOT
COMMON_GIT="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)"
readonly COMMON_GIT
readonly SHARED_ROOT="$(dirname "$COMMON_GIT")"
NPU_WORK="$(dirname "$(dirname "$COMMON_GIT")")"
readonly NPU_WORK
GATE_ROOT=
if [[ -n "$GATE_ARM" ]]; then
    command -v realpath >/dev/null || {
        echo "missing required tool: realpath" >&2
        exit 1
    }
    GATE_ROOT="$(realpath -e "$GATE_ROOT_INPUT")"
    [[ -d "$GATE_ROOT" ]] || {
        echo "real column-gate pair directory is not a directory: $GATE_ROOT" >&2
        exit 1
    }
fi
readonly GATE_ROOT
readonly MLIR_AIE_PATH="$NPU_WORK/mlir-aie"
readonly REGISTER_DB="$MLIR_AIE_PATH/lib/Dialect/AIE/Util/aie_registers_aie2.json"
readonly FIXTURE_ROOT="$NPU_WORK/fixtures/phoenix-vfio-user/v1"
readonly FROZEN_ROOT="$FIXTURE_ROOT/add_one_using_dma"
readonly FROZEN_TEST="$FROZEN_ROOT/test.exe"
readonly ELF_ROOT="$FIXTURE_ROOT/add_one_objFifo_elf"
readonly ELF_TEST="$ELF_ROOT/test.exe"
readonly REPARTITION_ROOT="$FIXTURE_ROOT/device_width/chess"
readonly REPARTITION_SOURCE="$ROOT/tools/phoenix-vfio-user/context-repartition.cpp"
readonly ERROR_PDI="${XDNA_ERROR_PDI:-$SHARED_ROOT/build/experiments/firmware-error-network-phoenix-20260804/error-main.pdi}"
readonly ERROR_INIT_CDO="${ERROR_PDI%/*}/error-init.cdo"
readonly MLIR_AIE_BUILD="${MLIR_AIE_BUILD:-$MLIR_AIE_PATH/build}"
readonly ASYNC_CDO_ROOT="$MLIR_AIE_BUILD/test/npu-xrt/add_one_using_dma/chess/aie_arch.mlir.prj/cdo_main"
readonly ASYNC_CONTROL_ELF="$MLIR_AIE_BUILD/test/npu-xrt/add_one_using_dma/chess/aie_arch.mlir.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf"
readonly AIE_TRANSLATE="$MLIR_AIE_PATH/install/bin/aie-translate"
readonly BOOTGEN="$MLIR_AIE_PATH/install/bin/bootgen"
readonly PEANO_INSTALL="$NPU_WORK/llvm-aie/install"
readonly XRT_ROOT=/opt/xilinx/xrt
readonly XRT_COREUTIL="$XRT_ROOT/lib/libxrt_coreutil.so.2"
readonly XRT_COREUTIL_VERSIONED="$XRT_ROOT/lib/libxrt_coreutil.so.2.26.0"
readonly XRT_CORE="$XRT_ROOT/lib/libxrt_core.so.2"
readonly XRT_CORE_VERSIONED="$XRT_ROOT/lib/libxrt_core.so.2.26.0"
readonly XRT_XDNA="$XRT_ROOT/lib/libxrt_driver_xdna.so.2"
readonly XRT_XDNA_VERSIONED="$XRT_ROOT/lib/libxrt_driver_xdna.so.2.26.0"
readonly XRT_RUNNER="$XRT_ROOT/bin/unwrapped/xrt-runner"
readonly XRT_PHOENIX_ARCHIVE="$XRT_ROOT/share/amdxdna/bins/xrt_smi_phx.a"
readonly GATE_FIXTURE="$SHARED_ROOT/build/experiments/phoenix-pm-fault-array-ordering/20260804T194245Z/edge-compute-mm2s"
readonly GATE_XCLBIN="$GATE_FIXTURE/fault-package/aie.xclbin"
readonly GATE_MLIR="$GATE_FIXTURE/fault-package/work/input_with_addresses.mlir"
readonly GATE_EXPECTED_OUTPUT="$GATE_FIXTURE/hw.out.bin"
readonly GATE_CANARY_INSTS="$SHARED_ROOT/build/experiments/phoenix-pm-clock-characterization/20260805T232931Z-shim-witness/full-witness-fault.insts.bin"
readonly GATE_RUNNER="$ROOT/bridge-runner/build/bridge-trace-runner"
readonly GATE_CLASSIFIER="$ROOT/target/debug/libxdna_emu.so"
readonly GATE_CLOCK_QUERY_SOURCE="$ROOT/tools/xdna-clock-query.cpp"
readonly NPI_READ_PATCH="$ROOT/docs/patches/0004-LOCAL-phoenix-read-only-npi-lock-probe.patch"
readonly PROTECTED_GATE_PATCH="$ROOT/docs/patches/0005-LOCAL-phoenix-protected-column-gate.patch"
export MLIR_AIE_PATH
readonly SERVER="$ROOT/build/tools/phoenix-vfio-user/phoenix-vfio-user"
readonly DRIVER_PIN=216cefececd74effcd7a88350c71b99f5ef9a215
readonly FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin
readonly FIRMWARE_SHA256=d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e
readonly GUEST_KERNEL_VERSION=7.1.6-custom+
readonly GUEST_KERNEL=/boot/vmlinuz-7.1.6-custom+
readonly GUEST_KERNEL_SHA256=b56fcaca980ece4c3f8f783086aceca2a1c4ae018dda994e5cb1657a90c4b63f
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_ID
if [[ -n "$GATE_ARM" ]]; then
    RUN_DIR="$GATE_ROOT/kvm/$GATE_ARM-$RUN_ID"
else
    RUN_DIR="$ROOT/build/experiments/phoenix-vfio-user/$RUN_ID"
fi
readonly RUN_DIR
readonly VFIO_SOCKET="/tmp/xdna-emu-vfio-$$.sock"
readonly MONITOR_SOCKET="/tmp/xdna-emu-monitor-$$.sock"
readonly RESPONSE="$RUN_DIR/server-nonce.bin"
readonly DRIVER_SOURCE="$RUN_DIR/driver-source"
readonly GUEST_ROOT="$RUN_DIR/guest-root"
readonly INITRAMFS="$RUN_DIR/initramfs.cpio.gz"
readonly GUEST_LOG="$RUN_DIR/guest.log"
readonly GATE_CLOCK_QUERY="$RUN_DIR/xdna-clock-query"
readonly GATE_INSTS="$GATE_CANARY_INSTS"
readonly GATE_CONTROL_MARKER="${GATE_ROOT:+$GATE_ROOT/kvm/control-safety-qualified}"
readonly REPARTITION_PRODUCER="$RUN_DIR/context-repartition"
readonly ASYNC_ROOT="$RUN_DIR/async-error"
readonly ASYNC_XCLBIN="$ASYNC_ROOT/aie.xclbin"
readonly ASYNC_PM_XCLBIN="$ASYNC_ROOT/PM.xclbin"
readonly ASYNC_PM_ELF="$ASYNC_ROOT/PM.elf"
readonly ASYNC_PM_MLIR="$ASYNC_ROOT/PM.mlir"
readonly ASYNC_PM_CDO_ROOT="$ASYNC_ROOT/pm-cdo"
readonly ASYNC_PM_BIF="$ASYNC_ROOT/PM.bif"
readonly ASYNC_PM_PDI="$ASYNC_ROOT/PM.pdi"
readonly ASYNC_INSTS_PM="$ASYNC_ROOT/PM.insts"
readonly ASYNC_INSTS_BATCH_CORE="$ASYNC_ROOT/batch-core-event.insts"
readonly ASYNC_INSTS_BATCH="$ASYNC_ROOT/BATCH.insts"
readonly ASYNC_INSTS_A="$ASYNC_ROOT/A.insts"
readonly ASYNC_INSTS_B="$ASYNC_ROOT/B.insts"
readonly ASYNC_INSTS_C="$ASYNC_ROOT/C.insts"
readonly ASYNC_INSTS_D="$ASYNC_ROOT/D.insts"
readonly ASYNC_INSTS_E="$ASYNC_ROOT/E.insts"
readonly ASYNC_INSTS_SHIM_CLEAR="$ASYNC_ROOT/shim-clear.insts"
readonly ASYNC_INSTS_S2MM="$ASYNC_ROOT/S2MM.insts"
readonly ASYNC_INSTS_F="$ASYNC_ROOT/F.insts"

for tool in git nice qemu-system-x86_64; do
    command -v "$tool" >/dev/null || {
        echo "missing required tool: $tool" >&2
        exit 1
    }
done
if [[ "$MODE" == "--map-smoke" ]]; then
    required_tools=(socat xxd)
else
    required_tools=(awk cpio cp depmod dpkg dpkg-query find gzip install ldd
        ln lspci make modinfo modprobe sed sha256sum sort tar tr)
    [[ "$MODE" != "--run-npu-direct" ]] || required_tools+=(ar)
    if [[ "$MODE" == "--run-context-repartition" || "$MODE" == "--run-async-error" ]]; then
        required_tools+=(c++)
    fi
    [[ "$MODE" != "--run-async-error" ]] || required_tools+=(python3 xclbinutil)
    [[ -z "$GATE_ARM" ]] || required_tools+=(base64 c++ cmake python3)
fi
for tool in "${required_tools[@]}"; do
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

if [[ -n "$GATE_ARM" || "$MODE" == "--probe-phoenix-npi-read" ]]; then
    [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
        echo "physical-candidate KVM run requires a clean source worktree" >&2
        exit 1
    }
fi
if [[ -n "$GATE_ARM" ]]; then
    [[ "$(sha256sum "$GATE_INSTS" | awk '{print $1}')" == \
        f6329e498d8d254e6522eb0a960c3b8305991f758344e3575f42bc11596f5af1 &&
        "$(sha256sum "$GATE_XCLBIN" | awk '{print $1}')" == \
        d25ab5b8b45a0119c7a62efbe291599020adf86e27609fdc01a6346637ab51b3 &&
        "$(sha256sum "$GATE_MLIR" | awk '{print $1}')" == \
        1e8ef843bca74767fd4b41f8da92dbfbe98b95fe15df425158ffc1fff46baf45 &&
        "$(sha256sum "$GATE_EXPECTED_OUTPUT" | awk '{print $1}')" == \
        64ed86b909d6d0502b64b28db0ea1272ffb358e20e9b1d88b63ccb07fa900cf5 ]] || {
        echo "real column-gate artifacts do not match the protected-gate pins" >&2
        exit 1
    }
    if [[ "$GATE_ARM" == treatment && ! -s "$GATE_CONTROL_MARKER" ]]; then
        echo "treatment requires a safety-qualified KVM control marker: $GATE_CONTROL_MARKER" >&2
        exit 1
    fi
fi

mkdir -p "$RUN_DIR"
if [[ -n "$GATE_ARM" ]]; then
    nice -n 19 cmake --build "$ROOT/bridge-runner/build" \
        --target bridge-trace-runner >"$RUN_DIR/runner-build.log" 2>&1
    nice -n 19 c++ -std=c++20 -O2 -Wall -Wextra -Werror \
        -I"$NPU_WORK/xdna-driver/include/uapi" \
        "$GATE_CLOCK_QUERY_SOURCE" -o "$GATE_CLOCK_QUERY" \
        >"$RUN_DIR/clock-query-build.log" 2>&1
fi
nice -n 19 "$ROOT/tools/phoenix-vfio-user/build.sh" >"$RUN_DIR/build.log" 2>&1

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

wait_for_guest_result() {
    local qemu=$1
    local server=$2
    local attempts=1800
    if [[ "$MODE" == "--run-async-error" || -n "$GATE_ARM" ]]; then
        attempts=9000
    fi

    for ((attempt = 0; attempt < attempts; ++attempt)); do
        grep -Fq "PHOENIX_DRIVER_PROBE_PASS" "$GUEST_LOG" && return 0
        grep -Fq "PHOENIX_DRIVER_PROBE_FAIL:" "$GUEST_LOG" && return 2
        kill -0 "$qemu" 2>/dev/null || return 3
        kill -0 "$server" 2>/dev/null || return 4
        sleep 0.1
    done
    return 5
}

copy_host_file() {
    local source=$1
    local destination="$GUEST_ROOT$source"

    mkdir -p "${destination%/*}"
    cp -L --preserve=mode,timestamps "$source" "$destination"
}

extract_guest_section() {
    local begin=$1
    local end=$2
    local destination=$3

    awk -v begin="$begin" -v end="$end" '
        $0 == begin { copying = 1; next }
        $0 == end { copying = 0; found_end = 1; exit }
        copying { print }
        END { exit !found_end }
    ' "$GUEST_LOG" >"$destination"
}

extract_guest_blob() {
    local name=$1
    local destination=$2
    local encoded="$RUN_DIR/$name.base64"

    extract_guest_section \
        "PHOENIX_REAL_COLUMN_GATE_BLOB_${name}_BEGIN" \
        "PHOENIX_REAL_COLUMN_GATE_BLOB_${name}_END" "$encoded"
    base64 -d "$encoded" | gzip -dc >"$destination"
}

classify_real_column_gate_run() {
    local qualified reason admitted disposition output_ok canary_ok kvm_gate
    local clock_before clock_after

    grep -Fqx "PHOENIX_REAL_COLUMN_GATE_MISMATCH_PASS" "$GUEST_LOG"
    grep -Fqx "PHOENIX_REAL_COLUMN_GATE_ARM_PASS $GATE_ARM" "$GUEST_LOG"
    grep -Fqx "PHOENIX_REAL_COLUMN_GATE_CANARY_PASS" "$GUEST_LOG"
    grep -Fqx "PHOENIX_REAL_COLUMN_GATE_GUEST_PASS $GATE_ARM" "$GUEST_LOG"
    grep -Fqx "force_cmdlist=Y" "$GUEST_LOG"

    extract_guest_section \
        PHOENIX_REAL_COLUMN_GATE_CLOCK_BEFORE_BEGIN \
        PHOENIX_REAL_COLUMN_GATE_CLOCK_BEFORE_END \
        "$RUN_DIR/clock-before.json"
    extract_guest_section \
        PHOENIX_REAL_COLUMN_GATE_CLOCK_AFTER_BEGIN \
        PHOENIX_REAL_COLUMN_GATE_CLOCK_AFTER_END \
        "$RUN_DIR/clock-after.json"
    for artifact in arm.stdout arm.stderr mismatch.stdout mismatch.stderr \
        canary.stdout canary.stderr; do
        extract_guest_section \
            "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_BEGIN" \
            "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_END" \
            "$RUN_DIR/$artifact"
    done
    extract_guest_blob arm.out.bin "$RUN_DIR/arm.out.bin"
    extract_guest_blob arm.trace.bin "$RUN_DIR/arm.trace.bin"
    extract_guest_blob canary.out.bin "$RUN_DIR/canary.out.bin"
    extract_guest_blob canary.trace.bin "$RUN_DIR/canary.trace.bin"

    grep -Fq \
        "live hardware context placement mismatch: expected 2:1, got 1:1" \
        "$RUN_DIR/mismatch.stderr"
    grep -Fq "verified live hw_context placement 1:1" "$RUN_DIR/arm.stderr"
    grep -Fq "verified live hw_context placement 1:1" "$RUN_DIR/canary.stderr"
    for runner_log in "$RUN_DIR/arm.stderr" "$RUN_DIR/canary.stderr"; do
        grep -Fq "classifier loaded from /run-real-column-gate/libxdna_emu.so" \
            "$runner_log"
        grep -Fq \
            "classifier roles: arg0=data_mm2s arg2=data_s2mm arg3=data_s2mm" \
            "$runner_log"
    done
    awk '
        /xdna_mailbox\.[0-9]+: opcode 0x18 size 24 id / {
            requests++
            if (requests == 2 && !destroyed_between)
                bad_order = 1
        }
        /xdna_mailbox\.[0-9]+: opcode 0x18 size 12 id / { responses++ }
        requests == 1 && /xdna_mailbox\.[0-9]+: opcode 0x3 size 4 id / {
            destroyed_between = 1
        }
        END {
            exit requests != 2 || responses != 2 ||
                !destroyed_between || bad_order
        }
    ' "$RUN_DIR/dmesg.log" || {
        echo "real column-gate submission/canary lifecycle differed; evidence: $RUN_DIR" >&2
        return 1
    }

    PYTHONPATH="$MLIR_AIE_PATH/install/python${PYTHONPATH:+:$PYTHONPATH}" \
        nice -n 19 python3 "$ROOT/tools/parse-trace.py" \
        --trace-bin "$RUN_DIR/arm.trace.bin" --xclbin-mlir "$GATE_MLIR" \
        --out-events "$RUN_DIR/arm.events.json" \
        --out-cycles "$RUN_DIR/arm.cycles.txt" \
        >"$RUN_DIR/parser.log" 2>&1

    nice -n 19 python3 "$ROOT/tools/phoenix-pm-clock-characterize.py" \
        classify-real-column-gate --arm "$GATE_ARM" \
        --events "$RUN_DIR/arm.events.json" \
        --output "$RUN_DIR/arm.out.bin" \
        --expected-output "$GATE_EXPECTED_OUTPUT" \
        --clock-before "$RUN_DIR/clock-before.json" \
        --clock-after "$RUN_DIR/clock-after.json" \
        --canary-output "$RUN_DIR/canary.out.bin" \
        --result "$RUN_DIR/result.json" \
        >"$RUN_DIR/classifier.log" 2>&1 || :
    [[ -f "$RUN_DIR/result.json" ]] || {
        echo "real column-gate classifier produced no result; evidence: $RUN_DIR" >&2
        return 1
    }
    IFS=$'\t' read -r qualified reason admitted disposition output_ok canary_ok < <(
        python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
print(str(r["qualified"]).lower(), r["classification"]["reason"],
      str(r["kvm_disposition"]["admitted"]).lower(),
      r["kvm_disposition"]["reason"],
      str(r["output"]["matches"]).lower(),
      str(r["canary"]["matches"]).lower(), sep="\t")
' "$RUN_DIR/result.json"
    )
    kvm_gate=FAIL
    if [[ "$admitted" == true ]]; then
        kvm_gate=PASS
    fi
    clock_before="$(tr -d '\n' <"$RUN_DIR/clock-before.json")"
    clock_after="$(tr -d '\n' <"$RUN_DIR/clock-after.json")"
    cat >"$RUN_DIR/receipt.md" <<EOF
# Phoenix real column-gate KVM $GATE_ARM receipt

- Behavioral result: **${qualified^^}** ($reason).
- KVM structural/lifecycle gate: **$kvm_gate** ($disposition).
- Physical freeze/resume conclusion: not established.
- Live placement: exact 1:1; deliberate 2:1 mismatch stopped before submission.
- Command completion: pass.
- Output exact: $output_ok.
- Fresh-context canary exact: $canary_ok.
- Clock before: $clock_before.
- Clock after: $clock_after.
- Raw arm evidence: arm.trace.bin, arm.events.json, arm.out.bin.
- Raw canary evidence: canary.trace.bin, canary.out.bin.
- Software tuple and hashes: tuple.txt; full classifier result: result.json.
- KVM boundary: this is driver-to-signed-firmware-to-emulated-array evidence. A physical host run remains unauthorized pending review of the complete KVM pair.
EOF
    if [[ "$admitted" != true ]]; then
        echo "phoenix vfio-user real column-gate $GATE_ARM: STOP ($reason; $disposition)" >&2
        echo "evidence: $RUN_DIR" >&2
        return 1
    fi
    if [[ "$GATE_ARM" == control ]]; then
        printf 'run=%s\ndisposition=%s\n' \
            "$RUN_DIR" "$disposition" >"$GATE_CONTROL_MARKER"
    fi
    echo "phoenix vfio-user real column-gate $GATE_ARM: KVM SAFETY PASS ($disposition)"
}

prepare_driver_guest() {
    local driver_repo
    local driver_archive="$RUN_DIR/driver-source.tar"
    local driver_module="$DRIVER_SOURCE/drivers/accel/amdxdna/amdxdna.ko"
    local driver_build_log="$RUN_DIR/driver-build.log"
    local initramfs_cpio="$RUN_DIR/initramfs.cpio"
    local -a async_pdis
    local dependency
    local elf_insts
    local elf_xclbin
    local elf_xclbin_hash
    local firmware_hash
    local frozen_xclbin
    local frozen_xclbin_hash
    local frozen_insts
    local guest_kernel_hash
    local library
    local module_signing_pem
    local module_vermagic
    local module_path
    local npu_direct_dir="$RUN_DIR/npu-direct"
    local qemu_package_version
    local repartition_insts="$REPARTITION_ROOT/insts.bin"
    local repartition_xclbin="$REPARTITION_ROOT/final.xclbin"
    local signature_field

    [[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]] || {
        echo "KVM is required for the pinned driver probe" >&2
        return 1
    }
    [[ -r "$REGISTER_DB" ]] || {
        echo "required AIE2 register database is missing: $REGISTER_DB" >&2
        return 1
    }
    qemu_package_version="$(dpkg-query -W -f='${Version}' qemu-system-x86)"
    [[ "$qemu_package_version" == "1:10.2.1+ds-1ubuntu3.1" ]] || {
        echo "installed QEMU package does not match the pinned version" >&2
        return 1
    }
    [[ -z "$(dpkg --verify qemu-system-x86)" ]] || {
        echo "installed QEMU package has locally modified files" >&2
        return 1
    }
    firmware_hash="$(sha256sum "$FIRMWARE" | awk '{print $1}')"
    [[ "$firmware_hash" == "$FIRMWARE_SHA256" ]] || {
        echo "Phoenix firmware does not match the pinned hash" >&2
        return 1
    }
    guest_kernel_hash="$(sha256sum "$GUEST_KERNEL" | awk '{print $1}')"
    [[ "$guest_kernel_hash" == "$GUEST_KERNEL_SHA256" ]] || {
        echo "guest kernel does not match the pinned hash" >&2
        return 1
    }
    if [[ -n "$FROZEN_COMPILER" ]]; then
        frozen_xclbin="$FROZEN_ROOT/$FROZEN_COMPILER/aie.xclbin"
        frozen_insts="$FROZEN_ROOT/$FROZEN_COMPILER/insts.bin"
        case "$FROZEN_COMPILER" in
            chess)
                frozen_xclbin_hash=b9f0e6bc43574859d1f1806e2ecb9ecd9af10ce0745f23f652f7aa860998f954
                ;;
            peano)
                frozen_xclbin_hash=71deb139ac91bba3a50099bfd0c3a4a966f00e1977eab017589ef51a36d63865
                ;;
        esac
        [[ "$(sha256sum "$FROZEN_TEST" | awk '{print $1}')" == \
            1888754a3efa669018c63de16c4f02773e75060547180f942c41464d9b60bb1b &&
            "$(sha256sum "$frozen_xclbin" | awk '{print $1}')" == "$frozen_xclbin_hash" &&
            "$(sha256sum "$frozen_insts" | awk '{print $1}')" == \
            ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896 ]] || {
            echo "frozen $FROZEN_COMPILER artifacts do not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ -n "$ELF_COMPILER" ]]; then
        elf_xclbin="$ELF_ROOT/$ELF_COMPILER/aie.xclbin"
        elf_insts="$ELF_ROOT/$ELF_COMPILER/insts.elf"
        case "$ELF_COMPILER" in
            chess)
                elf_xclbin_hash=46f9f27c66b89f388e21beb02a9c3731f686f4fd509701f9dd159e02e334b3fb
                ;;
            peano)
                elf_xclbin_hash=50f1a15df65a12b64bc2f3e6c3e647be0ee2c7798eeb8a1277c1111a2f55e7ca
                ;;
        esac
        [[ "$(sha256sum "$ELF_TEST" | awk '{print $1}')" == \
            2b4512e8c03ffdd1e078e35f533aa8e486be84c3901a9eafa75cc0915a7e725b &&
            "$(sha256sum "$elf_xclbin" | awk '{print $1}')" == "$elf_xclbin_hash" &&
            "$(sha256sum "$elf_insts" | awk '{print $1}')" == \
            23ff36c71ee6fc43265959921a00cae53bea2b44985c3c373bdc0df51065ca72 ]] || {
            echo "pinned ELF $ELF_COMPILER artifacts do not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ "$MODE" == "--run-context-repartition" || "$MODE" == "--run-async-error" ]]; then
        [[ "$(sha256sum "$FROZEN_ROOT/chess/aie.xclbin" | awk '{print $1}')" == \
            b9f0e6bc43574859d1f1806e2ecb9ecd9af10ce0745f23f652f7aa860998f954 &&
            "$(sha256sum "$FROZEN_ROOT/chess/insts.bin" | awk '{print $1}')" == \
            ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896 ]] || {
            echo "frozen Chess artifacts do not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ "$MODE" == "--run-context-repartition" ]]; then
        [[ "$(sha256sum "$repartition_xclbin" | awk '{print $1}')" == \
            837f287e8982d1ec29b61a907ada5b0a6faa0823485cf7a681d010c1a11b057b &&
            "$(sha256sum "$repartition_insts" | awk '{print $1}')" == \
            f6b358372f584f0f0c220ae3dcc83066ae8922d9a15617ca84f3472d4a787941 ]] || {
            echo "context-repartition artifacts do not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ "$MODE" == "--run-async-error" ]]; then
        [[ "$(sha256sum "$ERROR_PDI" | awk '{print $1}')" == \
            58d57cfff96ccfc8ad6b6f8d5c325e548b4f844e2a39b1256aa2294d3a09d77e ]] || {
            echo "signed async-error PDI is missing or does not match the pinned hash: $ERROR_PDI" >&2
            return 1
        }
        [[ -x "$AIE_TRANSLATE" && -x "$BOOTGEN" ]] || {
            echo "mlir-aie CDO tools are missing" >&2
            return 1
        }
        [[ "$(sha256sum "$ASYNC_CONTROL_ELF" | awk '{print $1}')" == \
            52348d78481d99482d56c55bc41d74f3e94f6f77d79508a14f89d9efac9dd75b &&
            "$(sha256sum "$ASYNC_CDO_ROOT/main_aie_cdo_init.bin" | awk '{print $1}')" == \
            bfd4e5fd0a6d7d6c84a44983d241fe192f034f78ce85f332a8203737c6052a02 &&
            "$(sha256sum "$ASYNC_CDO_ROOT/main_aie_cdo_enable.bin" | awk '{print $1}')" == \
            0b3a15b32569661290cc4a7adde899ccb310749584d4e77f0e24112927be4594 &&
            "$(sha256sum "$ERROR_INIT_CDO" | awk '{print $1}')" == \
            bb8e6ab1f30827f692fae587177bbc7e23bd9778368d73ba61dfcbf39ac95a65 ]] || {
            echo "PM-address fault inputs do not match the pinned hashes" >&2
            return 1
        }
        mkdir -p "$ASYNC_ROOT"
        python3 "$ROOT/tools/patch-aie2-pm-address-fault.py" \
            --peano "$PEANO_INSTALL" "$ASYNC_CONTROL_ELF" "$ASYNC_PM_ELF" \
            >"$ASYNC_ROOT/pm-elf.log"
        mkdir -p "$ASYNC_PM_CDO_ROOT"
        cat >"$ASYNC_PM_MLIR" <<EOF
module {
  aie.device(npu1_1col) {
    %tile = aie.tile(0, 2)
    %core = aie.core(%tile) {
      aie.end
    } {elf_file = "$ASYNC_PM_ELF"}
  }
}
EOF
        "$AIE_TRANSLATE" --aie-generate-cdo \
            --work-dir-path="$ASYNC_PM_CDO_ROOT" "$ASYNC_PM_MLIR" \
            >"$ASYNC_ROOT/pm-cdo.log" 2>&1
        [[ "$(sha256sum "$ASYNC_PM_CDO_ROOT/main_aie_cdo_elfs.bin" | awk '{print $1}')" == \
            201a02428f9aed28c1b31ba8ad796c84a554ba8637d25bc61bfe591d79e04b64 ]] || {
            echo "derived PM-address ELF CDO does not match the pinned hash" >&2
            return 1
        }
        cat >"$ASYNC_PM_BIF" <<EOF
all:
{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  image
  {
    name=aie_image, id=0x1c000000
    { type=cdo
      file=$ASYNC_PM_CDO_ROOT/main_aie_cdo_elfs.bin
      file=$ASYNC_CDO_ROOT/main_aie_cdo_init.bin
      file=$ERROR_INIT_CDO
      file=$ASYNC_CDO_ROOT/main_aie_cdo_enable.bin
    }
  }
}
EOF
        "$BOOTGEN" -arch versal -image "$ASYNC_PM_BIF" \
            -o "$ASYNC_PM_PDI" -w >"$ASYNC_ROOT/pm-bootgen.log" 2>&1
        [[ "$(sha256sum "$ASYNC_PM_PDI" | awk '{print $1}')" == \
            b5ffdd10feebf9f3155602299dae406e5a11d3ac5f0314bcdfbb98aba0d67ea5 ]] || {
            echo "derived PM-address signed-error PDI does not match the pinned hash" >&2
            return 1
        }
        mkdir -p "$ASYNC_ROOT/partition"
        xclbinutil --input "$FROZEN_ROOT/chess/aie.xclbin" \
            --dump-section "AIE_PARTITION:JSON:$ASYNC_ROOT/partition/partition.json" \
            --force >"$ASYNC_ROOT/xclbin-dump.log" 2>&1
        mapfile -t async_pdis < <(find "$ASYNC_ROOT/partition" -maxdepth 1 \
            -type f -name '*.pdi' -print)
        [[ "${#async_pdis[@]}" -eq 1 ]] || {
            echo "expected one PDI in the frozen Chess AIE partition" >&2
            return 1
        }
        install -m 0644 "$ERROR_PDI" "${async_pdis[0]}"
        xclbinutil --input "$FROZEN_ROOT/chess/aie.xclbin" \
            --add-replace-section \
            "AIE_PARTITION:JSON:$ASYNC_ROOT/partition/partition.json" \
            --output "$ASYNC_XCLBIN" --force \
            >"$ASYNC_ROOT/xclbin-replace.log" 2>&1
        install -m 0644 "$ASYNC_PM_PDI" "${async_pdis[0]}"
        xclbinutil --input "$FROZEN_ROOT/chess/aie.xclbin" \
            --add-replace-section \
            "AIE_PARTITION:JSON:$ASYNC_ROOT/partition/partition.json" \
            --output "$ASYNC_PM_XCLBIN" --force \
            >"$ASYNC_ROOT/xclbin-replace-pm.log" 2>&1
        install -m 0644 "$FROZEN_ROOT/chess/insts.bin" "$ASYNC_INSTS_PM"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$ASYNC_INSTS_PM" --col 0 --row 2 --tile-type core \
            --insert-event-generate 65 \
            --before-last-tct --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_BATCH_CORE" \
            >"$ASYNC_ROOT/insts-batch-core-event.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$ASYNC_INSTS_BATCH_CORE" --col 0 --row 4 --tile-type memmod \
            --insert-register-write DMA_S2MM_1_Start_Queue 15 \
            --before-last-tct --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_BATCH" \
            >"$ASYNC_ROOT/insts-batch.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 2 --tile-type core \
            --insert-event-generate 70 --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_A" \
            >"$ASYNC_ROOT/insts-a.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 3 --tile-type core \
            --insert-event-generate 70 --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_B" \
            >"$ASYNC_ROOT/insts-b.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 2 --tile-type memmod \
            --insert-register-write DMA_S2MM_1_Start_Queue 15 \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_C" \
            >"$ASYNC_ROOT/insts-c.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 1 --tile-type memtile \
            --insert-register-write DMA_S2MM_2_Start_Queue 24 \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_D" \
            >"$ASYNC_ROOT/insts-d.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 2 --tile-type memmod \
            --insert-register-write DMA_MM2S_1_Start_Queue 15 \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_E" \
            >"$ASYNC_ROOT/insts-e.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$FROZEN_ROOT/chess/insts.bin" --col 0 --row 0 --tile-type shim \
            --insert-register-write DMA_BD14_7 0 --before-last-tct \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_SHIM_CLEAR" \
            >"$ASYNC_ROOT/insts-shim-clear.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$ASYNC_INSTS_SHIM_CLEAR" --col 0 --row 0 --tile-type shim \
            --insert-register-write DMA_S2MM_0_Task_Queue 14 --before-last-tct \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_S2MM" \
            >"$ASYNC_ROOT/insts-s2mm.log"
        python3 "$ROOT/tools/trace-patch-events.py" \
            "$ASYNC_INSTS_SHIM_CLEAR" --col 0 --row 0 --tile-type shim \
            --insert-register-write DMA_MM2S_1_Task_Queue 14 --before-last-tct \
            --register-db "$REGISTER_DB" \
            --output "$ASYNC_INSTS_F" \
            >"$ASYNC_ROOT/insts-f.log"
    fi
    if [[ "$MODE" == "--run-context-repartition" || "$MODE" == "--run-async-error" ]]; then
        nice -n 19 c++ -std=c++17 -O2 -Wall -Wextra -Werror \
            -isystem "$XRT_ROOT/include" "$REPARTITION_SOURCE" \
            -L"$XRT_ROOT/lib" -Wl,-rpath,"$XRT_ROOT/lib" -lxrt_coreutil \
            -o "$REPARTITION_PRODUCER"
    fi
    if $NEEDS_XRT; then
        [[ "$(dpkg-query -W -f='${Version}' xrt-base)" == 2.26.0 &&
            "$(dpkg-query -W -f='${Version}' xrt-npu)" == 2.26.0 &&
            "$(dpkg-query -W -f='${Version}' xrt_plugin-amdxdna)" == 2.26 ]] || {
            echo "installed XRT packages do not match the pinned versions" >&2
            return 1
        }
        [[ -z "$(dpkg --verify xrt-base xrt-npu xrt_plugin-amdxdna)" ]] || {
            echo "installed XRT packages have locally modified files" >&2
            return 1
        }
        [[ "$(sha256sum "$XRT_COREUTIL_VERSIONED" | awk '{print $1}')" == \
            d6a6ea581c95d4c6c09732f7ba2a4d09b4b4a76f5f13657da4a00a9a6e42ca90 &&
            "$(sha256sum "$XRT_CORE_VERSIONED" | awk '{print $1}')" == \
            e4630adc3a2f3066858a2b7d52777ab8233e048096a8bea7f82a1c60731f2e2f &&
            "$(sha256sum "$XRT_XDNA_VERSIONED" | awk '{print $1}')" == \
            c8e5fe57eabb845cc9058631a611f921e6eb9bbf01d083c8c90fd6daa734bb96 ]] || {
            echo "installed XRT runtime does not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ "$MODE" == "--run-npu-direct" ]]; then
        [[ "$(sha256sum "$XRT_PHOENIX_ARCHIVE" | awk '{print $1}')" == \
            0970f2038ee7dcf33dbc704c2ac55271b94687b5a17181cdd2c9118ff195c508 &&
            "$(sha256sum "$XRT_RUNNER" | awk '{print $1}')" == \
            b63ab283f7c72fee5a53245bc5556221ba308a70beaa5ff572de837111063fa6 ]] || {
            echo "installed Phoenix XRT validation producer does not match the pinned hashes" >&2
            return 1
        }
        mkdir -p "$npu_direct_dir"
        (
            cd "$npu_direct_dir"
            ar x "$XRT_PHOENIX_ARCHIVE" \
                recipe_latency.json profile_latency.json validate.xclbin nop.elf
        )
        [[ "$(sha256sum "$npu_direct_dir/recipe_latency.json" | awk '{print $1}')" == \
            ca8b824cec50a8e41fda8c873978f363d6c7f52f728edb603bbc001ee96d8fba &&
            "$(sha256sum "$npu_direct_dir/profile_latency.json" | awk '{print $1}')" == \
            b44e2a96c10370461afe34f920d3c7ab0900cb8d08bd7064f42dc2a9769d3639 &&
            "$(sha256sum "$npu_direct_dir/validate.xclbin" | awk '{print $1}')" == \
            64e41d6bf7ce9668fc75bbbe699df9612056349ff81f17f0df524c6b5016ebf4 &&
            "$(sha256sum "$npu_direct_dir/nop.elf" | awk '{print $1}')" == \
            00338b532eeeea01611a36c916c07963b8085c34a6910e0ea80582bfa76fe00e ]] || {
            echo "extracted Phoenix XRT validation artifacts do not match the pinned hashes" >&2
            return 1
        }
    fi
    grep -Fqx "CONFIG_MODULE_SIG_FORCE=y" \
        "/boot/config-$GUEST_KERNEL_VERSION" && {
        echo "guest kernel would reject the unsigned pinned module" >&2
        return 1
    }

    driver_repo="$NPU_WORK/xdna-driver"
    mkdir -p "$DRIVER_SOURCE"
    git -C "$driver_repo" archive --format=tar --output="$driver_archive" \
        "$DRIVER_PIN" drivers/accel/amdxdna \
        drivers/accel/tools/configure_kernel.sh include
    tar -xf "$driver_archive" -C "$DRIVER_SOURCE"
    if [[ -n "$GATE_ARM" ]]; then
        git -C / apply --directory="${DRIVER_SOURCE#/}" "$NPI_READ_PATCH"
        git -C / apply --directory="${DRIVER_SOURCE#/}" "$PROTECTED_GATE_PATCH"
    elif [[ "$MODE" == "--probe-phoenix-npi-read" ]]; then
        git -C / apply --directory="${DRIVER_SOURCE#/}" "$NPI_READ_PATCH"
    fi
    KERNEL_VER="$GUEST_KERNEL_VERSION" \
        KERNEL_SRC="/usr/src/linux-headers-$GUEST_KERNEL_VERSION" \
        OUT="$DRIVER_SOURCE/drivers/accel/amdxdna/config_kernel.h" \
        sh "$DRIVER_SOURCE/drivers/accel/tools/configure_kernel.sh" \
        >"$driver_build_log" 2>&1
    nice -n 19 make -C "$DRIVER_SOURCE/drivers/accel/amdxdna" \
        KERNEL_VER="$GUEST_KERNEL_VERSION" \
        KERNEL_SRC="/usr/src/linux-headers-$GUEST_KERNEL_VERSION" \
        XDNA_HASH="$DRIVER_PIN" XDNA_DATE=20260728 LLVM=1 modules \
        >>"$driver_build_log" 2>&1
    if [[ -n "$GATE_ARM" || "$MODE" == "--probe-phoenix-npi-read" ]]; then
        module_signing_pem="$(sed -n \
            's/^CONFIG_MODULE_SIG_KEY="\(.*\)"$/\1/p' \
            "/boot/config-$GUEST_KERNEL_VERSION")"
        [[ -r "$module_signing_pem" &&
            -x "/usr/src/linux-headers-$GUEST_KERNEL_VERSION/scripts/sign-file" ]] || {
            echo "host module-signing key or sign-file is unavailable" >&2
            return 1
        }
        "/usr/src/linux-headers-$GUEST_KERNEL_VERSION/scripts/sign-file" \
            sha512 "$module_signing_pem" "$module_signing_pem" "$driver_module"
        for signature_field in signer sig_id sig_key sig_hashalgo; do
            [[ -n "$(modinfo -F "$signature_field" "$driver_module")" &&
                "$(modinfo -F "$signature_field" "$driver_module")" == \
                "$(modinfo -F "$signature_field" amdxdna)" ]] || {
                echo "pinned driver module signature does not match installed amdxdna: $signature_field" >&2
                return 1
            }
        done
        [[ "$(modinfo -F sig_id "$driver_module")" == PKCS#7 &&
            "$(modinfo -F sig_hashalgo "$driver_module")" == sha512 ]] || {
            echo "pinned driver module does not have the required PKCS#7/SHA-512 signature" >&2
            return 1
        }
    fi
    module_vermagic="$(modinfo -F vermagic "$driver_module")"
    [[ "$module_vermagic" == "$GUEST_KERNEL_VERSION "* ]] || {
        echo "pinned driver module has the wrong vermagic" >&2
        return 1
    }
    modinfo -F firmware "$driver_module" |
        grep -Fqx "amdnpu/1502_00/npu.dev.sbin" || {
        echo "pinned primary module does not request npu.dev.sbin" >&2
        return 1
    }

    mkdir -p "$GUEST_ROOT"/{bin,dev,etc,lib/firmware/amdnpu/1502_00,proc,run,sbin,sys,tmp,usr/bin}
    install -m 0755 /usr/bin/busybox "$GUEST_ROOT/bin/busybox"
    install -m 0755 \
        "$ROOT/tools/phoenix-vfio-user/guest-driver-probe-init.sh" \
        "$GUEST_ROOT/init"
    for applet in cat dmesg find grep mount poweroff sh sleep sync timeout uname wc; do
        ln -s busybox "$GUEST_ROOT/bin/$applet"
    done
    ln -s ../bin/busybox "$GUEST_ROOT/sbin/modprobe"
    install -m 0755 /usr/bin/lspci "$GUEST_ROOT/usr/bin/lspci"
    {
        ldd /usr/bin/lspci
        if $NEEDS_XRT; then
            if [[ -n "$FROZEN_COMPILER" ]]; then
                ldd "$FROZEN_TEST"
            elif [[ -n "$ELF_COMPILER" ]]; then
                ldd "$ELF_TEST"
            elif [[ -n "$GATE_ARM" ]]; then
                ldd "$GATE_RUNNER" "$GATE_CLASSIFIER" "$GATE_CLOCK_QUERY"
            elif [[ "$MODE" == "--run-context-repartition" || "$MODE" == "--run-async-error" ]]; then
                ldd "$REPARTITION_PRODUCER"
            else
                ldd "$XRT_RUNNER"
            fi
            ldd "$XRT_COREUTIL_VERSIONED" \
                "$XRT_CORE_VERSIONED" "$XRT_XDNA_VERSIONED"
            printf '  %s\n' "$XRT_COREUTIL" "$XRT_CORE" "$XRT_XDNA"
        fi
    } |
        awk '/=> \// {print $3} /^[[:space:]]+\// {print $1}' |
        sort -u >"$RUN_DIR/guest-libraries.txt"
    [[ -s "$RUN_DIR/guest-libraries.txt" ]] || {
        echo "failed to derive guest runtime libraries" >&2
        return 1
    }
    while IFS= read -r library; do
        copy_host_file "$library"
    done <"$RUN_DIR/guest-libraries.txt"
    [[ -x "$GUEST_ROOT/lib64/ld-linux-x86-64.so.2" ]] || {
        echo "guest image is missing lspci's dynamic loader" >&2
        return 1
    }
    if [[ -n "$FROZEN_COMPILER" ]]; then
        mkdir -p "$GUEST_ROOT/run-frozen"
        install -m 0755 "$FROZEN_TEST" "$GUEST_ROOT/run-frozen/test.exe"
        install -m 0644 "$frozen_xclbin" "$GUEST_ROOT/run-frozen/aie.xclbin"
        install -m 0644 "$frozen_insts" "$GUEST_ROOT/run-frozen/insts.bin"
        printf '%s\n' "$FROZEN_COMPILER" >"$GUEST_ROOT/run-frozen/compiler"
        printf '%s\n' "$FROZEN_EXECUTION" >"$GUEST_ROOT/run-frozen/execution-mode"
    fi
    if [[ -n "$ELF_COMPILER" ]]; then
        mkdir -p "$GUEST_ROOT/run-elf"
        install -m 0755 "$ELF_TEST" "$GUEST_ROOT/run-elf/test.exe"
        install -m 0644 "$elf_xclbin" "$GUEST_ROOT/run-elf/aie.xclbin"
        install -m 0644 "$elf_insts" "$GUEST_ROOT/run-elf/insts.elf"
        printf '%s\n' "$ELF_COMPILER" >"$GUEST_ROOT/run-elf/compiler"
    fi
    if [[ -n "$GATE_ARM" ]]; then
        mkdir -p "$GUEST_ROOT/run-real-column-gate"
        install -m 0755 "$GATE_RUNNER" \
            "$GUEST_ROOT/run-real-column-gate/bridge-trace-runner"
        install -m 0755 "$GATE_CLASSIFIER" \
            "$GUEST_ROOT/run-real-column-gate/libxdna_emu.so"
        install -m 0755 "$GATE_CLOCK_QUERY" \
            "$GUEST_ROOT/run-real-column-gate/xdna-clock-query"
        install -m 0644 "$GATE_XCLBIN" \
            "$GUEST_ROOT/run-real-column-gate/aie.xclbin"
        install -m 0644 "$GATE_INSTS" \
            "$GUEST_ROOT/run-real-column-gate/arm.insts.bin"
        install -m 0644 "$GATE_CANARY_INSTS" \
            "$GUEST_ROOT/run-real-column-gate/canary.insts.bin"
        printf '%s\n' "$GATE_ARM" >"$GUEST_ROOT/run-real-column-gate/arm"
    fi
    if [[ "$MODE" == "--probe-phoenix-npi-read" ]]; then
        mkdir -p "$GUEST_ROOT/run-phoenix-npi-read"
    fi
    if [[ "$MODE" == "--run-npu-direct" ]]; then
        mkdir -p "$GUEST_ROOT/run-npu"
        copy_host_file "$XRT_RUNNER"
        install -m 0644 "$npu_direct_dir/recipe_latency.json" \
            "$GUEST_ROOT/run-npu/recipe_latency.json"
        install -m 0644 "$npu_direct_dir/profile_latency.json" \
            "$GUEST_ROOT/run-npu/profile_latency.json"
        install -m 0644 "$npu_direct_dir/validate.xclbin" \
            "$GUEST_ROOT/run-npu/validate.xclbin"
        install -m 0644 "$npu_direct_dir/nop.elf" \
            "$GUEST_ROOT/run-npu/nop.elf"
    fi
    if [[ "$MODE" == "--run-context-repartition" ]]; then
        mkdir -p "$GUEST_ROOT/run-repartition"
        install -m 0755 "$REPARTITION_PRODUCER" \
            "$GUEST_ROOT/run-repartition/context-repartition"
        install -m 0644 "$FROZEN_ROOT/chess/aie.xclbin" \
            "$GUEST_ROOT/run-repartition/A.xclbin"
        install -m 0644 "$FROZEN_ROOT/chess/insts.bin" \
            "$GUEST_ROOT/run-repartition/A.insts"
        install -m 0644 "$repartition_xclbin" \
            "$GUEST_ROOT/run-repartition/B.xclbin"
        install -m 0644 "$repartition_insts" \
            "$GUEST_ROOT/run-repartition/B.insts"
    fi
    if [[ "$MODE" == "--run-async-error" ]]; then
        mkdir -p "$GUEST_ROOT/run-async-error"
        install -m 0755 "$REPARTITION_PRODUCER" \
            "$GUEST_ROOT/run-async-error/async-error-probe"
        install -m 0644 "$ASYNC_XCLBIN" \
            "$GUEST_ROOT/run-async-error/aie.xclbin"
        install -m 0644 "$ASYNC_PM_XCLBIN" \
            "$GUEST_ROOT/run-async-error/PM.xclbin"
        install -m 0644 "$ASYNC_INSTS_PM" \
            "$GUEST_ROOT/run-async-error/PM.insts"
        install -m 0644 "$ASYNC_INSTS_BATCH" \
            "$GUEST_ROOT/run-async-error/BATCH.insts"
        install -m 0644 "$ASYNC_INSTS_A" \
            "$GUEST_ROOT/run-async-error/A.insts"
        install -m 0644 "$ASYNC_INSTS_B" \
            "$GUEST_ROOT/run-async-error/B.insts"
        install -m 0644 "$ASYNC_INSTS_C" \
            "$GUEST_ROOT/run-async-error/C.insts"
        install -m 0644 "$ASYNC_INSTS_D" \
            "$GUEST_ROOT/run-async-error/D.insts"
        install -m 0644 "$ASYNC_INSTS_E" \
            "$GUEST_ROOT/run-async-error/E.insts"
        install -m 0644 "$ASYNC_INSTS_S2MM" \
            "$GUEST_ROOT/run-async-error/S2MM.insts"
        install -m 0644 "$ASYNC_INSTS_F" \
            "$GUEST_ROOT/run-async-error/F.insts"
        if $ASYNC_BATCH_ONLY; then
            : >"$GUEST_ROOT/run-async-error/batch-only"
        fi
    fi
    if [[ -f /usr/share/misc/pci.ids ]]; then
        copy_host_file /usr/share/misc/pci.ids
    elif [[ -f /usr/share/hwdata/pci.ids ]]; then
        copy_host_file /usr/share/hwdata/pci.ids
    fi

    {
        modprobe --show-depends 8250
        while IFS= read -r dependency; do
            [[ -z "$dependency" ]] ||
                modprobe --show-depends "$dependency"
        done < <(modinfo -F depends "$driver_module" | tr ',' '\n')
    } | awk '$1 == "insmod" {print $2}' | sort -u \
        >"$RUN_DIR/module-paths.txt"
    while IFS= read -r module_path; do
        copy_host_file "$module_path"
    done <"$RUN_DIR/module-paths.txt"
    install -D -m 0644 "$driver_module" \
        "$GUEST_ROOT/lib/modules/$GUEST_KERNEL_VERSION/extra/amdxdna.ko"
    for module_path in modules.builtin modules.builtin.modinfo modules.order; do
        copy_host_file "/lib/modules/$GUEST_KERNEL_VERSION/$module_path"
    done
    depmod -b "$GUEST_ROOT" "$GUEST_KERNEL_VERSION" \
        >"$RUN_DIR/depmod.log" 2>&1
    install -m 0644 "$FIRMWARE" \
        "$GUEST_ROOT/lib/firmware/amdnpu/1502_00/npu.dev.sbin"

    (
        cd "$GUEST_ROOT"
        find . -print0 | sort -z |
            cpio --null -o --format=newc >"$initramfs_cpio"
    ) 2>>"$RUN_DIR/guest-build.log"
    gzip -n -9 -c "$initramfs_cpio" >"$INITRAMFS"

    {
        echo "driver_commit=$DRIVER_PIN"
        echo "driver_tree=drivers/accel/amdxdna"
        echo "xdna_emu_commit=$(git -C "$ROOT" rev-parse HEAD)"
        echo "guest_kernel_version=$GUEST_KERNEL_VERSION"
        echo "qemu_package=$(dpkg-query -W -f='${Version}' qemu-system-x86)"
        echo "libvfio_user_commit=$(git -C "$ROOT/build/deps/libvfio-user" rev-parse HEAD)"
        echo "mlir_aie_commit=$(git -C "$MLIR_AIE_PATH" rev-parse HEAD)"
        echo "force_iova=default-false"
        echo "viommu=absent"
        sha256sum "$REGISTER_DB" "$FIRMWARE" "$GUEST_KERNEL" "$driver_archive" \
            "$driver_module" "$INITRAMFS"
        if [[ -n "$FROZEN_COMPILER" ]]; then
            echo "frozen_compiler=$FROZEN_COMPILER"
            echo "frozen_execution=$FROZEN_EXECUTION"
            sha256sum "$FROZEN_TEST" "$frozen_xclbin" "$frozen_insts"
        fi
        if [[ -n "$ELF_COMPILER" ]]; then
            echo "elf_compiler=$ELF_COMPILER"
            echo "xrt_execution=direct-exec-dpu-data-plane"
            sha256sum "$ELF_TEST" "$elf_xclbin" "$elf_insts"
        fi
        if [[ -n "$GATE_ARM" ]]; then
            echo "real_column_gate_arm=$GATE_ARM"
            echo "xrt_execution=signed-firmware-chain-exec-npu"
            echo "research_probe=phoenix-protected-column-gate"
            echo "expected_live_placement=1:1"
            echo "bridge_runner_async_context=0"
            echo "bridge_runner_reuse_context=0"
            sha256sum "$NPI_READ_PATCH" "$PROTECTED_GATE_PATCH" \
                "$GATE_INSTS" "$GATE_XCLBIN" "$GATE_MLIR" \
                "$GATE_EXPECTED_OUTPUT" "$GATE_RUNNER" \
                "$GATE_CLASSIFIER" \
                "$ROOT/bridge-runner/bridge-trace-runner.cpp" \
                "$GATE_CLOCK_QUERY" "$GATE_CLOCK_QUERY_SOURCE"
        fi
        if [[ "$MODE" == "--probe-phoenix-npi-read" ]]; then
            echo "research_probe=phoenix-read-only-npi-lock"
            echo "expected_management_request=opcode:0x203,size:24,type:2,row:0,col:0,offset:0x1000000c"
            echo "expected_system_read=0xac00000c"
            sha256sum "$NPI_READ_PATCH"
        fi
        if [[ "$MODE" == "--run-context-repartition" ]]; then
            echo "xrt_execution=context-repartition-cmdlist"
            sha256sum "$REPARTITION_SOURCE" "$REPARTITION_PRODUCER" \
                "$FROZEN_ROOT/chess/aie.xclbin" \
                "$FROZEN_ROOT/chess/insts.bin" \
                "$repartition_xclbin" "$repartition_insts"
        fi
        if [[ "$MODE" == "--run-async-error" ]]; then
            echo "xrt_execution=signed-firmware-async-error-cmdlist"
            if $ASYNC_BATCH_ONLY; then
                echo "async_error_scope=batch-only"
            else
                echo "async_error_scope=lifecycle"
            fi
            echo "mlir_aie_build=$MLIR_AIE_BUILD"
            sha256sum "$REPARTITION_SOURCE" "$REPARTITION_PRODUCER" \
                "$ERROR_PDI" "$ERROR_INIT_CDO" "$ASYNC_CONTROL_ELF" \
                "$ROOT/tools/patch-aie2-pm-address-fault.py" \
                "$ASYNC_PM_ELF" "$ASYNC_PM_CDO_ROOT/main_aie_cdo_elfs.bin" \
                "$ASYNC_PM_PDI" "$ASYNC_PM_XCLBIN" "$ASYNC_INSTS_PM" \
                "$ASYNC_INSTS_BATCH_CORE" "$ASYNC_INSTS_BATCH" \
                "$ASYNC_XCLBIN" "$ASYNC_INSTS_A" \
                "$ASYNC_INSTS_B" "$ASYNC_INSTS_C" "$ASYNC_INSTS_D" \
                "$ASYNC_INSTS_E" "$ASYNC_INSTS_S2MM" "$ASYNC_INSTS_F"
        fi
        if $NEEDS_XRT; then
            dpkg-query -W -f='${Package}=${Version}\n' \
                xrt-base xrt-npu xrt_plugin-amdxdna
            sha256sum "$XRT_COREUTIL_VERSIONED" "$XRT_CORE_VERSIONED" \
                "$XRT_XDNA_VERSIONED"
        fi
        if [[ "$MODE" == "--run-npu-direct" ]]; then
            echo "xrt_execution=direct-exec-dpu"
            sha256sum "$XRT_RUNNER" "$XRT_PHOENIX_ARCHIVE" \
                "$npu_direct_dir/recipe_latency.json" \
                "$npu_direct_dir/profile_latency.json" \
                "$npu_direct_dir/validate.xclbin" "$npu_direct_dir/nop.elf"
        fi
        modinfo "$driver_module"
    } >"$RUN_DIR/tuple.txt"
}

if [[ "$MODE" != "--map-smoke" ]]; then
    prepare_driver_guest
    server_args=("$VFIO_SOCKET")
    export RUST_LOG="${RUST_LOG:-xdna_emu::firmware::mmio=debug,warn}"
else
    server_args=(--map-smoke "$VFIO_SOCKET")
fi

"$SERVER" "${server_args[@]}" \
    >"$RUN_DIR/server.log" 2>&1 &
server_pid=$!
wait_for_socket "$VFIO_SOCKET" "$server_pid" || {
    echo "vfio-user server did not open its socket; see $RUN_DIR/server.log" >&2
    exit 1
}

readonly VFIO_DEVICE="{\"driver\":\"vfio-user-pci\",\"socket\":{\"path\":\"$VFIO_SOCKET\",\"type\":\"unix\"}}"
if [[ "$MODE" != "--map-smoke" ]]; then
    guest_qemu=(
        qemu-system-x86_64
        -no-user-config
        -nodefaults
        -machine "q35,accel=kvm,memory-backend=ram"
        -cpu "host,hypervisor=off"
        -smp 4
        -m 2G
        -object "memory-backend-memfd,id=ram,size=2G,share=on"
        -kernel "$GUEST_KERNEL"
        -initrd "$INITRAMFS"
        -append "console=ttyS0,115200n8 rdinit=/init panic=-1 memmap=256M\$1536M"
        -chardev "file,id=serial0,path=$GUEST_LOG"
        -device "isa-serial,chardev=serial0"
        -display none
        -no-reboot
        -device "$VFIO_DEVICE"
    )
    printf '%q ' "${guest_qemu[@]}" >"$RUN_DIR/qemu-command.txt"
    printf '\n' >>"$RUN_DIR/qemu-command.txt"

    : >"$GUEST_LOG"
    "${guest_qemu[@]}" >"$RUN_DIR/qemu.log" 2>&1 &
    qemu_pid=$!
    guest_result=0
    wait_for_guest_result "$qemu_pid" "$server_pid" || guest_result=$?
    if [[ "$guest_result" -ne 0 ]]; then
        echo "guest driver probe failed with result $guest_result; evidence: $RUN_DIR" >&2
        exit 1
    fi
    wait_for_exit "$qemu_pid" || {
        echo "QEMU did not exit after guest poweroff; evidence: $RUN_DIR" >&2
        exit 1
    }
    if ! wait "$qemu_pid"; then
        echo "QEMU exited unsuccessfully; evidence: $RUN_DIR" >&2
        exit 1
    fi
    qemu_pid=

    kill "$server_pid"
    wait_for_exit "$server_pid" || {
        echo "vfio-user server did not stop; evidence: $RUN_DIR" >&2
        exit 1
    }
    if ! wait "$server_pid"; then
        echo "vfio-user server exited unsuccessfully; evidence: $RUN_DIR" >&2
        exit 1
    fi
    server_pid=

    sed -i 's/\r$//' "$GUEST_LOG"
    sed -n '/PHOENIX_LSPCI_BEGIN/,/PHOENIX_LSPCI_END/p' "$GUEST_LOG" \
        >"$RUN_DIR/lspci.log"
    sed -n '/PHOENIX_MSIX_BEGIN/,/PHOENIX_MSIX_END/p' "$GUEST_LOG" \
        >"$RUN_DIR/msix.log"
    sed -n '/PHOENIX_DMESG_BEGIN/,/PHOENIX_DMESG_END/p' "$GUEST_LOG" \
        >"$RUN_DIR/dmesg.log"
    grep -Fqx "accel_node=/dev/accel/accel0" "$GUEST_LOG"
    grep -Fqx "probe=complete" "$GUEST_LOG"
    grep -Fqx "force_iova=N" "$GUEST_LOG"
    grep -Fqx "carveout=0x10000000@0x60000000" "$GUEST_LOG"
    grep -Fqx "count=16" "$GUEST_LOG"
    grep -Fq "MSI-X: Enable+ Count=16" "$RUN_DIR/lspci.log"
    grep -Fq "Kernel driver in use: amdxdna" "$RUN_DIR/lspci.log"
    grep -Fq "Load firmware amdnpu/1502_00/npu.dev.sbin" \
        "$RUN_DIR/dmesg.log"

    if $NEEDS_XRT; then
        grep -Eq \
            'firmware mailbox X2I tail 0x030da000=.*source 37 asserted=true' \
            "$RUN_DIR/server.log" || {
            echo "context channel-5 X2I publication was not observed; evidence: $RUN_DIR" >&2
            exit 1
        }
        grep -Fq "firmware service msix=0x20 " "$RUN_DIR/server.log" || {
            echo "context channel-5 MSI-X completion was not observed; evidence: $RUN_DIR" >&2
            exit 1
        }
    fi

    if [[ "$MODE" == "--probe-phoenix-npi-read" ]]; then
        grep -Fqx "PHOENIX_NPI_READ_BEGIN" "$GUEST_LOG"
        grep -Fqx "PHOENIX_NPI_READ value=0x00000000" "$GUEST_LOG"
        grep -Fqx "PHOENIX_NPI_READ_PASS" "$GUEST_LOG"
        grep -Fqx "PHOENIX_DRIVER_PROBE_PASS" "$GUEST_LOG"
        "$ROOT/scripts/phoenix-real-column-gate-host.py" \
            _validate_npi_lifecycle "$RUN_DIR/dmesg.log" || {
            echo "Phoenix NPI probe mailbox lifecycle differed; evidence: $RUN_DIR" >&2
            exit 1
        }
        echo "phoenix vfio-user read-only management NPI probe: PASS"
    elif [[ -n "$GATE_ARM" ]]; then
        classify_real_column_gate_run
    elif [[ "$MODE" == "--run-async-error" ]]; then
        if $ASYNC_BATCH_ONLY; then
            grep -Fqx "PHOENIX_ASYNC_ERROR_BATCH_BEGIN" "$GUEST_LOG"
            grep -Eq '^PHOENIX_ASYNC_ERROR_ONE err_code=0x2040304000b ts_us=[1-9][0-9]* ex_err_code=0x401$' \
                "$GUEST_LOG"
            grep -Fqx "PHOENIX_ASYNC_ERROR_BATCH_PASS" "$GUEST_LOG"
            grep -Fqx "force_cmdlist=Y" "$GUEST_LOG"
            grep -Fq "Row: 2, Col: 1, module 1, event ID 65, category 3" \
                "$RUN_DIR/dmesg.log"
            grep -Fq "Row: 4, Col: 1, module 0, event ID 98, category 8" \
                "$RUN_DIR/dmesg.log"
            awk '
                /AIE error: 00000000: 00000002 00000000 [[:xdigit:]]+ 00000102$/ {
                    headers++
                    if ((getline line1) > 0 &&
                        line1 ~ /AIE error: 00000010: 00000001 00000041 00000104 00000000$/ &&
                        (getline line2) > 0 &&
                        line2 ~ /AIE error: 00000020: 00000062 /)
                        matches++
                }
                END { exit !(headers == 1 && matches == 1) }
            ' "$RUN_DIR/dmesg.log"
            echo "phoenix vfio-user signed-firmware async-error batch: PASS"
        else
        grep -Fqx "PHOENIX_ASYNC_ERROR_PM_BEGIN" "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_ONE_STATE state=[0-9]+$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_ONE err_code=0x20303040006 ts_us=[1-9][0-9]* ex_err_code=0x201$' \
            "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_PM_PASS" "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_S2MM_BEGIN" "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_ONE_NONCOMPLETION state=[0-9]+$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_ONE err_code=0x2070304000b ts_us=[1-9][0-9]* ex_err_code=0x1$' \
            "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_ONE_PASS" "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_S2MM_PASS" "$GUEST_LOG"
        for marker in A B C D E F; do
            grep -Fqx "PHOENIX_ASYNC_ERROR_${marker}_PASS" "$GUEST_LOG"
        done
        grep -Eq '^PHOENIX_ASYNC_ERROR_FIRST err_code=0x20303040008 ts_us=[1-9][0-9]* ex_err_code=0x201$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_SECOND err_code=0x20303040008 ts_us=[1-9][0-9]* ex_err_code=0x301$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_THIRD err_code=0x2040304000b ts_us=[1-9][0-9]* ex_err_code=0x201$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_FOURTH err_code=0x2040304000b ts_us=[1-9][0-9]* ex_err_code=0x101$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_FIFTH err_code=0x2040304000b ts_us=[1-9][0-9]* ex_err_code=0x201$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_SIXTH err_code=0x2070304000b ts_us=[1-9][0-9]* ex_err_code=0x1$' \
            "$GUEST_LOG"
        grep -Eq '^PHOENIX_ASYNC_ERROR_F_NONCOMPLETION state=[0-9]+$' \
            "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_PASS" "$GUEST_LOG"
        grep -Fqx "PHOENIX_ASYNC_ERROR_GUEST_PASS" "$GUEST_LOG"
        grep -Fqx "force_cmdlist=Y" "$GUEST_LOG"
        grep -Fq "Row: 2, Col: 1, module 1, event ID 70, category 5" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 2, Col: 1, module 1, event ID 65, category 3" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 3, Col: 1, module 1, event ID 70, category 5" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 2, Col: 1, module 0, event ID 98, category 8" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 1, Col: 1, module 0, event ID 133, category 8" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 2, Col: 1, module 0, event ID 100, category 8" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 0, Col: 1, module 2, event ID 72, category 8" \
            "$RUN_DIR/dmesg.log"
        grep -Fq "Row: 0, Col: 1, module 2, event ID 73, category 8" \
            "$RUN_DIR/dmesg.log"
        echo "phoenix vfio-user signed-firmware async-error lifecycle: PASS"
        fi
    elif [[ "$MODE" == "--run-context-repartition" ]]; then
        grep -Fqx "PHOENIX_CONTEXT_REPARTITION_PASS" "$GUEST_LOG"
        for marker in A1 B A2; do
            grep -Fqx "PHOENIX_REPARTITION_${marker}_PASS" "$GUEST_LOG"
        done
        grep -Fqx "PHOENIX_REPARTITION_A_DESTROYED" "$GUEST_LOG"
        grep -Fqx "PHOENIX_REPARTITION_B_DESTROYED" "$GUEST_LOG"
        grep -Fqx "PHOENIX_REPARTITION_PASS" "$GUEST_LOG"
        grep -Fqx "force_cmdlist=Y" "$GUEST_LOG"
        awk '
            /xdna_mailbox\.[0-9]+: opcode 0x2 size 28 id / {
                awaiting_create_payload = 1
                create_requests++
                next
            }
            awaiting_create_payload && /req data: 00000010:/ {
                line = $0
                sub(/^.*req data: 00000010:[[:space:]]*/, "", line)
                split(line, words, /[[:space:]]+/)
                widths[++width_count] = words[2]
                awaiting_create_payload = 0
            }
            /xdna_mailbox\.[0-9]+: opcode 0x2 size 76 id / { create_responses++ }
            /xdna_mailbox\.[0-9]+: opcode 0x106 size 20 id / { map_requests++ }
            /xdna_mailbox\.[0-9]+: opcode 0x106 size 4 id / { map_responses++ }
            /xdna_mailbox\.[0-9]+: opcode 0x11 size 132 id / { config_requests++ }
            /xdna_mailbox\.[0-9]+: opcode 0x11 size 4 id / { config_responses++ }
            /xdna_mailbox\.[0-9]+: opcode 0x18 size 24 id / { execute_requests++ }
            /xdna_mailbox\.[0-9]+: opcode 0x18 size 12 id / { execute_responses++ }
            /xdna_mailbox\.[0-9]+: opcode 0x3 size 4 id / { destroy_messages++ }
            END {
                exit create_requests != 3 || create_responses != 3 ||
                    map_requests != 3 || map_responses != 3 ||
                    config_requests != 3 || config_responses != 3 ||
                    execute_requests != 3 || execute_responses != 3 ||
                    destroy_messages != 6 || width_count != 3 ||
                    widths[1] != "00000101" || widths[2] != "00000201" ||
                    widths[3] != "00010203"
            }
        ' "$RUN_DIR/dmesg.log" || {
            echo "context repartition mailbox lifecycle differed; evidence: $RUN_DIR" >&2
            exit 1
        }
        if grep -Fq "status 0x2000006" "$RUN_DIR/dmesg.log"; then
            echo "context repartition hit MGMT_ERT_BUSY; evidence: $RUN_DIR" >&2
            exit 1
        fi
        echo "phoenix vfio-user context repartition/reconnect: PASS"
    elif [[ -n "$FROZEN_COMPILER" ]]; then
        grep -Fqx "PHOENIX_FROZEN_PASS $FROZEN_COMPILER" "$GUEST_LOG"
        grep -Fqx "PASS!" "$GUEST_LOG"
        awk '
            /^Correct output / {
                expected = count + 2
                if ($0 != "Correct output " expected " == " expected)
                    bad = 1
                else
                    count++
            }
            END { exit bad || count != 64 }
        ' "$GUEST_LOG" || {
            echo "frozen output was not the ordered range 2..65; evidence: $RUN_DIR" >&2
            exit 1
        }
        if [[ "$FROZEN_EXECUTION" == direct ]]; then
            grep -Fqx "force_cmdlist=N" "$GUEST_LOG"
            grep -Eq 'xdna_mailbox\.[0-9]+: opcode 0xc size 80 id ' "$RUN_DIR/dmesg.log" || {
                echo "direct EXECUTE_BUFFER_CF request was not observed; evidence: $RUN_DIR" >&2
                exit 1
            }
            grep -Eq 'xdna_mailbox\.[0-9]+: opcode 0xc size 4 id ' "$RUN_DIR/dmesg.log" || {
                echo "direct EXECUTE_BUFFER_CF response was not observed; evidence: $RUN_DIR" >&2
                exit 1
            }
            if grep -Eq 'xdna_mailbox\.[0-9]+: opcode 0x18 size 24 id ' "$RUN_DIR/dmesg.log"; then
                echo "direct run used CHAIN_EXEC_NPU; evidence: $RUN_DIR" >&2
                exit 1
            fi
            destroy_count="$(awk '/xdna_mailbox\.[0-9]+: opcode 0x3 size 4 id / { count++ } END { print count + 0 }' "$RUN_DIR/dmesg.log")"
            [[ "$destroy_count" -eq 2 ]] || {
                echo "DESTROY_CONTEXT request/response pair was not observed; evidence: $RUN_DIR" >&2
                exit 1
            }
            echo "phoenix vfio-user frozen direct $FROZEN_COMPILER kernel: PASS"
        else
            grep -Fqx "force_cmdlist=Y" "$GUEST_LOG"
            echo "phoenix vfio-user frozen $FROZEN_COMPILER kernel: PASS"
        fi
    elif [[ -n "$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ]]; then
        if [[ -n "$ELF_COMPILER" ]]; then
            grep -Fqx "PHOENIX_PINNED_ELF_PASS $ELF_COMPILER" "$GUEST_LOG"
            grep -Fqx "PASS!" "$GUEST_LOG"
            awk '
                /^Correct output / {
                    expected = count + 42
                    if ($0 != "Correct output " expected " == " expected)
                        bad = 1
                    else
                        count++
                }
                END { exit bad || count != 64 }
            ' "$GUEST_LOG" || {
                echo "pinned ELF output was not the ordered range 42..105; evidence: $RUN_DIR" >&2
                exit 1
            }
        else
            grep -Fqx "PHOENIX_EXEC_DPU_PASS" "$GUEST_LOG"
        fi
        grep -Fqx "force_cmdlist=N" "$GUEST_LOG"
        awk '
            /xdna_mailbox\.[0-9]+: opcode 0x10 size 160 id / {
                request_count++
                request_id = $NF
            }
            /xdna_mailbox\.[0-9]+: opcode 0x10 size 4 id / {
                response_count++
                response_id = $NF
            }
            END {
                exit request_count != 1 || response_count != 1 ||
                    request_id != response_id
            }
        ' "$RUN_DIR/dmesg.log" || {
            echo "matched EXEC_DPU request/response pair was not observed; evidence: $RUN_DIR" >&2
            exit 1
        }
        if grep -Eq 'xdna_mailbox\.[0-9]+: opcode 0x18 size 24 id ' "$RUN_DIR/dmesg.log"; then
            echo "direct DPU run used CHAIN_EXEC_NPU; evidence: $RUN_DIR" >&2
            exit 1
        fi
        if [[ -n "$ELF_COMPILER" ]] &&
            grep -Eq 'xdna_mailbox\.[0-9]+: opcode 0xc size 80 id ' "$RUN_DIR/dmesg.log"; then
            echo "pinned ELF run used EXECUTE_BUFFER_CF; evidence: $RUN_DIR" >&2
            exit 1
        fi
        destroy_count="$(awk '/xdna_mailbox\.[0-9]+: opcode 0x3 size 4 id / { count++ } END { print count + 0 }' "$RUN_DIR/dmesg.log")"
        [[ "$destroy_count" -eq 2 ]] || {
            echo "DESTROY_CONTEXT request/response pair was not observed; evidence: $RUN_DIR" >&2
            exit 1
        }
        if [[ -n "$ELF_COMPILER" ]]; then
            echo "phoenix vfio-user pinned ELF $ELF_COMPILER data plane: PASS"
        else
            echo "phoenix vfio-user direct EXEC_DPU no-op: PASS"
        fi
    else
        echo "phoenix vfio-user pinned driver probe: PASS"
    fi
    echo "evidence: $RUN_DIR"
    exit 0
fi

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
