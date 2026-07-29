#!/usr/bin/env bash
set -euo pipefail

case "${1:-}:$#" in
    --map-smoke:1 | --driver-probe:1 | --run-npu-direct:1) ;;
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
        echo "usage: $0 --map-smoke | --driver-probe | --run-frozen chess|peano | --run-frozen-direct chess|peano | --run-npu-direct | --run-pinned-elf chess|peano" >&2
        exit 2
        ;;
esac

readonly MODE=$1
FROZEN_COMPILER=
ELF_COMPILER=
case "$MODE" in
    --run-frozen | --run-frozen-direct) FROZEN_COMPILER=$2 ;;
    --run-pinned-elf) ELF_COMPILER=$2 ;;
esac
readonly FROZEN_COMPILER ELF_COMPILER
FROZEN_EXECUTION=
case "$MODE" in
    --run-frozen) FROZEN_EXECUTION=cmdlist ;;
    --run-frozen-direct) FROZEN_EXECUTION=direct ;;
esac
readonly FROZEN_EXECUTION
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
readonly ROOT
COMMON_GIT="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)"
readonly COMMON_GIT
NPU_WORK="$(dirname "$(dirname "$COMMON_GIT")")"
readonly NPU_WORK
readonly MLIR_AIE_PATH="$NPU_WORK/mlir-aie"
readonly REGISTER_DB="$MLIR_AIE_PATH/lib/Dialect/AIE/Util/aie_registers_aie2.json"
readonly FROZEN_ROOT="$MLIR_AIE_PATH/build/test/npu-xrt/add_one_using_dma"
readonly FROZEN_TEST="$FROZEN_ROOT/test.exe"
readonly ELF_ROOT="$MLIR_AIE_PATH/build/test/npu-xrt/add_one_objFifo_elf"
readonly ELF_TEST="$ELF_ROOT/test.exe"
readonly XRT_ROOT=/opt/xilinx/xrt
readonly XRT_COREUTIL="$XRT_ROOT/lib/libxrt_coreutil.so.2"
readonly XRT_COREUTIL_VERSIONED="$XRT_ROOT/lib/libxrt_coreutil.so.2.23.0"
readonly XRT_CORE="$XRT_ROOT/lib/libxrt_core.so.2"
readonly XRT_CORE_VERSIONED="$XRT_ROOT/lib/libxrt_core.so.2.23.0"
readonly XRT_XDNA="$XRT_ROOT/lib/libxrt_driver_xdna.so.2"
readonly XRT_XDNA_VERSIONED="$XRT_ROOT/lib/libxrt_driver_xdna.so.2.23.0"
readonly XRT_RUNNER="$XRT_ROOT/bin/unwrapped/xrt-runner"
readonly XRT_PHOENIX_ARCHIVE="$XRT_ROOT/share/amdxdna/bins/xrt_smi_phx.a"
export MLIR_AIE_PATH
readonly SERVER="$ROOT/build/tools/phoenix-vfio-user/phoenix-vfio-user"
readonly DRIVER_PIN=216cefececd74effcd7a88350c71b99f5ef9a215
readonly FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin
readonly FIRMWARE_SHA256=d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e
readonly GUEST_KERNEL_VERSION=7.1.5-custom+
readonly GUEST_KERNEL=/boot/vmlinuz-7.1.5-custom+
readonly GUEST_KERNEL_SHA256=4c069ffa4da7a3b9e2ab5b16d514a1f0fd208c059221938a2c30e8aa47347bb4
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_ID
readonly RUN_DIR="$ROOT/build/experiments/phoenix-vfio-user/$RUN_ID"
readonly VFIO_SOCKET="/tmp/xdna-emu-vfio-$$.sock"
readonly MONITOR_SOCKET="/tmp/xdna-emu-monitor-$$.sock"
readonly RESPONSE="$RUN_DIR/server-nonce.bin"
readonly DRIVER_SOURCE="$RUN_DIR/driver-source"
readonly GUEST_ROOT="$RUN_DIR/guest-root"
readonly INITRAMFS="$RUN_DIR/initramfs.cpio.gz"
readonly GUEST_LOG="$RUN_DIR/guest.log"

for tool in git qemu-system-x86_64; do
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

wait_for_guest_result() {
    local qemu=$1
    local server=$2

    for ((attempt = 0; attempt < 1800; ++attempt)); do
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

prepare_driver_guest() {
    local driver_repo
    local driver_archive="$RUN_DIR/driver-source.tar"
    local driver_module="$DRIVER_SOURCE/drivers/accel/amdxdna/amdxdna.ko"
    local driver_build_log="$RUN_DIR/driver-build.log"
    local initramfs_cpio="$RUN_DIR/initramfs.cpio"
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
    local module_vermagic
    local module_path
    local npu_direct_dir="$RUN_DIR/npu-direct"
    local qemu_package_version

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
                frozen_xclbin_hash=c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6
                ;;
            peano)
                frozen_xclbin_hash=71deb139ac91bba3a50099bfd0c3a4a966f00e1977eab017589ef51a36d63865
                ;;
        esac
        [[ "$(sha256sum "$FROZEN_TEST" | awk '{print $1}')" == \
            511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806 &&
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
    if [[ -n "$FROZEN_COMPILER$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ]]; then
        [[ "$(dpkg-query -W -f='${Version}' xrt-base)" == 2.23.0 &&
            "$(dpkg-query -W -f='${Version}' xrt-npu)" == 2.23.0 &&
            "$(dpkg-query -W -f='${Version}' xrt_plugin-amdxdna)" == 2.23.1 ]] || {
            echo "installed XRT packages do not match the pinned versions" >&2
            return 1
        }
        [[ -z "$(dpkg --verify xrt-base xrt-npu xrt_plugin-amdxdna)" ]] || {
            echo "installed XRT packages have locally modified files" >&2
            return 1
        }
        [[ "$(sha256sum "$XRT_COREUTIL_VERSIONED" | awk '{print $1}')" == \
            461d3a9de0db09080ea1ad6e66476f012f983bc186772f14730d7eb03c356e76 &&
            "$(sha256sum "$XRT_CORE_VERSIONED" | awk '{print $1}')" == \
            69d585730b671dfbe6c48fa7000e398803880fac4ce204c9c274e50d47017fdd &&
            "$(sha256sum "$XRT_XDNA_VERSIONED" | awk '{print $1}')" == \
            4d6ed092a3ed805edd93053561b02946daa1187c3135a39674630b604455fd91 ]] || {
            echo "installed XRT runtime does not match the pinned hashes" >&2
            return 1
        }
    fi
    if [[ "$MODE" == "--run-npu-direct" ]]; then
        [[ "$(sha256sum "$XRT_PHOENIX_ARCHIVE" | awk '{print $1}')" == \
            0970f2038ee7dcf33dbc704c2ac55271b94687b5a17181cdd2c9118ff195c508 &&
            "$(sha256sum "$XRT_RUNNER" | awk '{print $1}')" == \
            f39e2399ab4d70f6bd646a2ad2b5a2b339cee2339c4c4597073d93dc7e3e6089 ]] || {
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
    [[ "$(git -C "$driver_repo" rev-parse HEAD)" == "$DRIVER_PIN" ]] || {
        echo "primary driver repository is not at the pinned commit" >&2
        return 1
    }
    [[ -z "$(git -C "$driver_repo" status --porcelain \
        --untracked-files=all -- drivers/accel/amdxdna \
        drivers/accel/tools/configure_kernel.sh include)" ]] || {
        echo "primary driver sources used by the guest are dirty" >&2
        return 1
    }

    mkdir -p "$DRIVER_SOURCE"
    git -C "$driver_repo" archive --format=tar --output="$driver_archive" \
        "$DRIVER_PIN" drivers/accel/amdxdna \
        drivers/accel/tools/configure_kernel.sh include
    tar -xf "$driver_archive" -C "$DRIVER_SOURCE"
    KERNEL_VER="$GUEST_KERNEL_VERSION" \
        KERNEL_SRC="/usr/src/linux-headers-$GUEST_KERNEL_VERSION" \
        OUT="$DRIVER_SOURCE/drivers/accel/amdxdna/config_kernel.h" \
        sh "$DRIVER_SOURCE/drivers/accel/tools/configure_kernel.sh" \
        >"$driver_build_log" 2>&1
    make -C "$DRIVER_SOURCE/drivers/accel/amdxdna" \
        KERNEL_VER="$GUEST_KERNEL_VERSION" \
        KERNEL_SRC="/usr/src/linux-headers-$GUEST_KERNEL_VERSION" \
        XDNA_HASH="$DRIVER_PIN" XDNA_DATE=20260728 LLVM=1 modules \
        >>"$driver_build_log" 2>&1
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
        if [[ -n "$FROZEN_COMPILER$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ]]; then
            if [[ -n "$FROZEN_COMPILER" ]]; then
                ldd "$FROZEN_TEST"
            elif [[ -n "$ELF_COMPILER" ]]; then
                ldd "$ELF_TEST"
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
        if [[ -n "$FROZEN_COMPILER$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ]]; then
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

    if [[ -n "$FROZEN_COMPILER$ELF_COMPILER" || "$MODE" == "--run-npu-direct" ]]; then
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

    if [[ -n "$FROZEN_COMPILER" ]]; then
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
