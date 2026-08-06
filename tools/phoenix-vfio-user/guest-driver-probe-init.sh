#!/bin/busybox sh
set -eu

mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t debugfs debugfs /sys/kernel/debug

modprobe 8250
exec </dev/ttyS0 >/dev/ttyS0 2>&1

fail()
{
	echo "PHOENIX_DMESG_BEGIN"
	dmesg
	echo "PHOENIX_DMESG_END"
	echo "PHOENIX_DRIVER_PROBE_FAIL: $1"
	poweroff -f
	while :; do sleep 1; done
}

echo "PHOENIX_DRIVER_PROBE_BEGIN"
echo "kernel=$(uname -r)"
echo "cmdline=$(cat /proc/cmdline)"
elf_compiler=
frozen_compiler=
frozen_execution=
npu_direct=
context_repartition=
async_error=
async_error_batch=
real_column_gate=
real_column_gate_arm=
phoenix_npi_read=
if [ -f /run-frozen/compiler ]; then
	frozen_compiler=$(cat /run-frozen/compiler)
	case "$frozen_compiler" in
		chess | peano) ;;
		*) fail "invalid frozen compiler $frozen_compiler" ;;
	esac
	[ -x /run-frozen/test.exe ] || fail "frozen test executable is missing"
	[ -r /run-frozen/aie.xclbin ] || fail "frozen xclbin is missing"
	[ -r /run-frozen/insts.bin ] || fail "frozen instructions are missing"
	[ -r /run-frozen/execution-mode ] || fail "frozen execution mode is missing"
	frozen_execution=$(cat /run-frozen/execution-mode)
	case "$frozen_execution" in
	cmdlist | direct) ;;
	*) fail "invalid frozen execution mode $frozen_execution" ;;
	esac
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
elif [ -f /run-elf/compiler ]; then
	elf_compiler=$(cat /run-elf/compiler)
	case "$elf_compiler" in
		chess | peano) ;;
		*) fail "invalid pinned ELF compiler $elf_compiler" ;;
	esac
	[ -x /run-elf/test.exe ] || fail "pinned ELF test executable is missing"
	[ -r /run-elf/aie.xclbin ] || fail "pinned ELF xclbin is missing"
	[ -r /run-elf/insts.elf ] || fail "pinned transaction ELF is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
elif [ -f /run-npu/recipe_latency.json ]; then
	npu_direct=1
	[ -r /run-npu/profile_latency.json ] ||
		fail "NPU profile is missing"
	[ -r /run-npu/validate.xclbin ] ||
		fail "NPU validation xclbin is missing"
	[ -r /run-npu/nop.elf ] ||
		fail "NPU no-op ELF is missing"
	[ -x /opt/xilinx/xrt/bin/unwrapped/xrt-runner ] ||
		fail "XRT runner is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
elif [ -d /run-real-column-gate ]; then
	real_column_gate=1
	[ -r /run-real-column-gate/arm ] ||
		fail "real column-gate arm identity is missing"
	real_column_gate_arm=$(cat /run-real-column-gate/arm)
	case "$real_column_gate_arm" in
	control | treatment) ;;
	*) fail "invalid real column-gate arm $real_column_gate_arm" ;;
	esac
	for artifact in bridge-trace-runner libxdna_emu.so xdna-clock-query aie.xclbin arm.insts.bin canary.insts.bin; do
		[ -r "/run-real-column-gate/$artifact" ] ||
			fail "real column-gate artifact $artifact is missing"
	done
	[ -x /run-real-column-gate/bridge-trace-runner ] ||
		fail "real column-gate runner is not executable"
	[ -x /run-real-column-gate/xdna-clock-query ] ||
		fail "real column-gate clock query is not executable"
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
elif [ -d /run-phoenix-npi-read ]; then
	phoenix_npi_read=1
elif [ -d /run-async-error ]; then
	async_error=1
	[ ! -f /run-async-error/batch-only ] || async_error_batch=1
	[ -x /run-async-error/async-error-probe ] ||
		fail "async-error producer is missing"
	for artifact in aie.xclbin PM.xclbin PM.insts BATCH.insts A.insts B.insts C.insts D.insts E.insts S2MM.insts F.insts; do
		[ -r "/run-async-error/$artifact" ] ||
			fail "async-error artifact $artifact is missing"
	done
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
elif [ -d /run-repartition ]; then
	context_repartition=1
	[ -x /run-repartition/context-repartition ] ||
		fail "context-repartition producer is missing"
	for artifact in A.xclbin A.insts B.xclbin B.insts; do
		[ -r "/run-repartition/$artifact" ] ||
			fail "context-repartition artifact $artifact is missing"
	done
	[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ] ||
		fail "XRT core runtime is missing"
	[ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2 ] ||
		fail "normal XDNA XRT plugin is missing"
else
	[ ! -e /opt/xilinx/xrt ] || fail "XRT is present in driver-only probe"
fi
[ ! -e /opt/xilinx/xrt/lib/libxrt_driver_emu.so.2 ] ||
	fail "emulator XRT plugin is present"
grep -qw hypervisor /proc/cpuinfo &&
	fail "guest CPU still advertises a hypervisor"
[ ! -e /lib/modules/"$(uname -r)"/extra/amdxdna_legacy.ko ] ||
	fail "legacy driver is present"

npu_bdf=
for device_path in /sys/bus/pci/devices/*; do
	[ "$(cat "$device_path/vendor")" = 0x1022 ] || continue
	[ "$(cat "$device_path/device")" = 0x1502 ] || continue
	[ -z "$npu_bdf" ] || fail "multiple Phoenix functions found"
	npu_bdf=${device_path##*/}
done
[ -n "$npu_bdf" ] || fail "Phoenix PCI function not found"
echo "bdf=$npu_bdf"

if [ -n "$frozen_compiler" ] || [ -n "$elf_compiler" ] ||
	[ -n "$npu_direct" ] || [ -n "$context_repartition" ] ||
	[ -n "$async_error" ] || [ -n "$real_column_gate" ]; then
	if [ "$frozen_execution" = direct ] || [ -n "$elf_compiler" ] ||
		[ -n "$npu_direct" ]; then
		modprobe amdxdna dyndbg=+p tdr_timeout_ms=0 force_cmdlist=N ||
			fail "primary amdxdna module did not load"
	else
		modprobe amdxdna dyndbg=+p tdr_timeout_ms=0 ||
			fail "primary amdxdna module did not load"
	fi
else
	modprobe amdxdna dyndbg=+p ||
		fail "primary amdxdna module did not load"
fi
if { [ -n "$frozen_compiler" ] || [ -n "$elf_compiler" ] ||
	[ -n "$npu_direct" ] || [ -n "$context_repartition" ] ||
	[ -n "$async_error" ] || [ -n "$real_column_gate" ]; } &&
	[ "$(cat /sys/module/amdxdna/parameters/tdr_timeout_ms)" != 0 ]; then
	fail "driver TDR was not disabled for slow emulation"
fi
if [ -n "$frozen_compiler" ] || [ -n "$elf_compiler" ] ||
	[ -n "$npu_direct" ] || [ -n "$context_repartition" ] ||
	[ -n "$async_error" ] || [ -n "$real_column_gate" ]; then
	echo "tdr_timeout_ms=0"
	force_cmdlist=$(cat /sys/module/amdxdna/parameters/force_cmdlist)
	if [ "$frozen_execution" = direct ] || [ -n "$elf_compiler" ] ||
		[ -n "$npu_direct" ]; then
		[ "$force_cmdlist" = N ] ||
			fail "force_cmdlist is $force_cmdlist in EXEC_DPU mode"
	else
		[ "$force_cmdlist" = Y ] ||
			fail "force_cmdlist is $force_cmdlist in command-list mode"
	fi
	echo "force_cmdlist=$force_cmdlist"
fi

attempt=0
while [ ! -e /dev/accel/accel0 ] && [ "$attempt" -lt 60 ]; do
	sleep 1
	attempt=$((attempt + 1))
done
[ -e /dev/accel/accel0 ] || fail "/dev/accel/accel0 was not created"
echo "accel_node=/dev/accel/accel0"
echo "probe=complete"

force_iova=$(cat /sys/module/amdxdna/parameters/force_iova)
[ "$force_iova" = N ] || fail "force_iova is not at its false default"
echo "force_iova=$force_iova"

[ -d /sys/kernel/debug ] || fail "debugfs root is missing"
carveouts=$(find /sys/kernel/debug -type f -name carveout)
carveout_count=$(find /sys/kernel/debug -type f -name carveout | wc -l)
[ "$carveout_count" -eq 1 ] ||
	fail "expected exactly one carveout node, found $carveout_count"
carveout=$carveouts
echo "0x10000000@0x60000000" >"$carveout"
carveout_value=$(cat "$carveout")
[ "$carveout_value" = "0x10000000@0x60000000" ] ||
	fail "carveout readback was $carveout_value"
echo "carveout=$carveout_value"

msix_count=$(find "/sys/bus/pci/devices/$npu_bdf/msi_irqs" \
	-mindepth 1 -maxdepth 1 | wc -l)
[ "$msix_count" -eq 16 ] || fail "expected 16 MSI-X vectors, found $msix_count"

if [ -n "$phoenix_npi_read" ]; then
	echo "PHOENIX_NPI_READ_BEGIN"
	npi_lock_nodes=$(find /sys/kernel/debug -type f -name phoenix_npi_lock)
	npi_lock_count=$(find /sys/kernel/debug -type f -name phoenix_npi_lock | wc -l)
	[ "$npi_lock_count" -eq 1 ] ||
		fail "expected exactly one Phoenix NPI lock node, found $npi_lock_count"
	npi_lock_value=$(cat "$npi_lock_nodes") || fail "Phoenix NPI lock read failed"
	printf '%s\n' "$npi_lock_value" | grep -Eq '^0x[0-9a-f]{8}$' ||
		fail "invalid Phoenix NPI lock value $npi_lock_value"
	echo "PHOENIX_NPI_READ value=$npi_lock_value"
	echo "PHOENIX_NPI_READ_PASS"
fi

if [ -n "$frozen_compiler" ]; then
	echo "PHOENIX_FROZEN_BEGIN $frozen_compiler"
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	if ! /run-frozen/test.exe -x /run-frozen/aie.xclbin \
		-k MLIR_AIE -i /run-frozen/insts.bin; then
		fail "frozen $frozen_compiler kernel failed"
	fi
	echo "PHOENIX_FROZEN_PASS $frozen_compiler"
fi
if [ -n "$elf_compiler" ]; then
	echo "PHOENIX_PINNED_ELF_BEGIN $elf_compiler"
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	if ! /run-elf/test.exe -x /run-elf/aie.xclbin \
		-k MLIR_AIE -i /run-elf/insts.elf; then
		fail "pinned ELF $elf_compiler kernel failed"
	fi
	echo "PHOENIX_PINNED_ELF_PASS $elf_compiler"
fi
if [ -n "$npu_direct" ]; then
	echo "PHOENIX_EXEC_DPU_BEGIN"
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	if ! timeout -k 5 120 /opt/xilinx/xrt/bin/unwrapped/xrt-runner \
		--recipe /run-npu/recipe_latency.json \
		--profile /run-npu/profile_latency.json \
		--iterations 1 --dir /run-npu --report -; then
		fail "direct EXEC_DPU no-op failed"
	fi
	echo "PHOENIX_EXEC_DPU_PASS"
fi
if [ -n "$real_column_gate" ]; then
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	export XDNA_EMU_DIR=/run-real-column-gate
	export BRIDGE_RUNNER_ASYNC_CTX=0
	export BRIDGE_RUNNER_REUSE_CONTEXT=0
	gate_dir=/run-real-column-gate

	echo "PHOENIX_REAL_COLUMN_GATE_MISMATCH_BEGIN"
	if timeout -k 5 650 "$gate_dir/bridge-trace-runner" \
		--xclbin "$gate_dir/aie.xclbin" \
		--instr "$gate_dir/arm.insts.bin" \
		--trace-out "$gate_dir/mismatch.trace.bin" \
		--output "$gate_dir/mismatch.out.bin" \
		--qos-gops 1 --qos-fps 1000 --expect-placement 2:1 -v \
		>"$gate_dir/mismatch.stdout" 2>"$gate_dir/mismatch.stderr"; then
		fail "mismatched placement unexpectedly reached submission"
	fi
	cat "$gate_dir/mismatch.stderr"
	grep -Fq "live hardware context placement mismatch: expected 2:1, got 1:1" \
		"$gate_dir/mismatch.stderr" ||
		fail "mismatched placement did not fail at the live-placement guard"
	[ ! -e "$gate_dir/mismatch.trace.bin" ] ||
		fail "mismatched placement produced a trace"
	[ ! -e "$gate_dir/mismatch.out.bin" ] ||
		fail "mismatched placement produced output"
	echo "PHOENIX_REAL_COLUMN_GATE_MISMATCH_PASS"

	echo "PHOENIX_REAL_COLUMN_GATE_CLOCK_BEFORE_BEGIN"
	"$gate_dir/xdna-clock-query" || fail "before-run clock query failed"
	echo "PHOENIX_REAL_COLUMN_GATE_CLOCK_BEFORE_END"
	echo "PHOENIX_REAL_COLUMN_GATE_ARM_BEGIN $real_column_gate_arm"
	if ! timeout -k 5 650 "$gate_dir/bridge-trace-runner" \
		--xclbin "$gate_dir/aie.xclbin" \
		--instr "$gate_dir/arm.insts.bin" \
		--trace-out "$gate_dir/arm.trace.bin" \
		--output "$gate_dir/arm.out.bin" \
		--qos-gops 1 --qos-fps 1000 --expect-placement 1:1 -v \
		>"$gate_dir/arm.stdout" 2>"$gate_dir/arm.stderr"; then
		cat "$gate_dir/arm.stderr"
		fail "real column-gate $real_column_gate_arm command failed"
	fi
	echo "PHOENIX_REAL_COLUMN_GATE_CLOCK_AFTER_BEGIN"
	"$gate_dir/xdna-clock-query" || fail "after-run clock query failed"
	echo "PHOENIX_REAL_COLUMN_GATE_CLOCK_AFTER_END"

	for artifact in arm.out.bin arm.trace.bin; do
		echo "PHOENIX_REAL_COLUMN_GATE_BLOB_${artifact}_BEGIN"
		/bin/busybox gzip -c "$gate_dir/$artifact" | /bin/busybox base64
		echo "PHOENIX_REAL_COLUMN_GATE_BLOB_${artifact}_END"
	done
	for artifact in arm.stdout arm.stderr mismatch.stdout mismatch.stderr; do
		echo "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_BEGIN"
		cat "$gate_dir/$artifact"
		echo "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_END"
	done
	echo "PHOENIX_REAL_COLUMN_GATE_ARM_PASS $real_column_gate_arm"

	echo "PHOENIX_REAL_COLUMN_GATE_CANARY_BEGIN"
	if ! timeout -k 5 650 "$gate_dir/bridge-trace-runner" \
		--xclbin "$gate_dir/aie.xclbin" \
		--instr "$gate_dir/canary.insts.bin" \
		--trace-out "$gate_dir/canary.trace.bin" \
		--output "$gate_dir/canary.out.bin" \
		--qos-gops 1 --qos-fps 1000 --expect-placement 1:1 -v \
		>"$gate_dir/canary.stdout" 2>"$gate_dir/canary.stderr"; then
		cat "$gate_dir/canary.stderr"
		fail "fresh-context canary command failed"
	fi
	for artifact in canary.out.bin canary.trace.bin; do
		echo "PHOENIX_REAL_COLUMN_GATE_BLOB_${artifact}_BEGIN"
		/bin/busybox gzip -c "$gate_dir/$artifact" | /bin/busybox base64
		echo "PHOENIX_REAL_COLUMN_GATE_BLOB_${artifact}_END"
	done
	for artifact in canary.stdout canary.stderr; do
		echo "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_BEGIN"
		cat "$gate_dir/$artifact"
		echo "PHOENIX_REAL_COLUMN_GATE_TEXT_${artifact}_END"
	done
	echo "PHOENIX_REAL_COLUMN_GATE_CANARY_PASS"
	echo "PHOENIX_REAL_COLUMN_GATE_GUEST_PASS $real_column_gate_arm"
fi
if [ -n "$async_error" ]; then
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	if [ -n "$async_error_batch" ]; then
		echo "PHOENIX_ASYNC_ERROR_BATCH_BEGIN"
		if ! timeout -k 5 650 /run-async-error/async-error-probe \
				--async-error-one /dev/accel/accel0 \
				/run-async-error/aie.xclbin /run-async-error/BATCH.insts \
				0x2040304000b 0x401; then
			fail "batched core/compute-memory async-error producer failed"
		fi
		echo "PHOENIX_ASYNC_ERROR_BATCH_PASS"
	else
		echo "PHOENIX_ASYNC_ERROR_PM_BEGIN"
		if ! timeout -k 5 650 /run-async-error/async-error-probe \
				--async-error-one-observe-state /dev/accel/accel0 \
				/run-async-error/PM.xclbin /run-async-error/PM.insts \
				0x20303040006 0x201; then
			fail "core PM-address async-error producer failed"
		fi
		echo "PHOENIX_ASYNC_ERROR_PM_PASS"
		echo "PHOENIX_ASYNC_ERROR_S2MM_BEGIN"
		if ! timeout -k 5 650 /run-async-error/async-error-probe \
				--async-error-one /dev/accel/accel0 /run-async-error/aie.xclbin \
				/run-async-error/S2MM.insts 0x2070304000b 0x1; then
			fail "shim S2MM async-error producer failed"
		fi
		echo "PHOENIX_ASYNC_ERROR_S2MM_PASS"
		echo "PHOENIX_ASYNC_ERROR_BEGIN"
		# Debug emulation takes about 140 seconds per signed-firmware error service.
		if ! timeout -k 5 800 /run-async-error/async-error-probe \
				--async-error /dev/accel/accel0 /run-async-error/aie.xclbin \
				/run-async-error/A.insts /run-async-error/B.insts \
				/run-async-error/C.insts /run-async-error/D.insts \
				/run-async-error/E.insts /run-async-error/F.insts; then
			fail "async-error producer failed"
		fi
		echo "PHOENIX_ASYNC_ERROR_GUEST_PASS"
	fi
fi
if [ -n "$context_repartition" ]; then
	echo "PHOENIX_CONTEXT_REPARTITION_BEGIN"
	export XILINX_XRT=/opt/xilinx/xrt
	export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
	if ! timeout -k 5 120 /run-repartition/context-repartition \
		/run-repartition/A.xclbin /run-repartition/A.insts \
		/run-repartition/B.xclbin /run-repartition/B.insts; then
		fail "context repartition producer failed"
	fi
	echo "PHOENIX_CONTEXT_REPARTITION_PASS"
fi

echo "PHOENIX_LSPCI_BEGIN"
lspci -nnvv -s "$npu_bdf" || fail "lspci failed"
echo "PHOENIX_LSPCI_END"
echo "PHOENIX_MSIX_BEGIN"
echo "count=$msix_count"
cat /proc/interrupts
echo "PHOENIX_MSIX_END"
echo "PHOENIX_DMESG_BEGIN"
dmesg
echo "PHOENIX_DMESG_END"
echo "PHOENIX_DRIVER_PROBE_PASS"
sync
poweroff -f
while :; do sleep 1; done
