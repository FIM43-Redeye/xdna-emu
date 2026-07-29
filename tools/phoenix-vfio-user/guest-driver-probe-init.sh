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
grep -qw hypervisor /proc/cpuinfo &&
	fail "guest CPU still advertises a hypervisor"
[ ! -e /lib/modules/"$(uname -r)"/extra/amdxdna_legacy.ko ] ||
	fail "legacy driver is present"
[ ! -e /opt/xilinx/xrt ] || fail "XRT or emulator plugin is present"

npu_bdf=
for device_path in /sys/bus/pci/devices/*; do
	[ "$(cat "$device_path/vendor")" = 0x1022 ] || continue
	[ "$(cat "$device_path/device")" = 0x1502 ] || continue
	[ -z "$npu_bdf" ] || fail "multiple Phoenix functions found"
	npu_bdf=${device_path##*/}
done
[ -n "$npu_bdf" ] || fail "Phoenix PCI function not found"
echo "bdf=$npu_bdf"

if ! modprobe amdxdna dyndbg=+p; then
	fail "primary amdxdna module did not load"
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
