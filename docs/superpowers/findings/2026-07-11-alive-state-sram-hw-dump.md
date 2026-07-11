# Alive-state device-SRAM HW dump: the mgmt channel struct is real, alive-publish is past the wall

Date: 2026-07-11
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`
Branch: `feat/m2c-mapping-boot-to-idle` (unmerged)

## Verdict: option C (writer past 0x8cb1), HW-confirmed. PA-0 alias not the mechanism.

The memory-map check asked whether the Xtensa management core's local PA 0 (where
the pre-wall `0x5044` publisher writes the pointer `0x030bb000`) physically aliases
device SRAM `0x030bf000` (FW_ALIVE_OFF). Two prior conclusions, now joined by a HW
dump, settle it:

1. Open source documents no such alias (driver/aie-rt/RyzenAI-SW sweep). The
   management core's local memory map is not described anywhere; only the
   host-side contract is (driver consumes FW_ALIVE_OFF as a **device-absolute**
   SRAM pointer, de-based by `sram_dev_addr = 0x3080000`).
2. The Xtensa's private PA 0 is off every host BAR and can never be host-read, so
   the alias is not host-confirmable in principle.
3. **HW dump (this finding):** the firmware builds a full, structured
   `mgmt_mbox_chann_info` object in device SRAM at `0x030bb000`, packed with
   device-absolute pointers it composed itself. Our emulator produces none of this
   before the `0x8cb1` framing wall (it executes exactly one SRAM-band store,
   `0x030b27c0 <- 0`). Therefore all struct-building -- and the alive-pointer
   publish that points at it -- happens past the wall, via direct device-absolute
   stores. The pre-wall PA-0 write is local bookkeeping, not the host doorbell.

Positing a PA-0 alias only for the final pointer, while the firmware builds the
entire surrounding struct by direct device-absolute stores and demonstrably can
address `0x030bf000` the same way, is gratuitous and unsupported. Occam favors a
direct device-absolute store past the wall. (Not disprovable with 100% certainty --
PA 0 is unreadable -- but it is the strictly-less-parsimonious story.)

## The dump

Host userspace read of BAR2 (`resource2`, device SRAM base `0x3080000`) on a live,
alive NPU (`0000:c6:00.1`), fat DEVEL driver, `autosuspend_ms=-1`. Read-only,
single pass -- BAR2 SRAM, not the BAR0 mgmt aperture (the hard-reset hazard).
Tool: `build/experiments/hw-introspect/res_read`; raw log `sram_dump.log`.

### FW_ALIVE_OFF (BAR2:0x3f000 = device 0x030bf000)

```
+0x3f000: 0x00000000    <- alive pointer slot, cleared by driver post-init
+0x3f004: 0x00000001
```

The driver clears FW_ALIVE_OFF after reading the pointer at init
(`aie2_pci.c` "Must clear address at FW_ALIVE_OFF"). Its transient value at alive
was `0x030bb000` (2026-07-07 kernel-side boot_capture, 72.8 ms). The persistent
struct it pointed at is the tell:

### mgmt_mbox_chann_info (BAR2:0x3b000 = device 0x030bb000)

One 0x40-byte unit (repeats across the read window):

```
+0x00: 0x030ec000   ring A head ptr loc   (device-absolute; in mbox aperture BAR4)
+0x04: 0x030ec004   ring A tail ptr loc
+0x08: 0x030bc000   ring A buffer base     (BAR2:0x3c000, live)
+0x0c: 0x00000400   ring A size = 1024
+0x10: 0x030ed000   ring B head ptr loc
+0x14: 0x030ed004   ring B tail ptr loc
+0x18: 0x030bd000   ring B buffer base     (BAR2:0x3d000, live)
+0x1c: 0x00000400   ring B size = 1024
+0x20: 0x55504e5f   MAGIC "_NPU" = MGMT_MBOX_MAGIC
+0x24: 0x0000000e   (14)
+0x28: 0x00000005   protocol major = 5
+0x2c: 0x00000008   protocol minor = 8   (5.8 -- matches npu1 fw_feature_table)
+0x30..0x3c: 0
```

### Live ring traffic (buffer bases from the struct)

```
x2i BAR2:0x3c000: 0000000c 0001000c 1d000000 0000010a 00000002 00000001 ...
i2x BAR2:0x3d000: 00000004 00010004 1d000000 0000010a 00000000 00000004 ...
```

`1d000000 0000010a` are mailbox message headers -- the channel is not just
initialized, it is actively carrying traffic.

## Consequence for the emulator arc

The blocker to observable-alive was never the alias question; it is the `0x8cb1`
framing collision, because the struct-build + pointer-publish live past it. Both a
hypothetical alias and the (favored) direct-store mechanism require the post-wall
code, so crossing the wall is required regardless.

Crossing it does NOT force PSP-loader RE (JTAG/BIOS, blocked). This dump is the
alive-state **ground truth**: exact struct layout, MAGIC, version bytes, ring
buffer addresses/sizes. That is a behavioral target for reconstructing the
post-wall overlay by matching emulated device-SRAM writes against observed HW
output -- HW-in-the-loop derivation, a legitimate source tier, no PSP RE.

## Reproduction

```
pkexec sh -c 'R=.../hw-introspect/res_read; P=/sys/bus/pci/devices/0000:c6:00.1/resource2;
  "$R" "$P" 0x3f000 4; "$R" "$P" 0x3b000 64; "$R" "$P" 0x3c000 8; "$R" "$P" 0x3d000 8'
```

Requires: fat DEVEL driver loaded with `autosuspend_ms=-1`, NPU alive. Read-only,
single-pass, BAR2 only. Never sustained-poll; never target BAR0.
