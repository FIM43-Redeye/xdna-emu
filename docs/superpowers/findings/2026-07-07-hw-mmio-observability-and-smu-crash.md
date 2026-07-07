# Host MMIO observability of the NPU mgmt processor — capability, limits, and an SMU hard-crash

**Date:** 2026-07-07
**Issue:** #140 firmware-emulation dream / boot-to-idle, HW-in-loop revisit.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).
**Status:** Capability PROVEN. Boot-to-idle completion event NOT yet captured (the
capture experiment hard-reset the machine). Banked with a safe resume path.

## Why this exists

Boot-to-idle was banked as "the completion contract lives in AIE-array behavior,
not derivable from the firmware alone" — with the escape hatch: revisit with
**hardware in the loop**. This session opened that hatch: the Phoenix NPU's
management registers are directly readable from the host. The point (per Maya):
get *proper timing from observed hardware fact* instead of calibrating blindly on
mechanisms we don't understand.

## TL;DR

- **Direct host MMIO into the NPU works.** `/dev/mem` and PCI sysfs `resourceN`
  mmap read the BARs. `CONFIG_STRICT_DEVMEM` is off, no lockdown LSM. Verified:
  userspace read of BAR2+0x3c000 == the driver's `ringbuf`, byte-identical.
- **The all-0xFF cold reads were power-gating, not an access wall.** The internal
  fabric idle-suspends (5s autosuspend) and floats reads high. `autosuspend_ms=-1`
  pins it on — verified a 20s cold read returns live data. "You're screwed" is
  refuted.
- **fw-log/trace is firmware-gated dead on NPU1.** `aie2_config_fw_log`
  (`aie2_message.c:487`) returns `-EOPNOTSUPP` because Phoenix firmware lacks
  `MSG_OP_CONFIG_FW_LOG`; silently disabled. Not host-fixable.
- **The event register that gates the boot wall is host-readable.** fw
  `0x27010d28` -> drv `0x3010d28` -> **BAR0:0x10d28** (the wake-path event source,
  the only-writer of readiness `[0x10f40]`). Reads 0 at idle.
- **A sustained, ungapped MMIO poll of the mgmt aperture HARD-RESET the machine.**
  See Hazard below. Single/occasional reads are safe; tight-loop polling is not.

## Fat DEVEL driver recipe (the enabling step)

Stock DKMS `amdxdna` is the lean upstream build (bare debugfs). The vendor
superset `xdna-driver/src/driver/amdxdna` builds with `-DAMDXDNA_DEVEL` and
exposes `dpm_level`, `powerstate`, `nputest`, `ringbuf`, `msg_queue`,
`telemetry_*`, `dump_fw_log/trace` (last two dead on NPU1).

- Build: `cd src/driver/amdxdna && make LLVM=1`  (kernel is **clang 21** + KCFI +
  RANDSTRUCT; gcc cannot build a loadable module for it. Build against the running
  kernel tree so the RANDSTRUCT seed matches.)
- Sign (module-sig enforcement is on): `sign-file sha512 <MOK.priv> <MOK.pem> <ko>`
  using the enrolled MOK at `/var/lib/shim-signed/mok/` (the `MOK.pem` bundle
  carries the trusted `298B…` module key).
- Load: `rmmod amdxdna && insmod <ko> autosuspend_ms=-1`. Revert: `rmmod amdxdna
  && modprobe amdxdna` (or reboot — reboot reverts to stock automatically).

## NPU1 BAR / register map (verified)

BAR0 = `resource0`, phys 0x90c00000, 512K, aperture base drv 0x3000000.
BAR2 = `resource2`, phys 0x8e20800000, 256K, SRAM, base 0x3080000.
BAR4 = mbox regs, base 0x30C0000.  (drv = fw − 0x24000000; off = drv − aperbase.)

| what | fw | drv | BAR:off | idle |
|------|----|-----|---------|------|
| fw wake-path event source | 0x27010d28 | 0x3010d28 | BAR0:0x10d28 | 0 |
| PWAITMODE (PSP wait) | — | 0x3010034 | BAR0:0x10034 | 1 |
| PSP_SEC_INTR | — | 0x3010090 | BAR0:0x10090 | 0 |
| SMU_PWRMGMT_INTR | — | 0x3010094 | BAR0:0x10094 | 1 |
| PSP/SMU scratch2..9 | — | 0x30100A0.. | BAR0:0x100a0..bc | mixed |
| x2i / i2x mailbox rings | — | — | BAR2:0x3c000 / 0x3d000 | live |
| FW_ALIVE_OFF | — | 0x30BF000 | BAR2:0x3f000 | 0 post-boot |

Per-column poll pages fw `0x2727_n000` -> drv `0x3270000` are **outside every host
BAR** (above 0x3100000) — not directly host-readable. The event source (0x10d28)
is the wall-crossing trigger and IS readable; that's the target signal.

## HAZARD — do not sustained-poll the mgmt aperture

**A 30s ungapped tight-loop read of BAR0 `0x10c00–0x10e00` (PSP/SMU-adjacent),
~58M reads, silently hard-reset the whole machine.** Evidence: `pkexec mmio_watch`
launched 16:51:46, watcher duration 30s, platform dead at 16:52:17 (exactly
watcher-end). **Zero** kernel log — no MCE, no panic, no amdxdna error — because
wedging the SMU (which owns the SoC's power/clocks) takes the system down below
the OS's ability to log. Machine recovered fully on reboot (stock driver, no
damage).

Rules:
- **Never** run an ungapped / high-rate poll over the mgmt/PSP/SMU register space.
- Single or occasional one-shot reads (`res_read`) are safe — we did many all
  session with no issue.
- To capture a transient event, do it **kernel-side, one read per event**, hooked
  to the completion path — never by host polling. See resume path.

## Safe resume path (when we return to firmware)

Goal: capture what the array/completion hardware writes to `0x10d28` (and
neighbors) when a column completes — the completion→event mapping that un-banks
boot-to-idle and yields the timing the firmware emulation needs.

Do it without polling: add a tiny debugfs to the fat driver that **snapshots
0x10d28 + the scratch/interrupt regs once, from inside the mailbox-RX / completion
IRQ handler** (`amdxdna_mailbox.c` irq path). One kernel read at the exact moment,
per completion. Drive it with the `add_one_using_dma` workload
(`mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/`, run
`./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`, non-root). Correlate the
captured event value with the RE'd wake path.

The captured `wl_watch.log` (before the crash) shows BAR0:0x10d7c byte-shifting
`05060706 -> a0a0a0a0` during the 50 workloads — a live mgmt register moves on
workload activity, so the signal is there; we just need to read it safely.

## Artifacts

- Tools + plan: `build/experiments/hw-introspect/` (gitignored, on-disk):
  `res_read`, `mmio_read`, `mmio_watch` (+ `.c`), `PLAN.md`, `wl_watch.log`.
- Do NOT reuse `mmio_watch` against the mgmt aperture without a large gap and a
  narrow, non-PSP/SMU target — or better, replace it with the kernel-side hook.
