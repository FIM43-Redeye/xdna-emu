# amdxdna TDR recovery incomplete on Phoenix dev box

**Status:** Open. Investigation gated on the next reboot of the dev box;
the device is currently wedged from the run that triggered this finding.

**Related:**
[`2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`](2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md)
(prior `tdr_dump_ctx=1` history),
[`2026-05-19-interrupt-tier-c-tdr.md`](2026-05-19-interrupt-tier-c-tdr.md)
(emulator-side TDR shipped same day as this observation).

## What happened

`./scripts/emu-bridge-test.sh` (full sweep, HW + EMU, dual-compiler).
Phase 3 ran HW serially. Through test [55] every HW test PASSed in
~0.3s. Test [56] `HW debug_halt_probe (peano)` hit TDR at +85s with the
sweep's `*** TDR DETECTED ***` banner; the script's "HW retry: 1 TDR
result(s) rerunning serially" path fired. Every HW test from [57]
onward (71 of them) FAILed in ~7.7s — the precise user-space
`run.wait()` timeout. The wedge was not bounded: the cascade ran
through test [128] without ever recovering. EMU-side runs were
unaffected (they bypass the device entirely).

`debug_halt_probe` exists specifically to deliberately wedge the NPU
and exercise the kernel-driver TDR-recovery path. The kernel driver
*did* detect the wedge — the TDR fired — but the recovery action did
not restore the device to a state where subsequent submissions could
complete.

The host is uninterrupted. `xrt-smi examine` still reports the device
present; PCIe is alive; `amdxdna` module is loaded. From every
user-space-visible signal, the device looks fine. It just doesn't
finish any new submission.

## Why this is the interesting bug

Per [`2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`](2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md),
we removed `tdr_dump_ctx=1` from `/etc/modprobe.d/amdxdna.conf` on
2026-05-13 specifically because `tdr_dump_ctx=1` made TDR dump-only
(no `aie2_rq_stop_all/restart_all`) and turned every firmware
silent-drop into a permanent wedge. The trade was: with TDR allowed to
actually recover, *most* wedges should self-heal at the cost of an
occasional `modprobe -r` getting stuck in `synchronize_srcu` (the
poisoned-mailbox case from the chain-exec finding).

This case shows TDR firing *and the recovery path running* but the
device still failing every subsequent submission. So either:

- The recovery action (`aie2_rq_stop_all` + `aie2_rq_restart_all`)
  completes but leaves the firmware / hardware in a state that can't
  service new submissions.
- The recovery action itself blocks on something (e.g.,
  `aie2_dump_ctx` calls `aie2_get_app_health` which sends a mailbox
  message; if the mailbox is wedged, that send hangs).
- The user-space side (XRT / our plugin) holds context state that
  needs to be torn down + recreated to match the post-recovery driver
  state, and we don't.

Which one is it? Unknown. That's the investigation.

## Diagnostic surface (verified on this kernel)

The amdxdna driver at `/home/triple/npu-work/xdna-driver/src/driver/amdxdna/`
exposes a substantial diagnostic surface. None of it has been exercised
during a real wedge yet.

**Module parameters** (`/sys/module/amdxdna/parameters/`, world-readable):
- `tdr_dump_ctx` (currently `N` per the 2026-05-13 change) — `Y` = TDR
  dumps state and skips recovery. Flipping to `Y` mid-investigation
  would let us see "what does the dump look like *without* the recovery
  also having run."
- `timeout_in_sec` (default 2) — TDR watchdog cadence. Bumping
  temporarily can buy time to inspect mid-wedge state without TDR
  firing concurrently.
- `mailbox_polling` — switch mailbox between IRQ and polling. If IRQ
  delivery dies post-TDR, polling can confirm.
- `fw_log_level`, `fw_log_size`, `fw_trace_size`, `fw_trace_categories`,
  `poll_fw_trace` — firmware logging knobs. We have not turned these
  on; we should before the next attempt.
- `hws_debug_mode` — hardware-scheduler debug mode. Unknown effect;
  worth reading the source.
- `fw_reload` — force firmware reload. Possibly the recovery hammer.

**sysfs** (`/sys/class/accel/accel0/`, world-readable):
- `vbnv`, `device_type`, `fw_version` — confirm FW didn't crash/reload
  silently.

**debugfs** (`/sys/kernel/debug/accel/<bdf>/`, e.g.
`/sys/kernel/debug/accel/0000:c6:00.1/`, mode 0700 owned root —
**requires `pkexec` for any read**. The newer DRM accel subsystem
hangs the amdxdna debugfs node here, not under
`/sys/kernel/debug/dri/<id>/` like older accel drivers.):
- `tdr_control` — read shows TDR state (`started`, `counter`, `status`,
  `progress`, `timeout_sec`, `dump_only`). Write `dump` to force
  context dump; write `recover` to force recovery.
- `ctx_rq` — full snapshot of all partitions + contexts + queue state.
  The "tell me everything" file.
- `ringbuf`, `msg_queue` — mailbox ring buffer state + pending messages.
  Reveals whether mailbox is making progress.
- `get_app_health` — FW-reported state for context #1: DPU PC,
  transaction OP ID, exception PC/type, fatal error type. Sends a
  mailbox message — may hang if mailbox is dead.
- `telemetry_disabled`, `telemetry_health`, `telemetry_error_info`,
  `telemetry_profiling`, `telemetry_debug` — five 8 KB FW telemetry
  blocks.
- `dump_fw_log_buffer`, `dump_fw_trace_buffer` — raw 8 KB FW log /
  trace hex dumps.
- `nputest` — diagnostic harness. `echo 1 > nputest` runs mailbox
  health check (calls `aie2_check_protocol_version()`), reports pass /
  fail to dmesg. **This is the single most useful "is the mailbox
  alive?" probe.**
- `dump_fw_log`, `dump_fw_trace` — non-`_buffer` variants exposed
  alongside the buffer dumps; treat as the same data via a different
  entry point.

Note: `msg_queue`, `powerstate`, `dpm_level` were listed in earlier
revisions of this doc but are not actually present in the current
amdxdna debugfs node on this kernel — verified by `find` post-reboot
2026-05-20. The diagnostic surface is just what's enumerated above.

**Tracepoints** (`/sys/kernel/tracing/events/amdxdna_trace/`, requires
pkexec to enable). On this kernel only two tracepoints are exposed:
`amdxdna_debug_point` and `__amdxdna_trace_point`. The richer set
(`xdna_job`, `mbox_set_tail`, `mbox_set_head`, `mbox_irq_handle`,
`mbox_rx_worker`, `uc_irq_handle`, `uc_wakeup`, `mbox_poll_handle`)
listed in earlier revisions of this doc requires a build that defines
those tracepoints — our DKMS build does not. If we want them, we'd
have to enable them in the driver source and rebuild via
`./build.sh -release -refresh_dkms`.

**TDR code path** lives in
`/home/triple/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c`:
- `aie2_tdr_detect()` — checks `aie2_rq_is_all_context_stuck()`. False
  → TDR doesn't fire.
- `aie2_tdr_work()` — every `timeout_in_sec`, calls
  `aie2_tdr_force_recover()`.
- `aie2_tdr_force_recover()` — unconditional `aie2_rq_dump_all()`;
  then if `!dump_only`: `aie2_rq_stop_all()` + `aie2_rq_restart_all()`.
- Context stop/restart implementation is in `aie2_ctx_runqueue.c`
  (holds `io_sem`, calls `part_ctx_stop_wait()`).

## Post-reboot capture procedure

The amdxdna debugfs node is `/sys/kernel/debug/accel/<bdf>/` — find
the BDF with `ls /sys/kernel/debug/accel/`. On this dev box it is
`0000:c6:00.1`; substitute as needed. To get fw_log and fw_trace
content, the corresponding module params (`fw_log_level`,
`fw_trace_size`, `fw_trace_categories`, `poll_fw_trace`) must be set
*before* the run — they're not enabled by default and unset params
produce empty buffers (`dd: IO error: Invalid input`).

```bash
# Optional: enable FW logging before baseline (else fw_log/fw_trace
# come back empty). Persist by adding to /etc/modprobe.d/amdxdna.conf
# and reloading the module.
pkexec sh -c 'echo 3 > /sys/module/amdxdna/parameters/fw_log_level &&
              echo 65536 > /sys/module/amdxdna/parameters/fw_trace_size'

# Baseline (clean device, no submissions yet) — scripted form:
tools/amdxdna-debug-capture.sh baseline /tmp/claude-1000/amdxdna-baseline
```

The helper resolves the debugfs BDF automatically, runs the privileged
half in a single pkexec, and chowns the output back to the calling
user. Manual equivalent for reference:

```bash
DBG=/sys/kernel/debug/accel/0000:c6:00.1
B=/tmp/claude-1000/amdxdna-baseline-$(date +%Y%m%d-%H%M%S) && mkdir -p $B
cat /sys/class/accel/accel0/device/{vbnv,device_type,fw_version} > $B/userland.txt
pkexec sh -c "cp $DBG/{tdr_control,ctx_rq,ringbuf} $B/
              dd if=$DBG/dump_fw_log_buffer of=$B/fw_log.bin status=none
              dd if=$DBG/dump_fw_trace_buffer of=$B/fw_trace.bin status=none
              chown -R $USER:$USER $B"
```

Then run **just** `debug_halt_probe` with the trace daemon already
running:

```bash
./scripts/emu-bridge-test.sh --with-debug-halt-probe -v debug_halt_probe
```

Wait for the cascade (the script will mark TDR and continue trying;
the second submission will hang for ~7.7s). Then, **before any
recovery action**, capture forensics:

```bash
tools/amdxdna-debug-capture.sh wedged /tmp/claude-1000/amdxdna-wedged
```

The helper runs the privileged half in a single pkexec: it copies
tdr_control / ctx_rq / ringbuf, dumps FW log+trace buffers, runs the
mailbox liveness probe (`echo 1 > nputest`), captures get_app_health
with a 5s timeout (so a dead mailbox doesn't hang the script), and
chowns everything back. dmesg captures (pre-probe + nputest verdict)
land in the same output dir.

Then try recovery experiments, in order. Test with a known-good kernel
after each — `add_one_objFifo/peano` is the canonical "object-fifo
shape" we use to confirm the device services real submissions:

```bash
# (a) Driver-level recover (does what TDR did internally)
DBG=/sys/kernel/debug/accel/0000:c6:00.1
pkexec sh -c "echo recover > $DBG/tdr_control"
XDNA_EMU= ./mlir-aie/build/test/npu-xrt/add_one_objFifo/peano/test.exe

# (b) FW reload — the last hammer before reboot
pkexec sh -c 'echo 1 > /sys/module/amdxdna/parameters/fw_reload'
XDNA_EMU= ./mlir-aie/build/test/npu-xrt/add_one_objFifo/peano/test.exe
```

## Open questions the capture is meant to answer

1. **What does dmesg show during the TDR sequence?** We've never read
   it. The expected sequence is "Device isn't making progress" ...
   "TDR start" ... `aie2_dump_ctx` output ... `aie2_rq_stop_all`
   completion ... `aie2_rq_restart_all` completion. If the sequence
   stops early, we know which stage hung.
2. **Is the mailbox alive after TDR?** `nputest` answers this directly.
   If yes, the wedge is downstream of mailbox (probably context
   scheduler state). If no, the wedge is firmware-level.
3. **Did the firmware reload?** `fw_version` before/after compares
   trivially. If it changed, FW did reload but didn't reattach; if it
   didn't change, recovery is purely a driver/scheduler concern.
4. **What does `tdr_control` report after the cascade?** Especially
   `status`, `progress`, `counter`. If `status` shows "recovered"
   while user-space still times out, the driver thinks recovery is
   complete but something downstream (XRT? our plugin? our context
   state?) isn't aware.
5. **Does `echo recover > tdr_control` un-stick the device?** If yes,
   the original recovery had a bug we can attribute to a specific
   debug-halt-probe-related code path; if no, recovery itself is
   broken irrespective of trigger.
6. **Does `fw_reload` un-stick it?** Last-resort hammer; if this works,
   we have a recovery primitive that avoids reboot.
7. **Module parameter sweep**: does `mailbox_polling=1` (poll instead
   of IRQ) survive the wedge any differently?

## Workaround currently in place

`scripts/emu-bridge-test.sh` skips `debug_halt_probe` by default
(commit a2b0360). Opt back in with `--with-debug-halt-probe` when
deliberately validating TDR-recovery changes on a known-good
kernel/driver state. This unblocks the sweep but does nothing to fix
the underlying problem.

## What success looks like

We can answer at least Q1-Q5 above with captured forensic data,
locate the failing step in `aie2_tdr_force_recover()` (or its callees
in `aie2_ctx_runqueue.c`), and either land a workaround in our test
infrastructure or report a reproducible bug upstream to AMD. If
`fw_reload` works, ideally bake it into a post-TDR recovery script so
the dev box doesn't need a full reboot for every cascade.
