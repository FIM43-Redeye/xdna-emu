# amdxdna TDR recovery incomplete on Phoenix dev box

**Status:** Investigated 2026-05-20 (after the first reboot). Root cause
located, all userspace recovery primitives proven non-functional on
Phoenix, reboot confirmed as the only recovery. See "Investigation
results" section below. Forensics preserved at
`build/experiments/2026-05-20-amdxdna-tdr-forensics/`.

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

## Investigation results (2026-05-20, post-reboot)

Triggered the wedge by running `debug_halt_probe/peano/test.exe`
directly (bypassing the bridge harness to keep the pkexec count
minimal). Captured full forensic state at
`build/experiments/2026-05-20-amdxdna-tdr-forensics/`.

### Q1 — dmesg TDR sequence (answered, but corrects the doc)

The single-context recovery path is **NOT** `aie2_rq_stop_all` +
`aie2_rq_restart_all`. Actual sequence (full trace in
`wedged-post-debug-halt-probe/dmesg-pre-probe.txt`):

```
T+0.000  Job submitted on user mailbox 136, opcode 0x18, msg_id 0x1d000001
T+1.545  Poll: ctx.88452.1 @[1, 1] submitted 1 completed 0
T+3.594  Poll: same
T+5.641  Poll: same
T+5.641  aie2_tdr_work: Device isn't making progress... Count 1 timeout 2 dump_only 0
T+5.641  aie2_dump_ctx: Dumping ctx (sub=1, comp=0)
T+5.641  Get app health unsupported for the device or firmware version
T+5.641  ctx.88452.1 @[1, 1] stop          ← per-ctx stop (not stop_all)
T+5.641  xdna_mailbox.136: IRQ disabled and RX work cancelled
T+5.641  Send destroy on MGMT mailbox 145 (opcode 0x3)
T+10.83  xdna_send_msg_wait: Wait for completion timeout       ← MGMT mailbox dead
T+10.83  aie2_destroy_context: destroy context failed, ret -62
T+10.83  Driver continues teardown anyway; releases mailbox 136
T+10.83  User-space process tears down BOs and exits
```

Notable: no `aie2_rq_stop_all` or `aie2_rq_restart_all` ever fires.
The per-ctx destroy path runs first and hangs on MGMT mailbox; the
escalation to rq_stop_all/restart_all (which the original doc and
earlier reading of `aie2_tdr_force_recover()` predicted) does not
happen. Either the driver path is different on this version or the
escalation requires the per-ctx destroy to succeed first.

### Q2 — mailbox alive after TDR? (NO, both channels dead)

Direct evidence from `wedged-post-debug-halt-probe/`:

- `nputest` probe: `aie2_check_protocol_version: Failed to get protocol version, ret -62` (`-ETIME`)
- `get_app_health`: timed out (also: driver itself logs `Get app health unsupported for the device or firmware version` during TDR dump on this FW)
- TDR's own destroy command on MGMT mailbox 145: timed out
- All subsequent submissions fail at `DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed (err=-5): Input/output error`

Both mailbox channels are dead. The doc's earlier framing ("MGMT
mailbox might be alive even if user-context mailbox is dead") does
not hold here — they go down together because the firmware is hung
at a level below per-channel state.

Caveat for future captures: `get_app_health` is not a reliable
post-TDR liveness probe on Phoenix because the FW (1.5.5.391) doesn't
support that mailbox opcode in normal operation either. The
authoritative liveness probe is `echo 1 > nputest` → check dmesg for
`NPU health check failed: ret=-62` (dead) vs success.

### Q3 — did the firmware reload? (NO)

`fw_version` unchanged at `1.5.5.391` throughout. No recovery action
ever caused a FW reload.

### Q4 — tdr_control state after the cascade

`counter=1, status=wait, progress=0`. **The driver's TDR state
machine thinks the recovery cycle completed successfully.** There is
no machine-readable indication that the recovery action's
mailbox-level operations actually failed. The kernel-side bookkeeping
is decoupled from FW-side actual recovery — a structural blind spot.

After "successful" TDR, `aie2_tdr_detect` returns false on subsequent
polls (no contexts → none stuck → no need to recover again), so TDR
doesn't fire repeatedly.

### SMU response code interpretation

The dmesg line `[drm] *ERROR* smu cmd 4 failed, 0xff` comes from
`aie_smu_exec()` in `xdna-driver/src/driver/amdxdna/aie_smu.c:46-48`.
The xdna driver collapses any non-OK SMU response to `-EINVAL`, so
the returned errno is uninformative -- but the raw 0xff in the
dmesg line carries the SMU's actual response code, which uses
the canonical AMD PPSMC convention (also documented in the modern
amdgpu driver at `drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c:77-83`):

| Code | Constant | Meaning |
|------|----------|---------|
| `0x00` | `SMU_RESP_NONE` | SMU never wrote a response (poll timed out -- SMU itself dead) |
| `0x01` | `SMU_RESP_OK` | Success |
| `0xFB` | `SMU_RESP_DEBUG_END` | Debug command terminus |
| `0xFC` | `SMU_RESP_BUSY_OTHER` | SMU busy with another command (transient -- retry may help) |
| `0xFD` | `SMU_RESP_CMD_BAD_PREREQ` | Prerequisites not met (state machine in wrong state) |
| `0xFE` | `SMU_RESP_CMD_UNKNOWN` | SMU doesn't recognize the command (FW/SMU version mismatch) |
| `0xFF` | `SMU_RESP_CMD_FAIL` | Generic "I tried and failed" |

**`0xFF` is significant**: it means the SMU acknowledged the command
(not `0xFE`) and tried to execute (not `0x00`), but execution failed.
**The MP1/SMU controller is alive and responsive.** What's broken is
the underlying action -- power-off of an NPU whose internal state
(managed by the hung FW) won't allow a clean power cycle. This
refines the earlier "SMU wedged" framing: the SMU itself isn't
wedged, but it can't unstick the wider NPU subsystem because the FW
that mediates the NPU's internal state machines is hung.

This also explains why system-level power cycling (suspend or
reboot) works while modprobe -r/modprobe cannot: suspend/reboot
drops the whole SoC to retention voltage, taking the NPU's internal
state machines down with it; modprobe-driven SMU POWER_OFF respects
NPU-internal sequencing that requires FW cooperation.

For future captures, **always read the raw `0x%x` from the dmesg
`smu cmd N failed` line** -- it carries diagnostic information that
the kernel-return errno does not.

### Q5 — do the userspace recovery primitives work? (NO, none of them)

| Primitive | Result | Why |
|-----------|--------|-----|
| `echo recover > tdr_control` | NO-OP | Counter doesn't increment; no contexts to operate on (already destroyed by initial TDR). Subsequent submission still EIO. |
| `echo 1 > fw_reload` | NO-OP | **`fw_reload` is in `aie4_pci.c` only — it does not exist in the aie2 path.** The module param accepts the write (it's a global), but the aie2 driver never reads it. Effectively a no-op on Phoenix. |
| `modprobe -r amdxdna && modprobe amdxdna` | **PARTIAL** | modprobe -r completed without hanging in `synchronize_srcu` (good — the documented risk did not materialize here). modprobe loaded the kernel module successfully. **But the device probe failed at `aie2_smu_start: Access power failed, ret -22` (`smu cmd 4 failed, 0xff`) — the on-NPU SMU controller is wedged.** Per CLAUDE.md the SMU lives downstream of the PCIe reset domain, so SBR wouldn't fix it either. |

### Additional findings

**A. The `synchronize_srcu` modprobe -r wedge risk is failure-mode-specific.**
The CLAUDE.md operational note about modprobe -r wedging in
`drm_dev_unplug -> synchronize_srcu` did NOT materialize for the
MGMT-mailbox-death-via-debug-halt-probe failure mode. It may be
specific to AIE_RW_ACCESS memtile-read timeout poisoning (where the
user-context mailbox dies mid-submission with outstanding work
holding the rq lock). When TDR has run to "completion" first
(bookkeeping-wise) and released the user mailbox cleanly, modprobe -r
appears safe. This is a useful distinction worth carrying forward.

**B. `fw_reload` is AIE4-only — remove from Phoenix recovery
suggestions.** `aie4_pci.c:44` defines `fw_reload` as a static
int module param used in `aie4_fw_reload()`. There is no equivalent
in `aie2_pci.c`. The CLAUDE.md operational notes and this doc's
earlier "diagnostic surface" section should be updated to reflect
that on Phoenix, `fw_reload` cannot recover anything.

**C. Both mailbox channels (user 136, MGMT 145) die together** when
debug_halt_probe wedges the FW. They are not independent — once the
FW is hung at the level the probe targets, all mailbox traffic stops
regardless of channel type.

**D. The full failure mode for debug_halt_probe on Phoenix:**
user submission → FW silent on user mailbox → TDR detects after 2s →
TDR's per-ctx destroy times out on MGMT mailbox → driver bookkeeping
marks ctx destroyed; FW is actually stuck → SMU also wedged → no
PCIe-layer recovery possible. Reboot is the only path.

### Open questions still standing

- **Q1 sub-question: when DOES `aie2_rq_stop_all` / `aie2_rq_restart_all`
  fire?** The code exists; presumably escalates from per-ctx stop when
  multiple contexts are stuck, or on a second TDR cycle. We only have
  one-ctx single-cycle evidence so far.
- **Q7 (mailbox_polling=1)**: Not exercised. Per "B" above this is
  unlikely to help (polling can't get a response from dead FW), but
  worth one cheap data point on a future investigation.
- **Is there a way to provoke a real FW reload without rebooting?**
  Suspend/resume drops the SoC to retention voltage and clears
  on-NPU controller state (per CLAUDE.md) — but S3 suspend is broken
  on this devbox (post-resume frequency issues). On other Phoenix
  systems where S3 works, that may be the recovery primitive that
  avoids reboot.

### Status

Findings documented. No code changes proposed yet — the upstream
amdxdna driver has structural limitations that we don't have leverage
to fix locally (SMU recovery requires either FW changes or a hardware
reset domain that includes the SMU). The actionable items from this
investigation are:

1. CLAUDE.md operational notes update: clarify that `fw_reload` is
   AIE4-only and shouldn't be in the Phoenix recovery escalation
   chain.
2. CLAUDE.md note that the `synchronize_srcu` modprobe -r risk
   appears to be specific to AIE_RW_ACCESS-style poisoning, not
   generic to any wedge.
3. Future: test `mailbox_polling=1` (cheap experiment, possibly worth
   one more debug_halt_probe cycle on a future session).
4. Future: investigate whether AMD has a Phoenix FW that supports the
   `app_health` opcode (would give us a better post-TDR liveness
   probe than `nputest`).
5. Future: patch experiment described in next section ("Driver
   workaround candidate").

## Direct SMU probing (2026-05-20, deeper dig)

After the recovery-experiment phase concluded, we directly probed the
NPU SMU via MMIO writes to `/sys/bus/pci/devices/<bdf>/resource0`
(`BAR0`) to see if there were commands beyond what xdna-driver wraps.
Tool: `tools/smu-probe.py`.  Goal: find an SMU command that could
unstick the NPU without rebooting.

### Method

The NPU1 (Phoenix) SMU registers are at fixed offsets within `BAR0`:

| Register | BAR0 offset | Source (driver) |
|----------|-------------|-----------------|
| SMU_CMD  | `0x100AC`   | `MPNPU_PUB_SCRATCH5` (`npu1_regs.c:112`) |
| SMU_RESP | `0x100B0`   | `MPNPU_PUB_SCRATCH6` |
| SMU_ARG  | `0x100B4`   | `MPNPU_PUB_SCRATCH7` (also SMU_OUT) |
| SMU_INTR | `0x10094`   | `MPNPU_PUB_PWRMGMT_INTR` |

We mirror `aie_smu_exec()`: clear RESP, write ARG, write CMD, kick
INTR (0 then 1), poll RESP.  `tools/smu-probe.py` does this and
decodes the response code.

The SMU lives in the NPU's private register aperture
(`MPNPU_APERTURE0_BASE`), not in the MP1 shared with the GPU.  So
undocumented commands here can only affect the NPU subsystem -- blast
radius bounded to "NPU more wedged", which is where we already are.
This made the probing safe to attempt on the wedged device.

### Findings: full SMU command map (Phoenix NPU SMU FW 76.101.0)

| Cmd | Behavior on wedged NPU                  | Identification |
|-----|-----------------------------------------|----------------|
| 0x1 | OK, `out = arg + 1` (verified, 5 args)  | `TestMessage` (canonical AMD PPSMC). The SMU is alive and bidirectional. |
| 0x2 | OK, `out = 0x004C6500` (arg-insensitive) | `GetSmuVersion`.  Decodes as major=`0x4C`=76, minor=`0x65`=101, patch=`0x00`, i.e. **SMU firmware version 76.101.0**. |
| 0x3 | OK, `out = 1` (arg-insensitive)         | `POWER_ON` (driver-known).  Works even on the wedged device. |
| 0x4 | CMD_FAIL, `out = arg` (echo)            | `POWER_OFF` (driver-known).  Persistently fails with `0xFF` regardless of arg or recent POWER_ON.  See "Structural recovery limitation" below. |
| 0x5 | OK, `out = 0x190` (clamped to 400 MHz)  | `SET_MPNPUCLK_FREQ` (driver-known).  Still works on wedged device.  Confirms SMU has live interaction with NPU clock infrastructure. |
| 0x6 | OK, `out = 0x320` (clamped to 800 MHz)  | `SET_HCLK_FREQ` (driver-known).  Same. |
| 0x7, 0x8 | **UNKNOWN_CMD** (`0xFE`)             | `SET_SOFT_DPMLEVEL` / `SET_HARD_DPMLEVEL` -- driver-defined in `aie_common.h` but **not implemented on Phoenix's NPU SMU**.  These are used only by `npu4_set_dpm()` (Strix path).  On NPU1 they're dead code. |
| 0x0, 0x9, 0xA–0x20 (gaps) | UNKNOWN_CMD              | Nothing else exists in the probed range.  The NPU SMU's command set is exactly **6 commands (0x1–0x6)**, narrower than typical MP1 vocabularies. |

Probing UNKNOWN_CMD codes is completely safe: they're rejected by the
SMU in the same ~20ms as documented commands, with no side effects on
register state beyond the SMU_CMD/SMU_RESP echo.

### Structural recovery limitation

The POWER_OFF asymmetry is the actionable finding.  POWER_ON
succeeds on a wedged device; POWER_OFF persistently fails.  The
likely explanation is a design asymmetry:

- POWER_ON doesn't need FW cooperation -- FW isn't running yet at
  POWER_ON time, so the SMU just programs hardware power gates
  directly and returns OK.
- POWER_OFF *does* need FW cooperation -- the SMU presumably waits
  for a FW shutdown handshake (clean-state save, clocks ramp-down
  acknowledgement) before reporting OK.  If FW is hung, this
  handshake never arrives and the SMU's internal timer fires CMD_FAIL.

This is the root cause of the modprobe-reload recovery failure.
`aie2_smu_start()` in `xdna-driver/src/driver/amdxdna/aie2_smu.c:134-156`
does **POWER_OFF first** as a defensive "ensure we're in a clean OFF
state before powering on":

```c
int aie2_smu_start(struct amdxdna_dev_hdl *ndev)
{
    /*
     * If the hardware was not powered off properly, try to set
     * power off. Failing to power off indicates an unrecoverable
     * issue, return failure.
     */
    ret = aie2_smu_set_power_off(ndev);   // <-- THIS FAILS (FW hung)
    if (ret) {
        XDNA_ERR(ndev->xdna, "Access power failed, ret %d", ret);
        return ret;                       // <-- modprobe init aborts here
    }
    ret = aie2_smu_set_power_on(ndev);    // <-- never reached
    ...
}
```

So the driver can never reach POWER_ON during re-probe because its
defensive POWER_OFF fails first.  But **POWER_ON works on its own**.

### Driver workaround candidate (untested)

If `aie2_smu_start()` skipped the defensive POWER_OFF (or treated its
failure as non-fatal) on the wedged-device path, re-probe might
complete successfully:

```c
int aie2_smu_start(struct amdxdna_dev_hdl *ndev)
{
    int ret;

    ret = aie2_smu_set_power_off(ndev);
    if (ret) {
        XDNA_WARN(ndev->xdna, "defensive POWER_OFF failed (NPU may be wedged), "
                              "proceeding to POWER_ON anyway");
        /* fall through to POWER_ON */
    }

    ret = aie2_smu_set_power_on(ndev);
    if (ret) {
        XDNA_ERR(ndev->xdna, "Power on failed, ret %d", ret);
        return ret;
    }
    return 0;
}
```

This is **untested** (would require DKMS rebuild + reload to verify on
a wedged device).  Risks of the patch:

- If the NPU is in some intermediate power state between ON and OFF,
  calling POWER_ON without the defensive POWER_OFF first might leave
  hardware in an inconsistent state.  Whether the SMU's POWER_ON
  internally handles this case is unknown.
- May surface different bugs further into init (e.g.,
  `aie2_pm_init()` or `aie2_mgmt_fw_init()` could fail differently if
  hardware state wasn't fully reset).

If the patch works, it could replace reboot as the recovery primitive
for FW-mailbox-dead wedges on Phoenix.  That would close the loop on
this finding: a software fix avoiding the reboot requirement.

### Tool reference

`tools/smu-probe.py` -- reads or executes commands on the Phoenix NPU
SMU via direct MMIO.  Modes:

- `read`: snapshot current SMU and PSP register state (read-only,
  always safe).
- `exec CMD [ARG] [--yes-destructive]`: send a command, wait up to
  1s for the SMU's response, decode it.  Refuses to send command
  numbers outside the safe set (0x1, 0x2, 0x3-0x8) without
  `--yes-destructive` opt-in.

Requires root via `pkexec`.  Default targets BDF `0000:c6:00.1`;
override with `--bdf`.  Don't run `exec` mode on a healthy device --
some command numbers are documented but most aren't, and writing an
undocumented command number is undefined behavior.
