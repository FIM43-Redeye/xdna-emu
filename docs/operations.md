# Operational Runbook

Durable operational procedures for working in this repo and on this devbox:
build discipline, formatting enforcement, test-suite economics, hardware
testing rules, NPU recovery escalation, and the current machine's environment
state.

CLAUDE.md carries a compact quick-reference of the must-know-always rules and
points here for the full procedures. When the NPU wedges, you are setting up a
fresh checkout, or you hit a stale-`.so` phantom bug, this is the file.

## Build discipline

**Rebuild before testing.** The XRT plugin loads `libxdna_emu.so` at
runtime. `cargo test --lib` does NOT trigger a plugin rebuild, and
ISA/bridge tests load whatever `.so` is on disk. After any Rust source
change, run `cargo build` (and `cargo build --release` if release is
being exercised). Stale `.so`s have produced phantom bugs, including a
memorable "concurrency bug" that was really just a stale debug lib.

**Profile clarification.** The rule "one build at a time" means one
invocation of a given target, not one build overall. `cargo build` and
`cargo build --release` can run concurrently -- cargo handles the
locking between them. Don't run the same command twice concurrently.

## Code formatting

The repo has a tuned `rustfmt.toml` at the root. Stable rustfmt only;
non-default settings preserve the project's semantic conventions
(import groups, compact struct literals, ~110-char lines, in-line method
chains).

The codebase is **fully fmt'd** as of the Path B convergence sprint
(2026-04-29). New code stays clean via two enforcement layers:

1. **`PostToolUse` hook** in `.claude/settings.json` auto-runs rustfmt
   on any `.rs` file Claude edits (via `.claude/hooks/rustfmt-edited.sh`,
   which uses stdin mode to avoid module recursion).
2. **Editor format-on-save** handles the same role for human edits.
3. **Pre-commit hook** at `scripts/git-hooks/pre-commit`. Runs
   `cargo fmt --check` on commits that touch `*.rs`. Blocks the
   commit on drift with an actionable message.
4. **CI check** (TODO -- no CI workflow yet): once added, run
   `cargo fmt --check` as a required step.

**One-time local setup per checkout (two configs):**

```bash
git config core.hooksPath scripts/git-hooks
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

The first activates the pre-commit hook. The second makes `git blame`
skip the chunked `cargo fmt` commits listed in
`.git-blame-ignore-revs` and attribute lines to the original content
author. GitHub honors the blame-revs file automatically; no per-user
setup there.

**Don't run `cargo fmt` repo-wide in one commit.** Even now that
everything's clean, future bulk reformats (e.g., from a config tweak)
should stay chunked by subtree -- each commit reviewable, each SHA
appended to `.git-blame-ignore-revs`.

## Test suite costs

- `./scripts/isa-test.sh`: ~5-10 minutes (release build + 123 HW batches)
- `./scripts/emu-bridge-test.sh`: ~15-30 minutes (dual-compiler, HW + EMU)

These are expensive; don't re-run them just to "check progress." Run
once after a batch of fixes and examine results. For targeted re-runs,
use filter arguments or single-test invocations.

**Use tee for long runs.** When backgrounding a test, pipe through tee
so output is both live-monitorable and logged:

```bash
./scripts/isa-test.sh 2>&1 | tee /tmp/isa-test.log
```

A bare redirect hides progress from both sides. (`/tmp` is fine here --
the log is ephemeral.)

## Hardware testing

**Never run two hardware test suites concurrently.** Bridge tests and
ISA tests both target the NPU device; running them in parallel causes
them to fight for the device and both must be killed. Run hardware
suites sequentially. Pure `cargo test --lib` unit tests are safe to run
alongside since they don't touch hardware.

## NPU recovery

When the NPU wedges, recovery escalates through:

1. **Driver reload** -- handles most TDR recoveries:
   ```bash
   pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'
   ```
   **Caveat: not safe in a poisoned-mailbox state.** If the user-context
   mailbox has been killed by an `AIE_RW_ACCESS` memtile-read timeout
   (or an equivalent firmware-level hang), `modprobe -r` itself wedges
   uninterruptibly in `drm_dev_unplug -> synchronize_srcu` and reboot
   becomes the only recovery path.  We previously set `tdr_dump_ctx=1`
   (the `src/driver`-tree TDR knob) to disable TDR recovery and avoid
   that poisoning, but that turned every firmware silent-drop into a
   permanent `aie2_hmm_invalidate` wedge requiring reboot anyway (see
   `docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`).
   We removed it on 2026-05-13: TDR now actually recovers, so
   `modprobe -r` is again attempted-first when the NPU wedges.

   On the canonical `drivers/accel` tree (now the loaded module) the
   TDR knob is `tdr_dump_only` (bool, default `N` = recovery active);
   there is no `tdr_dump_ctx`.  A ctrl_packet sweep reproduced a wedge
   on `drivers/accel` on 2026-05-22 -- TDR fired but recovery was
   incomplete (mgmt mailbox left rejecting CREATE/DESTROY context).
   `modprobe -r`/reload still recovered the device cleanly in ~3s (no
   `synchronize_srcu` hang) because the HW runner had *exited* -- no
   process pinning `/dev/accel/accel0` -- and the mgmt mailbox stayed
   responsive.  See
   `docs/superpowers/findings/2026-05-22-ctrl-packet-wedge-drivers-accel.md`.

   **2026-05-20 refinement** (see
   `docs/superpowers/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`):
   The `synchronize_srcu` wedge risk appears to be **failure-mode-specific**
   to AIE_RW_ACCESS-style poisoning, where the user mailbox dies
   mid-submission with outstanding work holding the rq lock.  For the
   debug_halt_probe failure mode (TDR runs to "completion"
   bookkeeping-wise and releases the user mailbox cleanly, but FW and
   MGMT mailbox are dead), `modprobe -r` completed in ~10s without
   hanging.  modprobe re-probe still fails -- the dmesg `smu cmd 4
   failed, 0xff` is `SMU_RESP_CMD_FAIL` from the PPSMC convention,
   meaning the SMU itself is alive and responding but cannot complete
   POWER_OFF on an NPU whose FW-managed internal state machines are
   hung.  So reboot is still the only recovery, but at least the
   reload attempt itself is safe to try.  Also: **`fw_reload` is
   AIE4-only** (`aie4_pci.c:44`); on Phoenix the module param is
   accepted but does nothing.  Don't include it in the Phoenix
   recovery escalation chain.

   The SMU response codes follow PPSMC convention (see also
   amdgpu's `drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c`): `0x01`=OK,
   `0xFC`=busy, `0xFD`=bad prereq, `0xFE`=unknown command,
   `0xFF`=generic fail, `0x00`=no response (SMU dead).  Always read
   the raw `0x%x` from the dmesg line; the kernel-return `-EINVAL`
   from `aie_smu_exec` collapses all non-OK codes to one value.

   **Phoenix NPU SMU command map** (post-probe 2026-05-20, SMU FW
   76.101.0; full discussion in
   `docs/superpowers/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`):

   | Cmd | Name | Notes |
   |-----|------|-------|
   | 0x1 | `TestMessage` | Returns `arg+1`.  Use as SMU heartbeat. |
   | 0x2 | `GetSmuVersion` | Returns packed version `0x004C6500` = `76.101.0`. |
   | 0x3 | `POWER_ON` | Works even on wedged device (no FW handshake needed). |
   | 0x4 | `POWER_OFF` | Requires FW cooperation; fails CMD_FAIL when FW hung. |
   | 0x5 | `SET_MPNPUCLK_FREQ` | Returns clamped value. |
   | 0x6 | `SET_HCLK_FREQ` | Returns clamped value. |
   | 0x7, 0x8 | `SET_SOFT/HARD_DPMLEVEL` | Defined in `aie_common.h` but **not implemented on Phoenix**; Strix-only. |

   Nothing else exists in the 0x0-0x20 range (probed and confirmed
   UNKNOWN_CMD).  The POWER_OFF asymmetry explains why driver reload
   fails on a wedged device: `aie2_smu_start()` runs POWER_OFF first
   as a defensive cleanup; it fails CMD_FAIL because the hung FW
   can't acknowledge the shutdown handshake, and the function returns
   early before reaching POWER_ON (which would succeed).  A candidate
   driver patch that demotes the defensive POWER_OFF failure to a
   warning and proceeds to POWER_ON is described in the finding doc
   -- untested.

   Tool: `tools/smu-probe.py read | exec CMD [ARG]` -- direct MMIO
   to `/sys/bus/pci/devices/<bdf>/resource0`, requires pkexec.

2. **Bridge PM-cycle** -- reset the upstream bridge function:
   ```bash
   pkexec modprobe -r amdxdna
   pkexec sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:08.2/reset'
   sleep 5
   pkexec modprobe amdxdna
   ```
   Note: this is what the kernel calls "reset" on the bridge; per the
   bridge's `reset_method = pm`, it's a D0->D3hot->D0 cycle of the bridge
   function, NOT a true Secondary Bus Reset on bus c6.

3. **True Secondary Bus Reset** -- toggle BCR.SBR on the bridge:
   ```bash
   pkexec modprobe -r amdxdna
   pkexec setpci -s 00:08.2 BRIDGE_CONTROL=0x42  # assert SBR
   sleep 0.1
   pkexec setpci -s 00:08.2 BRIDGE_CONTROL=0x02  # deassert
   pkexec sh -c 'echo 1 > /sys/bus/pci/rescan'
   pkexec modprobe amdxdna
   ```
   This actually pulses PERST# on bus c6 (resets c6:00.0 + c6:00.1).
   Useful when step 2 doesn't suffice. Confirmed working at the PCIe
   layer (BAR enable cycles), but does NOT recover SMU/MGMT_ERT-level
   wedges -- the on-NPU controllers live downstream of the PCIe reset
   domain.

4. **Suspend/resume** -- `systemctl suspend` drops the SoC to retention
   voltage, clearing on-NPU controller state. Required for SMU wedges
   from bring-up failures.

5. **Reboot** -- last resort.

FLR on the NPU function is advertised in DevCap (`FLReset+`) and listed
as the kernel's `reset_method = flr` for c6:00.1, but per upstream
xdna-driver feedback (May 2026) it's unreliable and unsupported in
practice. SBR on the upstream bridge is the working PCIe-layer path.
Always remove `amdxdna` before any PCIe reset -- the driver holds state
that corrupts the device on hot-reset. PCIe BDFs shift when hardware
changes (e.g., GPU swaps renumber the bus); `lspci | grep "IPU Device"`
finds the current BDF.

## Working-directory conventions

**Never put tools, scripts, or persistent work products in `/tmp`.**
This PC cannot suspend and reboots often, which wipes `/tmp`. Tools
live under `xdna-emu/tools/` or `xdna-emu/scripts/`. Experiment results
go under `xdna-emu/build/experiments/` or `~/npu-work/experiments/`.
Only truly ephemeral data (log tees for a specific run, throwaway
scratch) should use `/tmp`.

## Developer environment state

These describe the current machine's setup. Other contributors will
substitute their own values.

- **Kernel**: custom `7.1.2-custom+`. Out-of-tree `amdxdna` is
  managed by DKMS via `xrt-amdxdna/2.23.0`, source at
  `/usr/src/xrt-amdxdna-2.23.0/`.  Caveat: if the kernel is *rebuilt*
  in place (same `uname -r`, new build), `CONFIG_RANDSTRUCT`
  re-randomizes `struct module` and the DKMS module fails to load
  with `Exec format error` / a `.gnu.linkonce.this_module section
  size` mismatch.  DKMS does not auto-detect this -- force a rebuild:
  `pkexec sh -c 'dkms build xrt-amdxdna/2.23.0 -k $(uname -r) --force
  && dkms install xrt-amdxdna/2.23.0 -k $(uname -r) --force'` (plain
  `dkms install --force` re-copies the stale artifact -- it does not
  recompile).  Userspace plugin (the SHIM at
  `/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.*`) is delivered by the
  `xrt_plugin-amdxdna` .deb.  A bare `./build.sh -release` BUILDS
  both halves but INSTALLS neither; the `.deb` sits at
  `build/Release/xrt_plugin.*-amdxdna.deb` until `dpkg -i`'d.  Module
  is signed at install time with our MOK key
  (`/var/lib/shim-signed/mok/MOK.{priv,der}`), so `modprobe amdxdna`
  works after every kernel upgrade with no manual signing.  After
  editing driver or SHIM source, refresh both halves with:
  ```bash
  cd ~/npu-work/xdna-driver/build
  ./build.sh -release -refresh_dkms
  ```
  `-refresh_dkms` is our local addition (commit 3509b2a, simplified
  2026-05-12).  It `pkexec dpkg -i`s the OS-matched
  `build/Release/xrt_plugin.*_${VERSION_ID}-*.deb`; the .deb postinst
  does the rest -- runs `dkms_driver.sh --install` to populate
  `/usr/src/xrt-amdxdna-2.23.0/` from a single canonical source
  (script + tarball + config, NOT a piecemeal rsync that would miss
  `configure_kernel.sh` and produce feature-probe mismatches like
  `num_rqs == 0`), then `rmmod amdxdna && modprobe amdxdna` to swap
  the loaded module.  Single auth prompt, both halves of the install
  on disk, kernel module reloaded.  Stale SHIM bytes silently mask
  source changes -- always pass `-refresh_dkms` after touching
  driver OR SHIM code.

  Caveat: the .deb postinst uses `rmmod` which fails (and the install
  aborts) if the device is busy or the module is wedged.  In that case,
  free the device first (kill any process holding `/dev/accel/accel0`),
  or fall back to a manual `pkexec sh -c 'modprobe -r amdxdna; dpkg -i
  build/Release/xrt_plugin.*_${VERSION_ID}-*.deb'`.
- **amdxdna runtime autosuspend pinned off**: the NPU has been
  observed to wedge on auto-resume after certain firmware failure
  modes, so runtime autosuspend is disabled.  The canonical
  `drivers/accel` tree has *no* `autosuspend_ms` module parameter
  (that was a `src/driver`-tree knob) -- runtime PM is standard PCI
  runtime PM.  Autosuspend is pinned off via
  `/etc/udev/rules.d/71-amdxdna-no-autosuspend.rules`, which sets
  `power/control=on` on the NPU PCI function (`[1022:1502]`).
  Verify with `cat /sys/bus/pci/devices/<bdf>/power/control`
  (should report `on`).  Workaround for development; revert (rule
  to `auto`, or remove it) once the underlying wedges are fixed.

  The old `/etc/modprobe.d/amdxdna.conf` (`options amdxdna
  autosuspend_ms=-1`) was removed 2026-05-22 -- on `drivers/accel`
  the param does not exist and modprobe logged it as "unknown
  parameter ... ignored".  The loaded module's parameters are:
  `aie2_max_col`, `tdr_dump_only`, `tdr_timeout_ms`, `force_cmdlist`,
  `force_iova`.

  **History -- `tdr_dump_ctx` (removed 2026-05-13).** On the
  `src/driver` tree the TDR knob was `tdr_dump_ctx=1`, set so TDR
  was dump-only (no `aie2_rq_stop_all/restart_all`), avoiding the
  `synchronize_srcu` wedge in `modprobe -r` after the recover path
  poisoned the mailbox on Phoenix. But disabling TDR recovery made
  every firmware silent-drop (e.g. CHAIN_EXEC_NPU on ctrl_packet
  sweeps) into a permanent `aie2_hmm_invalidate` wedge -- the
  driver-design comment at `aie2_ctx.c:1017` explicitly relies on
  TDR to terminate the ctx if firmware doesn't respond, so without
  recovery the `dma_resv_wait_timeout(MAX_SCHEDULE_TIMEOUT)` in
  hmm_invalidate waits forever. We removed it 2026-05-13 -- letting
  TDR recover trades a permanent wedge for a possible (not certain)
  modprobe -r wedge. On `drivers/accel` the equivalent knob is
  `tdr_dump_only` (bool, default `N`), left at default (recovery
  active). See
  `docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`.
- **amdxdna driver + firmware logging**: driver-internal `pr_debug`
  output is enabled via `/etc/modprobe.d/amdxdna-verbose-debug.conf`
  (`options amdxdna dyndbg=+p`; persists across reloads). Firmware
  logging/tracing is NOT a module parameter on `drivers/accel` --
  the `src/driver`-era `fw_log_level` / `fw_trace_categories` /
  `fw_log_size` / `fw_trace_size` / `poll_fw_trace` params are gone.
  It is now the runtime "DPT" (firmware Debug/Profile/Trace)
  framework in `amdxdna_dpt.c`: `FW_LOG` auto-starts at `WARN`,
  `FW_TRACE` is opt-in, both driven by DRM ioctls
  (`SET_FW_LOG_STATE` / `SET_FW_TRACE_STATE`, read back via
  `GET_ARRAY` with `FW_LOG` / `FW_TRACE`) and surfaced through
  `xrt-smi`. Kernel tracepoints live under
  `/sys/kernel/debug/tracing/events/amdxdna/` -- the `amdxdna-trace`
  daemon rides `xdna_job` + the `mbox_*` set.
- **dmesg is unrestricted**: kernel built with `kernel.dmesg_restrict=0`
  (or equivalent), so `dmesg` works without `pkexec`. Don't wrap dmesg
  in pkexec on this machine.
- **Chess license**: `HOSTID=f4289d05121f` (bound to current Wi-Fi
  card; 2 of 3 vendor-permitted swaps remaining).
- **DNS**: UConn DNS is broken. Fix per-session:
  `resolvectl dns wlp5s0 8.8.8.8 8.8.4.4` (does not persist across
  reboot).
- **mlir-aie venv**: `/home/triple/npu-work/mlir-aie/ironenv/`
- **PYTHONPATH**: `/home/triple/npu-work/mlir-aie/install/python`
- **XRT plugin**: `./scripts/rebuild-plugin.sh` builds and installs the
  debug `.so` by default (`--release` for release). Activation:
  `XDNA_EMU=1` (any value) routes XRT to the emulator at `xrt::device(0)`.
  Profile: `XDNA_EMU_RUNTIME=release|debug` (default `debug`); the plugin
  picks the matching `.so` via `XDNA_EMU_DIR` or installed symlinks.
- **Trace column offset**: emulator col=0 vs HW col=start_col (cosmetic;
  trace tools should normalize).
