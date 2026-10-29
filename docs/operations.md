# Operational Runbook

Durable operational procedures for working in this repo: build discipline,
toolchain discovery, formatting enforcement, test-suite economics, hardware
testing rules, and NPU recovery escalation.

AGENTS.md carries a compact quick-reference of the must-know-always rules and
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

## Toolchain discovery

Required derivation inputs use one provider-neutral resolver. Its per-component
precedence is:

| Component | Resolution order |
|-----------|------------------|
| mlir-aie | `MLIR_AIE_PATH`, `MLIR_AIE_DIR`, `NPU_WORK_DIR/mlir-aie`, ancestor discovery |
| llvm-aie | `LLVM_AIE_PATH`, `LLVM_AIE_DIR`, `NPU_WORK_DIR/llvm-aie`, ancestor discovery |
| aie-rt | `AIE_RT_PATH`, `NPU_WORK_DIR/aie-rt/driver/src`, ancestor discovery |

An explicit variable is authoritative. Blank values, missing directories, or
missing sentinels are errors; the resolver never hides a bad override by
falling through. Automatic discovery instead continues upward past incomplete
shadow directories. Successful paths are canonical and absolute.

The required sentinels are:

- mlir-aie:
  `lib/Dialect/AIE/Util/aie_registers_aie2.json`
- llvm-aie: `llvm/lib/Target/AIE/AIE2.td` and executable
  `build/bin/llvm-config`
- aie-rt: `global/xaiemlgbl_reginit.c` beneath the selected `driver/src`

Activation scripts may still populate these variables and Python paths, but a
normal build does not depend on activation. To exercise the worktree-independent
contract directly:

```bash
env -u NPU_WORK_DIR \
  -u MLIR_AIE_PATH -u MLIR_AIE_DIR \
  -u LLVM_AIE_PATH -u LLVM_AIE_DIR \
  -u AIE_RT_PATH -u PYTHONPATH \
  -u TABLEGEN_210_PREFIX \
  cargo test -p xdna-archspec --lib coverage::
```

The LLVM used to compile the Rust `tblgen-rs` dependency is a host dependency,
separate from llvm-aie/Peano. It defaults to system LLVM 21 on `PATH`;
`TABLEGEN_210_PREFIX` is an optional explicit host-LLVM override. The resolved
llvm-aie root supplies AIE `.td` sources and matching decoder libraries.

## Code formatting

The repo has a tuned `rustfmt.toml` at the root. Stable rustfmt only;
non-default settings preserve the project's semantic conventions
(import groups, compact struct literals, ~110-char lines, in-line method
chains).

The codebase is **fully fmt'd** as of the Path B convergence sprint
(2026-04-29). New code stays clean via two enforcement layers:

1. **Format-on-save** in your editor (or agent harness) for edits as they
   happen. Format edited files in stdin mode (`rustfmt --emit stdout <
   file`) so lib.rs/mod.rs edits don't reformat the whole module tree.
2. **Pre-commit hook** at `scripts/git-hooks/pre-commit`. Runs
   `cargo fmt --check` on commits that touch `*.rs`. Blocks the
   commit on drift with an actionable message.
3. **CI check** (TODO -- no CI workflow yet): once added, run
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
   sudo sh -c 'modprobe -r amdxdna && modprobe amdxdna'
   ```
   **Caveat: not safe in a poisoned-mailbox state.** If the user-context
   mailbox has been killed by a Phoenix `AIE_RW_ACCESS` timeout caused by the
   legacy/current wire-layout mismatch (or an equivalent firmware-level hang),
   `modprobe -r` itself wedges
   uninterruptibly in `drm_dev_unplug -> synchronize_srcu` and reboot
   becomes the only recovery path.  We previously set `tdr_dump_ctx=1`
   (the `src/driver`-tree TDR knob) to disable TDR recovery and avoid
   that poisoning, but that turned every firmware silent-drop into a
   permanent `aie2_hmm_invalidate` wedge requiring reboot anyway (see
   `xdna-emu-priv: research/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`).
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
   `xdna-emu-priv: research/findings/2026-05-22-ctrl-packet-wedge-drivers-accel.md`.

   **2026-05-20 refinement** (see
   `xdna-emu-priv: research/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`):
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
   `xdna-emu-priv: research/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`):

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
   to `/sys/bus/pci/devices/<bdf>/resource0`, requires root.

2. **Bridge PM-cycle** -- reset the upstream bridge function:
   ```bash
   sudo modprobe -r amdxdna
   sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:08.2/reset'
   sleep 5
   sudo modprobe amdxdna
   ```
   Note: this is what the kernel calls "reset" on the bridge; per the
   bridge's `reset_method = pm`, it's a D0->D3hot->D0 cycle of the bridge
   function, NOT a true Secondary Bus Reset on bus c6.

3. **True Secondary Bus Reset** -- toggle BCR.SBR on the bridge:
   ```bash
   sudo modprobe -r amdxdna
   sudo setpci -s 00:08.2 BRIDGE_CONTROL=0x42  # assert SBR
   sleep 0.1
   sudo setpci -s 00:08.2 BRIDGE_CONTROL=0x02  # deassert
   sudo sh -c 'echo 1 > /sys/bus/pci/rescan'
   sudo modprobe amdxdna
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

**Never put tools, scripts, or persistent work products in `/tmp`**,
which a reboot wipes. Tools
live under `xdna-emu/tools/` or `xdna-emu/scripts/`. Experiment results
go under `xdna-emu/build/experiments/`.
Only truly ephemeral data (log tees for a specific run, throwaway
scratch) should use `/tmp`.
