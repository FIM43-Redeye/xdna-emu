---
name: bug #6 memtile_dmas hang still state-dependent post num_rqs packaging fix 2026-05-12
description: After fixing the num_rqs=0 packaging bug on the rebased xdna-driver, the memtile_dmas/* tests still hang at run.wait() on first run after some reload conditions, with HW reporting "total completed jobs 1" but the host never seeing completion. Process is killable (no D-state, no TDR, no oops). State-dependent: clears after one or more module reload cycles. Bug #6 is unchanged by the rebase; the morning's "not reproducible" finding was premature -- it conflated the (now-fixed) num_rqs oops with the quieter underlying memtile-DMA-completion bug. Supersedes 2026-05-12-memtile-dmas-bug6-not-reproducible.md.
type: finding
---

# Bug #6 memtile_dmas hang -- still state-dependent on rebased driver

**Supersedes** [2026-05-12-memtile-dmas-bug6-not-reproducible.md](../../archive/findings/2026-05-12-memtile-dmas-bug6-not-reproducible.md).
That earlier doc's title and TL;DR are now misleading: the
"reproducible TDR" the morning sweep saw was actually the
*num_rqs=0 packaging oops*, not bug #6 proper.  Bug #6 is the
quieter, state-dependent memtile-DMA-completion hang documented
here.

## TL;DR

- The num_rqs=0 oops from this morning was a separate **packaging
  bug** in xdna-driver's DKMS flow (stale `configure_kernel.sh`
  in /usr/src).  Fixed via `build.sh -refresh_dkms`; see
  xdna-driver commit `3509b2a` and the matching xdna-emu CLAUDE.md
  update.
- With the num_rqs oops gone, the **underlying bug #6 memtile-DMA
  completion hang remains** in the rebased driver, with the same
  characterization as pre-rebase:
  - `memtile_dmas/*` tests hang at `run.wait()` after printing
    "Running Kernel."
  - dmesg shows `ctx.NNN.1 total completed jobs 1` -- driver thinks
    HW finished something.
  - Process is **killable** (no D-state, no TDR, no kernel oops).
  - `xrt-smi validate` + compute-tile DMA tests (e.g. `add_one_using_dma`)
    pass cleanly on the same driver state.
  - State-dependent: hangs on "first" run after some reload conditions,
    clears after one or more module reloads.
- Rebase did **not** fix bug #6.  Upstream's HMM revert (the morning's
  hopeful theory) has no observable effect on the hang behavior.
- Rebase did **not** introduce bug #6.  The hang pattern matches the
  pre-rebase finding from earlier today; the rebase is bug-neutral
  for this issue.

## Confirmed symptoms

After loading the fixed (post num_rqs) module on rebased driver,
2026-05-12 evening session:

| Test | First-run result | Subsequent runs (after one or more reloads) |
|---|---|---|
| `writebd` | HANG (rc=124, killed by 30s timeout) | PASS (5/5 in loop) |
| `writebd_tokens` | HANG (same pattern) | PASS |
| `xrt-smi validate` | PASS | PASS |
| `add_one_using_dma` | PASS | PASS |

The "first run" qualifier is fuzzy: same kernel binary
(srcversion `5CA2BD72D0B1431D02E62BF`) flipped from "hangs every
attempt" to "passes every attempt" across an intermediate module
reload cycle.  We have not yet pinned down the exact state delta.

## Ruled out

The morning + evening sessions ruled out the following as
*the* cause of bug #6 (each on its own, not collectively):

1. **num_rqs=0 oops** -- a separate packaging bug.  Fixed; bug #6
   persists.
2. **Our 3 uncommitted patches** (Makefile, aie2_message.c,
   hwctx.cpp) -- stashed/reapplied selectively in the evening
   session.  Reapplying just `aie2_message.c` (the mailbox -ETIME
   teardown removal, plausible memtile-related suspect) on the
   same binary produced the same `5CA2BD72...` srcversion and
   passing tests, so these patches do not gate bug #6 behavior.
3. **The rebase as cause** -- the morning's pre-rebase bridge
   sweep showed the same state-dependent hang behavior
   (eventually-passing after a reload), so bug #6 predates the
   rebase.
4. **HMM revert as fix** -- the morning's working theory was that
   upstream commit reverting `49a1ee4` (HMM invalidate common to
   AIE2/AIE4) would fix bug #6.  Refuted: rebase brings in the
   revert, bug #6 still present.

## What's unknown

These are the unanswered questions that block a real root-cause
diagnosis.  Pick any one for the next investigation:

1. **What is "total completed jobs 1" actually counting?**  Is it
   the user kernel job, an internal config_ctx setup job, a sync
   BO operation, or some firmware ack we don't recognize?  Driver
   code search starting from the log call site would answer this.
2. **Did the firmware actually run the user kernel, or just a
   preparation step?**  Without firmware logs we can only infer.
   Possible to add a guard: check whether the runtime sequence's
   final lock/sync event ever fired by examining stream switch
   counters or BD state post-hang.
3. **Is the completion mailbox response arriving from firmware
   at all during a hang?**  Driver-side instrumentation on
   `xdna_mailbox` resp paths would tell us.  Add a printk on every
   user-channel response with msg_id, opcode, status.
4. **If the message arrives, is the IRQ fired?  Is the right fence
   signaled?**  `/proc/interrupts` snapshots before/after the
   hang state, plus driver-side prints in the IRQ handler and the
   `drm_sched_fence` signal path, would localize this.
5. **What state changes across the reload that clears the hang?**
   Possibilities: firmware DRAM state, NPU SMU clock/DPM state,
   PCIe link state, mailbox doorbell counters.  Hardest to chase
   without targeted instrumentation but most explanatory if found.

## Recommended next investigation

The fastest path to a real root cause is **driver-side
instrumentation** of the completion path, then capture passing
vs hanging dmesg for line-by-line diff.

### Repro recipe for a reliable hang state

The hang is most reliably triggered immediately after a **fresh
boot** with the rebased + num_rqs-fixed driver auto-loaded.
After we've done one or more module reloads in a session, the
hang stops reproducing.  So:

1. Reboot.
2. Verify amdxdna loaded clean (no kernel oops in dmesg, refcount 0,
   `tdr_dump_ctx=Y`, `autosuspend_ms=-1`).
3. `xrt-smi validate` -- should pass; gives us a hwctx baseline.
4. Single `writebd` test -- should hang (rc=124 on 30s timeout).
5. Capture dmesg, /proc/interrupts, kernel /sys state.
6. Try a second run -- maybe still hangs, maybe not.  Capture either way.
7. `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`
8. Single `writebd` test -- now passes.  Capture dmesg + /proc/interrupts diff.

### Driver instrumentation targets

In order of likely yield, all in `xdna-driver/src/driver/amdxdna/`:

1. `aie2_message.c` / `amdxdna_mailbox.c` -- printk on every user-channel
   response received from firmware: msg_id, opcode, status, size.
2. `aie2_ctx.c` -- printk where "total completed jobs N" is logged;
   correlate which job id was counted.
3. `aie2_hwctx.c` -- printk in the IRQ handler and drm_sched fence
   signal path.  Specifically the place where `drm_sched_job` is
   marked complete.
4. Any TDR-detection-but-no-recovery code path (`tdr_dump_ctx=1`
   path).  Confirm we're not silently TDR-detecting and just not
   recovering, which would leave the fence un-signaled.

Build via `./build.sh -release -refresh_dkms` (the new flag we
landed today, see [CLAUDE.md "Developer environment state"](../../../CLAUDE.md)).
Reboot, reproduce, save dmesg.  Then repeat after a reload and
diff the two dmesg outputs.

## State of the world at wrap

| Item | State |
|---|---|
| xdna-driver branch | `xdna-rebase-2026-05-12` (5 commits on `upstream/main` d28f140 + 3 uncommitted patches: Makefile, aie2_message.c, hwctx.cpp; build.sh fix committed 3509b2a) |
| xrt submodule | `xrt-rebase-2026-05-12` (3 commits on `origin/master` 8223bea51 + 1 uncommitted patch in pcidrv.h) |
| Rollback tags | `pre-rebase-2026-05-12` on both repos |
| xdna-emu branch | `dev` (CLAUDE.md update committed 33e4ad2) |
| Running module | rebased+fixed (srcversion `5CA2BD72D0B1431D02E62BF`) |
| Module options | `autosuspend_ms=-1 tdr_dump_ctx=1` pinned in `/etc/modprobe.d/amdxdna.conf` |
| Test quarantine | empty (memtile_dmas entries removed in xdna-emu 907a1d2 -- in retrospect premature, but reverting it would be a bandage; better to fix bug #6 properly) |

## Should we roll back the rebase?

**Probably not.**  The rebase is bug-neutral for bug #6 -- doesn't
fix it, doesn't worsen it.  Keeping the rebased state has these
benefits:

- We're on current upstream, which simplifies any future PR or
  issue we file (`Xilinx/xdna-driver#...`).
- The num_rqs packaging fix in `build.sh` is general -- it would
  apply on pre-rebase too, but the bug only surfaces against the
  newer configure_kernel.sh script.
- Our local cherry-picks are still preserved.

Roll back only if a downstream issue emerges that's specific to
upstream commits in the rebase range.  Otherwise, fix bug #6 on
the rebased branch.

## Cross-references

- [2026-05-12-memtile-dmas-bug6-not-reproducible.md](../../archive/findings/2026-05-12-memtile-dmas-bug6-not-reproducible.md) -- the morning's premature "fixed" doc, now superseded.
- `build/experiments/2026-05-12-bug6-rebase-attempt/POST-REBOOT-FINDINGS.md` -- detailed kernel oops trace from this morning's num_rqs repro (kept under build/ since it's experiment scratch, not committed).
- xdna-driver commit `3509b2a` -- `build: add -refresh_dkms flag for in-tree DKMS iteration` (the num_rqs packaging fix).
- xdna-emu commit `33e4ad2` -- `claude.md: switch DKMS refresh shortcut to build.sh -refresh_dkms`.
- 2026-05-09 narrowing of bug #6 / `Xilinx/mlir-aie#3062` -- the original 5-test quarantine and TDR observation.
- 2026-05-07 finding `aie-rw-access-memtile-dm-half-impl.md` -- the mailbox poisoning chain that motivated removing the mgmt_chann teardown in our aie2_message.c patch.
