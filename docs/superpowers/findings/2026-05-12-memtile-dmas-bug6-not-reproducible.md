---
name: memtile_dmas (bug #6) not reproducible 2026-05-12 -- task #29 closed
description: The 5 memtile_dmas tests quarantined on 2026-05-09 under bug #6 (TDR on Phoenix NPU1, runtime-sequence path) no longer reliably TDR. Bridge sweep, bare test.exe, and lit all PASS after a single modprobe reload at session start.  No source change in xdna-driver since 5/9; bisect task #29 closed without identifying a regressing commit because no deterministic reproducer remained.
type: finding
---

# memtile_dmas TDR (bug #6) no longer reproducible in our environment

## TL;DR

The five `test/npu-xrt/memtile_dmas/*` tests quarantined 2026-05-09 under
bug #6 / `Xilinx/mlir-aie#3062` -- which the 5/9 narrowing reliably TDR'd
via direct lit invocation -- now PASS in our environment.  Verified
2026-05-12 via three independent invocation paths:

| Path | Result |
|---|---|
| Bare `./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin` | PASS |
| `lit -v --filter memtile_dmas/writebd/run.lit` (5/9 canonical repro) | PASS |
| `scripts/emu-bridge-test.sh memtile_dmas` (full sweep) | 5/5 chess + 2/2 peano PASS, traces CLEAN |

Quarantine entries removed from `scripts/test-quarantine.txt`.  Task #29
(bisect XRT/XDNA 2.21 vs 2.23 for the regression) closed: no
deterministic reproducer means no usable bisect oracle.

## Session timeline

### Session start (post-compact, post-reboot): bug reproduced ONCE

System state:
- xdna-driver HEAD `c347d62` (May 9), kernel module built May 11 04:36
  (DKMS auto-rebuild, code-identical)
- xrt 2.23.0 (`0e2c0a82ba`, built 2026-05-09)
- xrt_plugin-amdxdna 2.23.1
- NPU FW 1.5.5.391
- Persistent modprobe options: `autosuspend_ms=-1 tdr_dump_ctx=1`
- Uptime at session start: hours-old boot, possibly several hwctx
  create/destroy cycles from prior session work on task #36

Initial reproducer:
```
$ pkexec sh -c 'echo 120 > /sys/module/amdxdna/parameters/timeout_in_sec'
$ cd .../memtile_dmas/writebd && timeout 150 ./test.exe ...
# Hung past 120s TDR, killed by SIGTERM, rc=143
```

dmesg showed:
- `ctx -> partition [1, 1]`, `created hwctx 5` (i.e., 4+ prior hwctxs)
- Mailbox opcode 0x18 (EXEC_CMD) sent at job submit -- never returned
- `submitted 1 completed 0` (TDR dump-only at 120s)
- Test killed; ctx torn down normally; mgmt mailbox stayed alive

### Mid-session: modprobe reload "fixed" everything

While probing `force_cmdlist` (the AIE2 cmdlist-vs-execbuf knob in
`aie2_ctx.c:446`), did `modprobe -r amdxdna && modprobe amdxdna ...`
to apply a command-line param (which didn't take, sysfs override used
instead).  After the reload:

- writebd via bare test.exe: PASS in <1s
- 10 writebd iterations in a row: all PASS
- 4 mixed tests x 5 rounds (20 hwctx cycles): all PASS
- writebd via `lit --filter memtile_dmas/writebd/run.lit` (canonical
  5/9 repro): PASS
- `force_cmdlist=N` vs `=Y`: both PASS post-reload (confirming the
  cmdlist code path wasn't the differentiator)

### Bridge sweep, quarantine lifted: all 5 pass

Removed the 5 memtile_dmas entries from `scripts/test-quarantine.txt`,
ran `scripts/emu-bridge-test.sh memtile_dmas`:
- 5/5 Chess HW PASS, 5/5 Chess BRIDGE PASS, 5/5 trace CLEAN
- 2/2 Peano HW PASS (where Peano-buildable), 2/2 BRIDGE PASS, 2 CLEAN
- 32 total HW invocations including trace-prep, no hang

## What changed between 5/9 and 5/12

Audited:
- `xdna-driver/src/driver/amdxdna/` -- no commits since 2026-05-09
  (HEAD `c347d62`).
- `xrt` submodule -- no commits since 2026-05-09.
- Kernel module rebuilt 2026-05-11 04:36 via DKMS -- triggered by a
  kernel update, but source was identical, so the binary is
  functionally equivalent.
- NPU firmware 1.5.5.391 unchanged (still the bundled blob).
- aietools 2025.2 / RyzenAI 1.5.x unchanged.

Things we cannot audit:
- On-chip persistent firmware state across reboots.  Phoenix firmware
  reset behavior is opaque to us.
- Microcode updates to SMU or related blocks delivered via routine
  kernel/firmware updates.
- Whether the 5/9 session was in a subtle "wedged" state that
  persisted across one or more reboots and finally cleared.

## Hypotheses

In rough order of plausibility, with the available evidence:

1. **The 5/9 failure was state-dependent, and the state has been
   cleared by intervening reboots / module reloads.**  Supporting
   evidence: a single modprobe-r/+r mid-session cleared it; today's
   bare/lit/bridge invocations all pass; we never re-triggered it
   after the reload despite 30+ HW runs.

2. **Something in our environment changed unobserved.**  Possible
   firmware microcode update via kernel upgrade, or a side-effect of
   our `tdr_dump_ctx=1` persistent mod-param interacting with some FW
   state machine that 5/9's session lacked.  No direct evidence, but
   we can't rule it out.

3. **The 5/9 reproducibility report was overstated.**  The 5/9 doc
   describes empirical narrowing via "_diag_memtile_write/" scratch
   tests, suggesting many reproductions during that session, so this
   feels weak.  Worth keeping in mind anyway.

The most actionable next step would be reproducing the failure
intentionally (e.g., long-running pc-anchored sweep, then probe
memtile_dmas) -- but each attempt costs HW time and there's no
priority forcing it.  Deferred.

## Implications for task #29

The bisect task is **closed without a SHA**.  Bisect requires a
deterministic good/bad oracle; we have a stochastic one at best.
Without a reliable trigger, walking 333 driver commits is a slot
machine.  If the failure returns:

- Reproduce it deterministically first.  Note what state the device
  was in when it appeared, what tests/workloads ran before, FW
  version, kernel module load duration.
- Then bisect, anchored at 2.21.68 (or earlier if needed).
- Erika's `Xilinx/mlir-aie#3062` may want a status update -- our PASS
  on the same hardware family suggests her side may also have
  resolved, or our setups differ in ways the issue doesn't capture.

## Mitigations to retain

- `tdr_dump_ctx=1` persistent in `/etc/modprobe.d/amdxdna.conf`.  If
  the hang returns, we want dumps-only (not poison-the-mailbox
  recovery).  This is still load-bearing for other failure modes
  (the original 2026-05-09 finding).
- The 2-second default TDR timeout is fine for our workloads; even
  with the bug present, 120s didn't help, so a longer TDR window
  isn't a workaround.

## See also

- `2026-05-09-bridge-test-contention-and-trace-wedge.md` -- the
  prior session that documented the failure and quarantined the
  tests.  Section "Continuation: 2026-05-09 evening" has the
  original narrowing.
- `2026-05-07-aie-rw-access-memtile-dm-half-impl.md` -- separate
  memtile-read failure mode that motivates the
  `tdr_dump_ctx=1`-as-default operational rule.
- `scripts/test-quarantine.txt` -- now empty of test entries; the
  five memtile_dmas lines were removed in the same commit as this
  doc landing.
- `Xilinx/mlir-aie#3062` -- upstream issue still open as of
  2026-05-12; consider posting a follow-up with this finding.
