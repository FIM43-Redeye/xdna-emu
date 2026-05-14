---
name: 'Bridge test reliability: HW/EMU contention masked v1 failures; memtile_dmas family is broken on HW *independent* of trace'
description: The mysterious memtile_dmas/cascade/packet_flow HW failures in v1 sweeps were artifacts of EMU at -j16 starving HW dispatch. Switching Phase 3+4 to sequential (HW first solo, then EMU parallel) makes 90/95 HW tests pass cleanly. Five script tooling improvements landed. The "trace wedge on memtile-only tests" hypothesis (Bug #6) was a RED HERRING -- post-reboot bisect showed memtile_dmas/* TDR even with --no-trace, both compilers, repeatedly. The trace prologue is bit-identical to passing tests. Real cause is unknown; suspect memtile DMA programming / firmware state. Bug #6 remains open with new scope.
type: project
---

# Bridge test reliability + memtile_dmas regression -- 2026-05-09

## TL;DR

A long debugging session went through three phases:

1. **CPU contention masquerading as HW bugs.** Phase 3+4 originally ran
   HW (-j1) and EMU (-j16) concurrently. EMU at -j16 starved HW's
   userspace dispatch enough to cause spurious TDRs and even data
   corruption (FAIL) on memtile_dmas tests. Confirmed by switching to
   sequential ordering. The "30/30 PASS" reliability run that supported
   this hypothesis was either run with trace enabled (different code
   path than my variants) or hit a transient working window -- post-
   reboot the same tests now fail consistently with --no-trace.

2. **Trace pipeline crash on shim+memtile-only tests.** `mlir-trace-inject.py`
   bailed early when a device had no compute tiles, then `_build_trace_config`
   crashed downstream on `min(t["col"] for t in tiles_traced)`. Affected
   the entire `memtile_dmas/*` family plus several others. Real bug,
   now fixed (commit `534383e`).

3. **"Trace wedges memtile-only tests" -- RED HERRING.** With trace
   injection enabled, `memtile_dmas/dma_configure_task_lock` reproducibly
   TDRs the NPU within 9-15s. Initial bisect showed both shim-only and
   memtile-only injection variants fail, suggesting a shared trace
   plumbing issue. **However, follow-up MLIR-level binary-search variants
   (A-F below) ALL TDR, including a variant that drops the entire
   shim trace BD setup. After clearing build caches and re-running with
   --no-trace, the test STILL TDRs.** All four memtile_dmas tests we
   tried (blockwrite_using_locks, dma_configure_task_lock,
   dma_configure_task_token, writebd, writebd_tokens) TDR with --no-trace
   on both compilers. add_one_using_dma --no-trace passes; xrt-smi
   validate passes; system is healthy. The trace was incidental.

The HW-only sweep with sequential ordering ran 95/95 PASS in 26 minutes
in this session, but that was BEFORE the memtile_dmas regression
manifested -- they appear in the doc-section "Tests confirmed PASSING"
below but no longer pass.

## Bridge test improvements landed

Five commits on `dev` between `9e5401e` and `9a0b793`:

- `425e798` -- bridge-test: parallel Phase 5b, sequential Phase 3+4,
  elapsed-time progress
- `7e9c071` -- bridge-test: skip Phase 5 trace compare when HW or EMU
  is disabled
- `534383e` -- mlir-trace-inject: don't skip devices without compute
  tiles
- `9a0b793` -- bridge-test: nuke stale per-test artifacts for sides not
  in this run

(Also one earlier commit during the same session for trace-inject
itself.)

The Phase 5b refactor splits the event sweep into two arms: HW serial
(NPU is single-tenant), EMU parallel via `xargs -P $JOBS` (no device
contention). Same HW-first ordering as Phase 3+4. This is where the
biggest wall-clock savings come from on `--sweep` runs.

The stale-artifact guard fixes a misleading-report bug we walked into:
running with `--no-emu` left old `.emu/` dirs in place from a previous
run, so Phase 5 silently compared today's fresh HW against yesterday's
stale EMU and the report displayed misleading "PASS PASS CLEAN" rows.
The guard nukes the disabled side's artifacts before Phase 3+4 starts.

## Bug #6 (rescoped): memtile_dmas/* family TDRs on HW with --no-trace

### Repro

```bash
pkexec modprobe -r amdxdna && pkexec modprobe amdxdna
xrt-smi validate                      # confirm NPU healthy
# clear cache so bridge test actually rebuilds
rm -rf ../mlir-aie/build/test/npu-xrt/memtile_dmas/dma_configure_task_lock/{chess,peano,traced,test.exe,test.cpp}
nice -n 19 ./scripts/emu-bridge-test.sh --no-emu --no-trace --chess-only \
    'memtile_dmas/dma_configure_task_lock$'
```

Result: HW TDR at ~7s, 100% reproducible. add_one_using_dma --no-trace
PASSES under the same conditions, so the system itself is healthy.

### Variant-bisect ruling out the trace-prologue hypothesis (2026-05-09)

Hand-edited `aie_traced.mlir` and recompiled via aiecc, then ran
test_traced.exe directly with the modified xclbin/insts.bin. All
variants TDR'd:

| Variant | What was changed | Result |
|---|---|---|
| Baseline | trace as injected | TDR @10s |
| A | drop issue_token bit on trace BD push (0x8000000F → 0xF) | TDR |
| B | drop shim event broadcast/timer setup (0x34000/0x3404C/0x34008) | TDR |
| C | drop memtile trace config (0x94*** writes) + packet flows | TDR |
| D | drop entire shim trace BD setup (writebd 15 + push) | TDR |
| F | keep all writes, drop just the packet_flow blocks | TDR |

After this dead-end, cleared the build cache and re-ran the bridge
test with `--no-trace`. **Still TDR.** That eliminated trace as a
cause entirely. Also confirmed:

- 4/4 other memtile_dmas tests (blockwrite_using_locks,
  dma_configure_task_token, writebd, writebd_tokens) all TDR with
  --no-trace on chess.
- The trace prologue is **bit-identical** to add_one_using_dma's
  trace prologue (same controller_id collision pattern, same shim S2MM
  1 BD push with token, same memtile Timer_Control = 0x9D00).
  add_one_using_dma passes with trace, memtile_dmas tests TDR with or
  without trace.

So the original "trace wedges memtile-only tests" bisect (which I'd
done by toggling --shim-sweep-events vs --memtile-sweep-events flags)
was finding the same underlying TDR each time, NOT a trace-specific
bug.

### What dmesg actually shows

```
amdxdna 0000:c6:00.1: aie2_tdr_work: Device isn't making progress... Count 13 timeout 2
amdxdna 0000:c6:00.1: aie2_dump_ctx: Dumping ctx ...
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	op: 0x0
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	msg: 0x1d000001
amdxdna 0000:c6:00.1: aie2_dump_ctx: 	fence: unsignaled
```

Generic "kernel never completed". No firmware-side diagnostic. The
kernel is just hung somewhere in the runtime sequence's DMA waits.

### Open questions for next session

- Are these tests ACTUALLY broken, or is there persistent state from
  trace TDRs that a stronger reset (suspend/resume, reboot) would
  clear? `pkexec modprobe -r amdxdna && pkexec modprobe amdxdna` does
  NOT clear it -- need to escalate.
- If tests are genuinely broken: when did they regress? They were
  added by mlir-aie commit `9c92428136` on 2026-02-06 and were
  reportedly passing in earlier sessions.
- WHERE is the wedge happening? The runtime sequence has chained
  memtile S2MM 0 BDs (0,1,2,3) gated on prod_lock/cons_lock, then
  memtile MM2S 0 (BD 4) acquires cons_lock 4× and reads, then shim
  S2MM 0 receives. dmesg can't tell us which step hangs. Need to
  read AIE registers post-TDR via `xrt::hw_context::read_aie_reg`
  (#356 added this on NPU1) to see lock state, BD state, channel
  state, etc.

### Suggested next steps (post-reboot)

1. Reboot to clear any persistent NPU/firmware state.
2. Re-run memtile_dmas/dma_configure_task_lock --no-trace and verify
   it still TDRs (confirm regression vs accumulated state).
3. If still TDRs: capture AIE register state at TDR time via
   `xrt::hw_context::read_aie_reg`. Read in order of likely hangs:
   - memtile prod_lock / cons_lock values
   - memtile S2MM 0 channel CTRL + STATUS + current_BD
   - memtile MM2S 0 channel CTRL + STATUS + current_BD
   - shim S2MM 0 channel CTRL + STATUS
   - shim MM2S 0 channel CTRL + STATUS
4. From register state, identify which DMA is stuck on which lock.
5. Cross-reference with mlir-aie git history near 9c92428136 to see
   if any subsequent commit changed lock semantics or BD chain
   behavior.

## Tests reportedly PASSING earlier in session (now contradicted)

The `bahg41enj` reliability run logged 30/30 PASS for these (3 iters,
chess+peano):

- `cascade_flows` (chess)
- `memtile_dmas/blockwrite_using_locks` (chess)
- `memtile_dmas/dma_configure_task_lock` (chess + peano)
- `memtile_dmas/dma_configure_task_token` (chess + peano)
- `memtile_dmas/writebd` (chess)
- `memtile_dmas/writebd_tokens` (chess)
- `packet_flow_fanout` (chess + peano)

**But these now consistently TDR with --no-trace.** Possible
explanations:

- The reliability run may have used trace artifacts (it ran during
  the same session as trace experiments and may have inherited
  cached trace-injected artifacts).
- Or the NPU was in a different state then; the many trace TDRs
  since then have left persistent state nothing short of a reboot
  clears.
- Or a recent emulator/tooling change perturbed something.

This needs to be re-verified after a reboot.

## HW-only full sweep summary

`bridge-sweep-20260509-hw-only.log` (started 15:39, 26m runtime):

- Chess: 61/62 compiled, **53/53 HW pass, 0 fail, 0 TDR** (10 skip)
- Peano: 54/55 compiled, **42/42 HW pass, 0 fail, 0 TDR** (11 skip,
  1 XFAIL)
- Trace prep: 59 OK, 10 FAIL (the bug #1 cluster, fixed in this
  session -- expect 69 OK on next sweep)

10 trace prep FAILs all hit the `min() iterable argument is empty`
crash now fixed by `534383e`:

- `bd_chain_repeat_on_memtile`
- `dynamic_object_fifo/sliding_window_conditional`
- `matrix_multiplication_using_cascade`
- `memtile_dmas/{blockwrite_using_locks, dma_configure_task_lock,
   dma_configure_task_token, writebd, writebd_tokens}`
- `objectfifo_repeat/simple_repeat`
- `ctrl_packet_reconfig_1x4_cores`

## Other artifacts from the session

- **PR amd/xdna-driver#1298** -- 25 typo fixes across driver, shim,
  uapi headers, and tools. Doc-only, doesn't depend on anything else.
- **`tools/dma-fill-measure.py`** built earlier in the day (commit
  `6b31370`) but not yet exercised against fresh data. The HW-only
  sweep produced 60 chess `.sweep/` dirs that could be ingested for
  #359 calibration once the EMU side is also captured.

## Continuation: 2026-05-09 evening -- read_aie_reg dead-end and the cascade we already knew about

After a reboot to confirm the bug #6 repro on a clean state (TDR
confirmed at 11s, EMU passes the same logic, system healthy), a second
session set out to build a register-readback harness so we could see
which DMA was stuck on which lock at hang time. The session ended in
exactly the same SMU-wedge cascade that `2026-05-07-aie-rw-access-
memtile-dm-half-impl.md` already documented two days earlier, and
should have been cross-referenced before any harness work began.

### What we built (and why it cannot debug bug #6 on Phoenix)

`tools/aie-reg-probe/` -- a small C++ sidecar (CMake + ~280 LOC) that
opens a hw_context, calls `force_connect`, optionally launches the
failing kernel, and reads a curated set of memtile + shim DMA
registers via `xrt::hw_context::read_aie_reg`.  Builds cleanly,
links against `/opt/xilinx/xrt/lib/libxrt_coreutil.so`, runs on
compute-tile reads.

It cannot read memtile registers on Phoenix.  Per the May 7 finding,
`MSG_OP_AIE_RW_ACCESS` (opcode 0x203) round-trips for compute-tile
register space but **silently never responds for any memtile address**
tested -- both memtile DM (0x10000) and memtile registers (TIMER_LOW
at 0x940F8).  The driver kills the user-context mailbox after a 5s
timeout and the device enters a degraded state from which `modprobe
-r` cascades to SMU wedge and only reboot recovers.

The harness still has a future: it works for compute-tile-side bugs.
Don't throw it out, but don't aim it at the memtile, ever, on this
firmware.

### TDR knob mechanics (newly mapped)

Two module parameters on `amdxdna`:

- `timeout_in_sec` (default 2; 0 disables TDR).  **Honored only at
  probe time** -- `aie2_tdr_start` checks `if (!timeout_in_sec)` and
  skips timer setup.  Setting it to 0 at runtime via sysfs does
  NOT stop the running timer; the next `mod_timer(t, jiffies + 0)`
  fires immediately, DOS-firing the TDR work.  To disable TDR
  entirely you must pass it as a module-load arg
  (`/etc/modprobe.d/amdxdna.conf` is the right place).
- `tdr_dump_ctx` (default 0).  **Live at runtime.**  Each TDR work
  invocation re-reads the global; with `dump_only=true`
  `aie2_tdr_force_recover` calls `aie2_rq_dump_all(rq)` and returns
  -- no `stop_all`, no `restart_all`.  Hung context stays alive.
  This is the right knob for keeping a hung kernel quiescent while
  a debugger reads state.

The TDR detect path (`aie2_tdr_work` -> `aie2_rq_dump_all` ->
`aie2_dump_ctx`) calls `aie2_get_app_health` to retrieve a structured
firmware health report (DPU PC, TXN OP ID, fatal error info, "timed
out sub command ID").  We never see this block in dmesg because
`MSG_OP_GET_APP_HEALTH` is gated -- listed in `npu4_regs.c` at FW
6.18+, **not in `npu1_regs.c`**.  On Phoenix the call returns
`-EOPNOTSUPP` from the gate, the `if (!ret)` guard skips the print
block, and we get only the bare `JOB[0]: op/msg/fence` info.  That
silence is by design of the gate, not a firmware bug.  Adding
`MSG_OP_GET_APP_HEALTH` to `npu1_msg_op_tbl` is tempting (firmware
might or might not have a handler), but `MSG_OP_GET_COREDUMP` was
tried by the same path (commits `7e641c9` then `0c5d393`) and
reverted -- proceed cautiously.

### The actual cascade we walked into

Sequence, captured live this session:

1. Bridge test runs `memtile_dmas/dma_configure_task_lock` chess
   `--no-trace` with `tdr_dump_ctx=0` (default).  HW TDR @ 11s.
   Driver's `aie2_tdr_force_recover` runs `aie2_rq_stop_all +
   aie2_rq_restart_all`.  Test exits.
2. `xrt-smi validate` PASSES.  System looks healthy by SMI metrics.
3. `tdr_dump_ctx=1` set via `pkexec tee
   /sys/module/amdxdna/parameters/tdr_dump_ctx`.  Verified `Y`.
4. `aie-reg-probe passive` against `add_one_using_dma`'s xclbin
   tries first read on memtile (col 0, row 1, lock0_value @ 0xC0000).
5. Mailbox req opcode 0x203 sent; firmware does not respond.
6. After 5s, `xdna_send_msg_wait: Wait for completion timeout` ->
   `aie2_rw_aie_reg: AIE reg read failed, ret -62`.  Driver
   destroys the user-context mailbox channel.
7. Probe is in D-state, eventually exits when each subsequent
   read also `-ETIME`s.
8. `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna && echo
   1 > .../tdr_dump_ctx'` is issued.  `modprobe -r` enters D-state
   in `drm_dev_unplug` -> `__synchronize_srcu`.  No reader is
   visibly active (`/proc/*/wchan` shows nothing else stuck), but
   the DRM unplug SRCU grace period never completes.  modprobe
   stuck for 7+ minutes; cannot be SIGKILL'd; only reboot recovers.
9. The `&& modprobe amdxdna` second half never runs.  Device is
   unbound from PCI driver, partial module state remains, `xrt-smi
   examine` reports 0 devices.

This matches the May 7 finding step-for-step.  We are now confident
that **any memtile read attempt on Phoenix produces this cascade**
and that **once the cascade starts, only reboot recovers** --
including from the modprobe-r side, which itself wedges
uninterruptibly.

### Side fix that landed this session: pkexec fingerprint

`pkexec` had been falling back to password despite an enrolled
fingerprint and `pam_fprintd.so` being first in
`/etc/pam.d/common-auth`.  Root cause: pkexec's PAM service is
`polkit-1`; with no `/etc/pam.d/polkit-1` file the chain fell
through to `/etc/pam.d/other` -> `common-auth`, where the chain
starts with

```
auth [success=ignore default=1] pam_exec.so quiet \
        /usr/local/bin/check-keyring-unlocked.sh
```

and the `default=1` skip on a non-zero exit jumps over `pam_fprintd`.
Even when the script returned 0 (keyring unlocked, fingerprint
allowed), some interaction in the pkexec PAM context was bypassing
fprintd silently -- the password prompt appeared instantly with no
fingerprint timeout.

Fix: created `/etc/pam.d/polkit-1` with

```
auth      sufficient   pam_fprintd.so timeout=10 max-tries=1
@include common-auth
@include common-account
@include common-password
@include common-session-noninteractive
```

`sufficient` means fingerprint short-circuits the chain on success
and falls through to common-auth (with its full keyring-aware
chain) on failure.  Verified working: `pkexec true` now prompts the
sensor first; password is the fallback if no swipe.

### Plan revision for bug #6 (after the reboot)

1. **Persist the TDR knob.**  Add to `/etc/modprobe.d/amdxdna.conf`,
   joining the existing `autosuspend_ms=-1` line:
   ```
   options amdxdna autosuspend_ms=-1 tdr_dump_ctx=1
   ```
   This is general defense -- every future failing test will dump
   on TDR rather than recover.  It does not unblock memtile reads,
   but it stops the recover-side damage to the firmware that
   compounds debugging.

2. **Drop the `read_aie_reg` plan for memtile state.**  The May 7
   wall is firm.  Until the Phoenix firmware grows memtile
   `AIE_RW_ACCESS` support (or we get a non-mailbox path), memtile
   register state is not directly readable on this hardware.

3. **Pivot to MLIR-level diagnostics.**  Three options, cheapest
   first:

   - **Bisect via intermediate `aiex.npu.dma_await_task`.**  The
     test ends with a single `dma_await_task(%t3)` on shim S2MM 0.
     Replace with progressive awaits on `t0`, `t1`, `t2` to
     identify which task's completion is missing when the test
     hangs.  Cheap iteration.
   - **Diff the runtime sequence.**  `add_one_using_dma` (passes)
     and `dma_configure_task_lock` (TDRs) both go shim->memtile
     ->shim with the same flow shape.  Decode `insts.bin` for both
     via `tools/parse-trace.py --raw` and find the structural
     delta (lock counts? BD-chain wraparound? `issue_token`
     placement?).  EMU passes both, so EMU isn't enforcing
     whatever HW constraint differs.
   - **Embed `aiex.npu.read32` reads in the runtime sequence.**
     Routes via control packets to a host BO, doesn't depend on
     the firmware mailbox.  Reads after the hang point never
     execute, so this is incremental: read state up to point N,
     advance N until we find the line where state goes wrong.

4. **Keep `tools/aie-reg-probe/` for compute-tile bugs.**  The
   harness is fine for any future bug that lives in compute-tile
   register state.  Don't throw it out; just don't aim it at
   memtile on Phoenix.

### Lessons / process notes

- **Always grep findings first.**  `grep -rin AIE_RW_ACCESS
  docs/superpowers/findings/` would have surfaced the May 7 doc
  before any harness work this session.  Cost of skipping: one
  reboot.
- **Driver reload is not safe in a poisoned-mailbox state.**  The
  CLAUDE.md operational note says "modprobe -r/+r handles most
  TDR recoveries" -- true for default TDR-recover state, false
  once the user mailbox has been killed.  In the killed-mailbox
  state, modprobe -r itself wedges in `synchronize_srcu` and
  reboot is the only path.  Update the recovery-escalation note.

## Continuation: 2026-05-09 evening (post-reboot) -- bug #6 narrowed and filed upstream

After the reboot, with `tdr_dump_ctx=1` made persistent in
`/etc/modprobe.d/amdxdna.conf`, the failure mode is recoverable:
hung tests dump every 2s but never poison the firmware mailbox, and
modprobe-r/+r works cleanly afterward.  This unblocked iterative
MLIR-level diagnostics that were impossible before.

### Empirical narrowing of bug #6

Constructed a series of minimal MLIR tests under
`mlir-aie/test/npu-xrt/_diag_memtile_write/` (ephemeral; deleted at
end of session) to isolate the trigger:

| Variant | Result |
|---------|--------|
| Single `aiex.npu.write32` to memtile (col 0, row 1) LOCK0_VALUE | PASS |
| Single `aiex.npu.writebd` to memtile (program BD, no exec) | PASS |
| `writebd` + push to memtile S2MM TASK_QUEUE (channel start) | PASS |
| Full `writebd` test, locks + chains as upstream-shipped | TDR |
| Full `writebd` with `use_next_bd=0` (no self-loop) | TDR |
| Full `writebd` with all `lock_acq_enable=0` (no locks) | TDR |
| `add_one_using_dma` (static `aie.memtile_dma` block via CDO) | PASS |

Conclusion: runtime-sequence writes to memtile registers,
single-BD programming, and channel pushes all work fine on Phoenix.
Static (CDO-time) memtile programming works.  But multi-channel
runtime-programmed memtile DMA flow shim -> memtile -> shim never
delivers data to the shim S2MM receiver.  Removing self-loops or
locks doesn't fix it.  EMU passes the same xclbin and test logic.

The hang is at `run.wait()` -> `drm_syncobj_array_wait_timeout` --
sync waits forever for a TCT from shim S2MM that never arrives.
The runtime sequence itself completes cleanly (verified by removing
the trailing sync: kernel returns, test fails on data check rather
than TDR).

### Bridge runner ruled out as a confounder

Suspected our bridge runner might be perturbing test execution
relative to upstream lit.  Reproduced via native lit invocation:

```
ironenv/bin/lit -v --filter "memtile_dmas/writebd/run.lit" build/test/
```

Same hang.  Bridge runner is faithfully reproducing the upstream
failure.  Process tree at the moment of failure:

```
lit -> bash run.lit.script -> python run_on_npu.py npu1 -> ./test.exe
                                                           (drm_syncobj_array_wait_timeout)
```

### Upstream CI and version comparison

Checked `Xilinx/mlir-aie` CI runs (`buildAndTestRyzenAI.yml`).  CI
matrix runs on `amd7940hs` (Phoenix, identical chip family to ours)
and `amdhx370` (Strix).  Most recent `main` CI run (2026-05-09
`Align buffer allocate (#3037)`) succeeded; lit summary reports 892
PASSED on `amd7940hs` but does not enumerate passing tests by name,
so it's not visible whether `memtile_dmas/*` actually ran.  On
`amdhx370`, `writebd` and adjacent tests show in the Unsupported
list -- expected, since `REQUIRES: ryzen_ai_npu1` isn't met on
Strix.  `dma_configure_task_lock` doesn't appear by name in either
runner's enumerated lists.

Environment delta vs upstream CI:

| Component | Us | Upstream CI |
|-----------|-----|-------------|
| Chip | Ryzen 9 7940HS (Phoenix) | `amd7940hs` (same family) |
| NPU FW | 1.5.5.391 | unknown -- log doesn't print |
| XRT | 2.23.0 | `/opt/xilinx/xrt` (version unknown) |
| amdxdna | 2.23.0\_20260509 (HEAD `c347d62`) | unknown |
| mlir-aie | HEAD `b37dc33d41` | HEAD CI runs main |
| Vitis | aietools 2025.2 (RyzenAI 1.5.x) | `/opt/ryzen_ai-1.3.0.1/vitis_aie_essentials` |

Strongest hypothesis: AMD's CI is on RyzenAI 1.3.0.1 while we are
on 1.5.x; the bundled NPU firmware may differ.

### Filed upstream

`Xilinx/mlir-aie#3062` -- "test/npu-xrt/memtile_dmas/* tests TDR on
Phoenix (NPU1) -- do these actually pass in CI?"  Asks for status
on the Phoenix runner and offers to file an XFAIL PR if upstream
confirms broken.

### Local mitigation landed

`scripts/test-quarantine.txt` extended to cover the five
memtile_dmas tests.  Bridge runner now skips them entirely (no
compile, no run, no trace) with a "SKIP (test-quarantined)" line.
Comment in the file references the upstream issue so it's clear
when to remove.

The quarantine file's docstring was generalized to allow two
reasons for entries: (1) structural problems, (2) persistent
failures on this hardware/firmware configuration that have been
triaged.  Existing usage was solely (1); adding the second
category covers our case without overloading the meaning.

## See also

- `../../archive/findings/2026-05-07-aie-rw-access-memtile-dm-half-impl.md`
  -- the prior session that documented the memtile-read hang and
  SMU-wedge cascade.  Should be cross-referenced before any
  read_aie_reg harness work on Phoenix.
- `../../archive/findings/2026-05-06-355a-host-latency-response.md`
  -- prior #359/#355a calibration progress (the original target this
  session was supposed to advance, before the bridge test mess
  surfaced).
- task #359 -- still in_progress; this session unblocks it by
  proving HW results are reliable.
- task #378 -- open follow-up for upstream maskwrite32 trace patch.
