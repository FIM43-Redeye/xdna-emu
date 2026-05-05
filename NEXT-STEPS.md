# Next Steps

Working notes for resuming from 2026-05-05. Granular task state lives in
the task list (#XXX references below); this file is the orientation
document — read it first when picking up where we left off.

## State at session end (2026-05-05)

### What landed today

- **#351 fixed**: `aiex.npu.address_patch arg_idx` is the *BO arg
  index* in the kernel regmap (after the opcode/instr_BO/ninstr
  prefix), not the XRT kernel slot. Our injector was passing
  `3 + max_existing_memref_args` (= 6 for 3-memref tests), which made
  firmware patch BD15 with garbage. Fix: pass `max_existing_memref_args`
  (the BO index) for `arg_idx` and keep `3 + max_existing_memref_args`
  as the host-side XRT slot. Verified on real HW with
  `add_one_using_dma`: trace_raw.bin now populated on both compilers,
  both sides. See `docs/superpowers/findings/2026-05-05-trace-arg-idx-bug.md`.
- **Bridge test default flipped**: `NO_TRACE=false` is the new default.
  The old `NO_TRACE=true` made `--no-trace` runs silently produce no
  trace data while looking like a successful run. The script now does
  what its name suggests by default.
- **Calibration framework + cycle-cost model** committed (sweep
  analyzers, sweep configs, provisional NPU1 fast-mode constants).
- **Trace start pipelining + multi-tile timer reset** committed.
  Closes some of the gaps documented in `docs/coverage/`.

### What we discovered but haven't acted on yet

#### `read_aie_reg` is firmware-blocked on NPU1 (today's sidetrack)

The earlier "closed-source" diagnosis was wrong. The host stack is
fully open: `xrt::hw_context::read_aie_reg` → `xrt_core::query::aie_read`
→ `DRM_AMDXDNA_AIE_TILE_READ` → `aie2_aie_tile_read` →
`aie2_rw_aie_reg`. The block is at
`aie2_is_supported_msg(MSG_OP_AIE_RW_ACCESS)`, which checks a per-device
firmware feature table:

- `npu1_regs.c` (Phoenix) lists only `MSG_OP_CHAIN_EXEC_NPU` at
  protocol 5.8. **`MSG_OP_AIE_RW_ACCESS` (opcode 0x203) is absent —
  Phoenix firmware never implemented it.**
- `npu4_regs.c` (Strix) lists `MSG_OP_AIE_RW_ACCESS` at protocol 6.24+.

We tried bpftrace `override()` (rejected by BPF verifier — function not
in `ALLOW_ERROR_INJECTION` allowlist), then a signed livepatch module
(loaded fine — but `aie2_is_supported_msg` is **fully inlined by GCC at
-O2** into all callers, so symbol-level livepatch doesn't redirect any
actual call site). Disassembly of `aie2_rw_aie_reg` shows zero
`R_X86_64_PLT32` relocations to `aie2_is_supported_msg`. Dead end via
livepatch.

**The path that's still open**: we build `amdxdna.ko` from source via
DKMS. A one-line driver-source patch removes the firmware-feature
check entirely (or, equivalently, adds `MSG_OP_AIE_RW_ACCESS` to
`npu1_regs.c`'s op table). Recompile, reload, retry. This is the
proper way to bypass the gate, given we're already building the
driver. **Most likely outcome**: Phoenix firmware returns "unknown
opcode" status → driver returns `-EINVAL`. Lower probability: timeout
→ mgmt channel destroyed → driver reload. All recovery paths
documented in `xdna-emu/CLAUDE.md`.

We *also* have a separate lifecycle bug in
`bridge-runner/bridge-trace-runner.cpp`: the pre-launch
`read_aie_reg` call hits `Column 0 outside partition range [0, 0)`
because `hwctx->num_col` is zero until the first `run.start()`. Move
the perf-counter init/reset to after `run.start()` returns (the
post-wait read works fine — it gets EOPNOTSUPP from the firmware
gate, not from the partition check).

References:
- src: `npu-work/xdna-driver/src/driver/amdxdna/aie2_message.c:1772`
  (`aie2_rw_aie_reg`'s firmware-support check)
- src: `npu-work/xdna-driver/src/driver/amdxdna/npu1_regs.c`
  (Phoenix op table — add entry here)
- src: `npu-work/xdna-driver/src/driver/amdxdna/npu4_regs.c:40`
  (Strix already has it at FW 6.24)
- finding: `docs/superpowers/findings/2026-05-04-control-path-cycle-calibration.md`
  (corrected end-of-doc — the path *is* open, it's just firmware-gated)
- task: #356 (closed; firmware diagnosis confirmed)
- task: #357 (open; retry on NPU4 hardware when available)

#### EMU trace divergence has multiple distinct causes (#321 re-scope)

On `add_one_using_dma` post-fix, HW captures 67 trace events spanning
~12.6k cycles (slots 0/6/7 = PERF_CNT_0, LOCK_STALL,
INSTR_LOCK_ACQUIRE_REQ); EMU captures only 16 events spanning 264
cycles, all slot 7 (INSTR_LOCK_ACQUIRE_REQ). The original task #321
("trace-stop timing") was too narrow:

- **#321** (now narrowed): trace-stop timing — verify
  USER_EVENT_2 / `trace_done` broadcast fires at the right cycle in
  EMU vs HW.
- **#353**: EMU never emits `LOCK_STALL` events at all. Either the
  EMU's lock-acquire model treats stalls as instant, or the
  `LOCK_STALL` event ID isn't wired into the trace unit's
  event-detection path.
- **#354**: EMU never emits `PERF_CNT_0` anchor pulses. Either the
  perf counter isn't incrementing, or the threshold-match isn't
  emitting the event into the trace stream, or the kernel doesn't
  reach the threshold (depends on #355).
- **#355**: EMU's effective kernel-activity cycle range is ~50x
  shorter than HW. This is probably **upstream of #353 and #354** —
  if EMU runs the kernel in 250 cycles when HW takes 12.6k, neither
  PERF_CNT_0 thresholds (every 1024 cy) nor LOCK_STALL events have
  enough wall-time to fire.

**Recommended order**: investigate #355 first. If we fix the cycle
model to put HW and EMU on comparable timelines, #353 and #354 may
resolve as side effects. #321 (trace-stop timing) is probably the
last to address — it's only meaningful once events generate at the
right rate.

This sequence overlaps with #342 (differential validation of
calibrated `CycleCostModel`) — they're the same problem viewed from
different angles. Worth treating as one workstream.

## Pending work (in priority-suggested order)

1. **Verify #351 fix on full bridge sweep** — single test verified
   today; need to run the whole matrix to confirm no regressions and
   measure populated-trace rate across all ~75 tests. Single command:
   `./scripts/emu-bridge-test.sh --compile`.

2. **#355 → #354 → #353 → #321** — EMU trace divergence chain.
   Start with #355 cycle-range investigation; expect that the upstream
   cycle-cost gap is responsible for most of the missing-events
   symptoms. Coordinate with #342 differential validation.

3. **NPU1 driver patch experiment** (one-liner) — when we have time
   for a low-priority diagnostic, patch `aie2_rw_aie_reg` to skip the
   firmware-feature check, rebuild via DKMS (we do this anyway for
   driver updates), retry `bridge-trace-runner --read-perf-counter`.
   Also fix the partition-range lifecycle bug in
   `bridge-trace-runner.cpp` regardless. Expected return: `-EINVAL`
   from firmware "unknown opcode" rejection; small chance of useful
   signal. Cleanup if it doesn't pan out: revert the one-liner.

4. **#357 re-test on NPU4 when hardware arrives** — Strix firmware
   ≥ 6.24 implements `MSG_OP_AIE_RW_ACCESS`, so the same
   `bridge-trace-runner --read-perf-counter` flow that fails on Phoenix
   should succeed on Strix. If it does, on-NPU
   `Performance_Counter0` readback becomes a trace-independent ground
   truth and major chunks of our trace pipeline become optional rather
   than load-bearing.

5. **Calibration follow-ups** (#342, #347) — differential validation,
   broadcast event propagation latency measurement.

## Cleanup before next session

- `/tmp/claude-1000/livepatch-aie2/` — experimental kmodule, can be
  deleted (`/tmp` is wiped on reboot anyway, but for clarity).
- `/tmp/claude-1000/perfcnt-test/` — input/output/trace artifacts from
  the read_aie_reg experiment; same.
- The task list is current; tasks #353, #354, #355, #357 are pending
  and self-contained.

## What didn't get committed but is worth preserving

Nothing — all source changes from today are committed (eight commits,
last one being the cycle_cost.rs comment correction). The findings
docs include the corrected `read_aie_reg` analysis. Working tree is
clean apart from this NEXT-STEPS.md file.
