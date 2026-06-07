# aiesim Failure Triage — three-way timing campaign

**Status:** ACTIVE investigation (campaign running 2026-06-06/07)
**Companion:** [trace-event-sweep-campaign.md](trace-event-sweep-campaign.md)

The all-72 three-way (HW / interp / aiesim) timing campaign (#85) surfaced a set
of aiesim failures. This doc catalogs them, their failure modes, and the
fix-first plan. **None come from this session's harness fixes** (B-full and the
budget-cap removal both touch only the HW/EMU path; the aiesim verdict path was
untouched except the xfail-labeling fix).

## Session harness fixes (committed)

- **`139f265` B-full** — the cycle-diff leg now derives from the *main run*
  (`parse-trace --out-cycles` on the already-captured trace BO); `_classify_cycle_diff`
  rewritten to read the main-run per-side `events.json` + run `trace-compare`
  directly; the redundant re-execution pipeline (`_run_trace_cycles_pipeline` /
  `_run_trace_compare`) retired (~180 lines). Fixed the broken total-cycle leg
  (was 1 `cycles.HW.txt`, now 139).
- **`7acab8b` budget cap removed** — B-full producing `cycles.HW.txt` activated a
  dormant "dual-bound" block that set `XDNA_EMU_MAX_CYCLES = span * 2`. But that
  scalar is the trace event *span* (~10-70k) while the cap is on the emulator's
  *absolute* cycle counter (runs to hundreds of thousands) -> every EMU run
  returned BUDGET. Dropped the cap; the 600 s wall-clock is the runaway guard.
- **`179a1b0` aiesim xfail** — `run_one_aiesim` now applies `is_xfail` like
  HW/EMU (was reporting xfail kernels as hard FAIL).

## Campaign results (run3, FINAL — `build/bridge-test-results/20260606/`)

Campaign completed cleanly (exit 0, all 154 jobs).

- **EMU/bridge:** all PASS except the HW-quarantined kernels; **0 BUDGET**
  (budget fix held across the full run).
- **cycle-diff column:** chess `58 DRIFT 13 EMPTY 2 NO_CORE 25 skip`; peano
  `46 DRIFT 17 EMPTY 3 NO_CORE 32 skip`. 0 MATCH / 0 *_TRACE_BUG / 0 COMPARE-ERR.
  DRIFT is the expected signal — emulator timing is ballpark, not exact.
- **aiesim:** chess `75 pass, 6 fail (5 timeout)`; peano `62 pass, 11 fail
  (8 timeout)`. The harness flagged **12 aiesim-vs-HW calibration mismatches**
  (HW=PASS, aiesim != PASS) — the genuine model-gap set, cataloged below.

## aiesim failure families (FINAL catalog)

The control-**read-response** clone patch (`aiesim-bridge/src/cluster_clone_patch.cpp`,
prints `[aiesim-clone] ACTIVE`) **is compiled in and active**. The failures below
are *different* issues, not regressions of that fix. Each kernel below **PASSES
on HW + EMU** unless noted.

### A. Control-packet reconfig routing
- **`ctrl_packet_reconfig_elf`** (chess+peano) — **FAIL** via our PL-egress panic
  guard (`88be997`): `DATA entered shim->PL egress port 9 ... d0=0x00202001
  d1=0xf0000000 ... after 0 TCTs` — classified as *real* data routing to PL on an
  all-NoC NPU. Plus invalid AXI-MM reads at col=5 (reconfig reading config back).
  *Open question:* is the PL-egress data actually trace egress? `--no-trace` test
  in flight (if so, the panic clears with trace off → not a routing bug).
- **`ctrl_packet_reconfig_1x4_cores`** (chess+peano) — **TIMEOUT**: hang
  mid-execution (146-word reconfig CDO, 1787 instructions started, wedged).
- **`ctrl_packet_reconfig`, `ctrl_packet_reconfig_4x1_cores`** (chess+peano) —
  aiesim TIMEOUT, but **FAIL on HW too** (quarantined). Not a calibration
  mismatch; not aiesim-specific.

### B. DMA completion wedge — large transfer
- **`dma_task_large_linear`** (peano) — **TIMEOUT**: hung at `Executing 10 NPU
  instructions` for the full timeout (472-word CDO, large linear DMA). The
  transfer never signals completion in the aiesim model, so the wait hangs. Same
  family as #76 (S2MM completion under BD reuse), #78 (start-queue overrun), #79
  (sync-token wedge), for the large-linear case. Also EMPTY-trace on EMU.

### C. Repeat-count / objectfifo-repeat (NEW — surfaced by the full campaign)
All **peano-only TIMEOUT**, all also EMPTY-trace on EMU. The repeat/iteration BD
mechanism appears to wedge the aiesim model.
- **`nd_memcpy_linear_repeat`** (peano) — TIMEOUT.
- **`objectfifo_repeat/compute_repeat`** (peano) — TIMEOUT.
- **`objectfifo_repeat/init_values_repeat`** (peano) — TIMEOUT.
- (`objectfifo_repeat/simple_repeat` PASSES — NO_CORE/DMA-only passthrough;
  `distribute_repeat` is the XFAIL data-mismatch case, see G.)

### D. Packet-flow fan-in (NEW)
- **`packet_flow_fanin`** (chess+peano) — **TIMEOUT**. Note `packet_flow` and
  `packet_flow_fanout` both PASS; only the fan-in topology wedges. Suggests a
  many-to-one stream-merge / arbitration gap in the aiesim model.

### E. vec_mul_trace_distribute_lateral (NEW)
- **`vec_mul_trace_distribute_lateral`** (peano) — **FAIL** (chess build failed,
  so chess is `FAIL*`=compile-fail, not an aiesim verdict). Name implies a
  trace-distribute kernel — a prime `--no-trace` candidate.

### F. Documented non-bug — cascade c2c shared memory (#82)
- **`matrix_multiplication_using_cascade/buffer`** (chess) — **TIMEOUT**. Already
  diagnosed (#82) as an **AMD aiesim AIE2 core-to-core shared-memory limitation**,
  not our bug. Excluded from the fix queue.

### G. xfail labeling
- **`objectfifo_repeat/distribute_repeat`** (peano) — aiesim reproduced the same
  expected data mismatch the kernel XFAILs with on HW/EMU. Fix `179a1b0` maps
  aiesim FAIL→XFAIL when `is_xfail`; the campaign shows `FAIL` because its process
  **started before `179a1b0` landed**. Will read XFAIL next run. No action.

## CRITICAL: the 1200s timeout cutoff is too close to the legitimate PASS tail

Before treating any TIMEOUT as a model gap, note the aiesim PASS-time
distribution from the campaign:

- 137 aiesim PASS; **slowest PASS = 1081.6s** (`packet_flow_fanout` peano).
- 4 PASS > 1000s, 6 > 900s, 9 > 700s.
- Timeout backstop = **1200s** — only ~120s above the legit PASS tail.

aiesim wall-time is enormous and highly variable (hundreds of seconds to ~1100s
even for kernels that *pass*). The 1200s cutoff cuts through the tail of the
healthy distribution, so **an unknown fraction of the TIMEOUT verdicts are
false** — slow-but-progressing sims guillotined just past the PASS cluster, not
wedged. Smoking gun:

```
nd_memcpy_linear_repeat (chess): PASS    1027.3s
nd_memcpy_linear_repeat (peano): TIMEOUT 1200.3s   <- SAME kernel, different compiler
```

Mechanism: `dma_wait`'s hard sim-time backstop is `poll_max_ns()` default = 50ms
*sim* time, which at cycle-accurate speed is ~1200s *wall* — so the timeouts
cluster at 1200s because that is the natural sim-budget limit, not a wedge.

## Next-step plan (CLASSIFY-then-FIX)

Per Maya: fix the genuinely-broken ones — but first separate genuine wedges from
merely-slow, or we'd "fix" kernels that just needed more wall-clock.

1. **(done) Campaign finished**; full catalog above.
2. **Wedge-vs-slow classification** (IN FLIGHT, `20260607-ratelog`): re-run the
   6 genuine-timeout kernels with the bridge's built-in instrumentation —
   `XDNA_AIESIM_RATE_LOG=1` (`[rate]` sim-vs-wall heartbeat), `XDNA_AIESIM_TRACE=1`
   (`[dma_wait]` status + `quiescent wedge` detector), `XDNA_AIESIM_POLL_MAX_NS=0`
   (disable the 50ms sim backstop, rely on quiescence), `AIESIM_TIMEOUT=2400`.
   Read the heartbeat trajectory:
   - sim-time climbing + DMA txns advancing → **slow, false timeout** → no model
     fix; raise the cap (and consider why peano builds are slower).
   - `[dma_wait] quiescent wedge` fires fast → **genuine DMA wedge** → fix.
   - sim-time flat but no quiescence trip → wedged outside `dma_wait` (compute /
     other wait path) → investigate that path.
3. **`--no-trace` arm** (Maya's overwhelm hypothesis), for any that are slow:
   does removing trace injection bring them under budget? If yes, the
   fewer-events option is the lever.
4. **Fix the genuine wedges** (priority) — real model gaps only.
5. **Build the fewer-than-8-trace-events option** (trace-prepare / harness) — for
   trace-overwhelm kernels + sweep capacity (some tiles have only 4 trace slots).
6. (Last resort) an `aiesim-known-timeout` skip list — only for genuinely
   unfixable gaps, after fix attempts.

The separate FAILs (not timeouts) are distinct, fast-failing, real:
- `ctrl_packet_reconfig_elf` — PL-egress panic (FAIL at ~360s).
- `vec_mul_trace_distribute_lateral` (peano) — FAIL at ~540s.

## Then: trace-event-sweep characterization campaign

Separate thread, after three-way closes. Build the cross-kernel sweep aggregator
(the one missing piece — `trace-sweep.py` emits per-tile matrices but nothing
rolls them up into a per-event HW-vs-EMU fidelity report), then run the
full-corpus sweep (chess + peano, all events). See companion doc.

## In-flight state

- Campaign running in the background; results land in `20260606/`. Pre-fix partial
  runs preserved as `20260606-partial-prebfull` and `-partial-budgetbug`.
- **On completion:** pull the full aiesim catalog + Phase 5d three-way report,
  then start the `--no-trace` triage (step 2).
