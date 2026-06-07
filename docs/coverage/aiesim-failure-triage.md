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

## Campaign results (run3, `build/bridge-test-results/20260606/`)

- **EMU/bridge:** 149 PASS, 4 FAIL, 1 XFAIL, **0 BUDGET** (budget fix held).
- **cycle-diff column:** 104 DRIFT, 30 EMPTY, 5 NO_CORE, 10 NO_DATA (classifier
  working; DRIFT is the expected signal — emulator timing is ballpark, not exact).
- **aiesim:** ~116 PASS, 3 FAIL, 7 TIMEOUT (campaign still finishing as of writing).

## aiesim failure families (pre-existing model gaps)

The control-**read-response** clone patch (`aiesim-bridge/src/cluster_clone_patch.cpp`,
prints `[aiesim-clone] ACTIVE`) **is compiled in and active**. The failures below
are *different* issues, not regressions of that fix.

### 1. Control-packet reconfig routing
- **`ctrl_packet_reconfig_elf`** (chess+peano) — **ABORT** via our PL-egress
  panic guard (`88be997`): `DATA entered shim->PL egress port 9 ... d0=0x00202001
  d1=0xf0000000 ... after 0 TCTs` — classified as *real* data routing to PL on an
  all-NoC NPU (a routing-fidelity bug, not a TCT artifact). Plus invalid AXI-MM
  reads at col=5 (reconfig reading config back). **PASSES on HW + EMU.**
- **`ctrl_packet_reconfig_1x4_cores`** (chess+peano) — **TIMEOUT**: hang
  mid-execution (loaded 146-word reconfig CDO, started 1787 instructions, wedged).
  **PASSES on HW + EMU.**
- **`ctrl_packet_reconfig`, `ctrl_packet_reconfig_4x1_cores`** — broadly broken:
  **FAIL on HW too** (quarantined). Not aiesim-specific.

### 2. DMA completion wedge
- **`dma_task_large_linear`** (peano) — **TIMEOUT**: hung at `Executing 10 NPU
  instructions` for the full 1200 s (472-word CDO, large linear DMA). The transfer
  never signals completion in the aiesim model, so the wait instruction hangs.
  Same family as #76 (S2MM completion under BD reuse), #78 (start-queue overrun),
  #79 (sync-token wedge), for the large-linear case. **PASSES on HW + EMU.**

### 3. xfail labeling
- **`objectfifo_repeat/distribute_repeat`** — aiesim reproduced the same expected
  data mismatch the kernel XFAILs with on HW/EMU, but the verdict path labeled it
  FAIL. **Fixed in `179a1b0`.**

## Next-step plan (FIX-FIRST)

Per Maya: once we know which kernels time out, **fix them** — don't just
skip-list them.

1. **Let the campaign finish**; collect the full aiesim timeout/fail catalog +
   the Phase 5d three-way report.
2. **`--no-trace` triage** on each family: run the kernel in aiesim with
   `--no-trace`. Passes -> trace injection is overwhelming it (fewer-events is the
   fix). Still wedges/panics -> a genuine model gap independent of trace.
   *Bet:* `dma_task_large_linear` is a pure model gap; `_1x4_cores` may be
   trace-sensitive.
3. **Fix the timeout kernels** (priority): the aiesim model gaps —
   DMA-completion for large-linear; control-packet reconfig routing / PL-egress.
4. **Build the fewer-than-8-trace-events option** (trace-prepare / harness): the
   fix for any trace-overwhelm kernels, and independently useful for sweep
   capacity management (some tiles compile with only 4 trace slots).
5. (Last resort) an `aiesim-known-timeout` skip list — only for genuinely
   unfixable gaps, after fix attempts.

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
