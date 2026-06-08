# aiesim Failure Triage — three-way timing campaign

**Status:** RESOLVED for the non-PASS set (2026-06-07). Re-verify on the current
`.so` cleared ~13 stale verdicts; the two genuine fast FAILs are root-caused (one
fixed, one characterized + tied to a device-model gap). One new FAIL surfaced
(`init_values_repeat`, open). See the **2026-06-07 resolution** section directly
below; the campaign-era catalog further down is retained as history.
**Companion:** [trace-event-sweep-campaign.md](trace-event-sweep-campaign.md)

## 2026-06-07 resolution (re-verify on current `.so` + the two fast FAILs)

The campaign verdicts (`20260606/`) were captured on an older `.so`/kernels. A
focused **re-verify** of the 17 non-PASS (test×compiler) pairs on the current
`.so` (shipping defaults: settle 4096, AIESIM_TIMEOUT 3000) — aiesim-only,
`build/bridge-test-results/20260607-reverify/` — separated stale from genuine:

- **~13 stale → now PASS/XFAIL.** Every campaign TIMEOUT in the set now PASSes:
  `ctrl_packet_reconfig{,_1x4_cores,_4x1_cores}`, `dma_task_large_linear`,
  `nd_memcpy_linear_repeat`, `objectfifo_repeat/compute_repeat`,
  `packet_flow_fanin` (all slow-not-wedged — confirms the #90 classification end
  to end). `objectfifo_repeat/distribute_repeat` → **XFAIL** (the `179a1b0` fix
  now applies on re-run, as predicted).
- **`ctrl_packet_reconfig_elf` (chess+peano) — RESOLVED (was the family-A FAIL).**
  Root cause: the panic data (`d0=0x00202001` = a stream packet header,
  pkt_type=2; `AIETargetNPU.cpp:324`) is **harness-injected TRACE egress**, not
  real data. VCD (`sShimTrace → mSouth3 → shim_to_pl3`) confirms it's the shim
  trace port. It lands on PL because the shim **demux** South-selector (0x1F004)
  is still at its PL **reset-default (00)** when the first trace beats arrive: the
  runtime's South3→DMA demux write (`XAie_EnableAieToShimDmaStrmPort`, value
  `0x40/0x50`) lands at **18827ns**, but the trace beats reach the shim at
  **18790ns** — a 37ns startup-ordering race in the cluster's timing. **The demux
  write IS present and the bridge routes it faithfully** — not a bridge gap, not a
  device-JSON flaw (`demux_dma_streamids=[2,3]` is correct topology), not missing
  driver replication. The result data still travels the shim-DMA aximm path, so
  the PL beats are redundant. **Fix:** PL-egress panic demoted to **drain+warn**
  by default; strict abort is now opt-in via `XDNA_AIESIM_PL_PANIC=1` (was: panic
  default, `XDNA_AIESIM_PL_DRAIN` to opt out). reconfig_elf now PASSes on aiesim
  with default env (one drain WARNING, then `PASS!`). TCT-on-PL path unchanged.
  Standing diagnostic added: `XDNA_AIESIM_MUXLOG=1` logs shim mux/demux
  (0x1F000/0x1F004) writes.
- **`vec_mul_trace_distribute_lateral` (peano) — ROOT-CAUSED (#93, 2026-06-07).**
  Compute PASSes; only the trace-buffer check fails (empty buffer → `parse_trace`
  JSONDecodeError → FAIL). Precise mechanism, from a fresh full-array VCD
  (`/tmp/.../distlat.vcd`, analyzed per-tile/per-direction):
  - The trace-start **does** fire. The kernel issues `Event_Generate=127` on a
    shim; the model honors it — `shim.tile_2_0.event_trace.event127_user_event_1`
    goes high @2237ns, and one tick later that shim drives its broadcast bus to
    **bit-15** (`event_broadcast_a.*_m @2238ns`). So "Event_Generate not modeled"
    is **refuted**.
  - Broadcast-15 **floods the entire shim row east-west** — reaches every shim
    column (col2@2238 → col1@2240 → col0@2242, eastward to col3/col4), on both the
    `_a` and `_b` networks. The horizontal broadcast network works.
  - But it **never climbs north out of row 0.** Every memtile (`mem_row.tile_X_1`)
    and every array tile (`tile_X_{2..7}`) broadcast bus is flat; even where a shim
    asserts its `north_m` with bit-15 (col2), the memtile directly above never
    receives it. **The shim→memtile vertical broadcast link is unwired in the
    cluster model.** Hence no memtile/core sees broadcast-15, every
    `event_trace.state` stays flat-0 array-wide, no trace packets are produced.
  - This **corrects** the earlier coarse note ("broadcast network flat"): the
    broadcast is alive in the shim row; it just cannot cross the shim/array
    boundary. Same *class* as the PL-egress finding — generic-Versal cluster
    internal connectivity ≠ XDNA.
  - **Answer to Maya's device-file question: NO.** The decrypted device JSON
    (`NPU1.json`) describes the event *register map* (offsets like
    `ctrl_event_generate=0x008`, `ctrl_event_broadcast_block_{n,s,e,w}`), event-ID
    enums, stream-switch connectivity (`DeviceConnections`), and the ME↔PL **event
    boundary** ports (`ME_PL_M_EVENTS`→plsink, `ME_PL_S_EVENTS`←0). It has **no
    internal tile-to-tile broadcast adjacency** — that wiring is structural in the
    closed `libaie2_cluster_msm`. Nothing in the device-file schema controls the
    missing shim→array hop. (Also: trace-start uses **no** `Event_Broadcast_Block`
    writes — broadcast is meant to flood freely — so it's not a blocked-route or
    config issue either.)
  - **Scope of the limitation:** EVERY standard mlir-aie trace setup starts the
    trace unit on broadcast-15 generated at a shim (row 0) and consumed in the
    array (row ≥ 2). Since that vertical hop is severed, **aiesim cannot oracle
    hardware trace for any such kernel** — distribute_lateral isn't special, it's
    just the one kernel that *validates* trace contents and thus FAILs visibly;
    other trace kernels "pass" only because they never check the buffer.
  - **Secondary observation (separate, non-gating):** `Event_Generate` was
    addressed to col1 (`shim_noc_tile_1_0`) but `user_event_1` fired on col2's
    shim — an apparent 1-column offset in the shim address remap. It does **not**
    gate trace (broadcast floods all columns anyway), but it's worth a separate
    look. Flagged, not chased here.
  - **Disposition (UPDATED 2026-06-07, supersedes the "not pursued" note below):**
    a local broadcast-bridge **was built and committed** (`7d93a83`, #97). It does
    **not** fabricate fidelity — it completes each dropped inter-sub-model seam with
    the model's **own real broadcast values** (shim combined value -> memtile per
    column; memtile -> compute vertical flood, each EB read from its wired south
    wire, rippling 1 row/posedge to match HW timing). Result: all 20 compute tiles
    flood, core+mem trace units fire bcast-15, `trace.txt` 0->18719 B, test PASS.
    **HW-validated** (`881d238`): same peano xclbin on real NPU1 — compute region
    cycle-EXACT (12297 ns/invocation), 32-event deterministic order byte-identical;
    only stall micro-timing diverges (an aiesim-ISS limit, decomposed in
    `docs/known-fidelity-gaps.md`). Gated `XDNA_AIESIM_BCAST_BRIDGE`, LOCAL-ONLY,
    fail-safe; cluster lib never modified on disk. So aiesim **is** now a trace
    *structure/flow* oracle for this kernel class; HW remains the stall-timing
    oracle. (Original characterize-and-accept reasoning preserved below for the
    record.) ~ A local broadcast-bridge patch was originally judged a "much deeper
    lift... not pursued absent a Maya decision" — that decision was taken and the
    lift completed.
- **`objectfifo_repeat/init_values_repeat` (peano) — NEW open FAIL.** Surfaced by
  the re-verify (campaign had it as TIMEOUT). Fails fast (log ends after 5 NPU
  instructions; no PL-PANIC, no Sync-timeout, no quiescent-wedge) — a distinct
  early-stop, NOT a settle wedge and NOT the trace-on-PL panic. #90 noted it needs
  settle≥16384, but the observed mode doesn't match a settle wedge. **Needs
  dedicated investigation** (deferred — separate from the two fast FAILs).
- **PL routing on HW** is a deeper standing item (Maya: "been acting weird for a
  while") — to revisit; the drain+warn default lets aiesim runs proceed meanwhile.

---

## Campaign-era catalog (HISTORY — 2026-06-06/07, older `.so`)

The all-72 three-way (HW / interp / aiesim) timing campaign (#85) surfaced a set
of aiesim failures. This doc catalogs them, their failure modes, and the
fix-first plan. **None come from this session's harness fixes** (B-full and the
budget-cap removal both touch only the HW/EMU path; the aiesim verdict path was
untouched except the xfail-labeling fix). *Superseded by the 2026-06-07
resolution above — retained for the per-kernel failure-mode notes.*

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

## Classification RESULT (`20260607-ratelog`, sim backstop off, 2400s cap)

Re-running the 6 genuine-timeout kernels with `XDNA_AIESIM_POLL_MAX_NS=0` and a
2400s wall cap **resolved 5 of 7 as merely slow** (Maya's instinct was right):

| Kernel | Campaign | Reclassified |
|--------|----------|--------------|
| `ctrl_packet_reconfig_1x4_cores` (chess/peano) | TIMEOUT | **PASS** @ 1098 / 1119s |
| `dma_task_large_linear` (peano) | TIMEOUT | **PASS** @ 1527s |
| `nd_memcpy_linear_repeat` (peano) | TIMEOUT | **PASS** @ 935s |
| `packet_flow_fanin` (chess/peano) | TIMEOUT | **PASS** @ 697 / 915s |
| `objectfifo_repeat/compute_repeat` (peano) | TIMEOUT | quiescent wedge (txns stall @64) |
| `objectfifo_repeat/init_values_repeat` (peano) | TIMEOUT | quiescent wedge (txns=0) |

Heartbeat ratio ~5000 ms_wall/µs_sim — at 1200s wall the sim has advanced only
~240µs, so these are slow-but-progressing, not stuck.

**CORRECTION:** family B above ("`dma_task_large_linear` — never signals
completion") is **wrong** — it completes at 1527s. The transfer is fine; it was
just slower than the 1200s cutoff. Not a model gap.

The two `objectfifo_repeat` wedges trip at ~131µs sim — suspiciously close to the
default 512-quanta settle window (~131µs). Disambiguation
(`20260607-settlecheck`, `XDNA_AIESIM_SETTLE_QUANTA=16384`): **both PASS**
(`compute_repeat` @ 609.7s, `init_values_repeat` @ 1828.8s). So they were
**settle-window FALSE-TRIPS, not real wedges.**

## FINAL verdict: ZERO genuine aiesim model wedges in the timeout set

All 7 campaign "timeouts" were **our own instrumentation defaults**, not aiesim
model gaps:

- **5 = `AIESIM_TIMEOUT` (1200s) too short.** They pass with budget
  (`dma_task_large_linear` @1527s, `init_values_repeat` @1829s low-contention).
- **2 = `settle_quanta` (512) too short** for repeat-mode inter-iteration DMA
  gaps. Measured max legitimate idle gap = **512 quanta** (`init_values_repeat`)
  / 495 (`compute_repeat`) — i.e. the default settle window sits exactly *at* the
  legitimate gap, so it borderline-false-trips a still-live transfer as wedged.

The aiesim oracle is therefore **more faithful than the campaign suggested** —
the control-read clone patch and all prior DMA/TCT fixes hold; the remaining
"failures" were measurement artifacts. The only genuinely-failing kernels are the
two fast FAILs (next section).

### Two instrumentation fixes (derived, not magic numbers) — APPLIED
1. **`settle_quanta` default 512 → 4096** (`npu_replay.cpp`). Exceeds the measured
   ~512-quanta legitimate gap by 8×. Tension: at ~5000 ms_wall/µs_sim a larger
   settle window means a *real* wedge burns `settle × ratio` wall before it's
   flagged (512→~675s, 4096→~5250s) — accepted for false-trip safety.
2. **`AIESIM_TIMEOUT` default 1200 → 3000** (`emu-bridge-test.sh`).
   `init_values_repeat` needed 1829s low-contention (2373s under the 9-job
   `--no-trace` arm); 3000s covers it with headroom.

### Trace tax is ZERO (refutes the overwhelm hypothesis for wall-clock)

`--no-trace` arm (`20260607-notrace-arm`) vs trace-on (`20260607-ratelog`),
compared on **final sim-time** (deterministic, contention-free) for the slow
cluster:

| Kernel | sim ON | sim OFF | tax |
|--------|--------|---------|-----|
| ctrl_packet_reconfig_1x4_cores.chess | 124.552µs | 124.552µs | +0.0% |
| dma_task_large_linear.peano | 252.035µs | 252.035µs | +0.0% |
| nd_memcpy_linear_repeat.peano | 100.531µs | 100.531µs | +0.0% |
| packet_flow_fanin.peano | 96.730µs | 96.669µs | +0.1% |

Byte-identical sim-time: trace events are observational (side-DMA into a trace
buffer; they don't gate compute), so they add no cycles to the critical path.
**`--no-trace` is not a lever for campaign wall-clock** — the slowness is
intrinsic to cycle-accurate sim of each kernel's cycle count. The two default
fixes are correct and sufficient; a fewer-events option is not needed for
performance (may still matter for sweep trace-slot capacity — separate concern).

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
