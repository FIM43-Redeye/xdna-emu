# 2026-04-25 Session Summary

Resumption of the paused refactor + cycle-budget validation work that
sat behind the in-tree trace decoder. Threads B, A, C, D walked
through; net 6 commits.

## Commits this session

| Commit | Thread | Net effect |
|--------|--------|------------|
| `d026e40` | **B** | parse-trace.py default → in-tree decoder; lazy mlir-aie import; docs |
| `daed9a7` | **A** | 3 findings docs (A.1/A.3/A.4/A.5 closed); A.2 deferred |
| `bfe3720` | **C** | 3 findings docs (C.1/C.2/C.4 scoped); C.3/#6 debugfs-blocked |
| `66be355` | **D-1** | TABLEGEN_210_PREFIX (D.6); stale-files cleanup (D.2); columns+1 overflow guard (D.7) |
| `07128a2` | **D-2** | archspec dead-code (D.1); arch-generic MemoryRegion (D.5); memtile/shim DMA promotion (D.4); cascade-flows finding correction |
| (pending) | **D-3** | D.3 design doc (implementation deferred to its own brainstorm) |

`cargo test --lib` green at 2755 / 5 ignored throughout. No bridge test
regressions touched (last full run: cascade_flows + ctrl_packet_reconfig
+ vector_scalar_using_dma all behave as documented in the findings).

## What's open

### A.2 -- thin defaults + sweep + PC-anchored joining
Biggest validation-strategy reframe (user's reframe 2026-04-25).
Concept: thin the default 8-event set down to grounding-only
(INSTR_EVENT_0/1, possibly PERF_CNT_0 if perfcnt-enable plumbing
lands), sweep the remaining slots to cover all events, join
HW vs EMU per (event, PC) using mode-1 (EVENT_PC) traces.

Why deferred: substantial rethink that needs its own
brainstorm -> writing-plans -> execute cycle. Prerequisites done
this session: in-tree decoder authoritative, cascade-flows finding
correctly framed, cycle accumulator confirmed deterministic.

Pickup: brainstorm skill on "thin grounding-event set + per-event sweep
+ PC-anchored joining for cycle-diff validation."

### C.3 + lead #6 -- ftrace + FW log/trace rings
Both gated on debugfs being mounted, which is gated on the kernel
rebuild the user is waiting on. Pick up after the rebuild lands.

Pickup: lead #4 says subscribe to xdna_job + mbox_* tracepoints;
lead #6 says read `dump_fw_log_buffer` / `dump_fw_trace_buffer`.
`docs/observability-leads.md` has the full surface.

### D.3 -- extend DeviceOp to non-CDO write paths
Largest D item. Touches 9+ call sites in coordinator, executor,
control-packet dispatch with subtle side-effect ordering risk
(tile-local effects, broadcast propagation). Subsystem 8 audit
flagged it as audit-first refactor scope.

Design doc landed today at
`docs/superpowers/specs/2026-04-25-d3-deviceop-universal-design.md`
covering: problem, three complications (tile-local side effects,
subsystem dispatch, lowering helper visibility/placement),
recommended Option A vs B path (Option A first, with rationale),
8-step migration plan, acceptance criteria.

Pickup: read the design doc + the Subsystem 8 audit's
"Load-bearing finding: non-CDO write paths still bypass DeviceOp"
section; run brainstorming on Option A vs B; produce the plan.

### D.8 -- master merge milestone
Process step. Suggested: tag `phase1-complete` after D.3 lands, then
decide squash- or fast-forward-merge dev → master at user discretion.

Optional: also tag the in-tree-decoder + observability-design point
as a way-station, since the cycle-budget validation pipeline is
materially different from where master sits.

## Bridge-trace-runner ctrlpkt protocol

Surfaced during A.4: bridge-trace-runner doesn't replicate test.exe's
ctrlpkt-then-run-seq submission protocol, so ctrl_packet_reconfig
fails on both HW and EMU through the bridge path while passing
under direct test.exe. The classifier subsystem in
`libxdna_emu.so` doesn't signal "ctrlpkt arg present, must be
submitted as separate run."

Documented in
`docs/superpowers/findings/2026-04-25-ctrl-packet-reconfig-bridge-runner.md`
as a real bridge-runner bug. Not blocking the validation pipeline
(other tests give us the cycle-diff signal). Worth fixing when the
runner gets another iteration cycle.

## Surprising findings worth re-reading later

1. **EMU_TRACE_BUG on cascade_flows was already resolved** by
   commit 13beeff (broader event set, 4/23 15:08) -- 35 minutes
   after the Phase E run that recorded it. Reframed correction in
   `docs/superpowers/findings/2026-04-25-cascade-flows-emu-trace-bug.md`
   (initial draft incorrectly attributed "short trace window" to
   wallclock outpacing; broadcasts are runtime-stream-fired, not
   host-wallclock-fired, so window boundaries match -- the actual
   divergence is on which events fire inside the window).

2. **Cycle accumulator is bit-deterministic.** cascade_flows EMU
   ran cycles=2256 exactly across two consecutive runs.
   EMU_SECONDS_PER_CYCLE is for wallclock-timeout calculation only;
   cycle-diff is dimensionless. Findings:
   `docs/superpowers/findings/2026-04-25-cycle-accumulator-status.md`.

3. **Cycle-span scalar misleads when event sets disagree.**
   cascade_flows DRIFT(ratio=0.01, diverge=7) reflects HW firing
   INSTR_VECTOR×2 + LOCK_STALL×2 while EMU fires
   MEMORY_STALL×3 + INSTR_LOCK_*×2 -- different event sets, so
   `max(ts) - min(ts)` measures different things. PC-anchored joining
   (A.2) is the right fix.

4. **D.6 was misnamed.** NEXT-STEPS called this LLVM_SYS_210_PREFIX
   (the llvm-sys crate convention); the tblgen-rs fork actually
   reads TABLEGEN_<MAJOR>0_PREFIX. Set in `.cargo/config.toml`
   relative-to-config-file, so no PATH dance needed.

5. **column_specific is correctly NO_CORE today** (DMA-only test;
   no aie.core). Phase E doc's "trace-inject silently skipped" was
   stale; the inject now runs and produces traced MLIR -- there's
   just no core for the trace to monitor. Updated inline.

## File map for the next session

**Read first:**
- This file (session summary).
- `docs/superpowers/specs/2026-04-25-d3-deviceop-universal-design.md` (D.3 plan).
- `docs/observability-leads.md` (C.3, C.4, lead #6 — the trace-sweep action priorities at the end).
- `NEXT-STEPS.md` (refactor recovery doc — needs an update to reflect today's hygiene).

**Today's findings (per-thread):**
- `docs/superpowers/findings/2026-04-25-cascade-flows-emu-trace-bug.md`
- `docs/superpowers/findings/2026-04-25-cycle-accumulator-status.md`
- `docs/superpowers/findings/2026-04-25-ctrl-packet-reconfig-bridge-runner.md`
- `docs/superpowers/findings/2026-04-25-event-bounded-trace-summary.md`
- `docs/superpowers/findings/2026-04-25-perfcnt-sidecar-design.md`
- `docs/superpowers/findings/2026-04-25-hwctx-query-design.md`

**Updated inline:**
- `docs/superpowers/plans/2026-04-23-phase-e-validation.md` (4 entries marked resolved/reframed)
- `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md` (deferred trace-comparison replacement marked landed)
