# SP-5b Kernel/HW Bring-up -- Design (rev2)

Issue #140, timer-sync faithful-broadcast arc. Execution sub-project of SP-5b:
the measurement-apparatus **software core** is landed+merged (`af8b1208`); this
spec brings up the **kernels and hardware path** that produce the observations
that core consumes. Parent design:
`2026-06-30-sp5b-measurement-apparatus-design.md` (rev2) -- read it first; this
spec inherits its instrument roles, load-bearing assumptions, column frames, and
sign conventions rather than re-deriving them.

**rev2 folds in two adversarial Opus reviews (epistemics + silicon-feasibility).**
The material changes from rev1: `r1_observe` is reshaped from reading the
Start-marker origin (an emulator-only annotation) to the real silicon method --
cross-domain event-pair offset minus emulator `Delta_wall` (epistemics #1); every
HW gate gains a `b`-vector range-0-across-runs reproducibility check, so a gate
can fail on a physically-meaningless measurement and not just on bad geometry
(epistemics #2); the emu inject-and-recover loop runs the real compiled xclbin
in-process (it needs real compute-event pairs, which `propagate_broadcasts` alone
cannot emit -- feasibility #3); R3b-PC's counter **readback** is specified as a
control-packet register-read (`write32` is write-only -- feasibility #1); R3b-PC
floods reduce to `Event_Broadcast`+`Event_Generate` with explicit counter-arm
ordering (feasibility #4); the R1 geometry is a taller **real** spine (the lean
seed has 2 cores, not the claimed 4 -- feasibility #2); and the emu-loop / sign
"pin" are relabeled as emu self-consistency, not silicon validation (epistemics
#3/#4).

**This is alpha hardware-characterization apparatus, not production software.**
SP-5b bring-up produces apparatus that **runs** on Phoenix and provably satisfies
the cross-language contract; it produces **no skew number**. Numbers are SP-5c.

## 0. What is already built (do not rebuild)

The software core (merged) provides, consumed unchanged here:

- `tools/calibration/skew/r1_extract.py` -- `extract_r1(observations, reflected=True)`
  -> `{d_v, intra_core, intra_mem, fit_residual}`. Consumes
  `{"dn_v": int, "kind": str, "origin": float}` dicts. 6/6 synthetic tests.
- `tools/calibration/skew/r3b_extract.py` -- `extract_r3b(observations, reference=0)`
  -> `{d_h, d_v, fit_residual}`. Consumes `{"dn_h": int, "dn_v": int, "r": float}`
  dicts. 5/5 synthetic tests.
- `tools/calibration/skew/_solve.py` -- `solve_design_matrix(A, b, min_rank)`,
  `RankDeficientError`.
- `tools/calibration/skew/schema.py` -- `skew_constants.json` read/write.
- The runtime-override seam on `DeviceState`
  (`broadcast_timing_override` + `effective_broadcast_timing` + setter;
  `src/device/state/effects.rs`), byte-identical when unset (3 neutrality guards
  green), plus the Task-4 surfacing test (`override_origin_offset_surfaces_in_encoded_start`).

The gap this spec closes: **nothing yet produces real observation dicts.** The
extractors are exercised only on synthetic input. Bring-up builds the kernels
that run on silicon and the observation bridges that turn their trace/buffer
output into those dicts -- the "real-trace cross-language contract."

## 1. Scope and non-goals

Three kernels + two observation bridges + one emulator regression loop + HW
runnability gates, sequenced in three phases, each independently landable.

| In SP-5b bring-up | Out (deferred to SP-5c) |
|---|---|
| R1 within-column kernel (= SP-3 gate-carrier) | The numeric Phoenix skew measurement |
| R3b PerfCounter kernel (primary) | R3b `LDA_TM` kernel *unless* Phase-3 go/no-go says go |
| `r1_observe.py` / `r3b_observe.py` (trace/buffer -> dicts) | Flipping `calibrated`; updating the 3 regression guards |
| R1 emu inject-and-recover (plumbing/regression) | Causal-vs-HW validation; P1 round-trip gate |
| Cheap HW runnability/repro gates on built kernels | `intra_mem` reliability determination |
| Exercising the whole pipeline shape end-to-end | Any value assertion / measured constant |

**Non-goals.** No numeric skew measurement. No `calibrated` flip. No engine
changes. **No new emulator features** -- the seam is the only emulator surface and
it is already landed; bring-up touches the emulator *only* by driving that
existing seam and the existing in-process xclbin runner from a new integration
test. The bring-up runs the full pipeline end-to-end but asserts **shape and
non-degeneracy, never values**.

**Design forks resolved** (brainstorm + adversarial review, 2026-06-30):

- **Build order = R1 first.** R1 is *lower*-risk than R3b (it builds the whole
  trace->observation->extractor->gate contract before R3b stresses it with
  hand-authored register programming and off-chip readback) -- but it is **not** a
  drop-in reskin: the lean seed has 2 cores, and R1 needs a taller real spine
  (Sec.4.1).
- **R1 vertical geometry = extend the real dataflow spine** (not idle traced
  tiles). Every traced tile is a real active tile with proven Start emission; no
  untested idle-tile-capture risk. Cost: re-prove Q=0/no-TDR on the taller spine.
- **R1 emu loop = heavy (real xclbin in-process).** The loop needs real
  compute-event pairs (Sec.4.2 method), which `propagate_broadcasts` alone cannot
  emit; it runs the compiled xclbin through the in-process runner and shakes out
  the entire decode+bridge on emu before Phoenix (Sec.4.3).
- **`intra_mem` = best-effort, folded into the range-0 reproducibility check.** R1
  traces the compute-tile mem module via the broadcast reset event (event 122),
  dodging SP-3's DMA-port-event unreliability (confirmed correct: the origin lives
  in the Start marker, not a DMA-port event). The HW gate includes the mem event
  in the range-0 check; if it fails, `intra_mem` is emitted `null`/provisional,
  not a clean number. `d_v` + `intra_core` still land.
- **R3b `LDA_TM` = gated go/no-go after R3b-PC's HW gate.** Build R3b-PC now;
  decide `LDA_TM` with the information PC's bring-up yields. The cross-check's
  epistemic value lands at SP-5c measurement time, and under this gating the TM
  kernel still gets built (if go) well before any `calibrated` flip.

## 2. Route taxonomy note (why R1 and R3b, no R2)

The parent skew-limit doc (`docs/trace/cross-domain-skew-limit.md:246-301`) lists
three routes. **Route 1** (in-domain verification + subtract toolchain latency)
became R1. **Route 2** (round-trip closure) was a *corroborating* idea that bounds
coupling *sums* -- it cannot isolate per-hop `d_h/d_v/intra`, and R3b supersedes it
as an independent cross-check that beats a different wall, so R2 was dropped as
redundant. **Route 3b** (two-source timer read) is the self-synchronizing
realization of the flawed bare Route 3 (whose free-running un-reset timer is
unsynchronized) -- there is no separate "R3a kernel." R1 and R3b are the two
survivors, chosen for independence.

## 3. Column frames (inherited, load-bearing)

Per the parent spec Sec.3: specify all kernel geometry in the **virtualized
frame** (relative cols 0..N-1, all shim-bearing, N<=4; rows 0 shim / 1 memtile /
2-5 core). Firmware relocates the partition onto physical cols 1..N at load; hop
*distances* are relocation-invariant. Never reason about a shim at physical
column 0. R1 is single-column; **R3b needs a >=3-column partition**
(`npu1_3col`/`4col`) for its horizontal axis (Sec.5.1).

## 4. Phase 1 -- R1 within-column (the rails)

### 4.1 Kernel (`mlir-aie/test/npu-xrt/sp5_skew_r1/`)

Seeded from the *pattern* of `mlir-aie/test/npu-xrt/spike_bringup/of_q0_lean.py`
(within-column, pure lock/DMA handshakes, emu-faithful timing, no TDR) -- but that
seed has only **2 cores** (rows 2,3). R1 **extends the dataflow spine to >=3 real
compute cores** (e.g. prod row 2 -> cons row 3 -> cons row 4), staying Q=0, and
**re-proves Q=0/no-TDR/reproducible** on the taller spine (this re-proof is a
first-class task, not assumed from the seed). Geometry in the virtualized frame,
single column:

- **Core modules at rows 2-4 (>=3 real active cores)** -- >=3 collinear
  compute-tile cores at >=3 distinct vertical hop-distances `dn_v`, all of the
  **same kind (`core`)**. This is the load-bearing set for falsifying per-hop
  vertical uniformity: >=3 same-kind points are required (parent Sec.4.5), and we
  do **not** borrow the shim (also `core`-group) as the third point, because that
  would assume the `{core,shim}` grouping SP-5c is meant to validate.
- **memtile (row 1)** and **shim (row 0)** as additional kind/distance anchors
  (`memtile` -> mem group, `shim` -> core group), used for the intra offsets and
  cross-checks -- **not** counted toward the >=3-same-kind-core uniformity set.
- **>=1 compute tile traced on BOTH its core and mem module** -- the intra-offset
  lever. The IRON API yields this by listing a tile twice in the trace config
  (`of_q0_iron_trace.py:46-48` -> core+mem traces; confirmed real, e.g.
  `vec_mul_event_trace/aie.mlir:94-108` traces `type=mem` on `tile(0,2)`); the mem
  trace uses broadcast event 122 (compute-tile memory-module `BROADCAST_15` = 122,
  confirmed).

`REQUIRES: ryzen_ai_npu1`. Traceable via the existing declarative
`configure_trace` / `start_trace` path. IRON Python -- one flood
(`start_broadcast=15`, auto-generated), **no hand-authored `write32`**. Also lands
the never-shipped SP-3 gate-carrier.

### 4.2 Observation bridge (`tools/calibration/skew/r1_observe.py`)

Net-new. **Implements the real silicon R1 method** (parent Sec.4.2), which works
**identically on emu and silicon traces**:

> For a within-column cross-domain event pair `(x in domain A, y in domain B)`
> that fires deterministically together, the trace gives
> `soc(x) - soc(y) = Delta_wall + skew(A, B)`. `r1_observe` isolates
> `skew = measured_offset - Delta_wall`, where `Delta_wall` is supplied from an
> emulator run of the same kernel at **zero constants** (Sec.4.3/4.4).

It emits `{dn_v, kind, origin}` where `origin` = the recovered `skew` in the
reflected convention; `r1_extract` negates it (`reflected=True`). Driven by a
**`geometry.json` shipped in the kernel dir** (source tile; each traced module's
`dn_v`/`kind`; and the co-firing cross-domain event pairs), so nothing is
hardcoded. Fails loud on unknown kind, missing pair, or malformed trace.

**Do not confuse** this with the Start-marker `origin_offset` the emulator injects
(`max_delay - module_delay`): that is the *engine's* `ModelDerived` annotation
(SP-4b/5a), it exists only because the emulator writes it, and on silicon the
Start marker carries no such value. R1's measurement is the event-pair residual
above, not the injected annotation.

**Pair-differences -> per-module origins (the sharpest open modeling detail in
R1).** The physical measurement yields cross-domain *differences*
`skew(A,B) = origin_A - origin_B`, not per-module absolutes -- but the merged
`r1_extract` consumes per-module `origin` relative to `source = 0`. `r1_observe`
reconciles this by pinning the **flood-source module as the zero-reference**
(`origin_source == 0`, physically justified: the reset originates there with
`module_delay = 0`) and resolving each measured module's origin from its
co-firing-pair offsets back to the source. This requires the kernel to emit
co-firing cross-domain pairs that form a **connected chain to the source** -- a
real kernel-design constraint the geometry (Sec.4.1) must satisfy. **Fallback:**
if the chain proves fragile, extract over pairs *directly* with a differencing
solver structurally identical to `r3b_extract`'s reference-differencing (each row
= `[dn_v(A)-dn_v(B), core_ind(A)-core_ind(B), mem_ind(A)-mem_ind(B)]`) -- a small
new companion beside the merged `r1_extract`, not a change to it. The plan
resolves which; the emu loop (Sec.4.3) validates whichever is chosen end-to-end.

Unit-tested against a **frozen decoded-trace fixture** (pair extraction, sign,
kind mapping, `Delta_wall` subtraction, chain-resolution, fail-loud paths).

`geometry.json` schema (non-degenerate example -- >=3 distinct `dn_v` of kind
`core`):

```json
{
  "source": {"col": 0, "row": 0},
  "modules": [
    {"col": 0, "row": 0, "kind": "shim",    "dn_v": 0},
    {"col": 0, "row": 1, "kind": "memtile", "dn_v": 1},
    {"col": 0, "row": 2, "kind": "core",    "dn_v": 2},
    {"col": 0, "row": 2, "kind": "mem",     "dn_v": 2},
    {"col": 0, "row": 3, "kind": "core",    "dn_v": 3},
    {"col": 0, "row": 4, "kind": "core",    "dn_v": 4}
  ],
  "pairs": [
    {"a": {"col": 0, "row": 2, "kind": "core"},
     "b": {"col": 0, "row": 2, "kind": "mem"}, "event": "..."}
  ]
}
```

### 4.3 Emu inject-and-recover loop (Rust, plumbing/regression -- heavy)

Runs the **compiled R1 xclbin through the in-process xclbin runner**
(`src/testing/xclbin_suite.rs`), which produces real compute-event pairs and a
real decodable trace (precedent: `xclbin_suite.rs` runs xclbins in unit tests).
Two runs:

1. **Injected** -- set constants via the seam, run, decode. The co-firing pair
   offsets read `Delta_wall + skew_injected`.
2. **Zero** -- unset override, run, decode. The pair offsets read `Delta_wall`.

`r1_observe(injected_trace, Delta_wall = zero_run_offsets)` -> recovered skew;
assert **recovered == injected exactly**. `Delta_wall` cancels between the two
runs because injected constants touch *only* the timer-reset target and the Start
label, leaving execution byte-identical -- so the differencing is exact. This is
**arithmetic self-consistency, not physics**: it validates the seam ->
timer-reset -> in-process-run -> decode -> `r1_observe` -> `r1_extract` pipeline
end-to-end on emu, shaking out the whole decode+bridge before Phoenix. It
provides **no evidence any silicon number is correct** (that is SP-5c's human
causal-vs-HW gate).

**Sign** is pinned only for **emu self-consistency** (inject in convention X,
recover in X). The **emu<->silicon sign correspondence is an open SP-5c question**
needing a silicon anchor (a known-geometry pair whose sign is independently
predictable); "sign-pinned" here does not mean silicon-validated.

**Clobber ordering (concrete, not a footnote).** `origin_offset` /`reset_target`
are overwritten per-flood, last-writer-wins (`effects.rs:678-679`). The injected
run's own trace-start flood must not clobber the injected override: the override
must be **in force before the run's trace-start flood fires**. The test sets the
override on `DeviceState` before invoking the runner, and asserts (via a watch or
readback) that the armed reset target reflects the injected value at flood time.

### 4.4 HW runnability gate (`build/experiments/sp5-skew/r1_gate.sh`)

Templated off `build/experiments/sp3-spike-trace/task3_gate.sh`, driving
`bridge-trace-runner` on an `mlir-trace-inject.py`-produced xclbin, decoded with
`tools/trace_decoder/`. Run with `env -u XDNA_EMU`. Procedure: N serial Phoenix
runs of the R1 xclbin; one emulator run at zero constants for `Delta_wall`;
`r1_observe(hw_trace, Delta_wall)` per run.

Asserts: rc-0, no TDR/reset in dmesg, reproducible tile/event set across runs,
non-degenerate geometry (>=3 distinct `dn_v` of kind `core`, rank-sufficient),
**and the `b`-vector (recovered per-pair offsets) is range-0 (or within a stated
tolerance) across the N runs** -- the real check on the *measurement*, mirroring
SP-3's 20-run range-0 evidence. **No value assertions** (the range-0 target is a
reproducibility bound, not a skew value). For the **mem module**: the mem event is
included in the range-0 check; if it fails, `intra_mem` is emitted
`null`/provisional (Q2-A). First real silicon test of the within-column trace
geometry + the taller spine's stability.

## 5. Phase 2 -- R3b PerfCounter (the hard instrument)

### 5.1 Kernel (`mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/`)

Hand-authored MLIR (Shape A -- IRON hides register writes). Net-new; **not** a
reuse of SP-3's free-running PERF_CNT anchor (parent Sec.5.3).

- **Two hand-authored floods.** `s1` from corner A, `s2` from a *different*
  corner B, each = `Event_Broadcast{N}_A` + `Event_Generate` on distinct
  broadcast channels with distinct generate events. **No `Timer_Control.Reset_Event`
  on the measured tiles** -- the perf counter (`0x31500`) is a *separate HW unit
  from the timer* (`0x34000`) and counts start-event->stop-event cycles regardless
  of timer state, so R3b-PC needs no timer sync at all. (The
  `Timer_Control`+`Event_Broadcast`+`Event_Generate` "trio" in
  `AIEInsertTraceFlows.cpp:676-723` is R1's *trace-sync* machinery; the
  `Reset_Event` write does not belong in R3b-PC. It is reserved for the TM path,
  Sec.6, where `s2` must **not** reset measured tiles' timers.)
  Namespace note: the generate event id depends on the generating module's event
  space -- shim has only `USER_EVENT_0`=126/`USER_EVENT_1`=127; core/mem
  `USER_EVENT_2`=126. State which module's space each generate uses.
- **Perf-counter per measured tile:** `Performance_Control0` (core `0x31500`;
  `Cnt0_Start_Event` bits 6:0, `Cnt0_Stop_Event` bits 14:8 -- 7-bit fields, event
  122 fits, no whitelist, confirmed) with start = s1's broadcast event, stop =
  s2's. All config via `aiex.npu.write32` at the AM025-DB offsets.
- **Readback (the critical path -- `write32` is write-only).** After the run,
  issue a **control-packet register-read** (`aiex.npu.control_packet` read opcode,
  `AIEX.td:944`, lowered by `AIECtrlPacketToDma.cpp`) of `Performance_Counter0`
  (`0x31520`) on each measured tile; a **net-new readback host** binds and dumps
  the readback BO. This is post-run and non-perturbing (preserving PC's jitter-free
  advantage). Fallback if control-packet read proves unworkable: post-run
  core-`LDA 0x31520` -> store -> DMA out (heavier; core-program shape). Budget the
  readback as the highest-effort R3b task -- the hard part is *readout*, not config.
- **Runtime-sequence ordering (load-bearing):** configure counters on **all**
  measured tiles -> `Event_Generate(s1)` -> `Event_Generate(s2)` -> readback.
  Counter config MUST precede `generate(s1)` or the start event is missed
  (counter reads 0/garbage).
- **Geometry spans BOTH axes** -- >=3 collinear tiles horizontally (for `d_h`,
  needing a >=3-column partition) and >=3 vertically (for `d_v`), per parent
  Sec.5.4.

**`s1`-before-`s2` is sequencing-derived, not route-topology.** The order is
dominated by the program-order gap between the two `Event_Generate` writes (tens
of cycles), so it is satisfiable by the runtime sequence above; `const = (T0_2 -
T0_1)` is identical across tiles within a run and cancels cleanly in `r_X - r_Y`.

**Why cross-column is fine here but not for R1:** R3b reads a **local
single-clock interval** (one tile's perf counter), immune to the ~30-cycle
cross-column trace-crossing jitter that forced R1 within-column. That immunity is
why R3b -- not R1 -- is primary for `d_h`.

### 5.2 Observation bridge (`tools/calibration/skew/r3b_observe.py`)

Net-new; **shared by both R3b kernels** (PC and TM produce the same
`{dn_h, dn_v, r}` contract). Reads the control-packet readback buffer, maps each
tile to `{dn_h, dn_v}` via a `geometry.json` in the kernel dir, emits the dicts
`r3b_extract` consumes. Fails loud on malformed/short buffer. Unit-tested against
a **frozen readback-buffer fixture**.

### 5.3 No emu loop

R3b is emulator-independent by design (parent Sec.5.4); an in-emu loop would be
circular. The emulator already handles two independent floods and a
distinct-channel `s2` does not trip the ch15 single-source guard
(`effects.rs:633`), so **no emulator work is needed** (parent Sec.5.3). The rank-2
solve is already synthetic-tested (5/5).

### 5.4 HW runnability gate (`build/experiments/sp5-skew/r3b_pc_gate.sh`)

N serial runs: rc-0, no TDR, counters non-degenerate and rank-sufficient across
both axes, an explicit **`s1`-before-`s2` non-inversion check** (zero/garbage
counter on any tile = inversion, flagged -- this catches *inversion only*, not
interval correctness), **and the counter `b`-vector is range-0 across the N runs**
(a systematically-contaminated-but-nonzero counter is often non-reproducible, so
range-0 is the real correctness proxy the non-inversion check cannot give).
Physical sanity bound (interval within the plausible `const + hop*d` envelope) if
feasible. First real silicon test of the two-flood + perf-counter config +
control-packet readback. **No value assertions.**

## 6. Phase 3 -- R3b `LDA_TM` go/no-go

An explicit checkpoint with the human after Phase 2's HW gate, informed by: did
PC bring up clean, is the two-flood + readback config solid, how much Phoenix
runway remains.

- **If go:** `mlir-aie/test/npu-xrt/sp5_skew_r3b_tm/` -- core event-triggered on
  `s2`, executes `__builtin_aiev2_read_tm` (`IntrinsicsAIE2.td:469`) -> `LDA_TM`
  -> store -> DMA out. Here `s2` **must not** reset the measured tiles' timers
  (the timer-reset caveat lives here, not in R3b-PC). Reuses `r3b_observe` (same
  `{dn_h, dn_v, r}` contract). Own HW gate (same range-0 discipline). **Must never
  enter the R1 emu loop** (parent Sec.5.2 -- it reads its own timer, breaking
  `Delta_wall` cancellation).
- **If no-go:** `LDA_TM` deferred to SP-5c-if-needed; R3b-PC stands as the single
  R3b instrument, documented.

## 7. Testing tiers

| Tier | Status |
|---|---|
| 1. Synthetic extractor unit tests | **Built** (11/11: R1 6, R3b 5) |
| 2. Runtime-override seam neutrality (Rust) | **Built** (software core) |
| 3. Observation-bridge unit tests (`r1_observe` event-pair/`Delta_wall`, `r3b_observe`) vs frozen fixtures | **New** |
| 4. R1 emu inject-and-recover (real xclbin in-process; recovered == injected exactly) | **New** |
| 5. HW runnability gates (R1, R3b-PC, R3b-TM-if-go) -- incl. `b`-vector range-0 | **New** |

`cargo test --lib` stays green throughout -- the seam is unset by default, so
behavior is byte-identical and the 3 neutrality guards hold.

## 8. Error handling (consolidated)

- Rank-deficient / degenerate geometry -> `RankDeficientError` (built) -> gate
  fails loud.
- Observation bridges fail loud on unknown kind, missing pair/module, malformed
  buffer/trace.
- HW gates fail on rc != 0, TDR, missing tiles/events, degenerate output,
  `b`-vector **not** range-0 across runs, or (R3b) `s1`-before-`s2` inversion.
- Emu loop asserts exact recovery; clobber-ordering asserted at flood time.

## 9. File inventory

Net-new:

- `mlir-aie/test/npu-xrt/sp5_skew_r1/` -- kernel (taller spine) + `geometry.json`
  + host + `REQUIRES`
- `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/` -- kernel + `geometry.json` +
  control-packet readback host
- `mlir-aie/test/npu-xrt/sp5_skew_r3b_tm/` -- *if Phase-3 go*
- `tools/calibration/skew/r1_observe.py` (event-pair + `Delta_wall`) + test
- `tools/calibration/skew/r3b_observe.py` + test
- `build/experiments/sp5-skew/{r1,r3b_pc,r3b_tm}_gate.sh`
- Rust emu inject-and-recover integration test (drives `xclbin_suite.rs`)
- A compiled R1 xclbin test fixture (for the in-process emu loop)

Consumed unchanged: `r1_extract`, `r3b_extract`, `_solve`, `schema`, the seam,
`xclbin_suite.rs`, `trace_decoder`.

Note: the `sp5_skew_*` dirs are **experiment fixtures consumed by the gate
scripts** (`bridge-trace-runner` + `mlir-trace-inject.py`), not
`emu-bridge-test.sh` lit-discovered tests; `REQUIRES` is documentary here.

## 10. Risks

Bring-up-specific (adversarial-review priorities):

- **[R3b-PC, critical path] Counter readback.** `write32` cannot read; the
  control-packet register-read + net-new readback host is the highest-effort task
  and gates the *primary* instrument. Fallback: core-`LDA` readback.
- **[R1] Taller-spine no-TDR re-proof.** The >=3-core spine is net-new dataflow;
  Q=0/no-TDR/reproducibility must be re-proven, not inherited from the 2-core seed.
- **[R1] Emu-loop in-process plumbing.** Running the compiled xclbin through
  `xclbin_suite.rs` + decoding + `r1_observe` is net-new test wiring; the clobber
  ordering (Sec.4.3) is load-bearing.
- **[R1] `intra_mem` traceability.** Q2-A tries the broadcast-event path and folds
  the mem event into the range-0 check; may prove flaky -> `intra_mem` provisional,
  documented for SP-5c.
- **[R3b] Two-flood config + counter-arm ordering + `s1`-before-`s2`** all untested
  on Phoenix. The HW gate (range-0 + non-inversion) is the check; `LDA_TM` is the
  gated fallback.
- **[both] Structure-vs-values.** >=3 collinear same-kind tiles per axis or neither
  instrument can falsify per-hop non-uniformity. Baked into Sec.4.1/5.1.
- **[R1] Sign correspondence emu<->silicon** is an open SP-5c question (Sec.4.3);
  the emu loop pins only self-consistency.

## 11. SP-5b -> SP-5c handoff

SP-5b bring-up ships apparatus that (a) provably recovers known constants through
the **real event-pair/`Delta_wall` bridge** on emu (Sec.4.3, in-process xclbin),
and (b) provably runs on silicon with reproducible (range-0) output (Sec.4.4/5.4).
**No measured number is produced, and SP-5b provides no evidence any measured
number is correct** -- correctness is entirely SP-5c's human causal-vs-HW gate.

SP-5c (Phoenix): runs the apparatus, reads `d_h/d_v` (R3b primary) and
`d_v`/intra (R1, cross-checking R3b's `d_v`), resolves the structure questions
(per-hop uniformity via the >=3-collinear residual; whether `d_h/d_v` collapse;
intra-tile sign; **the emu<->silicon sign correspondence**), writes
`skew_constants.json`, loads it via the runtime-override seam (then bakes into
archspec for the permanent flip), flips `calibrated`, updates the three regression
guards -- verified locations (parent Sec.9):
`crates/xdna-archspec/src/runtime.rs:807`,
`src/interpreter/engine/coordinator.rs:4078`, `src/device/state/effects.rs:1355`
-- and validates causal-vs-HW.

**Deferred to SP-5c** (carried from the software-core final review + this
bring-up): the 3 software-core minors (fail-loud unknown-kind -- now partly covered
by the observation bridges; per-axis `min_rank` identifiability; double
`matrix_rank` polish), `intra_mem` reliability (Q2-A), sign correspondence
(Sec.4.3), R3b-`LDA_TM`-if-no-go.

## 12. Reference map

- Parent design (roles, assumptions, frames, sign):
  `2026-06-30-sp5b-measurement-apparatus-design.md`
- Epistemic boundary + routes: `docs/trace/cross-domain-skew-limit.md`
- SP-3 substrate (untracked): `mlir-aie/test/npu-xrt/spike_bringup/of_q0_lean.py`
  (2-core seed), `build/experiments/sp3-spike-trace/` (README, SP4A-HW-TARGETS.md,
  TASK3-RESULT.md, `task3_gate.sh` + `task3_tally.py`)
- Kernel authoring: `mlir-aie/test/npu-xrt/vec_mul_event_trace/aie.mlir` (Shape A,
  core+mem trace), `of_q0_iron_trace.py` (Shape B trace),
  `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp:676-723` (write32 flood
  template; `:519-576` per-tile timer-sync receive)
- Readback: `include/aie/Dialect/AIEX/IR/AIEX.td:944` (`control_packet` read),
  `lib/Dialect/AIEX/Transforms/AIECtrlPacketToDma.cpp`
- Register DB: `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
  (`Performance_Control0` `Cnt0_Start_Event` 6:0 / `Cnt0_Stop_Event` 14:8,
  `Performance_Counter0` `0x31520`, `Timer_Control` `0x34000`);
  events `mlir-aie/{build,install}/lib/regdb/events_database.json`
  (`BROADCAST_15`=122 core/mem, `USER_EVENT_2`=126 core/mem, shim
  `USER_EVENT_0`=126/`USER_EVENT_1`=127)
- In-process runner + decode: `src/testing/xclbin_suite.rs`, `tools/trace_decoder/`,
  `tools/mlir-trace-inject.py`
- Software core (built): `tools/calibration/skew/*`, the seam in
  `src/device/state/effects.rs`
- Bridge harness (dir layout, `REQUIRES`, dual-compiler reference only):
  `scripts/emu-bridge-test.sh`
