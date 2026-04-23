# Phase E Task 6 Findings — Trace-Compare Landmines & Real Drift Signal

Date: 2026-04-23

Context: We finished Phase E Tasks 2–6 in one sitting. Task 6 ("invoke
trace-compare and capture the report") committed cleanly, but analyzing the
resulting report surfaced three things worth pinning down before Task 7
(classification) and before anyone else touches `trace-compare`. This note
captures them.

> **UPDATE (2026-04-23, later same day):** Finding #3's "254 115-cycle HW
> startup anomaly" turned out to be a bug in our **own** Rust trace decoder,
> not a real HW behavior. Our `decode_per_tile` decoded each 32-byte packet
> as if its timer started at 0 (unless a Start token reset it), which
> manufactured a phantom gap whenever a second packet continued the timer
> stream without a Start token. mlir-aie's `parse_trace` handles packet
> continuity correctly. We've since migrated the whole pipeline onto
> mlir-aie's decoder via `tools/parse-trace.py`; `trace-compare` now reads
> the events JSON that script emits rather than re-decoding binaries in
> Rust. The real drift on `vector_scalar_using_dma` is **+2 cycles per
> iteration, Stable, 0 diverged, 0 count mismatch** — essentially perfect.
> The rest of the findings below are preserved as-written for history;
> treat the "big startup anomaly" narrative as a debugging artifact.
> Relevant commits: migration from in-Rust decoder to events-JSON
> ingestion, new `tools/parse-trace.py`, deletion of `trace-to-cycles.py`
> in favor of a single-pass decoder.

## Commits landed in this session (for cold re-entry)

```
6588a62  feat(bridge-test): run trace-compare with --remap-columns
09741a7  fix(trace-compare): --extended no longer clobbers --remap-columns
e699c9f  feat(bridge-test): run trace-compare between HW and EMU bins        [Task 6]
098d00d  feat(bridge-test): capture EMU trace + cycles under --with-cycle-diff [Task 5]
5e54869  refactor(bridge-test): unify HW/EMU trace-cycles helper (side param) [Task 4]
5719904  feat(bridge-test): add --with-cycle-diff flag scaffolding            [Task 3]
798c087  refactor(bridge-test): rename HW trace bin to trace_hw.<variant>.bin [Task 2]
895a632  fix(trace-inject): target device with runtime_sequence, not first   [bonus — unrelated]
f1182d6  fix(trace): don't emit unit events for NOP slots                     [Bug 2]
f5789f6  fix(trace): thread real simulation cycle through event-unit writes   [Bug 1]
2edcba9  feat(plugin): profile-named libxdna_emu_{debug,release}.so install   [plugin]
```

Tasks completed: 1, 2, 3, 4, 5, 6.  Pending: 7 through 14.  Phase E plan doc
is at `docs/superpowers/plans/2026-04-23-phase-e-trace-diff-cycle-budget.md`.

---

## Finding #1 — Trace-compare CLI had an order-dependent flag clobber

**Symptom.** Invoked

```
target/release/trace-compare --hw ... --emu ... --remap-columns --stalls --extended -o report.txt
```

and got an un-remapped report. Two tiles, `(0,2)` and `(1,2)`, despite HW
only emitting on col=1 and EMU only on col=0.

**Root cause.** The `--extended` handler in `src/bin/trace_compare.rs` was
doing `opts = AnalysisOptions::extended();`, which *replaces* the whole
struct with fresh defaults. Any flag parsed earlier on the same line
(notably `--remap-columns`) got wiped.

**Fix.** Commit `09741a7` — OR-in the extended bits (`iterations`,
`stalls`, `cross_tile`) instead of replacing. CLI flags are now
order-independent.

**Lesson.** If you see a setter that *replaces* a struct rather than
mutating fields, and the struct is driven by CLI args, suspect clobber.
This is the second time we've been bitten by state-wipe in a CLI handler
this season; worth codifying somewhere as a pattern to watch for.

**Debug artifact that pinned this down.**  Adding an `eprintln!` printed
`opts.remap_columns=true` after `--remap-columns` alone, and
`opts.remap_columns=false` after `--remap-columns --extended`.  Clean
smoking gun.

---

## Finding #2 — Column remap works, but only matters when the two sides have disjoint physical columns

**Symptom.** HW binary reports all events at tile **(1, 2)**; EMU binary
reports them at tile **(0, 2)**. Same kernel. Same row. Different column
tuple. `trace-compare` without `--remap-columns` treats them as entirely
separate tiles.

**Why.** HW schedulers (xdna-driver) pick `start_col=1` or higher for
`vector_scalar_using_dma` on this NPU. The emulator's `TileArray` is
0-indexed and doesn't know about `start_col`; the kernel always runs on
col=0.

**Existing `--remap-columns` implementation
(`src/trace/compare.rs:660`).** Per side, collect the sorted set of
columns actually present in the trace and renumber them to 0, 1, 2, ….
HW: `{1} → {1→0}`; EMU: `{0} → {0→0}`. Both sides end up at `(0, 2, 0)`.
Events pair.

**When it quietly fails.** The current algorithm only looks at columns
that carry at least one packet. For our case that's fine. Two scenarios
to watch later:

- **Empty-column pollution.** If HW emits packets on both col=0 (e.g.
  control tile metadata) and col=1 (real kernel), the remap produces
  `{0→0, 1→1}` — identity — and the kernel events stay on col=1. If EMU
  only emits on col=0, pairing still fails. We dodged this because the
  ctrl tile on row 0 uses a *different pkt_type*, so the `TileKey`
  doesn't collide. Not guaranteed for other kernels.
- **Distinct physical column ranges.** HW uses cols 3–4, EMU uses cols
  0–1. Remap gives both sides logical `{0, 1}` and pairs fine. Good —
  this is the *intended* use case.

**Already committed.** Bridge-test invokes trace-compare with
`--remap-columns --stalls --extended` (commits `6588a62`, `09741a7`).
No further action needed for the Phase B 7-test batch.

**Possible future improvement.** In `remap_tile_columns`
(`src/trace/compare.rs:660`), drop tile keys whose event list is empty
before collecting the column set. That would make the algorithm robust
to the "empty-column pollution" case above without changing behavior on
current tests.

---

## Finding #3 — What the clean report actually tells us about emulator accuracy

This is the important finding. Full report:
`build/bridge-test-results/20260423/vector_scalar_using_dma.chess.cycles.compare.txt`.

### Per-iteration signal

```
Tile (0,2) Core -- slot1 (4 iterations, StepChange at #1 (magnitude 254115))
  Iter   HW_period  EMU_period   dt_period   cum_drift
     1      264423       10308     +254115     +254115  <<< ANOMALY
     2       10310       10308          +2     +254117
     3       10310       10308          +2     +254119
```

**Steady-state drift is 2 cycles out of ~10 310 = 0.019 %.**  That's
basically perfect cycle accuracy for the inner loop.  Whatever the
emulator is modeling, it's modeling it right.

**Iteration 1 has a 254 115-cycle HW anomaly the emulator doesn't
reproduce.**  HW spent roughly 25× the steady-state period doing
something on the first pass.  Plausible candidates, in rough order of
probability:

1. Firmware / driver setup work that runs before HW's trace buffer
   starts sampling but after EMU's trace buffer is already live.  I.e.
   the 254k cycles did happen, but HW attributed them to "iteration 1"
   because the first trace event landed late.
2. First-touch DRAM / cache effects on real silicon that the emulator
   collapses to an idealized latency.
3. Power-state wake-up.  NPU may start in an idle state whose transition
   to active is measured in O(100k) cycles.
4. BD chain prime / DMA warm-up.  The first descriptor may block on
   something subsequent descriptors don't.

Likely (1) or (2); both are outside the cycle-accurate compute model and
probably shouldn't be chased as emulator bugs.  **This is exactly the
kind of signal Phase E was built to surface — "how real is our cycle
accuracy, and where does it diverge from silicon?"**

### Count mismatches

```
[edge] slot0                            0/1 COUNTS DIFFER    OK
[edge] slot1                            4/4                  DIVERGES at #1
[edge] slot2                            4/3 COUNTS DIFFER    DIVERGES at #0
```

- `slot0: 0/1` — EMU captures one `INSTR_EVENT_0` that HW doesn't. Almost
  certainly the very first `event0()` call firing before HW's trace buffer
  starts recording. Trace-buffer coverage artifact, not a compute bug.
- `slot2: 4/3` — EMU captures one extra `INSTR_EVENT_1`. Same kind of
  boundary artifact at the other end of the run.

These are trace-window-coverage differences, not kernel-behavior
differences.  The kernel produces identical output on both sides; the
capture window differs at the edges by a handful of events.

### Absolute cycles

HW: 41181.  EMU: 41176.  Δ = 5 cycles (0.012 %).  This is the number we'd
feed to a coarse ratio classifier.  Fine as-is, but it hides the
structure that the per-iteration breakdown reveals.

---

## Implications for Task 7 (classification)

**Problem with the plan as written.**  The Phase E spec
(`docs/superpowers/plans/2026-04-23-phase-e-trace-diff-cycle-budget.md`)
§Task 7 says MATCH requires

> 0 diverged AND 0 count mismatch on both Edge and Level lines AND
> ratio (EMU/HW) ∈ [lower, upper]

On *this* kernel — where steady-state accuracy is 0.02 % and the only
mismatches are a startup anomaly + boundary capture artifacts — the
plan's classifier will say `DRIFT`.  Technically correct but not
actionable: almost every test we run is going to trip the "count
mismatch" clause because trace-buffer boundaries are asymmetric between
HW and EMU, and almost every test is going to have SOME startup
difference because the emulator doesn't model firmware setup.

**Options for Task 7:**

- **A. Implement verbatim.**  Every real test shows `DRIFT`.  We lean on
  `cycle-drift-overrides.txt` (Task 11) to silence noise per-test. This
  creates busy work: we'd have to hand-enter an override for every test
  we care about.  The override file becomes a giant noise filter rather
  than surfacing drift issues.

- **B. Classify on steady-state per-iteration ratio.**  When the report
  has a per-iteration breakdown, use the *median steady-state period
  ratio* (excluding anomalies) as the primary signal.  Count mismatches
  at start/end become `MATCH-with-boundary-mismatch` (warning) instead of
  `DRIFT` (error).  Step changes become `MATCH-with-startup-anomaly`.
  Overrides only needed for tests where steady-state itself is off.

- **C. Hybrid.**  Implement A literally (simple), but add an extra
  classification "MATCH*" for the "zero-diverged but has count-mismatch
  at boundaries only" case.  Postpones the harder call until we have more
  data from the 7-test batch.

**My lean is B**, but B requires parsing the per-iteration breakdown out
of the report, which adds parsing complexity to the helper.  C is the
pragmatic middle.  The user and I haven't decided yet; worth picking
this up explicitly when Task 7 starts.

---

## Implications for the Phase E plan doc

The plan was written before we had end-to-end data. A few spots to
update when we revisit:

- **Task 7 classification rules** — whichever of A/B/C we pick above,
  the "Rules in bash" table in the plan needs to match.  Currently it
  specifies A (strict).
- **Task 11 overrides file** — if we pick B or C, the override file is
  mostly unused rather than load-bearing.  Scope should shrink.
- **Task 14 validation pass** — expected outcomes should reflect what
  we now know the 7-test batch will look like (DRIFT everywhere under
  rule A; mostly MATCH-with-boundary-mismatch under rule B).
- **Non-obvious fact #11** worth adding to the spec's preamble: HW
  trace buffers don't capture the first ~250k cycles of a run; the
  emulator does. Count-mismatches at boundaries are normal.

---

## Follow-up investigations to file (not blocking Phase E)

1. **What actually consumes the 254k-cycle HW startup window?**
   Hypotheses above; confirm with a VCD capture of the first 300k cycles
   on real silicon.  Out of Phase E's scope.  Goes in the "emulator
   fidelity" backlog.

2. **Trace-compare robustness backlog:**
   - `remap_tile_columns` should drop empty-event tiles before collecting
     columns (see Finding #2).
   - Tile header line "anchor: HW cy 0" is ambiguous: it can mean
     "first event at cy 0" or "no HW events on this tile".  Use a
     sentinel or explicit text.
   - `find_edge_anchor` anchors on the *first shared edge slot*.  If the
     shared slot happens to have a boundary event on one side and a real
     event on the other, the anchor is misaligned.  Prefer the *median*
     shared event or pick by event-count match.
   - "0 diverged" in the summary doesn't include count mismatches.
     Reasonable but surprising; spell it out.

3. **cascade_flows is still producing empty EMU trace** (halt at 2253
   cycles, no packets).  Known, pre-dates this session.  Multi-tile
   cascade trace-unit bug in the emulator.  Separate investigation.

4. **ctrl_packet_reconfig_1x4_cores** fails aiecc routing when
   trace-injected (4 compute tiles + trace routes saturate the router).
   Exposed but not caused by the trace-inject device-targeting fix
   (commit `895a632`).  Routing capacity issue in aiecc; file upstream
   if we care.

---

## How to re-enter this work cold

1. `cat docs/superpowers/plans/2026-04-23-phase-e-trace-diff-cycle-budget.md`
   for the plan.
2. `git log --oneline -20` — includes the decoder-migration commits added
   later on 2026-04-23 (parse-trace.py + trace-compare events-JSON swap).
3. Run the smoke test to see the current state of the pipeline:
   ```bash
   source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
   cd /home/triple/npu-work/xdna-emu
   ./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout --chess-only vector_scalar_using_dma
   LATEST=$(ls -dt build/bridge-test-results/2026*/ | head -1)
   cat $LATEST/vector_scalar_using_dma.chess.cycles.compare.txt
   ```
   Expected post-migration: single `Tile (0,2)` entry, 2 edge event types
   (INSTR_EVENT_0 / INSTR_EVENT_1), 4 iterations each, **Stable** drift
   classification, constant `+2` cycles/iter EMU-fast-of-HW. No count
   mismatch, no divergence. This is the clean baseline for Task 7.
4. Task 7 is up next. See "Post-decoder-fix rethinking Task 7" below.

---

## Post-decoder-fix rethinking Task 7

With the decoder migration, the landscape changed materially:

- The previously-scary "254k anomaly" vanished — it was a decoder artifact.
- `vector_scalar_using_dma` shows **0 diverged, 0 count mismatch, Stable
  iteration drift of +2 cycles/iter**. Cycles-total ratio EMU/HW = 0.9999.
- Per-iteration output is structured (`IterationResult.drift_classification
  ∈ {Stable, StepChange, Accumulating, Irregular}`) and already
  categorizes anomaly shape for us.

This makes the classification problem much simpler than I thought in the
A/B/C discussion above. What we actually need from Task 7:

1. **MATCH** — diverged=0, count mismatch=0, ratio in threshold. The
   common case now.
2. **DRIFT** — anything with diverged>0, or ratio outside threshold, or
   iteration classification ∈ {StepChange, Accumulating, Irregular}.
3. **EMPTY** — no events on one side (trace didn't capture).
4. **COMPILE-FAIL(traced)** — already handled by existing plumbing.

Prefer a thin classifier that reads *both* the scalar cycles.txt files
**and** greps the compare report summary lines ("Edge event types: N
clean, M diverged, K count mismatch"). We don't need per-iteration
parsing; the Rust side already summarized it.

Overrides (Task 11) stay cheap: per-test "accept up to this drift in
cycles per iter", matching the thin classifier. Override file is only
needed for tests where legit emulator simplifications produce known
larger drift.

Summary for whoever picks up Task 7: the plan as written largely works
now. The "strict classifier gives DRIFT everywhere" concern that
motivated options A/B/C was false; it was built on bad decoder output.
Implement the strict classifier (option A in the original framing); it
will produce useful signal on real data.
