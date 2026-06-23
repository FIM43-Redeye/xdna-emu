# Trace-Experimenter Loop: Active HW Convergence — Design

**Date:** 2026-06-22
**Issue:** #140 (byte-identical emulator/HW trace reports)
**Status:** Design approved (interactive brainstorm). Ready for implementation plan.
**Predecessor:** program_path through-core (merged, `b3a62ea7`) — static `CoreLockRelay` edge,
three-witness validated *offline*. Plan 1 trace-inference-engine (merged) — chainer, verifier,
degeneracy reasoner, planner, and closed loop, all running against a `MockInstrument`.
**Successor (separate, immediately slated):** the actuator folds — trace-GROUPS + Z3 groups phase,
then per-tile mode threading. Driven in part by the honest-block findings this loop surfaces.

---

## 1. Goal, ceiling, non-goals

**Goal.** Close the inference loop against **real Phoenix (NPU1/AIE2) hardware** and demonstrate it
**converges on a suite of kernels**. The planner emits proven-gain `MEASURE-NEXT` batches, the
actuator runs them on the NPU, the verifier admits/rejects empirical rules, and the engine re-chains
to a fixpoint — terminating per kernel in one of two honest states: **full placement** (every
configured trace event placed on the derivation backbone) or **falsifiable halt** (a precise
degeneracy / blocked report, every verdict carrying measured provenance).

**Why now.** Plan 1 built the loop's control flow, planner, and termination guarantee, but drove
them against a `MockInstrument`. The actuator (`trace_capture.capture()` over a long-lived
`RunnerSession`) and the verifier (`correlates`/`deterministic`/`coincident`) are real and
HW-validated. The two genuinely-missing pieces are (a) an **HW adapter** that lets the existing loop
drive real captures, and (b) a **populated reachability self-model** — today an empty frame. Phoenix
is a one-way door (it is replaced, not augmented, at the Strix swap), so the HW-ground-truth half of
the engine must be built while the silicon is here.

**Ceiling.** This loop validates the **placement self-model against hardware** — that the engine's
derivation backbone (which event derives from which, what is a stochastic root, what is degenerate)
is a sound, falsifiable account of what Phoenix actually traces. It is **not** an emulator-vs-HW
byte-identical comparison; that is a downstream consumer of this validated model, not this plan.

**Non-goals (deferred or out of scope by construction):**
- **Actuator folds** — trace-GROUPS (with the Z3 groups phase) and per-tile EVENT_TIME/EVENT_PC mode
  threading. Out of *this* plan, but explicitly the **immediately-following plans** (the broad goal
  is 100% characterization; deferral here means "one reasonable thing per plan," not abandonment). If
  a suite kernel's only separating batch needs >8 events/tile or PC mode, the planner emits an honest
  **blocked-needs-fold** finding (the humility property working as designed); that finding becomes a
  driver for the very next plan.
- **Emu-vs-HW byte-identical trace comparison.** Downstream consumer of the validated placement
  model. Out of scope here.
- **General kernel coverage.** The starter suite is three topologically-distinct kernels (§5); the
  loop is built to be kernel-agnostic, but proving convergence on the full corpus is later work.

---

## 2. Current state (what already exists — do not rebuild)

From the stack inventory (2026-06-22). All paths under `xdna-emu/`.

| Layer | Component | State |
|-------|-----------|-------|
| Instrument (read) | `tools/parse-trace.py` decoder (per-tile mode auto-detect) | EXISTS |
| Actuator (write+run) | `tools/trace_capture.py` `capture()` / `configure_batch()`; `tools/trace_runner.py` `RunnerSession` / `ParseSession` (extracted) | EXISTS, HW-validated 2026-06-17 |
| Verifier (empirical) | `tools/inference/verifier.py` `correlates` / `deterministic` / `coincident` over `trace_join.pair_derivability` + `trace_variance.aggregate` | EXISTS |
| Engine | `tools/inference/{chainer,degeneracy,classify,engine}.py` | EXISTS (union-find v1; Z3 deferred to the groups fold) |
| Planner | `tools/inference/planner.py` `propose_next` / `plan_cotrace` / `seed_plan` / `Batch` | EXISTS |
| Loop | `tools/inference/loop.py` chain→fixpoint→act-on-stall→converge/halt, `MockInstrument`, termination ranking | EXISTS |
| Self-model | `tools/inference/reachability.py` `ReachabilityModel` (constraints w/ provenance) | EXISTS but **instantiated empty** (`loop.py:97`) |
| Static reachability (gain source) | `tools/config_extract/reachability.py` `Reachability` BFS (mirrors Rust `route_graph`) | EXISTS |
| Generator | `tools/config_extract/generator.py` `generate_ledger` (config_path + program_path) | EXISTS |

**The two gaps this plan fills:** the **HW adapter** (loop currently drives `MockInstrument`) and the
**self-model population** (the `ReachabilityModel` frame is empty; nothing fills legality / gain /
empirical limits).

---

## 3. Architecture

**Approach: HW-adapter (chosen over a new orchestrator or a two-phase batch-dump).** Preserve the
proven loop, planner, verifier, and termination guarantee untouched; swap the mock for hardware and
add the self-model. A new orchestrator would duplicate termination-guaranteed code; a two-phase
"emit the whole plan upfront, run all, chain once" would destroy the adaptive act-on-the-stall
property that is the entire point of the loop.

**Data flow (per kernel):**

```
config dump (Rust extractor)  ─┐
xclbin configured-event set   ─┼─▶  static self-model (legality + gain)
route graph / Reachability    ─┘            │
                                            ▼
                            loop.py seed ──▶ planner.propose_next (proven gain)
                                            │
                                            ▼  Batch (≤8 EVENT_TIME slots/tile)
                              HwInstrument.capture() ── runs on Phoenix ──▶ batch_*/hw/trace.events.json
                                            │
                                            ▼
                       loader re-emits fired atoms ──▶ verifier (correlates/deterministic/coincident)
                                            │
                                            ▼
                              engine chains to fixpoint
                                            │
                                  ┌─────────┴──────────┐
                          full placement?        stall? ──▶ planner inspects
                                  │                         │       │
                                 done              proven-gain   no gain / blocked
                                                  (next batch)   ──▶ empirical constraint
                                                                     into self-model;
                                                                     falsifiable halt
```

**`HwInstrument` (new, the adapter).** Conforms to the same interface `loop.py` already calls on
`MockInstrument` (`capture(batch) -> run_dirs`). Internally it owns a persistent `RunnerSession`,
translates the planner's `Batch` into `configure_batch()`'s patch spec, invokes the real
`trace_capture.capture()` against Phoenix, and returns loader-readable run dirs
(`batch_*/hw/trace.events.json`). Anchoring, RESET handshake, and trace-size negotiation are already
owned by `RunnerSession`/`HwRunner` — the adapter is thin.

**Entry point (new).** A small CLI (e.g. `tools/inference/run_experiment.py`) that, given a kernel
build dir, loads the config dump + configured-event set, builds the static self-model, constructs the
`HwInstrument`, runs `loop.py`'s loop to termination, and writes a per-kernel **convergence report**
(placement backbone + admitted/rejected rules + degeneracy/blocked findings, each with provenance).

---

## 4. The reachability self-model (the crux — hybrid)

Three parts, matching the existing `ReachabilityModel` frame and the engine's humility principle.

**4a. Static legality (offline, from the config).** For each candidate batch the planner must prove,
*before running*, that the batch is physically traceable:
- the `(col, row, pkt_type)` tiles it names are trace-configurable for this xclbin,
- the named events are valid for each tile-type,
- the per-tile slot budget is ≤8 (anchor in slot 0 + ≤7 events).
The xclbin's **configured-event set is the termination domain** (the fixed, finite top component of
the lexicographic ranking) — already the `seed_plan(configured_events)` input.

**4b. Static gain (offline, reusing the existing `Reachability`).** A batch has gain iff it
co-traces a pair the KB currently lists as **correlate-unknown** (never co-traced) **and** an
**orientable** relationship exists for that pair — i.e. the generator ledger holds a derives-edge
(`config_path` or `program_path`) linking them, per the same `config_extract/reachability.py` BFS
that mirrors the Rust route graph. This is what lets the planner satisfy **"never emit-then-discover"**:
gain is *proven* from config before a batch is spent. No new reachability engine — the generator's
`Reachability` is the gain oracle.

**4c. Empirical limits (online, with measured provenance).** When a batch runs and reality contradicts
the static optimism, the result is folded back into the self-model as a `ReachabilityModel` constraint
**citing the batch that demonstrated it**:
- a configured event that **never fires** on HW → a constraint that removes it from achievable
  placement (and decrements the ranking's top component honestly),
- a pair whose cross-run `std > eps` → **not** `correlates`; the rule is *rejected* (a finding), not
  silently dropped,
- a co-trace that cannot be realized because of a **confound** (e.g. memmod row-2 vs core mode-0
  forcing) → a `cannot_cotrace` constraint that blocks re-proposal and downgrades any dependent
  degeneracy verdict to `unconfirmable-structural`.

This is exactly the property that keeps the loop honest: the static side proposes, the hardware
disposes, and every limit the hardware imposes is recorded as a falsifiable, provenance-bearing fact.

---

## 5. Kernel suite (starter, extensible)

Three topologically-distinct kernels from `mlir-aie/test/npu-xrt/`. The loop is kernel-agnostic; this
is the first cut, not a coverage claim.

1. **`add_one_using_dma`** — anchor. Compute tile + memtile; exercises `config_path` memtile relays
   **and** the `program_path` through-core edge (the full edge-kind stack). The kernel all three prior
   plans validated against.
2. **`memtile_dmas`** — memtile-centric DMA topology; `config_path` only, no through-core. Confirms
   the loop converges where program_path is correctly absent.
3. **`two_col`** — multi-column; exercises cross-column routing reachability (the BFS over inter-tile
   edges spanning columns).

Per-kernel terminal state, both counting as convergence:
- **Full placement** — every configured trace event placed on the backbone, or
- **Falsifiable halt** — degeneracy (`structural-candidate` / `irreducible-by-instrument`) or
  **blocked-needs-fold**, each verdict carrying the measured provenance (the batch) that justifies it.

---

## 6. Testing & falsifiability

The discipline carried from Plans 1–3: **a loop that runs ≠ a loop that genuinely converged via
hardware.** Tests must prove the loop *used* the silicon and that withholding evidence changes the
outcome.

- **Offline (unchanged):** the existing `MockInstrument` loop tests stay green — the control flow and
  termination ranking are not modified.
- **Self-model units (new, offline):** an over-budget batch (>8 slots/tile) is rejected by legality;
  an already-co-traced pair returns `NO_GAIN`; an unorientable pair returns `NO_GAIN`; an empirical
  `cannot_cotrace` constraint downgrades a dependent degeneracy verdict.
- **HW integration (new, Phoenix-gated):** run the loop on `add_one` end-to-end and assert
  1. it **converges** (terminates in a defined state),
  2. it places **at least one event that is not config-derivable** — its placement required a measured
     HW co-trace, proving the loop actually consumed hardware evidence (the "loads ≠ derives" gate),
  3. a **withheld or forced-wrong batch changes the outcome** (genuine oracle, not vacuous): e.g. a
     forced co-trace `std` above eps must *reject* the corresponding `correlates` rule.
- **Suite (new, Phoenix-gated):** each of the three kernels reaches a defined terminal state and the
  per-kernel convergence report is non-empty and provenance-complete.

HW batches use the long-lived `RunnerSession` batch-stdin path (cheap — many small batches are
expected and fine).

---

## 7. Risks

- **R1 — self-model legality drift.** The static legality predicate must match what the patcher/HW
  actually accept (tile-type event validity, slot budget). Mitigation: legality unit tests against the
  real configured-event set extracted from each suite xclbin; an illegal batch must be rejected
  *before* a capture is spent.
- **R2 — gain proof too weak or too strong.** If "gain" admits pairs the ledger can't orient, the loop
  spends batches for nothing (emit-then-discover); if too strict, it halts prematurely. Mitigation:
  gain is defined strictly as correlate-unknown ∧ ledger-orientable; covered by NO_GAIN units and the
  HW convergence assertion (b).
- **R3 — empirical constraints vs termination.** Folding a "never-fired" event back must *decrease*
  the ranking's top component (event leaves the achievable set), not loop. Mitigation: constraint
  application is monotone on the configured-event domain; assert the ranking strictly decreases.
- **R4 — a suite kernel needs a fold.** `two_col`/`memtile_dmas` may have a tile whose only separating
  batch exceeds 8 events. Expected and handled: the loop emits **blocked-needs-fold** (a clean
  terminal state), which seeds the next plan. Not a failure of this plan.
- **R5 — adapter interface mismatch.** `MockInstrument` and the real `capture()` must present an
  identical `capture(batch) -> run_dirs` contract. Mitigation: define the interface explicitly; the
  HW integration test exercises the real path; the offline tests exercise the mock against the same
  interface.

---

## 8. Provisional task shape (to be expanded by writing-plans)

- **P0 — HwInstrument adapter + interface.** Define the `capture(batch) -> run_dirs` contract; build
  `HwInstrument` over `RunnerSession`/`trace_capture.capture()`; smoke it on `add_one` (one
  hand-built batch) to confirm the loop can drive real HW and the loader reads the result.
- **P1 — Static self-model (legality + gain).** Populate legality from the config dump + configured-
  event set; wire gain to the existing `config_extract/reachability.py`. Unit tests (over-budget
  reject, NO_GAIN cases).
- **P2 — Empirical limits.** Fold never-fired / std>eps / confound results back into
  `ReachabilityModel` with measured provenance; assert ranking monotonicity.
- **P3 — Entry point + convergence report.** `run_experiment.py`: kernel dir → loop → per-kernel
  report (backbone + admitted/rejected rules + findings, provenance-complete).
- **P4 — HW convergence on `add_one`.** The three-part falsifiability integration test.
- **P5 — Suite convergence.** `memtile_dmas` + `two_col` reach defined terminal states; reports
  non-empty and provenance-complete; full-suite regression (offline engine tests unchanged).

Actuator folds (trace-GROUPS + Z3, per-tile mode threading): **the immediately-following plans.**
