# Trace Inference Engine — Design

**Date:** 2026-06-21
**Issue:** #140 (byte-identical emulator/hardware trace reports)
**Status:** Design approved; implementation plan to follow.
**Precursor:** the trace-capture engine (`tools/trace_capture.py`, HW-validated
2026-06-17) is this engine's measurement organ. See
`docs/superpowers/findings/2026-06-17-capture-engine-validation.md`.

## Purpose

Reconstruct the **full per-kernel trace** — every configured event placed in a
proven relationship to the others — by treating nondeterminism as a
**logical-inference puzzle**: given facts we have measured to be true, follow a
chain of verified rules to the truth, and where the chain stalls, drive the
*next measurement* until it converges or proves it cannot.

The capture engine answers "trace exactly these events next, on demand, with
honest coverage." This engine is the reasoner that *decides what to measure
next and why*, turning a pile of trace scripts into a single modeling
instrument with a read path, an actuator, and a reasoner.

## Keystone principle: nothing is an axiom until verified

Every rule is a **hypothesis**. The engine empirically verifies each rule holds
across measured data *before* it is allowed to derive truth. A rule that fails
verification is never used — and its failure is itself recorded as a finding.

This is not a coding convention we promise to remember; the data model makes it
**unviolable by construction** (see "no `given` support type" below). The
existing `trace_variance.py` `check_span_law` / `law_violations` pair is the
prototype of this verify-before-trust pattern; the engine generalizes it.

## Build, not adopt (decided)

A small custom forward-chainer, built on the numpy verification we already own,
**plus Z3 as a reserved tenant of one predicate** (degeneracy entailment, below).

The engine has two layers with different needs, and every off-the-shelf option
is wrong for at least one:

- **Logic layer (categorical):** anchor identity, group membership, derivation.
  Forward-chain to fixpoint over a few hundred facts per kernel — ~150 lines,
  *obviously correct by reading it*. That readability matters more here than
  anywhere: a sound engine must be able to trust its own chainer.
- **Numeric layer (temporal/arithmetic):** `std(X−S) < eps ⇒ derives`, span
  laws, coincidence. **Verification here is empirical, not symbolic** — we check
  whether a hypothesis holds across measured runs (numpy), we do not solve for a
  model. We already wrote the prototype (`trace_variance.py`).

Why each adopt option loses: pyDatalog is dead (~2015); Soufflé is pure-logic,
batch, heavy, no native numeric/temporal constraints; Z3/SMT is a *solver* not a
forward-chainer and its symbolic power is idle for the empirical 95%; RETE
engines (experta, durable_rules) are unmaintained or business-rules-shaped and
mismatch verify-then-admit.

**Z3 is reserved for exactly one corner** where reasoning is genuinely symbolic
rather than empirical: proving degeneracy (below). It sits behind a clean
predicate interface — the landlord is our chainer; Z3 is one tenant.

## Section 1 — The knowledge base: facts and rules

**A fact** is an immutable record: `predicate`, `args`, and a **support** field
that is *either* `measured` *or* `derived(rule, [premise_fact_ids])`. **There is
no `given` support type.** A fact with no measured ancestry and no
verified-rule derivation cannot exist — this is the formal teeth behind
no-axioms. Even anchor identity is a measured-and-verified fact.

**Core predicates** (the whole vocabulary the chainer reasons over):

| Predicate | Support | Meaning |
|-----------|---------|---------|
| `fired(event_key, run, anchored_ts)` | measured | straight from capture data; `event_key` = 4-part `col\|row\|pkt\|name` |
| `deterministic(event_key)` | derived | `std(ts)` across runs < eps |
| `derives(child, parent, offset)` | derived | `std(child.ts − parent.ts) < eps` across *all* runs — the backbone relationship |
| `stochastic_root(event_key)` | derived | varies across runs **and** no parent derives it |
| `group_fired(group_key, run, ts)` | measured | a group slot fired |
| `group_member(group_key, member)` | derived | membership implication, verified on data |
| `coincident(A, B)` | derived | A and B fire at identical anchored ts in *every* run — the degeneracy candidate |

**A rule** is a hypothesis paired with a verifier. For the **numeric rules the
rule body and its verifier are the same computation** — applying `derives` *is*
checking the offset holds across runs; self-verification falls out for free.
Only the **implication rules** (groups: "group fired ⇒ some member fired") need a
*separate* verifier that confirms the implication holds on measured data before
the chainer may use it.

The KB is three sets: **facts**, **admitted rules** (verified), **rejected
rules** (each a finding).

## Section 2 — The loop: chain to fixpoint, then act on the stall

1. **Chain.** Apply every admitted rule until a full pass adds nothing. Because
   every numeric rule self-verifies as it fires, the entire fixpoint is *proven*
   — the closure is sound by construction. Each event ends in one of three
   states: `derived` (on the backbone via a proven offset), `stochastic_root`
   (proven independent), or **unresolved**.

2. **Inspect the stall.** Each unresolved event is unresolved for a *diagnosable
   reason*, and the reason picks the next action:
   - **Not-yet-measured** — never co-traced with a candidate parent in one batch,
     so `derives` *couldn't* fire → emit **MEASURE-NEXT**: a concrete batch
     (anchor + event + candidate parents). Re-infer.
   - **Measured-but-ambiguous** — e.g. `group_fired` with multiple active
     members, or two candidate parents both within eps → MEASURE-NEXT a
     *disambiguating* batch (rivals alone, or group split to singletons).
   - **Provably degenerate** — resolved by Section 2b → **stop and report**.

3. **Converge or halt.** Ends when the worklist is empty (full trace
   reconstructed) *or* every remaining unresolved event is provably degenerate
   (halt with a precise irreducibility report).

**Termination guarantee — the formal heart of "strictly sound + closed loop":**
the engine never guesses to make progress. It either *measures* its way out of
an ambiguity or *proves* the ambiguity cannot be measured away and reports it.
There is no third branch that picks a plausible answer.

## Section 2b — Degeneracy as entailment over a verified causal structure

Empirical-exhaustion ("we ran out of batches") gives **one** verdict and
silently fuses two distinct realities. Designing degeneracy in from the start
forces an explicit **causal structure** into the engine — a representation both
layers share — and buys a **trichotomy**.

Every event reduces, through proven `derives` edges, to `(stochastic root) +
(accumulated constant)`. So the only irreducible question is ever: **are two
stochastic roots the same source, or two sources that co-vary?** That collapse
is what keeps the hard problem small.

For a coincident root pair `(r1, r2)`, Z3 reasons over the **causal structure** —
a graph whose edges are *verified hypotheses* (this shim-DMA drives that
memtile-port, established empirically, never assumed) — and checks entailment:

- **Structurally degenerate** — `causal_structure ∧ (r1 ≠ r2)` is **UNSAT**.
  They cannot differ under any valuation: one physical source seen twice (e.g. a
  DMA event observed at both shim and memtile). Report **irreducibly-one-DOF,
  with a proof**.
- **Observationally degenerate** — **SAT** (they *can* differ), but the
  measurement-reachability model proves **no batch we can run** adds an edge that
  separates them. Genuinely independent in silicon, but the trace instrument
  cannot demonstrate it. Report **irreducible-by-instrument** — a finding about
  the *limits of the trace hardware itself*.
- **Separable** — SAT, and a batch *does* reach the distinguishing edge →
  **MEASURE-NEXT**.

This trichotomy is the answer to the overlapping-DMA-gaps soundness worry:
"same DOF" and "two DOFs we can't separate" stay distinct, *with proofs*.

**The subtlety resolved:** coincidence is a *cross-run statistical* property,
but Z3 works over a *single symbolic model*. The bridge is the division of
labor — **numpy establishes and verifies the causal edges empirically** (across
runs); **Z3 reasons over the resulting structure symbolically** (forced vs.
contingent). The causal graph is the shared artifact, and **every edge is itself
a verified fact, never a given** — aie-rt knowledge tells us only *where to
look*, never *what is true*.

## Section 3 — The instrument stack, and where the folds land

The trace ecosystem reorganizes into five layers, each a unit with one job:

1. **Instrument (read)** — `decoder` / `parse-trace`. Raw bytes → labeled
   events. Unchanged; the bottom.
2. **Actuator (write + run)** — `trace_capture` (over `RunnerSession`). Already
   built and HW-validated. The engine's primitive for *execute one MEASURE-NEXT*.
3. **Verifier (empirical)** — generalized from `trace_variance`'s
   `check_span_law` / `law_violations`. Establishes and checks facts and causal
   edges across runs.
4. **Inference engine** — NEW. Logic chainer + Z3 degeneracy reasoner. Consumes
   `trace_join`'s primitives; `build_derivability_graph` becomes the *seed* of
   the fact base, not a standalone endpoint.
5. **Planner** — NEW. Holds the measurement-reachability self-model; turns
   ambiguities into concrete MEASURE-NEXT batches.

**The four folds, each landing in exactly one layer:**

- **Trace GROUPS** → a *measurement capability* in the Planner's self-model. A
  group-batch is a cheaper, coarser measurement (more events per 8-slot budget,
  lower specificity). The planner *chooses* group-batches for broad coverage and
  singleton-batches for disambiguation — groups become a tactical trade-off, not
  a special feature.
- **Memmod row-2 co-trace** → a *verified constraint* in the self-model. "Can
  core + memmod co-trace on row 2 through one egress?" is a reachability fact the
  engine **verifies rather than assumes**; the validation run's "memmod produced
  no packets" is the first hypothesis the self-model tests and records. The
  instrument's own limits are measured.
- **Seed / discovery** → **Phase 0** of the loop. The over-counting seed was
  never wrong, just premature — it's the *initial coverage sweep* (ideal use for
  group-batches) that discovers which events fire before the inference loop works
  the real active set.
- **Mode-threading refactor** → falls out for free. The MEASURE-NEXT directive
  is the *single source of truth* for a batch — mode included — threaded once
  from planner → actuator → decoder. The current mode=0-duplicated-at-two-sites
  problem disappears.

**Consolidation by proven parity, not deletion-on-faith.** The old `trace-sweep`
sweep/matrix/regression orchestration is our regression gate for the emulator
refactor. The engine subsumes it as the new top — but the old sweep stays alive
as the gate **until the engine demonstrably reproduces its verdicts** (see the
parity gate in Testing).

## Section 4 — Testing: who verifies the verifier?

A self-verifying engine must answer this or the keystone is hollow. The answer
is **two axes for two failure modes** — machinery bugs vs. reality-mismatch —
needing different oracles.

**Axis 1 — prove the machinery correct (synthetic ground-truth, no HW), TDD:**

1. **Chainer soundness as a property test.** On *every* fixpoint, assert every
   derived fact's provenance DAG bottoms out *only* in `measured` leaves. One
   property test mechanically pins the entire no-axiom keystone.
2. **Verifiers against known answers.** Synthetic run-sets where we *know* the
   offset holds or not. Adversarial cases are the point: an offset that holds for
   N−1 runs and breaks on the Nth (must reject — eps boundary); two events
   coincident by *sample luck* but structurally independent (must not over-claim
   — the DMA-gaps worry in miniature).
3. **Degeneracy trichotomy as Z3 oracle tests.** Hand-built causal structures
   with known class: same-source → assert `UNSAT(r1≠r2)` (structural);
   independent + blocking reachability → assert SAT-but-no-batch (observational);
   independent + reachable → assert SAT-plus-emitted-batch (separable). Pure
   symbolic, fast, no hardware.
4. **Closed-loop convergence on a mock instrument.** Run the full loop against a
   *synthetic instrument* emitting events from a known ground-truth causal model.
   Assert convergence to that ground truth, and that **each MEASURE-NEXT strictly
   reduces ambiguity** (monotone progress — the infinite-loop guard).

**Axis 2 — prove the results match silicon (HW ground-truth).** On
`add_one_using_dma`: the engine must *independently re-derive* what the
capture-engine validation established — the 5 stochastic roots, the deterministic
backbone — then correctly classify any structural degeneracy (shim-DMA-seen-at-
memtile is the natural first confirmation).

**The bridge:** synthetic tests prove the machinery cannot lie; HW proves its
output matches reality. Neither alone suffices — a sound engine on a wrong causal
model passes Axis 1 and fails Axis 2; a lucky-on-this-kernel engine passes Axis 2
and fails Axis 1.

**Parity gate (retiring the old sweep).** Differential test: the engine's
verdicts must reproduce `trace-sweep`'s regression findings and
`trace_variance`'s `law_violations` on the same data before the old sweep is
demoted. Consolidation earns its way in.

## Open questions deferred to implementation planning

- Concrete fact/rule representation (dataclasses vs. tuples; how provenance DAGs
  are stored and walked for the soundness property test).
- The measurement-reachability self-model's schema — how slot budget, co-trace
  egress constraints, and group capabilities are encoded so the planner can
  enumerate reachable batches.
- Z3 encoding details for the causal structure (QF_LRA vs. QF_LIA; how verified
  `derives` edges and group implications translate to constraints).
- Whether group membership is "exactly one member" or true disjunction (multiple
  members firing the same cycle) — itself a hypothesis to verify.

## Glossary

- **anchored ts** — an event's timestamp relative to the anchor
  (`1|2|0|PERF_CNT_2`), making times comparable across runs.
- **eps** — the tolerance under which a cross-run std counts as "stable."
- **backbone** — the deterministic skeleton of `derives` edges, identical across
  runs within eps.
- **stochastic root** — an event that varies across runs and is not derived from
  any parent; the genuine degrees of freedom (all DMA-delivery-driven on
  `add_one_using_dma`).
