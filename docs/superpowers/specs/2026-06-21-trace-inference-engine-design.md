# Trace Inference Engine — Design

**Date:** 2026-06-21
**Issue:** #140 (byte-identical emulator/hardware trace reports)
**Status:** Design approved (revised after adversarial review, 2026-06-21);
implementation plan to follow.
**Precursor:** the trace-capture engine (`tools/trace_capture.py`, HW-validated
2026-06-17) is this engine's measurement organ. See
`docs/superpowers/findings/2026-06-17-capture-engine-validation.md`.

> **Revision note (2026-06-21).** Three independent adversarial reviews found
> that the first draft equated a verified *arithmetic* property (stable cross-run
> offset) with a verified *causal* one, overstated several "by construction"
> claims, and specified a degeneracy branch that was vacuous as written. This
> revision: (1) makes `derives` causal-by-intervention, not correlational;
> (2) adds the `same_source` identity predicate and a third, audited `structural`
> support type so the structural-degeneracy branch is non-vacuous and Z3 stays
> justified; (3) splits the soundness claim into three theorems and gives the
> reachability self-model the same no-axiom teeth as facts; (4) reclassifies
> three "reuse" claims (groups, the derivability graph, the verifier) as the
> net-new work they actually are. The review findings are archived alongside
> this spec.

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

## Keystone principle: no *unaudited* axioms

Every rule is a **hypothesis**, empirically verified against measured data
*before* it may derive truth; a rule that fails verification is never used, and
its failure is itself a finding. The first draft stated this as "nothing is an
axiom until verified." Review surfaced one unavoidable exception — *physical
identity* ("these two trace observations are the same hardware event") is not
measurable at the trace-event level; it is structural knowledge from aie-rt.
Rather than smuggle it in disguised as a measurement, we make it **visible**.

There are exactly **three support types**, and no fact may exist without one:

- **`measured`** — straight from capture data (a decoded, labeled event).
- **`derived(rule, premises)`** — produced by an admitted (verified) rule from
  existing facts.
- **`structural(citation)`** — an identity assertion sourced from aie-rt, **and
  recorded in an auditable ledger** with its citation. This is the *only* support
  type that is not earned from measurement, it applies to *exactly one predicate*
  (`same_source`, below), and every instance is enumerable for review.

So the keystone has teeth *and* honesty: a fact is `measured`, `derived` from
verified rules, or a **listed, citable** `structural` exception. Nothing hides.
The existing `trace_variance.py` aggregate/classify machinery is the seed of the
empirical verifier (see Section 3, layer 3 — it is a seed, not a finished
prototype).

## Build, not adopt (decided), with Z3 retained

A small custom forward-chainer over the numpy verification we own, **plus Z3 for
the degeneracy entailment** (Section 2b). Review confirmed Z3 earns its place for
two reasons union-find cannot cover: the causal structure mixes **equalities**
(`same_source`, `derives` offsets) with **disjunctions** (group implications,
"group fired ⇒ some member fired"), and degeneracy entailment over equalities +
disjunctions is genuine SMT, not pure congruence closure.

Two layers, different needs:

- **Logic layer (categorical):** anchor identity, group membership, derivation,
  identity. Forward-chain to fixpoint over a few hundred facts per kernel —
  ~150 lines, *obviously correct by reading it*. A sound engine must trust its
  own chainer.
- **Numeric/empirical layer:** `std(X−S) < eps`, intervention tracking, span
  laws, coincidence — all checked across measured runs with numpy. Verification
  here is **empirical, not symbolic**: we check whether a hypothesis holds across
  runs, we do not solve for a model.

Adopt options lose for the bulk of the work (pyDatalog dead; Soufflé pure-logic,
batch, heavy, no numeric/temporal constraints; RETE engines unmaintained or
business-rules-shaped). Z3 is reserved for the one corner that is genuinely
symbolic — degeneracy entailment — behind a clean predicate interface. The
chainer is the landlord; Z3 is one tenant.

## Section 1 — The knowledge base: facts and rules

**A fact** is an immutable record: `predicate`, `args`, and a **support** field
(`measured` | `derived(rule, premises)` | `structural(citation)`). A fact with no
valid support cannot exist — the formal teeth behind the keystone.

**Core predicates:**

| Predicate | Support | Meaning |
|-----------|---------|---------|
| `fired(event_key, run, anchored_ts)` | measured | a decoded event; `event_key` = 4-part `col\|row\|pkt\|name` |
| `deterministic(event_key)` | derived | `std(ts)` across runs < eps |
| `correlates(a, b, offset)` | derived | `std(a.ts − b.ts) < eps` across runs — a *symmetric, non-causal* co-variation. Not a backbone edge. |
| `derives(child, parent, offset)` | derived | `correlates` **plus** a causal discriminator (below). Oriented child←parent. The backbone. |
| `stochastic_root(event_key)` | derived | varies across runs **and** no parent derives it |
| `same_source(a, b)` | structural | a and b are the *same physical hardware event* observed at two trace units (e.g. one DMA transfer seen at shim and at memtile). aie-rt-cited, ledgered. |
| `group_fired(group_key, run, ts)` | measured | a group slot fired |
| `group_member(group_key, member)` | derived | membership, verified on data |
| `coincident(a, b)` | derived | a and b fire at identical anchored ts (within eps) in *every* measured run — a degeneracy *candidate*, revisable by future runs |

**`derives` is causal, not correlational (Theme-1 fix).** Stable offset alone is
symmetric (cannot orient) and non-causal (two deterministic events share a
constant offset trivially; two co-varying independent stochastic events do too).
`correlates` captures that raw co-variation and is explicitly *not* a backbone
edge. `derives` is admitted only with all three of:

1. **Parent is stochastic.** A deterministic "parent" carries no jitter to
   transmit, so a constant offset to it is not derivation — it eliminates the
   spurious deterministic-clique complete graph.
2. **Variance transfer (intervention where reachable).** The child must *track*
   the parent under perturbation: across batches whose configuration shifts the
   parent's timing, the child–parent offset stays stable while both move in
   absolute terms. Emitting these perturbing batches is the MEASURE-NEXT
   machinery we already have — `derives` becomes an experiment, not a passive
   correlation. This also **orients** the edge (the cause is the thing we
   perturbed).
3. **Honest fallback.** Where no reachable perturbation isolates the parent, the
   edge is admitted only as `correlates` (never `derives`), and *that* is a
   finding: "co-varies but causal direction unestablished." Such pairs route to
   the degeneracy analysis (Section 2b) as candidates — we never upgrade
   correlation to causation we could not show.

This keeps the keystone honest exactly where the first draft broke it: we verify
the proposition we actually claim.

**A rule** is a hypothesis paired with a verifier. For the **numeric rules** the
rule body and its verifier coincide (applying the rule *is* checking it across
runs). The **implication rules** (group membership) carry a *separate* verifier
that confirms the implication on measured data before the chainer may use it.

The KB is four sets: **facts**, **admitted rules**, **rejected rules** (each a
finding), and the **structural ledger** (every `same_source` citation, for audit).

## Section 2 — The loop: chain to fixpoint, then act on the stall

1. **Chain** every admitted rule to fixpoint. Each numeric rule self-verifies as
   it fires; the closure over `measured`/`structural` leaves is sound *given*
   those leaves and rules (see the three-part soundness claim in Section 4 — this
   is chaining-soundness, one of three obligations, not the whole story). Each
   event ends `derived` (on the backbone), `stochastic_root`, or **unresolved**.

2. **Inspect the stall.** Every unresolved event has a diagnosable reason that
   picks the next action. The reasons are exhaustive by the **ranking function**
   below, which guarantees the loop is not just sound but *terminating*:
   - **Not-yet-measured** — never co-traced with a candidate parent in one batch,
     so the causal test could not run → emit **MEASURE-NEXT** (anchor + event +
     candidate parents, with the perturbation needed for variance-transfer).
   - **Ambiguous-but-separable** — multiple candidate parents, or a multi-member
     group, **and the planner can prove a reachable batch distinguishes them**
     (Section 2b "separable") → emit that proven-separating batch.
   - **Provably degenerate** — Section 2b returns structural or observational
     degeneracy → **stop and report** this pair/cluster as irreducible.

   **Progress is guaranteed, not hoped-for.** A MEASURE-NEXT is emitted *only*
   when the planner has first proven (Section 2b machinery) that the batch will
   add a separating edge — never emit-then-discover. Termination rests on a
   well-founded ranking function: lexicographic `(# unresolved events,
   # untested candidate edges)`, both bounded below by the finite configured-event
   set, both strictly decreasing per iteration (a separating batch resolves an
   event or retires a candidate edge; a non-separating batch is *never emitted*
   because it could not be proven to separate). No livelock branch exists because
   "ambiguous and no batch separates" is **not** routed back to MEASURE-NEXT — it
   is degeneracy, and it halts.

3. **Converge or halt.** Worklist empty (full trace reconstructed) or every
   remainder provably degenerate (halt with a precise, *falsifiable*
   irreducibility report — see Section 2b on revisability).

**Termination guarantee — the formal heart of "strictly sound + closed loop":**
the engine never guesses. It measures its way out, or proves it cannot and
reports, and a ranking function proves it always reaches one of those.

## Section 2b — Degeneracy as entailment over a verified causal structure

Every event reduces, through proven `derives` edges and `same_source` identities,
toward stochastic roots. Because `derives` is now oriented and causal (parent
stochastic + variance-transfer), the reduced structure is a **DAG**, not the
symmetric bidirectional graph the first draft's predicate produced. The only
irreducible question is then: **are two stochastic roots the same source, or two
sources that co-vary?**

For a coincident root pair `(r1, r2)`, Z3 checks entailment over the causal
structure — `same_source` identity edges (structural, ledgered), `derives` offset
equalities, and group-implication disjunctions:

- **Structurally degenerate** — `structure ∧ (r1 ≠ r2)` is **UNSAT**. A chain of
  `same_source`/`derives` edges (or a forcing group disjunction) makes them
  provably one source. Report **irreducibly-one-DOF, with a proof relative to the
  ledgered structure** (the qualifier matters — see the audit obligation below).
  This branch is now *reachable*: without `same_source`, two roots are
  disconnected by construction and UNSAT was vacuous; the identity edges are
  exactly what can connect them.
- **Observationally degenerate** — SAT, but **no reachable batch** adds a
  separating edge. This is finite enumeration over the (verified) reachability
  self-model, not symbolic entailment — stated honestly as such. They are
  independent in silicon but the trace instrument cannot demonstrate it →
  **irreducible-by-instrument**, a finding about the instrument's own limits.
- **Separable** — SAT and a reachable batch reaches the distinguishing edge →
  **MEASURE-NEXT** (this is the proof that gates emission in Section 2 step 2).

**Audit obligation.** A structural-degeneracy verdict is only as sound as its
ledgered `same_source` edges, which are aie-rt-sourced, not measured. Therefore
every structural verdict ships a **falsifiable HW prediction** ("if these are one
source, no batch separates them — here is the batch most likely to; run it and
confirm it does not"). The UNSAT is a proof about the *structure*; the HW
non-separation is what ties the structure to silicon. Both are reported.

**Revisability.** `coincident` is within-eps over the *sampled* runs, so it can be
sample-luck. Every terminal degeneracy verdict is a **revisable hypothesis**: any
later batch that separates the pair retroactively invalidates the verdict. Stop
and report does not mean "never reconsider."

**The empirical/symbolic bridge.** numpy establishes and verifies the causal
edges across runs; Z3 reasons over the resulting structure symbolically. The
causal graph is the shared artifact, and every edge is either `derived` (verified)
or `structural` (ledgered) — never an invisible given.

## Section 3 — The instrument stack, and where the folds land

Five layers, each a unit with one job:

1. **Instrument (read)** — `decoder` / `parse-trace`. Raw bytes → labeled events.
   Already supports per-tile mode auto-detect (`decode_words(mode=None)`); the
   capture path just doesn't use it yet (see fold 4).
2. **Actuator (write + run)** — `trace_capture` over `RunnerSession`. HW-validated.
   The engine's "execute one MEASURE-NEXT" primitive. **Prerequisite:**
   `RunnerSession`/`ParseSession` must be *extracted* out of the 2500-line
   `trace-sweep.py` into a shared runner module before the actuator can depend on
   them without dragging in the sweep orchestration (today `HwRunner` reaches into
   `trace-sweep.py` by `importlib`).
3. **Verifier (empirical)** — **net-new**, seeded by `trace_variance`'s
   `aggregate`/`classify` (the real cross-run machinery), *not* `check_span_law`
   (a hardcoded `==64` single-run one-liner — a worked example, not a reusable
   engine). Provides: a rule/hypothesis type, a cross-run verifier harness, and
   the admitted/rejected-rule ledger.
4. **Inference engine** — NEW. Logic chainer + Z3 degeneracy reasoner.
5. **Planner** — NEW. Holds the measurement-reachability self-model; turns
   ambiguities into *proven-separating* MEASURE-NEXT batches.

**The derivability graph is a reference algorithm to port, not a fact-base
seed.** `build_derivability_graph` returns `stochastic_roots` (greedy,
order-dependent, single-pass — unverified, would smuggle `given`-style facts if
imported as-is) and `edges` (anchor-relative, deterministic events unlinked), and
it *discards* the per-run `fired` atoms that are the measured leaves of the whole
provenance DAG. The engine re-emits `fired` per run and re-derives edges under the
verified causal rule. We keep `build_derivability_graph`'s *algorithm* as
reference; we do not seed facts from its output.

**The four folds:**

- **Trace GROUPS** → **a first-class actuator phase, not a Planner knob.** Review
  confirmed groups are unbuilt: the patcher (`trace-patch-events.py`) has no
  group-enable register family, `configure_batch` cannot emit group masks, and the
  decoder has no group-slot→member attribution. Implementing groups means (a)
  reconciling which register gates AIE2 group events (emulator uses
  `0x4500+group*4`; mlir-aie-bridge lists `0x340E0`-family — resolve against the
  AM025 register DB), (b) a new patch op, (c) `configure_batch` group emission,
  (d) decoder attribution. *Then* the Planner can choose group-batches for cheap
  coverage vs. singleton-batches for disambiguation.
- **Memmod row-2 co-trace** → a *verified* reachability constraint — and the
  "memmod produced no packets" validation result is **confounded** with the core
  mode-0 forcing and with routing/enable, so the hypothesis must *control for
  routing/enable* before concluding "instrument cannot reach memmod row-2."
  Otherwise the self-model records a config bug as an instrument limit.
- **Seed / discovery** → **Phase 0** of the loop: the initial coverage sweep
  (ideal use for group-batches once built) that discovers which events fire before
  the inference loop works the real active set.
- **Mode-threading** → consolidate, but **per-tile, not per-batch**: only cores
  support EVENT_PC (mode 1); memmod/memtile/shim are always mode 0
  (`trace-sweep`'s `_build_lockstep_patch_spec` already encodes this). The
  MEASURE-NEXT directive carries per-tile mode on the write side; the read side
  switches to the decoder's existing `mode=None` auto-detect — *more* reuse than
  the first draft credited, in a different place.

**Consolidation by proven parity.** The old `trace-sweep` sweep/matrix/regression
orchestration is our regression gate. It is retired only after (a) `RunnerSession`
is extracted and (b) the engine's verdicts reproduce `trace-sweep`'s regression
findings and `trace_variance`'s `law_violations` on the same data.

## Section 4 — Testing: who verifies the verifier?

Soundness of *conclusions* = **leaf-validity ∧ rule-soundness ∧ chaining-
soundness**. The first draft's property test pins only the third; this revision
tests all three. Two axes for two failure modes — machinery bugs vs.
reality-mismatch — with different oracles.

**Axis 1 — prove the machinery correct (synthetic ground-truth, no HW), TDD:**

1. **Chaining-soundness property test.** On *every* fixpoint, assert every derived
   fact's provenance DAG bottoms out only in `measured` or `structural` leaves,
   **and every `structural` leaf is present in the audit ledger with a citation.**
   This pins chaining + the no-unaudited-axiom keystone — but *not* the other two
   obligations, which is now stated plainly.
2. **Leaf-validity.** `measured` facts carry capture provenance (run id, mode,
   traced_col, decoder version); the same `(event_key, run)` measured under
   independent conditions must agree (**replicated**, not trusted). This is the
   gate that would have caught the mode and start_col bugs the precursor hit.
3. **Rule-soundness against known answers.** Synthetic run-sets where we *know* the
   answer. Adversarial cases are the point: an offset stable for N−1 runs that
   breaks on the Nth (must reject — eps boundary); two events coincident by
   *sample luck* but independent (must not over-claim); a constant offset to a
   *deterministic* "parent" (must be `correlates`, never `derives`); a co-varying
   pair with no isolating perturbation (must stay `correlates` + finding).
4. **Degeneracy trichotomy as Z3 oracle tests.** Hand-built structures with known
   class: a `same_source` chain → assert `UNSAT(r1≠r2)` (structural, now
   reachable); independent + blocking reachability → SAT-but-no-batch
   (observational); independent + reachable → SAT-plus-emitted-batch (separable).
   Include a group-disjunction case (the union-find-can't, Z3-can witness).
5. **Closed-loop convergence + termination on a mock instrument.** Run the loop
   against a synthetic instrument emitting from a known ground-truth model. Assert
   convergence to ground truth *and* that the ranking function strictly decreases
   each iteration (the termination proof, executable).

**Axis 2 — prove the results match silicon (HW ground-truth).** On
`add_one_using_dma`: independently re-derive the capture-engine validation's 5
stochastic roots and deterministic backbone; confirm the first structural
degeneracy (shim-DMA-seen-at-memtile) *and run its falsifiable non-separation
prediction*.

**The bridge:** synthetic tests prove the machinery cannot lie; HW proves its
output matches reality. A sound engine on a wrong causal/structural model passes
Axis 1 and fails Axis 2; a lucky-on-this-kernel engine passes Axis 2 and fails
Axis 1. Both axes required.

**Parity gate (retiring the old sweep).** Differential test: engine verdicts
reproduce `trace-sweep`'s regression findings and `trace_variance`'s
`law_violations` on the same data before the old sweep is demoted.

## Reachability self-model — the one place the keystone could leak

The measurement-reachability model (what batches can run, what edge each reveals)
is consulted as ground truth by the observational-degeneracy branch, which *stops
measuring*. If it is incomplete, a separable pair is misclassified as
irreducible-by-instrument and the error is **self-sealing** (halting prevents the
contradicting measurement). Therefore the self-model is a **first-class verified
artifact**: every reachability constraint ("these two modules cannot co-trace
through one egress") carries `measured` provenance (a batch that *demonstrated*
the limit), and an observational verdict is **blocked** until every constraint it
relies on is discharged. Axis-1 adds a property test: no `irreducible-by-
instrument` verdict rests on an unverified reachability constraint. (This is why
the memmod row-2 confound above matters — it is the first such constraint.)

## Open questions deferred to implementation planning

- Concrete fact/rule representation (dataclasses vs. tuples; provenance-DAG storage
  and walking for the soundness property test).
- The reachability self-model schema (slot budget, co-trace egress constraints,
  group capabilities) the planner enumerates over.
- Z3 encoding (QF_LRA vs. QF_LIA; how `derives` equalities, `same_source`
  identities, and group disjunctions translate to constraints and how UNSAT cores
  surface as the structural proof).
- Group membership semantics — "exactly one member" vs. true disjunction — and
  whether a member fires identically alone vs. in-group (egress contention): both
  hypotheses to verify before groups feed disambiguation.
- The structural-ledger format and review workflow (how `same_source` citations
  are recorded, diffed, and audited).

## Glossary

- **anchored ts** — timestamp relative to the anchor (`1|2|0|PERF_CNT_2`),
  comparable across runs.
- **eps** — tolerance under which a cross-run std counts as stable. *Open:* a hard
  global eps on an n=6 std is fragile; implementation should consider a per-pair
  statistical test against measurement-noise variance.
- **backbone** — the deterministic-and-causal skeleton of `derives` edges.
- **stochastic root** — an event that varies across runs and is derived from no
  parent; the genuine degrees of freedom (all DMA-delivery-driven on
  `add_one_using_dma`).
- **structural support** — the audited, aie-rt-cited exception to no-axioms,
  applying only to `same_source`.
