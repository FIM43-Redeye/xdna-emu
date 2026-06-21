# Trace Inference Engine — Design

**Date:** 2026-06-21
**Issue:** #140 (byte-identical emulator/hardware trace reports)
**Status:** Design approved (revised twice after adversarial review,
2026-06-21); implementation plan to follow.
**Precursor:** the trace-capture engine (`tools/trace_capture.py`, HW-validated
2026-06-17) is this engine's measurement organ. See
`docs/superpowers/findings/2026-06-17-capture-engine-validation.md`.

> **Revision history (2026-06-21).** Two rounds of independent adversarial
> review shaped this design; findings archived in
> `2026-06-21-trace-inference-engine-review.md`.
> **Round 1** caught the first draft equating a verified *arithmetic* property
> (stable cross-run offset) with causation, and a vacuous degeneracy branch.
> **Round 2** caught the deeper truth behind round 1: we have **no lever to
> perturb the silicon**, so causation cannot be *measured* at all — and that at
> v1 the degeneracy model is pure-equality, where union-find suffices and Z3 is
> premature.
> **The organizing principle of this revision is humility:** claim only what the
> instrument can *observe* (co-variation, determinism) plus what we can *audit*
> (aie-rt structure). We stop inventing capabilities (intervention) to prop up
> stronger claims. Concretely: backbone **orientation is structural**
> (aie-rt-cited), not measured-causal; the degeneracy reasoner ships **union-find
> at v1** and **Z3 phases in with trace groups** (when disjunctions make SMT
> genuinely necessary).

## Purpose

Reconstruct the **full per-kernel trace** — every configured event placed in a
proven relationship to the others — by treating nondeterminism as a
**logical-inference puzzle**: given facts we have measured to be true, follow a
chain of verified rules (and audited structural facts) to the truth, and where
the chain stalls, drive the *next measurement* until it converges or proves it
cannot.

The capture engine answers "trace exactly these events next, on demand, with
honest coverage." This engine is the reasoner that *decides what to measure next
and why*, turning a pile of trace scripts into a single modeling instrument with
a read path, an actuator, and a reasoner.

## The observational regime (why orientation is structural)

We do **not** control the silicon's stochasticity. Every run replays the same
instructions on the same inputs; the DMA arbitration, lock-stall, and port
timing we study arise from the hardware, and we *observe their natural variation
across identical repeated runs*. There is no knob that perturbs one event's
timing independently of others. This is **causal discovery from observational
data**, which cannot orient an edge (distinguish cause from effect, or direct
cause from common cause) from measurement alone.

So we are honest about the division: **measurement gives us undirected facts**
(co-variation, offsets, determinism, coincidence); **direction and identity come
from aie-rt structure**, audited and ledgered. We never claim to have *measured*
causation. The backbone is *measured co-variation, structurally oriented*.

## Keystone principle: no *unaudited* axioms

Every rule is a **hypothesis**, empirically verified against measured data before
it may derive truth; a rule that fails is never used, and its failure is a
finding. The one thing measurement cannot give us — *direction and physical
identity* — comes from aie-rt, and rather than smuggle it in disguised as
measurement, we make it **visible and citable**.

Three support types, and no fact may exist without one:

- **`measured`** — straight from capture data (a decoded, labeled event), and
  *replicated* (see Section 4).
- **`derived(rule, premises)`** — produced by an admitted (verified) rule from
  existing facts.
- **`structural(citation)`** — an aie-rt-sourced fact that measurement cannot
  establish, recorded in an auditable **ledger** with its citation. It applies to
  exactly two things — **physical identity** (`same_source`) and **dataflow
  direction** (the orientation of a `derives` edge) — and every instance is
  enumerable for review.

So the keystone has teeth *and* honesty: a fact is `measured`, `derived` from
verified rules, or a **listed, citable** `structural` exception. Nothing hides.

## Build, not adopt — union-find at v1, Z3 phases in with groups

A small custom forward-chainer over the numpy verification we own. The degeneracy
reasoner's substrate is **chosen by what the model actually contains**:

- **v1 — union-find (congruence closure).** Before trace groups exist, the causal
  structure is *pure equalities* (`same_source` identities + `derives` offset
  equalities). Equality closure is exactly union-find's job, it is "obviously
  correct by reading it," and it is complete for this model. Z3 would be
  speculative infrastructure here — its only advantage over union-find is
  disjunctive reasoning, which the v1 model has none of.
- **Groups phase — Z3 arrives.** Trace-group implications ("group fired ⇒ member1
  OR member2") introduce **disjunctions** into the structure. Equalities +
  disjunctions is genuine SMT, beyond congruence closure — *that* is where Z3
  earns its place, and it ships in the same phase as groups, behind the same
  clean predicate interface union-find used. Z3 is on the roadmap, sequenced to
  its justification, not dropped.

Adopt options lose for the bulk of the work (pyDatalog dead; Soufflé pure-logic,
batch, heavy; RETE engines unmaintained or business-rules-shaped). The chainer is
the landlord; the degeneracy solver (union-find, later Z3) is one tenant.

## Section 1 — The knowledge base: facts and rules

**A fact** is an immutable record: `predicate`, `args`, and a **support** field
(`measured` | `derived(rule, premises)` | `structural(citation)`). A fact with no
valid support cannot exist.

| Predicate | Support | Meaning |
|-----------|---------|---------|
| `fired(event_key, run, anchored_ts)` | measured | a decoded event; `event_key` = 4-part `col\|row\|pkt\|name` |
| `deterministic(event_key)` | derived | `std(ts)` across runs < eps |
| `correlates(a, b, offset)` | derived | `std(a.ts − b.ts) < eps` across runs — *symmetric, undirected* co-variation. The measured backbone substrate. |
| `derives(child, parent, offset)` | derived + structural | `correlates(child, parent, offset)` **plus** a `structural` aie-rt dataflow orientation child←parent. The oriented backbone edge. |
| `stochastic_root(event_key)` | derived | varies across runs **and** no parent derives it |
| `same_source(a, b)` | structural | a and b are the *same physical hardware event* at two trace units (one DMA transfer seen at shim and memtile). aie-rt-cited, ledgered. |
| `group_fired(group_key, run, ts)` | measured | a group slot fired *(groups phase)* |
| `group_member(group_key, member)` | derived | membership, verified on data *(groups phase)* |
| `coincident(a, b)` | derived | a and b fire at identical anchored ts (within eps) in *every* measured run — a degeneracy *candidate*, revisable |

**`derives` = measured co-variation + structural orientation (Theme-1 fix,
round 2).** Stable offset alone is symmetric and cannot orient; intervention is
not available to us (the observational regime, above). So a `derives` edge is
admitted only when:

1. **`correlates(child, parent, offset)` holds** — measured, the offset is stable
   across runs.
2. **The parent is stochastic** — a deterministic "parent" transmits no jitter,
   so a constant offset to it is not derivation; this excludes the spurious
   deterministic-clique complete graph.
3. **aie-rt cites a dataflow path** parent→child — a `structural` orientation,
   ledgered. This is the *only* source of direction; we do not pretend to measure
   it.

A co-varying pair with **no aie-rt dataflow citation** stays `correlates`
(undirected) and routes to the degeneracy analysis — we never invent a direction
we cannot cite. This keeps the keystone honest: the measured part (co-variation)
is measured; the structural part (direction) is audited.

**A rule** is a hypothesis paired with a verifier; for the numeric rules the rule
body and its verifier coincide. The KB is four sets: **facts**, **admitted
rules**, **rejected rules** (each a finding), and the **structural ledger** (every
`same_source` and every `derives` orientation citation, for audit).

## Section 2 — The loop: chain to fixpoint, then act on the stall

1. **Chain** every admitted rule to fixpoint over `measured`/`structural` leaves.
   Each event ends `derived` (on the oriented backbone), `stochastic_root`, or
   **unresolved**.

2. **Inspect the stall.** Each unresolved event has a diagnosable reason picking
   the next action:
   - **Not-yet-measured** — never co-traced with a candidate in one batch → emit
     **MEASURE-NEXT**.
   - **Ambiguous-but-separable** — multiple candidates, *and the planner first
     proves a reachable batch distinguishes them* → emit that proven-separating
     batch.
   - **Provably degenerate** — Section 2b returns structural or observational
     degeneracy → **stop and report** (provisionally — see the gate below).

   **Progress is guaranteed, not hoped-for.** MEASURE-NEXT is emitted *only* after
   the planner proves the batch adds a separating edge. Termination rests on a
   well-founded **lexicographic ranking function with three components**:
   `(# configured-but-unfired events, # unresolved fired events, # untested
   candidate edges)`. The top component covers **discovery**: a batch may surface
   a never-before-fired event (composition-dependent), which would raise the
   middle count — but it strictly *decreases the top* (an event moves from
   unfired to fired), and the top is bounded below by the finite configured-event
   set. Each component is bounded below and strictly decreases at its level; no
   livelock branch exists because "ambiguous and no batch separates" is degeneracy
   (it halts), never re-queued.

3. **Converge or halt.** Worklist empty (full trace reconstructed) or every
   remainder provably degenerate (halt with a precise, *falsifiable*
   irreducibility report).

## Section 2b — Degeneracy: union-find now, Z3 with groups

Every event reduces, through oriented `derives` edges and `same_source`
identities, toward stochastic roots — a **DAG**, because orientation comes from
aie-rt's (acyclic) dataflow, not from a symmetric measured predicate. The only
irreducible question is then: **are two stochastic roots the same source, or two
sources that co-vary?**

For a coincident root pair `(r1, r2)`, the reasoner classifies:

- **Structurally degenerate** — the equality closure (union-find over
  `same_source` + `derives`-offset edges) puts r1 and r2 in the same class. They
  are provably one source *relative to the ledgered structure*. **This is the
  honest version of the round-1 "UNSAT" claim:** with `same_source` an aie-rt
  fact, this *is* an identity collapse, and union-find names it as exactly that —
  an identity collapse over audited edges — rather than dressing it as symbolic
  entailment. **Gate:** the verdict is **provisional (`structural-candidate`)
  until its falsifiable HW prediction is run** — "if these are one source, no
  batch separates them; here is the batch most likely to" — and confirmed
  non-separating. Only then is it `structurally-degenerate`. A wrong (mis-cited)
  `same_source` is caught here, before the verdict is trusted, symmetric with how
  the reachability model gates observational verdicts.
- **Observationally degenerate** — not in the same class, **and** finite
  enumeration over the (verified) reachability self-model finds **no reachable
  batch** that separates them. Honest about being enumeration, not entailment.
  → **irreducible-by-instrument**, a finding about the instrument's limits. *But*
  a strong `correlates` pair (co-varies tightly, no aie-rt orientation) is a
  **prime MEASURE-NEXT candidate first** — the planner must attempt separation
  before reporting irreducible, so we never launder "we lack a separating batch"
  into "silicon is indistinguishable."
- **Separable** — a reachable batch reaches a distinguishing edge → MEASURE-NEXT.

**Groups phase (Z3).** When group implications add disjunctions, equality closure
is no longer complete — a root's identity can be mediated through a disjunctive
group constraint that only SMT can resolve. Z3 replaces union-find as the
degeneracy solver behind the same interface, and the structural branch becomes a
genuine `UNSAT(r1 ≠ r2)` entailment over equalities + disjunctions.

**Revisability.** `coincident` is within-eps over *sampled* runs, so every
terminal degeneracy verdict is a revisable hypothesis: any later separating batch
retroactively invalidates it.

## Section 3 — The instrument stack, and where the folds land

Five layers, each a unit with one job:

1. **Instrument (read)** — `decoder` / `parse-trace`. Already supports per-tile
   mode auto-detect (`decode_words(mode=None)`); the capture path just doesn't use
   it yet (fold 4).
2. **Actuator (write + run)** — `trace_capture` over `RunnerSession`. HW-validated.
   **Prerequisite:** extract `RunnerSession`/`ParseSession` out of the 2500-line
   `trace-sweep.py` into a shared runner module before the actuator can depend on
   them without dragging in the sweep orchestration.
3. **Verifier (empirical)** — **net-new**, seeded by `trace_variance`'s
   `aggregate`/`classify` (the real cross-run machinery), *not* `check_span_law`
   (a hardcoded `==64` single-run one-liner). Provides the rule/hypothesis type,
   the cross-run verifier harness, and the admitted/rejected-rule ledger.
4. **Inference engine** — NEW. Logic chainer + degeneracy reasoner (union-find;
   Z3 in the groups phase).
5. **Planner** — NEW. Holds the measurement-reachability self-model; turns
   ambiguities into *proven-separating* MEASURE-NEXT batches.

**The derivability graph is a reference algorithm to port, not a fact-base
seed.** `build_derivability_graph`'s roots are greedy/order-dependent/unverified
and it *discards* the per-run `fired` atoms that are the measured leaves. The
engine re-emits `fired` per run and re-derives edges under the verified rule. We
keep the *algorithm* as reference; we do not seed facts from its output.

**The four folds:**

- **Trace GROUPS** → **a first-class actuator phase** (and the phase that brings Z3
  online). Groups are unbuilt: the patcher has no group-enable register family,
  `configure_batch` cannot emit group masks, the decoder has no group-slot→member
  attribution. Implementing groups means (a) reconciling which register gates AIE2
  group events (emulator `0x4500+group*4` vs. mlir-aie-bridge `0x340E0`-family —
  resolve against the AM025 register DB), (b) a new patch op, (c) `configure_batch`
  group emission, (d) decoder attribution. *Then* the Planner trades group-batches
  (cheap coverage) against singleton-batches (disambiguation), and Z3 replaces
  union-find for the now-disjunctive degeneracy model.
- **Memmod row-2 co-trace** → a *verified* reachability constraint — and the "no
  memmod packets" validation result is **confounded** with core mode-0 forcing and
  routing/enable, so the hypothesis must *control for routing/enable* before
  concluding "instrument cannot reach memmod row-2."
- **Seed / discovery** → **Phase 0**: the initial coverage sweep that discovers
  which events fire before the inference loop works the real active set (the top
  ranking-function component tracks its progress).
- **Mode-threading** → consolidate **per-tile, not per-batch**: only cores support
  EVENT_PC (mode 1); memmod/memtile/shim are always mode 0. The MEASURE-NEXT
  directive carries per-tile mode on the write side; the read side uses the
  decoder's existing `mode=None` auto-detect.

**Consolidation by proven parity.** The old `trace-sweep` orchestration is our
regression gate, retired only after (a) `RunnerSession` is extracted and (b) the
engine's verdicts reproduce `trace-sweep`'s regression findings and
`trace_variance`'s `law_violations` on the same data.

## Section 4 — Testing: who verifies the verifier?

Soundness of *conclusions* = **leaf-validity ∧ rule-soundness ∧ chaining-
soundness**, and leaf-validity has **two sub-cases** because there are two
non-derived leaf kinds. Two axes for two failure modes.

**Axis 1 — prove the machinery correct (synthetic ground-truth, no HW), TDD:**

1. **Chaining-soundness property test.** On every fixpoint, assert every derived
   fact's provenance DAG bottoms out only in `measured` or `structural` leaves,
   **and every `structural` leaf is present in the ledger with a citation.** Pins
   chaining + the no-unaudited-axiom keystone (presence, not correctness — the
   other two obligations cover correctness).
2. **Leaf-validity, two sub-cases.**
   - *Measured leaves* → **replication**: the same `(event_key, run)` measured
     under independent conditions must agree (would have caught the precursor's
     mode and start_col bugs).
   - *Structural leaves* → **falsifiable-non-separation discharge**: a
     `same_source`/orientation citation that drives a degeneracy verdict is not
     *trusted* until its HW non-separation prediction is run and confirmed (the
     Section-2b gate, as a test). A mis-cited structural leaf is caught here.
3. **Rule-soundness against known answers.** Adversarial synthetic run-sets: an
   offset stable for N−1 runs that breaks on the Nth (reject — eps boundary); a
   sample-luck coincident-but-independent pair (don't over-claim); a constant
   offset to a *deterministic* parent (must be `correlates`, never `derives`); a
   co-varying pair with no aie-rt orientation (must stay `correlates`).
4. **Degeneracy classification oracle.** v1: union-find oracle tests — a
   `same_source` chain collapses r1, r2 into one class (structural); independent +
   blocking reachability → no-batch (observational); independent + reachable →
   emitted batch (separable). Groups phase adds the Z3 disjunction case
   (union-find-can't, Z3-can).
5. **Closed-loop convergence + termination on a mock instrument.** Converge to a
   known ground-truth model, and assert the three-component ranking function
   strictly decreases each iteration **including a discovery step that surfaces a
   new fired event** (the top component must absorb it).

**Axis 2 — prove the results match silicon (HW ground-truth).** On
`add_one_using_dma`: independently re-derive the validation's 5 stochastic roots
and the deterministic backbone; take the first `same_source` candidate
(shim-DMA-seen-at-memtile) through the **full gate** — emit the non-separation
prediction, run it, confirm non-separation before trusting the structural verdict.

**The bridge:** synthetic tests prove the machinery cannot lie; HW proves its
output matches reality. Both axes required.

**Parity gate (retiring the old sweep).** Engine verdicts reproduce
`trace-sweep`'s regression findings and `trace_variance`'s `law_violations` on the
same data before the old sweep is demoted.

## Reachability self-model — the one place the keystone could leak

The reachability model (what batches can run, what edge each reveals) is consulted
as ground truth by the observational branch, which *stops measuring*. If
incomplete, a separable pair is misclassified as irreducible and the error is
**self-sealing**. So the self-model is a **first-class verified artifact**: every
constraint ("these two modules cannot co-trace through one egress") carries
`measured` provenance (a batch that *demonstrated* the limit), and an
observational verdict is **blocked** until every constraint it relies on is
discharged. Axis-1 adds a property test: no `irreducible-by-instrument` verdict
rests on an unverified reachability constraint. (The memmod row-2 confound is the
first such constraint.)

## Open questions deferred to implementation planning

- Concrete fact/rule representation (dataclasses vs. tuples; provenance-DAG storage
  and walking).
- The reachability self-model schema (slot budget, co-trace egress constraints,
  group capabilities).
- `eps` calibration — a hard global eps on an n=6 std is fragile and is itself a
  soundness precondition (a wrong eps mis-fires `deterministic`/`correlates`/
  `coincident`); consider a per-pair statistical test against measurement-noise
  variance, and state how many runs give the test power.
- The structural-ledger format and review workflow (how `same_source` and
  orientation citations are recorded, diffed, audited).
- Groups-phase specifics: group membership semantics ("exactly one member" vs.
  true disjunction), whether a member fires identically alone vs. in-group (egress
  contention), and the Z3 encoding (QF_LRA vs. QF_LIA; UNSAT cores as the
  structural proof).

## Glossary

- **anchored ts** — timestamp relative to the anchor (`1|2|0|PERF_CNT_2`),
  comparable across runs.
- **observational regime** — we observe natural cross-run variation but cannot
  perturb it; hence orientation is structural, not measured.
- **backbone** — measured co-variation (`correlates`), structurally oriented
  (`derives`) where aie-rt cites a dataflow path.
- **stochastic root** — an event that varies across runs and is derived from no
  parent; the genuine degrees of freedom (all DMA-delivery-driven on
  `add_one_using_dma`).
- **structural support** — the audited, aie-rt-cited exception to no-axioms,
  applying only to identity (`same_source`) and direction (`derives` orientation).
