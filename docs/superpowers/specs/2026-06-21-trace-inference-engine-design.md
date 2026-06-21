# Trace Inference Engine — Design

**Date:** 2026-06-21
**Issue:** #140 (byte-identical emulator/hardware trace reports)
**Status:** Design approved (revised three times after adversarial review,
2026-06-21); implementation plan to follow.
**Precursor:** the trace-capture engine (`tools/trace_capture.py`, HW-validated
2026-06-17) is this engine's measurement organ. See
`docs/superpowers/findings/2026-06-17-capture-engine-validation.md`.

> **Revision history (2026-06-21).** Three rounds of independent adversarial
> review shaped this design; findings archived in
> `2026-06-21-trace-inference-engine-review.md`.
> **Round 1:** the first draft equated a stable cross-run offset (arithmetic)
> with causation, and specified a vacuous degeneracy branch.
> **Round 2:** we have **no lever to perturb the silicon**, so causation cannot
> be *measured*; and at v1 the degeneracy model is pure-equality where union-find
> suffices and Z3 is premature.
> **Round 3:** we do not *need* causation at all — only **placement**. The
> backbone reconstructs *where* each event sits relative to the anchor (its
> offset), never *what causes what*. Orientation is a *derived rule over the
> loaded configuration*, not an aie-rt assertion; the oriented graph may contain
> **cycles** (condensed to irreducible groups); identity closure is separate from
> the placement structure.
> **Organizing principle: humility** — claim only what the instrument can
> *observe* (co-variation, determinism), what the loaded binary *contains*
> (configuration), and what *follows by verified rule*. Nothing else.

## Purpose

Reconstruct the **full per-kernel trace** — every configured event *placed* at a
proven offset relative to the others — by treating nondeterminism as a
**logical-inference puzzle**: from facts we have measured, the configuration we
loaded, and rules we have verified, chain to the truth; where the chain stalls,
drive the *next measurement* until it converges or proves it cannot.

The capture engine answers "trace exactly these events next, on demand, with
honest coverage." This engine is the reasoner that *decides what to measure next
and why* — turning a pile of trace scripts into one modeling instrument with a
read path, an actuator, and a reasoner.

## Placement, not causation (the round-3 clarity)

For byte-identical trace reports we need to know *where each event sits* relative
to the anchor — its offset and whether that offset is stable. We do **not** need
to know what causes what. This matters because timing-causation and dataflow
direction can diverge: a `STREAM_STARVATION` event fires from *downstream*
backpressure, so its causal arrow runs opposite to the data-flow arrow — a causal
claim would be backwards, but a *placement* claim (it sits at a stable offset
from its co-varying neighbor) is correct regardless. So `derives` means
**"dataflow-upstream-of, at a stable offset"** — pure placement — and explicitly
**disclaims timing-causation.**

## The observational regime

We do not control the silicon's stochasticity. Every run replays the same
instructions on the same inputs; the DMA, lock, and port timing we study arise
from the hardware, and we *observe their natural variation across identical
repeated runs*. There is no knob that perturbs one event independently. So
measurement yields only **undirected** facts (co-variation, offsets, determinism,
coincidence); **direction** comes from the loaded configuration, by a verified
rule.

## Keystone principle: no *unaudited* axioms

Every rule is a **hypothesis**, empirically verified against measured data before
it may derive truth; a failed rule is never used and is itself a finding. The one
thing measurement cannot give us — *direction and identity* — is not asserted on
authority; it is **derived from the configuration we loaded**, which is ground
truth *by construction* (it is the binary we are executing).

Three support types, and no fact may exist without one:

- **`measured`** — straight from capture data (a decoded, labeled event),
  *replicated* (Section 4).
- **`structural(citation)`** — a fact read directly from the **loaded
  configuration** (stream-switch routes, BD chains, lock pairings, in the
  xclbin/CDO/insts), ledgered to its exact config location. Ground truth by
  construction — not an assertion *about* hardware behavior, but a *quote of the
  binary*.
- **`derived(rule, premises)`** — produced by an admitted (verified) rule from
  existing `measured`/`structural` facts.

Note the round-3 shift: orientation and identity are **`derived`**, not primitive
`structural`. Their premises *include* `structural` config facts, but the
inference (config route ⇒ this event is upstream of that one) is a **stated,
verified rule**, not a hidden one. The keystone gets stronger: the only
non-measured primitives are *quotes of the loaded binary*.

## Build, not adopt — union-find at v1, Z3 phases in with groups

A small custom forward-chainer over the numpy verification we own. The degeneracy
reasoner's substrate is chosen by what the model contains, behind a stable
two-method interface:

- **Interface:** `same_class(a, b) -> bool` (the trichotomy's verdict query) and
  `classes() -> partition` (the planner's enumerate-all-roots need). Both backends
  expose both — union-find gives the partition natively; Z3 reconstructs it by
  pairwise `same_class` / model enumeration (cost noted in the groups-phase open
  questions).
- **v1 — union-find (congruence closure) over identity edges only.** Before
  groups, the identity structure is `same_source` equalities; equality closure is
  union-find's job and is "obviously correct by reading it." (At v1 with one or
  two identity edges this degenerates to a tiny transitive closure — union-find is
  chosen for shape and forward-compatibility, and we say so rather than pretend it
  is load-bearing yet.)
- **Groups phase — Z3 arrives.** Group implications ("group fired ⇒ member1 OR
  member2") introduce **disjunctions**; equalities + disjunctions is genuine SMT,
  beyond congruence closure. Z3 replaces union-find behind the same interface.
  Z3 is on the roadmap, sequenced to its justification — not dropped.

The chainer is the landlord; the degeneracy solver is one tenant.

## Section 1 — The knowledge base: facts and rules

**A fact** is an immutable record: `predicate`, `args`, **support**
(`measured` | `structural(citation)` | `derived(rule, premises)`).

| Predicate | Support | Meaning |
|-----------|---------|---------|
| `fired(event_key, run, anchored_ts)` | measured | a decoded event; `event_key` = 4-part `col\|row\|pkt\|name` |
| `config_path(a, b, cite)` | structural | the loaded configuration routes a's producer to b's consumer (stream route / BD chain / lock pairing), cited to its config location |
| `deterministic(event_key)` | derived | `std(ts)` across runs < eps |
| `correlates(a, b, offset)` | derived | `std(a.ts − b.ts) < eps` across runs — *symmetric, undirected* co-variation. The measured placement substrate. |
| `derives(child, parent, offset)` | derived | `correlates(child, parent, offset)` + stochastic parent + a `config_path(parent, child)` orientation. Means **dataflow-upstream + stable offset = placement**, *not* timing-causation. |
| `same_source(a, b)` | derived | a and b are the *same physical event* at two trace units — premises: a `config_path` showing identity + measured coincidence. |
| `stochastic_root(event_key)` | derived | varies across runs **and** no parent derives it |
| `group_fired(group_key, run, ts)` | measured | a group slot fired *(groups phase)* |
| `group_member(group_key, member)` | derived | membership, verified on data *(groups phase)* |
| `coincident(a, b)` | derived | a and b fire at identical anchored ts (within eps) in *every* measured run — a degeneracy *candidate*, revisable |

**`derives` admission** (all three): (1) `correlates` holds (measured stable
offset); (2) the parent is stochastic (a deterministic parent transmits no
jitter, so a constant offset to it is not placement-derivation — excludes the
spurious deterministic-clique graph); (3) a `config_path(parent, child)` exists,
giving orientation by verified rule, citing the config location. A co-varying pair
with **no `config_path`** stays `correlates` (undirected) and routes to degeneracy
— we never invent a direction we cannot derive from the loaded binary.

**A rule** is a hypothesis paired with a verifier; for the numeric rules the rule
body and verifier coincide. The KB is four sets: **facts**, **admitted rules**,
**rejected rules** (each a finding), and the **structural ledger** (every
`config_path` citation — auto-generated from the loaded configuration, so the
audit target is the *generator + spot-check*, not N hand-written entries).

## Section 2 — The loop: chain to fixpoint, then act on the stall

1. **Chain** to fixpoint over `measured`/`structural` leaves. Each event ends
   `derived` (placed on the backbone), `stochastic_root`, or **unresolved**.

2. **Inspect the stall:**
   - **Not-yet-measured / not-yet-co-traced** — the pair was never in one batch,
     so `correlates` could not be measured → **MEASURE-NEXT** (co-trace them). This
     is the genuine-gain case.
   - **Ambiguous-but-separable** — multiple candidates, *and the planner first
     proves a reachable batch distinguishes them* → emit that batch.
   - **Provably degenerate** — Section 2b → **stop and report** (provisionally).

   **Progress is guaranteed.** MEASURE-NEXT is emitted only after the planner
   proves the batch adds a separating/co-tracing gain — never emit-then-discover;
   a *fully-measured* tight `correlates` pair with a stable offset and no
   orientation goes **straight to observational degeneracy without burning a
   batch**. Termination rests on a three-component lexicographic measure:
   `(# configured-but-unfired events, # unresolved fired events, # untested
   candidate edges)`. The top component's domain is the **static configured-event
   set from the xclbin** (fixed and finite, independent of which batch observes
   it). A discovery batch that surfaces a never-before-fired event raises the
   middle count but strictly *decreases the top* (an event moves unfired→fired);
   each component is bounded below and strictly decreases at its level. No livelock
   branch exists — "ambiguous and no batch separates" is degeneracy (it halts),
   never re-queued.

3. **Converge or halt.** Worklist empty (full trace placed) or every remainder
   provably degenerate (halt with a precise, *falsifiable* report).

## Section 2b — Degeneracy: union-find now, Z3 with groups

The placement structure (`derives` edges) reduces events toward stochastic roots.
**The oriented graph may contain cycles** — circular BD chains, lock round-trips,
ping-pong double-buffering are real in AIE configuration. The reducer **detects
cycles and condenses each strongly-connected component into a single irreducible
group**, reported as such; the reduction then runs over the acyclic condensation.
(The round-1 "rooted forest" assumption is satisfied by the *condensation*, not
asserted of the raw graph.)

The identity structure is **separate**: a union-find equality closure over
**`same_source` edges only**. A non-zero `derives` offset is *proof two events are
not identical*, so `derives` edges never enter the identity closure — they are the
placement DAG, not equality edges.

For a coincident root pair `(r1, r2)`, the reasoner classifies:

- **Structurally degenerate** — `same_class(r1, r2)` (identity closure over
  `same_source`). They are one physical event, named honestly as an **identity
  collapse over audited config-derived edges**, not symbolic entailment. **Gate:**
  the verdict is **provisional (`structural-candidate`)** until its falsifiable HW
  prediction is run — "if one source, no batch separates them; here is the batch
  most likely to" — and confirmed non-separating. If that confirmation batch is
  *itself unreachable* per the verified self-model, the verdict downgrades to
  **`unconfirmable-structural`** — a distinct, honest finding ("we cite a
  structure but cannot run the experiment that would falsify it"), never a
  silently-trusted collapse and never an infinite candidate.
- **Observationally degenerate** — not same-class, **and** finite enumeration over
  the verified reachability self-model finds **no reachable batch** that separates
  them → **irreducible-by-instrument**. (Honest about being enumeration, not
  entailment.)
- **Separable** — a reachable batch reaches a distinguishing edge → MEASURE-NEXT.

**Groups phase (Z3).** Group disjunctions make equality closure incomplete; Z3
replaces union-find behind the `same_class`/`classes` interface and the structural
branch becomes genuine `UNSAT(r1 ≠ r2)` entailment over equalities + disjunctions.

**Revisability.** `coincident` is within-eps over *sampled* runs, so every
terminal degeneracy verdict is a revisable hypothesis: any later separating batch
retroactively invalidates it.

## Section 3 — The instrument stack, and where the folds land

Five layers, each a unit with one job:

1. **Instrument (read)** — `decoder` / `parse-trace`. Already supports per-tile
   mode auto-detect (`decode_words(mode=None)`); the capture path doesn't use it
   yet (fold 4).
2. **Actuator (write + run)** — `trace_capture` over `RunnerSession`. HW-validated.
   **Prerequisite:** extract `RunnerSession`/`ParseSession` out of the 2500-line
   `trace-sweep.py` into a shared runner module before the actuator can depend on
   them without dragging in the sweep orchestration.
3. **Verifier (empirical)** — net-new in shape (rule/hypothesis type, cross-run
   harness, admitted/rejected ledger), but the cross-run-offset machinery already
   exists: `trace_variance.aggregate` + `trace_join.pair_derivability` already
   compute `std(a.ts − b.ts)` — exactly the `correlates` verifier. *Not*
   `check_span_law` (a hardcoded `==64` one-liner).
4. **Inference engine** — NEW. Logic chainer + degeneracy reasoner (union-find;
   Z3 in the groups phase).
5. **Planner** — NEW. Holds the reachability self-model; turns ambiguities into
   *proven-gain* MEASURE-NEXT batches.

**The derivability graph is a reference algorithm to port, not a fact-base seed.**
Its roots are greedy/order-dependent/unverified; it *discards* the per-run `fired`
atoms — *but those atoms survive on disk* in `batch_*/hw/trace.events.json`
(written by `capture()`), so re-emitting `fired` per run needs no upstream change.
We re-emit `fired` and re-derive edges under the verified rule, keeping the
*algorithm* as reference only.

**The four folds:**

- **Trace GROUPS** → **a first-class actuator phase** (and the phase that brings Z3
  online). Unbuilt: the patcher has no group-enable register family,
  `configure_batch` cannot emit group masks, the decoder has no group-slot→member
  attribution. Implementing means (a) reconciling which register gates AIE2 group
  events (emulator `0x4500+group*4` vs. mlir-aie-bridge `0x340E0`-family — resolve
  against the AM025 register DB), (b) a new patch op, (c) `configure_batch` group
  emission, (d) decoder attribution. *Then* the Planner trades group-batches
  against singleton-batches, and Z3 replaces union-find for the now-disjunctive
  model.
- **Memmod row-2 co-trace** → a *verified* reachability constraint — and "no memmod
  packets" is **confounded** with core mode-0 forcing and routing/enable, so the
  hypothesis must *control for routing/enable* before concluding "instrument cannot
  reach memmod row-2."
- **Seed / discovery** → **Phase 0**: the initial coverage sweep that discovers
  which events fire (the ranking function's top component tracks its progress
  against the static xclbin event set).
- **Mode-threading** → consolidate **per-tile, not per-batch**: only cores support
  EVENT_PC (mode 1); memmod/memtile/shim are always mode 0. The MEASURE-NEXT
  directive carries per-tile mode on the write side; the read side uses the
  decoder's existing `mode=None` auto-detect.

**Consolidation by proven parity.** The old `trace-sweep` orchestration is our
regression gate, retired only after (a) `RunnerSession` is extracted and (b) the
engine's verdicts reproduce `trace-sweep`'s regression findings and
`trace_variance`'s `law_violations` on the same data.

## Section 4 — Testing: who verifies the verifier?

Soundness of conclusions = **leaf-validity ∧ rule-soundness ∧ chaining-
soundness**, with leaf-validity in two sub-cases. Two axes for two failure modes.

**Axis 1 — prove the machinery correct (synthetic ground-truth, no HW), TDD:**

1. **Chaining-soundness property test.** On every fixpoint, every derived fact's
   provenance DAG bottoms out only in `measured` or `structural` leaves, and every
   `structural` leaf is in the ledger with a config citation. Pins chaining + the
   no-unaudited-axiom keystone.
2. **Leaf-validity, two sub-cases.** *Measured leaves* → **replication** (same
   `(event_key, run)` under independent conditions must agree — would have caught
   the precursor's mode and start_col bugs). *Structural leaves* → the citation
   must resolve to a real config location in the loaded binary, and any
   degeneracy verdict it drives is gated on the falsifiable-non-separation HW
   discharge (Section 2b).
3. **Rule-soundness against known answers.** Adversarial synthetic run-sets: an
   offset stable for N−1 runs that breaks on the Nth (reject — eps boundary); a
   sample-luck coincident-but-independent pair (don't over-claim); a constant
   offset to a *deterministic* parent (must be `correlates`, never `derives`); a
   co-varying pair with no `config_path` (must stay `correlates`); **a backpressure
   event whose timing-cause opposes its dataflow direction (must still be *placed*
   correctly and must NOT be labeled causal).**
4. **Degeneracy classification oracle.** v1 union-find: a `same_source` chain
   collapses r1, r2 into one class (structural); independent + blocking reachability
   → no-batch (observational); independent + reachable → emitted batch (separable);
   **a cycle (lock round-trip) condenses to one irreducible group.** Groups phase
   adds the Z3 disjunction case.
5. **Closed-loop convergence + termination on a mock instrument.** Converge to a
   known ground-truth model; assert the three-component ranking function strictly
   decreases each iteration including a discovery step that surfaces a new fired
   event (top must absorb it).

**Axis 2 — prove the results match silicon (HW ground-truth).** On
`add_one_using_dma`: independently re-derive the validation's 5 stochastic roots
and the deterministic backbone; take the first `same_source` candidate
(shim-DMA-seen-at-memtile) through the **full gate** — emit the non-separation
prediction, run it, confirm before trusting the structural verdict.

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
constraint carries `measured` provenance (a batch that *demonstrated* the limit),
and an observational verdict is **blocked** until every constraint it relies on is
discharged. Axis-1 adds a property test: no `irreducible-by-instrument` verdict
rests on an unverified reachability constraint. (The memmod row-2 confound is the
first such constraint.)

## Open questions deferred to implementation planning

- Concrete fact/rule representation (dataclasses vs. tuples; provenance-DAG storage
  and walking).
- The `config_path` derivation rule: exactly how stream-switch routes / BD chains /
  lock pairings in the loaded binary map to event-pair orientation, and its
  verifier.
- The reachability self-model schema (slot budget, co-trace egress constraints,
  group capabilities).
- `eps` calibration — a hard global eps on an n=6 std is fragile and is itself a
  soundness precondition; consider a per-pair statistical test against
  measurement-noise variance, and how many runs give it power.
- The structural-ledger format and generator-audit workflow.
- Groups-phase specifics: membership semantics ("exactly one" vs. true
  disjunction), member-fires-identically-alone-vs-in-group (egress contention), the
  Z3 encoding (QF_LRA vs. QF_LIA; UNSAT cores as the structural proof; partition
  reconstruction cost from a SAT/UNSAT backend).

## Glossary

- **anchored ts** — timestamp relative to the anchor (`1|2|0|PERF_CNT_2`),
  comparable across runs.
- **placement, not causation** — the backbone reconstructs *where* an event sits
  (its offset), not *what causes it*; `derives` = dataflow-upstream + stable
  offset, with timing-causation explicitly disclaimed.
- **observational regime** — we observe natural cross-run variation but cannot
  perturb it; hence orientation is config-derived, not measured.
- **structural support** — a *quote of the loaded binary* (route / BD / lock
  config), ground truth by construction; the audited input from which orientation
  and identity are *derived* by verified rule.
- **stochastic root** — an event that varies across runs and is derived from no
  parent; the genuine degrees of freedom (all DMA-delivery-driven on
  `add_one_using_dma`).
