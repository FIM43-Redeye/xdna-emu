# Trace Inference Engine — Adversarial Review Findings (first draft)

**Date:** 2026-06-21
**Subject:** `2026-06-21-trace-inference-engine-design.md` (first draft)
**Method:** three independent adversarial reviewers — logical soundness, the Z3
degeneracy model, and feasibility-vs-code — each told to find holes, not praise,
and to read the actual tool source where the spec made claims about it.

This archives *why* the spec was revised. The revised spec addresses every
BLOCKER and SHOULD-FIX below; line references are to the first draft.

## Reviewer A — logical soundness

- **A1 (BLOCKER).** The no-axiom property test ("every derived fact bottoms out in
  `measured` leaves") pins *chaining*-soundness, not *conclusion*-soundness. A
  wrong `measured` leaf or a bad admitted rule both pass it green — and the
  precursor doc shows two wrong `measured` leaves actually happened (mode,
  start_col). → Fixed: soundness split into leaf-validity ∧ rule-soundness ∧
  chaining; leaf-validity via replicated measurement (Section 4 tests 1–3).
- **A2 (BLOCKER).** `derives` conflates constant-offset with causation. Offset is
  symmetric (can't orient) and non-causal (two deterministic events share it
  trivially → spurious complete graph; two co-varying independent stochastic
  events share it → two DOFs collapsed before Z3 sees them). → Fixed: `correlates`
  (symmetric, non-backbone) vs. `derives` (parent-stochastic + variance-transfer
  intervention + honest `correlates` fallback).
- **A3 (SHOULD-FIX→BLOCKER).** Multi-valid-parent "ambiguous" has no well-defined
  resolution; "rivals alone" re-confirms both edges → no ambiguity reduction →
  **livelock**. "Strictly reduces ambiguity" was asserted as a *test*, not
  guaranteed by design. → Fixed: MEASURE-NEXT gated by a prior separating-proof;
  "ambiguous and no batch separates" routes to degeneracy, not back to the loop;
  ranking function added.
- **A4 (BLOCKER).** The reducibility collapse assumes a rooted forest the
  symmetric predicate can't produce (2-cycles everywhere; no-root components;
  diamonds = multi-root events). → Fixed: causal `derives` is oriented → DAG;
  no-root deterministic cliques reported as such; degeneracy reasons over the DAG.
- **A5 (SHOULD-FIX).** Structural-degeneracy UNSAT is sound only relative to a
  hand-built encoding, re-introducing an axiom in the corner marketed "with a
  proof." → Fixed: structural verdicts ship a falsifiable HW non-separation
  prediction; the audit ledger makes the structural input visible.
- **A6 (SHOULD-FIX).** A single global `eps` silently controls every predicate's
  truth on an n=6 std. → Noted in glossary as an implementation open question
  (per-pair statistical test).
- **A7 (SHOULD-FIX).** Termination lacked a well-founded ranking function. → Fixed
  (Section 2 step 2).

## Reviewer B — the Z3 degeneracy model

- **B1 (BLOCKER).** The structural-degeneracy UNSAT branch is **vacuous as
  specified**: two stochastic roots are disconnected by construction (they are
  roots *because* no offset-edge links them), so `structure ∧ (r1≠r2)` is always
  SAT. The trichotomy silently collapsed to a dichotomy — the very failure 2b
  existed to prevent. → Fixed: `same_source` identity edges can connect roots,
  making UNSAT reachable.
- **B2 (BLOCKER).** No predicate encoded "same physical source," and identity is
  arguably unmeasurable at trace-event level (readout timestamps can't distinguish
  one-event-read-twice from two-events-with-forced-zero-gap). Sourcing it from
  aie-rt would violate the keystone. → Fixed via the decision below: `same_source`
  + audited `structural` support type (Option A).
- **B3 (SHOULD-FIX).** "Observationally degenerate" is finite enumeration over the
  reachability model, not symbolic entailment — mislabeled, and if B1 stood, Z3's
  entire justification collapsed. → Fixed: observational branch stated honestly as
  enumeration; Z3 re-justified by `same_source`/`derives` equalities mixed with
  group **disjunctions** (genuine SMT, beyond union-find).
- **B4 (SHOULD-FIX).** The reachability self-model is the one unguarded axiom; a
  false observational verdict is self-sealing (halting prevents contradiction). →
  Fixed: self-model is a first-class verified artifact (dedicated Section +
  property test).
- **B5 (MINOR).** `coincident` within-eps inherits eps-boundary fragility at the
  degeneracy gate. → Fixed: terminal verdicts are revisable hypotheses.
- **B6 (MINOR).** Group disjunctions are the one non-linear construct and a
  possible genuine UNSAT path; spec didn't connect them to degeneracy. → Fixed:
  group disjunctions are now an explicit Z3 justification.

## Reviewer C — feasibility vs. code

- **C1 (BLOCKER).** Trace GROUPS are unbuilt — no group-enable register family in
  `trace-patch-events.py`, no `configure_batch` emission, no decoder attribution
  (and an unreconciled register-address discrepancy: emulator `0x4500+group*4`
  vs. mlir-aie-bridge `0x340E0`-family). It is a phase, not a Planner knob. →
  Fixed: reclassified as a first-class actuator phase with the four sub-tasks.
- **C2 (SHOULD-FIX).** `build_derivability_graph` provides ~1.5 of 7 predicates;
  its roots are greedy/order-dependent/unverified; it *discards* the per-run
  `fired` atoms that are the measured leaves. Seeding from it would smuggle
  `given`-style facts. → Fixed: it is a reference *algorithm to port*, not a seed;
  `fired` re-emitted per run.
- **C3 (SHOULD-FIX).** `check_span_law`/`law_violations` is a hardcoded `==64`
  single-run one-liner, not the verify-before-trust pattern; the real cross-run
  machinery is `aggregate`/`classify`. The verifier layer is essentially net-new.
  → Fixed: layer 3 reclassified net-new, seeded by `aggregate`/`classify`.
- **C4 (MINOR).** Mode-threading: the two sites are real and easy to consolidate,
  but mode is **per-tile** (only cores do mode-1), and the decoder already has
  `mode=None` auto-detect. → Fixed: per-tile mode in the directive; read side uses
  auto-detect.
- **C5 (SHOULD-FIX).** The 5-layer stack has a back-edge: the actuator
  (`HwRunner`) imports `RunnerSession` from `trace-sweep.py`, so trace-sweep can't
  be retired until `RunnerSession`/`ParseSession` are extracted. → Fixed: extraction
  is a named prerequisite of the parity gate.
- **C6 (MINOR).** "Memmod produced no packets" is confounded with the core mode-0
  forcing and routing/enable; not yet a clean reachability fact. → Fixed: the
  hypothesis must control for routing/enable.

## Decision recorded

On the one architectural fork (B1/B2 — make `same_source` measurable, or admit it
structurally, or drop the structural branch and Z3 for union-find), the decision
was **Option A**: keep Z3, add `same_source` with an audited `structural` support
type. Rationale: Option A restores the non-vacuous UNSAT branch *and* the
group-disjunction justification keeps Z3 doing work union-find cannot. The
keystone evolves from "no axioms" to "no *unaudited* axioms" — the aie-rt identity
input becomes a visible, citable ledger rather than a hidden assumption.
