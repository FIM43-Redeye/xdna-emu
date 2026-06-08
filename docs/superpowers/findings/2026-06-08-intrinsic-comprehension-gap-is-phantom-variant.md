# The AIE2 "intrinsics" comprehension gap is a phantom variant

**Date:** 2026-06-08. **Status:** examined; closure deferred (see Disposition).

## TL;DR

The single AIE2 comprehension gap reported by the coverage generator —
`target-specific intrinsics (SemanticOp::Intrinsic): 1 node(s) UNSPECIFIED` — is
**not** a class of unmodeled intrinsics. It is a vestigial enum variant that the
live decode pipeline **never constructs**, sitting at the coverage bootstrap's
pessimistic `Unspecified` default. Intrinsic *behavior* is fully modeled via a
different path. There is no execution work hiding here.

## What the gap node is

- `SemanticOp::Intrinsic(u32)` (`crates/xdna-archspec/src/aie2/isa/types.rs:375`)
  is a catch-all variant: "an intrinsic we could not classify to a concrete
  SemanticOp."
- The coverage model hand-inserts a canonical `Intrinsic(0)` node
  (`coverage/mod.rs:126`, `coverage/artifacts.rs:117,310`); it derives to
  `Category::NeedsTriage` (`coverage/derive.rs:164`) → default verdict
  `Unspecified / Unverified` (`derive.rs:192`) → `is_comprehension_gap()` true
  (`coverage/verdict.rs:85`). That is the entire "1 node."

## Why it is a phantom

Two `from_intrinsic` functions exist; only one is live:

- `SemanticOp::from_intrinsic()` (`types.rs:459`) — runtime name-classifier.
  **Dead code**: never called anywhere outside its own unit tests (verified by
  whole-repo grep). The would-be producer of `Intrinsic(idx)` is unwired.
- `semantic_from_intrinsic()` (`build_helpers/extract.rs:461`, called at
  `:623` and `:1142`) — the **build-time** extractor that maps intrinsic-backed
  instruction patterns to concrete SemanticOps. This is the live path.

Consequences, all verified:

1. **No live construction site** of `SemanticOp::Intrinsic(_)` exists in any
   decoder / resolver / build path (grep-confirmed empty).
2. The ISA build asserts **100% semantic coverage** — every decoded instruction
   has `semantic.is_some()` or the build breaks (`aie2/isa/mod.rs:125-131`). So
   every intrinsic-backed instruction already carries a concrete SemanticOp
   (`Mac`, `MatMul`, `Srs`, `Ups`, `Convert`, `Shuffle`, …), each of which is
   `Wired` to a real handler (`interpreter/coverage/surface_probe.rs`).
3. The catch-all is **fail-loud, not silent-wrong**: the surface probe classes
   `Intrinsic(_)` as `Absent` (`surface_probe.rs:162`); were one ever to reach
   execution it hits a hard `ExecuteResult::Error` (`cycle_accurate.rs:178`),
   never a wrong value.
4. The full bridge corpus (~75 kernels incl. matmul) passes, so nothing emits
   `Intrinsic` in practice.

Net: intrinsic *semantics* are modeled (via the build-time extractor + the
structural resolver → concrete SemanticOps); the correctness of the **vector**
ones is tracked separately as the perishable-queue item "vector compute,
aietools-modeled, UNVERIFIED" — that is the real open work, not this node.

## Disposition (sequenced, agreed 2026-06-08)

"Modeling the class" has no execution target. The chosen path:

1. **(3) Correctness audit, folded into vector-compute verification.** The
   substantive question worth asking is not "is the `Intrinsic` variant
   handled" (it is never produced) but "is every build-time
   intrinsic→SemanticOp mapping behaviourally *correct*, not merely present?"
   That overlaps exactly the perishable-queue vector-compute verification, so it
   is done there, not as separate intrinsic work.
2. **(1) Accept{rationale} — after (3).** Close the comprehension gap via the
   framework's sanctioned mechanism, with a rationale written against
   *verified* mappings: intrinsics resolve to concrete SemanticOps at build time
   (100% coverage asserted); the `Intrinsic(_)` catch-all is never constructed
   and is a fail-loud sink; no behavior is unmodeled.
3. **(2) Dead-code cleanup — after (1).** Retire the unwired
   `SemanticOp::from_intrinsic()` (and consider whether the `Intrinsic` variant
   itself should go, given the 100%-coverage build assertion is the real
   forcing function, not the catch-all).

Deferring (1)/(2) until after (3) is deliberate: the Accept rationale is
stronger written against confirmed-correct mappings than assumed ones.
