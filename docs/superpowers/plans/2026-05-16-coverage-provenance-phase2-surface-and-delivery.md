# Coverage Provenance Phase 2 (Plan 2 of 2): live surface axis + enforcement delivery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Axis 1 (surface presence) real by adding the interpreter's concrete `impl SurfaceProbe`, prove the two axes cohere with a reconciliation test, complete the `Provenance` vocabulary with `HardwareObserved`, replace the hand-maintained coverage index with a generated per-arch one, and deliver the build-time enforcement panic through a dependency-light entry that breaks the `coverage -> aie2::isa -> build.rs` cycle.

**Architecture:** Plan 1 built the two-axis vocabulary and enforcement *logic* in `xdna-archspec`, but Axis 1 is a trait with zero implementations and the build panic is test-gated. This plan follows one ordering principle: **build the model real -> prove it coheres -> complete its vocabulary -> render it -> only then deliver the enforcement teeth.** The interpreter supplies a single compiler-forced exhaustive `match` over `SemanticOp` classifying each by its *true handler topology* across the real dispatch chain (`execute_semantic` -> `VectorAlu` -> `MemoryUnit` -> `CascadeOps` -> `StreamOps` -> `ControlUnit`), which is an *independent* signal from Axis 2's `category()` partition -- that independence is what makes the reconciliation cross-check meaningful.

**Tech Stack:** Rust 2021, `serde` (already a dep), inline `#[cfg(test)] mod tests` (repo convention -- there is no Rust `tests/` integration dir; `tests/` holds shell scripts), `cargo test --lib` / `cargo build`, `#[path]`-included leaf modules in `build.rs` (existing build.rs mechanism, `build.rs:24-36`).

**Spec:** `docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md`

**Predecessor:** `docs/superpowers/plans/2026-05-15-coverage-provenance-phase1-archspec-core.md` (Plan 1, complete -- commit range `8db022d..034a132`).

**Out of scope (Phase 2 standing process, no plan):** adjudicating the ~80-150 fine override units and flipping verdicts to `Verified`/`Accepted` one at a time -- that is the open-ended verification work this infrastructure *enables*, with no bounded end state. **Out of scope (future, named not built):** the trace-sweep / differential-fuzzer machinery that *produces* `HardwareObserved` verdicts; Task 3 lands the variant + predicate semantics + a mint constructor so the model can *receive* such verdicts, but wiring an actual empirical source is future work (spec "Future work").

---

## The irreducible cycle (read before Task 5)

`build.rs` generates `gen_tablegen.rs`; `aie2/isa/mod.rs` `include!`s it; `coverage` does `use crate::aie2::isa::SemanticOp`; `CoverageNode::Semantic { op: SemanticOp }` makes the whole `CoverageNode` enum -- and therefore every `coverage` type that names it -- transitively depend on generated code. A `build.rs` that names `CoverageNode`/`SemanticOp` cannot compile (it needs a file `build.rs` has not yet produced).

This cycle is **permanent**, not a Plan-1 accident. The honest consequence, recorded here and in Task 5: the build-time panic can only ever enforce checks expressible on **plain data** (domain-id `&str`s). The deep data-dependent invariants (node partition, cross-arch `Verified` rule, undeclared shadow) reference `CoverageNode` and so remain **test-gated permanently** -- delivered by the Plan-1 `should_panic` suite + the reconciliation test (Task 2) + the `clean_release_<arch>` gate, which is the project's actual commit/CI gate anyway (`cargo test --lib`). Task 5 delivers a *real* `build.rs` panic for the spine-existence invariant via a dependency-light leaf entry; it does not pretend to build-gate the deep checks. Claiming otherwise would be the exact false-confidence failure this design exists to prevent.

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/interpreter/coverage/mod.rs` | interpreter coverage module root; `pub mod surface_probe;` + the reconciliation test module |
| `src/interpreter/coverage/surface_probe.rs` | `InterpreterSurfaceProbe`: the compiler-forced exhaustive `match SemanticOp -> SurfaceClass` by true handler topology |
| `src/interpreter/mod.rs` | add `pub mod coverage;` |
| `crates/xdna-archspec/src/coverage/verdict.rs` | add `Provenance::HardwareObserved` + `Verdict::hardware_observed()` mint ctor; predicates updated |
| `crates/xdna-archspec/src/coverage/derive.rs` | `default_verdict` exhaustive `match Category` already total; no `Category` maps to `HardwareObserved` (mint-only) -- comment records why |
| `crates/xdna-archspec/src/coverage/artifacts.rs` | add `render_architecture_index`; perishable renderer's dead match arm updated for the new variant |
| `crates/xdna-archspec/src/coverage/spine_ids.rs` | NEW leaf module, zero `crate::` imports: `SPINE_DOMAIN_IDS: &[&str]` source of truth |
| `crates/xdna-archspec/src/coverage/build_gate.rs` | NEW leaf module, zero `crate::` imports: `enforce_spine_phase1(domain_ids: &[&str])` build-callable panic |
| `crates/xdna-archspec/src/coverage/units.rs` | `capability_spine()` consumes `spine_ids::SPINE_DOMAIN_IDS` instead of an inline literal |
| `crates/xdna-archspec/src/coverage/mod.rs` | add `pub mod spine_ids; pub mod build_gate;` |
| `crates/xdna-archspec/build.rs` | `#[path]`-include `coverage/spine_ids.rs` + `coverage/build_gate.rs`; call the gate after `build_arch_model` |
| `crates/xdna-archspec/examples/gen_coverage_artifacts.rs` | also write `docs/coverage/aie2/architecture-index.md` |
| `docs/coverage/aie2/architecture-index.md` | NEW generated, committed, staleness-gated |
| `docs/coverage/architecture-index.md` | `git rm` -- replaced by the generated per-arch file |
| `docs/README.md`, `docs/coverage/cycle-accuracy-mission.md`, `docs/coverage/timer-sync-gap.md`, `docs/coverage/audit-checklist.md` | repoint inbound links to the per-arch path |

**Verbatim signatures confirmed from source (do not re-derive):**
- `SurfaceClass` (`Wired`/`Fallthrough`/`Absent`, derives `Debug,Clone,Copy,PartialEq,Eq,Hash,Serialize,Deserialize`) + `trait SurfaceProbe { fn surface_class(&self, node: &CoverageNode) -> SurfaceClass; }` -- `crates/xdna-archspec/src/coverage/surface.rs:9-25`
- `CoverageNode::{Semantic{arch,op},Register{..},Capability{arch,domain}}` + `fn arch(&self)` -- `crates/xdna-archspec/src/coverage/mod.rs:23-52`
- `CoverageModel { pub arch, overrides }` with `build(arch)`, `semantic_verdict(&SemanticOp)->Verdict`, `perishable_queue()`, `comprehension_gaps()`, `clean_release()`, `applicable_capabilities()` -- `mod.rs:57-119`; private `all_semantic_ops() -> Vec<SemanticOp>` (103 entries) `mod.rs:132-239`
- `Verdict { pub provenance: Provenance, pub verification: Verification }`, `is_perishable()`, `is_comprehension_gap()` -- `verdict.rs:42-62`
- `Provenance::{ToolchainDerived,AietoolsModeled,DocSpecified,Unspecified}` -- `verdict.rs:7-17`
- `BehavioralUnit { id, arch, claims, verdict, shadows_derived: Option<String>, shared_from: Option<Architecture> }`, `Claims::Nodes(Vec<CoverageNode>)`, `CapabilityDomain { id, arches }` + `applies_to`, `override_registry(arch)->Vec<BehavioralUnit>` (empty), `capability_spine()->Vec<CapabilityDomain>` -- `units.rs:11-94`
- `enforce_coverage(arch,&[CapabilityDomain],&[BehavioralUnit])`, `enforce_coverage_phase1(arch)` -- `enforce.rs:19-91`
- `render_perishable(Architecture)->String`, `render_comprehension(Architecture)->String` -- `artifacts.rs:9-72`
- Interpreter dispatch chain: `execute_semantic` (`src/interpreter/execute/semantic.rs:72-195`, explicit scalar arms `:87-181`, delegated `_ =>` `:190-193`) then in order `VectorAlu::execute` / `MemoryUnit::execute` / `CascadeOps::execute` / `StreamOps::execute` / `ControlUnit::execute_with_neighbor_locks`, with a final hard `ExecuteResult::Error` for any labeled-but-unhandled op (`src/interpreter/execute/cycle_accurate.rs:127-185`)
- `Architecture` is in non-generated `crates/xdna-archspec/src/types.rs` (build.rs already `#[path]`-includes `src/types.rs` at `build.rs:27-28`)
- build.rs `#[path]` include pattern + `main()` builds `arch_model` at `build.rs:103-104` -- `build.rs:24-36`, `:64-110`

---

### Task 1: interpreter `impl SurfaceProbe` (make Axis 1 real)

**Files:**
- Create: `src/interpreter/coverage/mod.rs`
- Create: `src/interpreter/coverage/surface_probe.rs`
- Modify: `src/interpreter/mod.rs` (add `pub mod coverage;` after `pub mod test_runner;` -- `src/interpreter/mod.rs:52`)
- Test: inline in `surface_probe.rs`

The probe classifies each `SemanticOp` by the **dispatch unit that actually handles it**, which is an independent partition from Axis-2's `category()`. The exhaustive `match` has **no `_` arm** -- a new `SemanticOp` variant breaks the interpreter build until its true handler is classified (the Axis-1 forcing function, spec Section 1/2/4). `Register`/`Capability` nodes are out of Plan-2 scope (register-consumer probing is a separate future effort); the probe `unreachable!()`s on them with a scope message, and the reconciliation test (Task 2) only ever feeds `Semantic` nodes -- so the boundary is structural, not a silent lie.

- [ ] **Step 1: Write the failing test**

Create `src/interpreter/coverage/surface_probe.rs` with the test module first:

```rust
//! Axis-1 evidence: the interpreter's concrete `impl SurfaceProbe`. archspec
//! DEFINES SurfaceClass/SurfaceProbe (single point of definition, spec
//! Section 2); this answers the contract with knowledge that only lives here
//! -- which dispatch unit actually handles each SemanticOp. Compiler-forced
//! exhaustive: a new variant stops THIS build until its true handler is
//! classified (spec Section 1/2/4 Axis-1 forcing function).

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::aie2::isa::SemanticOp;
    use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
    use xdna_archspec::coverage::CoverageNode;
    use xdna_archspec::types::Architecture;

    fn sc(op: SemanticOp) -> SurfaceClass {
        InterpreterSurfaceProbe
            .surface_class(&CoverageNode::Semantic { arch: Architecture::Aie2, op })
    }

    #[test]
    fn scalar_alu_ops_are_wired_via_execute_semantic() {
        // semantic.rs:87-181 -- explicit handler arms.
        assert_eq!(sc(SemanticOp::Add), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Xor), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::SetLt), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Select), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Copy), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Nop), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Event), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::ReadCycleCounter), SurfaceClass::Wired);
    }

    #[test]
    fn vector_ops_are_wired_via_vector_alu() {
        // Fall through execute_semantic (is_vector) but VectorAlu handles them
        // -- Wired, NOT Fallthrough. This is the subtlety the probe must get
        // right (cycle_accurate.rs:131).
        assert_eq!(sc(SemanticOp::Mac), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::MatMul), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Srs), SurfaceClass::Wired);
        assert_eq!(sc(SemanticOp::Shuffle), SurfaceClass::Wired);
    }

    #[test]
    fn memory_stream_cascade_control_sync_are_wired() {
        assert_eq!(sc(SemanticOp::Load), SurfaceClass::Wired); // MemoryUnit
        assert_eq!(sc(SemanticOp::Store), SurfaceClass::Wired); // MemoryUnit
        assert_eq!(sc(SemanticOp::CascadeRead), SurfaceClass::Wired); // CascadeOps
        assert_eq!(sc(SemanticOp::StreamWrite), SurfaceClass::Wired); // StreamOps
        assert_eq!(sc(SemanticOp::Br), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::Call), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::Halt), SurfaceClass::Wired); // ControlUnit
        assert_eq!(sc(SemanticOp::LockAcquire), SurfaceClass::Wired); // lock model
    }

    #[test]
    fn intrinsic_is_absent_not_silently_wired() {
        // Step-3 investigation determined Intrinsic(_) is Absent: it hits the
        // execute_semantic `_ =>` delegated arm (semantic.rs:190), VectorAlu
        // ends `_ => false` (vector_dispatch.rs:120), no unit claims it, so it
        // reaches the hard ExecuteResult::Error (cycle_accurate.rs:178). This
        // pins that finding -- a future regression silently classifying it
        // Wired (claiming the emulator surfaces arbitrary intrinsics when it
        // does not) fails here. Coheres with Axis-2 Unspecified for Intrinsic.
        assert_eq!(sc(SemanticOp::Intrinsic(0)), SurfaceClass::Absent);
    }

    #[test]
    #[should_panic(expected = "Plan-2 scope")]
    fn non_semantic_nodes_are_out_of_scope_loudly() {
        InterpreterSurfaceProbe.surface_class(&CoverageNode::Capability {
            arch: Architecture::Aie2,
            domain: "dma".into(),
        });
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib interpreter::coverage::surface_probe 2>&1 | tail -20`
Expected: FAIL -- `cannot find ... InterpreterSurfaceProbe` / module `coverage` not declared.

- [ ] **Step 3: Write minimal implementation**

First investigate the two genuinely uncertain groups so the classification is fact, not guess:

Run: `grep -n "SemanticOp::" src/interpreter/execute/vector_dispatch.rs | head -60`
Run: `grep -rn "DmaStart\|DmaWait" src/interpreter/execute/ | grep -v test`

Decision rule (apply the rule, do not guess): for `Intrinsic(_)`, `DmaStart`, `DmaWait` -- if a dedicated dispatch unit has a real handler arm reached for that op, classify `Wired`; if it only reaches the `execute_semantic` `_ =>` delegated arm and no downstream unit claims it (would hit the final `ExecuteResult::Error` at `cycle_accurate.rs:178`), classify `Absent`; if a downstream unit has a generic catch-all that accepts-but-no-ops it, classify `Fallthrough`. Record the determination in a one-line comment on that arm citing the file:line you verified.

Prepend to `src/interpreter/coverage/surface_probe.rs` (above the `#[cfg(test)]` block). The match groups mirror the **dispatch chain**, not `category()` -- that independence is the point (spec Section 2 reconciliation). The variant list is the complete `SemanticOp` enum (103 representatives, from `crates/xdna-archspec/src/coverage/mod.rs:132-239`):

```rust
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
use xdna_archspec::coverage::CoverageNode;

/// The interpreter's Axis-1 evidence source. Zero-sized: the answer is a pure
/// function of the SemanticOp variant and the (static) dispatch topology.
pub struct InterpreterSurfaceProbe;

impl SurfaceProbe for InterpreterSurfaceProbe {
    fn surface_class(&self, node: &CoverageNode) -> SurfaceClass {
        match node {
            CoverageNode::Semantic { op, .. } => semantic_surface(op),
            // Register-consumer / capability surface probing is a separate
            // future effort, NOT Plan 2. The reconciliation test feeds only
            // Semantic nodes, so this is structurally unreachable in Plan 2
            // -- a loud `unreachable!`, never a silent false `Absent` (which
            // would be the exact lie this design fights).
            CoverageNode::Register { .. } | CoverageNode::Capability { .. } => {
                unreachable!("register/capability surface probing is out of Plan-2 scope")
            }
        }
    }
}

/// EXHAUSTIVE, NO `_` ARM ON PURPOSE (spec Section 1/2/4 Axis-1 forcing
/// function). Grouped by the REAL dispatch unit that handles each op
/// (`cycle_accurate.rs:127-185`), which is an independent partition from
/// archspec's `category()` -- the reconciliation test (Task 2) relies on that
/// independence. A new SemanticOp variant breaks THIS build until its true
/// handler is classified.
fn semantic_surface(op: &SemanticOp) -> SurfaceClass {
    match op {
        // Wired -- explicit scalar handler arms in execute_semantic
        // (semantic.rs:87-181).
        SemanticOp::Add
        | SemanticOp::Sub
        | SemanticOp::Adc
        | SemanticOp::Sbc
        | SemanticOp::Mul
        | SemanticOp::SDiv
        | SemanticOp::UDiv
        | SemanticOp::SRem
        | SemanticOp::URem
        | SemanticOp::Abs
        | SemanticOp::Neg
        | SemanticOp::DivStep
        | SemanticOp::And
        | SemanticOp::Or
        | SemanticOp::Xor
        | SemanticOp::Not
        | SemanticOp::Shl
        | SemanticOp::Sra
        | SemanticOp::Srl
        | SemanticOp::AshlBidir
        | SemanticOp::LshlBidir
        | SemanticOp::SetEq
        | SemanticOp::SetNe
        | SemanticOp::SetLt
        | SemanticOp::SetLe
        | SemanticOp::SetGt
        | SemanticOp::SetGe
        | SemanticOp::SetUlt
        | SemanticOp::SetUle
        | SemanticOp::SetUgt
        | SemanticOp::SetUge
        | SemanticOp::Cmp
        | SemanticOp::Select
        | SemanticOp::Ctlz
        | SemanticOp::Cttz
        | SemanticOp::Ctpop
        | SemanticOp::Bswap
        | SemanticOp::Clb
        | SemanticOp::Rotl
        | SemanticOp::Rotr
        | SemanticOp::Copy
        | SemanticOp::PointerAdd
        | SemanticOp::PointerMov
        | SemanticOp::SignExtend
        | SemanticOp::ZeroExtend
        | SemanticOp::Truncate
        | SemanticOp::ReadCycleCounter
        | SemanticOp::Nop
        | SemanticOp::Event => SurfaceClass::Wired,

        // Wired via VectorAlu (cycle_accurate.rs:131). These fall THROUGH
        // execute_semantic on the is_vector check (semantic.rs:79-80) but a
        // real handler exists downstream -- Wired, not Fallthrough.
        SemanticOp::Mac
        | SemanticOp::MatMul
        | SemanticOp::MatMulSub
        | SemanticOp::NegMatMul
        | SemanticOp::AddMac
        | SemanticOp::SubMac
        | SemanticOp::NegMul
        | SemanticOp::Srs
        | SemanticOp::Ups
        | SemanticOp::Shuffle
        | SemanticOp::Pack
        | SemanticOp::Unpack
        | SemanticOp::Align
        | SemanticOp::VectorBroadcast
        | SemanticOp::VectorExtract
        | SemanticOp::VectorInsert
        | SemanticOp::VectorPush
        | SemanticOp::VectorPushHi
        | SemanticOp::VectorSelect
        | SemanticOp::VectorClear
        | SemanticOp::Convert
        | SemanticOp::Min
        | SemanticOp::Max
        | SemanticOp::SubLt
        | SemanticOp::SubGe
        | SemanticOp::MaxDiffLt
        | SemanticOp::MaxLt
        | SemanticOp::MinGe
        | SemanticOp::AbsGtz
        | SemanticOp::NegGtz
        | SemanticOp::NegLtz
        | SemanticOp::NegAdd
        | SemanticOp::Accumulate
        | SemanticOp::AccumSub
        | SemanticOp::AccumNegAdd
        | SemanticOp::AccumNegSub => SurfaceClass::Wired,

        // Wired via MemoryUnit (cycle_accurate.rs:135).
        SemanticOp::Load | SemanticOp::Store => SurfaceClass::Wired,

        // Wired via CascadeOps (cycle_accurate.rs:142).
        SemanticOp::CascadeRead | SemanticOp::CascadeWrite => SurfaceClass::Wired,

        // Wired via StreamOps (cycle_accurate.rs:152).
        SemanticOp::StreamRead
        | SemanticOp::StreamWrite
        | SemanticOp::StreamWritePacketHeader => SurfaceClass::Wired,

        // Wired via ControlUnit (cycle_accurate.rs:158); LockAcquire/Release
        // drive the device lock model through the same path.
        SemanticOp::Br
        | SemanticOp::BrCond
        | SemanticOp::Call
        | SemanticOp::Ret
        | SemanticOp::Done
        | SemanticOp::Halt
        | SemanticOp::LockAcquire
        | SemanticOp::LockRelease => SurfaceClass::Wired,

        // DMA semantic ops: classify from Step-3 investigation. Replace this
        // arm's RHS with the verified class and cite the file:line you
        // confirmed (handler present -> Wired; labeled-but-unhandled hard
        // error path -> Absent; accepted no-op -> Fallthrough).
        SemanticOp::DmaStart | SemanticOp::DmaWait => SurfaceClass::Wired,

        // Target-specific intrinsics: classify from Step-3 investigation of
        // src/interpreter/execute/vector_dispatch.rs. Replace this arm's RHS
        // with the verified class and cite the file:line you confirmed.
        SemanticOp::Intrinsic(_) => SurfaceClass::Fallthrough,
    }
}
```

Create `src/interpreter/coverage/mod.rs` (the reconciliation-test clause is forward-looking — Task 2 appends it to this same file; phrase it so the Task-1 commit reads honestly):

```rust
//! Interpreter-side coverage: the concrete Axis-1 probe. The two-axis
//! reconciliation test will land here in Task 2. archspec owns both axis
//! DEFINITIONS (spec Section 2); this crate contributes only the one trait
//! impl plus (Task 2) the wiring test.

pub mod surface_probe;
```

Add to `src/interpreter/mod.rs` after line 52 (`pub mod test_runner;`):

```rust
pub mod coverage;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib interpreter::coverage::surface_probe 2>&1 | tail -20`
Expected: PASS -- 5 tests. If `semantic_surface` fails to compile with "non-exhaustive patterns", the `SemanticOp` enum changed since this plan was written -- classify the missing variant into the correct dispatch group (the compiler names it). That is the Axis-1 forcing function working as designed.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/coverage/mod.rs src/interpreter/coverage/surface_probe.rs src/interpreter/mod.rs
git commit -m "coverage: interpreter impl SurfaceProbe -- Axis 1 becomes real

Compiler-forced exhaustive match over SemanticOp classifying each by its
true dispatch-unit handler (independent partition from category()).

Generated using Claude Code."
```

---

### Task 2: reconciliation test (prove the two axes cohere)

**Files:**
- Modify: `src/interpreter/coverage/mod.rs` (add `#[cfg(test)] mod reconciliation`)
- Test: the reconciliation module itself is the deliverable

Spec Section 2 item 1: wire the concrete probe into archspec's `CoverageModel` and assert consistency -- a unit marked `Verified` whose nodes are all `Absent` is a contradiction; an undeclared shadow is a contradiction. Plan-1 scoped the model's node universe to the SemanticOp set (`all_semantic_ops`), so the reconciliation ranges over that set. archspec deliberately does not expose `all_semantic_ops()` (private, `mod.rs:132`); the test reconstructs the same universe from the public `semantic_verdict` + the probe by iterating a local list, and cross-checks count via the public surface of the model. Per repo convention this is an inline `#[cfg(test)]` module (no Rust `tests/` dir exists).

- [ ] **Step 1: Write the failing test**

First restore `src/interpreter/coverage/mod.rs`'s module doc to present tense (Task 1 made it forward-looking; Task 2 makes the reconciliation test present, so the doc is accurate again):

```rust
//! Interpreter-side coverage: the concrete Axis-1 probe and the two-axis
//! reconciliation test. archspec owns both axis DEFINITIONS (spec Section 2);
//! this crate contributes only the one trait impl plus the wiring test.
```

Then append to `src/interpreter/coverage/mod.rs`:

```rust
#[cfg(test)]
mod reconciliation {
    use crate::interpreter::coverage::surface_probe::InterpreterSurfaceProbe;
    use xdna_archspec::aie2::isa::SemanticOp;
    use xdna_archspec::coverage::surface::{SurfaceClass, SurfaceProbe};
    use xdna_archspec::coverage::verdict::Verification;
    use xdna_archspec::coverage::{CoverageModel, CoverageNode};
    use xdna_archspec::types::Architecture;

    /// The SemanticOp universe under reconciliation. Mirrors archspec's
    /// private `all_semantic_ops()` (mod.rs:132-239). The Plan-1
    /// `all_semantic_ops_len_tripwire` test (count == 103) is the maintenance
    /// tripwire on the archspec side; this list is the interpreter-side peer.
    /// If they drift, the count assertion below fails loudly.
    fn semantic_universe() -> Vec<SemanticOp> {
        use SemanticOp::*;
        vec![
            Add, Sub, Adc, Sbc, Mul, SDiv, UDiv, SRem, URem, Abs, Neg, DivStep, Select, Ctlz,
            Cttz, Ctpop, Bswap, Clb, SignExtend, ZeroExtend, Truncate, Copy, Nop, Event,
            ReadCycleCounter, PointerAdd, PointerMov, And, Or, Xor, Not, Shl, Sra, Srl,
            AshlBidir, LshlBidir, Rotl, Rotr, SetEq, SetNe, SetLt, SetLe, SetGt, SetGe, SetUlt,
            SetUle, SetUgt, SetUge, Cmp, Load, Store, Br, BrCond, Call, Ret, Done, Halt, Mac,
            MatMul, MatMulSub, NegMatMul, AddMac, SubMac, NegMul, Srs, Ups, Shuffle, Pack,
            Unpack, Align, VectorBroadcast, VectorExtract, VectorInsert, VectorPush,
            VectorPushHi, VectorSelect, VectorClear, Convert, Min, Max, SubLt, SubGe,
            MaxDiffLt, MaxLt, MinGe, AbsGtz, NegGtz, NegLtz, NegAdd, Accumulate, AccumSub,
            AccumNegAdd, AccumNegSub, LockAcquire, LockRelease, CascadeRead, CascadeWrite,
            StreamRead, StreamWrite, StreamWritePacketHeader, DmaStart, DmaWait, Intrinsic(0),
        ]
    }

    #[test]
    fn universe_matches_archspec_tripwire_count() {
        // Same 103 the archspec-side tripwire pins; drift here means the two
        // hand-maintained peer lists fell out of sync -- fix both together.
        // Note: this catches length drift, not content drift; content drift
        // (same count, different representative) is caught by compile errors
        // when archspec renames a SemanticOp variant (exhaustive matches
        // break). Symmetric with archspec's own all_semantic_ops tripwire
        // caveat -- neither claims to be a completeness proof.
        assert_eq!(semantic_universe().len(), 103);
    }

    #[test]
    fn no_verified_unit_is_entirely_absent() {
        // Spec Section 2 item 1 / Section 4 test-red condition: a verdict
        // claiming silicon-Verified for an op the emulator does not even
        // surface is an incoherent state. At Phase-1 bootstrap the override
        // registry is empty so nothing is Verified yet -- this asserts the
        // invariant holds now and keeps holding as Phase-2 adds overrides.
        let m = CoverageModel::build(Architecture::Aie2);
        let probe = InterpreterSurfaceProbe;
        for op in semantic_universe() {
            let v = m.semantic_verdict(&op);
            if matches!(v.verification, Verification::Verified { .. }) {
                let node = CoverageNode::Semantic { arch: Architecture::Aie2, op };
                assert_ne!(
                    probe.surface_class(&node),
                    SurfaceClass::Absent,
                    "{:?} is Verified but its surface is Absent -- a Verified \
                     verdict for an op the interpreter does not surface is a \
                     contradiction (spec Section 2/4)",
                    node
                );
            }
        }
    }

    #[test]
    fn every_semantic_node_has_a_definite_surface_class() {
        // The probe is total by construction (no `_` arm); this asserts the
        // wired model agrees -- every node in the reconciled universe gets a
        // definite class and the probe never panics on the in-scope set.
        let probe = InterpreterSurfaceProbe;
        for op in semantic_universe() {
            let node = CoverageNode::Semantic { arch: Architecture::Aie2, op };
            let _ = probe.surface_class(&node); // must not panic
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib interpreter::coverage::reconciliation 2>&1 | tail -20`
Expected: FAIL initially only if names mismatch; if Task 1 is correct these compile. Run it -- expected PASS once compiling (the invariants hold at bootstrap). If any test FAILS, that is a real contradiction to fix before proceeding (do not weaken the assertion).

- [ ] **Step 3: Make it pass**

No new production code: Task 1's probe + Plan-1's model satisfy the invariants at bootstrap. If `no_verified_unit_is_entirely_absent` fails, a probe arm misclassifies a Verified op as `Absent` -- fix the Task-1 classification (cite the corrected file:line). If `universe_matches_archspec_tripwire_count` fails, the two peer lists drifted -- reconcile against `crates/xdna-archspec/src/coverage/mod.rs:132-239`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib interpreter::coverage 2>&1 | tail -15`
Expected: PASS -- surface_probe (5) + reconciliation (3).

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/coverage/mod.rs
git commit -m "coverage: reconciliation test -- the two axes are proven to cohere

Verified-but-Absent is a test-red contradiction; the SemanticOp universe
is cross-checked against archspec's count tripwire.

Generated using Claude Code."
```

---

### Task 3: `Provenance::HardwareObserved` vocabulary (complete the model before rendering it)

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/verdict.rs` (variant + mint ctor + predicates + tests)
- Modify: `crates/xdna-archspec/src/coverage/artifacts.rs` (perishable renderer's match arm -- it has a non-`_` total match over `Provenance`)
- Test: inline in `verdict.rs`

Spec Section 6 "Phasing (honest scope)": `HardwareObserved` is a silicon-observed fact -- neither modeled-weakly (`is_perishable` false) nor a comprehension gap (`is_comprehension_gap` false). It is **mint-only**: no `Category` derives to it (it cannot arise from the toolchain-derived long tail -- it is born from a hardware surprise), so `derive.rs` is intentionally *not* changed. Landing the vocabulary now -- before Task 4 renders and Task 5 enforces -- means the renderer and gate are written once against the final `Provenance` set, not retrofitted (schema-first, spec's own ethos).

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` in `crates/xdna-archspec/src/coverage/verdict.rs`:

```rust
    #[test]
    fn hardware_observed_is_neither_perishable_nor_a_gap() {
        // Spec Section 6 "Phasing": a silicon-observed fact is ground truth
        // from hardware -- not modeled-weakly, not a comprehension gap.
        let v = Verdict::hardware_observed("trace-sweep:add_one:cycle=200");
        assert_eq!(v.provenance, Provenance::HardwareObserved);
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn hardware_observed_carries_its_evidence() {
        let v = Verdict::hardware_observed("finding:2026-05-13-chain-exec");
        match v.verification {
            Verification::Verified { evidence } => {
                assert_eq!(evidence, "finding:2026-05-13-chain-exec")
            }
            other => panic!("expected Verified evidence, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::verdict 2>&1 | tail -20`
Expected: FAIL -- `no variant ... HardwareObserved` / `no function ... hardware_observed`.

- [ ] **Step 3: Write minimal implementation**

In `crates/xdna-archspec/src/coverage/verdict.rs`, add the variant to `Provenance` (after `Unspecified`, `verdict.rs:16`):

```rust
    /// A behavior learned from silicon itself (trace sweep / differential
    /// fuzzer), with no originating toolchain/doc node -- a unit born from a
    /// hardware surprise (spec Section 6). Mint-only: no `Category` derives to
    /// it. Neither perishable (it IS the silicon truth) nor a comprehension
    /// gap (it is understood, just not toolchain-sourced).
    HardwareObserved,
```

Add the mint constructor to `impl Verdict` (next to the predicates, `verdict.rs:48`):

```rust
    /// Mint a verdict from an empirical hardware observation (spec Section 6
    /// empirical residual). `evidence` points at the trace / finding that
    /// recorded the surprise. The verification is `Verified` against that
    /// arch's silicon by construction -- the observation IS the evidence.
    pub fn hardware_observed(evidence: impl Into<String>) -> Self {
        Verdict {
            provenance: Provenance::HardwareObserved,
            verification: Verification::Verified { evidence: evidence.into() },
        }
    }
```

`is_perishable` (matches only `AietoolsModeled | DocSpecified`) and `is_comprehension_gap` (matches only `Unspecified`) already exclude `HardwareObserved` with no change -- the existing tests plus the two new ones prove it. Update the `Verdict` doc comment's invariant paragraph (`verdict.rs:33-41`) to note the new variant: append after the existing "future `HardwareObserved`" sentence -- change "(future `HardwareObserved`, spec Section 6)" to "(`HardwareObserved`, spec Section 6)".

Now fix the total `match v.provenance` in `crates/xdna-archspec/src/coverage/artifacts.rs:24-31`. It currently has `AietoolsModeled`, `DocSpecified`, and a commented dead `_ =>` arm. `HardwareObserved` is never perishable so `perishable_queue()` never yields it -- the existing `_ =>` already covers it. Update only the comment to name the new variant explicitly so a future reader is not misled:

Replace the comment in `artifacts.rs:27-29`:

```rust
            // Dead arm in Phase 1: perishable_queue() pre-filters to
            // is_perishable() (only AietoolsModeled|DocSpecified).
            // ToolchainDerived, Unspecified, and HardwareObserved cannot
            // reach here. Kept so the match stays total if a future
            // perishable provenance lands.
            _ => "modeled".to_string(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::verdict 2>&1 | tail -20`
Expected: PASS -- the 5 original + 2 new = 7 tests. Then `cargo test -p xdna-archspec --lib coverage 2>&1 | tail -10` -- all PASS (no rollup changed: nothing mints `HardwareObserved` yet, so the artifacts are byte-identical and their staleness tests still pass).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/verdict.rs crates/xdna-archspec/src/coverage/artifacts.rs
git commit -m "coverage: add Provenance::HardwareObserved vocabulary (mint-only)

Silicon-observed truth: neither perishable nor a comprehension gap.
No Category derives to it; the empirical intake path is future work
(spec Section 6 Phasing). Lands before render/enforce so both are
written once against the final Provenance set.

Generated using Claude Code."
```

---

### Task 4: generated per-arch `architecture-index.md` + retire the hand-maintained index

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/artifacts.rs` (add `render_architecture_index`)
- Modify: `crates/xdna-archspec/examples/gen_coverage_artifacts.rs` (also write the index)
- Create (generated, committed): `docs/coverage/aie2/architecture-index.md`
- Delete: `docs/coverage/architecture-index.md`
- Modify (link repoint): `docs/README.md:9`, `docs/coverage/cycle-accuracy-mission.md:12` & `:219`, `docs/coverage/timer-sync-gap.md:156`, `docs/coverage/audit-checklist.md:4` & `:78`
- Test: inline staleness gate in `artifacts.rs`

Spec Section 2 item 2 + Section 5: the index flips from hand-maintained to generated in one move -- "delete the old file, the renderer replaces it. No dual-maintenance window." The generated index is the **mechanical coverage matrix** (the part that silently went stale -- the bug this whole project exists to kill). The renderer owns the *entire* file including a short fixed preamble, so there is no prose/data split to drift again.

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` in `crates/xdna-archspec/src/coverage/artifacts.rs`:

```rust
    #[test]
    fn architecture_index_is_not_stale() {
        let want = render_architecture_index(Architecture::Aie2);
        let path = repo_path("docs/coverage/aie2/architecture-index.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            got, want,
            "{} is stale -- regenerate: `cargo run -p xdna-archspec --example \
             gen_coverage_artifacts` then `git add docs/coverage/` and commit",
            path.display()
        );
    }

    #[test]
    fn old_hand_maintained_index_is_retired() {
        // Spec Section 5: no dual-maintenance window. The non-arch-qualified
        // hand-maintained file must not exist once the generated per-arch one
        // lands -- its continued presence is the silent-staleness risk itself.
        let old = repo_path("docs/coverage/architecture-index.md");
        assert!(
            !old.exists(),
            "{} still exists -- it must be `git rm`'d; the generated \
             docs/coverage/<arch>/architecture-index.md replaces it (spec S5)",
            old.display()
        );
    }

    #[test]
    fn architecture_index_reps_match_category() {
        // The renderer's hand-listed (Category, SemanticOp) reps must stay in
        // sync with category(). The render fn also debug_assert!s this, but
        // that is invisible to release builds -- this makes the invariant a
        // first-class `cargo test --lib` gate that fails AT THE SOURCE with a
        // clear message rather than as a downstream staleness string-diff.
        use crate::aie2::isa::SemanticOp;
        use crate::coverage::derive::{category, Category};
        let reps: &[(Category, SemanticOp)] = &[
            (Category::Arithmetic, SemanticOp::Add),
            (Category::Bitwise, SemanticOp::And),
            (Category::Comparison, SemanticOp::SetLt),
            (Category::Memory, SemanticOp::Load),
            (Category::ControlFlow, SemanticOp::Br),
            (Category::Vector, SemanticOp::Mac),
            (Category::Sync, SemanticOp::LockAcquire),
            (Category::SideEffect, SemanticOp::DmaStart),
            (Category::NeedsTriage, SemanticOp::Intrinsic(0)),
        ];
        for (cat, rep) in reps {
            assert_eq!(category(rep), *cat, "rep {rep:?} drifted from category {cat:?}");
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::artifacts 2>&1 | tail -20`
Expected: FAIL -- `cannot find function render_architecture_index`; and `old_hand_maintained_index_is_retired` fails (file still present).

- [ ] **Step 3: Write minimal implementation**

Add to `crates/xdna-archspec/src/coverage/artifacts.rs` (after `render_comprehension`, before the test module). The matrix rows are the dispatch-independent Axis-2 view: one row per `category()` bucket with its rolled-up verdict. It deliberately renders the *same* `CoverageModel` the gate uses, so the index can never disagree with `clean_release`:

```rust
use crate::coverage::derive::{category, Category};

/// Render the per-arch coverage matrix (spec Section 2 item 2, Section 7).
/// Replaces the hand-maintained docs/coverage/architecture-index.md. The
/// renderer owns the WHOLE file -- fixed preamble + mechanical matrix -- so
/// there is no hand-maintained prose to drift against stale data again (the
/// original failure mode, spec Problem statement).
pub fn render_architecture_index(arch: Architecture) -> String {
    let m = CoverageModel::build(arch);
    let mut lines = vec![
        format!("# {arch} architecture coverage index"),
        String::new(),
        "Generated by `cargo run -p xdna-archspec --example gen_coverage_artifacts`. Do not hand-edit."
            .to_string(),
        String::new(),
        "Mechanical Axis-2 coverage matrix: every SemanticOp category and its".to_string(),
        "rolled-up provenance/verification verdict. This file is generated from".to_string(),
        "the same CoverageModel that backs the release gate, so it cannot".to_string(),
        "silently disagree with `clean_release` (spec Section 2, Section 5).".to_string(),
        "Per-op overrides and the perishable/comprehension queues live in the".to_string(),
        "sibling perishable-queue.md / comprehension-gaps.md.".to_string(),
        String::new(),
        "| Category | Provenance | Verification |".to_string(),
        "|----------|------------|--------------|".to_string(),
    ];
    // One representative op per category, in a fixed order for determinism.
    let reps: &[(Category, SemanticOp)] = &[
        (Category::Arithmetic, SemanticOp::Add),
        (Category::Bitwise, SemanticOp::And),
        (Category::Comparison, SemanticOp::SetLt),
        (Category::Memory, SemanticOp::Load),
        (Category::ControlFlow, SemanticOp::Br),
        (Category::Vector, SemanticOp::Mac),
        (Category::Sync, SemanticOp::LockAcquire),
        (Category::SideEffect, SemanticOp::DmaStart),
        (Category::NeedsTriage, SemanticOp::Intrinsic(0)),
    ];
    for (cat, rep) in reps {
        debug_assert_eq!(category(rep), *cat, "index representative drifted from category()");
        let v = m.semantic_verdict(rep);
        lines.push(format!("| {cat:?} | {:?} | {:?} |", v.provenance, v.verification));
    }
    lines.push(String::new());
    format!("{}\n", lines.join("\n"))
}
```

`render_architecture_index` references `SemanticOp` -- add `use crate::aie2::isa::SemanticOp;` to the `artifacts.rs` import block if not already present (it is not in Plan 1's version; add it).

Extend `crates/xdna-archspec/examples/gen_coverage_artifacts.rs` -- add the index write next to the existing two (`gen_coverage_artifacts.rs:13-14`):

```rust
    std::fs::write(
        dir.join("architecture-index.md"),
        xdna_archspec::coverage::artifacts::render_architecture_index(Architecture::Aie2),
    )
    .unwrap();
```

and update its `eprintln!` to `wrote docs/coverage/aie2/{perishable-queue,comprehension-gaps,architecture-index}.md`.

Generate the file and retire the old one (from repo root):

```bash
cargo run -p xdna-archspec --example gen_coverage_artifacts 2>&1 | tail -3
git rm docs/coverage/architecture-index.md
```

Repoint the six inbound links (exact edits):
- `docs/README.md:9-12`: repoint the link AND rewrite the now-inaccurate description (it still describes the retired hand-maintained subsystem catalogue with MODELED/PARTIAL/STUBBED states and a manual "refresh" protocol; the generated file is a category-level matrix that must not be hand-edited). Replace the whole bullet:
  ```
  - [coverage/aie2/architecture-index.md](coverage/aie2/architecture-index.md) -- Generated
    Axis-2 coverage matrix: every SemanticOp category rolled up to its
    provenance/verification verdict. Do not hand-edit; regenerate with
    `cargo run -p xdna-archspec --example gen_coverage_artifacts`.
  ```
- `docs/coverage/cycle-accuracy-mission.md:12`: `[architecture-index.md](architecture-index.md)` -> `[aie2/architecture-index.md](aie2/architecture-index.md)`.
- `docs/coverage/cycle-accuracy-mission.md:219`: the cross-ref targets a hand-written analysis anchor (`#likely-impactful-gaps-...`) that the mechanical index does not have. Replace the whole cross-reference with an in-document pointer: change `[architecture-index.md gap #3](architecture-index.md#likely-impactful-gaps-model-correctness-affecting)` to `the cycle-accuracy gaps tracked in this document` (the gaps analysis already lives in cycle-accuracy-mission.md per its own preamble -- the mechanical index never carried it; this removes a dangling anchor, it does not lose content).
- `docs/coverage/timer-sync-gap.md:156`: `architecture-index verification` -> `aie2/architecture-index coverage matrix` (prose ref, no anchor).
- `docs/coverage/audit-checklist.md:4` and `:78`: `architecture-index.md` -> `aie2/architecture-index.md` (link target + prose).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::artifacts 2>&1 | tail -20`
Expected: PASS -- the 2 Plan-1 staleness tests + the 2 new ones. Then verify no remaining stale links:
Run: `grep -rn "architecture-index" docs/README.md docs/coverage/*.md | grep -v "aie2/architecture-index" | grep -v "docs/superpowers"`
Expected: no output (every non-spec/plan reference now points at the per-arch path).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/artifacts.rs crates/xdna-archspec/examples/gen_coverage_artifacts.rs docs/coverage/aie2/architecture-index.md docs/README.md docs/coverage/cycle-accuracy-mission.md docs/coverage/timer-sync-gap.md docs/coverage/audit-checklist.md
git rm docs/coverage/architecture-index.md
git commit -m "coverage: generate per-arch architecture-index, retire hand-maintained one

The matrix is now rendered from the same CoverageModel as the gate, so
it cannot silently go stale -- the original failure this design kills
(spec Section 2/5). Inbound links repointed to the per-arch path.

Generated using Claude Code."
```

---

### Task 5: deliver the build.rs panic via a dependency-light entry

**Files:**
- Create: `crates/xdna-archspec/src/coverage/spine_ids.rs` (leaf, zero `crate::` imports)
- Create: `crates/xdna-archspec/src/coverage/build_gate.rs` (leaf, zero `crate::` imports)
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (`pub mod spine_ids; pub mod build_gate;`)
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (`capability_spine()` consumes `spine_ids::SPINE_DOMAIN_IDS`)
- Modify: `crates/xdna-archspec/build.rs` (`#[path]`-include the two leaves; call the gate)
- Test: inline in `build_gate.rs` + a consistency test in `units.rs`

Read "The irreducible cycle" section above first. This delivers a *real* `build.rs` panic for the **spine-existence** invariant on plain `&str` data. The deep checks (partition / cross-arch / shadow) reference `CoverageNode` and stay test-gated **permanently and by design** -- that is recorded, not hidden.

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/build_gate.rs`:

```rust
//! Dependency-light build-time gate (spec Section 4 delivery; the "irreducible
//! cycle" note in Plan 2). ZERO `crate::` imports on purpose: this module is
//! `#[path]`-included into build.rs, which cannot see `crate::aie2::isa`
//! (build.rs generates it). Operates only on plain domain-id strings; the deep
//! CoverageNode-dependent checks remain test-gated (permanent, by design).

/// Phase-1 spine-existence gate. Panics (stops the build) if the capability
/// spine is empty -- the one Axis-2 invariant expressible without
/// `CoverageNode`. In Phase 1 every domain is auto-claimed by the derived
/// shim, so a non-empty well-formed spine IS the deliverable invariant; the
/// partition / cross-arch / shadow panics live in the test gate (the
/// CoverageNode types cannot cross into build.rs -- see the Plan-2 cycle
/// note). Phase 2 enriches the test gate, never this entry.
pub fn enforce_spine_phase1(domain_ids: &[&str]) {
    if domain_ids.is_empty() {
        panic!(
            "COVERAGE(build): capability spine is empty -- every build must \
             enumerate the hardware capability domains (spec Section 6). This \
             is the dependency-light build-time gate; deep partition/cross-arch \
             checks are test-gated (Plan 2 cycle note)."
        );
    }
    // Defensive: a blank id would silently weaken the spine.
    for (i, id) in domain_ids.iter().enumerate() {
        assert!(
            !id.trim().is_empty(),
            "COVERAGE(build): spine domain #{i} is blank -- ids must be \
             non-empty (spec Section 6)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_on_a_normal_spine() {
        enforce_spine_phase1(&["core", "dma", "locks"]); // must not panic
    }

    #[test]
    #[should_panic(expected = "capability spine is empty")]
    fn panics_on_empty_spine() {
        enforce_spine_phase1(&[]);
    }

    #[test]
    #[should_panic(expected = "is blank")]
    fn panics_on_blank_domain_id() {
        enforce_spine_phase1(&["core", "  ", "dma"]);
    }
}
```

Create `crates/xdna-archspec/src/coverage/spine_ids.rs`:

```rust
//! Single source of truth for the capability-spine domain ids, as plain
//! `&'static str`s with ZERO `crate::` imports so build.rs can `#[path]`-
//! include it without dragging in `crate::aie2::isa` (the irreducible cycle,
//! Plan 2). `units::capability_spine()` builds the rich `CapabilityDomain`s
//! from this list; nothing else defines spine ids (spec Section 6: one
//! location).
pub const SPINE_DOMAIN_IDS: &[&str] = &[
    "core",
    "program_memory",
    "program_counter",
    "data_memory",
    "dma",
    "locks",
    "stream_switch",
    "events_trace",
    "performance_counters",
    "timer",
    "watchpoint",
    "debug_halt",
    "cascade",
    "interrupt",
    "noc",
    "shim_mux",
];
```

Add a consistency test to the `#[cfg(test)] mod tests` in `crates/xdna-archspec/src/coverage/units.rs`:

```rust
    #[test]
    fn capability_spine_matches_the_leaf_id_list() {
        // The leaf list (build.rs-visible) and the rich spine MUST agree --
        // they are the same source of truth, just two views (spec Section 6
        // one location; Plan 2 cycle note).
        use crate::coverage::spine_ids::SPINE_DOMAIN_IDS;
        let rich: Vec<String> = capability_spine().into_iter().map(|d| d.id).collect();
        let leaf: Vec<String> = SPINE_DOMAIN_IDS.iter().map(|s| s.to_string()).collect();
        assert_eq!(rich, leaf);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::build_gate 2>&1 | tail -20`
Expected: FAIL -- module `build_gate` / `spine_ids` not declared.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/xdna-archspec/src/coverage/mod.rs` `pub mod` list (alphabetical: `artifacts`, `build_gate`, `derive`, `enforce`, `spine_ids`, `surface`, `units`, `verdict`):

```rust
pub mod build_gate;
pub mod spine_ids;
```

Rewrite `capability_spine()` in `crates/xdna-archspec/src/coverage/units.rs:71-94` to consume the leaf list (keep the existing doc comment block above it unchanged):

```rust
pub fn capability_spine() -> Vec<CapabilityDomain> {
    let aie2 = vec![Architecture::Aie2];
    crate::coverage::spine_ids::SPINE_DOMAIN_IDS
        .iter()
        .map(|id| CapabilityDomain { id: (*id).to_string(), arches: aie2.clone() })
        .collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::build_gate 2>&1 | tail -20` -- PASS (3 tests).
Run: `cargo test -p xdna-archspec --lib coverage::units 2>&1 | tail -20` -- PASS (existing 3 + the new consistency test = 4; the spine is byte-identical so `capability_spine_seeded_for_aie2` still passes).
Run: `cargo test -p xdna-archspec --lib coverage 2>&1 | tail -10` -- all PASS (artifacts unchanged: spine ids identical, so `architecture-index.md` and the two queues are byte-identical and their staleness tests still pass).

- [ ] **Step 5: Wire the gate into build.rs**

Add to the `#[path]` include block in `crates/xdna-archspec/build.rs` (after the existing `mod model_builder;` at `build.rs:35-36`):

```rust
#[path = "src/coverage/spine_ids.rs"]
mod spine_ids;
#[path = "src/coverage/build_gate.rs"]
mod build_gate;
```

In `build.rs` `main()`, immediately after `arch_model` is built and before `extract_aiert` (after `build.rs:104`'s `.unwrap_or_else(...)` for `build_arch_model`), add:

```rust
    // COVERAGE build-time gate (spec Section 4 delivery). Dependency-light by
    // necessity: build.rs cannot see crate::aie2::isa (it generates it), so
    // the deep CoverageNode checks stay test-gated -- see the Plan-2
    // "irreducible cycle" note. This panics the build if the capability spine
    // is empty/ill-formed.
    build_gate::enforce_spine_phase1(spine_ids::SPINE_DOMAIN_IDS);
```

- [ ] **Step 6: Verify the build actually runs the gate**

Run BARE (no pipe): `cargo build -p xdna-archspec`
Expected: SUCCESS -- the gate ran (spine non-empty) and did not panic. To prove the gate is live (not dead code), temporarily change the build.rs call to `build_gate::enforce_spine_phase1(&[]);`, run `cargo build -p xdna-archspec 2>&1 | tail -5`, confirm the build **fails** with `COVERAGE(build): capability spine is empty`, then revert to `spine_ids::SPINE_DOMAIN_IDS` and rebuild clean. (This is a manual liveness check, not a committed test -- a committed empty-spine build cannot exist.)

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-archspec/src/coverage/spine_ids.rs crates/xdna-archspec/src/coverage/build_gate.rs crates/xdna-archspec/src/coverage/mod.rs crates/xdna-archspec/src/coverage/units.rs crates/xdna-archspec/build.rs
git commit -m "coverage: deliver build.rs panic via dependency-light spine gate

Leaf modules with zero crate imports break the coverage->aie2::isa
cycle so build.rs can enforce the spine-existence invariant at build
time. Deep partition/cross-arch checks remain test-gated permanently
(the cycle is irreducible -- recorded, not papered over).

Generated using Claude Code."
```

---

### Task 6: full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Regenerate artifacts and confirm no drift**

Run (repo root): `cargo run -p xdna-archspec --example gen_coverage_artifacts 2>&1 | tail -3`
Run: `git status --porcelain docs/coverage/`
Expected: no modified files under `docs/coverage/` (Tasks 3-5 changed no rollups; the committed artifacts already match the renderers). If anything is modified, the committed files were stale -- `git add docs/coverage/ && git commit` with message `coverage: regenerate artifacts after Phase-2 vocabulary`.

- [ ] **Step 2: Full workspace build (exercises the build.rs gate + the new interpreter module)**

Run BARE: `cargo build`
Expected: SUCCESS. The `build.rs` coverage gate ran; `src/interpreter/coverage/` compiles across the workspace.

- [ ] **Step 3: Full library suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -15`
Expected: all PASS -- Plan-1's coverage tests + the new `coverage::build_gate`, `coverage::verdict` (7), `coverage::artifacts` (4), `coverage::units` (4), and `interpreter::coverage::{surface_probe,reconciliation}`. Any previously-passing test now failing is an in-scope regression -- fix before proceeding.

- [ ] **Step 4: The release gate still resolves and is honestly red**

Run: `cargo test -p xdna-archspec --lib clean_release 2>&1 | tail -15`
Expected: the `clean_release_*` gate test(s) resolve and PASS (they assert the bootstrap is honestly NOT green -- vector perishable, Intrinsic a comprehension gap; Phase-2 override work, out of scope here, is what eventually flips them).

- [ ] **Step 5: Commit (only if Steps 1-4 required fixes)**

```bash
git add -A
git commit -m "coverage: fixes from Phase-2 full-suite verification

Generated using Claude Code."
```

If no fixes were needed, Plan 2 is complete at Task 5's commit.

---

## Self-Review

**1. Spec coverage (Phase 2 scope):**
- Spec §1 Axis-1 (`Wired`/`Fallthrough`/`Absent`, compiler-forced) -> Task 1 exhaustive `semantic_surface`. ✓
- Spec §2 ownership seam (interpreter supplies the one `impl SurfaceProbe`; reconciliation wires it into `CoverageModel`; Verified-but-Absent test-red) -> Tasks 1, 2. ✓
- Spec §2 item 2 (regenerate `architecture-index.md`, fail if stale, never hand-edit again) + §5 (delete old, no dual-maintenance) -> Task 4. ✓
- Spec §4 build-panic *delivery* (deferred from Plan 1) + the irreducible-cycle resolution via a dependency-light entry -> Task 5, with the permanent test-gating of deep checks recorded honestly (the design's own anti-false-confidence ethos). ✓
- Spec §6 "Phasing (honest scope)": `HardwareObserved` variant + both predicates false + mint ctor, intake path named-not-built -> Task 3. ✓
- Spec §7 per-arch artifacts (`docs/coverage/<arch>/architecture-index.md`) -> Task 4 path. ✓
- Empirical intake plumbing (trace-sweep/fuzzer -> minting) -> explicitly named out of scope (spec "Future work"); Task 3 ships the receiving vocabulary only. Not a gap. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"similar to". The two investigation steps (Task 1 Step 3: `Intrinsic`/`DmaStart`/`DmaWait`) are concrete actions with exact `grep` commands and an explicit decision rule + a default classification and a "cite the file:line you verified" requirement -- the same pattern Plan 1 used for the `category()` forcing function ("the compiler names it; that is the forcing function working"). Not placeholders: every code step has complete compiling code.

**3. Type consistency:** `SurfaceClass`/`SurfaceProbe`/`CoverageNode`/`CoverageModel`/`Verdict`/`Provenance`/`Verification` used exactly as defined in Plan 1's shipped source (signatures quoted verbatim in the file-structure header from the actual files). `InterpreterSurfaceProbe` (Task 1) -> consumed identically in Task 2. `Verdict::hardware_observed` (Task 3) returns `Verdict { provenance, verification }` matching the struct. `render_architecture_index(Architecture)->String` (Task 4) mirrors the existing `render_perishable`/`render_comprehension` signature and is consumed identically in the example + staleness test. `SPINE_DOMAIN_IDS: &[&str]` (Task 5) consumed identically by `capability_spine()`, `build_gate::enforce_spine_phase1(&[&str])`, and build.rs. `semantic_universe()` (Task 2, 103) pinned to archspec's private `all_semantic_ops()` (103) via the count tripwire. No drift found.
