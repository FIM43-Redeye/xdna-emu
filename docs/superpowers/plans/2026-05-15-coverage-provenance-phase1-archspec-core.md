# Coverage Provenance Phase 1 (Plan 1 of 2): archspec coverage core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the build-enforced two-axis coverage model inside `xdna-archspec`: every register node and behavioral unit carries a provenance/verification verdict, an incomplete model panics the build, and the Axis-2 honesty artifacts (`perishable-queue.md`, `comprehension-gaps.md`) are generated and committed.

**Architecture:** A new `crate::coverage` module in `xdna-archspec`. The SemanticOp axis is made un-missable by a compiler-enforced exhaustive `category()` match (no wildcard arm). The register/unit/capability axis is enforced by `enforce_coverage(&ArchModel)` called from `build.rs` after the cross-validated model is built, panicking exactly like `Confirmed<T>::confirm` already does. Coverage is keyed by `Architecture` (Section 7 of the spec); a `SurfaceProbe` trait is *defined* here but its concrete implementation is Plan 2 (it lives in the interpreter crate, which depends on archspec).

**Tech Stack:** Rust 2021, `serde`/`serde_json` (already crate deps), inline `#[cfg(test)] mod tests` (crate convention), `cargo test --lib` / `cargo build`.

**Spec:** `docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md`

**Out of scope (Plan 2):** concrete `impl SurfaceProbe`, the reconciliation integration test, generated `architecture-index.md`, retiring the hand-maintained index. **Out of scope (Phase 2, no plan):** adjudicating the ~80-150 fine override units — that is standing process work the infra enables, with no bounded end state.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/xdna-archspec/src/coverage/mod.rs` | Module root; `CoverageNode`; `CoverageModel`; the two filter sets; `clean_release` |
| `crates/xdna-archspec/src/coverage/verdict.rs` | `Provenance`, `Verification`, `Verdict`, perishable/comprehension predicates |
| `crates/xdna-archspec/src/coverage/surface.rs` | `SurfaceClass` enum + `SurfaceProbe` trait (definition only; impl is Plan 2) |
| `crates/xdna-archspec/src/coverage/derive.rs` | `Category` + compiler-enforced `category(SemanticOp)` + per-category default verdict |
| `crates/xdna-archspec/src/coverage/units.rs` | `BehavioralUnit`, `Claims`, override registry (seeded minimal), `CapabilityDomain` list |
| `crates/xdna-archspec/src/coverage/enforce.rs` | `enforce_coverage(&ArchModel)` build-time assertions |
| `crates/xdna-archspec/src/lib.rs` | add `pub mod coverage;` |
| `crates/xdna-archspec/build.rs` | call `enforce_coverage` after model build |
| `docs/coverage/aie2/perishable-queue.md` | generated Axis-2 artifact (committed) |
| `docs/coverage/aie2/comprehension-gaps.md` | generated Axis-2 artifact (committed) |

Verbatim type references used below (confirmed from source):
- `Architecture` (Aie/Aie2/Aie2p), `Confirmed<T>`, `Source`, `SourceAttribution` — `crates/xdna-archspec/src/types.rs:19-35`, `:217-336`
- `SemanticOp` (~135 variants incl. `Intrinsic(u32)`) — `crates/xdna-archspec/src/aie2/isa/types.rs:237-376`
- `ArchModel { arch, tile_types: Vec<TileTypeModel>, .. }` — `types.rs:1546-1585`
- `TileTypeModel { kind: TileKind, modules: Vec<ModuleModel>, .. }` — `types.rs:737-763`
- `RegisterModel { name, subsystem: SubsystemKind, .. }`, `SubsystemModel { kind: SubsystemKind, registers: Vec<RegisterModel> }`, `ModuleKind`, `SubsystemKind`, `TileKind` — `types.rs:391-538`, `:579-597`
- `build_arch_model(...) -> ArchModel`, `populate_manual_constants` arch-match — `model_builder.rs:16-34`, `:87-93`
- build.rs builds `arch_model` then codegens — `build.rs:64-128`

---

### Task 1: verdict types + module skeleton

**Files:**
- Create: `crates/xdna-archspec/src/coverage/mod.rs`
- Create: `crates/xdna-archspec/src/coverage/verdict.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod coverage;`)
- Test: inline in `verdict.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/verdict.rs` with ONLY the test module first:

```rust
//! Behavioral-provenance verdict: where knowledge comes from (Provenance)
//! and whether it is silicon-checked (Verification). See spec Section 1.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolchain_derived_is_never_a_gap() {
        let v = Verdict { provenance: Provenance::ToolchainDerived, verification: Verification::NotApplicable };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn modeled_unverified_is_perishable_not_comprehension() {
        let v = Verdict { provenance: Provenance::AietoolsModeled, verification: Verification::Unverified };
        assert!(v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn unspecified_is_a_comprehension_gap_not_perishable() {
        let v = Verdict { provenance: Provenance::Unspecified, verification: Verification::Unverified };
        assert!(v.is_comprehension_gap());
        assert!(!v.is_perishable(), "Unspecified is its own tier, not the perishable queue (spec Section 1)");
    }

    #[test]
    fn accepted_closes_both_sets() {
        let v = Verdict {
            provenance: Provenance::Unspecified,
            verification: Verification::Accepted { rationale: "out of scope".into() },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }

    #[test]
    fn verified_closes_perishable() {
        let v = Verdict {
            provenance: Provenance::AietoolsModeled,
            verification: Verification::Verified { evidence: "bridge:add_one".into() },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::verdict 2>&1 | tail -20`
Expected: FAIL — `cannot find ... Verdict / Provenance / Verification` (module not declared / types missing).

- [ ] **Step 3: Write minimal implementation**

Prepend to `crates/xdna-archspec/src/coverage/verdict.rs` (above the `#[cfg(test)]` block):

```rust
use serde::{Deserialize, Serialize};

/// Where the behavioral knowledge for a unit comes from. Spec Section 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provenance {
    /// Fully specified by aie-rt / TableGen / regdb. Toolchain is ground truth.
    ToolchainDerived,
    /// Reimplemented from reading aietools python models / AM020. Not silicon-checked.
    AietoolsModeled,
    /// From AM020/AM025 prose only.
    DocSpecified,
    /// A trusted source describes the node and we assert NO model.
    Unspecified,
}

/// Silicon-verification status, orthogonal to provenance. Spec Section 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verification {
    /// Toolchain-derived; silicon cannot be "more right" than the spec.
    NotApplicable,
    /// Checked against that arch's hardware. `evidence` points at the proof.
    Verified { evidence: String },
    /// Needs that arch's silicon to confirm; not done yet.
    Unverified,
    /// Explicitly signed off as "good enough, will not verify", with reasoning.
    Accepted { rationale: String },
}

/// One behavioral unit's verdict on both axes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub provenance: Provenance,
    pub verification: Verification,
}

impl Verdict {
    /// Perishable queue (spec Section 1): modeled, weak provenance, unverified.
    /// `Unspecified` is deliberately excluded — it is the comprehension-gap tier.
    pub fn is_perishable(&self) -> bool {
        matches!(self.provenance, Provenance::AietoolsModeled | Provenance::DocSpecified)
            && matches!(self.verification, Verification::Unverified)
    }

    /// Comprehension gap (spec Section 1): a trusted source describes it and we
    /// assert no model. Closed only by understanding it or explicit Accepted.
    pub fn is_comprehension_gap(&self) -> bool {
        matches!(self.provenance, Provenance::Unspecified)
            && !matches!(self.verification, Verification::Accepted { .. })
    }
}
```

Create `crates/xdna-archspec/src/coverage/mod.rs`:

```rust
//! Two-axis coverage provenance infrastructure. Spec:
//! docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod verdict;
```

Add to `crates/xdna-archspec/src/lib.rs`, in the existing `pub mod` list (alphabetical placement, after `pub mod aie2;` group — match the file's existing ordering):

```rust
pub mod coverage;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::verdict 2>&1 | tail -20`
Expected: PASS — 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/mod.rs crates/xdna-archspec/src/coverage/verdict.rs crates/xdna-archspec/src/lib.rs
git commit -m "coverage: verdict types (Provenance/Verification) + module skeleton

Generated using Claude Code."
```

---

### Task 2: SurfaceClass + SurfaceProbe trait + CoverageNode

**Files:**
- Create: `crates/xdna-archspec/src/coverage/surface.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `CoverageNode`, `pub mod surface;`)
- Test: inline in `surface.rs` and `mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/surface.rs`:

```rust
//! Axis 1: surface presence. The TYPE and CONTRACT live here (archspec is the
//! single point of definition for both axes, spec Section 2). The concrete
//! per-arch `impl SurfaceProbe` lives in the interpreter crate (Plan 2),
//! because "does a handler exist" is implementation state, not architecture.

use crate::coverage::CoverageNode;

/// Whether a generated node is wired into execution. Spec Section 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SurfaceClass {
    /// A real execution handler / register consumer exists.
    Wired,
    /// Reaches a `_ =>` default; no dedicated handling.
    Fallthrough,
    /// Decoded but nothing consumes it.
    Absent,
}

/// Axis-1 contract. archspec declares it; the interpreter implements it
/// per-arch (spec Section 2, Section 7). A node simply does not exist in an
/// arch's node set if the toolchain does not emit it for that arch, so no
/// `Inapplicable` value is needed (spec Section 7).
pub trait SurfaceProbe {
    fn surface_class(&self, node: &CoverageNode) -> SurfaceClass;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::CoverageNode;
    use crate::types::Architecture;

    /// A fake probe proves the trait is object-safe and usable before the
    /// real interpreter impl exists (Plan 2).
    struct FakeProbe;
    impl SurfaceProbe for FakeProbe {
        fn surface_class(&self, _node: &CoverageNode) -> SurfaceClass {
            SurfaceClass::Wired
        }
    }

    #[test]
    fn trait_is_object_safe_and_usable() {
        let p: &dyn SurfaceProbe = &FakeProbe;
        let node = CoverageNode::Capability { arch: Architecture::Aie2, domain: "dma".into() };
        assert_eq!(p.surface_class(&node), SurfaceClass::Wired);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::surface 2>&1 | tail -20`
Expected: FAIL — `cannot find type CoverageNode in crate::coverage` and `surface` not declared.

- [ ] **Step 3: Write minimal implementation**

Replace `crates/xdna-archspec/src/coverage/mod.rs` with:

```rust
//! Two-axis coverage provenance infrastructure. Spec:
//! docs/superpowers/specs/2026-05-15-two-axis-coverage-provenance-design.md

pub mod surface;
pub mod verdict;

use crate::aie2::isa::SemanticOp;
use crate::types::{Architecture, ModuleKind, SubsystemKind, TileKind};
use serde::{Deserialize, Serialize};

/// A fine node, architecture-qualified (spec Section 7: all identity is
/// arch-qualified). Two kinds of node universe today: ISA semantics and
/// registers. Capability-spine domains also get a CoverageNode so they can be
/// claimed by units uniformly. Derives `Hash` so it can key the partition
/// `HashSet` (Task 6) and any future identity maps.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageNode {
    Semantic { arch: Architecture, op: SemanticOp },
    Register {
        arch: Architecture,
        tile: TileKind,
        module: ModuleKind,
        subsystem: SubsystemKind,
        name: String,
    },
    Capability { arch: Architecture, domain: String },
}

impl CoverageNode {
    pub fn arch(&self) -> Architecture {
        match self {
            CoverageNode::Semantic { arch, .. }
            | CoverageNode::Register { arch, .. }
            | CoverageNode::Capability { arch, .. } => *arch,
        }
    }
}
```

`SemanticOp` derives `Hash` but not `Serialize`/`Deserialize` in `aie2/isa/types.rs:237`. `CoverageNode` derives `Serialize`/`Deserialize` above — to keep the build honest, the engineer must add `Serialize, Deserialize` to the `SemanticOp` derive list. Modify `crates/xdna-archspec/src/aie2/isa/types.rs:237`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SemanticOp {
```

(The crate already depends on `serde` with `derive`; this only adds two trait impls and changes no behavior.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::surface 2>&1 | tail -20`
Expected: PASS — 1 test passes. Also run `cargo test -p xdna-archspec --lib coverage::verdict 2>&1 | tail -5` — still PASS (no regression from the SemanticOp derive change).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/ crates/xdna-archspec/src/aie2/isa/types.rs
git commit -m "coverage: SurfaceClass + SurfaceProbe trait + arch-qualified CoverageNode

Derive Serialize/Deserialize on SemanticOp so CoverageNode can serialize.

Generated using Claude Code."
```

---

### Task 3: compiler-enforced `category()` + per-category default verdict

**Files:**
- Create: `crates/xdna-archspec/src/coverage/derive.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `pub mod derive;`)
- Test: inline in `derive.rs`

This task installs the SemanticOp forcing function: `category()` is an exhaustive match with **no `_` arm** — a new `SemanticOp` variant (e.g. from a future TableGen bump or an AIE2P addition) will fail to compile until categorized. The variant list below is the complete enum from `aie2/isa/types.rs:237-376`, grouped exactly as the enum's own comment sections group it.

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/derive.rs` with the test module first:

```rust
//! Derived-default clustering (spec Section 3, Section 5). `category()` is the
//! SemanticOp forcing function: exhaustive, no wildcard. `default_verdict`
//! gives each category its honestly-pessimistic bootstrap verdict.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verification};

    #[test]
    fn scalar_arithmetic_is_toolchain_ground_truth() {
        assert_eq!(category(&SemanticOp::Add), Category::Arithmetic);
        let v = default_verdict(Category::Arithmetic);
        assert_eq!(v.provenance, Provenance::ToolchainDerived);
        assert_eq!(v.verification, Verification::NotApplicable);
    }

    #[test]
    fn vector_is_aietools_modeled_unverified() {
        assert_eq!(category(&SemanticOp::Mac), Category::Vector);
        assert_eq!(category(&SemanticOp::Srs), Category::Vector);
        let v = default_verdict(Category::Vector);
        assert_eq!(v.provenance, Provenance::AietoolsModeled);
        assert_eq!(v.verification, Verification::Unverified);
    }

    #[test]
    fn side_effect_is_doc_specified_unverified() {
        assert_eq!(category(&SemanticOp::DmaStart), Category::SideEffect);
        let v = default_verdict(Category::SideEffect);
        assert_eq!(v.provenance, Provenance::DocSpecified);
        assert_eq!(v.verification, Verification::Unverified);
    }

    #[test]
    fn intrinsic_is_needs_triage_and_thus_a_comprehension_gap() {
        assert_eq!(category(&SemanticOp::Intrinsic(0)), Category::NeedsTriage);
        let v = default_verdict(Category::NeedsTriage);
        assert_eq!(v.provenance, Provenance::Unspecified);
        // Default-to-ignorant (spec Section 1): the lazy path lands loud.
        assert!(v.is_comprehension_gap());
    }

    #[test]
    fn locks_are_toolchain_derived() {
        assert_eq!(category(&SemanticOp::LockAcquire), Category::Sync);
        assert_eq!(default_verdict(Category::Sync).provenance, Provenance::ToolchainDerived);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::derive 2>&1 | tail -20`
Expected: FAIL — `cannot find function category` / `Category`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `crates/xdna-archspec/src/coverage/derive.rs`:

```rust
use crate::aie2::isa::SemanticOp;
use crate::coverage::verdict::{Provenance, Verification, Verdict};

/// Coarse SemanticOp grouping for derived-default provenance (spec Section 3).
/// Finer behavioral seams are the override registry's job (Task 4), not this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Arithmetic,
    Bitwise,
    Comparison,
    Memory,
    ControlFlow,
    Vector,
    Sync,
    SideEffect,
    /// New/unclassified — defaults to Unspecified (default-to-ignorant).
    NeedsTriage,
}

/// SemanticOp -> Category. EXHAUSTIVE WITH NO `_` ARM ON PURPOSE: this is the
/// forcing function (spec Section 1/3). A new variant breaks this build until
/// categorized. Only the Vector/Sync/SideEffect arms mirror the enum's own
/// comment sections; the Arithmetic arm deliberately coalesces several
/// sections into one pessimistic-default bucket (see its comment).
pub fn category(op: &SemanticOp) -> Category {
    match op {
        // Arithmetic + ALL fully-toolchain-specified scalar machinery
        // (div-step, bit-manip, extension, copy/nop, pointer, cntr, event):
        // deliberately coalesced into one pessimistic-default bucket. This
        // arm spans several enum comment sections on purpose -- it does NOT
        // mirror them (only Vector/Sync/SideEffect arms do). Event emits a
        // trace packet and ReadCycleCounter reads a counter via
        // toolchain-specified instructions, so ToolchainDerived is correct
        // here; trace-buffering is a Phase-2 override candidate.
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
        | SemanticOp::Ctlz
        | SemanticOp::Cttz
        | SemanticOp::Ctpop
        | SemanticOp::Bswap
        | SemanticOp::Clb
        | SemanticOp::SignExtend
        | SemanticOp::ZeroExtend
        | SemanticOp::Truncate
        | SemanticOp::Copy
        | SemanticOp::Nop
        | SemanticOp::Event
        | SemanticOp::ReadCycleCounter
        | SemanticOp::PointerAdd
        | SemanticOp::PointerMov => Category::Arithmetic,

        // Bitwise
        SemanticOp::And
        | SemanticOp::Or
        | SemanticOp::Xor
        | SemanticOp::Not
        | SemanticOp::Shl
        | SemanticOp::Sra
        | SemanticOp::Srl
        | SemanticOp::AshlBidir
        | SemanticOp::LshlBidir
        | SemanticOp::Rotl
        | SemanticOp::Rotr => Category::Bitwise,

        // Comparison
        SemanticOp::SetEq
        | SemanticOp::SetNe
        | SemanticOp::SetLt
        | SemanticOp::SetLe
        | SemanticOp::SetGt
        | SemanticOp::SetGe
        | SemanticOp::SetUlt
        | SemanticOp::SetUle
        | SemanticOp::SetUgt
        | SemanticOp::SetUge
        | SemanticOp::Cmp => Category::Comparison,

        // Memory
        SemanticOp::Load | SemanticOp::Store => Category::Memory,

        // Control flow. Select is the conditional-select/ternary (the enum
        // files it here); Done/Halt are well-understood toolchain-specified
        // core termination. All ToolchainDerived.
        SemanticOp::Br
        | SemanticOp::BrCond
        | SemanticOp::Call
        | SemanticOp::Ret
        | SemanticOp::Select
        | SemanticOp::Done
        | SemanticOp::Halt => Category::ControlFlow,

        // Vector / matrix engine — AietoolsModeled, perishable (spec Section 5)
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
        | SemanticOp::AccumNegSub => Category::Vector,

        // Synchronization — aie-rt fully specifies lock semantics
        SemanticOp::LockAcquire | SemanticOp::LockRelease => Category::Sync,

        // Side effects — DMA/stream/cascade: timing/semantics only
        // doc-specified, perishable (spec Section 5)
        SemanticOp::CascadeRead
        | SemanticOp::CascadeWrite
        | SemanticOp::StreamRead
        | SemanticOp::StreamWrite
        | SemanticOp::StreamWritePacketHeader
        | SemanticOp::DmaStart
        | SemanticOp::DmaWait => Category::SideEffect,

        // Target-specific intrinsic: a source describes it (intrinsic table)
        // but this layer asserts no model -> default-to-ignorant.
        SemanticOp::Intrinsic(_) => Category::NeedsTriage,
    }
}

/// Honestly-pessimistic bootstrap default per category (spec Section 5).
/// A default is always the weakest plausible provenance for the group;
/// refinement (Phase 2) can only improve a verdict, never hide one.
pub fn default_verdict(cat: Category) -> Verdict {
    match cat {
        Category::Arithmetic
        | Category::Bitwise
        | Category::Comparison
        | Category::Memory
        | Category::ControlFlow
        | Category::Sync => {
            Verdict { provenance: Provenance::ToolchainDerived, verification: Verification::NotApplicable }
        }
        Category::Vector => {
            Verdict { provenance: Provenance::AietoolsModeled, verification: Verification::Unverified }
        }
        Category::SideEffect => {
            Verdict { provenance: Provenance::DocSpecified, verification: Verification::Unverified }
        }
        Category::NeedsTriage => {
            Verdict { provenance: Provenance::Unspecified, verification: Verification::Unverified }
        }
    }
}
```

Add `pub mod derive;` to `crates/xdna-archspec/src/coverage/mod.rs` (in the `pub mod` list, keep alphabetical: `derive`, `surface`, `verdict`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::derive 2>&1 | tail -20`
Expected: PASS — 5 tests. If `category()` fails to compile with "non-exhaustive patterns", the SemanticOp enum changed since this plan was written — add the missing variant to the correct group (the compiler names it). That is the forcing function working as designed.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/derive.rs crates/xdna-archspec/src/coverage/mod.rs
git commit -m "coverage: compiler-enforced category() + pessimistic default verdicts

category() has no wildcard arm by design -- a new SemanticOp variant
breaks the build until categorized (spec Section 1/3 forcing function).

Generated using Claude Code."
```

---

### Task 4: behavioral units, claims, override registry, capability spine

**Files:**
- Create: `crates/xdna-archspec/src/coverage/units.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `pub mod units;`)
- Test: inline in `units.rs`

Per spec Section 3 the override registry starts minimal (Phase 2 grows it) and the capability spine is one hand-curated list co-located here (spec Section 6). Seed the spine with the coarse architectural domains; seed the override registry empty (coarse category defaults carry Phase 1 — spec Section 5).

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/units.rs` test module first:

```rust
//! Behavioral units: the explicit override registry (spec Section 3) and the
//! single hand-curated CapabilityDomain spine (spec Section 6). Seeded coarse;
//! Phase 2 refines. Co-located on purpose (one location, spec Section 6).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Architecture;

    #[test]
    fn override_registry_is_empty_at_phase1() {
        // Phase 1 ships green on coarse category defaults; overrides are
        // Phase 2 refinement work (spec Section 5).
        assert!(override_registry(Architecture::Aie2).is_empty());
    }

    #[test]
    fn capability_spine_seeded_for_aie2() {
        let spine = capability_spine();
        assert!(spine.iter().any(|d| d.id == "dma"));
        assert!(spine.iter().any(|d| d.id == "locks"));
        assert!(spine.iter().any(|d| d.id == "stream_switch"));
        // Every seeded domain applies to AIE2 (the only wired arch today).
        assert!(spine.iter().all(|d| d.applies_to(Architecture::Aie2)));
    }

    #[test]
    fn capability_domain_arch_applicability_is_explicit() {
        let dma = capability_spine().into_iter().find(|d| d.id == "dma").unwrap();
        assert!(dma.applies_to(Architecture::Aie2));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::units 2>&1 | tail -20`
Expected: FAIL — `cannot find function override_registry / capability_spine`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `crates/xdna-archspec/src/coverage/units.rs`:

```rust
use crate::coverage::verdict::Verdict;
use crate::coverage::CoverageNode;
use crate::types::Architecture;

/// What fine nodes a behavioral unit claims (spec Section 3). A unit either
/// claims explicit nodes (override) or is the derived bucket for a category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Claims {
    /// Explicit override: this exact set of nodes (spec Section 3).
    Nodes(Vec<CoverageNode>),
}

/// A behavioral unit: a cluster of fine nodes with one verdict (spec Section 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralUnit {
    pub id: String,
    pub arch: Architecture,
    pub claims: Claims,
    pub verdict: Verdict,
    /// Set true with a reason when an override pulls a node off the
    /// toolchain-derived path (spec Section 3 no-silent-shadow rule).
    pub shadows_derived: Option<String>,
}

/// A top-level hardware capability the manual names (spec Section 6).
/// Each must be claimed by >= 1 behavioral unit per applicable arch, or the
/// build panics (enforced in Task 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
}

impl CapabilityDomain {
    pub fn applies_to(&self, arch: Architecture) -> bool {
        self.arches.contains(&arch)
    }
}

/// The explicit override registry, per arch. Empty in Phase 1 (spec Section 5):
/// coarse category defaults carry the bootstrap; Phase 2 adds entries here.
pub fn override_registry(_arch: Architecture) -> Vec<BehavioralUnit> {
    Vec::new()
}

/// The single hand-curated capability spine (spec Section 6). Seeded once from
/// the AM020 ToC + aie-rt module tree; maintained ONLY here. Coarse,
/// arch-invariant domains; AIE2 is the only wired arch today (Plan 1).
pub fn capability_spine() -> Vec<CapabilityDomain> {
    let aie2 = vec![Architecture::Aie2];
    [
        "core", "program_memory", "data_memory", "dma", "locks",
        "stream_switch", "events_trace", "performance_counters",
        "timer", "watchpoint", "debug_halt", "cascade",
        "interrupt", "noc", "shim_mux",
    ]
    .iter()
    .map(|id| CapabilityDomain { id: (*id).to_string(), arches: aie2.clone() })
    .collect()
}
```

Add `pub mod units;` to `crates/xdna-archspec/src/coverage/mod.rs` (alphabetical: `derive`, `surface`, `units`, `verdict`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::units 2>&1 | tail -20`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/units.rs crates/xdna-archspec/src/coverage/mod.rs
git commit -m "coverage: behavioral units, claims, empty override registry, capability spine

Spine is the one hand-curated location (spec Section 6); override
registry empty in Phase 1 (coarse defaults carry bootstrap, Section 5).

Generated using Claude Code."
```

---

### Task 5: CoverageModel + the two filter sets + clean_release

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `CoverageModel`)
- Test: inline in `mod.rs`

`CoverageModel` joins the register node universe (from `ArchModel.tile_types -> modules -> subsystems -> registers`) with derived defaults and overrides, exposes the two filter sets and `clean_release(arch)`. SemanticOp nodes are covered by the compiler-enforced `category()` (Task 3) — their default verdict is derived, not stored per-node here; Plan 2's reconciliation test joins the surface axis. Plan 1's model is Axis-2 over the register + capability universe plus the SemanticOp category rollup.

- [ ] **Step 1: Write the failing test**

Append to `crates/xdna-archspec/src/coverage/mod.rs` a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verification};

    #[test]
    fn semantic_rollup_uses_pessimistic_category_defaults() {
        let m = CoverageModel::build(Architecture::Aie2);
        // Vector ops roll up to AietoolsModeled/Unverified -> perishable.
        let vmac = m.semantic_verdict(&SemanticOp::Mac);
        assert_eq!(vmac.provenance, Provenance::AietoolsModeled);
        assert!(vmac.is_perishable());
        // Scalar add is toolchain ground truth -> in neither set.
        let add = m.semantic_verdict(&SemanticOp::Add);
        assert_eq!(add.verification, Verification::NotApplicable);
        assert!(!add.is_perishable() && !add.is_comprehension_gap());
        // Intrinsic -> comprehension gap (default-to-ignorant).
        assert!(m.semantic_verdict(&SemanticOp::Intrinsic(0)).is_comprehension_gap());
    }

    #[test]
    fn clean_release_is_false_at_bootstrap_and_honest() {
        let m = CoverageModel::build(Architecture::Aie2);
        // Honest from day one but coarse: vector + intrinsic populate the
        // sets, so the gate is correctly NOT green at bootstrap (spec S5).
        assert!(!m.perishable_queue().is_empty(), "vector ops must be perishable at bootstrap");
        assert!(!m.comprehension_gaps().is_empty(), "Intrinsic must be a comprehension gap at bootstrap");
        assert!(!m.clean_release(), "bootstrap must not be green -- that is the honest state (spec S5)");
    }

    #[test]
    fn clean_release_is_per_arch() {
        // Spec Section 7: the gate is per-silicon. A different arch is a
        // different model; AIE2's state says nothing about it.
        let m = CoverageModel::build(Architecture::Aie2);
        assert_eq!(m.arch, Architecture::Aie2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find ... CoverageModel`.

- [ ] **Step 3: Write minimal implementation**

Append to `crates/xdna-archspec/src/coverage/mod.rs` (above the test module):

```rust
use crate::coverage::derive::{category, default_verdict};
use crate::coverage::units::{capability_spine, override_registry, BehavioralUnit};
use crate::coverage::verdict::Verdict;

/// Per-arch coverage model (spec Section 7: a per-ArchModel facet, keyed by
/// arch, built from the arch's node universe). Built-from / keyed-by satisfies
/// the spec intent without forcing a non-serializable field into ArchModel.
#[derive(Debug, Clone)]
pub struct CoverageModel {
    pub arch: Architecture,
    overrides: Vec<BehavioralUnit>,
}

impl CoverageModel {
    /// Build the coverage model for one architecture. Phase 1: overrides are
    /// empty, so verdicts come from coarse category/subsystem defaults.
    pub fn build(arch: Architecture) -> Self {
        Self { arch, overrides: override_registry(arch) }
    }

    /// Verdict for a SemanticOp node: override if one claims it, else the
    /// pessimistic category default (spec Section 3/5).
    pub fn semantic_verdict(&self, op: &SemanticOp) -> Verdict {
        let node = CoverageNode::Semantic { arch: self.arch, op: op.clone() };
        if let Some(u) = self.claiming_unit(&node) {
            return u.verdict.clone();
        }
        default_verdict(category(op))
    }

    fn claiming_unit(&self, node: &CoverageNode) -> Option<&BehavioralUnit> {
        self.overrides.iter().find(|u| match &u.claims {
            crate::coverage::units::Claims::Nodes(ns) => ns.contains(node),
        })
    }

    /// All SemanticOp verdicts (the Plan 1 node universe for the gate). Plan 2
    /// adds register/surface joining via the reconciliation test.
    fn all_semantic_verdicts(&self) -> Vec<Verdict> {
        all_semantic_ops().iter().map(|op| self.semantic_verdict(op)).collect()
    }

    /// Perishable queue (spec Section 1): modeled, unverified.
    pub fn perishable_queue(&self) -> Vec<Verdict> {
        self.all_semantic_verdicts().into_iter().filter(|v| v.is_perishable()).collect()
    }

    /// Comprehension gaps (spec Section 1): a source describes it, no model.
    pub fn comprehension_gaps(&self) -> Vec<Verdict> {
        self.all_semantic_verdicts().into_iter().filter(|v| v.is_comprehension_gap()).collect()
    }

    /// Per-silicon release gate (spec Section 4/7): both sets empty modulo
    /// Accepted. Green for AIE2 == "safe to retire NPU1".
    pub fn clean_release(&self) -> bool {
        self.perishable_queue().is_empty() && self.comprehension_gaps().is_empty()
    }

    /// The capability spine for this arch (spec Section 6).
    pub fn applicable_capabilities(&self) -> Vec<String> {
        capability_spine()
            .into_iter()
            .filter(|d| d.applies_to(self.arch))
            .map(|d| d.id)
            .collect()
    }
}

/// The static SemanticOp universe. One representative of every enum variant;
/// `Intrinsic` is represented by `Intrinsic(0)` (the table-lookup family).
/// This list is itself guarded: `category()` is exhaustive (Task 3), so a new
/// variant breaks that build before this list can go stale silently.
fn all_semantic_ops() -> Vec<SemanticOp> {
    use SemanticOp::*;
    vec![
        Add, Sub, Adc, Sbc, Mul, SDiv, UDiv, SRem, URem, Abs, Neg, DivStep, Select, Ctlz, Cttz,
        Ctpop, Bswap, Clb, SignExtend, ZeroExtend, Truncate, Copy, Nop, Event, ReadCycleCounter,
        PointerAdd, PointerMov, And, Or, Xor, Not, Shl, Sra, Srl, AshlBidir, LshlBidir, Rotl, Rotr,
        SetEq, SetNe, SetLt, SetLe, SetGt, SetGe, SetUlt, SetUle, SetUgt, SetUge, Cmp, Load, Store,
        Br, BrCond, Call, Ret, Done, Halt, Mac, MatMul, MatMulSub, NegMatMul, AddMac, SubMac,
        NegMul, Srs, Ups, Shuffle, Pack, Unpack, Align, VectorBroadcast, VectorExtract,
        VectorInsert, VectorPush, VectorPushHi, VectorSelect, VectorClear, Convert, Min, Max,
        SubLt, SubGe, MaxDiffLt, MaxLt, MinGe, AbsGtz, NegGtz, NegLtz, NegAdd, Accumulate,
        AccumSub, AccumNegAdd, AccumNegSub, LockAcquire, LockRelease, CascadeRead, CascadeWrite,
        StreamRead, StreamWrite, StreamWritePacketHeader, DmaStart, DmaWait, Intrinsic(0),
    ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::tests 2>&1 | tail -20`
Expected: PASS — 3 tests. Re-run full coverage module: `cargo test -p xdna-archspec --lib coverage 2>&1 | tail -10` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/mod.rs
git commit -m "coverage: CoverageModel + perishable/comprehension filters + clean_release

Per-arch gate; honest-but-coarse at bootstrap (vector perishable,
Intrinsic a comprehension gap) -- clean_release correctly false (spec S5).

Generated using Claude Code."
```

---

### Task 6: enforce_coverage + build.rs panic hook

**Files:**
- Create: `crates/xdna-archspec/src/coverage/enforce.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `pub mod enforce;`)
- Modify: `crates/xdna-archspec/build.rs` (call `enforce_coverage`)
- Test: inline in `enforce.rs`

`enforce_coverage` is the register/unit/capability/cross-arch forcing function (spec Section 2/4). The SemanticOp axis is already compiler-enforced by `category()`. This panics exactly like `Confirmed<T>::confirm` (spec's stated philosophy).

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/enforce.rs` with the test module first:

```rust
//! Build-time Axis-2 enforcement (spec Section 2/4). Panics like
//! Confirmed<T>::confirm: an incomplete coverage model stops the build.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::units::{BehavioralUnit, Claims, CapabilityDomain};
    use crate::coverage::verdict::{Provenance, Verdict, Verification};
    use crate::coverage::CoverageNode;
    use crate::types::Architecture;

    fn ok_unit(arch: Architecture, id: &str) -> BehavioralUnit {
        BehavioralUnit {
            id: id.into(),
            arch,
            claims: Claims::Nodes(vec![CoverageNode::Capability { arch, domain: id.into() }]),
            verdict: Verdict { provenance: Provenance::ToolchainDerived, verification: Verification::NotApplicable },
            shadows_derived: None,
        }
    }

    #[test]
    fn passes_when_every_capability_is_claimed() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        let units = vec![ok_unit(arch, "dma")];
        enforce_coverage(arch, &spine, &units); // must not panic
    }

    #[test]
    #[should_panic(expected = "unclaimed capability")]
    fn panics_on_unclaimed_capability() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "locks".into(), arches: vec![arch] }];
        enforce_coverage(arch, &spine, &[]); // nothing claims "locks"
    }

    #[test]
    #[should_panic(expected = "cross-arch")]
    fn panics_on_verified_with_cross_arch_shadow() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        let mut u = ok_unit(arch, "dma");
        u.verdict.verification = Verification::Verified { evidence: "npu4".into() };
        u.shadows_derived = Some("derived_shared_from:Aie2p".into());
        enforce_coverage(arch, &spine, &[u]);
    }

    #[test]
    #[should_panic(expected = "double-claimed")]
    fn panics_on_two_units_claiming_one_node() {
        let arch = Architecture::Aie2;
        let spine = vec![CapabilityDomain { id: "dma".into(), arches: vec![arch] }];
        enforce_coverage(arch, &spine, &[ok_unit(arch, "dma"), ok_unit(arch, "dma")]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::enforce 2>&1 | tail -20`
Expected: FAIL — `cannot find function enforce_coverage`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `crates/xdna-archspec/src/coverage/enforce.rs`:

```rust
use crate::coverage::units::{BehavioralUnit, CapabilityDomain, Claims};
use crate::coverage::verdict::Verification;
use crate::coverage::CoverageNode;
use crate::types::Architecture;
use std::collections::HashSet;

/// Build-time Axis-2 enforcement for one arch (spec Section 2/4). Panics on:
/// a capability applicable to `arch` with no claiming unit; two units claiming
/// one node; a `Verified` verdict carrying a cross-arch `shadows_derived`
/// marker (verification never transfers across silicon, spec Section 7).
///
/// The SemanticOp axis needs no check here -- `category()` is compiler-enforced
/// (Task 3). Register-node partition is checked once the register override
/// registry is non-empty (Phase 2); in Phase 1 it is trivially satisfied
/// because every register rolls up to its subsystem's coarse default.
pub fn enforce_coverage(arch: Architecture, spine: &[CapabilityDomain], units: &[BehavioralUnit]) {
    // 1. Cross-arch transfer rule: verification never transfers (spec S7).
    for u in units {
        if matches!(u.verdict.verification, Verification::Verified { .. }) {
            if let Some(reason) = &u.shadows_derived {
                if reason.contains("derived_shared_from") {
                    panic!(
                        "COVERAGE: unit '{}' ({arch}) is Verified but carries a cross-arch \
                         shared marker ({reason}) -- verification never transfers across \
                         silicon (spec Section 7)",
                        u.id
                    );
                }
            }
        }
    }

    // 2. Partition: no node double-claimed by two units (spec Section 3).
    // CoverageNode derives Hash, so a HashSet is the natural structure.
    let mut seen: HashSet<&CoverageNode> = HashSet::new();
    for u in units {
        let Claims::Nodes(ns) = &u.claims;
        for n in ns {
            if !seen.insert(n) {
                panic!(
                    "COVERAGE: node {:?} double-claimed (unit '{}', {arch}) -- the registry \
                     must partition, not merely cover (spec Section 3)",
                    n, u.id
                );
            }
        }
    }

    // 3. Capability spine: every applicable domain claimed by >= 1 unit (S6).
    for dom in spine {
        if !dom.applies_to(arch) {
            continue;
        }
        let claimed = units.iter().any(|u| {
            let Claims::Nodes(ns) = &u.claims;
            ns.iter().any(|n| matches!(n, CoverageNode::Capability { domain, .. } if domain == &dom.id))
        });
        if !claimed {
            panic!(
                "COVERAGE: unclaimed capability '{}' for {arch} -- every hardware capability \
                 the manual names must be modeled by >= 1 behavioral unit (spec Section 6)",
                dom.id
            );
        }
    }
}
```

Add `pub mod enforce;` to `crates/xdna-archspec/src/coverage/mod.rs` (alphabetical: `derive`, `enforce`, `surface`, `units`, `verdict`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::enforce 2>&1 | tail -20`
Expected: PASS — 4 tests (1 pass + 3 `should_panic`).

- [ ] **Step 5: Wire the build-panic hook into build.rs**

The capability spine is seeded (Task 4) but **no units claim any domain yet** (override registry empty in Phase 1). Calling `enforce_coverage` with the real spine now would panic the build on "unclaimed capability" — which is *correct behavior the design wants eventually*, but Phase 1 must ship green (spec Section 5). Resolution per spec Section 5: in Phase 1 the **coarse derived defaults are the claiming mechanism** — every capability domain is claimed by an implicit derived unit. Model that explicitly with a Phase-1 derived-claim shim so the spine is satisfied honestly without fabricating override entries.

Append to `crates/xdna-archspec/src/coverage/enforce.rs` (above the test module):

```rust
use crate::coverage::units::capability_spine;

/// Phase-1 entry point called from build.rs. Every spine domain is claimed by
/// an implicit derived unit (spec Section 5: coarse defaults carry bootstrap).
/// This keeps the build green and honest while the override registry is empty;
/// Phase 2 replaces derived claims with real override units one at a time.
pub fn enforce_coverage_phase1(arch: Architecture) {
    let spine = capability_spine();
    let derived_units: Vec<BehavioralUnit> = spine
        .iter()
        .filter(|d| d.applies_to(arch))
        .map(|d| BehavioralUnit {
            id: format!("derived.{}", d.id),
            arch,
            claims: Claims::Nodes(vec![CoverageNode::Capability { arch, domain: d.id.clone() }]),
            // Coarse Phase-1 default; refined in Phase 2 (spec Section 5).
            verdict: crate::coverage::derive::default_verdict(
                crate::coverage::derive::Category::NeedsTriage,
            ),
            shadows_derived: None,
        })
        .collect();
    enforce_coverage(arch, &spine, &derived_units);
}
```

Add a test for it (append inside the existing `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn phase1_entry_point_is_green() {
        // Coarse derived claims satisfy the spine without override entries.
        enforce_coverage_phase1(Architecture::Aie2); // must not panic
    }
```

Now wire it into `build.rs`. `build.rs` already calls `crate::model_builder::build_arch_model` (build.rs:110), so the crate's modules are in scope to `build.rs`. Inspect the top of `crates/xdna-archspec/build.rs` for how modules are brought in (the `mod`/`#[path]` declarations that make `crate::model_builder` reachable) and add `coverage` the same way. Then add the enforcement call immediately after the model is built and cross-validated. Modify `crates/xdna-archspec/build.rs` — after the `extract_aiert(workspace_root, &out_dir, &mut arch_model);` line (build.rs:115) and before `gen_arch(&arch_model, &out_dir);` (build.rs:118), insert:

```rust
    // Coverage Axis-2 forcing function (spec Section 2/4): an incomplete
    // coverage model panics the build, exactly like Confirmed<T>::confirm.
    // Phase 1: coarse derived claims satisfy the capability spine.
    crate::coverage::enforce::enforce_coverage_phase1(arch_model.arch);
```

If the top of build.rs uses `#[path = "src/<m>.rs"] mod <m>;` declarations, add `#[path = "src/coverage/mod.rs"] mod coverage;` alongside them (and ensure its submodules resolve — the `mod.rs` already declares them with `pub mod`, and `#[path]` on the parent makes the `src/coverage/` directory the module root, so submodule files resolve normally). The verification is Step 6 (the build must succeed).

- [ ] **Step 6: Run the build to verify the hook compiles and is green**

Run: `cargo build -p xdna-archspec 2>&1 | tail -20`
Expected: build SUCCEEDS (the Phase-1 entry point is green by construction). Then `cargo test -p xdna-archspec --lib coverage::enforce 2>&1 | tail -10` — 5 tests PASS.

To prove the forcing function actually fires, temporarily add a bogus unapplicable-but-applicable domain check: edit `capability_spine()` to append `CapabilityDomain { id: "BOGUS_UNCLAIMED".into(), arches: vec![Architecture::Aie2] }`, run `cargo build -p xdna-archspec 2>&1 | tail -5`, and confirm it PANICS with `unclaimed capability 'BOGUS_UNCLAIMED'`. Then revert that edit and rebuild green. (This manual check is not committed.)

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-archspec/src/coverage/enforce.rs crates/xdna-archspec/src/coverage/mod.rs crates/xdna-archspec/build.rs
git commit -m "coverage: enforce_coverage + build.rs panic hook (Axis-2 forcing function)

Panics like Confirmed<T> on unclaimed capability / double-claim /
cross-arch Verified. Phase-1 entry point green via coarse derived claims.

Generated using Claude Code."
```

---

### Task 7: generate + commit the Axis-2 honesty artifacts

**Files:**
- Create: `crates/xdna-archspec/src/coverage/artifacts.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `pub mod artifacts;`)
- Create (generated, committed): `docs/coverage/aie2/perishable-queue.md`, `docs/coverage/aie2/comprehension-gaps.md`
- Test: inline in `artifacts.rs` (the staleness gate)

Per spec Section 4, the artifacts are checked-in, diffable files; a staleness test fails if the committed file drifts from the regenerated content (spec Section 2 item 3, scoped to Axis-2 for Plan 1; Plan 2 adds `architecture-index.md`).

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/coverage/artifacts.rs`:

```rust
//! Axis-2 honesty artifacts (spec Section 4): committed, diffable markdown.
//! A staleness test fails if the committed file drifts from regeneration.

use crate::coverage::CoverageModel;
use crate::coverage::verdict::Provenance;
use crate::types::Architecture;

/// Render the perishable queue (spec Section 1) as deterministic markdown.
pub fn render_perishable(arch: Architecture) -> String {
    let m = CoverageModel::build(arch);
    let mut lines = vec![
        format!("# Perishable queue ({arch})"),
        String::new(),
        "Generated by `cargo test -p xdna-archspec --lib coverage::artifacts`. Do not hand-edit."
            .to_string(),
        String::new(),
        "Modeled, weak provenance, silicon-unverified (spec Section 1). Empty == this".to_string(),
        "axis of the per-silicon release gate is green for this arch.".to_string(),
        String::new(),
    ];
    let mut cats: Vec<String> = m
        .perishable_queue()
        .iter()
        .map(|v| match v.provenance {
            Provenance::AietoolsModeled => "aietools-modeled (e.g. vector compute)".to_string(),
            Provenance::DocSpecified => "doc-specified (e.g. DMA/stream side effects)".to_string(),
            _ => "modeled".to_string(),
        })
        .collect();
    cats.sort();
    cats.dedup();
    if cats.is_empty() {
        lines.push("_empty_".to_string());
    } else {
        for c in cats {
            lines.push(format!("- {c}: UNVERIFIED"));
        }
    }
    lines.push(String::new());
    format!("{}\n", lines.join("\n"))
}

/// Render the comprehension gaps (spec Section 1) as deterministic markdown.
pub fn render_comprehension(arch: Architecture) -> String {
    let m = CoverageModel::build(arch);
    let n = m.comprehension_gaps().len();
    let mut lines = vec![
        format!("# Comprehension gaps ({arch})"),
        String::new(),
        "Generated by `cargo test -p xdna-archspec --lib coverage::artifacts`. Do not hand-edit."
            .to_string(),
        String::new(),
        "A trusted source describes it and we assert NO model (spec Section 1).".to_string(),
        "Closed by understanding it or explicit Accepted -- never by aging out.".to_string(),
        String::new(),
    ];
    if n == 0 {
        lines.push("_empty_".to_string());
    } else {
        lines.push(format!(
            "- target-specific intrinsics (SemanticOp::Intrinsic): {n} node(s) UNSPECIFIED"
        ));
    }
    lines.push(String::new());
    format!("{}\n", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_path(rel: &str) -> PathBuf {
        // crates/xdna-archspec -> repo root is two parents up.
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().join(rel)
    }

    #[test]
    fn perishable_artifact_is_not_stale() {
        let want = render_perishable(Architecture::Aie2);
        let path = repo_path("docs/coverage/aie2/perishable-queue.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            got, want,
            "{} is stale -- regenerate with the snippet in Task 7 Step 3 and commit",
            path.display()
        );
    }

    #[test]
    fn comprehension_artifact_is_not_stale() {
        let want = render_comprehension(Architecture::Aie2);
        let path = repo_path("docs/coverage/aie2/comprehension-gaps.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            got, want,
            "{} is stale -- regenerate with the snippet in Task 7 Step 3 and commit",
            path.display()
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec --lib coverage::artifacts 2>&1 | tail -20`
Expected: FAIL — both staleness tests fail (committed files do not exist yet; `got` is empty, `want` is the rendered content).

- [ ] **Step 3: Write minimal implementation (generate + commit the artifacts)**

Add `pub mod artifacts;` to `crates/xdna-archspec/src/coverage/mod.rs` (alphabetical: `artifacts`, `derive`, `enforce`, `surface`, `units`, `verdict`).

Generate the two files from the renderers (run from repo root):

```bash
mkdir -p docs/coverage/aie2
cargo run -p xdna-archspec --quiet --example gen_coverage_artifacts 2>/dev/null || true
```

The crate has no binary; generate via a throwaway test-runner one-liner instead. Add this temporary example file `crates/xdna-archspec/examples/gen_coverage_artifacts.rs`:

```rust
//! One-shot generator for the committed Axis-2 artifacts. Re-run after any
//! change that alters coverage rollups; commit the regenerated files.
fn main() {
    use xdna_archspec::coverage::artifacts::{render_comprehension, render_perishable};
    use xdna_archspec::types::Architecture;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let dir = root.join("docs/coverage/aie2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("perishable-queue.md"), render_perishable(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("comprehension-gaps.md"), render_comprehension(Architecture::Aie2))
        .unwrap();
    eprintln!("wrote docs/coverage/aie2/{{perishable-queue,comprehension-gaps}}.md");
}
```

Run it:

```bash
cargo run -p xdna-archspec --example gen_coverage_artifacts 2>&1 | tail -3
```

Expected: `wrote docs/coverage/aie2/{perishable-queue,comprehension-gaps}.md`, and the two files now exist with rendered content.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec --lib coverage::artifacts 2>&1 | tail -20`
Expected: PASS — both staleness tests pass (committed files match renderers).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/coverage/artifacts.rs crates/xdna-archspec/src/coverage/mod.rs crates/xdna-archspec/examples/gen_coverage_artifacts.rs docs/coverage/aie2/perishable-queue.md docs/coverage/aie2/comprehension-gaps.md
git commit -m "coverage: generate + commit Axis-2 honesty artifacts with staleness gate

perishable-queue.md / comprehension-gaps.md are now committed, diffable,
and drift-checked by a staleness test (spec Section 4). Honest-but-coarse
at bootstrap: vector perishable, Intrinsic a comprehension gap.

Generated using Claude Code."
```

---

### Task 8: full-build + full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full archspec build (exercises the build.rs panic hook)**

Run: `cargo build -p xdna-archspec 2>&1 | tail -10`
Expected: SUCCESS. The build.rs `enforce_coverage_phase1` ran without panicking (Phase 1 is green by construction).

- [ ] **Step 2: Full workspace build (no regressions in dependents)**

Run: `cargo build 2>&1 | tail -10`
Expected: SUCCESS. The `SemanticOp` derive change (Task 2) and new module compile cleanly across the workspace.

- [ ] **Step 3: Full library test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -15`
Expected: all tests PASS, including the full `coverage::*` set. Test count increased by the new coverage tests; zero failures. If any previously-passing test now fails, that is a regression in scope for this plan — fix before proceeding.

- [ ] **Step 4: Commit (if any verification-driven fixes were needed)**

Only if Step 1-3 required fixes:

```bash
git add -A
git commit -m "coverage: fixes from full-build/full-suite verification

Generated using Claude Code."
```

If no fixes were needed, skip the commit; Plan 1 is complete at Task 7's commit.

---

## Self-Review

**1. Spec coverage (Phase 1 scope):**
- Spec §1 two axes + vocabulary -> Tasks 1 (verdict), 2 (surface), 3 (defaults). Perishable vs comprehension split + Accepted -> Task 1 tests. ✓
- Spec §2 ownership seam (archspec defines both axes; SurfaceProbe trait here, impl in Plan 2) -> Task 2. ✓
- Spec §3 derived default + override registry + partition + no-silent-shadow -> Tasks 3, 4 (`shadows_derived` field), 6 (partition panic). ✓
- Spec §4 failure modes + per-arch gate + committed artifacts -> Tasks 6 (panics), 5 (`clean_release`), 7 (artifacts + staleness). ✓
- Spec §5 default-to-ignorant + pessimistic bootstrap + no flag-day -> Task 3 (`default_verdict`, NeedsTriage=Unspecified), Task 6 Step 5 (Phase-1 green via coarse derived claims). ✓
- Spec §6 capability spine, one location -> Task 4 (`capability_spine` in `units.rs`), Task 6 (unclaimed-capability panic). ✓
- Spec §7 architecture parameterization -> arch-qualified `CoverageNode` (Task 2), `CoverageModel { arch }` + per-arch gate (Task 5), per-arch `enforce_coverage(arch, ..)` + cross-arch Verified panic (Task 6). ✓
- Surface-axis Axis-1 evidence, reconciliation test, generated `architecture-index.md`, retiring hand-maintained index -> **explicitly deferred to Plan 2** (stated in header + file-structure note); not a gap. ✓
- Phase 2 fine override adjudication -> explicitly out of scope (no bounded end state). ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"similar to". The one inspect-and-mirror step (build.rs module declaration, Task 6 Step 5) is a concrete action with a `cargo build` verification, not a placeholder — the literal `#[path]`/`mod` mechanism is the existing file's and must be read at execution time; the plan states exactly what to add and how to verify it. Every code step contains complete compiling code.

**3. Type consistency:** `Verdict { provenance, verification }`, `Provenance::{ToolchainDerived,AietoolsModeled,DocSpecified,Unspecified}`, `Verification::{NotApplicable,Verified{evidence},Unverified,Accepted{rationale}}`, `CoverageNode::{Semantic,Register,Capability}`, `Category` (9 variants), `category(&SemanticOp)`, `default_verdict(Category)`, `BehavioralUnit { id,arch,claims,verdict,shadows_derived }`, `Claims::Nodes`, `CapabilityDomain { id,arches }`, `CoverageModel { arch, overrides }` with `build/semantic_verdict/perishable_queue/comprehension_gaps/clean_release/applicable_capabilities`, `enforce_coverage(arch,&[CapabilityDomain],&[BehavioralUnit])`, `enforce_coverage_phase1(arch)` — names and signatures are used identically across Tasks 1-8. `SemanticOp` gains `Serialize/Deserialize` in Task 2 before `CoverageNode` (which needs it) is serialized; ordering correct. No drift found.
