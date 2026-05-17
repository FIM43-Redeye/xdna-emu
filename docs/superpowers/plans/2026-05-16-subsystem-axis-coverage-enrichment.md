# Subsystem-Axis Coverage Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the capability-spine (subsystem) axis into a first-class generated, staleness-gated coverage view woven bidirectionally to the SemanticOp-category axis, recovering the retired hand-maintained architecture index's per-subsystem rows as generated data.

**Architecture:** Extend `Verification` with one `Modeled { completeness }` variant (+ `is_implementation_gap`); enrich `CapabilityDomain` with seeded source/location/narrative/verdict fields; extend the spine 16→20; add a category→domain link with a worst-wins rollup and a directional drift cross-check; render two new generated artifacts and re-weave the existing one; repoint inbound docs.

**Tech Stack:** Rust (`xdna-archspec` crate), serde, inline `#[cfg(test)] mod` tests, generated+committed markdown gated by staleness tests.

**Spec:** `docs/superpowers/specs/2026-05-16-subsystem-axis-coverage-enrichment-design.md` (read it; this plan implements it section-by-section).

**Source of seed content:** the retired index is in git at `1afdb20^:docs/coverage/architecture-index.md`. Run `git show 1afdb20^:docs/coverage/architecture-index.md` to read it; Tasks 3-4 fold its rows onto domains per the spec Appendix.

**Conventions (all tasks):**
- Build/test ground truth is bare `cargo build -p xdna-archspec` and `cargo test -p xdna-archspec --lib` run from `/home/triple/npu-work/xdna-emu`. Never pipe through a filter. Rust-analyzer/harness diagnostics may lag — bare cargo is authoritative.
- Sandbox-safe test runs: prefix `TMPDIR=/tmp/claude-1000`.
- No emoji anywhere. Commit messages end with a line `Generated using Claude Code.`
- The repo has a rustfmt PostToolUse hook; do not hand-reformat.
- Per spec Section 8 cross-task ordering rule: any task that changes a verdict, seed value, category→domain tag, or the spine list MUST regenerate and commit all affected artifacts in that same task (the generator + commit steps are written into each such task below).

---

### Task 1: Vocabulary — extend `Verification`

Implements spec Section 1. Purely additive: no existing exhaustive match on `Verification` exists (current sites use `matches!` or `{:?}` Debug), so this compiles without forced edits. Step 2 proves that with the compiler.

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/verdict.rs`

- [ ] **Step 1: Add the `Completeness` enum, the `Modeled` variant, the predicate, and the documented invariant**

In `crates/xdna-archspec/src/coverage/verdict.rs`, add `Completeness` immediately above `pub enum Verification` (after the `Provenance` enum):

```rust
/// Implementation completeness of a modeled subsystem (spec Section 1).
/// Orthogonal in meaning to provenance/verification but folded into the one
/// verdict vocabulary deliberately (spec Risks). `Partial` names the absent
/// sub-behavior so the gap is self-documenting and not free prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Completeness {
    /// Built and exercised by tests.
    Full,
    /// Layout / some behavior present; the named sub-behavior is absent.
    Partial { missing: String },
    /// Placeholder defaults, no real state machine.
    Stub,
}
```

Add the new variant as the LAST arm of `pub enum Verification` (after `Accepted { rationale: String }`):

```rust
    /// The emulator implements this to `completeness`; verification status is
    /// implied by provenance, not asserted here (spec Section 1). Minted on
    /// `CapabilityDomain` seeds, never by a category default.
    Modeled { completeness: Completeness },
```

Add this method inside `impl Verdict` (after `is_comprehension_gap`, before `hardware_observed`):

```rust
    /// Implementation gap (spec Section 1, third gap class): the subsystem is
    /// only partially built or a stub. Evaluated over `CapabilityDomain`
    /// seeded verdicts ONLY -- never the SemanticOp universe, where no
    /// category default is ever `Modeled` (that would be a permanently empty
    /// queue, the silent failure this design exists to kill -- spec S1/M1).
    pub fn is_implementation_gap(&self) -> bool {
        matches!(
            self.verification,
            Verification::Modeled {
                completeness: Completeness::Partial { .. } | Completeness::Stub
            }
        )
    }
```

Extend the `Verdict` struct doc comment (the `/// Invariant:` block above `pub struct Verdict`) by appending this sentence to the existing invariant paragraph:

```rust
/// `Provenance::Unspecified` is likewise never paired with
/// `Verification::Modeled`: asserting *no model* and asserting *a model
/// exists* are contradictory. `enforce_coverage` rejects this pairing on
/// seeded domain verdicts (test-gated, spec Section 2).
```

- [ ] **Step 2: Verify the crate still compiles (proves predicates/sites unchanged)**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec`
Expected: builds clean. If the compiler flags any exhaustive `match` on `Verification` missing the `Modeled` arm, that is safety mechanism (2) from spec Section 1 working — add an explicit `Verification::Modeled { .. } =>` arm at each flagged site with no `_` wildcard, matching the surrounding code's intent, then rebuild.

- [ ] **Step 3: Write the failing tests**

Append to the `#[cfg(test)] mod tests` block at the bottom of `verdict.rs`:

```rust
    #[test]
    fn modeled_full_is_no_gap() {
        let v = Verdict {
            provenance: Provenance::ToolchainDerived,
            verification: Verification::Modeled { completeness: Completeness::Full },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
        assert!(!v.is_implementation_gap());
    }

    #[test]
    fn modeled_partial_and_stub_are_implementation_gaps_only() {
        let part = Verdict {
            provenance: Provenance::ToolchainDerived,
            verification: Verification::Modeled {
                completeness: Completeness::Partial { missing: "scrubber".into() },
            },
        };
        let stub = Verdict {
            provenance: Provenance::DocSpecified,
            verification: Verification::Modeled { completeness: Completeness::Stub },
        };
        for v in [&part, &stub] {
            assert!(v.is_implementation_gap());
            assert!(!v.is_perishable(), "completeness is orthogonal to perishable");
            assert!(!v.is_comprehension_gap());
        }
    }

    #[test]
    fn existing_predicates_ignore_modeled() {
        // Safety mechanism (1), spec S1: is_perishable / is_comprehension_gap
        // keep their exact arms; Modeled matches neither, by construction.
        let v = Verdict {
            provenance: Provenance::AietoolsModeled,
            verification: Verification::Modeled { completeness: Completeness::Full },
        };
        assert!(!v.is_perishable());
        assert!(!v.is_comprehension_gap());
    }
```

- [ ] **Step 4: Run the tests**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib verdict::`
Expected: the three new tests plus all pre-existing `verdict::` tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/verdict.rs
git commit -m "coverage: extend Verification with Modeled { completeness }

Adds Completeness (Full/Partial{missing}/Stub) and the third gap-class
predicate is_implementation_gap. Purely additive; existing predicates
provably unchanged (Modeled matches neither). Spec Section 1.

Generated using Claude Code."
```

---

### Task 2: Spine extension + `CapabilityDomain` enrichment + enforce loop

Implements spec Section 2 (struct + spine extension + the N2 plumbing) with an honest non-curated skeleton seed. The curated per-domain content is Tasks 3-4; this task makes the model compile, validate, and stay green with every domain in an honest `Partial{missing:"…pending seed…"}` state (it will appear in the Task-6 implementation-gaps queue until seeded — that is the correct self-correcting behavior).

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/spine_ids.rs`
- Modify: `crates/xdna-archspec/src/coverage/units.rs`
- Modify: `crates/xdna-archspec/src/coverage/enforce.rs`

- [ ] **Step 1: Extend the spine id leaf 16→20**

In `crates/xdna-archspec/src/coverage/spine_ids.rs`, add four ids to the END of the `SPINE_DOMAIN_IDS` array (after `"shim_mux",`):

```rust
    "control_packets",
    "clock_control",
    "tile_isolation",
    "binary_load",
```

Do not add any `use`/`crate::` — the leaf must keep zero imports (build.rs `#[path]`-includes it).

- [ ] **Step 2: Enrich the `CapabilityDomain` struct**

In `crates/xdna-archspec/src/coverage/units.rs`, replace the `CapabilityDomain` struct with:

```rust
/// A top-level hardware capability the manual names (spec Section 6), now
/// carrying the retired index's per-subsystem detail as seeded data (spec
/// Section 2). Each must be claimed by >= 1 behavioral unit per applicable
/// arch, or the build panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
    /// Authoritative source (aie-rt path / AM025 / device model). Non-empty,
    /// enforced.
    pub source_ref: String,
    /// Our emulator src/ path(s). Non-empty unless an OOS/MISSING narrative is
    /// present (enforced).
    pub src_locations: Vec<String>,
    /// The old "Notes / known gaps" column.
    pub narrative: String,
    /// The domain's own asserted coverage verdict (spec Section 1/2).
    pub verdict: Verdict,
    /// Documents a deliberate own-vs-rollup divergence (spec Section 3); a
    /// silent material drift is a hard failure, an annotated one is allowed.
    pub drift_rationale: Option<String>,
}
```

Add `use crate::coverage::verdict::{Completeness, Verdict, Verification};` to the imports if not already covering these (the file already imports `Verdict`; add `Completeness, Verification`).

- [ ] **Step 3: Rewrite `capability_spine()` with the 20-domain skeleton seed**

Replace the `pub fn capability_spine()` body. Extend its doc comment to document the four new folds (append to the existing `///` folds list, before the `pub fn`):

```rust
///   - `control_packets` covers on-chip control-packet reassembly, packet
///     handler status, and the NPU host instruction stream (spec Appendix)
///   - `clock_control` covers module/column/tile clock + reset control
///   - `tile_isolation` covers Tile_Control isolation bits / N-S-E-W gates
///   - `binary_load` covers CDO/ELF/XCLBIN ingest, SS-routing reconstruction,
///     and array-topology construction (spec Appendix N1 resolutions)
```

Body:

```rust
pub fn capability_spine() -> Vec<CapabilityDomain> {
    let aie2 = vec![Architecture::Aie2];
    // Task-2 skeleton seed: every domain is honestly Partial pending its
    // curated seed (Tasks 3/4). source_ref/src_locations are non-empty so the
    // enforce loop passes; the Partial{missing} verdict is true (the catalogue
    // entry is genuinely incomplete) and self-corrects out of the
    // implementation-gaps queue as Tasks 3/4 replace it. NOT a placeholder in
    // the plan-failure sense -- it is an accurate coarse bootstrap (the same
    // honest-coarse pattern as Phase-1 category defaults).
    let pending = |missing: &str| Verdict {
        provenance: crate::coverage::verdict::Provenance::Unspecified,
        verification: Verification::Modeled {
            completeness: Completeness::Partial { missing: missing.to_string() },
        },
    };
    crate::coverage::spine_ids::SPINE_DOMAIN_IDS
        .iter()
        .map(|id| CapabilityDomain {
            id: (*id).to_string(),
            arches: aie2.clone(),
            source_ref: "pending domain seed (plan Task 3/4)".to_string(),
            src_locations: vec!["pending domain seed (plan Task 3/4)".to_string()],
            narrative: format!("{id}: curated seed pending (plan Task 3/4)"),
            verdict: pending("curated domain seed pending (plan Task 3/4)"),
            drift_rationale: None,
        })
        .collect()
}
```

Note: `Provenance::Unspecified + Modeled` would violate the Section-1 invariant — so the skeleton uses it deliberately *only* to force every domain into the implementation-gaps queue until seeded AND to make the Task-2 enforce test below assert the invariant fires. Replace `Provenance::Unspecified` with `Provenance::ToolchainDerived` in the `pending` closure (Unspecified+Modeled is rejected by the enforce loop you add in Step 5; the skeleton must pass enforcement). Final `pending` closure provenance: `Provenance::ToolchainDerived`.

- [ ] **Step 4: Update existing `units.rs` tests for the new fields**

In the `units.rs` `#[cfg(test)] mod tests`, the existing `capability_spine_seeded_for_aie2`, `capability_domain_arch_applicability_is_explicit`, and `capability_spine_matches_the_leaf_id_list` only read `.id`/`.applies_to`, so they still compile and pass. Add one assertion to `capability_spine_seeded_for_aie2` after the existing asserts:

```rust
        // Spine extended to 20 (spec Section 2 hybrid decision).
        assert_eq!(spine.len(), 20);
        for id in ["control_packets", "clock_control", "tile_isolation", "binary_load"] {
            assert!(spine.iter().any(|d| d.id == id), "missing new domain {id}");
        }
        // Every domain carries non-empty source_ref + src_locations.
        assert!(spine.iter().all(|d| !d.source_ref.is_empty()));
        assert!(spine.iter().all(|d| !d.src_locations.is_empty()));
```

- [ ] **Step 5: Add the per-domain validation loop to `enforce_coverage`**

In `crates/xdna-archspec/src/coverage/enforce.rs`, add `use crate::coverage::verdict::{Completeness, Provenance, Verification};` (the file imports `Verification`; add `Completeness, Provenance`). Append a new numbered block at the end of the `enforce_coverage` fn body (after block 3, before the closing brace):

```rust
    // 4. Per-domain seed validity (spec Section 2 N2). The spine argument now
    //    carries fully-seeded domains (capability_spine()); validate them.
    for dom in spine {
        if !dom.applies_to(arch) {
            continue;
        }
        let oos_or_missing = match &dom.verdict.verification {
            Verification::Accepted { .. } => true,
            Verification::Modeled { completeness } => {
                matches!(completeness, Completeness::Stub | Completeness::Partial { .. })
            }
            Verification::NotApplicable | Verification::Verified { .. } | Verification::Unverified => {
                false
            }
        };
        if dom.source_ref.trim().is_empty() {
            panic!(
                "COVERAGE: domain '{}' ({arch}) has empty source_ref -- every \
                 subsystem must name its authoritative source (spec Section 2)",
                dom.id
            );
        }
        if dom.src_locations.iter().all(|s| s.trim().is_empty()) && !oos_or_missing {
            panic!(
                "COVERAGE: domain '{}' ({arch}) has no src_locations and is not \
                 an explicit OOS/MISSING state -- name where it is implemented \
                 or mark it OOS/MISSING with a narrative (spec Section 2)",
                dom.id
            );
        }
        if matches!(dom.verdict.provenance, Provenance::Unspecified)
            && matches!(dom.verdict.verification, Verification::Modeled { .. })
        {
            panic!(
                "COVERAGE: domain '{}' ({arch}) is Unspecified + Modeled -- \
                 asserting no-model and a-model are contradictory (spec S1 \
                 invariant)",
                dom.id
            );
        }
    }
```

- [ ] **Step 6: Write the failing enforce tests**

In the `enforce.rs` `#[cfg(test)] mod tests`, add (the test module already imports `CapabilityDomain`; add a small builder):

```rust
    fn dom(id: &str, v: Verdict) -> CapabilityDomain {
        CapabilityDomain {
            id: id.into(),
            arches: vec![Architecture::Aie2],
            source_ref: "aie-rt".into(),
            src_locations: vec!["src/x".into()],
            narrative: "n".into(),
            verdict: v,
            drift_rationale: None,
        }
    }

    #[test]
    #[should_panic(expected = "empty source_ref")]
    fn enforce_rejects_empty_source_ref() {
        let mut d = dom("dma", Verdict {
            provenance: Provenance::ToolchainDerived,
            verification: Verification::NotApplicable,
        });
        d.source_ref = "  ".into();
        enforce_coverage(Architecture::Aie2, &[d.clone()], &[ok_unit(Architecture::Aie2, "dma")]);
    }

    #[test]
    #[should_panic(expected = "Unspecified + Modeled")]
    fn enforce_rejects_unspecified_modeled_domain() {
        let d = dom("dma", Verdict {
            provenance: Provenance::Unspecified,
            verification: Verification::Modeled { completeness: Completeness::Stub },
        });
        enforce_coverage(Architecture::Aie2, &[d.clone()], &[ok_unit(Architecture::Aie2, "dma")]);
    }

    #[test]
    fn enforce_allows_oos_missing_without_src_locations() {
        let mut d = dom("ecc", Verdict {
            provenance: Provenance::DocSpecified,
            verification: Verification::Accepted { rationale: "out of scope".into() },
        });
        d.src_locations = vec![];
        enforce_coverage(Architecture::Aie2, &[d.clone()], &[ok_unit(Architecture::Aie2, "ecc")]);
    }
```

Add `use crate::coverage::verdict::Completeness;` to the test module imports if not already present.

- [ ] **Step 7: Build, run tests**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib`
Expected: full crate builds; all `units::`, `enforce::`, and pre-existing coverage tests PASS (including `enforce_coverage_phase1`-driven tests, now exercising the seeded spine). `capability_spine_matches_the_leaf_id_list` still passes (compares `.id` only).

- [ ] **Step 8: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/spine_ids.rs crates/xdna-archspec/src/coverage/units.rs crates/xdna-archspec/src/coverage/enforce.rs
git commit -m "coverage: extend spine 16->20, enrich CapabilityDomain, add enforce loop

Adds control_packets/clock_control/tile_isolation/binary_load ids;
seeded source_ref/src_locations/narrative/verdict/drift_rationale
fields; enforce_coverage per-domain validity loop (source_ref,
src_locations vs OOS, Unspecified+Modeled). Skeleton seed is an honest
Partial-pending bootstrap; Tasks 3/4 curate it. Spec Section 2.

Generated using Claude Code."
```

---

### Task 3: Seed the 16 original hardware-subsystem domains

Implements spec Section 5 + Appendix for the original 16 ids. Replaces each domain's skeleton `source_ref`/`src_locations`/`narrative`/`verdict` with curated values folded from the retired index. Read it first: `git show 1afdb20^:docs/coverage/architecture-index.md`.

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (`capability_spine`)

- [ ] **Step 1: Read the source-of-truth content**

Run: `cd /home/triple/npu-work/xdna-emu && git show 1afdb20^:docs/coverage/architecture-index.md`
Use its per-row Source / Coverage / "Our location" / Notes cells as the seed material, folded per the spec Appendix table.

- [ ] **Step 2: Replace the skeleton map with explicit per-domain seeds (16 originals)**

> **Plan correction (applied during execution, commit `a4efe52`):** the
> code block below originally listed only 15 `d(...)` entries — `debug_halt`
> was omitted in error. The implementer correctly added it, folded from the
> retired index's "Core debug (halt/step/breakpoint)" row per the spec
> Appendix: `d("debug_halt", "AM025 Debug_*; aie-rt core/",
> &["src/device/core_debug/"], "Halt + status bits modeled. Programmable
> breakpoints and single-step PC trap not wired through interpreter.",
> Modeled { completeness: Partial { missing: "programmable breakpoints +
> single-step PC trap".into() } })`. Additionally, entries are emitted in
> `SPINE_DOMAIN_IDS` order (not the order written below) because
> `capability_spine_matches_the_leaf_id_list` asserts ordered equality.

Rewrite `capability_spine()` to construct each of the 16 original domains explicitly (keep the four new ids on the skeleton `pending` path for now — Task 4 curates them). Use a helper to cut boilerplate. Verdict mapping rule from the retired "Coverage" column: `MODELED`→`Modeled{Full}`; `PARTIAL`→`Modeled{Partial{missing:"…"}}`; `STUBBED`→`Modeled{Stub}`; `MISSING`→`Modeled{Stub}` with a MISSING narrative; `OUT_OF_SCOPE`→`Accepted{rationale:"out of scope: …"}`. Provenance is `ToolchainDerived` for aie-rt/AM025-sourced subsystems.

```rust
pub fn capability_spine() -> Vec<CapabilityDomain> {
    use crate::coverage::verdict::{Completeness::*, Provenance::*, Verification::*};
    let aie2 = || vec![Architecture::Aie2];
    let full = || Modeled { completeness: Full };
    let d = |id: &str, source_ref: &str, locs: &[&str], narrative: &str, v: Verification| {
        CapabilityDomain {
            id: id.into(),
            arches: aie2(),
            source_ref: source_ref.into(),
            src_locations: locs.iter().map(|s| s.to_string()).collect(),
            narrative: narrative.into(),
            verdict: Verdict { provenance: ToolchainDerived, verification: v },
            drift_rationale: None,
        }
    };
    // Four new domains stay on the honest skeleton until Task 4.
    let pending = |id: &str| CapabilityDomain {
        id: id.into(),
        arches: aie2(),
        source_ref: "pending domain seed (plan Task 4)".into(),
        src_locations: vec!["pending domain seed (plan Task 4)".into()],
        narrative: format!("{id}: curated seed pending (plan Task 4)"),
        verdict: Verdict {
            provenance: ToolchainDerived,
            verification: Modeled { completeness: Partial { missing: "curated seed pending (plan Task 4)".into() } },
        },
        drift_rationale: None,
    };
    vec![
        d("core", "aie-rt core/, llvm-aie TableGen; AM025 Core_Control/Core_Status/Error_Halt_*",
          &["src/interpreter/", "src/device/core_debug/"],
          "VLIW core, control (enable/done/reset), and error-halt path. 100% ISA decode; SemanticOp coverage ~33%, rest in legacy handlers. Generic error_halt fires INSTR_ERROR (event 69) on every CoreStatus::Error; ECC fires ECC_ERROR_STALL. Saturation/watchdog error sources not yet detected.",
          full()),
        d("program_memory", "AM025",
          &["src/parser/elf.rs"],
          "16KB program memory; ELF load -> run.", full()),
        d("program_counter", "AM025 PC_Event0..3 (0x32020/4/8/C); aie-rt xaiemlgbl_params.h",
          &["src/interpreter/", "src/device/core_debug/"],
          "PC sampling + PC_Event0..3 / Core_PC_Range matching; drives event-halt selector.",
          full()),
        d("data_memory", "aie-rt memory/, AM025",
          &["src/device/banking.rs", "src/device/state/memtile.rs", "src/interpreter/timing/memory.rs"],
          "64KB compute (8 banks x 128-bit) / 512KB memtile; conflict detection done; per-bank MEM_CONFLICT_DM_BANK_N fired. ECC: status bit readable, no scrubber/fault-injection -- accepted out of scope unless workloads require.",
          full()),
        d("watchpoint", "AM025 Compute WatchPoint0/1 (2), MemTile WatchPoint0..3 (4)",
          &["src/interpreter/execute/cycle_accurate.rs"],
          "Compute 2 / memtile 4 slots; WriteStrobes==0xF gate, direction + address comparator, AXI/DMA/quadrant origin filters, scalar+vector+DMA-engine paths, modifier-register effective address. Locked by 17+ unit tests.",
          full()),
        d("dma", "aie-rt dma/, AM025 (112/433/144 reg)",
          &["src/device/dma/"],
          "BDs per tile-type 16/48/16 (memtile 48 not 64 -- aie-rt xaiemlgbl_reginit.c), channels 2/6/2, address dims 3/4/3; n-d addressing, padding, lock coupling, packet header, compression. Repeat / out-of-order BD execution: verify.",
          full()),
        d("locks", "aie-rt locks/, AM025",
          &["src/device/tile/locks.rs"],
          "Counts per tile-type 16/64/16 (aie-rt xaiemlgbl_reginit.c; the 192 in AieMlMemTileDmaMod.NumLocks is the cross-tile reference range, not slot count). acquire/release/get/set, semaphore semantics, round-robin arbiter.",
          full()),
        d("stream_switch", "aie-rt stream_switch/, AM025 (160/119/149 reg)",
          &["src/device/stream_switch/"],
          "Circuit + packet, FIFOs, port events, packet-header matching. Parse-time routing reconstruction is a binary_load concern (spec Appendix N1); see binary_load narrative.",
          full()),
        d("shim_mux", "aie-rt, AM025 (shim Mux/Demux 2 reg); aie-rt pl/ (PL Interface)",
          &["src/device/stream_switch/"],
          "Shim master/slave NoC-facing mux/demux. PL Interface (Upsizer/Downsizer) is Versal-FPGA stream-width adaptation -- NPU1 exposes no programmable PL: accepted out of scope.",
          full()),
        d("cascade", "aie-rt, aietools events",
          &["src/interpreter/execute/cascade.rs"],
          "Tile<->tile cascade read/write. Deadlock detection is a placeholder (deadlock.rs) -- promote to real detection or remove (verification follow-up).",
          full()),
        d("events_trace", "aie-rt events/ + trace/, AM025 (128/161/51 events)",
          &["src/device/events/", "src/device/trace_unit/"],
          "Events (broadcast 16ch, combo, group, port), cross-tile broadcast network, trace unit modes 0/1/2, pipelined start/stop + multi-tile timer sync. Combo/edge generator boundary cases need targeted tests; L2 broadcast propagation verify.",
          full()),
        d("performance_counters", "aie-rt perfcnt/, AM025 (compute/memtile/shim 4/11/6 reg)",
          &["src/device/perf_counters/"],
          "4 counters, threshold events. DMA/stream FIFO-size events not emitted (cycle-accuracy gap, tracked in cycle-accuracy-mission.md).",
          full()),
        d("timer", "aie-rt timer/, AM025 (5 reg)",
          &["src/device/timer.rs"],
          "Free-running 64-bit per-module; Reset_Event consumed via pending_reset latch; multi-tile timer-sync modeled. Trig_Event_Low/High_Value write effect: verification follow-up.",
          full()),
        d("interrupt", "aie-rt interrupt/, AM025 (L2 shim_intc_l2 23 reg)",
          &["src/device/interrupts/l1.rs", "src/device/interrupts/l2.rs"],
          "L1 per-tile 20 IRQs mask/enable/status MODELED. L2 NoC aggregator present but 23-reg surface exhaustiveness unconfirmed and privilege gating not modeled.",
          Modeled { completeness: Partial { missing: "L2 23-reg surface exhaustiveness + privilege gating".into() } }),
        d("noc", "AM025 (NoC_Interface_AIE_to_NoC 4 reg, AIE_AXIMM_Config); aie-rt npi/; hardware spec",
          &[],
          "Direct NoC control / AIE_AXIMM_Config / NoC fabric latency-arbitration are not modeled (NoC fudged; impacts cycle-accuracy more than functional correctness; cycle-accuracy-mission.md tracks calibration). NPI privileged register access is driver-side privilege -- emulator gives unrestricted access: accepted out of scope. No emulator src for the unmodeled NoC surface.",
          Modeled { completeness: Stub }),
        // 4 new domains: skeleton until Task 4.
        pending("control_packets"),
        pending("clock_control"),
        pending("tile_isolation"),
        pending("binary_load"),
    ]
}
```

(`noc` legitimately has empty `src_locations` because the NoC surface is unmodeled — the enforce loop permits this because its verdict is `Modeled{Stub}`, an explicit MISSING state, and the narrative says so.)

- [ ] **Step 3: Build and run the full coverage test suite**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib`
Expected: builds; all tests PASS. The enforce loop accepts every seeded domain (every `source_ref` non-empty; `noc` empty `src_locations` allowed by its `Modeled{Stub}` state; no `Unspecified+Modeled`).

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/units.rs
git commit -m "coverage: seed the 16 original hardware-subsystem domains

Folds the retired index's per-subsystem Source / location / coverage /
known-gaps cells onto the 16 original spine domains per spec Appendix.
AIE-ML constants (locks 16/64/16, DMA BD 16/48/16) folded into
locks/dma narratives. Spec Section 5.

Generated using Claude Code."
```

---

### Task 4: Seed the 4 new domains + OOS/driver dispositions

Implements spec Section 5 + Appendix N1 resolutions for `control_packets`, `clock_control`, `tile_isolation`, `binary_load`.

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (`capability_spine`)

- [ ] **Step 1: Replace the four `pending(...)` entries with curated seeds**

In `capability_spine()`, replace the four `pending("…")` lines with explicit `d(...)` constructions:

```rust
        d("control_packets", "AM025 Control_Packet_Handler_Status (0x3FF30/0xB0F30); XRT host protocol",
          &["src/device/control_packets/", "src/device/tile/mod.rs", "src/npu/"],
          "Keystone subsystem the retired index exists because we missed it. Control-packet headers, reassembly, register read/write effects, response packets MODELED. Packet handler status sticky bits + write-1-to-clear; Second_Header_Parity_Error wired, Tlast/SLVERR/ID_Parity not (no detecting path). NPU host instruction stream (WRITE32/BLOCKWRITE/BLOCKSET/MASKWRITE/MASKPOLL/CONFIG_SHIMDMA_*/DDR_PATCH) MODELED.",
          Modeled { completeness: Partial { missing: "Tlast/SLVERR/ID_Parity packet-handler sticky bits".into() } }),
        d("clock_control", "AM025 Module_Clock_Control / Column_Clock_Control / Reset_Control_1 / AIE_Tile_Column_Reset",
          &[],
          "Module/column/tile clock-gating writes accepted but no effect on cycle counts / power model. Tile column reset (partition teardown) and multi-clock-domain semantics not simulated. Functionally inert today -- MISSING; relevant once partition lifecycle / power modeling is built.",
          Modeled { completeness: Stub }),
        d("tile_isolation", "aie-rt pm/xaie_tilectrl.c, AM025 (Tile_Control compute 0x36030 / memtile 0x96030)",
          &["src/device/tile/mod.rs", "src/device/state/effects.rs", "src/device/array/routing.rs", "src/interpreter/execute/memory/neighbor.rs", "src/interpreter/engine/coordinator.rs"],
          "Tile_Control low 4 bits (S/W/N/E) snapshotted on register write. Inter-tile stream transfers, cross-tile NeighborMemory snapshots/reads/buffered writes, and NeighborLocks slices all consult the destination/own isolation byte. Shim isolation snapshotted; only memtile->shim south-bound routing gate consults it today. Clock-gating bits of Tile_Control pass through unmodeled (see clock_control).",
          Modeled { completeness: Full }),
        d("binary_load", "XRT container / CDO / ELF formats; mlir-aie device model (tools/aie-device-models.json)",
          &["src/parser/xclbin.rs", "src/parser/cdo/", "src/parser/elf.rs", "src/parser/stream_switch_topology.rs", "src/device/array/"],
          "XCLBIN container, CDO framing/syntax/semantics -> DeviceOps, per-core ELF load, all MODELED. Stream-switch routing reconstruction from CDO writes (parse-side, distinct from the runtime stream_switch subsystem -- spec Appendix N1). Tile array topology (5x6 NPU1) constructed from the device model at load: folded here as the array-constructed-from-binary concern (spec Appendix N1 rationale), not a reachability-forced tag.",
          Modeled { completeness: Full }),
```

Add a one-line cross-reference into the `stream_switch` narrative (Task-3 text) by appending to its narrative string: ` See binary_load for parse-time routing reconstruction.` (it already references it; ensure the wording matches).

- [ ] **Step 2: Build and run the full coverage suite**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib`
Expected: builds; all tests PASS. No domain is on the `pending` skeleton. `clock_control`/`noc` empty `src_locations` allowed (both `Modeled{Stub}` MISSING).

- [ ] **Step 3: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/units.rs
git commit -m "coverage: seed control_packets/clock_control/tile_isolation/binary_load

Curates the four hybrid-extension domains + the Appendix N1 resolutions
(SS-routing-reconstruction parse-side ownership; array-topology
documented fold). Spec Section 5 / Appendix.

Generated using Claude Code."
```

---

### Task 5: Category→domain link, rollup, directional drift

Implements spec Section 3 + the category-orphan model (Section 2) + the lattice. New focused module so units.rs stays data-only and artifacts.rs stays render-only.

**Files:**
- Create: `crates/xdna-archspec/src/coverage/subsystem.rs`
- Modify: `crates/xdna-archspec/src/coverage/mod.rs` (add `pub mod subsystem;`)

- [ ] **Step 1: Write the failing tests first**

Create `crates/xdna-archspec/src/coverage/subsystem.rs` containing ONLY the test module to start:

```rust
//! Subsystem-axis weave (spec Section 3): the coarse category->domain link,
//! the worst-wins coverage-strength lattice, and the directional drift
//! cross-check. The link is single-sourced on the category side; the
//! domain->category direction is computed here, never stored.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::derive::Category;
    use crate::coverage::verdict::{Completeness, Provenance, Verdict, Verification};
    use crate::types::Architecture;

    fn v(p: Provenance, ver: Verification) -> Verdict {
        Verdict { provenance: p, verification: ver }
    }

    #[test]
    fn lattice_total_order_is_pinned() {
        // Spec Section 3 binding shape: closed (Verified==NotApplicable==
        // Accepted) > Modeled{Full} > Modeled{Partial} > Modeled{Stub} >
        // Unverified; provenance tie-break Unspecified weakest.
        let verified = v(Provenance::AietoolsModeled, Verification::Verified { evidence: "e".into() });
        let na = v(Provenance::ToolchainDerived, Verification::NotApplicable);
        let accepted = v(Provenance::DocSpecified, Verification::Accepted { rationale: "r".into() });
        let full = v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Full });
        let partial = v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Partial { missing: "m".into() } });
        let stub = v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Stub });
        let unver = v(Provenance::DocSpecified, Verification::Unverified);
        let unspec = v(Provenance::Unspecified, Verification::Unverified);

        assert_eq!(coverage_rank(&verified), coverage_rank(&na));
        assert_eq!(coverage_rank(&na), coverage_rank(&accepted));
        assert!(coverage_rank(&accepted) > coverage_rank(&full));
        assert!(coverage_rank(&full) > coverage_rank(&partial));
        assert!(coverage_rank(&partial) > coverage_rank(&stub));
        assert!(coverage_rank(&stub) > coverage_rank(&unver));
        assert!(coverage_rank(&unver) > coverage_rank(&unspec));
    }

    #[test]
    fn every_category_tags_at_least_one_domain() {
        for cat in [
            Category::Arithmetic, Category::Bitwise, Category::Comparison,
            Category::Memory, Category::ControlFlow, Category::Vector,
            Category::Sync, Category::SideEffect, Category::NeedsTriage,
        ] {
            assert!(!category_domains(cat).is_empty(), "category {cat:?} tags no domain");
        }
    }

    #[test]
    fn every_domain_is_tagged_or_explicitly_orphan() {
        use crate::coverage::units::capability_spine;
        for d in capability_spine() {
            let tagged = tagged_categories(&d.id).next().is_some();
            assert!(
                tagged || is_category_orphan(&d.id),
                "domain '{}' is neither category-tagged nor an explicit orphan \
                 (no fabricated tags allowed -- spec Section 2)",
                d.id
            );
        }
    }

    #[test]
    fn drift_is_directional_optimistic_only() {
        // Domain claims MORE than its rollup -> material (must flag).
        let optimistic = drift_is_material(
            &v(Provenance::ToolchainDerived, Verification::Modeled { completeness: Completeness::Full }),
            &v(Provenance::DocSpecified, Verification::Unverified),
        );
        assert!(optimistic, "optimistic over-claim must be material");
        // Domain claims LESS than its rollup -> safe over-report (must not).
        let pessimistic = drift_is_material(
            &v(Provenance::DocSpecified, Verification::Unverified),
            &v(Provenance::ToolchainDerived, Verification::NotApplicable),
        );
        assert!(!pessimistic, "pessimistic divergence is safe over-reporting");
        // Equal rank -> not material.
        assert!(!drift_is_material(
            &v(Provenance::ToolchainDerived, Verification::NotApplicable),
            &v(Provenance::AietoolsModeled, Verification::Verified { evidence: "e".into() }),
        ));
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /home/triple/npu-work/xdna-emu && cargo test -p xdna-archspec --lib subsystem::`
Expected: FAIL — `coverage_rank`, `category_domains`, `tagged_categories`, `is_category_orphan`, `drift_is_material` undefined. (You must also add `pub mod subsystem;` to `mod.rs` alphabetically — between `pub mod spine_ids;` and `pub mod surface;` — for the module to compile at all; do that now and re-run to get the "undefined fn" failures.)

- [ ] **Step 3: Implement the module**

Prepend to `subsystem.rs` (above the test module):

```rust
use crate::coverage::derive::Category;
use crate::coverage::units::capability_spine;
use crate::coverage::verdict::{Completeness, Provenance, Verdict, Verification};

/// Coarse category->domain link (spec Section 2): the single source of truth,
/// stored only on the category side. 9 rows. Category-orphan domains
/// deliberately appear in no row (spec Section 2 -- no fabricated tags).
pub fn category_domains(cat: Category) -> &'static [&'static str] {
    match cat {
        Category::Arithmetic | Category::Bitwise | Category::Comparison
        | Category::ControlFlow => &["core"],
        Category::Memory => &["data_memory", "program_memory"],
        Category::Vector => &["core"],
        Category::Sync => &["locks"],
        Category::SideEffect => &["dma", "stream_switch", "cascade"],
        Category::NeedsTriage => &["core"],
    }
}

/// Domains intentionally reachable from NO category (spec Section 2). Their
/// coverage is solely their own seeded verdict; no rollup, no drift check.
pub fn is_category_orphan(domain_id: &str) -> bool {
    !ALL_CATEGORIES.iter().any(|c| category_domains(*c).contains(&domain_id))
}

const ALL_CATEGORIES: [Category; 9] = [
    Category::Arithmetic, Category::Bitwise, Category::Comparison,
    Category::Memory, Category::ControlFlow, Category::Vector,
    Category::Sync, Category::SideEffect, Category::NeedsTriage,
];

/// Inverse index, computed (never stored): categories tagged to `domain_id`.
pub fn tagged_categories(domain_id: &str) -> impl Iterator<Item = Category> + '_ {
    ALL_CATEGORIES
        .into_iter()
        .filter(move |c| category_domains(*c).contains(&domain_id))
}

/// Coverage-strength rank (spec Section 3 lattice). Higher == more covered.
/// Closed/terminal states tie at the top; provenance tie-breaks toward weaker.
pub fn coverage_rank(v: &Verdict) -> u32 {
    let base = match &v.verification {
        Verification::Verified { .. }
        | Verification::NotApplicable
        | Verification::Accepted { .. } => 50,
        Verification::Modeled { completeness: Completeness::Full } => 40,
        Verification::Modeled { completeness: Completeness::Partial { .. } } => 30,
        Verification::Modeled { completeness: Completeness::Stub } => 20,
        Verification::Unverified => 10,
    };
    // Provenance tie-break: Unspecified is weakest, shaving 1 so an
    // Unspecified/Unverified ranks strictly below any other Unverified.
    let tiebreak = match v.provenance {
        Provenance::Unspecified => 0,
        _ => 1,
    };
    base + tiebreak
}

/// Worst-wins rollup of the categories tagged to `domain_id` (spec Section 3).
/// `None` == category-orphan (no rollup; row shows `-`, drift exempt).
pub fn rolled_up_verdict(domain_id: &str) -> Option<Verdict> {
    use crate::coverage::CoverageModel;
    use crate::types::Architecture;
    let m = CoverageModel::build(Architecture::Aie2);
    let cats: Vec<Category> = tagged_categories(domain_id).collect();
    if cats.is_empty() {
        return None;
    }
    // Worst-wins: the weakest tagged category's representative verdict.
    cats.into_iter()
        .map(|c| m.semantic_verdict(&crate::coverage::subsystem::category_rep(c)))
        .min_by_key(|v| coverage_rank(v))
}

/// One representative SemanticOp per category (mirrors artifacts.rs reps; the
/// rollup only needs each category's default verdict, which is rep-invariant).
fn category_rep(cat: Category) -> crate::aie2::isa::SemanticOp {
    use crate::aie2::isa::SemanticOp::*;
    match cat {
        Category::Arithmetic => Add,
        Category::Bitwise => And,
        Category::Comparison => SetLt,
        Category::Memory => Load,
        Category::ControlFlow => Br,
        Category::Vector => Mac,
        Category::Sync => LockAcquire,
        Category::SideEffect => DmaStart,
        Category::NeedsTriage => Intrinsic(0),
    }
}

/// Directional drift (spec Section 3): material ONLY when the domain's own
/// verdict ranks strictly higher (claims more coverage) than the rollup.
/// Pessimistic divergence is safe over-reporting and is NOT material.
pub fn drift_is_material(own: &Verdict, rolled_up: &Verdict) -> bool {
    coverage_rank(own) > coverage_rank(rolled_up)
}

/// A domain's drift status for rendering / the no-silent-drift test:
/// `Some(true)` material+unannotated (hard fail), `Some(false)`
/// material+annotated (allowed), `None` not material or orphan.
pub fn unannotated_material_drift(domain_id: &str) -> bool {
    let Some(rolled) = rolled_up_verdict(domain_id) else { return false };
    let Some(dom) = capability_spine().into_iter().find(|d| d.id == domain_id) else {
        return false;
    };
    drift_is_material(&dom.verdict, &rolled) && dom.drift_rationale.is_none()
}
```

Note: `category_rep` is private; the `rolled_up_verdict` body calls it as `crate::coverage::subsystem::category_rep` — change that to plain `category_rep(c)` (same module). Keep one definition.

- [ ] **Step 4: Add the no-silent-drift gate test**

Append to the `subsystem.rs` test module:

```rust
    #[test]
    fn no_subsystem_drifts_silently() {
        // Spec Section 3: a material optimistic drift is allowed ONLY with a
        // drift_rationale. Any unannotated material drift is a hard failure.
        use crate::coverage::units::capability_spine;
        for d in capability_spine() {
            assert!(
                !unannotated_material_drift(&d.id),
                "domain '{}' claims more coverage than its tagged categories \
                 justify with no drift_rationale (spec Section 3)",
                d.id
            );
        }
    }
```

- [ ] **Step 5: Run tests; resolve any real drift**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib subsystem::`
Expected: all `subsystem::` tests PASS. If `no_subsystem_drifts_silently` fails for a domain, that is a real finding: the domain's seeded verdict (Task 3/4) over-claims vs its tagged categories. Resolve by EITHER correcting the domain's verdict in `units.rs` to be honest, OR (if the divergence is deliberate and defensible) setting that domain's `drift_rationale: Some("…why the subsystem is genuinely more covered than its op-level categories…".into())` in `capability_spine()`. Do not weaken the test. Re-run; if `units.rs` changed, the Task-3/4 commits are already in — add a follow-up commit in Step 6.

- [ ] **Step 6: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/subsystem.rs crates/xdna-archspec/src/coverage/mod.rs crates/xdna-archspec/src/coverage/units.rs
git commit -m "coverage: category->domain link, worst-wins rollup, directional drift

New coverage::subsystem module: single-sourced category_domains (9
rows), computed inverse index, explicit category-orphan set, pinned
coverage-strength lattice, optimistic-only material-drift check + the
no_subsystem_drifts_silently gate. Spec Section 2/3.

Generated using Claude Code."
```

---

### Task 6: Generated artifacts

Implements spec Section 4. Adds `render_subsystem_index` + `render_implementation_gaps`, re-weaves `render_architecture_index`, wires the generator, commits regenerated docs (spec Section 8 ordering rule).

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/artifacts.rs`
- Modify: `crates/xdna-archspec/examples/gen_coverage_artifacts.rs`
- Create: `docs/coverage/aie2/subsystem-index.md` (generated)
- Create: `docs/coverage/aie2/implementation-gaps.md` (generated)
- Modify: `docs/coverage/aie2/architecture-index.md` (regenerated, schema change)

- [ ] **Step 1: Add the two renderers + re-weave the index renderer**

In `artifacts.rs`, add imports: `use crate::coverage::units::capability_spine; use crate::coverage::subsystem::{rolled_up_verdict, is_category_orphan, tagged_categories, unannotated_material_drift};`.

Add `render_subsystem_index`:

```rust
/// Render the per-subsystem coverage index (spec Section 4): the retired
/// hand-maintained index's per-subsystem rows, regenerated.
pub fn render_subsystem_index(arch: Architecture) -> String {
    let mut lines = vec![
        format!("# {arch} subsystem coverage index"),
        String::new(),
        "Generated by `cargo run -p xdna-archspec --example gen_coverage_artifacts`. Do not hand-edit."
            .to_string(),
        String::new(),
        "Per-subsystem rows recovered from the retired hand-maintained index as".to_string(),
        "generated data (spec 2026-05-16-subsystem-axis-coverage-enrichment).".to_string(),
        "`rolled-up` is the worst-wins verdict of the categories tagged to the".to_string(),
        "domain, or `-` for category-orphan domains. Sibling queues:".to_string(),
        "implementation-gaps.md / perishable-queue.md / comprehension-gaps.md.".to_string(),
        String::new(),
        "| Domain | Source | Our location | Own verdict | Rolled-up | Drift | Narrative |".to_string(),
        "|--------|--------|--------------|-------------|-----------|-------|-----------|".to_string(),
    ];
    for d in capability_spine().into_iter().filter(|d| d.applies_to(arch)) {
        let rolled = match rolled_up_verdict(&d.id) {
            Some(v) => format!("{:?}/{:?}", v.provenance, v.verification),
            None => "-".to_string(),
        };
        let drift = if is_category_orphan(&d.id) {
            "orphan".to_string()
        } else if unannotated_material_drift(&d.id) {
            "MATERIAL".to_string()
        } else if d.drift_rationale.is_some() {
            "annotated".to_string()
        } else {
            "ok".to_string()
        };
        let locs = if d.src_locations.is_empty() { "-".to_string() } else { d.src_locations.join(", ") };
        lines.push(format!(
            "| {} | {} | {} | {:?}/{:?} | {} | {} | {} |",
            d.id, d.source_ref, locs, d.verdict.provenance, d.verdict.verification, rolled, drift,
            d.narrative.replace('|', "\\|").replace('\n', " ")
        ));
    }
    lines.push(String::new());
    format!("{}\n", lines.join("\n"))
}
```

Add `render_implementation_gaps` (iterates DOMAINS, never the semantic universe — spec S1/M1):

```rust
/// Render the implementation-gap queue (spec Section 4): domains whose own
/// seeded verdict is a `Modeled{Partial|Stub}` (or explicit MISSING). Sourced
/// from `capability_spine()` domain verdicts -- NEVER a category rollup, which
/// can never be `Modeled` (a permanently empty queue otherwise -- spec S1/M1).
pub fn render_implementation_gaps(arch: Architecture) -> String {
    let mut lines = vec![
        format!("# Implementation gaps ({arch})"),
        String::new(),
        "Generated by `cargo run -p xdna-archspec --example gen_coverage_artifacts`. Do not hand-edit."
            .to_string(),
        String::new(),
        "Subsystems only partially built or stubbed (spec Section 1, third gap".to_string(),
        "class). Empty == every named subsystem is Full/closed for this arch.".to_string(),
        String::new(),
    ];
    let gaps: Vec<_> = capability_spine()
        .into_iter()
        .filter(|d| d.applies_to(arch) && d.verdict.is_implementation_gap())
        .collect();
    if gaps.is_empty() {
        lines.push("_empty_".to_string());
    } else {
        for d in gaps {
            let what = match &d.verdict.verification {
                Verification::Modeled { completeness: Completeness::Partial { missing } } => {
                    format!("PARTIAL -- missing: {missing}")
                }
                Verification::Modeled { completeness: Completeness::Stub } => "STUB".to_string(),
                Verification::Modeled { completeness: Completeness::Full } => unreachable!(),
                Verification::NotApplicable
                | Verification::Verified { .. }
                | Verification::Unverified
                | Verification::Accepted { .. } => unreachable!(),
            };
            lines.push(format!("- {}: {what}", d.id));
        }
    }
    lines.push(String::new());
    format!("{}\n", lines.join("\n"))
}
```

Add `use crate::coverage::verdict::{Completeness, Verification};` to artifacts.rs imports (the inner `Completeness`/`Verification` match must be exhaustive with NO `_` — spec Section 1 N4 task constraint; the explicit unreachable arms above satisfy it).

Re-weave `render_architecture_index`: change the table header line and the row `format!` to add a Domains column. Replace the header push:

```rust
        "| Category | Provenance | Verification | Domains |".to_string(),
        "|----------|------------|--------------|---------|".to_string(),
```

and the in-loop row push:

```rust
        let doms: Vec<String> = tagged_categories_for_render(*cat);
        lines.push(format!(
            "| {cat:?} | {:?} | {:?} | {} |",
            v.provenance, v.verification,
            if doms.is_empty() { "-".to_string() } else { doms.join(", ") }
        ));
```

where `tagged_categories_for_render` is a tiny local helper added near the top of `artifacts.rs`:

```rust
fn tagged_categories_for_render(cat: Category) -> Vec<String> {
    crate::coverage::subsystem::category_domains(cat).iter().map(|s| s.to_string()).collect()
}
```

Add a header preamble line to `render_architecture_index` (insert into the `lines` vec after the existing "sibling perishable-queue.md / comprehension-gaps.md." line):

```rust
        "Each category lists the spine domains it tags; per-subsystem detail".to_string(),
        "is in the generated subsystem-index.md.".to_string(),
```

- [ ] **Step 2: Update the staleness tests and the reps test**

In the `artifacts.rs` test module add two staleness tests mirroring `architecture_index_is_not_stale`:

```rust
    #[test]
    fn subsystem_index_is_not_stale() {
        let want = render_subsystem_index(Architecture::Aie2);
        let path = repo_path("docs/coverage/aie2/subsystem-index.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(got, want, "{} is stale -- regenerate: `cargo run -p xdna-archspec --example gen_coverage_artifacts` then git add docs/coverage/ and commit", path.display());
    }

    #[test]
    fn implementation_gaps_is_not_stale() {
        let want = render_implementation_gaps(Architecture::Aie2);
        let path = repo_path("docs/coverage/aie2/implementation-gaps.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(got, want, "{} is stale -- regenerate: `cargo run -p xdna-archspec --example gen_coverage_artifacts` then git add docs/coverage/ and commit", path.display());
    }

    #[test]
    fn implementation_gaps_source_is_the_spine_not_semantic() {
        // Spec S1/M1 guard: render must iterate capability_spine() domains. No
        // category default is ever Modeled, so a semantic-universe source
        // would be a permanently empty queue. With the Task-2/3/4 seeds at
        // least one domain (e.g. interrupt/clock_control/noc) is a gap.
        let out = render_implementation_gaps(Architecture::Aie2);
        assert!(!out.contains("_empty_"), "implementation-gaps queue is empty -- generator likely sourced the semantic universe, not the spine (spec S1/M1)");
    }
```

`architecture_index_reps_match_category` is unaffected (reps unchanged); leave it. The in-fn `debug_assert_eq!` on reps stays.

- [ ] **Step 3: Wire the generator**

In `crates/xdna-archspec/examples/gen_coverage_artifacts.rs`, extend the `use` to include `render_subsystem_index, render_implementation_gaps`, add two writes, and update the eprintln:

```rust
    std::fs::write(dir.join("subsystem-index.md"), render_subsystem_index(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("implementation-gaps.md"), render_implementation_gaps(Architecture::Aie2)).unwrap();
```
```rust
    eprintln!("wrote docs/coverage/aie2/{{perishable-queue,comprehension-gaps,architecture-index,subsystem-index,implementation-gaps}}.md");
```

- [ ] **Step 4: Generate the artifacts, then run the suite**

Run: `cd /home/triple/npu-work/xdna-emu && cargo run -p xdna-archspec --example gen_coverage_artifacts`
Expected: writes 5 files including the 2 new ones and a re-woven `architecture-index.md`.

Run: `cd /home/triple/npu-work/xdna-emu && cargo build -p xdna-archspec && TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib`
Expected: ALL tests pass, including the 3 new artifact tests and the now-regenerated `architecture_index_is_not_stale` (it changed schema — the regen in this step is what makes it pass; that is the spec Section 8 ordering rule in action).

- [ ] **Step 5: Commit (code + regenerated docs together — spec S8 rule)**

```bash
cd /home/triple/npu-work/xdna-emu
git add crates/xdna-archspec/src/coverage/artifacts.rs crates/xdna-archspec/examples/gen_coverage_artifacts.rs docs/coverage/aie2/
git commit -m "coverage: generate subsystem-index + implementation-gaps, reweave index

render_subsystem_index / render_implementation_gaps (domain-sourced,
exhaustive no-underscore matches); architecture-index gains a Domains
column + subsystem-index link; staleness gates + the M1 silent-empty
guard; generator + regenerated docs committed together (spec S8).

Generated using Claude Code."
```

---

### Task 7: Inbound re-linking + audit-checklist prose

Implements spec Section 5 inbound-relinking. No code; doc edits + a final full-suite check.

**Files:**
- Modify: `docs/coverage/audit-checklist.md`
- Modify: `docs/coverage/cycle-accuracy-mission.md`
- Modify: `docs/README.md` (and any other file found in Step 1)

- [ ] **Step 1: Find every inbound reference to the old/category index**

Run: `cd /home/triple/npu-work/xdna-emu && grep -rn "architecture-index" docs/ --include=*.md`
For each hit decide: a reference about *subsystem* detail repoints to `aie2/subsystem-index.md`; a reference genuinely about the *category* matrix stays on `aie2/architecture-index.md`. List the decisions before editing.

- [ ] **Step 2: Repoint subsystem-detail references**

Edit each file from Step 1 per the decision. The `audit-checklist.md` references to grading "a row" in the index are subsystem-level → repoint to `aie2/subsystem-index.md`. `cycle-accuracy-mission.md` and `docs/README.md` subsystem-catalogue references → `aie2/subsystem-index.md`.

- [ ] **Step 3: Extend the audit-checklist workflow prose**

In `docs/coverage/audit-checklist.md`, extend the "## How to use this checklist" section so verdict flips also feed the **implementation-gaps** queue and the **subsystem axis**, not only perishable/comprehension. Add this paragraph at the end of that section:

```markdown
A subsystem that is only partially built or stubbed lands in the
generated [aie2/implementation-gaps.md](aie2/implementation-gaps.md) by
its `CapabilityDomain` verdict (`Modeled{Partial|Stub}`), the same way a
weak-provenance op lands in the perishable queue. Closing it means
raising that domain's seeded verdict in `capability_spine()` to
`Modeled{Full}` (or a closed state) with honest evidence -- the
subsystem-index and implementation-gaps docs then regenerate to match.
```

- [ ] **Step 4: Full repo test suite (no regressions)**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: whole-workspace build clean (live build.rs spine gate runs with the 20-id leaf); all `--lib` tests pass; no regression vs the pre-plan baseline. Docs-only task so no staleness churn here.

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/
git commit -m "coverage: repoint subsystem refs to subsystem-index, extend audit prose

Inbound docs that wanted per-subsystem detail now point at the
generated subsystem-index.md; audit-checklist workflow extended for the
implementation-gaps queue + subsystem axis. Spec Section 5.

Generated using Claude Code."
```

---

## Self-Review

**1. Spec coverage:** Section 1 → Task 1 (+ enforce invariant Task 2). Section 2 struct/spine/N2 → Task 2; category-orphan → Task 5. Section 3 rollup/lattice/drift → Task 5. Section 4 artifacts → Task 6. Section 5 seed → Tasks 3-4; meta-section retirement → folded into Task 3/4 narratives (constants) + git-history-only (no task needed, by design); inbound re-linking → Task 7. Section 6 edge cases → covered by Task 1/2/5/6 tests. Section 7 testing → distributed across each task's test steps (predicate T1, invariant T2, lattice T5, staleness T6, drift T5, gaps-source T6, totality T5, enforce T2). Section 8 ordering rule → baked into Task 6 (regen+commit together) and the conventions block. Appendix fold map → Tasks 3-4. No gap.

**2. Placeholder scan:** The Task-2 skeleton uses literal `"pending domain seed (plan Task 3/4)"` strings — these are *intentional honest bootstrap data* fully replaced by Tasks 3/4 (called out explicitly), not plan-failure TBDs; every step has real code/commands. No "TODO/implement later/similar to Task N".

**3. Type consistency:** `Completeness`/`Verification::Modeled`/`is_implementation_gap` (T1) used identically in T2/T5/T6. `CapabilityDomain` field names (`source_ref`, `src_locations`, `narrative`, `verdict`, `drift_rationale`) consistent T2→T6. `coverage_rank`/`category_domains`/`tagged_categories`/`is_category_orphan`/`drift_is_material`/`rolled_up_verdict`/`unannotated_material_drift` defined in T5, consumed in T6 with matching signatures. One fix applied inline: `rolled_up_verdict` calls `category_rep(c)` (same-module private fn), not a `crate::coverage::subsystem::` path.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-subsystem-axis-coverage-enrichment.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec-compliance + code-quality) between tasks, controller adjudication, plan/spec kept in lockstep. This is the Plan-2 discipline that just caught a Critical at spec time.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.

Which approach?
