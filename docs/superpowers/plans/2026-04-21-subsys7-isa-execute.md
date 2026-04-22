# Subsystem 7 -- ISA Execute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the 20-file `src/interpreter/execute/` surface plus the 9-file `interpreter/timing/` submodule against AIE1/AIE2P divergence evidence, then introduce an `IsaExecutor` trait seam in `xdna-archspec` (method count audit-driven, expected 2-5 methods), migrate arch-specific data (VMAC crossbar routing, timing values, memory-hierarchy constants as audit finds) to archspec, update call sites, and tag `phase1-subsys-isa-execute`.

**Architecture:** Audit-first discipline per the design spec. Part A (this document) covers Tasks 1 and 2 -- the audit itself, plus a neutral scaffold: `xdna_archspec::isa_execute` module with an empty `IsaExecutor` trait, `Aie2IsaExecutor` ZST, `AIE2_ISA_EXECUTOR` singleton, `ArchConfig::isa_executor()` dispatch method, `arch_handle::isa_executor()` accessor, and a dispatch test. Part B (authored as a plan amendment after Task 1 commits) fills in the audit-driven work: data migrations (vmac_routing, timing, memory as applicable), trait-method additions with AIE2 impls, call-site migrations, and the completion+tag task. Keeping Part B deferred respects the spec's "trait surface is audit-driven" commitment -- we don't guess the method list, we read it from evidence.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, `&'static dyn IsaExecutor` trait-object dispatch via `OnceLock`, evidence sources for the audit = llvm-aie TableGen (`AIE1InstrInfo.td` vs `AIE2InstrInfo.td` where relevant), aie-rt per-arch headers (`driver/src/`), AM025 register DB JSON, existing codebase grep/count.

**Spec:** [docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md](../specs/2026-04-21-subsys7-isa-execute-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

**Prior subsystem:** `phase1-subsys-stream-switch` (Subsystem 5, 2026-04-22).

---

## Two-Phase Structure

Subsystem 7 is the largest Phase 1b subsystem by source volume. Rather
than speculatively write 8-10 concrete tasks off a trait shape we have
not yet confirmed against evidence, the plan is structured in two
phases matching Subsystem 6's Part A / Part B precedent:

- **Part A (this document as authored).** Task 1 produces the audit
  artifact. Task 2 lands the neutral scaffold (empty trait + concrete
  ZST + accessors + dispatch test). Both tasks can proceed without
  knowing the final trait method list.
- **Part B (authored as an amendment after Task 1 commits).** With
  the audit in hand, the amendment writes concrete tasks for: data
  migrations identified by the audit (expected: vmac_routing
  wholesale move, timing values, memory-hierarchy constants as
  applicable), one task per trait method added, call-site migrations,
  and the final completion + gate + tag task.

**Amendment protocol.** After Task 1 commits:

1. Re-enter `writing-plans` (or just edit this file directly if the
   changes are mechanical).
2. Read `docs/arch/subsys7-audit.md` in full.
3. Append Part B tasks to this file starting at `### Task 3: ...`
   (numbering continues from Part A).
4. Commit the amendment as `docs(plan): Subsys 7 Part B tasks from audit findings`.
5. Resume execution from Task 3.

This is not a placeholder -- Part A's tasks are concrete and
executable on their own terms, and Part B's content is explicitly
deferred to when the evidence it needs exists.

Branch: `dev`. Tag at end (Part B final task): `phase1-subsys-isa-execute`.

---

## Scope Note (Part A)

Part A scope is two tasks:

- **Task 1 (Audit):** Produce `docs/arch/subsys7-audit.md` with
  per-file findings for 20 execute/ files + 9 timing/ files. Also
  produce `docs/arch/isa-execute-model.md` scaffold (the
  mandatory per-seam design note, filled in at Part B close).
- **Task 2 (Scaffold):** Create `xdna_archspec::isa_execute` module
  (empty trait), `xdna_archspec::aie2::isa_execute_model` module
  (`Aie2IsaExecutor` ZST + `AIE2_ISA_EXECUTOR` singleton),
  `ArchConfig::isa_executor()` dispatch method, runtime-side
  `arch_handle::isa_executor()` accessor, and a dispatch test
  (`isa_executor_dispatches_to_aie2_for_aie2_family`).

No data migration, no call-site changes, no trait methods in Part A.
That all lands in Part B after the audit reveals exactly what belongs
where.

- Estimated file-count Part A: 6-7 files touched (2 new markdown docs,
  2 new archspec source files, 3 edits to existing archspec + xdna-emu
  source).
- Estimated commits Part A: 4-5 (audit commit + scaffold commit +
  wiring commit + dispatch test commit).

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green. Baseline at `phase1-subsys-stream-switch`: `2684 passed; 0 failed; 5 ignored` (verified at HEAD before plan write).
- `cargo test -p xdna-archspec --lib` green. Baseline: `297 passed; 0 failed; 2 ignored` (verified at HEAD before plan write).
- `cargo build` green. `cargo build --release` clean is required before the Part B final tag, not every commit.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green after rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`). Required at Part A close and Part B tag.
- No commit introduces `TODO` / `FIXME` / `unimplemented!()` without an open-issue reference. The `unimplemented!("AIE1 IsaExecutor ...")` in the Part A scaffold accessor is the sanctioned exception, mirroring `dma_model()`, `lock_model()`, and `stream_switch_model()`.
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`, `refactor(archspec):`); no emoji; ends with `Generated using Claude Code.`.
- All work on `dev`. No merges to `master` during this plan.
- **Every `cargo` call** must have `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` prepended (tblgen needs llvm-config 21.x, not mlir-aie's 23.x).

---

## File Structure

**Current layout (post-Subsystem 5, at `phase1-subsys-stream-switch` / HEAD):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── arch_handle.rs           # dma_model(), lock_model(),
│   │   │                            #   stream_switch_topology() accessors
│   │   └── ...
│   ├── interpreter/
│   │   ├── execute/
│   │   │   ├── mod.rs               # dispatcher tree (17 submodules)
│   │   │   ├── semantic.rs          # 63K: TableGen-driven scalar dispatch
│   │   │   ├── vector_dispatch.rs   # 5K: vector dispatch tree
│   │   │   ├── vector_arith.rs      # 100K
│   │   │   ├── vector_permute.rs    # 73K
│   │   │   ├── vector_srs.rs        # 53K
│   │   │   ├── vector_helpers.rs    # 51K
│   │   │   ├── vector_semantic.rs   # 50K
│   │   │   ├── vector_float.rs      # 47K
│   │   │   ├── control.rs           # 44K
│   │   │   ├── vector_misc.rs       # 41K
│   │   │   ├── vector_config.rs     # 41K
│   │   │   ├── vector_compare.rs    # 40K
│   │   │   ├── cycle_accurate.rs    # 38K
│   │   │   ├── vector_pack.rs       # 37K
│   │   │   ├── vector_ups.rs        # 34K
│   │   │   ├── vector_convert.rs    # 22K
│   │   │   ├── stream.rs            # 16K
│   │   │   ├── cascade.rs           # 13K
│   │   │   ├── vector_validate.rs   # 8.5K
│   │   │   ├── vmac_routing.rs      # 234K: PURE GENERATED DATA (probed)
│   │   │   ├── vmac_hw.rs           # 69K
│   │   │   ├── vector_matmul/       # bf16_pipeline.rs, helpers.rs, mod.rs
│   │   │   └── memory/
│   │   │       ├── mod.rs           # 121K: load/store handlers
│   │   │       └── neighbor.rs      # 6.4K
│   │   ├── timing/
│   │   │   ├── arbitration.rs
│   │   │   ├── barrier.rs
│   │   │   ├── deadlock.rs
│   │   │   ├── hazards.rs
│   │   │   ├── latency.rs           # ~30 AIE2-specific refs
│   │   │   ├── memory.rs
│   │   │   ├── mod.rs
│   │   │   ├── slots.rs
│   │   │   └── sync.rs
│   │   └── traits.rs                # Decoder/Executor/StateAccess -- out of scope
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                   # (will gain `pub mod isa_execute;`)
        ├── aie2/
        │   ├── mod.rs               # (will gain `pub mod isa_execute_model;`)
        │   └── ...                  # stream_switch_model, lock_model, etc.
        └── runtime.rs               # dma_model/lock_model/stream_switch_model
                                     #   isa_executor joins in Task 2
```

**Target layout (post-Part A, before Part B starts):**

```
xdna-emu/
├── docs/
│   └── arch/
│       ├── subsys7-audit.md         # NEW (Task 1): per-file audit findings
│       └── isa-execute-model.md     # NEW (Task 1): design note scaffold
├── src/
│   ├── device/
│   │   └── arch_handle.rs           # + isa_executor() accessor (Task 2)
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                   # + pub mod isa_execute;
        ├── isa_execute/
        │   └── mod.rs               # NEW (Task 2): IsaExecutor trait (empty body)
        ├── aie2/
        │   ├── mod.rs               # + pub mod isa_execute_model;
        │   └── isa_execute_model.rs # NEW (Task 2): Aie2IsaExecutor ZST +
        │                            #               AIE2_ISA_EXECUTOR singleton
        └── runtime.rs               # + isa_executor() trait method + impl +
                                     #   dispatch test
```

Part A new LOC: ~80 source LOC (trait + ZST + singleton + accessor + dispatch test) + ~600 audit LOC + ~100 design-note-scaffold LOC. No deletions or migrations in Part A.

**Target layout (post-Part B)** is finalized in the Part B amendment. Expected additions: `xdna_archspec::aie2::vmac::routing` (absorbs the 234K vmac_routing.rs), possibly `xdna_archspec::aie2::timing` (migrated latency values), one method per row of the audit's final trait-method list.

---

## Baseline to Preserve

Before Task 1, capture current numbers so later regression checks have a target:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: test result: ok. 2684 passed; 0 failed; 5 ignored; 0 measured

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: test result: ok. 297 passed; 0 failed; 2 ignored; 0 measured
```

Known pre-existing failure (carries through, documented in `NEXT-STEPS.md`):
- `bd_chain_repeat_on_memtile` EMU deadlock in bridge. Not blocking.

---

## Task 1: Audit + design-note scaffold

**Files:**
- Create: `docs/arch/subsys7-audit.md`
- Create: `docs/arch/isa-execute-model.md`

**Goal:** Catalogue per-file divergence evidence for the 20 files in
`src/interpreter/execute/` plus the 9-file `interpreter/timing/`
submodule, plus the adjacent entanglements (`vmac_hw.rs` constants,
memory-sized constants, `ProcessorModel` intrinsic-name coverage).
Close the audit with the three-question summary the spec requires
(tentative trait method list, data-migration list, AIE1 projection).
Also scaffold `docs/arch/isa-execute-model.md` so Part B's closing
task has a skeleton to fill in.

**Approach:** The audit's per-file work is grep-driven and evidence-
gathering, suited for parallel Explore agents. Task 1 uses one Explore
agent per functional area (six areas -- see subsections below). Each
agent returns ~200-800 words of per-file findings that the executing
agent merges into a single `subsys7-audit.md`. The three-question
closing section is synthesized by the executing agent from all six
agents' findings, not dispatched separately.

- [ ] **Step 1: Create the audit scaffold with baseline + structure**

Create `docs/arch/subsys7-audit.md` with this exact content:

```markdown
# Subsystem 7 -- ISA Execute Audit

**Subsystem:** 7 of 8 (Phase 1b of the device-family refactor)
**Spec:** [../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md](../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md)
**Plan:** [../superpowers/plans/2026-04-21-subsys7-isa-execute.md](../superpowers/plans/2026-04-21-subsys7-isa-execute.md)

## Baseline (pre-subsystem, at phase1-subsys-stream-switch tag / HEAD)

- `cargo test --lib`: 2684 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 297 passed; 0 failed; 2 ignored
- `cargo build --release`: clean
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit methodology

Per the spec, this audit is a per-file deep dive (Question 3 option
B) over the 20 files in `src/interpreter/execute/` plus the 9-file
`interpreter/timing/` submodule, grouped by functional area rather
than alphabetically.

Per-file subsection template:

- **Size + responsibility.** One sentence.
- **AIE2 hardcode count.** Grep count of literal `"AIE2"`,
  `AIE_ML_*`, `aie2`/`Aie2` identifiers, and arch-branded constants.
- **Divergence risks vs AIE1/AIE2P.** Evidence from file comments,
  llvm-aie TableGen, aie-rt per-arch headers.
- **Prescribed migration verb.** `move-to-archspec` /
  `read-archspec-via-accessor` / `wrap-in-trait` / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines
  added archspec-side.

Two files get ~2 pages each: `vmac_routing.rs` and `memory/mod.rs`.

---

## 1. Dispatcher / orchestration

Files: `execute/mod.rs`, `semantic.rs`, `cycle_accurate.rs`,
`vector_dispatch.rs`.

(Filled in by Task 1 Step 3.)

## 2. Scalar / control / stream / cascade

Files: `control.rs`, `stream.rs`, `cascade.rs`.

(Filled in by Task 1 Step 4.)

## 3. Memory

Files: `memory/mod.rs`, `memory/neighbor.rs`. (Deep dive for `mod.rs`.)

(Filled in by Task 1 Step 5.)

## 4. Vector ALU

Files: `vector_arith.rs`, `vector_compare.rs`, `vector_misc.rs`,
`vector_pack.rs`, `vector_ups.rs`, `vector_srs.rs`, `vector_helpers.rs`,
`vector_semantic.rs`, `vector_permute.rs`, `vector_float.rs`,
`vector_config.rs`, `vector_convert.rs`, `vector_validate.rs`.

(Filled in by Task 1 Step 6.)

## 5. VMAC / matmul

Files: `vmac_routing.rs` (deep dive), `vmac_hw.rs`, `vector_matmul/`.

(Filled in by Task 1 Step 7.)

## 6. Timing

Files: `interpreter/timing/{arbitration, barrier, deadlock, hazards,
latency, memory, mod, slots, sync}.rs`, plus `execute/cycle_accurate.rs`
latency tables.

(Filled in by Task 1 Step 8.)

---

## Closing summary

(Filled in by Task 1 Step 9.)

### Tentative trait method list

### Data migration list

### AIE1 projection

---

## Completion

(Filled in at the end of Subsystem 7, in the Part B final task.)
```

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs: Subsystem 7 audit scaffold

Per-file ISA execute audit scaffold for Subsystem 7. Sections for
all six functional areas (dispatcher, scalar/control, memory,
vector ALU, VMAC/matmul, timing), plus the closing three-question
summary (trait list, data migrations, AIE1 projection) and the
Completion section that lands at subsystem close.

Content filled in over Steps 3-9 of Task 1.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 2: Create the design-note scaffold**

Create `docs/arch/isa-execute-model.md` with this exact content:

```markdown
# ISA Execute Model -- Design Note

**Subsystem:** 7 (Phase 1b)
**Tag:** `phase1-subsys-isa-execute`
**Spec:** [../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md](../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains which shape differences
justify each method of the `IsaExecutor` trait (if any survive the
audit), what data lives where, and what an AIE1 port would look like.

Filled in after Task 1 (audit) lands and the Part B tasks execute.

---

## What lives where

(Filled in at Part B close.)

## Trait surface

(Filled in at Part B close.)

## The shape-vs-values rule, applied to ISA execute

(Filled in at Part B close.)

## What would AIE1 look like?

(Filled in at Part B close -- audit provides the one-paragraph
projection as part of its closing summary; the design note expands
that into ~300 words.)

## Alternatives rejected

(Filled in at Part B close. Expected content: Approach A if trait is
non-empty, Approach C always, pre-audit commitment always, sub-
subsystem decomposition always.)
```

Run:
```bash
git add docs/arch/isa-execute-model.md
git commit -m "docs: Subsystem 7 design-note scaffold

Scaffold for the per-seam design note required by the parent
device-family refactor. Content lands at Part B close.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 3: Dispatch Explore agent for Dispatcher / orchestration area**

Dispatch via the `Explore` agent (subagent_type=Explore) with this prompt:

```
You are auditing four files under /home/triple/npu-work/xdna-emu/src/interpreter/execute/
for the Subsystem 7 ISA Execute refactor of the xdna-emu project
(AMD NPU emulator). Read the spec first:
/home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md

Then audit these files:
- mod.rs (small; dispatcher tree)
- semantic.rs (~63K; TableGen-driven scalar dispatch via SemanticOp)
- cycle_accurate.rs (~38K; pipeline stages, hazards, timing)
- vector_dispatch.rs (~5K; vector dispatch tree)

For each file, produce a subsection in this exact format:

### `<filename>`

- **Size + responsibility.** [one sentence]
- **AIE2 hardcode count.** [grep for literal "AIE2", AIE_ML, aie2/Aie2,
  hardcoded magic numbers like 256/512 when they are AIE2-specific;
  report a count and list the most notable ones with line numbers]
- **Divergence risks vs AIE1/AIE2P.** [evidence from comments,
  reference llvm-aie TableGen at /home/triple/npu-work/llvm-aie/llvm/lib/Target/AIE/
  where relevant, reference aie-rt at /home/triple/npu-work/aie-rt/driver/src/
  where relevant]
- **Prescribed migration verb.** One of: `move-to-archspec`,
  `read-archspec-via-accessor`, `wrap-in-trait`, `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines
  added archspec-side.

Finish with a one-paragraph summary: does this area warrant any
`IsaExecutor` trait methods? Any arch-specific data to migrate?

Target length: 400-800 words total. Use file:line references
throughout. Do NOT edit any source files; read only.
```

Wait for the agent to return. Paste its output into the audit doc at
section "## 1. Dispatcher / orchestration" (replacing the
`(Filled in by Task 1 Step 3.)` placeholder).

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 dispatcher/orchestration findings

Per-file audit of execute/mod.rs, semantic.rs, cycle_accurate.rs,
vector_dispatch.rs. Captures AIE2 hardcode count, divergence risks,
migration verb per file.

Generated using Claude Code."
```

Expected: clean commit with ~400-800 words of audit content.

- [ ] **Step 4: Dispatch Explore agent for Scalar / control / stream / cascade area**

Dispatch via the `Explore` agent with this prompt:

```
You are auditing three files under /home/triple/npu-work/xdna-emu/src/interpreter/execute/
for the Subsystem 7 ISA Execute refactor. Read the spec first:
/home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md

Files:
- control.rs (~44K; branch/lock/halt/stream-wait, plus ControlUnit struct)
- stream.rs (~16K; stream I/O ops routed through StreamOps)
- cascade.rs (~13K; 384-bit cascade link for matmul chains)

For each file, produce a subsection in the same format as Step 3
(Size + responsibility, AIE2 hardcode count, Divergence risks,
Prescribed migration verb, Estimated LOC impact).

Pay specific attention to:
- Cascade register width (AIE1=no cascade, AIE2=384-bit, AIE2P=?)
- Stream port enumeration (tile-kind-dependent, probably already
  archspec-resident via Subsystem 5 stream_switch data)
- Control-flow semantics (branch delay slots, LR update timing;
  usually arch-generic)

Finish with a one-paragraph summary covering the area.

Target length: 300-600 words total. Use file:line references. Do NOT
edit any source files.
```

Merge the agent's output into the audit doc at "## 2. Scalar / control / stream / cascade".

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 scalar/control/stream/cascade findings

Per-file audit of control.rs, stream.rs, cascade.rs. Cascade width
(384-bit AIE2; absent AIE1) and stream port enumeration flagged as
divergence points.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 5: Dispatch Explore agent for Memory area (deep dive)**

Dispatch via the `Explore` agent with this prompt:

```
You are doing a DEEP-DIVE audit (target ~2 pages) of the memory
subsystem for Subsystem 7 ISA Execute.

Files under /home/triple/npu-work/xdna-emu/src/interpreter/execute/memory/:
- mod.rs (~121K; load/store handlers)
- neighbor.rs (~6.4K; neighbor-tile memory access)

Also read adjacent context:
- /home/triple/npu-work/xdna-emu/src/device/tile/core_state.rs (memory size)
- crates/xdna-archspec/src/aie2/memory_sizes/ or similar (existing archspec memory data)
- /home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md

Your deliverable is a ~2-page audit section covering:

1. **Per-file breakdown** (same template as Step 3: size,
   responsibility, AIE2 hardcode count, divergence risks, migration
   verb, LOC impact).
2. **Structural breakdown of `memory/mod.rs`.** The 121K file almost
   certainly does multiple things. Enumerate them: which functions
   handle which instructions? Which handle which memory regions? How
   much is arch-generic (element-wise dispatch by ElementType/size)
   vs arch-specific (memory-bank topology, tile-kind-specific address
   layout)? Cite specific functions and line ranges.
3. **Key question.** Does the audit justify a `memory_load` or
   `memory_store` trait method on `IsaExecutor`, or is the arch-
   specific content reducible to data constants readable via
   `arch_handle::*` accessors?
4. **Data migration candidates.** Which constants, tables, or
   dispatch tables in these files are candidates for moving to
   archspec? Be specific.

Target length: 900-1500 words. Use file:line references throughout.
Do NOT edit any source files.
```

Merge the agent's output into the audit doc at "## 3. Memory".

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 memory deep dive

Deep-dive audit of memory/mod.rs (121K) and memory/neighbor.rs.
Structural breakdown of the 121K file, arch-generic vs arch-specific
split, trait-method-vs-data question resolved.

Generated using Claude Code."
```

Expected: clean commit with ~900-1500 words.

- [ ] **Step 6: Dispatch Explore agent for Vector ALU area**

Dispatch via the `Explore` agent with this prompt:

```
You are auditing 13 files under /home/triple/npu-work/xdna-emu/src/interpreter/execute/
for Subsystem 7 ISA Execute. Read the spec first:
/home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md

Files (size in parentheses):
- vector_arith.rs (100K)
- vector_permute.rs (73K)
- vector_srs.rs (53K)
- vector_helpers.rs (51K)
- vector_semantic.rs (50K)
- vector_float.rs (47K)
- vector_misc.rs (41K)
- vector_config.rs (41K)
- vector_compare.rs (40K)
- vector_pack.rs (37K)
- vector_ups.rs (34K)
- vector_convert.rs (22K)
- vector_validate.rs (8.5K)

Per-file template (Step 3 format): Size + responsibility, AIE2 hardcode
count, Divergence risks, Prescribed migration verb, Estimated LOC
impact.

Pay specific attention (highest-priority questions for Part B):

1. **SRS/UPS/Pack/Convert rounding and saturation.** `vector_srs.rs`,
   `vector_ups.rs`, `vector_pack.rs`, `vector_convert.rs` -- these are
   the "pipeline" ops whose behavior is configured by operation config
   word. Does the rounding/saturation implementation read arch-specific
   data (config field layouts, rounding-mode enumerations)? Could AIE1
   plug in different config-field layouts without changing algorithm,
   or would the algorithm itself need to change? Cross-reference
   aietools Python models ONLY if open-source TableGen/aie-rt is
   silent:
   - /home/triple/npu-work/aietools/data/aie_ml/lib/python_model/model/srs_ups.py
   - /home/triple/npu-work/aietools/data/aie_ml/lib/python_model/model/mulmac.py
   - (reading reference only; never copy code)

2. **Accumulator-width rules.** Any file referencing 384/512-bit
   accumulators, or mixed-width promotion (v8acc32 -> v8acc64). Flag
   which rules are AIE2-specific.

3. **Permute/shuffle.** vector_permute.rs and Shuffle handling in
   vector_dispatch.rs -- how much is data-table-driven (archspec
   material) vs algorithmic?

Finish with a one-paragraph summary: which files in this area warrant
trait methods (if any)? Which warrant data migration? How many files
can stay leave-alone?

Target length: 1200-1800 words. Most files get one paragraph; SRS/UPS/
permute get 2-3 paragraphs. Use file:line references. Do NOT edit.
```

Merge into audit at "## 4. Vector ALU".

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 vector ALU findings

Per-file audit of 13 vector_*.rs files. Focus on SRS/UPS/Pack/Convert
pipeline-op semantics (rounding + saturation config), accumulator
width rules, and permute/shuffle data-vs-algorithm split. Identifies
which files warrant IsaExecutor trait methods.

Generated using Claude Code."
```

Expected: clean commit with ~1200-1800 words.

- [ ] **Step 7: Dispatch Explore agent for VMAC / matmul area (deep dive)**

Dispatch via the `Explore` agent with this prompt:

```
You are doing a DEEP-DIVE audit (target ~2 pages) of the VMAC / matmul
subsystem for Subsystem 7 ISA Execute.

Files under /home/triple/npu-work/xdna-emu/src/interpreter/execute/:
- vmac_routing.rs (~234K; PURE generated data from probed C++ ISS)
- vmac_hw.rs (~69K; VMAC hardware model)
- vector_matmul/mod.rs
- vector_matmul/bf16_pipeline.rs
- vector_matmul/helpers.rs

Also skim:
- /home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md
- /home/triple/npu-work/aietools/data/aie_ml/lib/python_model/model/mulmac.py
  (READ ONLY; reading reference for AIE2 VMAC semantics; never copy)

Your deliverable is a ~2-page audit section covering:

1. **vmac_routing.rs breakdown.** What EXACTLY is the data shape?
   (It's described as 789 m-bits, 15808 route entries, prmx/prmy tables.)
   Does it move wholesale to archspec, or should only specific tables
   move? What is the consumer interface that vmac_hw.rs currently
   uses? Any call to vmac_routing content outside vmac_hw.rs? Use
   grep to verify.

2. **vmac_hw.rs structural breakdown.** How much of the 69K is:
   (a) Arch-generic VMAC algorithm (mul + accumulate + routing lookup
       + rounding)?
   (b) Arch-specific constants (accumulator width, lane counts,
       pmode enumeration)?
   (c) Direct consumer of vmac_routing.rs static tables?

   Cite specific functions and line ranges.

3. **vector_matmul/.** Is this a separate code path from vmac_hw.rs,
   or does it ultimately call into vmac_hw? What is the boundary?

4. **Trait candidates.** Does the audit support a `vmac_route(mbit, pmode)`
   trait method (spec's best-guess)? Or is the routing data directly
   accessible via `arch_handle::vmac_routing_table()` with the algorithm
   in xdna-emu? The difference: trait method = algorithm per arch;
   accessor = data per arch, algorithm shared.

5. **AIE1 projection.** Given aie-rt and llvm-aie have AIE1 data --
   would AIE1's VMAC crossbar be structurally similar (smaller but
   same shape) or fundamentally different? Cite evidence.

Target length: 900-1500 words. Use file:line references throughout.
Do NOT edit source files.
```

Merge into audit at "## 5. VMAC / matmul".

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 VMAC/matmul deep dive

Deep-dive audit of vmac_routing.rs (234K pure data) and vmac_hw.rs
(69K VMAC hardware model) plus vector_matmul/. Resolves wholesale
move vs partial extraction; trait-method-vs-accessor question for
crossbar routing; AIE1 crossbar-shape projection.

Generated using Claude Code."
```

Expected: clean commit with ~900-1500 words.

- [ ] **Step 8: Dispatch Explore agent for Timing area**

Dispatch via the `Explore` agent with this prompt:

```
You are auditing the timing subsystem for Subsystem 7 ISA Execute.
Read the spec first:
/home/triple/npu-work/xdna-emu/docs/superpowers/specs/2026-04-21-subsys7-isa-execute-design.md

Note: per brainstorming question 4 and the spec, timing is IN SCOPE
for data migration but NOT for a trait seam this round. The audit
should confirm that stance or flag evidence it's wrong.

Files under /home/triple/npu-work/xdna-emu/src/interpreter/timing/:
- arbitration.rs
- barrier.rs
- deadlock.rs
- hazards.rs
- latency.rs  (~30 AIE2-specific references)
- memory.rs
- mod.rs
- slots.rs
- sync.rs

Also audit the timing-related portions of:
- /home/triple/npu-work/xdna-emu/src/interpreter/execute/cycle_accurate.rs
  (specifically any per-instruction latency tables)

Also skim:
- crates/xdna-archspec/src/aie2/isa/types.rs (check what timing data
  the already-archspec-resident ProcessorModel includes)

Per-file template (Step 3 format). Emphasize:

1. **What's already in archspec.** ProcessorModel/itineraries shipped
   in Subsystem 6 -- which timing values does it already carry? Which
   remain hardcoded in xdna-emu?

2. **Data-migration candidates.** For each remaining hardcoded AIE2
   timing value in xdna-emu, what's its archspec destination? Extend
   ProcessorModel? New timing module under aie2? Be specific.

3. **Trait-seam argument (if any).** If any of these files has
   genuinely arch-specific algorithm (not just values), flag it. The
   spec predicts none; the audit's job is to confirm or refute.

Finish with a one-paragraph summary confirming or refuting the
"data migration only, no trait seam" stance.

Target length: 600-1000 words. Use file:line references. Do NOT edit.
```

Merge into audit at "## 6. Timing".

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 timing findings

Audit of interpreter/timing/ 9 files plus cycle_accurate.rs latency
tables. Confirms (or refutes) the spec's 'data-migration only, no
trait seam' stance. Identifies remaining AIE2 hardcoded timing values
outside ProcessorModel.

Generated using Claude Code."
```

Expected: clean commit with ~600-1000 words.

- [ ] **Step 9: Synthesize closing summary (trait list + data migrations + AIE1 projection)**

Read the six area sections you just filled in (dispatcher, scalar/
control, memory, vector ALU, VMAC, timing). Synthesize into the
audit's closing section.

Fill in the audit doc's `## Closing summary` with three subsections:

**Tentative trait method list.** For each candidate trait method
surfaced by the audit, write one line:

```markdown
- `<method_name>`: <one-line description>. Justified by: <which file/
  which finding>. Rejected alternatives: <accessor / constant /
  leave-alone + why trait is needed instead>.
```

If the audit surfaces zero trait methods, write:
```markdown
**No trait methods warranted.** The audit found that every expected
shape divergence reduces to data that can live in archspec and be
read via `arch_handle::*` accessors. `IsaExecutor` ships as an empty
trait in Part A; Part B does data migrations only and does not add
trait methods. (This matches Approach A from the spec's
alternatives-considered.)
```

**Data migration list.** One row per item:

| Source (xdna-emu) | Destination (archspec) | Pattern | Task (Part B) |
|---|---|---|---|
| ... | ... | `move-to-archspec` / `extract-constants` / `extend-existing` | TBD |

Include rows for: vmac_routing.rs (expected), any timing values,
memory constants if found, intrinsic indexing if found, anything
else the audit surfaces.

**AIE1 projection.** ~100 words. Reuse the spec's AIE1 projection as
a starting point, refine based on audit findings. The design-note
file (`docs/arch/isa-execute-model.md`) will expand this into ~300
words at Part B close.

Run:
```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs(audit): Subsystem 7 closing summary

Synthesized closing summary: tentative IsaExecutor trait method list,
data-migration table, AIE1 projection. Drives Part B task authoring.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 10: Verify audit integrity and global invariants**

Sanity-check that the audit is complete and the repo is still green.

Run:
```bash
# All six area sections have content (no unfilled placeholders)
grep -n "Filled in by Task 1 Step" /home/triple/npu-work/xdna-emu/docs/arch/subsys7-audit.md
# Expected: no output (or only in the Completion section, which is
# correct to still be unfilled)

# Tests still green
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: 2684 passed; 0 failed; 5 ignored

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: 297 passed; 0 failed; 2 ignored
```

If any placeholder remains in sections 1-6 or in the closing summary,
go back and fix it before Task 2.

No commit for this step -- it's a verification checkpoint only.

---

## Task 2: Trait scaffold + runtime wiring + dispatch test

**Files:**
- Create: `crates/xdna-archspec/src/isa_execute/mod.rs`
- Create: `crates/xdna-archspec/src/aie2/isa_execute_model.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (one line add)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (one line add)
- Modify: `crates/xdna-archspec/src/runtime.rs` (trait method + impl + dispatch test)
- Modify: `src/device/arch_handle.rs` (new `isa_executor()` accessor)

**Goal:** Land the neutral scaffold for `IsaExecutor`. Empty trait
(method list stays empty until Part B tasks add methods based on
audit findings), concrete `Aie2IsaExecutor` ZST, `AIE2_ISA_EXECUTOR`
singleton, `ArchConfig::isa_executor()` dispatch method (matches
`dma_model`/`lock_model`/`stream_switch_model` pattern), runtime-side
`arch_handle::isa_executor()` accessor via `OnceLock`, one dispatch
test. After Task 2 lands, adding a trait method in Part B is one edit
to `isa_execute/mod.rs` + one method body in `aie2/isa_execute_model.rs`.

- [ ] **Step 1: Create the empty `IsaExecutor` trait module**

Create `crates/xdna-archspec/src/isa_execute/mod.rs`:

```rust
//! ISA execute model -- per-arch behavioral seam for instruction
//! execution semantics.
//!
//! Subsystem 7 (Phase 1b) of the device-family refactor introduces
//! this module. The `IsaExecutor` trait is the seam by which
//! execute-layer code reaches arch-specific behavior where AIE1,
//! AIE2, and AIE2P have fundamentally different algorithmic shape
//! (as opposed to different values, which live in archspec data
//! modules and are read via `arch_handle::*` accessors).
//!
//! The trait ships empty in Part A (audit establishes the method
//! list; Part B populates). See the per-seam design note at
//! `docs/arch/isa-execute-model.md`.

/// Per-arch execute-layer behavioral seam.
///
/// Implementations cover the operation-level behavior where arches
/// genuinely diverge in algorithmic shape. Typical candidates:
/// VMAC expansion, vector pipeline rounding/saturation, accumulator-
/// width promotion rules.
///
/// The method list is audit-driven. An empty trait is a valid
/// landing -- it means the audit concluded that all divergence is
/// data-expressible and this seam is reserved for future use.
pub trait IsaExecutor: Send + Sync + core::fmt::Debug {
    // Methods added in Part B per audit findings.
}
```

- [ ] **Step 2: Create the AIE2 impl + singleton**

Create `crates/xdna-archspec/src/aie2/isa_execute_model.rs`:

```rust
//! AIE2 implementation of `IsaExecutor`.
//!
//! Shipped in Part A with no methods (mirroring the empty trait in
//! `crate::isa_execute`). Part B of Subsystem 7 adds method bodies
//! as trait methods are added, per the audit's findings.

use crate::isa_execute::IsaExecutor;

/// Zero-sized type representing the AIE2 execute model.
#[derive(Debug, Default)]
pub struct Aie2IsaExecutor;

impl IsaExecutor for Aie2IsaExecutor {
    // Method impls added in Part B.
}

/// Process-global singleton used by `ArchConfig::isa_executor()`.
pub static AIE2_ISA_EXECUTOR: Aie2IsaExecutor = Aie2IsaExecutor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aie2_isa_executor_is_zst() {
        assert_eq!(core::mem::size_of::<Aie2IsaExecutor>(), 0);
    }

    #[test]
    fn aie2_isa_executor_impls_trait() {
        // Static assertion: the singleton can be borrowed as
        // &'static dyn IsaExecutor (the shape the accessor will use).
        let _: &'static dyn IsaExecutor = &AIE2_ISA_EXECUTOR;
    }
}
```

- [ ] **Step 3: Wire the modules into `lib.rs` and `aie2/mod.rs`**

Modify `crates/xdna-archspec/src/lib.rs`. Find the line `pub mod stream_switch;` and add after it:

```rust
pub mod isa_execute;
```

Modify `crates/xdna-archspec/src/aie2/mod.rs`. Find the line `pub mod stream_switch_model;` and add after it:

```rust
pub mod isa_execute_model;
```

(The exact existing line may differ; use Grep to find a similar `pub mod *_model;` line and add the new one immediately after to preserve ordering.)

- [ ] **Step 4: Add `isa_executor()` method to ArchConfig trait + impl**

Modify `crates/xdna-archspec/src/runtime.rs`. Find the `stream_switch_model` trait method declaration on `ArchConfig` and add immediately after it:

```rust
    /// Per-arch ISA execute model (Subsystem 7 behavioral seam).
    ///
    /// Returns `&'static dyn IsaExecutor` covering operation-level
    /// behavior that varies in algorithmic shape between arch
    /// families. Ships empty in Part A; Part B tasks add methods.
    fn isa_executor(&self) -> &'static dyn crate::isa_execute::IsaExecutor;
```

Find the `stream_switch_model` impl on `ModelConfig` (the one that returns `&AIE2_STREAM_SWITCH_MODEL`) and add an impl of the new method immediately after it:

```rust
    fn isa_executor(&self) -> &'static dyn crate::isa_execute::IsaExecutor {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::isa_execute_model::AIE2_ISA_EXECUTOR
            }
            Architecture::Aie => unimplemented!(
                "AIE1 IsaExecutor not populated; Phase 1 refactor ships AIE2 seam only"
            ),
        }
    }
```

- [ ] **Step 5: Add dispatch test**

In the same `crates/xdna-archspec/src/runtime.rs`, find the existing test `stream_switch_model_dispatches_to_aie2_for_aie2_family` and add a sibling test immediately after it:

```rust
    #[test]
    fn isa_executor_dispatches_to_aie2_for_aie2_family() {
        let aie2 = ModelConfig {
            architecture: Architecture::Aie2,
            ..ModelConfig::default()
        };
        let executor = aie2.isa_executor();
        // Confirms dispatch succeeds and returns the AIE2 singleton.
        // Pointer equality against the ZST singleton's address.
        let expected: &dyn crate::isa_execute::IsaExecutor =
            &crate::aie2::isa_execute_model::AIE2_ISA_EXECUTOR;
        assert!(core::ptr::eq(executor as *const _ as *const (), expected as *const _ as *const ()));

        let aie2p = ModelConfig {
            architecture: Architecture::Aie2p,
            ..ModelConfig::default()
        };
        let executor_aie2p = aie2p.isa_executor();
        assert!(core::ptr::eq(executor_aie2p as *const _ as *const (), expected as *const _ as *const ()));
    }
```

- [ ] **Step 6: Verify archspec builds and tests pass**

Run:
```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -5
```

Expected: test result OK. Count should be `300 passed; 0 failed; 2 ignored` (previous 297 + 3 new: `aie2_isa_executor_is_zst`, `aie2_isa_executor_impls_trait`, `isa_executor_dispatches_to_aie2_for_aie2_family`).

If the test count is not 300, investigate: check all three new tests exist in the modules just created, and that the modules are wired into `lib.rs` / `aie2/mod.rs`.

- [ ] **Step 7: Commit archspec-side scaffold**

```bash
git add crates/xdna-archspec/src/isa_execute/mod.rs \
        crates/xdna-archspec/src/aie2/isa_execute_model.rs \
        crates/xdna-archspec/src/lib.rs \
        crates/xdna-archspec/src/aie2/mod.rs \
        crates/xdna-archspec/src/runtime.rs

git commit -m "feat(archspec): IsaExecutor trait + Aie2IsaExecutor scaffold

Part A of Subsystem 7 (ISA Execute). Introduces the neutral scaffold:

- \`xdna_archspec::isa_execute::IsaExecutor\` trait (empty body; Part B
  populates methods per audit findings)
- \`xdna_archspec::aie2::isa_execute_model::Aie2IsaExecutor\` ZST +
  \`AIE2_ISA_EXECUTOR\` singleton
- \`ArchConfig::isa_executor()\` trait method + ModelConfig impl
  (returns \`&'static dyn IsaExecutor\`; AIE1 arm is \`unimplemented!\`)
- Dispatch test: \`isa_executor_dispatches_to_aie2_for_aie2_family\`

Trait body stays empty in Part A. Part B tasks add methods as the
audit's trait-method list requires them.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 8: Add `isa_executor()` accessor to `arch_handle.rs`**

Modify `src/device/arch_handle.rs`. Find the `stream_switch_topology()` accessor (the most recent addition from Subsystem 5) and add a sibling accessor immediately after it.

Near the top of the file, the existing imports likely include `use xdna_archspec::stream_switch::StreamSwitchTopology;`. Add:

```rust
use xdna_archspec::isa_execute::IsaExecutor;
```

Find the `STREAM_SWITCH_TOPOLOGY` `OnceLock` declaration and add immediately after it:

```rust
static ISA_EXECUTOR: OnceLock<&'static dyn IsaExecutor> = OnceLock::new();
```

Find the `pub fn stream_switch_topology()` function body and add a sibling function immediately after:

```rust
/// Per-arch ISA execute seam for operation-level behavioral divergence.
///
/// Returns the `&'static dyn IsaExecutor` for the runtime's default
/// arch. Ships empty in Part A; Part B adds methods as the audit
/// requires.
pub fn isa_executor() -> &'static dyn IsaExecutor {
    *ISA_EXECUTOR.get_or_init(|| {
        xdna_archspec::runtime::default_arch().isa_executor()
    })
}
```

- [ ] **Step 9: Verify xdna-emu builds and tests pass**

Run:
```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
```

Expected: clean build.

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: `2684 passed; 0 failed; 5 ignored` (no change -- no new xdna-emu tests yet, accessor is not yet consumed).

- [ ] **Step 10: Commit xdna-emu-side wiring**

```bash
git add src/device/arch_handle.rs
git commit -m "feat: isa_executor() accessor in arch_handle

Adds runtime-side \`arch_handle::isa_executor()\` accessor for the
IsaExecutor trait seam introduced in the prior archspec commit.
OnceLock-cached, returns \`&'static dyn IsaExecutor\`, matches
established pattern alongside dma_model/lock_model/stream_switch_topology.

Accessor is not yet consumed (the trait is empty in Part A). Part B
tasks will migrate call sites to it as trait methods are added.

Generated using Claude Code."
```

Expected: clean commit.

- [ ] **Step 11: Bridge smoke + full-tree sanity**

Rebuild the FFI cdylib, then run a bridge smoke test:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -20
```

Expected: build clean, bridge smoke PASS for add_one_cpp_aiecc (Chess + Peano).

If the smoke test fails, rebuild the FFI first (`cargo clean -p xdna-emu-ffi && cargo build -p xdna-emu-ffi`) and re-run. Stale `.so` files have produced phantom regressions (documented in NEXT-STEPS.md).

No commit for this step -- it's a verification checkpoint only.

---

## Part A Completion

At the close of Task 2 Step 11, Part A is complete. State:

- **Audit:** `docs/arch/subsys7-audit.md` has all six area sections
  filled in and a closing summary with tentative trait method list +
  data migration list + AIE1 projection.
- **Scaffold:** Empty `IsaExecutor` trait + `Aie2IsaExecutor` ZST +
  singleton + `ArchConfig::isa_executor()` dispatch method +
  `arch_handle::isa_executor()` accessor + dispatch test. Test counts:
  xdna-emu 2684/0/5 (unchanged); archspec 300/0/2 (+3 from scaffold).
- **Design-note scaffold:** `docs/arch/isa-execute-model.md` exists
  with placeholders; filled in at Part B close.

Commit graph at this point (approximate):

```
<task2 step10>  feat: isa_executor() accessor in arch_handle
<task2 step7>   feat(archspec): IsaExecutor trait + Aie2IsaExecutor scaffold
<task1 step9>   docs(audit): Subsystem 7 closing summary
<task1 step8>   docs(audit): Subsystem 7 timing findings
<task1 step7>   docs(audit): Subsystem 7 VMAC/matmul deep dive
<task1 step6>   docs(audit): Subsystem 7 vector ALU findings
<task1 step5>   docs(audit): Subsystem 7 memory deep dive
<task1 step4>   docs(audit): Subsystem 7 scalar/control/stream/cascade findings
<task1 step3>   docs(audit): Subsystem 7 dispatcher/orchestration findings
<task1 step2>   docs: Subsystem 7 design-note scaffold
<task1 step1>   docs: Subsystem 7 audit scaffold
```

~11 commits total for Part A.

**NO TAG at Part A close.** The `phase1-subsys-isa-execute` tag lands
at Part B's final task. If work must pause at the end of Part A, that
is a natural stopping point, but the subsystem is not "done" until
Part B.

---

## Part B -- Authored 2026-04-21 After Audit Landed

The Subsystem 7 audit (committed through `63fca4a`) concluded
**Approach A: zero `IsaExecutor` trait methods warranted.** Every
candidate divergence reduces to data-expressible differences. Part B
is therefore data-migration-only: ~6,300 LOC moving from xdna-emu
into `xdna_archspec::aie2::*` submodules plus ~90 scattered accessor
migrations, followed by the completion + tag task.

**Pre-verified findings driving Part B** (see audit closing summary):

- `RoundingMode` enum duplication in `vector_srs.rs` (lines 34-87) and
  `vector_float.rs` (lines 34-70) is **structurally identical**. Only
  cosmetic doc-comment differences: `vector_float.rs` has the more
  accurate enum-level doc ("Hardware rounding modes for the SRS
  instruction and bf16 conversion"); `vector_srs.rs` has fuller
  per-variant docs. Merge: keep `vector_float.rs`'s enum-level doc +
  `vector_srs.rs`'s per-variant docs. Verified read-only 2026-04-21.
- `vmac_routing.rs` is `include!`'d from `vmac_hw.rs` line 15 with
  only two consumers (`eval_prmx`/`eval_prmy` called at lines 1053,
  1062 of `vmac_hw.rs`). Move is mechanical.
- `has_cascade_link: bool` is a genuinely new archspec addition, not a
  simple move. Plumbing: add field to `ProcessorModel`, wire through
  `from_arch_model` (with default `true` for AIE2 arches), add
  accessor, gate `cascade.rs` call sites. Estimated LOC is closer to
  20-30 than the +2 the audit table suggested.

**Task order is safest-first** (reversed from the audit's
size-ordering): accessor bundles warm up the pattern, medium
consolidations follow, big wholesale moves land last when the
pattern is proven. This preserves green tests at every commit.

---

### Task 3: Accessor migrations (5 bundles)

**Files:**
- Modify: `src/interpreter/execute/semantic.rs` (control register IDs)
- Modify: `src/interpreter/execute/control.rs` (lock quadrant boundaries)
- Modify: `src/interpreter/timing/latency.rs` (latency constants)
- Modify: `src/interpreter/execute/memory/mod.rs` (PROC_BUS_*)
- Modify: `src/interpreter/execute/cycle_accurate.rs` (LatencyTable::aie2 + delay slot)
- Possibly modify: `crates/xdna-archspec/src/aie2/` (extend existing modules if any constant isn't already archspec-resident)

**Goal:** Replace hardcoded AIE2 literals in five distinct execute-layer call-site bundles with archspec accessor calls. Each bundle gets its own commit. The sub-migrations are independent; they may be reordered if one is blocked, but all five must land in this task.

For each sub-bundle below, the pattern is: (1) verify the constant's archspec destination exists; if not, add it to archspec with a test; (2) find all call sites in the target xdna-emu file via grep; (3) replace each literal with the accessor call; (4) `cargo test --lib` (expect unchanged count); (5) commit.

- [ ] **Step 1: Bundle 3a -- Control register IDs in `semantic.rs`**

Audit reference: `semantic.rs` uses lock-quadrant-style IDs `crSat`, `crRnd`, `crSRSSign` and q-regs (~30 sites). Find them with:

```bash
cd /home/triple/npu-work/xdna-emu
grep -nE "crSat|crRnd|crSRSSign|crSat0|crUPSSign" src/interpreter/execute/semantic.rs
```

For each match, verify the constant exists in archspec (likely in `crates/xdna-archspec/src/aie2/aiert/` or `aie2::processor`). If the constant is NOT already in archspec, add it to the appropriate archspec module (with a drift-detection test asserting it matches the aie-rt header value). Then update the `semantic.rs` site to read via the accessor, e.g.:

```rust
// Before
const CR_SAT: u32 = 0x1D0;  // or a literal usage
// After
use xdna_archspec::aie2::<path>::CR_SAT;
```

(Or via `arch_handle::processor_model().ctrl_regs().sat()` if the archspec shape prefers a method.)

Commit:
```bash
cd /home/triple/npu-work/xdna-emu
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: 2684 passed (unchanged) OR +N if drift tests added.
git add src/interpreter/execute/semantic.rs crates/xdna-archspec/src/aie2/  # paths as modified
git commit -m "refactor: control register IDs read from archspec in semantic.rs

Migrates the ~30 hardcoded control-register-ID sites in semantic.rs
to read via archspec accessors. Part of Subsystem 7 Part B accessor
migrations (audit item 7).

Generated using Claude Code."
```

- [ ] **Step 2: Bundle 3b -- Lock quadrant boundaries in `control.rs`**

Audit reference: the quadrant boundaries are 0-15 (South), 16-31 (West), 32-47 (North), 48-63 (East/Internal). Find them:

```bash
grep -nE "\b(16|32|48)\b.*lock|lock.*(16|32|48)" src/interpreter/execute/control.rs
```

(Adjust grep if literal hits are noisy.) Verify the archspec `aie2::locks` module has these quadrant boundaries. If not, add them. Update the `control.rs` sites to read via accessor.

Commit (same style as 3a):
```bash
git commit -m "refactor: lock quadrant boundaries read from archspec in control.rs

Migrates the ~20 lock-quadrant literal sites (0-15/16-31/32-47/48-63)
in control.rs to read via archspec's aie2::locks accessor. Part of
Subsystem 7 Part B accessor migrations (audit item 8).

Generated using Claude Code."
```

- [ ] **Step 3: Bundle 3c -- Latency constants in `timing/latency.rs`**

Audit reference: 8 raw numeric literals duplicating archspec values (`LATENCY_MEMORY`, `LATENCY_SCALAR_MUL`, etc.). Find them:

```bash
grep -nE "^\s*const LATENCY_|pub const LATENCY_" src/interpreter/timing/latency.rs
```

For each, check if `ProcessorModel` (or another archspec module) already has the value. If yes, replace the xdna-emu const with an accessor usage. If no, add it to archspec first. Update any call sites that referenced the removed consts.

Commit: `refactor: latency constants read from archspec in timing/latency.rs`

- [ ] **Step 4: Bundle 3d -- PROC_BUS_* in `memory/mod.rs`**

Audit reference: `PROC_BUS_BASE`, `PROC_BUS_END` literals. Find them:

```bash
grep -nE "PROC_BUS_BASE|PROC_BUS_END" src/interpreter/execute/memory/mod.rs
```

Verify the constants exist in archspec `aie2::compute` (or equivalent). If not, add them (with drift test). Update the ~10 call sites in `memory/mod.rs`.

Commit: `refactor: PROC_BUS_* literals read from archspec in memory/mod.rs`

- [ ] **Step 5: Bundle 3e -- LatencyTable::aie2() + delay-slot in `cycle_accurate.rs`**

Audit reference: `LatencyTable::aie2()` constructor call + delay-slot constant. Find them:

```bash
grep -nE "LatencyTable::aie2|DELAY_SLOT" src/interpreter/execute/cycle_accurate.rs
```

The `LatencyTable::aie2()` call at line ~87 should route through `default_arch()` or `arch_handle::processor_model()`; the delay-slot constant should come from archspec. Verify archspec has these or extend it.

Commit: `refactor: cycle_accurate reads LatencyTable + delay slot from archspec`

- [ ] **Step 6: Verify Task 3 close**

```bash
cd /home/triple/npu-work/xdna-emu
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: 2684 passed (or +N from any drift tests added to archspec).
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: 300 passed or +N from any drift tests added.
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -20
# Expected: PASS on both Chess and Peano.
```

Five commits for Task 3 if all sub-bundles land cleanly. If any sub-bundle surfaces unexpected archspec plumbing (e.g., a constant is deeply embedded and reaching it via accessor requires more structural work), commit what's done and BLOCK-escalate the rest for plan revision.

---

### Task 4: Medium data-moves (4 migrations)

**Files:**
- Create: `crates/xdna-archspec/src/aie2/rounding.rs` (or extend existing module -- verify at Step 1)
- Create: `crates/xdna-archspec/src/aie2/matmul.rs` (or extend if exists)
- Create: `crates/xdna-archspec/src/aie2/ups.rs` (or extend if exists)
- Modify: `crates/xdna-archspec/src/aie2/processor.rs` (or equivalent; add `has_cascade_link` + `CASCADE_WORDS`)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (wire any new modules)
- Modify: `src/interpreter/execute/vector_srs.rs` (RoundingMode dedup)
- Modify: `src/interpreter/execute/vector_float.rs` (RoundingMode dedup)
- Modify: `src/interpreter/execute/vector_config.rs` (matmul geometry tables)
- Modify: `src/interpreter/execute/vector_ups.rs` (UPS mode table)
- Modify: `src/interpreter/execute/cascade.rs` (gate on `has_cascade_link`)
- Modify: `src/device/arch_handle.rs` (new accessor(s) if needed for `has_cascade_link` gating pattern)

**Goal:** Move four bounded data artifacts from execute/ to archspec, each with a drift-detection test and its own commit.

- [ ] **Step 1: `RoundingMode` dedup (pre-verified safe)**

`vector_srs.rs:34-87` and `vector_float.rs:34-70` define the same 10-variant enum plus an identical `from_raw()` method. Verified identical 2026-04-21 apart from cosmetic doc-comment variation. Merge:

1. Create `crates/xdna-archspec/src/aie2/rounding.rs` (or extend an existing SRS-adjacent module if one exists -- check `ls crates/xdna-archspec/src/aie2/` first):

```rust
//! Hardware rounding modes for the SRS instruction and bf16 conversion.
//!
//! The mode index values match the AIE2 hardware encoding in the
//! configuration word. Valid indices are 0-3 and 8-13 (indices 4-7
//! are reserved).

/// Hardware rounding modes for the SRS instruction and bf16 conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    /// Mode 0: Floor -- truncate toward negative infinity.
    /// Discards all fractional bits. Equivalent to arithmetic right shift.
    Floor = 0,
    /// Mode 1: Ceiling -- round toward positive infinity.
    /// Adds 1 if any discarded bits are nonzero and value is not already exact.
    Ceil = 1,
    /// Mode 2: Symmetric floor -- round toward zero (positive) or away (negative).
    /// Sign-dependent: positive values truncate, negative values round away from zero.
    SymFloor = 2,
    /// Mode 3: Symmetric ceiling -- round away from zero (positive) or toward (negative).
    /// Opposite of SymFloor.
    SymCeil = 3,
    /// Mode 8: Round half toward negative infinity.
    /// At the exact halfway point, rounds toward -inf; otherwise rounds to nearest.
    NegInf = 8,
    /// Mode 9: Round half toward positive infinity.
    /// At the exact halfway point, rounds toward +inf; otherwise rounds to nearest.
    PosInf = 9,
    /// Mode 10: Round half toward zero (symmetric).
    /// At the exact halfway point, rounds toward zero; otherwise rounds to nearest.
    SymZero = 10,
    /// Mode 11: Round half away from zero (symmetric).
    /// At the exact halfway point, rounds away from zero; otherwise rounds to nearest.
    SymInf = 11,
    /// Mode 12: Convergent rounding to even (IEEE 754 banker's rounding).
    /// At the exact halfway point, rounds to the nearest even value.
    ConvEven = 12,
    /// Mode 13: Convergent rounding to odd.
    /// At the exact halfway point, rounds to the nearest odd value.
    ConvOdd = 13,
}

impl RoundingMode {
    /// Convert a raw hardware mode index to a `RoundingMode`.
    /// Returns `None` for reserved indices (4-7, 14-15).
    pub fn from_raw(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Floor),
            1 => Some(Self::Ceil),
            2 => Some(Self::SymFloor),
            3 => Some(Self::SymCeil),
            8 => Some(Self::NegInf),
            9 => Some(Self::PosInf),
            10 => Some(Self::SymZero),
            11 => Some(Self::SymInf),
            12 => Some(Self::ConvEven),
            13 => Some(Self::ConvOdd),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_mode_from_raw_valid_indices() {
        for (raw, expected) in [
            (0, RoundingMode::Floor),
            (1, RoundingMode::Ceil),
            (2, RoundingMode::SymFloor),
            (3, RoundingMode::SymCeil),
            (8, RoundingMode::NegInf),
            (9, RoundingMode::PosInf),
            (10, RoundingMode::SymZero),
            (11, RoundingMode::SymInf),
            (12, RoundingMode::ConvEven),
            (13, RoundingMode::ConvOdd),
        ] {
            assert_eq!(RoundingMode::from_raw(raw), Some(expected));
        }
    }

    #[test]
    fn rounding_mode_from_raw_reserved_indices_return_none() {
        for raw in [4, 5, 6, 7, 14, 15, 200, 255] {
            assert_eq!(RoundingMode::from_raw(raw), None);
        }
    }
}
```

2. Wire the module: add `pub mod rounding;` to `crates/xdna-archspec/src/aie2/mod.rs`.

3. Remove the two local definitions from `vector_srs.rs` (lines 34-108 -- enum + impl) and `vector_float.rs` (lines 34-91). Replace each with `use xdna_archspec::aie2::rounding::RoundingMode;`.

4. `cargo test -p xdna-archspec --lib`: expect 302 passed (+2 from new tests). `cargo test --lib`: expect 2684 unchanged.

Commit:
```bash
git add crates/xdna-archspec/src/aie2/rounding.rs \
        crates/xdna-archspec/src/aie2/mod.rs \
        src/interpreter/execute/vector_srs.rs \
        src/interpreter/execute/vector_float.rs
git commit -m "refactor: consolidate RoundingMode to archspec::aie2::rounding

Deduplicates the identical 10-variant RoundingMode enum + from_raw()
impl from vector_srs.rs and vector_float.rs into a single definition
under xdna_archspec::aie2::rounding. Verified structurally identical
pre-dedup (only cosmetic doc-comment variation). Both source files
now use the shared archspec definition.

Part of Subsystem 7 Part B (audit item 3).

Generated using Claude Code."
```

- [ ] **Step 2: Matmul geometry tables**

`vector_config.rs` has `DENSE_GEOMETRY_TABLE` and `SPARSE_GEOMETRY_TABLE`. Find them:

```bash
grep -nE "DENSE_GEOMETRY_TABLE|SPARSE_GEOMETRY_TABLE" src/interpreter/execute/vector_config.rs
```

Move the table constants to `crates/xdna-archspec/src/aie2/matmul.rs` (new module). Wire `pub mod matmul;` in `aie2/mod.rs`. Update `vector_config.rs` to `use xdna_archspec::aie2::matmul::{DENSE_GEOMETRY_TABLE, SPARSE_GEOMETRY_TABLE};`. Add a drift-detection test in the new archspec module locking the table to the exact byte values before/after the move.

Commit: `refactor: matmul geometry tables moved to archspec::aie2::matmul`

- [ ] **Step 3: UPS mode table**

`vector_ups.rs::ups_mode()` contains a 4-entry valid type-pair table. Find it:

```bash
grep -nB2 -A15 "fn ups_mode" src/interpreter/execute/vector_ups.rs
```

Move the table to `crates/xdna-archspec/src/aie2/ups.rs` (new module). Update `vector_ups.rs` to read the table via the archspec module. Drift test.

Commit: `refactor: UPS mode table moved to archspec::aie2::ups`

- [ ] **Step 4: Cascade data + `has_cascade_link` feature flag**

This is the one migration with real plumbing work. Three sub-parts, all in one commit:

a. **Add the flag to archspec's ProcessorModel** (or wherever processor-level metadata lives). Grep:

```bash
grep -rn "pub struct ProcessorModel\|pub struct ArchProcessorMetadata" crates/xdna-archspec/src/
```

Add a `pub has_cascade_link: bool` field to the struct. Add `pub const CASCADE_WORDS: usize = 6;` alongside (or as a method). Default it to `true` for AIE2/AIE2P constructors.

b. **Wire `has_cascade_link` through `from_arch_model`** or whatever constructor path ModelConfig uses. AIE2 arches get `true`. AIE1 gets `false` (though we never construct AIE1 here today, set the default correctly for when we do).

c. **Add an accessor** if the existing processor-model accessor path doesn't already cover it. Probably `arch_handle::processor_model().has_cascade_link` works directly if an accessor already exists; otherwise extend.

d. **Gate `cascade.rs` call sites** on `has_cascade_link`. Find them:

```bash
grep -nE "cascade|Cascade" src/interpreter/execute/cascade.rs | head -20
```

At each cascade operation entry point, early-return (no-op) when the flag is false. The concrete cascade operations stay unchanged for AIE2 (flag is true).

e. **Remove the `CASCADE_WORDS = 6` literal from `cascade.rs`**; read via archspec instead.

Add a test in archspec: `has_cascade_link_true_for_aie2_family`.

Expected test-count change: archspec +1-3 (drift tests + feature-flag test), xdna-emu unchanged.

Commit:
```bash
git commit -m "refactor(archspec): cascade data + has_cascade_link feature flag

Moves CASCADE_WORDS=6 from cascade.rs to archspec and adds a new
has_cascade_link: bool flag to ProcessorModel. Wires the flag through
from_arch_model (AIE2/AIE2P default true; AIE1 default false). Gates
cascade dispatch in src/interpreter/execute/cascade.rs on the flag,
so a future AIE1 port gets no-op cascade handlers without touching
execute-layer code.

Part of Subsystem 7 Part B (audit item 6).

Generated using Claude Code."
```

- [ ] **Step 5: Verify Task 4 close**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -20
```

Expected: all tests green, bridge smoke PASS. Four commits for Task 4.

---

### Task 5: `vector_permute.rs` tables wholesale move

**Files:**
- Create: `crates/xdna-archspec/src/aie2/permute.rs` (~3200 LOC, static data + enum)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (add `pub mod permute;`)
- Modify: `src/interpreter/execute/vector_permute.rs` (remove local tables + enum; add imports)

**Goal:** Move `SHUFFLE_ROUTING: [[u8; 64]; 48]` (3072 bytes of hardware-probed routing data) and the 26-variant `MacPermuteMode` enum from `vector_permute.rs` into `xdna_archspec::aie2::permute`. This is the second-largest data migration after `vmac_routing.rs`.

- [ ] **Step 1: Locate the tables + enum in `vector_permute.rs`**

```bash
grep -nE "SHUFFLE_ROUTING|pub enum MacPermuteMode|^pub enum MacPermuteMode" src/interpreter/execute/vector_permute.rs
```

Note the line ranges for:
- The `SHUFFLE_ROUTING: [[u8; 64]; 48]` static
- The `MacPermuteMode` enum (all 26 variants)
- Any associated `impl` blocks (e.g., `MacPermuteMode::from_raw`, `as_u8`, etc.)

- [ ] **Step 2: Create `crates/xdna-archspec/src/aie2/permute.rs`**

Open a new file and transplant the `SHUFFLE_ROUTING` static + `MacPermuteMode` enum + attached impl blocks verbatim. Preserve existing doc comments. Add a module header:

```rust
//! AIE2 permute/shuffle data: SHUFFLE_ROUTING lookup table and
//! MacPermuteMode enum.
//!
//! Moved from src/interpreter/execute/vector_permute.rs as part of
//! Subsystem 7 Part B (audit item 2). Pure data -- no algorithmic
//! content. The shuffle routing table is hardware-probed; the
//! MacPermuteMode enum mirrors the 26 valid hardware permute modes.

// ... (transplanted content)
```

Add drift-detection tests at the end of the new module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_routing_table_size() {
        assert_eq!(SHUFFLE_ROUTING.len(), 48);
        assert_eq!(SHUFFLE_ROUTING[0].len(), 64);
    }

    #[test]
    fn mac_permute_mode_variant_count() {
        // Locks at 26 variants; if this changes, audit the migration.
        // Count MacPermuteMode enumerations the compiler knows about.
        // This test is a placeholder until a better variant-count check
        // is available via #[derive(EnumCount)] or similar.
        // Substitute with actual variant enumeration if the enum derives
        // strum::EnumCount.
        let _smoke: MacPermuteMode = MacPermuteMode::from_raw(0).unwrap();
    }
}
```

(Adjust tests to match how the codebase actually locks table invariants elsewhere -- see `aie2_topology_matches_generated_constants` in Subsystem 5 for the pattern. If the variant enum doesn't derive `EnumCount`, do a manual assertion listing all expected variants.)

- [ ] **Step 3: Wire the module in `aie2/mod.rs`**

Add `pub mod permute;` after the other model-ish modules in `aie2/mod.rs`.

- [ ] **Step 4: Remove local definitions from `vector_permute.rs`**

Delete the `SHUFFLE_ROUTING` static, the `MacPermuteMode` enum, and all attached `impl` blocks. Add at the top of `vector_permute.rs`:

```rust
use xdna_archspec::aie2::permute::{SHUFFLE_ROUTING, MacPermuteMode};
```

- [ ] **Step 5: Build + test**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -10
# Expected: clean build.
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: 2684 passed (unchanged; pure move).
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: +2 from drift tests.
```

If the build fails with "unresolved import" errors, trace each back -- it's likely a transplanted impl that referenced a private type in vector_permute.rs. Either make the referenced type `pub` (if it's appropriate for archspec) or split: leave the offending impl in vector_permute.rs and reach back through archspec for the data parts.

- [ ] **Step 6: Bridge smoke + commit**

```bash
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -20
```
Expected: PASS on both Chess and Peano.

Commit:
```bash
git add crates/xdna-archspec/src/aie2/permute.rs \
        crates/xdna-archspec/src/aie2/mod.rs \
        src/interpreter/execute/vector_permute.rs
git commit -m "refactor: move SHUFFLE_ROUTING + MacPermuteMode to archspec::aie2::permute

Wholesale transplant of ~3200 LOC of hardware-probed permute data
from src/interpreter/execute/vector_permute.rs to
xdna_archspec::aie2::permute. Pure data move -- no algorithmic
changes. Includes SHUFFLE_ROUTING lookup table (48 x 64 bytes) and
the 26-variant MacPermuteMode enum.

Part of Subsystem 7 Part B (audit item 2).

Generated using Claude Code."
```

One commit for Task 5.

---

### Task 6: `vmac_routing.rs` wholesale move

**Files:**
- Create: `crates/xdna-archspec/src/aie2/vmac/` (directory)
- Create: `crates/xdna-archspec/src/aie2/vmac/mod.rs` (or `routing.rs`) with the transplanted content
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (add `pub mod vmac;`)
- Delete: `src/interpreter/execute/vmac_routing.rs` (2862 LOC)
- Modify: `src/interpreter/execute/vmac_hw.rs` (update `include!` or switch to `use`)
- Modify: `src/interpreter/execute/mod.rs` if it declares `mod vmac_routing;` there

**Goal:** Move the entire 234 KB static crossbar routing data file wholesale from execute/ to archspec. Consumers (`vmac_hw.rs` line 15 `include!` + uses of `eval_prmx`/`eval_prmy`) update to read through archspec module path.

- [ ] **Step 1: Inspect the current consumer pattern**

```bash
head -30 /home/triple/npu-work/xdna-emu/src/interpreter/execute/vmac_hw.rs
grep -nE "vmac_routing|eval_prmx|eval_prmy" src/interpreter/execute/vmac_hw.rs
```

Identify exactly how `vmac_hw.rs` pulls in `vmac_routing.rs`: is it `include!("vmac_routing.rs")`, or `mod vmac_routing;` in `execute/mod.rs` + `use`, or something else? The audit reported it's `include!`'d from line 15 of `vmac_hw.rs`; verify.

- [ ] **Step 2: Create the archspec destination**

Create `crates/xdna-archspec/src/aie2/vmac/mod.rs`:

```rust
//! AIE2 VMAC crossbar routing.
//!
//! Moved from src/interpreter/execute/vmac_routing.rs as part of
//! Subsystem 7 Part B (audit item 1). Pure static data probed from
//! the AMD C++ ISS: 789 active m-bits, 15808 route entries (PRMX
//! tables), 26 x 512 = 13312 Y-route entries (PRMY tables).
//!
//! The two consumer functions `eval_prmx` and `eval_prmy` are
//! re-exported here for direct use from `vmac_hw.rs`.

pub mod routing;

pub use routing::{eval_prmx, eval_prmy};
```

Create `crates/xdna-archspec/src/aie2/vmac/routing.rs` and COPY the contents of `src/interpreter/execute/vmac_routing.rs` into it verbatim (including the auto-generated-file comment banner at the top). Any `pub fn eval_prmx`, `pub fn eval_prmy` (or whatever the exported symbols are) should remain `pub`.

- [ ] **Step 3: Wire the vmac module in `aie2/mod.rs`**

Add `pub mod vmac;` to `crates/xdna-archspec/src/aie2/mod.rs` in the appropriate alphabetical position.

- [ ] **Step 4: Update `vmac_hw.rs` consumer**

Replace the `include!` (or `mod vmac_routing; use vmac_routing::*;`) with an import from archspec:

```rust
// Replace line 15's `include!("vmac_routing.rs");` or equivalent with:
use xdna_archspec::aie2::vmac::{eval_prmx, eval_prmy};
```

(Or whatever symbol set the file actually exports. Verify first.)

- [ ] **Step 5: Delete `src/interpreter/execute/vmac_routing.rs`**

```bash
git rm src/interpreter/execute/vmac_routing.rs
```

If `execute/mod.rs` declared `mod vmac_routing;`, remove that line.

- [ ] **Step 6: Build + test**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -10
# Expected: clean build.
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: 2684 passed (unchanged; pure move).
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: +1 from drift-detection smoke test.
```

Add a drift-detection smoke test in `crates/xdna-archspec/src/aie2/vmac/mod.rs` (or a sibling tests file):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_prmx_callable() {
        // Smoke test: call eval_prmx with a known input. This locks
        // the symbol's existence + calling convention across the
        // wholesale move; content-level drift is caught by the
        // existing tests in vmac_hw.rs that exercise the full pipeline.
        let input = [0u8; 16];
        let _output = eval_prmx(&input, 0);
        // No assertion on value; the test exercises the code path.
    }
}
```

(Adjust to match the actual signatures of `eval_prmx`/`eval_prmy`.)

- [ ] **Step 7: Bridge smoke + commit**

```bash
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -20
```
Expected: PASS.

Commit:
```bash
git add crates/xdna-archspec/src/aie2/vmac/ \
        crates/xdna-archspec/src/aie2/mod.rs \
        src/interpreter/execute/vmac_hw.rs \
        src/interpreter/execute/
# The removal of vmac_routing.rs is already staged via `git rm`.
git commit -m "refactor: move vmac_routing.rs wholesale to archspec::aie2::vmac

Transplants 2862 lines of hardware-probed AIE2 VMAC crossbar routing
data from src/interpreter/execute/vmac_routing.rs (234K pure data,
zero algorithm) into xdna_archspec::aie2::vmac::routing with
eval_prmx + eval_prmy re-exported from aie2::vmac. vmac_hw.rs
switches from include! to use xdna_archspec::aie2::vmac::*.

Part of Subsystem 7 Part B (audit item 1). Largest single migration
of the subsystem.

Generated using Claude Code."
```

One commit for Task 6.

---

### Task 7: Completion + gate + tag

**Files:**
- Modify: `docs/arch/subsys7-audit.md` (fill `## Completion` section)
- Modify: `docs/arch/isa-execute-model.md` (expand all placeholders with content)
- Modify: `NEXT-STEPS.md` (move Subsystem 8 to "up next")

**Goal:** Close Part B and land the subsystem tag. This task has no source-code changes -- it documents the state, runs the full gate, and tags.

- [ ] **Step 1: Fill in `docs/arch/isa-execute-model.md`**

Replace each `(Filled in at Part B close.)` placeholder with content. The design-note template is structured as:

- **What lives where.** Summarize the archspec landing: `xdna_archspec::aie2::rounding` (RoundingMode), `aie2::permute` (SHUFFLE_ROUTING + MacPermuteMode), `aie2::vmac` (crossbar data + eval_prmx/eval_prmy), `aie2::matmul` (geometry tables), `aie2::ups` (UPS mode table), extended `ProcessorModel` (has_cascade_link, CASCADE_WORDS, control register IDs, latency values, PROC_BUS_*), and unchanged: `crate::isa_execute` (empty IsaExecutor trait anchor).

- **Trait surface.** Explain the empty trait: its purpose is as an anchor for future seams, not as a current dispatch point. Reference the audit's Approach A conclusion.

- **The shape-vs-values rule, applied to ISA execute.** ~2-3 paragraphs explaining why execute landed at values + feature flags rather than shapes. Cite `has_cascade_link` as the prototypical feature flag and `RoundingMode::from_raw` as the prototypical data-table consolidation.

- **What would AIE1 look like?** Expand the audit's ~100-word projection into ~300 words. Per the audit: narrower vectors/accumulators (parameterized by `VEC_BYTES`), checkerboard memory (already reads `IS_CHECKERBOARD`), no cascade (`has_cascade_link: false`), different VMAC pipeline (separate AIE1 matmul module, AIE2 untouched), different rounding-mode set (new AIE1 variants in the archspec enum or separate enum + feature-gated handlers).

- **Alternatives rejected.** Four entries matching the spec's alternatives section: Approach A-as-different-landing, Approach C (full trait), pre-audit commitment, sub-subsystem decomposition.

No commit yet (combine with Step 2 below).

- [ ] **Step 2: Fill in `## Completion` section of `docs/arch/subsys7-audit.md`**

Replace `(Filled in at the end of Subsystem 7, in the Part B final task.)` with a completion section structured like Subsystem 5's audit completion:

```markdown
## Completion

**Subsystem 7 closed at:** <tag phase1-subsys-isa-execute, commit SHA>
**Date:** <fill in>

### Commits landed (since phase1-subsys-stream-switch)

<output of `git log --oneline phase1-subsys-stream-switch..HEAD`>

### Test counts

- `cargo test --lib`: <N passed; 0 failed; 5 ignored>
- `cargo test -p xdna-archspec --lib`: <M passed; 0 failed; 2 ignored>
- `cargo build --release`: clean
- Bridge full run: <Chess / Peano summary, tee log path>
- ISA test: <result, tee log path>

### Success criteria check

- [x] Approach A landed (zero IsaExecutor trait methods).
- [x] 12 audit-list data migrations landed (list with commit refs).
- [x] Execute algorithms untouched for AIE2; arch-specific data lives in archspec.
- [x] AIE1 / AIE2P port is now "populate archspec data" -- no execute/*.rs edits.
- [x] Tests green at every commit. Bridge smoke green at every task close.

### Net delta

- xdna-emu LOC: <delta>
- archspec LOC: <delta>
- Total: ~<delta> LOC moved, 0 LOC rewritten.

### Follow-ups flagged for AIE1-landing pass (not in Subsystem 7 scope)

- (list anything surfaced during migration that's a genuinely-separate workstream)
```

- [ ] **Step 3: Update `NEXT-STEPS.md`**

Follow the pattern from Subsystem 5's update:
- Update "Last updated" to today's date.
- Update "Latest tag" to `phase1-subsys-isa-execute`.
- In the Phase 1b pass-order table, mark Subsystem 7 **Done** with one-line summary. Mark Subsystem 8 (Parser) **Up next**.
- Rewrite the "How to Pick Up Subsystem 7 (ISA Execute)" section as "How to Pick Up Subsystem 8 (Parser)" with the shaping questions the spec will need.
- Update the test-count expectations in the Useful Commands section.

Commit:
```bash
git add docs/arch/subsys7-audit.md docs/arch/isa-execute-model.md NEXT-STEPS.md
git commit -m "docs: Subsystem 7 completion + NEXT-STEPS update

Subsystem 7 (ISA Execute) complete. Approach A landed: empty
IsaExecutor trait anchor + 12 data migrations to archspec,
no trait methods, execute algorithms untouched for AIE2. AIE1
and AIE2P ports now require only archspec additions.

Generated using Claude Code."
```

- [ ] **Step 4: Run the full-tree gate (sequential)**

Per CLAUDE.md: never run two hardware test suites concurrently. Run bridge first, then ISA.

```bash
cd /home/triple/npu-work/xdna-emu
export PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH

# Rebuild FFI (critical -- bridge reads the .so, and stale ones
# produce phantom regressions)
cargo clean -p xdna-emu-ffi
cargo build -p xdna-emu-ffi

# Unit + archspec
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3

# Release build
cargo build --release 2>&1 | tail -5

# Full bridge (sequential)
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-subsys7.log
# Wait for complete; capture Chess + Peano summaries + any failures.

# ISA (after bridge completes)
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys7.log
# Expected: 4815/4815.
```

Expected outcomes:
- `cargo test --lib`: 2684 + any drift tests added during Part B (should be small; mostly archspec-side). Zero failures.
- `cargo test -p xdna-archspec --lib`: 300 + significant growth (drift tests for each new archspec data module + the has_cascade_link test + RoundingMode tests). Zero failures.
- `cargo build --release`: clean.
- Bridge: same pass/fail profile as `phase1-subsys-stream-switch` (Chess PASS on 62+, Peano PASS on 53+, known pre-existing deadlocks still pre-existing).
- ISA: 4815/4815.

If any of these fail unexpectedly, do NOT tag. Diagnose and fix before tagging.

- [ ] **Step 5: Tag `phase1-subsys-isa-execute`**

```bash
git tag phase1-subsys-isa-execute
git log --oneline phase1-subsys-stream-switch..phase1-subsys-isa-execute | head -30
```

Report the tag SHA.

- [ ] **Step 6: Amend the Completion section with final numbers**

Re-open `docs/arch/subsys7-audit.md` and fill in the actual SHA, date, commit count, test counts, bridge/ISA results into the Completion section's placeholders from Step 2. Commit with:

```bash
git commit --amend --no-edit
```

(Amending is acceptable here ONLY because the completion-doc commit just landed; the subsystem tag has NOT been pushed yet. If any ambiguity, just make a new commit.)

Alternatively, make a new commit:

```bash
git add docs/arch/subsys7-audit.md
git commit -m "docs: fill Subsystem 7 Completion section with final numbers"
```

Tag does not need to be re-issued if the tag was placed on the earlier commit -- the Completion-fill commit sits at HEAD after the tag; the tag still points at the working state.

Preferred: make Step 5 tag AFTER Step 6 completes, so the tag points at the final completion-fill commit.

---

## Part B Completion

After Task 7 Step 5 (or Step 6 followed by Step 5, per the preferred ordering note), Subsystem 7 is closed. State:

- Tag `phase1-subsys-isa-execute` placed.
- `cargo test --lib` ≈ 2684 + drift tests; archspec ≈ 300 + significant drift tests.
- Bridge full run captured at `/tmp/claude-1000/bridge-subsys7.log`; ISA at `/tmp/claude-1000/isa-subsys7.log`.
- Audit and design-note docs fully populated.
- `NEXT-STEPS.md` points at Subsystem 8 (Parser) as the next work item.

Expected total commit count across Part A + Part B: ~25-30.

**Ready to pause / hand off / proceed to Subsystem 8.**

---

## Known Pre-Existing Failures (carry through, do not block)

Documented in `NEXT-STEPS.md`:
- `bd_chain_repeat_on_memtile` EMU deadlock in bridge suite.
- `cargo test -p xdna-archspec --lib`: `test_full_parse_all_devices`
  failure FIXED in Subsystem 3; should remain fixed.
- Peano bridge EMU timeouts on `dma_task_large_linear` and
  `objectfifo_repeat/init_values_repeat`.
- Generated-file warnings (unused constants in `gen_aiert_*.rs`).

---

## Self-Review Checklist

Run before closing Part A:

- [ ] `docs/arch/subsys7-audit.md` exists, all six area sections have
      content, closing summary has the three subsections.
- [ ] `docs/arch/isa-execute-model.md` scaffold exists (placeholders
      for Part B close are fine).
- [ ] `crates/xdna-archspec/src/isa_execute/mod.rs` exists with
      empty `IsaExecutor` trait.
- [ ] `crates/xdna-archspec/src/aie2/isa_execute_model.rs` exists
      with `Aie2IsaExecutor` + `AIE2_ISA_EXECUTOR` + 2 tests.
- [ ] `crates/xdna-archspec/src/lib.rs` has `pub mod isa_execute;`.
- [ ] `crates/xdna-archspec/src/aie2/mod.rs` has `pub mod isa_execute_model;`.
- [ ] `crates/xdna-archspec/src/runtime.rs` has `isa_executor` trait
      method + impl + dispatch test.
- [ ] `src/device/arch_handle.rs` has `isa_executor()` accessor +
      `OnceLock`.
- [ ] `cargo test --lib` = 2684/0/5.
- [ ] `cargo test -p xdna-archspec --lib` = 300/0/2.
- [ ] Bridge smoke `--no-hw -v add_one_cpp_aiecc` = green.
- [ ] ~11 commits since `phase1-subsys-stream-switch`.

If any item is missing or incorrect, fix before declaring Part A
complete.
