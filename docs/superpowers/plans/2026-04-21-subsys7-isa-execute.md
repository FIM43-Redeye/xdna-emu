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

## Part B -- Authored After Task 1 Audit Lands

Part B tasks are authored as a plan amendment once Task 1 (the audit)
commits. The amendment adds concrete tasks starting at `### Task 3:`
and ending with a final gate + tag task.

**Expected Part B task shape** (not commitments; the audit drives the
exact list):

- **Tasks 3-5: Data migrations.** One task per data-migration
  destination from the audit's closing summary. Expected to include:
  - Wholesale move of `src/interpreter/execute/vmac_routing.rs` into
    `xdna_archspec::aie2::vmac::routing` with consumer update in
    `vmac_hw.rs`. Drift-detection test.
  - AIE2 latency values from `timing/latency.rs` -> archspec
    (extending `ProcessorModel` or new `aie2::timing` module).
  - Memory-hierarchy constants from `memory/mod.rs` if the audit
    surfaces any outside archspec.
  - Any other data-migration rows the audit surfaces.

- **Tasks 6-(N-1): Trait-method additions + call-site migrations.**
  One task per trait method added to `IsaExecutor`, each task:
  (a) adds the method signature to `isa_execute/mod.rs`;
  (b) adds the `Aie2IsaExecutor` impl in `aie2/isa_execute_model.rs`;
  (c) migrates the call sites in `src/interpreter/execute/*.rs` to
      use `arch_handle::isa_executor().<method>(...)`;
  (d) adds before-after equivalence tests where the call site did
      not already have coverage.

  If the audit's closing summary concludes **no trait methods**, this
  block is empty and Part B consists only of data migrations + the
  final gate task.

- **Task N: Completion + gate + tag.** Fills in
  `docs/arch/isa-execute-model.md` (expanding the audit's trait
  rationale into ~300 words + AIE1 projection), fills in the audit's
  `## Completion` section with commit list and verification results,
  updates `NEXT-STEPS.md` (move Subsystem 8 to "up next"), runs the
  full-tree gate:
  - `cargo test --lib`
  - `cargo test -p xdna-archspec --lib`
  - `cargo build --release`
  - `cargo build -p xdna-emu-ffi`
  - `./scripts/emu-bridge-test.sh` (full, ~30 min, tee to
    `/tmp/claude-1000/bridge-subsys7.log`)
  - `./scripts/isa-test.sh` (~10 min, tee to
    `/tmp/claude-1000/isa-subsys7.log`)
  - sequential, never concurrent (per CLAUDE.md rule)

  Finally, tags `phase1-subsys-isa-execute`.

**Amendment protocol (repeated for clarity):**

1. After Task 2 commits, read `docs/arch/subsys7-audit.md` in full,
   especially the closing summary.
2. Edit this file, replacing this "Part B -- Authored After..."
   section with concrete `### Task 3: ...`, `### Task 4: ...`, etc.
3. Commit as `docs(plan): Subsys 7 Part B tasks from audit findings`.
4. Resume execution from Task 3.

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
