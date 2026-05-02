# Subsystem 8 -- Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the `src/parser/*` surface (XCLBIN, AIE Partition, CDO,
ELF -- 2,299 LOC across 5 files) against AIE1/AIE2P divergence
evidence; migrate arch-specific data to `xdna-archspec`; decide the
`BinaryLoader` trait question; split `cdo.rs` into framing /
syntax / semantics layers; introduce a `DeviceOp` vocabulary to
clean the parser/device-state boundary; dedupe ELF loading across
5 call sites; add rich diagnostics + test fixtures + control-packet
parser alignment; tag `phase1-subsys-parser-arch`,
`phase1-subsys-parser-coupling`, and `phase1-subsys-parser-ergonomics`.

**Architecture:** Three stages, each tagged. 8a audits + migrates
arch data + lands the trait decision (audit-driven, Subsystem 7
pattern). 8b restructures the CDO layer in two halves around a
bisect-safe intermediate checkpoint: Half 1 renames `CdoCommand` ->
`CdoRaw` and splits `cdo.rs` into framing/syntax/semantics (no
behavior change); Half 2 introduces the `DeviceOp` vocabulary and
moves the `device/state` boundary onto it. **Explicit user gate
between 8a and 8b's Half 2:** user reviews the refined `DeviceOp`
proposal (audit output) before Half 2 starts. 8c lands diagnostics,
test fixtures, ELF deduplication, and control-packet alignment as
four independent sub-deliverables.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate,
`smallvec`, `thiserror`/`anyhow` for diagnostics, `zerocopy` for
existing byte parsing. Evidence sources for the 8a audit: aie-rt
per-arch headers (`../aie-rt/driver/src/`), AM025 register DB JSON,
llvm-aie TableGen (if ELF relocation shapes turn out to vary), real
XCLBINs on disk under `mlir-aie/build/test/npu-xrt/**/*.xclbin` for
CDO command frequency counts, existing codebase grep.

**Spec:** [docs/superpowers/specs/2026-04-23-subsys8-parser-design.md](../specs/2026-04-23-subsys8-parser-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

**Prior subsystem:** Subsystem 7 ISA Execute. Tag `phase1-subsys-isa-execute` referenced in docs but not yet applied to the repo (pending Subsystem 7 close-out commit; see "Pre-plan housekeeping" below).

---

> **Sweep-as-of 2026-05-01:** Subsystem 8 completed across three tags: phase1-subsys-parser-arch, phase1-subsys-parser-coupling, phase1-subsys-parser-ergonomics. Plan was split into stages 8a-8c during execution (see commit 79afb3c 'Subsystem 8 complete -- Phase 1 done'). Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Pre-plan housekeeping

Before Task 1, three loose ends from Subsystem 7's close need committing. These are not part of Subsystem 8 work; they just clear the working tree so Subsystem 8 starts from a clean state.

- `NEXT-STEPS.md` modifications (points at Subsystem 8).
- `docs/arch/isa-execute-model.md` modifications (Subsystem 7 design note polish).
- `docs/arch/subsys7-audit.md` modifications (Subsystem 7 audit close).

- [x] **Step: Commit the Subsystem 7 close-out polish**

Run:
```bash
git status --short
# Expect (among possibly other entries):
#  M NEXT-STEPS.md
#  M docs/arch/isa-execute-model.md
#  M docs/arch/subsys7-audit.md

git add NEXT-STEPS.md docs/arch/isa-execute-model.md docs/arch/subsys7-audit.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 7 close-out polish

Final-state edits to NEXT-STEPS.md, isa-execute-model.md, and
subsys7-audit.md that were not in the Subsystem 7 commit series.
Pointing the recovery doc at Subsystem 8 and closing the audit /
design note.

Generated using Claude Code.
EOF
)"
```

Then tag Subsystem 7 (the tag was referenced in docs but never applied):

```bash
git tag phase1-subsys-isa-execute HEAD
# Confirm:
git tag | grep phase1-subsys
# Expect phase1-subsys-isa-execute in the list.
```

Rationale: NEXT-STEPS.md claims `phase1-subsys-isa-execute` is the latest tag; this makes that claim real.

---

## Three-stage structure

Subsystem 8 lands in three stages with three tags. Each stage has its own rollback posture per the spec's §"Rollback posture"; Half 1 of 8b is the intermediate bisect anchor documented there.

**Tags, in order:**

1. `phase1-subsys-parser-arch` -- end of Stage 8a. Audit + data migrations + `BinaryLoader` trait decision + AIE1 design note.
2. `phase1-subsys-parser-coupling` -- end of Stage 8b. CDO split + `DeviceOp` boundary.
3. `phase1-subsys-parser-ergonomics` -- end of Stage 8c. Diagnostics + fixtures + ELF dedup + control-packet alignment. Phase 1 complete.

**Gate between 8a and 8b (Half 2):** After Tasks 1--6 commit and `phase1-subsys-parser-arch` tags, **pause execution** and hand the refined `DeviceOp` proposal (audit section 4 output) to the user for review. Task 10 (Half 2 start) is gated on explicit user confirmation. See §"Amendment protocol" below -- Task 10's detailed steps are authored *after* the audit lands and the user signs off.

---

## Scope Note

### Stage 8a (Tasks 1--6)

Audit-first pass matching Subsystems 3--7. Produces:
- `docs/arch/subsys8-audit.md` with nine sections (per spec §"Audit methodology (8a)").
- Data migrations surfaced by the audit (likely candidates: CDO opcode identity, ELF machine type constants, XCLBIN `SectionKind` AIE-membership table). Specified in Task 4 via amendment after Task 3's audit commits.
- `BinaryLoader` trait decision (populated / empty anchor / no trait) landed per Task 5.
- `docs/arch/binary-loader-model.md` design note with the "what would AIE1 look like?" answer.
- **Refined `DeviceOp` proposal** (spec text update within `2026-04-23-subsys8-parser-design.md`, §"`DeviceOp` vocabulary"). Becomes the ground truth for Stage 8b Half 2.
- Tag `phase1-subsys-parser-arch`.

### Stage 8b (Tasks 7--9, Half 1; Tasks 10--13, Half 2)

Structural-knot pass, two halves with an intermediate bisect-safe checkpoint.

**Half 1 (fully specifiable now):** Rename `CdoCommand` -> `CdoRaw`; split `cdo.rs` into `cdo/mod.rs` + `cdo/framing.rs` + `cdo/syntax.rs` + `cdo/semantics.rs` (semantics is pass-through). No behavior change. Half 1's closing commit is the **bisect anchor** for Half 2's rollback.

**Half 2 (partially specifiable; completed in amendment after user gate):** Introduce `DeviceOp` module. Rewrite `semantics::lower` to emit `DeviceOp` per the refined audit proposal. Migrate `device/state/cdo.rs` and `device/state/mod.rs` to consume `DeviceOp`. Tag `phase1-subsys-parser-coupling`.

### Stage 8c (Tasks 14--18)

Ergonomics pass. Four independent sub-deliverables:
- Task 14: Diagnostics (`ParseError` enum with `{offset, expected, got, hex_context}`).
- Task 15: Test fixtures (`XclbinBuilder`, `CdoBuilder`, `ElfBuilder` under `src/parser/testing/`).
- Task 16: ELF deduplication (single canonical `AieElf::load_into`; migrate 5 call sites).
- Task 17: Control-packet parser alignment (decision driven by Task 3 audit §7).
- Task 18: Stage close + tag `phase1-subsys-parser-ergonomics` + Phase 1 closeout (update `NEXT-STEPS.md`, note Phase 1 complete).

**Estimated task count:** 18 tasks (not counting pre-plan housekeeping). Estimated commits: ~30--40 (some tasks produce multiple atomic commits; some span one commit each).

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green. Baseline at HEAD: `2686 passed; 0 failed; 5 ignored`.
- `cargo test -p xdna-archspec --lib` green. Baseline at HEAD: `320 passed; 0 failed; 2 ignored`.
- `cargo build` green per commit. `cargo build --release` clean required before each stage tag, not every commit.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green as per-commit smoke during Stage 8b (FFI boundary-moving is highest risk). Bridge smoke runs after `cargo build -p xdna-emu-ffi` rebuilds the .so.
- Full bridge (`./scripts/emu-bridge-test.sh`) + ISA (`./scripts/isa-test.sh`) green before each stage tag.
- No commit introduces `TODO` / `FIXME` / `unimplemented!()` without an open-issue reference. Sanctioned exception: `unimplemented!("AIE1 BinaryLoader ...")` in any Stage 8a trait-dispatch accessor, mirroring Subsystems 3--7's pattern for other models.
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `feat:`, `refactor(archspec):`, `refactor(parser):`); no emoji; ends with `Generated using Claude Code.`.
- All work on `dev`. No merges to `master` during this plan.
- **Every `cargo` call** must have `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` prepended (tblgen needs llvm-config 21.x, not mlir-aie's 23.x; documented in NEXT-STEPS.md).

---

## File Structure

### Current layout (post-Subsystem 7, at HEAD)

```
xdna-emu/
├── src/
│   ├── parser/
│   │   ├── mod.rs                  # 18 LOC, re-exports
│   │   ├── xclbin.rs               # 456 LOC
│   │   ├── aie_partition.rs        # 372 LOC
│   │   ├── cdo.rs                  # 835 LOC -- splits in Stage 8b
│   │   └── elf.rs                  # 618 LOC
│   ├── device/
│   │   ├── state/
│   │   │   ├── mod.rs              # imports CdoCommand -- updates in 8b
│   │   │   ├── cdo.rs              # the apply_command dispatch -- updates in 8b
│   │   │   ├── compute.rs
│   │   │   ├── dispatch.rs
│   │   │   ├── effects.rs
│   │   │   ├── memtile.rs
│   │   │   └── tests.rs
│   │   └── control_packets/
│   │       ├── mod.rs
│   │       ├── parser.rs           # Task 3 §7 audit target
│   │       ├── processor.rs
│   │       ├── reassembler.rs
│   │       └── response.rs
│   ├── interpreter/
│   │   ├── decode/
│   │   │   ├── crossref.rs         # uses AieElf -- updates in 8c
│   │   │   └── decoder.rs          # uses AieElf -- updates in 8c
│   │   ├── engine/
│   │   │   └── coordinator.rs      # uses AieElf, MemoryRegion -- updates in 8c
│   │   └── test_runner.rs          # uses AieElf -- updates in 8c
│   ├── integration/
│   │   └── elfanalyzer.rs          # uses AieElf -- updates in 8c
│   ├── testing/
│   │   └── xclbin_suite.rs         # uses CdoCommand + AieElf
│   └── main.rs                     # uses CdoCommand + AieElf
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs
        ├── aie2/
        │   └── ...                 # dma_model, lock_model, stream_switch, isa_execute
        └── runtime.rs              # per-arch accessors
```

### Target layout (post-Stage 8a)

```
docs/
└── arch/
    ├── subsys8-audit.md            # NEW (Task 1+2): per-area audit findings
    └── binary-loader-model.md      # NEW (Task 1): design note scaffold -> filled Task 5
src/
└── parser/                         # unchanged; no structural moves yet
crates/xdna-archspec/src/
├── binary_loader/                  # NEW IF Task 5 lands populated/anchor trait
│   └── mod.rs
├── aie2/
│   ├── binary_loader_model.rs      # NEW IF Task 5 lands trait
│   └── ...                         # audit-driven: possibly cdo_opcodes, elf_machine
└── runtime.rs                      # + binary_loader() accessor if trait lands
```

### Target layout (post-Stage 8b, Half 1)

```
src/parser/
├── mod.rs                          # re-exports, unchanged function surface
├── xclbin.rs                       # unchanged
├── aie_partition.rs                # unchanged
├── cdo/
│   ├── mod.rs                      # public API, re-exports (replaces 835-LOC cdo.rs)
│   ├── framing.rs                  # NEW: header parse, version, command framing
│   ├── syntax.rs                   # NEW: emits CdoRaw (renamed from CdoCommand)
│   └── semantics.rs                # NEW: pass-through in Half 1; emits DeviceOp in Half 2
└── elf.rs                          # unchanged in 8b
```

### Target layout (post-Stage 8b, Half 2)

```
src/device/
├── ops.rs                          # NEW: DeviceOp enum
└── state/
    ├── cdo.rs                      # applies DeviceOp (was CdoCommand)
    └── mod.rs                      # imports DeviceOp (was CdoCommand)
```

### Target layout (post-Stage 8c)

```
src/parser/
├── error.rs                        # NEW (Task 14): ParseError enum
├── testing/
│   ├── mod.rs                      # NEW (Task 15): builders module
│   ├── xclbin_builder.rs           # NEW
│   ├── cdo_builder.rs              # NEW
│   └── elf_builder.rs              # NEW
├── elf.rs                          # refactored: AieElf::load_into canonical API
└── framing.rs                      # POSSIBLY NEW (Task 17, if control-packets share)
```

### Parallel changes in consumer sites

- `src/main.rs`: `use xdna_emu::parser::cdo::{find_cdo_offset, CdoRaw};` (Half 1) then `use xdna_emu::device::ops::DeviceOp;` (Half 2; main.rs is a pretty-printer, stays on `CdoRaw`).
- `src/device/state/cdo.rs`: `apply_command(&CdoCommand)` -> `apply_command(&CdoRaw)` (Half 1) -> `apply(&DeviceOp)` (Half 2).
- `src/testing/xclbin_suite.rs`: `CdoCommand` -> `CdoRaw` (Half 1 rename).
- Five ELF consumer files in 8c Task 16.

---

## Baseline to Preserve

Before Task 1, capture current numbers:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: test result: ok. 2686 passed; 0 failed; 5 ignored

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: test result: ok. 320 passed; 0 failed; 2 ignored

# Bridge smoke (quick sanity)
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -10
# Expected: both Chess and Peano PASS.
```

Full bridge + ISA baseline (captured in Task 1 Step 1, teed to /tmp/claude-1000/ so Stage tags have something to compare against):

```bash
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-baseline-bridge.log
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-baseline-isa.log
```

Known pre-existing failures (carry through, documented in NEXT-STEPS.md):
- `bd_chain_repeat_on_memtile` EMU bridge deadlock. Not blocking.
- `dma_task_large_linear` + `objectfifo_repeat/init_values_repeat` Peano EMU timeouts. Not blocking.

---

## Amendment protocol

This plan is 90% spec'd at write time. Three points require amendment after earlier tasks commit:

1. **After Task 3 commits (audit landed):**
   - Task 4's steps (data migrations) are authored based on the audit's migration-list output.
   - Task 5's steps (`BinaryLoader` trait decision) are authored based on the audit's trait-decision output.
   - Amendment commit: `docs(plan): Subsys 8 Task 4+5 steps from audit findings`.

2. **After Task 6 commits (Stage 8a tag, user gate):**
   - Task 10's steps (`DeviceOp` module creation) are authored based on the user-confirmed refined `DeviceOp` proposal.
   - Task 11's steps (semantics::lower rewrite) are authored based on the same.
   - Amendment commit: `docs(plan): Subsys 8 Task 10+11 steps from user-gated DeviceOp proposal`.

3. **After Task 3 commits but specific to Task 17:**
   - Task 17's detailed steps (control-packet alignment decision) depend on §7 of the audit. Can be authored as part of the Task 4+5 amendment above.

**Amendment mechanics:** re-enter `writing-plans` skill, or edit this file directly if mechanical. Update the relevant Task section's steps. Commit the plan as a documentation commit. Resume execution from the amended task.

---

## Task 1: Baseline capture + audit scaffold + design-note scaffold

**Files:**
- Read: (baselines only, no file edits)
- Create: `docs/arch/subsys8-audit.md`
- Create: `docs/arch/binary-loader-model.md`

**Goal:** Capture the starting state for regression checks. Scaffold the audit document with section placeholders (one per audit area in the spec's §9 methodology) and the design note with section placeholders the `BinaryLoader` decision will fill in.

- [x] **Step 1: Capture baseline numbers**

Run in sequence (heavy, ~15+ min for full bridge):

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3

# Full bridge + ISA baselines (these are the comparison targets for each stage tag)
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-baseline-bridge.log
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-baseline-isa.log
```

Note the final counts in a scratch file; they go into the audit doc in Step 2.

- [x] **Step 2: Create `docs/arch/subsys8-audit.md` scaffold**

Write this exact content:

```markdown
# Subsystem 8 -- Parser Audit

**Subsystem:** 8 of 8 (Phase 1b of the device-family refactor, final)
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Plan:** [../superpowers/plans/2026-04-23-subsys8-parser-plan.md](../superpowers/plans/2026-04-23-subsys8-parser-plan.md)

## Baseline (pre-subsystem, at phase1-subsys-isa-execute tag / HEAD)

- `cargo test --lib`: 2686 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 320 passed; 0 failed; 2 ignored
- `cargo build --release`: (capture from Step 1 scratch file)
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green
- Full bridge pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-bridge.log tail)
- ISA test pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-isa.log tail)

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite).
- `dma_task_large_linear` + `objectfifo_repeat/init_values_repeat` Peano EMU timeouts.

## Audit methodology

Per the spec, this audit runs nine sections. Sections 1--6 are per-area
deep dives (XCLBIN, AIE Partition, CDO syntax, CDO semantics, device-state
consumer, ELF consumer). Sections 7--9 are cross-cutting (control-packet
overlap, design note output, trait decision).

Per-area subsection template:

- **Files audited.** Exact paths + LOC.
- **AIE2 hardcode count.** Grep for literal `"AIE2"`, `AIE_ML_*`,
  `aie2`/`Aie2` identifiers, arch-branded constants, hardcoded offsets
  that appear in archspec (drift candidates).
- **Arch variance evidence.** From aie-rt per-arch headers, real XCLBINs
  on disk, llvm-aie TableGen (for ELF), AM025 register DB.
- **Prescribed migration.** `move-to-archspec` / `read-archspec-via-accessor`
  / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines added
  archspec-side.

---

## 1. XCLBIN section-kind classification

Files: `src/parser/xclbin.rs` (456 LOC).

(Filled in by Task 2 Step 1.)

## 2. AIE Partition wrapper

Files: `src/parser/aie_partition.rs` (372 LOC).

(Filled in by Task 2 Step 2.)

## 3. CDO syntax (byte format)

Files: `src/parser/cdo.rs` lines 1--412 approximately (framing + `CdoOpcode` +
`RawCdoHeader` + `Cdo` + `CdoCommandIterator`).

(Filled in by Task 2 Step 3.)

## 4. CDO semantics (device effect per command)

Files: `src/parser/cdo.rs` lines 219--285 approximately (`CdoCommand` enum),
plus `src/device/state/cdo.rs` (`apply_command`).

(Filled in by Task 2 Step 4.)

## 5. Device-state consumer

Files: `src/device/state/cdo.rs`, `src/device/state/mod.rs`.

(Filled in by Task 2 Step 5.)

## 6. ELF consumer

Files: `src/parser/elf.rs` (618 LOC) + 8 consumer files
(see plan §File Structure).

(Filled in by Task 2 Step 6.)

## 7. Control-packet parser overlap

Files: `src/device/control_packets/parser.rs`, compared against
`src/parser/*`.

(Filled in by Task 2 Step 7.)

## 8. Design note output

`docs/arch/binary-loader-model.md` -- "what would AIE1 look like?"
for the parser layer.

(Filled in by Task 5.)

## 9. Trait-or-no-trait decision

(Filled in by Task 2 Step 9.)

---

## Closing summary

(Filled in by Task 2 Step 9.)

### Data migration list

### Trait decision (populated / anchor / none) + reasoning

### Refined `DeviceOp` enum proposal

### AIE1 projection (one paragraph)

---

## Completion

(Filled in at the end of Stage 8a, in Task 6.)
```

Run:
```bash
git add docs/arch/subsys8-audit.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 8 audit scaffold

Per-area parser audit scaffold. Nine sections: XCLBIN, AIE
Partition, CDO syntax, CDO semantics, device-state consumer, ELF
consumer, control-packet overlap, design-note output, trait
decision. Plus closing-summary section for the data-migration list,
trait decision reasoning, refined DeviceOp proposal, and AIE1
projection. Content fills in over Task 2 steps.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

- [x] **Step 3: Create `docs/arch/binary-loader-model.md` scaffold**

Write this exact content:

```markdown
# Binary Loader Model -- Design Note

**Subsystem:** 8 (Phase 1b)
**Tag:** `phase1-subsys-parser-arch`
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Audit:** [subsys8-audit.md](subsys8-audit.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains where arch-specific data
lives after Stage 8a's migrations, what the `BinaryLoader` trait's
surface is (if any), and what an AIE1 port of the parser would
look like.

Filled in after Task 2 (audit) lands and Task 5 (trait decision)
lands.

---

## What lives where

(Filled in at Task 5. Expected table: per-data-item, which module
owns it after migration -- archspec constants, archspec tables,
xdna-emu types, etc.)

## Trait surface

(Filled in at Task 5. One of: populated with 1--3 methods, empty
anchor mirroring `IsaExecutor`, or "no trait at all" mirroring
`IsaDecoder`. Reasoning tied to audit §9 rejection table.)

## The shape-vs-values rule, applied to parser

(Filled in at Task 5. Applies Subsystem 6's "types from toolchain"
and Subsystem 7's "data vs algorithms" framings to parser concerns.)

## What would AIE1 look like?

(Filled in at Task 5 -- audit §8 supplies the projection as part of
its closing summary; the design note expands to ~200--300 words.)

## Alternatives rejected

(Filled in at Task 5. Expected content: any trait shape the audit
considered and rejected, mirroring isa-execute-model.md §"Alternatives
rejected" structure.)
```

Run:
```bash
git add docs/arch/binary-loader-model.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 8 design-note scaffold

BinaryLoader design-note scaffold. Sections for data placement,
trait surface, shape-vs-values rule, AIE1 projection, and
alternatives rejected. Filled in at Task 5 after the audit lands.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

---

## Task 2: Perform the audit (9 sections)

**Files:**
- Modify: `docs/arch/subsys8-audit.md` (fill in sections 1--9)

**Goal:** Catalogue per-area divergence evidence for the parser and its consumers. The audit's per-area work is grep-driven and evidence-gathering, suited for parallel Explore agents (one per section).

**Approach:** Seven Explore agents dispatched in parallel for sections 1--7. Section 8 (design note output) stays empty until Task 5 (design note filled); its placeholder here is structural. Section 9 (trait decision) is synthesized in Step 9 by the executing agent from sections 1--7 findings.

- [x] **Step 1: Section 1 -- XCLBIN section-kind classification**

Dispatch Explore agent with this prompt:

```
Audit src/parser/xclbin.rs (456 LOC) for arch variance. Produce a
Markdown section that replaces "(Filled in by Task 2 Step 1.)" in
docs/arch/subsys8-audit.md §1.

Structure:
- Files audited: src/parser/xclbin.rs
- LOC: 456
- AIE2 hardcode count: grep -c for "AIE2", "aie2", "Aie2" in the
  file. Report.
- Section-kind classification table: for each of the 35 SectionKind
  variants (enum at xclbin.rs:28-64), classify as
  (arch-agnostic | AIE-wide | AIE2-specific | unused).
  Evidence: cross-reference mlir-aie's XCLBIN conventions at
  ../mlir-aie/lib/Dialect/AIE/*, aie-rt's xclbin handling at
  ../aie-rt/driver/src/*.
- Drift candidates: hardcoded offsets/magics in xclbin.rs that
  duplicate anything in archspec.
- Prescribed migration per drift candidate.
- Estimated LOC impact.
- Notes: anything surprising.

Return ~300-500 words of Markdown ready to paste.
```

Paste the returned content into §1 of `docs/arch/subsys8-audit.md`.

- [x] **Step 2: Section 2 -- AIE Partition wrapper**

Dispatch Explore agent:

```
Audit src/parser/aie_partition.rs (372 LOC) for arch variance.
Same structure as Step 1 above. Focus: the wrapper is thin; look
for NPU1-vs-NPU4-vs-NPU6 assumptions in partition extraction,
embedded-PDI handling, column-start parsing.

Evidence sources: aie-rt's ../aie-rt/driver/src/xaie_elfloader.c
(or equivalent), xdna-driver's partition definitions at
../xdna-driver/**.

Return ~200-300 words.
```

Paste into §2.

- [x] **Step 3: Section 3 -- CDO syntax (byte format)**

Dispatch Explore agent:

```
Audit src/parser/cdo.rs lines 1-412 (framing + CdoOpcode +
RawCdoHeader + Cdo struct + CdoCommandIterator) for byte-format
variance across AIE1/AIE2/AIE2P.

Structure as Step 1 above, plus:
- CdoOpcode enum (cdo.rs:114) variant list. For each, cite the
  aie-rt CDO command reference (look at
  ../aie-rt/driver/src/cdo/xaie_cdo_*.c or equivalent).
- Per-arch byte format difference evidence.
- Whether the command-length framing (16-bit length + 16-bit
  opcode, per cdo.rs comment) holds across arches.
- AIE2P-specific variants that currently error as Unknown.

Return ~400-500 words.
```

Paste into §3.

- [x] **Step 4: Section 4 -- CDO semantics (device effect per command)**

Dispatch Explore agent:

```
Audit src/parser/cdo.rs CdoCommand enum (lines 219-285) + its
consumer src/device/state/cdo.rs's apply_command function. Produce
a per-CdoCommand-variant table:

| Variant | Device effect | Archspec rep exists? | Proposed DeviceOp |
|---------|---------------|----------------------|-------------------|
| ...     | ...           | yes/no/partial       | DeviceOp::...     |

For each variant, characterize:
- What the device actually does when this command applies (read
  device/state/cdo.rs's apply_command).
- Whether archspec already knows this semantics (Subsystems 1/3/4/5
  migrated memory-map / DMA / Locks / Stream Switch data).
- Which DeviceOp variant it should lower to (from spec's starting
  hypothesis: RegWrite, RegMask, RegBurst, BdConfigure, LockInit,
  StreamRoute, CoreEnable, DmaStart -- or propose a new variant).
- Additionally, count the command's frequency in real XCLBINs:
  `find mlir-aie/build/test/npu-xrt -name '*.xclbin' | head -30 | \
   xargs -I{} ... count commands by opcode ...`. Approximate counts
  fine.

Also scan for variants NOT in spec's starting hypothesis -- e.g.,
MaskPoll (cdo.rs:85), Delay, Marker, EndMark, Unknown. These need
DeviceOp variants too.

Return ~500-800 words.
```

Paste into §4. **This section is the primary input to the refined `DeviceOp` proposal in the closing summary.**

- [x] **Step 5: Section 5 -- Device-state consumer audit**

Dispatch Explore agent:

```
Walk src/device/state/cdo.rs's apply_command function match by
match. For each CdoCommand branch, classify:

- "moves to semantics::lower" if the branch does archspec-lookup
  or decode-then-write work (will move to semantics layer in 8b).
- "stays in apply as DeviceOp consumer" if the branch does pure
  device-state mutation (target for DeviceOp match arms).
- "is dead code" if the branch is unreachable (CdoCommands that
  never appear) or effectively a no-op.

Also audit src/device/state/mod.rs for any CdoCommand-typed imports
beyond the one at line 41 -- surface everything that'd migrate.

Return ~200-400 words.
```

Paste into §5.

- [x] **Step 6: Section 6 -- ELF consumer audit**

Dispatch Explore agent:

```
Survey the 8 files that use AieElf (per plan §File Structure:
src/device/host_memory.rs, src/device/mod.rs,
src/integration/elfanalyzer.rs, src/interpreter/decode/crossref.rs,
src/interpreter/decode/decoder.rs,
src/interpreter/engine/coordinator.rs,
src/interpreter/test_runner.rs, src/main.rs,
src/testing/xclbin_suite.rs).

For each consumer, tabulate:
- Line number where AieElf is used.
- What it's used for: program loading / data loading / symbol
  lookup / section iteration / other.
- What conventions (endianness, section-to-region mapping, symbol
  naming) are assumed.
- Would AieElf::load_into(&mut CoreMemory) cover this use, or
  does this consumer need something else (e.g., elfanalyzer
  probably wants iter_sections, not load_into)?

Also: scan src/parser/elf.rs for arch variance -- e.g., EM_*
machine type constants. Evidence from llvm-aie's LLVMContext
definitions at ../llvm-aie/llvm/include/llvm/BinaryFormat/*.

Return ~400-500 words.
```

Paste into §6.

- [x] **Step 7: Section 7 -- Control-packet parser overlap**

Dispatch Explore agent:

```
Compare src/device/control_packets/parser.rs to src/parser/*
byte-by-byte for framing / header-parsing / byte-ordering /
error-handling primitive overlap.

Produce a field-level comparison table:
| Concern | control_packets/parser.rs | parser/* | Same? |
|---------|---------------------------|----------|-------|

For each concern (header magic, endianness convention, length
framing, opcode dispatch, error type, etc.), list what each side
does.

Apply the spec's decision threshold:
- If >=3 framing primitives are effectively duplicated: extract
  shared module (recommend name + location).
- If 1-2 are similar: leave as is with an explanatory comment.
- If coincidental only: document non-overlap, no action.

Output: which outcome and why, plus specific module name if
shared-primitives path is chosen.

Return ~300-500 words.
```

Paste into §7.

- [x] **Step 8: Section 8 placeholder (filled later)**

§8 in the audit doc points at `docs/arch/binary-loader-model.md`. Leave the placeholder text that's already there -- it's filled in at Task 5. No action this step.

- [x] **Step 9: Section 9 -- Trait-or-no-trait synthesis + closing summary**

This step is synthesized by the executing agent (you), not dispatched. Working from sections 1--7 output, produce §9 + closing summary sections.

**§9 content:** Rejection table mirroring `docs/arch/isa-execute-model.md`'s format. Columns:
```
| Candidate trait method | Is there algorithmic variance? | Is the variance data-expressible? | Decision |
```

Candidate methods to evaluate (start with these; add/remove per audit findings):
- `parse_container_sections(&self, bytes) -> Vec<Section>` -- XCLBIN container parsing
- `extract_partition_payload(&self, section) -> PartitionPayload` -- AIE Partition extraction
- `decode_cdo_command(&self, frame) -> CdoRaw` -- byte-level CDO decode
- `lower_cdo(&self, raw) -> Iterator<DeviceOp>` -- semantics layer
- `load_elf(&self, bytes, &mut CoreMemory)` -- ELF loading

For each: does this differ across AIE1/AIE2/AIE2P at the algorithmic level, or just in the tables/constants it reads? Most will land as "data-expressible, reject trait method" per the Subsystem 6/7 pattern. Document the specific reasoning for each.

Decision: one of three landings per spec.
- **Populated trait** (1--3 methods) if surviving methods exist.
- **Empty anchor** mirroring `IsaExecutor` if no methods survive but dispatch pathway worth reserving.
- **No trait** mirroring `IsaDecoder` if even the anchor adds no value.

**Closing-summary content:**

Write four subsections (see audit doc scaffold):

1. **Data migration list.** Concrete items + migration verb + estimated LOC each. Example: `CdoOpcode enum -> xdna_archspec::aie2::cdo::opcodes (move-to-archspec, ~50 LOC)`.

2. **Trait decision + reasoning.** One-paragraph summary pointing at §9.

3. **Refined `DeviceOp` enum proposal.** Concrete Rust enum block. Starting from spec's hypothesis, add variants for anything §4 surfaced that isn't covered (MaskPoll? Delay? Marker?). For each: say why it's a standalone variant vs. folded into a catch-all. This is the proposal the user gates on.

4. **AIE1 projection** (one paragraph, ~100 words). What would AIE1 need? Expected answer: new `CdoOpcode` variants for AIE1-only commands (if any), different `EM_*` ELF machine type, same framing / wrapper / section layout. No algorithmic rewrites.

Run:
```bash
git add docs/arch/subsys8-audit.md
git commit -m "$(cat <<'EOF'
docs(audit): Subsystem 8 parser audit (sections 1-9 + summary)

Per-area parser audit complete. Sections 1-7 are evidence-gathering
per-area deep dives (XCLBIN, AIE Partition, CDO syntax, CDO
semantics, device-state consumer, ELF consumer, control-packet
overlap). Section 9 synthesizes the trait-or-no-trait decision
with a rejection table in the Subsystem 7 format. Closing summary
lists data migrations, trait decision, refined DeviceOp proposal
(the Stage 8a -> 8b user gate), and AIE1 projection.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

---

## Task 3: Plan amendment -- Tasks 4, 5, 17 detailed steps

**Files:**
- Modify: `docs/superpowers/plans/2026-04-23-subsys8-parser-plan.md` (this file)

**Goal:** With the audit in hand, author the concrete steps for audit-driven Tasks 4, 5, and 17. Without this amendment, those tasks are placeholders.

- [x] **Step 1: Re-read the audit**

Read `docs/arch/subsys8-audit.md` in full. Take particular note of:
- Closing summary's data migration list -> Task 4 steps.
- Closing summary's trait decision -> Task 5 steps.
- Section 7's control-packet decision -> Task 17 steps.

- [x] **Step 2: Write Task 4 steps**

Template per migration item:
- Create destination file (`crates/xdna-archspec/src/aie2/<thing>.rs` or equivalent).
- Move content. Apply minimal Rust edits to compile at destination.
- Wire accessor through `ArchConfig` trait + `Aie2ArchConfig` impl + `arch_handle::<thing>()`.
- Write drift test (mirrors Subsystem 7's `instruction_latency` drift test).
- Delete or forward-ref the original site in xdna-emu.
- Commit per migration item (one migration == one commit).

See Subsystem 7 Part B tasks 3-5 for pattern.

- [x] **Step 3: Write Task 5 steps**

Two possible shapes based on audit §9:

**If audit says "populated trait" (1-3 methods):**
- Create `crates/xdna-archspec/src/binary_loader/mod.rs` with the trait + method signatures.
- Create `crates/xdna-archspec/src/aie2/binary_loader_model.rs` with `Aie2BinaryLoader` ZST + `AIE2_BINARY_LOADER` singleton + impls.
- Wire `ArchConfig::binary_loader()` method.
- Wire `arch_handle::binary_loader()` accessor in `src/device/arch_handle.rs`.
- Dispatch test in `runtime.rs` (mirror Subsystem 7's `aie2_isa_executor_*` tests).
- One test per trait method verifying AIE2 behavior.

**If audit says "empty anchor":**
- Same scaffold as populated, but trait body is empty with doc comment pointing at audit §9 for reasoning.
- No per-method tests; just the dispatch test.

**If audit says "no trait":**
- No archspec binary_loader/ module. Skip directly to Task 5's close.
- `docs/arch/binary-loader-model.md` §"Trait surface" says "No trait" with explicit reasoning.

In all three cases, fill in the design-note body (`binary-loader-model.md` §What lives where, §Trait surface, §Shape-vs-values, §AIE1 projection, §Alternatives rejected).

- [x] **Step 4: Write Task 17 steps**

Three possible shapes based on audit §7:

**If "extract shared module":** Create `src/parser/framing.rs` (or audit-chosen name). Migrate duplicated primitives. Update both `src/parser/*` and `src/device/control_packets/parser.rs` consumers. Commit.

**If "leave as is":** Edit `src/device/control_packets/parser.rs` to add a module-level comment explaining the overlap decision with a pointer to audit §7. Commit as `docs(control_packets): document parser overlap decision with src/parser`.

**If "coincidental only":** Same as "leave as is."

- [x] **Step 5: Commit the plan amendment**

```bash
git add docs/superpowers/plans/2026-04-23-subsys8-parser-plan.md
git commit -m "$(cat <<'EOF'
docs(plan): Subsys 8 Task 4/5/17 steps from audit findings

Populates the audit-driven steps of Stage 8a: data migrations
(Task 4), BinaryLoader trait decision (Task 5), and the
control-packet alignment direction (Task 17). All three were
marked as amendment-protocol in the original plan header; content
now lives in the relevant task sections.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

---

## Task 4: Data migrations (audit-driven)

**Goal:** Migrate the 2 items surfaced by audit §Closing Summary "Data migration list":
1. `EM_AIE: u16 = 264` (ELF machine type literal)
2. `AieArchitecture` enum (Aie1=0x01, Aie2=0x02, Aie2P=0x03)

Both move from `src/parser/elf.rs` to a new `xdna_archspec::elf` module. One commit for
both migrations (they're tightly coupled; ~17 LOC total archspec addition).

**Files:**
- Create: `crates/xdna-archspec/src/elf.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod elf;`)
- Modify: `src/parser/elf.rs`

- [x] **Step 1: Create `crates/xdna-archspec/src/elf.rs`**

Content:

```rust
//! ELF format constants shared across AIE architectures.
//!
//! These are toolchain-derived identifiers (from llvm-aie backends):
//! - `EM_AIE` is the ELF machine type number LLVM emits for AIE ELFs.
//! - `AieArchitecture` is the per-arch flag value carried in the ELF
//!   header's `e_flags` field.
//!
//! Kept in archspec (not xdna-emu's parser) so that a future AIE1
//! implementation populates its arch constants in the same place as
//! memory-map, DMA model, and ISA data.

/// ELF machine type for AIE cores. LLVM's AIE backend emits this value
/// in `Elf::e_machine`. Source: llvm-aie/llvm/include/llvm/BinaryFormat/ELF.h.
pub const EM_AIE: u16 = 264;

/// AIE architecture variant encoded in ELF `e_flags`.
///
/// Source: llvm-aie AIEELFObjectWriter + aie-rt's ELF loader expects
/// this enum's values in the low byte of `e_flags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AieArchitecture {
    Aie1 = 0x01,
    Aie2 = 0x02,
    Aie2P = 0x03,
}

impl AieArchitecture {
    /// Decode from the low byte of ELF `e_flags`. Returns `None` for
    /// unrecognized values rather than defaulting, so callers can
    /// decide how to handle unknown arches.
    pub fn from_e_flags(flags: u32) -> Option<Self> {
        match (flags & 0xFF) as u8 {
            0x01 => Some(Self::Aie1),
            0x02 => Some(Self::Aie2),
            0x03 => Some(Self::Aie2P),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn em_aie_matches_llvm_aie_value() {
        assert_eq!(EM_AIE, 264);
    }

    #[test]
    fn aie_architecture_roundtrips_through_e_flags() {
        for (flag, expected) in [
            (0x01u32, AieArchitecture::Aie1),
            (0x02u32, AieArchitecture::Aie2),
            (0x03u32, AieArchitecture::Aie2P),
        ] {
            assert_eq!(AieArchitecture::from_e_flags(flag), Some(expected));
        }
    }

    #[test]
    fn aie_architecture_rejects_unknown_flags() {
        assert_eq!(AieArchitecture::from_e_flags(0x00), None);
        assert_eq!(AieArchitecture::from_e_flags(0xFF), None);
    }
}
```

- [x] **Step 2: Add `pub mod elf;`** to `crates/xdna-archspec/src/lib.rs` at the
  appropriate alphabetical spot (after existing `pub mod` declarations).

- [x] **Step 3: Update `src/parser/elf.rs`** to import from archspec rather than
  declaring locally.

  - At top: `use xdna_archspec::elf::{EM_AIE, AieArchitecture};`
  - Remove the local `const EM_AIE: u16 = 264;` definition.
  - Remove the local `pub enum AieArchitecture { ... }` definition.
  - Update all call sites (in the same file) that referenced local `AieArchitecture`
    or `EM_AIE`; they now reach the archspec versions through the `use` statement.
  - If any consumer outside `src/parser/elf.rs` imports these (grep to check),
    update their imports too.

- [x] **Step 4: Verify**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expect: 320 + 3 new tests = 323 passed, 0 failed.

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expect: 2686 passed (xdna-emu tests unchanged; elf.rs only re-imports).

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build --release 2>&1 | tail -3
```

- [x] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/elf.rs crates/xdna-archspec/src/lib.rs src/parser/elf.rs
# plus any other files Step 3 touched
git commit -m "$(cat <<'EOF'
refactor(archspec): migrate EM_AIE + AieArchitecture to archspec

Moves the two toolchain-derived ELF identifiers from the xdna-emu
parser into a new xdna_archspec::elf module:
- EM_AIE (ELF machine type number, LLVM-defined).
- AieArchitecture enum (per-arch flag value in ELF e_flags).

Both audit §Closing Summary data-migration-list items landed.
xdna_archspec now owns all toolchain-derived ELF constants; a future
AIE1 implementation adds arch-specific flag handling in the same
place as its memory map / DMA model / ISA data.

Generated using Claude Code.
EOF
)"
```

Verification:
- `cargo test --lib` green (2686 passed, unchanged).
- `cargo test -p xdna-archspec --lib` green (323 passed -- up 3 from the new tests).
- Archspec `AieArchitecture::from_e_flags` drift tests pass.

---

## Task 5: BinaryLoader trait decision + design note

**Goal:** Audit §9 concluded **no trait** (mirrors Subsystem 6 `IsaDecoder` precedent,
not Subsystem 7's empty anchor). No `crates/xdna-archspec/binary_loader/` module is
created; no dispatch test is added. The only Task 5 deliverable is filling in
`docs/arch/binary-loader-model.md` with the audit's reasoning and AIE1 projection.

**Files:**
- Modify: `docs/arch/binary-loader-model.md`

- [x] **Step 1: Fill in `## What lives where` section**

Replace `(Filled in at Task 5. Expected table: ...)` with a concrete table reflecting
the post-migration state:

```markdown
| Data / code | Module | Notes |
|-------------|--------|-------|
| `EM_AIE` (ELF machine type = 264) | `xdna_archspec::elf` | Moved from `src/parser/elf.rs` in Task 4. |
| `AieArchitecture` enum + `from_e_flags` | `xdna_archspec::elf` | Moved from `src/parser/elf.rs` in Task 4. |
| XCLBIN container parsing (`Xclbin`, `SectionKind`) | `src/parser/xclbin.rs` | Arch-agnostic; stays in xdna-emu. |
| AIE Partition wrapper (`AiePartition`) | `src/parser/aie_partition.rs` | Arch-agnostic framing; stays in xdna-emu. |
| CDO framing (`CdoVersion`, `RawCdoHeader`, `find_cdo_offset`) | `src/parser/cdo/framing.rs` | Arch-uniform byte-level framing. Created in Stage 8b Half 1. |
| CDO syntax (`CdoOpcode`, `CdoRaw`, `Cdo`) | `src/parser/cdo/syntax.rs` | Typed commands; arch-uniform byte format. Created in Stage 8b Half 1. |
| CDO semantics (`lower`) + arch-handle-consuming lowering | `src/parser/cdo/semantics.rs` | Reads archspec (memory map, DMA model, etc.) through existing accessors. No new arch dispatch. Created in Stage 8b Half 2. |
| ELF parsing (`AieElf`, `MemoryRegion`, `load_into`) | `src/parser/elf.rs` | Arch-agnostic ELF reader; uses `xdna_archspec::elf` constants. Consolidated in Stage 8c. |
| `DeviceOp` enum (arch-generic device ops) | `src/device/ops.rs` | New vocabulary between parser and device-state. Created in Stage 8b Half 2. |
```

- [x] **Step 2: Fill in `## Trait surface` section**

Replace `(Filled in at Task 5. One of: populated ... / empty anchor ... / "no trait" ...)` with:

```markdown
**No trait.** The parser layer does not define a `BinaryLoader` trait in
`xdna-archspec`. This matches the Subsystem 6 `IsaDecoder` precedent and
reflects three findings from the audit:

1. **XCLBIN / AIE Partition / CDO framing is arch-uniform.** Every AIE2 NPU
   (and every AIE1 Versal design that the audit checked) uses the same XCLBIN
   magic, the same AIE Partition struct, and the same CDO v2 byte-level format.
   There is no variance to dispatch on at the container-parsing layer.

2. **CDO semantics variance is data-expressible.** The `semantics::lower`
   function consults `ArchHandle` accessors (memory map, DMA model, lock layout,
   stream switch topology) for address decoding and BD field parsing. All four
   of those accessors are already arch-aware through their respective model
   traits (Subsystems 1, 3, 4, 5). `lower` adds no new arch-dispatch surface;
   it's a plain free function that parameterises through pre-existing accessors.

3. **Parsing is not on a hot path.** Unlike ISA execute (where the interpreter
   inner loop might benefit from a trait anchor to monomorphise later), the
   parser runs once at XCLBIN load. There is no dispatch-pathway-reservation
   argument that would motivate the empty-anchor pattern Subsystem 7 used.

The single small archspec deliverable is `xdna_archspec::elf` (Task 4), which is
a data module (two constants + one enum), not a trait.
```

- [x] **Step 3: Fill in `## The shape-vs-values rule, applied to parser` section**

Replace the placeholder with:

```markdown
Subsystem 6 established the rule:

> A type belongs in archspec iff it is derivable from the toolchain without
> reference to emulator execution state.

Subsystem 7 sharpened it:

> Among arch-specific content, *data* (tables, enums, constants, feature flags)
> lives in archspec while *algorithms* stay in xdna-emu.

Applied to the parser:

- **Data** that the toolchain (llvm-aie + aie-rt) defines -- `EM_AIE`,
  `AieArchitecture`, CDO opcode identity -- lives in archspec. The Task 4
  migration moves `EM_AIE` and `AieArchitecture` specifically; `CdoOpcode` stays
  in xdna-emu because its byte-level opcode values are not defined by the
  toolchain (they're defined by AMD's CDO specification, which is arch-uniform
  and doesn't warrant the migration overhead for one enum).
- **Algorithms** -- XCLBIN section iteration, AIE Partition extraction, CDO
  command decoding, ELF section-to-memory copying -- stay in xdna-emu. These
  read archspec data where needed (Task 4 adds one such read site in
  `parser::elf`) but are themselves pure functions of bytes-in, typed-out.
- **The one interesting layering departure** is `cdo::semantics::lower`, which
  reads archspec freely (memory map, DMA model, etc.) during the byte-to-op
  translation. This is deliberate -- the spec's design philosophy §"Elegant
  over pristine purity" explicitly allows the parser to be arch-aware in the
  semantics sublayer. The syntax and framing sublayers stay arch-blind for
  testability; the semantics sublayer is arch-aware because the lowering task
  demands it.
```

- [x] **Step 4: Fill in `## What would AIE1 look like?` section**

Copy the AIE1 projection from `docs/arch/subsys8-audit.md` §Closing Summary (the
one-paragraph version), then expand with the specific archspec additions AIE1 would
need:

```markdown
An AIE1 port of the parser layer would require no algorithm changes and one
small archspec addition.

**Archspec additions:**
- `xdna_archspec::aie1::elf` -- if AIE1's ELF flag values ever diverge from
  AIE2's (e.g., AIE1 uses `e_flags = 0x01`). Today this is handled by the
  existing `AieArchitecture::Aie1 = 0x01` variant, so the module may not even
  need to be created.
- AIE1-specific `CdoOpcode` variants -- if AIE1 Versal designs include PM-domain
  or NPI commands not present in today's enum. These would be additions to
  `CdoOpcode`'s `Unknown(u16)` catch-all, which is in xdna-emu not archspec; the
  catch-all already handles unknown opcodes gracefully, so they don't need to
  be named unless the parser needs to interpret them.

**Parser-side changes:** **none.** `xclbin::parse`, `aie_partition::parse`,
`cdo::framing::parse_header`, `cdo::syntax::decode`, `elf::AieElf::parse`, and
`elf::AieElf::load_into` are all byte-level parsers that work against
arch-uniform formats. `cdo::semantics::lower` dispatches through `ArchHandle`
accessors already -- when `ArchConfig::Aie` is populated with AIE1 values, the
accessors return AIE1 data and the lowering works unchanged.

**Device-state side changes:** none specific to the parser -- `device::state::apply`
consumes `DeviceOp`, which is already arch-generic.

The smoothness of this projection is precisely why the audit concluded "no trait":
there is no variance at the parsing layer that a trait would dispatch on.
```

- [x] **Step 5: Fill in `## Alternatives rejected` section**

```markdown
### Populated `BinaryLoader` trait (1--3 methods)

Rejected. §9's rejection table evaluated five candidate methods
(`parse_container_sections`, `extract_partition_payload`,
`decode_cdo_command`, `lower_cdo`, `load_elf`). None survived -- every
candidate either has zero algorithmic variance (container framing is
arch-uniform) or is data-expressible through existing archspec accessors
(CDO semantics). A populated trait would be ceremony.

### Empty anchor trait (mirroring `IsaExecutor`)

Rejected. Subsystem 7 justified the `IsaExecutor` empty anchor on two
grounds: (a) preserving a dispatch pathway for future seams without
cross-subsystem plumbing, and (b) hot-path dispatch might eventually
benefit from monomorphisation. Neither applies to the parser:

- Future parser seams would most likely be new archspec data (new
  opcodes, new ELF flags), not new dispatch. An anchor doesn't help
  data migrations.
- Parsing is a one-shot cold path. There is no hot-path motivation.

The cost of an empty anchor (one module, one ZST, one singleton, one
dispatch test, one accessor) was deemed not worth paying for a pathway
we don't expect to populate.

### Pre-audit commitment

Rejected by spec. The parent device-family refactor explicitly requires
audit-first design. Subsystem 5's `PortLayout` extension trait (231 LOC
of dead code deleted after the fact) is the cautionary precedent.
```

- [x] **Step 6: Verify no placeholders remain**

```bash
grep -n "Filled in at Task\|TODO\|TBD" docs/arch/binary-loader-model.md
# Expect: no output (all placeholders filled).
```

- [x] **Step 7: Commit**

```bash
git add docs/arch/binary-loader-model.md
git commit -m "$(cat <<'EOF'
docs(design-note): Subsystem 8 binary-loader-model populated

Fills the design note body with the "no trait" landing per audit §9.
Five sections now populated: data placement table, trait-surface
reasoning, shape-vs-values rule applied, AIE1 projection,
alternatives rejected (populated trait, empty anchor, pre-audit
commitment).

The audit's rejection-table logic lives in subsys8-audit.md §9;
this doc is the durable design-note companion per the parent
refactor's "what would AIE1 look like?" requirement.

Generated using Claude Code.
EOF
)"
```

Verification:
- `cargo test --lib` green (unchanged by doc-only commit).
- `grep "Filled in at Task" docs/arch/binary-loader-model.md` returns nothing.

---

## Task 6: Stage 8a close + tag + user gate

**Files:**
- Modify: `NEXT-STEPS.md`
- Modify: `docs/arch/subsys8-audit.md` (fill §Completion)
- Modify: `docs/superpowers/specs/2026-04-23-subsys8-parser-design.md` (update §"`DeviceOp` vocabulary" with refined proposal)

**Goal:** Close Stage 8a, tag `phase1-subsys-parser-arch`, hand the refined `DeviceOp` proposal to the user.

- [x] **Step 1: Update spec's `DeviceOp` vocabulary section**

Read audit closing-summary §"Refined `DeviceOp` enum proposal". Apply that enum block to `docs/superpowers/specs/2026-04-23-subsys8-parser-design.md` §"`DeviceOp` vocabulary" > "Shape". Replace the starting-hypothesis enum with the refined proposal. Keep the design rules (1-5) unchanged unless the audit surfaced a reason to revise them.

Commit:
```bash
git add docs/superpowers/specs/2026-04-23-subsys8-parser-design.md
git commit -m "$(cat <<'EOF'
docs(spec): Subsys 8 -- refined DeviceOp proposal from audit

Replaces the starting-hypothesis DeviceOp enum with the audit-driven
proposal (see docs/arch/subsys8-audit.md §4 + closing summary).
Variant list reflects actual CDO command frequencies observed in
real XCLBINs and the device-state consumer classification.

This is the Stage 8a -> 8b gate. Half 2 of 8b does not start until
the user reviews and confirms this proposal.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 2: Fill audit §Completion**

Write `docs/arch/subsys8-audit.md` §Completion with:
- Commit-series summary (7--15 commits, per audit results).
- Deliverables: audit complete, data migrations complete, trait decision landed, design note filled, refined DeviceOp in spec.
- Follow-ups flagged (Phase 2 hygiene items surfaced during audit).
- Mirror the §Completion sections of subsys{2-7}-audit.md.

Commit:
```bash
git add docs/arch/subsys8-audit.md
git commit -m "$(cat <<'EOF'
docs(audit): Subsystem 8 Stage 8a completion log

Completion section for Stage 8a (arch audit + data migrations +
BinaryLoader trait decision). Summarises commits, deliverables,
and follow-ups flagged for Phase 2 hygiene.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 3: Update NEXT-STEPS.md**

Edit NEXT-STEPS.md to:
- Change "Latest tag" to `phase1-subsys-parser-arch`.
- Mark Subsystem 8 row status as "Stage 8a done; gate; Stage 8b up next".
- Add a §"Gate between 8a and 8b" section pointing at the refined DeviceOp proposal and explaining that Half 2 work is paused pending user confirmation.

Commit:
```bash
git add NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: update NEXT-STEPS for Subsystem 8 Stage 8a completion

Latest tag advances to phase1-subsys-parser-arch. Stage 8b Half 2
is gated on user review of the refined DeviceOp proposal now in
the subsys8 spec.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 4: Run full verification gates**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
cargo build --release 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8a-bridge.log
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8a-isa.log
```

Compare bridge + ISA logs against `/tmp/claude-1000/subsys8-baseline-*.log` from Task 1. Expect no regressions; pre-existing failures carry through.

- [x] **Step 5: Apply tag**

```bash
git tag phase1-subsys-parser-arch HEAD
git tag | grep phase1-subsys
# Expect phase1-subsys-parser-arch in list.
```

- [x] **Step 6: Pause for user gate**

**STOP.** Stage 8b Half 2 does not begin until the user reviews the refined DeviceOp proposal (in the spec + audit) and confirms. Output to the user:

> "Stage 8a complete and tagged as `phase1-subsys-parser-arch`. The refined `DeviceOp` enum proposal is in `docs/arch/subsys8-audit.md` §closing-summary and now reflected in `docs/superpowers/specs/2026-04-23-subsys8-parser-design.md` §'`DeviceOp` vocabulary'. Please review before Stage 8b starts Half 2 -- that's the gated point. Half 1 (rename + internal split, no `DeviceOp` yet) is fully specifiable and can start immediately on your word; only Half 2 is gated."

Wait for user response before proceeding. If user requests DeviceOp reshape, re-enter brainstorming skill with the current proposal + user's concerns.

---

## Task 7: Stage 8b Half 1 -- rename `CdoCommand` -> `CdoRaw`

**Files:**
- Modify: `src/parser/cdo.rs`
- Modify: `src/main.rs`
- Modify: `src/device/state/cdo.rs`
- Modify: `src/device/state/mod.rs`
- Modify: `src/testing/xclbin_suite.rs`
- Any other files grep surfaces

**Goal:** Rename `CdoCommand` -> `CdoRaw` throughout. No behavior change. Type-only refactor.

- [x] **Step 1: Verify scope with grep**

```bash
grep -rnw "CdoCommand" src/ tests/ --include='*.rs'
```

Expect the 5 files listed in plan §File Structure plus possibly xclbin_suite.rs. If the grep surfaces more, all of them get updated in this task.

- [x] **Step 2: Rename in the definition site**

In `src/parser/cdo.rs` (line 219 per grep), change `pub enum CdoCommand {` to `pub enum CdoRaw {`. All internal references in the file to `CdoCommand` become `CdoRaw`, including:
- The return type of `Cdo::commands()` iterator
- `CdoCommandIterator`'s `Item` type
- Doc comments mentioning `CdoCommand`

Do NOT rename `CdoCommandIterator` in this task -- that's the iterator's name and can stay for bisect friendliness. Rename it in Task 8 as part of the file split.

- [x] **Step 3: Update consumers**

For each file in Step 1's grep output (excluding `src/parser/cdo.rs` now renamed):
- Replace `CdoCommand` with `CdoRaw` at each site.
- Update `use` statements.

Specifically:
- `src/main.rs:8` -- `use xdna_emu::parser::cdo::{find_cdo_offset, CdoRaw};`
- `src/main.rs:179,285,301,305,309,320` -- all `CdoCommand::X` -> `CdoRaw::X`
- `src/device/state/cdo.rs:31` -- `fn apply_command(&mut self, cmd: &CdoRaw) -> Result<()>`
- `src/device/state/cdo.rs:33,37,45,55,64,73,85,90,95,99` -- all `CdoCommand::X` -> `CdoRaw::X`
- `src/device/state/mod.rs:41` -- `use crate::parser::cdo::{Cdo, CdoRaw};`
- `src/testing/xclbin_suite.rs:320` -- `use crate::parser::cdo::CdoRaw;`

- [x] **Step 4: Verify build + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: build clean, 2686 passed, 0 failed.

- [x] **Step 5: Commit**

```bash
git add src/parser/cdo.rs src/main.rs src/device/state/cdo.rs src/device/state/mod.rs src/testing/xclbin_suite.rs
# plus any others from Step 1 grep
git commit -m "$(cat <<'EOF'
refactor(parser): rename CdoCommand -> CdoRaw

Type-only rename, no behavior change. Positions the type name
for the CDO module split in Task 8 (framing/syntax/semantics
layers), where CdoRaw is the byte-level typed-command vocabulary
that flows from syntax.rs to semantics.rs.

All consumers updated: main.rs (pretty-printer), device/state/cdo.rs
(apply_command), device/state/mod.rs (imports),
testing/xclbin_suite.rs.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

---

## Task 8: Stage 8b Half 1 -- split `cdo.rs` into framing / syntax / semantics

**Files:**
- Delete: `src/parser/cdo.rs` (835 LOC)
- Create: `src/parser/cdo/mod.rs`
- Create: `src/parser/cdo/framing.rs`
- Create: `src/parser/cdo/syntax.rs`
- Create: `src/parser/cdo/semantics.rs`
- Modify: `src/parser/mod.rs` (re-export path unchanged, but mod declaration may need adjusting)

**Goal:** Split the 835-LOC `cdo.rs` into four cohesive files along framing / syntax / semantics lines. No behavior change; `semantics.rs` is a pass-through that returns `CdoRaw` unchanged. External surface (`Cdo`, `CdoRaw`, `find_cdo_offset`) is preserved via `src/parser/cdo/mod.rs` re-exports.

- [x] **Step 1: Create directory + four target files**

```bash
mkdir -p src/parser/cdo
touch src/parser/cdo/{mod.rs,framing.rs,syntax.rs,semantics.rs}
```

- [x] **Step 2: Populate `src/parser/cdo/framing.rs`**

Content (moved from `src/parser/cdo.rs`):
- `CDO_MAGIC_CDO`, `CDO_MAGIC_XLNX`, `CDO_HEADER_SIZE` constants.
- `find_cdo_offset` function.
- `CdoVersion` enum + its helpers.
- `RawCdoHeader` struct + parsing.
- Any header-validation / version-detection code.
- Module-level doc comment explaining: "Framing: byte-level CDO
  container parsing (header, version, command length frames).
  Independent of command opcode meaning."

No arch imports. Pure byte-level parsing.

- [x] **Step 3: Populate `src/parser/cdo/syntax.rs`**

Content (moved from `src/parser/cdo.rs`):
- `CdoOpcode` enum.
- `CdoRaw` enum (the renamed `CdoCommand`).
- `Cdo` struct + its `commands()` method.
- `CdoCommandIterator` struct (rename to `CdoRawIterator` as part of this move -- in the same commit since it's an internal type).
- Module-level doc comment explaining: "Syntax: typed commands
  (CdoRaw) decoded from CDO byte frames. Shape-aware of command
  opcodes but not of their device-level effects."

Imports `framing::{find_cdo_offset, CdoVersion, RawCdoHeader}` as needed.

- [x] **Step 4: Populate `src/parser/cdo/semantics.rs`**

Content (Half 1 is pass-through):

```rust
//! Semantics: CdoRaw -> DeviceOp lowering.
//!
//! Half 1 of Subsystem 8 Stage 8b: this module is a pass-through.
//! It accepts CdoRaw and returns CdoRaw unchanged, preserving the
//! parser/device-state interface while the CDO file is split into
//! layers. Half 2 rewrites `lower` to emit DeviceOp (arch-generic
//! device-facing ops), consulting &ArchHandle for address decoding
//! and BD field parsing.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md
//! §Stage 8b for the two-halves rationale.

use super::syntax::CdoRaw;

/// Pass-through in Half 1 of Stage 8b.
///
/// Half 2 replaces this with a proper lowering that consults an
/// ArchHandle and emits a device-facing op stream.
pub fn lower(raw: CdoRaw) -> CdoRaw {
    raw
}
```

Tests: `semantics_pass_through_returns_input_unchanged` verifying the identity behavior.

- [x] **Step 5: Populate `src/parser/cdo/mod.rs`**

```rust
//! CDO (Configuration Data Object) parser.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md for the
//! two-layer design: framing (byte-level), syntax (CdoRaw typed
//! commands), semantics (CdoRaw -> DeviceOp in Half 2; pass-through
//! in Half 1).

pub mod framing;
pub mod syntax;
pub mod semantics;

pub use framing::{find_cdo_offset, CdoVersion, RawCdoHeader, CDO_MAGIC_CDO, CDO_MAGIC_XLNX, CDO_HEADER_SIZE};
pub use syntax::{Cdo, CdoOpcode, CdoRaw};
```

The re-exports preserve the existing `crate::parser::cdo::{Cdo, CdoRaw, find_cdo_offset, ...}` surface. Consumers' `use` statements don't change.

- [x] **Step 6: Delete old `src/parser/cdo.rs`**

```bash
rm src/parser/cdo.rs
```

- [x] **Step 7: Update `src/parser/mod.rs` if needed**

Check current content:
```bash
cat src/parser/mod.rs
```

The existing `pub mod cdo;` + `pub use cdo::Cdo;` should continue to work once `src/parser/cdo/mod.rs` exists. If `cdo` was previously a file-module and needs to be a directory-module, no source change is typically needed in Rust 2021 (both `cdo.rs` and `cdo/mod.rs` count as the same module).

- [x] **Step 8: Verify build + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build 2>&1 | tail -10
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -5
```

Expected: build clean, 2687 passed (up one from the new semantics pass-through test), 0 failed.

- [x] **Step 9: Commit**

```bash
git add src/parser/cdo/ src/parser/cdo.rs src/parser/mod.rs
git commit -m "$(cat <<'EOF'
refactor(parser): split cdo.rs into framing/syntax/semantics

Stage 8b Half 1. src/parser/cdo.rs (835 LOC) is split along
functional lines:

- cdo/framing.rs: header, version, command framing (byte-level).
- cdo/syntax.rs: CdoRaw typed-command decode from byte frames.
- cdo/semantics.rs: pass-through in Half 1; rewritten to emit
  DeviceOp in Half 2.
- cdo/mod.rs: public API re-exports (consumers' use statements
  unchanged).

No behavior change. Closing commit of Half 1 -- the bisect anchor
for Half 2's rollback posture (see spec §Rollback posture).

Generated using Claude Code.
EOF
)"
```

Expected: clean commit.

---

## Task 9: Stage 8b Half 1 checkpoint -- intermediate verification

**Goal:** Confirm Half 1 closing commit is bisect-safe (passes full per-stage-tag gates, not just per-commit).

- [x] **Step 1: Full verification gates**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
cargo build --release 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw 2>&1 | tee /tmp/claude-1000/subsys8-8b-half1-bridge-nohw.log
```

Expected:
- `cargo test --lib`: 2687 passed, 0 failed.
- archspec: 320 passed, 0 failed.
- `cargo build --release`: clean.
- FFI build: clean.
- Bridge `--no-hw`: matches baseline.

- [x] **Step 2: No commit (documentation only)**

Half 1's checkpoint is not a separate commit -- the Task 8 commit is the anchor. This task is a verification gate, not a tag step. Document the bisect anchor hash:

```bash
git rev-parse HEAD
# Record this hash. If Half 2 (Tasks 10-12) needs to roll back, it's
# the commit to checkout.
```

Record the hash in a scratch file at `/tmp/claude-1000/subsys8-half1-anchor.sha` so Task 10/11/12 can reference it if rollback becomes relevant.

---

## Task 10: Stage 8b Half 2 -- introduce `DeviceOp` module (audit-gated)

**Files:**
- Create: `src/device/ops.rs`

**Goal:** Define the `DeviceOp` enum per the user-confirmed refined proposal in the spec. One new module, no consumer changes yet.

**Detailed steps:** Authored via Task 3 Step 2 amendment using the refined `DeviceOp` enum from spec §"`DeviceOp` vocabulary" (post-Task-6 spec state).

Until the user confirms the gate, this task's steps are amendment-pending. The expected shape is:

- [x] **Step 1: Create `src/device/ops.rs`** with the refined enum, `TileAddr` / `BdFields` / `StreamRouteSpec` imports, Copy-where-possible derives, unit tests for each variant construction.

- [x] **Step 2: Register the module** in `src/device/mod.rs` (`pub mod ops;`).

- [x] **Step 3: Verify build + tests.**

- [x] **Step 4: Commit.**

```bash
# Commit message skeleton -- filled by amendment
feat(device): add DeviceOp vocabulary for parser -> state boundary
```

---

## Task 11: Stage 8b Half 2 -- rewrite `semantics::lower` to emit `DeviceOp`

**Files:**
- Modify: `src/parser/cdo/semantics.rs`

**Goal:** Replace the pass-through `lower(CdoRaw) -> CdoRaw` with the real lowering that consults `&ArchHandle` and emits `Iterator<Item = DeviceOp>`.

**Detailed steps:** Authored via Task 3 Step 2 amendment using the refined `DeviceOp` variant list from audit §4 and the refined `DeviceOp::apply` patterns.

Expected shape:

- [x] **Step 1: Import `DeviceOp` + `ArchHandle`** in `src/parser/cdo/semantics.rs`.

- [x] **Step 2: Replace pass-through `lower`** with real implementation. Match on every `CdoRaw` variant; produce `SmallVec<[DeviceOp; 4]>` for each. Free function, no trait method (per spec §Performance stance).

- [x] **Step 3: Per-variant unit tests.** For each `DeviceOp` variant the refined proposal surfaces: construct a `CdoRaw`, apply `lower` with a test `ArchHandle`, verify `DeviceOp` sequence.

- [x] **Step 4: Verify build + tests. Commit.**

```bash
# Commit message skeleton
feat(parser): semantics::lower emits DeviceOp from CdoRaw
```

---

## Task 12: Stage 8b Half 2 -- migrate `device/state` to consume `DeviceOp`

**Files:**
- Modify: `src/device/state/cdo.rs` (primary)
- Modify: `src/device/state/mod.rs`
- Possibly: `src/device/state/{compute,dispatch,effects,memtile}.rs` if they reference `CdoRaw`

**Goal:** `device/state/cdo.rs::apply_command` stops matching on `CdoRaw`; consumes `DeviceOp` emitted by `semantics::lower`. Parser's public surface to device/state is now `impl Iterator<Item = DeviceOp>`.

**Detailed steps:** Authored via Task 3 Step 2 amendment.

Expected shape:

- [x] **Step 1: Rewrite `apply_command`** to accept `&DeviceOp` instead of `&CdoRaw`. Per-DeviceOp-variant match arms.

- [x] **Step 2: Update call chain.** Caller (in `device/state/mod.rs` or wherever `apply_command` is invoked) now calls `parser::cdo::semantics::lower(raw, arch)` and applies each `DeviceOp`.

- [x] **Step 3: Verify build + tests + bridge smoke.**

- [x] **Step 4: Commit.**

```bash
# Commit message skeleton
refactor(device): state::apply consumes DeviceOp (was CdoRaw)
```

---

## Task 13: Stage 8b close + tag

- [x] **Step 1: Full verification gates**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
cargo build --release 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8b-bridge.log
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8b-isa.log
```

Compare against baseline + 8a logs. Expect no regressions.

- [x] **Step 2: Update NEXT-STEPS.md**

Advance "Latest tag" to `phase1-subsys-parser-coupling`. Update Stage 8b row status.

- [x] **Step 3: Apply tag**

```bash
git tag phase1-subsys-parser-coupling HEAD
```

- [x] **Step 4: Commit NEXT-STEPS update + tag**

```bash
git add NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: update NEXT-STEPS for Subsystem 8 Stage 8b completion

Stage 8b tag: phase1-subsys-parser-coupling. CDO layer split +
DeviceOp vocabulary + device/state boundary move landed. Stage 8c
(diagnostics / fixtures / ELF dedup / control-packet alignment)
up next.

Generated using Claude Code.
EOF
)"
```

---

## Task 14: Stage 8c -- diagnostics (`ParseError` enum)

**Files:**
- Create: `src/parser/error.rs`
- Modify: `src/parser/xclbin.rs`, `aie_partition.rs`, `cdo/framing.rs`, `cdo/syntax.rs`, `elf.rs`

**Goal:** Replace most `anyhow::bail!` and `anyhow!` sites in the parser module with a structured `ParseError` enum carrying `{offset, expected, got, hex_context}`. Error-path messages become actionable.

- [x] **Step 1: Write the failing test for ParseError display**

Create `src/parser/error.rs`:

```rust
//! Structured parser diagnostics.
//!
//! ParseError carries byte offset, expected vs got, and a hex context
//! window around the offending byte for every fallible boundary in
//! the parser. The Display impl renders this as a human-friendly
//! message suitable for CLI output; the Debug impl gives full
//! machine-readable context.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected magic at offset 0x{offset:x}: expected {expected}, got {got}")]
    BadMagic {
        offset: usize,
        expected: String,
        got: String,
        hex_context: Vec<u8>,
    },

    #[error("truncated at offset 0x{offset:x}: expected {expected_bytes} more bytes, have {available} ({context})")]
    Truncated {
        offset: usize,
        expected_bytes: usize,
        available: usize,
        context: &'static str,
    },

    #[error("unknown {kind} {value:#x} at offset 0x{offset:x} ({context})")]
    Unknown {
        offset: usize,
        kind: &'static str,
        value: u64,
        context: &'static str,
    },

    // Further variants added as audit surfaces additional error
    // shapes. Task 3 amendment may refine.
}

impl ParseError {
    /// Format a hex-dump of a byte window around an offset, returning
    /// a string suitable for attaching to error messages.
    pub fn hex_window(data: &[u8], offset: usize, window: usize) -> Vec<u8> {
        let start = offset.saturating_sub(window);
        let end = (offset + window).min(data.len());
        data[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_magic_renders_offset_in_hex() {
        let err = ParseError::BadMagic {
            offset: 0x1234,
            expected: "xclbin2\\0".to_string(),
            got: "deadbeef".to_string(),
            hex_context: vec![0xde, 0xad, 0xbe, 0xef],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x1234"));
        assert!(msg.contains("xclbin2"));
        assert!(msg.contains("deadbeef"));
    }

    #[test]
    fn truncated_message_shows_shortfall() {
        let err = ParseError::Truncated {
            offset: 0x100,
            expected_bytes: 64,
            available: 12,
            context: "AIE Partition header",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x100"));
        assert!(msg.contains("64"));
        assert!(msg.contains("12"));
        assert!(msg.contains("AIE Partition"));
    }

    #[test]
    fn hex_window_extracts_bytes_around_offset() {
        let data: Vec<u8> = (0..=255).collect();
        let window = ParseError::hex_window(&data, 128, 4);
        assert_eq!(window, vec![124, 125, 126, 127, 128, 129, 130, 131]);
    }
}
```

- [x] **Step 2: Register module + run tests**

Add `pub mod error;` to `src/parser/mod.rs`. Add `pub use error::ParseError;` if desired.

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib parser::error 2>&1 | tail -5
```

Expected: 3 tests pass.

- [x] **Step 3: Migrate one call site as pilot**

Pick one `bail!` / `anyhow!` site in `src/parser/xclbin.rs` (XCLBIN magic check at line ~100). Replace with `Err(ParseError::BadMagic { ... }.into())`. Verify tests still pass.

- [x] **Step 4: Commit pilot**

```bash
git add src/parser/error.rs src/parser/mod.rs src/parser/xclbin.rs
git commit -m "$(cat <<'EOF'
feat(parser): ParseError enum + pilot migration

Structured diagnostics for parser fallible boundaries. ParseError
carries offset, expected, got, and hex context window.

First migration: XCLBIN magic check in xclbin.rs. Remaining
parser fallible sites migrate in follow-up commits.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 5: Migrate remaining call sites**

Iterate through the parser module, replacing `bail!` and `anyhow!` with `ParseError` variants where the shape fits. Commit per parser file (one commit each for xclbin.rs, aie_partition.rs, cdo/framing.rs, cdo/syntax.rs, elf.rs). Skip sites where `anyhow::Context` chaining is more appropriate (e.g., wrapping lower-level errors from `zerocopy`).

Each commit message:
```
feat(parser): migrate <file>.rs fallible sites to ParseError
```

- [x] **Step 6: Verify final state**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: no regression from Task 13 baseline.

Estimated commits for Task 14: 5-7 (pilot + per-file migrations + final sweep).

---

## Task 15: Stage 8c -- test fixtures (builders)

**Files:**
- Create: `src/parser/testing/mod.rs`
- Create: `src/parser/testing/xclbin_builder.rs`
- Create: `src/parser/testing/cdo_builder.rs`
- Create: `src/parser/testing/elf_builder.rs`

**Goal:** Fluent builders that construct minimal-valid XCLBIN / CDO / ELF byte streams for parser unit tests. Enables edge-case coverage (truncated sections, malformed headers, unknown opcodes) without real binaries.

- [x] **Step 1: Write failing test for XclbinBuilder**

Create `src/parser/testing/xclbin_builder.rs`:

```rust
//! Minimal XCLBIN builder for parser unit tests.
//!
//! Not a general-purpose XCLBIN generator -- only the subset of
//! sections that parser code paths actually read. Keeps tests
//! independent of real XCLBINs on disk.

/// Fluent builder for a minimal valid XCLBIN byte stream.
pub struct XclbinBuilder {
    // fields: version, sections, partition bytes, etc.
    sections: Vec<(u32, Vec<u8>)>,  // (section_kind, payload)
}

impl XclbinBuilder {
    pub fn new() -> Self {
        Self { sections: Vec::new() }
    }

    /// Add a Partition section with the given bytes.
    pub fn with_partition(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        // Section kind 32 = AiePartition per src/parser/xclbin.rs.
        self.sections.push((32, bytes.into()));
        self
    }

    /// Add a CDO section (nested inside an AIE Partition in real usage,
    /// but builder flattens for test convenience).
    pub fn with_cdo(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.sections.push((32, bytes.into()));
        self
    }

    /// Build the byte stream.
    pub fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        // Write XCLBIN header + magic + section count + offsets + sections
        // (specific layout follows src/parser/xclbin.rs parsing expectations)
        out.extend_from_slice(b"xclbin2\\0");
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for (kind, payload) in &self.sections {
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        out
    }
}

impl Default for XclbinBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_produces_header_only() {
        let bytes = XclbinBuilder::new().build();
        assert_eq!(&bytes[..8], b"xclbin2\\0");
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
    }

    #[test]
    fn with_partition_adds_section() {
        let bytes = XclbinBuilder::new()
            .with_partition(vec![0xde, 0xad, 0xbe, 0xef])
            .build();
        // Section count = 1
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
    }
}
```

**Note:** The exact byte layout above is illustrative. The real implementation must match what `src/parser/xclbin.rs::Xclbin::parse` actually reads. The builder's test pairs it against the parser: construct bytes with `XclbinBuilder`, parse with `Xclbin::parse`, assert round-trip.

- [x] **Step 2: Create `cdo_builder.rs` + `elf_builder.rs` analogously**

`cdo_builder.rs`: minimal CDO byte stream. Methods: `.with_version(v)`, `.with_write32(addr, val)`, `.with_mask_write32(addr, mask, val)`, `.with_dma_write(addr, data)`, `.with_marker(id)`, `.build()`.

`elf_builder.rs`: minimal AIE ELF. Methods: `.with_program_bytes(b)`, `.with_data_bytes(b)`, `.with_symbol(name, addr)`, `.build()`.

Each builder writes enough bytes to make `src/parser/cdo.rs::Cdo::parse` / `src/parser/elf.rs::AieElf::parse` succeed.

- [x] **Step 3: Create `mod.rs` re-exporting**

```rust
//! Test fixtures for parser unit tests.
//!
//! Builders for minimal-valid XCLBIN / CDO / ELF byte streams.
//! Enable parser tests to cover edge cases without real-binary
//! dependencies.

pub mod xclbin_builder;
pub mod cdo_builder;
pub mod elf_builder;

pub use xclbin_builder::XclbinBuilder;
pub use cdo_builder::CdoBuilder;
pub use elf_builder::ElfBuilder;
```

Add `#[cfg(test)] pub mod testing;` to `src/parser/mod.rs` (or `pub mod` if builders are useful to consumers outside tests -- default `#[cfg(test)]` for hygiene).

- [x] **Step 4: Round-trip tests per builder**

For each builder: construct minimal bytes, parse with corresponding parser, verify parsed structure matches what the builder said.

Example for XclbinBuilder:
```rust
#[cfg(test)]
#[test]
fn xclbin_builder_round_trips_through_parser() {
    let bytes = XclbinBuilder::new()
        .with_partition(vec![0; 64])
        .build();
    let parsed = crate::parser::xclbin::Xclbin::parse(&bytes).unwrap();
    assert_eq!(parsed.sections().count(), 1);
}
```

- [x] **Step 5: Migrate one existing parser test to use builders (pilot)**

Pick one test in `src/parser/xclbin.rs` or `src/parser/cdo/syntax.rs` that currently reads a file from disk. Rewrite to use `XclbinBuilder` or `CdoBuilder`. Verify it still passes.

- [x] **Step 6: Verify + commit**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib parser::testing 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

```bash
git add src/parser/testing/ src/parser/mod.rs
git commit -m "$(cat <<'EOF'
feat(parser): test-fixture builders (XclbinBuilder, CdoBuilder,
ElfBuilder)

Fluent builders for minimal-valid XCLBIN / CDO / ELF byte streams.
Parser unit tests can now exercise edge cases (truncated headers,
unknown opcodes, malformed sections) without depending on real
binaries. One pilot migration included.

Generated using Claude Code.
EOF
)"
```

Estimated commits for Task 15: 2-4 (one per builder, plus pilot migration).

---

## Task 16: Stage 8c -- ELF loading deduplication

**Files:**
- Modify: `src/parser/elf.rs` (add canonical `load_into`)
- Modify: `src/device/host_memory.rs`
- Modify: `src/device/mod.rs`
- Modify: `src/integration/elfanalyzer.rs`
- Modify: `src/interpreter/decode/crossref.rs`
- Modify: `src/interpreter/decode/decoder.rs`
- Modify: `src/interpreter/engine/coordinator.rs`
- Modify: `src/interpreter/test_runner.rs`
- Modify: `src/main.rs`
- Modify: `src/testing/xclbin_suite.rs`

**Goal:** One canonical `AieElf::load_into(&mut CoreMemory)` + helper methods. Five existing consumers migrate; duplicated ELF-parsing conventions consolidate.

- [x] **Step 1: Read audit §6 -- ELF consumer table**

Task 3's amendment surfaced the needs-per-consumer breakdown. Re-read `docs/arch/subsys8-audit.md` §6. Identify:
- Which consumers need `load_into` (program/data writes to CoreMemory).
- Which need `iter_sections` / `resolve_symbol` / other helpers.
- The superset API to land.

- [x] **Step 2: Write failing test for `AieElf::load_into`**

In `src/parser/elf.rs`:

```rust
#[cfg(test)]
#[test]
fn load_into_writes_program_bytes_to_core_memory() {
    use crate::parser::testing::ElfBuilder;
    use crate::device::tile::CoreMemory; // or wherever CoreMemory lives

    let elf_bytes = ElfBuilder::new()
        .with_program_bytes(vec![0x11, 0x22, 0x33, 0x44])
        .build();
    let elf = AieElf::parse(&elf_bytes).unwrap();
    let mut mem = CoreMemory::new_for_test();
    elf.load_into(&mut mem).unwrap();
    // Verify program bytes landed at the expected program-memory offset.
    assert_eq!(mem.read_program(0, 4), &[0x11, 0x22, 0x33, 0x44]);
}
```

- [x] **Step 3: Implement `AieElf::load_into`**

Pattern: pull the ELF-to-CoreMemory logic from `src/interpreter/test_runner.rs` (the most complete existing implementation) and wrap it as a method on `AieElf`. Other consumers replicate with minor drift; this canonicalises.

```rust
impl AieElf {
    /// Write program + data sections into the given CoreMemory.
    /// Single canonical ELF-loading implementation; replaces the
    /// scattered copies in test_runner, coordinator, host_memory,
    /// etc.
    pub fn load_into(&self, mem: &mut CoreMemory) -> Result<()> {
        for section in self.sections() {
            match section.region() {
                MemoryRegion::Program => mem.write_program(section.offset(), section.bytes())?,
                MemoryRegion::Data => mem.write_data(section.offset(), section.bytes())?,
                MemoryRegion::Symbol => {} // metadata only
            }
        }
        Ok(())
    }
}
```

(Exact signature depends on `CoreMemory`'s API; adjust to match.)

- [x] **Step 4: Commit the canonical API**

```bash
git add src/parser/elf.rs
git commit -m "$(cat <<'EOF'
feat(parser): AieElf::load_into canonical loader

Consolidates ELF-to-CoreMemory logic into a single canonical method
on AieElf. Follow-up commits migrate the 5 consumer sites to use
it instead of their local reimplementations.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 5: Migrate each consumer, one commit per consumer**

For each of: `test_runner.rs`, `coordinator.rs`, `decoder.rs`, `crossref.rs`, `host_memory.rs`, `integration/elfanalyzer.rs`, `main.rs`, `testing/xclbin_suite.rs`:

- Find the local ELF-loading snippet.
- Replace with `elf.load_into(&mut mem)?;`.
- Delete now-dead helpers.
- Verify `cargo test --lib` green.
- Commit: `refactor(<file>): use AieElf::load_into canonical loader`.

- [x] **Step 6: Verify final state + bridge smoke**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -10
```

Expected: no regressions; bridge smoke green.

Estimated commits for Task 16: 1 canonical + 5-8 per-consumer migrations = 6-9 commits.

---

## Task 17: Stage 8c -- control-packet parser alignment

**Goal:** Audit §7 concluded "coincidental only" -- `src/device/control_packets/parser.rs`
shares 0 framing primitives with `src/parser/*`. They are structurally different parsers
(control-packets are NPU runtime command dispatch; src/parser/* is XCLBIN/CDO/ELF static
configuration). No shared module. Task 17 is a one-line documentation commit:

**Files:**
- Modify: `src/device/control_packets/parser.rs`

- [x] **Step 1: Add explanatory module-level comment**

At the top of `src/device/control_packets/parser.rs` (above existing imports), add:

```rust
//! Control packet parser (runtime command dispatch).
//!
//! Structurally distinct from `src/parser/*` (the XCLBIN/CDO/ELF static
//! configuration parsers): control packets carry NPU runtime commands
//! wrapped in a 1-word header + payload, whereas CDO carries a 20-byte
//! header + variable-length command stream, and XCLBIN is a Xilinx
//! container. No framing primitives overlap. Subsystem 8 audit §7
//! (docs/arch/subsys8-audit.md) evaluated shared-module extraction
//! and found zero overlap -- the two parsers stay separate by design.
```

Preserve any existing module-level doc comment by appending to it rather than replacing.

- [x] **Step 2: Verify build + tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: 2686 passed (unchanged by doc-only edit).

- [x] **Step 3: Commit**

```bash
git add src/device/control_packets/parser.rs
git commit -m "$(cat <<'EOF'
docs(control_packets): document non-overlap with src/parser

Subsystem 8 audit §7 evaluated shared-module extraction between the
control-packet parser and the src/parser/* static-configuration
parsers and found zero framing-primitive overlap. The two parsers
remain separate by design. Module-level comment added to
src/device/control_packets/parser.rs pointing at the audit for
future readers.

Generated using Claude Code.
EOF
)"
```

---

## Task 18: Stage 8c close + tag + Phase 1 complete

**Files:**
- Modify: `NEXT-STEPS.md`

**Goal:** Close Stage 8c, tag `phase1-subsys-parser-ergonomics`, close Phase 1.

- [x] **Step 1: Full verification gates**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
cargo build --release 2>&1 | tail -3
cargo build -p xdna-emu-ffi 2>&1 | tail -3
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8c-bridge.log
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys8-8c-isa.log
```

Compare against all three prior baselines (pre-subsystem, 8a, 8b). Expect no regressions.

- [x] **Step 2: Apply tag**

```bash
git tag phase1-subsys-parser-ergonomics HEAD
git tag | grep phase1
```

- [x] **Step 3: Transition NEXT-STEPS.md to Phase 1 complete**

Edit NEXT-STEPS.md:
- Change "Last updated" header to current date.
- Change "Latest tag" to `phase1-subsys-parser-ergonomics`.
- Update Subsystem 8 row in the pass-order table to "Done".
- Add a new section at top of document: `## Phase 1 Complete`. Note all 8 subsystem tags. Transition text: Phase 2 hygiene is next, with its own brainstorm/plan cycle.
- Optionally tag `phase1-complete` after this commit (at user's discretion).

- [x] **Step 4: Commit**

```bash
git add NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 8 complete -- Phase 1 done

Subsystem 8 (Parser) tagged at phase1-subsys-parser-ergonomics.
Phase 1b is complete: all eight subsystems done, all arch seams
landed, all per-seam design notes written.

NEXT-STEPS advances to Phase 2 hygiene (separate brainstorm/plan
cycle when that work begins).

Generated using Claude Code.
EOF
)"
```

- [x] **Step 5: Notify user of Phase 1 completion**

> "Subsystem 8 complete. Phase 1 of the device-family refactor is done -- all eight subsystems landed, all per-seam design notes written, all arch data migrated to `xdna-archspec`. Tags:
>
> - phase1-subsys-regs-mem (1)
> - phase1-subsys-tile-topo (2)
> - phase1-subsys-dma (3)
> - phase1-subsys-locks (4)
> - phase1-subsys-stream-switch (5)
> - phase1-subsys-isa-decode (6)
> - phase1-subsys-isa-execute (7)
> - phase1-subsys-parser-arch / -coupling / -ergonomics (8)
>
> Next is Phase 2 hygiene (not automatically started). Or `phase1-complete` milestone tag on master merge, at your discretion."

---

## Known Pre-Existing Failures (carry through, do not block)

These predate Subsystem 8. Do not investigate during this subsystem; they're noted so downstream agents don't chase them:

- `bd_chain_repeat_on_memtile` bridge EMU deadlock (both Chess and Peano).
- `dma_task_large_linear` Peano EMU timeout.
- `objectfifo_repeat/init_values_repeat` Peano EMU timeout.
- `cargo test -p xdna-archspec --lib::test_full_parse_all_devices` -- pre-existing 13 vs 12 device count mismatch. Still passes at HEAD per baseline capture; if it fails during Subsystem 8, it's likely regression-caused; investigate before tagging.

---

## Self-Review Checklist (run at plan-end, before invoking executing-plans)

- [x] Every task has exact file paths (Create/Modify/Delete) listed upfront.
- [x] Every step produces a commit or a verification gate; no mystery steps.
- [x] Global invariants restated per-task where specific values apply (pass counts, expected outputs).
- [x] No `TBD` / `TODO` / vague steps except explicit amendment markers (Tasks 4, 5, 10, 11, 12, 17 flag amendment-pending explicitly and say when they'll be filled).
- [x] Amendment protocol is explicit: when to amend, what to amend, which task gates on the amendment.
- [x] Tags + tag verification commands appear in each stage-close task.
- [x] Known pre-existing failures are listed.
- [x] Baseline + per-stage logs teed to /tmp/claude-1000/ with explicit filenames.

---

## Plan amendment history

- 2026-04-23: initial plan, written after spec committed (571 lines).
- (future): Task 3 amendments per audit findings, Task 6 post-gate amendments per user confirmation, Task 17 amendment per audit §7.
