# Agent Team: Project Overhaul Plan

Date: 2026-03-05
Status: Ready to execute after compaction

## Context

The emulator has grown significantly. Documentation is stale, the README
claims specific test counts that go stale within sessions, and several
subsystems have disconnected pieces. This plan deploys a team of parallel
agents to address everything at once.

## Agent Team Layout

### Agent 1: P2 Fix -- DMA Zero-Padding Element-vs-Word Bug (Code)

**Goal**: Fix the 4 failing bridge tests caused by zero-padding treating
element counts as word counts.

**Inputs**: `memory/p2-padding-root-cause.md` has the full root cause analysis.

**Tasks**:
1. Read `src/device/dma/transfer.rs` (ZeroPadState) and `src/device/dma/bd.rs`
   (to_bd_config)
2. Determine element width from BD context (buffer_length / d0_wrap ratio,
   or from the CDO write that programs the BD)
3. Convert padding fields to word units in `to_bd_config()` or make
   ZeroPadState element-aware
4. Update unit tests in transfer.rs to cover sub-word element types (i8, i16)
5. Run `cargo test --lib` to verify no regressions
6. Run bridge tests for the 4 affected tests: add_12_i8, add_21_i8, add_378, add_256

**Isolation**: Can use a worktree -- changes are in `src/device/dma/` only.

---

### Agent 2: README and Top-Level Documentation (Docs)

**Goal**: Rewrite README.md and update ROADMAP.md to reflect actual project state.

**Principles**:
- No hardcoded test counts (say "run `cargo test --lib` for current count")
- No stale claims about what works or doesn't -- describe the architecture
  and capabilities, not numerical state
- Keep it useful for newcomers and contributors
- Reference component docs (`.claude/components/`) for details

**Tasks**:
1. Read current README.md, ROADMAP.md, and all docs/roadmap/phase*.md
2. Read current `cargo test --lib` output for a sense of coverage
3. Audit every claim in README against reality (does the GUI work? does
   XRT bridge work? what actually runs?)
4. Rewrite README: clear project description, quick start, architecture
   overview, build/test instructions, links to deeper docs
5. Update ROADMAP.md: remove stale confidence markers, update status
6. Check docs/roadmap/ phase files for obviously wrong claims

---

### Agent 3: Codebase Audit -- Redundancy and Dead Code (Code/Explore)

**Goal**: Find dead code, duplicated logic, unused files, and stale modules.

**Tasks**:
1. Scan `src/` for unused modules (`#[allow(dead_code)]`, unused imports)
2. Check `tools/` for scripts that were superseded (old trace tools, old
   planners -- some were removed but check for stragglers)
3. Look for duplicated logic across subsystems (e.g., lock ID mapping code
   in multiple places, BD parsing done differently in different modules)
4. Check if any `.claude/components/` docs reference deleted code
5. Report findings as a list: file, issue, recommendation (delete/merge/fix)

**Note**: READ ONLY. Do not delete or modify anything. Report findings.

---

### Agent 4: Integration Wiring Audit (Explore)

**Goal**: Find pieces that were supposed to be connected but aren't.

**Tasks**:
1. Check `src/integration/mod.rs` -- is it still a placeholder?
2. Check if the GUI (`src/visual/`) can actually observe emulator state or
   if it's been fully disconnected by architecture changes
3. Check if `npu-test` binary still compiles and its tests are current
4. Check if the XRT plugin (`xrt-plugin/`) is in sync with the current FFI
   (`src/ffi/`) -- any functions declared but not implemented?
5. Check if `tools/aie-device-dump.py` still works with current mlir-aie
6. Check if `tools/trace-inject.py` and `tools/trace-sweep.py` have any
   remaining references to removed code (old planner functions, etc.)
7. Report findings: what's disconnected, what's stale, what needs wiring.

---

### Agent 5: Bug Triage and Test Status Report (Explore)

**Goal**: Produce an accurate snapshot of what passes, what fails, and why.

**Tasks**:
1. Read the latest bridge test results from `/tmp/emu-bridge-results-20260305/`
2. For each failing test, categorize the failure:
   - Emulator bug (data wrong)
   - Hardware bug (TDR, IOMMU)
   - Missing feature (unimplemented instruction, unsupported pattern)
   - Compiler difference (Chess OK, Peano fails)
3. Cross-reference with known issues in `memory/MEMORY.md`
4. Read the trace sweep results (if available) for trace accuracy status
5. Produce a summary table: test name, HW status, EMU status, failure category,
   known issue reference (if any)

---

## Execution Notes

- Agents 1 and 2 modify files -- they should use worktrees or coordinate
  to avoid conflicts (Agent 1 touches src/device/dma/, Agent 2 touches
  docs/ and README)
- Agents 3, 4, 5 are read-only explorers -- can run in parallel freely
- All agents should avoid claiming things are "fixed" or "done" without
  running verification (Agent 1 must run tests before claiming success)
- After all agents report, consolidate findings and commit changes

## Session Flow

1. Start all 5 agents in parallel
2. Wait for results
3. Review Agent 1's fix (P2) -- test and commit
4. Review Agent 2's doc rewrites -- edit and commit
5. Triage findings from Agents 3, 4, 5 into actionable items
6. Commit everything, update MEMORY.md

## Prerequisites

- mlir-aie on `fix/pathfinder-addfixedconnection-source-port` branch (or
  merge back to main if the fix is solid)
- Trace sweep results in `/tmp/emu-bridge-results-20260305/` (from today's run)
- `cargo test --lib` passing (baseline before changes)
