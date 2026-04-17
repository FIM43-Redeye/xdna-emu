# Emulator Accuracy Sweep -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development
> to implement this plan. Each task dispatches parallel agents on isolated
> files. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring xdna-emu to full hardware fidelity through systematic
cross-verification against three oracles (aie-rt, aiesimulator, real NPU).

**Architecture:** Four sequential workstreams, each internally parallelized
across independent agents. WS2 (foundation audit) first, then WS1 (vector
compute), WS3 (aiesim bridge, gated), WS4 (test expansion). Hybrid triage:
trivial fixes in-place, complex fixes cataloged.

**Tech Stack:** Rust, bash, aie-rt (C), aietools Python models (read-only
reference), VCD comparison infrastructure (already built).

**Spec:** `docs/superpowers/specs/2026-03-12-emulator-accuracy-sweep-design.md`

---

## Chunk 1: Setup and WS2 (aie-rt Cross-Verification)

### Task 0: Setup

**Files:**
- Create: `docs/accuracy-sweep/catalog.md`
- Create: `docs/accuracy-sweep/verified/.gitkeep`
- Create: `docs/accuracy-sweep/results/.gitkeep`

- [ ] **Step 1: Create output directory structure**

```bash
mkdir -p docs/accuracy-sweep/verified docs/accuracy-sweep/results
touch docs/accuracy-sweep/verified/.gitkeep docs/accuracy-sweep/results/.gitkeep
```

- [ ] **Step 2: Create catalog header**

Create `docs/accuracy-sweep/catalog.md`:
```markdown
# Emulator Accuracy Sweep -- Bug Catalog

Structured divergence entries from all workstreams. Each workstream
appends to per-subsystem catalog files (`catalog-<subsystem>.md`),
merged here after each workstream completes.

## Summary

| Subsystem | Critical | High | Medium | Low | Fixed |
|-----------|----------|------|--------|-----|-------|
| (populated after WS2) | | | | | |
```

- [ ] **Step 3: Commit**

```bash
git add docs/accuracy-sweep/
git commit -m "chore: create accuracy sweep output directories"
```

---

### Task 1: WS2 -- aie-rt Cross-Verification (6 parallel agents)

**Goal:** Audit every device subsystem against aie-rt source code. Produce
match reports, catalog divergences, fix trivial issues in-place.

**Dispatch 6 agents in parallel**, each in its own worktree. Each agent:
1. Reads the aie-rt reference files for its subsystem
2. Reads the corresponding emulator files
3. Compares function-by-function
4. Writes a match report to `docs/accuracy-sweep/verified/<subsystem>.md`
5. Writes divergences to `docs/accuracy-sweep/catalog-<subsystem>.md`
6. Applies trivial fixes directly (with tests)
7. Commits its changes

**Agent F: DMA Engine**
- Our code: `src/device/dma/` (9 files, ~8,800 lines total)
  - `mod.rs` (422) -- module root, DmaEngine type, re-exports
  - `bd.rs` (941) -- BD field parsing
  - `channel.rs` (530) -- channel FSM
  - `engine.rs` (3,903) -- DMA engine orchestration
  - `addressing.rs` (798) -- multi-dimensional addressing
  - `transfer.rs` (1,461) -- data transfer execution
  - `compression.rs` (264), `stream_io.rs` (332), `timing.rs` (125)
- **Owns all 9 DMA files.** Agent I audits shim-specific behavior but catalogs
  fixes to DMA files rather than applying directly (Agent F applies them).
- aie-rt: `../aie-rt/driver/src/dma/xaie_dma_aieml.c` and `.h`
- Existing validation: `src/device/aiert_validation.rs` (build on this)
- Key audit points:
  - BD field parsing matches `XAie_DmaBdSetLock`, `XAie_DmaBdSetAxi`
  - Channel FSM matches `XAie_DmaChannelEnable`/`Disable`
  - Completion polling matches `_XAieMl_DmaWaitForDone`
  - Multi-dim addressing matches `XAie_DmaBdSetMultiDimAddr`
  - Zero-padding mode matches `XAie_DmaBdSetPad`
- Output: `docs/accuracy-sweep/verified/dma.md`, `docs/accuracy-sweep/catalog-dma.md`

**Agent G: Lock Arbiter**
- Our code: `src/device/tile.rs` lock section (~500 lines within 2,897 total)
- aie-rt: `../aie-rt/driver/src/locks/xaie_locks_aieml.c` and `.h`
- Existing validation: `src/device/aiert_validation.rs`
- Key audit points:
  - Acquire/release value semantics match `XAie_LockAcquire`/`Release`
  - Overflow/underflow behavior (6-bit unsigned, 0-63 range)
  - Quadrant routing: IDs 48-63 = own tile memory module locks 0-15
  - Lock register stride: 16 bytes apart (not 4)
- Output: `docs/accuracy-sweep/verified/locks.md`, `docs/accuracy-sweep/catalog-locks.md`

**Agent H: Stream Switch**
- Our code: `src/device/stream_switch.rs` (2,429 lines)
- aie-rt: `../aie-rt/driver/src/stream_switch/xaie_ss.c` and `.h`
- Key audit points:
  - Port type assignments per bundle (in `src/device/stream_switch.rs` and `src/lib.rs` arch module)
  - Packet routing table structure (header matching, mask, ID routing)
  - Circuit-mode setup/teardown sequence
  - Arbiter/msel configuration
- Output: `docs/accuracy-sweep/verified/stream-switch.md`, `docs/accuracy-sweep/catalog-stream-switch.md`

**Agent I: Shim/MemTile/Cascade**
- Our code: `src/interpreter/execute/cascade.rs` (340 lines)
- Reads (but does not modify): `src/device/dma/` shim-related code paths
- aie-rt: `../aie-rt/driver/src/global/xaiemlgbl_params.h` + shim sections
- **Note**: Agent I catalogs DMA-file fixes for Agent F to apply. Agent I
  directly owns only `cascade.rs` and its own report/catalog files.
- Key audit points:
  - Shim BD buffer length field (18-bit, 0x3FFFF)
  - Shim mux port mapping (`shim_mux_ports["DMA"]` not `switchbox_ports["DMA"]`)
  - MemTile lock addressing (8-bit field, 192-entry space: W/Own/E at 0/64/128)
  - Cascade register setup, data movement semantics (currently stub)
- Output: `docs/accuracy-sweep/verified/shim-memtile.md`, `docs/accuracy-sweep/catalog-shim-memtile.md`

**Agent I+: Trace Unit**
- Our code: `src/device/trace_unit.rs` (725 lines)
- aie-rt: `../aie-rt/driver/src/events/xaie_events_aieml.h` + `events/xaie_events.c` + AM025 event tables
- Key audit points:
  - Event routing register layout
  - Trace packet format (header fields, timestamp encoding)
  - Event ID mapping (compute vs memory module events)
  - Trace control registers (start/stop triggers)
- Output: `docs/accuracy-sweep/verified/trace-unit.md`, `docs/accuracy-sweep/catalog-trace-unit.md`

**Agent J: Memory Model**
- Our code: `src/device/tile.rs` (memory sections), `src/device/banking.rs` (125 lines), `src/device/host_memory.rs` (626 lines)
- aie-rt: AM025 JSON + `../aie-rt/driver/src/global/` headers
- Key audit points:
  - Bank conflict rules (bank width should derive from ArchModel, not hardcoded)
  - Address space layout (local memory, cross-tile offsets)
  - Cross-tile CardDir routing (`_XAie_GetTargetTileLoc`)
  - Memory bank width derivation (currently `BANK_WIDTH_BYTES = 128/8`)
- Output: `docs/accuracy-sweep/verified/memory.md`, `docs/accuracy-sweep/catalog-memory.md`

- [ ] **Step 1: Dispatch all 6 agents in parallel worktrees**

Each agent prompt should include:
- The subsystem scope and file paths listed above
- Instructions to read aie-rt FIRST, then compare to our code
- The output format (match report + catalog entries)
- Permission to fix trivial issues in-place with tests
- Instruction to commit all changes before completing

- [ ] **Step 2: Merge agent worktrees**

After all agents complete, merge their worktree branches into dev.
Resolve any conflicts (unlikely -- agents own distinct files).

- [ ] **Step 3: Merge catalog files**

Combine individual `catalog-<subsystem>.md` files into unified
`docs/accuracy-sweep/catalog.md`. Update the summary table.

- [ ] **Step 4: Regression gate**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
./scripts/emu-bridge-test.sh --no-hw
```

All tests must pass before proceeding to WS1.

- [ ] **Step 5: Commit merged catalog**

```bash
git add docs/accuracy-sweep/
git commit -m "docs: WS2 complete -- aie-rt cross-verification results"
```

---

## Chunk 2: WS1 (Vector Compute Completion)

### Task 2: WS1 -- Vector Compute Completion (5 parallel agents)

**Goal:** Complete vector compute implementation: migrate ops to semantic
dispatch, implement config word extraction, fill remaining gaps.

**WS2 carryover -- CASCADE-2:** Cascade instructions are not connected to
FIFO data movement. This crosses the interpreter/device boundary and belongs
in WS1. Agent A should wire cascade put/get instructions in the semantic
dispatcher to the device cascade FIFOs as part of the migration work.

**Dispatch 5 agents in parallel.** Each agent owns a specific file and
does NOT touch files owned by other agents.

**LESSON FROM WS2:** Do NOT use worktrees for audit/read-heavy agents.
Worktrees snapshot at creation time and diverge as other agents modify dev.
WS2 round 1 audited stale code, producing false positives and missing real
bugs. Use worktrees only for agents that need file-level isolation for
concurrent writes. For WS1, worktrees ARE appropriate since agents write
to different files.

**Agent A: Semantic Dispatch Migration**
- Create: `src/interpreter/execute/vector_semantic.rs`
- Modify: `src/interpreter/execute/vector.rs` (3,335 lines -- **only Agent A touches this**)
- Modify: `src/interpreter/execute/semantic.rs` (1,393 lines -- add new dispatch calls)
- Modify: `src/interpreter/execute/mod.rs` (151 lines -- **only Agent A touches this**)
- Reference: existing `vector.rs` dispatch patterns + `semantic.rs` SemanticOp variants
- Scope:
  - Create `vector_semantic.rs` with clean functions for each vector arithmetic/comparison op
  - Migrate vadd, vsub, vmul, vcmp, vmin, vmax, vconvert from `vector.rs`
  - Wire into `execute_semantic()` in `semantic.rs`
  - Register ALL new WS1 modules in `mod.rs` (vector_semantic, vector_config,
    vector_float, vector_matmul_sparse). Agent A is the sole owner of `mod.rs`.
  - Add sparse matmul dispatch hook in `vector.rs` pointing to Agent E's new
    file (Agent E provides the implementation, Agent A wires it in)
  - Legacy `vector.rs` shrinks; remaining ops (SRS/UPS/MatMul/pack/shuffle) stay until later
  - **Do NOT touch**: `vector_pack.rs`, `vector_srs.rs`, `vector_ups.rs`, `vector_validate.rs`
  - Each migrated op gets a unit test in the new file
- Success: `cargo test --lib` passes, migrated ops produce identical results

**Agent B: Config Word Framework**
- Create: `src/interpreter/execute/vector_config.rs`
- Reference: `aietools/data/aie_ml/lib/python_model/model/mulmac.py`, `constants.py`
- Note: `vector_matmul.rs` (997 lines) has hardcoded geometry tables -- read it first
- **Boundary with Agent C**: Agent B owns config word *parsing* (instruction bits
  -> `MatMulConfig` struct). Agent C owns permutation *table entries* and shuffle
  logic. The `MacPermuteConfig` struct in `vector_permute.rs` is Agent C's territory.
- Scope:
  - Parse MatMul config word bits: tile geometry (rows/inner/cols), accumulator mode
  - Parse X/Y buffer sign extension control bits
  - Do NOT parse MAC PMODE -- that is Agent C's scope via `MacPermuteConfig`
  - Export clean API: `MatMulConfig::from_instruction_bits(bits: u32) -> Self`
  - Unit tests for every config variant against Python model expected values
- Success: config word parsing produces correct geometry for all 6 type combinations

**Agent C: MAC Permutation Modes**
- Extend: `src/interpreter/execute/vector_permute.rs` (918 lines)
- Reference: `aietools/data/aie_ml/lib/python_model/model/constants.py` (permute sections), `mulmac.py`
- Scope:
  - Identify which MAC permutation modes are missing (shuffle modes 0-47 exist)
  - Implement missing PMODE_* modes from `constants.py` permutation tables
  - Each mode gets a unit test with known input/output pairs from Python model
- Success: all permutation modes produce correct results for representative inputs

**Agent D: Float/BF16 Accuracy**
- Create: `src/interpreter/execute/vector_float.rs`
- Reference: `aietools/data/aie_ml/lib/python_model/model/bfloat16.py`, `srs.py`, `ups.py`
- Note: existing `vector_srs.rs` (848 lines) and `vector_ups.rs` (520 lines) are
  NOT in scope -- Agent D creates new float utilities, does not modify SRS/UPS files
- Scope:
  - BF16 denormalization (flush-to-zero vs gradual underflow -- which does AIE2 do?)
  - FP32 rounding edge cases for vector ops (ties-to-even vs other modes)
  - NaN propagation rules (quiet vs signaling, payload preservation)
  - IEEE 754 compliance verification for existing bf16-related matmul code
  - Unit tests for edge cases: denorms, infinity, NaN, max/min, zero signs
- Success: float operations match `bfloat16.py` behavior for all edge cases

**Agent E: Sparse MatMul**
- Create: `src/interpreter/execute/vector_matmul_sparse.rs`
- Does NOT modify `vector.rs` or `mod.rs` (Agent A owns those -- Agent A wires
  in the sparse dispatch hook; Agent E only provides the implementation)
- Reference: `aietools/data/aie_ml/lib/python_model/model/mulmac.py` sparse sections
- Scope:
  - Implement real sparse matmul (currently aliased to dense in vector.rs ~line 186)
  - Sparse mask extraction from input vector
  - Zero-skip accumulation (only accumulate non-zero elements)
  - Support all element type combinations that dense matmul supports
  - Unit tests with known sparse patterns and expected accumulator outputs
- Success: sparse matmul produces different (correct) results from dense for sparse inputs

- [ ] **Step 1: Dispatch all 5 agents in parallel worktrees**

Each agent prompt should include:
- The file ownership rules (bold in descriptions above)
- Instructions to read the Python model reference FIRST for behavioral understanding
- Write original Rust -- do not copy Python code
- Comprehensive unit tests for every operation
- Commit all changes before completing

- [ ] **Step 2: Merge agent worktrees**

Merge in order: Agent A first (touches shared files), then B-E (new files only).

- [ ] **Step 3: Regression gate**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
./scripts/emu-bridge-test.sh --no-hw
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: WS1 complete -- vector compute completion"
```

---

## Chunk 3: WS3 (aiesim Reverse Bridge) and WS4 (Bridge Test Expansion)

### Task 3: WS3 -- aiesim Reverse Bridge (3 parallel agents)

**Gate check:** Before starting, verify one of:
- fifield has responded to issue #2955 confirming NPU sim is possible
- Decision made to use Versal (`xcve2802`) target as fallback

If neither gate is met, skip to Task 4 and revisit WS3 later.

**Dispatch 3 agents in parallel**, each in its own worktree.

**Agent K: Bridge Protocol**
- Create: `src/aiesim/mod.rs` (new module)
- Create: `src/aiesim/ess_ffi.rs` -- Rust FFI implementations of `ess_*` callbacks
- Create: `src/aiesim/harness.rs` -- subprocess management for `aie2simmsm`
- Reference: `aietools/include/drivers/aiengine/xioutils.h` (ess_* API surface)
- Reference: `mlir-aie/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp` (wrapper pattern)
- Scope:
  - `ess_Write32`, `ess_Read32`: record (addr, data) to structured transaction log
  - `ess_WriteGM`, `ess_ReadGM`: record bulk memory operations
  - `ess_WriteCmd`: record tile commands (col, row, cmd, words)
  - Subprocess harness: spawn `aie2simmsm --pkg-dir <dir>`, manage lifecycle
  - Collect output VCD after simulation completes
- Success: can launch aiesim on a test sim package and collect its VCD output

**Agent L: State Comparison**
- Create: `src/aiesim/compare.rs` -- state snapshot and diff engine
- Scope:
  - Define `StateSnapshot` struct (register files, memory regions, lock values)
  - Capture snapshot from emulator at checkpoint (DMA complete, core halt)
  - Parse aiesim output files for equivalent state
  - Diff engine: report per-field divergences
  - Output format matches bug catalog entry structure
- Success: can diff two state snapshots and produce structured divergence report

**Agent M: VCD Cross-Check**
- Create: `src/aiesim/vcd_bridge.rs` -- automated VCD comparison orchestration
- Uses: existing `src/vcd/compare.rs`, `src/vcd/mapping.rs`, `src/bin/vcd_compare.rs`
- Scope:
  - Orchestrate: run emulator (with `vcd-recording`), run aiesim, collect both VCDs
  - Invoke `vcd_compare` programmatically on paired outputs
  - Report per-subsystem signal match rates
  - Integrate with tolerance config (`src/vcd/tolerance.rs`)
- Success: end-to-end pipeline produces comparison report for a test kernel

- [ ] **Step 1: Dispatch 3 agents (or skip if gate not met)**
- [ ] **Step 2: Merge worktrees**
- [ ] **Step 3: Regression gate** (`cargo test --lib`)
- [ ] **Step 4: Commit**

---

### Task 4: WS4 -- Bridge Test Expansion (3 agents, partially sequential)

**Agent N runs first** (survey), then **O and P run in parallel** (authoring
+ refactor are independent).

**Agent N: Test Gap Survey** (runs alone)
- Read: `mlir-aie/test/npu-xrt/` (all test directories)
- Read: current bridge test list in `scripts/emu-bridge-test.sh`
- Output: `docs/accuracy-sweep/results/test-gap-survey.md`
- Scope:
  - List every kernel in mlir-aie that we could run but don't
  - Identify hardware features with zero/thin coverage
  - Categorize by test category (smoke, compute, data-movement, sync, stress, control)
  - Prioritize: which gaps would catch the most bugs?
- Success: structured gap list with priorities

**Agent O: Kernel Authoring** (after N completes)
- Create: new test kernels in `mlir-aie/test/npu-xrt/` (or our own test dir)
- Reference: gap list from Agent N
- Scope:
  - Write targeted kernels for top-priority gaps
  - Focus: multi-hop stream routing, cascade flows, concurrent lock contention
  - Each kernel has a known-correct expected output
  - Add kernels to bridge test discovery
- Success: new kernels compile and run on HW, added to bridge test suite

**Agent P: Script Refactor**
- Modify: `scripts/emu-bridge-test.sh` (1,894 lines -- becomes thin orchestrator)
- Create: `scripts/bridge/` directory with component scripts
- Scope:
  - Extract phases into: `discover.sh`, `compile.sh`, `run-hw.sh`, `run-emu.sh`,
    `report.sh`, `trace.sh`
  - Extract shared code into: `lib/common.sh`, `lib/config.sh`
  - Create stub: `run-aiesim.sh` (populated when WS3 delivers)
  - Preserve ALL existing flags and behavior -- pure refactor, no behavior changes
  - Each script independently runnable
  - Test: run refactored suite, diff output against original
- Success: `./scripts/emu-bridge-test.sh --no-hw` produces identical output to before

- [ ] **Step 1: Dispatch Agent N alone**
- [ ] **Step 2: Dispatch Agents O and P in parallel worktrees**
- [ ] **Step 3: Merge worktrees**
- [ ] **Step 4: Final regression gate**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
./scripts/emu-bridge-test.sh --no-hw
```

- [ ] **Step 5: Final commit and summary**

```bash
git commit -m "feat: WS4 complete -- bridge test expansion and script refactor"
```

Update `docs/accuracy-sweep/catalog.md` with final summary table showing
all issues found, fixed, and remaining.

- [ ] **Step 6: Hardware validation (requires user presence)**

```bash
./scripts/emu-bridge-test.sh
```

Run full bridge tests including hardware. Compare EMU vs HW results.
This is the final oracle check.

---

### Task 5: Acceptance Checklist

Verify against spec Section 8 success criteria:

- [ ] All migrated vector operations bit-identical to Python models
  (WS1 unit tests pass for all edge cases)
- [ ] Every aie-rt subsystem has match report or fix
  (all files present in `docs/accuracy-sweep/verified/`)
- [ ] aiesim VCD comparison shows zero per-tile divergences
  (WS3 pipeline produces clean report -- or N/A if WS3 gated)
- [ ] Bridge tests cover all major HW features
  (gap survey shows no HIGH-priority untested features)
- [ ] `emu-bridge-test.sh --full` produces structured report
  (JSON summary with per-test pass/fail)
- [ ] Bug catalog: no CRITICAL/HIGH items in OPEN status
  (all resolved or explicitly deferred with justification)
