# Emulator Accuracy Sweep -- Design Specification

**Date**: 2026-03-12
**Status**: Approved
**Goal**: Bring xdna-emu to full hardware fidelity through systematic
cross-verification against three oracles, with parallel agent workstreams
operating on isolated files.

---

## 1. Overview

The xdna-emu emulator is architecturally sound (1,789+ unit tests; real NPU
hardware baseline 40/40 Chess pass; emulator bridge 24/42 Chess pass) but
has specific refinement gaps across vector compute, data movement, and
subsystem semantics. This plan addresses all known gaps
through four sequential workstreams, each internally parallelized across
independent agents.

### Three-Oracle Verification Model

Every subsystem is verified against up to three independent oracles:

| Oracle | What It Validates | Availability |
|--------|-------------------|--------------|
| **aie-rt source code** | Structural correctness: register layouts, state machine logic, field masks, programming sequences | Immediate (open-source, local clone) |
| **aiesimulator (reverse bridge)** | Behavioral correctness: per-tile execution semantics, timing, internal state | Gated on fifield response (issue #2955) or Versal target fallback |
| **Real NPU hardware** | End-to-end correctness: full binary execution, output comparison | Immediate (bridge test infrastructure exists) |

### Triage Model: Hybrid Fix-or-Catalog

- **Trivial fixes** (wrong constant, off-by-one, missing mask): fixed in-place
  by the verification agent that finds them.
- **Complex fixes** (multi-file changes, behavioral redesign): cataloged in
  `docs/accuracy-sweep/catalog.md` with structured entries for dedicated fix
  agents.

---

## 2. Execution Order

Workstreams run sequentially (one active set of parallel agents at a time).
Within each workstream, agents run in parallel on isolated files.

```
WS2: aie-rt Cross-Verification  (foundation -- fixes affect everything downstream)
  |
WS1: Vector Compute Completion   (builds on verified DMA/lock/stream foundation)
  |
WS3: aiesim Reverse Bridge       (likely unblocked by this point)
  |
WS4: Bridge Test Expansion        (consumes all fixes, exercises everything)
```

**Rationale**: WS2 first because if DMA or lock semantics are wrong, vector
compute results will not match either. WS1 second, building on a verified
foundation. WS3 third (gated dependency resolved by then). WS4 last,
exercising the complete corrected emulator.

**Regression gate between workstreams**: Before starting each subsequent
workstream, run `cargo test --lib` and the bridge test suite to verify no
regressions were introduced by the previous workstream's changes.

**Setup step**: Before WS2 begins, create the output directories:
`docs/accuracy-sweep/`, `docs/accuracy-sweep/verified/`,
`docs/accuracy-sweep/results/`.

---

## 3. Workstream 2: aie-rt Cross-Verification

**Oracle**: aie-rt source code (`../aie-rt/driver/src/`, branch `xlnx_rel_v2025.2`)

**Goal**: For every hardware subsystem, read the aie-rt implementation that
programs real silicon, compare to our emulator code, and produce a match
report or divergence catalog entry.

### Agent Decomposition

| Agent | Subsystem | Our Code | aie-rt Reference | Key Audit Points |
|-------|-----------|----------|-------------------|------------------|
| F | DMA Engine | `src/device/dma/` | `dma/xaie_dma_aieml.c/.h` | BD field parsing, channel FSM, completion polling (`_XAieMl_DmaWaitForDone`), multi-dim addressing, zero-padding mode |
| G | Lock Arbiter | `src/device/tile.rs` (lock section) | `locks/xaie_locks_aieml.c/.h` | Acquire/release semantics, value overflow/underflow behavior, quadrant routing (IDs 48-63 mapping) |
| H | Stream Switch | `src/device/stream_switch.rs` | `stream_switch/xaie_ss.c/.h` | Port type assignments per bundle, packet routing table structure, circuit-mode setup/teardown, arbiter behavior |
| I | Shim/MemTile/Cascade | `src/device/dma/` + shim paths + `src/interpreter/execute/cascade.rs` | `global/xaiemlgbl_params.h` + shim sections + cascade registers | Shim BD field differences (18-bit buffer length), shim mux port mapping, memtile lock addressing (8-bit field, 192-entry space), cascade register setup and data movement semantics |
| I+ | Trace Unit | `src/device/trace_unit.rs` | `events/xaie_events_aieml.c` + AM025 event tables | Event routing registers, trace packet format, event ID mapping, trace control registers |
| J | Memory Model | `src/device/tile.rs` (memory sections), `banking.rs`, `host_memory.rs` | AM025 JSON + `global/` headers | Bank conflict rules, address space layout, cross-tile CardDir routing, memory bank width derivation |

### Agent Outputs

**Existing work**: `src/device/aiert_validation.rs` (212 lines) already
contains aie-rt cross-validation tests comparing extracted constants against
the register database. WS2 agents (especially F, G, I) should build on this
existing infrastructure rather than reinventing it.

Each agent produces:

1. **Match report** (`docs/accuracy-sweep/verified/<subsystem>.md`):
   function-by-function comparison with specific file:line citations from
   both our code and aie-rt.

2. **Divergence catalog** (`docs/accuracy-sweep/catalog-<subsystem>.md`):
   per-agent catalog file to avoid merge conflicts during parallel execution.
   A merge step after the workstream completes combines them into a unified
   `catalog.md`.
   ```
   ## [SUBSYSTEM] Description
   - **Severity**: CRITICAL / HIGH / MEDIUM / LOW
   - **Our behavior**: what the emulator currently does
   - **aie-rt behavior**: what the hardware does (with file:line citation)
   - **Impact**: which tests or kernels this affects
   - **Suggested fix**: brief description
   - **Fixed in-place**: yes/no (if trivial)
   ```

3. **Trivial fixes**: applied directly (wrong constant, missing mask,
   off-by-one) with test updates as needed.

---

## 4. Workstream 1: Vector Compute Completion

**Oracle**: aietools Python models (`aietools/data/aie_ml/lib/python_model/model/`)

**Goal**: Complete the vector compute implementation by migrating all
operations to the clean semantic dispatch path, implementing config word
extraction, and filling remaining gaps (sparse matmul, permutation modes,
float edge cases).

### Agent Decomposition

| Agent | File | Scope | Reference |
|-------|------|-------|-----------|
| A | `vector_semantic.rs` (new) | Migrate vector ops from legacy `vector.rs` (~47 SemanticOp variants across 3,335 lines) to clean semantic dispatch. Focus on arithmetic/comparison ops first; SRS/UPS/MatMul already have dedicated files. Legacy file shrinks as code moves out. **Owns `vector.rs` modifications; other WS1 agents do NOT touch it.** | Existing `vector.rs` + `semantic.rs` |
| B | `vector_config.rs` (new) | Config word framework: extract MatMul tile geometry from instruction bits, MAC PMODE permutation mode selection, X/Y buffer sign extension bits. **Note**: `vector_matmul.rs` (997 lines) already has hardcoded geometry tables -- coordinate with Agent C to avoid duplication. | `mulmac.py`, `constants.py` |
| C | `vector_permute.rs` (extend existing, 918 lines) | Add missing MAC permutation modes to existing shuffle/permute infrastructure. Full permutation table derived from `constants.py` permute sections. | `constants.py` (permute sections), `mulmac.py` |
| D | `vector_float.rs` (new) | BF16 denormalization, FP32 rounding edge cases, NaN propagation rules, IEEE 754 compliance for vector ops | `bfloat16.py`, `srs.py` |
| E | `vector_matmul_sparse.rs` (new) | Real sparse matmul support (currently aliased to dense). Sparse mask extraction, zero-skip accumulation. | `mulmac.py` sparse sections |

### Design Constraints

- Each new file has a clean trait/function interface callable from the
  semantic dispatcher.
- Legacy `vector.rs` remains functional throughout migration -- no big-bang
  rewrite. Operations move one at a time.
- Every operation has unit tests that verify against the Python model's
  expected outputs for representative inputs (including edge cases).
- Each agent reads the aietools Python models as behavioral reference,
  understands the hardware facts, then writes original Rust implementations
  with comprehensive tests.

---

## 5. Workstream 3: aiesim Reverse Bridge

**Oracle**: aiesimulator SystemC models (`aie2simmsm`)

**Goal**: Build infrastructure to run both our emulator and aiesimulator on
the same kernel, then compare internal state (registers, memory, locks) and
event traces (VCDs).

### Gate Condition

Blocked until one of:
- fifield confirms NPU simulation is "untested" (not "never") and our
  PR #2956 lands or equivalent fix is accepted
- We fall back to compiling test kernels against `xcve2802` (Versal AIE2
  target) for simulation, using VCD mapping tree coordinate remapping.
  **Note**: the fallback path requires additional design work -- existing
  bridge test kernels are NPU-specific (use NPU instructions, shim DMA
  patterns). A subset of simpler kernels (pure compute, no NPU instructions)
  would need to be identified or authored for Versal compilation. This
  detail is resolved when WS3 begins, informed by fifield's response.

### Agent Decomposition

| Agent | Phase | Scope |
|-------|-------|-------|
| K | Bridge Protocol | Implement `ess_Write32`, `ess_Read32`, `ess_WriteGM`, `ess_ReadGM`, `ess_WriteCmd` as Rust FFI functions that record all transactions to a structured log. Build subprocess harness to launch `aie2simmsm` with a sim package directory. Handle process lifecycle (spawn, feed config, wait for completion, collect VCD). |
| L | State Comparison | At checkpoints (DMA complete, lock transition, core halt), snapshot both emulator and simulator state. Diff register files, memory contents, lock values. Produce structured divergence reports. Integrate with the bug catalog format from WS2. |
| M | VCD Cross-Check | Connect to VCD infrastructure (Tasks 1-15 from prior work). aiesim produces VCDs natively; our emulator produces them via `vcd-recording` feature. Automate `vcd_compare` execution on paired outputs. Report per-subsystem signal match rates. |

### Architecture Note

The bridge operates at the **register transaction level**, not instruction
level. We do not step both simulators in lockstep (different cycle models).
Instead: feed the same CDO configuration to both, run both to completion,
compare final state + event traces. This is complementary to hardware bridge
tests (those compare outputs; this compares internal state).

### Versal Support

This workstream has permanent value beyond NPU validation. The emulator will
eventually support Versal targets (AIE1, AIE2 on FPGA), and aiesimulator is
the primary simulation tool for those platforms. The reverse bridge becomes
the standard validation path for Versal accuracy.

---

## 6. Workstream 4: Bridge Test Expansion

**Oracle**: Real NPU hardware (Phoenix, NPU1)

**Goal**: Expand bridge test coverage to exercise every major hardware feature,
refactor the test infrastructure into maintainable scripts, and produce
structured pass/fail reporting.

### Agent Decomposition

| Agent | Phase | Scope |
|-------|-------|-------|
| N | Test Gap Survey | Audit all test kernels in `mlir-aie/test/npu-xrt/` against our bridge test list. Identify which kernels exist but we are not running, and which hardware features (cascade, multi-core sync, packet routing, control packets, memtile DMA) have zero or thin coverage. Produce prioritized gap list. |
| O | Kernel Authoring | Write targeted test kernels for gaps identified by Agent N. Focus on data movement patterns: multi-hop stream routing, cascade flows, shim-to-memtile-to-compute DMA chains, concurrent lock contention. Each kernel has a known-correct expected output. |
| P | Script Refactor | Break down monolithic `emu-bridge-test.sh` into a script suite with single-responsibility components. |

### Script Suite Structure

```
scripts/
  emu-bridge-test.sh              # Orchestrator (thin, sequences phases)
  bridge/
    discover.sh                   # Find test sources, check compilation cache
    compile.sh                    # Parallel compilation (Chess + Peano)
    run-hw.sh                     # Hardware execution (-j5, serial option)
    run-emu.sh                    # Emulator execution (-j nproc)
    run-aiesim.sh                 # aiesimulator execution (stub until WS3 delivers)
    report.sh                     # Summary generation (text + JSON dashboard)
    trace.sh                      # Trace processing (sweep, trim, merge)
    lib/
      common.sh                   # Shared functions, color output, path resolution
      config.sh                   # Test categories, compiler flags, timeouts
```

Each script is independently runnable and testable. The orchestrator
sequences them based on command-line flags (`--no-hw`, `--chess-only`,
`--sweep`, `--category=data-movement`, etc.).

### Test Categories

| Category | Coverage Target |
|----------|----------------|
| `smoke` | Basic functionality: add_one, passthrough, simple DMA |
| `compute` | Vector arithmetic, matmul, SRS/UPS, MAC |
| `data-movement` | Multi-hop routing, cascade, memtile DMA, shim chains |
| `sync` | Lock contention, multi-core barriers, DMA sync |
| `stress` | Large buffers, all columns active, concurrent DMAs |
| `control` | Control packets, packet routing, overlay configuration |

---

## 7. Shared Artifacts

### Bug Catalog (`docs/accuracy-sweep/catalog.md`)

Structured divergence entries from all workstreams. Format:

```markdown
## [WS#] [SUBSYSTEM] Short description

- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **Found by**: Agent letter (e.g., Agent F)
- **Our behavior**: what the emulator currently does
- **Expected behavior**: what aie-rt / aiesim / hardware does
- **Evidence**: file:line citations, test output diffs
- **Impact**: which tests or kernels this affects
- **Suggested fix**: brief description
- **Status**: OPEN / IN-PROGRESS / FIXED (commit hash)
```

### Match Reports (`docs/accuracy-sweep/verified/`)

Per-subsystem documents confirming "our implementation matches the oracle."
Each report cites specific functions and line numbers from both codebases.
These serve as living documentation of verification status.

### Test Results (`docs/accuracy-sweep/results/`)

Per-workstream test summaries. JSON format for dashboard consumption,
human-readable text for quick review.

---

## 8. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| All migrated vector operations bit-identical to Python models | Unit tests per operation, edge cases included (scope: ops handled by WS1 agents) |
| Every aie-rt subsystem function has match report or fix | Coverage checklist in verified/ directory |
| aiesim VCD comparison: zero per-tile divergences | `vcd_compare` output with per-subsystem tolerance |
| Bridge tests cover all major HW features | Gap survey shows no untested features at HIGH priority |
| `emu-bridge-test.sh --full` produces structured report | JSON summary with per-test pass/fail and divergence details |
| Bug catalog entries all resolved or triaged | No CRITICAL/HIGH items in OPEN status |

---

## 9. Future Considerations

- **Versal support**: aiesim bridge (WS3) becomes the primary validation
  path for AIE1 and AIE2-on-FPGA targets.
- **Agent teams**: when Claude Code supports managed multi-team orchestration,
  workstreams can run truly in parallel with cross-communication.
- **Differential fuzzer**: once all workstreams complete, the emulator is
  accurate enough to serve as one side of a fuzz-test harness (emulator vs
  hardware, random valid kernels).
- **Continuous regression**: the refactored script suite (WS4) becomes a CI
  pipeline that runs on every commit.
