# NPU1 Phase 0 Baseline Integrity Implementation Plan

**Goal:** Make xdna-emu resolve its required local toolchain components
identically from the main checkout and linked worktrees, bind every mlir-aie
query to the selected tree, and reject missing generated architecture inputs.

**Architecture:** One std-only resolver source in `xdna-archspec` supplies
component-specific and complete `ToolchainPaths` resolution. The archspec build
script compiles it directly; the library exposes the same implementation to
the runtime bridge. Existing generators consume resolved paths and fail
loudly. No wrapper, new config format, crate, or dependency is added.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents and no Halo. Each behavior follows a
witnessed RED, the smallest GREEN, and focused regression reruns before its
commit.

**Approved design:**
[`2026-07-29-npu1-phase0-baseline-integrity-design.md`](../specs/2026-07-29-npu1-phase0-baseline-integrity-design.md)

## Baseline

The targeted coverage gate passed 55 tests when all canonical roots were
supplied explicitly. The fresh full-library baseline passed 4,269 tests,
ignored 32, and failed these two tests because runtime `BridgePath` selected a
system Python without `numpy` from the nested worktree:

```text
integration::bridge::tests::test_invoke_trace_events
trace::tests::test_validate_trace_events_passes
```

Those failures are part of Phase 0. The worktree was clean before
implementation.

Until Task 4 removes the hidden path dependency, focused Rust tests use the
known-good explicit environment:

```bash
env MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
    LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie \
    AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src \
    PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
    TABLEGEN_210_PREFIX=/usr/lib/llvm-21 \
    <cargo command>
```

## Task 1: Shared Toolchain Resolver

**Files:**

- Add: `crates/xdna-archspec/src/toolchain_paths.rs`
- Modify: `crates/xdna-archspec/src/lib.rs`
- Add: `crates/xdna-archspec/tests/toolchain_paths.rs`

### RED

- [ ] Add temporary-layout tests for:
  - main checkout and nested `.worktrees/<name>` discovery;
  - `MLIR_AIE_PATH`, `MLIR_AIE_DIR`, `LLVM_AIE_PATH`,
    `LLVM_AIE_DIR`, `AIE_RT_PATH`, and `NPU_WORK_DIR`;
  - component-specific precedence;
  - blank and invalid configured paths failing without fallback;
  - missing AM025, AIE2 TableGen, llvm-config, and aie-rt sentinels;
  - `AIE_RT_PATH` resolving `driver/src`; and
  - canonical absolute results.
- [ ] Inject environment values into the resolver rather than mutating the
  process environment, so tests remain parallel-safe.
- [ ] Run:

```bash
cargo test -p xdna-archspec --test toolchain_paths
```

- [ ] Require failure because the resolver API does not exist.

### GREEN

- [ ] Implement only the approved precedence and sentinel checks with
  `std::path`, `std::env`, and `std::fs`.
- [ ] Keep per-component resolution available so runtime mlir-aie queries do
  not require llvm-aie or aie-rt.
- [ ] Distinguish absent automatic discovery from an invalid configured
  source; configured mistakes are errors.
- [ ] Emit actionable errors naming component, source, candidate, and missing
  sentinel.
- [ ] Re-run the focused test and require all cases pass.
- [ ] Run `cargo fmt --all -- --check` and `git diff --check`.
- [ ] Commit the resolver and tests.

## Task 2: Explicit mlir-aie Python Provenance

**Files:**

- Modify: `tools/test_mlir_aie_bridge_topology.py`
- Modify: `tools/mlir-aie-bridge.py`

### RED

- [ ] Add subprocess-isolated fake-package tests proving:
  - an explicit root wins over an importable ambient fake package;
  - a broken explicit root fails instead of importing ambient bindings; and
  - the reported binding file is under the selected root.
- [ ] Run:

```bash
python3 -m pytest tools/test_mlir_aie_bridge_topology.py
```

- [ ] Require the explicit-root cases to fail under the current ambient-first
  implementation.

### GREEN

- [ ] Search an explicit root's `build/python` and `install/python` before any
  ambient import.
- [ ] Reject a broken explicit root and a binding imported from outside it.
- [ ] Report the canonical binding path in `device-model` and `trace-events`
  JSON.
- [ ] Preserve ambient discovery only when no explicit root was supplied.
- [ ] Re-run the focused pytest module and commit the bridge change.

## Task 3: Fail-Loud aie-rt Extraction

**Files:**

- Add: `crates/xdna-archspec/build_helpers/aiert.rs`
- Modify: `crates/xdna-archspec/build_helpers/mod.rs`
- Add: `crates/xdna-archspec/tests/aiert_build_support.rs`
- Modify: `crates/xdna-archspec/build.rs`

### RED

- [ ] Add direct tests of the exact build helper for:
  - missing `global/xaiemlgbl_reginit.c`;
  - an unavailable preprocessor executable;
  - a preprocessor that exits non-zero; and
  - missing required compute, memtile, shim, master, or slave generated
    table categories.
- [ ] Use temporary real files and executables, not source-text assertions.
- [ ] Run:

```bash
cargo test -p xdna-archspec --test aiert_build_support
```

- [ ] Require failure because fail-loud support does not exist.

### GREEN

- [ ] Move only preprocessor execution and required-category validation into
  the testable helper.
- [ ] Return detailed `Result` errors and make `build.rs` abort on them.
- [ ] Delete `write_aiert_stubs`; no absence path emits architecture
  constants.
- [ ] Preserve the existing parsers and generators.
- [ ] Re-run the focused tests and commit the aie-rt change.

## Task 4: Wire Every Build Consumer

**Files:**

- Modify: `.cargo/config.toml`
- Modify: `crates/xdna-archspec/build.rs`

### RED

- [ ] Use a new ignored target directory and run:

```bash
env -u NPU_WORK_DIR \
    -u MLIR_AIE_PATH -u MLIR_AIE_DIR \
    -u LLVM_AIE_PATH -u LLVM_AIE_DIR \
    -u AIE_RT_PATH -u PYTHONPATH \
    -u TABLEGEN_210_PREFIX \
    CARGO_TARGET_DIR="$PWD/build/phase0-red-target" \
    cargo test -p xdna-archspec --lib coverage::
```

- [ ] Require failure at the current linked-worktree path assumption.

### GREEN

- [ ] Compile the shared resolver source into `build.rs` and resolve all three
  components once.
- [ ] Feed those roots to AM025 loading, TableGen extraction, decoder FFI,
  explicit-root trace generation, and aie-rt extraction.
- [ ] Require the trace bridge's reported binding path to remain inside the
  resolved mlir-aie root.
- [ ] Remove the checkout-relative `TABLEGEN_210_PREFIX` from
  `.cargo/config.toml`; retain user overrides and system LLVM 21 discovery.
- [ ] Emit Cargo rebuild triggers for every resolver input and selected
  sentinel.
- [ ] Re-run the sanitized targeted gate in the ordinary target and a second
  new ignored target directory.
- [ ] Run `cargo test -p xdna-archspec`.
- [ ] Commit the build-consumer change.

## Task 5: Runtime Bridge Uses the Same Resolver

**Files:**

- Modify: `src/integration/bridge.rs`
- Modify direct callers in:
  - `src/integration/chess_build.rs`
  - `src/trace/mod.rs`

### RED

- [ ] Add a controlled temporary-layout test whose fake interpreter accepts
  only:

```text
<bridge-script> --mlir-aie-path <resolved-root> <subcommand>
```

- [ ] Require the current `BridgePath` to select the wrong interpreter or omit
  the explicit root.
- [ ] Retain the already witnessed two-test full-library RED.

### GREEN

- [ ] Resolve only mlir-aie through the shared archspec resolver.
- [ ] Prefer the resolved root's executable ironenv Python; otherwise use
  `python3`.
- [ ] Store the resolved mlir-aie root in `BridgePath` and pass it before every
  bridge subcommand.
- [ ] Preserve optional absence, but propagate invalid configured paths.
- [ ] Update callers for the explicit discovery result without swallowing
  configuration errors.
- [ ] Run the focused runtime bridge and trace tests.
- [ ] Run `cargo test --lib` and require the two baseline failures to close
  without new failures.
- [ ] Commit the runtime bridge change.

## Task 6: Documentation and Acceptance

**Files:**

- Modify: `AGENTS.md`
- Modify: `docs/operations.md`
- Modify: `docs/components/tablegen.md`
- Update:
  `docs/superpowers/specs/2026-07-29-npu1-phase0-baseline-integrity-design.md`

- [ ] Document the resolution order, aliases, fail-closed configured paths,
  `AIE_RT_PATH=.../driver/src`, and activation-as-convenience contract.
- [ ] Correct the host LLVM 21 versus resolved llvm-aie distinction.
- [ ] Record RED/GREEN counts and exact fresh-target acceptance evidence in
  the design.
- [ ] Run:

```bash
python3 -m pytest tools/test_mlir_aie_bridge_topology.py
cargo test -p xdna-archspec
cargo test --lib
cargo fmt --all -- --check
git diff --check
```

- [ ] Confirm the worktree contains only Phase 0 source, tests, and
  documentation; ignored target directories remain uncommitted.
- [ ] Commit the evidence update last.

## Explicit Non-Actions

- Do not change coverage or `clean_release` semantics.
- Do not touch firmware, emulator timing, NPU hardware, KVM, vfio-user, Halo,
  or Maya's dirty main worktree.
- Do not change compiler pins or rebuild toolchains.
- Do not add a wrapper, path config file, crate, dependency, or activation
  requirement.
