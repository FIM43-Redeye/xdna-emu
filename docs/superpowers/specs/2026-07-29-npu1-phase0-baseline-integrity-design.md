# NPU1 Phase 0 Baseline Integrity -- Design

**Date:** 2026-07-29

**Status:** Implemented and validated

**Scope:** Provider-neutral, worktree-independent toolchain discovery and
fail-loud generated architecture inputs

## Purpose

Phase 0 makes the local build and validation environment trustworthy before
the NPU1 research-reserve program changes coverage or retirement semantics.
The same checkout must derive the same architecture facts whether it is built
from the main tree, a linked worktree, Claude Code, Codex, or an ordinary
shell.

This phase fixes the repository contract, not one shell session. Activation
scripts remain useful conveniences, but correctness must not depend on a
provider-specific `BASH_ENV`, inherited `PYTHONPATH`, or the checkout's depth
below `npu-work`.

## Current Diagnosis

The failed firmware-worktree build exposed path assumptions rather than a
broken compiler or broken mlir-aie Python installation:

- `xdna-archspec` falls back from its workspace root to `../mlir-aie`,
  `../llvm-aie`, and `../aie-rt/driver/src`;
- from
  `xdna-emu/.worktrees/firmware-priors`, those paths resolve beneath
  `.worktrees/` instead of `/home/triple/npu-work`;
- `.cargo/config.toml` similarly makes `TABLEGEN_210_PREFIX` relative to the
  checkout, so it points at the wrong tree in a linked worktree;
- the trace-event build step constructs the ironenv path from the same
  immediate-parent assumption and does not pass the already selected mlir-aie
  root to `mlir-aie-bridge.py`;
- the runtime `BridgePath` repeats that immediate-parent assumption, so the
  firmware worktree selects the ambient system Python instead of mlir-aie's
  ironenv and its trace-event tests fail because `numpy` is unavailable;
- the runtime register-database loader independently uses the project config's
  `../mlir-aie` value, causing required AM025 data to disappear from a linked
  worktree when activation inputs are removed;
- the bridge currently checks an ambient import before its explicit root, so
  an unrelated package on `PYTHONPATH` can override the requested toolchain;
  and
- aie-rt preprocessing failures silently generate empty stubs, converting a
  missing derivation source into an apparently successful build.

The canonical local components are intact. With their roots and Python
bindings supplied explicitly, the targeted semantic-coverage suite passed all
55 tests. A fresh full-library baseline then passed 4,269 tests, ignored 32,
and failed only the two runtime trace-bridge tests described above. Phase 0
makes the intended toolchain selection the ordinary build and runtime behavior
and leaves regression tests that fail under the old assumptions.

## Selected Approach

Add one small, shared `ToolchainPaths` resolver to `xdna-archspec`. `build.rs`
compiles the source directly, the library exposes the same implementation to
the runtime bridge and required register-database loader, and integration tests
exercise that implementation rather than a duplicate model of the search
rules.

The resolver owns only path selection and structural validation. Existing
generators continue owning parsing and code generation. No new configuration
file, dependency, wrapper command, background service, or activation mechanism
is introduced.

## Resolution Contract

Resolution is independent for each component. The first present source wins:

| Component | Resolution order |
|-----------|------------------|
| mlir-aie | `MLIR_AIE_PATH`, `MLIR_AIE_DIR`, `NPU_WORK_DIR/mlir-aie`, upward ancestor discovery |
| llvm-aie | `LLVM_AIE_PATH`, `LLVM_AIE_DIR`, `NPU_WORK_DIR/llvm-aie`, upward ancestor discovery |
| aie-rt | `AIE_RT_PATH`, `NPU_WORK_DIR/aie-rt/driver/src`, upward ancestor discovery |

`MLIR_AIE_DIR` and `LLVM_AIE_DIR` preserve the aliases already exported by the
shared NPU activation script. `AIE_RT_PATH` continues to mean the
`aie-rt/driver/src` directory, not the repository root.

Upward discovery walks from the xdna-emu workspace root through its ancestors
and checks the standard `npu-work` component layout at each level. Therefore
both:

```text
/home/triple/npu-work/xdna-emu
/home/triple/npu-work/xdna-emu/.worktrees/firmware-priors
```

discover components beneath `/home/triple/npu-work` without knowing either
absolute path in source.

An environment variable that is present is authoritative. A blank value,
missing directory, or missing sentinel is an error; resolution must not hide a
bad explicit configuration by falling through to another source. The same
fail-closed rule applies when `NPU_WORK_DIR` is present but does not contain a
required component.

Successful results are canonical absolute paths. A failure identifies:

- the component;
- the selected variable or discovery source;
- the candidate path;
- the missing or unusable sentinel; and
- for automatic discovery, the ancestor locations that were checked.

## Component Validation

The resolver and the immediate consumer together establish that each selected
component can provide the facts used by the build.

### mlir-aie

The selected root must contain:

```text
lib/Dialect/AIE/Util/aie_registers_aie2.json
```

It must also expose usable Python bindings from the selected tree. The build
prefers `<root>/ironenv/bin/python3` when it is executable; otherwise it uses
`python3` and lets the bridge insert the selected root's `build/python` or
`install/python` directory. Importing
`aie._mlir_libs._aie.get_target_model` is the functional check.

The runtime register-database loader uses this same selected root for AM025
rather than the project config's checkout-relative fixture path.

### llvm-aie

The selected root must contain:

```text
llvm/lib/Target/AIE/AIE2.td
build/bin/llvm-config
```

The `llvm-config` executable must be usable and belongs to the same resolved
llvm-aie tree. That single root then supplies the AIE TableGen sources and the
decoder FFI build; those consumers must not rediscover llvm-aie independently.

### aie-rt

The selected `driver/src` directory must contain:

```text
global/xaiemlgbl_reginit.c
```

Successful preprocessing must produce the required NPU1 DMA-module,
lock-module, and stream-port tables. Missing source, missing compiler,
preprocessor failure, parse failure, or an empty required table aborts the
build with the command context and underlying error. The generated empty-stub
fallback is removed.

## Cargo and TableGen Contract

The `tblgen` Rust dependency's host ABI and the Peano source tree solve
different problems:

- `tblgen-rs` uses LLVM 21 and, when no override is present, finds the system
  LLVM 21 `llvm-config` on `PATH`;
- an explicitly supplied `TABLEGEN_210_PREFIX` remains supported by that
  dependency; and
- the resolved llvm-aie tree supplies the target's AIE definitions and decoder
  libraries, regardless of the host `tblgen-rs` LLVM version.

The checkout-relative `TABLEGEN_210_PREFIX` entry is removed from
`.cargo/config.toml`. It is not replaced with another guessed path.

`build.rs` emits `cargo:rerun-if-env-changed` for every resolver input and
`cargo:rerun-if-changed` for the selected sentinel and generator inputs. A
changed toolchain selection or changed authoritative source therefore cannot
reuse generated output from the previous selection.

## Python Bridge Contract

Every build-time and runtime mlir-aie query is tied to the resolved root:

```text
<selected-python> tools/mlir-aie-bridge.py \
  --mlir-aie-path <resolved-mlir-aie-root> trace-events
```

When `--mlir-aie-path` is supplied, its `build/python` and `install/python`
locations are checked before any ambient import. A broken explicit root fails;
it never falls back to an unrelated package already visible through
`PYTHONPATH`.

Bridge output records the imported binding's physical location. The build
rejects output whose binding location is outside the selected mlir-aie root.
This turns “the import succeeded” into evidence that the intended toolchain
answered the query.

Ambient lookup remains available only to direct bridge users who do not pass
an explicit root. The xdna-archspec build and Rust `BridgePath` always pass
one.

The runtime bridge resolves only mlir-aie; it does not require llvm-aie or
aie-rt merely to answer an mlir-aie query. Missing optional bridge sources may
still make bridge discovery unavailable, but a present invalid explicit
mlir-aie path is reported as an error rather than silently treated as absence.

## Data Flow

```text
workspace root + process environment
  -> ToolchainPaths
     -> mlir-aie root
        -> AM025 register database
        -> explicit-root Python bridge
        -> trace-event tables
     -> llvm-aie root
        -> AIE TableGen definitions
        -> matching decoder FFI libraries
     -> aie-rt driver/src
        -> preprocessed DMA, lock, and stream-port tables
  -> generated xdna-archspec sources
  -> semantic coverage and emulator consumers

runtime BridgePath
  -> the same mlir-aie resolver
  -> selected interpreter + explicit mlir-aie root
  -> bridge query

runtime register database
  -> the same required mlir-aie resolver
  -> AM025 JSON
```

No downstream generator performs a second fallback search.

## Test-Driven Proof

Implementation begins with tests that reproduce the current failures.

### Resolver tests

The integration test compiles the exact build-support resolver against
temporary directory layouts and proves:

- canonical main-checkout discovery;
- nested `.worktrees/<name>` discovery;
- each explicit component override;
- activation-alias compatibility;
- per-component precedence;
- `NPU_WORK_DIR` resolution;
- an invalid explicit path fails without fallback;
- every missing sentinel is named;
- `AIE_RT_PATH` means `driver/src`; and
- every successful result is absolute.

A focused runtime test proves that a nested-worktree `BridgePath` selects the
resolved mlir-aie ironenv and passes the resolved root explicitly. It uses a
temporary layout and executable probe rather than depending on the developer's
installed packages. Companion tests pin optional automatic absence,
fail-closed explicit configuration, and the non-executable-ironenv fallback.
The required runtime register-database test no longer converts path-resolution
failure into a skip.

### Python tests

The existing bridge test module gains cases proving:

- an explicit mlir-aie root wins over a fake ambient `aie` package;
- a broken explicit root fails instead of importing the ambient package; and
- the reported binding path belongs to the selected root.

### Generator failure tests

Focused tests prove that aie-rt extraction rejects:

- a missing source tree;
- an unavailable preprocessor;
- non-zero preprocessing status;
- malformed output; and
- empty DMA, lock, or port tables.

No test accepts generated stubs as a valid absence mode.

## Acceptance Gate

The primary proof runs from the linked firmware worktree with all optional
activation inputs removed:

```bash
env -u NPU_WORK_DIR \
    -u MLIR_AIE_PATH -u MLIR_AIE_DIR \
    -u LLVM_AIE_PATH -u LLVM_AIE_DIR \
    -u AIE_RT_PATH -u PYTHONPATH \
    -u TABLEGEN_210_PREFIX \
    cargo test -p xdna-archspec --lib coverage::
```

The exact command is repeated once with a new ignored
`CARGO_TARGET_DIR` beneath `build/` so a cached build cannot satisfy the proof.
The implementation then passes:

```bash
python3 -m pytest tools/test_mlir_aie_bridge_topology.py
cargo test -p xdna-archspec
cargo test --lib
cargo fmt --all -- --check
git diff --check
```

The temporary-layout tests cover the canonical main-checkout shape without
building from or modifying Maya's dirty main worktree. Phase 0 needs no NPU
run: it verifies derivation inputs and build reproducibility, not hardware
behavior.

### Acceptance evidence

The witnessed RED states were:

- a new sanitized target failed because the checkout-relative
  `TABLEGEN_210_PREFIX` could not find a matching LLVM 21 `llvm-config`;
- the old bridge invocation was rejected by the executable probe as
  `bridge.py trace-events`, with no explicit mlir-aie root;
- the explicit-root Python tests allowed an ambient fake package to win;
- missing aie-rt input still produced a successful stub-backed build;
- the initial full-library baseline passed 4,269 tests, ignored 32, and failed
  the two live runtime bridge tests; and
- after closing the bridge edge, the fully sanitized library gate exposed the
  missed runtime AM025 consumer: 4,092 passed, 183 failed, 32 ignored, all
  cascading from `../mlir-aie/.../aie_registers_aie2.json`.

The GREEN evidence was:

- a genuinely new ignored target at `build/phase0-green-fresh-target` passed
  all 55 targeted coverage tests with every activation input removed;
- `cargo test -p xdna-archspec` passed 397 unit tests (2 ignored), 9 aie-rt
  support tests, 15 resolver tests, and 1 doc test;
- `pytest tools/test_mlir_aie_bridge_topology.py` passed all 4 tests;
- the sanitized `cargo test --lib` passed 4,275 tests, ignored 32, and failed
  none; and
- `cargo fmt --all -- --check` and `git diff --check` passed.

## Documentation Outcome

Repository guidance now states:

- the provider-neutral resolution order and override names;
- that activation is optional convenience;
- that `AIE_RT_PATH` points to `driver/src`;
- that system LLVM 21 is the default host dependency for `tblgen-rs`, distinct
  from the resolved llvm-aie/Peano tree; and
- that required aie-rt derivations fail loudly rather than degrading to stubs.

The shared activation script remains unchanged because its existing directory
aliases are accepted by the resolver.

## Alternatives Rejected

1. **A repository wrapper script.** It would make one entry point work while
   leaving direct Cargo, IDE, and test invocations dependent on hidden shell
   state.
2. **Repairing only Claude/Codex activation.** Provider hooks are useful
   conveniences, not a portable build contract.
3. **A new TOML toolchain-path file.** The standard sibling layout and existing
   environment overrides already express the needed policy; a second
   configuration source adds precedence and staleness problems.
4. **Keeping aie-rt stubs.** Empty generated tables violate the
   derive-from-the-toolchain rule and make later coverage results
   untrustworthy.

## Expected Implementation Surface

The implementation should remain within the existing seams:

- `.cargo/config.toml`;
- `crates/xdna-archspec/build.rs`;
- the shared resolver source and `crates/xdna-archspec/src/lib.rs`;
- the focused aie-rt build helper;
- resolver and build-support integration tests under
  `crates/xdna-archspec/tests/`;
- `src/integration/bridge.rs`, the required runtime register-database loader,
  and their direct callers;
- `tools/mlir-aie-bridge.py` and its existing Python test module; and
- the directly affected operations and TableGen documentation.

The implementation plan may narrow this list. It must not add a crate,
dependency, wrapper, or configuration format merely to satisfy Phase 0.

## Explicitly Out of Scope

- changing semantic coverage or `clean_release` behavior;
- designing the research ledger or capture catalogue;
- auditing the historical 5.2 GiB evidence corpus;
- changing compiler pins or rebuilding toolchain components;
- changing the shared activation script;
- emulator behavior, firmware behavior, or hardware characterization;
- NPU, KVM, or vfio-user runs; and
- remote Halo use.

Those belong to later research-reserve phases or independent toolchain work.
