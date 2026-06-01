# aiesim C++ Bridge + AiesimBackend Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (Parts 0 + I) or drive inline (Part II). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `AiesimBackend` — a second `NpuBackend` that drives AMD's
closed AIE2 cluster ISS in-process through a new C++ bridge library — selected at
runtime via `XDNA_BACKEND=aiesim`, with the interpreter path unchanged.

**Architecture:** A thin feature-gated Rust backend (`crates/xdna-emu-ffi/src/aiesim/`)
talks a small C ABI to a new out-of-tree C++ library `libxdna_aiesim_bridge.so`
(`aiesim-bridge/`, mirroring `xrt-plugin/`). The bridge embeds SystemC, constructs
the closed cluster (`create_math_engine`) once inside a long-lived service thread,
and provides the `ess_*()` weak symbols the open `-D__AIESIM__` HAL routes through.
The data path is **parser-driven**: our existing CDO parser's op-stream is replayed
as `ess_Write32`/blockwrite calls (HAL-driven independent replay is phase 2, out of
scope here).

**Tech Stack:** Rust (FFI crate, cargo `aiesim` feature, `libloading`/`dlopen`),
C++17 + SystemC 2.3.1 (Accellera, aietools), CMake (mirrors the plugin), the closed
`libaie2_cluster_msm_v1_0_0.osci.so`.

---

## Read first

- Spec: `docs/superpowers/specs/2026-06-01-aiesim-integrator-design.md` (the
  approved design; this plan implements it).
- Feasibility + embedding recipe (flags, host globals, key paths, factory
  signature): `docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md`.
- PS-bridge model (the `ess_*()` ⇄ TLM contract to twin):
  `../mlir-aie/aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp`.
- Real instantiation call site (the E513 fix derives from this):
  `<aietools>/data/systemc/simlibs/aie_xtlm/aie_xtlm_v1_0_0/src/aie_xtlm.cpp`
  (`create_cluster()` / `create_math_engine` call, ~lines 266–357).
- HAL SIM backend: `../aie-rt/driver/src/io_backend/ext/xaie_sim.c`.
- CMake aietools discovery module: `../mlir-aie/cmake/modulesXilinx/FindAIETools.cmake`.

---

## Key design decisions (review these before executing)

1. **`run()` seam = full unification (decided 2026-06-01).** `run` and
   `execute_npu_instructions` become first-class trait methods; the trait is the
   complete execution contract, and `execution.rs` collapses to a thin dispatcher
   with **no `as_interpreter()` downcast in the run path**. Two layering fixes fall
   out: (a) `npu_executor` — interpreter-specific runtime-sequence machinery that
   today sits misplaced on `XdnaEmuHandle` — moves into a new FFI-side
   `InterpreterBackend { engine, npu_executor }` wrapper that impls `NpuBackend`
   (core's pure ISA `InterpreterEngine` is untouched; `as_interpreter()` returns
   the inner engine); (b) the per-cycle async-callback firing
   (`fire_async_callbacks_for(handle)`, which took the whole handle — the borrow
   tangle Plan A dodged) becomes a clean `RunObserver` passed into `run()`. Each
   backend keeps its **own loop body** (interpreter per-cycle; aiesim batch
   `sc_start()`-to-quiescence) — we unify the *entry and state ownership*, not the
   loop mechanics. Unification is pure behavior-preserving relocation, gated by
   `cargo test --lib`; the interpreter's computation does not change. Sequenced
   (I.2→I.5) so the one fiddly step (the observer untangle) lands in isolation and
   no single task is a large atomic migration. The earlier dispatch design
   (`run_to_halt` returning `Option`) is **dropped**.

2. **`AiesimBackend` lives in the FFI crate, not `src/aiesim/` (core).** The spec
   §4 said "joins the existing `src/aiesim/`", but `NpuBackend` is private to
   `crates/xdna-emu-ffi`. A type implementing it must live there too. New home:
   `crates/xdna-emu-ffi/src/aiesim/`. The core crate's `src/aiesim/` (VCD harness)
   is unrelated and untouched. (Minor deviation from spec wording; same design.)

3. **The bridge ABI is behind a Rust trait (`BridgeAbi`).** `AiesimBackend` holds
   `Box<dyn BridgeAbi>`. The real impl `dlopen`s `libxdna_aiesim_bridge.so`; a
   `MockBridge` lets all of `AiesimBackend`'s marshalling be unit-tested in
   `cargo test --lib` with **zero aietools dependency**. This is how Part I stays
   sandbox-green.

4. **The real `ess_*` symbol set is a discovery step, not an assumption.** The
   proven seam in `genwrapper_for_ps.cpp` is six symbols: `ess_Write32`,
   `ess_Read32`, `ess_Write128`, `ess_Read128`, `ess_WriteGM`, `ess_ReadGM`. The
   spec also names `ess_WriteCmd` — **not present in that reference.** Part II's
   first bridge task enumerates exactly which `ess_*` weak symbols the
   `-D__AIESIM__` HAL actually references (`nm` on `libxaienginecdo`), and the
   bridge provides precisely that set. Do not hardcode `ess_WriteCmd` until
   confirmed.

5. **Part II cannot run in the Claude Code sandbox.** Building the bridge needs
   aietools libs on disk (present); *running* the cluster trips the same
   license-check / filesystem-isolation walls as a real-NPU capture. Part II tasks
   build and verify **out of sandbox**, on the dev box, the same way bridge tests
   do. There is a hard STOP-and-confirm checkpoint between Part I (sandbox-green)
   and Part II.

---

## File structure

**Part 0 — cleanups (modify):**
- `crates/xdna-emu-ffi/src/lib.rs` — `pub use` → `pub(crate)`; `select_backend`
  signature (lazy engine construction).

**Part I — Rust seam, full unification (create / modify):**
- Modify `crates/xdna-emu-ffi/Cargo.toml` — add `aiesim` feature + optional
  `libloading` dep.
- Modify `crates/xdna-emu-ffi/src/backend.rs` — `RunOutcome` / `HaltKind` (5
  variants) / `RunObserver`; `InterpreterBackend` wrapper; `run` +
  `execute_npu_instructions` trait methods; the relocated run loop +
  `build_*_signals` helpers.
- Modify `crates/xdna-emu-ffi/src/execution.rs` — `xdna_emu_run` becomes a thin
  dispatcher (build observer → `backend.run` → map outcome);
  `xdna_emu_execute_npu_instructions` routes to the backend.
- Modify `crates/xdna-emu-ffi/src/lib.rs` — drop `npu_executor` from
  `XdnaEmuHandle`; `select_backend` builds `InterpreterBackend`; feature-gated
  `aiesim` arm.
- Modify `crates/xdna-emu-ffi/src/async_errors.rs` — `fire_async_callbacks_for`
  → `CallbackObserver: RunObserver`; update the one direct-call test.
- Create `crates/xdna-emu-ffi/src/aiesim/mod.rs` — feature-gated module root.
- Create `crates/xdna-emu-ffi/src/aiesim/abi.rs` — C-ABI mirror + the `BridgeAbi`
  trait + `MockBridge` (test).
- Create `crates/xdna-emu-ffi/src/aiesim/bridge.rs` — real `dlopen` `BridgeAbi`.
- Create `crates/xdna-emu-ffi/src/aiesim/backend.rs` — `AiesimBackend: NpuBackend`.

**Part II — C++ bridge (create):**
- `aiesim-bridge/CMakeLists.txt`
- `aiesim-bridge/src/sc_bootstrap.cpp` — SystemC main + host globals.
- `aiesim-bridge/src/aiesim_top.{h,cpp}` — `sc_module`, owns `MathEngine*`.
- `aiesim-bridge/src/ps_bridge.{h,cpp}` — `ess_*()` ⇄ TLM, our `PSIP_ps_i3` twin.
- `aiesim-bridge/src/cdo_replay.{h,cpp}` — parser op-stream → `ess_*()`.
- `aiesim-bridge/src/service_thread.{h,cpp}` — elaborate-once thread + command queue.
- `aiesim-bridge/src/c_abi.cpp` — the `extern "C"` surface.
- `aiesim-bridge/include/xdna_aiesim_bridge.h` — the C ABI header (Rust mirrors it).
- `scripts/build-aiesim-bridge.sh` — driver, mirrors `rebuild-plugin.sh`.

---

# Part 0 — Plan-A cleanups

Wrap these in first (Maya's ask). Both are pure refactors; the suite stays green.

### Task 0.1: Tighten `NpuBackend` visibility to `pub(crate)`

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs:50`

- [ ] **Step 1: Confirm no external consumer**

Run: `grep -rn "NpuBackend" crates/xdna-emu-ffi/tests tests 2>/dev/null; echo "exit:$?"`
Expected: no hits in `tests/` (the trait is used only inside the crate). If a
test references it, stop and keep `pub use` — otherwise proceed.

- [ ] **Step 2: Narrow the re-export**

In `crates/xdna-emu-ffi/src/lib.rs`, change line 50 from:
```rust
pub use backend::NpuBackend;
```
to:
```rust
pub(crate) use backend::NpuBackend;
```

- [ ] **Step 3: Build + test**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Expected: compiles, all FFI tests pass (no `unused import` / `private in public`).

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi: scope NpuBackend re-export to pub(crate)

The trait is an internal dispatch seam; nothing outside the FFI crate
consumes it. Tightening the re-export keeps the public surface minimal.

Generated using Claude Code."
```

### Task 0.2: Make `select_backend` construct the engine lazily

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs:190-201` (the `select_backend` fn)
- Modify: `crates/xdna-emu-ffi/src/lib.rs:228-240` (call site in `xdna_emu_create`)
- Modify: `crates/xdna-emu-ffi/src/lib.rs:1033-1051` (the three selector tests)

**Why:** today `xdna_emu_create` builds an `InterpreterEngine` and passes it to
`select_backend`, which then *discards* it on the `aiesim`/unknown error paths.
Take a factory closure so only the interpreter arm constructs an engine. Plan B's
`aiesim` arm builds an `AiesimBackend` instead and never touches the closure.

- [ ] **Step 1: Change the signature to take a factory**

Replace the `select_backend` fn (lib.rs ~186-201) with:
```rust
/// Choose a backend from the `XDNA_BACKEND` value. Pure (no env read) so it is
/// directly testable. The interpreter engine is built lazily via `make_interp`
/// so error paths (and, in Plan B, the aiesim arm) never construct one need-
/// lessly. An unsupported request fails loudly rather than silently falling back.
pub(crate) fn select_backend(
    kind: &str,
    make_interp: impl FnOnce() -> InterpreterEngine,
) -> Result<Box<dyn crate::backend::NpuBackend>, String> {
    match kind {
        "interpreter" => Ok(Box::new(make_interp())),
        "aiesim" => {
            Err("XDNA_BACKEND=aiesim: this build has no aiesim backend (Plan B not yet built)".to_string())
        }
        other => Err(format!("XDNA_BACKEND={other}: unknown backend")),
    }
}
```

- [ ] **Step 2: Update the call site**

In `xdna_emu_create` (lib.rs ~228-240), replace the eager engine build + call:
```rust
    let config = xdna_emu_core::config::Config::get();
    let mut engine = InterpreterEngine::new_npu1();
    engine.set_stall_threshold(config.stall_threshold());

    let backend_kind = std::env::var("XDNA_BACKEND").unwrap_or_else(|_| "interpreter".to_string());
    let backend = match select_backend(&backend_kind, engine) {
```
with:
```rust
    let config = xdna_emu_core::config::Config::get();
    let make_interp = || {
        let mut engine = InterpreterEngine::new_npu1();
        engine.set_stall_threshold(config.stall_threshold());
        engine
    };

    let backend_kind = std::env::var("XDNA_BACKEND").unwrap_or_else(|_| "interpreter".to_string());
    let backend = match select_backend(&backend_kind, make_interp) {
```

- [ ] **Step 3: Update the three selector tests**

The tests at lib.rs ~1033-1051 pass an `eng` value; change each to pass a closure.
For `select_backend_interpreter_ok`:
```rust
    fn select_backend_interpreter_ok() {
        assert!(select_backend("interpreter", || InterpreterEngine::new_npu1()).is_ok());
    }
```
For `select_backend_aiesim_unsupported` (keep the assertion on the error text):
```rust
    fn select_backend_aiesim_unsupported() {
        let r = select_backend("aiesim", || InterpreterEngine::new_npu1());
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("aiesim"));
    }
```
For `select_backend_unknown_rejected`:
```rust
    fn select_backend_unknown_rejected() {
        assert!(select_backend("bogus", || InterpreterEngine::new_npu1()).is_err());
    }
```
(Match the existing test bodies; only the second argument changes form. If a test
constructs the engine differently, preserve that inside the closure.)

- [ ] **Step 4: Build + test**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Expected: compiles, selector tests pass, no behavior change.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi: build interpreter engine lazily in select_backend

Take a factory closure so the aiesim/unknown error paths no longer
construct and discard an InterpreterEngine. Prepares the aiesim arm,
which builds its own backend and never invokes the closure.

Generated using Claude Code."
```

---

# Part I — Rust seam (sandbox-green, feature-gated)

All of Part I builds and tests inside `cargo test --lib` with **no aietools**. The
`aiesim` feature gates the new code; the `MockBridge` stands in for the C++ library.
**Gate at the end of Part I: `cargo test --lib` green both with and without
`--features aiesim`.**

### Task I.1: Add the `aiesim` cargo feature

**Files:**
- Modify: `crates/xdna-emu-ffi/Cargo.toml`

- [ ] **Step 1: Add the feature and optional dep**

In `crates/xdna-emu-ffi/Cargo.toml`, after the `[dependencies]` block add:
```toml
[features]
# Off by default. Gates the AiesimBackend Rust code. With the feature off,
# XDNA_BACKEND=aiesim returns a clean "built without aiesim support" error.
# Even with the feature ON there is no build-time aietools dependency: the
# bridge .so is loaded at runtime via dlopen (see src/aiesim/bridge.rs).
aiesim = ["dep:libloading"]
```
And in `[dependencies]` add:
```toml
libloading = { version = "0.8", optional = true }
```

- [ ] **Step 2: Verify both feature states still build**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi`
Expected: builds (feature off; no `libloading`).
Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi --features aiesim`
Expected: builds (pulls `libloading`; nothing references it yet — allow the
`unused crate` warning for now, resolved in Task I.9 where the dlopen bridge uses it).

- [ ] **Step 3: Commit**

```bash
git add crates/xdna-emu-ffi/Cargo.toml
git commit -m "ffi: add off-by-default aiesim cargo feature

Gates the AiesimBackend code; pulls libloading only when enabled. No
build-time aietools dependency in either state -- the bridge is dlopened
at runtime.

Generated using Claude Code."
```

### Task I.2: Add `RunOutcome` / `HaltKind` / `RunObserver` to `backend.rs`

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs`

Purely additive types — no behavior change. `HaltKind` has **five** variants
mirroring `XdnaEmuHaltReason` (the interpreter loop produces all five).
`RunObserver` is the seam that decouples async-callback firing from the loop.

- [ ] **Step 1: Add the types**

Near the top of `backend.rs` (after the imports, before the trait), add. Note the
new import:
```rust
use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
```
```rust
/// How a `run()` ended. Mirrors `XdnaEmuHaltReason`; `execution.rs::map_halt`
/// translates it to the FFI exec-status triple. Kept FFI-struct-free so
/// backend.rs has no dependency on the C ABI types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaltKind {
    /// Kernel ran to natural completion (cores halted, syncs satisfied).
    Completed,
    /// Cycle budget reached before natural completion.
    Budget,
    /// A MaskPoll could not be satisfied (engine quiescent, condition unmet).
    MaskPollUnsatisfied,
    /// The in-flight submission wedged; the context was marked Failed.
    WedgeRecovered,
    /// Error during execution (FFI fault, executor error, bad precondition).
    Error,
}

/// Result of a `run()`.
#[derive(Debug, Clone, Copy)]
pub struct RunOutcome {
    pub cycles: u64,
    pub halt: HaltKind,
}

/// Observer the FFI passes into `run()`. The backend reports newly-recorded
/// async errors at whatever granularity it supports (interpreter: every cycle;
/// aiesim: per run / at sync points). This replaces `fire_async_callbacks_for`,
/// which took the whole handle -- the borrow tangle Plan A dodged by deferring
/// `run()`. The FFI's `CallbackObserver` (async_errors.rs) fires the registered
/// C callback for each record.
pub trait RunObserver {
    fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]);
}
```

- [ ] **Step 2: Add a use-site test**

In the `#[cfg(test)] mod tests`, add a trivial observer + outcome construction so
the types are exercised:
```rust
#[test]
fn run_outcome_and_observer_compose() {
    use super::{HaltKind, RunObserver, RunOutcome};
    use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;

    struct Counting(u32);
    impl RunObserver for Counting {
        fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]) {
            self.0 += records.len() as u32;
        }
    }
    let mut o = Counting(0);
    o.on_async_errors(&[]);
    assert_eq!(o.0, 0);
    let out = RunOutcome { cycles: 7, halt: HaltKind::Completed };
    assert_eq!(out.cycles, 7);
}
```

- [ ] **Step 3: Build + test**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib run_outcome_and_observer_compose`
Expected: PASS. (If `AmdxdnaAsyncError`'s path differs, grep
`grep -rn "pub struct AmdxdnaAsyncError" ..` and fix the import.)

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs
git commit -m "ffi: add RunOutcome/HaltKind/RunObserver execution types

Five-variant HaltKind mirrors XdnaEmuHaltReason; RunObserver is the seam
that will decouple async-callback firing from the run loop. Additive, no
behavior change.

Generated using Claude Code."
```

### Task I.3: Introduce the `InterpreterBackend` wrapper

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (`select_backend` builds the wrapper)

Move `impl NpuBackend` off the bare `InterpreterEngine` onto a wrapper
`InterpreterBackend` that (next task) will also own the runtime-sequence executor.
The wrapper delegates every existing trait method to its inner engine, and
`as_interpreter()` returns that engine — so all downcast-hatch consumers (query,
async-error reads, context, firmware, the run loop) are unchanged. This task adds
no executor yet; `XdnaEmuHandle.npu_executor` stays put. Fully green, mechanical.

- [ ] **Step 1: Replace the `impl NpuBackend for InterpreterEngine` block**

In `backend.rs`, delete `impl NpuBackend for InterpreterEngine { ... }` and add:
```rust
/// The interpreter backend: the pure-ISA `InterpreterEngine` plus (Task I.5) its
/// runtime-sequence executor. The engine itself stays a clean core type; this
/// FFI-side wrapper is where host/firmware-level driving lives.
pub(crate) struct InterpreterBackend {
    pub(crate) engine: InterpreterEngine,
}

impl InterpreterBackend {
    pub(crate) fn new(engine: InterpreterEngine) -> Self {
        Self { engine }
    }
}

impl NpuBackend for InterpreterBackend {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        self.engine.device_mut().apply_cdo(cdo).map_err(|e| e.to_string())
    }
    fn set_start_col(&mut self, start_col: u8) {
        self.engine.device_mut().set_start_col(start_col);
    }
    fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String> {
        self.engine.load_elf_bytes(col, row, data)
    }
    fn host_memory_mut(&mut self) -> &mut HostMemory {
        self.engine.host_memory_mut()
    }
    fn sync_cores_from_device(&mut self) {
        self.engine.sync_cores_from_device();
    }
    fn reset_for_new_context(&mut self) {
        self.engine.reset_for_new_context();
    }
    fn reset_context(&mut self, cid: ContextId) -> Result<(), ()> {
        self.engine.device_mut().reset_context(cid).map_err(|_| ())
    }
    fn cols(&self) -> usize {
        self.engine.device().cols()
    }
    fn rows(&self) -> usize {
        self.engine.device().rows()
    }
    fn arch_name(&self) -> String {
        self.engine.device().arch_name().to_string()
    }
    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        Some(&self.engine)
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        Some(&mut self.engine)
    }
}
```

- [ ] **Step 2: Update the Plan A `backend.rs` tests**

The `interpreter_implements_backend_and_downcasts` test uses
`InterpreterEngine::new_npu1()` directly as a `&mut dyn NpuBackend`. Wrap it:
```rust
    let mut be = super::InterpreterBackend::new(InterpreterEngine::new_npu1());
    let b: &mut dyn NpuBackend = &mut be;
```
(Keep the rest of the assertions — `cols()`/`rows()`/`arch_name()`/`as_interpreter`
all still hold through the wrapper.)

- [ ] **Step 3: Build the interpreter backend in `select_backend`**

In `lib.rs`, change the interpreter arm to wrap the engine:
```rust
        "interpreter" => Ok(Box::new(crate::backend::InterpreterBackend::new(make_interp()))),
```
(The `make_interp` closure from Task 0.2 is unchanged.)

- [ ] **Step 4: Build + full FFI suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Expected: green. `execution.rs` and the async-error/query/context/firmware files
are untouched — they reach the engine through `as_interpreter()`, which now lands
on `self.engine` transparently.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi: wrap the interpreter in InterpreterBackend

Move impl NpuBackend off bare InterpreterEngine onto a wrapper that
as_interpreter() unwraps, so every downcast-hatch consumer is unchanged.
Sets up relocating npu_executor into the backend (next task).

Generated using Claude Code."
```

### Task I.4: Extract the run loop into `run_interpreter` + the observer untangle

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs` (new free fn + the `build_*_signals`)
- Modify: `crates/xdna-emu-ffi/src/execution.rs` (call the free fn; add `map_halt`)
- Modify: `crates/xdna-emu-ffi/src/async_errors.rs` (`CallbackObserver`; drop
  `fire_async_callbacks_for`; fix the one direct-call test)

The fiddly step, in isolation. Lift the entire `xdna_emu_run` loop body into a free
function taking `(engine, executor, max_cycles, observer)`, replacing
`fire_async_callbacks_for(handle)` with `observer.on_async_errors(...)`. The
*current* `xdna_emu_run` keeps using `handle.npu_executor` (still a handle field)
and calls the new function. Behavior is byte-for-byte identical (same loop, same
per-cycle firing). The executor doesn't move until Task I.5.

- [ ] **Step 1: Add `CallbackObserver` in `async_errors.rs`, delete `fire_async_callbacks_for`**

Replace `fire_async_callbacks_for` with:
```rust
/// Observer that fires the registered C callback for each newly-recorded async
/// error. Built fresh per `xdna_emu_run` from the handle's callback (copied out,
/// so it holds no handle borrow). Replaces `fire_async_callbacks_for`.
pub(crate) struct CallbackObserver {
    pub(crate) cb: Option<crate::AsyncErrorCallback>,
}

impl crate::backend::RunObserver for CallbackObserver {
    fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]) {
        let Some(cb) = self.cb else { return };
        for rec in records {
            let xrec = XdnaEmuAsyncError::from(rec);
            // SAFETY: cb.func is a valid C fn ptr registered via
            // xdna_emu_set_async_event_callback; fired on the run thread per
            // the handle-serialization contract.
            unsafe { (cb.func)(&xrec as *const _, cb.user_data) };
        }
    }
}
```
(`AmdxdnaAsyncError` is already imported in async_errors.rs.)

- [ ] **Step 2: Fix the direct-call test**

`set_async_event_callback_registers_and_fires_on_drain` calls
`fire_async_callbacks_for(&mut *h)`. Replace that line with an observer drive:
```rust
                // Drain + fire through the observer (what run_interpreter does).
                let recs = (*h)
                    .backend
                    .as_interpreter_mut()
                    .expect("interpreter backend")
                    .device_mut()
                    .async_errors
                    .drain_newly_recorded();
                let mut obs = CallbackObserver { cb: (*h).async_callback };
                use crate::backend::RunObserver;
                obs.on_async_errors(&recs);
```
Do the same for `set_async_event_callback_with_none_unregisters`.

- [ ] **Step 3: Move `build_engine_signals` / `build_executor_signals` to `backend.rs`**

Cut both functions from `execution.rs` and paste them into `backend.rs` (make them
`pub(crate)` if needed by the free fn; they already take `&InterpreterEngine` /
`&mut NpuExecutor`). Add `use xdna_emu_core::npu::NpuExecutor;` to backend.rs.

- [ ] **Step 4: Add `run_interpreter` in `backend.rs`**

Move the whole loop body of `xdna_emu_run` (from the entry-guard through the final
`XdnaEmuExecStatus` computation) into:
```rust
/// The interpreter's run loop. Lifted verbatim from `xdna_emu_run`; the only
/// changes are: takes engine+executor+observer instead of the handle, fires
/// async errors via `observer`, and returns `RunOutcome` instead of the FFI
/// status struct. Behavior is unchanged.
pub(crate) fn run_interpreter(
    engine: &mut InterpreterEngine,
    executor: &mut NpuExecutor,
    max_cycles: u64,
    observer: &mut dyn RunObserver,
) -> RunOutcome {
    // ... entire current loop body, with these mechanical substitutions:
    //   handle.backend.as_interpreter().expect(...)      -> engine
    //   handle.backend.as_interpreter_mut().expect(...)  -> engine
    //   handle.npu_executor                              -> executor
    //   fire_async_callbacks_for(handle);                ->
    //       let recs = engine.device_mut().async_errors.drain_newly_recorded();
    //       observer.on_async_errors(&recs);
    //   each early `return XdnaEmuExecStatus { .. Error .. }` -> `return RunOutcome { cycles, halt: HaltKind::Error }`
    //   the wedge return     -> RunOutcome { cycles, halt: HaltKind::WedgeRecovered }
    //   the final return     -> RunOutcome { cycles, halt: <Completed|Budget|MaskPollUnsatisfied> }
}
```
Map the final halt determination (the existing `maskpoll_unsatisfied` /
`natural_halt` logic) to `HaltKind`:
- `maskpoll_unsatisfied` ⇒ `MaskPollUnsatisfied`
- else `natural_halt` ⇒ `Completed`
- else ⇒ `Budget`
The wedge branch (which calls `mark_failed` + returns) ⇒ `WedgeRecovered`. The
context `note_submission_complete()` on clean completion stays inside this fn.

- [ ] **Step 5: Rewrite `xdna_emu_run` to call it (executor still on the handle)**

Replace the body after the null check with:
```rust
    let handle = &mut *handle;
    let max = handle.max_cycles;
    let mut observer = crate::async_errors::CallbackObserver { cb: handle.async_callback };

    // Disjoint field borrows: backend (engine) and npu_executor.
    let Some(engine) = handle.backend.as_interpreter_mut() else {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::ExecutionError,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    };
    let outcome = crate::backend::run_interpreter(engine, &mut handle.npu_executor, max, &mut observer);
    let (result, halted, halt_reason) = map_halt(outcome.halt);
    XdnaEmuExecStatus { result, cycles_executed: outcome.cycles, halted, halt_reason }
```
And add `map_halt` (module scope, `execution.rs`):
```rust
fn map_halt(halt: crate::backend::HaltKind) -> (XdnaEmuResult, bool, XdnaEmuHaltReason) {
    use crate::backend::HaltKind;
    match halt {
        HaltKind::Completed => (XdnaEmuResult::Success, true, XdnaEmuHaltReason::Completed),
        // Budget = cycle cap hit before quiescence: run NOT done -> halted=false
        // (matches the original `maskpoll_unsatisfied || natural_halt` truth table).
        HaltKind::Budget => (XdnaEmuResult::Success, false, XdnaEmuHaltReason::Budget),
        HaltKind::MaskPollUnsatisfied => (XdnaEmuResult::Success, true, XdnaEmuHaltReason::MaskPollUnsatisfied),
        HaltKind::WedgeRecovered => (XdnaEmuResult::Success, true, XdnaEmuHaltReason::WedgeRecovered),
        HaltKind::Error => (XdnaEmuResult::ExecutionError, false, XdnaEmuHaltReason::Error),
    }
}
```

- [ ] **Step 6: Build + full lib suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: same green count as before. The loop relocated but is byte-for-byte the
same; async callbacks fire per-cycle exactly as before (now via the observer).
Watch for borrow errors at the `run_interpreter` call — `engine` and
`&mut handle.npu_executor` are disjoint fields, so it compiles; if the checker
complains, bind `let exec = &mut handle.npu_executor;` *before* taking `engine`.

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs crates/xdna-emu-ffi/src/execution.rs crates/xdna-emu-ffi/src/async_errors.rs
git commit -m "ffi: extract run_interpreter free fn + RunObserver untangle

Lift the run loop into run_interpreter(engine, executor, observer) and
replace fire_async_callbacks_for (which took the whole handle) with a
CallbackObserver. Behavior byte-for-byte identical; xdna_emu_run now
calls the free fn. Isolates the borrow untangle before promoting run()
to the trait.

Generated using Claude Code."
```

### Task I.5: Relocate `npu_executor` + promote `run` / `execute_npu_instructions` to the trait

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs` (executor field; trait methods)
- Modify: `crates/xdna-emu-ffi/src/execution.rs` (thin dispatcher; route exec-npu)
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (drop `npu_executor` from the handle)

The atomic flip — now small, because `run_interpreter` already exists. The executor
moves from the handle into `InterpreterBackend`; `run` and `execute_npu_instructions`
become trait methods; `execution.rs` collapses to a dispatcher with **no downcast
in the run path**.

- [ ] **Step 1: Grep the executor constructor**

Run: `grep -n "npu_executor" crates/xdna-emu-ffi/src/lib.rs; grep -rn "impl NpuExecutor\|fn new\|Default for NpuExecutor" $(find .. -path '*npu*' -name '*.rs' | head)`
Note how `xdna_emu_create` builds `npu_executor` (e.g. `NpuExecutor::new()` /
`::default()`); use the same in `InterpreterBackend::new`.

- [ ] **Step 2: Add the executor field + trait methods**

In `backend.rs`, add the field and the two trait methods. First extend the struct
+ constructor:
```rust
pub(crate) struct InterpreterBackend {
    pub(crate) engine: InterpreterEngine,
    pub(crate) npu_executor: NpuExecutor,
}
impl InterpreterBackend {
    pub(crate) fn new(engine: InterpreterEngine) -> Self {
        Self { engine, npu_executor: NpuExecutor::new() } // match Step 1
    }
}
```
Add to `pub trait NpuBackend` (after `reset_context`):
```rust
    // --- execution (the unified seam) ---
    /// Load the runtime-sequence (NPU instruction) stream for this submission.
    /// Interpreter: feed its executor. aiesim: encode + buffer for register-write
    /// replay (Part II).
    fn execute_npu_instructions(
        &mut self,
        stream: &xdna_emu_core::npu::NpuInstructionStream,
    ) -> Result<(), String>;
    /// Run the configured submission to quiescence or `max_cycles` (0 =
    /// unbounded). Reports async errors via `observer`.
    fn run(&mut self, max_cycles: u64, observer: &mut dyn RunObserver) -> RunOutcome;
```
Implement on `InterpreterBackend`:
```rust
    fn execute_npu_instructions(
        &mut self,
        stream: &xdna_emu_core::npu::NpuInstructionStream,
    ) -> Result<(), String> {
        self.npu_executor.load(stream);
        Ok(())
    }
    fn run(&mut self, max_cycles: u64, observer: &mut dyn RunObserver) -> RunOutcome {
        run_interpreter(&mut self.engine, &mut self.npu_executor, max_cycles, observer)
    }
```
And on `MockBackend` (test module):
```rust
    fn execute_npu_instructions(
        &mut self,
        _stream: &xdna_emu_core::npu::NpuInstructionStream,
    ) -> Result<(), String> {
        Ok(())
    }
    fn run(&mut self, _max_cycles: u64, _observer: &mut dyn RunObserver) -> RunOutcome {
        RunOutcome { cycles: 0, halt: HaltKind::Completed }
    }
```

- [ ] **Step 3: Make `xdna_emu_run` a thin dispatcher**

Replace the Task I.4 body of `xdna_emu_run` (after the null check) with:
```rust
    let handle = &mut *handle;
    let max = handle.max_cycles;
    let mut observer = crate::async_errors::CallbackObserver { cb: handle.async_callback };
    let outcome = handle.backend.run(max, &mut observer);
    let (result, halted, halt_reason) = map_halt(outcome.halt);
    XdnaEmuExecStatus { result, cycles_executed: outcome.cycles, halted, halt_reason }
```
No `as_interpreter` in the run path. `run_interpreter` stays a `pub(crate)` free fn
in backend.rs, now called only by `InterpreterBackend::run`.

- [ ] **Step 4: Route `execute_npu_instructions` through the backend**

In `xdna_emu_execute_npu_instructions`, replace `handle.npu_executor.load(&stream)`
with:
```rust
    if let Err(e) = handle.backend.execute_npu_instructions(&stream) {
        log::error!("execute_npu_instructions: {}", e);
        set_last_error(e);
        return XdnaEmuResult::ExecutionError;
    }
    XdnaEmuResult::Success
```
(Drop the now-unused `XdnaEmuResult::Success` tail if it duplicates.)

- [ ] **Step 5: Drop `npu_executor` from the handle**

In `lib.rs`: remove the `pub(crate) npu_executor: NpuExecutor,` field from
`XdnaEmuHandle`, remove its initializer in `xdna_emu_create`, and remove
`use xdna_emu_core::npu::NpuExecutor;` if now unused in lib.rs (it lives in
backend.rs now). Grep for any other `handle.npu_executor` / `.npu_executor`
references and route them through the backend or delete.

- [ ] **Step 6: Build + full lib suite + rebuild cdylib**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: same green count. Interpreter behavior identical; the trait is now the
complete execution contract.
Run: `cargo build -p xdna-emu-ffi`
Expected: fresh `.so`.

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-emu-ffi/src
git commit -m "ffi: unify run() + execute_npu_instructions onto NpuBackend

npu_executor moves from XdnaEmuHandle into InterpreterBackend; run and
execute_npu_instructions become trait methods. xdna_emu_run is now a thin
dispatcher with no downcast in the run path -- the trait is the complete
execution contract. Interpreter behavior unchanged.

Generated using Claude Code."
```

### Task I.6: The C-ABI surface + `BridgeAbi` trait + `MockBridge`

**Files:**
- Create: `crates/xdna-emu-ffi/src/aiesim/mod.rs`
- Create: `crates/xdna-emu-ffi/src/aiesim/abi.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (declare the gated module)

This task defines the Rust mirror of the C ABI and a backend-facing trait, with a
fully in-memory `MockBridge` so the next task's `AiesimBackend` is unit-testable.

- [ ] **Step 1: Declare the gated module**

In `crates/xdna-emu-ffi/src/lib.rs`, with the other `mod` lines (~39-47), add:
```rust
#[cfg(feature = "aiesim")]
mod aiesim;
```

- [ ] **Step 2: Write `mod.rs`**

Create `crates/xdna-emu-ffi/src/aiesim/mod.rs`:
```rust
//! aiesim backend: a `NpuBackend` that drives the closed AIE2 cluster ISS
//! in-process through the C++ bridge library `libxdna_aiesim_bridge.so`.
//!
//! Layering:
//! - `abi`     -- the C-ABI surface + the `BridgeAbi` trait + `MockBridge`.
//! - `bridge`  -- the real `dlopen`-backed `BridgeAbi` impl.
//! - `backend` -- `AiesimBackend`, which implements `NpuBackend` over a
//!                `Box<dyn BridgeAbi>`.
//!
//! Feature-gated behind `aiesim`; nothing here compiles in a default build.

pub(crate) mod abi;
pub(crate) mod backend;
pub(crate) mod bridge;
```

- [ ] **Step 3: Write the failing `MockBridge` test (drives the API shape)**

Create `crates/xdna-emu-ffi/src/aiesim/abi.rs` with the trait + mock + a test:
```rust
//! The bridge ABI: a Rust trait the backend calls, plus the raw C-ABI extern
//! declarations the real impl binds to. Keeping the backend behind a trait lets
//! us unit-test all marshalling against an in-memory `MockBridge` -- no aietools.

/// Result codes the bridge C ABI returns (0 = ok).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum BridgeStatus {
    Ok = 0,
    Error = 1,
}

/// How a bridge `run` ended (mirrors the C ABI's enum; maps to HaltKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum BridgeHalt {
    Completed = 0,
    Budget = 1,
    Error = 2,
}

/// The operations `AiesimBackend` needs from the bridge. One method per C-ABI
/// entry. Slices/lengths are marshalled to raw pointers only in the real impl.
pub(crate) trait BridgeAbi {
    /// Construct the cluster for `arch` ("aie2"/"aie2ps"/"aie") using the device
    /// JSON at `device_json`. Called once; the bridge owns the service thread.
    fn create(&mut self, arch: &str, device_json: &str) -> BridgeStatus;
    /// Replay a config op-stream (already serialized by cdo_replay-side encoding;
    /// here it is the raw bytes our parser produced). Returns Ok/Error.
    fn load_cdo(&mut self, ops: &[u8]) -> BridgeStatus;
    /// Replay a runtime-sequence (NPU instruction) op-stream as register writes.
    /// Same tagged wire format as `load_cdo`; separate entry so the bridge can
    /// stage it distinctly (it arrives via `execute_npu_instructions`, before run).
    fn exec_npu(&mut self, ops: &[u8]) -> BridgeStatus;
    /// Write host (DDR/GM) memory.
    fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus;
    /// Read host (DDR/GM) memory into `out`.
    fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus;
    /// Run to quiescence or `budget` cycles. On Ok, `*cycles_out` is set.
    fn run(&mut self, budget: u64, cycles_out: &mut u64) -> BridgeHalt;
    /// Tier-2: zero-time backdoor register read.
    fn read_reg(&mut self, addr: u64) -> u32;
    /// Reset logical state between submissions (re-apply CDO follows).
    fn reset(&mut self) -> BridgeStatus;
}

#[cfg(test)]
pub(crate) mod mock {
    use super::*;
    use std::collections::HashMap;

    /// In-memory bridge: records calls + models GM as a sparse map so the
    /// backend's write_gm/read_gm round-trips are testable without a cluster.
    #[derive(Default)]
    pub(crate) struct MockBridge {
        pub created: Option<(String, String)>,
        pub cdo_loads: u32,
        pub npu_loads: u32,
        pub runs: u32,
        pub gm: HashMap<u64, u8>,
        pub next_run_halt: Option<BridgeHalt>,
        pub next_run_cycles: u64,
    }

    impl BridgeAbi for MockBridge {
        fn create(&mut self, arch: &str, device_json: &str) -> BridgeStatus {
            self.created = Some((arch.to_string(), device_json.to_string()));
            BridgeStatus::Ok
        }
        fn load_cdo(&mut self, _ops: &[u8]) -> BridgeStatus {
            self.cdo_loads += 1;
            BridgeStatus::Ok
        }
        fn exec_npu(&mut self, _ops: &[u8]) -> BridgeStatus {
            self.npu_loads += 1;
            BridgeStatus::Ok
        }
        fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus {
            for (i, b) in data.iter().enumerate() {
                self.gm.insert(addr + i as u64, *b);
            }
            BridgeStatus::Ok
        }
        fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = *self.gm.get(&(addr + i as u64)).unwrap_or(&0);
            }
            BridgeStatus::Ok
        }
        fn run(&mut self, _budget: u64, cycles_out: &mut u64) -> BridgeHalt {
            self.runs += 1;
            *cycles_out = self.next_run_cycles;
            self.next_run_halt.unwrap_or(BridgeHalt::Completed)
        }
        fn read_reg(&mut self, _addr: u64) -> u32 {
            0
        }
        fn reset(&mut self) -> BridgeStatus {
            BridgeStatus::Ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockBridge;
    use super::*;

    #[test]
    fn mock_bridge_round_trips_gm() {
        let mut b = MockBridge::default();
        assert_eq!(b.create("aie2", "/dev/null"), BridgeStatus::Ok);
        assert_eq!(b.write_gm(0x1000, &[1, 2, 3, 4]), BridgeStatus::Ok);
        let mut out = [0u8; 4];
        assert_eq!(b.read_gm(0x1000, &mut out), BridgeStatus::Ok);
        assert_eq!(out, [1, 2, 3, 4]);
    }
}
```

- [ ] **Step 4: Run the test (feature on)**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --features aiesim --lib mock_bridge_round_trips_gm`
Expected: PASS.

- [ ] **Step 5: Confirm the default build is untouched**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Expected: PASS, and the `aiesim` module is absent (not compiled).

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-emu-ffi/src/aiesim/mod.rs crates/xdna-emu-ffi/src/aiesim/abi.rs crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi(aiesim): BridgeAbi trait + MockBridge + C-ABI mirror types

The backend talks to the C++ bridge through a Rust trait so all
marshalling is unit-testable against an in-memory mock with no aietools.
Feature-gated; default build unaffected.

Generated using Claude Code."
```

### Task I.7: `AiesimBackend` over `BridgeAbi`

**Files:**
- Create: `crates/xdna-emu-ffi/src/aiesim/backend.rs`

`AiesimBackend` implements `NpuBackend` by translating each trait call to a
`BridgeAbi` call — including the unified `run` and `execute_npu_instructions`. It
holds the parser→op-stream encoding for `apply_cdo` / `execute_npu_instructions`
and a host write/read buffer view. Tier-3 interpreter hatches stay `None` (an
aiesim backend is not an interpreter).

- [ ] **Step 1: Write the failing dispatch test**

Create `crates/xdna-emu-ffi/src/aiesim/backend.rs`:
```rust
//! `AiesimBackend` -- `NpuBackend` backed by the C++ bridge (via `BridgeAbi`).

use crate::aiesim::abi::{BridgeAbi, BridgeHalt, BridgeStatus};
use crate::backend::{HaltKind, NpuBackend, RunObserver, RunOutcome};
use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::npu::NpuInstructionStream;
use xdna_emu_core::parser::Cdo;

/// Cycle budget passed to the bridge when `max_cycles == 0` (unbounded request).
const DEFAULT_CYCLE_BUDGET: u64 = 100_000_000;

pub(crate) struct AiesimBackend {
    bridge: Box<dyn BridgeAbi>,
    /// Host-memory mirror. The bridge owns the authoritative GM model; we keep
    /// a Rust-side `HostMemory` so the FFI's `host_memory_mut()`-based read/write
    /// path is unchanged, and flush writes through to the bridge on run.
    host: HostMemory,
    start_col: u8,
    cols: usize,
    rows: usize,
    arch: String,
    /// Pending host-memory writes to push to the bridge before the next run.
    /// (Populated in apply path; see Step 3 note.)
    dirty: bool,
    /// Registered host-buffer regions (addr, size). The interpreter's executor
    /// uses these for runtime-sequence address patching; the cluster uses real
    /// DDR addresses directly, so aiesim just tracks them for GM/DDR setup.
    host_buffers: Vec<(u64, usize)>,
}

impl AiesimBackend {
    /// Construct from a ready `BridgeAbi` (real or mock). `create` must already
    /// have been called on the bridge by the caller (lib.rs selector).
    pub(crate) fn new(bridge: Box<dyn BridgeAbi>, arch: String, cols: usize, rows: usize) -> Self {
        Self {
            bridge,
            host: HostMemory::default(),
            start_col: 0,
            cols,
            rows,
            arch,
            dirty: false,
            host_buffers: Vec::new(),
        }
    }
}

impl NpuBackend for AiesimBackend {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        // Parser-driven data path: serialize the CDO op-stream and hand it to
        // the bridge, which replays it as ess_*() writes. The exact serialization
        // is defined in Task I.8 (encode_cdo); for now encode-then-load.
        let ops = encode_cdo(cdo);
        match self.bridge.load_cdo(&ops) {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err("aiesim bridge: load_cdo failed".to_string()),
        }
    }
    fn set_start_col(&mut self, start_col: u8) {
        self.start_col = start_col;
    }
    fn load_elf_bytes(&mut self, _col: usize, _row: usize, _data: &[u8]) -> Result<u32, String> {
        // ELF core images are delivered to the cluster as part of CDO/config in
        // the aiesim path; a standalone load is a no-op here. Return 0 (bytes
        // ack'd) to match the interpreter's contract. (Revisit if the bridge
        // needs an explicit core-image push -- tracked in Task II.6.)
        Ok(0)
    }
    fn host_memory_mut(&mut self) -> &mut HostMemory {
        self.dirty = true;
        &mut self.host
    }
    fn sync_cores_from_device(&mut self) {
        // No Rust-side core mirror for aiesim; the cluster is authoritative.
    }
    fn reset_for_new_context(&mut self) {
        let _ = self.bridge.reset();
        self.dirty = false;
    }
    fn reset_context(&mut self, _cid: ContextId) -> Result<(), ()> {
        match self.bridge.reset() {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err(()),
        }
    }
    fn execute_npu_instructions(&mut self, stream: &NpuInstructionStream) -> Result<(), String> {
        // Encode the runtime-sequence ops into the same tagged wire format and
        // hand them to the bridge for register-write replay. encode_npu is a
        // placeholder until Task I.8; the bridge stages them for the next run.
        let ops = encode_npu(stream);
        match self.bridge.exec_npu(&ops) {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err("aiesim bridge: exec_npu failed".to_string()),
        }
    }
    fn run(&mut self, max_cycles: u64, _observer: &mut dyn RunObserver) -> RunOutcome {
        // Flush any dirty host memory into the bridge GM model first.
        if self.dirty {
            flush_host_to_bridge(&self.host, self.bridge.as_mut());
            self.dirty = false;
        }
        // max_cycles == 0 means unbounded; pass the backend default to the bridge.
        let budget = if max_cycles == 0 { DEFAULT_CYCLE_BUDGET } else { max_cycles };
        let mut cycles = 0u64;
        let halt = match self.bridge.run(budget, &mut cycles) {
            BridgeHalt::Completed => HaltKind::Completed,
            BridgeHalt::Budget => HaltKind::Budget,
            BridgeHalt::Error => HaltKind::Error,
        };
        // Async-error surfacing through `observer` is a tier-3 item (Part II):
        // aiesim errors come back via error registers, not a Rust drain. For now
        // the bridge reports none; the observer is intentionally unused here.
        RunOutcome { cycles, halt }
    }
    fn add_host_buffer(&mut self, address: u64, size: usize) {
        // The cluster's shim-DMA uses real DDR addresses (no patching needed);
        // track the region so the GM/DDR model is set up before run.
        self.host_buffers.push((address, size));
    }
    fn clear_host_buffers(&mut self) {
        self.host_buffers.clear();
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn arch_name(&self) -> String {
        self.arch.clone()
    }
    // as_interpreter / as_interpreter_mut: default None -- aiesim is not an
    // interpreter. Tier-3 interpreter-only introspection is correctly absent.
}

/// Serialize a parsed CDO into the byte op-stream the bridge replays.
/// Placeholder until Task I.8 defines the real encoding; returns empty so the
/// mock-driven tests in this task compile.
pub(crate) fn encode_cdo(_cdo: &Cdo<'_>) -> Vec<u8> {
    Vec::new()
}

/// Serialize a runtime-sequence (NPU instruction) stream into the same tagged
/// wire format. Placeholder until Task I.8; returns empty for now.
pub(crate) fn encode_npu(_stream: &NpuInstructionStream) -> Vec<u8> {
    Vec::new()
}

/// Push populated host-memory regions into the bridge GM model.
/// Placeholder until Task I.8; no-op against the mock.
fn flush_host_to_bridge(_host: &HostMemory, _bridge: &mut dyn BridgeAbi) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aiesim::abi::mock::MockBridge;

    fn backend_with_mock() -> AiesimBackend {
        AiesimBackend::new(Box::new(MockBridge::default()), "aie2".to_string(), 5, 6)
    }

    #[test]
    fn topology_and_arch_reported() {
        let b = backend_with_mock();
        assert_eq!(b.cols(), 5);
        assert_eq!(b.rows(), 6);
        assert_eq!(b.arch_name(), "aie2");
        assert!(b.as_interpreter().is_none());
    }

    #[test]
    fn run_reports_outcome() {
        use crate::backend::RunObserver;
        use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
        struct NullObs;
        impl RunObserver for NullObs {
            fn on_async_errors(&mut self, _r: &[AmdxdnaAsyncError]) {}
        }
        let mut b = backend_with_mock();
        let out = b.run(0, &mut NullObs);
        assert_eq!(out.halt, HaltKind::Completed);
    }
}
```

- [ ] **Step 2: Run it to verify it fails, then passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --features aiesim --lib aiesim::backend`
Expected: compiles and PASSES (the placeholders are intentionally minimal). If
`HostMemory::default()` is not available, grep its constructors
(`grep -n "impl HostMemory\|fn new\|fn default" $(find .. -name host_memory.rs)`)
and use the real one inside `new`.

- [ ] **Step 3: Commit**

```bash
git add crates/xdna-emu-ffi/src/aiesim/backend.rs
git commit -m "ffi(aiesim): AiesimBackend implements NpuBackend over BridgeAbi

Each trait call maps to one bridge call, including the unified run and
execute_npu_instructions. CDO/NPU encoding + host-memory flush are
placeholders refined in the next task. Mock-driven unit tests.

Generated using Claude Code."
```

### Task I.8: CDO + NPU op-stream encoding + host-memory flush

**Files:**
- Modify: `crates/xdna-emu-ffi/src/aiesim/backend.rs` (replace the three placeholders)

Define how a parsed `Cdo` and an `NpuInstructionStream` become the byte streams the
bridge replays, and how populated host-memory regions flush into the bridge GM.
**Keep the encoding a faithful pass-through of the parser's op-stream** — the same
ops the interpreter consumes — so a cross-check isolates execution divergence (spec
§5, Path 2).

- [ ] **Step 1: Inspect the parser's op-stream shape**

Run: `grep -n "pub enum\|pub struct\|Write32\|BlockWrite\|MaskPoll" $(find .. -path '*parser*' -name '*.rs' | xargs grep -l "Cdo" | head)`
and read the `Cdo` op iterator API (how the interpreter walks it today — find the
interpreter's `apply_cdo` consumer). Capture the op variants (Write32, BlockWrite,
MaskPoll, NoOp, etc.) and their fields.

- [ ] **Step 2: Write the failing encoding test**

In `backend.rs` tests, add (adjust op constructors to the real parser API found in
Step 1):
```rust
#[test]
fn encode_cdo_emits_one_record_per_op() {
    // Build a tiny CDO with a couple of Write32s (use the real parser builder
    // or a fixture); assert encode_cdo produces a non-empty, length-consistent
    // byte stream that decodes back to the same (addr,value) pairs.
    // (Fill in with the concrete parser API from Step 1.)
}
```

- [ ] **Step 3: Define a stable wire format and implement `encode_cdo`**

Replace the placeholder `encode_cdo` with a real encoder. Wire format (little-
endian, one tagged record per op) — keep it minimal and mirror it in the C++
`cdo_replay` decoder (Task II.5):
```rust
// Op tags (must match aiesim-bridge/src/cdo_replay.cpp).
const OP_WRITE32: u8 = 1;
const OP_BLOCKWRITE: u8 = 2;
const OP_MASKPOLL: u8 = 3;

pub(crate) fn encode_cdo(cdo: &Cdo<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for op in cdo.ops() {       // <-- use the real iterator from Step 1
        match op {
            // Write32 { addr: u64, value: u32 }
            //   [OP_WRITE32][addr u64][value u32]
            // BlockWrite { addr: u64, words: &[u32] }
            //   [OP_BLOCKWRITE][addr u64][count u32][words...]
            // MaskPoll { addr: u64, mask: u32, value: u32 }
            //   [OP_MASKPOLL][addr u64][mask u32][value u32]
            // (Match on the actual parser enum; push bytes accordingly.)
            _ => {}
        }
    }
    out
}
```
Implement each arm with `out.extend_from_slice(&field.to_le_bytes())`. Document
that the **C++ decoder in Task II.5 is the consumer** and the two must stay in
lockstep (a shared header comment in both files referencing each other).

- [ ] **Step 4: Implement `encode_npu`**

Replace the `encode_npu` placeholder. The runtime sequence (`NpuInstructionStream`)
reduces to the same register-write primitives (DMA BD config, lock ops, MaskPoll),
so reuse the **same tag set and wire format** as `encode_cdo`. Walk the stream's
ops (grep the `NpuInstructionStream` / `NpuExecutor` op model — the interpreter's
`try_advance` consumer shows the variants) and emit the matching tagged records.
Add a round-trip test mirroring Step 2 for a small instruction stream. If the NPU
op model has primitives beyond Write32/BlockWrite/MaskPoll, extend the shared tag
enum (and note it for the Task II.5 decoder).

- [ ] **Step 5: Implement `flush_host_to_bridge`**

Replace the placeholder. Iterate the populated regions of `HostMemory` (use its
real region API found via grep) and call `bridge.write_gm(addr, bytes)` per region.

- [ ] **Step 6: Run the encoding tests + full FFI suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --features aiesim --lib`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-emu-ffi/src/aiesim/backend.rs
git commit -m "ffi(aiesim): real CDO + NPU op-stream encoding + host-memory flush

encode_cdo/encode_npu serialize the same parser op-streams the interpreter
consumes into a shared tagged little-endian wire format (decoder twin in
cdo_replay.cpp). flush_host_to_bridge pushes populated GM regions before
run. Round-trip unit-tested.

Generated using Claude Code."
```

### Task I.9: Wire the `aiesim` arm in `select_backend` (feature-gated)

**Files:**
- Create: `crates/xdna-emu-ffi/src/aiesim/bridge.rs` (real dlopen `BridgeAbi`)
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (`select_backend` aiesim arm, gated)

The `aiesim` arm builds the real `DlopenBridge`, calls `create`, then wraps it in
`AiesimBackend`. Under `--features aiesim` but **without** a bridge `.so` present,
it must fail gracefully (clear error), and Part I's tests must not require the `.so`.

- [ ] **Step 1: Write `bridge.rs` (dlopen the C ABI)**

Create `crates/xdna-emu-ffi/src/aiesim/bridge.rs`. It `dlopen`s
`libxdna_aiesim_bridge.so` (path from `XDNA_AIESIM_BRIDGE` env or a default next to
the emulator `.so`), resolves the C-ABI symbols, and implements `BridgeAbi` by
calling them. Use `libloading`. Sketch:
```rust
//! Real `BridgeAbi`: dlopen `libxdna_aiesim_bridge.so` and bind its C ABI.

use super::abi::{BridgeAbi, BridgeHalt, BridgeStatus};
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

type CreateFn = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void;
type LoadCdoFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int;
type ExecNpuFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int;
type WriteGmFn = unsafe extern "C" fn(*mut c_void, u64, *const u8, usize) -> c_int;
type ReadGmFn = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> c_int;
type RunFn = unsafe extern "C" fn(*mut c_void, u64, *mut u64) -> c_int;
type ReadRegFn = unsafe extern "C" fn(*mut c_void, u64) -> u32;
type ResetFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void);

pub(crate) struct DlopenBridge {
    _lib: Library, // kept alive for the symbols' lifetime
    handle: *mut c_void,
    // resolved fn pointers (stored as raw to avoid lifetime gymnastics)
    load_cdo: LoadCdoFn,
    exec_npu: ExecNpuFn,
    write_gm: WriteGmFn,
    read_gm: ReadGmFn,
    run: RunFn,
    read_reg: ReadRegFn,
    reset: ResetFn,
    destroy: DestroyFn,
}

impl DlopenBridge {
    /// Open the bridge and construct the cluster. Returns a clear error if the
    /// .so is missing (feature on but bridge not built) or a symbol is absent.
    pub(crate) fn open(arch: &str, device_json: &str) -> Result<Self, String> {
        let path = std::env::var("XDNA_AIESIM_BRIDGE")
            .unwrap_or_else(|_| "libxdna_aiesim_bridge.so".to_string());
        // SAFETY: loading a trusted, locally-built bridge; RTLD_GLOBAL so the
        // cluster can resolve the bridge's host globals.
        let lib = unsafe { Library::new(&path) }
            .map_err(|e| format!("aiesim: cannot load {path}: {e} (build it with scripts/build-aiesim-bridge.sh)"))?;
        unsafe {
            let create: Symbol<CreateFn> = lib.get(b"aiesim_create\0")
                .map_err(|e| format!("aiesim: missing aiesim_create: {e}"))?;
            let c_arch = CString::new(arch).unwrap();
            let c_json = CString::new(device_json).unwrap();
            let handle = create(c_arch.as_ptr(), c_json.as_ptr());
            if handle.is_null() {
                return Err("aiesim: aiesim_create returned null".to_string());
            }
            // Resolve the rest; copy the fn pointers out so we don't hold Symbols.
            let load_cdo = *lib.get::<LoadCdoFn>(b"aiesim_load_cdo\0").map_err(sym_err)?;
            let exec_npu = *lib.get::<ExecNpuFn>(b"aiesim_exec_npu\0").map_err(sym_err)?;
            let write_gm = *lib.get::<WriteGmFn>(b"aiesim_write_gm\0").map_err(sym_err)?;
            let read_gm = *lib.get::<ReadGmFn>(b"aiesim_read_gm\0").map_err(sym_err)?;
            let run = *lib.get::<RunFn>(b"aiesim_run\0").map_err(sym_err)?;
            let read_reg = *lib.get::<ReadRegFn>(b"aiesim_read_reg\0").map_err(sym_err)?;
            let reset = *lib.get::<ResetFn>(b"aiesim_reset\0").map_err(sym_err)?;
            let destroy = *lib.get::<DestroyFn>(b"aiesim_destroy\0").map_err(sym_err)?;
            Ok(Self { _lib: lib, handle, load_cdo, exec_npu, write_gm, read_gm, run, read_reg, reset, destroy })
        }
    }
}

fn sym_err(e: libloading::Error) -> String {
    format!("aiesim: missing bridge symbol: {e}")
}

impl Drop for DlopenBridge {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) };
    }
}

impl BridgeAbi for DlopenBridge {
    fn create(&mut self, _arch: &str, _device_json: &str) -> BridgeStatus {
        // Construction happens in `open`; this is a no-op so the trait shape
        // matches the mock. (Always Ok once `open` succeeded.)
        BridgeStatus::Ok
    }
    fn load_cdo(&mut self, ops: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.load_cdo)(self.handle, ops.as_ptr(), ops.len()) };
        if rc == 0 { BridgeStatus::Ok } else { BridgeStatus::Error }
    }
    fn exec_npu(&mut self, ops: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.exec_npu)(self.handle, ops.as_ptr(), ops.len()) };
        if rc == 0 { BridgeStatus::Ok } else { BridgeStatus::Error }
    }
    fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus {
        let rc = unsafe { (self.write_gm)(self.handle, addr, data.as_ptr(), data.len()) };
        if rc == 0 { BridgeStatus::Ok } else { BridgeStatus::Error }
    }
    fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus {
        let rc = unsafe { (self.read_gm)(self.handle, addr, out.as_mut_ptr(), out.len()) };
        if rc == 0 { BridgeStatus::Ok } else { BridgeStatus::Error }
    }
    fn run(&mut self, budget: u64, cycles_out: &mut u64) -> BridgeHalt {
        let rc = unsafe { (self.run)(self.handle, budget, cycles_out as *mut u64) };
        match rc {
            0 => BridgeHalt::Completed,
            1 => BridgeHalt::Budget,
            _ => BridgeHalt::Error,
        }
    }
    fn read_reg(&mut self, addr: u64) -> u32 {
        unsafe { (self.read_reg)(self.handle, addr) }
    }
    fn reset(&mut self) -> BridgeStatus {
        let rc = unsafe { (self.reset)(self.handle) };
        if rc == 0 { BridgeStatus::Ok } else { BridgeStatus::Error }
    }
}

// SAFETY: the bridge is a process singleton driven from one thread at a time
// (the plugin serializes handle access; the SystemC kernel is global). The raw
// `handle` pointer is never shared across threads concurrently.
unsafe impl Send for DlopenBridge {}
```

- [ ] **Step 2: Gate the `aiesim` arm in `select_backend`**

In `crates/xdna-emu-ffi/src/lib.rs`, replace the `"aiesim"` arm. Keep the
feature-off arm returning the existing clear error; add a feature-on arm:
```rust
        "aiesim" => {
            #[cfg(feature = "aiesim")]
            {
                // Arch + device JSON come from parsed-xclbin context in the real
                // flow; for the initial backend default to aie2 + discovered JSON.
                // (Threading real arch/JSON through is Task II.6.)
                let arch = "aie2";
                let device_json = std::env::var("XDNA_AIESIM_DEVICE_JSON")
                    .map_err(|_| "XDNA_BACKEND=aiesim: set XDNA_AIESIM_DEVICE_JSON to the cluster device model".to_string())?;
                let bridge = crate::aiesim::bridge::DlopenBridge::open(arch, &device_json)?;
                // Topology is queried from the cluster in the bridge; default to
                // NPU1 geometry here until the query is wired (Task II.6).
                let backend = crate::aiesim::backend::AiesimBackend::new(
                    Box::new(bridge), arch.to_string(), 5, 6);
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "aiesim"))]
            {
                Err("XDNA_BACKEND=aiesim: this build has no aiesim support (rebuild with --features aiesim)".to_string())
            }
        }
```
(Note: `select_backend` no longer invokes `make_interp` on this arm — the lazy
construction cleanup from Task 0.2 pays off here.)

- [ ] **Step 3: Build both feature states**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi`
Expected: builds (feature off; aiesim module absent).
Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi --features aiesim`
Expected: builds (DlopenBridge compiles; not exercised without a `.so`).

- [ ] **Step 4: Full suite, both states**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --features aiesim --lib`
Expected: both green. (No test opens the real `.so`; `DlopenBridge::open` is only
reached via `XDNA_BACKEND=aiesim`, not exercised in unit tests.)

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/aiesim/bridge.rs crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi(aiesim): dlopen bridge + feature-gated aiesim selector arm

DlopenBridge binds the C ABI of libxdna_aiesim_bridge.so and implements
BridgeAbi; missing .so / symbols fail with a clear message. The aiesim
arm of select_backend builds it under --features aiesim, else returns a
built-without-support error.

Generated using Claude Code."
```

### Task I.10: Part I gate — full library suite, both feature states

**Files:** none (verification only)

- [ ] **Step 1: Default suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: same green count as before Plan B (interpreter path unchanged).

- [ ] **Step 2: aiesim-feature suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --features aiesim --lib`
Expected: green, including the new `aiesim::*` tests.

- [ ] **Step 3: Rebuild the cdylib (both profiles touched by FFI)**

Run: `cargo build -p xdna-emu-ffi`
Expected: fresh `target/debug/libxdna_emu.so` (the interpreter path is byte-for-
byte the same; this just keeps the `.so` current per the operational rule).

**STOP — Part I complete and sandbox-green. Confirm with Maya before Part II.**
Part II builds + runs the C++ bridge, which requires aietools and runs **out of
sandbox** on the dev box (license/filesystem walls, same as bridge tests). This is
the right checkpoint to switch from subagent-driven to inline/exploratory.

---

# Part II — C++ bridge (gated, out-of-sandbox, drive inline)

Part II builds `libxdna_aiesim_bridge.so`. It is exploratory (the E513 fix) and
cannot be unit-tested in the sandbox. Drive it inline, building and verifying on
the dev box. Each task ends with an explicit out-of-sandbox verification command.

> Per CLAUDE.md: build once per target, never pipe build output through tail/grep,
> use `env -u XDNA_EMU` for any HW-adjacent invocation, and combine privileged ops
> into one `pkexec` call if any are needed (none expected here — all userspace).

### Task II.1: CMake skeleton + build script (compiles an empty bridge)

**Files:**
- Create: `aiesim-bridge/CMakeLists.txt`
- Create: `aiesim-bridge/include/xdna_aiesim_bridge.h`
- Create: `aiesim-bridge/src/c_abi.cpp` (stubs returning error)
- Create: `scripts/build-aiesim-bridge.sh`

- [ ] **Step 1: Write the C ABI header** (`aiesim-bridge/include/xdna_aiesim_bridge.h`)

The Rust `DlopenBridge` (Task I.9) is the consumer; signatures must match exactly:
```c
#ifndef XDNA_AIESIM_BRIDGE_H
#define XDNA_AIESIM_BRIDGE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Construct the cluster for `arch` ("aie2"/"aie2ps"/"aie") using device model
 * JSON at `device_json`. Spawns the SystemC service thread, elaborates once.
 * Returns an opaque handle, or NULL on failure. */
void *aiesim_create(const char *arch, const char *device_json);
/* Replay a CDO op-stream (tagged LE wire format; see cdo_replay.cpp). 0 = ok. */
int aiesim_load_cdo(void *h, const uint8_t *ops, size_t len);
/* Replay a runtime-sequence (NPU instruction) op-stream -- same wire format,
 * staged before run (arrives via execute_npu_instructions). 0 = ok. */
int aiesim_exec_npu(void *h, const uint8_t *ops, size_t len);
/* Host (GM/DDR) memory write/read. 0 = ok. */
int aiesim_write_gm(void *h, uint64_t addr, const uint8_t *data, size_t len);
int aiesim_read_gm(void *h, uint64_t addr, uint8_t *out, size_t len);
/* Run to quiescence or `budget` cycles. Sets *cycles_out. Returns
 * 0 = completed, 1 = budget, 2 = error. */
int aiesim_run(void *h, uint64_t budget, uint64_t *cycles_out);
/* Tier-2 zero-time backdoor register read. */
uint32_t aiesim_read_reg(void *h, uint64_t addr);
/* Reset logical state between submissions (CDO re-applied after). 0 = ok. */
int aiesim_reset(void *h);
/* Park the service thread + free logical state. */
void aiesim_destroy(void *h);
#ifdef __cplusplus
}
#endif
#endif
```

- [ ] **Step 2: Stub `c_abi.cpp`** returning error/NULL for every entry (so the
`.so` links before the real pieces exist):
```cpp
#include "xdna_aiesim_bridge.h"
extern "C" {
void *aiesim_create(const char *, const char *) { return nullptr; }
int aiesim_load_cdo(void *, const uint8_t *, size_t) { return 1; }
int aiesim_exec_npu(void *, const uint8_t *, size_t) { return 1; }
int aiesim_write_gm(void *, uint64_t, const uint8_t *, size_t) { return 1; }
int aiesim_read_gm(void *, uint64_t, uint8_t *, size_t) { return 1; }
int aiesim_run(void *, uint64_t, uint64_t *) { return 2; }
uint32_t aiesim_read_reg(void *, uint64_t) { return 0; }
int aiesim_reset(void *) { return 1; }
void aiesim_destroy(void *) {}
}
```

- [ ] **Step 3: Write `CMakeLists.txt`** mirroring `xrt-plugin/CMakeLists.txt`,
using the embedding recipe from the findings doc. Key elements: locate aietools
via `XILINX_VITIS_AIETOOLS` env (set by `activate-npu-env.sh`) or
`../mlir-aie/cmake/modulesXilinx/FindAIETools.cmake`; include
`<aietools>/data/osci_systemc/include`; link nothing closed (cluster is dlopened
at runtime by Task II.2); apply `-z execstack` and default-visibility for the host
globals; use **system** libstdc++. Skeleton:
```cmake
cmake_minimum_required(VERSION 3.16)
project(xdna_aiesim_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# aietools discovery (env from activate-npu-env.sh, else FindAIETools).
if(DEFINED ENV{XILINX_VITIS_AIETOOLS})
  set(AIETOOLS_DIR "$ENV{XILINX_VITIS_AIETOOLS}")
else()
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-aie/cmake/modulesXilinx")
  find_package(AIETools REQUIRED)
endif()
message(STATUS "aietools: ${AIETOOLS_DIR}")

set(SC_INCLUDE "${AIETOOLS_DIR}/data/osci_systemc/include")
set(SC_MAIN    "${AIETOOLS_DIR}/data/osci_systemc/sc_main")

add_library(xdna_aiesim_bridge SHARED
  src/c_abi.cpp
  # added as they are written:
  # src/sc_bootstrap.cpp src/aiesim_top.cpp src/ps_bridge.cpp
  # src/cdo_replay.cpp src/service_thread.cpp
  # ${SC_MAIN}/sc_main.cpp ${SC_MAIN}/sc_main_main.cpp
)
target_include_directories(xdna_aiesim_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${SC_INCLUDE}"
)
# Embedding recipe (findings doc): executable stack + exported host globals.
target_link_options(xdna_aiesim_bridge PRIVATE -z execstack -rdynamic)
target_link_libraries(xdna_aiesim_bridge PRIVATE ${CMAKE_DL_LIBS})
```

- [ ] **Step 4: Write `scripts/build-aiesim-bridge.sh`** mirroring
`rebuild-plugin.sh` (configure if needed, `make -j$(nproc)`, report the `.so`
path). It must source/respect `XILINX_VITIS_AIETOOLS`; print a clear message and
exit non-zero if aietools is absent.

- [ ] **Step 5: Build (out of sandbox)** — verification

Run (dev box, env activated):
`source toolchain-build/activate-npu-env.sh && ./scripts/build-aiesim-bridge.sh`
Expected: `aiesim-bridge/build/libxdna_aiesim_bridge.so` exists; `nm -D` shows the
eight `aiesim_*` symbols. (Cluster not yet loaded — stubs only.)

- [ ] **Step 6: Commit**

```bash
git add aiesim-bridge/CMakeLists.txt aiesim-bridge/include aiesim-bridge/src/c_abi.cpp scripts/build-aiesim-bridge.sh
git commit -m "aiesim-bridge: CMake skeleton + C ABI header + build script

Empty bridge that links and exports the eight aiesim_* C-ABI symbols the
Rust DlopenBridge binds. Mirrors xrt-plugin's CMake + rebuild script;
applies the in-process embedding flags. No closed deps at link time.

Generated using Claude Code."
```

### Task II.2: `sc_bootstrap` — host globals + SystemC main (banner prints)

**Files:**
- Create: `aiesim-bridge/src/sc_bootstrap.cpp`
- Modify: `aiesim-bridge/CMakeLists.txt` (add the source + the two `sc_main_*.cpp`)

Reproduce the proven spike: define the two host globals, compile aietools'
`sc_main`/`sc_main_main`, declare our `extern "C" sc_main`. This task only proves
SystemC embeds + runs in-process from the bridge `.so` (kernel banner), no cluster.

- [ ] **Step 1: Write `sc_bootstrap.cpp`** per the findings recipe:
```cpp
// In-process SystemC bootstrap for the aiesim bridge.
// Recipe: docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
#include <systemc.h>
#include <iostream>

// Host-executable contract: the dlopened cluster resolves these.
extern "C" {
bool sc_stop_at_end_of_main = false;
int  plio_complete = 0;
}

// The SystemC main bootstrap (sc_main_main.cpp) calls this with C linkage.
extern "C" int sc_main(int argc, char *argv[]) {
    std::cout << "[aiesim-bridge] sc_main entered (SystemC embedded)" << std::endl;
    // Service-thread loop is added in Task II.6; for now just prove we run.
    return 0;
}
```

- [ ] **Step 2: Add sources to CMake** — uncomment `sc_bootstrap.cpp` and the two
`${SC_MAIN}/sc_main*.cpp` in `add_library`. Link the system SystemC at runtime via
`LD_LIBRARY_PATH=<aietools>/lib/lnx64.o` (documented in the build script, not
hardcoded into the link).

- [ ] **Step 3: Build + smoke (out of sandbox)** — verification

Build the bridge, then a tiny driver that `dlopen`s it and calls into the bootstrap
(or temporarily have `aiesim_create` invoke the SystemC start). Expected: the
`[aiesim-bridge] sc_main entered` banner prints. This reproduces the spike's
proven gate from inside the bridge `.so`.

- [ ] **Step 4: Commit**

```bash
git add aiesim-bridge/src/sc_bootstrap.cpp aiesim-bridge/CMakeLists.txt
git commit -m "aiesim-bridge: SystemC bootstrap + host globals (banner prints)

Defines the two host globals and our extern-C sc_main, compiling
aietools' SystemC main bootstrap. Reproduces the proven in-process
SystemC embed from inside the bridge .so.

Generated using Claude Code."
```

### Task II.3: `aiesim_top` — E513-free cluster instantiation (the first real task)

**Files:**
- Create: `aiesim-bridge/src/aiesim_top.h`, `aiesim-bridge/src/aiesim_top.cpp`
- Modify: `aiesim-bridge/CMakeLists.txt`

**This is the spec's designated first implementation task** — the clean,
error-free `create_math_engine`. The findings doc records E533 (bare call) and
E513 (wrapped-but-incomplete-context). The fix: replicate `aie_xtlm.cpp`'s
**complete** pre-construction setup inside an `sc_module` constructed within live
elaboration (in our `sc_main`), not a throwaway context.

- [ ] **Step 1: Trace the real instantiation path**

Read `<aietools>/data/systemc/simlibs/aie_xtlm/aie_xtlm_v1_0_0/src/aie_xtlm.cpp`
around `create_cluster()` / the `create_math_engine` call (~266-357). Capture, in
order: every global/env it sets, the SystemC hierarchy it is inside, the exact
`create_math_engine(name, device_json, is_fast_pm, is_fast_dm)` arguments, and any
`sc_module_name` it threads in. Write these findings as a comment block at the top
of `aiesim_top.cpp` (this is the E513 root-cause record).

- [ ] **Step 2: Write `aiesim_top`** — an `sc_module` whose constructor performs
that setup and calls the factory:
```cpp
// aiesim_top.h
#pragma once
#include <systemc.h>
class aiesim_top : public sc_module {
public:
    SC_HAS_PROCESS(aiesim_top);
    aiesim_top(sc_module_name name, const char *arch, const char *device_json);
    ~aiesim_top();
    void *math_engine() const { return me_; }
private:
    void *me_ = nullptr;     // MathEngine* (opaque; closed type)
    void *cluster_lib_ = nullptr; // dlopen handle for libaie2_cluster_msm_*.so
};
```
```cpp
// aiesim_top.cpp
// E513 root-cause + fix (derived from aie_xtlm.cpp create_cluster, lines ~266-357):
//   <fill in the exact pre-construction sequence found in Step 1>
#include "aiesim_top.h"
#include <dlfcn.h>
#include <stdexcept>

typedef void *(*create_math_engine_fn)(const char *, const char *, bool, bool);

aiesim_top::aiesim_top(sc_module_name name, const char *arch, const char *device_json)
    : sc_module(name) {
    // 1. Reproduce aie_xtlm pre-construction setup (env/globals/order) HERE,
    //    inside live elaboration -- this is what satisfies E513.
    // 2. dlopen the per-arch cluster lib (RTLD_GLOBAL so it resolves our host
    //    globals; name from an arch->lib map: aie2 -> libaie2_cluster_msm_v1_0_0.osci.so).
    const char *lib = /* arch_to_cluster_lib(arch) */ "libaie2_cluster_msm_v1_0_0.osci.so";
    cluster_lib_ = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!cluster_lib_) throw std::runtime_error(std::string("dlopen cluster: ") + dlerror());
    auto factory = (create_math_engine_fn)dlsym(cluster_lib_, "create_math_engine");
    if (!factory) throw std::runtime_error("dlsym create_math_engine failed");
    // is_fast_pm / is_fast_dm: match aie_xtlm's defaults (functional vs timed).
    me_ = factory(/*name*/"aie", device_json, /*is_fast_pm*/false, /*is_fast_dm*/false);
    if (!me_) throw std::runtime_error("create_math_engine returned null");
    // 3. (Socket binding lives in Task II.4 once ps_bridge exists.)
}
aiesim_top::~aiesim_top() {
    if (cluster_lib_) dlclose(cluster_lib_);
}
```

- [ ] **Step 3: Instantiate it from `sc_main`** (replace the banner-only body with
constructing `aiesim_top top("top", arch, device_json);` inside elaboration).

- [ ] **Step 4: Build + run (out of sandbox)** — the gate

Build, then drive `aiesim_create("aie2", "<aietools>/data/aie_ml/devices/VC2802.json")`.
Expected: **no E513 / E533**; the cluster constructs cleanly. This is the
hello-cluster gate — the spec's first-task success criterion. If E513 persists,
the Step-1 trace is incomplete: diff our setup against `aie_xtlm` line-by-line
(this is the exploratory core; iterate here).

- [ ] **Step 5: Commit**

```bash
git add aiesim-bridge/src/aiesim_top.h aiesim-bridge/src/aiesim_top.cpp aiesim-bridge/CMakeLists.txt
git commit -m "aiesim-bridge: E513-free cluster instantiation (hello-cluster)

aiesim_top is an sc_module whose ctor replicates aie_xtlm's complete
pre-construction setup inside live elaboration, then dlopens the per-arch
cluster and calls create_math_engine. Resolves the E513 the spike left
open. Cluster constructs cleanly from the bridge .so.

Generated using Claude Code."
```

### Task II.4: `ps_bridge` — the `ess_*()` seam + socket bindings

**Files:**
- Create: `aiesim-bridge/src/ps_bridge.h`, `aiesim-bridge/src/ps_bridge.cpp`
- Modify: `aiesim-bridge/src/aiesim_top.cpp` (bind sockets), `CMakeLists.txt`

Twin `PSIP_ps_i3` (`genwrapper_for_ps.cpp`): provide the `ess_*()` weak symbols as
TLM transactions, with the two bindings (host→cluster config/MMIO via
`get_ss_aximm_rd/wr`; cluster→host DDR via `shim_dma_rd/wr_socket`).

- [ ] **Step 1: Enumerate the real `ess_*` symbol set** (the discovery the spec
left ambiguous). Build the open HAL `-D__AIESIM__` (or inspect the prebuilt
`libxaienginecdo`) and list referenced weak symbols:
`nm -uC <libxaienginecdo>.so | grep ess_` — and cross-check against
`genwrapper_for_ps.cpp` (Write32/Read32/Write128/Read128/WriteGM/ReadGM). Record
the authoritative set in a comment; **provide exactly those**. Note whether
`ess_WriteCmd` actually exists (the spec assumed it; the reference lacks it).

- [ ] **Step 2: Write `ps_bridge`** modeled on `PSIP_ps_i3`: an `IPBlock`/`sc_module`
holding the initiator socket to the cluster's aximm rd/wr and a DDR-model target
for shim-DMA. Implement `write32/read32/write128/read128/writeGM/readGM` as
`b_transport` calls (the genwrapper's `aximm_transaction` is the template) plus a
backing `std::map`/buffer for GM. Provide the `ess_*()` free functions delegating
to the singleton, exactly as genwrapper does:
```cpp
void ess_Write32(uint64_t a, uint32_t d) { ps_bridge::instance()->write32(a, d); }
uint32_t ess_Read32(uint64_t a) { return ps_bridge::instance()->read32(a); }
// ... ess_Write128/Read128/WriteGM/ReadGM, and ess_WriteCmd ONLY if Step 1 found it.
```

- [ ] **Step 3: Add a zero-time backdoor read for tier-2**

For `aiesim_read_reg` (tier-2), use the cluster's `transport_dbg_cb` (zero sim-time
backdoor) rather than a timed `aximm_transaction`, so register introspection never
perturbs the model. Wire `ps_bridge::read32_backdoor(addr)` to it; the timed
`read32` stays for the functional path. (Resolve `transport_dbg_cb` from the
`MathEngine` vtable/symbol per the 61-method capture in the findings/spec.)

- [ ] **Step 4: Bind the sockets in `aiesim_top`** — after `create_math_engine`,
construct the `ps_bridge`, bind host→cluster (`me->get_ss_aximm_rd/wr()`) and
cluster→host (`me->shim_dma_rd/wr_socket(col)` → ps_bridge DDR target), per spec §5.

- [ ] **Step 5: Build + verify a single register write/read round-trips** (out of
sandbox): drive `aiesim_create`, then `aiesim_read_reg` on a known register after
an `ess_Write32` through `load_cdo` of a one-op stream; confirm the value matches.

- [ ] **Step 6: Commit**

```bash
git add aiesim-bridge/src/ps_bridge.h aiesim-bridge/src/ps_bridge.cpp aiesim-bridge/src/aiesim_top.cpp aiesim-bridge/CMakeLists.txt
git commit -m "aiesim-bridge: ps_bridge ess_*() seam + socket bindings

Twin of PSIP_ps_i3: provides the exact ess_* weak symbols the __AIESIM__
HAL references (enumerated, not assumed) as TLM transactions, with
host->cluster aximm and cluster->host DDR bindings. Tier-2 register reads
use the zero-time transport_dbg_cb backdoor.

Generated using Claude Code."
```

### Task II.5: `cdo_replay` — decode the op-stream, drive `ess_*()`

**Files:**
- Create: `aiesim-bridge/src/cdo_replay.h`, `aiesim-bridge/src/cdo_replay.cpp`
- Modify: `aiesim-bridge/src/c_abi.cpp` (`aiesim_load_cdo` + `aiesim_exec_npu` call
  it), `CMakeLists.txt`

Decode the **exact** tagged LE wire format `encode_cdo`/`encode_npu` produce (Task I.8) and
replay each op via `ess_*()`. The two files are a matched pair — keep the tag
constants identical and cross-reference in comments. The decoder is shared:
`aiesim_load_cdo` (config) and `aiesim_exec_npu` (runtime sequence) both feed the
same wire format through it.

- [ ] **Step 1: Write the decoder** mirroring the Rust encoder's tags
(`OP_WRITE32=1`, `OP_BLOCKWRITE=2`, `OP_MASKPOLL=3`):
```cpp
// cdo_replay.cpp -- decoder twin of crates/xdna-emu-ffi/src/aiesim/backend.rs::encode_cdo
#include "cdo_replay.h"
#include <cstring>
// ess_*() declared in ps_bridge.h
enum { OP_WRITE32 = 1, OP_BLOCKWRITE = 2, OP_MASKPOLL = 3 };

int cdo_replay(const uint8_t *ops, size_t len) {
    size_t i = 0;
    auto rd_u64 = [&](uint64_t &v){ std::memcpy(&v, ops + i, 8); i += 8; };
    auto rd_u32 = [&](uint32_t &v){ std::memcpy(&v, ops + i, 4); i += 4; };
    while (i < len) {
        uint8_t tag = ops[i++];
        if (tag == OP_WRITE32) {
            uint64_t a; uint32_t d; rd_u64(a); rd_u32(d); ess_Write32(a, d);
        } else if (tag == OP_BLOCKWRITE) {
            uint64_t a; uint32_t n; rd_u64(a); rd_u32(n);
            for (uint32_t w = 0; w < n; ++w) { uint32_t d; rd_u32(d); ess_Write32(a + 4*w, d); }
        } else if (tag == OP_MASKPOLL) {
            uint64_t a; uint32_t m, v; rd_u64(a); rd_u32(m); rd_u32(v);
            // MaskPoll: poll until (read32(a) & m) == v, with a cycle cap.
            // (Implement against ess_Read32 + a bounded loop / sc_start step.)
        } else {
            return 1; // unknown tag -- encoder/decoder drift
        }
    }
    return 0;
}
```
(Implement `OP_BLOCKWRITE` with a native blockwrite if the HAL exposes one;
per-word `ess_Write32` is the safe baseline.)

- [ ] **Step 2: Call it from both `aiesim_load_cdo` and `aiesim_exec_npu`** in
`c_abi.cpp` (both stage the same wire format; runtime-sequence ops just arrive via
the exec-npu entry).

- [ ] **Step 3: Build + verify** (out of sandbox): load a small real CDO op-stream
(produced by the Rust encoder from a parsed fixture xclbin) and confirm the
expected registers read back via `aiesim_read_reg`.

- [ ] **Step 4: Commit**

```bash
git add aiesim-bridge/src/cdo_replay.h aiesim-bridge/src/cdo_replay.cpp aiesim-bridge/src/c_abi.cpp aiesim-bridge/CMakeLists.txt
git commit -m "aiesim-bridge: cdo_replay decodes op-stream, drives ess_*()

Decoder twin of the Rust encode_cdo wire format; replays Write32/
BlockWrite/MaskPoll via the ess_*() seam. aiesim_load_cdo wires it in.

Generated using Claude Code."
```

### Task II.6: `service_thread` — elaborate-once lifecycle + command queue

**Files:**
- Create: `aiesim-bridge/src/service_thread.h`, `aiesim-bridge/src/service_thread.cpp`
- Modify: `aiesim-bridge/src/c_abi.cpp` (all entries marshal onto the queue), `CMakeLists.txt`

Implement the spec §6 model: the first `aiesim_create` spawns a thread that runs
`sc_elab_and_sim(sc_main)`; `sc_main` constructs `aiesim_top` once, then loops
pulling commands (LOAD_CDO / WRITE_GM / READ_GM / READ_REG = backdoor zero-time;
RUN = `sc_start()` to quiescence). The C-ABI calls push a command + block for the
reply. Process-singleton; one SystemC sim per process.

- [ ] **Step 1: Write the command-queue + service-thread**

A thread-safe queue of tagged commands with reply slots (condvar handshake). The
service thread owns `aiesim_top`; the C-ABI calls (on the plugin's thread) enqueue
and wait. `aiesim_run` maps to a RUN command → `sc_start()` until `plio_complete`
(completion) or the cycle budget → reply `{halt, cycles}`. READ_REG uses the
backdoor (zero sim-time), so it can be serviced without advancing the kernel.

- [ ] **Step 2: Route `c_abi.cpp` through the queue** — each `aiesim_*` entry
becomes "enqueue command, block for reply." `aiesim_create` is special: first call
spawns the thread + waits for elaboration to finish; returns the singleton handle.
`aiesim_destroy` parks the thread (real `end_of_simulation` at process exit).

- [ ] **Step 3: Query topology from the cluster** — implement the arch/topology
query the Rust selector stubbed (Task I.9 used hardcoded 5x6): expose
`aiesim_create` filling cols/rows from `me->get_num_cols()` etc., and thread real
arch + device JSON through (replacing the `XDNA_AIESIM_DEVICE_JSON` stopgap once
the selector can pass them from parsed-xclbin context). Update the Rust side to
read topology back if the ABI is extended; otherwise document the query stays
bridge-internal and the Rust geometry is informational.

- [ ] **Step 4: Build + run a full submission** (out of sandbox): `create` →
`load_cdo` → `write_gm` → `run` → `read_gm`, asserting the kernel actually advances
(cycles > 0) and completes. This is the first end-to-end batch run through the
service thread.

- [ ] **Step 5: Commit**

```bash
git add aiesim-bridge/src/service_thread.h aiesim-bridge/src/service_thread.cpp aiesim-bridge/src/c_abi.cpp aiesim-bridge/CMakeLists.txt
git commit -m "aiesim-bridge: elaborate-once service thread + command queue

First aiesim_create spawns the SystemC thread, constructs the cluster
once, then services LOAD_CDO/WRITE_GM/READ_GM/READ_REG(backdoor)/RUN
commands. RUN drives sc_start to quiescence and maps to halt+cycles.
Process-singleton lifecycle per spec section 6.

Generated using Claude Code."
```

### Task II.7: Tier-2 bring-up — hello-cluster + in-process-vs-swap exact-match

**Files:**
- Create: `tests/aiesim_bringup.rs` (gated integration test) or a scripted harness
  under `scripts/` — whichever fits the out-of-sandbox runner.

- [ ] **Step 1: Hello-cluster end-to-end through the FFI** — with the bridge built
and `--features aiesim`, run a trivial kernel xclbin through
`XDNA_BACKEND=aiesim` + the FFI path; assert correct output. (This re-confirms
Task II.6 through the *real* `AiesimBackend` + `select_backend` seam, not the C ABI
directly.)

- [ ] **Step 2: In-process vs. the proven ELF-swap path** — same kernel, both
aiesim routes (in-process backend vs. the `docs/aiesimulator.md` swap), assert
**exact-match** output. This isolates "wired the in-process backend correctly"
from "aiesim itself correct."

- [ ] **Step 3: Grid cells** — {Peano, Chess} core ELFs × aiesim-in-process produce
correct output, completing the 2×3 grid through the real seam.

- [ ] **Step 4: Commit** the gated tests + a short `aiesim-bridge/README.md`
documenting the build + run (env, device JSON, `XDNA_BACKEND=aiesim`).

```bash
git add tests/aiesim_bringup.rs aiesim-bridge/README.md
git commit -m "aiesim-bridge: tier-2 bring-up tests (hello-cluster, vs-swap, grid)

Gated integration tests through the real XDNA_BACKEND=aiesim seam:
hello-cluster end-to-end, in-process vs ELF-swap exact-match, and the
Peano/Chess x aiesim grid. README documents build + run.

Generated using Claude Code."
```

### Task II.8: Tier-3 oracle role — differential vs. interpreter + third runtime

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (add aiesim as a third runtime, gated)
- Create: a differential harness (interpreter vs. aiesim, same FFI seam)

- [ ] **Step 1: Differential interpreter-vs-aiesim** on a small kernel corpus
through the same FFI seam: compare output **and** tier-2 register/memory state at
sync points (via `read_reg`/`read_gm`). Triage divergences → the "never fall
behind" backlog. (The Phoenix-survival corpus is the natural input once that
thread executes — link, don't block on it.)

- [ ] **Step 2: aiesim as a third runtime in `emu-bridge-test.sh`** — HW vs. EMU
vs. aiesim, gated on the bridge `.so` being present (absent → silently skip the
aiesim column, never fail the run). This is the "third bridge-test runtime" unlock.

- [ ] **Step 3: Commit**

```bash
git add scripts/emu-bridge-test.sh scripts/aiesim-diff.sh
git commit -m "aiesim: third bridge-test runtime + interpreter differential

emu-bridge-test.sh gains an optional aiesim column (HW vs EMU vs aiesim),
gated on the bridge .so. A differential harness compares interpreter vs
aiesim output + tier-2 state, feeding divergences to the never-fall-
behind backlog.

Generated using Claude Code."
```

---

## Out of scope (explicit, per spec §10)

- **Phase 2 — HAL-driven independent CDO replay** (`hal_driver`): a second,
  independent CDO interpretation as a stricter oracle. Sequenced after this plan.
  Gating unknown: whether aie-rt exposes a clean raw-CDO→`XAie_*` ingest path.
- **Custom device-model generation** — emitting our own device JSON (one memtile
  row, real NPU1 5×6) — gated on decoding the binary `XbV18.3` format.
- **Interpreter feature backlog** — closing tier-3 gaps aiesim exposes (guidance
  JSON, memory-violation diagnostics, FIFO guidance, native watchpoints,
  event-trace, VCD).

## Self-review notes (carried into execution)

- **Sandbox boundary is the Part I/II line.** Everything through Task I.10 is
  `cargo test --lib` green with no aietools. Part II builds + runs out of sandbox.
- **Encoder/decoder are a matched pair** (Task I.8 ↔ II.5). Tag constants and
  field order must stay identical; both files cross-reference each other.
- **Three stubs in Task I.7** (`encode_cdo`, `encode_npu`, `flush_host_to_bridge`)
  are intentionally minimal so the backend compiles; Task I.8 replaces them with
  the real impls.
  Don't ship I.5 to bring-up without I.6.
- **Topology hardcode** in the Task I.7 selector (5×6) is a stopgap; Task II.6
  queries the cluster and is where it gets resolved.
- **`ess_WriteCmd` is unverified** — Task II.4 Step 1 is the authority on the real
  symbol set; do not provide symbols the HAL doesn't reference.
- **`run()` seam decision** (dispatch, not unification) is the one design call to
  confirm with Maya before Part I lands — it is deliberately not a single unified
  loop, to keep interpreter behavior byte-for-byte unchanged.
