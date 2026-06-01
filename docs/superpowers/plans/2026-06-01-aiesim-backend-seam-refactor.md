# aiesim Integrator — Plan A: NpuBackend Seam Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce an `NpuBackend` trait behind `XdnaEmuHandle` so the FFI no
longer hardwires `InterpreterEngine`, establishing the seam a second backend
(aiesim, Plan B) plugs into — with behavior byte-for-byte unchanged.

**Architecture:** A narrow object-safe trait (`Box<dyn NpuBackend>`) carries the
cross-backend operations (config, host memory, lifecycle, topology); deep
interpreter-only introspection routes through an `as_interpreter()` downcast
hatch that returns `None` for non-interpreter backends. `run()` is deliberately
NOT abstracted in this plan — it stays interpreter-coupled via
`as_interpreter_mut()` because it is entangled with `npu_executor`, the TDR
detectors, and per-cycle FFI async-callback firing; the clean `run()` seam is
designed in Plan B where a real second implementation forces its shape. An
`XDNA_BACKEND` env selector chooses the backend at `xdna_emu_create`; only
`interpreter` is wired here (`aiesim` returns a clean "not supported" error).

**Tech Stack:** Rust (`crates/xdna-emu-ffi`), `xdna-emu-core`
(`InterpreterEngine`, `DeviceState`, `HostMemory`, `Cdo`, `ContextId`).

**Scope boundary (explicitly OUT of this plan — they are Plan B / later):** the
C++ `aiesim-bridge`, the E513 hello-cluster instantiation, the `ess_*()` data
path, the SystemC service-thread lifecycle, the cargo `aiesim` feature + CMake +
runtime dlopen, the phase-2 HAL replay, custom device-JSON generation, the
interpreter feature backlog. This plan ships a working emulator behind a trait;
nothing aiesim-specific is built yet.

**Reference spec:** `docs/superpowers/specs/2026-06-01-aiesim-integrator-design.md`
(§3 the seam, tiers 1-3).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `crates/xdna-emu-ffi/src/backend.rs` | **Create** | The `NpuBackend` trait, `impl NpuBackend for InterpreterEngine`, and a `cfg(test)` `MockBackend`. |
| `crates/xdna-emu-ffi/src/lib.rs` | Modify | `XdnaEmuHandle.engine` → `backend: Box<dyn NpuBackend>`; `mod backend`; `xdna_emu_create` selector; `xdna_emu_reset_context`. |
| `crates/xdna-emu-ffi/src/config.rs` | Modify | Migrate 5 `engine` sites to trait methods. |
| `crates/xdna-emu-ffi/src/memory.rs` | Modify | Migrate 5 `host_memory_mut` sites (receiver swap). |
| `crates/xdna-emu-ffi/src/execution.rs` | Modify | Migrate the run loop + 2 helpers via `as_interpreter_mut()`/`as_interpreter()`. |
| `crates/xdna-emu-ffi/src/query.rs` | Modify | Migrate topology to trait methods; diagnostics to `as_interpreter()`. |
| `crates/xdna-emu-ffi/src/context.rs` | Modify | Migrate context-state read to `as_interpreter()`. |
| `crates/xdna-emu-ffi/src/async_errors.rs` | Modify | Migrate 11 `device()`/`device_mut()` sites to `as_interpreter[_mut]()`. |
| `crates/xdna-emu-ffi/src/firmware.rs` | Modify | Migrate 1 real + test-only `device()` sites. |

**Convention used throughout the migration:**
- A call that maps to a **trait method** (`cols`, `rows`, `arch_name`,
  `set_start_col`, `apply_cdo`, `load_elf_bytes`, `sync_cores_from_device`,
  `reset_for_new_context`, `reset_context`, `host_memory_mut`): `handle.engine.X`
  → `handle.backend.X`.
- A call that reaches **interpreter-only `device()`/`device_mut()` internals**:
  `handle.engine.device()` → `handle.backend.as_interpreter().expect("Plan A: interpreter backend")`
  (and `as_interpreter_mut()` for `device_mut()`). The `expect` documents the
  Plan-A invariant: the selector guarantees an interpreter backend at runtime.
  FFI functions whose contract allows a clean failure instead use a `match …
  { Some(e) => …, None => <graceful error/zero> }` (Task 6).
- **Migration tables show only the fragment to replace.** Preserve everything
  around it — trailing casts (`as u8`), `?`, `.is_err()`, method chains, and the
  enclosing statement are untouched.

---

## Task 1: Define the `NpuBackend` trait + `MockBackend` test double

**Files:**
- Create: `crates/xdna-emu-ffi/src/backend.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (add `mod backend;`)

- [ ] **Step 1: Create `backend.rs` with the trait and a mock**

```rust
//! Backend abstraction behind `XdnaEmuHandle`.
//!
//! The FFI dispatches over `dyn NpuBackend` so a second backend (aiesim,
//! Plan B) can replace the hand-rolled interpreter without touching call
//! sites. The trait is intentionally narrow: it carries the cross-backend
//! operations the FFI needs, while deep interpreter-only introspection
//! routes through the `as_interpreter()` downcast hatch.
//!
//! `run()` is NOT on this trait in Plan A — it is entangled with the FFI's
//! `npu_executor`, the TDR detectors, and per-cycle async-callback firing.
//! `xdna_emu_run` reaches the interpreter via `as_interpreter_mut()`. The
//! clean `run()` seam is designed in Plan B.

use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::interpreter::engine::InterpreterEngine;
use xdna_emu_core::parser::Cdo; // same path config.rs imports (re-exported at parser root)

/// The operations the FFI performs that are common to every backend.
pub trait NpuBackend {
    // --- configuration ---
    // Cdo is lifetime-generic (`Cdo<'a>`); the elided `<'_>` keeps the trait
    // object-safe and lets callers pass any borrowed CDO.
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String>;
    fn set_start_col(&mut self, start_col: u8);
    fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String>;

    // --- host memory (tier-2) ---
    fn host_memory_mut(&mut self) -> &mut HostMemory;

    // --- lifecycle ---
    fn sync_cores_from_device(&mut self);
    fn reset_for_new_context(&mut self);
    fn reset_context(&mut self, cid: ContextId) -> Result<(), ()>;

    // --- topology / identity (tier-2, cross-backend) ---
    fn cols(&self) -> usize;
    fn rows(&self) -> usize;
    fn arch_name(&self) -> String;

    // --- downcast hatch (tier-3: interpreter-only introspection) ---
    fn as_interpreter(&self) -> Option<&InterpreterEngine> { None }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> { None }
}

#[cfg(test)]
pub(crate) mod mock {
    use super::*;

    /// A do-nothing backend used to prove FFI dispatch and the graceful
    /// "not an interpreter" path without constructing a real engine.
    #[derive(Default)]
    pub(crate) struct MockBackend {
        pub apply_cdo_calls: u32,
        pub start_col: u8,
        pub reset_for_new_context_calls: u32,
    }

    impl NpuBackend for MockBackend {
        fn apply_cdo(&mut self, _cdo: &Cdo<'_>) -> Result<(), String> {
            self.apply_cdo_calls += 1;
            Ok(())
        }
        fn set_start_col(&mut self, start_col: u8) { self.start_col = start_col; }
        fn load_elf_bytes(&mut self, _c: usize, _r: usize, _d: &[u8]) -> Result<u32, String> {
            Ok(0)
        }
        fn host_memory_mut(&mut self) -> &mut HostMemory {
            unimplemented!("MockBackend has no host memory")
        }
        fn sync_cores_from_device(&mut self) {}
        fn reset_for_new_context(&mut self) { self.reset_for_new_context_calls += 1; }
        fn reset_context(&mut self, _cid: ContextId) -> Result<(), ()> { Ok(()) }
        fn cols(&self) -> usize { 5 }
        fn rows(&self) -> usize { 6 }
        fn arch_name(&self) -> String { "mock".to_string() }
        // as_interpreter / as_interpreter_mut use the default None impls.
    }
}
```

- [ ] **Step 2: Wire the module into the crate**

In `crates/xdna-emu-ffi/src/lib.rs`, add to the module list (near line 39-46,
with the other `mod` declarations):

```rust
mod backend;
```

And re-export the trait (near the `pub use` block, lines 48-55):

```rust
pub use backend::NpuBackend;
```

- [ ] **Step 3: Write the failing test (mock defaults + dispatch)**

Append to `crates/xdna-emu-ffi/src/backend.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::mock::MockBackend;
    use super::NpuBackend;

    #[test]
    fn mock_backend_is_not_an_interpreter() {
        let m = MockBackend::default();
        assert!(m.as_interpreter().is_none());
    }

    #[test]
    fn mock_backend_dispatches_trait_methods() {
        let mut m: Box<dyn NpuBackend> = Box::new(MockBackend::default());
        m.set_start_col(3);
        m.reset_for_new_context();
        assert_eq!(m.cols(), 5);
        assert_eq!(m.rows(), 6);
        assert_eq!(m.arch_name(), "mock");
    }
}
```

- [ ] **Step 4: Run the test — verify it fails to compile first, then passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib backend`
Expected: compiles and PASSES. If `Cdo`, `HostMemory`, or `ContextId` import
paths are wrong, the compiler names the correct path — fix the `use` lines in
Step 1 to match (these three types are defined in `xdna-emu-core`; confirm with
`grep -rn "pub struct Cdo\|pub struct HostMemory\|pub struct ContextId" ../../src ../xdna-archspec/src` from the crate dir if needed).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi: add NpuBackend trait + MockBackend test double

Establishes the backend seam (Plan A). Trait is narrow: cross-backend
config/memory/lifecycle/topology ops, with an as_interpreter() downcast
hatch for interpreter-only introspection. run() intentionally deferred to
Plan B. No call sites migrated yet.

Generated using Claude Code."
```

---

## Task 2: Implement `NpuBackend` for `InterpreterEngine`

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `backend.rs`:

```rust
#[test]
fn interpreter_implements_backend_and_downcasts() {
    use xdna_emu_core::interpreter::engine::InterpreterEngine;
    let mut eng = InterpreterEngine::new_npu1();

    // Trait methods reflect the real device.
    let b: &mut dyn NpuBackend = &mut eng;
    assert_eq!(b.cols(), 5);
    assert_eq!(b.rows(), 6);
    assert!(b.arch_name().to_lowercase().contains("npu")
        || b.arch_name().to_lowercase().contains("aie"));

    // The downcast hatch returns the engine.
    assert!(b.as_interpreter().is_some());
    assert!(b.as_interpreter_mut().is_some());
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib interpreter_implements_backend`
Expected: FAIL — `the trait bound InterpreterEngine: NpuBackend is not satisfied`.

- [ ] **Step 3: Implement the trait for `InterpreterEngine`**

Add to `backend.rs` (after the trait definition, before the `cfg(test)` mock):

```rust
impl NpuBackend for InterpreterEngine {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        // DeviceState::apply_cdo returns a custom Result<()>; normalize the
        // error to String (config.rs only Displays it).
        self.device_mut().apply_cdo(cdo).map_err(|e| e.to_string())
    }
    fn set_start_col(&mut self, start_col: u8) {
        self.device_mut().set_start_col(start_col);
    }
    fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String> {
        InterpreterEngine::load_elf_bytes(self, col, row, data)
    }
    fn host_memory_mut(&mut self) -> &mut HostMemory {
        InterpreterEngine::host_memory_mut(self)
    }
    fn sync_cores_from_device(&mut self) {
        InterpreterEngine::sync_cores_from_device(self);
    }
    fn reset_for_new_context(&mut self) {
        InterpreterEngine::reset_for_new_context(self);
    }
    fn reset_context(&mut self, cid: ContextId) -> Result<(), ()> {
        self.device_mut().reset_context(cid).map_err(|_| ())
    }
    fn cols(&self) -> usize { self.device().cols() }
    fn rows(&self) -> usize { self.device().rows() }
    fn arch_name(&self) -> String { self.device().arch_name().to_string() }

    fn as_interpreter(&self) -> Option<&InterpreterEngine> { Some(self) }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> { Some(self) }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib backend`
Expected: PASS (all backend tests). If `reset_context`'s error type isn't unit,
the compiler shows the real `Err` type — adjust the `map_err` accordingly. If
`arch_name()` returns `&str` vs `String`, the `.to_string()` already normalizes;
if it returns something else, match it.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs
git commit -m "ffi: implement NpuBackend for InterpreterEngine

Delegates trait methods to existing engine/device methods; as_interpreter()
returns Some(self). The interpreter is now a drop-in dyn NpuBackend.

Generated using Claude Code."
```

---

## Task 3: Migrate `XdnaEmuHandle` to `Box<dyn NpuBackend>`

This is one atomic change: a Rust field-type flip breaks every `handle.engine`
site until all are migrated, so this task ends in a single compiling commit.
Each step migrates one file; the build is only expected to pass after Step 9.

**Files:** all 9 FFI source files (see File Structure table).

- [ ] **Step 1: Flip the handle field and `xdna_emu_create`**

In `crates/xdna-emu-ffi/src/lib.rs`:

Change the field (line ~98) from:
```rust
    pub(crate) engine: InterpreterEngine,
```
to:
```rust
    pub(crate) backend: Box<dyn crate::backend::NpuBackend>,
```

In `xdna_emu_create` (lines ~210-222), change:
```rust
    let mut engine = InterpreterEngine::new_npu1();
    engine.set_stall_threshold(config.stall_threshold());
    let handle = Box::new(XdnaEmuHandle {
        engine,
```
to:
```rust
    let mut engine = InterpreterEngine::new_npu1();
    engine.set_stall_threshold(config.stall_threshold());
    let handle = Box::new(XdnaEmuHandle {
        backend: Box::new(engine),
```

In `xdna_emu_reset_context` (lines ~272-276), change:
```rust
    if handle.engine.device_mut().reset_context(cid).is_err() {
        log::error!("xdna_emu_reset_context: invalid context_id {}", context_id);
        return XdnaEmuResult::ExecutionError;
    }
    handle.engine.reset_for_new_context();
```
to:
```rust
    if handle.backend.reset_context(cid).is_err() {
        log::error!("xdna_emu_reset_context: invalid context_id {}", context_id);
        return XdnaEmuResult::ExecutionError;
    }
    handle.backend.reset_for_new_context();
```

- [ ] **Step 2: Migrate `config.rs`** (trait-method receiver swaps)

| Line | From | To |
|---|---|---|
| 95 | `handle.engine.device_mut().set_start_col(start_col as u8)` | `handle.backend.set_start_col(start_col as u8)` |
| 152 | `handle.engine.device_mut().apply_cdo(&cdo)` | `handle.backend.apply_cdo(&cdo)` |
| 236 | `handle.engine.load_elf_bytes(col as usize, row as usize, &elf_data)` | `handle.backend.load_elf_bytes(col as usize, row as usize, &elf_data)` |
| 253 | `handle.engine.sync_cores_from_device()` | `handle.backend.sync_cores_from_device()` |
| 281 | `handle.engine.device_mut().set_start_col(start_col)` | `handle.backend.set_start_col(start_col)` |

- [ ] **Step 3: Migrate `memory.rs`** (receiver swap, 5 sites)

Replace every `handle.engine.host_memory_mut()` with `handle.backend.host_memory_mut()`
(lines 37, 74, 112, 201, 254). The `npu_executor` calls on lines 41, 132, 153,
214 are unchanged (separate field).

- [ ] **Step 4: Migrate `query.rs`** (topology → trait; diagnostics → hatch)

| Line | From | To |
|---|---|---|
| 191 | `handle.engine.device().cols()` | `handle.backend.cols()` |
| 204 | `handle.engine.device().rows()` | `handle.backend.rows()` |
| 226 | `handle.engine.device().arch_name()` | `handle.backend.arch_name()` |
| 32, 65, 109, 162 | `handle.engine.device_mut()` | `handle.backend.as_interpreter_mut().expect("Plan A: interpreter backend").device_mut()` |
| 286, 338, 394, 442 | `handle.engine.device()` | `handle.backend.as_interpreter().expect("Plan A: interpreter backend").device()` |

Note line 191/204: the original lines end in `as u8` (e.g.
`handle.engine.device().cols() as u8`) — the table shows only the fragment to
swap; **preserve the surrounding `as u8`**, giving `handle.backend.cols() as u8`.

Note line 226: `arch_name()` now returns an owned `String` (the inherent method
returns `&str`). The original binds `let arch_name = handle.engine.device().arch_name();`
(a `&str`); change it to own the value: `let arch_name = handle.backend.arch_name();`.
Downstream `&str` uses coerce from `String` via `Deref` (or call `.as_str()` if a
function needs `&str` explicitly).

- [ ] **Step 5: Migrate `context.rs`** (context-state read → hatch)

| Line | From | To |
|---|---|---|
| 37 | `handle.engine.device()` | `handle.backend.as_interpreter().expect("Plan A: interpreter backend").device()` |

Lines 67, 86 are inside `#[cfg(test)]`; migrate them the same way
(`handle_mut.engine.device_mut()` → `handle_mut.backend.as_interpreter_mut().expect("test interpreter backend").device_mut()`,
and `handle_ref.engine.device()` → `handle_ref.backend.as_interpreter().expect("test interpreter backend").device()`).

- [ ] **Step 6: Migrate `async_errors.rs`** (all → hatch, 11 sites)

Replace each `handle.engine.device()` / `(*h).engine.device()` with
`…as_interpreter().expect("Plan A: interpreter backend").device()`, and each
`…device_mut()` variant with `…as_interpreter_mut().expect("Plan A: interpreter backend").device_mut()`.
Sites: 69 (`device().async_errors.last_cache()`), 94 (`device_mut()`), 129
(`device().async_errors.ring(col_u8)`), 158 (`device()`), 201 (`device_mut()`),
240/261/309/345/382/441 (`(*h).engine.device_mut()`). For the `(*h).engine.…`
form, the receiver is `(*h)`, so it becomes
`(*h).backend.as_interpreter_mut().expect("Plan A: interpreter backend").device_mut()`.

- [ ] **Step 7: Migrate `firmware.rs`**

| Line | From | To |
|---|---|---|
| 67 | `handle.engine.device_mut()` | `handle.backend.as_interpreter_mut().expect("Plan A: interpreter backend").device_mut()` |

Lines 123-158 are inside `#[cfg(test)]` assertions of the form
`h.engine.device().array.clock()…`; replace each `h.engine.device()` with
`h.backend.as_interpreter().expect("test interpreter backend").device()`.

- [ ] **Step 8: Migrate `execution.rs`** (run loop via hatch; helpers take `&InterpreterEngine`)

The run loop holds no long-lived borrow of the backend (that would conflict with
`handle.npu_executor` and `fire_async_callbacks_for(handle)`), so replace each
`handle.engine` **inline at its use site**:

- Spine methods stay as trait calls: `handle.engine.enabled_cores()` →
  `handle.backend.as_interpreter().expect("Plan A: interpreter backend").enabled_cores()`
  (these are interpreter-only methods not on the trait — `enabled_cores`,
  `step`, `all_cores_blocked`, `force_running`, `flush_ctrl_packets`,
  `flush_trace_to_host`, `device_and_host_memory`, `status`, etc. — so they go
  through the hatch).

Exact sites (replace `handle.engine` with
`handle.backend.as_interpreter_mut().expect("Plan A: interpreter backend")` for
`&mut` uses, `as_interpreter()` for shared reads):

| Line | Method | Hatch |
|---|---|---|
| 107 | `.device()` (read guard) | `as_interpreter()` |
| 139 | `.enabled_cores()` | `as_interpreter()` |
| 142 | `.step()` | `as_interpreter_mut()` |
| 144 | `.all_cores_blocked()` | `as_interpreter()` |
| 164 | `.device_mut().array.set_dma_cycle(cycles)` | `as_interpreter_mut()` |
| 169 | `.device_and_host_memory()` | `as_interpreter_mut()` |
| 174 | `.flush_trace_to_host()` | `as_interpreter_mut()` |
| 198 | `.flush_ctrl_packets()` | `as_interpreter_mut()` |
| 206 | `.force_running()` | `as_interpreter_mut()` |
| 209 | `.step()` | `as_interpreter_mut()` |
| 220 | `.device_mut()` | `as_interpreter_mut()` |
| 248 | `.flush_trace_to_host()` | `as_interpreter_mut()` |
| 252 | `.device_mut()` | `as_interpreter_mut()` |
| 268 | `.device_mut()` | `as_interpreter_mut()` |

For the two helper calls at lines 217-218, the helpers keep their
`&InterpreterEngine` signatures; pass the downcast:
```rust
    let interp = handle.backend.as_interpreter().expect("Plan A: interpreter backend");
    let engine_signals = build_engine_signals(interp);
    let executor_signals = build_executor_signals(&mut handle.npu_executor, interp);
```
(Bind `interp` immediately before these two lines and let it drop before line 220's
`as_interpreter_mut()`.) Add a guard at the top of `xdna_emu_run` (right after
`let handle = &mut *handle;`, line ~87) for defense-in-depth:
```rust
    if handle.backend.as_interpreter().is_none() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::ExecutionError,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    }
```

- [ ] **Step 9: Build the whole crate and run the suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib`
Expected: compiles clean, all FFI tests PASS. Then the full lib suite:
Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: same green count as before the refactor (behavior unchanged).

- [ ] **Step 10: Commit**

```bash
git add crates/xdna-emu-ffi/src
git commit -m "ffi: migrate XdnaEmuHandle to Box<dyn NpuBackend>

Atomic field flip engine -> backend across all 9 FFI files. Cross-backend
ops go through trait methods; interpreter-only device() introspection and
the run loop route through the as_interpreter()/as_interpreter_mut() hatch.
Behavior unchanged: full --lib suite green.

Generated using Claude Code."
```

---

## Task 4: Add the `XDNA_BACKEND` selector to `xdna_emu_create`

The selection *logic* is extracted into a pure helper so it can be tested
directly — without mutating the process-global `XDNA_BACKEND` env var, which
would race against every other test that calls `xdna_emu_create()` in parallel.
`xdna_emu_create` reads the env once and delegates to the helper.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs`

- [ ] **Step 1: Write the failing test**

In the `#[cfg(test)] mod tests` block in `lib.rs` (starts ~line 315):

```rust
#[test]
fn select_backend_interpreter_ok() {
    let eng = xdna_emu_core::interpreter::engine::InterpreterEngine::new_npu1();
    assert!(select_backend("interpreter", eng).is_ok());
}

#[test]
fn select_backend_aiesim_unsupported() {
    // Plan A has no aiesim backend compiled; requesting it must fail cleanly
    // rather than silently falling back to the interpreter.
    let eng = xdna_emu_core::interpreter::engine::InterpreterEngine::new_npu1();
    let r = select_backend("aiesim", eng);
    assert!(r.is_err());
    assert!(r.err().unwrap().contains("aiesim"));
}

#[test]
fn select_backend_unknown_rejected() {
    let eng = xdna_emu_core::interpreter::engine::InterpreterEngine::new_npu1();
    assert!(select_backend("bogus", eng).is_err());
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib select_backend`
Expected: FAIL — `select_backend` does not exist yet (cannot find function).

- [ ] **Step 3: Implement the helper and call it from `xdna_emu_create`**

Add the helper near the top of `lib.rs` (after the `XdnaEmuHandle` definition,
before `xdna_emu_create`):

```rust
/// Choose a backend from the `XDNA_BACKEND` value. Pure (no env read) so it is
/// directly testable. Plan A wires only the interpreter; aiesim arrives in Plan
/// B behind a cargo feature (which will replace the `"aiesim"` arm). An
/// unsupported request fails loudly rather than silently using the interpreter.
pub(crate) fn select_backend(
    kind: &str,
    engine: InterpreterEngine,
) -> Result<Box<dyn crate::backend::NpuBackend>, String> {
    match kind {
        "interpreter" => Ok(Box::new(engine)),
        "aiesim" => Err(
            "XDNA_BACKEND=aiesim: this build has no aiesim backend (Plan B not yet built)".to_string(),
        ),
        other => Err(format!("XDNA_BACKEND={other}: unknown backend")),
    }
}
```

In `xdna_emu_create`, replace the handle construction (lines ~210-222) so the
backend is selected:

```rust
    let mut engine = InterpreterEngine::new_npu1();
    engine.set_stall_threshold(config.stall_threshold());

    let backend_kind = std::env::var("XDNA_BACKEND").unwrap_or_else(|_| "interpreter".to_string());
    let backend = match select_backend(&backend_kind, engine) {
        Ok(b) => b,
        Err(msg) => {
            log::error!("{}", msg);
            set_last_error(msg);
            return std::ptr::null_mut();
        }
    };

    let handle = Box::new(XdnaEmuHandle {
        backend,
        xclbin_path: None,
        npu_executor: NpuExecutor::new(),
        max_cycles: config.max_cycles(),
        next_alloc_addr: 0x8000_0000_0000,
        free_list: Vec::new(),
        async_callback: None,
    });
```

(This supersedes the `backend: Box::new(engine)` form from Task 3 Step 1 — the
selector now owns construction. The behavior for the default/unset case is
identical: an interpreter-backed handle.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib select_backend`
Expected: PASS — all three `select_backend_*` tests.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs
git commit -m "ffi: XDNA_BACKEND selector via pure select_backend() helper

interpreter (default) is wired; aiesim and unknown values fail cleanly with
a null handle + last_error. Logic is a pure helper (no env mutation) so the
tests don't race other create() calls. Plan B fills the aiesim arm behind a
cargo feature.

Generated using Claude Code."
```

---

## Task 5: Tier-1 mock-dispatch test (prove the seam is real)

This proves the FFI genuinely dispatches over `dyn NpuBackend` (not the concrete
engine) and that a non-interpreter backend is observable — the test
infrastructure Plan B builds on.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/backend.rs`

- [ ] **Step 1: Write the test**

Add to the `tests` module in `backend.rs`:

```rust
#[test]
fn dyn_dispatch_hits_the_concrete_backend() {
    use super::mock::MockBackend;

    // Own the concrete mock; dispatch through a `&mut dyn` (vtable) reference.
    let mut mock = MockBackend::default();
    {
        let backend: &mut dyn NpuBackend = &mut mock;
        backend.set_start_col(2);
        backend.reset_for_new_context();
        backend.reset_for_new_context();

        // The downcast hatch correctly reports "not an interpreter".
        assert!(backend.as_interpreter().is_none());
        assert!(backend.as_interpreter_mut().is_none());
    } // dyn borrow ends here

    // The dynamic calls landed on the concrete MockBackend (proves dispatch
    // reached it, not a default).
    assert_eq!(mock.start_col, 2);
    assert_eq!(mock.reset_for_new_context_calls, 2);
}
```

- [ ] **Step 2: Run it**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib dyn_dispatch`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/xdna-emu-ffi/src/backend.rs
git commit -m "ffi: tier-1 test proving dyn NpuBackend dispatch reaches the backend

Generated using Claude Code."
```

---

## Task 6: Final verification + rebuild the FFI cdylib

**Files:** none (verification + the `.so` rebuild the bridge tests load).

- [ ] **Step 1: Full library suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all green, same count as the pre-refactor baseline. If any test
regressed, that is a refactor bug to fix before proceeding (a migrated site that
changed behavior).

- [ ] **Step 2: Rebuild the FFI cdylib (so a later bridge/ISA test loads the new code)**

Run: `cargo build -p xdna-emu-ffi`
Expected: builds `target/debug/libxdna_emu.so` clean. (Per CLAUDE.md: `cargo
test --lib` does NOT rebuild this `.so`; the bridge path loads whatever is on
disk.) Do not run a bridge/HW suite here — this plan is sandbox-only; the bridge
path is exercised when Plan B's backend exists.

- [ ] **Step 3: Confirm no stray `handle.engine` remains**

Run: `grep -rn "\.engine\b" crates/xdna-emu-ffi/src/ | grep -v "as_interpreter\|InterpreterEngine\|//"`
Expected: no output (every `handle.engine` site is migrated). The only remaining
references to the engine type are the `InterpreterEngine` import in `backend.rs`
and the helper signatures in `execution.rs`.

- [ ] **Step 4: Commit (if Step 2/3 produced any fixups)**

```bash
git add -A crates/xdna-emu-ffi
git commit -m "ffi: finalize NpuBackend seam refactor (Plan A complete)

Full --lib suite green, FFI cdylib rebuilt, no residual handle.engine sites.
The seam is ready for the aiesim backend (Plan B).

Generated using Claude Code."
```

---

## Done criteria

- [ ] `NpuBackend` trait exists; `InterpreterEngine` implements it; `MockBackend` exercises it.
- [ ] `XdnaEmuHandle` owns `Box<dyn NpuBackend>`; no `handle.engine` sites remain.
- [ ] `XDNA_BACKEND=interpreter` (default) works; `aiesim`/unknown fail cleanly (null + last_error).
- [ ] `cargo test --lib` green at the pre-refactor count; FFI cdylib rebuilt.
- [ ] Behavior byte-for-byte unchanged — this is a pure seam refactor.

**Next:** Plan B — the aiesim C++ bridge + `AiesimBackend` (E513 hello-cluster
first), which adds the `run()` trait method with its proper seam and fills the
`aiesim` selector arm behind a cargo feature.
