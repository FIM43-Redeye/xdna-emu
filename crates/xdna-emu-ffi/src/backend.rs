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

use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::interpreter::engine::InterpreterEngine;
use xdna_emu_core::parser::Cdo; // same path config.rs imports (re-exported at parser root)

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
/// which took the whole handle -- the borrow tangle a prior phase dodged by
/// deferring `run()`. The FFI's `CallbackObserver` (async_errors.rs) fires the
/// registered C callback for each record.
pub trait RunObserver {
    fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]);
}

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
    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        None
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        None
    }
}

impl NpuBackend for InterpreterEngine {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        // DeviceState::apply_cdo returns anyhow::Result<()>; normalize the
        // error to String (Display is sufficient for the FFI boundary).
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
    // Deliberately delegate to `self.device().X()`, NOT `self.X()` — the latter
    // would re-enter this trait method and infinitely recurse. (The methods
    // above use `InterpreterEngine::X(self)` UFCS for the same reason.)
    fn cols(&self) -> usize {
        self.device().cols()
    }
    fn rows(&self) -> usize {
        self.device().rows()
    }
    fn arch_name(&self) -> String {
        self.device().arch_name().to_string()
    }

    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        Some(self)
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        Some(self)
    }
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
        fn set_start_col(&mut self, start_col: u8) {
            self.start_col = start_col;
        }
        fn load_elf_bytes(&mut self, _c: usize, _r: usize, _d: &[u8]) -> Result<u32, String> {
            Ok(0)
        }
        fn host_memory_mut(&mut self) -> &mut HostMemory {
            unimplemented!("MockBackend has no host memory")
        }
        fn sync_cores_from_device(&mut self) {}
        fn reset_for_new_context(&mut self) {
            self.reset_for_new_context_calls += 1;
        }
        fn reset_context(&mut self, _cid: ContextId) -> Result<(), ()> {
            Ok(())
        }
        fn cols(&self) -> usize {
            5
        }
        fn rows(&self) -> usize {
            6
        }
        fn arch_name(&self) -> String {
            "mock".to_string()
        }
        // as_interpreter / as_interpreter_mut use the default None impls.
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockBackend;
    use super::NpuBackend;

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

    #[test]
    fn interpreter_implements_backend_and_downcasts() {
        use xdna_emu_core::interpreter::engine::InterpreterEngine;
        let mut eng = InterpreterEngine::new_npu1();

        // Trait methods reflect the real device.
        let b: &mut dyn NpuBackend = &mut eng;
        assert_eq!(b.cols(), 5);
        assert_eq!(b.rows(), 6);
        assert!(b.arch_name().to_lowercase().contains("npu") || b.arch_name().to_lowercase().contains("aie"));

        // The downcast hatch returns the engine.
        assert!(b.as_interpreter().is_some());
        assert!(b.as_interpreter_mut().is_some());
    }
}
