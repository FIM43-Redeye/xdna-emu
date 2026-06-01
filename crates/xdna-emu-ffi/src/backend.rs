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
    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        None
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        None
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
