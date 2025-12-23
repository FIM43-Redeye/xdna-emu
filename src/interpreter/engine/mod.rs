//! Multi-core interpreter engine.
//!
//! The `InterpreterEngine` coordinates execution across all AIE2 compute cores
//! in a tile array. It manages core interpreters, handles synchronization,
//! and provides a unified interface for the GUI/CLI.
//!
//! # Execution Model
//!
//! Cores run "conceptually in parallel" - each step advances all enabled cores
//! by one cycle. Lock/DMA synchronization is resolved between steps.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::InterpreterEngine;
//! use xdna_emu::device::DeviceState;
//!
//! let device = DeviceState::new_npu1();
//! let mut engine = InterpreterEngine::new(device);
//!
//! // Step all cores once
//! engine.step();
//!
//! // Run for 1000 cycles
//! engine.run(1000);
//! ```

mod coordinator;

pub use coordinator::{InterpreterEngine, EngineStatus};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceState;

    #[test]
    fn test_engine_creation() {
        let device = DeviceState::new_npu1();
        let engine = InterpreterEngine::new(device);

        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
    }
}
