//! Modular AIE2 interpreter.
//!
//! This module provides an accurate, modular implementation of the AIE2
//! instruction set interpreter. It is designed for:
//!
//! - **Accurate VLIW handling**: Proper 128-bit bundle decoding with 7 slots
//! - **TableGen-driven decoding**: Instruction definitions parsed from llvm-aie
//! - **Swappable execution modes**: Fast mode for speed, cycle-accurate for debugging
//! - **Easy testing**: Trait abstractions enable mocking and unit testing
//!
//! # Architecture
//!
//! The interpreter is organized into several submodules:
//!
//! - [`bundle`]: VLIW bundle representation and slot operations
//! - [`decode`]: TableGen-driven instruction decoder
//! - [`execute`]: Execution units (scalar ALU, vector ALU, memory)
//! - [`state`]: Processor state (registers, flags, context)
//! - [`core`]: Per-core interpreter (replaces `emu_stub::CoreExecutor`)
//! - [`engine`]: Multi-core coordinator (replaces `emu_stub::Engine`)
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::{InterpreterEngine, FastDecoder, FastExecutor};
//! use xdna_emu::device::DeviceState;
//!
//! let state = DeviceState::new_npu1();
//! let mut engine = InterpreterEngine::new(state);
//!
//! engine.step();  // Execute one cycle on all cores
//! engine.run(1000);  // Run for up to 1000 cycles
//! ```
//!
//! # Migration from emu_stub
//!
//! | emu_stub | interpreter |
//! |----------|-------------|
//! | `Engine` | `InterpreterEngine` |
//! | `CoreExecutor` | `CoreInterpreter` |
//! | `Instruction` | `VliwBundle` |
//! | `InstructionKind` | `Operation` |

pub mod traits;
pub mod bundle;
pub mod decode;
pub mod state;
pub mod execute;
pub mod core;
pub mod engine;
pub mod timing;
pub mod test_runner;

// Re-export key types for convenience
pub use traits::{Decoder, Executor, StateAccess, ExecuteResult, DecodeError};

// Bundle types
pub use bundle::{
    BundleFormat, Operation, Operand, SlotIndex, SlotMask, SlotOp, VliwBundle,
    BranchCondition, ElementType, MemWidth, PostModify, Predicate, ShufflePattern,
};

// Decoder types
pub use decode::PatternDecoder;

// State types
pub use state::{ExecutionContext, SpRegister, TimingContext};
pub use state::{
    ScalarRegisterFile, VectorRegisterFile, AccumulatorRegisterFile,
    PointerRegisterFile, ModifierRegisterFile,
};

// Execute types
pub use execute::{FastExecutor, CycleAccurateExecutor, CycleAccurateStats, ScalarAlu, VectorAlu, MemoryUnit, ControlUnit};

// Core types
pub use core::{CoreInterpreter, CoreStatus, StepResult};

// Engine types
pub use engine::{InterpreterEngine, EngineStatus};

// Timing types
pub use timing::{LatencyTable, OperationTiming, MemoryModel, HazardDetector};

// Test harness
pub use test_runner::{TestRunner, TestResult};
