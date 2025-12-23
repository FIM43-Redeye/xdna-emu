//! **DEPRECATED**: Legacy emulation stub.
//!
//! This module contains the original simplified emulation implementation.
//! It is preserved as a reference and fallback, but should not be used
//! for new development.
//!
//! **Use `xdna_emu::interpreter` instead for accurate AIE2 emulation.**
//!
//! # Why This Exists
//!
//! The original implementation was designed for visualization and debugging
//! rather than cycle-accurate simulation. The instruction decoder uses
//! heuristic pattern matching rather than proper VLIW decoding, and many
//! operations (DMA, vector) are stubbed.
//!
//! # Migration
//!
//! Replace uses of:
//! - `emu_stub::Engine` → `interpreter::InterpreterEngine`
//! - `emu_stub::CoreExecutor` → `interpreter::CoreInterpreter`
//! - `emu_stub::Instruction` → `interpreter::VliwBundle`
//!
//! # Original Description
//!
//! This module provides the emulation engine that executes AIE programs.
//! It handles:
//! - Instruction fetch, decode, and execute for AIE2 cores
//! - DMA engine simulation with n-dimensional addressing
//! - Lock synchronization between tiles
//! - Stream switch packet routing
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::emu_stub::Engine;
//! use xdna_emu::device::DeviceState;
//!
//! let mut state = DeviceState::new_npu1();
//! // ... apply CDO to configure device ...
//!
//! let mut engine = Engine::new(state);
//! engine.step();  // Execute one cycle
//! engine.run();   // Run until breakpoint/halt
//! ```

#![allow(deprecated)]

pub mod instruction;
pub mod core;
pub mod engine;

#[deprecated(since = "0.2.0", note = "Use interpreter::VliwBundle instead")]
pub use instruction::{Instruction, InstructionKind, DecodeError};
#[deprecated(since = "0.2.0", note = "Use interpreter::CoreInterpreter instead")]
pub use core::{CoreExecutor, CoreStatus, ExecuteResult};
#[deprecated(since = "0.2.0", note = "Use interpreter::InterpreterEngine instead")]
pub use engine::{Engine, EngineStatus, Breakpoint};
