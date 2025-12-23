//! Core emulation logic for AMD XDNA NPUs.
//!
//! This module provides the emulation engine that executes AIE programs.
//! It handles:
//! - Instruction fetch, decode, and execute for AIE2 cores
//! - DMA engine simulation with n-dimensional addressing
//! - Lock synchronization between tiles
//! - Stream switch packet routing
//!
//! # Architecture
//!
//! The emulator runs all cores concurrently (no global clock sync).
//! Each core maintains its own PC and executes independently.
//! Synchronization happens via locks and DMA completion events.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::emu::Engine;
//! use xdna_emu::device::DeviceState;
//!
//! let mut state = DeviceState::new_npu1();
//! // ... apply CDO to configure device ...
//!
//! let mut engine = Engine::new(state);
//! engine.step();  // Execute one cycle
//! engine.run();   // Run until breakpoint/halt
//! ```

pub mod instruction;
pub mod core;
pub mod engine;

pub use instruction::{Instruction, InstructionKind, DecodeError};
pub use core::{CoreExecutor, CoreStatus, ExecuteResult};
pub use engine::{Engine, EngineStatus, Breakpoint};
