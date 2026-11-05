//! Cycle-accurate timing infrastructure for AIE2 emulation.
//!
//! This module provides timing models for accurate cycle counting:
//!
//! - **Latency tables**: Per-operation cycle counts from AM020
//! - **Memory model**: Bank conflict detection and access latencies
//! - **Hazard detection**: Register dependency tracking (RAW, WAW, WAR)
//! - **Pipeline model**: In-flight instruction tracking
//! - **Synchronization**: Lock contention tracking and timing
//!
//! # Architecture
//!
//! The timing module is used by `CycleAccurateExecutor` to determine
//! how many cycles each operation takes, including stalls from:
//!
//! - Instruction latency (e.g., multiply = 2 cycles)
//! - Memory bank conflicts (same bank accessed twice)
//! - Register hazards (reading a register still being written)
//! - Lock contention (waiting for synchronization primitives)
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::{LatencyTable, MemoryModel};
//! use xdna_emu::tablegen::SemanticOp;
//!
//! let latencies = LatencyTable::aie2();
//! let timing = LatencyTable::timing_from_semantic(SemanticOp::Mul, false);
//! assert_eq!(timing.latency, 2);
//! ```

pub mod latency;
pub mod memory;
pub mod hazards;
pub mod sync;

pub use latency::{LatencyTable, OperationTiming, LATENCY_MEMORY};
pub use memory::{MemoryModel, MemoryAccess, BankConflict, AlignmentError, MemoryQuadrant, CROSS_TILE_LATENCY};
pub use hazards::{HazardDetector, HazardType, HazardStats, Hazard, StallReason};
pub use sync::{LockTimingState, LockStats, SyncTimingConfig, AggregateStats};
