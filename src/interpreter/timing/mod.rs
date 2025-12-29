//! Cycle-accurate timing infrastructure for AIE2 emulation.
//!
//! This module provides timing models for accurate cycle counting:
//!
//! - **Latency tables**: Per-operation cycle counts from AM020
//! - **Memory model**: Bank conflict detection and access latencies
//! - **Hazard detection**: Register dependency tracking (RAW, WAW, WAR)
//! - **Pipeline model**: In-flight instruction tracking
//! - **Synchronization**: Lock contention tracking and timing
//! - **Deadlock detection**: Circular wait detection for locks
//! - **Barrier tracking**: Multi-core barrier timing and statistics
//!
//! # Architecture
//!
//! The timing module is used by `CycleAccurateExecutor` to determine
//! how many cycles each operation takes, including stalls from:
//!
//! - Instruction latency (e.g., multiply = 2 cycles)
//! - Memory bank conflicts (same bank accessed twice)
//! - Register hazards (reading a register still being written)
//! - Structural hazards (resource conflicts between slots)
//! - Lock contention (waiting for synchronization primitives)
//! - Barrier synchronization (waiting for all participants)
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::interpreter::timing::{LatencyTable, MemoryModel};
//!
//! let latencies = LatencyTable::aie2();
//! let cycles = latencies.get(&Operation::ScalarMul);
//! assert_eq!(cycles, 2);
//! ```

pub mod latency;
pub mod memory;
pub mod hazards;
pub mod sync;
pub mod deadlock;
pub mod barrier;
pub mod slots;
pub mod arbitration;

pub use latency::{LatencyTable, OperationTiming};
pub use memory::{MemoryModel, MemoryAccess, BankConflict, AlignmentError, MemoryQuadrant, CROSS_TILE_LATENCY};
pub use hazards::{HazardDetector, HazardType, HazardStats, Hazard, StallReason};
pub use sync::{LockTimingState, LockStats, SyncTimingConfig, AggregateStats};
pub use deadlock::{DeadlockDetector, DeadlockCycle, DeadlockConfig, TileId, LockId};
pub use barrier::{BarrierTracker, BarrierState, BarrierConfig, BarrierStats, BarrierPhase, BarrierId, AggregateBarrierStats};
pub use slots::{ExecutionResource, StructuralHazard, check_bundle_conflicts, bundle_structural_penalty};
pub use arbitration::{MemTileArbiter, ArbiterSource, ArbiterStats};
