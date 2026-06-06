//! VCD deep extraction: parse, emit, and compare aiesimulator VCD files.
//!
//! This module provides:
//! - [`state_path::StatePath`]: canonical signal identity bridging VCD names and emulator state
//! - [`mapping`]: hierarchical signal mapping tree (VCD signal names <-> StatePaths)
//! - [`coverage`]: coverage audit (mapped vs unmapped signals)
//! - [`compare`]: subsystem-level comparison engine
//! - [`tolerance`]: configurable timing tolerance bands
//! - [`report`]: text and JSON report generation
//! - [`emit`]: VCD emission from emulator (behind `vcd-recording` feature flag)

pub mod state_path;
pub mod mapping;
pub mod coverage;
pub mod tolerance;
pub mod compare;
pub mod cycles;
pub mod report;

#[cfg(feature = "vcd-recording")]
pub mod emit;

// Subsystem mapping subtrees
pub mod lock_mapping;
pub mod dma_mapping;
pub mod stream_mapping;
pub mod core_mapping;
pub mod event_mapping;
