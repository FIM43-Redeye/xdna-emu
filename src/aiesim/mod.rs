//! aiesimulator integration for cross-validation.
//!
//! This module provides subprocess management for running AMD's aiesimulator
//! and collecting VCD output for comparison against the emulator.
//!
//! # Architecture
//!
//! aiesim is run as an external process, not linked in. The flow:
//! 1. mlir-aie compiles a kernel targeting xcve2802 (Versal AIE2)
//! 2. [`AiesimHarness`] launches aiesimulator on the sim package
//! 3. VCD output is collected from `aiesimulator_output/`
//! 4. Existing `vcd::compare` engine diffs against emulator VCD
//!
//! # Relationship to `integration::aiesimulator`
//!
//! The [`crate::integration::aiesimulator`] module handles data-I/O simulation
//! runs (input buffers in, output buffers out) used by the bridge test suite.
//! This module is specifically for VCD cross-validation: launch, collect VCD,
//! and provide structured results for the comparison pipeline.

pub mod harness;

pub use harness::{AiesimConfig, AiesimError, AiesimHarness, AiesimResult};
