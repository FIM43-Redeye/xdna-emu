//! xdna-emu library
//!
//! Core emulation logic for AMD XDNA NPUs.
//!
//! # Module Organization
//!
//! - [`config`]: Configuration management (paths, settings)
//! - [`parser`]: Binary format parsers (xclbin, CDO, ELF)
//! - [`device`]: Hardware state model (tiles, registers, memory)
//! - [`interpreter`]: Accurate AIE2 interpreter
//! - [`tablegen`]: TableGen parser for instruction definitions
//! - [`npu`]: Host-to-NPU instruction execution
//! - [`visual`]: Trace comparison visualizer (egui-based)
//! - [`integration`]: External tool integration
//! - [`testing`]: Test harness for XCLBIN binary compatibility
//! - `ffi`: C-compatible FFI (separate crate: `xdna-emu-ffi`)
//! - [`fuzzer`]: Differential logic fuzzer for emulator validation
//! - [`aiesim`]: aiesimulator subprocess harness for VCD cross-validation

// Core emulation engine (always available)
pub mod config;
pub mod parser;
pub mod device;
pub mod interpreter;
pub mod tablegen;
pub mod npu;
pub mod trace;
pub mod aiesim;
pub mod debug;

// GUI (requires eframe/egui)
#[cfg(feature = "gui")]
pub mod visual;

// VCD deep extraction (requires wellen)
#[cfg(feature = "analysis")]
pub mod vcd;

// Test harness, external tool wrappers, fuzzer (requires crossterm)
#[cfg(feature = "tooling")]
pub mod build_progress;
#[cfg(feature = "tooling")]
pub mod testing;
#[cfg(feature = "tooling")]
pub mod integration;
#[cfg(feature = "tooling")]
pub mod fuzzer;

/// Compile-time architecture constants forwarded from xdna_archspec::aie2.
///
/// All generator outputs now live in xdna_archspec::aie2. The singular
/// `subsystem` shim survives for consumer compatibility; Task 11 renames
/// consumers to xdna_archspec::aie2::subsystems directly and removes
/// this block.
pub mod arch {
    pub use xdna_archspec::aie2::*;

    /// Singular-name compatibility shim for existing consumers.
    /// Task 11 renames them to `subsystems`.
    pub mod subsystem {
        pub use xdna_archspec::aie2::subsystems::*;
    }
}
