//! xdna-emu library
//!
//! Core emulation logic for AMD XDNA NPUs.
//!
//! # Module Organization
//!
//! - [`config`]: Configuration management (paths, settings)
//! - [`parser`]: Binary format parsers (xclbin, CDO, ELF)
//! - [`device`]: Hardware state model (tiles, registers, memory)
//! - [`firmware`]: in-tree Xtensa interpreter running the real NPU management firmware
//! - [`interpreter`]: Accurate AIE2 interpreter
//! - [`npu`]: Host-to-NPU instruction execution
//! - [`visual`]: Trace comparison visualizer (egui-based)
//! - [`integration`]: External tool integration
//! - [`testing`]: Test harness for XCLBIN binary compatibility
//! - `ffi`: C-compatible FFI (separate crate: `xdna-emu-ffi`)
//! - [`fuzzer`]: Differential logic fuzzer for emulator validation
//!
//! Arch-data (register definitions, TableGen model, subsystems, ISA decode
//! FFI) lives in the `xdna-archspec` crate at `xdna_archspec::aie2::*`.

// Core emulation engine (always available)
pub mod config;
pub mod parser;
pub mod device;
pub mod firmware;
pub mod interpreter;
pub mod npu;
pub mod trace;
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
