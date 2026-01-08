//! xdna-emu library
//!
//! Core emulation logic for AMD XDNA NPUs.
//!
//! # Module Organization
//!
//! - [`parser`]: Binary format parsers (xclbin, CDO, ELF)
//! - [`device`]: Hardware state model (tiles, registers, memory)
//! - [`interpreter`]: Accurate AIE2 interpreter
//! - [`tablegen`]: TableGen parser for instruction definitions
//! - [`npu`]: Host-to-NPU instruction execution
//! - [`visual`]: GUI visualization (egui-based)
//! - [`integration`]: External tool integration
//! - [`testing`]: Test harness for XCLBIN binary compatibility
//! - [`ffi`]: C-compatible Foreign Function Interface

pub mod parser;
pub mod device;
pub mod interpreter;
pub mod tablegen;
pub mod npu;
pub mod visual;
pub mod integration;
pub mod testing;
pub mod ffi;
