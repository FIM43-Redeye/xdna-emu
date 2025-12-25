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
//! - [`visual`]: GUI visualization (egui-based)
//! - [`integration`]: External tool integration

pub mod parser;
pub mod device;
pub mod interpreter;
pub mod tablegen;
pub mod visual;
pub mod integration;
