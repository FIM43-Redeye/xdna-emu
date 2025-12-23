//! xdna-emu library
//!
//! Core emulation logic for AMD XDNA NPUs.
//!
//! # Module Organization
//!
//! - [`parser`]: Binary format parsers (xclbin, CDO, ELF)
//! - [`device`]: Hardware state model (tiles, registers, memory)
//! - [`interpreter`]: Accurate AIE2 interpreter (recommended)
//! - [`emu_stub`]: Legacy emulation stub (deprecated)
//! - [`visual`]: GUI visualization (egui-based)
//! - [`integration`]: External tool integration

pub mod parser;
pub mod device;
pub mod interpreter;
pub mod visual;
pub mod integration;

/// Legacy emulation stub - deprecated, use `interpreter` instead.
#[deprecated(since = "0.2.0", note = "Use the interpreter module instead")]
pub mod emu_stub;

/// Backwards compatibility alias for the legacy `emu` module.
///
/// **Deprecated**: Use `emu_stub` or preferably `interpreter` instead.
#[deprecated(since = "0.2.0", note = "Module renamed to emu_stub; use interpreter for new code")]
pub use emu_stub as emu;
