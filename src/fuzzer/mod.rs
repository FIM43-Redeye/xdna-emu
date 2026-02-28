//! Differential logic fuzzer for NPU emulator validation.
//!
//! Generates random kernel programs, compiles them via Peano (and optionally
//! Chess), runs them on the emulator and real NPU hardware, and compares
//! outputs. Mismatches indicate emulator bugs.
//!
//! See `docs/plans/2026-02-28-logic-fuzzer-design.md` for the full design.

pub mod ast;
pub mod params;
