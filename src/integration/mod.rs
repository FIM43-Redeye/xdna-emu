//! External tool integration (XRT, Peano, aietools).
//!
//! Each sub-module wraps an optional external tool, providing:
//! - Discovery (is it installed? where?)
//! - Invocation (construct Commands with correct environment)
//! - Output parsing (structured data from tool stdout)
//!
//! All integrations are optional -- the emulator works without them.

pub mod aietools;
pub mod aiesimulator;
pub mod bridge;
pub mod chess_build;
pub mod elfanalyzer;
