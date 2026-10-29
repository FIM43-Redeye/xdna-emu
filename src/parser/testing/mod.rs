//! Test fixtures for parser unit tests.
//!
//! Builders for minimal-valid XCLBIN / CDO / ELF byte streams.
//! Enable parser tests to cover edge cases (truncated headers,
//! unknown opcodes, malformed sections) without real-binary
//! dependencies. Compile only under #[cfg(test)].

pub mod xclbin_builder;
pub mod cdo_builder;
pub mod elf_builder;

pub use xclbin_builder::XclbinBuilder;
pub use cdo_builder::CdoBuilder;
pub use elf_builder::ElfBuilder;
