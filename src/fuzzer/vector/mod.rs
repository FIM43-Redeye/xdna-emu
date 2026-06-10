//! Vector differential fuzzer: 512-bit `aie::vector` op chains.
//!
//! Each fuzz case is a chain of 8-16 vector stages that compose type-legally;
//! every stage's result is stored to its own output slice. The op table in
//! [`table`] defines what a stage can be. Intrinsic spellings are grounded in
//! the Peano-reach spike (`docs/superpowers/specs/2026-06-10-vector-fuzzer-spike-findings.md`);
//! every listed family compiles to native AIE2 vector code (no scalar loops).

pub mod table;
