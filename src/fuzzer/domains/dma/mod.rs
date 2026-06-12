//! DMA / data-movement fuzzer domain (framework Step 3a).
//!
//! Fuzzes DDR buffer-descriptor access patterns across the shim and memtile DMA
//! engines, differentially against silicon, with per-region localization. See
//! `docs/superpowers/specs/2026-06-11-dma-data-movement-domain.md`.
pub mod chain;
pub mod domain;
pub mod gen;
pub mod lower;
pub mod runner;
pub mod table;
