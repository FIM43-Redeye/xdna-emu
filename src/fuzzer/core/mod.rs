//! Domain-agnostic differential-fuzzing engine: the campaign loop, coverage
//! ledger, parallel compile, banking/replay, and reporting that every fuzzer
//! domain (vector, and later scalar/DMA/...) plugs into as a `Domain` tenant.
//!
//! Lifted out of the vector fuzzer in the framework Step 1 refactor
//! (`docs/superpowers/plans/2026-06-11-framework-step1-lift-vector.md`).

pub mod domain;
pub mod engine;
pub mod ledger;
pub mod toolchain;

pub use domain::{Backend, Banked, CampaignOptions, Domain};
pub use engine::run_campaign;
pub use ledger::Ledger;
