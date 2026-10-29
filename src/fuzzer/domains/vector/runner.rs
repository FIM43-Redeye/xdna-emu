//! Vector fuzz entry point: a thin tenant wrapper over the generic engine.
//! All orchestration lives in `core::engine`; all vector specifics in
//! `super::domain::VectorDomain`.

use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;
use crate::fuzzer::domains::vector::domain::VectorDomain;

/// Campaign knobs for the `fuzz-vector` subcommand. The vector runner reuses the
/// generic engine's [`CampaignOptions`] verbatim; this alias keeps the CLI and
/// `main.rs` call sites (`VecFuzzOptions { .. }`) unchanged.
pub use crate::fuzzer::core::domain::CampaignOptions as VecFuzzOptions;

/// Run (or report/replay) the vector fuzz campaign.
pub fn run_vector_fuzz(opts: &CampaignOptions) {
    run_campaign(&VectorDomain, opts);
}
