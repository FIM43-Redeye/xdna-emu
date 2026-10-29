//! Scalar fuzz entry point: dispatches the coverage-driven campaign (the
//! default `fuzz` path) or the legacy trace-sweep path (`--trace-sweep`).
//! Campaign orchestration lives in `core::engine`; scalar specifics in
//! `super::domain::ScalarDomain`. The trace path stays on the legacy
//! single-accumulator kernel in `fuzzer::runner` until trace becomes a
//! framework mode.

use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;
use crate::fuzzer::domains::scalar::domain::ScalarDomain;
use crate::fuzzer::runner::{run_trace_sweep_legacy, FuzzOptions};

/// Scalar `fuzz` knobs: the generic campaign options plus the two legacy
/// trace-sweep fields (used only when `trace_sweep` is set).
pub struct ScalarFuzzOptions {
    pub campaign: CampaignOptions,
    pub trace_sweep: bool,
    pub trace_sweep_reps: usize,
}

/// Run (or report/replay) the scalar fuzz campaign, or the legacy trace sweep.
pub fn run_scalar_fuzz(opts: &ScalarFuzzOptions) {
    if opts.trace_sweep {
        let legacy = FuzzOptions {
            verbose: opts.campaign.verbose,
            jobs: opts.campaign.jobs,
            hw: opts.campaign.hw,
            max_cycles: opts.campaign.max_cycles,
            fuzz_iterations: opts.campaign.iterations,
            fuzz_seed: opts.campaign.seed,
            trace_sweep: true,
            trace_sweep_reps: opts.trace_sweep_reps,
        };
        run_trace_sweep_legacy(&legacy);
        return;
    }
    run_campaign(&ScalarDomain, &opts.campaign);
}
