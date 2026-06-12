//! DMA fuzz campaign runner.
use super::domain::DmaDomain;
use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;

pub struct DmaFuzzOptions {
    pub campaign: CampaignOptions,
}

pub fn run_dma_fuzz(opts: &DmaFuzzOptions) {
    run_campaign(&DmaDomain, &opts.campaign);
}
